#
# MINIMAL BREAKING EXAMPLE
#
# N=3 agents, T=3 timesteps, scalar state/control
# ℓ₁ consensus: minimize total control while reaching agreement
#
# SYMPTOM: μ goes NEGATIVE at iteration 6, then Cholesky fails
#   Iter 5: μ = 7.6e-6
#   Iter 6: μ = -4.2e-6  ← IMPOSSIBLE (should always be positive)
#   ERROR: PosDefException
#
# The solver steps outside the cone (p or d goes negative for POS blocks)
#

using SheafSDP
using CommonSolve: solve
using LinearAlgebra
using BlockSparseArrays: blocksparse, block

N = 3; T = 3; ū = 1.0
edges = [(1,2), (1,3), (2,3)]

# Variables per agent: 3 states (NOC) + 6 controls (POS)
vars_per_agent = T + 3*(T-1)
col_x(i, t) = (i-1)*vars_per_agent + t
col_up(i, t) = (i-1)*vars_per_agent + T + 3*(t-1) + 1
col_um(i, t) = (i-1)*vars_per_agent + T + 3*(t-1) + 2
col_w(i, t) = (i-1)*vars_per_agent + T + 3*(t-1) + 3

# Constraints: init + dynamics + box + consensus
rows_per_agent = 1 + (T-1) + (T-1)
row_init(i) = (i-1)*rows_per_agent + 1
row_dyn(i, t) = (i-1)*rows_per_agent + 1 + t
row_box(i, t) = (i-1)*rows_per_agent + T + t
row_coord(e) = N*rows_per_agent + e

row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

for i in 1:N
    push!(row_ids, row_init(i)); push!(col_ids, col_x(i,1)); push!(blocks, ones(1,1))
    for t in 1:T-1
        # Dynamics: x_{t+1} = x_t + u⁺ - u⁻
        push!(row_ids, row_dyn(i,t)); push!(col_ids, col_x(i,t)); push!(blocks, -ones(1,1))
        push!(row_ids, row_dyn(i,t)); push!(col_ids, col_x(i,t+1)); push!(blocks, ones(1,1))
        push!(row_ids, row_dyn(i,t)); push!(col_ids, col_up(i,t)); push!(blocks, -ones(1,1))
        push!(row_ids, row_dyn(i,t)); push!(col_ids, col_um(i,t)); push!(blocks, ones(1,1))
        # Box: u⁺ + u⁻ + w = ū
        push!(row_ids, row_box(i,t)); push!(col_ids, col_up(i,t)); push!(blocks, ones(1,1))
        push!(row_ids, row_box(i,t)); push!(col_ids, col_um(i,t)); push!(blocks, ones(1,1))
        push!(row_ids, row_box(i,t)); push!(col_ids, col_w(i,t)); push!(blocks, ones(1,1))
    end
end
# Consensus: x_i^T = x_j^T
for (e, (i,j)) in enumerate(edges)
    push!(row_ids, row_coord(e)); push!(col_ids, col_x(i,T)); push!(blocks, -ones(1,1))
    push!(row_ids, row_coord(e)); push!(col_ids, col_x(j,T)); push!(blocks, ones(1,1))
end

B = blocksparse(row_ids, col_ids, blocks)
c = zeros(27); for i in 1:N, t in 1:T-1; c[col_up(i,t)] = c[col_um(i,t)] = 1.0; end
g = zeros(18); for i in 1:N; g[row_init(i)] = Float64(i); for t in 1:T-1; g[row_box(i,t)] = ū; end; end
Q = SheafSDP.allocblockdiag(B); fill!(Q, 0)
for i in 1:N, t in 1:T; block(Q, col_x(i,t), col_x(i,t), col_x(i,t))[1,1] = 2e-4; end

cones = vcat([[:NOC,:NOC,:NOC,:POS,:POS,:POS,:POS,:POS,:POS] for _ in 1:N]...)

prob = IPMProblem(c, g, B, Q, cones)
result = solve(prob, IPMSettings{Float64}(verbose=true, itmax=20))
