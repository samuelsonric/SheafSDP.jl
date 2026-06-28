using AppleAccelerate
using SheafSDP
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse, colrange, rowrange
using Printf

Random.seed!(42)

# Simple LP problem to see CG trace
N, T = 50, 50
nx, nu = 4, 2
h = 0.1

A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

x0 = [randn(nx) for _ in 1:N]
edges = [(i, j) for i in 1:N for j in i+1:N]
ū = 100.0

blocks_per_agent = T + 3 * (T - 1)
col_x(i, t) = (i - 1) * blocks_per_agent + t
col_up(i, t) = (i - 1) * blocks_per_agent + T + 3 * (t - 1) + 1
col_um(i, t) = (i - 1) * blocks_per_agent + T + 3 * (t - 1) + 2
col_w(i, t)  = (i - 1) * blocks_per_agent + T + 3 * (t - 1) + 3

rows_per_agent = T + (T - 1)
row_init(i) = (i - 1) * rows_per_agent + 1
row_dyn(i, t) = (i - 1) * rows_per_agent + 1 + t
row_box(i, t) = (i - 1) * rows_per_agent + T + t
row_coord(e) = N * rows_per_agent + e

row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

for i in 1:N
    push!(row_ids, row_init(i)); push!(col_ids, col_x(i, 1)); push!(blocks, Matrix(1.0I, nx, nx))
    for t in 1:T-1
        push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t)); push!(blocks, -A_dyn)
        push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t + 1)); push!(blocks, Matrix(1.0I, nx, nx))
        push!(row_ids, row_dyn(i, t)); push!(col_ids, col_up(i, t)); push!(blocks, -B_dyn)
        push!(row_ids, row_dyn(i, t)); push!(col_ids, col_um(i, t)); push!(blocks, B_dyn)
        push!(row_ids, row_box(i, t)); push!(col_ids, col_up(i, t)); push!(blocks, ones(nu, nu))
        push!(row_ids, row_box(i, t)); push!(col_ids, col_um(i, t)); push!(blocks, ones(nu, nu))
        push!(row_ids, row_box(i, t)); push!(col_ids, col_w(i, t)); push!(blocks, ones(nu, nu))
    end
end

for (e, (i, j)) in enumerate(edges)
    push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T)); push!(blocks, -P_proj)
    push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T)); push!(blocks, P_proj)
end

B = blocksparse(row_ids, col_ids, blocks)

c = zeros(size(B, 2))
for i in 1:N, t in 1:T-1
    c[colrange(B, col_up(i, t))] .= 1.0
    c[colrange(B, col_um(i, t))] .= 1.0
end

g = zeros(size(B, 1))
for i in 1:N
    g[rowrange(B, row_init(i))] .= x0[i]
    for t in 1:T-1
        g[rowrange(B, row_box(i, t))] .= ū
    end
end

Q = SheafSDP.allocblockdiag(B)
fill!(Q, 0)

nv = N * blocks_per_agent
cones = Vector{SheafSDP.AbstractCone}(undef, nv)
for i in 1:N
    for t in 1:T
        cones[col_x(i, t)] = SheafSDP.CofreeCone()
    end
    for t in 1:T-1
        cones[col_up(i, t)] = SheafSDP.PositiveCone()
        cones[col_um(i, t)] = SheafSDP.PositiveCone()
        cones[col_w(i, t)]  = SheafSDP.PositiveCone()
    end
end

prob = SheafSDP.IPMProblem(Q, B, c, g, cones)
settings = SheafSDP.IPMSettings{Float64}(
    kkt=SheafSDP.UzawaSettings{Float64}(raug=1e7),
    feas_tol=1e-6, gap_tol=1e-6, itmax=100
)
result = solve(prob, settings)

println("Per-iteration CG trace (LP N=$N, T=$T):")
println("iter |      μ      | pred | corr | total")
println("-----|-------------|------|------|------")
for (i, row) in enumerate(result.history)
    @printf("%4d | %11.3e | %4d | %4d | %5d\n", i, row.μ, row.npred, row.ncorr, row.npred + row.ncorr)
end
println()
println("Status: $(result.status)")
