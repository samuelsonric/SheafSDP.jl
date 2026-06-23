using SheafSDP
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: vtxs, colrange, rowrange, ncols, blocksparse, block

Random.seed!(42)
N, T_steps, p = 10, 15, 3.0
α_pow = 1/p
ū = 100.0
nx = 4; nu = 2; h = 0.1

A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

x0 = [randn(nx) for _ in 1:N]
edges = [(i, i+1) for i in 1:N-1]

num_pow_per_agent = nu * (T_steps - 1)
blocks_per_agent = T_steps + num_pow_per_agent + 2 * (T_steps - 1)

col_x(i, t) = (i - 1) * blocks_per_agent + t
col_pow(i, t, k) = (i - 1) * blocks_per_agent + T_steps + (t - 1) * nu + k
col_sp(i, t) = (i - 1) * blocks_per_agent + T_steps + num_pow_per_agent + 2 * (t - 1) + 1
col_sm(i, t) = (i - 1) * blocks_per_agent + T_steps + num_pow_per_agent + 2 * (t - 1) + 2

rows_per_agent = 1 + (T_steps - 1) + num_pow_per_agent + 2 * (T_steps - 1)

row_init(i) = (i - 1) * rows_per_agent + 1
row_dyn(i, t) = (i - 1) * rows_per_agent + 1 + t
row_x2(i, t, k) = (i - 1) * rows_per_agent + T_steps + (t - 1) * nu + k
row_boxp(i, t) = (i - 1) * rows_per_agent + T_steps + num_pow_per_agent + 2 * (t - 1) + 1
row_boxm(i, t) = (i - 1) * rows_per_agent + T_steps + num_pow_per_agent + 2 * (t - 1) + 2
row_coord(e) = N * rows_per_agent + e

row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

for i in 1:N
    push!(row_ids, row_init(i))
    push!(col_ids, col_x(i, 1))
    push!(blocks, Matrix(1.0I, nx, nx))

    for t in 1:T_steps-1
        push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t)); push!(blocks, -A_dyn)
        push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t + 1)); push!(blocks, Matrix(1.0I, nx, nx))

        for k in 1:nu
            B_col_k = B_dyn[:, k:k]
            pick_x3_apply_Bk = B_col_k * [0.0 0.0 1.0]
            push!(row_ids, row_dyn(i, t))
            push!(col_ids, col_pow(i, t, k))
            push!(blocks, -pick_x3_apply_Bk)
        end

        for k in 1:nu
            push!(row_ids, row_x2(i, t, k))
            push!(col_ids, col_pow(i, t, k))
            push!(blocks, reshape([0.0, 1.0, 0.0], 1, 3))
        end

        for k in 1:nu
            push!(row_ids, row_boxp(i, t))
            push!(col_ids, col_pow(i, t, k))
            blk = zeros(nu, 3)
            blk[k, 3] = 1.0
            push!(blocks, blk)
        end
        push!(row_ids, row_boxp(i, t))
        push!(col_ids, col_sp(i, t))
        push!(blocks, Matrix(1.0I, nu, nu))

        for k in 1:nu
            push!(row_ids, row_boxm(i, t))
            push!(col_ids, col_pow(i, t, k))
            blk = zeros(nu, 3)
            blk[k, 3] = -1.0
            push!(blocks, blk)
        end
        push!(row_ids, row_boxm(i, t))
        push!(col_ids, col_sm(i, t))
        push!(blocks, Matrix(1.0I, nu, nu))
    end
end

for (e, (i, j)) in enumerate(edges)
    push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T_steps)); push!(blocks, -P_proj)
    push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T_steps)); push!(blocks, P_proj)
end

B = blocksparse(row_ids, col_ids, blocks)

c = zeros(size(B, 2))
for i in 1:N, t in 1:T_steps-1, k in 1:nu
    c_rng = colrange(B, col_pow(i, t, k))
    c[c_rng[1]] = 1.0
end

g = zeros(size(B, 1))
for i in 1:N
    g[rowrange(B, row_init(i))] .= x0[i]
    for t in 1:T_steps-1, k in 1:nu
        g[rowrange(B, row_x2(i, t, k))] .= 1.0
    end
    for t in 1:T_steps-1
        g[rowrange(B, row_boxp(i, t))] .= ū
        g[rowrange(B, row_boxm(i, t))] .= ū
    end
end

Q = SheafSDP.allocblockdiag(B)
fill!(Q, 0)

nv = N * blocks_per_agent
cones = Vector{SheafSDP.Cone}(undef, nv)
for i in 1:N
    for t in 1:T_steps
        cones[col_x(i, t)] = SheafSDP.CofreeCone()
    end
    for t in 1:T_steps-1
        for k in 1:nu
            cones[col_pow(i, t, k)] = SheafSDP.PowerCone(α_pow)
        end
        cones[col_sp(i, t)] = SheafSDP.PositiveCone()
        cones[col_sm(i, t)] = SheafSDP.PositiveCone()
    end
end

prob = IPMProblem(c, g, B, Q, cones)

# Test with different rgmax values
for rgmax in [1e-6, 1e-4, 1e-2, 1.0]
    settings = IPMSettings{Float64}(
        kkt=UzawaSettings{Float64}(raug=1e4, rgmax=rgmax),
        feas_tol=1e-6, gap_tol=1e-6, itmax=200,
        verbose=false
    )
    result = solve(prob, settings)
    println("rgmax=$rgmax: status=$(result.status), iters=$(result.iterations)")
end
