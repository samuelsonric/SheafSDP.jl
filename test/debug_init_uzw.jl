using SheafSDP
using LinearAlgebra
using CliqueTrees.Multifrontal: ChordalTriangular, FChordalTriangular, symbolic, fronts, diagblock, offdblock
using BlockSparseArrays: BlockSparseMatrix, blocksparse, selectvtxs

# Test the full init_uzw! path
# Old: L as ChordalTriangular
# New: L as BlockSparseMatrix

function compare_init_uzw(B::BlockSparseMatrix{T, I}) where {T, I}
    weights, graph = SheafSDP.weightedgraph(B)
    R, P, S = symbolic(weights, graph)
    B_perm = selectvtxs(B, R.perm)

    # Create test A matrix (block diagonal)
    A = SheafSDP.allocblockdiag(B_perm)
    for i in 1:length(A.val)
        A.val[i] = randn()
    end

    α = 1e4

    # Old path: L as ChordalTriangular
    L_ct = FChordalTriangular{:N, :L, T, I}(S)
    copyto!(L_ct, B_perm' * B_perm)

    F_old = FChordalTriangular{:N, :L, T, I}(S)
    SheafSDP.copyblockdiag!(F_old, A)
    axpy!(α, L_ct, F_old)

    # New path: L as BlockSparseMatrix
    L_bsm = B_perm' * B_perm

    F_new = FChordalTriangular{:N, :L, T, I}(S)
    SheafSDP.copyblockdiag!(F_new, A)
    LinearAlgebra.axpy!(α, L_bsm, F_new)

    # Compare
    max_diff = 0.0
    for f in fronts(F_old)
        fD_old, res = diagblock(F_old, f)
        fD_new, _ = diagblock(F_new, f)
        diff = maximum(abs.(parent(fD_old) .- parent(fD_new)))
        max_diff = max(max_diff, diff)

        fL_old, sep = offdblock(F_old, f)
        fL_new, _ = offdblock(F_new, f)
        if !isempty(sep)
            diff_off = maximum(abs.(fL_old .- fL_new))
            max_diff = max(max_diff, diff_off)
        end
    end

    return max_diff
end

# Use the actual problem setup from pow.jl
using Random
Random.seed!(42)

N = 10
T_steps = 15

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

println("Testing full init_uzw! path...")
max_diff = compare_init_uzw(B)
println("Max difference: ", max_diff)

if max_diff < 1e-10
    println("PASS: init_uzw! paths are equivalent")
else
    println("FAIL: init_uzw! paths differ!")
end
