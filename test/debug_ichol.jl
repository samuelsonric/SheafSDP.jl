using SheafSDP
using SheafSDP: blocktri, copyblockdiag, cholblockdiag!, ldivblockdiag!, rdivblockdiag!, ichol!, lmulblockdiag!, block, vtxs, srcrange, weightedgraph
using BlockSparseArrays: rowrange, colrange, selectvtxs
using CliqueTrees: RCM
using CliqueTrees.Multifrontal: symbolic
using LinearAlgebra
using Random

Random.seed!(42)

src = [1, 2, 2, 3]
dst = [2, 1, 3, 2]
maps = [randn(4, 6) for _ in 1:4]
B = sheaf(src, dst, maps)

# Apply RCM ordering
weights, graph = weightedgraph(B)
P, Q, S = symbolic(weights, graph; alg=RCM())
B = selectvtxs(B, P.perm)

α = 0.1

function to_dense(L)
    n = size(L, 1)
    M = zeros(n, n)
    for v in vtxs(L)
        for e in srcrange(L, v)
            u = L.tgt[e]
            blk = block(L, u, v, e)
            rows = rowrange(L, u)
            cols = colrange(L, v)
            M[rows, cols] .= blk
        end
    end
    return M
end

# Original C
BtB = B' * B
C_orig = blocktri(BtB, Val(:L))
axpy!(α, I, C_orig)
C_dense_orig = to_dense(C_orig)

# Working copy
C = blocktri(BtB, Val(:L))
axpy!(α, I, C)

D = copyblockdiag(C)
cholblockdiag!(D, :L)

# Scale
ldivblockdiag!(C, D, Val(:L))
rdivblockdiag!(C, D, Val(:L))
C_scaled = to_dense(C)

# Symmetrize
C_scaled_sym = C_scaled + transpose(C_scaled) - Diagonal(diag(C_scaled))

println("Scaled C (symmetrized), size $(size(C_scaled_sym)):")
display(round.(C_scaled_sym, digits=3))

# Do ichol
ichol!(C, Val(:L))
M = to_dense(C)

println("\nIncomplete Cholesky factor M:")
display(round.(M, digits=3))

# Check M * transpose(M)
MMt = M * transpose(M)
println("\nM * M^T:")
display(round.(MMt, digits=3))

println("\nDifference M*M^T - scaled_C_sym:")
diff = MMt - C_scaled_sym
println("  ||diff|| = ", round(norm(diff), digits=6))
println("  relative error = ", round(norm(diff) / norm(C_scaled_sym), digits=6))
