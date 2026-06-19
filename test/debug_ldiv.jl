using SheafSDP
using SheafSDP: blocktri, block, vtxs, srcrange
using BlockSparseArrays: rowrange, colrange
using LinearAlgebra
using SparseArrays
using Random

Random.seed!(42)

src = [1, 2, 2, 3]
dst = [2, 1, 3, 2]
maps = [randn(4, 6) for _ in 1:4]
B = sheaf(src, dst, maps)

L = blocktri(B' * B, Val(:L))

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

L_dense = to_dense(L)
L_sparse = sparse(L_dense)

x = randn(size(L, 2))

# BlockSparseMatrix transpose solve
y_block = copy(x)
ldiv!(LowerTriangular(L)', y_block)

# SparseMatrixCSC transpose solve
y_sparse = LowerTriangular(L_sparse)' \ x

# Dense transpose solve (reference)
y_dense = LowerTriangular(L_dense)' \ x

println("Input x:")
println("  ", round.(x[1:6], digits=4), " ...")

println("\nTranspose solve L' \\ x:")
println("  block:  ", round.(y_block[1:6], digits=4), " ...")
println("  sparse: ", round.(y_sparse[1:6], digits=4), " ...")
println("  dense:  ", round.(y_dense[1:6], digits=4), " ...")

println("\nElement-by-element comparison (first 6):")
for i in 1:6
    println("  [$i] block=$(round(y_block[i], digits=4)), dense=$(round(y_dense[i], digits=4)), diff=$(round(y_block[i] - y_dense[i], digits=4))")
end

println("\nVerify L' * y_dense == x:")
residual = LowerTriangular(L_dense)' * y_dense - x
println("  ||L' * y_dense - x|| = ", norm(residual))

println("\nVerify L' * y_block == x:")
residual_block = LowerTriangular(L_dense)' * y_block - x
println("  ||L' * y_block - x|| = ", norm(residual_block))
