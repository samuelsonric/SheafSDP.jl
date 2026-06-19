# Minimal reproduction of BlockSparseArrays transpose ldiv! bug
using SheafSDP
using SheafSDP: blocktri
using LinearAlgebra
using Random

Random.seed!(42)

# Build a small sheaf: 3 vertices, path graph 1-2-3
src = [1, 2, 2, 3]
dst = [2, 1, 3, 2]
maps = [randn(4, 6) for _ in 1:4]
B = sheaf(src, dst, maps)

# L is lower triangular part of B'B
L = blocktri(B' * B, Val(:L))

# Test vector
x = randn(size(L, 2))

# Forward solve works
y_fwd = copy(x)
ldiv!(LowerTriangular(L), y_fwd)
println("Forward solve residual: ", norm(LowerTriangular(L) * y_fwd - x))

# Transpose solve is broken
y_trans = copy(x)
ldiv!(LowerTriangular(L)', y_trans)
println("Transpose solve residual: ", norm(LowerTriangular(L)' * y_trans - x))
