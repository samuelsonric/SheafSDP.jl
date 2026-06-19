# Simplified debug: compare BlockSparse vs Dense IC(0) step by step
using SheafSDP
using SheafSDP: blocktri, block, vtxs, srcrange
using LinearAlgebra
using Random

Random.seed!(42)

# Build small path graph sheaf
nv = 3  # Just 3 vertices for simplicity
dv, de = 3, 2
src = [1, 2, 2, 3]
dst = [2, 1, 3, 2]
maps = [randn(de, dv) for _ in 1:4]
B = sheaf(src, dst, maps)

n = size(B, 2)
println("n = $n (should be $(nv * dv))")

# Form lower triangular part of B'B + αI
α = 0.1
L_block = blocktri(B' * B, Val(:L))
using LinearAlgebra: axpy!
axpy!(α, I, L_block)

# Convert to dense
L_dense = Matrix(L_block)

println("\nInitial matrices match: ", norm(Matrix(L_block) - L_dense) < 1e-14)

# Now do IC(0) step by step on both
I_int = eltype(L_block.xsrc)

for v in 1:nv
    cols = (v-1)*dv+1:v*dv

    println("\n" * "="^50)
    println("VERTEX $v")
    println("="^50)

    # Get diagonal blocks
    e_diag = L_block.xsrc[v]
    blk_diag = block(L_block, v, v, e_diag)
    dense_diag = L_dense[cols, cols]

    println("\nDiagonal block BEFORE chol:")
    println("  Block sparse:\n", round.(blk_diag, digits=4))
    println("  Dense:\n", round.(dense_diag, digits=4))
    println("  Match: ", norm(blk_diag - dense_diag) < 1e-14)

    # Cholesky
    F_blk = cholesky!(Symmetric(blk_diag, :L); check=false)
    F_dense = cholesky(Symmetric(dense_diag, :L))
    L_dense[cols, cols] .= F_dense.L

    println("\nDiagonal block AFTER chol:")
    println("  Block sparse (via block()):\n", round.(block(L_block, v, v, e_diag), digits=4))
    println("  Dense:\n", round.(L_dense[cols, cols], digits=4))
    println("  Match: ", norm(block(L_block, v, v, e_diag) - L_dense[cols, cols]) < 1e-14)

    # Scale off-diagonal blocks
    estrt = L_block.xsrc[v]
    estop = L_block.xsrc[v + one(I_int)] - one(I_int)

    for e in estrt + one(I_int):estop
        u = L_block.tgt[e]
        rows = (u-1)*dv+1:u*dv

        blk_offdiag = block(L_block, u, v, e)
        dense_offdiag = L_dense[rows, cols]

        println("\nOff-diag block ($u,$v) BEFORE scale:")
        println("  Block sparse:\n", round.(blk_offdiag, digits=4))
        println("  Dense:\n", round.(dense_offdiag, digits=4))
        println("  Match: ", norm(blk_offdiag - dense_offdiag) < 1e-14)

        # Get the diagonal factor to scale by
        diag_factor_blk = LowerTriangular(block(L_block, v, v, e_diag))'
        diag_factor_dense = LowerTriangular(L_dense[cols, cols])'

        println("\n  Diagonal factor for scaling:")
        println("    Block sparse:\n", round.(Matrix(diag_factor_blk), digits=4))
        println("    Dense:\n", round.(Matrix(diag_factor_dense), digits=4))
        println("    Match: ", norm(Matrix(diag_factor_blk) - Matrix(diag_factor_dense)) < 1e-14)

        # Scale
        rdiv!(blk_offdiag, diag_factor_blk)
        L_dense[rows, cols] .= L_dense[rows, cols] / diag_factor_dense

        println("\nOff-diag block ($u,$v) AFTER scale:")
        println("  Block sparse:\n", round.(block(L_block, u, v, e), digits=4))
        println("  Dense:\n", round.(L_dense[rows, cols], digits=4))
        println("  Match: ", norm(block(L_block, u, v, e) - L_dense[rows, cols]) < 1e-14)
    end

    # Update next diagonal block
    for e in estrt + one(I_int):estop
        u = L_block.tgt[e]
        rows_u = (u-1)*dv+1:u*dv

        blk_Luv = block(L_block, u, v, e)
        dense_Luv = L_dense[rows_u, cols]

        # Find diagonal block of u
        e_u_diag = L_block.xsrc[u]
        blk_Luu = block(L_block, u, u, e_u_diag)
        dense_Luu = L_dense[rows_u, rows_u]

        println("\nUpdate L[$u,$u] -= L[$u,$v] * L[$u,$v]'")
        println("  L[$u,$u] BEFORE:\n", round.(blk_Luu, digits=4))
        println("  Dense BEFORE:\n", round.(dense_Luu, digits=4))

        # Do the update
        mul!(blk_Luu, blk_Luv, blk_Luv', -1.0, 1.0)
        L_dense[rows_u, rows_u] .-= dense_Luv * dense_Luv'

        println("  L[$u,$u] AFTER:\n", round.(block(L_block, u, u, e_u_diag), digits=4))
        println("  Dense AFTER:\n", round.(L_dense[rows_u, rows_u], digits=4))
        println("  Match: ", norm(block(L_block, u, u, e_u_diag) - L_dense[rows_u, rows_u]) < 1e-14)
    end
end

println("\n" * "="^50)
println("FINAL COMPARISON")
println("="^50)
L_block_final = Matrix(L_block)
A_reg = Matrix(B' * B) + α * I

println("||BlockSparse - Dense|| (raw) = ", norm(L_block_final - L_dense))

# The issue: upper triangle has garbage
println("\nUsing LowerTriangular wrapper:")
L_block_tri = LowerTriangular(L_block_final)
L_dense_tri = LowerTriangular(L_dense)
println("||LowerTri(BlockSparse) - LowerTri(Dense)|| = ", norm(L_block_tri - L_dense_tri))
println("||L * L' - A|| (BlockSparse with LowerTri) = ", norm(L_block_tri * L_block_tri' - A_reg))
println("||L * L' - A|| (Dense with LowerTri) = ", norm(L_dense_tri * L_dense_tri' - A_reg))

println("\nWithout LowerTriangular (includes upper triangle garbage):")
println("||L * L' - A|| (BlockSparse raw) = ", norm(L_block_final * L_block_final' - A_reg))
println("||L * L' - A|| (Dense raw) = ", norm(L_dense * L_dense' - A_reg))
