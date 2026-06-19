# Compare ichol_impl! vs manual IC(0) step by step
using SheafSDP
using SheafSDP: blocktri, block, vtxs, srcrange, copyblockdiag, cholblockdiag!, ldivblockdiag!, rdivblockdiag!
using LinearAlgebra
using Random

Random.seed!(42)

# Path graph with 5 vertices
nv = 5
dv, de = 4, 3
src, dst = Int[], Int[]
for i in 1:nv-1
    push!(src, i); push!(dst, i+1)
    push!(src, i+1); push!(dst, i)
end
maps = [randn(de, dv) for _ in 1:length(src)]
B = sheaf(src, dst, maps)

α = 0.1

# Create two identical copies
C1 = blocktri(B' * B, Val(:L))
C2 = blocktri(B' * B, Val(:L))
using LinearAlgebra: axpy!
axpy!(α, I, C1)
axpy!(α, I, C2)

# Scale both the same way
D = copyblockdiag(C1)
cholblockdiag!(D, :L)
ldivblockdiag!(C1, D, Val(:L))
rdivblockdiag!(C1, D, Val(:L))

D2 = copyblockdiag(C2)
cholblockdiag!(D2, :L)
ldivblockdiag!(C2, D2, Val(:L))
rdivblockdiag!(C2, D2, Val(:L))

println("After scaling, C1 == C2: ", norm(Matrix(C1) - Matrix(C2)) < 1e-14)

# Now run IC(0) manually on C1 and ichol_impl! on C2
I_int = eltype(C1.xsrc)

println("\n" * "="^60)
println("RUNNING MANUAL IC(0) ON C1")
println("="^60)

for v in vtxs(C1)
    estrt = C1.xsrc[v]
    estop = C1.xsrc[v + one(I_int)] - one(I_int)

    Lvv = block(C1, v, v, estrt)

    println("\nVertex $v:")
    println("  Diagonal block BEFORE chol: ", round.(diag(Lvv), digits=4))

    Fvv = cholesky!(Symmetric(Lvv, :L); check=false)

    println("  Diagonal block AFTER chol: ", round.(diag(Lvv), digits=4))

    # Scale off-diagonal
    for e in estrt + one(I_int):estop
        u = C1.tgt[e]
        blk = block(C1, u, v, e)
        println("  Scaling L[$u,$v]: ||before|| = ", round(norm(blk), digits=4))
        rdiv!(blk, LowerTriangular(Lvv)')
        println("             ||after|| = ", round(norm(blk), digits=4))
    end

    # Update - SIMPLIFIED: just do L[u,u] -= L[u,v] * L[u,v]'
    for e in estrt + one(I_int):estop
        u = C1.tgt[e]
        Luv = block(C1, u, v, e)
        e_diag_u = C1.xsrc[u]
        Luu = block(C1, u, u, e_diag_u)
        println("  Updating L[$u,$u]: diag before = ", round.(diag(Luu), digits=4))
        mul!(Luu, Luv, Luv', -1.0, 1.0)
        println("             diag after = ", round.(diag(Luu), digits=4))
    end
end

println("\n" * "="^60)
println("RUNNING ichol_impl! ON C2")
println("="^60)

# This is the actual ichol_impl! with print statements
for v in vtxs(C2)
    estrt = C2.xsrc[v]
    estop = C2.xsrc[v + one(I_int)] - one(I_int)

    Lvv = block(C2, v, v, estrt)

    println("\nVertex $v:")
    println("  Diagonal block BEFORE chol: ", round.(diag(Lvv), digits=4))

    Fvv = cholesky!(Symmetric(Lvv, :L); check=false)

    println("  Diagonal block AFTER chol: ", round.(diag(Lvv), digits=4))

    # Scale off-diagonal
    for e in estrt + one(I_int):estop
        u = C2.tgt[e]
        blk = block(C2, u, v, e)
        println("  Scaling L[$u,$v]: ||before|| = ", round(norm(blk), digits=4))
        rdiv!(blk, LowerTriangular(Lvv)')
        println("             ||after|| = ", round(norm(blk), digits=4))
    end

    # Update - ichol_impl! style with the nested loops
    for e in estrt + one(I_int):estop
        u = C2.tgt[e]
        Luv = block(C2, u, v, e)

        estrtu = C2.xsrc[u]
        estopu = C2.xsrc[u + one(I_int)] - one(I_int)

        eu = estrtu

        for ev in e:estop
            while eu ≤ estopu && C2.tgt[eu] < C2.tgt[ev]
                eu += one(I_int)
            end

            eu ≤ estopu || break

            uu = C2.tgt[eu]
            uv = C2.tgt[ev]

            if uu == uv
                target = block(C2, uu, u, eu)
                source = block(C2, uv, v, ev)
                println("  Update L[$uu,$u]: uu=$uu, uv=$uv, target edges $eu")
                println("    diag before = ", round.(diag(target), digits=4))
                mul!(target, source, Luv', -1.0, 1.0)
                println("    diag after = ", round.(diag(target), digits=4))
            end
        end
    end
end

println("\n" * "="^60)
println("COMPARISON AFTER ichol!")
println("="^60)

L1 = LowerTriangular(Matrix(C1))
L2 = LowerTriangular(Matrix(C2))

println("||C1 - C2|| (lower triangular) = ", norm(L1 - L2))

# CHECK: Does L_scaled * L_scaled' = C_scaled?
# Get the scaled matrix (before ichol!)
C_scaled = blocktri(B' * B, Val(:L))
axpy!(α, I, C_scaled)
D_check = copyblockdiag(C_scaled)
cholblockdiag!(D_check, :L)
ldivblockdiag!(C_scaled, D_check, Val(:L))
rdivblockdiag!(C_scaled, D_check, Val(:L))

# Symmetrize for comparison
C_scaled_dense = Matrix(C_scaled)
C_scaled_sym = C_scaled_dense + C_scaled_dense' - Diagonal(diag(C_scaled_dense))

println("\nSCALED matrix quality check:")
println("||L_scaled * L_scaled' - C_scaled|| = ", norm(L1 * L1' - C_scaled_sym))
println("This should be ~0 for perfect IC(0)")

# Now scale back
using SheafSDP: lmulblockdiag!
lmulblockdiag!(C1, D, Val(:L))
lmulblockdiag!(C2, D2, Val(:L))

println("\n" * "="^60)
println("AFTER SCALING BACK")
println("="^60)

L1_final = LowerTriangular(Matrix(C1))
L2_final = LowerTriangular(Matrix(C2))

println("||C1 - C2|| (lower triangular) = ", norm(L1_final - L2_final))

# Compare with exact Cholesky
A_reg = Matrix(B' * B) + α * I
L_exact = cholesky(Symmetric(A_reg, :L)).L

println("\nComparison with exact Cholesky:")
println("||C1 - L_exact|| / ||L_exact|| = ", norm(L1_final - L_exact) / norm(L_exact))
println("||C1 * C1' - A_reg|| / ||A_reg|| = ", norm(L1_final * L1_final' - A_reg) / norm(A_reg))

# Check if the reconstruction error is consistent
println("\nDoes L L' = A?")
println("||L_exact * L_exact' - A_reg|| = ", norm(L_exact * L_exact' - A_reg))
println("||C1 * C1' - A_reg|| = ", norm(L1_final * L1_final' - A_reg))
