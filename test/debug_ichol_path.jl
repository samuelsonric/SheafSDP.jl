#
# Debug: Why does IChol struggle on path graphs?
# Path graphs have no fill-in, so IC(0) should give exact Cholesky
#
using SheafSDP
using SheafSDP: Jacobi, SSOR, IChol, weightedgraph, blocktri, block, vtxs, srcrange
using BlockSparseArrays: rowrange, colrange
using CliqueTrees: RCM
using CliqueTrees.Multifrontal: symbolic
using BlockSparseArrays: selectvtxs
using LinearAlgebra
using Random

Random.seed!(42)

function build_sheaf(src, dst, dv, de)
    maps = [randn(de, dv) for _ in 1:length(src)]
    return sheaf(src, dst, maps)
end

function path_graph(n)
    src, dst = Int[], Int[]
    for i in 1:n-1
        push!(src, i); push!(dst, i+1)
        push!(src, i+1); push!(dst, i)
    end
    return src, dst
end

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

println("="^60)
println("DEBUG: IChol on Path Graph")
println("="^60)

# Small path graph
nv = 5
dv, de = 4, 3
src, dst = path_graph(nv)
B = build_sheaf(src, dst, dv, de)

println("\nPath graph: $nv vertices, dv=$dv, de=$de")
println("B size: $(size(B))")

# Try both with and without RCM
println("\n--- WITHOUT RCM (natural ordering) ---")
B_ordered = B  # No reordering

# Form A = B'B
A = B_ordered' * B_ordered
n = size(A, 2)

println("A size: $n x $n")

# Convert A directly to dense
A_dense = Matrix(A)

println("A eigenvalues (sorted): ", round.(sort(eigvals(A_dense)), sigdigits=3)[1:5], " ...")
println("A min eigenvalue: ", minimum(eigvals(A_dense)))

# A = B'B is positive semi-definite, so add regularization for comparison
α = 0.1
A_reg = A_dense + α * I

println("\nComputing exact Cholesky of A + α*I (α=$α)...")
F_exact = cholesky(Symmetric(A_reg, :L))
L_exact = F_exact.L

println("Exact Cholesky computed successfully")

# Now build IChol preconditioner
println("\nBuilding IChol preconditioner (α=0.1)...")
P_ic = IChol(B_ordered; α=0.1)

# Extract the IChol factor as dense
L_ic_dense = to_dense(P_ic.L)

println("\nComparing factors:")
println("  ||L_ic||_F = $(norm(L_ic_dense))")
println("  ||L_exact||_F = $(norm(L_exact))")

# Check if L_ic L_ic' ≈ A + α*I  (since IChol adds regularization)
A_reconstructed = L_ic_dense * L_ic_dense'
println("\n  ||L_ic * L_ic' - (A + αI)|| / ||A + αI|| = $(norm(A_reconstructed - A_reg) / norm(A_reg))")

# Check direct comparison of factors
println("  ||L_ic - L_exact|| / ||L_exact|| = $(norm(L_ic_dense - L_exact) / norm(L_exact))")

# Check with different alpha values
for α_test in [0.01, 0.1, 1.0]
    println("\nBuilding IChol preconditioner (α=$α_test)...")
    P_ic_test = IChol(B_ordered; α=α_test)
    L_ic_test = to_dense(P_ic_test.L)

    A_reg_test = A_dense + α_test * I
    F_exact_test = cholesky(Symmetric(A_reg_test, :L))
    L_exact_test = F_exact_test.L

    A_reconstructed_test = L_ic_test * L_ic_test'
    println("  ||L_ic * L_ic' - (A + αI)|| / ||A + αI|| = $(norm(A_reconstructed_test - A_reg_test) / norm(A_reg_test))")
    println("  ||L_ic - L_exact|| / ||L_exact|| = $(norm(L_ic_test - L_exact_test) / norm(L_exact_test))")
end

# Check sparsity patterns
println("\nSparsity check:")
println("  L_exact nonzeros: $(count(!iszero, L_exact))")
println("  L_ic nonzeros: $(count(!iszero, L_ic_dense))")

# Print condition numbers
println("\nCondition numbers:")
println("  cond(A + αI) = $(cond(A_reg))")
println("  cond(L_exact) = $(cond(L_exact))")
println("  cond(L_ic) = $(cond(L_ic_dense))")

# Check if there's something wrong with the Jacobi scaling
println("\n" * "="^60)
println("CHECKING JACOBI SCALING STEP")
println("="^60)

# Manually trace through IChol construction
C = blocktri(B_ordered' * B_ordered, Val(:L))
using SheafSDP: copyblockdiag, cholblockdiag!, ldivblockdiag!, rdivblockdiag!, lmulblockdiag!

# Add regularization
using LinearAlgebra: axpy!
axpy!(α, I, C)

# Copy diagonal blocks
D = copyblockdiag(C)

# Cholesky of diagonal blocks
cholblockdiag!(D, :L)

println("After cholblockdiag! of D:")
D_dense = to_dense(D)
println("  D diagonal block structure looks correct: $(all(D_dense[i,j] == 0 for i in 1:size(D_dense,1), j in 1:size(D_dense,2) if i != j && (i-1)÷dv != (j-1)÷dv))")

# Scale: L^{-1} C
ldivblockdiag!(C, D, Val(:L))
C_after_ldiv = to_dense(C)

# Scale: (L^{-1} C) L^{-T}
rdivblockdiag!(C, D, Val(:L))
C_after_rdiv = to_dense(C)

println("\nScaled matrix (D^{-1} C D^{-T}) diagonal blocks should be identity:")
for v in 1:nv
    rows = (v-1)*dv+1:v*dv
    diag_block = C_after_rdiv[rows, rows]
    println("  Block $v: ||diag - I|| = $(norm(diag_block - I))")
end

# Now let's look at what IC(0) does on the scaled matrix
println("\n" * "="^60)
println("TRACING IC(0) STEP BY STEP")
println("="^60)

# Restart fresh
C2 = blocktri(B_ordered' * B_ordered, Val(:L))
axpy!(α, I, C2)

D2 = copyblockdiag(C2)
cholblockdiag!(D2, :L)
ldivblockdiag!(C2, D2, Val(:L))
rdivblockdiag!(C2, D2, Val(:L))

C2_before_ichol = to_dense(C2)

# Make it properly symmetric by reflecting
C2_sym = C2_before_ichol + C2_before_ichol' - Diagonal(diag(C2_before_ichol))

println("Scaled C (symmetrized) eigenvalues: ", round.(sort(eigvals(C2_sym))[1:5], sigdigits=3), " ...")
println("Min eigenvalue of scaled C: ", minimum(eigvals(C2_sym)))

# Check the diagonal blocks are I
println("\nDiagonal blocks of scaled C (should be I):")
for v in 1:nv
    rows = (v-1)*dv+1:v*dv
    diag_block = C2_sym[rows, rows]
    println("  Block $v: ||diag - I|| = $(norm(diag_block - I(dv)))")
end

# Make a dense version and do exact Cholesky on it for comparison
F_scaled_exact = cholesky(Symmetric(C2_sym, :L))
L_scaled_exact = F_scaled_exact.L

println("\nExact Cholesky of scaled C:")
println("  ||L_scaled * L_scaled' - C_scaled_sym|| = ", norm(L_scaled_exact * L_scaled_exact' - C2_sym))

# Manual IC(0) to trace where the error comes from
println("\n" * "="^60)
println("MANUAL IC(0) TRACE")
println("="^60)

# Fresh copy of scaled matrix
C3 = blocktri(B_ordered' * B_ordered, Val(:L))
axpy!(α, I, C3)
D3 = copyblockdiag(C3)
cholblockdiag!(D3, :L)
ldivblockdiag!(C3, D3, Val(:L))
rdivblockdiag!(C3, D3, Val(:L))

C3_dense = to_dense(C3) + to_dense(C3)' - Diagonal(diag(to_dense(C3)))

println("Initial scaled matrix C3 diagonal blocks (should all be I):")
for v in 1:nv
    rows = (v-1)*dv+1:v*dv
    println("  Block $v: diag = ", round.(diag(C3_dense[rows, rows]), digits=4))
end

# Compare block from BlockSparseMatrix vs dense
println("\nComparing blocks from BlockSparseMatrix vs dense extraction:")
for v in 1:nv
    rows = (v-1)*dv+1:v*dv
    cols = rows

    # Block from dense
    dense_block = C3_dense[rows, cols]

    # Block from BlockSparseMatrix directly
    e = C3.xsrc[v]  # First edge in column v (should be diagonal)
    sparse_block = block(C3, v, v, e)

    err = norm(dense_block - sparse_block)
    println("  Block ($v,$v): ||dense - sparse|| = $err")
    if err > 1e-10
        println("    Dense:\n", round.(dense_block, digits=3))
        println("    Sparse:\n", round.(sparse_block, digits=3))
    end
end

# Check if diagonal block is at first edge position for each column
println("\nChecking block structure of C3:")
for v in vtxs(C3)
    estrt = C3.xsrc[v]
    estop = C3.xsrc[v + 1] - 1
    println("  Column $v: edges $estrt to $estop")
    for e in estrt:min(estop, estrt+2)  # Just show first few
        u = C3.tgt[e]
        println("    Edge $e: row $u ($(u == v ? "DIAGONAL" : "off-diag"))")
    end
end

# Do IC(0) manually, step by step
L_manual = Matrix(C3)  # Use proper conversion

for v in 1:nv
    cols = (v-1)*dv+1:v*dv

    # Current diagonal block
    Lvv = L_manual[cols, cols]

    # Cholesky of diagonal block
    Fvv = cholesky(Symmetric(Lvv, :L))
    L_manual[cols, cols] .= Fvv.L

    println("\nAfter processing vertex $v:")
    println("  L[$v,$v] chol succeeded, ||L[$v,$v]|| = ", norm(L_manual[cols, cols]))

    # Scale off-diagonal blocks in this column
    if v < nv
        rows_next = v*dv+1:(v+1)*dv
        Luv = L_manual[rows_next, cols]
        # L[v+1,v] = L[v+1,v] / L[v,v]'
        L_manual[rows_next, cols] .= Luv / LowerTriangular(L_manual[cols, cols])'
        println("  Scaled L[$(v+1),$v], ||L[$(v+1),$v]|| = ", norm(L_manual[rows_next, cols]))

        # Update next diagonal block: L[v+1,v+1] -= L[v+1,v] * L[v+1,v]'
        rows_next2 = rows_next
        cols_next = rows_next
        L_manual[rows_next2, cols_next] .-= L_manual[rows_next, cols] * L_manual[rows_next, cols]'
        println("  Updated L[$(v+1),$(v+1)], diag = ", round.(diag(L_manual[rows_next2, cols_next]), digits=4))
    end
end

println("\nManual IC(0) result:")
println("  ||L_manual * L_manual' - C3_sym|| = ", norm(L_manual * L_manual' - C3_dense))
println("  ||L_manual - L_scaled_exact|| = ", norm(L_manual - L_scaled_exact))

# Now trace actual ichol_impl! step by step
println("\n" * "="^60)
println("TRACING ACTUAL ichol_impl!")
println("="^60)

C4 = blocktri(B_ordered' * B_ordered, Val(:L))
axpy!(α, I, C4)
D4 = copyblockdiag(C4)
cholblockdiag!(D4, :L)
ldivblockdiag!(C4, D4, Val(:L))
rdivblockdiag!(C4, D4, Val(:L))

# Manual implementation of ichol_impl! with tracing
I_int = eltype(C4.xsrc)
for v in vtxs(C4)
    estrt = C4.xsrc[v]
    estop = C4.xsrc[v + one(I_int)] - one(I_int)

    Lvv = block(C4, v, v, estrt)

    println("\nVertex $v (estrt=$estrt, estop=$estop):")
    println("  Before chol: diag(Lvv) = ", round.(diag(Lvv), digits=4))

    # No α added since we use schedule α=0
    Fvv = cholesky!(Symmetric(Lvv, :L); check=false)

    println("  After chol: diag(Lvv) = ", round.(diag(Lvv), digits=4))

    # Scale off-diagonal blocks
    for e in estrt + one(I_int):estop
        u = C4.tgt[e]
        blk = block(C4, u, v, e)

        # Compare with what the manual implementation would do
        rows_u = (u-1)*dv+1:u*dv
        cols_v = (v-1)*dv+1:v*dv
        manual_blk_before = L_manual[rows_u, cols_v]

        println("  Scaling block ($u,$v) at edge $e:")
        println("    BlockSparse before: ", round.(blk[1,:], digits=4))
        println("    Manual before:      ", round.(manual_blk_before[1,:], digits=4))
        println("    Difference before:  ", norm(blk - manual_blk_before))

        # Scale both the same way
        rdiv!(blk, LowerTriangular(Lvv)')
        L_manual[rows_u, cols_v] .= L_manual[rows_u, cols_v] / LowerTriangular(L_manual[cols_v, cols_v])'

        manual_blk_after = L_manual[rows_u, cols_v]
        println("    BlockSparse after:  ", round.(blk[1,:], digits=4))
        println("    Manual after:       ", round.(manual_blk_after[1,:], digits=4))
        println("    Difference after:   ", norm(blk - manual_blk_after))
    end

    # Update future blocks
    for e in estrt + one(I_int):estop
        u = C4.tgt[e]
        Luv = block(C4, u, v, e)

        estrtu = C4.xsrc[u]
        estopu = C4.xsrc[u + one(I_int)] - one(I_int)

        eu = estrtu

        for ev in e:estop
            while eu ≤ estopu && C4.tgt[eu] < C4.tgt[ev]
                eu += one(I_int)
            end

            eu ≤ estopu || break

            uu = C4.tgt[eu]
            uv = C4.tgt[ev]

            if uu == uv
                target_blk = block(C4, uu, u, eu)
                source_blk = block(C4, uv, v, ev)
                println("  Update: L[$uu,$u] -= L[$uv,$v] * L[$u,$v]'")
                println("    before: diag(L[$uu,$u]) = ", round.(diag(target_blk), digits=4))
                mul!(target_blk, source_blk, Luv', -one(Float64), one(Float64))
                println("    after:  diag(L[$uu,$u]) = ", round.(diag(target_blk), digits=4))
            end
        end
    end
end

C4_result = Matrix(C4)
println("\nResult from manual trace:")
println("  ||L * L' - C_sym|| = ", norm(C4_result * C4_result' - C2_sym))
println("  ||L - L_manual|| = ", norm(C4_result - L_manual))

# Also check block by block directly
println("\nBlock-by-block comparison after ichol trace:")
for v in 1:nv
    cols = (v-1)*dv+1:v*dv
    e = C4.xsrc[v]
    blk = block(C4, v, v, e)
    manual_blk = L_manual[cols, cols]
    println("  Block ($v,$v): ||trace - manual|| = ", norm(blk - manual_blk))
end

# Now compare with actual ichol!
using SheafSDP: ichol!
C2_copy = blocktri(B_ordered' * B_ordered, Val(:L))
axpy!(α, I, C2_copy)
D2_copy = copyblockdiag(C2_copy)
cholblockdiag!(D2_copy, :L)
ldivblockdiag!(C2_copy, D2_copy, Val(:L))
rdivblockdiag!(C2_copy, D2_copy, Val(:L))
ichol!(C2_copy, Val(:L))

C2_after_ichol = to_dense(C2_copy)
println("\nActual ichol! result:")
println("  ||L_ic_scaled * L_ic_scaled' - C3_sym|| = ", norm(C2_after_ichol * C2_after_ichol' - C3_dense))
println("  ||L_ic_scaled - L_scaled_exact|| / ||L_scaled_exact|| = ", norm(C2_after_ichol - L_scaled_exact) / norm(L_scaled_exact))
println("  ||L_ic_scaled - L_manual|| = ", norm(C2_after_ichol - L_manual))

# Now scale back
lmulblockdiag!(C2, D2, Val(:L))

C2_final = to_dense(C2)
println("\nAfter scaling back (D * L_ic_scaled):")
println("  ||L_final * L_final' - (A + αI)|| = ", norm(C2_final * C2_final' - A_reg))

# Compare with the IChol preconditioner
println("\n  ||L_final - L_ic_from_prec|| = ", norm(C2_final - L_ic_dense))

# Check block structure
println("\n" * "="^60)
println("BLOCK-BY-BLOCK COMPARISON")
println("="^60)

for v in 1:nv
    cols = (v-1)*dv+1:v*dv
    for u in v:nv
        rows = (u-1)*dv+1:u*dv
        blk_ic = L_ic_dense[rows, cols]
        blk_exact = L_exact[rows, cols]

        if norm(blk_ic) > 1e-10 || norm(blk_exact) > 1e-10
            err = norm(blk_ic - blk_exact)
            println("  Block ($u,$v): ||L_ic - L_exact|| = $(round(err, sigdigits=3)), " *
                    "||L_ic|| = $(round(norm(blk_ic), sigdigits=3)), " *
                    "||L_exact|| = $(round(norm(blk_exact), sigdigits=3))")
        end
    end
end
