using SheafSDP
using SheafSDP: IChol, blocktri
using LinearAlgebra
using Random

Random.seed!(42)

# Small path graph
nv = 5
dv, de = 4, 3
src, dst = Int[], Int[]
for i in 1:nv-1
    push!(src, i); push!(dst, i+1)
    push!(src, i+1); push!(dst, i)
end
maps = [randn(de, dv) for _ in 1:length(src)]
B = sheaf(src, dst, maps)

n = size(B, 2)
A = B' * B
α = 0.1
A_reg = Matrix(A) + α * I

# Build IChol
P_ic = IChol(B; α=α)

# Test preconditioner solve: M^{-1} b where M = L L'
x_true = randn(n)
b = A_reg * x_true

# Apply preconditioner
y = similar(b)
ldiv!(y, P_ic, b)

# If IChol is perfect, y ≈ x_true (since M ≈ A)
println("||M^{-1} A x - x|| / ||x|| = ", norm(y - x_true) / norm(x_true))

# Check residual after one preconditioner application
# r = b - A * (M^{-1} b) = b - A y
r = b - A_reg * y
println("||b - A (M^{-1} b)|| / ||b|| = ", norm(r) / norm(b))

# Compare with exact inverse
x_exact = A_reg \ b
println("||A^{-1} b - x_true|| / ||x_true|| = ", norm(x_exact - x_true) / norm(x_true))

# Trace through the solve step by step
println("\n--- Tracing ldiv! ---")
L = LowerTriangular(P_ic.L)

# Step 1: y = b
y2 = copy(b)
println("After copy: ||y|| = ", norm(y2))

# Step 2: L y = b, solve for y
println("Before L solve: ||y|| = ", norm(y2))
ldiv!(L, y2)
println("After L solve: ||y|| = ", norm(y2))

# Step 3: L' y = prev_y, solve for y
println("Before L' solve: ||y|| = ", norm(y2))
ldiv!(L', y2)
println("After L' solve: ||y|| = ", norm(y2))

# Check if this matches the original ldiv!
println("\n||y2 - y|| = ", norm(y2 - y))

# Now manually check with dense matrices
L_dense = LowerTriangular(Matrix(P_ic.L))
y3 = copy(b)
ldiv!(L_dense, y3)
ldiv!(L_dense', y3)
println("\nUsing dense LowerTriangular:")
println("||y_dense - y_block|| = ", norm(y3 - y))
println("||M_dense^{-1} b - x|| / ||x|| = ", norm(y3 - x_true) / norm(x_true))

# Check the factor itself
println("\n--- Checking IChol factor ---")
L_ic = LowerTriangular(Matrix(P_ic.L))
M_ic = L_ic * L_ic'
println("||L L' - A_reg|| / ||A_reg|| = ", norm(M_ic - A_reg) / norm(A_reg))

# Compute exact Cholesky for comparison
L_exact = cholesky(Symmetric(A_reg, :L)).L
println("||L_ic - L_exact|| / ||L_exact|| = ", norm(L_ic - L_exact) / norm(L_exact))

# Check if it's the Jacobi scaling that's wrong
println("\n--- Checking intermediate steps ---")
using SheafSDP: blocktri, copyblockdiag, cholblockdiag!, ldivblockdiag!, rdivblockdiag!, lmulblockdiag!

# Trace through IChol construction
C = blocktri(B' * B, Val(:L))
using LinearAlgebra: axpy!
axpy!(α, I, C)

# Before scaling
println("C before scaling, diag block 1:")
println("  ", round.(Matrix(C)[1:dv, 1:dv], digits=3))

D = copyblockdiag(C)
cholblockdiag!(D, :L)
ldivblockdiag!(C, D, Val(:L))
rdivblockdiag!(C, D, Val(:L))

# After scaling
println("\nC after scaling, diag block 1 (should be I):")
println("  ", round.(Matrix(C)[1:dv, 1:dv], digits=3))

# Run ichol! - but first let's trace which α it uses
using SheafSDP: ICHOL_SCHEDULE
println("\nICHOL_SCHEDULE: ", ICHOL_SCHEDULE)

# Try ichol_impl! manually with α=0 to see if it succeeds
W = similar(C)
using SheafSDP: ichol_impl!
copyto!(W, C)
success_alpha0 = ichol_impl!(W, 0.0, Val(:L))
println("ichol_impl! with α=0 succeeded: ", success_alpha0)

if !success_alpha0
    # Try with larger α
    for α_try in ICHOL_SCHEDULE[2:end]
        copyto!(W, C)
        success = ichol_impl!(W, α_try, Val(:L))
        if success
            println("ichol_impl! succeeded with α=$α_try")
            break
        end
    end
end

# Now run actual ichol!
using SheafSDP: ichol!
C2 = blocktri(B' * B, Val(:L))
axpy!(α, I, C2)
D2 = copyblockdiag(C2)
cholblockdiag!(D2, :L)
ldivblockdiag!(C2, D2, Val(:L))
rdivblockdiag!(C2, D2, Val(:L))
ichol!(C2, Val(:L))

# The result should match W if α=0 worked
println("||W - C2|| after ichol!: ", norm(Matrix(W) - Matrix(C2)))

println("\nC2 after ichol!:")
for v in 1:nv
    rows = (v-1)*dv+1:v*dv
    println("  Block ($v,$v): ", round.(diag(Matrix(C2)[rows, rows]), digits=3))
end

# Scale back
lmulblockdiag!(C2, D2, Val(:L))

println("\nC2 after scaling back:")
L_final = LowerTriangular(Matrix(C2))
for v in 1:nv
    rows = (v-1)*dv+1:v*dv
    L_blk = L_final[rows, rows]
    L_ex_blk = L_exact[rows, rows]
    println("  Block ($v,$v): ||L - L_exact|| = ", round(norm(L_blk - L_ex_blk), digits=4))
end

println("\nFull factor comparison:")
println("  ||LowerTri(C) - L_exact|| / ||L_exact|| = ", norm(L_final - L_exact) / norm(L_exact))
println("  ||LowerTri(C) * LowerTri(C)' - A_reg|| / ||A_reg|| = ", norm(L_final * L_final' - A_reg) / norm(A_reg))

# Compare with P_ic.L
println("\n||C2 - P_ic.L|| = ", norm(Matrix(C2) - Matrix(P_ic.L)))
