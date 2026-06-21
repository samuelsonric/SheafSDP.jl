using SheafSDP
using SheafSDP: identity!, EXP
using CommonSolve: solve
using LinearAlgebra
using BlockSparseArrays: blocksparse

println("=== Paper Example (19): EXP cone integration test ===")
println("min x₁ + x₂  s.t. x₁ + x₂ + x₃ = 1, x ∈ K_exp\n")

# Build the problem
# min c'p  s.t. Bp = g, p ∈ K_exp
# Here: c = [1, 1, 0], g = [1], B = [1 1 1] (1×3 matrix)

T = Float64
c = T[1, 1, 0]
g = T[1]

# B is a 1×1 block matrix with one 1×3 block
row_ids = [1]
col_ids = [1]
blocks = [reshape(T[1, 1, 1], 1, 3)]
B = blocksparse(row_ids, col_ids, blocks)

# Q = 0 (no quadratic term)
Q = SheafSDP.allocblockdiag(B)
fill!(Q, zero(T))

# Single EXP cone
cones = [:EXP]

prob = IPMProblem(c, g, B, Q, cones)

println("Problem built. Running IPM with verbose output...\n")

# Run with verbose output to see gap sequence
# Use looser tolerances for now (conditioning issues at high precision)
settings = IPMSettings{T}(
    feas_tol=1e-6,
    gap_tol=1e-6,
    itmax=20,
    verbose=true
)

result = solve(prob, settings)

println("\nFinal status: ", result.status)
println("Iterations: ", result.iterations)
println("\nFinal solution p = ", result.p)
println("Final dual d = ", result.d)
println("Objective: ", dot(c, result.p))

# Check constraint
Bp = B.val[1] * result.p[1] + B.val[2] * result.p[2] + B.val[3] * result.p[3]
println("\nConstraint check: Bp = ", Bp, " (should be 1.0)")

# Expected gap sequence from paper Table 1:
# k=0: 4.0e+00
# k=1: 9.3e-01
# k=2: 4.3e-02
# k=3: 2.1e-04
# k=4: 2.7e-08
# k=5: 5.5e-10
# k=6: 8.4e-15
println("\n=== Expected gap sequence from paper Table 1 ===")
println("k=0: 4.0e+00")
println("k=1: 9.3e-01")
println("k=2: 4.3e-02")
println("k=3: 2.1e-04")
println("k=4: 2.7e-08")
println("k=5: 5.5e-10")
println("k=6: 8.4e-15")
