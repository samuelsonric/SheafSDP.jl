#
# Debug: Understand sdpcorr! with 2×2 matrices
#

using LinearAlgebra
using Random

# Weighted mean used in sdpcorr!
weightedmean(a, b, x, y) = (a * x + b * y) / (a + b)

# Start with some random PSD matrices P, D
Random.seed!(42)
A = randn(2, 2)
P = A * A' + I
A = randn(2, 2)
D = A * A' + I

println("P = "); display(P)
println("\nD = "); display(D)

# Cholesky factors
LP = cholesky(Symmetric(P)).L
LD = cholesky(Symmetric(D)).L

println("\nLP (Cholesky of P) = "); display(Matrix(LP))
println("\nLD (Cholesky of D) = "); display(Matrix(LD))

# SVD of LP' * LD
F = svd(LP' * LD)
U = F.U
s = F.S
V = F.Vt'

println("\nSVD of LP' * LD:")
println("U = "); display(U)
println("Σ = ", s)
println("V = "); display(V)

# Check: U' * LP' * LP * U vs Σ²
println("\n" * "="^50)
println("Key question: What is U' * LP' * LP * U?")
println("="^50)
UtLtLU = U' * LP' * LP * U
println("\nU' LP' LP U = "); display(UtLtLU)
println("\nΣ² = ", s.^2)
println("\nAre they equal? ", UtLtLU ≈ Diagonal(s.^2))

# Also check U' * LP' * U (without squaring L)
println("\nU' LP' U = "); display(U' * LP' * U)

# What about the full NT scaling point?
# W = R R' where R = LP * U * Σ^{-1/2}
R = LP * U * Diagonal(1 ./ sqrt.(s))
W = R * R'
println("\n" * "="^50)
println("NT scaling point W = R R' where R = LP U Σ^{-1/2}")
println("="^50)
println("\nW = "); display(W)
println("\nW⁻¹ = "); display(inv(W))

# Now let's trace through sdpcorr! with some ΔP, ΔD
println("\n" * "="^50)
println("Tracing sdpcorr!")
println("="^50)

# Random symmetric ΔP, ΔD (directions)
ΔP = randn(2,2); ΔP = (ΔP + ΔP')/2
ΔD = randn(2,2); ΔD = (ΔD + ΔD')/2
σμ = 0.5

println("\nΔP = "); display(ΔP)
println("\nΔD = "); display(ΔD)
println("\nσμ = ", σμ)

# What we want to compute: σμ I - ½(ΔP ΔD + ΔD ΔP)
target = σμ * I - (ΔP * ΔD + ΔD * ΔP) / 2
println("\nTarget: σμ I - ½(ΔP ΔD + ΔD ΔP) = "); display(target)

# Step 1: X = U' * LP⁻¹ * ΔP * ΔD * LP * U
X = U' * (LP \ (ΔP * ΔD * LP)) * U
println("\nX = U' LP⁻¹ ΔP ΔD LP U = "); display(X)

# Step 2: Build W element-wise (the sdpcorr! loop)
W_middle = zeros(2, 2)
for j in 1:2
    sj = s[j]
    for i in 1:j-1
        si = s[i]
        W_middle[i, j] = W_middle[j, i] = -weightedmean(si, sj, X[i, j], X[j, i])
    end
    W_middle[j, j] = σμ - sj^2 - X[j, j]
end
println("\nW_middle (from loop) = "); display(W_middle)

# Step 3: Transform back: result = LP⁻' * U * W_middle * U' * LP⁻¹
result = LP' \ (U * W_middle * U') / LP
println("\nResult = LP⁻' U W_middle U' LP⁻¹ = "); display(result)

# Compare to target
println("\n" * "="^50)
println("Comparison")
println("="^50)
println("\nTarget:  "); display(target)
println("\nResult:  "); display(result)
println("\nDifference: ", norm(result - target))

# If they don't match, let's see what the result actually equals
println("\n" * "="^50)
println("Analysis: What did we actually compute?")
println("="^50)

# Maybe it's related to W-scaled versions?
println("\nP ∘ D (Jordan product) = "); display((P*D + D*P)/2)
println("\nIn diagonal basis, P and D become Σ:")
println("  Σ = ", s)
println("  Σ² = ", s.^2)

# Maybe the result is W-scaled?
println("\n" * "="^50)
println("Testing W-scaled versions")
println("="^50)

Winv = inv(W)
jordan_prod = (ΔP * ΔD + ΔD * ΔP) / 2

# Option 1: W⁻¹ (σμ I - ΔP∘ΔD) W⁻¹
opt1 = Winv * (σμ * I - jordan_prod) * Winv
println("\nW⁻¹ (σμ I - ΔP∘ΔD) W⁻¹ = "); display(opt1)
println("Matches result? ", opt1 ≈ result)

# Option 2: W (σμ I - ΔP∘ΔD) W
opt2 = W * (σμ * I - jordan_prod) * W
println("\nW (σμ I - ΔP∘ΔD) W = "); display(opt2)
println("Matches result? ", opt2 ≈ result)

# Option 3: σμ W⁻² - W⁻¹ (ΔP∘ΔD) W⁻¹
opt3 = σμ * Winv^2 - Winv * jordan_prod * Winv
println("\nσμ W⁻² - W⁻¹ (ΔP∘ΔD) W⁻¹ = "); display(opt3)
println("Matches result? ", opt3 ≈ result)

# Option 4: Maybe it's scaled differently
# The Hessian in NT scaling is W⁻¹ ⊗ W⁻¹
# So the corrector might need to account for that

# Let's also check what the code ACTUALLY uses for in the IPM
# The result goes into f, which is the RHS of the Newton system
# The Newton system uses H = W⁻¹ ⊗ W⁻¹ (Hessian)

# Option 5: σμ (P⁻¹ + D⁻¹)/2 - something?
Pinv = inv(P)
Dinv = inv(D)
opt5 = σμ * (Pinv + Dinv) / 2
println("\nσμ (P⁻¹ + D⁻¹)/2 = "); display(opt5)

# Let's try: the result should work in svec form with the Hessian
# In the scaled system, directions are in W-scaled coordinates
# ΔP_scaled = W⁻¹ ΔP W⁻¹, etc.

ΔP_scaled = Winv * ΔP * Winv
ΔD_scaled = Winv * ΔD * Winv
jordan_scaled = (ΔP_scaled * ΔD_scaled + ΔD_scaled * ΔP_scaled) / 2
opt6 = σμ * I - jordan_scaled
println("\nσμ I - (W⁻¹ΔPW⁻¹) ∘ (W⁻¹ΔDW⁻¹) = "); display(opt6)
println("Matches result? ", opt6 ≈ result)

# Let me trace through more carefully what each part computes
println("\n" * "="^50)
println("Detailed trace")
println("="^50)

# The diagonal of W_middle is: σμ - sⱼ² - X[j,j]
# So W_middle[j,j] = σμ - sⱼ² - (U' LP⁻¹ ΔP ΔD LP U)[j,j]
println("\nDiagonal breakdown:")
for j in 1:2
    println("  W_middle[$j,$j] = σμ - s[$j]² - X[$j,$j]")
    println("                  = $σμ - $(s[j]^2) - $(X[j,j])")
    println("                  = $(σμ - s[j]^2 - X[j,j])")
end

# The sⱼ² term is interesting. What is Σ² in the original basis?
# s comes from SVD of LP' * LD, so Σ² relates to P and D somehow
println("\n\nWhat does Σ² represent?")
println("Σ² = diag($(s[1]^2), $(s[2]^2))")

# In the original basis: LP' LD LD' LP = U Σ² U'
# So Σ² = U' LP' LD LD' LP U = U' LP' D LP U
LPt_D_LP = LP' * D * LP
println("\nLP' D LP = "); display(LPt_D_LP)
println("\nU' (LP' D LP) U = "); display(U' * LPt_D_LP * U)
println("(Should have Σ² on diagonal)")

# Similarly: LD' LP LP' LD = V Σ² V'
# So LD' P LD = V Σ² V'
LDt_P_LD = LD' * P * LD
println("\nLD' P LD = "); display(LDt_P_LD)
println("\nV' (LD' P LD) V = "); display(V' * LDt_P_LD * V)

# So maybe the formula involves P and D explicitly?
# Let's try: σμ P⁻¹ - something
println("\n" * "="^50)
println("Trying P⁻¹ and D⁻¹ based formulas")
println("="^50)

# In NT scaling, the corrector is in the scaled space
# Maybe: LP⁻' (σμ I - ...) LP⁻¹  where the σμ I part becomes σμ (LP LP')⁻¹ = σμ P⁻¹?
opt7 = σμ * Pinv - Pinv * jordan_prod * Pinv
println("\nσμ P⁻¹ - P⁻¹ (ΔP∘ΔD) P⁻¹ = "); display(opt7)
println("Matches result? ", opt7 ≈ result)

# Or with D?
opt8 = σμ * Dinv - Dinv * jordan_prod * Dinv
println("\nσμ D⁻¹ - D⁻¹ (ΔP∘ΔD) D⁻¹ = "); display(opt8)
println("Matches result? ", opt8 ≈ result)

# Since Σ² = eigenvalues of LP' D LP, maybe:
# W_middle = σμ I - U'(LP' D LP)U - U'(LP⁻¹ ΔP ΔD LP)U
#          = U' [σμ (LP LP')⁻¹ - D - LP⁻¹(ΔP ΔD + ΔD ΔP)/2 LP⁻'] U  ???

# Let's work backwards. We have:
# result = LP⁻' U W_middle U' LP⁻¹
# So: LP' result LP = U W_middle U'
# And: W_middle = U' LP' result LP U

# What is LP' result LP?
LPt_result_LP = LP' * result * LP
println("\n" * "="^50)
println("Working backwards: LP' result LP = ")
display(LPt_result_LP)

# And U W_middle U' should equal this
U_Wmid_Ut = U * W_middle * U'
println("\nU W_middle U' = ")
display(U_Wmid_Ut)
println("\nMatch? ", LPt_result_LP ≈ U_Wmid_Ut)

# So we need to understand what W_middle is in terms of original matrices
# W_middle = σμ I - Σ² - (Jordan product term)
# But Σ² = U' LP' D LP U

# So U W_middle U' = σμ U U' - U Σ² U' - U (JP term) U'
#                  = σμ I - LP' D LP - (something)

# Let's verify:
println("\n" * "="^50)
println("Testing: U W_middle U' = σμ I - LP'D LP - ???")
println("="^50)

Sigma_sq_matrix = U * Diagonal(s.^2) * U'
println("\nU Σ² U' = LP' D LP? ")
println("U Σ² U' = "); display(Sigma_sq_matrix)
println("LP' D LP = "); display(LP' * D * LP)
println("Match? ", Sigma_sq_matrix ≈ LP' * D * LP)

# So the σμ I - Σ² part gives σμ I - LP' D LP in the U-basis
# And the X term: U X U' = U U' LP⁻¹ ΔP ΔD LP U U' = LP⁻¹ ΔP ΔD LP

# So U W_middle U' = σμ I - LP' D LP - JP_term_in_U_basis
#                  = σμ I - LP' D LP - (weighted mean stuff in U basis)

# The weighted mean with s[i], s[j] is the Jordan product in the diagonal basis!
# So: U (JP of X w.r.t. Σ) U' = ???

# Let me just compute what σμ I - LP'D LP - result of JP gives
JP_in_U_basis = zeros(2,2)
for j in 1:2
    for i in 1:j-1
        JP_in_U_basis[i,j] = JP_in_U_basis[j,i] = weightedmean(s[i], s[j], X[i,j], X[j,i])
    end
    JP_in_U_basis[j,j] = X[j,j]  # diagonal is just X[j,j]
end
println("\nJordan product of X in Σ-basis (U coords): "); display(U * JP_in_U_basis * U')

# So U W_middle U' should be:
predicted = σμ * I - Sigma_sq_matrix - U * JP_in_U_basis * U'
println("\nσμ I - U Σ² U' - U JP(X) U' = "); display(predicted)
println("\nActual U W_middle U' = "); display(U_Wmid_Ut)
println("Match? ", predicted ≈ U_Wmid_Ut)

println("\n" * "="^50)
println("FINAL FORMULA")
println("="^50)

# result = LP⁻' [σμ I - LP' D LP - U JP(X) U'] LP⁻¹
#        = σμ LP⁻' LP⁻¹ - LP⁻' LP' D LP LP⁻¹ - LP⁻' U JP(X) U' LP⁻¹
#        = σμ P⁻¹ - D - LP⁻' U JP(X) U' LP⁻¹

formula_part1 = σμ * Pinv - D
JP_term = LP' \ (U * JP_in_U_basis * U') / LP

println("\nσμ P⁻¹ - D = "); display(formula_part1)
println("\nLP⁻' U JP(X) U' LP⁻¹ = "); display(JP_term)

full_formula = formula_part1 - JP_term
println("\nσμ P⁻¹ - D - JP_term = "); display(full_formula)
println("\nActual result = "); display(result)
println("\nMatch? ", full_formula ≈ result)

# So the formula is:
# sdpcorr! computes: σμ P⁻¹ - D - (Σ-weighted JP of LP⁻¹ ΔP ΔD LP, transformed back)

# Now what is the JP term in terms of ΔP, ΔD?
# X = U' LP⁻¹ ΔP ΔD LP U
# JP(X) in Σ-basis uses weighted mean with s[i], s[j]
# Then we transform: LP⁻' U JP(X) U' LP⁻¹

# This is the Σ-weighted Jordan product of (LP⁻¹ ΔP ΔD LP),
# then conjugated by LP⁻' ... LP⁻¹

# For comparison, what is P⁻¹ (ΔP ∘ ΔD) P⁻¹?
println("\n" * "="^50)
println("Comparing to simpler formulas")
println("="^50)

simple_jp = Pinv * jordan_prod * Pinv
println("\nP⁻¹ (ΔP ∘ ΔD) P⁻¹ = "); display(simple_jp)

# What about the Σ-weighted version?
# Maybe the Σ-weighting gives us exactly this?
println("\nJP_term = "); display(JP_term)
println("\nDifference from P⁻¹ (ΔP ∘ ΔD) P⁻¹: ", norm(JP_term - simple_jp))

# So the formula might be:
# result = σμ P⁻¹ - D - P⁻¹ (ΔP ∘ ΔD) P⁻¹  ???
opt_final = σμ * Pinv - D - simple_jp
println("\nσμ P⁻¹ - D - P⁻¹(ΔP∘ΔD)P⁻¹ = "); display(opt_final)
println("Matches result? ", opt_final ≈ result)

println("\n" * "="^50)
println("CONCLUSION")
println("="^50)
println()
println("sdpcorr! computes:")
println()
println("   σμ P⁻¹ - D - (Σ-weighted JP transform)")
println()
println("NOT simply σμ I - ½(ΔP ΔD + ΔD ΔP)")
println()
println("The P⁻¹ and D terms come from the NT scaling structure.")
println()

# For the IPM, the corrector RHS is (in matrix form):
#   -D - Rd + σμ·E - ΔP ∘ ΔD
# where E is the identity element.
#
# In the scaled formulation, this becomes something involving P⁻¹.
# The σμ P⁻¹ term might be σμ times the "identity direction" in the
# P-scaled tangent space.

# Let's verify: P⁻¹ = identity direction at P in the cone
# D = current dual slack
# So: σμ P⁻¹ - D = σμ (identity at P) - D = centering toward P ∘ D = σμ I
# And the JP term corrects for the quadratic Δp ∘ Δd term

# Actually, let's check: is P⁻¹ ∘ P = I? (Is P⁻¹ the identity element in P-coords?)
PinvP_jordan = (Pinv * P + P * Pinv) / 2
println("P⁻¹ ∘ P = "); display(PinvP_jordan)
println("(Should be I if P⁻¹ is identity element at P)")

println("\n" * "="^50)
println("VERIFICATION: Is result = LP⁻ᵀ (σμ I - ΔP∘ΔD) LP⁻¹ ?")
println("="^50)

# If result = LP⁻ᵀ (σμ I - ΔP∘ΔD) LP⁻¹, then LP' result LP = σμ I - ΔP∘ΔD = target
LPt_result_LP = LP' * result * LP
println("\nLP' * result * LP = "); display(LPt_result_LP)
println("\ntarget = σμ I - ΔP∘ΔD = "); display(target)
println("\nAre they equal? ", LPt_result_LP ≈ target)
println("Difference: ", norm(LPt_result_LP - target))

# What IS LP' result LP then?
# We know result = σμ P⁻¹ - D - JP_term
# So LP' result LP = σμ LP' P⁻¹ LP - LP' D LP - LP' JP_term LP
#                  = σμ LP' LP⁻ᵀ LP⁻¹ LP - LP' D LP - ...
#                  = σμ I - LP' D LP - ...

println("\n" * "="^50)
println("What IS LP' result LP?")
println("="^50)

println("\nWe established: result = σμ P⁻¹ - D - JP_term")
println("\nSo LP' result LP = σμ LP'P⁻¹LP - LP'D LP - LP'(JP_term)LP")
println("                 = σμ I - LP'D LP - ...")

println("\nLP' D LP = "); display(LP' * D * LP)
println("\nσμ I - LP' D LP = "); display(σμ * I - LP' * D * LP)
println("\nLP' result LP = "); display(LPt_result_LP)

# The difference should be the JP term transformed
diff = (σμ * I - LP' * D * LP) - LPt_result_LP
println("\nDifference (= LP' JP_term LP) = "); display(diff)

# Compare to U J U' (the JP in the U-basis)
println("\nU J U' (JP in original coords) = "); display(U * JP_in_U_basis * U')

println("\n" * "="^50)
println("Maybe sdpcorr! computes the FULL corrector including -D?")
println("="^50)

# Full corrector (unscaled) = -D + σμ I - ΔP∘ΔD
full_corrector_unscaled = -D + σμ * I - jordan_prod
println("\nFull corrector (unscaled) = -D + σμ I - ΔP∘ΔD = ")
display(full_corrector_unscaled)

# Scaled version
full_corrector_scaled = LP' \ full_corrector_unscaled / LP
println("\nLP⁻ᵀ (full corrector) LP⁻¹ = ")
display(full_corrector_scaled)

println("\nActual result = ")
display(result)

println("\nAre they equal? ", full_corrector_scaled ≈ result)
println("Difference: ", norm(full_corrector_scaled - result))

println("\n" * "="^50)
println("What about W⁻¹ scaling?")
println("="^50)

# W⁻¹ = LP⁻ᵀ U Σ Uᵀ LP⁻¹
Winv_computed = LP' \ (U * Diagonal(s) * U') / LP
println("\nW⁻¹ = LP⁻ᵀ U Σ Uᵀ LP⁻¹ = ")
display(Winv_computed)
println("\nW⁻¹ (from inv(W)) = ")
display(Winv)
println("Match? ", Winv_computed ≈ Winv)

# Maybe: result = W⁻¹ (something) W⁻¹ ?
# Then W result W = something
W_result_W = W * result * W
println("\nW result W = ")
display(W_result_W)

# Or maybe σμ W⁻¹ - something?
println("\nσμ W⁻¹ = ")
display(σμ * Winv)

# Let's see what -D + σμ e - ΔP∘ΔD looks like with various scalings
println("\n" * "="^50)
println("Trying W-based scalings")
println("="^50)

# Full corrector with W⁻¹ scaling
opt_w = Winv * full_corrector_unscaled * Winv
println("\nW⁻¹ (-D + σμ I - ΔP∘ΔD) W⁻¹ = ")
display(opt_w)
println("Matches result? ", opt_w ≈ result)

println("\n" * "="^50)
println("EXPLICIT MATRIX FORMULA")
println("="^50)

# P = LP LP'
println("\nP = LP LPᵀ:")
println("LP * LP' = "); display(LP * LP')
println("P = "); display(P)

# So P⁻¹ = LP⁻ᵀ LP⁻¹
println("\nP⁻¹ = LP⁻ᵀ LP⁻¹:")
println("LP⁻ᵀ LP⁻¹ = "); display(LP' \ (LP \ I(2)))
println("P⁻¹ = "); display(Pinv)

# The Σ² comes from LP' D LP diagonalized
# Σ² = U' LP' D LP U
println("\nΣ² = U' LP' D LP U (diagonal):")
println("diag = ", diag(U' * LP' * D * LP * U))
println("s² = ", s.^2)

# Full formula for what sdpcorr! computes:
#
# Let X = U' LP⁻¹ ΔP ΔD LP U  (transformed product)
# Let J = Σ-weighted Jordan product of X:
#     J[j,j] = X[j,j]
#     J[i,j] = (sᵢ X[i,j] + sⱼ X[j,i]) / (sᵢ + sⱼ)  for i≠j
#
# Then W_middle = σμ I - Σ² - J (element-wise in diagonal basis)
#
# Result = LP⁻ᵀ U W_middle Uᵀ LP⁻¹

println("\n" * "="^50)
println("Algebraically, W_middle can be written as:")
println("==================================================")
println()
println("  W_middle = Uᵀ [ σμ I - LPᵀ D LP - JP_matrix ] U")
println()
println("where JP_matrix is U · J · Uᵀ with J the Σ-weighted")
println("Jordan product of X = Uᵀ LP⁻¹ ΔP ΔD LP U.")
println()
println("And the final result is:")
println()
println("  result = LP⁻ᵀ U W_middle Uᵀ LP⁻¹")
println("         = LP⁻ᵀ [ σμ I - LPᵀ D LP - JP_matrix ] LP⁻¹")
println("         = σμ P⁻¹ - D - LP⁻ᵀ JP_matrix LP⁻¹")
