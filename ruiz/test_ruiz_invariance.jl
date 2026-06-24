#
# Ruiz equilibration — invariance / correctness unit tests
#
# Verifies the properties the scaling MUST satisfy regardless of conditioning:
#   1. norm balancing      — after equilibration, row & block ∞-norms ≈ 1
#   2. round-trip          — unscale(scale(v)) == v  (p, d, y all recover)
#   3. pairing invariance  — ⟨p̂, d̂⟩ == ⟨p, d⟩
#   4. cone membership     — block-constant column scaling keeps p ∈ K, d ∈ K*
#   5. data consistency    — B̂ = E B D, ĝ = E g, ĉ = D c hold exactly
#

include("ruiz_common.jl")
using Printf

const PASS = Ref(0)
const FAIL = Ref(0)
check(name, ok) = (ok ? PASS[] += 1 : FAIL[] += 1; @printf("   %-46s %s\n", name, ok ? "PASS" : "FAIL"))

println("="^70)
println("Ruiz equilibration — invariance & correctness")
println("="^70)

d, k = 5, 8
prob, Bdense, Qdense, blocks = build_corrupted_qp(d, k, 1e6; seed=7)

# ---- 1. norm balancing -----------------------------------------------------
println("\n1. Norm balancing (row & block ∞-norms → 1):")
# copy problem data and equilibrate in place
B = deepcopy(prob.B)
Q = deepcopy(prob.Q)
c = copy(prob.c)
g = copy(prob.g)
n = size(B, 2)
m = size(B, 1)
scaling = Scaling{Float64}(n, m)
equilibrate!(scaling, B, Q, c, g; itmax=20, tol=1e-6)

Bh, Qh = apply_scaling_dense(scaling, Bdense, Qdense)

spread_before = norm_spread(Bdense, Qdense, blocks)
spread_after  = norm_spread(Bh, Qh, blocks)
@printf("   spread before = %.3e\n", spread_before)
@printf("   spread after  = %.3e\n", spread_after)
check("spread collapses below 10", spread_after < 10)
check("spread improved by ≥ 1e3", spread_before / spread_after > 1e3)

# ---- 2. round-trip ---------------------------------------------------------
println("\n2. Round-trip  unscale(scale(v)) == v:")
p̂ = randn(n); d̂ = randn(n); ŷ = randn(m)
p = copy(p̂); dd = copy(d̂); y = copy(ŷ)
unscale!(p, dd, y, scaling)
# re-apply the forward map and compare
p̂2 = p ./ scaling.cscl
d̂2 = scaling.cscl .* dd
ŷ2 = y ./ scaling.rscl
check("primal round-trip", maximum(abs, p̂2 .- p̂) ≤ 1e-10 * (1 + maximum(abs, p̂)))
check("dual   round-trip", maximum(abs, d̂2 .- d̂) ≤ 1e-10 * (1 + maximum(abs, d̂)))
check("y      round-trip", maximum(abs, ŷ2 .- ŷ) ≤ 1e-10 * (1 + maximum(abs, ŷ)))

# ---- 3. pairing invariance -------------------------------------------------
println("\n3. Pairing invariance ⟨p̂,d̂⟩ == ⟨p,d⟩:")
pair_hat = dot(p̂, d̂)
pair_org = dot(p, dd)
relerr = abs(pair_hat - pair_org) / max(abs(pair_org), eps())
@printf("   rel err = %.3e\n", relerr)
check("pairing preserved", relerr < 1e-12)

# ---- 4. cone membership under block-constant scaling -----------------------
println("\n4. Cone membership preserved by per-block scalars:")
# nonneg block: scaling is one positive scalar ⇒ sign preserved
xrange = collect(colrange(prob.B, 1))
srange = collect(colrange(prob.B, 2))
t_pos = scaling.cscl[first(srange)]   # scaling on POS block
check("POS block scalar > 0", t_pos > 0)
# a primal POS point stays nonneg after column scaling
spos = abs.(randn(k)) .+ 0.1
check("D·s ≥ 0 stays in K₊", all((scaling.cscl[srange] .* spos) .>= 0))
# scol is constant within each block (the defining property)
check("scol constant on free block",  maximum(scaling.cscl[xrange]) ≈ minimum(scaling.cscl[xrange]))
check("scol constant on nonneg block", maximum(scaling.cscl[srange]) ≈ minimum(scaling.cscl[srange]))

# ---- 5. data consistency B̂ = E B D, ĝ = E g, ĉ = D c -------------------------
println("\n5. Scaled-data consistency:")
Bh_expect = (scaling.rscl .* Bdense) .* scaling.cscl'
# pull B̂ out of the scaled copy densely
Bh_actual = todense(B)
check("B̂ == E B D", maximum(abs, Bh_actual .- Bh_expect) ≤ 1e-10 * (1 + maximum(abs, Bh_expect)))
check("ĝ == E g",  maximum(abs, g .- scaling.rscl .* prob.g) ≤ 1e-10 * (1 + maximum(abs, prob.g)))
check("ĉ == D c", maximum(abs, c .- scaling.cscl .* prob.c) ≤ 1e-10 * (1 + maximum(abs, prob.c)))

println("\n" * "="^70)
@printf("RESULT: %d passed, %d failed\n", PASS[], FAIL[])
println("="^70)
@assert FAIL[] == 0 "ruiz invariance tests failed"
