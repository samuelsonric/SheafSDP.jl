#
# Ruiz equilibration — conditioning stress test
#
# Holds the geometry fixed and drives the data's dynamic range up by `kappa`.
# Without equilibration the KKT operator's condition number and norm spread
# blow up like a power of kappa; equilibration keeps both ~flat.
#
# This is the "advantage" test for the static linear algebra: it isolates the
# preprocessing from the IPM iteration.
#

include("ruiz_common.jl")
using Printf

println("="^78)
println("Ruiz equilibration — conditioning collapse across a κ-sweep")
println("="^78)
println()
println("  d = free vars, k = nonneg slacks; data spans κ orders of magnitude")
println()
@printf("  %-8s | %-13s %-13s | %-13s %-13s | %-9s\n",
        "κ", "spread(raw)", "spread(ruiz)", "cond(raw)", "cond(ruiz)", "cond gain")
println("  " * "-"^74)

d, k = 6, 10
spreads_raw  = Float64[]
spreads_ruiz = Float64[]
conds_raw    = Float64[]
conds_ruiz   = Float64[]

for kappa in [1e0, 1e2, 1e4, 1e6, 1e8]
    prob, Bdense, Qdense, blocks = build_corrupted_qp(d, k, kappa; seed=11)

    # copy problem data and equilibrate in place
    B = deepcopy(prob.B)
    Q = deepcopy(prob.Q)
    c = copy(prob.c)
    g = copy(prob.g)
    n = size(B, 2)
    m = size(B, 1)
    scaling = Scaling{Float64}(n, m)
    equilibrate!(scaling, B, Q, c, g; itmax=25, tol=1e-8)

    Bh, Qh = apply_scaling_dense(scaling, Bdense, Qdense)

    sr = norm_spread(Bdense, Qdense, blocks)
    sh = norm_spread(Bh, Qh, blocks)
    cr = cond(kkt_dense(Bdense, Qdense))
    ch = cond(kkt_dense(Bh, Qh))

    push!(spreads_raw, sr); push!(spreads_ruiz, sh)
    push!(conds_raw, cr);   push!(conds_ruiz, ch)

    @printf("  %-8.0e | %-13.3e %-13.3e | %-13.3e %-13.3e | %-9.1e\n",
            kappa, sr, sh, cr, ch, cr / ch)
end

println()
println("Assertions:")
const C_PASS = Ref(0); const C_FAIL = Ref(0)
ck(name, ok) = (ok ? C_PASS[] += 1 : C_FAIL[] += 1; @printf("   %-52s %s\n", name, ok ? "PASS" : "FAIL"))

# equilibrated spread stays ~1 across the whole sweep
ck("ruiz spread ≤ 10 at every κ", all(spreads_ruiz .<= 10))
# raw spread grows with κ, ruiz spread does not
ck("raw spread grows ≥ 1e6 at κ=1e8", spreads_raw[end] >= 1e6)
ck("ruiz spread flat (max/min ≤ 10)", maximum(spreads_ruiz) / minimum(spreads_ruiz) <= 10)
# conditioning advantage widens with κ
ck("cond gain monotone in κ", all(diff(conds_raw ./ conds_ruiz) .>= 0))
ck("cond gain ≥ 1e8 at κ=1e8", (conds_raw[end] / conds_ruiz[end]) >= 1e8)
# ruiz conditioning is essentially κ-independent
ck("ruiz cond flat (max/min ≤ 1e2)", maximum(conds_ruiz) / minimum(conds_ruiz) <= 1e2)

println()
println("="^78)
@printf("RESULT: %d passed, %d failed\n", C_PASS[], C_FAIL[])
println("="^78)
@assert C_FAIL[] == 0 "ruiz conditioning tests failed"
