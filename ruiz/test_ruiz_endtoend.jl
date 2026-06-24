#
# Ruiz equilibration — end-to-end IPM stress test
#
# Solves the same badly-scaled conic QP two ways across a κ-sweep:
#   (a) solve(prob, settings_plain)  — scale_itmax=0, no equilibration
#   (b) solve(prob, settings_equil)  — scale_itmax>0, with equilibration
#
# The problem is a fixed well-scaled QP corrupted by a known scaling spanning κ
# orders of magnitude, so all κ share the *same* optimum (in mapped coords) and
# the objective is directly comparable. The advantage shows up as: equilibrated
# stays OPTIMAL with stable iteration / KKT counts and tiny residuals, while the
# plain solve needs more work and loses accuracy (or fails) as κ grows.
#

include("ruiz_common.jl")
using Printf

println("="^92)
println("Ruiz equilibration — end-to-end solve: plain vs equilibrated")
println("="^92)
println()
@printf("  %-7s | %-22s | %-22s | %-10s\n", "κ",
        "plain (it/kkt/status)", "equil (it/kkt/status)", "obj match")
println("  " * "-"^88)

d, k = 6, 10

# plain: no equilibration
settings_plain = IPMSettings{Float64}(
    kkt = UzawaSettings{Float64}(raug = 1e5),
    feas_tol = 1e-8, gap_tol = 1e-8, itmax = 200,
    scale_itmax = 0,
)

# equilibrated: with equilibration
settings_equil = IPMSettings{Float64}(
    kkt = UzawaSettings{Float64}(raug = 1e5),
    feas_tol = 1e-8, gap_tol = 1e-8, itmax = 200,
    scale_itmax = 20,
)

rows = NamedTuple[]

for kappa in [1e0, 1e2, 1e4, 1e6, 1e8]
    prob, _, _, _ = build_corrupted_qp(d, k, kappa; seed=23)

    res_p = solve(prob, settings_plain)
    res_e = solve(prob, settings_equil)

    obj_p = objective(prob, res_p.p)
    obj_e = objective(prob, res_e.p)

    rp_e = primal_res(prob, res_e.p) / (1 + norm(prob.g))
    rd_e = dual_res(prob, res_e.p, res_e.d, res_e.y) / (1 + norm(prob.c))

    objmatch = abs(obj_p - obj_e) / max(abs(obj_e), 1.0)

    push!(rows, (kappa=kappa,
                 it_p=res_p.iterations, kkt_p=res_p.kkt_iters, st_p=res_p.status,
                 it_e=res_e.iterations, kkt_e=res_e.kkt_iters, st_e=res_e.status,
                 rp_e=rp_e, rd_e=rd_e, objmatch=objmatch))

    @printf("  %-7.0e | %3d / %5d / %-9s | %3d / %5d / %-9s | %.2e\n",
            kappa, res_p.iterations, res_p.kkt_iters, res_p.status,
            res_e.iterations, res_e.kkt_iters, res_e.status, objmatch)
end

println()
println("Equilibrated residuals across the sweep:")
for r in rows
    @printf("   κ=%-7.0e  ‖rp‖=%.2e  ‖rd‖=%.2e\n", r.kappa, r.rp_e, r.rd_e)
end

println()
println("Assertions:")
const E_PASS = Ref(0); const E_FAIL = Ref(0)
ek(name, ok) = (ok ? E_PASS[] += 1 : E_FAIL[] += 1; @printf("   %-56s %s\n", name, ok ? "PASS" : "FAIL"))

# equilibrated solver stays healthy at every κ
ek("equilibrated OPTIMAL or NEAR_OPTIMAL ∀κ",
   all(r -> r.st_e in (SheafSDP.OPTIMAL, SheafSDP.NEAR_OPTIMAL), rows))
ek("equilibrated primal residual ≤ 1e-6 ∀κ", all(r -> r.rp_e ≤ 1e-6, rows))
ek("equilibrated dual residual ≤ 1e-6 ∀κ",   all(r -> r.rd_e ≤ 1e-6, rows))

# objectives agree whenever the plain solve also converged
converged_plain = filter(r -> r.st_p in (SheafSDP.OPTIMAL, SheafSDP.NEAR_OPTIMAL), rows)
ek("obj(plain) ≈ obj(equil) where plain converged",
   all(r -> r.objmatch ≤ 1e-4, converged_plain))

# the optimum is κ-independent (mapped); equilibrated objective must be stable
objs_e = [objective(build_corrupted_qp(d, k, r.kappa; seed=23)[1],
                    solve(build_corrupted_qp(d, k, r.kappa; seed=23)[1], settings_equil).p)
          for r in rows]
ek("equilibrated objective κ-stable (spread ≤ 1e-3)",
   maximum(objs_e) - minimum(objs_e) ≤ 1e-3 * (1 + abs(objs_e[1])))

# advantage: at the largest κ, equilibration does no more KKT work than plain
ek("equil KKT work ≤ plain at κ=1e8 (or plain not optimal)",
   rows[end].kkt_e ≤ rows[end].kkt_p ||
   !(rows[end].st_p in (SheafSDP.OPTIMAL, SheafSDP.NEAR_OPTIMAL)))

println()
println("="^92)
@printf("RESULT: %d passed, %d failed\n", E_PASS[], E_FAIL[])
println("="^92)
@assert E_FAIL[] == 0 "ruiz end-to-end tests failed"
