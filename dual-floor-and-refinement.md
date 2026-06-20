# The dual-variable floor in the augmented KKT solve, and how to remove it with iterative refinement

## Summary

The augmented (Uzawa) KKT solver drives the **primal** equation `B·Δp = rp` to machine
precision but satisfies the **dual / stationarity** equation `A·Δp − Bᵀ·Δy = f` only to a
floor of roughly `α·‖B‖·√eps`. The augmentation parameter `α` (set from `raug`) therefore
controls a *third* quantity beyond the two conditioning numbers we already knew about — the
accuracy of the recovered dual direction — and it does so linearly. This note explains where
the floor comes from, when it actually matters, and how to buy the accuracy back with
**iterative refinement** that reuses the existing factorization. Refinement is a pure add-on:
with the round count set to 0 it is a no-op, so it can be merged without changing current
behavior.

---

## 1. Where the floor comes from

The Newton system each IPM iteration is

```
[ A  Bᵀ ] [ Δp ]   [ f  ]
[ B  0  ] [ Δy ] = [ rp ]      A = H_cone + Q
```

The Uzawa path forms the augmented block `M = A + α·BᵀB`, factorizes it (chordal Cholesky),
solves a Schur-complement system `S·Δy = r` with `S = B·M⁻¹·Bᵀ` via an inner CG, and recovers
`Δp` by back-substitution. Writing the recovered pair `(Δp, Δy)`, an exact-arithmetic algebra
of the recovery gives

```
A·Δp − Bᵀ·Δy − f  =  α·Bᵀ·(rp − B·Δp).
```

That is the whole story in one line. The right-hand side is `α·Bᵀ` times the **primal**
residual `rp − B·Δp`. The inner CG drives that primal residual down to its tolerance `ε_CG`
(default `√eps`), and **everything else is exact**, so:

- **Primal equation** `B·Δp = rp`: solved to `ε_CG` (≈ machine after the final back-solve).
- **Dual equation** `A·Δp − Bᵀ·Δy = f`: off by `α·‖B‖·ε_CG`.

This is not a conditioning effect. Even with a perfectly conditioned `M` and a perfectly
conditioned Schur complement, the dual equation is still off by `α·‖B‖·ε_CG`, because the
augmentation parameter sits *literally in front of* the primal residual in the dual equation.
That is why it is invisible to both conditioning analyses and felt like it came from nowhere.

With `α = raug·‖A‖/‖B‖²`, the floor is

```
‖dual residual‖  ≈  α·‖B‖·ε_CG  =  raug·(‖A‖/‖B‖)·ε_CG.
```

The harness reports this directly as **`KKT dual-eq res`** (`kkt_res`); the corresponding
primal quantity is **`KKT primal-eq res`** (`kkt_pres`) and goes to machine.

---

## 2. The trilemma

`raug` is not a two-way tradeoff, it is a three-way one, and the three corners do not align:

| Quantity                          | Large `α`            | Small `α`            |
|-----------------------------------|----------------------|----------------------|
| Schur complement conditioning     | **better** (CG likes it) | worse (CG stalls) |
| Conditioning of `M = A + α·BᵀB`   | worse (large rank-deficient `α·BᵀB`) | **better** (≈ `A`) |
| Dual recovery accuracy (floor)    | worse (linear in `α`) | **better** |

There is no single `α` that is simultaneously best for all three. The dual floor moves *with*
Schur conditioning and *against* augmented-matrix conditioning, so the `α` that is best for
the two conditioning numbers is generally not the `α` that is best for dual accuracy. The
SOC stall we saw at tiny `raug` was the small-`α` Schur corner; the dual floor is the
large-`α` corner. The point of iterative refinement is to **decouple the dual-accuracy corner**
so that `α` can be chosen purely for the conditioning balance.

---

## 3. When the floor actually matters (often: never)

The IPM is an **inexact Newton method**: the directions are solved approximately, and by the
Dembo–Eisenstat–Steihaug theory the convergence rate is preserved as long as the linear-solve
residual stays below a forcing fraction of the current nonlinear residual
`‖F‖ = ‖(rp, rd, complementarity)‖`.

- **Early iterations:** `‖F‖ = O(1)`, the floor `α·‖B‖·√eps ≈ 1e-5` is far beneath it. The
  directions are effectively exact and the floor is invisible.
- **Endgame:** `‖F‖` shrinks until it *meets* the floor. Only here does the floor bite.

Because the asymmetry puts the floor entirely on the dual side, the fingerprint is sharp:

```
rp  →  ~1e-14        (primal feasibility closes)
rd  →  plateaus at  α·‖B‖·√eps  and refuses to go lower
```

The outer loop re-attacks `rd` every step — the Newton step for `rd` *is* the floored dual
block — but it can never punch below the floor. Two regimes:

- **floor `< feas_tol`** → `rd` reaches tolerance, the solver converges, the dual was never a
  problem. **Do nothing.**
- **floor `> feas_tol`** → `rd` parks above tolerance, the convergence test never fires, the
  solver **stalls** with `rd` stuck.

`kkt_res` is exactly the predictor: `kkt_res < feas_tol` ⇒ safe regime; `kkt_res > feas_tol` ⇒
plan to refine.

**Likely instance of this:** the ℓ1 problem stalled at the largest conic `raug` (=1000), i.e.
the largest dual floor. Worth confirming by logging final `rp` and `rd` separately: if `rp` is
tiny and `rd` sits right at the printed `kkt_res`, the "stall" is the dual floor colliding with
`feas_tol`, not a solver bug.

---

## 4. The fix: iterative refinement

The dual residual is a vector you already hold after the solve. Form the **full KKT residual**
of the computed direction and solve for a correction with the **same factorization**:

```
ρ_p = f  − A·Δp + Bᵀ·Δy          (dual / stationarity residual; this is the floored one)
ρ_d = rp − B·Δp                  (primal residual; already ~machine)

solve   K·δ = (ρ_p, ρ_d)   using the existing augmented Uzawa solve
update  (Δp, Δy) += δ
```

No refactor. Each round is one more Uzawa solve (a Schur CG plus back-solves) applied to the
residual right-hand side. `A` is block-diagonal and `B` is sparse, so forming `ρ` is cheap.

### Why it converges in one or two rounds

Iterative refinement contracts the residual by `‖I − K·G‖` per round, where `G` is the Uzawa
solve operator. But `‖I − K·G‖` is *exactly the relative dual floor* — the residual that `G`
itself leaves — which is `α·‖B‖·ε_CG`. Therefore:

```
contraction factor per round  =  the floor itself.
```

With a floor of `3e-5`, one round takes the dual residual to `~1e-9`, a second to `~1e-13`.
The very smallness that made the floor tolerable is what makes refinement converge almost
instantly. You are not fighting the augmentation; you are using its own (small) error as the
contraction rate.

**Precondition:** the floor must be `< 1` for refinement to contract at all — trivially true in
the operating regime. It only fails when `α` is so large that `M` itself is mush, which is the
regime avoided for the *other* conditioning reason anyway.

---

## 5. The attainable lower bound (read this before claiming "machine")

Refinement does **not** reach `u ≈ 1e-16` unconditionally. The fixed-precision floor is

```
‖dual residual‖  ≳  cond(K)·u
```

where `K = [A Bᵀ; B 0]` is the saddle matrix and `u` is unit roundoff. Refinement contracts
until the correction it computes is dominated by rounding in *forming and solving against the
residual*, and that is `cond(K)·u`, not `u`.

The catch specific to IPMs: **`cond(K)` grows in the endgame, independent of `α`.** As `μ → 0`
the cone Hessian blows up (`d/p` on active POS components, `W⁻¹⊗W⁻¹` with spreading singular
values on SDP), so `cond(K)` grows like `~1/μ`. The refinement floor `cond(K)·u` is therefore
*moving up* exactly where we want to deploy refinement. Two effects race:

- dual floor being removed: `α·‖B‖·√eps` (≈ constant), vs.
- floor refinement can reach: `cond(K)·u` (growing as `~1/μ`).

So fixed-precision refinement delivers `1e-12`…`1e-14` on benign problems, degrading toward a
degenerate / low-rank-active solution where it may stall at `1e-8` or worse — possibly above
`feas_tol`. **Honest fixed-precision bound: `max(cond(K)·u, residual-evaluation error)`, not `u`.**

### Breaking the `cond(K)` ceiling: extended-precision residual

To reach *true* machine accuracy when `K` is ill-conditioned, compute the **residual** `ρ = b − K·x`
in higher precision than the solve (Wilkinson / mixed-precision refinement). Factor and back-solve
in Float64 as now, but evaluate the residual matvecs `A·Δp`, `Bᵀ·Δy`, `B·Δp` and the subtraction
in double width (Float128, compensated/Kahan, or `widen`). Then the attainable accuracy returns
to `~u` of the working precision in the leading term, as long as `cond(K) < 1/u`. Cost: one
extended-precision matvec and subtraction per round; the factorization and Schur solve stay in
Float64. Since the endgame `K` is ill-conditioned essentially by construction, this is precisely
the regime where the extended-precision residual earns its place.

**Bottom line on accuracy:**

| Variant                       | Dual residual reaches | Cost                              | Notes |
|-------------------------------|-----------------------|-----------------------------------|-------|
| Fixed-precision refinement    | `~cond(K)·u` (1e-12…1e-14 benign; worse in endgame) | 1–2 extra Uzawa solves | nearly free; reuses factor |
| Extended-precision residual   | `~u` (true machine)   | + extended-precision matvec/round | robust to endgame `cond(K)` |

---

## 6. Adaptive policy (keep speed everywhere, pay only in the endgame)

Run cheap floored solves throughout; refine only when the floor is the thing standing between
you and convergence. The trigger should be **residual-driven, not iteration-driven**:

```
refine when:  rp is already at/below feas_tol         (primal converged)
        AND:  rd has been ~flat over the last k steps  (dual stalled at the floor)
```

`rd` stalling while `rp` is already tiny is the unique fingerprint of the dual floor. This
policy:

- pays nothing on early iterations (floor invisible),
- auto-disables on easy problems where the floor was below `feas_tol` all along and `rd`
  closes on its own,
- and never touches `α`, so the Schur-vs-`M` conditioning balance stays exactly where it was
  tuned.

A `max_refine_rounds = 0` default makes the whole feature a no-op until switched on.

---

## 7. Implementation sketch

Reuse the factorization already living in the Uzawa workspace; only the residual RHS changes.

```julia
# After the normal corrector solve, (Δp, Δy, Δd) are in hand and the augmented
# factor F (of M = A + α BᵀB) is current for this iteration.

"""
    refine_kkt!(Δp, Δy, wrk, set, A, B, f, rp; rounds=0, extended=false)

Iterative refinement of the KKT direction (Δp, Δy) against
    A Δp − Bᵀ Δy = f ,   B Δp = rp
reusing the augmented factorization in `wrk`. Returns the achieved dual residual.
`rounds = 0` is a no-op. Set `extended = true` to evaluate the residual in
higher precision (needed to beat cond(K)·u in the endgame).
"""
function refine_kkt!(Δp, Δy, wrk, set, A, B, f, rp; rounds::Int = 0, extended::Bool = false)
    rounds == 0 && return nothing
    n = length(Δp); m = length(Δy)
    ρp = similar(f)        # dual / stationarity residual
    ρd = similar(rp)       # primal residual
    δp = similar(Δp); δy = similar(Δy); δd = similar(Δp)

    for _ in 1:rounds
        # ρp = f − A Δp + Bᵀ Δy ;  ρd = rp − B Δp
        # (compute these in extended precision when `extended`, then narrow)
        residual_dual!(ρp, f, A, B, Δp, Δy; extended)     # ρp = f − A Δp + Bᵀ Δy
        residual_primal!(ρd, rp, B, Δp; extended)         # ρd = rp − B Δp

        # solve K δ = (ρp, ρd) with the SAME augmented factor; newton!-style.
        # δy carries the sign convention used by newton! (lmul!(-1, ...)).
        newton!(δp, δy, δd, wrk, set, A, B, ρp, ρd, /*Q=*/zero_or_Q)

        @. Δp += δp
        @. Δy += δy
        # if you also track Δd, refine it from the corrected (Δp, Δy):
        # Δd  = rd − Bᵀ Δy + Q Δp   (recompute, do not accumulate δd blindly)
    end

    # report the achieved dual residual for logging / trigger
    residual_dual!(ρp, f, A, B, Δp, Δy; extended)
    return norm(ρp)
end
```

Notes:

- The inner solve is the existing `newton!` / `solve_kkt!` with the residual as RHS — same
  factor `F`, same Schur CG. No new linear algebra primitives.
- `Δd` is recovered from the corrected `(Δp, Δy)` via the standard
  `Δd = rd − Bᵀ·Δy + Q·Δp`, not by accumulating a `δd`; this keeps `Δd` consistent with the
  refined primal/dual pair.
- `residual_dual!` is where the extended-precision option lives: evaluate `A·Δp`, `Bᵀ·Δy` and
  the subtraction in `widen(T)` (or a compensated dot), then narrow. Everything else stays in `T`.
- Hook the trigger into the outer loop using the existing history: detect `rp ≤ feas_tol` and
  `rd` flat over the stall window, then call `refine_kkt!` with `rounds = 1` or `2`.

---

## 8. Validating it with the harness

`invariants.jl` already instruments both halves of the story:

- **`KKT dual-eq res`** = the floor itself. After refinement it should drop by
  `~1/floor` per round (e.g. `3e-5 → 1e-9 → 1e-13`).
- **`KKT primal-eq res`** = already machine; should stay there.
- Final **`rd`** vs **`rp`**: the trigger fingerprint. Log them per iteration (unconditionally —
  for a study, pull the KKT-residual computation out of the `nrp > feas_tol` gate so the endgame
  iterations are not filtered out).

Suggested experiments:

1. **Confirm the diagnosis.** On the stalling ℓ1 run, log `rp`, `rd`, `kkt_res` every iteration.
   Expect `rp → machine`, `rd` flat at `≈ kkt_res`.
2. **Measure the refinement contraction.** Plot dual residual vs refinement round at a few outer
   iterations. The slope is the floor; where it *stalls* is `cond(K)·u`. If the stall tracks
   `1/μ`, you have measured `cond(K)·u` directly and know whether fixed precision suffices or the
   endgame wants the extended-precision residual.
3. **Map the trilemma.** Sweep `raug` on one fixed problem; per `α` record `kkt_res` (dual floor,
   linear in `α`), `worst CG iterations` (Schur conditioning), and a smallest-pivot / condition
   readout of `M = A + α·BᵀB` (augmented-matrix conditioning). Three curves, one plot — the
   trilemma made measurable, and the gap between "best `α` for iterations" and "best `α` for the
   dual floor" is the quantitative cost the refinement buys back.

---

## 9. One-paragraph version

The augmented solve leaves the dual equation off by `α·‖B‖·√eps` because that augmentation
parameter multiplies the (small) primal residual in the dual equation; it is a consistency
floor, not a conditioning effect, and it is the third, previously-hidden corner of the `raug`
trilemma. It only bites in the endgame, where its signature is `rp → machine` while `rd`
plateaus; if that plateau is below `feas_tol` you never needed an accurate dual. When it is
above `feas_tol`, iterative refinement against the existing factorization removes it, contracting
at a rate equal to the floor itself (so 1–2 rounds), down to `cond(K)·u` in fixed precision or
true machine precision with an extended-precision residual. Gate it behind a residual-driven
trigger with a `rounds = 0` default and it costs nothing until the endgame of a hard problem.
