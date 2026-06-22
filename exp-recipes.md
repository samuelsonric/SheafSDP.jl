# Recipes: exponential-cone node objectives → conic standard form

A companion to `conic-recipes.md`, scoped to the `:EXP` cone (`cone/exp.jl`). The
exp cone earns its place wherever a node objective or private constraint is
genuinely *transcendental* — `log`, `exp`, entropy, KL divergence. The
coordination layer stays exactly as in the parent doc: a thin coboundary on the
terminals. Only the per-agent stalk changes.

Read `conic-recipes.md` first; this doc only states the deltas. Section numbers
prefixed **M** refer to the master doc (e.g. M§2 is its master template, M§12 is
its three-backend oracle).

Code references: `cone/exp.jl`, `ipm.jl`, `sheaf.jl`.

---

## 0. The cone, and the convention you must respect

`exp.jl` implements the standard exponential cone via the barrier argument
`ψ(x) = x₂ log(x₁/x₂) − x₃` (`exp_psi`). Membership is

```
(x₁, x₂, x₃) ∈ K_exp   ⟺   x₂ log(x₁/x₂) − x₃ ≥ 0,  x₁ > 0, x₂ > 0
                       ⟺   x₁ ≥ x₂ · exp(x₃ / x₂)
```

with `degree = 3`, `cachesize = 25`, and a hard `@assert n == 3`: **an `:EXP`
block is always exactly three-dimensional.** There is no vector exp cone. `K`
scalar log/exp terms means `K` separate 3-dim `:EXP` column-blocks, each with its
own leaf rows. This is the single biggest structural difference from `:POS`
(one block, any dim) and `:SOC` (one block, any arm length).

> **The reordering trap (this is M§12's "translation layer = code that can be
> wrong", made concrete).** Your solver's slot order is `(x₁, x₂, x₃)` with the
> exponential on `x₃`. JuMP/MOI `MOI.ExponentialCone()` is `(a, b, c)` with
> `c ≥ b·exp(a/b)` — the exponential is on the **first** slot. The map between
> them is
>
> ```
> solver x₁  ↔  MOI c        (the "≥" side)
> solver x₂  ↔  MOI b        (the perspective scale)
> solver x₃  ↔  MOI a        (the exponent argument)
> ```
>
> i.e. a reversal, not a shift. Every leg-R assembly that pulls cone data out of
> MOI, and every place the high-level model hands a triple to JuMP, crosses this
> seam. Unit-test it in isolation (M§12) before trusting any H-vs-R comparison.

`initp` seeds each `:EXP` block at the central point
`(1.2909…, 0.8051…, −0.8278…)` via `identity!`, so a feasible-interior start is
automatic, exactly as in M§8. You do not need to construct one.

---

## 1. The two modeling primitives (signs worked out)

Everything below is a special case of "graft a 3-dim `:EXP` leaf onto agent `i`,
then pin its three slots with leaf rows." The objective weight rides on whichever
slot is the epigraph. Derivations are short enough to keep inline so you can
re-check signs at the assembly site.

**log epigraph** `t ≤ log(r)`, `r > 0`:
`log r ≥ t ⟺ r ≥ exp(t) ⟺ r ≥ 1·exp(t/1)`. So the leaf is `(x₁,x₂,x₃) = (r, 1, t)`.
Pin `x₁ = r` (leaf row), `x₂ = 1` (leaf row, `g = 1`); `x₃` *is* `t`.

**exp epigraph** `t ≥ exp(r)`:
`t ≥ 1·exp(r/1) ⟺ (t, 1, r) ∈ K_exp`. Pin `x₂ = 1`, `x₃ = r` (leaf row); `x₁`
is `t`.

**negative-entropy / `x log x`** `x log x ≤ t`, `x > 0`:
set `x₁ = 1, x₂ = x`: `ψ = x log(1/x) − x₃ = −x log x − x₃ ≥ 0 ⟺ x log x ≤ −x₃`.
Take `x₃ = −t`: leaf is `(1, x, −t)`. Pin `x₁ = 1` (`g = 1`), `x₂ = x` (couple to
the actual probability/mass variable); `x₃ = −t`, objective rides on `−x₃`.

**KL / relative-entropy term** `p log(p/q) ≤ t`, `p > 0`, `q > 0`:
set `x₁ = q, x₂ = p`: `ψ = p log(q/p) − x₃ = −p log(p/q) − x₃ ≥ 0 ⟺ p log(p/q) ≤ −x₃`.
Take `x₃ = −t`: leaf is `(q, p, −t)`. If `q` is **data**, `x₁ = q` is a constant
leaf row (`g = q`); if `q` is a decision var, couple it. `x₂ = p` couples to the
mass variable; `x₃ = −t`.

Note `x log x` and the KL term are *literally* `exp_psi`'s functional form with
no rearrangement — these are the most natural fits the cone has.

### Degree / duality-measure bookkeeping (delta from M§3)

Each `:EXP` block adds **3** to `ν = Σ_v degree(cones[v], …)` (`conedegree` in
`ipm.jl`), so `K` exp leaves per agent add `3K(N)` to the global `ν`, and
`μ = dot(p,d)/ν` is scaled accordingly. Contrast: `:SOC` adds 2 regardless of arm
length, `:NOC` adds 0. Unlike M§3's free blocks, exp leaves *do* carry an
inequality and so *do* inflate the duality measure — correct, and expected.

---

## 2. The lifting principle for exp (delta from M§3)

The M§3 dictionary carries over unchanged except for one new row and one caveat:

| convex modeling object | lands in the solver as |
|---|---|
| `log`/`exp`/entropy/KL term in `f_i` | a **new** 3-dim `:EXP` leaf stalk per scalar term |
| affine definition of each cone slot | a **new** private leaf row (touches only agent `i`) |
| objective weight on the term | a `c` entry on the leaf's **epigraph slot** (see §1) |

**Caveat — no `Q` on `:EXP` blocks.** Curvature comes entirely from the barrier
(`exphess!` copies the cached Tunçel scaling matrix `M`). Do not put a `Q` block
on an `:EXP` stalk; leave `Q` zero there. The M§0 rule (cross-block curvature must
travel through `B`, never `Q`) is unchanged.

---

## 3. Recipe A — log-barrier minimum-fuel (build this first)

The regression test. Smallest possible diff from `test/small/lp.jl`: keep the
planar double integrator, terminal-position consensus (`b = 0`), and the residual
split `u = u⁺ − u⁻`. Replace the *hard* actuator box with a **soft log barrier in
the objective**, so the box-slack blocks `w` disappear and `:EXP` leaves take
their place.

**Modeling object.** `f_i = −Σ_t Σ_k log(ū − u_i^{t,+} − u_i^{t,-})` — control is
pushed away from saturation `±ū` by an infinite penalty at the boundary.

**Reformulation.** Minimize `Σ τ` with `τ ≥ −log(arg)`, `arg = ū − u⁺ − u⁻ > 0`.
By §1 (log epigraph, negated): `(arg, 1, −τ) ∈ K_exp`, objective rides on `−x₃`.

**Construction (per agent, per timestep `t`, per actuator channel `k`).**
- Column-blocks: states `x_i^t` (`:NOC`), `u_i^{t±}` (`:POS`), and one
  `:EXP` block `ξ_i^{t,k} = (x₁, x₂, x₃)` per channel.
- Leaf rows: `x₁ + u_i^{t,+}_k + u_i^{t,-}_k = ū` (the `arg` row, private);
  `x₂ = 1` (`g = 1`).
- Dynamics rows: unchanged from the LP recipe (`−A, I, −B, +B` on
  `x^t, x^{t+1}, u^{t+}, u^{t-}`).
- `c`: `−1` on each `x₃` slot (so `c'p = Σ x₃ = −Σ τ`, and minimizing it
  maximizes `Σ log(arg)`… check the sign: we minimize `−Σ log arg = Σ τ = −Σ x₃`,
  so `c[x₃] = −1`). Zero elsewhere — **no `ℓ₁` term**, the barrier replaces it.
- `Q`: zero.
- `g`: `c_i` on `init`, `0` on `dyn`, `ū` on each `arg` row, `1` on each `x₂` row.

**Why first.** Optimum is cross-checkable three ways (M§12): leg H is a JuMP model
with `log` in the objective (native nonlinear — independent of the conic lift),
leg R is the explicit `:EXP`-leaf JuMP model (`MOI.ExponentialCone`, minding the
§0 reordering), leg S is your `IPMProblem`. Mosek, ECOS, SCS, Clarabel all do exp
natively, so all three legs run. And it directly exercises `expmaxstep`,
`expcorr!`, `expscale!` on a problem you control.

---

## 4. Recipe B — entropic / Boltzmann effort

**Modeling object.** Nonnegative control mix `u ≥ 0`, penalty
`f_i = Σ_t Σ_k u_k log u_k` (negative entropy — concentrates control into few
channels; an information-theoretic cousin of the `ℓ₁` schedule from M§5).

**Reformulation.** Per §1 (`x log x`): leaf `(1, u_k, −τ_k)`, `τ_k ≥ u_k log u_k`,
`c` on `−x₃`. The `u_k` slot (`x₂`) couples to the control variable that also
feeds the dynamics row, so `u` is shared between its `:POS`/`:NOC` home block and
the cone's `x₂` slot via a leaf equality. Objective `Σ τ`.

Same assembly shape as Recipe A; the only change is which slot carries the
variable and that `x₁ = 1` instead of `x₂ = 1`.

---

## 5. Recipe C — KL belief consensus (the showcase)

This is the one in the spirit of the master doc, because the exp cone and the
*sheaf* structure reinforce each other instead of the cone being bolted onto an
otherwise-linear consensus.

**Setup.** Each agent carries a terminal distribution `p_i ∈ Δⁿ` (a belief,
resource split, mixed strategy). Coordination is plain linear consensus
`δ p = 0` (`b = 0`, always realizable — M§1 caveat satisfied for free). The
transcendental content is a per-agent objective `KL(p_i ‖ q_i)` against a private
prior `q_i` (data): "agents reconcile private beliefs into a consensus
distribution, each staying close to its own prior in KL."

**Construction (per agent).**
- The distribution `p_i` is an `n`-dim `:POS` block (nonnegativity built in).
- One `:EXP` leaf per coordinate `k`: `(q_{i,k}, p_{i,k}, −τ_{i,k})` per §1 (KL).
  Leaf rows: `x₁ = q_{i,k}` (`g = q_{i,k}`, data), `x₂ = p_{i,k}` (couple to the
  `:POS` block). `c = −1` on each `x₃`.
- Simplex normalization `Σ_k p_{i,k} = 1`: one private row per agent (`g = 1`).
- Coordination rows: `δ_F` on the terminal `p_i`, exactly `sheaf(I,J,V)` as in
  M§7 — the simplex is the stalk, the equality is the coboundary.

This exercises exp cones across the M§9 `N`-on-`K_N` and `T` knobs (see §8 below
for which of those actually apply) while keeping the coordination layer a textbook
coboundary.

---

## 6. What does *not* fit cleanly (so you don't lose a day)

- **Geometric-programming dynamics** (`x_{t+1} = a·x_t^α·u_t^β`). Exp-cone
  representable after a log change of variables, and superficially
  "homological-on-the-time-path" — but the change of variables makes the
  **dynamics rows nonlinear in the original states**, so `B` is no longer the
  affine coboundary the KKT path (M§0) assumes. This wants the variables to *be*
  the logs from the outset, which changes what consensus means. Different problem,
  not a drop-in.
- **Log-Chebyshev / geometric-mean objectives across agents.** These couple
  agents *through the objective*; per M§0 that curvature must travel through `B`
  as constraints, never a shared `Q`. Possible with enough leaf rows, but no
  longer a thin coordination layer.

---

## 7. The three-backend oracle for exp (delta from M§12)

The H/R/S structure is unchanged. Exp-specific notes:

- **Solvers with native exp:** Mosek, ECOS, SCS, Clarabel, Hypatia. So both leg H
  and leg R have a commercial/reference backend.
- **Make leg H independent of the lift.** Write H with `log`/`x log x` *directly*
  as a JuMP nonlinear objective, not as a cone. Then **H-vs-R** is a pure check of
  "did I reformulate the transcendental term into the cone correctly," with no
  dependence on your solver — run it first, as M§12 insists.
- **The reordering is the prime suspect at the seam.** Per §0, leg R's
  `MOI.ExponentialCone` triples are in `(a,b,c)` order while leg S's blocks are in
  `(x₁,x₂,x₃)`. Round-trip a known interior point through your MOI-↔-solver map
  and assert membership both ways *before* comparing objectives. A silent
  reordering bug produces a plausible-but-wrong optimum that H-vs-R will catch and
  H-vs-S alone will not.
- **Solution-level comparison needs a unique optimum.** KL and the log-barrier
  objectives are strictly convex in their arguments, so the optimum is unique
  (unlike the `ℓ₁` case in M§12) — solution-level `atol` comparison is meaningful
  here without an ε-regularizer.

---

## 8. Cone-level unit tests (the part that catches `exp.jl` bugs)

Objective-value agreement (M§12) tests the *assembly*; it is weak at catching
bugs *inside* `exp.jl`, because a wrong scaling can still converge to the right
optimum more slowly, or fail in a way that the `NEAR_OPTIMAL`/`STALLED` logic
masks. Test the primitives directly. These need only `LinearAlgebra` — runnable in
total isolation, no `BlockSparseArrays`/Mosek.

**Finite-difference the barrier.** For random interior `x`:
- `exp_barrier_grad!` vs FD of `F(x) = −log ψ − log x₁ − log x₂`.
- `R Rᵀ` (built from `exp_barrier_factor!`) vs FD of the gradient — this is the
  Hessian, and it is the assertion that would have caught the rank-1 `mul3!`
  issue on the first off-central probe.
- `exp_barrier_hess_dir!` `F'''[u]` vs FD of the Hessian along `u`.

**Assert the scaling secants.** After `expscale!` fills `M`, both must hold to
tolerance:
- `M x = s` (primal),
- `M δx = δs` with `δx = x − μx̃`, `δs = s − μs̃`, `μ = μv`.
These are the defining properties of the Tunçel scaling; if either fails the step
is wrong even when membership and FD pass.

**Force the off-central branch.** This is the subtle one. A start with `p ∥ d ∥`
central point makes `z = x × x̃ = 0`, `rel_z = 0`, and `expscale!` takes the
`μ F''` *fallback* — so a parallel-start test exercises the fallback only, never
the closed-form `ssᵀ/⟨x,s⟩ + δsδsᵀ/⟨δx,δs⟩ + t zzᵀ` path. Construct `p`, `d`
deliberately **non-parallel** (and not near-central) so `rel_z > sqrt(eps)`, and
assert you actually entered the Tunçel branch (e.g. check `‖x×x̃‖` is above the
guard). Otherwise coverage silently skips the branch most likely to be wrong.

**Check the fallback crossover too.** Probe `rel_z` just above and just below
`sqrt(eps)` and confirm `M` stays PD and the secants degrade gracefully across the
switch.

**Status, not just objective.** `exp_shadow_primal!` can hit its
line-search/`maxiter` warning; `expmaxstep` is bisection. Assert
`result.status ∈ (OPTIMAL, NEAR_OPTIMAL)` and that no shadow-primal warning fired,
in addition to objective agreement — a degraded path can still land near the
optimum and look fine on objective alone.

---

## 9. Scaling knobs & conditioning (delta from M§9)

| knob | applies to exp? | note |
|---|---|---|
| `N` on `K_N` | **yes** | terminal clique → dense Schur fill, same as M§9 |
| `N` on sparse `G` | **yes** | chordal-friendly, same as M§9 |
| `T` | **yes** | number of leaves grows linearly; no fill penalty |
| `m` (cone dim) | **no** | `:EXP` is fixed 3-dim — there is no arm-length axis |

So the exp tests have **two** scaling axes (`N`, `T`), not three. The "large
stalk" story from `test/large/*` does not transfer: you scale the *count* of exp
leaves, never their size.

Two conditioning facts that change test expectations relative to the symmetric
cones:

- **Tunçel scaling is quasi-Newton, not a true NT scaling.** Unlike `:SDP`/`:SOC`/
  `:POS`, the exp cone is not self-scaled; `M` is an approximation. Expect **more
  IPM iterations** and less crisp terminal convergence. Loosen `gap_tol`/`feas_tol`
  and raise `itmax` relative to the LP/SOC tests; don't treat extra iterations as a
  bug. The `analytic factor R` (condition `√cond(F'')` instead of `cond(F'')`) is
  what keeps this tractable — but it's still harder than the symmetric cones.
- **`expmaxstep` is bisection** (53 iters, no closed form), so each step costs more
  than `posmaxstep`/`socmaxstep`. Wall-clock per iteration is higher; budget for it
  in any timing comparison.

---

## 10. Pitfalls (delta from M§10)

- **Parallel start masks the real scaling path.** Covered in §8 — the most
  important exp-specific testing trap. A "passing" parallel-start test may be
  exercising only the fallback branch.
- **Slot reordering at the MOI seam (§0).** The prime suspect for an
  H-vs-R/H-vs-S disagreement. Reversal, not shift.
- **`arg → 0` blows up the barrier.** Every leaf's positive argument
  (`ū − u⁺ − u⁻`, a probability `p_k`, …) must stay strictly interior. If a
  problem can drive an argument to the boundary at optimum (e.g. a control that
  *wants* to saturate, or a coordinate that *wants* probability 0), the log
  barrier is the wrong model — it forbids the boundary. Pick instances whose
  optimum is interior, or expect `NUMERICAL_FAILURE`.
- **No vector exp.** `K` scalar terms = `K` blocks = `K` sets of leaf rows. If you
  catch yourself wanting a dim-`n` `:EXP` block, you want `n` blocks.
- **Realizability (M§1/M§10) is unaffected.** Exp lives only in objectives and
  private rows, so the coordination coboundary's `b ∈ im δ` requirement is
  untouched — consensus (`b = 0`) remains the safe default, as in every other
  recipe.

---

## 11. Suggested file layout

Mirror the existing `test/small/*` pair:

- `test/small/exp_barrier.jl` — Recipe A (§3). Three-backend oracle vs Mosek/ECOS,
  plus the cone-level unit tests of §8 inlined as a first `@testset`. This is the
  **regression test** and the thing to write first; it both validates the recipe
  and pins `exp.jl` behavior.
- `test/small/kl_consensus.jl` — Recipe C (§5). The **showcase**: exp cone and
  sheaf coordination reinforcing each other, scaled across the `N`/`T` axes of §9.

Build §8's isolated FD + secant checks before either — they need nothing but
`LinearAlgebra`, run in milliseconds, and localize a cone bug to the exact
primitive, whereas an oracle mismatch only tells you "something between the model
and the optimum is wrong."
