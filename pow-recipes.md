# Recipes: power-cone node objectives → conic standard form

A companion to `conic-recipes.md`, scoped to the `:POW` cone (`cone/pow.jl`). The
power cone earns its place wherever a node objective or private constraint is
genuinely a **power law** — an `ℓ_p` norm (`p ≠ 2`), a fractional-power running
cost, a geometric mean, a Cobb–Douglas product. The one-line justification for
the whole cone: your existing recipes already pin three corners of the same
family — min-fuel is `ℓ₁` (`:POS`/LP), min-energy is `ℓ₂` (`:SOC`), min-peak is
`ℓ∞` (box/`:POS`) — and `:POW` is the **`p ≠ 2` interior** they bracket but can't
reach. The coordination layer stays exactly as in the parent doc: a thin linear
coboundary on the terminals. Only the per-agent stalk changes.

Read `conic-recipes.md` first, and `exp-recipes.md` second — this doc is closer
to the exp companion than to the master, and only states the deltas. Section
numbers prefixed **M** refer to the master doc; **E** refers to `exp-recipes.md`.

Code references: `cone/pow.jl`, `ipm.jl`, `sheaf.jl`.

---

## 0. The cone, and the conventions you must respect

`pow.jl` implements the 3-D power cone via the barrier argument
`φ(x) = x₁^(2α) x₂^(2(1-α)) − x₃²` (`powphi`). Membership is

```
(x₁, x₂, x₃) ∈ P_α   ⟺   x₁^α x₂^(1-α) ≥ |x₃|,  x₁ > 0, x₂ > 0,   α ∈ (0,1)
```

with `degree = 3`, `cachesize = 25`, and a hard `@assert n == 3`: **a `:POW`
block is always exactly three-dimensional.** As with `:EXP`, there is no vector
power cone — `K` power terms means `K` separate 3-D `:POW` column-blocks, each
with its own leaf rows.

Three conventions decide whether the assembly is honest. Two of them are *easier*
than the exp cone; one is genuinely new.

> **The `α` field — the new thing, and why the interface already handles it.**
> Unlike `:EXP` (a singleton — every exponential cone is identical), the power
> cone is a **one-parameter family**: each `α` is a distinct cone, so the type
> carries it (`PowerCone{T}(α)`, validated `0 < α < 1` in the constructor). The
> nonsymmetric notes flag this as the "POW carries a parameter" wrinkle (§8.4
> there), worried about a `Vector{Symbol}` + `tocone` interface that assumes
> nullary tags. **That wrinkle is already resolved in the code you have:** the
> `IPMProblem` consumes `cones::Vector{Cone}` of cone *objects*, so
> `PowerCone(α)` drops straight in next to `PositiveCone()`, `SecondOrderCone()`,
> `ExponentialCone()`, … and `α` rides inside the object. No `Dict{Int,Float64}`,
> no `Union`, no parallel channel. Each `:POW` block may carry its **own** `α` —
> which the geometric-mean tower in §5 exploits directly.

> **The slot/exponent seam — aligned, not reversed (contrast E§0).** The exp
> cone's MOI translation is a *reversal* (`(x₁,x₂,x₃) ↔ (c,b,a)`), the prime
> suspect for every H-vs-R disagreement. The power cone is kinder:
> `MOI.PowerCone(α)` is `(x,y,z)` with `x^α y^(1-α) ≥ |z|`, **the same slot order
> and the same exponent** as `pow.jl`. The map is the identity, not a permutation.
> Two things still need pinning at the seam, so unit-test them in isolation
> (M§12) before trusting any comparison: (i) `α` vs `1-α` — which arm is the
> `α`-weighted one — and (ii) that the `|·|` coordinate is slot 3 on both sides.
> Note also that the `:SOC` test's isometric `1/√2` rescale (`B_on_ζ`,
> `extract_u_box` in `test/small/soc.jl`) **does not recur**: power-cone
> membership matches MOI's verbatim, so the dynamics/cost read slot values with
> no scaling factor. Do not cargo-cult the `√2`.

`identity!` seeds each `:POW` block at the closed-form central point
`(√(1+α), √(2-α), 0)` (verified `x₀ + F′(x₀) = 0`, `⟨x₀,s₀⟩ = 3 = ϑ`), so a
feasible-interior start is automatic, exactly as in M§8 — and unlike `:EXP`,
which needs an offline-computed start. You do not construct one.

---

## 1. The three modeling primitives (signs worked out)

Everything below is "graft a 3-dim `:POW` leaf onto agent `i`, then pin its three
slots with leaf rows." The objective weight rides on whichever slot is the
epigraph. All three lifts were checked numerically; the `α` is stated for each.

**Power epigraph** `t ≥ |u|^p`, `p > 1`:
`|u| ≤ t^{1/p} = t^{1/p}·1^{1-1/p} ⟺ (t, 1, u) ∈ P_{1/p}`. Leaf
`(x₁,x₂,x₃) = (t, 1, u)`. Pin `x₂ = 1` (leaf row, `g = 1`); `x₃` *is* `u` (read by
dynamics, exactly as `soc.jl` reads `u` from the tail of `ζ`); `x₁` is `t`,
objective rides on `+x₁`. **`α = 1/p`.** This is the most natural fit the cone has
— a single scalar power, no auxiliary blocks.

**p-norm epigraph** `‖u‖_p ≤ t`, `u ∈ ℝ^m`, `p > 1`:
introduce `r_k ≥ 0` and require, per channel, `(r_k, t, u_k) ∈ P_{1/p}` (i.e.
`r_k^{1/p} t^{1-1/p} ≥ |u_k|`) together with `Σ_k r_k = t`. Then
`Σ_k|u_k|^p ≤ Σ_k r_k t^{p-1} = t^p`, so `‖u‖_p ≤ t` (verified, both directions;
tight at `r_k = |u_k|^p / t^{p-1}`). **`α = 1/p`.** The `m` channels couple
through the **shared** `t` — that coupling is the honest cost of a *norm* (vs the
separable sum-of-powers above), and the structure §3/§4 split on.

**Weighted geometric mean / Cobb–Douglas** `t ≤ x₁^α x₂^(1-α)`:
`(x₁, x₂, t) ∈ P_α` directly — the cone *is* this constraint. Maximize `t`
(objective on `+x₃`). The `m`-term equal-weight mean `(∏_k x_k)^{1/m} ≥ t` builds
as a small **tower** of 2-arg power cones with **mixed `α`** (verified: 3-term
mean = `P_{1/2}` feeding `P_{1/3}`; 4-term = two `P_{1/2}` feeding `P_{1/2}`).
This is the lift that exercises the `α`-as-field machinery — see §5.

### Degree / duality-measure bookkeeping (delta from M§3)

Each `:POW` block adds **3** to `ν = Σ_v degree(cones[v], …)` (`conedegree` in
`ipm.jl`), identical to `:EXP`. `K` power leaves per agent add `3K(N)` to the
global `ν`, and `μ = ⟨p,d⟩/ν` scales accordingly. Power leaves carry an
inequality and so *do* inflate the duality measure — correct and expected.

---

## 2. The lifting principle for power (delta from M§3)

The M§3 dictionary carries over with one new row, one caveat, and one extra
bookkeeping column relative to exp:

| convex modeling object | lands in the solver as |
|---|---|
| `ℓ_p` / `|·|^p` / geo-mean / Cobb–Douglas term in `f_i` | a **new** 3-dim `:POW` leaf stalk per scalar relation, **tagged with its `α`** |
| affine definition of each cone slot | a **new** private leaf row (touches only agent `i`) |
| objective weight on the term | a `c` entry on the leaf's **epigraph slot** (see §1) |

**The extra column is `α`.** Because the cone is a family, the lift must record
*which* `α` each leaf carries, and the builder must place the matching
`PowerCone(α)` object in `cones`. For a single-`p` objective this is one constant;
for a geometric-mean tower (§5) it is a small set of distinct values.

**Caveat — no `Q` on `:POW` blocks.** Curvature comes entirely from the barrier
(`powscale!` fills the Tunçel scaling `M`). Do not put a `Q` block on a `:POW`
stalk; leave it zero there. The M§0 rule (cross-block curvature travels through
`B`, never `Q`) is unchanged.

---

## 3. Recipe A — minimum-effort `ℓ_p^p` control (build this first)

The regression test, and the smallest possible diff from `test/small/qp.jl` /
`soc.jl`: keep the planar double integrator, terminal-position consensus
(`b = 0`), and replace the energy/fuel objective with a **separable**
`ℓ_p`-power cost. Separable because each scalar control gets its own epigraph —
no shared `t` — which is the lightest assembly and the cleanest first probe of
`powscale!`, `powmaxstep`, `powcorr!`.

**Modeling object.** `f_i = Σ_t Σ_k |u_i^{t}[k]|^p`, `p ∈ (1,2)∪(2,∞)`. For
`p ∈ (1,2)` this is a "soft-sparse" effort (between fuel and energy); for `p > 2`
it spreads effort and approaches peak-limiting as `p → ∞`. Both are quantities
one actually wants, not surrogates.

> **The honest non-2 instance.** If you want a `p ≠ 2` motivated by physics
> rather than by interpolation: power to overcome aerodynamic/hydrodynamic drag
> scales as `|v|³` (force `∝ v²`, power `= force·v`), so minimum-energy cruise for
> a drag-dominated vehicle minimizes `Σ_t |v_t|³` — a genuine **`α = 1/3`** cost.
> Drop it in as the `p = 3` case of this recipe with the velocity components in
> the `x₃` slots.

**Reformulation.** Per §1 (power epigraph): `t_k ≥ |u_k|^p ⟺ (t_k, 1, u_k) ∈
P_{1/p}`, objective on `+x₁`.

**Construction (per agent `i`, timestep `t`, channel `k`).**
- Column-blocks: states `x_i^t` (`:NOC`), and one `:POW` block
  `ξ_i^{t,k} = (x₁, x₂, x₃)` per channel, `cones[...] = PowerCone(1/p)`.
- Leaf row: `x₂ = 1` (`g = 1`). No row for `x₃` — the dynamics read it directly.
- Dynamics rows: as in the LP/SOC recipe, but the control enters from the `:POW`
  block's third slot. The block on `ξ_i^{t,k}` is `B_dyn[:,k] · [0 0 1]`
  (an `nx×3` matrix with `B_dyn[:,k]` in column 3). **No `1/√2`** — see §0.
- `c`: `+1` on each `x₁` slot (`c[colrange(B, ξ)[1]] = 1`), so
  `c'p = Σ t_k = Σ|u|^p`. Zero elsewhere.
- `Q`: zero.
- `g`: `x0_i` on `init`, `0` on `dyn`, `1` on each `x₂` leaf row.

**Why first.** Three independent cross-checks (M§12): leg H is a JuMP model with
`|u|^p` written directly (nonlinear, or `MOI.NormCone`/`NormPowerCone` where
available — independent of the lift); leg R is the explicit `:POW`-leaf JuMP model
(`MOI.PowerCone(1/p)`, slot-aligned per §0); leg S is your `IPMProblem`. And it
**hits the `α = 1/2` regression for free**: set `p = 2`, then
`(t,1,u) ∈ P_{1/2} ⟺ t ≥ u²`, so `Σ t = ‖u‖₂²` — run the *same* instance through
`PowerCone(0.5)` and through your rotated-SOC / `qp.jl` energy path and assert the
objective *and* iteration counts match closely (impl-guide §10.1). If they
disagree, the bug is in the oracle, not the assembly. Sweeping `p` then sweeps
`α`, the family axis exp never had (§9).

---

## 4. Recipe B — minimum-`ℓ_p`-norm control (the coupling showcase)

Recipe A summed per-channel powers; this one takes the **norm**, which couples a
whole control vector through a shared epigraph. The difference is real: `‖u‖_p`
vs `Σ|u_k|^p = ‖u‖_p^p`. The norm version is the honest "bound the size of the
control vector" objective and it stresses more of the assembly — one shared
scalar adjacent to `m` power leaves per timestep.

**Modeling object.** `f_i = Σ_t ‖u_i^t‖_p`, `p ∈ (1,2)∪(2,∞)`.

**Reformulation.** Per §1 (p-norm epigraph), per timestep `t`: a scalar bound
`τ_i^t`, weights `r_i^t ∈ ℝ_+^m`, the `m` cones `(r_k, τ, u_k) ∈ P_{1/p}`, and the
summation `Σ_k r_k = τ`. Objective `Σ_t τ_i^t`.

**Construction (per agent `i`, timestep `t`).**
- Column-blocks: states `x_i^t` (`:NOC`); a scalar `τ_i^t` (`:POS`); `m` `:POW`
  blocks `ξ_i^{t,k} = (x₁, x₂, x₃)`, `PowerCone(1/p)` each.
- Leaf rows per channel `k`: `x₁ = r_k` is *implicit* (slot 1 is the weight, kept
  `≥0` by the cone — no separate `:POS` block needed); `x₂ = τ_i^t` (couple slot 2
  to the shared scalar); `x₃ = u_k` read by dynamics as in §3.
- Summation row (private): `Σ_k x₁(ξ_i^{t,k}) − τ_i^t = 0`.
- `c`: `+1` on `τ_i^t`, zero elsewhere.
- `Q`: zero. `g`: `x0_i` on init, `0` on dyn and the summation row.

The slot-2 coupling (`x₂(ξ_k) = τ` for all `k`) is the apex-free way to share the
bound: `τ` lives in its own `:POS` block and each leaf pins to it, so no single
column touches all `m` cones except through private rows — the same "local copy
instead of a global scalar" discipline the dissipativity doc uses for `γ`.

**`α = 1/2` regression.** As in §3, `p = 2` gives `‖u‖₂ ≤ τ`, the second-order
cone epigraph — so this recipe's `p = 2` case maps to `test/small/soc.jl`
*exactly* (the `(τ; u)` SOC block), a different existing test than Recipe A's. Two
recipes, two independent `α = 1/2` anchors.

---

## 5. Recipe C — proportional-fair resource split (the showcase)

The power analogue of the exp KL-consensus showcase (E§5), and the one where the
cone and the *sheaf* reinforce each other instead of the cone being bolted onto an
otherwise-linear consensus. It is also the only recipe that exercises **multiple
distinct `α` in one problem**, which is the structural fact that separates power
from exp.

**Setup.** Each agent `i` holds a terminal allocation `a_i ∈ ℝ_+^m` (effort split
across `m` actuators/tasks/resources). Coordination is plain linear consensus
`δ a = 0` (`b = 0`, always realizable — M§1 satisfied for free): agents must agree
on a shared allocation at the terminal. The objective is **proportional fairness /
Nash bargaining**: maximize the geometric mean `(∏_k a_{i,k})^{1/m}` — the honest
"fair" aggregate, and *literally* a product of powers, not a surrogate for it.
(The `log`-sum form of the same argmax is the exp cone's territory; the geometric
mean is power's.)

**Construction (per agent `i`).**
- The allocation `a_i` is an `m`-dim `:POS` block (nonnegativity built in).
- A geometric-mean **tower** lifting `g_i ≤ (∏_k a_{i,k})^{1/m}`: a binary
  reduction of 2-arg power cones with **mixed `α`**. For `m = 3`:
  `(a₂, a₃, w) ∈ P_{1/2}` then `(a₁, w, g_i) ∈ P_{1/3}` gives
  `g_i ≤ a₁^{1/3}(a₂a₃)^{1/3} = (a₁a₂a₃)^{1/3}` (verified). Each tower node is a
  `:POW` block with its own `α`; the intermediate `w` is a private `:NOC`/`:POS`
  leaf pinned by the cone slots.
- `c`: `−1` on each `g_i` epigraph slot (maximize `Σ g_i`).
- Coordination rows: `δ_F` on the terminal `a_i`, exactly `sheaf(I,J,V)` as in
  M§7 — the allocation is the stalk, the equality is the coboundary.

This exercises power cones across the M§9 `N`-on-`K_N` and `T` knobs **and** the
`α` family (the tower's `1/2, 1/3, …`), while keeping the coordination layer a
textbook linear coboundary. For a fairness objective across agents (min instead of
sum), maximize a single `g` with `g ≤ g_i` rows — still linear coupling, the
geometric mean stays private.

---

## 6. What does *not* fit cleanly (so you don't lose a day)

- **`p = 1` and `p = ∞`.** These are the `α → 1` and `α → 0` boundaries — *not*
  in the cone (`0 < α < 1`), and already handled by `:POS`/box. The power cone is
  for `p ∈ (1,∞)\{2}`; `p = 2` is better served by native `:SOC` in production and
  kept only as the `α = 1/2` regression check (§3/§4). Don't ship `α = 1/2` power
  blocks where SOC is cheaper.
- **Geometric-programming dynamics** (`x_{t+1} = a·x_t^α·u_t^β`). Power-cone
  representable after a log change of variables, but — exactly the E§6 trap — the
  change of variables makes the **dynamics rows nonlinear in the original
  states**, so `B` is no longer the affine coboundary the KKT path (M§0) assumes.
  This wants the variables to *be* the logs from the outset. Different problem,
  not a drop-in.
- **Geometric-mean / power *consensus*.** Putting the power law on the
  **coordination layer** (agents agreeing on a geometric mean of their states)
  makes the restriction map nonlinear and kills the `δ`-structure — the same
  failure mode as agreeing on *storage* instead of *compliance* in
  `dissipativity-sdp.md` §2. Keep the power content in node objectives and private
  rows; let consensus stay linear. Recipe C is the honest factoring (geo-mean
  private, budget consensus linear).

---

## 7. The three-backend oracle for power (delta from M§12)

The H/R/S structure is unchanged. Power-specific notes:

- **Solvers with native power:** Mosek, Clarabel, SCS, Hypatia. Note **ECOS does
  *not*** do the power cone (it does SOC + exp) — so unlike the exp oracle, ECOS
  drops out of leg R. Mosek (commercial) and Clarabel/SCS (reference) remain.
- **Make leg H independent of the lift.** Write the `ℓ_p` / `|·|^p` / geo-mean
  term *directly* as a JuMP nonlinear objective (or `MOI.NormCone(p, …)` where
  your MOI version exposes it), not as a stack of power cones. Then **H-vs-R** is
  a pure check of "did I lift the power term into the cone correctly," independent
  of your solver — run it first, as M§12 insists.
- **The seam is aligned (E§0 reversal does *not* recur).** Leg R's
  `MOI.PowerCone(α)` triples are already in `(x₁,x₂,x₃)` order with exponent `α`.
  Still round-trip a known interior point and assert membership both ways before
  comparing objectives — the residual suspects are `α` vs `1-α` and the constant
  slot (`x₂ = 1` in the power epigraph; a wrong constant produces a
  plausible-but-wrong optimum).
- **Solution-level comparison is meaningful here.** `ℓ_p` with `p > 1` and the
  geometric mean are strictly convex/concave in their arguments, so the optimum is
  unique (unlike the `ℓ₁`/LP ties in M§12) — solution-level `atol` comparison
  works without an ε-regularizer. The `p = 2` separable case additionally
  cross-checks against `qp.jl`'s quadratic optimum.

---

## 8. Cone-level unit tests (the part that catches `pow.jl` bugs)

`test/small/pow_cone.jl` already exists and covers the battery; this section is
the map from those tests to the recipes, plus the power-specific traps. As in
E§8, objective-value agreement tests the *assembly* and is weak at catching bugs
*inside* `pow.jl` — test the primitives directly (they need only
`LinearAlgebra`).

The battery, against the impl-guide's identities:
- **Finite-difference the barrier.** `powbarrgrad!` vs FD of `F`; `powhess!`
  (and the `powbarr!` factor `L Lᵀ`) vs FD of the gradient; `powbarrhess!`
  `F‴[u]` vs FD of the Hessian along `u`. (Tests 4–7, 13.)
- **Log-homogeneity.** `⟨F′,x⟩ = -3`, `F″x = -F′`, `F‴[x] = -2F″`. (Tests 1–3.)
- **Shadow primal is closed-form** — the one place power is strictly *easier* than
  exp, which needs a Newton kernel (E's `exp_shadow_primal!`). `powdualgrad!`
  solves one monotone scalar equation for `ρ`; assert `F′(x̃) + s = 0` (Test 8)
  and the `s = -F′(x) ⇒ x̃ = x` identity (Test 9). **Exercise both shortcut
  branches:** `s₃ = 0 ⇒ ρ* = 1` (the symmetric slice) and `α = 1/2 ⇒` the
  quadratic closed form.
- **Secants.** After `powscale!` fills `M`, assert `M x = s` and
  `M δx = δs` (`δx = x − μx̃`, `δs = s − μs̃`) at an **off-central** `(x,s)`, plus
  `M` symmetric PD. (Test 10 does the first secant.)
- **Self-concordance** `|F‴[u,u,u]| ≤ 2(F″[u,u])^{3/2}` (tops out at exactly 1.0
  for a genuine 3-LHSCB). (Test 11.)
- **Test across multiple `α`** (the family) — the existing tests already randomize
  `α ∈ (0.1, 0.9)` — and keep `α = 1/2` as a permanent regression target
  (Test 12), where the cone is the rotated SOC and the shadow primal takes the
  quadratic branch.

**Force the off-central branch (same trap as E§8).** A near-central or parallel
start makes `z = x × x̃ ≈ 0`, `rel_z < eps`, and `powscale!` takes the `μ F″`
*fallback* — never the closed-form `ssᵀ/⟨x,s⟩ + δsδsᵀ/⟨δx,δs⟩ + t zzᵀ` Tunçel
path. Construct `(p,d)` deliberately non-parallel and assert `rel_z > sqrt(eps)`
so the branch most likely to be wrong is actually covered, and probe just above
and below the crossover for graceful degradation.

---

## 9. Scaling knobs & conditioning (delta from M§9)

| knob | applies to power? | note |
|---|---|---|
| `N` on `K_N` | **yes** | terminal clique → dense Schur fill, same as M§9 |
| `N` on sparse `G` | **yes** | chordal-friendly, same as M§9 |
| `T` | **yes** | number of leaves grows linearly; no fill penalty |
| `m` (control dim) | **as a count** | not a block-size axis (`:POW` is fixed 3-D); but Recipe B/C turn `m` into a *count* of leaves per timestep |
| **`α` (= `1/p`)** | **yes — power-only** | the family axis exp lacks; doubles as a conditioning sweep |

So power has the exp pair `(N, T)` **plus** an `α` axis. Two conditioning facts
shape test expectations:

- **`α → 0` / `α → 1` (i.e. `p → ∞` / `p → 1`) degrades.** The cone approaches a
  polyhedral corner; barrier conditioning worsens and the Tunçel scaling needs
  more IPM iterations. **`α = 1/2` is best-conditioned** — there the cone is the
  self-scaled rotated SOC, the BFGS scalar `t` collapses toward the NT value, and
  convergence should be crisp. So `α = 1/2` is simultaneously the correctness
  regression *and* the conditioning baseline; benchmark `α` away from it on both
  sides and expect a U-shaped iteration count.
- **Tunçel scaling is quasi-Newton** (as for exp): expect more iterations and less
  crisp terminal convergence than the symmetric cones, *except* at `α = 1/2`.
  Loosen `gap_tol`/`feas_tol` and raise `itmax` relative to LP/SOC. The analytic
  `(3,1,2)` factor (`cond(L) = √cond(F″)`) is what keeps it tractable.
  `powmaxstep` is bisection (53 iters, no closed form), so per-step cost is higher
  — budget for it in timing comparisons.

---

## 10. Pitfalls (delta from M§10)

- **`α = 1/2` must reproduce SOC.** The single most important regression. If the
  `p = 2` instance disagrees with your rotated-SOC/`soc.jl`/`qp.jl` answer
  (objective *or* iteration count), the bug is in the oracle, not the solver
  (impl-guide §10.1).
- **`α` vs `1-α` at the seam.** The seam is *aligned* (no exp reversal), so the
  one thing left to get wrong is which arm carries `α` and the constant slot
  (`x₂ = 1`). Round-trip an interior point (§7).
- **The boundary is more forgiving than exp's log.** A control that *wants* to be
  zero is fine: `x₃ = 0` is the symmetric slice, well-conditioned and
  interior-feasible — unlike the exp log-barrier, which forbids its argument from
  reaching the boundary at all (E§10). The caution instead is `α → 1` (`p → 1`):
  the `r_k`/weights want the boundary and conditioning suffers — keep `p` bounded
  away from 1.
- **No vector power.** `K` scalar terms = `K` blocks = `K` sets of leaf rows. If
  you want a dim-`m` `:POW` block, you want `m` blocks (Recipe B) — or a norm
  epigraph, which is `m` blocks plus a shared scalar.
- **Don't reuse one `PowerCone(α)` object across different `α`.** The object keys
  on `α`; a builder that caches a single cone object and reuses it for a
  geometric-mean tower (which needs several `α`) will silently apply the wrong
  exponent. Construct one `PowerCone(α)` per distinct `α`.
- **Realizability (M§1/M§10) is unaffected.** Power lives only in objectives and
  private rows, so the coordination coboundary's `b ∈ im δ` requirement is
  untouched — consensus (`b = 0`) remains the safe default.

---

## 11. Suggested file layout

Mirror the existing `test/small/*` pair (and the already-present
`pow_cone.jl`):

- `test/small/pow.jl` — Recipe A (§3). Three-backend oracle vs Mosek/Clarabel,
  with the **`α = 1/2 ≡` rotated-SOC** regression `@testset` (same instance
  through `PowerCone(0.5)` and the SOC/QP path) and a reference to the
  `pow_cone.jl` cone-level battery. This is the **regression test** and the thing
  to write first; it validates the recipe and pins `pow.jl` behavior on an
  end-to-end problem.
- `test/small/fairsplit.jl` — Recipe C (§5). The **showcase**: power cone and
  sheaf coordination reinforcing each other, with a geometric-mean tower carrying
  several `α`, scaled across the `N`/`T`/`α` axes of §9.

`test/small/pow_cone.jl` already provides §8's isolated FD + secant + factor +
self-concordance checks — they need nothing but `LinearAlgebra`, run in
milliseconds, and localize a cone bug to the exact primitive, whereas an oracle
mismatch only tells you "something between the model and the optimum is wrong."
Keep running them first.
