# Implementation Plan: Multi-Cone Support for the Sheaf IPM

## 1. What we are building

The solver currently handles a single cone type — the PSD cone — on every sheaf
vertex stalk. We want to support three symmetric cones, chosen per stalk:

- **POS** — the nonnegative orthant ℝⁿ₊ (rank *n*)
- **SOC** — the second-order / Lorentz cone (rank 2, regardless of dimension)
- **SDP** — the PSD cone 𝕊ᵈ₊ (rank *d*) — the existing path

Each vertex `v ∈ vtxs(B)` carries one cone. The coboundary `B`, the constraint
`Bp = g`, the objective `c'p`, and the entire KKT / chordal-factorization
machinery stay structurally the same. Only the *per-block* cone arithmetic changes.

## 2. Load-bearing design decisions

These are the decisions that everything else depends on. They are recorded here so
the rationale survives the implementation.

**D1 — Work in original (unscaled) coordinates, not v-space.** Each cone operates
in its own native coordinates. There is no shared "scaled" frame. The SDP path is
already a factored, fused, matrix-free implementation in original coordinates and
is both faster and more numerically stable than the unfused v-space rewrite. We
keep it verbatim. The v-space formulation buys *code uniformity*, which is only
worth it when cones are symmetric in importance and none is pre-tuned — not our
situation.

**D2 — Every cone emits a dense per-vertex Hessian block.** Each cone produces a
dense `embdim × embdim` block (its `W⁻¹ ⊗ₛ W⁻¹` analogue) that folds into the
augmented `(1,1)` block `F = H + α B'B` exactly as the SDP `skron` block does
today. **The chordal Cholesky, `factor_kkt!`, `solve_kkt_factored!`, and the Schur
CG never learn that cones exist.** This preserves the crown jewel: the sheaf
Laplacian sparsity that drives the factorization. The vertex block is already dense
(SDP `skron`, and `B'B`'s own `Σ_e F_e'F_e` block), so folding in POS/SOC curvature
adds no fill.

**D3 — No Woodbury / low-rank peeling of SOC curvature.** Each SOC contributes a
local rank-≤2 term, but peeling them via Woodbury couples all SOCs globally through
`F_base⁻¹` and only pays off when removing the curvature actually sparsifies the
factor — which it does **not** here, because `F_base` still contains dense SDP and
`Σ_e F_e'F_e` blocks. Fold, don't peel. Keep a global-Schur Woodbury path as a
documented escape hatch, to be revisited *only* if profiling against ECOS demands
it. It would be one optional code path inside the Schur solve, never a `Cone`
method.

**D4 — The cone is a strategy object; the scaling cache is a per-cone type.** A
`Cone` exposes a fixed set of methods (Section 3). Its scaling cache is itself an
abstract type so storage can diverge: a bare vector for POS, `(β, w)` for SOC, and
the existing `(L_P, L_D, U, sv)` bundle for SDP, all behind one dispatch.

## 3. The Cone interface

A cone is a strategy object dispatched in the per-block loops. The contract:

| method | meaning | POS | SOC | SDP |
|---|---|---|---|---|
| `degree(c)` | rank → feeds `conedegree` | n | 2 | d |
| `embdim(c)` | stored block length (= `ncols(B,v)`) | n | n | trinum(d) |
| `identity!(v,c,ξ)` | ξ·e for `initialize!` | ξ·1 | ξ·(1,0,…,0) | svec(ξI) |
| `update_scaling!(cache,c,p,d)` | compute + cache NT scaling | w=√(p./d) | (β,w) | L_P,L_D,U,sv |
| `hessian_block!(H,cache,c)` | emit W⁻¹⊗ₛW⁻¹ analogue (dense) | Diag(d./p) | W⁻² | skron(W⁻¹) |
| `corrector_term!(rc,cache,c,Δp,Δd,σμ)` | 2nd-order RHS (`σμ·z⁻¹ − s − cross`) | elementwise | H½/arrow (§4) | fused (existing) |
| `max_step(cache,c,Δ)` | step to boundary | min(−p./Δp) | det-quadratic | eigmin |

The KKT solver depends on none of these. The cone-specific call sites are exactly:
`conedegree`, `initialize!`, `allocate_hess` (block sizing), `hess!`,
`corrector_rhs!`, and `step_to_boundary`.

## 4. Per-cone math reference

Captured here so the implementation has a single source of truth. The SDP column is
the existing code; POS and SOC are new.

### POS (ℝⁿ₊), rank n

POS is diagonal in every coordinate system, so original-coords and v-space coincide.

- scaling: `w = sqrt(p ./ d)`; spectral values `λ = sqrt(p .* d)`
- Hessian block: `Diagonal(d ./ p)`
- corrector contribution: `σμ ./ d .- p .- (Δp .* Δd) ./ d`
- step: `τ = min(1, γ · min_i(−p_i/Δp_i))` over indices with `Δp_i < 0`
- identity: `ones(n)`

### SOC (Lorentz), rank 2

Notation: `x = (x₀, x̄)`, `J = diag(1, −1, …, −1)`, `jdot(a,b) = a₀b₀ − ā·b̄`,
determinant `det x = jdot(x,x) = x₀² − ‖x̄‖²` (> 0 in the interior),
Jordan product `s ∘ z = (jdot(s,z), s₀z̄ + z₀s̄)`, identity `e = (1, 0)`,
cone inverse `x⁻¹ = Jx / det x`.

**Arrow inverse** (solve `L(z) u = b` where `L(z)` is z's arrow matrix) — derived
in closed form by block elimination:

```
δ  = jdot(z, z)                       # = det z > 0
u₀ = (z₀·b₀ − z̄·b̄) / δ
ū  = (b̄ − u₀·z̄) / z₀
```

Self-check: `L(z) e = z`, so `b = z` must return `u = e = (1, 0)`. ✓ (pinned)

**NT scaling point and Hessian.** *(All formulas below are numerically verified to
machine precision across n ∈ {2,3,4,7,12}, 200 random interior pairs each — see
"Trust ledger" and the `/tmp/soc4.py`, `/tmp/corr.py` experiments.)*

> ⚠ **Correction to an earlier draft.** The vector `w = (s̄ + Jz̄)/‖s̄+Jz̄‖_J` is the
> normalized **NT scaling point** `w̄_pt`, **not** the axis of a boost `W`. Do **not**
> build `W = β(2wwᵀ − J)` and expect `Wz = W⁻¹s`; that `W` is a valid boost but the
> *wrong* one — it equalizes directions (`Wz ∝ s`, `W⁻¹s ∝ z`), which presents as
> "determinants correct, vectors off by a rotation." The object you want is the
> barrier Hessian **at** the scaling point.

Definitions: `s̄ = s/√det s`, `z̄ = z/√det z`,
`w = (s̄ + Jz̄)/√(2(1 + s̄·z̄))` (Euclidean dot; satisfies `wᵀJw = 1`),
`β = (det s/det z)^{1/4}`, `a = Jw`, `η = 1/β² = √(det z/det s)`.

- **Hessian block** (the `W⁻¹⊗ₛW⁻¹` analogue, satisfies `H s = z`, PD in-cone):
  **`H = η·(2 a aᵀ − J)`**, a literal rank-one-plus-diagonal. Apply in O(n):
  `H x = η(2a(aᵀx) − Jx)`; materialize trivially for the dense block. (Verified
  `Hs = z` to 1e-15 and `H = W⁻²` for the genuine symmetric boost to ~1e-8.)
- **What you do NOT need:** in original coordinates the boost `W` and the scaled
  point `λ` are **not** required for the Hessian, the affine RHS, the step length,
  or μ. The affine RHS is `r_c = −s` ⇒ `f = H(−s) − r_d = −z − r_d` (verified).
- consistency check (if you do form `λ`): `det λ = √(det s · det z)` (geometric mean).

**Step to boundary.** `det(z + τΔz)` is quadratic in τ: `aτ² + bτ + c` with
`a = jdot(Δz,Δz)`, `b = 2·jdot(z,Δz)`, `c = det z > 0`. The max step is the smaller
positive root (capped at 1, scaled by γ) with side condition `z₀ + τΔz₀ ≥ 0`. No
eigensolve, no `W`.

**Corrector.** The cone-generic corrector RHS, derived from
`(λ+d_s)∘(λ+d_z) = σμ e` (using `L(λ)⁻¹e = λ⁻¹` and `L(λ)⁻¹(λ∘λ) = λ`), is
`r̃_c = σμ λ⁻¹ − λ − L(λ)⁻¹(d_sᵃ ∘ d_zᵃ)`, mapped to original coords by `primal_out`.
The SDP `corrector_rhs!` is **exactly** this instance (verified equal to 9e-15). The
SOC instance, using the two verified simplifications `H^{-1/2}λ = s` and
`H^{-1/2}λ⁻¹ = z⁻¹`, collapses to the same `σμ(dual)⁻¹ − primal − cross` shape as the
SDP code:

```
H½, H⁻½ = boost half-rapidity of H   # closed form below
λ   = H½ s                            # scaled point (= H⁻½ z)
d_s = H½ Δsᵃ ;  d_z = H⁻½ Δzᵃ         # affine dirs into v-space
t   = d_s ∘ d_z                       # SOC Jordan product
q   = arrow_inv(λ, t)                 # L(λ)⁻¹, the O(n) arrowhead
r_c = σμ·z⁻¹ − s − H⁻½ q              # z⁻¹ = Jz/det z
```

`H½` is the only place the corrector needs the boost square root. Closed form
(half-rapidity of the boost-form `H = η(2aaᵀ−J)`): with `a = (a₀, ā)`,
`a' = (√((a₀+1)/2), ā/√(2(a₀+1)))` (J-unit), then
`H½ = √η (2a'a'ᵀ − J)`, `H⁻½ = (1/√η)(2(Ja')(Ja')ᵀ − J)`. (Verified
`(H½)² = H`, symmetric, `H⁻½H½ = I` to 1e-16.)

### SDP (𝕊ᵈ₊), rank d — unchanged

Existing `meanblock!` / `hessblock!` / `skron!` / `corrector_rhs!` /
`step_length_block`. Becomes the SDP implementation of the interface verbatim, with
**no override needed** — under D1 every cone is native in its own coordinates, so
the fused SDP corrector simply *is* SDP's `corrector_term!`.

## 5. Implementation phases

Each phase ends with a verification gate that must pass before the next begins. The
ordering is chosen so that plumbing is de-risked before math, and the easy cone
shakes out the interface before the hard one.

### Phase 0 — Interface scaffolding, zero behavior change

Introduce `abstract type Cone`, the three concrete types, and the abstract scaling
cache. Implement only the SDP cone, as a thin wrapper over existing code. Thread a
`cones::Vector{Cone}` aligned with `vtxs(B)` through `conedegree`, `allocate_hess`,
`initialize!`, `hess!`, `corrector_rhs!`, and `step_to_boundary`, replacing
hardcoded `triroot(ncols(B,v))` with `degree(cone)` and block sizing with
`embdim(cone)`. No new cone math.

> **Gate 0.** Run `compare_mosek.jl`. The pure-SDP iteration log — `μ`, `‖r_p‖`,
> `‖r_d‖`, `τ_p`, `τ_d` at every iteration — must reproduce the pre-refactor log to
> roundoff, and the objective must still match Mosek. This proves the plumbing is
> inert. Also assert the standing identity `dot(view(p,r),view(d,r)) ≈ sum(sv_v.^2)`
> per block as a cheap correctness tripwire.

*Open question to resolve here:* how cones are specified at `sheaf(...)`
construction and how a stalk's cone determines its `B` column-block dimension
(`embdim`). For SDP, `ncols(B,v) = trinum(d)`; for POS/SOC it is the stalk dimension
`n`. This affects how restriction maps `F_e` are sized and assembled. Settle the
construction-side threading before Phase 1.

### Phase 1 — POS cone

Implement the POS strategy: `degree = n`, `embdim = n`, diagonal Hessian, elementwise
corrector, `min(−p./Δp)` step, `ones` identity. This is the trivial cone whose only
job is to shake out the interface on something with a different rank and a non-svec
storage layout.

> **Gate 1.** Construct a POS-only problem (an LP in the sheaf form) and solve it
> against Mosek; demand relative objective agreement < 1e-6. Cross-check one small
> LP whose optimum is known by hand. Confirm `conedegree` now returns `Σ n_v` and
> that convergence behaves like an LP (no SDP-specific assumptions leaked in).

### Phase 2 — SOC kernels in isolation

Before any solver wiring, implement and unit-test the SOC math standalone:
`jdot`, `det`, `∘`, cone inverse `z⁻¹`, `arrow_inv!`, the scaling point `w` + `β`,
the Hessian `H = η(2aaᵀ−J)`, the boost half-rapidity `H½`/`H⁻½`, and `soc_max_step`.

> **Gate 2 (standalone unit tests, mirroring `/tmp/soc4.py` + `/tmp/corr.py`).**
> 1. **Hessian** — on random interior `(s, z)`: assert `wᵀJw ≈ 1`, `H s ≈ z`,
>    `H = Hᵀ`, `H ≻ 0`. (This is the test that *would have caught the original
>    boost-axis bug* — the old `‖Wz − W⁻¹s‖` check was testing the wrong object.)
> 2. **Boost root** — assert `(H½)² ≈ H`, `H½ = (H½)ᵀ`, `H⁻½ H½ ≈ I`, and
>    `H½ s ≈ H⁻½ z` (`= λ`).
> 3. **Arrow inverse** — assert `arrow_inv!(·, z, z) ≈ e` (since `L(z)e = z`).
> 4. **Corrector** — assert the SOC corrector equals its v-space form
>    `H⁻½(σμλ⁻¹ − λ − L(λ)⁻¹(d_s∘d_z))`, i.e. the two simplifications hold.
> 5. **Step length** — for a `Δz` pointing out of the cone, assert the returned τ
>    lands on the boundary: `det(z + τΔz) ≈ 0` to roundoff (with γ = 1).
>
> No solver involvement yet — pure algebra checks, the SOC analogue of "diff the
> iteration log."

### Phase 3 — SOC cone integration

Wrap the Phase-2 kernels in the SOC strategy and wire into the six call sites.

> **Gate 3.** Solve a SOC-only problem against Mosek; relative objective < 1e-6.
> Watch the step lengths and `μ` decrease — a SOC scaling-constant error typically
> shows up as τ collapse or stalling rather than a wrong answer, so check the full
> trajectory, not just the final objective.

### Phase 4 — Mixed cones

Heterogeneous stalks: a problem mixing POS, SOC, and SDP vertices.

> **Gate 4.** Solve against Mosek; relative objective < 1e-6. Verify `conedegree`
> sums mixed ranks correctly and that `allocate_hess` produces correctly-sized
> blocks per cone. This is the real acceptance test for the whole feature.

### Phase 5 — (deferred) performance and escape hatches

Only after correctness is established across Gates 0–4:
- Profile against ECOS / Mosek on POS/SOC-heavy problems.
- If a large SOC on sparse stalks dominates, *then* evaluate the global-Schur
  Woodbury path from D3 — as one optional branch in the Schur solve, gated behind a
  test comparing it to the dense-fold result.
- Consider the closed-form `W⁻²` expansion if the columnwise materialization shows
  up hot.

## 6. Summary of what changes vs. what doesn't

**Untouched:** `factor_kkt!`, `solve_kkt_factored!`, `solve_kkt!`, the chordal
Cholesky, the Schur CG, `residuals!`, `mu`, all `B`-products, and the SDP per-block
math.

**Generalized (cone-dispatched):** `conedegree`, `initialize!`, `allocate_hess`,
`hess!`, `corrector_rhs!`, `step_to_boundary`.

**New:** `abstract type Cone` + `POSCone`/`SOCCone`/`SDPCone`, the abstract scaling
cache, and the POS/SOC kernels in Section 4.

## 7. Trust ledger

What is verified vs. what still needs a check. Most SOC formulas are now numerically
verified to machine precision (n ∈ {2,3,4,7,12}, 200 trials each); the experiments
live in `/tmp/soc4.py` (Hessian) and `/tmp/corr.py` (corrector).

- **Verified to machine precision:** `β = (det s/det z)^{1/4}`;
  the scaling-point identity (`w` is `w̄_pt`, **not** a boost axis);
  the Hessian `H = η(2aaᵀ−J)` satisfies `Hs=z`, is PD in-cone, and equals the
  genuine symmetric boost's `W⁻²`; the affine shortcut `f = −z − r_d`;
  the boost half-rapidity `H½` (`(H½)²=H`, symmetric, `H⁻½H½=I`);
  `λ = H½s = H⁻½z`, `det λ = √(det s·det z)`;
  the simplifications `H⁻½λ = s` and `H⁻½λ⁻¹ = z⁻¹`;
  the cone-generic corrector reproduces the SDP `corrector_rhs!` exactly (9e-15);
  arrow inverse (self-checks via `L(z)e = z`); all POS formulas.
- **Still to check during implementation (Gates 2–4):** that the Julia transcription
  matches these reference kernels (port-level bugs), and that the *assembled* SOC
  solve converges against Mosek — i.e. that nothing in μ/σ bookkeeping, the
  inner-product/rank convention (`⟨s,z⟩=sᵀz`, rank 2), or `initialize!` leaks an
  SDP-specific assumption. The math is pinned; the wiring is not yet.
