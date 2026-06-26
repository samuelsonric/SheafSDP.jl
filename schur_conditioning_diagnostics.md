# Diagnosing the augmented Schur complement in SheafSDP.jl

## Purpose

The Uzawa solver augments the (1,1) block to condition the Schur complement and
then runs **unpreconditioned** CG on it (`solve_uzw!`, the `it!(itrwrk, S, r, …)`
call with no `M`). `raug = 1e6` is the price of having no preconditioner, and that
price is paid in the Cholesky of `F = A + α·BᵀB` (the `rgmin…rgmax` diagonal
rescue in `init_uzw!`, plus the `refine_kkt!` mop-up).

Before building any preconditioner we need to know **what actually creates the
small eigenvalues** of the augmented Schur complement, because the two candidate
causes want opposite cures:

- **(A) Structural.** The small modes come from the sheaf Laplacian `L = BᵀB`
  having small *nonzero* eigenvalues (small spectral gap / near-harmonic modes).
  These depend only on `B`, which is **fixed across IPM iterations**, so the bad
  subspace does not move. → *Cure: deflate the bottom-k eigenvectors of `L` once.*

- **(B) Barrier-driven.** The small modes come from `A` (the barrier Hessian)
  degenerating at the cone boundary as the IPM approaches optimality. These are
  `A`-dependent, they **drift** every Newton step, and they localize on the active
  set. → *Cure: recycle Ritz vectors across the Newton sequence (or an
  `A`-dependent coarse space).*

It can be a mix. The tests below quantify the split and pick the cheap path.

---

## What actually causes the small modes (the corrected picture)

Let `B` be the coboundary (`m × n`, edge × vertex, `m > n` for you), `A` the
(1,1) block, `L = BᵀB` the sheaf Laplacian, and

```
S(α) = B (A + αL)⁻¹ Bᵀ           # m×m, what CG actually solves
S₀   = B A⁻¹ Bᵀ                  # the un-augmented Schur complement
```

On `range(B)` the eigenvalues of `S(α)` are `μᵢ / (1 + α μᵢ)`, where `μᵢ` are the
eigenvalues of `S₀`; on `ker(Bᵀ)` (the cycle space, dimension `m − rank B`) they
are exactly `0`.

Two consequences that matter for the diagnosis:

1. **`ker(Bᵀ)` is an exact kernel**, not ill-conditioning. CG on the consistent
   semidefinite system stays in `range(B)`; this is *not* what `raug` is fighting.

2. **Exact `H⁰ = ker(L) = ker(B)` is invisible to `S(α)`** (annihilated by `B`).
   `H⁰` only shows up in the *primal* solve, where `F = A + αL` reduces to `A` on
   `H⁰` (the augmentation does nothing there). So `H⁰` is a Cholesky-accuracy
   concern, **not** a Schur-conditioning concern.

The Schur-complement bad modes are therefore the **small nonzero `μᵢ`**, and

```
κ(S(α) | range B) ≈ κ(S₀) · (1 + α μ_min)/(1 + α μ_max) ≈ 1 + 1/(α μ_min)   (α moderate)
```

So everything hinges on `μ_min` and on **whether `μ_min` is set by `L`'s spectral
gap (structural) or by `A`'s boundary blow-up (barrier).** That is the quantity
every test below targets.

---

## Minimal instrumentation

You already have the operators; the diagnostics mostly need logging hooks.

- In `solve_uzw!`, the augmented Schur complement is already a `LinearOperator`
  (`S = LinearOperator(T, m, m, true, true, schur!)`). Reuse `schur!` as the
  mat-vec for any eigensolver (Arpack/KrylovKit).
- Extend `History` (history.jl) to record, per Newton step: duality gap `μ_gap`,
  `‖A‖`, `α`, CG iteration count, `rgmin` rescue count, and (optionally) the
  smallest Ritz value from the CG/Lanczos run.
- For the *exact* spectral forensics, run on the **small** instances
  (`test/small/*`) where you can densify `L`, `S₀`, and `S(α)` and call a full
  `eigen`. Use the **large** instances (`test/large/*`) only for behavioral
  (iteration-count / drift) tests.

```julia
# Densify helpers for small instances (adapt to your accessors).
Ld  = Matrix(Symmetric(B' * B, :L))          # sheaf Laplacian
# S₀ and S(α): apply the existing schur!-style closure to basis vectors,
# or assemble B * inv(Matrix(A)) * B' / B * inv(Matrix(A + α*Ld)) * B' for small m.
```

---

## Tests

### T1 — Sheaf-Laplacian spectrum (the structural floor)

**Purpose.** Measure the structural lower bound on conditioning, completely
independent of the IPM.

**Procedure.** Compute the smallest eigenvalues of `L = BᵀB`:

```julia
λ = eigvals(Symmetric(Matrix(B'B)))          # small instances
# large: KrylovKit.eigsolve(x -> B'*(B*x), n, k, :SR)   # smallest, careful: 0s = H⁰
```

**Measure.**
- `dim H⁰` = number of eigenvalues at ~0 (machine-zero cluster).
- `λ₁⁺` = smallest **nonzero** eigenvalue (the sheaf spectral gap).
- The shape of the low end: a clean gap above `λ₁⁺`, or a fat cluster of tiny
  nonzero eigenvalues?

**Signature.**
- Fat low cluster / very small `λ₁⁺` (poorly connected sheaf, near-disconnected
  components, long-diameter graph) → **(A) likely**; structural deflation has a
  fixed, identifiable target.
- Large `λ₁⁺` with a clean gap, `dim H⁰` small/zero → structural floor is benign;
  any trouble must be **(B)**.

---

### T2 — `μ_min(S(α))` vs duality gap, across the IPM run

**Purpose.** The single most decisive test. Does the Schur conditioning collapse
*as the iterate approaches the boundary*?

**Procedure.** Instrument the IPM. At each Newton step `k`, after `A` and `α` are
set, compute the smallest few **nonzero** eigenvalues of `S(α)` (matrix-free via
`schur!`, deflating `ker Bᵀ`; or densely on small instances), and also of `S₀`
(set `α = 0`). Log alongside the duality gap.

**Measure.** `μ_min(S₀)` and `μ_min(S(α))` as functions of the duality gap
`μ_gap` (equivalently, of `‖A‖`).

**Signature.**
- `μ_min` **roughly constant** across the run (flat vs `μ_gap`) → **(A)
  structural.** The floor is `B`'s, not `A`'s.
- `μ_min` **decreases monotonically** as `μ_gap → 0`, e.g. like a power of the gap
  → **(B) barrier-driven.** Plot `log μ_min` vs `log μ_gap`; a clear negative
  slope is the smoking gun for (B).

---

### T3 — Identity and drift of the bad eigenvector

**Purpose.** Confirm T2 by checking *what* the smallest mode is and *whether it
moves*.

**Procedure.** At each Newton step, take the eigenvector `y₁` of `S(α)` for the
smallest nonzero eigenvalue. Compute two overlaps:

```julia
# overlap with the structural (sheaf-Laplacian) low space  -> hypothesis (A)
θ_struct = subspace_angle(Bᵀ * y₁, lowspan(L, k))     # lowspan = bottom-k eigvecs of L
# overlap with high-curvature (near-active) blocks of A    -> hypothesis (B)
θ_active = subspace_angle(Bᵀ * y₁, highcurv(A, k))    # blocks where A's eigvals are largest
# drift between consecutive Newton steps
θ_drift  = subspace_angle(span(y₁ᵏ), span(y₁ᵏ⁺¹))
```

**Signature.**
- High, **stable** `θ_struct` overlap and small `θ_drift` → **(A).** The bad
  subspace is fixed and equals `L`'s low modes ⇒ deflate-once works.
- High `θ_active` overlap, large `θ_drift`, mode concentrating on the settling
  active set → **(B).** A fixed deflation space cannot track it ⇒ recycling.

---

### T4 — The deflate-once experiment (direct test of A)

**Purpose.** Operational test of whether a *fixed* structural coarse space fixes
the problem.

**Procedure.** Compute `Z` = bottom-k eigenvectors of `L` **once** (mapped into
edge space via `B`). Run the IPM with the Schur CG deflated against `Z` (project
`Z` out, or use it as a coarse correction). Sweep `k ∈ {dim H⁰, +5, +10, +20}`.

**Measure.** CG iteration count per Newton step, across the whole run.

**Signature.**
- CG counts drop and **stay low for all iterations** → **(A) confirmed**; the
  cheap structural deflation is the answer.
- CG counts drop early but **climb again near optimality** → **(B) present**;
  barrier modes leak in beyond the fixed `Z`. Note the iteration where it breaks
  down — that is where the active set starts dominating.

---

### T5 — `raug` sweep at fixed iterates (the trade curve)

**Purpose.** Map the CG-speed vs Cholesky-accuracy trade, and see how the sweet
spot moves between mid-run and near-optimal.

**Procedure.** Freeze two iterates: one mid-run (well-centered `A`) and one
near-optimal (degenerate `A`). At each, sweep `raug ∈ {1, 10, …, 1e6}` and record:

```
CG iterations           # should fall as α grows
‖F Fᵀ − (A+αL)‖ / ‖A+αL‖  # Cholesky accuracy; should worsen as α grows
rgmin rescue triggered?  # the ceiling indicator
```

**Measure / Signature.**
- The crossover `raug` where CG gets cheap. If it is far larger for the
  near-optimal iterate than the mid-run one → the bad modes are **(B)
  barrier-driven** (you need ever-more α as you approach the boundary, which is
  exactly the failure mode).
- Quantifies how far you could *back off* `raug` once a preconditioner/deflation
  absorbs `μ_min`.

---

### T6 — Recycling pilot (direct test of B's cure)

**Purpose.** Test whether Krylov recycling fixes the barrier-driven case cheaply.

**Procedure.** Add minimal Ritz recycling: after each Newton-step CG solve, extract
the few smallest Ritz pairs (from the Lanczos tridiagonal, or use a recycling
solver — GCRO-DR / RMINRES / recycled CG). Carry them as a deflation space into
the next step. Drop `raug` toward the bare `‖A‖/‖B‖²` value (`raug ≈ 1`).

**Measure.** CG counts across the run at low `raug`, vs (i) bare CG at `raug=1e6`,
(ii) fixed-`L` deflation (T4). Also log `subspace_angle` between consecutive
recycled spaces.

**Signature.**
- Recycled-CG at `raug≈1` matches or beats bare CG at `raug=1e6`, with accurate
  Cholesky → **(B) cure confirmed**; this is the cheap robust path.
- Small consecutive subspace angles → slow drift, recycle every step is overkill;
  large angles near optimality → refresh the recycled space each iteration.

---

### T7 — Controlled synthetic sheaves (mechanism isolation)

**Purpose.** Separate the mechanisms by construction, so the behavioral tests have
a ground truth.

**Procedure.** Build two synthetic instances with the **same cones / same IPM**,
differing only in the sheaf graph + restriction maps:

1. **Small-gap sheaf.** Two well-connected clusters joined by a single weak edge
   (or a long path / near-disconnected restriction maps). Forces tiny `λ₁⁺(L)`
   ⇒ structural (A) by construction.
2. **Expander sheaf.** Well-connected graph, trivial `H⁰`, large `λ₁⁺(L)`. Any
   conditioning trouble here must be barrier-driven (B).

Run T2/T4 on both.

**Signature.** Confirms the readings: instance 1 should show flat `μ_min` fixed by
the structural deflation; instance 2 should show `μ_min` collapsing only as
`μ_gap → 0`. If your real instances (`dissipativity`, `l2gain`, `sdp`, …) behave
like 1, structural; like 2, barrier.

---

## Suggested instance coverage

From the repo, run the exact spectral tests (T1–T3) on **small** instances and the
behavioral tests (T4–T6) on **large**:

| Instance | Why include it |
|---|---|
| `lp`, `qp` | polyhedral / mild conic geometry — `A` degeneration is gentler; baseline |
| `soc`, `sdp` | strong cone-boundary behavior — most likely to expose (B) |
| `dissipativity`, `l2gain` | control LMIs with a genuine sheaf graph — most likely to expose (A); also the case where restriction maps are general (non-orthogonal) |
| `elasticnet`, `fairsplit` | additional graph topologies / mixed cones |
| T7 synthetic small-gap + expander | ground-truth controls |

---

## Decision table

| T1 low end | T2 `μ_min` vs gap | T3 drift | T4 deflate-once | Conclusion | Cheap cure |
|---|---|---|---|---|---|
| fat / tiny `λ₁⁺` | flat | small | stays low | **(A) structural** | deflate bottom-k of `L` once |
| clean gap | collapses with gap | large | breaks near optimality | **(B) barrier** | recycle Ritz vectors; back off `raug` |
| fat low end **and** collapse | both | mixed | helps but not enough | **(A)+(B)** | fixed `L`-deflation **+** recycling on top |

---

## Notes / gotchas

- **Nullspace handling.** `S(α)` is singular on `ker(Bᵀ)`. For smallest-eigenvalue
  computations, either restrict to `range(B)` (project with `B(BᵀB)⁻¹Bᵀ`) or target
  smallest *nonzero* (shift, or compute on `range`). Do **not** report the exact
  zeros as "small modes."
- **Cost.** Exact eigen on small instances is free; on large instances use a few
  Lanczos steps via `schur!`. T2 only needs `μ_min` (a handful of vectors), not the
  full spectrum.
- **`α` moves with the iterate.** Since `α = raug·‖A‖/‖B‖²` is recomputed each step
  and `‖A‖ → ∞` near optimality, the augmentation self-strengthens exactly where
  the Cholesky is weakest — keep this in mind when reading T5.
- **Orthogonal vs general maps** still matters downstream: if (A) dominates *and*
  your restriction maps are orthogonal, the structural deflation can be replaced by
  a connection-Laplacian approximate-Cholesky (Laplacians.jl). If maps are general
  (expected for `dissipativity`/`l2gain`), use the eigenvector deflation from T4.
- **Reproducibility.** Fix the RNG seed for any randomized instance/solver, and log
  the per-step `History` for every run so the plots (T2, T4, T5) are regenerable.
