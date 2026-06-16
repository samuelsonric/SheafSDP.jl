# Interior Point Method for Sheaf-Structured SDP — Implementation Plan

A detailed, phased successor to `plan.md`. Each phase has a concrete deliverable and a
verification checkpoint you can run before moving on. The math is worked out where it
touches an interface (residuals, NT scaling, the reduced Newton system, Mehrotra), because
those formulas decide the shapes of the objects you hand to `solve_kkt!`.

---

## 0. Notation and dimension bookkeeping

This is the part that bites. Three different "sizes" live on every vertex; keep them separate.

| symbol | meaning | per-vertex | total |
|--------|---------|-----------|-------|
| `d_v`  | side of the matrix block `P_v` (`d_v × d_v`) | `d_v` | — |
| `n_v`  | svec length = `trinum(d_v) = d_v(d_v+1)/2` | `n_v` | `n = Σ n_v` |
| `ν`    | cone degree / barrier parameter = `Σ d_v` | `d_v` | `ν = Σ d_v` |

- `p, d ∈ ℝⁿ` are stacked svec vectors; `n = length(p) = size(B, 2) = Σ ncols(B, v)`.
- `colrange(B, v)` is the `n_v`-length slice for vertex `v`; `ncols(B, v) = n_v` (svec length, **not** `d_v`).
- `d_v = triroot(ncols(B, v))`. This is the conversion we already fixed; it appears anywhere
  svec crosses back to matrix form.
- Edge/row side: `m = size(B, 1) = length(y) = length(g)`.

**svec convention (matters for the math below).** The code uses `α = √2` on off-diagonals,
which makes svec an *isometry*: `⟨P, D⟩ = tr(PD) = svec(P)' svec(D) = p'd`. Two consequences we
rely on:
1. The complementarity inner product is just `p'd` — no correction factor.
2. `svec(N M Nᵀ) = (N ⊗ₛ N) svec(M)` with the **symmetric Kronecker** `⊗ₛ`. This is what makes
   the barrier Hessian a clean Kronecker operator (Phase 2).

**Two name clashes to avoid** (both already latent in the code):
- `α` in `solve_kkt!` is the *augmented-system regularization* weight (`F = A + α BᵀB`), **not**
  the line-search step. Use a different name (`τ`, `γ`) for the step length.
- The comment in `solve_kkt!` says `S = Bᵀ F⁻¹ B`, but `schur!` computes `B F⁻¹ Bᵀ` (the `m × m`
  operator you actually want for `S Δy = r`). The code is right; the comment is flipped.

---

## 1. The problem and the central path

**Primal:**  min `c'p`  s.t.  `Bp = g`,  `P ⪰ 0`
**Dual:**    max `g'y`  s.t.  `Bᵀy + d = c`,  `D ⪰ 0`

`P = diag(P_v)`, `D = diag(D_v)` block-diagonal over vertices. The central path (parameter `μ > 0`):

```
Bp = g                         (primal feasibility)
Bᵀy + d = c                    (dual feasibility)
P_v D_v = μ I_{d_v}   ∀v        (perturbed complementarity, per block)
```

At `μ → 0` this is the KKT system of the SDP. The complementarity is a *matrix* equation and
`P_v D_v` is generally **not symmetric**, even though `P_v, D_v` are — this is the entire reason
SDP IPMs need a symmetrization/scaling, and why NT scaling shows up.

**Residuals** (define once, reuse everywhere):

```
r_p = g − B p              ∈ ℝᵐ     (want 0)
r_d = c − Bᵀy − d          ∈ ℝⁿ     (want 0)
μ   = ⟨p, d⟩ / ν = (p'd) / Σ d_v    (NOTE: ν, not n)
```

The `/ν` is easy to get wrong — it's the total *matrix* dimension `Σ d_v`, because
`tr(P_v D_v) = μ d_v` at the central point, so `Σ tr = μ Σ d_v = μ ν`.

> **Checkpoint 1.** Write `residuals!` and `mu`. Test on a hand-built feasible point
> (e.g. `P = D = I` block-diagonal, `y` chosen so `r_d = 0`): confirm `r_p = r_d = 0`,
> `μ = ⟨I,I⟩/ν = 1`, and that `p'd` equals `Σ tr(P_v D_v)` computed independently via `smat!`.

---

## 2. Nesterov–Todd scaling and the barrier Hessian `H`

### 2.0 Coordinate choice — DECIDED: natural

There is a fork in how the NT scaling enters the linear algebra:

- **Natural coordinates** (chosen): the scaling lives in the `(1,1)` block, `A = H = W⁻¹ ⊗ₛ W⁻¹`,
  and `B` is left untouched. `F = H + τ BᵀB`.
- **Scaled coordinates** (not chosen): the scaling is pushed onto the coboundary, `B̃ = B·G` with
  `G_v = W_v^{1/2} ⊗ₛ W_v^{1/2}` (symmetric, so `G' = G`), the `(1,1)` block collapses to `A = I`,
  and `F = I + τ B̃ᵀB̃ = I + τ GBᵀBG`. Requires the substitution `Δp = G q` and unscaling on exit.

**Why natural:** it matches the existing `solve_kkt!` interface exactly — `A` is already a
block-diagonal `BlockSparseMatrix` you assemble, so nothing in the solver changes. No `B̃`
plumbing, no `Δp = Gq` unscaling.

**What the two share (so we don't over-claim):** the reduced Schur complement is the *same*
operator either way — `B̃B̃ᵀ = B G² Bᵀ = B (W ⊗ₛ W) Bᵀ = B H⁻¹ Bᵀ` — so the inner iterative solve
(`it!` on `S`) sees identical conditioning. Scaled does **not** give a better-conditioned `S`.

**The one real difference** is in the matrix that gets the chordal Cholesky, the `(1,1)` block `F`:
- Natural: `λ_min(H) = 1/λ_max(W)² → 0` near the boundary, so `F = H + τ BᵀB` can go nearly
  singular from below in directions `BᵀB` doesn't cover; `κ(F) ~ κ(W)²`.
- Scaled: `F = I + τB̃ᵀB̃ ⪰ I` is floored, `κ(F) ~ λ_max(W)²`.

So scaled is mildly more stable *in the factorization of `F`*, not in the inner solve. We accept
the natural-coordinate factorization risk and **manage it with `τ`** (next note), in exchange for
zero interface change.

> **Consequence — `τ` is load-bearing, not cosmetic.** In natural coordinates the augmentation
> weight `τ` (the `α` arg of `solve_kkt!`) is what keeps `F = H + τBᵀB` away from the small-eigenvalue
> collapse as iterates approach the boundary. Keep it a live tunable; expect to raise it late in the
> solve. Do **not** set-and-forget.

### 2.1 What we need

`solve_kkt!` wants its `(1,1)` block `A` to be a **block-diagonal `BlockSparseMatrix`** whose
`v`-th diagonal block is the `n_v × n_v` SPD matrix `H_v` representing the linear map

```
H_v : svec(M) ↦ svec(W_v⁻¹ M W_v⁻¹)
```

where `W_v` is the NT scaling point for `(P_v, D_v)`. Equivalently `H_v = W_v⁻¹ ⊗ₛ W_v⁻¹`.
The key property that makes everything line up:

```
H_v p_v = d_v          (because W_v⁻¹ P_v W_v⁻¹ = D_v, see below)
```

### 2.2 Definition

`W_v` is the unique SPD matrix with

```
W_v D_v W_v = P_v       ⟺      W_v⁻¹ P_v W_v⁻¹ = D_v.
```

The common scaled point `V_v = W_v^{-1/2} P_v W_v^{-1/2} = W_v^{1/2} D_v W_v^{1/2}` is SPD, and the
central path in scaled coordinates is simply `V_v = √μ · I`.

### 2.3 Stable computation (Cholesky + SVD)

Do **not** form `W_v` from `P^{1/2}`/`D^{1/2}` directly. Use:

```
L_P = chol(P_v)                  # P_v = L_P L_Pᵀ
L_D = chol(D_v)                  # D_v = L_D L_Dᵀ
G   = L_Pᵀ L_D                   # d_v × d_v
SVD: G = U S Vᵀ                  # S = diag(s₁..s_{d_v}), s_i > 0
W_v = L_P U S⁻¹ Uᵀ L_Pᵀ          # = R Rᵀ with R = L_P U S^{-1/2}
```

**Why this is correct** (worth keeping in a comment): `L_Pᵀ D_v L_P = G Gᵀ = U S² Uᵀ`, so

```
W_v D_v W_v = L_P U S⁻¹Uᵀ (L_Pᵀ D_v L_P) U S⁻¹Uᵀ L_Pᵀ
            = L_P U S⁻¹ (S²) S⁻¹ Uᵀ L_Pᵀ = L_P L_Pᵀ = P_v.  ✓
```

The singular values carry the geometry: `s_i = √λ_i(P_v^{1/2} D_v P_v^{1/2})` and
`Σ s_i² = ⟨P_v, D_v⟩`. They are also the **eigenvalues of the scaled point** `V_v`
(`eig(V_v) = s`, verified numerically to machine precision), which the corrector's Lyapunov solve
`L_V⁻¹` reuses (Phase 4). Note they do **not** substitute for the step-to-boundary bound — that is a
*fresh* eigenproblem per direction; see §5.

> This is **not** what the current `meanblock!` computes — it builds a non-symmetric
> `S ≈ Q L⁻¹` and (separately) had the `n_v` vs `d_v` conflation. Replace the body with the
> Cholesky+SVD construction above, or verify the existing block against §2.4 before trusting it.

### 2.4 Assembling `H_v` into `A`

Two ways; build the first, keep the second for speed once correctness is established.

**(a) Column-by-column (simple, robust).** For `k = 1..n_v`: take svec basis vector `eₖ`,
`smat!` → `Eₖ` (`d_v × d_v` symmetric), form `W_v⁻¹ Eₖ W_v⁻¹` (two triangular solves with `R`),
`svec!` → column `k` of `H_v`. Place as `block(A, v, v, v)`.

**(b) Symmetric Kronecker (fast).** `H_v = W_v⁻¹ ⊗ₛ W_v⁻¹` directly in svec coordinates. Same
result; avoids `n_v` matvecs. Validate it equals (a) on random SPD blocks before switching.

`A` is block-diagonal with the **same column structure as `B`** (block `v` is `n_v × n_v` on the
diagonal). `solve_kkt!`'s `copydia!` copies these diagonal blocks into `F`, so the structure must
match exactly.

> **Checkpoint 2.** On random SPD `(P_v, D_v)`:
> 1. `‖W_v D_v W_v − P_v‖ < tol` and `W_v = W_vᵀ`, `W_v ≻ 0`.
> 2. `H_v` SPD; `H_v` built by (a) and (b) agree.
> 3. **The load-bearing identity:** `H_v · svec(P_v) ≈ svec(D_v)` (i.e. `H p = d` blockwise).
>    If this fails, the `A`/residual wiring downstream is wrong no matter what else checks out.
> 4. `Σ s_i² ≈ ⟨P_v, D_v⟩` and `s_i² ≈ λ_i(P^{1/2} D P^{1/2})`.

---

## 3. The reduced Newton system and the `solve_kkt!` mapping

### 3.1 Derivation

Linearize the three central-path equations and symmetrize the complementarity via NT (work in
scaled space, where it's clean). With scaled directions
`ΔP̃ = W^{-1/2} ΔP W^{-1/2}`, `ΔD̃ = W^{1/2} ΔD W^{1/2}`, the symmetrized complementarity is

```
ΔP̃ + ΔD̃ = σμ V⁻¹ − V − L_V⁻¹(2nd-order term)      (scaled, symmetric)
```

where `L_V(X) = ½(VX + XV)` is the Lyapunov operator (see §4 — for the centering/first-order part
`L_V⁻¹` evaluates in closed form and disappears; for the 2nd-order term it does not).
Unscaling (`apply W^{1/2}(·)W^{1/2}`, use `W D W = P`, `W^{-1}PW^{-1}=D`) turns the RHS into a
matrix `R_c` and gives, in svec coordinates:

```
Δp + 𝒲 Δd = r_c       where  𝒲 = H⁻¹,  r_c = svec(R_c)
⟹  Δd = H (r_c − Δp)
```

Substitute into the linearized feasibility equations `B Δp = r_p` and `Bᵀ Δy + Δd = r_d`:

```
H Δp − Bᵀ Δy = H r_c − r_d
     B Δp      = r_p
```

### 3.2 Mapping onto `solve_kkt!`

`solve_kkt!` solves `[A Bᵀ; B 0][x; y] = [f; g]`. Match by setting `w = −Δy`:

```
A = H                        (block-diagonal NT Hessian, Phase 2)
x = Δp
y = w = −Δy                  (recover Δy = −y after the solve)
f = H r_c − r_d
g = r_p
```

Then recover the dual slack direction from dual feasibility (cheaper than re-applying `H`):

```
Δd = r_d − Bᵀ Δy
```

This is the precise version of plan.md's "Connection to solve_kkt!" — note `f` is **not** just
the scaled dual residual, and there's a **sign flip** on the recovered `Δy`.

> Pick the augmentation weight `τ` (the `α` arg of `solve_kkt!`) on the order of `1`–`10`; per §2.0
> it conditions `F = H + τBᵀB` and does not change the solution. In natural coordinates this is
> load-bearing near the boundary — see the §2.0 consequence note.

> **Checkpoint 3.** Pick `r_c` for a centered step (§4 predictor RHS), call `solve_kkt!`, then
> verify the *original* (unaugmented) system residuals directly:
> `‖B Δp − r_p‖`, `‖H Δp − BᵀΔy − (H r_c − r_d)‖`, and after recovering `Δd`,
> `‖BᵀΔy + Δd − r_d‖`, all `≲ √eps`. This checks the sign conventions end-to-end.

---

## 4. Predictor / corrector right-hand sides

Everything above is generic; only `r_c` changes between steps.

**Affine / predictor (`σ = 0`, no centering, no 2nd order):**
```
R_c,v = −P_v      ⟹   r_c = −p      ⟹   f = H(−p) − r_d = −d − r_d
```
(using `H p = d`). Solve once → `(Δp^aff, Δy^aff, Δd^aff)`.

**Centering RHS — derivation (not just assertion).** The NT-symmetrized linearized
complementarity in scaled space is the clean `dp + dd = σμ V⁻¹ − V` (the NT direction's defining
virtue — no Sylvester operator left over *for the first-order part*). Unscale via `W^{1/2}(·)W^{1/2}`,
using `W^{1/2}VW^{1/2}=P` and `W^{1/2}V⁻¹W^{1/2}=WP⁻¹W=D⁻¹`:
```
ΔP + W ΔD W = W^{1/2}(σμV⁻¹ − V)W^{1/2}  ⟹  R_c = σμ D⁻¹ − P   (centering part)
```
(For the second-order term, below, this clean cancellation **fails** and `L_V⁻¹` survives.)

**Centering parameter (Mehrotra):**
```
α_p^aff, α_d^aff  = step-to-boundary for P, D along affine dir   (Phase 5)
μ_aff = ⟨p + α_p^aff Δp^aff,  d + α_d^aff Δd^aff⟩ / ν
σ     = (μ_aff / μ)³                                  # clamp to [0,1]
```

**Corrector (centered + 2nd order) — needs the `V`-eigendecomposition.** Earlier drafts of this
plan claimed the cross-term "collapses" to `sym(ΔP^a ΔD^a W)` with no operator. **That is wrong** —
it silently drops an inverse Lyapunov solve. The correct derivation:

The Jordan-symmetrized complementarity `P̃ ∘ D̃ = σμ I` (`X ∘ Y = ½(XY+YX)`), linearized about the
scaled point `P̃ = D̃ = V`, is
```
V ∘ (dp + dd) + dp^a ∘ dd^a = σμ I − V²
```
with `dp = W^{-1/2}ΔP W^{-1/2}`, `dd = W^{1/2}ΔD W^{1/2}`. Inverting the Lyapunov operator
`L_V(X) = V ∘ X = ½(VX + XV)`:
```
dp + dd = σμ V⁻¹ − V − L_V⁻¹(dp^a ∘ dd^a)
```
The two **first-order** terms are eigen-images of `L_V` (`L_V⁻¹(σμI) = σμV⁻¹`, `L_V⁻¹(V²) = V`), so
they pass through cleanly and unscale to the centering RHS `σμ D⁻¹ − P`. The **second-order** term
does **not**: `dp^a ∘ dd^a` is a generic symmetric matrix that does not commute with `V`, so `L_V⁻¹`
stays. The full corrector RHS, **per block**, is therefore
```
R_c = σμ D⁻¹ − P − W^{1/2} · L_V⁻¹(dp^a ∘ dd^a) · W^{1/2},   sym(X) = (X + Xᵀ)/2
```

> **Why the bare `sym(ΔP^a ΔD^a W)` is wrong.** Pushing the half-scalings through gives the *identity*
> `sym(ΔP^a ΔD^a W) = W^{1/2}(dp^a ∘ dd^a)W^{1/2}` — i.e. the Jordan product *without* `L_V⁻¹`.
> The LP special case makes the omission unmistakable: there `L_V` is multiplication by the scalar
> `V = √(xz)`, and the textbook corrector term is `dx^a dz^a / V` — the `1/V` is exactly `L_V⁻¹`.
> Dropping it leaves a second-order complementarity residual of `(I − L_V)(dp^a ∘ dd^a)`, nonzero
> away from `V ∝ I`. It still **converges** (the first-order part is exact) but the Mehrotra
> acceleration is degraded — more iterations, more erratic steps. A bare-product corrector is a
> defensible cheap *approximation*, but it is not the exact NT Mehrotra corrector; don't present it
> as one.

**Computing the corrector term — no new factorization.** The Lyapunov solve looks like it needs
the eigendecomposition of `V` (`L_V⁻¹(M) = Q[(QᵀMQ)_{ij}/(½(s_i+s_j))]Qᵀ` for `V = Q diag(s) Qᵀ`),
but **you never form `V`, `W^{1/2}`, or a fresh eig/SVD.** Two facts collapse it onto quantities
`meanblock!` already produces:
1. The eigenvalues `s_i` of `V` are exactly the SVD singular values from §2.3 (`eig(V) == s`,
   verified to machine precision).
2. The solve folds into a congruence by `R = L_P U S^{-1/2}` (already built — `W = R Rᵀ`) plus an
   entrywise divide, because `R = W^{1/2} O` (polar) with `O` the eigenvectors of `V`, so the
   eigenvector conjugations cancel.

The full corrector cross-term `W^{1/2} L_V⁻¹(dp^a ∘ dd^a) W^{1/2}` therefore equals
```
C = R · [ (R⁻¹ K R⁻ᵀ)_{ij} / (½(s_i + s_j)) ] · Rᵀ ,      K = sym(ΔP^a ΔD^a W)
```
with `R⁻¹ = S^{1/2} Uᵀ L_P⁻¹` (Cholesky/SVD factors + one triangular solve) and `K` needing only `W`
and the predictor directions. Verified to `2e-15` against an explicit `eig(V)` Lyapunov solve.

Per block per iteration this adds only a couple of small dense matmuls, one triangular solve against
`L_P`, and an entrywise divide — reusing `s_i`, `U`, `L_P`, `W` from the scaling. **No eigendecomposition,
no SVD, no matrix square root, no `V` formed.** (Forming `V` and eig-ing it would be the *more*
expensive route — it needs `W^{1/2}` plus an eig of `V`, i.e. two small eigs; avoid it.) The full
corrector RHS is then `R_c = σμ D⁻¹ − P − C`; `svec` it and form `f = H·r_c − r_d` with the
already-assembled `H`. Solve a second time (same `H`, same `B`, same `F` factorization — only `f`
changes).

> **Phasing tip.** Implement and verify a **centered-only** step first: drop the 2nd-order term
> and use a fixed schedule `σ ∈ (0,1)` (e.g. `σ = 0.1`–`0.5`). This gives a working short/long-step
> path-follower (Phase 6) you can validate before adding the fiddly Mehrotra correction (Phase 7),
> which is the single most error-prone formula in the whole method — and the one whose missing
> `L_V⁻¹` is easy to overlook because the solver still converges without it.

> **Checkpoint 4.** With the centered-only RHS, one step from a feasible interior point should
> *decrease* `μ` by roughly the factor `σ` and keep `r_p, r_d` near zero (they're already zero and
> the step preserves feasibility to linear order). Confirm `μ⁺ ≈ σμ`.
>
> **Checkpoint 4b (corrector).** On random SPD blocks, assert the second-order complementarity
> residual `‖V∘(dp+dd) + dp^a∘dd^a − (σμI − V²)‖` is at tolerance with the `L_V⁻¹` corrector, and
> confirm it is **large** (≈ `‖(I − L_V)(dp^a∘dd^a)‖`) if you drop `L_V⁻¹`. This is the cheapest
> guard against silently reintroducing the bug.

---

## 5. Step length to the cone boundary

Largest `τ` with `P_v + τ ΔP_v ⪰ 0`, per block, then take the min. Reuse `L_P = chol(P_v)`
from Phase 2:

```
M = L_P⁻¹ ΔP_v L_P⁻ᵀ            # symmetric
λ = λ_min(M)                    # most negative eigenvalue (or 0 if none)
τ_p,v = (λ < 0) ? min(1, −γ/λ) : 1      # γ ∈ (0.9, 0.99), or adaptive
τ_p = min_v τ_p,v
```

Same for `D` with `L_D`. Use possibly-different `τ_p`, `τ_d` for primal and dual (standard, and
helps). The eigenvalue can be the cheap symmetric one on a `d_v × d_v` matrix. Note: this is a *fresh*
eigenproblem per direction — it depends on `ΔP_v`, so the SVD `s_i` from the scaling (which
capture the current P–D geometry, `s_i² = λ_i(P^{1/2}DP^{1/2})`) do **not** substitute for it.
Reuse `L_P, L_D` (the factors), not `s_i`.

> **Checkpoint 5.** After `τ_p, τ_d`, assert `cholesky` of `P + τ_p ΔP` and `D + τ_d ΔD`
> *succeeds* for every block (the definitive PD test). With `γ < 1`, it should always succeed;
> if a block ever fails, the eigenvalue/sign logic is off.

---

## 6. Path-following loop (centered, no Mehrotra) — first working solver

```
initialize (P, D, y)                      # Phase 8
while not converged:
    residuals! → r_p, r_d, μ
    NT scaling → W_v, assemble A = H       # Phase 2
    r_c = centered RHS (fixed σ)           # Phase 4, no 2nd-order
    solve_kkt! → Δp, Δy; recover Δd
    τ_p, τ_d step-to-boundary              # Phase 5
    p += τ_p Δp;  d += τ_d Δd;  y += τ_d Δy
```

**Convergence test:**
```
‖r_p‖/(1+‖g‖) < εfeas,  ‖r_d‖/(1+‖c‖) < εfeas,  μ < εμ
```

> **Checkpoint 6.** Solve a *tiny* SDP with a known/independent answer:
> - a `1×1`-block-per-vertex case (this is an LP / weighted graph problem — sanity floor), then
> - a single `2×2` or `3×3` block where you can cross-check the optimum by a dense reference
>   (e.g. brute-force KKT, or Convex.jl/JuMP on the same data).
> Verify the final `μ`, primal/dual objectives match (`c'p ≈ g'y`, the duality gap closes), and
> the KKT residuals are at tolerance. Track `μ` per iteration — it should contract roughly
> geometrically. If `τ` collapses toward 0, suspect the scaling (Checkpoint 2.3) or the RHS signs.

---

## 7. Mehrotra predictor–corrector — add once Phase 6 is solid

Replace the fixed-`σ` step with: affine solve → `σ = (μ_aff/μ)³` → corrected solve (§4, **with the
`L_V⁻¹` Lyapunov term**). Reuse the **same `A = H` and the same `F` factorization** for both solves;
only `f` changes, so the expensive chordal factorization in `solve_kkt!` is done once per iteration
and the corrector is a cheap re-solve. The only extra work the corrector adds over the predictor is
the per-block Lyapunov solve `L_V⁻¹` — and that is just a congruence by `R` plus an entrywise divide
(no new eig/SVD; see §4). (This is the payoff of the structure — note it explicitly in the loop.)

> **Checkpoint 7.** On the same problems as Checkpoint 6, the Mehrotra version should reach the
> same optimum in **noticeably fewer iterations** (typically <½). Same answer, fewer steps — if it
> diverges or stalls, the 2nd-order term `L_V⁻¹(dp^aff ∘ dd^aff)` is the prime suspect; temporarily
> zero it to fall back to centered-corrector and confirm the rest is intact. If it converges but
> *slowly* (no better than centered), suspect the `L_V⁻¹` was dropped — see Checkpoint 4b.

---

## 8. Initialization, termination, robustness

- **Start.** Infeasible-start primal–dual is simplest: `P = D = ξ·I` block-diagonal (`ξ` scaled to
  the data, e.g. from `‖c‖, ‖g‖`), `y = 0`. Carry `r_p, r_d ≠ 0` through the iterations — the
  derivation in §3 already handles nonzero residuals, so no change needed.
- **Termination.** `εfeas, εμ ≈ 1e-8` (Float64). Also cap iterations and watch for `μ` stalling.
- **Infeasibility / failure.** If `τ` stays tiny while residuals plateau, flag possible
  infeasibility or numerical trouble rather than looping to `itmax`. Optionally monitor the
  duality gap `c'p − g'y` alongside `μ`.
- **Conditioning.** Keep the augmentation weight `τ_aug` (the `α` of `solve_kkt!`) as a tunable;
  raise it if the inner iterative solve (`it!`) needs many iterations near convergence.

---

## Build order summary (with the checkpoint gating each)

| Phase | Deliverable | Gate |
|------|-------------|------|
| 0 | `triroot`/`trinum`, svec isometry, dim bookkeeping | round-trip + isometry tests |
| 1 | `residuals!`, `mu` | feasible-point gives 0 residuals, `μ=1` |
| 2 | NT scaling (Cholesky+SVD), `H_v`, assemble `A` | `WDW=P`, `H p = d`, (a)≡(b) |
| 3 | reduced-system wiring to `solve_kkt!` | original KKT residuals `≲√eps` |
| 4 | predictor/corrector RHS (centered first) | `μ⁺ ≈ σμ`; 2nd-order residual at tol (4b) |
| 5 | step-to-boundary | `chol(P+τΔP)` succeeds every block |
| 6 | centered path-follower | matches reference SDP, gap closes |
| 7 | Mehrotra (with `L_V⁻¹` corrector) | same answer, fewer iterations |
| 8 | init / termination / robustness | clean exit on tiny problems |

---

## Watch-list (carried over from the code review)

- `meanblock!` / `cholblock!`: `n = ncols(B,v)` is the **svec** length; the matrix side is
  `triroot(n)`. Reshape workspaces to `d_v × d_v`, slice `p`/`d` over the `n_v`-length range.
- `meanblock!` currently computes a non-symmetric `S ≈ Q L⁻¹`; the plan's §2.3 SPD construction is
  what the `H = W⁻¹ ⊗ₛ W⁻¹` interface needs. Reconcile or replace, then gate on Checkpoint 2.
- **`corrector_rhs!`: the 2nd-order term needs the inverse Lyapunov solve `L_V⁻¹(dp^a ∘ dd^a)`, not
  the bare `sym(ΔP^a ΔD^a W)`** — the latter is the Jordan product missing `L_V⁻¹` (see §4). Reuse
  `s_i = eig(V)` from the scaling for the Lyapunov eigenvalues. Omitting `L_V⁻¹` still converges but
  kills the Mehrotra speedup; guard it with Checkpoint 4b.
- `solve_kkt!` comment `S = BᵀF⁻¹B` should read `B F⁻¹ Bᵀ` (code is correct).
- Don't overload `α`: augmentation weight vs. line-search step are different quantities.
