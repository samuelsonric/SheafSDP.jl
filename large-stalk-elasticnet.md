# Large-stalk elastic-net (the capstone)

The synthesis instance. It combines the dense quadratic of the large-stalk QP
(`large-stalk-qp.md`) with the `ℓ₁` orthant of the large-stalk POS
(`large-stalk-instances.md` §3), on coupled blocks, at stalk size `≥ 30` — using
the case-(c) reification whose *semantics* were worked out in `regularized-fuel.md`.

What earns it a page: it is the **only** instance in the suite where a dense `Q`
block and a barrier-curved cone coexist on the same logical variable **with
`ν > 0`**, so it is the only one that drives the full `hess!` + corrector path at
realistic block size. The plain QP (`ν = 0`) collapses the outer loop; the plain
POS/SOC run `Q = 0`. This one runs everything at once.

References: `regularized-fuel.md` (reification semantics), `large-stalk-*.md`
(chassis §2, restriction maps §6, knobs §7), `conic-recipes.md` §0 (the
block-separability rule), §3 (lifting), §12 (oracle).

---

## 1. The objective

Static chassis (no dynamics), elastic-net node cost on the decision cochain:

```
min   Σ_i  ½ x_i' R_i x_i + c_i' x_i + λ ‖x_i‖₁,     R_i ⪰ 0,   λ > 0
s.t.  δ_F x = b           (b = δx₀, realizable)
```

With `R_i` a **dense** SPD block (cross terms), the `ℓ₁` split couples the orthant
halves through `−R` (the §0 obstruction, `regularized-fuel.md` §2), so the
dense-`R` route is **case (c): reify the logical variable**.

---

## 2. The instance (case c, at scale)

Same chassis as the other large-stalk instances: `N = 6` on `P₆`, `n_v ∈ {30,48}`,
`d_e = 16`, orthonormal-row restriction maps, `b = δx₀`.

Per agent, one logical variable `x_i ∈ ℝ^{n_v}` expands to **three large blocks**
plus a large private coupling row:

| column-block | dim | cone | `Q` | `c` |
|---|---|---|---|---|
| `x_i` (reified) | `n_v` (30/48) | `:NOC` | `R_i` | `c_i` |
| `x_i⁺` | `n_v` | `:POS` | — | `λ𝟏` |
| `x_i⁻` | `n_v` | `:POS` | — | `λ𝟏` |

| row-block | dim | equation | `g` |
|---|---|---|---|
| `split_i` | `n_v` | `x_i − x_i⁺ + x_i⁻ = 0` | `0` |
| `coord_e` | `16` | `F_{i⊴e} x_i − F_{j⊴e} x_j = b_e` | `b_e` |

The reified `x_i` carries the quadratic and the coordination coupling; the halves
carry only the `ℓ₁`. (Coercivity from `R_i ≻ 0` plus the coupling bounds it; add a
box on the reified `x_i`, two `:POS` slacks, if you want `R_i` singular tests.)

---

## 3. Why it's the capstone — four stress points the others miss

**(a) Dense `Q` and a cone, coexisting, with `ν > 0`.** The reified `:NOC` block
gets `H_v = R_i` (dense, from `Q`); the `:POS` halves get `H_v = diag(d/p)`
(barrier). Both feed the same global `H = blockdiag(...)` and the same
`F = H + αB'B`. Unlike the plain QP, `degree(::POS,n) = n` makes
`ν = 2 Σ_i n_v > 0`, so the **centering actually runs**: `μ`, the affine measure
`μa`, and `σ` are all well-defined (the `ν = 0` hazard of `large-stalk-qp.md` §4
does *not* arise here). The corrector loops over mixed cones —
`poscorr!` on the halves, the `:NOC` `corr!` contributing zero — so `μ` reflects
only the conic part, correctly. This is the genuine end-to-end exercise of
`hess!` + `corrector!` with a populated dense `Q`.

**(b) Mixed conditioning.** As the IPM approaches the boundary, `diag(d/p)` on the
halves spreads toward `0`/`∞` while the `R_i` blocks stay fixed and dense. The KKT
factorization must stay stable across blocks whose conditioning diverges for
*different reasons* (barrier vs. data) — a regime neither the all-barrier POS/SOC
nor the all-`Q` QP produces. Sweep `κ(R_i)` and `λ` to probe it.

**(c) Large private (non-coordination) rows.** `split_i` is an `n_v × 3n_v` block
`[I | −I | I]` coupling the reified variable to both halves — a *private* row
local to agent `i`, distinct from the cross-agent coordination rows. In `B'B` it
makes `{x_i, x_i⁺, x_i⁻}` a **clique within each agent**, on top of the
cross-agent sheaf-Laplacian coupling. The line graph therefore carries per-agent
triangles plus the path coupling — a richer elimination structure than the POS
box rows (which couple only *pairs* `{x_i, w_i}`). Good stress for
`weightedgraph → linegraph → symbolic` at scale.

**(d) The widest cone/curvature mix in one solve.** It is the single instance that
simultaneously holds large dense `Q` blocks, large `:POS` orthants, large private
coupling rows, and the high-dimensional coordination coboundary. If you want one
problem that lights up the most code at realistic size, this is it.

---

## 4. Reductions and oracle

It degrades cleanly to its parents, which makes it a good regression anchor:

| set | becomes |
|---|---|
| `λ → 0` | large-stalk plain QP (`large-stalk-qp.md`) — and inherits its `ν = 0` caveat, so keep `λ > 0` unless you want that path |
| `R_i → 0` | large-stalk distributed LP (POS, `large-stalk-instances.md` §3) |
| `n_v → small` | the elastic-net 2b (`regularized-fuel.md`) |

For the oracle (`conic-recipes.md` §12): with `R_i ≻ 0` and `λ > 0` the objective
is **strictly convex**, so the optimum is unique and solution-level comparison
against Clarabel (native quadratic objective — no epigraphing of the `½x'Rx`
term) and Mosek is meaningful. Because both the dense `Q` and the orthant are
exercised, a discrepancy here that *isn't* present in the isolated QP or POS legs
points specifically at the `Q`-meets-barrier assembly — the `axpy!(block(Q,v,v,v), Hv)`
interacting with `diag(d/p)` in the same `F`.
