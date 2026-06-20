# Large-stalk plain QP (the `:NOC` base case)

Completes the large-stalk trio alongside `large-stalk-instances.md` (POS, SOC).
It reuses that document's chassis (§2), restriction-map design (§6), and scaling
knobs (§7) verbatim — only the node objective and cone change. What makes it worth
a separate page is that it is *not* just "the third cone": it is the **base case**
the entire framework reduces to, and it drives the one solver path the POS/SOC
instances leave completely untested — the `Q` path — while also surfacing the
`ν = 0` corner of `step!`.

---

## 1. What it is — the variational form of a sheaf-Laplacian solve

Plain distributed QP under hard sheaf consensus:

```
min   Σ_i  ½ x_i' R_i x_i + c_i' x_i,     R_i ⪰ 0,   x_i free
s.t.  δ_F x = b           (b ∈ im δ; b = δx₀ for a realizable target)
```

There are **no inequality constraints** — every block is free (`:NOC`). Its KKT
system is *linear*:

```
[ blockdiag(R_i)   δ' ] [ x ] = [ -c ]
[      δ           0  ] [ y ]   [  b ]
```

whose Schur complement `δ R⁻¹ δ'` is a **weighted sheaf Laplacian**. So this
instance is exactly the variational/optimization form of "solve a sheaf-Laplacian
system" — the linear core that the heat-equation and projection steps in the paper
sit on top of, and that everything in the conic recipes reduces to once the cones
are stripped away. For completeness of the test suite it is the foundation, not an
afterthought.

---

## 2. The instance

Same chassis as the POS/SOC instances: `N = 6` on the path `P₆`, heterogeneous
stalks `n_v ∈ {30, 48}`, edge stalks `d_e = 16`, orthonormal-row restriction maps
`F_{i⊴e} ∈ ℝ^{16×n_v}`, target `b = δx₀`.

**Modeling object.** `f_i(x_i) = ½ x_i' R_i x_i + c_i' x_i`, with `R_i` a **dense
SPD** `n_v × n_v` block (draw `R_i = G_i G_i' + εI`).

| column-block | dim | cone | `Q` | `c` |
|---|---|---|---|---|
| `x_i` | `n_v` (30 or 48) | `:NOC` | `R_i` | `c_i` |

| row-block | dim | equation | `g` |
|---|---|---|---|
| `coord_e` | `16` | `F_{i⊴e} x_i − F_{j⊴e} x_j = b_e` | `b_e` |

That is the entire problem: free variables, a dense quadratic per block, coupled
only by the coordination coboundary. No box, no epigraph, no slacks.

---

## 3. What's new under test — the `Q` path at scale

The POS and SOC instances both have `Q = 0`; their curvature comes from the
barrier (`poshess!`, `sochess!`). **This is the first instance that populates `Q`
with large dense blocks**, so it is the only one that exercises:

- `residuals!` → `mul!(rd, Symmetric(Q, :L), p, 1, 1)` with a dense `n_v × n_v`
  `Q` block (≥ `30×30`);
- `hess!` (`ipm.jl`) → `axpy!(true, block(Q, v, v, v), Hv)` folding `R_i` into the
  vertex Hessian. For `:NOC`, `hess!` in `noc.jl` fills `Hv` with zeros first, so
  the entire `(1,1)` block is `H_v = R_i` — the Hessian is supplied *purely* by
  `Q`.

It is therefore the natural place to validate the `Symmetric(Q, :L)` /
`block(Q, v, v, v)` machinery at realistic block size, and to test the `:NOC`
curvature requirement (`noc.jl`: *"Requires Q_v ≻ 0"*) when `Q_v` is a genuine
dense SPD matrix rather than a `4×4` toy. With `R_i ≻ 0` the block `F = H + αB'B`
is PD without help; with `R_i` singular you rely on the `αB'B` augmentation to
cover the nullspace — worth testing both by tuning `ε` down to `0`.

---

## 4. The `ν = 0` property — isolation, and a caveat to guard

Because every block is `:NOC` and `degree(::NOC, n) = 0`, `conedegree` returns
**`ν = 0`**. Two consequences, one good and one to watch.

**The good: it isolates the KKT/linear-algebra core.** With `ν = 0` there is no
barrier, no complementarity, no centering — the problem is a single linear KKT
solve. `step!` converges as soon as the residuals are small, so the IPM outer
loop collapses and you are testing the Uzawa/ADMM chordal factorization of
`blockdiag(R_i) + αL_F` *in isolation*, with no interior-point dynamics on top.
For debugging the block factorization and the `B'B = L_F` assembly at large stalk
size, this is the cleanest possible probe.

**The caveat: the Mehrotra centering arithmetic isn't guarded for `ν = 0`.** In
`step!`, `μ` is correctly guarded (`iszero(s.ν) → μ = zero(T)`), but the affine
duality measure and the centering ratio are not:

```
μa = dot(pa, da) / s.ν          # ν = 0 → division by zero
σ  = clamp((μa / μ)^3, 0, 1)     # μ = 0 → μa / μ ill-defined
```

so a pure-`:NOC` instance produces `σ = NaN`, then `σ·μ = NaN` poisons
`corrector!` and the step. Note this bites **any** all-equality instance — even
the recipe §4 pure-effort QP without a box. Two clean resolutions:

1. **Guard `ν = 0` in `step!`:** skip centering (`σ = 0`) and take the affine
   direction with a unit step — which is *exact* for a convex equality-constrained
   QP, since the affine Newton step already solves the linear KKT system. This is
   the principled fix and makes the plain-QP case a first-class citizen.
2. **Keep one token conic block** (e.g. a single bound rendered as a `:POS` slack)
   so `ν > 0` and the existing path runs. Cheaper, but it means you can't test the
   genuinely-unconstrained problem.

Recommend (1) — it's a few lines and it turns the base case into a legitimate
"solve the sheaf-Laplacian system" entry point through the same `IPMProblem`
interface.

---

## 5. Knobs, conditioning, oracle

Scaling knobs are the §7 set from `large-stalk-instances.md` (`n_v`, `d_e`, `N`,
graph fill), with one QP-specific axis: the **conditioning of `R_i`**. Sweeping
`κ(R_i)` from well-conditioned to near-singular (via `ε`) stresses the augmentation
and the chordal factor's numerical stability in a way the barrier-driven POS/SOC
blocks don't.

For the oracle (§8 / `conic-recipes.md` §12): `R_i ≻ 0` makes the QP **strictly
convex**, so the optimum is unique and solution-level comparison against Mosek /
Clarabel is meaningful, not just objective-level. And because the problem is a
pure linear solve, the commercial solvers return it to machine precision — making
this the tightest-tolerance leg of the whole suite and the best detector of a
sign or scaling error in the `B = δ_F` assembly.
