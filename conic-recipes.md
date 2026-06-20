# Recipes: convex homological programs → conic standard form

A construction guide for translating convex nonlinear homological programs (Hanks
et al., *Distributed Multi-agent Coordination over Cellular Sheaves*, eq. 7) into
the standard form consumed by the IPM solver. Scope is the **convex** slice:
quadratic edge potentials, convex node objectives. Nonconvex potentials (the
quartic distance/flocking terms) are out of scope and stay with ADMM.

Code references point at the solver sources: `ipm.jl`, `sheaf.jl`, `cone/*.jl`,
`utils.jl`, `kkt/*.jl`.

---

## 0. The target: what the solver actually consumes

The IPM solves a conic (possibly quadratic) program

```
min   c'p + ½ p'Q p
s.t.  B p = g
      p ∈ K
```

with the dual

```
max   g'y - ½ p'Q p
s.t.  B'y + d = c + Q p
      d ∈ K*
```

where `p`, `d` are the `svec` representations of block-diagonal primal/dual
variables, `K = ∏_v K_v` is a product of per-block cones, and `B` is a
`BlockSparseMatrix` whose **column-blocks are variables (vertices)** and
**row-blocks are constraints (outputs/edges)**.

The problem is handed over as an `IPMProblem{T,I}` (`ipm.jl`):

| field | type | meaning |
|---|---|---|
| `c` | `Vector{T}` | linear objective term, length `n = ncols(B)` |
| `g` | `Vector{T}` | constraint RHS, length `m = nrows(B)` |
| `B` | `BlockSparseMatrix{T,I}` | constraint matrix, `m × n` in block terms |
| `Q` | `BlockSparseMatrix{T,I}` | quadratic term, **block-diagonal**, `n × n` |
| `cones` | `Vector{Symbol}` | one symbol per column-block: `:SDP`, `:POS`, `:SOC`, `:NOC` |

`tocone` (`ipm.jl`) maps each symbol to its cone singleton. Build everything in
**natural order**; `init` (`CommonSolve.init`) applies the fill-reducing
permutation for you (`p = P*initp(...)`, `Q = selectvtxs(Q0, R.perm)`,
`cones = cones0[R.perm]`). Do not pre-permute.

> **Why `Q` must be block-diagonal.** `residuals!` applies the *full* matrix via
> `mul!(rd, Symmetric(Q, :L), p, 1, 1)`, but `hess!` (the `BlockSparseMatrix`
> method in `ipm.jl`) folds only the **diagonal** block into each vertex
> Hessian: `axpy!(true, block(Q, v, v, v), Hv)`. An off-diagonal block in `Q`
> would appear in the gradient `rd` but not in the Newton Hessian `H`, breaking
> the step. So all curvature that couples *different* blocks must travel through
> `B` as a constraint, never through `Q`. This is the single most important
> structural rule in this document.

---

## 1. The reduction every convex coordination goal obeys

Equation 7 imposes the coordination goal as the hard constraint
`L^{∇U}_F X[:,T] = 0`, where `L^{∇U}_F = δ' ∘ ∇U ∘ δ` is the nonlinear sheaf
Laplacian. With a **quadratic** potential `U_e(y) = ½‖y - b_e‖²` we have
`∇U(y) = y - b`, so

```
L^{∇U}_F x = δ'(δ x - b) = δ'δ x - δ' b.
```

Setting this to zero is `δ'δ x = δ' b`. Whenever `b` is **realizable**
(`b ∈ im δ` — automatic for consensus `b = 0`, and true for any consistent
formation, since pairwise displacements that close around every cycle are by
definition a coboundary), this is *equivalent* to the plain coboundary
constraint:

```
δ'δ x = δ' b   ⟺   δ x = b        (b ∈ im δ)
```

**Proof.** `b = δ x₀ ⇒ δ'δ(x - x₀) = 0 ⇒ x - x₀ ∈ ker δ'δ = ker δ ⇒ δ x = b.`

Consequence: you never form `δ'δ`. The coordination goal becomes coboundary rows
`δ_F X[:,T] = b` in `B`, which is exactly what `sheaf(I, J, V)` builds, at
coboundary sparsity (not its square). This is the cleanest possible mapping and
it preserves the `weightedgraph → linegraph → symbolic` chordal structure the
KKT solvers exploit.

---

## 2. The master template: a time-expanded coboundary

Every instance has the same skeleton — a block-diagonal stack of per-agent
dynamics blocks, plus a thin coordination layer coupling only the terminal
states.

```
        agent 1 cols     agent 2 cols     agent 3 cols
      ┌──────────────┬────────────────┬────────────────┐
 A1   │   B₁ (D₁)    │                │                │   per-agent
 A2   │              │   B₂ (D₂)      │                │   dynamics +
 A3   │              │                │   B₃ (D₃)      │   local cones
      ├──────────────┴────────────────┴────────────────┤
coord │  ±P on x_i^T columns only  (δ_F X[:,T] = b)     │   coupling
      └─────────────────────────────────────────────────┘
```

- **Per-agent block `B_i`** is the dynamics sheaf `D_i` from Example 4 of the
  paper: a path graph in time. Its rows are the LTI recursion and initial
  condition; its columns are the per-timestep states and controls, each carrying
  a cone. Convex node objectives `f_i` add *leaf* columns and rows to this block
  (Sections 4–6).
- **Coordination rows** are the coboundary `δ_F` of the coordination sheaf `F`,
  restricted to the terminal states `x_i^T`. For consensus `b = 0`; for
  formation `b = vec(b_e)`.

### Sign convention (`sheaf.jl`)

For each ordered endpoint pair, `sheaf` assigns `sign = -1` if `i < j` else
`+1`, and pushes `(edge, vertex i, sign·M)`. So for edge `e = ij` with `i < j`:

```
B[e, x_i^T] = -F_{i⊴e}     B[e, x_j^T] = +F_{j⊴e}
```

i.e. `B = -δ` under this convention. For consensus (`b = 0`) the sign is
immaterial. For formation, match `g` to the chosen sign.

### Running matrices (planar double integrator, the paper's examples)

State `x = (p_x, p_y, v_x, v_y) ∈ ℝ⁴`, control `u ∈ ℝ²`, Euler step `h`:

```
A = [ I₂  hI₂ ]      B_dyn = [ 0  ]      P = [ I₂  0₂ ]   (position projection,
    [ 0   I₂  ]              [ hI₂]                          the restriction map)
```

`P` is the restriction map for position consensus: `(δX[:,T])_e = P x_i^T - P x_j^T = p_i^T - p_j^T`.

---

## 3. The lifting principle (general convex `f_i`)

Write each agent's problem — dynamics, constraints, and node cost — as a
**cone-LP** (conic program with a block-diagonal quadratic):

```
min   c_i'x_i + ½ x_i'Q_i x_i
s.t.  A_i x_i = b_i
      x_i ∈ K_i
```

where `x_i` now bundles the original trajectory/controls **plus any
epigraph/auxiliary variables** the conic representation needs. Then graft it onto
the master template by the following dictionary:

| convex modeling object | lands in the solver as |
|---|---|
| original variable (state, control) | a column-block with cone `:NOC` (free) |
| epigraph / slack / aux variable | a **new** column-block (a *leaf stalk*) with the cone its membership requires |
| linear defining relation for an aux var | a **new** row-block (a *private* constraint, touches only agent `i`) |
| quadratic node cost | the diagonal block `Q_i` (block-diagonal only — see §0) |
| linear node cost | entries of `c` |
| conic membership `x_i ∈ K_i` | the `cones` symbol for that column-block |
| coordination goal | the shared coboundary rows of §1–2 |

This is the precise sense in which "stitch the individual CLPs together": each
`f_i` becomes a small sheaf grafted onto agent `i`'s time-path, and the
coordination coboundary couples the agents. The graph sparsity is preserved
because private rows touch only their own agent's columns; only the coordination
rows cross agents.

### Curvature bookkeeping (`NOC` and the augmentation)

A free column-block (`:NOC`) supplies **no** barrier curvature — `hess!` in
`noc.jl` fills zeros, and the header note reads *"Requires Q_v ≻ 0 to supply all
curvature in the (1,1) block."* If a `:NOC` block also has `Q = 0`, its only
curvature comes from the `α B'B` augmentation inside the KKT solve
(`init_kkt!`, `init_uzw!` / `init_admm!`). This works when the constraint rows
pin the block uniquely, but a small regularization is friendlier (§4).

`:POS`, `:SOC`, `:SDP` blocks supply their own curvature through the barrier
(`poshess!` gives `d/p`; `sochess!`; `sdphess!`), so they never need `Q`.

### Barrier parameter and free blocks

`conedegree` (`ipm.jl`) sets `ν = Σ_v degree(cones[v], ncols(B,v))` and
`μ = dot(p,d)/ν`. Degrees are `:POS → n`, `:SOC → 2`, `:SDP → triroot(n)`,
`:NOC → 0`. Free blocks contribute nothing to `ν` (their dual cone is `{0}`), so
adding states/controls as `:NOC` does not inflate the duality measure — correct,
since they carry no inequality.

---

## 4. Recipe — QP / `:NOC` (quadratic node cost)

The base case. No leaves, no extra rows; curvature lives entirely in `Q`.

**Modeling object.** `f_i = Σ_t (u_i^t)'R u_i^t` (control effort), optionally plus
tracking `Σ_t ‖x_i^t - r_i^t‖²_S`.

**Construction.**
- Column-blocks: `x_i^t` (`:NOC`), `u_i^t` (`:NOC`).
- Rows: `init` (`x_i^1 = c_i`), `dyn_t` (`x_i^{t+1} - A x_i^t - B u_i^t = 0`).
- `Q`: diagonal block `2R` on each `u_i^t`; if tracking, `2S` on each `x_i^t`
  with `c` entry `-2S r_i^t`.
- `g`: `c_i` on `init`, `0` on `dyn`.

**The two `Q` variants (both shown; pick per test):**

1. *Pure effort, `S = 0`.* Every state block is `:NOC` with `Q = 0`. Feasible set
   pins the states (init + dynamics), so Uzawa converges via the augmentation,
   but the bare singular `(1,1)` block leans on `α B'B`. Use this to probe how the
   augmentation copes.
2. *ε-regularized.* Add `Q_{x_i^T} = ε I` (and `c` term if tracking a target).
   Now the terminal `:NOC` blocks are honestly curved and the KKT is better
   conditioned. Recommended default for a first working instance.

---

## 5. Recipe — POS (`ℓ₁` effort, box, polytope)

The first recipe that adds leaves and rows. `:POS` blocks bring their own
curvature (`poshess!`: `H[i,i] = d[i]/p[i]`; `posmaxstep` does the ratio test),
so no `Q` is needed on them.

**Modeling object.** Minimum-fuel `f_i = Σ_t ‖u_i^t‖₁` (sparse / bang-off-bang
control), with the actuator box `|u_i^t| ≤ ū`.

**Reformulation.**
- *Residual split.* `u_i^t = u_i^{t+} - u_i^{t-}`, with `u_i^{t±} ≥ 0` two
  `:POS` blocks. Substitute directly into the dynamics row — no extra equality:
  `x_i^{t+1} - A x_i^t - B(u_i^{t+} - u_i^{t-}) = 0`.
- *Objective.* `‖u‖₁ = Σ (u⁺ + u⁻)` at optimum, so `c = 𝟏` on every `u^±`
  block, `Q = 0`.
- *Box as a slack row.* `u_i^{t+} + u_i^{t-} + w_i^t = ū 𝟏`, with `w_i^t ≥ 0` a
  `:POS` block. Feasibility forces `u⁺ + u⁻ ≤ ū` componentwise, hence
  `|u| ≤ ū` (since `|u| ≤ u⁺ + u⁻` always), even pre-optimality.

**Construction (per agent).**
- Column-blocks: `x_i^t` (`:NOC`), then `u_i^{t+}, u_i^{t-}, w_i^t` (`:POS`).
- Rows: `init`, `dyn_t` (now with `-B, +B` on `u^{t+}, u^{t-}`), `box_t`
  (`I, I, I` on `u^{t+}, u^{t-}, w^t`).
- `c`: `𝟏` on the `u^±` blocks, `0` elsewhere. `Q`: zero (or the §4 ε-term on
  states).
- `g`: `c_i` on `init`, `0` on `dyn`, `ū 𝟏` on `box`.

**Scaling knob.** `n = T` grows the orthant dimension `2(T-1)` per agent linearly
— this exercises `posmaxstep`'s ratio test and yields a genuinely sparse control
schedule to contrast against the smooth QP one.

---

## 6. Recipe — SOC (un-squared norm, slew, terminal ball, chance)

`:SOC` blocks bundle a scalar epigraph with a vector; curvature from
`sochess!`/`socscale!`, step from `socmaxstep`. Cache is `1 + n` (`soc.jl`).

**Modeling object.** Un-squared effort `f_i = Σ_t ‖u_i^t‖₂` (time-group-sparse —
whole control vectors switch off), or any norm-bounded cost/constraint.

**Reformulation (epigraph bundle).**
- Add a scalar `s_i^t ≥ 0` and form the `:SOC` block `ζ_i^t = (s_i^t; u_i^t) ∈ Q^{1+m}`.
- Feed the *tail* of `ζ_i^t` into the dynamics row in place of `u_i^t`.
- Objective is linear again: `c = 1` on the head `s_i^t`, `0` on the tail,
  `Q = 0`.

**On-theme variant — slew / jerk penalty.** `Σ_t ‖u_i^{t+1} - u_i^t‖₂`. Introduce
`d_i^t` with a leaf row `d_i^t - u_i^{t+1} + u_i^t = 0` — which is *itself a
coboundary on the time path*, a within-agent edge potential — then put the
`:SOC` block on `(s_i^t; d_i^t)`. This slots into the same homological assembly
as the dynamics rather than bolting on as a foreign block.

**Other one-liners (same move).**
- Terminal-ball reach `‖x_i^T - g_i‖₂ ≤ τ`: `:SOC` on `(τ; x_i^T - g_i)`, with a
  leaf row defining the shifted argument.
- Gaussian chance constraint `Pr(a'x ≤ b) ≥ 1-ε` (`ε ≤ ½`):
  `a'μ + Φ⁻¹(1-ε)‖Σ^{1/2}a‖ ≤ b`, an `:SOC` constraint on `Σ^{1/2}a`.

**Scaling knobs.** `n = m` (control dim = cone arm length) stresses
`socscale!`/`socmaxstep` on large cones; `n = T` gives many small cones.

---

## 7. Worked example: minimum-fuel POS, `N = 3` on `K₃`, `T = 3`

Consensus on terminal position (`b = 0`), planar double integrators. This is the
instance drawn in the accompanying diagrams.

### Column-blocks (variables) and cones, per agent `i ∈ {1,2,3}`

| block | dim | cone |
|---|---|---|
| `x_i^1, x_i^2, x_i^3` | 4 each | `:NOC` |
| `u_i^{1+}, u_i^{1-}, u_i^{2+}, u_i^{2-}` | 2 each | `:POS` |
| `w_i^1, w_i^2` | 2 each | `:POS` |

`cones` (natural order, agents concatenated):
`[:NOC,:NOC,:NOC,:POS,:POS,:POS,:POS,:POS,:POS]` × 3 agents = 27 column-blocks.

### Row-blocks (constraints)

Per agent `i`:

| row | dim | equation | `g` |
|---|---|---|---|
| `init` | 4 | `x_i^1 = c_i` | `c_i` |
| `dyn₁` | 4 | `-A x_i^1 + x_i^2 - B u_i^{1+} + B u_i^{1-} = 0` | `0` |
| `dyn₂` | 4 | `-A x_i^2 + x_i^3 - B u_i^{2+} + B u_i^{2-} = 0` | `0` |
| `box₁` | 2 | `u_i^{1+} + u_i^{1-} + w_i^1 = ū𝟏` | `ū𝟏` |
| `box₂` | 2 | `u_i^{2+} + u_i^{2-} + w_i^2 = ū𝟏` | `ū𝟏` |

Coordination (shared), edges `12, 13, 23`:

| row | dim | equation | `g` |
|---|---|---|---|
| `e=12` | 2 | `+P x_1^3 - P x_2^3 = 0` | `0` |
| `e=13` | 2 | `+P x_1^3 - P x_3^3 = 0` | `0` |
| `e=23` | 2 | `+P x_2^3 - P x_3^3 = 0` | `0` |

### `B` as block triples `(row_block, col_block, matrix)`

Per agent `i`:
```
(init_i, x_i^1, I₄)
(dyn₁_i, x_i^1, -A)   (dyn₁_i, x_i^2, I₄)   (dyn₁_i, u_i^{1+}, -B)   (dyn₁_i, u_i^{1-}, +B)
(dyn₂_i, x_i^2, -A)   (dyn₂_i, x_i^3, I₄)   (dyn₂_i, u_i^{2+}, -B)   (dyn₂_i, u_i^{2-}, +B)
(box₁_i, u_i^{1+}, I₂) (box₁_i, u_i^{1-}, I₂) (box₁_i, w_i^1, I₂)
(box₂_i, u_i^{2+}, I₂) (box₂_i, u_i^{2-}, I₂) (box₂_i, w_i^2, I₂)
```
Coordination (matches `sheaf.jl` `-/+` for `i < j`):
```
(e12, x_1^3, -P)  (e12, x_2^3, +P)
(e13, x_1^3, -P)  (e13, x_3^3, +P)
(e23, x_2^3, -P)  (e23, x_3^3, +P)
```
The coordination sub-block alone is exactly `sheaf(I, J, V)` with `V = P`. The
dynamics rows are assembled directly via `blocksparse(CI, CJ, CV)`.

### Objective

`c`: `𝟏` on all `u^±` blocks, `0` elsewhere. `Q`: all-zero block-diagonal
(`allocblockdiag(B)` then leave blocks zero) — or set `Q_{x_i^3} = ε I₄` for the
regularized variant.

---

## 8. Assembly checklist (mapping to `IPMProblem`)

1. **Enumerate column-blocks** in natural order; record `(dim, cone_symbol)` for
   each. Build `cones::Vector{Symbol}`.
2. **Enumerate row-blocks**; record dims.
3. **Emit `B`** as `(row_block_id, col_block_id, block_matrix)` triples and build
   the `BlockSparseMatrix` (via `blocksparse` for the coboundary part; assemble
   the dynamics rows directly).
4. **`c`** — length `ncols(B)`; fill linear costs (`𝟏` on `ℓ₁`/epigraph blocks).
5. **`g`** — length `nrows(B)`; init conditions, box RHS, coordination `b`.
6. **`Q`** — `allocblockdiag(B)`; populate diagonal blocks only (QP curvature /
   ε-regularization). Leave zero for `:POS`/`:SOC`-only objectives.
7. **Construct** `IPMProblem(c, g, B, Q, cones)` and call
   `solve(prob, settings)` / `init` + `step!`. `init` permutes; `solve!`
   unpermutes (`p = P \ s.p`).

`initp` seeds the interior via `identity!` per cone (`:POS → 𝟏`, `:SOC → e₁`,
`:SDP → I`, `:NOC → 0`), so a feasible-interior start is automatic.

---

## 9. Scaling knobs and conditioning

| knob | what grows | KKT effect |
|---|---|---|
| `N` on `K_N` | terminal vertices form an `N`-clique | dense Schur complement, `O(N³)` fill — stresses `weightedgraph→linegraph→symbolic` and the Uzawa/ADMM solve |
| `N` on sparse `G` (cycle, grid) | sparse terminal coupling | chordal-friendly, low fill |
| `T` | per-agent time band (a path → tree) | linear growth, **no** fill penalty |
| `m` (control / cone dim) | block sizes | stresses per-cone kernels (`socscale!`, `skron!`, etc.) |

So `N`-on-`K_N` and `T` probe different parts of the machinery and make a clean
pair of benchmark axes. The augmentation strength is
`α = aaug + raug·‖A‖` (ADMM) or `α = aaug + raug·‖A‖/‖B‖²` (Uzawa), set in
`init_kkt!`.

---

## 10. Pitfalls

- **`Q` off-diagonal blocks** silently break Newton (§0). Keep all cross-block
  curvature in `B`.
- **Realizability of `b`.** The coboundary reduction (§1) needs `b ∈ im δ`.
  Consensus (`b = 0`) and consistent formations qualify; an arbitrary displacement
  vector that doesn't close around cycles does not, and then you genuinely need
  the Laplacian-stationarity form or a lifted edge variable.
- **Bare `:NOC` with `Q = 0`** relies on the augmentation for curvature. Fine if
  constraints pin the block, but prefer a small ε-regularization for
  conditioning.
- **POS split** is only valid because the `ℓ₁` objective drives
  `min(u⁺, u⁻) = 0` at optimum; the box row bounds `u⁺ + u⁻` regardless, so
  feasibility already implies `|u| ≤ ū`.
- **Sign convention.** `sheaf.jl` uses `-` for `i < j`, `+` otherwise
  (`B = -δ`). Immaterial for `b = 0`; match `g` for formation.

---

## 11. Beyond QP/POS/SOC (pointer)

The `:SDP` cone (`sdp.jl`: `svec`/`smat` with `√2` off-diagonal scaling,
symmetric Kronecker `skron!`, cache `3d² + d`) is the cone none of the paper's
examples exercise. The natural lift is **covariance steering**: replace an
agent's vector stalk with a PSD matrix `Σ_t ⪰ 0`, the recursion
`Σ_{t+1} = A_t Σ_t A_t' + B_t Θ_t B_t'` is affine in the decision matrices, the
cost `Σ_t tr Θ_t` is linear, and a terminal *inequality* `Σ_T ⪯ Σ_f` keeps it
convex (the equality version is nonconvex — stay with `⪯`). Consensus *in the PSD
cone* (agents agreeing on an uncertainty ellipsoid) lifts the same idea to the
coordination layer. Same template, same assembly checklist — only the stalk type
and cone change.

---

## 12. Testing the reformulations: a three-backend oracle

The cleanest way to validate a recipe is to solve the *same* problem three ways
and compare. The three legs are diagnostic — each disagreement localizes a
different bug.

### The three legs

| run | what it is | a discrepancy means |
|---|---|---|
| **H** | high-level convex model in JuMP, commercial solver | (reference) |
| **R** | reformulated conic model in JuMP, commercial solver | — |
| **S** | reformulated model → your `IPMProblem` triple → IPM | — |
| **H vs R** | same solver, two formulations | bug in the **reformulation algebra** (sign, missing `√2`, coboundary orientation, a box that doesn't imply `\|u\|≤ū`). Pure modeling check — independent of your solver. **Run first.** |
| **R vs S** | same problem, two solvers | given H≈R passed, bug is in **your solver** (KKT assembly, a cone's `hess!`/`scale!`, the augmentation). Your regression oracle. |
| **H vs S** | end to end | the claim you care about, but **uninformative alone** — only meaningful as the conjunction of the other two. |

The legs work because R and S are literally the same optimization problem as H:
they share the optimal objective, and the lifted variables are determined
functions of the originals (`u⁺ - u⁻`, the epigraph `s`, …), so the
natural-variable optimum agrees too.

### What to compare, and how

- **Objective value is the primary signal** — tolerance-robust and
  solver-agnostic. Assert agreement to a tolerance; report per leg.
- **Natural-variable solutions only secondarily**, and only when the optimum is
  *unique*. The min-fuel `ℓ₁` objective is the cautionary case: LP-type, frequent
  ties, so `u★` can differ between solvers at identical objective. Use the
  **ε-regularized `Q`** variant (§4) for solution-level comparison — it makes the
  optimum unique, so the regularization knob doubles as a test-harness knob.

### Conventions at the seam (translation layer = code that can be wrong)

Commercial conic solvers and JuMP's `MOI` bridges use **standard** SOC scaling
and **un-scaled** PSD vectorization; your stack uses the `√2`-weighted `svec`
(`sdp.jl`). The same mathematical constraint therefore needs its coefficients in
*your* convention before reaching the IPM. Unit-test that translation
independently: round-trip `svec`/`smat` and check `⟨svec(X), svec(Y)⟩ = ⟨X, Y⟩`.

### Architecture

One builder emits the abstract problem (graph, sheaf, per-agent `f_i`,
potentials) into **three backends**: a high-level JuMP model, a reformulated JuMP
model, and your `IPMProblem` triple `(c, g, B, Q, cones)`. Have the reformulated
JuMP model and the IPM consume the **same block structure** (§7/§8) — one
assembling it as JuMP constraints, the other as `blocksparse` — so a discrepancy
in the assembly can't hide behind two different reformulations.

**Fork — where R's conic data comes from:**

- *Extract from JuMP* via `lp_matrix_data` / `MOI` `ListOfConstraints` +
  `constraint_object`. Less code, but you then map MOI's cone tags and ordering
  onto your column-block/`cones` layout — a second translation layer to debug.
- *Hand-assemble independently.* More work, but R and S can't share a bug —
  the stronger oracle. Preferred for a *test* harness.

### Driver sketch

```
for instance in suite:                # (N, T, graph, mode) tuples
    H = solve_highlevel(instance)      # JuMP + commercial
    R = solve_reformulated(instance)   # JuMP + commercial, conic
    S = solve_ipm(instance)            # IPMProblem + your IPM
    assert isapprox(obj(H), obj(R); rtol)   # reformulation algebra
    assert isapprox(obj(R), obj(S); rtol)   # your solver
    if unique_optimum(instance):            # e.g. ε-regularized
        assert isapprox(sol_natural(R), sol_natural(S); atol)
    report(instance, H, R, S)
```

Scale the same `(N, T, graph)` knobs from §9 across the suite so the oracle also
charts where the IPM's accuracy or iteration count degrades relative to the
commercial baseline.
