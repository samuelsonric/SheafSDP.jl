# Large-stalk stress instances: POS and SOC

A standalone companion to `conic-recipes.md`. The earlier examples all have tiny
stalks (state `∈ ℝ⁴`, control `∈ ℝ²`, edge `∈ ℝ²`), so the sheaf Laplacian
`L_F = δ'δ` is a block matrix in name only — its blocks are `4×4` and `2×2`. This
document builds instances with **vertex stalks ≥ 30** and **edge stalks ≥ 15**, so
`L_F` is a genuine block matrix and the block linear algebra (chordal Cholesky
fronts, `block(...)` operations, the per-cone kernels) runs at realistic size.

References `§N` point at `conic-recipes.md`.

---

## 1. The lever: edge-stalk dimension lives in the restriction map's codomain

In the paper's vehicle examples the restriction maps are *projections to position*
(`P = [I₂ | 0₂]`), so edge stalks are `ℝ²`/`ℝ³`. To get a large edge stalk you
need restriction maps with a **high-dimensional codomain**:

```
F_{i⊴e} : ℝ^{n_v} → ℝ^{d_e},      a genuine  d_e × n_v  block,   d_e ≥ 15.
```

This is the **matrix-weighted / general sheaf** setting (Table I, row 2), not the
constant sheaf. The resulting Laplacian blocks are then full-sized:

```
diagonal   (L_F)_ii = Σ_{e∋i} F_{i⊴e}' F_{i⊴e}      (n_v × n_v)
off-diag   (L_F)_ij = − F_{i⊴e}' F_{j⊴e}            (n_v × n_v,  rank ≤ d_e)
```

With `n_v ≥ 30` these are `30×30`+ blocks; the off-diagonals are low-rank
(`rank ≤ d_e`) but still dense `n_v × n_v`. `L_F = B'B` is exactly what the Uzawa
path forms and factorizes (`copyto!(L, B'B)` in `make_kkt`, `F = H + αB'B` in
`init_uzw!`), and what the ADMM inner CG applies — so big stalks directly enlarge
the chordal fronts (`FChordalTriangular`, `weightedgraph → linegraph → symbolic`)
and the per-vertex `cholblockdiag!`.

---

## 2. A clean static chassis (no dynamics)

For stalk-size stress, drop the time expansion — it only clutters the block
sizes. Use the dynamics-free distributed optimization that generalizes the
paper's Example 3:

```
min   Σ_i f_i(x_i)
s.t.  δ_F x = b           (hard sheaf consensus, b ∈ im δ; §1 reduction)
      x_i ∈ K_i
```

Vertex stalk `F(i) = ℝ^{n_v}` is the agent's decision vector directly; edge stalk
`F(e) = ℝ^{d_e}` is the shared interface the agents must agree on through the
restriction maps. `f_i` and `K_i` pick the cone. `b = 0` for consensus; for a
nonzero interface target, generate `b = δ x₀` from a random `x₀` so realizability
is guaranteed by construction (§10 pitfall in the recipes).

---

## 3. Instance A — large-stalk POS (distributed box LP)

Each agent allocates a nonnegative vector subject to a box, minimizing a linear
cost, coupled by a high-dimensional consensus interface.

**Parameters (concrete).** `N = 6` on the path `P₆` (edges `12,23,34,45,56`).
Heterogeneous stalks: `n_v = 48` for agents `{1,3,5}`, `n_v = 30` for `{2,4,6}`
(all ≥ 30). Edge stalks `d_e = 16` (≥ 15). Restriction maps `F_{i⊴e} ∈ ℝ^{16×n_v}`
with **orthonormal rows** (`F F' = I₁₆`), so each contributes a rank-16
projection to the Laplacian and the diagonal blocks stay well-conditioned.

**Modeling object.** `f_i(x_i) = c_i' x_i`, with `0 ≤ x_i ≤ u_i`.

**Reformulation (recipe §5 box machinery, at scale).**

| column-block | dim | cone |
|---|---|---|
| `x_i` | `n_v` (30 or 48) | `:POS` |
| `w_i` | `n_v` | `:POS` |

| row-block | dim | equation | `g` |
|---|---|---|---|
| `box_i` | `n_v` | `x_i + w_i = u_i` | `u_i` |
| `coord_e` | `16` | `F_{i⊴e} x_i − F_{j⊴e} x_j = b_e` | `b_e` (0 for consensus) |

- `c`: `c_i` on `x_i`, `0` on `w_i`. `Q`: zero (the `:POS` barrier supplies all
  curvature, `poshess!` gives `d/p`).
- The coordination sub-block of `B` is `sheaf(I, J, V)` with `V = F_{i⊴e}` — now a
  `16×n_v` block instead of a `2×4` projection.

**Block structure that gets exercised.** The diagonal block of `B'B` at agent `i`
is `I_{n_v}` (from `box_i`, on `x_i` and `w_i`) plus `Σ_{e∋i} F_{i⊴e}'F_{i⊴e}`
(from coordination) — a dense `n_v × n_v` PD block. Off-diagonals are the
rank-16 `−F_{i⊴e}'F_{j⊴e}`. So `F = H + αB'B` carries `30×30`–`48×48` fronts
through the chordal Cholesky.

**Why POS makes the barrier dimension large too.** `degree(::POS, n) = n`, so
`conedegree` gives `ν = 2 Σ_i n_v ≈ 2·N·n̄` — a large complementarity dimension.
Big `:POS` stalks stress *both* the block factorization and the barrier
(`posmaxstep`'s ratio test now runs over `n_v ≥ 30` coordinates per block).

---

## 4. Instance B — large-stalk SOC (distributed least-norm)

Same chassis, but the node objective is an un-squared norm, producing one **large
second-order cone per agent** with arm `≥ 30`.

**Parameters.** Same graph and stalks as Instance A (`n_v ∈ {30,48}`, `d_e = 16`,
orthonormal-row restriction maps). Interface target `b = δ x₀` (realizable).

**Modeling object.** `f_i(x_i) = ‖x_i‖₂`, i.e. the least-2-norm cochain
consistent with the interface targets `b_e`.

**Reformulation (recipe §6 epigraph, at scale).**

| column-block | dim | cone |
|---|---|---|
| `ζ_i = (s_i; x_i)` | `1 + n_v` (31 or 49) | `:SOC` |

| row-block | dim | equation | `g` |
|---|---|---|---|
| `coord_e` | `16` | `F_{i⊴e} x_i − F_{j⊴e} x_j = b_e` | `b_e` |

- The coordination row reads the **tail** of `ζ_i`: its block is `[0 | F_{i⊴e}]`
  of size `16 × (1+n_v)`, zero on the `s` head.
- `c`: `1` on each head `s_i`, `0` on the tail. `Q`: zero (the `:SOC` barrier,
  `sochess!`/`socscale!`, supplies curvature on the whole `(1+n_v)` block).

**What gets exercised — and how it differs from POS.** The cone is `Q^{1+n_v}`
with arm `n_v ≥ 30`, so `socscale!`, `sochess!`, and `socmaxstep` (the quadratic
ratio test) all run on long vectors. But `degree(::SOC, n) = 2` regardless of
arm, so `conedegree` gives only `ν = 2N` — a *tiny* barrier dimension. This is the
clean contrast:

- **POS (Instance A):** big stalk ⇒ big `ν` ⇒ stresses iteration-count /
  complementarity *and* block size.
- **SOC (Instance B):** big arm ⇒ `ν` stays at `2N` ⇒ stresses **per-iteration
  cone linear algebra** on long arms, not iteration count.

Running both at the same `n_v` isolates those two cost drivers against each other.

---

## 5. Extending §6: the aggregate energy budget (large arm via the horizon)

For the *dynamic* chassis, the small-cone SOC example (§6, per-timestep
`(s_i^t; u_i^t) ∈ Q^{1+m}`) only ever produces `Q^3` blocks — the `n = T`
many-small-cones regime. To reach a large arm there, replace the per-timestep
norm with a per-agent **total energy budget**:

```
‖ vec(u_i^1, …, u_i^{T-1}) ‖₂ ≤ E_i
```

a single `:SOC` block of dimension `1 + m(T−1)`:

| column-block | dim | cone |
|---|---|---|
| `η_i = (E_i ; u_i^1; …; u_i^{T-1})` | `1 + m(T−1)` | `:SOC` |

with `E_i` either a fixed head (a hard budget) or a free epigraph minimized in
`c`. Each dynamics row `x_i^{t+1} − A x_i^t − B u_i^t = 0` reads its own
timestep's sub-vector out of the shared tail. Now the arm grows with `T` (or `m`),
so this is the `n = m` long-arm knob *inside* the time-expanded problem — the same
`socscale!`/`socmaxstep` stress as Instance B, but coupled to dynamics.

This gives the SOC recipe coverage of both cone regimes:

| variant | cone(s) | arm | knob (§9) |
|---|---|---|---|
| per-timestep effort (§6) | many `Q^{1+m}` | small (2–4) | `n = T` |
| terminal-ball / slew (§6) | `Q^{1+m}` | small | — |
| energy budget (§5 here) | one `Q^{1+m(T-1)}` | large via `T`,`m` | `n = m` |
| least-norm (§4 here) | one `Q^{1+n_v}` | large via `n_v` | `n = m` |

---

## 6. Designing the restriction maps

The restriction maps are the whole game for stalk size and conditioning.

- **Codomain sets `d_e`.** Pick `d_e ≥ 15` and `d_e ≤ n_v` (a restriction map into
  a space larger than the stalk is rank-deficient and couples nothing extra).
- **Orthonormal rows** (`F_{i⊴e} F_{i⊴e}' = I_{d_e}`) make each edge contribute a
  clean rank-`d_e` projection `F'F` to the Laplacian diagonal — well-conditioned,
  easy to reason about. Draw a random `n_v × d_e` matrix and take its thin `Q`
  factor.
- **Full-rank diagonal blocks.** A single edge contributes rank `d_e < n_v`, so an
  agent needs enough incident edges (or a box/identity row) for its Laplacian
  diagonal to be full rank. The box row in Instance A guarantees this; the SOC
  barrier guarantees it in Instance B. Without either, lean on the `αB'B`
  augmentation (the bare `:NOC` case, §3/§4 of the recipes).
- **Heterogeneity.** Mixing `n_v ∈ {30, 48}` and varying `d_e` per edge is the
  point — it produces non-uniform fronts and exercises the symbolic factorization's
  handling of unequal block sizes, which uniform tiny stalks never do.

For nonzero interface targets, build `b = δ x₀` from a random `x₀ ∈ C⁰` so
`b ∈ im δ` and the hard constraint stays feasible (§1).

---

## 7. Scaling knobs (extends §9)

| knob | what grows | what it stresses |
|---|---|---|
| `n_v` | vertex stalk | chordal front size, `block(...)` ops, `cholblockdiag!`; for POS also `ν` and `posmaxstep`; for SOC the cone-kernel arm |
| `d_e` | edge stalk | rank of Laplacian off-diagonals, coordination row height, the `F'F`/`F_i'F_j` products |
| `N` | agents | number of blocks / fronts |
| graph `G` | coupling topology | fill: path/tree → none; cycle → one chord; clique → dense Schur |

For a pure block-size test, scale `n_v` on a **path** (zero fill, so only block
size moves). To bring in fill at large block size, switch the graph to a cycle or
clique and hold `n_v` fixed. The two knobs are separable, which makes them a clean
grid for profiling.

---

## 8. What to verify (ties to §12)

These instances are also good oracle cases because the block sizes make
convention bugs visible:

- The `:SOC` and (eventual) `:SDP` legs route through your `√2`-weighted `svec`
  (`sdp.jl`); at arm `≥ 30` a missing scale factor shifts the objective by a
  clearly measurable amount, unlike at arm 2. Keep the round-trip
  `⟨svec(X), svec(Y)⟩ = ⟨X, Y⟩` unit test.
- Compare against Clarabel (native quadratic objective, chordal SDP, IPM →
  tight tolerance) and Mosek; the large-stalk POS/SOC problems are squarely in
  the regime both handle well, so disagreement localizes cleanly to your solver
  per §12.
- With orthonormal-row restriction maps and `b = δx₀`, the least-norm SOC
  instance has a **unique** optimum (strictly convex on the feasible affine
  space), so solution-level comparison — not just objective — is meaningful.
