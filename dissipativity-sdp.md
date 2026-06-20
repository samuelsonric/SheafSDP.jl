# Compositional dissipativity: an SDP that starts graph-shaped

The SDP companion to the large-stalk POS/SOC/QP/elastic-net docs. The point of
this one is to get the `:SDP` cone into the framework **without** a chordal
decomposition — the problem is graph-shaped from the start (one PSD-valued
certificate per subsystem, coupled along the communication/interconnection graph),
not a monolithic SDP someone sliced into cliques. The distinction matters because
a clique-tree sheaf's restriction maps are 0/1 selections (indexing), whereas this
one's are **congruences** — dense `svec` operators (matmul) — which is the path
worth testing.

References: `conic-recipes.md` §0/§3/§12, `large-stalk-instances.md` (chassis,
knobs), `sdp.jl` (`svec`/`smat`, `skron!`).

---

## 1. The control backbone

Each subsystem `i` is `ẋᵢ = Aᵢxᵢ + Bᵢuᵢ`, `yᵢ = Cᵢxᵢ + Dᵢuᵢ`, with
`xᵢ ∈ ℝ^{nᵢ}`, `uᵢ ∈ ℝ^{mᵢ}`, `yᵢ ∈ ℝ^{pᵢ}`. Dissipativity w.r.t. a quadratic
supply rate is certified by a storage function `V(x) = x'Px`, `P ≻ 0`, satisfying
a KYP/dissipation LMI (Moylan–Hill; Arcak–Meissen–Packard for the compositional
version). Networks of dissipative subsystems certify **compositionally** when the
interconnection is power-continuous — which is the regime this example lives in,
and the one your group's compositional framing is built around.

That power-continuous interconnection is also the answer to "where does the
node-to-node coupling come from." With supply rates fixed, the textbook
decentralized certificate is a *purely local* LMI in each `Pᵢ` — the `Pᵢ` never
see each other and the sheaf is trivial. The genuine coupling appears when
subsystems **share a physical interface** and must agree on the energy stored
there. That agreement is the load-bearing modeling step, and it has one subtlety
that decides whether the example is honest.

---

## 2. The load-bearing step: agree on compliance, not storage

Let edge `e = ij` carry a shared interface with boundary DOFs `ξₑ ∈ ℝ^{dₑ}`,
reached from subsystem `i`'s coordinates by `ξₑ = Cₑ⁽ⁱ⁾ xᵢ`, with
`Cₑ⁽ⁱ⁾ ∈ ℝ^{dₑ×nᵢ}` the (dense, frame-dependent) interface map.

The tempting condition — agree on **interface storage** — is *not* linear. The
energy attributable to a boundary motion `ξ` is the constrained minimum

```
min_{x : Cₑx = ξ}  x'Pᵢx  =  ξ' (Cₑ Pᵢ⁻¹ Cₑ')⁻¹ ξ,
```

so storage agreement is `(Cₑ⁽ⁱ⁾Pᵢ⁻¹Cₑ⁽ⁱ⁾')⁻¹ = (Cₑ⁽ʲ⁾Pⱼ⁻¹Cₑ⁽ʲ⁾')⁻¹` — a Schur
complement, **nonlinear in `Pᵢ`**. Carried on the storage matrix, the restriction
map is harmonic and the coboundary structure dies. That is exactly the failure
mode that would make this fake.

The fix is to carry the **compliance / admittance** `Gᵢ := Pᵢ⁻¹ ≻ 0` as the node
SDP variable. Compliance marginalizes by **congruence**:

```
   Cₑ⁽ⁱ⁾ Gᵢ Cₑ⁽ⁱ⁾'  =  Cₑ⁽ʲ⁾ Gⱼ Cₑ⁽ʲ⁾'        ∈ 𝕊^{dₑ}
```

— linear in `Gᵢ`, dense, and physically it is **interface admittance matching**:
the two substructures induce the same force–displacement relation at the shared
port. This is `δ_F G = 0` with congruence restriction maps, and it is the same
sheaf object the covariance-consensus sketch produced — the structure is *forced*,
not chosen, which is the reassuring sign.

Switching to `G = P⁻¹` linearizes the node LMI too (congruence by
`diag(G, I, …)`), so **every block of `B` becomes a dense `svec`-linear map** —
Lyapunov operators on nodes, congruences on edges, indexing nowhere. That is the
maximal exercise of your `svec` matmul path, which is what makes this the SDP
stress instance.

---

## 3. Variant A — passivity (the clean, small case)

Supply rate `u'y` (so `mᵢ = pᵢ`). The dual passivity LMI in `Gᵢ` (congruence of
the standard KYP form by `diag(Gᵢ, I)`):

```
            ⎡ AᵢGᵢ + GᵢAᵢ'    Bᵢ − GᵢCᵢ' ⎤
𝒟ᵢ(Gᵢ)  =  ⎢                            ⎥  ⪯ 0 ,     on 𝕊^{nᵢ+mᵢ}.
            ⎣ Bᵢ' − CᵢGᵢ     −(Dᵢ + Dᵢ') ⎦
```

Linear in `Gᵢ`. Introduce slack `Sᵢ := −𝒟ᵢ(Gᵢ) ⪰ 0` and the constraint is the
equality `Sᵢ + 𝒟ᵢ(Gᵢ) = 0`.

**Objective — named honestly.** Passivity certification is a **feasibility**
problem; any valid `{Gᵢ}` certifies. To get a well-posed *optimization* test, add
the standard "smallest certificate" selection `min Σᵢ tr(Gᵢ)` (here in compliance
coordinates). This is a selection, not a physical objective — say so in the doc.
In your `√2`-weighted `svec`, `tr(G) = ⟨svec(I), svec(G)⟩`, so `c = svec(Iₙᵢ)` on
each `Gᵢ`, `Q = 0`.

---

## 4. Variant B — L₂-gain (the showcase, honest objective)

Here the objective stops being a selection. Certifying `‖·‖∞ ≤ γ` and minimizing
`γ` is *literally the quantity you want* — the best certifiable gain — not a
surrogate for an off-cone objective. The dual bounded-real LMI, linear in
`(Gᵢ, γᵢ)`:

```
              ⎡ AᵢGᵢ + GᵢAᵢ'   Bᵢ      GᵢCᵢ' ⎤
𝒟ᵢ(Gᵢ,γᵢ) =  ⎢ Bᵢ'           −γᵢ I    Dᵢ'   ⎥  ⪯ 0 ,   on 𝕊^{nᵢ+mᵢ+pᵢ}.
              ⎣ CᵢGᵢ           Dᵢ     −γᵢ I  ⎦
```

(`γᵢ`, not `γᵢ²` — the objective stays linear. Check the block placement against
your BRL reference; the shape and the linearity in `(Gᵢ, γᵢ)` are the
load-bearing facts.) Slack `Sᵢ := −𝒟ᵢ(Gᵢ,γᵢ) ⪰ 0`, on the larger `𝕊^{nᵢ+mᵢ+pᵢ}`
— the input *and* output channels, so `Sᵢ` is bigger than the passivity slack,
which is the block-size heterogeneity that stresses the symbolic factorization.

**Keep γ a consensus variable, not a global scalar.** A single shared `γ` enters
every node LMI, making it one apex column-block adjacent to all `Gᵢ` — a star,
not a clique, but still an apex that must be eliminated *last* or it densifies the
Schur complement. The sheaf-native fix is a local copy `γᵢ` (`:POS` scalar, the
constant sheaf `ℝ` over `G`) with `δγ = 0`, i.e. `γᵢ = γⱼ` on edges. Each node LMI
then touches only its own `(Gᵢ, γᵢ)`, the agreement rides the same graph edges as
everything else, and the apex is gone — the consensus mechanism applied to the
performance level, a feature rather than a wart. (`min Σ tr Gᵢ` sidesteps it
differently — fully separable, no shared variable — but it's the selection
objective, not the honest gain.)

---

## 5. Assembly and the small-stalk instance

`N = 3` on the path `P₃` (edges `12, 23`). SISO subsystems: `nᵢ = 3` modal
coordinates (`Gᵢ ∈ 𝕊³`, svec 6), `mᵢ = pᵢ = 1`, interface `dₑ = 2` (`𝕊²`, svec 3,
`Cₑ⁽ⁱ⁾ ∈ ℝ^{2×3}` dense mode shapes — rotation-into-interface-frame ∘ projection).

**Passivity** — per node `i`:

| column-block | space | svec dim | cone |
|---|---|---|---|
| `Gᵢ` | `𝕊³` | 6 | `:SDP` |
| `Sᵢ` | `𝕊⁴` | 10 | `:SDP` |

| row-block | svec dim | equation | `g` |
|---|---|---|---|
| `dissᵢ` (private) | 10 | `Sᵢ + ℒᵢ(Gᵢ) = −svec(𝒟₀⁽ⁱ⁾)` | `−svec(𝒟₀⁽ⁱ⁾)` |
| `agreeₑ` (coord) | 3 | `(Cₑ⁽ⁱ⁾⊗ₛCₑ⁽ⁱ⁾) svec Gᵢ − (Cₑ⁽ʲ⁾⊗ₛCₑ⁽ʲ⁾) svec Gⱼ = 0` | `0` |

Here `ℒᵢ` is the linear-in-`G` part of `𝒟ᵢ` (a dense `10×6` svec operator), and
`𝒟₀⁽ⁱ⁾` is the constant part (`Bᵢ` off-diagonal, `−(Dᵢ+Dᵢ')` corner). `c = svec(I₃)`
on each `Gᵢ`, `Q = 0`. Cones, natural order: `[:SDP,:SDP]` per node.

**L₂-gain** — add `γᵢ` (`:POS` scalar) per node, enlarge `Sᵢ` to `𝕊⁵` (svec 15),
and add the consensus rows:

| extra column | space | cone | extra row | dim | equation |
|---|---|---|---|---|---|
| `γᵢ` | `ℝ` | `:POS` | `γconsₑ` (coord) | 1 | `γᵢ − γⱼ = 0` |

`dissᵢ` is now svec 15 and also touches `γᵢ` (the `−γᵢI` blocks, linear in `γ`).
`c = 1` on each `γᵢ`, else 0; `Q = 0`. Cones per node `[:SDP,:SDP,:POS]`.

---

## 6. Large-stalk instance

The physical scale-up is a **substructured flexible system** (component mode
synthesis): each subsystem is a flexible component with many retained modes, the
interfaces are shared boundaries with many coupling DOFs, and you certify the
assembled structure's passivity / gain from component certificates that match
interface admittance. This is a real engineering setting, not a blow-up of the toy.

- `nᵢ ≥ 30` modes ⇒ `Gᵢ ∈ 𝕊³⁰`, svec dim `C(31,2) = 465` — a large `:SDP` block.
- interface `dₑ ≥ 15` ⇒ agreement on `𝕊¹⁵`, svec dim 120; `Cₑ⁽ⁱ⁾ ∈ ℝ^{15×30}` dense.
- MIMO `mᵢ = pᵢ = 5` ⇒ `Sᵢ ∈ 𝕊^{40}`, svec dim 820.

What it stresses, by component:

| object | size | code path |
|---|---|---|
| SDP barrier Hessian on `Gᵢ` | `465 × 465` dense per node | `skron!` (`W⁻¹⊗ₛW⁻¹`), `sdphess!` — `O(n⁴)` per block |
| congruence coboundary block | `120 × 465` dense | the rectangular `svec`-Kronecker (§7) |
| `L_F = B'B` off-diagonals | `465 × 465`, rank ≤ 120 | chordal fronts, `FChordalTriangular` |
| node LMI slack `Sᵢ` | `𝕊^{40}`, svec 820 | larger-block `:SDP` arithmetic, heterogeneous fronts |

`degree(::SDP, n)` is the matrix dimension, so `conedegree` gives a large `ν`
(≈ `nᵢ + (nᵢ+mᵢ+pᵢ)` per node ≈ 70), i.e. big barrier *and* heavy per-iteration
block work — the SDP analogue of the large-stalk POS instance, but with dense
`svec` operators in every block instead of selections.

---

## 7. Implementation note: the rectangular `svec`-congruence builder

The congruence `G ↦ CGC'` with `C ∈ ℝ^{dₑ×nᵢ}` is, in `svec` coordinates, the
**rectangular** symmetric Kronecker `C ⊗ₛ C`, a `C(dₑ+1,2) × C(nᵢ+1,2)` dense
block (`120 × 465` at large stalk). Your `skron!` (`sdp.jl`) is the **square**
special case `dₑ = nᵢ`, `C = A`: it builds `A ⊗ₛ A` for the node Hessian. The
agreement blocks need the rectangular cousin — rows indexed by symmetric pairs in
`[dₑ]`, columns by symmetric pairs in `[nᵢ]`, entries the symmetrized products
`C[a,i]·C[b,j]` with the same `√2` off-diagonal weighting as `skron!`. Two
consequences worth writing down:

- It is a **construction-time** builder, not a hot-loop kernel: the sheaf is
  fixed, so each block is built once and handed to the IPM as a plain dense block.
  The cone's hot loop keeps using square `skron!`.
- The two endpoints of an edge use **different** maps from **different-dimensional**
  stalks (`Cₑ⁽ⁱ⁾` vs `Cₑ⁽ʲ⁾`, sizes `dₑ×nᵢ` vs `dₑ×nⱼ`), so unlike `sheaf.jl`'s
  constant-sheaf path you cannot reuse one block with a sign flip — build each
  endpoint independently. This asymmetry is the concrete face of "this sheaf wants
  different handling."

Because both the node stalk and the edge stalk use the same `√2`-weighted `svec`,
`⟨svec A, svec B⟩ = ⟨A,B⟩_F` on both ends and the congruence builder is
scaling-clean — but that is precisely the place a missing factor hides, so unit-
test `svec(CGC') == (C ⊗ₛ C) svec(G)` directly (§12).

---

## 8. Is it fake?

No — conditional on the power-continuous (port-Hamiltonian) interconnection, which
is the compositional regime, not a dodge. The two things that would have made it
fake are both avoided: the objective is honest (`min Σγᵢ` is the genuine best
certifiable gain; passivity's trace selection is labeled as a selection, not a
physical claim), and the restriction map is **linear** because we carry compliance
`G = P⁻¹` rather than storage `P`. It starts graph-shaped — independent component
certificates coupled by an imposed interface-agreement constraint — so it passes
the "not a decomposed monolith" test that chordal SDP fails. Its honest limitation
is that it shares the covariance example's *homological skeleton* (PSD nodes
agreeing on congruence-restricted marginals); dissipativity buys a real objective
and a real story on top of that skeleton, not a structurally new linear-algebra
path.

For the oracle (§12): both variants go to Mosek and Clarabel (native SDP; Clarabel
also takes the `:POS` `γ` and any quadratic regularizer directly). `min Σγᵢ` with
a connected graph has a unique gain, so objective-level comparison is sharp; for
solution-level, add a small `ε Σ tr(Gᵢ²)` (a `Q = εI` on the `Gᵢ` svec blocks) to
make the compliance unique. The `√2` convention shows up in *both* the node LMI
operators and the congruence blocks, so the large-stalk version — svec dim 465 —
is where a scaling bug is most visible.
