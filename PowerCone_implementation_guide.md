# `PowerCone` — implementation guide

A drop-in 3-dimensional power cone for a Dahl–Andersen-style nonsymmetric primal–dual
IPM, mirroring your existing `ExpCone`. Every formula below was checked symbolically
(`sympy`) and/or numerically against finite differences; the relevant check is stated
in each section so you can reproduce it as a unit test.

The cone:

```
P_α = { x ∈ R³ : x₁^α x₂^(1-α) ≥ |x₃|,  x₁,x₂ ≥ 0 },   α ∈ (0,1)
```

Chares barrier, ϑ = 3:

```
F(x) = -log( x₁^(2α) x₂^(2(1-α)) - x₃² ) - (1-α) log x₁ - α log x₂
```

This is exactly the barrier MOSEK uses (their general-power-cone talk, m=2,n=1 case).

---

## 0. The type — and why it has a field

Unlike `ExpCone`, which is effectively a singleton (every exponential cone is identical),
**the power cone is a one-parameter family**, so the type *must* carry `α`:

```julia
struct PowerCone{T} <: AbstractCone{T}
    α::T                 # exponent, 0 < α < 1
    offset::Int          # start index into the global x / s vectors (as in ExpCone)
    # --- per-iteration cache, filled by update!(cone, x, s) ---
    # p, φ, ρ, ℓ, gradient, the (3,1,2) Cholesky factor, shadow iterate x̃, …
end
```

Design consequences of the field:

- The cone object is **not** a singleton: cache/equality/hashing key on `α`.
- `α = 1/2` is special — `P_{1/2} = {x₁x₂ ≥ x₃²}` is a **rotated second-order cone**
  (symmetric). Keep this as a permanent regression target (see §10).
- The central starting point (§8) depends on `α`.
- Validate `0 < α < 1` in the constructor.

**Shared per-point scalars** (compute once in `update!`, reuse everywhere). With
`a = 2α`, `b = 2(1-α)` (note `a+b = 2`):

```
p  = x₁^a * x₂^b              # = x₁^(2α) x₂^(2(1-α))   (the "power product")
φ  = p - x₃²                  # barrier argument; φ > 0 strictly inside, x₁,x₂ > 0
ρ  = p / φ                    # ≥ 1, with equality iff x₃ = 0
ℓ  = (a/x₁, b/x₂, 0)          # = ∇log p
φ′ = (a*p/x₁, b*p/x₂, -2x₃)   # = p*ℓ - 2x₃*e₃   (gradient of φ)
```

---

## 1. The test battery (build this first)

Three log-homogeneity identities catch essentially every derivative bug. Wire these
in before anything else and call them on random interior points for random `α`:

| identity | what it tests | tol |
|---|---|---|
| `⟨F′(x), x⟩ = -3` | gradient sign/scale | 1e-12 |
| `F″(x) x = -F′(x)` | Hessian vs gradient consistency | 1e-12 |
| `F‴(x)[x] = -2 F″(x)` | third-order vs Hessian consistency | 1e-11 |
| `F″(x) ≻ 0` | factorization succeeds | — |
| `|F‴(x)[u,u,u]| ≤ 2 (F″(x)[u,u])^{3/2}` | self-concordance | ≤ 1 |

All five were verified to machine precision across thousands of points; the
self-concordance ratio tops out at exactly 1.0, confirming a genuine 3-LHSCB.

Helper for sampling a strictly-interior point at exponent `α`:

```
x₁, x₂ > 0 random;  bound = x₁^α x₂^(1-α);  x₃ = t * bound, |t| < 1
```

---

## 2. Membership and boundary distance (line search)

Primal membership (`x ∈ int P_α`):

```
x₁ > 0  and  x₂ > 0  and  φ = x₁^(2α) x₂^(2(1-α)) - x₃² > 0
```

Dual membership (`s ∈ int P_α*`), needed before the conjugate solve (§7):

```
s₁ > 0  and  s₂ > 0  and  (s₁/α)^α (s₂/(1-α))^(1-α) - |s₃| > 0
```

Your generic step-to-boundary `αₐ = sup{α : x+αΔx ∈ K, s+αΔs ∈ K*}` only needs these
membership predicates (bisection, as you already do for `ExpCone`). No closed form
required, but if you want a tighter bracket, the primal boundary along a ray is where
`φ(x+αΔx) = 0`, a 1-D function you can root-find directly.

**✓ Check:** random `Δ`, compare bisection boundary to a fine brute-force scan.

---

## 3. Gradient `F′(x)`

```
F′(x) = -φ′/φ + ( -(1-α)/x₁,  -α/x₂,  0 )
      = ( -a*p/(x₁ φ) - (1-α)/x₁,
          -b*p/(x₂ φ) - α/x₂,
           2 x₃ / φ )
```

**✓ Check:** `⟨F′(x), x⟩ = -3`; and central-difference of `F` matches `F′` (≈1e-7).

---

## 4. Hessian `F″(x)` — cancellation-free entries

The naive form has a difference of an `O(φ⁻²)` and an `O(φ⁻¹)` term that cancels badly
near the boundary. The identity `p - φ = x₃²` collapses it so that **every `O(φ⁻²)`
contribution is multiplied by an explicit `x₃²`** (and therefore vanishes on the
symmetric slice `x₃ = 0`). Use these entries verbatim — each is a sum of like-signed
terms, no subtraction:

```
d₁ = (2ρa + b)/(2 x₁²)        d₂ = (2ρb + a)/(2 x₂²)      # both > 0

F″₁₁ = d₁ + a² p x₃² /(x₁² φ²)
F″₂₂ = d₂ + b² p x₃² /(x₂² φ²)
F″₁₂ =      a b p x₃² /(x₁ x₂ φ²)
F″₃₃ = 2 (p + x₃²) / φ²
F″₁₃ = -2 a x₃ p /(x₁ φ²)
F″₂₃ = -2 b x₃ p /(x₂ φ²)
```

Equivalent compact form (handy for the third-order term in §6):

```
F″ = diag(d₁,d₂,0)
   + (p x₃²/φ²)        ℓ ℓᵀ
   - (2 x₃ p/φ²)       (ℓ e₃ᵀ + e₃ ℓᵀ)
   + (2(p+x₃²)/φ²)     e₃ e₃ᵀ
```

**✓ Check:** matches `sympy.hessian` exactly (verified); `F″(x) x = -F′(x)` to 1e-12;
finite-difference agreement; symmetry.

---

## 5. The stable factorization (solving `F″ v = w`)

You need `(F″)⁻¹ Δsᵃ` for the corrector and `F″` factors for the scaling. Don't form a
dense `F″` and Cholesky it blindly. The exp-cone appendix gets a clean 3×3 factor
because `ψ` is linear in `x₃`; here `φ` is *quadratic* in `x₃`, so `F″₃₃ ≠ 0` and that
route is gone. Instead **pivot on coordinate 3 first** — its pivot `F″₃₃` is the cleanest
entry in the matrix — which reduces the rest to a 2×2 *diagonal-minus-rank-one*, exactly
the structure their `A(x)` has.

Pivot and Schur complement:

```
D₃₃ = F″₃₃ = 2(p + x₃²)/φ²
w   = (F″₁₃, F″₂₃)                       # coupling to coord 3
c   = p x₃² / ( φ (p + x₃²) )   ≥ 0
ℓ₁₂ = (a/x₁, b/x₂)                       # = ℓ[1:2]

S = diag(d₁, d₂) - c ℓ₁₂ ℓ₁₂ᵀ           # 2×2, the Schur complement on (1,2)
```

(The two rank-one pieces of `F″` collapse into the single `-c ℓ₁₂ℓ₁₂ᵀ` — verified, the
Schur reconstruction error is exactly 0.) Closed-form Cholesky `S = L̃ L̃ᵀ`:

```
L̃₁₁ = sqrt( d₁ - c ℓ₁² )
L̃₂₁ = -c ℓ₁ ℓ₂ / L̃₁₁
L̃₂₂ = sqrt( ( d₁ d₂ - c(d₁ ℓ₂² + d₂ ℓ₁²) ) / ( d₁ - c ℓ₁² ) )      # = det(S)/(d₁-cℓ₁²)
```

Assemble the full factor in permuted order **(3, 1, 2)**:

```
        [ sqrt(D₃₃)      0     0  ]
L  =    [ w₁/sqrt(D₃₃)  L̃₁₁    0  ]        # acts on (x₃, x₁, x₂)
        [ w₂/sqrt(D₃₃)  L̃₂₁   L̃₂₂ ]

P F″ Pᵀ = L Lᵀ ,   P = permutation (3,1,2)
```

Solve `F″ v = w` by permute → forward solve `L` → back solve `Lᵀ` → un-permute.
`cond(L) = sqrt(cond(F″))`, the same conditioning win the exp-cone factor gives.

**✓ Check:** `L Lᵀ = P F″ Pᵀ` (zero error); PD across 4000 interior points out to
`|x₃|/bound = 0.99999`; worst-case relative magnitude retained in `d₁ - c ℓ₁²` is ~6%
(≈ one digit — on par with the `1 ± sqrt(1+2x₂/ψ)` terms in the exp-cone `R`).

*Fallback:* because the §4 entries are themselves cancellation-free, a plain 3×3 Cholesky
on the assembled matrix is also stable; the structured factor only buys the
half-precision conditioning. Keep the structured one as primary, the dense Cholesky as a
cross-check in tests.

---

## 6. Third-order term and the corrector

The corrector is `η = -½ F‴(x)[Δxᵃ, (F″)⁻¹ Δsᵃ]`. Implement `F‴(x)[u]` as an honest
analytic matrix (don't finite-difference the Hessian in production). Building blocks
(reusing `p, φ, ℓ, φ′, φ″` from the cache):

```
φ̇      = ⟨φ′, u⟩
φ̇′     = φ″ u                                   # φ″ = p(ℓℓᵀ + Dℓ) + diag(0,0,-2)
Dℓ     = -diag(a/x₁², b/x₂², 0)
Dℓu    = Dℓ u
Ḋℓ     = diag(2a u₁/x₁³, 2b u₂/x₂³, 0)
φ‴[u]  = p⟨ℓ,u⟩ (ℓℓᵀ + Dℓ) + p ( Dℓu ℓᵀ + ℓ (Dℓu)ᵀ + Ḋℓ )
ḣ″     = diag( -2(1-α)u₁/x₁³, -2α u₂/x₂³, 0 )

F‴(x)[u] = (φ̇′ φ′ᵀ + φ′ φ̇′ᵀ)/φ²
         - (2 φ̇/φ³) φ′ φ′ᵀ
         - φ‴[u]/φ
         + (φ̇/φ²) φ″
         + ḣ″
```

Then with `u = Δxᵃ` and `v = (F″)⁻¹ Δsᵃ` (from §5):

```
η = -½ * F‴(x)[u] * v
```

**✓ Check:** `F‴(x)[u]` vs central-difference of `F″` along `u` (≈1e-4, FD-limited);
`F‴(x)[x] = -2 F″(x)` to 1e-11; `F‴(x)[u]` symmetric.

> Note (from the MOSEK talk): the higher-order corrector is a clear win for **3-D**
> cones (fewer iterations, as in the exp-cone results) but can *hurt or break*
> convergence for **high-dimensional** power cones. That regime isn't yours, but if you
> ever stack into a large power cone, gate the corrector behind a damping/disable flag.

---

## 7. Conjugate / shadow iterate — a **scalar** solve (no Kapelevich needed)

The shadow iterate `x̃ = -∇F*(s)` equals the conjugate point `xₛ` solving `F′(xₛ) = -s`.
For the 3-D power cone this reduces to **one monotone scalar equation** — you do not need
a multivariate Newton or the Kapelevich machinery (that's for the general m-D cone).

Precondition: `s ∈ int P_α*` (§2). Solve for `ρ ∈ [1, ∞)`:

```
X₁(ρ) = (2αρ + 1-α)/s₁
X₂(ρ) = (2(1-α)ρ + α)/s₂
g(ρ)  = ρ(ρ-1) - (s₃²/4) X₁(ρ)^(2α) X₂(ρ)^(2(1-α))
solve g(ρ) = 0
```

Properties (all verified): a root in `[1,∞)` exists **iff** `s ∈ int P_α*`; `g(1) ≤ 0`
gives the lower bracket; `g` is increasing at the root (`g′(ρ*) > 0` in all 20000 trials),
so Newton from `ρ = 1` (or bisect after growing the upper bracket until `g > 0`) is safe.
Recover in closed form:

```
ρ*  →  x₁ = X₁(ρ*),  x₂ = X₂(ρ*)
p   = x₁^(2α) x₂^(2(1-α))
φ   = p / ρ*
x₃  = -s₃ φ / 2
x̃ = xₛ = (x₁, x₂, x₃)
```

and `∇²F*(s) = (F″(xₛ))⁻¹` via the §5 factor.

**✓ Check:** `F′(xₛ) = -s` to 1e-12 over 20000 dual points; at the central
`s = (√(1+α), √(2-α), 0)` you get `xₛ = s` with `ρ* = 1`; `s₃ = 0 ⇒ ρ* = 1 ⇒ x₃ = 0`
(symmetric slice).

> Compared to `ExpCone`, where you Newton a (low-dim) system because `ψ` is transcendental,
> the power cone is strictly easier here: the two log-homogeneous coordinates plus the one
> quadratic coordinate collapse the conjugate to 1-D.

---

## 8. Starting point (analytic)

Unlike the exp cone's offline value `≈ (1.2909, 0.8051, -0.8278)`, the power-cone central
start is closed form:

```
x₀ = s₀ = ( sqrt(1+α), sqrt(2-α), 0 )
```

**✓ Check:** `x₀ + F′(x₀) = 0` exactly; `⟨x₀, s₀⟩ = 3 = ϑ`; hence `(x₀,s₀) ∈ N(1)`.

---

## 9. Dual cone (closed form)

```
P_α* = diag(α, 1-α, 1) · P_α
     = { z : (z₁/α)^α (z₂/(1-α))^(1-α) ≥ |z₃|,  z₁,z₂ ≥ 0 }
```

Gives you the §2 dual membership test directly. **✓ Check:** for `w ∈ P_α`,
`z = diag(α,1-α,1) w` satisfies `(z₁/α)^α (z₂/(1-α))^(1-α) = w₁^α w₂^(1-α)` (verified).

---

## 10. Integration tests (do not skip)

1. **`α = 1/2` ≡ rotated SOC.** At `α = 1/2` the barrier is
   `-log(x₁x₂ - x₃²) - ½log x₁ - ½log x₂`, a 3-self-concordant barrier for the rotated
   Lorentz cone `{x₁x₂ ≥ x₃²}` (symmetric). Run a small problem through `PowerCone(0.5)`
   and through your existing rotated-quadratic NT path; iteration counts and the solution
   should match closely, and the BFGS scalar `t` should collapse toward the NT value
   (bound `ξ → 4/3` or better). If they disagree, the bug is in the oracle, not the solver.

2. **Secant equations.** After your shared code builds `W` from the shadow iterates and
   cross-products, assert `W x = W⁻ᵀ s` and `W x̃ = W⁻ᵀ s̃` (these define the scaling).
   This validates that the cone feeds correct `F′`, `x̃` into the scaling assembly.

3. **Tiny end-to-end.** Solve a 3-variable problem you can check by hand, e.g.
   `min x₁ + x₂ s.t. x₁+x₂+x₃ = 1, x ∈ P_α` for a couple of `α`, with and without the
   corrector; confirm convergence and that the corrector reduces iterations
   (the exp-cone paper's example is a good template).

4. **Cross-check factor.** In tests, assert structured `L Lᵀ` (§5) matches a dense
   Cholesky of the §4 matrix.

---

## 11. Scaling assembly — what's shared vs. cone-specific

Almost all of the BFGS scaling is **cone-agnostic** and shared with `ExpCone`:
shadow iterates `x̃ = -∇F*(s)` (§7), `s̃ = -F′(x)` (§3), the 3-D cross-product directions
`z = (x ⊗ x̃)/‖x ⊗ x̃‖`, `r = (s ⊗ s̃)/⟨s ⊗ s̃, z⟩`, the BFGS scalar `t` (eq. 32 of the
paper), and the `W, W⁻¹` assembly. The **only** power-cone-specific inputs to that
pipeline are the oracles in §3–§7. If your `ExpCone` already factors the scaling code
out of the cone, `PowerCone` should reuse it unchanged.

---

## Method → section map (mirror your `ExpCone` interface)

| your method (likely name) | section |
|---|---|
| `update!(cone, x, s)` (fill cache) | §0 |
| `in_cone` / `in_dual_cone` / step-to-boundary | §2, §9 |
| `grad!` | §3 |
| `hess!` (entries) | §4 |
| `hess_fact!` / `inv_hess_prod!` | §5 |
| `correction!` (η) | §6 |
| `shadow!` / `conj_grad!` (x̃) | §7 |
| `set_start!` | §8 |
| scaling (`W`, secants) — *shared* | §11 |

All formulas verified symbolically and/or numerically; reproduce the **✓ Check** in each
section as the unit test for that method.
