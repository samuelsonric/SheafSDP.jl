# Adding a convex quadratic objective `½pᵀQp` and a free-variable (NOC) cone

## 0. Summary of the change

We extend the primal objective from `cᵀp` to `cᵀp + ½pᵀQp`, with `Q = blkdiag(Q_v) ⪰ 0`
sharing the block structure of the NT Hessian `H`. The net effect on the solver is:

- **Operator:** the KKT (1,1) block becomes `H + Q` instead of `H`.
- **Residual:** dual feasibility gains a `Qp` term.
- **Newton recovery:** `Δd` gains a `QΔp` term.

Everything else — `corr!`, `corrector_rhs!`, the affine RHS, `μ`, `σ`, `step_to_boundary` —
is untouched, because those are properties of the **cone**, and `Q` is a property of the
**objective**. The two never mix.

The `NOC` (no-cone / free variable) option is then just the cone `K_v = ℝ^{n_v}`. Its dual
cone is `{0}`, so its slack is identically zero and it carries no barrier; all of its curvature
in the (1,1) block must come from `Q_v`, which is why `Q_v ≻ 0` is required there.

---

## 1. Theory

### 1.1 Problem statement

Original conic program:

```
primal:  min  cᵀp                 dual:  max  gᵀy
         s.t. Bp = g                     s.t. Bᵀy + d = c
              p ∈ K                            d ∈ K*
```

With the quadratic term:

```
primal:  min  cᵀp + ½pᵀQp ,    Q = blkdiag(Q_v) ⪰ 0,    p ∈ K,   Bp = g
```

### 1.2 KKT conditions

Stationarity in `p` now includes the gradient of the quadratic, `Qp`:

```
(stationarity / dual feas.)   c + Qp − Bᵀy − d = 0,   d ∈ K*
(primal feasibility)          Bp = g,                 p ∈ K
(complementarity)             p ∘ d = 0     (handled via NT scaling)
```

The only condition that changes is dual feasibility: `Bᵀy + d = c` becomes
`Bᵀy + d − Qp = c`. Primal feasibility and cone complementarity are unchanged.

### 1.3 Residuals

```
r_p = g − Bp                  (unchanged)
r_d = c + Qp − Bᵀy − d        (gains the +Qp term)
```

This is the single change to `residuals!`.

### 1.4 The Newton system

We linearize the three groups. Complementarity is encoded in this solver through the NT
Hessian `H = W⁻¹ ⊗ₛ W⁻¹` and the centering target `r_c` (it is `−p` for the affine step;
the corrector term otherwise). Crucially, **`Q` appears nowhere in complementarity** — it is
not a cone quantity:

```
(complementarity, linearized)   H Δp + Δd = H r_c
(dual feas., linearized)        Bᵀ Δy + Δd − Q Δp = r_d
(primal feas., linearized)      B Δp = r_p
```

Eliminate `Δd`. From complementarity, `Δd = H r_c − H Δp`. Substitute into dual feasibility:

```
Bᵀ Δy + (H r_c − H Δp) − Q Δp = r_d
⟹  (H + Q) Δp − Bᵀ Δy = H r_c − r_d
```

Together with `B Δp = r_p`, the reduced (saddle) system is:

```
┌ H+Q   Bᵀ ┐ ┌ Δp ┐   ┌ H r_c − r_d ┐
│           │ │     │ = │             │
└  B    0  ┘ └ −Δy ┘   └    r_p      ┘
```

### 1.5 Why `Q` is in the operator but **not** the RHS

This is the load-bearing observation, and it is what keeps the change small.

- `Q` reached the left-hand side **only** through the term `−Q Δp` — i.e. it multiplies the
  *unknown* `Δp`. That is why it joins the (1,1) operator block, giving `H + Q`.
- The right-hand side term `H r_c` came from the **complementarity** equation, which never
  contained `Q`. There is no path by which `Q` can multiply `r_c`.
- The only way `Q` reaches the RHS at all is through `r_d` (via the `Qp` residual), which we
  already recompute every iteration.

Intuition: `r_c` is a *cone-centering* correction (distance from the central path in the
cone's geometry); `Q` is *objective curvature*. Objective curvature only acts on a search
direction when that direction perturbs the gradient `Qp`. So `Q` is structurally confined to
the operator (`·Δp`) and the gradient residual (`r_d`).

**Code consequence.** `H r_c` is produced by `corr!` straight from the NT cache
(`sdpcorr!` via `LP,U,s`; `poscorr!` via `p,d`) — never from the assembled `H` block.
Therefore folding `Q` into the assembled blocks (§2.3) cannot contaminate the RHS. The
`H` inside `H+Q` and the `H` inside `H r_c` are computed by two different code paths, which
is exactly the separation we want.

### 1.6 Mapping to `solve_kkt!`

`solve_kkt!` solves `[A Bᵀ; B 0][x; w] = [f; g]`. We set

```
A = H + Q ,   x = Δp ,   w = −Δy ,   f = H r_c − r_d ,   g = r_p
```

After the solve, `Δy = −w` and the dual-slack recovery is read off the linearized dual
feasibility equation:

```
Δd = r_d − Bᵀ Δy + Q Δp        (was: Δd = r_d − Bᵀ Δy)
```

### 1.7 Affine and corrector RHS are unchanged in form

- **Affine:** `r_c = −p`, and `H r_c = H(−p) = −d` by the NT property `Hp = d`, so
  `f_aff = −d − r_d = −(d + r_d)`. The existing line `@. f = -(d + r_d)` is already correct;
  it simply inherits the new `r_d`.
- **Corrector:** `corrector_rhs!` writes `H r_c` per block via `corr!`, then `axpy!(-1, r_d, f)`
  gives `f = H r_c − r_d`. Again automatically correct under the new `r_d`.

No edits to `corr!`, `corrector_rhs!`, or the affine RHS.

### 1.8 The NOC (free-variable) cone

A free block is `K_v = ℝ^{n_v}`, whose dual cone is `K*_v = {0}`. Consequences:

- `d_v ≡ 0`: the dual slack on a free block is identically zero. There is no barrier and no
  complementarity for that block.
- `degree(NOC, n) = 0`: free variables carry no barrier parameter, so they drop out of
  `ν = Σ degree`. They also contribute nothing to `⟨p,d⟩` since `d_v = 0`, so `μ = ⟨p,d⟩/ν`
  stays consistent.
- The (1,1) block is `A_v = H_v + Q_v = 0 + Q_v = Q_v`. For the chordal factorization of
  `F = A + αBᵀB` to remain definite we need `A_v` to be PD on its own, hence **`Q_v ≻ 0`**.

The free-block row of the reduced system is `Q_v Δp_v − (BᵀΔy)_v = −r_{d,v}`, and the
recovery `Δd = r_d − BᵀΔy + QΔp` then gives `Δd_v = 0` automatically — the invariant
`d_v ≡ 0` maintains itself. (See §3 for a defensive re-zeroing note, since the Schur solve
is only to CG tolerance.)

### 1.9 Convergence is still valid for the QP

At a point with `p ∈ K`, `d ∈ K*`, `r_p = r_d = 0` and `⟨p,d⟩ = 0`, the full KKT system
above is satisfied, so the point is optimal for the QP. Since `r_d` now properly includes
`Qp`, the existing stopping test (`‖r_p‖`, `‖r_d‖`, `μ` all small) is unchanged and correct.
The one edge case is `ν = 0` (all blocks free → an equality-constrained QP): then `μ = 0/0`,
so skip the gap test and converge on feasibility alone.

---

## 2. Implementation changes

`Q` is best stored as a **block-diagonal `BlockSparseMatrix`** with the same structure as
`H` (reuse `allocate_H`), because we need it both as a matvec operator (in `residuals!` and
`newton_step!`) and as blocks to fold into the (1,1) operator.

> **Storage note (important):** store each `Q_v` as a **full symmetric** dense block, not
> just a triangle. The factorization only reads one triangle (so a triangle would suffice
> there), but `mul!(·, Q, ·)` in `residuals!`/`newton_step!` reads the **full** matrix. A
> lower-only `Q_v` would give a wrong matvec.

Throughout, `Q` is optional (`nothing` ⇒ exact current behavior).

### 2.1 `solve!` signature (ipm.jl)

Add a keyword:

```julia
Q::Union{Nothing, BlockSparseMatrix{T}} = nothing,
```

and thread it into `residuals!`, `hess!`, and the two `newton_step!` calls. The `α` scaling
line `α = kkt_frac * norm(Symmetric(H, :L)) / norm_B_sq` needs **no change**: because we fold
`Q` into `H` during assembly (§2.3) *before* `α` is computed, `norm(Symmetric(H,:L))` already
reflects `H + Q`.

### 2.2 `residuals!` — add `Qp` to `r_d`

```julia
function residuals!(rp, rd, B, p, d, y, c, g, Q=nothing)
    # r_p = g − Bp
    copyto!(rp, g)
    mul!(rp, B, p, -1, 1)

    # r_d = c + Qp − Bᵀy − d
    copyto!(rd, c)
    Q !== nothing && mul!(rd, Q, p, 1, 1)   # rd += Q p
    mul!(rd, B', y, -1, 1)
    axpy!(-1, d, rd)
    return rp, rd
end
```

### 2.3 `hess!` assembly (ipm.jl) — fold `Q_v` into each block

The per-cone `hess!` overwrites `H_v` fresh each iteration (POS/SOC/SDP all fill it), so
adding `Q_v` afterward is safe — no accumulation:

```julia
function hess!(H, caches, cones, p, d, B, Q=nothing)
    for (i, (v, cone)) in enumerate(zip(vtxs(B), cones))
        r   = colrange(B, v)
        H_v = block(H, v, v, v)
        c   = cache(caches, i, cone)
        p_v = view(p, r); d_v = view(d, r)

        scale!(p_v, d_v, c)
        hess!(H_v, p_v, d_v, c)                 # H_v ← W⁻¹⊗ₛW⁻¹   (0 for NOC)
        Q !== nothing && axpy!(true, block(Q, v, v, v), H_v)   # H_v ← H_v + Q_v
    end
end
```

After this, the matrix handed to `factor_kkt!` *is* `H + Q`; `factor_kkt!`, `solve_kkt!`,
and the Schur solve need no edits.

### 2.4 `newton_step!` — add `QΔp` to the `Δd` recovery

```julia
function newton_step!(Δp, Δy, Δd, divwrk, itrwrk, r, F, B, f, r_p, r_d, Q=nothing; α, atol, rtol, itmax)
    solve_kkt!(divwrk, itrwrk, Δp, Δy, r, F, B, f, r_p; α, atol, rtol, itmax)
    lmul!(-1, Δy)                       # Δy = −w

    # Δd = r_d − Bᵀ Δy + Q Δp
    copyto!(Δd, r_d)
    mul!(Δd, B', Δy, -1, 1)
    Q !== nothing && mul!(Δd, Q, Δp, 1, 1)
    return
end
```

Pass `Q` in both call sites (affine and corrector).

### 2.5 Convergence guard for `ν = 0` (ipm.jl, inside `solve!`)

```julia
μ_curr = ν > 0 ? mu(p, d, ν) : zero(T)
push!(μ_history, μ_curr)
...
gap_ok = ν == 0 || μ_curr < params.gap_tol
if norm_rp < params.feas_tol && norm_rd < params.feas_tol && gap_ok
    status = :optimal
    ...
end
```

(Setting `μ_curr = 0` when `ν = 0` also keeps `is_stalled` from seeing NaNs.)

### 2.6 New file `src/cone/noc.jl`

```julia
#
# NOC cone (no cone / free variables, K = ℝⁿ, K* = {0})
#
# Requires Q_v ≻ 0 to supply all curvature in the (1,1) block.
#

struct NOC <: Cone end

struct NOCCache <: AbstractCache{NOC}
    cone::NOC
end
NOCCache() = NOCCache(NOC())

degree(::NOC, n::Int) = 0          # no barrier ⇒ drops out of ν and μ
cachesize(::NOC, n::Int) = 0
cache(::Caches, ::Int, c::NOC) = NOCCache(c)

# free start: p_v = 0 is fine; d_v = 0 is required (K* = {0})
function identity!(x::AbstractVector{T}, ::NOC) where {T}
    fill!(x, zero(T))
    return x
end

# no NT scaling
scale!(::AbstractVector, ::AbstractVector, ::NOCCache) = nothing

# H_v = 0; the Q_v fold in hess!-assembly supplies the block
function hess!(H::AbstractMatrix{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::NOCCache) where {T}
    fill!(H, zero(T))
    return H
end

# H r_c = 0 for a zero Hessian, for both affine and corrector
function corr!(r::AbstractVector{T}, ::AbstractVector{T}, ::AbstractVector{T},
               ::AbstractVector{T}, ::AbstractVector{T}, ::Real, ::NOCCache) where {T}
    fill!(r, zero(T))
    return r
end

# no boundary to stay inside
maxstep(x::AbstractVector{T}, ::AbstractVector{T}, ::Bool, ::Real, ::NOCCache) where {T} = one(T)
```

### 2.7 Wiring (cone.jl + SheafSDP.jl)

- `src/cone/cone.jl`: add `include("noc.jl")` next to the other cone includes.
- `src/SheafSDP.jl`: add `NOC` to the `export Cone, SDP, POS, SOC` line.

### 2.8 Objective in the comparison scripts

The reported objective must include the quadratic:

```julia
obj_sheaf = dot(c, result.p) + 0.5 * dot(result.p, Q * result.p)
```

and the JuMP model objective becomes `Min, cᵀp + ½ pᵀQp` (e.g. add the block
`@expression`s for each `Q_v`).

---

## 3. Checklist / edge cases

- [ ] `Q_v` stored **full symmetric** dense (matvec reads the full block).
- [ ] `Q ⪰ 0` overall; `Q_v ≻ 0` on every `NOC` block (required for definiteness of `F`).
- [ ] `Q` is optional everywhere (`nothing` ⇒ identical to current behavior).
- [ ] `α` scaling unchanged — works because `Q` is folded into `H` before `norm(...)`.
- [ ] `corr!`, `corrector_rhs!`, affine `f = -(d+r_d)` all left as-is.
- [ ] `ν = 0` guard added (pure equality-constrained QP; skip the `μ` test).
- [ ] **Defensive (optional):** because the Schur system is solved to CG tolerance, the
      automatic `Δd_v = 0` on free blocks is only approximate, so `d_v` can drift by
      `O(kkt_rtol)` over many iterations. Cheap insurance: re-zero `d` on `NOC` blocks once
      per iteration (e.g. in `initialize!`'s spirit, or right after the `axpy!(τ_d, Δd, d)`
      update), and likewise ensure `d_v` starts at `0` (handled by `identity!(·, NOC)`).
- [ ] Test: a pure-QP-with-cones problem (Q ≻ 0, all SDP/POS/SOC) should still converge and
      match Mosek with the quadratic in the objective; a mixed problem with some `NOC` blocks
      exercises the free-variable path.
```