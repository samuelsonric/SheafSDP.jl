# ADMM for the equality-constrained QP — `src/kkt/admm.jl`

The **operator-splitting** member of the `KKTWorkspace` family. It solves the same ECQP as `UzawaWorkspace`,

```
min  ½ xᵀA x − fᵀx
s.t. B x = g                    (B = δ, the sheaf coboundary)
```

with `A` block-diagonal by stalk — but it shares **none** of Uzawa's factorization stack. That is the whole point: Uzawa factors the coupled, fill-prone `A + αBᵀB` (where chordal Cholesky earns its keep); ADMM factors only the **block-diagonal** `A + αI` (`N` independent stalk-sized Choleskys, zero fill) and pushes all graph coupling into a matrix-free **sheaf-diffusion** projection. So this file mentions chordal machinery *nowhere*. It reuses `BlockSparseMatrix`, `copydia!`, `it!`, and `LinearOperator` from the rest of the family, and borrows no symbolic-factorization infrastructure.

When to reach for it (we worked through this): for an *easy* node objective on a *low-treewidth* sheaf, `UzawaWorkspace` dominates — exact on both halves off one cheap factor. ADMM earns its place on the **scale exit** (the coupled factor won't fit / fill-in explodes), the **modeling exit** (the node objective grows a nonsmooth or constrained term, so the split does real work), and the **deployment exit** (the robust one-knob method with a fast low-accuracy phase). Included here as a correct, first-class option for those.

---

## The idea: consensus splitting

Introduce a copy `z ∈ C⁰` of the primal and move the homological constraint onto it:

```
min  f(x) + χ_C(z)        s.t.  x − z = 0,     C = { z : L_F z = Bᵀg }
```

with `f(x) = ½xᵀA x − fᵀx`, `L_F = BᵀB` the linear sheaf Laplacian, and `χ_C` the indicator of the feasible set `C = δ⁺g + H⁰`. Scaled-dual ADMM gives:

```
x ← (A + αI)⁻¹ (f + α(z − u))     [block-diagonal node solve]
z ← Π_C(x + u)                    [coupling — sheaf diffusion]
u ← u + x − z                     [vertex dual ascent]
```

Three structural facts:

1. **The node solve is block-diagonal.** The augmentation is `‖x − z‖²`, so the `x`-update factor is `A + αI` — no `B`, no coupling, no fill. It is `N` independent stalk-sized `L Lᵀ` solves. This is what lets ADMM sidestep the scale exit.

2. **All coupling is in the `z`-update, and it is sheaf diffusion.** Projecting onto the affine set `C` means solving the singular system `L_F z = Bᵀg`, and **(warm-started) `it!` on `L_F` is exactly that projection.** With a `RiWorkspace` it is literally the heat equation `z ← z − τ Bᵀ(Bz − g)`; with `CgWorkspace`/`CrWorkspace` it is the Krylov-accelerated version. The seed **must** be `x + u` and must not be re-zeroed: every iterate correction lives in `im Bᵀ = (H⁰)⊥`, so the `H⁰` component of the seed is conserved at machine precision, which is what makes the limit the *orthogonal* projection of `x + u`. (This is the entire reason `it!` needed a warm-start path — see below.)

3. **The dual is on vertices.** `u ∈ C⁰`, the multiplier of `x − z = 0` — forced by the split, the opposite of the edge dual the rest of the stack carries. To honor the public contract (`solve_kkt!` writes `(x, y)` with `y ∈ C¹`), we **recover** the edge multiplier at the end: at convergence `Bᵀy = α u`, so the min-norm `y = B w` where `L_F w = α u` — one more `it!` solve, cold. Gated by `recover`.

Feasibility caveat, unchanged but worth restating because the split *hides* it: `L_F z = Bᵀg` is always consistent, so an infeasible `g ∉ im δ` does not stall the way it does in the edge formulation — the diffusion quietly returns the least-squares section. Test `g ∈ im B` separately if you need to catch it.

---

## Prerequisite: `it!` warm-start

The projection and the dual recovery both run through `it!`. The projection needs to **start from a seed and not zero it**; recovery starts cold. So `it!` (and `ri!`) gain a warm overload taking an optional **positional `x0`**, mirroring Krylov's own `cg!(solver, A, b, x0)`. Presence of `x0` *is* the warm start; absence is the cold start. Every existing call site (Uzawa's Schur solve) binds the cold method and is unchanged.

```julia
# ── ri! : cold zeros the iterate, warm seeds it from x0 ──────────────
function ri!(workspace::RiWorkspace{T}, S, b::AbstractVector{T};
             α::Real=1.0, atol::Real=√eps(T), rtol::Real=√eps(T), itmax::Integer=1000) where {T}
    fill!(workspace.x, 0)
    return ri_impl!(workspace, S, b; α, atol, rtol, itmax)
end

function ri!(workspace::RiWorkspace{T}, S, b::AbstractVector{T}, x0::AbstractVector{T};
             α::Real=1.0, atol::Real=√eps(T), rtol::Real=√eps(T), itmax::Integer=1000) where {T}
    copyto!(workspace.x, x0)
    return ri_impl!(workspace, S, b; α, atol, rtol, itmax)
end

# shared loop (assumes workspace.x already initialized)
function ri_impl!(workspace::RiWorkspace{T}, S, b::AbstractVector{T};
                  α::Real=1.0, atol::Real=√eps(T), rtol::Real=√eps(T), itmax::Integer=1000) where {T}
    x = workspace.x
    r = workspace.r
    ε = atol + rtol * norm(b)
    for k in 1:itmax
        mul!(r, S, x); axpby!(1, b, -1, r)
        norm(r) ≤ ε && (workspace.niter = k; return workspace)
        axpy!(α, r, x)
    end
    @warn "ri! did not converge in $itmax iterations"
    workspace.niter = itmax; return workspace
end

# ── it! : cold and warm, dispatching on the presence of x0 ──────────
function it!(itrwrk::IterationWorkspace{T}, S, b::AbstractVector{T};
             α::Real=1.0, atol::Real=√eps(T), rtol::Real=√eps(T), itmax::Integer=1000) where {T}
    itrwrk isa RiWorkspace ? ri!(itrwrk, S, b; α, atol, rtol, itmax) :
    itrwrk isa CgWorkspace ? cg!(itrwrk, S, b;    atol, rtol, itmax) :
                             cr!(itrwrk, S, b;    atol, rtol, itmax)
    return itrwrk
end

function it!(itrwrk::IterationWorkspace{T}, S, b::AbstractVector{T}, x0::AbstractVector{T};
             α::Real=1.0, atol::Real=√eps(T), rtol::Real=√eps(T), itmax::Integer=1000) where {T}
    itrwrk isa RiWorkspace ? ri!(itrwrk, S, b, x0; α, atol, rtol, itmax) :
    itrwrk isa CgWorkspace ? cg!(itrwrk, S, b, x0;    atol, rtol, itmax) :
                             cr!(itrwrk, S, b, x0;    atol, rtol, itmax)
    return itrwrk
end
```

For Richardson "warm" means *don't zero* (`copyto!` the seed instead); for Krylov it means *forward `x0`* (which starts internally from zero otherwise). Same intent — use the resident iterate as the start — expressed per backend. All three preserve the `H⁰` component of the seed on the singular `L_F`, because every correction lives in the Krylov space (or `im S` for Richardson) `⊆ (H⁰)⊥`.

---

## How it maps onto the API

**Penalty `α` — auto-scaled, different formula than Uzawa.** Uzawa's penalty multiplies `BᵀB`, so its natural scale is `‖A‖/‖B‖²` (the `/nrm`). ADMM's penalty multiplies `I` (it weights `‖x − z‖²` against the node curvature), so the `/nrm` drops:

```
α = max(aaug, raug · ‖A‖)        # multiplies I, not BᵀB ⇒ no /nrm
```

**Diffusion step `τ` — auto from the same `nrm = ‖B‖²`.** Richardson on `L_F` converges iff `0 < τ < 2/λ_max(L_F)`, and `nrm = ‖B‖_F² ≥ σ_max(B)² = λ_max(L_F)`, so `τ = 1/nrm` is always safely convergent. (Used only when `itrwrk` is a `RiWorkspace`; CG/CR self-tune and ignore it.) The one precomputed `‖B‖²` sets the penalty scale in Uzawa and the diffusion-step scale here.

**Workspace — no chordal anything.** `F` is a `BlockSparseMatrix` holding `A + αI` factored in place (per-block lower Cholesky). No `facwrk`, no `divwrk`, no `ChordalTriangular`, no `UPLO` parameter (a block-diagonal lower factor is lower by construction). Plus the persistent vertex state `z, u` (warm-started across calls; `x, y` are pure outputs), the `itrwrk` that drives both inner solves (sized `n`, the projection lives in `C⁰`), scratch, and the `α`/`τ`/`nrm` cells. There is no `L` field: the Laplacian is only ever a matrix-free `LinearOperator`, never assembled.

**Settings.** Outer `atol`/`rtol`/`itmax`, the `aaug`/`raug` penalty knobs, inner `iatol`/`irtol`/`iitmax` for the projection and recovery solves, and `recover` for the edge-dual reconstruction.

---

## The code

```julia
@kwdef struct AdmmSettings{T} <: KKTSettings{T}
    aaug::T   = zero(T)      # absolute penalty floor
    raug::T   = one(T)       # relative penalty  (α = max(aaug, raug·‖A‖))
    atol::T   = √eps(T)      # outer (ADMM) tolerance
    rtol::T   = √eps(T)
    itmax::Int = 1000        # outer iterations
    iatol::T  = √eps(T)      # inner (projection / recovery) tolerance
    irtol::T  = √eps(T)
    iitmax::Int = 1000       # inner iterations
    recover::Bool = true     # reconstruct edge multiplier y ∈ C¹
end

struct AdmmWorkspace{T, M <: BlockSparseMatrix{T}, ItrWrk <: IterationWorkspace{T}} <: KKTWorkspace{T}
    F::M                # block-diagonal A + αI, factored in place (per-stalk lower chol)
    itrwrk::ItrWrk      # projection / recovery solver, sized n
    z::Vector{T}        # section iterate    ∈ C⁰   [persistent / warm-start]
    u::Vector{T}        # scaled vertex dual ∈ C⁰   [persistent / warm-start]
    zprev::Vector{T}    # dual-residual scratch
    t::Vector{T}        # C⁰ scratch
    s::Vector{T}        # C¹ scratch (edge)
    α::Scalar{T}        # penalty
    τ::Scalar{T}        # diffusion Richardson step (Ri only)
    nrm::T              # ‖B‖²  (sets τ)
end

function AdmmWorkspace(F::BlockSparseMatrix{T}, B::BlockSparseMatrix{T}) where {T}
    m, n = size(B)
    @assert size(F, 1) == n
    itrwrk = CgWorkspace(n, n, Vector{T})     # default; swap to Ri/Cr as desired
    z = zeros(T, n); u = zeros(T, n); zprev = zeros(T, n)
    t = zeros(T, n); s = zeros(T, m)
    return AdmmWorkspace(F, itrwrk, z, u, zprev, t, s, ones(T), ones(T), norm(B)^2)
end

# cold-start: drop the warm-started iteration state
reset!(w::AdmmWorkspace) = (fill!(w.z, 0); fill!(w.u, 0); w)

#
# L_F = BᵀB as a matrix-free SPD operator (closure over edge scratch s)
#
function sheaf_laplacian(B::BlockSparseMatrix{T}, s::AbstractVector{T}) where {T}
    n = size(B, 2)
    return LinearOperator(T, n, n, true, true, (y, w) -> (mul!(s, B, w); mul!(y, B', s)))
end

function init_kkt!(w::AdmmWorkspace{T}, set::AdmmSettings{T}, A::BlockSparseMatrix) where {T}
    α = max(set.aaug, set.raug * norm(Symmetric(A, :L)))   # NB: no /nrm — penalty multiplies I
    w.α[] = α
    w.τ[] = one(T) / w.nrm                                  # safe Richardson step for L_F
    init_admm!(w.F, A, α)
    return w
end

#
# form and factor the block-diagonal augmented block, by hand, block by block:
#
#   F = A + α I,   then   Fᵥᵥ = Lᵥ Lᵥᵀ   per stalk v
#
function init_admm!(F::BlockSparseMatrix{T}, A::BlockSparseMatrix{T}, α::T) where {T}
    @assert size(F, 1) == size(A, 1)
    copydia!(F, A)                          # F = A   (diagonal blocks)
    for v in 1:nvtxs(F)
        Fvv = block(F, v, v, v)
        for i in 1:size(Fvv, 1)
            Fvv[i, i] += α                  # F = A + α I  on this stalk block
        end
        cholesky!(Symmetric(Fvv, :L))       # in-place lower Cholesky of the block
    end
    return F
end

function solve_kkt!(
        w::AdmmWorkspace{T},
        set::AdmmSettings{T},
        x::AbstractVector{T},
        y::AbstractVector{T},
        B::BlockSparseMatrix{T},
        f::AbstractVector{T},
        g::AbstractVector{T}
    ) where {T}
    niter = solve_admm!(w.itrwrk, w.F, x, w.z, w.u, w.zprev, w.t, w.s, B, f, g,
                        w.α[], w.τ[], set.atol, set.rtol, set.itmax,
                        set.iatol, set.irtol, set.iitmax)
    if set.recover
        recover_dual!(w.itrwrk, y, w.t, w.s, B, w.u, w.α[], w.τ[],
                      set.iatol, set.irtol, set.iitmax)
    else
        fill!(y, 0)
    end
    return niter
end

#
# consensus-split ADMM:
#
#   x ← (A + αI)⁻¹ (f + α(z − u))      [block-diagonal L Lᵀ solve]
#   z ← Π_C(x + u)                     [warm it! on L_F z = Bᵀg, seed x+u]
#   u ← u + x − z                      [vertex dual ascent]
#
# z, u persist in the workspace and warm-start across calls (reset! to cold-start).
#
function solve_admm!(
        itrwrk::IterationWorkspace{T},
        F::BlockSparseMatrix{T},
        x::AbstractVector{T},
        z::AbstractVector{T},
        u::AbstractVector{T},
        zprev::AbstractVector{T},
        t::AbstractVector{T},
        s::AbstractVector{T},
        B::BlockSparseMatrix{T},
        f::AbstractVector{T},
        g::AbstractVector{T},
        α::T, τ::T,
        atol::T, rtol::T, itmax::Int,
        iatol::T, irtol::T, iitmax::Int
    ) where {T}
    m, n = size(B)
    @assert length(x) == n && length(z) == n && length(u) == n && length(f) == n
    @assert length(s) == m && length(g) == m

    L  = sheaf_laplacian(B, s)
    LF = LowerTriangular(F)                 # block-diagonal lower factor; ldiv! "just works"
    ε  = atol + rtol * norm(f)

    for k in 1:itmax
        #
        # x-update:  (A + αI) x = f + α(z − u)        [no B — block diagonal]
        #
        copyto!(t, f); axpy!(α, z, t); axpy!(-α, u, t)   # t = f + α(z − u)
        copyto!(x, t)
        ldiv!(LF, x); ldiv!(LF', x)                      # L Lᵀ x = t
        #
        # z-update:  z = Π_C(x + u)  via warm it!,  seed = x + u,  RHS = Bᵀg
        #
        copyto!(zprev, z)
        zs = solution(itrwrk)
        copyto!(zs, x); axpy!(1, u, zs)                  # seed the iterate in place
        mul!(t, B', g)                                   # RHS = Bᵀg
        it!(itrwrk, L, t, zs; α=τ, atol=iatol, rtol=irtol, itmax=iitmax)
        copyto!(z, solution(itrwrk))
        #
        # u-update:  u ← u + x − z
        #
        axpy!(1, x, u); axpy!(-1, z, u)
        #
        # residuals:  primal ‖x − z‖,  dual α‖z − zprev‖
        #
        copyto!(t, x); axpy!(-1, z, t);     rp = norm(t)
        copyto!(t, z); axpy!(-1, zprev, t); rd = α * norm(t)
        if rp ≤ ε && rd ≤ ε
            return k
        end
    end

    @warn "solve_admm! did not converge in $itmax iterations"
    return itmax
end

#
# recover the edge multiplier:  at convergence  Bᵀy = α u.
# min-norm  y = B w  where  L_F w = α u   (cold it!, w₀ = 0).
#
function recover_dual!(
        itrwrk::IterationWorkspace{T},
        y::AbstractVector{T},
        t::AbstractVector{T},
        s::AbstractVector{T},
        B::BlockSparseMatrix{T},
        u::AbstractVector{T},
        α::T, τ::T,
        iatol::T, irtol::T, iitmax::Int
    ) where {T}
    L = sheaf_laplacian(B, s)
    copyto!(t, u); rmul!(t, α)                           # RHS = α u
    it!(itrwrk, L, t; α=τ, atol=iatol, rtol=irtol, itmax=iitmax)   # cold (w₀ = 0)
    mul!(y, B, solution(itrwrk))                         # y = B w  ⇒  Bᵀy = α u
    return y
end
```

---

## Slotting it in

**1. Register the file** (after `uzawa.jl`, which defines `copydia!`):

```julia
abstract type KKTWorkspace{T} end
abstract type KKTSettings{T} end

include("it.jl")        # ← now carries the warm it!/ri! overloads
include("uzawa.jl")
include("admm.jl")      # ← add
```

`copydia!` already copies a `BlockSparseMatrix`'s diagonal blocks; ADMM reuses it. The shift and per-block factor are done by hand in `init_admm!` — no chordal symbolic step.

**2. Build `F` as a block-diagonal `BlockSparseMatrix`** sized to the vertex stalks — *not* a chordal factor, and *not* Uzawa's `A + αBᵀB` pattern. It holds `A + αI` and is factored in place per block. That structural triviality (zero fill) is the entire reason ADMM sidesteps the scale exit, so handing it Uzawa's filled factor would defeat the method.

**3. Construct and run:**

```julia
wrk = AdmmWorkspace(F_blockdiag, B)        # F_blockdiag: block-diagonal BSM over the stalks
set = AdmmSettings{Float64}(raug = 1.0)    # tune aaug/raug, inner tols, recover

init_kkt!(wrk, set, A)                      # factors A + αI per block, sets α and τ
solve_kkt!(wrk, set, x, y, B, f, g)         # writes x (and y if set.recover)
```

**4. Warm-starting.** `x, y` are outputs; the live state is `wrk.z, wrk.u`. Across a sequence of related solves (an MPC horizon on fixed topology) just call `solve_kkt!` again — `z, u` warm-start automatically and cut the outer count. `reset!(wrk)` cold-starts. `init_kkt!` only re-factors `A + αI`; it does not touch `z, u`, so re-initializing the Hessian and keeping the warm start are independent.

**5. Inner solver choice.** `itrwrk` picks how the projection (and recovery) is solved, exactly as it picks Uzawa's Schur solver: `CgWorkspace` is the fast default (self-tunes, ignores `τ`); `RiWorkspace` recovers literal sheaf diffusion with the fixed local step `τ = 1/nrm` — the distributed-friendly form; `CrWorkspace` is the safer Krylov choice on the singular operator. "Sheaf diffusion" is just "the projection, solved with Richardson."

**6. `recover` and `y`.** `true` gives the edge multiplier `y ∈ C¹` (one extra cold `it!` solve), so `(x, y)` matches the Uzawa output contract and is interchangeable downstream; `false` zeros `y`. Either way `y` is determined mod `H¹` (the same gauge freedom as the Uzawa dual); `recover_dual!` returns the min-norm representative.

---

## Tuning, in one breath

Set `raug` and walk away — ADMM converges for **any** `α > 0`, no stability ceiling (unlike AH's steps). Larger `α` weights consensus harder: faster feasibility, slower objective progress. `τ` is auto-safe from `nrm`; the only inner knob worth touching is `iitmax`/`iatol` — loosen the projection tolerance early and tighten it as you converge to trim inner iterations. The method's calling card is a fast low-accuracy phase: a handful of outer iterations buys a few digits, ideal inside a control loop. For high accuracy it tails off — which is exactly where, memory permitting, `UzawaWorkspace` is the better tool.

---

## What this file does *not* touch

No `ChordalCholesky`, `ChordalTriangular`, `FactorizationWorkspace`, or `DivisionWorkspace` — the entire chordal stack stays in `uzawa.jl` where the coupled factor needs it. ADMM's primal factor is `N` independent dense blocks; its coupling is matrix-free. The type signature now tells the truth: ADMM and Uzawa share the `BlockSparseMatrix`/`copydia!`/`it!`/`LinearOperator` substrate and nothing else.
