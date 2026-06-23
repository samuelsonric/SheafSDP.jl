struct ExponentialCone <: Cone end

struct ExponentialConeCache{T} <: AbstractCache{ExponentialCone}
    cone::ExponentialCone
    M::FMatrixView{T}     # scaling Gram matrix (3×3)
    R::FMatrixView{T}     # analytic factor of F''(x) (3×3)
    xs::FVectorView{T}    # shadow primal x̃ (3)
    ss::FVectorView{T}    # shadow dual s̃ = -F'(x) (3)
    μv::FScalarView{T}    # block-local μ = ⟨x,s⟩/3
end

# degree = 3 always (EXP is intrinsically 3D)
function degree(::ExponentialCone, n::Int)
    @assert n == 3 "EXP cone is 3-dimensional"
    return 3
end

# cache size: M(9) + R(9) + xs(3) + ss(3) + μv(1) = 25
function cachesize(::ExponentialCone, n::Int)
    @assert n == 3 "EXP cone is 3-dimensional"
    return 25
end

# construct view-based cache from Caches
function cache(c::Caches{T}, i::Int, cone::ExponentialCone) where T
    data = view(c.val, c.xblk[i]:c.xblk[i+1]-1)
    M  = reshape(view(data, 1:9), 3, 3)
    R  = reshape(view(data, 10:18), 3, 3)
    xs = view(data, 19:21)
    ss = view(data, 22:24)
    μv = view(data, 25)
    ExponentialConeCache(cone, M, R, xs, ss, μv)
end

# Central point on the exp cone central path
# From Dahl-Andersen: x* ≈ (1.290928, 0.805102, -0.827838)
function identity!(x::AbstractVector, ::ExponentialCone)
    x[1] =  1.2909282315382298
    x[2] =  0.8051015526498357
    x[3] = -0.8278379086082098
    return x
end

# Initialize cache.xs to identity point for warm-starting shadow primal
function initcache!(cache::ExponentialConeCache)
    identity!(cache.xs, cache.cone)
    return cache
end

#
# EXP barrier and derivatives
#
# Barrier: F(x) = -log(ψ(x)) - log(x₁) - log(x₂)
# where ψ(x) = x₂ log(x₁/x₂) - x₃
#

# Barrier argument ψ(x) = x₂ log(x₁/x₂) - x₃
function exppsi(x::AbstractVector{T}) where {T}
    return x[2] * log(x[1] / x[2]) - x[3]
end

# Gradient of ψ: ψ'(x) = (x₂/x₁, log(x₁/x₂) - 1, -1)
function exppsigrad!(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    g[1] = x[2] / x[1]
    g[2] = log(x[1] / x[2]) - one(T)
    g[3] = -one(T)
    return g
end

#
# Note on ψ accuracy (investigated and rejected: compensation does not help)
#
# ψ = x₂·log(x₁/x₂) − x₃ has relative error that grows as 1/ψ (k=1) as ψ → 0.
# This was initially diagnosed as cancellation in the subtraction, suggesting
# compensation (twosum/twoprod) could help. However, measurement revealed:
#
#   1. The subtraction (x₂L) − x₃ is EXACT for Float64 inputs (no rounding)
#   2. The absolute error is FLAT at ~0.5u·phi, not growing
#   3. The 1/ψ relative error growth is purely representation amplification:
#      fixed absolute error ÷ shrinking ψ = growing relative error
#
# The ~0.5u absolute error comes from log(r) itself — a correctly-rounded
# library primitive whose ~0.5 ulp approximation error enters BEFORE our
# arithmetic. Compensation (twosum/twoprod) can only capture rounding from
# operations WE perform; it cannot reach behind a transcendental.
#
# This is the discriminator: flat absolute error at ~0.5u·|terms| indicates
# representation floor (not reducible), not cancellation (reducible). The
# relative error table alone cannot distinguish these — only the absolute
# error column can. ψ is representation-floored; k=1 is bedrock here.
#
# Contrast with SOC: det = x₀² − ‖x̄‖² has cancellation in YOUR arithmetic
# (the subtraction), so twosum captures it. ψ's error is in log, so it doesn't.
#
# Double64 log was also tested: computing log in Double64 gives ~50 orders of
# magnitude more accurate ψ. But the endgame μ was IDENTICAL (both stall at
# μ ≈ 458, same iterations, same status). The extra accuracy is non-binding:
# cond(M) ~ 1/μ floors the KKT solve first, and the extra digits die there.
# The solver never reads them. This is the same pattern as compensation —
# "more accurate" is not the binding question; "does a contract change" is.
#

# Barrier gradient: F'(x) = -ψ'(x)/ψ(x) - (1/x₁, 1/x₂, 0)
function expbarrgrad!(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    ψ = exppsi(x)
    exppsigrad!(g, x)

    # g = -ψ'/ψ - h' where h' = (1/x₁, 1/x₂, 0)
    g[1] = -g[1] / ψ - inv(x[1])
    g[2] = -g[2] / ψ - inv(x[2])
    g[3] = -g[3] / ψ  # = 1/ψ

    return g
end

#
# Change 1: Analytic factor R(x) such that F''(x) = R(x) R(x)ᵀ
#
# The factor has entries that scale like ψ⁻¹ (not ψ⁻² like F''),
# so solves through R have condition √cond(F'') instead of cond(F'').
#
# Structure: full 2×2 top-left, third column (1/ψ)(ψ'₁, ψ'₂, -1), zeros at (3,1), (3,2).
#
# σ = √(1 + 2x₂/ψ), then:
# R = [ (1-σ)/(2x₁)   (1+σ)/(2x₁)   (1/ψ)(x₂/x₁)        ]
#     [ (1+σ)/(2x₂)   (1-σ)/(2x₂)   (1/ψ)(log(x₁/x₂)-1) ]
#     [     0              0            -1/ψ             ]
#

function expbarr!(R::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    x1, x2 = x[1], x[2]
    ψ = exppsi(x)
    ψinv = inv(ψ)

    σ = sqrt(one(T) + 2 * x2 * ψinv)

    # Top-left 2×2
    R[1,1] = (one(T) - σ) / (2 * x1)
    R[1,2] = (one(T) + σ) / (2 * x1)
    R[2,1] = (one(T) + σ) / (2 * x2)
    R[2,2] = (one(T) - σ) / (2 * x2)

    # Third column: (1/ψ)(ψ'₁, ψ'₂, -1)
    R[1,3] = ψinv * (x2 / x1)
    R[2,3] = ψinv * (log(x1 / x2) - one(T))
    R[3,3] = -ψinv

    # Zeros
    R[3,1] = zero(T)
    R[3,2] = zero(T)

    return R
end


#
# Third-order directional derivative F'''(x)[u] as a 3×3 symmetric matrix
#

function expbarrhess!(D::AbstractMatrix{T}, x::AbstractVector{T}, u::AbstractVector{T}) where {T}
    ψ = exppsi(x)
    x1, x2 = x[1], x[2]
    u1, u2, u3 = u[1], u[2], u[3]

    # ψ' and ψ'·u
    ψg1 = x2 / x1
    ψg2 = log(x1 / x2) - one(T)
    ψg3 = -one(T)
    ψgu = ψg1 * u1 + ψg2 * u2 + ψg3 * u3

    # ψ'' (only non-zero entries)
    ψH11 = -x2 / x1^2
    ψH21 = inv(x1)
    ψH22 = -inv(x2)

    # ψ''u
    ψHu1 = ψH11 * u1 + ψH21 * u2
    ψHu2 = ψH21 * u1 + ψH22 * u2
    ψHu3 = zero(T)

    # ψ'''[u] (only non-zero entries)
    ψ3_11 = 2 * x2 * u1 / x1^3 - u2 / x1^2
    ψ3_21 = -u1 / x1^2
    ψ3_22 = u2 / x2^2

    ψ2 = ψ^2
    ψ3 = ψ^3

    # Build D = F'''[u]
    # Term 1: -2 ψ'ψ'ᵀ (ψ'·u) / ψ³
    # Term 2: (ψ'(ψ''u)ᵀ + (ψ''u)ψ'ᵀ) / ψ²
    # Term 3: ψ'' (ψ'·u) / ψ²
    # Term 4: -ψ'''[u] / ψ
    # Term 5: h'''[u] = diag(-2u₁/x₁³, -2u₂/x₂³, 0)

    ψg = (ψg1, ψg2, ψg3)
    ψHu = (ψHu1, ψHu2, ψHu3)
    ψH = ((ψH11, ψH21, zero(T)), (ψH21, ψH22, zero(T)), (zero(T), zero(T), zero(T)))

    for j in 1:3, i in j:3
        D[i,j] = -2 * ψg[i] * ψg[j] * ψgu / ψ3
        D[i,j] += (ψg[i] * ψHu[j] + ψHu[i] * ψg[j]) / ψ2
        D[i,j] += ψH[i][j] * ψgu / ψ2
    end

    # Term 4: -ψ'''[u] / ψ
    D[1,1] -= ψ3_11 / ψ
    D[2,1] -= ψ3_21 / ψ
    D[2,2] -= ψ3_22 / ψ

    # Term 5: h'''[u]
    D[1,1] -= 2 * u1 / x1^3
    D[2,2] -= 2 * u2 / x2^3

    # Symmetrize
    D[1,2] = D[2,1]
    D[1,3] = D[3,1]
    D[2,3] = D[3,2]

    return D
end

#
# Cone membership predicates
#

function expincone(x::AbstractVector{T}) where {T}
    x[1] > 0 && x[2] > 0 && exppsi(x) > 0
end

function expindual(z::AbstractVector{T}) where {T}
    z[1] > 0 && z[3] < 0 && ℯ * z[1] >= -z[3] * exp(z[2] / z[3])
end

#
# Shadow primal computation (Newton iteration, Change 6: warm start + decrement)
#
# Find x̃ such that F'(x̃) = -s via Newton with analytic factor (Change 1).
#
# Change 6 improvements:
#   - Warm start: if xs is already interior, use it (IPM caches x̃ between iters)
#   - Newton decrement convergence: ‖R'Δ‖ < tol, not ‖F'+s‖ < tol
#     (raw gradient has 1/ψ scale, badly conditioned near boundary)
#   - Tighter line search floor (1e-14) to handle boundary approaches
#

function expdualgrad!(xs::AbstractVector{T}, s::AbstractVector{T}; maxiter::Int=50, tol::T=T(1e-12)) where {T}
    # Warm start: use incoming xs if interior, else cold start
    if !expincone(xs)
        # Smart cold start: for EXP cone, x̃ satisfies F'(x̃) = -s.
        # At the central path identity point e ≈ (1.29, 0.81, -0.83), F'(e) = -e.
        # For general s, scale e by a factor that approximately matches ‖s‖.
        # Since F' has 1/ψ scaling, x̃ ∝ e·‖e‖/‖s‖^(1/2) is a rough guess.
        scale = sqrt(max(T(3.0) / (abs(s[1]) + abs(s[2]) + abs(s[3]) + one(T)), T(1e-8)))
        xs[1] = scale * T(1.29)
        xs[2] = scale * T(0.81)
        xs[3] = scale * T(-0.83)
        # Ensure interior
        while !expincone(xs)
            xs[1] *= 2
            xs[2] *= 2
            xs[3] -= one(T)
        end
    end

    # Workspaces
    g      = zeros(T, 3)
    R      = zeros(T, 3, 3)
    Δ      = zeros(T, 3)
    RtΔ    = zeros(T, 3)
    xs_new = zeros(T, 3)

    for iter in 1:maxiter
        # Residual: F'(x̃) + s
        expbarrgrad!(g, xs)
        axpy3!(1, s, g)

        # Newton step: Δ = -F''(x̃)⁻¹ (F'(x̃) + s)
        expbarr!(R, xs)
        axpby!(-1, g, 0, Δ)
        ldiv3!(R, Δ); ldiv3!(R', Δ)

        # Newton decrement = ‖R'Δ‖ (the natural self-concordant metric)
        mul3!(RtΔ, R', Δ)
        decrement = norm3(RtΔ)

        # Converged when decrement is small (scale-invariant criterion)
        if decrement < tol
            return xs
        end

        # Line search with tighter floor for boundary approaches
        θ = one(T)
        while θ > T(1e-14)
            copy3!(xs_new, xs)
            axpy3!(θ, Δ, xs_new)
            if expincone(xs_new)
                copy3!(xs, xs_new)
                break
            end
            θ *= T(0.5)
        end

        if θ <= T(1e-14)
            @warn "Shadow primal Newton line search failed"
            break
        end
    end

    return xs
end

#
# BFGS t computation via contracted form with cancellation-free d (Change 4+5)
#
# A := F'' − s̃s̃ᵀ/ϑ − vvᵀ/d is rank-1 along z (it kills span{x,x̃}),
# so t = μ‖A‖_F = μ·zᵀAz = μ(‖R'z‖² − (s̃ᵀz)²/ϑ − (vᵀz)²/d).
#
# Contracting with z first avoids forming the 1/ψ²-scale matrix and the 1/d².
# A valid (PD) scaling needs t > 0; t ≤ 0 signals fall back to μF''.
#
# Cancellation-free d (Change 5):
# The old d = ‖R'x̃‖² − 3μ̃² subtracted two O(3) numbers to get O(ε²), losing
# ~7 digits near the central path. Using log-homogeneity identities:
#   w = x̃ − μ̃x           (gap direction, → 0 on the central path)
#   d = ⟨w, F''w⟩ = ‖R'w‖²  (sum of squares, manifestly ≥ 0)
#   v = F''w              ⟹ vᵀz = ⟨R'w, R'z⟩
# The cancellation is now in forming w (one O(1)−O(1)→O(ε) subtraction),
# giving ~u/ε relative error instead of ~u/ε². Also v is never formed.
#
# Reference: Dahl & Andersen, Math. Program. 194 (2022), eq. (32).
#
function expbfgs(
        R::AbstractMatrix{T},
        xs::AbstractVector{T},
        ss::AbstractVector{T},
        z::AbstractVector{T},
        x::AbstractVector{T},
        μv::T,
        μt::T
    ) where {T}
    # Workspaces
    w   = zeros(T, 3)
    Rtw = zeros(T, 3)
    Rtz = zeros(T, 3)

    # Gap direction w = x̃ − μ̃x (→ 0 on the central path)
    copy3!(w, xs)
    axpy3!(-μt, x, w)

    # d = ⟨w, F''w⟩ = ‖R'w‖² (sum of squares, ≥ 0)
    # v = F''w ⟹ vᵀz = ⟨R'w, R'z⟩
    mul3!(Rtw, R', w)
    d = dot3(Rtw, Rtw)

    # d ≤ 0 only on the central path; the rel_z gate already covers that.
    # Guard defensively for numerical edge cases.
    d ≤ zero(T) && return zero(T)

    # zᵀAz = ‖R'z‖² − (s̃ᵀz)²/3 − (vᵀz)²/d
    mul3!(Rtz, R', z)
    fppzz = dot3(Rtz, Rtz)        # ‖R'z‖² = zᵀF''z
    sz    = dot3(ss, z)           # s̃ᵀz
    pz    = dot3(Rtw, Rtz)        # vᵀz = ⟨R'w, R'z⟩

    return μv * (fppzz - sz^2 / 3 - pz^2 / d)
end

#
# Scale computation (Tunçel scaling in 3D)
#
# Changes 1-4 applied:
#   - Analytic factor R(x)
#   - Closed form M = ssᵀ/⟨x,s⟩ + δsδsᵀ/⟨δx,δs⟩ + tzzᵀ
#   - Unified guard on gap = μμ̃ - 1
#   - BFGS t via R-products
#

function expscale!(
        M::AbstractMatrix{T},
        R::AbstractMatrix{T},
        xs::AbstractVector{T},
        ss::AbstractVector{T},
        x::AbstractVector{T},
        s::AbstractVector{T}
    ) where {T}

    # Workspace
    z  = zeros(T, 3)
    δx = zeros(T, 3)
    δs = zeros(T, 3)

    # Stage 1: Analytic factor R(x), shadow dual s̃ = -F'(x)
    expbarr!(R, x)
    expbarrgrad!(ss, x)
    lmul!(-1, ss)

    # Stage 2: Shadow primal x̃ (Newton, uses R internally)
    expdualgrad!(xs, s)

    # Block-local μ and μ̃
    μv = dot3(x, s) / 3
    μt = dot3(xs, ss) / 3

    # Stage 3: Orthogonal completion z = x × x̃
    cross3!(z, x, xs)
    nz = norm3(z)

    # Change 3: Unified guard on centrality
    #
    # rel_z = ‖x × x̃‖ / (‖x‖·‖x̃‖) = sin(angle between x and x̃) is the
    # trustworthy centrality signal: it's geometric and doesn't route through
    # the gap = μμ̃-1 computation which suffers from cancellation.
    #
    # When rel_z < sqrt(eps) (≈1e-8), the iterate is close to the
    # central path and the δsδsᵀ/⟨δx,δs⟩ term in M becomes inaccurate due to
    # subtraction cancellation in δx = x - μx̃. The crossover where fallback
    # becomes better is at rel_z ~ 6.5e-9 (see compare_M_choices.jl).
    #
    # In the fallback, we use M = μF''(x) which is accurate (uses analytic
    # factor R) and positive definite.
    #
    nx = norm3(x)
    nxs = norm3(xs)
    rel_z = nz / (nx * nxs + eps(T))

    # Tighten threshold to only catch true central-path (rel_z ~ 1e-15),
    # not boundary corners where x ∥ x̃ by coincidence (rel_z ~ 1e-8).
    used_fallback = rel_z < eps(T)

    if used_fallback
        # On or near central path: M = μ F''(x) = μ R Rᵀ
        mul3!(M, R, R', μv, 0)
    else
        # Normalize z (safe since rel_z > sqrt(eps) implies nz > 0)
        ldiv3!(nz, z)

        # Stage 4: BFGS t via contracted form with cancellation-free d (Change 4+5)
        t = expbfgs(R, xs, ss, z, x, μv, μt)

        if !(t > 0) || !isfinite(t)
            # BFGS construction degraded off-central; μF'' is always PD.
            mul3!(M, R, R', μv, 0)
        else
            # Change 2: Closed form for M
            # M = ssᵀ/⟨x,s⟩ + δsδsᵀ/⟨δx,δs⟩ + tzzᵀ
            # where δs = s - μs̃

            xs_dot = 3 * μv  # ⟨x,s⟩

            # δx = x - μx̃, δs = s - μs̃
            copy3!(δx, x);  axpy3!(-μv, xs, δx)
            copy3!(δs, s);  axpy3!(-μv, ss, δs)

            # δ_dot = ⟨δx, δs⟩ = 3μ(μμ̃−1) > 0 off-central
            # Note: The ⟨δx,δs⟩ form has k≈1 error scaling, better than the μ⟨q,s̃⟩ form (k≈2).
            # Dotting two O(ε) vectors attenuates errors; dotting O(ε) with O(1) does not.
            # (Empirically verified via BigFloat reference harness.)
            δ_dot = dot3(δx, δs)

            ger3!(M, s, s, inv(xs_dot), 0)
            ger3!(M, δs, δs, inv(δ_dot), 1)
            ger3!(M, z, z, t, 1)
        end
    end

    return μv
end

function scale!(H::AbstractMatrix{T}, p::AbstractVector{T}, d::AbstractVector{T}, cache::ExponentialConeCache{T}) where {T}
    cache.μv[] = expscale!(cache.M, cache.R, cache.xs, cache.ss, p, d)
    copyto!(H, cache.M)
    return H
end

#
# Corrector (uses ldiv3! for F''⁻¹)
#
# r = -d - σμ·F'(p) - η
# where η = -½ F'''(p)[Δpₐ, F''(p)⁻¹Δdₐ]
#

function expcorr!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        p::AbstractVector{T},
        d::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real
    ) where {T}

    # Workspaces
    Fp = zeros(T, 3)
    v  = zeros(T, 3)
    η  = zeros(T, 3)
    D  = zeros(T, 3, 3)

    # F'(p)
    expbarrgrad!(Fp, p)

    # v = F''(p)⁻¹Δd = (RR')⁻¹Δd
    ldiv3!(v, R, Δd); ldiv3!(R', v)

    # η = -½ F'''(p)[Δp, v]
    expbarrhess!(D, p, Δp)
    mul3!(η, D, v, -0.5, 0)

    # r = -d - σμ·F'(p) - η
    copy3!(r, d)
    axpby3!(-σμ, Fp, -1, r)
    axpy3!(-1, η, r)

    return r
end

function corr!(
        r::AbstractVector{T},
        p::AbstractVector{T},
        d::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        cache::ExponentialConeCache{T}
    ) where {T}
    return expcorr!(r, cache.R, p, d, Δp, Δd, σμ)
end

#
# Max step by bisection on cone membership
#

function expmaxstep(incone, x::AbstractVector{T}, Δx::AbstractVector{T}, γ::Real) where {T}
    w = zeros(T, 3)

    τ = binarysearchlast(zero(T), one(T), eps(T), 53) do τ
        copy3!(w, x)
        axpy3!(τ, Δx, w)
        return incone(w)
    end

    return γ * τ
end

function maxsteps(p::AbstractVector{T}, Δp::AbstractVector{T}, d::AbstractVector{T}, Δd::AbstractVector{T}, γ::Real, ::ExponentialConeCache{T}) where {T}
    return expmaxstep(expincone, p, Δp, γ), expmaxstep(expindual, d, Δd, γ)
end
