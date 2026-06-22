struct EXP <: Cone end

struct EXPCache{T} <: AbstractCache{EXP}
    cone::EXP
    M::FMatrixView{T}     # scaling Gram matrix (3×3)
    R::FMatrixView{T}     # analytic factor of F''(x) (3×3)
    xs::FVectorView{T}    # shadow primal x̃ (3)
    ss::FVectorView{T}    # shadow dual s̃ = -F'(x) (3)
    μv::FScalarView{T}    # block-local μ = ⟨x,s⟩/3
end

# degree = 3 always (EXP is intrinsically 3D)
function degree(::EXP, n::Int)
    @assert n == 3 "EXP cone is 3-dimensional"
    return 3
end

# cache size: M(9) + R(9) + xs(3) + ss(3) + μv(1) = 25
function cachesize(::EXP, n::Int)
    @assert n == 3 "EXP cone is 3-dimensional"
    return 25
end

# construct view-based cache from Caches
function cache(c::Caches{T}, i::Int, cone::EXP) where T
    data = view(c.val, c.xblk[i]:c.xblk[i+1]-1)
    M  = reshape(view(data, 1:9), 3, 3)
    R  = reshape(view(data, 10:18), 3, 3)
    xs = view(data, 19:21)
    ss = view(data, 22:24)
    μv = view(data, 25)
    EXPCache(cone, M, R, xs, ss, μv)
end

# Central point on the exp cone central path
# From Dahl-Andersen: x* ≈ (1.290928, 0.805102, -0.827838)
function identity!(x::AbstractVector, ::EXP)
    x[1] =  1.2909282315382298
    x[2] =  0.8051015526498357
    x[3] = -0.8278379086082098
    return x
end

#
# 3×3 linear algebra helpers
#

# Cross product of two 3-vectors: z = x × y
function cross3!(z::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    z[1] = x[2] * y[3] - x[3] * y[2]
    z[2] = x[3] * y[1] - x[1] * y[3]
    z[3] = x[1] * y[2] - x[2] * y[1]
    return z
end

# Cross product returning new vector
function cross3(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    z = similar(x)
    return cross3!(z, x, y)
end

# 3×3 gemm: C = α*A*B + β*C (works for matrix-vector and matrix-matrix)
function mul3!(C::AbstractArray, A::AbstractArray, B::AbstractArray, α=1, β=0)
    n = size(B, 2)
    for j in 1:n, i in 1:3
        C[i,j] = α * (A[i,1] * B[1,j] + A[i,2] * B[2,j] + A[i,3] * B[3,j]) + β * C[i,j]
    end
    return C
end

# 3-element BLAS-style operations
function copy3(x::AbstractVector)
    return [x[1], x[2], x[3]]
end

function copy3!(y::AbstractVector, x::AbstractVector)
    y[1] = x[1]; y[2] = x[2]; y[3] = x[3]
    return y
end

function axpy3!(a, x::AbstractVector, y::AbstractVector)
    y[1] += a * x[1]; y[2] += a * x[2]; y[3] += a * x[3]
    return y
end

function axpby3!(a, x::AbstractVector, b, y::AbstractVector)
    y[1] = a * x[1] + b * y[1]
    y[2] = a * x[2] + b * y[2]
    y[3] = a * x[3] + b * y[3]
    return y
end

function scal3!(a, x::AbstractVector)
    x[1] *= a; x[2] *= a; x[3] *= a
    return x
end

function dot3(x::AbstractVector, y::AbstractVector)
    return x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
end

function norm3(x::AbstractVector)
    return sqrt(x[1]^2 + x[2]^2 + x[3]^2)
end

# M ← α x yᵀ + β M (3×3 rank-1 update)
@inline function ger3!(M, x, y, α, β)
    for j in 1:3, i in 1:3
        M[i,j] = α * x[i] * y[j] + β * M[i,j]
    end
    return M
end


# Solve 2×2 system [a b; c d] [x; y] = [e; f]
function solve2x2(a::T, b::T, c::T, d::T, e::T, f::T) where {T}
    det = a * d - b * c
    return (d * e - b * f) / det, (a * f - c * e) / det
end

#
# EXP barrier and derivatives
#
# Barrier: F(x) = -log(ψ(x)) - log(x₁) - log(x₂)
# where ψ(x) = x₂ log(x₁/x₂) - x₃
#

# Barrier argument ψ(x) = x₂ log(x₁/x₂) - x₃
function exp_psi(x::AbstractVector{T}) where {T}
    return x[2] * log(x[1] / x[2]) - x[3]
end

# Gradient of ψ: ψ'(x) = (x₂/x₁, log(x₁/x₂) - 1, -1)
function exp_psi_grad!(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    g[1] = x[2] / x[1]
    g[2] = log(x[1] / x[2]) - one(T)
    g[3] = -one(T)
    return g
end

# Barrier gradient: F'(x) = -ψ'(x)/ψ(x) - (1/x₁, 1/x₂, 0)
function exp_barrier_grad!(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    ψ = exp_psi(x)
    exp_psi_grad!(g, x)

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

function exp_barrier_factor!(R::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    x1, x2 = x[1], x[2]
    ψ = exp_psi(x)
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
# Solve F''(x) v = b using the analytic factor R (Change 1)
#
# Since F'' = R Rᵀ, we solve R w = b then Rᵀ v = w.
# Structure of R allows efficient 2×2 solves.
#
# Forward (R w = b):
#   w₃ = b₃ / r₃₃
#   solve 2×2 [r₁₁ r₁₂; r₂₁ r₂₂] [w₁; w₂] = [b₁ - r₁₃ w₃; b₂ - r₂₃ w₃]
#
# Backward (Rᵀ v = w):
#   solve 2×2 [r₁₁ r₂₁; r₁₂ r₂₂] [v₁; v₂] = [w₁; w₂]
#   v₃ = (w₃ - r₁₃ v₁ - r₂₃ v₂) / r₃₃
#

function expsolve!(R::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    r11, r12, r13 = R[1,1], R[1,2], R[1,3]
    r21, r22, r23 = R[2,1], R[2,2], R[2,3]
    r33 = R[3,3]

    # Forward: R w = b
    w3 = b[3] / r33
    w1, w2 = solve2x2(r11, r12, r21, r22, b[1] - r13 * w3, b[2] - r23 * w3)

    # Backward: Rᵀ v = w
    v1, v2 = solve2x2(r11, r21, r12, r22, w1, w2)
    v3 = (w3 - r13 * v1 - r23 * v2) / r33

    b[1], b[2], b[3] = v1, v2, v3
    return b
end

#
# Third-order directional derivative F'''(x)[u] as a 3×3 symmetric matrix
#

function exp_barrier_hess_dir!(D::AbstractMatrix{T}, x::AbstractVector{T}, u::AbstractVector{T}) where {T}
    ψ = exp_psi(x)
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

function in_exp_primal(x::AbstractVector{T}) where {T}
    x[1] > 0 && x[2] > 0 && exp_psi(x) > 0
end

function in_exp_dual(z::AbstractVector{T}) where {T}
    z[1] > 0 && z[3] < 0 && ℯ * z[1] >= -z[3] * exp(z[2] / z[3])
end

#
# Shadow primal computation (Newton iteration)
#
# Find x̃ such that F'(x̃) = -s via Newton with analytic factor (Change 1)
#

function exp_shadow_primal!(xs::AbstractVector{T}, s::AbstractVector{T}; maxiter::Int=50, tol::T=T(1e-12)) where {T}
    # Initialize near the central point
    xs[1] = T(1.0)
    xs[2] = T(1.0)
    xs[3] = T(-0.5)

    # Ensure starting point is interior
    while !in_exp_primal(xs)
        xs[1] *= 2
        xs[2] *= 2
        xs[3] -= one(T)
    end

    g = zeros(T, 3)
    R = zeros(T, 3, 3)
    Δ = zeros(T, 3)

    for iter in 1:maxiter
        # Residual: F'(x̃) + s
        exp_barrier_grad!(g, xs)
        axpy3!(1, s, g)

        if norm3(g) < tol
            return xs
        end

        # Newton step: Δ = -F''(x̃)⁻¹ (F'(x̃) + s)
        # Using analytic factor (Change 1)
        exp_barrier_factor!(R, xs)
        axpby!(-1, g, 0, Δ)
        expsolve!(R, Δ)

        # Line search
        θ = one(T)
        while θ > T(1e-10)
            xs_new = copy3(xs); axpy3!(θ, Δ, xs_new)
            if in_exp_primal(xs_new)
                copyto!(xs, xs_new)
                break
            end
            θ *= T(0.5)
        end

        if θ <= T(1e-10)
            @warn "Shadow primal Newton line search failed"
            break
        end
    end

    return xs
end

#
# BFGS t computation via R-products (Change 4)
#
# t_BFGS = μ ‖F'' - s̃s̃ᵀ/3 - vvᵀ/d‖_F
# where v = F''x̃ - μ̃s̃ and d = ⟨x̃, F''x̃⟩ - 3μ̃²
#
# All products use R: F''w = R(Rᵀw), ‖F''‖_F = ‖RᵀR‖_F, ⟨F'', wwᵀ⟩ = ‖Rᵀw‖²
#

function exp_bfgs_t(R::AbstractMatrix{T}, xs::AbstractVector{T}, ss::AbstractVector{T}, μv::T, μt::T) where {T}
    # Rᵀx̃
    Rtxs = zeros(T, 3)
    mul3!(Rtxs, R', xs)

    # F''x̃ = R(Rᵀx̃)
    Fppxs = zeros(T, 3)
    mul3!(Fppxs, R, Rtxs)

    # d = ⟨x̃, F''x̃⟩ - 3μ̃² = ‖Rᵀx̃‖² - 3μ̃²
    d = dot3(Rtxs, Rtxs) - 3 * μt^2

    # v = F''x̃ - μ̃s̃
    v = copy3(Fppxs)
    axpy3!(-μt, ss, v)

    # Rᵀs̃
    Rtss = zeros(T, 3)
    mul3!(Rtss, R', ss)

    # Rᵀv
    Rtv = zeros(T, 3)
    mul3!(Rtv, R', v)

    # ‖F''‖_F² = ‖RᵀR‖_F² = Σᵢⱼ (Σₖ Rₖᵢ Rₖⱼ)²
    # But simpler: ‖RᵀR‖_F² = tr((RᵀR)²) = ‖RRᵀ‖_F² = Σᵢⱼ F''ᵢⱼ²
    # We compute ‖RᵀR‖_F directly
    RtR = zeros(T, 3, 3)
    for j in 1:3, i in 1:3
        for k in 1:3
            RtR[i,j] += R[k,i] * R[k,j]
        end
    end
    norm_Fpp_sq = zero(T)
    for j in 1:3, i in 1:3
        norm_Fpp_sq += RtR[i,j]^2
    end

    # ‖s̃‖⁴/9
    ss_norm_sq = dot3(ss, ss)

    # ‖Rᵀs̃‖²
    Rtss_norm_sq = dot3(Rtss, Rtss)

    # ‖v‖⁴/d²
    v_norm_sq = dot3(v, v)

    # ‖Rᵀv‖²
    Rtv_norm_sq = dot3(Rtv, Rtv)

    # ⟨s̃, v⟩
    ss_dot_v = dot3(ss, v)

    # ‖A‖_F² = ‖F''‖_F² + ‖s̃‖⁴/9 + ‖v‖⁴/d² - (2/3)‖Rᵀs̃‖² - (2/d)‖Rᵀv‖² + (2/3d)⟨s̃,v⟩²
    if abs(d) > T(1e-14)
        norm_A_sq = norm_Fpp_sq + ss_norm_sq^2 / 9 + v_norm_sq^2 / d^2 -
                    (2/3) * Rtss_norm_sq - (2/d) * Rtv_norm_sq +
                    (2 / (3*d)) * ss_dot_v^2
    else
        # d ≈ 0 means on central path; skip the v term
        norm_A_sq = norm_Fpp_sq + ss_norm_sq^2 / 9 - (2/3) * Rtss_norm_sq
    end

    # Handle potential numerical issues
    norm_A_sq = max(norm_A_sq, zero(T))

    return μv * sqrt(norm_A_sq)
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

    # Stage 1: Analytic factor R(x), shadow dual s̃ = -F'(x)
    exp_barrier_factor!(R, x)
    exp_barrier_grad!(ss, x)
    lmul!(-1, ss)

    # Stage 2: Shadow primal x̃ (Newton, uses R internally)
    exp_shadow_primal!(xs, s)

    # Block-local μ and μ̃
    μv = dot3(x, s) / 3
    μt = dot3(xs, ss) / 3

    # Stage 3: Orthogonal completion z = x × x̃
    z = cross3(x, xs)
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

    used_fallback = rel_z < sqrt(eps(T))

    if used_fallback
        # On or near central path: M = μ F''(x) = μ R Rᵀ
        mul3!(M, R, R', μv, 0)
    else
        # Normalize z (safe since rel_z > sqrt(eps) implies nz > 0)
        ldiv!(nz, z)

        # Stage 4: BFGS t via R-products (Change 4)
        t = exp_bfgs_t(R, xs, ss, μv, μt)

        # Change 2: Closed form for M
        # M = ssᵀ/⟨x,s⟩ + δsδsᵀ/⟨δx,δs⟩ + tzzᵀ
        # where δx = x - μx̃, δs = s - μs̃

        xs_dot = 3 * μv  # ⟨x,s⟩

        δx = copy3(x);  axpy3!(-μv, xs, δx)
        δs = copy3(s);  axpy3!(-μv, ss, δs)
        δ_dot = dot3(δx, δs)  # ⟨δx,δs⟩

        ger3!(M, s, s, inv(xs_dot), 0)
        ger3!(M, δs, δs, inv(δ_dot), 1)
        ger3!(M, z, z, t, 1)
    end

    return μv
end

function scale!(p::AbstractVector{T}, d::AbstractVector{T}, cache::EXPCache{T}) where {T}
    cache.μv[] = expscale!(cache.M, cache.R, cache.xs, cache.ss, p, d)
end

#
# Hessian: copy cached M
#

function exphess!(H::AbstractMatrix{T}, M::AbstractMatrix{T}) where {T}
    copyto!(H, M)
    return H
end

function hess!(
        H::AbstractMatrix{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        cache::EXPCache{T}
    ) where {T}
    exphess!(H, cache.M)
    return H
end

#
# Corrector (uses expsolve! for F''⁻¹, Change 1)
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

    # F'(p)
    Fp = zeros(T, 3)
    exp_barrier_grad!(Fp, p)

    # v = F''(p)⁻¹Δd using expsolve! (Change 1)
    v = copy(Δd)
    expsolve!(R, v)

    # η = -½ F'''(p)[Δp, v]
    D = zeros(T, 3, 3)
    exp_barrier_hess_dir!(D, p, Δp)

    η = zeros(T, 3)
    mul3!(η, D, v)
    ldiv!(-2, η)

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
        cache::EXPCache{T}
    ) where {T}
    return expcorr!(r, cache.R, p, d, Δp, Δd, σμ)
end

#
# Max step by bisection on cone membership
#

function expmaxstep(x::AbstractVector{T}, Δx::AbstractVector{T}, primal::Bool, γ::Real) where {T}
    membership = primal ? in_exp_primal : in_exp_dual

    τ_lo = zero(T)
    τ_hi = one(T)

    x_test = similar(x)

    # Check if full step is feasible
    copyto!(x_test, x)
    axpy!(τ_hi, Δx, x_test)
    if membership(x_test)
        return one(T)  # boundary is beyond 1, take full step
    end

    # Bisection
    for _ in 1:53
        τ_mid = (τ_lo + τ_hi) / 2
        copyto!(x_test, x)
        axpy!(τ_mid, Δx, x_test)

        if membership(x_test)
            τ_lo = τ_mid
        else
            τ_hi = τ_mid
        end

        if τ_hi - τ_lo < eps(T)
            break
        end
    end

    return γ * τ_lo
end

function maxstep(
        x::AbstractVector{T},
        Δx::AbstractVector{T},
        primal::Bool,
        γ::Real,
        ::EXPCache{T}
    ) where {T}
    return expmaxstep(x, Δx, primal, γ)
end
