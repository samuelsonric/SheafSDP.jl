"""
    ExponentialCone <: Cone

The exponential cone, consisting of all triples (x, y, z)
such that x > 0, y > 0, and y log(x/y) ≥ z.
"""
struct ExponentialCone <: Cone end

struct ExponentialConeCache{T} <: AbstractCache{ExponentialCone}
    cone::ExponentialCone
    #
    # The analytic factor of the barrier
    # Hessian
    #
    #   f''(p) = R Rᵀ
    #
    R::FMatrixView{T}
    #
    # The dual "shadow" iterate
    #
    #   d* = -f'(p)
    #
    ss::FVectorView{T}
    #
    # warm-start for computing the primal
    # "shadow" iterate, which solves
    #
    #   -f'(p*) = d 
    #
    x2::FScalarView{T}
end

function degree(::ExponentialCone, n::Int)
    @assert n == 3
    return 3
end

function cachesize(::ExponentialCone, n::Int)
    @assert n == 3
    return 13
end

function cache(c::Caches{T}, i::Int, cone::ExponentialCone) where T
    data = cachedata(c, i)
    R  = reshape(view(data, 1:9), 3, 3)
    ss = view(data, 10:12)
    x2 = view(data, 13)
    ExponentialConeCache(cone, R, ss, x2)
end

# compute the fixed point
#
#   f'(e) = -e
#
function expid!(x::AbstractVector)
    x[1] =  1.2909282315382298
    x[2] =  0.8051015526498357
    x[3] = -0.8278379086082098
    return x
end

function identity!(x::AbstractVector, ::ExponentialCone)
    return expid!(x)
end

function initcache!(cache::ExponentialConeCache{T}) where {T}
    cache.x2[] = 0.8051015526498357
    return cache
end

# the barrier argument
#
#   ψ(x) = x₂ log(x₁/x₂) - x₃
#
function exppsi(x::AbstractVector)
    return x[2] * log(x[1] / x[2]) - x[3]
end

# the gradient of ψ
#
#   ψ'(x) = (x₂/x₁, log(x₁/x₂) - 1, -1)
#
function exppsigrad!(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    g[1] = x[2] / x[1]
    g[2] = log(x[1] / x[2]) - one(T)
    g[3] = -one(T)
    return g
end

# The gradient
#
#   f'(x) = -ψ'(x)/ψ(x) - (1/x₁, 1/x₂, 0)
#
# of the barrier function.
function expbarrgrad!(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    ψ = exppsi(x)
    exppsigrad!(g, x)

    g[1] = -g[1] / ψ - inv(x[1])
    g[2] = -g[2] / ψ - inv(x[2])
    g[3] = -g[3] / ψ

    return g
end

# Compute the analytic factor R
#
#   f''(p) = R Rᵀ
#
# if the barrier function f.
function expbarr!(R::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    x1, x2 = x[1], x[2]

    ψ = exppsi(x)

    σ = sqrt(one(T) + 2x2 / ψ)

    R[1,1] = (one(T) - σ) / 2x1
    R[1,2] = (one(T) + σ) / 2x1
    R[2,1] = (one(T) + σ) / 2x2
    R[2,2] = (one(T) - σ) / 2x2

    R[1,3] = x2 / x1 / ψ
    R[2,3] = (log(x1 / x2) - one(T)) / ψ
    R[3,3] = -inv(ψ)

    R[3,1] = zero(T)
    R[3,2] = zero(T)

    return R
end


# Compute the third-order directional derivative
#
#   f'''(x)[u]
#
# as a 3x3 matrix.
function expbarrhess!(D::AbstractMatrix{T}, x::AbstractVector{T}, u::AbstractVector{T}) where {T}
    ψ = exppsi(x)

    x1, x2     = x[1], x[2]
    u1, u2, u3 = u[1], u[2], u[3]

    ψg1 = x2 / x1
    ψg2 = log(x1 / x2) - one(T)
    ψgu = ψg1 * u1 + ψg2 * u2 - u3

    ψH11 = -x2 / x1^2
    ψH21 =  inv(x1)
    ψH22 = -inv(x2)

    ψHu1 = ψH11 * u1 + ψH21 * u2
    ψHu2 = ψH21 * u1 + ψH22 * u2

    ψ2 = ψ^2
    ψ3 = ψ^3

    α = -2ψgu / ψ3

    D[1,1] =  α * ψg1^2 + (2ψg1 * ψHu1 + ψH11 * ψgu) / ψ2 - (2x2 * u1 / x1^3 - u2 / x1^2) / ψ - 2u1 / x1^3
    D[2,1] =  α * ψg1 * ψg2 + (ψg1 * ψHu2 + ψg2 * ψHu1 + ψH21 * ψgu) / ψ2 + u1 / x1^2 / ψ
    D[3,1] = -α * ψg1 - ψHu1 / ψ2
    D[2,2] =  α * ψg2^2 + (2ψg2 * ψHu2 + ψH22 * ψgu) / ψ2 - u2 / x2^2 / ψ - 2u2 / x2^3
    D[3,2] = -α * ψg2 - ψHu2 / ψ2
    D[3,3] =  α

    D[1,2] = D[2,1]
    D[1,3] = D[3,1]
    D[2,3] = D[3,2]

    return D
end

# Determine if x is in the exponential cone.
function expincone(x::AbstractVector)
    x[1] > 0 && x[2] > 0 && exppsi(x) > 0
end

# Determine if z is in the dual exponential cone.
function expindual(z::AbstractVector)
    z[1] > 0 && z[3] < 0 && ℯ * z[1] >= -z[3] * exp(z[2] / z[3])
end

# compute the "shadow" primal, solving
#
#   f'(p*) = -d
#
# using a 1-D scalar root-find on the function
#
#   h(p₂*) = d₃ (log(p₁* / p₂*) - 1) - 1 / p₂* + d₂
#
# with
#
#   p₁* = (1 - p₂* d₃) / d₁
#   p₃* = p₂* log(p₁* / p₂*) + 1 / d₃
#
function expdualgrad!(xs::AbstractVector{T}, seed::T, d::AbstractVector{T}) where {T}
    d1, d2, d3 = d[1], d[2], d[3]

    function h(x2)
        d3 * (log((one(T) - x2 * d3) / (d1 * x2)) - one(T)) - inv(x2) + d2
    end

    function hp(x2)
        -d3^2 / (one(T) - x2 * d3) - d3 / x2 + inv(x2^2)
    end
    #
    # bracket the function h, finding points lo and hi
    # such that
    #
    #   h(lo) < 0 < h(hi)
    #
    if h(seed) < 0
        lo = seed
        hi = seed * 2

        while h(hi) < 0
            hi *= 2
        end
    else
        hi = seed
        lo = seed / 2

        while h(lo) > 0
            lo /= 2
        end
    end
    #
    # find p₂* such that
    #
    #   h(p₂*) = 0
    #
    x2 = rtsafe(h, hp, lo, hi, seed)
    #
    # recover p* as
    #
    #   p₁* = (1 - p₂* d₃) / d₁
    #   p₃* = p₂* log(p₁* / p₂*) + 1 / d₃
    #
    xs[1] = x1 = (one(T) - x2 * d3) / d1
    xs[2] = x2
    xs[3] = x2 * log(x1 / x2) + inv(d3)

    return x2
end

# Compute the coefficient t in the rank-1 term
# tzzᵀ of the Tuncel scaling matrix M:
#
#   M = ⟨p, d⟩⁻¹   d  dᵀ
#     + ⟨δp,δd⟩⁻¹ δd δdᵀ
#     + t          z  zᵀ,
#
# where
#
#   δp = p - μ p*
#   δd = d - μ d*
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
    t = zero(T)

    w   = zeros(T, 3)
    Rtw = zeros(T, 3)
    Rtz = zeros(T, 3)
    #
    # compute the gap direction
    #
    #   w = p* - μ* p
    #
    # where is the dual centrality parameter
    #
    #   μ* = ⟨p*, d*⟩ / ν.
    #
    copy3!(w, xs)
    axpy3!(-μt, x, w)
    #
    # compute the norm
    #
    #   ⟨w, f''(p) w⟩ = ‖Rᵀ w‖²
    #
    mul3!(Rtw, R', w)
    d = dot3(Rtw, Rtw)

    if d > 0
        #
        # compute the norm
        #
        #   ⟨z, f''(p) z⟩ = ‖Rᵀ z‖²
        #
        mul3!(Rtz, R', z)
        fppzz = dot3(Rtz, Rtz)
        #
        # compute the dot product
        #
        #   ⟨d*, z⟩
        #
        sz = dot3(ss, z)
        #
        # compute the dot product
        #
        #   ⟨w, f''(p) z⟩ = ⟨Rᵀ w, Rᵀ z⟩
        #
        pz = dot3(Rtw, Rtz)
        #
        # compute t:
        #
        #   t = μ ⟨z, f''(p) z⟩
        #     - μ ⟨d*,       z⟩² / ν
        #     - μ ⟨w, f''(p) z⟩² / ⟨w, f''(p) w⟩
        #
        t = μv * (fppzz - sz^2 / 3 - pz^2 / d)
    end

    return t
end

# Assemble the Tuncel scaling matrix
#
#   M = ⟨p, d⟩⁻¹   d  dᵀ
#     + ⟨δp,δd⟩⁻¹ δd δdᵀ
#     + t          z  zᵀ,
#
# where
#
#   δp = p - μ p*
#   δd = d - μ d*
#
function expscale!(
        H::AbstractMatrix{T},
        R::AbstractMatrix{T},
        ss::AbstractVector{T},
        x2_seed::T,
        x::AbstractVector{T},
        s::AbstractVector{T}
    ) where {T}

    xs = zeros(T, 3)
    z  = zeros(T, 3)
    δx = zeros(T, 3)
    δs = zeros(T, 3)
    #
    # compute the analytic factor
    #
    #   f''(p) = R Rᵀ
    #
    expbarr!(R, x)
    #
    # compute the "shadow" dual
    #
    #   d* = -f'(p)
    #
    expbarrgrad!(ss, x)
    lmul3!(-1, ss)
    #
    # compute the "shadow" primal, solving
    #
    #   d = -f'(p*)
    #
    x2_new = expdualgrad!(xs, x2_seed, s)
    #
    # compute the centrality parameters
    #
    #   μ  = ⟨p,  d ⟩ / ν
    #   μ* = ⟨p*, d*⟩ / ν
    #
    μv = dot3(x,  s)  / 3
    μt = dot3(xs, ss) / 3
    #
    # compute the cross-product
    #
    #   z = p × p*
    #
    cross3!(z, x, xs)
    #
    # compute the sine of the angle θ between p and p*:
    #
    #   ‖p × p*‖ / (‖p‖ ‖p*‖) = sin(θ)
    #
    # when this quantity is small, the iterate is close to
    # the central path and the term δd δdᵀ / ⟨δp, δd⟩ term in M
    # becomes innaccurate due to cancellation in the difference
    #
    #   δx = p - μ p*
    #
    # in this case, we fall back to the approximation
    #
    #   M ≈ μ f''(p)
    #
    nz  = norm3(z)
    nx  = norm3(x)
    nxs = norm3(xs)

    if nz < eps(T) * (nx * nxs + eps(T))
        #
        # approximate M by
        #
        #    M ≈ μ f''(p) = μ R Rᵀ
        #
        mul3!(H, R, R', μv, 0)
    else
        #
        # normalize z:
        #
        #   z = z / ‖z‖
        #
        ldiv3!(nz, z)
        #
        # compute the coefficent t in the rank-1 term
        #
        #   tzzᵀ
        #
        t = expbfgs(R, xs, ss, z, x, μv, μt)

        if t ≤ 0 || !isfinite(t)
            #
            # approximate M by
            #
            #    M ≈ μ f''(p) = μ R Rᵀ
            #
            mul3!(H, R, R', μv, 0)
        else
            #
            # construct M:
            #
            #   M = ⟨p, d⟩⁻¹   d  dᵀ
            #     + ⟨δp,δd⟩⁻¹ δd δdᵀ
            #     + t          z  zᵀ,
            #
            copy3!(δx, x); axpy3!(-μv, xs, δx)
            copy3!(δs, s); axpy3!(-μv, ss, δs)

            xs_dot = 3μv
            δ_dot  = dot3(δx, δs)

            ger3!(H,  s,  s, inv(xs_dot), 0)
            ger3!(H, δs, δs, inv(δ_dot),  1)
            ger3!(H,  z,  z, t,           1)
        end
    end

    return x2_new
end

function scale!(H::AbstractMatrix{T}, p::AbstractVector{T}, d::AbstractVector{T}, cache::ExponentialConeCache{T}) where {T}
    cache.x2[] = expscale!(H, cache.R, cache.ss, cache.x2[], p, d)
    return H
end

# Compute the Mehrotra corrector term
#
#   -d - σμ f'(p) - η,
#
# where η is the third-order correction
#
#   η = -½ f'''(p)[Δp, f''(p)⁻¹ Δd].
#
function expcorr!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        ss::AbstractVector{T},
        p::AbstractVector{T},
        d::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real
    ) where {T}

    v  = zeros(T, 3)
    η  = zeros(T, 3)
    D  = zeros(T, 3, 3)
    #
    # solve for v in
    #
    #   f''(p) v =  Δd
    #
    # using the analytic factorization
    #
    #   f''(p) = R Rᵀ.
    #
    ldiv3!(v, R, Δd); ldiv3!(R', v)
    #
    # compute the third-order correction
    #
    #   η = -½ f'''(p)[Δp, v]
    #
    expbarrhess!(D, p, Δp)
    mul3!(η, D, v, -0.5, 0)
    #
    # compute the Mehrotra corrector term
    #
    #   -d - σμ f'(p) - η
    #
    copy3!(r, d)
    axpby3!(σμ, ss, -1, r)
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
    return expcorr!(r, cache.R, cache.ss, p, d, Δp, Δd, σμ)
end

# use bisection to find the largest number 0 < τ ≤ 1\
# such that
#
#   x + τ Δx
#
# is in the exponential cone (or its dual).
function expmaxstep(incone, x::AbstractVector{T}, Δx::AbstractVector{T}) where {T}
    w = zeros(T, 3)

    τ = binarysearchlast(zero(T), one(T), eps(T), 53) do τ
        copy3!(w, x)
        axpy3!(τ, Δx, w)
        return incone(w)
    end

    return τ
end

function maxsteps(p::AbstractVector{T}, Δp::AbstractVector{T}, d::AbstractVector{T}, Δd::AbstractVector{T}, ::ExponentialConeCache{T}) where {T}
    return expmaxstep(expincone, p, Δp), expmaxstep(expindual, d, Δd)
end
