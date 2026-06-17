#
# SOC cone (second-order / Lorentz cone)
#
# x = (x₀, x̄) ∈ SOC iff x₀ ≥ ‖x̄‖
#

struct SOC <: Cone end

struct SOCCache{T} <: AbstractCache{SOC}
    cone::SOC
    β::FScalarView{T}  # scaling factor (0-dim view)
    w::FVectorView{T}  # direction vector (satisfies w'Jw = 1)
end

# degree = 2 (always, regardless of dimension)
degree(::SOC, n::Int) = 2

# cache size: β(1) + w(n)
cachesize(::SOC, n::Int) = 1 + n

# construct view-based cache from Caches
function cache(c::Caches{T}, i::Int, cone::SOC) where T
    data = view(c.val, c.xblk[i]:c.xblk[i+1]-1)
    β = view(data, 1)
    w = view(data, 2:length(data))
    SOCCache(cone, β, w)
end

function identity!(x::AbstractVector{T}, ::SOC) where {T}
    # e = (1, 0, ..., 0)
    fill!(x, zero(T))
    x[1] = one(T)
    return x
end

#
# SOC helper functions
#

# determinant: det(x) = x₀² - ‖x̄‖²
function socdet(x::AbstractVector{T}) where {T}
    n = length(x); d = x[1]^2

    for i in 2:n
        d -= x[i]^2
    end

    return d
end

# SOC multiplication (Jordan product): out = x ∘ y
function socmul!(out::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    n = length(x)

    x1 = x[1]
    y1 = y[1]

    out[1] = dot(x, y)

    for i in 2:n
        out[i] = y1 * x[i] + x1 * y[i]
    end

    return out
end

# SOC division: b ← z \ b (solve L(z)b_new = b_old)
function socdiv!(z::AbstractVector{T}, b::AbstractVector{T}) where {T}
    n = length(z)
    δ = socdet(z)

    z1 = z[1]
    b1 = b[1] * z1

    for i in 2:n
        b1 -= z[i] * b[i]
    end

    b[1] = b1 /= δ

    for i in 2:n
        b[i] = (b[i] - b1 * z[i]) / z1
    end

    return b
end

# In-place H½ or H⁻½ application: x ← H½x (flag=false) or x ← H⁻½x (flag=true)
function socroot!(x::AbstractVector{T}, w::AbstractVector{T}, β::T, flag::Bool) where {T}
    n = length(x)

    if !flag
        σ = -one(T)
        α =  inv(β)
    else
        σ =  one(T)
        α =      β
    end

    w1p1 = w[1] + 1

    wpdx = w1p1 * x[1]

    for i in 2:n
        wpdx += σ * w[i] * x[i]
    end

    wpdx /= w1p1

    x[1] = α * (wpdx * w1p1 - x[1])

    for i in 2:n
        x[i] = α * (σ * wpdx * w[i] + x[i])
    end

    return x
end

function socscale!(
        w::AbstractVector{T},
        β::FScalarView{T},
        p::AbstractVector{T},
        d::AbstractVector{T}
    ) where {T}
    det_p = socdet(p)
    det_d = socdet(d)
    β[] = (det_p / det_d)^(1/4)

    sqrt_det_p = sqrt(det_p)
    sqrt_det_d = sqrt(det_d)

    n = length(p)
    s_dot_z = dot(p, d) / (sqrt_det_p * sqrt_det_d)
    sc = sqrt(2 * (1 + s_dot_z))
    w[1] = (p[1] / sqrt_det_p + d[1] / sqrt_det_d) / sc
    for i in 2:n
        w[i] = (p[i] / sqrt_det_p - d[i] / sqrt_det_d) / sc
    end

    return
end

function scale!(p::AbstractVector{T}, d::AbstractVector{T}, cache::SOCCache{T}) where {T}
    socscale!(cache.w, cache.β, p, d)
end

function sochess!(H::AbstractMatrix{T}, w::AbstractVector{T}, β::T) where {T}
    n = length(w)
    η = inv(β^2)
    w1 = w[1]

    H[1, 1] = 2η * w1^2 - η

    for i in 2:n
        H[i, 1] = -2η * w[i] * w1
    end

    for j in 2:n
        wj = w[j]
        H[j, j] = 2η * wj^2 + η
        for i in j + 1:n
            H[i, j] = 2η * w[i] * wj
        end
    end

    return H
end

function hess!(
        H::AbstractMatrix{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        cache::SOCCache{T}
    ) where {T}
    sochess!(H, cache.w, cache.β[])
end

function soccorr!(
        r::AbstractVector{T},
        w::AbstractVector{T},
        β::T,
        p::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real
    ) where {T}
    n = length(p)

    socroot!(Δp, w, β, false)
    socroot!(Δd, w, β, true)

    socmul!(r, Δp, Δd)

    copyto!(Δp, p)
    socroot!(Δp, w, β, false)

    socdiv!(Δp, r)

    axpy!(one(T), Δp, r)
    socroot!(r, w, β, false)

    pdet = socdet(p)

    r[1] = σμ * p[1] / pdet - r[1]

    for i in 2:n
        r[i] = -σμ * p[i] / pdet - r[i]
    end

    return r
end

function corr!(
        r::AbstractVector{T},
        p::AbstractVector{T},
        ::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        cache::SOCCache{T}
    ) where {T}
    soccorr!(r, cache.w, cache.β[], p, Δp, Δd, σμ)
end

function socmaxstep(x::AbstractVector{T}, Δx::AbstractVector{T}, γ::Real) where {T}
    n = length(x)

    # compute scalars
    #
    #   a = Δx J Δx
    #   b = 2x J Δx
    #   c =  x J  x
    #
    # such that that
    #
    #   det(x + Δx τ) = (x + Δx τ)ᵀ J (x + Δx τ)
    #                 = aτ² + bτ + c
    #
    a = socdet(Δx)
    c = socdet( x)

    b = x[1] * Δx[1]

    for i in 2:n
        b -= x[i] * Δx[i]
    end

    b *= 2

    # find the largest number τ < 1 such that
    #
    #   1. a τ² +   b τ + c  ≥ 0
    #   2.        Δx₁ τ + x₁ ≥ 0
    #
    τ = one(T)
    #
    # ensure that
    #
    #   a τ² + b τ + c  ≥ 0
    #
    if abs(a) > eps(T)
        #
        # d is the discriminant
        #
        #   d = b² - 4ac
        #
        d = b^2 - 4*a*c

        if d > -eps(T)
            #
            # - a > 0: roots have the same sign
            #          and τ1 is the smaller one
            #
            # - a < 0: roots have opposite signs
            #          and τ1 is the positive one
            #
            s = sqrt(max(d, zero(T)))
            q = -(b + copysign(s, b)) / 2

            τ1 = b ≥ 0 ? q / a : c / q

            if τ1 > 0
                τ = min(τ, γ * τ1)
            end
        end
    elseif b < -eps(T)
        τ = min(τ, -γ * c / b)
    end
    #
    # ensure that
    #
    #   Δx₁ τ + x₁ ≥ 0
    #
    if Δx[1] < 0
        τ = min(τ, -γ * x[1] / Δx[1])
    end

    return τ
end

function maxstep(
        x::AbstractVector{T},
        Δx::AbstractVector{T},
        ::Bool,
        γ::Real,
        ::SOCCache{T}
    ) where {T}
    socmaxstep(x, Δx, γ)
end
