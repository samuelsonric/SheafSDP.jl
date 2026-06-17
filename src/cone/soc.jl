#
# SOC cone (second-order / Lorentz cone)
#
# x = (x₀, x̄) ∈ SOC iff x₀ ≥ ‖x̄‖
#

struct SOC <: Cone end

# View-based cache for SOC
struct SOCCache{T}
    β::FScalarView{T}  # scaling factor (0-dim view)
    w::FVectorView{T}  # direction vector (satisfies w'Jw = 1)
end

# degree = 2 (always, regardless of dimension)
degree(::SOC, n::Int) = 2

# cache size: β(1) + w(n)
cache_size(::SOC, n::Int) = 1 + n

# construct view-based cache from Caches
function cache(c::Caches{T}, i::Int, ::SOC) where T
    data = view(c.val, c.xblk[i]:c.xblk[i+1]-1)
    β = view(data, 1)
    w = view(data, 2:length(data))
    SOCCache(β, w)
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

# SOC Jordan product: (x ∘ y)₀ = ⟨x,y⟩, (x ∘ y)ᵢ = x₀yᵢ + y₀xᵢ
function jordan_prod!(out::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    n = length(x)

    x1 = x[1]
    y1 = y[1]

    out[1] = dot(x, y)

    for i in 2:n
        out[i] = y1 * x[i] + x1 * y[i]
    end

    return out
end

# In-place arrow inverse: solve L(z)b = b_old, i.e., b ← L(z)⁻¹b
# L(z)e = z, so L(z)⁻¹z = e
function arrow_inv!(z::AbstractVector{T}, b::AbstractVector{T}) where {T}
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
function apply_H_half!(x::AbstractVector{T}, cache::SOCCache{T}, flag::Bool) where {T}
    n = length(x)
    w = cache.w
    β = cache.β[]

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

function update_scaling!(cache::SOCCache{T}, ::SOC,
                         p::AbstractVector{T}, d::AbstractVector{T}) where {T}
    # β = (det(p) / det(d))^{1/4}
    det_p = socdet(p)
    det_d = socdet(d)
    cache.β[] = (det_p / det_d)^(1/4)

    # Normalized: s̃ = p/√det(p), z̃ = d/√det(d)
    sqrt_det_p = sqrt(det_p)
    sqrt_det_d = sqrt(det_d)

    # w = (s̃ + J z̃) / √(2(1 + s̃·z̃))
    # Note: s̃·z̃ is the ordinary Euclidean dot product, not the J-inner product
    # This follows from: ‖s̃ + Jz̃‖_J² = det(s̃) + 2s̃ᵀz̃ + det(z̃) = 2(1 + s̃·z̃)
    n = length(p)
    w = cache.w

    # Compute s̃·z̃ (ordinary Euclidean dot product)
    s_dot_z = dot(p, d) / (sqrt_det_p * sqrt_det_d)

    # w = (s̃ + J z̃) / √(2(1 + s̃·z̃))
    scale = sqrt(2 * (1 + s_dot_z))
    w[1] = (p[1] / sqrt_det_p + d[1] / sqrt_det_d) / scale
    for i in 2:n
        w[i] = (p[i] / sqrt_det_p - d[i] / sqrt_det_d) / scale
    end

    return
end

function hessian_block!(H::AbstractMatrix{T}, cache::SOCCache{T}, ::SOC) where {T}
    n = length(cache.w)
    w = cache.w
    β = cache.β[]
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

function corrector_term!(rc::AbstractVector{T}, cache::SOCCache{T}, ::SOC,
                         p::AbstractVector{T}, d::AbstractVector{T},
                         Δp::AbstractVector{T}, Δd::AbstractVector{T},
                         σμ::Real) where {T}
    # Full SOC corrector: r_c = σμ·z⁻¹ - s - H⁻½ L(λ)⁻¹(d_s ∘ d_z)
    # where λ = H½ s, d_s = H½ Δs, d_z = H⁻½ Δz
    # Note: Δp, Δd are overwritten (not needed after this function returns)
    n = length(p)

    # Transform Δp, Δd in-place to d_s, d_z
    apply_H_half!(Δp, cache, false)  # Δp ← H½ Δp = d_s
    apply_H_half!(Δd, cache, true)   # Δd ← H⁻½ Δd = d_z

    # rc = d_s ∘ d_z (Jordan product)
    jordan_prod!(rc, Δp, Δd)

    # Reuse Δp for λ = H½ p (zero allocations)
    copyto!(Δp, p)
    apply_H_half!(Δp, cache, false)

    # rc ← L(λ)⁻¹ rc (in-place arrow inverse)
    arrow_inv!(Δp, rc)

    # rc ← H⁻½ rc (in-place)
    apply_H_half!(rc, cache, true)

    # r_c = σμ·d⁻¹ - p - rc
    # d⁻¹ = Jd / det(d)
    ddet = socdet(d)

    rc[1] = σμ * d[1] / ddet - p[1] - rc[1]

    for i in 2:n
        rc[i] = -σμ * d[i] / ddet - p[i] - rc[i]
    end

    return rc
end

function max_step(cache::SOCCache{T},
                  x::AbstractVector{T}, Δx::AbstractVector{T},
                  primal::Bool, γ::Real) where {T}
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

            if b ≥ 0
                τ1 = q / a
            else
                τ1 = c / q
            end

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
