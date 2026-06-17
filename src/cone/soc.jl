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

function identity!(x::AbstractVector{T}, ::SOC, ξ::Real, uplo::Val) where {T}
    # e = (1, 0, ..., 0)
    fill!(x, zero(T))
    x[1] = T(ξ)
    return x
end

#
# SOC helper functions
#

# J-inner product: jdot(a,b) = a₀b₀ - ā·b̄
function jdot(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    a[1] * b[1] - dot(view(a, 2:length(a)), view(b, 2:length(b)))
end

# determinant: det(x) = x₀² - ‖x̄‖² = jdot(x, x)
det_soc(x::AbstractVector) = jdot(x, x)

# J * x = (x₀, -x̄)
function jmul!(y::AbstractVector{T}, x::AbstractVector{T}) where {T}
    y[1] = x[1]
    for i in 2:length(x)
        y[i] = -x[i]
    end
    return y
end

function jmul(x::AbstractVector{T}) where {T}
    y = similar(x)
    jmul!(y, x)
end

# SOC Jordan product: (x ∘ y)₀ = jdot(x,y), (x ∘ y)ᵢ = x₀yᵢ + y₀xᵢ
function jordan_prod!(out::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    out[1] = jdot(x, y)
    x0, y0 = x[1], y[1]
    for i in 2:length(x)
        out[i] = x0 * y[i] + y0 * x[i]
    end
    return out
end

# Arrow inverse: solve L(z)u = b where L(z) is z's arrow matrix
# L(z)e = z, so L(z)⁻¹z = e
function arrow_inv!(u::AbstractVector{T}, z::AbstractVector{T}, b::AbstractVector{T}) where {T}
    n = length(z)
    δ = det_soc(z)
    z0 = z[1]

    # u₀ = (z₀·b₀ - z̄·b̄) / δ
    u0 = (z0 * b[1] - dot(view(z, 2:n), view(b, 2:n))) / δ
    u[1] = u0

    # ū = (b̄ - u₀·z̄) / z₀
    for i in 2:n
        u[i] = (b[i] - u0 * z[i]) / z0
    end
    return u
end

# Apply H½ or H⁻½ to x
# H = η(2aaᵀ - J) where a = Jw
# Half-rapidity: a' = (√((a₀+1)/2), ā/√(2(a₀+1)))
# H½ = √η (2a'a'ᵀ - J)
# H⁻½ = (1/√η)(2(Ja')(Ja')ᵀ - J)
function apply_H_half!(out::AbstractVector{T}, cache::SOCCache{T}, x::AbstractVector{T}, inverse::Bool) where {T}
    n = length(x)
    β = cache.β[]
    η = inv(β^2)
    sqrt_η = inv(β)

    # a = Jw
    w = cache.w
    a0 = w[1]  # (Jw)₀ = w₀

    # a' = (√((a₀+1)/2), ā/√(2(a₀+1)))
    denom = sqrt(2 * (a0 + 1))
    a_prime_0 = sqrt((a0 + 1) / 2)

    if inverse
        # H⁻½ = (1/√η)(2(Ja')(Ja')ᵀ - J)
        # Ja' = (a'₀, -a'̄) = (a_prime_0, a'̄ with sign flip)
        # (Ja')ᵀx = a_prime_0 * x₀ + Σᵢ>₀ (-a'ᵢ) * xᵢ
        #         = a_prime_0 * x₀ + Σᵢ>₀ (wᵢ/denom) * xᵢ  (since a'ᵢ = -wᵢ/denom for i>0)
        Ja_prime_dot_x = a_prime_0 * x[1]
        for i in 2:n
            Ja_prime_dot_x += (w[i] / denom) * x[i]  # -aᵢ = -(-wᵢ) = wᵢ, then a'ᵢ = aᵢ/denom
        end

        # H⁻½ x = (1/√η)(2(Ja')(Ja'ᵀx) - Jx)
        coeff = 2 * Ja_prime_dot_x / sqrt_η
        out[1] = coeff * a_prime_0 - x[1] / sqrt_η
        for i in 2:n
            # (Ja')ᵢ = -a'ᵢ = -(-wᵢ/denom) = wᵢ/denom
            out[i] = coeff * (w[i] / denom) + x[i] / sqrt_η  # -(-xᵢ) = +xᵢ
        end
    else
        # H½ = √η (2a'a'ᵀ - J)
        # a'ᵀx = a_prime_0 * x₀ + Σᵢ>₀ a'ᵢ * xᵢ
        # a'ᵢ = aᵢ/denom = -wᵢ/denom for i > 0
        a_prime_dot_x = a_prime_0 * x[1]
        for i in 2:n
            a_prime_dot_x += (-w[i] / denom) * x[i]
        end

        # H½ x = √η(2a'(a'ᵀx) - Jx)
        coeff = 2 * sqrt_η * a_prime_dot_x
        out[1] = coeff * a_prime_0 - sqrt_η * x[1]
        for i in 2:n
            out[i] = coeff * (-w[i] / denom) + sqrt_η * x[i]  # -(-xᵢ) = +xᵢ
        end
    end
    return out
end

function update_scaling!(cache::SOCCache{T}, ::SOC,
                         p::AbstractVector{T}, d::AbstractVector{T}, uplo::Val) where {T}
    # β = (det(p) / det(d))^{1/4}
    det_p = det_soc(p)
    det_d = det_soc(d)
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

function hessian_block!(H::AbstractMatrix{T}, cache::SOCCache{T}, ::SOC, uplo::Val{UPLO}) where {UPLO, T}
    n = length(cache.w)
    w = cache.w
    β = cache.β[]
    η = inv(β^2)

    fill!(H, zero(T))

    for j in 1:n
        if isone(j)
            αj =  one(T)
        else
            αj = -one(T)
        end

        ηαjwj = 2η * αj * w[j]

        if UPLO === :L
            r = j:n
        else
            r = 1:j
        end

        if UPLO === :L
            H[j, j] -= αj * η
        end

        for i in r
            if isone(i)
                αi =  one(T)
            else
                αi = -one(T)
            end

            H[i, j] += αi * w[i] * ηαjwj
        end

        if UPLO === :U
            H[j, j] -= αj * η
        end
    end

    return H
end

function corrector_term!(rc::AbstractVector{T}, cache::SOCCache{T}, ::SOC,
                         p::AbstractVector{T}, d::AbstractVector{T},
                         Δp::AbstractVector{T}, Δd::AbstractVector{T},
                         σμ::Real, uplo::Val) where {T}
    # Full SOC corrector: r_c = σμ·z⁻¹ - s - H⁻½ L(λ)⁻¹(d_s ∘ d_z)
    # where λ = H½ s, d_s = H½ Δs, d_z = H⁻½ Δz
    n = length(p)

    # Allocate temporaries
    λ = similar(p)
    d_s = similar(p)
    d_z = similar(p)
    t = similar(p)
    q = similar(p)
    H_inv_half_q = similar(p)

    # λ = H½ s (scaled point)
    apply_H_half!(λ, cache, p, false)

    # d_s = H½ Δp, d_z = H⁻½ Δd (affine dirs in v-space)
    apply_H_half!(d_s, cache, Δp, false)
    apply_H_half!(d_z, cache, Δd, true)

    # t = d_s ∘ d_z (Jordan product)
    jordan_prod!(t, d_s, d_z)

    # q = L(λ)⁻¹ t (arrow inverse)
    arrow_inv!(q, λ, t)

    # H⁻½ q
    apply_H_half!(H_inv_half_q, cache, q, true)

    # r_c = σμ·d⁻¹ - p - H⁻½ q
    # d⁻¹ = Jd / det(d)
    det_d = det_soc(d)
    rc[1] = σμ * d[1] / det_d - p[1] - H_inv_half_q[1]
    for i in 2:n
        rc[i] = -σμ * d[i] / det_d - p[i] - H_inv_half_q[i]
    end

    return rc
end

function max_step(cache::SOCCache{T}, ::SOC,
                  x::AbstractVector{T}, Δx::AbstractVector{T},
                  primal::Bool, γ::Real, uplo::Val) where {T}
    # det(x + τΔx) = aτ² + bτ + c where
    # a = jdot(Δx, Δx) = det(Δx)
    # b = 2·jdot(x, Δx)
    # c = det(x) > 0
    a = det_soc(Δx)
    b = 2 * jdot(x, Δx)
    c = det_soc(x)

    τ = one(T)

    # Quadratic constraint from determinant
    if abs(a) > eps(T)
        disc = b^2 - 4*a*c
        if disc >= 0
            sqrt_disc = sqrt(disc)
            if a > 0
                τ1 = (-b - sqrt_disc) / (2*a)
                τ2 = (-b + sqrt_disc) / (2*a)
                if τ1 > 0
                    τ = min(τ, γ * τ1)
                elseif τ2 > 0
                    τ = min(τ, γ * τ2)
                end
            else
                τ1 = (-b - sqrt_disc) / (2*a)
                τ2 = (-b + sqrt_disc) / (2*a)
                if τ1 > 0 && τ2 > 0
                    τ = min(τ, γ * min(τ1, τ2))
                elseif τ1 > 0
                    τ = min(τ, γ * τ1)
                elseif τ2 > 0
                    τ = min(τ, γ * τ2)
                end
            end
        end
    elseif abs(b) > eps(T)
        τ_lin = -c / b
        if τ_lin > 0
            τ = min(τ, γ * τ_lin)
        end
    end

    # Linear constraint: x₀ + τΔx₀ ≥ 0
    if Δx[1] < 0
        τ = min(τ, -γ * x[1] / Δx[1])
    end

    return τ
end
