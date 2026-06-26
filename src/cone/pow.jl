"""
    PowerCone{T} <: Cone
 
The three-dimensional power cone with parameter α ∈ (0, 1),
consisting of all triples (x₁, x₂, x₃) such that
x₁ ≥ 0, x₂ ≥ 0, and x₁^α x₂^(1-α) ≥ |x₃|.
"""
struct PowerCone{T} <: Cone
    α::T

    function PowerCone{T}(α) where {T}
        @assert 0 < α < 1
        return new{T}(α)
    end
end

function PowerCone(α::T) where {T}
    return PowerCone{T}(α)
end

struct PowerConeCache{T} <: AbstractCache{PowerCone{T}}
    cone::PowerCone{T}
    #
    # The permuted Cholesky factor of the
    # barrier Hessian
    #
    #   P f''(p) Pᵀ = L Lᵀ
    #
    # with pivot order (3, 1, 2).
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
    # "shadow" iterate p*, which solves
    #
    #   -f'(p*) = d
    #
    ρ::FScalarView{T}
end

function degree(::PowerCone, n::Integer)
    @assert n == 3
    return 3
end

function cachesize(::PowerCone, n::Integer)
    @assert n == 3
    return 13
end

function cache(c::Caches, i::Integer, cone::PowerCone)
    data = cachedata(c, i)
    R  = reshape(view(data, 1:9), 3, 3)
    ss = view(data, 10:12)
    ρ  = view(data, 13)
    PowerConeCache(cone, R, ss, ρ)
end

# construct the identity element
#
#   e = (√(1 + α), √(2 - α), 0)
#
function identity!(x::AbstractVector, cone::PowerCone)
    α = cone.α
    x[1] = sqrt(1 + α)
    x[2] = sqrt(2 - α)
    x[3] = false
    return x
end

function initcache!(cache::PowerConeCache)
    cache.ρ[] = true
    return cache
end

# evaluate the barrier argument
#
#   φ(p) = p₁ᵃ p₂ᵇ - p₃²
#
# where
#
#   - a = 2α
#   - b = 2 - 2α
#
function powphi(x::AbstractVector, α)
    a = 2α
    b = 2 - 2α
    return x[1]^a * x[2]^b - x[3]^2
end

# evaluate the gradient
#
#   f'(p)
#
# of the barrier function f.
function powbarrgrad!(g::AbstractVector, x::AbstractVector, α)
    a = 2α
    b = 2 - 2α
    p = x[1]^a * x[2]^b
    φ = p - x[3] * x[3]
    ρ = p / φ

    g[1] = -(a * ρ + 1 - α) / x[1]
    g[2] = -(b * ρ     + α) / x[2]
    g[3] =  2x[3] / φ

    return g
end

# evaluate the Hessian
#
#   f''(p)
#
# of the barrier function f.
function powbarrhess!(H::AbstractMatrix, x::AbstractVector, α)
    x1, x2, x3 = x[1], x[2], x[3]

    a = 2α
    b = 2 - 2α
    p = x1^a * x2^b
    φ = p - x3 * x3

    ρ  = p / φ
    w  = p / (φ * φ)
    wx = w * x3 * x3
    l1 = a / x1
    l2 = b / x2

    d1 = (2ρ * a + b) / (2x1 * x1)
    d2 = (2ρ * b + a) / (2x2 * x2)

    H[1,1] = l1 * l1 * wx + d1
    H[2,2] = l2 * l2 * wx + d2
    H[1,2] = l1 * l2 * wx

    H[3,3] = 2(p + x3 * x3) / (φ * φ)

    H[1,3] = -2l1 * x3 * w
    H[2,3] = -2l2 * x3 * w

    H[2,1] = H[1,2]
    H[3,1] = H[1,3]
    H[3,2] = H[2,3]

    return H
end

# factorize the Hessian
#
#   P f''(p) Pᵀ = L Lᵀ
#
# of the barrier function f, using
# the pivot order (3, 1, 2).
function powchol3!(L::AbstractMatrix, H::AbstractMatrix, x::AbstractVector, α)
    x1, x2, x3 = x[1], x[2], x[3]

    a = 2α
    b = 2 - 2α
    p = x1^a * x2^b
    φ = p - x3^2

    ρ = p / φ
    d1 = (2ρ * a + b) / 2x1^2
    d2 = (2ρ * b + a) / 2x2^2

    c = p * x3^2 / (φ * (p + x3^2))

    l1 = a / x1
    l2 = b / x2

    D33 = H[3,3]

    r1  =  d1      - c *  l1^2
    r2  = (d1 * d2 - c * (l2^2 * d1 + l1^2 * d2)) / r1

    L[1,1] = sqrt(H[3,3])

    L[2,1] = H[1,3] / L[1,1]
    L[3,1] = H[2,3] / L[1,1]

    L[2,2] = sqrt(r1)
    L[3,3] = sqrt(r2)

    L[3,2] = -c * l1 * l2 / L[2,2]

    L[1,2] = false
    L[1,3] = false
    L[2,3] = false

    return L
end

# solve for x in
#
#   f''(p) x = b
#
# using a pre-computed factorization
#
#   P f''(p) Pᵀ = L Lᵀ
#
function powldiv3!(L::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    b1, b2, b3 = b[1], b[2], b[3]

    y1 =  b3                              / L[1,1]
    y2 = (b1 - L[2,1] * y1)               / L[2,2]
    y3 = (b2 - L[3,1] * y1 - L[3,2] * y2) / L[3,3]

    z3 = y3                               / L[3,3]
    z2 = (y2 - L[3,2] * z3)               / L[2,2]
    z1 = (y1 - L[2,1] * z2 - L[3,1] * z3) / L[1,1]

    b[1] = z2
    b[2] = z3
    b[3] = z1
    return b
end

# compute the third-order directional derivative
#
#   f'''(p)[u]
#
# as a 3x3 matrix
function powbarrthird!(D::AbstractMatrix, x::AbstractVector, u::AbstractVector, α)
    x1, x2, x3 = x[1], x[2], x[3]
    u1, u2, u3 = u[1], u[2], u[3]

    a = 2α
    b = 2 - 2α
    p = x1^a * x2^b
    φ = p - x3 * x3

    l1 = a / x1
    l2 = b / x2
    m1 = u1 / x1
    m2 = u2 / x2

    φp1 = p * l1
    φp2 = p * l2
    φp3 = -2x3

    φdot = φp1 * u1 + φp2 * u2 + φp3 * u3

    φpp11 = p * l1 * (l1 - inv(x1))
    φpp22 = p * l2 * (l2 - inv(x2))
    φpp12 = p * l1 * l2

    φdotp1 = φpp11 * u1 + φpp12 * u2
    φdotp2 = φpp12 * u1 + φpp22 * u2
    φdotp3 = -2u3

    φ2 = φ * φ
    φ3 = φ * φ * φ

    c2 =  φdot / φ2
    c3 = 2φdot / φ3

    K  = p / φ * a * b * (a - 1) * (m2 - m1)

    D[1,1] = 2φdotp1 * φp1                 / φ2 - c3 * φp1 * φp1 - K / (x1 * x1) +  c2 * φpp11 - b * m1 / (x1 * x1)
    D[2,2] = 2φdotp2 * φp2                 / φ2 - c3 * φp2 * φp2 - K / (x2 * x2) +  c2 * φpp22 - a * m2 / (x2 * x2)
    D[2,1] = (φdotp2 * φp1 + φp2 * φdotp1) / φ2 - c3 * φp2 * φp1 + K / (x1 * x2) +  c2 * φpp12
    D[3,3] = 2φdotp3 * φp3                 / φ2 - c3 * φp3 * φp3                 - 2c2
    D[3,1] = (φdotp3 * φp1 + φp3 * φdotp1) / φ2 - c3 * φp3 * φp1
    D[3,2] = (φdotp3 * φp2 + φp3 * φdotp2) / φ2 - c3 * φp3 * φp2

    D[1,2] = D[2,1]
    D[1,3] = D[3,1]
    D[2,3] = D[3,2]

    return D
end

# Determine if x is in the power cone.
function powincone(x::AbstractVector{T}, α::T) where {T}
    x[1] > 0 && x[2] > 0 && powphi(x, α) > 0
end

# Determine if s is in the dual power cone.
function powindual(s::AbstractVector{T}, α::T) where {T}
    s[1] > 0 && s[2] > 0 && (s[1] / α)^α * (s[2] / (one(T) - α))^(one(T) - α) > abs(s[3])
end

# compute the primal "shadow" iterate x*, solving
#
#   -f'(p*) = d,
#
# using a 1-D scalar root-find on the function
#
#   g(ρ) = ρ(ρ - 1) - ¼ d₃² ((aρ + ½b) / d₁)ᵃ ((bρ + ½a) / d₂)ᵇ  
#
# with
#
#   a = 2α
#   b = 2 - 2α
#
# and
#
#   p₁* = ((aρ + ½b) / d₁)
#   p₂* = ((bρ + ½a) / d₂)
#   p₃* = -d₃ / 2ρ (p₁*)ᵃ (p₂*)ᵇ
#
function powdualgrad!(xs::AbstractVector{T}, seed::T, s::AbstractVector{T}, α::T) where {T}
    s1, s2, s3 = s[1], s[2], s[3]

    a =     2α
    b = 2 - 2α

    function g(ρ)
        return ρ * (ρ - 1) - s3^2 / 4 * ((a * ρ + 1 - α) / s1)^a * ((b * ρ + α) / s2)^b
    end

    function gp(ρ)
        return (2ρ - 1) - s3^2 / 4 * ((a * ρ + 1 - α) / s1)^a * ((b * ρ + α) / s2)^b * (a^2 / (a * ρ + 1 - α) + b^2 / (b * ρ + α))
    end

    hi = 2max(seed, 1)

    while g(hi) ≤ 0
        hi *= 2
    end

    ρ = rtsafe(g, gp, one(T), hi, seed)

    xs[1] = x1 = (a * ρ + 1 - α) / s1
    xs[2] = x2 = (b * ρ +     α) / s2
    xs[3] = -s3 * (x1^a * x2^b / ρ) / 2

    return ρ
end

# Compute the coefficient t in the rank-1 term
# t z zᵀ of the Tuncel scaling matrix M:
#
#   M = ⟨x, s⟩⁻¹   s  sᵀ
#     + ⟨δx,δs⟩⁻¹ δs δsᵀ
#     + t          z  zᵀ,
#
# where
#
#   δx = x - μ x*
#   δs = s - μ s*
#
function powbfgs(
        H::AbstractMatrix{T},
        xs::AbstractVector{T},
        ss::AbstractVector{T},
        z::AbstractVector{T},
        x::AbstractVector{T},
        μv::T,
        μt::T
    ) where {T}
    t = zero(T)

    w  = zeros(T, 3)
    Hw = zeros(T, 3)
    Hz = zeros(T, 3)
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
    mul3!(Hw, H, w)
    d = dot3(w, Hw)

    if d > 0
        #
        # compute the norm
        #
        #   ⟨z, f''(p) z⟩ = ‖Rᵀ z‖²
        #
        mul3!(Hz, H, z)
        fppzz = dot3(z, Hz)
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
        pz = dot3(Hw, z)
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
#   M = ⟨x, s⟩⁻¹   s  sᵀ
#     + ⟨δx,δs⟩⁻¹ δs δsᵀ
#     + t          z  zᵀ,
#
# where
#
#   δx = x - μ x*
#   δs = s - μ s*
#
function powscale!(
        H::AbstractMatrix{T},
        R::AbstractMatrix{T},
        ss::AbstractVector{T},
        ρ_seed::T,
        x::AbstractVector{T},
        s::AbstractVector{T},
        α::T
    ) where {T}

    xs = zeros(T, 3)
    z  = zeros(T, 3)
    δx = zeros(T, 3)
    δs = zeros(T, 3)
    #
    # compute the Hessian
    #
    #   f''(p)
    #
    powbarrhess!(R, x, α)
    #
    # compute the "shadow" dual
    #
    #   d* = -f'(p)
    #
    powbarrgrad!(ss, x, α)
    lmul3!(-1, ss)
    #
    # compute the "shadow" primal, solving
    #
    #   d = -f'(p*)
    #
    ρ_new = powdualgrad!(xs, ρ_seed, s, α)
    #
    # compute the centrality parameters
    #
    #   μ  = ⟨p,  d ⟩ / ν
    #   μ* = ⟨p*, d*⟩ / ν
    #
    μv = dot3(x, s)   / 3
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
    nz = norm3(z)
    nx = norm3(x)
    nxs = norm3(xs)

    if nz < eps(T) * (nx * nxs + eps(T))
        #
        # approximate M by
        #
        #    M ≈ μ f''(p)
        #
        copyto!(H, R)
        lmul3!(μv, H)
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
        t = powbfgs(R, xs, ss, z, x, μv, μt)

        if t ≤ 0 || !isfinite(t)
            #   
            # approximate M by
            #
            #    M ≈ μ f''(p)
            #
            copyto!(H, R)
            lmul3!(μv, H)
        else
            #
            # construct M:
            #
            #   M = ⟨p, d⟩⁻¹   d  dᵀ
            #     + ⟨δp,δd⟩⁻¹ δd δdᵀ
            #     + t          z  zᵀ,
            #
            copy3!(δx, x);  axpy3!(-μv, xs, δx)
            copy3!(δs, s);  axpy3!(-μv, ss, δs)

            xs_dot = 3μv
            δ_dot = dot3(δx, δs)

            ger3!(H,  s,  s, inv(xs_dot), 0)
            ger3!(H, δs, δs, inv(δ_dot),  1)
            ger3!(H,  z,  z, t,           1)
        end
    end

    powchol3!(R, R, x, α)
    return ρ_new
end

function scale!(H::AbstractMatrix{T}, p::AbstractVector{T}, d::AbstractVector{T}, cache::PowerConeCache{T}) where {T}
    cache.ρ[] = powscale!(H, cache.R, cache.ss, cache.ρ[], p, d, cache.cone.α)
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
function powcorr!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        ss::AbstractVector{T},
        p::AbstractVector{T},
        d::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        α::T
    ) where {T}

    v  = zeros(T, 3)
    η  = zeros(T, 3)
    D  = zeros(T, 3, 3)
    #
    # solve for v in
    #
    #   f''(p) v = Δd
    #
    # using the structured Cholesky factor stored in R
    #
    copy3!(v, Δd)
    powldiv3!(R, v)
    #
    # compute the third-order correction
    #
    #   η = -½ f'''(p)[Δp, v]
    #
    powbarrthird!(D, p, Δp, α)
    mul3!(η, D, v, -0.5, 0)
    #
    # compute the Mehrotra corrector term
    #
    #   -d - σμ f'(p) - η,
    #
    # using s* = -f'(p):  -d + σμ s* - η
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
        cache::PowerConeCache{T}
    ) where {T}
    return powcorr!(r, cache.R, cache.ss, p, d, Δp, Δd, σμ, cache.cone.α)
end

# use bisection to find the largest number 0 < τ ≤ 1
# such that
#
#   x + τ Δx
#
# is in the power cone (or its dual).
function powmaxstep(incone, x::AbstractVector{T}, Δx::AbstractVector{T}, α::T) where {T}
    w = zeros(T, 3)

    τ = binarysearchlast(zero(T), one(T), eps(T), 53) do τ
        copy3!(w, x)
        axpy3!(τ, Δx, w)
        return incone(w, α)
    end

    return τ
end

function maxsteps(p::AbstractVector{T}, Δp::AbstractVector{T}, d::AbstractVector{T}, Δd::AbstractVector{T}, cache::PowerConeCache{T}) where {T}
    α = cache.cone.α
    return powmaxstep(powincone, p, Δp, α), powmaxstep(powindual, d, Δd, α)
end
