#
# PowerCone: 3D power cone with parameter α ∈ (0,1)
#
# P_α = { x ∈ R³ : x₁^α x₂^(1-α) ≥ |x₃|, x₁,x₂ ≥ 0 }
#
# Barrier: F(x) = -log(x₁^(2α) x₂^(2(1-α)) - x₃²) - (1-α)log(x₁) - α log(x₂)
#

struct PowerCone{T} <: Cone
    α::T

    function PowerCone{T}(α::T) where {T}
        (0 < α < 1) || throw(ArgumentError("PowerCone requires 0 < α < 1, got α = $α"))
        return new{T}(α)
    end
end

PowerCone(α::T) where {T} = PowerCone{T}(α)

struct PowerConeCache{T} <: AbstractCache{PowerCone{T}}
    cone::PowerCone{T}
    M::FMatrixView{T}     # scaling Gram matrix (3×3)
    R::FMatrixView{T}     # factor of F''(x) in (3,1,2) order (3×3)
    xs::FVectorView{T}    # shadow primal x̃ (3)
    ss::FVectorView{T}    # shadow dual s̃ = -F'(x) (3)
    μv::FScalarView{T}    # block-local μ = ⟨x,s⟩/3
end

# degree = 3 always (POW is intrinsically 3D)
function degree(::PowerCone, n::Int)
    @assert n == 3 "PowerCone is 3-dimensional"
    return 3
end

# cache size: M(9) + R(9) + xs(3) + ss(3) + μv(1) = 25
function cachesize(::PowerCone, n::Int)
    @assert n == 3 "PowerCone is 3-dimensional"
    return 25
end

# construct view-based cache from Caches
function cache(c::Caches{T}, i::Int, cone::PowerCone{T}) where T
    data = view(c.val, c.xblk[i]:c.xblk[i+1]-1)
    M  = reshape(view(data, 1:9), 3, 3)
    R  = reshape(view(data, 10:18), 3, 3)
    xs = view(data, 19:21)
    ss = view(data, 22:24)
    μv = view(data, 25)
    PowerConeCache(cone, M, R, xs, ss, μv)
end

# Central point: x₀ = s₀ = (√(1+α), √(2-α), 0)
function identity!(x::AbstractVector{T}, cone::PowerCone{T}) where {T}
    α = cone.α
    x[1] = sqrt(one(T) + α)
    x[2] = sqrt(2 * one(T) - α)
    x[3] = zero(T)
    return x
end

# Initialize cache.xs to identity point
function initcache!(cache::PowerConeCache{T}) where {T}
    identity!(cache.xs, cache.cone)
    return cache
end

#
# Shared scalars
#
# a = 2α, b = 2(1-α)
# p = x₁^a * x₂^b (power product)
# φ = p - x₃² (barrier argument)
# ρ = p / φ (≥ 1, equality iff x₃ = 0)
#

function powphi(x::AbstractVector{T}, α::T) where {T}
    a = 2 * α
    b = 2 * (one(T) - α)
    p = x[1]^a * x[2]^b
    return p - x[3]^2
end

#
# Gradient F'(x)
#
# F'(x) = (-a*p/(x₁ φ) - (1-α)/x₁, -b*p/(x₂ φ) - α/x₂, 2x₃/φ)
#

function powbarrgrad!(g::AbstractVector{T}, x::AbstractVector{T}, α::T) where {T}
    a = 2 * α
    b = 2 * (one(T) - α)
    p = x[1]^a * x[2]^b
    φ = p - x[3]^2

    g[1] = -a * p / (x[1] * φ) - (one(T) - α) / x[1]
    g[2] = -b * p / (x[2] * φ) - α / x[2]
    g[3] = 2 * x[3] / φ

    return g
end

#
# Hessian F''(x) with cancellation-free entries
#
# Entries:
#   d₁ = (2ρa + b)/(2 x₁²)
#   d₂ = (2ρb + a)/(2 x₂²)
#   F″₁₁ = d₁ + a² p x₃² /(x₁² φ²)
#   F″₂₂ = d₂ + b² p x₃² /(x₂² φ²)
#   F″₁₂ = a b p x₃² /(x₁ x₂ φ²)
#   F″₃₃ = 2 (p + x₃²) / φ²
#   F″₁₃ = -2 a x₃ p /(x₁ φ²)
#   F″₂₃ = -2 b x₃ p /(x₂ φ²)
#

function powhess!(H::AbstractMatrix{T}, x::AbstractVector{T}, α::T) where {T}
    x1, x2, x3 = x[1], x[2], x[3]
    a = 2 * α
    b = 2 * (one(T) - α)
    p = x1^a * x2^b
    φ = p - x3^2
    ρ = p / φ

    d1 = (2 * ρ * a + b) / (2 * x1^2)
    d2 = (2 * ρ * b + a) / (2 * x2^2)

    H[1,1] = d1 + a^2 * p * x3^2 / (x1^2 * φ^2)
    H[2,2] = d2 + b^2 * p * x3^2 / (x2^2 * φ^2)
    H[1,2] = a * b * p * x3^2 / (x1 * x2 * φ^2)
    H[2,1] = H[1,2]
    H[3,3] = 2 * (p + x3^2) / φ^2
    H[1,3] = -2 * a * x3 * p / (x1 * φ^2)
    H[3,1] = H[1,3]
    H[2,3] = -2 * b * x3 * p / (x2 * φ^2)
    H[3,2] = H[2,3]

    return H
end

#
# Structured Cholesky factorization with (3,1,2) pivot order
#
# The naive (1,2,3) Cholesky loses all digits when φ → 0 because the
# Schur complement accumulates O(φ⁻²) terms that cancel. Pivoting
# coordinate 3 first and using the symbolic collapse
#
#   c = p x₃² / (φ (p + x₃²))
#
# which is O(φ⁻¹), preserves ~6 more digits near the boundary.
#
# L is stored in (3,1,2) permuted order:
#   row/col 1 of L corresponds to original coord 3
#   row/col 2 of L corresponds to original coord 1
#   row/col 3 of L corresponds to original coord 2
#

function powchol3!(L::AbstractMatrix{T}, H::AbstractMatrix{T}, x::AbstractVector{T}, α::T) where {T}
    x1, x2, x3 = x[1], x[2], x[3]
    a = 2 * α
    b = 2 * (one(T) - α)
    p = x1^a * x2^b
    φ = p - x3^2

    # Diagonal scalars from powhess! (recomputed for numerical stability)
    ρ = p / φ
    d1 = (2 * ρ * a + b) / (2 * x1^2)
    d2 = (2 * ρ * b + a) / (2 * x2^2)

    # Schur collapse coefficient: O(φ⁻¹) instead of O(φ⁻²)
    c = p * x3^2 / (φ * (p + x3^2))

    # Log-gradient components
    ℓ1 = a / x1
    ℓ2 = b / x2

    # Radicands for the three pivots
    D33 = H[3,3]                                    # 2(p + x₃²)/φ²
    r1  = d1 - c * ℓ1^2
    r2  = (d1 * d2 - c * (d1 * ℓ2^2 + d2 * ℓ1^2)) / r1

    # Build L in (3,1,2) permuted storage
    L[1,1] = sqrt(D33)
    L[2,1] = H[1,3] / L[1,1]
    L[3,1] = H[2,3] / L[1,1]
    L[2,2] = sqrt(r1)
    L[3,2] = -c * ℓ1 * ℓ2 / L[2,2]
    L[3,3] = sqrt(r2)

    # Zero upper triangle (not strictly necessary but clean)
    L[1,2] = zero(T)
    L[1,3] = zero(T)
    L[2,3] = zero(T)

    return L
end

# Solve H⁻¹ b via structured Cholesky L Lᵀ = P H Pᵀ (permutation 3,1,2)
function powldiv3!(L::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    # Forward substitution: solve L y = Pb
    # Permute b from (1,2,3) to (3,1,2): [b₃, b₁, b₂]
    b1, b2, b3 = b[1], b[2], b[3]
    y1 = b3 / L[1,1]
    y2 = (b1 - L[2,1] * y1) / L[2,2]
    y3 = (b2 - L[3,1] * y1 - L[3,2] * y2) / L[3,3]

    # Backward substitution: solve Lᵀ z = y
    z3 = y3 / L[3,3]
    z2 = (y2 - L[3,2] * z3) / L[2,2]
    z1 = (y1 - L[2,1] * z2 - L[3,1] * z3) / L[1,1]

    # Unpermute from (3,1,2) back to (1,2,3): z1→slot3, z2→slot1, z3→slot2
    b[1] = z2
    b[2] = z3
    b[3] = z1
    return b
end

# Hessian + in-place Cholesky (legacy, uses naive order)
function powbarr!(L::AbstractMatrix{T}, x::AbstractVector{T}, α::T) where {T}
    powhess!(L, x, α)
    chol3!(L)
    return L
end

#
# Third-order directional derivative F'''(x)[u]
#

function powbarrhess!(D::AbstractMatrix{T}, x::AbstractVector{T}, u::AbstractVector{T}, α::T) where {T}
    x1, x2, x3 = x[1], x[2], x[3]
    u1, u2, u3 = u[1], u[2], u[3]
    a = 2 * α
    b = 2 * (one(T) - α)
    p = x1^a * x2^b
    φ = p - x3^2

    # ℓ = (a/x₁, b/x₂, 0) = ∇log p
    ℓ1 = a / x1
    ℓ2 = b / x2
    ℓ3 = zero(T)

    # φ′ = (a*p/x₁, b*p/x₂, -2x₃)
    φp1 = a * p / x1
    φp2 = b * p / x2
    φp3 = -2 * x3

    # φ̇ = ⟨φ′, u⟩
    φdot = φp1 * u1 + φp2 * u2 + φp3 * u3

    # Dℓ = -diag(a/x₁², b/x₂², 0)
    Dℓ11 = -a / x1^2
    Dℓ22 = -b / x2^2

    # φ″ = p(ℓℓᵀ + Dℓ) + diag(0,0,-2)
    # φ″₁₁ = p(ℓ₁² + Dℓ₁₁) = p(ℓ₁² - a/x₁²)
    # φ″₂₂ = p(ℓ₂² + Dℓ₂₂) = p(ℓ₂² - b/x₂²)
    # φ″₁₂ = p ℓ₁ ℓ₂
    # φ″₃₃ = -2
    # φ″₁₃ = φ″₂₃ = 0

    φpp11 = p * (ℓ1^2 + Dℓ11)
    φpp22 = p * (ℓ2^2 + Dℓ22)
    φpp12 = p * ℓ1 * ℓ2
    φpp33 = -2 * one(T)

    # φ̇′ = φ″ u
    φdotp1 = φpp11 * u1 + φpp12 * u2
    φdotp2 = φpp12 * u1 + φpp22 * u2
    φdotp3 = φpp33 * u3

    # Dℓu = Dℓ u
    Dℓu1 = Dℓ11 * u1
    Dℓu2 = Dℓ22 * u2
    Dℓu3 = zero(T)

    # ⟨ℓ,u⟩
    ℓu = ℓ1 * u1 + ℓ2 * u2

    # Ḋℓ = diag(2a u₁/x₁³, 2b u₂/x₂³, 0)
    Ddotℓ11 = 2 * a * u1 / x1^3
    Ddotℓ22 = 2 * b * u2 / x2^3

    # φ‴[u] = p⟨ℓ,u⟩ (ℓℓᵀ + Dℓ) + p (Dℓu ℓᵀ + ℓ (Dℓu)ᵀ + Ḋℓ)
    # φ‴[u]₁₁ = p ℓu (ℓ₁² + Dℓ₁₁) + p (2 Dℓu₁ ℓ₁ + Ḋℓ₁₁)
    # φ‴[u]₂₂ = p ℓu (ℓ₂² + Dℓ₂₂) + p (2 Dℓu₂ ℓ₂ + Ḋℓ₂₂)
    # φ‴[u]₁₂ = p ℓu ℓ₁ ℓ₂ + p (Dℓu₁ ℓ₂ + ℓ₁ Dℓu₂)
    # φ‴[u]₃₃ = 0, φ‴[u]₁₃ = φ‴[u]₂₃ = 0

    φ3u11 = p * ℓu * (ℓ1^2 + Dℓ11) + p * (2 * Dℓu1 * ℓ1 + Ddotℓ11)
    φ3u22 = p * ℓu * (ℓ2^2 + Dℓ22) + p * (2 * Dℓu2 * ℓ2 + Ddotℓ22)
    φ3u12 = p * ℓu * ℓ1 * ℓ2 + p * (Dℓu1 * ℓ2 + ℓ1 * Dℓu2)

    # ḣ″ = diag(-2(1-α)u₁/x₁³, -2α u₂/x₂³, 0)
    hdot11 = -2 * (one(T) - α) * u1 / x1^3
    hdot22 = -2 * α * u2 / x2^3

    φ2 = φ^2
    φ3 = φ^3

    # F‴(x)[u] = (φ̇′ φ′ᵀ + φ′ φ̇′ᵀ)/φ²
    #          - (2 φ̇/φ³) φ′ φ′ᵀ
    #          - φ‴[u]/φ
    #          + (φ̇/φ²) φ″
    #          + ḣ″

    # Compute each entry
    for j in 1:3, i in j:3
        φpi = (φp1, φp2, φp3)[i]
        φpj = (φp1, φp2, φp3)[j]
        φdotpi = (φdotp1, φdotp2, φdotp3)[i]
        φdotpj = (φdotp1, φdotp2, φdotp3)[j]

        # Term 1: (φ̇′ φ′ᵀ + φ′ φ̇′ᵀ)/φ²
        term1 = (φdotpi * φpj + φpi * φdotpj) / φ2

        # Term 2: -(2 φ̇/φ³) φ′ φ′ᵀ
        term2 = -2 * φdot / φ3 * φpi * φpj

        # Term 3: -φ‴[u]/φ (only has 1,1 and 2,2 and 1,2 entries)
        φ3uij = zero(T)
        if i == 1 && j == 1
            φ3uij = φ3u11
        elseif i == 2 && j == 2
            φ3uij = φ3u22
        elseif (i == 1 && j == 2) || (i == 2 && j == 1)
            φ3uij = φ3u12
        end
        term3 = -φ3uij / φ

        # Term 4: (φ̇/φ²) φ″
        φppij = zero(T)
        if i == 1 && j == 1
            φppij = φpp11
        elseif i == 2 && j == 2
            φppij = φpp22
        elseif (i == 1 && j == 2) || (i == 2 && j == 1)
            φppij = φpp12
        elseif i == 3 && j == 3
            φppij = φpp33
        end
        term4 = φdot / φ2 * φppij

        # Term 5: ḣ″
        hdotij = zero(T)
        if i == 1 && j == 1
            hdotij = hdot11
        elseif i == 2 && j == 2
            hdotij = hdot22
        end

        D[i,j] = term1 + term2 + term3 + term4 + hdotij
    end

    # Symmetrize
    D[1,2] = D[2,1]
    D[1,3] = D[3,1]
    D[2,3] = D[3,2]

    return D
end

#
# Cone membership predicates
#

function powincone(x::AbstractVector{T}, α::T) where {T}
    x[1] > 0 && x[2] > 0 && powphi(x, α) > 0
end

function powindual(s::AbstractVector{T}, α::T) where {T}
    s[1] > 0 && s[2] > 0 && (s[1] / α)^α * (s[2] / (one(T) - α))^(one(T) - α) > abs(s[3])
end

#
# Shadow primal computation (1D scalar solve for ρ)
#
# Find x̃ such that F'(x̃) = -s by solving for ρ ∈ [1, ∞):
#   X₁(ρ) = (2αρ + 1-α)/s₁
#   X₂(ρ) = (2(1-α)ρ + α)/s₂
#   g(ρ) = ρ(ρ-1) - (s₃²/4) X₁(ρ)^(2α) X₂(ρ)^(2(1-α)) = 0
#
# Uses bracketed bisection (not Newton, which fails ~12% of points).
# Shortcuts: s₃=0 ⟹ ρ*=1; α=1/2 ⟹ quadratic closed form.
#

function powdualgrad!(xs::AbstractVector{T}, s::AbstractVector{T}, α::T) where {T}
    s1, s2, s3 = s[1], s[2], s[3]
    a = 2 * α
    b = 2 * (one(T) - α)

    # Shortcut 1: s₃ = 0 ⟹ ρ* = 1 (symmetric slice)
    if iszero(s3)
        ρ = one(T)
    # Shortcut 2: α = 1/2 ⟹ quadratic closed form
    elseif α == one(T)/2
        k = s3^2 / (4 * s1 * s2)
        ρ = ((one(T) + k) + sqrt(one(T) + 3*k)) / (2 * (one(T) - k))
    else
        # General case: bracketed bisection
        function gval(ρ)
            X1 = (a * ρ + one(T) - α) / s1
            X2 = (b * ρ + α) / s2
            return ρ * (ρ - one(T)) - (s3^2 / 4) * X1^a * X2^b
        end

        # Build bracket: lo=1 (g(1)≤0 always), grow hi until g(hi)>0
        lo = one(T)
        hi = 2 * one(T)
        while gval(hi) ≤ 0
            hi *= 2
        end

        # Bisection: find largest ρ where g(ρ) ≤ 0
        ρ = binarysearchlast(ρ -> gval(ρ) ≤ 0, lo, hi, eps(T), 64)
    end

    # Recover x̃ from ρ
    X1 = (a * ρ + one(T) - α) / s1
    X2 = (b * ρ + α) / s2
    p = X1^a * X2^b
    φ = p / ρ
    x3 = -s3 * φ / 2

    xs[1] = X1
    xs[2] = X2
    xs[3] = x3

    return xs
end

#
# BFGS t computation (same structure as ExpCone)
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
    # Workspaces
    w  = zeros(T, 3)
    Hw = zeros(T, 3)
    Hz = zeros(T, 3)

    # Gap direction w = x̃ − μ̃x
    copy3!(w, xs)
    axpy3!(-μt, x, w)

    # Hw = F'' w, Hz = F'' z
    mul3!(Hw, H, w)
    mul3!(Hz, H, z)

    d = dot3(w, Hw)
    d ≤ zero(T) && return zero(T)

    fppzz = dot3(z, Hz)   # zᵀF''z
    sz = dot3(ss, z)      # s̃ᵀz
    pz = dot3(Hw, z)      # (F''w)ᵀz

    return μv * (fppzz - sz^2 / 3 - pz^2 / d)
end

#
# Scale computation
#

function powscale!(
        M::AbstractMatrix{T},
        H::AbstractMatrix{T},
        xs::AbstractVector{T},
        ss::AbstractVector{T},
        x::AbstractVector{T},
        s::AbstractVector{T},
        α::T
    ) where {T}

    # Workspace
    z  = zeros(T, 3)
    δx = zeros(T, 3)
    δs = zeros(T, 3)

    # Stage 1: Hessian F''(x), shadow dual s̃ = -F'(x)
    powhess!(H, x, α)
    powbarrgrad!(ss, x, α)
    lmul!(-1, ss)

    # Stage 2: Shadow primal x̃
    powdualgrad!(xs, s, α)

    # Block-local μ and μ̃
    μv = dot3(x, s) / 3
    μt = dot3(xs, ss) / 3

    # Stage 3: Orthogonal completion z = x × x̃
    cross3!(z, x, xs)
    nz = norm3(z)

    # Centrality check
    nx = norm3(x)
    nxs = norm3(xs)
    rel_z = nz / (nx * nxs + eps(T))

    if rel_z < eps(T)
        # On or near central path: M = μ F''(x)
        copyto!(M, H)
        lmul3!(μv, M)
    else
        # Normalize z
        ldiv3!(nz, z)

        # BFGS t (uses H for matvecs)
        t = powbfgs(H, xs, ss, z, x, μv, μt)

        if !(t > 0) || !isfinite(t)
            # Fallback to μF''
            copyto!(M, H)
            lmul3!(μv, M)
        else
            # M = ssᵀ/⟨x,s⟩ + δsδsᵀ/⟨δx,δs⟩ + tzzᵀ
            xs_dot = 3 * μv

            copy3!(δx, x);  axpy3!(-μv, xs, δx)
            copy3!(δs, s);  axpy3!(-μv, ss, δs)

            δ_dot = dot3(δx, δs)

            ger3!(M, s, s, inv(xs_dot), 0)
            ger3!(M, δs, δs, inv(δ_dot), 1)
            ger3!(M, z, z, t, 1)
        end
    end

    # Structured Cholesky H for corrector solve (pivot order 3,1,2)
    powchol3!(H, H, x, α)

    return μv
end

function scale!(H::AbstractMatrix{T}, p::AbstractVector{T}, d::AbstractVector{T}, cache::PowerConeCache{T}) where {T}
    α = cache.cone.α
    cache.μv[] = powscale!(cache.M, cache.R, cache.xs, cache.ss, p, d, α)
    copyto!(H, cache.M)
    return H
end

#
# Corrector
#

function powcorr!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        p::AbstractVector{T},
        d::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        α::T
    ) where {T}

    # Workspaces
    Fp = zeros(T, 3)
    v  = zeros(T, 3)
    η  = zeros(T, 3)
    D  = zeros(T, 3, 3)

    # F'(p)
    powbarrgrad!(Fp, p, α)

    # v = F''(p)⁻¹Δd via structured Cholesky factor (stored in R, permuted 3,1,2)
    copy3!(v, Δd)
    powldiv3!(R, v)

    # η = -½ F'''(p)[Δp, v]
    powbarrhess!(D, p, Δp, α)
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
        cache::PowerConeCache{T}
    ) where {T}
    return powcorr!(r, cache.R, p, d, Δp, Δd, σμ, cache.cone.α)
end

#
# Max step by bisection on cone membership
#

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
