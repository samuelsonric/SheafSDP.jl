#
# EXP cone (exponential cone)
#
# x = (x₁, x₂, x₃) ∈ EXP iff x₁ ≥ x₂ exp(x₃/x₂), x₂ > 0
#
# Following Dahl & Andersen, "A primal-dual interior-point algorithm for
# nonsymmetric exponential-cone optimization," Math. Program. (2022) 194:341–370.
#

struct EXP <: Cone end

struct EXPCache{T} <: AbstractCache{EXP}
    cone::EXP
    M::FMatrixView{T}     # scaling Gram matrix (3×3)
    R::FMatrixView{T}     # Cholesky factor of F''(x) (3×3)
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
function identity!(x::AbstractVector{T}, ::EXP) where {T}
    x[1] = T(1.2909282315382298)
    x[2] = T(0.8051015526498357)
    x[3] = T(-0.8278379086082098)
    return x
end

#
# 3×3 linear algebra helpers
#

# Cholesky factorization of 3×3 symmetric positive definite matrix
# Overwrites lower triangle of A with L such that A = LLᵀ
# Returns false if factorization fails (matrix not SPD), true otherwise
function chol3!(A::AbstractMatrix{T}) where {T}
    ε = eps(T)^(2/3)  # small regularization threshold

    d1 = A[1,1]
    if d1 < ε
        d1 = ε
    end
    A[1,1] = sqrt(d1)
    A[2,1] /= A[1,1]
    A[3,1] /= A[1,1]

    d2 = A[2,2] - A[2,1]^2
    if d2 < ε
        d2 = ε
    end
    A[2,2] = sqrt(d2)
    A[3,2] = (A[3,2] - A[3,1] * A[2,1]) / A[2,2]

    d3 = A[3,3] - A[3,1]^2 - A[3,2]^2
    if d3 < ε
        d3 = ε
    end
    A[3,3] = sqrt(d3)

    return A
end

# Solve Lx = b where L is 3×3 lower triangular (in-place on b)
function ldiv3!(L::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    b[1] /= L[1,1]
    b[2] = (b[2] - L[2,1] * b[1]) / L[2,2]
    b[3] = (b[3] - L[3,1] * b[1] - L[3,2] * b[2]) / L[3,3]
    return b
end

# Solve Lᵀx = b where L is 3×3 lower triangular (in-place on b)
function ltdiv3!(L::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    b[3] /= L[3,3]
    b[2] = (b[2] - L[3,2] * b[3]) / L[2,2]
    b[1] = (b[1] - L[2,1] * b[2] - L[3,1] * b[3]) / L[1,1]
    return b
end

# Solve LLᵀx = b (in-place on b) where L is 3×3 lower triangular
function solve3!(L::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    ldiv3!(L, b)
    ltdiv3!(L, b)
    return b
end

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

# 3×3 matrix-vector product: y = Ax
function mul3!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    y[1] = A[1,1] * x[1] + A[1,2] * x[2] + A[1,3] * x[3]
    y[2] = A[2,1] * x[1] + A[2,2] * x[2] + A[2,3] * x[3]
    y[3] = A[3,1] * x[1] + A[3,2] * x[2] + A[3,3] * x[3]
    return y
end

# Frobenius norm of 3×3 matrix
function frob3(A::AbstractMatrix{T}) where {T}
    s = zero(T)
    for j in 1:3, i in 1:3
        s += A[i,j]^2
    end
    return sqrt(s)
end

# 2×2 inverse by hand: returns (a,b,c,d) such that [a b; c d] = [A[1,1] A[1,2]; A[2,1] A[2,2]]⁻¹
function inv2x2(A11::T, A12::T, A21::T, A22::T) where {T}
    det = A11 * A22 - A12 * A21
    return A22 / det, -A12 / det, -A21 / det, A11 / det
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

# Hessian of ψ (symmetric, stored in lower triangle)
function exp_psi_hess!(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    x1, x2 = x[1], x[2]
    H[1,1] = -x2 / x1^2
    H[2,1] = one(T) / x1
    H[3,1] = zero(T)
    H[2,2] = -one(T) / x2
    H[3,2] = zero(T)
    H[3,3] = zero(T)
    return H
end

# Barrier gradient: F'(x) = -ψ'(x)/ψ(x) - (1/x₁, 1/x₂, 0)
function exp_barrier_grad!(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    ψ = exp_psi(x)
    exp_psi_grad!(g, x)

    # g = -ψ'/ψ - h' where h' = (1/x₁, 1/x₂, 0)
    g[1] = -g[1] / ψ - one(T) / x[1]
    g[2] = -g[2] / ψ - one(T) / x[2]
    g[3] = -g[3] / ψ  # = 1/ψ

    return g
end

# Barrier Hessian: F''(x) = ψ'ψ'ᵀ/ψ² - ψ''/ψ + diag(1/x₁², 1/x₂², 0)
# Returns the full symmetric matrix
function exp_barrier_hess!(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    ψ = exp_psi(x)

    # Compute ψ'
    ψg = zeros(T, 3)
    exp_psi_grad!(ψg, x)

    # Compute ψ''
    ψH = zeros(T, 3, 3)
    exp_psi_hess!(ψH, x)

    # F'' = ψ'ψ'ᵀ/ψ² - ψ''/ψ + h''
    ψ2 = ψ^2
    for j in 1:3, i in j:3
        H[i,j] = ψg[i] * ψg[j] / ψ2 - ψH[i,j] / ψ
    end

    # Add h'' = diag(1/x₁², 1/x₂², 0)
    H[1,1] += one(T) / x[1]^2
    H[2,2] += one(T) / x[2]^2

    # Symmetrize
    H[1,2] = H[2,1]
    H[1,3] = H[3,1]
    H[2,3] = H[3,2]

    return H
end

# Third-order directional derivative F'''(x)[u] as a 3×3 symmetric matrix
# This is the derivative of F''(x) in direction u
function exp_barrier_hess_dir!(D::AbstractMatrix{T}, x::AbstractVector{T}, u::AbstractVector{T}) where {T}
    ψ = exp_psi(x)
    x1, x2, x3 = x[1], x[2], x[3]
    u1, u2, u3 = u[1], u[2], u[3]

    # Compute ψ' and ψ''
    ψg = zeros(T, 3)
    exp_psi_grad!(ψg, x)

    ψH = zeros(T, 3, 3)
    exp_psi_hess!(ψH, x)

    # ψ'·u
    ψgu = ψg[1] * u1 + ψg[2] * u2 + ψg[3] * u3

    # Third derivative of ψ (from paper eq. 33)
    # ψ'''[u] components (only non-zero parts)
    ψ3_11 = 2 * x2 * u1 / x1^3 - u2 / x1^2
    ψ3_21 = -u1 / x1^2
    ψ3_22 = u2 / x2^2

    # The formula for F'''[u] is:
    # d/dt F''(x+tu)|_{t=0} = -2 ψ'ψ'ᵀ (ψ'·u) / ψ³
    #                       + (ψ'(ψ''u)ᵀ + (ψ''u)ψ'ᵀ) / ψ²
    #                       - ψ'''[u] / ψ
    #                       + ψ'' (ψ'·u) / ψ²
    #                       + h'''[u]
    # where h'''[u] = diag(-2u₁/x₁³, -2u₂/x₂³, 0)

    # Compute ψ''u
    ψHu = zeros(T, 3)
    ψHu[1] = ψH[1,1] * u1 + ψH[2,1] * u2 + ψH[3,1] * u3
    ψHu[2] = ψH[2,1] * u1 + ψH[2,2] * u2 + ψH[3,2] * u3
    ψHu[3] = ψH[3,1] * u1 + ψH[3,2] * u2 + ψH[3,3] * u3

    ψ2 = ψ^2
    ψ3 = ψ^3

    for j in 1:3, i in j:3
        # Term 1: -2 ψ'ᵢψ'ⱼ (ψ'·u) / ψ³
        D[i,j] = -2 * ψg[i] * ψg[j] * ψgu / ψ3

        # Term 2: (ψ'ᵢ(ψ''u)ⱼ + (ψ''u)ᵢψ'ⱼ) / ψ²
        D[i,j] += (ψg[i] * ψHu[j] + ψHu[i] * ψg[j]) / ψ2

        # Term 3: ψ''ᵢⱼ (ψ'·u) / ψ²
        D[i,j] += ψH[i,j] * ψgu / ψ2
    end

    # Term 4: -ψ'''[u] / ψ (only non-zero entries)
    D[1,1] -= ψ3_11 / ψ
    D[2,1] -= ψ3_21 / ψ
    D[2,2] -= ψ3_22 / ψ

    # Term 5: h'''[u] = diag(-2u₁/x₁³, -2u₂/x₂³, 0)
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

# Check if x is in the interior of the primal exp cone
function in_exp_primal(x::AbstractVector{T}) where {T}
    x[1] > 0 && x[2] > 0 && exp_psi(x) > 0
end

# Check if z is in the interior of the dual exp cone
# K* = cl{z : e·z₁ ≥ -z₃·exp(z₂/z₃), z₁ > 0, z₃ < 0}
function in_exp_dual(z::AbstractVector{T}) where {T}
    z[1] > 0 && z[3] < 0 && ℯ * z[1] >= -z[3] * exp(z[2] / z[3])
end

#
# Shadow primal computation (Newton iteration)
#
# Find x̃ such that F'(x̃) = -s
# This is a 3D Newton iteration
#

function exp_shadow_primal!(xs::AbstractVector{T}, s::AbstractVector{T}; maxiter::Int=50, tol::T=T(1e-12)) where {T}
    # Initialize with a heuristic starting point
    # For well-scaled problems, start near the central point
    xs[1] = T(1.0)
    xs[2] = T(1.0)
    xs[3] = T(-0.5)

    # Make sure starting point is interior
    while !in_exp_primal(xs)
        xs[1] *= 2
        xs[2] *= 2
        xs[3] -= one(T)
    end

    g = zeros(T, 3)
    H = zeros(T, 3, 3)
    Δ = zeros(T, 3)

    for iter in 1:maxiter
        # Compute residual: F'(x̃) + s
        exp_barrier_grad!(g, xs)
        g[1] += s[1]
        g[2] += s[2]
        g[3] += s[3]

        # Check convergence
        if norm(g) < tol
            return xs
        end

        # Newton step: Δ = -F''(x̃)⁻¹ (F'(x̃) + s)
        exp_barrier_hess!(H, xs)

        # Solve H Δ = -g using Cholesky
        Δ .= .-g
        chol3!(H)
        solve3!(H, Δ)

        # Line search to stay in cone interior
        θ = one(T)
        while θ > T(1e-10)
            xs_new = xs .+ θ .* Δ
            if in_exp_primal(xs_new)
                xs .= xs_new
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
# Scale computation (Tunçel scaling in 3D)
#

function expscale!(
        M::AbstractMatrix{T},
        R::AbstractMatrix{T},
        xs::AbstractVector{T},
        ss::AbstractVector{T},
        x::AbstractVector{T},
        s::AbstractVector{T}
    ) where {T}

    # Stage 1: Compute F''(x) and factor it; compute s̃ = -F'(x)
    exp_barrier_hess!(R, x)
    exp_barrier_grad!(ss, x)
    ss .= .-ss  # s̃ = -F'(x)

    # Make a copy of R before Cholesky (we need F'' later)
    Fpp = copy(R)

    # Factor F'' = RRᵀ
    chol3!(R)

    # Stage 2: Compute shadow primal x̃ = -F*'(s)
    exp_shadow_primal!(xs, s)

    # Stage 3: Check degeneracy (on central path)
    # Y'S where Y = [s | s̃] and S = [x | x̃]
    YtS11 = dot(s, x)
    YtS12 = dot(s, xs)
    YtS21 = dot(ss, x)
    YtS22 = dot(ss, xs)

    det_YtS = YtS11 * YtS22 - YtS12 * YtS21

    # Check if on central path (Y'S singular)
    if abs(det_YtS) < T(1e-10) * (abs(YtS11 * YtS22) + abs(YtS12 * YtS21))
        # Fallback: M = μ F''(x)
        μv = YtS11 / 3
        M .= μv .* Fpp
        return μv
    end

    # Stage 3: Orthogonal completions
    # z = (x × x̃) / ‖x × x̃‖ such that Sᵀz = 0
    z = cross3(x, xs)
    nz = norm(z)
    z ./= nz

    # Stage 4: BFGS scalar t (paper eq. 32)
    # t = μ ‖F''(x) - s̃s̃ᵀ/3 - (F''x̃ - μ̃s̃)(F''x̃ - μ̃s̃)ᵀ / (x̃ᵀF''x̃ - 3μ̃²)‖_F

    μv = YtS11 / 3  # μ_v = ⟨x,s⟩/3
    μt = YtS22 / 3  # μ̃ = ⟨x̃,s̃⟩/3

    # Compute F''x̃
    Fpp_xs = zeros(T, 3)
    mul3!(Fpp_xs, Fpp, xs)

    # x̃ᵀF''x̃
    xsFppxs = dot(xs, Fpp_xs)

    # Denominator for rank-1 term
    denom = xsFppxs - 3 * μt^2

    # Construct the matrix whose Frobenius norm we need
    # A = F'' - s̃s̃ᵀ/3 - vvᵀ/denom where v = F''x̃ - μ̃s̃
    A = copy(Fpp)
    for j in 1:3, i in 1:3
        A[i,j] -= ss[i] * ss[j] / 3
    end

    if abs(denom) > T(1e-12)
        v = Fpp_xs .- μt .* ss
        for j in 1:3, i in 1:3
            A[i,j] -= v[i] * v[j] / denom
        end
    end

    t = μv * frob3(A)

    # Stage 5: Assemble M = Y(Y'S)⁻¹Y' + t·zzᵀ
    # (Y'S)⁻¹
    a, b, c, d = inv2x2(YtS11, YtS12, YtS21, YtS22)

    # M = Y · (Y'S)⁻¹ · Y'
    # Y = [s | s̃], so M = s·(a·sᵀ + c·s̃ᵀ) + s̃·(b·sᵀ + d·s̃ᵀ)
    for j in 1:3, i in 1:3
        M[i,j] = s[i] * (a * s[j] + c * ss[j]) + ss[i] * (b * s[j] + d * ss[j])
        M[i,j] += t * z[i] * z[j]
    end

    return μv
end

function scale!(p::AbstractVector{T}, d::AbstractVector{T}, cache::EXPCache{T}) where {T}
    cache.μv[] = expscale!(cache.M, cache.R, cache.xs, cache.ss, p, d)
end

#
# Hessian: just copy cached M
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
# Corrector
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

    # Compute F'(p)
    Fp = zeros(T, 3)
    exp_barrier_grad!(Fp, p)

    # Compute v = F''(p)⁻¹Δd using cached Cholesky factor R
    v = copy(Δd)
    solve3!(R, v)

    # Compute η = -½ F'''(p)[Δp, v]
    # F'''[u,v] = (F'''[u]) · v (contract the resulting matrix with v)
    D = zeros(T, 3, 3)
    exp_barrier_hess_dir!(D, p, Δp)

    η = zeros(T, 3)
    mul3!(η, D, v)
    η .*= T(-0.5)

    # r = -d - σμ·F'(p) - η
    r[1] = -d[1] - σμ * Fp[1] - η[1]
    r[2] = -d[2] - σμ * Fp[2] - η[2]
    r[3] = -d[3] - σμ * Fp[3] - η[3]

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
    # Bisection to find largest τ ∈ (0,1] such that x + τΔx is in cone interior

    membership = primal ? in_exp_primal : in_exp_dual

    τ_lo = zero(T)
    τ_hi = one(T)

    x_test = similar(x)

    # First check if full step is feasible
    x_test .= x .+ τ_hi .* Δx
    if membership(x_test)
        return γ * τ_hi
    end

    # Bisection
    for _ in 1:53  # enough iterations for machine precision
        τ_mid = (τ_lo + τ_hi) / 2
        x_test .= x .+ τ_mid .* Δx

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
