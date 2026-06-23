#
# SemidefiniteCone (PSD cone 𝕊ᵈ₊)
#

struct SemidefiniteCone <: Cone end

struct SemidefiniteConeCache{T} <: AbstractCache{SemidefiniteCone}
    cone::SemidefiniteCone
    LP::FMatrixView{T}    # lower triangular Cholesky factor of P (d×d)
    LD::FMatrixView{T}    # lower triangular Cholesky factor of D (d×d)
    U::FMatrixView{T}     # orthogonal matrix from SVD (d×d)
    s::FVectorView{T}     # singular values (d)
end

#
# SDP math utilities
#

# triangular number inverse
function triroot(n::Integer)
    return (isqrt(1 + 8n) - 1) ÷ 2
end

function roottwo(::Type{T}) where {T}
    return sqrt(T(2))
end

# symmetrize a matrix by copying lower to upper
function symmetrize!(M::AbstractMatrix)
    for j in axes(M, 1)
        for i in 1:j - 1
            M[i, j] = M[j, i]
        end
    end

    return M
end

# svec: vectorize symmetric matrix with √2 scaling on off-diagonals (lower triangle)
function svec!(v::AbstractVector{T}, M::AbstractMatrix{T}) where {T}
    n = size(M, 1); k = 0
    α = roottwo(T)

    for j in 1:n
        k += 1; v[k] = M[j, j]

        for i in j + 1:n
            k += 1; v[k] = α * M[i, j]
        end
    end

    return v
end

# smat: inverse of svec (unvectorize into lower triangle of matrix)
function smat!(M::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
    n = size(M, 1); k = 0
    α = roottwo(T)

    for j in 1:n
        k += 1; M[j, j] = v[k]

        for i in j + 1:n
            k += 1; M[i, j] = v[k] / α
        end
    end

    return M
end

# symmetric Kronecker product: H = B ⊗ₛ B (fills full matrix; symmetric iff B is)
# svec(B X B') = (B ⊗ₛ B) svec(X)
function skron!(H::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    α = roottwo(T)
    tll = 1

    @inbounds for l in 1:n
        tij = 0

        for j in 1:n
            Ajl = A[j, l]

            tij += 1; H[tij, tll] = Ajl^2

            for i in j + 1:n
                tij += 1; H[tij, tll] = α * A[i, l] * Ajl
            end
        end

        tkl = tll

        for k in l + 1:n
            tkl += 1; tij = 0

            for j in 1:n
                Ajk = A[j, k]
                Ajl = A[j, l]

                tij += 1; H[tij, tkl] = α * Ajk * Ajl

                for i in j + 1:n
                    tij += 1; H[tij, tkl] = A[i, k] * Ajl + A[i, l] * Ajk
                end
            end
        end

        tll += n - l + 1
    end

    return H
end

#
# SDP cone interface
#

# degree = triroot(n) where n = d(d+1)/2
function degree(::SemidefiniteCone, n::Int)
    return triroot(n)
end

# cache size: LP(d²) + LD(d²) + U(d²) + s(d) = 3d² + d
function cachesize(::SemidefiniteCone, n::Int)
    d = triroot(n)
    return 3 * d^2 + d
end

# construct view-based cache from Caches
function cache(c::Caches{T}, i::Int, cone::SemidefiniteCone) where T
    n = c.xcol[i+1] - c.xcol[i]
    d = triroot(n)
    data = view(c.val, c.xblk[i]:c.xblk[i+1]-1)

    # Layout: LP(d²), LD(d²), U(d²), s(d)
    d2 = d^2
    LP = reshape(view(data, 1:d2), d, d)
    LD = reshape(view(data, d2+1:2d2), d, d)
    U  = reshape(view(data, 2d2+1:3d2), d, d)
    s  = view(data, 3d2+1:3d2+d)

    SemidefiniteConeCache(cone, LP, LD, U, s)
end

function identity!(x::AbstractVector{T}, ::SemidefiniteCone) where {T}
    d = triroot(length(x))
    k = 1

    fill!(x, zero(T))

    for j in 1:d
        x[k] = one(T); k += d - j + 1
    end

    return x
end

function sdpscale!(
        LP::AbstractMatrix{T},
        LD::AbstractMatrix{T},
        U::AbstractMatrix{T},
        s::AbstractVector{T},
        p::AbstractVector,
        d::AbstractVector
    ) where {T}
    n = size(LP, 1)

    V = zeros(T, n, n)
    work = zeros(T, 1)
    iwork = zeros(BlasInt, 8n)

    smat!(LP, p)
    smat!(LD, d)
    #
    # factorize P:
    #
    #   P = LP LPᵀ
    #
    cholesky!(Symmetric(LP, :L))
    #
    # factorize D:
    #
    #   D = LD LDᵀ
    #
    cholesky!(Symmetric(LD, :L))
    #
    # factorize the product LPᵀ LD
    #
    #   LPᵀ LD = U Σ Vᵀ
    #
    tril!(LD)
    mul!(U, LowerTriangular(LP)', LD)
    svd!(s, U, V, work, iwork)

    return
end

function scale!(H::AbstractMatrix{T}, p::AbstractVector{T}, d::AbstractVector{T}, cache::SemidefiniteConeCache{T}) where {T}
    sdpscale!(cache.LP, cache.LD, cache.U, cache.s, p, d)

    # compute H_v = W⁻¹ ⊗ₛ W⁻¹
    # W⁻¹ = L⁻ᵀ U Σ Uᵀ L⁻¹ where W = R Rᵀ with R = L U Σ^{-1/2}
    L, U, s = cache.LP, cache.U, cache.s
    n = size(L, 1)

    W = zeros(T, n, n)
    X = zeros(T, n, n)

    mul!(W, U, Diagonal(s))
    mul!(X, W, U')
    ldiv!(LowerTriangular(L)', X)
    rdiv!(X, LowerTriangular(L))

    return skron!(H, X)
end

# H·R_c = L⁻ᵀ U [σμI - Σ² - Σ B_mat Σ] Uᵀ L⁻¹
function sdpcorr!(
        r::AbstractVector{T},
        L::AbstractMatrix{T},
        U::AbstractMatrix{T},
        s::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real
    ) where {T}
    n = size(L, 1)

    ΔP = zeros(T, n, n)
    ΔD = zeros(T, n, n)
    W = zeros(T, n, n)
    X = zeros(T, n, n)

    smat!(ΔP, Δp)
    smat!(ΔD, Δd)

    mul!(W, Symmetric(ΔD, :L), LowerTriangular(L))
    mul!(X, Symmetric(ΔP, :L), W)
    ldiv!(LowerTriangular(L), X)

    mul!(W, U', X)
    mul!(X, W, U)

    for j in 1:n
        sj = s[j]

        for i in 1:j - 1
            si = s[i]
            W[i, j] = W[j, i] = -weightedmean(si, sj, X[i, j], X[j, i])
        end

        W[j, j] = σμ - sj^2 - X[j, j]
    end

    mul!(X, W, U')
    mul!(W, U, X)

    ldiv!(LowerTriangular(L)', W)
    rdiv!(W, LowerTriangular(L))

    svec!(r, W)
    return r
end

function corr!(
        r::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        cache::SemidefiniteConeCache{T}
    ) where {T}
    return sdpcorr!(r, cache.LP, cache.U, cache.s, Δp, Δd, σμ)
end

# Find the largest number 0 < τ ≤ 1 such that
#
#   L Lᵀ + τ ΔX = L (I + τ M) Lᵀ
#
# is positive definite, where
#
#   M = L⁻¹ ΔX L⁻ᵀ.
#
# This matrix is positive definite if and
# only if M is, so the solution is given by
#
#   τ⁻¹ = max {1, -λ},
#
# where λ is the smallest eigenvalue of L⁻¹ ΔX L⁻ᵀ.
function sdpmaxstep(L::LowerTriangular{T}, Δx::AbstractVector{T}) where {T}
    n = size(L, 1)

    M = zeros(T, n, n)
    W = Vector{T}(undef, n)
    work = zeros(T, 1)
    iwork = zeros(BlasInt, 1)
    #
    # compute the product
    #
    #   M = L⁻¹ ΔX L⁻ᵀ
    #
    smat!(M, Δx)
    symmetrize!(M)
    ldiv!(L, M)
    rdiv!(M, L')
    #
    # λ is the smallest eigenvalue of M
    #
    λ = eigmin!(M, W, work, iwork)

    return inv(max(one(T), -λ))
end

function maxsteps(::AbstractVector{T}, Δp::AbstractVector{T}, ::AbstractVector{T}, Δd::AbstractVector{T}, cache::SemidefiniteConeCache{T}) where {T}
    τp = sdpmaxstep(LowerTriangular(cache.LP), Δp)
    τd = sdpmaxstep(LowerTriangular(cache.LD), Δd)
    return τp, τd
end
