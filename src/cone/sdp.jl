#
# SDP cone (PSD cone 𝕊ᵈ₊)
#

struct SDP <: Cone end

struct SDPCache{T} <: AbstractCache{SDP}
    cone::SDP
    LP::FMatrixView{T}    # lower triangular Cholesky factor of P (d×d)
    LD::FMatrixView{T}    # lower triangular Cholesky factor of D (d×d)
    U::FMatrixView{T}     # orthogonal matrix from SVD (d×d)
    s::FVectorView{T}     # singular values (d)
end

#
# SDP math utilities
#

# triangular number inverse
triroot(n::Integer) = (isqrt(1 + 8n) - 1) ÷ 2
roottwo(::Type{T}) where {T} = sqrt(T(2))

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

# symmetric Kronecker product: H = B ⊗ₛ B (lower triangle)
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

# compute NT scaling factors for a single block via Cholesky + SVD
#
# L_P = chol(P),  L_D = chol(D)
# G = L_Pᵀ L_D
# SVD: G = U Σ Vᵀ
#
# Stores L_P, L_D, U, and Σ (singular values). These define:
#   R = L_P U Σ^{-1/2}
#   W = R Rᵀ  (NT scaling, satisfies W D W = P)
#   W⁻¹ = L_P⁻ᵀ U Σ U' L_P⁻¹
function meanblock!(
        LP_out::AbstractMatrix{T},
        LD_out::AbstractMatrix{T},
        U_out::AbstractMatrix{T},
        s_out::AbstractVector{T},
        P::AbstractMatrix{T},
        D::AbstractMatrix{T}
    ) where {T}
    # L_P = chol(P)
    copyto!(LP_out, P)
    cholesky!(Symmetric(LP_out, :L))
    L_P = LowerTriangular(LP_out)

    # L_D = chol(D)
    copyto!(LD_out, D)
    cholesky!(Symmetric(LD_out, :L))
    L_D = LowerTriangular(LD_out)

    # G = L_Pᵀ L_D
    G = L_P' * L_D

    # SVD: G = U Σ Vᵀ
    F = svd(G)

    # Store U and singular values
    copyto!(U_out, F.U)
    copyto!(s_out, F.S)

    return
end

# compute H_v = W⁻¹ ⊗ₛ W⁻¹
# W⁻¹ = L_P⁻ᵀ U Σ U' L_P⁻¹ where W = R R' with R = L_P U Σ^{-1/2}
function hessblock!(
        H::AbstractMatrix{T},
        LP::AbstractMatrix{T},
        U::AbstractMatrix{T},
        s::AbstractVector{T}
    ) where {T}
    L_P = LowerTriangular(LP)

    # W⁻¹ = L_P⁻ᵀ U Σ U' L_P⁻¹
    UΣUt = U * Diagonal(s) * U'
    Winv = L_P' \ UΣUt / L_P

    return skron!(H, Winv)
end

#
# SDP cone interface
#

# degree = triroot(n) where n = d(d+1)/2
degree(::SDP, n::Int) = triroot(n)

# cache size: LP(d²) + LD(d²) + U(d²) + s(d) = 3d² + d
function cachesize(::SDP, n::Int)
    d = triroot(n)
    return 3 * d^2 + d
end

# construct view-based cache from Caches
function cache(c::Caches{T}, i::Int, cone::SDP) where T
    n = c.xcol[i+1] - c.xcol[i]
    d = triroot(n)
    data = view(c.val, c.xblk[i]:c.xblk[i+1]-1)

    # Layout: LP(d²), LD(d²), U(d²), s(d)
    d2 = d^2
    LP = reshape(view(data, 1:d2), d, d)
    LD = reshape(view(data, d2+1:2d2), d, d)
    U  = reshape(view(data, 2d2+1:3d2), d, d)
    s  = view(data, 3d2+1:3d2+d)

    SDPCache(cone, LP, LD, U, s)
end

function identity!(x::AbstractVector{T}, ::SDP) where {T}
    d = triroot(length(x))
    fill!(x, zero(T))
    k = 1
    for j in 1:d
        x[k] = one(T)
        k += d - j + 1
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
    d_v = size(LP, 1)

    P = zeros(T, d_v, d_v)
    D = zeros(T, d_v, d_v)
    smat!(P, p)
    smat!(D, d)

    meanblock!(LP, LD, U, s, Symmetric(P, :L), Symmetric(D, :L))
    return
end

function scale!(p::AbstractVector, d::AbstractVector, cache::SDPCache{T}) where {T}
    sdpscale!(cache.LP, cache.LD, cache.U, cache.s, p, d)
end

function hess!(
        H::AbstractMatrix{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        cache::SDPCache{T}
    ) where {T}
    hessblock!(H, cache.LP, cache.U, cache.s)
    return H
end

# H·R_c = L_P⁻ᵀ U [σμI - Σ² - Σ B_mat Σ] Uᵀ L_P⁻¹
function sdpcorr!(
        r::AbstractVector{T},
        LP::AbstractMatrix{T},
        U::AbstractMatrix{T},
        s::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real
    ) where {T}
    d_v = size(LP, 1)

    ΔP_v = zeros(T, d_v, d_v)
    ΔD_v = zeros(T, d_v, d_v)

    smat!(ΔP_v, Δp)
    smat!(ΔD_v, Δd)

    L_P = LowerTriangular(LP)

    X = L_P \ (Symmetric(ΔP_v, :L) * Symmetric(ΔD_v, :L) * L_P)
    Y = (U' * X * U) ./ s'
    M = (Y + Y') ./ (s .+ s')

    M .*= -(s .* s')
    for i in 1:d_v
        M[i,i] += σμ - s[i]^2
    end

    H_R_c = L_P' \ (U * M * U') / L_P

    svec!(r, H_R_c)
    return r
end

function corr!(
        r::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        cache::SDPCache{T}
    ) where {T}
    sdpcorr!(r, cache.LP, cache.U, cache.s, Δp, Δd, σμ)
end

function sdpmaxstep(
        LP::AbstractMatrix{T},
        LD::AbstractMatrix{T},
        Δx::AbstractVector{T},
        primal::Bool,
        γ::Real
    ) where {T}
    d_v = size(LP, 1)

    if primal
        L = LowerTriangular(LP)
    else
        L = LowerTriangular(LD)
    end

    ΔX = zeros(T, d_v, d_v)
    smat!(ΔX, Δx)

    # M = L⁻¹ ΔX L⁻ᵀ
    M = L \ Symmetric(ΔX, :L) / L'
    M = (M + M') / 2

    λ_min = eigmin(Symmetric(M, :L))

    if λ_min < 0
        return min(one(T), -γ / λ_min)
    else
        return one(T)
    end
end

function maxstep(
        ::AbstractVector{T},
        Δx::AbstractVector{T},
        primal::Bool,
        γ::Real,
        cache::SDPCache{T}
    ) where {T}
    sdpmaxstep(cache.LP, cache.LD, Δx, primal, γ)
end
