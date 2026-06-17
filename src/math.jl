#
# Basic mathematical utilities
#

#
# triangular number and its inverse
#
trinum(n::Integer) = n * (n + 1) ÷ 2
triroot(n::Integer) = (isqrt(1 + 8n) - 1) ÷ 2
roottwo(::Type{T}) where {T} = sqrt(T(2))

#
# symmetrize a matrix by copying off-diagonals
#
function symmetrize!(M::AbstractMatrix, uplo::Val{UPLO}) where {UPLO}
    for j in axes(M, 1)
        for i in 1:j - 1
            if UPLO === :L
                M[i, j] = M[j, i]
            else
                M[j, i] = M[i, j]
            end
        end
    end
    return M
end

#
# svec: vectorize symmetric matrix with √2 scaling on off-diagonals
#
function svec!(v::AbstractVector{T}, M::AbstractMatrix{T}, uplo::Val{UPLO}) where {UPLO, T}
    n = size(M, 1); k = 0
    α = roottwo(T)

    for j in 1:n
        if UPLO === :L
            k += 1; v[k] = M[j, j]
        end

        if UPLO === :L
            r = j + 1:n
        else
            r = 1:j - 1
        end

        for i in r
            k += 1; v[k] = α * M[i, j]
        end

        if UPLO === :U
            k += 1; v[k] = M[j, j]
        end
    end
    return v
end

#
# smat: inverse of svec (unvectorize into symmetric matrix)
#
function smat!(M::AbstractMatrix{T}, v::AbstractVector{T}, uplo::Val{UPLO}) where {UPLO, T}
    n = size(M, 1); k = 0
    α = roottwo(T)

    for j in 1:n
        if UPLO === :L
            k += 1; M[j, j] = v[k]
        end

        if UPLO === :L
            r = j + 1:n
        else
            r = 1:j - 1
        end

        for i in r
            k += 1; M[i, j] = v[k] / α
        end

        if UPLO === :U
            k += 1; M[j, j] = v[k]
        end
    end
    return v
end

#
# symmetric Kronecker product: H = B ⊗ₛ B
#
# svec(B X B') = (B ⊗ₛ B) svec(X)
#
function skron!(H::AbstractMatrix{T}, A::AbstractMatrix{T}, uplo::Val{UPLO}) where {T, UPLO}
    n = size(A, 1)
    α = roottwo(T)
    tll = 1

    @inbounds for l in 1:n
        tij = 0

        for j in 1:n
            Ajl = A[j, l]

            if UPLO === :L
                tij += 1; H[tij, tll] = Ajl^2
            end

            if UPLO === :L
                r = j + 1:n
            else
                r = 1:j - 1
            end

            for i in r
                tij += 1; H[tij, tll] = α * A[i, l] * Ajl
            end

            if UPLO === :U
                tij += 1; H[tij, tll] = Ajl^2
            end
        end

        if UPLO === :L
            tkl = tll
        else
            tkl = tll - l
        end

        if UPLO === :L
            s = l + 1:n
        else
            s = 1:l - 1
        end

        for k in s
            tkl += 1; tij = 0

            for j in 1:n
                Ajk = A[j, k]
                Ajl = A[j, l]

                if UPLO === :L
                    tij += 1; H[tij, tkl] = α * Ajk * Ajl
                end

                if UPLO === :L
                    r = j + 1:n
                else
                    r = 1:j - 1
                end

                for i in r
                    tij += 1; H[tij, tkl] = A[i, k] * Ajl + A[i, l] * Ajk
                end

                if UPLO === :U
                    tij += 1; H[tij, tkl] = α * Ajk * Ajl
                end
            end
        end

        if UPLO === :L
            tll += n - l + 1
        else
            tll += l + 1
        end
    end

    return H
end

#
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
#
function meanblock!(LP_out::AbstractMatrix{T}, LD_out::AbstractMatrix{T},
                    U_out::AbstractMatrix{T}, s_out::AbstractVector{T},
                    P::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
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

#
# compute H_v = W⁻¹ ⊗ₛ W⁻¹
#
# W⁻¹ = L_P⁻ᵀ U Σ U' L_P⁻¹ where W = R R' with R = L_P U Σ^{-1/2}
#
function hessblock!(H::AbstractMatrix{T}, LP::AbstractMatrix{T}, U::AbstractMatrix{T},
                    s::AbstractVector{T}, work::AbstractMatrix{T}, uplo::Val{UPLO}) where {T, UPLO}
    L_P = LowerTriangular(LP)

    # W⁻¹ = L_P⁻ᵀ U Σ U' L_P⁻¹
    mul!(work, U * Diagonal(s), U')      # work = U Σ U'
    Winv = L_P' \ work / L_P             # W⁻¹ = L_P⁻ᵀ work L_P⁻¹

    return skron!(H, Winv, uplo)
end

#
# step length to boundary for a single SDP block
#
# Computes largest τ such that X + τ ΔX ⪰ 0
# where X is SPD and ΔX is symmetric
#
# M = L⁻¹ ΔX L⁻ᵀ where X = L Lᵀ
# λ_min = minimum eigenvalue of M
# τ_max = (λ_min < 0) ? -γ/λ_min : 1.0
#
function step_length_block(L::LowerTriangular{T}, ΔX::AbstractMatrix{T}, γ::Real) where {T}
    # M = L⁻¹ ΔX L⁻ᵀ (L is precomputed Cholesky factor)
    M = L \ ΔX / L'

    # Symmetrize M (for numerical stability)
    M = (M + M') / 2

    # Minimum eigenvalue (only need the smallest one)
    λ_min = eigmin(Symmetric(M))

    # Step length
    if λ_min < 0
        return min(one(T), -γ / λ_min)
    else
        return one(T)
    end
end
