@kwdef struct JacobiSettings{T} <: PreconditionerSettings{T}
    areg::T = zero(T)
    rreg::T = zero(T)
end

struct Jacobi{UPLO, T, I <: Integer} <: Preconditioner{T}
    R::BlockSparseMatrix{T, I}
end

function Jacobi{UPLO}(B::BlockSparseMatrix{T, J}; α::T = zero(T)) where {UPLO, T, J}
    R = congblockdiag(B)
    axpy!(α, I, R)
    cholblockdiag!(R, UPLO)
    return Jacobi{UPLO, T, J}(R)
end

function Jacobi(B::BlockSparseMatrix{T}; α::T = zero(T)) where {T}
    return Jacobi{:L}(B; α)
end

function Jacobi{UPLO}(B::BlockSparseMatrix{T}, set::JacobiSettings{T}) where {UPLO, T}
    α = set.areg + set.rreg * norm(B)^2
    return Jacobi{UPLO}(B; α)
end

function Jacobi(B::BlockSparseMatrix{T}, set::JacobiSettings{T}) where {T}
    return Jacobi{:L}(B, set)
end

function make_prec(set::JacobiSettings{T}, B::BlockSparseMatrix{T, I}) where {T, I}
    R = NaturalPermutation{I}(nvtxs(B))
    P = NaturalPermutation{I}(ncols(B))
    M = Jacobi(B, set)
    return R, P, B, M
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::Jacobi{UPLO}, x::AbstractVector) where {UPLO}
    copyto!(y, x)

    if UPLO === :L
        L = LowerTriangular(P.R)
    else
        L = UpperTriangular(P.R)'
    end

    ldiv!(L,  y)
    ldiv!(L', y)

    return y
end
