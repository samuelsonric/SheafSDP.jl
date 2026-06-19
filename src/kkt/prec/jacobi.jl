struct Jacobi{UPLO, T, I <: Integer} <: AbstractPreconditioner{T}
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
