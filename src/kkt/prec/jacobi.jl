struct Jacobi{UPLO, T, I <: Integer} <: AbstractPreconditioner{T}
    R::BlockSparseMatrix{T, I}
end

function Jacobi{UPLO}(B::BlockSparseMatrix{T, I}) where {UPLO, T, I}
    R = cholblockdiag(B, UPLO, false)
    return Jacobi{UPLO, T, I}(R)
end

function Jacobi(B::BlockSparseMatrix)
    return Jacobi{:L}(B)
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
