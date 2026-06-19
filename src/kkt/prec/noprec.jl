struct NoPrecSettings{T} <: PreconditionerSettings{T} end
NoPrecSettings() = NoPrecSettings{Float64}()

struct NoPrec{T} <: Preconditioner{T} end

function make_prec(::NoPrecSettings{T}, B::BlockSparseMatrix{T, I}) where {T, I}
    R = NaturalPermutation{I}(nvtxs(B))
    P = NaturalPermutation{I}(ncols(B))
    return R, P, B, NoPrec{T}()
end

LinearAlgebra.ldiv!(y::AbstractVector, ::NoPrec, x::AbstractVector) = copyto!(y, x)
