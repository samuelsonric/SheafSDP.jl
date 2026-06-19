struct SSORWorkspace{UPLO, T, I <: Integer}
    R::BlockSparseMatrix{T, I}
    z::Vector{T}
    q::Vector{T}
    s::Vector{T}
end

function SSORWorkspace{UPLO}(B::BlockSparseMatrix{T, J}; α::T = zero(T)) where {UPLO, T, J <: Integer}
    m, n = size(B)

    R = congblockdiag(B)
    axpy!(α, I, R)
    cholblockdiag!(R, UPLO)
    z = zeros(T, n)
    q = zeros(T, m)
    s = zeros(T, maximum(ncols(B, v) for v in vtxs(B)))

    return SSORWorkspace{UPLO, T, J}(R, z, q, s)
end

function SSORWorkspace(B::BlockSparseMatrix{T}; α::T = zero(T)) where {T}
    return SSORWorkspace{:L}(B; α)
end

@kwdef struct SSORSettings{T} <: PreconditionerSettings{T}
    areg::T = zero(T)
    rreg::T = zero(T)
    relax::T = one(T)
end

struct SSOR{UPLO, T, I <: Integer} <: Preconditioner{T}
    wrk::SSORWorkspace{UPLO, T, I}
    set::SSORSettings{T}
    B::BlockSparseMatrix{T, I}
end

function SSOR{UPLO}(B::BlockSparseMatrix{T, J}, set::SSORSettings{T}) where {UPLO, T, J <: Integer}
    α = set.areg + set.rreg * norm(B)^2
    wrk = SSORWorkspace{UPLO}(B; α)
    return SSOR{UPLO, T, J}(wrk, set, B)
end

function SSOR(B::BlockSparseMatrix{T}, set::SSORSettings{T}) where {T}
    return SSOR{:L}(B, set)
end

function SSOR{UPLO}(B::BlockSparseMatrix{T, J}; α::T = zero(T), ω::T = one(T)) where {UPLO, T, J <: Integer}
    set = SSORSettings{T}(areg=α, rreg=zero(T), relax=ω)
    wrk = SSORWorkspace{UPLO}(B; α)
    return SSOR{UPLO, T, J}(wrk, set, B)
end

function SSOR(B::BlockSparseMatrix{T}; α::T = zero(T), ω::T = one(T)) where {T}
    return SSOR{:L}(B; α, ω)
end

function make_prec(set::SSORSettings{T}, B::BlockSparseMatrix{T, I}) where {T, I}
    R = NaturalPermutation{I}(nvtxs(B))
    P = NaturalPermutation{I}(ncols(B))
    M = SSOR(B, set)
    return R, P, B, M
end

function ssor_sweep!(
        z::AbstractVector,
        q::AbstractVector,
        s::AbstractVector,
        R::BlockSparseMatrix,
        B::BlockSparseMatrix,
        r::AbstractVector,
        ω::Number,
        ::Val{UPLO},
        ::Val{FWD},
    ) where {UPLO, FWD}
    m, n = size(B)

    @assert length(z) == n
    @assert length(q) == m
    @assert length(r) == n

    if FWD
        vr =         vtxs(B)
    else
        vr = reverse(vtxs(B))
    end

    @inbounds for v in vr
        cols = colrange(B, v)
        ncol =    ncols(B, v)

        sv = copyto!(view(s, oneto(ncol)), view(r, cols))

        if FWD
            er =         srcrange(B, v)
        else
            er = reverse(srcrange(B, v))
        end

        for e in er
            u = B.tgt[e]
            rows = rowrange(B, u)
            mul!(sv, block(B, u, v, e)', view(q, rows), -1, 1)
        end

        if UPLO === :L
            Rv = LowerTriangular(block(R, v, v, v))
        else
            Rv = UpperTriangular(block(R, v, v, v))
        end

        ldiv!(Rv,  sv)
        ldiv!(Rv', sv)
        lmul!(ω,   sv)
        axpy!(1, sv, view(z, cols))

        for e in er
            u = B.tgt[e]
            rows = rowrange(B, u)
            mul!(view(q, rows), block(B, u, v, e), sv, 1, 1)
        end
    end

    return z
end

function ssor_impl!(
        z::AbstractVector,
        q::AbstractVector,
        s::AbstractVector,
        R::BlockSparseMatrix,
        B::BlockSparseMatrix,
        r::AbstractVector,
        ω::Number,
        uplo::Val{UPLO},
    ) where {UPLO}
    fill!(z, false)
    fill!(q, false)

    ssor_sweep!(z, q, s, R, B, r, ω, uplo, Val(true))
    ssor_sweep!(z, q, s, R, B, r, ω, uplo, Val(false))

    rmul!(z, ω * (2 - ω))

    return z
end

function ssor!(wrk::SSORWorkspace{UPLO, T}, B::BlockSparseMatrix{T}, r::AbstractVector{T}, set::SSORSettings{T}) where {UPLO, T}
    return ssor_impl!(wrk.z, wrk.q, wrk.s, wrk.R, B, r, set.relax, Val(UPLO))
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::SSOR, x::AbstractVector)
    ssor!(P.wrk, P.B, x, P.set)
    copyto!(y, P.wrk.z)
    return y
end
