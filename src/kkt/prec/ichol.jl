@kwdef struct ICholSettings{T} <: PreconditionerSettings{T}
    areg::T = zero(T)
    rreg::T = zero(T)
end

struct IChol{UPLO, T, I <: Integer} <: Preconditioner{T}
    L::BlockSparseMatrix{T, I}
    z::Vector{T}
end

function IChol{UPLO}(L::BlockSparseMatrix{T, I}, z::Vector{T}) where {UPLO, T, I}
    return IChol{UPLO, T, I}(L, z)
end

const ICHOL_SCHEDULE = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

function IChol{UPLO}(B::BlockSparseMatrix{T}; α::T = one(T)) where {UPLO, T}
    z = zeros(T, size(B, 2))

    C = blocktri(B' * B, Val(UPLO))
    axpy!(α, I, C)

    D = copyblockdiag(C)
    cholblockdiag!(D, UPLO)

    ldivblockdiag!(C, D, Val(UPLO))
    rdivblockdiag!(C, D, Val(UPLO))

    ichol!(C, Val(UPLO))

    for v in vtxs(C)
        tril!(block(C, v, v, C.xsrc[v]))
    end

    if UPLO === :L
        lmulblockdiag!(C, D, Val(UPLO))
    else
        rmulblockdiag!(C, D, Val(UPLO))
    end

    return IChol{UPLO}(C, z)
end

function IChol(B::BlockSparseMatrix{T}; α::T = one(T)) where {T}
    return IChol{:L}(B; α)
end

function IChol{UPLO}(B::BlockSparseMatrix{T}, set::ICholSettings{T}) where {UPLO, T}
    α = set.areg + set.rreg * norm(B)^2
    return IChol{UPLO}(B; α)
end

function IChol(B::BlockSparseMatrix{T}, set::ICholSettings{T}) where {T}
    return IChol{:L}(B, set)
end

function make_prec(set::ICholSettings{T}, B::BlockSparseMatrix{T, I}) where {T, I}
    weights, graph = weightedgraph(B)
    R, P, S = symbolic(weights, graph)
    B = selectvtxs(B, R.perm)
    M = IChol(B, set)
    return R, P, B, M
end

function ichol!(L::BlockSparseMatrix{T}, uplo::Val) where {T}
    W = similar(L)

    for α in ICHOL_SCHEDULE
        ichol_impl!(copyto!(W, L), α, uplo) && return copyto!(L, W) 
    end

    error()
end

function ichol_impl!(L::BlockSparseMatrix{T, I}, α::Number, uplo::Val{:L}) where {T, I}
    for v in vtxs(L)
        estrt = L.xsrc[v]
        estop = L.xsrc[v + one(I)] - one(I)        

        Lvv = block(L, v, v, estrt)

        for i in diagind(Lvv)
            Lvv[i] += α
        end

        Fvv = cholesky!(Symmetric(Lvv, :L); check=false) 
        issuccess(Fvv) || return false

        for e in estrt + one(I):estop
            u = L.tgt[e]
            rdiv!(block(L, u, v, e), LowerTriangular(Lvv)')
        end

        for e in estrt + one(I):estop
            u = L.tgt[e]

            Luv = block(L, u, v, e)

            estrtu = L.xsrc[u]
            estopu = L.xsrc[u + one(I)] - one(I)

            eu = estrtu

            for ev in e:estop
                while eu ≤ estopu && L.tgt[eu] < L.tgt[ev]
                    eu += one(I)
                end

                eu ≤ estopu || break

                uu = L.tgt[eu]
                uv = L.tgt[ev]

                if uu == uv
                    mul!(block(L, uu, u, eu), block(L, uv, v, ev), Luv', -one(T), one(T))
                end                        
            end
        end
    end

    return true
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::IChol{UPLO}, x::AbstractVector) where {UPLO}
    copyto!(y, x)

    if UPLO === :L
        L = LowerTriangular(P.L)
    else
        L = UpperTriangular(P.L)'
    end

    ldiv!(L,  y)
    ldiv!(L', y)

    return y
end
