#
# POS cone (nonnegative orthant ℝⁿ₊)
#

struct POS <: Cone end

struct POSCache <: AbstractCache{POS}
    cone::POS
end

function POSCache()
    return POSCache(POS())
end

function degree(::POS, n::Int)
    return n
end

function cachesize(::POS, n::Int)
    return 0
end

function cache(::Caches, ::Int, c::POS)
    return POSCache(c)
end

function identity!(x::AbstractVector{T}, ::POS) where {T}
    fill!(x, one(T))
    return x
end

function scale!(::AbstractVector, ::AbstractVector, ::POSCache)
    return
end

function poshess!(H::AbstractMatrix{T}, p::AbstractVector{T}, d::AbstractVector{T}) where {T}
    fill!(H, zero(T))

    for i in eachindex(p)
        H[i, i] = d[i] / p[i]
    end

    return H
end

function hess!(
        H::AbstractMatrix{T},
        p::AbstractVector{T},
        d::AbstractVector{T},
        ::POSCache
    ) where {T}
    poshess!(H, p, d)
end

function poscorr!(
        r::AbstractVector{T},
        p::AbstractVector{T},
        d::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real
    ) where {T}
    for i in eachindex(r)
        r[i] = (σμ - Δp[i] * Δd[i]) / p[i] - d[i]
    end

    return r
end

function corr!(
        r::AbstractVector{T},
        p::AbstractVector{T},
        d::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        ::POSCache
    ) where {T}
    return poscorr!(r, p, d, Δp, Δd, σμ)
end

function posmaxstep(x::AbstractVector{T}, Δx::AbstractVector{T}, γ::Real) where {T}
    τ = one(T)

    for i in eachindex(x)
        Δxi = Δx[i]

        if Δxi < 0
            τ = min(τ, -γ * x[i] / Δxi)
        end
    end

    return τ
end

function maxstep(
        x::AbstractVector{T},
        Δx::AbstractVector{T},
        ::Bool,
        γ::Real,
        ::POSCache
    ) where {T}
    return posmaxstep(x, Δx, γ)
end
