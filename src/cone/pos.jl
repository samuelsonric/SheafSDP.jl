#
# PositiveCone (nonnegative orthant ℝⁿ₊)
#

struct PositiveCone <: Cone end

struct PositiveConeCache <: AbstractCache{PositiveCone}
    cone::PositiveCone
end

function PositiveConeCache()
    return PositiveConeCache(PositiveCone())
end

function degree(::PositiveCone, n::Int)
    return n
end

function cachesize(::PositiveCone, n::Int)
    return 0
end

function cache(::Caches, ::Int, c::PositiveCone)
    return PositiveConeCache(c)
end

function identity!(x::AbstractVector{T}, ::PositiveCone) where {T}
    fill!(x, one(T))
    return x
end

function scale!(H::AbstractMatrix{T}, p::AbstractVector{T}, d::AbstractVector{T}, ::PositiveConeCache) where {T}
    fill!(H, zero(T))
    for i in eachindex(p)
        H[i, i] = d[i] / p[i]
    end
    return H
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
        ::PositiveConeCache
    ) where {T}
    return poscorr!(r, p, d, Δp, Δd, σμ)
end

function posmaxstep(x::AbstractVector{T}, Δx::AbstractVector{T}) where {T}
    τ = one(T)

    for i in eachindex(x)
        Δxi = Δx[i]

        if Δxi < 0
            τ = min(τ, -x[i] / Δxi)
        end
    end

    return τ
end

function maxsteps(p::AbstractVector{T}, Δp::AbstractVector{T}, d::AbstractVector{T}, Δd::AbstractVector{T}, ::PositiveConeCache) where {T}
    return posmaxstep(p, Δp), posmaxstep(d, Δd)
end
