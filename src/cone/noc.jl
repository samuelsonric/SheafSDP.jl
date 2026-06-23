#
# CofreeCone (no cone / free variables, K = ℝⁿ, K* = {0})
#

struct CofreeCone <: Cone end

struct CofreeConeCache <: AbstractCache{CofreeCone}
    cone::CofreeCone
end

function CofreeConeCache()
    return CofreeConeCache(CofreeCone())
end

function degree(::CofreeCone, n::Int)
    return 0
end

function cachesize(::CofreeCone, n::Int)
    return 0
end

function cache(::Caches, ::Int, c::CofreeCone)
    return CofreeConeCache(c)
end

function identity!(x::AbstractVector{T}, ::CofreeCone) where {T}
    fill!(x, zero(T))
    return x
end

function scale!(H::AbstractMatrix{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::CofreeConeCache) where {T}
    fill!(H, zero(T))
    return H
end

function corr!(
        r::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        ::Real,
        ::CofreeConeCache
    ) where {T}
    fill!(r, zero(T))
    return r
end

function maxsteps(::AbstractVector{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::Real, ::CofreeConeCache) where {T}
    return one(T), one(T)
end
