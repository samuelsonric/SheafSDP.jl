#
# NOC cone (no cone / free variables, K = ℝⁿ, K* = {0})
#
# Requires Q_v ≻ 0 to supply all curvature in the (1,1) block.
#

struct NOC <: Cone end

struct NOCCache <: AbstractCache{NOC}
    cone::NOC
end

function NOCCache()
    return NOCCache(NOC())
end

function degree(::NOC, n::Int)
    return 0
end

function cachesize(::NOC, n::Int)
    return 0
end

function cache(::Caches, ::Int, c::NOC)
    return NOCCache(c)
end

function identity!(x::AbstractVector{T}, ::NOC) where {T}
    fill!(x, zero(T))
    return x
end

function scale!(::AbstractVector, ::AbstractVector, ::NOCCache)
    return
end

function hess!(
        H::AbstractMatrix{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        ::NOCCache
    ) where {T}
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
        ::NOCCache
    ) where {T}
    fill!(r, zero(T))
    return r
end

function maxstep(
        ::AbstractVector{T},
        ::AbstractVector{T},
        ::Bool,
        ::Real,
        ::NOCCache
    ) where {T}
    return one(T)
end
