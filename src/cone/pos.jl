"""
    PositiveCone <: Cone

An n-dimensional positive orthant.
"""
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

# construct the ones vector
#
#   e = (1, …, 1)
#
function identity!(x::AbstractVector, ::PositiveCone)
    fill!(x, true)
    return x
end

# construct the diagonal scaling matrix
#
#   H = diag(w)⁻²
#
# where
#
#   wᵢ = √pᵢ / √dᵢ
#
# is the Nesterov-Todd scaling point.
#
function scale!(H::AbstractMatrix, p::AbstractVector, d::AbstractVector, ::PositiveConeCache)
    fill!(H, false)

    for i in eachindex(p)
        H[i, i] = d[i] / p[i]
    end

    return H
end

# Compute the corrector term
#
#   rᵢ = (σμ - Δpᵢ Δdᵢ) / pᵢ - dᵢ.
#
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

# Find the largest number 0 < τ ≤ 1 such that
#
#   x + τ Δx ≥ 0.
#
# This is precisely τ = min {1, κ}, where
#
#   κ = min { -xᵢ / Δxᵢ : Δxᵢ < 0 }.
#
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
