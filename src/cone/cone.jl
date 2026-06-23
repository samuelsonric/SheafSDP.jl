#
# Abstract cone type and interface
#

abstract type Cone end
abstract type AbstractCache{C<:Cone} end

using Base: ReshapedArray
using FixedSizeArrays: FixedSizeArrayDefault

const FArray{T, N} = FixedSizeArrayDefault{T, N}
const FMatrix{T} = FArray{T, 2}
const FVector{T} = FArray{T, 1}
const FScalar{T} = FArray{T, 0}
const Scalar{T} = Array{T, 0}

"""
    degree(cone::Cone, n::Int) -> Int

Return the rank of the cone given embedding dimension n.
- POS: n
- SOC: 2
- SDP: triroot(n)
"""
function degree end

"""
    identity!(x::AbstractVector, cone::Cone)

Set x to the identity element of the cone.
"""
function identity! end

"""
    scale!(H, p, d, cache)

Compute the NT scaling from primal p and dual d, cache it,
and write the Hessian block W⁻¹ ⊗ₛ W⁻¹ (or its analogue) into H.
"""
function scale! end

"""
    corr!(r, p, d, Δp, Δd, σμ, cache)

Compute the H-applied corrector term directly.
"""
function corr! end

"""
    maxsteps(p, Δp, d, Δd, cache) -> (τp, τd)

Compute the maximum primal and dual steps τ ∈ (0,1] such that
p + τp·Δp and d + τd·Δd stay in the cone interior.

The IPM applies a step fraction γ after calling this function.
"""
function maxsteps end

"""
    cachesize(cone::Cone, n::Int) -> Int

Return the number of T values needed in the cache for this cone with embdim n.
"""
function cachesize end

"""
    initcache!(cache)

Initialize a cone's cache to a valid starting state.
Most cones need no initialization; EXP sets xs to the identity point.
"""
function initcache! end

# Default no-op for cones that don't need initialization
initcache!(c::AbstractCache) = c

# View types for cache structs
const FScalarView{T} = SubArray{T, 0, FVector{T}, Tuple{Int64}, true}
const FVectorView{T} = SubArray{T, 1, FVector{T}, Tuple{UnitRange{Int64}}, true}
const FMatrixView{T} = ReshapedArray{T, 2, FVectorView{T}, Tuple{}}

#
# Unified cache storage
#
struct Caches{T, I}
    xcol::FVector{I}  # colptr for embdim: xcol[i]:xcol[i+1]-1 gives colrange
    xblk::FVector{I}  # colptr for val: xblk[i]:xblk[i+1]-1 gives cache data for vertex i
    val::FVector{T}   # all cache data, flat
end

"""
    cache(caches::Caches, i::Int, cone::Cone)

Return a view-based cache struct for vertex i with the given cone type.
"""
function cache end

function Caches(cones::AbstractVector, B::BlockSparseMatrix{T, I}) where {T, I}
    xcol = FVector{I}(undef, nvtxs(B) + one(I))
    xblk = FVector{I}(undef, nvtxs(B) + one(I))

    c = zero(I)
    b = zero(I)

    for v in vtxs(B)
        ncol = ncols(B, v)
        xcol[v] = c + one(I); c += ncol
        xblk[v] = b + one(I); b += cachesize(cones[v], ncol)
    end

    val = FVector{T}(undef, b)

    xcol[nvtxs(B) + one(I)] = c + one(I)
    xblk[nvtxs(B) + one(I)] = b + one(I)

    return Caches(xcol, xblk, val)
end

include("sdp.jl")
include("pos.jl")
include("soc.jl")
include("noc.jl")
include("utils.jl")
include("exp.jl")
include("pow.jl")
