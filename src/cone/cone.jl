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
    scale!(p, d, cache)

Compute and cache the NT scaling from primal p and dual d.
"""
function scale! end

"""
    hess!(H, p, d, cache)

Compute the Hessian block W⁻¹ ⊗ₛ W⁻¹ (or its analogue) into H.
SDP/SOC use the cache; POS computes directly from p, d.
"""
function hess! end

"""
    corr!(r, p, d, Δp, Δd, σμ, cache)

Compute the H-applied corrector term directly.
"""
function corr! end

"""
    maxstep(x, Δx, primal, γ, cache) -> Real

Compute the maximum step τ ∈ (0,1] such that x + τΔx stays in the cone interior.
"""
function maxstep end

"""
    cachesize(cone::Cone, n::Int) -> Int

Return the number of T values needed in the cache for this cone with embdim n.
"""
function cachesize end

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
include("exp.jl")
