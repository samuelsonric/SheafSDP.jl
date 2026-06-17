#
# Abstract cone type and interface
#

abstract type Cone end

using Base: ReshapedArray
using FixedSizeArrays: FixedSizeArrayDefault

const FArray{T, N} = FixedSizeArrayDefault{T, N}
const FMatrix{T} = FArray{T, 2}
const FVector{T} = FArray{T, 1}
const FScalar{T} = FArray{T, 0}

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
    update_scaling!(cache, cone::Cone, p::AbstractVector, d::AbstractVector)

Compute and cache the NT scaling from primal p and dual d.
"""
function update_scaling! end

"""
    hessian_block!(H::AbstractMatrix, cache, cone::Cone)

Compute the Hessian block W⁻¹ ⊗ₛ W⁻¹ (or its analogue) into H.
"""
function hessian_block! end

"""
    corrector_term!(rc::AbstractVector, cache, cone::Cone,
                    p::AbstractVector, d::AbstractVector,
                    Δp::AbstractVector, Δd::AbstractVector, σμ::Real)

Compute the second-order corrector contribution in original coordinates.
"""
function corrector_term! end

"""
    max_step(cache, cone::Cone, x::AbstractVector, Δx::AbstractVector,
             primal::Bool, γ::Real) -> Real

Compute the maximum step τ ∈ (0,1] such that x + τΔx stays in the cone interior.
"""
function max_step end

"""
    cache_size(cone::Cone, n::Int) -> Int

Return the number of T values needed in the cache for this cone with embdim n.
"""
function cache_size end

# View types for cache structs
const FScalarView{T} = SubArray{T, 0, FVector{T}, Tuple{Int64}, true}
const FVectorView{T} = SubArray{T, 1, FVector{T}, Tuple{UnitRange{Int64}}, true}
const FMatrixView{T} = ReshapedArray{T, 2, FVectorView{T}, Tuple{}}

#
# Unified cache storage
#
struct Caches{T, I}
    val::FVector{T}   # all cache data, flat
    xcol::FVector{I}  # colptr for embdim: xcol[i]:xcol[i+1]-1 gives colrange
    xblk::FVector{I}  # colptr for val: xblk[i]:xblk[i+1]-1 gives cache data for vertex i
end

"""
    cache(caches::Caches, i::Int, cone::Cone)

Return a view-based cache struct for vertex i with the given cone type.
"""
function cache end

include("sdp.jl")
include("pos.jl")
include("soc.jl")
