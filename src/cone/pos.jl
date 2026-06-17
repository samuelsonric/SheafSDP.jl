#
# POS cone (nonnegative orthant ℝⁿ₊)
#

struct POS <: Cone end

# View-based cache for POS
struct POSCache{T}
    w::FVectorView{T}  # NT scaling w = sqrt(p ./ d)
end

# degree = n (dimension equals rank for POS)
degree(::POS, n::Int) = n

# cache size: w(n)
cache_size(::POS, n::Int) = n

# construct view-based cache from Caches
function cache(c::Caches{T}, i::Int, ::POS) where T
    data = view(c.val, c.xblk[i]:c.xblk[i+1]-1)
    return POSCache(data)
end

function identity!(x::AbstractVector{T}, ::POS) where {T}
    fill!(x, one(T))
    return x
end

function update_scaling!(cache::POSCache{T}, ::POS, p::AbstractVector{T}, d::AbstractVector{T}) where {T}
    w = cache.w

    for i in eachindex(w)
        w[i] = sqrt(p[i] / d[i])
    end

    return
end

function hessian_block!(H::AbstractMatrix{T}, cache::POSCache{T}, ::POS) where {T}
    w = cache.w; fill!(H, zero(T))

    for i in eachindex(w)
        H[i, i] = inv(w[i]^2)
    end

    return H
end

function corrector_term!(rc::AbstractVector{T}, cache::POSCache{T}, ::POS,
                         p::AbstractVector{T}, d::AbstractVector{T},
                         Δp::AbstractVector{T}, Δd::AbstractVector{T},
                         σμ::Real) where {T}
    for i in eachindex(rc)
        rc[i] = (σμ - Δp[i] * Δd[i]) / d[i] - p[i]
    end

    return rc
end

function max_step(cache::POSCache{T}, ::POS, x::AbstractVector{T}, Δx::AbstractVector{T}, primal::Bool, γ::Real) where {T}
    #                                                                                                                                                                                                                                                                        
    # compute                                                                                                                                                                                                                                                                
    #                                                                                                                                                                                                                                                                        
    # τ = min -γ xᵢ / Δxᵢ                                                                                                                                                                                                                                                    
    #     Δxᵢ < 0                                                                                                                                                                                                                                                            
    #
    τ = one(T)

    for i in eachindex(x)
        Δxi = Δx[i]

        if Δxi < 0
            τ = min(τ, -γ * x[i] / Δxi)
        end
    end

    return τ
end
