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
    POSCache(data)
end

function identity!(x::AbstractVector{T}, ::POS, ξ::Real, uplo::Val) where {T}
    fill!(x, T(ξ))
    return x
end

function update_scaling!(cache::POSCache{T}, ::POS,
                         p::AbstractVector{T}, d::AbstractVector{T}, uplo::Val) where {T}
    # w = sqrt(p ./ d)
    cache.w .= sqrt.(p ./ d)
    return
end

function hessian_block!(H::AbstractMatrix{T}, cache::POSCache{T}, ::POS,
                        uplo::Val) where {T}
    # H = Diagonal(1 ./ w.^2)
    n = length(cache.w)
    fill!(H, zero(T))
    w = cache.w
    for i in 1:n
        H[i, i] = one(T) / (w[i]^2)
    end
    return H
end

function corrector_term!(rc::AbstractVector{T}, cache::POSCache{T}, ::POS,
                         p::AbstractVector{T}, d::AbstractVector{T},
                         Δp::AbstractVector{T}, Δd::AbstractVector{T},
                         σμ::Real, uplo::Val) where {T}
    # rc = σμ ./ d .- p .- (Δp .* Δd) ./ d
    rc .= σμ ./ d .- p .- (Δp .* Δd) ./ d
    return rc
end

function max_step(cache::POSCache{T}, ::POS,
                  x::AbstractVector{T}, Δx::AbstractVector{T},
                  primal::Bool, γ::Real, uplo::Val) where {T}
    # τ = min(1, γ · min_i(-x_i/Δx_i)) over Δx_i < 0
    τ = one(T)
    for i in eachindex(x)
        if Δx[i] < 0
            τ = min(τ, -γ * x[i] / Δx[i])
        end
    end
    return τ
end
