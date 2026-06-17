#
# SDP cone (PSD cone 𝕊ᵈ₊)
#

struct SDP <: Cone end

# View-based cache for SDP
struct SDPCache{T}
    LP::FMatrixView{T}    # lower triangular Cholesky factor of P (d×d)
    LD::FMatrixView{T}    # lower triangular Cholesky factor of D (d×d)
    U::FMatrixView{T}     # orthogonal matrix from SVD (d×d)
    sv::FVectorView{T}    # singular values (d)
    work::FMatrixView{T}  # workspace (d×d)
end

# degree = triroot(n) where n = trinum(d)
degree(::SDP, n::Int) = triroot(n)

# cache size: LP(d²) + LD(d²) + U(d²) + sv(d) + work(d²) = 4d² + d
function cache_size(::SDP, n::Int)
    d = triroot(n)
    return 4 * d^2 + d
end

# construct view-based cache from Caches
function cache(c::Caches{T}, i::Int, ::SDP) where T
    n = c.xcol[i+1] - c.xcol[i]
    d = triroot(n)
    data = view(c.val, c.xblk[i]:c.xblk[i+1]-1)

    # Layout: LP(d²), LD(d²), U(d²), sv(d), work(d²)
    d2 = d^2
    LP   = reshape(view(data, 1:d2), d, d)
    LD   = reshape(view(data, d2+1:2d2), d, d)
    U    = reshape(view(data, 2d2+1:3d2), d, d)
    sv   = view(data, 3d2+1:3d2+d)
    work = reshape(view(data, 3d2+d+1:4d2+d), d, d)

    SDPCache(LP, LD, U, sv, work)
end

function identity!(x::AbstractVector{T}, ::SDP, ξ::Real, uplo::Val{UPLO}) where {T, UPLO}
    d = triroot(length(x))
    Id = ξ * Matrix{T}(I, d, d)
    svec!(x, Id, uplo)
    return x
end

function update_scaling!(cache::SDPCache{T}, ::SDP,
                         p::AbstractVector, d::AbstractVector, uplo::Val{UPLO}) where {T, UPLO}
    d_v = size(cache.LP, 1)

    # Build P and D from svec
    P = zeros(T, d_v, d_v)
    D = zeros(T, d_v, d_v)
    smat!(P, p, uplo)
    symmetrize!(P, uplo)
    smat!(D, d, uplo)
    symmetrize!(D, uplo)

    # Compute scaling factors via meanblock!
    meanblock!(cache.LP, cache.LD, cache.U, cache.sv, P, D)
    return
end

function hessian_block!(H::AbstractMatrix, cache::SDPCache, ::SDP,
                        uplo::Val{UPLO}) where {UPLO}
    hessblock!(H, cache.LP, cache.U, cache.sv, cache.work, uplo)
    return H
end

function corrector_term!(rc::AbstractVector, cache::SDPCache{T}, ::SDP,
                         p::AbstractVector, d::AbstractVector,
                         Δp::AbstractVector, Δd::AbstractVector,
                         σμ::Real, uplo::Val{UPLO}) where {T, UPLO}
    d_v = size(cache.LP, 1)

    # Build matrices from svec
    P_v = zeros(T, d_v, d_v)
    ΔP_v = zeros(T, d_v, d_v)
    ΔD_v = zeros(T, d_v, d_v)

    smat!(P_v, p, uplo)
    symmetrize!(P_v, uplo)
    smat!(ΔP_v, Δp, uplo)
    symmetrize!(ΔP_v, uplo)
    smat!(ΔD_v, Δd, uplo)
    symmetrize!(ΔD_v, uplo)

    L_P = LowerTriangular(cache.LP)
    L_D = LowerTriangular(cache.LD)
    U_v = cache.U
    s_v = cache.sv

    # D⁻¹ = L_D⁻ᵀ L_D⁻¹
    D_inv = L_D' \ inv(L_D)

    # Inverse Lyapunov solve
    X = L_P \ (ΔP_v * ΔD_v * L_P)
    Y = U_v' * X * U_v
    B_mat = (Y ./ s_v' + Y' ./ s_v) ./ (s_v .+ s_v')
    C = U_v * B_mat * U_v'
    cross_sym = L_P * C * L_P'

    # R_c = σμ D⁻¹ - P - cross_sym
    R_c = σμ * D_inv - P_v - cross_sym

    svec!(rc, R_c, uplo)
    return rc
end

function max_step(cache::SDPCache{T}, ::SDP,
                  x::AbstractVector{T}, Δx::AbstractVector{T},
                  primal::Bool, γ::Real, uplo::Val{UPLO}) where {T, UPLO}
    d_v = size(cache.LP, 1)

    # Use precomputed Cholesky factor from cache
    L = LowerTriangular(primal ? cache.LP : cache.LD)

    # Build ΔX from svec
    ΔX = zeros(T, d_v, d_v)
    smat!(ΔX, Δx, uplo)
    symmetrize!(ΔX, uplo)

    return step_length_block(L, ΔX, γ)
end
