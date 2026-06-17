#
# primal-dual iterate for SDP
#
# primal: min c'p  s.t. Bp = g, P ≻ 0
# dual:   max g'y  s.t. B'y + d = c, D ≻ 0
#
# p, d are svec representations of block-diagonal P, D
#
struct Iterate{T}
    p::Vector{T}    # primal (svec)
    d::Vector{T}    # dual slack (svec)
    y::Vector{T}    # dual multiplier
end

function Iterate{T}(n::Integer, m::Integer) where {T}
    p = zeros(T, n)
    d = zeros(T, n)
    y = zeros(T, m)
    return Iterate{T}(p, d, y)
end

#
# cone degree ν = Σ degree(cone_v, n_v)
#
function conedegree(cones::Vector{<:Cone}, B::BlockSparseMatrix)
    ν = 0
    for (cone, v) in zip(cones, vtxs(B))
        ν += degree(cone, ncols(B, v))
    end
    return ν
end

#
# complementarity measure
#
#   μ = ⟨p, d⟩ / ν
#
function mu(p::AbstractVector, d::AbstractVector, ν::Integer)
    return dot(p, d) / ν
end

#
# residuals
#
#   r_p = g − B p          (primal feasibility)
#   r_d = c − Bᵀy − d      (dual feasibility)
#
function residuals!(rp::AbstractVector, rd::AbstractVector, B, p::AbstractVector, d::AbstractVector, y::AbstractVector, c::AbstractVector, g::AbstractVector)
    # r_p = g - B p
    copyto!(rp, g)
    mul!(rp, B, p, -1, 1)

    # r_d = c - Bᵀy - d
    copyto!(rd, c)
    mul!(rd, B', y, -1, 1)
    axpy!(-1, d, rd)

    return rp, rd
end

#
# allocate unified cache storage
#
function allocate_caches(::Type{T}, ::Type{I}, cones::Vector{<:Cone}, B::BlockSparseMatrix) where {T, I<:Integer}
    nv = nvtxs(B)

    # Build xcol (same structure as B's colptr)
    xcol = FVector{I}(undef, nv + 1)
    xcol[1] = 1
    for (i, v) in enumerate(vtxs(B))
        xcol[i + 1] = xcol[i] + ncols(B, v)
    end

    # Build xblk (colptr into val)
    xblk = FVector{I}(undef, nv + 1)
    xblk[1] = 1
    for (i, (cone, v)) in enumerate(zip(cones, vtxs(B)))
        n_v = ncols(B, v)
        xblk[i + 1] = xblk[i] + cache_size(cone, n_v)
    end

    # Allocate val
    total_size = xblk[nv + 1] - 1
    val = FVector{T}(undef, total_size)

    return Caches(val, xcol, xblk)
end

#
# allocate block-diagonal Hessian H only
#
function allocate_H(::Type{T}, B::BlockSparseMatrix) where {T}
    nv = nvtxs(B)
    H_blocks = [zeros(T, ncols(B, v), ncols(B, v)) for v in vtxs(B)]
    return blocksparse(1:nv, 1:nv, H_blocks, nv, nv)
end

#
# assemble block-diagonal Hessian H using cone interface
#
function hess!(H::BlockSparseMatrix{T}, caches::Caches{T},
               cones::Vector{<:Cone}, p::AbstractVector{T}, d::AbstractVector{T},
               B::BlockSparseMatrix{T}, uplo::Val{UPLO}) where {T, UPLO}
    for (i, (v, cone)) in enumerate(zip(vtxs(B), cones))
        r = colrange(B, v)
        H_v = block(H, v, v, v)
        c = cache(caches, i, cone)

        # Update scaling cache
        update_scaling!(c, cone, view(p, r), view(d, r), uplo)

        # Compute Hessian block
        hessian_block!(H_v, c, cone, uplo)
    end
end

#
# compute Newton step via solve_kkt!
#
# The IPM Newton system is:
#   H Δp − Bᵀ Δy = H r_c − r_d
#        B Δp    = r_p
#
# solve_kkt! solves [A Bᵀ; B 0][x; y] = [f; g]
# so we set: A = H, x = Δp, y = -Δy, f = H r_c - r_d, g = r_p
#
# After solving, recover Δy = -y and Δd = r_d - Bᵀ Δy
#
function newton_step!(
    Δp::AbstractVector{T},
    Δy::AbstractVector{T},
    Δd::AbstractVector{T},
    divwrk::DivisionWorkspace{T},
    itrwrk::IterationWorkspace{T},
    r::AbstractVector{T},
    F::ChordalCholesky{UPLO, T},
    B::BlockSparseMatrix{T},
    H::BlockSparseMatrix{T},
    r_c::AbstractVector{T},
    r_p::AbstractVector{T},
    r_d::AbstractVector{T};
    α::Real=1.0,
    atol::Real=√eps(T),
    rtol::Real=√eps(T),
    itmax::Integer=1000
) where {UPLO, T}
    n = length(Δp)
    m = length(Δy)

    # f = H r_c - r_d
    f = H * r_c - r_d

    # solve [H Bᵀ; B 0][Δp; w] = [f; r_p] where w = -Δy
    # assumes F is already factored
    solve_kkt_factored!(divwrk, itrwrk, Δp, Δy, r, F, B, f, r_p; α, atol, rtol, itmax)

    # recover Δy = -w (solve_kkt! returns w in Δy)
    lmul!(-1, Δy)

    # recover Δd = r_d - Bᵀ Δy
    copyto!(Δd, r_d)
    mul!(Δd, B', Δy, -1, 1)

    return
end

#
# affine RHS: r_c = -p
#
# For the affine (predictor) step with σ = 0:
#   R_c = -P  →  r_c = -p
#   f = H r_c - r_d = H(-p) - r_d = -d - r_d
#
function affine_rhs!(r_c::AbstractVector{T}, p::AbstractVector{T}) where {T}
    copyto!(r_c, p)
    lmul!(-one(T), r_c)
    return r_c
end

#
# corrector RHS using cone interface
#
function corrector_rhs!(r_c::AbstractVector{T}, caches::Caches{T},
                        cones::Vector{<:Cone}, p::AbstractVector{T}, d::AbstractVector{T},
                        Δp::AbstractVector{T}, Δd::AbstractVector{T},
                        σμ::Real, B::BlockSparseMatrix{T}, uplo::Val{UPLO}) where {T, UPLO}
    for (i, (v, cone)) in enumerate(zip(vtxs(B), cones))
        r = colrange(B, v)
        c = cache(caches, i, cone)
        corrector_term!(view(r_c, r), c, cone, view(p, r), view(d, r),
                        view(Δp, r), view(Δd, r), σμ, uplo)
    end
    return r_c
end

#
# step to boundary using cone interface
#
function step_to_boundary(p::AbstractVector{T}, d::AbstractVector{T},
                          Δp::AbstractVector{T}, Δd::AbstractVector{T},
                          caches::Caches{T}, cones::Vector{<:Cone},
                          B::BlockSparseMatrix{T}, uplo::Val{UPLO}; γ::Real=0.99) where {T, UPLO}
    τ_p = one(T)
    τ_d = one(T)

    for (i, (v, cone)) in enumerate(zip(vtxs(B), cones))
        r = colrange(B, v)
        c = cache(caches, i, cone)

        # Step lengths for this block
        τ_p_v = max_step(c, cone, view(p, r), view(Δp, r), true, γ, uplo)
        τ_d_v = max_step(c, cone, view(d, r), view(Δd, r), false, γ, uplo)

        τ_p = min(τ_p, τ_p_v)
        τ_d = min(τ_d, τ_d_v)
    end

    return τ_p, τ_d
end

#
# initialize iterates for infeasible-start primal-dual IPM
#
# Sets P = D = ξ·e (cone identity), y = 0
# ξ is scaled based on problem data: ξ = max(1, ||c||, ||g||)
#
function initialize!(
    p::AbstractVector{T},
    d::AbstractVector{T},
    y::AbstractVector{T},
    c::AbstractVector{T},
    g::AbstractVector{T},
    B::BlockSparseMatrix{T},
    cones::Vector{<:Cone},
    uplo::Val{UPLO};
    ξ::Union{Nothing, Real}=nothing
) where {T, UPLO}
    # Default scaling based on problem data
    if ξ === nothing
        ξ = max(one(T), norm(c), norm(g))
    end

    # y = 0
    fill!(y, zero(T))

    # P = D = ξ·e for each block (cone identity)
    for (v, cone) in zip(vtxs(B), cones)
        r = colrange(B, v)
        identity!(view(p, r), cone, ξ, uplo)
        identity!(view(d, r), cone, ξ, uplo)
    end

    return p, d, y
end

# Legacy: infer SDP cones from B structure
function initialize!(
    p::AbstractVector{T},
    d::AbstractVector{T},
    y::AbstractVector{T},
    c::AbstractVector{T},
    g::AbstractVector{T},
    B::BlockSparseMatrix{T},
    uplo::Val{UPLO};
    ξ::Union{Nothing, Real}=nothing
) where {T, UPLO}
    # Default scaling based on problem data
    if ξ === nothing
        ξ = max(one(T), norm(c), norm(g))
    end

    # y = 0
    fill!(y, zero(T))

    # P = D = ξ I for each block
    for v in vtxs(B)
        r = colrange(B, v)
        n_v = ncols(B, v)
        d_v = triroot(n_v)

        # Create ξ I
        block = ξ * Matrix{T}(I, d_v, d_v)

        # svec into p and d
        svec!(view(p, r), block, uplo)
        svec!(view(d, r), block, uplo)
    end

    return p, d, y
end

#
# Result struct for solver diagnostics
#
struct SolverResult{T}
    p::Vector{T}
    d::Vector{T}
    y::Vector{T}
    converged::Bool
    iterations::Int
    μ_history::Vector{T}
    τ_p_history::Vector{T}
    τ_d_history::Vector{T}
    rp_history::Vector{T}
    rd_history::Vector{T}
    status::Symbol  # :optimal, :max_iter, :stalled, :infeasible, :numerical_failure
end

#
# detect stalling: μ not decreasing sufficiently
#
function is_stalled(μ_history::Vector{T}; window::Int=5, threshold::Real=0.99) where {T}
    if length(μ_history) < window + 1
        return false
    end
    # Check if μ decreased by less than (1 - threshold) over the window
    μ_old = μ_history[end - window]
    μ_new = μ_history[end]
    return μ_new > threshold * μ_old
end

#
# detect numerical failure: τ collapsing while residuals plateau
#
function is_numerical_failure(
    τ_p_history::Vector{T},
    τ_d_history::Vector{T},
    rp_history::Vector{T},
    rd_history::Vector{T};
    window::Int=3,
    τ_threshold::Real=1e-6,
    res_threshold::Real=0.9
) where {T}
    if length(τ_p_history) < window
        return false
    end
    # Check if τ is consistently tiny
    τ_avg = sum(τ_p_history[end-window+1:end]) / window
    τ_avg = min(τ_avg, sum(τ_d_history[end-window+1:end]) / window)
    if τ_avg > τ_threshold
        return false
    end
    # Check if residuals are not decreasing
    if length(rp_history) < window + 1
        return true
    end
    rp_old = rp_history[end - window]
    rp_new = rp_history[end]
    rd_old = rd_history[end - window]
    rd_new = rd_history[end]
    return rp_new > res_threshold * rp_old || rd_new > res_threshold * rd_old
end

#
# robust solve with diagnostics and failure detection
#
# This is the main user-facing solver that includes:
# - Automatic initialization if not provided
# - Failure detection (stalling, numerical issues, possible infeasibility)
# - Detailed diagnostics
#
function solve!(
    p::AbstractVector{T},
    d::AbstractVector{T},
    y::AbstractVector{T},
    c::AbstractVector{T},
    g::AbstractVector{T},
    B::BlockSparseMatrix{T},
    F::ChordalCholesky{UPLO, T},
    L::ChordalTriangular{:N, UPLO, T};
    cones::Vector{<:Cone}=[SDP() for _ in vtxs(B)],
    γ::Real=0.99,
    ε_feas::Real=1e-8,
    ε_μ::Real=1e-8,
    max_iter::Integer=100,
    τ_aug::Real=1.0,
    atol::Real=√eps(T),
    rtol::Real=√eps(T),
    itmax::Integer=1000,
    verbose::Bool=false,
    stall_window::Int=5,
    stall_threshold::Real=0.99,
    τ_collapse_threshold::Real=1e-6
) where {UPLO, T}

    n = length(p)
    m = length(y)
    ν = conedegree(cones, B)

    # Workspaces
    facwrk = FactorizationWorkspace(F)
    divwrk = DivisionWorkspace(F, 1)
    itrwrk = CgWorkspace(m, m, Vector{T})
    r = zeros(T, m)

    r_p = zeros(T, m)
    r_d = zeros(T, n)
    r_c = zeros(T, n)

    # Direction vectors
    Δp_aff = zeros(T, n)
    Δy_aff = zeros(T, m)
    Δd_aff = zeros(T, n)
    Δp = zeros(T, n)
    Δy = zeros(T, m)
    Δd = zeros(T, n)

    H = allocate_H(T, B)
    caches = allocate_caches(T, Int, cones, B)

    # History tracking
    μ_history = T[]
    τ_p_history = T[]
    τ_d_history = T[]
    rp_history = T[]
    rd_history = T[]

    uplo = Val(UPLO)
    status = :max_iter
    norm_B_sq = norm(B)^2

    for iter in 1:max_iter
        # Compute residuals and μ
        residuals!(r_p, r_d, B, p, d, y, c, g)
        μ_curr = mu(p, d, ν)
        push!(μ_history, μ_curr)

        # Track residual norms
        norm_rp = norm(r_p) / (1 + norm(g))
        norm_rd = norm(r_d) / (1 + norm(c))
        push!(rp_history, norm_rp)
        push!(rd_history, norm_rd)

        if verbose
            println("Iter $iter: μ = $μ_curr, ||r_p|| = $norm_rp, ||r_d|| = $norm_rd")
        end

        # Check convergence
        if norm_rp < ε_feas && norm_rd < ε_feas && μ_curr < ε_μ
            status = :optimal
            return SolverResult{T}(
                copy(p), copy(d), copy(y), true, iter,
                μ_history, τ_p_history, τ_d_history, rp_history, rd_history,
                status
            )
        end

        # Check for stalling
        if is_stalled(μ_history; window=stall_window, threshold=stall_threshold)
            status = :stalled
            if verbose
                println("Warning: μ stalling detected")
            end
        end

        # Assemble H (NT scaling + Hessian) via cone-dispatched interface
        hess!(H, caches, cones, p, d, B, uplo)

        # Scale α so that τ_aug=1 is a reasonable default
        α = τ_aug * norm(Symmetric(H, UPLO)) / norm_B_sq

        # Factor F = H + α B'B once per iteration
        factor_kkt!(facwrk, F, L, H; α)

        # ===== Predictor (affine) step =====
        affine_rhs!(r_c, p)
        newton_step!(Δp_aff, Δy_aff, Δd_aff, divwrk, itrwrk, r, F, B, H,
                     r_c, r_p, r_d; α, atol, rtol, itmax)

        # Step to boundary for affine direction
        τ_p_aff, τ_d_aff = step_to_boundary(p, d, Δp_aff, Δd_aff, caches, cones, B, uplo; γ=one(T))

        # Compute μ_aff
        p_aff = p + τ_p_aff * Δp_aff
        d_aff = d + τ_d_aff * Δd_aff
        μ_aff = mu(p_aff, d_aff, ν)

        # Adaptive centering parameter
        σ = clamp((μ_aff / μ_curr)^3, zero(T), one(T))

        # ===== Corrector step (reuses same factorization) =====
        corrector_rhs!(r_c, caches, cones, p, d, Δp_aff, Δd_aff, σ * μ_curr, B, uplo)
        newton_step!(Δp, Δy, Δd, divwrk, itrwrk, r, F, B, H,
                     r_c, r_p, r_d; α, atol, rtol, itmax)

        # Step to boundary
        τ_p, τ_d = step_to_boundary(p, d, Δp, Δd, caches, cones, B, uplo; γ)
        push!(τ_p_history, τ_p)
        push!(τ_d_history, τ_d)

        # Check for numerical failure
        if is_numerical_failure(τ_p_history, τ_d_history, rp_history, rd_history;
                                 τ_threshold=τ_collapse_threshold)
            status = :numerical_failure
            if verbose
                println("Warning: numerical failure detected (τ collapse + residual plateau)")
            end
        end

        # Update iterates
        axpy!(τ_p, Δp, p)
        axpy!(τ_d, Δd, d)
        axpy!(τ_d, Δy, y)
    end

    return SolverResult{T}(
        copy(p), copy(d), copy(y), false, max_iter,
        μ_history, τ_p_history, τ_d_history, rp_history, rd_history,
        status
    )
end
