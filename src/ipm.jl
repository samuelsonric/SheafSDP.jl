#
# primal-dual interior point method
#
# primal: min c'p  s.t. Bp = g, P ≻ 0
# dual:   max g'y  s.t. B'y + d = c, D ≻ 0
#
# p, d are svec representations of block-diagonal P, D
#

#
# solver parameters
#
@kwdef struct IPMSettings{T}
    step_frac::T = 0.99                     # step size damping
    feas_tol::T = 1e-8                # feasibility tolerance
    gap_tol::T = 1e-8                   # duality gap tolerance
    itmax::Int = 100             # max IPM iterations
    verbose::Bool = false           # verbosity flag
    stall_window::Int = 5           # stall detection window
    stall_threshold::T = 0.99       # stall detection threshold
    τ_collapse_threshold::T = 1e-6  # step collapse threshold
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
function residuals!(rp::AbstractVector, rd::AbstractVector, B, p::AbstractVector, d::AbstractVector, y::AbstractVector, c::AbstractVector, g::AbstractVector, Q)
    # r_p = g - B p
    copyto!(rp, g)
    mul!(rp, B, p, -1, 1)

    # r_d = c + Qp - Bᵀy - d
    copyto!(rd, c)
    mul!(rd, Symmetric(Q, :L), p, 1, 1)
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
        xblk[i + 1] = xblk[i] + cachesize(cone, n_v)
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
               B::BlockSparseMatrix{T}, Q) where {T}
    for (i, (v, cone)) in enumerate(zip(vtxs(B), cones))
        r = colrange(B, v)
        H_v = block(H, v, v, v)
        c = cache(caches, i, cone)

        p_v = view(p, r)
        d_v = view(d, r)

        # Update scaling cache
        scale!(p_v, d_v, c)

        # Compute Hessian block
        hess!(H_v, p_v, d_v, c)

        # Fold Q_v into H_v
        axpy!(true, block(Q, v, v, v), H_v)
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
# so we set: A = H, x = Δp, y = -Δy, f = (provided), g = r_p
#
# After solving, recover Δy = -y and Δd = r_d - Bᵀ Δy + Q Δp
#
function newton_step!(
    Δp::AbstractVector{T},
    Δy::AbstractVector{T},
    Δd::AbstractVector{T},
    kktwrk::KKTWorkspace{T},
    kktset::KKTSettings{T},
    H::BlockSparseMatrix{T},
    B::BlockSparseMatrix{T},
    f::AbstractVector{T},
    r_p::AbstractVector{T},
    r_d::AbstractVector{T},
    Q
) where {T}
    # solve [H Bᵀ; B 0][Δp; w] = [f; r_p] where w = -Δy
    # assumes F is already factored
    solve_kkt!(kktwrk, kktset, Δp, Δy, H, B, f, r_p)

    # recover Δy = -w (solve_kkt! returns w in Δy)
    lmul!(-1, Δy)

    # recover Δd = r_d - Bᵀ Δy + Q Δp
    copyto!(Δd, r_d)
    mul!(Δd, B', Δy, -1, 1)
    mul!(Δd, Symmetric(Q, :L), Δp, 1, 1)

    return
end

#
# affine RHS: r_c = -p
#
#
# corrector RHS using cone interface
#
function corrector_rhs!(r_c::AbstractVector{T}, caches::Caches{T},
                        cones::Vector{<:Cone}, p::AbstractVector{T}, d::AbstractVector{T},
                        Δp::AbstractVector{T}, Δd::AbstractVector{T},
                        σμ::Real, B::BlockSparseMatrix{T}) where {T}
    for (i, (v, cone)) in enumerate(zip(vtxs(B), cones))
        r = colrange(B, v)
        c = cache(caches, i, cone)
        corr!(view(r_c, r), view(p, r), view(d, r),
              view(Δp, r), view(Δd, r), σμ, c)
    end
    return r_c
end

#
# step to boundary using cone interface
#
function step_to_boundary(p::AbstractVector{T}, d::AbstractVector{T},
                          Δp::AbstractVector{T}, Δd::AbstractVector{T},
                          caches::Caches{T}, cones::Vector{<:Cone},
                          B::BlockSparseMatrix{T}; step_frac::Real=0.99) where {T}
    τ_p = one(T)
    τ_d = one(T)

    for (i, (v, cone)) in enumerate(zip(vtxs(B), cones))
        r = colrange(B, v)
        c = cache(caches, i, cone)

        # Step lengths for this block
        τ_p_v = maxstep(view(p, r), view(Δp, r), true, step_frac, c)
        τ_d_v = maxstep(view(d, r), view(Δd, r), false, step_frac, c)

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
    cones::Vector{<:Cone};
    ξ::Union{Nothing, Real}=nothing
) where {T}
    # Default scaling based on problem data
    if ξ === nothing
        ξ = max(one(T), norm(c), norm(g))
    end

    # y = 0
    fill!(y, zero(T))

    # P = D = ξ·e for each block (cone identity)
    for (v, cone) in zip(vtxs(B), cones)
        r = colrange(B, v)
        identity!(view(p, r), cone)
        identity!(view(d, r), cone)
    end
    rmul!(p, ξ)
    rmul!(d, ξ)

    return p, d, y
end

# Legacy: infer SDP cones from B structure
function initialize!(
    p::AbstractVector{T},
    d::AbstractVector{T},
    y::AbstractVector{T},
    c::AbstractVector{T},
    g::AbstractVector{T},
    B::BlockSparseMatrix{T};
    ξ::Union{Nothing, Real}=nothing
) where {T}
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
        svec!(view(p, r), block)
        svec!(view(d, r), block)
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
    status::Symbol  # :optimal, :itmax, :stalled, :infeasible, :numerical_failure
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
    F::ChordalTriangular{:N, UPLO, T},
    L::ChordalTriangular{:N, UPLO, T};
    Q::BlockSparseMatrix{T},
    cones::Vector{<:Cone}=[SDP() for _ in vtxs(B)],
    step_frac::Real=0.99,
    feas_tol::Real=1e-8,
    gap_tol::Real=1e-8,
    itmax::Integer=100,
    kkt::KKTSettings{T}=UzawaSettings{T}(),
    verbose::Bool=false,
    stall_window::Int=5,
    stall_threshold::Real=0.99,
    τ_collapse_threshold::Real=1e-6
) where {UPLO, T}

    n = length(p)
    m = length(y)
    ν = conedegree(cones, B)

    ipmset = IPMSettings{T}(; step_frac, feas_tol, gap_tol, itmax,
                           verbose, stall_window, stall_threshold, τ_collapse_threshold)
    kktwrk = UzawaWorkspace(F, L, B)

    r_p = zeros(T, m)
    r_d = zeros(T, n)
    r_c = zeros(T, n)
    f = zeros(T, n)

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

    status = :itmax

    for iter in 1:ipmset.itmax
        # Compute residuals and μ
        residuals!(r_p, r_d, B, p, d, y, c, g, Q)
        μ_curr = ν > 0 ? mu(p, d, ν) : zero(T)
        push!(μ_history, μ_curr)

        # Track residual norms
        norm_rp = norm(r_p) / (1 + norm(g))
        norm_rd = norm(r_d) / (1 + norm(c))
        push!(rp_history, norm_rp)
        push!(rd_history, norm_rd)

        if ipmset.verbose
            println("Iter $iter: μ = $μ_curr, ||r_p|| = $norm_rp, ||r_d|| = $norm_rd")
        end

        # Check convergence
        gap_ok = ν == 0 || μ_curr < ipmset.gap_tol
        if norm_rp < ipmset.feas_tol && norm_rd < ipmset.feas_tol && gap_ok
            status = :optimal
            return SolverResult{T}(
                copy(p), copy(d), copy(y), true, iter,
                μ_history, τ_p_history, τ_d_history, rp_history, rd_history,
                status
            )
        end

        # Check for stalling
        if is_stalled(μ_history; window=ipmset.stall_window, threshold=ipmset.stall_threshold)
            status = :stalled
            if ipmset.verbose
                println("Warning: μ stalling detected")
            end
        end

        # Assemble H (NT scaling + Hessian) via cone-dispatched interface
        hess!(H, caches, cones, p, d, B, Q)

        # Factor F = H + α B'B once per iteration
        init_kkt!(kktwrk, kkt, H)

        # ===== Predictor (affine) step =====
        # f = H·(-p) - r_d = -d - r_d (by NT property: H·p = d)
        @. f = -(d + r_d)
        newton_step!(Δp_aff, Δy_aff, Δd_aff, kktwrk, kkt, H, B, f, r_p, r_d, Q)

        # Step to boundary for affine direction
        τ_p_aff, τ_d_aff = step_to_boundary(p, d, Δp_aff, Δd_aff, caches, cones, B; step_frac=one(T))

        # Compute μ_aff
        p_aff = p + τ_p_aff * Δp_aff
        d_aff = d + τ_d_aff * Δd_aff
        μ_aff = mu(p_aff, d_aff, ν)

        # Adaptive centering parameter
        σ = clamp((μ_aff / μ_curr)^3, zero(T), one(T))

        # ===== Corrector step (reuses same factorization) =====
        # corrector_rhs! now returns H·r_c directly per block
        corrector_rhs!(f, caches, cones, p, d, Δp_aff, Δd_aff, σ * μ_curr, B)
        axpy!(-1, r_d, f)
        newton_step!(Δp, Δy, Δd, kktwrk, kkt, H, B, f, r_p, r_d, Q)

        # Step to boundary
        τ_p, τ_d = step_to_boundary(p, d, Δp, Δd, caches, cones, B; step_frac=ipmset.step_frac)
        push!(τ_p_history, τ_p)
        push!(τ_d_history, τ_d)

        # Check for numerical failure
        if is_numerical_failure(τ_p_history, τ_d_history, rp_history, rd_history;
                                 τ_threshold=ipmset.τ_collapse_threshold)
            status = :numerical_failure
            if ipmset.verbose
                println("Warning: numerical failure detected (τ collapse + residual plateau)")
            end
        end

        # Update iterates
        axpy!(τ_p, Δp, p)
        axpy!(τ_d, Δd, d)
        axpy!(τ_d, Δy, y)
    end

    return SolverResult{T}(
        copy(p), copy(d), copy(y), false, ipmset.itmax,
        μ_history, τ_p_history, τ_d_history, rp_history, rd_history,
        status
    )
end
