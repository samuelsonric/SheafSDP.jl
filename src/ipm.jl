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
    s::Vector{T}    # NT scaling (svec)
end

function Iterate{T}(n::Integer, m::Integer) where {T}
    p = zeros(T, n)
    d = zeros(T, n)
    y = zeros(T, m)
    s = zeros(T, n)
    return Iterate{T}(p, d, y, s)
end

#
# symmetrize a matrix by copying off-diagonals
#
#
# triangular number and its inverse
#
trinum(n::Integer) = n * (n + 1) ÷ 2
triroot(n::Integer) = (isqrt(1 + 8n) - 1) ÷ 2
roottwo(::Type{T}) where {T} = sqrt(T(2))

#
# svec index for (i,j) matrix entry in n×n matrix
#
# For :U (upper triangle, column-major): (i,j) with i ≤ j
#
# cone degree ν = Σ d_v (sum of matrix dimensions, not svec dimensions)
#
function conedegree(B::BlockSparseMatrix)
    ν = 0
    for v in vtxs(B)
        ν += triroot(ncols(B, v))
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

function symmetrize!(M::AbstractMatrix, uplo::Val{UPLO}) where {UPLO}
    for j in axes(M, 1)
        for i in 1:j - 1
            if UPLO === :L
                M[i, j] = M[j, i]
            else
                M[j, i] = M[i, j]
            end
        end
    end

    return M
end

#
# compute NT scaling point W for a single block via Cholesky + SVD
#
# L_P = chol(P),  L_D = chol(D)
# G = L_Pᵀ L_D
# SVD: G = U Σ Vᵀ
# W = L_P U Σ⁻¹ Uᵀ L_Pᵀ
#
# satisfies W D W = P
#
# Returns W in matrix form (stored in WP workspace)
#
function meanblock!(WP::AbstractMatrix{T}, WD::AbstractMatrix{T}, P::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n = size(P, 1)

    # L_P = chol(P)
    copyto!(WP, P)
    cholesky!(Symmetric(WP, :L))
    L_P = LowerTriangular(WP)

    # L_D = chol(D)
    copyto!(WD, D)
    cholesky!(Symmetric(WD, :L))
    L_D = LowerTriangular(WD)

    # G = L_Pᵀ L_D
    G = L_P' * L_D

    # SVD: G = U Σ Vᵀ
    F = svd(G)

    # W = L_P U Σ⁻¹ Uᵀ L_Pᵀ
    # Compute R = L_P U Σ^{-1/2}, then W = R Rᵀ
    R = L_P * F.U * Diagonal(1 ./ sqrt.(F.S))
    mul!(WP, R, R')

    return WP
end

#
# symmetric Kronecker product: H = B ⊗ₛ B
#
# svec(B X B') = (B ⊗ₛ B) svec(X)
#
# Entries:
#   H[diag_i, diag_k]         = B[i,k]²
#   H[offdiag_ij, diag_k]     = √2 · B[i,k] · B[j,k]
#   H[diag_i, offdiag_kl]     = √2 · B[i,k] · B[i,l]
#   H[offdiag_ij, offdiag_kl] = B[i,k]·B[j,l] + B[i,l]·B[j,k]
#
function skron!(H::AbstractMatrix{T}, A::AbstractMatrix{T}, uplo::Val{UPLO}) where {T, UPLO}
    n = size(A, 1)
    α = roottwo(T)
    tll = 1

    @inbounds for l in 1:n
        tij = 0

        for j in 1:n
            Ajl = A[j, l]

            if UPLO === :L
                tij += 1; H[tij, tll] = Ajl^2
            end

            if UPLO === :L
                r = j + 1:n
            else
                r = 1:j - 1
            end

            for i in r
                tij += 1; H[tij, tll] = α * A[i, l] * Ajl
            end

            if UPLO === :U
                tij += 1; H[tij, tll] = Ajl^2
            end
        end

        if UPLO === :L
            tkl = tll
        else
            tkl = tll - l
        end

        if UPLO === :L
            s = l + 1:n
        else
            s = 1:l - 1
        end

        for k in s
            tkl += 1; tij = 0

            for j in 1:n
                Ajk = A[j, k]
                Ajl = A[j, l]

                if UPLO === :L
                    tij += 1; H[tij, tkl] = α * Ajk * Ajl
                end

                if UPLO === :L
                    r = j + 1:n
                else
                    r = 1:j - 1
                end

                for i in r
                    tij += 1; H[tij, tkl] = A[i, k] * Ajl + A[i, l] * Ajk
                end

                if UPLO === :U
                    tij += 1; H[tij, tkl] = α * Ajk * Ajl
                end
            end
        end

        if UPLO === :L
            tll += n - l + 1
        else
            tll += l + 1
        end
    end

    return H
end

#
# compute H_v = W⁻¹ ⊗ₛ W⁻¹
#
function hessblock!(H::AbstractMatrix{T}, W::AbstractMatrix{T}, work::AbstractMatrix{T}, uplo::Val{UPLO}) where {T, UPLO}
    copyto!(work, W)
    C = cholesky!(Symmetric(work, :L))
    Winv = inv(C)
    return skron!(H, Winv, uplo)
end

function svec!(v::AbstractVector{T}, M::AbstractMatrix{T}, uplo::Val{UPLO}) where {UPLO, T}
    n = size(M, 1); k = 0

    α = roottwo(T)

    for j in 1:n
        if UPLO === :L
            k += 1; v[k] = M[j, j]
        end

        if UPLO === :L
            r = j + 1:n
        else
            r = 1:j - 1
        end

        for i in r
            k += 1; v[k] = α * M[i, j]
        end

        if UPLO === :U
            k += 1; v[k] = M[j, j]
        end
    end

    return v
end

function smat!(M::AbstractMatrix{T}, v::AbstractVector{T}, uplo::Val{UPLO}) where {UPLO, T}
    n = size(M, 1); k = 0

    α = roottwo(T)

    for j in 1:n
        if UPLO === :L
            k += 1; M[j, j] = v[k]
        end

        if UPLO === :L
            r = j + 1:n
        else
            r = 1:j - 1
        end

        for i in r
            k += 1; M[i, j] = v[k] / α
        end

        if UPLO === :U
            k += 1; M[j, j] = v[k]
        end
    end

    return v
end

#
# assemble block-diagonal Hessian H from primal/dual iterates
#
# H_v = W_v⁻¹ ⊗ₛ W_v⁻¹ where W_v is NT scaling point for (P_v, D_v)
#
# Returns a BlockSparseMatrix with diagonal blocks H_v
#
function hess!(H_blocks::Vector{Matrix{T}}, W_blocks::Vector{Matrix{T}}, p::AbstractVector{T}, d::AbstractVector{T}, B::BlockSparseMatrix{T}, uplo::Val{UPLO}) where {T, UPLO}
    nv = nvtxs(B)

    for v in vtxs(B)
        r = colrange(B, v)
        n_v = ncols(B, v)
        d_v = triroot(n_v)

        # Get or resize workspace matrices
        if length(H_blocks) < v
            push!(H_blocks, zeros(T, n_v, n_v))
        elseif size(H_blocks[v]) != (n_v, n_v)
            H_blocks[v] = zeros(T, n_v, n_v)
        end

        if length(W_blocks) < v
            push!(W_blocks, zeros(T, d_v, d_v))
        elseif size(W_blocks[v]) != (d_v, d_v)
            W_blocks[v] = zeros(T, d_v, d_v)
        end

        H_v = H_blocks[v]
        W_v = W_blocks[v]

        # Build P_v and D_v from svec
        P_v = zeros(T, d_v, d_v)
        D_v = zeros(T, d_v, d_v)
        smat!(P_v, view(p, r), uplo)
        symmetrize!(P_v, uplo)
        smat!(D_v, view(d, r), uplo)
        symmetrize!(D_v, uplo)

        # Compute W_v via meanblock!
        WD = zeros(T, d_v, d_v)  # workspace
        meanblock!(W_v, WD, P_v, D_v)

        # Compute H_v via hessblock!
        work = zeros(T, d_v, d_v)
        hessblock!(H_v, W_v, work, uplo)
    end

    # Build BlockSparseMatrix from diagonal blocks
    return blocksparse(1:nv, 1:nv, H_blocks, nv, nv)
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
    facwrk::FactorizationWorkspace{T, I},
    divwrk::DivisionWorkspace{T, I},
    itrwrk::IterationWorkspace{T},
    r::AbstractVector{T},
    F::ChordalCholesky{UPLO, T},
    L::L_t,
    B::BlockSparseMatrix{T},
    B_sp,  # sparse(B) for matvec
    H::BlockSparseMatrix{T},
    H_sp,  # sparse(H) for matvec
    r_c::AbstractVector{T},
    r_p::AbstractVector{T},
    r_d::AbstractVector{T};
    τ::Real=1.0,
    atol::Real=√eps(T),
    rtol::Real=√eps(T),
    itmax::Integer=1000
) where {UPLO, T, I, L_t <: ChordalTriangular}
    n = length(Δp)
    m = length(Δy)

    # f = H r_c - r_d
    f = H_sp * r_c - r_d

    # solve [H Bᵀ; B 0][Δp; w] = [f; r_p] where w = -Δy
    solve_kkt!(facwrk, divwrk, itrwrk, Δp, Δy, r, F, L, B, H, f, r_p; α=τ, atol, rtol, itmax)

    # recover Δy = -w (solve_kkt! returns w in Δy)
    lmul!(-1, Δy)

    # recover Δd = r_d - Bᵀ Δy
    copyto!(Δd, r_d)
    mul!(Δd, B_sp', Δy, -1, 1)

    return nothing
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
# corrector RHS: r_c = svec(σμ D⁻¹ - P - sym(ΔP^a ΔD^a W)) per block
#
# Full Mehrotra corrector with 2nd-order term:
#   R_c,v = σμ D_v⁻¹ - P_v - sym(ΔP_v^a ΔD_v^a W_v)
#
# where sym(X) = (X + Xᵀ)/2
#
function corrector_rhs!(r_c::AbstractVector{T}, p::AbstractVector{T}, d::AbstractVector{T}, Δp::AbstractVector{T}, Δd::AbstractVector{T}, W_blocks::Vector{Matrix{T}}, σμ::Real, B::BlockSparseMatrix{T}, uplo::Val{UPLO}) where {T, UPLO}
    for v in vtxs(B)
        r = colrange(B, v)
        n_v = ncols(B, v)
        d_v = triroot(n_v)

        # Build matrices from svec
        P_v = zeros(T, d_v, d_v)
        D_v = zeros(T, d_v, d_v)
        ΔP_v = zeros(T, d_v, d_v)
        ΔD_v = zeros(T, d_v, d_v)

        smat!(P_v, view(p, r), uplo)
        symmetrize!(P_v, uplo)
        smat!(D_v, view(d, r), uplo)
        symmetrize!(D_v, uplo)
        smat!(ΔP_v, view(Δp, r), uplo)
        symmetrize!(ΔP_v, uplo)
        smat!(ΔD_v, view(Δd, r), uplo)
        symmetrize!(ΔD_v, uplo)

        W_v = W_blocks[v]

        # Compute D_v⁻¹ via Cholesky
        D_inv = inv(cholesky(Symmetric(D_v)))

        # 2nd-order term: sym(ΔP_v ΔD_v W_v)
        cross = ΔP_v * ΔD_v * W_v
        cross_sym = (cross + cross') / 2

        # R_c,v = σμ D_v⁻¹ - P_v - cross_sym
        R_c_v = σμ * D_inv - P_v - cross_sym

        # svec into r_c
        svec!(view(r_c, r), R_c_v, uplo)
    end

    return r_c
end

#
# step length to boundary for a single block
#
# Computes largest τ such that X + τ ΔX ⪰ 0
# where X is SPD and ΔX is symmetric
#
# M = L⁻¹ ΔX L⁻ᵀ where X = L Lᵀ
# λ_min = minimum eigenvalue of M
# τ_max = (λ_min < 0) ? -γ/λ_min : 1.0
#
function step_length_block(X::AbstractMatrix{T}, ΔX::AbstractMatrix{T}, γ::Real) where {T}
    # Factor X = L Lᵀ
    C = cholesky(Symmetric(X))
    L = C.L

    # M = L⁻¹ ΔX L⁻ᵀ
    M = L \ ΔX / L'

    # Symmetrize M (for numerical stability)
    M = (M + M') / 2

    # Minimum eigenvalue
    λ_min = minimum(eigvals(Symmetric(M)))

    # Step length
    if λ_min < 0
        return min(one(T), -γ / λ_min)
    else
        return one(T)
    end
end

#
# step to boundary for full problem
#
# Computes τ_p, τ_d such that:
#   P + τ_p ΔP ⪰ 0 (all blocks)
#   D + τ_d ΔD ⪰ 0 (all blocks)
#
# γ ∈ (0, 1) is the fraction of the way to the boundary
# (e.g., γ = 0.99 stays 1% away from boundary)
#
function step_to_boundary(p::AbstractVector{T}, d::AbstractVector{T}, Δp::AbstractVector{T}, Δd::AbstractVector{T}, B::BlockSparseMatrix{T}, uplo::Val{UPLO}; γ::Real=0.99) where {T, UPLO}
    τ_p = one(T)
    τ_d = one(T)

    for v in vtxs(B)
        r = colrange(B, v)
        n_v = ncols(B, v)
        d_v = triroot(n_v)

        # Build matrices from svec
        P_v = zeros(T, d_v, d_v)
        D_v = zeros(T, d_v, d_v)
        ΔP_v = zeros(T, d_v, d_v)
        ΔD_v = zeros(T, d_v, d_v)

        smat!(P_v, view(p, r), uplo)
        symmetrize!(P_v, uplo)
        smat!(D_v, view(d, r), uplo)
        symmetrize!(D_v, uplo)
        smat!(ΔP_v, view(Δp, r), uplo)
        symmetrize!(ΔP_v, uplo)
        smat!(ΔD_v, view(Δd, r), uplo)
        symmetrize!(ΔD_v, uplo)

        # Step lengths for this block
        τ_p_v = step_length_block(P_v, ΔP_v, γ)
        τ_d_v = step_length_block(D_v, ΔD_v, γ)

        τ_p = min(τ_p, τ_p_v)
        τ_d = min(τ_d, τ_d_v)
    end

    return τ_p, τ_d
end

#
# initialize iterates for infeasible-start primal-dual IPM
#
# Sets P = D = ξ·I block-diagonal, y = 0
# ξ is scaled based on problem data: ξ = max(1, ||c||, ||g||)
#
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
    B_sp,
    F::ChordalCholesky{UPLO, T},
    L::L_t;
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
) where {UPLO, T, I, L_t <: ChordalTriangular}

    n = length(p)
    m = length(y)
    ν = conedegree(B)

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

    H_blocks = Matrix{T}[]
    W_blocks = Matrix{T}[]

    # History tracking
    μ_history = T[]
    τ_p_history = T[]
    τ_d_history = T[]
    rp_history = T[]
    rd_history = T[]

    uplo = Val(UPLO)
    status = :max_iter

    for iter in 1:max_iter
        # Compute residuals and μ
        residuals!(r_p, r_d, B_sp, p, d, y, c, g)
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

        # Assemble H (NT scaling + Hessian)
        H = hess!(H_blocks, W_blocks, p, d, B, uplo)
        H_sp = sparse(H)

        # ===== Predictor (affine) step =====
        affine_rhs!(r_c, p)
        newton_step!(Δp_aff, Δy_aff, Δd_aff, facwrk, divwrk, itrwrk, r, F, L, B, B_sp, H, H_sp,
                     r_c, r_p, r_d; τ=τ_aug, atol, rtol, itmax)

        # Step to boundary for affine direction
        τ_p_aff, τ_d_aff = step_to_boundary(p, d, Δp_aff, Δd_aff, B, uplo; γ=one(T))

        # Compute μ_aff
        p_aff = p + τ_p_aff * Δp_aff
        d_aff = d + τ_d_aff * Δd_aff
        μ_aff = mu(p_aff, d_aff, ν)

        # Adaptive centering parameter
        σ = clamp((μ_aff / μ_curr)^3, zero(T), one(T))

        # ===== Corrector step =====
        corrector_rhs!(r_c, p, d, Δp_aff, Δd_aff, W_blocks, σ * μ_curr, B, uplo)
        newton_step!(Δp, Δy, Δd, facwrk, divwrk, itrwrk, r, F, L, B, B_sp, H, H_sp,
                     r_c, r_p, r_d; τ=τ_aug, atol, rtol, itmax)

        # Step to boundary
        τ_p, τ_d = step_to_boundary(p, d, Δp, Δd, B, uplo; γ)
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
