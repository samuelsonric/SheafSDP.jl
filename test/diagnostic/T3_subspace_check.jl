#
# T3 Subspace Check - Coordinate Consistency Verification
#
# Purpose: Properly compare S₀'s bad subspace with L's bad subspace
# by mapping them to the same coordinate space.
#
# Key issue: S₀ is m×m (edges), L is n×n (vertices)
# - S₀'s eigenvectors are m-dimensional
# - L's eigenvectors are n-dimensional
# - To compare: map L's eigenvectors to edge space via B
#
# Two distinct questions:
# 1. S₀ vs B·L overlap: Does S₀'s bad subspace come from L's mapped eigenvectors?
# 2. S₀ step-to-step: Does S₀'s bad subspace rotate across IPM steps?
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block, blocksparse, colrange

# Compute principal angle between two subspaces
function subspace_alignment(V1, V2)
    if V1 === nothing || V2 === nothing
        return NaN
    end
    if size(V1, 2) == 0 || size(V2, 2) == 0
        return NaN
    end

    Q1, _ = qr(V1)
    Q2, _ = qr(V2)

    k1 = min(size(V1, 2), size(Q1, 2))
    k2 = min(size(V2, 2), size(Q2, 2))

    Q1 = Matrix(Q1)[:, 1:k1]
    Q2 = Matrix(Q2)[:, 1:k2]

    σ = svdvals(Q1' * Q2)
    return minimum(σ)  # smallest cosine = worst alignment
end

# Compute bottom-k eigenvectors of L = B'B (n-dimensional, vertex space)
function compute_L_bottom_k(B, k)
    Bdense = Matrix(B)
    L = Bdense' * Bdense
    L = Symmetric(L)

    F = eigen(L)
    λ = F.values
    V = F.vectors

    perm = sortperm(λ)
    λ_sorted = λ[perm]
    V_sorted = V[:, perm]

    # Skip kernel
    tol = 1e-10 * maximum(abs, λ)
    idx = findfirst(x -> x > tol, λ_sorted)

    if idx === nothing
        return nothing, nothing
    end

    end_idx = min(idx + k - 1, size(V, 2))
    V_k = V_sorted[:, idx:end_idx]
    λ_k = λ_sorted[idx:end_idx]

    return V_k, λ_k
end

# Compute bottom-k eigenvectors of S₀ = B A⁻¹ B' (m-dimensional, edge space)
function compute_S0_bottom_k(B, A, k)
    m, n = size(B)
    N = SheafSDP.nvtxs(B)

    Bdense = Matrix(B)

    # Build block-diagonal A
    Adense = zeros(n, n)
    for v in 1:N
        rng = SheafSDP.colrange(B, v)
        Av = Matrix(block(A, v, v, v))
        if any(isnan, Av) || any(isinf, Av)
            return nothing, nothing
        end
        Adense[rng, rng] .= Av
    end

    eigA = eigvals(Symmetric(Adense))
    if minimum(eigA) <= 0
        return nothing, nothing
    end

    try
        S0 = Bdense * (Adense \ Bdense')
        S0 = Symmetric((S0 + S0') / 2)

        F = eigen(S0)
        λ = F.values
        V = F.vectors

        perm = sortperm(λ)
        λ_sorted = λ[perm]
        V_sorted = V[:, perm]

        tol = 1e-10 * maximum(abs, λ)
        idx = findfirst(x -> x > tol, λ_sorted)

        if idx === nothing
            return nothing, nothing
        end

        end_idx = min(idx + k - 1, size(V, 2))
        return V_sorted[:, idx:end_idx], λ_sorted[idx:end_idx]
    catch
        return nothing, nothing
    end
end

#=============================================================================
   SDP Problem Builder
=============================================================================#

function svecdim(n)
    return div(n * (n + 1), 2)
end

function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C)
    α = roottwo(T)
    H = zeros(T, svecdim(d), svecdim(n))

    tkl = 1
    @inbounds for l in 1:n
        tab = 0
        for b in 1:d
            Cbl = C[b, l]
            tab += 1
            H[tab, tkl] = Cbl^2
            for a in b + 1:d
                tab += 1
                H[tab, tkl] = α * C[a, l] * Cbl
            end
        end
        for k in l + 1:n
            tkl += 1
            tab = 0
            for b in 1:d
                Cbk = C[b, k]
                Cbl = C[b, l]
                tab += 1
                H[tab, tkl] = α * Cbk * Cbl
                for a in b + 1:d
                    Cak = C[a, k]
                    Cal = C[a, l]
                    tab += 1
                    H[tab, tkl] = Cak * Cbl + Cal * Cbk
                end
            end
        end
        tkl += 1
    end
    return H
end

function passivity_lmi_operator(A::AbstractMatrix{T}, B_mat::AbstractMatrix{T},
                                 C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    m = size(B_mat, 2)
    nm = n + m

    sv_G = svecdim(n)
    sv_D = svecdim(nm)

    L = zeros(T, sv_D, sv_G)
    d0 = zeros(T, sv_D)

    G = zeros(T, n, n)
    M = zeros(T, nm, nm)
    v = zeros(T, sv_D)

    for k in 1:sv_G
        fill!(G, zero(T))
        smat!(G, setindex!(zeros(T, sv_G), one(T), k))
        for i in 1:n, j in 1:i-1
            G[j, i] = G[i, j]
        end

        M[1:n, 1:n] .= A * G .+ G * A'
        M[1:n, n+1:nm] .= -G * C'
        M[n+1:nm, 1:n] .= -C * G
        M[n+1:nm, n+1:nm] .= zero(T)

        svec!(v, M)
        L[:, k] .= v
    end

    fill!(M, zero(T))
    M[1:n, n+1:nm] .= B_mat
    M[n+1:nm, 1:n] .= B_mat'
    M[n+1:nm, n+1:nm] .= -(D .+ D')
    svec!(d0, M)

    return L, d0
end

function random_passive_system(n::Int, rng=Random.default_rng())
    Q = randn(rng, n, n)
    Q = Q'Q + I
    A = -Q
    B_mat = randn(rng, n, 1)
    C = B_mat'
    D = fill(1.0 + abs(randn(rng)), 1, 1)
    return A, B_mat, C, D
end

function build_sdp_problem(N, n_i; topology=:chain)
    T = Float64
    Random.seed!(42)

    m_i = 1
    d_e = min(2, n_i)

    if topology == :chain
        edges = [(i, i+1) for i in 1:N-1]
    elseif topology == :complete
        edges = [(i, j) for i in 1:N for j in i+1:N]
    end
    n_edges = length(edges)

    base_system = random_passive_system(n_i)
    systems = [base_system for _ in 1:N]

    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for _ in 1:n_edges
        C = zeros(T, d_e, n_i)
        for k in 1:d_e
            C[k, k] = 1.0
        end
        push!(interface_maps, (copy(C), copy(C)))
    end

    sv_G = svecdim(n_i)
    sv_S = svecdim(n_i + m_i)
    sv_edge = svecdim(d_e)

    col_G(i) = 2*(i-1) + 1
    col_S(i) = 2*(i-1) + 2
    row_diss(i) = i
    row_agree(e) = N + e

    n_col_blocks = 2 * N

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]
    g_vec = T[]

    for i in 1:N
        A, B_mat, C, D = systems[i]
        L, d0 = passivity_lmi_operator(A, B_mat, C, D)

        push!(row_ids, row_diss(i))
        push!(col_ids, col_S(i))
        push!(blocks, Matrix{T}(I, sv_S, sv_S))

        push!(row_ids, row_diss(i))
        push!(col_ids, col_G(i))
        push!(blocks, L)

        append!(g_vec, -d0)
    end

    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        K_i = skronr(C_i)
        K_j = skronr(C_j)

        push!(row_ids, row_agree(e))
        push!(col_ids, col_G(i))
        push!(blocks, K_i)

        push!(row_ids, row_agree(e))
        push!(col_ids, col_G(j))
        push!(blocks, -K_j)

        append!(g_vec, zeros(T, sv_edge))
    end

    B = blocksparse(row_ids, col_ids, blocks)
    g = g_vec

    c_vec = zeros(T, size(B, 2))
    I_n = Matrix{T}(I, n_i, n_i)
    svec_I = zeros(T, sv_G)
    svec!(svec_I, I_n)
    for i in 1:N
        c_vec[colrange(B, col_G(i))] .= svec_I
    end

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, zero(T))

    cones = Vector{Cone}(undef, n_col_blocks)
    for i in 1:N
        cones[col_G(i)] = SemidefiniteCone()
        cones[col_S(i)] = SemidefiniteCone()
    end

    return IPMProblem(c_vec, g, B, Q, cones), "SDP_$(topology)_N$(N)_n$(n_i)"
end

# LP problem for comparison
function build_lp_problem(N, d_v, d_e; topology=:chain)
    Random.seed!(42)
    T = Float64

    src, tgt, maps = Int[], Int[], Matrix{T}[]

    if topology == :chain
        for i in 1:N-1
            push!(src, i); push!(tgt, i+1); push!(maps, randn(d_e, d_v))
            push!(src, i+1); push!(tgt, i); push!(maps, randn(d_e, d_v))
        end
    elseif topology == :complete
        for i in 1:N, j in i+1:N
            push!(src, i); push!(tgt, j); push!(maps, randn(d_e, d_v))
            push!(src, j); push!(tgt, i); push!(maps, randn(d_e, d_v))
        end
    end

    B = SheafSDP.sheaf(src, tgt, maps)

    c = abs.(randn(size(B, 2))) .+ 0.1
    x_bar = ones(size(B, 2)) .+ abs.(randn(size(B, 2)))
    g = B * x_bar

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0.0)

    cones = [PositiveCone() for _ in 1:N]
    return IPMProblem(c, g, B, Q, cones), "LP_$(topology)_N$(N)"
end

#=============================================================================
   Main Diagnostic
=============================================================================#

function run_subspace_check(prob, name; k=5, raug=1e6, max_iters=30)
    println("\n" * "="^80)
    println("T3 Subspace Check: $name")
    println("="^80)

    T = Float64
    kkt = UzawaSettings{T}(raug=raug)
    settings = IPMSettings{T}(kkt=kkt, verbose=false, itmax=max_iters)

    solver = SheafSDP.init(prob, settings)
    B = solver.B
    Bdense = Matrix(B)
    m, n = size(Bdense)

    println("\nDimensions:")
    @printf("  B: %d × %d (edges × vertices)\n", m, n)
    @printf("  S₀: %d × %d (edge space)\n", m, m)
    @printf("  L:  %d × %d (vertex space)\n", n, n)

    # Get L's bottom-k eigenvectors
    V_L, λ_L = compute_L_bottom_k(B, k)
    if V_L === nothing
        println("  [Could not compute L eigenvectors]")
        return
    end

    # Map L's eigenvectors to edge space: B * V_L
    BV_L = Bdense * V_L

    # Orthonormalize the mapped vectors (they may not be orthonormal after mapping)
    Q_BVL, _ = qr(BV_L)
    Q_BVL = Matrix(Q_BVL)[:, 1:min(k, size(Q_BVL, 2))]

    println("\nL = B'B spectrum (bottom-$k non-kernel):")
    for (i, λ) in enumerate(λ_L)
        @printf("  λ_%d(L) = %.6e\n", i, λ)
    end

    println("\nTracking across IPM steps:")
    println("-"^80)
    @printf("%5s │ %12s │ %12s │ %12s │ %12s\n",
            "Step", "Gap", "S₀_vs_BL", "S₀_vs_step1", "S₀_vs_prev")
    println("─"^65)

    V_S0_first = nothing
    V_S0_prev = nothing

    iteration = 0
    while true
        iteration += 1

        ok = SheafSDP.step!(solver)
        gap = dot(solver.p, solver.d) / solver.ν

        # Get S₀'s bottom-k eigenvectors (m-dimensional, edge space)
        V_S0, λ_S0 = compute_S0_bottom_k(B, solver.H, k)

        if V_S0 === nothing
            @printf("%5d │ %12.4e │ %12s │ %12s │ %12s\n",
                    iteration, gap, "NaN", "NaN", "NaN")
        else
            # Comparison 1: S₀ vs B·L (structural overlap)
            # Both are now in edge space (m-dimensional)
            align_BL = subspace_alignment(V_S0, Q_BVL)

            # Comparison 2: S₀ step 1 vs S₀ current (drift)
            if V_S0_first === nothing
                V_S0_first = copy(V_S0)
            end
            align_first = subspace_alignment(V_S0_first, V_S0)

            # Comparison 3: S₀ previous vs S₀ current (step-to-step drift)
            align_prev = V_S0_prev === nothing ? 1.0 : subspace_alignment(V_S0_prev, V_S0)

            @printf("%5d │ %12.4e │ %12.4f │ %12.4f │ %12.4f\n",
                    iteration, gap, align_BL, align_first, align_prev)

            V_S0_prev = copy(V_S0)
        end

        if !ok || iteration >= max_iters
            break
        end
    end

    println("\n" * "-"^80)
    println("INTERPRETATION:")
    println("-"^80)
    println("  S₀_vs_BL:    Overlap of S₀'s bad modes with B·(L's bad modes)")
    println("               High → structural (L-deflation could work)")
    println("               Low  → S₀'s bad modes are NOT from L's eigenvectors")
    println()
    println("  S₀_vs_step1: Overlap of current S₀ bad modes with step 1's")
    println("               High → stable subspace (fixed deflation works)")
    println("               Low  → subspace drifts (need recycling)")
    println()
    println("  S₀_vs_prev:  Step-to-step overlap")
    println("               High → slow drift (recycling every N steps)")
    println("               Low  → fast drift (recycling every step)")
end

function main()
    println("="^80)
    println("T3 SUBSPACE CHECK: Coordinate Consistency")
    println("="^80)
    println()
    println("Resolving the tension: is 0.0003 alignment 'drift' or 'wrong subspace'?")
    println()
    println("Two distinct questions:")
    println("  1. Does S₀'s bad subspace overlap with B·(L's bad eigenvectors)?")
    println("  2. Does S₀'s bad subspace rotate across IPM steps?")
    println()
    println("Both are measured in edge space (m-dimensional) for consistency.")

    # LP Chain
    prob, name = build_lp_problem(10, 4, 2; topology=:chain)
    run_subspace_check(prob, name; k=5)

    # LP Complete
    prob, name = build_lp_problem(8, 4, 2; topology=:complete)
    run_subspace_check(prob, name; k=5)

    # SDP Chain
    prob, name = build_sdp_problem(5, 4; topology=:chain)
    run_subspace_check(prob, name; k=5)

    # SDP Complete - try larger k to capture full degenerate eigenspace
    prob, name = build_sdp_problem(5, 4; topology=:complete)
    run_subspace_check(prob, name; k=10)  # L has 4-fold degeneracy, need k > 4

    println("\n" * "="^80)
    println("T3 SUBSPACE CHECK COMPLETE")
    println("="^80)
end

main()
