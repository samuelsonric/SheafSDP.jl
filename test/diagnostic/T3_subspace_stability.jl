#
# T3 Subspace Stability - Final Measurement
#
# Purpose: Measure whether the bottom-k SUBSPACE is stable across steps,
# even though individual eigenvectors rotate within the degenerate cluster.
#
# Key metric: min singular value of Q_k' * Q_{k+1}
#   - High (>0.9) = subspace nearly static, build once and reuse
#   - Low = subspace drifts, need per-step refresh
#
# This quotienting out the in-cluster rotation that fooled earlier measurements.
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block, blocksparse, colrange

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

function build_sdp_problem(N, n_i; topology=:complete)
    T = Float64
    Random.seed!(42)
    m_i = 1
    d_e = min(2, n_i)

    if topology == :chain
        edges = [(i, i+1) for i in 1:N-1]
    else
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

    return IPMProblem(c_vec, g, B, Q, cones), n_i, m_i
end

# Compute bottom-k orthonormal basis of S₀
function compute_S0_subspace(B, A, k)
    m, n = size(B)
    N = SheafSDP.nvtxs(B)

    Bdense = Matrix(B)
    Adense = zeros(n, n)

    for v in 1:N
        rng = SheafSDP.colrange(B, v)
        Av = Matrix(block(A, v, v, v))
        if any(isnan, Av) || any(isinf, Av)
            return nothing
        end
        Adense[rng, rng] .= Av
    end

    eigA = eigvals(Symmetric(Adense))
    if minimum(eigA) <= 0
        return nothing
    end

    try
        S0 = Bdense * (Adense \ Bdense')
        S0 = Symmetric((S0 + S0') / 2)

        F = eigen(S0)
        λ = F.values
        V = F.vectors

        # Sort by eigenvalue
        perm = sortperm(λ)
        λ_sorted = λ[perm]
        V_sorted = V[:, perm]

        # Skip kernel, take bottom-k non-kernel eigenvectors
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

# Subspace alignment: min singular value of Q1' * Q2
# This measures how much of Q1's span is captured by Q2's span
function subspace_alignment(Q1, Q2)
    if Q1 === nothing || Q2 === nothing
        return NaN
    end
    # Ensure same dimensions
    k = min(size(Q1, 2), size(Q2, 2))
    Q1 = Q1[:, 1:k]
    Q2 = Q2[:, 1:k]

    M = Q1' * Q2
    σ = svdvals(M)
    return minimum(σ)  # Smallest singular value
end

function run_analysis(prob, name, n_i, m_i, N; k=8, max_iters=12)
    println("\n" * "="^90)
    println("Subspace Stability Analysis: $name")
    println("="^90)

    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=max_iters)

    solver = SheafSDP.init(prob, settings)
    B = solver.B

    println("\nTracking bottom-$k subspace stability (not individual eigenvectors)")
    println("-"^90)

    # Store subspaces for each step
    subspaces = Vector{Union{Nothing, Matrix{Float64}}}()
    eigenvalues = Vector{Union{Nothing, Vector{Float64}}}()

    for iter in 1:max_iters
        ok = SheafSDP.step!(solver)
        gap = dot(solver.p, solver.d) / solver.ν

        # Get bottom-k subspace
        Q, λ = compute_S0_subspace(B, solver.H, k)
        push!(subspaces, Q)
        push!(eigenvalues, λ)

        @printf("\nStep %d: gap = %.2e\n", iter, gap)

        if λ !== nothing
            # Show eigenvalue structure
            println("  Bottom-$k eigenvalues: ", [@sprintf("%.2e", e) for e in λ[1:min(k,length(λ))]])

            # Identify clusters (consecutive ratio < 1.1)
            clusters = Int[]
            cluster_start = 1
            for i in 2:length(λ)
                if λ[i] / λ[i-1] > 1.5  # Gap between clusters
                    push!(clusters, i - cluster_start)
                    cluster_start = i
                end
            end
            push!(clusters, length(λ) - cluster_start + 1)
            println("  Cluster sizes: ", clusters)
        end

        # Compute subspace alignment with previous step
        if iter > 1 && subspaces[iter] !== nothing && subspaces[iter-1] !== nothing
            align = subspace_alignment(subspaces[iter-1], subspaces[iter])
            @printf("  Subspace alignment with step %d: %.4f", iter-1, align)
            if align > 0.95
                println(" (STABLE)")
            elseif align > 0.8
                println(" (MILDLY DRIFTING)")
            else
                println(" (DRIFTING)")
            end
        end

        if !ok
            break
        end
    end

    # Summary analysis from step 4 onward
    println("\n" * "-"^90)
    println("SUBSPACE STABILITY SUMMARY (from step 4)")
    println("-"^90)

    if length(subspaces) >= 5
        # Compute all pairwise alignments from step 4
        println("\nStep-to-step subspace alignment (min σ of Q_k' Q_{k+1}):")
        alignments = Float64[]
        for i in 4:length(subspaces)-1
            if subspaces[i] !== nothing && subspaces[i+1] !== nothing
                align = subspace_alignment(subspaces[i], subspaces[i+1])
                push!(alignments, align)
                @printf("  Step %d → %d: %.4f\n", i, i+1, align)
            end
        end

        if !isempty(alignments)
            avg_align = sum(alignments) / length(alignments)
            min_align = minimum(alignments)
            @printf("\n  Average alignment: %.4f\n", avg_align)
            @printf("  Minimum alignment: %.4f\n", min_align)

            if min_align > 0.95
                println("\n  → SUBSPACE IS STATIC: Build coarse space once at step 4, reuse")
            elseif min_align > 0.8
                println("\n  → SUBSPACE MILDLY DRIFTS: Refresh every 3-5 steps")
            else
                println("\n  → SUBSPACE DRIFTS: Need per-step refresh")
            end
        end

        # Also check alignment with step 4 reference
        println("\nAlignment with step 4 reference:")
        if subspaces[4] !== nothing
            for i in 5:length(subspaces)
                if subspaces[i] !== nothing
                    align = subspace_alignment(subspaces[4], subspaces[i])
                    @printf("  Step 4 → %d: %.4f\n", i, align)
                end
            end
        end
    end
end

function main()
    println("="^90)
    println("T3 SUBSPACE STABILITY - FINAL MEASUREMENT")
    println("="^90)
    println()
    println("Testing whether the bottom-k SUBSPACE is stable across steps,")
    println("even though individual eigenvectors rotate within degenerate clusters.")
    println()
    println("Key metric: min singular value of Q_k' * Q_{k+1}")
    println("  - High (>0.9) = subspace nearly static, build once and reuse")
    println("  - Low = subspace drifts, need per-step refresh")
    println()

    # SDP Complete - expected stable subspace despite 4-fold degeneracy
    N = 5
    n_i = 4
    prob, n_i_out, m_i = build_sdp_problem(N, n_i; topology=:complete)
    run_analysis(prob, "SDP Complete (N=$N, n=$n_i)", n_i_out, m_i, N; k=8, max_iters=12)

    # SDP Chain - expected mildly drifting
    prob, n_i_out, m_i = build_sdp_problem(N, n_i; topology=:chain)
    run_analysis(prob, "SDP Chain (N=$N, n=$n_i)", n_i_out, m_i, N; k=8, max_iters=12)

    println("\n" * "="^90)
    println("ANALYSIS COMPLETE")
    println("="^90)
end

main()
