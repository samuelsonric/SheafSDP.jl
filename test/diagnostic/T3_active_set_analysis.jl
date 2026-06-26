#
# T3 Active Set Analysis
#
# Purpose: Determine whether the 0.0 consecutive alignment is:
# (A) Cluster artifact: S₀'s bottom-k eigenvalues are degenerate, causing
#     cluster-internal rotation even when the active-face subspace is stable
# (B) Genuine churn: The active set (near-boundary PSD blocks) changes each step
#
# Measurements:
# 1. S₀'s bottom-k eigenvalues - are they clustered or well-separated?
# 2. Per-vertex min(eig(X_v)) - which PSD blocks are near the boundary?
# 3. Track stability of the "near-active" set from step 3 onward
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

# Compute matrix size from svec length: L = n*(n+1)/2 => n = (-1 + sqrt(1+8L))/2
function svec_to_matsize(L)
    return Int((-1 + sqrt(1 + 8*L)) / 2)
end

# Extract X_v from primal svec and compute min eigenvalue
function get_vertex_min_eigs(solver, n_i, m_i, N)
    B = solver.B
    p = solver.p  # primal variables in svec form

    min_eigs_G = Float64[]
    min_eigs_S = Float64[]

    for i in 1:N
        # G_i block (Lyapunov certificate)
        col_G = 2*(i-1) + 1
        rng_G = colrange(B, col_G)
        svec_G = p[rng_G]

        # Compute actual matrix size from svec length
        n_G = svec_to_matsize(length(svec_G))
        G_mat = zeros(n_G, n_G)
        smat!(G_mat, svec_G)
        # Symmetrize
        for a in 1:n_G, b in 1:a-1
            G_mat[b, a] = G_mat[a, b]
        end
        push!(min_eigs_G, minimum(eigvals(Symmetric(G_mat))))

        # S_i block (dissipation slack)
        col_S = 2*(i-1) + 2
        rng_S = colrange(B, col_S)
        svec_S = p[rng_S]

        n_S = svec_to_matsize(length(svec_S))
        S_mat = zeros(n_S, n_S)
        smat!(S_mat, svec_S)
        for a in 1:n_S, b in 1:a-1
            S_mat[b, a] = S_mat[a, b]
        end
        push!(min_eigs_S, minimum(eigvals(Symmetric(S_mat))))
    end

    return min_eigs_G, min_eigs_S
end

# Compute bottom-k eigenvalues of S₀
function compute_S0_eigenvalues(B, A, k)
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

        λ = eigvals(S0)
        sort!(λ)

        tol = 1e-10 * maximum(abs, λ)
        idx = findfirst(x -> x > tol, λ)
        if idx === nothing
            return nothing
        end

        end_idx = min(idx + k - 1, length(λ))
        return λ[idx:end_idx]
    catch
        return nothing
    end
end

function run_analysis(prob, name, n_i, m_i, N; k=10, max_iters=12)
    println("\n" * "="^90)
    println("Active Set Analysis: $name")
    println("="^90)

    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=max_iters)

    solver = SheafSDP.init(prob, settings)
    B = solver.B

    println("\nTracking per-vertex min eigenvalues and S₀ spectrum clustering")
    println("-"^90)

    active_sets = Vector{Set{Int}}()

    for iter in 1:max_iters
        ok = SheafSDP.step!(solver)
        gap = dot(solver.p, solver.d) / solver.ν

        # Get min eigenvalues of each vertex's PSD blocks
        min_eigs_G, min_eigs_S = get_vertex_min_eigs(solver, n_i, m_i, N)

        # Identify "near-active" vertices (min eigenvalue < threshold)
        threshold = 1e-2
        near_active_G = Set(findall(x -> x < threshold, min_eigs_G))
        near_active_S = Set(findall(x -> x < threshold, min_eigs_S))
        push!(active_sets, near_active_G ∪ near_active_S)

        # Compute S₀'s bottom-k eigenvalues
        λ_S0 = compute_S0_eigenvalues(B, solver.H, k)

        # Print step info
        @printf("\nStep %d: gap = %.2e\n", iter, gap)

        # Vertex min eigenvalues
        println("  Vertex G min-eigs: ", [@sprintf("%.2e", e) for e in min_eigs_G])
        println("  Vertex S min-eigs: ", [@sprintf("%.2e", e) for e in min_eigs_S])
        println("  Near-active (λ<1e-2): G=", collect(near_active_G), " S=", collect(near_active_S))

        # S₀ spectrum
        if λ_S0 !== nothing
            println("  S₀ bottom-$k eigenvalues:")
            for (i, λ) in enumerate(λ_S0)
                @printf("    μ_%d = %.6e", i, λ)
                if i > 1
                    ratio = λ_S0[i] / λ_S0[i-1]
                    @printf("  (ratio to prev: %.2f)", ratio)
                end
                println()
            end

            # Check clustering: ratio of consecutive eigenvalues
            max_ratio = maximum(λ_S0[i] / λ_S0[i-1] for i in 2:length(λ_S0))
            min_ratio = minimum(λ_S0[i] / λ_S0[i-1] for i in 2:length(λ_S0))
            @printf("  Clustering: max_ratio=%.2f, min_ratio=%.2f\n", max_ratio, min_ratio)
            if max_ratio < 2.0
                println("  → CLUSTERED (ratios < 2)")
            else
                println("  → SEPARATED (some ratio > 2)")
            end
        else
            println("  S₀ eigenvalues: [could not compute]")
        end

        if !ok
            break
        end
    end

    # Analyze active set stability
    println("\n" * "-"^90)
    println("ACTIVE SET STABILITY ANALYSIS")
    println("-"^90)

    if length(active_sets) >= 4
        # Compare step 3 onward
        ref_set = active_sets[3]
        stable = true
        for i in 4:length(active_sets)
            if active_sets[i] != ref_set
                stable = false
                println("  Step 3 vs Step $i: DIFFERENT")
                println("    Step 3: ", collect(ref_set))
                println("    Step $i: ", collect(active_sets[i]))
            end
        end

        if stable
            println("  Active set is STABLE from step 3 onward: ", collect(ref_set))
            println("  → 0.0 alignment is likely CLUSTER ARTIFACT")
            println("  → Block-recycling should work")
        else
            println("  Active set is CHURNING")
            println("  → 0.0 alignment is GENUINE rotation")
            println("  → Need A-dependent coarse space")
        end
    end
end

function main()
    println("="^90)
    println("T3 ACTIVE SET ANALYSIS")
    println("="^90)
    println()
    println("Distinguishing cluster artifact from genuine active-set churn")
    println()

    # SDP Complete
    N = 5
    n_i = 4
    prob, n_i_out, m_i = build_sdp_problem(N, n_i; topology=:complete)
    run_analysis(prob, "SDP Complete (N=$N, n=$n_i)", n_i_out, m_i, N; k=10, max_iters=10)

    # SDP Chain for comparison
    prob, n_i_out, m_i = build_sdp_problem(N, n_i; topology=:chain)
    run_analysis(prob, "SDP Chain (N=$N, n=$n_i)", n_i_out, m_i, N; k=10, max_iters=10)

    println("\n" * "="^90)
    println("ANALYSIS COMPLETE")
    println("="^90)
end

main()
