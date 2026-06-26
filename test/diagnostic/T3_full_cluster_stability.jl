#
# T3 Full Cluster Stability
#
# The bottom-4 period-2 is a truncation artifact from cutting through
# the μ₄/μ₅ near-boundary. The FULL bottom-8 cluster should be stable.
#
# Final check: min(svdvals(Q8_k' * Q8_{k+1})) for steps 4→5→6→7
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using BlockSparseArrays: block, blocksparse, colrange

# Problem builder
function svecdim(n); div(n * (n + 1), 2); end
function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C); α = roottwo(T); H = zeros(T, svecdim(d), svecdim(n)); tkl = 1
    @inbounds for l in 1:n; tab = 0; for b in 1:d; Cbl = C[b, l]; tab += 1; H[tab, tkl] = Cbl^2
        for a in b + 1:d; tab += 1; H[tab, tkl] = α * C[a, l] * Cbl; end; end
    for kk in l + 1:n; tkl += 1; tab = 0; for b in 1:d; Cbk, Cbl = C[b, kk], C[b, l]; tab += 1; H[tab, tkl] = α * Cbk * Cbl
        for a in b + 1:d; tab += 1; H[tab, tkl] = C[a, kk] * Cbl + C[a, l] * Cbk; end; end; end; tkl += 1; end; return H; end
function passivity_lmi_operator(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n, m = size(A, 1), size(B, 2); nm = n + m; sv_G, sv_D = svecdim(n), svecdim(nm)
    L, d0 = zeros(T, sv_D, sv_G), zeros(T, sv_D); G, M, v = zeros(T, n, n), zeros(T, nm, nm), zeros(T, sv_D)
    for kk in 1:sv_G; fill!(G, zero(T)); smat!(G, setindex!(zeros(T, sv_G), one(T), kk))
        for ii in 1:n, jj in 1:ii-1; G[jj, ii] = G[ii, jj]; end
        M[1:n, 1:n] .= A * G .+ G * A'; M[1:n, n+1:nm] .= -G * C'; M[n+1:nm, 1:n] .= -C * G; M[n+1:nm, n+1:nm] .= zero(T)
        svec!(v, M); L[:, kk] .= v; end
    fill!(M, zero(T)); M[1:n, n+1:nm] .= B; M[n+1:nm, 1:n] .= B'; M[n+1:nm, n+1:nm] .= -(D .+ D'); svec!(d0, M); return L, d0; end
function random_passive_system(n::Int, rng=Random.default_rng()); Q = randn(rng, n, n); Q = Q'Q + I; A = -Q; B = randn(rng, n, 1); C = B'; D = fill(1.0 + abs(randn(rng)), 1, 1); return A, B, C, D; end
function build_sdp_problem(N, n_i)
    T = Float64; Random.seed!(42); m_i = 1; d_e = min(2, n_i); edges = [(ii, jj) for ii in 1:N for jj in ii+1:N]
    base_system = random_passive_system(n_i); systems = [base_system for _ in 1:N]
    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}(); for _ in edges; C = zeros(T, d_e, n_i); for kk in 1:d_e; C[kk,kk] = 1.0; end; push!(interface_maps, (copy(C), copy(C))); end
    sv_G, sv_S, sv_edge = svecdim(n_i), svecdim(n_i + m_i), svecdim(d_e)
    col_G(idx) = 2*(idx-1)+1; col_S(idx) = 2*(idx-1)+2; row_diss(idx) = idx; row_agree(idx) = N + idx
    row_ids, col_ids, blocks, g_vec = Int[], Int[], Matrix{T}[], T[]
    for vi in 1:N; A, B, C, D = systems[vi]; L, d0 = passivity_lmi_operator(A, B, C, D)
        push!(row_ids, row_diss(vi)); push!(col_ids, col_S(vi)); push!(blocks, Matrix{T}(I, sv_S, sv_S))
        push!(row_ids, row_diss(vi)); push!(col_ids, col_G(vi)); push!(blocks, L); append!(g_vec, -d0); end
    for (e, (vi, vj)) in enumerate(edges); C_i, C_j = interface_maps[e]; K_i, K_j = skronr(C_i), skronr(C_j)
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(vi)); push!(blocks, K_i)
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(vj)); push!(blocks, -K_j); append!(g_vec, zeros(T, sv_edge)); end
    B = blocksparse(row_ids, col_ids, blocks); c_vec = zeros(T, size(B, 2)); I_n = Matrix{T}(I, n_i, n_i); svec_I = zeros(T, sv_G); svec!(svec_I, I_n)
    for vi in 1:N; c_vec[colrange(B, col_G(vi))] .= svec_I; end
    Q = SheafSDP.allocblockdiag(B); fill!(Q, zero(T)); cones = Vector{Cone}(undef, 2*N)
    for vi in 1:N; cones[col_G(vi)] = SemidefiniteCone(); cones[col_S(vi)] = SemidefiniteCone(); end
    return IPMProblem(c_vec, g_vec, B, Q, cones); end

function compute_S0_eigen(B, A)
    m, n = size(B); N = SheafSDP.nvtxs(B); Bdense, Adense = Matrix(B), zeros(n, n)
    for v in 1:N; rng = SheafSDP.colrange(B, v); Av = Matrix(block(A, v, v, v))
        any(isnan, Av) || any(isinf, Av) && return nothing, nothing; Adense[rng, rng] .= Av; end
    minimum(eigvals(Symmetric(Adense))) <= 0 && return nothing, nothing
    try
        S0 = Bdense * (Adense \ Bdense'); S0 = Symmetric((S0 + S0') / 2)
        F = eigen(S0); return F.values, F.vectors
    catch; return nothing, nothing; end
end

# Get orthonormal basis for bottom-k subspace, cutting at first real gap
function get_cluster_basis(λ, V; gap_threshold=1.5)
    perm = sortperm(λ); λ_sorted = λ[perm]; V_sorted = V[:, perm]
    tol = 1e-10 * maximum(abs, λ); idx = findfirst(x -> x > tol, λ_sorted)
    idx === nothing && return nothing, nothing, 0

    # Find first real gap (ratio > threshold)
    k = 1
    for i in idx+1:length(λ_sorted)
        ratio = λ_sorted[i] / λ_sorted[i-1]
        if ratio > gap_threshold
            break
        end
        k += 1
    end

    V_k = V_sorted[:, idx:idx+k-1]
    λ_k = λ_sorted[idx:idx+k-1]
    Q, _ = qr(V_k)
    return Matrix(Q), λ_k, k
end

function main()
    println("="^90)
    println("FULL CLUSTER STABILITY CHECK")
    println("="^90)
    println()
    println("Testing the full bottom-cluster (to first real gap), not fixed k")
    println()

    N, n_i = 5, 4
    prob = build_sdp_problem(N, n_i)
    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=10)
    solver = SheafSDP.init(prob, settings)
    B = solver.B

    # Store bases for each step
    bases_k4 = Matrix{Float64}[]
    bases_k8 = Matrix{Float64}[]
    bases_adaptive = Matrix{Float64}[]
    cluster_sizes = Int[]

    println("Step-by-step eigenvalue structure:")
    println("-"^90)

    for iter in 1:10
        SheafSDP.step!(solver)
        λ, V = compute_S0_eigen(B, solver.H)
        if λ === nothing; continue; end

        gap = dot(solver.p, solver.d) / solver.ν

        # Fixed k=4
        perm = sortperm(λ); λ_sorted = λ[perm]; V_sorted = V[:, perm]
        tol = 1e-10 * maximum(abs, λ); idx = findfirst(x -> x > tol, λ_sorted)

        V_k4 = V_sorted[:, idx:idx+3]; Q4, _ = qr(V_k4); push!(bases_k4, Matrix(Q4))
        V_k8 = V_sorted[:, idx:idx+7]; Q8, _ = qr(V_k8); push!(bases_k8, Matrix(Q8))

        # Adaptive: cut at first gap > 1.5
        Q_adapt, λ_adapt, k_adapt = get_cluster_basis(λ, V; gap_threshold=1.5)
        push!(bases_adaptive, Q_adapt)
        push!(cluster_sizes, k_adapt)

        @printf("Step %d: gap=%.2e, cluster_size=%d\n", iter, gap, k_adapt)
        @printf("  λ₁-λ₄:  [%.2e, %.2e, %.2e, %.2e]\n",
                λ_sorted[idx], λ_sorted[idx+1], λ_sorted[idx+2], λ_sorted[idx+3])
        @printf("  λ₅-λ₈:  [%.2e, %.2e, %.2e, %.2e]\n",
                λ_sorted[idx+4], λ_sorted[idx+5], λ_sorted[idx+6], λ_sorted[idx+7])
        @printf("  λ₉-λ₁₂: [%.2e, %.2e, %.2e, %.2e]\n",
                λ_sorted[idx+8], λ_sorted[idx+9], λ_sorted[idx+10], λ_sorted[idx+11])
        @printf("  Ratios: μ₅/μ₄=%.2f, μ₉/μ₈=%.2f\n",
                λ_sorted[idx+4]/λ_sorted[idx+3], λ_sorted[idx+8]/λ_sorted[idx+7])
    end

    println("\n" * "="^90)
    println("SUBSPACE ALIGNMENT COMPARISON")
    println("="^90)

    println("\n--- Fixed k=4 (the artifact-prone truncation) ---")
    for i in 4:min(7, length(bases_k4)-1)
        M = bases_k4[i]' * bases_k4[i+1]
        σ = svdvals(M)
        @printf("Step %d → %d: min(σ)=%.4f, all σ=%s\n", i, i+1, minimum(σ),
                [@sprintf("%.3f", s) for s in σ])
    end

    println("\n--- Fixed k=8 (full merged cluster) ---")
    for i in 4:min(7, length(bases_k8)-1)
        M = bases_k8[i]' * bases_k8[i+1]
        σ = svdvals(M)
        @printf("Step %d → %d: min(σ)=%.4f, all σ=%s\n", i, i+1, minimum(σ),
                [@sprintf("%.3f", s) for s in σ[1:min(8,length(σ))]])
    end

    println("\n--- Adaptive (cut at first gap > 1.5) ---")
    for i in 4:min(7, length(bases_adaptive)-1)
        k_i, k_j = cluster_sizes[i], cluster_sizes[i+1]
        if k_i == k_j && bases_adaptive[i] !== nothing && bases_adaptive[i+1] !== nothing
            M = bases_adaptive[i]' * bases_adaptive[i+1]
            σ = svdvals(M)
            @printf("Step %d → %d (k=%d): min(σ)=%.4f\n", i, i+1, k_i, minimum(σ))
        else
            @printf("Step %d → %d: cluster size changed (%d → %d)\n", i, i+1, k_i, k_j)
        end
    end

    println("\n" * "="^90)
    println("VERDICT")
    println("="^90)

    # Check if k=8 is stable
    stable_k8 = true
    for i in 4:min(6, length(bases_k8)-1)
        M = bases_k8[i]' * bases_k8[i+1]
        if minimum(svdvals(M)) < 0.8
            stable_k8 = false
            break
        end
    end

    if stable_k8
        println("\nk=8 (full cluster) is STABLE: build W once at step 4, refresh at cardinality change")
        println("The period-2 in k=4 was a truncation artifact from cutting through μ₄/μ₅ boundary")
    else
        println("\nk=8 also drifts - need to investigate further or use recycling")
    end
end

main()
