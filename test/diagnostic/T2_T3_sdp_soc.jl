#
# T2/T3 - SDP/SOC Diagnosis
#
# Purpose: Test whether μ_min(S₀) collapses and whether the bottom-k subspace
# drifts for the hard cones (SDP/SOC) where the barrier Hessian can DEGENERATE
# rather than uniformly grow.
#
# Key differences from LP:
# - SDP barrier: A = X⁻¹ ⊗ X⁻¹ has both huge AND tiny eigenvalues near boundary
# - SOC barrier: similar anisotropic behavior near cone apex
#
# Critical fix from v1: Track bottom-k SUBSPACE angle, not single eigenvector.
# Chains have nearly degenerate low eigenvalues, so single-vector can rotate
# freely even when the bottom-k subspace is stable.
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block, blocksparse, colrange

struct StepData
    iteration::Int
    gap::Float64
    μ_min_S0::Float64
    subspace_angle_k5::Float64   # angle with FIRST step's bottom-5 subspace
    subspace_angle_prev::Float64 # angle with PREVIOUS step's bottom-5 subspace
    kkt_iters::Int
end

# Compute principal angle between two subspaces (returns cos of smallest angle)
# cos = 1 means perfect alignment, cos = 0 means orthogonal
function subspace_alignment(V1, V2)
    if V1 === nothing || V2 === nothing
        return NaN
    end
    if size(V1, 2) == 0 || size(V2, 2) == 0
        return NaN
    end

    # Orthonormalize both
    Q1, _ = qr(V1)
    Q2, _ = qr(V2)

    k1 = min(size(V1, 2), size(Q1, 2))
    k2 = min(size(V2, 2), size(Q2, 2))

    Q1 = Matrix(Q1)[:, 1:k1]
    Q2 = Matrix(Q2)[:, 1:k2]

    # Principal angles: cosines are singular values of Q1'*Q2
    σ = svdvals(Q1' * Q2)

    # Return smallest cosine (worst alignment in the subspace)
    return minimum(σ)
end

function compute_schur_bottom_k(B, A, k; tol=1e-10)
    m, n = size(B)
    N = SheafSDP.nvtxs(B)

    Bdense = Matrix(B)

    # Build block-diagonal A
    Adense = zeros(n, n)
    for v in 1:N
        rng = SheafSDP.colrange(B, v)
        Av = Matrix(block(A, v, v, v))
        if any(isnan, Av) || any(isinf, Av)
            return NaN, nothing
        end
        Adense[rng, rng] .= Av
    end

    # Check A is positive definite
    # Add small regularization if near-singular (for mixed cones with CofreeCone)
    eigA = eigvals(Symmetric(Adense))
    min_eig = minimum(eigA)
    max_eig = maximum(eigA)

    if min_eig <= 0
        # Try regularization for CofreeCone (μ=0) blocks
        reg = max(1e-8, 1e-8 * max_eig)
        Adense = Adense + reg * I
        eigA = eigvals(Symmetric(Adense))
        if minimum(eigA) <= 0
            return NaN, nothing
        end
    end

    try
        # S₀ = B A⁻¹ B'
        S0 = Bdense * (Adense \ Bdense')
        S0 = Symmetric((S0 + S0') / 2)

        F = eigen(S0)
        λ = F.values
        V = F.vectors

        # Sort by eigenvalue
        perm = sortperm(λ)
        λ_sorted = λ[perm]
        V_sorted = V[:, perm]

        # Find first non-kernel eigenvalue
        max_λ = maximum(abs, λ)
        tol_eig = max(tol, 1e-10 * max_λ)
        idx = findfirst(x -> x > tol_eig, λ_sorted)

        if idx === nothing
            return NaN, nothing
        end

        μ_min = λ_sorted[idx]

        # Extract bottom-k eigenvectors (non-kernel)
        end_idx = min(idx + k - 1, size(V, 2))
        V_bottom_k = V_sorted[:, idx:end_idx]

        return μ_min, V_bottom_k
    catch e
        return NaN, nothing
    end
end

function run_instrumented_ipm(prob; raug=1e6, verbose=false, max_iters=50, k_subspace=5)
    T = Float64
    kkt = UzawaSettings{T}(raug=raug)
    settings = IPMSettings{T}(kkt=kkt, verbose=verbose, itmax=max_iters)

    solver = SheafSDP.init(prob, settings)

    step_data = StepData[]
    B = solver.B

    V_first = nothing  # First step's bottom-k subspace
    V_prev = nothing   # Previous step's bottom-k subspace

    iteration = 0
    while true
        iteration += 1

        ok = SheafSDP.step!(solver)

        μ_gap = dot(solver.p, solver.d) / solver.ν
        kkt_iters = solver.kkt_iters

        # Compute bottom-k eigenvectors of S₀
        μ_min, V_k = try
            compute_schur_bottom_k(B, solver.H, k_subspace)
        catch
            (NaN, nothing)
        end

        # Store first step's subspace
        if V_first === nothing && V_k !== nothing
            V_first = copy(V_k)
        end

        # Compute subspace alignments
        align_first = subspace_alignment(V_first, V_k)
        align_prev = if V_prev !== nothing
            subspace_alignment(V_prev, V_k)
        else
            1.0  # First step trivially aligned
        end

        push!(step_data, StepData(
            iteration, μ_gap, μ_min, align_first, align_prev, kkt_iters
        ))

        if V_k !== nothing
            V_prev = copy(V_k)
        end

        if verbose
            @printf("Step %3d: gap=%.2e, μ_min(S₀)=%.2e, align_k5_first=%.4f, align_k5_prev=%.4f, KKT=%d\n",
                    iteration, μ_gap, μ_min, align_first, align_prev, kkt_iters)
        end

        if !ok || iteration >= max_iters
            break
        end
    end

    return step_data, solver
end

#=============================================================================
   SDP Problem: Passivity/Dissipativity LMI (from test/small/dissipativity.jl)
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

function build_sdp_chain_problem(N, n_i; topology=:chain)
    T = Float64
    Random.seed!(42)

    m_i = 1
    d_e = min(2, n_i)

    if topology == :chain
        edges = [(i, i+1) for i in 1:N-1]
    elseif topology == :complete
        edges = [(i, j) for i in 1:N for j in i+1:N]
    else
        error("Unknown topology: $topology")
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

#=============================================================================
   SOC Problem: Group-sparse control (from test/small/soc.jl)
=============================================================================#

function build_soc_problem(N, T_steps; topology=:complete)
    Random.seed!(42)
    T = Float64

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
    ū = 100.0

    x0 = [randn(nx) for _ in 1:N]

    if topology == :complete
        edges = [(i, j) for i in 1:N for j in i+1:N]
    elseif topology == :chain
        edges = [(i, i+1) for i in 1:N-1]
    else
        error("Unknown topology: $topology")
    end

    blocks_per_agent = T_steps + 3 * (T_steps - 1)

    col_x(i, t) = (i - 1) * blocks_per_agent + t
    col_ζ(i, t) = (i - 1) * blocks_per_agent + T_steps + 3 * (t - 1) + 1
    col_sp(i, t) = (i - 1) * blocks_per_agent + T_steps + 3 * (t - 1) + 2
    col_sm(i, t) = (i - 1) * blocks_per_agent + T_steps + 3 * (t - 1) + 3

    rows_per_agent = 1 + 3 * (T_steps - 1)

    row_init(i) = (i - 1) * rows_per_agent + 1
    row_dyn(i, t) = (i - 1) * rows_per_agent + 1 + t
    row_boxp(i, t) = (i - 1) * rows_per_agent + T_steps + (t - 1) + 1
    row_boxm(i, t) = (i - 1) * rows_per_agent + T_steps + (T_steps - 1) + (t - 1) + 1
    row_coord(e) = N * rows_per_agent + e

    row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

    invrt2 = 1 / sqrt(2.0)
    B_on_ζ = [zeros(nx, 1) B_dyn] .* invrt2
    extract_u_box = [zeros(nu, 1) Matrix(1.0I, nu, nu)] .* invrt2

    for i in 1:N
        push!(row_ids, row_init(i))
        push!(col_ids, col_x(i, 1))
        push!(blocks, Matrix(1.0I, nx, nx))

        for t in 1:T_steps-1
            push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t)); push!(blocks, -A_dyn)
            push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t + 1)); push!(blocks, Matrix(1.0I, nx, nx))
            push!(row_ids, row_dyn(i, t)); push!(col_ids, col_ζ(i, t)); push!(blocks, -B_on_ζ)

            push!(row_ids, row_boxp(i, t)); push!(col_ids, col_ζ(i, t)); push!(blocks, extract_u_box)
            push!(row_ids, row_boxp(i, t)); push!(col_ids, col_sp(i, t)); push!(blocks, Matrix(1.0I, nu, nu))

            push!(row_ids, row_boxm(i, t)); push!(col_ids, col_ζ(i, t)); push!(blocks, -extract_u_box)
            push!(row_ids, row_boxm(i, t)); push!(col_ids, col_sm(i, t)); push!(blocks, Matrix(1.0I, nu, nu))
        end
    end

    for (e, (i, j)) in enumerate(edges)
        push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T_steps)); push!(blocks, -P_proj)
        push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T_steps)); push!(blocks, P_proj)
    end

    B = blocksparse(row_ids, col_ids, blocks)

    c = zeros(size(B, 2))
    for i in 1:N, t in 1:T_steps-1
        ζ_range = colrange(B, col_ζ(i, t))
        c[ζ_range[1]] = invrt2
    end

    g = zeros(size(B, 1))
    for i in 1:N
        g[SheafSDP.rowrange(B, row_init(i))] .= x0[i]
        for t in 1:T_steps-1
            g[SheafSDP.rowrange(B, row_boxp(i, t))] .= ū
            g[SheafSDP.rowrange(B, row_boxm(i, t))] .= ū
        end
    end

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0)

    nv = N * blocks_per_agent
    cones = Vector{SheafSDP.Cone}(undef, nv)
    for i in 1:N
        for t in 1:T_steps
            cones[col_x(i, t)] = SheafSDP.CofreeCone()
        end
        for t in 1:T_steps-1
            cones[col_ζ(i, t)] = SheafSDP.SecondOrderCone()
            cones[col_sp(i, t)] = SheafSDP.PositiveCone()
            cones[col_sm(i, t)] = SheafSDP.PositiveCone()
        end
    end

    return IPMProblem(c, g, B, Q, cones), "SOC_$(topology)_N$(N)_T$(T_steps)"
end

#=============================================================================
   Analysis
=============================================================================#

function analyze_trajectory(step_data, name, k_subspace)
    println("\n" * "="^80)
    println("T2/T3: $name (bottom-$k_subspace subspace)")
    println("="^80)

    @printf("\n%5s │ %12s │ %12s │ %12s │ %12s │ %5s\n",
            "Step", "Gap μ", "μ_min(S₀)", "align_first", "align_prev", "KKT")
    println("─"^70)

    for d in step_data
        @printf("%5d │ %12.4e │ %12.4e │ %12.4f │ %12.4f │ %5d\n",
                d.iteration, d.gap, d.μ_min_S0, d.subspace_angle_k5, d.subspace_angle_prev, d.kkt_iters)
    end

    # Filter valid data
    valid = filter(d -> isfinite(d.gap) && d.gap > 0 && isfinite(d.μ_min_S0) && d.μ_min_S0 > 0, step_data)

    if length(valid) < 4
        println("\n  [Not enough valid data for analysis]")
        return NaN, NaN, "INCONCLUSIVE"
    end

    n = length(valid)

    # T2 analysis: μ_min vs gap slope
    log_gaps = [log10(d.gap) for d in valid]
    log_μ = [log10(d.μ_min_S0) for d in valid]

    x_mean = sum(log_gaps) / n
    y_mean = sum(log_μ) / n
    slope = sum((log_gaps .- x_mean) .* (log_μ .- y_mean)) / sum((log_gaps .- x_mean).^2)

    first_half = valid[1:n÷2]
    last_half = valid[n÷2+1:end]
    μ_early = sum(d.μ_min_S0 for d in first_half) / length(first_half)
    μ_late = sum(d.μ_min_S0 for d in last_half) / length(last_half)
    μ_ratio = μ_late / μ_early

    # T3 analysis: subspace drift
    drift_data = filter(d -> isfinite(d.subspace_angle_k5), valid[2:end])  # Skip first
    avg_align_first = if !isempty(drift_data)
        sum(d.subspace_angle_k5 for d in drift_data) / length(drift_data)
    else
        NaN
    end
    avg_align_prev = if !isempty(drift_data)
        sum(d.subspace_angle_prev for d in drift_data) / length(drift_data)
    else
        NaN
    end

    println("\n" * "-"^70)
    println("T2 ANALYSIS (μ_min vs gap):")
    println("-"^70)
    @printf("  Log-log slope (∂log μ_min / ∂log gap): %.4f\n", slope)
    @printf("  μ_min ratio (late/early): %.4f\n", μ_ratio)
    @printf("  Early avg μ_min: %.4e, Late avg μ_min: %.4e\n", μ_early, μ_late)

    println("\n" * "-"^70)
    println("T3 ANALYSIS (bottom-$k_subspace subspace drift):")
    println("-"^70)
    @printf("  Avg alignment with step 1 subspace: %.4f\n", avg_align_first)
    @printf("  Avg step-to-step alignment: %.4f\n", avg_align_prev)

    # Diagnosis
    println("\n" * "-"^70)

    # T2 diagnosis
    if slope > 0.2
        t2_diag = "BARRIER"
        println("T2 → BARRIER: μ_min DECREASES as gap → 0 (slope > 0)")
        println("     The Schur complement conditioning DEGRADES near optimality.")
    elseif slope < -0.2 || μ_ratio > 3.0
        t2_diag = "BENIGN"
        println("T2 → BENIGN: μ_min INCREASES as gap → 0 (slope < 0)")
        println("     The barrier helps conditioning (common for LP).")
    else
        t2_diag = "STRUCTURAL"
        println("T2 → STRUCTURAL: μ_min roughly CONSTANT (slope ≈ 0)")
        println("     The conditioning floor is from L = B'B.")
    end

    # T3 diagnosis
    if avg_align_first > 0.8
        t3_diag = "STABLE"
        println("T3 → STABLE: Bottom-$k_subspace subspace stays aligned with step 1 (>0.8)")
        println("     Deflate-once would work.")
    elseif avg_align_first < 0.5
        t3_diag = "DRIFTING"
        println("T3 → DRIFTING: Bottom-$k_subspace subspace drifts significantly (<0.5)")
        println("     Need recycling, not static deflation.")
    else
        t3_diag = "MIXED"
        println("T3 → MIXED: Moderate subspace drift (0.5-0.8)")
        println("     May need combination of deflation + recycling.")
    end

    # Combined diagnosis
    println("\n" * "-"^70)
    println("COMBINED DIAGNOSIS:")
    println("-"^70)
    if t2_diag == "BARRIER" && t3_diag == "DRIFTING"
        println("  → BARRIER-DRIVEN: Use Ritz recycling + A-dependent coarse space")
    elseif t2_diag == "STRUCTURAL" && t3_diag == "STABLE"
        println("  → STRUCTURAL: Use fixed deflation of bottom-k eigenvectors of L")
    elseif t2_diag == "BENIGN"
        println("  → BENIGN: No preconditioning needed for this cone type")
    else
        println("  → MIXED: Consider combination of deflation + recycling")
    end

    return slope, avg_align_first, "$t2_diag/$t3_diag"
end

function main()
    println("="^80)
    println("T2/T3: SDP/SOC DIAGNOSTICS")
    println("="^80)
    println()
    println("Testing the HARD cones where barrier can DEGENERATE:")
    println("  - SDP: barrier Hessian A = X⁻¹ ⊗ X⁻¹ has anisotropic structure")
    println("  - SOC: similar behavior near cone apex")
    println()
    println("Key improvement: tracking bottom-k SUBSPACE angle, not single eigenvector")
    println("(single eigenvector rotates freely in degenerate eigenvalue clusters)")
    println()

    k_subspace = 5
    results = []

    # SDP Chain (dissipativity-style)
    println("\n" * "="^80)
    println("SDP Chain: Passivity LMI on path graph")
    println("="^80)
    prob, name = build_sdp_chain_problem(5, 4; topology=:chain)
    data, _ = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=40, k_subspace=k_subspace)
    slope, align, diag = analyze_trajectory(data, name, k_subspace)
    push!(results, (name=name, slope=slope, align=align, diagnosis=diag))

    # SDP Complete (well-connected)
    println("\n" * "="^80)
    println("SDP Complete: Passivity LMI on complete graph")
    println("="^80)
    prob, name = build_sdp_chain_problem(5, 4; topology=:complete)
    data, _ = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=40, k_subspace=k_subspace)
    slope, align, diag = analyze_trajectory(data, name, k_subspace)
    push!(results, (name=name, slope=slope, align=align, diagnosis=diag))

    # SOC Chain
    println("\n" * "="^80)
    println("SOC Chain: Group-sparse control on chain")
    println("="^80)
    prob, name = build_soc_problem(6, 8; topology=:chain)
    data, _ = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=40, k_subspace=k_subspace)
    slope, align, diag = analyze_trajectory(data, name, k_subspace)
    push!(results, (name=name, slope=slope, align=align, diagnosis=diag))

    # SOC Complete
    println("\n" * "="^80)
    println("SOC Complete: Group-sparse control on complete graph")
    println("="^80)
    prob, name = build_soc_problem(5, 8; topology=:complete)
    data, _ = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=40, k_subspace=k_subspace)
    slope, align, diag = analyze_trajectory(data, name, k_subspace)
    push!(results, (name=name, slope=slope, align=align, diagnosis=diag))

    # Summary
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    @printf("\n%-30s │ %8s │ %8s │ %20s\n", "Instance", "Slope", "Align_k5", "Diagnosis")
    println("─"^75)
    for r in results
        @printf("%-30s │ %8.4f │ %8.4f │ %20s\n", r.name, r.slope, r.align, r.diagnosis)
    end

    println("\n" * "-"^75)
    println("KEY:")
    println("  Slope > 0  → μ_min degrades (BARRIER)")
    println("  Slope ≈ 0  → μ_min constant (STRUCTURAL)")
    println("  Slope < 0  → μ_min improves (BENIGN)")
    println("  Align > 0.8 → subspace stable (deflate-once works)")
    println("  Align < 0.5 → subspace drifts (need recycling)")

    println("\n" * "="^80)
    println("T2/T3 SDP/SOC COMPLETE")
    println("="^80)
end

main()
