#
# T7 - Controlled Synthetic Sheaves (Mechanism Isolation)
#
# Purpose: Separate structural vs barrier-driven mechanisms by construction,
# providing ground truth for behavioral tests.
#
# Two synthetic instances with same cones / same IPM, differing only in graph:
#
# 1. Small-gap sheaf: Two clusters joined by weak edge → tiny λ₁⁺(L) → structural (A)
# 2. Expander sheaf: Well-connected graph → large λ₁⁺(L) → any trouble is barrier (B)
#
# Run T2/T4 style diagnostics on both to confirm the readings.
#

using SheafSDP
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block

struct T7StepData
    iteration::Int
    gap::Float64
    μ_min_S0::Float64
    μ_min_L::Float64  # for reference
    kkt_iters::Int
end

struct T7Result
    name::String
    λ1_L::Float64                 # spectral gap of L
    dim_kernel::Int               # dimension of kernel H⁰
    step_data::Vector{T7StepData}
    diagnosis::String             # STRUCTURAL, BARRIER, or MIXED
end

# Build small-gap sheaf: two clusters joined by a weak edge
# Cluster 1: vertices 1..N1, complete graph
# Cluster 2: vertices N1+1..N1+N2, complete graph
# Bridge: single weak edge between vertex N1 and N1+1
function build_small_gap_sheaf(N1, N2, d_v, d_e; bridge_scale=0.01, seed=42)
    Random.seed!(seed)
    N = N1 + N2

    src, tgt, maps = Int[], Int[], Matrix{Float64}[]

    # Cluster 1: complete graph on vertices 1..N1
    for i in 1:N1, j in i+1:N1
        push!(src, i); push!(tgt, j); push!(maps, randn(d_e, d_v))
        push!(src, j); push!(tgt, i); push!(maps, randn(d_e, d_v))
    end

    # Cluster 2: complete graph on vertices N1+1..N
    for i in N1+1:N, j in i+1:N
        push!(src, i); push!(tgt, j); push!(maps, randn(d_e, d_v))
        push!(src, j); push!(tgt, i); push!(maps, randn(d_e, d_v))
    end

    # Weak bridge: scale down the restriction maps
    push!(src, N1); push!(tgt, N1+1); push!(maps, bridge_scale * randn(d_e, d_v))
    push!(src, N1+1); push!(tgt, N1); push!(maps, bridge_scale * randn(d_e, d_v))

    B = SheafSDP.sheaf(src, tgt, maps)

    c = abs.(randn(size(B, 2))) .+ 0.1
    x_bar = ones(size(B, 2)) .+ abs.(randn(size(B, 2)))
    g = B * x_bar

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0.0)

    cones = [PositiveCone() for _ in 1:N]
    return IPMProblem(c, g, B, Q, cones), "SmallGap(N1=$N1,N2=$N2,scale=$bridge_scale)"
end

# Build expander sheaf: random regular graph with good expansion
# Use a random d-regular graph construction for large spectral gap
function build_expander_sheaf(N, d_v, d_e, degree; seed=42)
    Random.seed!(seed)

    src, tgt, maps = Int[], Int[], Matrix{Float64}[]

    # Simple approach: add random edges until each vertex has ~degree edges
    # For a proper expander, we use a complete graph on small N
    # or Paley/Ramanujan construction idea: add edges with good mixing

    # For simplicity, use complete graph (perfect expander for small N)
    for i in 1:N, j in i+1:N
        push!(src, i); push!(tgt, j); push!(maps, randn(d_e, d_v))
        push!(src, j); push!(tgt, i); push!(maps, randn(d_e, d_v))
    end

    B = SheafSDP.sheaf(src, tgt, maps)

    c = abs.(randn(size(B, 2))) .+ 0.1
    x_bar = ones(size(B, 2)) .+ abs.(randn(size(B, 2)))
    g = B * x_bar

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0.0)

    cones = [PositiveCone() for _ in 1:N]
    return IPMProblem(c, g, B, Q, cones), "Expander(N=$N,complete)"
end

# Build chain sheaf: long path, intermediate spectral gap
function build_chain_sheaf(N, d_v, d_e; seed=42)
    Random.seed!(seed)

    src, tgt, maps = Int[], Int[], Matrix{Float64}[]

    for i in 1:N-1
        push!(src, i); push!(tgt, i+1); push!(maps, randn(d_e, d_v))
        push!(src, i+1); push!(tgt, i); push!(maps, randn(d_e, d_v))
    end

    B = SheafSDP.sheaf(src, tgt, maps)

    c = abs.(randn(size(B, 2))) .+ 0.1
    x_bar = ones(size(B, 2)) .+ abs.(randn(size(B, 2)))
    g = B * x_bar

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0.0)

    cones = [PositiveCone() for _ in 1:N]
    return IPMProblem(c, g, B, Q, cones), "Chain(N=$N)"
end

# Analyze sheaf Laplacian spectrum
function analyze_laplacian(B)
    Bdense = Matrix(B)
    L = Bdense' * Bdense
    L = Symmetric(L)

    λ = eigvals(L)
    sort!(λ)

    # Find kernel dimension and spectral gap
    tol = 1e-10 * maximum(abs, λ)
    kernel_idx = findall(x -> abs(x) <= tol, λ)
    dim_kernel = length(kernel_idx)

    nonzero = filter(x -> x > tol, λ)
    λ1 = isempty(nonzero) ? NaN : minimum(nonzero)
    λ_max = isempty(nonzero) ? NaN : maximum(nonzero)

    return λ1, λ_max, dim_kernel, λ
end

function compute_schur_min_eigenvalue(B, A; tol=1e-10)
    m, n = size(B)
    N = SheafSDP.nvtxs(B)

    Bdense = Matrix(B)

    # Build block-diagonal A
    Adense = zeros(n, n)
    for v in 1:N
        rng = SheafSDP.colrange(B, v)
        Av = Matrix(block(A, v, v, v))
        if any(isnan, Av) || any(isinf, Av)
            return NaN
        end
        Adense[rng, rng] .= Av
    end

    eigA = eigvals(Symmetric(Adense))
    if minimum(eigA) <= 0
        return NaN
    end

    try
        # S₀ = B A⁻¹ B'
        S0 = Bdense * (Adense \ Bdense')
        S0 = Symmetric((S0 + S0') / 2)

        μ = eigvals(S0)
        sort!(μ)

        # Find smallest nonzero eigenvalue
        max_μ = maximum(abs, μ)
        tol_μ = max(tol, 1e-10 * max_μ)
        pos = filter(x -> x > tol_μ, μ)

        return isempty(pos) ? NaN : minimum(pos)
    catch e
        return NaN
    end
end

function run_instrumented_ipm(prob; raug=1e6, verbose=false, max_iters=50)
    T = Float64
    kkt = UzawaSettings{T}(raug=raug)
    settings = IPMSettings{T}(kkt=kkt, verbose=verbose, itmax=max_iters)

    solver = SheafSDP.init(prob, settings)

    step_data = T7StepData[]
    B = solver.B

    # Get baseline L eigenvalue
    Bdense = Matrix(B)
    L = Bdense' * Bdense
    λ_L = eigvals(Symmetric(L))
    sort!(λ_L)
    tol = 1e-10 * maximum(abs, λ_L)
    nonzero_L = filter(x -> x > tol, λ_L)
    μ_min_L = isempty(nonzero_L) ? NaN : minimum(nonzero_L)

    iteration = 0
    while true
        iteration += 1

        ok = SheafSDP.step!(solver)

        μ_gap = dot(solver.p, solver.d) / solver.ν
        kkt_iters = solver.kkt_iters

        # Compute minimum eigenvalue of Schur complement
        μ_min_S0 = try
            compute_schur_min_eigenvalue(B, solver.H)
        catch
            NaN
        end

        push!(step_data, T7StepData(iteration, μ_gap, μ_min_S0, μ_min_L, kkt_iters))

        if verbose
            @printf("Step %3d: gap=%.2e, μ_min(S₀)=%.2e, μ_min(L)=%.2e, KKT=%d\n",
                    iteration, μ_gap, μ_min_S0, μ_min_L, kkt_iters)
        end

        if !ok || iteration >= max_iters
            break
        end
    end

    return step_data
end

function diagnose_behavior(step_data, λ1_L)
    # Filter valid data
    valid = filter(d -> isfinite(d.gap) && d.gap > 0 && isfinite(d.μ_min_S0) && d.μ_min_S0 > 0, step_data)

    if length(valid) < 5
        return "INCONCLUSIVE", NaN, NaN
    end

    n = length(valid)
    first_half = valid[1:n÷2]
    last_half = valid[n÷2+1:end]

    μ_early = sum(d.μ_min_S0 for d in first_half) / length(first_half)
    μ_late = sum(d.μ_min_S0 for d in last_half) / length(last_half)

    # Compute log-log slope: ∂log(μ_min) / ∂log(gap)
    # If slope > 0: μ_min decreases as gap decreases (BAD - barrier-driven)
    # If slope < 0: μ_min increases as gap decreases (GOOD - benign)
    # If slope ≈ 0: μ_min constant (structural floor)
    log_gaps = [log10(d.gap) for d in valid]
    log_μ = [log10(d.μ_min_S0) for d in valid]

    x_mean = sum(log_gaps) / length(log_gaps)
    y_mean = sum(log_μ) / length(log_μ)
    slope = sum((log_gaps .- x_mean) .* (log_μ .- y_mean)) /
            sum((log_gaps .- x_mean).^2)

    ratio = μ_late / μ_early

    # Diagnose based on slope and ratio
    # slope > 0.3: μ_min decreases with gap → BARRIER-DRIVEN degradation
    # slope ≈ 0: μ_min constant → STRUCTURAL floor (bad modes from L)
    # slope < -0.3: μ_min increases as gap decreases → BENIGN (barrier helps!)
    if slope > 0.3
        return "BARRIER", slope, ratio
    elseif abs(slope) < 0.2 && ratio > 0.3 && ratio < 3.0
        return "STRUCTURAL", slope, ratio
    elseif slope < -0.3 || ratio > 5.0
        return "BENIGN", slope, ratio
    else
        return "MIXED", slope, ratio
    end
end

function analyze_instance(prob, name; raug=1e6, verbose=true)
    println("\n" * "="^80)
    println("T7: $name")
    println("="^80)

    B = prob.B

    # Analyze Laplacian spectrum
    λ1_L, λ_max_L, dim_kernel, λ_all = analyze_laplacian(B)

    println("\nSheaf Laplacian L = B'B spectrum:")
    @printf("  dim(kernel H⁰) = %d\n", dim_kernel)
    @printf("  λ₁⁺ (spectral gap) = %.6e\n", λ1_L)
    @printf("  λ_max = %.6e\n", λ_max_L)
    if isfinite(λ1_L) && isfinite(λ_max_L) && λ1_L > 0
        @printf("  κ(L|range) = %.2e\n", λ_max_L / λ1_L)
    end

    # Show bottom eigenvalues
    println("\n  Bottom 10 eigenvalues of L:")
    for (i, λ) in enumerate(λ_all[1:min(10, length(λ_all))])
        @printf("    λ_%d = %.6e\n", i, λ)
    end

    # Run instrumented IPM
    println("\nRunning IPM...")
    step_data = run_instrumented_ipm(prob; raug=raug, verbose=verbose)

    # Analyze trajectory
    println("\n" * "-"^80)
    println("IPM Trajectory:")
    println("-"^80)
    @printf("%5s │ %12s │ %12s │ %5s\n", "Step", "Gap μ", "μ_min(S₀)", "KKT")
    println("─"^45)

    for d in step_data
        @printf("%5d │ %12.4e │ %12.4e │ %5d\n",
                d.iteration, d.gap, d.μ_min_S0, d.kkt_iters)
    end

    # Diagnosis
    diagnosis, slope, ratio = diagnose_behavior(step_data, λ1_L)

    println("\n" * "-"^80)
    println("DIAGNOSIS: $diagnosis")
    println("-"^80)
    @printf("  Log-log slope (∂log μ_min / ∂log gap): %.3f\n", slope)
    @printf("  Ratio μ_late/μ_early: %.3f\n", ratio)
    println()

    if diagnosis == "STRUCTURAL"
        println("  μ_min(S₀) stays roughly constant across the run.")
        println("  The conditioning floor comes from the small spectral gap λ₁⁺(L).")
        println("  Cure: Deflate bottom-k eigenvectors of L once.")
    elseif diagnosis == "BARRIER"
        println("  μ_min(S₀) decreases as the duality gap → 0.")
        println("  The conditioning degrades as A approaches the cone boundary.")
        println("  Cure: Recycle Ritz vectors or use A-dependent coarse space.")
    elseif diagnosis == "BENIGN"
        println("  μ_min(S₀) INCREASES as the duality gap → 0.")
        println("  The barrier Hessian A = diag(1/x²) helps conditioning.")
        println("  This is good! No preconditioning cure needed for this case.")
        println("  (Common for LP/PositiveCone where A grows as x → boundary)")
    else
        println("  Mixed behavior: both structural and barrier components present.")
        println("  Cure: Combine fixed deflation with recycling.")
    end

    return T7Result(name, λ1_L, dim_kernel, step_data, diagnosis)
end

function main()
    println("="^80)
    println("T7: CONTROLLED SYNTHETIC SHEAVES")
    println("="^80)
    println()
    println("Creating synthetic instances with known spectral properties")
    println("to validate the diagnostic framework.")
    println()
    println("Instance 1: Small-gap sheaf (two clusters + weak bridge)")
    println("            → tiny λ₁⁺(L) → should be STRUCTURAL")
    println()
    println("Instance 2: Expander sheaf (well-connected complete graph)")
    println("            → large λ₁⁺(L) → any trouble should be BARRIER")
    println()
    println("Instance 3: Chain sheaf (intermediate case)")
    println("            → moderate λ₁⁺(L) → may show MIXED behavior")

    results = T7Result[]

    # Instance 1: Small-gap sheaf
    prob, name = build_small_gap_sheaf(5, 5, 4, 2; bridge_scale=0.01)
    r1 = analyze_instance(prob, name; raug=1e6, verbose=true)
    push!(results, r1)

    # Instance 2: Expander sheaf
    prob, name = build_expander_sheaf(8, 4, 2, 4)
    r2 = analyze_instance(prob, name; raug=1e6, verbose=true)
    push!(results, r2)

    # Instance 3: Chain sheaf
    prob, name = build_chain_sheaf(12, 4, 2)
    r3 = analyze_instance(prob, name; raug=1e6, verbose=true)
    push!(results, r3)

    # Summary table
    println("\n" * "="^80)
    println("T7 SUMMARY")
    println("="^80)
    @printf("\n%-30s │ %12s │ %8s │ %12s\n",
            "Instance", "λ₁⁺(L)", "dim(H⁰)", "Diagnosis")
    println("─"^70)

    for r in results
        @printf("%-30s │ %12.4e │ %8d │ %12s\n",
                r.name, r.λ1_L, r.dim_kernel, r.diagnosis)
    end

    println("\n" * "-"^80)
    println("INTERPRETATION:")
    println("-"^80)
    println()
    println("For LP with PositiveCone, the barrier Hessian A = diag(1/x²)")
    println("tends to IMPROVE conditioning as x → boundary (A grows large).")
    println("This leads to BENIGN behavior where μ_min(S₀) increases with convergence.")
    println()
    println("The STRUCTURAL/BARRIER distinction is more relevant for:")
    println("  - SDP (PSDCone) where A has complex structure")
    println("  - SOC (SecondOrderCone) near the cone apex")
    println("  - Problems where A degenerates instead of growing")
    println()
    println("Key diagnostic indicators:")
    println("  - slope > 0  → BARRIER (μ_min degrades with convergence)")
    println("  - slope ≈ 0  → STRUCTURAL (fixed conditioning floor)")
    println("  - slope < 0  → BENIGN (conditioning improves with convergence)")
    println()
    println("For real instances, run T2 to see the actual slope behavior.")

    println("\n" * "="^80)
    println("T7 COMPLETE")
    println("="^80)

    return results
end

main()
