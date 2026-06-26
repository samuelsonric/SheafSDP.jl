#
# T2 - μ_min(S(α)) vs Duality Gap Across IPM Run
#
# Purpose: The single most decisive test. Does the Schur conditioning
# collapse as the iterate approaches the boundary?
#
# Signature:
# - μ_min roughly constant → (A) STRUCTURAL
# - μ_min decreases as gap → 0 → (B) BARRIER-DRIVEN
#

using SheafSDP
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block, blocksparse, colrange, rowrange

struct T2StepData
    iteration::Int
    gap::Float64           # duality gap μ
    norm_A::Float64        # ||A|| (Hessian norm)
    α::Float64             # augmentation parameter
    cg_iters::Int          # CG iterations this step
    μ_min_S0::Float64      # smallest nonzero eigenvalue of S₀ = B A⁻¹ B'
    μ_min_Sα::Float64      # smallest nonzero eigenvalue of S(α) = B (A+αL)⁻¹ B'
end

function compute_schur_eigenvalues(B, A, α; tol=1e-10)
    m, n = size(B)
    N = SheafSDP.nvtxs(B)

    # Build dense matrices for small instances
    Bdense = Matrix(B)

    # Build block-diagonal A (the Hessian)
    Adense = zeros(n, n)
    for v in 1:N
        rng = SheafSDP.colrange(B, v)
        Av = Matrix(block(A, v, v, v))
        # Check for NaN/Inf
        if any(isnan, Av) || any(isinf, Av)
            return NaN, NaN
        end
        Adense[rng, rng] .= Av
    end

    # Build L = B'B
    L = Bdense' * Bdense

    # Check A is positive definite
    eigA = eigvals(Symmetric(Adense))
    min_eig = minimum(eigA)
    max_eig = maximum(eigA)
    if min_eig <= 0
        return NaN, NaN
    end

    # S₀ = B A⁻¹ B' (unaugmented)
    # Use regularized inverse for numerical stability
    try
        # Add small regularization if A is very ill-conditioned
        if max_eig / min_eig > 1e14
            # Very ill-conditioned: use pseudoinverse-like approach
            Adense_reg = Adense + 1e-12 * max_eig * I
            A_fac = cholesky(Symmetric(Adense_reg))
        else
            A_fac = cholesky(Symmetric(Adense))
        end
        # Use direct backslash for numerical stability
        # S0 = B A⁻¹ B' computed as B * (A \ B')
        S0 = Bdense * (Adense \ Bdense')
        S0 = Symmetric((S0 + S0') / 2)

        # S(α) = B (A + αL)⁻¹ B' (augmented)
        Aα = Adense + α * L
        Sα = Bdense * (Aα \ Bdense')
        Sα = Symmetric((Sα + Sα') / 2)

        # Compute eigenvalues
        μ_S0 = eigvals(S0)
        μ_Sα = eigvals(Sα)

        sort!(μ_S0)
        sort!(μ_Sα)

        # Find smallest nonzero eigenvalues (skip kernel)
        max_S0 = maximum(abs, μ_S0)
        max_Sα = maximum(abs, μ_Sα)
        tol_S0 = max(tol, 1e-10 * max_S0)
        tol_Sα = max(tol, 1e-10 * max_Sα)

        # Filter eigenvalues > tolerance
        pos_S0 = filter(x -> x > tol_S0, μ_S0)
        pos_Sα = filter(x -> x > tol_Sα, μ_Sα)

        μ_min_S0 = isempty(pos_S0) ? NaN : minimum(pos_S0)
        μ_min_Sα = isempty(pos_Sα) ? NaN : minimum(pos_Sα)

        return μ_min_S0, μ_min_Sα
    catch e
        debug && println("  [debug] Exception in Schur: $e")
        return NaN, NaN
    end
end

function run_instrumented_ipm(prob; raug=1e6, verbose=false, max_iters=100)
    T = Float64
    kkt = UzawaSettings{T}(raug=raug)
    settings = IPMSettings{T}(kkt=kkt, verbose=verbose, itmax=max_iters)

    # Initialize solver
    solver = SheafSDP.init(prob, settings)

    step_data = T2StepData[]
    B = solver.B

    iteration = 0
    while true
        iteration += 1

        # Take one IPM step first (this initializes H properly)
        ok = SheafSDP.step!(solver)

        # Get current state after step
        μ_gap = dot(solver.p, solver.d) / solver.ν
        norm_A = norm(Symmetric(solver.H, :L))
        α = solver.wrk.α[]

        # Compute Schur eigenvalues
        μ_min_S0, μ_min_Sα = try
            compute_schur_eigenvalues(B, solver.H, α)
        catch e
            (NaN, NaN)
        end

        # Get CG iterations from the step
        cg_iters = solver.kkt_iters

        # Record data
        push!(step_data, T2StepData(
            iteration, μ_gap, norm_A, α, cg_iters, μ_min_S0, μ_min_Sα
        ))

        if verbose
            @printf("Step %3d: gap=%.2e, ||A||=%.2e, α=%.2e, μ_min(S₀)=%.2e, μ_min(S(α))=%.2e, CG=%d\n",
                    iteration, μ_gap, norm_A, α, μ_min_S0, μ_min_Sα, cg_iters)
        end

        # Check termination: step! returns false when converged or failed
        if !ok || iteration >= max_iters
            break
        end
    end

    return step_data, solver
end

# Build LP problem with PositiveCone that is guaranteed feasible
# Structure: Complete graph K_N with edge-based consensus
# Variables: x_i ∈ ℝ^d_v with x_i ≥ 0 (element-wise)
# Constraints: F_{ij} x_i = F_{ji} x_j (interface agreement)
# Objective: min c'x for random c > 0
function build_lp_problem(N, d_v, d_e)
    Random.seed!(42)

    # Complete graph K_N
    src, tgt, maps = Int[], Int[], Matrix{Float64}[]
    for i in 1:N, j in i+1:N
        F_ij = randn(d_e, d_v)
        F_ji = randn(d_e, d_v)
        push!(src, i); push!(tgt, j); push!(maps, F_ij)
        push!(src, j); push!(tgt, i); push!(maps, F_ji)
    end

    B = SheafSDP.sheaf(src, tgt, maps)

    # Objective: random positive cost (bounded LP if x ≥ 0)
    c = abs.(randn(size(B, 2))) .+ 0.1

    # Feasible RHS: start with x̄ > 0, compute g = B x̄
    # This ensures primal feasibility
    x_bar = ones(size(B, 2)) .+ abs.(randn(size(B, 2)))
    g = B * x_bar

    # Q = 0 (no quadratic term)
    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0.0)

    # PositiveCone for all vertices
    cones = [PositiveCone() for _ in 1:N]

    return IPMProblem(c, g, B, Q, cones)
end

# Build LP problem on grid graph
function build_grid_lp_problem(side, d_v, d_e)
    Random.seed!(42)
    N = side^2

    src, tgt, maps = Int[], Int[], Matrix{Float64}[]
    for i in 1:side, j in 1:side
        v = (i-1)*side + j
        if j < side
            u = v + 1
            push!(src, v); push!(tgt, u); push!(maps, randn(d_e, d_v))
            push!(src, u); push!(tgt, v); push!(maps, randn(d_e, d_v))
        end
        if i < side
            u = v + side
            push!(src, v); push!(tgt, u); push!(maps, randn(d_e, d_v))
            push!(src, u); push!(tgt, v); push!(maps, randn(d_e, d_v))
        end
    end

    B = SheafSDP.sheaf(src, tgt, maps)

    c = abs.(randn(size(B, 2))) .+ 0.1
    x_bar = ones(size(B, 2)) .+ abs.(randn(size(B, 2)))
    g = B * x_bar

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0.0)

    cones = [PositiveCone() for _ in 1:N]
    return IPMProblem(c, g, B, Q, cones)
end

# Build LP problem on chain graph
function build_chain_lp_problem(N, d_v, d_e)
    Random.seed!(42)

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
    return IPMProblem(c, g, B, Q, cones)
end

function analyze_trajectory(step_data, name)
    println("\n" * "="^80)
    println("T2: μ_min vs Duality Gap - $name")
    println("="^80)

    # Print trajectory
    @printf("\n%5s │ %12s │ %12s │ %12s │ %12s │ %12s │ %4s\n",
            "Step", "Gap μ", "||A||", "α", "μ_min(S₀)", "μ_min(S(α))", "CG")
    println("─"^80)

    for d in step_data
        @printf("%5d │ %12.4e │ %12.4e │ %12.4e │ %12.4e │ %12.4e │ %4d\n",
                d.iteration, d.gap, d.norm_A, d.α, d.μ_min_S0, d.μ_min_Sα, d.cg_iters)
    end

    # Filter valid data points
    valid_data = filter(d -> isfinite(d.gap) && d.gap > 0 && isfinite(d.μ_min_S0) && d.μ_min_S0 > 0, step_data)

    if length(valid_data) >= 3
        n = length(valid_data)
        first_third = valid_data[1:n÷3]
        last_third = valid_data[2n÷3+1:end]

        μ_min_early = mean(d.μ_min_S0 for d in first_third)
        μ_min_late = mean(d.μ_min_S0 for d in last_third)

        gap_early = mean(d.gap for d in first_third)
        gap_late = mean(d.gap for d in last_third)

        # Compute log-log slope
        log_gaps = [log10(d.gap) for d in valid_data]
        log_μ = [log10(d.μ_min_S0) for d in valid_data]

        if length(log_gaps) >= 3
            n_pts = length(log_gaps)
            x_mean = sum(log_gaps) / n_pts
            y_mean = sum(log_μ) / n_pts
            slope = sum((log_gaps .- x_mean) .* (log_μ .- y_mean)) /
                    sum((log_gaps .- x_mean).^2)

            println("\n" * "-"^80)
            println("ANALYSIS:")
            println("-"^80)
            @printf("  Early iterations (steps 1-%d): avg μ_min(S₀) = %.4e, avg gap = %.4e\n",
                    n÷3, μ_min_early, gap_early)
            @printf("  Late iterations (steps %d-%d):  avg μ_min(S₀) = %.4e, avg gap = %.4e\n",
                    2n÷3+1, n, μ_min_late, gap_late)
            @printf("  Ratio (late/early): μ_min ratio = %.4f\n", μ_min_late / μ_min_early)
            @printf("  Log-log slope (∂log μ_min / ∂log gap): %.4f\n", slope)

            println()
            if abs(slope) < 0.1 && μ_min_late / μ_min_early > 0.5
                println("  → DIAGNOSIS: (A) STRUCTURAL")
                println("    μ_min is roughly constant across the run.")
                println("    The conditioning floor comes from the sheaf Laplacian.")
                println("    Cure: Deflate bottom-k eigenvectors of L once.")
            elseif slope > 0.3 || μ_min_late / μ_min_early < 0.1
                println("  → DIAGNOSIS: (B) BARRIER-DRIVEN")
                println("    μ_min decreases as gap → 0.")
                println("    Conditioning degrades as A approaches cone boundary.")
                println("    Cure: Recycle Ritz vectors, or A-dependent coarse space.")
            else
                println("  → DIAGNOSIS: MIXED or INCONCLUSIVE")
                println("    Some drift but not dramatic. May need both approaches.")
            end
        end
    else
        println("\n  [Not enough valid data points for analysis]")
    end

    return step_data
end

function mean(itr)
    s = 0.0
    n = 0
    for x in itr
        s += x
        n += 1
    end
    return n > 0 ? s / n : NaN
end

function main()
    println("="^80)
    println("T2: μ_min(S(α)) VS DUALITY GAP ANALYSIS")
    println("="^80)
    println()
    println("Tracking smallest eigenvalue of Schur complement across IPM iterations.")
    println("Key question: Does μ_min stay flat (structural) or collapse (barrier)?")
    println()
    println("Using PositiveCone (LP) problems which have proper barrier Hessians.")
    println("Barrier Hessian A = diag(1/x_i²) changes as x → boundary.")

    # Test 1: Small LP (complete graph)
    println("\n" * "="^80)
    println("Instance: LP Complete Graph K_6, d_v=4, d_e=2")
    println("="^80)
    prob = build_lp_problem(6, 4, 2)
    data, solver = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=50)
    analyze_trajectory(data, "LP K_6")

    # Test 2: Chain graph (simpler structure)
    println("\n" * "="^80)
    println("Instance: LP Chain N=10, d_v=4, d_e=2")
    println("="^80)
    prob = build_chain_lp_problem(10, 4, 2)
    data, solver = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=50)
    analyze_trajectory(data, "LP Chain N=10")

    # Test 3: Small grid
    println("\n" * "="^80)
    println("Instance: LP Grid 4×4, d_v=4, d_e=2")
    println("="^80)
    prob = build_grid_lp_problem(4, 4, 2)
    data, solver = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=50)
    analyze_trajectory(data, "LP Grid 4×4")

    println("\n" * "="^80)
    println("T2 COMPLETE")
    println("="^80)
end

main()
