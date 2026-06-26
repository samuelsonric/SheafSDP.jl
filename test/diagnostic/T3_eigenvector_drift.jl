#
# T3 - Identity and Drift of Bad Eigenvector
#
# Purpose: Track whether the problematic (smallest) eigenvector of the
# Schur complement stays fixed across IPM iterations or drifts.
#
# Signature:
# - High alignment (>0.9) across steps → (A) STRUCTURAL
#   The bad mode is tied to L = B'B and doesn't change.
# - Alignment decays or is low → (B) BARRIER-DRIVEN
#   The bad mode depends on A and drifts each step.
#

using SheafSDP
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block

struct T3StepData
    iteration::Int
    gap::Float64
    μ_min_S0::Float64          # smallest nonzero eigenvalue of S₀
    v_alignment::Float64       # alignment with step 1 eigenvector
    v_alignment_prev::Float64  # alignment with previous step eigenvector
end

function compute_schur_bottom_eigenvector(B, A; tol=1e-10)
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

    eigA = eigvals(Symmetric(Adense))
    if minimum(eigA) <= 0
        return NaN, nothing
    end

    try
        # S₀ = B A⁻¹ B'
        S0 = Bdense * (Adense \ Bdense')
        S0 = Symmetric((S0 + S0') / 2)

        # Full eigendecomposition
        F = eigen(S0)
        λ = F.values
        V = F.vectors

        # Find smallest positive eigenvalue and its eigenvector
        max_λ = maximum(abs, λ)
        tol_eig = max(tol, 1e-10 * max_λ)

        # Sort by eigenvalue magnitude
        perm = sortperm(λ)
        λ_sorted = λ[perm]
        V_sorted = V[:, perm]

        # Find first eigenvalue > tolerance
        idx = findfirst(x -> x > tol_eig, λ_sorted)
        if idx === nothing
            return NaN, nothing
        end

        μ_min = λ_sorted[idx]
        v_min = V_sorted[:, idx]

        return μ_min, v_min
    catch e
        return NaN, nothing
    end
end

function run_instrumented_ipm(prob; raug=1e6, verbose=false, max_iters=100)
    T = Float64
    kkt = UzawaSettings{T}(raug=raug)
    settings = IPMSettings{T}(kkt=kkt, verbose=verbose, itmax=max_iters)

    solver = SheafSDP.init(prob, settings)

    step_data = T3StepData[]
    B = solver.B

    # Store first step's eigenvector for comparison
    v_first = nothing
    v_prev = nothing

    iteration = 0
    while true
        iteration += 1

        ok = SheafSDP.step!(solver)

        μ_gap = dot(solver.p, solver.d) / solver.ν

        # Compute bottom eigenvector of S₀
        μ_min_S0, v_S0 = try
            compute_schur_bottom_eigenvector(B, solver.H)
        catch e
            (NaN, nothing)
        end

        # Compute alignments
        if v_first === nothing && v_S0 !== nothing
            v_first = copy(v_S0)
        end

        v_alignment = if v_first !== nothing && v_S0 !== nothing
            abs(dot(v_first, v_S0))
        else
            NaN
        end

        v_alignment_prev = if v_prev !== nothing && v_S0 !== nothing
            abs(dot(v_prev, v_S0))
        else
            1.0  # First step, trivially aligned with itself
        end

        push!(step_data, T3StepData(
            iteration, μ_gap, μ_min_S0, v_alignment, v_alignment_prev
        ))

        if v_S0 !== nothing
            v_prev = copy(v_S0)
        end

        if verbose
            @printf("Step %3d: gap=%.2e, μ_min(S₀)=%.2e, align_first=%.4f, align_prev=%.4f\n",
                    iteration, μ_gap, μ_min_S0, v_alignment, v_alignment_prev)
        end

        if !ok || iteration >= max_iters
            break
        end
    end

    return step_data, solver
end

# Build LP problem with PositiveCone
function build_lp_problem(N, d_v, d_e)
    Random.seed!(42)

    src, tgt, maps = Int[], Int[], Matrix{Float64}[]
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
    return IPMProblem(c, g, B, Q, cones)
end

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

function analyze_drift(step_data, name)
    println("\n" * "="^80)
    println("T3: Eigenvector Drift Analysis - $name")
    println("="^80)

    @printf("\n%5s │ %12s │ %12s │ %12s │ %12s\n",
            "Step", "Gap μ", "μ_min(S₀)", "align_first", "align_prev")
    println("─"^60)

    for d in step_data
        @printf("%5d │ %12.4e │ %12.4e │ %12.4f │ %12.4f\n",
                d.iteration, d.gap, d.μ_min_S0, d.v_alignment, d.v_alignment_prev)
    end

    # Analysis
    valid_data = filter(d -> isfinite(d.v_alignment) && isfinite(d.v_alignment_prev), step_data)

    if length(valid_data) >= 3
        avg_align_first = sum(d.v_alignment for d in valid_data) / length(valid_data)
        avg_align_prev = sum(d.v_alignment_prev for d in valid_data) / length(valid_data)

        # Compute alignment decay
        n = length(valid_data)
        first_half = valid_data[1:n÷2]
        last_half = valid_data[n÷2+1:end]

        align_early = sum(d.v_alignment for d in first_half) / length(first_half)
        align_late = sum(d.v_alignment for d in last_half) / length(last_half)

        println("\n" * "-"^60)
        println("ANALYSIS:")
        println("-"^60)
        @printf("  Average alignment with step 1: %.4f\n", avg_align_first)
        @printf("  Average step-to-step alignment: %.4f\n", avg_align_prev)
        @printf("  Alignment with step 1 (early steps 1-%d): %.4f\n", n÷2, align_early)
        @printf("  Alignment with step 1 (late steps %d-%d): %.4f\n", n÷2+1, n, align_late)

        println()
        if avg_align_first > 0.9 && align_late > 0.8
            println("  → DIAGNOSIS: (A) STRUCTURAL")
            println("    The bottom eigenvector stays nearly constant.")
            println("    It's tied to the sheaf Laplacian L = B'B.")
            println("    Cure: Deflate once at initialization.")
        elseif avg_align_first < 0.5 || align_late < 0.5
            println("  → DIAGNOSIS: (B) BARRIER-DRIVEN")
            println("    The bottom eigenvector drifts significantly.")
            println("    It depends on the changing Hessian A.")
            println("    Cure: Recycle Ritz vectors or A-dependent coarse space.")
        else
            println("  → DIAGNOSIS: MIXED or PARTIAL DRIFT")
            println("    Moderate alignment suggests some structural component")
            println("    with barrier-driven perturbations.")
        end

        # Step-to-step drift analysis
        if avg_align_prev > 0.99
            println("\n  Note: Very high step-to-step alignment (>0.99).")
            println("        Mode changes are negligible.")
        elseif avg_align_prev < 0.9
            println("\n  Note: Moderate step-to-step drift (<0.9).")
            println("        The bad mode is shifting each iteration.")
        end
    else
        println("\n  [Not enough valid data points for analysis]")
    end

    return step_data
end

function main()
    println("="^80)
    println("T3: EIGENVECTOR DRIFT ANALYSIS")
    println("="^80)
    println()
    println("Tracking whether the bottom eigenvector of S₀ = B A⁻¹ B'")
    println("stays constant (structural) or drifts (barrier-driven).")
    println()
    println("Key metrics:")
    println("  - align_first: alignment with step 1's eigenvector")
    println("  - align_prev: alignment with previous step's eigenvector")

    # Test 1: Complete graph
    println("\n" * "="^80)
    println("Instance: LP Complete Graph K_6, d_v=4, d_e=2")
    println("="^80)
    prob = build_lp_problem(6, 4, 2)
    data, _ = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=30)
    analyze_drift(data, "LP K_6")

    # Test 2: Chain graph
    println("\n" * "="^80)
    println("Instance: LP Chain N=10, d_v=4, d_e=2")
    println("="^80)
    prob = build_chain_lp_problem(10, 4, 2)
    data, _ = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=30)
    analyze_drift(data, "LP Chain N=10")

    println("\n" * "="^80)
    println("T3 COMPLETE")
    println("="^80)
end

main()
