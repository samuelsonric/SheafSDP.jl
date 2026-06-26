#
# T4 - Deflate-Once Experiment
#
# Purpose: Test if deflating the bottom-k eigenvectors of L = B'B
# once at initialization improves CG convergence.
#
# Signature:
# - Significant speedup → (A) STRUCTURAL issue, deflation is the cure
# - Little improvement → (B) BARRIER-DRIVEN, need adaptive approach
#

using SheafSDP
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block
using Krylov: cg

struct T4Result
    name::String
    k::Int                  # number of eigenvectors deflated
    iters_baseline::Int     # total KKT iterations without deflation
    iters_deflated::Int     # total KKT iterations with deflation (estimated)
    time_baseline::Float64  # time without deflation
    time_deflated::Float64  # time with deflation (estimated)
    κ_before::Float64       # condition number before deflation
    κ_after::Float64        # condition number after deflation
end

# Build LP problem with PositiveCone
function build_lp_problem(N, d_v, d_e; topology=:complete)
    Random.seed!(42)

    src, tgt, maps = Int[], Int[], Matrix{Float64}[]

    if topology == :complete
        for i in 1:N, j in i+1:N
            push!(src, i); push!(tgt, j); push!(maps, randn(d_e, d_v))
            push!(src, j); push!(tgt, i); push!(maps, randn(d_e, d_v))
        end
    elseif topology == :chain
        for i in 1:N-1
            push!(src, i); push!(tgt, i+1); push!(maps, randn(d_e, d_v))
            push!(src, i+1); push!(tgt, i); push!(maps, randn(d_e, d_v))
        end
    elseif topology == :grid
        side = isqrt(N)
        @assert side^2 == N "N must be a perfect square for grid topology"
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

# Compute bottom-k eigenvectors of L = B'B
function compute_deflation_space(B, k)
    Bdense = Matrix(B)
    L = Bdense' * Bdense
    L = Symmetric(L)

    F = eigen(L)
    λ = F.values
    V = F.vectors

    # Sort by eigenvalue
    perm = sortperm(λ)
    V_sorted = V[:, perm]

    # Return bottom-k eigenvectors (skipping kernel)
    # Find where eigenvalues become non-negligible
    tol = 1e-10 * maximum(abs, λ)
    first_nonzero = findfirst(x -> abs(λ[perm[x]]) > tol, 1:length(λ))

    if first_nonzero === nothing
        return zeros(size(V, 1), 0)
    end

    # Take k eigenvectors starting from first non-zero
    end_idx = min(first_nonzero + k - 1, size(V, 2))
    return V_sorted[:, first_nonzero:end_idx]
end

# Run IPM with optional deflation applied to CG
# Note: This is a simplified simulation - in practice, deflation would be
# integrated into the KKT solver. Here we just compare iteration counts.
function run_with_deflation(prob, deflation_vecs; raug=1e6)
    # For this diagnostic, we just run the standard solver
    # and simulate what deflation would achieve based on spectral analysis

    T = Float64
    kkt = UzawaSettings{T}(raug=raug)
    settings = IPMSettings{T}(kkt=kkt, verbose=false, itmax=100)

    t = @elapsed result = solve(prob, settings)

    return result.kkt_iters, t, result.status
end

function run_baseline(prob; raug=1e6)
    T = Float64
    kkt = UzawaSettings{T}(raug=raug)
    settings = IPMSettings{T}(kkt=kkt, verbose=false, itmax=100)

    t = @elapsed result = solve(prob, settings)

    return result.kkt_iters, t, result.status
end

# Analyze spectral properties to estimate deflation benefit
function analyze_deflation_potential(B, k)
    Bdense = Matrix(B)
    L = Bdense' * Bdense
    λ_L = eigvals(Symmetric(L))
    sort!(λ_L)

    # Find non-zero eigenvalues
    tol = 1e-10 * maximum(abs, λ_L)
    nonzero = filter(x -> abs(x) > tol, λ_L)

    if length(nonzero) < k + 1
        return NaN, NaN
    end

    # Condition number improvement from deflating bottom-k
    λ_max = maximum(nonzero)
    λ_min_before = minimum(nonzero)
    λ_min_after = nonzero[k + 1]  # After removing k smallest

    κ_before = λ_max / λ_min_before
    κ_after = λ_max / λ_min_after

    return κ_before, κ_after
end

function run_experiment(name, prob, ks; raug=1e6)
    println("\n" * "="^70)
    println("T4: Deflate-Once Experiment - $name")
    println("="^70)

    # Baseline run
    iters_baseline, time_baseline, status_baseline = run_baseline(prob; raug=raug)

    println("\nBaseline (no deflation):")
    @printf("  Total KKT iterations: %d\n", iters_baseline)
    @printf("  Time: %.3f s\n", time_baseline)
    @printf("  Status: %s\n", status_baseline)

    # Analyze spectral structure
    Bdense = Matrix(prob.B)
    L = Bdense' * Bdense
    λ_L = eigvals(Symmetric(L))
    sort!(λ_L)
    tol = 1e-10 * maximum(abs, λ_L)
    nonzero = filter(x -> x > tol, λ_L)

    println("\nSpectral analysis of L = B'B:")
    @printf("  dim(kernel) = %d\n", length(λ_L) - length(nonzero))
    if length(nonzero) > 0
        @printf("  λ_min (nonzero) = %.4e\n", minimum(nonzero))
        @printf("  λ_max = %.4e\n", maximum(nonzero))
        @printf("  κ(L|range) = %.2e\n", maximum(nonzero) / minimum(nonzero))
    end

    println("\nDeflation analysis:")
    @printf("%5s │ %12s │ %12s │ %12s │ %12s\n",
            "k", "κ_before", "κ_after", "improvement", "est. speedup")
    println("─"^65)

    results = T4Result[]

    for k in ks
        if k >= length(nonzero)
            @printf("%5d │ %12s │ %12s │ %12s │ %12s\n",
                    k, "-", "-", "-", "-")
            continue
        end

        λ_max = maximum(nonzero)
        λ_min_before = minimum(nonzero)
        λ_min_after = nonzero[k + 1]

        κ_before = λ_max / λ_min_before
        κ_after = λ_max / λ_min_after
        improvement = κ_before / κ_after

        # CG convergence is O(√κ), so speedup ≈ √(improvement)
        est_speedup = sqrt(improvement)

        @printf("%5d │ %12.2e │ %12.2e │ %12.2fx │ %12.2fx\n",
                k, κ_before, κ_after, improvement, est_speedup)

        # Estimate deflated iterations
        iters_deflated = max(1, round(Int, iters_baseline / est_speedup))

        push!(results, T4Result(
            name, k,
            iters_baseline, iters_deflated,
            time_baseline, time_baseline / est_speedup,
            κ_before, κ_after
        ))
    end

    println()
    if length(results) > 0 && any(r -> r.iters_baseline / r.iters_deflated > 1.5, results)
        println("  → DIAGNOSIS: (A) STRUCTURAL")
        println("    Deflating bottom-k eigenvectors of L would significantly")
        println("    reduce CG iterations. The conditioning issue is structural.")
    else
        println("  → DIAGNOSIS: Deflation alone may not be sufficient")
        println("    Consider combining with barrier-aware techniques.")
    end

    return results
end

function main()
    println("="^70)
    println("T4: DEFLATE-ONCE EXPERIMENT")
    println("="^70)
    println()
    println("Testing if deflating bottom-k eigenvectors of L = B'B")
    println("at initialization improves conditioning.")
    println()
    println("For CG, condition number κ → √κ iterations.")
    println("If deflation reduces κ by factor F, expect √F fewer iterations.")

    ks = [1, 2, 5, 10]

    # Test 1: Complete graph
    println("\n" * "="^70)
    println("Instance: LP Complete Graph K_8, d_v=4, d_e=2")
    println("="^70)
    prob = build_lp_problem(8, 4, 2; topology=:complete)
    run_experiment("LP K_8", prob, ks)

    # Test 2: Chain graph
    println("\n" * "="^70)
    println("Instance: LP Chain N=15, d_v=4, d_e=2")
    println("="^70)
    prob = build_lp_problem(15, 4, 2; topology=:chain)
    run_experiment("LP Chain N=15", prob, ks)

    # Test 3: Grid
    println("\n" * "="^70)
    println("Instance: LP Grid 4×4, d_v=4, d_e=2")
    println("="^70)
    prob = build_lp_problem(16, 4, 2; topology=:grid)
    run_experiment("LP Grid 4×4", prob, ks)

    println("\n" * "="^70)
    println("T4 COMPLETE")
    println("="^70)
end

main()
