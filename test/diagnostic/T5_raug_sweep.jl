#
# T5 - raug Sweep at Fixed Iterates
#
# Purpose: Test how the augmentation parameter α = raug * ||A|| / ||L||
# affects conditioning and CG iterations.
#
# Smaller α → less augmentation → S(α) closer to S₀ → worse conditioning
# Larger α → more augmentation → S(α) closer to αL → better conditioning but less accurate
#

using SheafSDP
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block

struct T5Result
    raug::Float64
    α::Float64              # actual augmentation parameter
    kkt_iters::Int          # total KKT iterations
    ipm_iters::Int          # total IPM iterations
    time::Float64           # solve time
    gap_final::Float64      # final duality gap
    status::Any
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

function run_with_raug(prob, raug)
    T = Float64
    kkt = UzawaSettings{T}(raug=raug)
    settings = IPMSettings{T}(kkt=kkt, verbose=false, itmax=100)

    t = @elapsed result = solve(prob, settings)

    # Compute final gap from primal/dual
    gap = dot(result.p, result.d)

    # Compute α from first step (approximation)
    n = size(prob.B, 2)
    α_est = raug * 1.0 / n  # rough estimate

    return T5Result(
        raug, α_est,
        result.kkt_iters, result.iterations,
        t, gap, result.status
    )
end

function run_sweep(name, prob, raugs)
    println("\n" * "="^80)
    println("T5: raug Sweep - $name")
    println("="^80)

    results = T5Result[]

    @printf("\n%12s │ %8s │ %8s │ %10s │ %12s │ %10s\n",
            "raug", "KKT", "IPM", "Time (s)", "Final Gap", "Status")
    println("─"^75)

    for raug in raugs
        r = run_with_raug(prob, raug)
        push!(results, r)

        @printf("%12.2e │ %8d │ %8d │ %10.4f │ %12.4e │ %10s\n",
                r.raug, r.kkt_iters, r.ipm_iters, r.time, r.gap_final, r.status)
    end

    # Analysis
    println("\n" * "-"^80)
    println("ANALYSIS:")
    println("-"^80)

    # Find optimal raug
    successful = filter(r -> r.status == SheafSDP.OPTIMAL, results)
    if !isempty(successful)
        best = argmin(r -> r.kkt_iters, successful)
        @printf("  Optimal raug: %.2e (KKT iters = %d)\n", best.raug, best.kkt_iters)

        # Compare extremes
        small_raug = filter(r -> r.raug <= 1e3, successful)
        large_raug = filter(r -> r.raug >= 1e6, successful)

        if !isempty(small_raug) && !isempty(large_raug)
            avg_small = sum(r.kkt_iters for r in small_raug) / length(small_raug)
            avg_large = sum(r.kkt_iters for r in large_raug) / length(large_raug)

            @printf("  Average KKT iters (raug ≤ 1e3): %.1f\n", avg_small)
            @printf("  Average KKT iters (raug ≥ 1e6): %.1f\n", avg_large)

            if avg_small < avg_large * 0.8
                println("\n  → Small raug is better: less augmentation works")
            elseif avg_large < avg_small * 0.8
                println("\n  → Large raug is better: more augmentation needed")
            else
                println("\n  → Moderate dependence on raug")
            end
        end
    else
        println("  No successful solves!")
    end

    return results
end

function main()
    println("="^80)
    println("T5: raug SWEEP AT FIXED ITERATES")
    println("="^80)
    println()
    println("Testing how the augmentation parameter raug affects")
    println("conditioning and CG iterations.")
    println()
    println("α = raug × ||A|| / ||B'B||")
    println("  - Small raug → less regularization → S(α) ≈ S₀")
    println("  - Large raug → more regularization → S(α) better conditioned")

    raugs = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

    # Test 1: Complete graph
    prob = build_lp_problem(8, 4, 2; topology=:complete)
    run_sweep("LP K_8", prob, raugs)

    # Test 2: Chain graph
    prob = build_lp_problem(15, 4, 2; topology=:chain)
    run_sweep("LP Chain N=15", prob, raugs)

    # Test 3: Grid
    prob = build_lp_problem(16, 4, 2; topology=:grid)
    run_sweep("LP Grid 4×4", prob, raugs)

    println("\n" * "="^80)
    println("T5 COMPLETE")
    println("="^80)
end

main()
