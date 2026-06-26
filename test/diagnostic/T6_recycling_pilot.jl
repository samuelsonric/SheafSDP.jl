#
# T6 - Recycling Pilot
#
# Purpose: Test whether recycling Ritz vectors from previous CG solves
# would help convergence in subsequent solves.
#
# For barrier-driven conditioning, the slow modes change each step.
# If the change is gradual, recycling Ritz vectors could help.
#
# Metric: Alignment between consecutive steps' slow eigenvectors
# High alignment → recycling would help
# Low alignment → modes change too fast for recycling to help
#

using SheafSDP
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block

struct T6StepData
    iteration::Int
    gap::Float64
    kkt_iters::Int
    # Alignment of bottom-k eigenvectors with previous step
    align_k1::Float64   # alignment of bottom 1 eigenvector
    align_k5::Float64   # alignment of bottom 5 eigenvectors (subspace angle)
    align_k10::Float64  # alignment of bottom 10 eigenvectors
end

function compute_bottom_k_eigenvectors(B, A, k; tol=1e-10)
    m, n = size(B)
    N = SheafSDP.nvtxs(B)

    Bdense = Matrix(B)

    # Build block-diagonal A
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

        # Find non-kernel eigenvectors
        max_λ = maximum(abs, λ)
        tol_eig = max(tol, 1e-10 * max_λ)
        idx = findfirst(x -> x > tol_eig, λ_sorted)

        if idx === nothing
            return nothing
        end

        # Return bottom-k non-kernel eigenvectors
        end_idx = min(idx + k - 1, size(V, 2))
        return V_sorted[:, idx:end_idx]
    catch e
        return nothing
    end
end

# Compute principal angle between subspaces (cos of angle)
function subspace_alignment(V1, V2)
    if V1 === nothing || V2 === nothing
        return NaN
    end

    # Use SVD of V1'*V2 to get principal angles
    # cos(θ_min) = largest singular value
    k1, k2 = size(V1, 2), size(V2, 2)
    k_min = min(k1, k2)

    # Orthonormalize
    V1_orth, _ = qr(V1)
    V2_orth, _ = qr(V2)

    V1_orth = Matrix(V1_orth)[:, 1:min(k1, size(V1_orth, 2))]
    V2_orth = Matrix(V2_orth)[:, 1:min(k2, size(V2_orth, 2))]

    # Cosines of principal angles are singular values of V1'*V2
    σ = svdvals(V1_orth' * V2_orth)

    # Return smallest cosine (worst alignment)
    return minimum(σ)
end

function run_instrumented_ipm(prob; raug=1e6, verbose=false, max_iters=100)
    T = Float64
    kkt = UzawaSettings{T}(raug=raug)
    settings = IPMSettings{T}(kkt=kkt, verbose=verbose, itmax=max_iters)

    solver = SheafSDP.init(prob, settings)

    step_data = T6StepData[]
    B = solver.B

    V_prev_k1 = nothing
    V_prev_k5 = nothing
    V_prev_k10 = nothing

    iteration = 0
    while true
        iteration += 1

        ok = SheafSDP.step!(solver)

        μ_gap = dot(solver.p, solver.d) / solver.ν
        kkt_iters = solver.kkt_iters

        # Compute bottom-k eigenvectors
        V_k1 = compute_bottom_k_eigenvectors(B, solver.H, 1)
        V_k5 = compute_bottom_k_eigenvectors(B, solver.H, 5)
        V_k10 = compute_bottom_k_eigenvectors(B, solver.H, 10)

        # Compute alignments with previous step
        align_k1 = if V_prev_k1 !== nothing && V_k1 !== nothing
            abs(dot(V_prev_k1[:, 1], V_k1[:, 1]))
        else
            1.0
        end

        align_k5 = subspace_alignment(V_prev_k5, V_k5)
        align_k10 = subspace_alignment(V_prev_k10, V_k10)

        push!(step_data, T6StepData(
            iteration, μ_gap, kkt_iters, align_k1, align_k5, align_k10
        ))

        # Update previous
        V_prev_k1 = V_k1
        V_prev_k5 = V_k5
        V_prev_k10 = V_k10

        if verbose
            @printf("Step %3d: gap=%.2e, KKT=%d, align_k1=%.4f, align_k5=%.4f, align_k10=%.4f\n",
                    iteration, μ_gap, kkt_iters, align_k1, align_k5, align_k10)
        end

        if !ok || iteration >= max_iters
            break
        end
    end

    return step_data, solver
end

# Build LP problem
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

function analyze_recycling(step_data, name)
    println("\n" * "="^80)
    println("T6: Recycling Pilot - $name")
    println("="^80)

    @printf("\n%5s │ %12s │ %6s │ %10s │ %10s │ %10s\n",
            "Step", "Gap μ", "KKT", "align_k1", "align_k5", "align_k10")
    println("─"^70)

    for d in step_data
        @printf("%5d │ %12.4e │ %6d │ %10.4f │ %10.4f │ %10.4f\n",
                d.iteration, d.gap, d.kkt_iters, d.align_k1, d.align_k5, d.align_k10)
    end

    # Analysis
    valid_data = filter(d -> isfinite(d.align_k1), step_data[2:end])  # Skip first (trivial)

    if length(valid_data) >= 2
        avg_k1 = sum(d.align_k1 for d in valid_data) / length(valid_data)
        avg_k5 = sum(d.align_k5 for d in valid_data if isfinite(d.align_k5)) /
                 count(d -> isfinite(d.align_k5), valid_data)
        avg_k10 = sum(d.align_k10 for d in valid_data if isfinite(d.align_k10)) /
                  count(d -> isfinite(d.align_k10), valid_data)

        println("\n" * "-"^70)
        println("ANALYSIS:")
        println("-"^70)
        @printf("  Average step-to-step alignment (k=1): %.4f\n", avg_k1)
        @printf("  Average step-to-step alignment (k=5): %.4f\n", avg_k5)
        @printf("  Average step-to-step alignment (k=10): %.4f\n", avg_k10)

        println()
        if avg_k1 > 0.9 && avg_k5 > 0.8
            println("  → DIAGNOSIS: High subspace alignment")
            println("    Recycling Ritz vectors would likely help.")
            println("    The slow modes change slowly between steps.")
            println("    Cure: Recycle k Ritz vectors as initial subspace.")
        elseif avg_k1 > 0.7 || avg_k5 > 0.6
            println("  → DIAGNOSIS: Moderate subspace alignment")
            println("    Recycling may help partially.")
            println("    Consider combining with deflation.")
        else
            println("  → DIAGNOSIS: Low subspace alignment")
            println("    Modes change too rapidly for simple recycling.")
            println("    Consider A-dependent coarse space construction.")
        end

        # Also check if alignment varies across iterations
        early = valid_data[1:length(valid_data)÷2]
        late = valid_data[length(valid_data)÷2+1:end]
        if !isempty(early) && !isempty(late)
            avg_early = sum(d.align_k1 for d in early) / length(early)
            avg_late = sum(d.align_k1 for d in late) / length(late)
            @printf("\n  Alignment early: %.4f, late: %.4f\n", avg_early, avg_late)
        end
    else
        println("\n  [Not enough valid data points for analysis]")
    end

    return step_data
end

function main()
    println("="^80)
    println("T6: RECYCLING PILOT")
    println("="^80)
    println()
    println("Testing whether recycling Ritz vectors from previous CG solves")
    println("would help convergence in subsequent solves.")
    println()
    println("Key metric: Subspace alignment between consecutive steps")
    println("  - High alignment → recycling helps")
    println("  - Low alignment → modes change too fast")

    # Test 1: Complete graph
    println("\n" * "="^80)
    println("Instance: LP Complete Graph K_8, d_v=4, d_e=2")
    println("="^80)
    prob = build_lp_problem(8, 4, 2; topology=:complete)
    data, _ = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=30)
    analyze_recycling(data, "LP K_8")

    # Test 2: Chain graph
    println("\n" * "="^80)
    println("Instance: LP Chain N=15, d_v=4, d_e=2")
    println("="^80)
    prob = build_lp_problem(15, 4, 2; topology=:chain)
    data, _ = run_instrumented_ipm(prob; raug=1e6, verbose=true, max_iters=30)
    analyze_recycling(data, "LP Chain N=15")

    println("\n" * "="^80)
    println("T6 COMPLETE")
    println("="^80)
end

main()
