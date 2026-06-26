#
# T1 - Sheaf-Laplacian Spectrum (Structural Floor)
#
# Purpose: Measure the structural lower bound on conditioning,
# completely independent of the IPM.
#
# Measures:
# - dim H⁰ = number of eigenvalues at ~0 (kernel of L)
# - λ₁⁺ = smallest nonzero eigenvalue (spectral gap)
# - Shape of low end: clean gap or fat cluster?
#

using SheafSDP
using LinearAlgebra
using Printf
using Random

# Import test instance builders
include("../small/qp.jl")
include("../small/lp.jl")
include("../small/soc.jl")

function analyze_sheaf_laplacian(B; name="unknown", tol=1e-10)
    m, n = size(B)

    # Compute L = B'B (sheaf Laplacian)
    L = Matrix(B' * B)
    L = Symmetric(L)

    # Full eigendecomposition
    λ = eigvals(L)
    sort!(λ)

    # Count kernel dimension (eigenvalues < tol)
    dim_H0 = count(x -> abs(x) < tol, λ)

    # Find smallest nonzero eigenvalue
    nonzero_λ = filter(x -> abs(x) >= tol, λ)
    λ1_plus = isempty(nonzero_λ) ? NaN : minimum(nonzero_λ)
    λ_max = maximum(λ)

    # Spectral gap ratio
    gap_ratio = λ1_plus / λ_max

    # Check for fat low cluster: count eigenvalues < 10 * λ1_plus
    if !isnan(λ1_plus)
        low_cluster_count = count(x -> tol <= abs(x) < 10 * λ1_plus, λ)
    else
        low_cluster_count = 0
    end

    # Condition number of L restricted to range(B)
    if !isnan(λ1_plus)
        κ_L = λ_max / λ1_plus
    else
        κ_L = Inf
    end

    println("\n" * "="^70)
    println("T1: Sheaf Laplacian Spectrum - $name")
    println("="^70)
    println("Matrix dimensions: B is $m × $n (edges × vertices)")
    println()
    println("Kernel (H⁰):")
    println("  dim H⁰ = $dim_H0")
    println()
    println("Spectral gap:")
    println("  λ₁⁺ (smallest nonzero) = $(@sprintf("%.4e", λ1_plus))")
    println("  λ_max                  = $(@sprintf("%.4e", λ_max))")
    println("  gap ratio λ₁⁺/λ_max    = $(@sprintf("%.4e", gap_ratio))")
    println("  κ(L|range) = λ_max/λ₁⁺ = $(@sprintf("%.2e", κ_L))")
    println()
    println("Low-end shape:")
    println("  Eigenvalues in [λ₁⁺, 10λ₁⁺): $low_cluster_count")
    if low_cluster_count > 5
        println("  → FAT LOW CLUSTER detected (suggests structural issues)")
    else
        println("  → Clean gap (structural floor is benign)")
    end

    # Print first 10 eigenvalues
    println()
    println("First 15 eigenvalues:")
    for i in 1:min(15, length(λ))
        marker = abs(λ[i]) < tol ? " (kernel)" : ""
        println("  λ[$i] = $(@sprintf("%12.4e", λ[i]))$marker")
    end

    return (
        name = name,
        m = m,
        n = n,
        dim_H0 = dim_H0,
        λ1_plus = λ1_plus,
        λ_max = λ_max,
        gap_ratio = gap_ratio,
        κ_L = κ_L,
        low_cluster_count = low_cluster_count,
        eigenvalues = λ
    )
end

function build_qp_instance(N, T)
    # Build QP consensus problem (from test/small/qp.jl patterns)
    Random.seed!(42)

    # Complete graph K_N
    nx, nu = 4, 2
    d_v = T * (nx + nu)
    d_e = nx

    src, tgt, maps = Int[], Int[], Matrix{Float64}[]
    for i in 1:N, j in i+1:N
        push!(src, i)
        push!(tgt, j)
        push!(maps, randn(d_e, d_v))
        push!(src, j)
        push!(tgt, i)
        push!(maps, randn(d_e, d_v))
    end

    B = SheafSDP.sheaf(src, tgt, maps)
    return B
end

function build_chain_instance(N, d_v, d_e)
    # Chain graph: 1 - 2 - 3 - ... - N
    Random.seed!(42)

    src, tgt, maps = Int[], Int[], Matrix{Float64}[]
    for i in 1:N-1
        push!(src, i)
        push!(tgt, i+1)
        push!(maps, randn(d_e, d_v))
        push!(src, i+1)
        push!(tgt, i)
        push!(maps, randn(d_e, d_v))
    end

    B = SheafSDP.sheaf(src, tgt, maps)
    return B
end

function build_grid_instance(side, d_v, d_e)
    # Grid graph: side × side
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
    return B
end

function build_star_instance(N, d_v, d_e)
    # Star graph: hub connected to N-1 leaves
    Random.seed!(42)

    src, tgt, maps = Int[], Int[], Matrix{Float64}[]
    hub = 1
    for leaf in 2:N
        push!(src, hub)
        push!(tgt, leaf)
        push!(maps, randn(d_e, d_v))
        push!(src, leaf)
        push!(tgt, hub)
        push!(maps, randn(d_e, d_v))
    end

    B = SheafSDP.sheaf(src, tgt, maps)
    return B
end

function main()
    println("="^70)
    println("T1: SHEAF LAPLACIAN SPECTRUM ANALYSIS")
    println("="^70)
    println()
    println("Testing structural floor across different graph topologies.")
    println("Looking for: dim H⁰, spectral gap λ₁⁺, low-end cluster shape.")

    results = []

    # Test 1: Complete graph (QP-style)
    B = build_qp_instance(10, 5)
    push!(results, analyze_sheaf_laplacian(B; name="Complete K₁₀, T=5"))

    # Test 2: Chain graph
    B = build_chain_instance(20, 4, 2)
    push!(results, analyze_sheaf_laplacian(B; name="Chain N=20"))

    # Test 3: Grid graph (small)
    B = build_grid_instance(5, 4, 2)
    push!(results, analyze_sheaf_laplacian(B; name="Grid 5×5"))

    # Test 4: Grid graph (larger)
    B = build_grid_instance(10, 4, 2)
    push!(results, analyze_sheaf_laplacian(B; name="Grid 10×10"))

    # Test 5: Star graph
    B = build_star_instance(20, 4, 2)
    push!(results, analyze_sheaf_laplacian(B; name="Star N=20"))

    # Summary table
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println()
    @printf("%-20s │ %6s │ %10s │ %10s │ %10s │ %s\n",
            "Instance", "dim H⁰", "λ₁⁺", "κ(L)", "low cluster", "Diagnosis")
    println("─"^80)

    for r in results
        diagnosis = if r.low_cluster_count > 5
            "STRUCTURAL (A)"
        elseif r.κ_L > 1e6
            "STRUCTURAL (A)"
        else
            "BENIGN"
        end

        @printf("%-20s │ %6d │ %10.2e │ %10.2e │ %10d │ %s\n",
                r.name, r.dim_H0, r.λ1_plus, r.κ_L, r.low_cluster_count, diagnosis)
    end

    println()
    println("Interpretation:")
    println("  STRUCTURAL (A): Small λ₁⁺ or fat low cluster → deflate bottom-k of L once")
    println("  BENIGN: Clean gap, moderate κ(L) → structural floor not the problem")

    return results
end

main()
