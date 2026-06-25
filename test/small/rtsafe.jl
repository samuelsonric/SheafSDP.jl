#
# Test battery for shadow-primal 1-D solves (rtsafe)
#
# Layers:
#   0: Reference oracle (round-trip residual)
#   1: Random interior points
#   2: Near-convergence regime (μ → 0)
#   3: Boundary and degenerate inputs
#   4: Bracket and monotonicity invariants
#   5: Warm-start / cold-start behavior
#   6: Performance / regression
#

using SheafSDP
using LinearAlgebra
using Random
using Printf

# Import internal functions for testing
import SheafSDP: expdualgrad!, expbarrgrad!, exppsi, expincone, expindual
import SheafSDP: powdualgrad!, powbarrgrad!, powphi, powincone, powindual

println("="^70)
println("Shadow-Primal 1-D Solve Test Battery")
println("="^70)
println()

#
# Layer 0: Reference oracle (round-trip residual)
#
# The defining test: solve F'(x̃) = -d, then verify ‖F'(x̃) + d‖ ≈ 0
#

function test_exp_roundtrip(d::Vector{T}; tol::T = T(1e-10)) where {T}
    xs = zeros(T, 3)
    g = zeros(T, 3)

    expdualgrad!(xs, one(T), d)
    expbarrgrad!(g, xs)

    residual = norm(g .+ d)
    return residual < tol, residual
end

function test_pow_roundtrip(d::Vector{T}, α::T; tol::T = T(1e-10)) where {T}
    xs = zeros(T, 3)
    g = zeros(T, 3)

    powdualgrad!(xs, one(T), d, α)
    powbarrgrad!(g, xs, α)

    residual = norm(g .+ d)
    return residual < tol, residual
end

#
# Layer 1: Random interior points
#
# Generate valid dual points by d = -F'(x) for random interior x
#

function layer1_exp(; n_tests::Int = 1000)
    Random.seed!(42)
    failures = 0
    max_residual = 0.0

    for _ in 1:n_tests
        # Generate random interior point
        x = zeros(3)
        x[1] = 0.1 + 2.0 * rand()
        x[2] = 0.1 + 2.0 * rand()
        # Ensure ψ > 0: x₃ < x₂ log(x₁/x₂)
        ψ_bound = x[2] * log(x[1] / x[2])
        x[3] = ψ_bound - 0.1 - rand()

        # d = -F'(x)
        d = zeros(3)
        expbarrgrad!(d, x)
        d .*= -1

        pass, residual = test_exp_roundtrip(d)
        max_residual = max(max_residual, residual)
        if !pass
            failures += 1
        end
    end

    return failures, max_residual
end

function layer1_pow(; n_tests::Int = 1000, α_values = [0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.9])
    Random.seed!(42)
    failures = 0
    max_residual = 0.0

    for α in α_values
        for _ in 1:div(n_tests, length(α_values))
            # Generate random interior point
            x = zeros(3)
            x[1] = 0.1 + 2.0 * rand()
            x[2] = 0.1 + 2.0 * rand()
            # Ensure φ > 0: |x₃| < x₁^α x₂^(1-α)
            p_bound = x[1]^α * x[2]^(1-α)
            x[3] = (2 * rand() - 1) * 0.9 * p_bound

            # d = -F'(x)
            d = zeros(3)
            powbarrgrad!(d, x, α)
            d .*= -1

            pass, residual = test_pow_roundtrip(d, α)
            max_residual = max(max_residual, residual)
            if !pass
                failures += 1
            end
        end
    end

    return failures, max_residual
end

#
# Layer 2: Near-convergence regime (μ → 0)
#
# Walk down the central path with μ → 0, testing at each step
#

function layer2_exp(; μ_steps::Int = 40, μ_decay::Float64 = 0.6)
    Random.seed!(123)
    failures = 0
    max_residual = 0.0

    # Start from identity point
    x = zeros(3)
    x[1] = 1.2909282315382298
    x[2] = 0.8051015526498357
    x[3] = -0.8278379086082098

    μ = 1.0

    for k in 1:μ_steps
        μ *= μ_decay

        # d = -μ F'(x) with small perturbation off central path
        d = zeros(3)
        expbarrgrad!(d, x)
        d .*= -μ
        d .+= 0.1 * sqrt(μ) * randn(3) .* abs.(d)

        # Ensure d is still in dual interior (d₃ < 0)
        if d[3] >= 0
            d[3] = -0.01 * μ
        end

        pass, residual = test_exp_roundtrip(d; tol = 1e-8)
        max_residual = max(max_residual, residual)
        if !pass
            failures += 1
            @printf("  EXP Layer 2 fail at μ=%.2e: residual=%.2e\n", μ, residual)
        end
    end

    return failures, max_residual
end

function layer2_pow(; μ_steps::Int = 40, μ_decay::Float64 = 0.6, α::Float64 = 0.667)
    Random.seed!(123)
    failures = 0
    max_residual = 0.0

    # Start from identity point
    x = zeros(3)
    x[1] = sqrt(1 + α)
    x[2] = sqrt(2 - α)
    x[3] = 0.0

    μ = 1.0

    for k in 1:μ_steps
        μ *= μ_decay

        # d = -μ F'(x) with small perturbation
        d = zeros(3)
        powbarrgrad!(d, x, α)
        d .*= -μ
        d .+= 0.05 * sqrt(μ) * randn(3) .* max.(abs.(d), 0.01 * μ)

        # Ensure d is in dual interior:
        # d₁ > 0, d₂ > 0, (d₁/α)^α (d₂/(1-α))^(1-α) > |d₃|
        d[1] = max(d[1], 0.01 * μ)
        d[2] = max(d[2], 0.01 * μ)

        # Check and fix dual cone membership
        dual_bound = (d[1] / α)^α * (d[2] / (1 - α))^(1 - α)
        if abs(d[3]) >= 0.99 * dual_bound
            d[3] = sign(d[3]) * 0.5 * dual_bound
        end

        pass, residual = test_pow_roundtrip(d, α; tol = 1e-8)
        max_residual = max(max_residual, residual)
        if !pass
            failures += 1
            @printf("  POW Layer 2 fail at μ=%.2e: residual=%.2e\n", μ, residual)
        end
    end

    return failures, max_residual
end

#
# Layer 3: Boundary and degenerate inputs
#

function layer3_exp()
    failures = 0
    max_residual = 0.0

    # Test near d₃ → 0⁻ (ψ̃ → ∞)
    for d3 in [-1e-1, -1e-3, -1e-6, -1e-9, -1e-12]
        d = [1.0, 1.0, d3]
        pass, residual = test_exp_roundtrip(d; tol = 1e-6)
        max_residual = max(max_residual, residual)
        if !pass
            failures += 1
            @printf("  EXP Layer 3 fail at d₃=%.2e: residual=%.2e\n", d3, residual)
        end
    end

    return failures, max_residual
end

function layer3_pow()
    failures = 0
    max_residual = 0.0

    # Test d₃ → 0 (symmetric slice, ρ = 1 shortcut)
    for d3 in [0.0, 1e-12, -1e-12, 1e-6, -1e-6]
        d = [1.0, 1.0, d3]
        for α in [0.25, 0.5, 0.75]
            pass, residual = test_pow_roundtrip(d, α; tol = 1e-8)
            max_residual = max(max_residual, residual)
            if !pass
                failures += 1
                @printf("  POW Layer 3 fail at d₃=%.2e, α=%.2f: residual=%.2e\n", d3, α, residual)
            end
        end
    end

    # Test α → 0⁺ and α → 1⁻
    for α in [0.01, 0.05, 0.95, 0.99]
        d = [1.0, 1.0, 0.5]
        pass, residual = test_pow_roundtrip(d, α; tol = 1e-8)
        max_residual = max(max_residual, residual)
        if !pass
            failures += 1
            @printf("  POW Layer 3 fail at α=%.2f: residual=%.2e\n", α, residual)
        end
    end

    # Test α = 1/2 closed form matches general solve
    α = 0.5
    for _ in 1:100
        Random.seed!(42)
        d = [0.5 + rand(), 0.5 + rand(), (2 * rand() - 1) * 0.5]

        # Verify against roundtrip
        pass, residual = test_pow_roundtrip(d, α; tol = 1e-10)
        max_residual = max(max_residual, residual)
        if !pass
            failures += 1
        end
    end

    return failures, max_residual
end

#
# Layer 4: Bracket and monotonicity invariants
#

function layer4_exp(; n_tests::Int = 100)
    Random.seed!(456)
    failures = 0

    for _ in 1:n_tests
        # Generate valid d
        x = zeros(3)
        x[1] = 0.1 + 2.0 * rand()
        x[2] = 0.1 + 2.0 * rand()
        ψ_bound = x[2] * log(x[1] / x[2])
        x[3] = ψ_bound - 0.1 - rand()

        d = zeros(3)
        expbarrgrad!(d, x)
        d .*= -1

        # Check h is decreasing at the solution
        d1, d2, d3 = d[1], d[2], d[3]
        ψ = -1.0 / d3

        x1of(x2) = x2 / (ψ * d1) + 1.0 / d1
        h(x2) = (log(x1of(x2) / x2) - 1.0) / ψ + 1.0 / x2 - d2

        xs = zeros(3)
        expdualgrad!(xs, 1.0, d)
        x2_sol = xs[2]

        # Check h(x2_sol) ≈ 0
        if abs(h(x2_sol)) > 1e-8
            failures += 1
            @printf("  EXP Layer 4: |h(x̃₂)| = %.2e > tol\n", abs(h(x2_sol)))
        end

        # Check h is decreasing: h(x2_sol - ε) > 0 > h(x2_sol + ε)
        ε = 1e-6 * x2_sol
        if h(x2_sol - ε) < h(x2_sol + ε)
            failures += 1
            @printf("  EXP Layer 4: h not decreasing at solution\n")
        end
    end

    return failures
end

function layer4_pow(; n_tests::Int = 100)
    Random.seed!(456)
    failures = 0

    for α in [0.25, 0.667, 0.9]
        for _ in 1:div(n_tests, 3)
            # Generate valid d
            x = zeros(3)
            x[1] = 0.1 + 2.0 * rand()
            x[2] = 0.1 + 2.0 * rand()
            p_bound = x[1]^α * x[2]^(1-α)
            x[3] = (2 * rand() - 1) * 0.9 * p_bound

            d = zeros(3)
            powbarrgrad!(d, x, α)
            d .*= -1

            # Skip if d₃ ≈ 0 (shortcut case)
            if abs(d[3]) < 1e-10
                continue
            end

            # Check g(1) ≤ 0 always
            s1, s2, s3 = d[1], d[2], d[3]
            a = 2 * α
            b = 2 * (1 - α)

            X1(ρ) = (a * ρ + 1 - α) / s1
            X2(ρ) = (b * ρ + α) / s2
            g(ρ) = ρ * (ρ - 1) - (s3^2 / 4) * X1(ρ)^a * X2(ρ)^b

            if g(1.0) > 1e-10
                failures += 1
                @printf("  POW Layer 4: g(1) = %.2e > 0 (should be ≤ 0)\n", g(1.0))
            end

            # Check solution
            xs = zeros(3)
            powdualgrad!(xs, one(Float64), d, α)

            # Recover ρ from solution
            X1_sol = xs[1]
            ρ_sol = (X1_sol * s1 - (1 - α)) / a

            if abs(g(ρ_sol)) > 1e-6
                failures += 1
                @printf("  POW Layer 4: |g(ρ)| = %.2e > tol at α=%.2f\n", abs(g(ρ_sol)), α)
            end
        end
    end

    return failures
end

#
# Layer 5: Cold-start behavior
#
# Verify cold start (seed = 1) works for all cases
#

function layer5(; n_tests::Int = 200)
    Random.seed!(789)
    exp_failures = 0
    pow_failures = 0

    # EXP: test with varying scales of x̃₂
    for scale in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
        for _ in 1:div(n_tests, 6)
            x = zeros(3)
            x[1] = scale * (0.5 + rand())
            x[2] = scale * (0.5 + rand())
            ψ_bound = x[2] * log(x[1] / x[2])
            x[3] = ψ_bound - scale * 0.2 * (0.5 + rand())

            # Skip if not in cone
            if !expincone(x)
                continue
            end

            d = zeros(3)
            expbarrgrad!(d, x)
            d .*= -1

            # Skip if not in dual
            if !expindual(d)
                continue
            end

            pass, _ = test_exp_roundtrip(d; tol = 1e-8)
            if !pass
                exp_failures += 1
            end
        end
    end

    # POW: test with varying ρ scales
    for α in [0.25, 0.5, 0.75]
        for _ in 1:div(n_tests, 3)
            x = zeros(3)
            x[1] = 0.1 + 2.0 * rand()
            x[2] = 0.1 + 2.0 * rand()
            p_bound = x[1]^α * x[2]^(1-α)
            x[3] = (2 * rand() - 1) * 0.95 * p_bound  # close to boundary

            d = zeros(3)
            powbarrgrad!(d, x, α)
            d .*= -1

            pass, _ = test_pow_roundtrip(d, α; tol = 1e-8)
            if !pass
                pow_failures += 1
            end
        end
    end

    return exp_failures, pow_failures
end

#
# Layer 6: Performance check
#
# Count function evaluations (via timing proxy)
#

function layer6()
    Random.seed!(999)

    # Generate test cases
    exp_cases = []
    pow_cases = []

    for _ in 1:1000
        x = zeros(3)
        x[1] = 0.1 + 2.0 * rand()
        x[2] = 0.1 + 2.0 * rand()
        ψ_bound = x[2] * log(x[1] / x[2])
        x[3] = ψ_bound - 0.1 - rand()

        d = zeros(3)
        expbarrgrad!(d, x)
        d .*= -1
        push!(exp_cases, copy(d))
    end

    for _ in 1:1000
        α = 0.667
        x = zeros(3)
        x[1] = 0.1 + 2.0 * rand()
        x[2] = 0.1 + 2.0 * rand()
        p_bound = x[1]^α * x[2]^(1-α)
        x[3] = (2 * rand() - 1) * 0.9 * p_bound

        d = zeros(3)
        powbarrgrad!(d, x, α)
        d .*= -1
        push!(pow_cases, (copy(d), α))
    end

    # Warmup
    xs = zeros(3)
    x2_seed = 1.0
    ρ_seed = 1.0
    for d in exp_cases[1:10]
        x2_seed = expdualgrad!(xs, x2_seed, d)
    end
    for (d, α) in pow_cases[1:10]
        ρ_seed = powdualgrad!(xs, ρ_seed, d, α)
    end

    # Time EXP (with warm-starting)
    x2_seed = 1.0
    t_exp = @elapsed begin
        for d in exp_cases
            x2_seed = expdualgrad!(xs, x2_seed, d)
        end
    end

    # Time POW (with warm-starting)
    ρ_seed = 1.0
    t_pow = @elapsed begin
        for (d, α) in pow_cases
            ρ_seed = powdualgrad!(xs, ρ_seed, d, α)
        end
    end

    return t_exp, t_pow, length(exp_cases), length(pow_cases)
end

#
# Layer 7: Warm vs Cold iteration count comparison
#

import SheafSDP: rtsafe_count

function layer7_exp_warmstart()
    Random.seed!(999)

    # Generate a trajectory of EXP solves (simulating IPM iterations)
    x = zeros(3)
    x[1] = 1.2909282315382298
    x[2] = 0.8051015526498357
    x[3] = -0.8278379086082098

    trajectory = []
    μ = 1.0
    for _ in 1:50
        μ *= 0.8

        d = zeros(3)
        expbarrgrad!(d, x)
        d .*= -μ
        d .+= 0.02 * sqrt(μ) * randn(3) .* abs.(d)
        if d[3] >= 0
            d[3] = -0.01 * μ
        end

        push!(trajectory, copy(d))
    end

    # Cold start: always seed = 1
    cold_iters = Int[]
    for d in trajectory
        d1, d2, d3 = d[1], d[2], d[3]
        # negated h so it's increasing (rtsafe convention)
        h(x2) = d3 * (log((1.0 - x2 * d3) / (d1 * x2)) - 1.0) - 1.0 / x2 + d2
        hp(x2) = -d3^2 / (1.0 - x2 * d3) - d3 / x2 + 1.0 / x2^2

        seed = 1.0
        if h(seed) < 0
            lo = seed
            hi = 2 * seed
            while h(hi) < 0
                hi *= 2
            end
        else
            hi = seed
            lo = seed / 2
            while h(lo) > 0
                lo /= 2
            end
        end

        _, iters = rtsafe_count(h, hp, lo, hi, seed)
        push!(cold_iters, iters)
    end

    # Warm start: use previous x̃₂ as seed
    warm_iters = Int[]
    x2_prev = 1.0
    for d in trajectory
        d1, d2, d3 = d[1], d[2], d[3]
        # negated h so it's increasing (rtsafe convention)
        h(x2) = d3 * (log((1.0 - x2 * d3) / (d1 * x2)) - 1.0) - 1.0 / x2 + d2
        hp(x2) = -d3^2 / (1.0 - x2 * d3) - d3 / x2 + 1.0 / x2^2

        seed = x2_prev
        if h(seed) < 0
            lo = seed
            hi = 2 * seed
            while h(hi) < 0
                hi *= 2
            end
        else
            hi = seed
            lo = seed / 2
            while h(lo) > 0
                lo /= 2
            end
        end

        x2, iters = rtsafe_count(h, hp, lo, hi, seed)
        x2_prev = x2
        push!(warm_iters, iters)
    end

    return sum(cold_iters), sum(warm_iters), mean(cold_iters), mean(warm_iters)
end

function layer7_pow_warmstart()
    Random.seed!(999)

    # Generate a trajectory of POW solves (simulating IPM iterations)
    α = 0.667
    a = 2 * α
    b = 2 * (1 - α)

    # Generate 50 points along a trajectory
    trajectory = []
    x = zeros(3)
    x[1] = sqrt(1 + α)
    x[2] = sqrt(2 - α)
    x[3] = 0.0

    μ = 1.0
    for _ in 1:50
        μ *= 0.8  # typical IPM decay

        d = zeros(3)
        powbarrgrad!(d, x, α)
        d .*= -μ
        d .+= 0.02 * sqrt(μ) * randn(3) .* max.(abs.(d), 0.01 * μ)

        d[1] = max(d[1], 0.01 * μ)
        d[2] = max(d[2], 0.01 * μ)
        dual_bound = (d[1] / α)^α * (d[2] / (1 - α))^(1 - α)
        if abs(d[3]) >= 0.99 * dual_bound
            d[3] = sign(d[3]) * 0.5 * dual_bound
        end

        push!(trajectory, copy(d))
    end

    # Cold start: always seed = 1
    cold_iters = Int[]
    for d in trajectory
        s1, s2, s3 = d[1], d[2], d[3]

        if iszero(s3) || α == 0.5
            push!(cold_iters, 1)  # shortcut
            continue
        end

        k = s3^2 / 4
        X1(ρ) = (a * ρ + 1 - α) / s1
        X2(ρ) = (b * ρ + α) / s2
        g(ρ) = ρ * (ρ - 1) - k * X1(ρ)^a * X2(ρ)^b
        gp(ρ) = (2ρ - 1) - k * X1(ρ)^a * X2(ρ)^b * (a^2 / (a * ρ + 1 - α) + b^2 / (b * ρ + α))

        lo = 1.0
        hi = 2.0
        while g(hi) ≤ 0
            hi *= 2
        end

        _, iters = rtsafe_count(g, gp, lo, hi, 1.0)
        push!(cold_iters, iters)
    end

    # Warm start: use previous ρ as seed
    warm_iters = Int[]
    ρ_prev = 1.0
    for d in trajectory
        s1, s2, s3 = d[1], d[2], d[3]

        if iszero(s3) || α == 0.5
            ρ_prev = 1.0
            push!(warm_iters, 1)
            continue
        end

        k = s3^2 / 4
        X1(ρ) = (a * ρ + 1 - α) / s1
        X2(ρ) = (b * ρ + α) / s2
        g(ρ) = ρ * (ρ - 1) - k * X1(ρ)^a * X2(ρ)^b
        gp(ρ) = (2ρ - 1) - k * X1(ρ)^a * X2(ρ)^b * (a^2 / (a * ρ + 1 - α) + b^2 / (b * ρ + α))

        seed = ρ_prev >= 1 ? ρ_prev : 1.0
        lo = 1.0
        hi = max(2 * seed, 2.0)
        while g(hi) ≤ 0
            hi *= 2
        end

        ρ, iters = rtsafe_count(g, gp, lo, hi, seed)
        ρ_prev = ρ
        push!(warm_iters, iters)
    end

    return sum(cold_iters), sum(warm_iters), mean(cold_iters), mean(warm_iters)
end

function layer7_warmstart()
    exp_cold, exp_warm, exp_cold_mean, exp_warm_mean = layer7_exp_warmstart()
    pow_cold, pow_warm, pow_cold_mean, pow_warm_mean = layer7_pow_warmstart()
    return (exp_cold, exp_warm, exp_cold_mean, exp_warm_mean,
            pow_cold, pow_warm, pow_cold_mean, pow_warm_mean)
end

using Statistics: mean

#
# Run all layers
#

println("Layer 1: Random interior points")
println("-"^40)
exp_f1, exp_r1 = layer1_exp(n_tests=1000)
pow_f1, pow_r1 = layer1_pow(n_tests=1000)
@printf("  EXP: %d failures / 1000, max residual = %.2e\n", exp_f1, exp_r1)
@printf("  POW: %d failures / 1000, max residual = %.2e\n", pow_f1, pow_r1)
println()

println("Layer 2: Near-convergence regime (μ → 0)")
println("-"^40)
exp_f2, exp_r2 = layer2_exp(μ_steps=40)
pow_f2, pow_r2 = layer2_pow(μ_steps=40)
@printf("  EXP: %d failures / 40, max residual = %.2e\n", exp_f2, exp_r2)
@printf("  POW: %d failures / 40, max residual = %.2e\n", pow_f2, pow_r2)
println()

println("Layer 3: Boundary and degenerate inputs")
println("-"^40)
exp_f3, exp_r3 = layer3_exp()
pow_f3, pow_r3 = layer3_pow()
@printf("  EXP: %d failures, max residual = %.2e\n", exp_f3, exp_r3)
@printf("  POW: %d failures, max residual = %.2e\n", pow_f3, pow_r3)
println()

println("Layer 4: Bracket and monotonicity invariants")
println("-"^40)
exp_f4 = layer4_exp(n_tests=100)
pow_f4 = layer4_pow(n_tests=100)
@printf("  EXP: %d invariant violations / 100\n", exp_f4)
@printf("  POW: %d invariant violations / 100\n", pow_f4)
println()

println("Layer 5: Cold-start behavior")
println("-"^40)
exp_f5, pow_f5 = layer5(n_tests=200)
@printf("  EXP: %d cold-start failures / 200\n", exp_f5)
@printf("  POW: %d cold-start failures / 200\n", pow_f5)
println()

println("Layer 6: Performance")
println("-"^40)
t_exp, t_pow, n_exp, n_pow = layer6()
@printf("  EXP: %.1f μs/solve (%d solves)\n", 1e6 * t_exp / n_exp, n_exp)
@printf("  POW: %.1f μs/solve (%d solves)\n", 1e6 * t_pow / n_pow, n_pow)
println()

println("Layer 7: Warm vs Cold start iterations")
println("-"^40)
(exp_cold, exp_warm, exp_cold_mean, exp_warm_mean,
 pow_cold, pow_warm, pow_cold_mean, pow_warm_mean) = layer7_warmstart()
@printf("  EXP cold: %d total (%.1f mean), warm: %d total (%.1f mean), savings: %.0f%%\n",
        exp_cold, exp_cold_mean, exp_warm, exp_warm_mean, 100 * (1 - exp_warm / exp_cold))
@printf("  POW cold: %d total (%.1f mean), warm: %d total (%.1f mean), savings: %.0f%%\n",
        pow_cold, pow_cold_mean, pow_warm, pow_warm_mean, 100 * (1 - pow_warm / pow_cold))
println()

# Summary
println("="^70)
total_failures = exp_f1 + pow_f1 + exp_f2 + pow_f2 + exp_f3 + pow_f3 + exp_f4 + pow_f4 + exp_f5 + pow_f5
if total_failures == 0
    println("ALL LAYERS PASSED")
else
    @printf("TOTAL FAILURES: %d\n", total_failures)
end
println("="^70)
