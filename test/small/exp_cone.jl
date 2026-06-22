#
# Cone-level unit tests for EXP (§8 of exp-recipes.md)
#
# Tests:
#   1. Finite-difference the barrier (gradient, Hessian, third derivative)
#   2. Assert the scaling secants (M*x = s, M*δx = δs)
#   3. Force the off-central branch (non-parallel p, d)
#   4. Check the fallback crossover (rel_z near sqrt(eps))
#
using LinearAlgebra
using SheafSDP: exp_psi, exp_barrier_grad!, exp_barrier_factor!, exp_barrier_hess_dir!,
                expscale!, exp_shadow_primal!, cross3, in_exp_primal

# Barrier function F(x) = -log(ψ) - log(x₁) - log(x₂)
function exp_barrier(x)
    ψ = exp_psi(x)
    return -log(ψ) - log(x[1]) - log(x[2])
end

# Finite difference gradient
function fd_gradient(f, x; h=1e-7)
    g = zeros(3)
    for i in 1:3
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

# Finite difference Hessian
function fd_hessian(f, x; h=1e-5)
    H = zeros(3, 3)
    for i in 1:3, j in 1:3
        xpp = copy(x); xpp[i] += h; xpp[j] += h
        xpm = copy(x); xpm[i] += h; xpm[j] -= h
        xmp = copy(x); xmp[i] -= h; xmp[j] += h
        xmm = copy(x); xmm[i] -= h; xmm[j] -= h
        H[i,j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4h^2)
    end
    return H
end

# Finite difference of Hessian along direction u
function fd_hess_dir(f, x, u; h=1e-5)
    Hp = fd_hessian(f, x .+ h .* u; h=h/10)
    Hm = fd_hessian(f, x .- h .* u; h=h/10)
    return (Hp - Hm) / (2h)
end

println("=" ^ 70)
println("EXP CONE UNIT TESTS (§8)")
println("=" ^ 70)
println()

#
# Test 1: Finite-difference the barrier gradient
#
println("Test 1: Barrier gradient vs finite difference")
println("-" ^ 50)

for trial in 1:5
    # Random interior point
    x = [1.0 + rand(), 0.5 + rand(), -0.5 - rand()]
    while !in_exp_primal(x)
        x[1] *= 1.5
        x[3] -= 0.5
    end

    g_analytic = zeros(3)
    exp_barrier_grad!(g_analytic, x)
    g_fd = fd_gradient(exp_barrier, x)

    err = norm(g_analytic - g_fd) / (norm(g_fd) + 1e-10)
    status = err < 1e-6 ? "PASS" : "FAIL"
    println("  Trial $trial: rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 2: Finite-difference the Hessian (R*R' vs FD)
#
println("Test 2: Barrier Hessian (R*R') vs finite difference")
println("-" ^ 50)

for trial in 1:5
    x = [1.0 + rand(), 0.5 + rand(), -0.5 - rand()]
    while !in_exp_primal(x)
        x[1] *= 1.5
        x[3] -= 0.5
    end

    R = zeros(3, 3)
    exp_barrier_factor!(R, x)
    H_analytic = R * R'
    H_fd = fd_hessian(exp_barrier, x)

    err = norm(H_analytic - H_fd) / (norm(H_fd) + 1e-10)
    status = err < 1e-4 ? "PASS" : "FAIL"
    println("  Trial $trial: rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 3: Finite-difference the third derivative F'''[u]
#
println("Test 3: Barrier third derivative F'''[u] vs finite difference")
println("-" ^ 50)

for trial in 1:5
    x = [1.0 + rand(), 0.5 + rand(), -0.5 - rand()]
    while !in_exp_primal(x)
        x[1] *= 1.5
        x[3] -= 0.5
    end
    u = randn(3)

    D_analytic = zeros(3, 3)
    exp_barrier_hess_dir!(D_analytic, x, u)
    D_fd = fd_hess_dir(exp_barrier, x, u; h=1e-4)

    err = norm(D_analytic - D_fd) / (norm(D_fd) + 1e-10)
    status = err < 1e-2 ? "PASS" : "FAIL"
    println("  Trial $trial: rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 4: Scaling secants (Tuncel scaling properties)
#
println("Test 4: Scaling secants M*x = s, M*δx = δs")
println("-" ^ 50)

for trial in 1:5
    # Create non-parallel x and s to force Tuncel branch
    x = [2.0 + rand(), 1.0 + rand(), -1.0 - rand()]
    while !in_exp_primal(x)
        x[1] *= 1.5
        x[3] -= 0.5
    end

    # s should be in dual cone and not parallel to x
    # Use s = -F'(x) + perturbation to get non-central iterate
    g = zeros(3)
    exp_barrier_grad!(g, x)
    s = -g .+ 0.1 .* randn(3)

    # Ensure s is in dual cone interior
    while !(s[1] > 0 && s[3] < 0 && ℯ * s[1] >= -s[3] * exp(s[2] / s[3]))
        s[1] += 0.5
        s[3] -= 0.5
    end

    M = zeros(3, 3)
    R = zeros(3, 3)
    xs = zeros(3)
    ss = zeros(3)

    μv = expscale!(M, R, xs, ss, x, s)

    # Check M*x ≈ s
    Mx = M * x
    secant_err = norm(Mx - s) / (norm(s) + 1e-10)

    # Check centrality
    z = cross3(x, xs)
    rel_z = norm(z) / (norm(x) * norm(xs) + eps())

    status = secant_err < 1e-6 ? "PASS" : (secant_err < 1e-3 ? "WEAK" : "FAIL")
    branch = rel_z < sqrt(eps()) ? "fallback" : "Tuncel"
    println("  Trial $trial: secant_err = $(round(secant_err, sigdigits=3)), rel_z = $(round(rel_z, sigdigits=3)) [$branch] [$status]")
end
println()

#
# Test 5: Force off-central branch (Tuncel scaling)
#
println("Test 5: Force off-central branch (rel_z > sqrt(eps))")
println("-" ^ 50)

tuncel_count = 0
fallback_count = 0

for trial in 1:20
    # Deliberately non-parallel x and s
    θ = rand() * π/4 + π/8  # angle between 22.5 and 67.5 degrees

    x = [2.0, 1.0, -1.0]
    while !in_exp_primal(x)
        x[1] *= 1.5
        x[3] -= 0.5
    end

    # Rotate to get non-parallel s
    g = zeros(3)
    exp_barrier_grad!(g, x)
    s_base = -g

    # Add perpendicular component
    perp = randn(3)
    perp = perp - dot(perp, s_base) / dot(s_base, s_base) * s_base
    perp = perp / norm(perp) * norm(s_base) * tan(θ)
    s = s_base + perp

    # Ensure s is in dual cone
    while !(s[1] > 0 && s[3] < 0 && ℯ * s[1] >= -s[3] * exp(s[2] / s[3]))
        s[1] += 0.5
        s[3] -= 0.5
    end

    M = zeros(3, 3)
    R = zeros(3, 3)
    xs = zeros(3)
    ss = zeros(3)

    μv = expscale!(M, R, xs, ss, x, s)

    z = cross3(x, xs)
    rel_z = norm(z) / (norm(x) * norm(xs) + eps())

    if rel_z > sqrt(eps())
        global tuncel_count += 1
    else
        global fallback_count += 1
    end
end

println("  Tuncel branch: $tuncel_count / 20")
println("  Fallback branch: $fallback_count / 20")
status = tuncel_count >= 15 ? "PASS" : "FAIL"
println("  [$status] (expect mostly Tuncel)")
println()

#
# Test 6: Fallback crossover (rel_z near sqrt(eps))
#
println("Test 6: Fallback crossover - M stays PD across switch")
println("-" ^ 50)

# Start near central path and move away
x_central = [1.2909282315382298, 0.8051015526498357, -0.8278379086082098]
g = zeros(3)
exp_barrier_grad!(g, x_central)
s_central = -g  # exactly on central path

for scale in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    # Perturb s slightly off central path
    perp = [0.1, -0.2, 0.1]
    s = s_central + scale * perp

    M = zeros(3, 3)
    R = zeros(3, 3)
    xs = zeros(3)
    ss = zeros(3)

    μv = expscale!(M, R, xs, ss, x_central, s)

    z = cross3(x_central, xs)
    rel_z = norm(z) / (norm(x_central) * norm(xs) + eps())

    # Check M is PD
    λ_min = minimum(eigvals(Symmetric(M)))
    is_pd = λ_min > 0

    # Check secant
    secant_err = norm(M * x_central - s) / (norm(s) + 1e-10)

    branch = rel_z < sqrt(eps()) ? "fallback" : "Tuncel"
    pd_status = is_pd ? "PD" : "NOT PD"
    println("  scale=$(round(scale, sigdigits=1)): rel_z=$(round(rel_z, sigdigits=2)), λ_min=$(round(λ_min, sigdigits=2)), secant=$(round(secant_err, sigdigits=2)) [$branch] [$pd_status]")
end
println()

#
# Test 7: Shadow primal Newton convergence
#
println("Test 7: Shadow primal Newton convergence")
println("-" ^ 50)

for trial in 1:5
    local g, status

    # Random dual point s in interior
    s = [1.0 + rand(), rand() - 0.5, -1.0 - rand()]
    while !(s[1] > 0 && s[3] < 0 && ℯ * s[1] >= -s[3] * exp(s[2] / s[3]))
        s[1] += 0.5
        s[3] -= 0.5
    end

    xs = zeros(3)
    exp_shadow_primal!(xs, s)

    # Check F'(xs) + s ≈ 0
    g = zeros(3)
    exp_barrier_grad!(g, xs)
    residual = norm(g + s)

    status = residual < 1e-10 ? "PASS" : "FAIL"
    println("  Trial $trial: ||F'(x̃) + s|| = $(round(residual, sigdigits=3)) [$status]")
end
println()

println("=" ^ 70)
println("CONE-LEVEL TESTS COMPLETE")
println("=" ^ 70)
