#
# Cone-level unit tests for PowerCone
#
# Tests (from implementation guide):
#   1. Log-homogeneity: <F'(x), x> = -3
#   2. Log-homogeneity: F''(x) x = -F'(x)
#   3. Log-homogeneity: F'''(x)[x] = -2 F''(x)
#   4. Finite-difference gradient
#   5. Finite-difference Hessian
#   6. Finite-difference third derivative
#   7. Shadow primal: F'(xs) + s = 0
#   8. Scaling secants: M*x = s
#   9. Self-concordance bound
#

using LinearAlgebra
using SheafSDP: powphi, powbarrgrad!, powbarr!, powbarrhess!, powhess!,
                powscale!, powdualgrad!, cross3!, powincone, powindual,
                ldiv3!, copy3!

# Barrier function F(x) = -log(φ) - (1-α) log(x₁) - α log(x₂)
function pow_barrier(x, α)
    φ = powphi(x, α)
    return -log(φ) - (1 - α) * log(x[1]) - α * log(x[2])
end

# Generate random interior point
function random_interior_point(α; margin=0.5)
    x1 = 0.5 + rand()
    x2 = 0.5 + rand()
    bound = x1^α * x2^(1-α)
    t = margin * (2 * rand() - 1)  # |t| < margin
    x3 = t * bound
    return [x1, x2, x3]
end

# Generate random dual interior point
function random_dual_point(α; margin=0.5)
    s1 = 0.5 + rand()
    s2 = 0.5 + rand()
    bound = (s1/α)^α * (s2/(1-α))^(1-α)
    t = margin * (2 * rand() - 1)
    s3 = t * bound
    return [s1, s2, s3]
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

# Compute Hessian from Cholesky factor: F'' = L L'
function hessian_from_factor(L)
    return LowerTriangular(L) * LowerTriangular(L)'
end

# Direct Hessian computation (for testing)
function direct_hessian(x, α)
    x1, x2, x3 = x[1], x[2], x[3]
    a = 2α
    b = 2(1-α)
    p = x1^a * x2^b
    φ = p - x3^2
    ρ = p / φ

    d1 = (2ρ*a + b) / (2x1^2)
    d2 = (2ρ*b + a) / (2x2^2)

    H = zeros(3, 3)
    H[1,1] = d1 + a^2 * p * x3^2 / (x1^2 * φ^2)
    H[2,2] = d2 + b^2 * p * x3^2 / (x2^2 * φ^2)
    H[1,2] = a * b * p * x3^2 / (x1 * x2 * φ^2)
    H[2,1] = H[1,2]
    H[3,3] = 2(p + x3^2) / φ^2
    H[1,3] = -2a * x3 * p / (x1 * φ^2)
    H[3,1] = H[1,3]
    H[2,3] = -2b * x3 * p / (x2 * φ^2)
    H[3,2] = H[2,3]
    return H
end

println("=" ^ 70)
println("POWER CONE UNIT TESTS")
println("=" ^ 70)
println()

#
# Test 1: Log-homogeneity <F'(x), x> = -3
#
println("Test 1: Log-homogeneity <F'(x), x> = -3")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α)

    g = zeros(3)
    powbarrgrad!(g, x, α)
    inner = dot(g, x)

    err = abs(inner + 3)
    status = err < 1e-12 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): <F'(x),x> + 3 = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 2: Log-homogeneity F''(x) x = -F'(x)
#
println("Test 2: Log-homogeneity F''(x) x = -F'(x)")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α)

    g = zeros(3)
    powbarrgrad!(g, x, α)

    H = direct_hessian(x, α)
    Hx = H * x

    err = norm(Hx + g) / (norm(g) + 1e-10)
    status = err < 1e-12 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 3: Log-homogeneity F'''(x)[x] = -2 F''(x)
#
println("Test 3: Log-homogeneity F'''(x)[x] = -2 F''(x)")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α)

    H = direct_hessian(x, α)

    D = zeros(3, 3)
    powbarrhess!(D, x, x, α)

    err = norm(D + 2H) / (norm(H) + 1e-10)
    status = err < 1e-11 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 4: Gradient vs finite difference
#
println("Test 4: Barrier gradient vs finite difference")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α)

    g_analytic = zeros(3)
    powbarrgrad!(g_analytic, x, α)
    g_fd = fd_gradient(y -> pow_barrier(y, α), x)

    err = norm(g_analytic - g_fd) / (norm(g_fd) + 1e-10)
    status = err < 1e-6 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 5: Hessian (direct) vs finite difference
#
println("Test 5: Direct Hessian vs finite difference")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α)

    H_analytic = direct_hessian(x, α)
    H_fd = fd_hessian(y -> pow_barrier(y, α), x)

    err = norm(H_analytic - H_fd) / (norm(H_fd) + 1e-10)
    status = err < 1e-4 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 6: Factor reconstruction F'' = L L'
#
println("Test 6: Factor reconstruction F'' = L L'")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α)

    L = zeros(3, 3)
    powbarr!(L, x, α)

    H_factor = hessian_from_factor(L)
    H_direct = direct_hessian(x, α)

    err = norm(H_factor - H_direct) / (norm(H_direct) + 1e-10)
    status = err < 1e-12 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 7: Third derivative vs finite difference
#
println("Test 7: Third derivative F'''[u] vs finite difference")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α)
    u = randn(3)

    D_analytic = zeros(3, 3)
    powbarrhess!(D_analytic, x, u, α)
    D_fd = fd_hess_dir(y -> pow_barrier(y, α), x, u; h=1e-4)

    err = norm(D_analytic - D_fd) / (norm(D_fd) + 1e-10)
    status = err < 1e-2 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 8: Shadow primal F'(xs) + s = 0
#
println("Test 8: Shadow primal F'(xs) + s = 0")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    s = random_dual_point(α)

    xs = zeros(3)
    powdualgrad!(xs, s, α)

    g = zeros(3)
    powbarrgrad!(g, xs, α)
    residual = norm(g + s)

    status = residual < 1e-10 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): ||F'(xs) + s|| = $(round(residual, sigdigits=3)) [$status]")
end
println()

#
# Test 9: Central point F'(x₀) + x₀ = 0
#
println("Test 9: Central point F'(x₀) + x₀ = 0")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    x0 = [sqrt(1 + α), sqrt(2 - α), 0.0]

    g = zeros(3)
    powbarrgrad!(g, x0, α)

    err = norm(g + x0)
    status = err < 1e-12 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): ||F'(x₀) + x₀|| = $(round(err, sigdigits=3)) [$status]")
end
println()

#
# Test 10: Scaling secants M*x ≈ s
#
println("Test 10: Scaling secants M*x = s")
println("-" ^ 50)

for trial in 1:5
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α; margin=0.3)

    g = zeros(3)
    powbarrgrad!(g, x, α)
    s = -g .+ 0.1 .* randn(3)

    # Ensure s is in dual cone
    while !powindual(s, α)
        s[1] += 0.5
        s[2] += 0.5
    end

    M = zeros(3, 3)
    R = zeros(3, 3)
    xs = zeros(3)
    ss = zeros(3)

    μv = powscale!(M, R, xs, ss, x, s, α)

    Mx = M * x
    secant_err = norm(Mx - s) / (norm(s) + 1e-10)

    status = secant_err < 1e-6 ? "PASS" : (secant_err < 1e-3 ? "WEAK" : "FAIL")
    println("  Trial $trial (α=$(round(α, sigdigits=3))): secant_err = $(round(secant_err, sigdigits=3)) [$status]")
end
println()

#
# Test 11: Self-concordance |F'''[u,u,u]| ≤ 2 (F''[u,u])^(3/2)
#
println("Test 11: Self-concordance bound")
println("-" ^ 50)

max_ratio = 0.0
for trial in 1:100
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α; margin=0.9)
    u = randn(3)
    u = u / norm(u)

    H = direct_hessian(x, α)
    Huu = dot(u, H * u)

    D = zeros(3, 3)
    powbarrhess!(D, x, u, α)
    Duuu = dot(u, D * u)

    ratio = abs(Duuu) / (Huu^(3/2) + 1e-20)
    global max_ratio = max(max_ratio, ratio)
end

status = max_ratio <= 2.0 + 1e-6 ? "PASS" : "FAIL"
println("  Max ratio |F'''[u,u,u]| / (F''[u,u])^(3/2) = $(round(max_ratio, sigdigits=4)) [$status]")
println()

#
# Test 12: α = 0.5 is rotated SOC (symmetric)
#
println("Test 12: α = 0.5 (rotated SOC special case)")
println("-" ^ 50)

α = 0.5
x = random_interior_point(α)
s = random_dual_point(α)

# At α = 0.5, primal and dual cones should be identical
println("  x in primal: $(powincone(x, α))")
println("  x in dual: $(powindual(x, α))")

# Central point should be (√1.5, √1.5, 0)
x0 = [sqrt(1.5), sqrt(1.5), 0.0]
g = zeros(3)
powbarrgrad!(g, x0, α)
err = norm(g + x0)
status = err < 1e-12 ? "PASS" : "FAIL"
println("  Central point at α=0.5: ($(round(x0[1], sigdigits=4)), $(round(x0[2], sigdigits=4)), 0) [$status]")
println()

#
# Test 13: Solve via Cholesky factor matches direct solve
#
println("Test 13: Factor solve F''⁻¹ w via chol3! + ldiv3!")
println("-" ^ 50)

for trial in 1:5
    local α, x, err, status
    α = 0.1 + 0.8 * rand()
    x = random_interior_point(α)
    w = randn(3)

    L = zeros(3, 3)
    powbarr!(L, x, α)  # computes Hessian then in-place Cholesky

    v_factor = zeros(3)
    copy3!(v_factor, w)
    ldiv3!(LowerTriangular(L), v_factor)
    ldiv3!(LowerTriangular(L)', v_factor)

    H = direct_hessian(x, α)
    v_direct = H \ w

    err = norm(v_factor - v_direct) / (norm(v_direct) + 1e-10)
    status = err < 1e-10 ? "PASS" : "FAIL"
    println("  Trial $trial (α=$(round(α, sigdigits=3))): rel_err = $(round(err, sigdigits=3)) [$status]")
end
println()

println("=" ^ 70)
println("POWER CONE TESTS COMPLETE")
println("=" ^ 70)
