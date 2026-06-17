#
# Debug SOC corrector behavior
#
using SheafSDP
using SparseArrays
using LinearAlgebra
using Random
using SheafSDP: SOC, SOCCache, Caches, FVector, cache, cache_size,
                update_scaling!, max_step, jdot, det_soc, jmul,
                apply_H_half!, jordan_prod!, arrow_inv!, corrector_term!
using BlockSparseArrays: vtxs, colrange, ncols

Random.seed!(42)

# Small problem to trace
nv = 5
dv = 4
de = 2

edges = [(i, i+1) for i in 1:nv-1]  # simple chain
ne = length(edges)

src, dst, maps = Int[], Int[], Matrix{Float64}[]
for (e_idx, (u, v)) in enumerate(edges)
    push!(src, u); push!(dst, e_idx); push!(maps, randn(de, dv))
    push!(src, v); push!(dst, e_idx); push!(maps, randn(de, dv))
end

P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
B_sp = sparse(B)
n = size(B_sp, 2)
m = size(B_sp, 1)

# Helper: generate random SOC interior point
function rand_soc_interior(dim; margin=0.5)
    x̄ = randn(dim - 1)
    x₀ = norm(x̄) + margin + rand()
    return [x₀; x̄]
end

# Build feasible initial point (SOC interior)
p0 = zeros(n)
d0 = zeros(n)
for v in vtxs(B)
    r = colrange(B, v)
    dim = length(r)
    p0[r] .= rand_soc_interior(dim)
    d0[r] .= rand_soc_interior(dim)
end
y0 = randn(m)
c = B_sp' * y0 + d0
g = B_sp * p0

cones = [SOC() for _ in 1:nv]

# Create a cache for testing
function make_test_cache(n)
    val = FVector{Float64}(undef, 1 + n)
    xcol = FVector{Int}(undef, 2)
    xblk = FVector{Int}(undef, 2)
    xcol[1] = 1
    xcol[2] = n + 1
    xblk[1] = 1
    xblk[2] = 1 + n + 1
    caches = Caches(val, xcol, xblk)
    return cache(caches, 1, SOC())
end

println("=== SOC Corrector Debug ===\n")

# Test on first block
v = first(vtxs(B))
r = colrange(B, v)
dim = length(r)

p_v = p0[r]
d_v = d0[r]

println("Block dim: $dim")
println("p_v: $p_v")
println("d_v: $d_v")
println("det(p): $(det_soc(p_v)), det(d): $(det_soc(d_v))")
println()

# Set up scaling
c_test = make_test_cache(dim)
update_scaling!(c_test, SOC(), p_v, d_v, Val(:L))

β = c_test.β[]
w = Vector(c_test.w)
println("Scaling: β = $β")
println("w = $w")
println("wᵀJw = $(jdot(w, w))")  # Should be 1
println()

# Build H and verify Hp = d
η = 1 / (β^2)
a = jmul(w)  # Jw
H = η * (2 * a * a' - Diagonal([1; fill(-1, dim-1)]))
println("Hp ≈ d? $(norm(H * p_v - d_v))")
println()

# Test H½
λ = similar(p_v)
apply_H_half!(λ, c_test, p_v, false)
println("λ = H½ p = $λ")
println("det(λ) = $(det_soc(λ))")
println("Expected det(λ) = √(det(p)det(d)) = $(sqrt(det_soc(p_v) * det_soc(d_v)))")
println()

# Create a random affine direction for testing
Δp = 0.1 * randn(dim)
Δd = 0.1 * randn(dim)

# Compute corrector
σμ = 0.5 * jdot(p_v, d_v)  # Some centering
rc = similar(p_v)
corrector_term!(rc, c_test, SOC(), p_v, d_v, Δp, Δd, σμ, Val(:L))

println("σμ = $σμ")
println("Δp = $Δp")
println("Δd = $Δd")
println("r_c = $rc")
println()

# Verify: H r_c should equal the desired RHS
H_rc = H * rc
println("H r_c = $H_rc")

# The first-order terms should dominate: σμ/d - d
# For SOC: d⁻¹ = Jd / det(d)
det_d = det_soc(d_v)
d_inv = jmul(d_v) / det_d
expected_first_order = σμ * d_inv - p_v
println("Expected first-order (σμ d⁻¹ - p): $expected_first_order")
println()

# Check step sizes
τ_p = max_step(c_test, SOC(), p_v, Δp, true, 0.99, Val(:L))
τ_d = max_step(c_test, SOC(), d_v, Δd, false, 0.99, Val(:L))
println("Step sizes: τ_p = $τ_p, τ_d = $τ_d")

# Check det after stepping
p_new = p_v + τ_p * Δp
d_new = d_v + τ_d * Δd
println("det(p + τΔp) = $(det_soc(p_new))")
println("det(d + τΔd) = $(det_soc(d_new))")
println()

# Now run the actual solver with verbose mode
println("=== Running solver with verbose output ===\n")
p, d, y = copy(p0), copy(d0), copy(y0)
result = solve!(p, d, y, c, g, B, F, L;
                cones, ε_feas=1e-8, ε_μ=1e-8, max_iter=20, τ_aug=1000.0,
                verbose=true)

println("\nFinal status: $(result.status)")
println("Iterations: $(result.iterations)")
println("Final μ: $(result.μ_history[end])")
println("Step sizes (last 5): τ_p=$(result.τ_p_history[max(1,end-4):end])")
println("Step sizes (last 5): τ_d=$(result.τ_d_history[max(1,end-4):end])")
