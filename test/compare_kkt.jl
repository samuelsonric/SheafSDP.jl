#
# Compare ADMM vs Uzawa on ECQP via KKT internals
#
using SheafSDP
using LinearAlgebra
using Random
using BlockSparseArrays: vtxs, colrange, block

Random.seed!(42)

# Problem size
nv = 100
dv = 8
de = 4

# Create edges (sparse graph)
edges = Tuple{Int,Int}[]
for i in 1:nv, j in i+1:nv
    rand() < 0.1 && push!(edges, (i, j))
end
edge_set = Set(edges)
for v in 1:nv
    has_edge = any(e -> e[1] == v || e[2] == v, edges)
    if !has_edge
        neighbor = v == 1 ? 2 : 1
        e = v < neighbor ? (v, neighbor) : (neighbor, v)
        if e ∉ edge_set
            push!(edges, e)
            push!(edge_set, e)
        end
    end
end
ne = length(edges)

# Build restriction maps
src, dst, maps = Int[], Int[], Matrix{Float64}[]
for (u, v) in edges
    push!(src, u); push!(dst, v); push!(maps, randn(de, dv))
    push!(src, v); push!(dst, u); push!(maps, randn(de, dv))
end

# Build sheaf structure
B = sheaf(src, dst, maps)
n = size(B, 2)
m = size(B, 1)

println("ECQP size: n=$n, m=$m, nv=$nv vertices, ne=$ne edges")
println()

# Build block-diagonal SPD Hessian A
A = SheafSDP.allocate_H(Float64, B)
for v in vtxs(A)
    Av = block(A, v, v, v)
    d = size(Av, 1)
    M = randn(d, d)
    Av .= M' * M + I
end

# Random RHS - ensure g is feasible (g ∈ im(B))
x_true = randn(n)
y_true = randn(m)
f = zeros(n)
for v in vtxs(A)
    rv = colrange(A, v)
    Av = block(A, v, v, v)
    f[rv] .= Av * x_true[rv]
end
f .+= B' * y_true  # f = A x_true + B' y_true
g = B * x_true      # g = B x_true (feasible!)

# Output vectors
x_uzw = zeros(n)
y_uzw = zeros(m)
x_admm = zeros(n)
y_admm = zeros(m)

# Uzawa workspace and settings via make_kkt
uzw_set_warmup = UzawaSettings{Float64}(atol=1e-10, rtol=1e-10, itmax=2000)
perm_uzw, B_uzw, uzw_wrk = SheafSDP.make_kkt(uzw_set_warmup, B)

# Permute A, f, x for Uzawa (block permutation)
A_uzw = SheafSDP.selectvtxs(A, perm_uzw)
f_uzw = SheafSDP.blockpermute(f, B, perm_uzw)
x_uzw_perm = zeros(n)
y_uzw_perm = zeros(m)

# Warmup Uzawa
SheafSDP.init_kkt!(uzw_wrk, uzw_set_warmup, A_uzw)
SheafSDP.solve_kkt!(uzw_wrk, uzw_set_warmup, x_uzw_perm, y_uzw_perm, A_uzw, B_uzw, f_uzw, g)

# Try different augmentation parameters for Uzawa
println("Uzawa augmentation sweep:")
for aaug in [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    uzw_set = UzawaSettings{Float64}(aaug=aaug, atol=1e-10, rtol=1e-10, itmax=2000)
    SheafSDP.init_kkt!(uzw_wrk, uzw_set, A_uzw)
    fill!(x_uzw_perm, 0); fill!(y_uzw_perm, 0)
    t = @elapsed niter = SheafSDP.solve_kkt!(uzw_wrk, uzw_set, x_uzw_perm, y_uzw_perm, A_uzw, B_uzw, f_uzw, g)
    println("  aaug=$aaug: α=$(round(uzw_wrk.α[], sigdigits=4)), time=$(round(t*1000, digits=2)) ms, iters=$niter")
end
println()

uzw_set = UzawaSettings{Float64}(aaug=1e6, atol=1e-10, rtol=1e-10, itmax=2000)

# ADMM workspace and settings via make_kkt
admm_set_warmup = SheafSDP.ADMMSettings{Float64}(aaug=2e6, atol=1e-10, rtol=1e-10, itmax=2000, iatol=1e-6, irtol=1e-6, iitmax=500)
perm_admm, B_admm, admm_wrk = SheafSDP.make_kkt(admm_set_warmup, B)

# For ADMM, perm is identity so no permutation needed
A_admm = A  # same as original since perm is identity
f_admm = f
x_admm_perm = zeros(n)
y_admm_perm = zeros(m)

# Warmup ADMM
SheafSDP.init_kkt!(admm_wrk, admm_set_warmup, A_admm)
SheafSDP.solve_kkt!(admm_wrk, admm_set_warmup, x_admm_perm, y_admm_perm, A_admm, B_admm, f_admm, g)

# Try different augmentation parameters
println("ADMM augmentation sweep (relax=1.0):")
for aaug in [1e5, 5e5, 1e6]
    admm_set = SheafSDP.ADMMSettings{Float64}(aaug=aaug, atol=1e-10, rtol=1e-10, itmax=2000, iatol=1e-6, irtol=1e-6, iitmax=500)
    SheafSDP.init_kkt!(admm_wrk, admm_set, A_admm)
    fill!(x_admm_perm, 0); fill!(y_admm_perm, 0)
    fill!(admm_wrk.z, 0); fill!(admm_wrk.u, 0)
    t = @elapsed niter = SheafSDP.solve_kkt!(admm_wrk, admm_set, x_admm_perm, y_admm_perm, A_admm, B_admm, f_admm, g)
    println("  aaug=$aaug: α=$(round(admm_wrk.α[], sigdigits=4)), time=$(round(t*1000, digits=2)) ms, iters=$niter")
end
println()

# Try different relaxation parameters (with default aaug)
println("ADMM relaxation sweep (aaug=0, raug=1):")
for relax in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8]
    admm_set = SheafSDP.ADMMSettings{Float64}(relax=relax, atol=1e-10, rtol=1e-10, itmax=2000, iatol=1e-6, irtol=1e-6, iitmax=500)
    SheafSDP.init_kkt!(admm_wrk, admm_set, A_admm)
    fill!(x_admm_perm, 0); fill!(y_admm_perm, 0)
    fill!(admm_wrk.z, 0); fill!(admm_wrk.u, 0)
    t = @elapsed niter = SheafSDP.solve_kkt!(admm_wrk, admm_set, x_admm_perm, y_admm_perm, A_admm, B_admm, f_admm, g)
    println("  relax=$relax: α=$(round(admm_wrk.α[], sigdigits=4)), time=$(round(t*1000, digits=2)) ms, iters=$niter")
end
println()

admm_set = SheafSDP.ADMMSettings{Float64}(aaug=2e6, atol=1e-10, rtol=1e-10, itmax=2000, iatol=1e-6, irtol=1e-6, iitmax=500)

# Initialize both
SheafSDP.init_kkt!(uzw_wrk, uzw_set, A_uzw)
SheafSDP.init_kkt!(admm_wrk, admm_set, A_admm)

# Warmup
SheafSDP.solve_kkt!(uzw_wrk, uzw_set, x_uzw_perm, y_uzw_perm, A_uzw, B_uzw, f_uzw, g)
SheafSDP.solve_kkt!(admm_wrk, admm_set, x_admm_perm, y_admm_perm, A_admm, B_admm, f_admm, g)

# Reset outputs and ADMM workspace
fill!(x_uzw_perm, 0); fill!(y_uzw_perm, 0)
fill!(x_admm_perm, 0); fill!(y_admm_perm, 0)
fill!(admm_wrk.z, 0); fill!(admm_wrk.u, 0)

# Timed Uzawa
println("Uzawa:")
println("  α = $(uzw_wrk.α[])")
t_uzw = @elapsed niter_uzw = SheafSDP.solve_kkt!(uzw_wrk, uzw_set, x_uzw_perm, y_uzw_perm, A_uzw, B_uzw, f_uzw, g)
println("  time:       $(round(t_uzw * 1000, digits=3)) ms")
println("  iterations: $niter_uzw")

# Timed ADMM
println()
println("ADMM:")
println("  α = $(admm_wrk.α[]), τ = $(admm_wrk.τ[])")
t_admm = @elapsed niter_admm = SheafSDP.solve_kkt!(admm_wrk, admm_set, x_admm_perm, y_admm_perm, A_admm, B_admm, f_admm, g)
println("  time:       $(round(t_admm * 1000, digits=3)) ms")
println("  iterations: $niter_admm")

# Unpermute Uzawa results for comparison
SheafSDP.blockinvpermute!(x_uzw, x_uzw_perm, B, perm_uzw)
copyto!(y_uzw, y_uzw_perm)
# ADMM results don't need unpermuting (identity perm)
copyto!(x_admm, x_admm_perm)
copyto!(y_admm, y_admm_perm)

# Check accuracy
println()
println("Accuracy:")

# KKT residuals: A x + Bᵀ y = f, B x = g
function kkt_residuals(A, B, x, y, f, g)
    # r1 = f - A x - Bᵀ y
    Ax = zeros(length(x))
    for v in vtxs(A)
        rv = colrange(A, v)
        Av = block(A, v, v, v)
        Ax[rv] .= Av * x[rv]
    end
    Bty = B' * y
    r1 = f - Ax - Bty
    # r2 = g - B x
    r2 = g - B * x
    return norm(r1), norm(r2), norm(Ax), norm(Bty)
end

println("  ‖x_uzw‖  = $(norm(x_uzw)),  ‖y_uzw‖  = $(norm(y_uzw))")
println("  ‖x_admm‖ = $(norm(x_admm)), ‖y_admm‖ = $(norm(y_admm))")
println()

r1_uzw, r2_uzw, ax_uzw, bty_uzw = kkt_residuals(A, B, x_uzw, y_uzw, f, g)
r1_admm, r2_admm, ax_admm, bty_admm = kkt_residuals(A, B, x_admm, y_admm, f, g)

println("  ‖f‖ = $(norm(f)), ‖g‖ = $(norm(g))")
println()
println("  Uzawa:  ‖f - Ax - B'y‖ = $r1_uzw, ‖g - Bx‖ = $r2_uzw")
println("  ADMM:   ‖f - Ax - B'y‖ = $r1_admm, ‖g - Bx‖ = $r2_admm")

# Solution difference
println()
println("Agreement:")
println("  ‖x_uzw - x_admm‖ / ‖x‖ = $(norm(x_uzw - x_admm) / norm(x_admm))")
println("  ‖y_uzw - y_admm‖ / ‖y‖ = $(norm(y_uzw - y_admm) / norm(y_admm))")

# Summary
println()
println("Summary:")
println("  Uzawa: $(round(t_uzw * 1000, digits=2)) ms, $niter_uzw iters")
println("  ADMM:  $(round(t_admm * 1000, digits=2)) ms, $niter_admm iters")
