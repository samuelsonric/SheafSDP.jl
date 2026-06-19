#
# Compare ADMM (with different preconditioners) vs Uzawa on ECQP
#
using SheafSDP
using LinearAlgebra
using Random
using BlockSparseArrays: vtxs, colrange, block, halfselectvtxs

Random.seed!(42)

# Problem size
nv = 500
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
A = SheafSDP.allocblockdiag(B)
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

# KKT residuals helper
function kkt_residuals(A, B, x, y, f, g)
    Ax = zeros(length(x))
    for v in vtxs(A)
        rv = colrange(A, v)
        Av = block(A, v, v, v)
        Ax[rv] .= Av * x[rv]
    end
    Bty = B' * y
    r1 = f - Ax - Bty
    r2 = g - B * x
    return norm(r1), norm(r2)
end

# Test runner
function test_method(name, set, B, A, f, g)
    perm, B_perm, wrk = SheafSDP.make_kkt(set, B)
    A_perm = halfselectvtxs(halfselectvtxs(A, perm), perm)
    f_perm = SheafSDP.blockpermute(f, B, perm)

    n, m = size(B, 2), size(B, 1)
    x_perm = zeros(n)
    y_perm = zeros(m)

    # Warmup
    SheafSDP.init_kkt!(wrk, set, A_perm)
    SheafSDP.solve_kkt!(wrk, set, x_perm, y_perm, A_perm, B_perm, f_perm, g)

    # Reset
    fill!(x_perm, 0); fill!(y_perm, 0)
    if hasproperty(wrk, :z)
        fill!(wrk.z, 0); fill!(wrk.u, 0)
    end

    # Timed run
    SheafSDP.init_kkt!(wrk, set, A_perm)
    t = @elapsed niter = SheafSDP.solve_kkt!(wrk, set, x_perm, y_perm, A_perm, B_perm, f_perm, g)

    # Unpermute
    x = zeros(n)
    SheafSDP.blockinvpermute!(x, x_perm, B, perm)
    y = copy(y_perm)

    # Residuals
    r1, r2 = kkt_residuals(A, B, x, y, f, g)

    return (name=name, time_ms=t*1000, iters=niter, res_primal=r1, res_dual=r2)
end

println("="^70)
println("ADMM+IChol raug sweep (with rreg=0)")
println("="^70)
println()

raugs = [100.0, 500.0, 1000.0, 2000.0, 5000.0]
println(rpad("raug", 10), rpad("Time(ms)", 12), rpad("Iters", 8))
for raug in raugs
    set = ADMMSettings{Float64, ICholSettings{Float64}}(
        prec=ICholSettings{Float64}(),
        raug=raug, atol=1e-8, rtol=1e-8, itmax=2000,
        iatol=1e-10, irtol=1e-10, iitmax=500)
    r = test_method("IChol", set, B, A, f, g)
    println(rpad(raug, 10), rpad(round(r.time_ms, digits=1), 12), r.iters)
end
println()

println("="^70)
println("KKT SOLVER COMPARISON (optimal params)")
println("="^70)
println()

results = []

# Uzawa (raug=5000 was optimal)
uzw_set = UzawaSettings{Float64}(raug=5000.0, atol=1e-8, rtol=1e-8, itmax=2000)
push!(results, test_method("Uzawa", uzw_set, B, A, f, g))

# ADMM variants with raug=1000 (optimal for rreg=0)
for (name, Prec) in [("ADMM+NoPrec", NoPrecSettings{Float64}()),
                      ("ADMM+Jacobi", JacobiSettings{Float64}()),
                      ("ADMM+SSOR", SSORSettings{Float64}()),
                      ("ADMM+IChol", ICholSettings{Float64}())]
    set = ADMMSettings{Float64, typeof(Prec)}(
        prec=Prec,
        raug=1000.0, atol=1e-8, rtol=1e-8, itmax=2000,
        iatol=1e-10, irtol=1e-10, iitmax=500)
    push!(results, test_method(name, set, B, A, f, g))
end

println(rpad("Method", 15), rpad("Time(ms)", 12), rpad("Iters", 8), rpad("‖r_primal‖", 14), "‖r_dual‖")
println("-"^70)
for r in results
    println(rpad(r.name, 15),
            rpad(round(r.time_ms, digits=1), 12),
            rpad(r.iters, 8),
            rpad(round(r.res_primal, sigdigits=3), 14),
            round(r.res_dual, sigdigits=3))
end
println()
