#
# Compare preconditioners: Jacobi, SSOR, IChol
#
using SheafSDP
using SheafSDP: Jacobi, SSOR, IChol, weightedgraph
using LinearAlgebra
using Random

using CliqueTrees: RCM
using CliqueTrees.Multifrontal: symbolic
using BlockSparseArrays: selectvtxs

Random.seed!(42)

function build_problem(nv, dv, de, density)
    edges = Tuple{Int,Int}[]
    for i in 1:nv, j in i+1:nv
        rand() < density && push!(edges, (i, j))
    end
    # Ensure connected
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

    src, dst, maps = Int[], Int[], Matrix{Float64}[]
    for (u, v) in edges
        push!(src, u); push!(dst, v); push!(maps, randn(de, dv))
        push!(src, v); push!(dst, u); push!(maps, randn(de, dv))
    end

    B = sheaf(src, dst, maps)
    return B, length(edges)
end

# Simple preconditioned CG
function pcg(A, b, P; atol=1e-10, rtol=1e-10, itmax=1000)
    n = length(b)
    x = zeros(n)
    r = copy(b)
    z = similar(b)
    ldiv!(z, P, r)
    p = copy(z)
    rz = dot(r, z)

    bnorm = norm(b)
    tol = atol + rtol * bnorm

    for k in 1:itmax
        Ap = A * p
        α = rz / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap

        rnorm = norm(r)
        rnorm < tol && return x, k, rnorm

        ldiv!(z, P, r)
        rz_new = dot(r, z)
        β = rz_new / rz
        rz = rz_new
        p .= z .+ β .* p
    end

    return x, itmax, norm(r)
end

function test_preconditioner(name, P, A, b, x_true; atol=1e-8, rtol=1e-8, itmax=1000)
    n = length(b)

    # Measure apply time (average over several calls)
    y = similar(b)
    ldiv!(y, P, b)  # warmup
    napply = 20
    t_apply = @elapsed for _ in 1:napply
        ldiv!(y, P, b)
    end
    t_apply /= napply

    # Solve with PCG
    x, iters, res = pcg(A, b, P; atol, rtol, itmax)

    # Compute relative error
    rel_err = norm(x - x_true) / norm(x_true)

    return (
        name = name,
        apply_ms = t_apply * 1000,
        iterations = iters,
        converged = iters < itmax,
        rel_error = rel_err,
        residual = res
    )
end

function run_comparison(nv, dv, de, density; atol=1e-8, rtol=1e-8)
    println("="^60)
    println("Problem: nv=$nv, dv=$dv, de=$de, density=$density")

    B, ne = build_problem(nv, dv, de, density)
    n, m = size(B, 2), size(B, 1)
    println("  n=$n, m=$m, edges=$ne")

    # Apply fill-reducing ordering
    weights, graph = weightedgraph(B)
    P, Q, S = symbolic(weights, graph; alg=RCM())
    perm = P.perm
    B = selectvtxs(B, perm)

    # Form B'B
    A = B' * B

    # RHS in range of A (so system is consistent)
    x0 = randn(n)
    b = A * x0  # b = B'B x0, so b is in range(B'B)

    results = []

    # Jacobi
    print("  Jacobi:  building...")
    t_setup = @elapsed P_jac = Jacobi(B; α=0.1)
    println(" $(round(t_setup*1000, digits=2)) ms")
    r = test_preconditioner("Jacobi", P_jac, A, b, x0; atol, rtol)
    push!(results, (r..., setup_ms = t_setup * 1000))

    # SSOR
    print("  SSOR:    building...")
    t_setup = @elapsed P_ssor = SSOR(B; α=0.1)
    println(" $(round(t_setup*1000, digits=2)) ms")
    r = test_preconditioner("SSOR", P_ssor, A, b, x0; atol, rtol)
    push!(results, (r..., setup_ms = t_setup * 1000))

    # IChol
    print("  IChol:   building...")
    t_setup = @elapsed P_ic = IChol(B; α=0.1)
    println(" $(round(t_setup*1000, digits=2)) ms")
    r = test_preconditioner("IChol", P_ic, A, b, x0; atol, rtol)
    push!(results, (r..., setup_ms = t_setup * 1000))

    # Print results
    println()
    println("  Results:")
    println("  ", "-"^56)
    println("  ", rpad("Name", 8), rpad("Setup(ms)", 12), rpad("Apply(ms)", 12),
            rpad("Iters", 8), rpad("RelErr", 12))
    println("  ", "-"^56)
    for r in results
        println("  ", rpad(r.name, 8),
                rpad(round(r.setup_ms, digits=2), 12),
                rpad(round(r.apply_ms, digits=4), 12),
                rpad(r.iterations, 8),
                rpad(round(r.rel_error, sigdigits=3), 12))
    end
    println()

    return results
end

# Run comparisons on different problem sizes
println("\n", "="^60)
println("PRECONDITIONER COMPARISON")
println("="^60, "\n")

# Small problem
run_comparison(20, 6, 4, 0.15)

# Medium problem
run_comparison(50, 8, 5, 0.10)

# Larger problem
run_comparison(100, 6, 4, 0.08)
