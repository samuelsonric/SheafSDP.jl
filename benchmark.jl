using SheafSDP
using SparseArrays
using LinearAlgebra
using Random
using SheafSDP: trinum, triroot, svec!
using BlockSparseArrays: vtxs, colrange, ncols

"""
Build and solve a random sheaf SDP.

Parameters:
- nv: number of vertices
- dv: dimension of vertex stalks
- de: dimension of edge stalks
- density: edge probability (default 0.3)
"""
function build_and_solve(nv, dv, de; density=0.3, seed=42, verbose=false)
    Random.seed!(seed)

    # Generate random graph
    edges = Tuple{Int,Int}[]
    for i in 1:nv
        for j in i+1:nv
            if rand() < density
                push!(edges, (i, j))
            end
        end
    end
    ne = length(edges)

    # Build restriction maps
    src = Int[]
    dst = Int[]
    maps = Matrix{Float64}[]
    for (e_idx, (u, v)) in enumerate(edges)
        push!(src, u)
        push!(dst, e_idx)
        push!(maps, randn(de, dv))
        push!(src, v)
        push!(dst, e_idx)
        push!(maps, randn(de, dv))
    end

    # Build sheaf structure
    P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
    B_sp = sparse(B)

    n = size(F, 1)
    m = size(B, 1)

    # Build SPD initial point
    p = zeros(n)
    d = zeros(n)
    for v in vtxs(B)
        r = colrange(B, v)
        n_v = ncols(B, v)
        d_v = triroot(n_v)

        A = randn(d_v, d_v)
        P_v = A * A' + I
        A = randn(d_v, d_v)
        D_v = A * A' + I

        svec!(view(p, r), P_v, Val(:L))
        svec!(view(d, r), D_v, Val(:L))
    end

    # Feasible problem data
    y = randn(m)
    c = B_sp' * y + d
    g = B_sp * p

    if verbose
        println("Sheaf SDP: nv=$nv, dv=$dv, de=$de")
        println("  n=$n (primal dim), m=$m (dual dim), $ne edges")
    end

    # Solve
    t = @elapsed result = solve!(p, d, y, c, g, B, B_sp, F, L;
                                  ε_feas=1e-8, ε_μ=1e-8, max_iter=100, verbose=false)

    if verbose
        println("  time: $(round(t, digits=2))s, iters: $(result.iterations), converged: $(result.converged)")
        println("  final μ: $(result.μ_history[end])")
    end

    return result, t
end

# Warmup
println("Warming up...")
build_and_solve(30, 4, 3)
println()

# Main benchmark: ~2-3 second problem
# nv=400 vertices, dv=10 dimensional stalks, de=6 dimensional edge stalks
println("Running benchmark SDP...")
result, t = build_and_solve(400, 10, 6; verbose=true)
println()
println("Solve time: $(round(t, digits=2)) seconds")
