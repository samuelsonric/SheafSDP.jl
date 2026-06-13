using Test
using SparseArrays
using LinearAlgebra
using Random
using SheafSDP
using CliqueTrees.Multifrontal: FactorizationWorkspace, DivisionWorkspace
using Krylov: CgWorkspace
using BlockSparseArrays: blocksparse

"""
Build a random cellular sheaf on an Erdős-Rényi graph.
Ensures no isolated vertices.
Returns (src, dst, maps, nv, ne, edges, A_blocks) for use with sheaf().
"""
function random_sheaf_data(nv::Int, p::Float64, dv::Int, de::Int; seed=42)
    Random.seed!(seed)

    # Generate random edges
    edges = Tuple{Int,Int}[]
    for i in 1:nv, j in i+1:nv
        if rand() < p
            push!(edges, (i, j))
        end
    end

    # Ensure no isolated vertices
    has_edge = falses(nv)
    for (u, v) in edges
        has_edge[u] = has_edge[v] = true
    end
    for v in 1:nv
        if !has_edge[v]
            neighbor = v == nv ? 1 : v + 1
            push!(edges, (min(v, neighbor), max(v, neighbor)))
        end
    end
    ne = length(edges)

    # Build restriction map data
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

    # Block diagonal A (one dv × dv SPD block per vertex)
    A_blocks = [let M = randn(dv, dv); M' * M + I end for _ in 1:nv]

    return src, dst, maps, nv, ne, edges, A_blocks
end

@testset "sheaf solvers" begin
    # ~100ms problem size
    nv, p, dv, de = 47, 0.1, 60, 20

    src, dst, maps, nv, ne, edges, A_blocks = random_sheaf_data(nv, p, dv, de)

    println("Graph: $nv vertices, $ne edges")
    println("Stalk dims: vertex=$dv, edge=$de")

    # Build sheaf structure
    P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)

    # Build block diagonal A in permuted coordinates
    A_V = [A_blocks[P.invp[v]] for v in 1:nv]
    A = blocksparse(1:nv, 1:nv, A_V, nv, nv)

    # Dimensions
    n = size(F, 1)
    m = size(B, 1)
    println("Total dims: n=$n, m=$m")

    # Generate test vectors
    Random.seed!(123)
    f = randn(n)
    g = B * randn(n)  # feasible g

    # Sparse versions for residual checks
    A_sp = sparse(A)
    B_sp = sparse(B)

    # Test Direct
    println("\n--- solve_direct_sheaf ---")
    @time x_d, y_d = solve_direct_sheaf(B, A, f, g)
    println("  ||Ax + B'y - f|| = ", norm(A_sp * x_d + B_sp' * y_d - f))
    println("  ||Bx - g||       = ", norm(B_sp * x_d - g))

    @test norm(B_sp * x_d - g) < 1e-6

    # Pre-allocate workspaces
    facwrk = FactorizationWorkspace(F)
    divwrk = DivisionWorkspace(F, 1)
    riwork = RiWorkspace(m, Vector{Float64})

    # Pre-allocate vectors
    x = zeros(n)
    y = zeros(m)
    r = zeros(m)

    # Test Richardson
    println("\n--- solve_kkt! (γ=10.0) ---")
    @time iters_r = solve_kkt!(facwrk, divwrk, riwork, x, y, r, F, L, B, A, f, g; γ=10.0)
    println("Iterations: $iters_r")
    println("  ||Ax + B'y - f|| = ", norm(A_sp * x + B_sp' * y - f))
    println("  ||Bx - g||       = ", norm(B_sp * x - g))

    @test norm(B_sp * x - g) < 1e-6

    # Test Schur CG (rebuild F since Richardson modified it)
    println("\n--- solve_kkt! (γ=1.0) ---")
    P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
    A = blocksparse(1:nv, 1:nv, A_V, nv, nv)

    # Rebuild workspaces for new F
    facwrk = FactorizationWorkspace(F)
    divwrk = DivisionWorkspace(F, 1)
    itrwrk = CgWorkspace(m, m, Vector{Float64})

    @time iters_cg = solve_kkt!(facwrk, divwrk, itrwrk, x, y, r, F, L, B, A, f, g; γ=1.0)
    println("Iterations: $iters_cg")
    println("  ||Ax + B'y - f|| = ", norm(A_sp * x + B_sp' * y - f))
    println("  ||Bx - g||       = ", norm(B_sp * x - g))

    @test norm(B_sp * x - g) < 1e-6
end
