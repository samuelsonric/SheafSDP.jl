using AppleAccelerate
using SheafSDP
using SheafSDP: OPTIMAL, NEAR_OPTIMAL
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse, colrange, rowrange, block
using SparseArrays: sparse
using JuMP, MosekTools
using Printf

#
# Systematic exploration of SheafSDP vs Mosek for SOC problems
# Testing both linear (Q=0) and quadratic (Q≠0) objectives
#

#=============================================================================
  Graph topologies (reused from SDP exploration)
=============================================================================#

function path_edges(N)
    return [(i, i+1) for i in 1:N-1]
end

function star_edges(N)
    return [(1, i) for i in 2:N]
end

function cycle_edges(N)
    edges = [(i, i+1) for i in 1:N-1]
    push!(edges, (N, 1))
    return edges
end

function binary_tree_edges(N)
    edges = Tuple{Int,Int}[]
    for i in 1:N
        left, right = 2i, 2i + 1
        left <= N && push!(edges, (i, left))
        right <= N && push!(edges, (i, right))
    end
    return edges
end

function random_tree_edges(N; seed=123)
    Random.seed!(seed)
    N <= 2 && return N == 2 ? [(1, 2)] : Tuple{Int,Int}[]

    prufer = [rand(1:N) for _ in 1:(N-2)]
    degree = ones(Int, N)
    for node in prufer
        degree[node] += 1
    end

    edges = Tuple{Int,Int}[]
    prufer_idx = 1

    for _ in 1:(N-2)
        leaf = findfirst(==(1), degree)
        neighbor = prufer[prufer_idx]
        push!(edges, (min(leaf, neighbor), max(leaf, neighbor)))
        degree[leaf] -= 1
        degree[neighbor] -= 1
        prufer_idx += 1
    end

    remaining = findall(==(1), degree)
    length(remaining) == 2 && push!(edges, (remaining[1], remaining[2]))

    return edges
end

function erdos_renyi_edges(N, p; seed=456)
    Random.seed!(seed)
    edges = Set(random_tree_edges(N; seed=seed))
    for i in 1:N, j in (i+1):N
        rand() < p && !((i,j) in edges) && push!(edges, (i, j))
    end
    return collect(edges)
end

#=============================================================================
  SOC Problem builder
=============================================================================#

function build_soc_problem(N, n_v, d_e, edges; use_quadratic=false, quad_weight=0.1, seed=42)
    T = Float64
    ne = length(edges)
    invrt2 = 1 / sqrt(2.0)

    Random.seed!(seed)

    # Generate orthonormal-row restriction maps F_e^i, F_e^j for each edge
    function make_restriction_map(n)
        G = randn(n, d_e)
        Q, _ = qr(G)
        return Matrix(Q)'  # d_e × n
    end

    F = [(make_restriction_map(n_v), make_restriction_map(n_v)) for _ in 1:ne]

    # Realizable target: b = δx₀
    x0 = [randn(n_v) for _ in 1:N]
    b = [F[e][1] * x0[i] - F[e][2] * x0[j] for (e, (i, j)) in enumerate(edges)]

    # Column layout: ζ_i = (s_i; x_i), sp_i, sm_i (box slacks)
    col_ζ(i) = 3 * (i - 1) + 1
    col_sp(i) = 3 * (i - 1) + 2
    col_sm(i) = 3 * (i - 1) + 3

    # Row layout: boxp, boxm per node; coord per edge
    row_boxp(i) = 2 * (i - 1) + 1
    row_boxm(i) = 2 * (i - 1) + 2
    row_coord(e) = 2 * N + e

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]

    x_max = 100.0

    for i in 1:N
        # Extract x from ζ = (s; x), scaled by 1/√2
        extract_x = [zeros(n_v, 1) Matrix(1.0I, n_v, n_v)] .* invrt2

        # box+: x + sp = x_max
        push!(row_ids, row_boxp(i)); push!(col_ids, col_ζ(i)); push!(blocks, extract_x)
        push!(row_ids, row_boxp(i)); push!(col_ids, col_sp(i)); push!(blocks, Matrix{T}(I, n_v, n_v))

        # box-: -x + sm = x_max
        push!(row_ids, row_boxm(i)); push!(col_ids, col_ζ(i)); push!(blocks, -extract_x)
        push!(row_ids, row_boxm(i)); push!(col_ids, col_sm(i)); push!(blocks, Matrix{T}(I, n_v, n_v))
    end

    # Coordination rows: F_i x_i - F_j x_j = b_e
    for (e, (i, j)) in enumerate(edges)
        F_i, F_j = F[e]
        block_i = [zeros(d_e, 1) F_i] .* invrt2
        block_j = [zeros(d_e, 1) F_j] .* invrt2
        push!(row_ids, row_coord(e)); push!(col_ids, col_ζ(i)); push!(blocks, block_i)
        push!(row_ids, row_coord(e)); push!(col_ids, col_ζ(j)); push!(blocks, -block_j)
    end

    B_mat = blocksparse(row_ids, col_ids, blocks)

    # Cost: 1 on head (s_i), scaled by 1/√2
    c_vec = zeros(T, size(B_mat, 2))
    for i in 1:N
        ζ_range = colrange(B_mat, col_ζ(i))
        c_vec[ζ_range[1]] = invrt2
    end

    # RHS
    g = zeros(T, size(B_mat, 1))
    for i in 1:N
        g[rowrange(B_mat, row_boxp(i))] .= x_max
        g[rowrange(B_mat, row_boxm(i))] .= x_max
    end
    for e in 1:ne
        g[rowrange(B_mat, row_coord(e))] .= b[e]
    end

    # Q matrix (quadratic objective)
    Q = SheafSDP.allocblockdiag(B_mat)
    fill!(Q, zero(T))

    if use_quadratic
        # Add quadratic regularization on x (the tail of ζ = (s; x))
        # Q is block diagonal, so we add to the ζ blocks
        for i in 1:N
            Qv = block(Q, col_ζ(i), col_ζ(i), col_ζ(i))
            # ζ = (s; x) with √2 scaling, so x part starts at index 2
            # Add quad_weight to diagonal of x part (skip s at index 1)
            for k in 2:(1 + n_v)
                Qv[k, k] = 2 * quad_weight * (invrt2^2)  # factor of 2 for (1/2)x'Qx, scaled
            end
        end
    end

    # Cones
    cones = Vector{Cone}(undef, 3*N)
    for i in 1:N
        cones[col_ζ(i)] = SheafSDP.SecondOrderCone()
        cones[col_sp(i)] = SheafSDP.PositiveCone()
        cones[col_sm(i)] = SheafSDP.PositiveCone()
    end

    return IPMProblem(c_vec, g, B_mat, Q, cones), F, b, x0
end

#=============================================================================
  Mosek solver for comparison
=============================================================================#

function solve_soc_with_mosek(N, n_v, d_e, edges, F, b; use_quadratic=false, quad_weight=0.1)
    ne = length(edges)
    x_max = 100.0

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables
    x = [@variable(model, [1:n_v]) for _ in 1:N]
    s = @variable(model, [1:N], lower_bound=0)  # epigraph vars

    # SOC constraints: ‖x_i‖₂ ≤ s_i
    for i in 1:N
        @constraint(model, [s[i]; x[i]] in JuMP.SecondOrderCone())
    end

    # Box constraints
    for i in 1:N
        @constraint(model, -x_max .<= x[i] .<= x_max)
    end

    # Coordination constraints: F_i x_i - F_j x_j = b_e
    for (e, (i, j)) in enumerate(edges)
        F_i, F_j = F[e]
        @constraint(model, F_i * x[i] - F_j * x[j] .== b[e])
    end

    # Objective: Σ s (sum of norms) + optional quadratic
    if use_quadratic
        @objective(model, Min, sum(s) + 0.5 * quad_weight * sum(x[i]' * x[i] for i in 1:N))
    else
        @objective(model, Min, sum(s))
    end

    optimize!(model)

    return objective_value(model), solve_time(model)
end

#=============================================================================
  Benchmark runner
=============================================================================#

function find_best_raug_soc(prob, raug_values; feas_tol=1e-6, gap_tol=1e-6, refine_itmax=10)
    best_raug = nothing
    best_time = Inf
    best_result = nothing

    for raug in raug_values
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=raug),
            feas_tol=feas_tol,
            gap_tol=gap_tol,
            itmax=100,
            verbose=false,
            refine_itmax=refine_itmax
        )

        try
            t = @elapsed result = solve(prob, settings)

            if result.status in (OPTIMAL, NEAR_OPTIMAL)
                if t < best_time
                    best_time = t
                    best_raug = raug
                    best_result = result
                end
            end
        catch e
            continue
        end
    end

    return best_raug, best_time, best_result
end

function run_soc_benchmark(config; warmup=false, raug_values=[1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    topology = config.topology
    N = config.N
    n_v = config.n_v  # SOC arm dimension (vertex stalk)
    d_e = config.d_e  # edge stalk dimension
    use_quadratic = get(config, :quadratic, false)
    quad_weight = get(config, :quad_weight, 0.1)

    # Generate edges
    if topology == :path
        edges = path_edges(N)
    elseif topology == :star
        edges = star_edges(N)
    elseif topology == :cycle
        edges = cycle_edges(N)
    elseif topology == :bintree
        edges = binary_tree_edges(N)
    elseif topology == :randtree
        edges = random_tree_edges(N; seed=config.N + config.n_v)
    elseif topology == :erdos
        p = get(config, :p, 0.1)
        edges = erdos_renyi_edges(N, p; seed=config.N + config.n_v)
    else
        error("Unknown topology: $topology")
    end

    prob, F, b, x0 = build_soc_problem(N, n_v, d_e, edges; use_quadratic, quad_weight)

    if warmup
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=1e6),
            feas_tol=1e-4, gap_tol=1e-4, itmax=20, verbose=false
        )
        try; _ = solve(prob, settings); catch; end
        try; _ = solve_soc_with_mosek(N, n_v, d_e, edges, F, b; use_quadratic, quad_weight); catch; end
    end

    # Find best raug
    best_raug, t_sheaf, result = find_best_raug_soc(prob, raug_values)

    if isnothing(best_raug)
        return (
            topology=topology, N=N, n_v=n_v, d_e=d_e, quadratic=use_quadratic,
            nvars=size(prob.B, 2), ncons=size(prob.B, 1), nedges=length(edges),
            t_sheaf=NaN, t_mosek=NaN,
            best_raug=NaN, iters=0, status=:FAILED,
            speedup=NaN, obj_diff=NaN
        )
    end

    # Compute objective (linear + quadratic parts)
    obj_sheaf = dot(prob.c, result.p)
    if use_quadratic
        obj_sheaf += 0.5 * dot(result.p, Symmetric(sparse(prob.Q), :L) * result.p)
    end
    obj_mosek, t_mosek = solve_soc_with_mosek(N, n_v, d_e, edges, F, b; use_quadratic, quad_weight)

    return (
        topology=topology, N=N, n_v=n_v, d_e=d_e, quadratic=use_quadratic,
        nvars=size(prob.B, 2), ncons=size(prob.B, 1), nedges=length(edges),
        t_sheaf=t_sheaf*1000, t_mosek=t_mosek*1000,
        best_raug=best_raug, iters=result.iterations, status=result.status,
        speedup=t_mosek/t_sheaf, obj_diff=abs(obj_sheaf - obj_mosek)
    )
end

#=============================================================================
  QP Problem builder (for comparison: quadratic objective, no cones)
=============================================================================#

function build_qp_problem(N, n_v, d_e, edges; quad_weight=1.0, seed=42)
    T = Float64
    ne = length(edges)

    Random.seed!(seed)

    # Generate orthonormal-row restriction maps
    function make_restriction_map(n)
        G = randn(n, d_e)
        Q, _ = qr(G)
        return Matrix(Q)'
    end

    F = [(make_restriction_map(n_v), make_restriction_map(n_v)) for _ in 1:ne]

    # Realizable target
    x0 = [randn(n_v) for _ in 1:N]
    b = [F[e][1] * x0[i] - F[e][2] * x0[j] for (e, (i, j)) in enumerate(edges)]

    # Column layout: just x_i (no epigraph variables needed)
    col_x(i) = i

    # Row layout: coordination only (no box constraints for simplicity)
    row_coord(e) = e

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]

    # Coordination rows: F_i x_i - F_j x_j = b_e
    for (e, (i, j)) in enumerate(edges)
        F_i, F_j = F[e]
        push!(row_ids, row_coord(e)); push!(col_ids, col_x(i)); push!(blocks, F_i)
        push!(row_ids, row_coord(e)); push!(col_ids, col_x(j)); push!(blocks, -F_j)
    end

    B_mat = blocksparse(row_ids, col_ids, blocks)

    # Linear cost = 0
    c_vec = zeros(T, size(B_mat, 2))

    # RHS
    g = zeros(T, size(B_mat, 1))
    for e in 1:ne
        g[rowrange(B_mat, row_coord(e))] .= b[e]
    end

    # Quadratic objective: (1/2) * quad_weight * ||x||²
    Q = SheafSDP.allocblockdiag(B_mat)
    fill!(Q, zero(T))
    for i in 1:N
        Qv = BlockSparseArrays.block(Q, col_x(i), col_x(i), col_x(i))
        for k in 1:n_v
            Qv[k, k] = 2 * quad_weight  # factor of 2 because objective is (1/2) x'Qx
        end
    end

    # Cones: all CofreeCone (no cone constraint)
    cones = [SheafSDP.CofreeCone() for _ in 1:N]

    return IPMProblem(c_vec, g, B_mat, Q, cones), F, b, x0
end

function solve_qp_with_mosek(N, n_v, d_e, edges, F, b; quad_weight=1.0)
    ne = length(edges)

    model = Model(Mosek.Optimizer)
    set_silent(model)

    x = [@variable(model, [1:n_v]) for _ in 1:N]

    # Coordination constraints
    for (e, (i, j)) in enumerate(edges)
        F_i, F_j = F[e]
        @constraint(model, F_i * x[i] - F_j * x[j] .== b[e])
    end

    # Quadratic objective: (1/2) * quad_weight * ||x||²
    @objective(model, Min, 0.5 * quad_weight * sum(x[i]' * x[i] for i in 1:N))

    optimize!(model)

    return objective_value(model), solve_time(model)
end

function run_qp_benchmark(config; warmup=false, raug_values=[1e6, 1e7, 1e8, 1e9, 1e10])
    topology = config.topology
    N = config.N
    n_v = config.n_v
    d_e = config.d_e

    if topology == :path
        edges = path_edges(N)
    elseif topology == :star
        edges = star_edges(N)
    elseif topology == :cycle
        edges = cycle_edges(N)
    elseif topology == :erdos
        p = get(config, :p, 0.1)
        edges = erdos_renyi_edges(N, p; seed=config.N + config.n_v)
    else
        error("Unknown topology: $topology")
    end

    prob, F, b, x0 = build_qp_problem(N, n_v, d_e, edges)

    if warmup
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=1e8),
            feas_tol=1e-4, gap_tol=1e-4, itmax=20, verbose=false
        )
        try; _ = solve(prob, settings); catch; end
        try; _ = solve_qp_with_mosek(N, n_v, d_e, edges, F, b); catch; end
    end

    # Find best raug
    best_raug = nothing
    best_time = Inf
    best_result = nothing

    for raug in raug_values
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=raug),
            feas_tol=1e-6, gap_tol=1e-6, itmax=100, verbose=false
        )
        try
            t = @elapsed result = solve(prob, settings)
            if result.status in (OPTIMAL, NEAR_OPTIMAL) && t < best_time
                best_time = t
                best_raug = raug
                best_result = result
            end
        catch; end
    end

    if isnothing(best_raug)
        return (
            topology=topology, N=N, n_v=n_v, d_e=d_e, problem=:QP,
            nvars=size(prob.B, 2), ncons=size(prob.B, 1), nedges=length(edges),
            t_sheaf=NaN, t_mosek=NaN, best_raug=NaN, iters=0,
            speedup=NaN, obj_diff=NaN
        )
    end

    # Compute objective
    obj_sheaf = 0.5 * dot(best_result.p, prob.Q * best_result.p)
    obj_mosek, t_mosek = solve_qp_with_mosek(N, n_v, d_e, edges, F, b)

    return (
        topology=topology, N=N, n_v=n_v, d_e=d_e, problem=:QP,
        nvars=size(prob.B, 2), ncons=size(prob.B, 1), nedges=length(edges),
        t_sheaf=best_time*1000, t_mosek=t_mosek*1000,
        best_raug=best_raug, iters=best_result.iterations,
        speedup=t_mosek/best_time, obj_diff=abs(obj_sheaf - obj_mosek)
    )
end

#=============================================================================
  Main exploration
=============================================================================#

function main()
    println("="^110)
    println("SheafSDP vs Mosek: SOC Performance Exploration")
    println("="^110)
    println("\nProblem: sum-of-norms minimization with box constraints and sheaf consensus")
    println("n_v = SOC arm dimension (vertex stalk), d_e = edge stalk dimension")

    # Warmup
    println("\nWarming up...")
    run_soc_benchmark((topology=:path, N=10, n_v=20, d_e=8); warmup=true)

    results = []

    #=========================================================================
      Exploration 1: Regular graphs at various stalk sizes
    =========================================================================#
    println("\n" * "="^110)
    println("EXPLORATION 1: Regular graphs (varying n_v)")
    println("="^110)

    @printf("\n%10s %4s %4s %4s | %5s | %6s %6s | %10s %10s | %8s %5s | %8s\n",
            "topology", "N", "n_v", "d_e", "edges", "nvars", "ncons", "SheafSDP", "Mosek", "raug", "iters", "speedup")
    println("-"^115)

    for topology in [:path, :star, :cycle, :bintree]
        for (n_v, d_e) in [(10, 5), (20, 10), (40, 20)]
            for N in [20, 50]
                config = (topology=topology, N=N, n_v=n_v, d_e=d_e)
                r = run_soc_benchmark(config)
                push!(results, r)

                @printf("%10s %4d %4d %4d | %5d | %6d %6d | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                        r.topology, r.N, r.n_v, r.d_e, r.nedges, r.nvars, r.ncons,
                        r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
            end
        end
        println()
    end

    #=========================================================================
      Exploration 2: Irregular graphs
    =========================================================================#
    println("\n" * "="^110)
    println("EXPLORATION 2: Irregular graphs (n_v=20, d_e=10)")
    println("="^110)

    @printf("\n%10s %4s %4s %4s | %5s | %6s %6s | %10s %10s | %8s %5s | %8s\n",
            "topology", "N", "n_v", "d_e", "edges", "nvars", "ncons", "SheafSDP", "Mosek", "raug", "iters", "speedup")
    println("-"^115)

    for topology in [:randtree, :erdos]
        for N in [20, 50, 100]
            if topology == :erdos
                config = (topology=topology, N=N, n_v=20, d_e=10, p=0.1)
            else
                config = (topology=topology, N=N, n_v=20, d_e=10)
            end
            r = run_soc_benchmark(config)
            push!(results, r)

            @printf("%10s %4d %4d %4d | %5d | %6d %6d | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                    r.topology, r.N, r.n_v, r.d_e, r.nedges, r.nvars, r.ncons,
                    r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
        end
        println()
    end

    # Erdős-Rényi density sweep
    println("Erdős-Rényi with varying edge density (N=50, n_v=20):")
    println("-"^80)
    for p in [0.05, 0.1, 0.2, 0.3]
        config = (topology=:erdos, N=50, n_v=20, d_e=10, p=p)
        r = run_soc_benchmark(config)
        push!(results, r)

        @printf("  p=%.2f: %5d edges | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                p, r.nedges, r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
    end

    #=========================================================================
      Exploration 3: Stalk size sweep
    =========================================================================#
    println("\n" * "="^110)
    println("EXPLORATION 3: Stalk size sweep on path (N=50)")
    println("="^110)

    @printf("\n%10s %4s %4s %4s | %5s | %6s %6s | %10s %10s | %8s %5s | %8s\n",
            "topology", "N", "n_v", "d_e", "edges", "nvars", "ncons", "SheafSDP", "Mosek", "raug", "iters", "speedup")
    println("-"^115)

    for n_v in [5, 10, 20, 30, 50, 80]
        d_e = div(n_v, 2)
        config = (topology=:path, N=50, n_v=n_v, d_e=d_e)
        r = run_soc_benchmark(config)
        push!(results, r)

        @printf("%10s %4d %4d %4d | %5d | %6d %6d | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                r.topology, r.N, r.n_v, r.d_e, r.nedges, r.nvars, r.ncons,
                r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
    end

    #=========================================================================
      Exploration 4: Graph size scaling
    =========================================================================#
    println("\n" * "="^110)
    println("EXPLORATION 4: Graph size scaling (path, n_v=20)")
    println("="^110)

    @printf("\n%10s %4s %4s %4s | %5s | %6s %6s | %10s %10s | %8s %5s | %8s\n",
            "topology", "N", "n_v", "d_e", "edges", "nvars", "ncons", "SheafSDP", "Mosek", "raug", "iters", "speedup")
    println("-"^115)

    for N in [10, 20, 50, 100, 200]
        config = (topology=:path, N=N, n_v=20, d_e=10)
        r = run_soc_benchmark(config)
        push!(results, r)

        @printf("%10s %4d %4d %4d | %5d | %6d %6d | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                r.topology, r.N, r.n_v, r.d_e, r.nedges, r.nvars, r.ncons,
                r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
    end

    #=========================================================================
      Exploration 5: Linear vs Quadratic objective
    =========================================================================#
    println("\n" * "="^110)
    println("EXPLORATION 5: Linear vs Quadratic objective (SOC + quadratic regularization)")
    println("="^110)

    @printf("\n%10s %4s %4s %4s %5s | %5s | %6s %6s | %10s %10s | %8s %5s | %8s\n",
            "topology", "N", "n_v", "d_e", "quad", "edges", "nvars", "ncons", "SheafSDP", "Mosek", "raug", "iters", "speedup")
    println("-"^120)

    for topology in [:path, :star, :erdos]
        for use_quad in [false, true]
            if topology == :erdos
                config = (topology=topology, N=50, n_v=20, d_e=10, p=0.15, quadratic=use_quad, quad_weight=0.1)
            else
                config = (topology=topology, N=50, n_v=20, d_e=10, quadratic=use_quad, quad_weight=0.1)
            end
            r = run_soc_benchmark(config)
            push!(results, r)

            quad_str = r.quadratic ? "yes" : "no"
            @printf("%10s %4d %4d %4d %5s | %5d | %6d %6d | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                    r.topology, r.N, r.n_v, r.d_e, quad_str, r.nedges, r.nvars, r.ncons,
                    r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
        end
        println()
    end

    #=========================================================================
      Summary
    =========================================================================#
    println("\n" * "="^110)
    println("SUMMARY")
    println("="^110)

    valid = filter(r -> !isnan(r.speedup), results)
    wins = count(r -> r.speedup > 1.0, valid)
    losses = count(r -> r.speedup < 1.0, valid)
    failures = count(r -> isnan(r.speedup), results)

    println("\nSheafSDP wins: $wins / $(length(valid))")
    println("Mosek wins: $losses / $(length(valid))")
    println("Failures: $failures")

    if !isempty(valid)
        best = argmax(r -> r.speedup, valid)
        worst = argmin(r -> r.speedup, valid)

        println("\nBest speedup: $(round(best.speedup, digits=2))x")
        println("  Config: $(best.topology), N=$(best.N), n_v=$(best.n_v), d_e=$(best.d_e), edges=$(best.nedges)")

        println("\nWorst speedup: $(round(worst.speedup, digits=2))x")
        println("  Config: $(worst.topology), N=$(worst.N), n_v=$(worst.n_v), d_e=$(worst.d_e), edges=$(worst.nedges)")

        # Breakdown by topology
        println("\n" * "-"^60)
        println("By topology:")
        for topo in unique(r.topology for r in valid)
            topo_results = filter(r -> r.topology == topo, valid)
            avg_speedup = sum(r.speedup for r in topo_results) / length(topo_results)
            win_rate = count(r -> r.speedup > 1.0, topo_results) / length(topo_results)
            @printf("  %10s: avg %.2fx, win rate %.0f%%\n", topo, avg_speedup, 100*win_rate)
        end

        # Breakdown by stalk size
        println("\n" * "-"^60)
        println("By SOC arm size (n_v):")
        nv_groups = [(0, 15, "small"), (15, 35, "medium"), (35, 100, "large")]
        for (lo, hi, label) in nv_groups
            group = filter(r -> lo < r.n_v <= hi, valid)
            if !isempty(group)
                avg_speedup = sum(r.speedup for r in group) / length(group)
                win_rate = count(r -> r.speedup > 1.0, group) / length(group)
                @printf("  n_v %3d-%3d (%6s): avg %.2fx, win rate %.0f%%\n", lo, hi, label, avg_speedup, 100*win_rate)
            end
        end
    end

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
