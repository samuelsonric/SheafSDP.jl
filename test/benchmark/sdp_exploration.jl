using AppleAccelerate
using SheafSDP
using SheafSDP: svec!, smat!, roottwo, OPTIMAL, NEAR_OPTIMAL
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse, colrange, rowrange
using JuMP, MosekTools
using Printf

#
# Systematic exploration of SheafSDP vs Mosek performance
# across different problem configurations
#

function svecdim(n)
    return div(n * (n + 1), 2)
end

# rectangular symmetric Kronecker product
function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C)
    α = roottwo(T)
    H = zeros(T, svecdim(d), svecdim(n))
    tkl = 1

    @inbounds for l in 1:n
        tab = 0
        for b in 1:d
            Cbl = C[b, l]
            tab += 1
            H[tab, tkl] = Cbl^2
            for a in b + 1:d
                tab += 1
                H[tab, tkl] = α * C[a, l] * Cbl
            end
        end
        for k in l + 1:n
            tkl += 1
            tab = 0
            for b in 1:d
                Cbk = C[b, k]
                Cbl = C[b, l]
                tab += 1
                H[tab, tkl] = α * Cbk * Cbl
                for a in b + 1:d
                    tab += 1
                    H[tab, tkl] = C[a, k] * Cbl + C[a, l] * Cbk
                end
            end
        end
        tkl += 1
    end
    return H
end

# Build the passivity LMI operator
function passivity_lmi_operator(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                 C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    m = size(B, 2)
    nm = n + m
    sv_G = svecdim(n)
    sv_D = svecdim(nm)

    L = zeros(T, sv_D, sv_G)
    d0 = zeros(T, sv_D)
    G = zeros(T, n, n)
    M = zeros(T, nm, nm)
    v = zeros(T, sv_D)

    for k in 1:sv_G
        fill!(G, zero(T))
        smat!(G, setindex!(zeros(T, sv_G), one(T), k))
        for i in 1:n, j in 1:i-1
            G[j, i] = G[i, j]
        end
        M[1:n, 1:n] .= A * G .+ G * A'
        M[1:n, n+1:nm] .= -G * C'
        M[n+1:nm, 1:n] .= -C * G
        M[n+1:nm, n+1:nm] .= zero(T)
        svec!(v, M)
        L[:, k] .= v
    end

    fill!(M, zero(T))
    M[1:n, n+1:nm] .= B
    M[n+1:nm, 1:n] .= B'
    M[n+1:nm, n+1:nm] .= -(D .+ D')
    svec!(d0, M)

    return L, d0
end

# Generate a stable passive SISO system
function random_passive_system(n::Int, rng=Random.default_rng())
    Q = randn(rng, n, n)
    Q = Q'Q + I
    A = -Q
    B = randn(rng, n, 1)
    C = B'
    D = fill(1.0 + abs(randn(rng)), 1, 1)
    return A, B, C, D
end

#=============================================================================
  Graph topologies
=============================================================================#

function path_edges(N)
    return [(i, i+1) for i in 1:N-1]
end

function star_edges(N)
    # Node 1 is center, connects to all others
    return [(1, i) for i in 2:N]
end

function cycle_edges(N)
    edges = [(i, i+1) for i in 1:N-1]
    push!(edges, (N, 1))
    return edges
end

function grid_edges(N)
    # 2D grid: find closest square
    rows = isqrt(N)
    cols = div(N, rows)
    actual_N = rows * cols

    edges = Tuple{Int,Int}[]
    for i in 1:rows
        for j in 1:cols
            idx = (i-1)*cols + j
            # right neighbor
            if j < cols
                push!(edges, (idx, idx+1))
            end
            # down neighbor
            if i < rows
                push!(edges, (idx, idx+cols))
            end
        end
    end
    return edges, actual_N
end

function binary_tree_edges(N)
    # Binary tree: node i has children 2i and 2i+1
    edges = Tuple{Int,Int}[]
    for i in 1:N
        left = 2i
        right = 2i + 1
        if left <= N
            push!(edges, (i, left))
        end
        if right <= N
            push!(edges, (i, right))
        end
    end
    return edges
end

function random_tree_edges(N; seed=123)
    # Random spanning tree via random Prüfer sequence
    Random.seed!(seed)
    if N <= 2
        return N == 2 ? [(1, 2)] : Tuple{Int,Int}[]
    end

    # Generate random Prüfer sequence
    prufer = [rand(1:N) for _ in 1:(N-2)]

    # Decode Prüfer sequence to edges
    degree = ones(Int, N)
    for node in prufer
        degree[node] += 1
    end

    edges = Tuple{Int,Int}[]
    prufer_idx = 1

    for _ in 1:(N-2)
        # Find smallest leaf (degree 1)
        leaf = findfirst(==(1), degree)
        neighbor = prufer[prufer_idx]
        push!(edges, (min(leaf, neighbor), max(leaf, neighbor)))
        degree[leaf] -= 1
        degree[neighbor] -= 1
        prufer_idx += 1
    end

    # Connect last two nodes with degree 1
    remaining = findall(==(1), degree)
    if length(remaining) == 2
        push!(edges, (remaining[1], remaining[2]))
    end

    return edges
end

function erdos_renyi_edges(N, p; seed=456)
    # Erdős-Rényi random graph G(N, p)
    # Then take a spanning tree to ensure connectivity, plus some extra edges
    Random.seed!(seed)

    # Start with random spanning tree for connectivity
    edges = Set(random_tree_edges(N; seed=seed))

    # Add random edges with probability p
    for i in 1:N
        for j in (i+1):N
            if rand() < p && !((i,j) in edges)
                push!(edges, (i, j))
            end
        end
    end

    return collect(edges)
end

#=============================================================================
  Problem builders
=============================================================================#

function build_passivity_problem(N, n_i, m_i, d_e, edges; seed=42)
    T = Float64
    n_edges = length(edges)

    Random.seed!(seed)

    base_system = random_passive_system(n_i)
    systems = [base_system for _ in 1:N]

    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for _ in 1:n_edges
        C = zeros(T, d_e, n_i)
        for k in 1:min(d_e, n_i)
            C[k, k] = 1.0
        end
        push!(interface_maps, (copy(C), copy(C)))
    end

    sv_G = svecdim(n_i)
    sv_S = svecdim(n_i + m_i)
    sv_edge = svecdim(d_e)

    col_G(i) = 2*(i-1) + 1
    col_S(i) = 2*(i-1) + 2
    row_diss(i) = i
    row_agree(e) = N + e

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]
    g_vec = T[]

    for i in 1:N
        A, B, C, D = systems[i]
        L, d0 = passivity_lmi_operator(A, B, C, D)

        push!(row_ids, row_diss(i)); push!(col_ids, col_S(i)); push!(blocks, Matrix{T}(I, sv_S, sv_S))
        push!(row_ids, row_diss(i)); push!(col_ids, col_G(i)); push!(blocks, L)
        append!(g_vec, -d0)
    end

    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(i)); push!(blocks, skronr(C_i))
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(j)); push!(blocks, -skronr(C_j))
        append!(g_vec, zeros(T, sv_edge))
    end

    B_mat = blocksparse(row_ids, col_ids, blocks)

    c_vec = zeros(T, size(B_mat, 2))
    I_n = Matrix{T}(I, n_i, n_i)
    svec_I = zeros(T, sv_G)
    svec!(svec_I, I_n)
    for i in 1:N
        c_vec[colrange(B_mat, col_G(i))] .= svec_I
    end

    Q = SheafSDP.allocblockdiag(B_mat)
    fill!(Q, zero(T))

    cones = Vector{AbstractCone}(undef, 2N)
    for i in 1:N
        cones[col_G(i)] = SheafSDP.SemidefiniteCone()
        cones[col_S(i)] = SheafSDP.SemidefiniteCone()
    end

    return IPMProblem(Q, B_mat, c_vec, g_vec, cones), systems, interface_maps
end

#=============================================================================
  Mosek solver for comparison
=============================================================================#

function solve_with_mosek(systems, interface_maps, n_i, m_i, d_e, edges)
    N = length(systems)

    model = Model(Mosek.Optimizer)
    set_silent(model)

    G = [@variable(model, [1:n_i, 1:n_i] in PSDCone()) for _ in 1:N]

    for i in 1:N
        A, B, C, D = systems[i]
        nm = n_i + m_i
        Gi = G[i]

        TL = @expression(model, [a=1:n_i, b=1:n_i],
            -sum(A[a,k]*Gi[k,b] + Gi[a,k]*A[b,k] for k in 1:n_i))
        TR = @expression(model, [a=1:n_i, b=1:m_i],
            sum(Gi[a,k]*C[b,k] for k in 1:n_i) - B[a,b])
        BL = @expression(model, [a=1:m_i, b=1:n_i],
            sum(C[a,k]*Gi[k,b] for k in 1:n_i) - B[b,a])
        BR = @expression(model, [a=1:m_i, b=1:m_i], D[a,b] + D[b,a])

        M = [TL TR; BL BR]
        @constraint(model, Symmetric(M) in PSDCone())
    end

    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        for a in 1:d_e, b in 1:d_e
            lhs = sum(C_i[a,k] * G[i][k,l] * C_i[b,l] for k in 1:n_i, l in 1:n_i)
            rhs = sum(C_j[a,k] * G[j][k,l] * C_j[b,l] for k in 1:n_i, l in 1:n_i)
            @constraint(model, lhs == rhs)
        end
    end

    @objective(model, Min, sum(G[i][k,k] for i in 1:N, k in 1:n_i))
    optimize!(model)

    return objective_value(model), solve_time(model)
end

#=============================================================================
  Benchmark runner
=============================================================================#

function find_best_raug(prob, raug_values; feas_tol=1e-6, gap_tol=1e-6, refine_itmax=10)
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
            # Numerical failure - skip this raug
            continue
        end
    end

    return best_raug, best_time, best_result
end

function run_benchmark(config; warmup=false, raug_values=[1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    topology = config.topology
    N = config.N
    n_i = config.n_i
    d_e = config.d_e
    m_i = 1

    # Generate edges based on topology
    if topology == :path
        edges = path_edges(N)
    elseif topology == :star
        edges = star_edges(N)
    elseif topology == :cycle
        edges = cycle_edges(N)
    elseif topology == :grid
        edges, N = grid_edges(N)  # N may be adjusted
    elseif topology == :bintree
        edges = binary_tree_edges(N)
    elseif topology == :randtree
        edges = random_tree_edges(N; seed=config.N + config.n_i)
    elseif topology == :erdos
        p = get(config, :p, 0.1)
        edges = erdos_renyi_edges(N, p; seed=config.N + config.n_i)
    else
        error("Unknown topology: $topology")
    end

    prob, systems, interface_maps = build_passivity_problem(N, n_i, m_i, d_e, edges)

    if warmup
        # Quick warmup solve
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=1e6),
            feas_tol=1e-4, gap_tol=1e-4, itmax=20, verbose=false
        )
        try
            _ = solve(prob, settings)
        catch
        end
        try
            _ = solve_with_mosek(systems, interface_maps, n_i, m_i, d_e, edges)
        catch
        end
    end

    # Find best raug for SheafSDP
    best_raug, t_sheaf, result = find_best_raug(prob, raug_values)

    if isnothing(best_raug)
        return (
            topology=topology, N=N, n_i=n_i, d_e=d_e,
            nvars=size(prob.B, 2), ncons=size(prob.B, 1), nedges=length(edges),
            sv_G=svecdim(n_i), sv_S=svecdim(n_i+m_i),
            t_sheaf=NaN, t_mosek=NaN,
            best_raug=NaN, iters=0, status=:FAILED,
            speedup=NaN, obj_diff=NaN
        )
    end

    obj_sheaf = dot(prob.c, result.p)

    # Solve with Mosek
    obj_mosek, t_mosek = solve_with_mosek(systems, interface_maps, n_i, m_i, d_e, edges)

    return (
        topology=topology, N=N, n_i=n_i, d_e=d_e,
        nvars=size(prob.B, 2), ncons=size(prob.B, 1), nedges=length(edges),
        sv_G=svecdim(n_i), sv_S=svecdim(n_i+m_i),
        t_sheaf=t_sheaf*1000, t_mosek=t_mosek*1000,
        best_raug=best_raug, iters=result.ipm_niter, status=result.status,
        speedup=t_mosek/t_sheaf, obj_diff=abs(obj_sheaf - obj_mosek)
    )
end

#=============================================================================
  Main exploration
=============================================================================#

function main()
    println("="^110)
    println("SheafSDP vs Mosek: Systematic Performance Exploration")
    println("="^110)
    println("\nConstraint: sv_G ≤ 100 (n_i ≤ 13) so graphs can have real structure")
    println("sv_G = n_i*(n_i+1)/2, d_e ≈ n_i/2")

    # Warmup
    println("\nWarming up...")
    run_benchmark((topology=:path, N=10, n_i=9, d_e=5); warmup=true)

    results = []

    #=========================================================================
      Exploration 1: Regular graphs at sweet spot (sv_G ≈ 45)
    =========================================================================#
    println("\n" * "="^110)
    println("EXPLORATION 1: Regular graphs (sv_G=45, n_i=9)")
    println("="^110)

    @printf("\n%10s %4s %4s %4s | %5s %5s | %6s %6s | %10s %10s | %8s %5s | %8s\n",
            "topology", "N", "n_i", "d_e", "sv_G", "edges", "nvars", "ncons", "SheafSDP", "Mosek", "raug", "iters", "speedup")
    println("-"^115)

    for topology in [:path, :cycle, :star, :bintree, :grid]
        for N in [20, 50, 100]
            config = (topology=topology, N=N, n_i=9, d_e=5)
            r = run_benchmark(config)
            push!(results, r)

            @printf("%10s %4d %4d %4d | %5d %5d | %6d %6d | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                    r.topology, r.N, r.n_i, r.d_e, r.sv_G, r.nedges, r.nvars, r.ncons,
                    r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
        end
        println()
    end

    #=========================================================================
      Exploration 2: Irregular graphs (random trees, Erdős-Rényi)
    =========================================================================#
    println("\n" * "="^110)
    println("EXPLORATION 2: Irregular graphs (sv_G=45, n_i=9)")
    println("="^110)

    @printf("\n%10s %4s %4s %4s | %5s %5s | %6s %6s | %10s %10s | %8s %5s | %8s\n",
            "topology", "N", "n_i", "d_e", "sv_G", "edges", "nvars", "ncons", "SheafSDP", "Mosek", "raug", "iters", "speedup")
    println("-"^115)

    for topology in [:randtree, :erdos]
        for N in [20, 50, 100]
            if topology == :erdos
                config = (topology=topology, N=N, n_i=9, d_e=5, p=0.1)
            else
                config = (topology=topology, N=N, n_i=9, d_e=5)
            end
            r = run_benchmark(config)
            push!(results, r)

            @printf("%10s %4d %4d %4d | %5d %5d | %6d %6d | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                    r.topology, r.N, r.n_i, r.d_e, r.sv_G, r.nedges, r.nvars, r.ncons,
                    r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
        end
        println()
    end

    # Try different edge densities for Erdős-Rényi
    println("Erdős-Rényi with varying edge density (N=50, sv_G=45):")
    println("-"^80)
    for p in [0.05, 0.1, 0.2, 0.3]
        config = (topology=:erdos, N=50, n_i=9, d_e=5, p=p)
        r = run_benchmark(config)
        push!(results, r)

        @printf("  p=%.2f: %5d edges | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                p, r.nedges, r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
    end

    #=========================================================================
      Exploration 3: Stalk size sweep (sv_G from 6 to 91)
    =========================================================================#
    println("\n" * "="^110)
    println("EXPLORATION 3: Stalk size sweep on path graph (N=50)")
    println("sv_G range: 6 to 91 (n_i = 3 to 13)")
    println("="^110)

    @printf("\n%10s %4s %4s %4s | %5s %5s | %6s %6s | %10s %10s | %8s %5s | %8s\n",
            "topology", "N", "n_i", "d_e", "sv_G", "edges", "nvars", "ncons", "SheafSDP", "Mosek", "raug", "iters", "speedup")
    println("-"^115)

    # n_i from 3 to 13 (sv_G from 6 to 91)
    for n_i in [3, 5, 7, 9, 11, 13]
        d_e = div(n_i, 2) + 1
        config = (topology=:path, N=50, n_i=n_i, d_e=d_e)
        r = run_benchmark(config)
        push!(results, r)

        @printf("%10s %4d %4d %4d | %5d %5d | %6d %6d | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                r.topology, r.N, r.n_i, r.d_e, r.sv_G, r.nedges, r.nvars, r.ncons,
                r.t_sheaf, r.t_mosek, r.best_raug, r.iters, r.speedup)
    end

    #=========================================================================
      Exploration 4: Graph size scaling at different stalk sizes
    =========================================================================#
    println("\n" * "="^110)
    println("EXPLORATION 4: Graph size scaling (path graphs)")
    println("="^110)

    @printf("\n%10s %4s %4s %4s | %5s %5s | %6s %6s | %10s %10s | %8s %5s | %8s\n",
            "topology", "N", "n_i", "d_e", "sv_G", "edges", "nvars", "ncons", "SheafSDP", "Mosek", "raug", "iters", "speedup")
    println("-"^115)

    for (n_i, d_e, label) in [(5, 3, "small"), (9, 5, "sweet"), (13, 7, "large")]
        println("Stalk: $label (sv_G=$(svecdim(n_i)))")
        for N in [20, 50, 100, 200]
            config = (topology=:path, N=N, n_i=n_i, d_e=d_e)
            r = run_benchmark(config)
            push!(results, r)

            @printf("%10s %4d %4d %4d | %5d %5d | %6d %6d | %8.1fms %8.1fms | %8.0e %5d | %8.2fx\n",
                    r.topology, r.N, r.n_i, r.d_e, r.sv_G, r.nedges, r.nvars, r.ncons,
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
        println("  Config: $(best.topology), N=$(best.N), n_i=$(best.n_i) (sv_G=$(best.sv_G)), edges=$(best.nedges)")

        println("\nWorst speedup: $(round(worst.speedup, digits=2))x")
        println("  Config: $(worst.topology), N=$(worst.N), n_i=$(worst.n_i) (sv_G=$(worst.sv_G)), edges=$(worst.nedges)")

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
        println("By stalk size (sv_G):")
        sv_groups = [(0, 15, "tiny"), (15, 40, "small"), (40, 70, "sweet"), (70, 100, "large")]
        for (lo, hi, label) in sv_groups
            group = filter(r -> lo < r.sv_G <= hi, valid)
            if !isempty(group)
                avg_speedup = sum(r.speedup for r in group) / length(group)
                win_rate = count(r -> r.speedup > 1.0, group) / length(group)
                @printf("  sv_G %3d-%3d (%5s): avg %.2fx, win rate %.0f%%\n", lo, hi, label, avg_speedup, 100*win_rate)
            end
        end
    end

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
