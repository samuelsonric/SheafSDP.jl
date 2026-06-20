#
# Large-stalk SOC (distributed least-norm with box)
#
# From large-stalk-instances.md §4:
#   - N = 6 on path P₆
#   - Heterogeneous stalks: n_v ∈ {48, 30} for odd/even agents
#   - Edge stalks d_e = 16
#   - Orthonormal-row restriction maps
#   - Objective: Σ ‖x_i‖₂ (sum of norms)
#   - ζ_i = (s_i; x_i) ∈ Q^{1+n_v} with √2 scaling
#
# This tests SOC cones with large arms (n_v ≥ 30).
#
using AppleAccelerate
using SheafSDP
using CommonSolve: solve
using SparseArrays
using LinearAlgebra
using Random
using JuMP
using MosekTools
using BlockSparseArrays: colrange, rowrange, blocksparse, block

function run_benchmark(; raug=1e7, scale=1, x_max=100.0)
    Random.seed!(42)

    # Graph: path P₆
    N = 6 * scale
    edges = [(i, i+1) for i in 1:N-1]
    ne = length(edges)

    # Heterogeneous vertex stalks: 48 for odd, 30 for even
    n_v = [isodd(i) ? 48 : 30 for i in 1:N]

    # Edge stalk dimension
    d_e = 16

    # Generate orthonormal-row restriction maps
    function make_restriction_map(n)
        G = randn(n, d_e)
        Q, _ = qr(G)
        return Matrix(Q)'
    end
    F = [(make_restriction_map(n_v[i]), make_restriction_map(n_v[j])) for (i, j) in edges]

    # Realizable target: b = δx₀
    x0 = [randn(n_v[i]) for i in 1:N]
    b = [F[e][1] * x0[i] - F[e][2] * x0[j] for (e, (i, j)) in enumerate(edges)]

    # Mosek reference
    function solve_mosek()
        model = Model(Mosek.Optimizer)
        set_silent(model)

        # Variables
        x = [@variable(model, [1:n_v[i]]) for i in 1:N]
        s = @variable(model, [1:N], lower_bound=0)  # epigraph vars

        # Coordination constraints
        for (e, (i, j)) in enumerate(edges)
            F_i, F_j = F[e]
            @constraint(model, F_i * x[i] - F_j * x[j] .== b[e])
        end

        # SOC constraints: ‖x_i‖₂ ≤ s_i
        for i in 1:N
            @constraint(model, [s[i]; x[i]] in SecondOrderCone())
        end

        # Box constraints
        for i in 1:N
            @constraint(model, -x_max .<= x[i] .<= x_max)
        end

        # Objective: Σ s (sum of norms)
        @objective(model, Min, sum(s))

        optimize!(model)
        return objective_value(model), [value.(x[i]) for i in 1:N]
    end

    # SheafSDP with SOC cones
    function solve_sheaf()
        invrt2 = 1 / sqrt(2.0)

        # Column layout: ζ_i = (s_i; x_i), sp_i, sm_i (box slacks)
        col_ζ(i) = 3 * (i - 1) + 1
        col_sp(i) = 3 * (i - 1) + 2
        col_sm(i) = 3 * (i - 1) + 3

        # Row layout: boxp, boxm, coord
        row_boxp(i) = 2 * (i - 1) + 1
        row_boxm(i) = 2 * (i - 1) + 2
        row_coord(e) = 2 * N + e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        # Extract x from ζ = (s; x), scaled by 1/√2
        for i in 1:N
            extract_x = [zeros(n_v[i], 1) Matrix(1.0I, n_v[i], n_v[i])] .* invrt2

            # box+: x + sp = x_max
            push!(row_ids, row_boxp(i)); push!(col_ids, col_ζ(i)); push!(blocks, extract_x)
            push!(row_ids, row_boxp(i)); push!(col_ids, col_sp(i)); push!(blocks, Matrix(1.0I, n_v[i], n_v[i]))

            # box-: -x + sm = x_max
            push!(row_ids, row_boxm(i)); push!(col_ids, col_ζ(i)); push!(blocks, -extract_x)
            push!(row_ids, row_boxm(i)); push!(col_ids, col_sm(i)); push!(blocks, Matrix(1.0I, n_v[i], n_v[i]))
        end

        # Coordination rows: F_i x_i - F_j x_j = b_e
        for (e, (i, j)) in enumerate(edges)
            F_i, F_j = F[e]
            block_i = [zeros(d_e, 1) F_i] .* invrt2
            block_j = [zeros(d_e, 1) F_j] .* invrt2
            push!(row_ids, row_coord(e)); push!(col_ids, col_ζ(i)); push!(blocks, block_i)
            push!(row_ids, row_coord(e)); push!(col_ids, col_ζ(j)); push!(blocks, -block_j)
        end

        B = blocksparse(row_ids, col_ids, blocks)

        # Cost: 1 on head (s_i), scaled by 1/√2
        c_vec = zeros(size(B, 2))
        for i in 1:N
            ζ_range = colrange(B, col_ζ(i))
            c_vec[ζ_range[1]] = invrt2
        end

        # RHS
        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_boxp(i))] .= x_max
            g[rowrange(B, row_boxm(i))] .= x_max
        end
        for e in 1:ne
            g[rowrange(B, row_coord(e))] .= b[e]
        end

        # Q = 0
        Q = SheafSDP.allocblockdiag(B)
        fill!(Q, 0)

        # Cones
        cones = Vector{Symbol}(undef, 3*N)
        for i in 1:N
            cones[col_ζ(i)] = :SOC
            cones[col_sp(i)] = :POS
            cones[col_sm(i)] = :POS
        end

        prob = IPMProblem(c_vec, g, B, Q, cones)
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=raug),
            feas_tol=1e-8, gap_tol=1e-8, itmax=200
        )
        result = solve(prob, settings)

        return dot(c_vec, result.p), result.iterations, result.kkt_iters, result.status
    end

    # Warmup
    solve_mosek()
    solve_sheaf()

    t_mosek = @elapsed (obj_mosek, x_mosek) = solve_mosek()
    t_sheaf = @elapsed (obj_sheaf, iters, kkt_iters, status) = solve_sheaf()

    return (
        N = N,
        ne = ne,
        n_v = n_v,
        d_e = d_e,
        t_mosek = t_mosek,
        t_sheaf = t_sheaf,
        iters = iters,
        kkt_iters = kkt_iters,
        status = status,
        obj_mosek = obj_mosek,
        obj_sheaf = obj_sheaf,
    )
end

# Run benchmark
println("Large-Stalk SOC Benchmark: SheafSDP vs Mosek")
println("=============================================")
println()
println("Problem: N agents on path P_N, heterogeneous stalks n_v ∈ {30, 48}")
println("Edge stalks: d_e = 16, orthonormal-row restriction maps")
println("Objective: Σ ‖x‖₂ (sum of norms)")
println("Constraint: box + hard sheaf consensus δ_F x = b")
println("SOC blocks with √2 scaling, ν = 2N")
println()

results = []
for scale in [1, 2, 3]
    r = run_benchmark(; raug=1e7, scale=scale)
    push!(results, r)
    if r.status != SheafSDP.OPTIMAL && r.status != SheafSDP.NEAR_OPTIMAL
        println("Warning: scale=$(scale) status=$(r.status)")
    end
end

# Print table
println("| N | Edges | Mosek | SheafSDP | KKT iters | vs Mosek |")
println("|---|-------|-------|----------|-----------|----------|")
for r in results
    mosek_ms = round(r.t_mosek * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    vs_mosek = round(r.t_mosek / r.t_sheaf, digits=1)
    println("| $(r.N) | $(r.ne) | $(mosek_ms) ms | $(sheaf_ms) ms | $(r.kkt_iters) | $(vs_mosek)x |")
end
println()

# Verify correctness
println("Correctness check:")
for r in results
    obj_diff = abs(r.obj_mosek - r.obj_sheaf)
    println("  N=$(r.N): |Mosek - Sheaf| = $(round(obj_diff, sigdigits=3))")
end
