#
# Large-stalk POS / distributed box LP
#
# From large-stalk-instances.md §3:
#   - N = 6 on path P₆
#   - Heterogeneous stalks: n_v ∈ {48, 30} for odd/even agents
#   - Edge stalks d_e = 16
#   - Orthonormal-row restriction maps F_{i⊴e} ∈ ℝ^{16×n_v}
#   - f_i(x_i) = c_i' x_i with 0 ≤ x_i ≤ u_i
#
# This tests POS cones at large stalk size (ν scales with n_v).
#
using AppleAccelerate
using SheafSDP
using CommonSolve: solve
using SparseArrays
using LinearAlgebra
using Random
using JuMP
using HiGHS
using MosekTools
using BlockSparseArrays: colrange, rowrange, blocksparse, block

function run_benchmark(; raug=1e8, scale=1)
    Random.seed!(42)

    # Graph: path P₆
    N = 6 * scale
    edges = [(i, i+1) for i in 1:N-1]
    ne = length(edges)

    # Heterogeneous vertex stalks: 48 for odd, 30 for even
    n_v = [isodd(i) ? 48 : 30 for i in 1:N]

    # Edge stalk dimension
    d_e = 16

    # Box upper bounds (random positive)
    u_box = [abs.(randn(n_v[i])) .+ 1.0 for i in 1:N]

    # Random linear cost
    c_data = [randn(n_v[i]) for i in 1:N]

    # Generate orthonormal-row restriction maps F_{i⊴e} ∈ ℝ^{d_e × n_v[i]}
    function make_restriction_map(n)
        G = randn(n, d_e)
        Q, _ = qr(G)
        return Matrix(Q)'  # d_e × n
    end

    # Restriction maps: F[e] = (F_i, F_j) for edge e = (i,j)
    F = [(make_restriction_map(n_v[i]), make_restriction_map(n_v[j])) for (i, j) in edges]

    # Generate realizable target: b = δx₀
    x0 = [rand(n_v[i]) .* u_box[i] for i in 1:N]  # feasible starting point
    b = [F[e][1] * x0[i] - F[e][2] * x0[j] for (e, (i, j)) in enumerate(edges)]

    # Reference solver
    function solve_reference(optimizer)
        model = Model(optimizer)
        set_silent(model)

        # Variables: 0 ≤ x_i ≤ u_i
        x = [@variable(model, [1:n_v[i]], lower_bound=0) for i in 1:N]
        for i in 1:N
            @constraint(model, x[i] .<= u_box[i])
        end

        # Coordination constraints: F_i x_i - F_j x_j = b_e
        for (e, (i, j)) in enumerate(edges)
            F_i, F_j = F[e]
            @constraint(model, F_i * x[i] - F_j * x[j] .== b[e])
        end

        # Objective: Σ_i c_i' x_i
        @objective(model, Min, sum(c_data[i]' * x[i] for i in 1:N))

        optimize!(model)
        return objective_value(model), [value.(x[i]) for i in 1:N]
    end

    # SheafSDP with POS cones
    function solve_sheaf()
        # Column layout: x_i (:POS), w_i (:POS slack for box)
        col_x(i) = 2 * (i - 1) + 1
        col_w(i) = 2 * (i - 1) + 2

        # Row layout: box_i, then coord_e
        row_box(i) = i
        row_coord(e) = N + e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        # Box rows: x_i + w_i = u_i
        for i in 1:N
            push!(row_ids, row_box(i)); push!(col_ids, col_x(i)); push!(blocks, Matrix(1.0I, n_v[i], n_v[i]))
            push!(row_ids, row_box(i)); push!(col_ids, col_w(i)); push!(blocks, Matrix(1.0I, n_v[i], n_v[i]))
        end

        # Coordination rows: F_i x_i - F_j x_j = b_e
        for (e, (i, j)) in enumerate(edges)
            F_i, F_j = F[e]
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(i)); push!(blocks, F_i)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(j)); push!(blocks, -F_j)
        end

        B = blocksparse(row_ids, col_ids, blocks)

        # Cost vector: c on x, 0 on w
        c_vec = zeros(size(B, 2))
        for i in 1:N
            c_vec[colrange(B, col_x(i))] .= c_data[i]
        end

        # RHS: u_box on box rows, b on coord rows
        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_box(i))] .= u_box[i]
        end
        for e in 1:ne
            g[rowrange(B, row_coord(e))] .= b[e]
        end

        # Q = 0 (LP)
        Q = SheafSDP.allocblockdiag(B)
        fill!(Q, 0)

        # Cones: all PositiveCone
        cones = [PositiveCone() for _ in 1:2*N]

        prob = IPMProblem(Q, B, c_vec, g, cones)
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=raug),
            feas_tol=1e-6, gap_tol=1e-6, itmax=100
        )
        result = solve(prob, settings)

        # Extract solution
        x_sol = [result.p[colrange(B, col_x(i))] for i in 1:N]

        return dot(c_vec, result.p), x_sol, result.ipm_niter, result.kkt_niter, result.status
    end

    # Warmup
    solve_reference(HiGHS.Optimizer)
    solve_reference(Mosek.Optimizer)
    solve_sheaf()

    t_highs = @elapsed (obj_highs, x_highs) = solve_reference(HiGHS.Optimizer)
    t_mosek = @elapsed (obj_mosek, x_mosek) = solve_reference(Mosek.Optimizer)
    t_sheaf = @elapsed (obj_sheaf, x_sheaf, iters, kkt_iters, status) = solve_sheaf()

    # Solution difference
    sol_diff = maximum(norm(x_mosek[i] - x_sheaf[i], Inf) for i in 1:N)

    return (
        N = N,
        ne = ne,
        n_v = n_v,
        d_e = d_e,
        t_highs = t_highs,
        t_mosek = t_mosek,
        t_sheaf = t_sheaf,
        iters = iters,
        kkt_iters = kkt_iters,
        status = status,
        obj_highs = obj_highs,
        obj_mosek = obj_mosek,
        obj_sheaf = obj_sheaf,
        sol_diff = sol_diff,
    )
end

# Run benchmark
println("Large-Stalk LP Benchmark: SheafSDP vs HiGHS vs Mosek")
println("=====================================================")
println()
println("Problem: N agents on path P_N, heterogeneous stalks n_v ∈ {30, 48}")
println("Edge stalks: d_e = 16, orthonormal-row restriction maps")
println("Objective: c'x (linear)")
println("Constraint: 0 ≤ x ≤ u, hard sheaf consensus δ_F x = b")
println("All blocks :POS")
println()

results = []
for scale in [1, 2, 3, 4]
    r = run_benchmark(; raug=1e8, scale=scale)
    push!(results, r)
    if r.status != SheafSDP.OPTIMAL && r.status != SheafSDP.NEAR_OPTIMAL
        println("Warning: scale=$(scale) status=$(r.status)")
    end
end

# Print table
println("| N | Edges | HiGHS | Mosek | SheafSDP | KKT iters | vs HiGHS | vs Mosek |")
println("|---|-------|-------|-------|----------|-----------|----------|----------|")
for r in results
    highs_ms = round(r.t_highs * 1000, digits=1)
    mosek_ms = round(r.t_mosek * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    vs_highs = round(r.t_highs / r.t_sheaf, digits=1)
    vs_mosek = round(r.t_mosek / r.t_sheaf, digits=1)
    println("| $(r.N) | $(r.ne) | $(highs_ms) ms | $(mosek_ms) ms | $(sheaf_ms) ms | $(r.kkt_iters) | $(vs_highs)x | $(vs_mosek)x |")
end
println()

# Verify correctness
println("Correctness check:")
for r in results
    obj_diff_h = abs(r.obj_highs - r.obj_sheaf)
    obj_diff_m = abs(r.obj_mosek - r.obj_sheaf)
    println("  N=$(r.N): |HiGHS-Sheaf| = $(round(obj_diff_h, sigdigits=3)), |Mosek-Sheaf| = $(round(obj_diff_m, sigdigits=3))")
end
