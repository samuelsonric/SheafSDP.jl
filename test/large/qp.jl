#
# Large-stalk plain QP (the :NOC base case)
#
# From large-stalk-qp.md:
#   - N = 6 on path P₆
#   - Heterogeneous stalks: n_v ∈ {48, 30} for odd/even agents
#   - Edge stalks d_e = 16
#   - Orthonormal-row restriction maps F_{i⊴e} ∈ ℝ^{16×n_v}
#   - Dense SPD quadratic R_i per agent
#   - All blocks :NOC (free variables)
#   - Target b = δx₀ (realizable)
#
# This tests the Q path at scale with ν = 0 (no barrier).
#
using AppleAccelerate
using SheafSDP
using CommonSolve: solve
using SparseArrays
using LinearAlgebra
using Random
using JuMP
using MosekTools
using OSQP
using BlockSparseArrays: colrange, rowrange, blocksparse, block

function run_benchmark(; raug=1e9, ε_R=0.01, scale=1)
    Random.seed!(42)

    # Graph: path P₆
    N = 6 * scale
    edges = [(i, i+1) for i in 1:N-1]
    ne = length(edges)

    # Heterogeneous vertex stalks: 48 for odd, 30 for even
    n_v = [isodd(i) ? 48 : 30 for i in 1:N]

    # Edge stalk dimension
    d_e = 16

    # Generate orthonormal-row restriction maps F_{i⊴e} ∈ ℝ^{d_e × n_v[i]}
    # For edge e = (i,j), we need F_i and F_j
    function make_restriction_map(n)
        # Random matrix, then take thin Q factor for orthonormal rows
        G = randn(n, d_e)
        Q, _ = qr(G)
        return Matrix(Q)'  # d_e × n
    end

    # Restriction maps: F[e] = (F_i, F_j) for edge e = (i,j)
    F = [(make_restriction_map(n_v[i]), make_restriction_map(n_v[j])) for (i, j) in edges]

    # Generate dense SPD R_i = G_i G_i' + ε_R * I
    function make_R(n)
        G = randn(n, n)
        return G * G' + ε_R * I
    end
    R = [make_R(n_v[i]) for i in 1:N]

    # Random linear cost
    c_data = [randn(n_v[i]) for i in 1:N]

    # Generate realizable target: b = δx₀
    x0 = [randn(n_v[i]) for i in 1:N]
    b = [F[e][1] * x0[i] - F[e][2] * x0[j] for (e, (i, j)) in enumerate(edges)]

    # Reference solver (shared model builder)
    function solve_reference(optimizer; set_tol=false)
        model = Model(optimizer)
        set_silent(model)
        if set_tol
            set_optimizer_attribute(model, "eps_abs", 1e-8)
            set_optimizer_attribute(model, "eps_rel", 1e-8)
        end

        # Variables: x_i ∈ ℝ^{n_v[i]}
        x = [@variable(model, [1:n_v[i]]) for i in 1:N]

        # Coordination constraints: F_i x_i - F_j x_j = b_e
        for (e, (i, j)) in enumerate(edges)
            F_i, F_j = F[e]
            @constraint(model, F_i * x[i] - F_j * x[j] .== b[e])
        end

        # Objective: Σ_i ½ x_i' R_i x_i + c_i' x_i
        @objective(model, Min,
            sum(0.5 * x[i]' * R[i] * x[i] + c_data[i]' * x[i] for i in 1:N))

        optimize!(model)
        return objective_value(model), [value.(x[i]) for i in 1:N]
    end

    # SheafSDP
    function solve_sheaf()
        # Column layout: one block per agent
        col_x(i) = i

        # Row layout: one coordination row per edge
        row_coord(e) = e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        # Coordination rows: F_i x_i - F_j x_j = b_e
        for (e, (i, j)) in enumerate(edges)
            F_i, F_j = F[e]
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(i)); push!(blocks, F_i)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(j)); push!(blocks, -F_j)
        end

        B = blocksparse(row_ids, col_ids, blocks)

        # Cost vector
        c_vec = zeros(size(B, 2))
        for i in 1:N
            c_vec[colrange(B, col_x(i))] .= c_data[i]
        end

        # RHS
        g = zeros(size(B, 1))
        for e in 1:ne
            g[rowrange(B, row_coord(e))] .= b[e]
        end

        # Quadratic: Q[x_i] = R_i
        Q = SheafSDP.allocblockdiag(B)
        fill!(Q, 0)
        for i in 1:N
            Qv = block(Q, col_x(i), col_x(i), col_x(i))
            for k in 1:n_v[i], l in 1:n_v[i]
                Qv[k, l] = R[i][k, l]
            end
        end

        # Cones: all :NOC
        cones = [:NOC for _ in 1:N]

        prob = IPMProblem(c_vec, g, B, Q, cones)
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=raug),
            feas_tol=1e-8, gap_tol=1e-8, itmax=100
        )
        result = solve(prob, settings)

        # Objective: ½ p'Qp + c'p
        obj = 0.5 * dot(result.p, Symmetric(sparse(Q), :L) * result.p) + dot(c_vec, result.p)

        # Extract solution
        x_sol = [result.p[colrange(B, col_x(i))] for i in 1:N]

        return obj, x_sol, result.iterations, result.kkt_iters, result.status
    end

    # Warmup
    solve_reference(Mosek.Optimizer)
    solve_reference(OSQP.Optimizer; set_tol=true)
    solve_sheaf()

    t_mosek = @elapsed (obj_mosek, x_mosek) = solve_reference(Mosek.Optimizer)
    t_osqp = @elapsed (obj_osqp, x_osqp) = solve_reference(OSQP.Optimizer; set_tol=true)
    t_sheaf = @elapsed (obj_sheaf, x_sheaf, iters, kkt_iters, status) = solve_sheaf()

    # Solution difference (max norm across all agents)
    sol_diff = maximum(norm(x_mosek[i] - x_sheaf[i], Inf) for i in 1:N)

    return (
        N = N,
        ne = ne,
        n_v = n_v,
        d_e = d_e,
        t_mosek = t_mosek,
        t_osqp = t_osqp,
        t_sheaf = t_sheaf,
        iters = iters,
        kkt_iters = kkt_iters,
        status = status,
        obj_mosek = obj_mosek,
        obj_osqp = obj_osqp,
        obj_sheaf = obj_sheaf,
        sol_diff = sol_diff,
    )
end

# Run benchmark
println("Large-Stalk Plain QP Benchmark: SheafSDP vs OSQP vs Mosek")
println("==================================================")
println()
println("Problem: N agents on path P_N, heterogeneous stalks n_v ∈ {30, 48}")
println("Edge stalks: d_e = 16, orthonormal-row restriction maps")
println("Objective: ½ x'Rx + c'x (dense SPD R per agent)")
println("Constraint: hard sheaf consensus δ_F x = b")
println("All blocks :NOC (free variables), ν = 0")
println()

results = []
for scale in [1, 2, 3, 4]
    r = run_benchmark(; raug=1e9, scale=scale)
    push!(results, r)
    if r.status != SheafSDP.OPTIMAL && r.status != SheafSDP.NEAR_OPTIMAL
        println("Warning: scale=$(scale) status=$(r.status)")
    end
end

# Print table
println("| N | Edges | OSQP | Mosek | SheafSDP | KKT iters | vs OSQP | vs Mosek |")
println("|---|-------|------|-------|----------|-----------|---------|----------|")
for r in results
    osqp_ms = round(r.t_osqp * 1000, digits=1)
    mosek_ms = round(r.t_mosek * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    vs_osqp = round(r.t_osqp / r.t_sheaf, digits=1)
    vs_mosek = round(r.t_mosek / r.t_sheaf, digits=1)
    println("| $(r.N) | $(r.ne) | $(osqp_ms) ms | $(mosek_ms) ms | $(sheaf_ms) ms | $(r.kkt_iters) | $(vs_osqp)x | $(vs_mosek)x |")
end
println()

# Verify correctness
println("Correctness check:")
for r in results
    obj_diff_m = abs(r.obj_mosek - r.obj_sheaf)
    obj_diff_o = abs(r.obj_osqp - r.obj_sheaf)
    println("  N=$(r.N): |Mosek-Sheaf| = $(round(obj_diff_m, sigdigits=3)), |OSQP-Sheaf| = $(round(obj_diff_o, sigdigits=3))")
end
