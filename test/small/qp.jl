#
# Three-backend oracle test: QP consensus (§4 recipe)
#
# Benchmark: SheafSDP vs OSQP on multi-agent consensus QP
#
using AppleAccelerate
using SheafSDP
using CommonSolve: solve
using SparseArrays
using LinearAlgebra
using Random
using JuMP
using OSQP
using BlockSparseArrays: vtxs, colrange, rowrange, ncols, blocksparse, block

function run_benchmark(N, T; raug=1e9)
    Random.seed!(42)

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
    R_cost = Matrix(1.0I, nu, nu)
    ε_reg = 0.01

    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, j) for i in 1:N for j in i+1:N]
    ne = length(edges)

    # OSQP
    function solve_osqp()
        model = Model(OSQP.Optimizer)
        set_silent(model)
        set_optimizer_attribute(model, "eps_abs", 1e-8)
        set_optimizer_attribute(model, "eps_rel", 1e-8)
        @variable(model, x[1:N, 1:T, 1:nx])
        @variable(model, u[1:N, 1:T-1, 1:nu])
        for i in 1:N
            @constraint(model, x[i, 1, :] .== x0[i])
        end
        for i in 1:N, t in 1:T-1
            @constraint(model, x[i, t+1, :] .== A_dyn * x[i, t, :] + B_dyn * u[i, t, :])
        end
        for (i, j) in edges
            @constraint(model, P_proj * x[i, T, :] .== P_proj * x[j, T, :])
        end
        @objective(model, Min, sum(u[i, t, :]' * R_cost * u[i, t, :] for i in 1:N, t in 1:T-1) + ε_reg * sum(x[i, T, :]' * x[i, T, :] for i in 1:N))
        optimize!(model)
        return objective_value(model)
    end

    # SheafSDP
    function solve_sheaf()
        col_x(i, t) = (i - 1) * (T + T - 1) + t
        col_u(i, t) = (i - 1) * (T + T - 1) + T + t
        row_init(i) = (i - 1) * T + 1
        row_dyn(i, t) = (i - 1) * T + 1 + t
        row_coord(e) = N * T + e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]
        for i in 1:N
            push!(row_ids, row_init(i)); push!(col_ids, col_x(i, 1)); push!(blocks, Matrix(1.0I, nx, nx))
            for t in 1:T-1
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t)); push!(blocks, -A_dyn)
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t + 1)); push!(blocks, Matrix(1.0I, nx, nx))
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_u(i, t)); push!(blocks, -B_dyn)
            end
        end
        for (e, (i, j)) in enumerate(edges)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T)); push!(blocks, -P_proj)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T)); push!(blocks, P_proj)
        end

        B = blocksparse(row_ids, col_ids, blocks)
        c = zeros(size(B, 2))
        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_init(i))] .= x0[i]
        end

        Q = SheafSDP.allocblockdiag(B); fill!(Q, 0)
        for i in 1:N
            for t in 1:T-1
                Qv = block(Q, col_u(i, t), col_u(i, t), col_u(i, t))
                for k in 1:nu; Qv[k, k] = 2 * R_cost[k, k]; end
            end
            Qv = block(Q, col_x(i, T), col_x(i, T), col_x(i, T))
            for k in 1:nx; Qv[k, k] = 2 * ε_reg; end
        end

        nv = N * (T + T - 1)
        cones = [CofreeCone() for _ in 1:nv]
        prob = IPMProblem(c, g, B, Q, cones)
        settings = IPMSettings{Float64}(kkt=UzawaSettings{Float64}(raug=raug), feas_tol=1e-8, gap_tol=1e-8, itmax=100)
        result = solve(prob, settings)
        return 0.5 * dot(result.p, Symmetric(sparse(Q), :L) * result.p), result.iterations, result.kkt_iters
    end

    # Warmup
    solve_osqp(); solve_sheaf()

    t_osqp = @elapsed obj_osqp = solve_osqp()
    t_sheaf = @elapsed (obj_sheaf, iters, kkt_iters) = solve_sheaf()

    return (
        N = N,
        T = T,
        ne = ne,
        t_osqp = t_osqp,
        t_sheaf = t_sheaf,
        iters = iters,
        kkt_iters = kkt_iters,
        obj_osqp = obj_osqp,
        obj_sheaf = obj_sheaf,
    )
end

# Run benchmark sweep
println("QP Consensus Benchmark: SheafSDP vs OSQP")
println("=========================================")
println()
println("Problem: N agents on complete graph K_N, T timesteps")
println("Dynamics: planar double integrator (nx=4, nu=2)")
println("Objective: quadratic control effort + terminal regularization")
println("Constraint: terminal position consensus")
println()

results = []
for (N, T) in [(20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70)]
    r = run_benchmark(N, T; raug=1e9)
    push!(results, r)
end

# Print table
println("| N,T | Edges | OSQP | SheafSDP | KKT iters | Speedup |")
println("|-----|-------|------|----------|-----------|---------|")
for r in results
    speedup = r.t_osqp / r.t_sheaf
    osqp_ms = round(r.t_osqp * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    speedup_str = round(speedup, digits=1)
    println("| $(r.N),$(r.T) | $(r.ne) | $(osqp_ms) ms | $(sheaf_ms) ms | $(r.kkt_iters) | $(speedup_str)x |")
end
println()

# Verify correctness
println("Correctness check (objective difference):")
for r in results
    diff = abs(r.obj_osqp - r.obj_sheaf)
    println("  N=$(r.N), T=$(r.T): |obj_osqp - obj_sheaf| = $(round(diff, sigdigits=3))")
end
