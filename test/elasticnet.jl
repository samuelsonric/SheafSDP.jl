#
# Example 2b: Regularized fuel (elastic-net control)
#
# Combines quadratic (R) and ℓ₁ (λ) costs:
#   f_i(u) = ½ Σ_t (u_i^t)' R u_i^t  +  λ Σ_t ‖u_i^t‖₁
#
# Uses case (c): reified logical control for general R
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

function run_benchmark(N, T; raug=1.0, ū=100.0, λ=1.0, ε_R=1.0)
    Random.seed!(42)

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
    R_cost = ε_R * Matrix(1.0I, nu, nu)  # diagonal R for uniqueness

    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, j) for i in 1:N for j in i+1:N]
    ne = length(edges)

    # OSQP reference (elastic-net via epigraph)
    function solve_osqp()
        model = Model(OSQP.Optimizer)
        set_silent(model)
        set_optimizer_attribute(model, "eps_abs", 1e-8)
        set_optimizer_attribute(model, "eps_rel", 1e-8)

        @variable(model, x[1:N, 1:T, 1:nx])
        @variable(model, u[1:N, 1:T-1, 1:nu])
        @variable(model, t_abs[1:N, 1:T-1, 1:nu] >= 0)  # |u| ≤ t_abs

        # Initial conditions
        for i in 1:N, k in 1:nx
            @constraint(model, x[i, 1, k] == x0[i][k])
        end

        # Dynamics
        for i in 1:N, t in 1:T-1, k in 1:nx
            @constraint(model, x[i, t+1, k] == sum(A_dyn[k,j] * x[i,t,j] for j in 1:nx) + sum(B_dyn[k,j] * u[i,t,j] for j in 1:nu))
        end

        # Consensus
        for (i, j) in edges, k in 1:size(P_proj, 1)
            @constraint(model, sum(P_proj[k,l] * x[i,T,l] for l in 1:nx) == sum(P_proj[k,l] * x[j,T,l] for l in 1:nx))
        end

        # Absolute value: -t_abs ≤ u ≤ t_abs
        for i in 1:N, t in 1:T-1, k in 1:nu
            @constraint(model, u[i, t, k] <= t_abs[i, t, k])
            @constraint(model, -u[i, t, k] <= t_abs[i, t, k])
        end

        # Box constraint: |u| ≤ ū
        for i in 1:N, t in 1:T-1, k in 1:nu
            @constraint(model, t_abs[i, t, k] <= ū)
        end

        # Elastic-net objective: ½ u'Ru + λ‖u‖₁
        @objective(model, Min,
            sum(sum(R_cost[k,l] * u[i,t,k] * u[i,t,l] for k in 1:nu, l in 1:nu) for i in 1:N, t in 1:T-1) / 2 +
            λ * sum(t_abs))

        optimize!(model)
        return objective_value(model)
    end

    # SheafSDP with case (c): reified control
    # Column blocks per agent per timestep t ∈ 1:T-1:
    #   u_i^t (reified, :NOC, Q=R), u_i^{t+} (:POS), u_i^{t-} (:POS), s_i^{t+} (:POS), s_i^{t-} (:POS)
    # Plus state blocks x_i^t for t ∈ 1:T
    function solve_sheaf()
        # Column layout per agent:
        #   T state blocks (x_i^1 ... x_i^T)
        #   (T-1) * 5 control blocks: for each t: u, u+, u-, s+, s-
        blocks_per_agent = T + 5 * (T - 1)

        col_x(i, t) = (i - 1) * blocks_per_agent + t
        col_u(i, t) = (i - 1) * blocks_per_agent + T + 5 * (t - 1) + 1   # reified
        col_up(i, t) = (i - 1) * blocks_per_agent + T + 5 * (t - 1) + 2
        col_um(i, t) = (i - 1) * blocks_per_agent + T + 5 * (t - 1) + 3
        col_sp(i, t) = (i - 1) * blocks_per_agent + T + 5 * (t - 1) + 4
        col_sm(i, t) = (i - 1) * blocks_per_agent + T + 5 * (t - 1) + 5

        # Row layout per agent:
        #   1 init row
        #   (T-1) dynamics rows
        #   (T-1) split rows: u - u+ + u- = 0
        #   (T-1) box+ rows: u + s+ = ū
        #   (T-1) box- rows: -u + s- = ū
        rows_per_agent = 1 + 4 * (T - 1)

        row_init(i) = (i - 1) * rows_per_agent + 1
        row_dyn(i, t) = (i - 1) * rows_per_agent + 1 + t
        row_split(i, t) = (i - 1) * rows_per_agent + T + (t - 1) + 1
        row_boxp(i, t) = (i - 1) * rows_per_agent + T + (T - 1) + (t - 1) + 1
        row_boxm(i, t) = (i - 1) * rows_per_agent + T + 2 * (T - 1) + (t - 1) + 1
        row_coord(e) = N * rows_per_agent + e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        for i in 1:N
            # init: x_i^1 = x0[i]
            push!(row_ids, row_init(i))
            push!(col_ids, col_x(i, 1))
            push!(blocks, Matrix(1.0I, nx, nx))

            for t in 1:T-1
                # dynamics: x_i^{t+1} - A x_i^t - B u_i^t = 0
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t)); push!(blocks, -A_dyn)
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t + 1)); push!(blocks, Matrix(1.0I, nx, nx))
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_u(i, t)); push!(blocks, -B_dyn)

                # split: u_i^t - u_i^{t+} + u_i^{t-} = 0
                push!(row_ids, row_split(i, t)); push!(col_ids, col_u(i, t)); push!(blocks, Matrix(1.0I, nu, nu))
                push!(row_ids, row_split(i, t)); push!(col_ids, col_up(i, t)); push!(blocks, -Matrix(1.0I, nu, nu))
                push!(row_ids, row_split(i, t)); push!(col_ids, col_um(i, t)); push!(blocks, Matrix(1.0I, nu, nu))

                # box+: u_i^t + s_i^{t+} = ū
                push!(row_ids, row_boxp(i, t)); push!(col_ids, col_u(i, t)); push!(blocks, Matrix(1.0I, nu, nu))
                push!(row_ids, row_boxp(i, t)); push!(col_ids, col_sp(i, t)); push!(blocks, Matrix(1.0I, nu, nu))

                # box-: -u_i^t + s_i^{t-} = ū
                push!(row_ids, row_boxm(i, t)); push!(col_ids, col_u(i, t)); push!(blocks, -Matrix(1.0I, nu, nu))
                push!(row_ids, row_boxm(i, t)); push!(col_ids, col_sm(i, t)); push!(blocks, Matrix(1.0I, nu, nu))
            end
        end

        # Coordination: P x_i^T - P x_j^T = 0
        for (e, (i, j)) in enumerate(edges)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T)); push!(blocks, -P_proj)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T)); push!(blocks, P_proj)
        end

        B = blocksparse(row_ids, col_ids, blocks)

        # Cost vector: c[u±] = λ
        c = zeros(size(B, 2))
        for i in 1:N, t in 1:T-1
            c[colrange(B, col_up(i, t))] .= λ
            c[colrange(B, col_um(i, t))] .= λ
        end

        # RHS: g[init] = x0, g[box±] = ū
        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_init(i))] .= x0[i]
            for t in 1:T-1
                g[rowrange(B, row_boxp(i, t))] .= ū
                g[rowrange(B, row_boxm(i, t))] .= ū
            end
        end

        # Quadratic: Q[u_i^t] = 2R on reified control blocks (objective is ½ p'Qp, so need 2R)
        Q = SheafSDP.allocblockdiag(B)
        fill!(Q, 0)
        for i in 1:N, t in 1:T-1
            Qv = block(Q, col_u(i, t), col_u(i, t), col_u(i, t))
            for k in 1:nu, l in 1:nu
                Qv[k, l] = 2 * R_cost[k, l]
            end
        end

        # Cones
        nv = N * blocks_per_agent
        cones = Vector{Symbol}(undef, nv)
        for i in 1:N
            for t in 1:T
                cones[col_x(i, t)] = :NOC
            end
            for t in 1:T-1
                cones[col_u(i, t)] = :NOC   # reified control is free
                cones[col_up(i, t)] = :POS
                cones[col_um(i, t)] = :POS
                cones[col_sp(i, t)] = :POS
                cones[col_sm(i, t)] = :POS
            end
        end

        prob = IPMProblem(c, g, B, Q, cones)
        settings = IPMSettings{Float64}(kkt=UzawaSettings{Float64}(raug=raug), feas_tol=1e-8, gap_tol=1e-8, itmax=100)
        result = solve(prob, settings)

        # Objective: ½ p'Qp + c'p
        obj = 0.5 * dot(result.p, Symmetric(sparse(Q), :L) * result.p) + dot(c, result.p)
        return obj, result.iterations, result.status
    end

    # Warmup
    solve_osqp()
    solve_sheaf()

    t_osqp = @elapsed obj_osqp = solve_osqp()
    t_sheaf = @elapsed (obj_sheaf, iters, status) = solve_sheaf()

    return (
        N = N,
        T = T,
        ne = ne,
        t_osqp = t_osqp,
        t_sheaf = t_sheaf,
        iters = iters,
        status = status,
        obj_osqp = obj_osqp,
        obj_sheaf = obj_sheaf,
    )
end

# Run benchmark sweep
println("Elastic-Net Control Benchmark: SheafSDP vs OSQP")
println("================================================")
println()
println("Problem: N agents on complete graph K_N, T timesteps")
println("Dynamics: planar double integrator (nx=4, nu=2)")
println("Objective: ½ u'Ru + λ‖u‖₁ (elastic-net)")
println("Constraints: dynamics + box |u| ≤ 100 + terminal position consensus")
println("Parameters: raug=1.0, λ=1.0, ε_R=1.0")
println()

results = []
for (N, T) in [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)]
    r = run_benchmark(N, T; raug=1.0, λ=1.0, ε_R=1.0)
    push!(results, r)
    if r.status != SheafSDP.OPTIMAL
        println("Warning: N=$(r.N), T=$(r.T) status=$(r.status)")
    end
end

# Print table
println("| N,T | Edges | OSQP | SheafSDP | Iters | Speedup |")
println("|-----|-------|------|----------|-------|---------|")
for r in results
    osqp_ms = round(r.t_osqp * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    speedup = round(r.t_osqp / r.t_sheaf, digits=1)
    println("| $(r.N),$(r.T) | $(r.ne) | $(osqp_ms) ms | $(sheaf_ms) ms | $(r.iters) | $(speedup)x |")
end
println()

# Verify correctness
println("Correctness check (objective difference):")
for r in results
    diff = abs(r.obj_osqp - r.obj_sheaf)
    println("  N=$(r.N), T=$(r.T): |obj_osqp - obj_sheaf| = $(round(diff, sigdigits=3))")
end
