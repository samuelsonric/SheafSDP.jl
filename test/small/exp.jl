#
# Small-stalk EXP benchmark: Log-barrier minimum-fuel (Recipe A, §3 of exp-recipes.md)
#
# Three-backend oracle: R = JuMP MOI.ExponentialCone (Mosek), S = SheafSDP
#
# Problem: N agents on path graph P_N, T timesteps
# Dynamics: planar double integrator (nx=4, nu=2)
# Objective: maximize Σ log(ū - |u|) = log-barrier pushing control away from saturation
# Constraints: dynamics + terminal position consensus
#
# EXP cone scaling axes (§9): N (agents), T (timesteps) — no arm-length axis
#
using AppleAccelerate
using SheafSDP
using CommonSolve: solve
using SparseArrays
using LinearAlgebra
using Random
using JuMP
using MosekTools
using BlockSparseArrays: vtxs, colrange, rowrange, ncols, blocksparse, block

function run_exp_benchmark(N, T; raug=1e6, ū=10.0)
    Random.seed!(42)

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

    x0 = [randn(nx) for _ in 1:N]
    # Path graph P_N for coordination
    edges = [(i, i+1) for i in 1:N-1]
    ne = length(edges)

    # Count exp cones: nu * (T-1) per agent
    n_exp = N * nu * (T - 1)

    #
    # Leg R: JuMP with explicit MOI.ExponentialCone (conic lift)
    #
    # MOI convention: (a, b, c) ∈ ExpCone means c ≥ b·exp(a/b)
    # Our convention: (x₁, x₂, x₃) means x₁ ≥ x₂·exp(x₃/x₂)
    # Mapping: a ↔ x₃, b ↔ x₂, c ↔ x₁
    #
    function solve_mosek()
        model = Model(Mosek.Optimizer)
        set_silent(model)

        @variable(model, x[1:N, 1:T, 1:nx])
        @variable(model, u[1:N, 1:T-1, 1:nu])
        @variable(model, up[1:N, 1:T-1, 1:nu] >= 0)
        @variable(model, um[1:N, 1:T-1, 1:nu] >= 0)
        @variable(model, τ[1:N, 1:T-1, 1:nu])  # log epigraph var

        # u = u⁺ - u⁻
        for i in 1:N, t in 1:T-1
            @constraint(model, u[i, t, :] .== up[i, t, :] - um[i, t, :])
        end

        # Initial conditions
        for i in 1:N
            @constraint(model, x[i, 1, :] .== x0[i])
        end

        # Dynamics
        for i in 1:N, t in 1:T-1
            @constraint(model, x[i, t+1, :] .== A_dyn * x[i, t, :] + B_dyn * u[i, t, :])
        end

        # Consensus
        for (i, j) in edges
            @constraint(model, P_proj * x[i, T, :] .== P_proj * x[j, T, :])
        end

        # Log epigraph via exp cone: τ ≤ log(arg), arg = ū - u⁺ - u⁻
        # (arg, 1, τ) in our convention → (τ, 1, arg) in MOI convention
        for i in 1:N, t in 1:T-1, k in 1:nu
            arg = ū - up[i, t, k] - um[i, t, k]
            @constraint(model, [τ[i, t, k], 1, arg] in MOI.ExponentialCone())
        end

        # Maximize Σ τ (since τ ≤ log(arg))
        @objective(model, Max, sum(τ))

        optimize!(model)
        return objective_value(model)
    end

    #
    # Leg S: SheafSDP with EXP cones (§3 construction)
    #
    function solve_sheaf()
        # Per agent: T state blocks (NOC), 2*(T-1) control blocks (POS for u⁺, u⁻),
        # nu*(T-1) exp leaves (EXP, one per actuator channel per timestep)
        num_exp_per_agent = nu * (T - 1)
        blocks_per_agent = T + 2 * (T - 1) + num_exp_per_agent

        col_x(i, t) = (i - 1) * blocks_per_agent + t
        col_up(i, t) = (i - 1) * blocks_per_agent + T + 2 * (t - 1) + 1
        col_um(i, t) = (i - 1) * blocks_per_agent + T + 2 * (t - 1) + 2
        col_exp(i, t, k) = (i - 1) * blocks_per_agent + T + 2 * (T - 1) + (t - 1) * nu + k

        # Block rows per agent: 1 (init) + (T-1) (dynamics) + 2*num_exp_per_agent (arg + x2)
        rows_per_agent = 1 + (T - 1) + 2 * num_exp_per_agent

        row_init(i) = (i - 1) * rows_per_agent + 1
        row_dyn(i, t) = (i - 1) * rows_per_agent + 1 + t
        row_arg(i, t, k) = (i - 1) * rows_per_agent + T + ((t - 1) * nu + k - 1) + 1
        row_x2(i, t, k) = (i - 1) * rows_per_agent + T + num_exp_per_agent + ((t - 1) * nu + k - 1) + 1
        row_coord(e) = N * rows_per_agent + e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        for i in 1:N
            # Initial condition: x[1] = x0[i]
            push!(row_ids, row_init(i))
            push!(col_ids, col_x(i, 1))
            push!(blocks, Matrix(1.0I, nx, nx))

            for t in 1:T-1
                # Dynamics: x[t+1] = A x[t] + B(u⁺ - u⁻)
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t)); push!(blocks, -A_dyn)
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t + 1)); push!(blocks, Matrix(1.0I, nx, nx))
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_up(i, t)); push!(blocks, -B_dyn)
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_um(i, t)); push!(blocks, B_dyn)

                for k in 1:nu
                    # arg row: x₁ + u⁺_k + u⁻_k = ū
                    push!(row_ids, row_arg(i, t, k))
                    push!(col_ids, col_exp(i, t, k))
                    push!(blocks, reshape([1.0, 0.0, 0.0], 1, 3))  # picks x₁

                    push!(row_ids, row_arg(i, t, k))
                    push!(col_ids, col_up(i, t))
                    push!(blocks, reshape([k == 1 ? 1.0 : 0.0, k == 2 ? 1.0 : 0.0], 1, 2))

                    push!(row_ids, row_arg(i, t, k))
                    push!(col_ids, col_um(i, t))
                    push!(blocks, reshape([k == 1 ? 1.0 : 0.0, k == 2 ? 1.0 : 0.0], 1, 2))

                    # x₂ row: x₂ = 1
                    push!(row_ids, row_x2(i, t, k))
                    push!(col_ids, col_exp(i, t, k))
                    push!(blocks, reshape([0.0, 1.0, 0.0], 1, 3))  # picks x₂
                end
            end
        end

        # Coordination: terminal position consensus
        for (e, (i, j)) in enumerate(edges)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T)); push!(blocks, -P_proj)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T)); push!(blocks, P_proj)
        end

        B = blocksparse(row_ids, col_ids, blocks)

        # Objective: minimize -Σ x₃ = maximize Σ τ
        c = zeros(size(B, 2))
        for i in 1:N, t in 1:T-1, k in 1:nu
            c_col = col_exp(i, t, k)
            c_rng = colrange(B, c_col)
            c[c_rng[3]] = -1.0  # x₃ slot
        end

        # RHS: g[init] = x0, g[dyn] = 0, g[arg] = ū, g[x2] = 1, g[coord] = 0
        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_init(i))] .= x0[i]
            for t in 1:T-1, k in 1:nu
                g[rowrange(B, row_arg(i, t, k))] .= ū
                g[rowrange(B, row_x2(i, t, k))] .= 1.0
            end
        end

        Q = SheafSDP.allocblockdiag(B)
        fill!(Q, 0)

        nv = N * blocks_per_agent
        cones = Vector{AbstractCone}(undef, nv)
        for i in 1:N
            for t in 1:T
                cones[col_x(i, t)] = CofreeCone()
            end
            for t in 1:T-1
                cones[col_up(i, t)] = PositiveCone()
                cones[col_um(i, t)] = PositiveCone()
                for k in 1:nu
                    cones[col_exp(i, t, k)] = ExponentialCone()
                end
            end
        end

        prob = IPMProblem(Q, B, c, g, cones)
        # Exp cone needs looser tolerances and more iterations (per §9)
        settings = IPMSettings{Float64}(kkt=UzawaSettings{Float64}(raug=raug), feas_tol=1e-5, gap_tol=1e-5, itmax=200)
        result = solve(prob, settings)

        # Return negated objective (to match Mosek's maximization)
        return -dot(c, result.p), result.ipm_niter, result.kkt_niter, result.status
    end

    # Warmup
    solve_mosek()
    try solve_sheaf() catch end

    # Timed runs
    t_mosek = @elapsed obj_mosek = solve_mosek()

    local obj_sheaf, iters, kkt_iters, status, t_sheaf
    t_sheaf = @elapsed begin
        try
            (obj_sheaf, iters, kkt_iters, status) = solve_sheaf()
        catch e
            if isa(e, PosDefException)
                obj_sheaf = NaN
                iters = 0
                kkt_iters = 0
                status = SheafSDP.NUMERICAL_FAILURE
            else
                rethrow(e)
            end
        end
    end

    return (
        N = N,
        T = T,
        ne = ne,
        n_exp = n_exp,
        t_mosek = t_mosek,
        t_sheaf = t_sheaf,
        obj_mosek = obj_mosek,
        obj_sheaf = obj_sheaf,
        iters = iters,
        kkt_iters = kkt_iters,
        status = status,
    )
end

# Run tests
println("=" ^ 70)
println("Small-Stalk EXP Benchmark: SheafSDP vs Mosek")
println("=" ^ 70)
println()
println("Recipe A: Log-Barrier Minimum-Fuel (§3 of exp-recipes.md)")
println("Problem: N agents on path P_N, T timesteps")
println("Dynamics: planar double integrator (nx=4, nu=2)")
println("Objective: maximize Σ log(ū - |u|)")
println("Constraints: dynamics + terminal position consensus")
println()
println("EXP scaling axes: N (agents), T (timesteps) — no arm-length axis")
println()

# Test configurations: vary N and T
# Note: exp needs larger T to converge (per §9: quasi-Newton, not true NT)
# Target: at least one config where Mosek hits ~50ms
configs = [
    (5, 15),   # small: 140 exp cones
    (10, 15),  # medium: 280 exp cones
    (15, 20),  # medium-large: 570 exp cones
    (20, 20),  # larger: 760 exp cones (Mosek ~35ms)
    (25, 25),  # target ~50ms: 1200 exp cones
]

results = []
for (N, T) in configs
    println("Running N=$(N), T=$(T)...")
    r = run_exp_benchmark(N, T)
    push!(results, r)
end

# Print table
println("| N | T | #EXP | Mosek | SheafSDP | IPM | KKT | Status | vs Mosek |")
println("|---|---|------|-------|----------|-----|-----|--------|----------|")
for r in results
    mosek_ms = round(r.t_mosek * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    vs_mosek = r.status == SheafSDP.OPTIMAL || r.status == SheafSDP.NEAR_OPTIMAL ?
               "$(round(r.t_mosek / r.t_sheaf, digits=2))x" : "-"
    println("| $(r.N) | $(r.T) | $(r.n_exp) | $(mosek_ms) ms | $(sheaf_ms) ms | $(r.iters) | $(r.kkt_iters) | $(r.status) | $(vs_mosek) |")
end
println()

# Correctness check
println("Correctness check (objective agreement):")
for r in results
    if r.status == SheafSDP.OPTIMAL || r.status == SheafSDP.NEAR_OPTIMAL
        obj_diff = abs(r.obj_mosek - r.obj_sheaf)
        rel_diff = obj_diff / (abs(r.obj_mosek) + 1e-10)
        println("  N=$(r.N), T=$(r.T): |Mosek - Sheaf| = $(round(obj_diff, sigdigits=3)) (rel: $(round(rel_diff, sigdigits=3)))")
    else
        println("  N=$(r.N), T=$(r.T): $(r.status) — skipped")
    end
end
println()

# Raug sweep: find optimal raug (powers of 10)
println("-" ^ 70)
println("Raug sweep (N=15, T=20):")
println("-" ^ 70)
println("| raug | Time | IPM | KKT | Status |")
println("|------|------|-----|-----|--------|")
for log_raug in 4:9
    raug = 10.0^log_raug
    r = run_exp_benchmark(15, 20; raug=raug)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    println("| 1e$(log_raug) | $(sheaf_ms) ms | $(r.iters) | $(r.kkt_iters) | $(r.status) |")
end
println()
println("=" ^ 70)
