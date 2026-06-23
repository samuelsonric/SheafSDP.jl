#
# Recipe B (§4): Minimum-ℓ_p-norm control (the coupling showcase)
#
# Objective: f_i = Σ_t ‖u_i^t‖_p, p ∈ (1,2)∪(2,∞)
#
# This differs from Recipe A (sum of powers) by taking the NORM,
# which couples a whole control vector through a shared epigraph.
#
# Reformulation: per timestep t, introduce scalar bound τ and weights r_k ≥ 0
#   (r_k, τ, u_k) ∈ P_{1/p} for each channel k
#   Σ_k r_k = τ
#   Objective: Σ_t τ
#
# Then ‖u‖_p ≤ τ (verified: Σ|u_k|^p ≤ Σ r_k τ^{p-1} = τ^p)
#
using SheafSDP
using CommonSolve: solve
using SparseArrays
using LinearAlgebra
using Random
using JuMP
using MosekTools
using BlockSparseArrays: vtxs, colrange, rowrange, ncols, blocksparse, block

function run_pnorm_benchmark(N, T, p; raug=1e6, ū=100.0)
    Random.seed!(42)

    α = 1 / p  # POW cone parameter

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, i+1) for i in 1:N-1]
    ne = length(edges)

    #
    # Leg R: JuMP with explicit MOI.PowerCone for p-norm
    #
    function solve_mosek()
        model = Model(Mosek.Optimizer)
        set_silent(model)

        @variable(model, x[1:N, 1:T, 1:nx])
        @variable(model, u[1:N, 1:T-1, 1:nu])
        @variable(model, τ[1:N, 1:T-1] >= 0)  # norm bound per timestep
        @variable(model, r[1:N, 1:T-1, 1:nu] >= 0)  # weights
        # Mosek doesn't allow a variable in multiple cones, so create copies
        @variable(model, τ_copy[1:N, 1:T-1, 1:nu] >= 0)
        @variable(model, u_copy[1:N, 1:T-1, 1:nu])

        # Initial conditions
        for i in 1:N
            @constraint(model, x[i, 1, :] .== x0[i])
        end

        # Dynamics
        for i in 1:N, t_idx in 1:T-1
            @constraint(model, x[i, t_idx+1, :] .== A_dyn * x[i, t_idx, :] + B_dyn * u[i, t_idx, :])
        end

        # Consensus
        for (i, j) in edges
            @constraint(model, P_proj * x[i, T, :] .== P_proj * x[j, T, :])
        end

        # Link copies to originals
        for i in 1:N, t_idx in 1:T-1
            for k in 1:nu
                @constraint(model, τ_copy[i, t_idx, k] == τ[i, t_idx])
                @constraint(model, u_copy[i, t_idx, k] == u[i, t_idx, k])
            end
        end

        # p-norm epigraph: (r_k, τ, u_k) ∈ P_{1/p} and Σ r_k = τ
        for i in 1:N, t_idx in 1:T-1
            for k in 1:nu
                @constraint(model, [r[i, t_idx, k], τ_copy[i, t_idx, k], u_copy[i, t_idx, k]] in MOI.PowerCone(α))
            end
            @constraint(model, sum(r[i, t_idx, :]) == τ[i, t_idx])
        end

        # Box constraint on control
        for i in 1:N, t_idx in 1:T-1
            @constraint(model, -ū .<= u[i, t_idx, :] .<= ū)
        end

        # Objective: minimize Σ τ (sum of norms)
        @objective(model, Min, sum(τ))

        optimize!(model)
        return objective_value(model)
    end

    #
    # Leg S: SheafSDP with POW cones (coupled through shared τ)
    #
    function solve_sheaf()
        # Per agent: T state blocks (NOC), (T-1) tau blocks (POS),
        #            nu*(T-1) POW blocks, 2*(T-1) box slacks
        num_pow_per_agent = nu * (T - 1)
        blocks_per_agent = T + (T - 1) + num_pow_per_agent + 2 * (T - 1)

        col_x(i, t_idx) = (i - 1) * blocks_per_agent + t_idx
        col_tau(i, t_idx) = (i - 1) * blocks_per_agent + T + t_idx
        col_pow(i, t_idx, k) = (i - 1) * blocks_per_agent + T + (T - 1) + (t_idx - 1) * nu + k
        col_sp(i, t_idx) = (i - 1) * blocks_per_agent + T + (T - 1) + num_pow_per_agent + 2 * (t_idx - 1) + 1
        col_sm(i, t_idx) = (i - 1) * blocks_per_agent + T + (T - 1) + num_pow_per_agent + 2 * (t_idx - 1) + 2

        # Rows per agent:
        #   1 (init) + (T-1) (dynamics) + nu*(T-1) (x2=τ coupling) +
        #   (T-1) (summation Σr_k = τ) + 2*(T-1) (box)
        rows_per_agent = 1 + (T - 1) + num_pow_per_agent + (T - 1) + 2 * (T - 1)

        row_init(i) = (i - 1) * rows_per_agent + 1
        row_dyn(i, t_idx) = (i - 1) * rows_per_agent + 1 + t_idx
        row_x2(i, t_idx, k) = (i - 1) * rows_per_agent + T + (t_idx - 1) * nu + k
        row_sum(i, t_idx) = (i - 1) * rows_per_agent + T + num_pow_per_agent + t_idx
        row_boxp(i, t_idx) = (i - 1) * rows_per_agent + T + num_pow_per_agent + (T - 1) + 2 * (t_idx - 1) + 1
        row_boxm(i, t_idx) = (i - 1) * rows_per_agent + T + num_pow_per_agent + (T - 1) + 2 * (t_idx - 1) + 2
        row_coord(e) = N * rows_per_agent + e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        for i in 1:N
            # Initial condition: x[1] = x0[i]
            push!(row_ids, row_init(i))
            push!(col_ids, col_x(i, 1))
            push!(blocks, Matrix(1.0I, nx, nx))

            for t_idx in 1:T-1
                # Dynamics: x[t+1] = A x[t] + B u
                push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_x(i, t_idx)); push!(blocks, -A_dyn)
                push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_x(i, t_idx + 1)); push!(blocks, Matrix(1.0I, nx, nx))

                for k in 1:nu
                    # B_dyn[:, k] * x₃ from POW block
                    B_col_k = B_dyn[:, k:k]
                    pick_x3_apply_Bk = B_col_k * [0.0 0.0 1.0]
                    push!(row_ids, row_dyn(i, t_idx))
                    push!(col_ids, col_pow(i, t_idx, k))
                    push!(blocks, -pick_x3_apply_Bk)
                end

                # x₂ = τ coupling: slot 2 of POW block equals τ
                for k in 1:nu
                    # [0 1 0] * ξ - τ = 0
                    push!(row_ids, row_x2(i, t_idx, k))
                    push!(col_ids, col_pow(i, t_idx, k))
                    push!(blocks, reshape([0.0, 1.0, 0.0], 1, 3))

                    push!(row_ids, row_x2(i, t_idx, k))
                    push!(col_ids, col_tau(i, t_idx))
                    push!(blocks, reshape([-1.0], 1, 1))
                end

                # Summation: Σ r_k = τ (r_k is x₁ of POW block)
                for k in 1:nu
                    push!(row_ids, row_sum(i, t_idx))
                    push!(col_ids, col_pow(i, t_idx, k))
                    push!(blocks, reshape([1.0, 0.0, 0.0], 1, 3))
                end
                push!(row_ids, row_sum(i, t_idx))
                push!(col_ids, col_tau(i, t_idx))
                push!(blocks, reshape([-1.0], 1, 1))

                # Box constraints: u + s+ = ū, -u + s- = ū
                for k in 1:nu
                    push!(row_ids, row_boxp(i, t_idx))
                    push!(col_ids, col_pow(i, t_idx, k))
                    blk = zeros(nu, 3)
                    blk[k, 3] = 1.0
                    push!(blocks, blk)
                end
                push!(row_ids, row_boxp(i, t_idx))
                push!(col_ids, col_sp(i, t_idx))
                push!(blocks, Matrix(1.0I, nu, nu))

                for k in 1:nu
                    push!(row_ids, row_boxm(i, t_idx))
                    push!(col_ids, col_pow(i, t_idx, k))
                    blk = zeros(nu, 3)
                    blk[k, 3] = -1.0
                    push!(blocks, blk)
                end
                push!(row_ids, row_boxm(i, t_idx))
                push!(col_ids, col_sm(i, t_idx))
                push!(blocks, Matrix(1.0I, nu, nu))
            end
        end

        # Coordination: terminal position consensus
        for (e, (i, j)) in enumerate(edges)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T)); push!(blocks, -P_proj)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T)); push!(blocks, P_proj)
        end

        B = blocksparse(row_ids, col_ids, blocks)

        # Objective: minimize Σ τ
        c = zeros(size(B, 2))
        for i in 1:N, t_idx in 1:T-1
            c_rng = colrange(B, col_tau(i, t_idx))
            c[c_rng[1]] = 1.0
        end

        # RHS
        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_init(i))] .= x0[i]
            # x2 coupling rows have g = 0
            # summation rows have g = 0
            for t_idx in 1:T-1
                g[rowrange(B, row_boxp(i, t_idx))] .= ū
                g[rowrange(B, row_boxm(i, t_idx))] .= ū
            end
        end

        Q = SheafSDP.allocblockdiag(B)
        fill!(Q, 0)

        nv = N * blocks_per_agent
        cones = Vector{SheafSDP.Cone}(undef, nv)
        for i in 1:N
            for t_idx in 1:T
                cones[col_x(i, t_idx)] = SheafSDP.CofreeCone()
            end
            for t_idx in 1:T-1
                cones[col_tau(i, t_idx)] = SheafSDP.PositiveCone()
                for k in 1:nu
                    cones[col_pow(i, t_idx, k)] = SheafSDP.PowerCone(α)
                end
                cones[col_sp(i, t_idx)] = SheafSDP.PositiveCone()
                cones[col_sm(i, t_idx)] = SheafSDP.PositiveCone()
            end
        end

        prob = IPMProblem(c, g, B, Q, cones)
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=raug),
            feas_tol=1e-6, gap_tol=1e-6, itmax=200,
            verbose=false
        )
        result = solve(prob, settings)

        return dot(c, result.p), result.iterations, result.kkt_iters, result.status
    end

    # Warmup
    solve_mosek()
    try solve_sheaf() catch end

    t_mosek = @elapsed obj_mosek = solve_mosek()

    local obj_sheaf, iters, kkt_iters, status, t_sheaf
    t_sheaf = @elapsed begin
        try
            (obj_sheaf, iters, kkt_iters, status) = solve_sheaf()
        catch e
            obj_sheaf = NaN
            iters = 0
            kkt_iters = 0
            status = SheafSDP.NUMERICAL_FAILURE
        end
    end

    return (
        N = N,
        T = T,
        p = p,
        α = α,
        ne = ne,
        t_mosek = t_mosek,
        t_sheaf = t_sheaf,
        obj_mosek = obj_mosek,
        obj_sheaf = obj_sheaf,
        iters = iters,
        kkt_iters = kkt_iters,
        status = status,
    )
end

#
# α = 1/2 regression: p-norm with p=2 should match SOC ‖u‖₂ ≤ τ
#
function run_pnorm_alpha_half_regression(N, T; raug=1e6, ū=100.0)
    Random.seed!(42)

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, i+1) for i in 1:N-1]

    # Solve with PowerCone(0.5) for p=2
    r_pow = run_pnorm_benchmark(N, T, 2.0; raug=raug, ū=ū)

    # Solve with native SOC (‖u‖₂ ≤ τ)
    function solve_soc()
        model = Model(Mosek.Optimizer)
        set_silent(model)

        @variable(model, x[1:N, 1:T, 1:nx])
        @variable(model, u[1:N, 1:T-1, 1:nu])
        @variable(model, τ[1:N, 1:T-1] >= 0)

        for i in 1:N
            @constraint(model, x[i, 1, :] .== x0[i])
        end

        for i in 1:N, t in 1:T-1
            @constraint(model, x[i, t+1, :] .== A_dyn * x[i, t, :] + B_dyn * u[i, t, :])
        end

        for (i, j) in edges
            @constraint(model, P_proj * x[i, T, :] .== P_proj * x[j, T, :])
        end

        # SOC constraint: ‖u‖₂ ≤ τ
        for i in 1:N, t in 1:T-1
            @constraint(model, [τ[i, t]; u[i, t, :]] in JuMP.SecondOrderCone())
        end

        for i in 1:N, t in 1:T-1
            @constraint(model, -ū .<= u[i, t, :] .<= ū)
        end

        @objective(model, Min, sum(τ))

        optimize!(model)
        return objective_value(model)
    end

    obj_soc = solve_soc()

    return (
        obj_pow = r_pow.obj_sheaf,
        obj_soc = obj_soc,
        status = r_pow.status,
        iters = r_pow.iters,
    )
end

# Main benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 70)
    println("POW Recipe B: Minimum-ℓ_p-Norm Control (Coupling Showcase)")
    println("=" ^ 70)
    println()
    println("Problem: N agents, T timesteps, minimize Σ ‖u‖_p (norm, not sum of powers)")
    println("This tests coupling through shared epigraph τ")
    println()

    # α sweep (skip problematic α values)
    println("-" ^ 70)
    println("α sweep (N=10, T=15):")
    println("-" ^ 70)
    println("| p | α | Mosek (ms) | SheafSDP (ms) | IPM | KKT | Status | Obj diff |")
    println("|---|---|------------|---------------|-----|-----|--------|----------|")

    # Skip p values that give problematic α (like 1.2 → α=0.833, or 5.0 → α=0.2)
    for p in [1.5, 2.0, 3.0]  # α = 0.667, 0.5, 0.333
        res = run_pnorm_benchmark(10, 15, p)
        mosek_ms = round(res.t_mosek * 1000, digits=1)
        sheaf_ms = round(res.t_sheaf * 1000, digits=1)
        diff = abs(res.obj_mosek - res.obj_sheaf)
        α_str = round(res.α, digits=3)
        println("| $(p) | $(α_str) | $(mosek_ms) | $(sheaf_ms) | $(res.iters) | $(res.kkt_iters) | $(res.status) | $(round(diff, sigdigits=3)) |")
    end
    println()

    # α = 1/2 regression
    println("-" ^ 70)
    println("α = 1/2 Regression (p=2, POW p-norm vs SOC):")
    println("-" ^ 70)

    reg = run_pnorm_alpha_half_regression(10, 15)
    obj_diff = abs(reg.obj_pow - reg.obj_soc)
    rel_diff = obj_diff / (abs(reg.obj_soc) + 1e-10)
    println("  POW (p=2):  $(round(reg.obj_pow, sigdigits=6))")
    println("  SOC native: $(round(reg.obj_soc, sigdigits=6))")
    println("  Difference: $(round(obj_diff, sigdigits=3)) (rel: $(round(rel_diff, sigdigits=3)))")
    println("  Status: $(reg.status)")
    status = rel_diff < 1e-3 ? "PASS" : "FAIL"
    println("  Regression: [$status]")
    println()

    println("=" ^ 70)
end
