#
# Recipe A (§3): Minimum-effort ℓ_p^p control
#
# Objective: f_i = Σ_t Σ_k |u_i^{t}[k]|^p, p ∈ (1,2)∪(2,∞)
#
# Reformulation: Power epigraph per scalar control
#   t_k ≥ |u_k|^p ⟺ (t_k, 1, u_k) ∈ P_{1/p}
#   Objective: Σ t_k
#
# Three-backend oracle: R = JuMP MOI.PowerCone (Mosek), S = SheafSDP
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

function run_pow_benchmark(N, T, p; raug=1e6, ū=100.0)
    Random.seed!(42)

    α = 1 / p  # POW cone parameter

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

    x0 = [randn(nx) for _ in 1:N]
    # Path graph P_N for coordination
    edges = [(i, i+1) for i in 1:N-1]
    ne = length(edges)

    # Count pow cones: nu * (T-1) per agent
    n_pow = N * nu * (T - 1)

    #
    # Leg R: JuMP with explicit MOI.PowerCone
    #
    # MOI.PowerCone(α): (x, y, z) with x^α y^(1-α) ≥ |z|, x,y ≥ 0
    # Power epigraph t ≥ |u|^p: (t, 1, u) ∈ P_{1/p}
    #
    function solve_mosek()
        model = Model(Mosek.Optimizer)
        set_silent(model)

        @variable(model, x[1:N, 1:T, 1:nx])
        @variable(model, u[1:N, 1:T-1, 1:nu])
        @variable(model, t[1:N, 1:T-1, 1:nu] >= 0)  # epigraph variables

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

        # Power epigraph: t ≥ |u|^p via (t, 1, u) ∈ P_{1/p}
        for i in 1:N, t_idx in 1:T-1, k in 1:nu
            @constraint(model, [t[i, t_idx, k], 1, u[i, t_idx, k]] in MOI.PowerCone(α))
        end

        # Box constraint on control
        for i in 1:N, t_idx in 1:T-1
            @constraint(model, -ū .<= u[i, t_idx, :] .<= ū)
        end

        # Objective: minimize Σ t (sum of powers)
        @objective(model, Min, sum(t))

        optimize!(model)
        return objective_value(model)
    end

    #
    # Leg S: SheafSDP with POW cones
    #
    # Per agent: T state blocks (NOC), nu*(T-1) POW blocks (t, 1, u), 2*(T-1) box slacks
    #
    function solve_sheaf()
        num_pow_per_agent = nu * (T - 1)
        blocks_per_agent = T + num_pow_per_agent + 2 * (T - 1)

        col_x(i, t_idx) = (i - 1) * blocks_per_agent + t_idx
        col_pow(i, t_idx, k) = (i - 1) * blocks_per_agent + T + (t_idx - 1) * nu + k
        col_sp(i, t_idx) = (i - 1) * blocks_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 1
        col_sm(i, t_idx) = (i - 1) * blocks_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 2

        # Rows per agent: 1 (init) + (T-1) (dynamics) + num_pow (x2=1) + 2*(T-1) (box)
        rows_per_agent = 1 + (T - 1) + num_pow_per_agent + 2 * (T - 1)

        row_init(i) = (i - 1) * rows_per_agent + 1
        row_dyn(i, t_idx) = (i - 1) * rows_per_agent + 1 + t_idx
        row_x2(i, t_idx, k) = (i - 1) * rows_per_agent + T + (t_idx - 1) * nu + k
        row_boxp(i, t_idx) = (i - 1) * rows_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 1
        row_boxm(i, t_idx) = (i - 1) * rows_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 2
        row_coord(e) = N * rows_per_agent + e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        # Selector matrices for POW block (t, 1, u) -> picks u component k
        # POW block is (x₁, x₂, x₃) = (t, 1, u)
        # Dynamics needs B_dyn * u, where u_k comes from x₃ of POW block (i,t,k)
        sel_u(k) = reshape([0.0, 0.0, k == 1 ? 1.0 : 0.0, 0.0, 0.0, k == 2 ? 1.0 : 0.0], nx, 3)

        for i in 1:N
            # Initial condition: x[1] = x0[i]
            push!(row_ids, row_init(i))
            push!(col_ids, col_x(i, 1))
            push!(blocks, Matrix(1.0I, nx, nx))

            for t_idx in 1:T-1
                # Dynamics: x[t+1] = A x[t] + B u
                # where u_k comes from x₃ slot of POW block
                push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_x(i, t_idx)); push!(blocks, -A_dyn)
                push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_x(i, t_idx + 1)); push!(blocks, Matrix(1.0I, nx, nx))

                for k in 1:nu
                    # B_dyn[:, k] * x₃ from POW block (i, t_idx, k)
                    B_col_k = B_dyn[:, k:k]  # nx × 1
                    # Block picks x₃ (slot 3) and applies B_dyn column k
                    pick_x3_apply_Bk = B_col_k * [0.0 0.0 1.0]  # nx × 3
                    push!(row_ids, row_dyn(i, t_idx))
                    push!(col_ids, col_pow(i, t_idx, k))
                    push!(blocks, -pick_x3_apply_Bk)
                end

                for k in 1:nu
                    # x₂ = 1 constraint: [0 1 0] * ξ = 1
                    push!(row_ids, row_x2(i, t_idx, k))
                    push!(col_ids, col_pow(i, t_idx, k))
                    push!(blocks, reshape([0.0, 1.0, 0.0], 1, 3))
                end

                # Box constraints: u + s+ = ū, -u + s- = ū
                # u_k = x₃ of POW block (i, t_idx, k)
                pick_x3 = reshape([0.0, 0.0, 1.0], 1, 3)
                for k in 1:nu
                    # u_k + s+_k = ū
                    push!(row_ids, row_boxp(i, t_idx))
                    push!(col_ids, col_pow(i, t_idx, k))
                    # Only pick component k
                    blk = zeros(nu, 3)
                    blk[k, 3] = 1.0
                    push!(blocks, blk)
                end
                push!(row_ids, row_boxp(i, t_idx))
                push!(col_ids, col_sp(i, t_idx))
                push!(blocks, Matrix(1.0I, nu, nu))

                for k in 1:nu
                    # -u_k + s-_k = ū
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

        # Objective: minimize Σ t = Σ x₁ of POW blocks
        c = zeros(size(B, 2))
        for i in 1:N, t_idx in 1:T-1, k in 1:nu
            c_rng = colrange(B, col_pow(i, t_idx, k))
            c[c_rng[1]] = 1.0  # x₁ slot
        end

        # RHS: g[init] = x0, g[dyn] = 0, g[x2] = 1, g[box] = ū, g[coord] = 0
        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_init(i))] .= x0[i]
            for t_idx in 1:T-1, k in 1:nu
                g[rowrange(B, row_x2(i, t_idx, k))] .= 1.0
            end
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
        n_pow = n_pow,
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
# α = 1/2 regression: POW with p=2 should match SOC/QP (rotated SOC)
#
function run_alpha_half_regression(N, T; raug=1e6, ū=100.0)
    Random.seed!(42)

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, i+1) for i in 1:N-1]

    # Solve with PowerCone(0.5) for p=2: Σ |u|² = ‖u‖²
    r_pow = run_pow_benchmark(N, T, 2.0; raug=raug, ū=ū)

    # Solve with native quadratic (no cones, just Q matrix)
    function solve_qp()
        model = Model(Mosek.Optimizer)
        set_silent(model)

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

        for i in 1:N, t in 1:T-1
            @constraint(model, -ū .<= u[i, t, :] .<= ū)
        end

        # Objective: Σ |u|² (sum of squares)
        @objective(model, Min, sum(u[i, t, k]^2 for i in 1:N for t in 1:T-1 for k in 1:nu))

        optimize!(model)
        return objective_value(model)
    end

    obj_qp = solve_qp()

    return (
        obj_pow = r_pow.obj_sheaf,
        obj_qp = obj_qp,
        status = r_pow.status,
        iters = r_pow.iters,
    )
end

# Main benchmark
println("=" ^ 70)
println("POW (ℓ_p^p Control) Benchmark: SheafSDP vs Mosek")
println("=" ^ 70)
println()
println("Recipe A (§3): Minimum-effort ℓ_p^p control")
println("Problem: N agents on path P_N, T timesteps")
println("Dynamics: planar double integrator (nx=4, nu=2)")
println("Objective: minimize Σ |u|^p")
println("Constraints: dynamics + box |u| ≤ 100 + terminal position consensus")
println()

# Test configurations with different p values
println("-" ^ 70)
println("Scaling sweep (p=1.5, α=0.667):")
println("-" ^ 70)
println("| N | T | #POW | Mosek | SheafSDP | IPM | KKT | Status | vs Mosek |")
println("|---|---|------|-------|----------|-----|-----|--------|----------|")

for (N, T) in [(5, 10), (10, 15), (15, 20), (20, 20)]
    r = run_pow_benchmark(N, T, 1.5; raug=1e6)
    mosek_ms = round(r.t_mosek * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    vs = r.status == SheafSDP.OPTIMAL || r.status == SheafSDP.NEAR_OPTIMAL ?
         "$(round(r.t_mosek / r.t_sheaf, digits=2))x" : "-"
    println("| $(r.N) | $(r.T) | $(r.n_pow) | $(mosek_ms) ms | $(sheaf_ms) ms | $(r.iters) | $(r.kkt_iters) | $(r.status) | $(vs) |")
end
println()

# Sweep α (different p values)
println("-" ^ 70)
println("α sweep (N=10, T=15):")
println("-" ^ 70)
println("| p | α | Mosek | SheafSDP | IPM | KKT | Status | Obj diff |")
println("|---|---|-------|----------|-----|-----|--------|----------|")

for p in [1.2, 1.5, 2.0, 3.0, 5.0]
    r = run_pow_benchmark(10, 15, p; raug=1e6)
    mosek_ms = round(r.t_mosek * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    obj_diff = abs(r.obj_mosek - r.obj_sheaf)
    α_str = round(r.α, digits=3)
    println("| $(p) | $(α_str) | $(mosek_ms) ms | $(sheaf_ms) ms | $(r.iters) | $(r.kkt_iters) | $(r.status) | $(round(obj_diff, sigdigits=3)) |")
end
println()

# α = 1/2 regression test
println("-" ^ 70)
println("α = 1/2 Regression (p=2, POW vs QP):")
println("-" ^ 70)

reg = run_alpha_half_regression(10, 15)
obj_diff = abs(reg.obj_pow - reg.obj_qp)
rel_diff = obj_diff / (abs(reg.obj_qp) + 1e-10)
println("  POW (p=2):  $(round(reg.obj_pow, sigdigits=6))")
println("  QP native:  $(round(reg.obj_qp, sigdigits=6))")
println("  Difference: $(round(obj_diff, sigdigits=3)) (rel: $(round(rel_diff, sigdigits=3)))")
println("  Status: $(reg.status)")
status = rel_diff < 1e-3 ? "PASS" : "FAIL"
println("  Regression: [$status]")
println()

println("=" ^ 70)
