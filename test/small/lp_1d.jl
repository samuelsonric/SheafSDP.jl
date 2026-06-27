#
# Same as lp.jl but with 1D positive cones instead of nu-dimensional
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
using BlockSparseArrays: vtxs, colrange, rowrange, ncols, blocksparse, block

function run_benchmark(N, T; raug=1e7, ū=100.0)
    Random.seed!(42)

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, j) for i in 1:N for j in i+1:N]
    ne = length(edges)

    # High-level formulation (direct ℓ₁ via epigraph)
    function solve_highlevel(optimizer)
        model = Model(optimizer)
        set_silent(model)

        @variable(model, x[1:N, 1:T, 1:nx])
        @variable(model, u[1:N, 1:T-1, 1:nu])
        @variable(model, t_abs[1:N, 1:T-1, 1:nu] >= 0)

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
            @constraint(model, u[i, t, :] .<= t_abs[i, t, :])
            @constraint(model, -u[i, t, :] .<= t_abs[i, t, :])
        end

        for i in 1:N, t in 1:T-1
            @constraint(model, t_abs[i, t, :] .<= ū)
        end

        @objective(model, Min, sum(t_abs))

        optimize!(model)
        return objective_value(model)
    end

    # SheafSDP with 1D POS cones
    function solve_sheaf()
        # Each agent has:
        #   T state blocks (nx each)
        #   (T-1) * 3 * nu scalar positive cones (u⁺, u⁻, w for each control component)
        blocks_per_agent = T + 3 * nu * (T - 1)

        col_x(i, t) = (i - 1) * blocks_per_agent + t
        # For 1D cones: index by (i, t, k) where k ∈ 1:nu
        col_up(i, t, k) = (i - 1) * blocks_per_agent + T + 3 * nu * (t - 1) + k
        col_um(i, t, k) = (i - 1) * blocks_per_agent + T + 3 * nu * (t - 1) + nu + k
        col_w(i, t, k)  = (i - 1) * blocks_per_agent + T + 3 * nu * (t - 1) + 2 * nu + k

        # Rows: dynamics are still nx-dimensional, but box constraints are now 1D each
        rows_per_agent = T + nu * (T - 1)

        row_init(i) = (i - 1) * rows_per_agent + 1
        row_dyn(i, t) = (i - 1) * rows_per_agent + 1 + t
        row_box(i, t, k) = (i - 1) * rows_per_agent + T + nu * (t - 1) + k
        row_coord(e) = N * rows_per_agent + e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        for i in 1:N
            push!(row_ids, row_init(i))
            push!(col_ids, col_x(i, 1))
            push!(blocks, Matrix(1.0I, nx, nx))

            for t in 1:T-1
                # Dynamics: x_{t+1} = A x_t + B(u⁺ - u⁻)
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t)); push!(blocks, -A_dyn)
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t + 1)); push!(blocks, Matrix(1.0I, nx, nx))

                # B_dyn columns split into 1D blocks
                for k in 1:nu
                    push!(row_ids, row_dyn(i, t)); push!(col_ids, col_up(i, t, k)); push!(blocks, -B_dyn[:, k:k])
                    push!(row_ids, row_dyn(i, t)); push!(col_ids, col_um(i, t, k)); push!(blocks,  B_dyn[:, k:k])
                end

                # Box constraints: u⁺_k + u⁻_k + w_k = ū (one row per k)
                for k in 1:nu
                    push!(row_ids, row_box(i, t, k)); push!(col_ids, col_up(i, t, k)); push!(blocks, ones(1, 1))
                    push!(row_ids, row_box(i, t, k)); push!(col_ids, col_um(i, t, k)); push!(blocks, ones(1, 1))
                    push!(row_ids, row_box(i, t, k)); push!(col_ids, col_w(i, t, k));  push!(blocks, ones(1, 1))
                end
            end
        end

        for (e, (i, j)) in enumerate(edges)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T)); push!(blocks, -P_proj)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T)); push!(blocks, P_proj)
        end

        B = blocksparse(row_ids, col_ids, blocks)

        c = zeros(size(B, 2))
        for i in 1:N, t in 1:T-1, k in 1:nu
            c[colrange(B, col_up(i, t, k))] .= 1.0
            c[colrange(B, col_um(i, t, k))] .= 1.0
        end

        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_init(i))] .= x0[i]
            for t in 1:T-1, k in 1:nu
                g[rowrange(B, row_box(i, t, k))] .= ū
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
            for t in 1:T-1, k in 1:nu
                cones[col_up(i, t, k)] = PositiveCone()
                cones[col_um(i, t, k)] = PositiveCone()
                cones[col_w(i, t, k)]  = PositiveCone()
            end
        end

        prob = IPMProblem(Q, B, c, g, cones)
        settings = IPMSettings{Float64}(kkt=UzawaSettings{Float64}(raug=raug), feas_tol=1e-6, gap_tol=1e-6, itmax=100)
        result = solve(prob, settings)

        return dot(c, result.p), result.ipm_niter, result.kkt_niter, result.status
    end

    # Warmup
    solve_highlevel(HiGHS.Optimizer)
    solve_highlevel(Mosek.Optimizer)
    solve_sheaf()

    t_highs = @elapsed obj_highs = solve_highlevel(HiGHS.Optimizer)
    t_mosek = @elapsed obj_mosek = solve_highlevel(Mosek.Optimizer)
    t_sheaf = @elapsed (obj_sheaf, iters, kkt_iters, status) = solve_sheaf()

    return (
        N = N,
        T = T,
        ne = ne,
        t_highs = t_highs,
        t_mosek = t_mosek,
        t_sheaf = t_sheaf,
        iters = iters,
        kkt_iters = kkt_iters,
        status = status,
        obj_highs = obj_highs,
        obj_mosek = obj_mosek,
        obj_sheaf = obj_sheaf,
    )
end

# Run benchmark sweep
println("POS/ℓ₁ Consensus Benchmark (1D cones): SheafSDP vs HiGHS vs Mosek")
println("==================================================================")
println()
println("Problem: N agents on complete graph K_N, T timesteps")
println("Dynamics: planar double integrator (nx=4, nu=2)")
println("Objective: minimum fuel ‖u‖₁")
println("Constraints: dynamics + box |u| ≤ 100 + terminal position consensus")
println("Note: Each positive variable is a separate 1D cone")
println()

results = []
for (N, T) in [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30), (35, 35), (40, 40)]
    r = run_benchmark(N, T)
    push!(results, r)
    if r.status != SheafSDP.OPTIMAL
        println("Warning: N=$(r.N), T=$(r.T) status=$(r.status)")
    end
end

# Print table
println("| N,T | Edges | HiGHS | Mosek | SheafSDP | KKT iters | vs HiGHS | vs Mosek |")
println("|-----|-------|-------|-------|----------|-----------|----------|----------|")
for r in results
    highs_ms = round(r.t_highs * 1000, digits=1)
    mosek_ms = round(r.t_mosek * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    vs_highs = round(r.t_highs / r.t_sheaf, digits=1)
    vs_mosek = round(r.t_mosek / r.t_sheaf, digits=1)
    println("| $(r.N),$(r.T) | $(r.ne) | $(highs_ms) ms | $(mosek_ms) ms | $(sheaf_ms) ms | $(r.kkt_iters) | $(vs_highs)x | $(vs_mosek)x |")
end
println()

# Verify correctness
println("Correctness check (objective difference):")
for r in results
    diff_h = abs(r.obj_highs - r.obj_sheaf)
    diff_m = abs(r.obj_mosek - r.obj_sheaf)
    println("  N=$(r.N), T=$(r.T): |HiGHS - SheafSDP| = $(round(diff_h, sigdigits=3)), |Mosek - SheafSDP| = $(round(diff_m, sigdigits=3))")
end
