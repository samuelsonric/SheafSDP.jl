#
# Recipe §6: SOC (un-squared norm / group-sparse control)
#
# Objective: f_i = Σ_t ‖u_i^t‖₂ (whole control vectors switch on/off)
#
# Reformulation: SOC epigraph bundle
#   ζ_i^t = (s_i^t; u_i^t) ∈ Q^{1+m}, c = 1 on s_i^t
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

function run_benchmark(N, T; raug=100.0, ū=100.0)
    Random.seed!(42)

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, j) for i in 1:N for j in i+1:N]
    ne = length(edges)

    # Mosek reference (SOC via JuMP's native support)
    function solve_mosek()
        model = Model(Mosek.Optimizer)
        set_silent(model)

        @variable(model, x[1:N, 1:T, 1:nx])
        @variable(model, u[1:N, 1:T-1, 1:nu])
        @variable(model, s[1:N, 1:T-1] >= 0)  # epigraph variables

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

        # SOC constraint: ‖u‖₂ ≤ s
        for i in 1:N, t in 1:T-1
            @constraint(model, [s[i,t]; u[i,t,:]] in JuMP.SecondOrderCone())
        end

        # Box constraint on control
        for i in 1:N, t in 1:T-1
            @constraint(model, -ū .<= u[i, t, :] .<= ū)
        end

        # Objective: Σ s (sum of norms)
        @objective(model, Min, sum(s))

        optimize!(model)
        return objective_value(model)
    end

    # SheafSDP with SOC cones
    # Column layout per agent:
    #   T state blocks x_i^t (:NOC)
    #   (T-1) SOC blocks ζ_i^t = (s; u) (:SOC, dim 1+nu)
    #   (T-1) * 2 box slack blocks s+, s- (:POS, dim nu each)
    function solve_sheaf()
        blocks_per_agent = T + 3 * (T - 1)  # x's + ζ's + box slacks

        col_x(i, t) = (i - 1) * blocks_per_agent + t
        col_ζ(i, t) = (i - 1) * blocks_per_agent + T + 3 * (t - 1) + 1      # SOC block (s; u)
        col_sp(i, t) = (i - 1) * blocks_per_agent + T + 3 * (t - 1) + 2     # box slack +
        col_sm(i, t) = (i - 1) * blocks_per_agent + T + 3 * (t - 1) + 3     # box slack -

        # Row layout per agent:
        #   1 init row
        #   (T-1) dynamics rows
        #   (T-1) box+ rows: u + s+ = ū
        #   (T-1) box- rows: -u + s- = ū
        rows_per_agent = 1 + 3 * (T - 1)

        row_init(i) = (i - 1) * rows_per_agent + 1
        row_dyn(i, t) = (i - 1) * rows_per_agent + 1 + t
        row_boxp(i, t) = (i - 1) * rows_per_agent + T + (t - 1) + 1
        row_boxm(i, t) = (i - 1) * rows_per_agent + T + (T - 1) + (t - 1) + 1
        row_coord(e) = N * rows_per_agent + e

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        # Matrix to extract u from ζ = (s; u): [0 I_nu] picks the tail
        # Scale by 1/√2 for isometric SOC representation
        invrt2 = 1 / sqrt(2.0)
        extract_u = [zeros(nx, 1) [B_dyn; zeros(nx - size(B_dyn, 1), nu)]]
        # Actually we need: dynamics uses B_dyn * u, so we need [0 B_dyn] on ζ
        B_on_ζ = [zeros(nx, 1) B_dyn] .* invrt2  # nx × (1+nu), extracts and applies B to u part

        # For box constraints: extract u from ζ
        extract_u_box = [zeros(nu, 1) Matrix(1.0I, nu, nu)] .* invrt2  # nu × (1+nu)

        for i in 1:N
            # init: x_i^1 = x0[i]
            push!(row_ids, row_init(i))
            push!(col_ids, col_x(i, 1))
            push!(blocks, Matrix(1.0I, nx, nx))

            for t in 1:T-1
                # dynamics: x_i^{t+1} - A x_i^t - B u_i^t = 0
                # where u comes from the tail of ζ
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t)); push!(blocks, -A_dyn)
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_x(i, t + 1)); push!(blocks, Matrix(1.0I, nx, nx))
                push!(row_ids, row_dyn(i, t)); push!(col_ids, col_ζ(i, t)); push!(blocks, -B_on_ζ)

                # box+: u + s+ = ū  (extract u from ζ)
                push!(row_ids, row_boxp(i, t)); push!(col_ids, col_ζ(i, t)); push!(blocks, extract_u_box)
                push!(row_ids, row_boxp(i, t)); push!(col_ids, col_sp(i, t)); push!(blocks, Matrix(1.0I, nu, nu))

                # box-: -u + s- = ū
                push!(row_ids, row_boxm(i, t)); push!(col_ids, col_ζ(i, t)); push!(blocks, -extract_u_box)
                push!(row_ids, row_boxm(i, t)); push!(col_ids, col_sm(i, t)); push!(blocks, Matrix(1.0I, nu, nu))
            end
        end

        # Coordination: P x_i^T - P x_j^T = 0
        for (e, (i, j)) in enumerate(edges)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T)); push!(blocks, -P_proj)
            push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T)); push!(blocks, P_proj)
        end

        B = blocksparse(row_ids, col_ids, blocks)

        # Cost vector: c = 1 on the head (s) of each ζ, 0 elsewhere
        # Scale by 1/√2 for isometric SOC representation
        c = zeros(size(B, 2))
        for i in 1:N, t in 1:T-1
            # ζ block is (s; u), so first element is s
            ζ_range = colrange(B, col_ζ(i, t))
            c[ζ_range[1]] = invrt2  # only the head, scaled
        end

        # RHS
        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_init(i))] .= x0[i]
            for t in 1:T-1
                g[rowrange(B, row_boxp(i, t))] .= ū
                g[rowrange(B, row_boxm(i, t))] .= ū
            end
        end

        # Q = 0 (no quadratic term)
        Q = SheafSDP.allocblockdiag(B)
        fill!(Q, 0)

        # Cones
        nv = N * blocks_per_agent
        cones = Vector{SheafSDP.Cone}(undef, nv)
        for i in 1:N
            for t in 1:T
                cones[col_x(i, t)] = SheafSDP.CofreeCone()
            end
            for t in 1:T-1
                cones[col_ζ(i, t)] = SheafSDP.SecondOrderCone()   # (s; u) ∈ Q^{1+nu}
                cones[col_sp(i, t)] = SheafSDP.PositiveCone()
                cones[col_sm(i, t)] = SheafSDP.PositiveCone()
            end
        end

        prob = IPMProblem(c, g, B, Q, cones)
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=raug),
            feas_tol=1e-8, gap_tol=1e-8, itmax=200,
            verbose=false
        )
        result = solve(prob, settings)

        return dot(c, result.p), result.iterations, result.kkt_iters, result.status
    end

    # Warmup
    solve_mosek()
    solve_sheaf()

    t_mosek = @elapsed obj_mosek = solve_mosek()
    t_sheaf = @elapsed (obj_sheaf, iters, kkt_iters, status) = solve_sheaf()

    return (
        N = N,
        T = T,
        ne = ne,
        t_mosek = t_mosek,
        t_sheaf = t_sheaf,
        iters = iters,
        kkt_iters = kkt_iters,
        status = status,
        obj_mosek = obj_mosek,
        obj_sheaf = obj_sheaf,
    )
end

# Run benchmark sweep
println("SOC (Group-Sparse Control) Benchmark: SheafSDP vs Mosek")
println("========================================================")
println()
println("Problem: N agents on complete graph K_N, T timesteps")
println("Dynamics: planar double integrator (nx=4, nu=2)")
println("Objective: Σ ‖u‖₂ (sum of norms, group-sparse)")
println("Constraints: dynamics + box |u| ≤ 100 + terminal position consensus")
println("Parameters: raug=1e6")
println()

results = []
for (N, T) in [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)]
    r = run_benchmark(N, T; raug=1e6)
    push!(results, r)
    if r.status != SheafSDP.OPTIMAL && r.status != SheafSDP.NEAR_OPTIMAL
        println("Warning: N=$(r.N), T=$(r.T) status=$(r.status)")
    end
end

# Print table
println("| N,T | Edges | Mosek | SheafSDP | KKT iters | vs Mosek |")
println("|-----|-------|-------|----------|-----------|----------|")
for r in results
    mosek_ms = round(r.t_mosek * 1000, digits=1)
    sheaf_ms = round(r.t_sheaf * 1000, digits=1)
    vs_mosek = round(r.t_mosek / r.t_sheaf, digits=1)
    println("| $(r.N),$(r.T) | $(r.ne) | $(mosek_ms) ms | $(sheaf_ms) ms | $(r.kkt_iters) | $(vs_mosek)x |")
end
println()

# Verify correctness
println("Correctness check (objective difference):")
for r in results
    diff = abs(r.obj_mosek - r.obj_sheaf)
    println("  N=$(r.N), T=$(r.T): |Mosek - SheafSDP| = $(round(diff, sigdigits=3))")
end
