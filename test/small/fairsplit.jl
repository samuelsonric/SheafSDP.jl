#
# Recipe C (§5): Proportional-fair resource split
#
# The showcase for power cones + sheaf coordination reinforcing each other.
#
# Setup: Each agent i holds a terminal allocation a_i ∈ R+^m
# Coordination: Linear consensus δa = 0 (agents agree on shared allocation)
# Objective: Maximize geometric mean (∏_k a_{i,k})^{1/m} — proportional fairness
#
# Geometric mean tower (m=3):
#   (a₂, a₃, w) ∈ P_{1/2}     gives w ≤ (a₂ a₃)^{1/2}
#   (a₁, w, g_i) ∈ P_{1/3}    gives g_i ≤ a₁^{1/3} w^{2/3} = (a₁ a₂ a₃)^{1/3}
#
# This exercises multiple distinct α in one problem — the structural fact
# that separates power from exp (§5 of pow-recipes.md).
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

function run_fairsplit_benchmark(N; m=3, raug=1e6)
    Random.seed!(42)

    # Path graph for coordination
    edges = [(i, i+1) for i in 1:N-1]
    ne = length(edges)

    # Budget constraints: Σ_k a_k ≤ b_i (each agent has a budget)
    budgets = [10.0 + 5.0 * rand() for _ in 1:N]

    #
    # Leg R: JuMP with explicit MOI.PowerCone
    #
    function solve_mosek()
        model = Model(Mosek.Optimizer)
        set_silent(model)

        @variable(model, a[1:N, 1:m] >= 0)  # allocations
        @variable(model, g[1:N])             # geometric mean epigraph

        # Auxiliary variables for tower (to avoid variable in multiple cones)
        @variable(model, w[1:N] >= 0)        # intermediate from first cone
        @variable(model, w2[1:N] >= 0)       # copy for second cone
        @variable(model, a2_copy[1:N] >= 0)  # copy of a₂
        @variable(model, a3_copy[1:N] >= 0)  # copy of a₃
        @variable(model, a1_copy[1:N] >= 0)  # copy of a₁

        # Budget constraints
        for i in 1:N
            @constraint(model, sum(a[i, :]) <= budgets[i])
        end

        # Consensus: agents agree on allocation at terminal
        for (i, j) in edges
            @constraint(model, a[i, :] .== a[j, :])
        end

        # Link copies to originals
        for i in 1:N
            @constraint(model, a1_copy[i] == a[i, 1])
            @constraint(model, a2_copy[i] == a[i, 2])
            @constraint(model, a3_copy[i] == a[i, 3])
            @constraint(model, w2[i] == w[i])
        end

        # Geometric mean tower (m=3):
        # (a₂, a₃, w) ∈ P_{1/2}
        # (a₁, w, g) ∈ P_{1/3}
        for i in 1:N
            @constraint(model, [a2_copy[i], a3_copy[i], w[i]] in MOI.PowerCone(0.5))
            @constraint(model, [a1_copy[i], w2[i], g[i]] in MOI.PowerCone(1/3))
        end

        # Maximize sum of geometric means
        @objective(model, Max, sum(g))

        optimize!(model)
        return objective_value(model), value.(a), value.(g)
    end

    #
    # Leg S: SheafSDP with POW cones
    #
    # Per agent:
    #   - a_i block (POS, dim m): allocation
    #   - slack_i block (POS, dim 1): budget slack
    #   - POW block 1 (dim 3): (a₂, a₃, w) ∈ P_{1/2}
    #   - POW block 2 (dim 3): (a₁, w, g) ∈ P_{1/3}
    #   - w block (NOC, dim 1): intermediate
    #   - g block (NOC, dim 1): epigraph
    #
    function solve_sheaf()
        blocks_per_agent = 6  # a, slack, pow1, pow2, w, g

        col_a(i) = (i - 1) * blocks_per_agent + 1
        col_slack(i) = (i - 1) * blocks_per_agent + 2
        col_pow1(i) = (i - 1) * blocks_per_agent + 3
        col_pow2(i) = (i - 1) * blocks_per_agent + 4
        col_w(i) = (i - 1) * blocks_per_agent + 5
        col_g(i) = (i - 1) * blocks_per_agent + 6

        # Rows per agent:
        #   - 1 budget row: Σ a_k + slack = b_i
        #   - 3 pow1 slot rows: x₁=a₂, x₂=a₃, x₃=w
        #   - 3 pow2 slot rows: x₁=a₁, x₂=w, x₃=g
        # Plus coordination rows
        rows_per_agent = 1 + 3 + 3

        row_budget(i) = (i - 1) * rows_per_agent + 1
        row_pow1_x1(i) = (i - 1) * rows_per_agent + 2
        row_pow1_x2(i) = (i - 1) * rows_per_agent + 3
        row_pow1_x3(i) = (i - 1) * rows_per_agent + 4
        row_pow2_x1(i) = (i - 1) * rows_per_agent + 5
        row_pow2_x2(i) = (i - 1) * rows_per_agent + 6
        row_pow2_x3(i) = (i - 1) * rows_per_agent + 7
        row_coord(e, k) = N * rows_per_agent + (e - 1) * m + k

        row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

        for i in 1:N
            # Budget constraint: sum(a) + slack = b_i
            push!(row_ids, row_budget(i))
            push!(col_ids, col_a(i))
            push!(blocks, ones(1, m))  # [1 1 1]
            push!(row_ids, row_budget(i))
            push!(col_ids, col_slack(i))
            push!(blocks, reshape([1.0], 1, 1))

            # POW1: (a₂, a₃, w) ∈ P_{1/2}
            # x₁ = a₂ (row picks a[2] from a block)
            push!(row_ids, row_pow1_x1(i))
            push!(col_ids, col_pow1(i))
            push!(blocks, reshape([1.0, 0.0, 0.0], 1, 3))  # pick x₁
            push!(row_ids, row_pow1_x1(i))
            push!(col_ids, col_a(i))
            sel = zeros(1, m); sel[1, 2] = -1.0  # -a₂
            push!(blocks, sel)

            # x₂ = a₃
            push!(row_ids, row_pow1_x2(i))
            push!(col_ids, col_pow1(i))
            push!(blocks, reshape([0.0, 1.0, 0.0], 1, 3))  # pick x₂
            push!(row_ids, row_pow1_x2(i))
            push!(col_ids, col_a(i))
            sel = zeros(1, m); sel[1, 3] = -1.0  # -a₃
            push!(blocks, sel)

            # x₃ = w
            push!(row_ids, row_pow1_x3(i))
            push!(col_ids, col_pow1(i))
            push!(blocks, reshape([0.0, 0.0, 1.0], 1, 3))  # pick x₃
            push!(row_ids, row_pow1_x3(i))
            push!(col_ids, col_w(i))
            push!(blocks, reshape([-1.0], 1, 1))

            # POW2: (a₁, w, g) ∈ P_{1/3}
            # x₁ = a₁
            push!(row_ids, row_pow2_x1(i))
            push!(col_ids, col_pow2(i))
            push!(blocks, reshape([1.0, 0.0, 0.0], 1, 3))
            push!(row_ids, row_pow2_x1(i))
            push!(col_ids, col_a(i))
            sel = zeros(1, m); sel[1, 1] = -1.0  # -a₁
            push!(blocks, sel)

            # x₂ = w
            push!(row_ids, row_pow2_x2(i))
            push!(col_ids, col_pow2(i))
            push!(blocks, reshape([0.0, 1.0, 0.0], 1, 3))
            push!(row_ids, row_pow2_x2(i))
            push!(col_ids, col_w(i))
            push!(blocks, reshape([-1.0], 1, 1))

            # x₃ = g
            push!(row_ids, row_pow2_x3(i))
            push!(col_ids, col_pow2(i))
            push!(blocks, reshape([0.0, 0.0, 1.0], 1, 3))
            push!(row_ids, row_pow2_x3(i))
            push!(col_ids, col_g(i))
            push!(blocks, reshape([-1.0], 1, 1))

            # Budget: sum(a) ≤ b_i
            # Since a is POS, we can add a slack. Actually let's use a budget block.
            # For simplicity, we'll just constrain via the consensus + minimum budget.
        end

        # Actually, let's add budget constraints properly
        # We need slack variables. Add them to each agent.

        # Simpler: skip explicit budget for now, consensus + positivity is enough
        # The problem is feasible as long as allocations are positive and agree.
        # The geometric mean will be bounded by the consensus constraint.

        # Coordination: a_i = a_j for all edges (component-wise)
        for (e, (i, j)) in enumerate(edges)
            for k in 1:m
                push!(row_ids, row_coord(e, k))
                push!(col_ids, col_a(i))
                sel = zeros(1, m); sel[1, k] = -1.0
                push!(blocks, sel)

                push!(row_ids, row_coord(e, k))
                push!(col_ids, col_a(j))
                sel = zeros(1, m); sel[1, k] = 1.0
                push!(blocks, sel)
            end
        end

        # Add budget constraints as additional rows
        # Σ a_k - s = b_i where s is a POS slack (absorbed into existing blocks? no)
        # Let's add explicit budget slacks

        # Actually, let's restructure to include budget properly
        # For now, add an upper bound on allocations via additional POS blocks

        B = blocksparse(row_ids, col_ids, blocks)

        # Objective: maximize Σ g = minimize -Σ g
        c = zeros(size(B, 2))
        for i in 1:N
            c_rng = colrange(B, col_g(i))
            c[c_rng[1]] = -1.0
        end

        # RHS: budget constraints (agent i has budget b_i), coupling rows are 0
        g = zeros(size(B, 1))
        for i in 1:N
            g[rowrange(B, row_budget(i))] .= budgets[i]
        end

        Q = SheafSDP.allocblockdiag(B)
        fill!(Q, 0)

        nv = N * blocks_per_agent
        cones = Vector{SheafSDP.Cone}(undef, nv)
        for i in 1:N
            cones[col_a(i)] = SheafSDP.PositiveCone()
            cones[col_slack(i)] = SheafSDP.PositiveCone()
            cones[col_pow1(i)] = SheafSDP.PowerCone(0.5)   # P_{1/2}
            cones[col_pow2(i)] = SheafSDP.PowerCone(1/3)   # P_{1/3}
            cones[col_w(i)] = SheafSDP.CofreeCone()
            cones[col_g(i)] = SheafSDP.CofreeCone()
        end

        prob = IPMProblem(c, g, B, Q, cones)
        settings = IPMSettings{Float64}(
            kkt=UzawaSettings{Float64}(raug=raug),
            feas_tol=1e-6, gap_tol=1e-6, itmax=200,
            verbose=false
        )
        result = solve(prob, settings)

        return -dot(c, result.p), result.iterations, result.kkt_iters, result.status
    end

    # Warmup
    solve_mosek()
    try solve_sheaf() catch end

    local obj_mosek, a_mosek, g_mosek
    t_mosek = @elapsed (obj_mosek, a_mosek, g_mosek) = solve_mosek()

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
        m = m,
        ne = ne,
        n_pow = 2 * N,  # 2 POW blocks per agent
        t_mosek = t_mosek,
        t_sheaf = t_sheaf,
        obj_mosek = obj_mosek,
        obj_sheaf = obj_sheaf,
        iters = iters,
        kkt_iters = kkt_iters,
        status = status,
        a_mosek = a_mosek,
        g_mosek = g_mosek,
    )
end

# Verify geometric mean tower correctness
function verify_geomean_tower()
    println("Verifying geometric mean tower construction...")

    # Test: (a₂, a₃, w) ∈ P_{1/2}, (a₁, w, g) ∈ P_{1/3}
    # Should give g = (a₁ a₂ a₃)^{1/3}

    a = [2.0, 3.0, 5.0]  # test values

    # Direct geometric mean
    gm_direct = (prod(a))^(1/3)

    # Via tower
    w = (a[2] * a[3])^0.5  # from P_{1/2}
    g_tower = (a[1]^(1/3)) * (w^(2/3))  # from P_{1/3}: x₁^α x₂^(1-α) with α=1/3

    println("  a = $a")
    println("  Direct (a₁a₂a₃)^{1/3} = $(round(gm_direct, digits=6))")
    println("  Tower result          = $(round(g_tower, digits=6))")
    println("  Match: $(abs(gm_direct - g_tower) < 1e-10 ? "PASS" : "FAIL")")
    println()
end

# Main
println("=" ^ 70)
println("Proportional-Fair Resource Split: SheafSDP vs Mosek")
println("=" ^ 70)
println()
println("Recipe C (§5): Maximize geometric mean with consensus")
println("Problem: N agents on path P_N, m=3 resources")
println("Objective: maximize Σ (∏_k a_{i,k})^{1/m}")
println("Constraints: consensus (all agents agree on allocation)")
println("Tower: P_{1/2} feeding P_{1/3} for 3-term geometric mean")
println()

verify_geomean_tower()

println("-" ^ 70)
println("Scaling sweep:")
println("-" ^ 70)
println("| N | #POW | Mosek | SheafSDP | IPM | KKT | Status | Obj diff |")
println("|---|------|-------|----------|-----|-----|--------|----------|")

results = []
for N in [3, 5, 10, 15, 20]
    res = run_fairsplit_benchmark(N)
    push!(results, res)
    mosek_ms = round(res.t_mosek * 1000, digits=1)
    sheaf_ms = round(res.t_sheaf * 1000, digits=1)
    diff = abs(res.obj_mosek - res.obj_sheaf)
    println("| $(res.N) | $(res.n_pow) | $(mosek_ms) ms | $(sheaf_ms) ms | $(res.iters) | $(res.kkt_iters) | $(res.status) | $(round(diff, sigdigits=3)) |")
end
println()

# Print allocation for one result
r = results[3]  # N=10 case
println("-" ^ 70)
println("Solution check (N=$(r.N)):")
println("-" ^ 70)
println("  Mosek allocation (agent 1): $(round.(r.a_mosek[1, :], digits=4))")
println("  Mosek geo-means: $(round.(r.g_mosek, digits=4))")
println("  Objective (sum of geo-means):")
println("    Mosek:    $(round(r.obj_mosek, digits=6))")
println("    SheafSDP: $(round(r.obj_sheaf, digits=6))")
println()

# Verify consensus
println("Consensus check (all allocations should match):")
all_match = all(r.a_mosek[i, :] ≈ r.a_mosek[1, :] for i in 2:r.N)
println("  All agents have same allocation: $(all_match ? "PASS" : "FAIL")")
println()

println("=" ^ 70)
