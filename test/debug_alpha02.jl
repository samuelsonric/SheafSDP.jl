using Printf
using SheafSDP
using CommonSolve: solve
using SparseArrays
using LinearAlgebra
using Random
using BlockSparseArrays: colrange, rowrange, ncols, blocksparse

function run_verbose_alpha02()
    Random.seed!(42)

    N, T, p = 10, 15, 5.0
    α = 1/p
    raug = 1e4
    ū = 100.0

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, i+1) for i in 1:N-1]

    num_pow_per_agent = nu * (T - 1)
    blocks_per_agent = T + num_pow_per_agent + 2 * (T - 1)

    col_x(i, t_idx) = (i - 1) * blocks_per_agent + t_idx
    col_pow(i, t_idx, k) = (i - 1) * blocks_per_agent + T + (t_idx - 1) * nu + k
    col_sp(i, t_idx) = (i - 1) * blocks_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 1
    col_sm(i, t_idx) = (i - 1) * blocks_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 2

    rows_per_agent = 1 + (T - 1) + num_pow_per_agent + 2 * (T - 1)

    row_init(i) = (i - 1) * rows_per_agent + 1
    row_dyn(i, t_idx) = (i - 1) * rows_per_agent + 1 + t_idx
    row_x2(i, t_idx, k) = (i - 1) * rows_per_agent + T + (t_idx - 1) * nu + k
    row_boxp(i, t_idx) = (i - 1) * rows_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 1
    row_boxm(i, t_idx) = (i - 1) * rows_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 2
    row_coord(e) = N * rows_per_agent + e

    row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]

    for i in 1:N
        push!(row_ids, row_init(i)); push!(col_ids, col_x(i, 1)); push!(blocks, Matrix(1.0I, nx, nx))

        for t_idx in 1:T-1
            push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_x(i, t_idx)); push!(blocks, -A_dyn)
            push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_x(i, t_idx + 1)); push!(blocks, Matrix(1.0I, nx, nx))

            for k in 1:nu
                B_col_k = B_dyn[:, k:k]
                push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_pow(i, t_idx, k)); push!(blocks, -B_col_k * [0.0 0.0 1.0])
            end

            for k in 1:nu
                push!(row_ids, row_x2(i, t_idx, k)); push!(col_ids, col_pow(i, t_idx, k)); push!(blocks, reshape([0.0, 1.0, 0.0], 1, 3))
            end

            for k in 1:nu
                blk = zeros(nu, 3); blk[k, 3] = 1.0
                push!(row_ids, row_boxp(i, t_idx)); push!(col_ids, col_pow(i, t_idx, k)); push!(blocks, blk)
            end
            push!(row_ids, row_boxp(i, t_idx)); push!(col_ids, col_sp(i, t_idx)); push!(blocks, Matrix(1.0I, nu, nu))

            for k in 1:nu
                blk = zeros(nu, 3); blk[k, 3] = -1.0
                push!(row_ids, row_boxm(i, t_idx)); push!(col_ids, col_pow(i, t_idx, k)); push!(blocks, blk)
            end
            push!(row_ids, row_boxm(i, t_idx)); push!(col_ids, col_sm(i, t_idx)); push!(blocks, Matrix(1.0I, nu, nu))
        end
    end

    for (e, (i, j)) in enumerate(edges)
        push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T)); push!(blocks, P_proj)
        push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T)); push!(blocks, -P_proj)
    end

    B = blocksparse(row_ids, col_ids, blocks)

    c = zeros(ncols(B))
    for i in 1:N, t_idx in 1:T-1, k in 1:nu
        c[colrange(B, col_pow(i, t_idx, k))[1]] = 1.0
    end

    g = zeros(size(B, 1))
    for i in 1:N
        g[rowrange(B, row_init(i))] .= x0[i]
        for t_idx in 1:T-1
            for k in 1:nu; g[rowrange(B, row_x2(i, t_idx, k))] .= 1.0; end
            g[rowrange(B, row_boxp(i, t_idx))] .= ū
            g[rowrange(B, row_boxm(i, t_idx))] .= ū
        end
    end

    Q = SheafSDP.allocblockdiag(B); fill!(Q, 0)

    nv = N * blocks_per_agent
    cones = Vector{SheafSDP.Cone}(undef, nv)
    for i in 1:N
        for t_idx in 1:T; cones[col_x(i, t_idx)] = SheafSDP.CofreeCone(); end
        for t_idx in 1:T-1
            for k in 1:nu; cones[col_pow(i, t_idx, k)] = SheafSDP.PowerCone(α); end
            cones[col_sp(i, t_idx)] = SheafSDP.PositiveCone()
            cones[col_sm(i, t_idx)] = SheafSDP.PositiveCone()
        end
    end

    prob = SheafSDP.IPMProblem(c, g, B, Q, cones)
    settings = SheafSDP.IPMSettings{Float64}(
        kkt=SheafSDP.UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-6, gap_tol=1e-6, itmax=200, verbose=true
    )
    result = solve(prob, settings)

    println("\nLast 10 iterations:")
    println("iter | τp       | τd       | μ        | rp       | rd")
    for i in max(1, length(result.history.τp)-9):length(result.history.τp)
        @printf("%4d | %.2e | %.2e | %.2e | %.2e | %.2e\n",
            i, result.history.τp[i], result.history.τd[i],
            result.history.μ[i], result.history.rp[i], result.history.rd[i])
    end

    return result
end

# Also test with some diagnostics on the scaling
function test_scaling_at_failure()
    # Reproduce a point close to failure
    α = 0.2

    # Near-optimal solution state (approximated from iteration 28)
    # x = (t, 1, u) for power cone epigraph
    # At convergence, t ≈ |u|^5, so for small u, t is very small

    # Let's test with a point that has small t (close to boundary)
    for t_val in [1.0, 0.1, 0.01, 0.001, 0.0001]
        u = 0.5
        x = [t_val, 1.0, u]  # (t, 1, u) in power cone

        # Check if in cone
        φ = SheafSDP.powphi(x, α)
        if φ ≤ 0
            @printf("t=%.4f: NOT IN CONE (φ=%.2e)\n", t_val, φ)
            continue
        end

        # Check Hessian condition
        H = zeros(3, 3)
        SheafSDP.powhess!(H, x, α)
        κ_H = cond(H)

        # Check if structured Cholesky works
        L = zeros(3, 3)
        chol_ok = false
        try
            SheafSDP.powchol3!(L, H, x, α)
            chol_ok = all(isfinite, L) && all(L[i,i] > 0 for i in 1:3)
        catch
        end

        @printf("t=%.4f: φ=%.2e, κ(H)=%.2e, chol=%s\n", t_val, φ, κ_H, chol_ok ? "OK" : "FAIL")
    end
end

println("\n" * "="^60)
println("Scaling diagnostics:")
println("="^60)
test_scaling_at_failure()

# Run and analyze the minimum φ across POW cones at the final iterate
function run_and_analyze()
    Random.seed!(42)

    N, T, p = 10, 15, 5.0
    α = 1/p
    raug = 1e4
    ū = 100.0

    nx = 4; nu = 2; h = 0.1

    A_dyn = [1.0 0.0 h 0.0; 0.0 1.0 0.0 h; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    B_dyn = [0.0 0.0; 0.0 0.0; h 0.0; 0.0 h]
    P_proj = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]

    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, i+1) for i in 1:N-1]

    num_pow_per_agent = nu * (T - 1)
    blocks_per_agent = T + num_pow_per_agent + 2 * (T - 1)

    col_pow(i, t_idx, k) = (i - 1) * blocks_per_agent + T + (t_idx - 1) * nu + k

    # Build problem (same as before, abbreviated)
    col_x(i, t_idx) = (i - 1) * blocks_per_agent + t_idx
    col_sp(i, t_idx) = (i - 1) * blocks_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 1
    col_sm(i, t_idx) = (i - 1) * blocks_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 2
    rows_per_agent = 1 + (T - 1) + num_pow_per_agent + 2 * (T - 1)
    row_init(i) = (i - 1) * rows_per_agent + 1
    row_dyn(i, t_idx) = (i - 1) * rows_per_agent + 1 + t_idx
    row_x2(i, t_idx, k) = (i - 1) * rows_per_agent + T + (t_idx - 1) * nu + k
    row_boxp(i, t_idx) = (i - 1) * rows_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 1
    row_boxm(i, t_idx) = (i - 1) * rows_per_agent + T + num_pow_per_agent + 2 * (t_idx - 1) + 2
    row_coord(e) = N * rows_per_agent + e

    row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]
    for i in 1:N
        push!(row_ids, row_init(i)); push!(col_ids, col_x(i, 1)); push!(blocks, Matrix(1.0I, nx, nx))
        for t_idx in 1:T-1
            push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_x(i, t_idx)); push!(blocks, -A_dyn)
            push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_x(i, t_idx + 1)); push!(blocks, Matrix(1.0I, nx, nx))
            for k in 1:nu
                B_col_k = B_dyn[:, k:k]
                push!(row_ids, row_dyn(i, t_idx)); push!(col_ids, col_pow(i, t_idx, k)); push!(blocks, -B_col_k * [0.0 0.0 1.0])
            end
            for k in 1:nu
                push!(row_ids, row_x2(i, t_idx, k)); push!(col_ids, col_pow(i, t_idx, k)); push!(blocks, reshape([0.0, 1.0, 0.0], 1, 3))
            end
            for k in 1:nu
                blk = zeros(nu, 3); blk[k, 3] = 1.0
                push!(row_ids, row_boxp(i, t_idx)); push!(col_ids, col_pow(i, t_idx, k)); push!(blocks, blk)
            end
            push!(row_ids, row_boxp(i, t_idx)); push!(col_ids, col_sp(i, t_idx)); push!(blocks, Matrix(1.0I, nu, nu))
            for k in 1:nu
                blk = zeros(nu, 3); blk[k, 3] = -1.0
                push!(row_ids, row_boxm(i, t_idx)); push!(col_ids, col_pow(i, t_idx, k)); push!(blocks, blk)
            end
            push!(row_ids, row_boxm(i, t_idx)); push!(col_ids, col_sm(i, t_idx)); push!(blocks, Matrix(1.0I, nu, nu))
        end
    end
    for (e, (i, j)) in enumerate(edges)
        push!(row_ids, row_coord(e)); push!(col_ids, col_x(i, T)); push!(blocks, P_proj)
        push!(row_ids, row_coord(e)); push!(col_ids, col_x(j, T)); push!(blocks, -P_proj)
    end

    B = blocksparse(row_ids, col_ids, blocks)
    c = zeros(ncols(B))
    for i in 1:N, t_idx in 1:T-1, k in 1:nu
        c[colrange(B, col_pow(i, t_idx, k))[1]] = 1.0
    end
    g = zeros(size(B, 1))
    for i in 1:N
        g[rowrange(B, row_init(i))] .= x0[i]
        for t_idx in 1:T-1
            for k in 1:nu; g[rowrange(B, row_x2(i, t_idx, k))] .= 1.0; end
            g[rowrange(B, row_boxp(i, t_idx))] .= ū
            g[rowrange(B, row_boxm(i, t_idx))] .= ū
        end
    end
    Q = SheafSDP.allocblockdiag(B); fill!(Q, 0)

    nv = N * blocks_per_agent
    cones = Vector{SheafSDP.Cone}(undef, nv)
    for i in 1:N
        for t_idx in 1:T; cones[col_x(i, t_idx)] = SheafSDP.CofreeCone(); end
        for t_idx in 1:T-1
            for k in 1:nu; cones[col_pow(i, t_idx, k)] = SheafSDP.PowerCone(α); end
            cones[col_sp(i, t_idx)] = SheafSDP.PositiveCone()
            cones[col_sm(i, t_idx)] = SheafSDP.PositiveCone()
        end
    end

    prob = SheafSDP.IPMProblem(c, g, B, Q, cones)
    settings = SheafSDP.IPMSettings{Float64}(
        kkt=SheafSDP.UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-6, gap_tol=1e-6, itmax=200, verbose=true
    )
    result = solve(prob, settings)

    # Analyze POW cone states in final primal
    println("\n" * "="^60)
    println("POW cone analysis at final iterate:")
    println("="^60)

    p_final = result.p
    min_φ = Inf
    worst_cone = nothing

    for i in 1:N, t_idx in 1:T-1, k in 1:nu
        col = col_pow(i, t_idx, k)
        rng = colrange(B, col)
        x_cone = p_final[rng]
        φ = SheafSDP.powphi(x_cone, α)
        if φ < min_φ
            min_φ = φ
            worst_cone = (i, t_idx, k, x_cone, φ)
        end
    end

    println("Minimum φ across all POW cones: ", min_φ)
    if worst_cone !== nothing
        i, t_idx, k, x_cone, φ = worst_cone
        println("Worst cone: agent=$i, t=$t_idx, k=$k")
        println("  x = ", x_cone)
        println("  φ = ", φ)

        # Run digit-loss report on worst cone
        include("pow_digitloss.jl")
        println("\nDigit-loss report for worst cone:")
        PowDigitLoss.report(x_cone, α)
    end

    return result
end

result = run_and_analyze()
println("\nFinal status: ", result.status)
