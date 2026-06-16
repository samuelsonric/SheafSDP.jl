using Test
using LinearAlgebra
using SparseArrays
using Random
using SheafSDP
using SheafSDP: trinum, triroot, svec!, smat!, symmetrize!
using SheafSDP: initialize!, solve!, SolverResult, is_stalled, is_numerical_failure
using SheafSDP: residuals!, conedegree, mu
using BlockSparseArrays: blocksparse, ncols, vtxs, colrange, nvtxs

@testset "Phase 8: initialization, termination, robustness" begin

    @testset "initialize! creates SPD blocks" begin
        Random.seed!(42)

        # Build sheaf
        nv = 3
        dv = 2
        de = 2
        edges = [(1, 2), (2, 3)]
        ne = length(edges)

        src = Int[]
        dst = Int[]
        maps = Matrix{Float64}[]
        for (e_idx, (u, v)) in enumerate(edges)
            push!(src, u)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
            push!(src, v)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
        end

        P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
        B_sp = sparse(B)

        n = size(F, 1)
        m = size(B, 1)

        # Create problem data
        c = randn(n)
        g = randn(m)

        # Initialize
        p = zeros(n)
        d = zeros(n)
        y = zeros(m)
        initialize!(p, d, y, c, g, B, Val(:U))

        # Check y = 0
        @test all(y .== 0)

        # Check p, d represent SPD blocks
        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            P_v = zeros(d_v, d_v)
            D_v = zeros(d_v, d_v)
            smat!(P_v, view(p, r), Val(:U))
            symmetrize!(P_v, Val(:U))
            smat!(D_v, view(d, r), Val(:U))
            symmetrize!(D_v, Val(:U))

            @test isposdef(Symmetric(P_v))
            @test isposdef(Symmetric(D_v))

            # P_v = D_v = ξ I where ξ = max(1, ||c||, ||g||)
            ξ = max(1.0, norm(c), norm(g))
            @test P_v ≈ ξ * I atol=1e-10
            @test D_v ≈ ξ * I atol=1e-10
        end
    end

    @testset "initialize! with custom ξ" begin
        Random.seed!(123)

        nv = 2
        dv = 2
        de = 2
        edges = [(1, 2)]
        ne = 1

        src = Int[]
        dst = Int[]
        maps = Matrix{Float64}[]
        for (e_idx, (u, v)) in enumerate(edges)
            push!(src, u)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
            push!(src, v)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
        end

        P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
        B_sp = sparse(B)

        n = size(F, 1)
        m = size(B, 1)

        c = randn(n)
        g = randn(m)
        p = zeros(n)
        d = zeros(n)
        y = zeros(m)

        # Custom ξ
        initialize!(p, d, y, c, g, B, Val(:U); ξ=5.0)

        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            P_v = zeros(d_v, d_v)
            smat!(P_v, view(p, r), Val(:U))
            symmetrize!(P_v, Val(:U))

            @test P_v ≈ 5.0 * I atol=1e-10
        end
    end

    @testset "is_stalled detection" begin
        # Not enough history
        μ_history = [1.0, 0.9, 0.8]
        @test !is_stalled(μ_history; window=5)

        # Clear decrease - not stalled
        μ_history = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03, 0.015]
        @test !is_stalled(μ_history; window=5, threshold=0.99)

        # Stalled - μ barely decreasing (less than 1% over window)
        # μ_old = μ_history[2] = 1.0, μ_new = μ_history[7] = 0.995
        # 0.995 > 0.99 * 1.0 = 0.995 > 0.99 ✓ (stalled)
        μ_history = [1.0, 1.0, 0.999, 0.998, 0.997, 0.996, 0.995]
        @test is_stalled(μ_history; window=5, threshold=0.99)
    end

    @testset "is_numerical_failure detection" begin
        # Not enough history
        τ_p = [0.5, 0.6]
        τ_d = [0.5, 0.6]
        rp = [0.1, 0.09]
        rd = [0.1, 0.09]
        @test !is_numerical_failure(τ_p, τ_d, rp, rd; window=3)

        # Normal operation - not failure
        τ_p = [0.9, 0.85, 0.8, 0.75]
        τ_d = [0.9, 0.85, 0.8, 0.75]
        rp = [0.1, 0.08, 0.06, 0.04]
        rd = [0.1, 0.08, 0.06, 0.04]
        @test !is_numerical_failure(τ_p, τ_d, rp, rd; window=3)

        # τ collapse with residual plateau
        τ_p = [0.9, 0.5, 1e-8, 1e-9, 1e-9]
        τ_d = [0.9, 0.5, 1e-8, 1e-9, 1e-9]
        rp = [0.1, 0.09, 0.095, 0.094, 0.093]
        rd = [0.1, 0.09, 0.095, 0.094, 0.093]
        @test is_numerical_failure(τ_p, τ_d, rp, rd; window=3, τ_threshold=1e-6)
    end

    @testset "SolverResult struct" begin
        result = SolverResult{Float64}(
            [1.0, 2.0], [1.0, 2.0], [0.5],
            true, 10,
            [1.0, 0.5, 0.1],
            [0.9, 0.85],
            [0.9, 0.85],
            [0.1, 0.05],
            [0.1, 0.05],
            :optimal
        )

        @test result.converged == true
        @test result.iterations == 10
        @test result.status == :optimal
        @test length(result.μ_history) == 3
        @test length(result.τ_p_history) == 2
    end

    @testset "solve! from initialized point" begin
        Random.seed!(456)

        # Build sheaf
        nv = 2
        dv = 2
        de = 2
        edges = [(1, 2)]
        ne = 1

        src = Int[]
        dst = Int[]
        maps = Matrix{Float64}[]
        for (e_idx, (u, v)) in enumerate(edges)
            push!(src, u)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
            push!(src, v)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
        end

        P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
        B_sp = sparse(B)

        n = size(F, 1)
        m = size(B, 1)

        # Build a feasible problem
        p_feas = zeros(n)
        d_feas = zeros(n)
        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            A = randn(d_v, d_v)
            P_v = A * A' + I
            A = randn(d_v, d_v)
            D_v = A * A' + I

            svec!(view(p_feas, r), P_v, Val(:U))
            svec!(view(d_feas, r), D_v, Val(:U))
        end

        y_feas = randn(m)
        c = B_sp' * y_feas + d_feas
        g = B_sp * p_feas

        # Initialize
        p = zeros(n)
        d = zeros(n)
        y = zeros(m)
        initialize!(p, d, y, c, g, B, Val(:U))

        # Solve with Mehrotra
        result = solve!(p, d, y, c, g, B, B_sp, F, L;
                        ε_feas=1e-6, ε_μ=1e-6,
                        max_iter=50, verbose=false)

        @test result.converged
        @test result.status == :optimal
        @test result.μ_history[end] < 1e-5

        # Check solution feasibility
        r_p = zeros(m)
        r_d = zeros(n)
        residuals!(r_p, r_d, B_sp, result.p, result.d, result.y, c, g)
        @test norm(r_p) / (1 + norm(g)) < 1e-5
        @test norm(r_d) / (1 + norm(c)) < 1e-5
    end

    @testset "solve! returns diagnostics" begin
        Random.seed!(999)

        nv = 2
        dv = 2
        de = 2
        edges = [(1, 2)]
        ne = 1

        src = Int[]
        dst = Int[]
        maps = Matrix{Float64}[]
        for (e_idx, (u, v)) in enumerate(edges)
            push!(src, u)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
            push!(src, v)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
        end

        P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
        B_sp = sparse(B)

        n = size(F, 1)
        m = size(B, 1)

        # Build feasible problem
        p_feas = zeros(n)
        d_feas = zeros(n)
        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            A = randn(d_v, d_v)
            P_v = A * A' + I
            A = randn(d_v, d_v)
            D_v = A * A' + I

            svec!(view(p_feas, r), P_v, Val(:U))
            svec!(view(d_feas, r), D_v, Val(:U))
        end

        y_feas = randn(m)
        c = B_sp' * y_feas + d_feas
        g = B_sp * p_feas

        p = zeros(n)
        d = zeros(n)
        y = zeros(m)
        initialize!(p, d, y, c, g, B, Val(:U))

        result = solve!(p, d, y, c, g, B, B_sp, F, L;
                        ε_feas=1e-8, ε_μ=1e-8,
                        max_iter=50, verbose=false)

        # Check that diagnostics are populated
        @test length(result.μ_history) >= 1
        @test length(result.τ_p_history) >= 1
        @test length(result.τ_d_history) >= 1
        @test length(result.rp_history) >= 1
        @test length(result.rd_history) >= 1

        # μ should decrease
        @test result.μ_history[end] < result.μ_history[1]

        # τ should be reasonable (not collapsed)
        @test all(result.τ_p_history .> 0)
        @test all(result.τ_d_history .> 0)
    end

    @testset "solve! larger problem from initialization" begin
        Random.seed!(1234)

        # Build a larger sheaf
        nv = 5
        dv = 3
        de = 2
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        ne = length(edges)

        src = Int[]
        dst = Int[]
        maps = Matrix{Float64}[]
        for (e_idx, (u, v)) in enumerate(edges)
            push!(src, u)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
            push!(src, v)
            push!(dst, e_idx)
            push!(maps, randn(de, dv))
        end

        P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
        B_sp = sparse(B)

        n = size(F, 1)
        m = size(B, 1)

        # Build feasible problem
        p_feas = zeros(n)
        d_feas = zeros(n)
        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            A = randn(d_v, d_v)
            P_v = A * A' + I
            A = randn(d_v, d_v)
            D_v = A * A' + I

            svec!(view(p_feas, r), P_v, Val(:U))
            svec!(view(d_feas, r), D_v, Val(:U))
        end

        y_feas = randn(m)
        c = B_sp' * y_feas + d_feas
        g = B_sp * p_feas

        # Initialize and solve
        p = zeros(n)
        d = zeros(n)
        y = zeros(m)
        initialize!(p, d, y, c, g, B, Val(:U))

        result = solve!(p, d, y, c, g, B, B_sp, F, L;
                        ε_feas=1e-7, ε_μ=1e-7,
                        max_iter=100, verbose=false)

        @test result.converged
        @test result.status == :optimal

        # Check feasibility
        r_p = zeros(m)
        r_d = zeros(n)
        residuals!(r_p, r_d, B_sp, result.p, result.d, result.y, c, g)
        @test norm(r_p) / (1 + norm(g)) < 1e-6
        @test norm(r_d) / (1 + norm(c)) < 1e-6
    end

end
