using Test
using LinearAlgebra
using SparseArrays
using Random
using SheafSDP
using SheafSDP: trinum, triroot, svec!, smat!, symmetrize!
using SheafSDP: hess!, affine_rhs!, newton_step!, residuals!
using SheafSDP: step_to_boundary, conedegree, mu
using BlockSparseArrays: blocksparse, ncols, vtxs, colrange, nvtxs
using CliqueTrees.Multifrontal: FactorizationWorkspace, DivisionWorkspace

@testset "Phase 6: solve! convergence" begin

    @testset "small 2x2 problem" begin
        Random.seed!(42)

        # Simple problem: 2 vertices with 2x2 blocks, 1 edge
        nv = 2
        dv = 2
        de = 2
        edges = [(1, 2)]
        ne = length(edges)

        # Build restriction maps
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

        P_perm, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
        B_sp = sparse(B)

        n = size(F, 1)
        m = size(B, 1)

        # Build SPD P, D
        p = zeros(n)
        d = zeros(n)
        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            A = randn(d_v, d_v)
            P_v = A * A' + I
            A = randn(d_v, d_v)
            D_v = A * A' + I

            svec!(view(p, r), P_v, Val(:U))
            svec!(view(d, r), D_v, Val(:U))
        end

        # Feasible point
        y = randn(m)
        c = B_sp' * y + d
        g = B_sp * p

        # Solve
        result = solve!(p, d, y, c, g, B, B_sp, F, L;
                        γ=0.99, ε_feas=1e-6, ε_μ=1e-6,
                        max_iter=50, verbose=false)

        @test result.converged
        @test result.μ_history[end] < 1e-5

        # Check feasibility
        @test norm(B_sp * result.p - g) < 1e-5
    end

    @testset "convergence rate check" begin
        Random.seed!(456)

        # Build a small sheaf
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
        ν = conedegree(B)

        # Build SPD P, D
        p = zeros(n)
        d = zeros(n)
        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            A = randn(d_v, d_v)
            P_v = A * A' + I
            A = randn(d_v, d_v)
            D_v = A * A' + I

            svec!(view(p, r), P_v, Val(:U))
            svec!(view(d, r), D_v, Val(:U))
        end

        # Make a feasible point
        y = randn(m)
        c = B_sp' * y + d
        g = B_sp * p

        # Solve
        result = solve!(p, d, y, c, g, B, B_sp, F, L;
                        γ=0.99, ε_feas=1e-8, ε_μ=1e-8,
                        max_iter=100, verbose=false)

        @test result.converged

        # μ should decrease roughly geometrically
        μ_hist = result.μ_history
        @test length(μ_hist) > 3

        # Check that μ decreases monotonically (mostly)
        decreasing = sum(μ_hist[2:end] .< μ_hist[1:end-1])
        @test decreasing >= length(μ_hist) - 2  # allow a few non-monotone steps

        # Final μ should be small
        @test μ_hist[end] < 1e-7
    end

    @testset "duality gap closes" begin
        Random.seed!(789)

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

        # Build SPD P, D
        p = zeros(n)
        d = zeros(n)
        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            A = randn(d_v, d_v)
            P_v = A * A' + I
            A = randn(d_v, d_v)
            D_v = A * A' + I

            svec!(view(p, r), P_v, Val(:U))
            svec!(view(d, r), D_v, Val(:U))
        end

        # Feasible point
        y = randn(m)
        c = B_sp' * y + d + 0.1 * randn(n)  # not exactly feasible in dual
        g = B_sp * p

        # Solve
        result = solve!(p, d, y, c, g, B, B_sp, F, L;
                        γ=0.99, ε_feas=1e-8, ε_μ=1e-8,
                        max_iter=100, verbose=false)

        @test result.converged

        # Primal objective: c'p
        primal_obj = dot(c, result.p)

        # Dual objective: g'y
        dual_obj = dot(g, result.y)

        # Duality gap should be small (equal to ν * μ at optimum)
        gap = primal_obj - dual_obj
        @test abs(gap) < 1e-6

        # At optimum: primal_obj ≈ dual_obj
        @test primal_obj ≈ dual_obj atol=1e-6
    end

    @testset "larger problem" begin
        Random.seed!(999)

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

        # Build SPD P, D
        p = zeros(n)
        d = zeros(n)
        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            A = randn(d_v, d_v)
            P_v = A * A' + I
            A = randn(d_v, d_v)
            D_v = A * A' + I

            svec!(view(p, r), P_v, Val(:U))
            svec!(view(d, r), D_v, Val(:U))
        end

        # Feasible point
        y = randn(m)
        c = B_sp' * y + d
        g = B_sp * p

        # Solve
        result = solve!(p, d, y, c, g, B, B_sp, F, L;
                        γ=0.99, ε_feas=1e-7, ε_μ=1e-7,
                        max_iter=100, verbose=false)

        @test result.converged

        # Check feasibility
        r_p = zeros(m)
        r_d = zeros(n)
        residuals!(r_p, r_d, B_sp, result.p, result.d, result.y, c, g)
        @test norm(r_p) / (1 + norm(g)) < 1e-6
        @test norm(r_d) / (1 + norm(c)) < 1e-6
    end

end
