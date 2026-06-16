using Test
using LinearAlgebra
using SparseArrays
using Random
using SheafSDP
using SheafSDP: trinum, triroot, svec!, smat!, symmetrize!, hess!, newton_step!, residuals!, conedegree, mu
using BlockSparseArrays: blocksparse, ncols, vtxs, colrange, nvtxs
using CliqueTrees.Multifrontal: FactorizationWorkspace, DivisionWorkspace

@testset "Phase 3: solve_kkt! mapping and sign conventions" begin

    @testset "hess! assembles block-diagonal H" begin
        # Create a simple block structure: 2 vertices with dims 2, 2
        dims = [2, 2]
        nv = length(dims)
        n = sum(trinum.(dims))  # svec length = 3 + 3 = 6

        # Build a minimal B (one edge connecting two vertices)
        edge_dim = 2
        ne = 1
        rows = [1, 1]
        cols = [1, 2]
        V = [randn(edge_dim, trinum(2)), randn(edge_dim, trinum(2))]
        B = blocksparse(rows, cols, V, ne, nv)

        # Random SPD P, D
        p = zeros(n)
        d = zeros(n)
        offset = 0
        for dim in dims
            k = trinum(dim)
            A = randn(dim, dim)
            P_v = A * A' + I
            A = randn(dim, dim)
            D_v = A * A' + I
            svec!(view(p, offset+1:offset+k), P_v, Val(:U))
            svec!(view(d, offset+1:offset+k), D_v, Val(:U))
            offset += k
        end

        # Assemble H
        H_blocks = Matrix{Float64}[]
        W_blocks = Matrix{Float64}[]
        H = hess!(H_blocks, W_blocks, p, d, B, Val(:U))

        # H should have the right structure
        @test nvtxs(H) == nv

        # Each diagonal block should be SPD
        for v in 1:nv
            H_v = H_blocks[v]
            @test isposdef(Symmetric(H_v))
        end

        # Load-bearing identity: H p = d (blockwise)
        H_sp = sparse(H)
        @test H_sp * p ≈ d atol=1e-10
    end

    @testset "newton_step! solves reduced Newton system" begin
        Random.seed!(42)

        # Build a simple sheaf: 3 vertices, 2 edges
        nv = 3
        dv = 2  # matrix dimension per vertex (not svec)
        de = 2  # edge stalk dimension
        edges = [(1, 2), (2, 3)]
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

        # Build sheaf
        P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
        B_sp = sparse(B)

        # Dimensions
        n = size(F, 1)  # svec total
        m = size(B, 1)  # edge total

        # Build random SPD P, D in the permuted order
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

        # Assemble H
        H_blocks = Matrix{Float64}[]
        W_blocks = Matrix{Float64}[]
        H = hess!(H_blocks, W_blocks, p, d, B, Val(:U))
        H_sp = sparse(H)

        # Choose c, g, y for a consistent (feasible) problem
        y = randn(m)
        c = B_sp' * y + d
        g = B_sp * p

        # Compute residuals (should be zero)
        r_p = zeros(m)
        r_d = zeros(n)
        residuals!(r_p, r_d, B_sp, p, d, y, c, g)
        @test norm(r_p) < 1e-12
        @test norm(r_d) < 1e-12

        # r_c for affine step
        r_c = -p

        # Workspaces
        facwrk = FactorizationWorkspace(F)
        divwrk = DivisionWorkspace(F, 1)
        itrwrk = RiWorkspace(m, Vector{Float64})
        r = zeros(m)

        # Direction vectors
        Δp = zeros(n)
        Δy = zeros(m)
        Δd = zeros(n)

        # Solve Newton system
        newton_step!(Δp, Δy, Δd, facwrk, divwrk, itrwrk, r, F, L, B, B_sp, H, H_sp,
                     r_c, r_p, r_d; τ=1.0, atol=1e-12, rtol=1e-12)

        # Verify original (unaugmented) KKT system residuals:
        # 1. B Δp = r_p
        res1 = B_sp * Δp - r_p
        @test norm(res1) < 1e-8

        # 2. H Δp - Bᵀ Δy = H r_c - r_d (the (1,1) block equation)
        f = H_sp * r_c - r_d
        res2 = H_sp * Δp - B_sp' * Δy - f
        @test norm(res2) < 1e-8

        # 3. Bᵀ Δy + Δd = r_d (dual feasibility recovery)
        res3 = B_sp' * Δy + Δd - r_d
        @test norm(res3) < 1e-10

        # For affine step from feasible point with r_c = -p:
        # f = H(-p) - r_d = -d - 0 = -d
        @test f ≈ -d atol=1e-10
    end

    @testset "newton_step! with infeasible start" begin
        Random.seed!(123)

        # Build sheaf
        nv = 4
        dv = 3
        de = 2
        edges = [(1, 2), (2, 3), (3, 4), (1, 4)]
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

        # Build random SPD P, D
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

        # Assemble H
        H_blocks = Matrix{Float64}[]
        W_blocks = Matrix{Float64}[]
        H = hess!(H_blocks, W_blocks, p, d, B, Val(:U))
        H_sp = sparse(H)

        # Choose c, g, y with NON-zero residuals (infeasible start)
        y = randn(m)
        c = randn(n)
        g = randn(m)

        # Compute residuals
        r_p = zeros(m)
        r_d = zeros(n)
        residuals!(r_p, r_d, B_sp, p, d, y, c, g)

        # r_c for affine step
        r_c = -p

        # Workspaces
        facwrk = FactorizationWorkspace(F)
        divwrk = DivisionWorkspace(F, 1)
        itrwrk = RiWorkspace(m, Vector{Float64})
        r = zeros(m)

        # Direction vectors
        Δp = zeros(n)
        Δy = zeros(m)
        Δd = zeros(n)

        # Solve Newton system
        newton_step!(Δp, Δy, Δd, facwrk, divwrk, itrwrk, r, F, L, B, B_sp, H, H_sp,
                     r_c, r_p, r_d; τ=1.0, atol=1e-12, rtol=1e-12)

        # Verify KKT residuals
        f = H_sp * r_c - r_d
        res1 = B_sp * Δp - r_p
        res2 = H_sp * Δp - B_sp' * Δy - f
        res3 = B_sp' * Δy + Δd - r_d

        @test norm(res1) < 1e-8
        @test norm(res2) < 1e-8
        @test norm(res3) < 1e-10
    end

end
