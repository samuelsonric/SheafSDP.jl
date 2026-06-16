using Test
using LinearAlgebra
using SparseArrays
using Random
using SheafSDP
using SheafSDP: trinum, triroot, svec!, smat!, symmetrize!, hess!, newton_step!
using SheafSDP: residuals!, conedegree, mu, affine_rhs!, corrector_rhs!
using BlockSparseArrays: blocksparse, ncols, vtxs, colrange, nvtxs
using CliqueTrees.Multifrontal: FactorizationWorkspace, DivisionWorkspace

@testset "Phase 4: predictor/corrector RHS" begin

    @testset "affine_rhs! produces r_c = -p" begin
        n = 10
        p = randn(n)
        r_c = zeros(n)

        affine_rhs!(r_c, p)

        @test r_c ≈ -p
    end

    @testset "affine step gives f = -d - r_d" begin
        Random.seed!(456)

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

        # Assemble H
        H_blocks = Matrix{Float64}[]
        W_blocks = Matrix{Float64}[]
        H = hess!(H_blocks, W_blocks, p, d, B, Val(:U))
        H_sp = sparse(H)

        # Arbitrary residuals for test
        y = randn(m)
        c = randn(n)
        g = randn(m)
        r_p = zeros(m)
        r_d = zeros(n)
        residuals!(r_p, r_d, B_sp, p, d, y, c, g)

        # Affine RHS
        r_c = zeros(n)
        affine_rhs!(r_c, p)

        # f = H r_c - r_d = H(-p) - r_d = -d - r_d
        f = H_sp * r_c - r_d
        @test f ≈ -d - r_d atol=1e-10
    end

    @testset "corrector_rhs! includes 2nd-order term" begin
        Random.seed!(789)

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

        # Assemble H to get W_blocks
        H_blocks = Matrix{Float64}[]
        W_blocks = Matrix{Float64}[]
        H = hess!(H_blocks, W_blocks, p, d, B, Val(:U))

        # Create some direction vectors
        Δp = 0.1 * randn(n)
        Δd = 0.1 * randn(n)

        σμ = 0.5
        r_c = zeros(n)
        corrector_rhs!(r_c, p, d, Δp, Δd, W_blocks, σμ, B, Val(:U))

        # Verify by computing manually for each block
        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            P_v = zeros(d_v, d_v)
            D_v = zeros(d_v, d_v)
            ΔP_v = zeros(d_v, d_v)
            ΔD_v = zeros(d_v, d_v)

            smat!(P_v, view(p, r), Val(:U))
            symmetrize!(P_v, Val(:U))
            smat!(D_v, view(d, r), Val(:U))
            symmetrize!(D_v, Val(:U))
            smat!(ΔP_v, view(Δp, r), Val(:U))
            symmetrize!(ΔP_v, Val(:U))
            smat!(ΔD_v, view(Δd, r), Val(:U))
            symmetrize!(ΔD_v, Val(:U))

            W_v = W_blocks[v]
            D_inv = inv(cholesky(Symmetric(D_v)))

            # 2nd-order term
            cross = ΔP_v * ΔD_v * W_v
            cross_sym = (cross + cross') / 2

            R_c_expected = σμ * D_inv - P_v - cross_sym

            r_c_expected = zeros(n_v)
            svec!(r_c_expected, R_c_expected, Val(:U))

            @test view(r_c, r) ≈ r_c_expected atol=1e-10
        end
    end

end
