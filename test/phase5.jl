using Test
using LinearAlgebra
using SparseArrays
using Random
using SheafSDP
using SheafSDP: trinum, triroot, svec!, smat!, symmetrize!
using SheafSDP: step_length_block, step_to_boundary
using BlockSparseArrays: blocksparse, ncols, vtxs, colrange, nvtxs

@testset "Phase 5: step-to-boundary" begin

    @testset "step_length_block with identity" begin
        # X = I, ΔX = -0.5 I
        # M = I * (-0.5 I) * I = -0.5 I
        # λ_min = -0.5
        # τ = -0.99 / (-0.5) = 1.98, clamped to 1.0
        d = 3
        X = Matrix{Float64}(I, d, d)
        ΔX = -0.5 * Matrix{Float64}(I, d, d)

        τ = step_length_block(X, ΔX, 0.99)
        @test τ ≈ 1.0  # clamped

        # With larger negative: ΔX = -2 I
        # λ_min = -2
        # τ = -0.99 / (-2) = 0.495
        ΔX = -2.0 * Matrix{Float64}(I, d, d)
        τ = step_length_block(X, ΔX, 0.99)
        @test τ ≈ 0.495 atol=1e-10
    end

    @testset "step_length_block preserves positive definiteness" begin
        Random.seed!(42)

        for d in [2, 3, 4]
            # Random SPD X
            A = randn(d, d)
            X = A * A' + I

            # Random symmetric ΔX (can be indefinite)
            B = randn(d, d)
            ΔX = (B + B') / 2

            γ = 0.99
            τ = step_length_block(X, ΔX, γ)

            # X + τ ΔX should be positive definite
            X_new = X + τ * ΔX
            @test isposdef(Symmetric(X_new))

            # At the computed τ, we should be just inside the boundary
            # (not at it, due to γ < 1)
            @test τ > 0
            @test τ <= 1
        end
    end

    @testset "step_to_boundary for full problem" begin
        Random.seed!(123)

        # Build a simple block structure
        nv = 3
        n = sum(trinum.([2, 2, 2]))  # 3 + 3 + 3 = 9

        # Build a minimal B
        rows = [1, 1, 2, 2]
        cols = [1, 2, 2, 3]
        V = [randn(2, 3), randn(2, 3), randn(2, 3), randn(2, 3)]
        B = blocksparse(rows, cols, V, 2, nv)

        # Random SPD P, D
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

        # Random directions (can push toward boundary)
        Δp = randn(n)
        Δd = randn(n)

        τ_p, τ_d = step_to_boundary(p, d, Δp, Δd, B, Val(:U); γ=0.99)

        # Step lengths should be positive and at most 1
        @test τ_p > 0
        @test τ_p <= 1
        @test τ_d > 0
        @test τ_d <= 1

        # After stepping, all blocks should be positive definite
        p_new = p + τ_p * Δp
        d_new = d + τ_d * Δd

        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            P_new = zeros(d_v, d_v)
            D_new = zeros(d_v, d_v)
            smat!(P_new, view(p_new, r), Val(:U))
            symmetrize!(P_new, Val(:U))
            smat!(D_new, view(d_new, r), Val(:U))
            symmetrize!(D_new, Val(:U))

            @test isposdef(Symmetric(P_new))
            @test isposdef(Symmetric(D_new))
        end
    end

    @testset "step_to_boundary with Newton direction from IPM" begin
        Random.seed!(456)

        # Build sheaf
        nv = 2
        dv = 2
        de = 2
        edges = [(1, 2)]
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

        # Assemble H
        using SheafSDP: hess!, affine_rhs!, newton_step!, residuals!
        using CliqueTrees.Multifrontal: FactorizationWorkspace, DivisionWorkspace

        H_blocks = Matrix{Float64}[]
        W_blocks = Matrix{Float64}[]
        H = hess!(H_blocks, W_blocks, p, d, B, Val(:U))
        H_sp = sparse(H)

        r_p = zeros(m)
        r_d = zeros(n)
        residuals!(r_p, r_d, B_sp, p, d, y, c, g)

        r_c = zeros(n)
        affine_rhs!(r_c, p)

        facwrk = FactorizationWorkspace(F)
        divwrk = DivisionWorkspace(F, 1)
        itrwrk = RiWorkspace(m, Vector{Float64})
        r = zeros(m)

        Δp = zeros(n)
        Δy = zeros(m)
        Δd = zeros(n)

        newton_step!(Δp, Δy, Δd, facwrk, divwrk, itrwrk, r, F, L, B, B_sp, H, H_sp,
                     r_c, r_p, r_d; τ=1.0, atol=1e-12, rtol=1e-12)

        # Compute step lengths
        τ_p, τ_d = step_to_boundary(p, d, Δp, Δd, B, Val(:U); γ=0.99)

        @test τ_p > 0
        @test τ_d > 0

        # After stepping, should remain positive definite
        p_new = p + τ_p * Δp
        d_new = d + τ_d * Δd

        for v in vtxs(B)
            r = colrange(B, v)
            n_v = ncols(B, v)
            d_v = triroot(n_v)

            P_new = zeros(d_v, d_v)
            D_new = zeros(d_v, d_v)
            smat!(P_new, view(p_new, r), Val(:U))
            symmetrize!(P_new, Val(:U))
            smat!(D_new, view(d_new, r), Val(:U))
            symmetrize!(D_new, Val(:U))

            @test isposdef(Symmetric(P_new))
            @test isposdef(Symmetric(D_new))
        end
    end

end
