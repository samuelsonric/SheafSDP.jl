using Test
using LinearAlgebra
using SparseArrays
using SheafSDP: trinum, triroot, svec!, smat!, conedegree, mu, residuals!
using BlockSparseArrays: blocksparse, ncols, vtxs, colrange

@testset "Phase 1: residuals and mu" begin

    @testset "conedegree" begin
        # Create a simple block structure: 3 vertices with dims 2, 3, 2
        # svec lengths: 3, 6, 3 = 12 total
        # cone degree ν = 2 + 3 + 2 = 7

        # Build a dummy B with the right column structure
        # B is m × n where n = 12 (sum of svec dims)
        # Use 2 edges, so m = 2 * edge_dim
        edge_dim = 2

        # Blocks: vertex 1 -> edge 1, vertex 2 -> edge 1, vertex 2 -> edge 2, vertex 3 -> edge 2
        rows = [1, 1, 2, 2]  # edge indices
        cols = [1, 2, 2, 3]  # vertex indices
        V = [randn(edge_dim, trinum(2)), randn(edge_dim, trinum(3)),
             randn(edge_dim, trinum(3)), randn(edge_dim, trinum(2))]

        B = blocksparse(rows, cols, V, 2, 3)

        @test conedegree(B) == 2 + 3 + 2
    end

    @testset "mu with P = D = I" begin
        # For P = D = I (identity blocks), μ should be 1
        # Because tr(I * I) = d_v per block, sum = ν, so μ = ν/ν = 1

        dims = [2, 3, 2]  # matrix dimensions
        ν = sum(dims)
        n = sum(trinum.(dims))  # svec length

        p = zeros(n)
        d = zeros(n)

        # Fill p and d with svec(I) for each block
        offset = 0
        for dim in dims
            k = trinum(dim)
            I_mat = Matrix{Float64}(I, dim, dim)
            svec!(view(p, offset+1:offset+k), I_mat, Val(:U))
            svec!(view(d, offset+1:offset+k), I_mat, Val(:U))
            offset += k
        end

        @test mu(p, d, ν) ≈ 1.0
        @test dot(p, d) ≈ ν  # isometry: p'd = tr(PD) = Σ d_v
    end

    @testset "mu isometry check" begin
        # Verify p'd = Σ tr(P_v D_v) for random SPD blocks

        dims = [2, 3, 4]
        ν = sum(dims)
        n = sum(trinum.(dims))

        p = zeros(n)
        d = zeros(n)

        trace_sum = 0.0
        offset = 0
        for dim in dims
            k = trinum(dim)

            # random SPD matrices
            A = randn(dim, dim)
            P = A * A' + I
            A = randn(dim, dim)
            D = A * A' + I

            svec!(view(p, offset+1:offset+k), P, Val(:U))
            svec!(view(d, offset+1:offset+k), D, Val(:U))

            trace_sum += tr(P * D)
            offset += k
        end

        @test dot(p, d) ≈ trace_sum
    end

    @testset "residuals at feasible point" begin
        # Build a simple problem where we know a feasible point
        # P = D = I, choose y so that Bᵀy + d = c

        dims = [2, 2]
        n = sum(trinum.(dims))  # 3 + 3 = 6
        edge_dim = 2
        m = edge_dim  # 1 edge

        # Simple B: one edge connecting two vertices
        rows = [1, 1]
        cols = [1, 2]
        V = [randn(edge_dim, trinum(2)), randn(edge_dim, trinum(2))]
        B = blocksparse(rows, cols, V, 1, 2)
        B_sp = sparse(B)

        # p = svec(I) for each block
        p = zeros(n)
        offset = 0
        for dim in dims
            k = trinum(dim)
            I_mat = Matrix{Float64}(I, dim, dim)
            svec!(view(p, offset+1:offset+k), I_mat, Val(:U))
            offset += k
        end

        # g = B p (so r_p = 0)
        g = B_sp * p

        # d = svec(I) for each block
        d = copy(p)

        # c = Bᵀy + d for some y (so r_d = 0)
        y = randn(m)
        c = B_sp' * y + d

        rp = zeros(m)
        rd = zeros(n)
        residuals!(rp, rd, B_sp, p, d, y, c, g)

        @test norm(rp) < 1e-12
        @test norm(rd) < 1e-12
    end

end
