using Test
using LinearAlgebra
using SheafSDP: trinum, triroot, svec!, smat!, symmetrize!, meanblock!, hessblock!

@testset "Phase 2: NT scaling and Hessian H" begin

    @testset "meanblock! produces valid W" begin
        for d in [2, 3, 4, 5]
            # random SPD matrices P, D
            A = randn(d, d)
            P = A * A' + I
            A = randn(d, d)
            D = A * A' + I

            WP = zeros(d, d)
            WD = zeros(d, d)

            W = meanblock!(WP, WD, P, D)

            # W should be symmetric
            @test W ≈ W'

            # W should be SPD (Cholesky should succeed)
            @test isposdef(W)

            # W D W = P
            @test W * D * W ≈ P atol=1e-10
        end
    end

    @testset "hessblock! produces SPD H" begin
        for d in [2, 3, 4]
            k = trinum(d)

            # random SPD W
            A = randn(d, d)
            W = A * A' + I

            H = zeros(k, k)
            work = zeros(d, d)

            hessblock!(H, W, work, Val(:U))

            # H should be symmetric
            @test H ≈ H'

            # H should be SPD
            @test isposdef(Symmetric(H))
        end
    end

    @testset "load-bearing identity: H p = d" begin
        for d in [2, 3, 4, 5]
            k = trinum(d)

            # random SPD P, D
            A = randn(d, d)
            P = A * A' + I
            A = randn(d, d)
            D = A * A' + I

            # compute W
            WP = zeros(d, d)
            WD = zeros(d, d)
            W = meanblock!(WP, WD, P, D)

            # compute H from W
            H = zeros(k, k)
            work = zeros(d, d)
            hessblock!(H, W, work, Val(:U))

            # svec P and D
            p = zeros(k)
            d_vec = zeros(k)
            svec!(p, P, Val(:U))
            svec!(d_vec, D, Val(:U))

            # H p should equal d
            # because H: svec(M) -> svec(W⁻¹ M W⁻¹)
            # and W⁻¹ P W⁻¹ = D (since W D W = P implies W⁻¹ P W⁻¹ = D)
            @test H * p ≈ d_vec atol=1e-10
        end
    end

    @testset "singular values and inner product" begin
        for d in [2, 3, 4]
            # random SPD P, D
            A = randn(d, d)
            P = A * A' + I
            A = randn(d, d)
            D = A * A' + I

            # compute W via meanblock! to get SVD internally
            # We'll verify Σ s_i² ≈ ⟨P, D⟩ by recomputing

            # L_P = chol(P), L_D = chol(D)
            L_P = cholesky(P).L
            L_D = cholesky(D).L

            # G = L_Pᵀ L_D
            G = L_P' * L_D

            # SVD
            F = svd(G)
            s = F.S

            # Σ s_i² should equal tr(P D) = ⟨P, D⟩
            @test sum(s.^2) ≈ tr(P * D) atol=1e-10

            # Also: s_i² = λ_i(P^{1/2} D P^{1/2})
            P_sqrt = sqrt(Symmetric(P))
            eigvals_check = eigvals(Symmetric(P_sqrt * D * P_sqrt))
            @test sort(s.^2) ≈ sort(eigvals_check) atol=1e-10
        end
    end

end
