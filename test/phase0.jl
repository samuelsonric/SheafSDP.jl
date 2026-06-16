using Test
using LinearAlgebra
using SheafSDP: trinum, triroot, svec!, smat!

@testset "Phase 0: dimension bookkeeping and svec isometry" begin

    @testset "trinum/triroot inverses" begin
        for n in 1:20
            k = trinum(n)
            @test triroot(k) == n
        end
    end

    @testset "svec/smat round-trip" begin
        for n in [1, 2, 3, 5, 10]
            k = trinum(n)

            # random symmetric matrix
            M = randn(n, n)
            M = (M + M') / 2

            v = zeros(k)
            M2 = zeros(n, n)

            # round-trip with :U (smat fills upper triangle only)
            svec!(v, M, Val(:U))
            smat!(M2, v, Val(:U))
            @test Symmetric(M2, :U) ≈ M

            # round-trip with :L (smat fills lower triangle only)
            fill!(M2, 0)
            svec!(v, M, Val(:L))
            smat!(M2, v, Val(:L))
            @test Symmetric(M2, :L) ≈ M
        end
    end

    @testset "svec isometry: p'd = tr(PD)" begin
        for n in [1, 2, 3, 5, 10]
            k = trinum(n)

            # random symmetric matrices
            P = randn(n, n); P = (P + P') / 2
            D = randn(n, n); D = (D + D') / 2

            p = zeros(k)
            d = zeros(k)

            svec!(p, P, Val(:U))
            svec!(d, D, Val(:U))

            # isometry: inner product in svec space equals trace
            @test p' * d ≈ tr(P * D)
        end
    end

    @testset "svec scaling: off-diagonals scaled by √2" begin
        # 2x2 case: check explicit values
        M = [1.0 2.0; 2.0 3.0]
        v = zeros(3)
        svec!(v, M, Val(:U))

        # Upper storage: (1,1), (1,2), (2,2) -> 1, 2√2, 3
        @test v[1] ≈ 1.0
        @test v[2] ≈ 2.0 * √2
        @test v[3] ≈ 3.0
    end

end
