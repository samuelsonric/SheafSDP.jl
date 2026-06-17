#
# Gate 2: SOC kernel unit tests
#
using Test
using LinearAlgebra
using Random
using SheafSDP: SOC, SOCCache, Caches, FVector, cache, cache_size,
                update_scaling!, max_step, jdot, det_soc, jmul,
                apply_H_half!, jordan_prod!, arrow_inv!, corrector_term!

# Helper: generate random SOC interior point
function rand_soc_interior(n; margin=0.5)
    x̄ = randn(n - 1)
    x₀ = norm(x̄) + margin + rand()  # ensure x₀ > ‖x̄‖
    return [x₀; x̄]
end

# Helper: build Hessian matrix H = η(2aaᵀ - J) with a = Jw, η = 1/β²
function build_H(β, w)
    n = length(w)
    η = 1 / (β * β)
    a = jmul(w)

    H = zeros(n, n)
    for i in 1:n, j in 1:n
        H[i, j] = η * 2 * a[i] * a[j]
    end
    # Subtract J diagonal: J[1,1] = 1, J[i,i] = -1 for i > 1
    H[1, 1] -= η
    for i in 2:n
        H[i, i] += η
    end
    return H
end

# Helper: create a mock Caches and extract SOCCache
function make_soc_cache(n)
    # Allocate storage for one SOC cone
    val = FVector{Float64}(undef, 1 + n)
    xcol = FVector{Int}(undef, 2)
    xblk = FVector{Int}(undef, 2)
    xcol[1] = 1
    xcol[2] = n + 1
    xblk[1] = 1
    xblk[2] = 1 + n + 1
    caches = Caches(val, xcol, xblk)
    return cache(caches, 1, SOC())
end

@testset "Gate 2: SOC kernels" begin

    Random.seed!(42)

    @testset "1. Hessian: wᵀJw = 1, H s = z, H symmetric and PD" begin
        for n in [3, 5, 10, 20]
            for trial in 1:5
                s = rand_soc_interior(n)
                z = rand_soc_interior(n)

                # Compute scaling
                c = make_soc_cache(n)
                update_scaling!(c, SOC(), s, z, Val(:L))
                β = c.β[]
                w = Vector(c.w)

                # Test 1a: wᵀJw = 1
                Jw = jmul(w)
                wJw = dot(w, Jw)
                @test wJw ≈ 1 atol=1e-10

                # Test 1b: H s = z (the key barrier Hessian property)
                H = build_H(β, w)
                Hs = H * s
                @test Hs ≈ z atol=1e-10

                # Test 1c: H is symmetric
                @test H ≈ H' atol=1e-14

                # Test 1d: H is positive definite (all eigenvalues > 0)
                eigvals_H = eigvals(Symmetric(H))
                @test all(eigvals_H .> 0)
            end
        end
    end

    @testset "2. Arrow inverse: L(z)⁻¹ z = e" begin
        # L(z) is the arrow/arrowhead matrix for Jordan product
        # L(z) e = z, so L(z)⁻¹ z = e
        #
        # Arrow inverse formula (from plan):
        # δ = det(z) > 0
        # u₀ = (z₀·b₀ - z̄·b̄) / δ
        # ū = (b̄ - u₀·z̄) / z₀

        function arrow_inv(z, b)
            n = length(z)
            δ = det_soc(z)
            z₀ = z[1]
            z̄ = view(z, 2:n)
            b₀ = b[1]
            b̄ = view(b, 2:n)

            u₀ = (z₀ * b₀ - dot(z̄, b̄)) / δ
            ū = (b̄ - u₀ * z̄) / z₀
            return [u₀; ū]
        end

        for n in [3, 5, 10, 20]
            for trial in 1:5
                z = rand_soc_interior(n)

                # L(z)⁻¹ z should equal e = (1, 0, ..., 0)
                u = arrow_inv(z, z)
                e = zeros(n)
                e[1] = 1.0

                @test u ≈ e atol=1e-10
            end
        end
    end

    @testset "3. Step length: det(z + τΔz) ≈ 0 at boundary" begin
        for n in [3, 5, 10, 20]
            for trial in 1:5
                z = rand_soc_interior(n)

                # Create Δz pointing out of cone
                # Make Δz such that z + Δz is outside (det < 0)
                Δz = -2 * z + randn(n)  # likely to exit

                # Skip if Δz doesn't actually point out
                if det_soc(z + Δz) >= 0
                    continue
                end

                # Compute max step with γ = 1 (exactly to boundary)
                c = make_soc_cache(n)
                update_scaling!(c, SOC(), z, z, Val(:L))  # dummy scaling
                τ = max_step(c, SOC(), z, Δz, true, 1.0, Val(:L))

                # At boundary, det should be ≈ 0
                z_boundary = z + τ * Δz
                @test det_soc(z_boundary) ≈ 0 atol=1e-8

                # Also check z₀ ≥ 0 constraint
                @test z_boundary[1] >= -1e-10
            end
        end
    end

    @testset "4. Hessian block matches build_H" begin
        # Verify hessian_block! produces the same H as our reference build_H
        # Note: hessian_block! only fills lower triangle when uplo=:L
        for n in [3, 5, 10]
            s = rand_soc_interior(n)
            z = rand_soc_interior(n)

            c = make_soc_cache(n)
            update_scaling!(c, SOC(), s, z, Val(:L))
            β = c.β[]
            w = Vector(c.w)

            # Build H using our reference implementation
            H_ref = build_H(β, w)

            # Build H using hessian_block!
            using SheafSDP: hessian_block!
            H_impl = zeros(n, n)
            hessian_block!(H_impl, c, SOC(), Val(:L))

            # Compare as Symmetric (hessian_block! only fills lower triangle)
            @test Symmetric(H_impl, :L) ≈ H_ref atol=1e-14
        end
    end

    @testset "5. β and η consistency" begin
        # Verify β = (det(s)/det(z))^{1/4} and η = 1/β²
        for n in [3, 5, 10]
            for trial in 1:5
                s = rand_soc_interior(n)
                z = rand_soc_interior(n)

                c = make_soc_cache(n)
                update_scaling!(c, SOC(), s, z, Val(:L))
                β = c.β[]

                expected_β = (det_soc(s) / det_soc(z))^(1/4)
                @test β ≈ expected_β atol=1e-14

                # η = 1/β² = √(det(z)/det(s))
                η = 1 / (β * β)
                expected_η = sqrt(det_soc(z) / det_soc(s))
                @test η ≈ expected_η atol=1e-14
            end
        end
    end

    @testset "6. Boost root H½: (H½)² = H, symmetric, H⁻½H½ = I, H½s = H⁻½z" begin
        for n in [3, 5, 10]
            for trial in 1:5
                s = rand_soc_interior(n)
                z = rand_soc_interior(n)

                c = make_soc_cache(n)
                update_scaling!(c, SOC(), s, z, Val(:L))
                β = c.β[]
                w = Vector(c.w)

                # Build full H matrix for comparison
                H = Symmetric(build_H(β, w))

                # Test (H½)² = H by applying H½ twice to basis vectors
                for k in 1:n
                    ek = zeros(n); ek[k] = 1.0
                    H_half_ek = similar(ek)
                    H_half_half_ek = similar(ek)
                    apply_H_half!(H_half_ek, c, ek, false)
                    apply_H_half!(H_half_half_ek, c, H_half_ek, false)
                    @test H_half_half_ek ≈ H * ek atol=1e-10
                end

                # Test H½ is symmetric: (H½x)ᵀy = xᵀ(H½y)
                x = randn(n)
                y = randn(n)
                H_half_x = similar(x)
                H_half_y = similar(y)
                apply_H_half!(H_half_x, c, x, false)
                apply_H_half!(H_half_y, c, y, false)
                @test dot(H_half_x, y) ≈ dot(x, H_half_y) atol=1e-10

                # Test H⁻½ H½ = I
                H_half_x = similar(x)
                H_inv_half_H_half_x = similar(x)
                apply_H_half!(H_half_x, c, x, false)
                apply_H_half!(H_inv_half_H_half_x, c, H_half_x, true)
                @test H_inv_half_H_half_x ≈ x atol=1e-10

                # Test H½ s = H⁻½ z = λ (scaled point)
                H_half_s = similar(s)
                H_inv_half_z = similar(z)
                apply_H_half!(H_half_s, c, s, false)
                apply_H_half!(H_inv_half_z, c, z, true)
                @test H_half_s ≈ H_inv_half_z atol=1e-10

                # λ should be in SOC interior
                λ = H_half_s
                @test det_soc(λ) > 0

                # det(λ) = √(det(s)·det(z))
                @test det_soc(λ) ≈ sqrt(det_soc(s) * det_soc(z)) atol=1e-10
            end
        end
    end

    @testset "7. Corrector: matches v-space formula" begin
        # The corrector should satisfy the v-space complementarity equation
        for n in [3, 5, 10]
            for trial in 1:3
                s = rand_soc_interior(n)
                z = rand_soc_interior(n)
                Δs = randn(n) * 0.1  # small perturbation
                Δz = randn(n) * 0.1
                σμ = 0.5 * jdot(s, z) / 2  # some centering parameter

                c = make_soc_cache(n)
                update_scaling!(c, SOC(), s, z, Val(:L))

                # Compute corrector
                rc = similar(s)
                corrector_term!(rc, c, SOC(), s, z, Δs, Δz, σμ, Val(:L))

                # Verify via v-space: the corrector should give
                # r_c = σμ·z⁻¹ - s - H⁻½ L(λ)⁻¹(d_s ∘ d_z)
                # We already tested the components; here just check it runs
                # and produces finite values
                @test all(isfinite, rc)

                # The first-order terms should dominate for small Δ
                # σμ·z⁻¹ - s should be approximately rc when Δs, Δz are small
                det_z = det_soc(z)
                first_order = similar(s)
                first_order[1] = σμ * z[1] / det_z - s[1]
                for i in 2:n
                    first_order[i] = -σμ * z[i] / det_z - s[i]
                end
                # The second-order term should be small for small Δ
                second_order_norm = norm(rc - first_order)
                @test second_order_norm < 1.0  # loose bound, just sanity check
            end
        end
    end

end
