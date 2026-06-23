using SheafSDP
using LinearAlgebra
using Printf

# Import internal functions for testing
import SheafSDP: powequil!, powscalein!, powscaleout!, powhess!, powphi, powincone, powindual

# Test the verification ladder from §5

function test_membership_invariance(α, x, s)
    # Check that D*x stays in cone and D⁻¹*s stays in dual cone
    t = zeros(3)
    xhat = zeros(3)
    shat = zeros(3)

    powequil!(t, x, s, α)
    powscalein!(xhat, t, x)
    powscaleout!(shat, t, s)

    in_primal = powincone(xhat, α)
    in_dual = powindual(shat, α)

    return in_primal && in_dual
end

function test_congruence_identity(α, x, s)
    # Check D F″(x̂) D = F″(x) to ~u
    t = zeros(3)
    xhat = zeros(3)
    Hx = zeros(3, 3)
    Hxhat = zeros(3, 3)

    powequil!(t, x, s, α)
    powscalein!(xhat, t, x)

    powhess!(Hx, x, α)
    powhess!(Hxhat, xhat, α)

    # D F″(x̂) D should equal F″(x)
    D = Diagonal([t[1], t[2], t[3]])
    DHD = D * Hxhat * D

    rel_err = norm(DHD - Hx) / norm(Hx)
    return rel_err
end

function test_conditioning_collapse(α, x₂_values)
    # Hold ρ (boundary proximity) fixed, drive x₂ → 0
    # Check that κ(F″(x̂)) is FLAT while κ(F″(x)) grows

    x₁ = 10.0  # fixed
    ρ = 100.0  # fixed boundary proximity

    results = []

    for x₂ in x₂_values
        # Compute x₃ to achieve the target ρ
        # ρ = p/φ = p/(p - x₃²) ⟹ x₃² = p(1 - 1/ρ)
        p = x₁^(2α) * x₂^(2(1-α))
        x₃_sq = p * (1 - 1/ρ)
        x₃_sq < 0 && continue
        x₃ = sqrt(x₃_sq)

        x = [x₁, x₂, x₃]

        # Check if in cone
        if !powincone(x, α)
            push!(results, (x₂, NaN, NaN, NaN))
            continue
        end

        # Compute s = -F'(x) (on central path)
        s = zeros(3)
        SheafSDP.powbarrgrad!(s, x, α)
        lmul!(-1, s)

        # Equilibrate
        t = zeros(3)
        xhat = zeros(3)
        powequil!(t, x, s, α)
        powscalein!(xhat, t, x)

        # Compute Hessians
        Hx = zeros(3, 3)
        Hxhat = zeros(3, 3)
        powhess!(Hx, x, α)
        powhess!(Hxhat, xhat, α)

        # Condition numbers
        κ_raw = cond(Hx)
        κ_equil = cond(Hxhat)
        stretch = x₁ / x₂

        push!(results, (x₂, stretch, log10(κ_raw), log10(κ_equil)))
    end

    return results
end

function test_pairing_invariance(α, x, s)
    # Check ⟨x̂,ŝ⟩ = ⟨x,s⟩
    t = zeros(3)
    xhat = zeros(3)
    shat = zeros(3)

    powequil!(t, x, s, α)
    powscalein!(xhat, t, x)
    powscaleout!(shat, t, s)

    pairing_orig = dot(x, s)
    pairing_scaled = dot(xhat, shat)

    rel_err = abs(pairing_scaled - pairing_orig) / abs(pairing_orig)
    return rel_err
end

# Run tests
println("="^70)
println("PowerCone Equilibration Verification (§5 ladder)")
println("="^70)
println()

# Test 1: Membership invariance
println("1. Membership invariance:")
for α in [0.2, 0.333, 0.5, 0.667, 0.8]
    x = [10.0, 1.0, 0.5]  # stretched
    s = zeros(3)
    SheafSDP.powbarrgrad!(s, x, α)
    lmul!(-1, s)

    pass = test_membership_invariance(α, x, s)
    println("   α = $α: ", pass ? "PASS" : "FAIL")
end
println()

# Test 2: Congruence identity D F″(x̂) D = F″(x)
println("2. Congruence identity (D F″(x̂) D = F″(x)):")
for α in [0.2, 0.333, 0.5, 0.667, 0.8]
    x = [10.0, 1.0, 0.5]
    s = zeros(3)
    SheafSDP.powbarrgrad!(s, x, α)
    lmul!(-1, s)

    rel_err = test_congruence_identity(α, x, s)
    status = rel_err < 1e-10 ? "PASS" : "FAIL"
    println(@sprintf("   α = %.3f: rel_err = %.2e  %s", α, rel_err, status))
end
println()

# Test 3: Conditioning collapse
println("3. Conditioning collapse (α = 0.2, x₁ = 10, varying x₂):")
println("   x₂       | stretch  | log₁₀κ(raw) | log₁₀κ(equil)")
println("   " * "-"^55)

x₂_values = [1.0, 0.1, 0.01, 0.001, 0.0001]
results = test_conditioning_collapse(0.2, x₂_values)

for (x₂, stretch, κ_raw, κ_equil) in results
    if isnan(κ_raw)
        println(@sprintf("   %.4f   | (out of cone)", x₂))
    else
        println(@sprintf("   %.4f   | %8.1f | %11.2f | %13.2f", x₂, stretch, κ_raw, κ_equil))
    end
end
println()

# Test 4: Pairing invariance ⟨x̂,ŝ⟩ = ⟨x,s⟩
println("4. Pairing invariance (⟨x̂,ŝ⟩ = ⟨x,s⟩):")
for α in [0.2, 0.333, 0.5, 0.667, 0.8]
    x = [10.0, 1.0, 0.5]
    s = zeros(3)
    SheafSDP.powbarrgrad!(s, x, α)
    lmul!(-1, s)

    rel_err = test_pairing_invariance(α, x, s)
    status = rel_err < 1e-14 ? "PASS" : "FAIL"
    println(@sprintf("   α = %.3f: rel_err = %.2e  %s", α, rel_err, status))
end
println()

println("="^70)
