# test/pow_digitloss.jl
#
# Componentwise digit-loss harness for the PowerCone Hessian factorization.
#
# Principle: the cone math is generic over T, so the BigFloat oracle is the
# SAME source run at higher precision — no reimplementation. For one iterate we
# compute every named intermediate in Float64 and in BigFloat and report
# digits-correct = -log10(relerr) per quantity. The quantity whose row falls off
# a cliff is where the digits die. κ(F'') is reported as *context* only: the gap
# between (17 - log10 κ) and what a method actually delivers is the structure it
# is (or isn't) exploiting.
#
# Usage:
#   include("test/pow_digitloss.jl")
#   PowDigitLoss.report(x, α)                 # one iterate: per-quantity table
#   PowDigitLoss.sweep(α)                     # boundary march: digits vs floor
#   PowDigitLoss.capture(x, s, α; tag="bad")  # dump a fixture from a live run
#   PowDigitLoss.replay("bad")                # report() on a captured fixture
#
# NOTE: logic + numbers validated in a Float64/mpmath twin; treat the Julia
# itself as needing one syntax pass. Adjust the qualified names to your module.

module PowDigitLoss

using LinearAlgebra, Printf, Serialization
import SheafSDP                       # powhess! lives here (internal, qualify it)

const HESS! = SheafSDP.powhess!       # (H, x, α) -> fills 3x3
setprecision(BigFloat, 256)

safe_sqrt(x::T) where {T} = x > 0 ? sqrt(x) : T(NaN)

# -log10 relative error of a Float64 quantity vs a BigFloat reference, in [0,17]
function digits_correct(approx, ref::BigFloat)
    a = BigFloat(approx)
    ref == 0 && return a == 0 ? 17.0 : 0.0
    (isnan(a) || isinf(a)) && return 0.0
    rel = abs(a - ref) / abs(ref)
    rel == 0 ? 17.0 : clamp(Float64(-log10(rel)), 0.0, 17.0)
end

# scalar building blocks, generic over T (mirror of pow.jl)
function pieces(x::AbstractVector{T}, α::T) where {T}
    a = 2α; b = 2 * (one(T) - α)
    p = x[1]^a * x[2]^b
    φ = p - x[3]^2
    ρ = p / φ
    d1 = (2ρ * a + b) / (2 * x[1]^2)
    d2 = (2ρ * b + a) / (2 * x[2]^2)
    return (; a, b, p, φ, ρ, d1, d2)
end

# --- naive: natural-order Cholesky on the assembled Hessian, radicands kept ---
function factor_naive(x::AbstractVector{T}, α::T) where {T}
    H = Matrix{T}(undef, 3, 3); HESS!(H, x, α)
    L = copy(H); r = zeros(T, 3)
    r[1] = L[1,1];                       L[1,1] = safe_sqrt(r[1])
    L[2,1] /= L[1,1]; L[3,1] /= L[1,1]
    r[2] = L[2,2] - L[2,1]^2;            L[2,2] = safe_sqrt(r[2])
    L[3,2] = (L[3,2] - L[3,1]*L[2,1]) / L[2,2]
    r[3] = L[3,3] - L[3,1]^2 - L[3,2]^2; L[3,3] = safe_sqrt(r[3])
    return (; L, rad = r, perm = (1, 2, 3))
end

# --- structured: pivot coord 3 first, symbolic Schur collapse (guide §5) ------
function factor_struct(x::AbstractVector{T}, α::T) where {T}
    H = Matrix{T}(undef, 3, 3); HESS!(H, x, α)   # reuse cancellation-free §4 entries
    pc = pieces(x, α)
    D33 = H[3,3]
    c   = pc.p * x[3]^2 / (pc.φ * (pc.p + x[3]^2))   # the collapse: O(φ⁻¹), not O(φ⁻²)
    ℓ1  = pc.a / x[1]; ℓ2 = pc.b / x[2]
    r1  = pc.d1 - c * ℓ1^2
    r2  = (pc.d1*pc.d2 - c*(pc.d1*ℓ2^2 + pc.d2*ℓ1^2)) / r1
    L = zeros(T, 3, 3)                               # stored in permuted (3,1,2) order
    L[1,1] = safe_sqrt(D33)
    L[2,1] = H[1,3] / L[1,1]; L[3,1] = H[2,3] / L[1,1]
    L[2,2] = safe_sqrt(r1)
    L[3,2] = -c * ℓ1 * ℓ2 / L[2,2]
    L[3,3] = safe_sqrt(r2)
    return (; L, rad = T[D33, r1, r2], perm = (3, 1, 2))
end

# solve F'' v = w using a factor returned above (handles the permutation)
function fsolve(fac, w::AbstractVector{T}) where {T}
    p = fac.perm; L = fac.L
    b = T[w[p[1]], w[p[2]], w[p[3]]]
    b[1] /= L[1,1]
    b[2]  = (b[2] - L[2,1]*b[1]) / L[2,2]
    b[3]  = (b[3] - L[3,1]*b[1] - L[3,2]*b[2]) / L[3,3]
    b[3] /= L[3,3]
    b[2]  = (b[2] - L[3,2]*b[3]) / L[2,2]
    b[1]  = (b[1] - L[2,1]*b[2] - L[3,1]*b[3]) / L[1,1]
    v = zeros(T, 3); v[p[1]] = b[1]; v[p[2]] = b[2]; v[p[3]] = b[3]
    return v
end

# collect every named intermediate in type T for one iterate
function trace(x, α, ::Type{T}; w = T[1, -0.5, 0.3]) where {T}
    xT = T.(x); αT = T(α)
    H = Matrix{T}(undef, 3, 3); HESS!(H, xT, αT)
    pc = pieces(xT, αT)
    fn = factor_naive(xT, αT); fs = factor_struct(xT, αT)
    q = Dict{String,T}()
    for (k, v) in pairs(pc); q[String(k)] = v; end
    q["H11"]=H[1,1]; q["H22"]=H[2,2]; q["H33"]=H[3,3]
    q["H12"]=H[1,2]; q["H13"]=H[1,3]; q["H23"]=H[2,3]
    q["naive_rad1"]=fn.rad[1]; q["naive_rad2"]=fn.rad[2]; q["naive_rad3"]=fn.rad[3]
    q["struct_D33"]=fs.rad[1]; q["struct_r1"]=fs.rad[2]; q["struct_r2"]=fs.rad[3]
    vn = fsolve(fn, T.(w)); vs = fsolve(fs, T.(w))
    q["solve_naive_1"]=vn[1]; q["solve_naive_2"]=vn[2]; q["solve_naive_3"]=vn[3]
    q["solve_struct_1"]=vs[1]; q["solve_struct_2"]=vs[2]; q["solve_struct_3"]=vs[3]
    return q
end

# normwise condition number of F'' (context only, so Float64 is fine)
function cond_big(x, α)
    H = Matrix{Float64}(undef, 3, 3); HESS!(H, Float64.(x), Float64(α))
    e = abs.(eigvals(Symmetric(H)))
    return maximum(e) / minimum(e)
end

# ---------------------------------------------------------------------------
# report: per-quantity digit-loss table for one iterate
# ---------------------------------------------------------------------------
function report(x, α; w = [1.0, -0.5, 0.3])
    qf = trace(Float64.(x), Float64(α), Float64; w = Float64.(w))
    qb = trace(BigFloat.(x), BigFloat(α), BigFloat; w = BigFloat.(w))
    κ = cond_big(x, α)
    @printf("φ = %.3e   log10 κ(F'') = %.1f   normwise floor = %.1f digits\n",
            Float64(pieces(BigFloat.(x), BigFloat(α)).φ), Float64(log10(κ)),
            17 - Float64(log10(κ)))
    println("-"^54)
    @printf("%-16s %10s   %s\n", "quantity", "digits", "")
    order = ["p","φ","ρ","d1","d2","H11","H22","H33","H12","H13","H23",
             "naive_rad3","struct_r2",
             "solve_naive_1","solve_naive_2","solve_naive_3",
             "solve_struct_1","solve_struct_2","solve_struct_3"]
    for k in order
        haskey(qf, k) || continue
        d = digits_correct(qf[k], qb[k])
        bar = d < 1 ? "  <-- DEAD" : d < 8 ? "  <-- lossy" : ""
        @printf("%-16s %10.1f%s\n", k, d, bar)
    end
end

# ---------------------------------------------------------------------------
# sweep: march toward the boundary, naive vs structured digits in the solve,
# against the normwise floor. (the table from the chat.)
# ---------------------------------------------------------------------------
function sweep(α; x1 = 1.3, x2 = 0.8, ts = 10.0 .^ (-2:-2:-15), w = [1.0,-0.5,0.3])
    a = 2α; b = 2*(1-α); p = x1^a * x2^b
    @printf("%9s %8s %7s | %7s %7s\n", "φ","log10κ","floor","naive","struct")
    for t in ts
        x3 = sqrt(p*(1-t)); x = [x1, x2, x3]
        κ = cond_big(x, α); floor = 17 - Float64(log10(κ))
        wb = BigFloat.(w)
        Hb = Matrix{BigFloat}(undef,3,3); HESS!(Hb, BigFloat.(x), BigFloat(α))
        vt = Hb \ wb
        dn = let v = fsolve(factor_naive(Float64.(x), Float64(α)), Float64.(w))
            minimum(digits_correct(v[i], vt[i]) for i in 1:3)
        end
        ds = let v = fsolve(factor_struct(Float64.(x), Float64(α)), Float64.(w))
            minimum(digits_correct(v[i], vt[i]) for i in 1:3)
        end
        @printf("%9.1e %8.1f %7.1f | %7.1f %7.1f\n",
                Float64(p*t), Float64(log10(κ)), floor, dn, ds)
    end
end

# ---------------------------------------------------------------------------
# fixture capture: drop this at the point a live solve fails (chol throws or
# cond exceeds a threshold) to serialize the exact iterate for replay.
# ---------------------------------------------------------------------------
function capture(x, s, α; tag = "bad", dir = joinpath(@__DIR__, "fixtures"))
    mkpath(dir)
    serialize(joinpath(dir, "$tag.jls"), (; x = Float64.(x), s = Float64.(s), α = Float64(α)))
    @info "captured fixture" tag
end

function replay(tag; dir = joinpath(@__DIR__, "fixtures"))
    f = deserialize(joinpath(dir, "$tag.jls"))
    report(f.x, f.α)
end

# ---------------------------------------------------------------------------
# regression assertions (drop into a @testset). The bar is structured-vs-its-
# own-BigFloat, NOT the normwise floor, plus no nonpositive radicands.
# ---------------------------------------------------------------------------
function assert_clean(x, α; minrad = 0.0, mindigits = 2.0)
    fs = factor_struct(Float64.(x), Float64(α))
    @assert all(r -> r > minrad, fs.rad) "structured radicand nonpositive: $(fs.rad)"
    qb = trace(BigFloat.(x), BigFloat(α), BigFloat)
    qf = trace(Float64.(x), Float64(α), Float64)
    for k in ("solve_struct_1","solve_struct_2","solve_struct_3")
        d = digits_correct(qf[k], qb[k])
        @assert d ≥ mindigits "structured solve only $d digits at φ-tight point ($k)"
    end
    return true
end

end # module
