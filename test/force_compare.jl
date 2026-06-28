using SheafSDP
using SheafSDP: svec!
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse
using Printf

Random.seed!(42)

svecdim(n) = div(n * (n + 1), 2)

# Simple SDP: max tr(CX) s.t. tr(AX) = b, X ≥ 0
function simple_sdp(n)
    sv = svecdim(n)
    A = randn(n, n); A = A + A'
    C = randn(n, n); C = C + C'
    b = 1.0

    a_vec = zeros(sv)
    svec!(a_vec, A)
    c_vec = zeros(sv)
    svec!(c_vec, C)

    row_ids = [1]
    col_ids = [1]
    blocks = [reshape(a_vec, sv, 1)]
    B = blocksparse(row_ids, col_ids, blocks)

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0.0)

    cones = [SheafSDP.SemidefiniteCone()]
    g_vec = [b]

    return SheafSDP.IPMProblem(Q, B, c_vec, g_vec, cones)
end

# Simple LP
function simple_lp(m, n)
    A = randn(m, n)
    b = abs.(randn(m))
    c = randn(n)

    row_ids = [1]
    col_ids = [1]
    blocks = [A]
    B = blocksparse(row_ids, col_ids, blocks)

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0.0)

    cones = [SheafSDP.PositiveCone()]

    return SheafSDP.IPMProblem(Q, B, c, b, cones)
end

# Simple SOC
function simple_soc(n)
    c = randn(n)
    A = randn(1, n)
    b = [10.0]

    row_ids = [1]
    col_ids = [1]
    blocks = [A]
    B = blocksparse(row_ids, col_ids, blocks)

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0.0)

    cones = [SheafSDP.SecondOrderCone()]

    return SheafSDP.IPMProblem(Q, B, c, b, cones)
end

function compare(name, prob)
    settings_off = SheafSDP.IPMSettings{Float64}(
        feas_tol=1e-8, gap_tol=1e-8, itmax=100, verbose=false,
        force_tol=0.0
    )
    settings_on = SheafSDP.IPMSettings{Float64}(
        feas_tol=1e-8, gap_tol=1e-8, itmax=100, verbose=false,
        force_tol=1e-3
    )

    r_off = solve(prob, settings_off)
    r_on = solve(prob, settings_on)

    cg_diff = r_on.kkt_niter - r_off.kkt_niter

    @printf("%-15s | OFF: %-12s %2d IPM %4d CG | ON: %-12s %2d IPM %4d CG | Δ=%+d\n",
        name,
        r_off.status, r_off.ipm_niter, r_off.kkt_niter,
        r_on.status, r_on.ipm_niter, r_on.kkt_niter,
        cg_diff)
end

println("Force-tol comparison: OFF (0) vs ON (1e-3)")
println("="^95)

compare("SDP 8x8", simple_sdp(8))
compare("SDP 16x16", simple_sdp(16))
compare("LP 20x50", simple_lp(20, 50))
compare("LP 50x100", simple_lp(50, 100))
compare("SOC n=20", simple_soc(20))
compare("SOC n=50", simple_soc(50))
