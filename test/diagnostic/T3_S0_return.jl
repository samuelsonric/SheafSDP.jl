#
# T3 S₀ Return Check
#
# Does S₀ at step 6 "return" to something similar to step 4?
# If ||S₀(6) - S₀(4)|| << ||S₀(5) - S₀(4)||, that explains the period-2.
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using BlockSparseArrays: block, blocksparse, colrange

# Abbreviated problem builder
function svecdim(n); div(n * (n + 1), 2); end
function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C); α = roottwo(T); H = zeros(T, svecdim(d), svecdim(n)); tkl = 1
    @inbounds for l in 1:n
        tab = 0; for b in 1:d; Cbl = C[b, l]; tab += 1; H[tab, tkl] = Cbl^2
            for a in b + 1:d; tab += 1; H[tab, tkl] = α * C[a, l] * Cbl; end; end
        for kk in l + 1:n; tkl += 1; tab = 0
            for b in 1:d; Cbk, Cbl = C[b, kk], C[b, l]; tab += 1; H[tab, tkl] = α * Cbk * Cbl
                for a in b + 1:d; tab += 1; H[tab, tkl] = C[a, kk] * Cbl + C[a, l] * Cbk; end; end; end
        tkl += 1; end; return H; end
function passivity_lmi_operator(A::AbstractMatrix{T}, B_mat::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n, m = size(A, 1), size(B_mat, 2); nm = n + m; sv_G, sv_D = svecdim(n), svecdim(nm)
    L, d0 = zeros(T, sv_D, sv_G), zeros(T, sv_D); G, M, v = zeros(T, n, n), zeros(T, nm, nm), zeros(T, sv_D)
    for kk in 1:sv_G; fill!(G, zero(T)); smat!(G, setindex!(zeros(T, sv_G), one(T), kk))
        for ii in 1:n, jj in 1:ii-1; G[jj, ii] = G[ii, jj]; end
        M[1:n, 1:n] .= A * G .+ G * A'; M[1:n, n+1:nm] .= -G * C'; M[n+1:nm, 1:n] .= -C * G; M[n+1:nm, n+1:nm] .= zero(T)
        svec!(v, M); L[:, kk] .= v; end
    fill!(M, zero(T)); M[1:n, n+1:nm] .= B_mat; M[n+1:nm, 1:n] .= B_mat'; M[n+1:nm, n+1:nm] .= -(D .+ D'); svec!(d0, M); return L, d0; end
function random_passive_system(n::Int, rng=Random.default_rng()); Q = randn(rng, n, n); Q = Q'Q + I; A = -Q; B_mat = randn(rng, n, 1); C = B_mat'; D = fill(1.0 + abs(randn(rng)), 1, 1); return A, B_mat, C, D; end
function build_sdp_problem(N, n_i)
    T = Float64; Random.seed!(42); m_i = 1; d_e = min(2, n_i); edges = [(ii, jj) for ii in 1:N for jj in ii+1:N]
    base_system = random_passive_system(n_i); systems = [base_system for _ in 1:N]
    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}(); for _ in edges; C = zeros(T, d_e, n_i); for kk in 1:d_e; C[kk,kk] = 1.0; end; push!(interface_maps, (copy(C), copy(C))); end
    sv_G, sv_S, sv_edge = svecdim(n_i), svecdim(n_i + m_i), svecdim(d_e)
    col_G(idx) = 2*(idx-1)+1; col_S(idx) = 2*(idx-1)+2; row_diss(idx) = idx; row_agree(idx) = N + idx
    row_ids, col_ids, blocks, g_vec = Int[], Int[], Matrix{T}[], T[]
    for vi in 1:N; A, B_mat, C, D = systems[vi]; L, d0 = passivity_lmi_operator(A, B_mat, C, D)
        push!(row_ids, row_diss(vi)); push!(col_ids, col_S(vi)); push!(blocks, Matrix{T}(I, sv_S, sv_S))
        push!(row_ids, row_diss(vi)); push!(col_ids, col_G(vi)); push!(blocks, L); append!(g_vec, -d0); end
    for (e, (vi, vj)) in enumerate(edges); C_i, C_j = interface_maps[e]; K_i, K_j = skronr(C_i), skronr(C_j)
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(vi)); push!(blocks, K_i)
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(vj)); push!(blocks, -K_j); append!(g_vec, zeros(T, sv_edge)); end
    B = blocksparse(row_ids, col_ids, blocks); c_vec = zeros(T, size(B, 2)); I_n = Matrix{T}(I, n_i, n_i); svec_I = zeros(T, sv_G); svec!(svec_I, I_n)
    for vi in 1:N; c_vec[colrange(B, col_G(vi))] .= svec_I; end
    Q = SheafSDP.allocblockdiag(B); fill!(Q, zero(T)); cones = Vector{Cone}(undef, 2*N)
    for vi in 1:N; cones[col_G(vi)] = SemidefiniteCone(); cones[col_S(vi)] = SemidefiniteCone(); end
    return IPMProblem(c_vec, g_vec, B, Q, cones); end

function compute_S0(B, A)
    m, n = size(B); N = SheafSDP.nvtxs(B); Bdense, Adense = Matrix(B), zeros(n, n)
    for v in 1:N; rng = SheafSDP.colrange(B, v); Av = Matrix(block(A, v, v, v))
        any(isnan, Av) || any(isinf, Av) && return nothing; Adense[rng, rng] .= Av; end
    minimum(eigvals(Symmetric(Adense))) <= 0 && return nothing
    try; S0 = Bdense * (Adense \ Bdense'); return Symmetric((S0 + S0') / 2); catch; return nothing; end
end

function main()
    println("="^90)
    println("S₀ RETURN CHECK: Does S₀ oscillate with period 2?")
    println("="^90)

    N, n_i = 5, 4
    prob = build_sdp_problem(N, n_i)

    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=8)
    solver = SheafSDP.init(prob, settings)
    B = solver.B

    # Collect S₀ at each step
    S0_list = Matrix{Float64}[]

    for iter in 1:8
        SheafSDP.step!(solver)
        S0 = compute_S0(B, solver.H)
        if S0 !== nothing
            push!(S0_list, Matrix(S0))
        end
        gap = dot(solver.p, solver.d) / solver.ν
        @printf("Step %d: gap = %.2e\n", iter, gap)
    end

    println("\n" * "="^90)
    println("S₀ MATRIX DISTANCES (Frobenius norm, relative to step 4)")
    println("="^90)

    S4 = S0_list[4]
    norm_S4 = norm(S4)

    println("\nRelative distance from S₀(step 4):")
    for i in 1:length(S0_list)
        rel_dist = norm(S0_list[i] - S4) / norm_S4
        @printf("  ||S₀(%d) - S₀(4)|| / ||S₀(4)|| = %.4f\n", i, rel_dist)
    end

    println("\n" * "-"^90)
    println("Consecutive distances:")
    for i in 1:length(S0_list)-1
        rel_dist = norm(S0_list[i+1] - S0_list[i]) / norm(S0_list[i])
        @printf("  ||S₀(%d) - S₀(%d)|| / ||S₀(%d)|| = %.4f\n", i+1, i, i, rel_dist)
    end

    println("\n" * "-"^90)
    println("Skip-one distances (checking period-2 in S₀ itself):")
    for i in 1:length(S0_list)-2
        rel_dist = norm(S0_list[i+2] - S0_list[i]) / norm(S0_list[i])
        @printf("  ||S₀(%d) - S₀(%d)|| / ||S₀(%d)|| = %.4f\n", i+2, i, i, rel_dist)
    end

    println("\n" * "="^90)
    println("ANALYSIS:")
    println("="^90)

    # Check if skip-one distances are smaller than consecutive
    consec_avg = sum(norm(S0_list[i+1] - S0_list[i]) / norm(S0_list[i]) for i in 4:6) / 3
    skip_avg = sum(norm(S0_list[i+2] - S0_list[i]) / norm(S0_list[i]) for i in 4:5) / 2

    @printf("\nAverage consecutive distance (steps 4-7): %.4f\n", consec_avg)
    @printf("Average skip-one distance (steps 4-7): %.4f\n", skip_avg)

    if skip_avg < 0.5 * consec_avg
        println("\n→ S₀ OSCILLATES: skip-one distance << consecutive distance")
        println("  The period-2 is in S₀ itself, not just its eigenvectors")
    else
        println("\n→ S₀ does NOT oscillate: distances are similar")
        println("  The period-2 is only in the eigenvector subspace, not S₀")
    end
end

main()
