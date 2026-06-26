#
# T3 Subspace Check - k=4 (first cluster only)
#
# Verify that with k=4 (cutting at the μ₅/μ₄ gap), the subspace is stable.
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block, blocksparse, colrange

# [Same problem builder code]
function svecdim(n); div(n * (n + 1), 2); end

function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C); α = roottwo(T)
    H = zeros(T, svecdim(d), svecdim(n)); tkl = 1
    @inbounds for l in 1:n
        tab = 0
        for b in 1:d
            Cbl = C[b, l]; tab += 1; H[tab, tkl] = Cbl^2
            for a in b + 1:d; tab += 1; H[tab, tkl] = α * C[a, l] * Cbl; end
        end
        for k in l + 1:n
            tkl += 1; tab = 0
            for b in 1:d
                Cbk, Cbl = C[b, k], C[b, l]; tab += 1; H[tab, tkl] = α * Cbk * Cbl
                for a in b + 1:d
                    tab += 1; H[tab, tkl] = C[a, k] * Cbl + C[a, l] * Cbk
                end
            end
        end
        tkl += 1
    end
    return H
end

function passivity_lmi_operator(A::AbstractMatrix{T}, B_mat::AbstractMatrix{T},
                                 C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n, m = size(A, 1), size(B_mat, 2); nm = n + m
    sv_G, sv_D = svecdim(n), svecdim(nm)
    L, d0 = zeros(T, sv_D, sv_G), zeros(T, sv_D)
    G, M, v = zeros(T, n, n), zeros(T, nm, nm), zeros(T, sv_D)
    for k in 1:sv_G
        fill!(G, zero(T)); smat!(G, setindex!(zeros(T, sv_G), one(T), k))
        for i in 1:n, j in 1:i-1; G[j, i] = G[i, j]; end
        M[1:n, 1:n] .= A * G .+ G * A'; M[1:n, n+1:nm] .= -G * C'
        M[n+1:nm, 1:n] .= -C * G; M[n+1:nm, n+1:nm] .= zero(T)
        svec!(v, M); L[:, k] .= v
    end
    fill!(M, zero(T)); M[1:n, n+1:nm] .= B_mat; M[n+1:nm, 1:n] .= B_mat'
    M[n+1:nm, n+1:nm] .= -(D .+ D'); svec!(d0, M)
    return L, d0
end

function random_passive_system(n::Int, rng=Random.default_rng())
    Q = randn(rng, n, n); Q = Q'Q + I; A = -Q
    B_mat = randn(rng, n, 1); C = B_mat'; D = fill(1.0 + abs(randn(rng)), 1, 1)
    return A, B_mat, C, D
end

function build_sdp_problem(N, n_i; topology=:complete)
    T = Float64; Random.seed!(42); m_i = 1; d_e = min(2, n_i)
    edges = topology == :chain ? [(ii, ii+1) for ii in 1:N-1] : [(ii, jj) for ii in 1:N for jj in ii+1:N]
    base_system = random_passive_system(n_i); systems = [base_system for _ in 1:N]
    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for _ in edges
        C = zeros(T, d_e, n_i)
        for k in 1:d_e; C[k,k] = 1.0; end
        push!(interface_maps, (copy(C), copy(C)))
    end
    sv_G, sv_S, sv_edge = svecdim(n_i), svecdim(n_i + m_i), svecdim(d_e)
    col_G(idx) = 2*(idx-1)+1
    col_S(idx) = 2*(idx-1)+2
    row_diss(idx) = idx
    row_agree(idx) = N + idx
    row_ids, col_ids, blocks, g_vec = Int[], Int[], Matrix{T}[], T[]
    for vi in 1:N
        A, B_mat, C, D = systems[vi]; L, d0 = passivity_lmi_operator(A, B_mat, C, D)
        push!(row_ids, row_diss(vi)); push!(col_ids, col_S(vi)); push!(blocks, Matrix{T}(I, sv_S, sv_S))
        push!(row_ids, row_diss(vi)); push!(col_ids, col_G(vi)); push!(blocks, L); append!(g_vec, -d0)
    end
    for (e, (vi, vj)) in enumerate(edges)
        C_i, C_j = interface_maps[e]; K_i, K_j = skronr(C_i), skronr(C_j)
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(vi)); push!(blocks, K_i)
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(vj)); push!(blocks, -K_j)
        append!(g_vec, zeros(T, sv_edge))
    end
    B = blocksparse(row_ids, col_ids, blocks)
    c_vec = zeros(T, size(B, 2)); I_n = Matrix{T}(I, n_i, n_i); svec_I = zeros(T, sv_G); svec!(svec_I, I_n)
    for vi in 1:N; c_vec[colrange(B, col_G(vi))] .= svec_I; end
    Q = SheafSDP.allocblockdiag(B); fill!(Q, zero(T))
    cones = Vector{Cone}(undef, 2*N)
    for vi in 1:N; cones[col_G(vi)] = SemidefiniteCone(); cones[col_S(vi)] = SemidefiniteCone(); end
    return IPMProblem(c_vec, g_vec, B, Q, cones)
end

function compute_S0_eigen(B, A)
    m, n = size(B); N = SheafSDP.nvtxs(B)
    Bdense, Adense = Matrix(B), zeros(n, n)
    for v in 1:N
        rng = SheafSDP.colrange(B, v); Av = Matrix(block(A, v, v, v))
        any(isnan, Av) || any(isinf, Av) && return nothing, nothing
        Adense[rng, rng] .= Av
    end
    minimum(eigvals(Symmetric(Adense))) <= 0 && return nothing, nothing
    try
        S0 = Bdense * (Adense \ Bdense'); S0 = Symmetric((S0 + S0') / 2)
        F = eigen(S0); return F.values, F.vectors
    catch; return nothing, nothing; end
end

function get_bottom_k_basis(λ, V, k)
    perm = sortperm(λ); λ_sorted, V_sorted = λ[perm], V[:, perm]
    tol = 1e-10 * maximum(abs, λ); idx = findfirst(x -> x > tol, λ_sorted)
    idx === nothing && return nothing, nothing
    end_idx = min(idx + k - 1, size(V, 2)); V_k, λ_k = V_sorted[:, idx:end_idx], λ_sorted[idx:end_idx]
    Q, R = qr(V_k); return Matrix(Q), λ_k
end

function main()
    println("="^90)
    println("SUBSPACE STABILITY WITH k=4 (bottom cluster only)")
    println("="^90)
    println()
    println("Testing if cutting at the μ₅/μ₄ gap gives stable subspace")
    println()

    N, n_i = 5, 4
    prob = build_sdp_problem(N, n_i; topology=:complete)

    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=10)
    solver = SheafSDP.init(prob, settings)
    B = solver.B

    bases = Vector{Union{Nothing, Matrix{Float64}}}()

    for iter in 1:10
        ok = SheafSDP.step!(solver)
        gap = dot(solver.p, solver.d) / solver.ν
        λ, V = compute_S0_eigen(B, solver.H)
        if λ !== nothing
            Q, λ_k = get_bottom_k_basis(λ, V, 4)  # k=4 only
            push!(bases, Q)
            @printf("Step %d: gap = %.2e, λ₁-λ₄ = [%.2e, %.2e, %.2e, %.2e]\n",
                    iter, gap, λ_k[1], λ_k[2], λ_k[3], λ_k[4])
        else
            push!(bases, nothing)
        end
        ok || break
    end

    println("\n" * "-"^90)
    println("SUBSPACE ALIGNMENT WITH k=4:")
    println("-"^90)

    println("\nConsecutive steps (Q_k' Q_{k+1}):")
    for i in 1:length(bases)-1
        if bases[i] !== nothing && bases[i+1] !== nothing
            M = bases[i]' * bases[i+1]
            σ = svdvals(M)
            @printf("  Step %d → %d: σ = %s, min(σ) = %.4f\n",
                    i, i+1, [@sprintf("%.3f", s) for s in σ], minimum(σ))
        end
    end

    println("\nFrom step 4 reference:")
    if bases[4] !== nothing
        for i in 5:length(bases)
            if bases[i] !== nothing
                M = bases[4]' * bases[i]
                σ = svdvals(M)
                @printf("  Step 4 → %d: σ = %s, min(σ) = %.4f\n",
                        i, [@sprintf("%.3f", s) for s in σ], minimum(σ))
            end
        end
    end

    # Final verdict
    println("\n" * "="^90)
    if length(bases) >= 6 && bases[4] !== nothing && bases[5] !== nothing && bases[6] !== nothing
        align_45 = minimum(svdvals(bases[4]' * bases[5]))
        align_56 = minimum(svdvals(bases[5]' * bases[6]))
        avg = (align_45 + align_56) / 2

        if avg > 0.9
            println("VERDICT: k=4 subspace is STABLE (min align = $(round(avg, digits=3)))")
            println("→ Build deflation space once at step 4 with k=4, reuse through convergence")
        else
            println("VERDICT: k=4 subspace still DRIFTS (min align = $(round(avg, digits=3)))")
        end
    end
    println("="^90)
end

main()
