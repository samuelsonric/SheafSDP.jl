#
# T3 Eigenvector Stability Check
#
# Check if the eigen() call produces consistent eigenvector orderings
# within degenerate clusters across consecutive calls.
#
# If the eigenvectors flip ordering/sign between calls, that could
# create fake period-2 patterns even with stable H.
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block, blocksparse, colrange

# Same problem builder as before (abbreviated)
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
        for kk in l + 1:n
            tkl += 1; tab = 0
            for b in 1:d
                Cbk, Cbl = C[b, kk], C[b, l]; tab += 1; H[tab, tkl] = α * Cbk * Cbl
                for a in b + 1:d; tab += 1; H[tab, tkl] = C[a, kk] * Cbl + C[a, l] * Cbk; end
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
    for kk in 1:sv_G
        fill!(G, zero(T)); smat!(G, setindex!(zeros(T, sv_G), one(T), kk))
        for ii in 1:n, jj in 1:ii-1; G[jj, ii] = G[ii, jj]; end
        M[1:n, 1:n] .= A * G .+ G * A'; M[1:n, n+1:nm] .= -G * C'
        M[n+1:nm, 1:n] .= -C * G; M[n+1:nm, n+1:nm] .= zero(T)
        svec!(v, M); L[:, kk] .= v
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

function build_sdp_problem(N, n_i)
    T = Float64; Random.seed!(42); m_i = 1; d_e = min(2, n_i)
    edges = [(ii, jj) for ii in 1:N for jj in ii+1:N]  # complete graph
    base_system = random_passive_system(n_i); systems = [base_system for _ in 1:N]
    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for _ in edges
        C = zeros(T, d_e, n_i); for kk in 1:d_e; C[kk,kk] = 1.0; end
        push!(interface_maps, (copy(C), copy(C)))
    end
    sv_G, sv_S, sv_edge = svecdim(n_i), svecdim(n_i + m_i), svecdim(d_e)
    col_G(idx) = 2*(idx-1)+1; col_S(idx) = 2*(idx-1)+2
    row_diss(idx) = idx; row_agree(idx) = N + idx
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

function compute_S0(B, A)
    m, n = size(B); N = SheafSDP.nvtxs(B)
    Bdense, Adense = Matrix(B), zeros(n, n)
    for v in 1:N
        rng = SheafSDP.colrange(B, v); Av = Matrix(block(A, v, v, v))
        any(isnan, Av) || any(isinf, Av) && return nothing
        Adense[rng, rng] .= Av
    end
    minimum(eigvals(Symmetric(Adense))) <= 0 && return nothing
    try
        S0 = Bdense * (Adense \ Bdense')
        return Symmetric((S0 + S0') / 2)
    catch; return nothing; end
end

function main()
    println("="^90)
    println("EIGENVECTOR STABILITY CHECK")
    println("="^90)
    println()
    println("Testing if eigen() produces consistent results across multiple calls")
    println("on the SAME matrix (checking for numerical tie-breaking artifacts)")
    println()

    N, n_i = 5, 4
    prob = build_sdp_problem(N, n_i)

    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=6)
    solver = SheafSDP.init(prob, settings)
    B = solver.B

    # Run to step 4 where we see the period-2 pattern
    for iter in 1:4
        SheafSDP.step!(solver)
    end

    println("At step 4, calling eigen() multiple times on the SAME S₀ matrix:")
    println("-"^90)

    S0 = compute_S0(B, solver.H)
    if S0 === nothing
        println("Could not compute S₀")
        return
    end

    # Call eigen() multiple times on the exact same matrix
    results = []
    for trial in 1:5
        F = eigen(S0)
        λ = F.values
        V = F.vectors

        # Sort by eigenvalue
        perm = sortperm(λ)
        λ_sorted = λ[perm]
        V_sorted = V[:, perm]

        # Skip kernel, get bottom-4
        tol = 1e-10 * maximum(abs, λ)
        idx = findfirst(x -> x > tol, λ_sorted)
        V_k = V_sorted[:, idx:idx+3]

        # Orthonormalize
        Q, _ = qr(V_k)
        push!(results, Matrix(Q))

        @printf("Trial %d: λ₁-λ₄ = [%.6e, %.6e, %.6e, %.6e]\n",
                trial, λ_sorted[idx], λ_sorted[idx+1], λ_sorted[idx+2], λ_sorted[idx+3])
    end

    println("\nAlignment between consecutive eigen() calls on SAME matrix:")
    for i in 1:length(results)-1
        M = results[i]' * results[i+1]
        σ = svdvals(M)
        @printf("  Trial %d vs %d: σ = %s, min(σ) = %.6f\n",
                i, i+1, [@sprintf("%.4f", s) for s in σ], minimum(σ))
    end

    println("\n" * "="^90)
    println("Now checking: does H actually change between steps 4 and 5?")
    println("="^90)

    # Store S0 at step 4
    S0_step4 = copy(S0)

    # Take step 5
    SheafSDP.step!(solver)

    S0_step5 = compute_S0(B, solver.H)
    if S0_step5 === nothing
        println("Could not compute S₀ at step 5")
        return
    end

    # Check if S0 actually changed
    S0_diff = norm(S0_step5 - S0_step4) / norm(S0_step4)
    @printf("\n||S₀(step5) - S₀(step4)|| / ||S₀(step4)|| = %.6e\n", S0_diff)

    if S0_diff > 0.1
        println("→ S₀ changed significantly between steps")
    else
        println("→ S₀ is nearly identical (unexpected!)")
    end

    # Compare eigenvectors
    F4 = eigen(S0_step4)
    F5 = eigen(S0_step5)

    perm4 = sortperm(F4.values)
    perm5 = sortperm(F5.values)

    tol = 1e-10 * maximum(abs, F4.values)
    idx4 = findfirst(x -> x > tol, F4.values[perm4])
    idx5 = findfirst(x -> x > tol, F5.values[perm5])

    V4 = F4.vectors[:, perm4[idx4:idx4+3]]
    V5 = F5.vectors[:, perm5[idx5:idx5+3]]

    Q4, _ = qr(V4); Q4 = Matrix(Q4)
    Q5, _ = qr(V5); Q5 = Matrix(Q5)

    M = Q4' * Q5
    σ = svdvals(M)

    println("\nSubspace alignment between step 4 and step 5:")
    @printf("  σ = %s\n", [@sprintf("%.4f", s) for s in σ])
    @printf("  min(σ) = %.4f\n", minimum(σ))

    println("\nEigenvalue comparison:")
    @printf("  Step 4 λ₁-λ₄: [%.4e, %.4e, %.4e, %.4e]\n",
            F4.values[perm4[idx4]], F4.values[perm4[idx4+1]],
            F4.values[perm4[idx4+2]], F4.values[perm4[idx4+3]])
    @printf("  Step 5 λ₁-λ₄: [%.4e, %.4e, %.4e, %.4e]\n",
            F5.values[perm5[idx5]], F5.values[perm5[idx5+1]],
            F5.values[perm5[idx5+2]], F5.values[perm5[idx5+3]])
end

main()
