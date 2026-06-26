#
# T3 Subspace Check - Bug Hunt
#
# Verify that subspace alignment is computed correctly:
# - Explicit QR orthonormalization
# - SVD of Q1' * Q2
# - Minimum singular value = cos(largest principal angle)
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using CommonSolve: solve
using BlockSparseArrays: block, blocksparse, colrange

#=============================================================================
   SDP Problem Builder (same as before)
=============================================================================#

function svecdim(n)
    return div(n * (n + 1), 2)
end

function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C)
    α = roottwo(T)
    H = zeros(T, svecdim(d), svecdim(n))
    tkl = 1
    @inbounds for l in 1:n
        tab = 0
        for b in 1:d
            Cbl = C[b, l]
            tab += 1
            H[tab, tkl] = Cbl^2
            for a in b + 1:d
                tab += 1
                H[tab, tkl] = α * C[a, l] * Cbl
            end
        end
        for k in l + 1:n
            tkl += 1
            tab = 0
            for b in 1:d
                Cbk = C[b, k]
                Cbl = C[b, l]
                tab += 1
                H[tab, tkl] = α * Cbk * Cbl
                for a in b + 1:d
                    Cak = C[a, k]
                    Cal = C[a, l]
                    tab += 1
                    H[tab, tkl] = Cak * Cbl + Cal * Cbk
                end
            end
        end
        tkl += 1
    end
    return H
end

function passivity_lmi_operator(A::AbstractMatrix{T}, B_mat::AbstractMatrix{T},
                                 C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    m = size(B_mat, 2)
    nm = n + m
    sv_G = svecdim(n)
    sv_D = svecdim(nm)
    L = zeros(T, sv_D, sv_G)
    d0 = zeros(T, sv_D)
    G = zeros(T, n, n)
    M = zeros(T, nm, nm)
    v = zeros(T, sv_D)
    for k in 1:sv_G
        fill!(G, zero(T))
        smat!(G, setindex!(zeros(T, sv_G), one(T), k))
        for i in 1:n, j in 1:i-1
            G[j, i] = G[i, j]
        end
        M[1:n, 1:n] .= A * G .+ G * A'
        M[1:n, n+1:nm] .= -G * C'
        M[n+1:nm, 1:n] .= -C * G
        M[n+1:nm, n+1:nm] .= zero(T)
        svec!(v, M)
        L[:, k] .= v
    end
    fill!(M, zero(T))
    M[1:n, n+1:nm] .= B_mat
    M[n+1:nm, 1:n] .= B_mat'
    M[n+1:nm, n+1:nm] .= -(D .+ D')
    svec!(d0, M)
    return L, d0
end

function random_passive_system(n::Int, rng=Random.default_rng())
    Q = randn(rng, n, n)
    Q = Q'Q + I
    A = -Q
    B_mat = randn(rng, n, 1)
    C = B_mat'
    D = fill(1.0 + abs(randn(rng)), 1, 1)
    return A, B_mat, C, D
end

function build_sdp_problem(N, n_i; topology=:complete)
    T = Float64
    Random.seed!(42)
    m_i = 1
    d_e = min(2, n_i)

    if topology == :chain
        edges = [(i, i+1) for i in 1:N-1]
    else
        edges = [(i, j) for i in 1:N for j in i+1:N]
    end
    n_edges = length(edges)

    base_system = random_passive_system(n_i)
    systems = [base_system for _ in 1:N]

    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for _ in 1:n_edges
        C = zeros(T, d_e, n_i)
        for k in 1:d_e
            C[k, k] = 1.0
        end
        push!(interface_maps, (copy(C), copy(C)))
    end

    sv_G = svecdim(n_i)
    sv_S = svecdim(n_i + m_i)
    sv_edge = svecdim(d_e)

    col_G(i) = 2*(i-1) + 1
    col_S(i) = 2*(i-1) + 2
    row_diss(i) = i
    row_agree(e) = N + e

    n_col_blocks = 2 * N

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]
    g_vec = T[]

    for i in 1:N
        A, B_mat, C, D = systems[i]
        L, d0 = passivity_lmi_operator(A, B_mat, C, D)
        push!(row_ids, row_diss(i))
        push!(col_ids, col_S(i))
        push!(blocks, Matrix{T}(I, sv_S, sv_S))
        push!(row_ids, row_diss(i))
        push!(col_ids, col_G(i))
        push!(blocks, L)
        append!(g_vec, -d0)
    end

    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        K_i = skronr(C_i)
        K_j = skronr(C_j)
        push!(row_ids, row_agree(e))
        push!(col_ids, col_G(i))
        push!(blocks, K_i)
        push!(row_ids, row_agree(e))
        push!(col_ids, col_G(j))
        push!(blocks, -K_j)
        append!(g_vec, zeros(T, sv_edge))
    end

    B = blocksparse(row_ids, col_ids, blocks)
    g = g_vec

    c_vec = zeros(T, size(B, 2))
    I_n = Matrix{T}(I, n_i, n_i)
    svec_I = zeros(T, sv_G)
    svec!(svec_I, I_n)
    for i in 1:N
        c_vec[colrange(B, col_G(i))] .= svec_I
    end

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, zero(T))

    cones = Vector{Cone}(undef, n_col_blocks)
    for i in 1:N
        cones[col_G(i)] = SemidefiniteCone()
        cones[col_S(i)] = SemidefiniteCone()
    end

    return IPMProblem(c_vec, g, B, Q, cones), n_i, m_i
end

# Compute S₀ and return its eigendecomposition
function compute_S0_eigen(B, A)
    m, n = size(B)
    N = SheafSDP.nvtxs(B)

    Bdense = Matrix(B)
    Adense = zeros(n, n)

    for v in 1:N
        rng = SheafSDP.colrange(B, v)
        Av = Matrix(block(A, v, v, v))
        if any(isnan, Av) || any(isinf, Av)
            return nothing, nothing
        end
        Adense[rng, rng] .= Av
    end

    eigA = eigvals(Symmetric(Adense))
    if minimum(eigA) <= 0
        return nothing, nothing
    end

    try
        S0 = Bdense * (Adense \ Bdense')
        S0 = Symmetric((S0 + S0') / 2)
        F = eigen(S0)
        return F.values, F.vectors
    catch
        return nothing, nothing
    end
end

# Get orthonormal basis for bottom-k subspace via explicit QR
function get_bottom_k_basis(λ, V, k)
    # Sort by eigenvalue
    perm = sortperm(λ)
    λ_sorted = λ[perm]
    V_sorted = V[:, perm]

    # Skip kernel
    tol = 1e-10 * maximum(abs, λ)
    idx = findfirst(x -> x > tol, λ_sorted)
    if idx === nothing
        return nothing, nothing
    end

    end_idx = min(idx + k - 1, size(V, 2))
    V_k = V_sorted[:, idx:end_idx]
    λ_k = λ_sorted[idx:end_idx]

    # Explicit QR to get orthonormal basis
    Q, R = qr(V_k)
    return Matrix(Q), λ_k
end

# Correct subspace alignment: min singular value of Q1' * Q2
function subspace_angle(Q1, Q2)
    if Q1 === nothing || Q2 === nothing
        return NaN, nothing
    end

    k = min(size(Q1, 2), size(Q2, 2))
    Q1 = Q1[:, 1:k]
    Q2 = Q2[:, 1:k]

    M = Q1' * Q2
    σ = svdvals(M)

    return minimum(σ), σ
end

function run_check(prob, name; k=8, max_iters=10)
    println("\n" * "="^90)
    println("Subspace Check: $name (k=$k)")
    println("="^90)

    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=max_iters)

    solver = SheafSDP.init(prob, settings)
    B = solver.B

    # Store orthonormal bases for each step
    bases = Vector{Union{Nothing, Matrix{Float64}}}()
    eigenvalues = Vector{Union{Nothing, Vector{Float64}}}()

    for iter in 1:max_iters
        ok = SheafSDP.step!(solver)
        gap = dot(solver.p, solver.d) / solver.ν

        λ, V = compute_S0_eigen(B, solver.H)
        if λ !== nothing
            Q, λ_k = get_bottom_k_basis(λ, V, k)
            push!(bases, Q)
            push!(eigenvalues, λ_k)

            @printf("\nStep %d: gap = %.2e\n", iter, gap)

            # Verify Q is orthonormal
            if Q !== nothing
                ortho_err = norm(Q'Q - I)
                @printf("  Q orthonormality error: %.2e\n", ortho_err)
                @printf("  Bottom-%d eigenvalues: %s\n", k,
                        [@sprintf("%.2e", e) for e in λ_k[1:min(4,length(λ_k))]])
            end
        else
            push!(bases, nothing)
            push!(eigenvalues, nothing)
        end

        if !ok
            break
        end
    end

    # Detailed analysis of steps 4, 5, 6
    println("\n" * "-"^90)
    println("DETAILED SUBSPACE ANALYSIS: Steps 4, 5, 6")
    println("-"^90)

    if length(bases) >= 6
        Q4, Q5, Q6 = bases[4], bases[5], bases[6]

        if Q4 !== nothing && Q5 !== nothing && Q6 !== nothing
            println("\nQ4 dimensions: ", size(Q4))
            println("Q5 dimensions: ", size(Q5))
            println("Q6 dimensions: ", size(Q6))

            # Q4' * Q5
            M45 = Q4' * Q5
            σ45 = svdvals(M45)
            println("\nQ4' * Q5:")
            println("  All singular values: ", [@sprintf("%.4f", s) for s in σ45])
            println("  min(σ) = ", @sprintf("%.4f", minimum(σ45)))

            # Q5' * Q6
            M56 = Q5' * Q6
            σ56 = svdvals(M56)
            println("\nQ5' * Q6:")
            println("  All singular values: ", [@sprintf("%.4f", s) for s in σ56])
            println("  min(σ) = ", @sprintf("%.4f", minimum(σ56)))

            # Q4' * Q6
            M46 = Q4' * Q6
            σ46 = svdvals(M46)
            println("\nQ4' * Q6:")
            println("  All singular values: ", [@sprintf("%.4f", s) for s in σ46])
            println("  min(σ) = ", @sprintf("%.4f", minimum(σ46)))

            # Sanity check: Q4' * Q4 should be identity
            M44 = Q4' * Q4
            σ44 = svdvals(M44)
            println("\nQ4' * Q4 (should be all 1s):")
            println("  All singular values: ", [@sprintf("%.4f", s) for s in σ44])

            # Check if the raw eigenvectors (before QR) are orthonormal
            println("\n" * "-"^50)
            println("RAW EIGENVECTOR CHECK (before QR):")

            λ4, V4 = compute_S0_eigen(B, solver.H)  # This is stale, need to recompute
        end
    end

    # Summary table
    println("\n" * "-"^90)
    println("CONSECUTIVE STEP ALIGNMENTS (min singular value of Q_k' Q_{k+1}):")
    println("-"^90)

    for i in 1:length(bases)-1
        align, σ = subspace_angle(bases[i], bases[i+1])
        @printf("  Step %d → %d: min(σ) = %.4f\n", i, i+1, align)
    end

    println("\n" * "-"^90)
    println("SKIP-ONE ALIGNMENTS (Q_k' Q_{k+2}):")
    println("-"^90)

    for i in 1:length(bases)-2
        align, σ = subspace_angle(bases[i], bases[i+2])
        @printf("  Step %d → %d: min(σ) = %.4f\n", i, i+2, align)
    end
end

function main()
    println("="^90)
    println("T3 SUBSPACE CHECK - BUG HUNT")
    println("="^90)
    println()
    println("Verifying subspace alignment computation is correct.")
    println("Using explicit QR orthonormalization and checking all singular values.")
    println()

    N = 5
    n_i = 4

    # Complete topology - k=8 (full merged cluster at step 4)
    prob, n_i_out, m_i = build_sdp_problem(N, n_i; topology=:complete)
    run_check(prob, "SDP Complete"; k=8, max_iters=8)

    # Chain topology
    prob, n_i_out, m_i = build_sdp_problem(N, n_i; topology=:chain)
    run_check(prob, "SDP Chain"; k=8, max_iters=8)

    println("\n" * "="^90)
    println("CHECK COMPLETE")
    println("="^90)
end

main()
