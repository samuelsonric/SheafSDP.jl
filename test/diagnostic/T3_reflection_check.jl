#
# T3 Reflection Check
#
# If the transformation between consecutive subspaces is a reflection (R² = I),
# that would explain the period-2 pattern.
#
# Test: Compute the transformation matrix T = Q4' * Q5 * Q5' * Q4
# If this is close to identity, the transformation is an involution (reflection-like).
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using BlockSparseArrays: block, blocksparse, colrange

# Same abbreviated problem builder
function svecdim(n); div(n * (n + 1), 2); end
function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C); α = roottwo(T); H = zeros(T, svecdim(d), svecdim(n)); tkl = 1
    @inbounds for l in 1:n; tab = 0; for b in 1:d; Cbl = C[b, l]; tab += 1; H[tab, tkl] = Cbl^2
        for a in b + 1:d; tab += 1; H[tab, tkl] = α * C[a, l] * Cbl; end; end
    for kk in l + 1:n; tkl += 1; tab = 0; for b in 1:d; Cbk, Cbl = C[b, kk], C[b, l]; tab += 1; H[tab, tkl] = α * Cbk * Cbl
        for a in b + 1:d; tab += 1; H[tab, tkl] = C[a, kk] * Cbl + C[a, l] * Cbk; end; end; end; tkl += 1; end; return H; end
function passivity_lmi_operator(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n, m = size(A, 1), size(B, 2); nm = n + m; sv_G, sv_D = svecdim(n), svecdim(nm)
    L, d0 = zeros(T, sv_D, sv_G), zeros(T, sv_D); G, M, v = zeros(T, n, n), zeros(T, nm, nm), zeros(T, sv_D)
    for kk in 1:sv_G; fill!(G, zero(T)); smat!(G, setindex!(zeros(T, sv_G), one(T), kk))
        for ii in 1:n, jj in 1:ii-1; G[jj, ii] = G[ii, jj]; end
        M[1:n, 1:n] .= A * G .+ G * A'; M[1:n, n+1:nm] .= -G * C'; M[n+1:nm, 1:n] .= -C * G; M[n+1:nm, n+1:nm] .= zero(T)
        svec!(v, M); L[:, kk] .= v; end
    fill!(M, zero(T)); M[1:n, n+1:nm] .= B; M[n+1:nm, 1:n] .= B'; M[n+1:nm, n+1:nm] .= -(D .+ D'); svec!(d0, M); return L, d0; end
function random_passive_system(n::Int, rng=Random.default_rng()); Q = randn(rng, n, n); Q = Q'Q + I; A = -Q; B = randn(rng, n, 1); C = B'; D = fill(1.0 + abs(randn(rng)), 1, 1); return A, B, C, D; end
function build_sdp_problem(N, n_i)
    T = Float64; Random.seed!(42); m_i = 1; d_e = min(2, n_i); edges = [(ii, jj) for ii in 1:N for jj in ii+1:N]
    base_system = random_passive_system(n_i); systems = [base_system for _ in 1:N]
    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}(); for _ in edges; C = zeros(T, d_e, n_i); for kk in 1:d_e; C[kk,kk] = 1.0; end; push!(interface_maps, (copy(C), copy(C))); end
    sv_G, sv_S, sv_edge = svecdim(n_i), svecdim(n_i + m_i), svecdim(d_e)
    col_G(idx) = 2*(idx-1)+1; col_S(idx) = 2*(idx-1)+2; row_diss(idx) = idx; row_agree(idx) = N + idx
    row_ids, col_ids, blocks, g_vec = Int[], Int[], Matrix{T}[], T[]
    for vi in 1:N; A, B, C, D = systems[vi]; L, d0 = passivity_lmi_operator(A, B, C, D)
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

function compute_S0_subspace(B, A, k)
    m, n = size(B); N = SheafSDP.nvtxs(B); Bdense, Adense = Matrix(B), zeros(n, n)
    for v in 1:N; rng = SheafSDP.colrange(B, v); Av = Matrix(block(A, v, v, v))
        any(isnan, Av) || any(isinf, Av) && return nothing; Adense[rng, rng] .= Av; end
    minimum(eigvals(Symmetric(Adense))) <= 0 && return nothing
    try
        S0 = Bdense * (Adense \ Bdense'); S0 = Symmetric((S0 + S0') / 2)
        F = eigen(S0); λ = F.values; V = F.vectors
        perm = sortperm(λ); λ_sorted = λ[perm]; V_sorted = V[:, perm]
        tol = 1e-10 * maximum(abs, λ); idx = findfirst(x -> x > tol, λ_sorted)
        idx === nothing && return nothing
        end_idx = min(idx + k - 1, size(V, 2)); V_k = V_sorted[:, idx:end_idx]
        Q, _ = qr(V_k); return Matrix(Q)
    catch; return nothing; end
end

function main()
    println("="^90)
    println("REFLECTION CHECK")
    println("="^90)

    N, n_i = 5, 4
    prob = build_sdp_problem(N, n_i)
    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=8)
    solver = SheafSDP.init(prob, settings)
    B = solver.B

    bases = Matrix{Float64}[]
    for iter in 1:8
        SheafSDP.step!(solver)
        Q = compute_S0_subspace(B, solver.H, 4)
        Q !== nothing && push!(bases, Q)
    end

    println("\nSubspace alignment matrix analysis (k=4):")
    println("-"^90)

    for i in 4:min(6, length(bases)-1)
        Qi = bases[i]
        Qj = bases[i+1]

        # The "transformation" in subspace coordinates
        M = Qi' * Qj

        println("\nStep $i → $(i+1):")
        println("  M = Q$i' * Q$(i+1):")
        for row in 1:4
            @printf("    [%.4f %.4f %.4f %.4f]\n", M[row, 1], M[row, 2], M[row, 3], M[row, 4])
        end

        # Check if M² ≈ I (involution)
        M2 = M * M
        println("  M² (should be ≈I if reflection):")
        for row in 1:4
            @printf("    [%.4f %.4f %.4f %.4f]\n", M2[row, 1], M2[row, 2], M2[row, 3], M2[row, 4])
        end

        # Eigenvalues of M
        eigM = eigvals(M)
        println("  Eigenvalues of M: ", [@sprintf("%.3f%+.3fi", real(e), imag(e)) for e in eigM])

        # Check if eigenvalues are on unit circle (orthogonal transformation)
        println("  |eigenvalues|: ", [@sprintf("%.4f", abs(e)) for e in eigM])
    end

    println("\n" * "="^90)
    println("CHECK: Is the pattern really period-2 in subspace basis?")
    println("="^90)

    if length(bases) >= 6
        Q4, Q5, Q6 = bases[4], bases[5], bases[6]

        # Q4 expressed in Q5 coordinates, then Q5→Q6 transformation applied
        M45 = Q4' * Q5
        M56 = Q5' * Q6

        # Composed transformation: Q4 → Q6 via Q5
        M46_composed = M45 * M56

        # Direct transformation Q4 → Q6
        M46_direct = Q4' * Q6

        println("\nM_{4→5} * M_{5→6} (composed):")
        for row in 1:4
            @printf("  [%.4f %.4f %.4f %.4f]\n",
                    M46_composed[row, 1], M46_composed[row, 2],
                    M46_composed[row, 3], M46_composed[row, 4])
        end

        println("\nM_{4→6} (direct):")
        for row in 1:4
            @printf("  [%.4f %.4f %.4f %.4f]\n",
                    M46_direct[row, 1], M46_direct[row, 2],
                    M46_direct[row, 3], M46_direct[row, 4])
        end

        println("\nDifference (should be near zero):")
        diff = M46_composed - M46_direct
        @printf("  ||composed - direct|| = %.6f\n", norm(diff))
    end
end

main()
