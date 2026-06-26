#
# T3 Structural Subspace Test
#
# The definitive test: does L's low subspace (B·V_L) stay inside
# S₀'s bottom cluster at every step?
#
# L = B'B is fixed. If B·V_L stays inside the bottom-8 throughout,
# we have a static deflation space and never need to refresh.
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using BlockSparseArrays: block, blocksparse, colrange

# Problem builder
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

function compute_S0_bottom_k(B, A, k)
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
        end_idx = min(idx + k - 1, size(V, 2))
        V_k = V_sorted[:, idx:end_idx]
        Q, _ = qr(V_k); return Matrix(Q)
    catch; return nothing; end
end

# Compute L = B'B and its low eigenvectors, transported to edge space
function compute_structural_subspace(B, k)
    Bdense = Matrix(B)
    L = Bdense' * Bdense
    L = Symmetric(L)

    F = eigen(L)
    λ = F.values
    V = F.vectors

    # Sort by eigenvalue
    perm = sortperm(λ)
    λ_sorted = λ[perm]
    V_sorted = V[:, perm]

    # Skip kernel (eigenvalues ≈ 0)
    tol = 1e-10 * maximum(abs, λ)
    idx = findfirst(x -> x > tol, λ_sorted)
    if idx === nothing
        return nothing, nothing
    end

    # Take bottom-k non-kernel eigenvectors
    end_idx = min(idx + k - 1, size(V, 2))
    V_L = V_sorted[:, idx:end_idx]
    λ_L = λ_sorted[idx:end_idx]

    # Transport to edge space: B * V_L
    BV_L = Bdense * V_L

    # Orthonormalize
    Q, _ = qr(BV_L)
    return Matrix(Q), λ_L
end

function main()
    println("="^90)
    println("STRUCTURAL SUBSPACE TEST")
    println("="^90)
    println()
    println("Does L's low subspace (B·V_L) stay inside S₀'s bottom cluster at every step?")
    println("L = B'B is fixed. If B·V_L stays inside, we have a static deflation space.")
    println()

    N, n_i = 5, 4
    prob = build_sdp_problem(N, n_i)
    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=10)
    solver = SheafSDP.init(prob, settings)
    B = solver.B

    # Compute L's structural subspace (fixed, computed once)
    println("Computing L = B'B structural subspace...")
    W_L4, λ_L4 = compute_structural_subspace(B, 4)
    W_L8, λ_L8 = compute_structural_subspace(B, 8)

    println("L's bottom-4 eigenvalues: ", [@sprintf("%.4e", e) for e in λ_L4])
    println("L's bottom-8 eigenvalues: ", [@sprintf("%.4e", e) for e in λ_L8])
    println("W_L4 dimensions: ", size(W_L4))
    println("W_L8 dimensions: ", size(W_L8))
    println()

    # Run IPM and check alignment at each step
    println("="^90)
    println("Alignment of fixed B·V_L with S₀'s bottom cluster at each step:")
    println("="^90)
    println()

    results = []

    for iter in 1:10
        SheafSDP.step!(solver)
        gap = dot(solver.p, solver.d) / solver.ν

        # Get S₀'s bottom-8 at this step
        Q8 = compute_S0_bottom_k(B, solver.H, 8)
        if Q8 === nothing
            @printf("Step %d: could not compute S₀\n", iter)
            continue
        end

        # Alignment: min singular value of W_L' * Q8
        # This measures how much of the fixed W_L lies inside Q8's span
        M4 = W_L4' * Q8
        σ4 = svdvals(M4)
        min_σ4 = minimum(σ4)

        M8 = W_L8' * Q8
        σ8 = svdvals(M8)
        min_σ8 = minimum(σ8)

        push!(results, (iter=iter, gap=gap, min_σ4=min_σ4, min_σ8=min_σ8, σ4=σ4, σ8=σ8))

        @printf("Step %d: gap=%.2e\n", iter, gap)
        @printf("  B·V_L(4) vs Q8: min(σ)=%.4f, all σ=%s\n", min_σ4, [@sprintf("%.3f", s) for s in σ4])
        @printf("  B·V_L(8) vs Q8: min(σ)=%.4f, all σ=%s\n", min_σ8, [@sprintf("%.3f", s) for s in σ8[1:min(8,length(σ8))]])
        println()
    end

    println("="^90)
    println("SUMMARY: Fixed B·V_L vs S₀'s bottom-8 at steps 4,5,6,7,8")
    println("="^90)
    println()

    println("Step  |  B·V_L(4) min(σ)  |  B·V_L(8) min(σ)")
    println("------|-------------------|------------------")
    for r in results
        if r.iter >= 4 && r.iter <= 8
            @printf("  %d   |      %.4f       |      %.4f\n", r.iter, r.min_σ4, r.min_σ8)
        end
    end

    println()
    println("="^90)
    println("VERDICT")
    println("="^90)

    # Check if L's subspace stays inside S₀'s bottom cluster
    steps_to_check = filter(r -> r.iter >= 4 && r.iter <= 8, results)
    min_align_L4 = minimum(r.min_σ4 for r in steps_to_check)
    min_align_L8 = minimum(r.min_σ8 for r in steps_to_check)

    println()
    @printf("Minimum alignment of B·V_L(4) with S₀ bottom-8 across steps 4-8: %.4f\n", min_align_L4)
    @printf("Minimum alignment of B·V_L(8) with S₀ bottom-8 across steps 4-8: %.4f\n", min_align_L8)
    println()

    if min_align_L4 > 0.9
        println("→ B·V_L(4) STAYS INSIDE S₀'s bottom cluster throughout!")
        println("→ Static structural deflation: build W from L once, never refresh")
        println("→ This is the cheapest possible outcome (free lunch confirmed)")
    elseif min_align_L4 > 0.7
        println("→ B·V_L(4) MOSTLY stays inside (some drift)")
        println("→ May need occasional refresh at cardinality changes")
    else
        println("→ B·V_L(4) wanders out of S₀'s bottom cluster")
        println("→ Need gap-triggered refresh or recycling")
    end
end

main()
