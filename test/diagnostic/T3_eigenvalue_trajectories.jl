#
# T3 Eigenvalue Trajectories
#
# Track the full spectrum of S₀ across steps to see if there are
# eigenvalue crossings that could cause the period-2 pattern.
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

function compute_S0_spectrum(B, A)
    m, n = size(B); N = SheafSDP.nvtxs(B); Bdense, Adense = Matrix(B), zeros(n, n)
    for v in 1:N; rng = SheafSDP.colrange(B, v); Av = Matrix(block(A, v, v, v))
        any(isnan, Av) || any(isinf, Av) && return nothing, nothing; Adense[rng, rng] .= Av; end
    minimum(eigvals(Symmetric(Adense))) <= 0 && return nothing, nothing
    try; S0 = Bdense * (Adense \ Bdense'); S0 = Symmetric((S0 + S0') / 2)
        F = eigen(S0); return sort(F.values), F.vectors[:, sortperm(F.values)]
    catch; return nothing, nothing; end
end

function main()
    println("="^90)
    println("EIGENVALUE TRAJECTORIES")
    println("="^90)

    N, n_i = 5, 4
    prob = build_sdp_problem(N, n_i)
    kkt = UzawaSettings{Float64}(raug=1e6)
    settings = IPMSettings{Float64}(kkt=kkt, verbose=false, itmax=8)
    solver = SheafSDP.init(prob, settings)
    B = solver.B

    # Track eigenvectors for specific eigenvalue indices
    eigenvec_history = Dict{Int, Vector{Vector{Float64}}}()
    for i in 1:12; eigenvec_history[i] = Vector{Float64}[]; end

    println("\nTracking eigenvalues 1-12 across steps:")
    println("-"^90)

    for iter in 1:8
        SheafSDP.step!(solver)
        λ, V = compute_S0_spectrum(B, solver.H)
        if λ === nothing; continue; end

        # Skip kernel
        tol = 1e-10 * maximum(abs, λ)
        idx = findfirst(x -> x > tol, λ)
        if idx === nothing; continue; end

        # Store eigenvectors
        for i in 1:min(12, length(λ) - idx + 1)
            push!(eigenvec_history[i], V[:, idx + i - 1])
        end

        @printf("Step %d:", iter)
        for i in 1:min(8, length(λ) - idx + 1)
            @printf("  λ%d=%.2e", i, λ[idx + i - 1])
        end
        println()
    end

    println("\n" * "="^90)
    println("EIGENVECTOR TRACKING (by eigenvalue index)")
    println("="^90)
    println("\nDoes eigenvector i at step k align with eigenvector i at step k+1?")
    println("(This tests if the 'i-th smallest' eigenvector is stable)")
    println()

    for ev_idx in 1:4
        println("Eigenvector $ev_idx:")
        vecs = eigenvec_history[ev_idx]
        for i in 1:length(vecs)-1
            # Alignment of eigenvector ev_idx at step i with step i+1
            align = abs(dot(vecs[i], vecs[i+1]))
            @printf("  Step %d → %d: |v_i · v_{i+1}| = %.4f\n", i, i+1, align)
        end
        println()
    end

    println("="^90)
    println("CROSS-EIGENVECTOR TRACKING")
    println("="^90)
    println("\nDoes eigenvector 1 at step k become eigenvector 2,3,4 at step k+1?")
    println()

    for step in 4:6
        if step >= length(eigenvec_history[1]) || step+1 > length(eigenvec_history[1])
            continue
        end

        @printf("Step %d → %d:\n", step, step+1)
        for i in 1:4
            vi = eigenvec_history[i][step]
            @printf("  v%d(step%d) aligns with: ", i, step)
            for j in 1:4
                vj = eigenvec_history[j][step+1]
                align = abs(dot(vi, vj))
                @printf("v%d(%.2f) ", j, align)
            end
            println()
        end
        println()
    end
end

main()
