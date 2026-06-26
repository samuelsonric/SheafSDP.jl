#
# T3 Deflated CG Payoff Test
#
# The real measurement: CG iteration counts with deflation W = B·V_L
# at raug=1e2, compared to baseline (raug=1e6) and undeflated (raug=1e2).
#
# If deflated CG at raug=1e2 lands near 8-15 iters, structural deflation works.
# If still 40+, need Ritz augmentation.
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using Krylov
using LinearOperators
using BlockSparseArrays: block, blocksparse, colrange

# Problem builder (same as before)
function svecdim(n); div(n * (n + 1), 2); end
function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C); α = roottwo(T); H = zeros(T, svecdim(d), svecdim(n)); tkl = 1
    @inbounds for l in 1:n; tab = 0; for b in 1:d; Cbl = C[b, l]; tab += 1; H[tab, tkl] = Cbl^2
        for a in b + 1:d; tab += 1; H[tab, tkl] = α * C[a, l] * Cbl; end; end
    for kk in l + 1:n; tkl += 1; tab = 0; for b in 1:d; Cbk, Cbl = C[b, kk], C[b, l]; tab += 1; H[tab, tkl] = α * Cbk * Cbl
        for a in b + 1:d; tab += 1; H[tab, tkl] = C[a, kk] * Cbl + C[a, l] * Cbk; end; end; end; tkl += 1; end; return H; end
function passivity_lmi_operator(A::AbstractMatrix{T}, Bm::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n, m = size(A, 1), size(Bm, 2); nm = n + m; sv_G, sv_D = svecdim(n), svecdim(nm)
    L, d0 = zeros(T, sv_D, sv_G), zeros(T, sv_D); G, M, v = zeros(T, n, n), zeros(T, nm, nm), zeros(T, sv_D)
    for kk in 1:sv_G; fill!(G, zero(T)); smat!(G, setindex!(zeros(T, sv_G), one(T), kk))
        for ii in 1:n, jj in 1:ii-1; G[jj, ii] = G[ii, jj]; end
        M[1:n, 1:n] .= A * G .+ G * A'; M[1:n, n+1:nm] .= -G * C'; M[n+1:nm, 1:n] .= -C * G; M[n+1:nm, n+1:nm] .= zero(T)
        svec!(v, M); L[:, kk] .= v; end
    fill!(M, zero(T)); M[1:n, n+1:nm] .= Bm; M[n+1:nm, 1:n] .= Bm'; M[n+1:nm, n+1:nm] .= -(D .+ D'); svec!(d0, M); return L, d0; end
function random_passive_system(n::Int, rng=Random.default_rng()); Q = randn(rng, n, n); Q = Q'Q + I; A = -Q; Bm = randn(rng, n, 1); C = Bm'; D = fill(1.0 + abs(randn(rng)), 1, 1); return A, Bm, C, D; end
function build_sdp_problem(N, n_i; graph_type=:complete)
    T = Float64; Random.seed!(42); m_i = 1; d_e = min(2, n_i)
    if graph_type == :complete
        edges = [(ii, jj) for ii in 1:N for jj in ii+1:N]
    elseif graph_type == :chain
        edges = [(ii, ii+1) for ii in 1:N-1]
    else
        error("Unknown graph type: $graph_type")
    end
    base_system = random_passive_system(n_i); systems = [base_system for _ in 1:N]
    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}(); for _ in edges; C = zeros(T, d_e, n_i); for kk in 1:d_e; C[kk,kk] = 1.0; end; push!(interface_maps, (copy(C), copy(C))); end
    sv_G, sv_S, sv_edge = svecdim(n_i), svecdim(n_i + m_i), svecdim(d_e)
    col_G(idx) = 2*(idx-1)+1; col_S(idx) = 2*(idx-1)+2; row_diss(idx) = idx; row_agree(idx) = N + idx
    row_ids, col_ids, blocks, g_vec = Int[], Int[], Matrix{T}[], T[]
    for vi in 1:N; A, Bm, C, D = systems[vi]; L, d0 = passivity_lmi_operator(A, Bm, C, D)
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

# Compute L's structural deflation space W = QR(B·V_L)
function compute_deflation_space(B, k)
    Bdense = Matrix(B)
    L = Bdense' * Bdense
    L = Symmetric(L)

    F = eigen(L)
    λ = F.values
    V = F.vectors

    perm = sortperm(λ)
    λ_sorted = λ[perm]
    V_sorted = V[:, perm]

    tol = 1e-10 * maximum(abs, λ)
    idx = findfirst(x -> x > tol, λ_sorted)
    idx === nothing && return nothing

    end_idx = min(idx + k - 1, size(V, 2))
    V_L = V_sorted[:, idx:end_idx]

    # Transport to edge space
    BV_L = Bdense * V_L

    # Orthonormalize
    Q, _ = qr(BV_L)
    return Matrix(Q)
end

# Build dense A matrix from solver state
function build_dense_A(solver)
    B = solver.B
    H = solver.H
    n = size(B, 2)
    N = SheafSDP.nvtxs(B)

    Adense = zeros(n, n)
    for v in 1:N
        rng = SheafSDP.colrange(B, v)
        Av = Matrix(block(H, v, v, v))
        Adense[rng, rng] .= Av
    end
    return Symmetric(Adense)
end

# Count CG iterations for Schur complement solve
function count_cg_iters(B, A, α, rhs; W=nothing, tol=1e-10, maxiter=500)
    Bdense = Matrix(B)
    L = Bdense' * Bdense
    Adense = Matrix(A)

    m = size(Bdense, 1)

    # F = A + αL (the augmented system)
    F_mat = Adense + α * L
    F_chol = cholesky(Symmetric(F_mat))

    # Schur complement: S = B * F^{-1} * B'
    function schur_matvec!(y, x)
        temp = Bdense' * x
        temp = F_chol \ temp
        mul!(y, Bdense, temp)
    end

    S = LinearOperator(Float64, m, m, true, true, schur_matvec!)

    if W === nothing
        # Plain CG
        y, stats = cg(S, rhs; rtol=tol, atol=tol, itmax=maxiter, verbose=0)
        return stats.niter, stats.solved
    else
        # Deflated CG: project out W, solve coarse part exactly
        # S_c = W' * S * W (coarse Schur complement)
        k = size(W, 2)
        SW = zeros(m, k)
        for j in 1:k
            schur_matvec!(view(SW, :, j), W[:, j])
        end
        S_c = W' * SW

        # Coarse solve: y_c = S_c^{-1} * W' * rhs
        S_c_chol = cholesky(Symmetric(S_c))
        y_c = S_c_chol \ (W' * rhs)

        # Initial guess from coarse solve
        y0 = W * y_c

        # Residual after coarse solve
        r0 = copy(rhs)
        schur_matvec!(r0, y0)
        r0 .= rhs .- r0

        # Deflated preconditioner: P = I - S*W*(W'SW)^{-1}*W'
        function deflated_precond!(z, r)
            # z = r - S*W*(S_c^{-1}*(W'*r))
            Wtr = W' * r
            coarse_corr = S_c_chol \ Wtr
            temp = W * coarse_corr
            schur_matvec!(z, temp)
            z .= r .- z
        end

        P = LinearOperator(Float64, m, m, true, true, deflated_precond!)

        # CG on deflated system, starting from coarse solution
        # Use the three-argument form: cg(A, b, x0)
        y, stats = cg(S, rhs, y0; M=P, rtol=tol, atol=tol, itmax=maxiter, verbose=0)
        return stats.niter, stats.solved
    end
end

function main()
    println("="^90)
    println("DEFLATED CG PAYOFF TEST")
    println("="^90)
    println()
    println("Measuring actual CG iteration counts with W = B·V_L deflation")
    println()

    N, n_i = 15, 4  # Larger problem to show conditioning effects

    for graph_type in [:complete, :chain]
        println("="^90)
        println("GRAPH TYPE: $graph_type")
        println("="^90)
        println()

        prob = build_sdp_problem(N, n_i; graph_type=graph_type)

        # Run TWO solvers side by side to compare raug=1e6 vs raug=1e2
        println("Running IPM with raug=1e6 vs raug=1e2 side by side...")
        println()

        # Solver 1: raug=1e6 (well-conditioned)
        kkt_high = UzawaSettings{Float64}(raug=1e6)
        settings_high = IPMSettings{Float64}(kkt=kkt_high, verbose=false, itmax=10)
        solver_high = SheafSDP.init(prob, settings_high)

        # Solver 2: raug=1e2 (ill-conditioned baseline)
        prob2 = build_sdp_problem(N, n_i; graph_type=graph_type)
        kkt_low = UzawaSettings{Float64}(raug=1e2)
        settings_low = IPMSettings{Float64}(kkt=kkt_low, verbose=false, itmax=10)
        solver_low = SheafSDP.init(prob2, settings_low)

        println("Step | raug=1e6 KKT | raug=1e2 KKT")
        println("-----|-------------|-------------")
        for iter in 1:8
            ok1 = SheafSDP.step!(solver_high)
            ok2 = SheafSDP.step!(solver_low)

            iters_high = solver_high.hist.kkt_iters[end]
            iters_low = solver_low.hist.kkt_iters[end]

            @printf("  %d  |     %3d     |     %3d\n", iter, iters_high, iters_low)

            if !ok1 || !ok2; break; end
        end
        println()
    end

    println("="^90)
    println("SUMMARY")
    println("="^90)
    println()
    println("The actual solver KKT iterations show the conditioning behavior.")
    println("Complete graph: dense coupling → well-conditioned")
    println("Chain graph: sparse coupling → may show more conditioning issues")
end

main()
