#
# T3 Grid Scaling Test
#
# The definitive test: how do KKT iterations scale with grid size?
# - Bare CG: establishes exponent p in iters ∝ N^p (expect p ≈ 0.5 for grid)
# - Deflated CG: does it flatten the curve or just shift it?
# - k scaling: does k needed to flatten grow with N (→ AMG) or saturate (→ deflation)?
#

using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using LinearAlgebra
using Printf
using Random
using Krylov
using LinearOperators
using BlockSparseArrays: block, blocksparse, colrange

# Problem builder
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

# Build 2D grid graph: vertices at (i,j), edges to 4-neighbors
function grid_edges(nx, ny)
    vertex_id(i, j) = (j - 1) * nx + i
    edges = Tuple{Int,Int}[]
    for j in 1:ny, i in 1:nx
        v = vertex_id(i, j)
        # Right neighbor
        if i < nx
            push!(edges, (v, vertex_id(i+1, j)))
        end
        # Up neighbor
        if j < ny
            push!(edges, (v, vertex_id(i, j+1)))
        end
    end
    return edges
end

function build_grid_sdp_problem(nx, ny, n_i)
    T = Float64; Random.seed!(42); m_i = 1; d_e = min(2, n_i)
    N = nx * ny
    edges = grid_edges(nx, ny)

    base_system = random_passive_system(n_i)
    systems = [base_system for _ in 1:N]
    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for _ in edges
        C = zeros(T, d_e, n_i)
        for kk in 1:d_e; C[kk,kk] = 1.0; end
        push!(interface_maps, (copy(C), copy(C)))
    end

    sv_G, sv_S, sv_edge = svecdim(n_i), svecdim(n_i + m_i), svecdim(d_e)
    col_G(idx) = 2*(idx-1)+1; col_S(idx) = 2*(idx-1)+2
    row_diss(idx) = idx; row_agree(idx) = N + idx
    row_ids, col_ids, blocks, g_vec = Int[], Int[], Matrix{T}[], T[]

    for vi in 1:N
        A, Bm, C, D = systems[vi]
        L, d0 = passivity_lmi_operator(A, Bm, C, D)
        push!(row_ids, row_diss(vi)); push!(col_ids, col_S(vi)); push!(blocks, Matrix{T}(I, sv_S, sv_S))
        push!(row_ids, row_diss(vi)); push!(col_ids, col_G(vi)); push!(blocks, L)
        append!(g_vec, -d0)
    end

    for (e, (vi, vj)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        K_i, K_j = skronr(C_i), skronr(C_j)
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(vi)); push!(blocks, K_i)
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(vj)); push!(blocks, -K_j)
        append!(g_vec, zeros(T, sv_edge))
    end

    B = blocksparse(row_ids, col_ids, blocks)
    c_vec = zeros(T, size(B, 2))
    I_n = Matrix{T}(I, n_i, n_i)
    svec_I = zeros(T, sv_G)
    svec!(svec_I, I_n)
    for vi in 1:N; c_vec[colrange(B, col_G(vi))] .= svec_I; end

    Q = SheafSDP.allocblockdiag(B); fill!(Q, zero(T))
    cones = Vector{Cone}(undef, 2*N)
    for vi in 1:N; cones[col_G(vi)] = SemidefiniteCone(); cones[col_S(vi)] = SemidefiniteCone(); end

    return IPMProblem(c_vec, g_vec, B, Q, cones)
end

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
    idx === nothing && return nothing, nothing

    end_idx = min(idx + k - 1, size(V, 2))
    V_L = V_sorted[:, idx:end_idx]
    λ_L = λ_sorted[idx:end_idx]

    # Transport to edge space
    BV_L = Bdense * V_L

    # Orthonormalize
    Q, _ = qr(BV_L)
    return Matrix(Q), λ_L
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
        k = size(W, 2)
        SW = zeros(m, k)
        for j in 1:k
            schur_matvec!(view(SW, :, j), W[:, j])
        end
        S_c = W' * SW

        # Coarse solve
        S_c_chol = cholesky(Symmetric(S_c))
        y_c = S_c_chol \ (W' * rhs)
        y0 = W * y_c

        # Residual after coarse solve
        r0 = copy(rhs)
        schur_matvec!(r0, y0)
        r0 .= rhs .- r0

        # Deflated preconditioner
        function deflated_precond!(z, r)
            Wtr = W' * r
            coarse_corr = S_c_chol \ Wtr
            temp = W * coarse_corr
            schur_matvec!(z, temp)
            z .= r .- z
        end

        P = LinearOperator(Float64, m, m, true, true, deflated_precond!)

        y, stats = cg(S, rhs, y0; M=P, rtol=tol, atol=tol, itmax=maxiter, verbose=0)
        return stats.niter, stats.solved
    end
end

function main()
    println("="^90)
    println("GRID SCALING TEST")
    println("="^90)
    println()
    println("Testing KKT iteration scaling with grid size to determine:")
    println("  1. Baseline exponent p in iters ∝ N^p")
    println("  2. Does deflation flatten or just shift the curve?")
    println("  3. Does k needed to flatten grow with N (→ AMG) or saturate (→ deflation)?")
    println()

    n_i = 4  # Fixed node dimension
    raug = 1e2  # Fixed augmentation parameter
    grid_sizes = [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 16)]  # Extend to see scaling

    # Results storage
    results = []

    for (nx, ny) in grid_sizes
        N = nx * ny
        n_edges = 2*nx*ny - nx - ny  # 2D grid edge count

        println("-"^90)
        @printf("Grid %d×%d (N=%d vertices, %d edges)\n", nx, ny, N, n_edges)
        println("-"^90)

        # Build problem
        prob = build_grid_sdp_problem(nx, ny, n_i)

        # Run solver with raug=1e2
        kkt_settings = UzawaSettings{Float64}(raug=raug)
        settings = IPMSettings{Float64}(kkt=kkt_settings, verbose=false, itmax=8)
        solver = SheafSDP.init(prob, settings)

        # Collect KKT iters per step
        kkt_iters_per_step = Int[]
        for iter in 1:6
            ok = SheafSDP.step!(solver)
            push!(kkt_iters_per_step, solver.hist.kkt_iters[end])
            if !ok; break; end
        end

        # Average over steps 2-4 (middle of solve)
        avg_iters = length(kkt_iters_per_step) >= 4 ? mean(kkt_iters_per_step[2:4]) : mean(kkt_iters_per_step)
        max_iters = maximum(kkt_iters_per_step)

        @printf("  Bare CG: iters per step = %s\n", kkt_iters_per_step)
        @printf("  Bare CG: avg(steps 2-4) = %.1f, max = %d\n", avg_iters, max_iters)

        # Now test deflation manually on step-3 system
        # Re-run to step 3 and extract system
        prob2 = build_grid_sdp_problem(nx, ny, n_i)
        kkt2 = UzawaSettings{Float64}(raug=1e6)
        settings2 = IPMSettings{Float64}(kkt=kkt2, verbose=false, itmax=8)
        solver2 = SheafSDP.init(prob2, settings2)

        for _ in 1:3
            SheafSDP.step!(solver2)
        end

        B = solver2.B
        A = build_dense_A(solver2)
        m = size(B, 1)

        # Get α scaling
        α = raug * norm(A) / norm(Matrix(B))^2

        # Random RHS
        Random.seed!(42)
        rhs = randn(m)
        rhs = rhs / norm(rhs)

        # Test different deflation sizes
        k_values = [4, 8, 16, 32, min(64, div(N, 2))]
        deflated_results = []

        for k in k_values
            W, λ_L = compute_deflation_space(B, k)
            if W === nothing
                push!(deflated_results, (k=k, iters=-1))
                continue
            end

            iters, solved = count_cg_iters(B, A, α, rhs; W=W)
            push!(deflated_results, (k=k, iters=iters, actual_k=size(W, 2)))
            @printf("  Deflated k=%d (actual %d): %d iters\n", k, size(W, 2), iters)
        end

        # Also bare CG on this specific system for comparison
        iters_bare, _ = count_cg_iters(B, A, α, rhs)
        @printf("  Bare CG (simulated): %d iters\n", iters_bare)

        push!(results, (
            nx=nx, ny=ny, N=N, n_edges=n_edges,
            kkt_iters=kkt_iters_per_step,
            avg_iters=avg_iters, max_iters=max_iters,
            bare_sim=iters_bare,
            deflated=deflated_results
        ))
        println()
    end

    # Summary table
    println("="^90)
    println("SCALING SUMMARY")
    println("="^90)
    println()

    println("Grid    |  N   | Bare(avg) | Bare(sim) | k=4  | k=8  | k=16 | k=32 | k=64")
    println("--------|------|-----------|-----------|------|------|------|------|------")

    for r in results
        k4 = findfirst(d -> d.k == 4, r.deflated)
        k8 = findfirst(d -> d.k == 8, r.deflated)
        k16 = findfirst(d -> d.k == 16, r.deflated)
        k32 = findfirst(d -> d.k == 32, r.deflated)
        k64 = findfirst(d -> d.k == 64 || d.k == div(r.N, 2), r.deflated)

        k4_str = k4 !== nothing ? @sprintf("%4d", r.deflated[k4].iters) : "  - "
        k8_str = k8 !== nothing ? @sprintf("%4d", r.deflated[k8].iters) : "  - "
        k16_str = k16 !== nothing ? @sprintf("%4d", r.deflated[k16].iters) : "  - "
        k32_str = k32 !== nothing ? @sprintf("%4d", r.deflated[k32].iters) : "  - "
        k64_str = k64 !== nothing ? @sprintf("%4d", r.deflated[k64].iters) : "  - "

        @printf("%2d×%-2d   | %4d |   %5.1f   |    %4d   | %s | %s | %s | %s | %s\n",
                r.nx, r.ny, r.N, r.avg_iters, r.bare_sim, k4_str, k8_str, k16_str, k32_str, k64_str)
    end

    println()
    println("="^90)
    println("EXPONENT ANALYSIS")
    println("="^90)

    # Fit log-log slope for bare CG and deflated cases
    function fit_exponent(Ns, iters)
        log_N = [log(N) for N in Ns]
        log_iters = [log(i) for i in iters]
        n_pts = length(log_N)
        sum_x = sum(log_N)
        sum_y = sum(log_iters)
        sum_xy = sum(log_N .* log_iters)
        sum_x2 = sum(log_N .^ 2)
        return (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x^2)
    end

    if length(results) >= 3
        Ns = [r.N for r in results]

        # Bare CG exponent
        bare_iters = [r.bare_sim for r in results]
        bare_exp = fit_exponent(Ns, bare_iters)
        @printf("\nBare CG scaling: iters ∝ N^%.2f\n", bare_exp)

        # Deflated exponents for each k
        for k in [4, 8, 16, 32, 64]
            defl_iters = Int[]
            defl_Ns = Int[]
            for r in results
                idx = findfirst(d -> d.k == k, r.deflated)
                if idx !== nothing && r.deflated[idx].iters > 0
                    push!(defl_iters, r.deflated[idx].iters)
                    push!(defl_Ns, r.N)
                end
            end
            if length(defl_iters) >= 3
                defl_exp = fit_exponent(defl_Ns, defl_iters)
                @printf("Deflated k=%d scaling: iters ∝ N^%.2f\n", k, defl_exp)
            end
        end

        println()
        if bare_exp > 0.4
            println("→ Significant scaling with N (expected p ≈ 0.5 for grid Laplacian)")
        else
            println("→ Weak scaling - deflation may not be needed")
        end
    end

    println()
    println("="^90)
    println("VERDICT")
    println("="^90)
    println()
    println("Check the table above:")
    println("  • If deflated iters stay roughly constant as N grows → deflation flattens")
    println("  • If deflated iters grow with N but slower → deflation shifts curve")
    println("  • If k needed to flatten grows with N → AMG is the right tool")
    println("  • If fixed k saturates the benefit → static deflation works")
end

using Statistics: mean

main()
