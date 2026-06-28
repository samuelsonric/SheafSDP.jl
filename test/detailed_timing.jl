using AppleAccelerate
using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse, colrange, rowrange
using Printf
using Statistics

Random.seed!(42)

svecdim(n) = div(n * (n + 1), 2)

function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C)
    α = roottwo(T)
    H = zeros(T, svecdim(d), svecdim(n))
    tkl = 1
    @inbounds for l in 1:n
        tab = 0
        for b in 1:d
            Cbl = C[b, l]
            tab += 1; H[tab, tkl] = Cbl^2
            for a in b + 1:d
                tab += 1; H[tab, tkl] = α * C[a, l] * Cbl
            end
        end
        for k in l + 1:n
            tkl += 1; tab = 0
            for b in 1:d
                Cbk, Cbl = C[b, k], C[b, l]
                tab += 1; H[tab, tkl] = α * Cbk * Cbl
                for a in b + 1:d
                    tab += 1; H[tab, tkl] = C[a, k] * Cbl + C[a, l] * Cbk
                end
            end
        end
        tkl += 1
    end
    return H
end

function l2gain_lmi_operator(A, B, C, D)
    T = Float64
    n = size(A, 1)
    m = size(B, 2)
    nm = n + m
    sv_P = svecdim(n)
    sv_D = svecdim(nm)

    L = zeros(T, sv_D, sv_P)
    d0 = zeros(T, sv_D)
    P = zeros(T, n, n)
    M = zeros(T, nm, nm)
    v = zeros(T, sv_D)

    for k in 1:sv_P
        fill!(P, zero(T))
        smat!(P, setindex!(zeros(T, sv_P), one(T), k))
        for ii in 1:n, jj in 1:ii-1
            P[jj, ii] = P[ii, jj]
        end
        M[1:n, 1:n] .= A' * P .+ P * A
        M[1:n, n+1:nm] .= P * B
        M[n+1:nm, 1:n] .= B' * P
        M[n+1:nm, n+1:nm] .= zero(T)
        svec!(v, M)
        L[:, k] .= v
    end

    fill!(M, zero(T))
    M[1:n, 1:n] .= C' * C
    M[1:n, n+1:nm] .= C' * D
    M[n+1:nm, 1:n] .= D' * C
    M[n+1:nm, n+1:nm] .= D' * D
    svec!(d0, M)

    return L, d0
end

function random_l2gain_system(n, m, p)
    Q = randn(n, n)
    A = -Q'Q - 10.0*I
    B = 0.05 * randn(n, m)
    C = 0.05 * randn(p, n)
    D = 0.01 * randn(p, m)
    return A, B, C, D
end

function build_problem(N, n_i)
    m_i, p_i, d_e = 1, 1, 10
    edges = [(ii, ii+1) for ii in 1:N-1]
    n_edges = length(edges)

    systems = [random_l2gain_system(n_i, m_i, p_i) for _ in 1:N]

    interface_maps = Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}()
    for _ in 1:n_edges
        C = zeros(Float64, d_e, n_i)
        for k in 1:d_e
            C[k, k] = 1.0
        end
        push!(interface_maps, (copy(C), copy(C)))
    end

    sv_P = svecdim(n_i)
    sv_S = svecdim(n_i + m_i)
    sv_edge = svecdim(d_e)

    col_P(ii) = 2*(ii-1) + 1
    col_S(ii) = 2*(ii-1) + 2
    col_μ = 2*N + 1

    row_lmi(ii) = ii
    row_agree(ee) = N + ee

    row_ids, col_ids, blocks = Int[], Int[], Matrix{Float64}[]
    g_vec = Float64[]

    E_μ = zeros(Float64, n_i + m_i, n_i + m_i)
    for k in n_i+1:n_i+m_i
        E_μ[k, k] = 1.0
    end
    svec_E_μ = zeros(Float64, sv_S)
    svec!(svec_E_μ, E_μ)

    for ii in 1:N
        A, B, C, D = systems[ii]
        L, d0 = l2gain_lmi_operator(A, B, C, D)

        push!(row_ids, row_lmi(ii)); push!(col_ids, col_S(ii)); push!(blocks, Matrix{Float64}(I, sv_S, sv_S))
        push!(row_ids, row_lmi(ii)); push!(col_ids, col_P(ii)); push!(blocks, L)
        push!(row_ids, row_lmi(ii)); push!(col_ids, col_μ); push!(blocks, reshape(-svec_E_μ, sv_S, 1))
        append!(g_vec, -d0)
    end

    for (ee, (ii, jj)) in enumerate(edges)
        C_i, C_j = interface_maps[ee]
        push!(row_ids, row_agree(ee)); push!(col_ids, col_P(ii)); push!(blocks, skronr(C_i))
        push!(row_ids, row_agree(ee)); push!(col_ids, col_P(jj)); push!(blocks, -skronr(C_j))
        append!(g_vec, zeros(Float64, sv_edge))
    end

    B = blocksparse(row_ids, col_ids, blocks)

    c_vec = zeros(Float64, size(B, 2))
    c_vec[colrange(B, col_μ)] .= 1.0

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, zero(Float64))

    cones = Vector{SheafSDP.AbstractCone}(undef, 2N + 1)
    for ii in 1:N
        cones[col_P(ii)] = SheafSDP.SemidefiniteCone()
        cones[col_S(ii)] = SheafSDP.SemidefiniteCone()
    end
    cones[col_μ] = SheafSDP.PositiveCone()

    return SheafSDP.IPMProblem(Q, B, c_vec, g_vec, cones)
end

# Problem dimensions
N, n_i = 100, 16
prob = build_problem(N, n_i)

# Show problem dimensions
println("Problem dimensions (N=$N, n_i=$n_i):")
n_sdp_cones = 2 * N
sdp_size = n_i  # Each SDP cone is n_i × n_i
sv_P = div(n_i * (n_i + 1), 2)
sv_S = div((n_i + 1) * (n_i + 2), 2)
println("  $n_sdp_cones SDP cones of size $n_i×$n_i (svec dim: $sv_P/$sv_S)")
println("  B matrix: $(size(prob.B, 1)) × $(size(prob.B, 2))")
println("  Total primal vars: $(length(prob.c))")
println("  Total dual vars: $(length(prob.g))")
println()

settings = SheafSDP.IPMSettings{Float64}(
    kkt=SheafSDP.UzawaSettings{Float64}(raug=1e6),
    feas_tol=1e-8, gap_tol=1e-8, itmax=100, verbose=false, refine_itmax=0
)

# Warmup
println("Warmup...")
solve(prob, settings)

# Time individual step! calls
println("\nStep-by-step timing:")

solver = SheafSDP.init(prob, settings)
step_times = Float64[]
step_cg_counts = Tuple{Int,Int}[]

# Run all steps, timing each one
while true
    t = @elapsed status = SheafSDP.step!(solver)
    push!(step_times, t)
    row = solver.hist[end]
    push!(step_cg_counts, (row.npred, row.ncorr))
    if status != SheafSDP.CONTINUE
        break
    end
end

println()
println("iter |  time (ms) | pred | corr | ms/CG")
println("-----|------------|------|------|-------")
for i in 1:length(step_times)
    t_ms = step_times[i] * 1000
    pred, corr = step_cg_counts[i]
    total_cg = pred + corr
    ms_per_cg = total_cg > 0 ? t_ms / total_cg : 0.0
    @printf("%4d | %10.1f | %4d | %4d | %5.1f\n", i, t_ms, pred, corr, ms_per_cg)
end

total_time = sum(step_times)
total_cg = sum(p + c for (p, c) in step_cg_counts)
println()
@printf("Total: %.1f ms, %d IPM iters, %d total CG\n", total_time*1000, length(step_times), total_cg)
@printf("Average: %.1f ms/CG iteration\n", total_time*1000/total_cg)

# Analyze where time goes
# Group by high-CG (>5) vs low-CG (<=5) iterations
high_cg_mask = [p + c > 5 for (p, c) in step_cg_counts]
low_cg_mask = .!high_cg_mask

high_cg_time = sum(step_times[high_cg_mask])
low_cg_time = sum(step_times[low_cg_mask])
high_cg_n = sum(p + c for (p, c) in step_cg_counts[high_cg_mask])
low_cg_n = sum(p + c for (p, c) in step_cg_counts[low_cg_mask])

println()
println("Breakdown by CG count:")
@printf("  High-CG iters (>5 CG/iter): %.1f ms (%.1f%%), %d CG, %.1f ms/CG\n",
    high_cg_time*1000, 100*high_cg_time/total_time, high_cg_n,
    high_cg_n > 0 ? high_cg_time*1000/high_cg_n : 0.0)
@printf("  Low-CG iters (≤5 CG/iter):  %.1f ms (%.1f%%), %d CG, %.1f ms/CG\n",
    low_cg_time*1000, 100*low_cg_time/total_time, low_cg_n,
    low_cg_n > 0 ? low_cg_time*1000/low_cg_n : 0.0)

# Also show step 1 specifically since it has most CG
println()
println("First iteration alone:")
@printf("  Time: %.1f ms (%.1f%% of total)\n", step_times[1]*1000, 100*step_times[1]/total_time)
@printf("  CG iters: %d (%.1f%% of total)\n", step_cg_counts[1][1] + step_cg_counts[1][2],
    100*(step_cg_counts[1][1] + step_cg_counts[1][2])/total_cg)
