using AppleAccelerate
using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse, colrange, rowrange, vtxs, block
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

# Build problem
N, n_i = 100, 16
prob = build_problem(N, n_i)
settings = SheafSDP.IPMSettings{Float64}(
    kkt=SheafSDP.UzawaSettings{Float64}(raug=1e6),
    feas_tol=1e-8, gap_tol=1e-8, itmax=100, verbose=false, refine_itmax=0
)

println("Problem: N=$N, n_i=$n_i (200 SDP cones of size 16×16)")
println()

# Warmup
println("Warmup...")
solve(prob, settings)

# Create solver and run a few steps
solver = SheafSDP.init(prob, settings)
for _ in 1:3
    SheafSDP.step!(solver)
end

# Time the scale! loop (from step! line 476-478)
println("\nTiming scale! loop (all 201 cones):")
n_trials = 10
function time_scale_loop!(s)
    for v in vtxs(s.B)
        SheafSDP.scale!(s.cones[v], v, s.H, s.caches, s.p, s.d, s.B, s.Q, s.conewrk)
    end
end
# warmup
time_scale_loop!(solver)
t_scale = mean([@elapsed time_scale_loop!(solver) for _ in 1:n_trials])
@printf("  scale! loop: %.1f ms\n", t_scale*1000)

# Time init_kkt!
println("\nTiming init_kkt!:")
t_init_kkt = mean([@elapsed SheafSDP.init_kkt!(solver.kkt, solver.settings.kkt, solver.H) for _ in 1:n_trials])
@printf("  init_kkt!: %.1f ms\n", t_init_kkt*1000)

# Time corrector! loop (wrapper for corr!)
println("\nTiming corrector! loop:")
function time_corr_loop!(s, σμ)
    for v in vtxs(s.B)
        SheafSDP.corrector!(s.cones[v], v, s.wrk.f, s.caches, s.p, s.d, s.wrk.Δpa, s.wrk.Δda, σμ, s.B, s.conewrk)
    end
end
time_corr_loop!(solver, 0.1)
t_corr = mean([@elapsed time_corr_loop!(solver, 0.1) for _ in 1:n_trials])
@printf("  corrector! loop: %.1f ms\n", t_corr*1000)

# Time maxsteps loop
println("\nTiming maxsteps loop:")
function time_maxsteps_loop(s)
    τp_min, τd_min = Inf, Inf
    for v in vtxs(s.B)
        τp, τd = SheafSDP.maxsteps(s.cones[v], v, s.p, s.d, s.wrk.Δp, s.wrk.Δd, s.caches, s.B, s.conewrk)
        τp_min = min(τp_min, τp)
        τd_min = min(τd_min, τd)
    end
    return τp_min, τd_min
end
time_maxsteps_loop(solver)
t_maxsteps = mean([@elapsed time_maxsteps_loop(solver) for _ in 1:n_trials])
@printf("  maxsteps loop: %.1f ms\n", t_maxsteps*1000)

# Time solve_kkt! and count CG iterations
println("\nTiming solve_kkt!:")
fill!(solver.wrk.rp, 0.0)
fill!(solver.wrk.rd, 0.0)
cg_counts = Int[]
times_kkt = Float64[]
for _ in 1:n_trials
    t = @elapsed n_cg = SheafSDP.solve_kkt!(solver.kkt, solver.settings.kkt, solver.wrk.Δp, solver.wrk.Δy, solver.H, solver.B, solver.wrk.rd, solver.wrk.rp, solver.y)
    push!(times_kkt, t)
    push!(cg_counts, n_cg)
end
t_solve_kkt = mean(times_kkt)
avg_cg = mean(cg_counts)
@printf("  solve_kkt!: %.1f ms (avg %.1f CG iters → %.1f ms/CG)\n", t_solve_kkt*1000, avg_cg, t_solve_kkt*1000/avg_cg)

# Summary
println("\n" * "="^50)
println("Summary (per IPM iteration with 2 CG):")
println("="^50)
total = t_scale + t_init_kkt + 2*t_corr + 2*t_maxsteps + 2*t_solve_kkt
@printf("  scale! loop:      %6.1f ms (%5.1f%%)\n", t_scale*1000, 100*t_scale/total)
@printf("  init_kkt!:        %6.1f ms (%5.1f%%)\n", t_init_kkt*1000, 100*t_init_kkt/total)
@printf("  2×corr! loop:     %6.1f ms (%5.1f%%)\n", 2*t_corr*1000, 100*2*t_corr/total)
@printf("  2×maxsteps loop:  %6.1f ms (%5.1f%%)\n", 2*t_maxsteps*1000, 100*2*t_maxsteps/total)
@printf("  2×solve_kkt!:     %6.1f ms (%5.1f%%)\n", 2*t_solve_kkt*1000, 100*2*t_solve_kkt/total)
println("-"^50)
@printf("  Total:            %6.1f ms\n", total*1000)
println()
println("Actual step! time (from earlier): ~145 ms")
println("Difference likely due to: residual computation, line search, history updates")
