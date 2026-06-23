using AppleAccelerate
using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse, colrange, rowrange
using JuMP, MosekTools
using Printf

Random.seed!(42)

#
# Small-stalk L₂-gain test with large graph
#
# N subsystems on path P_N
# n_i = 8 states (vertex stalk = 36+45 = 81)
# d_e = 5 interface dimension (edge stalk = 15)
#

function svecdim(n)
    return div(n * (n + 1), 2)
end

# rectangular symmetric Kronecker product for congruence
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

# Build L₂-gain LMI operator in svec coordinates
function l2gain_lmi_operator(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                              C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
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

# Generate stable system with finite L₂-gain
function random_l2gain_system(n::Int, m::Int, p::Int, rng=Random.default_rng())
    Q = randn(rng, n, n)
    A = -Q'Q - 10.0*I
    B = 0.05 * randn(rng, n, m)
    C = 0.05 * randn(rng, p, n)
    D = 0.01 * randn(rng, p, m)
    return A, B, C, D
end

# Build the SheafSDP problem for L₂-gain certification
function build_l2gain_problem(N, n_i, m_i, p_i, d_e, edges)
    T = Float64
    n_edges = length(edges)

    Random.seed!(42 + N + n_i)

    base_system = random_l2gain_system(n_i, m_i, p_i)
    systems = [base_system for _ in 1:N]

    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for _ in 1:n_edges
        C = zeros(T, d_e, n_i)
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

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]
    g_vec = T[]

    E_μ = zeros(T, n_i + m_i, n_i + m_i)
    for k in n_i+1:n_i+m_i
        E_μ[k, k] = 1.0
    end
    svec_E_μ = zeros(T, sv_S)
    svec!(svec_E_μ, E_μ)

    for ii in 1:N
        A, B, C, D = systems[ii]
        L, d0 = l2gain_lmi_operator(A, B, C, D)

        push!(row_ids, row_lmi(ii)); push!(col_ids, col_S(ii)); push!(blocks, Matrix{T}(I, sv_S, sv_S))
        push!(row_ids, row_lmi(ii)); push!(col_ids, col_P(ii)); push!(blocks, L)
        push!(row_ids, row_lmi(ii)); push!(col_ids, col_μ); push!(blocks, reshape(-svec_E_μ, sv_S, 1))
        append!(g_vec, -d0)
    end

    for (ee, (ii, jj)) in enumerate(edges)
        C_i, C_j = interface_maps[ee]
        push!(row_ids, row_agree(ee)); push!(col_ids, col_P(ii)); push!(blocks, skronr(C_i))
        push!(row_ids, row_agree(ee)); push!(col_ids, col_P(jj)); push!(blocks, -skronr(C_j))
        append!(g_vec, zeros(T, sv_edge))
    end

    B = blocksparse(row_ids, col_ids, blocks)

    c_vec = zeros(T, size(B, 2))
    c_vec[colrange(B, col_μ)] .= 1.0

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, zero(T))

    cones = Vector{Cone}(undef, 2N + 1)
    for ii in 1:N
        cones[col_P(ii)] = SemidefiniteCone()
        cones[col_S(ii)] = SemidefiniteCone()
    end
    cones[col_μ] = PositiveCone()

    return IPMProblem(c_vec, g_vec, B, Q, cones), systems, interface_maps
end

# Solve with Mosek for comparison (SISO only)
function solve_with_mosek(systems, interface_maps, n_i, m_i, p_i, d_e, edges)
    N = length(systems)
    @assert m_i == 1 && p_i == 1 "Mosek comparison only supports SISO"

    model = Model(Mosek.Optimizer)
    set_silent(model)

    P = [@variable(model, [1:n_i, 1:n_i] in PSDCone()) for _ in 1:N]
    @variable(model, μ >= 0)

    for ii in 1:N
        A, B, C, D = systems[ii]
        Pi = P[ii]
        CTC = C'C
        dtd = (D'D)[1,1]

        TL = @expression(model, [a=1:n_i, b=1:n_i],
            -sum(A[k,a]*Pi[k,b] + Pi[a,k]*A[k,b] for k in 1:n_i) - CTC[a,b])
        TR = @expression(model, [a=1:n_i, b=1:1],
            -sum(Pi[a,k]*B[k,1] for k in 1:n_i) - (C'D)[a,1])
        BL = @expression(model, [a=1:1, b=1:n_i],
            -sum(B[k,1]*Pi[k,b] for k in 1:n_i) - (D'C)[1,b])
        BR = @expression(model, [a=1:1, b=1:1], μ - dtd)

        M = [TL TR; BL BR]
        @constraint(model, Symmetric(M) in PSDCone())
    end

    for (ee, (ii, jj)) in enumerate(edges)
        C_i, C_j = interface_maps[ee]
        for a in 1:d_e, b in 1:d_e
            lhs = sum(C_i[a,k] * P[ii][k,l] * C_i[b,l] for k in 1:n_i, l in 1:n_i)
            rhs = sum(C_j[a,k] * P[jj][k,l] * C_j[b,l] for k in 1:n_i, l in 1:n_i)
            @constraint(model, lhs == rhs)
        end
    end

    @objective(model, Min, μ)
    optimize!(model)

    return objective_value(model), solve_time(model)
end

# Test runner
function run_test(N, n_i, d_e; warmup=false, raug=1e6)
    m_i, p_i = 1, 1
    edges = [(ii, ii+1) for ii in 1:N-1]

    prob, systems, interface_maps = build_l2gain_problem(N, n_i, m_i, p_i, d_e, edges)

    sv_P = svecdim(n_i)
    sv_S = svecdim(n_i + m_i)
    stalk_size = sv_P + sv_S

    settings = IPMSettings{Float64}(
        kkt=UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-8,
        gap_tol=1e-8,
        itmax=100,
        verbose=false,
        refine_itmax=0
    )

    if warmup
        _ = solve(prob, settings)
        _ = solve_with_mosek(systems, interface_maps, n_i, m_i, p_i, d_e, edges)
    end

    t_sheaf = @elapsed result = solve(prob, settings)
    μ_sheaf = result.p[end]
    γ_sheaf = sqrt(max(0, μ_sheaf))

    μ_mosek, t_mosek = solve_with_mosek(systems, interface_maps, n_i, m_i, p_i, d_e, edges)
    γ_mosek = sqrt(max(0, μ_mosek))

    return (
        N=N, n_i=n_i, d_e=d_e, stalk=stalk_size,
        nvars=size(prob.B, 2), ncons=size(prob.B, 1),
        t_sheaf=t_sheaf*1000, t_mosek=t_mosek*1000,
        iters=result.iterations, status=result.status,
        γ_sheaf=γ_sheaf, γ_mosek=γ_mosek,
        γ_diff=abs(γ_sheaf - γ_mosek)
    )
end

# Run large-stalk tests
println("Small-Stalk L2-gain SDP Test (Large Graph)")
println("="^90)

# Warmup
println("Warming up...")
run_test(50, 8, 5; warmup=true)

println("\n")
@printf("%4s %4s %4s %6s | %8s %8s | %10s %10s | %5s | %8s %8s\n",
        "N", "n_i", "d_e", "stalk", "vars", "cons", "SheafSDP", "Mosek", "iters", "γ_sheaf", "γ_mosek")
println("-"^90)

# Large-stalk test case
N, n_i, d_e = 50, 8, 5

r = run_test(N, n_i, d_e)
@printf("%4d %4d %4d %6d | %8d %8d | %8.1fms %8.1fms | %5d | %8.5f %8.5f\n",
        r.N, r.n_i, r.d_e, r.stalk, r.nvars, r.ncons,
        r.t_sheaf, r.t_mosek, r.iters, r.γ_sheaf, r.γ_mosek)

# Find optimal raug
println("\n")
println("="^60)
println("Optimal raug search (N=$N, n_i=$n_i, d_e=$d_e, stalk=$(r.stalk))")
println("="^60)

function run_with_raug(N, n_i, d_e, raug)
    m_i, p_i = 1, 1
    edges = [(ii, ii+1) for ii in 1:N-1]
    prob, _, _ = build_l2gain_problem(N, n_i, m_i, p_i, d_e, edges)

    settings = IPMSettings{Float64}(
        kkt=UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-8, gap_tol=1e-8, itmax=100, verbose=false, refine_itmax=0
    )

    t = @elapsed result = solve(prob, settings)
    return (raug=raug, time=t*1000, iters=result.iterations, status=result.status)
end

@printf("\n%12s | %10s | %5s | %8s\n", "raug", "time", "iters", "status")
println("-"^45)

# Note: raug >= 1e8 fails for L2-gain (POS cone + large raug causes numerical issues)
for raug in [1e4, 1e5, 1e6, 1e7]
    rr = run_with_raug(N, n_i, d_e, raug)
    @printf("%12.0e | %8.1fms | %5d | %8s\n", rr.raug, rr.time, rr.iters, rr.status)
end
