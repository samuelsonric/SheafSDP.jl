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
# Small-stalk passivity test with large graph
#
# N subsystems on path P_N
# n_i = 8 states (vertex stalk = 36+45 = 81)
# d_e = 5 interface dimension (edge stalk = 15)
#

function svecdim(n)
    return div(n * (n + 1), 2)
end

# rectangular symmetric Kronecker product: H = C ⊗ₛ C for C ∈ ℝᵈˣⁿ
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
                    tab += 1
                    H[tab, tkl] = C[a, k] * Cbl + C[a, l] * Cbk
                end
            end
        end
        tkl += 1
    end
    return H
end

# Build the passivity LMI operator in svec coordinates
function passivity_lmi_operator(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                 C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    m = size(B, 2)
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
    M[1:n, n+1:nm] .= B
    M[n+1:nm, 1:n] .= B'
    M[n+1:nm, n+1:nm] .= -(D .+ D')
    svec!(d0, M)

    return L, d0
end

# Generate a stable passive SISO system
function random_passive_system(n::Int, rng=Random.default_rng())
    Q = randn(rng, n, n)
    Q = Q'Q + I
    A = -Q
    B = randn(rng, n, 1)
    C = B'
    D = fill(1.0 + abs(randn(rng)), 1, 1)
    return A, B, C, D
end

# Parameterized problem builder
function build_passivity_problem(N, n_i, m_i, d_e, edges)
    T = Float64
    n_edges = length(edges)

    # Seed for reproducibility per problem size
    Random.seed!(42 + N + n_i)

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

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]
    g_vec = T[]

    for i in 1:N
        A, B, C, D = systems[i]
        L, d0 = passivity_lmi_operator(A, B, C, D)

        push!(row_ids, row_diss(i)); push!(col_ids, col_S(i)); push!(blocks, Matrix{T}(I, sv_S, sv_S))
        push!(row_ids, row_diss(i)); push!(col_ids, col_G(i)); push!(blocks, L)
        append!(g_vec, -d0)
    end

    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(i)); push!(blocks, skronr(C_i))
        push!(row_ids, row_agree(e)); push!(col_ids, col_G(j)); push!(blocks, -skronr(C_j))
        append!(g_vec, zeros(T, sv_edge))
    end

    B = blocksparse(row_ids, col_ids, blocks)

    c_vec = zeros(T, size(B, 2))
    I_n = Matrix{T}(I, n_i, n_i)
    svec_I = zeros(T, sv_G)
    svec!(svec_I, I_n)
    for i in 1:N
        c_vec[colrange(B, col_G(i))] .= svec_I
    end

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, zero(T))

    cones = Vector{Symbol}(undef, 2N)
    for i in 1:N
        cones[col_G(i)] = :SDP
        cones[col_S(i)] = :SDP
    end

    return IPMProblem(c_vec, g_vec, B, Q, cones), systems, interface_maps
end

# Solve with Mosek for comparison
function solve_with_mosek(systems, interface_maps, n_i, m_i, d_e, edges)
    N = length(systems)

    model = Model(Mosek.Optimizer)
    set_silent(model)

    G = [@variable(model, [1:n_i, 1:n_i] in PSDCone()) for _ in 1:N]

    for i in 1:N
        A, B, C, D = systems[i]
        nm = n_i + m_i
        Gi = G[i]

        TL = @expression(model, [a=1:n_i, b=1:n_i],
            -sum(A[a,k]*Gi[k,b] + Gi[a,k]*A[b,k] for k in 1:n_i))
        TR = @expression(model, [a=1:n_i, b=1:m_i],
            sum(Gi[a,k]*C[b,k] for k in 1:n_i) - B[a,b])
        BL = @expression(model, [a=1:m_i, b=1:n_i],
            sum(C[a,k]*Gi[k,b] for k in 1:n_i) - B[b,a])
        BR = @expression(model, [a=1:m_i, b=1:m_i], D[a,b] + D[b,a])

        M = [TL TR; BL BR]
        @constraint(model, Symmetric(M) in PSDCone())
    end

    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        for a in 1:d_e, b in 1:d_e
            lhs = sum(C_i[a,k] * G[i][k,l] * C_i[b,l] for k in 1:n_i, l in 1:n_i)
            rhs = sum(C_j[a,k] * G[j][k,l] * C_j[b,l] for k in 1:n_i, l in 1:n_i)
            @constraint(model, lhs == rhs)
        end
    end

    @objective(model, Min, sum(G[i][k,k] for i in 1:N, k in 1:n_i))
    optimize!(model)

    return objective_value(model), solve_time(model)
end

# Test runner
function run_test(N, n_i, d_e; warmup=false, raug=1e6)
    m_i = 1
    edges = [(i, i+1) for i in 1:N-1]

    prob, systems, interface_maps = build_passivity_problem(N, n_i, m_i, d_e, edges)

    sv_G = svecdim(n_i)
    sv_S = svecdim(n_i + m_i)
    stalk_size = sv_G + sv_S

    settings = IPMSettings{Float64}(
        kkt=UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-4,
        gap_tol=1e-4,
        itmax=100,
        verbose=false,
        refine_itmax=0
    )

    if warmup
        _ = solve(prob, settings)
        _ = solve_with_mosek(systems, interface_maps, n_i, m_i, d_e, edges)
    end

    t_sheaf = @elapsed result = solve(prob, settings)
    obj_sheaf = dot(prob.c, result.p)

    obj_mosek, t_mosek = solve_with_mosek(systems, interface_maps, n_i, m_i, d_e, edges)

    return (
        N=N, n_i=n_i, d_e=d_e, stalk=stalk_size,
        nvars=size(prob.B, 2), ncons=size(prob.B, 1),
        t_sheaf=t_sheaf*1000, t_mosek=t_mosek*1000,
        iters=result.iterations, status=result.status,
        obj_diff=abs(obj_sheaf - obj_mosek)
    )
end

# Run large-stalk tests
println("Small-Stalk Passivity SDP Test (Large Graph)")
println("="^90)

# Warmup
println("Warming up...")
run_test(50, 8, 5; warmup=true)

println("\n")
@printf("%4s %4s %4s %6s | %8s %8s | %10s %10s | %5s | %10s\n",
        "N", "n_i", "d_e", "stalk", "vars", "cons", "SheafSDP", "Mosek", "iters", "obj_diff")
println("-"^90)

# Large-stalk test case
N, n_i, d_e = 50, 8, 5

r = run_test(N, n_i, d_e; raug=1e10)
@printf("%4d %4d %4d %6d | %8d %8d | %8.1fms %8.1fms | %5d | %10.2e\n",
        r.N, r.n_i, r.d_e, r.stalk, r.nvars, r.ncons,
        r.t_sheaf, r.t_mosek, r.iters, r.obj_diff)

# Find optimal raug
println("\n")
println("="^60)
println("Optimal raug search (N=$N, n_i=$n_i, d_e=$d_e, stalk=$(r.stalk))")
println("="^60)

function run_with_raug(N, n_i, d_e, raug)
    m_i = 1
    edges = [(i, i+1) for i in 1:N-1]
    prob, _, _ = build_passivity_problem(N, n_i, m_i, d_e, edges)

    settings = IPMSettings{Float64}(
        kkt=UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-4, gap_tol=1e-4, itmax=100, verbose=false, refine_itmax=0
    )

    t = @elapsed result = solve(prob, settings)
    return (raug=raug, time=t*1000, iters=result.iterations, status=result.status)
end

@printf("\n%12s | %10s | %5s | %8s\n", "raug", "time", "iters", "status")
println("-"^45)

for raug in [1e4, 1e5, 1e6, 1e7, 1e8]
    r = run_with_raug(N, n_i, d_e, raug)
    @printf("%12.0e | %8.1fms | %5d | %8s\n", r.raug, r.time, r.iters, r.status)
end
