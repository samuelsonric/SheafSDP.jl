using AppleAccelerate
using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse, colrange, rowrange
using JuMP, MosekTools

Random.seed!(42)

#
# Small-stalk passivity test
#
# N = 3 subsystems on path P_3
# n_i = 3 states, m_i = p_i = 1 (SISO)
# d_e = 2 interface dimension
#

function svecdim(n)
    return div(n * (n + 1), 2)
end

# rectangular symmetric Kronecker product: H = C ⊗ₛ C for C ∈ ℝᵈˣⁿ
# svec(C G C') = (C ⊗ₛ C) svec(G)
# H has size svecdim(d) × svecdim(n)
function skronr(C::AbstractMatrix{T}) where {T}
    d, n = size(C)
    α = roottwo(T)
    H = zeros(T, svecdim(d), svecdim(n))
    #
    # column index (k,l) with k ≤ l → input svec(G)
    # row index (a,b) with a ≤ b → output svec(CGC')
    #
    tkl = 1

    @inbounds for l in 1:n
        # diagonal column: k = l
        tab = 0

        for b in 1:d
            Cbl = C[b, l]

            # diagonal row: a = b
            tab += 1
            H[tab, tkl] = Cbl^2

            # off-diagonal rows: a < b
            for a in b + 1:d
                tab += 1
                H[tab, tkl] = α * C[a, l] * Cbl
            end
        end

        # off-diagonal columns: k < l
        for k in l + 1:n
            tkl += 1
            tab = 0

            for b in 1:d
                Cbk = C[b, k]
                Cbl = C[b, l]

                # diagonal row: a = b
                tab += 1
                H[tab, tkl] = α * Cbk * Cbl

                # off-diagonal rows: a < b
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

# Build svec operator for Lyapunov: svec(AG + GA') = L_A * svec(G)
function lyapunov_svec_operator(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    sv = svecdim(n)
    L = zeros(T, sv, sv)
    G = zeros(T, n, n)
    AG = zeros(T, n, n)
    v = zeros(T, sv)

    for k in 1:sv
        # Set G to the k-th svec basis element
        fill!(G, zero(T))
        smat!(G, setindex!(zeros(T, sv), one(T), k))
        # Symmetrize G
        for i in 1:n, j in 1:i-1
            G[j, i] = G[i, j]
        end
        # Compute AG + GA'
        mul!(AG, A, G)
        AG .= AG .+ AG'
        # Extract svec
        svec!(v, AG)
        L[:, k] .= v
    end

    return L
end

# Build the passivity LMI operator in svec coordinates
# 𝒟(G) = [ AG + GA'    B - GC' ]
#        [ B' - CG    -(D + D') ]
# Returns (L, d0) such that svec(𝒟(G)) = L * svec(G) + d0
function passivity_lmi_operator(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                 C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    m = size(B, 2)
    p = size(C, 1)
    @assert m == p "Passivity requires m = p"

    nm = n + m
    sv_G = svecdim(n)
    sv_D = svecdim(nm)

    L = zeros(T, sv_D, sv_G)
    d0 = zeros(T, sv_D)

    G = zeros(T, n, n)
    M = zeros(T, nm, nm)
    v = zeros(T, sv_D)

    # Build L column by column
    for k in 1:sv_G
        fill!(G, zero(T))
        smat!(G, setindex!(zeros(T, sv_G), one(T), k))
        # Symmetrize
        for i in 1:n, j in 1:i-1
            G[j, i] = G[i, j]
        end

        # Top-left: AG + GA'
        M[1:n, 1:n] .= A * G .+ G * A'
        # Top-right: -GC'
        M[1:n, n+1:nm] .= -G * C'
        # Bottom-left: -CG
        M[n+1:nm, 1:n] .= -C * G
        # Bottom-right: 0 (constant part goes in d0)
        M[n+1:nm, n+1:nm] .= zero(T)

        svec!(v, M)
        L[:, k] .= v
    end

    # Build d0 (constant part)
    fill!(M, zero(T))
    M[1:n, n+1:nm] .= B
    M[n+1:nm, 1:n] .= B'
    M[n+1:nm, n+1:nm] .= -(D .+ D')
    svec!(d0, M)

    return L, d0
end

# Generate a stable passive SISO system (guaranteed passive by construction)
function random_passive_system(n::Int, rng=Random.default_rng())
    # Passive system construction:
    # A = -Q for Q > 0, B = C', D + D' > 0
    # This guarantees passivity with G = Q^{-1}

    Q = randn(rng, n, n)
    Q = Q'Q + I  # Q > 0
    A = -Q

    B = randn(rng, n, 1)
    C = B'  # C = B' makes system "symmetric"

    # D + D' > 0 for strict passivity
    D = fill(1.0 + abs(randn(rng)), 1, 1)

    return A, B, C, D
end

# Build the SheafSDP problem for passivity certification
function build_passivity_problem()
    T = Float64
    N = 3  # number of subsystems
    n_i = 3  # states per subsystem
    m_i = 1  # inputs = outputs (SISO)
    d_e = 2  # interface dimension

    edges = [(1, 2), (2, 3)]
    n_edges = length(edges)

    # Generate identical passive systems (ensures feasibility with same G)
    base_system = random_passive_system(n_i)
    systems = [base_system for _ in 1:N]

    # Generate interface maps C_e^(i) ∈ ℝ^{d_e × n_i}
    # Use simple selection matrices that pick the same interface states
    # This ensures the problem is feasible when all G_i are equal
    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for _ in 1:n_edges
        # Both endpoints use the same selection (first d_e states)
        C = zeros(T, d_e, n_i)
        for k in 1:d_e
            C[k, k] = 1.0
        end
        push!(interface_maps, (copy(C), copy(C)))
    end

    # Dimensions
    sv_G = svecdim(n_i)           # 6
    sv_S = svecdim(n_i + m_i)     # 10
    sv_edge = svecdim(d_e)        # 3

    # Column blocks per node: [G_i, S_i]
    # Row blocks per node: [diss_i] (private)
    # Row blocks per edge: [agree_e] (coordination)

    col_G(i) = 2*(i-1) + 1
    col_S(i) = 2*(i-1) + 2
    row_diss(i) = i
    row_agree(e) = N + e

    n_row_blocks = N + n_edges
    n_col_blocks = 2 * N

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]

    # Build dissipation constraints: S_i + ℒ_i(G_i) = -d0_i
    g_vec = T[]
    for i in 1:N
        A, B, C, D = systems[i]
        L, d0 = passivity_lmi_operator(A, B, C, D)

        # Row: diss_i
        # S_i block: identity
        push!(row_ids, row_diss(i))
        push!(col_ids, col_S(i))
        push!(blocks, Matrix{T}(I, sv_S, sv_S))

        # G_i block: L
        push!(row_ids, row_diss(i))
        push!(col_ids, col_G(i))
        push!(blocks, L)

        # RHS
        append!(g_vec, -d0)
    end

    # Build agreement constraints: (C_e^(i) ⊗_s C_e^(i)) svec(G_i) - (C_e^(j) ⊗_s C_e^(j)) svec(G_j) = 0
    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]

        # Congruence operators
        K_i = skronr(C_i)  # sv_edge × sv_G
        K_j = skronr(C_j)

        # Row: agree_e
        push!(row_ids, row_agree(e))
        push!(col_ids, col_G(i))
        push!(blocks, K_i)

        push!(row_ids, row_agree(e))
        push!(col_ids, col_G(j))
        push!(blocks, -K_j)

        # RHS = 0
        append!(g_vec, zeros(T, sv_edge))
    end

    B = blocksparse(row_ids, col_ids, blocks)
    g = g_vec

    # Objective: min Σ tr(G_i) = min Σ ⟨svec(I), svec(G_i)⟩
    c_vec = zeros(T, size(B, 2))
    I_n = Matrix{T}(I, n_i, n_i)
    svec_I = zeros(T, sv_G)
    svec!(svec_I, I_n)
    for i in 1:N
        c_vec[colrange(B, col_G(i))] .= svec_I
    end

    # Q = 0
    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, zero(T))

    # Cones: [G_i is :SDP, S_i is :SDP] per node
    cones = Vector{Symbol}(undef, n_col_blocks)
    for i in 1:N
        cones[col_G(i)] = :SDP
        cones[col_S(i)] = :SDP
    end

    return IPMProblem(c_vec, g, B, Q, cones), systems, interface_maps
end

# Solve with Mosek for comparison
function solve_with_mosek(systems, interface_maps, n_i, m_i, d_e, edges)
    N = length(systems)
    n_edges = length(edges)

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables: G_i ∈ 𝕊^{n_i} PSD for each subsystem (separate 2D matrices)
    G = [@variable(model, [1:n_i, 1:n_i] in PSDCone()) for _ in 1:N]

    # Passivity LMI constraints for each subsystem
    for i in 1:N
        A, B, C, D = systems[i]
        nm = n_i + m_i
        Gi = G[i]

        # Build the LMI matrix directly: -𝒟(G) ⪰ 0
        # 𝒟(G) = [ AG + GA'    B - GC' ]
        #        [ B' - CG    -(D + D') ]

        # Top-left block: -(AG + GA')
        TL = @expression(model, [a=1:n_i, b=1:n_i],
            -sum(A[a,k]*Gi[k,b] + Gi[a,k]*A[b,k] for k in 1:n_i))

        # Top-right block: -(B - GC') = GC' - B
        TR = @expression(model, [a=1:n_i, b=1:m_i],
            sum(Gi[a,k]*C[b,k] for k in 1:n_i) - B[a,b])

        # Bottom-left block: -(B' - CG) = CG - B'
        BL = @expression(model, [a=1:m_i, b=1:n_i],
            sum(C[a,k]*Gi[k,b] for k in 1:n_i) - B[b,a])

        # Bottom-right block: -(-( D + D')) = D + D'
        BR = @expression(model, [a=1:m_i, b=1:m_i], D[a,b] + D[b,a])

        # Assemble the full matrix
        M = [TL TR; BL BR]
        @constraint(model, Symmetric(M) in PSDCone())
    end

    # Interface agreement constraints
    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        # C_i * G_i * C_i' = C_j * G_j * C_j'
        for a in 1:d_e, b in 1:d_e
            lhs = sum(C_i[a,k] * G[i][k,l] * C_i[b,l] for k in 1:n_i, l in 1:n_i)
            rhs = sum(C_j[a,k] * G[j][k,l] * C_j[b,l] for k in 1:n_i, l in 1:n_i)
            @constraint(model, lhs == rhs)
        end
    end

    # Objective: min Σ tr(G_i)
    @objective(model, Min, sum(G[i][k,k] for i in 1:N, k in 1:n_i))

    optimize!(model)

    return objective_value(model), solve_time(model)
end

# Scaling study
using Printf

function run_scaling_test(N, n_i, d_e; warmup=false)
    m_i = 1
    edges = [(i, i+1) for i in 1:N-1]

    prob, systems, interface_maps = build_passivity_problem_params(N, n_i, m_i, d_e, edges)

    settings = IPMSettings{Float64}(
        kkt=UzawaSettings{Float64}(raug=1e6),
        feas_tol=1e-6,
        gap_tol=1e-6,
        itmax=100,
        verbose=false
    )

    if warmup
        _ = solve(prob, settings)
        _ = solve_with_mosek(systems, interface_maps, n_i, m_i, d_e, edges)
    end

    t_sheaf = @elapsed result = solve(prob, settings)
    obj_sheaf = dot(prob.c, result.p)

    obj_mosek, t_mosek = solve_with_mosek(systems, interface_maps, n_i, m_i, d_e, edges)

    return (
        N=N, n_i=n_i, d_e=d_e,
        nvars=size(prob.B, 2), ncons=size(prob.B, 1),
        t_sheaf=t_sheaf*1000, t_mosek=t_mosek*1000,
        iters=result.iterations, status=result.status,
        obj_diff=abs(obj_sheaf - obj_mosek)
    )
end

# Parameterized problem builder
function build_passivity_problem_params(N, n_i, m_i, d_e, edges)
    T = Float64
    n_edges = length(edges)

    # Generate identical passive systems
    base_system = random_passive_system(n_i)
    systems = [base_system for _ in 1:N]

    # Interface maps: select first d_e states
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

    n_row_blocks = N + n_edges
    n_col_blocks = 2 * N

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]
    g_vec = T[]

    for i in 1:N
        A, B, C, D = systems[i]
        L, d0 = passivity_lmi_operator(A, B, C, D)

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

    cones = Vector{Symbol}(undef, n_col_blocks)
    for i in 1:N
        cones[col_G(i)] = :SDP
        cones[col_S(i)] = :SDP
    end

    return IPMProblem(c_vec, g, B, Q, cones), systems, interface_maps
end

# Run scaling study
println("Passivity SDP Scaling Study")
println("="^80)

# Warmup with small problem
println("Warming up...")
run_scaling_test(3, 3, 2; warmup=true)

println("\n")
@printf("%4s %4s %4s | %6s %6s | %10s %10s | %5s | %10s\n",
        "N", "n_i", "d_e", "vars", "cons", "SheafSDP", "Mosek", "iters", "obj_diff")
println("-"^80)

# Test cases: (N, n_i, d_e)
test_cases = [
    (3, 3, 2),
    (5, 5, 3),
    (10, 10, 5),
    (20, 15, 8),
]

for (N, n_i, d_e) in test_cases
    r = run_scaling_test(N, n_i, d_e)
    @printf("%4d %4d %4d | %6d %6d | %8.2fms %8.2fms | %5d | %10.2e\n",
            r.N, r.n_i, r.d_e, r.nvars, r.ncons, r.t_sheaf, r.t_mosek, r.iters, r.obj_diff)
end

# Find optimal raug
println("\n")
println("="^80)
println("Optimal raug search (N=10, n_i=10, d_e=5)")
println("="^80)

function run_with_raug(N, n_i, d_e, raug)
    m_i = 1
    edges = [(i, i+1) for i in 1:N-1]
    prob, systems, interface_maps = build_passivity_problem_params(N, n_i, m_i, d_e, edges)

    settings = IPMSettings{Float64}(
        kkt=UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-6,
        gap_tol=1e-6,
        itmax=100,
        verbose=false
    )

    t = @elapsed result = solve(prob, settings)
    obj = dot(prob.c, result.p)

    return (raug=raug, time=t*1000, iters=result.iterations, status=result.status, obj=obj)
end

@printf("\n%12s | %10s | %5s | %8s\n", "raug", "time", "iters", "status")
println("-"^45)

raug_values = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
for raug in raug_values
    r = run_with_raug(10, 10, 5, raug)
    @printf("%12.0e | %8.2fms | %5d | %8s\n", r.raug, r.time, r.iters, r.status)
end
