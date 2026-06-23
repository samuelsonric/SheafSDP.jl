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
# L₂-gain certification with γ consensus
#
# N subsystems on path P_N
# Each subsystem (A_i, B_i, C_i, D_i) must satisfy:
#   [A'P + PA + C'C    PB + C'D  ]
#   [B'P + D'C         D'D - μI  ] ⪯ 0
# where μ = γ² is the shared L₂-gain bound squared
#
# Using compliance G = P⁻¹ and Schur complement:
#   [AG + GA'    GC' + BD'   B  ]
#   [CG + DB'    -μ⁻¹I       D  ] ⪯ 0  (needs μ⁻¹, nonlinear)
#
# Alternative: work with P directly (standard form)
# Interface agreement: C_e P_i C_e' = C_e P_j C_e'
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

# Build L₂-gain LMI operator in svec coordinates (for fixed μ)
# 𝒟(P) = [ A'P + PA + C'C    PB + C'D  ]
#        [ B'P + D'C         D'D - μI  ]
# Returns (L, d0) such that svec(𝒟(P)) = L * svec(P) + d0 (for μ=0)
function l2gain_lmi_operator(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                              C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    m = size(B, 2)  # inputs
    p = size(C, 1)  # outputs

    nm = n + m
    sv_P = svecdim(n)
    sv_D = svecdim(nm)

    L = zeros(T, sv_D, sv_P)
    d0 = zeros(T, sv_D)

    P = zeros(T, n, n)
    M = zeros(T, nm, nm)
    v = zeros(T, sv_D)

    # Build L column by column
    for k in 1:sv_P
        fill!(P, zero(T))
        smat!(P, setindex!(zeros(T, sv_P), one(T), k))
        # Symmetrize
        for i in 1:n, j in 1:i-1
            P[j, i] = P[i, j]
        end

        # Top-left: A'P + PA
        M[1:n, 1:n] .= A' * P .+ P * A
        # Top-right: PB
        M[1:n, n+1:nm] .= P * B
        # Bottom-left: B'P
        M[n+1:nm, 1:n] .= B' * P
        # Bottom-right: 0 (μ term and D'D go in d0/separate)
        M[n+1:nm, n+1:nm] .= zero(T)

        svec!(v, M)
        L[:, k] .= v
    end

    # Build d0 (constant part from C'C, C'D, D'C, D'D)
    fill!(M, zero(T))
    M[1:n, 1:n] .= C' * C
    M[1:n, n+1:nm] .= C' * D
    M[n+1:nm, 1:n] .= D' * C
    M[n+1:nm, n+1:nm] .= D' * D
    svec!(d0, M)

    return L, d0
end

# Generate a stable system with guaranteed finite L₂-gain
# Using dissipative construction: A = -Q, small B, C, D
function random_l2gain_system(n::Int, m::Int, p::Int, rng=Random.default_rng())
    # Strongly stable A (large stability margin)
    Q = randn(rng, n, n)
    A = -Q'Q - 5.0*I  # eigenvalues < -5

    # Small B, C to ensure small L₂-gain
    B = 0.1 * randn(rng, n, m)
    C = 0.1 * randn(rng, p, n)

    # D with D'D small relative to expected μ
    D = 0.05 * randn(rng, p, m)

    return A, B, C, D
end

# Build the SheafSDP problem for L₂-gain certification
# Variables: P_i (storage), S_i (slack for LMI), μ (shared gain²)
function build_l2gain_problem(N, n_i, m_i, p_i, d_e, edges)
    T = Float64
    n_edges = length(edges)

    # Generate identical systems (ensures feasibility with shared μ)
    base_system = random_l2gain_system(n_i, m_i, p_i)
    systems = [base_system for _ in 1:N]

    # Interface maps (select first d_e states)
    interface_maps = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for _ in 1:n_edges
        C = zeros(T, d_e, n_i)
        for k in 1:d_e
            C[k, k] = 1.0
        end
        push!(interface_maps, (copy(C), copy(C)))
    end

    # Dimensions
    sv_P = svecdim(n_i)
    sv_S = svecdim(n_i + m_i)
    sv_edge = svecdim(d_e)

    # Column blocks: [P_1, S_1, P_2, S_2, ..., P_N, S_N, μ]
    # μ is a scalar (1D)
    col_P(i) = 2*(i-1) + 1
    col_S(i) = 2*(i-1) + 2
    col_μ = 2*N + 1

    # Row blocks: [lmi_1, ..., lmi_N, agree_1, ..., agree_{N-1}]
    row_lmi(i) = i
    row_agree(e) = N + e

    n_row_blocks = N + n_edges
    n_col_blocks = 2*N + 1

    row_ids, col_ids, blocks = Int[], Int[], Matrix{T}[]
    g_vec = T[]

    # Build LMI constraints: S_i + ℒ_i(P_i) - μ * E_μ = -d0_i
    # where E_μ is the matrix with 1s on diagonal of (2,2) block
    E_μ = zeros(T, n_i + m_i, n_i + m_i)
    for k in n_i+1:n_i+m_i
        E_μ[k, k] = 1.0
    end
    svec_E_μ = zeros(T, sv_S)
    svec!(svec_E_μ, E_μ)

    for i in 1:N
        A, B, C, D = systems[i]
        L, d0 = l2gain_lmi_operator(A, B, C, D)

        # S_i block: identity
        push!(row_ids, row_lmi(i))
        push!(col_ids, col_S(i))
        push!(blocks, Matrix{T}(I, sv_S, sv_S))

        # P_i block: L
        push!(row_ids, row_lmi(i))
        push!(col_ids, col_P(i))
        push!(blocks, L)

        # μ block: -svec(E_μ) (since we want -μI in (2,2) block)
        push!(row_ids, row_lmi(i))
        push!(col_ids, col_μ)
        push!(blocks, reshape(-svec_E_μ, sv_S, 1))

        # RHS
        append!(g_vec, -d0)
    end

    # Build agreement constraints: K_i svec(P_i) - K_j svec(P_j) = 0
    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        K_i = skronr(C_i)
        K_j = skronr(C_j)

        push!(row_ids, row_agree(e))
        push!(col_ids, col_P(i))
        push!(blocks, K_i)

        push!(row_ids, row_agree(e))
        push!(col_ids, col_P(j))
        push!(blocks, -K_j)

        append!(g_vec, zeros(T, sv_edge))
    end

    B = blocksparse(row_ids, col_ids, blocks)
    g = g_vec

    # Objective: min μ (the last column block)
    c_vec = zeros(T, size(B, 2))
    c_vec[colrange(B, col_μ)] .= 1.0

    # Q = 0
    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, zero(T))

    # Cones: P_i is SemidefiniteCone, S_i is SemidefiniteCone, μ is PositiveCone
    cones = Vector{Cone}(undef, n_col_blocks)
    for i in 1:N
        cones[col_P(i)] = SemidefiniteCone()
        cones[col_S(i)] = SemidefiniteCone()
    end
    cones[col_μ] = PositiveCone()

    return IPMProblem(c_vec, g, B, Q, cones), systems, interface_maps
end

# Solve with Mosek for comparison (SISO only: m_i = p_i = 1)
function solve_with_mosek(systems, interface_maps, n_i, m_i, p_i, d_e, edges)
    N = length(systems)
    @assert m_i == 1 && p_i == 1 "Mosek comparison only supports SISO systems"

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables
    P = [@variable(model, [1:n_i, 1:n_i] in PSDCone()) for _ in 1:N]
    @variable(model, μ >= 0)

    # L₂-gain LMI constraints for each subsystem (SISO)
    for i in 1:N
        A, B, C, D = systems[i]
        nm = n_i + 1
        Pi = P[i]

        # Precompute constants (scalars for SISO)
        CTC = C'C  # n_i × n_i
        ctd = (C'D)[1]  # scalar (as 1-element matrix -> extract)
        dtc = (D'C)[1]  # scalar
        dtd = (D'D)[1,1]  # scalar

        # Top-left: -(A'P + PA + C'C)
        TL = @expression(model, [a=1:n_i, b=1:n_i],
            -sum(A[k,a]*Pi[k,b] + Pi[a,k]*A[k,b] for k in 1:n_i) - CTC[a,b])

        # Top-right: -(PB + C'D), size n_i × 1
        TR = @expression(model, [a=1:n_i, b=1:1],
            -sum(Pi[a,k]*B[k,1] for k in 1:n_i) - (C'D)[a,1])

        # Bottom-left: -(B'P + D'C), size 1 × n_i
        BL = @expression(model, [a=1:1, b=1:n_i],
            -sum(B[k,1]*Pi[k,b] for k in 1:n_i) - (D'C)[1,b])

        # Bottom-right: μ - D'D, size 1 × 1
        BR = @expression(model, [a=1:1, b=1:1], μ - dtd)

        M = [TL TR; BL BR]
        @constraint(model, Symmetric(M) in PSDCone())
    end

    # Interface agreement constraints
    for (e, (i, j)) in enumerate(edges)
        C_i, C_j = interface_maps[e]
        for a in 1:d_e, b in 1:d_e
            lhs = sum(C_i[a,k] * P[i][k,l] * C_i[b,l] for k in 1:n_i, l in 1:n_i)
            rhs = sum(C_j[a,k] * P[j][k,l] * C_j[b,l] for k in 1:n_i, l in 1:n_i)
            @constraint(model, lhs == rhs)
        end
    end

    # Objective: min μ
    @objective(model, Min, μ)

    optimize!(model)

    return objective_value(model), solve_time(model)
end

# Parameterized test runner
function run_l2gain_test(N, n_i, m_i, p_i, d_e; warmup=false, raug=1e6)
    # Seed for reproducibility
    Random.seed!(42 + N + n_i)

    edges = [(i, i+1) for i in 1:N-1]

    prob, systems, interface_maps = build_l2gain_problem(N, n_i, m_i, p_i, d_e, edges)

    settings = IPMSettings{Float64}(
        kkt=UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-6,
        gap_tol=1e-6,
        itmax=100,
        verbose=false
    )

    if warmup
        _ = solve(prob, settings)
        _ = solve_with_mosek(systems, interface_maps, n_i, m_i, p_i, d_e, edges)
    end

    t_sheaf = @elapsed result = solve(prob, settings)
    μ_sheaf = result.p[end]  # μ is the last variable
    γ_sheaf = sqrt(max(0, μ_sheaf))

    μ_mosek, t_mosek = solve_with_mosek(systems, interface_maps, n_i, m_i, p_i, d_e, edges)
    γ_mosek = sqrt(max(0, μ_mosek))

    return (
        N=N, n_i=n_i, m_i=m_i, p_i=p_i, d_e=d_e,
        nvars=size(prob.B, 2), ncons=size(prob.B, 1),
        t_sheaf=t_sheaf*1000, t_mosek=t_mosek*1000,
        iters=result.iterations, status=result.status,
        γ_sheaf=γ_sheaf, γ_mosek=γ_mosek,
        γ_diff=abs(γ_sheaf - γ_mosek)
    )
end

# Run tests
println("L₂-gain Certification with γ Consensus")
println("="^80)

# Warmup
println("Warming up...")
run_l2gain_test(3, 3, 1, 1, 2; warmup=true)

println("\n")
@printf("%4s %4s %4s %4s %4s | %6s %6s | %10s %10s | %5s | %8s %8s\n",
        "N", "n_i", "m_i", "p_i", "d_e", "vars", "cons", "SheafSDP", "Mosek", "iters", "γ_sheaf", "γ_mosek")
println("-"^95)

# Test cases: (N, n_i, m_i, p_i, d_e) - SISO systems only
test_cases = [
    (3, 3, 1, 1, 2),
    (5, 5, 1, 1, 3),
    (10, 10, 1, 1, 5),
    (20, 15, 1, 1, 8),
]

for (N, n_i, m_i, p_i, d_e) in test_cases
    r = run_l2gain_test(N, n_i, m_i, p_i, d_e; raug=1e6)
    @printf("%4d %4d %4d %4d %4d | %6d %6d | %8.2fms %8.2fms | %5d | %8.4f %8.4f\n",
            r.N, r.n_i, r.m_i, r.p_i, r.d_e, r.nvars, r.ncons,
            r.t_sheaf, r.t_mosek, r.iters, r.γ_sheaf, r.γ_mosek)
end
