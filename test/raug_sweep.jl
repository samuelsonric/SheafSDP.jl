using AppleAccelerate
using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse, colrange
using Printf

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

N, n_i = 100, 16
prob = build_problem(N, n_i)

println("raug sweep (N=$N, n_i=$n_i, force_tol=1e-3)")
println("="^60)

# Warmup
solve(prob, SheafSDP.IPMSettings{Float64}(
    kkt=SheafSDP.UzawaSettings{Float64}(raug=1e5),
    feas_tol=1e-8, gap_tol=1e-8, itmax=100, verbose=false,
    force_tol=1e-3
))

for raug in [1e4, 5e4, 1e5, 5e5, 1e6, 1e7]
    settings = SheafSDP.IPMSettings{Float64}(
        kkt=SheafSDP.UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-8, gap_tol=1e-8, itmax=100, verbose=false,
        force_tol=1e-3
    )
    t = @elapsed result = solve(prob, settings)
    @printf("raug=%5.0e | %12s | %2d IPM | %4d CG | %.1f sec\n",
        raug, result.status, result.ipm_niter, result.kkt_niter, t)
end
