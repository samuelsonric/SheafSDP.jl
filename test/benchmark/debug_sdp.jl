using SheafSDP
using SheafSDP: svec!, smat!, roottwo
using CommonSolve: solve
using LinearAlgebra
using Random
using BlockSparseArrays: blocksparse, colrange

Random.seed!(42)

function svecdim(n)
    return div(n * (n + 1), 2)
end

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
                Cbk, Cbl = C[b, k], C[b, l]
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

function passivity_lmi_operator(A, B, C, D)
    T = eltype(A)
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

function random_passive_system(n::Int)
    Q = randn(n, n)
    Q = Q'Q + I
    A = -Q
    B = randn(n, 1)
    C = B'
    D = fill(1.0 + abs(randn()), 1, 1)
    return A, B, C, D
end

# Build a simple test problem
N = 10
n_i = 9
d_e = 5
m_i = 1
edges = [(i, i+1) for i in 1:N-1]

println("Building problem: N=$N, n_i=$n_i, d_e=$d_e")

T = Float64
base_system = random_passive_system(n_i)
systems = [base_system for _ in 1:N]

interface_maps = [(zeros(T, d_e, n_i), zeros(T, d_e, n_i)) for _ in 1:length(edges)]
for (C1, C2) in interface_maps
    for k in 1:min(d_e, n_i)
        C1[k, k] = 1.0
        C2[k, k] = 1.0
    end
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

B_mat = blocksparse(row_ids, col_ids, blocks)

c_vec = zeros(T, size(B_mat, 2))
I_n = Matrix{T}(I, n_i, n_i)
svec_I = zeros(T, sv_G)
svec!(svec_I, I_n)
for i in 1:N
    c_vec[colrange(B_mat, col_G(i))] .= svec_I
end

Q = SheafSDP.allocblockdiag(B_mat)
fill!(Q, zero(T))

cones = Vector{Symbol}(undef, 2N)
for i in 1:N
    cones[col_G(i)] = :SDP
    cones[col_S(i)] = :SDP
end

prob = IPMProblem(c_vec, g_vec, B_mat, Q, cones)
println("Problem size: $(size(B_mat, 2)) vars, $(size(B_mat, 1)) cons")
println("sv_G = $sv_G, sv_S = $sv_S")

# Try different raug values
for raug in [1e4, 1e6, 1e8, 1e10]
    println("\n" * "="^50)
    println("Trying raug = $raug")
    println("="^50)
    settings = IPMSettings{Float64}(
        kkt=UzawaSettings{Float64}(raug=raug),
        feas_tol=1e-6,
        gap_tol=1e-6,
        itmax=30,
        verbose=true
    )

    try
        result = solve(prob, settings)
        println("\nFinal status: $(result.status), iterations: $(result.iterations)")
    catch e
        println("\nERROR: $e")
        showerror(stdout, e, catch_backtrace())
    end
end
