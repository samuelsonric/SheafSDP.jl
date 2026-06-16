using SheafSDP
using SparseArrays
using LinearAlgebra
using Random
using Printf
using SheafSDP: trinum, triroot, svec!
using BlockSparseArrays: vtxs, colrange, ncols

Random.seed!(42)
nv, dv, de = 400, 10, 6

edges = Tuple{Int,Int}[]
for i in 1:nv, j in i+1:nv
    rand() < 0.3 && push!(edges, (i, j))
end
ne = length(edges)

src, dst, maps = Int[], Int[], Matrix{Float64}[]
for (e_idx, (u, v)) in enumerate(edges)
    push!(src, u); push!(dst, e_idx); push!(maps, randn(de, dv))
    push!(src, v); push!(dst, e_idx); push!(maps, randn(de, dv))
end

P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
B_sp = sparse(B)
n, m = size(F, 1), size(B, 1)

p, d = zeros(n), zeros(n)
for v in vtxs(B)
    r = colrange(B, v)
    d_v = triroot(ncols(B, v))
    A = randn(d_v, d_v); svec!(view(p, r), A*A'+I, Val(:L))
    A = randn(d_v, d_v); svec!(view(d, r), A*A'+I, Val(:L))
end

y = randn(m)
c = B_sp' * y + d
g = B_sp * p

result = solve!(p, d, y, c, g, B, B_sp, F, L; ε_feas=1e-8, ε_μ=1e-8, max_iter=100, verbose=false)

println("Iteration history:")
println("iter |      μ      |   ||r_p||   |   ||r_d||")
println("-----|-------------|-------------|-------------")
for i in 1:result.iterations
    μ = result.μ_history[i]
    rp = result.rp_history[i]
    rd = result.rd_history[i]
    @printf "  %d  | %.3e | %.3e | %.3e\n" i μ rp rd
end
