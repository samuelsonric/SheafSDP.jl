using SheafSDP
using SparseArrays
using LinearAlgebra
using Random
using JuMP
using MosekTools
using SheafSDP: trinum, triroot, svec!, smat!, symmetrize!
using BlockSparseArrays: vtxs, colrange, ncols

Random.seed!(42)
nv, dv, de = 250, 10, 6  # (dv=10=trinum(4), de=6=trinum(3))

edges = Tuple{Int,Int}[]
for i in 1:nv, j in i+1:nv
    rand() < 0.1 && push!(edges, (i, j))  # 10% density
end
# Ensure no isolated vertices
edge_set = Set(edges)
for v in 1:nv
    has_edge = any(e -> e[1] == v || e[2] == v, edges)
    if !has_edge
        neighbor = v == 1 ? 2 : 1
        e = v < neighbor ? (v, neighbor) : (neighbor, v)
        if e ∉ edge_set
            push!(edges, e)
            push!(edge_set, e)
        end
    end
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

# Build feasible initial point for our solver
p0, d0 = zeros(n), zeros(n)
for v in vtxs(B)
    r = colrange(B, v)
    d_v = triroot(ncols(B, v))
    A = randn(d_v, d_v); svec!(view(p0, r), A*A'+I, Val(:L))
    A = randn(d_v, d_v); svec!(view(d0, r), A*A'+I, Val(:L))
end

y0 = randn(m)
c = B_sp' * y0 + d0
g = B_sp * p0

println("Problem size: n=$n, m=$m, ne=$ne edges")
println()

# Warmup SheafSDP
p, d, y = copy(p0), copy(d0), copy(y0)
solve!(p, d, y, c, g, B, B_sp, F, L; ε_feas=1e-8, ε_μ=1e-8, max_iter=100, τ_aug=10.0)

# Solve with our solver (timed after warmup)
println("Solving with SheafSDP...")
p, d, y = copy(p0), copy(d0), copy(y0)
t1 = @elapsed result = solve!(p, d, y, c, g, B, B_sp, F, L;
                               ε_feas=1e-8, ε_μ=1e-8, max_iter=100, τ_aug=10.0)
obj_sheaf = dot(c, result.p)
println("  time: $(round(t1, digits=3))s, iterations: $(result.iterations)")
println("  objective: $obj_sheaf")
println()

# Solve with Mosek
println("Solving with Mosek...")

t_build = @elapsed begin
model = Model(Mosek.Optimizer)
# set_silent(model)

# Create PSD block variables
block_sizes = [triroot(ncols(B, v)) for v in vtxs(B)]
num_blocks = length(block_sizes)

P_blocks = [@variable(model, [1:block_sizes[idx], 1:block_sizes[idx]], PSD) for idx in 1:num_blocks]

# Build p vector from blocks (svec representation with :L ordering)
# For :L: column j has diagonal first, then off-diagonals below
sqrt2 = sqrt(2.0)

p_expr = Vector{AffExpr}(undef, n)
vertices = collect(vtxs(B))
for (idx, v) in enumerate(vertices)
    r = colrange(B, v)
    d_v = block_sizes[idx]
    Pv = P_blocks[idx]

    k = 1
    for j in 1:d_v
        # diagonal entry
        p_expr[r[k]] = 1.0 * Pv[j, j]
        k += 1
        # off-diagonals below diagonal in column j
        for i in j+1:d_v
            p_expr[r[k]] = sqrt2 * Pv[i, j]
            k += 1
        end
    end
end

# Constraint: Bp = g
for i in 1:m
    row = B_sp[i, :]
    expr = AffExpr(0.0)
    for (j, val) in zip(findnz(row)...)
        add_to_expression!(expr, val, p_expr[j])
    end
    @constraint(model, expr == g[i])
end

# Objective: min c'p
obj = AffExpr(0.0)
for j in 1:n
    if c[j] != 0
        add_to_expression!(obj, c[j], p_expr[j])
    end
end
@objective(model, Min, obj)
end
println("  model build: $(round(t_build, digits=3))s")

# Warmup
optimize!(model)

# Timed run - rebuild model
model2 = Model(Mosek.Optimizer)
set_silent(model2)
P_blocks2 = [@variable(model2, [1:block_sizes[idx], 1:block_sizes[idx]], PSD) for idx in 1:num_blocks]
p_expr2 = Vector{AffExpr}(undef, n)
for (idx, v) in enumerate(vertices)
    r = colrange(B, v)
    d_v = block_sizes[idx]
    Pv = P_blocks2[idx]
    k = 1
    for j in 1:d_v
        p_expr2[r[k]] = 1.0 * Pv[j, j]
        k += 1
        for i in j+1:d_v
            p_expr2[r[k]] = sqrt2 * Pv[i, j]
            k += 1
        end
    end
end
for i in 1:m
    row = B_sp[i, :]
    expr = AffExpr(0.0)
    for (j, val) in zip(findnz(row)...)
        add_to_expression!(expr, val, p_expr2[j])
    end
    @constraint(model2, expr == g[i])
end
obj2 = AffExpr(0.0)
for j in 1:n
    if c[j] != 0
        add_to_expression!(obj2, c[j], p_expr2[j])
    end
end
@objective(model2, Min, obj2)

print("  solve (after warmup): ")
@time optimize!(model2)
obj_mosek = objective_value(model2)
println("  objective: $obj_mosek")
println()

# Compare
println("Comparison:")
println("  SheafSDP objective: $obj_sheaf")
println("  Mosek objective:    $obj_mosek")
println("  difference:         $(abs(obj_sheaf - obj_mosek))")
println("  relative diff:      $(abs(obj_sheaf - obj_mosek) / abs(obj_mosek))")
