#
# Compare SheafSDP (POS cones) vs Mosek on an LP
#
using SheafSDP
using SparseArrays
using LinearAlgebra
using Random
using JuMP
import MathOptInterface as MOI
using MosekTools
using SheafSDP: POS
using BlockSparseArrays: vtxs, colrange, ncols

Random.seed!(42)

# Problem size
nv = 200
dv = 10  # vertex stalk dimension (for POS, this is just a vector)
de = 5   # edge stalk dimension

# Create edges (sparse graph)
edges = Tuple{Int,Int}[]
for i in 1:nv, j in i+1:nv
    rand() < 0.05 && push!(edges, (i, j))
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

# Build restriction maps (random)
src, dst, maps = Int[], Int[], Matrix{Float64}[]
for (u, v) in edges
    push!(src, u); push!(dst, v); push!(maps, randn(de, dv))
    push!(src, v); push!(dst, u); push!(maps, randn(de, dv))
end

# Build sheaf structure
B = sheaf(src, dst, maps)
B_sp = sparse(B)
n = size(B_sp, 2)
m = size(B_sp, 1)

# Build feasible initial point (strictly positive for POS)
p0 = zeros(n)
d0 = zeros(n)
for v in vtxs(B)
    r = colrange(B, v)
    p0[r] .= rand(length(r)) .+ 0.1
    d0[r] .= rand(length(r)) .+ 0.1
end
y0 = randn(m)

# Feasible problem data
c = B_sp' * y0 + d0
g = B_sp * p0

println("LP size: n=$n, m=$m, nv=$nv vertices, ne=$ne edges")
println()

# Cones: all POS
cones = [POS() for _ in 1:nv]

# Q = 0 (no quadratic term)
Q_obj = SheafSDP.allocate_H(Float64, B)

settings = IPMSettings{Float64}(kkt=UzawaSettings{Float64}(raug=1000.0), feas_tol=1e-8, gap_tol=1e-8, itmax=100)

# Warmup SheafSDP
p, d, y = copy(p0), copy(d0), copy(y0)
solve!(p, d, y, c, g, B; Q=Q_obj, cones, settings)

# Solve with SheafSDP (timed)
println("Solving LP with SheafSDP (POS cones)...")
p, d, y = copy(p0), copy(d0), copy(y0)
t1 = @elapsed result = solve!(p, d, y, c, g, B; Q=Q_obj, cones, settings)
obj_sheaf = dot(c, result.p)
println("  time: $(round(t1, digits=3))s, iterations: $(result.iterations)")
println("  objective: $obj_sheaf")
println("  converged: $(result.converged)")
println()

# Solve with Mosek
println("Solving LP with Mosek...")

function build_mosek_lp(n, B_sp, g, c)
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, p_var[1:n] >= 0)
    @constraint(model, B_sp * p_var .== g)
    @objective(model, Min, dot(c, p_var))
    return model
end

# Warmup
model_warmup = build_mosek_lp(n, B_sp, g, c)
optimize!(model_warmup)

# Timed run
model = build_mosek_lp(n, B_sp, g, c)
t2 = @elapsed optimize!(model)
println("  time: $(round(t2, digits=3))s")
println("  simplex iters: $(MOI.get(model, MOI.SimplexIterations()))")
println("  barrier iters: $(MOI.get(model, MOI.BarrierIterations()))")
obj_mosek = objective_value(model)
println("  objective: $obj_mosek")
println()

# Compare
println("Comparison:")
println("  SheafSDP objective: $obj_sheaf")
println("  Mosek objective:    $obj_mosek")
println("  difference:         $(abs(obj_sheaf - obj_mosek))")
println("  relative diff:      $(abs(obj_sheaf - obj_mosek) / abs(obj_mosek))")
