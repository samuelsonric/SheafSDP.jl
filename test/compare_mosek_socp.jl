#
# Compare SheafSDP (SOC cones) vs Mosek on a SOCP
#
using SheafSDP
using SparseArrays
using LinearAlgebra
using Random
using JuMP
import MathOptInterface as MOI
using MosekTools
using SheafSDP: SOC
using BlockSparseArrays: vtxs, colrange, ncols

Random.seed!(42)

# Problem size - tuned for SheafSDP advantage (large stalks)
nv = 100
dv = 34  # vertex stalk dimension (SOC dimension)
de = 12  # edge stalk dimension

# Create edges (sparse graph)
edges = Tuple{Int,Int}[]
for i in 1:nv, j in i+1:nv
    rand() < 0.12 && push!(edges, (i, j))
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

# Helper: generate random SOC interior point
function rand_soc_interior(dim; margin=0.5)
    x̄ = randn(dim - 1)
    x₀ = norm(x̄) + margin + rand()
    return [x₀; x̄]
end

# Build feasible initial point (SOC interior)
p0 = zeros(n)
d0 = zeros(n)
for v in vtxs(B)
    r = colrange(B, v)
    dim = length(r)
    p0[r] .= rand_soc_interior(dim)
    d0[r] .= rand_soc_interior(dim)
end
y0 = randn(m)

# Feasible problem data
c = B_sp' * y0 + d0
g = B_sp * p0

println("SOCP size: n=$n, m=$m, nv=$nv vertices, ne=$ne edges")
println()

# Cones: all SOC
cones = [SOC() for _ in 1:nv]

# Q = 0 (no quadratic term)
Q_obj = SheafSDP.allocate_H(Float64, B)

# Warmup SheafSDP
p, d, y = copy(p0), copy(d0), copy(y0)
solve!(p, d, y, c, g, B; Q=Q_obj, cones, feas_tol=1e-8, gap_tol=1e-8, itmax=200, kkt=UzawaSettings{Float64}(raug=5000.0))

# Solve with SheafSDP (timed)
println("Solving SOCP with SheafSDP (SOC cones)...")
p, d, y = copy(p0), copy(d0), copy(y0)
t1 = @elapsed result = solve!(p, d, y, c, g, B;
                               Q=Q_obj, cones, feas_tol=1e-8, gap_tol=1e-8, itmax=200, kkt=UzawaSettings{Float64}(raug=5000.0))
obj_sheaf = dot(c, result.p)
println("  time: $(round(t1, digits=3))s, iterations: $(result.iterations)")
println("  objective: $obj_sheaf")
println("  converged: $(result.converged), status: $(result.status)")
println("  final μ: $(result.μ_history[end])")
println("  final rp: $(result.rp_history[end]), rd: $(result.rd_history[end])")
println("  step sizes (last 10): τ_p=$(round.(result.τ_p_history[end-min(9,length(result.τ_p_history)-1):end], digits=4))")
println("  μ trajectory (last 10): $(round.(result.μ_history[end-min(9,length(result.μ_history)-1):end], digits=6))")
println()

# Solve with Mosek
println("Solving SOCP with Mosek...")

function build_mosek_model(n, nv, B, B_sp, g, c)
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, p_var[1:n])
    for v in vtxs(B)
        r = colrange(B, v)
        @constraint(model, p_var[r] in SecondOrderCone())
    end
    @constraint(model, B_sp * p_var .== g)
    @objective(model, Min, dot(c, p_var))
    return model
end

# Warmup
model_warmup = build_mosek_model(n, nv, B, B_sp, g, c)
optimize!(model_warmup)

# Timed run
model = build_mosek_model(n, nv, B, B_sp, g, c)
t2 = @elapsed optimize!(model)
println("  time: $(round(t2, digits=3))s")
println("  iterations: $(MOI.get(model, MOI.SimplexIterations()))")  # for LP
println("  barrier iters: $(MOI.get(model, MOI.BarrierIterations()))")  # for conic
println("  status: $(termination_status(model))")
obj_mosek = termination_status(model) == MOI.OPTIMAL ? objective_value(model) : NaN
println("  objective: $obj_mosek")
println()

# Compare
println("Comparison:")
println("  SheafSDP objective: $obj_sheaf")
println("  Mosek objective:    $obj_mosek")
println("  difference:         $(abs(obj_sheaf - obj_mosek))")
println("  relative diff:      $(abs(obj_sheaf - obj_mosek) / abs(obj_mosek))")
