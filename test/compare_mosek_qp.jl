using SheafSDP
using CommonSolve: solve
using SparseArrays
using LinearAlgebra
using Random
using JuMP
import MathOptInterface as MOI
using MosekTools
using AppleAccelerate
using SheafSDP: triroot, svec!, smat!, symmetrize!
using BlockSparseArrays: vtxs, colrange, ncols, block, blocksparse

Random.seed!(42)
nv, dv, de = 100, 10, 6

edges = Tuple{Int,Int}[]
for i in 1:nv, j in i+1:nv
    rand() < 0.1 && push!(edges, (i, j))
end
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
for (u, v) in edges
    push!(src, u); push!(dst, v); push!(maps, randn(de, dv))
    push!(src, v); push!(dst, u); push!(maps, randn(de, dv))
end

B = sheaf(src, dst, maps)
B_sp = sparse(B)
n, m = size(B, 2), size(B, 1)

# Build Q matrix with same block structure as H
# Use diagonal Q in svec space (each Q_v is diagonal with positive entries)
H_template = SheafSDP.allocblockdiag(B)
fill!(H_template, 0)
q_diag = zeros(n)  # store diagonal for Mosek comparison
for v in vtxs(B)
    r = colrange(B, v)
    n_v = ncols(B, v)  # svec dimension
    Q_v = block(H_template, v, v, v)
    # Q_v is n_v × n_v, make it diagonal with positive entries
    diag_vals = abs.(randn(n_v)) .+ 0.1  # positive diagonal
    for i in 1:n_v
        Q_v[i, i] = diag_vals[i]
    end
    q_diag[r] .= diag_vals
end
Q = H_template

# Build feasible initial point
p0, d0 = zeros(n), zeros(n)
for v in vtxs(B)
    r = colrange(B, v)
    d_v = triroot(ncols(B, v))
    A = randn(d_v, d_v); svec!(view(p0, r), A*A'+I(d_v))
    A = randn(d_v, d_v); svec!(view(d0, r), A*A'+I(d_v))
end

y0 = randn(m)
c = B_sp' * y0 + d0
g = B_sp * p0

println("QP Problem size: n=$n, m=$m, ne=$ne edges")
println()

settings = IPMSettings{Float64}(kkt=UzawaSettings{Float64}(raug=50000.0), feas_tol=1e-8, gap_tol=1e-7, itmax=200)

# Build problem
cones = [:SDP for _ in 1:nv]
prob = IPMProblem(c, g, B, Q, cones)

# Warmup SheafSDP
solve(prob, settings)

# Solve with our solver
println("Solving QP with SheafSDP...")
t1 = @elapsed result = solve(prob, settings)
obj_sheaf = dot(c, result.p) + 0.5 * dot(result.p, Symmetric(Q, :L) * result.p)
println("  time: $(round(t1, digits=3))s, iterations: $(result.iterations)")
println("  converged: $(result.status == OPTIMAL)")
println("  objective: $obj_sheaf")
println()

# Solve with Mosek - simplified formulation without auxiliary variables
println("Solving QP with Mosek...")

block_sizes = [triroot(ncols(B, v)) for v in vtxs(B)]
num_blocks = length(block_sizes)
vertices = collect(vtxs(B))
sqrt2 = sqrt(2.0)

model = Model(Mosek.Optimizer)
set_silent(model)

# PSD block variables
P_blocks = [@variable(model, [1:block_sizes[idx], 1:block_sizes[idx]], PSD) for idx in 1:num_blocks]

# Build svec expressions for each entry (no auxiliary variables)
p_expr = Vector{AffExpr}(undef, n)
for (idx, v) in enumerate(vertices)
    r = colrange(B, v)
    d_v = block_sizes[idx]
    Pv = P_blocks[idx]
    k = 1
    for j in 1:d_v
        p_expr[r[k]] = 1.0 * Pv[j, j]
        k += 1
        for i in j+1:d_v
            p_expr[r[k]] = sqrt2 * Pv[i, j]
            k += 1
        end
    end
end

# Constraint: Bp = g (using the svec expressions directly)
for i in 1:m
    row = B_sp[i, :]
    expr = AffExpr(0.0)
    for (j, val) in zip(findnz(row)...)
        add_to_expression!(expr, val, p_expr[j])
    end
    @constraint(model, expr == g[i])
end

# Objective: min c'p + ½ p'Qp
# Linear part: c'p
obj_lin = sum(c[j] * p_expr[j] for j in 1:n)

# For diagonal Q, quadratic part is: ½ Σ_i q_i * p_i²
# p_i involves PSD matrix entries, so p_i² involves products of those entries
# For diagonal entries: p_ii = P[i,i], so p_ii² = P[i,i]²
# For off-diagonal: p_ij = √2 P[i,j], so p_ij² = 2 P[i,j]²
obj_quad_terms = []
for (idx, v) in enumerate(vertices)
    r = colrange(B, v)
    d_v = block_sizes[idx]
    Pv = P_blocks[idx]
    k = 1
    for j in 1:d_v
        # Diagonal entry: q * P[j,j]²
        q_val = q_diag[r[k]]
        if q_val != 0
            push!(obj_quad_terms, 0.5 * q_val * Pv[j, j]^2)
        end
        k += 1
        for i in j+1:d_v
            # Off-diagonal: q * (√2 P[i,j])² = q * 2 * P[i,j]²
            q_val = q_diag[r[k]]
            if q_val != 0
                push!(obj_quad_terms, 0.5 * q_val * 2 * Pv[i, j]^2)
            end
            k += 1
        end
    end
end

@objective(model, Min, obj_lin + sum(obj_quad_terms))

t2 = @elapsed optimize!(model)
println("  time: $(round(t2, digits=3))s")
println("  barrier iters: $(MOI.get(model, MOI.BarrierIterations()))")
println("  status: $(termination_status(model))")
println("  primal status: $(primal_status(model))")
obj_mosek = objective_value(model)
println("  objective: $obj_mosek")

# Extract Mosek solution in svec form directly from PSD matrices
p_mosek = zeros(n)
for (idx, v) in enumerate(vertices)
    r = colrange(B, v)
    d_v = block_sizes[idx]
    Pv = P_blocks[idx]
    k = 1
    for j in 1:d_v
        p_mosek[r[k]] = value(Pv[j, j])
        k += 1
        for i in j+1:d_v
            p_mosek[r[k]] = sqrt2 * value(Pv[i, j])
            k += 1
        end
    end
end
obj_mosek_check = dot(c, p_mosek) + 0.5 * dot(p_mosek, Symmetric(Q, :L) * p_mosek)

# Check constraint satisfaction
constraint_residual_sheaf = norm(B_sp * result.p - g)
constraint_residual_mosek = norm(B_sp * p_mosek - g)

println()
println("Comparison:")
println("  SheafSDP objective: $obj_sheaf")
println("  Mosek objective:    $obj_mosek")
println("  Mosek obj (check):  $obj_mosek_check")
println("  difference:         $(abs(obj_sheaf - obj_mosek_check))")
println("  relative diff:      $(abs(obj_sheaf - obj_mosek_check) / abs(obj_mosek_check))")
println()
println("Constraint residuals:")
println("  SheafSDP ||Bp - g||: $constraint_residual_sheaf")
println("  Mosek ||Bp - g||:    $constraint_residual_mosek")
