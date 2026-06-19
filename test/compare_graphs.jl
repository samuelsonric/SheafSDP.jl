#
# Compare preconditioners on different graph structures
#
using SheafSDP
using SheafSDP: Jacobi, SSOR, IChol, weightedgraph
using CliqueTrees: RCM
using CliqueTrees.Multifrontal: symbolic
using BlockSparseArrays: selectvtxs
using LinearAlgebra
using Random

Random.seed!(42)

function build_sheaf(src, dst, dv, de)
    maps = [randn(de, dv) for _ in 1:length(src)]
    return sheaf(src, dst, maps)
end

# Graph builders
function path_graph(n)
    src, dst = Int[], Int[]
    for i in 1:n-1
        push!(src, i); push!(dst, i+1)
        push!(src, i+1); push!(dst, i)
    end
    return src, dst
end

function cycle_graph(n)
    src, dst = path_graph(n)
    push!(src, n); push!(dst, 1)
    push!(src, 1); push!(dst, n)
    return src, dst
end

function grid_graph(nx, ny)
    src, dst = Int[], Int[]
    idx(i, j) = (j-1)*nx + i
    for j in 1:ny, i in 1:nx
        if i < nx
            push!(src, idx(i,j)); push!(dst, idx(i+1,j))
            push!(src, idx(i+1,j)); push!(dst, idx(i,j))
        end
        if j < ny
            push!(src, idx(i,j)); push!(dst, idx(i,j+1))
            push!(src, idx(i,j+1)); push!(dst, idx(i,j))
        end
    end
    return src, dst
end

function star_graph(n)
    src, dst = Int[], Int[]
    for i in 2:n
        push!(src, 1); push!(dst, i)
        push!(src, i); push!(dst, 1)
    end
    return src, dst
end

function complete_graph(n)
    src, dst = Int[], Int[]
    for i in 1:n, j in i+1:n
        push!(src, i); push!(dst, j)
        push!(src, j); push!(dst, i)
    end
    return src, dst
end

function random_graph(n, density)
    src, dst = Int[], Int[]
    edge_set = Set{Tuple{Int,Int}}()
    for i in 1:n, j in i+1:n
        if rand() < density
            push!(src, i); push!(dst, j)
            push!(src, j); push!(dst, i)
            push!(edge_set, (i, j))
        end
    end
    # Ensure connected
    for v in 1:n
        has_edge = any(e -> e[1] == v || e[2] == v, edge_set)
        if !has_edge
            neighbor = v == 1 ? 2 : 1
            push!(src, v); push!(dst, neighbor)
            push!(src, neighbor); push!(dst, v)
        end
    end
    return src, dst
end

# Simple PCG
function pcg(A, b, P; atol=1e-10, rtol=1e-10, itmax=500)
    x = zeros(length(b))
    r = copy(b)
    z = similar(b)
    ldiv!(z, P, r)
    p = copy(z)
    rz = dot(r, z)
    tol = atol + rtol * norm(b)

    for k in 1:itmax
        Ap = A * p
        α = rz / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        norm(r) < tol && return k
        ldiv!(z, P, r)
        rz_new = dot(r, z)
        p .= z .+ (rz_new / rz) .* p
        rz = rz_new
    end
    return itmax
end

function test_graph(name, src, dst; dv=6, de=4, α=0.1)
    B = build_sheaf(src, dst, dv, de)

    # Apply fill-reducing ordering
    weights, graph = weightedgraph(B)
    P, Q, S = symbolic(weights, graph; alg=RCM())
    B = selectvtxs(B, P.perm)

    n = size(B, 2)
    nv = length(unique(vcat(src, dst)))
    ne = length(src) ÷ 2

    A = B' * B
    x0 = randn(n)
    b = A * x0

    P_jac = Jacobi(B; α)
    P_ssor = SSOR(B; α)
    P_ic = IChol(B; α)

    it_jac = pcg(A, b, P_jac)
    it_ssor = pcg(A, b, P_ssor)
    it_ic = pcg(A, b, P_ic)

    println(rpad(name, 20), rpad("v=$nv e=$ne n=$n", 20),
            rpad("Jac: $it_jac", 12), rpad("SSOR: $it_ssor", 14), "IChol: $it_ic")
end

println("="^80)
println("PRECONDITIONER COMPARISON ON DIFFERENT GRAPH STRUCTURES")
println("="^80)
println()
println(rpad("Graph", 20), rpad("Size", 20), rpad("Jacobi", 12), rpad("SSOR", 14), "IChol")
println("-"^80)

# Path graphs
test_graph("Path(20)", path_graph(20)...)
test_graph("Path(50)", path_graph(50)...)

# Cycle graphs
test_graph("Cycle(20)", cycle_graph(20)...)
test_graph("Cycle(50)", cycle_graph(50)...)

# Grid graphs
test_graph("Grid(5x5)", grid_graph(5, 5)...)
test_graph("Grid(7x7)", grid_graph(7, 7)...)
test_graph("Grid(10x10)", grid_graph(10, 10)...)

# Star graphs
test_graph("Star(20)", star_graph(20)...)
test_graph("Star(50)", star_graph(50)...)

# Complete graphs
test_graph("Complete(10)", complete_graph(10)...)
test_graph("Complete(15)", complete_graph(15)...)

# Random graphs
test_graph("Random(30, 0.1)", random_graph(30, 0.1)...)
test_graph("Random(50, 0.1)", random_graph(50, 0.1)...)
test_graph("Random(50, 0.2)", random_graph(50, 0.2)...)
test_graph("Random(100, 0.05)", random_graph(100, 0.05)...)

println("-"^80)
