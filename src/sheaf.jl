"""
    sheaf(I, J, V, nv, ne, edges)

Construct a cellular sheaf from restriction map data.

Arguments:
- I: source vertex indices (one per restriction map)
- J: target edge indices (one per restriction map)
- V: restriction maps (each V[k] maps from vertex stalk I[k] to edge stalk J[k])
- nv: number of vertices
- ne: number of edges
- edges: vector of (u, v) tuples

Returns (P, Q, F, L, B) where:
- P: vertex permutation (block-aware fill-reducing)
- Q: secondary permutation
- F: ChordalTriangular working storage for factorization
- L: ChordalTriangular of sheaf Laplacian (B'B) - cached template
- B: coboundary matrix as BlockSparseMatrix, permuted by P
"""
function sheaf(I::Vector{Int}, J::Vector{Int}, V::Vector{<:AbstractMatrix},
               nv::Int, ne::Int, edges::Vector{Tuple{Int,Int}})

    # Extract vertex stalk dimensions from restriction maps
    # V[k] is d_edge × d_vertex, maps from vertex I[k]
    vertex_weights = zeros(Int, nv)
    for k in eachindex(V)
        v = I[k]
        dv = size(V[k], 2)
        if vertex_weights[v] == 0
            vertex_weights[v] = dv
        end
    end

    # Build the graph from edges
    graph = SimpleGraph(nv)
    for (u, v) in edges
        add_edge!(graph, u, v)
    end

    # Get block-aware symbolic factorization
    P, Q, S = symbolic(vertex_weights, graph)

    # Permute vertex indices using P
    perm = P.perm
    invp = P.invp

    # Build coboundary C with permuted vertex blocks
    # For each edge e = (u, v), coboundary has:
    #   C[e, perm[u]] = -F_u
    #   C[e, perm[v]] = +F_v

    # Group restriction maps by edge to identify pairs
    edge_maps = Dict{Int, Vector{Tuple{Int, Matrix{Float64}, Int}}}()  # edge -> [(vertex, map, sign), ...]

    # Need to figure out signs from edge orientation
    # For edge (u,v), the map from u has sign -1, from v has sign +1
    edge_to_vertices = Dict(e => (edges[e][1], edges[e][2]) for e in 1:ne)

    for k in eachindex(V)
        v = I[k]
        e = J[k]
        u1, u2 = edge_to_vertices[e]
        sign = (v == u1) ? -1 : +1
        if !haskey(edge_maps, e)
            edge_maps[e] = []
        end
        push!(edge_maps[e], (v, Matrix{Float64}(V[k]), sign))
    end

    # Build block sparse coboundary C (ne × nv blocks)
    C_I = Int[]      # row block indices (edges)
    C_J = Int[]      # column block indices (permuted vertices)
    C_V = Matrix{Float64}[]  # blocks (with signs applied)

    for e in 1:ne
        for (v, F, sign) in edge_maps[e]
            push!(C_I, e)
            push!(C_J, invp[v])  # permuted vertex index
            push!(C_V, sign * F)
        end
    end

    # Form block sparse coboundary
    B = blocksparse(C_I, C_J, C_V, ne, nv)

    # Build sheaf Laplacian as ChordalCholesky using sparse matrices
    B_sparse = sparse(B)
    L_sparse = B_sparse' * B_sparse
    L_chol = ChordalCholesky(L_sparse, S)

    # Return working factorization object and cached triangular
    F = triangular(similar(L_chol))
    L = triangular(L_chol)

    return P, Q, F, L, B
end
