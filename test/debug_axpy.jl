using SheafSDP
using LinearAlgebra
using SparseArrays
using BlockSparseArrays
using BlockSparseArrays: BlockSparseMatrix, blocksparse, colrange, rowrange, nvtxs, selectvtxs
using CliqueTrees.Multifrontal: ChordalTriangular, FChordalTriangular, symbolic, fronts, diagblock, offdblock

# Create a simple test case similar to what Uzawa uses
function test_axpy_equivalence()
    # Create a simple block sparse matrix B
    row_ids = [1, 1, 2, 2, 3]
    col_ids = [1, 2, 2, 3, 3]
    blocks = [
        randn(2, 2),
        randn(2, 3),
        randn(3, 3),
        randn(3, 2),
        randn(2, 2)
    ]
    B = blocksparse(row_ids, col_ids, blocks)

    # Compute L = B' * B as BlockSparseMatrix
    L_bsm = B' * B

    # Create symbolic factorization
    weights, graph = SheafSDP.weightedgraph(B)
    R, P, S = symbolic(weights, graph)
    B_perm = selectvtxs(B, R.perm)
    L_bsm_perm = B_perm' * B_perm

    # Create two FChordalTriangular matrices
    F1 = FChordalTriangular{:N, :L, Float64, Int}(S)
    F2 = FChordalTriangular{:N, :L, Float64, Int}(S)
    L_ct = FChordalTriangular{:N, :L, Float64, Int}(S)

    # Old path: copyto! to ChordalTriangular, then axpby!
    fill!(F1, 0.0)
    copyto!(L_ct, L_bsm_perm)
    axpby!(2.5, L_ct, 1, F1)

    # New path: direct axpy! from BlockSparseMatrix
    fill!(F2, 0.0)
    axpy!(2.5, L_bsm_perm, F2)

    # Compare
    println("Comparing old path (copyto! + axpby!) vs new path (axpy!):")

    max_diff = 0.0
    for f in fronts(F1)
        fD1, _ = diagblock(F1, f)
        fD2, _ = diagblock(F2, f)
        diff = maximum(abs.(parent(fD1) .- parent(fD2)))
        max_diff = max(max_diff, diff)
    end

    println("Max difference in diagonal blocks: ", max_diff)

    if max_diff < 1e-10
        println("PASS: Results match!")
    else
        println("FAIL: Results differ!")

        # Print details
        for f in fronts(F1)
            fD1, res = diagblock(F1, f)
            fD2, _ = diagblock(F2, f)
            println("\nFront residual range: ", res)
            println("Old path diagonal block:")
            display(parent(fD1))
            println("\nNew path diagonal block:")
            display(parent(fD2))
            println("\nDifference:")
            display(parent(fD1) .- parent(fD2))
        end
    end

    return max_diff < 1e-10
end

test_axpy_equivalence()
