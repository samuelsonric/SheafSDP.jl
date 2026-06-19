using BlockSparseArrays
using LinearAlgebra

# Check if there is a specialized method for LowerTriangular{BlockSparseMatrix}
println("Methods for ldiv! with LowerTriangular or BlockSparse:")
for m in methods(ldiv!)
    str = string(m)
    if occursin("LowerTriangular", str) || occursin("BlockSparse", str)
        println("  ", m)
    end
end
