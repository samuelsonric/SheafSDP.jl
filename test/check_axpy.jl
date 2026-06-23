using LinearAlgebra
using LinearAlgebra: I, axpy!
using SheafSDP

# Check if axpy!(r, I, F) works with ChordalTriangular
# The CliqueTrees package might have this implemented

# Let's try a simple test
using CliqueTrees.Multifrontal: ChordalTriangular, FChordalTriangular, ChordalSymbolic, symbolic
using Graphs

# Create a simple test case
G = Graphs.path_graph(5)
R, P, S = symbolic(ones(5), G)
F = FChordalTriangular{:N, :L, Float64, Int}(S)

# Fill with some values
fill!(F, 0.0)
for i in 1:5
    # Set diagonal to 1
end

println("Type of I: ", typeof(I))
println("Type of F: ", typeof(F))

# Try to see if axpy! with UniformScaling is defined
try
    axpy!(0.1, I, F)
    println("axpy!(r, I, F) works!")
catch e
    println("axpy!(r, I, F) failed: ", e)
end
