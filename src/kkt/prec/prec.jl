abstract type AbstractPreconditioner{T} end

include("jacobi.jl")
include("ssor.jl")
include("ichol.jl")
