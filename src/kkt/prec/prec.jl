abstract type Preconditioner{T} end
abstract type PreconditionerSettings{T} end

include("noprec.jl")
include("jacobi.jl")
include("ssor.jl")
include("ichol.jl")
