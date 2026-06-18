abstract type KKTWorkspace{T} end
abstract type KKTSettings{T} end

include("it.jl")
include("ssor.jl")
include("uzawa.jl")
include("admm.jl")
