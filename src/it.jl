mutable struct RiWorkspace{T}
    x::Vector{T}
    r::Vector{T}
    niter::Int
end

function RiWorkspace(n::Int, ::Type{Vector{T}}) where T
    return RiWorkspace(zeros(T, n), zeros(T, n), 0)
end

const IterationWorkspace = Union{RiWorkspace, CgWorkspace, CrWorkspace}

"""
    ri!(workspace, S, b; α=1.0, atol=1e-8, rtol=0.0, itmax=1000)

Richardson iteration for solving S x = b.

Arguments:
- workspace: RiWorkspace containing solution x and residual workspace r
- S: operator supporting mul!(y, S, x)
- b: right-hand side vector
- α: step size
- atol, rtol, itmax: convergence parameters

Solution is stored in workspace.x, iteration count in workspace.niter.
"""
function ri!(
    workspace::RiWorkspace,
    S,
    b::Vector;
    α::Float64=1.0,
    atol::Float64=1e-8,
    rtol::Float64=0.0,
    itmax::Int=1000
)
    x = workspace.x
    r = workspace.r
    #
    # initialize
    #
    #   x = 0
    #
    fill!(x, 0)

    for k in 1:itmax
        #
        # compute the residual
        #
        #   r = b - S x
        #
        mul!(r, S, x)
        axpby!(1, b, -1, r)

        if norm(r) < atol
            workspace.niter = k
            return workspace
        end
        #
        # update x:
        #
        #   x = x + α r
        #
        axpy!(α, r, x)
    end

    @warn "ri! did not converge in $itmax iterations"
    workspace.niter = itmax
    return workspace
end

function it!(itrwrk::RiWorkspace, S, b; α=1.0, atol=1e-8, rtol=0.0, itmax=1000)
    return ri!(itrwrk, S, b; α=α, atol=atol, rtol=rtol, itmax=itmax)
end

function it!(itrwrk::CgWorkspace, S, b; α=1.0, atol=1e-8, rtol=0.0, itmax=1000)
    return cg!(itrwrk, S, b; atol=atol, rtol=rtol, itmax=itmax)
end

function it!(itrwrk::CrWorkspace, S, b; α=1.0, atol=1e-8, rtol=0.0, itmax=1000)
    return cr!(itrwrk, S, b; atol=atol, rtol=rtol, itmax=itmax)
end

function niter(itrwrk::RiWorkspace)
    return itrwrk.niter
end

function niter(itrwrk::CgWorkspace)
    return itrwrk.stats.niter
end

function niter(itrwrk::CrWorkspace)
    return itrwrk.stats.niter
end
