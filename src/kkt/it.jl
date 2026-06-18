mutable struct RiWorkspace{T}
    x::Vector{T}
    r::Vector{T}
    niter::Int
end

function RiWorkspace(n::Int, ::Type{Vector{T}}) where T
    return RiWorkspace(zeros(T, n), zeros(T, n), 0)
end

const IterationWorkspace{T} = Union{RiWorkspace{T}, CgWorkspace{T}, CrWorkspace{T}}

#
# solve for x using Richardson iterations:
#
#   S x = b
#
function ri!(
    workspace::RiWorkspace{T},
    S,
    b::AbstractVector{T};
    α::Real=1.0,
    atol::Real=√eps(T),
    rtol::Real=√eps(T),
    itmax::Integer=1000
) where {T}
    fill!(workspace.x, 0)
    return ri_impl!(workspace, S, b; α, atol, rtol, itmax)
end

function ri!(
        workspace::RiWorkspace{T},
        S,
        b::AbstractVector{T},
        x0::AbstractVector{T};
        α::Real=1.0,
        atol::Real=√eps(T),
        rtol::Real=√eps(T),
        itmax::Integer=1000
    ) where {T}
    copyto!(workspace.x, x0)
    return ri_impl!(workspace, S, b; α, atol, rtol, itmax)
end

function ri_impl!(
        workspace::RiWorkspace{T},
        S,
        b::AbstractVector{T};
        α::Real=1.0,
        atol::Real=√eps(T),
        rtol::Real=√eps(T),
        itmax::Integer=1000
    ) where {T}
    x = workspace.x
    r = workspace.r
    #
    # compute stopping tolerance
    #
    #   tol = atol + rtol ‖b‖
    #
    tol = atol + rtol * norm(b)

    for k in 1:itmax
        #
        # compute the residual
        #
        #   r = b - S x
        #
        mul!(r, S, x)
        axpby!(1, b, -1, r)

        if norm(r) ≤ tol
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

    @warn "Richardson did not converge in $itmax iterations"
    workspace.niter = itmax
    return workspace
end

function it!(itrwrk::IterationWorkspace{T}, S, b::AbstractVector{T}; α::Real=1.0, atol::Real=√eps(T), rtol::Real=√eps(T), itmax::Integer=1000) where {T}
    if itrwrk isa RiWorkspace
        ri!(itrwrk, S, b; α, atol, rtol, itmax)
    elseif itrwrk isa CgWorkspace
        cg!(itrwrk, S, b; atol, rtol, itmax)
    else
        cr!(itrwrk, S, b; atol, rtol, itmax)
    end

    return itrwrk
end

function it!(itrwrk::IterationWorkspace{T}, S, b::AbstractVector{T}, x0::AbstractVector{T}; α::Real=1.0, atol::Real=√eps(T), rtol::Real=√eps(T), itmax::Integer=1000) where {T}
    if itrwrk isa RiWorkspace
        ri!(itrwrk, S, b, x0; α, atol, rtol, itmax)
    elseif itrwrk isa CgWorkspace
        cg!(itrwrk, S, b, x0; atol, rtol, itmax)
    else
        cr!(itrwrk, S, b, x0; atol, rtol, itmax)
    end

    return itrwrk
end

function solution(itrwrk::RiWorkspace)
    return itrwrk.x
end

function solution(itrwrk::CgWorkspace)
    return itrwrk.x
end

function solution(itrwrk::CrWorkspace)
    return itrwrk.x
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
