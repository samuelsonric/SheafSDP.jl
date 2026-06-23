abstract type KKTWorkspace{T} end
abstract type KKTSettings{T} end

include("it.jl")
include("uzawa.jl")

function refine_kkt!(
        Δp::AbstractVector{T},
        Δy::AbstractVector{T},
        wrk::KKTWorkspace{T},
        set::KKTSettings{T},
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        ξp::AbstractVector{T},
        ξy::AbstractVector{T},
        sp::AbstractVector{T},
        sy::AbstractVector{T},
        dp::AbstractVector{T},
        dy::AbstractVector{T};
        itmax::Int=10,
        atol::T=T(1e-12),
        rtol::T=T(1e-13)
    ) where {T}
    kkt_iters = 0
    normξ = max(norm(ξp, Inf), norm(ξy, Inf))
    tol = atol + rtol * normξ

    for _ in 1:itmax
        #
        # compute the primal residual:
        #
        #   sp = ξp - A Δp - Bᵀ Δy
        #
        copyto!(sp, ξp)
        mul!(sp, Symmetric(A, :L), Δp, -1, 1)
        mul!(sp, B', Δy, -1, 1)
        #
        # compute the dual residual:
        #
        #   sy = ξy - B Δp
        #
        copyto!(sy, ξy)
        mul!(sy, B, Δp, -1, 1)

        res_norm = max(norm(sp, Inf), norm(sy, Inf))

        if res_norm <= tol
            break
        end
        #
        # solve for dp and dy:
        #
        #   [ A  Bᵀ ] [ dp ] = [ sp ]
        #   [ B  0  ] [ dy ]   [ sy ]
        #
        kkt_iters += solve_kkt!(wrk, set, dp, dy, A, B, sp, sy)
        #
        # update the directions:
        #
        #   Δp = Δp + dp
        #   Δy = Δy + dy
        #
        axpy!(1, dp, Δp)
        axpy!(1, dy, Δy)
    end

    return kkt_iters
end
