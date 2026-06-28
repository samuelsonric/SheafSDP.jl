abstract type KKTWorkspace{T} end
abstract type KKTSettings{T} end

include("it.jl")
include("uzawa.jl")

#
# residual_stop: stop refinement when
#
#   (1) residual ≤ η ‖ξ‖  (target achieved), or
#   (2) res / res_prev > stall  (contraction stall: κ·u floor reached)
#
function residual_stop(η::T, ξnorm::T, stall::T) where {T}
    prev = Ref(typemax(T))
    return function (pass::Int, res::T, sy)
        hit = res ≤ η * ξnorm
        stalled = pass > 1 && res > stall * prev[]
        prev[] = res
        return hit || stalled
    end
end

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
        dy::AbstractVector{T},
        should_stop;
        itmax::Int = 10,
        rtolmin::T = zero(T),
    ) where {T}
    kkt_iters = 0

    for pass in 1:itmax
        #
        # compute full KKT residual:
        #
        #   sp = ξp - A Δp + Bᵀ Δy
        #   sy = ξy - B Δp
        #
        copyto!(sp, ξp)
        mul!(sp, Symmetric(A, :L), Δp, -1, 1)
        mul!(sp, B', Δy, 1, 1)

        copyto!(sy, ξy)
        mul!(sy, B, Δp, -1, 1)

        res = max(norm(sp, Inf), norm(sy, Inf))

        should_stop(pass, res, sy) && break
        #
        # solve for dp and dy:
        #
        #   [ A -Bᵀ ] [ dp ] = [ sp ]
        #   [ B  0  ] [ dy ]   [ sy ]
        #
        kkt_iters += solve_kkt!(wrk, set, dp, dy, A, B, sp, sy; rtolmin)
        #
        # update the directions:
        #
        #   Δp = Δp + dp
        #   Δy = Δy + dy
        #
        axpy!(one(T), dp, Δp)
        axpy!(one(T), dy, Δy)
    end

    return kkt_iters
end
