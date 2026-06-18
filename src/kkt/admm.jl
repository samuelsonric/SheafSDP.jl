@kwdef struct ADMMSettings{T} <: KKTSettings{T}
    aaug::T   = zero(T)
    raug::T   = one(T)
    relax::T  = one(T)
    atol::T   = √eps(T)
    rtol::T   = √eps(T)
    itmax::Int = 1000
    iatol::T  = √eps(T)
    irtol::T  = √eps(T)
    iitmax::Int = 1000
end

struct ADMMWorkspace{T, I <: Integer, M <: BlockSparseMatrix{T, I}, ItrWrk <: IterationWorkspace{T}} <: KKTWorkspace{T}
    F::M
    itrwrk::ItrWrk
    z::Vector{T}
    u::Vector{T}
    s::Vector{T}
    r::Vector{T}
    t::Vector{T}
    α::Scalar{T}
    τ::Scalar{T}
    nrm::T
end

function ADMMWorkspace(F::BlockSparseMatrix{T, I}, B::BlockSparseMatrix{T, I}) where {T, I <: Integer}
    m, n = size(B)
    @assert size(F, 1) == n
    itrwrk = CgWorkspace(n, n, Vector{T})
    z = zeros(T, n)
    u = zeros(T, n)
    s = zeros(T, n)
    r = zeros(T, n)
    t = zeros(T, m)
    α = ones(T)
    τ = ones(T)
    nrm = norm(B)^2
    return ADMMWorkspace(F, itrwrk, z, u, s, r, t, α, τ, nrm)
end

function init_kkt!(wrk::ADMMWorkspace{T}, set::ADMMSettings{T}, A::BlockSparseMatrix) where {T}
    α = set.aaug + set.raug * norm(Symmetric(A, :L))
    wrk.α[] = α
    wrk.τ[] = inv(wrk.nrm)
    init_admm!(wrk.F, A, α)
    return wrk
end

function init_admm!(F::BlockSparseMatrix{T}, A::BlockSparseMatrix{T}, α::T) where {T}
    @assert size(F, 1) == size(A, 1)

    for v in vtxs(F)
        Fv = block(F, v, v, v)
        Av = block(A, v, v, v)
        copyto!(Fv, Av)

        for i in diagind(Fv)
            Fv[i] += α
        end

        cholesky!(Symmetric(Fv, :L))
    end

    return F
end

function solve_kkt!(
        wrk::ADMMWorkspace{T},
        set::ADMMSettings{T},
        x::AbstractVector{T},
        y::AbstractVector{T},
        B::BlockSparseMatrix{T},
        f::AbstractVector{T},
        g::AbstractVector{T}
    ) where {T}
    F = LowerTriangular(wrk.F)
    return solve_admm!(wrk.itrwrk, F, x, y, wrk.z, wrk.u, wrk.s, wrk.r, wrk.t, B, f, g,
                       wrk.α[], wrk.τ[], set.relax, set.atol, set.rtol, set.itmax,
                       set.iatol, set.irtol, set.iitmax)
end

#
# solve the KKT system via consensus ADMM
#
#   [ A Bᵀ ] [ x ] = [ f ]
#   [ B 0  ] [ y ]   [ g ]
#
function solve_admm!(
        itrwrk::IterationWorkspace{T},
        F::LowerTriangular,
        x::AbstractVector{T},
        y::AbstractVector{T},
        z::AbstractVector{T},
        u::AbstractVector{T},
        s::AbstractVector{T},
        r::AbstractVector{T},
        t::AbstractVector{T},
        B::BlockSparseMatrix{T},
        f::AbstractVector{T},
        g::AbstractVector{T},
        α::T,
        τ::T,
        ρ::T,
        atol::T,
        rtol::T,
        itmax::Int,
        iatol::T,
        irtol::T,
        iitmax::Int
    ) where {T}
    m, n = size(B)

    @assert length(x) == n
    @assert length(z) == n
    @assert length(u) == n
    @assert length(s) == n
    @assert length(r) == n
    @assert length(f) == n
    @assert length(t) == m
    @assert length(g) == m

    function btb!(y, w)
        #
        # compute the product
        #
        #   y ← Bᵀ B w
        #
        mul!(t, B,  w)
        mul!(y, B', t)
    end
    #
    # L is the product
    #
    #   L = Bᵀ B
    #
    L = LinearOperator(T, n, n, true, true, btb!)
    niter = itmax
    mul!(r, B', g)

    for k in 1:itmax
        #
        # solve for x
        #
        #   (A + αI) x = f + α (z − u)
        #
        copyto!(x, f)
        axpy!( α, z, x)
        axpy!(-α, u, x)
        ldiv!(F,  x)
        ldiv!(F', x)
        #
        # relax x:
        #
        #   s = ρ x + (1 − ρ) z
        #
        copyto!(s, x)
        rmul!(s, ρ)
        axpy!(1 - ρ, z, s)
        xnrm = norm(s)
        #
        # update u:
        #
        #   u = u + s
        #
        axpy!(1, s, u)
        #
        # find the closest vector z' to u
        # that solves
        #
        #   L z' = Bᵀ g
        #
        it!(itrwrk, L, r, u; α=τ, atol=iatol, rtol=irtol, itmax=iitmax)
        #
        # update u:
        #
        #   u = u − z'
        #
        axpy!(-1, solution(itrwrk), u)
        #
        # compute primal residual
        #
        #   pres = ‖s − z'‖
        #        = ‖ρ x + (1 − ρ) z − z'‖
        #
        axpy!(-1, solution(itrwrk), s)
        pres = norm(s)
        #
        # compute dual residual
        #
        #   dres = α ‖z' − z‖
        #
        axpy!(-1, solution(itrwrk), z)
        dres = α * norm(z)
        #
        # update z:
        #
        #   z = z'
        #
        copyto!(z, solution(itrwrk))
        #
        # check convergence
        #
        #   pres ≤ atol + rtol max(‖x‖, ‖z‖)
        #   dres ≤ atol + rtol ‖α u‖
        #
        ptol = atol + rtol * max(xnrm, norm(z))
        dtol = atol + rtol * α * norm(u)

        if pres ≤ ptol && dres ≤ dtol
            niter = k
            break
        end
    end

    if niter == itmax
        @warn "ADMM did not converge in $itmax iterations"
    end
    #
    # solve for w
    #
    #   L w = α u
    #
    copyto!(s, u)
    rmul!(s, α)
    it!(itrwrk, L, s; α=τ, atol=iatol, rtol=irtol, itmax=iitmax)
    #
    # compute
    #
    #   y = B w
    #
    mul!(y, B, solution(itrwrk))

    return niter
end
