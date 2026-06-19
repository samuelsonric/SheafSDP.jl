@kwdef struct ADMMSettings{T} <: KKTSettings{T}
    aaug::T    = zero(T)
    raug::T    = one(T)
    atol::T    = √eps(T)
    rtol::T    = √eps(T)
    relax::T   = one(T)
    itmax::Int = 1000
    iatol::T   = 1e-6
    irtol::T   = 1e-6
    irelax::T  = one(T)
    iitmax::Int = 1000
end

struct ADMMWorkspace{UPLO, T, I <: Integer, ItrWrk <: IterationWorkspace{T}} <: KKTWorkspace{T}
    F::BlockSparseMatrix{T, I}
    itrwrk::ItrWrk
    M::SSOR{UPLO, T, I}
    z::Vector{T}
    u::Vector{T}
    s::Vector{T}
    r::Vector{T}
    t::Vector{T}
    α::Scalar{T}
    τ::Scalar{T}
    nrm::T
end

function ADMMWorkspace{UPLO}(F::BlockSparseMatrix{T, I}, B::BlockSparseMatrix{T, I}) where {UPLO, T, I <: Integer}
    m, n = size(B)
    @assert size(F, 1) == n
    itrwrk = CgWorkspace(n, n, Vector{T})
    M = SSOR{UPLO}(B)
    z = zeros(T, n)
    u = zeros(T, n)
    s = zeros(T, n)
    r = zeros(T, n)
    t = zeros(T, m)
    α = ones(T)
    τ = ones(T)
    nrm = norm(B)^2
    return ADMMWorkspace(F, itrwrk, M, z, u, s, r, t, α, τ, nrm)
end

function ADMMWorkspace(F::BlockSparseMatrix, B::BlockSparseMatrix)
    return ADMMWorkspace{:L}(F, B)
end

#
# Initialize workspace for ADMM method
#
# Returns (perm, B, workspace) where:
# - perm: identity permutation (oneto)
# - B: unchanged input B
# - workspace: ADMMWorkspace ready for solve_kkt!
#
function make_kkt(::ADMMSettings{T}, B::BlockSparseMatrix{T, I}) where {T, I}
    F = allocblockdiag(B)
    wrk = ADMMWorkspace(F, B)
    return oneto(nvtxs(B)), B, wrk
end

function init_kkt!(wrk::ADMMWorkspace{UPLO, T}, set::ADMMSettings{T}, A::BlockSparseMatrix) where {UPLO, T}
    α = set.aaug + set.raug * norm(Symmetric(A, :L))
    wrk.α[] = α
    wrk.τ[] = inv(wrk.nrm)
    wrk.M.ω[] = set.irelax
    init_admm!(wrk.F, A, α, UPLO)
    return wrk
end

function init_admm!(F::BlockSparseMatrix, A::BlockSparseMatrix, α::Number, uplo::Symbol)
    @assert size(F, 1) == size(A, 1)
    copyto!(F, A)
    return cholblockdiag!(F, uplo, α)
end

function solve_kkt!(
        wrk::ADMMWorkspace{UPLO, T},
        set::ADMMSettings{T},
        x::AbstractVector{T},
        y::AbstractVector{T},
        A::BlockSparseMatrix{T},
        B::BlockSparseMatrix{T},
        f::AbstractVector{T},
        g::AbstractVector{T}
    ) where {UPLO, T}

    if UPLO === :L
        F = LowerTriangular(wrk.F)
    else
        F = UpperTriangular(wrk.F)
    end

    return solve_admm!(wrk.itrwrk, wrk.M, F, x, y, wrk.z, wrk.u, wrk.s, wrk.r, wrk.t, A, B, f, g,
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
        M::SSOR{UPLO, T},
        F::AbstractMatrix{T},
        x::AbstractVector{T},
        y::AbstractVector{T},
        z::AbstractVector{T},
        u::AbstractVector{T},
        s::AbstractVector{T},
        r::AbstractVector{T},
        t::AbstractVector{T},
        A::BlockSparseMatrix{T},
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
    ) where {UPLO, T}
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
        it!(itrwrk, L, r, u; M, α=τ, atol=iatol, rtol=irtol, itmax=iitmax)
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
    # solve for w:
    #
    #   L z' = f - A x
    #
    copyto!(s, f)
    mul!(s, A, x, -1, 1)
    it!(itrwrk, L, s; M, α=τ, atol=iatol, rtol=irtol, itmax=iitmax)
    #
    # compute
    #
    #   y = B z'
    #
    mul!(y, B, solution(itrwrk))

    return niter
end
