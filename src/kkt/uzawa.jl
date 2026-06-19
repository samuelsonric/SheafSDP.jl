@kwdef struct UzawaSettings{T} <: KKTSettings{T}
    aaug::T = zero(T)
    raug::T = one(T)
    atol::T = √eps(T)
    rtol::T = √eps(T)
    itmax::Int = 1000
end

struct UzawaWorkspace{
        UPLO,
        T,
        I <: Integer,
        ItrWrk <: IterationWorkspace{T}
    } <: KKTWorkspace{T}
    F::FChordalTriangular{:N, UPLO, T, I}
    L::FChordalTriangular{:N, UPLO, T, I}
    facwrk::FactorizationWorkspace{T, I}
    divwrk::DivisionWorkspace{T, I}
    itrwrk::ItrWrk
    r::Vector{T}
    α::Scalar{T}
    nrm::T
end

function UzawaWorkspace(F::FChordalTriangular{:N, UPLO, T, I}, L::FChordalTriangular{:N, UPLO, T, I}, B::BlockSparseMatrix{T, I}) where {UPLO, T, I <: Integer}
    m = size(B, 1)
    facwrk = FactorizationWorkspace(F)
    divwrk = DivisionWorkspace(F, 1)
    itrwrk = CgWorkspace(m, m, Vector{T})
    r = zeros(T, m)
    α = ones(T)
    nrm = norm(B)^2
    return UzawaWorkspace(F, L, facwrk, divwrk, itrwrk, r, α, nrm)
end

#
# Initialize workspace for Uzawa method
#
# Returns (perm, B, workspace) where:
# - perm: block permutation (use to unpermute results)
# - B: permuted coboundary matrix
# - workspace: UzawaWorkspace ready for solve_kkt!
#
function make_kkt(::UzawaSettings{T}, B::BlockSparseMatrix{T, I}) where {T, I}
    weights, graph = weightedgraph(B)

    R, P, S = symbolic(weights, graph)

    B = selectvtxs(B, R.perm)

    F = FChordalTriangular{:N, :L, T, I}(S)
    L = FChordalTriangular{:N, :L, T, I}(S)

    copyto!(L, B' * B)

    wrk = UzawaWorkspace(F, L, B)

    return R, P, B, wrk
end

function init_kkt!(wrk::UzawaWorkspace{UPLO, T}, set::UzawaSettings{T}, A::BlockSparseMatrix) where {UPLO, T}
    α = set.aaug + set.raug * norm(Symmetric(A, :L)) / wrk.nrm
    wrk.α[] = α
    init_uzw!(wrk.facwrk, wrk.F, wrk.L, A, α)
    return wrk
end

# form the augmented block
#
#   F = A + α Bᵀ B
#
# and factorize it
function init_uzw!(
        facwrk::FactorizationWorkspace{T},
        F::ChordalTriangular{:N, UPLO, T},
        L::ChordalTriangular{:N, UPLO, T},
        A::BlockSparseMatrix{T},
        α::T
    ) where {UPLO, T}
    @assert size(F, 1) == size(L, 1) == size(A, 1)

    copyblockdiag!(F, A)
    axpby!(α, L, 1, F)
    cholesky!(facwrk, F)
    return F
end

function solve_kkt!(
    wrk::UzawaWorkspace{UPLO, T},
    set::UzawaSettings{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    A::BlockSparseMatrix{T},
    B::BlockSparseMatrix{T},
    f::AbstractVector{T},
    g::AbstractVector{T}
) where {UPLO, T}
    return solve_uzw!(wrk.divwrk, wrk.itrwrk, x, y, wrk.r, wrk.F, B, f, g, wrk.α[], set.atol, set.rtol, set.itmax)
end

#
# solve the KKT system
#
#   [ A Bᵀ ] [ x ] = [ f ]
#   [ B 0  ] [ y ]   [ g ]
#
function solve_uzw!(
        divwrk::DivisionWorkspace{T},
        itrwrk::IterationWorkspace{T},
        x::AbstractVector{T},
        y::AbstractVector{T},
        r::AbstractVector{T},
        F::ChordalTriangular{:N, UPLO, T},
        B::BlockSparseMatrix{T},
        f::AbstractVector{T},
        g::AbstractVector{T},
        α::T,
        atol::T,
        rtol::T,
        itmax::Int
    ) where {UPLO, T}
    m, n = size(B)

    @assert length(x) == n
    @assert length(y) == m
    @assert length(r) == m
    @assert length(f) == n
    @assert length(g) == m
    @assert size(F, 1) == n
    #
    # solve for x:
    #
    #   F F' x = f + α Bᵀ g
    #
    copyto!(x, f)
    mul!(x, B', g, α, 1)
    ldiv!(divwrk, F, x)
    ldiv!(divwrk, F', x)
    #
    # compute the residual
    #
    #   r = B x - g
    #
    copyto!(r, g)
    mul!(r, B, x, 1, -1)

    function schur!(u, b)
        #
        # compute
        #
        #   u = B x
        #
        # where x solves
        #
        #   F F' x = Bᵀ b
        #
        mul!(x, B', b)
        ldiv!(divwrk, F, x)
        ldiv!(divwrk, F', x)
        mul!(u, B, x)
    end
    #
    # S is the augmented Schur complement:
    #
    #   S = B (F F')⁻¹ Bᵀ
    #
    S = LinearOperator(T, m, m, true, true, schur!)
    #
    # solve for y:
    #
    #   S y = r
    #
    it!(itrwrk, S, r; α, atol, rtol, itmax)

    copyto!(y, solution(itrwrk))
    #
    # compute the dual correction
    #
    #   r = y - α g
    #
    copyto!(r, y)
    axpy!(-α, g, r)
    #
    # solve for x:
    #
    #   F F' x = f - Bᵀ r
    #
    copyto!(x, f)
    mul!(x, B', r, -1, 1)
    ldiv!(divwrk, F, x)
    ldiv!(divwrk, F', x)

    return niter(itrwrk)
end
