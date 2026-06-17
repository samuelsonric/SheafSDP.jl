struct UzawaWorkspace{
        UPLO,
        T,
        I <: Integer,
        Fac <: ChordalCholesky{UPLO, T, I},
        Tri <: ChordalTriangular{:N, UPLO, T, I},
        FacWrk <: FactorizationWorkspace{T, I},
        DivWrk <: DivisionWorkspace{T, I},
        ItrWrk <: IterationWorkspace{T}
    } <: KKTWorkspace{T}
    F::Fac
    L::Tri
    facwrk::FacWrk
    divwrk::DivWrk
    itrwrk::ItrWrk
    r::Vector{T}
    α::Scalar{T}
end

function UzawaWorkspace(F::ChordalCholesky{UPLO, T, I}, L::ChordalTriangular{:N, UPLO, T, I}, B::BlockSparseMatrix{T, I}) where {UPLO, T, I <: Integer}
    m = size(B, 1)
    facwrk = FactorizationWorkspace(F)
    divwrk = DivisionWorkspace(F, 1)
    itrwrk = CgWorkspace(m, m, Vector{T})
    r = zeros(T, m)
    α = ones(T)
    return UzawaWorkspace(F, L, facwrk, divwrk, itrwrk, r, α)
end

#
# copy a block diagonal matrix A into F:
#
#   F = A
#
function copydia!(F::ChordalCholesky, A::BlockSparseMatrix)
    copydia!(triangular(F), A)
    return F
end

function copydia!(L::ChordalTriangular, A::BlockSparseMatrix)
    fill!(L, false)

    v = 1

    for f in fronts(L)
        fL, fcol = diagblock(L, f)

        while v ≤ nvtxs(A) && colrange(A, v) ⊆ fcol
            vcol = colrange(A, v); vA = block(A, v, v, v)

            for j in vcol
                fj = j - first(fcol) + 1
                vj = j - first(vcol) + 1

                for i in vcol
                    fi = i - first(fcol) + 1
                    vi = i - first(vcol) + 1

                    parent(fL)[fi, fj] = vA[vi, vj]
                end
            end

            v += 1
        end
    end

    return L
end

function init_kkt!(kktwrk::UzawaWorkspace, A::BlockSparseMatrix; α::Real=1.0)
    kktwrk.α[] = α
    init_uzw!(kktwrk.facwrk, kktwrk.F, kktwrk.L, A; α)
end

# form the augmented block
#
#   F = A + α Bᵀ B
#
# and factorize it
function init_uzw!(
        facwrk::FactorizationWorkspace{T},
        F::ChordalCholesky{UPLO, T},
        L::ChordalTriangular{:N, UPLO, T},
        A::BlockSparseMatrix{T};
        α::Real=1.0
    ) where {UPLO, T}
    n = size(F, 1)
    @assert size(L, 1) == n

    copydia!(F.L, A)
    axpby!(α, L, 1, F.L)
    cholesky!(facwrk, F)
    return
end

function solve_kkt!(
    kktwrk::UzawaWorkspace{UPLO, T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    B::BlockSparseMatrix{T},
    f::AbstractVector{T},
    g::AbstractVector{T};
    atol::Real=√eps(T),
    rtol::Real=√eps(T),
    itmax::Integer=1000
) where {UPLO, T}
    return solve_uzw!(kktwrk.divwrk, kktwrk.itrwrk, x, y, kktwrk.r, kktwrk.F, B, f, g; α=kktwrk.α[], atol, rtol, itmax)
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
        F::ChordalCholesky{UPLO, T},
        B::BlockSparseMatrix{T},
        f::AbstractVector{T},
        g::AbstractVector{T};
        α::Real=1.0,
        atol::Real=√eps(T),
        rtol::Real=√eps(T),
        itmax::Integer=1000
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
    #   F x = f + α Bᵀ g
    #
    copyto!(x, f)
    mul!(x, B', g, α, 1)
    ldiv!(divwrk, F, x)
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
        #   F x = Bᵀ b
        #
        mul!(x, B', b)
        ldiv!(divwrk, F, x)
        mul!(u, B, x)
    end
    #
    # S is the augmented Schur complement:
    #
    #   S = B F⁻¹ Bᵀ
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
    #   F x = f - Bᵀ r
    #
    copyto!(x, f)
    mul!(x, B', r, -1, 1)
    ldiv!(divwrk, F, x)

    return niter(itrwrk)
end
