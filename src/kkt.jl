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

#
# solve the KKT system
#
#   [ A Bᵀ ] [ x ] = [ f ]
#   [ B 0  ] [ y ]   [ g ]
#
function solve_kkt!(
    facwrk::FactorizationWorkspace{T, I},
    divwrk::DivisionWorkspace{T, I},
    itrwrk::IterationWorkspace{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    r::AbstractVector{T},
    F::ChordalCholesky{UPLO, T},
    L::L_t,
    B::BlockSparseMatrix{T},
    A::BlockSparseMatrix{T},
    f::AbstractVector{T},
    g::AbstractVector{T};
    α::Real=1.0,
    atol::Real=√eps(T),
    rtol::Real=√eps(T),
    itmax::Integer=1000
) where {UPLO, T, I, L_t <: ChordalTriangular}
    m, n = size(B)

    @assert length(x) == n
    @assert length(y) == m
    @assert length(r) == m
    @assert length(f) == n
    @assert length(g) == m
    @assert size(F, 1) == n
    @assert size(L, 1) == n
    #
    # initialize
    #
    #   F = A + α Bᵀ B
    #
    # and factorize F.
    #
    copydia!(F.L, A)
    axpby!(α, L, 1, F.L)
    cholesky!(facwrk, F)
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
