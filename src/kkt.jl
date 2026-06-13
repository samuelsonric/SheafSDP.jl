# ============================================================================
# Sheaf-aware solvers using pre-computed symbolic structure
# ============================================================================

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

"""
    solve_direct_sheaf(B, A, f, g)

Solve the ECQP by direct factorization of the KKT system:

    [A  Bᵀ] [x]   [f]
    [B -εI] [y] = [g]

Arguments:
- B: coboundary as BlockSparseMatrix
- A: block diagonal vertex weights as BlockSparseMatrix
- f, g: RHS vectors

Returns (x, y).
"""
function solve_direct_sheaf(B::BlockSparseMatrix, A::BlockSparseMatrix,
                            f::Vector, g::Vector)
    n = size(A, 1)
    m = size(B, 1)

    A_sp = sparse(A)
    B_sp = sparse(B)

    ε = 1e-10
    K = [A_sp B_sp'; B_sp -ε*sparse(I, m, m)]
    rhs = [f; g]

    signs = [ones(Int, n); -ones(Int, m)]
    F = ldlt!(ChordalLDLt(K); signs)
    sol = F \ rhs

    x = sol[1:n]
    y = sol[n+1:end]

    return x, y
end

"""
    solve_kkt!(
        facwrk, divwrk, itrwrk,
        x, y, r,
        F, L, B, A,
        f, g;
        γ=1.0, tol=1e-8, maxiter=1000
    )

Solve the ECQP using iterative method on the Schur complement.
Dispatches to Richardson (RiWorkspace) or CG (CgWorkspace) based on itrwrk type.
Zero-allocation version with pre-allocated workspaces.

Arguments:
- facwrk: FactorizationWorkspace(F)
- divwrk: DivisionWorkspace(F, 1)
- itrwrk: RiWorkspace or CgWorkspace
- x: pre-allocated vector of length n (primal solution)
- y: pre-allocated vector of length m (dual solution)
- r: pre-allocated vector of length m (workspace)
- F: ChordalCholesky working object from sheaf()
- L: cached triangular (Laplacian values) from sheaf()
- B: coboundary as BlockSparseMatrix
- A: block diagonal vertex weights as BlockSparseMatrix
- f, g: RHS vectors

Returns iterations count. Solution is in x, y.
"""
function solve_kkt!(
    facwrk::FactorizationWorkspace,
    divwrk::DivisionWorkspace,
    itrwrk::IterationWorkspace,
    x::Vector,
    y::Vector,
    r::Vector,
    F::ChordalCholesky,
    L::ChordalTriangular,
    B::BlockSparseMatrix,
    A::BlockSparseMatrix,
    f::Vector,
    g::Vector;
    γ::Float64=1.0,
    tol::Float64=1e-8,
    maxiter::Int=1000
)
    m = size(B, 1)
    #
    # initialize
    #
    #   F = A + γ Bᵀ B
    #
    # and factorize F.
    #
    copydia!(F.L, A)
    axpby!(γ, L, 1, F.L)
    cholesky!(facwrk, F)
    #
    # solve for x:
    #
    #   F x = f + γ Bᵀ g
    #
    copyto!(x, f)
    mul!(x, B', g, γ, 1)
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
    # S is the augumented Schur complement:
    #
    #   S = Bᵀ F⁻¹ B
    #
    S = LinearOperator(Float64, m, m, true, true, schur!)
    #
    # solve for y:
    #
    #   S y = r
    #
    it!(itrwrk, S, r; α=γ, atol=tol, rtol=0.0, itmax=maxiter)
    copyto!(y, itrwrk.x)
    #
    # compute the dual correction
    #
    #   r = y - γ g
    #
    copyto!(r, y)
    axpy!(-γ, g, r)
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
