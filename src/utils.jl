function allocblockdiag(A::BlockSparseMatrix{T, I}) where {T, I}
    nout = nvtx = narc = nvtxs(A)
    ncol = nrow = ncols(A)

    nblk = zero(I)

    for v in vtxs(A)
        nblk += ncols(A, v)^2
    end

    D = BlockSparseMatrix{T, I}(nout, nvtx, narc, ncol, nrow, nblk)
    return allocblockdiag!(D, A)
end

function allocblockdiag!(D::BlockSparseMatrix{T, I}, A::BlockSparseMatrix{T, I}) where {T, I}
    nout = nvtx = narc = nvtxs(D)
    ncol = nrow = ncols(D)
    nblk = nblks(D)

    blk = zero(I)

    for v in vtxs(A)
        D.xsrc[v] = v
        D.xcol[v] = A.xcol[v]
        D.xrow[v] = A.xcol[v]
        D.xblk[v] = blk + one(I)
        D.tgt[v]  = v

        blk += ncols(A, v)^2
    end

    D.xsrc[nvtx + one(I)] = nout + one(I)
    D.xcol[nvtx + one(I)] = ncol + one(I)
    D.xrow[nout + one(I)] = nrow + one(I)
    D.xblk[narc + one(I)] = nblk + one(I)

    return D
end

function congblockdiag!(D::BlockSparseMatrix, A::BlockSparseMatrix)
    for v in vtxs(D)
        Dv = block(D, v, v, v)
        fill!(Dv, false)
    end

    for v in vtxs(A)
        Dv = block(D, v, v, v)

        for e in srcrange(A, v)
            u = A.tgt[e]
            Ae = block(A, u, v, e)
            mul!(Dv, Ae', Ae, true, true)
        end
    end

    return D
end

function congblockdiag(A::BlockSparseMatrix)
    D = allocblockdiag(A)
    return congblockdiag!(D, A)
end

function cholblockdiag!(D::BlockSparseMatrix, uplo::Symbol, α::Number)
    for v in vtxs(D)
        Dv = block(D, v, v, v)

        for i in diagind(Dv)
            Dv[i] += α
        end

        cholesky!(Symmetric(Dv, uplo))
    end

    return D
end

function cholblockdiag(A::BlockSparseMatrix, uplo::Symbol, α::Number)
    D = congblockdiag(A)
    return cholblockdiag!(D, uplo, α)
end
