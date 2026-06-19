function wmean(a, b, x, y)
    return (a * x + b * y) / (a + b)
end

function weightedgraph(B::BlockSparseMatrix{T, I}) where {T, I}
    weight = FVector{I}(undef, nvtxs(B))

    for v in vtxs(B)
        weight[v] = ncols(B, v)
    end

    graph = BipartiteGraph(nouts(B), nvtxs(B), narcs(B), B.xsrc, B.tgt)
    return weight, linegraph(graph)
end

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

function eigmin!(
        A::AbstractMatrix{T},
        work::Vector{T},
        iwork::Vector{BlasInt}
    ) where {T <: BlasFloat}
    n = size(A, 1)

    @assert size(A, 2) == n

    require_one_based_indexing(A, work, iwork)
    chkstride1(A)

    W = Vector{T}(undef, 1)
    m = Ref{BlasInt}()
    info = Ref{BlasInt}()
    lwork = BlasInt(-1)
    liwork = BlasInt(-1)

    for i in 1:2
        ccall((@blasfunc(dsyevr_), libblastrampoline), Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
               Ptr{T}, Ref{BlasInt}, Ref{T}, Ref{T},
               Ref{BlasInt}, Ref{BlasInt}, Ref{T}, Ptr{BlasInt},
               Ptr{T}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt},
               Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
               Ref{BlasInt}, Clong, Clong, Clong),
              'N', 'I', 'L', n,
              A, max(1, stride(A, 2)), zero(T), zero(T),
              1, 1, -one(T), m,
              W, A, 1, C_NULL,
              work, lwork, iwork, liwork,
              info, 1, 1, 1)

        chklapackerror(info[])

        if i == 1
            lwork = round(BlasInt, nextfloat(real(work[1])))
            liwork = iwork[1]

            if lwork > length(work)
                resize!(work, lwork)
            end

            if liwork > length(iwork)
                resize!(iwork, liwork)
            end
        end
    end

    return W[1]
end

function svd!(
        s::AbstractVector{T},
        A::AbstractMatrix{T},
        VT::AbstractMatrix{T},
        work::Vector{T},
        iwork::Vector{BlasInt}
    ) where {T <: BlasFloat}
    require_one_based_indexing(A, s, VT, work, iwork)
    chkstride1(A, VT)

    m, n = size(A)
    minmn = min(m, n)

    @assert m >= n
    @assert size(VT) == (n, n)
    @assert length(s) >= minmn
    @assert length(iwork) >= 8minmn

    info = Ref{BlasInt}()
    lwork = BlasInt(-1)

    for i in 1:2
        ccall((@blasfunc(dgesdd_), libblastrampoline), Cvoid,
              (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T},
               Ref{BlasInt}, Ptr{T}, Ptr{T}, Ref{BlasInt},
               Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt},
               Ptr{BlasInt}, Ref{BlasInt}, Clong),
              'O', m, n, A, max(1, stride(A, 2)), s, A, max(1, stride(A, 2)),
              VT, max(1, stride(VT, 2)), work, lwork, iwork, info, 1)

        chklapackerror(info[])

        if i == 1
            lwork = round(BlasInt, nextfloat(real(work[1])))

            if lwork > length(work)
                resize!(work, lwork)
            end
        end
    end

    return A, s, VT
end

function Base.copyto!(L::ChordalTriangular{DIAG, :L, T, I}, A::BlockSparseMatrix{T}) where {DIAG, T, I}
    fill!(L, zero(T))

    v = one(I)

    @inbounds for f in fronts(L)
        fD, res = diagblock(L, f)
        fL, sep = offdblock(L, f)

        rlo = first(res)
        rhi = last(res)

        if !isempty(sep)
            slo = first(sep)
            shi = last(sep)
        end

        while v ≤ nvtxs(A) && colrange(A, v) ⊆ res
            vcol = colrange(A, v)
            vlo = first(vcol)

            for e in srcrange(A, v)
                u = A.tgt[e]
                urow = rowrange(A, u)

                ulo = first(urow)
                uhi = last(urow)

                Ae = block(A, u, v, e)

                if uhi <= rhi
                    ulo < rlo && continue

                    for j in vcol
                        fj = j - rlo + one(I)
                        vj = j - vlo + one(I)

                        for i in urow
                            fi = i - rlo + one(I)
                            vi = i - ulo + one(I)
                            parent(fD)[fi, fj] = Ae[vi, vj]
                        end
                    end
                elseif !isempty(sep) && ulo >= slo && uhi <= shi
                    k = one(I)

                    while sep[k] < ulo
                        k += one(I)
                    end

                    for j in vcol
                        fj = j - rlo + one(I)
                        vj = j - vlo + one(I)

                        kk = k

                        for i in urow
                            vi = i - ulo + one(I)
                            fL[kk, fj] = Ae[vi, vj]
                            kk += one(I)
                        end
                    end
                end
            end

            v += one(I)
        end
    end

    return L
end

function Base.copyto!(L::ChordalTriangular{DIAG, :U, T, I}, A::BlockSparseMatrix{T}) where {DIAG, T, I}
    fill!(L, zero(T))

    vr = one(I)
    vs = one(I)

    @inbounds for f in fronts(L)
        fD, res = diagblock(L, f)
        fL, sep = offdblock(L, f)

        rlo = first(res)
        rhi = last(res)

        while vr ≤ nvtxs(A) && colrange(A, vr) ⊆ res
            vcol = colrange(A, vr)
            vlo = first(vcol)

            for e in srcrange(A, vr)
                u = A.tgt[e]
                urow = rowrange(A, u)

                ulo = first(urow)
                uhi = last(urow)

                Ae = block(A, u, vr, e)

                ulo < rlo && continue
                uhi > rhi && continue

                for j in vcol
                    fj = j - rlo + one(I)
                    vj = j - vlo + one(I)

                    for i in urow
                        fi = i - rlo + one(I)
                        vi = i - ulo + one(I)
                        parent(fD)[fi, fj] = Ae[vi, vj]
                    end
                end
            end

            vr += one(I)
        end

        isempty(sep) && continue

        while vs ≤ nvtxs(A) && last(colrange(A, vs)) < first(sep)
            vs += one(I)
        end

        while vs ≤ nvtxs(A) && colrange(A, vs) ⊆ sep
            vcol = colrange(A, vs)
            vlo = first(vcol)

            k = one(I)

            while sep[k] < vlo
                k += one(I)
            end

            for e in srcrange(A, vs)
                u = A.tgt[e]
                urow = rowrange(A, u)

                ulo = first(urow)
                uhi = last(urow)

                Ae = block(A, u, vs, e)

                ulo < rlo && continue
                uhi > rhi && continue

                for j in vcol
                    fj = k + (j - vlo)
                    vj = j - vlo + one(I)

                    for i in urow
                        fi = i - rlo + one(I)
                        vi = i - ulo + one(I)
                        fL[fi, fj] = Ae[vi, vj]
                    end
                end
            end

            vs += one(I)
        end
    end

    return L
end
