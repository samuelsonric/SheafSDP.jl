#
# Shared harness for the Ruiz-equilibration tests.
#
# Builds a strictly-convex conic QP and corrupts it with a known scaling whose
# entries span `kappa` orders of magnitude, so the corrupted problem is a badly
# conditioned *equivalent* of a fixed, well-scaled base problem.
#
#   base:       min ½‖x‖² + cₓ'x   s.t.  A x + s = b,  s ≥ 0
#   corruption: block-constant column scaling (Dx on the free x-block, Ds on the
#               nonneg slack block) + a per-row scaling E0 spanning `kappa`.
#
# Block layout (vertices = column/cone blocks, one output = row block):
#   vertex 1 : x      free   (CofreeCone),   d cols
#   vertex 2 : s      nonneg (PositiveCone),  k cols
#   output 1 : the k equality rows  A x + s = b
#

using SheafSDP
using LinearAlgebra
using Random
using CommonSolve: solve
using BlockSparseArrays: blocksparse, block, colrange, rowrange, vtxs, ncols, srcrange

"dense materialization of a BlockSparseMatrix (test-only helper)"
function todense(B)
    m, n = size(B)
    M = zeros(m, n)
    for v in vtxs(B)
        cr = colrange(B, v)
        for e in srcrange(B, v)
            u  = B.tgt[e]
            rr = rowrange(B, u)
            M[rr, cr] .= block(B, u, v, e)
        end
    end
    return M
end

"""
    build_corrupted_qp(d, k, kappa; seed=42)

Return `(prob, Bdense, Qdense, blocks)` where `prob::IPMProblem` is the corrupted
QP, `Bdense`/`Qdense` are dense copies of its operator/data-Hessian, and `blocks`
is a vector of 1-based `(start, len)` column ranges (one per cone block).
"""
function build_corrupted_qp(d::Int, k::Int, kappa::Float64; seed::Int=42)
    Random.seed!(seed)

    A     = randn(k, d)
    xfeas = randn(d)
    b     = A * xfeas .+ (0.5 .+ rand(k))     # s = b - A xfeas > 0  ⇒ strictly feasible
    cx    = randn(d)

    # known corruption scalars (block-constant on columns; per-row on E0)
    Dx = kappa
    Ds = 1 / sqrt(kappa)
    E0 = exp.(range(-log(sqrt(kappa)), log(sqrt(kappa)); length=k))   # geomspace, per row

    Acorr  = (E0 .* A) .* Dx
    Sblk   = Matrix(Diagonal(E0 .* Ds))
    gcorr  = E0 .* b
    cxcorr = Dx .* cx

    B = blocksparse([1, 1], [1, 2], [Acorr, Sblk])

    n = d + k
    c = zeros(n)
    c[colrange(B, 1)] .= cxcorr               # slack cost stays 0

    g = zeros(size(B, 1))
    g[rowrange(B, 1)] .= gcorr

    Q = SheafSDP.allocblockdiag(B)
    fill!(Q, 0)
    block(Q, 1, 1, 1) .= Matrix((Dx^2) * I, d, d)    # ½‖x‖² Hessian, scaled by Dx²

    cones    = Vector{Cone}(undef, 2)
    cones[1] = CofreeCone()
    cones[2] = PositiveCone()

    prob = IPMProblem(c, g, B, Q, cones)

    Bdense = hcat(Acorr, Sblk)
    Qdense = zeros(n, n)
    Qdense[1:d, 1:d] .= (Dx^2) * Matrix(I, d, d)
    blocks = [(1, d), (d + 1, k)]

    return prob, Bdense, Qdense, blocks
end

# ---- metrics on dense data -------------------------------------------------

"block ∞-norm (max abs over the block's columns of B and Q)"
function block_colnorm_inf(Bdense, Qdense, blocks)
    out = zeros(length(blocks))
    for (kk, (s, l)) in enumerate(blocks)
        cols = s:(s + l - 1)
        v = maximum(abs, @view Bdense[:, cols])
        v = max(v, maximum(abs, @view Qdense[cols, :]))
        out[kk] = v
    end
    return out
end

row_norm_inf(Bdense) = vec(maximum(abs, Bdense; dims=2))

"ratio max/min over all row and block ∞-norms (1.0 == perfectly equalized)"
function norm_spread(Bdense, Qdense, blocks)
    cn = block_colnorm_inf(Bdense, Qdense, blocks)
    rn = row_norm_inf(Bdense)
    a  = vcat(cn, rn)
    a  = a[a .> 0]
    return maximum(a) / minimum(a)
end

"dense KKT matrix with +I cone-Hessian proxy"
function kkt_dense(Bdense, Qdense)
    m, n = size(Bdense)
    H = Qdense + Matrix(I, n, n)
    return [H Bdense'; Bdense zeros(m, m)]
end

# apply a Scaling to dense copies (mirrors B̂ = E B D, Q̂ = D Q D)
function apply_scaling_dense(scaling::Scaling, Bdense, Qdense)
    D = scaling.cscl
    E = scaling.rscl
    Bh = (E .* Bdense) .* D'
    Qh = D .* Qdense .* D'
    return Bh, Qh
end

# ---- solution-space helpers ------------------------------------------------

function objective(prob, p)
    qp = similar(p)
    mul!(qp, Symmetric(prob.Q, :L), p)
    return dot(prob.c, p) + 0.5 * dot(p, qp)
end

function primal_res(prob, p)
    rp = copy(prob.g)
    mul!(rp, prob.B, p, -1, 1)
    return norm(rp)
end

function dual_res(prob, p, d, y)
    rd = copy(prob.c)
    mul!(rd, Symmetric(prob.Q, :L), p, 1, 1)
    mul!(rd, prob.B', y, -1, 1)
    rd .-= d
    return norm(rd)
end
