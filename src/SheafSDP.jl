module SheafSDP

using LinearAlgebra
using LinearAlgebra: chkstride1, BlasFloat, BlasInt, LowerTriangular, Adjoint
using LinearAlgebra.BLAS: @blasfunc, libblastrampoline
using LinearAlgebra.LAPACK: chklapackerror
using Base: require_one_based_indexing, ReshapedArray
using FixedSizeArrays: FixedSizeArrayDefault
using SparseArrays

const FArray{T, N} = FixedSizeArrayDefault{T, N}
const FMatrix{T} = FArray{T, 2}
const FVector{T} = FArray{T, 1}
const FScalar{T} = FArray{T, 0}
const Scalar{T} = Array{T, 0}

# View types for cache structs
const FScalarView{T} = SubArray{T, 0, FVector{T}, Tuple{Int64}, true}
const FVectorView{T} = SubArray{T, 1, FVector{T}, Tuple{UnitRange{Int64}}, true}
const FMatrixView{T} = ReshapedArray{T, 2, FVectorView{T}, Tuple{}}

using Graphs
using CliqueTrees: BipartiteGraph, linegraph
using CliqueTrees.Multifrontal: ChordalLDLt, ldlt!, ChordalCholesky, cholesky!, ChordalSymbolic,
                                 ChordalTriangular, FChordalTriangular, triangular, fronts, diagblock, offdblock,
                                 DivisionWorkspace, FactorizationWorkspace, symbolic, NaturalPermutation, FPermutation
using Krylov: cg!, CgWorkspace, cr!, CrWorkspace
using LinearOperators: LinearOperator
using BlockSparseArrays: BlockSparseMatrix, block, colrange, rowrange, srcrange, nvtxs, vtxs, ncols, nrows, nouts, outs, nbnzs, narcs, blocksparse, selectvtxs, halfselectvtxs, rows, cols
using CommonSolve: init, solve!, solve
using Base: oneto
using Core.Compiler: tmerge

import CommonSolve

include("utils.jl")
include("cone/cone.jl")
include("sheaf.jl")
include("kkt/kkt.jl")
include("scaling.jl")
include("history.jl")
include("ipm.jl")

export sheaf, solve_kkt!, factor_kkt!
export IPMProblem, IPMSettings, IPMSolver, IPMResult, History, IPMStatus, OPTIMAL, NEAR_OPTIMAL, STALLED, NUMERICAL_FAILURE, ITERATION_LIMIT
export step!
export AbstractCone, SemidefiniteCone, PositiveCone, SecondOrderCone, CofreeCone, ExponentialCone
export KKTSettings, UzawaSettings
export Scaling, equilibrate!, scale!, unscale!

end # module SheafSDP
