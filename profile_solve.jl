using SheafSDP
using SparseArrays
using LinearAlgebra
using Random
using SheafSDP: trinum, triroot, svec!, smat!, symmetrize!
using SheafSDP: residuals!, conedegree, mu, hess!, newton_step!
using SheafSDP: affine_rhs!, corrector_rhs!, step_to_boundary
using BlockSparseArrays: vtxs, colrange, ncols
using CliqueTrees.Multifrontal: FactorizationWorkspace, DivisionWorkspace
using SheafSDP: RiWorkspace

Random.seed!(42)
nv, dv, de = 400, 10, 6

edges = Tuple{Int,Int}[]
for i in 1:nv, j in i+1:nv
    rand() < 0.3 && push!(edges, (i, j))
end
ne = length(edges)

src, dst, maps = Int[], Int[], Matrix{Float64}[]
for (e_idx, (u, v)) in enumerate(edges)
    push!(src, u); push!(dst, e_idx); push!(maps, randn(de, dv))
    push!(src, v); push!(dst, e_idx); push!(maps, randn(de, dv))
end

P, Q, F, L, B = sheaf(src, dst, maps, nv, ne, edges)
B_sp = sparse(B)
n, m = size(F, 1), size(B, 1)

# Build feasible initial point
p, d = zeros(n), zeros(n)
for v in vtxs(B)
    r = colrange(B, v)
    d_v = triroot(ncols(B, v))
    A = randn(d_v, d_v); svec!(view(p, r), A*A'+I, Val(:L))
    A = randn(d_v, d_v); svec!(view(d, r), A*A'+I, Val(:L))
end

y = randn(m)
c = B_sp' * y + d
g = B_sp * p

println("Problem size: n=$n, m=$m, ne=$ne edges")
println()

# Setup workspaces (same as solve!)
ν = conedegree(B)
facwrk = FactorizationWorkspace(F)
divwrk = DivisionWorkspace(F, 1)
itrwrk = RiWorkspace(m, Vector{Float64})
r = zeros(m)
r_p, r_d, r_c = zeros(m), zeros(n), zeros(n)
Δp_aff, Δy_aff, Δd_aff = zeros(n), zeros(m), zeros(n)
Δp, Δy, Δd = zeros(n), zeros(m), zeros(n)
H_blocks = Matrix{Float64}[]
W_blocks = Matrix{Float64}[]
uplo = Val(:L)

# Warmup
residuals!(r_p, r_d, B_sp, p, d, y, c, g)
H = hess!(H_blocks, W_blocks, p, d, B, uplo)
H_sp = sparse(H)
affine_rhs!(r_c, p)
newton_step!(Δp_aff, Δy_aff, Δd_aff, facwrk, divwrk, itrwrk, r, F, L, B, B_sp, H, H_sp,
             r_c, r_p, r_d; τ=1.0, atol=√eps(), rtol=√eps(), itmax=1000)
step_to_boundary(p, d, Δp_aff, Δd_aff, B, uplo; γ=1.0)

# Time each component (average over multiple calls)
N = 10

println("Timing components ($N iterations each):")
println()

t = @elapsed for _ in 1:N
    residuals!(r_p, r_d, B_sp, p, d, y, c, g)
    mu(p, d, ν)
end
println("  residuals! + mu:     $(round(1000*t/N, digits=3)) ms")

t = @elapsed for _ in 1:N
    empty!(H_blocks)
    empty!(W_blocks)
    H = hess!(H_blocks, W_blocks, p, d, B, uplo)
end
println("  hess!:               $(round(1000*t/N, digits=3)) ms")

t = @elapsed for _ in 1:N
    H_sp = sparse(H)
end
println("  sparse(H):           $(round(1000*t/N, digits=3)) ms")

t = @elapsed for _ in 1:N
    affine_rhs!(r_c, p)
end
println("  affine_rhs!:         $(round(1000*t/N, digits=3)) ms")

t = @elapsed for _ in 1:N
    newton_step!(Δp_aff, Δy_aff, Δd_aff, facwrk, divwrk, itrwrk, r, F, L, B, B_sp, H, H_sp,
                 r_c, r_p, r_d; τ=1.0, atol=√eps(), rtol=√eps(), itmax=1000)
end
println("  newton_step!:        $(round(1000*t/N, digits=3)) ms")

t = @elapsed for _ in 1:N
    step_to_boundary(p, d, Δp_aff, Δd_aff, B, uplo; γ=1.0)
end
println("  step_to_boundary:    $(round(1000*t/N, digits=3)) ms")

μ_curr = mu(p, d, ν)
τ_p_aff, τ_d_aff = step_to_boundary(p, d, Δp_aff, Δd_aff, B, uplo; γ=1.0)
p_aff = p + τ_p_aff * Δp_aff
d_aff = d + τ_d_aff * Δd_aff
μ_aff = mu(p_aff, d_aff, ν)
σ = clamp((μ_aff / μ_curr)^3, 0.0, 1.0)

t = @elapsed for _ in 1:N
    corrector_rhs!(r_c, p, d, Δp_aff, Δd_aff, W_blocks, σ * μ_curr, B, uplo)
end
println("  corrector_rhs!:      $(round(1000*t/N, digits=3)) ms")

println()
println("Per iteration estimate: 2x newton_step! + 2x step_to_boundary + rest")

# Now drill into newton_step! / solve_kkt!
using SheafSDP: solve_kkt!, niter, copydia!, axpby!
import CliqueTrees.Multifrontal as MF

println()
println("Breaking down newton_step!:")

# Get iteration count from a fresh solve
f = H_sp * r_c - r_d
iters = solve_kkt!(facwrk, divwrk, itrwrk, Δp, Δy, r, F, L, B, H, f, r_p;
                   α=1.0, atol=√eps(), rtol=√eps(), itmax=1000)
println("  Schur CG iterations: $iters")
println("  Time per newton_step!: $(round(314.0, digits=1)) ms")
println("  Estimated time per Schur iter: $(round(314.0/iters, digits=1)) ms")

# Time B matvec (doesn't corrupt state)
t = @elapsed for _ in 1:N
    mul!(r, B, Δp)
end
println("  mul!(r, B, x):       $(round(1000*t/N, digits=3)) ms")

t = @elapsed for _ in 1:N
    mul!(Δp, B', Δy)
end
println("  mul!(x, B', y):      $(round(1000*t/N, digits=3)) ms")

# The Schur complement matvec does: B * (F \ (B' * y))
# So each iteration costs: 1 ldiv! + 2 B matvec
# ldiv! should dominate if factorization is dense
