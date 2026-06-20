#
# invariants.jl  —  "check my math" harness for the SheafSDP IPM
#
# Two layers:
#   (1) cone math, random interior points, no solver needed
#       -> svec isometry, skron identity, NT identity H·p=d, H≻0,
#          maxstep boundary, corr! baseline, identity idempotence
#   (2) solver level, instrumented step! on real problems
#       -> per-iteration affine/corrector identities (these are the ones
#          that catch a wrong corrector that still converges),
#          cone membership of every iterate, final feasibility / gap /
#          duality identity
#   (3) positive control: inject a wrong corrector and confirm (2) catches it
#
# NOTE: diagnostic_step! mirrors SheafSDP.step! line-for-line and inserts
# checks. If you change step!, mirror it here too.
#
# NOTE on rd: the complementarity identities are rd-FREE. Confirmed against the
# real newton! (ipm.jl): it recovers Δd = rd − BᵀΔy + QΔp, while step! passes
# f = corr_term − rd, so H_cone·Δp + Δd = (AΔp − BᵀΔy) + rd = f + rd = corr_term
# — the rd cancels exactly. So do NOT add an rd term to these identities (an
# earlier suggestion to use corr_term − rd / −⟨p,rd⟩ is wrong: it would also
# absorb a real corrector error and let a wrong-but-converging corrector pass).
# Each identity holds to the KKT solve tolerance (Uzawa atol=rtol=√eps), printed
# separately as the KKT-solve-residual floor.
#
# NOTE: complementarity checks are gated on ν>0 (conic content). For an all-NOC
# problem d≡0, ⟨p,d⟩=0, ν=0, so they are vacuous (0/0) and are skipped — that is
# why the old version blew up to ~1e10 on the pure-NOC QP, not a solver fault.
#
# NOTE: the SDP and SOC *problems* below are best-effort constructions.
# If the cone-math layer (1) passes but a problem in layer (2) fails its
# invariants, suspect the problem construction here before the solver.
#
using SheafSDP
using SheafSDP: triroot, svec!, smat!, skron!, socmul!, socdet, socroot!,
                degree, identity!, scale!, hess!, corr!, maxstep, cachesize,
                Caches, cache, allocblockdiag, residuals!, corrector!,
                maxsteps, newton!, init_kkt!, isstalled, isnumfail,
                SDP, POS, SOC, NOC, FVector
using SheafSDP: hess! as block_hess!         # disambiguate the block-level method
using CommonSolve: init
using BlockSparseArrays: vtxs, colrange, ncols, block, blocksparse, rowrange
using LinearAlgebra
using Random
using Printf

# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

# symmetric d×d  ->  svec (length d(d+1)/2)
function to_svec(M::AbstractMatrix)
    d = size(M, 1); n = d * (d + 1) ÷ 2
    v = zeros(eltype(M), n)
    svec!(v, M)
    return v
end

# svec -> symmetric d×d  (smat! fills lower triangle; we reflect)
function to_smat(v::AbstractVector, d::Int)
    M = zeros(eltype(v), d, d)
    smat!(M, v)
    for j in 1:d, i in 1:j-1
        M[i, j] = M[j, i]
    end
    return M
end

# build the *real* view-based cache for a single block of embedding dim n,
# exactly as the solver does it (so we test the real path)
function single_cache(cone, n::Int; T::Type = Float64)
    I = Int
    xcol = FVector{I}(undef, 2); xcol[1] = 1; xcol[2] = 1 + n
    csz  = cachesize(cone, n)
    xblk = FVector{I}(undef, 2); xblk[1] = 1; xblk[2] = 1 + csz
    val  = FVector{T}(undef, csz)
    return cache(Caches(xcol, xblk, val), 1, cone)
end

# Jordan product per cone (svec coords for SDP)
jmul(::POS, x, y) = x .* y
function jmul(::SOC, x, y)
    # socmul! on scaled inputs returns 2(x∘y)_phys — one factor of √2 too many.
    # Divide by √2 so cone-math on scaled elements is self-consistent.
    out = similar(x); socmul!(out, x, y); rmul!(out, 1 / sqrt(2)); out
end
function jmul(::SDP, x, y)
    d = triroot(length(x))
    X = to_smat(x, d); Y = to_smat(y, d)
    return to_svec((X * Y + Y * X) / 2)
end

# Jordan-algebra trace tr(z) = Σ eigenvalues, by cone.  Paired with the Euclidean
# dot it answers the design question ⟨x,y⟩ =? tr(x∘y):
#   POS: tr(z)=Σz, and ⟨x,y⟩=Σxy=tr(x∘y)         → factor 1
#   SDP: tr(z)=tr(smat z); svec is √2-scaled so ⟨svec X,svec Y⟩=tr(XY)=tr(X∘Y) → factor 1
#   SOC: tr(z)=2·z₁ (eigenvalues z₁±‖z̄‖); tr(x∘y)=2⟨x,y⟩  → factor 2 (= the degree)
jtrace(::POS, z) = sum(z)
jtrace(::SOC, z) = 2 * z[1]
jtrace(::SDP, z) = tr(to_smat(z, triroot(length(z))))

# strict interior membership
in_cone(::POS, x) = minimum(x) > 0
in_cone(::NOC, x) = true
in_cone(::SOC, x) = socdet(x) > 0 && x[1] > 0
function in_cone(::SDP, x)
    d = triroot(length(x))
    return eigmin(Symmetric(to_smat(x, d))) > 0
end

# random strictly-interior point
rand_interior(::POS, n) = rand(n) .+ 0.5
rand_interior(::NOC, n) = randn(n)
function rand_interior(::SOC, n)
    x̄ = randn(n - 1)
    return [norm(x̄) + 1.0; x̄]
end
function rand_interior(::SDP, n)
    d = triroot(n)
    M = randn(d, d)
    S = M * M'
    return to_svec((S + S') / 2 + d * I)
end

# pretty pass/fail
function report(io, name, val; tol)
    ok = val ≤ tol
    @printf(io, "  [%s] %-34s  %.3e   (tol %.0e)\n", ok ? "ok " : "XX!", name, val, tol)
    return ok
end

relnorm(a, b) = norm(a - b) / (norm(b) + eps())

# ----------------------------------------------------------------------
# layer 1a: svec / smat / skron  (SDP plumbing)
# ----------------------------------------------------------------------
function check_svec_skron(io, d; tol = 1e-10)
    Random.seed!(1)
    A = (X = randn(d, d); (X + X') / 2)
    Bm = (X = randn(d, d); (X + X') / 2)
    X = (Y = randn(d, d); (Y + Y') / 2)
    G = (Z = randn(d, d); (Z + Z') / 2)   # symmetric: skron! is only ever fed W⁻¹ (sym)
    n = d * (d + 1) ÷ 2

    println(io, "svec/skron (d=$d):")
    ok = true
    # isometry: ⟨svec A, svec B⟩ == tr(A B)
    ok &= report(io, "svec isometry", abs(dot(to_svec(A), to_svec(Bm)) - tr(A * Bm)); tol)
    # round trip
    ok &= report(io, "smat∘svec round trip", relnorm(to_smat(to_svec(A), d), A); tol)
    # symmetric Kronecker: skron(G)·svec(X) == svec(G X Gᵀ)
    H = zeros(n, n); skron!(H, G)
    lhs = Symmetric(H, :L) * to_svec(X)
    rhs = to_svec(G * X * G')
    ok &= report(io, "skron(G)·svec(X)=svec(GXGᵀ)", relnorm(lhs, rhs); tol)
    return ok
end

# ----------------------------------------------------------------------
# layer 1b: per-cone math at a random interior point
# ----------------------------------------------------------------------
function check_cone(io, cone, n; tol = 1e-8)
    Random.seed!(2)
    p = rand_interior(cone, n)
    d = (cone isa NOC) ? zeros(n) : rand_interior(cone, n)
    cv = single_cache(cone, n)

    println(io, "cone $(typeof(cone)) (embdim=$n):")
    ok = true

    # scaling + Hessian
    scale!(p, d, cv)
    H = zeros(n, n)
    hess!(H, p, d, cv)
    Hs = Symmetric(H, :L)

    if !(cone isa NOC)
        # *** NT scaling identity: H·p = d ***  (the big one)
        ok &= report(io, "NT identity  H·p = d", relnorm(Hs * p, d); tol)
        # H ≻ 0
        λmin = eigmin(Matrix(Hs))
        ok &= report(io, "H ≻ 0 (-eigmin)", max(0.0, -λmin); tol)
    else
        ok &= report(io, "NOC hess! = 0", norm(H); tol)
    end

    # corrector baseline: corr!(p,d,0,0,0) == -d   (sign / baseline)
    r  = zeros(n)
    z  = zeros(n)
    corr!(r, p, d, z, z, 0.0, cv)
    ok &= report(io, "corr!(·,0,0,0) = -d", relnorm(r, -d); tol = cone isa NOC ? 1e-10 : tol)

    # maxstep boundary: along Δx = -2x the cone is exited at τ = 1/2
    Δx = -2 .* p
    τ  = maxstep(p, Δx, true, 1.0, cv)
    expected = (cone isa NOC) ? 1.0 : 0.5
    ok &= report(io, "maxstep(-2x) = $(expected)", abs(τ - expected); tol)

    # identity element is idempotent under the Jordan product
    e = zeros(n); identity!(e, cone)
    if !(cone isa NOC)
        ok &= report(io, "identity idempotent  e∘e=e", relnorm(jmul(cone, e, e), e); tol)
    end

    # *** the canonical dot=trace check ***  The design invariant is ⟨x,y⟩ = tr(x∘y)
    # for EVERY cone — that is exactly what makes μ = ⟨p,d⟩/ν the correct barrier
    # Identity-norm check: dot(e,e)/ν should equal 1 when the stored Euclidean dot
    # equals the trace inner product (isometric coordinates).
    #   POS: ‖(1,…,1)‖²/n = 1.  SDP: ‖svec(I)‖²/d = 1.  SOC: old e=(1,0) gives 1/2;
    #   scaled e=(√2,0) gives 2/2=1.  This is the clean signal for the isometry fix.
    # (The old κ=tr(p∘d)/⟨p,d⟩ check is coordinate-invariant and reads 2 always.)
    if !(cone isa NOC)
        e = zeros(n); identity!(e, cone)
        ν = degree(cone, n)
        id_norm = dot(e, e) / ν
        ok &= report(io, "isometry  dot(e,e)/ν = 1", abs(id_norm - 1); tol)
        @printf(io, "       dot(e,e)/ν = %.4f   (cone degree ν = %d)\n", id_norm, ν)
    end

    return ok
end

# ----------------------------------------------------------------------
# layer 2: instrumented solver step  (mirror of SheafSDP.step!)
# ----------------------------------------------------------------------
Base.@kwdef mutable struct InvLog
    aff_assembly::Float64 = 0.0   # ‖H_cone·Δpa + Δda + d‖ — the Uzawa dual-eq residual floor
    cor_scalar::Float64 = 0.0     # corrector correctness bracket (floor-immune; the real catcher)
    cor_scalar_tr::Float64 = 0.0  # diagnostic: same bracket in the Jordan trace pairing (SOC ×2)
    cor_assembly::Float64 = 0.0   # ‖H_cone·Δp + Δd − corr_term‖ — dual-eq residual floor
    pos_vec::Float64 = 0.0        # exact POS Mehrotra vector identity
    nt_runtime::Float64 = 0.0     # ‖hcone_mul(p) − d‖/‖d‖ at the iterate (exact for POS; SOC/SDP boundary-sensitive)
    kkt_res::Float64 = 0.0        # KKT dual-eq residual ‖A·Δp − Bᵀ·Δy − f‖ (robust-normalized)
    kkt_pres::Float64 = 0.0       # KKT primal-eq residual ‖B·Δp − rp‖ (what the Schur CG controls)
    kkt_iters::Int = 0            # worst-case CG iteration count (≈itmax ⇒ CG not converging)
    member::Float64 = 0.0         # worst interior margin over iterates (≤0 = left cone)
    noc_dual::Float64 = 0.0       # ‖d on NOC blocks‖/‖d‖ — should be ~0 (free cone, dual={0})
    # decomposition captured AT the worst cor_scalar iteration (be = -σ/2·C + (1-σ/2)·N):
    pk_sigma::Float64 = 0.0       # σ there  (1/σ is the amplifier on the N term)
    pk_Ccone::Float64 = 0.0       # ⟨p,d⟩ over cone blocks
    pk_Nnoc::Float64  = 0.0       # ⟨p,d⟩ over NOC blocks (the contaminant)
    pk_B::Float64     = 0.0       # ⟨Δpa,Δda⟩ over cone blocks (other normalizer term)
    pk_be::Float64    = 0.0       # the bracket numerator itself
    # μ contamination: does NOC pollute the global dot(p,d) that defines μ?
    mu_noc_share::Float64 = 0.0   # |⟨p,d⟩_NOC| / |⟨p,d⟩| — should be ~0; if tenths, μ is contaminated
    mu_clean::Float64 = 0.0       # ⟨p,d⟩_cone / ν — μ with NOC excluded
    mu_used::Float64 = 0.0        # dot(p,d) / ν — the μ the solver actually uses
    d_noc_max::Float64 = 0.0      # max |d| on NOC blocks — the decisive instrument
    iters::Int = 0
end

upd!(l, f, v) = setfield!(l, f, max(getfield(l, f), v))

# H_cone · v   =   H·v − Q·v   (H = cone Hessian + Q, both block-diagonal)
function hcone_mul(s, v)
    a = similar(v); mul!(a, Symmetric(s.H, :L), v)
    b = similar(v); mul!(b, Symmetric(s.Q, :L), v)
    return a .- b
end

# Block inner product over the cones that carry complementarity (NOC, degree 0,
# is skipped — its free variables have d≡0 and an arbitrary affine cross term that
# is NOT part of any centering identity; sweeping it in via a global `dot` is what
# made the SOC Euclidean bracket read 24 instead of its true value).
#
# With isometric coordinates (SOC scaled by √2), the Euclidean dot equals the
# trace inner product, so trace=false is the correct corrector-arithmetic check.
# trace=true now double-counts (applies ×2 on already-scaled coordinates) and is
# RETIRED — kept only for informational/historical comparison.
function blockdot(s, x, y; trace::Bool)
    acc = 0.0
    for v in vtxs(s.B)
        s.cones[v] isa NOC && continue
        κ = (trace && s.cones[v] isa SOC) ? 2.0 : 1.0
        r = colrange(s.B, v)
        acc += κ * dot(view(x, r), view(y, r))
    end
    return acc
end

function diagnostic_step!(s::IPMSolver{T}, log::InvLog; bug::Symbol = :none) where {T}
    residuals!(s.rp, s.rd, s.B, s.p, s.d, s.y, s.c, s.g, s.Q)
    μ = iszero(s.ν) ? zero(T) : dot(s.p, s.d) / s.ν

    nrp = norm(s.rp) / (1 + norm(s.g))
    nrd = norm(s.rd) / (1 + norm(s.c))
    if nrp < s.settings.feas_tol && nrd < s.settings.feas_tol &&
       (iszero(s.ν) || μ < s.settings.gap_tol)
        s.status = OPTIMAL
        return false
    end

    block_hess!(s.H, s.caches, s.cones, s.p, s.d, s.B, s.Q)
    init_kkt!(s.wrk, s.settings.kkt, s.H)

    # NT identity at the ACTUAL iterate: hcone_mul(p) =? d.  Exact for POS (a
    # division), factorization-based and boundary-sensitive for SOC/SDP.  Reported
    # as nt_runtime; informational (the corrector bracket below is now floor-free
    # and does not depend on it).
    if !iszero(s.ν)
        upd!(log, :nt_runtime, norm(hcone_mul(s, s.p) .- s.d) / (norm(s.d) + eps()))
    end

    # --- affine direction ---
    axpby!(-1, s.d,  0, s.f)
    axpby!(-1, s.rd, 1, s.f)
    kkt_a = newton!(s.Δpa, s.Δya, s.Δda, s.wrk, s.settings.kkt, s.H, s.B, s.f, s.rp, s.rd, s.Q)

    # Complementarity identities are vacuous with no conic content (all-NOC):
    # d≡0, ⟨p,d⟩=0, ν=0 ⇒ 0/0.  Gate on ν>0.  rd-FREE (newton! recovers
    # Δd = rd−BᵀΔy+QΔp, step! passes f = corr_term−rd, so H_cone·Δp+Δd = corr_term).
    #
    # The assembly residual = the Uzawa DUAL-equation residual ‖A·Δp−BᵀΔy−f‖.
    # Augmented-Lagrangian Uzawa drives the PRIMAL eq BΔp=rp to ~machine but the
    # DUAL eq only to ≈α·‖B‖·√eps (= α·Bᵀ(rp−BΔp)).  So assembly sits on that
    # floor (e.g. ~3e-5 at raug=1000), NOT machine zero — that is expected.
    # The affine *scalar* ⟨p,Δda⟩+⟨d,Δpa⟩+⟨p,d⟩ equals ⟨p, assembly_resid⟩ exactly
    # (when nt_runtime≈0), i.e. it is the projection of that same residual — so it
    # is redundant with assembly, and normalizing it by the vanishing ⟨p,d⟩ only
    # manufactures a near-convergence false alarm.  Dropped; we report assembly.
    if !iszero(s.ν)
        upd!(log, :aff_assembly,
             norm(hcone_mul(s, s.Δpa) .+ s.Δda .+ s.d) / (norm(s.d) + norm(s.Δda) + eps()))
    end

    # --- centering parameter (verbatim from step!, guarded for ν=0) ---
    if iszero(s.ν)
        σ = zero(T)                      # no barrier ⇒ pure Newton, no centering
    else
        τpa, τda = maxsteps(s.p, s.d, s.Δpa, s.Δda, s.caches, s.cones, s.B; frac = one(T))
        pa = s.p + τpa * s.Δpa
        da = s.d + τda * s.Δda
        μa = dot(pa, da) / s.ν
        σ  = clamp((μa / μ)^3, zero(T), one(T))
    end

    # --- corrector direction ---
    # soccorr! OVERWRITES its Δp/Δd args (the affine directions) in place via
    # socroot! — a side effect of building corr_term in scaled space.  step! never
    # reuses Δpa/Δda afterward, but our bracket/POS checks below do, so snapshot the
    # clean affine directions first.  (POS/SDP corr! don't mutate; SOC does.)
    Δpa0 = copy(s.Δpa)
    Δda0 = copy(s.Δda)
    σμ_eff = (bug === :corrector_2sigma) ? 2 * σ * μ : σ * μ   # positive control
    corrector!(s.f, s.caches, s.cones, s.p, s.d, s.Δpa, s.Δda, σμ_eff, s.B)
    corr_term = copy(s.f)                       # capture before rd is folded in
    axpy!(-1, s.rd, s.f)
    kkt_c = newton!(s.Δp, s.Δy, s.Δd, s.wrk, s.settings.kkt, s.H, s.B, s.f, s.rp, s.rd, s.Q)
    upd!(log, :kkt_iters, max(kkt_a, kkt_c))    # CG iterations; near itmax ⇒ CG stalling

    # KKT solve residuals (the floor under every identity below). newton! solved
    #   A·Δp − Bᵀ·Δy = f   and   B·Δp = rp,   with A = s.H = H_cone + Q, Δy = −y_raw.
    # kkt_res is the dual-eq residual; kkt_pres is the primal-eq residual that the
    # Schur-complement CG drives to zero. If kkt_pres is tiny but kkt_res is not,
    # the dual residual is just α·Bᵀ(rp−BΔp) amplification, not a failed solve;
    # if BOTH are large AND kkt_iters≈itmax, the Uzawa CG genuinely did not converge.
    # GATED on nrp>feas_tol: once primal-feasible, rp→0 and ‖BΔp−rp‖/(…) is 0/0≈1
    # (an endgame artifact of the relative metric, NOT a solve failure).
    if nrp > s.settings.feas_tol
        let tmp = similar(s.Δp), tBy = similar(s.Δp), tg = similar(s.rp)
            mul!(tmp, Symmetric(s.H, :L), s.Δp)
            mul!(tBy, s.B', s.Δy)
            mul!(tg,  s.B, s.Δp)
            upd!(log, :kkt_res,  norm(tmp .- tBy .- s.f) / (norm(tmp) + norm(tBy) + norm(s.f) + eps()))
            upd!(log, :kkt_pres, norm(tg .- s.rp) / (norm(tg) + norm(s.rp) + eps()))
        end
    end

    if !iszero(s.ν)
        # corrector assembly residual (= the Uzawa dual-eq floor; informational):
        car = hcone_mul(s, s.Δp) .+ s.Δd .- corr_term
        upd!(log, :cor_assembly, norm(car) / (norm(corr_term) + norm(s.Δd) + eps()))

        # The corrector-correctness checks below divide by the centering scale
        # σ·⟨p,d⟩, which → 0 at convergence (σ→0); when it vanishes the floor-immune
        # bracket degenerates to nt_runtime/0 (the SOC 1e15 blow-up).  Gate on a
        # meaningful centering signal σμ > gap_tol so we only assert them where the
        # corrector actually does something.  Layer 3 (early iters, σ meaningful)
        # remains the definitive bug catcher.
        if σ * μ > s.settings.gap_tol
            # Corrector correctness, floor-FREE (uses only corr_term, the clean
            # affine snapshots, p, d, σ — never the solved directions, NT, or
            # assembly, so no floor can enter).  TWO forms:
            #
            #  cor_scalar  (pass/fail): EUCLIDEAN dot.  This encodes the design
            #    invariant ⟨x,y⟩ = tr(x∘y).  = 0 iff corr! is correct AND the cone
            #    satisfies that invariant.  POS/SDP satisfy it (svec's √2); if SOC
            #    does NOT, this is nonzero — a real finding, not noise.  Do not
            #    weight it away.
            #
            #  cor_scalar_tr (informational): same identity but with the Jordan
            #    TRACE pairing (SOC blocks ×2).  If cor_scalar fails while this is
            #    ~0, the corrector math is correct and the discrepancy is EXACTLY
            #    the missing trace normalization on SOC (the degree-2 factor) —
            #    i.e. SOC violates ⟨x,y⟩ = tr(x∘y), to be fixed in the solver
            #    (a √2-style block scaling, or the degree appearing in μ).
            pd_e = dot(s.p, s.d)
            Ccone = blockdot(s, s.p, s.d; trace=false)   # ⟨p,d⟩ over cone blocks
            Nnoc  = pd_e - Ccone                          # ⟨p,d⟩ over NOC blocks

            # NOC contamination of μ: how much of the global dot that defines μ
            # comes from degree-0 (NOC) blocks, which have no complementarity to contribute.
            upd!(log, :mu_noc_share, abs(Nnoc) / (abs(pd_e) + eps()))
            upd!(log, :mu_clean, Ccone / s.ν)             # μ with NOC excluded
            upd!(log, :mu_used, pd_e / s.ν)               # μ the solver actually uses

            Bcone = blockdot(s, Δpa0, Δda0; trace=false)
            # Corrector identity: ⟨p, corr_term⟩ + ⟨Δp_a, Δd_a⟩ = (σ-1)·⟨p,d⟩ over cone blocks.
            # The (σ-1) coefficient is correct because corr_term includes the -d baseline
            # (recall corr!(·,0,0,0) = -d), so ⟨p, corr_term⟩ picks up -⟨p,d⟩.
            # Use Ccone (cone-only ⟨p,d⟩) to exclude NOC blocks which have no complementarity.
            be = blockdot(s, s.p, corr_term; trace=false) + Bcone - (σ - 1) * Ccone
            cs = abs(be) / (abs(σ * pd_e) + abs(Bcone) + eps())
            if cs > log.cor_scalar                        # capture decomposition at the peak
                log.cor_scalar = cs
                log.pk_sigma = σ; log.pk_Ccone = Ccone; log.pk_Nnoc = Nnoc
                log.pk_B = Bcone; log.pk_be = be
            end

            bt = blockdot(s, s.p, corr_term; trace=true) +
                 blockdot(s, Δpa0, Δda0; trace=true) +
                 blockdot(s, s.p, s.d; trace=true) - σ * pd_e
            upd!(log, :cor_scalar_tr,
                 abs(bt) / (abs(σ * pd_e) + abs(blockdot(s, Δpa0, Δda0; trace=true)) + eps()))

            # POS Mehrotra vector identity, per component, made floor-immune by
            # subtracting p∘car (= the per-component dual-eq floor).  What remains
            # catches per-component corrector errors (σμ or cross-term) that can
            # cancel in the summed bracket:
            #   pᵢdᵢ + dᵢΔpᵢ + pᵢΔdᵢ + Δpa0ᵢ·Δda0ᵢ − σμ − pᵢ·carᵢ
            for v in vtxs(s.B)
                s.cones[v] isa POS || continue
                r    = colrange(s.B, v)
                pv   = view(s.p, r);   dv   = view(s.d, r)
                Δpv  = view(s.Δp, r);  Δdv  = view(s.Δd, r)
                Δav  = view(Δpa0, r);  Δdav = view(Δda0, r)
                carv = view(car, r)
                mehr  = pv .* dv .+ dv .* Δpv .+ pv .* Δdv .+ Δav .* Δdav .- σ * μ
                clean = mehr .- pv .* carv
                upd!(log, :pos_vec, maximum(abs, clean) / (abs(σ * μ) + maximum(abs, pv .* dv) + eps()))
            end
        end
    end

    # --- take the step ---
    τp, τd = maxsteps(s.p, s.d, s.Δp, s.Δd, s.caches, s.cones, s.B; frac = s.settings.step_frac)
    axpy!(τp, s.Δp, s.p)
    axpy!(τd, s.Δd, s.d)
    axpy!(τd, s.Δy, s.y)

    # cone membership of the new iterate
    for v in vtxs(s.B)
        cone = s.cones[v]
        cone isa NOC && continue
        r = colrange(s.B, v)
        margin_p = in_cone(cone, view(s.p, r)) ? 1.0 : -1.0
        margin_d = in_cone(cone, view(s.d, r)) ? 1.0 : -1.0
        upd!(log, :member, -min(margin_p, margin_d))   # >0 means a block left its cone
    end

    # NOC dual sanity: a free cone has dual cone {0}, so d should vanish on NOC
    # blocks.  A nonzero value contaminates the global μ = dot(p,d)/ν AND leaks an
    # unbounded ⟨p_NOC,d_NOC⟩ term into the Euclidean corrector bracket — this is the
    # source of an SOC bracket reading ≫ 0.5 (see the (1−σ/2)·N term).
    let nd = 0.0
        for v in vtxs(s.B)
            s.cones[v] isa NOC || continue
            r = colrange(s.B, v)
            nd += sum(abs2, view(s.d, r))
            upd!(log, :d_noc_max, maximum(abs, view(s.d, r)))
        end
        upd!(log, :noc_dual, sqrt(nd) / (norm(s.d) + eps()))
    end

    push!(s.hist, (μ = μ, τp = τp, τd = τd, rp = nrp, rd = nrd, kkt_iters = 0))
    s.iter += 1
    log.iters = s.iter

    if isstalled(s.hist; window = s.settings.stall_window, threshold = s.settings.stall_threshold)
        s.status = STALLED; return false
    end
    if isnumfail(s.hist; threshold = s.settings.step_collapse_threshold)
        s.status = NUMERICAL_FAILURE; return false
    end
    return s.iter < s.settings.itmax
end

function check_solver(io, prob, settings; label = "problem", tol = 1e-4)
    println(io, "\nsolver: $label")
    s = init(prob, settings)
    log = InvLog()
    while diagnostic_step!(s, log) end

    # final point, unpermuted, against the *original* problem data
    p = s.P \ s.p; d = s.P \ s.d; y = s.y
    B, Q, c, g = prob.B, prob.Q, prob.c, prob.g
    Qp = Symmetric(Q, :L) * p
    rp = g - B * p
    rd = c - d + Qp - B' * y
    gap   = dot(p, d)
    duality = abs(gap - (dot(p, Qp) + dot(c, p) - dot(g, y))) / (abs(gap) + eps())

    @printf(io, "  status = %s  iters = %d\n", s.status, log.iters)
    ok = true
    # --- correctness assertions (pass/fail) ---
    # The corrector-arithmetic catcher is the TRACE-pairing bracket: it is degree-correct
    # With isometric coordinates (SOC scaled by √2), the stored Euclidean dot equals the
    # trace inner product, so the Euclidean bracket is the correct corrector-arithmetic
    # check.  The trace-weighted bracket now double-counts the factor the coordinates
    # already carry and is retired.  This catches the 2σμ control.
    ok &= report(io, "corrector arithmetic (Euclidean)", log.cor_scalar; tol)
    ok &= report(io, "POS Mehrotra vector id (clean)",  log.pos_vec;      tol)
    ok &= report(io, "iterates stayed in cone",       log.member;       tol = 0.0)
    ok &= report(io, "final primal feas ‖g-Bp‖",      norm(rp) / (1 + norm(g)); tol = 10 * settings.feas_tol)
    ok &= report(io, "final dual feas ‖rd‖",          norm(rd) / (1 + norm(c)); tol = 10 * settings.feas_tol)
    ok &= report(io, "final gap ⟨p,d⟩",               abs(gap);                 tol = 10 * settings.gap_tol * max(1, s.ν))
    ok &= report(io, "duality id ⟨p,d⟩=pQp+cp-gy",    duality;                  tol)
    # --- informational floors (NOT pass/fail): the assembly residuals are the
    #     Uzawa dual-eq floor (≈α√eps); a few-e-2..e-5 reading is expected, benign,
    #     and is the same quantity as KKT dual-eq res — not a correctness signal. ---
    @printf(io, "  [..] %-34s  %.3e   (Uzawa dual-eq floor)\n",  "affine assembly residual",       log.aff_assembly)
    @printf(io, "  [..] %-34s  %.3e   (Uzawa dual-eq floor)\n",  "corrector assembly residual",    log.cor_assembly)
    @printf(io, "  [..] %-34s  %.3e   (NT id at iterate; exact for POS)\n", "NT runtime ‖H_cone·p−d‖", log.nt_runtime)
    @printf(io, "  [..] %-34s  %.3e   (RETIRED: trace-weighted now double-counts)\n", "corrector bracket (trace)", log.cor_scalar_tr)
    @printf(io, "  [..] %-34s  %.3e   (should be ~0; if not, μ & euclid-be contaminated)\n", "NOC dual ‖d_NOC‖/‖d‖", log.noc_dual)
    @printf(io, "  [..] %-34s  %.3e   (THE decisive instrument: ~0 ⇒ no contamination)\n", "max |d| on NOC blocks", log.d_noc_max)
    @printf(io, "  [..] %-34s  %.3e   (NOC share of dot(p,d); ≫0 ⇒ μ contaminated)\n", "μ NOC share |⟨p,d⟩_NOC|/|⟨p,d⟩|", log.mu_noc_share)
    @printf(io, "  [..] μ_used=%.3e  μ_clean=%.3e  ratio=%.4f\n", log.mu_used, log.mu_clean,
            log.mu_clean > 0 ? log.mu_used / log.mu_clean : 0.0)
    @printf(io, "  [..] euclid-bracket peak:  σ=%.3e  ⟨p,d⟩cone=%.3e  ⟨p,d⟩NOC=%.3e  ⟨Δpa,Δda⟩cone=%.3e  be=%.3e\n",
            log.pk_sigma, log.pk_Ccone, log.pk_Nnoc, log.pk_B, log.pk_be)
    @printf(io, "  [..]   model be=-σ/2·C+(1-σ/2)·N = %.3e ;  1/σ amplifier = %.1f\n",
            -log.pk_sigma/2*log.pk_Ccone + (1-log.pk_sigma/2)*log.pk_Nnoc,
            log.pk_sigma > 0 ? 1/log.pk_sigma : 0.0)
    @printf(io, "  [..] %-34s  %.3e   (= dual-eq floor, α·√eps)\n", "KKT dual-eq res ‖AΔp-BᵀΔy-f‖", log.kkt_res)
    @printf(io, "  [..] %-34s  %.3e   (CG-controlled, →machine)\n", "KKT primal-eq res ‖BΔp-rp‖", log.kkt_pres)
    @printf(io, "  [..] %-34s  %d   (itmax=%d ⇒ CG stalled)\n", "worst CG iterations", log.kkt_iters, s.settings.kkt.itmax)
    return ok
end

# Positive control: deliberately inject a wrong corrector (2σμ instead of σμ —
# the classic "still converges but wrong" bug) and CONFIRM the corrector scalar
# identity catches it.  If this fails, the scalar check has no teeth and any
# green run above is meaningless.  Needs a problem with conic content (σ>0).
function check_corrector_detection(io, prob, settings; label = "problem")
    println(io, "\npositive control — inject 2σμ corrector bug: $label")
    s = init(prob, settings)
    log = InvLog()
    n = 0
    while n < 12 && diagnostic_step!(s, log; bug = :corrector_2sigma)
        n += 1
    end
    @printf(io, "  cor_scalar under injected bug = %.3e   (must be ≫ solver tol)\n", log.cor_scalar)
    detected = log.cor_scalar > 1e-2
    return report(io, "injected corrector bug IS caught", detected ? 0.0 : 1.0; tol = 0.5)
end

# ----------------------------------------------------------------------
# problem builders
# ----------------------------------------------------------------------

# NOC-only QP (adapted from test/qp.jl, known-good)
function build_qp(N, T)
    Random.seed!(42)
    nx, nu, h = 4, 2, 0.1
    A = [1.0 0 h 0; 0 1 0 h; 0 0 1 0; 0 0 0 1]
    Bd = [0.0 0; 0 0; h 0; 0 h]
    P = [1.0 0 0 0; 0 1 0 0]
    R, ε = Matrix(1.0I, nu, nu), 0.01
    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, j) for i in 1:N for j in i+1:N]

    cx(i, t) = (i - 1) * (T + T - 1) + t
    cu(i, t) = (i - 1) * (T + T - 1) + T + t
    ri(i)    = (i - 1) * T + 1
    rd(i, t) = (i - 1) * T + 1 + t
    rc(e)    = N * T + e

    rid, cid, blk = Int[], Int[], Matrix{Float64}[]
    for i in 1:N
        push!(rid, ri(i)); push!(cid, cx(i, 1)); push!(blk, Matrix(1.0I, nx, nx))
        for t in 1:T-1
            push!(rid, rd(i, t)); push!(cid, cx(i, t));     push!(blk, -A)
            push!(rid, rd(i, t)); push!(cid, cx(i, t + 1)); push!(blk, Matrix(1.0I, nx, nx))
            push!(rid, rd(i, t)); push!(cid, cu(i, t));     push!(blk, -Bd)
        end
    end
    for (e, (i, j)) in enumerate(edges)
        push!(rid, rc(e)); push!(cid, cx(i, T)); push!(blk, -P)
        push!(rid, rc(e)); push!(cid, cx(j, T)); push!(blk, P)
    end
    Bm = blocksparse(rid, cid, blk)
    c = zeros(size(Bm, 2)); g = zeros(size(Bm, 1))
    for i in 1:N; g[rowrange(Bm, ri(i))] .= x0[i]; end
    Q = allocblockdiag(Bm); fill!(Q, 0)
    for i in 1:N
        for t in 1:T-1
            Qv = block(Q, cu(i, t), cu(i, t), cu(i, t)); for k in 1:nu; Qv[k, k] = 2R[k, k]; end
        end
        Qv = block(Q, cx(i, T), cx(i, T), cx(i, T)); for k in 1:nx; Qv[k, k] = 2ε; end
    end
    cones = [:NOC for _ in 1:N*(T + T - 1)]
    prob = IPMProblem(c, g, Bm, Q, cones)
    settings = IPMSettings{Float64}(kkt = UzawaSettings{Float64}(raug = 1e9),
                                    feas_tol = 1e-8, gap_tol = 1e-8, itmax = 100)
    return prob, settings
end

# POS + NOC consensus ℓ1 (adapted from test/pos.jl, known-good)
function build_pos(N, T)
    Random.seed!(42)
    nx, nu, h, ū = 4, 2, 0.1, 100.0
    A = [1.0 0 h 0; 0 1 0 h; 0 0 1 0; 0 0 0 1]
    Bd = [0.0 0; 0 0; h 0; 0 h]
    P = [1.0 0 0 0; 0 1 0 0]
    x0 = [randn(nx) for _ in 1:N]
    edges = [(i, j) for i in 1:N for j in i+1:N]
    bpa = T + 3 * (T - 1)
    cx(i, t)  = (i - 1) * bpa + t
    cup(i, t) = (i - 1) * bpa + T + 3 * (t - 1) + 1
    cum(i, t) = (i - 1) * bpa + T + 3 * (t - 1) + 2
    cw(i, t)  = (i - 1) * bpa + T + 3 * (t - 1) + 3
    rpa = 2T - 1
    ri(i)    = (i - 1) * rpa + 1
    rd(i, t) = (i - 1) * rpa + 1 + t
    rb(i, t) = (i - 1) * rpa + T + t
    rc(e)    = N * rpa + e

    rid, cid, blk = Int[], Int[], Matrix{Float64}[]
    for i in 1:N
        push!(rid, ri(i)); push!(cid, cx(i, 1)); push!(blk, Matrix(1.0I, nx, nx))
        for t in 1:T-1
            push!(rid, rd(i, t)); push!(cid, cx(i, t));     push!(blk, -A)
            push!(rid, rd(i, t)); push!(cid, cx(i, t + 1)); push!(blk, Matrix(1.0I, nx, nx))
            push!(rid, rd(i, t)); push!(cid, cup(i, t));    push!(blk, -Bd)
            push!(rid, rd(i, t)); push!(cid, cum(i, t));    push!(blk, Bd)
            push!(rid, rb(i, t)); push!(cid, cup(i, t));    push!(blk, Matrix(1.0I, nu, nu))
            push!(rid, rb(i, t)); push!(cid, cum(i, t));    push!(blk, Matrix(1.0I, nu, nu))
            push!(rid, rb(i, t)); push!(cid, cw(i, t));     push!(blk, Matrix(1.0I, nu, nu))
        end
    end
    for (e, (i, j)) in enumerate(edges)
        push!(rid, rc(e)); push!(cid, cx(i, T)); push!(blk, -P)
        push!(rid, rc(e)); push!(cid, cx(j, T)); push!(blk, P)
    end
    Bm = blocksparse(rid, cid, blk)
    c = zeros(size(Bm, 2)); g = zeros(size(Bm, 1))
    for i in 1:N, t in 1:T-1
        c[colrange(Bm, cup(i, t))] .= 1.0
        c[colrange(Bm, cum(i, t))] .= 1.0
    end
    for i in 1:N
        g[rowrange(Bm, ri(i))] .= x0[i]
        for t in 1:T-1; g[rowrange(Bm, rb(i, t))] .= ū; end
    end
    Q = allocblockdiag(Bm); fill!(Q, 0)
    cones = Vector{Symbol}(undef, N * bpa)
    for i in 1:N
        for t in 1:T; cones[cx(i, t)] = :NOC; end
        for t in 1:T-1
            cones[cup(i, t)] = :POS; cones[cum(i, t)] = :POS; cones[cw(i, t)] = :POS
        end
    end
    prob = IPMProblem(c, g, Bm, Q, cones)
    settings = IPMSettings{Float64}(kkt = UzawaSettings{Float64}(raug = 1000.0),
                                    feas_tol = 1e-7, gap_tol = 1e-7, itmax = 100)
    return prob, settings
end

# SOC: minimize ‖x - b‖ over x∈R^k subject to 1ᵀx = s, modeled with a
# Lorentz block z=(z0,z̄), z̄ = x - b, min z0.  x is NOC with tiny reg.
function build_soc(k)
    Random.seed!(7)
    b = randn(k); s = sum(b) + 1.0
    # columns: 1 = z (SOC, dim k+1), 2 = x (NOC, dim k)
    cz, cx = 1, 2
    # rows: 1..k  -> z̄ - x = -b ;  k+1 -> 1ᵀx = s
    rEq, rSum = 1, 2
    # Scale SOC columns by 1/√2 for isometric representation
    invrt2 = 1 / sqrt(2.0)
    Sel = hcat(zeros(k), Matrix(1.0I, k, k)) .* invrt2   # k×(k+1): picks z̄ from z, scaled
    rid, cid, blk = Int[], Int[], Matrix{Float64}[]
    push!(rid, rEq);  push!(cid, cz); push!(blk, Sel)
    push!(rid, rEq);  push!(cid, cx); push!(blk, -Matrix(1.0I, k, k))
    push!(rid, rSum); push!(cid, cx); push!(blk, ones(1, k))
    Bm = blocksparse(rid, cid, blk)
    c = zeros(size(Bm, 2)); c[colrange(Bm, cz)[1]] = invrt2   # minimize z0, scaled
    g = zeros(size(Bm, 1))
    g[rowrange(Bm, rEq)] .= -b
    g[rowrange(Bm, rSum)] .= s
    Q = allocblockdiag(Bm); fill!(Q, 0)
    # No regularization on NOC - problem is bounded by the SOC constraint z0 ≥ ‖z̄‖
    # Qx = block(Q, cx, cx, cx); for i in 1:k; Qx[i, i] = 0; end   # Q=0 to eliminate contamination
    cones = [:SOC, :NOC]
    prob = IPMProblem(c, g, Bm, Q, cones)
    settings = IPMSettings{Float64}(kkt = UzawaSettings{Float64}(raug = 1.0),
                                    feas_tol = 1e-8, gap_tol = 1e-8, itmax = 100)
    return prob, settings
end

# SDP: min ⟨C,X⟩ s.t. tr(X)=τ, ⟨A2,X⟩=b2, ⟨A3,X⟩=b3, X∈S^d_+
# row blocks are svec(Ai)' (1×n); single SDP column block.
function build_sdp(d)
    Random.seed!(11)
    n = d * (d + 1) ÷ 2
    M = randn(d, d); X0 = M * M' + d * I           # feasible interior point
    C  = (G = randn(d, d); (G + G') / 2)
    A2 = (G = randn(d, d); (G + G') / 2)
    A3 = (G = randn(d, d); (G + G') / 2)
    As = [Matrix(1.0I, d, d), A2, A3]               # A1 = I  -> trace constraint
    rid, cid, blk = Int[], Int[], Matrix{Float64}[]
    for (i, Ai) in enumerate(As)
        push!(rid, i); push!(cid, 1); push!(blk, collect(reshape(to_svec(Ai), 1, n)))
    end
    Bm = blocksparse(rid, cid, blk)
    c = to_svec(C)
    g = [dot(to_svec(Ai), to_svec(X0)) for Ai in As]
    Q = allocblockdiag(Bm); fill!(Q, 0)
    cones = [:SDP]
    prob = IPMProblem(c, g, Bm, Q, cones)
    settings = IPMSettings{Float64}(kkt = UzawaSettings{Float64}(raug = 1.0),
                                    feas_tol = 1e-8, gap_tol = 1e-8, itmax = 100)
    return prob, settings
end

# ----------------------------------------------------------------------
# run everything
# ----------------------------------------------------------------------
function main(io = stdout)
    all_ok = true
    println(io, "="^64)
    println(io, "LAYER 1  —  cone math (random interior points)")
    println(io, "="^64)
    all_ok &= check_svec_skron(io, 3)
    all_ok &= check_svec_skron(io, 5)
    all_ok &= check_cone(io, POS(), 6)
    all_ok &= check_cone(io, SOC(), 5)
    all_ok &= check_cone(io, SDP(), 6)     # d=3
    all_ok &= check_cone(io, SDP(), 10)    # d=4
    all_ok &= check_cone(io, NOC(), 4)

    println(io, "\n" * "="^64)
    println(io, "LAYER 2  —  instrumented solver on real problems")
    println(io, "="^64)
    all_ok &= check_solver(io, build_qp(15, 15)...;  label = "QP   (NOC)")
    all_ok &= check_solver(io, build_pos(12, 12)...; label = "ℓ1   (POS+NOC)")
    all_ok &= check_solver(io, build_soc(6)...;      label = "SOC  (+NOC)")
    all_ok &= check_solver(io, build_sdp(3)...;      label = "SDP")
    all_ok &= check_solver(io, build_sdp(4)...;      label = "SDP (d=4)")

    println(io, "\n" * "="^64)
    println(io, "LAYER 3  —  positive control (the checks must have teeth)")
    println(io, "="^64)
    all_ok &= check_corrector_detection(io, build_pos(12, 12)...; label = "ℓ1   (POS+NOC)")
    all_ok &= check_corrector_detection(io, build_soc(6)...;      label = "SOC  (+NOC)")
    all_ok &= check_corrector_detection(io, build_sdp(3)...;      label = "SDP")

    println(io, "\n" * "="^64)
    println(io, all_ok ? "ALL INVARIANTS PASSED" : "SOME INVARIANTS FAILED  (look for XX! above)")
    println(io, "="^64)
    return all_ok
end

main()
