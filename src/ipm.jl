@enum IPMStatus CONTINUE OPTIMAL NEAR_OPTIMAL STALLED NUMERICAL_FAILURE ITERATION_LIMIT

struct IPMProblem{T, I, C <: AbstractCone}
    Q::BlockSparseMatrix{T, I}
    B::BlockSparseMatrix{T, I}
    c::FVector{T}
    g::FVector{T}
    cones::FVector{C}

    function IPMProblem(Q::BlockSparseMatrix{T, I}, B::BlockSparseMatrix{T, I}, c::FVector{T}, g::FVector{T}, cones::FVector{C}) where {T, I, C <: AbstractCone}
        @assert nrows(B) == length(g)
        @assert ncols(B) == ncols(Q) == length(c)
        @assert nvtxs(B) == nvtxs(Q) == length(cones)

        for v in vtxs(B)
            @assert ncols(B, v) == ncols(Q, v)
        end

        return new{T, I, C}(Q, B, c, g, cones)
    end
end

function IPMProblem(Q::BlockSparseMatrix, B::BlockSparseMatrix, c::AbstractVector, g::AbstractVector, cones::AbstractVector)
    c = FVector(c)
    g = FVector(g)
    cones = FVector(cones)
    return IPMProblem(Q, B, c, g, cones)
end

@kwdef struct IPMSettings{T}
    kkt::KKTSettings{T} = UzawaSettings{T}()
    step_frac::T = 0.99
    feas_tol::T = 1e-8
    gap_tol::T = 1e-8
    itmax::Int = 100
    verbose::Bool = false
    near_factor::T = 1000.0
    step_collapse_threshold::T = 1e-6
    refine_itmax::Int = 10
    refine_atol::T = 1e-12
    refine_rtol::T = 1e-13
    scale_itmax::Int = 10
end

struct IPMResult{T}
    p::Vector{T}
    d::Vector{T}
    y::Vector{T}
    status::IPMStatus
    ipm_niter::Int
    kkt_niter::Int
    history::History{T}
end

struct IPMWorkspace{T}
    # residuals
    rp::FVector{T}
    rd::FVector{T}
    # Newton RHS
    f::FVector{T}
    # affine directions
    Δpa::FVector{T}
    Δya::FVector{T}
    Δda::FVector{T}
    # corrector directions
    Δp::FVector{T}
    Δy::FVector{T}
    Δd::FVector{T}
    # refinement workspace
    sp::FVector{T}
    sy::FVector{T}
    dp::FVector{T}
    dy::FVector{T}
end

function IPMWorkspace{T}(m::Integer, n::Integer) where {T}
    return IPMWorkspace{T}(
        FVector{T}(undef, m),  # rp
        FVector{T}(undef, n),  # rd
        FVector{T}(undef, n),  # f
        FVector{T}(undef, n),  # Δpa
        FVector{T}(undef, m),  # Δya
        FVector{T}(undef, n),  # Δda
        FVector{T}(undef, n),  # Δp
        FVector{T}(undef, m),  # Δy
        FVector{T}(undef, n),  # Δd
        FVector{T}(undef, n),  # sp
        FVector{T}(undef, m),  # sy
        FVector{T}(undef, n),  # dp
        FVector{T}(undef, m),  # dy
    )
end

struct IPMSolver{T, I, K, C}
    Q::BlockSparseMatrix{T, I}
    H::BlockSparseMatrix{T, I}
    B::BlockSparseMatrix{T, I}
    c::FVector{T}
    g::FVector{T}
    p::FVector{T}
    d::FVector{T}
    y::FVector{T}

    # cones
    cones::FVector{C}

    # scaling
    scaling::Scaling{T}

    # permutation
    P::FPermutation{I}

    # workspace
    wrk::IPMWorkspace{T}
    caches::Caches{T, I}
    conewrk::ConeWorkspace{T}
    kkt::K

    # state
    hist::History{T}
    ν::Int

    # settings
    settings::IPMSettings{T}
end

function IPMResult(s::IPMSolver{T}, status::IPMStatus) where {T}
    p = Vector{T}(undef, length(s.p))
    d = Vector{T}(undef, length(s.d))
    y = Vector{T}(undef, length(s.y))

    ldiv!(p, s.P, s.p)
    ldiv!(d, s.P, s.d)
    copyto!(y, s.y)

    unscale!(p, d, y, s.scaling)

    ipm_niter = 0
    kkt_niter = 0

    for row in s.hist
        ipm_niter += 1
        kkt_niter += row.niter
    end

    return IPMResult{T}(p, d, y, status, ipm_niter, kkt_niter, s.hist)
end

function conedegree(cones::AbstractVector, B::BlockSparseMatrix)
    ν = 0

    for v in vtxs(B)
        ν += degree(cones[v], ncols(B, v))
    end

    return ν
end

function residuals!(
        rp::AbstractVector,
        rd::AbstractVector,
        B::BlockSparseMatrix,
        p::AbstractVector,
        d::AbstractVector,
        y::AbstractVector,
        c::AbstractVector,
        g::AbstractVector,
        Q::BlockSparseMatrix,
    )
    #
    # compute the primal residual:
    #
    #   rp = g - B p
    #
    copyto!(rp, g)
    mul!(rp, B, p, -1, 1)
    #
    # compute the dual residual:
    #
    #   rd =  c - d + Q p - Bᵀ y
    #
    copyto!(rd, c)
    mul!(rd, Symmetric(Q, :L), p,  1, 1)
    mul!(rd,           B',     y, -1, 1)
    axpy!(-1, d, rd)

    return rp, rd
end

function scale!(
        cone::AbstractCone,
        v::Integer,
        H::BlockSparseMatrix,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        B::BlockSparseMatrix,
        Q::BlockSparseMatrix,
        conewrk::ConeWorkspace,
    )
    r = colrange(B, v)
    Hv = block(H, v, v, v)
    cv = cache(caches, v, cone)
    scale!(Hv, view(p, r), view(d, r), cv, conewrk)
    axpy!(true, block(Q, v, v, v), Hv)
    return
end

function newton!(
        Δp::AbstractVector,
        Δy::AbstractVector,
        Δd::AbstractVector,
        wrk::KKTWorkspace,
        set::KKTSettings,
        H::BlockSparseMatrix,
        B::BlockSparseMatrix,
        f::AbstractVector,
        rp::AbstractVector,
        rd::AbstractVector,
        Q::BlockSparseMatrix,
        sp::AbstractVector,
        sy::AbstractVector,
        dp::AbstractVector,
        dy::AbstractVector,
        y0 = nothing;
        itmax::Integer = 0,
        atol::Real = 1e-12,
        rtol::Real = 1e-13,
    )
    kkt_iters = solve_kkt!(wrk, set, Δp, Δy, H, B, f, rp, y0)

    if itmax > 0
        kkt_iters += refine_kkt!(Δp, Δy, wrk, set, H, B, f, rp, sp, sy, dp, dy; itmax, atol, rtol)
    end

    copyto!(Δd, rd)
    mul!(Δd, B', Δy, -1, 1)
    mul!(Δd, Symmetric(Q, :L), Δp, 1, 1)

    return kkt_iters
end

function corrector!(
        cone::AbstractCone,
        v::Integer,
        f::AbstractVector,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        σμ::Real,
        B::BlockSparseMatrix,
        conewrk::ConeWorkspace,
    )
    r = colrange(B, v)
    cv = cache(caches, v, cone)
    corr!(view(f, r), view(p, r), view(d, r), view(Δp, r), view(Δd, r), σμ, cv, conewrk)
    return
end

function maxsteps(
        cone::AbstractCone,
        v::Integer,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        caches::Caches,
        B::BlockSparseMatrix,
        conewrk::ConeWorkspace,
    )
    r = colrange(B, v)
    cv = cache(caches, v, cone)
    return maxsteps(view(p, r), view(Δp, r), view(d, r), view(Δd, r), cv, conewrk)
end

function startingpoint(B::BlockSparseMatrix{T}, g::AbstractVector{T}, c::AbstractVector{T}, cones::AbstractVector) where {T}
    m, n = size(B)

    p = FVector{T}(undef, n)
    d = FVector{T}(undef, n)
    y = FVector{T}(undef, m)
    z = FVector{T}(undef, m)

    for v in vtxs(B)
        r = colrange(B, v)
        identity!(view(p, r), cones[v])
        identity!(view(d, r), cones[v])
    end

    mul!(z, B, p)

    np = norm(p)
    nz = norm(z)

    if nz > eps(T) * np
        ξ = max(one(T), norm(g) / nz)
    else
        ξ = one(T)
    end

    if np > eps(T)
        η = max(one(T), norm(c) / np)
    else
        η = one(T)
    end

    lmul!(ξ, p)
    lmul!(η, d)

    fill!(y, zero(T))

    return p, d, y
end

function isstalled(hist::History{T}; window=5, threshold=0.99) where {T}
    n = length(hist)

    if n < window + 1
        return false
    end

    floor = √eps(T)

    for i in n - window + 1:n
        if hist.μ[i] < threshold * hist.μ[i - 1]
            return false
        end

        if hist.pres[i - 1] > floor && hist.pres[i] < threshold * hist.pres[i - 1]
            return false
        end

        if hist.dres[i - 1] > floor && hist.dres[i] < threshold * hist.dres[i - 1]
            return false
        end
    end

    return true
end

function isnearoptimal(hist::History; feas_tol, gap_tol, near_factor)
    if isempty(hist)
        return false
    end

    μ  = hist.μ[end]
    rp = hist.pres[end]
    rd = hist.dres[end]

    return rp < near_factor * feas_tol && rd < near_factor * feas_tol && μ < near_factor * gap_tol
end

function isnumfail(hist::History; window=3, threshold=1e-6)
    if length(hist.pstep) < window
        return false
    end

    τavg = sum(hist.pstep[end-window+1:end]) / window
    τavg = min(τavg, sum(hist.dstep[end-window+1:end]) / window)

    if τavg > threshold
        return false
    end

    if length(hist.pres) < window + 1
        return true
    end

    return hist.pres[end] > 0.9 * hist.pres[end - window] || hist.dres[end] > 0.9 * hist.dres[end - window]
end

function CommonSolve.init(prob::IPMProblem{T, I}, settings::IPMSettings{T}=IPMSettings{T}()) where {T, I}
    n = size(prob.B, 2)
    m = size(prob.B, 1)
    ν = conedegree(prob.cones, prob.B)
    #
    # equilibrate problem data
    #
    scaling = Scaling{T}(n, m)

    if settings.scale_itmax > 0
        B = copy(prob.B)
        Q = copy(prob.Q)
        c = copy(prob.c)
        g = copy(prob.g)

        equilibrate!(scaling, B, Q, c, g; itmax=settings.scale_itmax)
    else
        B = prob.B
        Q = prob.Q
        c = prob.c
        g = prob.g
    end
    #
    # initialize kkt solver
    #
    R, P, B, kkt = make_kkt(settings.kkt, B)
    #
    # permute problem data
    #
    c = P * c
    Q = halfselectvtxs(halfselectvtxs(Q, R.perm), R.perm)
    cones = tounion(prob.cones, R.perm)
    #
    # compute starting point
    #
    p, d, y = startingpoint(B, g, c, cones)
    #
    # initialize per-cone caches
    #
    caches = Caches(cones, B)

    for v in vtxs(B)
        initcache!(cache(caches, v, cones[v]))
    end

    H = allocblockdiag(B)
    conewrk = ConeWorkspace{T}(cones, B)
    ipmwrk = IPMWorkspace{T}(m, n)
    hist = History{T}()

    return IPMSolver(Q, H, B, c, g, p, d, y, cones,
        scaling, P, ipmwrk, caches, conewrk, kkt,
        hist, ν, settings
    )
end

function step!(s::IPMSolver{T}) where {T}
    w = s.wrk
    #
    # compute the primal and dual residuals:
    #
    #   rp = g - B p
    #   rd = c - d + Q p - Bᵀ y
    #
    residuals!(w.rp, w.rd, s.B, s.p, s.d, s.y, s.c, s.g, s.Q)

    if iszero(s.ν)
        μ = zero(T)
    else
        μ = dot(s.p, s.d) / s.ν
    end
    #
    # iterative refinement only in endgame
    #
    if μ < 100 * s.settings.gap_tol
        refine_itmax = s.settings.refine_itmax
    else
        refine_itmax = 0
    end
    #
    # check the quantities
    #
    #   - ‖rp‖ / (1 + ‖g‖)
    #   - ‖rd‖ / (1 + ‖c‖)
    #   - μ
    #
    # for convergence
    #
    nrp = norm(w.rp) / (1 + norm(s.g))
    nrd = norm(w.rd) / (1 + norm(s.c))

    if nrp < s.settings.feas_tol && nrd < s.settings.feas_tol && (iszero(s.ν) || μ < s.settings.gap_tol)
        return OPTIMAL
    end
    #
    # compute the sum
    #
    #   H = Hf(w) + Q
    #
    # where f is the barrier function and
    # w is the scaling point.
    #
    for v in vtxs(s.B)
        scale!(s.cones[v], v, s.H, s.caches, s.p, s.d, s.B, s.Q, s.conewrk)
    end

    if !init_kkt!(s.kkt, s.settings.kkt, s.H)
        if s.settings.verbose
            println("Warning: KKT factorization failed")
        end

        return NUMERICAL_FAILURE
    end
    #
    # compute the affine direction (Δpa, Δya, Δda)
    # by solving for Δpa, Δya in
    #
    #   [ H -Bᵀ ] [ Δpa ] = [ -d - rd ]
    #   [ B  0  ] [ Δya ]   [ rp      ]
    #
    # and setting Δda to the value
    #
    #   Δda = rd - Bᵀ Δya + Q Δpa
    #
    axpby!(-1, s.d,  0, w.f)
    axpby!(-1, w.rd, 1, w.f)

    kkt_iters_aff = newton!(w.Δpa, w.Δya, w.Δda, s.kkt, s.settings.kkt, s.H, s.B, w.f, w.rp, w.rd, s.Q, w.sp, w.sy, w.dp, w.dy;
                            itmax=refine_itmax, atol=s.settings.refine_atol, rtol=s.settings.refine_rtol)
    #
    # compute the centering parameter
    #
    #   σμ ∈ [0, μ]
    #
    τpa = one(T)
    τda = one(T)

    for v in vtxs(s.B)
        τpv, τdv = maxsteps(s.cones[v], v, s.p, s.d, w.Δpa, w.Δda, s.caches, s.B, s.conewrk)
        τpa = min(τpa, τpv)
        τda = min(τda, τdv)
    end

    σμ = zero(T)

    for j in cols(s.B)
        σμ += (s.p[j] + τpa * w.Δpa[j]) * (s.d[j] + τda * w.Δda[j])
    end

    σμ /= s.ν
    σμ  = clamp(σμ * (σμ / μ)^2, zero(T), μ)
    #
    # solve for the corrector direction (Δp, Δy, Δd)
    # by solving for Δp, Δy in
    #
    #   [ H -Bᵀ ] [ Δp ] = [ -d - rd + σμ e - Δpa ∘ Δda ]
    #   [ B  0  ] [ Δy ]   [ rp                         ]
    #
    # and setting Δd to the value
    #
    #   Δd = rd - Bᵀ Δy + Q Δp
    #
    for v in vtxs(s.B)
        corrector!(s.cones[v], v, w.f, s.caches, s.p, s.d, w.Δpa, w.Δda, σμ, s.B, s.conewrk)
    end

    axpy!(-1, w.rd, w.f)

    kkt_iters_corr = newton!(w.Δp, w.Δy, w.Δd, s.kkt, s.settings.kkt, s.H, s.B, w.f, w.rp, w.rd, s.Q, w.sp, w.sy, w.dp, w.dy, w.Δya;
                             itmax=refine_itmax, atol=s.settings.refine_atol, rtol=s.settings.refine_rtol)

    kkt_iters = kkt_iters_aff + kkt_iters_corr
    #
    # take a step in the direction
    #
    #   (Δp, Δy, Δd)
    #
    τp, τd = one(T), one(T)

    for v in vtxs(s.B)
        τpv, τdv = maxsteps(s.cones[v], v, s.p, s.d, w.Δp, w.Δd, s.caches, s.B, s.conewrk)
        τp = min(τp, τpv)
        τd = min(τd, τdv)
    end

    τp *= s.settings.step_frac
    τd *= s.settings.step_frac

    axpy!(τp, w.Δp, s.p)
    axpy!(τd, w.Δd, s.d)
    axpy!(τd, w.Δy, s.y)

    push!(s.hist, (μ=μ, pstep=τp, dstep=τd, pres=nrp, dres=nrd, niter=kkt_iters))

    if s.settings.verbose
        printrow(length(s.hist), s.hist[end])
    end

    if isstalled(s.hist)
        if isnearoptimal(s.hist; feas_tol=s.settings.feas_tol, gap_tol=s.settings.gap_tol, near_factor=s.settings.near_factor)
            if s.settings.verbose
                println("μ stalling detected but solution is near-optimal; accepting")
            end

            return NEAR_OPTIMAL
        else
            if s.settings.verbose
                println("Warning: μ stalling detected (μ=$(s.hist.μ[end]), rp=$(s.hist.pres[end]), rd=$(s.hist.dres[end]))")
            end

            return STALLED
        end
    end

    if isnumfail(s.hist; threshold=s.settings.step_collapse_threshold)
        if isnearoptimal(s.hist; feas_tol=s.settings.feas_tol, gap_tol=s.settings.gap_tol, near_factor=s.settings.near_factor)
            if s.settings.verbose
                println("Step collapse detected but solution is near-optimal; accepting")
            end

            return NEAR_OPTIMAL
        else
            if s.settings.verbose
                println("Warning: numerical failure detected")
            end

            return NUMERICAL_FAILURE
        end
    end

    if length(s.hist) >= s.settings.itmax
        return ITERATION_LIMIT
    end

    return CONTINUE
end

function CommonSolve.solve!(s::IPMSolver)
    status = CONTINUE

    while status == CONTINUE
        status = step!(s)
    end

    return IPMResult(s, status)
end

function CommonSolve.solve(prob::IPMProblem, settings::IPMSettings=IPMSettings{eltype(prob.c)}())
    return solve!(init(prob, settings))
end
