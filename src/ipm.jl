#
# primal-dual interior point method
#
# primal: min c'p  s.t. Bp = g, P ≻ 0
# dual:   max g'y  s.t. B'y + d = c, D ≻ 0
#
# p, d are svec representations of block-diagonal P, D
#

@enum IPMStatus OPTIMAL NEAR_OPTIMAL STALLED NUMERICAL_FAILURE ITERATION_LIMIT

struct IPMProblem{T, I}
    c::Vector{T}
    g::Vector{T}
    B::BlockSparseMatrix{T, I}
    Q::BlockSparseMatrix{T, I}
    cones::Vector{Symbol}
end

function tocone(s::Symbol)
    if s === :SDP
        return SDP()
    elseif s === :POS
        return POS()
    elseif s === :SOC
        return SOC()
    elseif s === :NOC
        return NOC()
    elseif s === :EXP
        return EXP()
    else
        error("Unknown cone: $s")
    end
end

@kwdef struct IPMSettings{T}
    kkt::KKTSettings{T} = UzawaSettings{T}()
    step_frac::T = 0.99
    feas_tol::T = 1e-8
    gap_tol::T = 1e-8
    itmax::Int = 100
    verbose::Bool = false
    stall_window::Int = 5
    stall_threshold::T = 0.99
    step_collapse_threshold::T = 1e-6
    refine_itmax::Int = 10
    refine_atol::T = 1e-12
    refine_rtol::T = 1e-13
end

const IPMHistoryRow{T} = @NamedTuple{μ::T, τp::T, τd::T, rp::T, rd::T, kkt_iters::Int}

struct IPMHistory{T} <: AbstractVector{IPMHistoryRow{T}}
    μ::Vector{T}
    τp::Vector{T}
    τd::Vector{T}
    rp::Vector{T}
    rd::Vector{T}
    kkt_iters::Vector{Int}
end

function IPMHistory{T}() where {T}
    return IPMHistory{T}(T[], T[], T[], T[], T[], Int[])
end

Base.size(h::IPMHistory) = (length(h.μ),)

function Base.getindex(h::IPMHistory, i::Int)
    return (μ=h.μ[i], τp=h.τp[i], τd=h.τd[i], rp=h.rp[i], rd=h.rd[i], kkt_iters=h.kkt_iters[i])
end

function Base.push!(h::IPMHistory, row::NamedTuple)
    push!(h.μ, row.μ)
    push!(h.τp, row.τp)
    push!(h.τd, row.τd)
    push!(h.rp, row.rp)
    push!(h.rd, row.rd)
    push!(h.kkt_iters, row.kkt_iters)
    return h
end

function printrow(i::Int, row::IPMHistoryRow)
    println("Iter $i: μ = $(row.μ), ||rp|| = $(row.rp), ||rd|| = $(row.rd), kkt = $(row.kkt_iters)")
end

struct IPMResult{T}
    p::Vector{T}
    d::Vector{T}
    y::Vector{T}
    status::IPMStatus
    iterations::Int
    kkt_iters::Int
    history::IPMHistory{T}
end

mutable struct IPMSolver{T, I, W, Perm}
    # problem data
    p::Vector{T}
    d::Vector{T}
    y::Vector{T}
    c::Vector{T}
    g::Vector{T}
    B::BlockSparseMatrix{T, I}
    Q::BlockSparseMatrix{T, I}
    cones::Vector{<:Cone}

    # permutation
    P::Perm

    # workspace
    rp::Vector{T}
    rd::Vector{T}
    f::Vector{T}
    Δpa::Vector{T}
    Δya::Vector{T}
    Δda::Vector{T}
    Δp::Vector{T}
    Δy::Vector{T}
    Δd::Vector{T}
    H::BlockSparseMatrix{T, I}
    caches::Caches{T, I}
    wrk::W
    # refinement workspace
    sp::Vector{T}
    sy::Vector{T}
    dp::Vector{T}
    dy::Vector{T}

    # state
    hist::IPMHistory{T}
    iter::Int
    kkt_iters::Int
    status::IPMStatus
    ν::Int

    # settings
    settings::IPMSettings{T}
end

function conedegree(cones::AbstractVector, B::BlockSparseMatrix)
    ν = 0
    for v in vtxs(B)
        ν += degree(cones[v], ncols(B, v))
    end
    return ν
end

function residuals!(rp, rd, B, p, d, y, c, g, Q)
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

function scale!(H::BlockSparseMatrix{T}, caches::Caches{T},
                cones::AbstractVector, p::AbstractVector{T}, d::AbstractVector{T},
                B::BlockSparseMatrix{T}, Q) where {T}
    for v in vtxs(B)
        r = colrange(B, v)
        Hv = block(H, v, v, v)
        cv = cache(caches, v, cones[v])

        scale!(Hv, view(p, r), view(d, r), cv)
        axpy!(true, block(Q, v, v, v), Hv)
    end

    return
end

function newton!(Δp, Δy, Δd, wrk, set, H, B, f, rp, rd, Q, sp, sy, dp, dy;
                 refine::Bool=false, refine_itmax::Int=10, refine_atol=1e-12, refine_rtol=1e-13)
    kkt_iters = solve_kkt!(wrk, set, Δp, Δy, H, B, f, rp)

    if refine
        kkt_iters += refine_kkt!(Δp, Δy, wrk, set, H, B, f, rp, sp, sy, dp, dy;
                                 itmax=refine_itmax, atol=refine_atol, rtol=refine_rtol)
    end

    lmul!(-1, Δy)

    copyto!(Δd, rd)
    mul!(Δd, B', Δy, -1, 1)
    mul!(Δd, Symmetric(Q, :L), Δp, 1, 1)

    return kkt_iters
end

function corrector!(f, caches, cones, p, d, Δp, Δd, σμ, B)
    for v in vtxs(B)
        r = colrange(B, v)
        cv = cache(caches, v, cones[v])
        corr!(view(f, r), view(p, r), view(d, r), view(Δp, r), view(Δd, r), σμ, cv)
    end

    return f
end

function maxsteps(p, d, Δp, Δd, caches, cones, B; frac=0.99)
    T = eltype(p)
    τp = one(T)
    τd = one(T)

    for v in vtxs(B)
        r = colrange(B, v)
        cv = cache(caches, v, cones[v])
        τp = min(τp, maxstep_prim(view(p, r), view(Δp, r), frac, cv))
        τd = min(τd, maxstep_dual(view(d, r), view(Δd, r), frac, cv))
    end

    return τp, τd
end

#
# scaled (Mehrotra-style) starting point
#
# Start from the block-diagonal cone identity e and scale the primal and
# dual copies independently so the two initial infeasibilities live on the
# same scale as the data:
#
#   primal:  ξ chosen so that ‖B (ξe)‖ ≈ ‖g‖   (reach the consensus RHS)
#   dual:    η chosen so that   ‖ηe‖   ≈ ‖c‖    (balance the cost)
#
# This removes the "scale-discovery" phase where μ shoots up several orders
# of magnitude on the first step because both copies start at e. A positive
# multiple of e is still perfectly centered, so no interior shift is needed,
# and y can stay at 0 (dual feasibility is reached quickly regardless).
#
function startingpoint(B::BlockSparseMatrix{T}, g::AbstractVector{T},
                       c::AbstractVector{T}, cones::AbstractVector) where {T}
    m, n = size(B)

    e = zeros(T, n)
    for v in vtxs(B)
        identity!(view(e, colrange(B, v)), cones[v])
    end

    Be = zeros(T, m)
    mul!(Be, B, e)

    ξ = max(one(T), norm(g) / max(eps(T), norm(Be)))
    η = max(one(T), norm(c) / max(eps(T), norm(e)))

    p = ξ .* e
    d = η .* e
    y = zeros(T, m)

    return p, d, y
end

function isstalled(hist::IPMHistory; window=5, threshold=0.99, noise_floor=1e-12)
    n = length(hist.μ)
    if n < window + 1
        return false
    end
    # Check if progress is being made on μ, rp, or rd
    # Ignore "improvements" when values are already at noise floor (machine precision)
    μ_improved = false
    rp_improved = false
    rd_improved = false

    for i in (n-window+1):n
        if hist.μ[i] < threshold * hist.μ[i-1]
            μ_improved = true
        end
        # Only count residual improvement if not at machine precision noise floor
        if hist.rp[i-1] > noise_floor && hist.rp[i] < threshold * hist.rp[i-1]
            rp_improved = true
        end
        if hist.rd[i-1] > noise_floor && hist.rd[i] < threshold * hist.rd[i-1]
            rd_improved = true
        end
    end

    # Stall if no measure improved within the window
    return !μ_improved && !rp_improved && !rd_improved
end

# Check if solution is "near optimal" - small enough residuals and gap even if stalled
# Hypatia uses near_factor=1000 by default
function isnearoptimal(hist::IPMHistory, feas_tol, gap_tol; near_factor=1000.0)
    if isempty(hist.μ)
        return false
    end
    μ = hist.μ[end]
    rp = hist.rp[end]
    rd = hist.rd[end]
    # Accept if residuals are small and either:
    # 1. μ is reasonably small (within near_factor of gap_tol), or
    # 2. Residuals are at machine precision (1e-10), indicating numerical convergence
    residuals_tiny = rp < 1e-10 && rd < 1e-10
    return rp < feas_tol && rd < feas_tol && (μ < near_factor * gap_tol || residuals_tiny)
end

function isnumfail(hist::IPMHistory; window=3, threshold=1e-6)
    if length(hist.τp) < window
        return false
    end

    τavg = sum(hist.τp[end-window+1:end]) / window
    τavg = min(τavg, sum(hist.τd[end-window+1:end]) / window)

    if τavg > threshold
        return false
    end

    if length(hist.rp) < window + 1
        return true
    end

    return hist.rp[end] > 0.9 * hist.rp[end - window] || hist.rd[end] > 0.9 * hist.rd[end - window]
end

function CommonSolve.init(prob::IPMProblem{T, I}, settings::IPMSettings{T}=IPMSettings{T}()) where {T, I}
    c0, g, B0, Q0 = prob.c, prob.g, prob.B, prob.Q
    cones0 = map(tocone, prob.cones)

    kktset = settings.kkt
    R, P, B, wrk = make_kkt(kktset, B0)

    n = size(B0, 2)
    m = size(B0, 1)

    p0, d0, y = startingpoint(B0, g, prob.c, cones0)

    p = P * p0
    d = P * d0
    c = P * c0

    Q = halfselectvtxs(halfselectvtxs(Q0, R.perm), R.perm)
    cones = cones0[R.perm]

    ν = conedegree(cones, B)

    rp = zeros(T, m)
    rd = zeros(T, n)
    f = zeros(T, n)

    Δpa = zeros(T, n)
    Δya = zeros(T, m)
    Δda = zeros(T, n)
    Δp = zeros(T, n)
    Δy = zeros(T, m)
    Δd = zeros(T, n)

    H = allocblockdiag(B)
    caches = Caches(cones, B)
    sp = zeros(T, n)
    sy = zeros(T, m)
    dp = zeros(T, n)
    dy = zeros(T, m)
    hist = IPMHistory{T}()

    return IPMSolver(
        p, d, y, c, g, B, Q, cones,
        P,
        rp, rd, f, Δpa, Δya, Δda, Δp, Δy, Δd, H, caches, wrk, sp, sy, dp, dy,
        hist, 0, 0, ITERATION_LIMIT, ν,
        settings
    )
end

function step!(s::IPMSolver{T}) where {T}
    #
    # compute the primal and dual residuals:
    #
    #   rp = g - B p
    #   rd = c - d + Q p - Bᵀ y
    #
    residuals!(s.rp, s.rd, s.B, s.p, s.d, s.y, s.c, s.g, s.Q)

    if iszero(s.ν)
        μ = zero(T)
    else
        μ = dot(s.p, s.d) / s.ν
    end
    #
    # iterative refinement only in endgame (when μ is close to gap_tol)
    #
    do_refine = μ < 100 * s.settings.gap_tol
    #
    # check the quantities
    #
    #   - ‖rp‖ / (1 + ‖g‖)
    #   - ‖rd‖ / (1 + ‖c‖)
    #   - μ
    #
    # for convergence
    #
    nrp = norm(s.rp) / (1 + norm(s.g))
    nrd = norm(s.rd) / (1 + norm(s.c))

    if nrp < s.settings.feas_tol && nrd < s.settings.feas_tol && (iszero(s.ν) || μ < s.settings.gap_tol)
        s.status = OPTIMAL
        return false
    end
    #
    # compute the sum
    #
    #   H = Hf(w) + Q
    #
    # where f is the barrier function and
    # w is the scaling point.
    #
    scale!(s.H, s.caches, s.cones, s.p, s.d, s.B, s.Q)

    if !init_kkt!(s.wrk, s.settings.kkt, s.H)
        s.status = NUMERICAL_FAILURE

        if s.settings.verbose
            println("Warning: KKT factorization failed")
        end

        return false
    end
    #
    # compute the affine direction (Δpa, Δya, Δda)
    # by solving for Δpa, Δya in
    #
    #   [ H  Bᵀ ] [ Δpa ] = [ -d - rd ]
    #   [ B  0  ] [ Δya ]   [ rp      ]
    #
    # and setting Δda to the value
    #
    #   Δda = rd - Bᵀ Δya + Q Δpa
    #
    axpby!(-1, s.d,  0, s.f)
    axpby!(-1, s.rd, 1, s.f)
    kkt_iters_aff = newton!(s.Δpa, s.Δya, s.Δda, s.wrk, s.settings.kkt, s.H, s.B, s.f, s.rp, s.rd, s.Q, s.sp, s.sy, s.dp, s.dy;
                            refine=do_refine, refine_itmax=s.settings.refine_itmax,
                            refine_atol=s.settings.refine_atol, refine_rtol=s.settings.refine_rtol)
    #
    # compute the centering parameter
    #
    #   σ ∈ [0, 1]
    #
    τpa, τda = maxsteps(s.p, s.d, s.Δpa, s.Δda, s.caches, s.cones, s.B; frac=one(T))

    pa = s.p + τpa * s.Δpa
    da = s.d + τda * s.Δda
    μa = dot(pa, da) / s.ν

    σ = clamp((μa / μ)^3, zero(T), one(T))
    #
    # solve for the corrector direction (Δp, Δy, Δd)
    # by solving for Δp, Δy in
    #
    #   [ H  Bᵀ ] [ Δp ] = [ -d - rd + σμ e - Δpa ∘ Δda ]
    #   [ B  0  ] [ Δy ]   [ rp                         ]
    #
    # and setting Δd to the value
    #
    #   Δd = rd - Bᵀ Δy + Q Δp
    #
    corrector!(s.f, s.caches, s.cones, s.p, s.d, s.Δpa, s.Δda, σ * μ, s.B)
    axpy!(-1, s.rd, s.f)
    kkt_iters_corr = newton!(s.Δp, s.Δy, s.Δd, s.wrk, s.settings.kkt, s.H, s.B, s.f, s.rp, s.rd, s.Q, s.sp, s.sy, s.dp, s.dy;
                             refine=do_refine, refine_itmax=s.settings.refine_itmax,
                             refine_atol=s.settings.refine_atol, refine_rtol=s.settings.refine_rtol)
    kkt_iters = kkt_iters_aff + kkt_iters_corr
    #
    # take a step in the direction
    #
    #   (Δp, Δy, Δd)
    #
    τp, τd = maxsteps(s.p, s.d, s.Δp, s.Δd, s.caches, s.cones, s.B; frac=s.settings.step_frac)

    axpy!(τp, s.Δp, s.p)
    axpy!(τd, s.Δd, s.d)
    axpy!(τd, s.Δy, s.y)

    push!(s.hist, (μ=μ, τp=τp, τd=τd, rp=nrp, rd=nrd, kkt_iters=kkt_iters))
    s.iter += 1
    s.kkt_iters += kkt_iters

    if s.settings.verbose
        printrow(s.iter, s.hist[end])
    end

    if isstalled(s.hist; window=s.settings.stall_window, threshold=s.settings.stall_threshold)
        # Check if we're "near optimal" - residuals and μ are small enough
        if isnearoptimal(s.hist, s.settings.feas_tol, s.settings.gap_tol)
            s.status = NEAR_OPTIMAL
            if s.settings.verbose
                println("μ stalling detected but solution is near-optimal; accepting")
            end
        else
            s.status = STALLED
            if s.settings.verbose
                println("Warning: μ stalling detected (μ=$(s.hist.μ[end]), rp=$(s.hist.rp[end]), rd=$(s.hist.rd[end]))")
            end
        end

        return false
    end

    if isnumfail(s.hist; threshold=s.settings.step_collapse_threshold)
        s.status = NUMERICAL_FAILURE

        if s.settings.verbose
            println("Warning: numerical failure detected")
        end

        return false
    end

    return s.iter < s.settings.itmax
end

function CommonSolve.solve!(s::IPMSolver{T}) where {T}
    while step!(s) end

    p = s.P \ s.p
    d = s.P \ s.d

    return IPMResult{T}(p, d, s.y, s.status, s.iter, s.kkt_iters, s.hist)
end

function CommonSolve.solve(prob::IPMProblem, settings::IPMSettings=IPMSettings{eltype(prob.c)}())
    return solve!(init(prob, settings))
end
