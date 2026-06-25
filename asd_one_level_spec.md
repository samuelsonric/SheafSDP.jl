# One-level ASd preconditioner for the Uzawa Schur solve — initial implementation & diagnostic spec

**Target:** `SheafSDP.jl`, the `UzawaWorkspace` Schur-CG path (`src/kkt/uzawa.jl`).
**Goal of this first cut:** add a *dense*, one-level **diagonally-assembled additive Schwarz (ASd)** preconditioner to the
outer CG on the augmented Schur complement $S_\gamma = B\,K_\gamma^{-1} B^\top$, wire it in behind a flag, and use it to
answer one question: **does a real DD preconditioner let us drop $\gamma$ (`raug`) without the CG iteration count
exploding?**

Everything here assumes **general (vector-stalk, dense) restriction maps**. No scalar-stalk shortcuts.

Explicitly **out of scope** for this cut (deferred until the dense version is validated):

- the §5.5 block-Woodbury apply (an *apply-cost* optimization of the same $T_v$ — swap in later, diff against dense);
- the two-level coarse space / cycle-space deflation (a *scaling* fix — only build if iteration counts still grow with
  $n_P$ after $\gamma$ is lowered);
- the large-$\gamma$ spectral wrap $t\mapsto t/(1+\gamma t)$ (unnecessary precisely in the low-$\gamma$ regime we are
  steering toward).

---

## 0. The structure we are exploiting (and where it lives in the code)

The Schur complement acts on **edge space** ($m$ = total edge-stalk DOFs). $B=\delta$ is stored column-major by vertex
in `BlockSparseMatrix`, which hands us the entire gather/scatter layout for free:

| Quantity | Meaning | API |
|---|---|---|
| vertices $v$ (blocks $k$) | column blocks of $B$ | `vtxs(B)`, `nvtxs(B)` |
| $q_v$ | vertex stalk dim | `ncols(B, v)` = `length(colrange(B, v))` |
| incident edges of $v$ | arcs in column $v$ | `e in srcrange(B, v)`, edge `u = B.tgt[e]` |
| edge (row block) $u$ | a row block of $B$ | `u in 1:nouts(B)`, range `rowrange(B, u)` |
| $\dim u$ | edge stalk dim | `length(rowrange(B, u))` |
| $\pm F_{v\to u}$ | signed restriction map | `block(B, u, v, e)` — shape `dim u × q_v` |
| $A_v$ | vertex Hessian (barrier + Q) | `block(H, v, v, v)`, lower-tri ⇒ `Symmetric(·, :L)` |

Two facts that this geometry guarantees and that the implementation leans on:

1. **Every edge-row is shared by exactly its two endpoints** (overlap $\omega_i\equiv 2$, no self-loops). So there are
   **no interior rows**, the assembled diagonal correction $\Delta_v$ is **strictly positive on every edge-block**
   (provided the *other* endpoint's restriction has full row rank), and ASd is **well-posed with no $(2,2)$
   regularization**. This is why we can start with a plain dense Cholesky of $T_v$ and skip §5.6 entirely.
2. **"Diagonally assembled" means block-diagonal per edge**, not scalar-diagonal. For edge $u=(a,b)$ the assembled
   block is the *true* diagonal block of $M$,
   $$(M)_{uu} = (S_a)_{uu} + (S_b)_{uu}, \qquad (S_v)_{uu} = F_{v\to u}\,A_v^{-1}\,F_{v\to u}^\top,$$
   and the only inter-block reduction in the whole build is "each edge sums its two endpoints' diagonal blocks."

The local operator we factor per vertex is then **own off-diagonals + assembled diagonal**:
$$
T_v \;=\; \underbrace{S_v}_{\text{own star, } p_v\times p_v} \quad\text{with each diagonal block } (S_v)_{uu}\ \text{overwritten by}\ (M)_{uu},
\qquad p_v = \sum_{u \ni v}\dim u .
$$
The preconditioner is the additive sum $M_{\mathrm{1L}}^{-1} = \sum_v N_v\, T_v^{-1}\, N_v^\top$, applied by gather →
local solve → scatter-add. Signs in the restriction maps are carried through the stored blocks, so we never strip them.

> **⚠ verify** (cheap, do once): `nouts(B)` is the number of edge/row blocks and edges are indexed `1:nouts(B)`;
> `block(H, v, v, v)` returns a readable $q_v\times q_v$ dense block (copy it before factoring if it is a mutable view).

---

## 1. Files to add / touch

- **add** `src/kkt/asd.jl` — the preconditioner (struct, static layout, per-step refresh, `ldiv!`).
- **touch** `src/kkt/uzawa.jl` — hold the preconditioner in `UzawaWorkspace`; refresh it in `init_kkt!`; pass it (and
  an optional warm-start `x0`) into the Schur `it!`.
- **add** `src/kkt/asd.jl` to the `include` list in `src/SheafSDP.jl` (after `kkt/kkt.jl` or inside it).
- **add** a `precond::Bool` and `warmstart::Bool` knob to `UzawaSettings` so the diagnostic can A/B against `M = I`.

---

## 2. The preconditioner: `src/kkt/asd.jl`

### 2.1 Static layout (built once from the permuted `B`)

The sparsity pattern is fixed for the whole solve, so the per-vertex gather ranges and local block offsets are computed
once. Only the numbers in `T_v` change per IPM step.

```julia
# Per-vertex gather/scatter layout + scratch + current Tᵥ factor.
struct VertexBlock{T}
    pv      :: Int                       # local dimension = Σ dim(u) over incident edges
    edges   :: Vector{Int}               # incident edge (row-block) ids u, in srcrange order
    arcs    :: Vector{Int}               # the arc id e for each incident edge (for block(B,u,v,e))
    gr      :: Vector{UnitRange{Int}}    # global row ranges rowrange(B,u), one per incident edge
    lc      :: Vector{UnitRange{Int}}    # local column ranges inside the pv-vector, one per incident edge
    rloc    :: Vector{T}                 # gather scratch (length pv)
    zloc    :: Vector{T}                 # solve scratch  (length pv)
    chol    :: Base.RefValue{Cholesky{T,Matrix{T}}}  # refreshed each IPM step
end

struct ASdPrecond{T, I, BS}
    B       :: BS                        # the permuted BlockSparseMatrix (same object used in solve_uzw!)
    blocks  :: Vector{VertexBlock{T}}    # one per vertex, indexed by v
    Mdiag   :: Vector{Matrix{T}}         # assembled diagonal block per edge u (length nouts(B))
    npd     :: Base.RefValue{Int}        # diagnostic: # vertices whose Tᵥ was not PD this step
end

function ASdPrecond(B::BlockSparseMatrix{T, I}) where {T, I}
    nv = nvtxs(B)
    blocks = Vector{VertexBlock{T}}(undef, nv)

    for v in vtxs(B)
        edges = Int[]; arcs = Int[]; gr = UnitRange{Int}[]; lc = UnitRange{Int}[]
        off = 0
        for e in srcrange(B, v)
            u  = B.tgt[e]
            rr = rowrange(B, u)
            d  = length(rr)
            push!(edges, u); push!(arcs, e); push!(gr, rr)
            push!(lc, (off+1):(off+d))
            off += d
        end
        pv = off
        # placeholder factor; overwritten on first refresh!
        ch = cholesky(Matrix{T}(I, pv, pv))
        blocks[v] = VertexBlock{T}(pv, edges, arcs, gr, lc,
                                   zeros(T, pv), zeros(T, pv), Ref(ch))
    end

    Mdiag = [zeros(T, length(rowrange(B, u)), length(rowrange(B, u))) for u in 1:nouts(B)]
    return ASdPrecond{T, I, typeof(B)}(B, blocks, Mdiag, Ref(0))
end
```

### 2.2 Per-step refresh (called once per IPM iteration from `init_kkt!`)

Two passes. **Pass 1** forms each $Y_v = L_v^{-1}\bar B_{(v)}^\top$ and $S_v = Y_v^\top Y_v$, and accumulates the
assembled per-edge diagonal blocks `Mdiag`. **Pass 2** overwrites each $T_v$'s diagonal blocks with the assembled ones
and factors. We hold the $S_v$ from pass 1 (small, $p_v\times p_v$) to avoid recomputing.

```julia
function refresh!(P::ASdPrecond{T}, H::BlockSparseMatrix{T}) where {T}
    B = P.B
    P.npd[] = 0
    for Md in P.Mdiag
        fill!(Md, zero(T))
    end

    Sstore = Vector{Matrix{T}}(undef, nvtxs(B))   # cache Sᵥ from pass 1

    # ---- Pass 1: Sᵥ = Yᵥ' Yᵥ and accumulate assembled diagonals ----
    for v in vtxs(B)
        blk = P.blocks[v]
        qv, pv = ncols(B, v), blk.pv

        # Aᵥ = Hᵥ ; Lᵥ = chol(Aᵥ)
        Av = Symmetric(Matrix(block(H, v, v, v)), :L)   # ⚠ copy via Matrix(...) guards against view aliasing
        Lv = cholesky(Av)

        # Ȳᵀ ∈ ℝ^{qv × pv}: horizontally stack the (signed) restriction blocks, transposed
        Ybar = zeros(T, qv, pv)
        for (i, u) in enumerate(blk.edges)
            e  = blk.arcs[i]
            Fi = block(B, u, v, e)                       # dim(u) × qv  (signed)
            @views Ybar[:, blk.lc[i]] .= transpose(Fi)
        end

        # Yᵥ = Lᵥ⁻¹ Ȳᵀ   (in place); then Sᵥ = Yᵥ' Yᵥ
        ldiv!(Lv.L, Ybar)
        Sv = Ybar' * Ybar                                # pv × pv, SPSD, rank ≤ qv
        Sstore[v] = Sv

        # accumulate this vertex's contribution to each incident edge's diagonal block
        for (i, u) in enumerate(blk.edges)
            lci = blk.lc[i]
            @views P.Mdiag[u] .+= Sv[lci, lci]
        end
    end

    # ---- Pass 2: Tᵥ = Sᵥ with diagonal blocks replaced by assembled (M)_uu ; factor ----
    for v in vtxs(B)
        blk = P.blocks[v]
        Tv  = copy(Sstore[v])
        for (i, u) in enumerate(blk.edges)
            lci = blk.lc[i]
            @views Tv[lci, lci] .= P.Mdiag[u]            # own off-diagonals kept; diagonal assembled
        end
        ch = cholesky(Symmetric(Tv); check = false)
        if !issuccess(ch)
            P.npd[] += 1
            # diagnostic fallback so the solve can proceed: jitter the assembled diagonal.
            ϵ = sqrt(eps(T)) * (tr(Tv) / max(blk.pv, 1))
            for d in 1:blk.pv
                Tv[d, d] += ϵ
            end
            ch = cholesky(Symmetric(Tv); check = false)
        end
        blk.chol[] = ch
    end
    return P
end
```

### 2.3 Apply — `ldiv!(z, P, r)` (this is what CG calls)

`it!` passes the preconditioner with `ldiv=true`, so Krylov applies it as `ldiv!(z, M, r)` to compute
$z = M_{\mathrm{1L}}^{-1} r$. The apply is the symmetric additive Schwarz loop. It runs once per CG iteration, so it is
allocation-free (all scratch lives in the layout).

```julia
function LinearAlgebra.ldiv!(z::AbstractVector{T}, P::ASdPrecond{T}, r::AbstractVector{T}) where {T}
    fill!(z, zero(T))
    for v in vtxs(P.B)
        blk = P.blocks[v]
        # gather: rloc ← r over incident edge ranges
        @inbounds for i in eachindex(blk.gr)
            @views blk.rloc[blk.lc[i]] .= r[blk.gr[i]]
        end
        # local solve: zloc = Tᵥ⁻¹ rloc
        copyto!(blk.zloc, blk.rloc)
        ldiv!(blk.chol[], blk.zloc)
        # scatter-add: z += Nᵥ zloc
        @inbounds for i in eachindex(blk.gr)
            @views z[blk.gr[i]] .+= blk.zloc[blk.lc[i]]
        end
    end
    return z
end

# Krylov sometimes probes `mul!`; route it through ldiv! since this object represents M⁻¹ directly.
LinearAlgebra.ldiv!(P::ASdPrecond, r::AbstractVector) = ldiv!(similar(r), P, r)
```

This operator is **symmetric and PSD by construction** (a sum of $N_v T_v^{-1} N_v^\top$ with each $T_v\succ 0$), so it
is CG-safe as written. No restricted-AS (RAS) weighting — that would break symmetry.

---

## 3. Wiring into the Uzawa workspace

### 3.1 Extend `UzawaWorkspace` and its constructor

```julia
struct UzawaWorkspace{UPLO, T, I, ItrWrk, P} <: KKTWorkspace{T}
    F::FChordalTriangular{:N, UPLO, T, I}
    L::BlockSparseMatrix{T, I}
    facwrk::FactorizationWorkspace{T, I}
    divwrk::DivisionWorkspace{T, I}
    itrwrk::ItrWrk
    r::Vector{T}
    α::Scalar{T}
    nrm::T
    precond::P            # ASdPrecond{T} or nothing
    y0::Vector{T}         # warm-start buffer (previous Schur solution), length m
end

function UzawaWorkspace(F, L, B; use_precond::Bool = true)
    m = size(B, 1)
    facwrk = FactorizationWorkspace(F)
    divwrk = DivisionWorkspace(F, 1)
    itrwrk = CgWorkspace(m, m, Vector{eltype(L)})
    r  = zeros(eltype(L), m)
    α  = ones(eltype(L))
    nrm = norm(B)^2
    P  = use_precond ? ASdPrecond(B) : nothing
    y0 = zeros(eltype(L), m)
    return UzawaWorkspace(F, L, facwrk, divwrk, itrwrk, r, α, nrm, P, y0)
end
```

`make_kkt` already has the permuted `B` in hand when it constructs the workspace, so the only change there is to pass
`use_precond = set.precond` (or thread it through). The `ASdPrecond` captures *that* `B`, the same object later handed
to `solve_uzw!`.

### 3.2 Refresh inside `init_kkt!` (already called once per IPM step)

```julia
function init_kkt!(wrk::UzawaWorkspace{UPLO, T}, set::UzawaSettings{T}, A::BlockSparseMatrix) where {UPLO, T}
    wrk.α[] = set.aaug + set.raug * norm(Symmetric(A, :L)) / wrk.nrm
    ok = init_uzw!(wrk.facwrk, wrk.F, wrk.L, A, wrk.α[], set.rgmin, set.rgmax)
    if ok && wrk.precond !== nothing
        refresh!(wrk.precond, A)          # A is H, block-diagonal vertex Hessians
        fill!(wrk.y0, zero(T))            # reset warm-start at the top of each IPM step (see note in §4)
    end
    return ok
end
```

### 3.3 Pass the preconditioner (and warm start) into the Schur CG

In `solve_uzw!`, the only line that changes is the Schur solve. Currently:

```julia
it!(itrwrk, S, r; α, atol, rtol, itmax)
```

becomes, threading `M` and the warm start:

```julia
M = wrk_precond            # === wrk.precond, or I if disabled
if warmstart
    it!(itrwrk, S, r, y0; M, ldiv=true, atol, rtol, itmax)
else
    it!(itrwrk, S, r;    M, ldiv=true, atol, rtol, itmax)
end
copyto!(y0, solution(itrwrk))   # carry this Schur solution into the next solve as x0
```

`M = I` reproduces today's behavior exactly, which is the A/B baseline. (Plumb `wrk.precond`, `wrk.y0`, and the
`warmstart` flag from `solve_kkt!` into `solve_uzw!`; they are all on `wrk`/`set` already.)

---

## 4. Warm start — what couples to what

The augmented operator $S_\gamma$ is **identical** for the affine and corrector solves within one IPM step (same
$K_\gamma$, same factor); only the RHS differs. So warm-starting the corrector from the affine Schur solution is a
strict win and costs nothing. Across IPM steps $S_\gamma$ drifts smoothly along the central path, so carrying `y0`
forward helps too, but more weakly and it can occasionally hurt right after a long step.

Recommended switch settings for the diagnostic:

- **within a step:** always warm-start corrector from affine (set `y0` from the affine solve, *do not* reset between
  affine and corrector).
- **across steps:** make it a flag. The `fill!(wrk.y0, 0)` in `init_kkt!` above is the *conservative* choice (reset each
  step). To test cross-step reuse, delete that `fill!` and measure both.

⚠ One correctness note: `y0` holds the **reduced** Schur unknown inside `solve_uzw!`, not the final $\Delta y$. That is
the right thing to warm-start (same operator/RHS family). Don't warm-start it with $\Delta y$ from `newton!`.

---

## 5. Diagnostic harness

Three measurements, in order of importance.

### 5.1 The headline experiment: does ASd decouple iterations from $\gamma$?

Sweep `raug` with the preconditioner **on** and **off**, holding everything else fixed, and record CG iterations per
Schur solve. The claim under test: **with `M = I`, CG iterations climb as `raug` drops; with ASd, they stay roughly
flat.**

```julia
using SheafSDP

function gamma_sweep(prob; raugs = (1e6, 1e4, 1e2, 1e0, 1e-2), precond = true, warmstart = true)
    for raug in raugs
        kkt = UzawaSettings(; raug = raug, precond = precond, warmstart = warmstart)
        set = IPMSettings(; kkt = kkt, verbose = false)
        res = solve(prob, set)
        h   = res.history
        # kkt_iters per IPM iteration is in the history; total is res.kkt_iters
        avg = res.kkt_iters / max(res.iterations, 1)
        @info "raug=$raug" status=res.status ipm_iters=res.iterations total_kkt=res.kkt_iters avg_kkt_per_ipm=avg
    end
end
```

Then run `gamma_sweep(prob; precond=false)` and `gamma_sweep(prob; precond=true)` and compare the two
`avg_kkt_per_ipm` curves against `raug`. **Success looks like:** the `precond=true` curve is roughly horizontal while
`precond=false` curve bends sharply upward as `raug → 1`.

Also watch, at the *low* end of the sweep:

- **final accuracy** `res.history.rp[end]`, `res.history.rd[end]`, terminal `μ`, and `res.status`;
- whether the endgame **iterative-refinement** rounds drop (lower $\gamma$ ⇒ better-conditioned chordal factor ⇒ the
  true KKT residual is met with fewer refinement Newton steps). Instrument `refine_kkt!`'s returned `kkt_iters` or add a
  counter.

The win is not "fewer CG iters at `raug=1e6`." It is "I can run at `raug≈1` (accuracy-optimal) without the CG count
blowing up, and I get better final residuals / fewer refinement rounds for it."

### 5.2 Preconditioner self-consistency (run once, small instance)

Before trusting the sweep, confirm the operator is what it claims to be:

```julia
function check_precond(P::ASdPrecond{T}, m::Int) where {T}
    # symmetry:  a'(P⁻¹ b) ≈ (P⁻¹ a)'b
    a = randn(T, m); b = randn(T, m)
    za = similar(a); zb = similar(b)
    ldiv!(za, P, a); ldiv!(zb, P, b)
    sym = abs(dot(a, zb) - dot(za, b)) / (norm(a) * norm(b))
    # positive-definiteness on a few probes:  r'(P⁻¹ r) > 0
    pos = minimum(let r = randn(T, m), z = similar(r); ldiv!(z, P, r); dot(r, z) end for _ in 1:20)
    @info "ASd self-check" symmetry_resid=sym min_quadratic_form=pos npd=P.npd[]
    @assert sym < 1e-10
    @assert pos > 0
end
```

Call it right after a `refresh!` at some mid-solve IPM iterate. `npd` (non-PD $T_v$ count) should be **0** for healthy
sheaves; a nonzero count flags vertices where the assembled diagonal failed to make $T_v$ PD — see §6.

### 5.3 Cycle-space / singular-Schur check (run once, small instance)

Confirm the topological picture on *your* sheaves: the near-null space of the unaugmented $M = B H^{-1} B^\top$ should
have dimension $m - \operatorname{rank}(B) = \dim\ker\delta^\top$ — the cycle space. This number is the coarse-space
dimension you would eventually target if §1's deferred two-level step becomes necessary.

```julia
# dense densify of a (small) BlockSparseMatrix, using only confirmed accessors
function densify(B::BlockSparseMatrix{T}) where {T}
    m, n = size(B)
    D = zeros(T, m, n)
    for v in vtxs(B), e in srcrange(B, v)
        u = B.tgt[e]
        @views D[rowrange(B, u), colrange(B, v)] .= block(B, u, v, e)
    end
    return D
end

function cycle_space_check(B::BlockSparseMatrix{T}, H::BlockSparseMatrix{T}; tol = 1e-8) where {T}
    m, n = size(B)
    Bd = densify(B)
    Hinv = zeros(T, n, n)
    for v in vtxs(B)
        r = colrange(B, v)
        @views Hinv[r, r] .= inv(Symmetric(Matrix(block(H, v, v, v)), :L))
    end
    M  = Symmetric(Bd * Hinv * Bd')
    λ  = eigvals(M)
    nz = count(<(tol * maximum(λ)), λ)          # near-null modes
    @info "cycle-space" m=m rankB=rank(Bd) predicted_kerδᵀ=(m - rank(Bd)) near_null_eigs=nz
    return λ
end
```

`near_null_eigs` should match `m - rankB`. If it does, the "low modes of $S_\gamma$ are the sheaf cycle space" story
holds on your data, and you know exactly what a coarse space would have to deflate.

---

## 6. Failure modes and what each one tells you

| Symptom | Likely cause | Reading / action |
|---|---|---|
| `npd > 0` in `refresh!` | a vertex $v$ has an incident edge whose **other** endpoint's restriction is rank-deficient, so $\Delta_v$ is singular on that block and $T_v$ isn't PD | this is the genuine "locally over-determined" case the markdown's §5.6 validity condition describes. The jitter fallback keeps the solve running for the diagnostic, but the principled fix is the $(2,2)$ regularization or routing those modes to a coarse space. Note which vertices. |
| CG iterations flat in `raug` even with `M = I` | the instance is too easy / too small to show the conditioning effect | scale up $n_P$; the $\gamma$-dependence only bites once $\mu_{\min}$ is small. |
| ASd helps at high `raug` but **not** at low `raug` | one-level Schwarz leaves the **global** low (cycle) modes untouched — exactly what §1's deferred coarse space fixes | this is the signal to build the two-level correction. Use §5.3's `near_null_eigs` as its size. |
| `self-check` symmetry residual not ~0 | a sign or transpose bug in `Ybar`/gather/scatter | the restriction signs must come *through* `block(B,…)`; don't strip them. |
| accuracy worse at low `raug` despite ASd | the inner direct solve `init_uzw!` may be hitting its `rgmin/rgmax` perturbation, or refinement tol too loose | check `init_uzw!` return and the refinement gating in `step!`. |
| build dominates runtime | dense $T_v$ Cholesky is $O(p_v^3)$ on tall vertices | this is the cue to switch on the §5.5 block-Woodbury apply (deferred), which removes the $p_v^3$ term. Validate dense first. |

---

## 7. Definition of done for this cut

1. `ASdPrecond` builds, `self-check` passes (symmetry ~0, quadratic form > 0, `npd = 0` on a healthy instance).
2. `M = I` path is bit-for-bit the current solver (regression check on an existing test instance).
3. The §5.1 sweep produces the two iteration-vs-`raug` curves, and they separate: ASd flat-ish, `I` rising.
4. At the low-`raug` end with ASd on, the IPM still reaches `OPTIMAL` with residuals no worse than the high-`raug`
   baseline, ideally with fewer refinement rounds.
5. §5.3 confirms `near_null_eigs == m - rank(B)` on a small instance.

If (3) and (4) hold, the strategy question is answered affirmatively and the next move is to lower the *default* `raug`
and decide — from whether the ASd curve is *truly* flat or merely *flatter* as $n_P$ grows — whether the coarse space is
needed. If the curve still creeps with $n_P$, that creep is the cycle space, and §5.3 already sized it for you.

---

## 8. Verification checklist before first run (cheap, do once)

- `nouts(B)` = number of edge/row blocks; edges indexed `1:nouts(B)`.
- `block(H, v, v, v)` is readable and `Matrix(...)` of it copies (no aliasing into `H`).
- `block(B, u, v, e)` has shape `length(rowrange(B,u)) × ncols(B,v)` and **includes the sign**.
- Krylov's `cg!` with `M = P, ldiv = true` calls `ldiv!(z, P, r)` (it does in current Krylov; confirm your pinned
  version).
- `size(B) == (m, n)` with `m` = edge DOFs (it is, per `solve_uzw!`).
