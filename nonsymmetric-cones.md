# Implementing the Exponential and Power Cones

*A guide to adding 3D nonsymmetric cones to SheafSDP, following Dahl & Andersen,
"A primal-dual interior-point algorithm for nonsymmetric exponential-cone
optimization," Math. Program. (2022) 194:341–370.*

---

## 0. Orientation

This document describes how to add two nonsymmetric cones — the **exponential
cone** `EXP` and the **3D power cone** `POW{α}` — to the existing cone interface
(`degree`, `cachesize`, `identity!`, `scale!`, `hess!`, `corr!`, `maxstep`).

The central facts that make this tractable:

1. **Both cones are intrinsically 3-dimensional.** There is no `EXP(n)` family.
   `ϑ = 3` always. Big models are built by *replication* — one 3D block per
   exp/log/entropy/power term — which is exactly the many-small-blocks regime the
   solver is fastest in. Each becomes a width-3 vertex with its own 3×3 Hessian
   block; the coboundary `B` does the coupling.

2. **One mechanism yields both cones.** Tunçel's primal-dual scaling, specialized
   to 3D, is generic in the barrier. The rank-3 / single-scalar construction,
   the secant bookkeeping, and the corrector mapping are *shared verbatim*. Only
   the barrier derivatives, the shadow-primal map, the membership predicates, and
   the central starting point are per-cone. So we build a `Barrier3D` interface
   once and plug in two barriers.

3. **The scaling reduces to two vectors and one scalar.** In 3D the secant
   equations pin a rank-2 piece of the scaling *exactly*; the only freedom is a
   single scalar `t` weighting a rank-1 piece in the leftover direction. We are
   not "computing a scaling matrix" so much as computing two orthogonal
   completion vectors and one number.

> **Scope of the first implementation.** Two deliberate divergences from the
> paper, both flagged in §9 (Next steps): we use the **standard Mehrotra cube**
> for the centering parameter `σ` (not the paper's clipped rule), and we **skip
> neighborhood enforcement** entirely. Both are correctness-preserving
> simplifications that cost iterations, not solutions, on the validation
> instances. They become important on harder problems.

---

## 1. The two cones and their duals

### 1.1 Exponential cone

$$
K_{\exp} = \mathrm{cl}\{x \in \mathbb{R}^3 \mid x_1 \ge x_2 \exp(x_3/x_2),\ x_2 > 0\},
\qquad \vartheta = 3.
$$

Its barrier (paper eq. 2):

$$
F(x) = -\log\bigl(x_2 \log(x_1/x_2) - x_3\bigr) - \log x_1 - \log x_2.
$$

The dual cone is **not equal** to the primal — this is what "nonsymmetric"
means:

$$
K_{\exp}^* = \mathrm{cl}\{z \in \mathbb{R}^3 \mid e\,z_1 \ge -z_3 \exp(z_2/z_3),\ z_1 > 0,\ z_3 < 0\},
$$

where $e$ is Euler's number. The two cones are linearly isomorphic via
$T K_{\exp} = K_{\exp}^*$ with

$$
T = \begin{bmatrix} e & 0 & 0 \\ 0 & 0 & -1 \\ 0 & -1 & 0 \end{bmatrix} \succ 0.
$$

**Operational consequence:** `maxstep` must test membership in *different* cones
for the primal and dual directions. The `primal::Bool` flag selects which
membership predicate to bisect against, not (as in the symmetric case) which
Cholesky factor to use.

### 1.2 Power cone

$$
K_{\mathrm{pow}}^{\alpha} = \{(x_1, x_2, x_3) \in \mathbb{R}^3 \mid x_1^{\alpha} x_2^{1-\alpha} \ge |x_3|,\ x_1, x_2 \ge 0\},
\qquad \alpha \in (0,1),\ \vartheta = 3.
$$

Barrier:

$$
F(x) = -\log\bigl(x_1^{2\alpha} x_2^{2(1-\alpha)} - x_3^2\bigr) - (1-\alpha)\log x_1 - \alpha \log x_2.
$$

The power cone is a **one-parameter family**: each $\alpha$ is a distinct cone.
The dual is

$$
(K_{\mathrm{pow}}^{\alpha})^* = \left\{ z \mid \left(\frac{z_1}{\alpha}\right)^{\alpha}\left(\frac{z_2}{1-\alpha}\right)^{1-\alpha} \ge |z_3|,\ z_1, z_2 \ge 0 \right\}.
$$

**Why power is the *easier* cone.** Its conjugate barrier is closed-form, so the
shadow primal $\tilde x = -F^{*\prime}(s)$ is a direct formula — the one iterative
kernel that the exp cone needs (a 3D Newton, §4) simply *vanishes* for power.

---

## 2. The shared mechanism: Tunçel scaling in 3D

We need the matrix $M = W^\top W$ (the scaling Gram matrix). This is the **only**
object `hess!` requires — we never form $W$ itself.

### 2.1 Data and the rank-2 + rank-1 structure

Write the current block's primal-dual pair as $x = p_v$, $s = d_v$ (each in
$\mathbb{R}^3$). Define the **shadow iterates**

$$
\tilde s := -F'(x), \qquad \tilde x := -F^{*\prime}(s),
$$

and stack

$$
S := [\,x \mid \tilde x\,] \in \mathbb{R}^{3\times 2},
\qquad
Y := [\,s \mid \tilde s\,] \in \mathbb{R}^{3\times 2}.
$$

The 3D characterization (paper p. 20) is

$$
\boxed{\;M = Y (Y^\top S)^{-1} Y^\top + t\, z z^\top,\quad S^\top z = 0,\ \|z\| = 1,\ t > 0.\;}
$$

- The first term is **rank 2**, lives in $\mathrm{span}(Y)$, and is *entirely
  fixed by the data*.
- The second is **rank 1**, in the direction $z$ orthogonal to both columns of
  $S$, with the single scalar $t$ the only freedom.

### 2.2 Why the secants hold for any `t`

The defining property — both secant equations — is

$$
M S = Y(Y^\top S)^{-1}(Y^\top S) + t\,z(S^\top z)^\top = Y + 0 = Y.
$$

So $Mx = s$ **and** $M\tilde x = \tilde s$ hold identically, *regardless of $t$*.
This is the concrete meaning of "the secants are exact; $t$ only buys
boundedness." A bad $t$ gives a valid scaling with a worse condition number,
never a wrong one.

Two consequences worth internalizing, because they save real work downstream:

- **$Mx = s$** means the affine right-hand side `f = -d - rd` is correct
  unchanged: the `-d` term *is* `-Mx = -Mp`, exactly as in the symmetric case.
- **$M\tilde x = \tilde s = -F'(x)$** converts the centering term in the corrector
  into a *closed-form gradient*, with no conjugate-barrier evaluation needed
  (§6).

### 2.3 The on-central-path degeneracy

The condition $Y^\top S \succ 0$ is equivalent to *being off the central path*
(paper p. 19, footnote 1). On the central path, $Y^\top S$ becomes singular and
the rank-2 term is undefined. There, fall back to

$$
M = \mu_v\, F''(x), \qquad \mu_v := \langle x, s\rangle / 3,
$$

which is the $T_2(x,s,1)$ scaling.

> **This is not an edge case you can ignore.** The warm start sits at
> $x = s = -F'(x)$, which *is* on the central path. So the very first `scale!`
> call of every solve hits the degenerate branch. Guard it explicitly:
> if $\det(Y^\top S)$ is below a tolerance (or any pivot non-positive), take the
> fallback.

### 2.4 The block-local `μ` — no global leakage

The anchor $\mu_v = \langle p_v, d_v\rangle/3$ is computed from **this block's own
two vectors**. It is *not* the solver's aggregate $\mu = \langle p,d\rangle/\nu$.
The exp/pow block asks nothing of the POS/SOC/SDP blocks. `scale!(p, d, cache)`
already receives exactly $p_v, d_v$ and manufactures $\mu_v$ in place — no extra
argument, no API change.

(There is a separate, *global* `μ` in `step!` used for the centering parameter
`σ`. That one legitimately aggregates across blocks, because `σ` is global step
policy — see §7. Two different `μ`'s, two different scopes. Keep them distinct.)

---

## 3. Barrier derivatives (closed form, per cone)

The shared scaffolding calls the barrier only through $F'$, $F''$, $F'''$. Here
are both cones' formulas.

### 3.1 Exponential cone

Let $\psi(x) = x_2 \log(x_1/x_2) - x_3$ (the barrier argument, $>0$ in the
interior). Splitting $F = g + h$ with $g = -\log\psi$ and
$h = -\log x_1 - \log x_2$:

**First order.**

$$
\psi'(x) = \left(\frac{x_2}{x_1},\ \log(x_1/x_2) - 1,\ -1\right),
\qquad
h'(x) = -\left(\frac{1}{x_1}, \frac{1}{x_2}, 0\right),
$$

$$
F'(x) = -\frac{\psi'(x)}{\psi(x)} + h'(x).
$$

Hence the shadow dual, **free of charge**:

$$
\tilde s = -F'(x) = \frac{\psi'(x)}{\psi(x)} - h'(x)
= \frac{\psi'(x)}{\psi(x)} + \left(\frac{1}{x_1}, \frac{1}{x_2}, 0\right).
$$

**Second order.**

$$
\psi''(x) = \begin{bmatrix} -x_2/x_1^2 & 1/x_1 & 0 \\ 1/x_1 & -1/x_2 & 0 \\ 0 & 0 & 0 \end{bmatrix},
\qquad
h''(x) = \mathrm{diag}(1/x_1^2,\ 1/x_2^2,\ 0),
$$

$$
F''(x) = \frac{\psi'(x)\,\psi'(x)^\top}{\psi(x)^2} - \frac{\psi''(x)}{\psi(x)} + h''(x).
$$

This is symmetric PD on the interior. Form it and Cholesky it: $F'' = R R^\top$.
Keep $R$ — the corrector needs it (§6). In 3D a direct `cholesky!` on the
assembled $F''$ is simpler than transcribing the paper's hand-factored $R(x)$
(appendix A.2); the paper prefers their factored form only to avoid squaring the
condition number, which is negligible at dimension 3.

**Third order (directional).** Needed only for the corrector. From paper eq. (33),
for a direction $u$:

$$
h''(x)[u] = -2\begin{bmatrix} u_1/x_1^3 & 0 \\ 0 & u_2/x_2^3 \end{bmatrix}
\quad(\text{leading } 2\times2\text{ part}),
$$

$$
\psi''(x)[u] = \begin{bmatrix} 2x_2 u_1/x_1^3 - u_2/x_1^2 & -u_1/x_1^2 \\ -u_1/x_1^2 & u_2/x_2^2 \end{bmatrix}
\quad(\text{leading part}).
$$

The full third-order form $F'''(x)[u]$ assembles from these plus the $g$-part
terms in $\psi', \psi''$ (see §6 for how it is contracted). This is the single
fiddliest transcription in the project and where a sign error will hide.

### 3.2 Power cone

Let $\phi(x) = x_1^{2\alpha} x_2^{2(1-\alpha)} - x_3^2$ (barrier argument, $>0$ in
interior). With $F = -\log\phi - (1-\alpha)\log x_1 - \alpha\log x_2$:

**First order.**

$$
\phi'(x) = \left( 2\alpha\, \frac{\phi + x_3^2}{x_1},\ 2(1-\alpha)\, \frac{\phi + x_3^2}{x_2},\ -2x_3 \right),
$$

(using $x_1^{2\alpha}x_2^{2(1-\alpha)} = \phi + x_3^2$), and

$$
F'(x) = -\frac{\phi'(x)}{\phi(x)} - \left( \frac{1-\alpha}{x_1},\ \frac{\alpha}{x_2},\ 0 \right).
$$

**Second order.** $F''(x) = \dfrac{\phi'(x)\phi'(x)^\top}{\phi(x)^2} -
\dfrac{\phi''(x)}{\phi(x)} + \mathrm{diag}\!\left(\dfrac{1-\alpha}{x_1^2},
\dfrac{\alpha}{x_2^2}, 0\right)$, with $\phi''$ the (closed-form) Hessian of
$\phi$:

$$
\phi''_{11} = 2\alpha(2\alpha-1)\frac{\phi+x_3^2}{x_1^2},\quad
\phi''_{22} = 2(1-\alpha)(1-2\alpha)\frac{\phi+x_3^2}{x_2^2},
$$
$$
\phi''_{12} = 4\alpha(1-\alpha)\frac{\phi+x_3^2}{x_1 x_2},\quad
\phi''_{13} = \phi''_{23} = 0,\quad \phi''_{33} = -2.
$$

**Shadow primal, closed form (the payoff).** Because $(K^\alpha_{\mathrm{pow}})^*$
is itself a scaled power cone, $\tilde x = -F^{*\prime}(s)$ has an explicit
formula: $F^*$ is the power-cone barrier with $\alpha$ and a scalar adjustment.
In practice evaluate $\tilde x$ by mapping $s$ through the dual-cone barrier
gradient directly. **No Newton iteration.** (Transcribe from the closed dual
barrier; this is the one place power genuinely diverges from exp and is
*simpler*.)

**Third order** follows from $\phi'''$, which is again closed-form polynomial in
$x$ — more terms than exp but no transcendental pieces.

---

## 4. The shadow primal for EXP — the one iterative kernel

$\tilde x = -F^{*\prime}(s)$ has **no** closed form for the exponential cone. But
it is not a mysterious conjugate evaluation — it is a **gradient inversion**.
From the conjugate's stationarity (paper p. 7), $\tilde x$ is the
$\xi \in \mathrm{int}(K)$ solving

$$
F'(\xi) = -s.
$$

This is a 3D Newton iteration with $F''(\xi)$ as Jacobian (which we already know
how to form and factor):

$$
\Delta\xi = -F''(\xi)^{-1}\bigl(F'(\xi) + s\bigr),
\qquad \xi \leftarrow \xi + \theta\,\Delta\xi,
$$

damped by $\theta \in (0,1]$ to maintain $\psi(\xi) > 0$, $\xi_1, \xi_2 > 0$.
Convergence is quadratic; a handful of iterations suffice. Fully local to the
block.

**Unit test for this kernel in isolation:** pick a random interior $x$, set
$s = -F'(x)$, and confirm the Newton recovers $\xi = x$ to machine precision.
That single test fully validates the hardest ~30 lines in the exp handler.

For POW, replace this entire kernel with the closed-form $\tilde x$ from §3.2.

---

## 5. Assembling `scale!` — stage by stage

The `scale!(p, d, cache)` routine, for either cone, runs:

**Stage 1 — barrier derivatives.** Compute $F'(x)$, $F''(x)$, factor
$F'' = RR^\top$. Form $\tilde s = -F'(x)$. *(Closed form, §3.)*

**Stage 2 — shadow primal.** EXP: Newton-invert $F'(\xi) = -s$ to get
$\tilde x$ (§4). POW: closed-form $\tilde x$ (§3.2).

**Stage 3 — orthogonal completions.** Cross products:

$$
z = \frac{x \times \tilde x}{\|x \times \tilde x\|},
\qquad
r = \frac{s \times \tilde s}{\langle s \times \tilde s,\ z\rangle}
$$

so that $S^\top z = 0$, $Y^\top r = 0$, $\langle r, z\rangle = 1$. (For the BFGS
path you can skip $r$ — $M$ needs only $z$. $r$ is for the optional optimal-$t$
bisection and for $M^{-1}$.)

**Stage 4 — the scalar $t$ (BFGS, paper eq. 32).** With
$\mu_v = \langle x,s\rangle/3$ and $\tilde\mu = \langle \tilde x,\tilde s\rangle/3$:

$$
t = \mu_v \left\| F''(x) - \frac{\tilde s\,\tilde s^\top}{3}
- \frac{\bigl(F''(x)\tilde x - \tilde\mu\,\tilde s\bigr)\bigl(F''(x)\tilde x - \tilde\mu\,\tilde s\bigr)^\top}
{\langle \tilde x,\ F''(x)\tilde x\rangle - 3\tilde\mu^2} \right\|_F.
$$

Every ingredient is in hand: $F''$ (Stage 1), $\tilde s$, $\tilde x$ (Stage 2),
one matvec $F''\tilde x$, the scalar $\tilde\mu$. It is the Frobenius norm of a
$3\times3$. This is the cheap, robust choice; the paper reports it never needed
the optimal bisection (largest observed $\xi$-bound 1.72 vs. conjectured 1.253).

**Stage 5 — assemble $M$.** Form the $2\times2$

$$
Y^\top S = \begin{bmatrix} \langle s, x\rangle & \langle s, \tilde x\rangle \\
\langle \tilde s, x\rangle & \langle \tilde s, \tilde x\rangle \end{bmatrix},
$$

invert by hand, and set

$$
M = Y(Y^\top S)^{-1}Y^\top + t\, z z^\top.
$$

Guard: if $Y^\top S$ is (near) singular — the on-central-path case, including the
very first iterate — use the fallback $M = \mu_v F''(x)$ instead (§2.3).

**Cache for `corr!`:** store $M$ (9 values) and $R$ the $F''$ factor (9 values);
optionally $\tilde s, \tilde x$ (3 each) if you prefer storing to recomputing.
Roughly 18–24 $T$-values per block, dense, view-reshaped exactly like
`SDPCache`.

---

## 6. The corrector `corr!`

### 6.1 The right-hand side, after the secants collapse it

The combined search direction (paper eq. 18) has the conic right-hand side
$-v + \gamma\mu\tilde v - W^{-\top}\eta$. Mapped back through $W$ and using the two
secants $-Mx = -s$ and $M\tilde x = \tilde s = -F'(x)$, the H-applied corrector
that `corr!` must return is

$$
\boxed{\; r = -d \;-\; \sigma\mu\, F'(p) \;-\; \eta, \qquad
\eta = -\tfrac{1}{2} F'''(p)\bigl[\Delta p_a,\ F''(p)^{-1}\Delta d_a\bigr]. \;}
$$

Every term is closed-form from data `corr!` already receives — `p`, `d`, `σμ`,
`Δp` ($=\Delta p_a$), `Δd` ($=\Delta d_a$) — plus the barrier derivatives.
**The centering term needs no conjugate-barrier evaluation**, because the second
secant turned $M\tilde x$ into the closed-form gradient $-F'(p)$.

Compare the existing `poscorr!`, which returns $(\sigma\mu - \Delta p\,\Delta d)/p
- d$: same shape — baseline $-d$, a centering term, a cross-term — with the
orthant specializations swapped for the exp/pow closed forms.

> **Note on $\sigma\mu$ vs $\gamma$.** The conic centering target is the *shadow*
> $\tilde v$, equivalently the gradient $-F'(p)$ above — **not** the identity
> $e$. The scalar passed in (`σμ` in the existing signature) is $\sigma\mu$ with
> $\sigma = \gamma$; the $\sigma \leftrightarrow \gamma$ identity (§7) means you
> pass the same number you already compute.

### 6.2 Evaluating the third-order term `η`

The contraction $F'''(p)[u, v]$ with $u = \Delta p_a$ and
$v = F''(p)^{-1}\Delta d_a$:

1. Solve $F''(p)\,v = \Delta d_a$ using the cached factor $R$ (so
   $v = R^{-\top}R^{-1}\Delta d_a$). **For stability, always go through the
   factor** — never form $F''^{-1}$ explicitly. (Paper, appendix: "for stability
   we solve for $v$ using the factored expression of $F''$.")
2. Contract $F'''(p)[u, v]$ from the §3 third-order building blocks. For EXP this
   assembles from the $\psi$- and $h$-derivative pieces in eq. (33); for POW from
   the polynomial $\phi'''$. The result is a 3-vector.
3. $\eta = -\tfrac12 F'''(p)[u, v]$.

This contraction is where corrector bugs live. The validation oracle in §8.1 is
designed to catch exactly a sign or scaling error here.

---

## 7. Where this plugs into `step!` — and the two first-cut simplifications

### 7.1 The centering parameter (FIRST-CUT: standard cube)

We keep the existing computation

$$
\sigma = \mathrm{clamp}\bigl((\mu_a/\mu)^3,\ 0,\ 1\bigr),
$$

unchanged, even when exp/pow blocks are present. This is a **deliberate
divergence** from the paper, justified and de-risked in §9.1. It is
correctness-preserving: the cube only ever over-centers relative to the paper's
rule, costing iterations, not solutions.

The affine-gap progress signal $\mu_a/\mu$ remains legitimate for nonsymmetric
cones (it follows from $\langle \Delta p_a, \Delta d_a\rangle = 0$ — paper Lemma
2 — which holds for any cone). What does *not* transfer is the exactness argument
that makes the *cube exponent* the natural choice; see §9.1.

### 7.2 Neighborhood enforcement (FIRST-CUT: skipped)

The paper restricts every iterate to a neighborhood $N(\beta)$ of the central
path and cuts the step back to stay inside it. **We skip this entirely** in the
first implementation. Consequence: on hard instances the scaling approximation
$W^\top W \approx \mu F''(x)$ can degrade unguarded. Acceptable for validation;
revisit per §9.2.

### 7.3 Divide-by-zero hygiene

With any exp/pow block present, $\nu \ge 3 > 0$, so the global
$\mu = \langle p,d\rangle/\nu$ is well-defined and the cube's division is safe.
The pre-existing all-`NOC` ($\nu = 0$) guard is unrelated and still needed; just
ensure whatever guard handles it also covers any future $\nu=0$-with-exp mixing
(there shouldn't be any, since one exp block forces $\nu \ge 3$).

---

## 8. The `Barrier3D` interface and the API surface

### 8.1 What is shared vs. per-cone

The scaffolding (`scale!`, `hess!`, `corr!`, `maxstep`'s bisection loop) is
written **once** against an interface supplying:

| Interface call | EXP | POW{α} |
|---|---|---|
| `F′(x)`, `F″(x)`, `F‴(x)[u]` | §3.1 | §3.2 |
| `shadow_primal(s)` → $\tilde x$ | Newton (§4) | closed form (§3.2) |
| `in_primal_cone(x)` | $\psi(x)>0,\ x_2>0$ | $x_1^\alpha x_2^{1-\alpha} \ge \lvert x_3\rvert,\ x_1,x_2 \ge 0$ |
| `in_dual_cone(z)` | $e z_1 \ge -z_3 e^{z_2/z_3},\ z_1>0, z_3<0$ | dual power predicate |
| `central_point()` | $(1.290928, 0.805102, -0.827838)$ | $\alpha$-dependent |

Everything numerically delicate — the rank-3 update, the corrector's
$M$-mapping, the secant bookkeeping — is shared and validated **once**.

### 8.2 Cone interface signatures — unchanged

The six existing methods absorb both cones with **no signature change** for the
first implementation:

- `degree(::EXP, 3) = 3`, `degree(::POW, 3) = 3`. The `n` argument is vestigial
  (`@assert n == 3`).
- `cachesize` ≈ 18–24.
- `identity!` returns `central_point()`. **Semantic note:** for these cones there
  is no Jordan identity; `identity!` means "canonical interior start," used only
  by `initp`. Crucially `initd = initp` is *correct* because the central start
  has $p_0 = d_0$ at that triple.
- `scale!(p, d, cache)` — manufactures $\mu_v$ internally; no $\mu$ argument.
- `hess!(H, p, d, cache)` — copies cached $M$; then `axpy!` the $Q$ block.
- `corr!(r, p, d, Δp, Δd, σμ, cache)` — returns $r$ from §6.
- `maxstep(x, Δx, primal, γ, cache)` — bisection (§8.3).

### 8.3 `maxstep` by bisection

No closed form. Bisect the largest $\tau \in (0,1]$ such that
$x + \tau\,\Delta x$ stays in the cone (primal predicate) or the dual cone (dual
predicate), per the `primal` flag. Standard interval bisection on the membership
predicate; the cache is unused (as for POS/SOC).

### 8.4 The one real API wrinkle: `POW` carries a parameter

`EXP` is a nullary tag; `POW{α}` is parametrized. The current
`IPMProblem.cones::Vector{Symbol}` and `tocone(s::Symbol)` assume **nullary cone
tags**. `POW` breaks that — $\alpha$ must travel from problem specification into
the barrier formulas and `central_point()`.

Options, in increasing invasiveness:
1. Keep `cones` as symbols for the nullary cones; allow a parallel parameter
   channel for `POW` blocks (e.g. a `Dict{Int,Float64}` of vertex → α).
2. Widen `cones` to admit parametrized entries (e.g. `Union{Symbol,
   Tuple{Symbol,Float64}}`, or a small cone-spec struct).

Decide this at the `IPMProblem` boundary. `EXP` alone needs none of it, so if you
build exp first, defer the decision until pow.

---

## 9. Next steps — divergences from the paper to revisit

These are the knobs we deliberately set to "simple" for the first cut. Each is a
real, motivated improvement from Dahl-Andersen that matters on harder instances.

### 9.1 Switch to the clipped centering rule when exp/pow is present

**What the paper does.** With $\alpha_a$ the affine step to the boundary,

$$
\gamma = (1 - \alpha_a)\cdot\min\{(1-\alpha_a)^2,\ \tfrac14\}.
$$

Since $(1-\alpha_a) = \mu_a/\mu$ along the affine direction (Lemma 2), this is
**exactly the cube, clipped:**

$$
\gamma = \min\{(\mu_a/\mu)^3,\ (\mu_a/\mu)/4\}.
$$

**Why the cube is unmoored for nonsymmetric cones.** For symmetric cones the
complementarity condition $X\circ S = \mu e$ is *bilinear*; first-order Newton
plus the Mehrotra cross-term captures the *entire* nonlinearity — the corrector
is exact, and the cube exponent is tuned to that exact picture. For exp/pow the
centrality condition $s = -\mu F'(x)$ is *transcendental*. The corrector is a
second-order Taylor correction capturing one further (third-order) term, not the
whole nonlinearity. So "the corrector is exact, pick $\sigma$ to match" no longer
holds.

**When they differ.** Identical whenever the affine step is good
($\mu_a/\mu \le \tfrac12$, the usual case): there $(\mu_a/\mu)^2 \le \tfrac14$ so
the min picks the cube. They diverge only when the affine step is poor
($\mu_a/\mu \to 1$): cube $\to 1$ (full centering), clip $\to \tfrac14$ (capped).
The clip deliberately holds back hard recentering exactly where
$W^\top W \approx \mu F''$ and the shadow-target $\mu\tilde v$ are least
trustworthy.

**Recommended design.** Do **not** apply the clip globally by default — the
cube's full-centering recovery is a valid and wanted escape hatch on hard
*symmetric* instances (degenerate $B$, rank-deficient Schur complement). Instead,
attach a centering trait to each cone — `SDP/POS/SOC/NOC → :cube`,
`EXP/POW → :clipped` — and have `step!` take the **most conservative** rule among
blocks present. Since $\sigma$ is global (one residual reduction $(1-\sigma)G(z)$
is shared across all blocks), it cannot be set per-block; "max over cone types"
is the correct aggregation. The clip only ever lowers $\sigma$, so symmetric
blocks tolerate it at the cost of at most a couple extra iterations.

**Caveat to settle when implementing the clip.** The affine $\mu_a$ currently
uses *separate* $\tau_{pa}, \tau_{da}$ (`frac=1`). With $\tau_{pa} \ne \tau_{da}$
the exact $(1-\alpha)\mu$ linearity is broken (the cross-term still vanishes by
orthogonality, but the two linear terms no longer collapse to one $(1-\tau)$
factor). The cube doesn't care; the clip's $\tfrac14$ knee is sensitive to it.
Either compute $\mu_a$ at a *common* affine step, or accept that "$(1-\alpha_a)$"
means $\min(\tau_{pa}, \tau_{da})$ and tune the knee accordingly.

### 9.2 Neighborhood enforcement

Add an $N(\beta)$ neighborhood test (paper §3, with $\beta$ small, e.g.
$10^{-6}$) and cut the step back to stay inside it. This is the guardrail on the
scaling approximation; without it, the quality of the step has no protection as
iterates leave the central-path vicinity. Most valuable on ill-conditioned
instances — which, given this solver's intended frontier (coboundary matrices
that go rank-deficient when $H^1 \ne 0$), is exactly where it will eventually
matter. The neighborhood test is a per-block conic computation
($\langle F'(x_i), [F''(x_i)]^{-1} F'(x_i)\rangle$-type quantities), so it fits
the existing per-vertex loop structure.

### 9.3 Optimal bounded scaling (the bisection on `t`)

We use the BFGS scalar $t$ (eq. 32). The paper also gives a bisection for the
*optimally bounded* $t$ achieving the $\xi^*$ of eq. (22), using the
cross-product point $r$ and $Q = [r \mid S]$. They report no significant
practical difference (BFGS bound 1.72 vs. optimal $\approx 1.253$). Implement
only if a conditioning problem surfaces that traces to the scaling; keep eq. (32)
as the default.

### 9.4 Homogeneous self-dual embedding

The paper embeds in the homogeneous model, which gives clean infeasibility
*certificates* and a canonical starting point. The current solver has no such
embedding, so exp/pow problems inherit whatever the existing infeasible loop
does — and get **no infeasibility certificate**. If certificates become a
requirement, this is the (substantial) piece of work that provides them.

### 9.5 Power cone parameter plumbing

Resolve §8.4 — how $\alpha$ travels from `IPMProblem` into the cone. Only needed
once `POW` is built; `EXP` defers it entirely.

---

## 10. Verification plan

Tests in dependency order — each isolates one new piece so a failure points at
one place.

### 10.1 Barrier derivative checks (both cones)

Finite-difference $F'$ against $F$, $F''$ against $F'$, and $F'''[u]$ against
$F''$, at random interior points. Catches transcription errors in §3 before they
contaminate anything. *Cheap; do first.*

### 10.2 Shadow-primal inversion (EXP)

Random interior $x$; set $s = -F'(x)$; run the Stage-2 Newton (§4); assert
$\xi = x$ to machine precision. Fully validates the only iterative kernel.
*(For POW: check the closed-form $\tilde x$ against a finite-difference of $F^*$,
or against the same $s = -F'(x) \Rightarrow \tilde x = x$ identity.)*

### 10.3 Secant equations (the core of the shared mechanism)

After `scale!`, with $M$ assembled, assert **both**

$$
\|M x - s\| \le \epsilon, \qquad \|M\tilde x - \tilde s\| \le \epsilon
$$

to machine precision, at a random *off-central-path* $(x,s)$. Also assert $M$ is
symmetric PD (Cholesky succeeds). This validates Stages 3–5 independently of the
IPM. *If this passes, the scaling is correct regardless of `t`.*

### 10.4 On-path fallback

At $x = s = -F'(x)$ (on the central path), assert the degenerate branch triggers
and returns $M = \mu_v F''(x)$, symmetric PD. This is the path the first solver
iterate actually takes.

### 10.5 The paper's Example (19) — the integration oracle (EXP)

$$
\min\ x_1 + x_2 \quad \text{s.t.}\quad x_1 + x_2 + x_3 = 1,\quad x \in K_{\exp}.
$$

Solve with the full handler and compare the complementarity-gap sequence
$\langle x_k, s_k\rangle$ against **Table 1** of the paper:

| $k$ | with corrector |
|---|---|
| 0 | 4.0e+00 |
| 1 | 9.3e−01 |
| 2 | 4.3e−02 |
| 3 | 2.1e−04 |
| 4 | 2.7e−08 |
| 5 | 5.5e−10 |
| 6 | 8.4e−15 |

This single 3D instance exercises `scale!`, the conjugate Newton, `corr!` (the
$\eta$ contraction), and `maxstep` together. Matching the
$4.0 \to 9.3\text{e–}1 \to 4.3\text{e–}2 \to 2.1\text{e–}4 \to 2.7\text{e–}8$
cascade is tight enough to catch a sign error in $\eta$. **This is the
acceptance test for the EXP handler.** Run it both with and without the corrector
(the paper's two columns) to confirm the corrector is doing what it should — the
no-corrector column needs ~12 iterations for the same accuracy.

### 10.6 Many-block replication (both cones)

A small sheaf with one exp (or power) block per vertex coupled by a real
coboundary — e.g. an entropy cost per vertex — to exercise the
many-small-blocks path through `Caches`, the per-vertex `hess!`/`scale!`/`corr!`
loops, and the KKT solve, rather than a single isolated cone. Cross-check the
objective against Mosek/ECOS (both support exp; Mosek supports power).

### 10.7 Power cone cross-check

Power has no Table 1. Use a small p-norm or geometric-mean problem with a
known/JuMP-Mosek reference, plus the secant tests (§10.3) which carry most of the
validation weight since the scaffolding is shared with the already-validated exp
path. Power-specific risk is confined to its four §8.1 plug-in functions.

---

## 11. Build order (recommended)

1. **`Barrier3D` interface** + EXP barrier (§3.1) — pass §10.1.
2. **Shadow-primal Newton** (§4) — pass §10.2.
3. **`scale!` Stages 3–5** (§5) + fallback (§2.3) — pass §10.3, §10.4.
4. **`hess!`** (trivial: copy $M$) and **`maxstep`** (§8.3).
5. **`corr!`** (§6) — the $\eta$ contraction is the risk; pass §10.5 *without*
   corrector first (tests everything but $\eta$), then *with*.
6. **Wire into `step!`**, `tocone`, exports; full §10.5 acceptance.
7. **`POW` barrier** (§3.2) behind the same interface + closed-form $\tilde x$;
   resolve §8.4 parameter plumbing; pass §10.7.
8. **Revisit §9** as harder instances demand.

The payoff of the abstracted build: once EXP passes §10.5, POW is a barrier swap
plus a closed-form $\tilde x$ (deleting the Newton) plus two predicates and a
start point — roughly a quarter the cost of a from-scratch cone. {EXP, POW} on
top of {POS, SOC, SDP, NOC} reaches the Lubin et al. modeling-complete basis.
