<p align="right"><a href="../en/validation.md">English</a> · <a href="../zh/validation.md">中文</a></p>

# Validation: what this solver has been checked against

This page gives the short answer. The solver has been checked at three levels. Analytic and constitutive tests confirm that the operator is the operator it claims to be. Published benchmark solutions confirm that the whole pipeline lands on published numbers. A code-to-code comparison with ASPECT measures how far two defensible discretizations can part on the same physics. The first two say the implementation is right; the third says how much the answer moves when the discretization changes.

## Steady benchmarks

The steady benchmarks are the Blankenbach et al. (1989) family on the quadtree solver. Every row is a run driven to a steady state at one refinement level, so the table shows how the answer converges rather than quoting a single point. `Nu` is the top wall flux from the one-sided GLL gradient and `Vrms` the volume RMS velocity; [guide section 1.3](guide.md#13-base-state-heating-and-nusselt-numbers) defines the Nusselt numbers and guide section 7.2 the velocity columns. The 1a, 1b, 1c and 2a runs behind this table are archived under [benchmarks/results/](../../benchmarks/results/README.md): one record per quoted number with the command, the input card and its SHA-256, the resolution, the environment and the log's SHA-256, and the compact logs themselves under `benchmarks/results/logs/`.

| Case | `Ra` | Viscosity | Level | Leaves | `Nu` | Reference | Δ | `Vrms` | Δ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1a | $10^4$ | constant | 3 | 1 024 | 4.936942 | 4.884409 | +1.08% | 42.8638 | −0.00% |
| 1a | $10^4$ | constant | 4 | 4 096 | 4.897925 | 4.884409 | +0.28% | 42.8616 | −0.01% |
| 1a | $10^4$ | constant | 5 | 16 384 | 4.888163 | 4.884409 | **+0.08%** | 42.8724 | +0.02% |
| 1b | $10^5$ | constant | 3 | 1 024 | 10.988574 | 10.534095 | +4.31% | 193.2127 | −0.00% |
| 1b | $10^5$ | constant | 4 | 4 096 | 10.669987 | 10.534095 | +1.29% | 193.2139 | −0.00% |
| 1b | $10^5$ | constant | 5 | 16 384 | 10.568610 | 10.534095 | **+0.33%** | 193.2137 | −0.00% |
| 1c | $10^6$ | constant | 3 | 1 024 | 23.687756 | 21.972465 | +7.81% | 833.3367 | −0.08% |
| 1c | $10^6$ | constant | 4 | 4 096 | 23.037486 | 21.972465 | +4.85% | 833.9917 | −0.00% |
| 1c | $10^6$ | constant | 5 | 16 384 | 22.315960 | 21.972465 | **+1.56%** | 834.0082 | +0.00% |
| 2a | $10^4$ | $\eta=e^{-\ln(10^3) T}$ | 3 + walls 4 | 1 600 | 10.124814 | 10.066 | **+0.58%** | 479.695 | −0.15% |
| 2b | $10^4$ | temperature and depth | 3 | 640 | 7.248805 | 6.9299 | +4.60% | 170.213 | −0.90% |
| 2b | $10^4$ | temperature and depth | 3 + walls 4 | 1 120 | 7.212885 | 6.9299 | +4.08% | 170.633 | −0.65% |
| 2b | $10^4$ | temperature and depth | 4 + walls 5 | 4 000 | 6.992620 | 6.9299 | **+0.90%** | 171.634 | −0.07% |

Read the 1a rows for the method. The deviations from the published value fall by a factor of 3.9 between level 3 and level 4, and by 3.5 between level 4 and level 5. 1b shows the same behavior, with ratios of 3.3 and 3.9. In 1a and 1b, `Vrms` sits within 0.02% of the reference on every level. That contrast is why both columns are here: the velocity field is a large-scale quantity, while `Nu` is a one-sided gradient inside a boundary layer thinner than the wall element.

Case 1c converges more slowly, 7.81% to 4.85% to 1.56%, a ratio of 1.6 and then 3.1. Those are ratios of the deviations, not an observed order: the successive differences do not decay, so no order is defined (see the uncertainty paragraph below). Its Rayleigh number is $10^6$ , a hundred times 1a's, so the wall boundary layer is thinner at the same leaf count and the wall element carries a larger share of the error at every level.

Cases 2a and 2b use the wall rule of [guide section 4.3](guide.md#43-hysteresis-and-the-wall-rule): every leaf within a strip of a wall, and the wall rows themselves, are held at a minimum level. 2b shows what that buys, moving from +4.60% at 640 leaves to +0.90% at 4 000, where the two wall fluxes agree to 0.39%. The 4.60% offset at level 3 is the wall boundary layer being under-resolved, not an implementation error, and agreement between the top and bottom fluxes is the steady-state acceptance test. Uniform refinement does not substitute for the wall rule: a uniform level-4 run on 2b (80 x 32) stopped before convergence at +5.31%. 2a lands at +0.58% on `Nu_top` and −0.15% on `Vrms` with the same rule at 1 600 leaves.

**What is measured, what is not.** Computed from the three finest recorded values of each ladder with the standard Richardson/Roache procedure (refinement ratio 2, safety factor 1.25), the observed order and the GCI of the finest value are: 1a **1.999 / 0.083 %** (Richardson value 4.884906, +0.010 % from the published 4.884409); 1b **1.652 / 0.560 %**, and the level-5 deviation +0.33 % is 0.59 of that GCI. Case 1c has no observed order: the successive differences do not decay (0.650 then 0.722) and a separate re-run L2–L4 ladder changes sign, so no GCI exists and its +1.56 % deviation carries no measured uncertainty; a percent-level bracket is the honest reading. Case 2a's quoted row is a wall-refined, non-uniform tree, so no Richardson analysis is defined for it; the uniform-tree L1–L4 study of the same card measures an order of 1.271 and a GCI of 2.88 % (1.21 % on `Vrms`), which brackets the quoted +0.58 % but is a property of the uniform card, not of the quoted rung. The three 2b rows are archived in
[blankenbach2b_nu-L3toL4_steady.md](../../benchmarks/results/blankenbach2b_nu-L3toL4_steady.md):
the level-3 and level-3+walls-4 runs end with their closing summaries, while the
level-4+walls-5 run was stopped by the host's out-of-memory killer before its summary, so its
evidence is the last printed step (7 550, top/bottom fluxes agreeing to 0.39 %).  That is why
the 2b rows are the least certain ones in the table.

### The 2b flux families

2b has a published record at a wall-refined level-4 state where the three Nusselt definitions can be compared against each other:

| State | `Nu` family | Value | Reference | Δ |
|---|---|---:|---:|---:|
| level 3, 40 x 16 | wall gradient | 6.9428 | 6.9299 | +0.19% |
| level 3, 40 x 16 | energy, historical | 6.9874 | 6.9299 | +0.83% |
| wall-refined L4 | wall gradient | 6.9480 | 6.9299 | +0.26% |
| wall-refined L4 | steady CBF | 6.9325 | 6.9299 | +0.04% |
| wall-refined L4 | energy | 6.9330 | 6.9299 | +0.05% |

On the wall-refined state the three families span 0.22%. That agreement is the reason the convergence table quotes the plain wall gradient: the choice of definition does not move the answer at this resolution.

## The 2-D cylindrical annulus

Stage 0.5 adds a second geometry to the solver, the closed annular ring, and with it the only published 2-D cylindrical-shell convection benchmark: the isoviscous free-slip $Ra = 10^5$ case of Davies et al. (2022) and of the ASPECT documentation. The card is $R\_0 = 1.22$, $R\_1 = 2.22$ (the gap $d = 1$ is the length unit), $T = 1$ at the inner circle and $T = 0$ at the outer, radial gravity of constant magnitude folded into $Ra = \rho g \alpha \Delta T d^3 / (\eta \kappa)$, no internal heating, free slip on both circles, the paper's initial condition (its Eq. 41), and the closed periodic ring, whose seam is a topology feature and not a wall. The two reference values are the Firedrake row of Davies et al. (2022, Table A1: $Nu = 9.541$, $V\_{RMS} = 193.26$ at $2048 \times 512$) and the ASPECT table at $1536 \times 128$ ($Nu\_T = 9.548$, $Nu\_B = 9.584$, $V\_{RMS} = 193.371$, $\langle T \rangle = 0.382$). The two codes differ by 0.07 % on $Nu$ and 0.06 % on $V\_{RMS}$. That is a spread between two single-resolution reference values, not a discretization uncertainty of either code: with the finest rung's own GCI (computed below) and an ASPECT-side error at least of the order of its 0.38 % wall-flux split, the total comparison uncertainty on $Nu$ is roughly **0.5–0.7 %**. The tolerances this page quotes — 0.5 % on $Nu$ and 0.2 % on $V\_{RMS}$ at ASPECT-like resolution, 0.1 % at Firedrake-like resolution — are a stated acceptance policy in that band, not a claim that a deviation inside 0.5 % is an accuracy.

The discrete card is Q2-Q1-Q2 on the quadtree with Galerkin advection and no entropy viscosity, matching the paper's Taylor-Hood element pair and its unstabilised steady formulation. The stop criterion is stated in the harness and gated: after the start-up, the relative temperature change and the changes of $Nu\_T$ and $V\_{RMS}$ per outer step stay below $10^{-6}$ for three consecutive outer steps.

The published Firedrake mesh, $2048 \times 512$ = 1 048 576 leaves or 8 396 800 velocity DOFs, cannot be run on this machine. The ladder's measured peak RSS grows from 2.3 GB at 12 288 leaves to 13.7 GB at 196 608 leaves, i.e. by 63–69 kB per leaf, so the Firedrake mesh needs **66–72 GB** against the 25 GB the shared-memory solver has, and the measured profile extrapolates it to at least two hours of wall clock even if it fit. The ladder therefore stops at $128 \times 1536$ = 196 608 leaves and 1 579 008 velocity DOFs, which is the ASPECT-like published resolution (the card's DOF arithmetic reproduces Table A1's Firedrake counts at $2048 \times 512$ exactly: 8 396 800 velocity and 4 198 400 temperature DOFs). Reaching a steady state there with CFL-limited time stepping would need thousands of steps to $t \sim 1$; the runs below use the documented steady-relaxation mode instead: a large fixed step makes the time derivative negligible, so each energy solve is the steady advection-diffusion problem and the step sequence is the Stokes/energy Picard iteration, damped at `pic_relax = 0.5` because the undamped iteration oscillates instead of converging. The fixed point is the same discrete steady state for every step size, because the time derivative vanishes there; the archived record [annulus_steady-dt_L4_relaxation.md](../../benchmarks/results/annulus_steady-dt_L4_relaxation.md) measures it: the $16 \times 192$ fixed point is the same for $dt = 10^3, 10^4, 10^5$ and for `pic_relax` $0.3$ and $0.5$, with a mutual spread of $6.8 \times 10^{-8}$ relative in $Nu\_T$, and the returned state satisfies the discrete equations to about $10^{-8}$ relative in $u$ and $\theta$ while the dropped continuity row stays at $10^{-17}$ of the velocity scale. A time-accurate march at $16 \times 192$ approaches the same state from its transient (within 0.3 % on all four quantities at $t = 0.15$, still drifting), and an under-damped run at `pic_relax = 0.8` did not converge inside the record's budget, consistent with the note that the undamped iteration oscillates.

| Resolution | Leaves | Velocity DOFs | Wall clock | Peak RSS | Steps | $Nu\_T$ | $Nu\_B$ | $V\_{RMS}$ | $\langle T \rangle$ | $\Delta Nu\_T$ vs ASPECT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $16 \times 192$ | 3 072 | 25 344 | 113 s | 1.9 GB | 6 | 10.31665 | 10.26887 | 193.1271 | 0.382440 | +8.05% |
| $32 \times 384$ | 12 288 | 99 840 | 193 s | 2.3 GB | 6 | 9.89994 | 9.94385 | 193.2584 | 0.382464 | +3.69% |
| $64 \times 768$ | 49 152 | 396 288 | 424 s | 4.8 GB | 6 | 9.64611 | 9.66529 | 193.2601 | 0.382466 | +1.03% |
| $128 \times 1536$ | 196 608 | 1 579 008 | 1 391 s | 13.7 GB | 6 | 9.56743 | 9.57303 | 193.2599 | 0.382465 | **+0.20%** |

The wall clocks are the harness's measured driver-call time (which includes the first-call JIT); Julia start-up, package load and space construction add about 1.5 minutes per rung. The steps are outer steps: each one is a converged burst of Picard iterations, so the first step carries the convergence and the following ones are cheap.

The ladder converges, but it does not by itself give the numerical uncertainty. Computed from the three finest rungs with the standard Richardson/Roache procedure (refinement ratio $r = 2$, safety factor $F\_{s} = 1.25$; record [annulus_nu-t_128x1536_rung4.md](../../benchmarks/results/annulus_nu-t_128x1536_rung4.md)), the observed order of $Nu\_T$ is **1.690** and the GCI of the finest value is **0.462 %**; $Nu\_B$ gives **1.594** and **0.597 %**. The Richardson value with the measured order is 9.5321 for $Nu\_T$ and 9.5273 for $Nu\_B$. $V\_{RMS}$ and $\langle T \rangle$ change sign between the two finest rungs, so no observed order and no GCI exist for them; they are converged to about $10^{-5}$ relative and $10^{-6}$ absolute instead. With $p = 2$ assumed instead of measured the GCI of $Nu\_T$ is 0.343 % and the Richardson value is 9.5412, which is the independent Firedrake value 9.541 to four digits: that supports the $p = 2$ assumption but does not measure the order, which stays 1.69.

`V_RMS` is the large-scale quantity, and it is already right on the coarse meshes: it lies within 0.13 % of the ASPECT value at every rung and within 0.07 % of the Firedrake value from $32 \times 384$ on, well inside the 0.2 % tolerance. $\langle T \rangle$ lies within $5 \times 10^{-4}$ of 0.382 everywhere, inside the three published digits. The two walls agree with each other to 0.46 % at $16 \times 192$, 0.44 % at $32 \times 384$ and 0.20 % at $64 \times 768$, against ASPECT's own 0.38 % wall-flux split at its published mesh; it decreases with resolution as the resolution diagnostic should.

`Nu_T` is the resolution-sensitive quantity. Its deviation from ASPECT falls 8.05 % → 3.69 % → 1.03 % → 0.20 % over the ladder, but those are error ratios, not an observed order: the measured order on the three finest rungs is **1.69**, below the nominal 2, so the sequence is still pre-asymptotic. What the deviation does show is agreement: the finest value is 0.2035 % above ASPECT's 9.548, which is 0.44 of the run's own 0.462 % GCI, so the values are **consistent with ASPECT inside their discretization uncertainty**. That is not an accuracy claim of 0.20 %: the finest rung's discretization uncertainty is 0.34–0.46 % (0.34 % if $p = 2$ is assumed), and that band is the number to quote next to the deviation. The bottom flux deviates −0.114 % from 9.584, 0.19 of its 0.597 % GCI. The error is a wall boundary-layer measurement error, not a different steady state: the same field gives `V_RMS` and $\langle T \rangle$ to better than 0.13 %, exactly the pattern of the box benchmarks above.

Two implementation questions the annulus stage had left open are quantified on these states. First, the **pressure pin**: on a curved free-slip wall only the nodal normal velocity is Dirichlet, so the constant pressure is only approximately in the null space of the reduced gradient and the nodal pin is a regularisation. The dropped continuity row is the volume integral of $\nabla \cdot u$; measured on the converged states it is $1.5 \times 10^{-13}$ at $16 \times 192$, $5.3 \times 10^{-15}$ at $32 \times 384$, $3.5 \times 10^{-14}$ at $64 \times 768$ and $9.1 \times 10^{-14}$ at $128 \times 1536$, i.e. $10^{-17}$–$10^{-18}$ of the velocity scale: the pin is an exact gauge on this discretisation to quadrature round-off, not merely a stable regularisation. Two controls agree. The same physical mesh with a different base tessellation (identical node set, different leaf and DOF ordering) reproduces the $32 \times 384$ state to $1.6 \times 10^{-7}$ absolute ($2 \times 10^{-8}$ relative) on `Nu_T`. And because the pin is the first free pressure DOF — the logical origin node on a uniform tree — moving it relative to the flow needs the ring's rotational symmetry: the four-fold initial condition rotated by $\pi/8$ (24 angular cells, grid aligned) has the same discrete solution rotated, so every diagnostic is identical if the pin is gauge-neutral; the measured shift is $-4 \times 10^{-9}$ absolute on `Nu_T` ($-4 \times 10^{-8}$ relative) at $32 \times 384$, about six orders of magnitude below the 0.1 % relative level the wall-flux ratios were suspected to be sensitive to. Second, the **advection scheme**: because the paper's transport stabilisation is not part of the fetched record, one $32 \times 384$ state was also run with the conservative `:skew` form instead of Galerkin; it moves `Nu_T` by $+1.8 \times 10^{-3}$ relative (+0.018 %, `Nu_B` +0.026 %, `V_RMS` +0.002 %), an order of magnitude below the 0.1 % level and two below the 0.5 % tolerance, so the choice does not move the comparison.

The net solid-body rotation is removed from the returned velocity once per solve (the solver's `:rotation` gauge, engaged in every run above), which is the convention both reference codes use; without it $V\_{RMS}$ is gauge dependent. No separate per-iteration removal was needed: the converged `V_RMS` matches the published values directly at every rung where it is resolved.

**What this comparison does and does not establish.** Three levels are kept separate. *Reproduced:* the published steady scalars at the ASPECT-like published resolution, within the comparison uncertainty stated above: $V\_{RMS}$ to 0.06 % of 193.371, $Nu\_T$ to 0.20 % of 9.548 (0.44 of its own GCI), $Nu\_B$ to 0.11 % of 9.584, $\langle T \rangle$ to the third published digit; and the wall-flux split is 0.06 %, smaller than ASPECT's own 0.38 % split at its mesh, decreasing with resolution as the benchmark expects. *Not an accuracy claim:* the finest rung's own discretization uncertainty is 0.34–0.46 % on $Nu\_T$ (0.34 % with $p = 2$ assumed) and 0.60 % on $Nu\_B$; the 0.20 % is the deviation, not the error bar, and it has not reached the 0.1 % Firedrake-like tolerance, which is defined at a mesh 5.3 times larger. The convergence is not measured to be second order: the observed order is 1.69 on $Nu\_T$ and 1.59 on $Nu\_B$, and only the $p = 2$ Richardson extrapolation — an assumption, though one that lands on the independent Firedrake value — is consistent with the reference. The transport scheme is plain Galerkin, stated as a choice because the fetched record does not fix the paper's stabilisation. *Could not be run:* the Firedrake-like $2048 \times 512$ mesh, for the measured memory reason above; any number quoted for it here is an extrapolation, not a measurement. The relaxation mode also means the paper's **transient** is not reproduced: the benchmark's published targets are steady-state scalars and the transient is not one of them, and the record's time-accurate coarse check approaches the same state from its transient but does not claim the paper's trajectory. Nothing was tuned to hit a target: the card, the advection scheme, the relaxation and the resolutions were fixed before the runs, and the deviations above are what came out.

## Boundary conditions

The per-wall surface of [guide section 2](guide.md#2-boundary-conditions) is verified against exact solutions whose boundary data are the inputs. Uniform viscosity and polynomial data make Q2 velocity and Q1 pressure exact, and the face quadrature exact for the polynomial traction, so every number below is the linear solve's round-off and not a discretization error.

| Check | What was compared | Measured |
|---|---|---|
| Traction-driven Couette | uniform viscosity `eta0 = 3`, tangential traction `tau0 = 2` on one wall, the other three no slip; exact `u = (tau0 y/eta0, 0)`, `p = 0` | velocity error 7.6e-15, pressure 2.6e-12 |
| Normal traction on an open wall | a rigid box with the top `:open` and `wall_tn = T`; exact `u = 0`, `p = -T` | `max|u| = 3.6e-16`, `max|p + T|/T = 8.7e-15` |
| Channel Poiseuille | an inflow normal velocity, no-slip channel walls and an open outflow carrying the exact total traction, in both driving modes | velocity 9.9e-15, pressure 2.3e-13, net boundary flux 1e-16 |
| Regional `validate` | the Couette–Poiseuille channel of the regional case, 13 gated analytic checks | 13 of 13 pass (velocity 1.4e-14, pressure 1.3e-12, net flux 9.0e-16) |

Each row discriminates something a wrong implementation would get wrong. The traction-driven Couette fixes the sign and the total-Cauchy-traction convention of `wall_t`, and its pressure check rules out a spurious pressure load. The normal-traction row gives the *data* as the exact pressure (`-T`, not the pinned zero), so a pinned gauge or a hidden outward-normal sign table fails it. The channel row needs all three mechanisms together: without the inflow normal velocity nothing drives the region, without the tangential traction the outflow is a different flow, and without the open wall with `wall_tn` the outflow is not at a prescribed pressure. The regional `validate` mode re-runs the same construction through the production driver and the wall-refined tree. The derived gauge is checked in the same tests: `LAST_SEMSTOKES_PRESSURE_GAUGE` is `:none` whenever a wall is open and `:pin` otherwise.

Every boundary addition defaults to `nothing`, and the closed free-slip card is unchanged: the pre-existing regression pins of the default path reproduce byte-identical state dumps.

## Cross-code comparisons

The ASPECT side is pinned and was run for this repository in the 2026-11-26 batch. The runs were
made inside the Docker image
`gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac`
(the tag `gassmoeller/aspect:latest` currently resolves to it; created 2023-04-19), which reports
**ASPECT 2.5.0-pre (main, 340c0d018), deal.II 9.4.0**. The image ships that commit twice: `aspect`
in DEBUG mode (the default on `PATH`) and `aspect-release` in OPTIMIZED mode. The measured cost is
the limitation that decided what could be run: on the Blankenbach 2b card at refinement 3 the
DEBUG build ran **1.56 s per step** against **0.0905 s per step** for `aspect-release` (17 x), i.e.
about 1.8 h against 6 min for the full 4 076-step card. Every comparison below therefore uses
`aspect-release`, and each record states the DEBUG cost measured beside it. The image digest, both
version banners and the commands are archived in
[logs/aspect_environment.txt](../../benchmarks/results/logs/aspect_environment.txt); the image is
identified by the digest, not by the mutable tag. The runs are launched from Windows PowerShell
with an ASCII mount because Docker Desktop's WSL integration is off in this environment.

### Reproduced from the pinned image

The published ASPECT rows below are now reproduced by this repository's own runs, at the mesh each
card ships with (or one refinement lower where the shipped mesh does not run in this image, see
below), and not copied from the ASPECT documentation table. Each is archived as a provenance
record under `benchmarks/results/`.

| Case | Quantity | ASPECT run | Reference it reproduces | ASPECT deviation | Record |
|---|---|---:|---|---:|---|
| Blankenbach 2b, ref 3 (40 x 16) | `Nu` | 6.445128 | image's shipped ASPECT ref-3 record 6.44510878 | +0.0003 % | [aspc-blankenbach2b_nu-vrms_ref3.md](../../benchmarks/results/aspc-blankenbach2b_nu-vrms_ref3.md) |
| Blankenbach 2b, ref 3 | `Vrms` | 168.5755 | 168.574818 (same record) | +0.0004 % | same |
| Davies 1.1, ref 5 (384 x 32) | `Nu_T` | 2.464699 | published ASPECT ref-7 row 2.464 | +0.028 % | [aspc-davies-case11_nu-vrms_ref5.md](../../benchmarks/results/aspc-davies-case11_nu-vrms_ref5.md) |
| Davies 1.1, ref 5 | `Nu_B` | 2.464702 | 2.468 | −0.134 % | same |
| Davies 1.1, ref 5 | `V_rms` | 19.051767 | 19.053 | −0.0065 % | same |
| Davies 1.1, ref 5 | mean `T` | 0.402951114 | 0.403 | −0.012 % | same |
| Davies 2.1, ref 4 (192 x 16) | `Nu_T` | 4.699036 | 4.7000 | −0.020 % | [aspc-davies-case21_nu-vrms_ref4.md](../../benchmarks/results/aspc-davies-case21_nu-vrms_ref4.md) |
| Davies 2.1, ref 4 | `Nu_B` | 4.699070 | 4.706 | −0.147 % | same |
| Davies 2.1, ref 4 | `V_rms` | 46.2422496 | 46.244 | −0.0038 % | same |
| Davies 2.1, ref 4 | mean `T` | 0.382056192 | 0.382 | +0.0147 % | same |
| Davies 2.2, ref 4 (192 x 16) | `Nu_T` | 9.511780 | 9.548 | −0.379 % | [aspc-davies-case22_nu-vrms_ref4.md](../../benchmarks/results/aspc-davies-case22_nu-vrms_ref4.md) |
| Davies 2.2, ref 4 | `Nu_B` | 9.500254 | 9.584 | −0.874 % | same |
| Davies 2.2, ref 4 | `V_rms` | 193.126704 | 193.371 | −0.126 % | same |
| Davies 2.2, ref 4 | mean `T` | 0.382437930 | 0.382 | +0.115 % | same |

The Blankenbach row is the reproduction check on the image itself: the shipped card runs to the
shipped termination criterion in exactly the 4 076 steps the image's own readme lists, and returns
the shipped ASPECT ref-3 value to six significant figures. The Davies rows are lower-resolution
reproductions of the published ASPECT ref-7 table taken from the shipped cards, and all of them
land within 0.9 %.

Where both codes were run on the same card, the comparison can be made at the same resolution.
That is the case at 3 072 cells, where the repository's `16 x 192` annulus rung and the ASPECT
ref-4 runs of cases 2.1 and 2.2 meet (same cells, same Q2-Q1-Q2 element pair, and identical
degree-of-freedom counts: 25 344 velocity + 3 264 pressure + 12 672 temperature on both sides):

| Case, 3 072 cells | Quantity | ASPECT ref 4 | QuaST `16 x 192` | Difference |
|---|---|---:|---:|---:|
| Davies 2.1, Ra = 1e4 | `V_rms` | 46.2422496 | 46.24225 | +0.000001 % |
| Davies 2.1, Ra = 1e4 | mean `T` | 0.382056192 | 0.382056 | −0.00005 % |
| Davies 2.1, Ra = 1e4 | `Nu_T` | 4.699036 | 4.862962 | +3.49 % |
| Davies 2.2, Ra = 1e5 | `V_rms` | 193.126704 | 193.12706 | −0.0002 % |
| Davies 2.2, Ra = 1e5 | mean `T` | 0.382437930 | 0.382440 | −0.0005 % |
| Davies 2.2, Ra = 1e5 | `Nu_T` | 9.511780 | 10.316646 | −7.80 % |

The state is the same to better than 1e-5 relative at both Rayleigh numbers; the wall-flux
functional at that coarse mesh is not. The QuaST number is its one-sided GLL wall gradient (family
A, [below](#the-three-nusselt-conventions)), whose coarse-mesh error on this case is the +8.05 %
measured in the ladder record
[annulus_nu-t_128x1536_rung4.md](../../benchmarks/results/annulus_nu-t_128x1536_rung4.md);
ASPECT's `heat flux statistics` is the consistent boundary flux (family C), which is already within
0.4-0.9 % of the ref-7 value at this mesh. This is the same pattern as the box benchmarks: velocity
and mean temperature are large-scale quantities and converge first; the wall gradient is a
boundary-layer measurement that needs the wall resolved.

The same-mesh comparison exists for Blankenbach 2b too, and it shows the same thing
from the other direction: at 40 x 16 cells ASPECT reads `Nu = 6.445128` (-6.995 % from the
published value) where QuaST reads 7.248805 (+4.60 %), while `Vrms` agrees to 0.97 %.  The two
codes are unconverged in the wall flux at that mesh and read it differently (one-sided GLL
gradient versus consistent boundary flux), so the 12.5 % same-mesh `Nu` difference is a
discretisation difference, not a physical disagreement; see
[the 2b record](../../benchmarks/results/blankenbach2b_nu-L3toL4_steady.md). The QuaST side of the Ra = 1e5 row is the
recorded ladder rung; the Ra = 1e4 row is a run of the same card for this comparison with only
`Ra` changed (the run script is archived as
[logs/annulus_ra1e4_16x192_script.jl](../../benchmarks/results/logs/annulus_ra1e4_16x192_script.jl)
and differs from `examples/annulus_benchmark.jl` in that one line; its log is
[logs/annulus_ra1e4_16x192.log](../../benchmarks/results/logs/annulus_ra1e4_16x192.log)), free slip
on both circles, `R0 = 1.22`, `R1 = 2.22`, periodic seam, `deta_T = 1`. The archived script's
printed card line still says `Ra = 1e5` because that label is hard-coded in the example's banner
text; the parameter line (`Params(Ra=1.0e4, ...)`) is what the run used, and the log records the
Steady-state criterion firing (`steady=1`, drifts 6.8e-7 and 1.5e-7).

Case 2.2 is also the one annulus case with a second, independent published row: the Firedrake
values of Davies et al. (2022, Table A1), `Nu = 9.541` and `V_RMS = 193.26` at 2048 x 512. The
refinement-4 ASPECT run is −0.31 % from that `Nu` (9.511780) and −0.07 % from that `V_RMS`
(193.126704). The repository's own annulus section above relates its rungs to both reference rows.

### What remains same-group or single-state

| Case | Quantity | Ours | Other code | Difference |
|---|---|---:|---:|---:|
| blob, `Ra = 1e8`, frame 25 | `Tmax` | 1.2643 | 1.2531 (ASPECT) | +0.9% |
| realistic, ASPECT state imported | `Nu_top` | 5.990543 | 5.990543 (ASPECT) | 0.000% |
| realistic, ASPECT state imported | `Nu_bot` | -3.074986 | -3.075076 (ASPECT) | +0.003% |

These three rows do not change. They are ASPECT 2.5.0-pre runs made by the same group on the same
development card, one imported state and one transient frame, so they are consistency checks of the
shared state and the shared wall functional, not independent validation. Their ASPECT-side output
was not archived in the earlier batch, and the `realistic` composite-rheology card was not re-run in
the new campaign: it needs the material-model plugin built and a production-length run, so the
campaign's time box went to the published benchmark cards, which are the only ASPECT-side numbers
that can be checked against an independent reference. The same-state rows remain the sharp test
they were: both codes start from the same imported temperature field, ASPECT's state at
`t_nd = 9.586e-4`, and evaluate the same wall functional, so a difference there is a
discretization difference and nothing else. The blob row is a 25-frame transient, so its `Tmax`
also carries the accumulated trajectory.

The comparison runs ASPECT 2.5.0-pre on the same 2.5 x 1 case, with the same composite rheology,
the same initial condition, and no tuned parameter. The top flux, which is the box's total heat
loss, agrees to **0.005 to 0.9 percent** over `t_nd = 1e-3 .. 5e-3`, with a cross-correlation of at
least 0.994 at every lag. The bottom flux carries the two solutions' different bottom-boundary-layer
history, so it is quoted with its window: **-6.11%** (consistent-boundary-flux family) or
**-8.44%** (gradient family) over `t_nd = 1e-6 .. 1e-3`, crossing zero at `1.42e-3`, and
**+6.05% / +5.83%** over the later window. On a shared field the two operators agree to the last
digits the comparison can resolve. The same-state test separates the operators from the history:
ASPECT's state at `t_nd = 9.586e-4` is imported into the QuaST mesh and both codes advance from
it, `Nu_bot` agrees to **0.003%** and `Nu_top` to **0.000%**.

### What could not be run, and the measured limitation

The published ASPECT mesh for the Davies cases is 7 global refinements (1536 x 128, about 2.5
million unknowns). It was not attempted in this batch. The reason is measured, not assumed: the
DEBUG build is 17 x slower per step, and the shipped free-slip cards (`case-2.1.prm`,
`case-2.2.prm`, `case-2.3.prm`) at their shipped 5 refinements **abort in timestep 1 in this
pinned image** with `AztecOO::Iterate error code -3: loss of precision` in the block-Schur
preconditioner's velocity solve. The abort reproduces with 1 and 4 MPI ranks, with the DEBUG and
the OPTIMIZED binary, with `Remove nullspace` disabled and with a looser linear-solver tolerance;
it tracks the 12 288-cell mesh, not the card physics. The cards run at refinement 4, which is what
the reproduced rows above use; refinement 5 also runs when the cheap Stokes solver is disabled
(`Number of cheap Stokes solver steps = 0`), but at a measured 5.1 s/step, and the published
1536 x 128 mesh remains out of reach for this batch. The abort logs are archived beside the
case-2.1 record. The three Davies cases that have only ASPECT's own table as a reference (1.1,
2.1, 2.3) are single-code values: reproduction from the pinned image, such as the rows above, is a
check of the image and the card, not agreement with an independent code.

Case 2.3 (variable viscosity, `eta = 1000^(-T)`) is the one case that was attempted and did not
produce a comparison. At refinement 4 it reached the card's `End time = 3` (step 2 056) but is not
steady there: `V_rms` still rises by 4.9 % over the last 100 steps and sits at 89.6 against the
published 79.6, mean `T` is 0.501 against 0.582, at a measured 2.16 s/step with 4 MPI ranks (the
worst of the four cases, because the variable-viscosity Stokes solve needs about 219 iterations per
step). The published case-2.3 values are steady-state scalars at 1536 x 128, so this run's numbers
are recorded as an unfinished comparison and marked UNVERIFIED-in-full in the batch record; they
are not quoted here as a reproduced value.

## The Tosi family

Tosi et al. (2015) extends the Blankenbach cases with a viscoplastic rheology. Case 1 has a temperature-dependent viscosity only and reaches a steady state. Case 5a adds a yield stress and settles into a limit cycle, so it is described by a period and a min/max pair rather than a single number.

Both cases are run on a $32 \times 32$ box with the solver's Tosi law. The card carries $Ra\_{\mathrm{eff}} = Ra \cdot \eta\_0 \cdot \eta\_{\mathrm{ref}} = 100$, the benchmark's Rayleigh number: the shipped default is $\eta\_0 = 0.1$ with $Ra = 1000$ (the example now defaults to that), which is the same physical problem as the paper's $\eta\_0 = 1$, $Ra = 100$, because the flow solve applies the law as $\eta\_{\mathrm{law}}/\eta\_{\mathrm{ref}}$ with the law already normalised by $\eta\_0$. The reference values in the first table are the **YACC column** of Table 2, not the medians; the second table gives the medians of the participating codes.

| Case | Quantity | Ours | YACC | Δ |
|---|---|---:|---:|---:|
| 1 | `Nu_top` | 3.4216 | 3.4298 | **−0.24%** |
| 1 | `Vrms` | 249.37 | 251.80 | **−0.96%** |
| 5a | period | 0.0783 | 0.0792 | **−1.1%** |
| 5a | `Nu_top`, minimum | 2.6924 | 2.6950 | **−0.1%** |
| 5a | `Nu_top`, maximum | 7.4889 | 7.2792 | +2.9% |
| 5a | mean `T`, minimum | 0.6505 | 0.6543 | −0.6% |
| 5a | mean `T`, maximum | 0.6686 | 0.6722 | −0.5% |

| Case | Quantity | Ours | Median | Δ |
|---|---|---:|---:|---:|
| 1 | `Nu_top` | 3.4216 | 3.4214 | +0.007% |
| 1 | `Vrms` | 249.37 | 249.573 | −0.08% |
| 5a | period | 0.0783 | 0.07686 | +1.9% |
| 5a | `Nu_top`, minimum | 2.6924 | 2.6899 | +0.09% |
| 5a | `Nu_top`, maximum | 7.4889 | 7.4030 | +1.2% |
| 5a | mean `T`, minimum | 0.6505 | 0.65102 | −0.08% |
| 5a | mean `T`, maximum | 0.6686 | 0.66921 | −0.09% |

The YACC column is one participating code, so a deviation from it is one code-to-code difference. The medians are 3.4214 and 249.573 for Case 1 (all eleven codes), and 0.07686 / 2.6899 / 7.4030 / 0.65102 / 0.66921 for Case 5a (the ten codes that resolved the limit cycle; MC3D found a steady state and its values are not part of the periodic band). Case 1 lands inside the eleven-code spread on both diagnostics and matches the medians. Case 5a's period, `Nu_top` minimum and mean-temperature range land inside the ten-code spread; its `Nu_top` maximum, 7.4889, is 0.8 % above the largest reported value (7.4314) and so is the one number that lies outside the spread. Case 5a carries particular weight because a limit cycle is hard to reach by accident, and its period, amplitude and mean temperature all come out in the right place at once.

The comparison is nonetheless single-resolution, and the uncertainties are not measured. Case 1 runs at the reference card's $32 \times 32$, but Case 5a runs at $32 \times 32$ while its reference card is $64 \times 64$, and no resolution study exists for either case: the deviations above are differences against single-resolution reference values, not accuracies. The paper's own spread is the only uncertainty the benchmark supplies.

## The three Nusselt conventions

Three discrete families measure the wall flux. Write $\theta$ for the temperature anomaly, $H$ for the internal heating, $h$ for the wall element height, and $\langle\cdot\rangle$ for the horizontal average.

Family **A** is the one runs report as `Nu_top` and `Nu_bot`. It uses the one-sided GLL derivative, and [guide section 1.3](guide.md#13-base-state-heating-and-nusselt-numbers) states it.

Family **B** is the steady consistent boundary flux (`semquad_nu_cbf`), the residual of the steady operator summed over the wall degrees of freedom. Family **C** is the same sum for the BDF2 step operator, which is what ASPECT's `heat flux statistics` writes. B and C agree to $10^{-4}$ ; A differs from them by $-\tfrac{h}{3}\,\theta''\_{\text{wall}}$ .

Quote **A**. It is the exact derivative of the discrete solution, its wall quadrature is exact, its truncation error is second order and measures $3\times10^{-5}$ to $9\times10^{-5}$ relative at the late states, and it is invariant under wall refinement at a fixed state. B is first order in the wall element, so it serves as a cross-check at fine walls and as the definition of choice when a wall is coarse. Report `Nu_top` and `Nu_bot` together with `h_wall` and `delta_T`, keep at least two wall cells across the layer and preferably four, and refine the wall strip when that fails. Use C for the energy budget, where it closes the discrete identity down to the solver floor.

## Viscoelastic verification

The Maxwell model of [guide section 1.8](guide.md#18-viscoelasticity) is verified on its own terms. The benchmarks above were all measured with `shear_modulus = 0`, so they say nothing about the stress history; these are the checks to hold a viscoelastic run to:

| Check | What was compared | Result |
|---|---|---|
| AMR transfer | the stress state before and after an AMR cycle | **bit-identical** |
| 20-step replay | a developed run's stress history, replayed from its own `(dt, u, theta)` | **bit-identical** |
| Rigid-body rotation | principal-axis error against the analytic objective solution | **1.6e-9 rad** |
| The same rotation without the co-rotational term | principal-axis error | $\pi/4$ |
| Pointwise fixed point | how closely the stored stress reproduces the pointwise fixed point | median residual 1.9e-5 on 51.6% of the nodes |
| Variable-step consistency | uniform, factor-10 alternating and random step sequences | observed order 1.988 to 1.997 |

The stress update takes the actual numerical `dt`, not a fixed elastic step, and the variable-step check is what says so: the observed order is 1.988 to 1.997 on the three sequences; no fixed-elastic-step option exists or is needed. The rigid-body test is the direct check of the co-rotational term: with it the principal axis follows the analytic objective solution to 1.6e-9 rad, and without it the same rotation is off by $\pi/4$ . The validity band of that term, and the report a run prints when it leaves the band, are in [guide section 1.8](guide.md#18-viscoelasticity).

## The reference viscosity

The equations are nondimensionalized with the reference viscosity that appears in the Rayleigh number, as [guide section 1.9](guide.md#19-the-reference-viscosity-and-the-rayleigh-number) states. The convention is checked by rescaling the material law and the reference viscosity together, and by rescaling the law alone:

| Scaling | Change in the flow |
|---|---|
| the law and `eta_ref` scaled together | L2-relative 2.6e-15 in the run history, 6.3e-16 in the `theta` field |
| the law alone | L2-relative 1.37e-1 in the run history, velocity by 9.4, `Vrms` falling by `1/c` |

The card's nominal `Ra` describes the calibration point, not the interior; [guide section 1.9](guide.md#19-the-reference-viscosity-and-the-rayleigh-number) gives the developed-state values and the report that a non-default `eta_ref` prints. `pic_mode = :newton` builds its viscosity from the raw law and refuses `eta_ref != 1`.

## The automated suite

`test/runtests.jl` includes 207 gate files covering analytic identities, constitutive cards, manufactured-solution convergence, the benchmark drivers (including the 2-D annulus card and its steady-state criterion), the blob regression, checkpoint round trips, the AMR rules, the boundary-condition surface, the regional case, and the solver paths. Run the whole suite with

```bash
julia --project=. --threads=8 test/runtests.jl
```

The one committed record of a full-suite run — the assertion totals and the two broken entries — is [test/README.md](../../test/README.md); the tree has 207 gate files, `runtests.jl` includes each of them exactly once, and `scripts/check_counts.jl` checks that record and the other documented surfaces against the tree. The two broken entries wait for checkpoints the repository does not ship. Each test pins its reference state at a stated relative tolerance; the benchmark snapshot pins its 20-step trajectory at `1e-6` relative and the symmetric case asserts `Nu_top = Nu_bot` to `1e-10`, so a change that moves a number surfaces before a benchmark has to be re-run. The same-state dump protocol keeps the box path bitwise: its nine state hashes are `0dda6974…545c`, unchanged through the geometry/mapping work and archived in [benchmarks/results/](../../benchmarks/results/README.md).

## Scope

The validation covers a two-dimensional Cartesian box with free-slip, rigid or open walls, and the 2-D polar annulus (a closed periodic ring or a wedge with symmetry cuts) with free-slip, rigid or open curved walls, on the Boussinesq mass form used by the benchmark and cross-code runs. The anelastic form with a TALA/ALA reference state has its own self-consistency check. It covers the shipped physics where that physics is exercised: composite isostress rheology, phase transitions with per-transition latent heat, internal heating, depth-dependent yielding, and strain weakening.

What the solver cannot do is this list:

- Three dimensions.
- Any geometry other than a Cartesian box or the 2-D polar annulus: no spherical shell, no 3-D spherical geometry, no dipping interface, no non-axis-aligned wall other than the annulus circles, no free surface.
- A free surface or topography.
- Compositional fields or tracers, and therefore no chemical buoyancy.
- Melt transport.
- Pressure- and temperature-dependent material properties from a table.
- Full compressibility: Boussinesq or anelastic only.
- Finite-strain elasticity: Maxwell only.
- Distributed-memory parallelism: shared-memory threads only. This is the measured limit on the annulus benchmark: the published Firedrake-like mesh needs 66–72 GB at the measured memory per leaf, against the 25 GB this machine has.
- Time-dependent boundary data.
- VTK or other visualization formats: the frames are raw binary plus CSV.

Several features are verified but not validated against independent data. No published comparison exists here for the anelastic/TALA reference state, phase transitions with latent heat, yielding or strain weakening beyond the Tosi 5a limit cycle (5a adds yielding but no strain weakening), the curved-wall `:rigid` and `:open` conditions (the annulus benchmark is free slip only), internal heating $H \neq 0$, or the effect of AMR on a benchmark answer (the benchmark ladders use fixed, non-adaptive trees). Those features are covered by analytic solutions and self-consistency identities — the sections above and the automated suite — not by published reference values. The viscoelasticity section is verification for the same reason.

For studies in this geometry and this physics, the deviations above and the uncertainties stated next to them are what to expect; they are not a guarantee. The benchmark and cross-code numbers are measured at the default `eta_ref = 1` and `shear_modulus = 0`; the viscoelastic and reference-viscosity sections above are the checks that cover the other settings.

Before quoting a number in a new regime, run a resolution check on the quantity you intend to quote. Convergence is necessary, and it does not by itself validate an answer.

**Next:** [FAQ](faq.md) — the practical questions these numbers raise.
