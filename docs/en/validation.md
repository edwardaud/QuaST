<p align="right"><a href="../en/validation.md">English</a> · <a href="../zh/validation.md">中文</a></p>

# Validation: what this solver has been checked against

This page gives the short answer. The solver has been checked at three levels. Analytic and constitutive tests confirm that the operator is the operator it claims to be. Published benchmark solutions confirm that the whole pipeline lands on published numbers. A code-to-code comparison with ASPECT measures how far two defensible discretizations can part on the same physics. The first two say the implementation is right; the third says how much the answer moves when the discretization changes.

## Steady benchmarks

The steady benchmarks are the Blankenbach et al. (1989) family on the quadtree solver. Every row is a run driven to a steady state at one refinement level, so the table shows how the answer converges rather than quoting a single point. `Nu` is the top wall flux from the one-sided GLL gradient and `Vrms` the volume RMS velocity; [guide section 1.3](guide.md#13-base-state-heating-and-nusselt-numbers) defines the Nusselt numbers and guide section 7.2 the velocity columns.

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

Read the 1a rows for the method. The error drops by a factor of 3.9 between level 3 and level 4, and by 3.5 between level 4 and level 5, against the factor of 4 a second-order element predicts. 1b shows the same behavior, with ratios of 3.3 and 3.9. In 1a and 1b, `Vrms` sits within 0.02% of the reference on every level. That contrast is why both columns are here: the velocity field is a large-scale quantity, while `Nu` is a one-sided gradient inside a boundary layer thinner than the wall element.

Case 1c converges more slowly, 7.81% to 4.85% to 1.56%, a ratio of 1.6 and then 3.1. Its Rayleigh number is $10^6$ , a hundred times 1a's, so the wall boundary layer is thinner at the same leaf count and the wall element carries a larger share of the error at every level.

Cases 2a and 2b use the wall rule of [guide section 4.3](guide.md#43-hysteresis-and-the-wall-rule): every leaf within a strip of a wall, and the wall rows themselves, are held at a minimum level. 2b shows what that buys, moving from +4.60% at 640 leaves to +0.90% at 4 000, where the two wall fluxes agree to 0.39%. The 4.60% offset at level 3 is the wall boundary layer being under-resolved, not an implementation error, and agreement between the top and bottom fluxes is the steady-state acceptance test. Uniform refinement does not substitute for the wall rule: a uniform level-4 run on 2b (80 x 32) stopped before convergence at +5.31%. 2a lands at +0.58% on `Nu_top` and −0.15% on `Vrms` with the same rule at 1 600 leaves.

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

Stage 0.5 adds a second geometry to the solver, the closed annular ring, and with it the only published 2-D cylindrical-shell convection benchmark: the isoviscous free-slip $Ra = 10^5$ case of Davies et al. (2022) and of the ASPECT documentation. The card is $R\_0 = 1.22$, $R\_1 = 2.22$ (the gap $d = 1$ is the length unit), $T = 1$ at the inner circle and $T = 0$ at the outer, radial gravity of constant magnitude folded into $Ra = \rho g \alpha \Delta T d^3 / (\eta \kappa)$, no internal heating, free slip on both circles, the paper's initial condition (its Eq. 41), and the closed periodic ring, whose seam is a topology feature and not a wall. The two reference values are the Firedrake row of Davies et al. (2022, Table A1: $Nu = 9.541$, $V\_{RMS} = 193.26$ at $2048 \times 512$) and the ASPECT table at $1536 \times 128$ ($Nu\_T = 9.548$, $Nu\_B = 9.584$, $V\_{RMS} = 193.371$, $\langle T \rangle = 0.382$). The two codes differ by 0.07 % on $Nu$ and 0.06 % on $V\_{RMS}$; those differences are the tolerances this page uses: 0.5 % on $Nu$ and 0.2 % on $V\_{RMS}$ at ASPECT-like resolution, 0.1 % at Firedrake-like resolution.

The discrete card is Q2-Q1-Q2 on the quadtree with Galerkin advection and no entropy viscosity, matching the paper's Taylor-Hood element pair and its unstabilised steady formulation. The stop criterion is stated in the harness and gated: after the start-up, the relative temperature change and the changes of $Nu\_T$ and $V\_{RMS}$ per outer step stay below $10^{-6}$ for three consecutive outer steps.

The published Firedrake mesh, $2048 \times 512$ = 1 048 576 leaves or 8 396 800 velocity DOFs, cannot be run on this machine. The ladder's measured peak RSS grows from 2.3 GB at 12 288 leaves to 13.7 GB at 196 608 leaves, i.e. by 63–69 kB per leaf, so the Firedrake mesh needs **66–72 GB** against the 25 GB the shared-memory solver has, and the measured profile extrapolates it to at least two hours of wall clock even if it fit. The ladder therefore stops at $128 \times 1536$ = 196 608 leaves and 1 579 008 velocity DOFs, which is the ASPECT-like published resolution (the card's DOF arithmetic reproduces Table A1's Firedrake counts at $2048 \times 512$ exactly: 8 396 800 velocity and 4 198 400 temperature DOFs). Reaching a steady state there with CFL-limited time stepping would need thousands of steps to $t \sim 1$; the runs below use the documented steady-relaxation mode instead: a large fixed step makes the time derivative negligible, so each energy solve is the steady advection-diffusion problem and the step sequence is the Stokes/energy Picard iteration, damped at `pic_relax = 0.5` because the undamped iteration oscillates instead of converging. The fixed point is the same discrete steady state for every step size, because the time derivative vanishes there; the record checks it two ways: a time-accurate march at $16 \times 192$ approaches the same state from its transient (within 0.3 % on all four quantities at $t = 0.15$, still drifting), and the damped iteration at $16 \times 192$ with `pic_relax = 0.3` reaches the same fixed point as the $0.5$ run to eight significant figures.

| Resolution | Leaves | Velocity DOFs | Wall clock | Peak RSS | Steps | $Nu\_T$ | $Nu\_B$ | $V\_{RMS}$ | $\langle T \rangle$ | $\Delta Nu\_T$ vs ASPECT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $16 \times 192$ | 3 072 | 25 344 | 113 s | 1.9 GB | 6 | 10.31665 | 10.26887 | 193.1271 | 0.382440 | +8.05% |
| $32 \times 384$ | 12 288 | 99 840 | 193 s | 2.3 GB | 6 | 9.89994 | 9.94385 | 193.2584 | 0.382464 | +3.69% |
| $64 \times 768$ | 49 152 | 396 288 | 424 s | 4.8 GB | 6 | 9.64611 | 9.66529 | 193.2601 | 0.382466 | +1.03% |
| $128 \times 1536$ | 196 608 | 1 579 008 | 1 391 s | 13.7 GB | 6 | 9.56743 | 9.57303 | 193.2599 | 0.382465 | **+0.20%** |

The wall clocks are the harness's measured driver-call time (which includes the first-call JIT); Julia start-up, package load and space construction add about 1.5 minutes per rung. The steps are outer steps: each one is a converged burst of Picard iterations, so the first step carries the convergence and the following ones are cheap.

`V_RMS` is the large-scale quantity, and it is already right on the coarse meshes: it lies within 0.13 % of the ASPECT value at every rung and within 0.07 % of the Firedrake value from $32 \times 384$ on, well inside the 0.2 % tolerance. $\langle T \rangle$ lies within $5 \times 10^{-4}$ of 0.382 everywhere, inside the three published digits. The two walls agree with each other to 0.46 % at $16 \times 192$, 0.44 % at $32 \times 384$ and 0.20 % at $64 \times 768$, against ASPECT's own 0.38 % wall-flux split at its published mesh; it decreases with resolution as the resolution diagnostic should.

`Nu_T` is the resolution-sensitive quantity, and the convergence is second order: the error falls 8.05 % → 3.69 % → 1.03 % → 0.20 %, ratios 2.2, 3.6 and 5.0 against the 4 a second-order element predicts. At the ASPECT-like published resolution the top flux is within **0.20 %** of ASPECT's 9.548 and the bottom flux within 0.11 % of 9.584, inside the 0.5 % tolerance the two reference codes' 0.07 % spread motivates. The error is a wall boundary-layer measurement error, not a different steady state: the same field gives `V_RMS` and $\langle T \rangle$ to better than 0.13 %, exactly the pattern of the box benchmarks above.

Two implementation questions the annulus stage had left open are quantified on these states. First, the **pressure pin**: on a curved free-slip wall only the nodal normal velocity is Dirichlet, so the constant pressure is only approximately in the null space of the reduced gradient and the nodal pin is a regularisation. The dropped continuity row is the volume integral of $\nabla \cdot u$; measured on the converged states it is $1.5 \times 10^{-13}$ at $16 \times 192$, $5.3 \times 10^{-15}$ at $32 \times 384$, $3.5 \times 10^{-14}$ at $64 \times 768$ and $9.1 \times 10^{-14}$ at $128 \times 1536$, i.e. $10^{-17}$–$10^{-18}$ of the velocity scale: the pin is an exact gauge on this discretisation to quadrature round-off, not merely a stable regularisation. Two controls agree. The same physical mesh with a different base tessellation (identical node set, different leaf and DOF ordering) reproduces the $32 \times 384$ state to $1.6 \times 10^{-7}$ absolute ($2 \times 10^{-8}$ relative) on `Nu_T`. And because the pin is the first free pressure DOF — the logical origin node on a uniform tree — moving it relative to the flow needs the ring's rotational symmetry: the four-fold initial condition rotated by $\pi/8$ (24 angular cells, grid aligned) has the same discrete solution rotated, so every diagnostic is identical if the pin is gauge-neutral; the measured shift is $-4 \times 10^{-9}$ absolute on `Nu_T` ($-4 \times 10^{-8}$ relative) at $32 \times 384$, about six orders of magnitude below the 0.1 % relative level the wall-flux ratios were suspected to be sensitive to. Second, the **advection scheme**: because the paper's transport stabilisation is not part of the fetched record, one $32 \times 384$ state was also run with the conservative `:skew` form instead of Galerkin; it moves `Nu_T` by $+1.8 \times 10^{-3}$ relative (+0.018 %, `Nu_B` +0.026 %, `V_RMS` +0.002 %), an order of magnitude below the 0.1 % level and two below the 0.5 % tolerance, so the choice does not move the comparison.

The net solid-body rotation is removed from the returned velocity once per solve (the solver's `:rotation` gauge, engaged in every run above), which is the convention both reference codes use; without it $V\_{RMS}$ is gauge dependent. No separate per-iteration removal was needed: the converged `V_RMS` matches the published values directly at every rung where it is resolved.

**What this comparison does and does not establish.** Three levels are kept separate. *Reproduced:* the published steady scalars at the ASPECT-like published resolution, within the tolerances the two reference codes' own 0.07 % spread supports: $V\_{RMS}$ to 0.06 % of 193.371, $Nu\_T$ to 0.20 % of 9.548 and $Nu\_B$ to 0.11 % of 9.584, $\langle T \rangle$ to the third published digit; and the wall-flux split is 0.06 %, smaller than ASPECT's own 0.38 % split at its mesh, decreasing with resolution as the benchmark expects. *Deviates:* $Nu\_T$ is still 0.20 % high at the finest mesh. That is a wall boundary-layer error converging at second order (the error falls by 2.2, 3.6 and 5.0), but it has not reached the 0.1 % Firedrake-like tolerance, which is defined at a mesh 5.3 times larger; the transport scheme is plain Galerkin, stated as a choice because the fetched record does not fix the paper's stabilisation. *Could not be run:* the Firedrake-like $2048 \times 512$ mesh, for the measured memory reason above; any number quoted for it here is an extrapolation, not a measurement. The relaxation mode also means the paper's **transient** is not reproduced: the benchmark's published targets are steady-state scalars and the transient is not one of them, and the record's time-accurate coarse check approaches the same state from its transient but does not claim the paper's trajectory. Nothing was tuned to hit a target: the card, the advection scheme, the relaxation and the resolutions were fixed before the runs, and the deviations above are what came out.

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

| Case | Quantity | Ours | Other code | Difference |
|---|---|---:|---:|---:|
| blob, `Ra = 1e8`, frame 25 | `Tmax` | 1.2643 | 1.2531 (ASPECT) | +0.9% |
| realistic, ASPECT state imported | `Nu_top` | 5.990543 | 5.990543 (ASPECT) | 0.000% |
| realistic, ASPECT state imported | `Nu_bot` | -3.074986 | -3.075076 (ASPECT) | +0.003% |

The same-state rows are the sharp test. Both codes start from the same imported temperature field, ASPECT's state at `t_nd = 9.586e-4`, and evaluate the same wall functional, so a difference there is a discretization difference and nothing else. The blob row is a 25-frame transient, so its `Tmax` also carries the accumulated trajectory.

The comparison runs ASPECT 2.5.0-pre on the same 2.5 x 1 case, with the same composite rheology, the same initial condition, and no tuned parameter. The top flux, which is the box's total heat loss, agrees to **0.005 to 0.9 percent** over `t_nd = 1e-3 .. 5e-3`, with a cross-correlation of at least 0.994 at every lag.

The bottom flux carries the two solutions' different bottom-boundary-layer history, so it is quoted with its window: **-6.11%** (consistent-boundary-flux family) or **-8.44%** (gradient family) over `t_nd = 1e-6 .. 1e-3`, crossing zero at `1.42e-3`, and **+6.05% / +5.83%** over the later window. On a shared field the two operators agree to the last digits the comparison can resolve.

## The Tosi family

Tosi et al. (2015) extends the Blankenbach cases with a viscoplastic rheology. Case 1 has a temperature-dependent viscosity only and reaches a steady state. Case 5a adds a yield stress and settles into a limit cycle, so it is described by a period and a min/max pair rather than a single number. The reference values are the medians of the eleven participating codes, Table 2 of the paper.

| Case | Quantity | Ours | Reference | Δ |
|---|---|---:|---:|---:|
| 1 | `Nu_top` | 3.4216 | 3.4298 | **−0.24%** |
| 1 | `Vrms` | 249.53 | 251.80 | **−0.91%** |
| 5a | period | 0.0783 | 0.0792 | **−1.1%** |
| 5a | `Nu_top`, minimum | 2.6924 | 2.6950 | **−0.1%** |
| 5a | `Nu_top`, maximum | 7.4889 | 7.2792 | +2.9% |
| 5a | mean `T`, minimum | 0.6505 | 0.6543 | −0.6% |
| 5a | mean `T`, maximum | 0.6686 | 0.6722 | −0.5% |

Both cases land inside the spread the eleven codes reported for themselves, which is the resolution the benchmark can offer. Case 5a carries particular weight: a limit cycle is hard to reach by accident, and its period, amplitude and mean temperature all come out in the right place at once.

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

`test/runtests.jl` includes 196 gate files covering analytic identities, constitutive cards, manufactured-solution convergence, the benchmark drivers (including the 2-D annulus card and its steady-state criterion), the blob regression, checkpoint round trips, the AMR rules, the boundary-condition surface, the regional case, and the solver paths. Run the whole suite with

```bash
julia --project=. --threads=8 test/runtests.jl
```

The last full run recorded 137 405 assertions: 137 403 passed, 0 failed, and 2 broken (36 minutes on eight threads, shared with the benchmark runs). The two broken entries wait for checkpoints the repository does not ship. Each test pins its reference state at a stated relative tolerance; the benchmark snapshot pins its 20-step trajectory at `1e-6` relative and the symmetric case asserts `Nu_top = Nu_bot` to `1e-10`, so a change that moves a number surfaces before a benchmark has to be re-run. The same-state dump protocol keeps the box path bitwise: its nine state hashes are `0dda6974…545c` on both the stage-0 commit and this working tree.

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

For studies in this geometry and this physics, the numbers above are the accuracy to expect. The benchmark and cross-code numbers are measured at the default `eta_ref = 1` and `shear_modulus = 0`; the viscoelastic and reference-viscosity sections above are the checks that cover the other settings.

Before quoting a number in a new regime, run a resolution check on the quantity you intend to quote. Convergence is necessary, and it does not by itself validate an answer.

**Next:** [FAQ](faq.md) — the practical questions these numbers raise.
