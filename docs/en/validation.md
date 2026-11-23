<p align="right"><a href="../en/validation.md">English</a> · <a href="../zh/validation.md">中文</a></p>

# Validation: what this solver has been checked against

This page gives the short answer. The full record, with the conditions and the run behind every number, is in the validation volume; the realistic case has its own volume. The solver has been checked at three levels. Analytic and constitutive tests confirm that the operator is the operator it claims to be. Published benchmark solutions confirm that the whole pipeline lands on published numbers. A code-to-code comparison with ASPECT measures how far two defensible discretizations can part on the same physics. The first two say the implementation is right; the third says how much the answer moves when the discretization changes.

## Steady benchmarks

The Blankenbach et al. (1989) family on the quadtree solver. Every row is a run driven to a steady state at one refinement level, so the table shows how the answer converges rather than quoting a single point. `Nu` is the top wall flux from the one-sided GLL gradient and `Vrms` the volume RMS velocity.

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

Read the 1a rows for the method. The error drops by a factor of 3.9 between level 3 and level 4, and by 3.5 between level 4 and level 5, against the factor of 4 a second-order element predicts. 1b shows the same behavior, with ratios of 3.3 and 3.9. `Vrms` sits within 0.02% of the reference on every level. That contrast is why both columns are here: the velocity field is a large-scale quantity, while `Nu` is a one-sided gradient inside a boundary layer thinner than the wall element.

Case 1c converges more slowly, 7.81% to 4.85% to 1.56%, a ratio of 1.6 and then 3.1. Its Rayleigh number is $10^6$, a hundred times 1a's, so the wall boundary layer is thinner at the same leaf count and the wall element carries a larger share of the error at every level.

Cases 2a and 2b use the wall rule: every leaf within a strip of a wall, and the wall rows themselves, are held at a minimum level. 2b shows what that buys, moving from +4.60% at 640 leaves to +0.90% at 4 000, where the two wall fluxes agree to 0.39%. 2a lands at +0.58% on `Nu_top` and −0.15% on `Vrms` with the same rule at 1 600 leaves.

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

## Cross-code comparisons

| Case | Quantity | Ours | Other code | Difference |
|---|---|---:|---:|---:|
| blob, `Ra = 1e8`, frame 25 | `Tmax` | 1.2644 | 1.2531 (ASPECT) | +0.9% |
| realistic, ASPECT state imported | `Nu_top` | 5.990543 | 5.990543 (ASPECT) | 0.000% |
| realistic, ASPECT state imported | `Nu_bot` | -3.074986 | -3.075076 (ASPECT) | +0.003% |

The same-state rows are the sharp test. Both codes start from the same imported temperature field and evaluate the same wall functional, so a difference there is a discretization difference and nothing else. The blob row is a 25-frame transient, so its `Tmax` also carries the accumulated trajectory.

### The ASPECT run behind those numbers

The comparison runs ASPECT 2.5.0-pre at commit `340c0d018` on the same 2.5 x 1 case, with the same composite rheology, the same initial condition, and no tuned parameter; sections 5 and 6 of the validation volume record the setup. The top flux, which is the box's total heat loss, agrees to **0.005 to 0.9 percent** over `t_nd = 1e-3 .. 5e-3`, with a cross-correlation of at least 0.994 at every lag.

The bottom flux carries the two solutions' different bottom-boundary-layer history, so it is quoted with its window: **-6.11%** (consistent-boundary-flux family) or **-8.44%** (gradient family) over `t_nd = 1e-6 .. 1e-3`, crossing zero at `1.42e-3`, and **+6.05% / +5.83%** over the later window.

The same-state test separates the operator from the history. ASPECT's state at `t_nd = 9.586e-4` was imported into the QuaST tree, and both codes advanced from it: `Nu_bot` agrees to **0.003%** and `Nu_top` to **0.000%**. On a shared field the two operators agree to the last digits the comparison can resolve.

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

## The two Nusselt conventions

Three discrete families measure the wall flux. Write $\theta$ for the temperature anomaly, $H$ for the internal heating, $h$ for the wall element height, and $\langle\cdot\rangle$ for the horizontal average.

Family **A**, the one runs report as `Nu_top` and `Nu_bot`, uses the one-sided GLL derivative:

$$
\mathrm{Nu}^{A}_{\text{top}} = \left(1 + \frac{H}{2}\right) - \left\langle \frac{\partial\theta}{\partial y} \right\rangle_{\text{top}},
\qquad
\mathrm{Nu}^{A}_{\text{bot}} = \left(1 - \frac{H}{2}\right) - \left\langle \frac{\partial\theta}{\partial y} \right\rangle_{\text{bot}}.
$$

Family **B** is the steady consistent boundary flux (`semquad_nu_cbf`), the residual of the steady operator summed over the wall degrees of freedom. Family **C** is the same sum for the BDF2 step operator, which is what ASPECT's `heat flux statistics` writes. B and C agree to $10^{-4}$; A differs from them by $-\tfrac{h}{3}\,\theta''\_{\text{wall}}$ .

Quote **A**. It is the exact derivative of the discrete solution, its wall quadrature is exact, its truncation error is second order and measures $3\times10^{-5}$ to $9\times10^{-5}$ relative at the late states, and it is invariant under wall refinement at a fixed state. B is first order in the wall element, so it serves as a cross-check at fine walls and as the definition of choice when a wall is coarse. Report `Nu_top` and `Nu_bot` together with `h_wall` and `delta_T`, keep at least two wall cells across the layer and preferably four, and refine the wall strip when that fails. Use C for the energy budget, where it closes the discrete identity down to the solver floor.

## Viscoelastic verification

The Maxwell model (guide section 1.8) is verified on its own terms. The benchmarks above were all measured with `shear_modulus = 0`, so they say nothing about the stress history; these are the checks to hold a viscoelastic run to:

| Check | What was compared | Result |
|---|---|---|
| AMR transfer | the stress state before and after an AMR cycle | **bit-identical** |
| 20-step replay | a developed run's stress history, replayed from its own `(dt, u, theta)` | **bit-identical** |
| Rigid-body rotation | principal-axis error against the analytic objective solution | **1.6e-9 rad** |
| The same rotation without the co-rotational term | principal-axis error | $\pi/4$ |
| Pointwise fixed point | how closely the stored stress reproduces the pointwise fixed point | median residual 1.9e-5 on 51.6% of the nodes |
| Variable-step consistency | uniform, factor-10 alternating and random step sequences | observed order 1.988 to 1.997 |

The stress update takes the actual numerical `dt`, not a fixed elastic step, and the variable-step check is what says so: the observed order is 1.988 to 1.997 on the three sequences; no fixed-elastic-step option exists or is needed. The co-rotational term is the linearized rotation $1+\Delta t\,ad\_W$, valid while $2|\mathbf W|\min(\Delta t,\tau\_{\mathrm M})$ is small; `MAXWELL_COROTATION_BOUND = 0.1` is that band, and the driver reports the run maxima of $2|\mathbf W|\Delta t$, $2\tau\_{\mathrm M}|\mathbf W|$ and the per-step rotation, and warns once at the first step outside it. The rigid-body test is the direct check of the term: with it the principal axis follows the analytic objective solution to 1.6e-9 rad, and without it the same rotation is off by $\pi/4$.

## The reference viscosity

The equations are nondimensionalized with the reference viscosity that appears in the Rayleigh number,

$$
\mathrm{Ra} = \frac{\rho g \alpha\,\Delta T\,d^3}{\kappa\,\eta_{\mathrm{ref}}},
$$

where $\eta\_{\mathrm{ref}}$ is `Params.eta_ref` (card knob `CASE_ETA_REF`, default `1.0`) in the units the material law returns. The flow solve uses `eta_law / eta_ref` wherever the law's viscosity enters, and the default leaves every existing run bit-identical.

| Scaling | Change in the flow |
|---|---|
| the law and `eta_ref` scaled together | L2-relative 2.6e-15 in the run history, 6.3e-16 in the `theta` field |
| the law alone | L2-relative 1.37e-1 in the run history, velocity by 9.4, `Vrms` falling by `1/c` |

That is the whole convention. Multiplying `comp_Ad` and `comp_As` by `1/c` multiplies every viscosity the law returns by `c`, so the same physical problem needs `eta_ref = c`; with `eta_ref` left at its default, `Ra` keeps its value while the physical Rayleigh number moves. `eta_min` and `eta_max` are absolute bounds in the law's units, applied before the normalization, so a rescaled law must rescale them too. The card's `Ra = 1e5` describes the calibration point: on the production card the developed state has `eta_interior = 4.280635e-3` and `Ra_interior = 2.336102e7`, and the label should be quoted with `Ra_interior`. `pic_mode = :newton` builds its viscosity from the raw law and refuses `eta_ref != 1`. The guide's section on the Rayleigh-number reference viscosity documents the knob and the report a non-default `eta_ref` prints at the first flow solve and at the end of the run.

## The automated suite

`test/runtests.jl` includes the gate files for analytic identities, constitutive cards, manufactured-solution convergence, the benchmark drivers, the blob regression, checkpoint round trips, the AMR rules, and the solver paths. Run the whole suite with

```bash
julia --project=. --threads=8 test/runtests.jl
```

It takes about 25 to 30 minutes on eight threads. Each gate pins its reference state at a stated relative tolerance; the 2b snapshot gate uses `1e-6` and its symmetry identity `1e-10`, so a change that moves a number surfaces before a benchmark has to be re-run.

## Scope

The validation covers a two-dimensional Cartesian box with free-slip or rigid walls and the Boussinesq mass form used by the benchmark and cross-code runs. It covers the shipped physics where that physics is exercised: composite isostress rheology, phase transitions with per-transition latent heat, internal heating, depth-dependent yielding, and strain weakening. The TALA/ALA reference state has its own self-consistency check.

For studies in this geometry and this physics, the numbers above are the accuracy to expect, and the realistic-case volume records the resolutions and rheologies at which they were measured. Three dimensions, a spherical shell, toroidal motion, plate boundaries, magnetic fields and a free surface lie outside what this code implements, and nothing in the performance record is measured in 3-D. Full elasticity is not a separate rheology either: the one elastic behavior this release carries is the Maxwell viscoelastic model (guide section 1.8). The benchmark and cross-code numbers above are measured at the default `eta_ref = 1` and `shear_modulus = 0`; the viscoelastic and reference-viscosity sections above are the checks that cover the other settings.

Before quoting a number in a new regime, run a resolution check on the quantity you intend to quote. Convergence is necessary, and it does not by itself validate an answer.
