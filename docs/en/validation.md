<p align="right"><a href="../en/validation.md">English</a> · <a href="../zh/validation.md">中文</a></p>

# Validation: how far can you trust this solver?

This page gives the short answer. The full record, with the conditions and the run behind every number, is in [the validation volume](../dev/VALIDATION.md); the realistic case has its own [volume](../dev/CASE_REALISTIC.md). The solver has been checked at three levels. Analytic and constitutive tests confirm that the operator is the operator it claims to be. Published benchmark solutions confirm that the whole pipeline lands on published numbers. A code-to-code comparison with ASPECT measures how far two defensible discretizations can part on the same physics. The first two say the implementation is right; the third says how much the answer can move when the discretization changes.

## Steady benchmarks

The Blankenbach et al. (1989) family, run to a steady state on the quadtree solver. Each row is one converged run at one refinement level, so the table shows how the answer converges rather than quoting a single point. `Nu` is the top wall flux from the one-sided GLL gradient and `Vrms` the volume RMS velocity. A run counts as converged only when the top and bottom fluxes agree, which they must in a steady state with no internal heating.

| Case | `Ra` | Viscosity | Level | `Nu` | Reference | Δ | `Vrms` | Δ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1a | $10^4$ | constant | 3 | 4.936942 | 4.884409 | +1.08% | 42.8638 | −0.00% |
| 1a | $10^4$ | constant | 4 | 4.897925 | 4.884409 | +0.28% | 42.8616 | −0.01% |
| 1a | $10^4$ | constant | 5 | 4.888163 | 4.884409 | **+0.08%** | 42.8724 | +0.02% |
| 1b | $10^5$ | constant | 3 | 10.988574 | 10.534095 | +4.31% | 193.2127 | −0.00% |
| 1b | $10^5$ | constant | 4 | 10.669987 | 10.534095 | **+1.29%** | 193.2139 | −0.00% |
| 1c | $10^6$ | constant | 3 | 23.687756 | 21.972465 | +7.81% | 833.3367 | −0.08% |
| 1c | $10^6$ | constant | 4 | 23.037486 | 21.972465 | **+4.85%** | 833.9917 | −0.00% |
| 2a | $10^4$ | $\eta=e^{-\ln(10^3) T}$ | 3 | not converged | 10.066 | — | 474.63 | −1.21% |
| 2b | $10^4$ | temperature and depth | 3 | 7.248805 | 6.9299 | +4.60% | 170.213 | −0.90% |

Read the 1a rows for the method. The error drops by a factor of 3.9 between level 3 and level 4, and by 3.5 between level 4 and level 5, against the factor of 4 a second-order element predicts. `Vrms` meanwhile sits at 0.01% on the coarsest level already. That contrast is why both columns are here: the velocity field is a large-scale quantity, while `Nu` is a one-sided gradient inside a boundary layer thinner than the wall element.

The 2a row at level 3 says "not converged" for a reason. Its two wall fluxes disagree by 5.4% (`Nu_top = 10.455`, `Nu_bot = 11.024`), and with no internal heating they can only differ while the temperature field is still evolving. Quoting its `Nu` would repeat the mistake this table exists to prevent.

Four runs were still in flight when this page was written: 1b level 5, 1c level 5, 2a level 4, and 2b level 4 with wall refinement. Their rows will replace or extend the entries above.

### The 2b flux families

The 2b case has a published record at a wall-refined level-4 state, where the three Nusselt definitions can be compared against each other:

| State | `Nu` family | Value | Reference | Δ |
|---|---|---:|---:|---:|
| level 3, 40 x 16 | wall gradient | 6.9428 | 6.9299 | +0.19% |
| level 3, 40 x 16 | energy, historical | 6.9874 | 6.9299 | +0.83% |
| wall-refined L4 | wall gradient | 6.9480 | 6.9299 | +0.26% |
| wall-refined L4 | steady CBF | 6.9325 | 6.9299 | +0.04% |
| wall-refined L4 | energy | 6.9330 | 6.9299 | +0.05% |

On the wall-refined state the three families span 0.22%. On a uniform coarse mesh they spread much wider, which is why the convergence table quotes the plain wall gradient and keeps the family comparison here. Uniform refinement alone does not close the gap: going from a uniform level 3 to a uniform level 4 changed `Nu` from +4.60% to +5.31% on this case, while wall refinement reached +0.26% with far fewer cells.

## Cross-code comparisons

| Case | Quantity | Ours | Other code | Difference |
|---|---|---:|---:|---:|
| blob, `Ra = 1e8`, frame 25 | `Tmax` | 1.2644 | 1.2531 (ASPECT) | +0.9% |
| realistic, ASPECT state imported | `Nu_top` | 5.990543 | 5.990543 (ASPECT) | 0.000% |
| realistic, ASPECT state imported | `Nu_bot` | -3.074986 | -3.075076 (ASPECT) | +0.003% |

The same-state rows are the sharp test. Both codes start from the same imported temperature field and evaluate the same wall functional, so any difference there is a discretization difference and nothing else. The blob row is a 25-frame transient, so its `Tmax` also carries the accumulated trajectory.

### The ASPECT run behind those numbers

The comparison runs ASPECT 2.5.0-pre at commit `340c0d018` on the same 2.5 x 1 case, with the same composite rheology, the same initial condition, and no tuned parameter; sections 5 and 6 of [the validation volume](../dev/VALIDATION.md) record the setup. The top flux, which is the box's total heat loss, agrees to **0.005 to 0.9 percent** over `t_nd = 1e-3 .. 5e-3`, with a cross-correlation of at least 0.994 at every lag.

The bottom flux does not agree. The difference is **-6.11%** (consistent-boundary-flux family) or **-8.44%** (gradient family) over `t_nd = 1e-6 .. 1e-3`, crosses zero at `1.42e-3`, and reaches **+6.05% / +5.83%** over the later window, up to +8.5% late.

> [!WARNING]
> The sign of that difference depends on the time window, so a bottom-flux number quoted without its window says nothing.

The same-state test separates the operator from the history. ASPECT's state at `t_nd = 9.586e-4` was imported into the QuaST tree, and both codes advanced from it: `Nu_bot` agrees to **0.003%** and `Nu_top` to **0.000%**. So the two operators agree on a shared field, and the 6 to 9% gap is accumulated bottom-boundary-layer history. A provenance study prices five candidate mechanisms: the definition costs 0.7%, wall resolution at most about 1 point, artificial viscosity 7 to 15%, the realization 2 to 4% with the wrong sign, and phases are excluded. That leaves 85 to 90% attributed to the different bottom-layer structure of the two solutions, with the mechanism still open.

## The two Nusselt conventions

Three discrete families measure the wall flux. Write $\theta$ for the temperature anomaly, $H$ for the internal heating, $h$ for the wall element height, and $\langle\cdot\rangle$ for the horizontal average.

Family **A**, the one runs report as `Nu_top` and `Nu_bot`, uses the one-sided GLL derivative:

$$
\mathrm{Nu}^{A}_{\text{top}} = \left(1 + \frac{H}{2}\right) - \left\langle \frac{\partial\theta}{\partial y} \right\rangle_{\text{top}},
\qquad
\mathrm{Nu}^{A}_{\text{bot}} = \left(1 - \frac{H}{2}\right) - \left\langle \frac{\partial\theta}{\partial y} \right\rangle_{\text{bot}}.
$$

Family **B** is the steady consistent boundary flux (`semquad_nu_cbf`), the residual of the steady operator summed over the wall degrees of freedom. Family **C** is the same sum for the BDF2 step operator, which is what ASPECT's `heat flux statistics` writes. B and C agree to $10^{-4}$; A differs from them by $-\tfrac{h}{3}\,\theta''\_{\text{wall}}$ .

Quote **A**. It is the exact derivative of the discrete solution, its wall quadrature is exact, its truncation error is second order and measures $3\times10^{-5}$ to $9\times10^{-5}$ relative at the late states, and it is invariant under wall refinement at a fixed state. B is only first order in the wall element, so treat it as a cross-check: at `h = 1/16` at the bottom wall it returns +1.34 where A returns -4.47, an error of 433% relative to the B value, and at `h = 1/64` the error falls to 0.0095%. At the production states the A-to-C spread is 0.27 to 1.19% at the top and 0.005 to 0.046% at the bottom. Report `Nu_top` and `Nu_bot` together with `h_wall` and `delta_T`, require at least two wall cells across the layer and preferably four, and refine the wall strip when that fails. Use C for the energy budget, where it closes the discrete identity down to the solver floor.

## The automated suite

`test/runtests.jl` includes 156 gate files: analytic identities, constitutive cards, manufactured-solution convergence, the 2b driver, the blob regression, checkpoint round trips, the AMR rules, and the solver paths. Run the whole suite with

```bash
julia --project=. --threads=4 test/runtests.jl
```

It takes about 22 minutes on eight threads. Each gate pins its reference state at a stated relative tolerance; the 2b snapshot gate uses `1e-6` and its symmetry identity `1e-10`, so a change that moves a number surfaces before a benchmark has to be re-run.

## Limits

The validation covers a **two-dimensional Cartesian box** with free-slip or rigid walls and the Boussinesq mass form used by the benchmark and cross-code runs. It covers the shipped physics where that physics is exercised: composite isostress rheology, phase transitions with per-transition latent heat, internal heating, depth-dependent yielding, and strain weakening. The TALA/ALA reference state has its own self-consistency check. It does not cover three dimensions, a spherical shell, toroidal motion, plate boundaries, viscoelasticity, compressible MHD, or a free surface, and nothing in the performance record is measured in 3-D.

The benchmark set is finite too. Within the Blankenbach family, 1a, 1b, 1c, 2a and 2b now have steady runs, though not all of them are refined far enough to quote at the percent level. The Tosi et al. (2015) law is implemented and tested against the paper's equations 6 to 8 (gates G119 and G124), but the benchmark cases themselves are not reproducible in this repository yet: the case-1 run drives a far too energetic initial transient, and case 5a is a limit cycle that needs a period measurement the current driver does not make. The state of both is recorded in section 2.4 of [the validation volume](../dev/VALIDATION.md) rather than reported as a number here. The blob is one 25-frame window, and the ASPECT comparison is one case, one rheology, and one direction of state import.

There is also no mesh-convergence study of the realistic solution itself. The $O(h^2)$ error of the pointwise flux rests on the measured third derivative, and the bottom-flux difference is not decomposed. Within these limits the solver is trustworthy for Rayleigh–Bénard and plume experiments in a 2-D box at the resolutions and rheologies named in [the realistic-case volume](../dev/CASE_REALISTIC.md). Outside them, run a resolution check and a benchmark check before quoting a number. Convergence is necessary, and it does not by itself validate an answer.

## What is no longer cross-checked

Until this revision the repository also carried the uniform structured-mesh solver it was originally written around, and several checks compared this solver against it: the energy operator against the uniform assembly (G34), the high-Rayleigh plume against the structured run (G64), and the `Ra = 1e8` windows against it (G65 to G67). That solver is gone, and those comparisons with it. What remains is manufactured-solution convergence (G151, G154), the p-convergence study, the Blankenbach drivers, the blob regression, and the ASPECT comparison. The uniform-mesh solver still exists in the development repository, so those comparisons can be rebuilt if a future change needs them.
