<p align="right"><a href="../en/validation.md">English</a> · <a href="../zh/validation.md">中文</a></p>

# Validation: can I trust this solver?

This page is the short answer. The full record, with the conditions and the run behind every
number, is [../dev/VALIDATION.md](../dev/VALIDATION.md); the realistic case is
[../dev/CASE_REALISTIC.md](../dev/CASE_REALISTIC.md). The solver has been checked at three
levels: analytic and constitutive checks that the operator is the operator it claims to be,
published benchmark solutions that the whole pipeline lands on the published numbers, and a
code-to-code comparison with ASPECT that sizes the discretisation uncertainty of a
time-dependent solution. The first two say the implementation is right. The third says how far
two defensible discretisations can part on the same physics.

## Benchmarks

| case | quantity | reference | computed | difference |
|---|---|---|---|---|
| Blankenbach 2b, level 3 (40 x 16) | `Nu`, energy family | 6.9299 | 6.9874 | +0.83% |
| Blankenbach 2b, level 3 (40 x 16) | `Vrms` | 171.755 | 171.18 | -0.33% |
| Blankenbach 2b, wall-refined L4 | `Nu`, wall gradient | 6.9299 | 6.9480 | +0.26% |
| Blankenbach 2b, wall-refined L4 | `Nu`, steady CBF | 6.9299 | 6.9325 | +0.04% |
| Blankenbach 2b, wall-refined L4 | `Vrms` | 171.755 | 172.010 | +0.15% |
| Blankenbach 1a | `Nu` / `Vrms` | 4.884409 / 42.864947 | not run (10-step snapshot only) | n/a |
| Blob, `Ra = 1e8`, frame 25 | `Tmax` | 1.2531 (ASPECT) | 1.2644 | +0.9% |
| Realistic, ASPECT state imported | `Nu_top` | 5.990543 | 5.990543 | 0.000% |
| Realistic, ASPECT state imported | `Nu_bot` | -3.075076 | -3.074986 | +0.003% |

The 2b rows are the pre-`266a656` record, kept for comparison; the blob and same-state rows are
current measurements. The 2b rows need their families: at level 3 the three Nusselt definitions
straddle the reference by +4.5 and -5.3 percent, which is the coarse wall element talking. On
the wall-refined level-4 state the wall, CBF and energy families span 0.22 percent (6.9480,
6.9325, 6.9330), and that is the number to compare. The same-state rows are the operator-level
check: ASPECT's field and ours give the same wall fluxes to 0.003 percent, so the difference
between the codes is accumulated history; the discrete operator agrees. The blob is a 25-frame
transient; the 2.87 percent field difference of the classic arm against ASPECT mixes `p = 4`
with Q2, two wall resolutions and two adaptive meshes, so the 0.9 percent last-frame `Tmax` is
the clean scalar. The commands that regenerate the 2b and blob records are in
[examples.md](examples.md).

The steady 2b rows are historical: they came from a stage-continuation driver that was removed
together with the uniform-mesh solver, and no script in this repository reproduces them today.
`blankenbach_2b.jl` runs the same case in time-dependent mode instead; at level 3 for
`tstop = 0.1` it ends at `Nu_top = 7.9039` and `Vrms = 143.64`, still in the startup transient.
What the suite pins is `t_g150_regression_snapshot`: a 20-step run from ASPECT's initial
condition at `1e-6` relative, `Nu_top = 7.0299964227238032`. Read the level-4 row when a
percent-level steady `Nu` matters.

## The cross-code comparison with ASPECT

The comparison runs ASPECT 2.5.0-pre (commit `340c0d018`) on the same 2.5 x 1 case, with the
same composite rheology, the same initial condition and no tuned parameter;
[../dev/VALIDATION.md](../dev/VALIDATION.md) sections 5 and 6. The top flux, which is the box's
total heat loss, agrees to **0.005 to 0.9 percent** over `t_nd = 1e-3 .. 5e-3`, with a
cross-correlation of at least 0.994 at every lag. The bottom flux does not agree: the difference
is **-6.11 percent** (CBF family) or **-8.44 percent** (gradient family) over
`t_nd = 1e-6 .. 1e-3`, crosses zero at `1.42e-3`, and is **+6.05 / +5.83 percent** over
`1e-3 .. 5e-3`, reaching +8.5 percent late. The sign depends on the window, so a bottom-flux
number without its window is meaningless.

The same-state test separates the operator from the history. ASPECT's state at `t_nd = 9.586e-4`
was imported into the QuaST tree and both codes advanced from it: `Nu_bot` agrees to
**0.003 percent** and `Nu_top` to **0.000 percent**. The operators agree on a shared field. The 6 to 9 percent is
accumulated bottom-boundary-layer history. The provenance study prices five candidate
mechanisms: definition 0.7 percent, wall resolution at most about 1 point, artificial viscosity
7 to 15 percent, realisation 2 to 4 percent with the wrong sign, and phase excluded; 85 to 90
percent remains attributed to the two solutions' bottom-layer structure, with the mechanism
open.

## Two Nusselt conventions

Three discrete families measure the wall flux. Write $\theta$ for the temperature anomaly,
$H$ for the internal heating, $h$ the wall element height and $\langle\cdot\rangle$ the
horizontal average.

Family **A**, the one runs report as `Nu_top` and `Nu_bot`, is the one-sided GLL derivative:

$$
\mathrm{Nu}^{A}_{\text{top}} = \left(1 + \frac{H}{2}\right) - \left\langle \frac{\partial\theta}{\partial y} \right\rangle_{\text{top}}, \qquad \mathrm{Nu}^{A}_{\text{bot}} = \left(1 - \frac{H}{2}\right) - \left\langle \frac{\partial\theta}{\partial y} \right\rangle_{\text{bot}}.
$$

Family **B** is the steady consistent boundary flux (`semquad_nu_cbf`), the residual of the
steady operator summed over the wall degrees of freedom; family **C** is the same sum for the
BDF2 step operator, which is what ASPECT's `heat flux statistics` writes. B and C agree to
$10^{-4}$; A differs from them by $-\tfrac{h}{3}\,\theta''_{\text{wall}}$.

Quote **A**. It is the exact derivative of the discrete solution, its wall quadrature is exact,
its truncation error is second order and measures 3e-5 to 9e-5 relative at the late states, and
it is invariant under wall refinement at a fixed state. B is first order in the wall element and
is only a cross-check: at `h = 1/16` at the bottom wall it returns +1.34 where A returns -4.47, a
433 percent error relative to the B value, and at `h = 1/64` the error is 0.0095 percent. At the
production states the A-to-C spread is 0.27 to 1.19 percent at the top and 0.005 to 0.046 percent
at the bottom.
Report `Nu_top` and `Nu_bot` with `h_wall` and `delta_T`, require at least two wall cells across
the layer and preferably four, and refine the wall strip when the gate fails. Use C for the
energy budget, where it closes the discrete identity to the solver floor.

## The automated suite

`test/runtests.jl` includes 156 gate files: analytic identities, constitutive cards,
manufactured-solution convergence, the 2b driver, the blob regression, checkpoint round trips,
the AMR rules and the solver paths. Run the whole suite with

```bash
julia --project=. --threads=4 test/runtests.jl
```

It takes about 22 minutes on eight threads. The gates pin their reference states to a stated
relative tolerance; the 2b snapshot gate uses `1e-6` and its symmetry identity `1e-10`, so a
change that moves a number is caught before a benchmark is re-run.

## Limits

The validation covers a **two-dimensional Cartesian box** with free-slip or rigid walls and the
Boussinesq mass form used by the benchmark and cross-code runs. The shipped physics is covered
where it is exercised: composite isostress rheology, phase transitions with per-transition
latent heat, internal heating, depth-dependent yielding and strain weakening. The TALA/ALA
reference state has its own self-consistency check. The validation
does not cover three dimensions, a spherical shell, toroidal motion, plate boundaries,
viscoelasticity, compressible MHD or a free surface. Nothing in the performance record is
measured in 3-D.

The benchmark set is also finite. Blankenbach 2b is done; 1a has only a 10-step transient, and
1b, 1c and 2a are open. Tosi Cases 1 and 5a are checked in
[../dev/PHYSICS.md](../dev/PHYSICS.md) section 5.2; the rest of the Tosi matrix is not. The blob
is one 25-frame window. The ASPECT comparison is one case, one rheology and one direction of
state import. There is no mesh-convergence study of the solution itself: the `O(h^2)` error of
the pointwise flux rests on the measured third derivative, and the bottom flux difference is not
decomposed. Within those limits, the solver is trustworthy for Rayleigh-Benard and plume
experiments in a 2-D box at the resolutions and rheologies named in
[../dev/CASE_REALISTIC.md](../dev/CASE_REALISTIC.md). Outside them, run a resolution check and a
benchmark check before quoting a number: convergence is necessary, and it does not by itself
validate a number.

## What is no longer cross-checked

Until this revision the repository also carried the uniform structured-mesh solver it was
originally written around, and several checks compared the quadtree solver against it: the
energy operator against the uniform assembly (G34), the high-Rayleigh plume against the
structured run (G64), and the `Ra = 1e8` windows against it (G65-G67). That solver is gone, and
those comparisons with it. What remains is the manufactured-solution convergence (G151, G154),
the p-convergence study, the Blankenbach 2b driver, the blob regression, and the ASPECT
comparison. The uniform mesh solver stays in the development repository, so the comparisons can
be rebuilt if a future change needs them.
