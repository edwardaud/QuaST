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

## Steady benchmarks

The Blankenbach et al. (1989) family. Each row is one run driven to its steady
state on the quadtree solver; `Nu` is the top wall flux from the one-sided GLL
gradient, `Vrms` the volume RMS velocity, and the last column is the difference
from the published value.

| case | `Ra` | heating | viscosity | `Nu` | ref. | `Vrms` | ref. | status |
|---|---:|---:|---|---:|---:|---:|---:|---|
| 1a | $10^4$ | 0 | constant | — | 4.884409 | — | 42.864947 | running: the flow has not settled by $t = 0.2$ |
| 1b | $10^4$ | 10 | constant | | 10.534095 | | 193.21454 | running |
| 1c | $10^4$ | 0 | $\Delta\eta\_T = 10^3$ | | 21.972465 | | 833.98977 | queued |
| 2a | $10^5$ | 10 | constant | | 10.066 | | 480.4334 | running |
| 2b | $10^4$ | 0 | joint $T$ and $z$ | 6.9428 | 6.9299 | 170.962 | 171.755 | done |

The 2b row is the level-3 record. Its three flux families disagree by more than
the family spread on a coarse wall element, so the table below separates them:

| 2b state | `Nu` family | value | ref. | difference |
|---|---|---:|---:|---:|
| level 3, 40 x 16 | wall gradient | 6.9428 | 6.9299 | +0.19% |
| level 3, 40 x 16 | energy, historical | 6.9874 | 6.9299 | +0.83% |
| wall-refined L4 | wall gradient | 6.9480 | 6.9299 | +0.26% |
| wall-refined L4 | steady CBF | 6.9325 | 6.9299 | +0.04% |
| wall-refined L4 | energy | 6.9330 | 6.9299 | +0.05% |

The three families span 0.22 percent on the wall-refined state, which is what a
percent-level `Nu` needs. On the coarse level-3 state the wall family is the
least converged, by 3.6 percent, which is why the 2b row above quotes it and the
history stays in [../dev/VALIDATION.md](../dev/VALIDATION.md) section 2.1.

## The Tosi family

Tosi et al. (2015) extends the Blankenbach cases with a viscoplastic rheology.
The reference values are the median of the eleven participating codes, Table 2 of
the paper. Case 1 is steady; case 5a is a limit cycle.

| case | quantity | reference | our value |
|---|---|---:|---|
| 1 | `Nu_top` | 3.4298 | running |
| 1 | `Nu_bot` | 3.3143 | running |
| 1 | mean `T` | 0.7767 | running |
| 5a | `Nu_top` range | 2.6950 to 7.2792 | not reproducible yet |
| 5a | mean `T` range | 0.6543 to 0.6722 | not reproducible yet |

Case 5a is periodic, not steady: comparing it needs a period and a min/max pair,
which the current benchmark driver does not measure. The state is recorded in
[../dev/VALIDATION.md](../dev/VALIDATION.md) section 2.4 rather than reported as a
number here.

## Cross-code comparisons

| case | quantity | ours | other code | difference |
|---|---|---:|---:|---:|
| blob, `Ra = 1e8`, frame 25 | `Tmax` | 1.2644 | 1.2531 (ASPECT) | +0.9% |
| realistic, ASPECT state imported | `Nu_top` | 5.990543 | 5.990543 (ASPECT) | 0.000% |
| realistic, ASPECT state imported | `Nu_bot` | -3.074986 | -3.075076 (ASPECT) | +0.003% |

The same-state rows are the sharp test: both codes start from the same imported
temperature field and evaluate the same wall functional, so a difference there is
a discretisation difference and nothing else. The blob is a 25-frame transient,
so its `Tmax` also carries the accumulated trajectory.

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
$10^{-4}$; A differs from them by $-\tfrac{h}{3}\,\theta''\_{\text{wall}}$.

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
