<p align="right"><a href="../en/faq.md">English</a> · <a href="../zh/faq.md">中文</a></p>

# FAQ

## How do I change the Rayleigh number and the heating?

In a library run, pass `Params(Ra=..., H=...)`. `Ra` is the Rayleigh number and `H` is the volumetric internal heating. The conductive base state is

$$
\bar T(y) = (1-y) + \frac H2\,y(1-y),
$$

so a conductive box has `Nu_top = 1 + H/2` and `Nu_bot = 1 - H/2`, which come to 6 and -4 at `H = 10`. In the realistic case, edit the literals `RA` and `HHEAT` in `examples/case_realistic_2d.jl`; neither has an environment variable.

## Why is the bottom Nusselt number negative?

The sign comes from the base-state gradient. With `Nu_bot = (1 - H/2) - <dtheta/dy>_bot` and `H = 10`, the conductive value is `1 - 5 = -4`. That much internal heating makes the base state carry heat downward at the bottom, where `dTbar/dy = +4`, so a negative bottom Nusselt number is the normal state of this case; the 20 000-step endpoint is `Nu_bot = -2.7636`. Always quote it with its time window, because the ASPECT bottom-flux difference changes sign at `t_nd = 1.42e-3` (see [validation](validation.md)).

## How do I add a phase transition?

Construct `PhaseTransition(z0, width, slope, Tref, drho, B)`, collect the ones you want in a vector, and pass it as `phases=ph` to `run_semquad_simulation`. `z0` is height, with 0 at the bottom and 1 at the top; `slope` is $\mathrm dz/\mathrm dT$ ; and

$$
B = \frac{\Delta\rho}{\rho_0\alpha_0\Delta T},
$$

where $\Delta\rho$ is the density jump, $\rho\_0$ the reference density, $\alpha\_0$ the thermal expansivity, and $\Delta T$ the temperature drop. For latent heat, set `latent_mode=:per_phase` and fill the coefficients with `per_phase_latent(pr, ph; rho_ref=..., cp=..., dT=..., g=..., d=..., T_surface=...)`. The realistic case already has a matched 410/660 pair in `phases()`, ready to copy.

## How do I change the rheology law?

Pick `eta_law` on `Params`:

| `eta_law` | viscosity | parameters |
|---|---|---|
| `:expT` | $\eta = \exp[-\ln(\Delta\eta\_T)\,T]$ | `deta_T` is the viscosity ratio |
| `:expTz` | $\eta = \exp(-bT + c(1-y))$ | `eta_b` = b, `eta_c` = c |
| `:arrhenius` | $\eta = \exp\{E[1/(T+T\_0) - 1/(1+T\_0)]\}$ | `eta_E` = E, `eta_T0` = T_0 |
| `:composite` | diffusion plus dislocation, isostress or isostrain | `comp_Ad`, `comp_As`, `comp_Ed`, `comp_Es`, `comp_Vd`, `comp_Vs`, `comp_n`, `comp_rule`, `comp_eps_min`, `eta_T0` |
| `:tosi` | the Tosi viscoplastic law in absolute units | `tosi=TosiLaw(...)` |

There is no keyword for anything else. `Params.eta_custom` is stored but no solver path reads it, so call `solve_semquad_stokes_var(ss, qt, eta_fun, fx, fz, uDx, uDz; ...)` directly with your law as `eta_fun(x, y)`. Note that `Params` accepts `eta_law=:const` while `run_semquad_simulation` rejects it; use `:expT` with `deta_T=1` for constant viscosity. The 2b law is $\eta = \exp(-9.70406\,T + 4.15888\,(1-y))$ , and the realistic composite card is `make_params()` in `examples/case_realistic_2d.jl`.

## Does it run in 3-D?

No. The mesh is a two-dimensional quadtree, the elements are two-dimensional GLL tensor elements, and the Stokes solver is built for two velocity components and one pressure. The repository has no 3-D path and no 3-D measurement. Section 8 of the performance volume says "nothing measured" in its 3-D column, and the thresholds do not transfer anyway: a `p = 2` hexahedral leaf has 27 nodes against 9 in two dimensions.

## How many threads should I use?

Run `julia --project=. --threads=8` and leave `linear=:auto` (the default) or `linear=:mg` alone. The matrix-free multigrid is the threaded path, and `:auto` sets `mg_threads = min(8, Threads.nthreads())` at the production size. At eight threads the warm production Stokes solve takes 1.36 s, against 3.37 s for the direct solve. The full solve gains 1.30 to 1.42x over a single thread, and the V-cycle gains 1.77 to 2.05x. The sparse direct path is serial UMFPACK, so extra threads do nothing for it; pin `linear=:direct` when you need the bitwise path.

## Why does the Stokes solve dominate the wall time?

Every step solves a variable-viscosity saddle point, while the energy equation is a scalar advection–diffusion system. On the production case at eight threads a warm step takes 2.72 s: 2.21 s (81%) in the Stokes solve, 0.31 s for the energy assembly and solve, 0.14 s for the work integral, and 0.04 s for diagnostics. To cut the Stokes cost, give the multigrid a larger leaf cap, loosen `mg_tol` within the driver's `stokes_tol`, or reduce `max_leaves`. Tightening the energy tolerances will not move the total.

## What do I do when the mesh stops refining?

Read `frames/frame_meta.csv`. Compare the `leaves` column against `CASE_MAXLEAVES` (default 10000) and the `nlev` column against `CASE_MAXLEV` (default 7), and you will see which cap you hit. If `leaves` is at its cap, raise `CASE_MAXLEAVES` (about 1.5 GB at 10 000 leaves), or lower `CASE_COARSEN` so the budget exchange frees fewer cells. If `nlev` is at 7, raise `CASE_MAXLEV`; each level halves the leaf size. If neither is at its cap, look at the wall strip set by `CASE_WALLMIN=5` and `CASE_WALLSTRIP=0.1`. A `freeze` run has `amr_every=0` and does not refine by design.

## What is `closure` in `history.csv`?

`closure` is `Nu_top - Nu_bot - (H - Ibase)` plus the latent term's time derivative over the area, with `Ibase = int u_y dTbar/dy`. It measures how far the wall fluxes sit from the steady balance, which means that at a transient state it is the enthalpy storage rate plus booking terms. No Nusselt definition closes it while the state is evolving, so a value of order `H` early in a run is normal; read it as an enthalpy rate. Setting `CASE_EXTDIAG=1` adds `resid_A`, the exact residual of the quoted pointwise pair against the scheme's own identity.

## How do I restart?

Use a checkpoint. The case writes one every 200 steps and one at exit. `resume` continues with adaptation on; `freeze` rewrites only `amr_every` to 0 and continues on the frozen mesh:

```bash
CASE_OUT=out_julia/demo CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

`CASE_STEPS` is an absolute step number, so it has to exceed the checkpoint's step, or the driver runs nothing and writes nothing. Repeat every stored setting the original run used (`CASE_SYMMETRY`, `CASE_AMR_GATE`, `CASE_WALLMIN`, `CASE_LINEAR`). On a mismatch the restart guard aborts and prints both values.

## How do I compare against ASPECT?

Use `aspect_runs/realistic/`, run ASPECT 2.5.0-pre with the `realistic_composite` plugin, and convert with the scale set and rheology mapping in section 5.1 of the validation volume. Compare `Nu_top` first: it agrees to 0.005 to 0.9 percent over `t_nd = 1e-3 .. 5e-3`. Quote `Nu_bot` with its window, because the difference is -6.11 percent over `1e-6 .. 1e-3` and +6.05 percent over `1e-3 .. 5e-3`. For a state-level check, import an ASPECT checkpoint with `examples/import_aspect_state.jl`; the two operators then agree to 0.003 percent at the bottom wall.

## How do I cite it?

[`CITATION.cff`](../../CITATION.cff) carries the release metadata: repository URL, version 0.1.0, MIT license. There is no DOI. Cite the repository URL you cloned together with the commit, plus the benchmark paper for whichever case you used — Blankenbach et al. (1989) for 2b, Tosi et al. (2015) for cases 1 and 5a, the ASPECT paper for the cross-code comparison. Whenever you quote a number, state the commit, the Julia version and the thread count. The bitwise pins depend on those three; the benchmark tolerances do not.
