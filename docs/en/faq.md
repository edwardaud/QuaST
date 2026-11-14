<p align="right"><a href="../en/faq.md">English</a> · <a href="../zh/faq.md">中文</a></p>

# FAQ

## How do I change the Rayleigh number and the heating?

For a library run, set `Params(Ra=..., H=...)`: `Ra` is the Rayleigh number and `H` is the
volumetric internal heating. The conductive base state is
$\bar T(y) = (1-y) + \frac{H}{2}y(1-y)$, so a conductive box has `Nu_top = 1 + H/2` and
`Nu_bot = 1 - H/2`, which is 6 and -4 at `H = 10`. For the realistic case, edit the literals
`RA` and `HHEAT` at `examples/case_realistic_2d.jl:146-147`; there is no environment variable
for either.

## Why is the bottom Nusselt number negative?

The sign is the base-state gradient: `Nu_bot = (1 - H/2) - <dtheta/dy>_bot`, and at `H = 10` the
conductive value is `1 - 5 = -4`. With that much internal heating the conductive base state
carries heat downward at the bottom (`dTbar/dy = +4` there), so a negative bottom Nusselt number
is the normal state; the 20 000-step E endpoint is `Nu_bot = -2.7636`. Report it with its time
window, because the ASPECT bottom-flux difference changes sign at `t_nd = 1.42e-3`
([validation.md](validation.md)).

## How do I add a phase transition?

Build a `PhaseTransition(z0, width, slope, Tref, drho, B)`, put it in a vector and pass the
vector as `phases=ph` to `run_semquad_simulation`; `z0` is height (0 bottom, 1 top), `slope` is
`dz/dT`, and `B = drho/(rho0 alpha0 dT)`. For latent heat, set `latent_mode=:per_phase` and fill
the coefficients with
`per_phase_latent(pr, ph; rho_ref=..., cp=..., dT=..., g=..., d=..., T_surface=...)`; the
ready-made pair is `phases()` at `examples/case_realistic_2d.jl:302-310`.

## How do I change the rheology law?

Pick `eta_law` on `Params`:

| `eta_law` | viscosity | parameters |
|---|---|---|
| `:expT` | $\eta = \exp[-\ln(\Delta\eta_T)\,T]$ | `deta_T` is the viscosity ratio |
| `:expTz` | $\eta = \exp(-bT + c(1-y))$ | `eta_b` = b, `eta_c` = c |
| `:arrhenius` | $\eta = \exp\{E[1/(T+T_0) - 1/(1+T_0)]\}$ | `eta_E` = E, `eta_T0` = T_0 |
| `:composite` | diffusion plus dislocation, isostress or isostrain | `comp_Ad`, `comp_As`, `comp_Ed`, `comp_Es`, `comp_Vd`, `comp_Vs`, `comp_n`, `comp_rule`, `comp_eps_min`, `eta_T0` |
| `:tosi` | Tosi viscoplastic law in absolute units | `tosi=TosiLaw(...)` |

For a law outside these five there is no keyword: `Params.eta_custom` is stored but no solver
path reads it, so call `solve_semquad_stokes_var(ss, qt, eta_fun, fx, fz, uDx, uDz; ...)`
directly with your viscosity as `eta_fun(x, y)`. `run_semquad_simulation` rejects
`eta_law=:const`, which `Params` accepts, so use `:expT` with `deta_T=1` for constant viscosity;
the 2b law is `eta = exp(-9.70406 T + 4.15888 (1-y))` and the realistic composite card is
`make_params()` at `examples/case_realistic_2d.jl:312-329`.

## Does it run in 3-D?

No. The mesh is a 2-D quadtree, the elements are 2-D GLL tensor elements, and the Stokes solver
is built for two velocity components and one pressure; the repository has no 3-D path and no 3-D
measurement. The 3-D column of [../dev/PERFORMANCE.md](../dev/PERFORMANCE.md) section 8 is
explicitly "nothing measured", and a `p = 2` hexahedral leaf has 27 nodes against 9, so the
measured thresholds do not transfer.

## How many threads should I use?

Use `julia --project=. --threads=8` and leave `linear=:auto` (the default) or `linear=:mg`
alone; the matrix-free multigrid is the threaded path, and `:auto` sets
`mg_threads = min(8, Threads.nthreads())` at the production size. The warm production Stokes
solve is 1.36 s at eight threads against 3.37 s for the direct solve, the full solve gains 1.30
to 1.42x over one thread, and the V-cycle gains 1.77 to 2.05x. The sparse direct path is serial
UMFPACK, so adding threads does not speed it up; pin `linear=:direct` when you need the bitwise
path.

## Why is the wall time dominated by the Stokes solve?

Every step solves a variable-viscosity saddle point, while the energy equation is a scalar
advection-diffusion system. On the production case at eight threads the warm step is 2.72 s, of
which 2.21 s (81 percent) is the Stokes solve, 0.31 s the energy assembly and solve, 0.14 s the
work integral and 0.04 s the diagnostics. Reduce the Stokes cost with a larger multigrid leaf
budget, a looser `mg_tol` inside the driver's `stokes_tol`, or a smaller `max_leaves`; changing
the energy tolerances will not move the total much.

## What do I do when the mesh stops refining?

Read `frames/frame_meta.csv`: the `leaves` column against `CASE_MAXLEAVES` (default 10000) and
the `nlev` column against `CASE_MAXLEV` (default 7) tell you which cap was hit. If `leaves` is
at the cap, raise `CASE_MAXLEAVES` (about 1.5 GB at 10 000 leaves) or lower `CASE_COARSEN` so
the budget exchange frees fewer cells; if `nlev` is at 7, raise `CASE_MAXLEV`, which halves the
leaf size per level. If neither is at its cap, check `CASE_WALLMIN=5` and `CASE_WALLSTRIP=0.1`
for the wall strip; a `freeze` run has `amr_every=0` and does not refine by design.

## What is `closure` in `history.csv`?

`closure = Nu_top - Nu_bot - (H - Ibase)` plus the latent time derivative over the area, where
`Ibase = int u_y dTbar/dy`; it is the departure of the wall fluxes from the steady balance, so
it is the enthalpy storage rate plus bookkeeping terms at a transient state. It does not close
for any Nusselt definition at a transient state, and a value of order `H` early in a run is
normal, so read it as an enthalpy rate. The opt-in `CASE_EXTDIAG=1` adds `resid_A`, the exact
residual of the quoted pointwise pair against the scheme's own identity.

## How do I restart?

Use a checkpoint; the case writes one every 200 steps and at exit. `resume` continues it with
AMR, while `freeze` rewrites only `amr_every` to 0 and continues on the frozen mesh:

```bash
CASE_OUT=out_julia/demo CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

`CASE_STEPS` is an absolute step number, so it must exceed the checkpoint's step or the driver
runs zero steps and writes nothing; repeat every stored setting the run used (`CASE_SYMMETRY`,
`CASE_AMR_GATE`, `CASE_WALLMIN`, `CASE_LINEAR`), and the restart guard aborts on a mismatch and
prints both values.

## How do I compare against ASPECT?

Use `aspect_runs/realistic/`, run ASPECT 2.5.0-pre with the `realistic_composite` plugin, and
convert with the scale set and rheology mapping in [../dev/VALIDATION.md](../dev/VALIDATION.md)
section 5.1. Compare `Nu_top` first: it agrees to 0.005 to 0.9 percent over
`t_nd = 1e-3 .. 5e-3`. Quote `Nu_bot` with its window, because the difference is -6.11 percent
over `1e-6 .. 1e-3` and +6.05 percent over `1e-3 .. 5e-3`; for a state-level check import an
ASPECT checkpoint with `examples/import_aspect_state.jl`, where the two operators agree to 0.003
percent at the bottom wall.

## How do I cite it?

[`CITATION.cff`](../../CITATION.cff) carries the release metadata: the repository URL, version
0.1.0 and the MIT licence, and there is no DOI. Cite the repository URL you cloned and the code
commit, plus the benchmark paper for the case you used: Blankenbach et al. (1989) for 2b, Tosi
et al. (2015) for Cases 1 and 5a, and the ASPECT paper for the cross-code comparison. State the
commit, the Julia version and the thread count with any number you quote; the bitwise pins are
tied to those, while the benchmark tolerances hold across builds.
