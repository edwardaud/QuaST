<p align="right"><a href="../en/faq.md">English</a> · <a href="../zh/faq.md">中文</a></p>

# FAQ

## How do I change the Rayleigh number and the heating?

In a library run, pass `Params(Ra=..., H=...)`; the base state that `H` builds is in [guide section 1.3](guide.md#13-base-state-heating-and-nusselt-numbers). In the realistic case, set `CASE_RA` in the environment. `H` is the literal `HHEAT` in `make_params()` of `examples/case_realistic_2d.jl` and has no environment variable. [Guide section 6.1](guide.md#61-environment-variables) lists the case variables.


## Why is the bottom Nusselt number negative?

The sign comes from the conductive base state. For `H = 10` the base state carries heat downward at the bottom, where its gradient is positive, so the conductive bottom Nusselt number is negative. A negative bottom Nusselt number is the normal state of this case. Always quote it with its time window: the ASPECT bottom-flux difference changes sign during the early transient, and [validation](validation.md) gives the windows. The definition is in [guide section 1.3](guide.md#13-base-state-heating-and-nusselt-numbers).


## How do I add a phase transition?

Construct `PhaseTransition(z0, width, slope, Tref, drho, B)`, collect the ones you want in a vector, and pass it as `phases=ph` to `run_semquad_simulation`. `z0` is height, with 0 at the bottom and 1 at the top; `slope` is $\mathrm dz/\mathrm dT$ ; and `B` is the buoyancy number defined in [guide section 1.5](guide.md#15-the-two-phase-transitions). For latent heat, set `latent_mode=:per_phase` and fill the coefficients with `per_phase_latent(pr, ph; rho_ref=..., cp=..., dT=..., g=..., d=..., T_surface=...)`. The realistic case already has a matched 410/660 pair in `phases()`, ready to copy; [guide section 1.6](guide.md#16-latent-heat) derives the coefficients.


## How do I change the rheology law?

Pick `eta_law` on `Params`. The laws, their parameters and the constant-viscosity limit are tabulated in [guide section 1.4](guide.md#14-rheology). There is no keyword for anything else. `Params.eta_custom` is stored but no solver path reads it, so call `solve_semquad_stokes_var(ss, qt, eta_fun, fx, fz, uDx, uDz; ...)` directly with your law as `eta_fun(x, y)`. The realistic composite card is `make_params()` in `examples/case_realistic_2d.jl`.


## What is `eta_ref` and when should I change it?

`eta_ref` is `Params.eta_ref`, the reference viscosity that the Rayleigh number is defined with. The flow solve uses `eta_law/eta_ref` wherever the law's viscosity enters. Change it when you rescale the law itself: multiplying `comp_Ad` and `comp_As` by `1/c` multiplies every viscosity by `c`, so the same physical problem needs `eta_ref = c`; `eta_min` and `eta_max` rescale with the law too, because the bounds are absolute in the law's units. Leave it at its default for the realistic card: the calibration defines the card's `Ra` with `eta_ref = 1`. A non-unit value makes the driver report the convention at the first flow solve and at the end of the run, and `pic_mode = :newton` refuses it. [Guide section 1.9](guide.md#19-the-reference-viscosity-and-the-rayleigh-number) gives the convention, the report and the developed-state numbers.


## Why is the reported `Ra` not the interior Rayleigh number?

`Ra` is normalized by one reference viscosity, `eta_ref`, while the material's actual viscosity varies by orders of magnitude across the box. A run with a non-unit `eta_ref` therefore also reports `eta_interior` and `Ra_interior = Ra/eta_interior`. Quote the nominal value together with `Ra_interior`; [guide section 1.9](guide.md#19-the-reference-viscosity-and-the-rayleigh-number) defines both and gives the numbers at the developed production state.


## Does it run in 3-D?

No. The mesh is a two-dimensional quadtree, the elements are two-dimensional GLL tensor elements, and the Stokes solver is built for two velocity components and one pressure. The [Scope section of validation](validation.md#scope) is the single list of what the code does not implement.


## How do I set an open outflow?

Set the wall to `:open` in `wall_bc` (or `CASE_WALL_BC` on the card), then give its normal traction with `wall_tn` and its tangential traction with `wall_t`; the normal traction sets the outflow pressure level. The configuration must keep a Dirichlet normal velocity on at least one x-wall and one y-wall, and `:schur_cg`/`:lagged` are refused with an open wall. [Guide section 2](guide.md#2-boundary-conditions) is the reference, and the channel recipe on [examples](examples.md) is a working card.


## Why does the pressure pin disappear?

The nodal pin is exact only while the normal velocity is Dirichlet on the whole boundary; an `:open` wall makes the constant pressure no longer a null mode of the reduced gradient, so the solver leaves the pressure unpinned and reports `LAST_SEMSTOKES_PRESSURE_GAUGE = :none`. The pressure level then comes from the `wall_tn` data. [Guide section 2](guide.md#2-boundary-conditions) derives both.


## How many threads should I use?

Run with `--threads=8` and leave `linear=:auto` (the default) or an explicit `linear=:mg` alone. The matrix-free multigrid is the threaded path; the direct solve is serial UMFPACK. [Guide section 9](guide.md#9-threads-and-memory) gives the scaling and the memory numbers.


## Why does the Stokes solve dominate the wall time?

Every step solves a variable-viscosity saddle point, while the energy equation is a scalar advection–diffusion system. At the 10 000-leaf cap the warm step is about 3.6 s, of which 3.1 s is the Stokes solve; [guide section 9](guide.md#9-threads-and-memory) gives the full split. To cut the Stokes cost, give the multigrid a larger leaf cap, loosen `mg_tol` within the driver's `stokes_tol`, or reduce `max_leaves`. Tightening the energy tolerances will not move the total.


## Why can a viscoelastic run be cheaper than its `shear_modulus = 0` twin?

Because `G` changes the problem rather than adding a cost: the effective viscosity is capped at `G*dt`, so the contrast the Stokes solve sees is smaller than the material contrast. [Guide section 1.8](guide.md#18-viscoelasticity) gives the closure, the two measured states and the cost of the machinery. A viscoelastic card is a different problem, so its diagnostics are not its `G = 0` twin's; record the expected effective contrast before interpreting a viscoelastic run.


## What do I do when the mesh stops refining?

Read `frames/frame_meta.csv`. Compare the `leaves` column against `CASE_MAXLEAVES` and the `nlev` column against `CASE_MAXLEV`; the defaults are in [guide section 6.1](guide.md#61-environment-variables). If `leaves` is at its cap, raise `CASE_MAXLEAVES`, or lower `CASE_COARSEN` so the budget exchange frees fewer cells. If `nlev` is at its cap, raise `CASE_MAXLEV`; each level halves the leaf size. If neither is at its cap, look at the wall strip set by `CASE_WALLMIN` and `CASE_WALLSTRIP`. A `freeze` run has `amr_every=0` and does not refine by design.


## Which AMR indicator should I use?

Keep the default, `:theta`, unless your observable is better served by `:velocity` or `:both`. The field selector `CASE_INDICATOR`, the rule that combines the two fields and the measured comparison of `:theta` against `:velocity` are in [guide section 4.1](guide.md#41-indicator).


## What is `closure` in `history.csv`?

`closure` is not a flux error. It is the steady flux balance evaluated at the current state, plus the latent term's time derivative, so at a transient state it is the enthalpy storage rate plus booking terms. [Guide section 7.2](guide.md#72-historycsv) gives the definition and the columns. Setting `CASE_EXTDIAG=1` adds `resid_A`, the exact residual of the quoted pointwise pair against the scheme's own identity.


## How do I restart?

Use a checkpoint. The commands are in the stopping and resuming section of the [quick start](quickstart.md), and the realistic-card section of [examples](examples.md) shows the frozen continuation; the modes are in [guide section 6.1](guide.md#61-environment-variables) and the configuration guard is in [guide section 8](guide.md#8-restarting-and-the-configuration-guard). `CASE_STEPS` is an absolute step number, so it has to exceed the checkpoint's step, and the guard refuses a run whose compared settings differ from the stored ones.


## How do I compare against ASPECT?

Use `aspect_runs/realistic/`, run ASPECT 2.5.0-pre with the `realistic_composite` plugin, and convert with `examples/import_aspect_state.jl`. [Validation](validation.md) gives the setup, the windows and the measured differences; compare `Nu_top` first, because the box's total heat loss is the sharpest shared quantity.


## Why is the regional case not comparable with published models?

It is a kinematic `Ra = 0` model with no dipping slab interface, no free surface, no compositional field and no self-consistent slab pull, so its numbers are not directly comparable with published kinematic-subduction models. The limitations are stated on the [examples](examples.md) page, and the [Scope section of validation](validation.md#scope) lists the missing physics.


## How do I cite it?

[`CITATION.cff`](../../CITATION.cff) carries the release metadata: repository URL, version 0.1.0, MIT license. There is no DOI. Cite the repository URL you cloned together with the commit, plus the benchmark paper for whichever case you used: Blankenbach et al. (1989) for 2b, Tosi et al. (2015) for cases 1 and 5a, the ASPECT paper for the cross-code comparison. Whenever you quote a number, state the commit, the Julia version and the thread count. The bitwise pins depend on those three; the benchmark tolerances do not.

**Next:** [Guide](guide.md) — the full reference.
