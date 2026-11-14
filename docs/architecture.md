# Architecture

`src/` is one Julia module. `QuaST.jl` (188 lines) fixes the include order and
the exports; eight class folders hold the rest, 59 Julia files and 19 806 lines.
The tree is organised by class of code: the grid knows nothing about the
rheology, and the rheology nothing about the solver. `test/`, `examples/`,
`benchmarks/` and `scripts/` sit outside the module.

## Where things live

| Folder | Files | Lines | What belongs there |
|---|---:|---:|---|
| `core/` | 9 | 2 217 | GLL and quadrature tables, the 2:1 quadtree, the scalar `SemQuad` space, hanging constraints, point evaluation, transfer |
| `physics/` | 11 | 3 112 | `Params`, the viscosity laws, phase transitions, latent heat, the reference state, the energy assembly and its step |
| `stokes/` | 11 | 4 317 | `SemStokes`, the saddle assembly, the matrix-free velocity operator, the solve and its `:auto` selector |
| `mg/` | 9 | 3 439 | The multigrid hierarchy, the V-cycle, the pressure block, GMRES and the saddle driver |
| `amr/` | 7 | 1 665 | The gradient-jump indicator, the leaf primitives, the 2:1 coarsening gate, the balanced cycle, the driver hooks |
| `driver/` | 8 | 3 700 | `run_semquad_simulation`, the Picard and Newton loops, the steady entries, the mass closures |
| `io/` | 1 | 383 | The HDF5 checkpoint and the restart configuration guard |
| `diagnostics/` | 3 | 973 | Nusselt numbers, Vrms, the energy budget, consistent boundary flux, entropy viscosity |

Counted over the `.jl` files only, with `awk 'END{print NR}'`; `src/QuaST.jl` is the sixtieth file.

## Include order

`include` pastes a file into the module body, so the order is constrained by
definition-time references only: a struct field type, a supertype, a method
signature, or a `const` initialiser. Function bodies may call anything defined
later, which is why `src/driver/run.jl` calls `save_semquad_state` before
`src/io/checkpoint.jl` is included.

```text
physics/     tosi_law refstate params constitutive phase
core/        gll fe quadtree semquad parallel constraints poisson eval transfer
amr/         leaf_ops indicator
diagnostics/ entropy_viscosity
physics/     latent energy_assembly energy eta weakening phase_buoyancy
amr/         cooldown coarsen_gate coarsen_plan cycle
stokes/      state selector direct assembly kop apply diag solve_const assembly_var solve_var operators
mg/          hierarchy levelcache build vcycle pressure bfbt saddle s1solve gmres
amr/         hooks
driver/      picard newton step_newton tosi_steady dp_steady steady_entry run mass
io/          checkpoint
diagnostics/ global flux
```

`physics/` appears twice on purpose. The first block is the type layer, before `core/`:
`TosiLaw` and `SemRefState` are field types of `Params`
(`src/physics/params.jl:75-79`), so `tosi_law.jl` and `refstate.jl` precede
`params.jl`; `constitutive.jl` and `phase.jl` annotate `pr::Params`, so they
follow it. The second block holds the rest of the physics: the laws that take a
`SemQuad`, plus the latent and buoyancy terms. The seam shows
in the diagnostics block: the first file after `core/` that names `Params` in a
signature is `diagnostics/entropy_viscosity.jl`; `physics/energy_assembly.jl`
names its `SEMEVGlobals` in the `evg` signature, so it must come first.
`amr/hooks.jl` follows `stokes/` because `semquad_amr_passes!(ss::SemStokes,
...)` names the type, and `driver/` follows both.

## The folders

**`core/`** `gll.jl` computes the GLL nodes, their barycentric weights and the
Gauss-Legendre rule; `fe.jl` builds `FEData`, pairing a $Q\_p$ velocity basis
with a $Q\_{p-1}$ pressure basis on a $p+2$ Gauss rule. `quadtree.jl` is the
mesh: leaves are `(level, ix, iy)`, `refine!` and `coarsen!` mutate, `balance!`
keeps 2:1, and `QuadFaceTable` answers `face_pairs` and `neighbors` without a
rescan. `semquad.jl` is `SemQuad`, the scalar $Q\_p$ space. Its split slices
follow: `constraints.jl` (hanging interpolation `C`, applied as
$C^\top A C\,u = C^\top b$), `eval.jl`, `transfer.jl` (nodal or L2),
`parallel.jl`, and `poisson.jl` for verification only.

**`physics/`** `params.jl` is `Params`, every switch a run takes (`Ra`, `H`, the
viscosity law, yield, reference state, latent heat, weakening), plus
`Tbar`/`dTbar`. `constitutive.jl` evaluates the temperature laws (`eta_of_T`,
`eta_at`, `_clip_eta`) and the composite law (`eta_composite_at`); `eta.jl` adds
the plastic branch (`semquad_yield_stress`, `semquad_visco_plastic_eta`,
`semquad_eta_at`) and the Newton derivatives (`semquad_deta_dp`,
`semquad_deta_chi`, `spd_factor`); `tosi_law.jl` defines `TosiLaw`. `phase.jl`
defines `PhaseTransition` with `phase_gamma` and `per_phase_latent`;
`latent.jl` supplies the per-transition latent coefficients,
`phase_buoyancy.jl` the buoyancy anomaly. `refstate.jl` is `SemRefState` with
`semquad_rho`, `semquad_alpha`, `semquad_cp`, `semquad_Tad` and
`semquad_drho_dz`. `energy_assembly.jl` assembles `M`, `K`, `C` and `F` for
$\partial\_t\theta + u\cdot\nabla\theta = \nabla^2\theta - u\_z\bar T'(z)$;
`energy.jl` steps it; `weakening.jl` carries the weakening field.

**`stokes/`** `state.jl` is `SemStokes` (a velocity and a pressure `SemQuad`)
plus the `LAST_SEMSTOKES_*` Refs. `assembly.jl` holds the reference-element
matrices, `make_semstokes` and the pressure operators; `assembly_var.jl`
assembles the variable-viscosity operator into COO buffers; `diag.jl` builds the
Jacobi diagonal. `kop.jl` is the matrix-free operator `SemQuadKOp`
(`semquad_kop`, `semquad_apply_K!`, `semquad_apply_Kred!`) on the assembler's own
Gauss rule; `apply.jl` adds `SemQuadParRegion`; `operators.jl` assembles the
density-weighted divergence and the compressible block. `solve_var.jl` is
`solve_semquad_stokes_var`; `selector.jl` is `semquad_choose_linear` (MG from
20 000 saddle dofs, or 2 000 at contrast at most 100); `direct.jl` the
stale-factor LU and shift ladder; `solve_const.jl` the constant-viscosity solve.

**`mg/`** `hierarchy.jl` defines `SemQuadMG` and coarsens the quadtree into
levels (`_mg_coarsen_tree`); `build.jl` is the cached `make_semquad_mg`.
`levelcache.jl` keeps the levels whose leaf set did not change. `vcycle.jl` is
the V-cycle (`_semquad_mg_vcycle!`, `semquad_mg_apply`); `saddle.jl` is the
outer FGMRES (`semquad_mg_saddle_solve`) and the velocity CG. `pressure.jl`
builds the Schur preconditioner (`SemQuadPressureCache`), `bfbt.jl` its BFBT
diagonal, `s1solve.jl` the `S1` inner solves, `gmres.jl` the restarted GMRES.

**`amr/`** `indicator.jl` marks: `semquad_gradient_jump` returns one error per
leaf from the normal-derivative jump. `leaf_ops.jl` has `_refine_pair!`,
`_coarsen_pair!` and the hard budget `semquad_enforce_leaf_budget!`.
`coarsen_gate.jl` is the 2:1 `coarsening_allowed` pre-check; `coarsen_plan.jl`
is the plan and funding merge. `cycle.jl` is `amr_cycle_balanced!`, with
thresholds $\tau\_r$, $\tau\_c$ and the deadband $\tau\_h = \min(\tau\_c, h\tau\_r)$
of the `hysteresis` keyword; `cooldown.jl` is `AmrCooldown`; `hooks.jl` is
`semquad_amr_passes!` (indicator, cycle, budget, rebuild `SemStokes`, transfer).

**`driver/`** `run.jl` is `run_semquad_simulation`: the spaces, the optional
restart, a local `solve_flow!` that builds the viscosity and calls the Stokes
solve, the Picard or Newton loop, the energy step, the CFL update, the AMR
block, the history row, the callback, the checkpoint and the steady test.
`picard.jl` has `semquad_strain_rate_field` and `blend_field`; `newton.jl`,
`tosi_steady.jl` and `dp_steady.jl` are the three steady Newton maps;
`step_newton.jl` is the per-step Newton; `steady_entry.jl` is
`semquad_steady_newton`; `mass.jl` builds the mass-form closures.

**`io/`** `checkpoint.jl` is the HDF5 state. `save_semquad_state` writes the
leaves, `theta`, `theta_prev`, `ux`, `uz`, `pv`, `t`, `dt`, `dt_prev`, `step`,
`p`, the tree geometry, the EV, weakening and cooldown datasets, and `config`, a
canonical string of every trajectory-changing setting. `load_semquad_state`
rebuilds the tree, and `run_semquad_simulation` compares `config` on restart, so
a resume cannot silently change its AMR or physics setup.

**`diagnostics/`** `global.jl` is the acceptance set: `semquad_nu_walls` (the
wall Nusselt numbers, $Nu\_{top} = 1 + H/2 - \langle\partial\_y\theta\rangle$),
`semquad_nu_energy`, `semquad_heat_budget`, `semquad_vrms`, `semquad_mean_T`,
`semquad_max_divergence`, `semquad_work_dissipation`, `semquad_surface_velocity`,
`semquad_exact_residual` and `semquad_level_stats`. `flux.jl` is the
consistent-boundary-flux pair (`semquad_cbf_theta_flux`, `semquad_nu_cbf`);
`entropy_viscosity.jl` is the EV field (`SEMEVGlobals`, `semquad_ev_globals`).

## Data flow

```text
Params (pr)                          Quadtree (qt)
    |                                    |
    |                      make_semquad(qt, p) -> SemQuad        core/
    |                      make_semstokes(qt, p) -> SemStokes     stokes/
    |                                    |
    +-------------> run_semquad_simulation(qt, pr, theta0)        driver/run.jl
                         |  while step < maxsteps && t < tstop
                         +-- solve_flow!  eta_fun(x,y) from theta, u, p
                         |     +-- solve_semquad_stokes_var(ss, qt, eta_fun, fz)
                         |           :auto -> semquad_choose_linear
                         |             :direct  sparse LU + the shift ladder
                         |             :mg      matrix-free V-cycle in FGMRES
                         |     -> u, p
                         +-- solve_semquad_energy -> theta_new      physics/energy.jl
                         +-- diagnostics row (Nu, Vrms, mean T)     diagnostics/
                         +-- semquad_amr_passes! every amr_every    amr/hooks.jl
                         |     -> new SemStokes, fields transferred
                         +-- save_semquad_state every checkpoint_every    io/
```

The inputs are `Params` and `Quadtree`. The driver builds `SemStokes` at
`src/driver/run.jl:504`, and `src/amr/hooks.jl:241` rebuilds it after every AMR pass.
The loop at `src/driver/run.jl:927` runs `solve_flow!` (buoyancy
`fz = Ra * theta` plus the phase anomaly), one energy step, the diagnostics row
(`src/driver/run.jl:1275`), the AMR pass (`src/driver/run.jl:1330`), the callback
(`src/driver/run.jl:1387`) and the checkpoint (`src/driver/run.jl:1394`). The driver
writes only the HDF5 checkpoint; `examples/case_realistic_2d.jl` writes the CSV
history, `run.log` and the frames.

## Where do I change X

| Change | The handle | The code |
|---|---|---|
| The Rayleigh number | `Params(Ra=...)`, `src/physics/params.jl:116` | `fz = pr.Ra * theta` in `solve_flow!`, `src/driver/run.jl:772-781`; the same `Ra` weights the work balance at `src/driver/run.jl:842` |
| The rheology | `Params.eta_law` and its parameters, `src/physics/params.jl:51-114` | `src/physics/constitutive.jl` (`eta_of_T`, `eta_at`, `_clip_eta`), `src/physics/eta.jl` (`semquad_visco_plastic_eta`, `semquad_eta_at`), `src/physics/tosi_law.jl` |
| A phase transition | the `phases=` vector of `run_semquad_simulation`, `src/driver/run.jl:229` | `src/physics/phase.jl` (`PhaseTransition`, `phase_gamma`), `src/physics/phase_buoyancy.jl`, `src/physics/latent.jl` |
| The adaptive policy | `amr_every`, `refine_frac`, `coarsen_frac`, `max_leaves`, `wall_min_level`, `src/driver/run.jl:167-173` | `src/amr/cycle.jl:137` (`amr_cycle_balanced!`), `src/amr/cooldown.jl`, `src/amr/indicator.jl:148` |
| The linear solver | `linear=:auto` or `:direct` or `:mg`, `src/driver/run.jl:210` | `src/stokes/selector.jl:131`, the dispatch at `src/stokes/solve_var.jl:773-883`, `src/mg/build.jl:221` (`make_semquad_mg`) |
| The output | `callback`, `checkpoint_path`, `checkpoint_every`, `src/driver/run.jl:184, 241-243` | the history row at `src/driver/run.jl:1275-1309`, `src/io/checkpoint.jl:110`; the CSV, log and frames in `examples/case_realistic_2d.jl` |
