# Changelog

All notable changes to this project are recorded here, in Keep a Changelog format with
semantic versioning; in the 0.x series the public API may still change. This is the first
public release: earlier development happened in a private repository.

## [Unreleased]

### Added

- `examples/benchmark_cases.jl`: one entry point per published case (Blankenbach 1a, 1b,
  1c, 2a, 2b and Tosi 1, 5a), each at a chosen level, with the wall rule and the Rayleigh
  number as environment knobs.
- The validation pages report each benchmark as a convergence table with the level, the
  leaf count and the difference from the published value, instead of a single number.
- `scripts/run_benchmarks.sh` reproduces the whole campaign with the settings the published
  benchmark cards use, and `scripts/benchmark_table.jl` turns its logs into table rows.
- `scripts/check_docs.jl`, run in CI, catches the Markdown constructs that would leave a
  formula rendered literally.
- `Params.eta_multiplier` scales whatever the viscosity law returned, and
  `weak_zone_factor` builds the factor of a straight weak zone: the cheap
  stand-in for a plate boundary, next to the law rather than instead of it.
- `Params.side_thermal = :fixed` holds the side walls at `Twall_left` and
  `Twall_right` instead of insulating them; `:insulating` stays the default.
- `Params.H_half_life` decays the radiogenic source as `H 2^(-t/H_half_life)`, the
  form ASPECT's `Radioactive decay` subsection uses. `Inf` keeps the constant rate
  bit for bit.
- Maxwell viscoelasticity: `Params.shear_modulus` turns on the deviatoric stress history,
  the effective-viscosity closure, the lagged elastic force term and the co-rotational
  update, and the driver reports the viscosity regime it produces. `CASE_SHEAR_MODULUS`
  exposes it on the realistic card.
- `Params.eta_ref` and `CASE_ETA_REF`: the viscosity the Rayleigh number is defined with,
  separate from the viscosity the material law returns. A non-unit value makes the driver
  report the convention, the interior Rayleigh number and the physical reference viscosity.
- `amr_indicator` and `CASE_INDICATOR`: the field the AMR jump estimator is applied to can
  be the temperature (`:theta`, the default), the velocity (`:velocity`), or the two after
  normalizing each by its own total (`:both`).
- Per-wall boundary conditions: `Params.wall_bc` and `Params.wall_u` set each of the four
  walls to free slip, rigid, or a prescribed tangential velocity.
- `Params.rho_multiplier`, the density hook a low-density sticky-air layer needs.
- A physics revision marker (`src/physics/revision.jl`): a checkpoint written under a
  different revision is refused at restart, and a key-less checkpoint is resumed with a
  warning that names what the revision changed.
- `corotation_guard` and `regime_report` keywords, plus `MAXWELL_COROTATION_BOUND`, which
  report how far a run sits from the linearized rotation's validity band and what viscosity
  contrast the effective closure produces.

### Changed

- The Maxwell stress update is deviatoric and co-rotational: the strain rate is projected
  onto its deviatoric part, the old stress is rotated with `dt (W tau - tau W)` before the
  backward-Euler relaxation, and the two changes together move the stress fixed point from
  `2 eta ehat` to `2 eta ehat + tau_M (W tau - tau W)`. A `shear_modulus > 0` run started
  from a checkpoint written before this change is refused by the physics revision guard.
- The `:both` AMR indicator adds the two fields after normalizing each by its own total;
  it used to be a plain sum, which the velocity term dominated by orders of magnitude.
- The user documentation was rewritten in both languages rather than translated, and the
  internal layer labels were removed from every page, comment and gate header. The set is
  now organized by question, one owner per fact, with the same hierarchy in both languages.
- The benchmark drivers now start from ASPECT's own initial fields, which are close to the
  steady state, and take `CASE_CFL` from the cards (`5.0` for the Blankenbach cases).

### Removed

- The steady-state Newton solver: `newton.jl`, `dp_steady.jl`, `tosi_steady.jl`,
  `steady_entry.jl`, the `steady_newton` keyword and the `steady` field of the returned
  state. The solver time-steps; steady state is reached, and detected, by continuing the
  transient, which is what ASPECT and CitcomS do as well.
- The classic AMR policy. `amr_policy` (and with it `:classic` and the
  `:threshold` dispatch), `amr_cycle!` (`src/dgquad.jl`),
  `amr_cycle_hysteresis!` (`src/amr_advanced.jl`), the DG prototype that shared
  `dgquad.jl`, and the classic-only keywords `amr_strategy`, `marking`,
  `refine_thresh`, `coarsen_thresh` and `coarsen_theta_tol` are gone. The
  balanced policy (`amr_cycle_balanced!`) is the only adaptive policy, and
  `run_semquad_simulation`'s `initial_refine` now runs it as well, so a run with
  the old defaults is bit-for-bit unchanged. Checkpoints keep the
  `amr_policy = "balanced"` config key so files written before the removal still
  match the restart guard.
- Sixteen one-off launcher scripts under `scripts/`, superseded by `run_benchmarks.sh`.
- Five Blankenbach 2b example variants (`blankenbach_2b{,_time,_stretch,_wall,_cbf_ev}.jl`),
  superseded by `benchmark_cases.jl`, and `examples/tosi_case5a_cycle.jl`, which the
  ordinary driver reproduces.

### Fixed

- The velocity-based AMR indicator threw at its first adaptation cycle (it read a field the
  pass never received), so every `:velocity` and `:both` run stopped there.
- The benchmark driver passed a full temperature as the anomaly when a case had no ASCII
  field, which drove the Tosi cases into a violent startup.
- The benchmark driver's diagnostics used the mesh from before an adaptation pass, so a
  wall-refined run printed a Nusselt number unrelated to the state.
- The wall rule was silently clipped when the tree depth equalled the interior level, so a
  "wall-refined" run was a uniform mesh.
- The limit-cycle analysis called `minmax` with a splatted slice and crashed after a case
  had finished; the trajectory is now written to a CSV as well.
- The energy step's entropy viscosity lives in `physics/` with the assembly that uses it,
  and the driver's per-step diagnostics row, the run configuration and the restart guard
  each have their own file.

## [0.1.0] - 2026-11-13

### Added

- The solver: Q_p velocity and Q_{p-1} pressure on Gauss-Legendre-Lobatto nodes over a
  quadtree kept 2:1 balanced, exact hanging-node constraints, variable-step BDF2 time
  stepping, entropy viscosity, and a wall strip held at a minimum level.
- The physics stack: composite diffusion-dislocation rheology with the isostress rule,
  depth-dependent Drucker-Prager yielding with strain weakening, phase transitions with
  per-transition latent heat, internal heating, an Adams-Williamson or Murnaghan reference
  state, and the Boussinesq or anelastic mass form.
- Two linear paths behind one `linear` switch: sparse direct factorization with a diagonal
  shift ladder, and matrix-free geometric multigrid with a w-BFBT pressure preconditioner.
  `linear = :auto` selects the multigrid at 20 000 saddle dofs or more at any contrast, and
  at 2 000 or more when the sampled viscosity contrast is at most 100; otherwise it selects
  the direct path and retries a failed multigrid solve there.
- The adaptive policy `amr_policy = :balanced`: Dörfler marking, forward coarsening with
  budget exchange, a hysteresis cooldown, a 2:1 coarsening gate, and the wall rule, with
  `:classic` kept as the historical policy.
- The gate suite of about 220 gate files and 133 600 assertions, run by
  `test/runtests.jl` or `test/run_gates.jl`, and the CI workflow that runs the gates
  expected to pass.
- The validation record, covering Blankenbach 2b, the blob digest, Tosi Cases 1 and 5a and
  the ASPECT constitutive card, and the ASPECT comparison kit in `aspect_runs/realistic/`:
  the `.prm` inputs, the small C++ plugin for ASPECT 2.5, and the analysis scripts.
- The user documentation: README, quickstart, guide, examples, validation, FAQ, and the
  engineering record in `docs/dev/`.

### Changed

- The repository is a standard Julia package at the root, so every command uses
  `--project=.`. The package is named `QuaST` (was `MantleSEM` under `julia/`); its UUID is
  unchanged.
- The three monolithic solver files were split into `src/stokes/`, `src/mg/` and
  `src/driver/`. Every chunk is a byte-contiguous slice of the original file, and
  concatenating the chunks in include order reproduces it exactly.
- The Nusselt convention is documented as the code returns it:
  `Nu_top = (1 + H/2) - <dtheta/dy>_top`, which is 6 at the top and -4 at the bottom at the
  conductive state for `H = 10`.

### Fixed

- The checkpoint writer creates the parent directory, so `blankenbach_2b.jl` and
  `blankenbach_2b_wall.jl` no longer fail with an HDF5 error at step 250 or 500 on a first
  run from a clean checkout.
- The Blankenbach 2b wall record: the level-4 family span is 0.22 percent (was stated as
  0.04), and the pre-`266a656` level-3 rows are kept as a labelled historical row.
- Source comments that named note files by old paths were repointed at the volumes in
  `docs/dev/` that absorbed them.

### Removed

- Eight files that no longer had a consumer: `dg2d.jl`, `interp.jl`, `newton.jl`,
  `production.jl`, `semquad_coupled.jl`, `semquad_saturation.jl`, `semquad_vtk.jl`
  and `trust_region.jl`, with the eight gates that pinned only them. None was on
  the solver path; the development repository keeps their history.

### Removed (second pass)

- The uniform structured-mesh solver this code was originally written around:
  `assembly.jl`, `timestep.jl`, `direct.jl`, `mesh.jl`, `bcs.jl`, `projection.jl`,
  `psi_stokes.jl`, `streamfunction.jl`, `rt.jl`, `rt1.jl`, `dg1d.jl`, `mg.jl`,
  `newton_solver.jl`, `refstate.jl`, the uniform-mesh `cbf.jl` and `io.jl`, the
  uniform-mesh diagnostics at the end of `physics.jl`, six examples that drove it,
  and the 31 gates that pinned it. The quadtree solver is unchanged: the ten-step
  production case reproduces `Nu_top = 5.999881`, `leaves = 2644`, `N = 11233`
  bit for bit after the removal, and the suite is 132,886 passing with no failure.

### Removed (third pass)

- The goal-oriented (DWR) AMR kernel `src/amr_dwr.jl` and the seventeen gates that
  pinned it. Nothing on the production path called it. The one function it carried
  that is still used, `assemble_semquad_velocity_mass`, moved verbatim to
  `src/semquad.jl`; `return_system` and `assemble_only` stay because the Newton and
  multigrid stack depends on them.

### Removed (fourth pass)

- `src/physics/rheology.jl`, the SI-unit ASPECT material model the uniform-mesh
  solver used. One function survived and moved to `physics/eta.jl`: `spd_factor`,
  the SPD stabilisation factor of the visco-plastic Jacobian, which the
  variable-viscosity assembly calls.
- The 110 exported names that bound nothing in the package. `export` of a missing
  name is legal Julia, so the whole surface of the earlier structured-grid layer (`Mesh`, `make_mesh`, `DG1D`,
  `rt0`, the psi-omega spaces, `run_simulation`) was still advertised to users and
  failed only when they touched it. The export list is now generated from what the
  module actually defines: 309 names to 199.
- Three ASCII initial conditions (`data/initial_temperature_case{1b,1c,2a}.txt`)
  whose only reader was a deleted gate, and `aspect_runs/realistic/compare.py`,
  superseded by `compare_E.py`. The initial conditions came back later, in
  `examples/data/`, as the benchmark drivers' starting fields.

### Fixed

- `t_g203`'s cost assertion timed a 200 microsecond loop 20 times, so the ratio
  moved between 1.16 and 1.55 on identical code. Both arms are warmed up and timed
  over 200 repetitions now; the 1.5x bound is unchanged.
