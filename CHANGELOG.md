# Changelog

All notable changes to this project are recorded here, in Keep a Changelog format with
semantic versioning; in the 0.x series the public API may still change. This is the first
public release: earlier development happened in a private repository.

## [0.1.0] - 2026-11-13

### Added

- The solver: Q_p velocity and Q_{p-1} pressure on Gauss-Legendre-Lobatto nodes over a
  quadtree kept 2:1 balanced, exact hanging-node constraints, variable-step BDF2 time
  stepping, entropy viscosity, and a wall strip held at a minimum level.
- The physics stack: composite diffusion-dislocation rheology with the isostress rule,
  depth-dependent Drucker-Prager yielding with strain weakening, phase transitions with
  per-transition latent heat, internal heating, an Adams-Williamson or Murnaghan reference
  state, and the Boussinesq or anelastic mass form.
- Two linear paths behind one `linear` switch: sparse direct factorisation with a diagonal
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
