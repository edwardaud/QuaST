# QuaST

**English** · [中文](README.zh.md)

A spectral-element solver for two-dimensional mantle convection, on an adaptively refined quadtree, with a matrix-free multigrid Stokes solver.

QuaST solves the incompressible Boussinesq or anelastic Stokes and energy equations with Taylor–Hood elements of degree $p$ and $p-1$ on Gauss–Legendre–Lobatto nodes. The mesh is a quadtree held at 2:1 balance, hanging nodes are eliminated by exact constraints rather than penalties, time stepping is BDF2, and the mesh adapts by trading error against a fixed leaf count. The Stokes saddle is solved either by sparse direct factorization or by matrix-free geometric multigrid, and `:auto` picks between them from the saddle size and the viscosity contrast.

This is a research code for two dimensions. It does not do three dimensions, plate boundaries, viscoelasticity, or compressible MHD.

## Install

Julia 1.12.

```bash
git clone git@github.com:edwardaud/QuaST.git
cd QuaST
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The instantiate step takes five to ten minutes and under 2 GB of disk.

## First run

Ten steps of the realistic card, about two minutes including compilation:

```bash
CASE_OUT=/tmp/quast_smoke julia --project=. examples/case_realistic_2d.jl smoke
```

Set `CASE_OUT` every time. Without it the card writes into `out_julia/case_realistic`, which may already hold a run you care about.

A production spin-up, about two hours on eight threads:

```bash
mkdir -p out_julia/run1
CASE_OUT=out_julia/run1 CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

Progress goes to `run.log`, per-step diagnostics accumulate in `history.csv`, and `frames/` holds the temperature field for plotting.

## What it does

- Quadtree adaptive meshing: balanced refinement, a 2:1 legality check, local rebalancing, and wall boundary layers that can be held at a minimum level.
- $Q\_p$ velocity and $Q\_{p-1}$ pressure on GLL nodes, with hanging nodes eliminated through exact constraints.
- Composite diffusion–dislocation rheology, depth-dependent Drucker–Prager yielding with strain weakening, per-transition latent heat, internal heating, and Adams–Williamson or Murnaghan reference states.
- Sparse direct and matrix-free multigrid Stokes solvers behind one `linear` switch; `:auto` chooses from the saddle size and the viscosity contrast.
- Bitwise restart: resuming from the same checkpoint with the same configuration reproduces the trajectory exactly.

## Validation

The published benchmarks have been run on several mesh levels, and the results are reported as convergence tables rather than single numbers. For case 1a ($Ra=10^4$), the `Nu` error falls from +1.08% at level 3 to +0.28% at level 4 and +0.08% at level 5. The error ratios are 3.9 and 3.5, against the factor of 4 a second-order element predicts, while `Vrms` stays within 0.02% at every level.

For case 2b, a wall-refined level-4 state brings the three Nusselt definitions — wall gradient, steady consistent boundary flux, and energy — within 0.22% of each other, and within 0.04% to 0.26% of the published value.

The ASPECT comparison runs at two levels: a cross-code comparison from a common initial condition, and a state-level comparison that imports an ASPECT checkpoint, where the two operators agree to 0.003% at the bottom wall. Over transient windows the bottom heat flux differs by −6.11% to +6.05% depending on the window. That gap is documented but not fully explained, and the [validation page](docs/en/validation.md) says so.

## Documentation

Every page carries a language switch. The full set starts at [docs/](docs/README.md).

| Document | What it covers |
|---|---|
| [Quick start](docs/en/quickstart.md) | Install, first run, stopping and resuming, the failures a new user hits |
| [Guide](docs/en/guide.md) | Equations and boundary conditions, discretization and linear solvers, every knob, the output formats |
| [Examples](docs/en/examples.md) | Four runnable cases with measured runtimes and numbers to check against |
| [Validation](docs/en/validation.md) | Benchmark convergence, the ASPECT comparison, the two Nusselt conventions |
| [FAQ](docs/en/faq.md) | Changing `Ra`, adding a phase, switching rheology, threading, restarting, citing |
| [Architecture](docs/architecture.md) | Where every file lives, and why |
| [Engineering record](docs/dev/README.md) | Derivations, performance history, defect ledger. For maintainers |

## Tests

```bash
julia --project=. test/runtests.jl                            # full suite, ~22 min
julia --project=. test/run_gates.jl t_g229_default_balanced.jl # one gate
```

The suite is 156 gate files and 131 851 assertions, covering bitwise regression pins, benchmark values, invariants, and accuracy classes. CI runs all of them.

## The ASPECT comparison

`aspect_runs/realistic/` holds the `.prm` inputs, the small C++ plugin that supplies the composite rheology to ASPECT 2.5, and the analysis scripts. Running them needs the ASPECT image, which is not part of this repository.

## License

MIT, see [LICENSE](LICENSE).
