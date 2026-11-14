# QuaST

A spectral-element solver for two-dimensional mantle convection on an adaptively refined
quadtree, with a matrix-free multigrid Stokes solver.

QuaST solves the incompressible Boussinesq or anelastic Stokes and energy equations with
Taylor-Hood elements of degree `p` and `p-1` on Gauss-Legendre-Lobatto nodes. The mesh is
a quadtree kept 2:1 balanced, hanging-node continuity is enforced exactly, time stepping is
BDF2, and the mesh adapts with a policy that trades error against a fixed leaf budget. The
Stokes saddle is solved either by a sparse direct factorisation or by matrix-free geometric
multigrid, and the solver picks between them from the problem size and the viscosity
contrast.

It is a research code for two dimensions. It does not do three dimensions, plate
boundaries, viscoelasticity or compressible MHD.

## Install

Julia 1.12.

```bash
git clone git@github.com:edwardaud/QuaST.git
cd QuaST
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The instantiate step takes five to ten minutes and under 2 GB.

## First run

Ten steps of the realistic card, about two minutes including compilation:

```bash
CASE_OUT=/tmp/quast_smoke julia --project=. examples/case_realistic_2d.jl smoke
```

Set `CASE_OUT` every time. Without it the card writes into `out_julia/case_realistic`,
which may already hold a run you care about.

A production spin-up, about two hours on eight threads:

```bash
mkdir -p out_julia/run1
CASE_OUT=out_julia/run1 CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

Progress goes to `run.log`, per-step diagnostics accumulate in `history.csv`, and
`frames/` holds the temperature field for the video pipeline.

## What it does

- Quadtree adaptive mesh with a balanced refinement policy, a 2:1 legality gate and local
  rebalancing; the wall boundary layer can be held at a minimum level.
- `Q_p` velocity and `Q_{p-1}` pressure on GLL nodes, with hanging nodes eliminated through
  exact constraints rather than penalties.
- Composite diffusion-dislocation rheology, depth-dependent Drucker-Prager yielding with
  strain weakening, phase transitions with per-transition latent heat, internal heating,
  and an Adams-Williamson or Murnaghan reference state.
- Sparse direct and matrix-free multigrid Stokes solvers behind one `linear` switch, with
  `:auto` choosing from the saddle size and the viscosity contrast.
- Bitwise-checkpointed runs: a resume from the same checkpoint and configuration
  reproduces the trajectory exactly.

## Validation

The Blankenbach 2b benchmark reproduces the published Nusselt number to the printed digits,
the blob regression reproduces its reference digest bitwise, the Tosi cases and the
composite rheology agree with the ASPECT constitutive card, and a case run in ASPECT from
the same initial condition agrees on the top heat flux to within one percent over the
window. The bottom heat flux differs by six to nine percent depending on the window; the
cause is documented but not fully decomposed. Details, tolerances and the limits of the
validation are in [docs/validation.md](docs/validation.md).

## Documentation

| Document | What it covers |
|---|---|
| [quickstart](docs/quickstart.md) | Install, smoke run, first production run, stopping and resuming, the failures a new user hits |
| [guide](docs/guide.md) | The physics, the method, every `CASE_*` variable and keyword, the output formats, threads and memory |
| [examples](docs/examples.md) | Four runnable cases with their runtimes and expected accuracy |
| [validation](docs/validation.md) | Can you trust it: benchmarks, the ASPECT comparison, the Nusselt conventions, the limits |
| [faq](docs/faq.md) | Changing `Ra`, adding a phase, threading, restarting, citing |
| [docs/dev/](docs/dev/README.md) | The engineering record: derivation, performance history, defect ledger. For maintainers, not a manual |

## Tests

```bash
julia --project=. test/runtests.jl                    # full suite, ~35 min
julia --project=. test/run_gates.jl t_g229_default_balanced.jl   # one gate
```

The suite is 156 gate files and 131 851 assertions, covering bitwise regression pins,
benchmark values, invariants and accuracy classes.

## The ASPECT comparison

`aspect_runs/realistic/` has the `.prm` inputs, the small C++ plugin that supplies the
composite rheology to ASPECT 2.5, and the analysis scripts. Running it needs the ASPECT
image, which is not part of this repository.

## Licence

MIT, see [LICENSE](LICENSE).
