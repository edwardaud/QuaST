# QuaST

<p align="center">
  <b>A spectral-element solver for two-dimensional mantle convection</b><br>
  adaptively refined quadtree · matrix-free multigrid Stokes solve
</p>

<p align="center">
  <a href="docs/en/README.md">Documentation</a> ·
  <a href="docs/en/quickstart.md">Quick start</a> ·
  <a href="docs/en/examples.md">Examples</a> ·
  <a href="docs/en/validation.md">Validation</a> ·
  <a href="README.zh.md">中文</a>
</p>

<p align="center">
  <a href="https://github.com/edwardaud/QuaST/actions/workflows/gates.yml"><img alt="Gates" src="https://github.com/edwardaud/QuaST/actions/workflows/gates.yml/badge.svg"></a>
  <img alt="Julia" src="https://img.shields.io/badge/Julia-1.12-9558B2?logo=julia&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-131%20851%20assertions-brightgreen">
  <img alt="Docs" src="https://img.shields.io/badge/docs-English%20%7C%20中文-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

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

> [!WARNING]
> Set `CASE_OUT` every time. Without it the card writes into `out_julia/case_realistic`, which may already hold a run you care about.

A production spin-up, about two hours on eight threads:

```bash
mkdir -p out_julia/run1
CASE_OUT=out_julia/run1 CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

Progress goes to `run.log`, per-step diagnostics accumulate in `history.csv`, and `frames/` holds the temperature field for plotting.

## What it does

| | |
|---|---|
| **Mesh** | Quadtree adaptation with balanced refinement, a 2:1 legality check, local rebalancing, and wall boundary layers held at a minimum level |
| **Discretization** | $Q\_p$ velocity and $Q\_{p-1}$ pressure on GLL nodes, hanging nodes eliminated through exact constraints |
| **Physics** | Composite diffusion–dislocation rheology, depth-dependent Drucker–Prager yielding with strain weakening, per-transition latent heat, internal heating, Adams–Williamson or Murnaghan reference states |
| **Linear algebra** | Sparse direct and matrix-free multigrid Stokes solvers behind one `linear` switch; `:auto` chooses from the saddle size and the viscosity contrast |
| **Restart** | Bitwise: resuming from the same checkpoint with the same configuration reproduces the trajectory exactly |

## Validation

**Table 1.** Published benchmarks against this solver. `Ours` is the best converged run for each
case; `ASPECT` is the same quantity from the reference statistics ASPECT ships with its
benchmark cards (Blankenbach) or from the ASPECT column of Table 2 in Tosi et al. (2015).
The last two columns are differences from the published value and from ASPECT. A dash means
the quantity is not defined for that case.

| Case | Quantity | Published | ASPECT | Ours | Δ vs published | Δ vs ASPECT |
|:-----------------|:-------------|-------------:|-------------:|-------------:|-------------:|-------------:|
| Blankenbach 1a | `Nu` | 4.884409 | 4.884409 | **4.888163** | +0.08% | +0.08% |
| | `Vrms` | 42.864947 | 42.864956 | **42.8724** | +0.02% | +0.02% |
| Blankenbach 1b | `Nu` | 10.534095 | 10.533912 | **10.568610** | +0.33% | +0.33% |
| | `Vrms` | 193.21454 | 193.22175 | **193.2137** | −0.00% | −0.00% |
| Blankenbach 1c | `Nu` | 21.972465 | 21.972523 | **22.315960** | +1.56% | +1.56% |
| | `Vrms` | 833.98977 | 833.99083 | **834.0082** | +0.00% | +0.00% |
| Blankenbach 2a | `Nu` | 10.066 | 10.065793 | **10.124814** | +0.58% | +0.59% |
| | `Vrms` | 480.4334 | 480.43579 | **479.695** | −0.15% | −0.15% |
| Blankenbach 2b | `Nu` | 6.9299 | 6.929727 | **6.992620** | +0.91% | +0.91% |
| | `Vrms` | 171.755 | 171.75354 | **171.634** | −0.07% | −0.07% |
| Tosi 1 | `Nu_top` | 3.4298 | 3.4305 | **3.4216** | −0.24% | −0.26% |
| | `u_rms` | 251.80 | 251.3069 | **249.53** | −0.90% | −0.71% |
| Tosi 5a | period | 0.0792 | — | **0.0783** | −1.14% | — |
| | `Nu_top`, min | 2.6950 | — | **2.6924** | −0.10% | — |
| | `Nu_top`, max | 7.2792 | — | **7.4889** | +2.88% | — |
| | mean `T`, min | 0.6543 | — | **0.6505** | −0.58% | — |
| | mean `T`, max | 0.6722 | — | **0.6686** | −0.54% | — |
| Blob, `Ra = 1e8`, frame 25 | `Tmax` | — | 1.2531 | **1.2644** | — | +0.90% |
| Realistic, ASPECT state imported | `Nu_top` | — | 5.990543 | **5.990543** | — | +0.000% |
| | `Nu_bot` | — | −3.075076 | **−3.074986** | — | +0.003% |

The last three rows compare against ASPECT rather than against a published number: the blob is
a 25-frame transient, and the two realistic rows start from an imported ASPECT state, so a
difference there is a discretization difference and nothing else — 0.003% at the bottom wall.
The details behind each row are in [the validation page](docs/en/validation.md).

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
