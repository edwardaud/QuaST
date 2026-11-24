# QuaST

<p align="center">
  <b>A spectral-element solver for two-dimensional mantle convection</b><br>
  adaptively refined quadtree · matrix-free multigrid Stokes solve
</p>

<p align="right"><a href="README.zh.md">中文</a></p>

QuaST solves the incompressible Boussinesq or anelastic Stokes and energy equations for thermal convection in a two-dimensional Cartesian box. Velocity and pressure are Taylor–Hood elements of polynomial degree `p` and `p - 1` on Gauss–Legendre–Lobatto nodes, the mesh is a 2:1-balanced quadtree that adapts to the solution, and the temperature is advanced with BDF2. The Stokes saddle is solved by sparse direct factorization or by matrix-free geometric multigrid, and `linear = :auto` chooses between them from the saddle size and the viscosity contrast. The material model covers composite diffusion–dislocation creep, depth-dependent Drucker–Prager yielding, strain weakening, phase transitions with latent heat, internal heating, and Maxwell viscoelasticity.

![The same realistic case run by QuaST and by ASPECT](docs/assets/quast_vs_aspect_realistic.gif)

The same realistic case run by QuaST (top) and by ASPECT (bottom), from the same initial state: temperature, its anomaly about the horizontal mean, viscosity on a logarithmic color scale, and the mesh colored by refinement level.

## What it does, and what it does not

| It does | It does not |
|---|---|
| Solve in a 2-D Cartesian box with free-slip or no-slip walls, set per wall, optionally with a prescribed wall velocity | Run in 3-D or in a spherical shell |
| Support the Boussinesq and anelastic mass forms and the TALA/ALA reference state | Model toroidal motion, plate boundaries, magnetic fields or a free surface |
| Adapt a 2:1-balanced quadtree with a gradient-jump indicator over temperature, velocity or both, and stabilize the energy step with entropy viscosity | Carry full elasticity as a separate rheology; the Maxwell model is the one elastic behavior |
| Carry composite diffusion–dislocation rheology, depth-dependent Drucker–Prager yielding, strain weakening, phase transitions with latent heat, and internal heating | Move the top boundary: the Maxwell stress history is not a free surface and produces no topography |
| Solve the saddle with a sparse direct factorization or matrix-free geometric multigrid, selected by `linear = :auto`, and restart bitwise from an HDF5 checkpoint | Use an unstructured mesh: the mesh is a quadtree of square leaves |

## Three commands to a first result

```bash
git clone git@github.com:edwardaud/QuaST.git && cd QuaST
julia --project=. -e 'using Pkg; Pkg.instantiate()'
CASE_OUT=/tmp/quast_smoke julia --project=. examples/case_realistic_2d.jl smoke
```

The second command takes five to ten minutes on a clean depot and under 2 GB of disk. The third runs ten steps of the shipped realistic card, about two minutes the first time and almost all of it compilation; it adapts the mesh at steps 5 and 10, writes `history.csv` and `frames/` under `CASE_OUT`, writes no checkpoint, and ends with a `[case] DONE mode=smoke steps=10` line. **Set `CASE_OUT` on every run:** without it the card writes into `out_julia/case_realistic`, which may hold an earlier spin-up. Julia 1.12 is the version the commands were checked on; `Project.toml` admits 1.10 and later.

The ten steps are a smoke test, not a result. For the files a run produces, the first settings to change, and what to do when something fails, see the [quick start](docs/en/quickstart.md).

## Verification

| Check | Result | Recorded in |
|---|---|---|
| Blankenbach et al. (1989) case 1a at refinement level 5, steady state: `Nu` against the published 4.884409 | **+0.08%** | [Validation](docs/en/validation.md) |
| The same state imported from ASPECT, advanced by both codes: `Nu_bot` against ASPECT | **+0.003%** | [Validation](docs/en/validation.md) |
| The automated test suite | **133 714** assertions passed, 0 failed, and 2 broken entries that wait for checkpoints the repository does not ship | [CONTRIBUTING.md](CONTRIBUTING.md) |

The validation page carries the convergence behind the Blankenbach number, the Tosi limit-cycle results, the viscoelastic checks, the two Nusselt conventions, and the paragraph that says where the evidence stops. The suite runs with `julia --project=. test/runtests.jl`.

## Documentation

Start at the [documentation map](docs/README.md), which lists every page and the reading order. For a first run, read the [quick start](docs/en/quickstart.md); for a case to copy, the [examples](docs/en/examples.md); for the method and every knob, the [guide](docs/en/guide.md). The [validation page](docs/en/validation.md) carries the evidence, the [FAQ](docs/en/faq.md) the common changes, and the [architecture page](docs/architecture.md) the code layout.

## License

MIT, see [LICENSE](LICENSE).

**Next:** the [documentation map](docs/README.md) gives the reading order; start with the [quick start](docs/en/quickstart.md).
