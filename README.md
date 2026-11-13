# QuaST

A quadtree spectral-element solver for two-dimensional mantle convection, with a
matrix-free multigrid Stokes solver verified against published benchmarks and against
ASPECT.

QuaST discretises the incompressible Boussinesq (or anelastic) Stokes and energy equations
with Taylor-Hood elements of degree `p` and `p-1` on Gauss-Legendre-Lobatto nodes, keeps
the mesh 2:1 balanced, enforces hanging-node continuity as hard constraints, integrates in
time with BDF2, and adapts the mesh with a policy that trades error against a fixed leaf
budget. Around that core sit the pieces a mantle-convection case needs: composite
diffusion-dislocation rheology, phase transitions with per-transition latent heat,
internal heating, depth-dependent yielding with strain weakening, and an equation-of-state
reference state. The Stokes saddle is solved either by a sparse direct factorisation or by
a matrix-free geometric multigrid whose defaults were measured, not guessed.

## What it is, and what it is not

It is a research solver for two-dimensional, incompressible mantle convection on an
adaptive mesh, written in Julia, with a gate suite that pins the numerics and a body of
benchmark and cross-code evidence behind every number in `docs/`.

It is not a general-purpose geodynamics package. It is two-dimensional, it does not do
plate boundaries, viscoelasticity, compressible MHD or three dimensions, and the validation
that exists covers the cases in `docs/VALIDATION.md`, not everything you may want to run.

## Requirements

Julia 1.12. About 1.5 GB of memory for the production case at the default budget, and
eight threads if you want the multigrid to be at its best.

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

The repository ships one flagship case, a 2.5 x 1 box with composite rheology, two phase
transitions and internal heating, and it has a ten-step smoke mode:

```bash
CASE_OUT=/tmp/quast_smoke julia --project=julia julia/scripts/case_realistic_2d.jl smoke
```

Always set `CASE_OUT`; the fallback output directory is one an earlier run may have used.
The smoke mode builds the default 10 000-leaf budget and takes about two minutes including
compilation. The other modes are `short`, `spinup`, `freeze`, `wallrefine`, `resume` and
`symseed`, and the environment variables that steer them are in
[docs/CASE_REALISTIC.md](docs/CASE_REALISTIC.md).

As a library, the entry point is `run_semquad_simulation`. A minimal case builds a tree,
a parameter card and an initial temperature, then advances it:

```julia
using MantleSEM

pr = Params(Ra=1.0e5, H=10.0, eta_law=:expTz,
            eta_b=9.70406052783923433184, eta_c=4.1588830833596718565,
            Twall_top=0.0, Twall_bot=1.0, bc=:free)

qt = Quadtree(5; cells_x=5, cells_y=2, Lx=2.5)   # level-5 seeds on a 5 x 2 base grid
for _ in 1:2, lf in collect(qt.leaves)            # two uniform refinements
    refine!(qt, lf)
end
balance!(qt)                                      # enforce the 2:1 rule

ss = make_semstokes(qt, 2)                        # the Q2/Q1 space on this tree
theta0 = [0.1 * sinpi(y) * cospi(x / qt.Lx) for (x, y) in ss.su.coords]

res = run_semquad_simulation(qt, pr, theta0; p=2, maxsteps=10, amr_every=1,
                             max_leaves=1_000, amr_policy=:balanced,
                             linear=:auto, cfl=0.5, checkpoint_path="/tmp/quast.h5")
```

`res` carries the history matrix and the end state. `julia/scripts/case_realistic_2d.jl`
is the full card, including the rheology, the transitions and the latent coefficients.

The defaults are `linear=:auto` and `amr_policy=:balanced`. The solver picks the linear
path from the saddle size and the sampled viscosity contrast, taking the matrix-free
multigrid for the larger or lower-contrast problems and the sparse direct factorisation
elsewhere, with a fallback that redoes the solve directly if the multigrid misses the
tolerance. `linear=:direct` and `linear=:mg` remain selectable, and `pic_mode=:newton`
requires `:direct` explicitly.

## What the numbers look like

On the production state (8 188 leaves, 69 388 saddle degrees of freedom, viscosity contrast
5.7e6) the warm Stokes solve at eight threads is 1.36 s with the matrix-free multigrid and
3.37 s with a direct factorisation that actually reaches the 1e-6 tolerance, a factor of
2.5, at lower memory. The crossover, measured on seven tree sizes, is documented in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) and [docs/MG.md](docs/MG.md).

Against ASPECT, run on the same physical card from the same initial condition, the top
heat flux agrees to between 0.005 and 0.9 percent over the window, the same-state import
reproduces ASPECT's diagnostics to 0.003 percent, and the bottom flux differs by six to
nine percent depending on the window with the cause not fully decomposed. The evidence,
including the two-arm linear-solver comparison, is in [docs/VALIDATION.md](docs/VALIDATION.md).

## Tests

The gate suite is the contract that the numerics have not drifted. It is about 220 gate
files and 133 600 assertions and takes roughly 35 minutes.

```bash
julia --project=julia julia/test/runtests.jl                       # everything
julia --project=julia julia/test/run_gates.jl t_g229_default_balanced.jl
```

## Documentation

| Document | What it answers |
|---|---|
| [SOLVER.md](docs/SOLVER.md) | The equations, the discretisation, the four linear paths and the `:auto` rule, the physics inventory, the limits |
| [PHYSICS.md](docs/PHYSICS.md) | The energy operator, the rheology laws, latent heat per transition, ALA versus TALA, the EOS reference state |
| [AMR.md](docs/AMR.md) | The quadtree, the balanced marking policy and its defaults, the 2:1 gate, the local rebalance, the hot-path costs |
| [MG.md](docs/MG.md) | The matrix-free multigrid: the algorithm, the ten optimisation stages with their measured numbers, the studies behind the defaults, the rejected levers |
| [MG_LITERATURE.md](docs/MG_LITERATURE.md) | What mature codes do, with sources, and what each claim cost to verify |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | Where a step's time goes, the direct-versus-multigrid verdict, the crossover table, allocations and scaling |
| [VALIDATION.md](docs/VALIDATION.md) | The benchmarks, the Nusselt-number conventions and their error terms, the energy budget, the ASPECT comparison |
| [CASE_REALISTIC.md](docs/CASE_REALISTIC.md) | The realistic card, its defaults, the launch protocol, the production runs and the three-arm verification |
| [VIDEOS.md](docs/VIDEOS.md) | The rendering pipeline for the comparison videos |
| [BUG_LEDGER.md](docs/BUG_LEDGER.md) | Every serious defect found, with symptom, mechanism, measurement, fix and the gate that pins it |
| [HANDOFF.md](docs/HANDOFF.md) | The current state, the open problems, the runbook and the rules the project learned the hard way |

The documents quote measured numbers and name the log each came from. Those logs are
development artifacts and are not shipped; the reproduction commands in each document
regenerate them.

## The ASPECT comparison

`aspect_runs/realistic/` holds the `.prm` inputs and the small C++ plugin that supplies the
composite rheology to ASPECT 2.5, plus the analysis scripts. Running the comparison needs
the ASPECT image, which is not part of this repository.

## Licence

MIT. See [LICENSE](LICENSE).
