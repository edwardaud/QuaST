<p align="right"><a href="../en/examples.md">English</a> · <a href="../zh/examples.md">中文</a></p>

# Examples

Four runs you can start immediately. Each comes with its command, the physics it computes, the measured runtime on eight threads, and a number to check against. Run everything from the repository root. The first command is needed once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The timings are wall clock on eight Julia threads, not CPU time. The direct solver is serial inside the process, so a run on that path leaves seven threads idle. The measurement records and their conditions are in the validation volume and the realistic-case volume.

| Example | Command | Wall clock, eight threads | What to check |
|---|---|---|---|
| Blankenbach 2b, steady | `benchmark_cases.jl 2b 3 0.2` | about 45 minutes | `Nu_top = Nu_bot` |
| Blob | `run_blob_kelly_1024.jl` | about two minutes | `Tmax25` and the two digests |
| Realistic case, smoke | `case_realistic_2d.jl smoke` | about two minutes the first time | `Nu_top = 5.999881` |
| Programmatic run | `build_and_run.jl` | about 84 s the first time | `Nu_top = Nu_bot = 1.000783` |

## 1. Blankenbach 2b

The 2b benchmark is a 2.5 x 1 box with free-slip walls, `Ra = 1e4`, no internal heating, and a viscosity that depends on both temperature and depth:

$$
\eta = \exp(-bT + c(1-z)),
\qquad
b = \ln 16384,
\qquad
c = \ln 64,
$$

where $T$ is temperature, $z$ is height, and $\eta$ is viscosity. The reference values are `Nu = 6.9299` and `Vrms = 171.755` (Blankenbach et al. 1989, Table 9).

### Running to a steady state

`examples/benchmark_cases.jl` is the entry point to use now. It starts from ASPECT's own temperature field, which is already close to the steady state, so a run needs hundreds or thousands of steps instead of tens of thousands:

```bash
julia --project=. --threads=8 examples/benchmark_cases.jl 2b 3 0.2
```

The arguments are `case level tstop`. Level 3 refines a 5 x 2 base grid three times, giving 40 x 16 square leaves at `p = 2` on the direct solver. The run prints

```text
[2b] steps=5492 t=2.0001e-01 wall=20530 s  Nu_top=7.248805 Nu_bot=7.250974 Vrms=170.213063  leaves=640
[2b] reference Nu=6.929900 (4.60%)  Vrms=171.755000 (-0.90%)
```

The two wall fluxes agree to 0.03%, so the state is steady. The 4.60% offset in `Nu` is the wall boundary layer being under-resolved at this mesh, not an implementation error. Agreement between the top and bottom fluxes is the acceptance test: with no internal heating, a steady state cannot have them differ.

### Why the wall needs its own refinement

`Nu` is a one-sided gradient inside the boundary layer, and uniform refinement spends its cells in the interior where nothing depends on them. The measurements make the point:

| 2b state | `Nu` against the reference |
|---|---:|
| uniform level 3 (40 x 16) | +4.60% |
| uniform level 4 (80 x 32) | +5.31% (stopped before convergence) |
| wall-refined level 4 | +0.26% |

Uniform refinement from level 3 to level 4 does not help and in fact makes the number slightly worse, while wall refinement reaches 0.26% with far fewer cells. Turn the wall rule on with

```bash
CASE_WALLMIN=5 CASE_INITREF=1 julia --project=. --threads=8 examples/benchmark_cases.jl 2b 4 0.2
```

`CASE_WALLMIN` sets the minimum level for the wall strip and the wall rows, and `CASE_INITREF=1` runs one adaptation pass on the initial condition. On that wall-refined state the three Nusselt definitions — wall gradient, steady consistent boundary flux, and energy — give 6.9480, 6.9325 and 6.9330, a spread of 0.22%.

## 2. The blob regression

The blob is the `Ra = 1e8` regression that pins the adaptive path. The box is 1 x 1, the walls are free slip, the heating is `H = 5`, the viscosity is the same two-exponent law, and the initial condition is the conductive base state plus one Gaussian:

$$
\theta_0(x,z) = 0.12\exp\left[-\left(\frac{x-0.5}{0.12}\right)^2-\left(\frac{z-0.35}{0.10}\right)^2\right]\sin(\pi z),
$$

where $\theta\_0$ is the initial temperature anomaly and $x$ and $z$ are the horizontal and vertical coordinates.

```bash
julia --project=. --threads=8 examples/run_blob_kelly_1024.jl
```

The script runs with `p = 4` on the direct solver, adapts every 5 steps, keeps a 1 024-leaf hard cap, and writes 256 x 256 frames every `4e-10` up to `t = 1e-8`. It clears its output directory at start, so point `BLOB_OUT` elsewhere if you want a different path. The run ends at 898 leaves and 25 frames. On this machine, eight threads finish the 126 steps in about two minutes including Julia startup.

The last two lines are

```text
done t=1.008e-08 steps=126 leaves=898 N=15361 frames=25 in 141.3 s
FINAL frames=25 end_theta_sha256=a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda end_u_sha256=1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51 Tmax25=1.264329738
```

`end_theta_sha256` is the bitwise pin and `Tmax25` is the scalar to compare. ASPECT reports `Tmax = 1.2531` at the same frame, so the computed 1.264330 is **+0.9%** high. The 2.87% field-level difference was measured for the classic arm and was never re-measured for the current balanced one, so do not quote 2.87% for this default run. The digest holds bitwise only on the pinned Julia, BLAS and thread stack; `Tmax25` is the check that tolerates a different machine.

## 3. The realistic case

The production case is a 2.5 x 1 box with `Ra = 1e5` and `H = 10`, composite isostress rheology, phase transitions at `z = 0.87` and `0.78` with per-transition latent heat, `p = 2`, a `max_level = 7` tree, a 10 000-leaf cap, and adaptation every 5 steps. The initial anomaly is

$$
\theta_0(x,y) = 0.05\sin(\pi y)\cos\left(\frac{2\pi x}{2.5}\right) + 0.02\sin(2\pi y)\cos\left(\frac{4\pi x}{2.5}\right),
$$

where $x$ and $y$ are the horizontal and vertical coordinates. Always point `CASE_OUT` at a scratch directory: the fallback path may hold files left by a production run.

The smoke test runs 10 steps with the full case:

```bash
CASE_OUT=out_julia/demo_smoke CASE_STEPS=10 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

The first run takes about two minutes, mostly Julia compilation and the first Stokes solve; on a warm machine it is 47.6 s. It must end with `[case] DONE mode=smoke steps=10` followed by

```text
[case] final: Nu_top=5.999881 Nu_bot=-4.029590 Vrms=6.808435e+03 meanT=1.333328 closure=-8.794e-04 leaves=2644 N=11233
```

The conductive reference for this base state is `Nu_top = 6`, so the top flux is right to **2e-5 relative** after ten steps. The bottom sits 0.74% below its `-4` reference because the bottom boundary layer is still forming.

The 2 000-step spin-up is the same card with adaptation on:

```bash
CASE_OUT=out_julia/demo CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

The tree grows from 640 leaves to the 10 000-leaf cap within the first 100 steps, and `spinup.h5` is written every 200 steps and again at exit. This page has not run the full 2 000 steps. At the project's record of 4.13 s per step at the cap, 2 000 steps is about **2.3 hours**, an upper bound because the early steps are cheaper while the tree is still small. The case has no analytic reference: after 2 000 steps it is still in the startup transient. Compare it against another run of the same card, not against the 20 000-step endpoint (`Nu_top = 18.2892`, `Nu_bot = -2.7636`), which is a different regime.

`resume` and `freeze` are two ways to continue the same checkpoint. `resume` keeps adaptation on; `freeze` rewrites only `amr_every` to 0 and continues on the frozen mesh:

```bash
CASE_OUT=out_julia/demo_frozen CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl freeze \
  out_julia/demo/spinup.h5

CASE_OUT=out_julia/demo_resume CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

Both carry the state from step 2 000 to step 4 000, about **2.3 hours** each at the recorded rate; neither was re-run for this page. Give each its own `CASE_OUT` so the `frames/` and `history.csv` do not collide. `CASE_STEPS` is an absolute step number, so a budget at or below the checkpoint step makes the driver run zero steps and write nothing. Asking `resume` to continue a frozen checkpoint fails the configuration guard, because that mode passes `amr_every = 5`; continue that branch with `freeze` on `frozen.h5`.

## 4. A programmatic run

`run_semquad_simulation` takes a tree, a `Params` card and an initial temperature anomaly. Save this block as `build_and_run.jl`; it is complete and prints one history row per step.

```julia
using QuaST, Printf

# 1. The tree: a unit box on a 2 x 2 base grid, three uniform refinements
#    (2*2*4^3 = 256 leaves, 2:1 balanced).
qt = Quadtree(3; Lx=1.0, cells_x=2, cells_y=2)
for _ in 1:3, lf in collect(qt.leaves)
    refine!(qt, lf)
end
balance!(qt)

# 2. The parameter card: Boussinesq, constant viscosity, free slip.
#    :expT with deta_T = 1 is the constant-viscosity limit of the T law.
pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
            Twall_top=0.0, Twall_bot=1.0, bc=:free)

# 3. The initial temperature anomaly, zero on both walls.
ss = make_semstokes(qt, 2)
theta0 = [0.01 * cospi(x) * sinpi(y) for (x, y) in ss.su.coords]

# 4. Advance 20 steps, then read the history back.
res = run_semquad_simulation(qt, pr, theta0; p=2, tstop=Inf, maxsteps=20,
                             dt0=1.0e-4, dtmax=1.0e-4, cfl=0.0,
                             linear=:direct, verbose=false)
@printf("steps=%d t=%.6e leaves=%d N=%d\n",
        res.steps, res.t, length(res.qt.leaves), res.ss.su.N)
@printf("%3s %12s %10s %10s %10s\n", "i", "t", "Nu_top", "Nu_bot", "Vrms")
for (i, row) in enumerate(res.history)
    @printf("%3d %12.5e %10.5f %10.5f %10.5f\n", i, row[1], row[3], row[4], row[5])
end
@printf("last row: Nu_top=%.6f Nu_bot=%.6f Vrms=%.6f\n",
        res.history[end][3], res.history[end][4], res.history[end][5])
```

```bash
julia --project=. --threads=8 build_and_run.jl
```

The first run takes about 84 s, almost all of it compiling the driver on first call. Each history row is `[t, dt, Nu_top, Nu_bot, Vrms, meanT, N, leaves, ...]`, so `row[3]` and `row[4]` are the wall Nusselt numbers and `row[5]` is `Vrms`. The output ends with

```text
steps=20 t=2.000000e-03 leaves=256 N=1089
last row: Nu_top=1.000783 Nu_bot=1.000783 Vrms=2.762497
```

The sharpest check here is symmetry: the flow is reflection-symmetric about the mid-plane, so `Nu_top` and `Nu_bot` must print the same value, and a difference above `1e-6` means the Stokes or energy solve was disturbed. The 0.000783 offset from the conductive value of 1 is the physical transient at `t = 2e-3`, not solver error. Rerunning on a different BLAS build should agree to `1e-5` in `Nu` and `Vrms`; the direct solver's round-off is far below that.
