# Examples

Four runs, each with the command, the physics, the measured runtime on eight threads and the
number to check. Run everything from the repository root. The first command is needed once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The timings below are wall clock on eight Julia threads, not CPU time; the direct solver is
serial inside that process, so a direct path does not use the other seven threads. The
measured records and their conditions are in [dev/VALIDATION.md](dev/VALIDATION.md) and
[dev/CASE_REALISTIC.md](dev/CASE_REALISTIC.md).

## 1. Blankenbach 2b

The 2b benchmark is a 2.5 x 1 box with free-slip walls, `Ra = 1e4`, no internal heating, and
the joint viscosity law `eta = exp(-b T + c (1 - z))` with `b = ln 16384` and `c = ln 64`.
The reference values are `Nu = 6.9299` and `Vrms = 171.755` (Blankenbach et al. 1989, Table 9).

```bash
julia --project=. --threads=8 examples/run_2b.jl
```

The arguments are `[nxe] [nze] [stage...]`; the defaults are 40 x 16, `p = 4`, and the four
continuation stages `s = 0.25, 0.5, 0.75, 1.0` that scale `b` and `c` from a mild contrast to
the full one. The initial temperature is `T0 = (1 - z) + 0.01 cos(pi x / 2.5) sin(pi z)`, and
each stage runs the steady Stokes-energy Picard solver with the direct solver, `maxiter = 80`
and `tol = 1e-8`. The run takes **116 s** on eight threads (measured with `time` on this
checkout), and the direct solves are serial, so the thread count does not change the numbers.

The last line is

```
STAGE s=1.000  Nu_top=6.942798 Nu_bot=6.938484 Vrms=170.962139 it=39  (ref 6.9299 / 171.755)
```

The pointwise wall `Nu_top` is **+0.19 percent** above the reference, and `Vrms` is
**-0.46 percent** below it. That level is inside the historical level-3 scatter: the pre-fix
record quotes the energy-family `Nu = 6.9874` (+0.83 percent) and `Vrms = 171.18` (-0.33
percent), and the wall family was the least converged at this resolution. Use the wall-refined
level-4 state for a percent-level `Nu`: there the wall, CBF and energy families span 0.22
percent (`6.9480, 6.9325, 6.9330` against 6.9299) and `Vrms = 172.010` (+0.15 percent).

The gate pins the 2b trajectory; the steady value is a published reference.
`t_g150_regression_snapshot` checks a 20-step 2b run from ASPECT's initial condition at
`1e-6` relative, and the same 20-step state has the FNV-1a `theta` hash
`d22fdab099ae84e0` and `Nu_top = 7.0299964227238032`, which reproduce bitwise on the pinned
stack ([dev/BUG_LEDGER.md](dev/BUG_LEDGER.md) section 1.3). The run above compares with
6.9299; no gate asserts 171.18.

## 2. The blob regression

The blob is the `Ra = 1e8` regression that pins the adaptive path. The box is 1 x 1, the walls
are free slip, the heating is `H = 5`, the viscosity is the same two-exponent law, and the
initial condition is the conductive base state plus one Gaussian:

```
theta0(x, z) = 0.12 exp(-((x - 0.5)/0.12)^2 - ((z - 0.35)/0.10)^2) sin(pi z)
```

```bash
julia --project=. --threads=8 examples/run_blob_kelly_1024.jl
```

The producer runs 126 steps with `p = 4`, the direct solver, Kelly marking every 5 steps, a
1024-leaf hard budget and 256 x 256 frames every `4e-10` up to `t = 1e-8`. It deletes its
output directory at start, so set `BLOB_OUT` if you want a different path. The default
(`BLOB_POLICY=balanced`) ends at 898 leaves and 25 frames. On this checkout the producer
reported **116.1 s** for the 126 steps, with 3 min 15 s total wall including Julia startup.

The final lines are

```
done t=1.008e-08 steps=126 leaves=898 N=15361 frames=25 in 116.1 s
FINAL policy=balanced frames=25 end_theta_sha256=a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda end_u_sha256=1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51 Tmax25=1.264329738
```

`end_theta_sha256` is the bitwise pin, and `Tmax25` is the scalar to compare. Against ASPECT's
`Tmax = 1.2531` at the same frame, the computed 1.264330 is **+0.9 percent**; the field-level
difference was 2.87 percent for the classic arm and was not re-measured for the balanced arm,
so do not quote 2.87 percent for this default. The classic arm is available with
`BLOB_POLICY=classic BLOB_OUT=out_julia/blob_classic julia --project=. --threads=8
examples/run_blob_kelly_1024.jl`; its documented digest is
`868a79d92a78bcfe89cfacd4774a0301c0eeb0a865284ab005155f6f458afe88` with `Tmax25=1.264427445`
(project record, not re-run here). The digest is bitwise only on the pinned Julia, BLAS and
thread stack; `Tmax25` is the platform-tolerant check.

## 3. The realistic card

The flagship case is a 2.5 x 1 box with `Ra = 1e5`, `H = 10`, composite isostress rheology, two
phase transitions at `z = 0.87` and `0.78` with per-transition latent heat, `p = 2`, a
`max_level = 7` tree, a 10000-leaf budget and AMR every 5 steps. The initial anomaly is
`0.05 sin(pi y) cos(2 pi x / 2.5) + 0.02 sin(2 pi y) cos(4 pi x / 2.5)`. Always set `CASE_OUT`
to a scratch directory; the fallback path is one a production run may have used.

The smoke is 10 steps with the full card:

```bash
CASE_OUT=out_julia/demo_smoke CASE_STEPS=10 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

It takes **108.5 s** on eight threads, most of it Julia compilation and the first Stokes
solve, and must end with `[case] DONE mode=smoke steps=10` and `[case] final: Nu_top=5.999881
Nu_bot=-4.029590 Vrms=6.808435e+03 ... leaves=2644 N=11233`. The conductive reference for this
base state is `Nu_top = 6`, so the top flux is right to **2e-5 relative** after 10 steps; the
bottom is 0.74 percent below its `-4` reference because the bottom boundary layer is still
forming.

The 2000-step spin-up is the same card with AMR on:

```bash
CASE_OUT=out_julia/demo CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

It grows the tree from 640 leaves to the 10000-leaf cap in the first 100 steps and writes
`spinup.h5` every 200 steps and at exit. I did not run the full 2000 steps for this page. The
project's production record is 4.13 s/step at the 10000-leaf cap, so 2000 steps is about
**2.3 h** at that rate; the early steps are cheaper while the tree grows, which makes 2.3 h an
upper bound. The card has no analytic reference: the 2000-step state is still the startup
transient, and the two production policies agree on `Nu_top` to 0.01 percent in the common
early window but differ by 5.1 percent at `t = 4.34e-3`. Compare a 2000-step run with another
run of the same card, not with the 20000-step E endpoint (`Nu_top = 18.2892`,
`Nu_bot = -2.7636`).

Freeze and resume are two ways to continue the same checkpoint. `resume` keeps AMR on;
`freeze` rewrites only `amr_every` to 0 and continues on the frozen mesh:

```bash
CASE_OUT=out_julia/demo_frozen CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl freeze \
  out_julia/demo/spinup.h5

CASE_OUT=out_julia/demo_resume CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

Both run 2000 further steps from step 2000 to step 4000. At the project's 4.13 s/step record
each is about **2.3 h**; neither full branch was re-run for this page. Use separate
`CASE_OUT` directories: each writes its own `frames/` and `history.csv`. `CASE_STEPS` is an
absolute step number, so a budget at or below the checkpoint step makes the driver run zero
steps and write nothing. `resume` on a frozen checkpoint fails the guard because the mode
passes `amr_every = 5`; continue that branch with `freeze` on `frozen.h5`.

## 4. A programmatic run

`run_semquad_simulation` takes a tree, a `Params` card and an initial anomaly. Save this block
as `build_and_run.jl`; it is complete and prints one history row per step.

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

# 4. Advance 20 steps and read the history back.
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

The first run takes **84 s** on eight threads, almost all of it Julia's first-call
compilation of the driver. The history rows are
`[t, dt, Nu_top, Nu_bot, Vrms, meanT, N, leaves, ...]`, so `row[3]` and `row[4]` are the wall
Nusselt numbers and `row[5]` is `Vrms`. The output ends at

```
steps=20 t=2.000000e-03 leaves=256 N=1089
last row: Nu_top=1.000783 Nu_bot=1.000783 Vrms=2.762497
```

The exact check is symmetry: the flow is reflection-symmetric, so `Nu_top` and `Nu_bot` print
the same value; a difference above `1e-6` means the Stokes or energy solve was disturbed. The
0.000783 offset from the conductive value 1 is the physical transient at `t = 2e-3`, not solver
error. A rerun on another BLAS build should agree to `1e-5` in `Nu` and `Vrms`; the direct
solver's round-off is far below the printed digits.
