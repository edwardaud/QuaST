<p align="right"><a href="../en/examples.md">English</a> · <a href="../zh/examples.md">中文</a></p>

# Examples

Seven runs you can start immediately. Each comes with its command and a number to check against. Run everything from the repository root, and run the first command once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The timings are wall clock on eight Julia threads, not CPU time. The direct solver is serial inside the process, so a run on that path leaves seven threads idle. A first run in a fresh Julia process is dominated by compiling the driver, about 70 s for the first `run_semquad_simulation` call; the simulations themselves are short.

| Example | Command | What to check |
|---|---|---|
| Realistic case, smoke | `julia --project=. --threads=8 examples/case_realistic_2d.jl smoke` | `Nu_top = 5.999881` |
| Blankenbach 2b, steady | `julia --project=. --threads=8 examples/benchmark_cases.jl 2b 3 0.2` | `Nu_top` and `Nu_bot` agree to 0.03% |
| Blob regression | `julia --project=. --threads=8 examples/run_blob_kelly_1024.jl` | `Tmax25 = 1.264329738` and the two digests |
| Regional channel | `julia --project=. --threads=8 examples/channel_flow_2d.jl` | `err_u` below `1e-10` |
| Regional subduction-style case | `julia --project=. --threads=8 examples/subduction_2d.jl rheology` | `Nu_top = 31.383` and `delta(0.8) = 0.0522` |
| 2-D annulus benchmark, `Ra = 1e5` | `ANN_CFL=0 ANN_DT0=1e3 ANN_MAXPIC=25 ANN_TMIN=0 ANN_WINDOW=3 ANN_TSTOP=1e12 julia --project=. --threads=8 examples/annulus_benchmark.jl 32x384` | `steady=1` and `Nu_T = 9.89994` |
| Programmatic run | `julia --project=. --threads=8 build_and_run.jl` | `Nu_top = Nu_bot = 1.000783` |

## The realistic card's modes

### Modes and step budgets

`examples/case_realistic_2d.jl` is the production card. Its physics is in guide section 1, its initial condition in guide section 1.7, and its seven modes (`smoke`, `short`, `spinup`, `freeze`, `wallrefine`, `resume`, `symseed`) and their step budgets are in guide section 6.1:

```bash
CASE_OUT="$HOME/quast/demo" julia --project=. --threads=8 \
  examples/case_realistic_2d.jl <mode> [checkpoint]
```

### The ten-step smoke test

The smoke test runs 10 steps with the full card:

```bash
CASE_OUT="$HOME/quast/demo_smoke" CASE_STEPS=10 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

The first run is dominated by compiling the driver; the [quick start](quickstart.md) gives the first-run expectation. The run must end with `[case] DONE mode=smoke steps=10` followed by

```text
[case] final: Nu_top=5.999881 Nu_bot=-4.029590 Vrms=6.808435e+03 meanT=1.333328 closure=-8.794e-04 leaves=2644 N=11233
```

The conductive reference for this base state is `Nu_top = 6`, so the top flux is right to **2e-5 relative** after ten steps. The bottom sits 0.74% below its `-4` reference because the bottom boundary layer is still forming.

### Spin-up, freeze and resume

The production spin-up of the same card is the first real run of the [quick start](quickstart.md): that command starts the tree from the same initial condition and grows it toward the 10 000-leaf cap, writing `spinup.h5` every 200 steps and again at exit. The case has no analytic reference, and after 2 000 steps it is still in the startup transient. Compare a short run against another run of the same card, not against the 20 000-step endpoint (`Nu_top = 18.2892`, `Nu_bot = -2.7636`), which is a different regime.

`resume` keeps adaptation on and is the quick start's stopping and resuming step. `freeze` rewrites only `amr_every` to 0 and continues on the frozen mesh:

```bash
CASE_OUT="$HOME/quast/frozen" CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl freeze \
  "$HOME/quast/run1/spinup.h5"
```

Give each continuation its own `CASE_OUT`, so its `frames/` and `history.csv` do not collide with the run it starts from. `CASE_STEPS` is an absolute step number, so a budget at or below the checkpoint step makes the driver run zero steps and write nothing. Asking `resume` to continue a frozen checkpoint fails the configuration guard, because that mode passes `amr_every = 5`; continue that branch with `freeze` on `frozen.h5`.

### Turning on viscoelasticity

Maxwell viscoelasticity is off by default, and every command on this page runs the purely viscous problem unless you add the card knob `CASE_SHEAR_MODULUS`, the nondimensional shear modulus $G$ (guide section 1.8). The ten-step smoke test is the quickest way to see it on:

```bash
CASE_OUT="$HOME/quast/demo_ve" CASE_STEPS=10 CASE_SHEAR_MODULUS=3.55e6 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

$G$ is nondimensional: it is scaled by the card's reference viscosity and time unit, so the same physical shear modulus is a different number when `Ra`, the reference viscosity or the scale set changes. The first header line of this run ends with `eta_ref=1 shear_modulus=3.55e+06`, and with `shear_modulus > 0` the driver adds the two Maxwell regime reports that guide section 1.8 describes, one at the first stress step and one at the end of the run. The smoke test writes no checkpoint, so the checkpoint cost of the stress history first appears on a spin-up.

## The Blankenbach 2b case

The 2b benchmark is a 2.5 x 1 box with free-slip walls, `Ra = 1e4`, no internal heating, and a viscosity that depends on both temperature and depth:

$$
\eta = \exp(-bT + c(1-z)),
\qquad
b = \ln 16384,
\qquad
c = \ln 64,
$$

where $T$ is temperature, $z$ is height, and $\eta$ is viscosity. The reference values are those of Blankenbach et al. (1989), Table 9, `Nu = 6.9299` and `Vrms = 171.755`; the script prints them next to its own result.

`examples/benchmark_cases.jl` is the entry point. It starts from ASPECT's own temperature field, which is already close to the steady state, so a run needs thousands of steps instead of tens of thousands:

```bash
julia --project=. --threads=8 examples/benchmark_cases.jl 2b 3 0.2
```

The arguments are `case level tstop`. Level 3 refines a 5 x 2 base grid three times, giving 40 x 16 square leaves and 640 leaves in total at `p = 2`; the `:auto` selector resolves to the direct solver at this size (guide section 5.3). The run ends with

```text
[2b] steps=5492 t=2.0001e-01 wall=<s> s  Nu_top=7.248805 Nu_bot=7.250974 Vrms=170.213063  leaves=640
```

The two wall fluxes agree to 0.03%, so the state is steady. Agreement between the top and bottom fluxes is the acceptance test: with no internal heating, a steady state cannot have them differ. The second printed line gives both differences from the published values; the level-by-level convergence record is in [validation](validation.md).

`Nu` is a one-sided gradient inside the boundary layer, and uniform refinement spends its cells in the interior where nothing depends on them. The wall rule of guide section 4.3 is what fixes that. Turn it on with

```bash
CASE_WALLMIN=5 CASE_INITREF=1 julia --project=. --threads=8 examples/benchmark_cases.jl 2b 4 0.2
```

`CASE_WALLMIN` and `CASE_INITREF` are knobs of `benchmark_cases.jl`, not of the realistic case. `CASE_WALLMIN` sets the minimum level for the wall strip and the wall rows, and `CASE_INITREF=1` runs one adaptation pass on the initial condition. The measured effect of the wall rule is part of the convergence table in [validation](validation.md).

## The 2-D annulus benchmark

`examples/annulus_benchmark.jl` is the 2-D cylindrical-annulus convection benchmark of Davies et al. (2022) and the ASPECT documentation: isoviscous Boussinesq at `Ra = 1e5`, free slip on both circles, `R0 = 1.22` and `R1 = 2.22` (the gap is the length unit), no internal heating, the paper's Eq. 41 initial condition, and the closed periodic ring. [validation](validation.md) carries the published values, the tolerances and the resolution record; this section is the command.

The published resolutions are `128x1536` (ASPECT-like) and `512x2048` (Firedrake-like), and the first is the largest this machine can run (the second needs about 66–72 GB at the measured memory per leaf). Reaching a steady state is done with the harness's documented relaxation mode: a large fixed step makes the time derivative negligible, so every energy solve is the steady advection-diffusion problem and the step sequence is a damped Picard iteration for the coupled steady state. The `32x384` rung, whose measured driver wall clock is 193 s (about four minutes end to end with start-up and compilation):

```bash
ANN_CFL=0 ANN_DT0=1e3 ANN_MAXPIC=25 ANN_PIC_RELAX=0.5 ANN_TMIN=0 \
ANN_WINDOW=3 ANN_TSTOP=1e12 ANN_CKPT_DIR=out_julia/ann_ckpt \
julia --project=. --threads=8 examples/annulus_benchmark.jl 32x384
```

The two lines to check are the result and its deviation from the published numbers:

```text
ANNULUS_RESULT L5 nx=32 ny=384 ... steps=6 ... NuT=9.899941 NuB=9.943853 Vrms=193.25842 meanT=0.382464 ... steady=1 ...
ANNULUS_DEV NuT_vs_aspect=3.6860% ... Vrms_vs_aspect=-0.0582% ... dNu_rel=-0.4426%
```

`steady=1` means the stated criterion fired (relative `theta`, `Nu_T` and `V_RMS` changes below `1e-6` for three consecutive outer steps), and the numbers to compare are `Nu_T = 9.89994`, `Nu_B = 9.94385`, `V_RMS = 193.258`, `<T> = 0.382464`. The next rungs are `64x768` (measured driver wall clock 424 s) and `128x1536` (1 391 s, 13.7 GB peak RSS, `Nu_T = 9.56743`, +0.20 % from ASPECT); a killed run resumes from `out_julia/ann_ckpt/` by rerunning the same command. Without the `ANN_*` settings the script runs the time-accurate `CFL = 1` mode, which is the physical transient and needs thousands of steps at these resolutions; the header of the script documents every knob and the steady-state criterion.

## The blob regression

The blob is the `Ra = 1e8` regression that pins the adaptive path. The box is 1 x 1, the walls are free slip, the heating is `H = 5`, the viscosity is the same two-exponent law as 2b, and the initial condition is the conductive base state plus one Gaussian:

$$
\theta_0(x,z) = 0.12\exp\left[-\left(\frac{x-0.5}{0.12}\right)^2-\left(\frac{z-0.35}{0.10}\right)^2\right]\sin(\pi z),
$$

where $\theta\_0$ is the initial temperature anomaly and $x$ and $z$ are the horizontal and vertical coordinates.

```bash
julia --project=. --threads=8 examples/run_blob_kelly_1024.jl
```

The script runs with `p = 4` on the direct solver, adapts every 5 steps, keeps a 1 024-leaf hard cap with a largest level of 6, and writes 256 x 256 frames every `4e-10` up to `t = 1e-8`. It clears its output directory at start when that directory holds nothing but its own output, so point `BLOB_OUT` elsewhere if you want a different path. The run ends at 898 leaves and 25 frames, 126 steps. On this machine, eight threads finish in about two minutes including Julia startup. The last line is

```text
FINAL frames=25 end_theta_sha256=a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda end_u_sha256=1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51 Tmax25=1.264329738
```

`end_theta_sha256` is the bitwise pin and `Tmax25` is the scalar to compare. The digest holds bitwise only on the pinned Julia, BLAS and thread stack; `Tmax25` is the check that tolerates a different machine. The comparison of this run against ASPECT is in [validation](validation.md).

## A regional channel

`examples/channel_flow_2d.jl` is the combined boundary-condition demonstration: a plane channel whose upstream wall prescribes the normal velocity of the inflow, whose downstream wall is `:open` with the exact total traction, and whose top and bottom are no-slip. Guide section 2 defines the wall types, the data arrays and their sign conventions; this section composes them into the regional recipe.

The card has this shape, in the wall order `(xminus, xplus, yminus, yplus)`:

| Wall | `wall_bc` | Data |
|---|---|---|
| Upstream x-wall | `:rigid` | `wall_un` = the normal velocity $u\_x(y)$ of the inflow profile; `wall_u` = the tangential velocity (zero for a normal inflow) |
| Downstream x-wall | `:open` | `wall_tn` = $-p\_{\mathrm{out}}$ (the dynamic pressure of the outflow) and `wall_t` = $\eta\,\mathrm du\_x/\mathrm dy$ (the shear traction) |
| Top and bottom | `:rigid` or `:free` | Nothing; `:rigid` is no slip and `:free` is free slip |

The inflow wall must be `:rigid`: a `:free` wall has zero tangential traction and cannot carry the shear $\eta\,\mathrm du\_x/\mathrm dy$ that a shear-carrying profile has at its entrance. Prescribing the same normal velocity on a `:free` wall is a genuinely different flow, measured at 5.6% of the centreline velocity; the recipe therefore always uses `:rigid` on the inflow, with `wall_u` carrying any tangential part. The velocity of a flow entering through a normal-velocity boundary is prescribed completely by `wall_un` and `wall_u`, and the interior sees the same Dirichlet data whichever of the two mechanisms writes the value: the per-wall `wall_un` route and the legacy global closure produce bit-identical solutions.

Three compatibility conditions make the combination well posed:

- **Mass.** Continuity fixes the flux. The integral of the inflow data is the flux the outflow must carry, and the discrete continuity equations enforce the balance. The net-flux guard of guide section 2.6 cannot double as the check here: with an open wall an inlet whose integral does not match is not an error, because its imbalance moves the outflow velocity. The open wall's free normal velocity absorbs it, and that is the regional model's outflow.
- **Pressure level.** The open wall removes the nodal pressure pin (guide section 2.4), so the pressure level is not fixed by a pin but by the normal traction data: `wall_tn` is the dynamic pressure of the outflow, $p\_{\mathrm{out}}$, with the sign of the outward normal. It is a dynamic pressure, not a lithostatic one.
- **Well-posedness.** At least one x-wall and one y-wall keep a Dirichlet normal component; the channel satisfies this with the rigid inflow, the two channel walls and the open outflow.

The card is a plain `Params`, run through `run_semquad_simulation` for one step:

```bash
julia --project=. --threads=8 examples/channel_flow_2d.jl
```

The channel is 2 x 1, the viscosity is constant, and the flow is driven by a pressure gradient `dp/dx = -G` with `G = 2`; the driver's body force is the buoyancy and has no horizontal component, so the pressure-driven mode is the one the card uses. With the open outflow held at dynamic pressure `p_out = 2`, the exact solution is plane Poiseuille,

$$
u_x = \frac{G\,y\,(H-y)}{2\eta},
\qquad
p = p_{\text{out}} + G\,(L-x),
$$

so the card carries `wall_un = y(1-y)`, `wall_t = 1 - 2y` (the shear traction $\eta\,\mathrm du/\mathrm dy$) and `wall_tn = -2` (the outflow dynamic pressure with the outward normal). The run ends with

```text
[channel] steps=1 leaves=128 err_u=1.832e-14 err_uz=3.289e-14 err_p=4.567e-13 max|u|=0.250000
```

`err_u` is the maximum profile error divided by the centreline velocity `G H^2/(8 eta) = 0.25`, and `err_uz` uses the same scale; `err_p` is absolute. Q2 velocity and Q1 pressure contain this solution exactly, so the few `1e-14` are the linear solve, not the discretization. The simulation itself is 0.04 s once the driver is compiled; the first-call compile in a fresh process is the process cost quoted in the introduction.

The same analytic solution also exists with a constant body force `f_x = G` and no pressure gradient, in which case the pressure is constant and `wall_tn` is $-p\_0$; the shipped script uses the pressure-driven form.

If the outflow tangential traction is omitted (`wall_t = nothing`), the open wall is a free-slip outlet. The Poiseuille profile is then **not** the solution, because its outflow shear $G(H/2-y)$ violates the zero-tangential-traction condition. The free-slip solution keeps the same inflow data and mass flux and differs from Poiseuille by exactly the response to the withheld shear traction: on this channel that difference is 18% of the centreline velocity, its strain energy is 2.4% of the derived energy scale $L\,\lVert\eta u'\rVert^2/(2\eta^2)$, and the difference is exactly linear in the withheld-traction scale.

The production use of the recipe is the regional subduction-style case below.

## A regional subduction-style case

`examples/subduction_2d.jl` is the regional subduction-style case built from the regional-channel recipe above. A 2 x 1 box (2 000 x 1 000 km) of asthenosphere enters through the far-field wall at `x = 0`, crosses under the overriding plate and leaves through the open trench-side wall at `x = 2`, while the cold surface grows the plate-cooling boundary layer as the material moves trenchward. It is a kinematic thermal model: `Ra = 0`, so the thermal and phase buoyancy are off and the two phase transitions act through their latent heat only. The inflow wall (`xminus`) is `:rigid` with `wall_un` = the reference-channel normal velocity, the top (`yplus`) is `:rigid` with `wall_u = U_p` for the overriding plate, the base (`yminus`) is no slip, and the outflow (`xplus`) is `:open` with `wall_tn = -p_out` and the far-field shear traction.

The five modes are:

| Mode | What it runs | Measured wall clock |
|---|---|---|
| `validate` | The isoviscous (`deta_T = 1`) analytic channel on the 512-leaf tree, `linear = :direct` | 15 s |
| `rheology` | The `deta_T = 1e3` plate-cooling stage on the production tree, 240 steps | 409 s |
| `phases` | The same stage with the 410-km and 660-km transitions and their latent heat, 240 steps | 463 s |
| `full` | The composite diffusion–dislocation law with frames and the `nu_cbf`/`extdiag` columns, 120 steps | 154 s |
| `converge` | The resolution sequence on the uniform level-3, uniform level-4 and production trees | 149 s at `CASE_STEPS=60` (220 s including the first-call compile); the full 200-step sequence is the manual run |

The wall clocks are the time each mode itself reports, on eight threads; they exclude the compile that a fresh process pays before the mode starts, about another minute.

```bash
julia --project=. --threads=8 examples/subduction_2d.jl validate
julia --project=. --threads=8 examples/subduction_2d.jl rheology
julia --project=. --threads=8 examples/subduction_2d.jl phases
julia --project=. --threads=8 examples/subduction_2d.jl full
julia --project=. --threads=8 examples/subduction_2d.jl converge
```

`validate` is the analytic stage: constant viscosity, and the exact solution is the Couette–Poiseuille flow between the moving plate and the no-slip base,

$$
u_x = u_{\text{ref}}(y) = U_p\,y + G\,y(1-y)/2, \qquad u_y = 0,
\qquad p = p_{\text{out}} + G\,(L_x-x),
$$

with `U_p = 1.6e3`, `G = 1e3`, `p_out = 1e3`, `L_x = 2` and constant viscosity 1. The upstream wall carries the normal velocity `u_ref(y)`, the plate carries `U_p`, and the open outflow carries the far-field total traction: `wall_tn = -p_out` and `wall_t = tau_ref(y) = U_p + G(1/2-y)`. Q2 velocity and Q1 pressure contain that solution exactly, so the printed errors are the direct linear solve, not the discretization. The mode ends with

```text
[regional] check  1 u-field        max|ux-uref|/Up   measured=1.4211e-14 ... PASS
[regional] check  2 p-field        max|p-pref|/pscale measured=1.2738e-12 ... PASS
[regional] check  3 flux-net       |sum Q|/Qref      measured=9.0091e-16 ... PASS
[regional] validate: 13 checks (13 gated): 13 PASS, 0 FAIL, 0 DIAG, 0 SKIP
[regional] validate: PASS
```

`rheology` adds the $10^3$ temperature-dependent viscosity and integrates the thermal field to the developed plate-cooling state. It ends with `Nu_top = 31.383`, `Nu_bot = 0.212`, `max|u| = 1.600e3 = U_p`, and the two stations report `delta(0.8) = 0.0522` (half-space 0.0519) and `delta(1.6) = 0.0739` (0.0734) with the surface anomaly flux 24.28 and 16.55 against 25.25 and 17.87: those are the numbers to compare after a rerun. `phases` adds the 410-km and 660-km transitions with their latent heat and prints the two coefficients (measured `-0.051193` and `+0.105288`); `full` runs the composite law and ends with `Nu_top = 31.374`, `Nu_bot = 0.283` and `max|u| = 1.6001e3`; `converge` is the resolution sequence and gates the production-versus-uniform-4 agreement once every arm reaches steady state. Each mode prints one line per check and writes `checks.csv` under `$CASE_OUT/<mode>/`; `validate`, `rheology`, `phases` and `full` also write `history.csv` there, while `converge` writes one history per arm under its own subdirectory.

The limitations are part of the case. There is no dipping slab interface and no free surface: the trench side is a vertical open wall, the slab sits outside the box, and the case models the incoming plate with the mantle channel beneath it. There is also no compositional field and no self-consistent slab pull, and `Ra = 0` makes the velocity prescribed at the walls, so the numbers are not directly comparable with published kinematic-subduction models.

## A programmatic run

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

**Next:** [Guide](guide.md) — the equations, the discretization and every knob behind these runs.
