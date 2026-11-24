<p align="right"><a href="../en/examples.md">English</a> · <a href="../zh/examples.md">中文</a></p>

# Examples

Six runs you can start immediately. Each comes with its command, the physics it computes, the measured runtime on eight threads, and a number to check against. Run everything from the repository root. The first command is needed once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The timings are wall clock on eight Julia threads, not CPU time. The direct solver is serial inside the process, so a run on that path leaves seven threads idle. A first run in a fresh Julia process is dominated by compiling the driver (about 70 s, the repository-wide cost of the first `run_semquad_simulation` call); the simulations themselves are short.

| Example | Command | Wall clock, eight threads | What to check |
|---|---|---|---|
| Blankenbach 2b, steady | `benchmark_cases.jl 2b 3 0.2` | about 45 minutes | `Nu_top = Nu_bot` |
| Blob | `run_blob_kelly_1024.jl` | about two minutes | `Tmax25` and the two digests |
| Realistic case, smoke | `case_realistic_2d.jl smoke` | about two minutes the first time | `Nu_top = 5.999881` |
| Regional channel | `channel_flow_2d.jl` | about 70 s the first time, under a second of simulation | profile error below `1e-10` |
| Regional subduction-style case | `subduction_2d.jl rheology` | about 8 min the first time | `Nu_top = 31.383` and `delta(0.8) = 0.0522` |
| Programmatic run | `build_and_run.jl` | about 84 s the first time | `Nu_top = Nu_bot = 1.000783` |

## Blankenbach 2b

The 2b benchmark is a 2.5 x 1 box with free-slip walls, `Ra = 1e4`, no internal heating, and a viscosity that depends on both temperature and depth:

$$
\eta = \exp(-bT + c(1-z)),
\qquad
b = \ln 16384,
\qquad
c = \ln 64,
$$

where $T$ is temperature, $z$ is height, and $\eta$ is viscosity. The reference values are those of Blankenbach et al. (1989), Table 9; the script prints them next to its own result.

### Running to a steady state

`examples/benchmark_cases.jl` is the entry point. It starts from ASPECT's own temperature field, which is already close to the steady state, so a run needs hundreds or thousands of steps instead of tens of thousands:

```bash
julia --project=. --threads=8 examples/benchmark_cases.jl 2b 3 0.2
```

The arguments are `case level tstop`. Level 3 refines a 5 x 2 base grid three times, giving 40 x 16 square leaves at `p = 2`; the `:auto` selector resolves to the direct solver at this size (guide section 4.3). The run ends with

```text
[2b] steps=5492 t=2.0001e-01 wall=<s> s  Nu_top=7.248805 Nu_bot=7.250974 Vrms=170.213063  leaves=640
```

The two wall fluxes agree to 0.03%, so the state is steady. Agreement between the top and bottom fluxes is the acceptance test: with no internal heating, a steady state cannot have them differ. The second printed line gives both differences from the published values; the level-by-level convergence record is in [validation](validation.md).

### Why the wall needs its own refinement

`Nu` is a one-sided gradient inside the boundary layer, and uniform refinement spends its cells in the interior where nothing depends on them. The wall rule of guide section 3.3 is what fixes that. Turn it on with

```bash
CASE_WALLMIN=5 CASE_INITREF=1 julia --project=. --threads=8 examples/benchmark_cases.jl 2b 4 0.2
```

`CASE_WALLMIN` and `CASE_INITREF` are knobs of `benchmark_cases.jl`, not of the realistic case. `CASE_WALLMIN` sets the minimum level for the wall strip and the wall rows, and `CASE_INITREF=1` runs one adaptation pass on the initial condition. The measured effect of the wall rule is part of the convergence table in [validation](validation.md).

## The blob regression

The blob is the `Ra = 1e8` regression that pins the adaptive path. The box is 1 x 1, the walls are free slip, the heating is `H = 5`, the viscosity is the same two-exponent law, and the initial condition is the conductive base state plus one Gaussian:

$$
\theta_0(x,z) = 0.12\exp\left[-\left(\frac{x-0.5}{0.12}\right)^2-\left(\frac{z-0.35}{0.10}\right)^2\right]\sin(\pi z),
$$

where $\theta\_0$ is the initial temperature anomaly and $x$ and $z$ are the horizontal and vertical coordinates.

```bash
julia --project=. --threads=8 examples/run_blob_kelly_1024.jl
```

The script runs with `p = 4` on the direct solver, adapts every 5 steps, keeps a 1 024-leaf hard cap with a largest level of 6, and writes 256 x 256 frames every `4e-10` up to `t = 1e-8`. It clears its output directory at start when that directory holds nothing but its own output, so point `BLOB_OUT` elsewhere if you want a different path. The run ends at 898 leaves and 25 frames. On this machine, eight threads finish the 126 steps in about two minutes including Julia startup.

The last two lines are

```text
done t=1.008e-08 steps=126 leaves=898 N=15361 frames=25 in 141.3 s
FINAL frames=25 end_theta_sha256=a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda end_u_sha256=1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51 Tmax25=1.264329738
```

`end_theta_sha256` is the bitwise pin and `Tmax25` is the scalar to compare. The digest holds bitwise only on the pinned Julia, BLAS and thread stack; `Tmax25` is the check that tolerates a different machine. The comparison of this run against ASPECT is in [validation](validation.md).

## The realistic case

This is the production card; its physics is in guide section 1, and its initial condition is in section 1.7. The smoke test runs 10 steps with the full case:

```bash
CASE_OUT="$HOME/quast/demo_smoke" CASE_STEPS=10 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

The first run takes about two minutes, mostly Julia compilation and the first Stokes solve; on a warm machine it is 47.6 s. It must end with `[case] DONE mode=smoke steps=10` followed by

```text
[case] final: Nu_top=5.999881 Nu_bot=-4.029590 Vrms=6.808435e+03 meanT=1.333328 closure=-8.794e-04 leaves=2644 N=11233
```

The conductive reference for this base state is `Nu_top = 6`, so the top flux is right to **2e-5 relative** after ten steps. The bottom sits 0.74% below its `-4` reference because the bottom boundary layer is still forming.

The production spin-up of the same card is the first real run of the [quick start](quickstart.md): that command starts the tree from the same initial condition. The tree grows from 640 leaves to the 10 000-leaf cap within the first 100 steps, and `spinup.h5` is written every 200 steps and again at exit. The case has no analytic reference, and after 2 000 steps it is still in the startup transient. Compare a short run against another run of the same card, not against the 20 000-step endpoint (`Nu_top = 18.2892`, `Nu_bot = -2.7636`), which is a different regime.

`resume` and `freeze` are two ways to continue the same checkpoint. `resume` keeps adaptation on and is the quick start's stopping and resuming step; `freeze` rewrites only `amr_every` to 0 and continues on the frozen mesh:

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

### Freezing the mesh

`resume` and `freeze` continue from the same checkpoint. `resume` keeps the adaptive mesh;
`freeze` sets `amr_every = 0` and continues on the frozen tree:

```bash
CASE_OUT=out_julia/demo_frozen CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl freeze \
  out_julia/demo/spinup.h5

CASE_OUT=out_julia/demo_resume CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

Both carry the state from step 2 000 to step 4 000. Give each its own `CASE_OUT`: they write
their own `frames/` and `history.csv`. `CASE_STEPS` is an absolute step number, so a budget
at or below the checkpoint step runs nothing.

## A regional channel

`examples/channel_flow_2d.jl` is the combined boundary-condition demonstration: a plane channel whose upstream wall prescribes the normal velocity of the inflow, whose downstream wall is `:open` with the exact total traction, and whose top and bottom are no-slip. The card is a plain `Params`, run through `run_semquad_simulation` for one step:

```bash
julia --project=. --threads=8 examples/channel_flow_2d.jl
```

The channel is 2 x 1, the viscosity is constant, and the flow is driven by a pressure gradient `dp/dx = -G` with `G = 2`; the driver has no horizontal body force. With the open outflow held at dynamic pressure `p_out = 2`, the exact solution is plane Poiseuille,

$$
u_x = \frac{G\,y\,(H-y)}{2\eta},
\qquad
p = p_{\text{out}} + G\,(L-x),
$$

so the card carries `wall_un = y(1-y)`, `wall_t = 1 - 2y` (the shear traction `eta du/dy`) and `wall_tn = -2` (the outflow dynamic pressure with the outward normal). Guide section 1.10 is the recipe. The run ends with

```text
[channel] steps=1 leaves=128 err_u=1.832e-14 err_uz=3.289e-14 err_p=4.567e-13 max|u|=0.250000
```

`err_u` is the maximum profile error divided by the centreline velocity `G H^2/(8 eta) = 0.25`, and `err_uz` uses the same scale; `err_p` is absolute. Q2 velocity and Q1 pressure contain this solution exactly, so the few `1e-14` are the linear solve, not the discretization. The run itself is 0.04 s once the driver is compiled; a fresh process spends about 70 s on that first-call compilation, and `julia -O0 --project=. examples/channel_flow_2d.jl` cuts the start-up to about 55 s without changing the answer at this accuracy.

The channel's gate, G264, also runs the same solution driven by a constant body force (`p` then constant, with `wall_tn = -p0`), and the free-slip outflow variant (`wall_t = nothing`): that variant is a different, bounded flow, because the exact Poiseuille outflow carries a non-zero shear, so the free-slip solution differs from Poiseuille by exactly the response to the withheld traction (18% of the centreline velocity here). The gate exercises this script in-process; the section is not documentation-only.

## A regional subduction-style case

`examples/subduction_2d.jl` is the production use of the regional-channel recipe of the [guide](guide.md), section 1.10. A 2 x 1 box (2 000 x 1 000 km) of asthenosphere enters through the far-field wall at `x = 0`, crosses under the overriding plate and leaves through the open trench-side wall at `x = 2`, while the cold surface grows the plate-cooling boundary layer as the material moves trenchward. It is a kinematic thermal model: `Ra = 0`, so the thermal and phase buoyancy are off and the two phase transitions act through their latent heat only.

The five modes are the staged plan of `docs/dev/REGIONAL_DESIGN.md`:

```bash
julia --project=. --threads=8 examples/subduction_2d.jl validate
julia --project=. --threads=8 examples/subduction_2d.jl rheology
julia --project=. --threads=8 examples/subduction_2d.jl phases
julia --project=. --threads=8 examples/subduction_2d.jl full
julia --project=. --threads=8 examples/subduction_2d.jl converge
```

`validate` is the analytic stage: constant viscosity, and the exact solution is the Couette-Poiseuille flow between the moving plate and the no-slip base,

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

`rheology` adds the 10^3 temperature-dependent viscosity and integrates the thermal field to the developed plate-cooling state. On this machine it ends with `Nu_top = 31.383`, `Nu_bot = 0.212`, `max|u| = 1.600e3 = U_p`, and the two stations report `delta(0.8) = 0.0522` (half-space 0.0519) and `delta(1.6) = 0.0739` (0.0734) with the surface anomaly flux 24.28 and 16.55 against 25.25 and 17.87: those are the numbers to compare after a rerun. `phases` adds the 410-km and 660-km transitions with their latent heat and prints the two coefficients (measured `-0.051193` and `+0.105288`); `full` runs the composite diffusion-dislocation law with frames and the extra `nu_cbf`/`extdiag` columns; `converge` is the resolution sequence of the design's section 6.6. Every mode prints one line per check and writes `checks.csv` (mode, check id, quantity, measured value, tolerance, PASS/FAIL/DIAG/SKIP) and `history.csv` under `out_julia/regional_channel/<mode>/`.

The check classes are explicit. Exact: the stage-(i) fields, the wall fluxes and the `deta_T = 1` continuation (1e-10, the linear solve). Conservation: the per-wall mass flux and its net-zero identity. Asymptotic: the plate-cooling profile against the half-space solution `T = T_inf erf((1-y)/(2 sqrt(x/U_p)))`, with 10 % in temperature and boundary-layer thickness and 15 % in the surface flux. The open-box heat budget (design section 6.3) and the latent-heat enthalpy identity (section 6.7, check 10) are reported as diagnostics, not gates; the reason is in the limitations below.

The honest limitations are the design's sections 7 and 9. There is no dipping slab interface: the trench side is a vertical open wall, the slab is outside the box, and the case models the incoming plate plus the mantle channel beneath it, not the wedge. There is no free surface, no topography, no compositional field and no self-consistent slab pull. `Ra = 0` makes the velocity prescribed at the walls and the thermal evolution kinematic, so the phase transitions contribute only latent heat. The side-wall temperature is a scalar, so a depth-dependent slab-top profile cannot be imposed and a published slab-top temperature curve is not a target for this field. For those reasons the case is not comparable to van Keken-class published wedge-flow numbers; it is the flat-slab limit of the corner flow, which is exactly Couette-Poiseuille.

Two further implementation facts a reader should know. The wall data are the reference channel's far-field *kinematics*, so the outflow tangential traction is `wall_t = tau_ref(y) * eta_far(y)/eta_ref` with `eta_far` the far-field viscosity at the card's hot side-wall temperature; the design's literal isoviscous *stress* (`CASE_WALL_T=stress`) was measured to trap an outflow shear layer of `max|u| ~ 1.7e5` against the 10^3 contrast, and is not the default. And the open-box heat budget built from the pointwise wall gradient does not close for a card with fixed side walls: the side-wall exchange is a Dirichlet reaction, not a resolved one-sided derivative, and the library's consistent boundary flux exists for the top and bottom walls only (`docs/dev/VALIDATION.md`, section 4.1). Both facts are recorded with the measured numbers in `docs/dev/REGIONAL_PROGRESS.md`.

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
