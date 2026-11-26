# The realistic card on the closed ring (ASPECT arm)

The ASPECT side of `docs/assets/quast_vs_aspect_annulus_demo.gif`: the card of
`aspect_runs/realistic/verify7000.prm` (the ASPECT arm of
`docs/assets/quast_vs_aspect_realistic.gif`) ported to the closed 2-D annulus.
`annulus_realistic_prod.prm` is the production card of the clip; the pilot and
probe cards beside it are the record of the two pilot runs that measured the
stability and the cost before the production run.

## The card

| Setting | Value |
|---|---|
| Geometry | `spherical shell`, inner radius `1.22*d = 3538000 m`, outer `2.22*d = 6438000 m`, `Opening angle = 360`; the gap `d = 2.9e6 m` is the length unit |
| Gravity | `radial constant`, magnitude `9.81 m/s^2` |
| Boundaries | free slip on both circles (`Tangential velocity boundary indicators = 0,1`), fixed temperatures `1500 K` inner / `500 K` outer (`spherical constant`), `Remove nullspace = net rotation` |
| Buoyancy / heating | Boussinesq, `H_r = 1.189061e-12 W/kg`, the two phase transitions with latent heat — verbatim from `verify7000.prm` |
| Material model | the tracked `realistic composite` plugin (`../realistic/realistic_composite.cc`): ASPECT's `diffusion dislocation` + the two phase transitions |
| Initial temperature | a `function` expression in dimensional units implementing the annular H-inclusive conductive base plus `0.05 sin(pi h) cos(2 phi) + 0.02 sin(2 pi h) cos(4 phi)`, `h = r/d - 1.22`, `phi = atan2(z,x)` (the same total temperature the QuaST card's perturbation is defined against) |
| AMR | **the original GIF card's block verbatim**: `Initial global refinement = 5`, `Initial adaptive refinement = 2`, `Time steps between mesh refinement = 5`, `Strategy = temperature`, `Refinement fraction = 0.2`, `Coarsening fraction = 0.005`, `Minimum refinement level = 5`.  Not edited |
| Time | CFL `0.5`, `Maximum time step = 4.2e14 s`, solver tolerances `1e-7` (Stokes) / `1e-9` (temperature) — verbatim from `verify7000.prm` |
| Production window | `End time = 2.6912e15 s` = `t_nd = 3.2e-4` (the box clip's window, starting from the card's own t = 0 state) |
| Output | VTU every `1.682e13 s` (161 graphical outputs over the window), ASPECT checkpoint every 100 steps |

The only changes from `verify7000.prm` are the six allowed ones: geometry,
gravity, the two velocity/temperature boundary cards, the initial temperature,
the output/cadence, and the `Nullspace removal` subsection the shell needs.
Everything else (material, heating, phases, formulation, solver tolerances,
CFL, maximum time step) is verbatim, and the AMR block is the original GIF's.

## Plugin build (measured)

The pinned image is `gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac`
(ASPECT 2.5.0-pre, deal.II 9.4.0).  Its build tree `CMAKE_BUILD_TYPE` is
`DebugRelease`, so `ASPECT_SETUP_PLUGIN` builds **both** variants from one
`cmake` configure — there is no debug/release mismatch:

```powershell
$img = "gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac"
docker run --rm --entrypoint /bin/bash -v C:/temp/quast_aspect/realistic_annulus:/work -w /work $img `
  -c "mkdir -p build && cd build && cmake -DAspect_DIR=/home/dealii/aspect/build .. && make -j8"
```

Result: `build/librealistic_composite.debug.so` and
`build/librealistic_composite.release.so`; the production card loads the
**release** one and runs with `/home/dealii/aspect/aspect-release`.  (The DEBUG
binary was not used; the repository's measured DEBUG/OPTIMIZED factor on the
Blankenbach card is 17.3x, `benchmarks/results/aspc-blankenbach2b_nu-vrms_ref3.md`.)

## Run the production card (Windows PowerShell, ASCII mounts)

Docker is on the Windows host and the repository path contains non-ASCII
characters, so both the work tree and the output are mounted from ASCII host
paths.  The output (~3 GB including the checkpoints) goes to the E: drive
because the C: drive does not hold it:

```powershell
$img = "gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac"
docker run -d --name quast_aspect_prod --entrypoint /bin/bash `
  -v C:/temp/quast_aspect/realistic_annulus:/work `
  -v E:/temp/quast_aspect_prod:/out -w /out $img `
  -c 'start=$(date +%s); mpirun --allow-run-as-root -n 8 /home/dealii/aspect/aspect-release /work/annulus_realistic_prod.prm > /out/logs/prod.log 2>&1; rc=$?; end=$(date +%s); echo "exit=$rc wall=$((end-start))s" >> /out/logs/prod.log'
```

Follow it with `docker ps`, `docker wait quast_aspect_prod` and the log under
`E:/temp/quast_aspect_prod/logs/prod.log`.  The rendered clip reads the output
from `/mnt/e/temp/quast_aspect_prod/output-annulus-realistic-prod` in WSL (see
`scripts/run_annulus_realistic_demo.sh`).  A run with the ASPECT checkpoint
files in that directory resumes when the card is restarted in that directory
(ASPECT rotates `restart.mesh_fixed.data` and its `.old` copy).

`annulus_realistic_probe.prm` is the one-step parameter/mesh probe used before
the pilots.  The pilot cards and their archived logs are the smaller
`End time` runs (pilot: `8.41e13 s`; the second pilot: `2.523e14 s`) that
measured the in-loop AMR stability and the per-step cost before the production
window was chosen.

## Measured production run (2026-11-27)

| Item | Measured |
|---|---|
| Steps / end | 2 666 time steps, `t = 2.6912e15 s` = `t_nd = 3.2e-4`, termination by the end-time criterion |
| Wall / ranks | **14 288 s = 3.97 h at 8 MPI ranks**, `aspect-release` (OPTIMIZED) |
| Mesh | `16 239` cells after the startup refinement (level 5 global + 2 adaptive passes); max **`77 229` cells**; **`67 866` cells at the end** (the temperature strategy ran the 0.5 % coarsening back down after the spin-up peak), levels 5-7 |
| In-loop AMR | `535` refine-mesh operations = 2 startup passes + 533 in-loop cycles at every 5 steps; the mesh is stable throughout (no spike, no solver failure, no warning) |
| Output | 161 graphical outputs (1 288 VTU files over 8 ranks), 2.94 GB total including 26 rotating checkpoint snapshots |
| Final state | RMS / max velocity `1.69e-10` / `1.19e-9 m/s` (`u_rms = 490` nondimensional); `T` min/avg/max `500 / 1 739 / 2 285 K` (`T_max = 1.785` nondimensional); boundary heat fluxes `9.738e4 W` / `2.477e5 W` |
| Net rotation | the shell has a rigid-rotation nullspace; `Remove nullspace = net rotation` is active for the whole run and the flow shows no visible rotation. The pilot measured the angular-momentum integral `-3.14e6 m^3/s`, i.e. `omega ~ 1e-21 1/s` against strain rates of order `1e-16 1/s` |
| Stability rule | the refinement ratchet is capped by `initial_global + initial_adaptive = 7` (the card leaves `Additional refinement times` empty), so the maximum mesh is the uniform level-7 shell (`12*4^7 = 196 608` cells); the run stayed below it |

Nothing in the AMR block, the solver tolerances or the formulation was changed:
the pilot measured this exact card stable on the shell, and so did the
production run.

## QuaST arm

The QuaST side is `examples/annulus_realistic_demo.jl` (direct solve, production
AMR settings, `nu_undefined = :nan` because the H = 10 ring's per-wall Nusselt
pair is undefined by construction).  The recorded run: 1 396 steps to
`t_nd = 3.200977e-4`, 3.26 h on 8 Julia threads, 10 000 leaves over levels 3-7,
218 adaptive cycles, measured `Ra_interior = 4.4218e7` (nominal `Ra = 1e5` at
the composite calibration point).  The frame stream is
`profile = :state` (total `T`, the six node arrays including `ux, uy`, the
composite `log10(eta/eta_ref)`, the mesh dump and the v3 meta table).

## Render

```bash
PY=/home/szh/dedalus-env/bin/python
python scripts/annulus_realistic_video.py \
    --quast out_julia/annulus_realistic/quast_prod \
    --aspect /mnt/e/temp/quast_aspect_prod/output-annulus-realistic-prod \
    --quast-log out_julia/annulus_realistic/quast_prod.log \
    --t0 1e-6 --t1 3.2e-4 --frames 120 --fps 10 \
    --out out_julia/annulus_realistic/render/realistic.gif \
    --also-poster --also-mp4 \
    --sidecar out_julia/annulus_realistic/render/realistic.json
```

`--check` prints the input tables before rendering (counts, time alignment,
level inference against the statistics, the ASPECT saved-viscosity cross-check,
and the shown-frame colour ranges).  The GIF is 1200 x 616 px, 120 frames,
7.94 MB with `--gif-dither none` (the default for this clip).

## Licence and provenance

The plugin sources under `../realistic/` are GPL-2.0-or-later (see that
directory's `LICENSE` and the top-level `NOTICE`); the `*.prm` cards here are
original work under the repository's MIT licence.
