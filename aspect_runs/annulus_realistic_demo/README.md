# The realistic card on the closed ring (ASPECT pilot)

The ASPECT side of the realistic-card-on-the-annulus demonstration: the card of
`aspect_runs/realistic/verify7000.prm` (the ASPECT arm of
`docs/assets/quast_vs_aspect_realistic.gif`) ported to the closed 2-D ring.  This
directory is the **pilot** record; the final clip card (end time, frame cadence,
output directory) is decided after the pilot.

## The card

| Setting | Value |
|---|---|
| Geometry | `spherical shell`, inner radius `1.22*d = 3538000 m`, outer `2.22*d = 6438000 m`, `Opening angle = 360`; the gap `d = 2.9e6 m` is the length unit |
| Gravity | `radial constant`, magnitude `9.81 m/s^2` |
| Boundaries | free slip on both circles (`Tangential velocity boundary indicators = 0,1`), fixed temperatures `1500 K` inner / `500 K` outer (`spherical constant`), `Remove nullspace = net rotation` |
| Buoyancy / heating | Boussinesq, `H_r = 1.189061e-12 W/kg`, the two phase transitions with latent heat — verbatim from `verify7000.prm` |
| Material model | the tracked `realistic composite` plugin (`../realistic/realistic_composite.cc`): ASPECT's `diffusion dislocation` + the two phase transitions |
| Initial temperature | a `function` expression in dimensional units implementing the annular H-inclusive conductive base plus `0.05 sin(pi h) cos(2 phi) + 0.02 sin(2 pi h) cos(4 phi)`, `h = r/d - 1.22`, `phi = atan2(z,x)` (the same total temperature the QuaST card's perturbation is defined against) |
| AMR | **the original GIF card's block verbatim**: `Initial global refinement = 5`, `Initial adaptive refinement = 2`, `Time steps between mesh refinement = 5`, `Strategy = temperature`, `Refinement fraction = 0.2`, `Coarsening fraction = 0.005`, `Minimum refinement level = 5`.  Not edited; the pilot measures what it does on the shell |
| Time | CFL `0.5`, `Maximum time step = 4.2e14 s`, solver tolerances `1e-7` (Stokes) / `1e-9` (temperature) — verbatim from `verify7000.prm` |
| Pilot window | `End time = 8.41e13 s` = `t_nd = 1e-5`, VTU every `2.6912e12 s` (the original fine cadence, `t_nd = 3.2e-7`) |

The only changes from `verify7000.prm` are the six allowed ones: geometry,
gravity, the two velocity/temperature boundary cards, the initial temperature,
the output/cadence, and the `Nullspace removal` subsection the shell needs.
Everything else (material, heating, phases, formulation, solver tolerances,
CFL, maximum time step) is verbatim.

## Plugin build (measured)

The pinned image is `gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac`
(ASPECT 2.5.0-pre, deal.II 9.4.0).  Its build tree `CMAKE_BUILD_TYPE` is
`DebugRelease`, so `ASPECT_SETUP_PLUGIN` builds **both** variants from the same
`cmake` configure — there is no debug/release mismatch to work around:

```powershell
$img = "gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac"
# stage the two tracked files at an ASCII host path (the CJK repo path cannot be a docker mount):
#   copy ..\realistic\realistic_composite.cc and ..\realistic\CMakeLists.txt to C:/temp/quast_aspect/realistic_annulus
docker run --rm --entrypoint /bin/bash -v C:/temp/quast_aspect/realistic_annulus:/work -w /work $img `
  -c "mkdir -p build && cd build && cmake -DAspect_DIR=/home/dealii/aspect/build .. && make -j8"
```

Result: `build/librealistic_composite.debug.so` and
`build/librealistic_composite.release.so`; the cards load the **release** one
and run with `/home/dealii/aspect/aspect-release`.  (The DEBUG binary was not
used; the repository's measured DEBUG/OPTIMIZED factor on the Blankenbach card
is 17.3x, `benchmarks/results/aspc-blankenbach2b_nu-vrms_ref3.md`.)  This
pilot's plugin build is archived in `out_julia/annulus_realistic/plugin_build/`.

## Run it (Windows PowerShell, ASCII mount)

```powershell
$img = "gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac"
docker run --rm --entrypoint /bin/bash -v C:/temp/quast_aspect/realistic_annulus:/work -w /work $img `
  -c "mpirun --allow-run-as-root -n 4 /home/dealii/aspect/aspect-release annulus_realistic_pilot.prm > logs/pilot.log 2>&1"
```

`annulus_realistic_probe.prm` is the one-step parameter/mesh probe used before
the pilot.  Output: `output-annulus-realistic-pilot/{statistics,solution/}`;
the archived log/statistics/derived trajectory are under
`out_julia/annulus_realistic/aspect_pilot/`.

## Measured pilot (2026-11-26)

| Item | Measured |
|---|---|
| **Probe** | Image parses the function IC (`atan2`, `ln` are muparser built-ins), loads the plugin, builds the shell: `12,288` cells after `Initial global refinement = 5`, `15,480` and `16,236` after the two initial adaptive refinements (levels 0-7); one Stokes solve `18 s` total |
| **Initial condition check** | outward heat flux through the two circle boundaries sums to `3.56753e5 W` = `H_r rho0` x area (internal heat production), i.e. the function IC is the exact H-inclusive annular conduction state; `T` ranges `500..2323.9 K`, area mean `1740.78 K`, `u_rms = 1.839e-9 m/s` at t = 0 |
| **Pilot run** | 4 MPI ranks, `aspect-release`, `End time = 8.41e13 s` (`t_nd = 1e-5`): **69 time steps**, **353 s** wall (5.04 s/step average; Stokes 157 s, postprocessing 48.5 s, temperature 43.5 s, Stokes preconditioner 39.4 s, Stokes assembly 35.4 s) |
| **In-loop AMR** | **15 refinement calls** = 2 initial + **13 in-loop** cycles (every 5 steps); the mesh grew monotonically `16,236 -> 101,994` cells (6.3x), all on 8 levels (max level 7) |
| **Stability** | **stable**: no spike, no solver failure, no warning; `u_rms` decays `5.333e3 -> 3.365e3` (nondim `u*d/kappa`), `u_max` `1.394e4 -> 8.48e3`, `T` min/avg/max stays `500 / 1740.78 / 2323.9 K`, Stokes cheap iterations 114 (t=0) then 30-47 |
| **Net rotation** | the measured angular-momentum integral of a sampled frame is `-3.14e6 m^3/s`, i.e. `omega = 1.3e-21 1/s` against a strain rate of order `1e-16 1/s`: the `net rotation` removal is active and the flow carries no visible rigid rotation, including across the in-loop cycles |
| **Cells cap** | ASPECT's `max_refinement_level` is `initial_global + initial_adaptive = 7` (it grows only through `Additional refinement times`, which the card leaves empty), so the temperature strategy's 20%-refine / 0.5%-coarsen ratchet approaches the **uniform level-7 shell mesh = 12*4^7 = 196,608 cells**, not an unbounded mesh.  The first pilot ended at 101,994 cells, still growing |
| **Second pilot (projection)** | `End time = 2.523e14 s` (`t_nd = 3e-5`), VTU every `2.6912e13 s`: **166 steps, 1428 s at 4 ranks (8.6 s/step), 16,236 -> 167,160 cells, 35 in-loop refinement calls**, same stable behaviour (the flow decays to `u_rms_nd = 2.94e3` near `t_nd ~ 2e-5` and starts to spin up again, reaching `3.50e3` at the end; cheap Stokes iterations 40-50).  Timer: Stokes solve 45%, temperature assembly 15%, Stokes assembly 13%, preconditioner 13%, **postprocessing 5.7%** at this cadence (vs 14% at the GIF's fine cadence); the cost proxy drops `1.007e-4 -> 8.07e-5 s per cell-step` |
| **Cost projection** | the saturated level-7 mesh costs ~15.8 s/step at 4 ranks (coarse output).  The two pilots bracket the step count: 166 steps reached `t_nd = 3e-5` with the spin-up only starting, and the box GIF's ASPECT arm needed 2362 steps for the whole window, so the window `t_nd = 1e-6..3.2e-4` projects to **O(1800-2200) steps and ~6-9 h at 4 ranks**.  The output cadence is the free lever: write VTU every ~`2.6912e13 s` (~200 outputs for 120 rendered frames), not every step.  This is the pilot's main cost finding and the reason the clip window/ranks are a lead decision |

## What is NOT done here

- The full production clip (window `t_nd = 1e-6..3.2e-4`, ~2000 steps) was not
  run: the pilot's job was to measure the per-step cost, the AMR behaviour and
  the stability.
- No timing optimisations, no AMR-block edits, no solver workarounds: the one
  pilot ran with the card as specified.  (The only solver choices reported are
  the ones `verify7000.prm` itself has.)

## Licence and provenance

The plugin sources under `../realistic/` are GPL-2.0-or-later (see that
directory's `LICENSE` and the top-level `NOTICE`); the `*.prm` cards here are
original work under the repository's MIT licence.
