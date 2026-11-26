# aspc-davies-case23: ASPECT run of Davies et al. case 2.3 (variable viscosity) at refinement 4 — UNVERIFIED, the state is not steady at the card's end time

- **What was run.** ASPECT 2.5.0-pre running the shipped
  `benchmarks/davies_et_al/case-2.3.prm` card (free slip, nominal Ra = 1e3, viscosity
  `eta = 1000^(-T)` from the `case-2.3-plugin` VoT material model, `Reference thermal expansion
  coefficient = 1e-5`, gravity 1e8), resumed from the refinement-4 case-2.1 checkpoint at t = 2
  and advanced to the card's `End time = 3`, on the **refinement-4 mesh** (192 tangential x 16
  radial = 3 072 cells). The run **did reach its end time** (step 2056, t = 3.0) but it is **not
  at a steady state there**, so it is **not a completed comparison** and no number from it is
  quoted as a reproduced ASPECT value.
- **Measured, for the record.** Last statistics row (step 2056, t = 3.0):

  | quantity | value at t = 3 | published ASPECT ref-7 steady row | apparent difference | drift over the last 100 steps |
  |---|---:|---:|---:|---:|
  | `Nu_T` (outer wall) | 4.832801 | 5.102 | −5.28 % | `Q_top` −0.83 % |
  | `Nu_B` (inner wall) | 4.910300 | 5.121 | −4.11 % | — |
  | `V_rms` | 89.6160520 | 79.632 | +12.54 % | **+4.89 %** |
  | mean `T` | 0.500763028 | 0.582 | −13.96 % | +0.15 % |

  The differences from the published row are **not** a comparison: the published case-2.3 values
  are steady-state scalars at 1536 x 128, and this refinement-4 run still has `V_rms` rising by
  about 5 % per 100 timesteps at t = 3. The card's `End time = 3` is not long enough for the
  variable-viscosity transient to settle on this coarse mesh, and extending it was not affordable:
  the measured cost of the resumed leg is **2 990 s for 1 384 new timesteps = 2.16 s/step** with
  4 MPI ranks (worst of the four Davies cases — 219 Stokes iterations per step in the variable
  viscosity solve vs 3-50 in the isoviscous cases), and the timestep is shrinking as the flow
  accelerates (dt = 9.8e-5 at the last step). Reaching a steady state would need several more
  time units at that cost.
- **`Nu` conversion.** `Nu_T = -Q_top ln(f)/(2 pi)`, `Nu_B = -Q_B ln(f)/(2 pi)`, r_min = 1.22,
  r_max = 2.22, f = 0.55, k = 1, unit gap; measured `Q_top = +50.7225602`,
  `Q_bot = -51.5359554` at t = 3.
- **Command.**
  ```text
  docker run --rm --entrypoint /bin/bash -v C:/temp/quast_aspect:/work \
    -w /work/davies_et_al \
    gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac \
    -c "mpirun --allow-run-as-root -n 4 \
        /home/dealii/aspect/aspect-release d23_ref4_resume.prm > d23_ref4_resume.log 2>&1"
  ```
  The variant card `d23_ref4_resume.prm` (SHA-256
  `9542305cb0a965a8dbe87339c19b62ef71fd9caaecd8d2934c424a80515519a0`) is the shipped
  `case-2.3.prm` with only the output directory (a copy of the case-2.1 ref-4 output), the
  refinement (4), the VTU/depth-average intervals and the plugin flavour
  (`./case-2.3-plugin/libVoT.release.so`, since the image ships
  `libVoT.release.so`/`libVoT.debug.so` but not the bare `libVoT.so` the card names) changed.
- **Input card / fixture.** Image path `/home/dealii/aspect/benchmarks/davies_et_al/case-2.3.prm`,
  SHA-256 `6be3a6e699dc3afddc99e243abbabe52df16f48c9332b92881599b25c627efea`.
- **Resolution.** 3 072 cells on 5 levels; 41 280 degrees of freedom = 25 344 velocity + 3 264
  pressure + 12 672 temperature, Q2 velocity / Q1 pressure / Q2 temperature.
- **Solver path.** `aspect-release` (OPTIMIZED build), 4 MPI processes, ASPECT default Stokes
  solver (GMRES + AMG block preconditioner), `CFL = 10`.
- **Wall clock.** 3 857 s cumulative "including restarts" minus the 867 s recorded in the
  case-2.1 checkpoint = **2 990 s** for 1 384 new timesteps (2.16 s/step, 4 ranks).
- **Environment.** Docker Desktop engine from Windows PowerShell (WSL integration off); image
  `gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac`
  (tag `gassmoeller/aspect:latest` currently resolves to it), created 2023-04-19; ASPECT
  2.5.0-pre (main, 340c0d018), deal.II 9.4.0, Trilinos 13.2.0, p4est 2.2.0, linux/amd64;
  container view 16 CPUs, 25 GB RAM. Version output archived in `logs/aspect_environment.txt`.
- **Log.** Full log 902 969 bytes, SHA-256
  `1c86c4bd3299993d78fa46b90c5ee686bcb69d9f1fef1805316067d37e059055` (not shipped). Compact
  excerpt `logs/aspect_d23_ref4_compact.log` (34 148 bytes, SHA-256
  `6e5e18894ab3a33df35113b653ac755838b790a48d068ca743b51be908b5355c`). Statistics excerpt
  `logs/aspect_d23_ref4_statistics.txt` (11 029 bytes, SHA-256
  `1dc6caaceb107fbb606e786356149205935c6647e9f9316ed60c5963e5f03448`; full statistics
  793 207 bytes, SHA-256
  `680e17e0d7e95dbbb85bb60d07c6b0782dfe27e168dea0641e13d7193407d911`). The statistics file
  includes the inherited case-2.1 rows; the case-2.3 leg is rows 674-2057.
- **Status.** **`UNVERIFIED-in-full`**: the card ran to its end time but the state is not steady,
  so it is not a completed comparison; the numbers above are the measured t = 3 state, not
  reproduced published values. The shipped free-slip card at refinement 5 aborts in this image
  (see `aspc-davies-case21_nu-vrms_ref4.md`), so the published 1536 x 128 state was not attempted.
  There is also no same-parameter QuaST run (the repository's annulus ladder is isoviscous
  Ra = 1e5), so the variable-viscosity case has neither an independent reference nor a
  cross-code counterpart in this repository.
- **Recorded.** 2026-11-26 by the ASPECT cross-code batch, repository commit
  `a93246984d359d5989e683b07cb711ad23a35019`; run tree `C:\temp\quast_aspect` on the Windows host.
