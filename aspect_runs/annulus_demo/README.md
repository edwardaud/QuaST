# Annulus AMR demonstration (QuaST vs ASPECT)

The ASPECT side of `docs/assets/quast_vs_aspect_annulus_demo.gif`: a
purpose-configured **demonstration** of the adaptive path on the closed 2-D
annulus, run by QuaST and by ASPECT with their own adaptive meshes. It is not a
validation and not a published benchmark; no accuracy claim is made.

## The card

| Setting | Value |
|---|---|
| Geometry | spherical shell / periodic ring, `Inner radius = 1.22`, `Outer radius = 2.22`, `Opening angle = 360`; the gap `d = 1` is the length unit |
| Boundaries | free slip on both circles (`Tangential velocity boundary indicators = 0,1`), fixed temperatures `T = 1` inner / `T = 0` outer |
| Buoyancy | Boussinesq, `rho = cp = k = 1`, radial constant gravity `1e9`, thermal expansion `1e-6`, reference viscosity 1, so **Ra = 1e3** at the reference viscosity; the hot inner material has `eta = 1e-3` and therefore local Ra = 1e6 |
| Rheology | the image's `VoT` material model: **eta = 1000^(-T)** |
| Heating | none (H = 0) |
| Initial condition | `spherical hexagonal perturbation`, angular mode 4, rotation offset 0: `s = (R1-r)/(R1-R0)`, `T = s + 0.2 s (1-s) cos(4 atan2(y,x))` (the same total T the QuaST example uses through `ANN_IC=aspect`) |
| Time | `t = 0 -> 0.2`, CFL 10, BDF2, VTU every `0.005` (60 visualisation outputs) |
| Mesh | `Strategy = temperature`, `Initial global refinement = 3`, `Initial adaptive refinement = 2`, `Refinement fraction = 0.2`, `Coarsening fraction = 0.05`; the adaptive startup mesh (levels 3-4) is held fixed during the time loop -- see "Measured deviations" |

The QuaST side of the same demonstration is the `demo` preset of
`examples/annulus_benchmark.jl`, driven by `scripts/run_annulus_demo.sh`; its
card and measured statistics are in the figure's JSON sidecar
(`out_julia/annulus_demo/quast_vs_aspect_annulus_demo.json`).

## The pinned image and the plugin build decision

All runs use the image the repository pins everywhere:

```text
gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac
```

which is ASPECT 2.5.0-pre (main, 340c0d018), deal.II 9.4.0, and ships two builds
(`aspect` DEBUG and `aspect-release` OPTIMIZED) plus the benchmark plugin
sources under `/home/dealii/aspect/benchmarks/`. This run used
**`aspect-release`** (the DEBUG build is ~17x slower per step, measured on the
Blankenbach card in `../README.md`).

**Build decision (stated):** `demo.prm` loads the **prebuilt plugin the image
ships**:

```text
/home/dealii/aspect/benchmarks/davies_et_al/case-2.3-plugin/libVoT.release.so
```

The source is tracked here as `VoT.cc` (the shipped file, with an added
provenance header) and `CMakeLists.txt` (the shipped build file, with an added
provenance header) so the binary can be rebuilt from this repository and so
the exact law (`eta = Viscosity * 1000^(-T)`) is auditable. To rebuild it inside
the image (from a host directory mounted at `/work`, as the run itself is):

```bash
docker run --rm --entrypoint /bin/bash \
  -v C:/temp/quast_aspect/annulus_demo:/work -w /work \
  gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac \
  -c "mkdir -p build && cd build && \
      cmake -DAspect_DIR=/home/dealii/aspect/build .. && make -j4"
```

The result is `build/libVoT.so`; point `Additional shared libraries` at it
instead of the image's `libVoT.release.so`.

## Run it

Docker Desktop's WSL integration is off in this environment, so `docker` is run
from **Windows PowerShell** and the inputs/outputs live on an ASCII path
(`C:/temp/quast_aspect/annulus_demo`), because the mount path cannot carry the
CJK repository name. Copy `demo.prm` there, then:

```powershell
docker run --rm --entrypoint /bin/bash `
  -v C:/temp/quast_aspect/annulus_demo:/work -w /work `
  gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac `
  -c "mpirun --allow-run-as-root -n 4 /home/dealii/aspect/aspect-release demo.prm"
```

The output directory is `/work/output-annulus-demo`: `solution/` holds the VTU
pieces (`solution-NNNNN.PPPP.vtu`, one piece per MPI rank) and `statistics` the
per-timestep ASPECT statistics table. Copy `solution/` to
`out_julia/annulus_demo/aspect_frames/solution/` and `statistics` to
`out_julia/annulus_demo/aspect_frames/statistics`, then render with
`scripts/run_annulus_demo.sh` (or its `scripts/annulus_video.py --amr`
command).

## Measured cost (recorded)

| Item | Measured |
|---|---|
| Ranks | 4 MPI (`mpirun -n 4`, `aspect-release`); 1 rank was not needed -- the run is stable at 4 |
| Time steps to `t = 0.3` | 356 (`*** Timestep 353: t = 0.3`) |
| Wall clock | **227 s** (ASPECT timer, including startup and the 60 VTU writes) |
| Mesh | 1176 cells, levels 3-4, held fixed after the initial adaptive refinement |
| Frames | 60 visualisation outputs (240 VTU pieces) |
| Stability | stable to `t = 0.3` (see "Measured deviations" for the in-loop AMR instability that fixed the mesh) |

## Measured deviations from the first proposed card

The figure's caption and JSON sidecar state these; this section records the
measurements.

1. **Ra = 1e3, not 1e6.** With `eta = 1000^(-T)` and the reference viscosity 1,
   the hot inner boundary has local `Ra = 1e6` already at `Ra_ref = 1e3`. The
   first proposed card (`gravity 1e12`, i.e. `Ra_ref = 1e6`, local Ra = 1e9)
   was run as a pilot: ASPECT measured `u_max = 6.73e4` at the initial Stokes
   solve and `2.19e5` after one step with a CFL-limited `dt = 1.76e-6`; at that
   rate `t = 0.3` needs O(1e5-1e6) steps, and the temperature solve failed
   (1000 iterations, residual 4.8e-3). `Ra_ref = 1e4` was also measured
   (`u_max 5.5e4`, `dt 2.6e-6`) and rejected for the same reason.
2. **H = 0, not 10.** Internal heating is not needed for the demonstration, and
   at `H = 10` on the annulus it is refused by QuaST's own wall-flux
   normalisation: the base conductive flux changes sign at the inner wall
   (`-dTbar/dr = -4.31` at `R0` against `+5.38` at `R1`), so the family-A
   Nusselt normalisation is undefined.
3. **In-loop mesh refinement off.** In this pinned pre-release ASPECT image, an
   in-loop adaptive cycle every 5 steps drove a reproducible unbounded velocity
   spike (u_max 5.4e6 at t = 0.016) and a temperature-solver failure, with
   several solver settings (cheap Stokes steps 0, full A block preconditioner,
   GMRES restart 100, tolerance 1e-6/1e-7) and AMR settings (refinement
   fraction 0.1/0.2, coarsening 0/0.05). A direct Stokes solver is not an
   option for this card because it is incompatible with the net-rotation
   nullspace removal. The ASPECT mesh is therefore the two-pass adaptive
   startup mesh, held fixed; QuaST refines in-loop (different solver, no such
   instability).
4. **Window `t = 0 -> 0.2`, not `0.3`.** The archived ASPECT run used `End time = 0.3`; the tracked card sets `End time = 0.2` and the figure uses its first 0.2 (41 frames). The QuaST leg must fit the measurement
   budget; the frame cadence stays at the card's `0.005` (41 frames). Ra = 1e3
   was the largest reference Ra that fits the budget with a visibly evolving
   trajectory.

## Licence and provenance

`VoT.cc` and `CMakeLists.txt` are the ASPECT benchmark plugin files, a work
derived from ASPECT, distributed under **GPL-2.0-or-later**: each carries the
original GNU GPL header, an added provenance note and an SPDX line, and the
licence text is in `LICENSE` beside them. `demo.prm` and this README are
original work under the repository's MIT licence; the per-directory split is
also recorded in the top-level `../NOTICE`.
