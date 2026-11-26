# aspc-davies-case11: ASPECT `Nu_T`, `Nu_B`, `V_rms`, mean `T` for Davies et al. case 1.1 (rigid walls), shipped 5-refinement mesh

- **Quantity.** ASPECT 2.5.0-pre running the shipped
  `benchmarks/davies_et_al/case-1.1.prm` card at its shipped `Initial global refinement = 5`
  (12 288 cells; 384 tangential x 32 radial, the mesh the card's own footnote calls "5 global
  refinements"), to the card's `End time = 2`. Last statistics row (step 529, t = 2.0):

  | quantity | ASPECT run, ref 5 | published ASPECT table, ref 7 (1536 x 128) | relative difference |
  |---|---:|---:|---:|
  | `Nu_T` (outer wall) | **2.464699** | 2.464 | +0.0284 % |
  | `Nu_B` (inner wall) | **2.464702** | 2.468 | −0.1336 % |
  | `V_rms` | **19.051767** | 19.053 | −0.0065 % |
  | mean `T` | **0.402951114** | 0.403 | −0.000049 absolute (−0.012 %) |

  The published values are the ASPECT documentation table for the Davies et al. benchmark
  (shipped in the same image as `benchmarks/davies_et_al/doc/davies_et_al.md`, table
  "Numerical results for some of the output quantities required by the benchmarks"), which the
  docs page states was obtained at **7 global refinements = 1536 tangential x 128 radial**. The run
  here is at the shipped 5 refinements (4 x fewer cells in each direction), so this is a
  same-code, lower-resolution reproduction of the published ASPECT row, not a same-mesh
  comparison; the fact that all four scalars still land within 0.14 % is the useful content: this
  case is already converged on this mesh for these quantities.
- **`Nu` conversion.** ASPECT's `heat flux statistics` postprocessor writes the integrated outward
  heat flux through each boundary. For the annulus benchmark radii (r_min = 1.22, r_max = 2.22,
  f = 0.55, k = 1, and r_max(1-f) = r_min(1-f) = 1 for the unit gap) the documentation's formulas
  reduce to `Nu_T = -Q_top ln(f)/(2 pi)` and `Nu_B = -Q_B ln(f)/(2 pi)` with `Q_B` the flux entering
  the domain through the inner wall (the negative of ASPECT's outward value). The measured fluxes
  are `Q_top = +25.8681967`, `Q_bot = -25.8682304`, giving the two numbers in the table.
- **Command.**
  ```text
  docker run --rm --entrypoint /bin/bash -v C:/temp/quast_aspect:/work \
    -w /work/davies_et_al \
    gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac \
    -c "/home/dealii/aspect/aspect-release d11_ref5_run.prm > d11_ref5_run.log 2>&1"
  ```
  The variant card `d11_ref5_run.prm` (SHA-256
  `e02865cbfc71b1828be62d8a3cb87528cf2a7dbf478c8fa0d28e9c1529887193`) is:
  ```text
  include case-1.1.prm

  set Output directory = output-case11-ref5-release

  subsection Postprocess
    subsection Visualization
      set Time between graphical output = 1e6
    end
    subsection Depth average
      set Time between graphical output = 1e6
    end
  end
  ```
  Only the output location and the VTU/depth-average output intervals differ from the shipped
  card; mesh, boundary conditions (zero velocity on `top`/`bottom`), gravity, material, `End time`
  and CFL are the shipped values. The shipped card runs as-is: unlike the free-slip cards it has
  no solid-body-rotation nullspace (see `aspc-davies-case21_nu-vrms_ref4.md` for the free-slip
  abort).
- **Input card / fixture.** Image path `/home/dealii/aspect/benchmarks/davies_et_al/case-1.1.prm`,
  SHA-256 `a0d8d9f852bf2408931f3a714b13be0105ecd3531899c2825a26aa4941986dde`.
- **Resolution.** 12 288 cells on 6 levels; 112 512 Stokes DOFs (99 840 velocity + 12 672
  pressure) and 49 920 temperature DOFs, Q2 velocity / Q1 pressure / Q2 temperature. This equals
  the repository's `128 x 1536` annulus ladder cell count divided by 16 (the ladder rung with the
  same tangential spacing is `32 x 384` = 12 288 cells, but it is a Ra = 1e5 free-slip run, so no
  same-parameter QuaST comparison exists for case 1.1; the repository's recorded annulus ladder
  is at Ra = 1e5).
- **Solver path.** `aspect-release` (OPTIMIZED build shipped in the image, same commit as the
  default DEBUG `aspect`); ASPECT default Stokes solver (GMRES + AMG block preconditioner),
  `CFL = 10`.
- **Wall clock.** **445 s** for 530 timesteps = 0.84 s/step (ASPECT's own timer). Single MPI rank.
- **Environment.** Docker Desktop engine from Windows PowerShell (WSL integration off); image
  `gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac`
  (tag `gassmoeller/aspect:latest` currently resolves to it), created 2023-04-19; ASPECT
  2.5.0-pre (main, 340c0d018), deal.II 9.4.0, Trilinos 13.2.0, p4est 2.2.0, linux/amd64;
  container view 16 CPUs, 25 GB RAM. Version output archived in `logs/aspect_environment.txt`.
- **Log.** Full log 309 817 bytes, SHA-256
  `e640a9ba5cbf9382e5c6762b1099193ec7fd6791fdbbb0b6b2d8a330e75cdf41` (not shipped).
  Archived compact excerpt `logs/aspect_d11_ref5_compact.log` (16 077 bytes, SHA-256
  `1f5858b1995e434b9c688519c5ffc1185b3e891a4c435a5d70e10559c44ce145`). Statistics excerpt
  `logs/aspect_d11_ref5_statistics.txt` (5 219 bytes, SHA-256
  `7a555e9c6c24e4d824acf5f0977cbc4443453be5a17dff8062aab76d4d43ca61`; full statistics
  204 252 bytes, SHA-256
  `2bf08c8a3e97727eacceb033c4f4927a82460d8e2b5fb1348fe263d8aa660ea9`).
- **Status.** `verified` — the card runs to its end time and its four steady scalars reproduce the
  published ASPECT ref-7 row to 0.14 % or better at 4 x fewer cells per direction. Note the
  resolution difference explicitly: it is a lower-resolution reproduction, not the published mesh.
- **Recorded.** 2026-11-26 by the ASPECT cross-code batch, repository commit
  `a93246984d359d5989e683b07cb711ad23a35019`; run tree `C:\temp\quast_aspect` on the Windows host.
