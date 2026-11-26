# aspc-davies-case21: ASPECT `Nu_T`, `Nu_B`, `V_rms`, mean `T` for Davies et al. case 2.1 (free slip, Ra = 1e4) at refinement 4

- **Quantity.** ASPECT 2.5.0-pre running the shipped
  `benchmarks/davies_et_al/case-2.1.prm` card (free slip, Ra = 1e4, isoviscous) with only its
  global refinement changed from the shipped 5 to **4** (192 tangential x 16 radial = 3 072 cells),
  to `End time = 2`. Last statistics row (step 672, t = 2.0):

  | quantity | ASPECT run, ref 4 | published ASPECT table, ref 7 (1536 x 128) | relative difference |
  |---|---:|---:|---:|
  | `Nu_T` (outer wall) | **4.699036** | 4.7000 | −0.0205 % |
  | `Nu_B` (inner wall) | **4.699070** | 4.706 | −0.1473 % |
  | `V_rms` | **46.2422496** | 46.244 | −0.0038 % |
  | mean `T` | **0.382056192** | 0.382 | +0.0147 % |

  The published reference is the ASPECT documentation table for the Davies et al. benchmark, which
  the page says was produced at 7 global refinements (1536 x 128), shipped in the image as
  `benchmarks/davies_et_al/doc/davies_et_al.md`. The run here is at the shipped card's structure
  but one refinement lower (16 x fewer cells), so it is a lower-resolution reproduction of the
  published ASPECT row. It reproduces it to 0.15 % or better on all four scalars, and the run's own
  top/bottom wall-flux split is 0.0007 % against ASPECT's published 0.13 % at ref 7.
- **`Nu` conversion.** ASPECT's `heat flux statistics` writes the integrated outward heat flux
  through each boundary. For r_min = 1.22, r_max = 2.22, f = 0.55, k = 1 and a unit gap
  (r_max(1-f) = r_min(1-f) = 1) the documentation formulas reduce to
  `Nu_T = -Q_top ln(f)/(2 pi)` and `Nu_B = -Q_B ln(f)/(2 pi)`, with `Q_B` the flux entering through
  the inner wall (the negative of ASPECT's outward value). Measured: `Q_top = +49.3186340`,
  `Q_bot = -49.3189890`.
- **Command.**
  ```text
  docker run --rm --entrypoint /bin/bash -v C:/temp/quast_aspect:/work \
    -w /work/davies_et_al \
    gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac \
    -c "/home/dealii/aspect/aspect-release d21_ref4_full.prm > d21_ref4_full.log 2>&1"
  ```
  The variant card `d21_ref4_full.prm` (SHA-256
  `978c01d61b6a46366085619e115a1ea606ddf76c873cfac1769c6fabe7cad622`) is:
  ```text
  include case-2.1.prm

  set Output directory = output-case21-ref4-release
  set End time         = 2

  subsection Mesh refinement
    set Initial global refinement = 4
  end

  subsection Postprocess
    subsection Visualization
      set Time between graphical output = 1e6
    end
    subsection Depth average
      set Time between graphical output = 1e6
    end
  end
  ```
  Only the output location, the VTU/depth-average intervals and the global refinement differ from
  the shipped card; the physics, boundary conditions and `End time` are the card's.
- **Why refinement 4 and not the shipped 5 — a measured upstream limitation.**
  The shipped card at `Initial global refinement = 5` (384 x 32, 12 288 cells) **aborts in timestep 1
  in this pinned image**: the block-Schur preconditioner's velocity solve stops with
  `AztecOO::Iterate error code -3: loss of precision` inside
  `BlockSchurPreconditioner::vmult` ("The iterative (top left) solver ... did not converge").  The
  abort was reproduced with 1 and with 4 MPI ranks, and with both the OPTIMIZED and the DEBUG
  binary; it persists with `Remove nullspace` switched off and with a looser linear-solver
  tolerance. It is not the card and not the MPI or build flavour but the mesh size: refinement 4
  runs, and the same card at refinement 5 also runs when the cheap Stokes solver is disabled
  (`Number of cheap Stokes solver steps = 0`, measured 5.06 s/step, 13 timesteps to t = 0.02).
  The published 7-refinement mesh (1536 x 128) was therefore not attempted in this batch; any
  ASPECT number at the published mesh is not reproduced here. The abort logs are archived:
  `logs/aspect_d21_ref5_freeslip_abort_compact.log` (excerpt of the 4-rank abort, 10 844
  bytes, SHA-256 `405291594a4f5d355bd8a68b76ed21bca1a767e60700ec393c35d3d672f22e1b`) and
  `logs/aspect_d21_ref5_cheap0_workaround_compact.log` (7 547 bytes, SHA-256
  `9150ac14ee33d00952bc8e49793e2eccdb3fcd886d591f743e4632ecb5f844cd`). Full abort logs and their
  SHA-256 are listed below.
- **Input card / fixture.** Image path `/home/dealii/aspect/benchmarks/davies_et_al/case-2.1.prm`,
  SHA-256 `8230f4ee8d2fdeb83c60fdc4b36a833568441f81b4af3b1e1592a19fa89ffff6`.
- **Resolution.** 3 072 cells on 5 levels; 41 280 degrees of freedom = 25 344 velocity + 3 264
  pressure + 12 672 temperature, Q2 velocity / Q1 pressure / Q2 temperature. This is the same cell
  count as the repository's `16 x 192` annulus ladder rung, so the same-resolution cross-code
  comparison is against a `16 x 192` QuaST run (see the "our side" paragraph below).
- **Solver path.** `aspect-release` (OPTIMIZED build shipped in the image), 1 MPI process, ASPECT
  default Stokes solver (GMRES + AMG block preconditioner), `CFL = 10`.
- **Wall clock.** **867 s** for 673 timesteps = 1.29 s/step (ASPECT's own timer). The DEBUG binary
  on the same card is about 17 x slower per step (measured on Blankenbach 2b; see
  `aspc-blankenbach2b_nu-vrms_ref3.md`).
- **Environment.** Docker Desktop engine from Windows PowerShell (WSL integration off); image
  `gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac`
  (tag `gassmoeller/aspect:latest` currently resolves to it), created 2023-04-19; ASPECT
  2.5.0-pre (main, 340c0d018), deal.II 9.4.0, Trilinos 13.2.0, p4est 2.2.0, linux/amd64;
  container view 16 CPUs, 25 GB RAM. Version output archived in `logs/aspect_environment.txt`.
- **Log.** Full log 414 034 bytes, SHA-256
  `2af25e357603331687c009b591f88c3c0266eb07a24e356d8c516376eb8e0036` (not shipped). Compact
  excerpt `logs/aspect_d21_ref4_compact.log` (19 277 bytes, SHA-256
  `47ece715bfe4de9d217cf08690beb21f4ccd0f316b80ad9f8c8313caaba81381`). Statistics excerpt
  `logs/aspect_d21_ref4_statistics.txt` (5 592 bytes, SHA-256
  `7a8625e2c61b05f9e912b813c2021d14c2d87728919db5ae0cd88d204ab15082`; full statistics
  258 348 bytes, SHA-256
  `ed634059627a725b309b294119edcafb861a8702e713f23e9c9a9da7d6b6f996`). Abort and workaround
  logs: `d21_ref5_release.log` (4 MPI ranks, 10 527 bytes, SHA-256
  `1ae185e3f1c9bb356b8b1f21a778fca17e00fd5369a2db0994bbc04e3e44bfdb`),
  `d21_probe_1rank.log` (1 rank, 10 525 bytes, SHA-256
  `7955f4e3f64ff44261d3ce351fe3547ba363f9fa5d48acb368bd2668c6d35ae8`),
  `d21_dbg_probe.log` (DEBUG binary, 10 690 bytes, SHA-256
  `727395f285cef6cae3b478f972394edf194563167061a752bb0d3cd9985595c6`),
  `diag_nsoff.log` (10 489 bytes, SHA-256
  `e7618592cf7017b106ebe5bc25051926edf4aa354e385250e7a692cea1e28b67`),
  `diag_lintol.log` (10 493 bytes, SHA-256
  `46da81ea801c38d12196643c182af2f9cfbb657490334f13d150fe18804e8ee0`),
  `diag_ref4.log` (6 107 bytes, SHA-256
  `455e221d7424181d0f73b1f870bba3bc8d7f4623ff12dc0ee24ee2356962c272`),
  `diag_cheap0.log` (11 457 bytes, SHA-256
  `c5b6501071bb6fd8ed9cab063c711eab4abca8c3feb845a4100bcf707cc55338`). They live in the
  campaign tree `C:\temp\quast_aspect\davies_et_al` and are not shipped.
- **Our side, same resolution.** A `16 x 192` QuaST run of the same card (`geom = :annulus`,
  `annulus_R0 = 1.22`, `annulus_R1 = 2.22`, `annulus_Phi = 2 pi`, `annulus_seam = :periodic`,
  `Ra = 1e4`, `deta_T = 1`, free slip, Davies et al. Eq. (41) initial condition) reaches its stated
  steady-state criterion (`steady=1`, drifts 6.8e-7 / 1.5e-7) with `Nu_T = 4.862962`,
  `Nu_B = 4.874556`, `V_rms = 46.24225`, mean `T` = 0.382056. Against this ASPECT ref-4 run at the
  same cell count the state agrees to 1e-5 relative or better (`V_rms` +0.000001 %, mean `T`
  −0.00005 %) while `Nu_T` differs by +3.49 %; that is the wall-gradient (family A) versus
  consistent-boundary-flux (family C) measurement difference discussed on the validation page, not a
  different state. The run script is archived at `logs/annulus_ra1e4_16x192_script.jl` (one line
  different from `examples/annulus_benchmark.jl`: `Ra = 1.0e4`; SHA-256
  `9276ebd9985cefe6c4c497fde9f04f0577221e585d80097698d65dc65af75945`) and its log at
  `logs/annulus_ra1e4_16x192.log` (1 143 bytes, SHA-256
  `51aee915451e73580a85ed6e41ac1bdff8aac63dce1b4df96edeb7e6c86aac98`).
- **Status.** `verified` for the refinement-4 reproduction. `UNVERIFIED-in-full` for the shipped
  refinement-5 card and the published refinement-7 mesh: the shipped card aborts in this pinned
  image (measured, log archived), and the published mesh was not attempted.
- **Recorded.** 2026-11-26 by the ASPECT cross-code batch, repository commit
  `a93246984d359d5989e683b07cb711ad23a35019`; run tree `C:\temp\quast_aspect` on the Windows host.
