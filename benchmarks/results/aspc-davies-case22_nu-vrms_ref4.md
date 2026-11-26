# aspc-davies-case22: ASPECT `Nu_T`, `Nu_B`, `V_rms`, mean `T` for Davies et al. case 2.2 (free slip, Ra = 1e5) at refinement 4

- **Quantity.** ASPECT 2.5.0-pre running the shipped `benchmarks/davies_et_al/case-2.2.prm`
  card, resumed from the refinement-4 case-2.1 checkpoint at t = 2 (the card's documented
  initial condition) and advanced to its `End time = 3`, on the **refinement-4 mesh**
  (192 tangential x 16 radial = 3 072 cells). Last statistics row (step 2226, t = 3.0):

  | quantity | ASPECT run, ref 4, t = 3 | published ASPECT table, ref 7 (1536 x 128) | relative difference | QuaST `16 x 192` rung (same 3 072 cells) |
  |---|---:|---:|---:|---:|
  | `Nu_T` (outer wall) | **9.511780** | 9.548 | −0.379 % | 10.316646 |
  | `Nu_B` (inner wall) | **9.500254** | 9.584 | −0.874 % | 10.268873 |
  | `V_rms` | **193.126704** | 193.371 | −0.126 % | 193.12706 |
  | mean `T` | **0.382437930** | 0.382 | +0.115 % | 0.382440 |

  The run is steady at t = 3: over the last 100 timesteps `V_rms` and mean `T` change by less than
  their printed precision (192.126704 -> 192.126704 on the printed digits; recorded values
  193.12670400 both), i.e. the card's end time is past the plateau, matching the docs figure.
  The published reference is the ASPECT documentation table (ref 7), shipped in the image; the
  same-row values of the Firedrake paper are `Nu = 9.541`, `V_RMS = 193.26` at 2048 x 512.
- **What the same-resolution cross-code comparison says.** At 3 072 cells on both sides the
  *state* agrees to round-off: `V_rms` differs by **−0.0002 %** and mean `T` by **−0.0005 %**
  between ASPECT and the repository's recorded `16 x 192` QuaST rung. The wall-flux functionals do
  not: `Nu_T` differs by **−7.80 %** and `Nu_B` by **−7.48 %**. That is the expected signature of
  the wall measurement, not of a different state: the QuaST number is its one-sided GLL wall
  gradient (family A), whose coarse-mesh error on this case is +8.05 % (the ladder record
  `annulus_nu-t_128x1536_rung4.md`), while ASPECT's `heat flux statistics` is the consistent
  boundary flux (family C), already within 0.4 % of the ref-7 value on this coarse mesh. The
  validation page states that pattern and this run measures it from the other code's side.
- **Our side, same card.** The QuaST run is the `16 x 192` ladder rung: `geom = :annulus`,
  `annulus_R0 = 1.22`, `annulus_R1 = 2.22`, `annulus_Phi = 2 pi`, `annulus_seam = :periodic`,
  `Ra = 1e5`, `deta_T = 1`, free slip, Davies et al. Eq. (41) initial condition, Q2-Q1-Q2,
  Galerkin advection, steady relaxation; recorded in `annulus_nu-t_128x1536_rung4.md` and the
  ladder log. The ASPECT card, by contrast, starts case 2.2 from the case-2.1 state (as the
  benchmark prescribes); both target the same steady scalars.
- **`Nu` conversion.** `Nu_T = -Q_top ln(f)/(2 pi)`, `Nu_B = -Q_B ln(f)/(2 pi)` with
  `Q_B` the flux entering through the inner wall (negative of ASPECT's outward value), for
  r_min = 1.22, r_max = 2.22, f = 0.55, k = 1 and unit gap. Measured `Q_top = +99.8306925`,
  `Q_bot = -99.7097162`.
- **Command.** Case 2.1's refinement-4 output directory was copied to
  `output-case22-ref4-release` first; the case-2.2 card then resumes from that copy's checkpoint.
  ```text
  docker run --rm --entrypoint /bin/bash -v C:/temp/quast_aspect:/work \
    -w /work/davies_et_al \
    gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac \
    -c "mpirun --allow-run-as-root -n 4 \
        /home/dealii/aspect/aspect-release d22_ref4_resume.prm > d22_ref4_resume.log 2>&1"
  ```
  The variant card `d22_ref4_resume.prm` (SHA-256
  `451b0179b78222f29912fe60f70b47d6e273e38a1cf56a613b0c55c429deb36e`) is:
  ```text
  include case-2.2.prm

  set Output directory = output-case22-ref4-release
  set End time         = 3

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
  Relative to the shipped card: output location, VTU/depth-average intervals and the global
  refinement (the resume mesh is the checkpoint's anyway) only; physics, gravity (Ra = 1e5) and
  end time are the card's. The refinement change is the same measured limitation as in
  `aspc-davies-case21_nu-vrms_ref4.md`: the shipped free-slip card at 5 refinements aborts in
  timestep 1 in this pinned image.
- **Input card / fixture.** Image path `/home/dealii/aspect/benchmarks/davies_et_al/case-2.2.prm`,
  SHA-256 `cc19697b0f355b62789eb994c386550a2f29e50e9c26fda8cc54b24b2a51b574`.
- **Resolution.** 3 072 cells on 5 levels; 41 280 degrees of freedom = 25 344 velocity + 3 264
  pressure + 12 672 temperature, Q2 velocity / Q1 pressure / Q2 temperature.
- **Solver path.** `aspect-release` (OPTIMIZED build), 4 MPI processes, ASPECT default Stokes
  solver (GMRES + AMG block preconditioner), `CFL = 10`.
- **Wall clock.** The resumed leg ran 1 554 new timesteps in **344 s** (ASPECT timer:
  1 211 s cumulative "including restarts" minus the 867 s recorded in the case-2.1 checkpoint) =
  0.22 s/step with 4 ranks. The whole case (2.1 + 2.2) therefore cost 867 + 344 s of OPTIMIZED
  time on 1 + 4 ranks. The DEBUG binary is about 17 x slower per step (measured on Blankenbach 2b).
- **Environment.** Docker Desktop engine from Windows PowerShell (WSL integration off); image
  `gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac`
  (tag `gassmoeller/aspect:latest` currently resolves to it), created 2023-04-19; ASPECT
  2.5.0-pre (main, 340c0d018), deal.II 9.4.0, Trilinos 13.2.0, p4est 2.2.0, linux/amd64;
  container view 16 CPUs, 25 GB RAM. Version output archived in `logs/aspect_environment.txt`.
- **Log.** Full log 943 801 bytes, SHA-256
  `16c458b1b17e215bc7ed98da52a53ceb32d253312b788aa9b4bdc554d4d1ce30` (not shipped). Compact
  excerpt `logs/aspect_d22_ref4_compact.log` (37 237 bytes, SHA-256
  `1adc89ea274e36583db7ac7158c1b9db05036324a64dbf4e8fbf9336b5e2a89d`). Statistics excerpt
  `logs/aspect_d22_ref4_statistics.txt` (11 801 bytes, SHA-256
  `d067724a124450438ba63af8ac00f5616f94bd1e3853f85623313791111271fd`; full statistics
  858 657 bytes, SHA-256
  `c6811196df2ee13166cf439a61f7f15ad2f23598e1b0be68592e433421ea99db`). The statistics file
  includes the inherited case-2.1 rows; the case-2.2 leg is rows 674-2227.
- **Status.** `verified` — the card runs to its end time, is steady, and reproduces the published
  ref-7 row to 0.87 % on `Nu` and 0.13 % on `V_rms` from a 16 x coarser mesh; the same-resolution
  cross-code state comparison with QuaST is a measured 2e-6 agreement.
- **Recorded.** 2026-11-26 by the ASPECT cross-code batch, repository commit
  `a93246984d359d5989e683b07cb711ad23a35019`; run tree `C:\temp\quast_aspect` on the Windows host.
