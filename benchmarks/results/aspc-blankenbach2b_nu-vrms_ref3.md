# aspc-blankenbach2b: ASPECT `Nu` and `Vrms` on the shipped Blankenbach 2b card, refinement 3

- **Quantity.** ASPECT 2.5.0-pre running the shipped
  `benchmarks/blankenbach/base_case2b.prm` card at its shipped `Initial global refinement = 3`
  (40 x 16 = 640 cells, the same cell count as the repository's level-3 2b rung) to the card's own
  steady-state termination criterion. Last statistics row (step 4076, t = 0.74438):

  | quantity | ASPECT value | shipped ASPECT ref-3 value (`readme.txt`) | relative difference |
  |---|---:|---:|---:|
  | `Nu` (top flux density, boundary id 3) | **6.44512825** | 6.44510878 | +0.00030 % |
  | `Vrms` | **168.575518** | 168.574818 | +0.00042 % |
  | `max|u|` | 816.712711 | — | — |
  | mean `T` | 0.430922931 | — | — |

  The agreement with the image's own shipped ASPECT ref-3 record (six significant figures) is the
  reproduction check: the pinned image reproduces the benchmark record it ships. The run also
  reproduces the shipped step count exactly: 4076 timesteps (the `readme.txt` lists
  "steps needed: 4076 output-case2b_ref3/statistics").
- **Comparison with the published value and with this repository.**
  Published Blankenbach et al. (1989, Table 9) reference, shipped beside the card as
  `case2b_reference.stat`: `Nu = 6.9299`, `Vrms = 171.755`.

  | source | `Nu` | vs published | `Vrms` | vs published |
  |---|---:|---:|---:|---:|
  | ASPECT run, ref 3 (640 cells, this record) | 6.445128 | −6.995 % | 168.5755 | −1.851 % |
  | ASPECT shipped value, ref 4 (2 560 cells) | 6.94481365 | +0.215 % | 169.976109 | −1.035 % |
  | ASPECT shipped value, ref 5 (10 240 cells) | 6.92988247 | −0.0003 % | 171.554254 | −0.117 % |
  | ASPECT shipped value, ref 6 (40 960 cells) | 6.92964921 | −0.004 % | 171.74538 | −0.006 % |
  | QuaST level 3 (640 leaves, wall gradient) | 7.248805 | +4.60 % | 170.213 | −0.90 % |
  | QuaST level 3 + walls 4 (1 120 leaves) | 7.212885 | +4.08 % | 170.633 | −0.65 % |
  | QuaST level 4 + walls 5 (4 000 leaves) | 6.992620 | +0.905 % | 171.634 | −0.070 % |

  Same-resolution cross-code difference (both codes on 40 x 16 cells, Q2-Q1-Q2, identical card
  physics): `Nu` 7.248805 (QuaST) against 6.445128 (ASPECT) = **+12.47 %**; `Vrms` 170.213 against
  168.5755 = **+0.97 %**.  Both codes are far from the published value at that mesh, with the
  wall-gradient `Nu` error of opposite sign: QuaST reads high by 4.6 %, ASPECT's consistent
  boundary-flux read is low by 7.0 %.  The velocity, a large-scale quantity, already agrees to
  about 1 %.  This is the honest content of the comparison: at the shipped coarse mesh neither code
  is converged on `Nu`, so a 12 % same-mesh difference is a discretisation difference, not a
  physical disagreement.
- **Where the QuaST 2b values come from.** The three QuaST rows above are the public table in
  `docs/en/validation.md` (section "Steady benchmarks"), whose 2b rows record the earlier
  wall-refined campaign (`Nu_top` = 7.248805 / 7.212885 / 6.992620 at level 3, level 3 + walls 4
  and level 4 + walls 5); the detailed run record is in `docs/dev/VALIDATION.md` §2.1, and the
  level-4 + walls-5 run was stopped by the host's out-of-memory killer before its closing summary,
  as that page states.  There is deliberately **no** `benchmarks/results/blankenbach2b_*.md`
  record in this tree: the QuaST 2b rows predate the provenance-record batch, and this ASPECT
  record does not invent one.  The published 2b reference is Blankenbach et al. (1989) Table 9,
  shipped in the image as `case2b_reference.stat` (6.9299 / 171.755).
- **Command.**
  ```text
  docker run --rm --entrypoint /bin/bash -v C:/temp/quast_aspect:/work \
    -w /work/blankenbach \
    gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac \
    -c "/home/dealii/aspect/aspect-release run2b_ref3_release.prm > run2b_ref3_release.log 2>&1"
  ```
  The variant card `run2b_ref3_release.prm` is a two-line overlay on the shipped card (SHA-256
  `13aa5c554236ff2537024640c3c7c204feb38839e3c63480f6ea2d018031781b`):
  ```text
  include base_case2b.prm
  set Additional shared libraries = ./plugin/libblankenbach.release.so
  set Output directory = output-2b-ref3-release
  set End time         = 1.0
  subsection Postprocess
    subsection Visualization
      set Time between graphical output = 1e6
    end
  end
  ```
  The only changes are the release plugin flavour (the image ships
  `plugin/libblankenbach.release.so` and `plugin/libblankenbach.debug.so`, but not the bare
  `plugin/libblankenbach.so` the shipped card names), the output location, and the VTU interval
  (raised to 1e6 s so the run does not write per-step VTU files). Physics, mesh, end time and the
  steady-state termination criterion are the shipped card's.
- **Input card / fixture.** Image path `/home/dealii/aspect/benchmarks/blankenbach/base_case2b.prm`,
  SHA-256 `0e5b108891a981ee7f09990166a28e18c2108cf5f813cb0304903c226c5ae04e`; initial temperature
  `initial_temperature_case2b.txt`, SHA-256
  `53879b76feb2bb88cb3d9992b6156619d5ebc60bf46f941d6cdd1967ca6eb2a1`; published reference
  `case2b_reference.stat`, SHA-256 `77f7dfe8173c9ee6d931b7e7abd49908fe1835b17e989d0db2365900627f2b94`.
- **Resolution.** 640 cells, 6 043 Stokes DOFs (velocity + pressure; the log's "Number of degrees
  of freedom" is not printed for this card, the statistics file's columns 5-6 give 6 043 Stokes and
  2 673 temperature DOFs), refinement 3 of the 5 x 2 base mesh.
- **Solver path.** The image ships two builds of the same commit: `aspect` (DEBUG) and
  `aspect-release` (OPTIMIZED). This comparison uses `aspect-release`; the DEBUG cost is measured
  separately below. Default ASPECT linear solver (GMRES + AMG block preconditioner), `CFL = 5`.
- **Wall clock.** **369 s** total for 4 077 Stokes solves = 0.0905 s/step (ASPECT's own timer
  table: 53 s assemble Stokes, 50.6 s assemble temperature, 73 s build Stokes preconditioner,
  59.8 s solve Stokes, 15 s solve temperature, 92.4 s postprocessing, all 4 077 calls).
  DEBUG-mode cost: a 40-timestep probe (`End time = 0.01`) took **64.1 s = 1.56 s/step** in
  ASPECT's timer, 17.3 x the OPTIMIZED step; extrapolated to the full 4 076-step card this is
  about **1.77 h**. A 240 s DEBUG cap reached timestep 164 (t = 0.03598, 4.0 % of the card's
  4 076 steps), consistent with the same rate. No full DEBUG run was attempted inside the batch's
  time box.
- **Environment.** Docker Desktop engine reached from Windows PowerShell (WSL integration off);
  image `gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac`
  (the tag `gassmoeller/aspect:latest` currently resolves to it), created 2023-04-19; ASPECT
  2.5.0-pre (main, 340c0d018), deal.II 9.4.0, Trilinos 13.2.0, p4est 2.2.0, linux/amd64; container
  view 16 CPUs, 25 GB RAM. Version output archived in `logs/aspect_environment.txt`.
- **Log.** Full run log 2 595 262 bytes, SHA-256
  `f465faa9140bbc9e5bfcb1b4b398a480f3a61e202df74a3114e9cebb25ae3003` (per-step lines; not
  shipped). Archived compact excerpt `logs/aspect_2b_ref3_compact.log` (23 078 bytes, SHA-256
  `344b95d5aba5f475d97d7d8ed27c6683c673953b09219a7dd8e4f70c1e946320`; preamble, one timestep
  block in 500, the last five blocks, the termination notice and the timer table). Archived
  statistics excerpt `logs/aspect_2b_ref3_statistics.txt` (9 765 bytes, SHA-256
  `7434703514075762a2332df4691cb9262efaa906eecf3d7b9d8753031a247e26`; column header, one row in
  200, last five rows; the full 1 383 254-byte statistics file has SHA-256
  `36ee71a6ae1b02eba839e6a9782d1474dc15e153f2ea1c9d83618b32e8fdfd5e`). DEBUG cost probe:
  `logs/aspect_2b_ref3_debug_cost_compact.log` (9 030 bytes, SHA-256
  `4e457248a23fcbca3e9a9143cdad0f663ab4778737c51bfc4c554bf21997712f`).
- **Status.** `verified` for the OPTIMIZED run (the shipped ASPECT record is reproduced to
  six figures and the step count matches); the complete-card DEBUG run is **not** done in this
  batch, only its cost is measured — it is a measured limitation, not a comparison.
- **Recorded.** 2026-11-26 by the ASPECT cross-code batch, repository commit
  `a93246984d359d5989e683b07cb711ad23a35019`; generated on the Windows host tree under
  `C:\temp\quast_aspect` (ASCII paths because the mount path must not contain the CJK repository
  name).
