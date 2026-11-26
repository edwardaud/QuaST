# aspc-davies-case22 annulus: ASPECT -> QuaST state import and the t = 3 same-state comparison

- **Quantity.** The 2-D cylindrical-annulus counterpart of the box ASPECT -> QuaST state
  import (`examples/import_aspect_state.jl`): extract one ASPECT visualization frame of the
  Davies et al. (2022) case-2.2 card (free slip, Ra = 1e5, R0 = 1.22, R1 = 2.22, closed
  periodic ring) as the exact Q2 field, import it into the repository's 16 x 192 Q2-Q1-Q2
  ring (3 072 cells, the same mesh and the same 25 344 velocity / 3 264 pressure / 12 672
  temperature DOFs), and measure (a) the imported state's diagnostics against ASPECT's
  statistics row, (b) the field-level same-state difference against our own converged t = 3
  state, and (c) a continuation that starts from the imported t = 2.00296 state and runs to
  t = 3 under our solver.

- **What was imported.**
  1. `solution-00041` of the pinned ASPECT case-2.2 continuation, **t = 3.000000000** (the
     final frame; 4 MPI pieces, VTU SHA-256 of the concatenated piece set
     `6ef8f5b2272fd59b8a5dbb099d8660e95dbcf440e46d475b7208f0c6d21477c7`), for the
     same-state comparison;
  2. `solution-00001`, the **first case-2.2 output**, exact time **t = 2.002962603857**
     (VTU SHA-256 `c5fa94baa2989ab51fb985e606b1d40ec0ffc38f062b71bd1aac56ad242e11d9`), as
     the continuation seed.

  The exact ASPECT t = 2 state is **not** available: the case-2.1 leg of the v3 batch ran
  with visualization off (its output directory holds only the t = 0 frame `solution-00000`).
  We did **not** re-run ASPECT (a 12-minute container run under the pinned image) and used
  the first case-2.2 frame instead: it is one ASPECT timestep after t = 2 (that step was
  dt = 2.962603856522e-3, the checkpoint's inherited 2.96e-3), so the continuation below
  starts 2.9626e-3 time units after t = 2 and skips exactly one ASPECT step. That offset is
  stated, not compensated.

- **The extractor.** New tracked script `aspect_runs/davies_et_al/extract_annulus_state.py`
  (the annulus twin of `aspect_runs/realistic/extract_aspect_state.py`). ASPECT's DataOut
  writes each Q2 cell as a 2 x 2 block of **bilinear** sub-quads on the cell's nine Q2 nodes;
  the extractor recovers the nine nodal values exactly, rebuilds the structured polar node
  grid and writes one small HDF5 state file per frame (datasets `r (33)`, `phi (384)`,
  `T (33, 384)`, `ux`, `uy`; attributes `R0`, `R1`, `cells_r`, `cells_t`, `t`, `frame`,
  `source_vtu_sha256`, the measured recovery errors, the family-A field diagnostics and the
  matched ASPECT statistics row).

  Measured on both frames (frame 41 / frame 1, identical because the mesh is fixed):

  | item | measured |
  |---|---|
  | VTU points -> unique grid nodes | 27 648 points -> 12 672 nodes = 33 x 384, complete |
  | node multiplicity (per-cell copies) | 1: 3 456, 2: 6 336, 4: 2 880 — the exact structured-grid pattern |
  | radial recovery residual | 1.217e-7 (node spacing 3.125e-2) — the Float32 coordinate floor |
  | angular recovery residual | 4.910e-8 (node spacing 1.636e-2) |
  | duplicate-copy value spread | 0.0 (bit-identical copies, as the renderer measured) |
  | r ladder vs the card's uniform grid | 2.06e-8 |
  | measured radius span | [1.219999998, 2.219999991] against the card's [1.22, 2.22] |
  | seam column | single-valued: phi = 2pi is stored once as phi = 0; the one-sided Q2 angular derivative jump across the seam is 0.1223 (t = 2.00296) and 2.626 (t = 3) — an O(h^2) discretisation diagnostic of the developed plumes, not a constraint |
  | wall temperature | inner exactly 1.0, outer exactly 0.0 (the Dirichlet nodes) |
  | wall normal velocity | max 5.5e-6 (inner) / 4.6e-6 (outer) at t = 3 — Float32-level free-slip data |
  | velocity components | 3 in the VTU; the out-of-plane component is checked to be exactly zero, the two in-plane ones are nonzero |

  The state file is the node cloud of the same 16 x 192 Q2-Q1-Q2 ring QuaST builds, so the
  import is a node transfer, not an O(h^2) interpolation. The extractor deliberately
  **duplicates** the VTU reading of `scripts/annulus_video.py`'s `AspectSource` instead of
  refactoring that frozen renderer (a refactor would force a re-render proof of the
  delivered v3 video); the duplication is pinned by `--cross-check`, which imports
  `AspectSource` itself and compares on the same frame: **12 672 unique renderer points,
  max difference 0.0 in T, 0.0 in speed**, 1.207e-7 in the reconstructed x/y (Float32), 12 288
  quads. The v3 GIF/video/poster were not re-rendered because the renderer's code is
  unchanged (only a new file was added).

- **The importer.** New tracked example `examples/import_annulus_aspect_state.jl` (annulus
  twin of the box importer; the gate includes it, so the CLI and the gate run the same
  code). It builds the ring with the harness's own rule `annulus_benchmark_tree(nx, ny)`,
  evaluates the ASPECT Q2 field at every QuaST GLL node by exact tensor-product Q2
  interpolation (radial nodes 2k, 2k+1, 2k+2; angular nodes 2m, 2m+1, 2m+2 with the index
  taken modulo nphi, so the seam is one column), sets
  theta = T_total - `semquad_base_T` (the radial log base state), zeroes theta on both walls
  and checks the periodic seam trace. Checkpoint: `save_semquad_state` with the template's
  `run_config`, t = the state's exact time, step = 0, theta_prev = theta, pressure zeros,
  dt/dt_prev from the template, velocity_stale = false, the imported velocity as ux/uz;
  the file is reloaded and its config (96 keys), geometry record, t and theta are checked.

  Measured import of the t = 3 frame (16 x 192 ring, template = the case-2.1 checkpoint
  `quast_ckpt/ann_16x192.h5`, dt = 5.276776e-3, Ra = 1e4, 96 config keys):

  | check | measured |
  |---|---|
  | Q2 evaluation at the state's own nodes | max abs error 3.331e-15 (relative 3.33e-15) in T, 1.364e-12 (relative 3.45e-15) in u — round-off |
  | ring nodes coinciding with state nodes | 12 672 of 12 672; max node offset 8.88e-16; max difference between the Q2 evaluation and the nearest stored node 7.82e-15 (relative) |
  | walls | 768 wall nodes, 0 with |theta| > 1e-8; max pre-zeroing |theta| 7.45e-15 (the state's wall T is exactly 1/0 and the base state is exactly 1/0 there) |
  | seam trace | 33 nodes on the phi = 0 ray; jump 3.33e-16 (relative) — round-off |
  | wall normal velocity of the imported state | max |u_r| = 1.02e-5 (Float32-derived free-slip data) |
  | imported-state diagnostics (family A) | Nu_top = **10.316813283**, Nu_bot = **10.269148790**, V_rms = **193.126705055**, Tmax = 1.000000000, mean T = **0.382437932783** |
  | ASPECT statistics row at t = 3 | Nu_top = 9.511780010, Nu_bot = 9.500253524, V_rms = 193.126705, mean T = 0.382437930 |
  | vs ASPECT statistics (Nu = family C) | Nu_top **+8.4635 %**, Nu_bot **+8.0934 %**, V_rms **+2.85e-10**, mean T **+7.28e-9** |
  | vs the extracted Q2 field's family-A diagnostics | Nu_top **+1.32e-5** relative, Nu_bot **+4.03e-5**, V_rms +1.16e-14, mean T +6.31e-9 |

  So the interpolation check is clean: V_rms and mean T reproduce ASPECT's row to the
  quadrature accuracy (1e-10 / 1e-9), the family-A Nu reproduces the exact-Q2-field value to
  the theta-vs-total-T wall-derivative convention (1e-5, the discretisation of the base
  slope), and the 8.46 % / 8.09 % gap to the statistics row is the **family** difference:
  evaluated on the **same** ASPECT field, family A gives 10.316677 and ASPECT's family C
  gives 9.511780, i.e. the 8.46 % is present on one and the same field.
  The same import of the t = 2.00296 frame gives Nu_top = 4.862964029, Nu_bot = 4.874564921,
  V_rms = 460.302783234, mean T = 0.382056194248 (vs the statistics row 4.790342381 /
  4.827097386 / 460.302783 / 0.382056192: +1.51 % / +0.97 % family gap at this transient,
  V_rms 5.1e-10, mean T -4.3e-10).

- **Same-state comparison at t = 3.** Both states live on the identical 12 672-node cloud
  (max coordinate difference 0.0) and the identical 3 072-leaf uniform ring. Our own state
  is the v3 continuation checkpoint `quast_ckpt_cont/ann_16x192.h5` at t = 3.000070640
  (7.1e-5 after ASPECT's 3.0; the plateau drift over that interval is below the printed
  precision, the run's last-window drifts are 2.6e-11 on Nu and 6.7e-10 on V_rms).

  | quantity | our own t = 3 | imported ASPECT t = 3 | relative difference |
  |---|---:|---:|---:|
  | Nu_top (family A) | 10.316647007 | 10.316813283 | -1.612e-5 |
  | Nu_bot (family A) | 10.268873460 | 10.269148790 | -2.681e-5 |
  | V_rms | 193.127060809 | 193.126705055 | +1.842e-6 |
  | mean T | 0.382439915523 | 0.382437932783 | +5.184e-6 |
  | max abs theta | 0.709682732 | 0.709729797 | |
  | L2 norm of theta | 32.06237432 | 32.06237003 | |
  | node-RMS of |u| | 194.435014948 | 194.434654072 | |

  Field difference (ours minus ASPECT, node by node over the 12 672 nodes):

  | norm / statistic | value |
  |---|---:|
  | max abs theta difference | 6.304903e-3 (0.89 % of max |theta|) |
  | L2 norm of the difference | 7.847889e-2 |
  | L2 relative to the theta norm | 2.447694e-3 |
  | RMS of the difference | 6.971565e-4 |
  | p50 / p90 / p99 / p99.9 / max | 4.006e-5 / 8.234e-4 / 3.328e-3 / 5.789e-3 / 6.305e-3 |
  | histogram of |dtheta| | 1 043 nodes below 1e-6, 2 121 in [1e-6, 1e-5), 4 831 in [1e-5, 1e-4), 3 639 in [1e-4, 1e-3), 1 038 in [1e-3, 1e-2); none above 1e-2 |
  | max |dtheta| per radial ring | between 4.19e-3 and 6.30e-3 on all 33 rings — broad, no localised mismatch |
  | max |dux|, |duz|, |d|u|| | 1.973, 2.063, 2.355 |
  | RMS difference of the velocity components / of the speed | 2.605e-3 / 1.982e-3 relative to node-RMS |u| |

  Interpretation. The integral diagnostics of the two states agree to 2e-6 (V_rms), 5e-6
  (mean T) and 2e-5 / 3e-5 (the family-A wall slopes) — the same-state check that the box
  import path established. The local fields differ by 2.4e-3 relative in L2 and 6.3e-3 in
  the maximum: that is the **cross-code discretisation difference** between two different
  Q2-Q1-Q2 formulations (different element-level quadrature, advection and boundary-flux
  treatment) on the same 3 072-cell mesh with identical DOF counts, not a transfer error —
  the transfer itself is exact at the nodes. In particular the 8.5 % / 8.1 % Nu gap cannot
  be attributed to this field difference: the same ASPECT field yields 10.3167 under family A
  and 9.5118 under family C, so it is a wall-functional definition difference, while the two
  fields differ by only 2.4e-3 in L2.

- **Continuation from the imported state.** Restart with the imported t = 2.00296
  checkpoint: the file carries the case-2.1 template's physics (Ra = 1e4), the run is case
  2.2 (Ra = 1e5), so the documented protocol applies — `ANN_RESTART_PHYSICS=allow` warns
  once with the differing key and resets the first resumed dt to dt0 = 1e-4; then
  CFL = 10, 16 x 192, to t = 3: **995 steps, t = 3.000073, 1 580 s wall clock**, writing
  41 frame files as well.
  Final diagnostics: Nu_T = 10.316647, Nu_B = 10.268873, V_rms = 193.12706, mean T =
  0.382440 — identical (to the printed digits) to our own-state v3 continuation.
  The field-level result is the sharp one:

  | comparison at t = 3 | max abs theta difference | L2 relative | diagnostics |
  |---|---:|---:|---|
  | continuation vs ASPECT state | 5.956e-3 | 2.359e-3 | V_rms +1.84e-6, mean T +5.18e-6, Nu_top -1.61e-5, Nu_bot -2.68e-5 (relative) |
  | continuation vs our own state | 3.890e-4 | 2.371e-4 | Nu_top +7.1e-10, Nu_bot +1.1e-9, V_rms +3.9e-10, mean T -5.9e-10 (relative) |

  Starting from the ASPECT state and evolving under our equations for ~1 time unit, the
  trajectory lands on our own discrete steady state to 2.4e-4 relative in L2 (max
  3.9e-4), while the remainder of the difference to ASPECT's t = 3 state is the same
  2.4e-3 cross-code discretisation difference; it is slightly smaller than the own-vs-ASPECT
  2.45e-3 because the start was ASPECT's state. The two QuaST trajectories themselves
  coincide from t about 2.25 onward (see the transient table below).

  Trajectory comparison (nearest available frame to each target time, no time
  interpolation; imported-continuation frames have t = 2.003063 ... 3.000073 and ASPECT
  rows are the raw statistics): values of V_rms relative difference to the ASPECT row.

  | target t | imported-continuation | our own v3 continuation | imported vs own |
  |---:|---:|---:|---:|
  | 2.025 | +1.0588 % | +54.4770 % | -34.5801 % |
  | 2.050 | +13.8951 % | +3.6918 % | +9.8400 % |
  | 2.075 | +1.8046 % | -4.6927 % | +6.8172 % |
  | 2.100 | +0.1063 % | -1.8649 % | +2.0087 % |
  | 2.150 | +0.0985 % | -0.2385 % | +0.3378 % |
  | 2.200 | +0.0078 % | -0.1159 % | +0.1238 % |
  | 2.250 | -0.0213 % | -0.0235 % | +0.0022 % |
  | 2.300 | -0.0097 % | -0.0005 % | -0.0091 % |
  | 2.500 | +0.0002 % | +0.0002 % | +0.0000 % |
  | 3.000 | +0.00018 % | +0.00018 % | +0.00000 % |

  In the steep transient (t below about 2.1) the nearest-frame percentages are dominated by
  the 0.025 frame cadence: the V_rms transient that follows the Ra = 1e4 -> 1e5 switch has a
  large time derivative, so a frame-time offset of ~1e-4 alone moves V_rms by several
  percent (the v3 record measured the same effect). From t about 2.15 on, where the state
  difference is meaningful, the imported trajectory is closer to ASPECT early
  (+0.10 % at 2.15 against our own -0.24 %) and both QuaST trajectories agree to 0.01 % or
  less by t = 2.3. The first frame also measures the momentum-equation difference: at
  t = 2.003063 (one dt = 1e-4 after the import) V_rms = 462.42 against ASPECT's 460.30 at
  t = 2.002963 (+0.46 % over 1e-4): the imported velocity is a seed, and the first Stokes
  solve under our operator replaces it.

- **Honest limitations.**
  1. The exact ASPECT t = 2 state does not exist in the available data; the continuation
     seed is one ASPECT timestep (2.9626e-3) later and the trajectory comparison therefore
     skips that step (stated above, not compensated).
  2. The field transfer is exact at the nodes **because** this card's two grids coincide
     node-for-node; the extractor does not add an interpolation error, only the Float32
     coordinate-recovery level (1.2e-7 radial, 4.9e-8 angular) of the VTU frame.
  3. Family A (our one-sided GLL wall gradient normalised by the per-wall base flux) and
     family C (ASPECT's consistent boundary flux) are different functionals; they are never
     compared as one measurement, and the same-field check above separates them from the
     state difference.
  4. The v3 video's two codes continued **their own** case-2.1 states; this batch exchanges
     exactly one state (t = 3 comparison, and the t = 2.00296 -> 3 continuation). The
     t = 2 -> 3 transient is not a published target.
  5. Our own t = 3 checkpoint is at t = 3.000070640, ASPECT's frame at t = 3.0; the run is at
     the plateau (last-window drifts 2.6e-11 / 6.7e-10), so the time offset contributes
     below the printed field difference.
  6. No solver source was changed by this batch; the importer's config is the template's
     (case-2.1, Ra = 1e4), so continuing a case-2.2 state needs the documented
     `restart_physics=:allow` protocol, which is what the continuation used.

- **Gate and mutation evidence.** New gate G292 (`test/t_g292_annulus_import.jl`, registered
  in `test/runtests.jl`) builds its synthetic states inside the gate (HDF5.jl) and pins:
  exactness (theta vs a quadratic R to 1.332e-14 relative; Nu pair to the analytic
  quadratic value to 1e-15; mean T to the direct radial quadrature to 1.510e-14; V_rms
  exactly 0), the zero-wall-derivative profile (quartic R: Nu_top = 1.001881908935,
  Nu_bot = 0.998965797793, matching the discrete one-sided Q2 value at the gate's 1e-12 tolerance and
  |Nu - 1| at the 1e-3 O(h^2) level), the wall zeroing (384 wall nodes zeroed, max pre-zero
  5.000e-7), the smooth-angular arm (max interpolation error 1.571e-5, 3.141e-4 relative to
  the field scale, bound 1e-3; seam trace jump <= 1e-12) and the checkpoint round trip plus a
  same-physics restart (2 steps, finite). The four refusal arms (non-finite T; a
  self-consistent state of another annulus with shifted radii; an odd-nphi / duplicated-seam
  angular grid; non-benchmark wall temperatures; non-annulus template geometry) each raise
  an actionable `ArgumentError`. The campaign record's red evidence: seven one-line
  mutations of the importer, each run against the gate arm it targets — drop the wall
  zeroing (384 of 384 wall nodes keep theta = 5.0e-7), drop the angular index modulo (Q2
  exactness fails), swap r/phi (seam-trace check fails), drop the non-finite check,
  drop the seam/nphi check, drop the R0/R1 geometry check, drop the template-geometry check
  (all four import where they must refuse) — **all seven red**
  (`out_julia/annulus_import/logs/g292_mutations.log`).

- **Verification.** The nine-hash state dump is unchanged (`out_julia/agent/b_rs.txt` SHA-256
  `0dda697418787882061e6978a289b04189f4295db654566fb37539029eba545c`, diff empty);
  `scripts/check_docs.jl` prints `docs: ok`; `scripts/check_manifest.jl` prints
  `manifest: ok (60 entries)`. The acceptance gate batch (G112, G120, G126, G273, G274, G275,
  G276, G277, G290, G291, G292) ran in one Julia process: **2 524 passed / 2 524 total, exit
  0** (`out_julia/annulus_import/logs/gates_annulus_import.log`); the full suite, README /
  CONTRIBUTING / test-README suite-count surfaces and the public docs pages are the lead's
  acceptance step and were not touched.

- **Commands.** Run from the repository root on the shared machine
  (`/home/szh/dedalus-env/bin/python` 3.12.3 with h5py 3.16 / numpy 2.5.4 / scipy 1.18.1;
  Julia 1.12.7; Ubuntu 24.04 WSL2). Scripts are archived under `out_julia/agent/`
  (`ai_extract.sh`, `ai_import.sh`, `ai_same_state.sh`, `ai_cont_compare.py`, `ai_gates.sh`,
  `ai_mutations.sh`).
  ```text
  # 1. extract the two frames (meshio + h5py; --cross-check also runs the renderer parser)
  python aspect_runs/davies_et_al/extract_annulus_state.py \
      --vtu out_julia/annulus_video_v3/aspect_frames --frame 41 \
      --stats out_julia/annulus_video_v3/logs/aspect/case22_statistics \
      --out out_julia/annulus_import/asp_t3_state.h5 --cross-check
  python aspect_runs/davies_et_al/extract_annulus_state.py \
      --vtu out_julia/annulus_video_v3/aspect_frames --frame 1 \
      --stats out_julia/annulus_video_v3/logs/aspect/case22_statistics \
      --out out_julia/annulus_import/asp_t2003_state.h5 --cross-check

  # 2. import into the 16 x 192 ring (template = our case-2.1 checkpoint)
  julia --project=. --threads=4 examples/import_annulus_aspect_state.jl \
      --state out_julia/annulus_import/asp_t3_state.h5 \
      --template out_julia/annulus_video_v3/quast_ckpt/ann_16x192.h5 \
      --out out_julia/annulus_import/quast_t3/ann_16x192.h5
  julia --project=. --threads=4 examples/import_annulus_aspect_state.jl \
      --state out_julia/annulus_import/asp_t2003_state.h5 \
      --template out_julia/annulus_video_v3/quast_ckpt/ann_16x192.h5 \
      --out out_julia/annulus_import/quast_t2003/ann_16x192.h5

  # 3. continuation from the imported t = 2.00296 state (Ra change -> explicit protocol)
  ANN_RA=1e5 ANN_RESTART=out_julia/annulus_import/quast_t2003/ann_16x192.h5 \
  ANN_RESTART_PHYSICS=allow ANN_CFL=10 ANN_DT0=1e-4 ANN_TMIN=1e9 ANN_TOL=0.0 \
  ANN_WINDOW=2 ANN_MAXSTEPS=6000 \
  ANN_FRAMES=out_julia/annulus_import/quast_cont_frames ANN_FRAME_DT=0.025 \
  ANN_FRAME_NR=200 ANN_FRAME_NPHI=1024 \
  ANN_CKPT_DIR=out_julia/annulus_import/quast_ckpt_cont ANN_CKPT_EVERY=1000000 \
  ANN_REPORT=50 ANN_LINEAR=auto ANN_TAG=case22_imported_v1 \
  julia --project=. --threads=8 examples/annulus_benchmark.jl 16x192 3.0

  # 4. same-state field comparison (imported t=3, our v3 t=3, imported continuation t=3)
  julia --project=. --threads=4 out_julia/agent/ai_same_state.jl \
      out_julia/annulus_import/quast_t3/ann_16x192.h5 \
      out_julia/annulus_video_v3/quast_ckpt_cont/ann_16x192.h5 \
      out_julia/annulus_import/quast_ckpt_cont/ann_16x192.h5

  # 5. gate (standalone or in the suite) and the mutation campaign
  julia --project=. --threads=4 -e 'using Test, QuaST; include("test/t_g292_annulus_import.jl")'
  julia --project=. --threads=4 out_julia/agent/ai_mutations.jl
  ```

- **Inputs and hashes** (`out_julia/annulus_import/logs/INPUT_HASHES.txt` has the complete
  list; ASPECT image pinned as
  `gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac`):

  | file | SHA-256 |
  |---|---|
  | raw ASPECT statistics `logs/aspect/case22_statistics` | `2613cf972f38ee6eddd3fb4d2c27efd629c17c8f782eda10d1669763f567f5ce` |
  | frame 41 VTU set (source attr) | `6ef8f5b2272fd59b8a5dbb099d8660e95dbcf440e46d475b7208f0c6d21477c7` |
  | frame 1 VTU set (source attr) | `c5fa94baa2989ab51fb985e606b1d40ec0ffc38f062b71bd1aac56ad242e11d9` |
  | extracted state t = 3 | `5a7a7b291d7aaf18448f48cbefd093311430e38351bbc0b46e78d1cbf317bdfa` |
  | extracted state t = 2.00296 | `092826152466a3508a9b9fcfe6043b446b8bd70e034bd8bfbe763196e07368c5` |
  | template (our case-2.1 checkpoint) | `2004830c4d9b83f0cafcd2ede48f8b7d2060a1225542d330048923491beaa71b` |
  | our own t = 3 state (`quast_ckpt_cont`) | `b0a1f41c6fed62a308c4b7cdc35706504b271d3aa56a8d30963ae61d5b253a59` |
  | imported checkpoint t = 3 | `4c7f77b0fab0d51d47a21ac497e8101f66886123c5578b095413fc6034507c76` |
  | imported checkpoint t = 2.00296 | `d4c5acb6bc86d0350f129137f50cb6325cad551258e7ea5d0d8ab52e8d8f247c` |
  | extractor `aspect_runs/davies_et_al/extract_annulus_state.py` | `d2b366ae3d696d99836120d5bd96b92aef7a7863d283715789875915abc0e8e4` |
  | importer `examples/import_annulus_aspect_state.jl` | `fbe9ab7af0c8edb6ba9dd23bef42ec3e8c76dbc7e49e95b6194cbc025a6f0961` |
  | gate `test/t_g292_annulus_import.jl` | `2ab11e59c839e31de43d3af1e98143de9d30962238cc1f4dfd44686ed2518d3d` |

- **Evidence files** (untracked, under `out_julia/annulus_import/`): `asp_t3_state.h5`,
  `asp_t2003_state.h5`, `quast_t3/ann_16x192.h5`, `quast_t2003/ann_16x192.h5`,
  `quast_ckpt_cont/ann_16x192.h5` (final imported-continuation state, t = 3.000073),
  `quast_cont_frames/` (41 frames + meta), `logs/import_t3_write.log`,
  `logs/import_t2003_write.log`, `logs/quast_imported_cont.log` and `.time`,
  `logs/same_state_t3.log`, `logs/cont_trajectory_compare.log`, `logs/g292.log`,
  `logs/g292_mutations.log`, `logs/gates_annulus_import.log`,
  `logs/nine_hash_dump.txt`, `logs/check_docs.log`, `logs/check_manifest.log`,
  `logs/extract_summary.log` (the extracted state files' datasets + attributes),
  `logs/INPUT_HASHES.txt`.

- **Status.** `recorded` — the state import is measured end to end on the real Davies et al.
  (2022) case-2.2 ref-4 frames; the same-state t = 3 check, the imported-state
  continuation and the interpolation/diagnostic checks are reported above. Not committed:
  the lead folds the results into the public validation pages and the suite-count surfaces.

- **Recorded.** 2026-11-26 by the annulus state-import batch; solver HEAD
  `a82c08f094ffb25c32a357844b1a08459581971e` plus the new untracked files above.
