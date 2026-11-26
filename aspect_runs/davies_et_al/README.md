# ASPECT counterpart runs for the 2-D annulus (Davies et al. 2022)

The scripts here produce and read the ASPECT side of the annulus cross-code
comparison. They are the annulus siblings of `aspect_runs/realistic/` (the
2-D realistic box): like that directory, nothing here is part of the solver
build or the gate suite; the scripts are provenance for `benchmarks/results/`
records.

- `extract_annulus_state.py` -- extract one ASPECT VTU frame of the
  Davies et al. (2022) case-2.2 card (or its case-2.1 seeding leg) as the
  **exact Q2 field** on the structured polar node grid, written as a small
  HDF5 state file for `examples/import_annulus_aspect_state.jl`. ASPECT's
  visualization output writes every Q2 cell as a 2 x 2 block of bilinear
  sub-quads on the cell's nine Q2 nodes, i.e. a bilinear view of a quadratic
  field; the extractor recovers the nine nodal values exactly, rebuilds the
  `(r, phi)` grid, verifies the point cloud lands on it (measured recovery
  error printed and stored as attributes), checks the seam is single-valued,
  and reports the family-A wall Nusselt pair plus V_rms and mean T of the
  exact Q2 field. For the refinement-4 mesh (3 072 cells) the grid is
  33 radial x 384 angular nodes = the node cloud of QuaST's `16x192`
  Q2-Q1-Q2 ring, so the import is a node transfer.

  ```text
  python extract_annulus_state.py --list --vtu <ASPECT output dir>
  python extract_annulus_state.py --vtu <ASPECT output dir> --frame 41 \
      --stats <raw ASPECT statistics> --out out_julia/annulus_import/asp_t3.h5 \
      --cross-check
  ```

  `--cross-check` imports `class AspectSource` of `scripts/annulus_video.py`
  (the delivered renderer's own VTU reader) and verifies that the renderer's
  unique point cloud and this extractor's recovered nodal values agree bit for
  bit on the frame -- the two parsers are deliberately duplicated (the
  renderer is frozen and refactoring it would force a re-render proof) and
  this is the pin that they cannot drift apart.

- `case22_ref4/` / `case21_ref4/` (if present) -- the pinned ASPECT cards and
  logs of the runs these frames come from; the frames used by the import
  record live (untracked) under
  `out_julia/annulus_video_v3/aspect_frames/solution/`, together with the raw
  statistics table `out_julia/annulus_video_v3/logs/aspect/case22_statistics`.
