# annulus_linear-paths: `linear=:direct`, `:mg` and `:auto` on the 2-D periodic annulus

- **Quantity.** The same published-card annulus state solved along the three linear
  paths, at two rungs of the benchmark ladder: an $8 \times 96$ ring (768 leaves,
  7 392 saddle DOFs) and the first published rung, $16 \times 192$ (3 072 leaves,
  28 608 saddle DOFs).  Same card throughout (`Ra = 1e5` isoviscous, free slip,
  periodic ring, `R0 = 1.22`, `R1 = 2.22`, paper initial condition, Galerkin
  advection, `ev = 0`, `maxpic = 1`), only `ANN_LINEAR` changes.  The final
  state's printed diagnostics:

  | run | saddle DOFs | `Nu_T` | `Nu_B` | `V_rms` | mean `T` | `div_vol` | `Ibase` | wall clock | log |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---|
  | 8 x 96, `:direct` | 7392 | 1.334815 | 0.752257 | 97.23519 | 0.451303 | 1.600e-14 | 1.919e-04 | 35.1 | `logs/annulus_paths_small_direct.log` |
  | 8 x 96, `:mg` | 7392 | 1.334815 | 0.752257 | 97.23519 | 0.451303 | -8.934e-15 | 1.917e-04 | 86.3 | `logs/annulus_paths_small_mg.log` |
  | 8 x 96, `:auto` | 7392 | 1.334815 | 0.752257 | 97.23519 | 0.451303 | -8.934e-15 | 1.917e-04 | 86.1 | `logs/annulus_paths_small_auto.log` |
  | 16 x 192, `:direct` | 28608 | 1.328648 | 0.731659 | 39.88487 | 0.451539 | -3.183e-14 | 4.117e-06 | 31.4 | `logs/annulus_paths_rung_direct.log` |
  | 16 x 192, `:mg` | 28608 | 1.328648 | 0.731659 | 39.88487 | 0.451539 | -2.659e-14 | 3.283e-06 | 84.7 | `logs/annulus_paths_rung_mg.log` |
  | 16 x 192, `:auto` | 28608 | 1.328648 | 0.731659 | 39.88487 | 0.451539 | -2.659e-14 | 3.283e-06 | 85.5 | `logs/annulus_paths_rung_auto.log` |

  At both rungs the three paths give the same printed scalars.  The only
  differences are in the quantities that are cancellation-amplified: the dropped
  continuity row is $\sim 10^{-14}$ in both, and the base-advection diagnostic
  `Ibase` (a small difference of large terms, $10^{-6}$ at
  $16 \times 192$) differs in its last digits.  This is the expected signature of
  two linear solves converged to their own tolerances, not of two different
  discretisations.

- **Which path the production ladder used.**  The recorded ladder
  (`annulus_nu-t_128x1536_rung4.md`) runs with the harness default
  `ANN_LINEAR=auto`.  The selector's rule (`src/stokes/selector.jl`) chooses the
  matrix-free multigrid above 2 000 saddle DOFs for viscosity contrast
  $\le 100$ (this card is isoviscous, contrast 1), so **every published rung,
  $16 \times 192$ upward, is a matrix-free-MG result**; the `:auto` run above
  matches the explicit `:mg` run in both wall clock and values, which confirms
  the resolution.  The direct sparse path is verified against it end to end at
  $16 \times 192$; at the finest recorded rung ($128 \times 1536$, $\sim 3.2
  \times 10^6$ saddle unknowns) a sparse direct factorisation is not attempted.
  Wall clocks in the table are the harness's driver-call time and include the
  first-call JIT; the production rung timings are in the ladder record
  (113 s / 193 s / 424 s / 1 391 s).

- **Mapped-geometry solver support.**  On a mapped space only `:direct`, `:auto`
  and `:mg` are accepted; `:schur_cg` and `:lagged` are refused with a message
  that says why: no mapped gate covers them (the development campaign measured
  both on the ring against `:direct` — 1e-12 relative for `:lagged`, iteration
  tolerance for `:schur_cg` — but a gate-coverage restriction is kept until a
  mapped gate lands).  The ring-AMR gate (`t_g276`) exercises AMR with
  `:direct`; the matrix-free MG on the ring is exercised on fixed meshes here
  and in `t_g276`'s `:mg` arm; an end-to-end **ring + AMR + MG** run is not
  separately gated.

- **Commands.**  From the repository root (Julia 1.12.7, 8 threads):
  ```text
  ANN_LINEAR=direct ANN_TSTOP=0.02  ANN_MAXSTEPS=24 ANN_MAXPIC=1 ANN_REPORT=1 \
  ANN_TOL=0 ANN_WINDOW=2 ANN_CFL=1.0 ANN_DT0=1e-5 ANN_DTMAX=1e-2 \
    julia --project=. --threads=8 examples/annulus_benchmark.jl 8x96 0.02

  ANN_LINEAR=mg     ... (as above) ... 8x96 0.02

  ANN_LINEAR=auto   ... (as above) ... 8x96 0.02

  ANN_LINEAR=direct ... (as above) ... 16x192 0.002
  ANN_LINEAR=mg     ... (as above) ... 16x192 0.002
  ANN_LINEAR=auto   ... (as above) ... 16x192 0.002
  ```
  `ANN_LINEAR=mg` additionally sets `mg_operator=:matrixfree` and
  `mg_threads=8` in the harness; `:auto` resolves internally and forces the
  same calibrated configuration.

- **Environment.**  Julia 1.12.7, Ubuntu 24.04 (WSL2), 8 threads, 25 GB RAM;
  the runs were made on 2026-11-26 with the tree of commit
  `3964f41` (the same tree whose nine state hashes are unchanged).

- **Archived logs** (trimmed? no: the runs are short, these are complete):
  - `logs/annulus_paths_small_direct.log` — 4022 bytes, SHA-256 `81c846d4256877d4139ef6cc531dba134eb453b52a21aad472dce1b963c7a11a`
  - `logs/annulus_paths_small_mg.log` — 4023 bytes, SHA-256 `25032f503ac63e1fb62b4c2dd8f561ff8db243abe94c536d381514d4beca38b2`
  - `logs/annulus_paths_small_auto.log` — 4023 bytes, SHA-256 `a6ab229ada588b317830e54fda98b66d58e295b21cc66c056d6255778c69475d`
  - `logs/annulus_paths_rung_auto.log` — 1333 bytes, SHA-256 `246bf2a1ccb8ed5974a2bd0b6965b54e5149849799a310adccf8d8420316423a`
  - `logs/annulus_paths_rung_mg.log` — 1333 bytes, SHA-256 `d4e8b67206d2c39a705f5bca3959773fc03f53b8a9daa2409d3267571570b744`
  - `logs/annulus_paths_rung_direct.log` — 1329 bytes, SHA-256 `2aef5eee05b5912d6592a51309adfb4952e13e2232436e38bd8a0dc25e8bd28f`

- **Status.** verified (the path-equivalence statement); the production-ladder
  matrix-free-MG attribution is `recorded` from the selector rule and the
  `:auto` runs above.
