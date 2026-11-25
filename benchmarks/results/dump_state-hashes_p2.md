# dump: the nine state hashes (box bitwise-invariance pin)

- **Quantity.** SHA-256 of the nine state arrays of the standard box dump card:
  `θ, ux, uz, pv, hist, leaves, txx, tyy, txy` after the 6-step card
  (`run_semquad_simulation`, 1 024 leaves, `p=2`, AMR on, `shear_modulus=1`, `eta_ref=2`,
  checkpoint written and resumed). The nine `L1` lines are byte-identical to the recorded
  baseline `out_julia/agent/b_rs.txt`, whose SHA-256 is
  **`0dda697418787882061e6978a289b04189f4295db654566fb37539029eba545c`**.
- **Command.** `julia --project=. --threads=8 out_julia/agent/state_dump_run.jl`.
  (The dump card itself lives in the git-ignored scratch tree, as recorded by batch P0 §14.12;
  it is reproduced here as the invariant check, not re-shipped.)
- **Input card / fixture.** the dump script's card `Params(Ra=1e3, H=10, eta_law=:composite,
  comp_Ad=1e-2, comp_As=2.5e-1, comp_n=3, eta_T0=0.5, deta_T=1, eta_min=1e-4, eta_max=1e4,
  bc=:free, shear_modulus=1, eta_ref=2)`, 4 096-leaf... `mktree()` = 4^2 → see the P0 record;
  the card hash is unchanged from the P0 baseline (git HEAD `3f4cca7`).
- **Resolution.** 1 024 leaves at the dumped state, p = 2, direct linear path.
- **Solver path.** `linear=:direct`, `maxpic=2`, `extdiag=true`, checkpoint at steps 3 and 6.
- **Wall clock.** ≈60 s process (one-off Julia JIT included).
- **Environment.** Julia 1.12.7, 8 threads, Ubuntu 24.04.5 LTS (WSL2), i5-13490F, 25 GB RAM,
  `Manifest.toml` sha256 `ff64fc791ddb601a3119bbb0ff5a3bba7300a86b12b1f90fece23c7de2a247cc`.
- **Log / output.** `out_julia/agent/p2/dump_run.log` and `out_julia/agent/p2/dump_l1.txt`
  (9 lines, reproduced here):

  ```text
  L1 theta 320bc5ccb0646254e3520072df5c00226e7828f1b9b475df096b941729e1bef9
  L1 ux 16af946301d1313d5782378be624d3abbdcf4f4a5cde929eb0502753b392196f
  L1 uz c7d180bad6c44c62bd0964b32aab1bec2fa69843adc7558c8868992c0f3fc733
  L1 pv c5415146a30ea0a19edbd1845ddbc6098abb6184eba60edb888a32900e1912e3
  L1 hist 09e4e0427a9335f4fd480f132c5c32769047e253fdc6d2d06f6613f37e82dc17
  L1 leaves 83675250c933b1a88b37c43d8a6ab0670d2734bb0b0a4f7904d7662f83a057d4
  L1 txx c3f44deb48faf9b372df136e4dd711c7d2ec98de4382ca0b07438b26770e30fd
  L1 tyy 79cfd3b804cbbf6d1c926b1985c79482f02405e9bec8eaad4b4e7150b6f369ca
  L1 txy 24a2c545e5125b80b70b3d2cdee557fe648f3228e12bd517f364216f115a6a65
  ```

- **Status.** `verified` — re-run twice in this batch (before and after the only production
  change, the Tosi default in `examples/benchmark_cases.jl`, which does not touch the solver);
  identical both times.
- **Recorded.** 2026-11-26, batch P2 part 1; campaign record `docs/dev/BUG_FIX_LOG.md` §15.
