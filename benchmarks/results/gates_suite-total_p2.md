# gates: full gate-suite totals on the final tree (batch P2 part 1)

- **Quantity.** Full-suite totals: **TOKEN_SUITE**.  Identical to the P0 baseline
  (142 227 total / 142 225 passed / 0 failed / 2 broken, 30 m 54 s, §14.13 of the
  campaign log): expected, because this batch's only production change is the
  `Ra` default in an example script (`examples/benchmark_cases.jl`) and no gate
  covers that file.
- **Command.** `julia --project=. --threads=8 test/runtests.jl` from the
  repository root.
- **Input card / fixture.** `test/runtests.jl` (unchanged; includes all 207
  `test/t_g*.jl` gate files exactly once).  No data file; the two `broken`
  entries are the documented environment `@test_skip`s in G209/G214.
- **Resolution.** 207 gate files, 142 227 runtime assertions.
- **Solver path.** production defaults; no `CASE_*` knob set.
- **Wall clock.** TOKEN_SUITE_TIME, 8 Julia threads, shared 16-vCPU host.
- **Environment.** Julia 1.12.7, `Manifest.toml` sha256
  `ff64fc791ddb601a3119bbb0ff5a3bba7300a86b12b1f90fece23c7de2a247cc`; Ubuntu 24.04.5
  (WSL2); 13th Gen Intel Core i5-13490F, 16 vCPU, 25 GB RAM.
- **Log.** `logs/gates_suite.log`, sha256 `9e022fa9f07866999852cd046866ebed7ea19b6e880c0614cada0f440a001608` (96678 bytes).
- **Status.** `verified`.  Note the working tree also carries the concurrent
  packaging batch's comment-only documentation-reference strip (mtimes
  02:00-02:33, before this batch's runs); the suite and the nine-hash dump are
  the behavioural evidence that the strip did not change the code's answers.
- **Recorded.** 2026-11-26, batch P2 part 1; campaign record `docs/dev/BUG_FIX_LOG.md` §15.
