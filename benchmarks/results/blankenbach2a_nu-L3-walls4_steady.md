# blankenbach2a: Nu_top of the published wall-refined L3 rung, and the uniform-tree ladder

- **Quantity.** `Nu_top`, `Nu_bot` and `V_rms` of the published wall-refined 2a rung (the
  single number the validation page quotes), plus the uniform-tree L1-L4 resolution study of
  the same card.  Reported below.
- **Published rung (reproduced).** Wall-refined card (level 3 with walls at level 4,
  1 600 leaves, `CASE_WALLMIN=4 CASE_INITREF=1`): `Nu_top = 10.124846`,
  `Nu_bot = 10.528856`, `V_rms = 479.698001`. Reference 10.066 / 480.4334; deltas
  **+0.5846 %** and **−0.1531 %**. The archived recorded run gives 10.124814 / 10.528853 /
  479.694864 — the reproduction agrees to 3.2e-6 in `Nu_top` (2.5e-6 in `Nu_bot`,
  6.5e-6 in `V_rms`), which is the dt-independence check for the CFL-20 re-run.
- **Uniform-tree resolution study.** Levels 1-4 on uniform trees (the same card without the
  wall rule): `Nu_top` = 8.925195, 11.265499, 10.468121, 10.137658;
  `V_rms` = 394.524, 467.536, 475.786, 480.075. Observed order on the three finest:
  p = **1.271** for `Nu_top` (f∞ = 9.904, **GCI = 2.88 %**) and p = **0.944** for `V_rms`
  (f∞ = 484.72, **GCI = 1.21 %**). The sequence is pre-asymptotic (the case has a 1e3 viscosity
  contrast), so this GCI is a coarse bound on the uniform card, not the published rung's error
  bar. It does bound the quoted deviation: +0.58 % is well inside 2.88 %.
- **No GCI for the quoted rung itself.** The published 2a row is a wall-refined, non-uniform
  tree: the refinement ratio is not constant, so a Richardson analysis is not defined for it
  (AUDIT_VV §3.2; confirmed here — the only quoted 2a number is a single rung). The honest
  bracket for it is the uniform-card study above plus the archived recorded run.
- **Command.**
  published rung: `CASE_CFL=20 CASE_STEADY=Inf CASE_WALLMIN=4 CASE_INITREF=1
  julia --project=. --threads=2 examples/benchmark_cases.jl 2a 3 0.2` (run solo);
  uniform ladder: the same with `CASE_WALLMIN=0` at levels 1-4.
  Card: `Ra=1e4`, `eta_law=:expT`, `deta_T=1e3`, `H=0`, free slip, `T=0/1`.
- **Input card / fixture.** `examples/benchmark_cases.jl` sha256
  `0251870b9857280a233b68640c83b78e7b52e09fba0d6c85dbe3b3d284e88941`; initial condition
  `examples/data/initial_temperature_case2a.txt` sha256
  `c4004b01037dd77b06a00726006f1f1bd8abda244d98b53f639529166c318794`.
- **Resolution / run length.** published rung 1 600 leaves, 850 outer steps to t = 0.2002;
  uniform ladder 64 / 256 / 1 024 / 4 096 leaves, 146 / 240 / 435 / 851 steps to t ≈ 0.200.
- **Solver path.** `linear = :auto`; `CASE_CFL=20`, `CASE_STEADY=Inf`.
- **Wall clock.** 252 s (published rung, solo); 16 / 24 / 167 / 1 042 s (uniform L1-L4, the L4
  sharing the host). The historical published rung at CFL 5 took 937 s.
- **Environment.** Julia 1.12.7, `Manifest.toml` sha256
  `ff64fc791ddb601a3119bbb0ff5a3bba7300a86b12b1f90fece23c7de2a247cc`; Ubuntu 24.04.5 (WSL2);
  i5-13490F, 16 vCPU, 25 GB RAM.
- **Logs.** `logs/run_2a_L3walls4.log` (sha256 `40398033a0e4303d7d23481d7df24c98cbe7f3c26c6528795b3774316feeb8ba` (33913 bytes)),
  `logs/run_2a_L1uniform.log` (sha256 `45653cad9e847bb6e5bade421828115175ab9d194d3b0a74b78b5ba19aea3bc7` (16231 bytes)),
  `logs/run_2a_L2uniform.log` (sha256 `ab8b48ed43eee633b9e2fe6b828140ba6aa1734db0735e708ce509349bfb5bbb` (26816 bytes)),
  `logs/run_2a_L3uniform.log` (sha256 `92d1d849e7ab12cf1e7d8e4df770dc82b7d58823fc88e95d8410f08f9178a712` (26179 bytes)),
  `logs/run_2a_L4uniform.log` (sha256 `d732a1d6630f1a924c071ea5ec252d75445103c7a7636576c49a4fdff4905678` (34305 bytes)).
- **Status.** `verified` (published rung reproduced; uniform study measured); **no GCI for the
  quoted non-uniform rung**, uniform-card GCI 2.88 % / 1.21 % labelled as such. Plateaus: the
  last-10 % drift on `Nu_top` is +8e-5 (published rung), ≤2e-4 (uniform levels).
- **Recorded.** 2026-11-26, batch P2 part 1; campaign record `docs/dev/BUG_FIX_LOG.md` §15.
