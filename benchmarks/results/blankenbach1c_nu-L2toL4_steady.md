# blankenbach1c: Nu at levels 2, 3, 4 on uniform trees (Ra = 1e6), steady state

- **Quantity.** `Nu_top` (top-wall one-sided GLL flux; H = 0 so the walls must agree): L2 =
  21.032435, L3 = 23.689323, L4 = 23.038085. Published reference (as printed in
  `docs/en/validation.md`) 21.972465; deltas −4.278 %, **+7.814 %**, +4.850 %. `V_rms`:
  844.798, 833.576, 834.010 against 833.98977 (+1.296 %, −0.050 %, +0.002 %).
- **Observed order / GCI.** **None exists.** The successive differences are not monotone
  (`f(L2) − f(L3) = −2.657` then `f(L3) − f(L4) = +0.651`), so the three-grid observed order is
  undefined and no Richardson/GCI can be quoted for this sequence. The recorded L3/L4/L5 ladder
  is likewise non-monotone with the opposite sign of difference (0.6503 then 0.7215), so no GCI
  exists for the quoted L5 value either. The only defensible statement is that at Ra = 1e6 the
  wall flux is **not yet in its asymptotic range** at these resolutions (the deviation scales
  7.81 % → 4.85 % → 1.56 % on the recorded ladder), and the L5 deviation +1.56 % carries **no
  measured uncertainty** — a percent-level uncertainty is the honest bracket.
- **Command.**
  `CASE_CFL=20 CASE_STEADY=Inf julia --project=. --threads=2 examples/benchmark_cases.jl 1c {2,3,4} 0.05`.
  CFL 20 is the dt-independence device described in the 1a record.
- **Input card / fixture.** `examples/benchmark_cases.jl` sha256
  `0251870b9857280a233b68640c83b78e7b52e09fba0d6c85dbe3b3d284e88941`; initial condition
  `examples/data/initial_temperature_case1c.txt` sha256
  `70e5ef60699c39e25dbabea92da51ee684d0adc49deaf20d513a54c4e8abd419`; `Ra = 1e6`, constant
  viscosity, `H = 0`, free slip.
- **Resolution / run length.** 256 / 1 024 / 4 096 leaves, 92 / 146 / 271 outer steps to
  t ≈ 0.050; `N` = 1 089 / 4 225 / 16 641 Q2 nodes, p = 2.
- **Solver path.** `linear = :auto`; `CASE_CFL=20`, `CASE_STEADY=Inf`, no AMR/wall rule.
- **Wall clock.** 75 s / 84 s / 264 s driver-call time (2 threads; another run shared the host).
- **Environment.** Julia 1.12.7, `Manifest.toml` sha256
  `ff64fc791ddb601a3119bbb0ff5a3bba7300a86b12b1f90fece23c7de2a247cc`; Ubuntu 24.04.5 (WSL2);
  i5-13490F, 16 vCPU, 25 GB RAM.
- **Logs.** `logs/run_1c_L2.log`, `logs/run_1c_L3.log`, `logs/run_1c_L4.log`
  (sha256 `3b3810791e4ad4f3fc0a63a99ceccee574e5aa3c4d118733e365c18a3b63079f` (10813 bytes); sha256 `edf5f5066082171890287bfd48640769828fb08aca234005dca214157c3aed76` (16888 bytes); sha256 `9a9692e887474798f1ce3a681d74f169ed844b2299ae46b669e80526493642b4` (30934 bytes)).
- **Status.** `verified` for the values; **no GCI** (non-monotone). Each run reaches its
  plateau (last-10 % drift +9e-6, −2.0e-4, −3.1e-5 relative on `Nu_top`; last Picard change
  2.8e-6 / 5.6e-5 / 5.8e-6). The L3/L4 values reproduce the recorded CFL 0.5 run
  (23.687756 / 23.037486) to +6.6e-5 and +2.6e-5 relative. L5 (16 384 leaves, ~2.5 h at
  CFL=20 in this budget) was **not re-run**; its recorded value is archived in
  `logs/recorded_1c_L5.log`.
- **Recorded.** 2026-11-26, batch P2 part 1; the raw log is archived beside this record.
