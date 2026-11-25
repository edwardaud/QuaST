# blankenbach1b: Nu at levels 2, 3, 4 on uniform trees (Ra = 1e5), steady state

- **Quantity.** `Nu_top` (H = 0 so the walls agree): L2 = 11.436454, L3 = 10.990515,
  L4 = 10.670756. Published reference 10.534095; deltas +8.567 %, +4.333 %, +1.297 %.
  `V_rms`: 193.0645, 193.2437, 193.2201 against 193.21454 (−0.078 %, +0.015 %, +0.003 %).
- **Observed order / GCI.** On the re-run L2-L4 sequence p = **0.480**, f∞ = 9.86,
  GCI(Fs = 1.25) = **9.49 %** — the coarse end is not in its asymptotic range (this is a
  Ra = 1e5 boundary layer, and L2/L3 are under-resolved). The **usable** GCI for the quoted
  numbers comes from the archived recorded L3/L4/L5 ladder: p = **1.652**, f∞ = **10.5213**,
  **GCI = 0.560 %** of the L5 value. The quoted L5 deviation (+0.3277 %) is 0.59 × that GCI, i.e.
  consistent inside its own uncertainty; the L4 re-run reproduces the recorded L4 value
  (10.669987) to +7.2e-5 relative.
- **Command.**
  `CASE_CFL=20 CASE_STEADY=Inf julia --project=. --threads=2 examples/benchmark_cases.jl 1b {2,3,4} 0.08`.
  CFL 20 only shortens the transient (BDF2 fixed point is dt-independent); the recorded
  CFL 0.5 value is the anchor reproduced below.
- **Input card / fixture.** `examples/benchmark_cases.jl` sha256
  `0251870b9857280a233b68640c83b78e7b52e09fba0d6c85dbe3b3d284e88941`; initial condition
  `examples/data/initial_temperature_case1b.txt` sha256
  `1cdcea5f6d3744c2218ab6dd1b33fe3a6d0ce1d92b617824987a8e42b3e0fcb5`; `Ra = 1e5`, constant
  viscosity, `H = 0`, free slip.
- **Resolution / run length.** 256 / 1 024 / 4 096 leaves, 77 / 86 / 120 outer steps to
  t ≈ 0.08; `N` = 1 089 / 4 225 / 16 641 Q2 nodes, p = 2.
- **Solver path.** `linear = :auto`; `CASE_CFL=20`, `CASE_STEADY=Inf`, no AMR/wall rule.
- **Wall clock.** 83 s / 81 s / 212 s driver-call time (2 threads; a second run shared the host
  for L2/L3 — the two are within noise of each other).
- **Environment.** Julia 1.12.7, `Manifest.toml` sha256
  `ff64fc791ddb601a3119bbb0ff5a3bba7300a86b12b1f90fece23c7de2a247cc`; Ubuntu 24.04.5 (WSL2);
  i5-13490F, 16 vCPU, 25 GB RAM.
- **Logs.** `logs/run_1b_L2.log` (sha256 `86051af649511c09c15d598f9cab485f1d12034bc25e00cb75156d8e90a9bd5b` (9183 bytes)), `logs/run_1b_L3.log`
  (sha256 `ac5a29c0321847a25667d9fa85f8021932f65a2fc410504156233e2862b5c6be` (10259 bytes)), `logs/run_1b_L4.log` (sha256 `bd438092e220c3b823166428636525d04d39c85c186c04281bbca6d117c28fed` (14155 bytes)).
- **Status.** `verified` values, **no usable GCI from the coarse re-run ladder** (p < 1); the
  recorded L3-L5 ladder's GCI (0.56 %) is the quoted uncertainty. Each run reaches its plateau
  (last-10 % drift +1.9e-4, −2.7e-5, −2.7e-5 relative on `Nu_top`). L5 (16 384 leaves, ~35 min at
  CFL=20 in this budget) was **not re-run**; its recorded log is archived as `logs/recorded_1b_L5.log`.
- **Recorded.** 2026-11-26, batch P2 part 1; campaign record `docs/dev/BUG_FIX_LOG.md` §15.
