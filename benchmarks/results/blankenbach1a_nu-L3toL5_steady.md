# blankenbach1a: Nu at levels 3, 4, 5 on uniform trees, steady state

- **Quantity.** `Nu_top` (top-wall one-sided GLL flux; `Nu_bot` identical because H = 0 and the
  walls agree): L3 = 4.937073, L4 = 4.898188, L5 = 4.888160. Published reference
  (Blankenbach et al. 1989, as printed in `docs/en/validation.md`) 4.884409; deltas
  +1.0782 %, +0.2821 %, +0.0768 %. `V_rms`: 42.8677, 42.8701, 42.8670 against 42.864947
  (+0.0063 %, +0.0121 %, +0.0049 % — converged at every level).
- **Convergence / GCI.** Observed order p = **1.955** from the three finest values
  (recorded campaign: 1.999). Richardson f∞ = **4.884675**; **GCI(Fs = 1.25) = 0.089 %** of the
  L5 value (recorded ladder: 0.083 %). |L5 − reference| = 0.0768 % is 0.86 × its own GCI, i.e.
  the finest value is consistent with the reference inside its discretization uncertainty.
- **Command.**
  `CASE_CFL=20 CASE_STEADY=Inf julia --project=. --threads=2 examples/benchmark_cases.jl 1a {3,4,5} 0.15`.
  CFL 20 (not the ASPECT-like 5) only shortens the transient: the BDF2 step's fixed point is the
  same discrete steady state for any dt (the time derivative vanishes there). The recorded
  CFL 0.5 value is the anchor and it is reproduced below.
- **Input card / fixture.** `examples/benchmark_cases.jl` sha256
  `0251870b9857280a233b68640c83b78e7b52e09fba0d6c85dbe3b3d284e88941`; initial condition
  `examples/data/initial_temperature_case1a.txt` sha256
  `e91c379d9b744505a664fe415d53478aaa219abed96792ae691f7abaede2dd2b`; `Ra = 1e4`, constant
  viscosity (`eta_law=:expT, deta_T=1`), `H=0`, free slip, `T=0` top / `1` bottom.
- **Resolution / run length.** 1 024 / 4 096 / 16 384 leaves, 89 / 92 / 112 outer steps to
  t ≈ 0.151. `N` (Q2 nodes) 4 225 / 16 641 / 66 049, p = 2.
- **Solver path.** `linear = :auto`; `CASE_CFL=20`, `CASE_STEADY=Inf`, no AMR, no wall
  refinement (`CASE_WALLMIN=0`).
- **Wall clock.** 104 s / 185 s / 470 s driver-call time (2 Julia threads; two runs shared the
  16-vCPU host). The historical CFL 0.5 runs took 269 s / 1 633 s / 13 640 s.
- **Environment.** Julia 1.12.7, `Manifest.toml` sha256
  `ff64fc791ddb601a3119bbb0ff5a3bba7300a86b12b1f90fece23c7de2a247cc`; Ubuntu 24.04.5 LTS
  (WSL2); 13th Gen Intel Core i5-13490F, 16 vCPU, 25 GB RAM.
- **Logs.** `logs/run_1a_L3.log`, `logs/run_1a_L4.log`, `logs/run_1a_L5.log`
  (sha256 `d025eef92b7f7d6f2c8980ea4f454b7708b602c06234a9aa3344b6920eb02672` (10348 bytes); sha256 `55816913f9f2ea013266fad94c01109e1352a77af5c2cb4c4cb7e15c42a11a74` (10770 bytes); sha256 `36823fca971fae49e3c760117883c177297f924a73330709ed1ea715aca1177c` (13069 bytes));
  raw, git-ignored: `out_julia/agent/p2/run3/1a_L{3,4,5}.log`.
- **Status.** `verified` — each run reaches its own plateau: over the last 10 % of the horizon
  (t = 0.136 → 0.151) `Nu_top` moves by +7.0e-5 / +1.6e-4 / +1.3e-4 relative, and the last
  Picard change is 5e-5 / 5e-5 / 1e-5. The time drift is below 15 % of the L4-L5 difference
  for L5 and below 1 % of the L3-L4 difference for the coarse levels, so the ladder is
  space-dominated. Reproduces the recorded CFL 0.5 ladder (4.936942 / 4.897925 / 4.888163) to
  +1.3e-4 / +2.7e-4 / −6.1e-7 relative; the L5 values agree to the last printed digit.
- **Recorded.** 2026-11-26, batch P2 part 1; the raw log is archived beside this record.
