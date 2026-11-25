# tosi1: Nu_top and V_rms of Tosi et al. (2015) Case 1 at level 3, fixed default Ra map

- **Quantity.** `Nu_top = 3.421632`, `V_rms = 249.370429` (32 x 32, t = 0.5002). Against the
  eleven-code medians of the paper's Table 2 (`Nu_top = 3.4214`, `V_rms = 249.573`, the median
  values used by `` §5-6): **+0.0068 %** and **−0.0811 %**. Both lie
  inside the eleven-code band (`Nu_top` 3.3129 … 3.5889; `V_rms` 248.9252 … 296.6156).
- **Command.**
  `CASE_CFL=10 CASE_STEADY=Inf julia --project=. --threads=2 examples/benchmark_cases.jl tosi1 3 0.5`
  with **no `CASE_RA` and no `CASE_ETA0`**: the shipped defaults after the P2 fix are
  `eta0 = 0.1`, `Ra = 100 / eta0 = 1000`, hence `Ra_eff = Ra · eta0 · eta_ref = 100`, which is
  the paper's Rayleigh number. The fixed point is step-size independent, so the transient is
  shortened with CFL 10 (the audit's plateau reference used CFL 5; both reach the same values).
- **Input card / fixture.** `examples/benchmark_cases.jl` sha256
  `0251870b9857280a233b68640c83b78e7b52e09fba0d6c85dbe3b3d284e88941` (HEAD `3f4cca7` plus the
  uncommitted P2 Tosi-default edit); analytic initial condition `θ₀ = 0.01 cos(πx) sin(πy)`,
  no data file. Card: `TosiLaw(etaT=1e5, etaZ=1, eta_star=0, sigma_y=0, eta_init=1e-1,
  eta0=0.1, eta_min=1e-6, eta_max=10, z_is_depth=true)`, `Ra=1000`, `H=0`, free slip.
- **Resolution.** Level 3 = 32 x 32 cells, 1 024 leaves, `N = 4 225` Q2 nodes, `p = 2`;
  reached t = 5.0022e-01 in 1 380 outer steps.
- **Solver path.** `linear = :auto` (the selector's choice), `CASE_CFL=10`,
  `CASE_STEADY=Inf` (no early stop), default `CASE_MAXPIC=1`, `linear`/`kmg` defaults otherwise.
- **Wall clock.** 522 s driver-call time (601 s process wall), 2 Julia threads; another
  benchmark run shared the 16-vCPU WSL2 host (see environment).
- **Environment.** Julia 1.12.7, `Manifest.toml` sha256
  `ff64fc791ddb601a3119bbb0ff5a3bba7300a86b12b1f90fece23c7de2a247cc`; Ubuntu 24.04.5 LTS
  (WSL2); 13th Gen Intel Core i5-13490F, 16 vCPU, 25 GB RAM.
- **Log.** `logs/run_tosi1_fixeddef.log`, sha256 `71e55435fdc58d5c60e1f52262e93dd58b86df477a0676128972d4a743621495` (21923 bytes)
  (raw, git-ignored: `out_julia/agent/p2/run3/tosi1_fixeddef.log`).
- **Status.** `verified` — reaches the recorded plateau: `Nu_top` moves by 3.5e-6 relative and
  `V_rms` by 1.7e-6 relative over the last 10 % of the horizon (t = 0.40 → 0.50), and the last
  Picard change is 1.3e-8. The first two outer steps reproduce the recorded
  `(eta0 = 0.1, Ra = 1000)` run digit for digit (`Nu = 1.00000/1.00039`,
  `Vrms = 6.3988`, closure −3.839e-04); the old `Ra = 10/eta0` default started at
  `Vrms = 6.3988e-01` (Ra_eff = 10), so the fix is confirmed by the run itself.
- **Recorded.** 2026-11-26, batch P2 part 1; the raw log is archived beside this record.
  No resolution study exists for the Tosi cases, so no GCI is quoted: the delta is
  single-resolution and the reference side carries the 11-code spread.
