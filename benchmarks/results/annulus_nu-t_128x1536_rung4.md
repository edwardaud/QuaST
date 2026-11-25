# annulus: Nu_T/Nu_B at 128 x 1536 (ladder rung 4) — observed order, Richardson and GCI

- **Quantity.** Recorded ladder rungs (16 x 192 → 128 x 1536), `Nu_T`:
  10.316646, 9.899941, 9.646110, **9.567429**; `Nu_B`: 10.268873, 9.943853, 9.665285,
  **9.573026**; `V_rms`: 193.12706, 193.25842, 193.26005, **193.25992**; `<T>`:
  0.382440, 0.382464, 0.382466, **0.382465**. Reference: ASPECT docs 1536 x 128
  (`Nu_T = 9.548`, `Nu_B = 9.584`, `V_RMS = 193.371`, `<T> = 0.382`); Davies et al. (2022)
  Table A1 Firedrake row (`Nu = 9.541`, `V_RMS = 193.26` at 2048 x 512).
- **Observed order / GCI** (three finest rungs, refinement ratio r = 2, Fs = 1.25):
  - `Nu_T`: p = **1.690**, Richardson f∞ = **9.53208**, **GCI = 0.462 %** of the finest value.
    With p = 2 *assumed* (not measured): GCI = 0.343 % (Fs = 1.25) or 0.822 % (2-grid Fs = 3).
  - `Nu_B`: p = **1.594**, f∞ = **9.52734**, **GCI = 0.597 %** (p = 2 assumed: 0.443 %).
  - `V_RMS` and `<T>`: the last-pair differences change sign, so no observed order and no GCI;
    they are converged to ≈1e-5 and ≈1e-6 absolute respectively (reference-side differences
    dominate there).
- **Verdict on the public "0.20 % agreement".** The headline deviation is
  `Nu_T = +0.2035 %` from ASPECT (9.567429 vs 9.548). Its own finest-rung discretization
  uncertainty is **0.462 %** (0.34 % with p = 2 assumed), i.e. |deviation|/GCI = **0.44**:
  **the agreement is consistent with ASPECT but is not an accuracy of 0.20 %.** `Nu_B` deviates
  −0.1145 % against a 0.597 % GCI (ratio 0.19) and is comfortably inside.
- **Verdict on "the convergence is second order".** The measured finest-pair order is 1.69
  (Nu_T) and 1.59 (Nu_B), not 2; the sequence is pre-asymptotic. The p = 2 extrapolation lands
  on the Firedrake value (9.5412 vs 9.541), which supports the second-order *assumption* but is
  not a measured order. The public page's "ratios 2.2, 3.6, 5.0 against 4" are error ratios,
  not an observed order (a ratio of 5.0 corresponds to p ≈ 2.3, outside the asymptotic range).
- **Command / card.** Recorded by `julia --project=. examples/annulus_benchmark.jl <rung>` with
  `ANN_CFL=0 ANN_DT0=1e3 ANN_MAXPIC=25 ANN_PIC_RELAX=0.5` (steady relaxation; each step is a
  converged steady energy solve, so the step sequence is the Stokes↔energy Picard iteration),
  `ANN_IC=firedrake` (Davies et al. 2022 Eq. 41, A = 0.02), Q2-Q1-Q2, Galerkin advection,
  no entropy viscosity, free slip, periodic ring, R0 = 1.22, R1 = 2.22, Ra = 1e5. Card hash:
  `examples/annulus_benchmark.jl` sha256
  `6dd7df1ce8e8ce6719338c4a1402c7fa1fc084a3418745bc45fe60fce6215cd8`.
- **Resolution.** 16 x 192 = 3 072 leaves / 25 344 velocity DOFs; 32 x 384 = 12 288 / 99 840;
  64 x 768 = 49 152 / 396 288; 128 x 1536 = 196 608 / 1 579 008. Recorded wall clocks
  113 s / 193 s / 424 s / 1 391 s, peak RSS 1.9 / 2.3 / 4.8 / 13.7 GB.
- **Environment.** Julia 1.12.7, 8 threads, Ubuntu 24.04 (WSL2), 25 GB RAM, `Manifest.toml`
  sha256 `ff64fc791ddb601a3119bbb0ff5a3bba7300a86b12b1f90fece23c7de2a247cc` (the recorded
  campaign predates the P2 batch; the card and tree are unchanged for this card).
- **Log.** `logs/recorded_annulus_ladder.log`, sha256 `5a0aa268b6f71b9a2d7e91cac5b8046571d94bb0a5bbcbbc51b1c4dcebdacc91` (8369 bytes).
- **Status.** `recorded` — the four rungs are the 2026-11-25 campaign's own `ANNULUS_RESULT`
  lines, archived here with their SHA-256. Per the P2 rule the 128 x 1536 rung was **not
  re-run** (1 391 s, 13.7 GB); the GCI above is computed from the recorded values, and the
  steady-state / relaxation-dt uncertainty is separately re-measured on the coarse rung in
  `annulus_steady-dt_L4_relaxation.md`.
- **Recorded.** 2026-11-26, batch P2 part 1; the raw log is archived beside this record.
