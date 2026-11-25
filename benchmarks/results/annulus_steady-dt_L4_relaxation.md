# annulus: steady-state and relaxation-dt uncertainty (AUDIT_VV F9)

- **Quantity.** The fixed point of the `16 x 192` annulus relaxation run (and one `32 x 384`
  cross-check) as a function of the relaxation parameters, and the residual evidence at the
  final state. The published ladder uses `ANN_CFL=0 ANN_DT0=1e3 ANN_MAXPIC=25
  ANN_PIC_RELAX=0.5`: each outer step is a converged *steady* energy solve, so the step
  sequence is the Stokes↔energy Picard iteration and the stop criterion is a fixed-point test
  (`docs/en/validation.md` §3.5 of AUDIT_VV: "steady" is an iterate-change statement; no
  momentum residual was reported, no dt study existed).
- **Fixed-point values** (16 x 192, 12 outer steps, `maxpic = 25`; recorded ladder L4 value in
  brackets = 10.316646 / 10.268873 / 193.12706 / 0.382440):

  | dt | pic_relax | steps | wall s | Nu_T | Nu_B | V_rms | ⟨T⟩ |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 1e3 | 0.3 | 12 | 191 | 10.316646310 | 10.268872985 | 193.1270583 | 0.382439912 |
  | 1e3 | 0.5 (ladder setting) | 12 | 218 | 10.316646985 | 10.268873444 | 193.1270608 | 0.382439916 |
  | 1e4 | 0.5 | 12 | 300 | 10.316647005 | 10.268873458 | 193.1270608 | 0.382439921 |
  | 1e5 | 0.5 | 12 | 192 | 10.316647012 | 10.268873463 | 193.1270608 | 0.382439921 |
  | 32×384, 1e3 | 0.5 | 12 | 267 | 9.899942297 | 9.943854260 | 193.2584248 | 0.382464313 |
  | 32×384, 1e5 | 0.5 | 12 | 421 | 9.899942324 | 9.943854283 | 193.2584248 | 0.382464313 |

  **Result:** the fixed point is independent of the relaxation step size over three decades of
  dt and of the damping factor (0.3 vs 0.5). The mutual spread in `Nu_T` is **6.8e-8 relative**
  (7.0e-7 absolute), `Nu_B` 4.6e-8, `V_rms` **1.3e-8**, `⟨T⟩` 8e-9 — five to six orders of
  magnitude below the 0.46 % GCI, which is what "the fixed point is the same discrete steady
  state for every dt" (the analytic argument in `docs/en/validation.md` §3.6) needed to be
  measured. `pic_relax = 0.8` **did not converge inside a 655 s budget** (killed, no result):
  consistent with the record's note that the undamped/undersamped iteration oscillates.
- **Residual evidence at the final state** (the audit's F9 request; the harness itself prints
  only `∫div u`):
  - `∫div u dA = −1.8e-13` (baseline), `−1.4e-13` (0.3), `−7.8e-14` (1e4), `−1.2e-13` (1e5),
    `−1.5e-14` (32×384); `div_rel = |∫div u|/(Umax·A) = 3.4e-17 … 4.3e-17` — the pressure-pin
    row is a pure gauge at quadrature round-off.
  - **linear Stokes residual** `LAST_SEMSTOKES_RESIDUAL[] = 7.20e-11` (16×192) / 6.63e-11
    (32×384) — the last flow solve's relative residual.
  - **momentum / fixed-point residual.** At the returned state the stored `u` was solved from
    the previous iterate, so the momentum residual of the state is measured by the next
    iteration's increment: `‖u_{k+1}−u_k‖/‖u‖ = 2.0e-9` (baseline), 1.9e-9 (1e4/1e5), 9.9e-9
    (0.3), 1.1e-9 (32×384); energy increment `‖θ_{k+1}−θ_k‖/‖θ‖` 2.0e-9 (baseline). The
    increments are at the round-off floor of the iteration (their own contraction ratio is
    ≈0.85-0.99, i.e. no longer contracting geometrically), so the fixed-point residual is
    **bounded by ≈1e-8 relative in u and θ**; it cannot be pushed lower in double precision.
    `ΔNu_T` over the last outer step is 2.2e-8 (baseline) — the same statement in the reported
    observable.
  - **Conclusion.** The published annulus scalars are insensitive to the relaxation mode and
    step size at the 1e-7 relative level, and the final state satisfies the discrete system to
    ~1e-8 (u, θ) with `∫div u` at 1e-17 of the velocity scale. F9 is closed for this benchmark:
    the steady-state claim is now backed by a residual statement, not only by an iterate-change
    criterion.
- **Command.**
  `ANN_NX=16 ANN_DT0={1e3,1e4,1e5} ANN_PIC_RELAX={0.3,0.5} ANN_STEPS=12 ANN_STEADY_TOL=0
  ANN_TAG=<tag> julia --project=. --threads=8 out_julia/agent/p2/p2_annulus_probe.jl`
  (and `ANN_NX=32` for the cross-check). The probe reuses `examples/annulus_benchmark.jl`'s
  card, geometry and diagnostics; card hash `examples/annulus_benchmark.jl` sha256
  `6dd7df1ce8e8ce6719338c4a1402c7fa1fc084a3418745bc45fe60fce6215cd8`.
- **Resolution.** 16 x 192 = 3 072 leaves / 28 608 saddle DOFs; 32 x 384 = 12 288 leaves /
  112 512 saddle DOFs; p = 2.
- **Solver path.** `ANN_CFL=0` (relaxation), `maxpic=25`, `linear=:auto`, `ANN_IC=firedrake`,
  fixed 12 outer steps (`ANN_STEADY_TOL=0` so the run cannot stop early).
- **Wall clock.** 191-300 s per run (8 Julia threads, shared host; Julia start-up and the first
  step's 25 Picard sweeps dominate). Budget: 1 200 s watchdog per run.
- **Environment.** Julia 1.12.7, `Manifest.toml` sha256
  `ff64fc791ddb601a3119bbb0ff5a3bba7300a86b12b1f90fece23c7de2a247cc`; Ubuntu 24.04.5 (WSL2);
  i5-13490F, 16 vCPU, 25 GB RAM.
- **Logs.** `logs/annulus_a16_dt1e3_r05.log` (sha256 `d1c3673a559f958b8ae38834b38bc045fbfe9d20213208938a7c3e40d1ac73b2` (1818 bytes)),
  `logs/annulus_a16_dt1e3_r03.log` (sha256 `6e32ff39d0e93609bf68223ab758477fb8a782719d100214cacc5ea874db474a` (1818 bytes)),
  `logs/annulus_a16_dt1e4_r05.log` (sha256 `5812f24dc9deebb0065c0d8e5455afd52103d4bdbccdd2f7cc504f4f372bef52` (1818 bytes)),
  `logs/annulus_a16_dt1e5_r05.log` (sha256 `8ed38fc9ef036ce66f8658dc5aadddf5337247eca279b9e53661a9202249eb18` (1818 bytes)),
  `logs/annulus_a32_dt1e3_r05.log` (sha256 `d2014cec45823a9ab5a43b9350ccd96f9cb1f35e8a051cf8ec51c01919328479` (1795 bytes)),
  `logs/annulus_a16_dt1e3_r08.log` (sha256 `96f7f3b47a367eeae201f7d084bf1b49910d93f9e3d9b4de1e4ac0efbdf0939a` (567 bytes), the non-converged 0.8 run).
- **Status.** `verified` — all settings listed converge to the same fixed point and the
  reproductions of the recorded 32×384 / 16×192 ladder values agree to ≤1e-6.
- **Recorded.** 2026-11-26, batch P2 part 1; the raw log is archived beside this record.
