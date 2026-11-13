# BUG_LEDGER.md: the register of serious defects and the gates that pin them

HEAD at the time of writing: `e790092`. Documentation only; no source file changed with this commit.

This is the document to open first when something breaks. Sections 1 and 2 are the two **bug classes** this codebase produces repeatedly. Sections 3 to 15 are the **concrete bugs**, in the order they
were found: the nine of the first revision (3 to 11), then the four the MF+MG campaign and the 2026-11-13 re-baseline added (12 to 15). Section 16 is the process failures, several of which cost
more wall time than the numerics. Section 17 is the gate and commit index, section 18 the reproduction commands.

Conventions: every number is measured unless marked. `(inference)` marks a statement I reasoned to but did not measure, `[arithmetic]` one derived from measured numbers. Code locations are `file:line` at the fix commit, except in sections 12 to 15,
which were written at `e790092` and quote that tree's lines. Gates are `G<n>` with the test file. The two pinned reference hashes are Blankenbach 2b (20 steps, `theta = d22fdab099ae84e0`, `Nu_top = 7.0299964227238032`) and blob frame 19 (1024 leaves, `theta = 45ac8acd099adadc`, `Nu_top = 5.9916941925454079`). Both are AMR-free (`amr_every = 0`) and reproduce bitwise under the current defaults; the classic-path pins below are reproduced by passing `amr_policy = :classic`, because since 2026-11-13 the library defaults of `run_semquad_simulation` are `:balanced` for the mesh policy (`4584e85`) and `:auto` for the linear solver (`2453f16`). The blob digest quoted below is the end-of-run `theta` at step 126, `868a79d9...458afe88`; frame 25 has its own digest, `dd364f70...5aebfecb`, and §14 records the difference.

**Run artifacts.** The `out_julia/...` paths in this document name run artifacts — logs, checkpoints, frame dumps
and audit scripts — from the development tree. They are cited so that a measured number can be traced to the record
that produced it; they are not part of this repository and are not shipped with it. The protocol in section 18 is
kept as the record of how each audit was run, not as a recipe that runs from this checkout.

---

## 0. The index

| # | defect | class | fix | gate |
|---|---|---|---|---|
| 1 | `Union{Nothing,Function}` / `Function` fields in hot loops | S15 | `74494d1`, `96e52a5`, `17d0e17` | G185, G202, G215 |
| 2 | loops that run without progress | S16 | `4bbdae9`, `96e52a5`, `17d0e17` | G186, G203, G215 |
| 3 | Newton: iteration-cheap, wall-expensive | algorithm | `96e52a5` | G202, G203 |
| 4 | `maxpic > 1` passed the iterate as the BDF2 `theta^n` | scheme | `86d03f6` | G193 |
| 5 | the AMR freeze | policy | `117cc9d`, `0b3f4bd`, `98554dc`, `75e629c` | G210, G211 |
| 6 | the mirror invariant | invariant | `398b245`, `e91beaf` | G204, G208 |
| 7 | the load-bearing blocked-key quirk | history | kept in `957a43b` | G211, G216 |
| 8 | the Nusselt definition and the closure column | measurement | `266a656`, `c95d596` | identities, `resid_A` |
| 9 | the near-null pressure space | discretisation | `aceeeac` | G214 |
| 10 | `STOKES-FAIL`: shift ladder and trigger | solver | `dea9718`, `5c54870`, `b23483f`, `8633073`, `16fdfa3`, `064747d` | G207, G209, G212, G213 |
| 11 | the constrained-Jacobi `1e12` amplifier (velocity smoother) | MG | `9250a3c` | G218, G226; the stage gates G219-G225 |
| 12 | the BFBT lumped diagonal: 16 non-positive entries, `1e12` contrast | MG | `503fd87` | G226 |
| 13 | the per-call costs: a 640 B kernel allocation and a 6-9 us fork that costs 100 us in place | MG | `ecc1c5c`, `2b3c2d3`, `262e7eb` | G222, G223, G224 |
| 14 | the `868a79` digest recorded without its object (end of run; frame 25 has its own) | measurement | `2b016a2` | G229; `frame_sha256.csv`, `run_summary.csv` |
| 15 | `run_config` blind to keys added after a checkpoint was written | history | `c7e0f55` | n/a (round trip in the commit; G229 writes the key) |
| 16 | process failures | process | `11e7744`, `fb14a08`, `46d0c2a`, `015b6f7` | n/a |

G219-G230 are the twelve gates the campaign and the default flips added, and §17 states what each one asserts: G219 the production defaults, G220 the reduced assembly, G221 the level cache, G222 the 32-lane kernel and its 640 B, G223 the region
V-cycle, G224 the region phases, G225 the eta lane table, G226 the pressure diagonal, G227 the `S1` inner solve, G228 the card's `mg_tol`, G229 the balanced default, G230 the auto default. The MG rows lean on
them: G218 is the velocity diagonal, G226 the pressure diagonal of the same clip class, G222-G225 the stages that made the repaired hierarchy measurable, and G228-G230 the tolerance and the two defaults the
production numbers are quoted under.

---

## 1. Class S15: type instability and boxing in hot loops

### 1.1 Symptom

Right answers, several times slower than the arithmetic justifies, with no error and no warning. First instance: the quadtree energy assembly on the blob tree (1024 leaves, `p = 4`) took `2.492 s` and
allocated `5.029 GiB` per step, a `4.552 s` frame-19 step. The arithmetic is one element matrix per leaf plus a `sparse` conversion, and the same code on comparable uniform trees was several times
faster. The profile landed on `_semquad_assemble_energy_block!`, where every line was a `push!` or a scalar multiply.

### 1.2 The pattern

Julia specialises on argument types and type parameters, never on a struct field's declared type. A field declared `Function`, or `Union{Nothing,Function}`, turns every read into a dynamic call
inferred `Any`, and every arithmetic operation downstream is boxed and dynamically dispatched, once per quadrature point per component. Five shapes appeared:

1. **Function-typed field.** `SemQuadAsmCfg.geo_coef :: Union{Nothing,Function}` (and `dp_coef`). Then `gc = d_a * deta_q / edot` is `Any`, and the four `Kgloc[...] += gc * ...` updates per `(k,l)`
  pair are boxed multiplies: 324 per quadrature point, 331 776 per assembly (`semquadstokes.jl:1923-1926`).
2. **`Function` field on the default path.** `SemRefProfile.alpha_fun`/`cp_fun` make `_prof_lambda :: Any`, and so `_prof_dT`, `_prof_d2T`, `semquad_Tad`, `semquad_base_gradient` and the energy
  kernel's `src`. Because `semquad_base_gradient` branches three ways on `pr.refstate`, the `Any` poisons the default Picard path (`refstate === nothing`), not only the EOS path.
3. **Closure over a reassigned variable (`Core.Box`).** `_fund_live!` was a closure in `amr_cycle_balanced!` capturing `nc`, `gate_extended`, `tau_r`. Julia lowers a captured variable assigned in the
  enclosing scope to a heap cell, so every read in the funding scan is an `Any` load. `@code_warntype` reported exactly 3 `Core.Box` entries; after hoisting, 0.
4. **Abstract container and per-call allocation in a scan.** `_budget_quartet` built a `2x2 Matrix{NTuple{3,Int}}` per leaf per scan; `_coarsening_allowed` built `Set(vec(kids))` plus four
  `vec(kids[:, d])` slices per call. `QuadFaceCache` did worse: every `refine!`/`coarsen!` ran `empty!` on a 10 000 to 40 000 entry `Dict` (0.022 ms at 10 k, 0.094 ms at 40 k), and the gate-on cycle
  performs about 570 mutations, i.e. 54 ms/cycle of pure cache clearing.
5. **Abstract argument and repeated sorting.** `_coarsen_plan`'s `lmin::Function` is an abstract argument (one dynamic dispatch per candidate), and the function re-sorted the whole leaf set on every
  call (1.18 MB, most of 6.26 ms) while rebuilding its sweep permutation inside the fixed-point loop.

### 1.3 The instances and their measured costs

| instance | before | after | ratio |
|---|---|---|---|
| `rho_ref::Union{Nothing,Function}`, energy element loop | 2.492 s, 5.029 GiB per assembly | 0.137 s, 45 MB | 18.2x, 112x |
| same fix at the step level (blob frame 19) | 4.552 s/step | 1.511 s/step | 3.01x |
| `geo_coef`/`dp_coef` fields | 17.49 ms Stokes assembly, 336 MB / 3 Newton steps | 3.30 ms, not in the top-25 | 5.30x (`Kgeo` branch 15.04 → 0.48 ms, 31x) |
| `SemRefProfile.alpha_fun`/`cp_fun` | 0.7425 ms, 1.858 MB per energy assembly | 0.3958 ms, 1.137 MB | 1.88x |
| — same on the 25-frame blob run (126 steps) | 18.72 s energy segment | 14.12 s | 1.33x (−4.6 s, 3.9 %) |
| `_coarsening_allowed` `Set` + slices | 864 B / 200 ns per quartet | 0 B / 78–81 ns | 2.6x |
| `_budget_quartet` 2x2 `Matrix` | 2 021 888 B / 1.98 ms per scan | 0 B / 0.67 ms | 3.0x; 108.24 MB / 4 cycles = 27 MB/cycle |
| `_coarsen_plan` re-sorting | 1 163 088 B / 6.26 ms | 442 872 B / 1.63 ms | 3.8x |
| `_fund_live!` `Core.Box` | 3 boxes, `Any` loads | 0 boxes, 0 `Any` in `amr_cycle_balanced!` | 0.163 → 0.067 s/cycle with the cache fix |
| face-cache `empty!` | 0.022–0.094 ms per mutation, ~570/cycle | one integer compare | ~54 ms/cycle removed |
| abstract `lmin::Function` | one dispatch per candidate | concrete | not separately timed |

The energy fix is the cleanest A/B in the ledger because it changed one expression: `rho = (rho_ref === nothing ? 1.0 : rho_ref(y))::Float64`, the same for `eta_q`, `rho_ad`, and the
`theta::Vector{Float64}` argument of `_semquad_eval_at`. No arithmetic changed, and the trajectory is bitwise: the 22-step blob slice keeps its `theta`/`(ux,uz)` SHA-256, the 25-frame classic-era run keeps
`868a79d9...458afe88` (the classic arm, bitwise re-verified 2026-11-13; the current balanced default is a separate
baseline, `a4fb8248...b8cdeda`, `docs/VALIDATION.md` §2.3.1), and G185 pins `sum(M)`, `sum(abs,K)`, `norm(C)`, `norm(F)` and the BDF2 solution against the pre-fix worktree.

### 1.4 The audit method

1. **`@code_warntype` with concrete argument types.** Keywords through `Core.kwcall`, the real loop body through the internal method (e.g. `#semquad_step_newton#350`). A warntype on an `Any` argument
  is a probe artefact; the Newton audit records one such false report (`semquad_steady_jv`) rather than acting on it.
2. **JET `@report_opt` in a separate environment** (`Pkg.develop(path="julia")` in `/home/szh/jetenv`, project file untouched). On the Newton path, 380 runtime dispatches, all of the boxed-closure
  family, plus one item warntype does not show: a dynamic kwarg entry on `assemble_semquad_energy`, 16 times per step. On the fixed AMR path, 0 reports for `_coarsen_plan`, `amr_cycle_balanced!`,
  `semquad_enforce_leaf_budget!` and `_coarsening_allowed`.
3. **Allocation attribution.** Julia 1.12 removed `--track-allocation`'s `clear_mallocdata`, so the Newton audit used `Profile.Allocs` at 1 % sampling (326 413 samples over 3 steps, 830.8 MB). The AMR
  audit used `--track-allocation=user`, which costs nothing in trajectory (same leaf and `theta` FNV) and about 35x in time (15.7 s/cycle), which is why the AMR pass, not the 10-step run, is what gets
  instrumented.
4. **A `git archive` baseline in the same session, with the working tree's `Manifest.toml` copied in.** NEWTON: `git archive 8b08af3` into `/home/szh/ms_base`. AMR: `git archive 0b3f4bd` into
  `/home/szh/ms_baseline_line`. S3 stage 2: `/tmp/ms_base` at `23f667e`. Without the manifest copy the package versions differ and the A/B is not an A/B.

### 1.5 What the numbers say

Allocation is not time, twice over. The four `Kgloc` lines were 336 MB of 831 MB (40.4 %) and about 30 % of the step. The `lagrange_eval`/`lagrange_values` scratch is 78 MB per 3 steps (9.4 %) and
about 0.3 us of each 0.9 us pointwise evaluation, under 2 % of the post-fix step, and was not fixed. A 40 % allocation share bought a 30 % time share; a 9 % share buys 2 %. The second result is that a
small allocation can be the whole loop: `_budget_quartet`'s matrix was 15 % of the AMR pass's bytes and 3.0x of its own scan, and the scan is 83 % of the hard budget pass.

### 1.6 The rule

`Union{Nothing,X}` on an **argument** is harmless: `typeassert` narrows it through `typeintersect` and the compiler specialises. A `Union{Nothing,Function}` on a **field** is not: it costs a dynamic
call and a boxed result at every use. Fix the field with a type parameter (`struct SemQuadAsmCfg{G,D}`), fix the closure by hoisting it to a top-level function returning its results, fix the abstract
argument by passing concrete data. The gate for a de-boxing change is a bitwise kernel comparison, not a timing threshold: G202 compares `Kgeo` and `Kp` entry by entry against a verbatim copy of the
pre-fix loop, pins SHA-256 of `K, B, Kgeo, Kp, Kred, fv` recorded in the pre-fix worktree, and asserts `geo_coef`'s field type is the concrete closure type (that assertion fails before the fix). 18
assertions. G185 does the same for the energy assembly.

---

## 2. Class S16: loops that keep running without progress

### 2.1 Symptom

A step takes 31.6 s where its neighbours take 32 ms, deterministically, at one leaf count. A Newton step burns 1.15 s on a linear solve whose best achievable residual is 3.3e-2 against a 1e-4
acceptance, then discards the direction. An AMR pass stops changing the mesh for 3000 cycles while the boundary layer it is meant to resolve thins. The signature is a loop whose exit is a counter that
increments while the state does not change. It is not hung; it is making work.

### 2.2 The four questions

Every loop in the audited files was asked: (1) what changes per iteration (the carrier: `it`, `alpha`, `cursor`, `granted`, the leaf set, `x`); (2) what bounds it (a cap, an invariant, a budget); (3)
whether the exit is **proved**, i.e. whether the carrier changes on every iteration that is not the last, because if it can return to its previous value the cap is the only exit and the cap is 10 000;
(4) whether a fired cap is recorded, since a silent cap converts a bug into a performance mystery.

### 2.3 The instances

| loop | was wrong | measured |
|---|---|---|
| GMRES restart (`gmres.jl:143`) | `maxit` honoured only at restart granularity: `m = restart` per cycle, so `krylov=80, restart=30` used `ceil(80/30)*30 = 90` vectors | 450 vectors per failing step (5 `Kgeo` attempts × 90), 1.148 s |
| GMRES stagnation | a stalling preconditioner burned the whole budget; everything past 30 vectors per attempt was waste | best `rel` any attempt reached 3.3e-2 against 1e-4; guard on: 300 vectors, 0.994 s, bitwise same outcome; budget 30: 0.49 s, same outcome |
| GMRES lucky breakdown | `den = hypot(H[k,k], H[k+1,k]) = 0` returned `rel = NaN` | one comparison per Arnoldi step; G203 returns finite `rel` and `x` |
| step-95 budget guard (`dgquad.jl`) | after coarsen + `balance!` the leaf set returned bitwise identical, so the next round sorted to the same lowest-`eta` quartet; `guard` was the only exit | 10 000 rounds, 10 001 `balance!` calls, 82 118 182 `neighbors`, 2 124 801 774 `_collect_face!`; 31.637 s → 0.0595 s (531x) after fixed-point detection |
| `_coarsen_plan` outer walk | bounded only by `limit >= n`, so a tree whose legal supply is below `need` cost about `n/target` rounds × `O(limit)` lookups each = `O(n^2/target)` | capped at `MAX_PLAN_ROUNDS = 16`; measured 1–3 rounds |
| `_coarsen_plan` inner sweep | monotone `granted` set but no enforced bound | capped at `MAX_PLAN_SWEEPS = 64`; measured 2–5 sweeps |
| `mirror_close!` | success flag returned and dropped by every caller | `LAST_MIRROR_CLOSE` (`refined`, `ok`, `rounds`, `capped`); caller records `mirror_ok` |
| `balance!` | return value dropped | recorded as `balance_ok` in `LAST_AMR_BALANCED` |
| isostress scalar Newton (`physics.jl:582`) | audited, not changed: 1024 quadrature points, max 6 iterations, mean 4.48, 0 cap hits | no guard needed; the `converged` flag is returned and never inspected |
| Newton → Picard fallback (`semquadconvection.jl:1096`) | audited, not changed: the Newton is called only at `it == 1` and never retried | cannot oscillate |

Two adjacent defects were fixed: `semquad_step_newton` returned `alpha = alpha_last` with `alpha_last` initialised to `1.0` and never updated (the accepted `alpha` was a loop local), so the diagnostic
always printed `alpha = 1` even when the line search halved to 0.25; and the verbose print of a damped continuation attempt omitted the Krylov count, which is what exposed the 90-vector burn once it
was added.

### 2.4 Detection and counters

The step-95 loop was found with a temporary counter recording, per AMR cycle, the guard iterations, the leaf count in and out, the `balance!` call and pass counts, and the `neighbors` count. The
decisive row: guard iterations 10 000, leaves 1030/1030, leaf set bitwise identical. The loop was idling, not converging slowly. A synthetic 1264-leaf tree reproduced 3.29 ms/round, 32.9 s for 10 000
rounds, confirming the mechanism without the production state. The detector generalises: record the state that is supposed to change, per iteration, and look for repeats. That is also how the GMRES
budget was found, since the diagnostic line showed 80, 80, 80, 80, 80 vectors on a failing step.

Counters added: `LAST_AMR_BALANCED` (`plan_sweeps`, `plan_rounds`, `plan_capped`, `fund_calls`, `fund_hits`, `balance_ok`, `mirror_ok`, `mirror_rounds`), `LAST_BUDGET_PASS` (`merged`, `scans`,
`guard`, `capped`), `LAST_MIRROR_CLOSE` (`refined`, `ok`, `rounds`, `capped`), `LAST_SEMSTOKES_LADDER_PHASE` (`:healthy`, `:tiny`, `:ladder`, `:refine`, `:fallback`, `:fail`). The two plan caps are
structural: when either fires, the plan is the prefix found so far, the cycle merges fewer quartets, and the hard budget pass (with its own `blocked` set) does the rest. Neither fires on the
production trees measured here; G215 pins them with a uniform 1024-leaf tree whose every quartet is illegal (`rounds == MAX_PLAN_ROUNDS`, `capped == true`, `refused > 0`).

### 2.5 The rule

A loop must name its progress measure and its exit. A cap that can fire must be recorded in a counter a gate can read. "Bounded by `maxit`" bounds the damage; it is not a proof of progress.

---

## 3. The Newton path: an efficient iteration with an expensive linear solve

### 3.1 What the path is, and what the audit found

`pic_mode = :newton` runs one Picard sweep, then a damped Newton-Krylov on the coupled residual `(r_S, r_C, r_E)`: Stokes momentum and continuity plus the energy equation. Each iteration assembles a
Jacobian-vector product (`semquad_step_jvp`), solves a reduced saddle with a preconditioner that drops the `theta`–`u` coupling, and does a line search with `Kgeo` continuation over `s in (1, 0.5, 0.25, 0.1, 0)`.

One Newton step at `Ra = 1e6`, 64 leaves, composite `:isostress`, 3 iterations, was 0.1981 s (median 0.2006). The phase split: residual evaluation 37 % (Stokes assembly 34 % of the step, of which the
`Kgeo` block was 15.04 ms of 16.3 ms), GMRES 52 %. A micro-benchmark of the `Kgeo` loop shape ran at 330 ns per quadrature point, so the 12.7 us the kernel spent was 38x the arithmetic; the cause was
the boxed `gc` of §1.2. After the two S15 fixes and the three S16 guards: 0.1021 s (median 0.1096), 1.94x, the fixed point bitwise unchanged (`change = 2.16668644868966926e-14` before and after,
history `3.2e-03 5.1e-07 3.4e-10 2.2e-14`, 3 iterations both times), allocations 830.8 → 428.9 MB. Implementation waste removed: 86 ms of 187, 46 %.

### 3.2 The lagged-factorization work

`linear = :lagged` (`68c5896`, G174) is the other answer to the same cost: cache the equilibrated `D S D` factorisation keyed by a tree fingerprint, warm-start GMRES against the current saddle, accept
on the true residual, refactor only on a cache miss or a stall. Measured: 2b L3 0.661 → 0.293 s/step (2.25x), blob `Ra = 1e8`, `p = 4` AMR 266 → 302.7 s (0.88x, a regression). At high viscosity
contrast the stale factor is too weak a preconditioner and the extra Krylov iterations cost more than the factorisation saved. The work stands as a load-bearing negative result: "avoid the
factorisation" is not the way to make Newton cheap.

### 3.3 Line search and `Kgeo` continuation

The line search is bounded: at most 8 halvings per continuation value and 5 values, so at most 40 residual evaluations (0.16 s) before the step declares failure and the driver falls back to Picard.
Measured: 1 evaluation per iteration on the healthy case (`alpha = 1`), 3 at the first iteration of the failing case (`alpha = 0.25`). The continuation multiplies GMRES waste, since each value costs
one LU plus one GMRES run: 5 attempts × 90 vectors = 450 in the failing step of §2.3. Verdict: the line search is algorithmic and correctly bounded, the continuation is algorithmic and correct, and
the waste was in the GMRES budget.

### 3.4 Rejected optimisations, with numbers

| candidate | measured | why not taken |
|---|---|---|
| de-box the Newton loop's closures (`let` or callable structs) | 30.2 → 27.3 ms GMRES, 1.11x before the kernel fixes; **0.94x** after | 1.11x on GMRES is 8.6 ms of 188 ms = 4.6 %, under the 5 % rule; after the kernel fixes the dispatch was never the bottleneck |
| skip the energy assembly's `M`/`K` in the JVP and preconditioner (they keep `C`, `F`) | ~4 ms = 4 % of the post-fix step | under the 5 % rule; needs a new kernel and a G202-style bitwise gate |
| memoise the pointwise rheology (`eta_fun`, `fz`, `deta_eps` recompute `T`, `|edot|`) | 2.2 ms per assembly = 67 % of the post-fix assembly, 8.8 ms/step = 8.6 %; a one-point memo cuts ~2/3 = 5.5 % | stateful cache whose hit rate depends on call order; the honest fix is an element-local evaluator |
| scratch buffers for `lagrange_eval` | 78 MB / 3 steps (9.4 % of bytes), ~0.3 us of each 0.9 us evaluation | under 2 % of the step, and it is a shared helper used by every solver |
| JET's 380 reports | boxed-closure family plus one dynamic kwarg entry per energy assembly | the kwarg removal is a design change; the rest is the 0.94x family |
| `_prof_phi` (anelastic mass form) | not measured | only reached with an EOS reference state; changing it blind is the speculative edit the rules exclude |

### 3.5 The L5 A/B: iteration-efficient, wall-expensive

Same case (`Ra = 1e6`, `:isostress`, `dt = 1e-5`, `linear = :direct`, `pic_tol = 1e-6`), 5 steps, one process per configuration, one untimed warm-up:

| quantity | 64 leaves | 1024 leaves (`krylov = 80`) | 1024 leaves (`krylov = 300`) |
|---|---|---|---|
| Picard `maxpic = 1` per step | 22.0 ms | 266 ms | 266 ms |
| Picard `maxpic = 20` iterations (5 steps) | 28 = 8/5/5/5/5 | 28 = 8/5/5/5/5 | 28 = 8/5/5/5/5 |
| Picard `maxpic = 20` per step | 60.8 ms | 1.31 s | 1.30 s |
| one Picard sweep | 10.9 ms | 233 ms | 233 ms |
| Newton iterations (5 steps) | 11 = 3/2/2/2/2 | 10 = **1 (fallback)**/3/2/2/2 | 11 = 3/2/2/2/2 |
| Newton per step | 113 ms | 16.1 s (first 37.1 s) | 34.9 s |
| one Newton iteration | 51.5 ms | 4.8 s | 15.8 s |
| Newton / Picard `maxpic = 20` | 1.87x | **12.3x** | **26.9x** |
| Newton / production `maxpic = 1` | 5.15x | **60.5x** | **131x** |
| `||dtheta||/||theta||` Picard20 vs Newton | 2.0e-6 | 1.10e-3 (fallback-contaminated) | **1.89e-6** |

The iteration counts favour Newton at both sizes: 28 Picard against 11 Newton at 64 leaves, 28 against 9 plus one fallback at 1024. The wall time does not. The phase split of one L5 Newton iteration:
GMRES with 80 vectors is 96 % of it (4.28 s of 4.44 s), of which the JVP is 45 %, the preconditioner 17 %, and bookkeeping/allocation 34 % (53.6 ms per vector). From 64 to 1024 leaves the cost per
vector grows 14x, in line with the 14.6x dof growth, and the vector count grows 10x (8 → 80). The second factor is algorithmic: the block-triangular preconditioner drops the `theta`–`u` coupling and
its iteration count degrades with resolution.

The shipped budget makes it worse: at 80 vectors the last linear solve of step 1 reaches only 3.2e-4 against the 1e-4 acceptance, all five `Kgeo` continuation values are rejected, the line search
fails, and the step falls back to Picard after 37.1 s. With 300 vectors the same step converges in 3 iterations (203/300/300 vectors, `rel` 9.9e-9/1.3e-8/4.9e-8) in 44.9 s, still 34x one
Picard-to-tolerance step (1.31 s). So the fallback is a budget problem and the 26.9x is an algorithm problem, a distinction the audit is what makes measurable.

**Status.** Production stays on `maxpic = 1`; Newton is the tool for small-size iteration accuracy (1.87x the cost of Picard-to-tolerance for the same fixed point, and a cleaner one) until the
preconditioner is replaced. Removing the remaining 34 % bookkeeping would still leave one L5 iteration at ~2.9 s, 12x a Picard sweep, so the conclusion does not depend on it.

### 3.6 The lesson

An iteration count is not a cost. The audit removed 46 % of the step before the A/B was allowed to be quoted, because before that the A/B measured the boxes, and the result is stable under the two
remaining implementation items, which is the property a positioning decision needs.

---

## 4. `maxpic > 1`: the iterate passed as the BDF2 `theta^n`

### 4.1 Symptom and discovery

Nothing on the production path. The defect was invisible at `maxpic = 1` and appeared only as confusing `maxpic > 1` measurements: "more iterations is worse", benchmark overshoot, results that did not
improve when the nonlinear solve was tightened. The natural reading was divergence or a wrong rheology; the actual reading is that the iteration converged to a different equation. It was found while
adding per-step globalization (`pic_mode = :newton`, G192/G193): the fixed point of the existing `maxpic > 1` loop was checked against the BDF2 step operator instead of against the previous iterate,
and it failed.

### 4.2 The mechanism

The BDF2 energy step is

```
A theta = rhs,   A = (a/dt) M + K + C,
rhs = (a/dt) M theta_n - (b/dt) M theta_{n-1} + (c/dt) M theta_{n-2} + F
```

with `M` the mass matrix, `K` diffusion, `C` skew advection, `theta_n` the current unknown, `theta_{n-1}`, `theta_{n-2}` the previous two levels, `dt` the step, `dt_old` the previous step, `w = dt/dt_old`, and BDF2 weights `a = (2w+1)/(1+w)`, `b = 1+w`, `c = w^2/(1+w)`; at constant `dt` they are `3/2, 2, 1/2`.

The iterated loop put the **running iterate** in the `theta_n` slot. With `theta^(k)` the iterate, it solved

```
(a/dt) M theta + K theta + C theta = (a/dt) M theta^(k) - (b/dt) M theta_{n-1} + (c/dt) M theta_{n-2} + F
```

and iterated. At the fixed point `theta = theta^(k)` the `(a/dt) M` terms cancel, leaving `K theta + C theta = -(b/dt) M theta_{n-1} + (c/dt) M theta_{n-2} + F`, which is the steady energy equation
with a frozen source. On the first step the mass terms cancel completely and the loop converges to that equation, as `86d03f6` states. A `change`-based test cannot see this: the iteration does
converge, to the wrong operator, and drives the wrong residual to zero.

### 4.3 The fix and the docs-only follow-up

`86d03f6` passes `theta_old` (the true `theta^n`) and keeps the iterate out of the time-level slots. With one sweep there is no iterate distinct from `theta^n`, so the two forms are the same
expression, which is why the production path never moved and why the pinned hashes are the evidence. `maxpic > 1` changes by design and is now a converged implicit step; G193's second test asserts the
backward-Euler residual is below `1e-8` **and** that the steady equation does not hold. The old driver loop (`run_simulation` in `julia/src/timestep.jl`) always passed `theta_n`, so it was never
affected at any `maxpic`.

A numerics fix that invalidates historical comparisons is not finished when the code is right. `0330edc` added `docs/BUG_LEDGER.md` (914 lines) and the correction table in `docs/PHYSICS.md` §12:
877 matched lines across tests, scripts, scratch analysis and docs, of which 83 are AFFECTED (run at `maxpic > 1` before `86d03f6`), 779 UNAFFECTED, 15 NEEDS-RERUN, each with an explicit disposition.
`1998f60` then marked one historical claim unverified because it does not reproduce at HEAD: a composite 20-step `Vrms` claim of `6.99e1 -> 6.00e3` gives ALA `3.15e2` / TALA `4.15e2` at `maxpic = 1`
and `1.54e3` / `2.05e3` at `maxpic = 20`, with no recorded `dt` or `maxpic`.

### 4.4 The lesson

A nonlinear iteration's fixed point must be checked against the scheme's residual, not against its own change. A scheme fix has a documentation blast radius, and the inventory is part of the fix. And
"more iterations is worse" is never a rheology result until the fixed point has been verified.

---

## 5. The AMR freeze: three mechanisms, one policy

### 5.1 Symptom

Production run E (`CASE_MAXLEAVES=10000`, `CASE_COARSEN=0.05`, `CASE_THETATOL=1e-2`, `CASE_WALLMIN=5`, `CASE_WALLSTRIP=0.1`, `linear = :direct`) sat at 9994 of 10000 leaves with an effectively frozen
mesh: 3720 AMR cycles, 723 with any change (19.4 %), the leaf count oscillating 9988–9994 (one mirror pair), last structural change at step 17225. The top boundary layer was thinning at the same time
(`delta_top` 0.157 at `Nu_top ~ 9` to 0.073 at `Nu_top ~ 18.5`), so the frozen mesh was becoming the accuracy limit of the one quantity that agreed with ASPECT to 1 %.

### 5.2 Mechanism (a): an absolute tolerance protecting 99 % of the tree

`coarsen_theta_tol = 1e-2` compares the per-leaf theta range `max(theta[leaf]) - min(theta[leaf])`. In a convecting field that range is never small: minimum 4.500e-3, p25 2.037e-2, median 3.092e-2,
p75 5.806e-2, p95 1.854e-1, maximum 3.003e-1. **9892 of 9994 leaves (99.0 %) exceed the tolerance** and enter the protect set; only 102 are below it, and no leaf has a range below `1e-3` or exactly
zero. The threshold is a blanket, not a representation-error guard. The `symmetry = :x` closure makes it worse: `_mirror_protect` adds the partner of every protected leaf (69 leaves whose partner was
not already protected by itself), so the effective set is 9980 of 9994 (99.9 %) and 14 leaves are coarsenable. A sibling quartet needs four of them.

### 5.3 Mechanism (a'): refused leaves are charged to the coarsening budget

`amr_cycle!`'s coarsen loop (`julia/src/dgquad.jl:433-445`) charges every walked leaf:

```julia
nc += _coarsen_pair!(qt, lf, symmetry, protect)   # 0 when refused
cum_c += max(eta[lf], 0.0)                        # charged even when refused
```

A protected leaf spends budget without freeing a slot, and a quartet merge is decided by one child's `eta` instead of the quartet's maximum. Measured ledger on the E tree against `0.05 * total`:
protected leaves **94.8 %** (raw indicator 94.2 %), incomplete quartets 5.2 %, mergeable free quartets **0.0 %**. The loop walks 5171 leaves, 39.5 % of them the zero-eta wall strip, and makes zero
merges. The accounting is per leaf; the operation is per quartet.

### 5.4 Mechanism (b): the budget guard breaks before the first candidate

At 9994 leaves the guard `length(qt.leaves) + 10 > max_leaves` (10004 > 10000) is true before the first candidate, so the refinement loop performs zero iterations and the cycle reports `refined = 0`.
The Dörfler `refine_frac = 0.2` of the card error is 82 leaves (+492, to 10486), and only one mirror pair (6 leaves) fits even with the `+10` slack removed. The driver reinforces the freeze with `rf = length(qt.leaves) < max_leaves ? refine_frac : 0.0` (`julia/src/semquadconvection.jl:1532`), which pins `refine_frac = 0` at the cap and then runs a cycle that may not coarsen either. `amr_cycle!`
cannot trade; it can only break.

### 5.5 Mechanism (c): the wall strip is protected and de-signalled

`wall_strip = 0.1` protects every leaf with `zc <= 0.1` or `zc >= 0.9` plus the two wall rows: 2040 leaves, 20.4 % of the tree, carrying 60.6 % of the raw indicator. Because `wall_min_level > 0`
zeroes the strip's indicator for refinement, the strip is protected from coarsening and removed from refinement at once, and it accounts for 39.5 % of the coarsening walk's steps.

### 5.6 The counterfactual and the redesign

Two controls on a copy of the same tree, same indicator, same `coarsen_frac = 0.05`. Wall rows only as the protect set (no `coarsen_theta_tol`): the from-below walk merges 1110 quartets, 9994 → 6664
leaves (6688 after `balance!` and `mirror_close!`), net −3306, removing 1.5 % of the indicator. Dörfler above 0.2 is 82 leaves on the card indicator. The low end of the distribution is flat: 5 % of
the error lives in 5513 leaves, 55.2 % of the cells.

The redesign is `amr_policy = :balanced` (`117cc9d`): two cumulative-error thresholds `tau_r` (from above, `refine_frac * total`) and `tau_c` (from below, `coarsen_frac * total`), a hysteresis `tau_h = min(tau_c, 0.5 tau_r)`, a quartet rule (`max(eta_child) <= tau_h`, four leaves, none protected, `level > l_min`), budget exchange instead of a break at `max_leaves`, and an `AmrCooldown`
(`no_refine` for a freshly merged quartet, `no_coarsen` for a freshly split leaf) so the two passes cannot undo each other. Measured, 200 steps from step 18800, same card, only `CASE_AMR` differs:

| quantity | `:classic` | `:balanced` |
|---|---|---|
| leaves | 9994 constant | 9994 → 10000 |
| AMR cycles with any change | 0 of 40 | 40 of 40 |
| refined / coarsened (total) | 0 / 0 | 1390 / 1388 |
| top wall row | L5 = 160 | L6 = 320 |
| `h/delta_top` | 0.213 | **0.106** |
| `Nu_top` (last 40 steps) | 18.326732 | 18.314807 (−0.065 %) |
| wall clock | 1076.8 s | 1123.1 s (+4.3 %) |
| AMR cycle cost | 0.24 s/cycle | 0.31 s/cycle (+0.25 %/step) |

The +4.3 % is not the policy: `N` grows 0.09 % and `eta_clip_frac` doubles (0.0076 → 0.0154) because the refined cold lid puts more nodes on the clamp. The policy's bookkeeping is +0.07 s/cycle,
+0.014 s/step. The trade saturates after about 8 cycles, and the measurement says why: 2932 of 10000 leaves are below `tau_h` and 481 quartets pass the gate; 80 touch a protected wall row and 160 have
a child at or below `wall_min_level`, leaving 241 candidates; of those, only 4 have a mirror quartet also below the gate. The temperature field is not mirror-symmetric (only the tree is), so the
pair-strict rule of §6 caps the coarsening supply by 60x and the cycle settles at 4 leaves/cycle.

### 5.7 The coarsening gate, and the lesson

The second stage adds a 2:1 pre-check (`coarsening_allowed`, `98554dc`) before a merge, a plan computed before any mutation (`_coarsen_plan`), and a supply rule that extends past `tau_h` when the
deadband cannot pay for the refinement. Measured on the operating tree: merge-undo recirculation falls from 100 % of 14 merges per cycle to 1.0 % of 4881, the exchange grows 29x in the 12-cycle replay
and 6.6x over 200 steps, at +18 % of the AMR pass (+0.03 s/step, +0.6 %). Over 200 steps the recirculation rate falls 54 % → 3.7 % and the absolute count falls 1251 → 576 leaves, missing a `<300`
absolute target because the exchange is 6.6x larger.

A mesh policy is a control loop, and the first measurement must be the actuator's authority. Here it was 14 coarsenable leaves out of 9994, so no controller tuning could have moved the mesh. An
absolute threshold on a non-stationary field becomes a blanket protect set; a guard that can only break turns a cap into a freeze. Both are invisible in the log unless the counts are printed.

---

## 6. The mirror defect: a formula, one-sided merges, and a count

### 6.1 The invariant and the formula defect

`symmetry = :x` promises the forest stays closed under `x -> Lx - x`: for every leaf `(l, ix, iy)` the partner `(l, nx0 * 2^l - 1 - ix, iy)` is a leaf, with `l` the level, `ix`, `iy` integer
coordinates at that level, and `nx0 = cells_x` the number of base cells in `x`. The violation count is now an API, `mirror_defect(qt)`, 0 meaning closed.

`_mirror_leaf` mirrored about `2^l - 1 - ix` instead of `cells_x * 2^l - 1 - ix` (`398b245`). On a `cells_x = 5` forest (`2.5 x 1`, the 2b geometry and the realistic case) the partner index is wrong
and negative for `ix < cells_x`; a negative or absent partner made the mirror silently no-op, so the mesh stayed asymmetric while every call reported success. With `cells_x = 1` the two formulas
coincide, which is why the unit-box tests never saw it. The fix uses `nside_x(qt, l)` and passes `qt` at every call site; `cells_x = 1` is bitwise unchanged by construction. G204 (217 assertions)
covers `cells_x = 1/2/5`, involution, range, a refined pair in the `2.5 x 1` tree, and the unit-box identity.

### 6.2 The one-sided coarsening pass

The marking pass and `_coarsen_pair!` were written for pairs, but four operations acted on one leaf at a time. Measured on a closed 910-leaf `cells_x = 5` tree:

| operation | what it did | measured |
|---|---|---|
| `_coarsen_pair!` | merged the `lf` quartet when the mirror quartet was in `protect`, and `protect` comes from the theta field (`coarsen_theta_tol`), which is only approximately mirror-symmetric | 0 → 81 non-mirrored in one pass at 30 % protect; `balance!` healed it to 53 |
| `semquad_enforce_leaf_budget!` | same, when the mirror quartet was protected or incomplete | 0 → 5 and 0 → 10 |
| `amr_cycle!`'s hard-budget guard | same, one quartet at a time | 120 → 230 in one production pass |
| `balance!` | split a single leaf for the 2:1 condition | not the first source (the badness test is mirror-invariant on a closed tree, 24 paired-refinement experiments at `cells_x = 1` and 5), but it cannot stop a defect from spreading |

One one-sided merge leaves five non-mirrored leaves: the new parent (its mirror parent is still split) plus the four children of the mirror parent (their mirrors are gone). That is why the realistic
counts move in fives: 0 at frame 161 (step ~805), 80 by step 6016, 120 in `pre_step6131.h5` (step 6130), 125 in the C seed (the seed's own paired merge bought one more).

### 6.3 The fix and what it costs

`quadtree.jl`, `dgquad.jl`, `semquadconvection.jl`: `balance!(qt; symmetry = :x)` splits mirror pairs whenever a leaf is split for 2:1 (both or neither, so a pair at `max_level` stays untouched;
`:none` is the old code verbatim); `_coarsen_pair!`, `semquad_enforce_leaf_budget!` and `amr_cycle!`'s hard-budget guard are pair-strict (both quartets complete and unprotected before either is
merged); `protect` is closed under the mirror first (`_mirror_protect`), because an asymmetric theta-range guard otherwise refuses ~94 % of the pairs at 30 % protect density and the tree stops
coarsening; `mirror_close!(qt; max_level)` is a repair pass, refinement only, computing the common refinement of the tree and its mirror, run for `:x` after balancing and before the hard budget; and
`refine_wall_to_min!(...; symmetry = :x)` refines wall pairs.

| tree / operation | before (`94e8bd3`) | after |
|---|---|---|
| `mirror_close!` alone on the C seed tree | — | 25 refinements, 125 → **0**, 8266 leaves, 2:1 held, area 2.5 |
| `wallrefine` seed on `pre_step6131.h5` | 120 → 125 | 120 → **0** (24 repairs, then 536 paired merges), 8188 leaves, budget reached |
| one runtime pass on `pre_step6131.h5` | 120 → 120 | 120 → **0** |
| one runtime pass on the C seed tree | 125 → 125 | 125 → **0** |

A paired merge drops 6 leaves (3 per quartet), so the budget is met in steps of 6 and a refused pair leaves the tree above `max_leaves` with `reached == false`. Measured: at protect density up to 10 %
the budget is reached exactly; at 20–30 % the tree sits +6 to +18 leaves over budget with `defect == 0`. G208 pins `<= max_leaves` for the mild case and `<= max_leaves + 18` for the 30 % case. The
repair only refines, so it can push a tree over budget: 125 defects on the C seed cost 25 refinements (+75 leaves, 8191 → 8266), and the C card protects 8184 of those 8266, so a resume from a pre-fix
checkpoint can merge nothing and ends one pass over budget with the documented warning.

### 6.4 The pins and the lesson

G208 (910 assertions) pins the defect arithmetic (one one-sided split == 5 defects), the repair including its `max_level` refusal, `balance!(:x)` pair splits, 15-cycle random AMR sequences at `cells_x = 1` and 5, the asymmetric-protect passes (the 0 → 81 and 0 → 5 operations), the +18 bound, and the `symmetry = :none` FNV pins (`35968c872ebeabf0`, `07812bd2ff43ed4d`, `0de7757f838fd16f`, `theta 84de84a66ab9dd69`, `(ux,uz) 6ea033597f291f32`, `history 8fcbf706331b1542`). The 25-frame classic blob run keeps `868a79d9...458afe88` (a new balanced arm has its own digest, `a4fb8248...b8cdeda`, `docs/VALIDATION.md` §2.3.1); that classic tree was already closed.

An invariant that is not measured is not an invariant. The defect lived in the code for months and was reported as success because no counter existed. Make the violation count an API so it can be
asserted, and make the repair a separate, idempotent, refinement-only pass, because a damaged tree cannot be fixed by tightening the merge rule. The pair-strict rule caps the coarsening supply by 60x
(§5.6), which is the documented price of the invariant.

---

## 7. The load-bearing blocked-key quirk

`semquad_enforce_leaf_budget!` (`julia/src/dgquad.jl`) merges the lowest-error legal quartet, and when the merge is undone by the following rebalance it adds the attempt to a `blocked` set. The
historical code pushes `parent`, `mparent` and, for each of the quartet's four children `k`, `(k[1], k[2] >> 1, k[3] >> 1)`. The last four look wrong: `k[1]` is the child level, so those keys are not
the merged parent; they are a real quartet parent one level **down**. A reader's first instinct is to delete them.

They are load-bearing. Those keys occasionally block an unrelated quartet, so the greedy trajectory depends on them. The stage-B work measured the clean version on 60 random budget-pass cases (levels
3–7, `cells_x` 1–3, `:none` and `:x`, random protect, random `eta`, random budget): **6 of 60 cases change**, 4 with a different leaf set and 2 only in `merged`, and in 2 of the 6 the clean pass
*reaches* the budget where the old one could not, the tell that it is not obviously worse, only different. The gate that pins the historical pass is G211 (200 assertions), which records the plan order
and per-cycle numbers of the `:balanced` coarsening gate; the clean version fails it, and G216 pins the same leaf sets again. For a bitwise stage the 6/60 class is decisive: the historical line is
kept verbatim and documented at the call site. For the same reason `blocked` does not receive the parents the rebalance splits: with the linear candidate scan a cascade-split parent has an incomplete
quartet and cannot be re-merged, so the entry would be inert, and adding it changes `blocked`, which changes the trajectory.

A historical quirk that changes 10 % of the measured cases is part of the specification, whatever it looks like. Record it with the A/B that proves it matters, at the call site and here, and never
combine a cleanup with a performance rewrite in one commit: if the leaf set moves, the two changes cannot be separated by bisection.

---

## 8. The Nusselt definition and the closure column

### 8.1 Symptom and the three definitions

Three wall Nusselt numbers believed to be 17 % apart (`7.62` pointwise against `6.33` CBF), and a per-step `closure` column whose residual was 1.44, so the energy budget appeared not to close. Both
readings were wrong, and the second was pointing at the wrong object.

One wall element, GLL nodes `theta_0` (interior edge), `theta_1` (mid), `theta_wall = 0` (Dirichlet), height `h`. Two linear functionals of the nodal values measure `int_wall dtheta/dy dx`:

```
pointwise   p = (theta_0 - 4 theta_1 + 3 theta_wall)/h     (semquad_nu_walls, family A)
steady CBF  q = (theta_0 - 8 theta_1)/(3h)                 (semquad_nu_cbf, family B)
q - p = -(2/(3h)) (theta_0 - 2 theta_1) = -(h/3) theta''_wall
```

`p` is the one-sided three-point derivative of the P2 interpolant, exactly the derivative of the discrete solution; `q` is the sum of the **steady** operator's residual over the wall dofs, a reaction;
`theta''_wall` is the profile's second derivative at the wall. Family C is the **time-step** CBF, the sum of the BDF2 step operator's residual over the wall dofs, `sum_wall (A theta - rhs)`.

### 8.2 The orders, exactly, and the 17 % claim

On a smooth solution the pointwise error is the interpolation error `-(h^2/12) theta'''`, second order; the CBF walls' error is `-(h/3) theta''`, first order. Verified on the production operator at `h = 1/64` over the whole wall:

| `theta` | exact `int dtheta/dy` | A | B |
|---|---|---|---|
| `u` | −1 | −1.000000000000 | −1.000000000000 |
| `u^2` | 0 | **+0.000000000000** | **−5.208333333e-3 = −h/3** |
| `u^3` | 0 | **+1.220703125e-4 = +h^2/2** | **+0.000000000000** |

On the four production states the closed form reproduces the measured gap to 1e-6 relative on the top wall and 1e-3 on the bottom. At `h = 1/64` the pointwise error is `h^2/2 = 1.2e-4`, the CBF's is
`h/3 = 5.2e-3`, 43x larger. The measured truncation error of A on the late states is 3e-5 to 9e-5 relative; the measured CBF term grows with `Nu` from 0.27 % (E5202) to 1.19 % (D7402) as the boundary
layer thins.

The 17 % claim does not exist at the audited states. At step 7202: `Nu_top` = **+7.6128 (A), +7.6923 (B), +7.6930 (C)**, a 1.0 % spread; `Nu_bot` = −2.8338 / −2.8341 / −2.8341, 0.02 %. No route
produces 6.33. The value 6.33 is what all three definitions give near step 6050 (6.304 at 6001, 6.433 at 6201), so the pair "7.62 pointwise / 6.33 CBF" mixes two states, not two definitions.

### 8.3 The closure column is a bookkeeping residual

The driver's `closure` is not the scheme's identity. The identity the assembled operator satisfies is `D := Ibase + latent_dgdt - F_pt = A * closure_area = (F_ts - F_pt) + (latent_adv - sum(C theta)) - dH/dt - eps`, where `Ibase` is the base-state heat input, `latent_dgdt` the latent source, `F_pt` the pointwise wall flux, `F_ts` the time-step CBF, `F_ss` the steady CBF, `latent_adv` the PDE-level
latent advection `int L Gamma_T u.grad theta`, `C` the skew advection matrix, `H` the enthalpy, `eps` the constraint and solver residual. Measured decomposition:

| item | E5202 | E7202 | E7602 | D7402 |
|---|---|---|---|---|
| `D` | −1.701752 | +1.188501 | +3.242800 | +5.188037 |
| `F_ts - F_pt` | −0.043293 (2.5 %) | −0.200961 (−16.9 %) | −0.221654 (−6.8 %) | −0.268183 (−5.2 %) |
| `latent_adv - sum(C theta)` | −7.507760 (441 %) | −0.234986 (−19.8 %) | −0.362173 (−11.2 %) | +0.953056 (18.4 %) |
| `-dH/dt` | +5.860927 (−344 %) | +1.625146 (136.7 %) | +3.819150 (117.8 %) | +4.496633 (86.7 %) |
| `-eps` | −0.011626 | −0.000698 | +0.007477 | +0.006530 |

Three formula defects carry it. It omits `dH/dt`, and `-dH/dt` is 86–137 % of `D` whenever the run is not stationary (the E run is strongly non-stationary, `Nu_top` climbing 6.30 → 7.61 → 8.16 → 8.93
between `t = 1.45e-3` and `2.21e-3`). It uses the PDE-level `latent_adv` where the skew assembly has `0.5 int (1 + L Gamma_T) u.grad theta`, so the two latent integrals nearly cancel and `0.5 int u.grad theta` does not (`int u.grad theta` = +14.82 at E5202, +0.506 at E7202). And it mixes a gradient-family `Nu_top - Nu_bot` with reaction-family terms, inheriting the `O(h)` wall functional. The
residual is a property of the formula, not of the flux: the scheme closes its own identity to `|R_C|/|rhs|` = 0.43 %, 0.05 %, 0.28 %, 0.14 %, while the pointwise value misses it by 4.2 %, 15.3 %, 15.9
%, 17.1 % of the budget scale (0.22–1.05 % of `H`).

### 8.4 Consequences and the lesson

Report A as `Nu_top`/`Nu_bot`: it is the physical conductive flux, invariant under wall refinement at a fixed state, and its error is smaller than the boundary layer's own resolution error (`h/delta = 0.06–0.10`, measured). Keep C for the discrete energy budget; it closes by construction, which makes it a tautology rather than evidence of accuracy. Do not mix A and C at late times. ASPECT's `heat flux statistics` is family C, so the cross-code comparison has a definitional floor of `h/3 theta''_wall`: 0.005–0.046 % of `Nu_bot` and 1.0–1.2 % of `Nu_top`. The existing −6.1 % window difference in
`Nu_bot` is therefore not definitional: the definitional offset is at most 0.0015 in `Nu_bot`, 0.7 % of the 0.21 window difference. The fix is in the closure formula and the comparison protocol, not
in `semquad_nu_walls`; `c95d596` adds the opt-in exact energy-identity residual `resid_A`, the instrument that would have caught the formula defect earlier.

When two diagnostics disagree, check first whether they are the same functional. A residual of the wrong formula can dominate the physics and send a project looking for a solver bug. "The budget does
not close" is a statement about a formula until the scheme's own identity is written down and decomposed.

---

## 9. The near-null pressure space

### 9.1 Symptom and the classification

`STOKES-FAIL` at three states (steps 6130, 6508, 7641), with unshifted direct residuals from `3.98e-4` to `4.69e7`, LU growth from `7.1e6` to `1.2e15`, and the failing residual entirely in the
continuity block. The suspects were a bad right-hand side, a broken constraint set, a degenerate tree, or a conditioning cliff. Four measurements, in order of decisiveness:

1. **The constraint test.** The same physical state (the `eta` field and buoyancy of the step-7641 checkpoint) on a **uniform L5 tree**: 10240 leaves, `h = 1/64`, zero hanging constraints. The six
  smallest singular values are `5.3922 / 5.4214 / 6.3762 / 6.4545 / 6.8266 / 6.9729e-10` against the AMR tree's `5.3909 / 5.4212 / 6.3761 / 6.4544 / 6.8326 / 7.0133e-10` (0.6 %), the same `lambda` to
  0.05 %, the same `beta_mode` (8.3e-3), the same wall elements. The cluster is not induced by the hanging-node elimination.
2. **The shape.** All six modes are 100 % pressure, 100 % in `y > 0.9`, 100 % on level-5 leaves, 99 % of their mass energy on the wall line `y = 1`, and the sign class of the four Q1 corner values is
  "checkerboard" (`s00 = s11 = -s10 = -s01`) for 1.000 of the mass energy in every supporting element. One element carries 52 % of mode 1 (the wall-row element at `x = 0`). `cos_M(p,1) = -2.8e-3` to
  `+3.4e-3`; orthogonality to every constraint row is `max |p . z| / (|p||z|) = 1.7e-25` over all 2922 rows.
3. **The local inf-sup table.** An isolated Q2/Q1 element has `beta = 0.95346` at L2 through L7, exactly level-independent; the 2:1 T-junction with its hanging constraints has `0.69870` against
  `0.70005` without them (constraints cost 0.2 %). The wall checkerboard on the real tree with `eta = 1` has `beta = 0.696`; with the physical lid (`eta = 1e4`) it has `7.9e-3 = 0.696/sqrt(1e4)`. The
  deficiency is the viscosity weighting.
4. **The level dependence.** The family exists at every level, with `beta` settled after L5 (1.20e-2 at L4, 7.9e-3 at L5, 7.4e-3 at L6) and `|lambda|/max|S|` = **3.3e-13** at L4 (range 3.34e-13 to
  5.38e-13), **6.8e-15** at L5 (6.85e-15 to 1.09e-14), **5.4e-18** at L6 (5.44e-18 to 2.62e-17). `lambda ~ -beta^2 p'Mp`, with `p'Mp` the pressure mass of the packet and `max|S|` pinned at the `eta`
  clamp, so the relative eigenvalue falls with refinement; the measured fall is steeper than the `h^2` of a fixed-size packet because the packet widens (11 elements at L5, 93 at L6). (Inference on the
  widening; the three relative eigenvalues are measured.)

The verdict is mechanism (iii): an element-level inf-sup deficiency, the viscosity-weighted checkerboard, present with zero constraints and at every level, at least 12–16 dimensional at L5 and growing
with the wall element count.

### 9.2 The gauge that failed

The minimum-norm gauge is the bordered system of the unpinned operator, `[S g 0; g' 0 1; 0' 1' 0]` with `g = -S c`, `c` the constant-pressure vector, and the exact null vector of the unpinned saddle
`[0; 1]`. On step 6600:

| solve | residual | `vel` | `div` | growth | `abs(x)` |
|---|---|---|---|---|---|
| pinned, unshifted | 4.46e+07 | 4.47e+07 | 5.50e+00 | 9.77e+14 | 2.05e+12 |
| bordered (mean-zero gauge) | 4.97e-02 | 4.97e-02 | **3.83e-08** | 6.79e+06 | 6.58e+05 |
| row/column equilibration of `S` alone | 1.01e+01 | 1.01e+01 | 1.39e-06 | 2.88e+12 | 7.57e+05 |

The gauge does what it promises: the continuity block drops to 3.8e-8, so the constant is removed. The momentum block stays at 5e-2, because the cluster is orthogonal to the constant. The code already
pins the first free pressure dof (`pf[first(pfree0)] = 0.0`, folded into the reduced `Cp`), and that pin is not the problem. A gauge cannot see a mode localised on the wall strip.

### 9.3 What works: symmetric equilibration

`stokes_scaling = :equilibrated` (`aceeeac`, default `:none`) applies an exact congruence before the direct LU:

```
D = diag( 1/sqrt(diag Kred),  2 max|Kred| / max(|Bt|, |Bs|) )
S -> D S D,   rhs -> D rhs,   x = D y
```

`Kred` is the reduced stiffness, `Bt`/`Bs` the saddle's pressure blocks, the two factors the velocity and pressure scalings. The congruence leaves the solution unchanged, so only the arithmetic of the
sparse LU changes. On the G-7641 tree the pressure factor is `d_p = 5.67e6`, i.e. `(d_p/d_p')^2 ~ 8e9` relative to a mass-normalised scaling; in that scaling the cluster's relative eigenvalue is
`4.0e-6`, so it is lifted far above the round-off floor, and the smallest mode of the scaled operator is a momentum mode at `4.2e-11` relative.

| state / tree | unscaled unshifted | shipped stopgap (1e-14 shift) | equilibrated, unshifted |
|---|---|---|---|
| G 7641 AMR (`n = 82 951`) | 4.20e2 | 3.72e-7 | **5.96e-10** |
| G 7641 cooldown-seeded | 1.37 | 3.85e-7 | **1.22e-10** |
| G 7600 checkpoint | 4.14e2 | 3.18e-7 | **7.45e-10** |
| uniform L5 (`n = 92 382`) | 1.58e6 | 1.19e-6 (fails 1e-6) | **3.65e-8** |
| uniform L4 (`n = 23 150`) | 1.96e-10 | — | 1.96e-10 (not engaged, bitwise) |
| uniform L6 (`n = 369 086`) | 4.53e5 | 1.92e-5 (19x the goal) | **3.64e-7** |

The option is gated on the historical health check (relative residual above `1e-8` or above the caller's tolerance), so a healthy solve is bitwise identical with the option on, and a solve that does
not benefit falls back to the unchanged unscaled ladder. `|du|/|u|` against the tiny-shift answer is 3.15e-5 (AMR) and 1.01e-4 (L5); `|dp|/|p|` is 4.4e-2 / 1.3e-2, because the pressure representative
of a singular system is not unique. G214 (31 assertions) pins the option surface, the diagonal, the congruence, the bitwise healthy and fallback behaviour, the untouched `:mg` path, the driver keyword
and restart guard, and the unshifted answer on the realistic state.

### 9.4 The negative results and the lesson

Iterative refinement against the unshifted operator with the unshifted LU as preconditioner diverges: residuals 7.20e2, 1.22e3, 4.85e2 at 1, 2, 4 iterations, because the error lives in the near-null
subspace and every iteration re-amplifies it. One-time band deflation is impractical: the band is at least 12-dimensional at L5 and 16 at L4 with no gap, its dimension grows with the wall element
count, and `k = 12` at L5 costs a 90 s shifted LU plus an `n x k` basis (8 MB at L5, 35 MB at L6). Pressure stabilisation is not needed for the solver and would perturb the wall flux that §8 pins; at
`h = 1/64` a Brezzi-Pitkaranta term `c h^2 int grad p . grad q` would need `c ~ 0.5` to lift the cluster to 1e-8 relative and `c ~ 500` to reach the momentum block's scale.

Classify before fixing. Three candidate mechanisms implied three different fixes (a gauge, an AMR/constraint change, a pressure-space change), and the decisive test cost 43 s and removed two of them.
The failed gauge is worth more than most successes: it proved the constant mode harmless (`abs(S c)` 4.0e-9 relative) and redirected the fix to the scaling.

---

## 10. `STOKES-FAIL`: the shift ladder, its trigger, and the resume order

### 10.1 Symptom

Two trajectories died in the same 5.9k–6.2k window on the same 8188-leaf tree, with residual ratios to `stokes_tol` of 1.10 and 1.32. Restarting from the last checkpoint re-reached the same failure
one step later: a treadmill. Later the C run died at step 6508 with an unshifted residual of `1.6e10`, and the G run at step 7641 with `4.465e-3` after the whole ladder ran.

### 10.2 Finding 1: the 1e-12 retry manufactured the fatal residual

The direct branch solved unshifted, and if the residual exceeded the internal `1e-8` gate it retried with a `1e-12 * max|diag(S)|` diagonal shift and accepted the retry unconditionally (G164 added
that rung for the wall trees, where it takes `1.6e-4` to `3e-11`). A direct solve of `(S + delta I) x = rhs` satisfies `S x - rhs = -delta x` exactly, so `|r| = delta |x|`: the residual of a shifted
solve is the shift times the size of the answer, not a measure of the solve.

On step 6131: `delta = 1e-12 * (4.6753e4 + 1) = 4.6754e-8`, `|r| = 1.317e-3 * 1.0112e3 = 1.332`, so `|x| = 2.85e7` and the effective amplification `|x|/|rhs|` is `2.85e7 / 1.7187e2 = 1.66e5`. The
rungs:

| shift | LU time | residual | `vel` | `div` | `abs(du)/abs(u)` |
|---|---|---|---|---|---|
| 0 (unshifted) | 1.35 s | **3.982e-04** | 3.982e-04 | 3.933e-05 | — |
| 1e-12 | 0.69 s | **1.317e-03** | 4.61e-06 | 1.317e-03 | 1.247e-02 |
| 1e-10 | 0.60 s | 9.359e-03 | 4.35e-04 | 9.359e-03 | 3.589e-01 |
| 1e-8 | 0.82 s | 6.048e-01 | 2.165e-01 | 6.048e-01 | 4.76 |

The unshifted solve passes the card's `1e-3` gate. The retry converts it to `1.317e-3`, exactly the fatal number (the run's 1.3168e-3 and this state's 1.317e-3 agree to four digits): the fatal
residual is the shift's perturbation, not a failed solve. Escalating the shift is the wrong medicine here, because the residual grows monotonically with the shift and the velocity moves 36 % at 1e-10
and 476 % at 1e-8. The state is also extremely sensitive: computing the 1e-12 rung with `dmax = max|diag S|` instead of `max|diag S| + 1` (the historical `+1.0`, a 2e-5 relative change in the shift)
moves that rung from 1.576e-3 to 1.317e-3, so no fix can rely on a specific shift value. Iterative refinement is the right medicine: one GMRES iteration against the unshifted operator, preconditioned
by the 1e-12 factorisation, buys a factor 37 (4.299e-5) and lands on the run's normal residual level.

### 10.3 The fix: a ladder

`dea9718` replaced the three lines with `_shift_ladder_solve(S, rhs; accept_tol, bad_tol = 1e-8, resid, ...)`, rules in order:

1. **Healthy**: unshifted residual `<= bad_tol` and `<= accept_tol`, one factorisation, shift 0. A non-finite residual is a failure and is recorded as `Inf` so the ladder fires; the old branch treated
  `NaN` as "not bad" because `NaN > 1e-8` is false (`b23483f`).
2. **First rung**: shifts `(1e-12, 1e-10, 1e-8)` in order, refactoring each time; the first whose residual is `<= accept_tol` is accepted. With `accept_tol = Inf` this is exactly the old single retry.
3. **Escalation**: with a finite `accept_tol`, a rung that still misses it escalates, the best solve seen (including unshifted) is remembered, and escalation stops as soon as that best passes, so a
  larger shift cannot throw away an answer that already passed (`5c54870`).
4. **Iterative refinement**: if nothing passes, up to 4 GMRES iterations against the unshifted operator, preconditioned by the best shifted rung, tolerance `min(1e-10, 0.01 accept_tol)`, keeping only
  an improvement.
5. **Return the best** with its residual; the driver's error stays the last resort.

Diagnostics: `LAST_SEMSTOKES_RESID_SPLIT`, `LAST_SEMSTOKES_SHIFT`, `LAST_SEMSTOKES_REFINE_ITERS`, `LAST_SEMSTOKES_LADDER`, later `LAST_SEMSTOKES_LADDER_PHASE`. On a miss the driver prints one stderr
line with the split residual, the eta range and clip fraction, and every rung tried. The failing state now returns the unshifted answer (3.982e-4) at the cost of one extra residual evaluation,
`LAST_SEMSTOKES_REGULARIZED` false, and the 180-step run from the step-6000 checkpoint is bitwise identical to the pre-fix run through step 6130 before passing the step that killed it. G207 (58, later
60 assertions) pins the control flow on a prescribed residual sequence, the semantics on the G164 wall tree, the bitwise no-op on a uniform mesh, and the reset of the records.

### 10.4 Finding 2: the tiny-shift trigger sat below the lost threshold

On step 6604 the unshifted solve returns `4.69e7` and the historical 1e-12 rung `1.08e-4`, while the near-null family sits at `|lambda| ~ 4.6e-10` and the rung's `delta = 7.8e-8` is 170x above it: the
rung's bias is the residual. A shift scan falls monotonically as the shift is lowered (1e-12 1.08e-4, 1e-13 1.42e-5, 1e-14 **3.72e-7**, 3e-15 1.11e-7) until the round-off floor `eps max|S| = 1.7e-11`
absolute.

The fix for that class (`8633073`) is a phase **before** the historical rungs: if the unshifted relative residual is finite and above `SEMSTOKES_LOST_TOL = 1.0`, i.e. the answer is worse than `x = 0`,
try `SEMSTOKES_TINY_SHIFTS = (1e-14, 1e-13)`, accepting a rung only when it both improves on the unshifted residual and meets the caller's tolerance. A finite residual below 1.0, which includes every
state that produced a usable answer up to 6130's 3.98e-4, never reaches the phase, which is what keeps those answers bitwise unchanged. A non-finite residual is excluded: that is a broken
factorisation, not a biased one.

Then the G run died at step 7641 with:

```
[case] STOKES-FAIL step=7641 res=4.465272e-03 (vel=1.125e-03 div=4.465e-03) stokes_tol=1.0e-03 shift=1e-10 refine=4 leaves=10000 N=43427 saddle=98568
[case] STOKES-FAIL ladder: 0e+00:9.654e-01 1e-12:1.561e-02 1e-10:8.245e-03 1e-08:6.279e+00 refine:4:4.465e-03
```

The unshifted residual is **0.9654**, 3.5 % below the 1.0 gate: a 96.5 % relative residual was classified as "the factorisation did solve the system", so 1e-14 and 1e-13 were never tried. The nearest
reproducible states have `r = 1.37` and `4.20e2`; both fire the phase and return 3.85e-7 and 3.72e-7, and a one-step replay of the failing update passes. The state is not unsolvable; the switch is
what differs.

The fix (`16fdfa3`, `064747d`) is rule 7: after the ladder and its refinement, if the best residual still misses `accept_tol`, a finite unshifted residual exists, the tiny list is non-empty and the
priority phase did not already run, try the tiny shifts in order, keep the best answer seen including the refinement's, stop as soon as `accept_tol` is met, and record the phase (`:healthy`, `:tiny`,
`:ladder`, `:refine`, `:fallback`, `:fail`). Measured on the three reproducible states: 3.7179e-7, 3.8464e-7, 3.1766e-7, bitwise equal to `lu(S + 1e-14 dmax I) \ rhs`, against historical bests of
7.32e-5, 7.38e-5 and 9.37e-5. A tightened gate (`accept_tol = 1e-12`) returns the same answer with phase `:fail`, so a gate that cannot be met still fails loudly. G212 (62 assertions) pins the exact
step-7641 sequence, the healthy bitwise path, the G164 `Inf` semantics, and the phase record.

The closed form is in G209's synthetic saddle: with `K = I_4`, `B = nb [1 1 0 0; 0 0 1 1]`, `nb = 1e-6`, the Schur complement is exactly `2 nb^2` and the shifted pressure is `p(delta) = (B f - g)/(2 nb^2 - delta)`. At `delta = 1e-12` (resonant) `|p| = 9.10e17` and the residual is `4.82e5`; at `delta = 1e-14` (100x below) `|p| = 1.84e6` and the residual is `9.74e-9`. The shift must sit below the
near-null scale, a property of the operator rather than a fit to one state.

### 10.5 Finding 3: the resume was not bitwise, for three reasons

1. **`dt_prev` was stored one step late.** The in-loop checkpoint was written before `dt_prev = dt` (the assignment sat after the steady-state block), so the file carried step `n-1`'s `dt_prev` while
  the next step would use step `n`'s. Measured on the 160-leaf synthetic probe: the step-10 checkpoint had `dt = 2.3643e-5` but `dt_prev = 2.5024e-5` (the step-9 value). The trajectory used the right
  value; only the file was wrong, which a bitwise resume test finds and a physics test does not.
2. **The final checkpoint stored the diagnostic re-solve velocity.** When a run ends on an AMR step, the driver re-solves the flow after the loop so the returned velocity belongs to the final mesh;
  the uninterrupted trajectory's next step starts from the zeros the AMR pass left. With a velocity-dependent law (`:composite`) that difference feeds the first `eta` evaluation and breaks the resume.
  The final checkpoint now carries a `velocity_stale` attribute and the driver zeroes the restored `ux/uz/pv` when it is set; old files and in-loop checkpoints are unchanged.
3. **The `:balanced` cooldown was not in the checkpoint at all.** `AmrCooldown` lived in memory, so a restart began empty and the first AMR cycle after a resume was not the cycle the uninterrupted run
  executed: `refined = 502` against `295` at step 7605/7606. `8bb11e6` persists it as two `Int64` datasets (`cooldown_no_refine`, 4 x n `level, ix, iy, remaining`, including the mirror half;
  `cooldown_no_coarsen`; plus a `cooldown_cycle` attribute), sorted by leaf key so the file is deterministic, and the restart restores it instead of resetting. It is state, not configuration, so it is
  not in `run_config` and an old checkpoint is not rejected; `freeze` and `symseed` copies carry it, `wall_checkpoint` starts empty because it changes the leaf set.

The evidence is a 20-step synthetic case (640-leaf `5x2` base, uniform L3, `eta_law = :expTz`, `cfl = 0.5`, `amr_every = 1`, `max_leaves = 640`, `symmetry = :x`): uninterrupted against
checkpoint-at-10-then-resume gives `theta = fce9712f4ea53377`, `(ux,uz) = 953262731fdf3e7f`, `pv = cdcad29324b84013`, history rows 11–20 `b25f651bcc2ad2a4`; the step-10 file carries `cycle = 10`, 76
`no_refine` (38 merged pairs, mirror halves included) and 144 `no_coarsen`, and the first resumed cycle reproduces the uninterrupted `36/36/36/0` and ages the counter to 11. The same file with the
cooldown stripped resumes to a different trajectory (`theta = abcda3f3749a4f8e`), so the test is load-bearing. The realistic card at `max_leaves = 1000` reproduces `theta = 61ca317e577d3dac`, leaf FNV
`= 5c23e8fa271648f7` across a 10 + resume-to-20 split, with rows 11–20 `diff`-identical. Old checkpoints keep their historical behaviour (`refined = 502`, `theta = 3a59871db6ce72db` under both
binaries); the 502-against-295 difference is a property of the file, not of the resume code. G213 (71 assertions) pins the round trip, the synthetic bitwise resume, the stripped-cooldown sensitivity,
the legacy path, and the classic FNV pins.

### 10.6 The lesson

A regularisation's bias is the defect: the 1e-12 rung was accepted because it made the factorisation succeed, and its `delta |x|` was larger than the gate. A trigger on a residual classifies the
*answer*, not the operator, and a threshold at 1.0 classifies a 96.5 % residual as healthy. A bitwise resume tests the file's contents, not the solver: `dt_prev`, the velocity, and the policy state
all had to be in the file before the trajectory could be reproduced.

---

## 11. The MG constrained-Jacobi defect

### 11.1 Symptom and the excluded suspects

`linear = :mg` on the failing production tree `F` (8188 leaves) ran 100 FGMRES iterations to a true residual of `9.718e-1`, a convergence factor of `(0.9718)^(1/100) = 0.99971` per iteration, and
printed `MG-FGMRES true relres=0.97 exceeds 1e-5 at level 7`. The iteration trace showed the preconditioner amplifying: `||M^{-1} v||/||v|| = 1.6e170` and `||v - K z||/||v|| = 4.3e173`. With
`mg_vel_solver = :cg` the same state let a `SingularException` escape from the FGMRES Hessenberg solve. The suspects were the pressure block, the Schur preconditioner, the 2:1 forest, and the hanging
constraints.

An exact velocity inverse with the same w-BFBT gives `F` 52 iterations to 6.88e-7 and `P1` 35 to 7.97e-7, so the pressure block is not the blocker. The failing MG run's `div` residual is 2.2e-7 and
100 % of the residual is velocity, so the wall-strip cluster is a direct-LU round-off problem rather than the cause. Every coarse tree is a coarsening of the fine one, `max abs(P) = 1.0`, sizes match
`nfs`, and `Cu` is reassembled with the hanging constraints on every level (`Cus[k]`, constrained counts 32/60/116/312/992/3244/6810/7562), with `P = C_f' P_full C_c` applied on both sides of every
transfer, so the forest and its constraints are handled correctly. The same outer solver reaches 6.88e-7 and 7.97e-7 with an exact velocity inverse.

### 11.2 The mechanism

`_semquad_diag_reduced` (`julia/src/semquadstokes.jl:1696`) returned `Cu' * diag(K)`, documented as "the standard lumped diagonal ... a diagonal that differs from the assembled one only on hanging
rows is harmless for a damped-Jacobi smoother". On a hanging row the Lagrange weights are negative, the lumped entry can cancel to zero or below, and the positivity clip `max(d, 1e-12 * dmax)` gives
that row a diagonal `1e12` times too small. The Jacobi update `x += omega r / D` then amplifies the row by `1e12` per sweep, at every level with a clipped row. Measured `D_mf / diag(K_assembled)` per
level:

| level | `F` min … max | `P1` min … max |
|---|---|---|
| 1–3 | 1.00 / 1.00 | 1.00 / 1.00 |
| 4 | 1.00 / 1.47 | 1.00 / 1.00 |
| 5 | **0.00 / 1.52** | 1.00 / 1.52 |
| 6 | **0.00 / 1.52** | 1.00 / 1.52 |
| 7 | **0.00 / 1.54** | 0.15 / 1.53 |
| 8 | **0.00 / 1.54** | 0.15 / 1.53 |

The `0.00` is exactly `1e-12`. One V-cycle on a random right-hand side gives `||z||/||b|| = 1.36e170` and `||b - A z||/||b|| = 3.56e173`; the per-level table shows one finest-level pre-smoothing pass
amplifying `1.01e3` to `4.50e11` (4.4e8) and the two clipped levels compounding to `4.3e176`. A stationary iteration diverges after the first V-cycle (`4.29e173`) and returns `NaN` after the second.
`P1` and `P2` never activate the clip, which is why the same code converges on them: `P2` 42 iterations to 6.38e-7, `P1` 65 to 7.04e-7.

### 11.3 The fix, what it buys, and the lesson

`9250a3c` computes the exact `diag(Cu' K Cu)` without forming `K` or the product: per leaf, the constraint rows of its local dofs give a dense `2n^2 x ncol` matrix `W`, the leaf contributes `diag(W' K_e W)`, and summing over leaves is exact because `K = sum_e K_e` and each `K_e` has no cross-leaf entries. `mode = :lumped` keeps the historical path (`mg_diag = :lumped`), `:constrained` is the
default. On a conforming tree `Cu` is a row selector, `Cu' diag(K)` *is* `diag(Cu' K Cu)`, and the old arithmetic is kept bit for bit. The sparse operator path always used `diag(A[k])` and is
untouched.

| state | before (lumped `D`) | after (constrained `D`) |
|---|---|---|
| `F` | 100 it (maxit), 9.718e-1 | **70 it, 6.18e-7** (shipped 70 it, 6.07e-7) |
| `P1` | 65 it, 7.04e-7 | **61 it, 7.12e-7** |
| `P2` | 42 it, 6.38e-7 | **42 it, 6.27e-7** |

The fixed finest diagonal equals the assembled `diag(Kred)` to `7.8e-17` (`F`), `8.7e-17` (`P1`) and `7.2e-19` (`P2`) relative; no row on any level sits on the clip (smallest `D/dmax` is `1.02e-7` on
`F`). `mg_vel_solver = :cg` no longer throws: 54 outer iterations, 534 inner CG, true residual `6.07e-7` in 35.2 s, agreeing with the `:vcycle` answer to `2.1e-6`. The exact diagonal costs 0.098 s for
all eight levels warm (lumped 0.010 s). G218 pins it on a 28-leaf 2:1 hanging-node tree at `p = 2, 3` and every level: `D` equals the assembled `diag(Cu' K Cu)` to round-off; no matrix-free row is on
the clip when the assembled diagonal is positive; one V-cycle is finite and non-amplifying (`0.20 < 1`); the `:cg` saddle converges (residual 5.0e-12, 34 inner iterations, velocity equal to direct to
6.9e-11); `:lumped` remains reachable and an invalid `mg_diag` is rejected.

The fix does not make MG competitive: the repaired solve is 9.8 s on `F` against 6.65 s for direct, because the setup alone exceeds one direct solve. The structural defects survive, with the
rediscretised coarse operator 20x and 32x too soft at the two coarsest levels (`1.91e1` and `3.10e1` relative), and the coarse space weak (the optimal correction of the same smoothed residual leaves
1.4–5.4x the input). Galerkin coarsening fixes the scale and takes `F` to 61 iterations and `P1` to 46, still far from a 50-iteration kill criterion on `F`.

**Update (2026-11-12).** The 9.8 s / 6.65 s sentence is the pre-tuning record. The
[MG.md](MG.md) §11-§13 tuning cut the setup and the cycle, and the
8-thread MG solve is now faster than the accuracy-matched equilibrated direct
solve on both production states ([PERFORMANCE.md](PERFORMANCE.md) §3.5); the
structural coarse-space diagnosis in this paragraph is unchanged and is why the
outer count is 81/65 before the exact BFBT diagonal and 71/65 after it (§12).

A smoother diagonal is an operator, not a scalar. A clip of the form `max(d, eps * dmax)` is a `1/(eps * dmax)` amplifier on every entry it touches, by construction, and it touches exactly the entries
with the most delicate cancellation. The `1e170` is the honest signature of a preconditioner that amplifies, and it should be read as a bug, not as slow convergence.

---

## 12. The BFBT lumped diagonal: 16 entries, `1e12` contrast

### 12.1 Symptom

The w-BFBT pressure block needs a positive `D` so that `S1 = B D^{-1} B'` is SPD. On the production `F` tree (8 188 leaves) the shipped weight `D = Cu' diag(M_u(sqrt(eta)))` had **16 non-positive entries out
of 61 964**. The `1e-12 * max|d|` clip lifted each of them to `2.761e-14` against a largest entry of `2.761e-2`, so the returned diagonal spanned `1.0e12` and `D^-1` reached `3.6e13`. The solve still
converged: 81 outer iterations to `relres` 8.619e-7. The residual was healthy and the weight was not.

### 12.2 Mechanism

`_semquad_bfbt_diag` (`julia/src/semquad_mg.jl:2330`) with `how=:leaf` takes the leaf mass diagonal `dfull` and restricts it, `Cux' * dfull + Cuz' * dfull` (`:2343-2348`). A lumped restriction samples the
diagonal of `M_u` and drops the cross terms a hanging row contributes to `diag(Cu' M_u Cu)`. On such a row the Lagrange weights change sign, the sum can be zero or negative, and the clip at `:2356-2360`
replaces it with `1e-12 * dmax`. The clip is unconditional: it also runs over the `:exact` and `:matrix` diagonals, where every entry is positive and it touches nothing.

The exact restricted diagonal is built leaf by leaf by `_semquad_bfbt_diag_exact` (`julia/src/semquad_mg.jl:2261`). Each leaf classifies its constrained rows from `Cu`, forms its local mass diagonal plus a
dense sub-block over its hanging rows and the masters they touch, and scatter-adds the quadratic forms; `M` and `C' M C` are never formed. `:auto` resolves to `:exact` on `mg_operator=:matrixfree` and to
`:leaf` on `:sparse` (`julia/src/semquadstokes.jl:3691-3693`).

### 12.3 The measurement

| weight `D` | raw non-positive | contrast | `F` it | `F` iteration s | `F` rel err `du` | `P1` it |
|---|---|---|---|---|---|---|
| shipped `Cu' diag(M_u)`, clip `1e-12` | **16** | 1.0e12 | 81 | 1.144 | 1.694e-6 | 65 |
| `diag(Cu' K Cu)` | 0 | 9.8e6 | 79 | 1.239 | 3.282e-6 | 70 |
| **`diag(Cu' M_u Cu)`, exact** | **0** | **5.923e5** | **71** | **1.003** | **7.956e-7** | 65 |
| shipped, re-clipped `1e-6` | 16 | 1.0e6 | 80 | 1.132 | 6.682e-7 | 65 |
| shipped, re-clipped `1e-4` | 16 | 1.0e4 | 155 | 2.170 | 7.364e-7 | 64 |

The two weights differ by **1.541 %** in the L2 norm on `F` and **0.0864 %** on `P1`, which is where the 10 iterations appear. Re-clipping the 16 entries does not reproduce
the gain and is not monotone: `1e-6` gives 80 iterations and `1e-4` gives 155. `diag(K)` buys two iterations on `F` and pays 2x the velocity error, 3.282e-6 against the 2.1e-6 class. The clip is inert on the
exact diagonal: zero entries are touched on either state. The same five rows are in `docs/MG.md` §6.1.

### 12.4 The fix

`503fd87` ships `:exact` on the matrix-free path. The leaf-local construction is cheaper than the weight it replaces at eight threads, 0.088 -> 0.028 s on `F` and 0.123 -> 0.032 s on `P1`, and 5 ms more
expensive at one (0.090 against 0.085 s), because 87 % of either build is the eta sampling (8 188 x 36 = 294 768 `wfun` calls on `F`). The warm solve at eight threads falls 1.548 -> 1.324 s (-11.5 to
-14.5 % across two sessions) and 2.070 -> 1.791 s at one thread (-13.5 %); charged for both builds, the total falls 10.1 % on `F`/8, 14.6 % on `F`/1, 7.8 % on `P1`/8 and 0.1 % on `P1`/1. Accuracy moves
with the count: `|du|/|u|` 1.694e-6 -> 7.956e-7, gauge-removed `|dp|/|p|` 2.165e-7 -> 2.267e-7. `relres` goes 8.619e-7 -> 6.782e-7, in the same direction. `docs/PERFORMANCE.md` §2.1 quotes the post-diagonal solve at 1.356 s, a later session than the 1.324 s above.

### 12.5 The gate

G226 (28 assertions) runs a 28-leaf 2:1 tree with a 1000x viscosity contrast. It asserts that the leaf-local exact diagonal equals the sparse reference `diag(Cux' Mv Cux + Cuz' Mv Cuz)` to round-off,
1.9e-16 on `F` and 2.3e-16 on `P1`, because the summation orders differ; that no entry is clipped and every entry is positive; that the exact and lumped diagonals differ, so the equality is not vacuous;
that two constructions are bitwise equal; that both weights solve to `relres <= 1e-6` and the exact one stays inside the MG class `|du|/|u| <= 2.1e-6`, gauge-removed `|dp|/|p| <= 8.8e-7`, against the
equilibrated direct solve; and that `:auto` resolves per operator, with an invalid value rejected.

### 12.6 The lesson

A smoother diagonal is an operator, the §11 argument one block over. The clip is a `1e12` amplifier on exactly the entries with a delicate cancellation, and sixteen entries of 61 964 move the count from 80
to 155. The exact object is worth building when it is available per leaf: it is cheaper at eight threads, it removes the clip instead of shrinking it, and it improved the answer at the same tolerance. Ship
the exact object and keep the approximation as the A/B.

---

## 13. The per-call costs: 640 B and a 100 us fork

### 13.1 Symptom

The matrix-free V-cycle paid per call. Every parallel op created its own task set, 6-9 us even for an empty body at eight workers, and the fork accounting put that at 1.2 % of a V-cycle. Timing the same
ops in place told a different story: one threaded `r = b - y` at 61 964 dofs costs **about 101 us inside the cycle** (506 us over five calls), against 19.6 us back-to-back and 21-23 us inside an open
region. The gap is the workers' state. A task set created, run for 10-300 us and destroyed leaves the other threads in the scheduler's idle state, on this hybrid CPU in an OS idle state; the
next call pays the wake-up and the barrier waits for the slowest thread. The back-to-back benchmark never sees it because the cores never go idle.

The kernel had the smaller version of the same shape: five lane vectors (`shx`, `shy`, `swf`, the unit scale, `kofs`) built with `zeros`/`ones` of length `BLK` per call, **640 B per call**. A threaded
level-6 apply allocated 25 456 B per call and an eight-thread V-cycle 1.83 MB; at 65 applies per cycle that is 1.3 MB per cycle and 90 MB per 70-iteration solve.

### 13.2 Mechanism

`_semquad_td_run!` (`julia/src/semquadstokes.jl:1517`) is the fork. `SEMQUAD_MG_SERIAL_DOFS = 20 480` (`julia/src/semquad_mg.jl:1129`, a `Ref`) keeps levels below that size on the calling task: at 318
dofs the serial subtract is 0.021 us and the Jacobi update 0.155 us, so a fork/join is 40 to 330 times the work it distributes. The threshold decides which levels open a region; inside an open region, a
level's pre-smoothing block, its 2+2 Jacobi sweeps, the residual and the restriction each still created a set.

`SemQuadParRegion` (`julia/src/semquadstokes.jl:1580`) is a persistent worker set of `nt` workers, one of them the calling task, in which an op is published as a phase: a job, a chunk count, an atomic
sequence number, a barrier. `_semquad_mg_vcycle_region!` (`julia/src/semquad_mg.jl:1482`) runs a threaded level's pre-smoothing, its Jacobi sweeps, the residual and the restriction in one fork/join, and
its prolongation, `x .+= py` and post-smoothing in a second, while the workers spin on the phase counter. The five kernel lane vectors moved into `SemQuadKScratch` (`julia/src/semquadstokes.jl:892`), one
instance per worker.

### 13.3 The measurement

| quantity | `F` | `P1` |
|---|---|---|
| forks per V-cycle | 15 -> **2** | 43 -> **4** |
| worker tasks | 120 -> 14 | 344 -> 28 |
| V-cycle, 8 threads | 6.94 -> 5.42 ms (1.28x) | 15.48 -> 10.98 ms (1.41x) |
| FGMRES iteration | 1.455 -> 1.339 s (1.09x) | 1.890 -> 1.614 s (1.17x) |
| warm step | 1.902 -> 1.835 s (1.04x) | 2.569 -> 2.406 s (1.07x) |

Thirteen forks removed at the measured 101 us would be 1.31 ms of the 1.52 ms the cycle lost; the remainder is the residual floor and the allocation [arithmetic]. The residual fork/join floor after the
rewrite is 35-39 us on `F` and 70-76 us on `P1`, 0.6-0.7 % of the cycle. Stage 7 adds the transfer and the two `Cu` products as phases inside the same regions: the V-cycle falls a further
5.78 -> 4.39 ms (1.32x) and 11.64 -> 8.61 ms (1.35x). The serial restriction is 1.8-2.1x slower inside a live region than isolated,
185.5 against 335.6 us for `F`'s level-6 to level-5 transfer at 15 634 dofs, which is why it became a phase (`docs/MG.md` §3.6-§3.7, `docs/PERFORMANCE.md` §7.2).

### 13.4 The gate

G222 (18 assertions) pins the batch width against G180/G190, 0 B of allocation on the serial apply and below 16 kB on the threaded one, and the padded final block. Its threaded assertion needs a warm-up
call because the first measured call pays 308 080 B of compilation. G223 (39 and 71) pins each op inside a region against the per-call form on every level, each V-cycle with the region on, off and against
the one-worker hierarchy for every serial threshold, and the fork count per cycle. G224 (93 and 180) pins the transfer and split-`Cu` phases bitwise and their phase counts on top of G223.

### 13.5 The lesson

A fork costs the state it leaves the other threads in, and a microbenchmark that calls it back to back cannot see that state. Measure in place, inside the cycle, or the record will quote 19.6 us while the
solve pays 101 us. The lever is the fork count and the workers' state together: a persistent worker set with phases took `F` from 15 forks to 2 and the cycle by 1.28x. The empty fork is 5.4-6.3 us at
eight workers and the in-place call is 101 us, so the other 95 us is the wake-up [arithmetic], and no pool size or chunking rule touches it.

---

## 14. The digest that lost its object

### 14.1 Symptom

`868a79d9...458afe88` is quoted as the 25-frame blob digest. It is the `theta` digest at the **end of the run, step 126**. Frame 25 has its own digest,
`dd364f70744923e1833a770c192f8d691dcb2a6b4eb2b32edaafc72d5aebfecb`, and its own `Tmax`, 1.264427445; at the 5-step cadence frame 25 is step 125 [arithmetic]. A reader who reproduced 25 frames and
hashed the last frame got a different number, and nothing in the quoted digest said which object it belonged to.

### 14.2 Mechanism

The driver calls the frame callback after an AMR cycle, with `ux` and `uz` zeroed, and re-solves only at the end of the run. A per-frame `u` digest is then a digest of zeros whose length is the frame's
dof count. In `out_julia/fresh_blob_kelly_1024_classic/frame_sha256.csv`, frames 19 to 25 all have 1024 leaves and all seven rows carry the same `u_sha256`,
`9862acd5230f7871052465fe9b0971ebbd0df0332aa8044168ca1e84de3f1209`; frames 1 to 18 all differ because the leaf count moves from 214 to 958. A per-frame `u` digest at this cadence is a leaf-count
checksum. The end-of-run `u` digest is the velocity pin: `c2a217d6...dc36af491` classic, `1cfcb5c6...84b0b51` balanced.

### 14.3 The fix

`2b016a2` promotes the producer out of the untracked `out_julia/agent/run_blob_kelly_1024.jl` into `julia/scripts/run_blob_kelly_1024.jl`, with `BLOB_POLICY` (default `balanced`) and `BLOB_OUT`, and
writes `frame_sha256.csv` and `run_summary.csv` next to the frame stream. `run_summary.csv` carries the object with the digest: `policy,frames,steps,end_theta_sha256,end_u_sha256,Tmax25,wall_s`. The 25
per-frame `theta` digests are tabulated in `docs/VALIDATION.md` §2.3.1, and the classic numbers are labelled classic-era instead of withdrawn.

### 14.4 The measurement

| arm | wall s | steps | end `theta` | end `(ux,uz)` | frame-25 `Tmax` | frame-25 leaves |
|---|---|---|---|---|---|---|
| classic | 129.8 | 126 | `868a79d9...458afe88` | `c2a217d6...dc36af491` | 1.264427445 | 1024 |
| balanced | 134.5 | 126 | `a4fb8248...b8cdeda` | `1cfcb5c6...84b0b51` | 1.264329738 | 898 |

The classic arm reproduces the documented classic-era record bitwise, so the correction is a relabelling. The balanced arm matches an independent run of `out_julia/agent/bench_blob_production_ab.jl`
bitwise and is the current default's baseline.

### 14.5 The gate

No gate hashes the blob; the digest is a record with a step. G229 is the closest pin: it exercises the balanced default and pins its leaves and three FNV hashes. A digest row belongs next to its producer
and its policy, and a document that quotes it has to quote the object.

### 14.6 The lesson

A hash measures an object, and the object is the state, the step and the policy. Drop one and the number stops being reproducible. The per-frame `u` case is sharper: a digest over a field that is zeroed
at that instant is a checksum of the mesh, and it stays stable and meaningless at the same time. Hash what the run produced, and write the label next to the hash.

---

## 15. `run_config`: a guard that reads only the checkpoint

### 15.1 Symptom

A watchdog restart of `E` with `CASE_AMR=balanced` starts silently and prints `amr=balanced ... gate=1` (`julia/scripts/case_realistic_2d.jl:460`) over a step history written by the classic policy. The
guard compared the file's 71 keys and found no mismatch, because the file carries no `amr_policy` key to compare. `amr_coarsen_gate` has the same hole on the F arm, whose 73 keys include `balanced` and no gate.

### 15.2 Mechanism

`run_semquad_simulation` builds `run_config` at `julia/src/semquadconvection.jl:673` and, on a resume, iterates the checkpoint's keys: `for (k, v) in restart.config`, then `haskey(run_config, k) || continue`
(`:813-817`). A key absent from the file is never compared, and the throw is `:829-831`. `stokes_scaling` is the single absent-means-`none` special case (`:823-828`), added because equilibration changes
the answer of a failing solve.

The direction of the loop is the defect. The checkpoint is treated as the schema, so every key added to the code after the file was written is invisible on that file, and those are the keys that decide
the trajectory. Measured on the nine production checkpoints (`docs/AMR.md` §5, 2026-11-13): E, D, C, B2 and B_auto76 carry 71 keys and neither `amr_policy` nor `amr_coarsen_gate`; F_nogate carries 73 and
`balanced` with no gate; G's early file 74 and both; G's late file 75 plus a cooldown. A restart of E under the new defaults therefore passes the guard and switches the mesh policy in the middle of a
trajectory, and `docs/AMR.md` §5 records the two pure arms: E's step-20000 state is `Nu_top 18.2892` and G's step-15195 state is `Nu_top 16.597`, and a hybrid arm matches neither.

### 15.3 The fix

`c7e0f55` stores `coarsen_theta_tol`, `refine_thresh` and `coarsen_thresh` unconditionally (`julia/src/semquadconvection.jl:686-687`), inert under `:balanced` and live under `:classic` and
`amr_strategy=:threshold`. Checked with a checkpoint round trip: stored 0.001/0.4/0.2, a changed tolerance raises `ArgumentError`, the matching config restarts, and the defaults `Inf`/0.5/0.1 are stored.
The same commit documents the driver default `:balanced` (since `4584e85`) and adds `t_g229` to the drivers-checkpoint CI group.

### 15.4 What it does not fix

The six files written before `amr_policy` existed still lack it, and F_nogate lacks `amr_coarsen_gate`; the loop still skips both. The three thresholds are guarded from this commit forward, and nothing
was retrofitted. The structural hole is a schema without a
version: comparing a key the file does not carry requires knowing what the file's writer would have written, which is either a version number or a per-key rule like `stokes_scaling`'s. A guard that reads
only the checkpoint's keys cannot protect that checkpoint's future.

### 15.5 The gate

No gate pins the absent-key behaviour; the round trip is in the commit message. G229 exercises the default path and writes `amr_policy` into `run_config`, which is the key the E restart lacked. A
missing-key test is two lines: write a checkpoint with one key deleted from `config`, restart, require the `ArgumentError`.

### 15.6 The lesson

A restart guard is a schema check, and a schema check needs a schema. Iterating the file's keys makes every new key optional by construction, and the new keys are the ones that change the mesh policy.
Store the defaults unconditionally, and give the record a version so the next key has somewhere to be compared.

---

## 16. Process lessons

### 16.1 Cold-start timing subtracted from warm timings

The direct-solve cost was reported for weeks as `6.65 s = 5.6 s assembly + 1.02 s LU`. It was a **cold** first assembly in a fresh process minus a **warm** raw `lu`; the assembly paid JIT and
first-touch. Warm, on `F`/`P1`/`P2`: 2.09 / 2.63 / 2.33 s per solve, of which assembly is 0.35–0.45 s (15–18 %) and the LU **pair** 1.58–1.99 s (73–83 %); one assembly per step; the kernel is
0.042–0.045 ms per leaf against the historical 0.431, so there was no kernel regression either. Warm step 3.58 s (Stokes 3.08, energy 0.31, work 0.14, diagnostics 0.04); the first JIT step is 20.4 s.
The corrected split is `015b6f7`, with `913c75c`, `daa4c86`, `70f84dd`, `7c0b9b7` as intermediate corrections. The consequence: the assembly opportunity collapsed from 5.4x the LU to 4–7 % of the
solve (0.15–0.25 s), below the bar, and the only multi-second lever left is the LU pair's second factorisation or UMFPACK's unsymmetric branch (MUMPS or PARDISO, 0.6–1.0 s), a dependency and numerics
change that breaks bitwise. Rule: warm up, then measure; A/B in one session, one process per arm; quote the ratio, not the seconds.

### 16.2 Shared-worktree commits sweeping other agents' unstaged lines

Several agents share one checkout. A commit made with `git add -A` or `git commit -a` sweeps whatever another agent had unstaged at that moment, and the result is a commit that cannot be bisected and
a file whose committed content nobody chose. The rule is mandatory here: every commit names its pathspec, and staging is by path. This document was committed as `git add docs/BUG_LEDGER.md` then `git commit -m "..." -- docs/BUG_LEDGER.md`, so the index for every other path is irrelevant to it. The archive baseline is taken with `git archive <commit>`, which reads the object database and never the
working tree. Before any commit in a shared checkout, read `git status --short` and `git diff --cached --name-only`, and be able to explain the committed set name by name.

### 16.3 CI cache keys longer than 512 characters

`11e7744`. `julia-actions/cache` derives its key from the job and matrix parameters. The `drivers-checkpoint` job's entry embeds the full gate list, which exceeds 512 characters, and GitHub rejects
the key with a Key Validation Error **before any test runs**, so the failure looked like a workflow syntax problem. The fix replaces the action with `actions/cache` and a short explicit key derived
from the `Manifest.toml` hash. CI cache keys are not a place for a specification.

### 16.4 Gate registration drift

`fb14a08`: 18 gates that existed on disk were never in `julia/test/runtests.jl` (G182–G191, G194, G197–G199, G202–G204), and 8 were missing from the CI selection (G194–G199, G202–G204). `46d0c2a` then
restored a curated CI selection because the previous commit had appended every gate on disk, helper fragments included. A gate exists only when `runtests.jl` and the CI selection both list it, and
both edits belong in the commit that adds the gate. The two failure directions are symmetric: a gate that never runs protects nothing, and a helper fragment mislabelled as a gate fails for the wrong
reason.

### 16.5 The disk-full failure at C: 10 GB

A long run failed its writes with the C: volume down to 10 GB free. The mechanism is inferred to be ENOSPC on the write path; the failure was a write error, not a numerics error, since the last
complete step was healthy. Keep `out_julia` (checkpoints 1.3 MB each at `checkpoint_every = 1`, frames, `.mem` files) on the data volume, and check free space before a multi-hour run, because the
failure arrives hours in. When a run dies, read the error class before the numerics: a write failure and a `STOKES-FAIL` both leave a truncated last line.

### 16.6 The performance-claim discipline

Every performance claim in this ledger is a same-session A/B against a `git archive` baseline of a named commit, with the same `Manifest.toml`, one process per arm, warm-up discarded, and the ratio
quoted. The audits that followed this rule produced numbers that survived re-measurement (energy boxing 3.0x, the step-95 guard 531x, the shift ladder's one extra residual evaluation). The audits that
did not were corrected twice: the blob 2.77x comparison had its "before" arms measured while 16 gates ran concurrently and added a back-to-back 22-step 3.07x control; the AMR pass's 159.8 → 128.1 s
had its attribution checked by removing the fixed step (the other 24 cycles cost 0.96 s before and 0.98 s after, so the entire win was the guard). The corollary is the 5 % rule: an optimisation below
about 5 % of the step is recorded with its numbers and not taken (§3.4). It is cheaper to write a row in a table than to own a new kernel.

### 16.7 Documentation as part of the fix

The `maxpic` fix invalidated every historical measurement taken at `maxpic > 1`. The inventory (`0330edc`, 877 matched lines, 83 AFFECTED) and the correction table were as necessary as the code
change, because a reader of the old measurements would otherwise conclude the rheology was wrong. When a fix changes a scheme, the blast radius of the claims is part of the change.

### 16.8 A converged `relres` does not certify a preconditioner change

The `K`-completed `S1` deflation rows on `P1` reached `relres` **7.21e-7** and 9.06e-7 with a gauge-removed `|dp|/|p|` of **1.0e-4** and 2.2e-5 against the equilibrated direct solve, 20 to 90 times the
baseline's 1.2e-6 (`docs/MG.md` §5.5). The residual gate passed and the pressure was wrong by 1e-4. A low-rank modification `S1 + Z (sigma - Lambda) Z'` went further: 5.7-12.4 s to refactor, 40-110x the
0.11-0.15 s cold w-BFBT build, 265, 285 and 500 iterations, relative errors 3e-4 to 25. Neither deflation shipped.

The rule that came out of it: every preconditioner change in this campaign is gated on `|du|/|u|` and the gauge-removed `|dp|/|p|` against `linear=:direct, stokes_scaling=:equilibrated` in the same
process, never on `relres`. The reference solve is not the limit: its own residual is 9.97e-11 on `F` and 1.84e-9 on `P1`. The exact BFBT diagonal is the positive case, `relres` 8.619e-7 -> 6.782e-7 and
`|du|/|u|` 1.694e-6 -> 7.956e-7 in the same direction, and G226 (e) asserts the MG class rather than the residual. The two commits that recorded the study are `e9171b7` and `f9027a9`.

---

## 17. Gate and commit index

| defect | fix | gate (assertions) | what it pins |
|---|---|---|---|
| S15 energy boxing | `74494d1` | G185 (25) | pre-fix kernel snapshots, `sum(M)`, `sum(abs,K)`, `norm(C)`, `norm(F)`, the BDF2 solution, identity-callback equality |
| S15 Newton kernels | `96e52a5` | G202 (18) | `Kgeo`/`Kp` entry by entry against the pre-fix loop, SHA-256 of `K, B, Kgeo, Kp, Kred, fv`, the concrete field type |
| S15/S16 AMR hot path | `17d0e17` | G215 (12 939) | pre-audit equivalence on 24 random trees, 0-B budgets with historical controls, the plan caps, the lazy cache |
| S16 GMRES guards | `96e52a5` | G203 (23) | `nit == maxit` for six `(maxit, restart)` pairs, stagnation fires and never fires, finite `rel` on breakdown |
| S16 step-95 guard | `4bbdae9` | G186 (265/267) | worklist-vs-full-balance equivalence, cache consistency, fixed-point detection, 2 s cost bound |
| S16 plan caps | `17d0e17` | G215 | `rounds == MAX_PLAN_ROUNDS` on an all-illegal tree, `plan_sweeps`, `plan_capped` |
| Newton path | `96e52a5` | G202, G203, G150 (12), G153 (210) | bitwise default path, kernel equality, loop guards |
| `maxpic > 1` | `86d03f6` | G193 (33) | backward-Euler residual < 1e-8 and the steady equation does not hold |
| AMR policy | `117cc9d`, `0b3f4bd` | G210 (263) | the `:balanced` policy, pinned cycles, `refused` never charged to `dropped` |
| coarsening gate | `98554dc`, `75e629c` | G211 (200) | `coarsening_allowed`, plan order, per-cycle numbers, mirror/2:1/area |
| mirror formula | `398b245` | G204 (217) | `cells_x = 1/2/5`, involution, range, the `2.5 x 1` pair, unit-box identity |
| mirror invariant | `e91beaf` | G208 (910) | defect arithmetic, repair, pair splits, 15-cycle random sequences, `:none` FNV pins |
| blocked-key quirk | `957a43b` | G211, G216 | the historical plan order and whole-pass equivalence |
| Nu definition | `266a656`, `c95d596` | identities, `resid_A` | the two wall functionals on `u`, `u^2`, `u^3`; the exact energy-identity residual |
| pressure space | `aceeeac` | G214 (31) | the equilibration diagonal and congruence, gated bitwise healthy/fallback, the `:mg` path |
| shift ladder | `dea9718`, `5c54870`, `b23483f` | G207 (58/60) | rung sequence, escalation with best-kept, non-finite handling, `Inf` semantics, uniform no-op |
| tiny shifts | `8633073` | G209 (39 + 1) | the lost trigger, the tiny rungs, the never-worse rule, `:mg` untouched |
| rule 7 fallback | `16fdfa3`, `064747d` | G212 (62) | the step-7641 sequence, the 1e-14 bitwise answer, the phase record, `:fail` on a tightened gate |
| resume bitwise | `8bb11e6`, `8abfb02` | G213 (71) | cooldown round trip, `velocity_stale`, `dt_prev`, stripped-cooldown sensitivity, legacy path |
| MG diagonal | `9250a3c` | G218 | `D == diag(Cu' K Cu)` at `p = 2, 3` on every level, no clip, non-amplifying V-cycle, `:cg` converges |
| MG production defaults | `8bb724c` | G219 (27) | coarsening 1.5 on the matrix-free path, 21 +/- 1 outer iterations at 1e-8, 2+2 on a non-top-heavy hierarchy and 3+3 on a top-heavy one, the finest-level assembly removal |
| MG reduced assembly | `463d4bf` | G220 (37) | `assemble_only=:mg` equals the full assembly except `K`/`Kred` are `nothing`, no `SEMSTOKES_K_ASSEMBLIES` increment, a whole solve forms no sparse `K`, the coarse geometry mode bitwise |
| MG level cache | `9827b4a` | G221 (57) | the topology key, a cache hit equals the from-scratch build, viscosity-dependent data left out of the cache |
| 32-lane kernel | `ecc1c5c` | G222 (18) | the lane width bitwise against G180/G190, 0 B on the serial apply and under 16 kB threaded (the first call pays 308 080 B of compilation), the serial threshold |
| region V-cycle | `2b3c2d3`, `262e7eb` | G223 (39 and 71) | every op bitwise inside a region, every V-cycle with the region on, off and serial, the fork count per cycle |
| region phases | `ff1c09c` | G224 (93 and 180) | the transfer and split-`Cu` phases bitwise, the phase count, no added fork |
| eta lane table | `7ec7566` | G225 (129 and 129) | the lane table as a pure re-layout, the padded final block, the apply and V-cycle bitwise |
| BFBT diagonal | `503fd87` | G226 (28) | the exact weight to round-off, no clipped entry, the cross terms real, determinism, the accuracy class, the `:auto` resolution |
| `S1` inner solve | `345ed8a` | G227 (15) | `:chol` bitwise, `:level` to round-off and thread-independent, `:cg_*` to tolerance, the cache key |
| card MG tolerance | `edf7488`, `c535115` | G228 (20) | the card declares `stokes_tol=1e-3` and `mg_tol=1e-8` and forwards both, and removing either declaration fails |
| balanced default | `4584e85` | G229 (13) | `run_config` records `balanced`, the diagnostics, two identical runs bitwise, the leaves and three FNV hashes |
| auto default | `2453f16` | G230 (27) | below and above the threshold, no fallback, two runs bitwise, `:direct`/`:mg` still selectable |
| CI cache key | `11e7744` | n/a | the workflow starts |
| gate registration | `fb14a08`, `46d0c2a` | n/a | the suite and the CI selection agree with the files on disk |

---

## 18. Reproducing the audits

Every audit has a scratch directory under `out_julia/` (git-ignored) and a fixed protocol. The essential commands:

```bash
# the baseline tree for any A/B (never the working tree)
rm -rf /home/szh/ms_base && mkdir -p /home/szh/ms_base
git archive 8b08af3 | tar -x -C /home/szh/ms_base
cp julia/Manifest.toml /home/szh/ms_base/julia/

J=/home/szh/opt/julia-1.12.7/bin/julia
# S15/S16, Newton path
$J --project=julia out_julia/agent/newton_audit/nb_bench.jl      # step time
$J --project=julia out_julia/agent/newton_audit/nb_parts.jl      # phase split
$J --project=julia out_julia/agent/newton_audit/nb_allocs.jl     # Profile.Allocs
$J --project=/home/szh/jetenv out_julia/agent/newton_audit/nb_jet.jl
bash out_julia/agent/newton_audit/run_gates.sh

# S15/S16, AMR path
bash out_julia/agent/s3hot/amr_ab.sh out_julia/s340/ab_none/resume.h5 12 phase none <tag>
$J --project=julia out_julia/agent/s3hot/s15_alloc.jl out_julia/s340/ab_none/resume.h5
$J --project=julia out_julia/agent/bitwise_theta_sha.jl          # the two pinned cases

# the AMR policy and the gate
$J --project=julia julia/scripts/amr_policy_probe.jl quantify <ckpt>
$J --project=julia julia/scripts/amr_coarsen_gate_probe.jl trace /tmp/s350/ab_none_19000.h5

# the MG campaign: the diagonal, the forks, the blob digest (§12-§14)
$J --project=julia --threads=8 julia/test/t_g226_bfbt_leafdiag.jl
$J --project=julia --threads=8 julia/test/t_g223_mg_region_vcycle.jl
$J --project=julia julia/scripts/bench_bfbt_leafdiag.jl F
BLOB_POLICY=classic BLOB_OUT=/tmp/blob_classic $J --project=julia julia/scripts/run_blob_kelly_1024.jl

# the pressure diagnosis (the §17.1 replay state; the original is gone)
$J --project=julia out_julia/agent/diag_pressure_null/probe_modes.jl <state> 6 40
$J --project=julia out_julia/agent/diag_pressure_null/probe_uniform.jl <state> 5 6 40
$J --project=julia out_julia/agent/pressure_fix/verify_final.jl <state> amr

# the shift ladder on a real failing state
G209_STATE=out_julia/agent/g212/run7641/resume.h5 \
  $J --project=julia julia/test/run_gates.jl t_g209_pressure_gauge.jl
```

`docs/MG.md`, `MG.md`, `docs/AMR.md`, `docs/PERFORMANCE.md`, `docs/VALIDATION.md` and `docs/CASE_REALISTIC.md` §11-§18 carry the full instrument, the negative results, and the states. This ledger is
the index to them; when the two disagree, the section doc wins and this file gets corrected.
