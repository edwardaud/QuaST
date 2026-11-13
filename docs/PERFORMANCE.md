# Performance: what a step costs, and which optimisations paid

The performance record of the solver, in reading order. §1 prices one step on the
production card and names the session behind the number. §2 opens the solve inside
that step, first on the matrix-free path that now runs it, then on the direct path it
is compared against. §3 states the direct-versus-MG verdict and the two changes of
comparison basis that produced it. §4 carries the crossover table, the old table it
replaced, and the selector rule that turns the table into a default. §5 and §6 keep
the allocation and scaling records. §7 is threading and determinism. §8 is the
decision table: what is worth doing, what is closed, and the number that closed it.

**How to read the numbers.** Every wall time below is a measurement from this
repository on the state named next to it, on one WSL2 Ubuntu-24.04 machine (16 cores
arranged as 6 performance and 4 efficiency cores with SMT, 23 GB RAM, Julia 1.12.7).
Unless stated otherwise the process ran one thread and `BLAS.set_num_threads(1)`, and
the time is the minimum or median of repeated warm solves. A ratio is quoted only when
both sides ran in the same session; the numbers that are deliberately cross-table are
labelled. Numbers obtained by adding measured components are marked *inference*.
Absolute seconds identify a session: the same shared per-step work has moved by 1.7x
between machines, so a time from another session is evidence about that session, and
not a baseline. Anything labelled **superseded** is kept because it explains a decision
that is still in the code; it is not a number to quote.

**The cards.** `F` is the production warm state, 8 188 leaves and p=2 on the step-6131
resume, 69 388 reduced saddle dofs (61 964 velocity and 7 424 pressure) and a
6-level hierarchy. `P1` is the s340 `ab_none` resume, 10 000 leaves and p=2, 86 544
saddle dofs (76 988 and 9 556) and 7 levels. `P2` is the third direct-solve state,
10 000 leaves and 82 951 saddle dofs. The uniform L4/L5/L6 trees (256, 1 024 and 4 096
leaves) and the frame-19 blob field are the scaling cases; the 25-frame blob run is the
end-to-end trajectory; the allocation card is a four-cycle AMR pass on the same
10 000-leaf state; the verify100 pair is two 500-step runs of the realistic card from
t = 0, one per linear path. §1 and §2 use the card and the three states, §3 and §4 the
production pair, §5 the allocation card, §6 the scale tables, §7 the L6 and the
production pair.

**Run artifacts.** The `out_julia/...` paths in this document name the logs and run
directories that produced a measured number in the development tree. They are cited as the
primary record and are not part of this repository; the numbers are quoted from those
records, not regenerated here.

**The verdict.** The MF+MG campaign took the warm production step from a 3.58 s
pre-MG baseline to 2.72 s, and a 500-step end-to-end A/B on the production card
measures 2.798 s/step for the matrix-free arm against 4.403 s/step for the equilibrated
direct arm, 1.57x. The matrix-free solve beats a direct solve that actually reaches
1e-6 by 1.30 to 2.31x across the four same-session cells (one and eight threads, cold
and warm), and by 2.49x for the shipped 71-iteration row against the eight-thread
direct column. It now wins on every measured cell of the crossover sweep, including
the 25-frame trajectory that used to lose, and `linear=:auto` is the library default
with the re-calibrated thresholds of §4. The pre-MG numbers (the 3.58 s clean step, the
5.2 s audit step, the 6.65 s cold solve, the 9.8-11 s MG rows, the old crossover table,
the 1.457 s/2.55x endpoint) are labelled in place and are superseded as performance
claims. The one thing that did not change is the arithmetic: every shipped stage except
two is bitwise, and the two that are not are gated on the equilibrated direct solution.

## 1. One step on the production card

### 1.1 The card, and the number that produced it

The card is the realistic case of [CASE_REALISTIC.md](CASE_REALISTIC.md): a 2.5 x 1
box, composite isostress rheology at `Ra=1e5`, two phase transitions with per-phase
latent heat, `H=10`, free slip, p=2, `max_level=7`, a hard 10 000-leaf budget,
`amr_every=5`, `linear=:auto` (which resolves to MG at this size, §4.4) and
`mg_tol=1e-8`. The driver accepts a Stokes solve at `stokes_tol=1e-3`, so the solve
sits five orders inside its own gate.

The number that produced the split is the verify100 MG arm: 500 steps from t = 0 on
that card, eight threads, the frame writer off, `wall=1398.9 s`, which is
**2.798 s/step** over 500 steps, with zero `auto_fallbacks`. The direct arm of the same
pair is 500 steps on the same card with `CASE_LINEAR=direct` and
`CASE_STOKES_SCALING=equilibrated`, `wall=2201.3 s`, which is **4.403 s/step**. The
ratio is 1.574x by wall time and 1.574x by step. The two arms are the same script, the
same card and the same 500 steps; the script refuses to start a second spin-up while
one is running, so they ran one after the other in the same session. The MG log closes
at 23:08 on 2026-11-12 and the direct log at 23:47; both are in `out_julia/` and are the
primary record for this paragraph.

That is the end-to-end number. The split below decomposes it into the five phases the
driver's profile hook names, using the per-phase measurements of the vintages cited in
each row.

### 1.2 The split

| phase | s | share | contents |
|---|---|---|---|
| Stokes solve (MG, `mg_tol=1e-8`) | 2.21 | 81.2 % | reduced saddle assembly, warm hierarchy, cached w-BFBT, 91 FGMRES iterations, final residual, unpack to `u` and `p` |
| energy | 0.31 | 11.4 % | energy operator assembly (M, K, C, F, latent heat, EV field) and the BDF2 solve |
| work | 0.14 | 5.1 % | viscous dissipation integral `work_dissipation` |
| diagnostics | 0.04 | 1.5 % | Nu, Vrms, mean temperature, divergence, surface flux |
| AMR | 0.025 | 0.9 % | one pass per five steps at 0.122-0.134 s per cycle |
| total | 2.72 | 100.1 % | the shares are rounded one decimal at a time |

The arithmetic is a cost split. Write `T_step` for the wall time of one step and
`q_s = T_s / T_step` for the share of phase `s`; then

```
T_step = T_Stokes + T_energy + T_work + T_diag + T_AMR,      sum_s q_s = 1,
```

with `T_Stokes` the one saddle solve the step pays, `T_energy` the energy assembly and
BDF2 solve, `T_work` the dissipation integral, `T_diag` the five scalar diagnostics and
`T_AMR` the amortised mesh pass (`0.122-0.134` s per cycle divided by the cadence of
five). The five shares are the `q_s`; the total is their sum, and the measured 500-step
mean is 2.798 s.

The relation between the two is worth stating rather than hiding. The split is a warm
plateau decomposition: its Stokes term is the isolated solver call with the card's
kwargs on the 10 000-leaf `P1` state, measured at 2.208 s for 91 iterations in the
session behind [CASE_REALISTIC.md](CASE_REALISTIC.md) §1.6, and its four other terms are
phase runs from the direct-card audit, unchanged by the solver swap. The 500-step mean
is 2.8 % above the sum. The mean carries two biases in opposite directions: the first
hundred steps run a tree that is still growing (640 leaves at step 1, 10 000 at step
100), which makes them cheaper, and the first step pays compilation and first touch,
which makes it dearer. The measured 2.8 % is the residual of those two, so read the
shares as the deliverable and the absolute sum as a decomposition of the mean to a few
percent. The direct arm checks the same decomposition the other way: its Stokes term is
3.885 s (the eight-thread equilibrated direct warm solve of §3.3), and
3.885 + 0.31 + 0.14 + 0.04 + 0.025 = 4.40 s against the measured 4.403 s/step.

One caveat belongs to the Stokes row. The driver's own Stokes slot in the old direct
card ran 17 % above the isolated phase run (3.08 s against 2.63 s, §2.3) because the
production `eta` closure and the step's other in-flight allocations are charged to it.
The 2.21 s above is the isolated-call convention. If the driver charges the MG solve
the same way, the slot is nearer 2.4-2.6 s and the 2.8 % residual is where the
difference lands. The shares move by a point or two; the ordering does not.

### 1.3 The four phases the campaign did not touch

The energy phase is the one that moved most in the previous period, and it did not move
in this one. Its element loop used to be dynamically typed: the `SemQuadEnergyCfg`
callables were stored as `Union{Nothing,Function}`, so 328 M boxed allocations and
5 GiB of allocation per assembly went through the loop. Fixing the boxing took the
frame-19 step from 4.55 to 1.51 s and the 25-frame run from 438.4/443.2 s to
158.6/159.4 s. That change is why the phase is 0.31 s on this card.

The work phase is the viscous dissipation integral. It had the same boxing pattern
through `semquad_rho`'s `Union{Nothing,Function}` field and was fixed 2.04x in the
stage-4 audit; the phase is too small to show at this card's resolution. Diagnostics are
the scalar outputs. The AMR pass is 0.122-0.134 s per cycle after the allocation work of
§5, one cycle per five steps, and §1.4 of the old record already said the pass could not
move the step much; nothing since has changed that.

### 1.4 The pre-MG baseline, and what actually changed

The 3.58 s clean step and the 5.2 s audit step are the **pre-MG baseline**. The 3.58 s
is the second step of the s340 `ab_none` resume (10 000 leaves, p=2, 86 544 saddle
unknowns, composite rheology at `Ra=1e5` and `H=10`) with the unscaled direct solve:
3.08 s Stokes, 0.31 energy, 0.14 work, 0.04 diagnostics, no AMR. The 5.2 s is the step
of the allocation audit, the figure the audit's 0.010-0.011 s/step was measured against
(§5.6). Both are direct-path numbers, and both were taken before the matrix-free solve
was the default.

The campaign changed one of the five terms. The Stokes solve at the card's tolerance is
2.21 s where the direct solve at 1e-6 was 2.71-3.09 s, and the solve it replaces is the
equilibrated direct one at 3.885 s, which is the reference of §3. The other four terms
are the same measurements as before, which is why the split is a clean before-and-after:
3.08/3.58 = 86.0 % of the pre-MG step was Stokes, and 2.21/2.72 = 81.2 % of the current
step is. The solve is still the step. It is 0.87 s cheaper, and the step is 0.86 s
cheaper, which is the whole campaign at this resolution.

The 200-step s340 resume averages 5.37 s/step (1 074.6 s over 200 resumed steps) because
it includes AMR, frame output, the energy solve and the occasional shift-ladder repair.
The full direct production runs measured 4.044 s/step (E, 20 000 steps) and 4.13 s/step
(G, 15 195 steps) including frames, AMR and checkpoint I/O. Those are the
production-length averages, and they are pre-MG. The verify100 arms deliberately turn
the frame writer off, so their 2.798 s and 4.403 s are clean step times rather than
production averages.

Per-step allocation on the direct path is 2.40 GB: about 1.9 GB for the saddle assembly
and its factorisation, 0.30 GB for the energy path and 0.1 GB for work (HEAD `8501b53`).
A four-cycle AMR pass after the fixes of §5 allocates 30.50 MB, about 260 times less
than one Stokes solve. Bytes are not seconds: in the 25-frame run 63 % of the bytes that
drive GC are the LU factor arrays, and GC there is 24.0 s over 127 full sweeps.
Removing small objects buys GC headroom, not step time. On the matrix-free path the
per-step allocation was never instrumented, and the memory evidence is the peak RSS of
§3.3: 1.30 GB at `F`/8 against 2.37 GB for the direct solve, both including a
1.29-1.40 GB Julia-plus-checkpoint baseline.

## 2. The warm solve, phase by phase

### 2.1 The matrix-free solve slot

The solve the card pays is one reduced saddle assembly, one warm hierarchy build, one
w-BFBT cache lookup and 91 FGMRES iterations. The composition was measured at 81
iterations on `F` and 65 on `P1`, before the exact BFBT diagonal of §2.2 took `F` to 71,
so the fractions are the shipped ones and the absolute seconds are that session's:

| phase | `F`/8 s | % | `P1`/8 s | % |
|---|---|---|---|---|
| assembly (reduced matrix-free geometry) | 0.236 | 15.8 | 0.313 | 17.2 |
| hierarchy build, warm geometry cache | 0.089 | 6.0 | 0.171 | 9.4 |
| w-BFBT build, warm pressure cache | 0.0003 | 0.0 | 0.0003 | 0.0 |
| setup subtotal | 0.325 | 21.8 | 0.484 | 26.5 |
| FGMRES iterations (slot) | 1.164 | 78.2 | 1.338 | 73.5 |
| warm solve | 1.489 | 100 | 1.822 | 100 |
| ms per iteration | 14.372 | | 20.586 | |

The setup is 21.8 % of a warm solve on `F` and 26.5 % on `P1`, and its composition
changed in the campaign. The assembly is the largest setup row, the warm hierarchy is
the second (0.08 s on `F`, 0.17 s on `P1`), and the w-BFBT build is a cache hit at
0.3-0.4 ms. The warm hierarchy is not free: `make_semquad_mg` hits the geometry cache
but still rebuilds every level's leaf-local operator, its eta field, the exact
constrained Jacobi diagonal and the dense coarse operator. On a cache miss the same two
calls cost 0.13-0.28 s and 0.11-0.15 s, paid once per tree.

The shipped row is the same solve after the diagonal stage: 71 iterations at 1.356 s
([MG.md](MG.md) §4.2). The profile's 14.372 ms per iteration
carries over, because the apply is unchanged and only the cached factor's diagonal
moved: 0.325 + 71 x 0.014372 = 1.345 s, 0.8 % below the measured row, which is the
session drift. At 71 iterations the setup share rises to 24.0 % and the iteration to
76.0 %, so the campaign's net effect on the shape is to give the setup back four points
of the solve.

The card runs `mg_tol=1e-8`, between the benchmark tolerance and the library default.
The three rows, on the same states, eight threads, warm best of three, are the
tolerance ladder:

| state | `mg_tol` | outer it | true res | warm s |
|---|---|---|---|---|
| `F` | `1e-10` (library default) | 92 | 7.85e-11 | 1.608 |
| `F` | `1e-8` (card) | 81 | 8.61e-9 | 1.414 |
| `F` | `1e-6` (benchmark) | 71 | 6.75e-7 | 1.260 |
| `P1` | `1e-10` | 108 | 8.73e-11 | 2.681 |
| `P1` | `1e-8` (card) | 91 | 9.44e-9 | 2.208 |
| `P1` | `1e-6` | 65 | 6.62e-7 | 1.703 |

The `1e-8` row is the card's choice because it keeps the true residual 5.0-5.2 orders
below the driver's 1e-3 gate while moving the velocity by at most 2.7e-8 and the
gauge-removed pressure by at most 7.9e-9, 78x and 111x inside the documented MG accuracy
class (`|du|/|u| <= 2.1e-6`, gauge-removed `|dp|/|p| <= 8.8e-7`). The `1e-6` row is
22-37 % faster and is the benchmark row of §3; it is not the card's row because it moves
`P1`'s gauge-removed pressure to 1.28e-6, above the class. The tension between "fastest"
and "inside the class" is the only tolerance decision in the file, and the card resolves
it at 12-20 % of the warm solve.

### 2.2 The shape of the iteration

Per outer iteration at eight threads, the split is measured by a bitwise replica of the
production V-cycle and saddle solve:

| component | `F`/8 ms/it | % | `P1`/8 ms/it | % |
|---|---|---|---|---|
| V-cycle | 5.095 | 35.5 | 9.680 | 47.0 |
| `Bs zc` (continuity of the correction) | 0.585 | 4.1 | 0.623 | 3.0 |
| w-BFBT block | 3.635 | 25.3 | 4.511 | 21.9 |
| of which first `chol(S1)` solve | 0.976 | 6.8 | 1.205 | 5.9 |
| of which `S2` apply (matrix-free) | 1.732 | 12.0 | 2.079 | 10.1 |
| of which second `chol(S1)` solve and scaling | 0.923 | 6.4 | 1.221 | 5.9 |
| outer operator `A v` | 2.023 | 14.1 | 2.311 | 11.2 |
| FGMRES own work | 3.033 | 21.1 | 3.461 | 16.8 |
| total | 14.372 | 100 | 20.586 | 100 |

Two readings carry into the decision table. The V-cycle is no longer the whole story: it
is 5.095 ms of 14.372, and the w-BFBT block plus the FGMRES own work are 6.668 ms. And
almost half of the eight-thread iteration is serial: the two `S1` solves, `Bt`, `Bs`, the
`Z`/`W` copies, the modified Gram-Schmidt, the true residual and `Bs zc` are 6.6 of
14.4 ms on `F`/8 (46 %) and 7.7 of 20.6 ms on `P1`/8 (37 %). What threads is the
V-cycle, the leaf part of the outer `K` apply and the leaf part of `S2`; the sparse `Cu`
products, the scatter and every BLAS-1 operation stay on the calling task.

The iteration is stagnation-limited, not rate-limited. The first iteration buys nothing
(`relres` 1.000), and 40 iterations on `F` and 27 on `P1` are spent above 0.9; those
iterations buy 9 % of the total reduction on `F` and under 10 % on `P1`. Once the
stagnation breaks, the contraction is 0.80 to 0.67 per iteration on `F` and 0.70 to 0.65
on `P1`. Measured at the eight-thread slot cost, the plateau iterations are 0.58 s on
`F` and 0.56 s on `P1`, 39 % and 31 % of the warm solve. That is the number behind the
coarse-space row of §8.

The exact BFBT diagonal is the campaign's last stage, and it is the one that changed the
iteration count. The w-BFBT pressure block needs `S1 = Bs D^-1 Bt`, with `Bs` and `Bt`
the reduced divergence and gradient blocks and `D` a diagonal weight approximating the
velocity mass. The shipped `D` was the restricted diagonal of the mass matrix weighted by
the square root of viscosity, a mass diagonal with no hanging-node cross terms. It is
now the exact diagonal of the restricted matrix:

```
D_j = M_jj + 2 sum_h w_hj M_jh + sum_{h,h'} w_hj w_h'j M_hh'
```

`j` is a free column of `Cu`, `h` and `h'` are hanging rows that name `j` as a master,
`w_hj` is the constraint weight of `h` on `j`, and `M` is the leaf mass matrix weighted
by the square root of eta. Every term is leaf-local, because a hanging row shares its
leaf with its constraint's local masters, so the diagonal is assembled per leaf and `M`
is never formed. The result is the iteration count: 81 to 71 on `F` (12.3 %), 65 to 65
on `P1`. The warm solve fell from 1.507-1.548 s to 1.324-1.334 s at eight threads across
two sessions and from 2.070 to 1.791 s at one, and the accuracy improved with it:
`|du|/|u|` went from 1.694e-6 to 7.956e-7, 2.1x closer to the equilibrated direct solve.
The mechanism is the cross terms: re-clipping the 16 non-positive entries of the lumped
restriction gives 80 or 155 iterations, not 71, and the relative L2 change in `D` is
1.54 % on `F` against 0.086 % on `P1`. The diagonal is bitwise identical at one and
eight threads, and its construction is 3x cheaper at eight threads (0.088 to 0.028 s on
`F`, 0.123 to 0.032 s on `P1`).

### 2.3 The direct solve, for the comparison

Everything above is the path that runs. The direct path still owns the accuracy
reference and the bitwise path, so its cost decomposition stays in the file. The phase
split is the same three-state measurement as before, and every number in it is the
**pre-MG baseline** as a step cost; the solve itself is current, because the direct path
did not change.

| phase | `F` | `P1` | `P2` | share of `P1` |
|---|---|---|---|---|
| assembly (whole `assemble_only`) | 0.353 | 0.420 | 0.452 | 15.5 % |
| LU pair in `_shift_ladder_solve` | 1.581 | 1.991 | 1.923 | 73.4 % |
| final residual and unpack | ~0.05 | ~0.05 | ~0.05 | 1.8 % |
| unattributed (GC, allocations, call overhead) | 0.11 | 0.25 | 0.11 | 9.3 % |
| warm solve, median of three | 2.092 | 2.713 | 2.534 | 100 % |
| warm solve, minimum of three | 1.983 | 2.632 | 2.327 | n/a |
| allocation per solve | 1.67 GB | 1.96 GB | 1.96 GB | n/a |

Write the solve as `T_solve = T_asm + T_ladder + T_end + T_other`, where `T_asm` is the
whole assembly, `T_ladder` the whole shift ladder including both factorisations, the
shift build, the residual evaluations and the back-substitutions, `T_end` the final
residual and unpack (`Kred*u + Bt*p - fr`, `Bs*u - gr`, `u0 + Cu*u`) and `T_other` GC,
allocation and call overhead. The components were measured separately and overlap by a
few percent, so they add to the total within plus or minus 10 %; the percentages are
shares of the `P1` median. The assembly is 15.5-17.8 % of the solve across the three
states, the ladder 73.4-75.9 %. The minimum-of-three column is why the earlier notes
quote 73-83 %: against the fastest sample the same ladder is a larger share.

### 2.4 The assembly

| item | `F` | `P1` | `P2` | share of `P1` assembly |
|---|---|---|---|---|
| leaf kernel (eta, fz, `K`, `B`, COO, `fv`) | 0.200 | 0.255 | 0.262 | 60.7 % |
| of which closure sampling (`eta_fun` + `fz`) | 0.145 | 0.170 | 0.170 | 40.5 % |
| of which `K` local block and COO pushes | 0.054 | 0.065 | 0.072 | 15.5 % |
| of which `B` block, pushes and `fv` | 0.007 | 0.020 | 0.020 | 4.8 % |
| `sparse(K)` | 0.0185 | 0.0217 | 0.0226 | 5.2 % |
| constraints (`Cu`, `Cp`) and affine rows | 0.050 | 0.047 | 0.080 | 11.1 % |
| `Kred = Cu'*K*Cu` | 0.021 | 0.024 | 0.032 | 5.8 % |
| `S = [Kred Bt; Bs 0]`, right-hand side | 0.0092 | 0.0217 | 0.0107 | 5.2 % |
| leftover (allocation, dispatch) | 0.035 | 0.029 | 0.021 | 6.9 % |
| total | 0.353 | 0.420 | 0.452 | 100 % |

The closures are 40 % of the assembly. Each quadrature point calls `T_at` twice (once for
eta, once for buoyancy) and `semquad_strain_rate_at` once; on `P1` that is 810 000
points at 0.110 s for eta and 0.059 s for buoyancy, 0.21 microseconds per point. The
count is what changed relative to the historical measurement, and the kernel is not.

| metric | historical frame-19 | `F` | `P1` | `P2` |
|---|---|---|---|---|
| leaves | 1 024 | 8 188 | 10 000 | 10 000 |
| p | 4 | 2 | 2 | 2 |
| velocity dofs | 17 609 | 61 964 | 76 988 | 74 154 |
| saddle dofs | 45 475 | 69 388 | 86 544 | 82 951 |
| quadrature points | 36 864 | 663 228 | 810 000 | 810 000 |
| assembly | 0.441 | 0.353 | 0.420 | 0.452 |
| ratio to historical | 1.00 | 0.80 | 0.95 | 1.03 |
| per leaf | 0.431 ms | 0.043 ms | 0.042 ms | 0.045 ms |
| per saddle dof | 9.7 us | 5.1 us | 4.9 us | 5.4 us |
| per quadrature point | 12.0 us | 0.53 us | 0.52 us | 0.56 us |

At p=2 a leaf does 7.7x less matrix work than at p=4, but the trees have 8-10x more
leaves and 2.25x more quadrature points per leaf; the two effects land on the same
absolute time. Every normalization says the p=2 kernel is faster: 0.042-0.045 ms per
leaf against 0.431, 4.9-5.4 us per saddle dof against 9.7, 0.52-0.56 us per quadrature
point against 12.0.

The matrix-free path does not pay this assembly. It pays a reduced assembly: eta and the
buoyancy closure on the leaves, the constraint matrices, and the coarse operator, with
no `K`, no `sparse(K)` and no `Kred` product. The 0.236 s of §2.1 against 0.420 s is
that difference, and it is the only part of the direct decomposition the campaign
removed.

### 2.5 The LU pair

| item | `F` | `P1` | `P2` |
|---|---|---|---|
| unshifted `lu(S)` | 0.906 | 1.058 | 1.015 |
| of which symbolic (analysis and ordering) | 0.162 | 0.209 | 0.200 |
| of which numeric | 0.751 | 0.890 | 0.826 |
| of which back-substitution | 0.027 | 0.031 | 0.030 |
| shifted `lu(S + a*dmax*I)` | ~0.58 | 0.595 | 0.560 |
| its back-substitution | 0.027 | 0.031 | 0.030 |
| shift build | ~0.007 | 0.007 | 0.007 |
| residual evaluations (two per rung) | ~0.04 | ~0.04 | ~0.04 |
| ladder total, two rungs | 1.581 | 1.991 | 1.923 |

The symbolic phase is 0.16-0.21 s, 7-8 % of the solve. The numeric factorization is the
top item, 0.75-0.89 s, and the unshifted factorisation alone is 39-43 % of the solve.
The ladder's answer differs by state: on `F` the unshifted residual 4.0e-4 already
passes the driver's `stokes_accept_tol=1e-3`, yet the ladder still tries the 1e-12 rung
before returning the unshifted answer, a control flow pinned by G207. On `P1` and `P2`
the unshifted residual is 2.2e9 and 420, and the 1e-14 rung is what makes the solve
usable (1.2e-6 and 3.7e-7). The equilibrated variant pays an unshifted LU before the
scaled one, which is the 0.78 s of §8 and the reason the accuracy-matched direct column
of §3 is 3.885 s rather than 3.08 s.

### 2.6 Why UMFPACK takes its unsymmetric branch

The saddle has a zero pressure block, so only the velocity rows carry a diagonal. Write
`n_v` for the velocity unknowns, `n_p` for the pressure unknowns, and
`f_diag = n_v / (n_v + n_p)` for the fraction of rows with a structural diagonal. On
`F`, 61 964 / 69 388 = 0.893; on `P1`, 76 988 / 86 544 = 0.890; on `P2`, 74 154 /
82 951 = 0.894. UMFPACK's Rule 3 wants `f_diag >= 0.9` before it uses its symmetric
strategy, and 0.89 is just under. The solver therefore takes the unsymmetric branch:
COLAMD ordering, threshold partial pivoting with no diagonal preference, and no 2x2
pivots.

The price is visible in the fill. Write `r_fill = nnz(L+U) / nnz(K)` for the fill ratio.
On `F` the factor has 31.3 M nonzeros against 2.14 M in `K`, `r_fill = 14.6`; on `P1`,
36.8 M against 2.61 M, `r_fill = 14.1`. The shifted factorisation adds a full diagonal,
so `f_diag` becomes 1.0, the same Rule 3 now selects the symmetric strategy, and the
fill drops to 12.0 M on `F`. The factorization drops from 0.91-1.06 s to 0.56-0.60 s
for the same reason. The branch assignment is inferred from the measured fill and the
published rule; the fill and the times are measured. The practical consequence is that
the ladder cannot be priced as `2 * lu(S)`.

### 2.7 The corrected history

The 6.65 s figure was measured on the first direct solve in a fresh process. The 5.6 s
was obtained by subtracting the warm raw `lu(S)` (1.02 s) from that cold solve. Run the
first/second-solve experiment and the shape reproduces.

| measurement, same process | `F` | `P1` |
|---|---|---|
| first `assemble_only`, JIT cold | 19.79 s (`--threads=8`, fresh) | 9.10 s (first call) |
| first direct solve | 5.24 s (`--threads=8`, fresh) | 16.40 s (first driver step) |
| second direct solve | 2.25 s | 3.08 s (second driver step) |
| cold / warm | 2.33x | 5.3x |
| warm solve, dedicated phase run | 2.09 s | 2.63 s |

So the 5.6 s is JIT, first touch and the second factorisation, none of which is
assembly. The warm direct solve is 2.09 s on `F`, 2.63 s on `P1` and 2.33 s on `P2`, and
the assembly inside it is 0.35-0.45 s. The 6.65 s belongs to the same family of
cold-start numbers and is labelled pre-MG wherever it appears. The s340 average of
5.37 s/step is not a direct-solve time either: it includes AMR, frame output, the energy
solve and the Picard loop. The clean numbers are the phase runs.

### 2.8 Cheap knobs, and what a rewrite would buy

Every cheap knob on the path was checked. `sparse(I,J,V,m,n)` is a two-pass counting
sort, and a `sortperm` of the same triplets costs 117-167 ms against 18.5-22.6 ms for
`sparse(K)` and 3.9-4.5 ms for `sparse(B)`; the COO `sizehint!` is already exact; the
constraints are global, one `Cu`/`Cp` per solve at 0.07-0.11 s including the affine rows
and `C'AC`; nothing is assembled twice, and the 1.67-1.96 GB allocated per solve is the
matrix and its factor, not waste.

The proposed rewrite targets the element kernel and COO pushes (0.065 s, gain
0.04-0.05), sparse construction (0.026 s, gain 0.01-0.02), constraints and `C'AC` by
local condensation (0.10 s, gain 0.05-0.08) and allocation pressure inside the 0.30 s
unattributed (0.05-0.10 s, inference). That totals 0.15-0.25 s, 4-7 % of the old 3.58 s
step and 5-9 % of the current 2.72 s step. It does not touch the closures (0.170 s), and
even a free assembly is 0.42 s, 12 % of the pre-MG step. The lever above the bar is the
LU pair: a symmetric-indefinite backend is a 0.6-1.0 s saving (inference) and breaks
bitwise, reordering alone is 0.1-0.3 s, and an explicit block factorization loses
40-100 s. §8 keeps the ledger.

The campaign's verdict on this list is the same one the direct phase split gave: the
assembly rewrite is below the bar on the current step, and the numbers that once seemed
to justify it (the 6.65 s and the 5.6 s) were cold-start arithmetic. The 0.15-0.25 s was
never measured as a step-time gain, and the matrix-free path removes 0.18 s of it by
construction.

## 3. Matrix-free multigrid against the direct solve

### 3.1 What the matrix-free path is

`mg_operator=:matrixfree` builds an h-multigrid hierarchy by merging each leaf quartet
one level at a time and restoring the 2:1 balance after each merge. The velocity
operator is never assembled: the leaf kernel applies `K(eta) v` as a tensor-product
contraction with `eta` sampled at the quadrature points, wrapped in `Cu` and `Cu'` so
the hanging constraints apply without forming `C'AC`. The pressure block is the
three-factor w-BFBT

```
S~^-1 = S1^-1 S2 S1^-1,   S1 = B D^-1 B',   S2 = B D^-1 K D^-1 B',   D = diag(M_u(sqrt(eta)))
```

with `B` the reduced divergence block (`B'` the gradient), `K` the viscosity-weighted
velocity stiffness that this path never forms, `M_u` the velocity mass matrix, `eta` the
viscosity field at the quadrature points, and `D` the diagonal of the mass matrix
weighted by the square root of viscosity. `S1` is factorised once per tree with a sparse
Cholesky (CHOLMOD) and applied twice per iteration; `S2` is applied matrix-free. The
three factors are what make the pressure preconditioner carry the physics, and the two
`S1` solves are what make the block serial (§2.2).

Smoothing is damped Jacobi on a per-level schedule. `:auto` selects `:taper`
(`1,2,2,4,4,8,8` pre and post from the finest level) with `omega = 0.55` when the
hierarchy is top-heavy (`nfs[end-1] > 0.5 nfs[end]`), and the uniform 2+2 with
`omega = 2/3` otherwise; on the production pair the resolved choice is 2+2 on `F` and
3+3 on `P1`. The hierarchy coarsens by at least 1.5x per level (`mg_coarsen_factor=1.5`
on the matrix-free path), which removes the near-duplicate level that used to cost a
third of every cycle and takes `F` from 8 levels to 6. The leaf kernel is 32-lane, and
the outer solver is FGMRES with one V-cycle per iteration.

### 3.2 The comparison basis changed twice

The first change was the direct solve under comparison. The unscaled direct ladder stops
at 1.317e-3 on `F`, 1 300x above the 1e-6 target, while the equilibrated solve reaches
9.97e-11. Comparing a 1e-6 iterative solve with a direct solve that returns a 1e-3
answer measures two different problems, and the earlier documents did exactly that. The
reference is now `linear=:direct, stokes_scaling=:equilibrated`, and the equilibrated
variant's own residual is 9.97e-11 on `F` and 1.84e-9 on `P1`, so the gate is not limited
by the reference.

The second change was the thread count. The direct assembly leaves `assemble_threads` at
1 and UMFPACK's LU is serial, so `--threads=8` leaves the direct path unchanged. The
eight-thread row is threaded MG against serial direct: it is the production
configuration's number, and it is not an equal-thread algorithmic comparison. The
equal-thread rows are the one-thread rows, and they are the conservative ones.

Both changes moved the ratio, and the direction of the move is the campaign. The old
numeral was "MG is 1.6x slower at 11 s against 6.65 s"; the 11 s and the 6.65 s are both
cold-start figures (§2.7), and the 1.6x was a comparison against a direct solve that
did not reach the stated tolerance.

### 3.3 The same-session head-to-head

One process per state and thread count, both paths on the same read-only checkpoint and
the same closure objects, `BLAS=1`, `tol=1e-6`. **Cold** clears the solver caches and
calls `GC.gc()` before the call, median of three samples; **warm** is the production
in-tree step, median of three consecutive solves on the same tree and eta, no forced GC.
Assembly and factorisation are timed separately on the same assembled system.

| state | thr | direct cold | direct warm | MG cold | MG warm | direct/MG cold | direct/MG warm | MG it |
|---|---|---|---|---|---|---|---|---|
| `F` | 1 | 3.713 | 3.773 | 2.283 | 1.929 | 1.63x | 1.96x | 81 |
| `F` | 8 | 3.454 | 3.372 | 1.790 | 1.457 | 1.93x | 2.31x | 81 |
| `P1` | 1 | 4.219 | 3.791 | 3.252 | 2.531 | 1.30x | 1.50x | 65 |
| `P1` | 8 | 3.758 | 3.885 | 2.247 | 1.850 | 1.67x | 2.10x | 65 |

The range over the four cells is 1.30 to 2.31x, and every cell favours MG. The eight
cells read down the table are the same-session comparison, and the MG column is the
pre-diagonal build (81 and 65 iterations); §3.5 carries the shipped row after the exact
diagonal. The direct column is the accuracy-matched equilibrated path, and its cold/warm
split is allocator and GC placement rather than a different algorithm, because the
direct path has no hierarchy cache. The MG warm row is the real in-tree reuse.

The component table behind those seconds is what shows where the margins come from:

| state | thr | path | assembly s | setup s | factor./solve s | iteration s (nit) | peak RSS GB |
|---|---|---|---|---|---|---|---|
| `F` | 1 | direct | 0.331 | n/a | 3.364 (1.089 + 2.282) | n/a | 2.50 |
| `F` | 1 | MG | 0.229 / 0.254 | 0.342 | n/a | 1.585 (81) | 1.73 |
| `F` | 8 | direct | 0.342 | n/a | 3.052 (1.072 + 2.137) | n/a | 2.37 |
| `F` | 8 | MG | 0.214 / 0.256 | 0.347 | n/a | 1.113 (81) | 1.30 |
| `P1` | 1 | direct | 0.458 | n/a | 3.355 (1.083 + 2.247) | n/a | 2.78 |
| `P1` | 1 | MG | 0.264 / 0.501 | 0.446 | n/a | 2.088 (65) | 1.44 |
| `P1` | 8 | direct | 0.410 | n/a | 3.039 (1.093 + 1.928) | n/a | 2.29 |
| `P1` | 8 | MG | 0.268 / 0.478 | 0.454 | n/a | 1.272 (65) | 1.32 |

The MG assembly column is **timed / inferred**: the first number is a standalone
`assemble_only=:mg` call, the second is the convention `cold - setup - iteration`, which
also carries the allocator/GC state of the whole call; they differ by 0.03-0.24 s. MG
setup is hierarchy + w-BFBT, with the hierarchy 0.155 / 0.193 / 0.297 / 0.304 s and the
w-BFBT build 0.187 / 0.154 / 0.149 / 0.150 s in table order. Direct factor./solve is the
saddle build, the unshifted LU, its residual check, the equilibrated LU and its residual
check; `cold` also includes the final residual, unpack and allocator/GC state, so the
separately timed assembly plus factor./solve need not add to the cold row exactly (the
difference is 0.02-0.41 s across the four rows). Peak RSS is the process high-water
mark, reset through `/proc/self/clear_refs` immediately before the cold solve; it
includes a 1.29-1.40 GB Julia-plus-checkpoint baseline, so the solve-only difference is
larger than the column suggests.

The old direct number is reproducible, but it is not the same accuracy. With
`stokes_scaling=:none`, the pre-equilibration production path, the same process measures
2.081/2.170 s cold/warm on `F` at one thread, 1.877/1.950 s at eight, 2.486/2.430 s on
`P1` at one and 2.337/2.201 s at eight. On `P1` the tiny-shift rung reaches 1.166e-6,
17 % above the 1e-6 target, so it is a near-target reference there; on `F` it stops at
1.317e-3 (phase `:ladder`), 1 300x above the target, so it cannot be the reference.
Against this control MG is 1.12x faster on `F` at one thread (1.929 median against 2.170
single sample) and 1.04x slower on `P1` at one thread (2.531 against 2.430), and faster
in the eight-thread warm step on both states (1.457 against 1.950 on `F`, 1.850 against
2.201 on `P1`).

### 3.4 The verdict

At eight threads, the production configuration at this size and the one `linear=:auto`
selects, the tuned MG solve beats a direct solve that actually reaches 1e-6 by 1.93x cold
and 2.31x warm on `F`, and by 1.67x cold and 2.10x warm on `P1`. At one thread it is
1.63x and 1.96x on `F`, 1.30x and 1.50x on `P1`. Stated as one range over the two
production states and the two protocols, **MG is 1.3 to 2.3x faster than a direct solve
that reaches the tolerance**.

Three caveats carry equal weight with the range, and each is a property of the
measurement rather than a hedge. The eight-thread direct column is a serial algorithm:
`assemble_threads=1` and UMFPACK's LU do not use the other seven cores, so the eight
threads belong to MG alone and the equal-thread algorithmic comparison is the one-thread
row. The unscaled direct path does not reach the 1e-6 tolerance on `F`, so a comparison
against it would be a comparison against a non-converged answer; the equilibrated path is
the reference because it is the one that converges. And the measurement is 2-D, at 8 188
and 10 000 leaves and 69 388 and 86 544 saddle dofs, on two states of one rheology: no
extrapolation to 3-D, to other contrasts or to a larger leaf budget is measured here,
and §8 lists those as work to do rather than results to quote.

### 3.5 The row the campaign shipped

The head-to-head above predates the exact BFBT diagonal. After it, the shipped `F` row
is 71 iterations and 1.356 s at eight threads, with `|du|/|u| = 7.96e-7` and the
gauge-removed pressure at 2.27e-7 against the equilibrated direct solve whose own
residual is 9.97e-11. Against the same session's direct column of 3.372 s that is
**2.49x**. The pairing is cross-session: the 3.372 s is the §3.3 session and the 1.356 s
is the later one, so 2.49x is the campaign's headline number and 2.31x is the
same-session floor for the same cell. The campaign's own closing note quotes 1.356 s and
2.49x, and marks the 1.457 s / 2.55x pair of the earlier session as superseded. The
1.457 s row carried 81 iterations, not 71; the two numbers are one stage apart.

The whole-family sweep at `c7e0f55` re-measured both columns in one session and gives the
post-diagonal picture at `mg_tol=1e-6`. At one thread the ratios are 2.27x on `F` and
1.60x on `P1`; at eight threads 2.66x and 2.06x. Those rows use the same protocol as
§3.3 and are the numbers the selector's table is built from (§4.1). For the card's own
tolerance (`mg_tol=1e-8`) the eight-thread warm solve is 1.414 s on `F` and 2.208 s on
`P1` (§2.1), which puts the card at 1.57-1.83x against the 3.372/3.885 s direct column
at the tighter tolerance; the tolerance is part of the comparison, and §2.1 is where it
is priced.

### 3.6 What this supersedes

| stale claim | where | superseded by | why |
|---|---|---|---|
| "the setup alone is already the cost of the direct solve" | §3.8 of the pre-campaign record, [MG_LITERATURE.md](MG_LITERATURE.md) §0 | this section | the setup is 0.34-0.45 s against a 2.1-4.2 s direct solve |
| "MG is 1.6x slower at 11 s against 6.65 s" | the commit message of `5b4c670` | §3.3 | 11 s and 6.65 s are cold-start figures; the warm pair is 1.93/1.46 s against 3.37/3.89 s |
| 9.8-11 s repaired MG rows | [HANDOFF.md](HANDOFF.md) §2.2, §3.4, the pre-tuning §3.8 | §3.3 | the tuned warm solve is 1.46-1.85 s |
| "direct stays the production default" | `5b4c670` | §4.4 | `linear=:auto` is the library default since `2453f16` and resolves to MG at both production states |
| "MG loses by 1.38x on the 25-frame run" | §6.3 of the pre-campaign record | §6.3 | the frame-by-frame re-measurement is 1.55x MG at 1e-6, every frame |
| the endpoint is 1.324 s and 2.55x | [MG.md](MG.md) §0 | §3.5 | same row, later session; 71 iterations and 7.96e-7 are unchanged |
| the old crossover table | §4.2 | §4.1 | it was measured at `mg_tol=1e-10` on the pre-campaign code |

### 3.7 The pre-tuning record, kept because it motivated the campaign

`F` used to fail in production: 100 FGMRES iterations, true residual 9.718e-1, 11.43 s
of iteration, 25.7 s end to end against 6.65 s for direct, and 1.38 GB peak. Continuity
stayed at 2.2e-7 throughout; all of the residual was velocity, which exonerated the
pressure block and the Schur preconditioner. The stall had one proximate cause, a
constrained-diagonal defect in the matrix-free smoother; [BUG_LEDGER.md](BUG_LEDGER.md)
carries the defect and its fix. With the exact diagonal the same hierarchy reaches 70
iterations and 6.07e-7, and `mg_vel_solver=:cg` stops throwing.

The repaired version still converged slowly, and the reason was the coarse space. The
stiff lid is a boundary layer the coarse Gauss points do not sample: the operator saw
`eta_max` of 1.4e2, 3.3e3 and 1e4 at the three coarsest levels on `F`, against 1e4 in
the fine operator. With `E_c = ||P' A_f P - A_c|| / ||A_c||` the relative inconsistency
is 19, 1.7 and 0.36 on `F`'s three coarsest levels, and 31, 18.3 and 5.95 on `P1`; the
coarse correction is 31x, 412x and 842x worse than the optimal correction in the same
coarse space on `P1`. Galerkin coarsening (`A_c = P' A_f P`) removes the scale error
exactly: it takes `F` from 69 iterations to 61 and `P1` from 65 to 46, and the coarse
space stays weak, with the optimal correction still 1.4-5.4x the input residual on `P1`.
That measurement is why the campaign attacked the kernel and the scheduling first: the
iteration count was not the cheapest lever, and the campaign's own diagnosis of the
plateau later put the upper bound at 1.93x/1.65x for a perfect coarse space (§8).

## 4. The crossover

### 4.1 The re-measured table

Re-measured at `c7e0f55` on 2026-11-13 after the campaign. Same-session A/B in one
process per state, both paths on the same tree and the same closure objects, `BLAS=1`,
direct `:equilibrated`, `:mg` with the shipped defaults at `mg_tol=1e-6` (the accuracy
the direct solve reaches), `t_min`; ratio = `t_direct / t_mg`, so above 1 means MG is
faster. One thread unless stated; the eight-thread ratios follow the table.

| state (leaves, p) | saddle dofs | contrast | direct cold / warm | MG cold / warm | ratio cold / warm | iters |
|---|---|---|---|---|---|---|
| uniform L2 (16, p=4) | 747 | 1 | 0.0084 / 0.0083 | 0.0072 / 0.0049 | 1.17 / 1.71 | 17 |
| uniform L3 (64, p=4) | 2 803 | 1 | 0.042 / 0.043 | 0.024 / 0.016 | 1.74 / 2.75 | 18 |
| uniform L4 at p=2 (256) | 2 467 | 1 | 0.019 / 0.018 | 0.013 / 0.009 | 1.48 / 1.91 | 15 |
| uniform L4 (256, p=4) | 10 851 | 1 | 0.222 / 0.262 | 0.094 / 0.068 | 2.36 / 3.86 | 19 |
| uniform L5 (1 024, p=4) | 42 691 | 1 | 1.516 / 1.555 | 0.469 / 0.211 | 3.23 / 7.36 | 11 |
| uniform L6 (4 096, p=4) | 169 347 | 1 | 8.334 / 8.477 | 2.338 / 1.216 | 3.56 / 6.97 | 11 |
| stratified 50x L3 (64, p=4) | 2 803 | 50 | 0.042 / 0.041 | 0.024 / 0.017 | 1.74 / 2.39 | 21 |
| stratified 50x L4 (256, p=4) | 10 851 | 50 | 0.226 / 0.274 | 0.103 / 0.068 | 2.20 / 4.04 | 22 |
| blob trajectory frame 1 (214) | 9 351 | 3 024 | 0.147 / 0.162 | 0.134 / 0.109 | 1.09 / 1.48 | 27 |
| blob trajectory frame 14 (628) | 28 135 | 5 087 | 0.910 / 0.778 | 0.691 / 0.387 | 1.32 / 2.01 | 29 |
| blob frame 19 (1 024) | 45 475 | 5 087 | 1.235 / 1.254 | 1.086 / 0.703 | 1.14 / 1.78 | 27 |
| blob frame 19 x4 (2 344) | 99 559 | 5 087 | 3.882 / 3.992 | 2.437 / 1.656 | 1.59 / 2.41 | 27 |
| blob frame 19 x16 (4 096) | 174 915 | 5 087 | 7.692 / 7.569 | 3.920 / 2.891 | 1.96 / 2.62 | 25 |
| production `F` (8 188, p=2) | 78 720 | 5.7e6 | 3.751 / 3.729 | 2.082 / 1.644 | 1.80 / 2.27 | 71 |
| production `P1` (10 000, p=2) | 95 308 | 3.3e6 | 4.112 / 3.912 | 2.962 / 2.449 | 1.39 / 1.60 | 65 |

The saddle-dof column uses the selector's convention, `2*Nu + Np` before Dirichlet and
hanging-node elimination; the reduced counts of §2 are 69 388 and 86 544 for the two
production states. At eight threads (`--threads=8`, `mg_threads=8`) the ratios are, cold
and warm: L2 0.91 / 1.09, L3 1.74 / 2.60, L4 at p=2 1.22 / 1.81, L4 2.41 / 3.94, L5
3.97 / 7.21, L6 4.38 / 7.59, stratified-50 L3 1.51 / 2.33, stratified-50 L4 2.15 / 3.48,
f19 1.17 / 1.55, f19x4 1.86 / 2.44, f19x16 2.07 / 2.77, `F` 2.33 / 2.66, `P1` 1.89 /
2.06. No cell of either table is below 1.0 in the warm protocol. The only sub-1.0 cell
in the whole campaign is the 747-dof L2 tree cold at eight threads, 0.91x on a 17 ms
solve.

The 25-frame trajectory is the row that flipped, and it is measured frame by frame on
the stored mesh and temperature of each frame: 214-1 024 leaves, 9 351-45 475 saddle
dofs, contrast 3 024-5 793. The per-frame warm ratio is 1.55 at one thread and 1.51 at
eight, time-weighted 1.60 and 1.53, range 1.24-1.99 and 1.11-1.91. Every one of the 25
frames is faster with MG at 1e-6. At the library default `mg_tol=1e-10` the same sweep is
still a win overall, mean 1.17 and time-weighted 1.21, but no longer on every frame: the
minimum is 0.81 at 17 135 and 19 413 dofs, the two frames the 20 000 constant keeps on
the direct side. That is the tolerance case the constant is set to protect against, and
it is why the flip is quoted at 1e-6.

**How much of this is the campaign and how much is the tolerance.** The old table was
measured at `mg_tol=1e-10`; the new rows are at 1e-6, matched to what direct reaches.
Repeating two rows at 1e-10 separates the two effects: uniform L4 is 2.03 cold / 3.11
warm (old table 3.04x) and blob f19 is 1.18 cold / 1.39 warm (old table 0.97x). So the
uniform rows barely moved at equal tolerance, which means they were calibrated
correctly, while f19 flips on the code change alone (0.97 to 1.39) and the 1e-6 protocol
adds a further 1.28x on the MG side. The FNV-level accuracy at 1e-10 is unchanged:
`|du|/|u| = 2.2e-11` on L4 and 2.9e-12 on f19.

### 4.2 The old table, kept as history

The table this volume used before the campaign, superseded by §4.1. It is a
fresh-process table: one process per case on the 2026-11-06 machine, `t_min` of 2-5
repeats, at the library default `mg_tol=1e-10`, on the code before the nine stages. Its
ratio column mixes two conventions, which is one of the reasons it was retired: the
3.0-3.9x entries are `t_direct / t_mg`, and the "direct 1.38x" entry is the reciprocal.

| regime | measured threshold | ratio |
|---|---|---|
| uniform, constant eta | from 8 450 velocity dofs, the smallest tested | 3.0-3.9x |
| high-contrast AMR | 1 024 leaves, 35 218 velocity dofs | 1.03x, a tie |
| high-contrast AMR | 2 344 leaves, 77 442 velocity dofs | 1.6x |
| high-contrast AMR | 4 096 leaves, 135 970 velocity dofs | 1.9x |
| production 25-frame trajectory | mean 639 leaves, about 30k saddle dofs | direct 1.38x |
| production tree class, extrapolated | 40-80k leaves, 4-8x today | inference, unmeasured |

Read it as the cold-comparison stance of its era, and it is wrong in three ways that
matter. The tolerance is the library default rather than the accuracy direct reaches, so
the blob rows are flattered for direct. The code predates the 32-lane kernel, the region
V-cycle, the 1.5 coarsening and the exact diagonal, so the matrix-free column is 2-5x
slower than the shipped code at the same state. And the last two rows are the ones the
campaign refuted: the trajectory flipped to an MG win on every frame, and the
40-80k-leaf extrapolation was replaced by two measured production points whose larger
state has the smaller margin (2.31x on `F`, 2.10x on `P1`).

### 4.3 The rule

The production selector runs before assembly, from two quantities: the saddle size
`saddle_dofs = 2*Nu + Np`, a node count, and the viscosity contrast measured on the eta
closure the solve is about to use, sampled on a fixed 33 x 33 grid (1 089 evaluations,
0.02 ms). The rule, in order:

```
if saddle_dofs >= 20_000                         return :mg
elseif contrast <= 100 && saddle_dofs >= 2_000    return :mg
else                                              return :direct
```

The symbols: `Nu` and `Np` are the velocity and pressure node counts before Dirichlet and
hanging-node elimination, so `saddle_dofs` is the selector's size convention and not the
reduced count. `contrast = max(eta) / min(eta)` over the 1 089 samples. When the rule
selects MG it also sets `mg_operator=:matrixfree` and `mg_threads=min(8, nthreads())`.

The constants are calibrated against measured crossings, and both moved down 3-4x from
the previous ladder because only the MG side got faster. **20 000** is the any-contrast
point. Every measured high-contrast state at or above it wins: 19 413 dofs 1.24x at eight
threads, 28 135 dofs 1.32x cold and 2.01x warm, 45 475 dofs 1.14x cold and 1.78x warm,
99 559 dofs 1.59x cold. Below it the high-contrast rows are ties or losses: 9 351 dofs
1.09x cold, and 17 135 and 19 413 dofs are 0.81-0.82x warm at the library default
`mg_tol=1e-10`, which is the tolerance case the constant also protects. It sits 2.1x
above the last measured high-contrast tie, where the old 60 000 sat 1.32x above its tie
at 45 475. **2 000** is the low and mid-contrast point: 747 dofs is the last measured
tie or loss (1.17x cold at one thread, 0.91x at eight), 2 467 dofs at p=2 is 1.22-1.48x
cold and 1.91x warm, 2 803 dofs at p=4 is 1.74x cold at both thread counts, and the
contrast-50 tree of the same size is 1.51-1.74x cold. The old 8 000 was never measured
below 10 851 dofs; the new 2 000 is bracketed on both sides.

The contrast-10 rung is gone. The two stratified-50 rows show that the
`10 < contrast <= 100` band costs the V-cycle nothing measurable against contrast 1, so
the band had no measurement behind its constant. The only contrast boundary left is 100,
which is where the MG pressure block switches from the mass inverse to w-BFBT: that is a
property of the MG configuration, and not a measured crossover. The full row-by-row
justification is in [SOLVER.md](SOLVER.md) §4.4 and pinned by `t_g191_auto_path.jl`.

The rule keeps a 1.5x margin: it picks MG only where the warm measurement is at least
1.5x, and keeps `:direct` only where the conservative cold measurement is a tie (at most
1.25x, ties going to the bitwise path). The rows between those bands (f19 1.14x cold,
1.78x warm; trajectory frame 1 1.09x cold, 1.48x warm) are resolved by the 20 000
constant. And `LAST_AUTO_FALLBACK` turns a wrong MG pick into time lost rather than a
failed run: if the MG residual misses the tolerance, the solve is redone directly.

### 4.4 What the selector resolves at the production states

At the two checkpoints the selector's own convention puts both states above the 20 000
rule: `saddle_dofs = 2*Nu + Np` is 78 720 on `F` (8 188 leaves) and 95 308 on `P1`
(10 000 leaves), so `linear=:auto` resolves to MG on both. On read-only copies with the
driver's kwargs (`stokes_tol=1e-3` forwarded as `auto_fallback_tol` and
`stokes_accept_tol`, and the card's `mg_tol=1e-8`) the auto solve measured 81 FGMRES
iterations and 8.6e-9 on `F` and 91 and 9.4e-9 on `P1`, with zero
`LAST_AUTO_FALLBACK` events; the warm best of three was 1.41 s and 2.21 s against
3.47 s and 3.92 s for direct. The verify100 MG arm runs the same path and reports
`auto_fallbacks=0` across 500 steps.

`linear=:auto` became the library default of `run_semquad_simulation` on 2026-11-13
(commit `2453f16`), and the realistic card passes it explicitly, so a bare driver call
and the card are the same policy. The solver call's own default is still
`linear=:direct`, and `:direct` remains the only bitwise path; `pic_mode=:newton`
requires it explicitly.

### 4.5 The cost of the decision

The sampling is 0.02 ms against 0.70, 1.63 and 2.58 ms for a full-field sample at 1 024,
2 344 and 4 096 leaves. The decision is 0.02 ms per step, which is 0.001 % of a 2.72 s
step, and the only way it can cost more than it saves is by picking MG where the measurement
says tie: that is what the 1.5x margin and the fallback are for. The selector is the
cheapest line in the file and the only one whose default is a measured rule rather than
a constant.

## 5. Assembly and transfer allocations

### 5.1 The instrument

The allocation card is `--track-allocation=user` over four production-shaped AMR cycles
of the same 10 000-leaf resume. The instrumenter costs about 35x in time (15 s per
cycle), so it measures bytes and attribution, not speed; budgets are pinned with
`@allocated` on a warm call plus a positive control on the historical implementation
(G215, G217). The per-call timings below are probe medians.

### 5.2 The chain

| stage | MB per 4 cycles | what changed |
|---|---|---|
| baseline `0b3f4bd` | 710.19 | n/a |
| budget-candidate fix | 601.95 | the 108.24 MB `Matrix` in `_budget_quartet` goes to 0.001 |
| stage A: flat neighbour table | 308.17 | 298.58 MB of per-query vectors gone |
| stage B: local 2:1 rebalance | 282.47 | 26.1 MB more; `balance!` 27.60 to 1.87 MB |
| indicator, transfer, energy audit | 30.50 | the remaining per-element temporaries |

### 5.3 Before and after

The before column is the post-AMR-refactor tree (`8501b53`), so this table isolates the
indicator, transfer and energy work from the tree work above.

| quantity | before | after | ratio |
|---|---|---|---|
| 4 AMR cycles, total | 282.47 MB | 30.50 MB | 9.3x |
| `gll.jl` | 166.02 MB | 0.004 MB | 41 000x |
| `semquad.jl` | 97.60 MB | 17.88 MB | 5.5x |
| indicator (20 444 face pairs) | 33.52 MB / 29.4 ms | 0.52 MB / 4.9 ms | 64x / 6.0x |
| nodal transfer (90 000 node evaluations) | 14.45 MB / 18.2 ms | 0.72 MB / 8.5 ms | 20x / 2.2x |
| energy assembly (latent and EV, 10 000 leaves) | 248.5 MB / 195.3 ms | 110.6 MB / 117.0 ms | 2.2x / 1.7x |
| energy solve (assemble and BDF2 solve) | 447.0 MB / 475.7 ms | 304.7 MB / 324.4 ms | 1.5x / 1.5x |
| `semquad_mass`, cold, `:l2` only | 1 372 MB / 518 ms | 505 MB / 240 ms | 2.7x / 2.2x |
| AMR pass, 12 cycles, mean | 0.1340 s/cycle | 0.0766 s/cycle | 1.75x |
| AMR pass, min of ten | 0.0982 s/cycle | 0.0524 s/cycle | 1.87x |
| 10 production steps, wall | 68.33 s | 66.94 s | 2.0 % |
| bitwise | n/a | identical everywhere | n/a |

The bitwise row is not decoration: the leaf-list FNV, the theta FNV, the 2b and blob
hashes and G217 all reproduce the pre-refactor values, so the 9.3x is a refactor, and
not a change of arithmetic.

### 5.4 What was wrong

Of the 282 MB, about 250 MB (88 %) was waste: per-element temporaries, a GLL rule
rebuilt per face pair and duplicate node evaluations. The out-parameter Lagrange forms
removed 160 B plus 80 B per hot call. The indicator got a workspace struct
(`SemQuadJumpWork`) and a cached GLL rule, so the per-pair buffer went from 1 584 B to
0, the 12.48 MB of `gll_rule` per four cycles disappeared, and the full indicator went
from 33.52 MB / 29.4 ms to 0.52 MB / 4.9 ms. The nodal transfer now evaluates each
unique global node once, 42 123 instead of 90 000, 14.45 MB / 18.2 ms to 0.72 MB /
8.5 ms.

The energy side was four hoists. The EV quadrature data (`make_fedata` and `d2tab`, one
`eigen` per leaf) is built once per assembly instead of once per leaf, removing 57.8 MB
and 44 ms. The nine COO arrays are pooled per block size, so the 58.3 MB per assembly at
p=2 is paid once. The latent temperature is evaluated leaf-locally, which removes the
search and takes 24.4 MB / 30 ms to zero at about 15 ms. The `semquad_mass` hint is the
cold-path win, 1 372 MB to 505 MB, and is not on the production card because
`transfer_mode=:nodal` is. `make_semquad` now sorts the leaf list once (7.02 to 6.34 MB),
and `face_pairs` uses a concrete `FacePair` type instead of a boxed
`Vector{NamedTuple}` (8.07 to 1.83 MB per four cycles).

### 5.5 The marking rule, and why the pass is a scan

The AMR pass decides what to refine with Dörfler marking. Write `eta_T` for the indicator
of leaf `T` (the gradient jump across its faces), `L` for the leaf set, and
`E = sum_{T in L} eta_T` for the total. For a Dörfler fraction `theta` (`refine_frac`,
0.2) the marked set is the smallest `M` contained in `L` with `sum_{T in M} eta_T >=
theta * E`; sorting `eta` once gives the threshold `tau_r` at the `theta * E` point from
above. Coarsening uses the same cumulative sum from below at `coarsen_frac` (0.05), with
a hysteresis threshold `tau_h = min(tau_c, 0.5 * tau_r)`, and a quartet merges when all
four children are leaves, none is protected, and `max(eta_child) <= tau_h`.

The fraction decides how many leaves are marked, not how much work the pass does. On the
10 000-leaf state the Dörfler set above 0.2 is 157 leaves on the raw indicator and 82 on
the card's indicator, while the indicator evaluates 20 444 face pairs per cycle and the
transfer visits 90 000 (leaf, node) pairs for 42 123 unique nodes. The pass cost follows
the full scan, which is why the fixes above were per-pair and per-node, and why the
allocation is dominated by scratch the marked set does not size.

### 5.6 The honest end-to-end effect

Ten production steps from the same checkpoint, `amr_every=5`, direct solver, two runs per
arm: 68.33 s before, 66.94 s after, a 2.0 % change that is inside this machine's
run-to-run spread. The AMR pass itself moves from 0.231 s to 0.122-0.134 s for the two
cycles in those ten steps, which is 0.010-0.011 s/step or 0.2 % of the 5.2 s
**pre-MG baseline** step of that audit. One AMR cycle is 1.8-3 % of a step, so even
deleting the pass could not move the step much.

The measurable win is the energy path: `solve_semquad_energy` 475.7 to 324.4 ms and
`assemble_semquad_energy` 195.3 to 117.0 ms in the same-process A/B, which is -0.15 to
-0.23 s/step expected and -0.09 to -0.16 s/step in the driver's own profile. The rest of
the audit's value is the 9.3x allocation (the pass no longer competes with the solver for
GC) and the three historical-implementation gates that pin the refactor bit for bit: the
pass was never where the step time was, and what it bought is headroom and a smaller
floor.

## 6. Scaling and memory

### 6.1 The uniform trees

The 2026-11-06 study (`SCALING`, earlier HEAD) ran one process per case with one thread,
BLAS 1, and a long-running unrelated job on four cores, so its absolute times are noisy;
every wall time is the best of 2-5 repeats in the same process.

| case | leaves | velocity dofs | direct | mg | mg / direct | cold direct | cold mg | peak direct | peak mg |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 256 | 8 450 | 0.278 | 0.117 | 2.4x | 1.397 | 1.350 | 1 034 | 971 |
| L5 | 1 024 | 33 282 | 1.696 | 0.735 | 2.3x | 3.026 | 2.708 | 1 906 | 1 203 |
| production 2.5:1 | 2 560 | 82 818 | 4.503 | 1.857 | 2.4x | 6.112 | 4.917 | 3 140 | 1 739 |
| L6 | 4 096 | 132 098 | 9.578 | 2.851 | 3.4x | 10.447 | 7.890 | 5 116 | 2 132 |

A log-log fit over the four steady-state points gives a time exponent of 1.26 for direct,
1.17 for mg and 1.18 for lagged; between L5 and L6 alone, 1.23 and 0.99. Solve memory
(peak minus the 0.93-1.0 GB Julia baseline) goes from 33 MB to 4.19 GB for direct, an
exponent of 1.74 over the four points and 1.08 between L5 and L6, which is what 2-D
sparse LU storage of order `n log n` looks like; mg goes from 0 to 1.24 GB, exponent
1.14. The peak-RSS exponents (0.56 and 0.28) are dominated by the constant baseline and
say nothing about the solver. The cold columns give mg/direct ratios of 0.97 / 0.89 /
0.80 / 0.76, the first-solve penalty: a 0.87-1.61 s allocator and GC warm-up, plus the
BFBT build for mg. There is no crossover in 2-D up to 1.32e5 velocity dofs: mg is
already 2.4x faster at 8 450 dofs and 3.4x at 132 098. The per-solve win is 2.4-3.4x and
the solve-memory win 2.8-3.5x. That is a speed-up, and not a change of regime.

The current code, at the pre-campaign HEAD of 2026-11-07, measured faster on the
matrix-free side at every size and on the direct side at L5 and L6:

| case | leaves | velocity dofs | direct | MF | MF / direct | peak direct | peak MF | solve mem direct | solve mem MF |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 256 | 8 450 | 0.353 | 0.116 | 3.0x | 1 118 | 1 013 | 149 | 39 |
| L5 | 1 024 | 33 282 | 1.604 | 0.505 | 3.2x | 1 772 | 1 408 | 698 | 229 |
| L6 | 4 096 | 132 098 | 8.857 | 2.286 | 3.9x | 5 139 | 2 597 | 3 328 | 923 |
| blob frame-19 | 1 024 | 35 218 | 0.971 | 0.939 | 1.03x | 1 640 | 1 314 | 455 | 161 |
| blob, 2.3x leaves | 2 344 | 77 442 | 3.648 | 2.273 | 1.6x | 2 784 | 1 970 | 1 258 | 527 |
| blob, 4x leaves | 4 096 | 135 970 | 7.374 | 3.920 | 1.9x | 4 447 | 2 633 | 2 665 | 888 |

The uniform ratios are 3.0-3.9x, the blob ratios 1.03-1.9x. Outer iterations are 26-28
on uniform trees and 37-40 on the blob field, and that difference is the whole reason the
crossover moves right: on a small tree the LU is cheap, while the iteration count is set
by the contrast and stays high. These are single numbers per side, best of 2-5 in one
process; §4.1 re-measures the same cases with a cold/warm protocol, which is why L6 reads
3.56x cold and 6.97x warm there and 3.9x here.

The L6 comparison has two readings that are easy to confuse. Serial against serial,
direct is 8.857 s and MF is 2.286 s, 3.9x. Cross-table, serial direct against the
8-thread MF full solve (1.269 s, §7.1) is 8.857 / 1.269 = 7.0x. The 7.0x is a valid
statement about the production default against the best measured configuration; it is
not a same-session controlled ratio for the solver alone, and §7 gives the controlled
threading number (1.48x at L6). The post-campaign controlled number at L6 is 4.38x cold
and 7.59x warm (§4.1), which is the one to quote.

### 6.2 The blob field

The blob results stand as the high-contrast record: a tie at frame 19 (1.03x) and 1.6x
and 1.9x at 2.3x and 4x the leaves in the pre-campaign table, against 1.48x, 2.01x,
1.78x, 2.41x and 2.62x warm in the re-measured table of §4.1. Outer iterations 37-40,
against 26-28 on uniform trees. The field is where the coarse-space weakness of §3.7
shows: the same contrast that makes the direct LU fill more also makes the matrix-free
iteration count higher, so the two effects partly cancel and the margin at 1 024 leaves
is the smallest in the file.

### 6.3 The 25-frame trajectory

| run | wall | note |
|---|---|---|
| reference era, direct | 266.0 s | earlier machine |
| reference era, MG | 328.6 s | MG 1.24x slower, pre-campaign code |
| parity era, MG | 449.1 s | 0.993x against direct 452.1 s, 2026-11-06 machine |
| before the energy fix, direct | 438.4 / 443.2 s | same session, 1.1 % apart |
| after the energy fix, direct | 158.6 / 159.4 s | 2.77x, bitwise identical |
| after the step-95 guard | 128.1 s | from 159.8 s; AMR cycle 31.637 to 0.0595 s |
| after the hot-path audit | 114.8 s | from 130.8 s, 1.14x |
| final A/B, direct | 119.40 / 117.72 s | two runs 12 minutes apart, 1.4 % |
| final A/B, MG (pre-campaign build) | 163.78 s | 1.37-1.39x slower at `mg_tol=1e-10` |
| final A/B, lagged | 178.58 s | 1.50x slower, 3 029 MB peak |
| frame-by-frame re-measurement | per-frame 1.55x MG at 1e-6 | every frame, §4.1 |

Only pairs within a row are comparisons. The 266 s and 452 s eras are different
machines; the 2026-11-06 machine is 1.7x slower on the shared per-step energy work than
the reference era. The 114.8 s and 119.4 s are the same HEAD within session noise
(2-4 %). The pre-campaign final A/B is the row that used to settle the production choice:
on a 25-frame run whose mean tree is 639 leaves, direct beat MG by 1.38x at the library
tolerance with the pre-campaign build, and the per-phase split said the whole difference
was the Stokes solve, 101.81 s against 138.97 s over 126 steps. That row is superseded:
the frame-by-frame re-measurement at 1e-6 makes every frame an MG win (mean 1.55x at one
thread, 1.51x at eight), and with Stokes at 79 % of the run a 1.55x Stokes win is about
1.39x on the whole trajectory (component arithmetic, marked inference). The old wall
clock stays in the table because it is the measurement that closed the pre-campaign
question, and the campaign's point is that the question was posed at the wrong
tolerance.

### 6.4 Memory

At L6 direct peaks at 5.14 GB and MF at 2.60 GB; the solve-only difference is 3.6x
(3 328 against 923 MB). The direct factor itself is only 46-56 MB at 69-87k saddle
dofs, with 31.3-36.8 M nonzeros, so the peak is the assembly plus the process, not the
factor. Peak RSS includes a 0.93-1.0 GB Julia baseline; compare solve memory when the
question is the solver. On the 25-frame run direct peaks at 2 183 / 2 423 MB and MG at
1 719 MB. Lagged is the memory pig at 3 029 MB because it keeps the stale LU plus an
equilibrated copy. The production pair at eight threads is 2.37 GB direct against
1.30 GB MG on `F`, and 2.29 against 1.32 on `P1` (§3.3), and that is the memory claim
the campaign makes: lower at every measured state, by 1.7-1.8x including the process
baseline.

## 7. Threading and determinism

### 7.1 What threads buy

The reference protocol is L6, 4 096 leaves, 131 070 velocity dofs, one process with
eight threads and BLAS 1, thread counts alternating inside the process, minimum of seven
samples. The table predates the 32-lane kernel and the region V-cycle, so its absolute
V-cycle milliseconds are the 8-lane code's; what it measures is the shape of the scaling.

| threads | V-cycle ms | V-cycle speedup | full solve ms | full solve speedup | bitwise |
|---|---|---|---|---|---|
| 1 | 36.41 | 1.00 | 1 875.9 | 1.00 | n/a |
| 2 | 31.72 | 1.15 | 1 629.3 | 1.15 | yes |
| 4 | 19.53 | 1.86 | 1 388.7 | 1.35 | yes |
| 8 | 16.41 | 2.22 | 1 268.8 | 1.48 | yes |

L5 scales worse on the same protocol: 8.49 ms V-cycle and 447.0 ms full solve at one
thread, 0.91x for both at two, 1.43x / 1.04x at four and 1.50x / 1.20x at eight. The
machine's efficiency cores take half the work with two workers, which is why two threads
lose. Pinning does not help: four physical cores' SMT siblings give 0.96-1.05x, and
eight physical cores give 1.93x V-cycle and 1.39x full solve, the same as the default
scheduler.

At 8 threads the V-cycle was 36 % of an outer iteration (16.41 of 45.3 ms) in that
protocol. Amdahl's bound for that share is `S_max = 1 / (1 - f)` with `f = 0.36`, which
is 1.57x for the full solve; the measured 1.48x is 94 % of the bound. The remaining
29 ms per iteration was FGMRES orthogonalisation, the true-residual BLAS-1 (an estimated
12-20 ms from memory traffic, not a measured split), three sparse products at about 1 ms
each, and two sparse Cholesky solves in the pressure block. A dot product is a
reduction, so threading it would break the bitwise property; that is the wall the stage
stopped at.

The post-campaign numbers are the ones that describe the shipped code. At the same
thread protocol the V-cycle is 9.024 ms at one thread and 5.095 ms at eight on `F`, a
1.77x, and 19.844 against 9.680 ms on `P1`, a 2.05x; the full solve moves 1.930 to
1.489 s on `F` (1.30x) and 2.591 to 1.822 s on `P1` (1.42x). The gap between the
V-cycle's speedup and the solve's is the serial half of the iteration (46 % on `F`/8,
37 % on `P1`/8) plus the setup, neither of which threads. Amdahl's bound from the serial
share alone is 1.85x on `F`; the measured 1.30x is 70 % of it, and the difference is the
21.8 % setup.

The threaded hierarchy build is more expensive (L6 12.9 to 53.6 ms, L5 3.6 to 9.7 ms)
because it builds a transposed prolongation and 32 scratch sets; the production rebuilds
once per tree. The bare velocity apply scales 4.28x at eight threads (3.62 to 0.85 ms)
and the constrained apply only 3.46x (4.11 to 1.19 ms) because the two `Cu` products
stay serial; threading those is worth about 0.3 ms per apply, 1 % under the bound. Those
two rows are 8-lane code; the 32-lane kernel below changes the constants and not the
shape.

### 7.2 What the campaign changed in threading

The campaign made the V-cycle threaded as a region and the apply 32-lane, and both
changes are stages with gates. The 32-lane apply is the largest single kernel change:
at 8 lanes LLVM unrolled the lane loop before the vectoriser saw it, leaving 0 `vaddpd`
and 248 `vaddsd` in the listing, and the raw kernel was 2 149.8 us against 2 187.4 us
for the historical scalar kernel, a 1.7 % win. At 32 lanes the lane axis becomes
`vaddpd`/`vmulpd`/`vsubpd` with `vgather` only for eta, and the raw kernel drops to
786-801 us, 2.72x. On one thread on `F` the level-6 apply went 2 436 to 1 056 us (2.3x),
the V-cycle 18.10 to 8.86 ms (2.04x), the cold solve 3.53 to 2.62 s (1.35x) and the
FGMRES iteration 2.66 to 1.58 s (1.68x). The width is a compile-time constant, and the
sweep that chose 32 measured the alternatives: 16 lanes is 1.05-1.11x slower per leaf,
8 lanes 3.02x slower (the vectoriser stops firing), and 64 lanes 1.14-1.15x slower
because the block overflows L1d.

The region is the other half. Every parallel op used to create its own task set, 6-9 us
even for an empty body; `SemQuadParRegion` publishes an op as a phase into a persistent
worker set, so a threaded level's pre-smoothing block, its 2+2 Jacobi sweeps, the
residual and the restriction share one fork/join, and its prolongation, `x .+= py` and
post-smoothing share a second. Forks per V-cycle fell from 15 to 2 on `F` and 43 to 4 on
`P1`, worker tasks from 120 to 14 and 344 to 28. The 8-thread V-cycle went 6.94 to
5.42 ms on `F` (1.28x) and 15.48 to 10.98 ms on `P1` (1.41x), and the iteration 1.455 to
1.339 s (1.09x) and 1.890 to 1.614 s (1.17x). The transfer and the two `Cu` products
became phases on top of that: the transfer is 1.8-2.1x slower inside a live region than
isolated, and in place the restriction column falls from 431 to 178 us on `F` and 322 to
247 us on `P1`; the `Cu` split takes a finest apply from 579.0 to 346.1 us on `F` and
821.8 to 417.1 us on `P1`. The aggregate from those two stages is the V-cycle 5.78 to
4.39 ms on `F` (1.32x) and 11.64 to 8.61 ms on `P1` (1.35x), and the warm step 1.824 to
1.698 s and 2.347 to 2.096 s.

The serial threshold is what makes the small levels cheap. A fork and join costs 6-9 us
regardless of the work, and at 318 dofs the serial subtract is 0.021 us and the Jacobi
update 0.155 us, so threading them is 40 to 330 times slower than running them.
`SEMQUAD_MG_SERIAL_DOFS = 20 480` runs the apply, the Jacobi sweeps, the subtract and the
transfers on the calling task below that size. On `F` at eight threads the level-1 and
level-2 applies fell from 34.15 and 49.20 us to 3.12 and 7.12 us, the V-cycle from 10.14
to 7.17 ms, and the iteration from 2.074 to 1.497 s (1.39x); `P1` went from 2.575 to
1.973 s (1.31x). The threshold is a 2-D measurement, quoted in §8 as a reason to
re-measure in 3-D: it is a level's dof count, and a 3-D level's dof count grows by 27/9
per leaf.

The eta lane table is the last kernel line. The apply used to read its viscosity one lane
at a time, `etav[kofs[b] + kq]`, so LLVM emitted a `vgather` per quadrature point and
lane vector. Each level's eta field is now re-laid into a block-major lane table once, at
operator construction, with the padded lanes of the last block repeating its final leaf
at weight 0.0. The `vgather` count in the compiled p=2 kernel goes from 24 to 0, the leaf
kernel is 1.158x faster on `F` level 6 and 1.162x on `P1` level 7, and the 8-thread
V-cycle is 1.075x on `F` and 1.063x on `P1`. The zero-allocation scratch is the
housekeeping that came with it: five lane vectors moved into `SemQuadKScratch`, the
serial apply allocates 0 B at every level, the threaded apply 4 976 B, and the 8-thread
V-cycle 1.05 MB, down from 1.83 MB. G222 pins 0 B serial and below 16 kB threaded.

### 7.3 Why the V-cycle stays bitwise

The threaded leaf kernel does not write partial sums into the node vector. It writes
each leaf's eight element vectors into per-leaf columns (`SemQuadKPar`, 25 x nleaf
doubles), and after the parallel loop it scatters them into the node vector serially in
leaf order. The serial kernel accumulates in leaf order and within a leaf in index
order, so the floating-point addition sequence is identical; the result is bitwise, not
merely deterministic. Restriction and prolongation use `muladd` to match the FMA kernel
that `mul!` uses for CSC; a hand-written `s += a*x` is not bitwise. G190 asserts
`out == ref` at 2, 4 and 8 threads on uniform and hanging-node trees, and that repeated
calls are bitwise repeatable. The transposed prolongation costs 31 MB at L6 and 7.7 MB
at L5, 3 % of solve memory.

### 7.4 Determinism, and the same-session rule

The direct path is bitwise by construction: G150 pins `linear=:direct` at 12/12, and the
2b and blob FNV hashes (`d22fdab099ae84e0`, `45ac8acd099adadc` and their companions) are
unchanged by every optimisation here. The matrix-free path is bitwise across thread
counts but not against direct: FGMRES and LU round differently, so the theta SHA
differs. Acceptance is 1e-10 on the fields plus 1e-6 on `Tmax`; the 25-frame A/B
reproduces the reference `Tmax` to a maximum relative 3.81e-10 (ten significant digits
in the reference), and MG and direct agree within 1e-12 in the printed values.

Every performance claim here is a same-session A/B unless it is marked cross-table, and
the rule is evidence-based. The same shared per-step energy work moved by 1.7x between
the reference era and the 2026-11-06 machine. Two direct 25-frame runs 12 minutes apart
differ by 1.4 %; the four pressure-cache samples span 159.60-165.25 s, plus or minus
1.7 %; direct phase components spread 5-12 %; ten-step wall times spread plus or minus
2 %. The campaign's own spread is 4-6 % run to run, and two measurements of the same
81-iteration `F` solve differed by 7.7 %. The same warm `F` row has been recorded at
1.324, 1.334, 1.356, 1.457, 1.489 and 1.505 s in different sessions, which is why this
file quotes shares and ratios and treats absolute seconds as session currency. Absolute
seconds identify a session; ratios measured inside one session survive the drift, and
ratios assembled across sessions do not. The one deliberate cross-table number here, the
7.0x L6 comparison, is labelled; the controlled statement beside it is 3.9x serial or
1.48x threaded.

## 8. The decision table

### 8.1 The bar

The bar for a funded performance change is 0.3-0.5 s per step. Against the pre-MG 3.58 s
card that is 8-14 %; against the current 2.72 s card it is 11-18 %. Below it, the change
cannot be distinguished from the machine's own spread in a production-length run.

### 8.2 What is still worth doing

| direction | measured or bounded gain | share of the warm solve | breaks bitwise | state |
|---|---|---|---|---|
| mixed precision in the apply | 1.2-1.8x on the apply; the V-cycle is 0.413 s of a 1.489 s `F`/8 solve, so 4.6-12.3 % | 5-12 % | yes, needs the accuracy gate | open, 3-5 days |
| a better-conditioned `S1` | the two `chol(S1)` solves are 1.90 ms/it, 13.2 % of the 8-thread iteration, against a 1.16 ms two-pass DRAM floor: 0.74 ms/it, 5.1 % of the iteration, is recoverable; an additive product would also let standard GMRES drop the `Z` basis (0.5-0.6 % of the iteration, 45 MB) | 4-6 % | yes | open, a redesign of the Schur block; the cheap routes measured 0.43x and 0.30-0.34x |
| 3-D | nothing measured; a p=2 hexahedral leaf has 27 nodes against 9, so the working set, the 20 480-dof threshold, the level dof counts and the `S1` conclusion (7 424 rows here against ASPECT's 1e5-1e6) are all 2-D numbers | n/a | n/a | measure, do not extrapolate |
| a 40k-leaf state | the premise is that the margin grows with size; the two measured states do not order that way (2.31x on `F` at 8 188 leaves, 2.10x on `P1` at 10 000) | n/a | n/a | measure with `bench_direct_vs_mg.jl` |
| a better coarse space (AMR-aware Galerkin) | Galerkin takes `F` 69 to 61 and `P1` 65 to 46; the plateau is 40 and 27 iterations, 0.58 and 0.56 s, 39 % and 31 % of the warm solve; a perfect coarse space bounds the warm solve at 1.93x and 1.65x, and a conservative 20 % iteration cut is 0.23 and 0.27 s, 16 % and 15 % | 15-39 % | yes | open, 2-6 weeks, below the first three |

The mixed-precision row is first because it needs no new preconditioner and no new
system. The evidence is external and direct: Kronbichler and Ljungkvist measure up to
+83 % from a single-precision V-cycle with a double-precision outer correction, and the
expected range here is 1.2-1.8x on the apply. It is not done because it cannot be
bitwise, so it forces a decision about carrying two verification bases, one bitwise for
the direct path and one accuracy-based for the matrix-free path.

The `S1` row is second because it is the only remaining item that attacks the serial
half of the iteration. The two `chol(S1)` solves are 1.90 ms of 14.37 ms at eight
threads and do not scale: 1.84 ms at one thread and 1.90 at eight, because CHOLMOD's
`cholmod_solve` is serial. One solve is 0.436 ms cache-resident, 0.34 ns per nonzero of
`L` (649 814 nonzeros, 7.8 MB); reading `L` twice at the measured 27.1 GB/s costs
0.581 ms per solve, so the pair has a two-pass DRAM floor of 1.16 ms. The remaining
0.74 ms, 5.1 % of the iteration, is what a smarter solve could recover, and both routes
that aimed at it measured slower than the exact one. In place a solve costs 0.948 ms,
2.2x the hot measurement, because the V-cycle and `S2` stream tens of MB through the
caches between two solves; just over half of the in-place cost is memory traffic. A
better-conditioned `S1` would need to change the Schur approximation itself, which
changes the preconditioner and therefore the iteration count, and it is the direction the
pressure-deflation probe points at without measuring.

The coarse-space row carries the largest measured bound and the longest schedule. It is
also the only one of the three that attacks the iteration count, and therefore the only
one that shrinks both the 46 % serial floor and the FGMRES own work. The 1.93x/1.65x is
an upper bound from extrapolating the late contraction rate back to iteration 1, and the
authors of that measurement say so; the conservative 20 % cut is the number to plan
against.

### 8.3 What is closed

| option | measured result | share of the step | breaks bitwise | verdict |
|---|---|---|---|---|
| assembly rewrite | 0.15-0.25 s | 5-9 % | no | closed: below the bar, and the 6.65 s premise was a cold-start subtraction |
| reordering round, same backend | 0.1-0.3 s (inference) | 4-11 % | yes | closed |
| symmetric-indefinite LU backend | 0.6-1.0 s (inference) | 22-37 % | yes | closed unless a funded multi-second win |
| equilibrated direct saddle | 0.78 s, unshifted residual 1.42e-10 | 29 % | yes | opt-in, default off |
| explicit block factorization | -40 to -100 s | negative | n/a | closed |
| Chebyshev smoothing | diverges: `F` 0.97, `P1` 0.99 after 500 iterations | n/a | yes | closed |
| slice-wise cache blocking | -2.0 % to +1.0 % | 0 | no | closed |
| `dest`/`tmp` aliasing | 682.5 to 682.0 us | 0 | no | closed |
| per-level hierarchy cache | 77.5 % hit rate, 77.9 s against 75.9 s | 0 | no | kept bitwise, not a speedup |
| lane width 16 / 8 / 64 | 1.05-1.11x / 3.02x / 1.14-1.15x slower | negative | n/a | closed at 32 |
| small-mode deflation | 83 to 79 iterations at best | at most 5 % | yes | closed |
| `S1`-based large-mode deflation | 2.6-3.2x the whole solve; subspace defect 5-100 | negative | yes | closed |
| standard GMRES | 196 iterations on `F`, `|du|/|u|` 3.58e-6 | negative | yes | closed |
| level-scheduled `S1` solve | 0.43x | negative | no | closed |
| inexact CG or IC(0) `S1` solve | 0.30-0.34x | negative | yes | closed |
| `bfbt_s1=:mass` | 500 iterations at `relres` 0.99 | negative | yes | closed |

### 8.4 What is done

| change | measured effect | share of the step | bitwise |
|---|---|---|---|
| energy element-loop de-boxing | 438.4 to 158.6 s on the 25-frame run (2.77x) | n/a | identical |
| step-95 AMR budget guard | 31.637 to 0.0595 s for the cycle, run 159.8 to 128.1 s | n/a | identical |
| stage-4 hot-path audit | 130.8 to 114.8 s (1.14x) | n/a | identical |
| AMR, indicator and transfer allocation | 282.47 to 30.50 MB; 10-step wall 2.0 % | 0.2 % pass | identical |
| pressure-cache threshold 1.0 | builds 41 to 21, wall -0.18 % | about 0 | identical |
| per-level V-cycle smoothing | frame-19 solve 1.22 to 1.06 s | about 4 % of a frame-19 step | identical |
| 32-lane apply | raw kernel 2.72x, V-cycle 2.04x at one thread | n/a | identical |
| zero-allocation kernel | 1.83 to 1.05 MB per 8-thread V-cycle | about 0 | identical |
| serial threshold 20 480 | iteration 1.39x on `F`, 1.31x on `P1` at eight threads | n/a | identical |
| region V-cycle | V-cycle 1.28x and 1.41x, iteration 1.09x and 1.17x | n/a | identical |
| transfer and `Cu` phases | V-cycle 1.32x and 1.35x, warm step 1.07x and 1.12x | n/a | identical |
| eta lane table | leaf kernel 1.16x, V-cycle 1.07x and 1.06x | n/a | identical |
| exact leaf-local BFBT diagonal | 81 to 71 iterations on `F`, warm 1.548 to 1.324 s | 11.5-14.5 % | not bitwise, gated |
| `linear=:auto` selector | 7/7 measured cases on the intended side, 0.02 ms sampling | about 0 | direct side bitwise |
| card `mg_tol=1e-8` | 12-20 % shorter warm solve than the library default, 5 orders inside the gate | 12-20 % | not bitwise, gated |

### 8.5 The reading

The campaign's result can be stated in one line: the matrix-free path became competitive
by removing duplicated work and by giving the kernel a shape the vectoriser keeps, and
every attempt to beat the remaining serial block with a better algorithm failed on
measurement. The next gain is a different arithmetic (mixed precision), a different
system (an augmented Lagrangian, months), a different pressure block (a better-conditioned
`S1`), or a different size (three dimensions, a larger leaf budget). The evidence for
each is in §8.2, and none of it is a promise.

One row that used to be open is now closed by default: `linear=:auto` resolves to MG at
both production states, so the direct path is the reference and the bitwise fallback
rather than the default. The reason to revisit the MG coarse space is the one-thread
iteration count, not break-even at a larger leaf budget, because the eight-thread
comparison already favours MG at 10 000 leaves.

## 9. Provenance

| date | commit | what it established |
|---|---|---|
| 2026-11-06 | `bd73018`, `1afae8b` | S3-press blob parity: MG 449.1 s against direct 452.1 s, 0.993x, with the honest 0.99-1.04x range |
| 2026-11-07 | `6c706a8`, `74494d1`, `23f667e` | per-step profile hook; the boxed energy loop: frame-19 step 4.55 to 1.51 s, run 438.4 to 158.6 s (2.77x) |
| 2026-11-07 | `4bbdae9` | step-95 AMR spike 31.637 to 0.0595 s, run 159.8 to 128.1 s |
| 2026-11-07 | `a04d2c2`, `58b9e15` | stage-4 audit 130.8 to 114.8 s; 83.6 s (72.8 %) is LU plus GC |
| 2026-11-07 | `3da92a3` | the 25-frame A/B, the 3.0-3.9x and 1.6-1.9x tables, the 2 000-leaf rule |
| 2026-11-07 | `471ef9b`, `502e30d`, `63b670c` | pressure-cache threshold 1.0; threaded V-cycle 1.48x at L6; `linear=:auto` and G191 |
| 2026-11-11 | `2d67185`, `861df6a`, `ab75905` | AMR hot-path audit and its reference |
| 2026-11-11 | `f9f32a3`, `957a43b`, `8501b53` | flat neighbour table, local 2:1 rebalance, 602 to 282 MB, and the merge |
| 2026-11-11 | `a967917`, `39dc2d2`, `2000627`, `17713c1` | SEM allocation reference, out-parameter Lagrange forms, scratch workspaces, G217 budgets |
| 2026-11-11 | `70f7438`, `bf5a219` | the assembly and transfer audit, 282 to 30.5 MB, 2.0 % wall |
| 2026-11-11 | `c05b3c9`, `09f9bb0` | the MG decision reference and the production-tree diagnosis |
| 2026-11-11 | `9250a3c`, `127b49e`, `5b4c670` | exact constrained diagonal: 0.972 to 70 iterations and 6.07e-7; the 11 s against 6.65 s verdict, later superseded |
| 2026-11-11 | `daa4c86` | sparse-direct reference: reordering, the 0.9 diagonal rule, the 5.6 s premise |
| 2026-11-11 | `913c75c`, `70f84dd`, `7c0b9b7`, `015b6f7` | the corrected warm cost split: one assembly per step, 0.35-0.45 s, the 0.15-0.25 s rewrite ceiling |
| 2026-11-12 | `83d649f` | MG tuning §11-§13 (32-lane kernel, region threading, transfer and `Cu` phases) and the first eight-thread win: 1.86/1.50 s on `F`, 2.25/1.85 s on `P1` against 3.28-3.63 cold and 3.51-3.91 warm |
| 2026-11-12 | the campaign's nine stages | the 1.5 coarsening, the finest-`Cu` dedup, the zero-allocation kernel, the 20 480 threshold, the region phases, the eta lane table, the exact diagonal: `F` 3.740 to 1.356 s at eight threads (2.76x), 71 iterations |
| 2026-11-12 | the verify100 pair | 500 steps from t = 0 on the production card: MG 1 398.9 s (2.798 s/step), direct 2 201.3 s (4.403 s/step), zero fallbacks |
| 2026-11-13 | `c7e0f55` | the crossover re-measurement at 1e-6, the 15-row table of §4.1 and the 8-thread ratios |
| 2026-11-13 | `8e648e6` | the crossover documented after the campaign |
| 2026-11-13 | `2453f16`, `fabbedd` | `linear=:auto` as the library default, its rule, the fallback safety net and the docs |

Three corrections are the point of the timeline. The first is the 6.65 s split: a
cold-start subtraction was read as an assembly cost and nearly funded a rewrite that the
warm phase run says is worth 0.15-0.25 s. The second is the parity era: 0.993x on the
blob was measured while 70 % of the step was shared energy and AMR work; after that work
was fixed, the same comparison is an MG win. The third is the comparison basis: the
pre-campaign verdict compared an iterative solve at 1e-6 with a direct solve that
returned 1e-3 on `F`, and once the direct side was the equilibrated path that reaches
1e-6, the sign of the verdict changed. All three came from keeping the pair inside one
session and checking what the denominator contains.

## Sources

| document | what this file takes from it |
|---|---|
| [MG.md](MG.md) | the nine shipped stages, the nine rejected directions, the endpoint, the BFBT diagonal, mixed precision and 3-D |
| [MG.md](MG.md) §11-§16 | the stage-by-stage numbers, the §14 head-to-head, the lane width and the slice-wise negative |
| [MG.md](MG.md) §11-§12 | the current solve composition, the serial half, the plateau, the standard-GMRES negative |
| [SOLVER.md](SOLVER.md) §4.2, §4.4 | the re-measured crossover table, the selector rule and its brackets, the fallback |
| [MG.md](MG.md) §4 | the shipped 71-iteration row and the two `S1`-solve negatives |
| [MG.md](MG.md) §10 | the exact leaf-local BFBT diagonal and its measurement |
| [CASE_REALISTIC.md](CASE_REALISTIC.md) §1.6, §3.4 | the card's tolerance rows and the E/G production walls |
| [AMR.md](AMR.md) | the 710.19 to 282.47 MB chain and the pass timing |
| [BUG_LEDGER.md](BUG_LEDGER.md) | the constrained-diagonal defect, the shift-ladder failure, the pressure classification |
| [MG.md](MG.md) §7 | the deflation negatives and the pressure-block conclusion |
| [MG_LITERATURE.md](MG_LITERATURE.md) | the external evidence behind the three named levers |
| [HANDOFF.md](HANDOFF.md) §4, §6 | the pre-MG baseline table and the open directions |
| `out_julia/verify100_mg.log`, `out_julia/verify100_direct.log` | the 500-step end-to-end A/B of §1 (2.798 and 4.403 s/step) |
