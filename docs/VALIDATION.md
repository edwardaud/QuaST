# VALIDATION.md - benchmarks, wall fluxes, the energy budget, and the ASPECT comparison

A number enters this document only with the conditions under which it was measured: the code path, the state or
window, the wall-flux family, and whether it is an instantaneous value, a window mean, or a bound. Every number is a
measurement or is marked as inference, and carries a tag naming the record it came from. The material merges five
records: the validation audit, the Nusselt-convention note, the cross-code comparison with ASPECT, the
diagnostics of `docs/CASE_REALISTIC.md` §7-§10, and the constitutive card of `docs/PHYSICS.md` §14. Tags point at them: `[CONV §n]`, `[AUDIT §n]`, `[COMPARE §n]`, `[CASE §n]`, `[RHEO §n]`,
`[REVIEW §n]` (the project-level review from the development tree, which is not part of this repository; see
[HANDOFF.md](HANDOFF.md)). Three wall-flux families recur. **A** is the
pointwise one-sided GLL derivative of the discrete field, `semquad_nu_walls`, reported by the production runs as
`Nu_top` and `Nu_bot`. **B** is the steady consistent boundary flux (CBF), `semquad_nu_cbf`, the residual of the
steady operator summed over the wall degrees of freedom. **C** is the time-step CBF, `Qc`, the same sum for the BDF2
step operator; ASPECT's `heat flux statistics` reports this family. An unqualified number is family A.

**Run artifacts.** The `out_julia/...` paths in this document name frame dumps, logs and
checkpoints from the development tree, and the `aspect_runs/realistic/output-*` paths name
ASPECT run directories from the same tree. They are cited so that each number can be traced
to its record; none of them is part of this repository.

## 1. What validation means here

Three activities with different powers. **Analytic and constitutive.** A manufactured solution is pushed through the
assembled operator and the error measured against the exact field, so the answer is a convergence order or a round-off
bound. `t_g154` reproduces the polynomial Stokes solution `u = (2xy, -y^2)`, `p = xy`, `eta = 1 + 0.2xy` to round-off
and shows spectral convergence in `p`; `t_g151` repeats this on a 2:1 tree with hanging constraints; `t_g26` measures
second-order BDF2 convergence. The constitutive card belongs here and is the strongest single check in the project:
our `eta_law = :composite` against an independent re-implementation of ASPECT's `diffusion dislocation` equations over
495 points in `(T, edot_II, z)`, with the branch laws and the parameter mapping agreeing to **6.2e-15** at worst over
four parameter sets `[RHEO §14.3]`, and the full isostress algorithm to **6.6e-15 .. 8.9e-15** after the 2026-11-08
switch `[RHEO §14.8.3]`. The card also caught a documentation error of 4.0e51 in `comp_Ad` and 6.1e15 in `comp_As`,
now fixed and pinned by `t_g200` `[RHEO §14.4g]`. This level establishes that the operator is the operator we think it
is; it says nothing about a convecting solution. **Benchmark solutions.** The whole pipeline, driven by a published
card, lands on a number other codes have reported. Its resolution is poor: four to six published digits and a sample
of one state. **Independent code comparison.** A second code gives a second solution, and the difference is the
measurable. This is the only level that can size the discretisation-induced uncertainty of a time-dependent solution,
and it cannot say which code is closer to the continuum.

### 1.1 Why the condition is the hard part

Two recorded incidents. A note circulated that the pointwise top flux and the CBF at step 7200 were 17 percent apart.
At the step-7202 state the measured spread across the three families is **1.0 percent** (`Nu_top` +7.6128 A, +7.6923
B, +7.6930 C) and **0.02 percent** at the bottom wall (-2.8338 / -2.8341 / -2.8341) `[AUDIT §0.3]`. No combination of
stored or recomputed EV field, zero velocity, steady or time-step CBF, or latent-capacity choice reproduces 17
percent; all three families give about **6.33** near step 6050, which is where the two halves of the quoted pair came
from `[AUDIT §0.3, §10]`. The bottom-flux difference over `t_nd = 1e-6 .. 1e-3` is -6.11 percent (CBF) and -8.44
percent (gradient) `[COMPARE §10.1, §12.5]`. Over `t_nd = 1e-3 .. 5e-3` the same comparison gives **+6.05 percent**
(CBF) and **+5.83 percent** (gradient), the instantaneous difference crosses zero at `t_nd = 1.42e-3`, and it
saturates at **+8.5 percent** in `[3.5e-3, 5e-3]` `[COMPARE §17.5, §17.7]`. One window mean quoted without its window
would have been wrong by 12 percentage points and by a sign. The same asymmetry explains why the codes agree on
`Nu_top` to 0.03 percent in the first window, where the top layer is thick, while the definitional floor at `t_nd >=
1.9e-3` is 1.0 to 1.2 percent `[COMPARE §10.1; AUDIT §3.1]`.

## 2. The benchmarks

### 2.1 Blankenbach 2b

A 2.5 by 1 box, free-slip walls, `Ra = 1e4`, no internal heating, and the joint viscosity law `eta = exp(-b T + c (1 -
z))` with `b = ln 16384 = 9.704060527839234` and `c = ln 64 = 4.158883083359672` `[REVIEW §2.1]`. The initial
condition is ASPECT's `ascii data` file (`julia/scripts/data/initial_temperature_case2b.txt`); the analytic function
in that input file was never used by ASPECT, and a near-conductive start produces a violent transient `[REVIEW §4.8]`.
The reference is Blankenbach et al. (1989) Table 9: `Nu = 6.9299`, `Vrms = 171.755`.

| diagnostic, level 3 (40 by 16) | value | relative to reference |
|---|---|---|
| `Vrms` | 171.18 | **-0.33 %** |
| energy family, `Nu` | 6.9874 | **+0.83 %** |
| wall gradient, `Nu` | 7.2405 | **+4.48 %** |
| steady CBF, `Nu` | 6.5622 | **-5.31 %** |
| mean of the two wall diagnostics | 6.9014 | **-0.41 %** |

`[REVIEW §4.1, §4.4, Appendix B]`. The wall families straddle the reference by +4.5 and -5.3 percent, and the ladder
says why. Holding the wall at level 6 and refining the interior:

| internal level | leaves | velocity dofs | `Vrms` | relative |
|---|---|---|---|---|
| 2 | 8 800 | 73 034 | 187.392 | +9.10 % |
| 3 | 9 040 | 74 890 | 174.742 | +1.74 % |
| 4 | 10 240 | 84 370 | 172.010 (EV 0.11) / 171.823 (EV 0) | +0.15 % / +0.04 % |

`[REVIEW §4.8]`. The 9 percent at level 2 is internal resolution; the entropy viscosity is immaterial on the steady
branch (0.11 percent between EV 0.11 and EV 0). On the level-4 wall-refined state the three families agree to **0.04
percent**: wall 6.9480, CBF 6.9325, energy 6.9330. The 4.5 percent at level 3 is a coarse-mesh error of the
functionals and disappears at the first refinement that resolves the wall layer, so a comparison quoting one wall
diagnostic at that resolution quotes the functional's error `[REVIEW §2.4]`. Two gates touch the case.
`t_g12_blankenbach` pins the viscosity law and its use in the Stokes assembly: the four corner values to 1e-14, the
contrast `16384 * 64` to 1e-12, and the refusal of the joint law by `eta_of_T`. `t_g150_regression_snapshot` pins a
20-step trajectory from ASPECT's initial condition on the 5x2 level-2 tree with EV on: `Nu_top = 7.029996422723808`,
`Nu_bot = 5.585292640587811`, `Vrms = 125.9039928721316`, at 1e-6 relative. The steady numbers above come from
`julia/scripts/run_2b.jl`, so they are a measured record; no gate asserts 171.18.

### 2.2 The rest of the Blankenbach family

| case | reference `Nu` / `Vrms` | our number | state |
|---|---|---|---|
| 1a | 4.884409 / 42.864947 | 10-step snapshot only | benchmark not run |
| 1b | 10.534095 / 193.21454 | none | open |
| 1c | 21.972465 / 833.98977 | none | open |
| 2a | 10.066 / 480.4334 | none | open |
| 2b | 6.9299 / 171.755 | 6.9874 / 171.18 | done, steady |

`[REVIEW §2.5]`. The 1a entry in `t_g150` is a 16x16, 10-step transient to `t = 2e-3` with `Nu_top = Nu_bot =
1.000761776406918` and `Vrms = 2.677656198451545`; the flow is still reflection symmetric there, and the gate asserts
`Nu_top = Nu_bot` to 1e-10, pinning the two wall normalisations against each other. ASPECT reaches `4.884409` on 1a at
refinement 6 and moves 0.2 percent between refinements 3 and 6 `[CONV §2]`. The open cases are a machine-time
question: about 500 steps for 1a at 1 to 2 s per step, about 5 000 for 1b, about 5e4 (a day) for 1c. The ASPECT
templates are in `aspect_runs/blankenbach_templates/`, rewritten from the image's own inputs because those crash in
debug postprocessing.

### 2.3 The blob at Ra = 1e8

The only case with the same initial condition, the same output cadence and a frame-by-frame comparison against a
second code `[REVIEW §4.10]`. A 1 by 1 box, free slip, `Ra = 1e8`, `H = 5` as constant heating, the same two-exponent
viscosity law, and the conductive base state plus a Gaussian anomaly, `(1 - z) + 2.5 z(1 - z) + 0.12 exp(-((x -
0.5)/0.12)^2 - ((z - 0.35)/0.10)^2) sin(pi z)`. Output every `4e-10` to `t = 9.6e-9`, 25 frames; ASPECT runs the
repository input file with only `End time` shortened, 8 ranks, 198 to 3816 points.

| classic-era 25-frame | ours | ASPECT | difference |
|---|---|---|---|
| last-frame `Tmax` | 1.2644 | 1.2531 | +0.9 % |
| relative L2 field difference, 256x256 | 2.87 % | | |

Both rows are the classic-era measurement and are superseded as the default baseline by the balanced re-run of the
same protocol, which gives frame-25 `Tmax = 1.264329738`, 7.7e-5 relative below the classic 1.264427445; the field
comparison has not been repeated under `:balanced`. The classic rows stay as the record of the classic arm (§2.3.1).

At equal `p`, time and leaf budget `[REVIEW §4.10]`:

| configuration | leaves | velocity nodes | 25-frame wall | last `Tmax` | field L2 |
|---|---|---|---|---|---|
| uniform L5 | 1024 | 16 641 | 199.5 s | 1.2532 | 2.55 % |
| adaptive, `max_leaves = 1024` | 1024 | 17 609 | 266.0 s | 1.2644 | 2.87 % |
| adaptive, `max_leaves = 1600` | 1600 | 27 045 | 244.6 s | 1.2644 | 2.87 % |

At equal leaf count the adaptive run is 33 percent slower and 0.32 points further from ASPECT; the larger budget adds
nodes without moving the field difference. All three rows are one session on 2026-11-05 and are quoted as a
same-session comparison; the uniform row was not re-run later, so the 33 percent is session-local. A later measurement
of the 1024-leaf adaptive protocol gives 119.4 s `[RHEO §15.1]`, a different budget that cannot be converted into the
1600-leaf row. These three rows are **classic-era** measurements: the Kelly protocol they use ran `amr_policy=:classic`
until the 2026-11-13 default flip. Under the balanced default the same 1024-leaf protocol ends at 898 leaves and
frame-25 `Tmax = 1.264329738` (§2.3.1); the classic row is not withdrawn, it records the classic arm. The regression
side is bitwise. Five FNV-1a hashes over `theta`, `ux`, `uz`, `pv` and the history
matrix, on a pristine `git archive` and on the tree `[BUG_LEDGER §1.3; AMR.md §5]`:

| case | `theta` | `ux` | `uz` | `pv` | history |
|---|---|---|---|---|---|
| 2b, 20 steps, EV + CFL | `d22fdab099ae84e0` | `2c60e7985fd9b8ec` | `def3383e3d7b6255` | `89de0ec00d6557ac` | `24dea03838128ba6` |
| blob frame 19, 1024 leaves, 2 steps | `45ac8acd099adadc` | `e41c4a173e290036` | `2371b0e7a8db9e6e` | `38bcc13372cfd60f` | `a8677df8737e63b5` |

The scalar diagnostics match to 17 printed digits (2b `Nu_top = 7.0299964227238032`, blob `5.9916941925454079`). Both
FNV rows are AMR-free. Their producer calls the driver with `amr_every = 0`, so `semquad_amr_passes!` is never called
and the `amr_policy` keyword is inert (`docs/BUG_LEDGER.md` §1.3; `AMR.md` §2.4). The
2026-11-13 default flip therefore left `d22fdab099ae84e0` and `45ac8acd099adadc` bitwise unchanged: the re-run under
`:balanced` returns the same `theta` hash and the same `Nu_top`, at 101 s against 100 s wall. That is why no
re-baseline was needed for these two rows.

The 25-frame trajectory was hashed per frame in the classic era (before the 2026-11-13 `amr_policy` default flip): all
25 SHA-256 digests of `theta` were identical between the pre- and post-optimisation trees, and the separate end-of-run
`theta` digest was `868a79d92a78bcfe89cfacd4774a0301c0eeb0a865284ab005155f6f458afe88`, while frame 25's
`Tmax = 1.264427445` reproduced the stored reference to `3.622e-10`. That is the **classic-era** record and it is not
withdrawn: with `BLOB_POLICY=classic`, `julia/scripts/run_blob_kelly_1024.jl` reproduces the digest bitwise on
2026-11-13 (126 steps, 1024 leaves, 129.8 s). Since the library default is now `:balanced`, the current record is the
balanced one:

| quantity | classic-era (`amr_policy=:classic`) | current default (`:balanced`) |
|---|---|---|
| end-of-run `theta` SHA-256 | `868a79d92a78bcfe89cfacd4774a0301c0eeb0a865284ab005155f6f458afe88` | `a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda` |
| end-of-run `(ux,uz)` SHA-256 | `c2a217d67c41973ba893d563d5ef200efc141545685e1d1653d7366dc36af491` | `1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51` |
| frame-25 `Tmax` | `1.264427445` | `1.264329738` |
| frame-25 leaves / `N` | 1024 / 17609 | 898 / 15361 |
| wall, 2026-11-13 session | 129.8 s | 134.5 s |

The classic digest `868a79d9...458afe88` is the end-of-run `theta` at step 126, the state after the last time step;
it is not frame 25. Frame 25 is written at `t = 1.0e-8`, one step earlier (step 125), and its classic `theta` digest
is `dd364f70...5aebfecb`, the field whose `Tmax` is 1.264427445. Older records paired the step-126 digest with the
frame-25 `Tmax`; `run_summary.csv` now prints `end_theta_sha256`, `end_u_sha256` and `Tmax25` on one row, and
`frame_sha256.csv` holds the per-frame rows, so the two objects are separable (`docs/BUG_LEDGER.md` §1.3, defect 17).

The per-frame `u` digests in `frame_sha256.csv` are hashes of the zero vector. The driver calls the frame callback
after an AMR cycle, where `ux`/`uz` have been zeroed for the next solve and are re-solved only at the end of the run
(`semquadconvection.jl:1718-1734`); the hash is over a zero vector whose length is the frame's `2N`, so it changes
only when `N` changes. Frames 23 to 25 of the balanced run, all at 898 leaves, share `110b5ff8...`; the end-of-run
`(ux,uz)` digest above is the meaningful velocity pin. The producer is now tracked as
`julia/scripts/run_blob_kelly_1024.jl`; it was `out_julia/agent/run_blob_kelly_1024.jl`, untracked, until 2026-11-13.
The gates guard the trajectory's preconditions, which is
what makes the hashes meaningful. `t_g173_leaf_budget` pins the hard budget (a refine-only cycle used to sit at 3016
leaves against `max_leaves = 1600`) and its quartet-merge and symmetry rules; `t_g186_balance_worklist` pins 2:1
balancing as a local worklist, the fix for a cycle that spent 31.5 to 31.9 s retrying one quartet 10 000 times.
`t_g56_h5_picard` pins the `H = 5`, `maxpic = 1` baseline; `t_g65`, `t_g66` and `t_g67` pin two-, five- and ten-step
CFL-respecting states at `Ra = 1e8`, and their headers state that they are stability and consistency gates rather than
statistical tests. `t_g202_newton_kernel_bitwise` pins a Newton kernel bitwise against the pre-fix worktree. The 2.87
percent field difference does not establish a convergence order: the window covers about a fifth of the event, `Tmax`
at the last frame differs by 0.9 percent, and the difference mixes `p = 4` against Q2, two boundary-layer resolutions
and two adaptive mesh distributions `[REVIEW §A3]`.

### 2.3.1 The 25 balanced per-frame digests

The balanced baseline of the same 25-frame protocol (run 2026-11-13, 134.5 s, 126 steps, `max_leaves = 1024`; the
budget exchange ends at 898 leaves from frame 19 on). `theta` SHA-256 is over the raw `Float64` bytes of the field the
frame callback receives, frame by frame; frame 25's field is one step before the end-of-run digest in §2.3.

| frame | `t` | `theta` SHA-256 | leaves | frame `Tmax` |
|---|---|---|---|---|
| 1 | 4.0e-10 | `42f363f31a754238bbaaa05337ac0b41c46e0fc455bfe214a62cf21874738380` | 268 | 1.318678647 |
| 2 | 8.0e-10 | `6f9ffbd72292b156d679a85d5be5f6b3c1264f704f2417c55edbbd7ff5f2f4b3` | 286 | 1.316914493 |
| 3 | 1.2e-9 | `66572116a1d6175a6dfbfdf9b8e561bb2035328897f3d6b3932e053056d64a19` | 304 | 1.313455366 |
| 4 | 1.6e-9 | `9d6fd660208cc94316e32aaf015b8ca00bfb0ae2679787ee09500d33a6888d47` | 322 | 1.308614442 |
| 5 | 2.0e-9 | `c256b27047f7ef801f59266e5c21843b27d30315dc9736953f0d61c8391ab574` | 352 | 1.303916437 |
| 6 | 2.4e-9 | `2a347835e2a91fe196522cbcad7d617c75949633a68ac8e7ede1659668c7eabe` | 376 | 1.298569699 |
| 7 | 2.8e-9 | `5c6a8d1a4181f57a62d4759e90be473bda230a26dd1bfe9ca190479f37320c05` | 400 | 1.291442003 |
| 8 | 3.2e-9 | `7b7d3e9a79e9b6ff5ca27d538949916aacb98b09cd63417cfec318dc3c5628b0` | 460 | 1.280015761 |
| 9 | 3.6e-9 | `cda6535da2c48ccab5d615a9429dd4b0d500f206d4969eab5c70c9e38cbb8505` | 502 | 1.274378717 |
| 10 | 4.0e-9 | `e1d7d45a93988ee6c6987fc4b7b5c2cf185c29ecac1a89b899bfc7bd2ade2fba` | 574 | 1.268971422 |
| 11 | 4.4e-9 | `484dbd02d95c7ff95db7eb6a6470537d67c287a0dee290ac82c1f42a39d7879b` | 616 | 1.267723062 |
| 12 | 4.8e-9 | `8f1247d13fbf0e5c0957825b0f81c902b8707b569f5b6831b67738ba05e4d866` | 658 | 1.267636892 |
| 13 | 5.2e-9 | `d9fedbe693707b29560e97db674f73008d0cfaba7dbc7c7947610f42100aa8ea` | 676 | 1.266783997 |
| 14 | 5.6e-9 | `1d120822e565fdaba66f090c96debb4f4f3749b291af6174e7f8fa39aad5992b` | 712 | 1.266438849 |
| 15 | 6.0e-9 | `11880cff42f4bac1299a0b830346c8e831e9c70c945db306d1f7440b20e771cf` | 736 | 1.266009403 |
| 16 | 6.4e-9 | `a56c20b9f76faaedad1a45db4b31efd32e577ba21be9796bc2c46d43e87a4f83` | 784 | 1.265939383 |
| 17 | 6.8e-9 | `fc5448a855f3ebcd52d8c0635ec3ca01347a660e1bf1dd7dc90d41d6360709f2` | 832 | 1.265992126 |
| 18 | 7.2e-9 | `4d309c982ddbb5385aab458005e24144b4f11b03f7d9c789b888d3efaa53a84d` | 868 | 1.265411661 |
| 19 | 7.6e-9 | `d081811b2cbfa7530139d715b8ab772655a423f0eaddec2220b5b1049f0edd96` | 898 | 1.264850274 |
| 20 | 8.0e-9 | `779733ffddcc08b3ca1d5058cab2ab513672fb71ac3fae9cb8aca25877b52339` | 898 | 1.264405827 |
| 21 | 8.4e-9 | `3eb69b8ee8a314b1bba12bc30be5f4696dc612840467d9a1cd234a40a26ca95e` | 898 | 1.264353469 |
| 22 | 8.8e-9 | `82ad62ca05d679652e98b2631aec79b0b4606ec09a2db1d1063e7ac3c45b4cee` | 898 | 1.264136381 |
| 23 | 9.2e-9 | `11956d1d79de08b3150d74e01469dbf91ef1984b89b08d410b121587ba74eeb8` | 898 | 1.264117179 |
| 24 | 9.6e-9 | `3c38d409c0b64330c0ac88eca148f6767a086dc58b9cd3b0634d8d36917fa2c8` | 898 | 1.264301836 |
| 25 | 1.0e-8 | `32eadb4028233645fea571868b887540e1c55696d4e07dfbf9a4352b9e90a6b3` | 898 | 1.264329738 |

The same record is written by the producer as `frame_sha256.csv` and `run_summary.csv` under its output directory.

### 2.4 TALA self-consistency

The truncated anelastic formulation is verified against analytic references, because no expressible ASPECT
configuration reproduces it `[RHEO §10.5]`. The code uses anomaly buoyancy `Ra rho(y) alpha(y) theta` with `div u = 0`
(TALA) or `div(rho u) = 0` (ALA), and subtracts the reference hydrostatic pressure analytically, so it is never
discretised `[RHEO §10.1]`. At `theta = 0` the assembled load is exactly zero and the solve returns `u = p = 0` with
residual `0.0`, independent of `h`. A resting *conductive* column is the interesting case, because the Q1 pressure
space cannot represent the antiderivative of its buoyancy anomaly. At 4, 16, 64 and 256 leaves: `||r||/||f|| =
2.795e-2, 7.292e-3, 1.905e-3, 4.893e-4` with orders 1.94, 1.94, 1.96; spurious `Vrms` 3.173e1, 1.855, 0.1126, 6.921e-3
with orders 4.10, 4.04, 4.02; spurious `max |u|` at 64 leaves 0.423 against `Ra |theta_c| ~ 3.8e5`. A run that needs a
truly resting column refines or accepts an `O(h^4)` drift `[RHEO §10.2]`. `T = Tad` is not steady (`lap(Tad) > 0`);
the conductive profile `T = 1 - z` is steady to `O(h^2)` (orders 1.92 and 2.49), with `drift/||theta_c|| = 9.276e-8,
2.451e-8, 4.355e-9` at 16, 64 and 256 leaves `[RHEO §10.3]`. The ALA versus TALA difference is a converged model
difference. Under the legacy rule `dVrms/Vrms = -7.557e-2` at **both** 64 and 256 leaves, identical to all printed
digits, so it is not a discretisation error; under the current isostress default it is `-6.879e-2` at 64 leaves. The
`Nu` difference is 1e-7 relative under both rules, so `Nu` does not discriminate mass forms here. Each field satisfies
its own constraint and only its own (`max |div u| = 3.8e-14` for TALA, `max |div(rho u)| = 8.1e-11` for ALA, wrong
residuals 3.87 and 1.68). The difference is concentrated at the bottom wall (14.5 percent of the local velocity),
decays upward by a factor 1e4, and vanishes at zero density contrast (`-2.0e-15` at `c = 0`, then `-2.0045e-2`,
`-3.8995e-2`, `-7.5568e-2` for `c = 0.25, 0.5, 1`). `t_g197_tala_consistency` pins all four checks `[RHEO §10.4]`.

## 3. The Nusselt number

### 3.1 Two functionals of one field

Take one wall element of height `h` with GLL nodes `theta_0` at the interior edge, `theta_1` at the midpoint, and
`theta_wall = 0` at the wall. Then

```
pointwise   p = (theta_0 - 4 theta_1 + 3 theta_wall) / h
steady CBF  q = (theta_0 - 8 theta_1) / (3 h)
q - p = -(2 / (3 h)) (theta_0 - 2 theta_1) = -(h/3) theta''_wall
```

`theta''_wall` is the second derivative at the wall, and the last equality follows from expanding a quadratic term
across the element. The first line is the standard one-sided three-point derivative, exact for a cubic. The second
comes from the wall row of the steady stiffness operator, whose top test function is `phi_top(y) = 2(y/h)^2 - y/h`;
against `theta = a u + b u^2 + c u^3`, with `u` the distance from the wall, its integral gives `-a - b h/3` `[AUDIT
§5]`. `p` is the exact derivative of the P2 interpolant, so its error against a smooth `theta` is the interpolation
error `-(h^2/12) theta'''`, second order in `h`; `q` differs from `p` by `-(h/3) theta''`, first order. At `h = 1/64`
the coefficients are 1.2e-4 and 5.2e-3, a factor 43. Exact checks at that `h`: `theta = u` gives `p = q =
-1.000000000000`; `theta = u^2` gives `p = 0.000000000000` and `q = -5.208333e-3 = -h/3`; `theta = u^3` gives `p =
+1.220703125e-4 = +h^2/2` and `q = 0.000000000000` `[AUDIT §5]`. On the four late production states the closed form
reproduces the measured gap:

| state | wall | measured `q - p` | closed form |
|---|---|---|---|
| E5202 | top | -0.037917353 | -0.037926511 |
| E7202 | top | -0.198680865 | -0.198681070 |
| E7602 | top | -0.219606919 | -0.219609295 |
| D7402 | top | -0.263641573 | -0.263645714 |
| E7202 | bottom | +0.000659513 | +0.000660024 |
| E7602 | bottom | +0.000387539 | +0.000388899 |
| D7402 | bottom | +0.003497160 | +0.003480483 |

The agreement is 1e-6 relative at the top wall and 1e-3 at the bottom. The E5202 bottom row is the exception at 3e-2
relative, because that state's advective wall term is -2.6e-3; the `K` part still matches `[AUDIT §5]`.

### 3.2 The measured spread

The four states are the E and D production runs: E5202 at `t_nd = 1.2292e-3`, E7202 at 1.9257e-3, E7602 at 2.0623e-3,
D7402 at 2.2069e-3, all with 9994 leaves and both walls at level 5 (`h = 1/64`) `[AUDIT §1, §3.1]`:

| state | `Nu_top` A | B | C | spread A to C | `Nu_bot` A | B | C | spread |
|---|---|---|---|---|---|---|---|---|
| E5202 | +6.074733 | +6.089900 | +6.090894 | 0.27 % | -2.822885 | -2.824167 | -2.824042 | 0.044 % |
| E7202 | +7.612832 | +7.692304 | +7.692973 | 1.05 % | -2.833834 | -2.834097 | -2.834077 | 0.009 % |
| E7602 | +8.160628 | +8.248471 | +8.249141 | 1.08 % | -2.836669 | -2.836824 | -2.836818 | 0.005 % |
| D7402 | +8.926261 | +9.031718 | +9.032216 | 1.19 % | -2.877316 | -2.878715 | -2.878635 | 0.046 % |

B and C agree to 1e-4, so the definitional gap is A against the reaction family, i.e. the `-(h/3) theta''_wall` term.
At the bottom wall the near-wall profile is almost linear, so the term is 0.005 to 0.046 percent of `Nu_bot`; at the
top wall it grows from 0.27 percent at `Nu_top = 6.07` to 1.19 percent at 8.93 as the layer thins. The thin-layer
explanation for that growth is rejected by measurement. The wall element is 12 to 16 times smaller than the top
boundary layer (`h/delta = 0.06 to 0.10`), only 6.1 to 8.5 percent of the layer's temperature drop falls inside it,
and the measured third derivative gives a pointwise truncation error of **3e-5 to 9e-5 relative** `[AUDIT §6]`:

| step | `t_nd` | `Nu_top` | `delta_top` | `T'''` | `-(h^2/12) T'''` | relative |
|---|---|---|---|---|---|---|
| 5201 | 1.229e-3 | +6.074 | 0.257 | +25.2 | -5.1e-4 | 8e-5 |
| 6001 | 1.446e-3 | +6.304 | 0.248 | +28.0 | -5.7e-4 | 9e-5 |
| 6601 | 1.673e-3 | +6.790 | 0.230 | +26.5 | -5.4e-4 | 8e-5 |
| 7201 | 1.925e-3 | +7.611 | 0.203 | +15.9 | -3.2e-4 | 4e-5 |
| 7601 | 2.062e-3 | +8.158 | 0.190 | +12.1 | -2.5e-4 | 3e-5 |

`delta_top` is the integral thickness `(T_bulk - T_wall)/|dT/dy|` and `T'''` the wall third derivative from the
x-averaged 256x256 frame; the frame-derived wall gradient reproduces `frame_meta` to 5e-5. What grows as the layer
thins is `theta''_wall`, and with it the CBF's error. The uniform-mesh sweep is the clean statement of an unresolved
wall `[RHEO §7.3.5]`:

| `h` | leaves | `Nu_bot` A | `Nu_bot` B | relative | `Nu_top` A | `Nu_top` B |
|---|---|---|---|---|---|---|
| 1/16 | 640 | -4.466022 | **+1.339976** | **433 %** | +5.999454 | +5.999670 |
| 1/32 | 2 560 | -4.094900 | -4.102413 | 0.183 % | +5.9999969 | +5.9999865 |
| 1/64 | 10 240 | -4.042907 | -4.043290 | 0.0095 % | +6.0000091 | +6.0000094 |

At 1/16 the bottom layer is thinner than the element (`~ Ra^{-1/3} = 0.02` against `h = 0.0625`), the pointwise value
is 10 percent off, and the reaction has the wrong sign. The top wall still agrees to 5.2e-4 at that `h`, because the
heated top layer is thick. An independent 32-step sweep on the same card reproduces the failure with slightly
different end states: -4.449346 against +1.469056, **402.9 percent** at 1/16, then -4.094900 against -4.102413 at 1/32
and -4.037315 against -4.036725 at 1/64 `[COMPARE §10.5]`. The sweeps differ at 1/16 and 1/64 because each level stops
at its own CFL-limited `t_nd` (2.77e-5, 2.10e-5, 1.40e-5) and `Nu_bot` still moves there; that attribution is
inference. The production wall shows the same ordering. At step 5801 the E-card top wall sits at level 3 or 4 (`h =
1/16 .. 1/32`, mean `h = 0.052`) while the bottom is at level 5. Measured: top +5.222130 pointwise against +6.095549
CBF, 14.3 percent; bottom -2.753533 against -2.819683, 2.40 percent `[CASE §7.2]`. Refining the wall strips to level 5
over 12 steps gives top +5.285596 against +5.508284, 4.0 percent: the pointwise value moved **1.2 percent** and the
CBF **9.6 percent**. On a *fixed* state the refinement leaves the pointwise value identical to 1e-7 (+5.629297 before
and after) while the CBF moves -9.7 percent, +6.536818 to +5.900182 `[CASE §14.4]`. The refined-wall gap is the CBF's
wall-resolution error.

### 3.3 Which number to quote

Quote the pointwise value. It is the exact derivative of the discrete solution, its wall quadrature is exact to
`0.0e0` (the GLL rule integrates the degree-2 wall derivative exactly `[AUDIT §7]`), its truncation error is second
order and measures 3e-5 to 9e-5 relative at the late states, and it is invariant under wall refinement at a fixed
state. Report `Nu_top` and `Nu_bot` with `h_wall` and `delta_T`, require at least two wall cells across the layer and
preferably four, and refine the wall strip when the gate fails `[CONV §4]`. Use the steady CBF as a cross-check on a
resolved wall: at or below one or two percent agreement quote the pointwise value, above it flag the wall. The CBF is
more *conservative*, because it closes the step budget by construction, and its wall functional is one order less
accurate: at `h = 1/64` its error is `h/3 = 5.2e-3` against the gradient's `h^2/2 = 1.2e-4` `[AUDIT §5]`. It is not a
cure for `delta_T < h`, since the 433 percent row is the reaction of a discrete equation inheriting the coarse wall
cell's error. Use the time-step CBF for the budget: its reaction is the boundary term of the step, so it closes the
discrete identity to the linear solver's floor (section 4). On a transient it is not a physical flux, and at step 1 it
is an artefact, reporting `Nu_bot = -28.4` against a wall gradient of -4.0 `[CONV §1; COMPARE §10.2]`; ASPECT's first
step shows -26 `[COMPARE §4]`. For cross-code work the floor is exact: the difference between families A and C is
`-(h/3) theta''_wall`, which grows with `Nu_top`, 0.27 percent at 6.07, 1.05 at 7.61, 1.08 at 8.16, 1.19 at 8.93, and
0.005 to 0.046 percent at the bottom wall `[AUDIT §3.1; COMPARE §13.1]`. In `Nu` units the bottom bound is at most
0.0015. A late-time top-wall comparison quotes that floor, uses ASPECT's own gradient family reconstructed from the
VTU, or compares C with C `[AUDIT §9.1]`.

## 4. The energy budget

### 4.1 The identity

With `phi` the tensor GLL basis, `rho = 1` for this card, `nu_ev` the elementwise entropy viscosity, `L_a` the latent
coefficient of transition `a`, `Gamma_T,a` its temperature derivative, and `lat_cp = sum_a L_a Gamma_T,a`, the step
operator is `A = ca M + K + C` with

```
M_ij = int (1 + lat_cp) phi_i phi_j                mass, with latent capacity
K_ij = int (1 + nu_ev) grad phi_i . grad phi_j     diffusion, with EV
C_ij = int (1 + lat_cp) phi_i (u . grad phi_j)     advection, skew
F_i  = int src phi_i                               source, base state included
```

and `rhs = cb M theta_n - cc M theta_{n-1} + F`, where `ca = a/dt`, `cb = b/dt`, `cc = c/dt`, and for `w = dt/dt_prev`
the BDF2 weights are `a = (2w+1)/(1+w)`, `b = 1+w`, `c = w^2/(1+w)` `[AUDIT §2.1]`. Since `sum_i phi_i = 1` and `sum_j
grad phi_j = 0`, row sums turn matrices into integrals: `1'(M v) = int (1+lat_cp) v`, `1'(K theta) = 0` to 1e-13,
`1'(C theta) = 0.5 int (1+lat_cp) u.grad theta`, and `-1'F = Ibase + latent_src`. Splitting the dofs into free,
Dirichlet-wall and hanging-constrained sets gives `1'r = Qc + hang + sum_free r`, with `Qc = sum_wall r`. The
constraint weights reproduce constants (`max |1 - sum_m w_m| = 2.2e-16`), so `hang` and `sum_free r` cancel to the
solver's backward error and the survivor is `eps`. Hence

```
Qc + eps = dH/dt + 0.5 int (1 + L Gamma_T) u.grad theta + Ibase + latent_src   (ID1)
```

`[AUDIT §2.2, §2.3]`. Read the symbols one at a time: `Qc` is the sum of the step residual over the Dirichlet wall
nodes, in flux units; `eps` is the part of the algebraic residual the constrained solve leaves outside the constraint
range, zero in exact arithmetic; `dH/dt` is the exact BDF2 mass term, the discrete rate of `int (1+lat_cp) theta`;
`0.5 int (1 + L Gamma_T) u.grad theta` is the volume advection of the skew form, with `u.grad theta` the advective
derivative and the `0.5` the skew factor; `Ibase = int u_z dTbar/dz` is the base-state advection, zero in exact
arithmetic for a divergence-free discrete velocity; and `latent_src = int sum_a L_a dGamma_a/dz u_z` is the latent
volume source. The PDE form would put the physical flux where `Qc` stands, and the two differ by exactly `eps`.

### 4.2 The historical closure column

The driver's `closure` column has been read as a budget miss for most of the project's life. With `A = 2.5` the box
area,

```
closure_area = Nut - Nub - (H - Ibase/A) + latent_dgdt/A
             = (Ibase + latent_dgdt - F)/A
```

where `Nut` and `Nub` are normalised wall fluxes, `H` the internal heating rate, `Ibase` the un-normalised base
integral, and `latent_dgdt = latent_src + latent_adv` with `latent_adv = int sum_a L_a dGamma_a/dT (u.grad theta)`
`[AUDIT §2.4; RHEO §7.3]`. Against the identity it has three defects: it omits `dH/dt`, it uses the PDE-level
`latent_adv` where the skew assembly has `0.5 int (1 + L Gamma_T) u.grad theta`, and it adds `Ibase` without dividing
by `A`, exact for the unit box every gate uses and wrong here by `Ibase (1 - 1/A)`, measured at +1.13 with `Ibase =
1.89` `[RHEO §7.3]`. The decomposition is exact. With `D := A * closure_area`,

```
D = (F_ts - F_pt) + (latent_adv - sum(C theta)) - dH/dt - eps          (DEC)
```

| item | E5202 | E7202 | E7602 | D7402 |
|---|---|---|---|---|
| `D` | -1.701752 | +1.188501 | +3.242800 | +5.188037 |
| `F_ts - F_pt` (wall mass plus wall functional) | -0.043293 | -0.200961 | -0.221654 | -0.268183 |
| `latent_adv - sum(C theta)` | -7.507760 | -0.234986 | -0.362173 | +0.953056 |
| `-dH/dt` | +5.860927 | +1.625146 | +3.819150 | +4.496633 |
| `-eps` | -0.011626 | -0.000698 | +0.007477 | +0.006530 |

`[AUDIT §4]`. `-dH/dt` supplies 87 to 137 percent of the residual at the three later states (136.7, 117.8, 86.7) and
-344 percent at E5202, where the latent and advection bookkeeping is larger. The wall-functional part is the CBF's own
`O(h)` error, and `eps` is the solver floor at 0.06 to 0.68 percent of `D`. The column is a transient term, and it
does not vanish for any `Nu` definition at a non-stationary state `[AUDIT §4, §9.2; CASE §7.2]`.

### 4.3 The measured residuals

| term | E5202 | E7202 | E7602 | D7402 |
|---|---|---|---|---|
| `dH/dt` | -5.860927 | -1.625146 | -3.819150 | -4.496633 |
| `sum(C theta)` | +7.308058 | +0.271102 | +1.756459 | +1.408371 |
| `Ibase` | +7.8919e-3 | +0.0233085 | +0.232710 | +0.183684 |
| `latent_src` | +1.246011 | +0.0124128 | -0.877439 | -1.866018 |
| `rhs` | +2.701034 | -1.318323 | -2.707420 | -4.770596 |
| `Qc` | +2.712660 | -1.317625 | -2.714897 | -4.777126 |
| `eps = Qc - rhs` | +1.1626e-2 | +6.982e-4 | -7.477e-3 | -6.530e-3 |
| free-row relative residual | 7.9e-3 | 3.5e-4 | 6.1e-3 | 8.3e-3 |

`eps` is 0.43, 0.05, 0.28 and 0.14 percent of `|rhs|`, the same size as the free-row residual, so it is the linear
solver's backward error on the ill-conditioned level-5 wall-strip operator. The hanging-node sums, `+74.0, +14.6,
+46.7, -23.4`, are not budget terms: `sum_free r` cancels them and the partition closes to 7.1e-15 .. 9.9e-14 `[AUDIT
§3.3, §7]`. The exact residual of the quoted family A pair is `O(h)`, and it is not the historical closure. Measured
as `R(F)/A` `[AUDIT §3.2]`:

| state | `R` A | `R` B | `R` C | `%H` A | `%H` C | `|R_C|/|rhs|` |
|---|---|---|---|---|---|---|
| E5202 | +5.492e-2 | +1.380e-2 | +1.163e-2 | 0.22 | 0.047 | 4.3e-3 |
| E7202 | +2.0166e-1 | +2.319e-3 | +6.982e-4 | 0.81 | 0.003 | 5.3e-4 |
| E7602 | +2.1418e-1 | -5.817e-3 | -7.477e-3 | 0.86 | 0.030 | 2.8e-3 |
| D7402 | +2.6165e-1 | -5.486e-3 | -6.530e-3 | 1.05 | 0.026 | 1.4e-3 |

Family C is 20 to 300 times closer to the identity than family A at every state, and family B is C plus the wall mass
term. Relative to `H`, the normalisation the case's acceptance criterion uses, family A misses 0.22 to 1.05 percent
while the reaction family misses at most 0.055 percent `[AUDIT §3.2; CASE §7.2]`. The one-step closure is the sharp
version. In a 1-D conductive column with one phase transition, the enthalpy rate from `semquad_heat_budget` matches
the consistent wall reaction to **1.6e-7** relative with latent heat and **1.9e-13** without, and the latent defect
halves with `dt` (the `O(dt)` capacity-placement error) `[RHEO §7.3.2]`. The multi-step budget against the physical
wall flux closes to 1.4e-2 at `h = 1/64` of that column, 5.1e-2 at `h = 1/16` and 4.9e-3 at `h = 1/256`, improving at
roughly second order in `h` and not with `dt`, identically with and without latent heat `[RHEO §7.3.1, §7.3.3]`. For
the realistic card the algebraic check was made at the B2 step-4604 state: `Qc = +2.582` against `dH/dt + 0.5 int(1 +
L Gamma_T) u.grad theta + Ibase + latent_src = +2.639`, a closure of 2.2 percent which is the hanging-node part of the
residual, with `latent_src = +11.3` inside the sum; dropping the latent terms gives -1.38 against +2.58, a miss of
3.96. The same state reports `closure = +0.430206` for `:per_phase` and -0.428968 for `:global`, with
`closure_nolatent = +0.233012`, `latent_dgdt/A = +0.19719`, `dH/dt/A = -1.306704` and `|Qc - flux|/A = 0.63` `[RHEO
§7.3]`. The budget closes, with the time-step CBF and `(ID1)`, to the linear solver's floor, and the historical column
is the enthalpy rate of a transient; neither statement depends on the `Nu` definition.

## 5. The cross-code comparison: mapping and runs

### 5.1 The scale set and the mapping

ASPECT is dimensional. The nondimensional card is fixed by four numbers (`Ra`, `H`, two composite prefactors) once a
scale set is chosen `[COMPARE §2.1]`:

| symbol | value | meaning |
|---|---|---|
| `d` | 2.9e6 m | box depth, the case's 2900 km |
| `kappa` | 1e-6 m^2/s | thermal diffusivity |
| `dT` | 1000 K | temperature drop |
| `rho0` | 3300 kg/m^3 | reference density |
| `g` | 9.81 m/s^2 | gravity |
| `alpha` | 3e-5 1/K | thermal expansivity |
| `cp` | 1000 J/(kg K) | specific heat |

With `Ra = rho0 g alpha dT d^3/(eta0 kappa) = 1e5`, `eta0 = 3300 * 9.81 * 3e-5 * 1000 * (2.9e6)^3/(1e5 * 1e-6) =
2.368635e23 Pa s`; with `H = H_r d^2/(cp kappa dT) = 10`, `H_r = 10 * 1000 * 1e-6 * 1000/(2.9e6)^2 = 1.189061e-12
W/kg`. The derived scales are `S = kappa/d^2 = 1.189061e-19 1/s` (strain rate), `P0 = rho0 g d = 9.388170e10 Pa`
(pressure), `u0 = kappa/d = 3.4483e-13 m/s` (velocity) and `t0 = d^2/kappa = 8.41e18 s` (time). A surface temperature
of `eta_T0 = 0.5` shifts the axis, `T_ASPECT = dT (T_nd + 0.5)`, so ASPECT's `exp(E/(R T_K))` with constant `E`
expresses an Arrhenius factor in `T_nd`; the walls become 500 K and 1500 K. The rheology mapping equates the branch
laws term by term `[COMPARE §2.2]`:

| ASPECT | value | from |
|---|---|---|
| `A_d` | 3.428173e-28 Pa^-1 m^3 s^-1 | `comp_Ad d_g^3/(2 eta0)` |
| `E_d` | 99773.5452 J/mol | `comp_Ed R dT` |
| `V_d` | 0 | `comp_Vd = 0` |
| `A_s` | 1.280082e-35 Pa^-3 s^-1 | `(comp_As/(2 eta0))^3 S^-2` |
| `E_s` | 141345.8557 J/mol | `comp_Es R dT` |
| `n`, `V_s` | 3, 0 | `comp_n`, `comp_Vs` |
| grain size | 1e-3 m | folded into `A_d` |
| bounds | 2.368635e19 / 2.368635e27 Pa s | `1e-4 eta0` / `1e4 eta0` |

with `R = 8.3144621 J/(mol K)`, ASPECT's `eta_diff = (1/2) A_d^-1 d_g^m exp((E_d + P V_d)/(R T))`, `eta_disl = (1/2)
A_s^(-1/n) edot^((1-n)/n) exp((E_s + P V_s)/(n R T))` and isostress combination, matching our two-branch composite law
and its isostress solve. The minimum strain rate is not mapped: ours is `1.19e-31 1/s`, ASPECT's default `1.4e-20 1/s`
is kept because the mapped value sits below its absolute Newton tolerance and would return `eta_min` at the
zero-velocity first evaluation `[COMPARE §2.6]`.

### 5.2 Verification of the mapping

`julia/scripts/validate_aspect_mapping.jl` evaluates `eta_composite_at` and an ASPECT-formula implementation of the
exact Newton solve at 180 points, `T in {0, 0.1, ..., 2.5}`, `edot_nd in 1e-4 .. 1e4`, `z in {0, 0.5, 0.9}` `[COMPARE
§2.2]`:

```
180 points: worst relative difference = 3.523e-06  (at the clamped point T=0.1, edot=1e-4)
interior rows: 8.8e-16, 0.0, 2.1e-16, 1.8e-16, 2.2e-16, 1.2e-16
```

The worst point is where the viscosity sits on `eta_max`; the unclamped law agrees to round-off. Independently, the
viscosity field of the real ASPECT run at `t = 0` matches the mapped diffusion branch to `1e-4` relative at interior
points, and both codes clamp the lid at `eta_max` with 0.94 percent of nodes clipped `[COMPARE §2.2, §4]`. The broader
check is the constitutive card of section 1 with the case's own parameters: 495 points, branch laws to 6.2e-15, full
isostress algorithm to 6.6e-15 .. 8.9e-15. The mapping's own history is a validation lesson: the published table had
`comp_Ad` and `comp_As` inverted, a factor 4.0e51 and 6.1e15, and the code never used the table so no run was
affected; `t_g200` now pins both factors `[RHEO §14.4g]`.

### 5.3 The runs

All runs use image `gassmoeller/aspect:latest` (2.5.0-pre, commit `340c0d018`, deal.II 9.4.0), the
`realistic_composite` plugin, `single Advection, single Stokes` (the analogue of our `maxpic = 1`), and no tuned
parameter `[COMPARE §5]`:

| run | window | steps | ranks | wall | cells |
|---|---|---|---|---|---|
| production | `t_nd = 0 .. 2.854e-5` (2.4e14 s) | 532 | 4 | 751 s | 1462, 12 832 at step 200, 10 816 at end |
| E, from scratch | `0 .. 1e-3` (8.41e15 s) | 4912 | 6 | 61.8 min | 1462 to 6739 |
| E-cont, E-long | `9.586e-4 .. 1.5e-3`, `.. 5e-3` | 1894, 9063 | 6 | 28.5 min, 2.11 h | 6742 to 4621 |
| verify100, matched 500-step window | `0 .. 3.2893e-5` (2.766e14 s) | 646 | 6 | 397 s | 1462 to 9553 |
| verify7000, long arm | `0 .. 3.2e-4` (2.6912e15 s) | 2362 | 6 | 10 618 s | 1462 to 6493 |

The verify rows are the ASPECT arms of §5.5. ASPECT needs 646 timesteps for the 500-step window because its CFL
controller takes a different path; the end time is the direct arm's `3.2893e-5`, and the MG frame is `3.2886e-5`.

The production run's `dt` falls from 1.9e12 s to 2.8e11 s at CFL about 0.5 on the finest cells, with 4 to 7
temperature and 26 to 37 Stokes iterations per step, 147 306 dofs at the end and no failure; `eta` stays inside its
bounds `[COMPARE §5]`. The E run advances at 0.90 s per step early and 1.09 to 1.13 s per step at 12 811 cells, with
`c = dt_nd * v_max = 1.95e-3` constant from step 50 `[COMPARE §9.3]`. The long continuation is deterministic:
`run_E_long` reproduces `run_E` row for row up to `t = 1e-3` (`max |dNu_bot| = 0.000e+00` over 4912 rows) and
`run_E_cont` over the overlap, and E6 reproduces its parent to 1e-6 in 1214 of 1215 steps `[COMPARE §17.1, §18.1]`.
The time conversion is one line with one trap: `t_nd = t kappa/d^2`, so one unit of `t_nd` is 8.41e18 s, `t_nd = 1e-3`
is 8.41e15 s (266 Myr), and the production window `2.854e-5` is 2.4e14 s. The number `8.41e12` is `d^2` alone and
would put `t_nd = 1e-3` 1000 times too early; the check is `2.854e-5 * 8.41e18 = 2.4e14 s` `[COMPARE §9.1]`. The
comparison script also uses `Vrms_nd = Vrms d/kappa`, `T_nd = T/dT - 0.5`, and `Nu = Q d/(Lx k dT)` with `k = 3.3 W/(m
K)` and `Lx = 2.5 d`, i.e. a normalisation `Lx k dT/d = 8250 W` `[COMPARE §6, §10.2]`.

### 5.4 The first window

At `t = 0` the codes start from the same state: `Vrms` 9594.1 against 9591.8 (0.02 percent), `Tmax` 1.8593 against
1.8578 (0.08 percent, the Q2 interpolation of the initial function), `Nu_top` 6.00000 against 5.99995, and `Nu_bot`
-4.004 (CBF) or -4.017 (VTU derivative) against -3.9995 `[COMPARE §4]`; the analytic values for this base state with
`H = 10` are exactly -4.000 and +6.000. Over the first window the thermal diagnostics agree and the dynamic ones do
not. At `t_nd = 2.854e-5`: `Nu_top` 6.00001 against 5.99908 (0.016 percent), `Tmax` 1.8589 against 1.8565 (0.13
percent), `Nu_bot` -5.600 against -4.868 (15 percent), `Vrms` 1.341e4 against 1.599e4 (16 percent) `[COMPARE §6.1]`.
The `Vrms` peak amplitudes agree to 2 percent, and the runs are not phase locked: a shift of `1e-6` in `t_nd` moves
`Vrms` by 10 to 20 percent, the leading contribution to that column `[COMPARE §6.2]`. Two level-5 controls remove the
model side: with latent heat off, `Vrms` at `t_nd = 1.19e-5` is 8.74e3 against 8.78e3 and `Nu_bot` -3.621 against
-3.623, and with both transitions off 8.80e3 and -3.622, so the controls agree with the full run to 0.5 percent and
0.06 percent; by `2.4e-5` their `Nu_bot` differs by 60 percent (-5.52 against -8.84), which is oscillation phase
`[COMPARE §6.3]`. One model difference was found and fixed: our latent operator carried one positive `L` for both
transitions where ASPECT's is per transition, so the two had opposite signs at the 410. The corrected per-transition
coefficient, -0.0594 at the 410 and +0.1306 at the 660, changes `Nu_top` by -6.7e-6 relative, `Nu_bot` by -1.0e-4 and
`Vrms` by -0.13 percent, so it was not the 5 to 16 percent dynamic difference `[COMPARE §2.5, §6.5]`.

### 5.5 The three-arm check at 500 steps

`julia/scripts/verify100_case.sh` launches one arm of a from-scratch comparison at `t = 0`: matrix-free MG, the
equilibrated sparse direct solve, and ASPECT 2.5. All three start from the card's own initial condition, not from a
checkpoint. The script writes every AMR knob explicitly (`CASE_AMR=balanced`, gate 1, `CASE_COARSEN=0.05`,
`CASE_THETATOL=1e-2`, `CASE_WALLMIN=5`, `CASE_SYMMETRY=none`, `CASE_MAXLEAVES=10000`), pins
`CASE_STOKES_SCALING=equilibrated`, names the output `out_julia/${RUN_ID}_<arm>` with `RUN_ID` defaulting to
`verify100`, and sets the frame stride through `CASE_FRAME`. The 500-step arms write 101 frames: steps 1, 6, ..., 496
and the final state at 500.

| arm | wall | s/step | end `t_nd` | `Nu_top` | `Nu_bot` | `Vrms` |
|---|---|---|---|---|---|---|
| matrix-free MG | 1398.9 s | 2.798 | 3.288644e-5 | 5.999914 | -7.915330 | 9605.85 |
| sparse direct | 2201.3 s | 4.403 | 3.289298e-5 | 5.999914 | -7.911161 | 9609.90 |
| ASPECT 2.5 | 397 s (646 steps) | n/a | 3.2893e-5 | 6.00000 | -8.32288 / -7.743888 | 8906.68 |

The two ASPECT `Nu_bot` values belong to two families: `-8.32288` is the gradient reconstructed from the final VTU,
`-7.743888` is the time-step CBF that `heat flux statistics` writes. The ASPECT values are from
`aspect_runs/realistic/output-verify100` (statistics and VTU at timestep 646); the Julia values are frame 100 of
`out_julia/verify100_mg/frames/frame_meta.csv` and `out_julia/verify100_direct/frames/frame_meta.csv`. The walls are
the logs' `[case] DONE` lines.

At frame 100 the two Julia arms agree to 3.7e-8 relative in `Nu_top`, 5.3e-4 in `Nu_bot` and 4.2e-4 in `Vrms`. The
256x256 temperature fields differ by rms 2.7e-4, which is 1.9e-3 of the anomaly's rms 0.142. The meshes part at step
136 (first `N` difference, 41 913 against 41 917) and first differ in leaf count at step 181 (9 961 against 9 958);
73 of the 101 frames differ, and the largest leaf gap is 222. The linear path moves the startup transient by at most
2e-3 relative.

Against ASPECT the same frame gives `Nu_top` 1.4e-5 relative, `Nu_bot` -5.2 percent in the gradient family
(-8.32288 against -7.915330) and `Vrms` -7.3 percent (8906.68 against 9605.85). ASPECT's own CBF value is 7.0 percent
above its gradient and 2.2 percent above ours; the gradient is the same-family comparison. The gap belongs to the
startup; the late-window gap of section 6 is a different measurement.

The window is the startup transient. 500 steps from `t = 0` end at `t_nd = 3.3e-5`; E reaches 8.56e-3, 260 times
later, and `Nu_top` climbs from 6.0 to 18.3 over that span `[CASE §3.1]`. At frame 100 all three arms still read
`Nu_top = 6.0`, and `Nu_bot` moves by 3.4e-2 in the last step. The check verifies the driver wiring, the two linear
paths and the frame writer. It does not measure a statistical state and does not replace the E/G comparison or the
`1e-3 .. 5e-3` window of section 6.

The long ASPECT arm reaches `t_nd = 3.2e-4` in 2362 timesteps at 6 ranks and 10 618 s
(`aspect_runs/realistic/output-verify7000/log.txt`), with `Nu_top` still 6.0. The wall clocks are not a code-speed
comparison. The MG arm runs 8 Julia threads; the direct arm is serial UMFPACK inside an 8-thread process, so its
4.403 s/step is one core's time. ASPECT runs 6 MPI ranks on its own mesh and solver. The 1.57x direct/MG ratio is a
wall-clock ratio on this card. `stokes_scaling=:equilibrated` engages on 365 of the direct arm's 500 steps and on none
of the MG arm's.

## 6. The bottom flux question

Over `t_nd = 1e-6 .. 1e-3`, sampled at 2000 matched times, ASPECT and our E card give `Nu_top` 6.0031 against 6.0012
(+0.03 percent), `Nu_bot` -3.6368 against -3.4274 (-6.11 percent), `Vrms` 1715.9 against 1684.3 (+1.87 percent) and
`Tmax` 1.8031 against 1.7916 (+0.64 percent) `[COMPARE §10.1]`. The top flux is the total heat loss and it agrees. The
bottom flux is a split of that total between the walls, and it does not.

### 6.1 The window is part of the answer

The subwindow means change sign inside the window: +8.38 percent over `[1e-6, 3e-6]`, +5.71 over `[3e-6, 1e-5]`, +0.99
over `[1e-5, 3e-5]`, -3.37 over `[3e-5, 1e-4]`, -0.51 over `[1e-4, 3e-4]` and **-9.68** over `[3e-4, 1e-3]`, which is
70 percent of the window's linear time. The cumulative mean moves from -2.45 percent at 10 percent of the window
through -3.57 at half to -6.11 at the end; the early positive side is the near-conductive startup, where the true
value is -4.000 `[COMPARE §10.1]`. Over longer windows the sign flips `[COMPARE §17.4]`:

| window | A CBF | A gradient | ours | diff CBF | diff gradient |
|---|---|---|---|---|---|
| `[1e-6, 1e-3]` | -3.6368 | -3.7237 | -3.4274 | -6.11 % | -8.44 % |
| `[1e-3, 1.5e-3]` | -2.9199 | -2.9515 | -2.8189 | -3.58 % | -4.70 % |
| `[1e-3, 2e-3]` | -2.8212 | -2.8435 | -2.8255 | +0.15 % | -0.64 % |
| `[1e-3, 3e-3]` | -2.7270 | -2.7402 | -2.8308 | +3.66 % | +3.20 % |
| `[1e-3, 4e-3]` | -2.6819 | -2.6909 | -2.8306 | +5.25 % | +4.94 % |
| `[2e-3, 4e-3]` | -2.6122 | -2.6146 | -2.8332 | +7.80 % | +7.71 % |
| `[1e-3, 5e-3]` | -2.6575 | -2.6640 | -2.8288 | **+6.05 %** | **+5.83 %** |
| `[3.5e-3, 5e-3]` | -2.5851 | -2.5846 | -2.8255 | **+8.51 %** | **+8.53 %** |

Over the same windows `Nu_top` (gradient family) agrees to 0.005 to 0.9 percent and `Tmax` to 0.3 to 1.1 percent, so
the whole difference sits in the bottom/top split. The long window covers 2.66 box overturns for ASPECT and 2.99 for
ours (`int Vrms dt`), 8911 steps and 2.1 h. The time structure is measured directly. The instantaneous difference is
-9.75 percent at `t_nd = 1e-3`, crosses zero at **1.42e-3**, reaches +5.67 percent at 2e-3 and +8.07 at 3e-3, and
saturates at +8.4 to +8.6 percent after 3.5e-3, a swing of 18 percentage points. Adjacent half-windows differ by 12.2
points (-3.58 to +8.59), the direct argument against quoting a fixed offset. The drift rate is not constant: about
+1.0 percent per 1e-4 across the zero crossing, +0.33 at 2.5e-3, +0.15 at 3e-3 and below 0.05 after 3.5e-3. Our own
`Nu_bot` over the long window is nearly constant (range 0.0143, 0.51 percent, over six windows) while our `Nu_top`
climbs from 6.13 to 12.3, so the split is what moves `[COMPARE §17.2, §17.5]`.

### 6.2 It is not realisation variance

Four members of the same ASPECT card, differing only in the initial-condition seed (amplitudes +0.2 and +0.4 percent,
and a 0.2 rad phase rotation, a 10 K probe) `[COMPARE §14.1, §14.3, §14.4]`:

| quantity | ensemble mean | sd | relative | min, max | ours | z |
|---|---|---|---|---|---|---|
| `Nu_bot`, window | -3.66543 | 0.06849 | **1.869 %** | -3.76795, -3.62615 | -3.4274 | **+3.48** |
| `Nu_bot`, late | -2.98836 | 7.985e-3 | **0.267 %** | -2.99521, -2.97850 | -2.7308 | **+32.3** |
| `Nu_top`, window | 6.002894 | 2.101e-4 | 0.0035 % | 6.002599, 6.003097 | 6.00123 | -7.90 |
| `Vrms`, window | 1716.84 | 30.745 | 1.791 % | 1673.72, 1740.26 | 1684.3 | -1.06 |
| `Tmax`, window | 1.80814 | 8.992e-3 | 0.497 % | 1.80305, 1.82161 | 1.7916 | -1.84 |

The three small members alone give a window spread of 0.00533 (0.147 percent), and the 10 K probe moves the window
mean 3.6 percent in the more negative direction (-3.7680), away from ours; no member is less negative than -3.6262, so
our value lies outside the ensemble while `Vrms` lies inside it at percentile 25. The honest budget line is a
window-mean realisation uncertainty of 2 to 4 percent including the startup transient, concentrated in `[1e-4, 3e-4]`,
with the late window at 0.27 percent: smaller than the difference, and the wrong way.

### 6.3 The same-state test

ASPECT was continued from its own checkpoint, its state at `t_nd = 9.586230e-4` was interpolated into our checkpoint
on our production tree (9988 leaves, 42 189 dofs, the same tree the E card had there), and both codes advanced from
that common state `[COMPARE §16.1, §16.2]`. The Q2 field is rebuilt exactly from the 3x3 patch vertices (a literal
bilinear read of the VTU would give a 10 to 25 percent wall-gradient error), with integer indexing instead of the
float32 coordinates `[COMPARE §16.4]`. On the common state `[COMPARE §16.3]`:

| quantity | statistics | VTU gradient | ours, imported | vs VTU |
|---|---|---|---|---|
| `Nu_bot` | -3.057280 | -3.075076 | -3.074986 | **+0.003 %** |
| `Nu_top` | +6.020498 | +5.990543 | +5.990543 | **0.000 %** |
| `Vrms` | 793.90 | 793.90 | 793.30 | -0.08 % |
| `Tmean` | 1.3185617 | 1.3185617 | 1.318491 | -0.005 % |

The 0.003 percent is the same state seen by two discretisations, so the operator-level agreement is essentially exact.
The 0.58 and 0.50 percent between ASPECT's own CBF and its own VTU gradient is the family floor, 200 times larger than
the interpolation error; three trees (9514, 9088 and 9988 leaves) give the same -3.074986 and 5.990543. From that
state the two solutions stay together for 0.84 box overturns `[COMPARE §16.5]`:

| quantity | at the common state | within 1.41e-4 | drift per 1e-4 | extrapolated to 1e-3 |
|---|---|---|---|---|
| `Nu_bot`, same family | -0.019 % | at most **0.59 %** | +0.275 | +2.75 % |
| `Nu_top`, same family | +0.009 % | at most 0.81 % | +0.57 | +5.7 % |
| `Nu_bot`, CBF family | -0.60 % | at most 0.60 % | -0.084 | -0.84 % |
| `Vrms` | -0.90 % | **14.8 %** | +1.67 | +16.7 % |
| `Nu_bot`, natural trajectory | +9.16 % | +6.2 .. +9.2 % | -2.1 | open |

Both runs go through the same two bursts (`Vrms` 794, 969, 654, 2483, 638) and their `Nu_bot` moves with them, so the
velocity field dephases and the bottom flux does not. The natural trajectory from `t = 0` over the same window is +6.2
to +9.2 percent away and parallel, so the 6 to 9 percent is accumulated history.
The extrapolations are the same order as the observed -6.11 and -8.44 percent and two to three times smaller; the rate
is larger in the burst phases, so they are lower bounds. The span is 0.84 overturns, not the full window: 541 steps at
4.56 s per step is 41 minutes, while a full `1e-3` window needs about 2500 steps and 3 hours `[COMPARE §16.6]`. The
reverse direction (our checkpoint into ASPECT) was not run, because it needs a plugin that reads our field format.

### 6.4 The candidate explanations, priced

| candidate | measurement | share of the difference |
|---|---|---|
| definition, family A against C | at most 0.0015 in `Nu_bot` units, against 0.2094 | **0.7 %** |
| ASPECT bottom-wall resolution, L5 to L6 | window -0.003 %, late +0.54 %, four late frames +0.78 %, gradient family -8.64 to -8.44 % | at most ~1 point |
| internal artificial viscosity, `beta = 0` | window +0.694 %, late +1.307 %, gradient 7.1 % | **7 to 15 %** |
| initial-condition realisation | late 0.27 %, window 1.9 %, wrong direction | **2 to 4 %, wrong sign** |
| phase | `Nu_top` cross-correlation >= 0.994 at every lag, `Nu_bot` at most 0.668; coupling slope +0.0277 against +0.0001 | excluded |

`[COMPARE §13.2, §13.4, §15.6, §14.4, §17.6]`. The E6 run behind the second row refines `y <= 0.05 d` to level 6, the
same `h = 1/64` as ours, at a cost of 1690 cells at `t = 0` against 1462 (+15.6 percent), 4930 steps against 4912
(+0.4 percent) and 81 minutes against 62 `[COMPARE §12.3, §12.4]`. Its whole-window zero cancels a late improvement
against a transient worsening (the `[1e-5, 1e-4]` subwindow goes 0.09 to 0.13 more negative); the profile moves 0.233
K at `y = 0.0156` where the codes differ by 3.797 K, 6 percent of the gap. The artificial-viscosity row needs its own
numbers, because the naive picture is wrong twice. CBF uses `kappa_eff = max(k, alpha_AV)`, and the wall cells are
exactly zero: 17 frames give `alpha_AV = 0` on the `y = 0` row, `kappa_eff/kappa = 1.000`, and at most 0.31 kappa
inside the layer `[COMPARE §12.2, §15.4]`. The interior does reach 5 to 36.5 kappa at plume transients, but the median
is 0.005 to 0.117 kappa, only 0.45 to 13.2 percent of nodes exceed kappa, and the box mean is 0.048 to 0.495 kappa.
The `beta = 0` run turns every branch off (`cR = 0` would not, because the first two steps return the upper bound) and
leaves `alpha_AV` identically zero in all 17 frames `[COMPARE §15.1, §15.2]`. In that run the bottom-layer profile
moves 21, 29 and 44 percent of the local temperature difference at the three sampled depths while the wall flux moves
7 to 15 percent, so the internal change does not transmit to the wall in proportion `[COMPARE §15.5]`. The premise
that ASPECT's artificial viscosity is much larger also fails: `cR = 0.11` and `beta = 0.052` are our `ev` and
`ev_cmax`, the cap is the same `c |u| h`, and our maxima (0.66 to 12.7 kappa) are the same order. What remains after
those five rows is 85 to 90 percent of the difference, and the profile measurements say what it is made of. Over the
four late frames the `Nu_bot` difference is -9.4 to -10.2 percent (mean -9.90). The temperature difference below `y =
0.05` is +0.0021 to +0.0065 (0.14 to 0.42 percent of `T_nd`), which maps to a viscosity ratio of only **1.007 to
1.028**, while a -9.9 percent flux difference would need `dln eta = +0.283`, a ratio of 1.33 `[COMPARE §11.1, §11.4]`.
The chain fails by 16.6 times in `dln eta` and 14 times in `Delta T`, and the peak-temperature version fails too: 0.6
percent at `Tmax` gives 2.5 percent in viscosity at `T_nd = 2.3`, or 5 percent at the wall, against the 33 percent
needed. The dislocation branch carries about 1e-4 of the strain rate inside the boundary layer (median `1 -
eta/eta_diff` between -7e-8 and +4e-6), so the viscosity there is a pure function of temperature and a 256-fold
strain-rate change moves the ratio by less than 1e-4 `[COMPARE §11.3]`. The transitions sit at `y = 0.78` and 0.87
with `Gamma = 1` to better than 1e-25 below `y = 0.05` `[COMPARE §11.5]`. The profiles show a vertical redistribution
of temperature. ASPECT is 0.25 to 0.6 percent warmer just above the wall, crosses over at `y = 0.05 .. 0.09`, and is 1
to 2.5 percent cooler above that (0.4 to 0.9 percent at mid-height), while the mean temperature agrees to 1e-5. The
wall slope difference is +9.5 percent. The boundary-layer thickness ratio is 1.016 to 1.104, while `eta^{1/3}` scaling
from the measured viscosity ratio predicts 1.002 to 1.009, so the thickness difference is 2 to 40 times what the
viscosity can explain `[COMPARE §11.2, §11.4]`. The top wall, thick and internally heated, agrees to 0.002 to 0.215
percent in `Nu_top` and 1.1e-4 to 2.8e-3 in the profile `[COMPARE §11.5]`.

### 6.5 The open item

ASPECT's bottom flux varies more than ours by 12 to 16 times. Over `[1e-3, 2.16e-3]` the standard deviation of
`Nu_bot` is **4.36 percent** for ASPECT and **0.37 percent** for us, a ratio of 11.7; the gradient family gives 4.59
against 0.37, 12.3. `Vrms` standard deviations differ by only 1.59 times in the same window, so the flux varies far
more than the flow `[COMPARE §18.3]`. The E6 run makes the spread larger: 4.36 to **5.82** percent on `[1e-3,
2.16e-3]` and 4.96 to **6.30** on `[1e-3, 3e-3]`, under all three definitions of the statistic, while the `Vrms`
spread falls from 50.1 to 39.8 percent. The variation is a slow drift, not a pulsation: 98 to 99 percent of the
variance is the trend at a `1e-4` smoothing width, and the residual is 0.07 percent at level 5, 0.09 at level 6 and
0.03 for us. Refinement improves the flow and leaves the flux alone, so the two are decoupled in the data `[COMPARE
§18.5]`. The decomposition is not done: the candidates left are the flow structure inside the bottom boundary layer
(cell number and phase, the near-wall `u_y`), the horizontal wall resolution (still 0.039 d in E6 against our 1/64 =
0.0156), and the two time-step and AMR histories `[COMPARE §13.5, §18.7]`; testing the first needs velocity frames,
which we do not store.

## 7. What the comparison establishes, and what it does not

It establishes the top flux. `Nu_top` agrees to **0.005 to 0.9 percent** over `t_nd = 1e-3 .. 5e-3`, and that is what
the box's total heat loss is made of: both codes walk the same internal-heating transient, `Nu_top` from about 6 to
about 18, on the same schedule, with a cross-correlation of at least 0.994 at every lag within `+/- 3e-4` `[COMPARE
§17.5, §17.6]`. The 0.9 percent end is partly ASPECT's own `Nu_top` sensitivity of 0.5 to 1.0 percent between level 5
and level 6 `[COMPARE §18.6]`. It establishes that the operators agree on a shared state: two discretisations, one
imported state, same-family wall fluxes, **0.003 percent** at the bottom and 0.000 percent at the top `[COMPARE
§16.3]`. Whatever separates the two codes is not what either operator does to a given temperature field. It
establishes the size of the bottom flux difference: **+/- 6 to 9 percent with a sign that depends on the window**,
from -6.11 (CBF) and -8.44 (gradient) over `[1e-6, 1e-3]` through +6.05/+5.83 over `[1e-3, 5e-3]` to +8.5 percent late
`[COMPARE §17.7]`; quote it with the window. It does not establish which code is right: there is no exact solution,
and the same-state experiment shows only that the two operators agree at a shared state. It does not decompose the
origin of the bottom flux difference. The location is pinned (the bottom boundary layer), the time structure is pinned
(a monotone, saturating redistribution crossing zero at `t_nd = 1.42e-3`), and five candidate mechanisms are excluded
by measurement: definition 0.7 percent, ASPECT wall resolution at most about 1 point, internal artificial viscosity 7
to 15 percent, realisation 0.27 to 1.9 percent with the wrong sign, and phase via the cross-correlation and the
coupling slope of +0.028 against +0.0001 `[COMPARE §17.6, §17.7]`. The remaining 85 to 90 percent is attributed to the
two solutions' bottom boundary layer structure, and that attribution names a location, with the mechanism open. Three
further limits constrain a future round: the drift rate is not constant, so extrapolating a short same-state span is a
lower bound; the same-state import ran in one direction only; and the `O(h^2)` error of the pointwise flux against the
continuum rests on the measured third derivative and on the functional being the exact derivative of the discrete
solution, with no mesh-convergence study of the *solution* `[AUDIT §10]`.

## 8. Gate index

The suite is `julia/test/runtests.jl` (all gates) or `julia/test/run_gates.jl <names>` (a subset). This table maps
each claim to the gate that keeps it true. Where no gate asserts a number, the row says so: a measured number and a
pinned number are different things.

| claim | gate | what the gate asserts |
|---|---|---|
| 2b viscosity law and its use in Stokes | `t_g12_blankenbach` | values at four corner states to 1e-14, contrast 16384*64, `eta_of_T` refusal |
| 2b and 1a transient diagnostics | `t_g150_regression_snapshot` | 20-step 2b `Nu_top`/`Nu_bot`/`Vrms`, 10-step 1a `Nu`/`Vrms`, Poisson L2, exact Stokes polynomial |
| second-order BDF2 energy step | `t_g26_dt_convergence` | perturbation decay against `exp(-2 pi^2 t)` |
| p-convergence, hanging constraints | `t_g151_p_convergence` | exponential error fall per order, same on a 2:1 tree, exact Qp/Qp-1 polynomial |
| manufactured Stokes and energy solutions | `t_g154_mms` | polynomial cases to round-off, trigonometric cases spectral in `p` |
| constitutive card | `t_g200_constitutive_card` | branch laws and mapping to 1e-12, one-mechanism limits, the combination-rule difference |
| isostress rule and tangents | `t_g201_isostress` | default rule against ASPECT's algorithm, tangents against finite differences |
| TALA self-consistency | `t_g197_tala_consistency` | anomaly equilibrium, resting-column orders, `T = Tad` not steady, ALA against TALA |
| heating budget and ALA/TALA sources | `t_g128_budget` | `Nu_top - Nu_bot = int(gamma_ad) + int(Phi)` with `Ra W = U` cancellation |
| time-step CBF on exact fields | `t_g58_cbf` | conductive `H = 5` gives bottom 1.5, top 3.5; `theta = 0.1 z` gives 1.6, 3.4 |
| steady CBF sign and normalisation | `t_g137_cbf` | conduction gives `Nu = 1` at `Lx = 1` and 2; short convective run agrees with the gradient to a few percent |
| EV field needed by CBF, checkpoint I/O | `t_g149_ev_cbf` | `ev_out` reproduces M, K, C, F bitwise; wall EV zero; CBF unchanged bitwise |
| wall-flux and budget columns | `t_g114_diagnostics` | conductive `H != 0`: `Nu_top = 1 + H/2`, `Nu_bot = 1 - H/2`, closure 0; history layout; averages |
| hard leaf budget under AMR | `t_g173_leaf_budget` | quartet merging, 2:1 maintenance, protection, symmetry pairing |
| 2:1 balance worklist and caches | `t_g186_balance_worklist` | local worklist equals the full-scan oracle; the outlier step cannot recur |
| `H = 5` extreme baseline | `t_g56_h5_picard` | `maxpic = 1` two-step values; the unrelaxed fixed point is rejected |
| `Ra = 1e8` stability and consistency | `t_g65`, `t_g66`, `t_g67` | 2-, 5- and 10-step CFL-respecting states; explicitly not statistical tests |
| Newton kernel bitwise | `t_g202_newton_kernel_bitwise` | kernel output against the pre-fix worktree |
| direct-solve ladder inert on healthy systems | `t_g207_direct_shift_ladder` | control flow on prescribed residuals; bitwise no-op when the residual is healthy |

Two things this table does not claim. The wall-functional closed forms of section 3.1 are pinned by a script,
`julia/scripts/nusselt_wall_functional_test.jl`, on exact polynomials `[AUDIT §5, §11]`; the steady 2b numbers and the
frame hashes are measured records from `julia/scripts/run_2b.jl` and `out_julia/agent/`, with `t_g150` pinning the
same trajectory's first 20 steps. The fast suite pins the operators and the short transients; the long comparisons are
measurements a gate would make too expensive to run on every change.
