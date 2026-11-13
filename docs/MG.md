# The matrix-free multigrid solver: ten stages, nine negatives, and the numbers

This document merges six measurement volumes of the matrix-free (MF) multigrid
campaign into one: the closing note, the stage-by-stage tuning record, the
component profile, the spectrum probe, the Schur-variant probe, and the cost page
for the two Cholesky solves. Nothing is re-measured here. Where a later session
corrected an earlier number, the later one is quoted and the earlier is named.

**Run artifacts.** The `out_julia/...` paths in this document name checkpoints and logs
from the development tree; `F` and `P1` were read from such checkpoints. They are cited so
that a measured number can be traced to the record that produced it, and are not part of
this repository.

`F` is 8 188 leaves, 69 388 saddle dofs (61 964 velocity plus 7 424 pressure), a
6-level hierarchy at p = 2, viscosity contrast 6.09e6, on a 2.5 x 1 free-slip box.
`P1` is 10 000 leaves, 86 544 saddle dofs (76 988 plus 9 556), 7 levels, contrast
3.32e6. Both come from realistic-case checkpoints,
`out_julia/_s7/case/pre_step6131.h5` and `out_julia/s340/ab_none/resume.h5`, read
through a temporary read-only copy. The machine is the i5-13490F WSL2 host, Julia
1.12.7, `BLAS` pinned to one thread. Unless a row says otherwise the tolerance is
1e-6, the FGMRES restart is 100, and "warm" means the in-tree production step with
the geometry cache and the pressure cache warm.

**The result.** A warm `F` solve at eight threads is **1.356 s and 71 outer
iterations**, with a true residual of 6.75e-7, `|du|/|u|` = 7.96e-7 and a
gauge-removed `|dp|/|p|` = 2.27e-7 against the equilibrated direct solve. That
direct solve needs 3.372 s warm in the same session, so the margin is **2.49x**.
At equal thread counts the margin is 1.96x on `F` and 1.50x on `P1`. Nine changes
shipped. Nine other directions were built or measured and rejected, each with the
number that rejected it.

One note before any seconds. Wall clock here drifts 4-6 % between sessions and
5-12 % across the wider record; the same 81-iteration warm `F` row has been
recorded at 1.324, 1.334, 1.356, 1.457, 1.489 and 1.505 s. Every ratio below is
two arms measured in one process on one machine. The iteration counts and the
residuals reproduce exactly and are the load-bearing numbers.

## 1. What the solver is

### 1.1 The saddle

After Dirichlet elimination, hanging-node elimination and the 2:1 constraint, a
Stokes solve is the reduced saddle system

```
[Kred  Bt] [u]   [fr]
[Bs    0 ] [p] = [gr]
```

`u` is the velocity unknown (61 964 entries on `F`), `p` the pressure (7 424).
`Kred = Cu' K Cu` is the reduced velocity stiffness, with `K` the leaf-local
viscosity-weighted stiffness operator and `Cu` the full-to-free constraint of the
level, mapping the unconstrained nodal velocity to the constrained one.
`Bs = -Cp' Bc Cu` is the reduced divergence, `Bt = Bs'` the reduced gradient, and
`fr`, `gr` the reduced right-hand sides. The system is symmetric indefinite,
which is the reason for the two-block treatment: the velocity block is SPD and
takes a multigrid cycle, and the pressure block is a Schur complement and takes
an approximation.

On the matrix-free path `K` is never assembled. The leaf kernel applies `K v` as
a tensor-product contraction over each leaf's element matrices with the viscosity
sampled at the quadrature points, wrapped in `Cu` and `Cu'` so the hanging
constraints cost two sparse products per level instead of a formed `C' K C`. A
production `linear=:mg` solve forms no sparse `K` and no `Kred` at all (`K =
nothing`, `Kred = nothing` in `solve_semquad_stokes_var`,
`julia/src/semquadstokes.jl:3193` and `:3311`), and `SEMSTOKES_K_ASSEMBLIES`
does not move across one.

### 1.2 The outer loop

The solver is flexible GMRES, `fgmres_restarted` from `semquad_mg_saddle_solve`
(`julia/src/semquad_mg.jl:2549`), restarted at 100, capped at 500, stopped on the
**true** reduced-saddle residual recomputed with one operator apply. Flexible
means it stores the preconditioned directions `Z[:,k]` and closes a cycle with
`x += Z y` instead of `x += V y`. The reason is measured in §5 and §9: on `F` the
computed pressure block fails additivity at 4.5e-7 on random vectors and 7.2e-2
on the direction the solve needs, so the preconditioner really is a different map
at different inputs in floating point, and `Z` is the stable way to form the
correction.

```julia
applyA(v):  ku = applyK(v[1:nf]); wA[1:nf] = ku + Bt*v[nf+1:end]
            wA[nf+1:end] = Bs*v[1:nf]
precond(v): zc = semquad_mg_apply(mg, v[1:nf])   # one V-cycle
            rp = v[nf+1:end] - Bs*zc             # the Schur residual
            wP[1:nf] = zc; wP[nf+1:end] = pressure_block(rp)
```

That is a block-triangular right preconditioner `P = [Khat Bt; 0 -Shat]`, with
`Khat` the approximate velocity inverse (the V-cycle) and `Shat` the approximate
Schur complement. The `Bs zc` product in `precond` is the row the profile calls
`Bs zc`, 4.1 % of an eight-thread iteration.

### 1.3 The V-cycle and the hierarchy

`semquad_mg_apply` runs `mg.cycles` V-cycles and that is 1 on the production
path, so one preconditioner call is one cycle. A cycle is recursive
(`_semquad_mg_vcycle!`, `julia/src/semquad_mg.jl:1437`): at the coarsest level
`x .= mg.cA \ b`, a dense 78 x 78 Cholesky solve; otherwise damped Jacobi
pre-smoothing (`pre[lev]` sweeps), the residual `b - A x`, restriction `P' r`, a
recursive cycle, prolongation `P ec`, `x .+= py`, and post-smoothing (`post[lev]`
sweeps). The damping is `omega = 2/3` for the uniform schedule and 0.55 for the
taper.

The apply count follows from the schedule: `F` does 25 matrix-free applies per
cycle (5 finest, 20 coarse) and 5 plus 5 transfers; `P1` does 42 (7 and 35) and 6
plus 6. Six levels and 25 applies is why the cycle cost is a schedule question.

`make_semquad_mg` builds the hierarchy bottom-up by merging every complete leaf
quartet one tree layer at a time, restoring the 2:1 balance after each merge
(`_mg_coarsen_layer!`, `_mg_hierarchy_trees`, `julia/src/semquad_mg.jl:269` and
`:328`). `mg_coarsen_factor` raises the bar in the merge loop

```julia
while done < layers || (factor > 1.0 && length(qc.leaves) * factor > n0)
    _mg_coarsen_layer!(qc) || break
end
```

which keeps merging until the step has coarsened by at least that factor, so
`factor = 1.5` deletes any step that would remove less than a third of the
leaves. The dof ladders are

```
F :  78, 318, 1 278, 4 930, 15 634,              61 964    (6 levels)
P1:  78, 318, 1 278, 5 118, 16 724, 40 776,      76 988    (7 levels)
```

Each level carries its constraint matrix `Cus[k]`, its prolongation
`Ps[k] = C_f' P_full C_c` (constraints on both sides of the transfer), its
free-row list, an exact constrained Jacobi diagonal, the viscosity sampled at its
quadrature points, and the leaf-local operator data. The coarsest operator is
materialised as a dense matrix from applies and factorised; that 78-dof solve is
0.03 ms of a cycle.

### 1.5 The w-BFBT pressure block

The pressure inverse is the three-factor weighted BFBT of Rudi, Stadler and
Ghattas (2017), built in `_semquad_build_pressure_precond`
(`julia/src/semquad_mg.jl:1838`) and applied in `_semquad_pressure_block!`
(`:2576`):

```
Shat^-1 = S1^-1 S2 S1^-1
S1      = Bs D^-1 Bt
S2      = Bs D^-1 K D^-1 Bt
D       = diag(Cu' M_u(sqrt(eta)) Cu)
```

`Bs` is the reduced divergence block, sparse, 7 424 x 61 964 on `F` (9 556 x
76 988 on `P1`). It turns a velocity field into a pressure residual, and its
transpose `Bt` is the gradient, which turns pressure into a velocity load. Both
depend on the tree, the boundary conditions and the element matrices, so both are
built once per tree and neither depends on viscosity.

`D` is a diagonal `nf x nf` weight, and `D^-1` in the formulas is elementwise
reciprocal. Its entries are the exact diagonal of the restricted velocity mass
matrix, `M_u(sqrt(eta))` being the leaf velocity mass with every leaf weighted by
the square root of its viscosity. `Cu' M_u Cu` is the mass matrix of the
constrained velocity space, and its diagonal is, for a free column `j` of `Cu`,

```
D_j = M_jj + 2 sum_h w_hj M_jh + sum_{h,h'} w_hj w_h'j M_hh'          (1)
```

where `h` and `h'` run over hanging rows naming `j` as a master, `w_hj` is the
constraint weight of `h` on `j`, and `M` is the weighted leaf mass. The first
term is the lumped diagonal, the other two are the hanging-node cross terms. All
are leaf-local, because a hanging row shares its leaf with its constraint's local
masters, so the diagonal is assembled per leaf and `M` is never formed. That
matters for cost (§3.9) and for the count (§2.5).

`S1 = Bs D^-1 Bt` is the mass-weighted Schur complement, sparse and symmetric,
198 490 nonzeros on `F`. It is symmetrised as `(S1 + S1')/2` and shifted by
`1e-14 * max|diag(S1)|` before the Cholesky, which exists only to make the
factorisation succeed. `F1 = chol(S1)` is the one factorisation on this path, and
each application of the block costs two solves with it; its factor has 649 814
nonzeros, a 3.27x fill, and is §7's subject.

`S2 = Bs D^-1 K D^-1 Bt` is the same sandwich with the stiffness `K` in the
middle. It is never formed (`Fp.S2 === nothing`); the apply is the sequence
`Bt r -> D^-1 -> K -> D^-1 -> Bs`, 2.10 ms per apply on `F` (1.60 ms of `S2` plus
0.50 ms of one `F1` solve), and it is the 12.0 % row of §4.1. Forming it as a
sparse matrix cost 0.42 s of build on the 1024-leaf tree with no per-apply gain.

The product is exact in one case, and that case is the intuition. If `D = K` then
`S1 = Bs K^-1 Bt = S` and `S2 = Bs K^-1 K K^-1 Bt = S`, where `S` is the true
Schur complement, so `Shat^-1 = S^-1 S S^-1 = S^-1`. The sandwich writes the
exact inverse as three cheap factors; with `D` the mass diagonal it approximates
it, and the quality is set by how well that diagonal stands in for `K` on the
pressure modes. The two `S1^-1` factors are large, `S2` is larger, and the
product works by cancelling them, which is where §6's knife edge comes from.

`pscale = 0.40` multiplies the pressure correction on output
(`wP[nf+1:end] = -pscale * rp2`) and relaxes the block. The sweep of §6 puts
`F`'s minimum at 0.20 (77 iterations) and `P1`'s at 0.30-0.40 (65), so one scalar
cannot be optimal for both and 0.40 is within 1 % of the best single value.

### 1.6 The four switches

| switch | default | what it selects |
|---|---|---|
| `mg_operator` | `:matrixfree` | the level operators. `:sparse` assembles `A[k]` per level, keeps `coarsen 1.0` and `D = :leaf`, and is the bitwise reference |
| `mg_vel_solver` | `:vcycle` | the velocity inverse. `:cg` runs CG on `Kred` preconditioned by the hierarchy, where a flexible outer method is mandatory |
| `mg_coarsen_factor` | `1.5` (`1.0` on `:sparse`) | the minimum coarsening ratio per step |
| `mg_smoothing` | `:auto` | the per-level Jacobi schedule and, with it, `omega` |

`:vcycle` is the default because it is 33 % faster per Stokes solve on the
1024-leaf blob tree: 40 outer with 40 V-cycles against 25 outer with 84 inner CG
iterations at 13 ms per cycle. `:sparse` is kept as the A/B arm of every bitwise
claim in §3, not as a competitor: `:matrixfree` builds its hierarchy 44-79x
faster and runs a 1.26-1.48x cheaper cycle at p = 4. Two settings behave like
defaults and are §2's business even though they are not switches of the method:
`mg_threads`, which the `:auto` selector sets to `min(8, Threads.nthreads())`,
and `mg_tol`, whose library default is 1e-10 and whose card value is 1e-8.

## 2. The defaults, and the measurement behind each

### 2.1 Coarsening factor 1.5

The uncoarsened `F` hierarchy has a near-duplicate step: level 7 holds 8 008
leaves and 60 900 dofs against the finest level's 8 188 and 61 964, a 1.7 % step,
and it cost 0.21 s to assemble and a third of every V-cycle. A factor of 1.5 is
the smallest value that deletes it, taking `F` to 6 levels while `P1` keeps 7
levels and its 40 776-dof level.

The alternative 2.0 cuts one level too deep: `P1` drops to 6 levels with a coarse
space too weak for that tree and the count triples from 61 to 177 (5.26 to 8.03 s
at one thread). On `F` 2.0 gives the same 6 levels and the same 81 iterations, so
it buys nothing there and costs a regression on `P1`. The shipped numbers are the
cold `F` solve 5.48 to 3.45 s at one thread (1.59x) and 3.74 to 2.47 s at eight
(1.51x), with `P1` at 1.02x and 1.10x. On uniform L5, uniform L6 and the
1024-leaf blob tree every step already coarsens by 2x or more, so 1.5 is a
bitwise no-op: 27, 28 and 39 iterations before and after.

### 2.2 Uniform 2+2, and 3+3 when the tree is top-heavy

`mg_smoothing = :auto` resolves to uniform 2+2 (`omega = 2/3`) for a 6-level
non-top-heavy p = 2 hierarchy, to uniform 3+3 for a top-heavy p = 2 hierarchy,
and to the taper at p >= 4. Top-heavy means the finest coarsening step keeps more
than half of the fine dofs (`_mg_smoothing_profile`,
`julia/src/semquad_mg.jl:199`).

The taper never won at p = 2. On `F`'s 6 levels uniform 2+2 gives 81 iterations
at 31.8 ms against 99 at 27.6 ms for the best taper, a tie on total time (3.36
against 3.33 s at one thread) and 2.473 against 2.498 s at eight. On `P1`'s 7
levels uniform 3+3 gives 65 at 34.5 ms against 97 at 25.0 ms, 3.38 against 3.53 s
at eight; at one thread the taper wins by about 2 % (5.01 against 5.12 s) and the
default follows the production thread count. At p = 4 the taper still wins
because one apply costs 4-16x more: on the blob tree 39 iterations at 21.1 ms
against uniform 3+3's 34 at 27.7 ms. The price is explicit: the count on `F` rose
from 70 to 81 when the taper was dropped, and §3.9 brought it back to 71.

### 2.3 The 32-lane kernel

`SEMQUAD_KBLK = 32` (`julia/src/semquadstokes.jl:1202`), a compile-time constant
because the width is what keeps LLVM's vectoriser in the loop. The batch was old:
`SEMQUAD_KBLK = 8` had shipped in `b97156c` and was pinned by G180, and at 8 lanes
it produced **zero** packed instructions, 0 `vaddpd`, 0 `vmulpd`, 0 `vsubpd`,
0 `vgather`, with 248 `vaddsd` and 464 `vmulsd`, because LLVM unrolled the
eight-lane loop completely before the vectoriser saw it. It measured 2 149.8 us
against 2 187.4 us for the scalar kernel, a 1.7 % win, and on the coarse levels
it was slower than scalar.

At 32 lanes the same kernel is 786-801 us, 2.72x, with 248 `vaddpd`, 496
`vmulpd`, 24 `vsubpd` and 48 `vgather`. On `F` at one thread the level-6 apply
went 2 436 to 1 056 us (2.3x), the V-cycle 18.10 to 8.86 ms (2.04x), the cold
solve 3.53 to 2.62 s (1.35x) and the iteration 2.66 to 1.58 s (1.68x). The
`@simd ivdep` annotations are load-bearing: removing them from the same code
gives 1 656 us and 12.63 ms.

### 2.4 The 20 480-dof serial threshold

`SEMQUAD_MG_SERIAL_DOFS = 20 480` (`julia/src/semquad_mg.jl:1129`) decides which
levels open a worker region; below it the apply, the Jacobi sweeps, the subtract
and the transfers stay on the calling task. A fork/join costs 6-9 us regardless
of the work, and at 318 dofs the serial subtract is 0.021 us and the Jacobi
update 0.155 us, so threading them is 40 to 330 times slower than running them.
The threshold keeps levels 1-5 serial on `F` and on `P1`, threads level 6 on `F`
and levels 6-7 on `P1`, and on `F` at eight threads it takes the level-1 and
level-2 applies from 34.15 and 49.20 us to 3.12 and 7.12 us, the V-cycle from
10.14 to 7.17 ms, and the iteration from 2.074 to 1.497 s (1.39x; `P1` 1.31x).

The value came from end-to-end solves and the sweep is flat either side of it:
V-cycle min/median at 6 144, 8 192, 15 634 and 20 480 dofs reads 7.60/8.58,
7.91/8.22, 7.75/8.10 and 7.32/8.13 ms, and 0 costs 7.84/9.80. The settling rows
are `F` at eight threads 2.215 s cold and 1.930 s warm at 20 480 against 2.261
and 1.988 at 8 192, and `P1` 2.960 and 2.643 against 2.938 and 2.794.

### 2.5 The exact leaf-local BFBT diagonal

`mg_pressure_bfbt_d = :auto` resolves to `:exact` on `mg_operator=:matrixfree`
and to `:leaf` on `:sparse`. The shipped weight is
`D = diag(Cu' M_u(sqrt(eta)) Cu)`, built per leaf by `_semquad_bfbt_diag_exact`
(`julia/src/semquad_mg.jl:2261`), replacing the restricted diagonal
`Cu' diag(M_u)`. The result is the count: **81 to 71 on `F`** (-12.3 %), **65 to
65 on `P1`**. `F`'s warm solve fell from 1.507-1.548 s to 1.324-1.334 s at eight
threads across two sessions (-11.5 to -14.5 %) and from 2.070 to 1.791 s at one,
and accuracy improved with it: `|du|/|u|` 1.694e-6 to 7.956e-7, with the
gauge-removed pressure at 2.267e-7 against 2.165e-7.

The mechanism is the cross terms of equation (1), not the 16 non-positive entries
of the lumped restriction. Re-clipping those 16 dofs gives 80 iterations at a
1e-6 clip and 155 at 1e-4, a non-monotone function of 16 values, while the
relative L2 change in `D` is 1.54 % on `F` against 0.086 % on `P1`, which is
exactly where the gain appears. The exact diagonal equals the sparse reference to
round-off (1.9e-16, 2.3e-16, a different summation order) and is bitwise
identical at one and eight threads, because the leaves are reduced in 16 fixed
contiguous blocks.

It also builds faster than the weight it replaces at eight threads: 0.028-0.034 s
against 0.088-0.105 s on `F`, 0.032-0.034 s against 0.123 s on `P1`. At one
thread it is a few milliseconds slower (0.090 against 0.085 s), because 87 % of
either build is the eta sampling: 8 188 x 36 = 294 768 `wfun` calls on `F`, 87 ms
with the production `sqrt(eta)` closure and 1.5 ms with `w = 1`.

### 2.6 `mg_tol = 1e-8` in the card

The library default is `mg_tol = 1e-10`. The realistic card passes 1e-8
(`MG_TOL` in `julia/scripts/case_realistic_2d.jl`) and its driver accepts a Stokes
solve at `stokes_tol = 1e-3`, so 1e-8 is five orders of margin below the gate and
1e-10 is seven. The card's choice is a measured 12-20 % of warm time:

| state | `mg_tol` | outer it | true res | warm s | `|du|/|u|` | gauge `|dp|/|p|` |
|---|---|---|---|---|---|---|
| `F` | 1e-10 | 92 | 7.85e-11 | 1.608 | - | - |
| `F` | **1e-8** | **81** | **8.61e-9** | **1.414** | **1.98e-8** | **3.26e-9** |
| `F` | 1e-6 | 71 | 6.75e-7 | 1.260 | 7.95e-7 | 2.27e-7 |
| `P1` | 1e-10 | 108 | 8.73e-11 | 2.681 | - | - |
| `P1` | **1e-8** | **91** | **9.44e-9** | **2.208** | **1.83e-8** | **7.86e-9** |
| `P1` | 1e-6 | 65 | 6.62e-7 | 1.703 | 9.84e-7 | 1.28e-6 |

The 1e-6 row is faster still and was rejected on accuracy: it moves `P1`'s
gauge-removed pressure by 1.28e-6, above the documented MG class of 8.8e-7, and
1e-5 fails the velocity class on all three states. The 1e-8 row moves velocity by
at most 2.7e-8 and pressure by at most 7.9e-9, 78x and 111x inside the class. The
library default stays at 1e-10 because the library does not know the driver's
gate; the card does. G228 pins the declarations and their forwarding.

## 3. The ten stages

Nine changes shipped, plus a one-line fix that came with the ninth, which is why
there are ten subsections. Seven are **bitwise**: they move scheduling, storage
or allocation and leave every dof with the same sequence of IEEE operations. The
two that are not, stages 1 and 9, change the preconditioner and therefore the
outer count, and both were gated on accuracy against the equilibrated direct
solve. That gating is the campaign's most reusable output, so it is three rules
here.

**Same-session A/B, against a built baseline.** Absolute seconds are session
currency, so a speed claim is a ratio of two arms in one process on one machine.
The baseline arm is built, not remembered: `git archive HEAD` unpacks the
pre-change tree, the same script and checkpoint run in both, and the raw outputs
are compared with `cmp`. The scripts alternate paths round by round for the same
reason.

**Bitwise comparison of the raw IEEE bits.** Anything claimed inert is compared
as `reinterpret(UInt64, x)`, because `==` would accept a `-0.0` for a `+0.0`.
`probe_mg_bitwise.jl` writes every `Float64` of a solve to a file and the
comparison is `cmp`. Each stage carries a gate: G219 (27 assertions), G220 (37),
G221 (57), G222 (18), G223 (39 and 71), G224 (93 and 180), G225 (129 and 129),
G226 (28), G227 (15).

**Accuracy against the equilibrated direct solve, never against `relres`.** A
converged residual does not certify a preconditioner change; §5.5 has a case
where `relres` 7.2e-7 came with a 1.0e-4 pressure error. Every preconditioner
change is gated on `|du|/|u|` and the gauge-removed `|dp|/|p|` against
`linear=:direct, stokes_scaling=:equilibrated` in the same process.

### 3.1 Stage 1: the coarsening factor and the smoothing schedule

**What was slow.** A hierarchy step that coarsens the dof count by 1.7 % and
costs a third of every V-cycle, and a taper that spent its sweeps where they were
cheap but not where they paid.

**What the measurement said.** Factor 1.5 removing that step was worth 1.59x on
`F` at one thread, the taper was a tie at p = 2, and factor 2.0 cost `P1` 61 to
177 iterations.

**What changed.** `mg_coarsen_factor = 1.5` on the matrix-free path and
`mg_smoothing = :auto`: 2+2 for a 6-level non-top-heavy p = 2 hierarchy, 3+3 for
a top-heavy p = 2 one, the taper at p >= 4.

**The number.** Cold cache, best of 3: `F` 5.48 to 3.45 s at one thread (1.59x)
and 3.74 to 2.47 s at eight (1.51x); `P1` 5.26 to 5.17 s (1.02x) and 3.72 to
3.38 s (1.10x). The gain is mostly setup, 0.88 s on `F` and 0.47 s on `P1`, and
the `F` cycle also got cheaper, 53.88 to 32.35 ms. The price is one extra outer
iteration per solve: 70 to 81 on `F`, back to 71 at stage 9.

### 3.2 Stage 2: the duplicated finest `Cu`

**What was slow.** `make_semquad_mg` assembled every level, the finest included,
with `eta = 1`, and read only `asm.Cu` from the result. `Cu` is a function of the
tree, the boundary conditions and the affine Dirichlet data; eta does not enter
it, so the finest row was duplicate work on a matrix the caller had built.

**What the measurement said.** 0.766 s of hierarchy build on `F` at one thread,
of which the finest level was 0.17 s in the anatomy pass and 0.41 s in a single
pass.

**What changed.** The caller's `Cu` is passed in as `fine_Cu`, and
`_semquad_mg_geometry` uses it instead of re-assembling.

**The number.** One thread, cold cache, best of 2: the build fell from 0.766 to
0.256 s, a 3.0x drop of 0.510 s. The A/B is bitwise on one process: `mg.geo.Cus`,
`mg.geo.Ps`, `mg.geo.free_rows`, `mg.D`, `mg.mfetas`, one `semquad_mg_apply` and
the returned FGMRES solution are all `==`, at 81 iterations and 8.619e-7 both
sides.

### 3.3 Stage 3: the 32-lane apply

**What was slow.** The leaf kernel ran eight leaves wide, and at that width LLVM
unrolled the lane loop completely before the vectoriser saw it, leaving eight
independent scalar dependency chains.

**What the measurement said.** The compiled kernel at 8 lanes has zero packed
arithmetic, 0 `vaddpd`, 0 `vmulpd`, 0 `vsubpd`, 0 `vgather`, with 248 `vaddsd`
and 464 `vmulsd`. It measured 2 149.8 us against 2 187.4 us for the scalar
kernel, a 1.7 % win, and it was slower than scalar at levels 1 and 4. At 16 lanes
it is 835 us, at 32 lanes 786-801 us with 248 `vaddpd`, 496 `vmulpd`, 24
`vsubpd` and 48 `vgather`, and at 64 lanes 890 us.

**What changed.** `SEMQUAD_KBLK = 32` with `@simd ivdep` on the lane loops. The
scatter and the padding lanes stay scalar: a scatter has loop-carried dependences
through shared nodes, and vectorising it would reorder the accumulation.

**The number.** On `F` at one thread: level-6 apply 2 436 to 1 056 us (2.3x),
V-cycle 18.10 to 8.86 ms (2.04x), cold solve 3.53 to 2.62 s (1.35x), iteration
2.66 to 1.58 s (1.68x). Removing the `@simd ivdep` from the same code gives
1 656 us and 12.63 ms.

### 3.4 Stage 4: the zero-allocation kernel

**What was slow.** The 32-lane kernel built five lane vectors inside itself on
every call, `shx`, `shy`, `swf`, the unit scale and `kofs`, with `zeros` and
`ones` of length `BLK`: 640 B per call, invisible per call and not per solve.

**What the measurement said.** The threaded level-6 apply allocated 25 456 B per
call (32 chunks of 640 B plus task objects) and an eight-thread V-cycle 1.83 MB.
At 65 applies per cycle that is 1.3 MB per cycle, 90 MB per 70-iteration solve.

**What changed.** The five fields moved into `SemQuadKScratch`
(`julia/src/semquadstokes.jl:892`), one instance per worker on the threaded path.

**The number.** The serial apply allocates 0 B at every level, the threaded apply
4 976 B (task objects only), and the eight-thread `F` V-cycle 1.05 MB. No
arithmetic changed, so the stage is bitwise by construction. G222 pins 0 B serial
and below 16 kB threaded; its threaded assertion needed a warm-up call, because
the first measured call now pays 308 080 B of one-time compilation.

### 3.5 Stage 5: the 20 480-dof serial threshold

**What was slow.** The eight-thread regression on the small levels was
thread-launch overhead, 6-9 us per fork and join regardless of the work.

**What the measurement said.** At 318 dofs the serial subtract is 0.021 us and
the Jacobi update 0.155 us, so a fork/join costs 40 to 330 times the work it
distributes. The level-1 and level-2 applies were 34.15 and 49.20 us at eight
threads against 4.56 and 11.70 us at one, 7.5x and 4.2x slower with more threads.

**What changed.** `SEMQUAD_MG_SERIAL_DOFS = 20 480`: below that size the apply,
the Jacobi sweeps, the subtract and the transfers stay on the calling task.

**The number.** On `F` at eight threads the level-1 and level-2 applies fell to
3.12 and 7.12 us, the V-cycle from 10.14 to 7.17 ms, and the iteration from
2.074 to 1.497 s (1.39x; `P1` 2.575 to 1.973 s, 1.31x).

### 3.6 Stage 6: the region, and the hundred microseconds

**What was slow.** Every parallel op created its own task set, 6-9 us even for an
empty body, and the fork accounting said that was 1.2 % of a V-cycle. So the
expected gain was small and the measured gain was not.

**What the measurement said.** Timing every op in place puts one threaded
`r = b - y` at 61 964 dofs at **about 101 us inside the cycle** (506 us over 5
calls) against 19.6 us back-to-back and 21-23 us inside a region. An empty
`_semquad_td_run!` costs 5.4-6.3 us at eight workers, so the 90 us difference is
not task creation. A task set created, run for 10-300 us and destroyed leaves the
other threads in the scheduler's idle state, and on this hybrid CPU in an OS idle
state; the next call pays the wake-up and the barrier waits for the slowest. The
back-to-back microbenchmark hides it because the cores never go idle.

**What changed.** `SemQuadParRegion`: a persistent worker set of `nt` workers,
one of them the calling task, in which an op is published as a phase (a job, a
chunk count, an atomic sequence number, a barrier). A threaded level's
pre-smoothing block, its Jacobi sweeps, the residual and the restriction run in
one fork/join, and its prolongation, `x .+= py` and post-smoothing in a second.
The workers spin on the phase counter while the calling task runs the serial `Cu`
products and the scatter.

**The number.** Forks per V-cycle 15 to 2 on `F` and 43 to 4 on `P1`; worker
tasks 120 to 14 and 344 to 28. The eight-thread V-cycle went 6.94 to 5.42 ms
(1.28x) and 15.48 to 10.98 ms (1.41x), the iteration 1.455 to 1.339 s (1.09x)
and 1.890 to 1.614 s (1.17x), the warm step 1.902 to 1.835 s (1.04x) and 2.569 to
2.406 s (1.07x). The residual fork/join floor is 35-39 us on `F` and 70-76 us on
`P1`, 0.6-0.7 % of the cycle.

### 3.7 Stage 7: the transfer and the two `Cu` products as phases

**What was slow.** Two costs sat on the calling task inside a live region. The
restriction into the next coarser level is below the serial threshold by design,
so it ran serially while seven workers spun; the threshold decides which *level*
opens a region, and a transfer inside an open region has no such choice to make.
And the two sparse `Cu` products are row-independent work on the calling task,
about 215 us of a finest apply.

**What the measurement said.** The serial restriction is slower inside a live
region than isolated, 1.8-2.1x, because the spinning workers contend with the
calling task: 185.5 us isolated, 335.6 us in a live region, for `F`'s level-6 to
level-5 transfer at 15 634 dofs.

**What changed.** Both became phases. `SEMQUAD_MG_REGION_TRANSFER` skips the dof
clamp when a region is given, and `SEMQUAD_MG_REGION_CU` splits each `Cu` product
by output entry. The row split needs the transpose, so `SemQuadMGGeom` carries
`Cuts`, one materialised transpose per level (a pure re-indexing: under 1 ms per
level, 2.4 MB in total on `F`), which keeps the `mul!` accumulation order.

**The number.** At eight threads the V-cycle went 5.78 to 4.39 ms on `F` (1.32x,
1.29-1.37x over three sessions) and 11.64 to 8.61 ms on `P1` (1.35x); the stage-6
arm re-measured at 5.78 ms here against 5.42 ms in its own session, which is
drift. The warm step 1.824 to 1.698 s (1.07x) and 2.347 to 2.096 s (1.12x), the
iteration 1.311 to 1.172 s (1.12x) and 1.644 to 1.366 s (1.20x). In place the
restriction column falls 431 to 178 us on `F`. The `Cu` split is the larger half:
`Cu * v` 102.2 to 74.9 us and `Cu' * y` 113.0 to 58.4 us at `F` level 6, and a
finest apply 579.0 to 346.1 us on `F`, 821.8 to 417.1 us on `P1`. Phases go 15 to
26 and 43 to 72, at 5.5-5.7 us of empty-body cost each.

### 3.8 Stage 8: the eta lane table

**What was slow.** The apply read its viscosity one lane at a time,
`etav[kofs[b] + kq]`, so LLVM emitted a `vgather` per quadrature point and lane
vector. Gathers are slow and non-pipelined on this AVX2 machine.

**What the measurement said.** The p = 2 kernel at 32 lanes had 48 `vgather`s per
block in the revision of §2.3 (24 after the unrolling changed in stages 6 and 7),
and a gather moves the same bytes as a contiguous load through a slower
instruction.

**What changed.** Each level's eta field is re-laid into a block-major lane table
once, at operator construction:

```
lane[(bi-1)*nq2*BLK + (kq-1)*BLK + b] = scalar[((bi-1)*BLK + b - 1)*nq2 + kq]
```

`bi` indexes the 32-lane block, `b` the lane inside it, `kq` the quadrature
point, `nq2 = nq^2` the points per leaf, `BLK = 32` the lane width, and `scalar`
the historical leaf-major field: for each block and quadrature point the eta of
the 32 leaves is contiguous. The padded lanes of the last block repeat its final
leaf, which the kernel already read with weight 0.0, so neither value nor
arithmetic changes. The threaded chunk boundaries moved onto the block grid as a
consequence, which is bitwise safe because a boundary only decides which worker
computes a leaf.

**The number.** The `vgather` count in the compiled p = 2 kernel goes 24 to 0 and
every other instruction count is unchanged (4 020 asm lines to 3 902). The leaf
kernel is 1.158x faster on `F` level 6 (800.57 to 691.33 us) and 1.162x on `P1`
level 7 (986.63 to 849.29 us). The eight-thread V-cycle is 1.075x on `F` (4 434
to 4 123 us) and 1.063x on `P1` (8 223 to 7 735 us); at one thread 1.083x and
1.101x. The table costs 116 us for all six `F` levels and 240 us for `P1`'s
seven, holds 1.05 and 1.28 MB, and is 0.03 % of the setup. The gain beats the
2-4 % projected for it because the coarse *serial* applies read the same field
through the same kernel: they are 41 % of `F`'s cycle.

### 3.9 Stage 9: the exact leaf-local BFBT diagonal

**What was slow.** The weight `D` was the restricted diagonal of the velocity
mass, `Cu' diag(M_u)`, a lumped approximation with no hanging-node cross terms.
It also had 16 non-positive entries on `F` out of 61 964, which the shipped
`1e-12 * max|d|` clip turned into a 1e12 contrast in `D` and hence into `D^-1`
entries up to 3.6e13.

**What the measurement said.** The exact diagonal, equation (1), changes `D` by
1.54 % in the L2 norm on `F` and 0.086 % on `P1`. That 1.5 % is worth 10 outer
iterations, and re-clipping the 16 outliers does not reproduce it: 1e-6 gives 80
iterations and 1e-4 gives 155.

**What changed.** `_semquad_bfbt_diag_exact` and `_semquad_bfbt_diag(...;
how=:exact)` build the diagonal per leaf. The implementation classifies each
leaf's local rows from `Cu` itself, computes the leaf mass diagonal plus a small
dense mass sub-block over the leaf's hanging rows and the local masters they
need, and scatter-adds the quadratic forms; `M` and `C' M C` are never formed.
The pressure cache key carries the weight variant, so switching weights inside
one process cannot reuse the other's factors.

**The number.** 81 to 71 iterations on `F` (-12.3 %), 65 to 65 on `P1`; `F`'s
warm solve 1.507-1.548 to 1.324-1.334 s at eight threads and 2.070 to 1.791 s at
one; `|du|/|u|` 1.694e-6 to 7.956e-7. The build is 0.028-0.034 s at eight
threads against 0.088-0.105 s for the weight it replaces.

### 3.10 Stage 9b: the `ldiv!` line

**What was slow.** Applying the factor as `rp2 .= Fp.F1 \ rp` allocates a fresh
281 432-byte result on every call, twice per outer iteration.

**What the measurement said.** `ldiv!(rp2, Fp.F1, rp)` is the same CHOLMOD
`cholmod_solve` on the same data, verified bitwise equal (`rp2 == xa` on the
production state).

**What changed.** One line in `_semquad_pressure_block!`
(`julia/src/semquad_mg.jl:2626`).

**The number.** 0.461 to 0.436 ms for a hot solve, 0.012-0.050 ms per solve,
0.025-0.10 ms per iteration, 0.2-0.5 % of the warm solve. The remaining 61 168 B
per call are `cholmod_solve`'s internal workspace; a persistent `cholmod_solve2`
cache would remove them and its measured ceiling is the same 0.012-0.050 ms, so
it was not implemented.

## 4. What the composition is now

The profile was taken on the 81-iteration configuration, the last before stage 9
changed the count. One warm `F` solve at eight threads is 1.489 s there: 0.325 s
of setup (21.8 %) and 1.164 s of FGMRES (78.2 %), at 14.37 ms per iteration.
Stage 9 then dropped the count to 71 without changing the per-iteration work, and
the shipped warm rows measure 1.324-1.356 s, so the shares below describe the
shipped per-iteration cost while the seconds belong to that session. [inference]

| phase, `F` at eight threads | s | % |
|---|---|---|
| assembly (reduced matrix-free geometry) | 0.236 | 15.8 |
| hierarchy build, warm geometry cache | 0.089 | 6.0 |
| w-BFBT build, warm pressure cache | 0.0003 | 0.0 |
| **setup** | **0.325** | **21.8** |
| FGMRES iterations (slot) | 1.164 | 78.2 |
| **warm solve** | **1.489** | 100 |

Setup is 19-27 % of a warm step across the four cells. The assembly is the
largest setup row (12.7-17.2 %), the warm hierarchy second (4.1-9.4 %), and the
w-BFBT build is a cache hit at 0.3-0.4 ms. The warm hierarchy is not free on a
hit, because every level still rebuilds its leaf-local operator, its eta field,
its exact Jacobi diagonal and the dense coarse operator; a miss costs 0.13-0.28 s
and 0.11-0.15 s, once per tree.

### 4.1 One iteration and one V-cycle

`F` at eight threads, 14.372 ms:

| component | ms | % |
|---|---|---|
| V-cycle | 5.095 | 35.5 |
| w-BFBT block | 3.635 | 25.3 |
| ... first `chol(S1)` solve | 0.976 | 6.8 |
| ... `S2` apply (matrix-free) | 1.732 | 12.0 |
| ... second solve and scaling | 0.923 | 6.4 |
| outer operator `A v` | 2.023 | 14.1 |
| ... matrix-free `K` apply | 0.817 | 5.7 |
| ... `Bt p` | 0.594 | 4.1 |
| ... `Bs u` | 0.569 | 4.0 |
| FGMRES own work | 3.033 | 21.1 |
| ... MGS | 1.091 | 7.6 |
| ... true residual | 0.591 | 4.1 |
| ... `Z`/`W` copies | 1.256 | 8.7 |
| ... norm and basis | 0.096 | 0.7 |
| `Bs zc` | 0.585 | 4.1 |
| **iteration** | **14.372** | 100 |

At one thread the iteration is 19.314 ms; on `P1`/8 it is 20.586 ms with the
V-cycle at 47.0 % and the w-BFBT at 21.9 %, because the larger tree has two
threaded levels. The V-cycle is 4.6x cheaper per iteration than the pre-campaign
baseline at one thread and 5.1x at eight; the w-BFBT only got 1.5x and 1.16x
cheaper and the FGMRES work did not move, so their shares roughly doubled while
the absolute iteration halved.

One correction belongs with the `Z`/`W` copies. That row's timer also wraps the
outer `A z` apply, and the FGMRES rows are scaled by the ratio of the measured
FGMRES total to the sum of its parts. Measured alone the two copies are 0.086 ms
per iteration on `F`/8 and 0.114 ms on `P1`/8, which is **0.5-0.6 %**, and the
rest of the row is the outer apply the `A v` row already charges.

Inside the V-cycle (5.095 ms on `F`/8, 25 applies and 5 plus 5 transfers) the
finest-level applies are 1.963 ms (38.5 %), of which the leaf phase is 1.395 and
the `Cu` products 0.567; the coarse applies are **2.025 ms (39.7 %)**, of which
the leaf phase is 1.550 and `Cu` 0.474; transfers are 0.466 (9.1 %), smoothing
vector ops 0.263 (5.2 %), the coarse Cholesky, `x .+= py` and `fill!` 0.101
(2.0 %), and region overhead with unaccounted time 0.279 (5.5 %). On `P1`/8 the
V-cycle is 9.680 ms with the coarse applies at 45.3 %. The structural fact is
that the coarse applies cost as much as the finest ones, 2.03 against 1.96 ms on
`F` and 4.39 against 3.57 on `P1`: the serial level 5 alone is 1.674 ms against
the threaded level 6's 2.403 ms. **Almost half of the eight-thread iteration is
serial**: the two `F1` solves, `Bt`, `Bs`, the `Z`/`W` copies, MGS, the true
residual and `Bs zc` are 6.6 of 14.4 ms on `F`/8 (46 %) and 7.7 of 20.6 ms on
`P1`/8 (37 %). That share is why every stage from §3.6 on shows a smaller
eight-thread gain than its one-thread gain.

### 4.2 The residual history, and what the plateau means

| k | `F` (81 iterations) | `P1` (65) |
|---|---|---|
| 1 | 9.9999e-1 | 1.0000 |
| 20 | 9.9862e-1 | 9.9401e-1 |
| 30 | 9.9340e-1 | 6.8674e-1 |
| 40 | 9.0887e-1 | 2.9892e-2 |
| 50 | 9.7417e-2 | 6.0157e-4 |
| 60 | 4.6429e-3 | 7.7590e-6 |
| 70 | 1.0131e-4 | - |
| 80 | 1.1494e-6 | - |
| last | 8.6192e-7 (81) | 6.7151e-7 (65) |

The solve is stagnation-limited, not rate-limited. `F` needs 41 iterations to
cross 0.9 and buys 9 % of its total reduction with them; `P1` needs 28. Once the
stagnation breaks the contraction is 0.80 to 0.67 on `F` and 0.70 to 0.65 on
`P1`. At the eight-thread slot cost those plateau iterations are 0.58 s on `F`
and 0.56 s on `P1`, 39 % and 31 % of the warm solve. If the late rate held from
iteration 1 the solve would take 31 and 30 iterations, a 62 % and 54 % cut, and
that is an upper bound because a superlinear drop cannot be extrapolated
backwards; a conservative 20 % cut is 0.23 and 0.27 s. §5 and §6 ask whether the
plateau can be bought.

## 5. The spectrum, and both deflation studies

### 5.1 The probe

`mg_spectrum_deflate.jl` runs a 150-step Arnoldi on `P^-1 A` from `v1 = b/||b||`,
with modified Gram-Schmidt and one reorthogonalisation pass, accumulating the
Hessenberg over both so that `A V_m = V_{m+1} H` holds. `P^-1 A` is
non-symmetric, so the **Ritz values of the Hessenberg matrix** are reported. A
Ritz pair `(theta, y)` has `x = V_m y` and residual `|h_{m+1,m} y_m|`, and the
smallest 20 and largest 10 are re-checked with an explicit `P^-1 A` apply. Two
controls make the small-end answer credible: the 100- and 120-step prefixes of
the same run, and the harmonic Ritz values with shift 0, the quantity that would
reveal a small eigenvalue ordinary Ritz values cannot see.

The deflation prototype uses the two-level preconditioner

```
z = P^-1 v + U (U' A U)^-1 U' (v - A P^-1 v)
```

with `U` an orthonormal real basis of the chosen Ritz vectors and `G = U' A U`
factorised once. With `U` empty the replica reproduces the production solve
exactly: 83 iterations and 5.377e-7 on `F`, 65 and 6.127e-7 on `P1`. `F` needs 83
here against the 81 of §4.2, two iterations at the stagnation cliff, at the same
residual and plateau shape; `P1` reproduces 65. The spectrum measured is the
spectrum of the operator the solver inverts.

### 5.2 The small end is a continuum

The smallest Ritz value of `P^-1 A` is **0.166 on `F` and 0.285 on `P1`**, and
both are converged: the 100, 120 and 150-step prefixes give 0.1725, 0.1681 and
0.1665 on `F` (1.5 % between 120 and 150) and 0.28562, 0.28543, 0.28543 on `P1`.

There is no gap. The consecutive ratios `|theta_{i+1}|/|theta_i|` are 1.038,
1.188, 1.125, 1.080, 1.073, 1.096, 1.073, **1.197**, 1.000, and the largest ratio
in the smallest 64 is 1.197 on `F` and 1.151 on `P1`. Seven of `F`'s 150 Ritz
values sit in [0.1, 0.3), one of `P1`'s. The harmonic Ritz values with shift 0
find **nothing below 0.166 / 0.285**, and the count below 0.1, 0.01 and 1e-3 is
0, 0, 0 on both. The mode structure agrees: the smallest modes are interior,
mixed velocity and pressure, and not wall-localised, with 65.6-98.3 % of `F`'s
velocity energy in the interior band and 74.7-89.8 % on `P1`. There is no single
stalled structure at the small end.

### 5.3 The large end carries the plateau

The other end is populated by a few huge eigenvalues: **1e2 to 1.1e5 on `F` and
4e1 to 3.0e3 on `P1`**, 23 and 15 Ritz values above 10. All are **pressure-only**
(`u%` = 0.0) and all are **numerically in the null space of `A`**: `||A u||` is
1.9e-3 to 4.5e-2 on `F` and 5e-11 to 6e-11 on `P1` against `|theta|` of 1e2 to
1e5. Yet `P^-1 A u = theta u`, so the preconditioner amplifies them by 3 to 14
orders of magnitude more than `A`. The mechanism is the `1e-14` shift: a pressure
mode with `Bt p` almost zero is almost annihilated by `A`, while `S1^-1` on the
corresponding near-null mode of `S1` is enormous. The eigenvector matrix of `H_m`
has condition number **7.5e3 on `F` and 5.9e2 on `P1`** against 1 for a normal
operator, so the preconditioned operator is strongly non-normal and the
non-normality is carried by these modes. That is why GMRES stalls while every
eigenvalue is comfortably away from zero. On `F` the modes sit in the interior
(62-99.7 % of the energy); on `P1` they are bottom and top wall modes (49-65 % in
one wall band, side bands under 2.3 %).

### 5.4 Small-end deflation does nothing; large-end deflation costs more than it saves

Deflating the 4, 8 and 16 smallest Ritz vectors changes the count from 83 to 82,
80 and 79 on `F` and from 65 to 67, 64 and 60 on `P1`. A random 8-direction
control gives 81 and 65, so the `P1` small-16 gain is at most five iterations and
indistinguishable from a generic coarse space of 17 directions; on `F` the
`k = 16` wall clock is *worse* than the baseline (1.230 against 1.164 s).

Deflating the 8, 16 and 32 **largest** Ritz vectors cuts the solve to 73, 65 and
**52** iterations on `F` and 57, 49 and **40** on `P1`, with wall clock 1.164 to
0.798 s (-31 %) and 1.380 to 0.848 s (-39 %). `P1` meets a < 50 target at
`k = 16`; `F` misses it by two at `k = 32`. That is a real iteration-count lever.
The subspace costs a 150-step Arnoldi of **3.07 s on `F` and 4.47 s on `P1`**,
2.6-3.2x the whole solve and 8x the iteration time it saves, and its `G` is
already at `cond = 2.8e4` on `F`.

The cheap route was to build the same pressure subspace from `S1` itself: its
smallest eigenpairs are the near-null modes `S1^-1` amplifies, `S1` is factorised
at setup, and block inverse iteration with `F1` costs a few triangular solves per
vector. It fails. The two pressure subspaces are 51-88 degrees apart on average
(mean 88 at `k = 8` on `F`, 69 on `P1`, 51 at `k = 32` on `F`), the deflated
solves go 80-119 iterations on `F` against 83 and 64-95 on `P1` against 65, and
the best S1 row is **11 % slower** than the baseline. The defect
`||(I - UU') P^-1 A U|| / ||P^-1 A U||` is **5-100 on `F` and 9-21 on `P1`**
against 2-3e-7 and 2-3e-14 for the Arnoldi basis: a deflatable subspace has
defect near zero, and this one is amplified by the operator it was meant to
deflate.

The reason is structural. The pressure part of a huge mode is a near-eigenvector
of the **product** `T = S1^-1 S2`, alignment 1.4-2.8 % on `F`, and only
accidentally close to an eigenvector of `S1` (single-mode overlap 0.61-0.90, `S1`
residual 6.3x the Rayleigh quotient). `S1^-1 p` lands in the near-null subspace
with fraction 0.997-1.000 on `F`, so the amplification happens where the mode
does not live. A `T`-based subspace is not affordable either: one `T` apply is one
`S2` apply plus one `F1` solve, 2.10 ms on `F` and 2.29 on `P1` per vector, so a
16/24/40-vector block is 0.34/0.51/0.84 s, 0.29-0.74x the whole solve for a
`k = 8` gain of 10 iterations.

### 5.5 The accuracy lesson

One result outlives both negatives and it is not about deflation. The
`K`-completed `S1` deflation rows on `P1` reach `relres` **7.21e-7** and 9.06e-7
with a gauge-removed `|dp|/|p|` of **1.0e-4** and 2.2e-5 against the equilibrated
direct solve, 20 to 90 times the baseline's 1.2e-6. The residual gate passes and
the answer is wrong. A low-rank modification `S1 + Z (sigma - Lambda) Z'` goes
further: 5.7-12.4 s to refactor, 40-110x the 0.11-0.15 s cold w-BFBT build, 265,
285 and 500 iterations, relative errors 3e-4 to 25.

So every preconditioner change after this is gated on `|du|/|u|` and the
gauge-removed `|dp|/|p|` against the equilibrated direct solve in the same
process, never on `relres`; the reference's own residual is 9.97e-11 on `F` and
1.84e-9 on `P1`, so the gate is not limited by it. Neither deflation shipped. What
would have to be true for a pressure-space deflation to pay: reachable from `F1`
alone (no `S2` apply per vector) **and** nearly invariant (defect below 1e-3),
and with this w-BFBT those cannot both hold because the invariant space belongs
to `T`. The lever that follows is a pressure block whose product is better
conditioned where `S2 >> S1`, which is §6.

## 6. The Schur variants probe

Four families were measured against the shipped block at the production tolerance
on both states, accuracy gated on the equilibrated direct solve: an adaptive
weight `D`, a coarse correction on `S1^-1` (two-level), a penalty added to `S1`
(the augmented-Lagrangian idea as a preconditioner-only change), and the exact
restricted diagonal. The protocol is `total = iteration + factor build`, the
iteration a median of three solves, with the rest of the warm solve (0.236 s
assembly, 0.089 s hierarchy on `F`) excluded. Two measurements of the same
81-iteration solve in one process differed by 7.7 % and rows with the same count
span 5-7 %, so differences below about 8 % are not claimed as wins.

### 6.1 The weight family: the exact diagonal wins

`D` is the only lever that changes *both* factors of `S1^-1 S2 S1^-1`, so it is
the only one that can move the `S2/S1` ratio. Five weights, zero per-apply cost:

| weight `D` | contrast | `F` it | `F` iter s | `|du|/|u|` | `P1` it |
|---|---|---|---|---|---|
| shipped `Cu' diag(M_u)`, clip 1e-12 | 1.0e12 | 81 | 1.144 | 1.694e-6 | 65 |
| `diag(K) = diag(Cu' K Cu)` | 9.8e6 | 79 | 1.239 | **3.282e-6** | 70 |
| **`diag(Cu' M_u Cu)` exact** | **5.9e5** | **71** | **1.003** | **7.956e-7** | **65** |
| shipped diagonal, re-clipped 1e-6 | 1.0e6 | 80 | 1.132 | 6.682e-7 | 65 |
| shipped diagonal, re-clipped 1e-4 | 1.0e4 | **155** | 2.170 | 7.364e-7 | 64 |

The exact restricted diagonal is the only weight that helps and the only row that
improves accuracy: 2.1x better in velocity on `F`, unchanged in pressure, `P1`
unchanged to 1 %. `diag(K)` buys two iterations on `F` and pays 2x the velocity
error (3.282e-6, outside the 2.1e-6 class) and five iterations on `P1`. The
re-clip rows show the gain comes from the cross terms: 1e-6 gives 80 and 1e-4
gives 155. `D^alpha` for alpha in {0.50, 0.75, 1.25, 1.50} costs 130, 274, 152 and
500 iterations on `F` with the same weight in both factors, and the `S1`-only
form hits the 500 cap at every alpha on `P1`. The scalar `pscale` is worth 1-4
iterations: `F` minimises at 0.20 with 77, `P1` at 0.30-0.40 with 65, and the
shipped 0.40 is four iterations off `F`'s optimum and within 1 % of the best
single value for the pair.

The exact diagonal shipped once the construction was leaf-local: `how=:matrix`
charged 0.87-0.99 s of sparse assembly and §3.9's leaf-local form is 0.028-0.034 s
at eight threads. With both weights' builds charged the totals fall 10.1 % on
`F`/8, 14.6 % on `F`/1, 7.8 % on `P1`/8 and 0.1 % on `P1`/1. The shipping rule was
"improve the warm solve by more than 5 % on at least one state and do not regress
the other", and the count 81 to 71 is the deterministic part of it.

### 6.2 The two-level family: every sign, every `k`, the same failure

`S1^-1` was augmented with `S1~^-1 = S1^-1 + Z C Z'` in three forms:
`C = (Z'S1Z)^-1` (the classical additive two-level form),
`C = (Z'M_pZ)^-1 - (Z'S1Z)^-1` (coarse replacement by the pressure mass) and
`C = -(Z'S1Z)^-1` (coarse deflation). The coarse spaces were piecewise-constant
aggregates on 2x2, 4x4, 8x8 and 16x8 partitions (`k` = 4, 16, 64, 128) and the 8
smallest `S1` eigenvectors.

Fifteen rows, **fourteen at the 500-iteration cap**. The one that converges takes
296 iterations at 4.179 s of iteration, 3.7x the baseline, and lands at
`|du|/|u|` = 8.902e-6 and `|dp|/|p|` = 6.689e-4, 5x and 3000x the baseline; its
correction size is 0.95, so it is not a correction at all. The number that
explains the rest is the smallest one: at `k = 4` the added term is **8.5e-5**
relative to the fine solve (`||Z C Z'r|| / ||F1^-1 r||`, the smallest of the
fifteen) and the count is still 500 at `relres` 0.32. The product amplifies a
perturbation of `S1^-1` by **1e4 to 1e6** on the directions the solve needs,
because its two `S1^-1` factors and the huge `S2` work by cancelling each other;
anything outside that cancellation destroys it. That is §5.5's phenomenon from
the other side, and it is why the `S1`-based deflation hit 80-119 iterations.

### 6.3 The penalty family: 92 iterations at the smallest lift

The penalty adds `(1/gamma) M_p` to `S1` with `M_p = Cp' M Cp`, parameterised by
the lift `s` of the near-null end: `mu = s * lambda_1 / (z1' M_p z1)`, so the
added term's Rayleigh quotient on the smallest eigenvector is exactly
`s * lambda_1`. At `s = 1`, a relative change of 1.7e-10 of the matrix, it already
costs 11 iterations on `F` (92) and 12 on `P1` (77), plus a 10x and 56x worse
pressure (`|dp|/|p|` 2.255e-6 and 7.153e-5 against 2.165e-7 and 1.274e-6). At
`s = 10` it is 419 iterations on `F` and 130 on `P1`; at `s >= 1e2` both hit the
cap. Re-tuning `pscale` gives 92-96 iterations, so the row is insensitive to the
one knob that might rescue it. The literature's augmented-Lagrangian counts
(15-27 iterations at a 1e10 dynamic ratio with a multigrid `(1,1)` block, 5-6
with an exact one) come with a different operator and a gamma-robust smoother,
which changes the system rather than the inverse; this probe measured the
preconditioner-only change.

### 6.4 The knife edge

Every family lands worse than the shipped block, and the reason is that the
iteration count at this contrast is not a smooth function of the Schur
approximation:

* an additive correction of relative size **8.5e-5** turns 81 iterations into 500;
* a **1e-9** relative change of `S1` (the penalty at `s = 10`) costs 419;
* a **1.5 %** change of `D` buys 10 iterations and improves the accuracy;
* re-clipping **16** of 61 964 diagonal entries swings the count from 80 to 155.

The shipped point was tuned by these measurements and then made exact where the
exact object was available; it was not derived from a conditioning argument, and
the sharpness of the response is the evidence for that [inference]. A wall-clock
win from the pressure block alone is also unavailable: even the winner is -10.4 %
of the `F` total with the build amortised, -8.2 % of the measured warm solve, and
0 % on `P1`. What remains are the levers that do not perturb the cancellation: a
cheaper application of the same block (§7), or a change to the outer solve.

## 7. The two `chol(S1)` solves

`S1` is 7 424 x 7 424 with 198 490 nonzeros on `F` (244 422 on `P1`'s 9 556). Its
factor has **649 814 nonzeros, a 3.27x fill**, 7.80 MB, and costs 16.4-16.5 ms,
14 % of a 113-119 ms pressure-block build and 0.02 % of a 1.3 s solve. One
`F1 \ b` is one forward and one backward substitution: two passes over 1.30 M
nonzeros, 15.6 MB of values and indices read.

| form | 1 thread | 8 threads |
|---|---|---|
| `rp2 .= F1 \ rp` (the pre-stage-10 form) | 0.461 ms, 281 432 B per call | 0.46-0.49 ms |
| `ldiv!(rp2, F1, rp)` (shipped) | **0.436 ms**, 61 168 B | 0.41-0.44 ms |
| in the outer iteration (the profile's slot) | 0.948 ms | 0.948-0.976 ms |

The solve **does not scale because there is nothing to scale**: SuiteSparse's
`cholmod_solve` is serial and has no threaded triangular solve, and this code
pins `BLAS` to one thread anyway. 0.436 ms is the cache-resident floor, 0.34 ns
per nonzero; in the iteration it costs 0.948 ms, **2.2x the hot measurement**,
because the V-cycle and `S2` stream tens of MB through the caches between two
solves. A sparse matvec over the same `L` costs 0.290 ms, 27.1 GB/s of `L`, so
one solve's two-pass DRAM floor is 0.581 ms and the pair's is 1.16 ms. Just over
**half** of the in-place 1.90 ms per iteration is memory traffic (0.948 minus
0.436 is 0.51 of 0.95) and the rest is the kernel at 0.34 ns per nonzero. The
pair is 13.2 % of the eight-thread iteration, does not scale (1.84 ms at one
thread, 1.90 at eight), and the 0.74 ms a smarter solve could recover is 5.1 % of
the iteration.

**Route (a), the level-scheduled solve, rejected.** `semquad_s1solve.jl`
renumbers `L` twice, in forward row-dependency and backward column-dependency
order, so a level's rows are contiguous in both, and publishes one region phase
per level whose work clears `minwork`. It is correct to 5.2e-16 against CHOLMOD
(the difference is row-sum order), bitwise independent of the thread count, and
costs 28 ms at eight threads or 309 ms at one to build, plus 16 MB for two CSR
copies. The structure rules it out first: the column etree is 1 004 deep, the
forward schedule has **1 005 levels** with median width 2 and maximum 338, and
the widest level holds **2 578 nonzeros, 0.9 us of work** at 0.32 ns per nonzero,
against a 7.7 us phase and a 26-44 us fork/join. A level would need about 1.9e5
nonzeros to break even, and 1 005 phases would cost 7.7 ms per solve, 18 times
the solve. Measured: 0.945 ms at one thread and 1.128 ms at eight against
CHOLMOD's 0.486 ms, **0.43x**, with the eight-thread run being the serial kernel
plus one empty fork/join because `minwork = 3000` lets **zero** levels qualify.
Lowering it to 600 publishes 93 forward and 226 backward phases at 0.2-0.9 us
each. `P1` says the same at 0.42x with a widest forward level of 3 020. Even with
perfect parallelism the pair ceiling is **2.69x** (1.96x forward, 4.28x
backward), and this kernel is 2.2x slower than CHOLMOD's cache-resident and at par
in the DRAM-bound state, so the two effects cancel.

**Route (b), the inexact inner solve, rejected.** The ASPECT and deal.II recipe:
CG on `S1` at a loose relative tolerance, preconditioned by `diag(S1)` or an IC(0)
(pattern = lower triangle, 99 k nonzeros, 6.6x fewer than `L`). The diagonal
preconditioner **does not converge at all**: 200 iterations is the cap and the
solve stalls at 1.6e-3. IC(0) converges in 51-74 iterations at 17.2-22.5 ms per
solve against 0.436 ms, 40-50x. Inside the warm solve the trade never pays: where
the inner solve converges the outer count is **unchanged** (71 at 1e-4, 1e-5 and
1e-6; 76 at 1e-3), so the solve takes 4.04-4.59 s against 1.356 s, **0.30-0.34x**.
At 1e-2 it breaks the outer solve (500 iterations, `relres` 0.83, `|du|/|u|` 1.7).
Accuracy where it converges is fine, so the rejection is on cost. The break-even
is the number to remember: an inexact solve must cost at most a third of 0.95 ms,
0.32 ms, and one CG iteration is already 0.11 ms with the diagonal or 0.30 ms
with IC(0), so the inner solve would have to converge in **1-2 iterations**, a
preconditioner 20-40x stronger than IC(0). `bfbt_s1 = :mass` is not a third route:
it builds `F1 = chol(Cp' M Cp)` and keeps `kind = :bfbt_m1`, so the same two
solves remain with a different factor, and on `F`/8 it measures 500 iterations at
`relres` 0.99 (7.37 s). The earlier claim that `:mass` removes the two solves is
wrong and corrected here.

**Why the problem is not ASPECT's A22.** ASPECT and step-56 work because their
`A22` is a **pressure mass matrix**, and a CG solve of a mass matrix takes 21-22
inner iterations independent of the mesh at 1e5 to 1e6 rows. `S1 = Bs D^-1 Bt` is
a **Schur complement** with condition number growing like `h^-2`, at 7 424 rows,
and at that size a sparse Cholesky solve costs less than two CG iterations. The
same arithmetic inverts the conclusion: here the exact factor is the cheap option
and the inexact solve the expensive one, the opposite of the regime that made the
ASPECT recipe standard.

What ships from this page is one bitwise line, §3.10. The two routes stay
selectable and non-default (`bfbt_s1_solve = :chol | :level | :cg_diag |
:cg_ichol`, with `bfbt_s1_tol`, `bfbt_s1_maxit`, `bfbt_s1_minwork`,
`bfbt_s1_chunk`), because the measurement is the point and it has to be
reproducible. G227 pins that `:chol` reproduces the shipped apply bitwise, that
`:level` matches it to round-off and is bitwise identical with one and eight
workers, and that the pressure cache is keyed on the inner solve.

<!-- PARTC -->

