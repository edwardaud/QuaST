# MG literature: what the mature codes do, and what it cost to check

**Scope.** This is the project's literature record for the matrix-free multigrid Stokes
solver. It merges five working notes into one volume: the speed reference, the matrix-free
vectorisation reference, the S1 variants reference, the S1 solve reference, and the
verification pass over four proposed levers. No number changed in the merge. The provenance
did: where a note cited a paper, this volume cites the paper, and where a note recorded one
of our measurements, this volume names the document and section that own it.

**Three labels.** *Quoted* means the number is in the cited document and this volume gives
the table or section it came from. *Measured here* means it came out of a run recorded in
this repository. *Inference* means we combined the two, and the arithmetic is written out so
a reader can disagree with the step instead of the conclusion. Nothing here is a fourth kind.

**What the merge leaves behind.** The five notes stay in the tree (they are deleted by hand,
later). Their section numbers survive in commit messages and a few cross references. The
material now lives here: per-cycle costs and smoothers in §2 and §3, the pressure block in
§4, the triangular solves in §5, coarsening in §6, kernels in §7, and the four levers, with
an explicit account of where the original proposal read a source loosely, in §8.

## 1. Why a literature volume exists

The rule this project ended up with is one sentence: a claim about what mature codes do needs
a source, and the source needs a number. A citation without a number is allowed, but it gets
a label, "bibliographic only", and it cannot price a change. That is the method. This file is
where the sentences that begin "mature codes do" come to be checked.

Two incidents bought the rule. The first was a correction. For a while the counts 231, 3279
and >10000 circulated in our notes as the iteration counts of weighted BFBT at high contrast.
They are the counts of the inverse-viscosity-weighted pressure mass matrix, Rudi et al.
Table 1a; the same three problems in their Table 1b give 47, 39 and 60 for w-BFBT. The gap
between those rows is the entire argument for a BFBT-type Schur complement, so reading the
wrong row would have inverted a conclusion. The second was an overturn. The 2026-11-11
statement that multigrid "does not pay at this size" was arithmetic on one state's setup,
0.42 s of assembly plus 2.2 s of hierarchy plus 0.2 s of BFBT against a 2.63 s warm direct
solve. After the tuning in [MG.md](MG.md) §11-§16 that setup is 0.325 s on `F`,
and the eight-thread MG solve is faster than the accuracy-matched direct solve
([PERFORMANCE.md](PERFORMANCE.md) §3.5). The literature did not change between those two
statements. Our accounting did.

The volume then does three jobs, with different failure modes.

**Reject.** Some directions are already answered by a number that exists. Unweighted pressure
mass diverges at our contrast (step-56 ships it; our `:mass` diverges at 16384x,
[SOLVER.md](SOLVER.md) §4.2). PCD and LSC are constant-viscosity Oseen tools, and with no
advection in the momentum equation PCD collapses to a pressure Laplacian at 2.7x the
per-apply cost of `:mass` for the same count. A level-scheduled triangular solve wins nothing
at 7.4k rows, and the vendor manual says so before we do.

**Price.** Some directions are plausible and need a number before they earn engineer-weeks.
Braess-Sarazin (4.0x an apply per V-cycle, and the paper reports no timings), augmented
Lagrangian (the only variant that changes the answer, months of work), adaptive Galerkin
coarsening (velocity block only, 1.2-1.4x by our arithmetic), mixed precision (+83 % in the
cited measurement, and not bitwise). Pricing is the bulk of this volume.

**Adopt.** Three things came in on a citation plus a measurement: the 32-lane batching that
gave 2.72x (deal.II's vectorise-over-cells argument, [MG.md](MG.md) §11.2), the
eta lane table that gave 1.06-1.10x on the V-cycle (step-37's coefficient layout,
[MG.md](MG.md) §15), and the exact restricted BFBT diagonal that gave 81 → 71
iterations on `F` (the true-versus-lumped distinction Rudi et al. report only in a figure,
[MG.md](MG.md) §2, §10).

The ledger, in one place. Each row is developed in the section named at the right.

| candidate | what the source gives | what settled it | verdict | where |
|---|---|---|---|---|
| Chebyshev smoother | degree 2-4, 1-2 sweeps/level | diverges on both production states | rejected, reverted | §3 |
| unweighted, then 1/η mass | shipped by step-56 and ASPECT | `:mass` diverges at 16384x; 1/η mass 68 vs BFBT 39 | keep the three-factor BFBT | §4 |
| `D = diag(K)` weight | "fast for k = 2", figure only | 79 vs 81 on `F`, 70 vs 65 on `P1`, 1.9x velocity error | rejected; exact restricted diagonal shipped | §4 |
| augmented Lagrangian | 5-6 at 1e10 with an exact (1,1) block | 92 iterations at the smallest lift | a reformulation, not a variant | §4 |
| inexact inner solve on `S1` | ASPECT 1e-6 RHS-relative; step-56 fixed 21-22 | 45-75 CG iterations, 0.30-0.34x | rejected | §5 |
| level-scheduled tri. solve | Anderson-Saad, Saltz; 12.07x fewer barriers | 0.9 µs of work in a 7.7 µs phase | rejected | §5 |
| slice-wise cache blocking | 1.26x at p = 8 in 3-D | -2.0 % to +1.0 %; cold cache costs 8.7 % | rejected | §8 |
| Braess-Sarazin | 6/7/9/9 at p = 2, μ = 1e6; 4.0x, no timings | not measured | prototype behind a gate | §8 |
| adaptive Galerkin | velocity block only, 1e10 dofs, 1e6 jumps | full Galerkin already 69 → 61, 65 → 46 | narrowed, 1.2-1.4x | §8 |
| libCEED partial assembly | optimal transfers, near-optimal flops | our apply already is element-level PA | not a lever | §8 |

A volume like this has an obvious failure mode: it becomes the place where plausible
sentences go to acquire citations. The countermeasure is that every section ends in a
decision that was measured or priced, and §8 states, item by item, where the original
proposal read its sources loosely.

## 2. The cost anatomy of a matrix-free V-cycle

Three numbers dominate this subject and they are in three units. HPGMG reports nanoseconds
per unknown per V-cycle, Clevenger and Heister report nanoseconds per dof for a whole GMG
Stokes iteration, and deal.II reports fractions of peak and of bandwidth. Comparing them
without converting is where most of the "we are five times slower" statements here came from.

**HPGMG.** One 512³ V-cycle on 8 MPI ranks × 8 threads takes 0.483 s at 2.78e8 dof/s, which is
3.6 ns per unknown per V-cycle, or 0.23 µs per dof per core
([HPGMG README](https://github.com/hpgmg/hpgmg#readme)). The two units are the same
measurement: with `t` the wall time, `N` the unknowns and `P` the cores, `t/N` is 3.6 ns and
`t·P/N` is 0.23 µs, because `P` is 64. The component split is the useful part: smoothing
64 %, residual 12.5 %, ghost exchange 8.9 %, interpolation 4.3 %, restriction 2.3 %. A fast
V-cycle is mostly smoothing, and everything else together is a third of it.

**Clevenger and Heister.** A whole GMG Stokes iteration costs 30-53 ns/dof on 32 cores in
3-D, Q2/Q1, at 18.5M dofs ([arXiv:1907.06696](https://arxiv.org/abs/1907.06696), Table 1),
which is 0.96-1.7 µs/dof/core by the same conversion.

**Ours.** On `F` (69 388 saddle dofs, 61 964 velocity and 7 424 pressure) one warm solve at
eight threads is 1.489 s for 81 iterations at 14.372 ms each
([MG.md](MG.md) §5). Per saddle dof that is 207 ns of wall time and 1.66 µs
per core, so the per-core number sits inside Clevenger's band and the per-dof wall number is
four to seven times theirs, which is mostly a statement about the cores in the denominator.
The earlier "4-6x that per-core rate" was computed on the pre-tuning state, 74 ms at 98k dofs
or 6.0 µs/dof/core; its conclusion, that the gap is implementation and not algorithm class,
survives, and the number that produced it no longer describes the code.

| where | what is counted | cost | source |
|---|---|---|---|
| HPGMG, 512³, 8 ranks × 8 threads | one V-cycle, per unknown | 3.6 ns (0.23 µs/dof/core) | HPGMG README |
| Clevenger, 18.5M dofs, 3-D, 32 cores | whole GMG Stokes iteration, per dof | 30-53 ns | arXiv:1907.06696, Table 1 |
| ours, `F`, 69 388 dofs, 8 threads | whole FGMRES iteration, per saddle dof | 207 ns (1.66 µs/dof/core) | MG_CAMPAIGN §5 |
| ours, `F`, 8 threads | the V-cycle inside that iteration | 5.095 ms, 35.5 % | MG_CAMPAIGN §5 |
| ours, L6, 169k dofs, 8 threads | one V-cycle of a 45.3 ms iteration | 16.41 ms, 36 % | MG_PROFILE §7.1 |

The rest of our iteration: the w-BFBT pressure block is 3.635 ms (25.3 %), the outer operator
apply 2.023 ms (14.1 %), FGMRES's own work 3.033 ms (21.1 %) and `Bs zc` 0.585 ms (4.1 %).
That is why this project spent a year on kernels while a production spectral-element code
spends its time elsewhere: in Nek5000 the pressure solve is 85 % of the runtime and the
BoomerAMG coarse grid 34 % ([Fischer](https://mfem.org/pdf/seminar/fischer.pdf)). Our pressure
block is a quarter and our V-cycle a third, and the Krylov wrapper neither code profiles is
the second largest item here.

**The arithmetic-intensity argument.** Let `F` be the flops a kernel performs, `B` the bytes it
moves and `N` the unknowns it touches. Then `F/N` is flops per dof, `B/N` is bytes per dof,
and the arithmetic intensity is `AI = (F/N)/(B/N) = F/B`, flops per byte. The roofline model
gives an achievable rate `min(P_peak, AI·B_peak)` for a peak rate `P_peak` and an achievable
bandwidth `B_peak`; the two cross at the ridge point `r = P_peak/B_peak`, below which a kernel
is bandwidth-bound and above which it is compute-bound
([Williams, Waterman & Patterson, CACM 52(4), 2009](https://doi.org/10.1145/1498765.1498785)).

For our leaf at p = 2 in 2-D: `n = 3` nodes per direction, `n_q = p+2 = 4` Gauss points, 18
velocity dofs per leaf. One 2-D contraction costs `n²n_q + n_q²n = 84` multiply-adds, the
kernel does eight of them (four gradient components, four output pieces), and there are four
pointwise blocks of 16 entries. So

```
flops/leaf = 8 x 84 x 2 + 4 x 16 x 5 = 1664   -> 92 flops/dof
bytes/leaf = 18x8 (v) + 18x8 (out) + 16x8 (eta) + 18x8 (ids) + 24 = 584  -> 32 B/dof
AI         = 92 / 32 = 2.9 flop/byte
```

The counts come from the code ([MG_LITERATURE.md](MG_LITERATURE.md)
§2); the arithmetic is an inference. The textbook figure is `4(p+1) = 12` multiply-adds per dof
per component, and we use `n_q = 4` rather than 3 with both components' gradients as four
separable pieces, which is where 37 comes from.

Ridge points: AVX2 gives 16 flop/cycle (2 FMA units × 4 doubles) and AVX-512 gives 32, so at
3-3.5 GHz a core peaks at 48-112 GFlop/s; per-core bandwidth is 6-8 GB/s from DRAM and
10-25 GB/s from L2/L3. Dividing gives a ridge of 6-15 flop/byte from DRAM, 2-5 from L2 and
below 1 from L1. At `AI = 2.9` the kernel sits on the L2 ridge, where the model cannot decide
and the measurement has to. Ours is 0.5 MB, 62k doubles, resident in L2/L3: at 15 ns/dof the
apply takes 0.93 ms at eight threads and moves about 2 MB, 2.2 GB/s, a few percent of DRAM
bandwidth. The kernel is issue-bound and latency-bound, and that reading has been confirmed
twice since: the 32-lane leaf moves 6.9 GB/s against a 35.4 GB/s copy and issues 9.0 GFLOP/s
against a 35 GFLOP/s AVX2 ceiling, about 20 % and 26 % of the two roofs, and a fully cold
cache costs it 8.7 % ([MG.md](MG.md) §16.3, §16.5).

That single fact orders every kernel lever here. Batching leaves `F/B` unchanged: one lane of
a block does the same 18 loads, 18 stores, 16 coefficient reads and 18 index reads as the
scalar kernel, and the SoA staging adds about 16 B/dof of L1 traffic. So batching can only buy
instruction issue, and it bought 2.72x. Cache blocking also leaves `F/B` unchanged, so it can
only pay if capacity binds; capacity does not bind, and it measured -2 % to +1 %. Mixed
precision halves the bytes per lane, so it moves `AI` as well as the instruction count, which
is why it is the one kernel lever the literature supports with a large number (up to +83 %
from a single-precision V-cycle with a double outer correction, Kronbichler & Ljungkvist,
TOPC 2019) and the one we have not measured.

Two benchmarks complete the picture. HPGMG ranks by MEquations per second for a full FMG solve
and defines textbook efficiency as under 10 work units for the whole solve, a work unit being
one operator apply ([HPGMG 1.0](https://escholarship.org/uc/item/00r9w79m)); Kohl and Rüde
reach that budget with an Uzawa-type monolithic multigrid
([arXiv:2010.13513](https://arxiv.org/abs/2010.13513)), which is why a monolithic smoother
stays on the list in §8.

## 3. Smoothers: Jacobi, Chebyshev, and the monolithic alternatives

Damped Jacobi applies one scalar to the whole upper spectrum, so its amplification is nearly
flat there. A degree-`k` Chebyshev semi-iteration has `k+1` coefficients chosen to minimise
the amplification on an interval `[λ_min, λ_max]`, so it suppresses high-frequency error far
better per apply, and it needs the operator, the diagonal and a spectrum estimate, with no
triangular solve
([step-37](https://dealii.org/current/doxygen/deal.II/step_37.html);
[Adams, Brezina, Hu & Tuminaro, JCP 188:593-610, 2003](https://doi.org/10.1016/S0021-9991(03)00194-3)).
The symbols: `A` is the level operator, `D = diag(A)`, `λ_max` its largest eigenvalue and
`λ_min` the smallest, and `κ = λ_max/λ_min` the window ratio. In practice `λ_max` comes from
10-20 power or Lanczos iterations or a Gershgorin bound, and `λ_min` is set to `λ_max/κ` with
`κ` between 2 and 30 (HPGMG uses 3.5, Jodlbauer et al. use 2).

What the mature codes run is modest in degree. ASPECT's ML preconditioner uses degree 2 and
calls it "strongly recommended"
([parameters.cc:504-524](https://github.com/geodynamics/aspect/blob/340c0d018/source/simulator/parameters.cc#L504-L524)).
Clevenger and Heister's GMG uses degree 4 with one sweep per level. HPGMG-FE uses two pre-
and three post-sweeps at degree 2 as just noted. Jodlbauer, Langer, Wick and Zulehner measured
the optimum once cost is counted: the best reduction per second comes at `(k_A, k_S) = (1, 1)`
with two smoothing steps in 2-D and `(3, 1)` with two in 3-D, the rates are h-independent and
the smoother cost grows linearly with degree
([arXiv:2205.15770](https://arxiv.org/abs/2205.15770), §3.3-4.4). Munch et al. run their
production V-cycle at degree 3 in single precision with AMG on the coarsest level
([arXiv:2203.12292](https://arxiv.org/abs/2203.12292)).

Ours is damped Jacobi with a taper of 1, 2, 2, 4, 4, 8, 8 sweeps from fine to coarse at
`ω = 0.55` (`semquad_mg.jl:158-176`). A taper cycle performs about `sum(pre + post + 1)`
applies weighted by level, roughly five full-grid applies here, so the coarsest level alone
burns 8 + 8 Jacobi sweeps, and the literature's argument is that one degree-3 Chebyshev
evaluation, three applies, replaces them.

**The measurement, which is the point of this section.** A Chebyshev semi-iteration was
implemented, with `λ_max` estimated once per hierarchy by power iteration on `D⁻¹A` and the
window `[λ_max/κ, λ_max]`. On a uniform low-contrast tree it converges: 27 iterations become
82 at degree 2 and 47 at degree 4. On both production states it diverges. `F` stalls at a true
residual of 0.97 and `P1` at 0.99 after 500 iterations, at degree 2 and at degree 4, from
several `λ_max` estimates ([MG.md](MG.md) §3.1). The mechanism is visible in
the spectrum: the reduced contrast operator has near-null modes outside the polynomial's
positive window, and the polynomial amplifies them. The code was reverted rather than shipped
behind a flag that can silently diverge.

The interesting part is that no citation was misread. Chebyshev smoothing is correct on the
problems it was measured on, which are elliptic operators whose small end is a smooth
Laplacian mode. Our small end is a near-null pressure band, and a window fitted to
`[λ_max/κ, λ_max]` does not contain it. That is the same band that limits the Schur complement
in §4 and the coarse space in §6, and it is recorded once here because one property of one
operator explains three failures. It also says the right place to attack is the spectrum: the
coarse-space work came after this, and the smoother stayed Jacobi.

**Braess-Sarazin and Vanka.** Wichrowski's patch-smoother paper is the only recent source with
contrast-dependent counts. At p = 2, 2-D, a 3×3 coarse grid, L = 4 and `μ = 10⁶` in the
central cell, FGMRES to 1e-8, Table 3 gives 6/7/9/9 iterations at distortion 0/10/25/35 %
against 6/7/8/10 for an exact local solve. On a single patch the jump is swept to `10⁸` "with
iteration counts showing only a minimal increase", and at p = 7 it degrades to 8/8/10/12
against 5/6/7/7 ([arXiv:2601.13230](https://arxiv.org/abs/2601.13230)). The cost is the
problem: "the total cost of the p-multigrid V-cycle ... is around 4.0 times the cost of a
single application", and the paper reports no timings, because "our performance metrics focus
primarily on iteration counts ... rather than absolute runtimes". Spies, Olson and MacLachlan
work on our element type, Q2-Q1 Taylor-Hood in 2-D, and find that monolithic multigrid can
outperform block-triangular preconditioning, with Vanka "up to 100 % slower than
Braess-Sarazin" on CPU and "more than 20 %" faster on GPU, at constant viscosity in the unit
square, and they flag the "high amount of algorithmic tuning" the relaxation needs
([10.1177/10943420241261989](https://doi.org/10.1177/10943420241261989)).

**The break-even arithmetic.** Let `n = 70` be the shipped outer count, `a` the applies one
taper V-cycle costs (about 5), and `c` the cost of the replacement cycle in the same unit.
The shipped solve costs `n·a`, the replacement `m·c`, and break-even is `m·c < n·a`. With
`c = 4.0` and `n·a = 350`, that is `m < 87`. The earlier note priced it as `m < 18` by
setting `a = 1`, which charges one apply per outer iteration; that answers "how far must the
count fall if the cycle costs one apply", and our cycle does not. Both readings agree on what
matters: 6-10 published iterations sit far below either threshold. What is in doubt is
whether a 6-10 count survives at 1e6-1e7 contrast on this operator and whether 4.0x is what a
vertex-patch smoother costs in 2-D here. The contrast evidence is a high-order patch setting,
the efficiency evidence is Q2-Q1 at constant viscosity, and neither reports a wall-clock
number. The verdict is a gated prototype, and the correction of 18 to 87 is recorded because
18 was being quoted as a measured threshold.

## 4. The Schur block

The saddle operator `A = [[K, Bᵗ], [Bs, 0]]` has `K` the viscous block from `2η ε:ε`, `B` the
discrete gradient, `Bs` the divergence (the negative transpose continuously, kept separate
because of the hanging-node constraints), and `S = Bs K⁻¹ Bᵗ` the Schur complement. Every code
here uses the same block-triangular preconditioner `P⁻¹ = [[K, Bᵗ], [0, S̃]]⁻¹`, and every
difference is in how `S̃⁻¹` approximates `S⁻¹`.

Our shipped form is a three-factor product, `S̃⁻¹ = S1⁻¹ S2 S1⁻¹`, with `S1 = Bs D⁻¹ Bᵗ` and
`S2 = Bs D⁻¹ Kred D⁻¹ Bᵗ`. `D` is the BFBT weight, a diagonal; `Kred` is the reduced viscous
operator whose inverse the middle factor approximates; the outer factors are exact sparse
Cholesky solves of `S1`. They make the product SPD, and dropping one leaves
`S1⁻¹ S2 ≈ S1⁻¹ S⁻¹`, the wrong operator unless `S1 ≈ I`
([MG_LITERATURE.md](MG_LITERATURE.md) §4). The cost is two `chol(S1)` solves and one
matrix-free `S2` apply per outer iteration, 3.635 ms or 25.3 % of the iteration
([MG.md](MG.md) §5).

**What the reference codes do.** ASPECT builds `A22` as `(1/η) p p`, a viscosity-weighted
pressure mass matrix, and solves it with an inner CG whose tolerance is relative to the
right-hand side, `SolverControl(1000, src.block(1).l2_norm() * S_block_tolerance)`,
preconditioned by an ILU of the block
([solver.cc#L274-L350](https://github.com/geodynamics/aspect/blob/340c0d018/source/simulator/solver.cc#L274-L350),
[assembly.cc#L419-L422](https://github.com/geodynamics/aspect/blob/340c0d018/source/simulator/assembly.cc#L419-L422)).
The defaults form a hierarchy: outer 1e-7, `S` block 1e-6, `A` block 1e-2 in the expensive
phase, and no `A` solve in the 200-step cheap phase, where one AMG sweep stands in; the `S`
block is solved in both phases. BFBT is "more expensive, but works better for large viscosity
variations", and the weighting comes from Kronbichler, Heister and Bangerth
([10.1111/j.1365-246X.2012.05609.x](https://doi.org/10.1111/j.1365-246X.2012.05609.x)).
deal.II's step-56 is the same shape with the mass matrix solved "approximately by using CG
with ILU" to `SolverControl(1000, 1e-6 * src.block(1).l2_norm())`, a fixed 21-22 inner
iterations independent of the mesh
([step-56](https://dealii.org/current/doxygen/deal.II/step_56.html)). Its table is the
cleanest statement of why the approximate route exists: at 15 468 dofs UMFPACK takes 2.65 s to
set up and 2.8 s to solve against 0.3 + 1.3 s for FGMRES with GMG, at 112 724 the pair is
236 + 237 s against 2.6 + 14 s, and at 859 812 the direct solve does not run while FGMRES with
GMG takes 20 + 101 s. Trilinos offers the same ingredients and no direct solve: Ifpack2's
incomplete factorisations (`RILUK`, `ILUT`, `AdditiveSchwarz`) and ML or MueLu, which deal.II
wraps as `TrilinosWrappers::PreconditionAMG` "based on the Trilinos ML implementation".

**Why their tolerance recipe does not transfer.** Both codes' `S̃` is a pressure mass matrix,
well conditioned, which is why 21-22 iterations suffice and the count is h-independent. Our
`S1` is a Schur complement, `Bs D⁻¹ Bᵗ`, whose condition number grows like `h⁻²`; an inner CG
on it needs 45-75 iterations and measures 0.30-0.34x of the exact solve
([MG.md](MG.md) §3.9, and §5 below). The lever they pull is available
because their block is a mass matrix. Ours is not that object.

**The contrast dependence.** Rudi, Stadler and Ghattas
([10.1137/16M108450X](https://doi.org/10.1137/16M108450X)) is the systematic study and the one
that validates our shipped choice: `Q_k × P_disc(k-1)`, GMRES restart 100, 1e-6, density ratio
`DR(µ)` from 1e4 to 1e10, 1-28 sinkers, 128³ elements. Table 1, iterations:

| sinkers \ DR | 1e4 | 1e6 | 1e8 | 1e10 |
|---|---|---|---|---|
| S16, `M_p(1/µ)` | 85 | 167 | 231 | 891 |
| S24, `M_p(1/µ)` | 117 | 286 | 3279 | 5983 |
| S28, `M_p(1/µ)` | 108 | 499 | 2472 | >10000 |
| S16, w-BFBT | 40 | 45 | 47 | 48 |
| S24, w-BFBT | 31 | 32 | 39 | 55 |
| S28, w-BFBT | 29 | 31 | 42 | 60 |

The left block is the inverse-viscosity-weighted mass matrix, the right block the weighted
BFBT. "Only 60 iterations are needed for the most extreme problem ... for which convergence
with `M_p(1/µ)` essentially stagnated", and "each iteration with the weighted BFBT ... is
computationally more costly". Table 3 adds the scaling: 40 iterations at `ℓ = 4`, 33-34 for
`ℓ = 5-10` (mesh-independent), 33 → 67 for `k = 2-8` (order-dependent). The row that matters
for us is the one that fails: the cheap choice our reference codes ship is Rudi's left block,
and at our contrast `:mass` diverges at 16384x while the 1/η-weighted mass needs 68 outer
iterations against BFBT's 39 ([SOLVER.md](SOLVER.md) §4.2). We pay a second Cholesky solve to
stay in the 29-60 band.

**The 2025 correction.** Burkhart, Kohl, Wohlmuth and Zawallich
([10.1007/s13137-025-00280-5](https://doi.org/10.1007/s13137-025-00280-5),
[arXiv:2506.04157](https://arxiv.org/abs/2506.04157)) use the same w-BFBT with `M_√η` and
Rudi's asymmetric boundary scaling, and solve `B M_√η⁻¹ Bᵗ` by CG with GMG to 1e-10. Their
"V-cycle BFBT" replaces the diagonal by `Â_C`, one A-block GMG V-cycle (`m_V = 1`, Chebyshev
degree 1), and solves `B Â_C⁻¹ Bᵗ` by CG with a mass preconditioner to `tolV-BFBT ≈ 0.1`, "a
relatively large tolerance ... sufficient for a solver suitable for geophysical applications
with realistic viscosity contrasts". It "needs the lowest number of iterations and is the
fastest", weighted BFBT with asymmetric scaling "is the slowest", plain mass is second
fastest, and "when solving to a small relative residual ... all approximations perform
comparably fast". Counts are in Figure 12 only. The mechanism is the one our diagnosis
predicted: the variant that helps puts the same approximate `K⁻¹` in both factors instead of a
diagonal.

**Augmented Lagrangian.** The only published family that beats the BFBT band changes the
operator. Shih, Stadler and Wechsung add `γ BᵗW⁻¹B` to the `(1,1)` block and approximate the
new Schur complement by `M_p(1/µ)⁻¹ + γM_p⁻¹` (P1) or `(1+γ)M_p(1/µ)⁻¹` (P2)
([10.1137/21M1430698](https://doi.org/10.1137/21M1430698)). Table 1, 2-D sinkers, 394 754
dofs, exact `(1,1)` solve, FGMRES to 1e-6, iterations at `DR(µ)` = 1e4/1e6/1e8/1e10:

| `γ` | 1e4 | 1e6 | 1e8 | 1e10 |
|---|---|---|---|---|
| 0, mass alone | 32 | 48 | 59 | 70 |
| 10, P1 / P2 | 7 / 10 | 9 / 16 | 10 / 20 | 13 / 24 |
| 1000, P1 / P2 | 2 / 2 | 3 / 4 | 4 / 5 | 5 / 6 |

With multigrid for the `(1,1)` block (Table 4, 41 992 563 velocity dofs in 3-D) the 2-D
`γ = 0` row is 55 at 1e6, `γ = 10` gives 11/22 at 1e6/1e10 and `γ = 1000` gives 13/15, and the
3-D 1e8 row is 51/51, 15/15 and 14/14. Their text: "values of γ from 10 to 1000 results in
convergence within 30 iterations in both 2D and 3D experiments". The price is Table 2:
standard GMG with Jacobi "fail[s] to converge within 300 iterations for γ = 10", AMG "fails to
converge for larger dynamic ratios or γ", and the fix is a discretisation-dependent smoother
and transfer. That is a new operator, smoother, transfer and setup path, and a different system
for FGMRES, so it is months of work and not a preconditioner change in the sense of §5.

The origin papers say what "robust in viscosity" means, and it is narrower: the ideal AL
preconditioner gives 9/9/10/10 iterations at `ν = 0.1`, 7/7/6/7 at 0.01 and 8/8/8/7 at 0.001
across 16²-128² grids, "independent on both the mesh size h and the viscosity ν", while the
modified AL degrades 14 → 65 as `ν` goes 0.1 → 0.001 at 128² (Benzi & Olshanskii
[10.1137/050646421](https://doi.org/10.1137/050646421); Benzi, Olshanskii & Wang
[10.1002/fld.2267](https://doi.org/10.1002/fld.2267)). That `ν` is a constant kinematic
viscosity, a statement about the Reynolds number and not about a heterogeneous field `η(x)`;
the variable-viscosity evidence is the 2023 paper.

**PCD and LSC.** Both are commutator approximations for the Oseen operator, whose pressure
convection-diffusion operator `F_p = A_p + C_p` carries advection. We have none, so `C_p = 0`,
`F_p = A_p`, and PCD collapses to a pressure Laplacian or mass inverse, measuring 2.7x the
per-apply cost of `:mass` for the same count, with LSC 6x worse. Rudi et al. say it from the
other side: "numerical experiments with `M_u`-BFBT are not presented, because it performs
poorly in the presence of spatially-varying viscosities". No paper we could access reports PCD
or LSC counts for a heterogeneous `η(x)` at contrast 1e6 or above.

**What the measurement added.** The reference notes ended with a ranked list whose first item
was a weight swap, `D = diag(K)`, because Rudi's §2.2 says diag(A)-BFBT "is able to achieve
fast convergence for discretization order k = 2" and ours is Q2. That experiment was run with
the rest of the family, and none cleared the 15 % wall-clock bar
([MG.md](MG.md) §2, §5). `pscale` is worth 1-4 iterations and the
two states disagree on its sign; `D^α` loses 60-240 %; `D = diag(K)` takes `F` from 81 to 79
and `P1` from 65 to 70 and makes the velocity error 1.9x worse (3.28e-6 against 1.69e-6),
outside the established 2.1e-6 class; the two-level correction hits the 500-iteration cap on
every row; the penalty needs 92 iterations at the smallest lift and 500 above it. The change
that paid was not on the list: the exact diagonal of the restricted mass matrix,
`D = diag(Cuᵗ M_u(√η) Cu)`, in place of the shipped lumped `Cuᵗ diag(M_u(√η))`, took `F` from
81 to 71 and, once leaf-local, bought 11.5-14.5 % of the warm solve on `F` and nothing on `P1`;
it shipped as stage 9 ([MG.md](MG.md) §2.9). The literature reports that
distinction only in a figure, which is why the note ranked the wrong knob.

One measurement from that probe closes a whole family. A coarse correction whose size is
`8.5e-5` relative to the fine solve, the smallest of fifteen rows, turns 81 iterations into 500
at `relres` 0.32, because the w-BFBT product amplifies a perturbation of `S1⁻¹` by 1e4-1e6. Any
additive correction to this `S1⁻¹`, even a numerically negligible one, is unavailable. The
deflation literature gives the equivalence results and no count at our contrast; what it
cannot give is a subspace that survives this amplification. The family is closed by
measurement.

**The decision.** The block is a quarter of the iteration, the three-factor form is the only
published family that holds at our contrast, and the second Cholesky solve is the price. What
is left is engineering: cache `S1` and `S2` across Picard and time steps, and loosen the inner
tolerance only with an outer-count gate, because +9 % of outer iterations erases the saving.

## 5. Sparse triangular solves

The serial cost of a triangular solve is `O(nnz(L))`, a dependent walk over the factor:
Davis, *Direct Methods for Sparse Linear Systems*, ch. 3
([10.1137/1.9780898718881.ch3](https://doi.org/10.1137/1.9780898718881.ch3)). Parallelism
exists in the dependency DAG's levels, generated by the elimination tree (Liu 1990,
[10.1137/0611010](https://doi.org/10.1137/0611010)), and level scheduling is Anderson and
Saad ([10.1142/S0129053389000056](https://doi.org/10.1142/S0129053389000056)) and Saltz
([10.1137/0911008](https://doi.org/10.1137/0911008)). The cost model is barriers, one per
level, so the interesting quantity is the work available per level divided by the cost of
exposing it.

The vendor manual makes the same point. cuSPARSE's `csrsv2_solve` warns that "The level
information may not improve the performance" and advises timing both policies and keeping
the faster (cuSPARSE 11.0). The state of the art confirms the framing without overturning
it: GrowLocal reports 1.42x over SpMP and 3.32x over HDagg through a "12.07x reduction in
the number of synchronization barriers" on a 22-core x86 machine
([arXiv:2503.05408](https://arxiv.org/abs/2503.05408)). None of the cited work reports a
serial-relative speedup at a 7k factor.

**Our factor.** `S1` is 7 424 × 7 424 with 198 490 nonzeros, factored once in 0.016 s. One
solve is 0.61 ms standalone and 0.976 / 0.923 ms inside the solver; the pair is 1.84 / 1.90 ms
per iteration on `F`, 13.2 / 11.8 % of the eight-thread iteration and 10.3 / 8.7 % of the warm
solve ([MG_LITERATURE.md](MG_LITERATURE.md) §1). CHOLMOD is already supernodal, and
its solve is built from `dtrsv`, `dgemv`, `dtrsm` and `dgemm`
([`cholmod_solve.c`](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Cholesky/cholmod_solve.c)),
which is BLAS-2 for a single right-hand side; the BLAS-3 routines are for several. It walks
supernodes in order, so the concurrency a level schedule would expose is not there. MUMPS
parallelises its solve over the assembly tree
([10.1137/S0895479899358194](https://doi.org/10.1137/S0895479899358194)), which is
parallelism inside a distributed multifrontal factor and not inside a 7.4k serial one.

**The barrier-to-work ratio.** A level-scheduled forward and backward solve over the same
factor was built: `L` renumbered twice so each level's rows are contiguous, one region phase
per level, one fork/join per solve. It is correct to 5.2e-16 and bitwise thread-independent,
and it is slower by more than a factor of two, 1.128 ms against 0.486 ms at eight threads
(0.43x) and 0.51x at one ([MG.md](MG.md) §3.9). The reason is the factor's
shape. The widest forward level holds 2 578 nonzeros, about 0.9 µs of work, and one region
phase costs 7.7 µs, so the barrier-to-work ratio is 8.6:1; the 1 005 phases would cost 7.7 ms
against a 0.42-0.46 ms exact solve, 18 times the thing they replace. A 2-D factor has O(10²)
levels and the solve is about a millisecond, so microsecond barriers are of the order of the
arithmetic. The earlier inference expected at most 1.2x at eight threads; the 1→8-thread A/B
measured 0.98x and 1.00x, and the full schedule measured 0.43x.

**The inexact route the literature recommends.** The solve reference's rank-1 recommendation
was to replace the exact factorisation by an inner CG at 1e-6 to 1e-2, on the strength of
ASPECT's 1e-6 RHS-relative tolerance, step-56's fixed 21-22 iterations and Burkhart's 0.1.
CG preconditioned by `diag(S1)` or an IC(0) needs 45-75 iterations, because `S1` is a Schur
complement with condition number growing like `h⁻²`; one CG iteration costs 0.11 ms with the
diagonal and 0.30 ms with IC(0) against 0.42-0.46 ms for the exact solve, so the route is
0.30-0.34x. The outer count does not move at 1e-4 through 1e-6, and at 1e-2 the outer solve
breaks: 500 iterations, `relres` 0.83, `|du|/|u|` 1.7. ASPECT and step-56 work because their
block is a mass matrix with an h-independent count. The break-even that made the route look
attractive, 7 % of warm if the outer count holds and +9 % erases it, was never close.

**What the block is worth.** One solve is 6.6 % of the iteration and 5.2 % of the warm solve;
the pair is 10.3 %. Making each solve three times cheaper would be about 7 % of warm if the
outer count held, and that is the entire upside, which is why the decision was hours rather
than weeks: workspace reuse through `cholmod_solve2` with a cached `X/Y/E` (Julia's `F \ b`
calls `cholmod_solve`, which allocates and frees the workspace every call) plus a check of
CHOLMOD's fill-reducing ordering together measure about 2 % of the iteration. Level
scheduling and the one-solve form are rejected, the first on the barrier-to-work ratio and
the second because the one-solve row is the inverse-mass row that fails at contrast.

## 6. Algebraic versus geometric coarsening on adaptive meshes

Clevenger and Heister, 18.5M dofs, Q2/Q1, 48 cores, Tables 3-4
([arXiv:1907.06696](https://arxiv.org/abs/1907.06696)):

| phase | AMG (Trilinos ML) | matrix-free GMG | ratio |
|---|---|---|---|
| setup | 12.6 s | 10.3 s | 1.2x |
| assemble | 32.5 s | 2.9 s | 11.2x |
| solve | 38.6 s | 14.8 s | 2.6x |
| total | 83.7 s | 28.0 s | 3.0x |
| iterations, 18M → 2.2B dofs | 53 → 83 | 27 → 28 | GMG h-independent |

AMG is cheaper per application and its count grows by more than 50 % under adaptive
refinement, while GMG's is constant; GMG's setup is not cheaper. The win is that assembly
disappears and the count does not drift, and for a time-dependent run that does not remesh
every step they report GMG about 4x faster end to end because setup is skipped. Munch et al.
compare local smoothing with global coarsening: global coarsening parallelises better but has
more cells, local smoothing wins in serial, hanging-node constraints cost under 20 % at high
order and more for linear elements, and their production configuration is Chebyshev degree 3,
a single-precision V-cycle and AMG on the coarsest level
([arXiv:2203.12292](https://arxiv.org/abs/2203.12292)).

Forest codes make the same bet at scale. Sampath and Biros solve 8e9 elasticity unknowns on
32 000 processors with an `O(N/n_p log(N/n_p))` 2:1-balanced octree GMG
([10.1137/090747774](https://doi.org/10.1137/090747774)); Sundar et al. use hybrid
geometric-algebraic multigrid on octree forests
([10.1109/SC.2012.91](https://doi.org/10.1109/SC.2012.91)); p4est supplies the forest and the
2:1 balance ([10.1137/100791634](https://doi.org/10.1137/100791634)); May, Brown and Le
Pourhiet's 3-D matrix-free multigrid preconditioner is the same bet at lithospheric scale
([10.1016/j.cma.2015.03.014](https://doi.org/10.1016/j.cma.2015.03.014), cited
bibliographically; its counts were not accessible and none is quoted).

**What this says about a forest mesh.** A balanced quadtree hands you the geometric hierarchy
for free, which is exactly the cost AMG pays at setup and what Clevenger's 11.2x assembly
ratio is about, so the argument for geometric coarsening transfers before any measurement.
What does not transfer is the coarse-space quality. Ours is weak in a measured way: the
rediscretised operator is 2-32x too soft because coarse Gauss points miss the wall layer, and
the coarse correction is 31x, 412x and 842x worse than optimal on the three coarsest levels.
Galerkin removes the scale error exactly and takes `F` from 69 to 61 and `P1` from 65 to 46,
but the space stays weak, with the optimal correction still 1.4-5.4x the input residual on
`P1` ([PERFORMANCE.md](PERFORMANCE.md) §3.4). The missing piece is the coarse space, not the
operator scale.

The 2025 paper that looks directly applicable is Böhm, Kohl, Köstler and Rüde's adaptive
Galerkin coarsening ([arXiv:2511.13109](https://arxiv.org/abs/2511.13109)): Galerkin "applied
locally in regions with large coefficient gradients, while ... direct coarse grid
approximation is used elsewhere", with 1e10 degrees of freedom, viscosity jumps of 1e6 and
105 456 processes, weakly scaling from 1.8e8 to 1.44e10 DoFs. The qualification is in their
assumptions: "we assume that we only Galerkin-coarsen the velocity block `A_L`, as only there
the coefficient enters". The saddle system keeps a block-triangular preconditioner with a
diag(A)-BFBT Schur complement, so AGCA replaces the velocity-block V-cycle and not a
coarsening of the coupled operator. They also report iteration growth at 16 sharp sinkers,
where "a switch from linear to operator-dependent interpolation may be necessary", and a
remark that grid-independent convergence "may require a more sophisticated weighting W in
the Schur complement approximation". The part AGCA adds on top of what we measured is local
Galerkin in the wall region, and our full-Galerkin run bounds it at roughly 1.2-1.4x end to
end. That is the narrowed version §8 prices.

**The setup a cache can and cannot remove.** A per-level hierarchy cache was built and
measured: it keys each level by the hash of its leaf set, reuses 77.5 % of the level slots on
a 30-step `amr_every=5` trajectory, saves a median 56 ms per cycle, and reproduces the
trajectory bit-identically. It buys nothing end to end, 77.9 s against 75.9 s, because the
part it can remove, 0.04-0.19 s of topology, is smaller than the part it cannot, 0.09-0.16 s
of eta sampling, diagonal and coarse factor ([MG.md](MG.md) §3.3). That is
Clevenger's point in miniature: skipping assembly is the win, and the hierarchy construction
that remains is not assembly.

The cadence question has also moved. The old statement was that if AMR changes every step,
setup alone makes GMG uncompetitive at 98k, because 2.8 s of setup sat against a 2.63 s direct
solve. At the current `F`/8 state setup is 0.325 s against a 1.489 s warm MG solve, 21.8 %, so
a rebuild every step is affordable and the cadence stops being the constraint. The pre-tuning
sentence stays in the record as the reason the question was asked, and these numbers are the
reason it is closed.

## 7. Vectorisation and kernels

deal.II's matrix-free infrastructure is the reference design. `MatrixFree` caches the dof
indices and constraints, the 1-D shape data and the mapped Jacobians; `FEEvaluation` evaluates
them by sum factorisation at `O(d²(p+1)^(d+1))` against `O(d(p+1)^{2d})` for a local matrix;
and the loop body is our body, `reinit; read_dof_values; evaluate; [pointwise]; integrate;
distribute_local_to_global`
([group doc](https://dealii.org/current/doxygen/deal.II/group__matrixfree.html),
[step-37](https://dealii.org/current/doxygen/deal.II/step_37.html)). The lane model is
`VectorizedArray<Number, width>`, one SIMD register holding 2 doubles on SSE2, 4 on AVX and 8
on AVX-512, with partial batches masked through `std::bitset<n_lanes>` rather than
scalar-epilogued
([classVectorizedArray](https://dealii.org/current/doxygen/deal.II/classVectorizedArray.html)).
Vectorisation is over cells, not points: a batch is one "cell" in the loop, and its dofs are
made contiguous at setup because index translation at evaluation time "would be absolutely
detrimental to performance".

The honest part is the ceiling. Isolated kernels reach "up to 60 % of arithmetic peak on
Intel Haswell and Broadwell and up to 50 % on Knights Landing", but the full operator
evaluation reaches "only about half that throughput due to memory bandwidth limitations from
loading the input and output vectors, MPI ghost exchange, as well as handling variable
coefficients and the geometry", landing "within 10 % of the available memory bandwidth ...
with the exception of the Cartesian mesh case where the cost of gather operations and MPI
communication are more substantial"
([Kronbichler & Kormann 2019](https://arxiv.org/abs/1711.03590)). The same group reports a Q2
apply about 8x faster than a sparse matvec on Skylake with 14 cores in 2018, and best
throughput per dof at p = 3-6.

**Our measured before and after.** At `SEMQUAD_KBLK = 8` the blocked kernel was 2 149.8 µs at
`F` level 6 against the scalar kernel's 2 187.4 µs, a 1.7 % win, because LLVM unrolled the lane
loop into scalar `vaddsd`/`vmulsd` before the vectoriser saw it: 0 packed adds, 248 scalar adds,
464 scalar multiplies. At 16 lanes it is 835 µs, 2.62x, and at 32 lanes 786-801 µs, 2.72x, with
248 `vaddpd`, 496 `vmulpd`, 24 `vsubpd` and 48 `vgather`; 64 lanes is 890 µs, slower again
([MG.md](MG.md) §11.2). The `@simd ivdep` annotations on the lane loops are
load-bearing at 32 lanes: removing them gives a 1 656 µs level-6 apply and a 12.63 ms V-cycle
against 1 056 µs and 8.86 ms (1.53x / 1.43x, same-session A/B). The eta pre-gather that
followed replaced 24 `vgather`s with a lane table and bought 1.083x on the `F` V-cycle and
1.101x on `P1` ([MG.md](MG.md) §15.4), and the constrained apply fell from the
46 ns/dof of the eight-lane era to 15.0 ns/dof at level 6, 928 µs over 61 964 dofs
([MG.md](MG.md) §15.3). The 46/15 pair quoted in the vectorisation note as a
one-thread/eight-thread scaling is really a before/after pair at one thread; the eight-thread
finest apply is about 600 µs.

The small-level regressions were the other surprise. At 318 and 1 278 dofs the threaded kernel
is 7.4x and 2.3x slower than serial, at 4 930 it breaks even, and at 15 634 it is 1.30x faster,
because `_semquad_td_run!` builds tasks, an atomic counter and a `@sync` barrier per call and
a V-cycle makes about 200 such calls
([MG_LITERATURE.md](MG_LITERATURE.md) §3). The fix is what mature
codes do: partition once, reduce parallelism at the bottom, run below a threshold on the
calling task. PETSc GAMG exposes it as `-pc_gamg_process_eq_limit`, which "reduce[s] the
number of MPI processes used directly on the coarse grids so that there are around limit
equations on each process" ([PCGAMG](https://petsc.org/release/manualpages/PC/PCGAMG/)). Ours
is a per-level dof threshold, and the measured control is that eight threads with a serial
V-cycle takes 5.52 s against 5.80 s at one thread while the threaded cycle takes 3.99 s.

**The Julia tooling, and what is not usable.** `@inbounds` removes bounds checks and changes
no arithmetic, so it is bitwise. `@simd` promises that iterations are independent and may be
reordered, is documented as experimental, and takes the manual's dot product from 1.95 to
17.58 GFlop/s. `@fastmath` "may change numerical results". `@turbo` "assumes that loop
iterations can be reordered", does no bounds checking, and is "a tool for generating efficient
computational kernels, leaving tasks of parallelization and cache efficiency to you"; its gemm
showcase goes 4.89 ms to 0.112 ms against MKL's 0.117 ms on a dense, unit-stride,
register-blockable loop, which is the shape we lack, since our inner operations index
`etav[kofs[b] + kq]` and `ids[idx, li]` and are the gather case it handles worst. More
decisively, `VectorizationBase`, the code generator underneath it, is "deprecated in Julia
1.11", and this project runs 1.12.7 against a `julia = "1.10"` bound, so the toolchain is out
before the gathers matter. Static tuples unroll but do not vectorise: broadcasting over an
`NTuple` becomes scalar code that LLVM's SLP may pack two to four ways at best, so the lane
axis has to be a contiguous array dimension, which in this code it already is.

Two caveats from the same note that the measurements resolved. Aliasing: Julia has no
`restrict`, so `_semquad_contract_blk!` taking `dest` and `tmp` as separate arrays means LLVM
must assume they may overlap, and step-37 records the same blocker. The prototype that wrapped
the read-only operand in `Base.Experimental.Const` inside `@aliasscope` is bitwise identical
and flat, 682.5 to 682.0 µs at one thread and 341.8 to 339.1 µs at eight, with more moves
rather than fewer, because the lane loop is already vectorised at 32 lanes and the kernel is
not load-bound ([MG.md](MG.md) §3.2). Unions in the kernel signature and the
zero-weight padded tail lane remain maintenance hazards.

**What is left at the kernel level.** Mixed precision is the one change the literature supports
with a large number and we have not measured: step-37 runs its multigrid levels in `float`,
Munch et al. run the whole V-cycle in single precision, and Kronbichler and Ljungkvist measure
up to +83 % from a single-precision V-cycle with a double outer correction. Our V-cycle is all
`Float64`; a `Float32` residual, smoother and transfer path with a `Float64` outer FGMRES is
contained but not bitwise, and it is priced at 1.2-1.8x on the V-cycle. The other remaining
item is dof renumbering so a block's 144 indices are contiguous, deal.II's
`DoFInfo::reorder_cells` argument, which touches every vector, constraint and transfer.
Everything cheaper has been taken and measured.

## 8. The four levers that were proposed, and what happened to each

The proposal audited here named four directions, each with a citation: slice-wise loops and
partial assembly from Chang et al.; matrix-free patch smoothers from Wichrowski, with Spies et
al. as the saddle-point companion; adaptive Galerkin coarsening from Böhm et al.; and libCEED
as the partial-assembly reference. A verification pass at HEAD `a1832df` checked all four
against Crossref, arXiv, Semantic Scholar and the web. All four exist, none is invented, and
two were used loosely.

| # | direction | what the source actually says | what we measured or priced | verdict |
|---|---|---|---|---|
| a | slice-wise loops, "PAop 1.26x" | 1.26x is a stage margin in a cumulative ablation at p = 8; PAop as a whole is 15.65x end to end and 54.19x on the kernel | -2.0 % to +1.0 % | rejected |
| b | Braess-Sarazin patch smoother | 6/7/9/9 at p = 2, μ = 1e6; 4.0x an apply per cycle; no timings | not measured; break-even at m < 87 | gated prototype |
| c | adaptive Galerkin coarsening | variable-coefficient Stokes, but the velocity block only | full Galerkin already 69 → 61, 65 → 46 | narrowed, 1.2-1.4x |
| d | libCEED partial assembly | optimal transfers, near-optimal flops at high order | our apply already is element-level PA, 0 B per call | not a lever |

**(a) Slice-wise loops.** Chang, Zhang, Zhang, Yang, Li and Kong
([10.1145/3844725](https://doi.org/10.1145/3844725), arXiv
[2601.08374](https://arxiv.org/abs/2601.08374)) is a real paper with real numbers. Table 7 is
a cumulative ablation at p = 8, 51.17M DoFs, Kunpeng 920, 64 ranks: adding slice-wise loops
after sum factorisation, a Voigt layout and fusion takes total time from 13.00 to 10.33 s,
1.26x, and kernel time from 5.38 to 2.74 s, 1.96x. But PAop as a whole is "15.65x" end to end
and "54.19x" on the kernel against MFEM v4.8, so 1.26x is the last stage's marginal
contribution, not the operator's speedup. That is the first loose reading. The mechanism is
cache capacity at high p: their reference-gradient working array "scales as 8d²(p+2)³ bytes
(about 72 KB at p = 8) and exceeds L1d", and their kernel "stays bandwidth-bound". They place
the CPU sweet spot "near p ≈ 2", and at p = 2 the whole stack is only a "2x" per-element FLOP
reduction, because their baseline is `O((p+1)⁶)` and ours already has sum factorisation.

Ours at p = 2 has no capacity problem to buy: one 32-lane block's live compute working set is
40 864 B, 83.1 % of this CPU's 48 KiB L1d and 47 008 B once the gather and scatter lines are
counted, and those lines are touched once per leaf in leaf order, so no reorder reaches them
([MG.md](MG.md) §16.1). The capacity test says smaller is slower per lane: 16
lanes is 1.05-1.11x slower and 8 lanes 3.02x, because the vectoriser loses the form. The
residency test says a cold cache costs 8.7 % and the penalty does not fall with the working
set, the signature of a compulsory stream rather than of capacity. Both slice forms were
built and measured in process: the lane slice is bitwise and 1.7-1.9x slower at 16 lanes, the
quadrature slice keeps the vector width and measures -2.0 % to +1.0 %, inside the
instrument's 2-4 % spread ([MG.md](MG.md) §16.2-§16.4). Rejection, with the
measurement behind it, and the number that predicted it was in the source: bandwidth-bound at
p = 8 and issue-bound at p = 2 are different regimes.

**(b) Braess-Sarazin.** The contrast claim is real and stronger than expected: at p = 2 and
`μ = 10⁶` it is within one iteration of an exact local solve, and single-patch jumps to `10⁸`
barely move the count. The cost claim is where the transfer fails. Wichrowski reports 4.0x an
apply per V-cycle and reports no timings on principle; Spies et al. timed Braess-Sarazin only
at constant viscosity in 2-D, where the ranking against Vanka reverses between CPU and GPU,
and they flag how much tuning the relaxation needs. Break-even is `m < 87` by the arithmetic
of §3, and the published 6-10 is not the risk. The risk is that no experiment combines a
heterogeneous `η(x)` at 1e6-1e7 with a wall-clock cost on Q2-Q1, the only combination that
prices it here. Hence a two-week prototype behind a gate, the gate being the outer count at
fixed accuracy.

**(c) Adaptive Galerkin coarsening.** Böhm et al. is variable-coefficient Stokes at 1e10
unknowns, 1e6 jumps and 105 456 processes, and the numbers are as advertised. The assumption
that bounds it is theirs verbatim: "we assume that we only Galerkin-coarsen the velocity block
`A_L`, as only there the coefficient enters". It is not a coarsening of the coupled saddle
operator, and a near-null pressure band is not in its scope. We have already measured the
operator-scale part of the same idea: full Galerkin takes `F` 69 → 61 and `P1` 65 → 46 with
the coarse space still weak. So the narrowed lever is local Galerkin on wall-layer coarse
levels plus operator-dependent interpolation, priced at 1.2-1.4x end to end and 2-6
engineer-weeks, and the missing piece it would still not supply is the coarse space itself.

**(d) libCEED.** The library is what it says: "high-order methods, implemented properly with
only partial assembly, require optimal amount of memory transfers ... and near-optimal
FLOPs", with "the bottleneck of performance ... mostly in the memory bandwidth"
([manual](https://libceed.org/en/latest/intro/)). Its BP3 note frames the gain as a
large-working-set, high-order, often GPU effect. Our apply already is element-level partial
assembly, leaf-local tensor contractions, batched, zero allocation, so a port would be a
foreign-function interface and not a new lever; our hanging-node `Cu` products and near-null
pressure handling have no libCEED equivalent. Expected ~1.0x for a multi-month port, and the
verdict is rejection.

**Three more, caught later by measurement.** The four-lever proposal was not the only place a
source was read against the wrong object. Chebyshev at degree 2-4, ASPECT's "strongly
recommended" setting, was applied to an operator whose small end is a near-null pressure band,
and it diverged on both production states (§3). ASPECT's 1e-6 right-hand-side-relative inner
tolerance was applied to `S1`, a Schur complement with `κ ~ h⁻²`, and the measured cost is
0.30-0.34x of the exact solve (§5). Rudi's "diag(A)-BFBT is able to achieve fast convergence
for discretization order k = 2" was our own rank-1 recommendation, and it measured 79 vs 81 on
`F` with 1.9x the velocity error and 70 vs 65 on `P1`, while the change that paid was the one
the paper reports only in a figure, the difference between a lumped and a true restricted
diagonal (§4).

What all five have in common is that the citation was correct and the object it was applied to
was not the object the citation measured. That is the failure mode this volume exists to
catch, and the countermeasure is cheap: state the object in the same sentence as the number.
The mature codes measure mass matrices, high-order bandwidth-bound kernels, constant-viscosity
Oseen operators and velocity blocks. We solve a 2-D variable-viscosity saddle system with a
near-null pressure band at p = 2 and 69k dofs, and almost every transfer has to cross one of
those differences.

## 9. An annotated bibliography

One line per source that changed something. "Number" means we took a figure from it;
"bibliographic" means it is cited for existence or theory only.

**Cost, benchmarks, hardware.**
- HPGMG 1.0, LBNL-6876E, and the README: one V-cycle at 3.6 ns/unknown with its component split, the MEq/s ranking and the <10 work-unit target. Number.
- Clevenger & Heister, arXiv:1907.06696: 30-53 ns/dof per iteration, the AMG-versus-GMG table, and iteration growth under AMR. Number.
- Kronbichler & Ljungkvist, ACM TOPC 2019: +83 % from a single-precision V-cycle with a double correction, 3.4x over HPGMG-FE, and the p = 3-5 optimum. Number.
- Fischer, *Exascale: A User's Perspective*: the production spectral-element profile, pressure 85 % and coarse AMG 34 %. Number.
- Williams, Waterman & Patterson, CACM 2009: the roofline and the arithmetic-intensity vocabulary used throughout. Method.
- PETSc `PCGAMG`: the coarse-grid process limit, the mature form of our per-level serial threshold. Number (semantics).

**Smoothers and multigrid.**
- Adams, Brezina, Hu & Tuminaro, JCP 2003: polynomial smoothing as a matrix-free alternative to Jacobi. Method.
- Jodlbauer, Langer, Wick & Zulehner, arXiv:2205.15770: the degree-versus-cost trade-off and (1,1) as the best reduction per second. Number.
- Munch, Heister, Prieto Saavedra & Kronbichler, arXiv:2203.12292: local smoothing against global coarsening, the hanging-node cost, and the degree-3 single-precision production V-cycle. Number.
- Wichrowski, arXiv:2601.13230: Braess-Sarazin counts at contrast, 4.0x per V-cycle, and no timings. Number.
- Spies, Olson & MacLachlan, IJHPCA 2025: monolithic multigrid against block-triangular on Q2-Q1, at constant viscosity. Number, with that limit.
- Böhm, Kohl, Köstler & Rüde, arXiv:2511.13109: adaptive Galerkin coarsening at 1e10 dofs, velocity block only. Number, with that limit.
- Kohl & Rüde, arXiv:2010.13513: Uzawa-type monolithic multigrid reaching the <10 work-unit budget. Number.
- May, Brown & Le Pourhiet, CMAME 2015; Gmeiner et al., SISC 2015: the 3-D matrix-free and HHG Stokes precedents. Bibliographic.
- Sampath & Biros, SISC 2010; Sundar et al., SC'12; Burstedde, Wilcox & Ghattas, SISC 2011: octree-forest GMG at 8e9 unknowns, hybrid geometric-algebraic multigrid, and the p4est forest with its 2:1 balance. Number (scaling claims only).

**The Schur block.**
- Rudi, Stadler & Ghattas, SISC 2017: the contrast study, Tables 1-3, and the 29-60 band that bounds every weight variant. Number. The paper that validates the shipped choice.
- Burkhart, Kohl, Wohlmuth & Zawallich, GEM 2025: V-cycle BFBT, the 0.1 inner tolerance, and the ranking of six approximations. Number (counts in a figure) and method.
- Shih, Stadler & Wechsung, SISC 2023: the `γ` table for augmented Lagrangian and the smoother/transfer price in Table 2. Number.
- Benzi & Olshanskii, SISC 2006; Benzi, Olshanskii & Wang, IJNMF 2011: ideal versus modified AL, and what "robust in viscosity" means for a constant `ν`. Number, with that limit.
- Kay, Loghin & Wathen, SISC 2002 (PCD); Elman et al., SISC 2006 (LSC): the commutator approximations that need an advection term we do not have. Number (constant-viscosity counts).
- Li & Widlund, SINUM 2006; Frank & Vuik, SISC 2001; Nabben & Vuik, NLAA 2008: the two-level and deflation theory behind the coarse-correction family. Theory only.
- Kronbichler, Heister & Bangerth, GJI 2012: ASPECT's block preconditioner and viscosity weighting. Method.

**Solvers and kernels.**
- deal.II step-56: the block-triangular preconditioner with a mass-matrix `S̃`, CG+ILU, and the fixed 21-22 inner iterations. Number.
- ASPECT at `340c0d018`: the tolerance hierarchy, the cheap and expensive phases, and the BFBT note. Number.
- Trilinos Ifpack2, ML and MueLu, through deal.II's `TrilinosWrappers::PreconditionAMG`: the approximate inner solvers a mature stack actually offers. Method.
- Saad, SISC 1993 (FGMRES); Simoncini & Szyld, SISC 2003 (inexact Krylov): the licence to solve the inner block inexactly. Theory.
- Davis, *Direct Methods*, ch. 3-4; Liu, SIMAX 1990; Anderson & Saad, IJHSC 1989; Saltz, SISC 1990; GrowLocal, arXiv:2503.05408; cuSPARSE 11.0: the triangular-solve cost model, the elimination tree, level scheduling, and barriers as its real cost. Theory and number.
- CHOLMOD (Chen, Davis, Hager & Rajamanickam, TOMS 2008) and its source; MUMPS (Amestoy, Duff, L'Excellent & Koster, SIMAX 2001): the supernodal BLAS-2 solve with a serial supernode walk, and assembly-tree parallelism in a distributed factor. Method.
- deal.II matrix-free group doc, `VectorizedArray`, `FEEvaluationBase`, step-37; Kronbichler & Kormann, Computers & Fluids 2012 and ACM TOMS 2019; Orszag, JCP 1980; Deville, Fischer & Mund, CUP 2002: sum factorisation, the lane model, vectorise-over-cells, and the 60 %-of-peak isolated-kernel ceiling. Number and method.
- libCEED, JOSS 2021: partial assembly as the reference design, and the L3-working-set argument. Number (qualitative) and method.
- Chang et al., ACM TACO 2026: slice-wise loops at p = 8, 1.26x as a stage margin, and the 72 KB working-set mechanism. Number, read carefully.
- Julia *Performance Tips* and *multi-threading* manuals; LoopVectorization.jl; VectorizationBase.jl: `@inbounds`, `@simd`, `@fastmath`, `@turbo`, the gather limitation, and the 1.11 deprecation that closes the route on 1.12. Number and constraint.

Two absences are worth recording. No accessible paper reports PCD or LSC iteration counts for a
heterogeneous viscosity at contrast 1e6 or above, and none reports a wall-clock cost for
Braess-Sarazin at that contrast on Q2-Q1. Those gaps are why §§3-4 end in "not measured here"
rather than in a number, and stating them is more useful than filling them with an inference.
