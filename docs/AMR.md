# AMR

A convection run at a fixed leaf budget has one obvious place to spend it: the
top boundary layer. A uniform grid fine enough for a thinning lid is
unaffordable in the interior, so the mesh moves leaves from the quiet middle of
the box to the wall. This file explains how that works in `julia/src`: the
quadtree and its flat face table, the gradient-jump indicator and the Dörfler
marking, the `:balanced` policy and the defaults it now owns, the 2:1 legality
gate, the local rebalance and the proof that it equals the global one, the
load-bearing quirk, and the hot-path ledger that made the pass cheap enough to
leave on.

Two changes on 2026-11-13 moved the ground under the old text, and both are now
part of the volume. The realistic card began to default to `balanced` with the
gate on, `symmetry = :none`, `coarsen_frac = 0.05`, `coarsen_theta_tol = 1e-2`
and `wall_min_level = 5` (`715e272`), and the library driver
`run_semquad_simulation` flipped its own `amr_policy` default from `:classic`
to `:balanced` (`4584e85`). The survey that priced the flip,
`AMR.md` (405 lines), is folded in at §4 and §5. It
stays in the tree as the raw measurement log; every number it produced that
decides something is quoted here at the place where it decides it.

Numbers are the production card unless a line says otherwise: a `[0, 2.5] x
[0, 1]` box with 5 x 2 base cells, square leaves, `p = 2`, `max_level = 7`,
`max_leaves = 10000`, `wall_min_level = 5`, `wall_strip = 0.1`,
`refine_frac = 0.2`, `coarsen_frac = 0.05`, `amr_every = 5`. The operating state
is the E checkpoint at step 18800 (9994 leaves, `N = 42177`) and the step-19000
state (10000 leaves) that the policy lives at. The audit numbers are WSL Ubuntu
24.04, Julia 1.12.7, one thread; the local-balance probes ran under Windows with
Julia 1.10 and only their ratios are used. Code references are at `fabbedd`.

## 1. The tree, the leaf set, and the flat face table

`Quadtree` is a forest of base cells, not a pointer tree. It holds the maximum
level, the physical width `Lx`, the base-cell counts `nx0` and `ny0`, a
`Set{NTuple{3,Int}}` of leaves, and a `QuadFaceCache` that holds the flat table
and the face-pair list. There are no per-cell objects and no stable ordering;
every pass that needs an order sorts the set once, and every pass that only
needs membership pays a hash lookup.

A leaf is the triple `(l, ix, iy)`. Here `l` is its level, `0 <= l <= maxlevel`;
`ix` is its column index, in `0:nx0*2^l - 1`; `iy` is its row index, in
`0:ny0*2^l - 1`. Its physical box is `[ix*hx, (ix+1)*hx] x [iy*hy, (iy+1)*hy]`,
with `hx = Lx/(nx0*2^l)` and `hy = 1/(ny0*2^l)`. The children in x are `2ix` and
`2ix + 1`, so one split appends four leaves `(l+1, 2ix+dx, 2iy+dy)` for `dx, dy`
in `{0, 1}`. The parent helper is the inverse,
`_pq(lf) = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)`. The tree is complete: a cell
exists exactly when no ancestor is a leaf. That is what lets the face queries be
written with index arithmetic instead of a node hierarchy.

Two face-adjacent leaves may differ by at most one level. The coarser side is
always the one at fault, so the repair rule is one-sided: split a leaf when a
face neighbour is at `level + 2` or finer. `balance!` applies that rule as a
worklist and returns the unique least 2:1 refinement of its input.
`max_face_level_jump(qt)` returns the worst face difference, and the invariant
is `<= 1`. Under `symmetry=:x`, `balance!` splits the mirror partner with every
split so the forest stays closed.

### 1.1 The mirror index, and the bug that looked like arithmetic

The x-mirror is the reflection `x -> Lx - x`. At level `l` the grid has
`n = nx0 * 2^l` columns, indexed `0` through `n - 1`. Reflection reverses the
column order and leaves the row alone, so the partner of `(l, ix, iy)` is
`(l, nx0*2^l - 1 - ix, iy)`. The `nx0` belongs there: it is `cells_x`, the
base-cell count, and each base cell contributes `2^l` columns at level `l`. The
full row has `nx0*2^l` columns, and the last index is one less. The helper is
`nside_x(qt, l) = nx0 * (1 << l)`, and the implementation is exactly
`(lf[1], nside_x(qt, lf[1]) - 1 - lf[2], lf[3])`.

The old formula was `2^l - 1 - ix`. That is the `nx0 = 1` specialization, and on
the unit square it is correct. On a forest with more than one base cell it is
wrong in two ways. For the 5 x 2 card at level 1 it gives `1 - ix`, so the
mirror of `(1, 0, 0)` is `(1, 1, 0)`, a real leaf in the second base column; the
geometric mirror is `(1, 9, 0)` in the fifth. For most other indices it goes
negative, for example `(6, 200, 0)` maps to `63 - 200`. Every call site guarded
the partner with `ml in qt.leaves`, so a negative or absent key was simply
skipped and a wrong but existing key was used. No exception was thrown and no
counter moved. The commit that fixed it described the effect as the mirror
silently no-oping. The existing-but-wrong key is the subtle case: on a symmetric
wide field the wrong partner is a leaf too, so a membership test cannot tell
(inference from the geometry; the commit measured the resulting asymmetry).
`G204`, 217 assertions, now pins the involution, the index range, and the
`cells_x = 1, 2, 5` cases.

`mirror_defect(qt)` counts leaves whose partner is not a leaf; zero is the
invariant. `mirror_close!` repairs a damaged tree by refining the coarser side of
each mismatch, and `_mirror_protect` closes the protect set under the mirror
because a set derived from the temperature field is only approximately
symmetric. A one-sided merge leaves five non-mirrored leaves: the new parent plus
the four children of the still-split mirror parent. That is why the old damage
counts moved in fives, 120 at step 6130 and 125 in the C seed.

### 1.2 The flat face table

The face query used to be a recursive descent from the base cells, returning a
fresh `Vector` per `(leaf, direction)` call. Under `--track-allocation=user` that
was 223.94 MB in `_collect_face!` plus 74.64 MB in `neighbors` per four AMR
cycles, 42 % of the pass. The mature-code answer is a per-tree flat array:
deal.II keeps `TriaLevel::neighbors` as a per-level array of `(level, index)`
pairs, and p4est keeps `quad_to_quad` and `quad_to_face`. We keep the same
shape, keyed by a dense leaf id.

`QuadFaceTable` holds `ids`, a `Dict{NTuple{3,Int},Int32}` from live leaf to
dense id `1:n`; `keys`, id to leaf key; `lev`, id to `Int8` level; `gen` and
`cur`, where `gen[id] == cur` means the four lists of `id` are current; `off`,
`slen`, `scap`, the per-slot offset, length, and capacity into `pool`; `pool`,
the neighbour ids as one `Int32` block per slot; and the bookkeeping fields
`used`, `freeids`, `live`, `holes`, `scratch`, and `rebuilds`. For leaf id `i`
and direction `d` in `1:4`, the slot is `4*(i - 1) + d`, and the neighbours are
`pool[off[slot] + k]` for `k = 0:slen[slot]-1`. The directions are in `FACE_DIRS`
order: `:xplus, :xminus, :yplus, :yminus`. A query returns ids, not keys, so a
2:1 test is `lev[id] >= lev[own] + 2` and never touches a hash.

A face list is a pure function of the leaf set, computed from topology rather
than float coordinates. The cell one step away from the face is in one of three
states. It is a leaf at the same level, and the list has one entry. It is
covered by a coarser leaf, found by the ancestor walk, and that leaf covers the
whole face. It is an internal node, and the list is the leaves of its subtree
that touch the face, found by descending only the near-edge children. In a
complete quadtree "no ancestor is a leaf" is exactly "the equal-level cell is
internal", so one ancestor walk decides the first two cases. The result is the
same list, in the same order (increasing coordinate along the face), as the
historical `_collect_face!` descent. `G216` checks 38 380 queries on random,
unbalanced, wall-graded, and multi-base-cell trees against that scan.

### 1.3 The incremental hooks

A table rebuilt on every mutation would be a per-query win and a per-mutation
loss. `balance!` mutates between the queries of one wave, so the table is
maintained incrementally. `refine!` calls `_faces_drop_leaf!` before the leaf
leaves the set, deletes it, inserts the four children through `_faces_add_leaf!`,
and bumps the version. `coarsen!` drops the four children, inserts the parent,
and bumps the version. `_faces_drop_leaf!` marks stale the dropped leaf and
every leaf whose list contains it: a split replaces one face by two sub-faces, a
merge does the reverse. It also clears `live[id]`, removes the key from `ids`,
and pushes the id onto `freeids`. `_faces_add_leaf!` reuses a freed id when one
is available, keeps the pool block, and gives the new entry `gen = 0`; its lists
are computed on the first query. `_touch_faces!` bumps the stamp and drops
`face_pairs`. `_invalidate_faces!` is the sledgehammer for a wholesale leaf-set
replacement (a checkpoint load or a test tree): it also sets `tab.cur = 0`, so
the next query rebuilds the whole table.

The measured invalidation is small. On a 617-leaf tree one `refine!` invalidates
8 leaves, the worst single mutation 13, and a 60-mutation sequence on a
3 136-leaf tree performed zero full rebuilds. A rebuild happens on the first
query of a tree, after `_invalidate_faces!`, or when the pool's dead space
exceeds half of it. The table stays populated between queries: at 10 000 leaves
it is at most about 3 MB, and the lazy clear keeps a mutation from paying for a
10 000-entry `Dict` that the next query may not need.

`face_pairs(qt)` is the per-version companion list. Each `FacePair` is
`(leafL, leafR, dir, offset, len)`: two leaves, the direction, and the parameter
interval of the sub-face they share. It is dropped by every mutation and rebuilt
lazily. Its element type is the concrete `FacePair` alias, not
`Vector{NamedTuple}`; §9 has the allocation history.

## 2. Marking: the indicator, the fractions, and the wall floor

### 2.1 The gradient-jump indicator

The indicator is per leaf, not per face. For a leaf `K` with normal element size
`h_K`,

    eta_K = (h_K / 2) * sum over faces F of K of integral_F [grad(theta) . n]^2 ds.

Here `theta` is the temperature perturbation, `n` is the face normal, and the
bracket is the jump between the two traces. The `h_K / 2` factor turns an
integral of a squared derivative jump into an error estimate. In the SEM
implementation the trace is evaluated at the fine-side GLL points. For an
interior face between `L` and `R` the quadrature sum is
`J = sum_q w_q * (dL_q - dR_q)^2`, where `w_q` are the face weights and `dL_q`,
`dR_q` are the normal derivatives on the two sides. The face contributes
`0.5*hL*J` to `eta[L]` and `0.5*hR*J` to `eta[R]`, with `hL`, `hR` the normal
element sizes: `hx` for a vertical face and `hy` for a horizontal one. On the top
and bottom boundaries the one-sided normal derivative is the only trace, so the
same formula runs with one side removed. The left and right walls are skipped:
they are insulating, the natural Neumann condition makes a one-sided x
derivative meaningless as a jump, and including it would refine the side columns
forever. On the E tree the total error is `E = 1.797e+01` with the wall strip
zeroed, as the classic policy does, and `4.564e+01` raw.

### 2.2 Dörfler marking by error fraction

Marking is by fraction of the total error, not by fraction of the cells. Sort the
leaves by `eta` descending. Walk that order and accumulate `eta` until the sum
first reaches `refine_frac * E`; the `eta` at that leaf is `tau_r`. Sort
ascending, accumulate until `coarsen_frac * E`; that leaf's `eta` is `tau_c`. The
production card uses `0.2` and `0.05` for both policies. The `:classic`
`amr_cycle!` defaults are `0.3` and `0.0` (`julia/src/dgquad.jl:390`), a
historical difference worth remembering when a run is compared with an old one.

The reason for an error fraction is visible in the distribution. On the E tree
5 % of the error lives in 5513 leaves, 55.2 % of the cells, so the low end is
very flat. A cumulative threshold can free thousands of slots without touching
the important part of the field, while a fixed cell fraction would have to guess
how many cells that is. In this marking mode `refine_frac` is an error fraction;
the cell count is an output.

One cycle's refinement demand is the Dörfler prefix: every leaf with
`eta >= tau_r`, `level < max_level`, and not in the `no_refine` cooldown. Each
such leaf adds `per = 3` leaves when split, or `6` under `symmetry=:x` because
the mirror partner is split too. The sum of those additions is `want`, used by
the budget exchange in §3.4. The classic path stops the same walk when
`length(qt.leaves) + 10 > max_leaves` (`julia/src/dgquad.jl:424`); that guard is
the subject of §3.5.

### 2.3 The wall-strip minimum level

The wall rule is deliberately an error criterion's complement. The boundary
layer is the quantity the production card exists to resolve, and its thickness
`delta_top` falls while the run proceeds, so the mesh needs a floor that does
not depend on today's indicator. `refine_wall_to_min!`
(`julia/src/semquadconvection.jl:38`, called at `:272` after every AMR cycle)
refines every leaf whose center lies within `wall_strip` of either wall, or that
is in the first or last leaf row, until it reaches `wall_min_level`, and
balances after each level. On the production card that is `min_level = 5`,
`strip = 0.1`. The pass runs after every AMR cycle, for both policies.

The `:balanced` cycle turns that into a merge constraint through
`l_min(lf) = wall_min_level` inside the strip and zero elsewhere
(`julia/src/amr_balanced.jl:591`). A quartet may be merged only when every child
has `level > l_min(child)`. A child at level 5 in the strip therefore cannot be
merged, which preserves the floor, while a child at level 6 or 7 can be merged
down to 5. The floor is a minimum-refinement function in the sense of ASPECT's
`minimum_refinement_function`, restricted to the strip because
`refine_wall_to_min!` already does the actual refinement. The gate only has to
stop the merge that the wall pass would immediately undo.

The classic policy does something much blunter with the same geometry. With
`wall_min_level > 0` it zeroes the indicator of every leaf in the strip and puts
that leaf in the protect set (`julia/src/semquadconvection.jl:138-152`). That is
2040 leaves, 20.4 % of the tree, carrying 60.6 % of the raw indicator. It also
makes the coarsening walk start with 2040 zero-eta leaves, 39.5 % of its 5171
steps. The balanced policy keeps only the first and last leaf rows in the protect
set, 320 leaves, 3.2 %, and lets the strip's indicator participate in marking.
The floor is handled by `l_min`, the wall rows by protection, and the rest of the
strip is allowed to refine and coarsen above the floor.

## 3. The balanced policy, and the defaults it now owns

The policy is one rule with several parts: the error criterion decides the mesh,
and the leaf budget is a target paid for by coarsening, never a reason to stop
deciding. The entry point is `amr_cycle_balanced!`
(`julia/src/amr_balanced.jl:540`); the driver selects it with
`amr_policy = :balanced`.

The policy is selected at four call sites, and they do not share a default,
which is the one thing to know before reading §4. `case_realistic_2d.jl` is the
production card and exposes every knob as a `CASE_*` variable;
`run_semquad_simulation` is the library driver and owns the kwarg defaults;
`semquad_amr_passes!` is the unit-level primitive and keeps
`amr_policy = :classic` (`julia/src/semquadconvection.jl:108`), so `t_g55`,
`t_g173`'s merge path and `t_g208`'s tree pins still test the classic primitive;
`amr_cycle_balanced!` holds one number neither caller exposes, the deadband
ratio `hysteresis = 0.5` (`julia/src/amr_balanced.jl:543`).

The two columns are not the same configuration, and the difference is the point.
A bare `run_semquad_simulation(qt, pr, theta0; amr_every = 5)` now runs
`:balanced`, with `coarsen_frac = 0.0`, `wall_min_level = 0` and no leaf ceiling.
The policy is on; the production numbers are not. The card closes the gap at
`:469-478`, where it passes `refine_frac = 0.2`, `coarsen_frac = COARSEN_FRAC`,
`coarsen_theta_tol = COARSEN_TOL`, the two wall keys, `amr_policy = AMR_POLICY`
and `amr_coarsen_gate = AMR_GATE == 1`. A caller that wants the card's behaviour
must pass the same eight keys, and the checkpoint guard (§5) will then hold it
to them.

### 3.1 The defaults, knob by knob, and where each came from

| knob | production card | library driver | provenance |
|---|---|---|---|
| `amr_policy` | `balanced` (`:211`) | `balanced` (`:492`) | card `715e272`, driver `4584e85` |
| `amr_coarsen_gate` | `1` (`:234`) | `true` (`:494`) | gate added `98554dc`, card default raised `715e272` |
| `amr_hysteresis` | `1` (`:212`) | `1` (`:493`) | cooldown length in AMR cycles |
| `coarsen_frac` | `0.05` (`:188`) | `0.0` (`:487`) | card is the production value |
| `coarsen_theta_tol` | `1e-2` (`:189`) | `Inf` (`:496`) | classic only, inert under `:balanced` (`:153`) |
| `wall_min_level` | `5` (`:201`) | `0` (`:490`) | card raised from 0 in `715e272` |
| `wall_strip` | `0.1` (`:202`) | `0.1` (`:490`) | same both |
| `symmetry` | `none` (`:225`) | `none` (`:491`) | card moved from `x` in `715e272` |
| `refine_frac` | `0.2` (`:474`) | `0.2` (`:484`) | same both |
| `marking` | `:error_fraction` (`:473`) | `:global` (`:485`) | card is the production value |
| `amr_every` | `5` (`:180`) | `0` (`:482`) | card |
| `max_leaves` | `10000` (`:147`) | `typemax(Int)` (`:488`) | card default `a442712` |
| `max_level` | `7` (`:146`) | `typemax(Int)` (`:489`) | card |
| `p` | `2` (`:148`) | `2` (`:481`) | same both |
| `hysteresis` | `0.5` (`amr_balanced.jl:543`) | `0.5` | deadband ratio, not exposed |

The card's policy pair was `classic / gate 0 / symmetry x / wall_min 0` before
`715e272`, which is why the four rows that changed there are exactly the four
that E's trajectory depends on. `coarsen_frac = 0.05` and
`coarsen_theta_tol = 1e-2` are older still: the card's own comment at `:181-187`
records the historical A/B pair as `0.005 / 1e-3`, and `0.05 / 1e-2` is the
production choice that followed it. `:classic` never reads `coarsen_theta_tol`
unless `coarsen_frac > 0`, and `:balanced` never reads it at all
(`julia/src/semquadconvection.jl:153`), so on the card today that row is stored,
forwarded, guarded and inert.

`:classic` stays selectable, at three levels, and it is the only tool that
reproduces the E/F/G history. The driver takes `amr_policy = :classic`; the card
takes `CASE_AMR=classic`; the unit helper is already classic by default. E is
the classic arm, and the full command line that reproduces it pins the other two
keys that the card moved in `715e272`:

    CASE_AMR=classic CASE_AMR_GATE=0 CASE_SYMMETRY=x \
    CASE_COARSEN=0.05 CASE_THETATOL=1e-2 CASE_WALLMIN=5 \
        julia --project=julia julia/scripts/case_realistic_2d.jl spinup

F and G are balanced arms, and their checkpoints record that. A G restart
therefore has to forward `CASE_AMR=balanced CASE_AMR_GATE=1` and, for the late
`resume.h5`, the cooldown datasets are in the file, so the restart is a genuine
continuation. §5 states that rule and the guard's blind spot.

### 3.2 Thresholds and the deadband

From §2.2, `tau_r` is the refine threshold and `tau_c` the coarsen threshold. The
policy adds `tau_h = min(tau_c, hysteresis * tau_r)` and
`deadband = hysteresis * tau_r < tau_c` (`julia/src/amr_balanced.jl:620-621`).
`hysteresis` defaults to `0.5`, so `tau_h = min(tau_c, 0.5*tau_r)`. The `min` is
the safe direction: a quartet is merged only when its worst child is at or below
`tau_h`, and `tau_h` never exceeds either threshold. When `0.5*tau_r < tau_c`
the bands overlap and the policy reports `deadband = true`; the gap between
`tau_h` and `tau_r` is a no-man's land where neither a merge nor a refinement is
triggered. The uniform check is clean: if every `eta` is equal, `tau_r = tau_c`,
`deadband = true`, `tau_h = 0.5*tau_r`, no quartet is below `tau_h`, and the
cycle coarsens nothing. `G210` pins that.

This is our knob, not ASPECT's. ASPECT has no hysteresis, no deadband, and no
per-cell memory; it relies on the default ten-step cadence and on complete
sibling coarsening. The closest mature analogue is deal.II's two flag sets and
the `coarsening_allowed` gate, which are geometric, not temporal.

### 3.3 What may be refined and merged

A leaf is refined when `eta >= tau_r`, `level < max_level`, and it is not in the
`no_refine` cooldown. A quartet is merged when all four children are leaves, none
is protected, every child has `level > l_min`, no child is in the `no_coarsen`
cooldown, and `max(eta_child) <= tau_h`. The quartet test is the important
change from the classic path. The decision uses the worst child, not one child,
and a rejected quartet is not charged to the coarsening budget. The mirror
quartet must pass the same test under `symmetry=:x`, and the two are merged
atomically by `_coarsen_pair!`. A quartet containing a child at or above `tau_r`
is never coarsened, which is deal.II's rule that a refine-flagged cell is not
coarsened in the same pass. A merge removes three leaves, or six for a mirror
pair. A refused quartet removes none and costs nothing.

### 3.4 The budget exchange

The policy needs a number of merges before it can refine. Let `n0` be the leaf
count at cycle entry, `want` the leaves the Dörfler prefix would add, and
`capcells = round(coarsen_frac*n0)`, the per-cycle cell cap (500 leaves at
10 000). Then `deficit = max(0, n0 + want - max_leaves)` and
`need = min(deficit, capcells)` (`julia/src/amr_balanced.jl:637-641`). When
`deficit` is zero the tree has room and the policy only refines, exactly like the
classic path on a tree below the budget. This is the demand-driven supply:
coarsening is not "5 % of the error every cycle". The measured reason is in
§3.5: one unconditional 5 % pass drops the E tree from 9994 to 6688 leaves, a
net -3306, for 1.5 % of the indicator. That is a one-way coarsening, not a trade.

When `need > 0` the plan selects the legal quartets whose summed indicator is
lowest, subject to the gate of §6, and executes them. The plan is sized to
`need`, so the overshoot is at most one batch. Then the refinement loop runs. If
`length(qt.leaves) + per > max_leaves`, the loop does not break. It first takes
the next planned quartet; if the plan is exhausted, it calls `_fund_live!`
(`julia/src/amr_balanced.jl:251`), which scans the indicator order for the first
complete quartet whose merge the post-merge rebalance will not undo. A
successful merge adds three slots (six under `:x`) and the loop continues. The
loop stops only when no eligible quartet is left, and then sets
`budget_limited = true`. A cycle that wants a refinement cannot silently report
`refined = 0` without that flag. If the tree is still above `max_leaves` after
the cycle, the caller's hard budget pass runs as before and `budget_infeasible`
is recorded; the leaf-count invariant stays in force.

The old guard was one line: `length(qt.leaves) + 10 > max_leaves && break`
(`julia/src/dgquad.jl:424`). At 9994 leaves against 10000 that is
`10004 > 10000`, true before the first candidate, so the refinement loop
performed zero iterations. The driver reinforced it: on the classic arm it
passes `refine_frac = 0` when the tree is at or above `max_leaves`
(`julia/src/semquadconvection.jl:1650-1651`), and the E tree sits just below
that, so its freeze was the guard's doing. Since the flip the same line reads
`rf = balanced ? refine_frac : (...)`, so the balanced arm never zeroes the
fraction; the exchange is what frees the slots. The `+10` was slack for six
leaves of a mirror pair plus balancing, but at this occupancy there was room for
one pair at most. The tree could not trade; it could only stop.

### 3.5 What the old policy did, with numbers

The production E run is the clean measurement. It is pinned at 9994 of 10000
leaves for 3720 AMR cycles; 723 cycles changed anything, 19.4 %; the last
structural change was at step 17225. The three mechanisms are separable.

The absolute theta guard, `coarsen_theta_tol = 1e-2`, compares the per-leaf range
`max(theta[leaf]) - min(theta[leaf])`. In a convecting field that range is never
small. The minimum over 9994 leaves is 4.500e-03, the median 3.092e-02, the
maximum 3.003e-01. 9892 leaves, 99.0 %, exceed the tolerance and enter the
protect set; only 102 are below it, and no leaf has a range below 1e-3 or exactly
zero. The mirror closure adds 69 partners, so the effective set is 9980 of 9994,
99.9 %, and only 14 leaves are coarsenable. A quartet needs four of them, and
there are none. The guard was intended as a representation-error check; on this
field it is a blanket freeze.

The accounting compounds it. The classic coarsen loop walks the leaves ascending
and, for every walked leaf, adds `_coarsen_pair!`'s return to `nc` and adds
`max(eta[lf], 0.0)` to `cum_c` whether the merge happened or not
(`julia/src/dgquad.jl:433-445`). A protected leaf spends the coarsening budget
without freeing a slot, and a quartet is decided by one child's `eta`. On the E
tree, with a target of `0.05*total`, 94.8 % of the budget is charged to protected
leaves, 5.2 % to incomplete quartets, and 0.0 % to mergeable free quartets. The
loop walks 5171 leaves, 39.5 % of them zero-eta wall strip, and makes zero
merges. The accounting is per leaf while the operation is per quartet.

The wall strip is the third piece. With `wall_strip = 0.1`, the strip is 2040
leaves, 20.4 % of the tree; the first and last rows alone are 320, 3.2 %. The
strip carries 60.6 % of the raw indicator, and the classic path zeroes it. The
same strip is protected from coarsening and removed from refinement.

The counterfactual shows the tree does want to trade. With the wall rows only as
the protect set and no theta guard, the from-below walk merges 1110 quartets,
taking 9994 to 6664 leaves and 6688 after `balance!` and `mirror_close!`, a net
-3306, while removing only 1.5 % of the indicator. The low end is flat enough to
free thousands of slots; it needs a per-cycle cap so the operation stays a trade.
Dörfler from above is small and sharp at the same time: 82 leaves on the card
indicator, 157 raw.

### 3.6 Cooldown hysteresis

`AmrCooldown` is a pair of dictionaries, `no_refine` and `no_coarsen`, plus a
cycle counter (`julia/src/amr_balanced.jl:59-70`). A merged quartet's parent goes
into `no_refine`; the four children of a refined leaf go into `no_coarsen`. A
value of `n` suppresses the operation for `n` more cycles. `_age_cooldown!` runs
at the start of each cycle, deletes keys for leaves that are gone, decrements, and
drops keys that go negative. The default is `amr_hysteresis = 1`, so a merge in
cycle `k` suppresses a refinement in cycle `k+1` and expires at the start of
`k+2`.

The mechanism is specific. For a smooth field the indicator falls steeply with
`h`; the policy note records `h^6`, so a merged parent carries about 16 times the
summed indicator of the four children it replaced (one child's indicator scales
by `2^6`, the quartet sum by `2^6/4`). Without the cooldown the two passes would
undo each other every other cycle. `G210` measures the effect directly: with
`amr_hysteresis = 1`, a quartet merged in cycle 1 is not re-refined in cycle 2;
with `amr_hysteresis = 0` it is, 12 of 20 parents.

The cooldown is state, not configuration. It is written to the checkpoint as two
integer datasets plus a cycle attribute and restored on resume
(`julia/src/semquad_checkpoint.jl:92-100`). `BUG_LEDGER.md` §10.5 measures the
result: a 640-leaf synthetic case and the realistic card at a 1000-leaf budget
both resume bitwise, and stripping the cooldown from the same checkpoint changes
the trajectory. Old checkpoints load an empty cooldown, which is exactly the
historical restart behaviour, and it is also why a policy switch has no
hysteresis memory on its first cycle (§5).

### 3.7 What it buys

The 200-step A/B from step 18800 is the policy's report card. Both arms use the
same card and differ only in `CASE_AMR`. The classic arm changes the mesh in 0 of
40 cycles; the balanced arm changes it in 40 of 40, refines 1390 leaves and
merges 1388 quartets, and settles at 10000 leaves. The top wall row moves from L5
to L6, `h/delta_top` goes from 0.213 to 0.106, and `Nu_top` moves by 0.065 %,
inside the 1 % agreement with ASPECT. The run costs 4.3 % more per step, of which
0.25 % is the policy; the rest is the finer wall, a 0.09 % larger `N`, and a
rougher viscosity field (twice as many nodes on the `eta_max = 1e4` clamp).

The trade saturates after about eight cycles. On the final balanced tree the
thresholds are `tau_r = 1.0907e-2` and `tau_c = tau_h = 1.6115e-3`; 2932 of
10000 leaves, 29.3 %, are below `tau_h`, and 481 complete quartets have
`max(eta_child) <= tau_h`. Of those, 80 touch a protected wall row and 160 have a
child at or below the wall floor, leaving 241. Under `symmetry=:x` only four of
the 241 have a mirror quartet that also passes the gate, because the temperature
field is not mirror-symmetric even though the tree is. The pair rule is the
binding constraint by a factor of 60, not the budget and not the criterion. A run
with `symmetry=:none` keeps 22.2 leaves per cycle against 7.2 and reaches
`h/delta_top = 0.053` at the same budget; that is the production choice, with
`:x` kept as the test invariant. The quartet gate itself is second order:
replacing `max(eta) <= tau_h` by `sum(eta) <= 4*tau_h` admits 590 instead of 481,
23 % more.

The A/B was measured on the direct saddle, which is what the card ran at the
time (`linear = :direct`). Since `2453f16` the library default is `linear =
:auto` and the card passes `:auto` explicitly (`julia/scripts/case_realistic_2d.jl:274`),
so the per-step cost that the 4.3 % refers to is now the smaller MG one. The
policy's own share, 0.07 s/cycle and 0.014 s/step, is unchanged; only the
denominator moved.

## 4. The default flip

`amr_policy` was the last default that could change a trajectory without a
caller asking for it. The flip was staged so that no regression pin could move
silently: Stage 0 pinned the callers (`5ddd50f`), Stage 1 flipped the driver
kwarg and added the gate that pins the new default (`4584e85`), Stage 2
re-baselined the one canonical baseline that runs AMR (`2b016a2`), Stage 3
closed the two adjacent gaps (`c7e0f55`). All four are dated 2026-11-13. The
survey `AMR.md` holds the reasoning and the
line-by-line dependency map; this section holds what the four commits did and
what the numbers became.

### 4.1 What relied on the old default

Twenty test gates contained an AMR cadence (`amr_every > 0`) and no
`amr_policy`, so a flip would have changed their mesh trajectory with no diff
to read: `t_g39`, `t_g48`, `t_g49`, `t_g56`, `t_g59`, `t_g60`, `t_g71`,
`t_g74`, `t_g75`, `t_g77`, `t_g79`, `t_g82`, `t_g90`, `t_g112`, `t_g153`,
`t_g173`, `t_g178`, `t_g196`, `t_g208`, `t_g221`. Eight scripts did the same:
`amr_highra_accept.jl`, `kelly_100.jl`, `kelly_100_frames.jl`,
`kelly_100_restart.jl`, `kelly_long.jl`, `restart_demo.jl`,
`run_blankenbach.jl`, `tosi_diag.jl`. `case_realistic_2d.jl` was the only
script that pinned the policy, at `:477`. Stage 0 added
`amr_policy = :classic` to all 28 files, 65 insertions against 8 deletions, with
a one-line comment at each site saying why. The suite before the flip was
133566 pass, 2 fail, 2 broken, and the 20 affected gates in one process were
1404 of 1404.

Untouched by a driver-only flip: every gate that calls the driver with
`amr_every = 0`, and every unit-level gate that calls `amr_cycle!` or
`semquad_amr_passes!` directly (`t_g32`, `t_g46`, `t_g55`, `t_g62`, `t_g215`,
`t_g216`), because the helper's own default stayed `:classic`
(`julia/src/semquadconvection.jl:108`). That split is the whole reason the
stage paid one day of edits and not a week of re-pinning.

### 4.2 The one gate that had to be re-pinned or kept classic: t_g208

The survey's table had exactly one row in the hard-failure column:
`t_g208_mirror_invariant`, whose driver-path block asserts `steps == 5`,
`leaves == 118` and three FNV hashes (`julia/test/t_g208_mirror_invariant.jl:443-447`).
Those five assertions were produced under the default policy, so the flip would
have changed all of them.

The resolution was to keep the gate classic, not to re-pin it. Stage 0 had
already put `amr_policy = :classic` in the call at `:441`, so the gate is
self-documenting; re-pinning it under `:balanced` would have duplicated the new
gate and deleted the only classic driver coverage in the suite. `G210`
(`:271-274`), `G211` (`:297-300`) and `G213` (`:322-325`) already asserted
`:classic` explicitly and needed no edit. The gate's three tree hashes,
`35968c872ebeabf0`, `07812bd2ff43ed4d` and `0de7757f838fd16f` (`:388`, `:399`,
`:418`), come from direct `semquad_amr_passes!` calls and never saw the driver
default at all.

### 4.3 The new gate that pins the default: t_g229

`julia/test/t_g229_default_balanced.jl` (83 lines) runs the same 5-step case as
G208(d) with no `amr_policy` kwarg: `Ra = 1e4`, `max_leaves = 120`,
`amr_every = 1`, `coarsen_frac = 0`, `refine_frac = 0.5`, `max_level = 5`. It
asserts that the checkpoint's `run_config` records `"balanced"` (`:66`), that
the `balanced:` diagnostic line is printed (`:68`), that two identical runs are
bitwise equal in leaves, `theta`, `ux`, `uz` and `history` (`:72-76`), and then
four pins of the new path: `leaves == 118` (`:79`), `theta`
`47ee97ea1cc249de` (`:80`), `(ux,uz)` `8a33b2262144a1fb` (`:81`), `history`
`2ceaf052be8af11c` (`:82`), 13 assertions in all. The suite after Stage 1 was
133579 pass, 2 fail, 2 broken, which is the Stage 0 baseline plus those 13.

One detail is worth keeping: both policies end this tiny case at 118 leaves and
different bytes. The leaf count is not a policy fingerprint; the FNV hashes are,
which is why the gate pinning the default carries three of them.

### 4.4 The two canonical hashes that the flip cannot move

The two hashes quoted in `docs/VALIDATION.md:137-138` do not go through AMR at
all. `bitwise_theta_sha.jl` passes `maxsteps = 20` and `maxsteps = 2` and no
`amr_every`, so `amr_every = 0` and the AMR branch at
`julia/src/semquadconvection.jl:1643` never fires.

* 2b, 20 steps, EV+CFL: `theta d22fdab099ae84e0`, `ux 2c60e7985fd9b8ec`,
  `uz def3383e3d7b6255`, `pv 89de0ec00d6557ac`, `history 24dea03838128ba6`,
  `Nu_top 7.0299964227238032`.
* blob frame 19, 1024 leaves, 2 steps: `theta 45ac8acd099adadc`,
  `ux e41c4a173e290036`, `uz 2371b0e7a8db9e6e`, `pv 38bcc13372cfd60f`,
  `history a8677df8737e63b5`, `Nu_top 5.9916941925454079`.

Measured on 2026-11-13, same session: shipped default 100 s, forced
`amr_policy = :balanced` on both driver calls 101 s, every hash and both
`Nu_top` values identical. No regeneration was needed and neither table row in
`docs/VALIDATION.md` changed. This is the survey's most useful finding and it
generalises: a hash that does not turn the feature on cannot pin the feature.

### 4.5 The 25-frame blob, re-baselined

This is the only canonical baseline that runs AMR, so it is the only one that
had to be re-measured. The protocol is fixed: 1 by 1 box, `Ra = 1e8`, `H = 5`,
`p = 4`, `max_level = 6`, `max_leaves = 1024`, Kelly marking every 5 steps,
`dt = 8e-11` held constant, `tstop = 1e-8`, 25 frames every `4e-10`,
`symmetry = :x`, `coarsen_frac = 0.005`, `coarsen_theta_tol = 1e-3`,
`linear = :direct`, `stokes_tol = 1e-6`
(`julia/scripts/run_blob_kelly_1024.jl:88-95`). Both arms are 126 steps and
25 frames:

| quantity | classic arm | balanced arm |
|---|---|---|
| end-of-run `theta` SHA-256 | `868a79d92a78bcfe89cfacd4774a0301c0eeb0a865284ab005155f6f458afe88` | `a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda` |
| end-of-run `(ux,uz)` SHA-256 | `c2a217d67c41973ba893d563d5ef200efc141545685e1d1653d7366dc36af491` | `1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51` |
| frame-25 `Tmax` | `1.264427445` | `1.264329738` |
| frame-25 leaves / `N` | 1024 / 17609 | 898 / 15361 |
| wall clock, 2026-11-13 session | 129.8 s | 134.5 s |

The classic arm is not withdrawn, and it is not a reconstruction: with
`BLOB_POLICY=classic` the producer reproduces the digest bitwise, 126 steps,
1024 leaves, 129.8 s. The balanced arm is the current record: the budget
exchange spends the 1024-leaf budget and settles at 898 leaves from frame 19
on, `frame-25 Tmax` drops by 9.7e-5, and the wall clock rises by 4.7 s. The 25
per-frame `theta` digests of the balanced arm are tabulated in
`docs/VALIDATION.md` §2.3.1; the producer writes the same rows as
`frame_sha256.csv` and one `run_summary.csv` row.

The re-baseline found a convention error in the old record. The digest that the
documentation quoted as the trajectory's last field is the **end-of-run**
`theta`, taken after step 126, and frame 25 is the callback field at
`t = 1.0e-8`, one step earlier (`julia/scripts/run_blob_kelly_1024.jl:98-101`).
In the balanced arm frame 25 hashes to
`32eadb4028233645fea571868b887540e1c55696d4e07dfbf9a4352b9e90a6b3` and the
end-of-run field to `a4fb8248...b8cdeda`, so the two are different objects with
different bytes. The classic-era table inherited the mislabel because the old
producer printed only the end-of-run digest, which reads as the last frame's.
The fix is the per-frame table plus the labelled summary row, and it is why
`docs/VALIDATION.md` §2.3 now says which field each digest is over.

### 4.6 What was not re-run, and what the flip cost

E is 22.47 h at 4.044 s/step and G is 17.45 h at 4.13 s/step, and neither was
re-run. They stay reproducible through the explicit `CASE_AMR` pin of §3.1, and
that is the only requirement the flip puts on them. The survey priced the whole
procedure at 1.0 to 1.5 h: 100 s for the two AMR-free baselines, 5 to 9 min for
the 20 gates, 2 to 5 min for the blob, 40 to 75 min for the full suite. The
alternative it priced as well was to keep `:classic`, add one gate that asserts
the documented default, and write one paragraph of prose, about 15 min and no
re-baseline. That option lost because the card had been balanced since
`715e272`: the library default was the only inconsistent piece left, and a
default that differs between the production script and the library is exactly
what the survey was written to find.

## 5. Restarts: the watchdog rule, and what a checkpoint remembers

`scripts/watchdog_case.sh` (550 lines) is the unattended restart for the long
realistic runs. It forwards `CASE_AMR=classic` (`:106`), `CASE_SYMMETRY=x`
(`:107`) and `CASE_AMR_GATE=0` (`:108`) into every resume it launches
(`:525-527`). Those are the pre-2026-11-13 card values on purpose: the script
only resumes existing checkpoints, while the card itself now defaults to
balanced, none and 1 (`julia/scripts/case_realistic_2d.jl:211`, `:225`,
`:234`). Its header states the rule at `:69-76`. A run started under the new
defaults has to export all three, because the restart guard compares them.

The guard is where the rule comes from. `run_semquad_simulation` writes
`run_config` at `julia/src/semquadconvection.jl:673-699` and, on a resume,
compares every key of the checkpoint's own `config` against the current one,
raising `ArgumentError` on the first mismatch (`:809-832`, throw at `:829-831`).
So a restart must keep the policy its checkpoint recorded: `:classic` for E,
`:balanced` with the matching gate and symmetry for F and G.

The guard also has a blind spot, and it is structural. The loop iterates over
the checkpoint's keys, `for (k, v) in restart.config`, with
`haskey(run_config, k) || continue` (`:813-817`), so a key that is absent from
the file is never compared. Only `stokes_scaling` gets a hardcoded
absent-means-`none` special case (`:823-828`). A checkpoint written before
`amr_policy` and `amr_coarsen_gate` existed does not carry them, and the guard
cannot compare what is not in the file. Measured over the nine production
checkpoints on 2026-11-13, with `h5py` and no run:

| case | file | step | `config` keys | `amr_policy` | `amr_coarsen_gate` | cooldown |
|---|---|---|---|---|---|---|
| E | `case_realistic_E/spinup.h5` | 20000 | 71 | absent | absent | none |
| D | `case_realistic_D/resume.h5` | 7400 | 71 | absent | absent | none |
| C | `case_realistic_C/resume.h5` | 6600 | 71 | absent | absent | none |
| B2 | `case_realistic_B2/{resume,spinup}.h5` | 6000 / 5800 | 71 | absent | absent | none |
| B_auto76 | `case_realistic_B_auto76/spinup.h5` | 76 | 71 | absent | absent | none |
| F (the `_nogate` arm is the only one on disk) | `case_realistic_F_nogate/spinup.h5` | 600 | 73 | `balanced` | absent | none |
| G, early | `case_realistic_G/spinup.h5` | 7600 | 74 | `balanced` | `true` | none |
| G, late | `case_realistic_G/resume.h5` | 15000 | 75 | `balanced` | `true` | `(702,4)`, `(2712,4)` |

A watchdog restart of E with `CASE_AMR=balanced` therefore starts silently, and
the banner prints `amr=balanced ... gate=1` (`julia/scripts/case_realistic_2d.jl:460`)
over a step history that was written by the classic policy. The same hazard
exists for `amr_coarsen_gate` on the F and G-early files, which carry no key. A
G restart is in the opposite position: the file records `balanced` and `true`,
so forwarding the watchdog's `classic` and `0` fails the comparison instead of
passing it, which is the behaviour we want.

The cooldown adds a second asymmetry. It is state, not configuration: absent
datasets load as an empty `AmrCooldown`
(`julia/src/semquadconvection.jl:950-957`,
`julia/src/semquad_checkpoint.jl:92-100`). A run that switches policy starts
with no hysteresis memory, so its first balanced cycles are more active than a
native balanced run at the same step. G's own `spinup.h5` is in that class: it
carries the policy string and no cooldown, because it was written after the
policy key existed (`117cc9d`) and before cooldown persistence (`8bb11e6`),
both on 2026-11-10. `G213`'s bitwise-restart property holds only for
checkpoints written after `8bb11e6`.

A policy switch in the middle of a trajectory reproduces neither arm. E's
step-20000 state is `Nu_top 18.2892`, `Nu_bot -2.7636`, `Vrms 913.7`,
`meanT 1.2620`, 9994 leaves; G's step-15195 state is `Nu_top 16.597`,
`Vrms 1113.2`, 10000 leaves. A hybrid matches neither, and it also breaks the
E-versus-G early-window comparison that relies on both arms being pure. The mesh
itself stays legal, because every pass ends with `mirror_close!`, `balance!` and
the hard budget pass (`julia/src/semquadconvection.jl:286-298`), so the loss is
provenance rather than correctness. A balanced continuation of E is a new arm
with a new output directory and a new frame stream, and it is not a resume.

## 6. Legality: the 2:1-aware coarsening pre-check

### 6.1 The gate

A merge at child level `L` produces a parent at level `L - 1`. If any face
neighbour of that parent ends the pass at level `L + 1` or finer, the merge is
illegal: the post-merge `balance!` splits the parent back, and the cycle pays for
a merge it does not keep. At the operating point this is the normal case, not an
edge case. On the `:none` tree, 14 of the 15 criterion-eligible quartets are
2:1-illegal; 1635 of 1873 complete quartets are legal, but the criterion never
selects them. The low-eta tail is by construction the coarse side of a 2:1 jump:
a coarse leaf next to a refined plume has a small gradient jump while its fine
neighbour carries it.

The gate is deal.II's `coarsening_allowed` (`tria.cc:10935-10996`, v9.4.0),
translated to one quartet. The candidate parent is `p = _pq(k00)`, its child
level is `L`, and the merged parent level is `L - 1`. For each of the four
external faces, and for each of the two boundary children on that face, look at
every leaf neighbour `nb` on the other side. Drop `nb` if it belongs to the
quartet itself: in our table the parent's own children appear in a parent-face
query, which is why the query is per boundary child. A neighbour that ends the
pass at level `lev(nb) + eff(nb)` is legal only when `lev(nb) + eff(nb) <= L`.
The effective level is
`eff(nb) = (nb in will_refine ? +1 : 0) - (_pq(nb) in granted ? -1 : 0)`, where
`will_refine` is the set of leaves the refinement plan splits and `granted` is
the set of parents whose quartet this plan already merged. The `+1` is deal.II's
refine flag: refinement runs after coarsening, so a neighbour that will be split
is one level finer in the final tree. The `-1` is deal.II's `user_flag_set` on a
refined neighbour whose children are all coarsening: that neighbour becomes a
leaf at its own level, so it is one level coarser before the refine pass.
`granted` wins when both apply, because a merged leaf is gone before the refine
pass runs. Skipping a neighbour coarser than the child is implicit in the
inequality.

`_merge_allowed` (`julia/src/amr_balanced.jl:205`) adds the complete-set rule:
all four children must be leaves, and under `symmetry=:x` the mirror quartet must
also be complete and pass the same gate. The pair is atomic, because
`_coarsen_pair!` merges both halves or neither. The pre-check is computed before
any mutation, in `_coarsen_plan` (`:320`), which mirrors deal.II's
`fix_coarsen_flags` (`tria.cc:14830-15032`): candidates are complete,
unprotected quartets at `level > l_min`, not cooled, with no child at or above
`tau_r`. They are ordered deadband first, then by summed indicator. Grants are
decided finest level first, each against the grants already made, and the sweep
repeats until no new grant appears. A coarser candidate that needed a finer
quartet is retried after that quartet is granted, which is deal.II's outer flag
loop. The whole plan is then executed, so its grants stay self-consistent;
stopping in the middle would leave a granted quartet facing a neighbour whose
merge did not happen. The final execution order is sorted by the quartet's
maximum child indicator.

### 6.2 The cheapest repro, and the cost

The smallest correct demonstration is one merge next to a two-level finer leaf.
The quartet at child level `L` merges to `L - 1`; the neighbour at `L + 1` is two
levels finer. `balance!` splits the new parent, and the leaf count returns to
where it started. The merge cost three leaves, and the balance restored three.
The gate refuses the merge up front, which is exactly the `-3` then `+3` that
the old cycle paid every time. Measured on the operating tree, merge-undo
recirculation falls from 100 % of 14 merges per cycle to 1.0 % of 4881; over 200
steps the rate falls from 54 % to 3.7 % and the absolute count from 1251 to 576
leaves. The gate costs about 0.1 us per neighbour query and about 0.2 ms per
pass at 10 000 leaves, with two sets of order 10^3 entries. That is noise
against the pass it saves.

## 7. The local 2:1 rebalance

### 7.1 The queue

Every budget merge used to be followed by a full `balance!`. At 10 000 leaves
that is a sort of the whole leaf set plus four face queries per leaf, 22.9 ms
per merge, and the pass does 14 to 29 merges per cycle. The local rebalance
replaces it with a FIFO of leaf ids and a queue bit per live id
(`julia/src/dgquad.jl:805-920`):

    push(x): x is a live leaf and not queued -> queue.push(x); queued[x] = true

    seeds: for the merged parent p (and p' = mirror(p) under :x):
           push(p); push(N(p));  [push(p'); push(N(p'))]

    while queue not empty:
        x = pop(); queued[x] = false            # clear on pop, not on push
        x is a live leaf                        || continue   # stale entry
        N = N(x)                                # snapshot BEFORE the split
        x violates 2:1                          || continue
        split(x); push(children(x)); push(N)    # Lemma 2: only N can cascade
        parents.push(_pq(x))

The queue is keyed by the dense leaf id, so a stale entry costs one array read.
The `queued` bit is cleared when the id is popped, not when it is pushed. A leaf
tested clean can become violated later, after a neighbour is split, and must be
testable again; clearing on push would silently drop that later violation. The
neighbour list is snapshotted before the split, because the split mutates the
table and reading afterwards returns the children's faces instead of the old
leaf's. Every split pushes the four children and the old neighbours, which is
exactly the set Lemma 2 allows to become newly violated.

Only the merged parent's neighbourhood matters because of Lemma 1. A merge
replaces four leaves by one parent. Every other leaf keeps the same neighbours
or sees a coarser one, so its level difference can only shrink. The new parent
is the only leaf that can be two levels below a neighbour. The queue starts
there, and each split can create a new violation only in the coarser neighbours,
one level up per wave. That is the entire cascade.

Under `symmetry=:x` the split is pair-atomic. If `x` has a mirror leaf, the
partner is split too, even when the partner does not violate; if the partner is
already split by an earlier pair, skipping it is not an error. A partner at
`max_level` counts as unsplittable rather than leaving a one-sided split. The
seeds are mirror-closed, so `{p, p'} union N(p) union N(p')` may cover two
disjoint patches, which is fine for a queue keyed by id. `mirror_close!` remains
the repair path for a tree that enters damaged; the local pass preserves closure,
it cannot create it.

### 7.2 Termination

The pass is refine-only. A split replaces one leaf by four, so the leaf count
grows by 3 per split, or 6 for a mirror pair. A leaf is split at most once per
pass, the queue receives at most four children and eight neighbours per split,
and the number of pops is capped structurally at `64 * max(n, 64) + 1024`,
recorded as `local_capped`. A leaf at `max_level` that still violates is counted
as `unsplittable` and not requeued; no refinement can fix it. The return value is
`ok = !capped && unsplittable == 0`, and the budget pass carries the counters in
`LAST_BUDGET_PASS` (`splits`, `tests`, `local_capped`, `entry_21`,
`unsplittable`). The worst case is real but bounded: a merge can force a cascade
through most of a graded tree, and then the pass is linear in the leaves it
touches. That is still no worse than the global `balance!`, which visits every
leaf and sorts them. On the production state the cascade is almost empty: 2
splits in 40 merges, 8.2 queue tests per merge.

### 7.3 Cost

| per merge, 10 000 leaves | global `balance!` | local `rebalance_local!` |
|---|---|---|
| tree work | sort all leaves, four queries per leaf | one parent and at most eight neighbour reads |
| allocation | 0.35 MB plus the old per-query vectors | 0 B steady state |
| wall clock | 22.9 ms | 7.2 us (median 6.7, max 13.2) |
| ratio |  | about 3 200x |

The 7.2 us are the seed expansion and a handful of queue pops. The flat table is
what makes them O(1). With the old cached `neighbors` the same pass measured
14.3 us, of which 13.3 us were the four seed queries alone, because every merge
invalidated the cache; the flat table and the local rebalance are one change, not
two. At the pass level, the hard budget pass goes from 0.3206 s/cycle to 0.0113
s/cycle, 28x. The AMR pass goes from 0.4412 to 0.1198 s/cycle, 3.7x, and the
allocation goes from 601.95 MB to 282.47 MB per four cycles. The remaining 11.3
ms is 83 % candidate scan (`_budget_quartet`, 0.67 ms per merge) and 0.1 ms
rebalancing; what is left is the entry check and bookkeeping. A priority queue
would change the merge order, which changes the bitwise trajectory, so the
linear scan stays until a separate audit.

### 7.4 Why local equals global

The local pass may replace `balance!` only because of two lemmas and one
theorem. Write `T` for a 2:1 tree, `T'` for the tree after one or more complete
quartet merges, `G = balance!(T')`, and `L` for the local queue seeded at the
merged parents and their face neighbours.

**Lemma 1.** If `T` is 2:1 and `T'` replaces one complete quartet of four leaves
at level `L` by its parent `p` at `L - 1`, the only leaf of `T'` that is not 2:1
is `p`.

Every other leaf of `T'` is a leaf of `T`, and merging only replaces leaves by
parents, so every neighbour level it sees is unchanged or decreased. A violation
at such a leaf would need a neighbour level that increased, and none did. The new
parent is the only leaf with new neighbours. Those neighbours are the old
neighbours of its four children, at levels at most `L + 1` by the 2:1 property of
`T`, and `p` is at `L - 1`, so it can be two levels below them. The batch form
follows: for any set of merges on one 2:1 tree, the only possibly violating
leaves are the merged parents, and two parents are never two levels apart because
their children were face neighbours in `T`. The seed set is therefore `{p} union
N(p)` per merge.

**Lemma 2.** If a violating leaf `x` is split, the only leaves that can newly
violate are the neighbours `y` of `x` with `lev(y) = lev(x) - 1`.

Only `x`'s children and `x`'s neighbours change adjacency. A child is at most one
level coarser than any old neighbour, so no child can violate. A neighbour `y`
now faces a child at `lev(x) + 1`; it violates exactly when
`lev(y) <= lev(x) - 1`, and 2:1 held before, so that is the only new case. This
is the one way a split propagates upward, one level per wave.

**Theorem.** If the queue empties, `L` and `G` have the same leaf set. `G` is the
least 2:1 refinement of `T'`. A split always creates the canonical four children,
so any 2:1 refinement is closed under "if a cell is split, its four children are
present". If a leaf `c` of `T'` has a neighbour at `lev(c) + 2` or finer, `c` must
be split in any 2:1 refinement: otherwise that neighbour refines a cell two
levels finer and the result is not 2:1. By induction every cell split by the
closure is split by every 2:1 refinement, and the closure is itself 2:1, so it is
the unique least one. `balance!` computes that closure. `L` computes the same
closure: Lemma 1 puts every forced split in the seed set, Lemma 2 makes the queue
closed under the only event that creates a new forced split, and the queue splits
only actual violators. Both end 2:1 and minimal, hence equal.

The measurements agree. On 150 random 2:1 trees with random complete quartets,
local and global agree on 197 of 197 merges; the parent was the violator in 75 of
150, 50 %, consistent with the 54 % pre-gate undo rate. On a 12 544-leaf
wall-graded tree, 60 of 60 merges agree. On mirror-closed `:x` trees, 118 of 118
pair merges agree with `mirror_defect == 0` and `max_face_level_jump <= 1`. The
whole budget pass agrees with the old implementation on 60 of 60 random cases,
including `merged`, `reached`, the leaf-set FNV, the jump, and the defect.

The counterexample if the 2:1 entry precondition is dropped: take `Quadtree(5)`
uniformly at level 2, 16 leaves. Refine `(2,1,3)` and then its child `(3,3,7)`
without balancing. That is 22 leaves, `max_face_level_jump = 2`, and the sole
violator is `(2,2,3)`, which faces a level-4 leaf. Merge the far quartet of
`(2,0,0)`, whose parent is `(1,0,0)`. The seed is
`[(1,0,0),(2,0,2),(2,1,2),(2,2,0),(2,2,1)]`. The local pass splits nothing, ends
at 19 leaves and jump 2, because the violator is not in the seeded
neighbourhood. The global `balance!` splits `(2,2,3)`, follows its cascade, and
ends at 22 leaves and jump 1. The budget pass cannot reach that state because it
starts each merge from a 2:1 tree, but a checkpoint restore, a `mirror_close!`
repair, or a batch plan can. `rebalance_local!` therefore checks the
precondition with `_tree_is_21`; a tree that enters unbalanced is repaired by
seeding every violator with `all_violations = true`, which computes the same
closure as `balance!` from that tree too. `G216` pins both the counterexample
and the repair.

## 8. Load-bearing quirks

One line in the budget pass looks like a bug and is deliberately kept. When a
merge is undone by the rebalance, `semquad_enforce_leaf_budget!` adds the merged
parent to `blocked` and then, for each of the quartet's four children `k`, pushes
`(k[1], k[2] >> 1, k[3] >> 1)` (`julia/src/dgquad.jl:908-913`). The key is built
from the child level `k[1]`, not from `k[1] - 1`, so it is not the merged parent.
`parent` is pushed separately. The four pushes collapse to one level-`L` key,
and that key can be the parent of a complete quartet one level deeper, so the old
pass occasionally blocked an unrelated quartet. That is the quirk: the keys are
at the children's own level, and they do not mean what the line appears to mean.

It is load-bearing. Removing the four pushes makes the pass merge more on 6 of
the 60 random cases used for the whole-pass A/B: four end with a different leaf
set, two differ only in `merged`, and in two of them the pass reaches the budget
where the old one could not. The clean version is arguably better greedy
behaviour, but it is not the old trajectory, and the stage was required to be
bitwise. For a bitwise stage the historical line stays, documented at the call
site in `julia/src/dgquad.jl:897-907`.

For the same reason `blocked` does not receive the parents the rebalance splits.
With the linear candidate scan, a cascade-split parent has an incomplete quartet
and cannot be re-merged, so the entry would be inert. Adding it would still
change `blocked`, and `blocked` changes the trajectory. The reference note's
advice to block cascade parents applies to a priority-queue order, which is not
what this pass uses.

The pins are layered. `G215` carries the pre-audit `_budget_quartet` and
`semquad_enforce_leaf_budget!` verbatim, including that push, and requires the
same answer on 24 random graded trees (2 140 quartets, 96 plans, 2 949
candidates, 159 budget merges, leaf-set FNV identical). `G211` pins the gate and
the plan: the illegal quartet is refused, the legal one is paid for, the
extension past `tau_h` is counted, `recirc == 0` after a gated cycle, `balance!`
is a no-op there, and the pre-gate arm does recirculate. `G216` pins the flat
table against the historical scan and the local rebalance against `balance!`,
including the counterexample and the `all_violations` repair. The bitwise hashes
(`G150`, `G208`, `G210`, and `G229` for the new default) and the pinned
default-path FNV values stay unchanged through all of it.

## 9. The hot-path audit

The audit looked for two bug classes: a per-cell lookup that allocates or boxes,
and a loop that spins on a fixed point. The S15 class had four items on the
`:balanced` path, plus one allocation hot spot in the candidate scan. All were
measured on the 10 000-leaf operating tree, in the same process, with the
historical implementation carried in the probe as the before column.

**`_coarsening_allowed`.** The old predicate built `Set(vec(kids))` and four
`vec(kids[:, d])` vectors on every call. One call was 864 B and 200 ns. The
replacement tests membership by parent key, `_pq(nb) == _pq(k01)`, because only a
level-`l` child of the same parent can be in the quartet, and reads the eight
boundary children directly. It is 0 B and 78 to 81 ns, 2.6x. On a standalone
12 544-leaf wall tree the same change measured 1 184 B and 493 ns to 0 B and
164 ns, 3.0x. The matrix method is kept as a forwarding wrapper because `G211`
calls it.

**The `Core.Box` in the funding closure.** `_fund_live!` used to be a closure
inside `amr_cycle_balanced!` that assigned `nc`, `gate_extended` and read `tau_r`.
Julia lowers a captured reassigned variable to a `Core.Box`, so every read inside
the funding scan was an `Any` load. The function is now top-level
(`julia/src/amr_balanced.jl:251`), takes `tau_r::Float64`, returns
`(merged, extended, leaf)`, and lets the caller do the accounting.
`@code_warntype` on the cycle went from three boxes to zero `Core.Box` and zero
`Any`; JET's `@report_opt` reports zero runtime dispatches on the four audited
entry points. The same audit removed the abstract `lmin::Function` annotation in
`_coarsen_plan`, which made every `lmin(...)` in the candidate loop a dynamic
dispatch.

**The repeated sort and the candidate matrix.** `_coarsen_plan` re-sorted the
whole leaf set on every call: 1 163 088 B and 6.26 ms. The cycle already has the
ascending order, sorted once for the thresholds, so the plan takes it as a
keyword; the sweep permutation is hoisted out of the fixed-point loop, and the
four-child tests use the tuple `_kids4`. The result is 442 872 B and 1.63 ms,
3.8x. `_budget_quartet` built a 2x2 `Matrix{NTuple{3,Int}}` per leaf per scan:
2 021 888 B and 1.98 ms for a full scan. It is now a 4-tuple in the same
column-major order, so the summed indicator is bit-identical, and the scan visits
one leaf per sibling quartet instead of four. That is 0 B and 0.67 ms, 3.0x. The
track-allocation run attributed 108.24 MB per four cycles, 27 MB/cycle and 15 %
of the pass, to this one row; it is now 0.001 MB.

**The lazy face-cache clear.** Every `refine!` and `coarsen!` used to `empty!` a
10 000 to 40 000 entry `Dict`. The clear alone measured 0.022 to 0.094 ms. The
gate-off cycle does about 28 mutations, so that was about 2.6 ms per cycle; the
gate-on cycle does more than 570, so about 54 ms per cycle, which is most of the
0.163 to 0.067 s drop in `amr_cycle_balanced!` after the fix. The cache became
version-stamped and lazily cleared, and stage A then replaced it with the
incremental table of §1.3.

The allocation ledger, four production-shaped cycles, MB per four cycles:

| file / function | baseline | after audit | stage A | stage B |
|---|---|---|---|---|
| total | 710.19 | 601.95 | 308.17 | 282.47 |
| `quadtree.jl` | 334.27 | 334.27 | 40.49 | 14.79 |
| `_collect_face!` | 223.94 | 223.94 | 0 | 0 |
| `neighbors` output vectors | 74.64 | 74.64 | 0 | 0 |
| `balance!` sort | 27.60 | 27.60 | 27.57 | 1.87 |
| `face_pairs` | 8.07 | 8.07 | 8.07 | 8.07 |
| `gll.jl` | 166.02 | 166.02 | 166.02 | 166.02 |
| `semquad.jl` | 97.60 | 97.60 | 97.60 | 97.60 |
| `_budget_quartet` | 108.24 | 0.001 | 0.001 | 0.001 |
| `amr_balanced.jl` | 3.12 | 3.12 | 3.12 | 3.12 |

Stage A is the flat table: the two neighbour-query rows are gone, 298.58 MB or
49.6 % of the baseline, and the total drops 48.8 %. Stage B is the local
rebalance: 25.7 MB more, because the one remaining `balance!` per cycle is
1.87 MB and `rebalance_local!` is 2 kB. The remaining `quadtree.jl` allocation
is one-off table growth: the pool doubles from empty to its 10 000-leaf size
(2.64 MB), and the id arrays grow with the leaf count (1.71 MB and 0.50 MB). A
longer run does not repeat them.

The time ledger tells the same story. The audit's own fixes took the gate-off
pass from 0.4729 to 0.4436 s/cycle, 6.2 %, with identical leaf and `theta` FNV
in all 12 cycles. The gate-on pass is 0.5037 s/cycle in the same protocol,
because the gate does 20 times the operations and 30 to 60 times the structural
work: 284 refinements and 284 coarsenings in the first cycle instead of 14 and
14, plus the plan walk and 545 funding scans. The increase was work, not a type
bug. After stage A and B, the gate-off pass is 0.1198 s/cycle and the gate-on
pass 0.1591 s/cycle; in the gate-off arm the hard budget pass is 9.4 % of the
cycle, down from 72.7 %.

The step-level effect is below this machine's noise. The audit's 10-step runs
spread over 2.6 %, and the pass change was 0.029 s/cycle, 0.006 s/step, 0.1 % of
a step. At 5.6 s/step the AMR pass is 1.6 % of an average step and 7.8 % of an AMR
step, so no AMR optimisation can move the step time. The value of the audit is
the bitwise-preserving 6 % of the pass, the removal of an O(n^2/target) plan
failure mode, and the fact that the production card's pass is now cheap enough
that the remaining work is supply and correctness. The 5.6 s denominator is the
direct-saddle step of the audit; after `2453f16` the library default is
`linear = :auto`, so an MG run has a smaller denominator and a larger AMR share
with the same numerator.

## 10. What is still open

The allocation is now 30.50 MB per four cycles, down from 282.47 MB, a 9.3x
reduction on the stage-B column above. Almost all of it is necessary structural
allocation: `semquad.jl` 17.88 MB (`make_semquad` 15.27, the transfer output
vector and its `Set` 2.61), `quadtree.jl` 8.55 MB (`face_pairs` 1.83 and the
one-off table growth), and `amr_balanced.jl` 3.12 MB, `semquadconvection.jl`
0.92 MB, `dgquad.jl` 0.001 MB, `gll.jl` 0.004 MB. That is the mesh rebuild and
the transfer, not per-cell temporaries. The assembly audit's §3 calls it the
necessary part; there is no per-call boxing left to remove on this path.

The `Vector{NamedTuple}` face-pair item is closed at HEAD, and the history is
worth keeping straight. The AMR audit listed it as abstract-element boxing; the
assembly audit changed the element type to the concrete `FacePair`, which cut
`face_pairs` from 8.07 to 1.83 MB per four cycles and removed the boxed field
reads in the indicator and the assembly. What remains open is the per-version
rebuild and sort of the pair list, 1.83 MB and one list build per mesh version. A
flat-array pair list is the next stage named in the audit, if the pair rebuild
ever matters against the indicator and the transfer.

A few things are worth naming for the next person. `_fund_live!`'s scan is
repeated, 545 calls in the first gate-on cycle, each walking the indicator order
until a legal quartet appears; that is inherent to the "pay with the lowest legal
quartet" rule, and caching rejections is unsound because a merge can make a
previously illegal quartet legal. The one remaining global `balance!` per cycle
is 1.87 MB and about 1.4 ms, and it is still there because the cycle has to leave
a 2:1 tree for the assembly. A priority queue in the budget pass would change the
merge sequence and the bitwise trajectory, so it needs its own audit before it
can replace the linear scan. And the 2:1 rule here is face-only; ASPECT also runs
the vertex rule, which is a documented deviation. It is irrelevant for the DG
assembly, but it means our meshes cannot be claimed bitwise equal to ASPECT's.
