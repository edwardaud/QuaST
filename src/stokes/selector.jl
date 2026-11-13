# --- S6: production path selector (`linear=:auto`) ---------------------------
#
# `linear=:auto` chooses between the sparse direct LU and the matrix-free MG
# from two properties of the *current* saddle: its size and the viscosity
# contrast of the eta field actually passed to the solve.  The constants below
# are calibrated against a same-session A/B of the two paths on the states the
# previous table named, re-measured at `c7e0f55` after the optimisation
# campaign of docs/dev/MG.md§16 (that campaign's endpoint is a 2.3-2.6x
# faster warm MG solve than the one the previous table -- docs/dev/SOLVER.md
# §18.3/§19.2 -- was calibrated on, while the direct path is untouched).  The
# A/B and every row: `benchmarks/bench_auto_crossover.jl`, one process per
# state, both paths on the same tree and closures, `BLAS=1`, direct
# `linear=:direct, stokes_scaling=:equilibrated` (the tolerance-reaching
# reference), MG `linear=:mg` with the shipped defaults and
# `mg_tol = 1e-6` (the direct reference's accuracy), warm = best of three
# in-tree solves after a cold build.  Ratio = t_direct / t_mg, so >1 means MG
# faster (the convention of docs/dev/SOLVER.md §4.2/§4.4, which carries the same
# table plus the cold column):
#
# | state                                 | saddle dofs | contrast | 1thr | 8thr |
# |---------------------------------------|-------------|----------|------|------|
# | uniform L4  (256 leaves, p=4)          |      10 851 |        1 | 3.86 | 3.94 |
# | uniform L5  (1 024 leaves, p=4)        |      42 691 |        1 | 7.36 | 7.21 |
# | uniform L6  (4 096 leaves, p=4)        |     169 347 |        1 | 6.97 | 7.59 |
# | uniform L4 at p=2 (256 leaves)         |       2 467 |        1 | 1.91 | 1.81 |
# | uniform L4, stratified eta (p=4)       |      10 851 |       50 | 4.04 | 3.48 |
# | blob f19 (1 024 leaves)                |      45 475 |    5 087 | 1.78 | 1.55 |
# | blob f19x4 (2 344 leaves)              |      99 559 |    5 087 | 2.41 | 2.44 |
# | blob f19x16 (4 096 leaves)             |     174 915 |    5 087 | 2.62 | 2.77 |
# | blob 25-frame trajectory (frame 1)     |       9 351 |    3 024 | 1.48 | 1.11 |
# | blob 25-frame trajectory (frame 14)    |      28 135 |    5 087 | 2.01 | 1.52 |
# | blob trajectory, 25-frame mean         | 9 351-45 475|    5 087 | 1.55 | 1.51 |
# | production `F` (8 188 leaves, p=2)     |      78 720 |  5.7e6   | 2.27 | 2.66 |
# | production `P1` (10 000 leaves, p=2)   |      95 308 |  3.3e6   | 1.60 | 2.06 |
#
# so the selector is: any contrast above 20k dofs, and contrast <= 100 above
# 2k dofs.  The old ladder (60k / 20k / 8k, with a contrast 10 rung) moved down
# 3-4x because MG got faster and the direct path did not, and it lost its
# middle contrast rung: the `10 < contrast <= 100` band used to earn MG only at
# 20k dofs with no measurement behind it, and the two stratified-50x rows above
# now show it costs the V-cycle nothing measurable against contrast 1 (1.74x
# cold at 2 803 dofs, 2.20x at 10 851, against 1.74x / 2.36x at contrast 1).
# The only contrast boundary left is 100, which is where the pressure block
# switches from the mass inverse to w-BFBT -- a property of the *MG
# configuration*, not a measured crossover.  Where each constant sits (cold
# ratios are the conservative column; `docs/dev/SOLVER.md` §4.2 lists them):
#
#   * 20 000 -- every measured high-contrast state at or above it wins
#     (19 413 dofs 1.24x at 8 threads, 28 135 dofs 1.32x cold / 2.01x warm,
#     45 475 dofs 1.14x cold / 1.78x warm, 99 559 1.59x cold); below it the
#     high-contrast rows are a tie (9 351 dofs, 1.09x cold; 17 135 and 19 413
#     dofs are 0.81-0.82x at `mg_tol=1e-10`).  The old 60k sat 1.32x above its
#     parity point (45 475); 20k sits 2.1x above the last measured
#     high-contrast tie and keeps the 17-19k frames that lose at the library
#     tolerance on the direct side.
#   * 2 000 -- 747 dofs is the last measured tie/loss (1.17x cold at one
#     thread, 0.91x cold at eight); 2 467 dofs (p=2) is 1.22-1.48x cold /
#     1.91x warm, 2 803 dofs (p=4) is 1.74x cold at both thread counts, and the
#     contrast-50 tree of the same size is 1.51-1.74x cold.  The old 8k was
#     never measured below 10 851; the new constant is bracketed by two.
#
# `saddle_dofs` uses the convention of that table: `2*Nu + Np`, i.e. the total
# velocity + pressure unknowns before Dirichlet/hanging elimination (the
# reduced nf+nfp is a few % smaller).
const SEMQUAD_AUTO_DOFS_MG_ANY = 20_000
const SEMQUAD_AUTO_DOFS_CONTRAST_100 = 2_000
const SEMQUAD_AUTO_CONTRAST_MID = 100.0
# Contrast sampler resolution: (GRID+1)^2 fixed Cartesian points spanning
# [0,Lx]x[0,1].  Measured on the production blob eta field, 32 intervals
# recovers the full quadrature contrast to -3%/-9% on the 1024/4096-leaf trees
# at ~23 us/1 089 evaluations, while the <=256 strided velocity-node sample
# misses by up to 5x (docs/dev/SOLVER.md §20).  Fixed cost, independent of N.
const SEMQUAD_AUTO_CONTRAST_GRID = 32
const SEMQUAD_AUTO_MG_MAX_THREADS = 8

# Diagnostics of the last `linear=:auto` resolution: the chosen concrete path
# (`:direct`/`:mg`; `:none` when `linear` was not `:auto`), a human-readable
# reason with the two measured properties, and the sampler cost.  Log it from a
# production driver (`QuaST.LAST_SEMSTOKES_AUTO_CHOICE[]`).
const _SEMSTOKES_AUTO_CHOICE_NONE =
    (choice=:none, reason="", saddle_dofs=0, contrast=1.0, samples=0,
     sample_ms=0.0, mg_threads=0)
const LAST_SEMSTOKES_AUTO_CHOICE = Ref(_SEMSTOKES_AUTO_CHOICE_NONE)

# Safety net of `linear=:auto`: when the selector picks `:mg` but that solve is
# not acceptable -- its true reduced-saddle residual is above the caller's
# `auto_fallback_tol` (the driver passes its `stokes_tol`), or the MG path
# throws -- the solve is redone with the sparse direct path instead of aborting
# the run.  `count` is the number of such solves and `last_reason` describes the
# last one; explicit `:mg` calls never touch either (they still fail loudly).
# Reset from a driver/gate with `LAST_AUTO_FALLBACK[] = (count=0,
# last_reason="")`.
const LAST_AUTO_FALLBACK = Ref((count=0, last_reason=""))

# Pre-built reasons for the explicit-path case: recording `:none` must not
# allocate on the hot per-solve path (this block runs for every explicit
# `:direct`/`:mg` call and is the *only* thing the selector adds to it).
const _SEMSTOKES_AUTO_OFF_REASONS = Dict{Symbol,String}(
    :direct => "linear=:direct (auto not requested)",
    :schur_cg => "linear=:schur_cg (auto not requested)",
    :mg => "linear=:mg (auto not requested)",
    :lagged => "linear=:lagged (auto not requested)")

"""Format a viscosity contrast for the `:auto` reason string."""
_fmt_eta_contrast(c::Real) = @sprintf("%.4g", c)

"""
    semquad_choose_linear(saddle_dofs, contrast) -> :direct | :mg

Production path selector behind `linear=:auto` (S6).  `saddle_dofs` is the
total number of velocity + pressure unknowns (`2*Nu + Np`, the convention of
the measured table in the comment above) and `contrast >= 1` is
`max(eta)/min(eta)` on the eta field the solve uses (sampled by
`semquad_eta_contrast`).  The rule, in order:

1. `saddle_dofs >= 20_000` -> `:mg`.  Measured: MG 1.24x faster at 19 413 dofs
   (high-contrast blob trajectory), 1.55-1.78x on the frame-19 tree (45 475
   dofs), 2.4-2.8x at 99 559-174 915, 6.97-7.59x on uniform trees; it also uses
   ~1/2-1/3 of the direct solve memory.
2. `contrast <= 100 && saddle_dofs >= 2_000` -> `:mg`.  Measured: 1.51-1.74x at
   2 803 dofs (uniform and stratified contrast-50 trees), 1.91-2.75x at
   2 467-10 851, 3.5-4.0x at 10 851 with contrast 50; the 747-dof tree is the
   last measured tie (0.91x cold at 8 threads).
3. otherwise -> `:direct`.  The measured high-contrast ties (9 351 dofs, 1.09x
   cold; 17 135-19 413 dofs at the library tolerance) and the sub-2k trees fall
   here; `:direct` is also the bitwise path, so ties go to it.

The returned symbol is what the solver actually runs; explicit `:direct` and
`:mg` calls never consult this function and keep their historical behaviour.
"""
function semquad_choose_linear(saddle_dofs::Integer, contrast::Real)
    choice, _ = _semquad_choose_linear_rule(saddle_dofs, contrast)
    return choice
end

"""
    _semquad_choose_linear_rule(saddle_dofs, contrast) -> (choice, reason)

The rule of [`semquad_choose_linear`](@ref) plus the human-readable reason used
for `LAST_SEMSTOKES_AUTO_CHOICE`.  Kept in one place so the logged reason can
never drift from the decision.
"""
function _semquad_choose_linear_rule(saddle_dofs::Integer, contrast::Real)
    dofs = Int(saddle_dofs)
    c = isnan(contrast) ? Inf : max(1.0, Float64(contrast))
    if dofs >= SEMQUAD_AUTO_DOFS_MG_ANY
        return (:mg, "saddle_dofs=$dofs >= $SEMQUAD_AUTO_DOFS_MG_ANY: " *
                     "MG wins 1.2-7.6x from this size (and uses less solver " *
                     "memory)")
    elseif c <= SEMQUAD_AUTO_CONTRAST_MID && dofs >= SEMQUAD_AUTO_DOFS_CONTRAST_100
        return (:mg, "saddle_dofs=$dofs >= $SEMQUAD_AUTO_DOFS_CONTRAST_100 and " *
                     "eta contrast $(_fmt_eta_contrast(c)) <= " *
                     "$SEMQUAD_AUTO_CONTRAST_MID (uniform-tree crossover is " *
                     "below 2.5k dofs)")
    end
    return (:direct, "saddle_dofs=$dofs and eta contrast " *
                     "$(_fmt_eta_contrast(c)) are below the measured MG " *
                     "crossover (high contrast needs 20k dofs; low contrast " *
                     "needs 2k)")
end

const _SEMQUAD_AUTO_SERIAL_WARNED = Ref(false)

"""
    semquad_auto_mg_threads() -> Int

Thread count used by `linear=:auto` when it picks the matrix-free MG:
`min(8, Threads.nthreads())`.  Measured (docs/dev/SOLVER.md §19.2), the threaded
V-cycle is 1.5x faster at 1024 leaves and 2.2x at 4096 leaves; 2 threads are
not worth it on this machine, 4/8 are.  A single-threaded process falls back to
1 and warns once.  This never applies to an explicit `linear=:mg` call, whose
`mg_threads` default stays 1 (bitwise-serial).
"""
function semquad_auto_mg_threads()
    nt = min(SEMQUAD_AUTO_MG_MAX_THREADS, Threads.nthreads())
    if nt == 1 && !_SEMQUAD_AUTO_SERIAL_WARNED[]
        _SEMQUAD_AUTO_SERIAL_WARNED[] = true
        @warn "linear=:auto selected the matrix-free MG, but Julia was " *
              "started with 1 thread: using the serial V-cycle " *
              "(mg_threads=1). Start Julia with --threads=$SEMQUAD_AUTO_MG_MAX_THREADS " *
              "for the measured 1.5-2.2x V-cycle speedup." maxlog=1
    end
    return nt
end

"""
    _semquad_resolve_auto(ss, eta_fun; schur_precond=:mass) -> (choice, mg_threads)

Resolve `linear=:auto` **before** the sparse assembly, so an MG decision never
pays for (or stores) the sparse velocity operator: `saddle_dofs = 2*Nu + Np`
(node counts, no full O(N) sample), `contrast` from the fixed 33x33-grid eta
sample (`semquad_eta_contrast`, ~4-23 us measured), then
[`semquad_choose_linear`](@ref).  On an MG decision it also forces the
calibrated MG configuration (`mg_operator=:matrixfree`, `mg_threads =
semquad_auto_mg_threads()`); `schur_precond=:mass` is required by the MG path,
so an auto request with another preconditioner falls back to `:direct` instead
of throwing.  Records `LAST_SEMSTOKES_AUTO_CHOICE`.
"""
function _semquad_resolve_auto(ss::SemStokes, eta_fun::Function;
                               schur_precond::Symbol=:mass)
    dofs = 2 * ss.su.N + ss.sp.N
    t0 = time_ns()
    contrast = semquad_eta_contrast(ss.su, eta_fun)
    sample_ms = (time_ns() - t0) / 1.0e6
    choice, reason = _semquad_choose_linear_rule(dofs, contrast)
    mg_threads = 0
    if choice === :mg && schur_precond !== :mass
        choice = :direct
        reason *= "; MG needs schur_precond=:mass, falling back to direct"
    elseif choice === :mg
        mg_threads = semquad_auto_mg_threads()
        reason *= "; using mg_operator=:matrixfree, mg_threads=$mg_threads"
    end
    LAST_SEMSTOKES_AUTO_CHOICE[] =
        (choice=choice, reason=reason, saddle_dofs=dofs, contrast=contrast,
         samples=(SEMQUAD_AUTO_CONTRAST_GRID + 1)^2, sample_ms=sample_ms,
         mg_threads=mg_threads)
    return choice, mg_threads
end

