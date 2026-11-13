using Test, MantleSEM, LinearAlgebra, Printf

# G191: production `linear=:auto` path selector (S6).
#
# 1. the decision table: every measured case of the re-calibrated A/B
#    (docs/dev/SOLVER.md §4.2/§4.4, `julia/scripts/bench_auto_crossover.jl`, this
#    session at c7e0f55) maps to the method the measurement says is faster, with
#    the margin policy the rule claims: MG only where the *warm* 1-thread ratio
#    (t_direct/t_mg, the production protocol) is at least 1.5x, direct wherever
#    the *cold* 1-thread ratio is a tie (<=1.25x, ties going to the bitwise
#    path) -- plus the branch boundaries and the monotonicity of the rule;
# 2. the contrast sampler recovers the full-field contrast on a localised eta
#    field, costs <=5 ms (fixed 33x33 grid; measured ~20 us on the production
#    blob closure) and is used from the *actual* eta field of the solve;
# 3. `:auto` -> direct reproduces the explicit direct answer **bitwise**
#    (small, high-contrast case), both in the direct solve API and through
#    `run_semquad_simulation`;
# 4. `:auto` -> MG reproduces the explicit `:mg` answer to <=1e-10 on the
#    uniform L4 tree (10 851 dofs, contrast 1), with the auto thread rule
#    `mg_threads=min(8, Threads.nthreads())`;
# 5. explicit `:direct`/`:mg` never consult or record an auto choice, and an
#    explicit `:mg` call before/after an auto call is bitwise unchanged.

const _G191_Z2 = (x, y) -> 0.0
const _G191_ONE = (x, y) -> 1.0

_choice_is_none() = LAST_SEMSTOKES_AUTO_CHOICE[].choice === :none

function _g191_refined(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g191_uniform(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# Measured table (this session, `bench_auto_crossover.jl`, 1 thread, one process
# per state, direct = `linear=:direct, stokes_scaling=:equilibrated`, MG = the
# shipped defaults at `mg_tol=1e-6`): same-session A/B `t_direct / t_mg`
# (>1 = MG faster), cold = caches cleared, warm = best of three in-tree solves.
# The two high-contrast rows that used to read "parity" and "MG 1.38x slower"
# (blob f19, the 25-frame trajectory) now win for MG and are MG decisions.
const _G191_MEASURED = [
    ("uniform L2 (16 leaves, p=4)",        747,      1.0, :direct, 1.17, 1.71),
    ("uniform L3 (64 leaves, p=4)",      2_803,      1.0, :mg,     1.74, 2.75),
    ("uniform L4 at p=2 (256 leaves)",   2_467,      1.0, :mg,     1.48, 1.91),
    ("uniform L4 (256 leaves, p=4)",    10_851,      1.0, :mg,     2.36, 3.86),
    ("uniform L5 (1 024 leaves, p=4)",  42_691,      1.0, :mg,     3.23, 7.36),
    ("uniform L6 (4 096 leaves, p=4)", 169_347,      1.0, :mg,     3.56, 6.97),
    ("mid-contrast L3 (stratified 50x)", 2_803,     50.0, :mg,     1.74, 2.39),
    ("mid-contrast L4 (stratified 50x)", 10_851,    50.0, :mg,     2.20, 4.04),
    ("blob traj frame 1 (214 leaves)",   9_351,  3_024.0, :direct, 1.09, 1.48),
    ("blob traj frame 14 (628 leaves)", 28_135,  5_087.0, :mg,     1.32, 2.01),
    ("blob f19 (1 024 leaves)",         45_475,  5_087.0, :mg,     1.14, 1.78),
    ("blob f19x4 (2 344 leaves)",       99_559,  5_087.0, :mg,     1.59, 2.41),
    ("blob f19x16 (4 096 leaves)",     174_915,  5_087.0, :mg,     1.96, 2.62),
    ("production F (8 188 leaves, p=2)", 78_720, 5.69e6,  :mg,     1.80, 2.27),
    ("production P1 (10 000 leaves, p=2)", 95_308, 3.29e6, :mg,    1.39, 1.60),
]

@testset "G191 auto path selector" begin
    # ---- 1. decision table ------------------------------------------------
    @testset "decision table" begin
        for (name, dofs, c, want, rcold, rwarm) in _G191_MEASURED
            got = semquad_choose_linear(dofs, c)
            @test got == want
            if want === :mg
                # MG is chosen only where the production (warm) measurement is
                # a clear win; the smallest is P1 at 1.60x.
                @test rwarm >= 1.5
            else
                # ... and direct only where the conservative (cold) measurement
                # is a tie, which goes to the bitwise path.
                @test rcold <= 1.25
            end
            @info "G191 table" case = name dofs = dofs contrast = c choice = got
        end
        # The 25-frame blob trajectory (9 351-45 475 dofs, contrast ~5 087)
        # splits at the any-contrast threshold: the frames below it are the
        # measured tie (cold 1.09-1.17x), the ones above it win 1.32-1.78x.
        for dofs in (9_351, 10_181, 17_135, 19_413)
            @test semquad_choose_linear(dofs, 5_087.0) == :direct
        end
        for dofs in (20_000, 28_135, 45_475)
            @test semquad_choose_linear(dofs, 5_087.0) == :mg
        end
    end

    # ---- 2. rule boundaries, monotonicity and reason consistency ----------
    @testset "thresholds" begin
        @test semquad_choose_linear(2_000, 1.0) == :mg      # low contrast
        @test semquad_choose_linear(1_999, 1.0) == :direct
        @test semquad_choose_linear(2_000, 50.0) == :mg     # mid contrast
        @test semquad_choose_linear(1_999, 50.0) == :direct
        @test semquad_choose_linear(2_000, 100.0) == :mg
        @test semquad_choose_linear(2_000, 100.001) == :direct
        @test semquad_choose_linear(19_999, 1.0e5) == :direct  # high contrast
        @test semquad_choose_linear(20_000, 1.0e5) == :mg
        # the any-contrast branch owns its dof count whatever the contrast is
        @test semquad_choose_linear(20_000, 100.001) == :mg
        @test semquad_choose_linear(0, 1.0) == :direct
        @test semquad_choose_linear(10^9, Inf) == :mg
        @test semquad_choose_linear(10^9, NaN) == :mg
        # the rule is monotone: a bigger tree never turns MG off ...
        for c in (1.0, 50.0, 100.0, 100.001, 1.0e5)
            was = false
            for dofs in (1_000, 2_000, 20_000, 100_000)
                got = semquad_choose_linear(dofs, c) === :mg
                @test !was || got
                was = got
            end
        end
        # ... and a lower contrast never turns it off either.
        for dofs in (1_000, 2_000, 20_000)
            was = true
            for c in (1.0, 50.0, 100.0, 101.0, 1.0e5)
                got = semquad_choose_linear(dofs, c) === :mg
                @test !got || was
                was = got
            end
        end
        # The logged reason comes from the same function as the decision.
        for (dofs, c) in ((10_851, 1.0), (9_351, 5_087.0), (45_475, 5_087.0),
                          (99_559, 5_087.0), (2_000, 50.0), (2_000, 5.0),
                          (100, 1.0), (19_999, 1.0e5), (20_000, 1.0e5))
            choice, reason = MantleSEM._semquad_choose_linear_rule(dofs, c)
            @test choice == semquad_choose_linear(dofs, c)
            @test occursin("saddle_dofs=$dofs", reason)
            @test !isempty(reason)
        end
    end

    # ---- 3. contrast sampler: accuracy, fixed cost, actual eta field ------
    qt4 = _g191_uniform(4)
    ss4 = make_semstokes(qt4, 4)
    @test 2 * ss4.su.N + ss4.sp.N == 10_851           # table convention
    @test semquad_eta_contrast(ss4.su, _G191_ONE) == 1.0

    # A localised high-viscosity blob: the contrast is set by a small region,
    # exactly the situation in which a strided node/leaf subset under-samples
    # (measured up to 5x on the real frame-19 field, §20).  The fixed grid must
    # stay within 2x of the full quadrature-field contrast.
    eta_blob = (x, y) -> exp(8.0 * exp(-((x - 0.37)^2 + (y - 0.61)^2) / 0.01))
    op4 = semquad_kop(ss4, qt4)
    ef4 = semquad_eta_field(op4, eta_blob)
    c_full = maximum(ef4) / minimum(ef4)
    c_grid = semquad_eta_contrast(ss4.su, eta_blob)
    @test 0.5 * c_full <= c_grid <= 2.0 * c_full
    @test c_grid > 100.0                                # high-contrast band

    # Fixed cost: 1 089 eta evaluations, independent of the tree size.
    tsamp = minimum([@elapsed semquad_eta_contrast(ss4.su, eta_blob) for _ in 1:3])
    @test tsamp <= 5.0e-3                              # "a few ms" upper bound
    @info "G191 contrast sampler" grid = c_grid full_quadrature = c_full t_s = tsamp points = (MantleSEM.SEMQUAD_AUTO_CONTRAST_GRID + 1)^2

    # ---- 4. :auto -> direct is bitwise the direct answer -------------------
    # Small 2b L2 tree, but a ~8 100x viscosity contrast (exp(9x)), so this also
    # checks that a high-contrast field is sampled from the actual eta_fun.
    qt_s = _g191_refined(2; cx=5, cy=2)
    ss_s = make_semstokes(qt_s, 2)
    eta_s = (x, y) -> exp(9.0 * (x / qt_s.Lx))
    fz_s = (x, y) -> sin(pi * x / qt_s.Lx) * sin(pi * y)
    res_d = solve_semquad_stokes_var(ss_s, qt_s, eta_s, _G191_Z2, fz_s,
                                     _G191_Z2, _G191_Z2; bc=:free)
    ud = res_d[1]; pd = res_d[2]
    rd = LAST_SEMSTOKES_RESIDUAL[]
    @test _choice_is_none()                              # explicit direct
    @test rd < 1.0e-9
    res_a = solve_semquad_stokes_var(ss_s, qt_s, eta_s, _G191_Z2, fz_s,
                                     _G191_Z2, _G191_Z2; bc=:free,
                                     linear=:auto)
    ua = res_a[1]; pa = res_a[2]
    ch = LAST_SEMSTOKES_AUTO_CHOICE[]
    @test ch.choice === :direct
    @test ch.saddle_dofs == 2 * ss_s.su.N + ss_s.sp.N
    @test ch.saddle_dofs < MantleSEM.SEMQUAD_AUTO_DOFS_MG_ANY
    @test 100.0 < ch.contrast < 20_000.0          # exp(9) ~ 8 103, high band
    @test ch.mg_threads == 0
    @test ua == ud && pa == pd                    # bitwise
    @test LAST_SEMSTOKES_RESIDUAL[] == rd
    # ... and an explicit direct call after the auto call is unchanged too.
    res_d2 = solve_semquad_stokes_var(ss_s, qt_s, eta_s, _G191_Z2, fz_s,
                                      _G191_Z2, _G191_Z2; bc=:free)
    @test _choice_is_none()
    @test res_d2[1] == ud && res_d2[2] == pd

    # ---- 5. run_semquad_simulation forwards :auto, bitwise direct ---------
    pr2b = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                  Twall_top=0.0, Twall_bot=1.0, bc=:free)
    function _g191_run(lin)
        qt = _g191_uniform(3)
        ss = make_semstokes(qt, 2)
        th = [0.01 * cos(pi * x) * sin(pi * y) for (x, y) in ss.su.coords]
        for (i, (x, y)) in enumerate(ss.su.coords)
            (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th[i] = 0.0)
        end
        choices = Symbol[]
        cb = st -> push!(choices, LAST_SEMSTOKES_AUTO_CHOICE[].choice)
        st = run_semquad_simulation(qt, pr2b, th; p=2, tstop=4.0e-4,
                                    dt0=2.0e-4, maxsteps=2, linear=lin,
                                    callback=cb, verbose=false)
        return st, choices
    end
    sd, cd = _g191_run(:direct)
    @test all(==(:none), cd)
    sa, ca = _g191_run(:auto)
    @test all(==(:direct), ca)
    @test sa.steps == sd.steps
    @test sa.theta == sd.theta
    @test sa.ux == sd.ux && sa.uz == sd.uz && sa.pv == sd.pv
    @test sa.history == sd.history

    # ---- 6. :auto -> MG on the uniform L4 tree vs explicit :mg ------------
    fz4 = (x, y) -> sin(pi * x) * sin(pi * y)
    res_mg = solve_semquad_stokes_var(ss4, qt4, _G191_ONE, _G191_Z2, fz4,
                                      _G191_Z2, _G191_Z2; bc=:free,
                                      linear=:mg, mg_operator=:matrixfree,
                                      mg_threads=1)
    u_mg = res_mg[1]; p_mg = res_mg[2]
    r_mg = LAST_SEMSTOKES_RESIDUAL[]
    @test _choice_is_none()
    @test r_mg < 1.0e-9
    res_au = solve_semquad_stokes_var(ss4, qt4, _G191_ONE, _G191_Z2, fz4,
                                      _G191_Z2, _G191_Z2; bc=:free,
                                      linear=:auto)
    u_au = res_au[1]; p_au = res_au[2]
    ch = LAST_SEMSTOKES_AUTO_CHOICE[]
    @test ch.choice === :mg
    @test ch.saddle_dofs == 10_851
    @test ch.contrast == 1.0
    @test ch.mg_threads == min(8, Threads.nthreads())
    @test ch.sample_ms <= 5.0                       # measured ~0.005-0.02 ms
    @test norm(u_au - u_mg) / max(norm(u_mg), 1.0) <= 1.0e-10
    @test norm(p_au - p_mg) / max(norm(p_mg), 1.0) <= 1.0e-10
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-9
    bitwise_mg = (u_au == u_mg && p_au == p_mg)
    # An explicit :mg call after the auto call is still the same solve: the
    # selector does not leak its mg_operator/mg_threads into explicit calls.
    res_mg2 = solve_semquad_stokes_var(ss4, qt4, _G191_ONE, _G191_Z2, fz4,
                                       _G191_Z2, _G191_Z2; bc=:free,
                                       linear=:mg, mg_operator=:matrixfree,
                                       mg_threads=1)
    @test _choice_is_none()
    @test res_mg2[1] == u_mg && res_mg2[2] == p_mg
    @info "G191 auto MG" dofs = ch.saddle_dofs contrast = ch.contrast threads = ch.mg_threads sample_ms = ch.sample_ms rel_u = norm(u_au - u_mg) / norm(u_mg) bitwise = bitwise_mg iterations = LAST_SEMSTOKES_SCHUR_ITERS[]
end
