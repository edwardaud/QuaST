# G212: the last-resort tiny shifts of the direct path — the lost-solve fallback.
using Test, QuaST, LinearAlgebra, SparseArrays

# G212: the last-resort tiny-shift fallback of the direct path
# (docs/dev/CASE_REALISTIC.md §17.4-§17.7).
#
# `_shift_ladder_solve` used to run the tiny shifts (1e-14 then 1e-13) only
# when the *unshifted* residual was above `SEMSTOKES_LOST_TOL = 1.0`.  The G run
# died at step 7641 with an unshifted residual of **0.9654** -- 3.5 % below the
# trigger -- while the historical ladder reached only 4.465e-3 against the
# card's 1e-3, and the same state class measures 3.7-3.9e-7 at a 1e-14 shift.
# The fix is rule 7: after the historical ladder (1e-12/1e-10/1e-8) *and* its
# four-iteration refinement, if the best answer still misses `accept_tol` and
# the priority tiny phase did not already run, the tiny shifts are tried as a
# last resort; the best answer (refinement included) is kept and the solve still
# fails if they all miss.
#
# This gate pins:
#   (1) the exact step-7641 ladder via a prescribed residual sequence:
#       r = 0.9654 < LOST_TOL skips the priority phase, the historical ladder and
#       the refinement fail, and the fallback returns the 1e-14 answer;
#   (2) that answer is the actual tiny-shift solve of the matrix, bit for bit;
#   (3) the fallback does not fire when a rung (or the refinement) already
#       passes, and it cannot discard a better refinement result;
#   (4) a healthy solve is a single unshifted factorisation, bitwise unchanged,
#       and `accept_tol = Inf` still is the historical single 1e-12 retry on the
#       G164 wall tree (bit for bit);
#   (5) on that real wall tree a failing-class ladder (unshifted 0.9654 < 1.0,
#       prescribed historical rungs) succeeds through the fallback and its answer
#       is the true 1e-14 solve; a deliberately tightened gate (1e-12) still
#       fails loudly -- the fallback never silently accepts a bad answer;
#   (6) the phase is recorded by `LAST_SEMSTOKES_LADDER_PHASE` for every path
#       (`:healthy`, `:ladder`, `:refine`, `:fallback`, `:fail`) and reset by a
#       non-direct solve.

@testset "G212 last-resort tiny shifts" begin
    ladder = QuaST._shift_ladder_solve
    # The historical ladder and the tiny rungs are public constants.
    @test QuaST.SEMSTOKES_SHIFT_LADDER == (1.0e-12, 1.0e-10, 1.0e-8)
    @test QuaST.SEMSTOKES_TINY_SHIFTS == (1.0e-14, 1.0e-13)
    @test QuaST.SEMSTOKES_LOST_TOL == 1.0

    # ---------------------------------------------------------------- (1) ----
    # Control flow on the G step-7641 ladder record (docs §17, verbatim):
    #   rungs 0:9.654e-01 1e-12:1.561e-02 1e-10:8.245e-03 1e-08:6.279
    #   refine:4.465e-03, and the measured 1e-14 answer 3.718e-07.
    n = 24
    Ssyn = sparse([1:n; 1:n-1], [1:n; 2:n],
                  [fill(2.0, n); fill(-1.0, n - 1)], n, n)
    bsyn = ones(n)
    function prescribed(seq)
        k = Ref(0)
        return x -> (k[] += 1; seq[min(k[], length(seq))])
    end
    gseq = [9.654e-01, 1.561e-02, 8.245e-03, 6.279e+00, 4.465e-03, 3.7179e-07]
    @test gseq[1] < QuaST.SEMSTOKES_LOST_TOL    # the trigger miss
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed(gseq))
    @test [g[1] for g in rungs] ==
          [0.0, 1.0e-12, 1.0e-10, 1.0e-8, :refine, 1.0e-14]
    @test rungs[5][2] == 4.465e-03
    @test rungs[end] == (1.0e-14, 3.7179e-07)
    @test r == 3.7179e-07 && sh == 1.0e-14
    @test nref >= 1                       # the refinement really ran first
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :fallback
    # the returned iterate *is* the 1e-14 rung: bit for bit
    dmax = maximum(abs, diag(Ssyn)) + 1.0
    @test x == lu(Ssyn + (1.0e-14 * dmax) * I) \ bsyn

    # ---------------------------------------------------------------- (2) ----
    # The historical rungs are unchanged when the fallback is not needed; only
    # then is the tiny shift tried.  (a) a rung passes:
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([5.0e-3, 4.0e-4]))
    @test length(rungs) == 2 && [g[1] for g in rungs] == [0.0, 1.0e-12]
    @test r == 4.0e-4 && sh == 1.0e-12 && nref == 0
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :ladder
    # (b) the refinement passes:
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([5.0e-3, 3.0e-3, 2.0e-3,
                                                     1.5e-3, 4.0e-4]))
    @test length(rungs) == 5 && rungs[5][1] === :refine
    @test r == 4.0e-4
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :refine
    # (c) a marginal but accepted unshifted answer is returned untouched:
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([3.98e-4, 1.32e-3]))
    @test length(rungs) == 2 && r == 3.98e-4 && sh == 0.0
    @test x == (Ssyn \ bsyn)
    # (d) the best answer *including the refinement* is kept: a tiny rung that
    # does not beat the refinement is recorded and the solve still fails.
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-4,
                                   resid=prescribed([9.654e-1, 1.561e-2,
                                                     8.245e-3, 6.279,
                                                     4.0e-4, 9.0e-4, 9.0e-4]))
    @test length(rungs) == 7
    @test [g[1] for g in rungs[6:7]] == [1.0e-14, 1.0e-13]
    @test r == 4.0e-4 && sh == 1.0e-10     # the refinement's preconditioner
    @test r > 1.0e-4
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :fail

    # ---------------------------------------------------------------- (3) ----
    # A healthy solve: one factorisation, the unshifted answer bit for bit,
    # nothing recorded.
    kref = Ref(0)
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=x -> (kref[] += 1; 9.0e-9))
    @test kref[] == 1
    @test x == (Ssyn \ bsyn) && r == 9.0e-9
    @test rungs === nothing && sh == 0.0 && nref == 0
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :healthy

    # ---------------------------------------------------------------- (4) ----
    # The phase switches: `tiny_shifts = ()` disables both tiny phases (the
    # historical ladder and its failure are back); `lost_tol = Inf` only
    # suppresses the *priority* phase, the last resort still runs (this is how
    # the failing-class replay is forced).
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed(gseq), tiny_shifts=())
    @test length(rungs) == 5 && rungs[end][1] === :refine
    @test r == 4.465e-3
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :fail
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed(gseq), lost_tol=Inf)
    @test r == 3.7179e-7 && sh == 1.0e-14
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :fallback

    # ---------------------------------------------------------------- (5) ----
    # The real SEM saddle: the G164 wall-refined free-slip tree.  Its unshifted
    # residual is 1.6e-4 (finite, *below* LOST_TOL) and its 1e-12 rung is exact,
    # so the default path never reaches the fallback.
    function wall_tree(base, wall)
        qt = Quadtree(max(base, wall); cells_x=5, cells_y=2)
        for _ in 1:base, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        refine_wall_to_min!(qt, wall; geom=CartesianBox(qt), strip=0.1)
        balance!(qt)
        return qt
    end
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=9.70406052783923433,
                eta_c=4.15888308335967186, Twall_top=0.0, Twall_bot=1.0,
                bc=:free)
    z2 = (x, y) -> 0.0
    qt = wall_tree(4, 5)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    theta = [0.5 * sin(pi * x) * sin(pi * y) for (x, y) in sq.coords]
    for g in 1:sq.N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end
    leaves = Set(qt.leaves)
    eta_fun = (x, y) -> semquad_eta_at(sq, theta, x, y, pr, leaves)
    fz = (x, y) -> pr.Ra * QuaST._semquad_eval_at(sq, theta, x, y, leaves)
    solve_wall(; kw...) = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2;
                                                   bc=:free, linear=:direct, kw...)

    # (5a) `accept_tol = Inf` is the historical single 1e-12 retry, bit for bit
    u_inf, _, _, _, _ = solve_wall()
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :ladder
    @test LAST_SEMSTOKES_SHIFT[] == 1.0e-12
    @test LAST_SEMSTOKES_REFINE_ITERS[] == 0
    @test length(LAST_SEMSTOKES_LADDER[]) == 2
    @test all(g[1] != 1.0e-14 for g in LAST_SEMSTOKES_LADDER[])
    # (5b) a finite gate that the 1e-12 rung passes: the same answer, bitwise
    u_gate, _, _, _, _ = solve_wall(stokes_accept_tol=1.0e-3)
    @test u_gate == u_inf
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :ladder
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-3
    # (5c) ... and it is exactly the explicit 1e-12 solve of the same saddle
    sys = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2; bc=:free,
                                   linear=:direct, assemble_only=true)
    nf = size(sys.Kred, 1); nfp = size(sys.Bs, 1)
    S = [sys.Kred sys.Bt; sys.Bs spzeros(Float64, nfp, nfp)]
    rhs = vcat(sys.fr, sys.gr)
    dmaxS = maximum(abs, diag(S)) + 1.0
    sol12 = lu(S + (1.0e-12 * dmaxS) * I) \ rhs
    @test u_inf == sys.u0 + sys.Cu * sol12[1:nf]

    # (5d) the failing-class state on the real saddle: the unshifted residual is
    # reported as the G failing 0.9654 (< LOST_TOL, so the priority phase is
    # skipped), the historical rungs as the failing record -- and the tiny rung
    # is then a *real* solve.  It passes the 1e-6 gate and is the true 1e-14
    # answer, bit for bit.
    rden = max(norm(sys.fr), norm(sys.gr), 1.0)
    relt = function (x)
        rv = norm(sys.Kred * x[1:nf] + sys.Bt * x[(nf + 1):end] - sys.fr) / rden
        rd = norm(sys.Bs * x[1:nf] - sys.gr) / rden
        max(rv, rd)
    end
    sev = [9.654e-1, 1.561e-2, 8.245e-3, 6.279, 4.465e-3]
    function failing_class(seq)
        k = Ref(0)
        return function (x)
            k[] += 1
            k[] <= length(seq) ? seq[k[]] : relt(x)
        end
    end
    x, r, sh, nref, rungs = ladder(S, rhs; accept_tol=1.0e-6,
                                   resid=failing_class(sev))
    @test [g[1] for g in rungs[1:5]] == [0.0, 1.0e-12, 1.0e-10, 1.0e-8, :refine]
    @test rungs[end][1] == 1.0e-14
    @test sh == 1.0e-14 && r < 1.0e-6
    @test relt(x) == r == rungs[end][2]       # a genuine residual, not prescribed
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :fallback
    @test x == lu(S + (1.0e-14 * dmaxS) * I) \ rhs

    # (5e) a deliberately tightened gate (1e-12) on the same failing class: even
    # the fallback misses, the *best* answer (the tiny rung, which improved on
    # the refinement) is returned with its true residual, and the phase says the
    # solve failed.  No silent acceptance of a bad answer.
    x, r, sh, nref, rungs = ladder(S, rhs; accept_tol=1.0e-12,
                                   resid=failing_class(sev))
    @test r > 1.0e-12
    @test sh == 1.0e-14
    @test [g[1] for g in rungs[6:7]] == [1.0e-14, 1.0e-13]
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :fail
    @test x == lu(S + (1.0e-14 * dmaxS) * I) \ rhs

    # ---------------------------------------------------------------- (6) ----
    # A healthy real solve: a uniform mesh takes the unshifted path, the new
    # phase changes nothing there, and a finite gate is bitwise identical.
    qu = Quadtree(4)
    for _ in 1:2, lf in collect(qu.leaves)
        refine!(qu, lf)
    end
    balance!(qu)
    ssu = make_semstokes(qu, 2)
    squ = ssu.su
    thu = [0.5 * sin(pi * x) * sin(pi * y) for (x, y) in squ.coords]
    lvu = Set(qu.leaves)
    etau = (x, y) -> semquad_eta_at(squ, thu, x, y, pr, lvu)
    fzu = (x, y) -> pr.Ra * QuaST._semquad_eval_at(squ, thu, x, y, lvu)
    uu0, pp0, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                                 bc=:free, linear=:direct)
    @test QuaST.LAST_SEMSTOKES_LADDER_PHASE[] === :healthy
    @test LAST_SEMSTOKES_LADDER[] === nothing
    @test LAST_SEMSTOKES_SHIFT[] == 0.0
    r_healthy = LAST_SEMSTOKES_RESIDUAL[]
    uu1, pp1, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                                 bc=:free, linear=:direct,
                                                 stokes_accept_tol=1.0e-3)
    @test uu1 == uu0 && pp1 == pp0
    @test LAST_SEMSTOKES_RESIDUAL[] == r_healthy
    @test LAST_SEMSTOKES_LADDER[] === nothing
    # a non-direct solve resets the phase record
    umg, _, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                               bc=:free, linear=:mg,
                                               mg_maxit=40, mg_restart=40)
    @test all(isfinite, umg)
    @test LAST_SEMSTOKES_LADDER_PHASE[] === nothing
    @test LAST_SEMSTOKES_LADDER[] === nothing
end
