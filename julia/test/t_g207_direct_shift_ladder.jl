using Test, MantleSEM, LinearAlgebra, SparseArrays

# G207: the direct path's diagonal-shift ladder (docs/dev/CASE_REALISTIC.md §13).
#
# The direct reduced-saddle solve used to retry *once* with a 1e-12 relative
# diagonal shift and then accept whatever came out.  On the 8192-leaf realistic
# tree that left 1.3e-3 against the card's 1e-3 and aborted the long run at step
# 6130.  `_shift_ladder_solve` keeps that rung first, escalates to 1e-10 then
# 1e-8 while the residual misses `stokes_accept_tol`, and finally refines with
# the last factorisation.  This gate pins:
#
#   (1) the control flow (which rungs, in which order, when it stops, when the
#       refinement runs) on a prescribed residual sequence -- deterministic,
#   (2) the semantics on a genuinely ill-conditioned saddle (the G164 wall tree,
#       whose unshifted LU has a growth factor ~1e15),
#   (3) `accept_tol = Inf` and a *healthy* system are bit-identical to the
#       historical behaviour, and the records are reset by every solve.

@testset "G207 direct shift ladder" begin
    # the helper is internal to the module (it takes the reduced saddle, not a
    # Stokes problem); the gate drives it directly for the control-flow tests
    ladder = MantleSEM._shift_ladder_solve
    shifts = (1.0e-12, 1.0e-10, 1.0e-8)

    # ---------------------------------------------------------------- (1) ----
    # Control flow on a prescribed residual sequence.  `_shift_ladder_solve`
    # accepts `resid`, so the decisions can be tested without a pathological
    # matrix: the solve itself is a deterministic 1-D Laplacian, only the
    # measured residual is prescribed.
    n = 24
    Ssyn = sparse([1:n; 1:n-1], [1:n; 2:n],
                  [fill(2.0, n); fill(-1.0, n - 1)], n, n)
    bsyn = ones(n)
    function prescribed(seq)
        k = Ref(0)
        return x -> (k[] += 1; seq[min(k[], length(seq))])
    end

    # healthy: one factorisation, nothing recorded
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([9.0e-9]))
    @test rungs === nothing && sh == 0.0 && nref == 0
    @test norm(Ssyn * x - bsyn) / max(norm(bsyn), 1.0) < 1.0e-12

    # the first rung meets the gate -> accepted, no escalation
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([5.0e-3, 5.0e-4]))
    @test rungs !== nothing && length(rungs) == 2
    @test rungs[1][1] == 0.0 && rungs[2][1] == 1.0e-12
    @test sh == 1.0e-12 && nref == 0 && r == 5.0e-4

    # the first rung misses -> escalate one step (1e-12 then 1e-10), stop there
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([5.0e-3, 3.0e-3, 2.0e-4]))
    @test length(rungs) == 3
    @test [g[1] for g in rungs] == [0.0, 1.0e-12, 1.0e-10]
    @test sh == 1.0e-10 && nref == 0 && r == 2.0e-4

    # every rung misses -> all three are tried, then the refinement runs and
    # the improved iterate is returned
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([5.0e-3, 3.0e-3, 2.0e-3,
                                                     1.5e-3, 4.0e-4]))
    @test length(rungs) == 5
    @test [g[1] for g in rungs[1:4]] == [0.0, 1.0e-12, 1.0e-10, 1.0e-8]
    @test rungs[5][1] === :refine
    @test r == 4.0e-4

    # the best rung is kept when the refinement does not improve on it -- and,
    # because that best still misses the gate, the last-resort tiny shifts run
    # after it (G212, docs/dev/CASE_REALISTIC.md §17.4).  The prescribed sequence is
    # exhausted, so both tiny rungs repeat 9.0e-3 and the 1.5e-3 rung still wins.
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([5.0e-3, 3.0e-3, 2.0e-3,
                                                     1.5e-3, 9.0e-3]))
    @test r == 1.5e-3 && sh == 1.0e-8
    @test rungs[5][1] === :refine && rungs[5][2] == 9.0e-3
    @test [g[1] for g in rungs[6:7]] == [1.0e-14, 1.0e-13]
    @test LAST_SEMSTOKES_LADDER_PHASE[] === :fail

    # the unshifted solve already passes and a worse shift must not replace it:
    # escalate no further, return the unshifted answer (measured shape on the
    # step-6130 realistic state: unshifted 4.0e-4, 1e-12 rung 1.6e-3)
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([5.0e-4, 2.0e-3, 1.0e-9]))
    @test length(rungs) == 2 && sh == 0.0 && nref == 0 && r == 5.0e-4
    @test rungs[2][1] == 1.0e-12 && rungs[2][2] == 2.0e-3

    # ... but a *better* shift is taken even when the unshifted one passes
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([5.0e-4, 3.0e-11]))
    @test length(rungs) == 2 && sh == 1.0e-12 && r == 3.0e-11

    # `accept_tol = Inf` (the default): the first rung is accepted
    # unconditionally, which is the historical behaviour bit for bit
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; resid=prescribed([5.0e-3, 7.0e-3]))
    @test length(rungs) == 2 && sh == 1.0e-12 && nref == 0 && r == 7.0e-3

    # a non-finite residual is a failure, not "healthy" (`NaN > tol` is false):
    # it is recorded as Inf, the ladder runs, and the finite rung is returned
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([NaN, 2.0e-3]))
    @test rungs !== nothing && length(rungs) == 5
    @test rungs[1][1] == 0.0 && rungs[1][2] == Inf
    @test isfinite(r) && r == 2.0e-3 && sh == 1.0e-12
    @test all(isfinite, x)

    # the rung history is exactly the ladder constant, in order
    @test MantleSEM.SEMSTOKES_SHIFT_LADDER == shifts

    # ---------------------------------------------------------------- (2) ----
    # A genuinely ill-conditioned saddle: the G164 wall-refined free-slip tree,
    # whose *unshifted* sparse LU has a growth factor ~1e15 and a backward
    # residual ~1e3.
    function wall_tree(base, wall)
        qt = Quadtree(max(base, wall); cells_x=5, cells_y=2)
        for _ in 1:base, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        refine_wall_to_min!(qt, wall; strip=0.1)
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
    fz = (x, y) -> pr.Ra * MantleSEM._semquad_eval_at(sq, theta, x, y, leaves)
    solve_wall(; kw...) = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2;
                                                   bc=:free, linear=:direct, kw...)

    # (2a) the historical single retry: `Inf` accepts the 1e-12 rung
    u_inf, _, _, _, _ = solve_wall()
    r_inf = LAST_SEMSTOKES_RESIDUAL[]
    @test LAST_SEMSTOKES_SHIFT[] == 1.0e-12
    @test LAST_SEMSTOKES_REFINE_ITERS[] == 0
    lad = LAST_SEMSTOKES_LADDER[]
    @test lad !== nothing && length(lad) == 2 && lad[1][1] == 0.0
    @test lad[2][1] == 1.0e-12
    # the unshifted LU really is bad here: measured 1.6e-4 (the internal 1e-8
    # gate fires long before the driver's 1e-3 one would)
    @test lad[1][2] > 1.0e-6
    @test r_inf < 1.0e-5              # the rung fixes it (G164 measured 3e-11)

    # (2b) a real gate (the driver's stokes_tol) accepts the same rung and gives
    # the *same* velocity: the escalation only fires when the gate fails
    u_gate, _, _, _, _ = solve_wall(stokes_accept_tol=1.0e-3)
    @test u_gate == u_inf
    @test LAST_SEMSTOKES_SHIFT[] == 1.0e-12
    @test LAST_SEMSTOKES_REFINE_ITERS[] == 0
    @test length(LAST_SEMSTOKES_LADDER[]) == 2
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-3

    # (2c) an unreachable gate (1e-12 against a 3e-11 rung) escalates through
    # both larger shifts and then refines; the returned residual is the best
    # the ladder saw and the last entry is the refinement
    u_tight, _, _, _, _ = solve_wall(stokes_accept_tol=1.0e-12)
    lad = LAST_SEMSTOKES_LADDER[]
    @test length(lad) == 5
    @test [g[1] for g in lad[1:4]] == [0.0, 1.0e-12, 1.0e-10, 1.0e-8]
    @test lad[5][1] === :refine
    @test LAST_SEMSTOKES_REFINE_ITERS[] >= 1
    @test LAST_SEMSTOKES_RESIDUAL[] <= minimum(g[2] for g in lad[1:4])
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9   # refinement removes the shift bias
    @test all(isfinite, u_tight)

    # ---------------------------------------------------------------- (3) ----
    # Healthy system: a uniform mesh takes the unshifted path, and the new
    # keyword changes nothing there -- bitwise.
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
    fzu = (x, y) -> pr.Ra * MantleSEM._semquad_eval_at(squ, thu, x, y, lvu)
    uu0, pp0, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                                 bc=:free, linear=:direct)
    @test !LAST_SEMSTOKES_REGULARIZED[]
    @test LAST_SEMSTOKES_SHIFT[] == 0.0
    @test LAST_SEMSTOKES_LADDER[] === nothing
    @test LAST_SEMSTOKES_REFINE_ITERS[] == 0
    r_healthy = LAST_SEMSTOKES_RESIDUAL[]
    @test r_healthy < 1.0e-10
    uu1, pp1, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                                 bc=:free, linear=:direct,
                                                 stokes_accept_tol=1.0e-3)
    @test uu1 == uu0                 # bitwise: the ladder never fired
    @test pp1 == pp0
    @test LAST_SEMSTOKES_RESIDUAL[] == r_healthy
    @test LAST_SEMSTOKES_LADDER[] === nothing

    # the split residual agrees with the scalar one, and the wall tree's is
    # dominated by the continuity block (the near-null mode is a pressure mode)
    sp = LAST_SEMSTOKES_RESID_SPLIT[]
    @test max(sp.vel, sp.div) == LAST_SEMSTOKES_RESIDUAL[]
    solve_wall()
    sp = LAST_SEMSTOKES_RESID_SPLIT[]
    @test sp.vel > 0.0 && sp.div > 0.0
    @test max(sp.vel, sp.div) == LAST_SEMSTOKES_RESIDUAL[]

    # `linear=:mg` resets the direct-path records and is not touched by the
    # ladder (the records describe the last solve only).  `LAST_SEMSTOKES_
    # REGULARIZED` is deliberately *not* in this list: it is only written by the
    # `:direct`/`:schur_cg`/`:lagged` branches (its documented contract), so
    # after an `:mg` solve it still describes the previous direct solve.
    solve_wall()
    @test LAST_SEMSTOKES_SHIFT[] > 0.0
    umg, _, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                               bc=:free, linear=:mg,
                                               mg_maxit=40, mg_restart=40)
    @test LAST_SEMSTOKES_SHIFT[] == 0.0
    @test LAST_SEMSTOKES_LADDER[] === nothing
    @test LAST_SEMSTOKES_REFINE_ITERS[] == 0
    @test all(isfinite, umg)
end
