using Test, MantleSEM, LinearAlgebra, SparseArrays

# G209: the near-null pressure modes of the wall-refined realistic saddle, and
# the tiny-shift rule of the direct path (docs/dev/CASE_REALISTIC.md §13.10).
#
# What §13.10 measured on the 8260-leaf realistic tree (the step-6600/6604
# states): the *unshifted* reduced-saddle solve has a growth factor 1.2e15 and
# returns a solution of norm 8.9e11 with a relative residual 4.7e7, while the
# same operator solved with a 1e-14 relative diagonal shift is stable (growth
# 1e-4) and accurate (residual 5.2e-7).  The near-null space is **not** the
# constant pressure (that direction's residual is 4e-9 relative and harmless):
# it is a cluster of pressure modes on the level-5 wall strip -- an inf-sup
# deficiency of the nonconforming wall strip -- with |eigenvalue| down to
# 4.6e-10 against sigma_max = 1.2e5.  The diagonal shift has two jobs there: it
# makes the structurally zero pressure block pivottable, and it biases the
# answer by `shift*|x|`.  A shift far above the cluster (the historical 1e-12)
# therefore leaves a 1e-4 bias; a shift at the cluster's own scale leaves 5e-7.
#
# This gate pins:
#   (1) the control flow of the rule on a prescribed residual sequence: it
#       fires only for a *lost* unshifted solve (r > SEMSTOKES_LOST_TOL), tries
#       the tiny shifts first, keeps the best, and falls through to the
#       historical ladder when none of them improves;
#   (2) a synthetic saddle with a *known* near-null pressure direction, where
#       the closed form p = (B f - g)/(2 nb^2 - delta) shows the historical rung
#       destroying the pressure (resonance with the Schur complement) and the
#       tiny rung reproducing it;
#   (3) a real SEM saddle (the G164 wall-refined tree): with the default trigger
#       the answer is bit-identical to the historical one, and a forced run
#       (stokes_lost_tol = 0) can only improve on it;
#   (4) the realistic state itself (step 6600/6604, 8260 leaves): 4.7e7 -> 5e-7
#       at shift 1e-14.  Data-dependent: point `G209_STATE` at the checkpoint to
#       enable it (checkpoints are not committed, so this test is skipped in a
#       bare checkout);
#   (5) `linear=:mg` is untouched.

"""
    probe_realistic(path)

Assemble the exact reduced saddle of a realistic checkpoint with the
`assemble_only` recipe of `out_julia/agent/probe_saddle.jl` and run the direct
path's ladder on it with the driver's gate.  Returns the measured quantities the
step-6600 assertions quote.
"""
function probe_realistic(path)
    RA = 1.0e5; HHEAT = 10.0
    COMP_AD = 1.6240183119e5; COMP_AS = 2.6796419238e-1
    B410 = 180.0 / (3300.0 * 3.0e-5 * 1000.0)
    B660 = 300.0 / (3300.0 * 3.0e-5 * 1000.0)
    scale = (rho_ref=3300.0, cp=1000.0, dT=1000.0, g=9.81, d=2.9e6, T_surface=0.5)
    ph = [PhaseTransition(0.87, 0.025, 0.032, 0.5, 180.0, B410),
          PhaseTransition(0.78, 0.025, -0.032, 0.5, 300.0, B660)]
    ph = per_phase_latent(Params(Ra=RA, H=HHEAT), ph; scale...)
    pr = Params(Ra=RA, H=HHEAT, eta_law=:composite, comp_rule=:isostress,
                comp_Ad=COMP_AD, comp_As=COMP_AS, comp_Ed=12.0, comp_Es=17.0,
                comp_Vd=0.0, comp_Vs=0.0, comp_n=3.0, comp_Tcut=0.0,
                comp_eps_min=1.0e-12, eta_T0=0.5, eta_min=1.0e-4,
                eta_max=1.0e4, bc=:free, latent_L=0.11, latent_mode=:per_phase,
                yield_c=Inf, yield_phi=0.0, yield_taumax=Inf, yield_c_grad=0.0,
                yield_mu_grad=0.0, weak_f=1.0, weak_eps_ref=0.0,
                rho_heating=false, rho_cp=false, Di=0.0,
                heating_adiabatic=false, heating_dissipation=false,
                refstate=nothing, deviatoric=false)
    st = load_semquad_state(path)
    qt = st.qt
    ss = make_semstokes(qt, st.p)
    sq = ss.su
    leaves = Set(sq.leaves)
    Tfun = (x, y) -> MantleSEM._semquad_eval_at(sq, st.theta, x, y, leaves)
    eta_fun = (x, y) -> begin
        T = Tbar(y, pr.H) + MantleSEM._semquad_eval_at(sq, st.theta, x, y, leaves)
        ed = semquad_strain_rate_at(sq, st.ux, st.uz, x, y, leaves)
        eta_composite_at(T, ed, y, pr)[1]
    end
    fz = (x, y) -> pr.Ra * Tfun(x, y) + semquad_phase_buoyancy(pr, ph, y, Tfun(x, y))
    z2 = (x, y) -> 0.0
    sys = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2; bc=:free,
                                   linear=:direct, assemble_only=true,
                                   mass_form=:boussinesq)
    nf = size(sys.Kred, 1); nfp = size(sys.Bs, 1)
    S = [sys.Kred sys.Bt; sys.Bs spzeros(Float64, nfp, nfp)]
    rhs = vcat(sys.fr, sys.gr)
    rden = max(norm(sys.fr), norm(sys.gr), 1.0)
    rel = function (x)
        rv = norm(sys.Kred * x[1:nf] + sys.Bt * x[(nf + 1):end] - sys.fr) / rden
        rd = norm(sys.Bs * x[1:nf] - sys.gr) / rden
        max(rv, rd)
    end
    x0 = S \ rhs
    r0 = rel(x0)
    x, r, sh, nref, _ = MantleSEM._shift_ladder_solve(S, rhs; accept_tol=1.0e-3,
                                                      bad_tol=1.0e-8, resid=rel)
    return (unshifted_res=r0, res=r, shift=sh, nref=nref, norms=norm(x),
            du=max(norm(x[1:nf] - x0[1:nf]) / max(norm(x0[1:nf]), 1.0), 0.0))
end

@testset "G209 near-null pressure modes and the tiny-shift rule" begin
    ladder = MantleSEM._shift_ladder_solve
    @test MantleSEM.SEMSTOKES_TINY_SHIFTS == (1.0e-14, 1.0e-13)
    @test MantleSEM.SEMSTOKES_LOST_TOL == 1.0
    @test MantleSEM.SEMSTOKES_SHIFT_LADDER == (1.0e-12, 1.0e-10, 1.0e-8)

    # ---------------------------------------------------------------- (1) ----
    # Control flow on a prescribed residual sequence (the solve is a
    # deterministic SPD tridiagonal system; only the *measured* residual is
    # prescribed, as in G207).
    n = 24
    Ssyn = sparse([1:n; 1:n-1], [1:n; 2:n],
                  [fill(2.0, n); fill(-1.0, n - 1)], n, n)
    bsyn = ones(n)
    function prescribed(seq)
        k = Ref(0)
        return x -> (k[] += 1; seq[min(k[], length(seq))])
    end

    # (1a) the step-6130 class: the unshifted answer is marginal (4e-4) but the
    # solve is not *lost*, so no tiny rung is tried -- the rungs are exactly the
    # historical ones, and the unshifted answer comes back bitwise.
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([3.98e-4, 1.32e-3]))
    @test rungs !== nothing && length(rungs) == 2
    @test [g[1] for g in rungs] == [0.0, 1.0e-12]
    @test sh == 0.0 && nref == 0 && r == 3.98e-4
    @test x == (Ssyn \ bsyn)              # the unshifted solve, untouched

    # (1b) the step-6600 class: the unshifted solve is lost (4.7e7 > 1), the
    # tiny rung is tried first and accepted, and the answer is a real solve.
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-6,
                                   resid=prescribed([4.69e7, 5.25e-7]))
    @test length(rungs) == 2
    @test rungs[1] == (0.0, 4.69e7) && rungs[2][1] == 1.0e-14
    @test sh == 1.0e-14 && r == 5.25e-7 && nref == 0
    @test norm(Ssyn * x - bsyn) / norm(bsyn) < 1.0e-12

    # (1c) none of the tiny rungs improves on the lost solve -> the phase gives
    # way and the historical ladder runs to completion.
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-6,
                                   resid=prescribed([4.69e7, 8.0e7, 9.0e7,
                                                     7.0e7, 3.0e7, 5.0e7, 5.0e-7]))
    @test [g[1] for g in rungs[1:6]] == [0.0, 1.0e-14, 1.0e-13,
                                         1.0e-12, 1.0e-10, 1.0e-8]
    @test rungs[end][1] === :refine
    @test r == 5.0e-7
    # the returned iterate is the refined one, so the reported shift is the
    # factorisation the refinement used: the *best* shifted rung, not the last
    @test sh == 1.0e-10

    # (1d) the tiny rung can also pass on its second try
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-6,
                                   resid=prescribed([4.69e7, 8.0e7, 2.0e-7]))
    @test length(rungs) == 3 && rungs[3][1] == 1.0e-13
    @test sh == 1.0e-13 && r == 2.0e-7 && nref == 0

    # (1e) `tiny_shifts = ()` and `lost_tol = Inf` both disable the phase, so a
    # caller that does not want it stays on the historical path
    for kw in ((; tiny_shifts=()), (; lost_tol=Inf))
        x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-6,
                                       resid=prescribed([4.69e7, 6.0e-7]), kw...)
        @test length(rungs) == 2 && rungs[2][1] == 1.0e-12
        @test sh == 1.0e-12 && r == 6.0e-7
    end

    # (1f) a healthy solve is still a single factorisation with no record
    x, r, sh, nref, rungs = ladder(Ssyn, bsyn; accept_tol=1.0e-3,
                                   resid=prescribed([9.0e-9]))
    @test rungs === nothing && sh == 0.0 && nref == 0

    # ---------------------------------------------------------------- (2) ----
    # A saddle whose pressure block has a *known* near-null direction.  With
    # K = I and B = nb [1 1 0 0; 0 0 1 1] the Schur complement is exactly
    # 2 nb^2, so the shifted solve has the closed form
    #     p(delta) = (B f - g)/(2 nb^2 - delta):
    # a shift at the Schur complement's own scale is resonant (the pressure
    # explodes), a shift two decades below it is a 1 % bias.  This is the
    # mechanism the realistic state exhibits at 1e-12 vs 1e-14.
    nb = 1.0e-6
    Kd = Matrix{Float64}(I, 4, 4)
    B = nb * [1.0 1 0 0; 0 0 1 1]
    S = sparse([Kd B'; B zeros(2, 2)])
    f = [1.0, -2.0, 0.5, 3.0]
    b = vcat(f, zeros(2))
    p0 = [1.0, -1.0] / sqrt(2.0)          # the pressure direction of the pair
    z0 = vcat(zeros(4), p0)
    @test norm(S * z0) <= 3.0 * nb        # |B' p0| = nb*sqrt(2)
    @test maximum(abs, S) == 1.0
    setprecision(BigFloat, 256) do
        # the exact solution (the Schur complement 2e-12 needs the precision)
        xe = Float64.(BigFloat.(Matrix(S)) \ BigFloat.(b))
        @test norm(S * xe - b) / norm(b) < 1.0e-13
        pe = xe[5:6]
        @test isapprox(norm(pe), 1.8200275e6; rtol=1.0e-6)
        dmax = maximum(abs, diag(S)) + 1.0
        # the historical rung sits on the Schur complement: the pressure is lost
        xh = lu(S + (1.0e-12 * dmax) * I) \ b
        @test norm(xh[5:6]) > 1.0e3 * norm(pe)       # measured 9.1e17
        # two decades below it: stable, 1 % bias, residual < 1e-6
        xt = lu(S + (1.0e-14 * dmax) * I) \ b
        @test isapprox(norm(xt[5:6]), norm(pe); rtol=0.05)   # measured 1.84e6
        @test norm(S * xt - b) / norm(b) < 1.0e-6
        # and the ladder with the default trigger leaves the (healthy)
        # unshifted answer untouched -- bitwise
        xd, rd, shd, nd, rungsd = ladder(S, b; accept_tol=1.0e-6)
        @test rungsd === nothing && shd == 0.0 && xd == (S \ b)
    end

    # ---------------------------------------------------------------- (3) ----
    # A real SEM saddle: the G164 wall-refined free-slip tree, whose unshifted
    # residual is 1.6e-4 and whose 1e-12 rung is exact.  The default trigger
    # leaves it bit-identical; forcing the phase can only improve the residual.
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
    lv = Set(qt.leaves)
    eta_fun = (x, y) -> semquad_eta_at(sq, theta, x, y, pr, lv)
    fz = (x, y) -> pr.Ra * MantleSEM._semquad_eval_at(sq, theta, x, y, lv)
    u_def, p_def, _, _, _ = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2;
                                                     bc=:free, linear=:direct,
                                                     stokes_accept_tol=1.0e-3)
    r_def = LAST_SEMSTOKES_RESIDUAL[]
    lad_def = LAST_SEMSTOKES_LADDER[]
    @test LAST_SEMSTOKES_SHIFT[] == 1.0e-12         # the historical rung
    @test [g[1] for g in lad_def] == [0.0, 1.0e-12] # no tiny rung: not lost
    # forced through the phase: same or better, never worse
    u_forced, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2;
                                                    bc=:free, linear=:direct,
                                                    stokes_accept_tol=1.0e-3,
                                                    stokes_lost_tol=0.0)
    @test LAST_SEMSTOKES_RESIDUAL[] <= r_def * (1 + 1.0e-12)
    @test LAST_SEMSTOKES_LADDER[][2][1] in MantleSEM.SEMSTOKES_TINY_SHIFTS
    @test all(isfinite, u_forced)

    # ---------------------------------------------------------------- (5) ----
    # `linear=:mg` is not touched: the direct-path records stay at their reset
    # values and the solve is finite.
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
    umg, _, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                               bc=:free, linear=:mg,
                                               mg_maxit=40, mg_restart=40)
    @test all(isfinite, umg)
    @test LAST_SEMSTOKES_SHIFT[] == 0.0
    @test LAST_SEMSTOKES_LADDER[] === nothing
    @test LAST_SEMSTOKES_REFINE_ITERS[] == 0

    # ---------------------------------------------------------------- (4) ----
    # The realistic state (data-dependent, not committed): the step-6600/6604
    # checkpoint on the 8260-leaf wall-refined tree.
    state = get(ENV, "G209_STATE", "")
    if isempty(state) || !isfile(state)
        @info "G209: realistic-state check skipped (set G209_STATE to the " *
              "step-6600 checkpoint to enable it)"
        @test_skip 1 + 1 == 2
    else
        prj = probe_realistic(state)
        @test prj.unshifted_res > 1.0e7       # measured 4.69e7 (lost)
        @test prj.res < 1.0e-6                # measured 5.25e-7
        @test prj.shift == 1.0e-14
        @test prj.norms < 2.0e6               # measured 6.58e5 (not 8.9e11)
        @test prj.nref == 0
    end
end
