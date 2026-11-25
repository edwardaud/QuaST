# G195: depth-dependent Drucker-Prager friction and cohesion.
#
# The yield surface is
#
#     c(d)  = yield_c + yield_c_grad*d
#     mu(d) = sin(yield_phi) + yield_mu_grad*d,        d = 1-y
#     tau_y = min(f(chi)*(c(d)*cos(yield_phi) + mu(d)*p), yield_taumax)
#
# and the plastic viscosity/tangents are those of G162/G163 with `p sin(phi)`
# replaced by `mu(d) p`.  This gate checks:
#
#   (1) the depth profiles evaluated at the assembly's quadrature points match
#       the analytic expression (bitwise against the same expression tree, and
#       to round-off against an independently re-associated form);
#   (2) with zero gradients the yield stress, plastic viscosity and all three
#       analytic derivatives are *bitwise* the historical constant-coefficient
#       expressions (the two bodies are copied verbatim from the pre-change
#       source into this file);
#   (3) `deta/dp`, `deta/dedot` and `deta/dT` against central differences at
#       two depths, i.e. the tangents really carry mu(d)/c(d);
#   (4) the per-step Newton's Jacobian with the Drucker-Prager `Kp` block
#       matches central differences of the residual (full direction and a
#       pressure-only direction that isolates Kp);
#   (5) a time-stepping case with a depth-dependent friction profile runs
#       stably with `pic_mode=:newton`: every step converges, the Newton and
#       Picard(`maxpic=20`) reach the same fixed point, the depth profile
#       changes the state measurably, and a 10-step CFL run stays finite;
#   (6) the parameter guards of the linear profiles.
using Test, QuaST, LinearAlgebra, Printf

# ---- (2) verbatim copies of the pre-change (HEAD ad6951c) expressions --------
function g195_old_yield_stress(p, pr)
    tau = pr.yield_c * cos(pr.yield_phi) + p * sin(pr.yield_phi)
    return min(tau, pr.yield_taumax)
end
function g195_old_eta(T, y, edot, p, pr)
    eta_v, _ = eta_at(T, y, pr)
    tau_y = g195_old_yield_stress(p, pr)
    eta_pl = tau_y / (2.0 * max(edot, pr.yield_eps_min))
    return _g195_clip(min(eta_v, g195_old_damper(eta_v, eta_pl, pr)), pr)[1]
end
function g195_old_damper(eta_v, eta_pl, pr)
    pr.yield_damper > 0.0 || return eta_pl
    return (pr.yield_damper + eta_pl) / (1.0 + pr.yield_damper / eta_v)
end
function _g195_clip(eta, pr)
    clipped = false
    if isfinite(pr.eta_min)
        if pr.eta_soft > 0.0
            t = (eta - pr.eta_min) / pr.eta_soft
            eta = pr.eta_min + pr.eta_soft * log1p(exp(min(t, 50.0)))
            clipped = t < 0.0
        elseif eta < pr.eta_min
            eta = pr.eta_min
            clipped = true
        end
    end
    if isfinite(pr.eta_max) && eta > pr.eta_max
        eta = pr.eta_max
        clipped = true
    end
    return eta, clipped
end
function g195_old_deta_dp(T, y, edot, p, pr)
    pr.yield_phi > 0.0 || return 0.0
    eta_v, _ = eta_at(T, y, pr)
    tau_raw = pr.yield_c * cos(pr.yield_phi) + p * sin(pr.yield_phi)
    tau_raw < pr.yield_taumax || return 0.0
    ed = max(edot, pr.yield_eps_min)
    eta_pl = tau_raw / (2.0 * ed)
    eta_pl < eta_v || return 0.0
    eta_b = g195_old_damper(eta_v, eta_pl, pr)
    _, clipped = _g195_clip(min(eta_v, eta_b), pr)
    clipped && return 0.0
    return (sin(pr.yield_phi) / (2.0 * ed)) /
           (1.0 + pr.yield_damper / eta_v)
end
function g195_old_deta_dp_eps(T, y, edot, p, pr)
    edot <= pr.yield_eps_min && return 0.0
    eta_v, _ = eta_at(T, y, pr)
    tau_y = g195_old_yield_stress(p, pr)
    eta_pl = tau_y / (2.0 * edot)
    eta_pl < eta_v || return 0.0
    eta_b = g195_old_damper(eta_v, eta_pl, pr)
    _, clipped = _g195_clip(min(eta_v, eta_b), pr)
    clipped && return 0.0
    return (-eta_pl / edot) / (1.0 + pr.yield_damper / eta_v)
end
function g195_old_deta_T(T, y, edot, p, pr)
    eta_v, _ = eta_at(T, y, pr)
    tau_y = g195_old_yield_stress(p, pr)
    eta_pl = tau_y / (2.0 * max(edot, pr.yield_eps_min))
    if eta_pl < eta_v
        eta_b = g195_old_damper(eta_v, eta_pl, pr)
        _, clipped = _g195_clip(min(eta_v, eta_b), pr)
        clipped && return 0.0
        (pr.yield_damper > 0.0) || return 0.0
        D = 1.0 + pr.yield_damper / eta_v
        return (pr.yield_damper + eta_pl) * pr.yield_damper *
               deta_dT_at(T, y, pr) / (eta_v^2 * D^2)
    end
    _, clipped = _g195_clip(min(eta_v, eta_pl), pr)
    clipped && return 0.0
    return deta_dT_at(T, y, pr)
end

@testset "G195 depth-dependent Drucker-Prager yield" begin
    # ---- (1) depth profiles at the assembly's quadrature points -----------
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                eta_min=1.0e-4, eta_max=10.0, bc=:free,
                yield_c_grad=0.5, yield_mu_grad=-0.15)
    qt = Quadtree(3)
    for _ in 1:1, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    sq = make_semquad(qt, 2)
    fd = make_fedata(2, 2)
    nd = 0
    for lf in sq.leaves
        hx = QuaST.leaf_hx(sq, lf[1]); hy = QuaST.leaf_hy(sq, lf[1])
        for qy in 1:fd.nq, qx in 1:fd.nq
            y = lf[3] * hy + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            x = lf[2] * hx + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            d = 1.0 - y
            c_an = 1.0 + 0.5 * d
            mu_an = sin(0.3) - 0.15 * d
            @test semquad_yield_cohesion(y, pr) == c_an
            @test semquad_yield_friction(y, pr) == mu_an
            # bitwise against the *same* expression tree the kernel uses, and
            # to round-off against a differently associated analytic form
            p = -0.7 + 2.4 * sin(3.1 * x + 1.7 * y)
            tau_same = semquad_yield_cohesion(y, pr) * cos(0.3) +
                       p * semquad_yield_friction(y, pr)
            @test semquad_yield_stress(p, y, pr) ==
                  min(tau_same, 1.0e6)
            tau_re = (1.0 + 0.5 * d) * cos(0.3) + (sin(0.3) - 0.15 * d) * p
            @test semquad_yield_stress(p, y, pr) ≈ min(tau_re, 1.0e6) rtol=1e-14
            nd += 1
        end
    end
    @test nd == length(sq.leaves) * fd.nq^2
    # 2-argument form is the surface value (y = 1): the historical constant law
    @test semquad_yield_stress(0.7, pr) == min(1.0 * cos(0.3) + 0.7 *
                                               sin(0.3), 1.0e6)

    # ---- (2) zero gradients: bitwise the historical expressions ------------
    pr0 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                 yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                 yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                 eta_min=1.0e-4, eta_max=10.0, bc=:free,
                 yield_c_grad=0.0, yield_mu_grad=0.0)
    nbit = 0
    for y in (0.0, 0.25, 0.5, 0.75, 1.0),
        p in (-2.0, -0.3, 0.0, 0.41, 3.0),
        ed in (1.0e-10, 1.0e-8, 1.0e-5, 1.0e-2, 1.0),
        T in (0.0, 0.2, 0.55, 1.0)
        @test semquad_yield_stress(p, y, pr0) == g195_old_yield_stress(p, pr0)
        @test semquad_visco_plastic_eta(T, y, ed, p, pr0) ==
              g195_old_eta(T, y, ed, p, pr0)
        @test semquad_deta_dp(T, y, ed, p, pr0) ==
              g195_old_deta_dp(T, y, ed, p, pr0)
        @test semquad_deta_dp_eps(T, y, ed, p, pr0) ==
              g195_old_deta_dp_eps(T, y, ed, p, pr0)
        @test semquad_dp_deta_T(T, y, ed, p, pr0) ==
              g195_old_deta_T(T, y, ed, p, pr0)
        nbit += 5
    end
    @test nbit == 5 * 5 * 5 * 4 * 5
    # gradients explicitly 0 and the defaults are the same object behaviour
    prd0 = Params(; Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                  yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                  yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                  eta_min=1.0e-4, eta_max=10.0, bc=:free)
    @test prd0.yield_c_grad == 0.0 && prd0.yield_mu_grad == 0.0
    @test semquad_yield_stress(0.41, 0.3, prd0) ==
          g195_old_yield_stress(0.41, prd0)

    # ---- (3) analytic tangents vs central differences at two depths --------
    # A yielded point: tau_y/(2 edot) < eta_v and above eta_min.  The viscous
    # law is temperature dependent so the `deta/dT` check also exercises the
    # plastic damper's `eta_v(T)` dependence (with a constant law both sides
    # would be zero and the assertion vacuous).
    prt = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=10.0,
                 yield_c=0.1, yield_phi=0.4, yield_taumax=1.0e6,
                 yield_eps_min=1.0e-12, yield_damper=2.0e-2,
                 eta_min=1.0e-6, eta_max=1.0e6, bc=:free,
                 yield_c_grad=0.3, yield_mu_grad=-0.1)
    for (y, p, ed, T) in ((1.0, 0.1, 0.8, 0.5), (0.2, 0.1, 0.8, 0.5))
        eta0 = semquad_visco_plastic_eta(T, y, ed, p, prt)
        @test 1.0e-6 < eta0 < 1.0e6          # not clamped
        h = 1.0e-6
        fdp = (semquad_visco_plastic_eta(T, y, ed, p + h, prt) -
               semquad_visco_plastic_eta(T, y, ed, p - h, prt)) / (2h)
        @test abs(semquad_deta_dp(T, y, ed, p, prt) - fdp) <=
              1.0e-6 * abs(fdp) + 1.0e-12
        fde = (semquad_visco_plastic_eta(T, y, ed + h, p, prt) -
               semquad_visco_plastic_eta(T, y, ed - h, p, prt)) / (2h)
        @test abs(semquad_deta_dp_eps(T, y, ed, p, prt) - fde) <=
              1.0e-6 * abs(fde) + 1.0e-12
        fdT = (semquad_visco_plastic_eta(T + h, y, ed, p, prt) -
               semquad_visco_plastic_eta(T - h, y, ed, p, prt)) / (2h)
        @test abs(fdT) > 1.0e-6              # the check is not vacuous
        @test abs(semquad_dp_deta_T(T, y, ed, p, prt) - fdT) <=
              1.0e-6 * abs(fdT) + 1.0e-12
    end

    # ---- (4) per-step Newton Jacobian with the Kp block --------------------
    qj = Quadtree(2; cells_x=2, cells_y=2)
    for _ in 1:1, lf in collect(qj.leaves); refine!(qj, lf); end
    balance!(qj)
    ssj = make_semstokes(qj, 2)
    sqj = ssj.su
    Nj = sqj.N
    leaves_j = Set(qj.leaves)
    thj = [0.05 * cospi(x) * sinpi(z) for (x, z) in sqj.coords]
    for (i, (x, z)) in enumerate(sqj.coords)
        (z <= 1.0e-12 || z >= 1.0 - 1.0e-12) && (thj[i] = 0.0)
    end
    make_eta_j = function (theta, ufull, pfull)
        uxv = ufull[1:Nj]; uzv = ufull[(Nj + 1):(2Nj)]
        T_at(x, y) = Tbar(y, pr.H) +
                     QuaST._semquad_eval_at(sqj, theta, x, y, leaves_j)
        ed_at(x, y) = semquad_strain_rate_at(sqj, uxv, uzv, x, y, leaves_j)
        p_at(x, y) = QuaST._semquad_eval_at(ssj.sp, pfull, x, y,
                                               Set(ssj.sp.leaves))
        ((x, y) -> semquad_visco_plastic_eta(T_at(x, y), y, ed_at(x, y),
                                            p_at(x, y), pr),
         (x, y) -> semquad_dp_deta_T(T_at(x, y), y, ed_at(x, y), p_at(x, y), pr),
         (x, y, ed) -> semquad_deta_dp_eps(T_at(x, y), y, ed, p_at(x, y), pr),
         (x, y, ed, pq) -> semquad_deta_dp(T_at(x, y), y, ed, pq, pr))
    end
    z2 = (x, y) -> 0.0
    ufull = zeros(2Nj)
    pfull = zeros(ssj.sp.N)
    for _ in 1:3
        ef = make_eta_j(thj, ufull, pfull)
        u, p, _, _, _ = solve_semquad_stokes_var(
            ssj, qj, ef[1], z2,
            (x, y) -> pr.Ra * QuaST._semquad_eval_at(sqj, thj, x, y, leaves_j),
            z2, z2; bc=pr.bc, linear=:direct)
        ufull = 0.7 * ufull + 0.3 * u
        pfull = 0.7 * pfull + 0.3 * p
    end
    ctx = QuaST._semquad_step_residual_ctx(ssj, qj, pr, thj, thj, thj,
                                              ufull, pfull, make_eta_j,
                                              leaves_j; dt=1.0e-5,
                                              dt_prev=1.0e-5, firststep=false)
    @test ctx.sys.Kp_red !== nothing
    @test size(ctx.sys.Kp_red) == (ctx.nf, ctx.nfp)
    Mv = assemble_semquad_pressure_mass(sqj)
    nf, nfp, nt = ctx.nf, ctx.nfp, ctx.nt
    resid_of = function (x)
        thx = thj + ctx.parts.Cfree * x[(nf + nfp + 1):end]
        ufx = ufull + ctx.sys.Cu * x[1:nf]
        pfx = pfull + ctx.sys.Cp * x[(nf + 1):(nf + nfp)]
        c = QuaST._semquad_step_residual_ctx(ssj, qj, pr, thj, thj, thx,
                                                ufx, pfx, make_eta_j, leaves_j;
                                                dt=1.0e-5, dt_prev=1.0e-5,
                                                firststep=false)
        return QuaST.semquad_step_residual(c, thx)
    end
    x = [sin(0.7 * k) for k in 1:(nf + nfp + nt)]
    x ./= norm(x)
    Jx = QuaST.semquad_step_jvp(ctx, thj, ufull, x, Mv; N=Nj, sq=sqj,
                                    q=ssj.q, pr=pr)
    for h in (1.0e-6, 1.0e-7)
        fd = (resid_of(h * x) - resid_of(-h * x)) / (2h)
        @test norm(Jx - fd) / norm(Jx) <= 1.0e-6
    end
    xp = zeros(nf + nfp + nt)
    prand = [cos(1.3 * k) for k in 1:size(ctx.sys.Cp, 2)]
    xp[(nf + 1):(nf + nfp)] = prand ./ norm(prand)
    Jp = QuaST.semquad_step_jvp(ctx, thj, ufull, xp, Mv; N=Nj, sq=sqj,
                                    q=ssj.q, pr=pr)
    for h in (1.0e-6, 1.0e-7)
        fd = (resid_of(h * xp) - resid_of(-h * xp)) / (2h)
        @test norm(Jp - fd) / norm(Jp) <= 1.0e-5
    end
    # dropping Kp must make the same check fail, so it is not vacuous
    ctxn = merge(ctx, (sys=merge(ctx.sys, (Kp_red=nothing,)),))
    Jp_nokp = QuaST.semquad_step_jvp(ctxn, thj, ufull, xp, Mv; N=Nj,
                                         sq=sqj, q=ssj.q, pr=pr)
    fd = (resid_of(1.0e-7 * xp) - resid_of(-1.0e-7 * xp)) / 2.0e-7
    @test norm(Jp_nokp - fd) / norm(Jp_nokp) > 1.0e-4

    # ---- (5) time stepping with a depth-dependent profile ------------------
    function g195_run(pru; nsteps=3, mode=:picard, maxpic=1)
        q = Quadtree(2; cells_x=2, cells_y=2)
        for _ in 1:1, lf in collect(q.leaves); refine!(q, lf); end
        balance!(q)
        ss = make_semstokes(q, 2)
        th0 = [0.05 * cospi(x) * sinpi(y) for (x, y) in ss.su.coords]
        for (i, (x, y)) in enumerate(ss.su.coords)
            (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
        end
        kw = mode === :newton ? (pic_mode=:newton, newton_maxit=12) :
             (maxpic=maxpic,)
        r = run_semquad_simulation(q, pru, th0; p=2, tstop=1.0, dt0=1.0e-6,
                                   dtmax=1.0e-4, cfl=2.0, maxsteps=nsteps,
                                   verbose=false, ev=0.0, advection=:skew,
                                   linear=:direct, pic_diag=true, kw...)
        (r=r, diag=copy(SEMQUAD_PIC_DIAG[]))
    end
    rp = g195_run(pr; nsteps=3, mode=:picard, maxpic=20)
    rn = g195_run(pr; nsteps=3, mode=:newton)
    rc = g195_run(Params(; Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                         yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                         yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                         eta_min=1.0e-4, eta_max=10.0, bc=:free);
                  nsteps=3, mode=:picard, maxpic=20)
    @test all(row -> row.mode === :newton, rn.diag)
    @test all(row -> last(row.changes) < 1.0e-6, rp.diag)
    @test all(row -> last(row.changes) < 1.0e-6, rn.diag)
    @test norm(rp.r.theta - rn.r.theta) / norm(rp.r.theta) < 1.0e-5
    vp = semquad_vrms(rp.r.ss.su, rp.r.qt, rp.r.ux, rp.r.uz)
    vn = semquad_vrms(rn.r.ss.su, rn.r.qt, rn.r.ux, rn.r.uz)
    @test abs(vp - vn) / vp < 1.0e-5
    # the depth profile changes the state measurably vs the constant law
    vc = semquad_vrms(rc.r.ss.su, rc.r.qt, rc.r.ux, rc.r.uz)
    @test abs(vc - vp) / vp > 1.0e-3
    @test sum(row.iters for row in rn.diag) < sum(row.iters for row in rp.diag)
    @printf("G195: depth DP 3 steps: Picard20 iters=%s Newton iters=%s |dtheta|/|theta|=%.2e  Vrms depth=%.6e const=%.6e\n",
            [row.iters for row in rp.diag], [row.iters for row in rn.diag],
            norm(rp.r.theta - rn.r.theta) / norm(rp.r.theta), vp, vc)
    # 10-step CFL run with the depth profile stays finite
    r10 = g195_run(pr; nsteps=10, mode=:newton)
    @test r10.r.steps == 10
    @test all(isfinite, r10.r.theta) && all(isfinite, vcat(r10.r.history...))
    @printf("G195: depth DP 10 steps Newton: iters=%s Nu_top=%.10f Vrms=%.6e\n",
            [row.iters for row in r10.diag], r10.r.history[end][3],
            semquad_vrms(r10.r.ss.su, r10.r.qt, r10.r.ux, r10.r.uz))

    # ---- (6) parameter guards ---------------------------------------------
    @test_throws ArgumentError Params(yield_c=0.5, yield_phi=0.3,
                                      yield_c_grad=-1.0, eta_min=1.0e-4)
    @test_throws ArgumentError Params(yield_c=0.5, yield_phi=0.3,
                                      yield_mu_grad=-1.0, eta_min=1.0e-4)
    @test_throws ArgumentError Params(yield_c=0.5, yield_phi=0.3,
                                      yield_c_grad=Inf, eta_min=1.0e-4)
    @test_throws ArgumentError Params(yield_c=0.5, yield_phi=0.3,
                                      yield_mu_grad=NaN, eta_min=1.0e-4)
end
