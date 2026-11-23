# G248: the density hook, `Params.rho_multiplier` (the density half of a
# "sticky air" layer).
#
# `Params.eta_multiplier` already scales whatever the viscosity law returns; the
# density had no equivalent.  This gate pins the new hook:
#
#   (a) the *parameter*: default `nothing`, a keyword that reads back the
#       function, and every field after it in the positional constructor still
#       receives its own value (the PHASE2 constructor-order trap: a field added
#       mid-struct shifts every later argument and shows up as a type-conversion
#       error, not a missing argument);
#   (b) `nothing` is bit-identical through the real driver (theta/ux/uz/history),
#       and a *unit* multiplier is too -- but the run really flows, so the
#       comparison is not 0 == 0;
#   (c) the hook sits exactly on the buoyancy load: the driver's first-step
#       velocity for a spatial multiplier equals a direct Stokes solve with
#       f_z = Ra * rho_multiplier(x,y) * theta;
#   (d) a uniform factor c is the identity the hook has to satisfy: the velocity
#       of a run with `Ra` and `rho_multiplier = c` equals the velocity of the
#       same run with `Ra*c` (to solver accuracy), and it is not the unscaled
#       run;
#   (e) the buoyancy-work diagnostic sees the same density (the solver's own
#       `Ra*W = U` balance stays closed);
#   (f) the anelastic mass constraint: `semquad_stokes_B` with the hook is the
#       Galerkin form of `div(rho_tot u) = 0` for `rho_tot = rho(y)*M(x,y)`,
#       including the two gradient terms, against a hand quadrature reference;
#       and the driver forwards the multiplier to that constraint;
#   (g) the Newton paths: a per-step Newton with the hook reaches the same fixed
#       point as the converged Picard loop with the hook, and the hook actually
#       changes the Newton answer (the Jacobian/`Mv` weighting is asserted
#       directly);
#   (h) a layered configuration does something physical: a dense load below a
#       low-density layer gives a non-zero vertical velocity whose sign under
#       the load is downward, stays finite, and does not reduce to the no-layer
#       run.
#
# Tolerances and why, where they are not a bitwise `==`:
#   * (d) is exact (`==`) in this configuration: the load is
#     `(Ra*c) * theta` on both sides with the same association, so one product
#     is bit-identical.  1e-12 * |u| is kept as the assertion floor instead of
#     `==` so a future reassociation of the closure is not a spurious failure.
#   * (e) `bal = (Ra*W - U)/max(...)` is the solver's own algebraic identity; it
#     is measured at 1e-14 on these runs and asserted below 1e-8.  With the
#     diagnostic left unwired the run with c=3.7 would give
#     `bal = 1 - 1/c = 0.73`, so the assertion is not vacuous.
#   * (f) the multiplier gradient is a central difference with h = 1e-6 (the
#     user function has no analytic derivative), so its round-off is
#     O(eps/h) ~ 1e-10 relative; the assertions use 1e-9..1e-8 absolute, which
#     is 100x the measured error (3.8e-12 / 2.0e-12) and the quadrature rule of
#     the reference is the same one the production matrix uses.
#   * (g) measured 1e-12 relative; 1e-9 is asserted.
#   * (h) the load is a cold Gaussian centred at (0.5, 0.5) below z_a = 0.6;
#     `u_z(0.5,0.5)` must be negative, measured -95 against a global max of 102,
#     so the sign margin 1e-2 of the max speed is far above round-off.  The
#     layered/no-layer velocity difference is 14% of the max speed and is
#     asserted above 1%: an unevaluated or misplaced multiplier gives 0.
using Test, QuaST, LinearAlgebra, Printf
using SparseArrays

@testset "G248 density hook" begin
    # A uniform tree of 4^nref leaves (level 4 base, like G247's helper).
    function g248_tree(nref::Int)
        qt = Quadtree(4)
        for _ in 1:nref, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        return qt
    end
    g248_pr(; m=nothing, Ra::Float64=1.0e4) =
        Params(Ra=Ra, H=0.0, eta_law=:expT, deta_T=1.0, rho_multiplier=m)

    # ---- (a) the parameter and the positional constructor -----------------
    prdef = Params()
    @test prdef.rho_multiplier === nothing
    # every field after the insertion point must still receive its own value:
    # eta_multiplier was the insertion neighbour, shear_modulus / wall_u /
    # rho_heating / rho_cp / Di / latent_L / weak_f follow it
    @test prdef.shear_modulus == 0.0
    @test prdef.wall_u == (nothing, nothing, nothing, nothing)
    @test prdef.rho_heating == false && prdef.rho_cp == false
    @test prdef.Di == 0.0 && prdef.latent_L == 0.0 && prdef.weak_f == 1.0
    @test prdef.wall_bc == (:same, :same, :same, :same)   # last struct field
    mfun = (x, y) -> 0.5 + 0.5 * x
    # `bc=:rigid` because `wall_u` sets a *tangential* value and since G258 a
    # value on a resolved `:free` wall is an error instead of a silent no-op
    # (the yplus value below is the field this check is about).
    prm = Params(; eta_multiplier=(x, y) -> 2.0, rho_multiplier=mfun,
                 shear_modulus=3.0, bc=:rigid,
                 wall_u=(nothing, nothing, nothing, (x, y) -> 1.0),
                 rho_heating=true, rho_cp=true, Di=2.0, latent_L=0.1,
                 weak_f=0.3, weak_mode=:linear)
    @test prm.rho_multiplier === mfun
    @test prm.eta_multiplier(0.5, 0.5) == 2.0
    @test prm.shear_modulus == 3.0
    @test prm.wall_u[4](0.0, 0.0) == 1.0
    @test prm.rho_heating && prm.rho_cp && prm.Di == 2.0
    @test prm.latent_L == 0.1 && prm.weak_f == 0.3

    # ---- shared small card: 4 leaves, one Stokes+energy step --------------
    qt = g248_tree(1)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    th = [0.1 * sinpi(x) * sinpi(z) for (x, z) in sq.coords]
    kw = (p=2, tstop=2.0e-3, dt0=1.0e-3, dtmax=1.0e-3, dtmin=1.0e-16,
          amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
          mass_form=:boussinesq)
    base = run_semquad_simulation(qt, g248_pr(), copy(th); kw...)
    @test maximum(abs, base.ux) > 1.0e-6      # the reference run flows

    # ---- (b) nothing and a unit multiplier are bit-identical --------------
    unit = run_semquad_simulation(qt, g248_pr(m=(x, y) -> 1.0), copy(th);
                                  kw...)
    @test base.theta == unit.theta
    @test base.ux == unit.ux && base.uz == unit.uz
    @test base.history == unit.history
    # and the buoyancy closure of the default path is the historical one
    th_at = (x, y) -> 0.1 * sinpi(x) * sinpi(y)
    f_hist = (x, y) -> 1.0e3 * th_at(x, y)
    f_hook = theta_buoyancy_load(Params(Ra=1.0e3), th_at)
    @test f_hook(0.3, 0.7) == f_hist(0.3, 0.7)
    @test f_hook(0.9, 0.1) == f_hist(0.9, 0.1)

    # ---- (c) the hook is on the buoyancy load ----------------------------
    msp = (x, y) -> 1.0 + 0.5 * sinpi(x) * sinpi(y)
    prsp = g248_pr(m=msp)
    r1 = run_semquad_simulation(qt, prsp, copy(th); kw..., maxsteps=1)
    z2 = (x, y) -> 0.0
    leaves = Set(qt.leaves)
    u_dir = solve_semquad_stokes_var(
        ss, qt, (x, y) -> 1.0, z2,
        (x, y) -> prsp.Ra * msp(x, y) *
                  QuaST._semquad_eval_at(sq, th, x, y, leaves),
        z2, z2; linear=:direct, bc=prsp.bc, wall_bc=prsp.wall_bc,
        wall_u=prsp.wall_u, deviatoric=prsp.deviatoric)[1]
    uscale = max(1.0, maximum(abs, u_dir))
    @test maximum(abs, r1.ux - u_dir[1:N]) <= 1.0e-12 * uscale
    @test maximum(abs, r1.uz - u_dir[(N + 1):(2N)]) <= 1.0e-12 * uscale
    # non-vacuous: the spatial multiplier changes the answer relative to no hook
    @test maximum(abs, r1.ux - base.ux) > 1.0e-3 * uscale

    # ---- (d) uniform c == Ra*c, the identity the hook must satisfy -------
    c = 3.7
    rscale = run_semquad_simulation(qt, g248_pr(m=(x, y) -> c), copy(th);
                                    kw...)
    rraise = run_semquad_simulation(qt, g248_pr(Ra=1.0e4 * c), copy(th);
                                    kw...)
    unorm = max(maximum(abs, rraise.ux), maximum(abs, rraise.uz), 1.0e-30)
    @test maximum(abs, rscale.ux - rraise.ux) <= 1.0e-12 * unorm
    @test maximum(abs, rscale.uz - rraise.uz) <= 1.0e-12 * unorm
    @test rscale.theta == rraise.theta
    @test maximum(abs, rscale.ux - base.ux) > 1.0e-2 * unorm   # not the plain run

    # ---- (e) the buoyancy-work diagnostic uses the same density ----------
    # history columns: [t, dt, Nut, Nub, Vrms, meanT, N, nleaves, divmax, hmin,
    #                   umax, cfl, closure, usrms_surf, umax_surf, W_last,
    #                   U_last, bal_last]
    row = rscale.history[end]
    W, U, bal = row[16], row[17], row[18]
    @test isfinite(W) && isfinite(U) && U > 0.0
    @test abs(bal) < 1.0e-8            # measured ~1e-14; unwired W gives 0.73

    # ---- (f) the anelastic mass constraint carries the density -----------
    # (f1) a unit multiplier is the historical matrix, bit for bit
    B0 = semquad_stokes_B(ss, qt)
    @test semquad_stokes_B(ss, qt; rho_mult=(x, y) -> 1.0) == B0
    # (f2) linear multiplier M = 1 + 0.3x + 0.2y with the constant base density:
    # u = (x, -y) is Q2-exact and div u = 0, so div(M u) = u.grad M = 0.3x-0.2y.
    # Bc*u must equal int q_i (0.3x - 0.2y), both gradient terms included.
    Mlin = (x, y) -> 1.0 + 0.3 * x + 0.2 * y
    ufull = vcat([c0[1] for c0 in sq.coords], [-c0[2] for c0 in sq.coords])
    r = semquad_stokes_B(ss, qt; rho_mult=Mlin) * ufull
    p = ss.sp.p
    nq1 = p + 1
    fd = make_fedata(2, p; nq=8)
    ref = zeros(Float64, ss.sp.N)
    for lf in ss.sp.leaves
        hx = QuaST.leaf_hx(ss.sp, lf[1]); hy = QuaST.leaf_hy(ss.sp, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = ss.sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            for b in 1:nq1, a in 1:nq1
                ref[ids[(b - 1) * nq1 + a]] +=
                    w * (0.3 * x - 0.2 * y) * fd.phi_p[a, qx] * fd.phi_p[b, qy]
            end
        end
    end
    @test maximum(abs, r - ref) < 1.0e-9        # measured 3.8e-12 (FD of M)
    @test maximum(abs, ref) > 1.0e-3            # not 0 == 0
    # (f3) a smooth stratified density and the same multiplier: compare against
    # the same quadrature rule and an *exact* gradient, so the residual is the
    # central-difference error of the multiplier only.
    rho = y -> exp(0.7 * (1.0 - y))
    drho = y -> -0.7 * exp(0.7 * (1.0 - y))
    Bm = semquad_stokes_B(ss, qt; rho_fun=rho, drho_fun=drho, rho_mult=Mlin)
    fdq = make_fedata(2, ss.sp.p)
    pref = zeros(Float64, ss.sp.N, 2 * N)
    pu = sq.p
    nv = pu + 1                       # velocity nodes per direction
    for lf in sq.leaves
        hx = QuaST.leaf_hx(sq, lf[1]); hy = QuaST.leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = sq.elem[lf]; pids = ss.sp.elem[lf]
        for qy in 1:fdq.nq, qx in 1:fdq.nq
            w = 0.25 * hx * hy * fdq.gl_w[qx] * fdq.gl_w[qy]
            x = x0 + 0.5 * (fdq.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fdq.gl_x[qy] + 1.0) * hy
            rtot = rho(y) * Mlin(x, y)
            drtot_dx = rho(y) * 0.3
            drtot_dy = drho(y) * Mlin(x, y) + rho(y) * 0.2
            for b in 1:nq1, a in 1:nq1
                pq = fdq.phi_p[a, qx] * fdq.phi_p[b, qy]
                pq == 0.0 && continue
                piq = pids[(b - 1) * nq1 + a]
                for j in 1:nv, i in 1:nv
                    uj = uids[(j - 1) * nv + i]
                    dpx = fdq.dphi_v[i, qx] * fdq.phi_v[j, qy] * (2.0 / hx)
                    dpy = fdq.phi_v[i, qx] * fdq.dphi_v[j, qy] * (2.0 / hy)
                    ph = fdq.phi_v[i, qx] * fdq.phi_v[j, qy]
                    pref[piq, uj] += w * pq * (rtot * dpx + drtot_dx * ph)
                    pref[piq, N + uj] += w * pq * (rtot * dpy + drtot_dy * ph)
                end
            end
        end
    end
    @test maximum(abs, Bm - pref) < 1.0e-8      # measured 2.0e-12
    @test maximum(abs, pref) > 1.0e-3
    # the y-derivative fallback of B (finite difference of rho) is itself
    # exercised through the hook path when the caller supplies no drho_fun
    Bm_fd = semquad_stokes_B(ss, qt; rho_fun=rho, rho_mult=Mlin)
    @test maximum(abs, Bm_fd - Bm) < 1.0e-8     # FD of rho against the analytic drho
    # (f4) the driver forwards the multiplier to the constraint: a run with a
    # reference state and `mass_form=:anelastic` satisfies the *multiplied*
    # constraint and not the unmultiplied one.
    rs = SemRefState(Tsurf=0.0, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                     rho_fun=rho)
    Msm = (x, y) -> 1.0 + 0.2 * x + 0.1 * y
    prref = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs,
                   rho_multiplier=Msm)
    prref0 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs)
    thm = [0.05 * sinpi(x) * sinpi(z) for (x, z) in sq.coords]
    kwm = (p=2, tstop=2.0e-5, dt0=1.0e-5, dtmax=1.0e-5, dtmin=1.0e-16,
           amr_every=0, ev=0.0, advection=:skew, linear=:direct, maxpic=1,
           pic_tol=1.0e-6, verbose=false)
    rm = run_semquad_simulation(qt, prref, copy(thm); kwm...)
    r0 = run_semquad_simulation(qt, prref0, copy(thm); kwm...)
    um = vcat(rm.ux, rm.uz)
    Bc_m = semquad_stokes_B(rm.ss, rm.qt; rho_fun=rho, drho_fun=drho,
                            rho_mult=Msm)
    Bc_0 = semquad_stokes_B(rm.ss, rm.qt; rho_fun=rho, drho_fun=drho)
    @test all(isfinite, um) && all(isfinite, rm.theta)
    @test maximum(abs, Bc_m * um) < 1.0e-8 * max(1.0, maximum(abs, um))
    @test maximum(abs, Bc_0 * um) > 1.0e-3 * maximum(abs, um)
    @test maximum(abs, rm.ux - r0.ux) > 1.0e-2 * maximum(abs, r0.ux)
    # (f5) the compressible reference-state block uses the same density:
    # Pcm[i,pj] = int phi_i(y) rho(y) M(x,y) Di gamma phi_pj dV (vertical rows).
    Di_c, gam_c = 2.0, 3.0
    P0c = semquad_stokes_Pcm(ss, qt; rho_fun=rho, Di=Di_c, gamma=gam_c)
    P1c = semquad_stokes_Pcm(ss, qt; rho_fun=rho, Di=Di_c, gamma=gam_c,
                             rho_mult=(x, y) -> 1.0)
    Pmc = semquad_stokes_Pcm(ss, qt; rho_fun=rho, Di=Di_c, gamma=gam_c,
                             rho_mult=Mlin)
    @test P1c == P0c
    pcm = ss.sp.p
    npq = pcm + 1
    fdp = make_fedata(sq.p, pcm)
    nv = sq.p + 1
    Pref = zeros(Float64, 2 * N, ss.sp.N)
    for lf in sq.leaves
        hx = QuaST.leaf_hx(sq, lf[1]); hy = QuaST.leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = sq.elem[lf]; pids = ss.sp.elem[lf]
        for qy in 1:fdp.nq, qx in 1:fdp.nq
            w = 0.25 * hx * hy * fdp.gl_w[qx] * fdp.gl_w[qy]
            x = x0 + 0.5 * (fdp.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fdp.gl_x[qy] + 1.0) * hy
            cc = Di_c * gam_c * rho(y) * Mlin(x, y)
            for b in 1:npq, a in 1:npq
                pq = fdp.phi_p[a, qx] * fdp.phi_p[b, qy]
                piq = pids[(b - 1) * npq + a]
                for j in 1:nv, i in 1:nv
                    uj = uids[(j - 1) * nv + i]
                    Pref[N + uj, piq] +=
                        w * cc * fdp.phi_v[i, qx] * fdp.phi_v[j, qy] * pq
                end
            end
        end
    end
    @test maximum(abs, Pmc - Pref) < 1.0e-12
    @test maximum(abs, Pref) > 1.0e-3

    # ---- (g) the Newton paths -------------------------------------------
    cn = 2.0
    prn = g248_pr(m=(x, y) -> cn)
    r_pic = run_semquad_simulation(qt, prn, copy(th); kw...,
                                   pic_mode=:picard, maxpic=20,
                                   pic_tol=1.0e-12)
    r_new = run_semquad_simulation(qt, prn, copy(th); kw...,
                                   pic_mode=:newton, pic_tol=1.0e-12)
    un = max(norm(r_pic.ux), norm(r_pic.uz), 1.0e-30)
    utn = max(norm(r_pic.theta), 1.0e-30)
    @test maximum(abs, r_new.ux - r_pic.ux) <= 1.0e-9 * un
    @test maximum(abs, r_new.uz - r_pic.uz) <= 1.0e-9 * un
    @test maximum(abs, r_new.theta - r_pic.theta) <= 1.0e-9 * utn
    r_nom = run_semquad_simulation(qt, g248_pr(), copy(th); kw...,
                                   pic_mode=:newton, pic_tol=1.0e-12)
    @test maximum(abs, r_nom.ux - r_new.ux) > 1.0e-3 * un   # hook reaches Newton
    # the tangent's mass matrix is the density-weighted one
    @test buoyancy_mass_matrix(sq, Params(Ra=1.0)) ==
          assemble_semquad_pressure_mass(sq)
    mw = (x, y) -> 1.0 + 0.5 * x
    prw = Params(Ra=1.0, rho_multiplier=mw)
    @test buoyancy_mass_matrix(sq, prw) ==
          assemble_semquad_pressure_mass(sq; weight=mw)
    @test maximum(abs, buoyancy_mass_matrix(sq, prw) -
                      assemble_semquad_pressure_mass(sq)) > 0.0

    # ---- (h) the layered sticky-air configuration ------------------------
    qtL = g248_tree(2)
    sqL = make_semquad(qtL, 2)
    NL = sqL.N
    za = 0.6
    thetaf = (x, z) -> -0.2 *
              exp(-((x - 0.5) / 0.15)^2 - ((z - 0.5) / 0.15)^2) * sinpi(z)
    thL = [thetaf(x, z) for (x, z) in sqL.coords]
    rho_air = (x, z) -> z > za ? 1.0e-3 : 1.0
    kwL = (p=2, tstop=1.5e-4, dt0=5.0e-5, dtmax=5.0e-5, dtmin=1.0e-16,
           amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
           mass_form=:boussinesq)
    prL = Params(Ra=1.0e5, H=0.0, eta_law=:expT, deta_T=1.0,
                 rho_multiplier=rho_air, bc=:free)
    prN = Params(Ra=1.0e5, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free)
    rL = run_semquad_simulation(qtL, prL, copy(thL); kwL...)
    rN = run_semquad_simulation(qtL, prN, copy(thL); kwL...)
    leavesL = Set(qtL.leaves)
    uz_load = QuaST._semquad_eval_at(sqL, rL.uz, 0.5, 0.5, leavesL)
    umaxL = maximum(abs, rL.uz)
    @test umaxL > 1.0e-3                       # non-zero vertical velocity
    @test uz_load < -1.0e-2 * umaxL            # downward under the load
    @test all(isfinite, rL.ux) && all(isfinite, rL.uz) &&
          all(isfinite, rL.theta)
    dz_diff = maximum(abs, rL.uz - rN.uz)
    @test dz_diff > 1.0e-2 * maximum(abs, rN.uz)   # the layer is not ignored
    # record the air-layer velocities for the dev-notes probe
    ga = [g for g in 1:NL if sqL.coords[g][2] > za]
    @printf("G248: layered max|uz|=%.3e u_z(load)=%.3e (%.2f of max) | max|uz| air=%.3e (%.0f%%) max|ux| air=%.3e | layered-vs-plain=%.1f%%\n",
            umaxL, uz_load, uz_load / umaxL, maximum(abs, rL.uz[ga]),
            100.0 * maximum(abs, rL.uz[ga]) / umaxL,
            maximum(abs, rL.ux[ga]),
            100.0 * dz_diff / maximum(abs, rN.uz))
end
