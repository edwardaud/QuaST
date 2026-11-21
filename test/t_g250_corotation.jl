# G250: the Maxwell stress is objective -- the co-rotational (Jaumann) term.
#
# `_advance_maxwell_stress!` used to advance the history with no spin at all,
#
#     tau^{n+1} = (tau^n + 2 G dt ehat) / (1 + dt/tau_M),
#
# so the *magnitude* relaxed correctly but the tensor never rotated with the
# material.  A Maxwell stress that does not rotate is not frame-indifferent, and
# every mantle-convection run is a large-strain run.  The update is now
#
#     tau^{n+1} = maxwell_stress_step(corotate(tau^n, wxy, dt), ehat, dt, eta, G)
#     corotate:   tau^n + dt (W tau^n - tau^n W),   W_xy = (du_x/dy - du_z/dx)/2,
#
# which is ASPECT's `viscoelastic` numerator.  This gate pins:
#   (a) the in-plane component form of `tau + dt (W tau - tau W)` against an
#       explicit matrix product `W*T - T*W`, for all three components (the xy
#       sign is the one that is easy to reverse), and its exact identity for
#       `wxy = 0` / `dt = 0`;
#   (b) a **rigid-body rotation** `u = (-Omega y, Omega x)` on the whole domain:
#       no strain at all, `ehat = 0`; the initial stress has its principal axes
#       at 45 degrees to the grid (`tau_0 = T0 [[0, 1], [1, 0]]`, hence
#       `tau_zz = 0`); the discrete solution is compared with the analytic
#       objective solution `tau(t) = exp(-t/tau_M) Q(t) tau_0 Q(t)^T`, `Q(t)`
#       the rotation by `Omega t`.  The discriminating observable is the
#       **orientation**, not the magnitude: four snapshots over a 45-degree
#       material rotation, at which the analytic stress is diagonal in the grid
#       axes (txy = 0, txx = -T0 exp(-t/tau_M)).  The **negative control** is
#       the same run with the term disabled -- which is exactly the pre-fix
#       update, the gated scalar kernel on the (zero) strain rate.  Its
#       magnitude still decays correctly (relative error `1e-2`), but the
#       components do not rotate: component-wise it differs from the analytic
#       curve by O(1) and its orientation error grows to pi/4.  This is the
#       direct evidence that a component comparison, not a magnitude one, is
#       what detects the missing term;
#   (c) a **pure shear** `u = (gamma y, gamma x)`, whose spin is zero: the new
#       update equals the spin-free update to round-off, so the term changes
#       nothing when the flow does not rotate;
#   (d) the analytic reference `maxwell_stress_exact_corotational` that amended
#       G246 (b) / G249 (a) use: it reduces to `maxwell_stress_exact` for
#       `wxy = 0`, is off for `G = 0`, and agrees with a fine discrete recursion
#       of the composed kernel (so the reference is pinned independently of the
#       kernel gate it is used to amend).
#
# Tolerances and their sources, stated before they are used:
# * component-wise relative error of the rotation run, `1e-3`: the co-rotational
#   term is the rotation *linearised* over the step, `1 + dt ad_W`, so its
#   magnitude grows by `sqrt(1 + (2 W_xy dt)^2) ~ 1 + 2 W_xy^2 dt^2` per step,
#   i.e. by `2 W_xy^2 dt T = 2 (T/20 000) T = 6.2e-5` over the run (T = pi/4,
#   W_xy = 1) -- measured 6.2e-5 at the last snapshot.  The backward-Euler
#   relaxation adds `T dt / (2 tau_M^2) ~ 1e-10`.  `1e-3` is 16x the measured
#   error and 700x below what the term-free run produces.
# * orientation error of the rotation run, `1e-5` rad: each step's linearised
#   rotation angle `atan(2 W dt)` differs from `2 W dt` by `(2 W dt)^3 / 3`, so
#   the total is `T (2 W dt)^2 / 3 ~ 1.6e-9` rad -- measured 1.6e-9.  `1e-5` is
#   6000x above that and five orders below the disabled term's pi/4.
# * the pure-shear round-off, `1e-14` of the stress scale: the discrete spin is
#   round-off (`~6e-16` from the GLL derivative rows), so the two updates can
#   differ only in the last ulp.
using Test, QuaST, LinearAlgebra, Printf

@testset "G250 co-rotational Maxwell stress" begin
    # ---- (a) the component form against an explicit matrix product --------
    function corot_matrix(txx, tyy, txy, w, dt)
        Wm = [0.0 w; -w 0.0]
        T = [txx txy; txy tyy]
        R = T + dt * (Wm * T - T * Wm)
        return (R[1, 1], R[2, 2], R[1, 2])
    end
    samples = ((1.0, -0.5, 0.3), (-2.0, 0.7, -1.1), (0.0, 0.0, 1.0),
               (3.0, 3.0, 0.0))
    maxerr = 0.0
    for (txx, tyy, txy) in samples, w in (-1.3, -0.2, 0.0, 0.4, 2.1),
        h in (0.7, 0.0)
        got = maxwell_stress_corotate(txx, tyy, txy, w, h)
        want = corot_matrix(txx, tyy, txy, w, h)
        maxerr = max(maxerr, maximum(abs.(got .- want)))
    end
    @test maxerr < 1.0e-15                 # measured 4.4e-16, matrix product
    # zero spin and zero step are exact identities: the term costs nothing when
    # there is nothing to rotate (the pure-shear block (d) uses this property)
    @test maxwell_stress_corotate(1.0, -0.5, 0.3, 0.0, 0.7) == (1.0, -0.5, 0.3)
    @test maxwell_stress_corotate(1.0, -0.5, 0.3, 0.4, 0.0) == (1.0, -0.5, 0.3)
    # the two normal increments are exact negatives, so the in-plane trace is
    # preserved and a deviatoric stress stays deviatoric: no projection is
    # needed on this term
    for w in (-1.3, 0.4), h in (0.7, 1.0e-3)
        a = maxwell_stress_corotate(2.0, -1.0, 0.35, w, h)
        b = maxwell_stress_corotate(-0.7, 0.4, -0.8, w, h)
        @test abs((a[1] + a[2]) - (2.0 + -1.0)) <= 1.0e-15
        @test abs((b[1] + b[2]) - (-0.7 + 0.4)) <= 1.0e-15
    end

    # ---- mesh for the driver-level blocks ---------------------------------
    function g250_tree(nref::Int)
        q = Quadtree(2)
        for _ in 1:nref, lf in collect(q.leaves)
            refine!(q, lf)
        end
        balance!(q)
        return q
    end
    qt = g250_tree(2)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    @test length(qt.leaves) == 16          # not one leaf (HANDOVER section 8)

    # ---- (b) rigid-body rotation against the objective analytic curve -----
    Omega = 1.0
    T_rot = pi / 4                 # 45-degree material rotation at the last snap
    tauM = 10.0 * T_rot
    eta0 = 1.0
    G0 = eta0 / tauM
    T0 = 0.8                       # tau_0 = T0 [[0, 1], [1, 0]]: axes at 45 deg
    nstep = 20_000
    dt = T_rot / nstep
    # u = (-Omega y, Omega x): W_xy = (du_x/dy - du_z/dx)/2 = -Omega and D = 0.
    # The material rotates counter-clockwise by Omega*t, so the analytic stress
    # is Q(Omega t) tau_0 Q(Omega t)^T times the Maxwell decay.
    ux = [-Omega * y for (_, y) in sq.coords]
    uz = [Omega * x for (x, _) in sq.coords]
    eps0 = QuaST.semquad_strain_rate_components(sq, ux, uz)
    @test maximum(abs, eps0.exx) <= 1.0e-14     # no strain at all
    @test maximum(abs, eps0.ezz) <= 1.0e-14
    @test maximum(abs, eps0.exz) <= 1.0e-14
    w = QuaST.semquad_spin_component(sq, ux, uz)
    @test maximum(abs.(w .+ Omega)) <= 1.0e-14  # W_xy = -Omega for this u
    Q(phi) = [cos(phi) -sin(phi); sin(phi) cos(phi)]
    function g250_analytic(t)
        R = exp(-t / tauM) .* (Q(Omega * t) * [0.0 T0; T0 0.0] * Q(Omega * t)')
        return (R[1, 1], R[2, 2], R[1, 2])
    end
    fnorm(txx, tyy, txy) = sqrt(txx^2 + tyy^2 + 2.0 * txy^2)
    orientation(txx, tyy, txy) = 0.5 * atan(2.0 * txy, txx - tyy)
    wrap_pi(x) = mod(x + pi / 2, pi) - pi / 2
    txx = zeros(sq.N); tyy = zeros(sq.N); txy = fill(T0, sq.N)
    eta = fill(eta0, sq.N)
    snaps = [nstep ÷ 4, nstep ÷ 2, 3 * nstep ÷ 4, nstep]
    run_vals = zeros(3, length(snaps))
    # the same rotation run with the term disabled: the pre-fix update, i.e. the
    # gated scalar kernel applied to the projected (here: zero) strain rate.
    nx = zeros(sq.N); ny = zeros(sq.N); nz = fill(T0, sq.N)
    off_vals = zeros(3, length(snaps))
    j = 0
    for k in 1:nstep
        QuaST._advance_maxwell_stress!(txx, tyy, txy, sq, ux, uz, dt, eta, G0)
        for g in eachindex(nx)
            trn = eps0.exx[g] + eps0.ezz[g]
            nx[g] = maxwell_stress_step(nx[g], eps0.exx[g] - trn / 3.0, dt,
                                        eta0, G0)
            ny[g] = maxwell_stress_step(ny[g], eps0.ezz[g] - trn / 3.0, dt,
                                        eta0, G0)
            nz[g] = maxwell_stress_step(nz[g], eps0.exz[g], dt, eta0, G0)
        end
        if j < length(snaps) && k == snaps[j + 1]
            j += 1
            run_vals[:, j] = [txx[1], tyy[1], txy[1]]
            off_vals[:, j] = [nx[1], ny[1], nz[1]]
        end
    end
    # the field is uniform and trace-free throughout (D = 0, tau_zz = 0)
    @test maximum(abs.(txx .- txx[1])) <= 1.0e-15 * T0
    @test maximum(abs.(tyy .- tyy[1])) <= 1.0e-15 * T0
    @test maximum(abs.(txy .- txy[1])) <= 1.0e-15 * T0
    @test maximum(abs.(txx .+ tyy)) <= 1.0e-15 * T0
    comp_err = zeros(length(snaps))       # co-rotational, relative
    comp_err_n = zeros(length(snaps))     # term disabled, relative
    ori_err = zeros(length(snaps))
    ori_err_n = zeros(length(snaps))
    mag_err_n = zeros(length(snaps))
    for (jj, k) in enumerate(snaps)
        t = k * dt
        ana = g250_analytic(t)
        scale = maximum(abs.(ana))
        comp_err[jj] = maximum(abs.(run_vals[:, jj] .- ana)) / scale
        comp_err_n[jj] = maximum(abs.(off_vals[:, jj] .- ana)) / scale
        ori_err[jj] = abs(wrap_pi(orientation(run_vals[:, jj]...)
                                  - orientation(ana...)))
        ori_err_n[jj] = abs(wrap_pi(orientation(off_vals[:, jj]...)
                                    - orientation(ana...)))
        mag_err_n[jj] = abs(fnorm(off_vals[:, jj]...) - fnorm(ana...)) / fnorm(ana...)
    end
    # (1) the co-rotational run follows the analytic objective solution
    @test maximum(comp_err) <= 1.0e-3      # measured 6.2e-5, see header
    @test maximum(ori_err) <= 1.0e-5       # measured 1.6e-9 rad, see header
    # the principal axis really turns counter-clockwise with the material: the
    # analytic curve moves it by Omega*t, pi/4 over the whole run.  Compare the
    # *unwrapped* orientation sequence for monotonicity and the wrapped angular
    # increment for the value (a principal axis is defined mod pi).
    oris = [orientation(run_vals[:, jj]...) for jj in eachindex(snaps)]
    ori0 = orientation(0.0, 0.0, T0)        # pi/4: axes at 45 degrees
    @test all(diff(oris) .> 0.0)
    @test abs(wrap_pi(oris[end] - ori0) - T_rot) <= 1.0e-5
    # the observable is concrete: the analytic stress is diagonal in the grid
    # at the last snapshot, and the disabled run still has its shear component
    ana_end = g250_analytic(T_rot)
    @test abs(ana_end[3]) <= 1.0e-12 * T0
    @test abs(off_vals[3, end]) > 0.5 * T0
    # (2) the negative control: magnitude right, components/orientation wrong
    @test maximum(mag_err_n) <= 1.0e-2     # the magnitude still decays correctly
    @test maximum(comp_err_n) >= 0.1       # components do not rotate
    @test maximum(ori_err_n) >= 0.5        # pi/4 of orientation is missing
    @test maximum(comp_err) <= 0.1 * maximum(comp_err_n)   # the discriminator

    # ---- (c) a spin-free pure shear is unchanged to round-off ------------
    gam = 0.3
    uxp = [gam * y for (_, y) in sq.coords]
    uzp = [gam * x for (x, _) in sq.coords]
    wp = QuaST.semquad_spin_component(sq, uxp, uzp)
    @test maximum(abs, wp) <= 1.0e-14      # measured ~6e-16 (GLL round-off)
    epsp = QuaST.semquad_strain_rate_components(sq, uxp, uzp)
    @test maximum(abs, epsp.exx) <= 1.0e-14
    @test maximum(abs, epsp.ezz) <= 1.0e-14
    @test maximum(abs.(epsp.exz .- gam)) <= 1.0e-14
    eta_p = fill(2.0, sq.N)
    G_p = 50.0
    dt_p = 1.0e-3
    n_p = 200
    px = zeros(sq.N); py = zeros(sq.N); pz = zeros(sq.N)   # with the term
    sx = zeros(sq.N); sy = zeros(sq.N); sz = zeros(sq.N)   # spin-free replay
    trp = epsp.exx .+ epsp.ezz
    for _ in 1:n_p
        QuaST._advance_maxwell_stress!(px, py, pz, sq, uxp, uzp, dt_p, eta_p, G_p)
        for g in eachindex(sx)
            sx[g] = maxwell_stress_step(sx[g], epsp.exx[g] - trp[g] / 3.0, dt_p,
                                        2.0, G_p)
            sy[g] = maxwell_stress_step(sy[g], epsp.ezz[g] - trp[g] / 3.0, dt_p,
                                        2.0, G_p)
            sz[g] = maxwell_stress_step(sz[g], epsp.exz[g], dt_p, 2.0, G_p)
        end
    end
    p_scale = max(maximum(abs, px), maximum(abs, py), maximum(abs, pz))
    spin_diff = max(maximum(abs.(px .- sx)), maximum(abs.(py .- sy)),
                    maximum(abs.(pz .- sz)))
    @test p_scale > 0.1                    # not the 0 == 0 case
    @test spin_diff <= 1.0e-14 * p_scale

    # ---- (d) the analytic co-rotational reference is itself pinned --------
    # (it is the reference G246 (b) and G249 (a) were amended to; comparing the
    # amended gates against a wrong reference formula would not be caught there,
    # so this block compares it with the composed update directly)
    # wxy = 0 is the gated no-spin curve
    for t in (0.3, 2.0), (e1, e2, e3) in ((0.3, -0.12, 0.2), (-0.2, 0.05, -0.4))
        c = maxwell_stress_exact_corotational(t, e1, e2, e3, 0.0, 2.0, 50.0)
        s = (maxwell_stress_exact(t, e1, 2.0, 50.0),
             maxwell_stress_exact(t, e2, 2.0, 50.0),
             maxwell_stress_exact(t, e3, 2.0, 50.0))
        @test maximum(abs.(c .- s)) <= 1.0e-14 * max(maximum(abs.(s)), 1.0e-300)
    end
    # G = 0 is off, as in maxwell_stress_exact
    @test maxwell_stress_exact_corotational(1.0, 0.3, -0.12, 0.2, 0.7, 2.0,
                                            0.0) == (0.0, 0.0, 0.0)
    # a fine recursion of the composed kernel converges to it (first order)
    Er_xx, Er_zz, Er_xz, w_r = 0.24, -0.18, 0.20, 0.30   # G246 (b) constants
    eta_r, G_r = 2.0, 50.0
    tauM_r = eta_r / G_r
    ref = maxwell_stress_exact_corotational(tauM_r, Er_xx, Er_zz, Er_xz, w_r,
                                            eta_r, G_r)
    n_fine = 100_000
    dt_fine = tauM_r / n_fine
    frx = 0.0; fry = 0.0; frz = 0.0
    for _ in 1:n_fine
        rx, ry, rz = maxwell_stress_corotate(frx, fry, frz, w_r, dt_fine)
        frx = maxwell_stress_step(rx, Er_xx, dt_fine, eta_r, G_r)
        fry = maxwell_stress_step(ry, Er_zz, dt_fine, eta_r, G_r)
        frz = maxwell_stress_step(rz, Er_xz, dt_fine, eta_r, G_r)
    end
    ref_err = maximum(abs.((frx, fry, frz) .- ref)) / maximum(abs.(ref))
    @test ref_err <= 1.0e-4                # measured ~3e-6, first order in dt

    @printf("G250: rotation Wxy=%+.3f tauM/T=%.1f dt=T/%d ; |tau-tau_ana|/|tau_ana| corot %.1e vs disabled %.1e ; orientation err corot %.1e rad vs disabled %.1e rad (tol 1e-5) ; disabled magnitude err %.1e ; W=0 round-off %.1e ; reference vs fine recursion %.1e\n",
            w[1], tauM / T_rot, nstep, maximum(comp_err),
            maximum(comp_err_n), maximum(ori_err), maximum(ori_err_n),
            maximum(mag_err_n), spin_diff / p_scale, ref_err)
    @printf("G250 (b) rigid rotation: Omega=%.1f, T=pi/4, tauM=%.4f, tau0=T0[[0,1],[1,0]], T0=%.1f\n",
            Omega, tauM, T0)
    for (jj, k) in enumerate(snaps)
        t = k * dt
        ana = g250_analytic(t)
        @printf("  t=%.4f analytic=(%+.6f, %+.6f, %+.6f) run=(%+.6f, %+.6f, %+.6f) comp-rel=%.2e orientation run=%.6f analytic=%.6f disabled=%.6f | disabled comp-rel=%.3f\n",
                t, ana..., run_vals[:, jj]..., comp_err[jj],
                orientation(run_vals[:, jj]...), orientation(ana...),
                orientation(off_vals[:, jj]...), comp_err_n[jj])
    end
end
