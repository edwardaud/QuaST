# G249: the Maxwell stress is deviatoric -- the update projects the strain rate.
#
# `_advance_maxwell_stress!` used to feed the *raw* strain-rate components into
# the Maxwell update (`txx <- step(exx)`, `tyy <- step(ezz)`).  The Maxwell
# stress is the deviatoric stress, so its input is
#
#     ehat_ij = eps_ij - (1/3) tr(eps) delta_ij,   tr(eps) = exx + ezz,
#
# where the out-of-plane normal strain is zero in 2-D plane strain but still
# enters the trace (so the third deviatoric normal strain rate is -tr/3).  The
# missing projection accumulated a spurious isotropic stress which the elastic
# force `div(a tau)` realised as a pressure-like load, and the old gates could
# not see it: G246 (b) was believed to drive a solenoidal field but its trace is
# 0.18, and its expected values were computed with the same raw mapping, so a
# wrong update was compared against a wrong reference (the lesson: the reference
# must come from the physics, not from the code under test).
#
# The update also carries the co-rotational (Jaumann) term now (G250), which
# couples the components: the shear equation reads the normal difference
# `W_xy (tau_yy - tau_xx)` and the normal equations read the shear.  Two things
# in this gate moved with it, and both are recorded at their sites: the (a)/(c)
# references became the analytic objective curve / co-rotational fixed point
# (G250 pins both), and the raw-mapping control now includes the same rotation
# so that it isolates the *projection* defect, which is what this gate is about.
# The trace, the first elastic step and the `G = 0` invariants are untouched.
#
# The stored state is still three in-plane components: `tau_zz` is not stored
# but implied by trace zero, `tau_zz = -(tau_xx + tau_yy)`.  The 2-D elastic
# force term has no out-of-plane normal component -- `div(a tau)` is
# `(d_x(a tau_xx) + d_y(a tau_xy), d_x(a tau_xy) + d_y(a tau_yy))`, which is
# exactly what `src/stokes/assembly_var.jl` assembles from the three fields --
# so no assembly ever needs the fourth component.  The co-rotational term is a
# pure out-of-plane rotation generator: it leaves `tau_zz` and the in-plane trace
# invariant, so it needs no projection either.
#
# This gate pins:
#   (a) a prescribed **non-solenoidal** velocity: the three updated components
#       follow the *objective* analytic curve against their deviatoric strain
#       rate, and are measurably away from what the raw (pre-projection) update
#       would give (the test the old gate lacked); for an (analytically)
#       solenoidal field the projection only removes a round-off trace, so the
#       update agrees with the raw-mapping control to round-off -- which is why
#       such a field would hide the bug;
#   (b) trace zero of the stored triple, with the out-of-plane component evolved
#       independently through the kernel from -tr/3 (so it is not the tautology
#       `-(txx+tyy)`);
#   (c) a driver run with `pr.deviatoric = true`: the elastic half and the
#       viscous half carry the same deviatoric measure, so the stress
#       equilibrates to the co-rotational fixed point built on `2 eta ehat` (the
#       spin-free `tau = 2 eta ehat` moved by G250, see there), the flow is the
#       material-viscosity one, nothing is counted twice, and the pre-fix raw
#       history is measurably inconsistent with the operator that drives it;
#   (d) `shear_modulus = 0` still disables every stress path (the full
#       bit-identity invariants are the G246/G247 suites, which must pass with
#       their amended expectations).
using Test, QuaST, LinearAlgebra, Printf

@testset "G249 deviatoric stress projection" begin
    z2 = (x, y) -> 0.0
    rel(a, b) = norm(a .- b) / max(norm(b), 1.0e-300)

    function g249_tree(nref::Int)
        qt = Quadtree(2)
        for _ in 1:nref, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        return qt
    end

    # ---- (a) a prescribed non-solenoidal velocity -------------------------
    # A quadratic velocity field (represented exactly by p = 2, so its nodal
    # strain-rate components are the analytic ones) with a trace that does not
    # vanish anywhere:
    #
    #     exx = 0.30 + 0.40 y      ezz = -0.12 + 0.14 y
    #     exz = 0.20 - 0.10 x      tr  = exx + ezz = 0.18 + 0.54 y
    #
    # hence ehat_xx = 0.24 + 0.22 y and ehat_zz = -0.18 - 0.04 y: the deviatoric
    # and raw normal components differ by 20-34% (xx) and by 50% or more (zz,
    # whose raw value passes through zero), so the assertions below cannot be
    # satisfied by the pre-fix code.
    qa = g249_tree(2)
    ssa = make_semstokes(qa, 2)
    sqa = ssa.su
    uxa = [0.30 * x - 0.10 * y + 0.40 * x * y + 0.02 for (x, y) in sqa.coords]
    uza = [0.50 * x - 0.12 * y - 0.30 * x^2 + 0.07 * y^2 - 0.03
           for (x, y) in sqa.coords]
    exx_a = [0.30 + 0.40 * y for (_, y) in sqa.coords]
    ezz_a = [-0.12 + 0.14 * y for (_, y) in sqa.coords]
    exz_a = [0.20 - 0.10 * x for (x, _) in sqa.coords]
    tr_a = exx_a .+ ezz_a
    ehat_xx = exx_a .- tr_a ./ 3
    ehat_zz = ezz_a .- tr_a ./ 3
    ehat_out = .-tr_a ./ 3                    # out-of-plane deviatoric part
    epsa = QuaST.semquad_strain_rate_components(sqa, uxa, uza)
    # the gate's own field is what it thinks it is -- and the accessor still
    # returns the *raw* components (the projection belongs to the update)
    @test maximum(abs.(epsa.exx .- exx_a)) < 1.0e-12
    @test maximum(abs.(epsa.ezz .- ezz_a)) < 1.0e-12
    @test maximum(abs.(epsa.exz .- exz_a)) < 1.0e-12
    @test maximum(abs.(epsa.exx .+ epsa.ezz .- tr_a)) < 1.0e-12
    @test maximum(abs, tr_a) > 0.2                     # non-solenoidal
    @test maximum(abs.(ehat_xx .+ ehat_zz .+ ehat_out)) < 1.0e-15
    # The co-rotational term (G250) makes this field rotate: its spin is the
    # analytic W_xy = ((dux/dy) - (duz/dx))/2 = (-0.1+0.4x - 0.5+0.6x)/2 =
    # -0.30 + 0.50 x, which the accessor has to reproduce (G250 pins the
    # accessor against a rigid rotation; here it is a spatially varying field).
    w_a = [-0.30 + 0.50 * x for (x, _) in sqa.coords]
    @test maximum(abs.(QuaST.semquad_spin_component(sqa, uxa, uza) .- w_a)) <
          1.0e-12
    eta_a = 2.0
    G_a = 50.0
    tauM_a = eta_a / G_a
    nstep = 500
    dt_a = tauM_a / nstep
    eta_field = fill(eta_a, sqa.N)
    # accumulate to t = tau_M, where the analytic curve is
    # `2 eta ehat (1 - 1/e)` for a rotation-free flow; with the spin it is the
    # objective curve `maxwell_stress_exact_corotational` (the reference G250
    # pins against the composed kernel).
    txx_a = zeros(sqa.N); tyy_a = zeros(sqa.N); txy_a = zeros(sqa.N)
    for _ in 1:nstep
        QuaST._advance_maxwell_stress!(txx_a, tyy_a, txy_a, sqa, uxa, uza,
                                       dt_a, eta_field, G_a)
    end
    dev_ref = [maxwell_stress_exact_corotational(tauM_a, ehat_xx[i],
                                                 ehat_zz[i], exz_a[i], w_a[i],
                                                 eta_a, G_a)
               for i in eachindex(ehat_xx)]
    dev_xx = [r[1] for r in dev_ref]
    dev_zz = [r[2] for r in dev_ref]
    dev_xz = [r[3] for r in dev_ref]
    @test maximum(abs.(txx_a .- dev_xx)) <= 2.0e-3 * maximum(abs, dev_xx)
    @test maximum(abs.(tyy_a .- dev_zz)) <= 2.0e-3 * maximum(abs, dev_zz)
    @test maximum(abs.(txy_a .- dev_xz)) <= 2.0e-3 * maximum(abs, dev_xz)
    # ... and the raw references are not close: this is the assertion the old
    # gate would fail.  The normal components are off by tr/3, which is 25% of
    # ehat_xx and 33% of ehat_zz at the smallest-trace node.  (The raw
    # references are the same objective curve driven by the *raw* strain rate,
    # so this control still isolates the projection, not the rotation.)
    raw_ref = [maxwell_stress_exact_corotational(tauM_a, exx_a[i], ezz_a[i],
                                                 exz_a[i], w_a[i], eta_a, G_a)
               for i in eachindex(exx_a)]
    raw_xx = [r[1] for r in raw_ref]
    raw_zz = [r[2] for r in raw_ref]
    @test minimum(abs.(raw_xx .- dev_xx) ./ abs.(dev_xx)) > 0.20
    @test minimum(abs.(raw_zz .- dev_zz) ./ abs.(dev_zz)) > 0.20
    @test maximum(abs.(txx_a .- raw_xx)) > 0.15 * maximum(abs, dev_xx)
    @test maximum(abs.(tyy_a .- raw_zz)) > 0.15 * maximum(abs, dev_zz)
    # the elastic first step from rest is 2 G ehat dt (not 2 G eps dt)
    dt_s = 1.0e-4 * tauM_a
    txxa = zeros(sqa.N); tyya = zeros(sqa.N); txya = zeros(sqa.N)
    QuaST._advance_maxwell_stress!(txxa, tyya, txya, sqa, uxa, uza, dt_s,
                                   eta_field, G_a)
    @test maximum(abs.(txxa .- 2.0 .* G_a .* ehat_xx .* dt_s)) <=
          1.0e-3 * maximum(abs.(2.0 .* G_a .* ehat_xx .* dt_s))
    @test maximum(abs.(tyya .- 2.0 .* G_a .* ehat_zz .* dt_s)) <=
          1.0e-3 * maximum(abs.(2.0 .* G_a .* ehat_zz .* dt_s))
    @test maximum(abs.(txya .- 2.0 .* G_a .* exz_a .* dt_s)) <=
          1.0e-3 * maximum(abs.(2.0 .* G_a .* exz_a .* dt_s))

    # The pre-fix update, as the control: the G244-gated scalar kernel mapped
    # over the *raw* components.  The co-rotational part is common to both sides
    # now (G250), so this isolates the projection defect the gate is about.  It
    # is expected to be round-off-identical when tr == 0 and measurably
    # different otherwise ("a gate must assert that behaviour changes").
    function g249_raw_replay(ux, uz, tx, ty, tz)
        eps = QuaST.semquad_strain_rate_components(sqa, ux, uz)
        w = QuaST.semquad_spin_component(sqa, ux, uz)
        for _ in 1:nstep, g in eachindex(tx)
            rxx, ryy, rxy = maxwell_stress_corotate(tx[g], ty[g], tz[g],
                                                    w[g], dt_a)
            tx[g] = maxwell_stress_step(rxx, eps.exx[g], dt_a, eta_a, G_a)
            ty[g] = maxwell_stress_step(ryy, eps.ezz[g], dt_a, eta_a, G_a)
            tz[g] = maxwell_stress_step(rxy, eps.exz[g], dt_a, eta_a, G_a)
        end
        return tx, ty, tz
    end
    rp_x, rp_y, rp_z = g249_raw_replay(uxa, uza, zeros(sqa.N), zeros(sqa.N),
                                       zeros(sqa.N))
    @test maximum(abs.(txx_a .- rp_x)) > 0.15 * maximum(abs, dev_xx)
    @test maximum(abs.(tyy_a .- rp_y)) > 0.15 * maximum(abs, dev_zz)
    # The shear component is not projected by the *strain-rate* projection: the
    # raw and deviatoric drives differ by an isotropic offset in the two normal
    # components only, and the shear equation sees their difference
    # `W_xy (tau_yy - tau_xx)`, which the offset cancels.  Since G250's term the
    # two histories are therefore equal to round-off, not bit-identical (the
    # offset is a common floating-point shift of the two normal components).
    @test maximum(abs.(rp_z .- txy_a)) <= 1.0e-12 * maximum(abs, txy_a)
    # the old raw history's normal trace is the spurious isotropic stress
    @test maximum(abs.(rp_x .+ rp_y)) > 0.5 * maximum(abs, dev_xx)
    @printf("G249 (a): tr in [%.3f, %.3f], ehat_xx/exx in [%.3f, %.3f]; |raw-dev| xx = %.2f%% of |dev|; |old txx+tyy|/|dev| = %.2f; |raw-projected| shear = %.1e (round-off)\n",
            minimum(tr_a), maximum(tr_a), minimum(ehat_xx ./ exx_a),
            maximum(ehat_xx ./ exx_a),
            100 * minimum(abs.(raw_xx .- dev_xx) ./ abs.(dev_xx)),
            maximum(abs.(rp_x .+ rp_y)) / maximum(abs, dev_xx),
            maximum(abs.(rp_z .- txy_a)))
    # an (analytically) solenoidal linear field: the projection then only
    # removes a round-off trace, so the update equals the raw-mapping control to
    # round-off -- an old gate built on such a field could never have detected
    # the missing projection.  (Exact bit-identity is not constructible through
    # the interpolated derivative: the GLL differentiation rows sum to zero only
    # to round-off, so `exx + ezz` is ~1e-16, not 0.0.)
    uxs = [0.30 * x + 0.50 * y + 0.02 for (x, y) in sqa.coords]
    uzs = [-0.10 * x - 0.30 * y - 0.03 for (x, y) in sqa.coords]
    txs = zeros(sqa.N); tys = zeros(sqa.N); tzs = zeros(sqa.N)
    for _ in 1:nstep
        QuaST._advance_maxwell_stress!(txs, tys, tzs, sqa, uxs, uzs, dt_a,
                                       eta_field, G_a)
    end
    rs_x, rs_y, rs_z = g249_raw_replay(uxs, uzs, zeros(sqa.N), zeros(sqa.N),
                                       zeros(sqa.N))
    tsmax = max(maximum(abs, txs), maximum(abs, tys), maximum(abs, tzs))
    @test maximum(abs.(txs .- rs_x)) <= 1.0e-12 * tsmax
    @test maximum(abs.(tys .- rs_y)) <= 1.0e-12 * tsmax
    @test maximum(abs.(tzs .- rs_z)) <= 1.0e-12 * tsmax
    @test maximum(abs, txs) > 0.0                    # not the 0 == 0 case

    # ---- (b) trace zero of the stored triple, out-of-plane implied --------
    # Evolve the *out-of-plane* component independently through the kernel from
    # its deviatoric strain rate -tr/3.  The stored triple then has to satisfy
    # `txx + tyy + tzz == 0` to round-off; that is the content of
    # `tau_zz = -(tau_xx + tau_yy)`, and it is not a tautology because `tzz`
    # never sees `txx`/`tyy`.  A 2-D plane-*stress* projection (divide by 2)
    # would give `txx + tyy == 0` and leave `tzz` alone, i.e. a trace of order
    # |tzz|, so the assertion also rules that convention out.
    tzz = zeros(sqa.N)
    for _ in 1:nstep, g in eachindex(tzz)
        tzz[g] = maxwell_stress_step(tzz[g], ehat_out[g], dt_a, eta_a, G_a)
    end
    dev_zz_out = [maxwell_stress_exact(tauM_a, e, eta_a, G_a) for e in ehat_out]
    @test maximum(abs.(tzz .- dev_zz_out)) <= 2.0e-3 * maximum(abs, dev_zz_out)
    tmax_b = max(maximum(abs, txx_a), maximum(abs, tyy_a), maximum(abs, tzz))
    @test maximum(abs.(txx_a .+ tyy_a .+ tzz)) <= 1.0e-12 * tmax_b
    @test maximum(abs, tzz) > 0.05 * tmax_b           # the check is not vacuous
    @printf("G249 (b): |tau_xx+tau_yy+tau_zz| = %.2e (scale %.1e, relative %.1e); tzz/tmax = %.3f\n",
            maximum(abs.(txx_a .+ tyy_a .+ tzz)), tmax_b,
            maximum(abs.(txx_a .+ tyy_a .+ tzz)) / tmax_b,
            maximum(abs, tzz) / tmax_b)

    # ---- (c) driver: pr.deviatoric = true, the two halves agree -----------
    # Lid-driven cavity with the deviatoric operator (`2 eta ehat : eps(v)` in
    # the stiffness) and G > 0.  R = G dt >> eta equilibrates the stress; the
    # backward-Euler split with the co-rotational term (G250) is
    #
    #     tau^{n+1} = 2 eta_eff ehat^{n+1} + a R(tau^n),
    #         R(tau) = tau + dt (W tau - tau W),
    #
    # whose fixed point is the Jaumann steady stress
    # `tau = 2 eta ehat + tau_M (W tau - tau W)`, not `tau = 2 eta ehat`.  The
    # deviatoric content survives: the in-plane trace and the implied `tau_zz`
    # are invariant under `R`, so the trace assertions below are unchanged.  The
    # viscous half and the elastic half still carry one and the same deviatoric
    # measure, nothing is counted twice, and the flow is still the
    # material-viscosity one (the velocity numbers below are the ones the gate
    # measured before the term landed).  The *stress* reference moved from
    # `2 eta ehat` to the co-rotational fixed point by 1.2e-2 of tmax -- real
    # rotation physics at this card's |W| dt, not a numerical artifact.
    #
    # The run takes 10 steps instead of 5: the rotation channel of the update
    # contracts at `a sqrt(1 + (2 W dt)^2)` (1.2e-2 per step here), so 5 steps
    # leave the stress 2.8e-10 from its fixed point and the 1e-12 tolerance
    # unchanged; 10 steps reach machine precision.  No tolerance was loosened.
    g249_eta_mult = (x, y) -> 1.0 + 0.5 * sinpi(x) * sinpi(y)
    function g249_card(; G::Float64, nref::Int, deviatoric::Bool)
        qt = g249_tree(nref)
        ss = make_semstokes(qt, 2)
        pr = Params(; Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0,
                    eta_multiplier=g249_eta_mult, shear_modulus=G, bc=:free,
                    wall_bc=(:free, :free, :free, :rigid),
                    wall_u=(nothing, nothing, nothing, (x, y) -> 1.0),
                    deviatoric=deviatoric)
        return qt, ss, pr
    end
    # co-rotational (Jaumann) fixed point of the same map (G250):
    # tau = 2 eta ehat + tau_M (W tau - tau W), closed form in the plane.
    function g249_corot_fixed_point(ehat_xx, ehat_zz, ehat_xz, w, eta, tauM)
        den = 1.0 .+ 4.0 .* tauM .^ 2 .* w .^ 2
        txy = (2.0 .* eta .* ehat_xz .+
               2.0 .* eta .* tauM .* w .* (ehat_zz .- ehat_xx)) ./ den
        txx = 2.0 .* eta .* ehat_xx .+ 2.0 .* tauM .* w .* txy
        tyy = 2.0 .* eta .* ehat_zz .- 2.0 .* tauM .* w .* txy
        return txx, tyy, txy
    end
    G_c = 1000.0
    R_c = 1000.0
    dt_c = R_c / G_c
    qtc, ssc, prc = g249_card(; G=G_c, nref=3, deviatoric=true)
    Nc = ssc.su.N
    rc = run_semquad_simulation(qtc, prc, zeros(Float64, Nc); p=2,
                                maxsteps=10, tstop=1.0e9, dt0=dt_c, dtmax=dt_c,
                                dtmin=1.0e-30, cfl=0.0, linear=:direct, ev=0.0,
                                maxpic=1)
    sqc = rc.ss.su
    epsc = QuaST.semquad_strain_rate_components(sqc, rc.ux, rc.uz)
    tr_c = epsc.exx .+ epsc.ezz
    ehatc_xx = epsc.exx .- tr_c ./ 3
    ehatc_zz = epsc.ezz .- tr_c ./ 3
    ehatc_out = .-tr_c ./ 3
    w_c = QuaST.semquad_spin_component(sqc, rc.ux, rc.uz)
    leaves = Set(sqc.leaves)
    thz = zeros(Float64, sqc.N)
    eta_mat(x, y) = QuaST.semquad_eta_at(sqc, thz, x, y, prc, leaves) *
                    prc.eta_multiplier(x, y)
    eta_nodes = [eta_mat(c[1], c[2]) for c in sqc.coords]
    tmax_c = max(maximum(abs, rc.txx), maximum(abs, rc.tyy),
                 maximum(abs, rc.txy))
    tauM_c = eta_nodes ./ G_c
    # 1. the fixed point is the material deviatoric stress plus the
    #    co-rotational (Jaumann) increment -- the update's own steady state
    tx_fp, ty_fp, tz_fp = g249_corot_fixed_point(ehatc_xx, ehatc_zz, epsc.exz,
                                                 w_c, eta_nodes, tauM_c)
    tau_dev_err = max(maximum(abs.(rc.txx .- tx_fp)),
                      maximum(abs.(rc.tyy .- ty_fp)),
                      maximum(abs.(rc.txy .- tz_fp)))
    @test tau_dev_err / tmax_c < 1.0e-12
    # 2. and it is traceless, with the out-of-plane component implied (the
    #    co-rotational term leaves the in-plane trace and tau_zz invariant)
    trace_c = maximum(abs.(rc.txx .+ rc.tyy .+
                            2 .* eta_nodes .* ehatc_out))
    @test trace_c / tmax_c < 1.0e-12
    # 3. the pre-fix raw fixed point (2 eta eps) is far away at this mesh: the
    # discrete velocity is not divergence-free, so the projection is active
    tau_raw_err = max(maximum(abs.(rc.txx .- 2 .* eta_nodes .* epsc.exx)),
                      maximum(abs.(rc.tyy .- 2 .* eta_nodes .* epsc.ezz)))
    @test tau_raw_err / tmax_c > 0.01
    eps_scale_c = max(maximum(abs, epsc.exx), maximum(abs, epsc.ezz),
                      maximum(abs, epsc.exz))
    @test maximum(abs, tr_c) > 0.01 * eps_scale_c
    # 4. the flow is the material-viscosity solution; the same fixed-point map
    # started from the run's fixed-point stress (the discretisation floor) bounds
    # the difference between the effective-viscosity operator, the material one
    # and the run.
    function g249_solve_ref(eta_fun; elastic=nothing)
        return solve_semquad_stokes_var(rc.ss, rc.qt, eta_fun, z2, z2, z2, z2;
                                        bc=prc.bc, wall_bc=prc.wall_bc,
                                        wall_u=prc.wall_u, deviatoric=true,
                                        linear=:direct,
                                        elastic_stress=elastic)[1]
    end
    u_mat = g249_solve_ref(eta_mat)
    eta_eff_fn(x, y) = begin
        e = eta_mat(x, y)
        1.0 / (1.0 / e + 1.0 / (G_c * dt_c))
    end
    u_eff = g249_solve_ref(eta_eff_fn)
    # the frozen backward-Euler load at the (objective) projected fixed point
    el_fp = maxwell_force_stress(tx_fp, ty_fp, tz_fp, eta_nodes, dt_c, G_c)
    u_def = g249_solve_ref(eta_eff_fn; elastic=el_fp)
    # the pre-fix load: the same map with the raw strain rate in the history --
    # its co-rotational fixed point, so the control isolates the projection (the
    # rotation is common to both histories since G250)
    tx_raw, ty_raw, tz_raw = g249_corot_fixed_point(epsc.exx, epsc.ezz,
                                                    epsc.exz, w_c, eta_nodes,
                                                    tauM_c)
    el_raw = maxwell_force_stress(tx_raw, ty_raw, tz_raw, eta_nodes, dt_c, G_c)
    u_old = g249_solve_ref(eta_eff_fn; elastic=el_raw)
    # the two histories are different tensors: the raw one adds the trace of the
    # strain rate to the normal components, so it asks the momentum equation for
    # a different measure than the deviatoric stiffness uses.  (The isotropic
    # part is pressure-like, hence the small velocity residual above; the sharp
    # evidence is the stress fixed point, this line sizes what changed.)
    el_stored = maxwell_force_stress(rc.txx, rc.tyy, rc.txy, eta_nodes, dt_c, G_c)
    rel_tau = max(rel(el_stored[1], el_raw[1]), rel(el_stored[2], el_raw[2]),
                  rel(el_stored[3], el_raw[3]))
    rel_mat_eff = rel(u_mat[1:Nc], u_eff[1:Nc])
    rel_run_mat = rel(rc.ux, u_mat[1:Nc])
    rel_def_mat = rel(u_def[1:Nc], u_mat[1:Nc])
    rel_old_mat = rel(u_old[1:Nc], u_mat[1:Nc])
    @test rel_mat_eff > 1.0e-6            # material and 2a differ: non-vacuous
    @test rel_def_mat > 1.0e-8            # the interpolated-stress defect is real
    @test rel_tau > 1.0e-3                # the measures differ materially
    @test rel_run_mat < 1.0e-3            # the run is the material flow ...
    @test rel_run_mat <= 2.0 * rel_def_mat      # ... to the defect floor
    @test rel_run_mat < rel_mat_eff             # and not the 2a one
    @test rel_old_mat > 1.1 * rel_run_mat       # the raw map is farther away
    @printf("G249 (c): |tau-tau_fp|/tmax=%.1e (max tauM|W|=%.1e) |trace|/tmax=%.1e |raw fixpoint|/tmax=%.2f rel(run,mat)=%.2e defect=%.2e rel(oldraw,mat)=%.2e rel(tau,raw)=%.2e rel(mat,2a)=%.2e\n",
            tau_dev_err / tmax_c, maximum(tauM_c .* abs.(w_c)),
            trace_c / tmax_c, tau_raw_err / tmax_c,
            rel_run_mat, rel_def_mat, rel_old_mat, rel_tau, rel_mat_eff)

    # ---- (d) the invariants ----------------------------------------------
    @test Params().shear_modulus == 0.0
    qd = g249_tree(1)
    ssd = make_semstokes(qd, 2)
    sqd = ssd.su
    uxd = [0.30 * x - 0.10 * y + 0.40 * x * y + 0.02 for (x, y) in sqd.coords]
    uzd = [0.50 * x - 0.12 * y - 0.30 * x^2 + 0.07 * y^2 - 0.03
           for (x, y) in sqd.coords]
    txd = fill(3.0, sqd.N); tyd = fill(-1.0, sqd.N); tzd = fill(0.5, sqd.N)
    QuaST._advance_maxwell_stress!(txd, tyd, tzd, sqd, uxd, uzd, 1.0e-3,
                                   fill(2.0, sqd.N), 0.0)
    @test txd == fill(3.0, sqd.N) && tyd == fill(-1.0, sqd.N) &&
          tzd == fill(0.5, sqd.N)          # G = 0: off, not elastic-zero
    @printf("G249: deviatoric projection active; trace = %.1e at tau_M; G = 0 leaves the state untouched\n",
            maximum(abs.(txx_a .+ tyy_a .+ tzz)))
end
