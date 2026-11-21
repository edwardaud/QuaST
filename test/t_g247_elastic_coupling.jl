# G247: the elastic force term, `div(tau)` in the Stokes right-hand side
# (docs/dev/PHASE2.md 2b step 2).
#
# The Maxwell stress history of 2b step 1 is now coupled to momentum: the
# backward-Euler update splits the stress into
#
#     tau^{n+1} = 2 eta_eff eps^{n+1} + a tau^n,   a = eta/(eta + G dt),
#
# the effective-viscosity part is already in the Stokes operator (2a), and the
# remaining `a tau^n` enters the load vector as `-int a tau^n : eps(v)` (the weak
# form of `div(a tau^n)`).  The driver advances the history *after* the step's
# momentum solve, so a solve always sees the previous step's stress.
#
# This gate pins, in order:
#   (a) `shear_modulus == 0` is bit-identical: no stress state, and an assembly
#       with an all-zero elastic tuple is bitwise the historical load vector;
#   (b) the algebra of the split (`maxwell_force_stress` and eta_eff + a*eta = eta)
#       and its guards;
#   (c) the assembled term is exactly `-int tau : eps(v)`: for a rotational
#       polynomial stress it reproduces the body force `div(tau)` to round-off,
#       and a uniform stress (div = 0) assembles to no load; the same load
#       reaches the matrix-free `:mg` assembly;
#   (d) the equilibrated limit through the real driver.  A lid-driven cavity
#       whose viscosity field is temperature-independent (`expT` with
#       `deta_T = 1` times `eta_multiplier`; `Ra = 0`), so the flow is steady and
#       the stress equilibrates to `2 eta eps`.  At `dt >> tau_M` the velocity
#       must agree with the 2a effective-viscosity solution; at `dt = 10 tau_M`
#       it must instead track the material-viscosity solution, proving the
#       coupling carries the right `a` rather than being absent or unscaled;
#   (e) the per-step order: the first solve is exactly the 2a solution (tau^0 =
#       0) and the second solve is reproduced bit for bit from `tau^1` advanced
#       from the *first* solve's velocity.
using Test, QuaST, LinearAlgebra, Printf

@testset "G247 elastic coupling" begin
    z2 = (x, y) -> 0.0
    rel(a, b) = norm(a .- b) / max(norm(b), 1.0e-300)
    function g247_tree(nref::Int)
        qt = Quadtree(4)
        for _ in 1:nref, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        return qt
    end
    # A viscosity field that does *not* depend on temperature: `:expT` with
    # `deta_T = 1` is exactly `eta = 1` for every T, and `eta_multiplier` gives
    # the spatial shape.  With `Ra = 0` the temperature then cannot feed back into
    # the momentum (no buoyancy) or into the viscosity, so the flow is steady even
    # though the temperature of a base profile advected by the lid is not.
    g247_eta_mult = (x, y) -> 1.0 + 0.5 * sinpi(x) * sinpi(y)
    function g247_card(; G::Float64, nref::Int=3)
        qt = g247_tree(nref)
        ss = make_semstokes(qt, 2)
        N = ss.su.N
        pr = Params(; Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0,
                    eta_multiplier=g247_eta_mult,
                    shear_modulus=G, bc=:free,
                    wall_bc=(:free, :free, :free, :rigid),
                    wall_u=(nothing, nothing, nothing, (x, y) -> 1.0))
        return qt, ss, N, pr
    end
    # `R = G*dt`; `dt >> tau_M = eta/G` means R >> eta (eta in [1, 1.5]).
    function g247_run(R::Float64; nsteps::Int, G::Float64=1000.0, nref::Int=3)
        dt = R / G
        qt, ss, N, pr = g247_card(; G=G, nref=nref)
        us = Vector{Vector{Float64}}()
        zs = Vector{Vector{Float64}}()
        cb = s -> begin
            push!(us, copy(s.ux))
            push!(zs, copy(s.uz))
            nothing
        end
        r = run_semquad_simulation(qt, pr, zeros(Float64, N); p=2,
                                   maxsteps=nsteps, tstop=1.0e9, dt0=dt,
                                   dtmax=dt, dtmin=1.0e-30, cfl=0.0,
                                   linear=:direct, ev=0.0, maxpic=1,
                                   callback=cb)
        return (r=r, us=us, zs=zs, dt=dt, G=G, pr=pr)
    end
    function g247_eta_closures(run)
        sq = run.r.ss.su
        leaves = Set(sq.leaves)
        thz = zeros(Float64, sq.N)
        eta_mat(x, y) = QuaST.semquad_eta_at(sq, thz, x, y, run.pr, leaves) *
                        run.pr.eta_multiplier(x, y)
        eta_eff(x, y) = begin
            e = eta_mat(x, y)
            1.0 / (1.0 / e + 1.0 / (run.G * run.dt))
        end
        return eta_mat, eta_eff
    end
    function g247_reference(run)
        eta_mat, eta_eff = g247_eta_closures(run)
        um = solve_semquad_stokes_var(run.r.ss, run.r.qt, eta_mat, z2, z2, z2,
                                      z2; bc=run.pr.bc,
                                      wall_bc=run.pr.wall_bc,
                                      wall_u=run.pr.wall_u,
                                      deviatoric=run.pr.deviatoric,
                                      linear=:direct)[1]
        ue = solve_semquad_stokes_var(run.r.ss, run.r.qt, eta_eff, z2, z2, z2,
                                      z2; bc=run.pr.bc,
                                      wall_bc=run.pr.wall_bc,
                                      wall_u=run.pr.wall_u,
                                      deviatoric=run.pr.deviatoric,
                                      linear=:direct)[1]
        return um, ue, eta_mat, eta_eff
    end
    # Nodal stress at the material solution, `2 eta eps(u)`, and one application
    # of the coupled map from it: the fixed-point defect measures how far the
    # nodal stress interpolation and the lag move the fixed point away from the
    # material-viscosity solution.  It is the discretisation floor the (d)
    # tolerances are justified against.
    function g247_defect(run, um, eta_nodes)
        sq = run.r.ss.su
        N = sq.N
        epsm = QuaST.semquad_strain_rate_components(sq, um[1:N], um[(N + 1):(2N)])
        tx = 2 .* eta_nodes .* epsm.exx
        ty = 2 .* eta_nodes .* epsm.ezz
        tz = 2 .* eta_nodes .* epsm.exz
        el = QuaST.maxwell_force_stress(tx, ty, tz, eta_nodes, run.dt, run.G)
        _, eta_eff = g247_eta_closures(run)
        ud = solve_semquad_stokes_var(run.r.ss, run.r.qt, eta_eff, z2, z2, z2,
                                      z2; bc=run.pr.bc,
                                      wall_bc=run.pr.wall_bc,
                                      wall_u=run.pr.wall_u,
                                      deviatoric=run.pr.deviatoric,
                                      linear=:direct,
                                      elastic_stress=el)[1]
        return ud
    end

    # ---- (a) shear_modulus = 0: no state, bit-identical assembly ----------
    # The historical path (G246(a) pattern): the default has no `shear_modulus`
    # key reachable above zero, the explicit 0 is the same run, and the stress
    # fields do not exist.  The full suite's bitwise regression gates (G150) pin
    # the pre-2b trajectory; here the same run twice pins the driver wiring.
    qt0, ss0, N0, pr0 = g247_card(; G=0.0, nref=2)
    function g247_g0_run(pr)
        run_semquad_simulation(g247_tree(2), pr, zeros(Float64, N0);
                               p=2, maxsteps=3, tstop=1.0e9,
                               dt0=1.0e-3, dtmax=1.0e-3,
                               dtmin=1.0e-30, cfl=0.0,
                               linear=:direct, ev=0.0, maxpic=1)
    end
    prdefault = Params(; Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0,
                       eta_multiplier=g247_eta_mult, bc=:free,
                       wall_bc=(:free, :free, :free, :rigid),
                       wall_u=(nothing, nothing, nothing,
                               (x, y) -> 1.0))
    @test prdefault.shear_modulus == 0.0
    rg0 = g247_g0_run(prdefault)
    rg0b = g247_g0_run(pr0)
    @test rg0.txx === nothing && rg0.tyy === nothing && rg0.txy === nothing
    @test rg0.theta == rg0b.theta
    @test rg0.ux == rg0b.ux && rg0.uz == rg0b.uz && rg0.pv == rg0b.pv
    @test rg0.history == rg0b.history
    @test rg0.weak == false
    # an all-zero elastic tuple is a bitwise no-op in the load vector: this is the
    # "not assembled when the term is zero" property at the assembly boundary
    # (`elastic_stress = nothing` is what the driver passes for G = 0, so the
    # branch is compiled out entirely; G188/G202 pin the nothing-path bits).
    fv0 = solve_semquad_stokes_var(ss0, qt0, (x, y) -> 1.0, z2, z2, z2, z2;
                                   assemble_only=true, bc=:free)[:fv]
    fvz = solve_semquad_stokes_var(ss0, qt0, (x, y) -> 1.0, z2, z2, z2, z2;
                                   assemble_only=true, bc=:free,
                                   elastic_stress=(zeros(N0), zeros(N0),
                                                   zeros(N0)))[:fv]
    @test fv0 == fvz

    # ---- (b) the algebra of the backward-Euler split ---------------------
    eta_b = [0.5, 1.0, 2.0, 4.0]
    txb = [1.0, -2.0, 3.0, -4.0]
    tyb = [-1.0, 2.0, -3.0, 4.0]
    tzb = [0.1, 0.2, -0.3, 0.4]
    Gb, dtb = 100.0, 1.0e-2
    a_b = eta_b ./ (eta_b .+ Gb * dtb)
    out = QuaST.maxwell_force_stress(txb, tyb, tzb, eta_b, dtb, Gb)
    @test out[1] == a_b .* txb
    @test out[2] == a_b .* tyb
    @test out[3] == a_b .* tzb
    # the identity that makes the equilibrated limit exact: the operator's
    # effective viscosity plus the explicit part is the material viscosity
    eta_eff_b = 1.0 ./ (1.0 ./ eta_b .+ 1.0 ./ (Gb * dtb))
    @test maximum(abs.(eta_eff_b .+ a_b .* eta_b .- eta_b)) < 1.0e-14
    @test_throws ArgumentError QuaST.maxwell_force_stress(txb, tyb, tzb, eta_b,
                                                          dtb, 0.0)
    @test_throws ArgumentError QuaST.maxwell_force_stress(txb, tyb, tzb, eta_b,
                                                          0.0, Gb)
    @test_throws ArgumentError QuaST.maxwell_force_stress(txb[1:3], tyb, tzb,
                                                          eta_b, dtb, Gb)

    # ---- (c) the assembled term is -int tau : eps(v) ----------------------
    # A rotational polynomial stress: tau = (a x y, b x y, c x^2/2) has
    # div tau = (a y, b x + c x), which is *not* a pressure gradient (curl != 0),
    # so the equivalent body-force solve has a non-trivial velocity to compare
    # against (a gradient force would be absorbed by the pressure and the test
    # would be vacuous).  The mass matrix is constant eta = 1, all walls rigid.
    qta = g247_tree(2)
    ssa = make_semstokes(qta, 2)
    sqa = ssa.su
    Na = sqa.N
    xs = [c[1] for c in sqa.coords]
    ys = [c[2] for c in sqa.coords]
    aa, bb, cc = 0.7, -0.3, 0.55
    tau_poly = (aa .* xs .* ys, bb .* xs .* ys, 0.5 .* cc .* xs .^ 2)
    fx_poly(x, y) = aa * y
    fz_poly(x, y) = (bb + cc) * x
    u_body = solve_semquad_stokes_var(ssa, qta, (x, y) -> 1.0, fx_poly,
                                      fz_poly, z2, z2; bc=:rigid,
                                      deviatoric=false,
                                      linear=:direct)[1]
    u_el = solve_semquad_stokes_var(ssa, qta, (x, y) -> 1.0, z2, z2, z2, z2;
                                    bc=:rigid, deviatoric=false,
                                    linear=:direct,
                                    elastic_stress=tau_poly)[1]
    @test maximum(abs, u_body) > 1.0e-4        # the comparison is not 0 = 0
    @test rel(u_body, u_el) < 1.0e-12          # measured 9e-15
    # a uniform stress has div tau = 0: no load, so no flow (the free dofs'
    # test functions vanish on the rigid boundary, which kills the boundary term)
    u_uniform = solve_semquad_stokes_var(ssa, qta, (x, y) -> 1.0, z2, z2, z2,
                                         z2; bc=:rigid, deviatoric=false,
                                         linear=:direct,
                                         elastic_stress=(fill(0.3, Na),
                                                         fill(-0.2, Na),
                                                         fill(0.1, Na)))[1]
    @test maximum(abs, u_uniform) < 1.0e-12
    # the same load reaches the reduced matrix-free `:mg` assembly, which does
    # not form K: its `fv` must be bitwise the direct assembly's
    sys_d = solve_semquad_stokes_var(ssa, qta, (x, y) -> 1.0, fx_poly,
                                     fz_poly, z2, z2; bc=:rigid,
                                     deviatoric=false, linear=:direct,
                                     assemble_only=true)
    sys_m = solve_semquad_stokes_var(ssa, qta, (x, y) -> 1.0, fx_poly,
                                     fz_poly, z2, z2; bc=:rigid,
                                     deviatoric=false, linear=:mg,
                                     mg_operator=:matrixfree,
                                     assemble_only=:mg)
    @test sys_m.K === nothing
    @test sys_d.fv == sys_m.fv
    # a wrong-length stress field is rejected before assembly
    @test_throws ArgumentError solve_semquad_stokes_var(
        ssa, qta, (x, y) -> 1.0, z2, z2, z2, z2; bc=:rigid,
        elastic_stress=(zeros(Na + 1), zeros(Na), zeros(Na)),
        assemble_only=true)

    # ---- (d) the equilibrated limit through the driver -------------------
    # R >> eta: dt = 1000 tau_M-ish.  The stress must reach 2 eta eps, and the
    # flow must reproduce the 2a effective-viscosity solution: both schemes then
    # differ only by the O(tau_M/dt) viscosity shift, and the coupled one is the
    # material-viscosity solution (eta_eff + a eta = eta).
    runA = g247_run(1000.0; nsteps=5)
    umA, ueA, eta_matA, eta_effA = g247_reference(runA)
    sqA = runA.r.ss.su
    NA = sqA.N
    eta_nodesA = [eta_matA(c[1], c[2]) for c in sqA.coords]
    # first solve = 2a solution exactly (tau^0 = 0); second solve = one step of
    # the coupled map from the first solve's velocity
    eps0 = QuaST.semquad_strain_rate_components(sqA, runA.us[1], runA.zs[1])
    t1 = ntuple(_ -> zeros(Float64, NA), 3)
    QuaST._advance_maxwell_stress!(t1[1], t1[2], t1[3], sqA, runA.us[1],
                                   runA.zs[1], runA.dt, eta_nodesA, runA.G)
    el1 = QuaST.maxwell_force_stress(t1[1], t1[2], t1[3], eta_nodesA, runA.dt,
                                     runA.G)
    _, eta_eff0 = g247_eta_closures(runA)
    u_rec = solve_semquad_stokes_var(runA.r.ss, runA.r.qt, eta_eff0, z2, z2,
                                     z2, z2; bc=runA.pr.bc,
                                     wall_bc=runA.pr.wall_bc,
                                     wall_u=runA.pr.wall_u,
                                     deviatoric=runA.pr.deviatoric,
                                     linear=:direct, elastic_stress=el1)[1]
    @test runA.us[1] == ueA[1:NA]             # tau^0 = 0: exactly the 2a answer
    @test runA.us[2] == u_rec[1:NA]           # the step used tau^1 from u_0
    @test runA.r.txx !== nothing && runA.r.tyy !== nothing &&
          runA.r.txy !== nothing
    # tau = 2 eta eps at the equilibrated state (machine precision: the update's
    # fixed point is exactly the material stress)
    epsA = QuaST.semquad_strain_rate_components(sqA, runA.r.ux, runA.r.uz)
    tmaxA = max(maximum(abs, runA.r.txx), maximum(abs, runA.r.tyy),
                maximum(abs, runA.r.txy))
    tau_err = max(maximum(abs.(runA.r.txx .- 2 .* eta_nodesA .* epsA.exx)),
                  maximum(abs.(runA.r.tyy .- 2 .* eta_nodesA .* epsA.ezz)),
                  maximum(abs.(runA.r.txy .- 2 .* eta_nodesA .* epsA.exz)))
    @test tau_err / tmaxA < 1.0e-12           # measured ~1e-15 after 5 steps
    # the stress is also 2 eta_eff eps to the O(tau_M/dt) the 2a closure implies
    eta_eff_nodesA = [eta_effA(c[1], c[2]) for c in sqA.coords]
    tau_eff_err = max(maximum(abs.(runA.r.txx .-
                                   2 .* eta_eff_nodesA .* epsA.exx)),
                      maximum(abs.(runA.r.tyy .-
                                   2 .* eta_eff_nodesA .* epsA.ezz)),
                      maximum(abs.(runA.r.txy .-
                                   2 .* eta_eff_nodesA .* epsA.exz)))
    @test tau_eff_err / tmaxA < 5.0e-3        # tau_M/dt <= 1.5e-3
    # the velocity: the fixed point of the coupled iteration is the material
    # solution (measured 6e-6), and at dt >> tau_M that is the 2a solution to
    # the predicted O(tau_M/dt) viscosity shift (measured 3e-5)
    uA = runA.r.ux
    @test rel(uA, umA[1:NA]) < 1.0e-4
    @test rel(uA, ueA[1:NA]) < 5.0e-5
    @test rel(uA, ueA[1:NA]) > 1.0e-6         # non-vacuous: no coupling = 0
    @test rel(umA[1:NA], ueA[1:NA]) > 1.0e-6  # the two references differ
    # the coupled answer departs from 2a by exactly the 2a answer's own departure
    # from the material solution (ratio 1.04 measured)
    @test 0.7 < rel(uA, ueA[1:NA]) / rel(umA[1:NA], ueA[1:NA]) < 1.3
    udefectA = g247_defect(runA, umA, eta_nodesA)
    @test rel(uA, umA[1:NA]) <= 2.0 * rel(udefectA, umA)

    # dt = 10 tau_M-ish: the same fixed point must still be the material
    # solution, but the O(tau_M/dt) 2a viscosity shift is now large enough to
    # see, so an absent or unscaled elastic force (which would give the 2a
    # answer) fails the "differs from 2a" assertion.
    runB = g247_run(10.0; nsteps=8)
    umB, ueB, eta_matB, _ = g247_reference(runB)
    sqB = runB.r.ss.su
    NB = sqB.N
    eta_nodesB = [eta_matB(c[1], c[2]) for c in sqB.coords]
    uB = runB.r.ux
    udefectB = g247_defect(runB, umB, eta_nodesB)
    @test rel(uB, umB[1:NB]) <= 2.0 * rel(udefectB, umB)
    @test rel(uB, ueB[1:NB]) > 1.0e-3         # measured 2.6e-3, 10x the 2a shift
    epsB = QuaST.semquad_strain_rate_components(sqB, runB.r.ux, runB.r.uz)
    tmaxB = max(maximum(abs, runB.r.txx), maximum(abs, runB.r.tyy),
                maximum(abs, runB.r.txy))
    tau_errB = max(maximum(abs.(runB.r.txx .- 2 .* eta_nodesB .* epsB.exx)),
                   maximum(abs.(runB.r.tyy .- 2 .* eta_nodesB .* epsB.ezz)),
                   maximum(abs.(runB.r.txy .- 2 .* eta_nodesB .* epsB.exz)))
    @test tau_errB / tmaxB < 1.0e-6           # measured 5e-9
    # the 2a departure scales like tau_M/dt: R = 1000 is 100x smaller than R = 10
    @test rel(uA, ueA[1:NA]) < rel(uB, ueB[1:NB]) / 30.0
    # and the iteration contracts at the backward-Euler rate (ratio ~1/a)
    ch = [rel(runA.us[k], runA.us[k - 1]) for k in 2:length(runA.us)]
    @test ch[3] < 0.1 * ch[2]
    @test ch[end] < 1.0e-6 * ch[1]
    @printf("G247: R=1000 rel(run,mat)=%.2e rel(run,2a)=%.2e |tau-2eta eps|=%.1e ; R=10 rel(run,mat)=%.2e rel(run,2a)=%.2e\n",
            rel(uA, umA[1:NA]), rel(uA, ueA[1:NA]), tau_err / tmaxA,
            rel(uB, umB[1:NB]), rel(uB, ueB[1:NB]))
end
