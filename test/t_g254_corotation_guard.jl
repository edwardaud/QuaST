# G254: the validity band of the linearised co-rotational (Jaumann) term is
# enforced at run time.
#
# `maxwell_stress_corotate` applies the rotation *linearised* over the step,
# `1 + dt ad_W`, not `exp(dt ad_W)`.  The documented band is `2|W|dt << 1` in
# the elastic regime (`dt < tau_M = eta/G`) and `2 tau_M |W|` not large in the
# fast-relaxation regime (`tau_M < dt`); both are the single per-step rotation
#
#     resolved_rotation = 2 |W| min(dt, tau_M),
#
# which `MAXWELL_COROTATION_BOUND` quantifies as 0.1.  Nothing used to check it:
# a card at `W dt ~ 1` was silently wrong.  This gate pins:
#
#   (a) the three measures `maxwell_corotation_band` returns, their exact
#       values in both regimes (the resolved rotation is the *shorter* of the
#       two timescales, so `2|W|dt` alone would be a false alarm in the
#       fast-relaxation regime and `2 tau_M |W|` alone one in the elastic
#       regime), the sign/magnitude convention and `wxy = 0`;
#   (b) `_advance_maxwell_stress!` measures the driving velocity on the domain:
#       for a rigid rotation the three maxima are the analytic ones; on a
#       non-uniform field they are exactly the maximum of the same accessors;
#       the measurement does not change a single stress component (the state is
#       compared bitwise with the explicit composition of the same kernels);
#       and `G = 0` returns `nothing` without sampling anything (so the guard
#       costs nothing when `shear_modulus = 0`);
#   (c) through the driver: an in-band run emits exactly one `Info` report with
#       the run maxima recomputed here from its callback states, and no warning;
#   (d) the guard fires on a card outside the band: exactly one `Warning` at
#       the first offending step, carrying that step's measured maxima, the
#       bound, and what it means; the run still completes (warn, not refuse --
#       the scheme remains consistent outside the band), and the final `Info`
#       report says OUTSIDE;
#   (e) `corotation_guard = false` silences both logs and leaves every returned
#       field and the history bit-identical, so the guard changes no number;
#       `shear_modulus = 0` logs nothing at all.
#
# The bound is pinned by value (0.1) because it is a stated accuracy claim, not
# an implementation detail: changing it should be a deliberate edit with a
# reason, exactly like the tolerances this suite quotes.
#
# The helper functions are defined at file scope rather than inside the testset:
# a function defined inside a `@testset` is a closure, and an assignment to a
# name that already exists in the testset's local scope would *modify* that
# outer variable (Julia nested-function scoping) -- the shape of the accumulator
# in `g254_measures` must not depend on where it is defined.
using Test, QuaST, Logging, Printf

function g254_tree(nref::Int, base::Int=4)
    qt = Quadtree(base)
    for _ in 1:nref, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# The same maxima accumulations the driver performs, from the callback states of
# the steps that advanced the stress: maxima over the domain per step and over
# the run.
function g254_measures(caps, sq, gg)
    runmax = (two_w_dt=0.0, two_taum_w=0.0, resolved=0.0)
    per = NamedTuple[]
    for cap in caps
        spin = QuaST.semquad_spin_component(sq, cap.ux, cap.uz)
        stepmax = (two_w_dt=0.0, two_taum_w=0.0, resolved=0.0)
        for g in eachindex(spin)
            band = QuaST.maxwell_corotation_band(spin[g], cap.dt, 1.0 / gg)
            stepmax = (two_w_dt=max(stepmax.two_w_dt, band.two_w_dt),
                       two_taum_w=max(stepmax.two_taum_w, band.two_taum_w),
                       resolved=max(stepmax.resolved, band.resolved_rotation))
        end
        push!(per, stepmax)
        runmax = (two_w_dt=max(runmax.two_w_dt, stepmax.two_w_dt),
                  two_taum_w=max(runmax.two_taum_w, stepmax.two_taum_w),
                  resolved=max(runmax.resolved, stepmax.resolved))
    end
    return runmax, per
end

# Lid-driven cavity with a temperature-independent viscosity (`expT` with
# `deta_T = 1` is exactly eta = 1), so the material Maxwell time is exactly
# 1/G and the callback states reconstruct the update exactly.
function g254_card_run(dt; nsteps::Int, G::Float64=10.0,
                       corotation_guard::Bool=true)
    qt = g254_tree(1)
    ss = make_semstokes(qt, 2)
    N = ss.su.N
    pr = Params(; Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0,
                shear_modulus=G, bc=:free,
                wall_bc=(:free, :free, :free, :rigid),
                wall_u=(nothing, nothing, nothing, (x, y) -> 1.0))
    caps = NamedTuple[]
    cb = s -> begin
        push!(caps, (step=s.step, dt=s.dt, ux=copy(s.ux), uz=copy(s.uz)))
        nothing
    end
    logs, res = Test.collect_test_logs(min_level=Logging.Info) do
        run_semquad_simulation(qt, pr, zeros(Float64, N); p=2,
                               maxsteps=nsteps, tstop=1.0e9, dt0=dt, dtmax=dt,
                               dtmin=1.0e-30, cfl=0.0, linear=:direct, ev=0.0,
                               maxpic=1, callback=cb,
                               corotation_guard=corotation_guard)
    end
    return (logs=logs, r=res, caps=caps, sq=ss.su, G=G)
end

g254_is_info(l) = l.level == Logging.Info &&
                  occursin("Maxwell co-rotation validity (G254):", l.message)
g254_is_warn(l) = l.level == Logging.Warn &&
                  occursin("Maxwell co-rotation validity band exceeded",
                           l.message)

@testset "G254 co-rotation validity guard" begin
    # ---- (a) the bound and the three measures, by hand -------------------
    @test MAXWELL_COROTATION_BOUND isa Float64
    @test MAXWELL_COROTATION_BOUND == 0.1
    # elastic regime: dt < tau_M -> the step, not the Maxwell time, limits it
    m_el = QuaST.maxwell_corotation_band(2.0, 0.01, 0.5)
    @test m_el.two_w_dt == 0.04           # 2*2.0*0.01
    @test m_el.two_taum_w == 2.0          # 2*0.5*2.0
    @test m_el.resolved_rotation == 0.04
    # fast-relaxation regime: tau_M < dt -> the Maxwell time limits it, and the
    # raw 2|W|dt (40 here) is *not* the quantity to bound
    m_fast = QuaST.maxwell_corotation_band(2.0, 10.0, 0.01)
    @test m_fast.two_w_dt == 40.0
    @test m_fast.two_taum_w == 0.04
    @test m_fast.resolved_rotation == 0.04
    @test m_fast.resolved_rotation == min(m_fast.two_w_dt, m_fast.two_taum_w)
    @test m_el.resolved_rotation == min(m_el.two_w_dt, m_el.two_taum_w)
    # magnitude convention, and zero spin is three zeros
    @test QuaST.maxwell_corotation_band(-2.0, 0.01, 0.5) == m_el
    z = QuaST.maxwell_corotation_band(0.0, 1.0e-3, 10.0)
    @test z.two_w_dt == 0.0 && z.two_taum_w == 0.0 && z.resolved_rotation == 0.0

    # ---- (b) the kernel measures the driving velocity on the domain ------
    qtb = g254_tree(2, 2)
    ssb = make_semstokes(qtb, 2)
    sqb = ssb.su
    @test length(qtb.leaves) == 16          # not one leaf (HANDOVER section 8)
    Omega = 1.0
    G0 = 40.0
    eta0 = 2.0
    dtb = 1.0e-2
    uxr = [-Omega * y for (_, y) in sqb.coords]   # rigid rotation: W_xy = -Omega
    uzr = [Omega * x for (x, _) in sqb.coords]
    wr = QuaST.semquad_spin_component(sqb, uxr, uzr)
    @test maximum(abs.(wr .+ Omega)) <= 1.0e-14
    tx = zeros(sqb.N); ty = zeros(sqb.N); tz = fill(0.7, sqb.N)
    ret = QuaST._advance_maxwell_stress!(tx, ty, tz, sqb, uxr, uzr, dtb,
                                         fill(eta0, sqb.N), G0)
    tauM = eta0 / G0
    @test ret !== nothing
    @test isapprox(ret.two_w_dt, 2.0 * Omega * dtb; rtol=1.0e-12)
    @test isapprox(ret.two_taum_w, 2.0 * tauM * Omega; rtol=1.0e-12)
    @test isapprox(ret.resolved_rotation, 2.0 * Omega * min(dtb, tauM);
                   rtol=1.0e-12)
    @test ret.resolved_rotation < MAXWELL_COROTATION_BOUND   # this card is fine
    # `G = 0` is the disabled convention: no state, no measurement, no cost
    txg = fill(3.0, sqb.N); tyg = fill(-1.0, sqb.N); tzg = fill(0.5, sqb.N)
    retg = QuaST._advance_maxwell_stress!(txg, tyg, tzg, sqb, uxr, uzr, dtb,
                                          fill(eta0, sqb.N), 0.0)
    @test retg === nothing
    @test txg == fill(3.0, sqb.N) && tyg == fill(-1.0, sqb.N) &&
          tzg == fill(0.5, sqb.N)

    # A non-uniform velocity field on the same tree: the maxima are the domain
    # maxima of the same accessors (not a node-1 sample), and one call of the
    # update is bitwise the explicit composition of the same kernels -- the
    # measurement reads, it does not change the state.
    uxu = [0.30 * x - 0.10 * y + 0.40 * x * y + 0.02 for (x, y) in sqb.coords]
    uzu = [0.50 * x - 0.12 * y - 0.30 * x^2 + 0.07 * y^2 - 0.03
           for (x, y) in sqb.coords]
    dtu = 1.0e-3
    etau = fill(2.0, sqb.N)
    Gu = 50.0
    txa = fill(0.3, sqb.N); tya = fill(-0.2, sqb.N); tza = fill(0.1, sqb.N)
    retu = QuaST._advance_maxwell_stress!(txa, tya, tza, sqb, uxu, uzu, dtu,
                                          etau, Gu)
    epsu = QuaST.semquad_strain_rate_components(sqb, uxu, uzu)
    wu = QuaST.semquad_spin_component(sqb, uxu, uzu)
    rxa = fill(0.3, sqb.N); rya = fill(-0.2, sqb.N); rza = fill(0.1, sqb.N)
    @inbounds for g in eachindex(rxa)
        tr = epsu.exx[g] + epsu.ezz[g]
        cx, cy, cz = maxwell_stress_corotate(rxa[g], rya[g], rza[g], wu[g], dtu)
        rxa[g] = maxwell_stress_step(cx, epsu.exx[g] - tr / 3.0, dtu, 2.0, Gu)
        rya[g] = maxwell_stress_step(cy, epsu.ezz[g] - tr / 3.0, dtu, 2.0, Gu)
        rza[g] = maxwell_stress_step(cz, epsu.exz[g], dtu, 2.0, Gu)
    end
    @test txa == rxa && tya == rya && tza == rza      # bitwise: a pure read
    # the returned maxima are exactly the maxima over the domain of the same
    # three per-node measures (same accessor, same arithmetic order)
    twodt = 0.0; twotaum = 0.0; resolved = 0.0
    for g in eachindex(wu)
        b = QuaST.maxwell_corotation_band(wu[g], dtu, etau[g] / Gu)
        twodt = max(twodt, b.two_w_dt)
        twotaum = max(twotaum, b.two_taum_w)
        resolved = max(resolved, b.resolved_rotation)
    end
    @test retu.two_w_dt == twodt
    @test retu.two_taum_w == twotaum
    @test retu.resolved_rotation == resolved
    gmax = argmax(abs.(wu))
    @test retu.two_w_dt == 2.0 * abs(wu[gmax]) * dtu
    @test retu.two_w_dt > 0.0                          # not the 0 == 0 case
    @test retu.resolved_rotation > 0.0

    # ---- (c) the driver reports the run maxima once, inside the band -----
    inb = g254_card_run(1.0e-3; nsteps=2)
    acc, per = g254_measures(inb.caps, inb.sq, inb.G)
    @test acc.resolved < MAXWELL_COROTATION_BOUND
    @test acc.two_w_dt > 0.0 && acc.two_taum_w > 0.0
    # elastic regime: dt < tau_M everywhere, so the report's resolved rotation
    # is the same quantity as max 2|W|dt
    @test acc.resolved == acc.two_w_dt
    infos = [l for l in inb.logs if g254_is_info(l)]
    @test length(infos) == 1                     # once per run, not per step
    itxt = isempty(infos) ? "" : infos[1].message
    @test occursin("over $(length(inb.caps)) stress step(s)", itxt)
    @test occursin(@sprintf("max 2|W|dt = %.6e", acc.two_w_dt), itxt)
    @test occursin(@sprintf("max 2*tau_M*|W| = %.6e", acc.two_taum_w), itxt)
    @test occursin(@sprintf("max per-step rotation 2|W|*min(dt, tau_M) = %.6e",
                            acc.resolved), itxt)
    @test occursin("bound = 0.1", itxt)
    @test occursin("inside the band", itxt)
    @test isempty([l for l in inb.logs if g254_is_warn(l)])
    @test inb.r.steps == 2 && inb.r.txx !== nothing
    @printf("G254 (c) in-band report:\n%s\n", itxt)

    # ---- (d) a card outside the band: one warning, with the measured value --
    out = g254_card_run(0.3; nsteps=2)          # tau_M = 0.1 < dt: fast regime
    acc2, per2 = g254_measures(out.caps, out.sq, out.G)
    @test acc2.resolved > MAXWELL_COROTATION_BOUND   # the card really is outside
    warns = [l for l in out.logs if g254_is_warn(l)]
    @test length(warns) == 1                     # once per run: first offender
    wtxt = isempty(warns) ? "" : warns[1].message
    first_bad = findfirst(p -> p.resolved > MAXWELL_COROTATION_BOUND, per2)
    @test first_bad !== nothing
    # the warning carries that step's *measured* maxima, the bound, and what it
    # means; the step number is the checkpoint/history step that finished
    @test occursin("exceeded at step $(out.caps[first_bad].step)", wtxt)
    @test occursin(@sprintf("2|W|dt = %.6e", per2[first_bad].two_w_dt), wtxt)
    @test occursin(@sprintf("2*tau_M*|W| = %.6e", per2[first_bad].two_taum_w),
                   wtxt)
    @test occursin(@sprintf("per-step rotation = %.6e",
                            per2[first_bad].resolved), wtxt)
    @test occursin("bound = 0.1", wtxt)
    @test occursin("linearised Jaumann rotation", wtxt)
    @test occursin("Above the bound", wtxt)
    @test occursin("The run continues", wtxt)
    @test occursin("reduce dt", wtxt)
    # the run completed and returned its state: warn, not refuse
    @test out.r.steps == 2 && out.r.txx !== nothing && all(isfinite, out.r.txx)
    infos2 = [l for l in out.logs if g254_is_info(l)]
    @test length(infos2) == 1
    itxt2 = isempty(infos2) ? "" : infos2[1].message
    @test occursin("OUTSIDE the band", itxt2)
    @test occursin(@sprintf("max per-step rotation 2|W|*min(dt, tau_M) = %.6e",
                            acc2.resolved), itxt2)
    @printf("G254 (d) over-band warning:\n%s\n", wtxt)

    # ---- (e) the guard changes no number, and G = 0 pays nothing ---------
    off = g254_card_run(0.3; nsteps=2, corotation_guard=false)
    @test isempty([l for l in off.logs if g254_is_info(l) || g254_is_warn(l)])
    @test off.r.theta == out.r.theta && off.r.ux == out.r.ux &&
          off.r.uz == out.r.uz && off.r.pv == out.r.pv
    @test off.r.t == out.r.t && off.r.steps == out.r.steps
    @test off.r.history == out.r.history
    @test off.r.txx == out.r.txx && off.r.tyy == out.r.tyy &&
          off.r.txy == out.r.txy
    @test off.r.qt.leaves == out.r.qt.leaves
    zero_g = g254_card_run(0.3; nsteps=2, G=0.0)
    @test zero_g.r.txx === nothing && zero_g.r.tyy === nothing &&
          zero_g.r.txy === nothing
    @test isempty([l for l in zero_g.logs
                   if g254_is_info(l) || g254_is_warn(l)])
    @printf("G254: bound %.3g; in-band max resolved %.3e; over-band max resolved %.3e; guard off bit-identical; G=0 silent\n",
            MAXWELL_COROTATION_BOUND, acc.resolved, acc2.resolved)
end
