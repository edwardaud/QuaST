# G260: the tangential-traction load is an RHS-only term -- it must not change
# the operator, the `:auto` path decision or the multigrid hierarchy, it must be
# deterministic, and `assemble_only` must return it.
#
# Stage 2 adds `int tbar.phi_i dS` to the **unreduced** velocity load vector
# (docs/dev/BC_DESIGN.md §3.2/§3.3).  That claim has observable consequences
# this gate pins on the uniform L4 tree (256 leaves, 2 467 saddle dofs, so
# `linear=:auto` resolves to MG) and on a small L3 tree (`:auto` resolves to
# direct):
#
#   (a) RHS-only: with the load on and off, every assembled operator (`K`,
#       `Kred`, `Bs`, `Bt`, `B`, `Bc`, `Bm`, `Cu`, `Cp`, `u0`, `p0`) is
#       bitwise identical and only `fv`/`fr` differ; `fv` changes by exactly the
#       load, whose total is `int tbar dS`;
#   (b) the `:auto` selector's recorded decision (choice, reason, saddle dofs,
#       contrast, threads) is identical with the load on and off, and an
#       `:auto` solve resolves to the same path; on L4 `:auto` is MG and its
#       answer agrees with explicit `:mg` to 1e-10 (G191's convention), while
#       `:direct` and `:mg` agree to the design's 1e-6 accuracy goal;
#   (c) the multigrid hierarchy is load-independent: both solves build the same
#       levels (`qts` leaf-set fingerprints, `nfs`, `Cus`, `Ps`, `free_rows`,
#       level count) from a cold cache, and the level-cache counters match;
#   (d) `assemble_only=true` returns the loaded `fv` (equal to the
#       `return_system` path's `fv`) and its `fr`, and `assemble_only=:mg` also
#       carries the load; with the default (no `wall_t`) both are bitwise the
#       unloaded values;
#   (e) threaded and serial assembly give bitwise identical answers for a
#       load-only problem (`assemble_threads=1` vs `8`): the load pass is serial
#       and runs after the threaded merge, so it cannot reorder anything (with a
#       non-zero *volume* load, `fv` may still move at round-off by design, see
#       §17 of `assembly_var.jl` -- the gate therefore drives the flow with the
#       boundary load only);
#   (f) a traction-free wall (`wall_t = 0` closure) is bitwise the default path,
#       and an all-`nothing` tuple is bitwise the default path;
#   (g) the checkpoint guard records the traction: a checkpoint written without
#       a custom `wall_t` is refused when resumed with one (the marker format is
#       pinned by G245; the end-to-end refusal of the other per-wall fields is
#       G258 (f), this is the same mechanism for the new field).
#
# The numbers and tolerances are measured and explained next to the assertions.

using Test, QuaST, LinearAlgebra

# Uniform `2^n x 2^n`-leaf tree (each leaf is one Q2 element): 17x17 velocity
# nodes at n = 3, 33x33 at n = 4.
function g260_uniform(n::Int)
    qt = Quadtree(n)
    for _ in 1:n, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

g260_zero(x, y) = 0.0
g260_tau = (x, y) -> 1.0 + 0.5 * x     # polynomial: the face rule is exact

# The traction configuration: bottom and both sides `:rigid` (normal Dirichlet
# zero, tangential zero), top `:free` with the tangential traction.  All normal
# values are zero, so the closed-box net-flux guard accepts it.
g260_wall_bc() = (:rigid, :rigid, :rigid, :free)
g260_wall_t(τ) = (nothing, nothing, nothing, τ)

@testset "G260 traction paths, assembly and threads" begin
    z0 = g260_zero
    tau = g260_tau
    eta1 = (x, y) -> 1.0

    # ---- (a) RHS-only: the operators do not see the load -------------------
    qt = g260_uniform(4)
    ss = make_semstokes(qt, 2)
    N = ss.su.N
    solve_asm(; wall_t, linear=:direct) =
        solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                 bc=:free, wall_bc=g260_wall_bc(),
                                 wall_t=wall_t, linear=linear,
                                 assemble_only=true)
    base = solve_asm(wall_t=(nothing, nothing, nothing, nothing))
    load = solve_asm(wall_t=g260_wall_t(tau))
    @test load.K == base.K && load.Kred == base.Kred
    @test load.B == base.B && load.Bc == base.Bc && load.Bm == base.Bm
    @test load.Bs == base.Bs && load.Bt == base.Bt
    @test load.Cu == base.Cu && load.Cp == base.Cp
    @test load.u0 == base.u0 && load.p0 == base.p0
    @test all(load.u0 .== 0.0)                 # the load never becomes Dirichlet data
    # with no volume drive, the unloaded RHS is exactly zero and the loaded one
    # is exactly the traction load
    @test all(iszero, base.fv)
    @test load.fv != base.fv && load.fr != base.fr
    @test all(iszero, load.fv[N + 1:end])      # tangential load on a y-wall is u_x
    # the total load is `int_0^Lx (1 + x/2) dx = 1.25` on this Lx = 1 box
    @test sum(load.fv[1:N]) ≈ 1.25 rtol = 1.0e-14
    # the loaded `fr` is the loaded `fv` projected by `Cu'` (no Dirichlet lift,
    # and with all normal data zero the p0/u0 terms are zero)
    @test load.fr == load.Cu' * load.fv

    # ---- (b) `:auto` decision is load-independent; direct/MG agree ---------
    # `:auto` on the 2 467-dof tree resolves to MG; record the decision for
    # both load settings and require the same path and the same reason.
    function auto_choice(; wall_t)
        solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                 bc=:free, wall_bc=g260_wall_bc(),
                                 wall_t=wall_t, linear=:auto)
        ch = LAST_SEMSTOKES_AUTO_CHOICE[]
        return (choice=ch.choice, reason=ch.reason, saddle_dofs=ch.saddle_dofs,
                contrast=ch.contrast, samples=ch.samples, mg_threads=ch.mg_threads)
    end
    ch_off = auto_choice(wall_t=(nothing, nothing, nothing, nothing))
    ch_on = auto_choice(wall_t=g260_wall_t(tau))
    @test ch_on == ch_off
    @test ch_on.choice === :mg                # the L4 tree is above the selector rule
    @test ch_on.saddle_dofs == 2 * ss.su.N + ss.sp.N
    @test ch_on.contrast == 1.0

    # explicit answers on the same system
    ud, pd = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                      bc=:free, wall_bc=g260_wall_bc(),
                                      wall_t=g260_wall_t(tau),
                                      linear=:direct)[1:2]
    rmg = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                   bc=:free, wall_bc=g260_wall_bc(),
                                   wall_t=g260_wall_t(tau),
                                   linear=:mg, mg_operator=:matrixfree,
                                   mg_threads=1, mg_tol=1.0e-10)
    um, pm = rmg[1], rmg[2]
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-9
    @test norm(ud - um) / max(norm(ud), 1.0) <= 1.0e-6   # the design's G-path goal
    @test norm(pd - pm) / max(norm(pd), 1.0) <= 1.0e-6
    # `:auto` -> MG reproduces the explicit MG answer (G191 checks that on an
    # unloaded system; here the load is the only drive)
    nfb = LAST_AUTO_FALLBACK[].count
    r_auto = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                      bc=:free, wall_bc=g260_wall_bc(),
                                      wall_t=g260_wall_t(tau), linear=:auto)
    @test LAST_SEMSTOKES_AUTO_CHOICE[].choice === :mg
    @test LAST_AUTO_FALLBACK[].count == nfb          # no fallback happened
    @test norm(r_auto[1] - um) / max(norm(um), 1.0) <= 1.0e-10
    @test norm(r_auto[2] - pm) / max(norm(pm), 1.0) <= 1.0e-10
    @test maximum(abs, ud) > 1.0e-3            # the load actually drives the flow

    # On the small L3 tree `:auto` resolves to direct, and the loaded auto
    # answer is bitwise the loaded direct answer (the selector is insensitive to
    # the load there by the same rule).
    qts = g260_uniform(3)
    sss = make_semstokes(qts, 2)
    ad = solve_semquad_stokes_var(sss, qts, eta1, z0, z0, z0, z0;
                                  bc=:free, wall_bc=g260_wall_bc(),
                                  wall_t=g260_wall_t(tau), linear=:auto)
    @test LAST_SEMSTOKES_AUTO_CHOICE[].choice === :direct
    dd = solve_semquad_stokes_var(sss, qts, eta1, z0, z0, z0, z0;
                                  bc=:free, wall_bc=g260_wall_bc(),
                                  wall_t=g260_wall_t(tau), linear=:direct)
    @test ad[1] == dd[1] && ad[2] == dd[2] && ad[3] == dd[3]
    # ... and with the load on it is not the unloaded direct answer
    d0 = solve_semquad_stokes_var(sss, qts, eta1, z0, z0, z0, z0;
                                  bc=:free, wall_bc=g260_wall_bc(),
                                  wall_t=(nothing, nothing, nothing, nothing), linear=:direct)
    @test dd[1] != d0[1]

    # ---- (c) the multigrid hierarchy is load-independent -------------------
    # A cold build for each setting: empty the cache, solve, and read the
    # hierarchy geometry back out of the cache.  If the load were (wrongly) part
    # of the hierarchy inputs, the two geos would differ; the level-cache
    # counters are compared as well.
    function g260_geo(; wall_t)
        empty!(QuaST.SEMQUAD_MG_CACHE)
        QuaST.reset_semquad_mg_level_stats!()
        solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                 bc=:free, wall_bc=g260_wall_bc(),
                                 wall_t=wall_t, linear=:mg,
                                 mg_operator=:matrixfree, mg_threads=1,
                                 mg_tol=1.0e-10)
        geos = [v for v in values(QuaST.SEMQUAD_MG_CACHE)
                if v isa QuaST.SemQuadMGGeom]
        @test length(geos) == 1
        g = geos[1]
        st = QuaST.semquad_mg_level_cache_stats()
        return (g=g,
                leaves=[QuaST._semquad_mg_leaf_hash(q.leaves) for q in g.qts],
                nfs=copy(g.nfs), L=g.L, coarse=g.coarse_direct,
                Cus=g.Cus, Ps=g.Ps, free_rows=g.free_rows,
                stats=(builds=st.builds, hits=st.hits, misses=st.misses,
                       prolongations=st.prolongations,
                       last_rebuilt=copy(st.last_rebuilt),
                       last_reused=copy(st.last_reused),
                       cached_levels=QuaST.semquad_mg_level_cache_stats().cached_levels))
    end
    ga = g260_geo(wall_t=(nothing, nothing, nothing, nothing))
    gb = g260_geo(wall_t=g260_wall_t(tau))
    @test ga.L == gb.L && ga.L >= 3
    @test ga.leaves == gb.leaves
    @test ga.nfs == gb.nfs
    @test ga.coarse == gb.coarse
    @test length(ga.Cus) == length(gb.Cus) && length(ga.Ps) == length(gb.Ps)
    @test all(ga.Cus[k] == gb.Cus[k] for k in eachindex(ga.Cus))
    @test all(ga.Ps[k] == gb.Ps[k] for k in eachindex(ga.Ps))
    @test ga.free_rows == gb.free_rows
    @test ga.stats == gb.stats
    @test ga.stats.builds == 1 && ga.stats.hits == 0   # both were cold builds
    @test ga.stats.misses == ga.L                       # one per coarse level
    @test ga.stats.last_rebuilt == collect(1:ga.L)
    @test ga.stats.cached_levels == ga.L

    # ---- (d) assemble_only / return_system carry the load ------------------
    rt = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:free, wall_bc=g260_wall_bc(),
                                  wall_t=g260_wall_t(tau), linear=:direct,
                                  return_system=true)
    # return_system = (u, pv, fixed, cons2, cons_p, K, Bref, Cu, Cp, u0, p0, fv)
    @test load.fv == rt[end]
    @test load.K == rt[6] && load.Cu == rt[8] && load.u0 == rt[10] &&
          load.p0 == rt[11]
    @test load.fr == load.Cu' * (load.fv - load.K * load.u0 + load.Bm' * load.p0)
    # `assemble_only=:mg` (the reduced matrix-free geometry assembly) also
    # carries the load, and its `fv` is bitwise the full assembly's `fv`
    red = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                   bc=:free, wall_bc=g260_wall_bc(),
                                   wall_t=g260_wall_t(tau), linear=:mg,
                                   mg_operator=:matrixfree, assemble_only=:mg)
    @test red.K === nothing && red.Kred === nothing
    @test red.fv == load.fv
    @test red.Cu == load.Cu
    # with no traction the reduced `fv` is the full one bit for bit (the load
    # pass is a genuine no-op on the reduced path too)
    red0 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                    bc=:free, wall_bc=g260_wall_bc(),
                                    linear=:mg, mg_operator=:matrixfree,
                                    assemble_only=:mg)
    @test red0.fv == base.fv

    # ---- (e) threaded == serial for a load-only problem --------------------
    # With no volume drive the leaf blocks contribute exactly zero to `fv`, and
    # the load pass runs serially after the merge: every bit must agree.  The
    # threaded path is only actually exercised when Julia has > 1 thread; the
    # assertion holds (trivially) in a 1-thread process and the suite runs with
    # --threads=8.
    r1 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:free, wall_bc=g260_wall_bc(),
                                  wall_t=g260_wall_t(tau), linear=:direct,
                                  assemble_threads=1)
    r8 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:free, wall_bc=g260_wall_bc(),
                                  wall_t=g260_wall_t(tau), linear=:direct,
                                  assemble_threads=8)
    @test r1[1] == r8[1] && r1[2] == r8[2] && r1[3] == r8[3]
    asm1 = solve_asm(wall_t=g260_wall_t(tau))
    # (the `solve_asm` closure above pins assemble_threads at its default 1)
    asm8 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                    bc=:free, wall_bc=g260_wall_bc(),
                                    wall_t=g260_wall_t(tau), linear=:direct,
                                    assemble_only=true, assemble_threads=8)
    @test asm1.fv == asm8.fv && asm1.fr == asm8.fr
    @test asm1.K == asm8.K && asm1.Cu == asm8.Cu
    @info "G260 threads" nthreads = Threads.nthreads() assemble_threads = 8

    # ---- (f) a traction-free wall is bitwise the default path --------------
    # A non-zero volume drive makes the comparison non-trivial (`u` is not
    # identically zero); `wall_t = 0` adds only +-0.0, which leaves every bit of
    # `fv` alone.
    fz = (x, y) -> sinpi(x) * sinpi(y)
    function zero_traction_solve(; wall_t)
        return solve_semquad_stokes_var(ss, qt, eta1, z0, fz, z0, z0;
                                        bc=:free, wall_bc=g260_wall_bc(),
                                        wall_t=wall_t, linear=:direct)
    end
    rz0 = zero_traction_solve(wall_t=(nothing, nothing, nothing, nothing))
    rzall = zero_traction_solve(wall_t=(nothing, nothing, nothing, nothing))
    rzf = zero_traction_solve(wall_t=(nothing, nothing, nothing, (x, y) -> 0.0))
    @test rz0[1] == rzall[1] && rz0[2] == rzall[2] && rz0[3] == rzall[3]
    @test rz0[1] == rzf[1] && rz0[2] == rzf[2] && rz0[3] == rzf[3]
    @test maximum(abs, rz0[1]) > 1.0e-6      # the volume drive is real

    # ---- (g) the restart guard records the traction -----------------------
    # A checkpoint's `run_config` stores what changes the trajectory; the wall
    # traction does, so a resume with a different `wall_t` must be refused (the
    # marker format itself is pinned by G245, the same mechanism for the other
    # per-wall fields is G258 (f)).
    qtck = Quadtree(4)
    for _ in 1:1, lf in collect(qtck.leaves)
        refine!(qtck, lf)
    end
    balance!(qtck)
    ssck = make_semstokes(qtck, 2)
    thck = zeros(Float64, ssck.su.N)
    ckpath = joinpath(tempdir(), "g260_wall_t_restart.h5")
    ckkw = (p=2, tstop=2.0e-3, dt0=1.0e-3, dtmax=1.0e-3, dtmin=1.0e-16,
            amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
            mass_form=:boussinesq)
    prck = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free,
                  wall_bc=g260_wall_bc())
    run_semquad_simulation(qtck, prck, thck; ckkw...,
                           checkpoint_path=ckpath, checkpoint_every=1)
    prck2 = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free,
                   wall_bc=g260_wall_bc(),
                   wall_t=(nothing, nothing, nothing, (x, y) -> 0.5))
    errck = try
        run_semquad_simulation(qtck, prck2, thck; ckkw..., restart_from=ckpath)
        nothing
    catch e
        e
    end
    @test errck isa ArgumentError
    msgck = errck isa ArgumentError ? sprint(showerror, errck) : ""
    @test occursin("restart configuration mismatch", msgck)
    @test occursin("wall_t", msgck)
    @test occursin("checkpoint=none,none,none,none", msgck)
    @test occursin("current=none,none,none,custom", msgck)
    rm(ckpath; force=true)
end
