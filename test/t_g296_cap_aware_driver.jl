# G296 driver plumbing and default identity for the opt-in cap-aware fallback.
#
# This gate does not try to force a fallback event through the driver (the
# annulus acceptance run does that); it pins the API contract:
#   (a) omitted and explicit `amr_cap_aware=false` are bitwise identical, and
#       the checkpoint config records the key as false;
#   (b) `amr_cap_aware=true` with a non-binding cap (max_level above the tree's
#       own maximum) is bitwise identical to the default and records true;
#   (c) `amr_cap_aware=true` with `amr_coarsen_gate=false` is refused before
#       setup (the legacy gate-off branch has no fallback mark protection).
using Test
using QuaST

@testset "G296 cap-aware driver plumbing" begin
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                eta_b=9.70406052783923433184,
                eta_c=4.1588830833596718565,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)

    function setup()
        qt = Quadtree(5; cells_x=5, cells_y=2, Lx=2.5)
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        th = [0.1 * sinpi(y) * cospi(x / qt.Lx) for (x, y) in ss.su.coords]
        for (g, (_, y)) in enumerate(ss.su.coords)
            (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[g] = 0.0)
        end
        return qt, th
    end

    function run_one(; cap_aware=:omitted, max_level=typemax(Int),
                     coarsen_gate=true)
        empty!(QuaST.SEMQUAD_MG_CACHE)
        qt, th = setup()
        ck = tempname() * ".h5"
        logp = tempname() * ".log"
        res = if cap_aware === :omitted
            open(logp, "w") do io
                redirect_stdout(io) do
                    run_semquad_simulation(qt, pr, th;
                        p=2, tstop=6.0e-4, dt0=1.0e-4, maxsteps=3,
                        amr_every=1, coarsen_frac=0.0, refine_frac=0.5,
                        max_level=max_level, max_leaves=120, linear=:direct,
                        amr_coarsen_gate=coarsen_gate,
                        checkpoint_path=ck)
                end
            end
        else
            open(logp, "w") do io
                redirect_stdout(io) do
                    run_semquad_simulation(qt, pr, th;
                        p=2, tstop=6.0e-4, dt0=1.0e-4, maxsteps=3,
                        amr_every=1, coarsen_frac=0.0, refine_frac=0.5,
                        max_level=max_level, max_leaves=120, linear=:direct,
                        amr_coarsen_gate=coarsen_gate,
                        amr_cap_aware=cap_aware,
                        checkpoint_path=ck)
                end
            end
        end
        st = load_semquad_state(ck)
        rm(ck; force=true)
        rm(logp; force=true)
        return (res=res, config=st.config)
    end

    function same_state(a, b)
        return (length(a.res.qt.leaves) == length(b.res.qt.leaves) &&
                a.res.theta == b.res.theta && a.res.ux == b.res.ux &&
                a.res.uz == b.res.uz && a.res.history == b.res.history)
    end

    # (a) omitted == explicit false, and the key is recorded false
    a = run_one(cap_aware=:omitted)
    b = run_one(cap_aware=false)
    @test same_state(a, b)
    @test a.config["amr_cap_aware"] == "false"
    @test b.config["amr_cap_aware"] == "false"

    # (b) a non-binding cap: explicit true is the same trajectory, records true
    c = run_one(cap_aware=true)
    @test same_state(a, c)
    @test c.config["amr_cap_aware"] == "true"
    # the flag reached the cycle (report echo), even though the cap cannot bind
    @test LAST_AMR_BALANCED[].cap_aware == true

    # The cycle never fires a cap-aware fallback in these runs (the cap cannot
    # bind); the report's `cap_aware` flag may still be true for the opt-in run.
    @test LAST_AMR_BALANCED[].cap_applied == false

    # (c) the invalid combination is refused before any work
    err = try
        run_one(cap_aware=true, coarsen_gate=false)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("amr_cap_aware", sprint(showerror, err))
    @test occursin("amr_coarsen_gate", sprint(showerror, err))
end
