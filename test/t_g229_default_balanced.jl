# G229: the default AMR path is the balanced policy — no policy keyword required.
using Test, QuaST, Printf

# G229: the library path of `run_semquad_simulation` is the balanced AMR policy.
# Since the classic policy was removed at `8c622df`
# the balanced cycle is the only adaptive policy, so this gate pins the
# *default* driver path itself: a caller that passes no AMR keyword at all gets
# the balanced cycle, its checkpoint records `amr_policy = "balanced"`, the
# diagnostics are printed, and the run is deterministic.
#
# Same 5-step case and tree as G208 (d) so the two gates were a policy A/B:
#   (a) the checkpoint's run_config records "balanced", and the balanced-cycle
#       diagnostics ("balanced: gate_refused=...") are printed;
#   (b) two identical default runs are bitwise equal (deterministic);
#   (c) the balanced leaves are pinned, so the default path is frozen.

"""FNV-1a 64 over the raw bytes of a Float64 vector (no external dependency)."""
function _fnv229(a)
    h = 0xcbf29ce484222325
    for b in reinterpret(UInt8, a)
        h ⊻= UInt64(b)
        h *= 0x100000001b3
    end
    return string(h, base=16, pad=16)
end

@testset "G229 default AMR path is the balanced policy" begin
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                eta_b=9.70406052783923433184,
                eta_c=4.1588830833596718565,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)

    # A gate that pins the default path has to start from the module's defaults.
    # The process-global MG cache outlives a run, so this gate clears it.  The
    # AMR hysteresis state no longer needs the reset it used to do here: stage 3a
    # made it a local of `run_semquad_simulation` (returned as `.cooldown`),
    # created fresh by every call, so a default run can no longer inherit the
    # cooldown that an earlier gate in the same process advanced.  Determinism is
    # therefore a property of the driver, not of a test-side global reset.
    function reset_module_state()
        empty!(QuaST.SEMQUAD_MG_CACHE)
        return nothing
    end

    function run_default()
        reset_module_state()
        qt5 = Quadtree(5; cells_x=5, cells_y=2, Lx=2.5)
        for _ in 1:2, lf in collect(qt5.leaves)
            refine!(qt5, lf)
        end
        balance!(qt5)
        ss5 = make_semstokes(qt5, 2)
        th5 = [0.1 * sinpi(y) * cospi(x / qt5.Lx) for (x, y) in ss5.su.coords]
        for (g, (_, y)) in enumerate(ss5.su.coords)
            (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th5[g] = 0.0)
        end
        ck = tempname() * ".h5"
        logpath = tempname()
        # No `amr_policy` kwarg: this is the default (and only) path.
        res = open(logpath, "w") do io
            redirect_stdout(io) do
                run_semquad_simulation(qt5, pr, th5;
                                       p=2, tstop=8.0e-4,
                                       dt0=1.0e-4, maxsteps=5, amr_every=1,
                                       coarsen_frac=0.0, refine_frac=0.5,
                                       max_level=5,
                                       max_leaves=120, linear=:direct,
                                       checkpoint_path=ck, verbose=true)
            end
        end
        st = load_semquad_state(ck)
        logtxt = read(logpath, String)
        rm(ck; force=true); rm(logpath; force=true)
        return (res=res, config=st.config, log=logtxt)
    end

    a = run_default()
    b = run_default()

    # (a) the run is the balanced cycle and records itself as such
    @test a.config["amr_policy"] == "balanced"
    @test b.config["amr_policy"] == "balanced"
    @test occursin("balanced:", a.log)

    # (b) the default path is deterministic, bit for bit
    @test length(a.res.qt.leaves) == length(b.res.qt.leaves)
    @test a.res.theta == b.res.theta
    @test a.res.ux == b.res.ux
    @test a.res.uz == b.res.uz
    @test a.res.history == b.res.history

    # (c) the default path is frozen
    @test length(a.res.qt.leaves) == 118

end
