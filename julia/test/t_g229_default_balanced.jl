using Test, MantleSEM, Printf

# G229: the library default of `run_semquad_simulation` is `amr_policy=:balanced`
# since 2026-11-13 (docs/dev/AMR.md, Stage 1).  G208 keeps the
# classic driver path pinned by passing `amr_policy=:classic` explicitly; this
# gate is the one that exercises the *default*, so a future revert to `:classic`
# fails here without touching G208 or G210.
#
# Same 5-step case and tree as G208 (d) so the two gates are a policy A/B:
#   (a) the checkpoint's run_config records "balanced", and the balanced-cycle
#       diagnostics ("balanced: gate_refused=...") are printed;
#   (b) two identical default runs are bitwise equal (deterministic);
#   (c) the balanced leaves and the three FNV hashes are pinned, so the new
#       default is frozen as tightly as the classic pin was.

"""FNV-1a 64 over the raw bytes of a Float64 vector (no external dependency)."""
function _fnv229(a)
    h = 0xcbf29ce484222325
    for b in reinterpret(UInt8, a)
        h ⊻= UInt64(b)
        h *= 0x100000001b3
    end
    return string(h, base=16, pad=16)
end

@testset "G229 default amr_policy is :balanced" begin
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                eta_b=9.70406052783923433184,
                eta_c=4.1588830833596718565,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)

    function run_default()
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
        # No `amr_policy` kwarg: this is the default under test.
        res = open(logpath, "w") do io
            redirect_stdout(io) do
                run_semquad_simulation(qt5, pr, th5; p=2, tstop=8.0e-4,
                                       dt0=1.0e-4, maxsteps=5, amr_every=1,
                                       coarsen_frac=0.0, refine_frac=0.5,
                                       marking=:error_fraction, max_level=5,
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

    # (a) the default is balanced, not classic
    @test a.config["amr_policy"] == "balanced"
    @test b.config["amr_policy"] == "balanced"
    @test occursin("balanced:", a.log)
    @test !occursin("amr_policy must be", a.log)

    # (b) the default path is deterministic, bit for bit
    @test length(a.res.qt.leaves) == length(b.res.qt.leaves)
    @test a.res.theta == b.res.theta
    @test a.res.ux == b.res.ux
    @test a.res.uz == b.res.uz
    @test a.res.history == b.res.history

    # (c) balanced pins (measured 2026-11-13 on this tree; G208 pins classic)
    @test length(a.res.qt.leaves) == 118
    @test _fnv229(reinterpret(UInt8, a.res.theta)) == "47ee97ea1cc249de"
    @test _fnv229(reinterpret(UInt8, vcat(a.res.ux, a.res.uz))) == "8a33b2262144a1fb"
    @test _fnv229(reinterpret(UInt8, reduce(vcat, a.res.history))) == "2ceaf052be8af11c"
end
