# G245: the restart guard records viscoelasticity and the per-wall tuple.
#
# A checkpoint compares its stored `run_config` against the current run's keys, so a
# parameter that changes the trajectory but is missing from that dictionary can be
# switched silently across a restart.  `shear_modulus` and `wall_bc` were both in that
# position.  The guard tolerates keys an older checkpoint lacks
# (`haskey(run_config, k) || continue`), so adding them is safe for existing files --
# which is exactly why this gate asserts the *source* keeps them: the safety of the
# addition depends on a property of the guard, and losing the keys would not fail
# anything else.

using Test, QuaST

@testset "G245 run_config records the new physics" begin
    src = read(normpath(joinpath(@__DIR__, "..", "src", "driver", "restart_guard.jl")), String)

    @test occursin("run_config[\"shear_modulus\"] = pr.shear_modulus", src)
    @test occursin("run_config[\"wall_bc\"] = join(string.(pr.wall_bc), \",\")", src)

    # the guard really does skip keys an old checkpoint does not carry; if this
    # property is ever removed, the two keys above become breaking changes
    @test occursin("haskey(run_config, k) || continue", src)

    # and the values the keys record are the ones the run uses
    pr = Params(shear_modulus=1.5e3, wall_bc=(:rigid, :free, :free, :rigid))
    @test pr.shear_modulus == 1.5e3
    @test join(string.(pr.wall_bc), ",") == "rigid,free,free,rigid"

    # ------------------------------------------------------------------
    # G258: the per-wall velocity *values* belong to the trajectory too.
    # `wall_u` was missing from the guard before G258 (only `wall_bc` was
    # recorded) and `wall_un` is new with it.  Both are tuples of functions,
    # which cannot be serialised or compared, so each is recorded with the
    # `eta_multiplier`/`rho_multiplier` convention: one `none`/`custom` marker
    # per wall.  A restart that changes a marker is refused.
    # ------------------------------------------------------------------
    @test occursin("run_config[\"wall_u\"] = _wall_value_marker(pr.wall_u)", src)
    @test occursin("run_config[\"wall_un\"] = _wall_value_marker(pr.wall_un)", src)
    mk(t) = QuaST._wall_value_marker(t)
    @test mk((nothing, nothing, nothing, nothing)) == "none,none,none,none"
    @test mk((nothing, nothing, nothing, (x, y) -> 1.0)) == "none,none,none,custom"
    @test mk(((x, y) -> 0.0, nothing, nothing, nothing)) == "custom,none,none,none"
    # the guard compares `config_value_string` of the stored value, so a changed
    # marker is a mismatch (the checkpoint string is the marker itself)
    @test QuaST.config_value_string(mk((nothing, nothing, nothing, nothing))) ==
          "none,none,none,none"
    @test QuaST.config_value_string(mk((nothing, nothing, nothing,
                                        (x, y) -> 1.0))) !=
          QuaST.config_value_string(mk((nothing, nothing, nothing, nothing)))
    # the end-to-end refusal (write a checkpoint with one `wall_un`, resume with
    # another, expect the guard's mismatch error) lives in G258, which already
    # pays the driver's compilation cost; this gate stays source/marker-only so
    # running it alone remains fast.
end
