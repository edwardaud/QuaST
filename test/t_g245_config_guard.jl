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
end
