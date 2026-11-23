# G241: the card still reads, validates and forwards CASE_WALL_BC.
#
# (G258 added the sibling `CASE_WALL_UN` knob; its wiring block is at the end
# and is asserted the same way.)
#
# G228's convention: the card is an example script, so the gate asserts on its
# source rather than spawning a julia process.  The behavioural side of the knob was
# validated by hand at three settings (unset, a mixed tuple, a bad value); what this
# gate prevents is the *wiring* being lost in a later edit -- the failure mode that
# bit `wall_bc` once already, when the parameter was accepted and never used.

using Test, QuaST

@testset "G241 the card wires CASE_WALL_BC" begin
    card = normpath(joinpath(@__DIR__, "..", "examples", "case_realistic_2d.jl"))
    @test isfile(card)
    src = read(card, String)

    # 1. read from the environment, with an empty default
    @test occursin("CASE_WALL_BC", src)
    @test occursin("get(ENV, \"CASE_WALL_BC\", \"\")", src)

    # 2. both validations are present: four entries, and the three allowed values
    @test occursin("length(parts) == 4", src)
    @test occursin("all(w -> w in (:same, :free, :rigid), parts)", src)

    # 3. forwarded into Params, which is where the driver reads it
    @test occursin("wall_bc=WALL_BC", src)
    @test occursin("bc=:free, wall_bc=WALL_BC", src)

    # 4. and printed, so a log records which wall was which
    @test occursin("wall_bc=%s", src)
    @test occursin("join(string.(WALL_BC), \",\")", src)

    # 5. the library side the card relies on
    pr = Params()
    @test pr.wall_bc == (:same, :same, :same, :same)
    @test Params(wall_bc=(:rigid, :free, :free, :rigid)).wall_bc ==
          (:rigid, :free, :free, :rigid)
    @test_throws ArgumentError Params(wall_bc=(:free, :free, :free, :nope))

    # 6. G258 added the per-wall *normal* velocity knob the same way: read from
    #    the environment with an empty (= all `nothing`) default, four
    #    comma-separated entries, forwarded into `Params`, and printed.
    @test occursin("CASE_WALL_UN", src)
    @test occursin("get(ENV, \"CASE_WALL_UN\", \"\")", src)
    @test occursin("isempty(raw)", src)
    @test occursin("length(parts) == 4", src)
    @test occursin("CASE_WALL_UN entry", src)
    @test occursin("must be finite", src)
    @test occursin("wall_un=WALL_UN", src)
    @test occursin("wall_un=%s", src)
    @test Params(wall_un=(nothing, nothing, nothing, (x, y) -> 2.0)).wall_un[4](0.0, 0.0) == 2.0
    @test Params().wall_un == (nothing, nothing, nothing, nothing)

    # 7. G259 added the per-wall *tangential traction* knob with the same shape
    #    (empty = all `nothing` = the historical card, four finite entries,
    #    forwarded into `Params`, printed in the header).
    @test occursin("CASE_WALL_T", src)
    @test occursin("get(ENV, \"CASE_WALL_T\", \"\")", src)
    @test occursin("CASE_WALL_T entry", src)
    @test occursin("wall_t=WALL_T", src)
    @test occursin("wall_t=%s", src)
    @test occursin("join((v === nothing ? \"-\" : string(v) for v in WALL_T_VALUES), \",\")", src)
    @test Params(wall_t=(nothing, nothing, nothing, (x, y) -> 2.0)).wall_t[4](0.0, 0.0) == 2.0
    @test Params().wall_t == (nothing, nothing, nothing, nothing)
    # the card source still parses as Julia (the knob's const block is only
    # executed when the card runs, so a syntax error would otherwise survive)
    @test Meta.parseall(src) isa Expr
end
