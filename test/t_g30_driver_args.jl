# G30: the high-Ra driver's argument mapping and validation.
#
# The extreme-case runs must be reproducible from the command line alone, with
# H, advection form, nonlinear iteration count and checkpoint cadence all
# explicit.  This gate freezes the positional contract.
using Test
using QuaST

include("../examples/highra_run.jl")

@testset "G30 highra driver argument mapping" begin
    base = ["16", "1e8", "1e3", "1e-4", "4", "caseA", "20", "element",
            "0.11", "0.052", "32", "2", "schur_cg", "-", "global", "1e-7"]

    a = parse_args(vcat(base, "5.0", "galerkin", "3", "1e-6"))
    @test a.n == 16
    @test a.Ra == 1.0e8
    @test a.deta == 1.0e3
    @test a.tstop == 1.0e-4
    @test a.nz == 32
    @test a.p == 2
    @test a.linear == :schur_cg
    @test a.H == 5.0
    @test a.adv == :galerkin
    @test a.maxpic == 3
    @test a.save_dt == 1.0e-6

    b = parse_args(String[])
    @test b.H == 0.0
    @test b.adv == :skew
    @test b.maxpic == 1
    @test b.save_dt == 0.0

    @test_throws ArgumentError parse_args(vcat(base, "5.0", "bogus"))
    @test_throws ArgumentError parse_args(vcat(base, "5.0", "galerkin", "0"))
    @test_throws ArgumentError parse_args(vcat(base, "-1.0"))
    @test_throws ArgumentError parse_args(vcat(base, "5.0", "galerkin", "3", "-1e-6"))
end
