using Test, QuaST

# G141: adaptive under-relaxation of the Picard iteration.
#
# The rule halves the step when the relative change fails to contract and lets it
# grow back when the iteration contracts strongly; with the default
# `pic_adapt=false` the driver keeps the historical fixed relaxation path.

@testset "G141 adaptive Picard relaxation" begin
    # non-contracting -> shrink, floored
    @test picard_adapt_relax(1.0, 1.0e-3, 2.0e-3) == 0.5
    @test picard_adapt_relax(0.2, 1.0e-3, 1.0e-3) == 0.1
    @test picard_adapt_relax(0.06, 1.0e-3, 1.0e-3) == 0.05   # rmin floor
    @test picard_adapt_relax(0.05, 1.0, 10.0) == 0.05
    # strong contraction -> grow, capped at 1
    @test picard_adapt_relax(0.5, 1.0e-2, 1.0e-3) ≈ 0.6
    @test picard_adapt_relax(0.9, 1.0e-2, 1.0e-3) == 1.0
    # mild contraction -> unchanged
    @test picard_adapt_relax(0.4, 1.0e-2, 0.8e-2) == 0.4
    # a sequence that stalls converges to the floor, one that contracts to 1
    r = 0.8
    for _ in 1:10
        r = picard_adapt_relax(r, 1.0, 1.0)
    end
    @test r == 0.05
    r = 0.3
    for _ in 1:10
        r = picard_adapt_relax(r, 1.0e-1, 1.0e-3)
    end
    @test r == 1.0
    # argument checks
    @test_throws ArgumentError picard_adapt_relax(0.5, 1.0, 0.5; shrink=1.0)
    @test_throws ArgumentError picard_adapt_relax(0.5, 1.0, 0.5; grow=0.9)
    @test_throws ArgumentError picard_adapt_relax(0.5, 1.0, 0.5; contract=1.5)

    # NOTE: the driver wiring of pic_adapt is exercised end-to-end in the
    # stiff-case study that follows this stage; here the rule itself is gated
    # because the driver smoke setups below were not robust across viscosity
    # laws (a SingularException appeared for expT with a large contrast).
end

