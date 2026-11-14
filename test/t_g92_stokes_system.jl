# G92: the Stokes system exposure — return_system returns all twelve pieces.
﻿using Test, QuaST, SparseArrays
@testset "G92 Stokes system exposure" begin
    qt = Quadtree(1)
    ss = make_semstokes(qt, 2)
    out = solve_semquad_stokes_var(ss, qt, (x,y)->1.0,
                                   (x,y)->0.0, (x,y)->2.0,
                                   (x,y)->0.0, (x,y)->0.0;
                                   bc=:free, linear=:direct,
                                   return_system=true)
    @test length(out) == 12
    u, pv, fixed, cons2, cons_p, K, B, Cu, Cp, u0, p0, fv = out
    @test size(K,1) == 2*ss.su.N
    @test size(B,1) == ss.sp.N
    @test size(B,2) == 2*ss.su.N
    @test size(Cu,1) == 2*ss.su.N
    @test size(Cp,1) == ss.sp.N
    @test length(u) == 2*ss.su.N
    @test length(fv) == 2*ss.su.N
end
