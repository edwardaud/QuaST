# G148: the HDF5 checkpoint round trip, including the base cell counts.
using Test, QuaST, LinearAlgebra

# G148: HDF5 checkpoint round trip, including the base cell counts.
#
# The checkpoint used to store only maxlevel, so a 5x2 (multi-root) state was
# rebuilt as a single-root tree and load_semquad_state rejected it with
# "leaf list is not a partition".  This gate runs a short simulation with
# checkpointing on a multi-root grid, reloads the file and checks that the state
# comes back exactly.

@testset "G148 checkpoint round trip" begin
    for (cx, cy) in ((1, 1), (5, 2))
        qt = Quadtree(2; cells_x=cx, cells_y=cy)
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
        th0 = [0.01 * cos(pi * x / (cx / cy)) * sin(pi * y) for (x, y) in ss.su.coords]
        path = joinpath(tempdir(), "g148_ck_$(cx)x$(cy).h5")
        st = run_semquad_simulation(qt, pr, th0; p=2, tstop=5.0e-4, dt0=1.0e-4,
                                    cfl=0.5, maxsteps=5, verbose=false,
                                    checkpoint_path=path, checkpoint_every=1)
        ld = load_semquad_state(path)
        # geometry comes back exactly
        @test ld.qt.nx0 == cx
        @test ld.qt.ny0 == cy
        @test ld.qt.Lx ≈ cx / cy
        @test length(ld.qt.leaves) == length(st.qt.leaves)
        @test Set(ld.qt.leaves) == Set(st.qt.leaves)
        # and so does the state
        @test ld.step == st.steps
        @test ld.t ≈ st.t rtol = 1.0e-12
        @test ld.theta ≈ st.theta rtol = 1.0e-12
        @test length(ld.theta) == length(st.theta)
        rm(path; force=true)
    end
end
