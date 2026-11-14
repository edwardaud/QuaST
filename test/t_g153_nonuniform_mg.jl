using Test, QuaST, LinearAlgebra

# G153: non-uniform quadtree multigrid and multi-root point location.
#
# The MG hierarchy is built by coarsening the tree one layer at a time (merge
# complete 2x2 sibling quartets, balance), so an adaptive/wall-refined forest
# gets a full hierarchy rather than collapsing to its base leaves.  Per level
# P_k = (fine free rows) * Pnodal * C_{k-1}; the free-row selector matters:
# using C_fine' instead would sum hanging rows onto masters and double count.
#
# This gate also locks a real point-location bug: `_semquad_leaf_at` used to
# check only the (0,0) base cell, so on a multi-root tree with a surviving
# level-0 leaf every point outside base cell 0 was evaluated in the wrong
# element.  Refined production meshes hid it because they have no level-0
# leaves; the MG coarse levels and AMR coarsening expose it.

function _g153_wall_tree(level::Int, wall::Int; cx::Int=5, cy::Int=2,
                         strip::Float64=0.1)
    qt = Quadtree(wall; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    refine_wall_to_min!(qt, wall; strip=strip)
    balance!(qt)
    return qt
end

@testset "G153 non-uniform MG and point location" begin
    # ---- multi-root point location with a level-0 leaf ----
    qtp = Quadtree(3; cells_x=5, cells_y=2)
    refine!(qtp, (0, 0, 0))
    refine!(qtp, (1, 0, 0))
    balance!(qtp)
    @test any(lf[1] == 0 for lf in qtp.leaves)
    @test length(unique(lf[1] for lf in qtp.leaves)) > 1
    sqp = make_semquad(qtp, 2)
    Lxp = qtp.Lx
    fld = [sin(pi * x / Lxp) * sin(pi * y) for (x, y) in sqp.coords]
    leavesp = Set(sqp.leaves)
    for gx in 0:20, gy in 0:8
        x = (gx + 0.5) / 21 * Lxp
        y = (gy + 0.5) / 9
        @test QuaST._semquad_eval_at(sqp, fld, x, y, leavesp) ≈
              semquad_eval(sqp, fld, x, y) atol = 1.0e-12
    end

    # ---- non-uniform MG: wall-refined 5x2 tree, 880 leaves ----
    qt = _g153_wall_tree(2, 4)
    @test max_face_level_jump(qt) <= 1
    @test length(unique(lf[1] for lf in qt.leaves)) > 1
    ss = make_semstokes(qt, 2)
    Lx = qt.Lx
    fz = (x, y) -> sin(pi * x / Lx) * sin(pi * y)
    z2 = (x, y) -> 0.0
    geo = QuaST._semquad_mg_geometry(ss, qt, 2, :rigid, false)
    @test geo.L >= 3
    @test issorted(geo.nfs)
    @test geo.nfs[end] == length(geo.free_rows[end])
    @test geo.nfs[end] == size(geo.Cus[end], 2)

    for (label, eta, bc, tol_du, maxit) in
        (("const", (x, y) -> 1.0, :rigid, 1.0e-8, 100),
         ("100x", (x, y) -> exp(-log(100.0) * (1 - y)), :rigid, 1.0e-8, 100),
         ("2b-like", (x, y) -> exp(-log(16384.0) * (1 - y)), :free, 1.0e-9, 100))
        u0, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, z2, fz, z2, z2;
                                                  bc=bc, linear=:direct)
        u1, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, z2, fz, z2, z2;
                                                  bc=bc, linear=:mg)
        @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
        @test norm(u1 - u0) / norm(u0) < tol_du
        @test LAST_SEMSTOKES_SCHUR_ITERS[] < maxit
    end

    # ---- AMR + MG: the tree changes during the run, so the hierarchy is
    # rebuilt for a non-uniform tree; this is the end-to-end path. ----
    qta = Quadtree(4; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(qta.leaves)
        refine!(qta, lf)
    end
    balance!(qta)
    ssa = make_semstokes(qta, 2)
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                eta_b=log(10.0), eta_c=log(2.0),
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th = [0.05 * sin(pi * x / qta.Lx) * sin(pi * y) for (x, y) in ssa.su.coords]
    for (i, (x, y)) in enumerate(ssa.su.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th[i] = 0.0)
    end
    # The comparison was calibrated on the classic cycle; pin it.
    st = run_semquad_simulation(qta, pr, th; p=2, tstop=1.0, dt0=1.0e-4,
                                cfl=5.0, maxsteps=4, amr_every=2,
                                refine_frac=0.3, ev=0.11, ev_cmax=0.052,
                                linear=:mg, verbose=false)
    @test st.steps == 4
    @test length(unique(lf[1] for lf in st.qt.leaves)) > 1
    @test all(isfinite, st.history[end])
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-8
end
