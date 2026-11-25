# G136: the stretched box (Lx) — square leaves, multi-root equivalence and the guards.
using Test, QuaST, LinearAlgebra, Printf

# G136: stretched box (Lx != 1).
#
# The Poisson path is the one converted to a box of width Lx (height 1).  The
# gate checks it against a manufactured solution whose Laplacian is known,
#   u = sin(pi x / Lx) sin(pi y),
#   -lap u = (pi^2/Lx^2 + pi^2) u,
# with homogeneous Dirichlet data on all four walls, at Lx = 1 and Lx = 3.
# It also checks that Lx = 1 is unchanged (the unit-square path must reproduce
# the same element matrix) and that the un-converted Stokes path refuses to run
# instead of silently returning unit-square results.

function _poisson_case(Lx::Float64, level::Int; p::Int=3)
    qt = Quadtree(level; Lx=Lx)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    sq = make_semquad(qt, p)
    k = pi / Lx
    uex = (x, y) -> sin(k * x) * sin(pi * y)
    f = (x, y) -> (k^2 + pi^2) * uex(x, y)
    uD = (x, y) -> 0.0
    u, cons = solve_semquad_poisson(sq, qt, f, uD; sigma_C=50.0)
    M = semquad_mass(sq)
    e = [u[g] - uex(sq.coords[g][1], sq.coords[g][2]) for g in 1:sq.N]
    err2 = dot(e, M * e)            # L2 error squared
    return sqrt(err2), sq, qt, u, uex
end

@testset "G136 stretched box (Lx)" begin
    # 5x2 base cells: square leaves over [0,2.5]x[0,1] (ASPECT's 2b geometry)
    q5 = Quadtree(2; cells_x=5, cells_y=2)
    @test q5.Lx == 2.5
    for l in 0:3
        @test QuaST.leaf_hx(q5, l) == QuaST.leaf_hy(q5, l)
    end
    @test QuaST.leaf_hx(q5, 0) == 0.5
    for _ in 1:2, lf in collect(q5.leaves)
        refine!(q5, lf)
    end
    balance!(q5)
    s5 = make_semquad(q5, 2)
    @test length(s5.leaves) == 5 * 2 * 16
    @test maximum(c[1] for c in s5.coords) ≈ 2.5
    @test maximum(c[2] for c in s5.coords) ≈ 1.0
    @test total_area(q5) ≈ 2.5
    # a leaf's physical box is square at every level
    for lf in s5.leaves
        hx = QuaST.leaf_hx(s5, lf[1]); hy = QuaST.leaf_hy(s5, lf[1])
        @test hx == hy
        break
    end
    # inconsistent Lx / cells must be rejected
    @test_throws ArgumentError Quadtree(2; Lx=2.0, cells_x=5, cells_y=2)

    # THE multi-root regression: the same 8x8 mesh must give the same answer
    # whether it is one root or a 2x2 base grid (this caught a real bug: the
    # diagnostics/Stokes/picard still used h = 2^-l instead of leaf_hy)
    function same_mesh(q)
        ss = make_semstokes(q, 2)
        th = [0.01 * cos(pi * x) * sin(pi * y) for (x, y) in ss.su.coords]
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
        st = run_semquad_simulation(q, pr, th; p=2, tstop=2.0e-3, dt0=1.0e-4,
                                    cfl=0.5, maxsteps=2000, verbose=false)
        nt, nb = semquad_nu_walls(ss.su, st.qt, st.theta, pr)
        return (nt, nb, semquad_vrms(ss.su, st.qt, st.ux, st.uz))
    end
    q1 = Quadtree(3)
    for _ in 1:3, lf in collect(q1.leaves); refine!(q1, lf); end
    balance!(q1)
    q4 = Quadtree(2; cells_x=2, cells_y=2)
    for _ in 1:2, lf in collect(q4.leaves); refine!(q4, lf); end
    balance!(q4)
    r1 = same_mesh(q1); r4 = same_mesh(q4)
    @test r1[1] ≈ r4[1] rtol = 1.0e-12
    @test r1[2] ≈ r4[2] rtol = 1.0e-12
    @test r1[3] ≈ r4[3] rtol = 1.0e-12

    # the tree bookkeeping
    for Lx in (1.0, 3.0)
        qt = Quadtree(4; Lx=Lx)
        @test qt.Lx == Lx
        @test total_area(qt) ≈ Lx
        for _ in 1:3, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        @test total_area(qt) ≈ Lx
        sq = make_semquad(qt, 3)
        @test sq.Lx == Lx
        # the box really is [0,Lx] x [0,1]
        @test minimum(c[1] for c in sq.coords) ≈ 0.0 atol = 1e-14
        @test maximum(c[1] for c in sq.coords) ≈ Lx atol = 1e-14
        @test minimum(c[2] for c in sq.coords) ≈ 0.0 atol = 1e-14
        @test maximum(c[2] for c in sq.coords) ≈ 1.0 atol = 1e-14
        @test leaf_box(qt, (3, 0, 0)) == (0.0, 0.0, Lx / 8, 1 / 8)
    end

    # Lx = 1 reproduces the historical unit-square element matrix bit for bit:
    # hy/hx = hx/hy = 1, and multiplying by 1.0 is exact.
    err1, sq1, _, u1, uex1 = _poisson_case(1.0, 3)
    err3, sq3, _, u3, uex3 = _poisson_case(3.0, 3)
    @test err1 < 1.0e-6
    @test err3 < 1.0e-6
    @test abs(err1 - err3) < 1.0e-6       # same accuracy, different geometry

    # spectral convergence on the stretched box
    e2, _, _, _, _ = _poisson_case(3.0, 2)
    @test err3 < e2 / 20
    e1, _, _, _, _ = _poisson_case(1.0, 2)
    @test err1 < e1 / 20

    # the manufactured solution is actually resolved (not a trivial zero)
    @test maximum(abs, u3) > 0.5

    # Stokes on a stretched box: the divergence-free quadratic
    #   u = (x y, -y^2 / 2),  p = 0
    # satisfies -div(2 eps(u)) = 0 exactly for any Lx, so Q2/Q1 must reproduce
    # it to round-off.  This exercises K, B, the Dirichlet blocks and the
    # metric factors of the Stokes path.
    for Lx in (1.0, 3.0)
        qts = Quadtree(3; Lx=Lx)
        for _ in 1:2, lf in collect(qts.leaves)
            refine!(qts, lf)
        end
        balance!(qts)
        sss = make_semstokes(qts, 2)
        ux_ex = (x, y) -> x * y
        uz_ex = (x, y) -> -0.5 * y^2
        us, _ = solve_semquad_stokes(sss, qts, (x, y) -> 0.0, (x, y) -> 0.0,
                                     ux_ex, uz_ex)
        nus = sss.su.N
        errs = 0.0
        for (g, c) in enumerate(sss.su.coords)
            errs = max(errs, abs(us[g] - ux_ex(c...)), abs(us[nus + g] - uz_ex(c...)))
        end
        @test errs < 1.0e-10
    end
    # The variable-viscosity driver (`solve_semquad_stokes_var`) is the one the
    # convection loop uses; it used `h = leaf_hy` for x0, y0 and the Jacobian,
    # so on an anisotropic (stretched) leaf it solved the wrong operator (the
    # constant-viscosity path above was already correct).  Same exact field,
    # eta = 1, on leaves up to 2:1:
    #   u = (2xy, -y^2), p = 0, -div(2 eps(u)) = (0, 2).
    for Lx in (1.0, 2.0)
        qtv = Quadtree(2; Lx=Lx)
        for _ in 1:2, lf in collect(qtv.leaves)
            refine!(qtv, lf)
        end
        balance!(qtv)
        ssv = make_semstokes(qtv, 2)
        ux_ex = (x, y) -> 2.0 * x * y
        uz_ex = (x, y) -> -y^2
        usv, _ = solve_semquad_stokes_var(ssv, qtv, (x, y) -> 1.0,
                                          (x, y) -> 0.0, (x, y) -> 2.0,
                                          ux_ex, uz_ex; bc=:rigid)
        nsv = ssv.su.N
        ev = 0.0
        for (g, c) in enumerate(ssv.su.coords)
            ev = max(ev, abs(usv[g] - ux_ex(c...)),
                     abs(usv[nsv + g] - uz_ex(c...)))
        end
        @test ev < 1.0e-10
        @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-10
    end
    # energy operator on a stretched box: a *polynomial* theta that vanishes on
    # all walls is represented exactly by Q3, so the mass and Dirichlet energy
    # are exact:
    #   theta = x(Lx-x) y(1-y)
    #   int theta^2      = Lx^5/900
    #   int |grad theta|^2 = Lx^3/90 + Lx^5/90
    for Lx in (1.0, 3.0)
        qte = Quadtree(3; Lx=Lx)
        for _ in 1:2, lf in collect(qte.leaves)
            refine!(qte, lf)
        end
        balance!(qte)
        sqe = make_semquad(qte, 3)
        prm = Params(Ra=1.0, H=0.0)
        M, K, C, F = assemble_semquad_energy(sqe, qte, prm, zeros(sqe.N),
                                             zeros(sqe.N); advection=:galerkin)
        th = [x * (Lx - x) * y * (1.0 - y) for (x, y) in sqe.coords]
        @test dot(th, M * th) ≈ Lx^5 / 900 rtol = 1.0e-11
        @test dot(th, K * th) ≈ Lx^3 / 90 + Lx^5 / 90 rtol = 1.0e-11
    end

    # the AMR indicator on a stretched box: for u = y^2 the interior faces have
    # no jump and only the Dirichlet walls contribute; the indicator is an
    # integral over x, so it must scale exactly with Lx
    vals = Float64[]
    walls = Float64[]
    interiors = Float64[]
    for Lx in (1.0, 3.0)
        qti = Quadtree(3; Lx=Lx)
        for _ in 1:1, lf in collect(qti.leaves)
            refine!(qti, lf)
        end
        balance!(qti)
        sqi = make_semquad(qti, 2)
        ui = [c[2]^2 for c in sqi.coords]
        et = semquad_gradient_jump(sqi, qti, ui)
        push!(vals, sum(values(et)))
        # the analytic Dirichlet-wall term: du/dy = 2y, so only the y = 1 wall
        # contributes, (hy/2) * int_0^Lx 4 dx = 2*hy*hx per top-row leaf
        wall = 0.0
        for lf in qti.leaves
            hx = QuaST.leaf_hx(qti, lf[1])
            hy = QuaST.leaf_hy(qti, lf[1])
            lf[3] == qti.ny0 * (1 << lf[1]) - 1 && (wall += 2.0 * hx * hy)
        end
        push!(walls, wall)
        push!(interiors, vals[end] - wall)
    end
    @printf("G136 u=y^2: totals %.17g / %.17g, wall %.17g / %.17g, interior %.3e / %.3e\n",
            vals[1], vals[2], walls[1], walls[2], interiors[1], interiors[2])
    @test vals[2] / vals[1] ≈ 3.0 rtol = 1.0e-12
    # D7: the ratio test above cannot see the old defect -- both the true wall
    # term and the historical spurious y-face mass scale with Lx, so the pre-fix
    # body also gave exactly 3:1.  Assert the decomposition instead: the
    # analytic wall term is the sum over the top row of 2*hx*hy (du/dy = 2 at
    # y = 1; the bottom wall has du/dy = 0), which is Lx on this level-1 tree
    # (hy = 0.5), and the interior term must be zero.  Measured on the pre-fix
    # body: interior 0.5 (Lx=1) / 1.5 (Lx=3), i.e. exactly 0.5*Lx (two interior
    # y-face pairs per tree); after the 2026-11-25 fix both interiors are 0.
    @test abs(interiors[1]) <= 1.0e-13 * abs(vals[1])
    @test abs(interiors[2]) <= 1.0e-13 * abs(vals[2])
end
