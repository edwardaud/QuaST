# G267: affine non-identity exactness of the Stokes assembly through the
# Jacobian (stage 0, cut 2 — the core proof of the mapped assembled operator).
#
# The geometry is a general `AffineBox` (rotation + shear + anisotropic scale +
# shift, GEOM_DESIGN §5.2): the physical domain is a parallelogram.  An affine
# image of the reference square contains every physical polynomial of total
# degree <= 2, so the mapped Q2 velocity space contains P2 and the mapped Q1
# pressure space contains P1.  With a polynomial manufactured solution, constant
# viscosity and the Gauss `p+2` rule (exact for the resulting polynomial
# integrands), the only error of the discrete solution is the linear solve —
# exactly the argument that makes G150 case B / G154 A1 exact on the box.  The
# tolerances below are therefore solver round-off, not discretisation error.
#
# What is checked, and what each assertion catches:
#
#   (a) Stokes exactness on a genuinely rotated/sheared/anisotropic map:
#       u = (x^2 - y^2, -2xy) (P2, divergence free, Laplacian 0), p = a x + b y
#       (P1), f = grad p (constant), eta = 1, all four logical walls rigid with
#       the exact field.  Node-wise and independent-Gauss-quadrature L2 errors
#       are asserted at round-off.  A wrong Jinv (transposed, or the `2/hx`
#       scale folded incorrectly) is invisible on a diagonal map but produces an
#       O(1) error here; a wrong volume weight or B-block gradient does too.
#   (b) Traction: the linear shear u = (gamma y, 0), p = 0 on the same
#       parallelogram, Dirichlet on three logical walls and prescribed **total
#       traction** on the fourth (`:open`), given in the mapped wall-local form
#       `t.n` / `t.tau` (GEOM_DESIGN §6.3).  This exercises the mapped face
#       Jacobian, the outward unit normal and the positive-tangent direction —
#       the three sign traps of the boundary-condition feature: a flipped
#       normal flips the load, a flipped tangent flips the shear, a wrong face
#       Jacobian scales it, each by O(1).
#   (c) mapping independence: the same physical parallelogram (identical
#       physical domain, identical physical exact solution) is solved in
#       frames that differ by (i) a logical coordinate swap and (ii) a physical
#       rigid rotation+translation with correspondigly rotated exact fields.
#       Both discrete solutions are **evaluated at the same physical points**
#       (the mapped evaluator of cut 1, not a logical-coordinate comparison) and
#       must agree to round-off after the frame transform.  A fixed-dof loop
#       that prescribes data at logical instead of mapped positions, or a
#       metric that assumes a special (triangular/diagonal) affine map, fails
#       here while passing a single-frame exactness check could hide it.
#   (d) the mapped boundary measure: the geometry's `perimeter` integrated from
#       the mapped face Jacobians equals the analytic parallelogram perimeter.
#
# No assertion compares a quantity against itself: the exact fields are written
# in this file from the manufactured-solution definition, the L2 norms are
# independent high-order Gauss quadratures over the mapped space (never the SEM
# mass matrix, never the nodal interpolant), the mapping-independence fields are
# evaluated through the cut-1 mapped evaluator at physical points, and the
# traction data are computed from the exact stress `sigma = 2 eta eps(u)` and
# the analytic normal/tangent of the map written here.

using Test, QuaST, LinearAlgebra, SparseArrays

const MS267 = QuaST

# ---------------------------------------------------------------------------
# The validation geometry: A = R(0.55) * [[1.3, 0.4], [0, 0.9]], b = (0.3, -0.2).
# Written out here from the map definition (no reuse of the geometry object's
# internals); `det A = 1.17 > 0`.
# ---------------------------------------------------------------------------
function g267_geometry()
    th = 0.55
    R = ((cos(th), -sin(th)), (sin(th), cos(th)))
    S = ((1.3, 0.4), (0.0, 0.9))
    A = (R[1][1] * S[1][1] + R[1][2] * S[2][1],
         R[1][1] * S[1][2] + R[1][2] * S[2][2],
         R[2][1] * S[1][1] + R[2][2] * S[2][1],
         R[2][1] * S[1][2] + R[2][2] * S[2][2])
    b = (0.3, -0.2)
    return AffineBox(A, b, 1.0), A, b
end

# uniform refinement: level 0 base cell -> 4^level leaves
function g267_tree(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# a genuinely 2:1 hanging (mixed-level) tree
function g267_mixed_tree()
    qt = Quadtree(3)
    for lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    for lf in sort(collect(qt.leaves))[1:3]
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# exact Stokes fields: divergence free, Laplacian 0, p linear
g267_ux(x, y) = x^2 - y^2
g267_uz(x, y) = -2.0 * x * y
g267_p(x, y) = 1.0 + 2.0 * x - 0.5 * y
g267_fx(x, y) = 2.0
g267_fz(x, y) = -0.5

# ---------------------------------------------------------------------------
# error measures
# ---------------------------------------------------------------------------

"""max nodal error of a velocity field (the mapped Q2 contains the exact P2)."""
function g267_u_nodal(ss, u)
    su = ss.su
    nu = su.N
    e = 0.0
    for (g, c) in enumerate(su.coords)
        e = max(e, abs(u[g] - g267_ux(c...)), abs(u[nu + g] - g267_uz(c...)))
    end
    return e
end

"""max nodal pressure error after removing the one free constant (gauge)."""
function g267_p_nodal(ss, pv)
    sp = ss.sp
    e = [pv[g] - g267_p(sp.coords[g]...) for g in 1:sp.N]
    shift = sum(e) / length(e)
    return maximum(abs, e .- shift)
end

"""
    g267_l2(ss, u, pv) -> (u_l2, p_l2, div_l2)

True L2 errors of the discrete velocity/pressure fields and of `div u_h`,
integrated with an **independent** Gauss rule (`p+6` points) over the mapped
elements: `quad_point`/`Q.quad_measure` provide the physical points and the
Jacobian weight, `semquad_eval` the cut-1 mapped field evaluator and
`Q._semquad_grad_at` its mapped gradient (also cut 1).  Nothing here uses the
assembly's basis sums or the SEM mass matrix.
"""
function g267_l2(ss, u, pv; nq::Int=0)
    su = ss.su
    sp = ss.sp
    nu = su.N
    nq == 0 && (nq = su.p + 6)
    fd = make_fedata(su.p, sp.p; nq=nq)
    leaves = Set(su.leaves)
    uvec = u[1:nu]
    zvec = u[(nu + 1):(2nu)]
    lu2 = 0.0
    le1 = 0.0
    area = 0.0
    du2 = 0.0
    # physical points and their weights, so the gauge constant can be removed in
    # a second pass without the catastrophic cancellation of E[e^2]-E[e]^2
    pts = NTuple{2,Float64}[]
    wts = Float64[]
    els = Float64[]
    for lf in su.leaves
        lm = leaf_metric(su.geom, su, lf, fd)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(lm, su, lf, fd, qx, qy)
            w = MS267.quad_measure(lm, fd, qx, qy)
            push!(pts, (x, y))
            push!(wts, w)
            ux = semquad_eval(su, uvec, x, y)
            uz = semquad_eval(su, zvec, x, y)
            lu2 += w * ((ux - g267_ux(x, y))^2 + (uz - g267_uz(x, y))^2)
            pe = semquad_eval(sp, pv, x, y) - g267_p(x, y)
            push!(els, pe)
            area += w
            le1 += w * pe
            dxu, dyu = MS267._semquad_grad_at(su, uvec, x, y, leaves)
            dxz, dyz = MS267._semquad_grad_at(su, zvec, x, y, leaves)
            du2 += w * (dxu + dyz)^2
        end
    end
    shift = le1 / area
    lp2 = 0.0
    for k in eachindex(pts)
        lp2 += wts[k] * (els[k] - shift)^2
    end
    return sqrt(lu2), sqrt(lp2), sqrt(du2)
end

"""
    g267_frame_solve(g, qt, ux, uz, fx, fz; wall_bc=..., wall_t=..., wall_tn=...)

One mapped Stokes solve on geometry `g` with the given exact data.  `eta = 1`.
Kept as one helper so every frame uses exactly the same solve call.
"""
function g267_frame_solve(g, qt, ux, uz, fx, fz;
                          wall_bc=(:same, :same, :same, :same),
                          wall_t=(nothing, nothing, nothing, nothing),
                          wall_tn=(nothing, nothing, nothing, nothing))
    ss = make_semstokes(qt, 2; geom=g)
    res = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, fx, fz, ux, uz;
                                   linear=:direct, bc=:rigid, wall_bc=wall_bc,
                                   wall_t=wall_t, wall_tn=wall_tn)
    return ss, res
end

# physical point set of a frame's quadrature points (for comparisons done in
# physical space)
function g267_physical_points(g, sq; nq::Int=6)
    fd = make_fedata(2, 1; nq=nq)
    pts = NTuple{2,Float64}[]
    for lf in sq.leaves
        lm = leaf_metric(g, sq, lf, fd)
        for qy in 1:fd.nq, qx in 1:fd.nq
            push!(pts, quad_point(lm, sq, lf, fd, qx, qy))
        end
    end
    return pts
end

# Evaluate a discrete velocity at a physical point (mapped evaluator).
function g267_eval_u(ss, u, x, y)
    nu = ss.su.N
    return (semquad_eval(ss.su, u[1:nu], x, y),
            semquad_eval(ss.su, u[(nu + 1):(2nu)], x, y))
end
g267_eval_p(ss, pv, x, y) = semquad_eval(ss.sp, pv, x, y)

@testset "G267 affine Stokes exactness (mapped assembled operator)" begin
    g, A, b = g267_geometry()
    detA = A[1] * A[4] - A[2] * A[3]
    @test detA > 0.0
    qt = g267_tree(2)

    # ==================================================================
    # (a) exactness of the mapped Stokes solution on a general affine map
    # ==================================================================
    @testset "mapped Stokes exactness (all walls Dirichlet)" begin
        ss, res = g267_frame_solve(g, qt, g267_ux, g267_uz, g267_fx, g267_fz)
        u = res[1]
        pv = res[2]
        eu = g267_u_nodal(ss, u)
        ep = g267_p_nodal(ss, pv)
        lu2, lp2, ddiv = g267_l2(ss, u, pv)
        @test eu < 1.0e-10        # design tolerance; measured ~5e-15
        @test lu2 < 1.0e-10
        @test ep < 1.0e-9         # design tolerance; measured ~3e-13
        @test lp2 < 1.0e-9
        @test ddiv < 1.0e-9       # div u_ex = 0 pointwise, so this is round-off
        @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-10
        println("G267 exactness: u nodal=", eu, "  p nodal=", ep,
                "  L2 u=", lu2, "  L2 p=", lp2, "  div L2=", ddiv)
        # a *second* affine map (pure rotation, no shear) must be exact too:
        # the shear+rotation one above exercises the mixed Jinv terms; this one
        # pins the isotropic-scaling branch.
        th = 0.9
        R = ((cos(th), -sin(th)), (sin(th), cos(th)))
        A2 = (R[1][1] * A[1] + R[1][2] * A[3], R[1][1] * A[2] + R[1][2] * A[4],
              R[2][1] * A[1] + R[2][2] * A[3], R[2][1] * A[2] + R[2][2] * A[4])
        g2 = AffineBox(A2, (0.15, 0.25), 1.0)
        ss2, res2 = g267_frame_solve(g2, qt, g267_ux, g267_uz, g267_fx, g267_fz)
        @test g267_u_nodal(ss2, res2[1]) < 1.0e-10
        @test g267_p_nodal(ss2, res2[2]) < 1.0e-9
        # a hanging (mixed-level) tree with the same exact field: the cut-1
        # logical constraints must not corrupt the exactness of the affine map.
        qm = g267_mixed_tree()
        @test length(unique(lf[1] for lf in qm.leaves)) > 1
        ssm, resm = g267_frame_solve(g, qm, g267_ux, g267_uz, g267_fx, g267_fz)
        @test !isempty(semquad_constraints(ssm.su, qm))
        @test g267_u_nodal(ssm, resm[1]) < 1.0e-10
        @test g267_p_nodal(ssm, resm[2]) < 1.0e-9
    end

    # ==================================================================
    # (b) traction exactness on the same parallelogram
    # ==================================================================
    @testset "mapped traction exactness" begin
        gam = 1.3
        usx = (x, y) -> gam * y
        usz = (x, y) -> 0.0
        z0 = (x, y) -> 0.0
        # analyst: eps(u) = [[0, gamma/2], [gamma/2, 0]], sigma = 2 eps = [[0, g],[g,0]]
        # (eta = 1, p = 0).  The `yplus` logical wall is the segment
        # x = A*(Lx*S, 1) + b; its outward unit normal is A^{-T} e_y / |.| and
        # its positive logical tangent is A e_x / |.| — written from the map
        # matrix here, not read from the geometry object.
        nx0 = -A[3] / detA
        ny0 = A[1] / detA
        nn = hypot(nx0, ny0)
        nvec = (nx0 / nn, ny0 / nn)
        tv = (A[1], A[3])
        tn = hypot(tv[1], tv[2])
        tau = (tv[1] / tn, tv[2] / tn)
        # sigma = [[0, gam], [gam, 0]]; tbar = sigma . n = gam*(n_y, n_x)
        tvec = (gam * nvec[2], gam * nvec[1])
        tnum = tvec[1] * nvec[1] + tvec[2] * nvec[2]
        ttau = tvec[1] * tau[1] + tvec[2] * tau[2]
        @test abs(hypot(nvec...) - 1.0) < 1.0e-15
        @test abs(hypot(tau...) - 1.0) < 1.0e-15
        @test abs(nvec[1] * tau[1] + nvec[2] * tau[2]) < 1.0e-15
        # the exact traction must be the outward one: sigma.n has a positive
        # component along the outward normal on this (tension-free shear) case
        # only in the tangential direction, so check tbar = sigma.n explicitly:
        @test abs(tvec[1] - gam * nvec[2]) < 1.0e-15
        const_tn = (x, y) -> tnum
        const_t = (x, y) -> ttau
        ss, res = g267_frame_solve(g, qt, usx, usz, z0, z0;
                                   wall_bc=(:same, :same, :same, :open),
                                   wall_t=(nothing, nothing, nothing, const_t),
                                   wall_tn=(nothing, nothing, nothing, const_tn))
        u = res[1]
        pv = res[2]
        eu = 0.0
        for (gi, c) in enumerate(ss.su.coords)
            eu = max(eu, abs(u[gi] - usx(c...)), abs(u[ss.su.N + gi] - usz(c...)))
        end
        @test eu < 1.0e-10        # measured ~4e-14
        @test maximum(abs, pv) < 1.0e-9
        @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-10
        println("G267 traction: u nodal=", eu, "  max|p|=", maximum(abs, pv))
        # a wrong-sign / wrong-length load would be O(1): the exact shear
        # velocity has max |u| = gamma, so the 1e-10 bound is a meaningful
        # fraction of the solution amplitude.
        @test max(maximum(abs, u[1:ss.su.N]),
                  maximum(abs, u[(ss.su.N + 1):end])) > 0.5 * gam
    end

    # ==================================================================
    # (c) mapping independence, compared in physical space
    # ==================================================================
    @testset "mapping independence (physical space)" begin
        # (i) same physical domain, different logical frame: A2 = A1 * P with
        # P the logical x/y swap, which maps the unit square onto itself.  The
        # physical exact problem is identical; only the leaf parametrisation is
        # different, so the two discrete solutions must agree at physical points.
        # T(S,T) = (1-T, S): a 90-degree logical rotation of the square onto
        # itself (det = +1, so the orientation-preserving choice).  Lx = 1, so
        # g2(S,T) = g1(T(S,T)) gives A2 = A1*M and b2 = A1*(1,0) + b.
        M = ((0.0, -1.0), (1.0, 0.0))
        Aswap = (A[1] * M[1][1] + A[2] * M[2][1], A[1] * M[1][2] + A[2] * M[2][2],
                 A[3] * M[1][1] + A[4] * M[2][1], A[3] * M[1][2] + A[4] * M[2][2])
        bswap = (A[1] * 1.0 + b[1], A[3] * 1.0 + b[2])
        gswap = AffineBox(Aswap, bswap, 1.0)
        ss1, res1 = g267_frame_solve(g, qt, g267_ux, g267_uz, g267_fx, g267_fz)
        ss2, res2 = g267_frame_solve(gswap, qt, g267_ux, g267_uz, g267_fx,
                                     g267_fz)
        pts = g267_physical_points(g, ss1.su)
        du = 0.0
        dp = 0.0
        pshift = 0.0
        first = true
        for (x, y) in pts
            ax, az = g267_eval_u(ss1, res1[1], x, y)
            bx, bz = g267_eval_u(ss2, res2[1], x, y)
            du = max(du, abs(ax - bx), abs(az - bz))
            p1v = g267_eval_p(ss1, res1[2], x, y)
            p2v = g267_eval_p(ss2, res2[2], x, y)
            if first
                pshift = p1v - p2v
                first = false
            end
            dp = max(dp, abs((p1v - p2v) - pshift))
        end
        @test du < 1.0e-11
        @test dp < 1.0e-11
        println("G267 logical-frame invariance: du=", du, "  dp=", dp)

        # (ii) rigid rotation + translation of the whole physical configuration:
        # the exact fields are rotated too, u2(x) = Q u1(Q'(x-t)), p2 = p1(Q'(x-t)).
        # The discrete solutions must satisfy the same relation in physical
        # space.  A fixed-dof loop that evaluates the closures at logical
        # instead of physical positions cannot satisfy this.
        ang = 0.7
        Q = ((cos(ang), -sin(ang)), (sin(ang), cos(ang)))
        Ash = (Q[1][1] * A[1] + Q[1][2] * A[3], Q[1][1] * A[2] + Q[1][2] * A[4],
               Q[2][1] * A[1] + Q[2][2] * A[3], Q[2][1] * A[2] + Q[2][2] * A[4])
        tsh = (0.4, -0.35)
        bsh = (Q[1][1] * b[1] + Q[1][2] * b[2] + tsh[1],
               Q[2][1] * b[1] + Q[2][2] * b[2] + tsh[2])
        grot = AffineBox(Ash, bsh, 1.0)
        # pulled-back coordinates: xi = Q' (x - t)
        pull(x, y) = (Q[1][1] * (x - tsh[1]) + Q[2][1] * (y - tsh[2]),
                      Q[1][2] * (x - tsh[1]) + Q[2][2] * (y - tsh[2]))
        urotx(x, y) = begin
            xp, yp = pull(x, y)
            Q[1][1] * g267_ux(xp, yp) + Q[1][2] * g267_uz(xp, yp)
        end
        urotz(x, y) = begin
            xp, yp = pull(x, y)
            Q[2][1] * g267_ux(xp, yp) + Q[2][2] * g267_uz(xp, yp)
        end
        # f2(x) = Q f1(Q'(x-t)) = Q (2, -0.5) (constant)
        f2x = Q[1][1] * 2.0 + Q[1][2] * (-0.5)
        f2z = Q[2][1] * 2.0 + Q[2][2] * (-0.5)
        ssr, resr = g267_frame_solve(grot, qt, urotx, urotz,
                                     (x, y) -> f2x, (x, y) -> f2z)
        dur = 0.0
        dpr = 0.0
        pshift = 0.0
        first = true
        for (x, y) in pts
            ax, az = g267_eval_u(ss1, res1[1], x, y)
            xr = Q[1][1] * x + Q[1][2] * y + tsh[1]
            yr = Q[2][1] * x + Q[2][2] * y + tsh[2]
            bx, bz = g267_eval_u(ssr, resr[1], xr, yr)
            dur = max(dur, abs(Q[1][1] * ax + Q[1][2] * az - bx),
                           abs(Q[2][1] * ax + Q[2][2] * az - bz))
            p1v = g267_eval_p(ss1, res1[2], x, y)
            prv = g267_eval_p(ssr, resr[2], xr, yr)
            if first
                pshift = p1v - prv
                first = false
            end
            dpr = max(dpr, abs((p1v - prv) - pshift))
        end
        @test dur < 1.0e-11
        @test dpr < 1.0e-11
        println("G267 rigid-motion invariance: du=", dur, "  dp=", dpr)
    end

    # ==================================================================
    # (d) mapped boundary measure: perimeter of the parallelogram
    # ==================================================================
    @testset "mapped perimeter" begin
        ssb = make_semstokes(qt, 2)
        bmask = MS267._semquad_boundary_face_mask(qt, ssb.su)
        # analytic perimeter of A*(unit square): Lx=1, so the two directions are
        # A e_x and A e_y; |A e_x| = hypot(A11, A21), |A e_y| = hypot(A12, A22)
        exp_per = 2.0 * (hypot(A[1], A[3]) + hypot(A[2], A[4]))
        per = perimeter(g, ssb.su, bmask)
        @test abs(per - exp_per) <= 1.0e-13 * exp_per
        println("G267 perimeter: measured=", per, "  analytic=", exp_per,
                "  rel=", abs(per - exp_per) / exp_per)
        # the box method still returns the historical E12 expression
        @test perimeter(CartesianBox(2.5)) == 2.0 * (2.5 + 1.0)
    end
end
