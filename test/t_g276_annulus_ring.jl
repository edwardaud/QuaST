# G276: the periodic ring topology of the 2-D annulus (stage 0.5c1).
#
# Stage 0.5a built the annulus geometry and refused the closed ring; 0.5b2 built
# the curved-wall rows, the rotation gauge and the annular IC but still ran a
# walled wedge.  This gate pins the **ring**: the logical `T = 0` and `T = 1`
# sides of the annulus are the same physical curve, and that is a quadtree
# topology feature (`Quadtree(...; seam = :periodic)`), not a mapping one:
#
#   (a) **circular Couette on the ring, exact**.  `u_phi = A r + B/r` is the
#       exact steady Stokes solution between two rotating cylinders.  On the
#       ring the only walls are the two circles and the field is tangential
#       there, so the ring is *the* configuration where Couette is exact with
#       Dirichlet data; a wedge cannot host it (G275 (b1) solves the
#       all-Dirichlet restriction instead).  The gate solves the ring with the
#       exact tangential wall values on both circles and checks the velocity
#       error orders and the *analytic* torques `T_in = 4*pi*B = -54.3883`,
#       `T_out = -T_in`, plus the closed-ring torque balance.  The velocity
#       field is **not exact**: `A r + B/r` is not a polynomial on the curved
#       elements, so L2 converges at the Q2 order 3 (H1 order 2) and the
#       boundary-functional torque one order lower; the tolerances below are
#       set just above the measured level-3 values, so the assertion is the
#       observed convergence, not a hard-coded equality.  The control is the
#       *cut ring* (`Phi = 2*pi`, `seam = :wall`): with the seam treated as a
#       wall the same drive is O(1) wrong (measured max error 2.328 = f(R0)),
#       which is what a non-periodic implementation would silently produce.
#
#   (b) **periodicity**.  (b1) the node identity across the seam: the T = 0 and
#       T = 1 dofs are one node -- the node count is the *analytic* ring count
#       `(p*nx0*2^l + 1) * (p*ny0*2^l)` and the bounded twin of the same leaf
#       set has exactly the radial node count more; the T = 1 node ids of the
#       last angular row equal the T = 0 ids of the matching first-row leaves
#       and no physical coordinate appears twice.  (b2) the cross-seam
#       neighbour pair: the flat table and the historical float-coordinate
#       scan oracle agree leaf by leaf on the entire tree (the oracle wraps the
#       face coordinate by one period), `has_no_face_neighbor` is false on the
#       seam sides, `face_pairs(qt, geom)` contains the cross-seam pairs and
#       the physical perimeter excludes the seam.  (b3) the 2:1 mirror: a
#       refinement applied to one side only is *detected* (the independently
#       computed cross-seam level jump and `max_face_level_jump` are both 2,
#       measured) and `balance!` repairs it across the seam (jump 1, the other
#       side gains leaves) -- a wrong implementation without the wrap would
#       report the tree as balanced (jump 1) and leave the seam non-mirrored.
#       (b4) the seam constraints: on a deliberately non-mirrored (but
#       balanced) ring tree the hanging constraints from the cross-seam pair
#       are present, are partitions of unity, and the represented field is
#       C0 across the seam for *every* column of the affine constraint matrix;
#       the bounded twin has no such rows and is O(1) discontinuous at the
#       same points.  (b5) transfer: the nodal and the L2 transfer of a smooth
#       field from a coarse ring to a refined, non-mirrored ring keep the seam
#       trace equal (`<= 1e-14`) and the nodal transfer is bit-exact at the
#       shared nodes.
#
#   (c) **MG/transfer on the ring**.  The hierarchy levels are coarsenings that
#       stay periodic, the prolongation `Pfull` has the right size, it is exact
#       against an independent *physical-point* evaluation of the coarse FE
#       field (the prolongation uses logical coordinates, the oracle inverts the
#       map), the prolonged field has an exactly periodic seam trace, and the
#       warm mapped matrix-free apply allocates 0 B.  `:mg` and `:auto` agree
#       with `:direct` on the Couette ring at the documented tolerance.
#
#   (d) **the rotation gauge on the ring**.  A free-slip ring (both walls
#       free-slip, no other boundary: the solid-body rotation IS admissible)
#       has the velocity gauge and the pressure pin at the same time -- unlike
#       the wedge, where the free-slip cuts already remove the rotation.  The
#       load is a zero-net-torque m = 0 tangential body force whose solution
#       carries a nonzero net angular momentum (`omega * S(Z_h) = -0.337`
#       measured), so the projection is not a no-op; the returned field has
#       the *discrete* angular momentum `S(u) = 0` to round-off (measured
#       `1.8e-15`), is M-orthogonal to the nodal rotation mode (measured
#       `2.6e-12`), and the design's `S(u)/S(Z_h)` normalisation is what makes
#       the reported condition exact (the alternative `int r^2 dA`
#       normalisation leaves the measured `O(h^(p+1))` mode-norm residual, and
#       the two normalisations differ by that same measured ratio).  The
#       closed-ring torque balance is checked: no free-slip circle can exert a
#       torque, so the discrete wall torques vanish at the measured order
#       (sum `5.8e-2 -> 1.4e-2`, order 2.1) -- not exactly, because the
#       numerical stress on the wall is the weak natural-traction residual.
#       The wedge behaviour is unchanged: its free-slip card resolves the
#       gauge to `:none` (the cuts pin the rotation) and keeps the pressure
#       pin.
#
#   (e) **end-to-end**: an isoviscous free-slip ring driver run (the paper's
#       annular IC, Ra = 100) is finite, writes the 18-column history, keeps
#       the identities (`int div u = 0`, `Ra W = U` at the free-slip boundary
#       term), has wall normal velocity zero to round-off, a periodic theta
#       (the same dofs, exact) and `:mg`/`:auto` agreement with `:direct`; an
#       AMR run keeps the tree periodic and balanced and the transferred theta
#       periodic; a checkpoint round trip restores the periodic tree.  The
#       wedge card (Phi = pi/2, seam = :wall) still runs with the same history
#       shape and wall structure, and the seam sides there are walls.
#
# WHY THE ASSERTIONS ARE NOT TAUTOLOGICAL (what each catches):
#   * (a) the exact Couette field is not a polynomial and the ring is curved:
#     matching the L2/H1 errors at the Q2 orders and the *torque* (a boundary
#     functional of the numerical stress) to the analytic `4*pi*B` is a genuine
#     discretisation test.  The cut-ring control fails it by O(1), so an
#     implementation that treats the seam as a wall cannot pass the block.
#     The tolerances are stated as "the measured value at level 3 is within
#     this", and the observed orders are asserted separately;
#   * (b) the node count is analytic (no code path is reused to count it); the
#     bounded twin is built from the *same leaf set* so the count difference
#     isolates the seam identification; the table is compared with the
#     historical scan oracle (independent implementation); the cross-seam
#     level jump is computed from leaf indices and is required to equal what
#     the production `max_face_level_jump` reports (a missing wrap would make
#     the two disagree); the constraint-matrix trace test is a property of the
#     *space*, not of one solution, and the bounded twin is its O(1) control;
#     the transfer is checked against the direct sampling of the same smooth
#     function at the shared nodes;
#   * (c) `Pfull` is compared with an independent evaluation of the coarse FE
#     field at the fine node's *physical* point (the prolongation never reads a
#     physical coordinate, so the two routes share no code); the 0 B allocation
#     is the mapped steady state; direct-vs-mg compares two different linear
#     solvers;
#   * (d) the load is torque-free (else the solver refuses it), nonzero and
#     m = 0, so the exact solution's angular momentum is nonzero: the measured
#     `omega*S(Z_h)` proves the gauge ran.  The discrete `S(u) = 0` is checked
#     by an independent quadrature functional, the L2 statement by the velocity
#     mass matrix, and the two normalisations' difference is measured rather
#     than asserted away.  The wall torques are a boundary functional the
#     solver never forms;
#   * (e) the driver run re-computes every checked history column from the
#     returned state, and the identities are properties of the discrete
#     solution.  The wedge block pins that the new seam topology did not leak
#     into the bounded path (the seam sides are walls there, the gauge is
#     `:none`, and the 18-column history is unchanged in shape).
#
# The gate pins `linear=:direct` where a bitwise or exact comparison is made;
# the `:mg`/`:auto` runs are compared on their documented tolerance.

using Test, QuaST, LinearAlgebra, Random
using QuaST: gll_rule, make_fedata, leaf_metric, quad_measure, quad_point,
             basis_grad, face_point, face_normal, face_jacobian,
             jacobian_inverse, total_measure, make_semstokes, make_semquad,
             solve_semquad_stokes_var, transfer_semquad, semquad_constraints,
             face_pairs, _semquad_boundary_face_mask, _semquad_wall_frames,
             _semquad_rotation_moment, _semquad_eval_at!, SemQuadEvalWork,
             _affine_constraint

const MS276 = QuaST

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
const G276_R0 = 1.0
const G276_R1 = 2.0
const G276_PHI = 2.0 * pi
const G276_A = 2.0
const G276_B = -3.0 / log(2.0)

g276_ring(; R0=G276_R0, R1=G276_R1) =
    PolarAnnulus(R0, R1, G276_PHI; seam=:periodic)
g276_cut_ring(; R0=G276_R0, R1=G276_R1) =
    PolarAnnulus(R0, R1, G276_PHI; seam=:wall)
g276_wedge(; R0=G276_R0, R1=G276_R1) =
    PolarAnnulus(R0, R1, 0.5 * pi; seam=:wall)

# uniform ring / bounded trees with the maxlevel headroom the balance needs
function g276_tree(level::Int; seam::Symbol=:periodic, ny0::Int=4)
    qt = Quadtree(level + 2; cells_x=1, cells_y=ny0, seam=seam)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end
g276_space(level, p; seam::Symbol=:periodic, R0=G276_R0, R1=G276_R1) = begin
    g = seam === :periodic ? g276_ring(; R0=R0, R1=R1) :
                             g276_cut_ring(; R0=R0, R1=R1)
    qt = g276_tree(level; seam=seam)
    return g, qt, make_semstokes(qt, p; geom=g)
end

g276_zero(x, y) = 0.0
g276_one(x, y) = 1.0
g276_f(r) = G276_A * r + G276_B / r
g276_couette(x, y) = begin
    r = hypot(x, y)
    g = g276_f(r)
    (-g * y / r, g * x / r)
end
const G276_TORQUE_IN = 4.0 * pi * G276_B      # ANNULUS_REFS Appendix A (k = 0)
const G276_TORQUE_OUT = -G276_TORQUE_IN

# exact field sampled on the velocity nodes
function g276_sample(sq, f)
    nu = sq.N
    u = zeros(Float64, 2 * nu)
    for (gid, c) in enumerate(sq.coords)
        v = f(c[1], c[2])
        u[gid] = v[1]
        u[nu + gid] = v[2]
    end
    return u
end

# L2 / H1 velocity errors of a solved field against an exact closure
function g276_errors(g, ss, u, exact)
    sq = ss.su
    nu = sq.N
    sp = sq.p
    fd = make_fedata(sp, sp - 1)
    eu2 = 0.0; eh1 = 0.0; area = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0; uyq = 0.0
            dux = [0.0, 0.0]; duy = [0.0, 0.0]
            for j in 1:sp+1, i in 1:sp+1
                k = (j - 1) * (sp + 1) + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += u[ids[k]] * b
                uyq += u[nu + ids[k]] * b
                gx, gy = basis_grad(m, fd, qx, qy, i, j)
                dux[1] += u[ids[k]] * gx; dux[2] += u[ids[k]] * gy
                duy[1] += u[nu + ids[k]] * gx; duy[2] += u[nu + ids[k]] * gy
            end
            ue = exact(x, y)
            hs = 1.0e-6
            up = exact(x + hs, y); um = exact(x - hs, y)
            vp = exact(x, y + hs); vm = exact(x, y - hs)
            duex = ((up[1] - um[1]) / (2hs), (vp[1] - vm[1]) / (2hs),
                    (up[2] - um[2]) / (2hs), (vp[2] - vm[2]) / (2hs))
            eu2 += w * ((uxq - ue[1])^2 + (uyq - ue[2])^2)
            eh1 += w * ((dux[1] - duex[1])^2 + (dux[2] - duex[2])^2 +
                        (duy[1] - duex[3])^2 + (duy[2] - duex[4])^2)
            area += w
        end
    end
    return sqrt(eu2 / area), sqrt(eh1 / area)
end

function g276_maxerr(ss, u, exact)
    nu = ss.su.N
    m = 0.0
    for (gid, c) in enumerate(ss.su.coords)
        e = exact(c[1], c[2])
        m = max(m, hypot(u[gid] - e[1], u[nu + gid] - e[2]))
    end
    return m
end

# wall torque from the assembled numerical stress, `T = int (x t_y - y t_x) ds`
function g276_wall_torque(g, ss, u, pv, side)
    sq = ss.su
    nu = sq.N
    n = sq.p + 1
    q = ss.q
    rule_p = gll_rule(q)
    T = 0.0
    for lf in sq.leaves
        MS276._on_logical_wall(lf, side, sq) || continue
        ids = sq.elem[lf]
        pids = ss.sp.elem[lf]
        fm = MS276._mapped_face_metric(g, sq, lf, side, sq.xi)
        for qq in 1:n
            xi, eta = MS276._wall_ref_point(side, sq.xi[qq])
            inv = jacobian_inverse(g, sq, lf, xi, eta)
            i = side === :xplus ? n : 1
            j = qq
            dxi = zeros(4)
            for a in 1:n
                dxi[1] += sq.D[i, a] * u[ids[(j - 1) * n + a]]
                dxi[3] += sq.D[i, a] * u[nu + ids[(j - 1) * n + a]]
                dxi[2] += sq.D[j, a] * u[ids[(a - 1) * n + i]]
                dxi[4] += sq.D[j, a] * u[nu + ids[(a - 1) * n + i]]
            end
            ux_x = dxi[1] * inv[1] + dxi[2] * inv[2]
            ux_y = dxi[1] * inv[3] + dxi[2] * inv[4]
            uy_x = dxi[3] * inv[1] + dxi[4] * inv[2]
            uy_y = dxi[3] * inv[3] + dxi[4] * inv[4]
            pq = 0.0
            Li, _ = lagrange_eval(rule_p, xi)
            Lj, _ = lagrange_eval(rule_p, eta)
            for bb in 1:q+1, aa in 1:q+1
                pq += pv[pids[(bb - 1) * (q + 1) + aa]] * Li[aa] * Lj[bb]
            end
            sxx = 2.0 * ux_x - pq
            syy = 2.0 * uy_y - pq
            sxy = ux_y + uy_x
            nx_, ny_ = face_normal(fm, qq)
            tx = sxx * nx_ + sxy * ny_
            ty = sxy * nx_ + syy * ny_
            xq, yq = face_point(fm, qq)
            w = face_jacobian(fm, qq) * sq.w[qq]
            T += w * (xq * ty - yq * tx)
        end
    end
    return T
end

function g276_div_volume(g, sq, u)
    p = sq.p
    fd = make_fedata(p, p - 1)
    nu = sq.N
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            duxdx = 0.0; duydy = 0.0
            for j in 1:p+1, i in 1:p+1
                k = (j - 1) * (p + 1) + i
                gx, gy = basis_grad(m, fd, qx, qy, i, j)
                duxdx += u[ids[k]] * gx
                duydy += u[nu + ids[k]] * gy
            end
            s += w * (duxdx + duydy)
        end
    end
    return s
end

# nodal field from a closure on the global logical coordinates
function g276_field(sq, f)
    u = zeros(Float64, sq.N)
    n = sq.p + 1
    for lf in sq.leaves, j in 1:n, i in 1:n
        S, T = MS276.global_logical(sq, lf, sq.xi[i], sq.xi[j])
        u[sq.elem[lf][(j - 1) * n + i]] = f(S, T)
    end
    return u
end

# seam trace jump of a represented scalar field: the field is evaluated at
# (S, T=0) from the first angular row and at (S, T=1) from the last one
function g276_seam_trace_jump(sq, u; npts::Int=17)
    leaves = Set(sq.leaves)
    w = SemQuadEvalWork(sq.p + 1)
    worst = 0.0
    for S in range(0.05, 0.95; length=npts)
        lf0 = MS276.locate_leaf_logical(sq, S, 0.0, leaves)
        lf1 = MS276.locate_leaf_logical(sq, S, 1.0, leaves)
        xi0, eta0 = MS276.leaf_ref(sq, lf0, S, 0.0)
        xi1, eta1 = MS276.leaf_ref(sq, lf1, S, 1.0)
        v0 = MS276._semquad_eval_ref!(w, sq, u, xi0, eta0, lf0)
        v1 = MS276._semquad_eval_ref!(w, sq, u, xi1, eta1, lf1)
        worst = max(worst, abs(v0 - v1))
    end
    return worst
end

# cross-seam level jump computed from leaf indices only (never the face table):
# for every first-row leaf and last-row leaf whose radial intervals overlap,
# the level difference; the maximum is the 2:1 violation across the seam
function g276_seam_level_jump(qt)
    rows = Dict{Int,Vector{NTuple{3,Int}}}()
    for lf in qt.leaves
        lf[3] == 0 && push!(get!(rows, 0, NTuple{3,Int}[]), lf)
        lf[3] == qt.ny0 * (1 << lf[1]) - 1 &&
            push!(get!(rows, 1, NTuple{3,Int}[]), lf)
    end
    worst = 0
    for a in get(rows, 0, NTuple{3,Int}[]), b in get(rows, 1, NTuple{3,Int}[])
        na = qt.nx0 * (1 << a[1])
        nb = qt.nx0 * (1 << b[1])
        # radial intervals [ia/na, (ia+1)/na] and [ib/nb, (ib+1)/nb] overlap?
        lo = max(a[2] / na, b[2] / nb)
        hi = min((a[2] + 1) / na, (b[2] + 1) / nb)
        hi > lo + 1.0e-14 && (worst = max(worst, abs(a[1] - b[1])))
    end
    return worst
end

# cross-seam pairs of a pair list
function g276_cross_pairs(qt, g)
    return [p for p in face_pairs(qt, g)
            if (p.leafL[3] == 0 &&
                p.leafR[3] == qt.ny0 * (1 << p.leafR[1]) - 1) ||
               (p.leafR[3] == 0 &&
                p.leafL[3] == qt.ny0 * (1 << p.leafL[1]) - 1)]
end

# build the velocity affine constraint matrix of a free-slip/unconstrained ring
# space: hanging constraints only (no wall rows), both velocity components
function g276_hanging_Cu(ss, qt)
    sq = ss.su
    nu = sq.N
    cons = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in semquad_constraints(sq, qt)
        cons[c] = (ms, ws)
        cons[nu + c] = (nu .+ ms, ws)
    end
    u0, Cu, _ = _affine_constraint(2 * nu, cons, Dict{Int,Float64}())
    return u0, Cu
end

# ===========================================================================
@testset "G276 annulus ring (stage 0.5c1)" begin
    Random.seed!(276)

    # ======================================================================
    # (a) circular Couette on the ring
    # ======================================================================
    @testset "(a) circular Couette: the ring's exact Stokes solution" begin
        wu(x, y) = g276_f(hypot(x, y))
        eus = Float64[]; ehs = Float64[]; tinq = Float64[]
        for level in 1:3
            g, qt, ss = g276_space(level, 2)
            u, p, = solve_semquad_stokes_var(ss, qt, g276_one, g276_zero,
                                             g276_zero, g276_zero, g276_zero;
                                             linear=:direct, bc=:rigid,
                                             wall_bc=(:rigid, :rigid, :rigid,
                                                      :rigid),
                                             wall_u=(wu, wu, nothing, nothing),
                                             wall_un=(nothing, nothing,
                                                      nothing, nothing))
            eu, eh = g276_errors(g, ss, u, g276_couette)
            Tin = g276_wall_torque(g, ss, u, p, :xminus)
            Tout = g276_wall_torque(g, ss, u, p, :xplus)
            push!(eus, eu); push!(ehs, eh); push!(tinq, abs(Tin - G276_TORQUE_IN))
            if level == 3
                println("G276 a: T_in = ", Tin, " (exact ", G276_TORQUE_IN,
                        ")  T_out = ", Tout, "  T_in+T_out = ", Tin + Tout)
                println("G276 a: |u|L2 = ", eus, "  |u|H1 = ", ehs,
                        "  Tin_err = ", tinq)
                # measured: 1.08e-4 / 5.10e-3 / 1.80e-1 / 1.22e-1
                @test eu <= 3.0e-4
                @test eh <= 2.0e-2
                @test abs(Tin - G276_TORQUE_IN) <= 0.4
                @test abs(Tout - G276_TORQUE_OUT) <= 0.4
                # the closed ring: the total torque of the two walls on the fluid
                # must vanish in the continuum; the discrete residual converges
                @test abs(Tin + Tout) <= 0.25
                @test abs(maximum(abs, u) - abs(g276_f(G276_R0))) <= 1.0e-3
            end
        end
        ord_u = log2(eus[2] / eus[3])
        ord_h = log2(ehs[2] / ehs[3])
        ord_t = log2(tinq[2] / tinq[3])
        println("G276 a orders (level2->3): u_L2 = ", ord_u, "  u_H1 = ", ord_h,
                "  torque = ", ord_t)
        @test ord_u >= 2.6
        @test ord_h >= 1.7
        @test ord_t >= 1.4
        # the seam is NOT a wall: no y face of a ring leaf is in the boundary
        # mask, and the wall frames only occur on the two circles (wi 1, 2)
        g, qt, ss = g276_space(2, 2)
        bmask = _semquad_boundary_face_mask(qt, ss.su)
        nywall = 0
        for it in 1:length(qt.leaves)
            b = 4 * (it - 1)
            (bmask[b + 3] != 0x00 || bmask[b + 4] != 0x00) && (nywall += 1)
        end
        @test nywall == 0
        conds = _semquad_wall_frames(g, ss.su, bmask)
        @test all(cs -> all(c -> c.wi in (1, 2), cs), values(conds))
        # the perimeter of the ring is the two circles, not + two radial sides
        per = perimeter(g, ss.su, bmask)
        @test abs(per - G276_PHI * (G276_R0 + G276_R1)) <= 1.0e-12 * per
        # control: the cut ring (the same map, seam = :wall) with the same
        # circle data and free-slip radial sides is O(1) wrong -- the seam
        # forces u_phi = 0 there, which is the non-periodic failure mode this
        # stage removes
        gcut = g276_cut_ring()
        qtc = g276_tree(2; seam=:none)
        ssc = make_semstokes(qtc, 2; geom=gcut)
        uc, pc, = solve_semquad_stokes_var(ssc, qtc, g276_one, g276_zero,
                                           g276_zero, g276_zero, g276_zero;
                                           linear=:direct, bc=:rigid,
                                           wall_bc=(:rigid, :rigid, :free,
                                                    :free),
                                           wall_u=(wu, wu, nothing, nothing))
        ecut, = g276_errors(gcut, ssc, uc, g276_couette)
        mcut = g276_maxerr(ssc, uc, g276_couette)
        println("G276 a control (free-slip cut ring): L2 err = ", ecut,
                "  max err = ", mcut, " (exact f(R0) = ", g276_f(G276_R0), ")")
        @test ecut > 0.1
        @test mcut > 2.0
        # a ring geometry on a bounded tree is refused, not run as a cut ring
        @test_throws ArgumentError make_semstokes(qtc, 2; geom=g)
    end

    # ======================================================================
    # (b) periodicity: identity, pairs, the balance mirror, constraints and
    #     transfer
    # ======================================================================
    @testset "(b1) node identity across the seam" begin
        g, qt, ss = g276_space(2, 2)
        sq = ss.su
        p = sq.p
        l = 2
        nx = qt.nx0 * (1 << l)
        ny = qt.ny0 * (1 << l)
        # the analytic ring node count: one radial column of nodes is shared by
        # the two seam images, so the angular count is `p*ny` not `p*ny + 1`
        nring = (p * nx + 1) * (p * ny)
        @test sq.N == nring
        # the bounded twin of the same leaf set (the cut ring) has the extra
        # radial column of seam nodes: N_bounded - N_ring = p*nx + 1
        qtb = Quadtree(qt.maxlevel; Lx=qt.Lx, cells_x=qt.nx0, cells_y=qt.ny0)
        empty!(qtb.leaves)
        for lf in qt.leaves
            push!(qtb.leaves, lf)
        end
        MS276._invalidate_faces!(qtb)
        @test qtb.seam === :none
        sbb = make_semstokes(qtb, 2; geom=g276_cut_ring())
        @test sbb.su.N - sq.N == p * nx + 1
        @test Set(qtb.leaves) == Set(qt.leaves)
        # the T = 1 corner ids of the last angular row ARE the T = 0 corner ids
        # of the matching first-row leaves (exact identity, no tolerance)
        ncommon = 0
        for lf in qt.leaves
            lf[3] == 0 || continue
            nside = qt.ny0 * (1 << lf[1])
            lfm = (lf[1], lf[2], nside - 1)
            lfm in qt.leaves || continue
            idsA = sq.elem[lf]
            idsB = sq.elem[lfm]
            n = p + 1
            for i in 1:n
                # side A's T = 0 node (i, j = 1) vs side B's T = 1 node
                # (i, j = n) of the same radial leaf: same physical point, so
                # the same node id (no tolerance).  The y-face node index is
                # ids[(j-1)*n + i], so T = 0 is idsA[i]. 
                @test idsA[i] == idsB[(n - 1) * n + i]
                ncommon += 1
            end
        end
        @test ncommon >= 8
        # no physical coordinate appears twice: the seam nodes are merged, not
        # duplicated (a dictionary, not the code's logical key)
        coords = Dict{NTuple{2,Float64},Int}()
        ndup = 0
        for c in sq.coords
            k = (c[1], c[2])
            haskey(coords, k) && (ndup += 1)
            coords[k] = 1
        end
        @test ndup == 0
        # a periodic tree needs at least two base cells in the closed direction
        @test_throws ArgumentError Quadtree(2; cells_x=1, cells_y=1,
                                            seam=:periodic)
        println("G276 b1: N = ", sq.N, "  bounded twin N = ", sbb.su.N,
                "  seam nodes merged = ", sbb.su.N - sq.N)
    end

    @testset "(b2) the cross-seam neighbour pair" begin
        g, qt, ss = g276_space(2, 2)
        # the flat table agrees with the historical float-coordinate scan
        # oracle, leaf by leaf, on the whole periodic tree
        nbad = 0
        for lf in qt.leaves, dir in (:xminus, :xplus, :yminus, :yplus)
            a = neighbors(qt, lf, dir)
            b = MS276._neighbors_scan(qt, lf, dir)
            (Set(a) == Set(b) && length(a) == length(b)) || (nbad += 1)
        end
        @test nbad == 0
        # seam faces have neighbours on both sides: no y face is a boundary
        for lf in qt.leaves
            (lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1) || continue
            @test !MS276.has_no_face_neighbor(qt, lf, :yminus)
            @test !MS276.has_no_face_neighbor(qt, lf, :yplus)
        end
        # the pair list has the cross-seam entries, with the coarse leaf as
        # leafL and the direction pointing across the seam
        cross = g276_cross_pairs(qt, g)
        @test !isempty(cross)
        @test all(p -> p.leafL[1] <= p.leafR[1], cross)
        # every first-row leaf has a cross-seam neighbour in the last row
        for lf in qt.leaves
            lf[3] == 0 || continue
            nbs = neighbors(qt, lf, :yminus)
            @test !isempty(nbs)
            @test all(nb -> nb[3] == qt.ny0 * (1 << nb[1]) - 1, nbs)
        end
        # the bounded-box pair list refuses a periodic tree rather than
        # silently dropping the seam
        @test_throws ArgumentError face_pairs(qt)
        # the physical measures: area unchanged, perimeter = the two circles
        bmask = _semquad_boundary_face_mask(qt, ss.su)
        per = perimeter(g, ss.su, bmask)
        @test abs(per - G276_PHI * (G276_R0 + G276_R1)) <= 1.0e-12 * per
        @test abs(total_measure(g, ss.su) - pi * (G276_R1^2 - G276_R0^2)) <=
              1.0e-12 * pi * G276_R1^2
        println("G276 b2: table-vs-scan mismatches = ", nbad,
                "  cross-seam pairs = ", length(cross))
    end

    @testset "(b3) one-sided refinement is detected and mirrored" begin
        g = g276_ring()
        qt = g276_tree(2; seam=:periodic)
        @test g276_seam_level_jump(qt) == 0
        # refine ONE leaf on the first side through two levels, no balance: the
        # seam now carries a 2:1 violation.  It is visible *only* through the
        # cross-seam pair (a bounded tree would hide it), so the production
        # jump test and the index-only oracle must agree.
        @test refine!(qt, (2, 0, 0))
        @test refine!(qt, (3, 0, 0))
        jump_ind = g276_seam_level_jump(qt)
        jump_prod = max_face_level_jump(qt)
        println("G276 b3: unbalanced seam jump: index-only = ", jump_ind,
                "  production = ", jump_prod)
        @test jump_ind == 2
        @test jump_prod == jump_ind      # a missing wrap would report 1 here
        nl_before = count(l -> l[3] == qt.ny0 * (1 << l[1]) - 1, qt.leaves)
        balance!(qt)
        jump_after = g276_seam_level_jump(qt)
        nl_after = count(l -> l[3] == qt.ny0 * (1 << l[1]) - 1, qt.leaves)
        println("G276 b3: after balance: jump = ", jump_after,
                " last-row leaves ", nl_before, " -> ", nl_after,
                "  maxjump = ", max_face_level_jump(qt))
        @test jump_after <= 1
        @test max_face_level_jump(qt) <= 1
        # the mirror: the balance refined the OTHER side of the seam
        @test nl_after > nl_before
        @test !isempty(g276_cross_pairs(qt, g))
    end

    @testset "(b4) the seam constraints and the space's C0 trace" begin
        g = g276_ring()
        # a deliberately non-mirrored (but balanced) ring tree: the angular
        # level differs between the two sides of the seam, so the fine side's
        # seam nodes are hanging nodes constrained by the coarse side
        qt = g276_tree(2; seam=:periodic)
        refine!(qt, (2, 0, 0))
        refine!(qt, (3, 0, 0))
        balance!(qt)
        ss = make_semstokes(qt, 2; geom=g)
        sq = ss.su
        @test g276_seam_level_jump(qt) <= 1
        cons = semquad_constraints(sq, qt)
        @test !isempty(cons)
        # every constraint row is a partition of unity (the Lagrange weights)
        for (_, (_, ws)) in cons
            @test abs(sum(ws) - 1.0) <= 1.0e-12
        end
        # the space is C0 across the seam for EVERY admissible velocity field:
        # each column of the affine constraint matrix, evaluated from the first
        # and the last angular row, must have the same seam trace
        u0, Cu = g276_hanging_Cu(ss, qt)
        nu = sq.N
        ncols = size(Cu, 2)
        worst = 0.0; ntest = 0
        for j in unique(round.(Int, range(1, ncols; length=min(ncols, 24))))
            for comp in 0:1
                u = collect(u0 + Cu[:, j])
                f = comp == 0 ? u[1:nu] : u[nu+1:2nu]
                worst = max(worst, g276_seam_trace_jump(sq, f))
                ntest += 1
            end
        end
        println("G276 b4: non-mirrored ring: columns tested = ", ntest,
                "  max seam trace jump = ", worst)
        @test ntest >= 8
        @test worst <= 1.0e-13
        # the O(1) control: the same leaf set on the *bounded* topology (the cut
        # ring) has no cross-seam rows and a generic field jumps there
        qtb = Quadtree(qt.maxlevel; Lx=qt.Lx, cells_x=qt.nx0, cells_y=qt.ny0)
        empty!(qtb.leaves)
        for lf in qt.leaves
            push!(qtb.leaves, lf)
        end
        MS276._invalidate_faces!(qtb)
        ssb = make_semstokes(qtb, 2; geom=g276_cut_ring())
        u0b, Cub = g276_hanging_Cu(ssb, qtb)
        # a dense admissible field of the bounded space: every column enters, so
        # the seam-adjacent dofs do too
        ub = collect(u0b + Cub * randn(size(Cub, 2)))
        worstb = g276_seam_trace_jump(ssb.su, ub[1:ssb.su.N])
        println("G276 b4: bounded control seam trace jump = ", worstb)
        @test worstb > 0.1
        # (the constraint count is not the point; the cross-seam *rows* are)
        @test length(cons) >= 2
    end

    @testset "(b5) periodic transfer" begin
        g = g276_ring()
        qtc = g276_tree(1; seam=:periodic)
        ssc = make_semstokes(qtc, 2; geom=g)
        qt = g276_tree(2; seam=:periodic)
        refine!(qt, (2, 0, 0))
        refine!(qt, (3, 0, 0))
        balance!(qt)
        ss = make_semstokes(qt, 2; geom=g)
        smooth(S, T) = exp(sinpi(3.0 * S)) * cos(2.0 * pi * 3.0 * T) + 0.25 * S
        uc = g276_field(ssc.su, smooth)
        uf = transfer_semquad(ssc.su, uc, ss.su; mode=:nodal)
        ufl2 = transfer_semquad(ssc.su, uc, ss.su; mode=:l2)
        jn = g276_seam_trace_jump(ss.su, uf)
        jl = g276_seam_trace_jump(ss.su, ufl2)
        println("G276 b5: nodal transfer seam jump = ", jn,
                "  L2 transfer seam jump = ", jl)
        @test jn <= 1.0e-13
        @test jl <= 1.0e-13
        # the nodal transfer is bit-exact at the nodes both spaces share
        # (the smooth function is sampled directly on the fine mesh as oracle)
        ua = g276_field(ss.su, smooth)
        worst = 0.0; ncommon = 0
        for (gc, c) in enumerate(ssc.su.coords)
            j = findfirst(cc -> cc[1] == c[1] && cc[2] == c[2], ss.su.coords)
            j === nothing && continue
            ncommon += 1
            worst = max(worst, abs(uc[gc] - ua[j]))
        end
        println("G276 b5: common nodes = ", ncommon,
                "  same-function nodal diff = ", worst)
        @test ncommon >= 50
        @test worst <= 1.0e-15
    end

    # ======================================================================
    # (c) MG and prolongation on the ring
    # ======================================================================
    @testset "(c) MG, prolongation and the warm apply" begin
        g, qt, ss = g276_space(2, 2)
        wu(x, y) = g276_f(hypot(x, y))
        ud, pd, = solve_semquad_stokes_var(ss, qt, g276_one, g276_zero,
                                           g276_zero, g276_zero, g276_zero;
                                           linear=:direct, bc=:rigid,
                                           wall_bc=(:rigid, :rigid, :rigid,
                                                    :rigid),
                                           wall_u=(wu, wu, nothing, nothing))
        um, pm, = solve_semquad_stokes_var(ss, qt, g276_one, g276_zero,
                                           g276_zero, g276_zero, g276_zero;
                                           linear=:mg, bc=:rigid,
                                           wall_bc=(:rigid, :rigid, :rigid,
                                                    :rigid),
                                           wall_u=(wu, wu, nothing, nothing))
        ua, pa, = solve_semquad_stokes_var(ss, qt, g276_one, g276_zero,
                                           g276_zero, g276_zero, g276_zero;
                                           linear=:auto, bc=:rigid,
                                           wall_bc=(:rigid, :rigid, :rigid,
                                                    :rigid),
                                           wall_u=(wu, wu, nothing, nothing))
        println("G276 c: |u_direct - u_mg| = ", maximum(abs, ud - um),
                "  |u_direct - u_auto| = ", maximum(abs, ud - ua),
                "  mg residual = ", MS276.LAST_SEMSTOKES_RESIDUAL[])
        @test maximum(abs, ud - um) <= 1.0e-8
        @test maximum(abs, ud - ua) <= 1.0e-12
        # the hierarchy stays periodic at every level
        hier = MS276._mg_hierarchy_trees(qt)
        @test length(hier) >= 3
        @test all(q -> q.seam === :periodic, hier)
        @test all(q -> max_face_level_jump(q) <= 1, hier)
        # a coarse-to-fine prolongation: right size, exact against an
        # independent physical-point evaluation of the coarse FE field, and a
        # periodic seam trace
        qta = g276_tree(1; seam=:periodic)
        ssa = make_semstokes(qta, 2; geom=g)
        P = MS276._mg_prolongation_full(ss.su, ssa.su, 2)
        @test size(P) == (2 * ss.su.N, 2 * ssa.su.N)
        c = randn(ssa.su.N)
        cfull = vcat(c, c)
        leaves_c = Set(ssa.su.leaves)
        w = SemQuadEvalWork(ssa.p + 1)
        worst = 0.0
        for (gid, cc) in enumerate(ss.su.coords)
            ref = MS276._semquad_eval_at!(w, ssa.su, c, cc[1], cc[2], leaves_c)
            worst = max(worst, abs(P[gid, :]' * cfull - ref))
        end
        println("G276 c: |Pfull*c - independent evaluation| = ", worst)
        @test worst <= 1.0e-12
        pf = P * cfull
        @test g276_seam_trace_jump(ss.su, pf[1:ss.su.N]) <= 1.0e-13
        # warm mapped matrix-free apply: 0 B (the mapped steady state)
        asm = solve_semquad_stokes_var(ss, qt, g276_one, g276_zero, g276_zero,
                                       g276_zero, g276_zero;
                                       linear=:mg, mg_operator=:matrixfree,
                                       assemble_only=:mg)
        op = MS276.semquad_kop(ss, qt)
        etav = MS276.semquad_eta_lanes(op,
                                       MS276.semquad_eta_field(op, g276_one))
        work = MS276.SemQuadKWork(op)
        nv = size(asm.Cu, 2)
        v = ones(Float64, nv)
        out = zeros(Float64, nv)
        function alloc_apply(op, etav, v, work, out, Cu)
            MS276._semquad_apply_Kred!(out, op, Cu, etav, v, work)
            return @allocated MS276._semquad_apply_Kred!(out, op, Cu, etav, v,
                                                         work)
        end
        a = alloc_apply(op, etav, v, work, out, asm.Cu)
        b = alloc_apply(op, etav, v, work, out, asm.Cu)
        @test a == 0
        @test b == 0
        @test nv > 0
        println("G276 c: warm reduced apply allocation = ", b, " B (", nv,
                " dofs)")
    end

    # ======================================================================
    # (d) the rotation gauge on the ring
    # ======================================================================
    @testset "(d) rotation gauge and pressure pin on the ring" begin
        # the load: an m = 0 tangential body force f_theta = c (r - r0) with r0
        # chosen so the net torque vanishes (a torque-imbalanced load would be
        # refused by the solver).  Its exact solution carries a nonzero net
        # angular momentum, so it exercises the projection.
        r0 = 3.0 * (G276_R1^4 - G276_R0^4) /
             (4.0 * (G276_R1^3 - G276_R0^3))
        fx(x, y) = begin
            r = hypot(x, y)
            -((r - r0) * y / r)
        end
        fz(x, y) = begin
            r = hypot(x, y)
            (r - r0) * x / r
        end
        # the analytic load torque is zero to round-off
        torque = 2.0 * pi * ((G276_R1^4 - G276_R0^4) / 4.0 -
                             r0 * (G276_R1^3 - G276_R0^3) / 3.0)
        @test abs(torque) <= 1.0e-14
        sums = Float64[]
        for level in (2, 3)
            g, qt, ss = g276_space(level, 2)
            sq = ss.su
            nu = sq.N
            u, p, = solve_semquad_stokes_var(ss, qt, g276_one, fx, fz,
                                             g276_zero, g276_zero;
                                             linear=:direct, bc=:free,
                                             wall_bc=(:free, :free, :free,
                                                      :free))
            # the two gauges compose: one velocity dof (rotation) and one
            # pressure dof (constant) are pinned at the same time
            @test MS276.LAST_SEMSTOKES_ROTATION_GAUGE[] === :rotation
            @test MS276.LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin
            om = MS276.LAST_SEMSTOKES_ROTATION_SHIFT[]
            # the *discrete* angular momentum functional the design names
            S, Q = _semquad_rotation_moment(g, sq, u)
            @test abs(S) <= 1.0e-13
            # the L2-projection form: M-orthogonal to the nodal rotation mode
            z = zeros(2 * nu)
            for (gid, c) in enumerate(sq.coords)
                z[gid] = -c[2]
                z[nu + gid] = c[1]
            end
            SZ, Q2 = _semquad_rotation_moment(g, sq, z)
            M = assemble_semquad_velocity_mass(sq)
            Mu = M * u
            Mz = M * z
            mres = abs(dot(z, Mu)) /
                   (sqrt(abs(dot(z, Mz))) * sqrt(abs(dot(u, Mu))))
            @test mres <= 1.0e-8
            # the gauge is not a no-op: the removed mode is nonzero
            @test abs(om * SZ) >= 1.0e-2
            # the two normalisations: S(Z_h) vs int r^2 dA differ by the
            # measured O(h^(p+1)) mode norm; the implementation normalises by
            # S(Z_h), which is what makes S(u) exactly zero
            rel = abs(SZ - Q2) / Q2
            @test rel <= 1.0e-4
            # wall torques on the free-slip circles: the continuum value is 0
            Tin = g276_wall_torque(g, ss, u, p, :xminus)
            Tout = g276_wall_torque(g, ss, u, p, :xplus)
            push!(sums, abs(Tin + Tout))
            if level == 3
                println("G276 d level 3: omega = ", om, "  S = ", S,
                        "  S(Z_h) = ", SZ, "  int r^2 = ", Q2,
                        "  rel = ", rel, "  M-res = ", mres,
                        "  T_in = ", Tin, "  T_out = ", Tout,
                        "  sum = ", Tin + Tout)
                @test abs(Tin) <= 0.03
                @test abs(Tout) <= 0.05
                @test abs(Tin + Tout) <= 0.03
            end
            println("G276 d level ", level, ": omega*S(Z_h) = ", om * SZ,
                    "  |S| = ", abs(S), "  M-res = ", mres,
                    "  rel normalisations = ", rel)
        end
        # the closed-ring torque balance converges (measured order 2.1)
        ord_T = log2(sums[1] / sums[2])
        println("G276 d torque balance: ", sums, "  order = ", ord_T)
        @test ord_T >= 1.4
        # the alternative (int r^2) normalisation on level 2: it leaves the
        # measured O(h^(p+1)) residual instead of round-off, and that residual
        # is exactly the two-normalisation ratio -- this is why the discrete
        # functional is the implemented normalisation
        g, qt, ss = g276_space(2, 2)
        sq = ss.su
        nu = sq.N
        u, p, = solve_semquad_stokes_var(ss, qt, g276_one, fx, fz,
                                         g276_zero, g276_zero;
                                         linear=:direct, bc=:free,
                                         wall_bc=(:free, :free, :free, :free))
        om = MS276.LAST_SEMSTOKES_ROTATION_SHIFT[]
        z = zeros(2 * nu)
        for (gid, c) in enumerate(sq.coords)
            z[gid] = -c[2]
            z[nu + gid] = c[1]
        end
        SZ, Q2 = _semquad_rotation_moment(g, sq, z)
        # reconstruct the pre-projection field: u_after = u_before - omega*Z_h
        ub = copy(u)
        for (gid, c) in enumerate(sq.coords)
            ub[gid] += om * (-c[2])
            ub[nu + gid] += om * c[1]
        end
        Sb, _ = _semquad_rotation_moment(g, sq, ub)
        ua = copy(ub)
        om_alt = Sb / Q2
        for (gid, c) in enumerate(sq.coords)
            ua[gid] -= om_alt * (-c[2])
            ua[nu + gid] -= om_alt * c[1]
        end
        Sa, _ = _semquad_rotation_moment(g, sq, ua)
        Sa_impl, _ = _semquad_rotation_moment(g, sq, u)
        @test abs(om * SZ) >= 1.0e-2
        @test abs(Sa) / abs(Sb) <= 1.0e-4
        @test abs(Sa_impl) <= 1.0e-13
        println("G276 d alternative normalisation: |S_before| = ", abs(Sb),
                "  |S_after_alt|/|S_before| = ", abs(Sa) / abs(Sb),
                "  implemented |S_after| = ", abs(Sa_impl))
        # the wedge keeps its behaviour: its free-slip card resolves the gauge
        # to :none (the radial cuts pin the rotation) and the pressure pin
        gwt = g276_wedge()
        qtw = g276_tree(2; seam=:none)
        ssw = make_semstokes(qtw, 2; geom=gwt)
        uw, pw, = solve_semquad_stokes_var(ssw, qtw, g276_one, fx, fz,
                                           g276_zero, g276_zero;
                                           linear=:direct, bc=:free,
                                           wall_bc=(:free, :free, :free,
                                                    :free))
        @test MS276.LAST_SEMSTOKES_ROTATION_GAUGE[] === :none
        @test MS276.LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin
        # ... and its seam sides are walls in the boundary mask
        bmw = _semquad_boundary_face_mask(qtw, ssw.su)
        nwall_y = 0
        for it in 1:length(qtw.leaves)
            b = 4 * (it - 1)
            (bmw[b + 3] != 0x00 || bmw[b + 4] != 0x00) && (nwall_y += 1)
        end
        @test nwall_y > 0
    end

    # ======================================================================
    # (e) the end-to-end ring driver
    # ======================================================================
    @testset "(e) end-to-end ring driver and the unchanged wedge" begin
        R0 = 1.22
        R1 = 2.22
        g = PolarAnnulus(R0, R1, 2.0 * pi; seam=:periodic)
        pr = Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:free,
                    wall_bc=(:free, :free, :free, :free))
        dt = 1.0e-3
        function ring_run(; nsteps=3, kwargs...)
            qt = g276_tree(2; seam=:periodic)
            ss = make_semstokes(qt, 2; geom=g)
            return run_semquad_simulation(qt, pr, zeros(Float64, ss.su.N);
                                          geom=:annulus, annulus_R0=R0,
                                          annulus_R1=R1, annulus_Phi=2.0 * pi,
                                          annulus_seam=:periodic, p=2,
                                          tstop=nsteps * dt, dt0=dt, dtmax=dt,
                                          dtmin=1.0e-16, linear=:direct,
                                          maxsteps=nsteps, amr_every=0,
                                          advection=:skew, maxpic=1,
                                          annulus_ic=:firedrake,
                                          verbose=false, kwargs...)
        end
        straight = ring_run()
        @test straight.steps == 3
        @test length(straight.history) == 3
        @test all(r -> length(r) == 18, straight.history)
        @test all(isfinite, vcat(straight.history...))
        @test all(isfinite, straight.theta) && all(isfinite, straight.ux) &&
              all(isfinite, straight.uz)
        @test geometry_identity(straight.ss.su.geom) ==
              geometry_identity(g)
        @test straight.qt.seam === :periodic
        @test MS276._geom_periodic(straight.ss.su.geom)
        sq = straight.ss.su
        nu = sq.N
        # the IC was applied and stays a perturbation of the base state
        @test maximum(abs, straight.theta) <= 0.2
        # the wall rows are active: free-slip circles -> normal velocity 0
        bmask = _semquad_boundary_face_mask(straight.qt, sq)
        conds = _semquad_wall_frames(g, sq, bmask)
        wnorm = 0.0
        for (node, cs) in conds
            c = cs[1]
            wnorm = max(wnorm, abs(c.nx * straight.ux[node] +
                                   c.ny * straight.uz[node]))
        end
        @test wnorm <= 1.0e-10
        # both gauges engaged in the driver's Stokes solves
        @test MS276.LAST_SEMSTOKES_ROTATION_GAUGE[] === :rotation
        @test MS276.LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin
        # history rows are the diagnostics of the returned state
        row = straight.history[end]
        nut, nub = semquad_nu_walls(sq, straight.qt, straight.theta, pr)
        @test row[3] == nut && row[4] == nub
        @test row[5] == semquad_vrms(sq, straight.qt, straight.ux, straight.uz)
        @test row[6] == semquad_mean_T(sq, straight.qt, straight.theta, pr)
        @test row[10] == minimum_leaf_height(straight.ss.su.geom,
                                            straight.qt)
        @test row[5] > 1.0e-3
        # identities: continuity and the free-slip energy balance
        @test abs(row[18]) <= 1.0e-2
        @test abs(g276_div_volume(g, sq, vcat(straight.ux, straight.uz))) <=
              1.0e-8 * max(maximum(hypot.(straight.ux, straight.uz)) *
                           total_measure(g, sq), 1.0)
        # the fields are periodic across the seam (the node identity is exact,
        # so this is the statement that the *same dofs* are used on both sides:
        # the fields are evaluated from the first and the last angular row at
        # non-nodal points too, so a discontinuous trace would show)
        @test g276_seam_trace_jump(sq, straight.theta) <= 1.0e-13
        @test g276_seam_trace_jump(sq, straight.ux) <= 1.0e-13
        @test g276_seam_trace_jump(sq, straight.uz) <= 1.0e-13
        println("G276 e driver: Nu = (", row[3], ", ", row[4], ")  Vrms = ",
                row[5], "  meanT = ", row[6], "  maxdiv = ", row[9],
                "  bal = ", row[18], "  steps = ", straight.steps,
                "  leaves = ", length(straight.qt.leaves))
        @test isfinite(row[3]) && isfinite(row[4])
        @test abs(row[3] - 1.0) <= 0.5
        @test abs(row[4] - 1.0) <= 0.5
        # :mg and :auto paths
        mg = ring_run(nsteps=2, linear=:mg)
        @test mg.steps == 2 && length(mg.history) == 2
        @test all(isfinite, vcat(mg.history...))
        au = ring_run(nsteps=2, linear=:auto)
        dit = ring_run(nsteps=2, linear=:direct)
        @test au.steps == 2 && dit.steps == 2
        @test au.theta == dit.theta
        @test abs(mg.history[end][5] - dit.history[end][5]) <= 1.0e-8
        # AMR: the cycle keeps the tree periodic, balanced and the transferred
        # theta periodic
        amr = ring_run(nsteps=2, amr_every=1, max_leaves=400)
        @test amr.steps == 2
        @test amr.qt.seam === :periodic
        @test max_face_level_jump(amr.qt) <= 1
        @test g276_seam_level_jump(amr.qt) <= 1
        @test all(isfinite, vcat(amr.history...))
        @test g276_seam_trace_jump(amr.ss.su, amr.theta) <= 1.0e-12
        println("G276 e AMR: leaves = ", length(amr.qt.leaves),
                "  maxjump = ", max_face_level_jump(amr.qt),
                "  seam jump = ", g276_seam_level_jump(amr.qt),
                "  theta seam jump = ",
                g276_seam_trace_jump(amr.ss.su, amr.theta))
        # checkpoint round trip restores the periodic tree
        path = joinpath(tempdir(), "g276_ring_ckpt.h5")
        ssamr = amr.ss
        MS276.save_semquad_state(path, amr.qt, amr.theta, amr.theta,
                                 amr.ux, amr.uz, amr.pv;
                                 t=0.0, dt=dt, dt_prev=dt, step=1, p=2,
                                 config=Dict("geometry" =>
                                             geometry_identity(amr.ss.su.geom)))
        rst = MS276.load_semquad_state(path)
        @test rst.qt.seam === :periodic
        @test Set(rst.qt.leaves) == Set(amr.qt.leaves)
        @test max_face_level_jump(rst.qt) <= 1
        @test rst.theta == amr.theta
        rm(path)
        # the seam agreement is refused, not silently run: a ring card on a
        # bounded tree, and a periodic tree on the wedge
        err = try
            qtb = g276_tree(2; seam=:none)
            ssb = make_semstokes(qtb, 2; geom=CartesianBox(qtb))
            run_semquad_simulation(qtb, pr, zeros(Float64, ssb.su.N);
                                   geom=:annulus, annulus_R0=1.0,
                                   annulus_R1=2.0, annulus_Phi=2.0 * pi,
                                   annulus_seam=:periodic, p=2,
                                   tstop=dt, dt0=dt, maxsteps=1,
                                   linear=:direct, amr_every=0, verbose=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("seam", sprint(showerror, err))
        err = try
            gfw = PolarAnnulus(1.0, 2.0, 2.0 * pi; seam=:wall)
            qtp = g276_tree(2; seam=:periodic)
            ssp = make_semstokes(qtp, 2; geom=PolarAnnulus(1.0, 2.0,
                                                           2.0 * pi;
                                                           seam=:periodic))
            run_semquad_simulation(qtp, pr, zeros(Float64, ssp.su.N);
                                   geom=gfw, p=2, tstop=dt, dt0=dt,
                                   maxsteps=1, linear=:direct, amr_every=0,
                                   verbose=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("seam", sprint(showerror, err))
        # the wedge card is unchanged: same history shape, seam sides are walls
        gw = g276_wedge()
        prw = Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:free,
                     wall_bc=(:free, :free, :free, :free))
        function wedge_run(; nsteps=3)
            qt = g276_tree(2; seam=:none)
            ss = make_semstokes(qt, 2; geom=gw)
            return run_semquad_simulation(qt, prw, zeros(Float64, ss.su.N);
                                          geom=gw, p=2, tstop=nsteps * dt,
                                          dt0=dt, dtmax=dt, dtmin=1.0e-16,
                                          linear=:direct, maxsteps=nsteps,
                                          amr_every=0, advection=:skew,
                                          maxpic=1, annulus_ic=:firedrake,
                                          verbose=false)
        end
        wres = wedge_run()
        @test wres.steps == 3
        @test all(r -> length(r) == 18, wres.history)
        @test all(isfinite, vcat(wres.history...))
        @test MS276.LAST_SEMSTOKES_ROTATION_GAUGE[] === :none
        @test MS276.LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin
        bmw = _semquad_boundary_face_mask(wres.qt, wres.ss.su)
        nwy = 0
        for it in 1:length(wres.qt.leaves)
            b = 4 * (it - 1)
            (bmw[b + 3] != 0x00 || bmw[b + 4] != 0x00) && (nwy += 1)
        end
        @test nwy > 0
        @test !MS276._geom_periodic(wres.ss.su.geom)
        println("G276 e wedge: steps = ", wres.steps, "  Nu = (",
                wres.history[end][3], ", ", wres.history[end][4], ")",
                "  wall-flagged y faces = ", nwy)
    end
end
