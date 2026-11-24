# G269: the energy assembly through the Jacobian — affine exactness,
# CurvedBox convergence orders, mapped-vs-box operator consistency and the
# map-curvature second-derivative checks (stage 0, cut 3).
#
# THE CONTRACT OF THIS GATE
#
# (a) **Affine exactness continuation for the energy operator.**  On a general
#     `AffineBox` (rotation + shear + anisotropic scale + shift), a polynomial
#     temperature field of total degree <= 2 and the polynomial Stokes velocity
#     `u = (x^2 - y^2, -2xy)` are reproduced by the mapped energy operator to
#     solver round-off.  Reason (the G154 A1 / G267 argument): an affine image
#     of the reference square contains every physical P2 polynomial, so the
#     mapped Q2 space contains the manufactured temperature and velocity; the
#     `p+2` Gauss rule is exact for the polynomial integrands (the advection
#     integrand has total degree 7 and the 4-point rule is exact to degree 7 in
#     each variable), so the only error is the linear solve.  The same statement
#     is checked on a hanging (mixed-level) tree, where the cut-1 logical
#     constraints are active.
#
# (b) **CurvedBox manufactured solutions, measured orders.**  The design
#     (GEOM_DESIGN Sec. 5.4/G269) predicts, for `p = 2` on a uniform refinement
#     sequence of the smooth non-affine map `x = Lx*S, y = T*(1+a*sin(pi S))`:
#
#       velocity L2 3, velocity H1 2, pressure L2 (mean removed) 2,
#       temperature theta L2 3, theta H1 2.
#
#     The thresholds asserted between the two finest levels are the design's:
#     >= 2.7 (velocity L2), >= 1.8 (velocity H1), >= 1.7 (pressure L2),
#     >= 2.7 (theta L2), >= 1.8 (theta H1), plus a monotone error decrease over
#     every level.  **Asymptotic-parallelogram caveat (recorded, not tuned):**
#     Q2-Q1 on non-affine quadrilateral meshes is known in the literature to
#     admit spurious pressure modes on *strongly distorted* meshes, i.e. when
#     the elements do not approach parallelograms under refinement; our map is
#     a single smooth global map, so the elements are asymptotically
#     parallelograms and the optimal orders should be recovered at the finest
#     pair.  The measured orders are printed.  If the finest pair fell below
#     the threshold the honest response is to record the sequence and the level
#     at which the asymptotic regime starts, not to loosen the threshold; this
#     gate does not tune.
#
#     Boundary data of the Stokes case: Dirichlet on the three straight logical
#     walls plus the exact **total traction** on the curved `yplus` wall, not
#     the design's "all four walls Dirichlet".  The solver's net-flux
#     compatibility guard refuses the all-Dirichlet configuration on a curved
#     boundary: the interpolant of the exact divergence-free field has an O(h^3)
#     mapped boundary flux (the normal varies along a curved face), so the
#     discrete continuity equations are incompatible.  See
#     `g269_top_traction` and GEOM_PROGRESS.md design entry C1.  The measured
#     orders still confirm the design's prediction.
#
# (c) **Mapped-vs-box cross-check (the G268 analogue for the energy path).**
#     The mapped energy kernel fed the **box map** (`AffineBox(I, 0, Lx)`) must
#     reproduce the untouched box kernel: `M`, `K`, `C`, `F`, the
#     entropy-viscosity coefficient per leaf, a full-physics assembly (heating,
#     latent phases) and one BDF2 `solve_semquad_energy` step (with EV on and
#     with a flux wall) agree to round-off on a dyadic, a non-dyadic and a
#     hanging tree; the mapped `lap_phi` fed the box map reproduces the
#     historical box `(2/h)^2` expression.  The two kernel bodies are
#     independent code paths (the box body is not routed through the geometry;
#     that is what keeps the nine state-dump hashes), so a shared mistake is
#     only possible in the data both read (`sq.coords`, the closures).
#
# (d) **A wrong second-derivative term is caught specifically.**  The map
#     curvature is checked three ways, each of which a missing/wrong term fails:
#
#     1. *Coordinate Laplacian identities.*  The physical Laplacian of the
#        coordinate monomials is known exactly (`lap x = 0`, `lap y = 0`,
#        `lap x^2 = 2`, `lap xy = 0`, `lap y^2 = 2`).  The curvature
#        coefficients are applied to the exact reference derivatives of those
#        functions (written in this file from the `CurvedBox` map definition) at
#        every quadrature point of a CurvedBox mesh.  Dropping the
#        map-curvature term `C` or the cross metric term `M12` is an O(1)
#        error; the measured magnitudes are printed and asserted to be large.
#     2. *High-order finite-difference reference.*  The mapped Laplacian of a
#        smooth nodal field (the production `lap_phi` applied to the nodal
#        values, which is the exact Laplacian of the interpolant) is compared
#        with a fourth-order central-difference Laplacian of the same
#        interpolant evaluated through the cut-1 mapped evaluator
#        (`semquad_eval`), i.e. an independent numerical reference that never
#        uses `map_curvature`.  The dropped-`C` variant is compared with the
#        same reference and its measured failure is asserted to be large.
#     3. *Base-state second derivative.*  `height_laplacian` on `CurvedBox` is
#        compared with a finite-difference Laplacian of `height`; the mapped
#        base-state source in the assembly uses exactly that scalar
#        (`Tbar''(h)|grad h|^2 + Tbar'(h) lap h`), so an error in it is an
#        error in the manufactured solution's source (parts (a)/(b) then fail).
#     The entropy-viscosity field on the CurvedBox is additionally checked to
#     be finite, non-negative, switched off on the logical wall rows and
#     non-zero on the interior, i.e. the mapped EV residual is a real field.
#
# Every assertion states what wrong implementation it catches.  No assertion
# compares a quantity against itself: the exact fields and their derivatives
# are written in this file, the L2/H1 norms are independent Gauss quadratures
# over the mapped elements (never the SEM mass matrix, never the assembly's own
# sums), and the FD references use only the evaluator.

using Test, QuaST, LinearAlgebra, SparseArrays
const MS269 = QuaST

# ---------------------------------------------------------------------------
# Manufactured fields (written here from their definitions, not read from the
# solver): the Stokes field is divergence free with zero Laplacian and a linear
# pressure, so eta = 1 and f = grad p give an exact solution on any geometry.
# The temperature field is a physical quadratic (its Laplacian is the constant
# `2d + 2f`).
# ---------------------------------------------------------------------------
g269_ux(x, y) = x^2 - y^2
g269_uz(x, y) = -2.0 * x * y
g269_p(x, y) = 1.0 + 2.0 * x - 0.5 * y
g269_fx(x, y) = 2.0
g269_fz(x, y) = -0.5

const G269_TA = 0.4
const G269_TB = 0.7
const G269_TC = -0.35
const G269_TD = 0.45
const G269_TE = 0.25
const G269_TF = -0.15
g269_theta(x, y) = G269_TA + G269_TB * x + G269_TC * y +
                   G269_TD * x^2 + G269_TE * x * y + G269_TF * y^2
g269_theta_dx(x, y) = G269_TB + 2.0 * G269_TD * x + G269_TE * y
g269_theta_dy(x, y) = G269_TC + G269_TE * x + 2.0 * G269_TF * y
g269_lap_theta(x, y) = 2.0 * G269_TD + 2.0 * G269_TF

# relative max-abs disagreement of two matrices/vectors (0 when bitwise equal)
function g269_rel(a, b)
    s = maximum(abs, b)
    d = maximum(abs, a - b)
    return s == 0.0 ? d : d / s
end

# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------
# uniform refinement of the unit-square base tree: after nref refinements
# nside = 2^nref, h = 2^-nref, 4^nref leaves
function g269_tree(nref::Int)
    qt = Quadtree(2 + nref)
    for _ in 1:nref, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# a genuinely 2:1 hanging tree (constraints active)
function g269_mixed_tree()
    qt = Quadtree(3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    for lf in sort(collect(qt.leaves))[1:5]
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# non-dyadic square-leaf tree: base cells 7x3 with Lx = 7/3 (leaf h = 1/12);
# the mapped coordinates are not bitwise the box ones (G268's tree)
function g269_cells73()
    qt = Quadtree(2; cells_x=7, cells_y=3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# the affine validation geometry: rotation * shear/anisotropic scale + shift
function g269_affine()
    th = 0.4
    R = ((cos(th), -sin(th)), (sin(th), cos(th)))
    S = ((1.2, 0.3), (0.0, 1.1))
    A = (R[1][1] * S[1][1] + R[1][2] * S[2][1],
         R[1][1] * S[1][2] + R[1][2] * S[2][2],
         R[2][1] * S[1][1] + R[2][2] * S[2][1],
         R[2][1] * S[1][2] + R[2][2] * S[2][2])
    return AffineBox(A, (0.15, -0.1), 1.0)
end

g269_boxmap(qt) = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), qt.Lx)

# ---------------------------------------------------------------------------
# Analytic physical height h(x,y), grad h and lap h for the validation
# geometries, written in the gate from the map definitions (not read from the
# geometry object; `g269_height_checks` compares the library accessors against
# finite differences separately).
# ---------------------------------------------------------------------------
function g269_height_terms(g::CurvedBox, x, y)
    S = x / g.Lx
    d = 1.0 + g.a * sinpi(S)
    dp = g.a * pi * cospi(S)
    dpp = -g.a * pi * pi * sinpi(S)
    T = y / d
    return T, -T * dp / (d * g.Lx), 1.0 / d,
           T * (-dpp / d + 2.0 * (dp / d)^2) / (g.Lx * g.Lx)
end
function g269_height_terms(g::AffineBox, x, y)
    A = g.A
    det = A[1] * A[4] - A[2] * A[3]
    h = (-A[3] * (x - g.b[1]) + A[1] * (y - g.b[2])) / det
    return h, -A[3] / det, A[1] / det, 0.0
end

# The manufactured source of the mapped theta equation.  With Tbar(h) = 1 - h
# (Params(H=0), refstate=nothing) the implemented theta equation is
#   theta_t + u.grad theta - lap(theta) = -u.grad h dTbar/dh + lap(Tbar) + g,
# dTbar/dh = -1 and lap(Tbar) = -lap(h), so
#   g = u.grad theta - lap(theta) - u.grad h + lap(h).
function g269_energy_source(g::AbstractGeometry, x, y)
    h, ghx, ghy, laph = g269_height_terms(g, x, y)
    ux = g269_ux(x, y); uz = g269_uz(x, y)
    ugh = ux * ghx + uz * ghy
    ugrad = ux * g269_theta_dx(x, y) + uz * g269_theta_dy(x, y)
    return ugrad - g269_lap_theta(x, y) - ugh + laph
end

# Exact total traction on the **curved** logical `yplus` wall of a `CurvedBox`.
# The wall is y = f(x) = 1 + a sin(pi x/Lx); its outward unit normal is
# (-f', 1)/|.| and the positive logical tangent is (1, f')/|.| (the image of
# d/dxi).  With eta = 1, p = 1 + 2x - 0.5y and the exact strain rate
# eps(u) = [[2x, -2y], [-2y, -2x]], the total stress is
# sigma = 2 eps - p I, and the prescribed data are t.n and t.tau (the mapped
# wall-local convention of cut 2).
#
# Why the Stokes case uses one traction wall instead of the design's "all four
# walls Dirichlet": the solver refuses a fully-Dirichlet configuration whose
# prescribed normal data have a non-zero net boundary flux (the discrete
# continuity equations are only compatible when ∫ u.n dS = 0, GEOM_DESIGN/
# BC_DESIGN §3.5).  On a straight boundary the *interpolant* of the exact
# divergence-free field has zero net flux on each side to round-off (that is why
# G267 uses it), but the mapped curved wall's normal is not constant, so the
# interpolant's boundary flux is O(h^3), not zero, and the guard (correctly)
# refuses.  Leaving the curved wall's normal component free via an `:open` wall
# with its exact traction removes the compatibility constraint and still tests
# the mapped assembly at its optimal order; the deviation is recorded in
# docs/dev/GEOM_PROGRESS.md.
function g269_top_traction(g::CurvedBox, x, y)
    f = 1.0 + g.a * sinpi(x / g.Lx)
    fp = g.a * pi * cospi(x / g.Lx) / g.Lx
    nn = hypot(-fp, 1.0)
    nx = -fp / nn; ny = 1.0 / nn
    tn = hypot(1.0, fp)
    tx = 1.0 / tn; ty = fp / tn
    pr = g269_p(x, y)
    sxx = 4.0 * x - pr
    syy = -4.0 * x - pr
    sxy = -4.0 * y
    ttx = sxx * nx + sxy * ny
    tty = sxy * nx + syy * ny
    return (ttx * nx + tty * ny, ttx * tx + tty * ty)
end

# ---------------------------------------------------------------------------
# Test-side energy solve: Dirichlet on all four logical walls, the load of the
# manufactured source integrated with the mapped measure, then the lifted
# solve of (K + C) theta = F + G.  The production code supplies only `M,K,C,F`
# and the constraint map.
# ---------------------------------------------------------------------------
function g269_dirichlet(sq::SemQuad, theta_ex::Function)
    fixed = Dict{Int,Float64}()
    n = sq.p + 1
    for lf in sq.leaves
        nsidex = MS269.leaf_nside_x(sq, lf[1])
        nsidey = MS269.leaf_nside_y(sq, lf[1])
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            onb = (i == 1 && lf[2] == 0) || (i == n && lf[2] == nsidex - 1) ||
                  (j == 1 && lf[3] == 0) || (j == n && lf[3] == nsidey - 1)
            onb || continue
            gid = ids[(j - 1) * n + i]
            fixed[gid] = theta_ex(sq.coords[gid]...)
        end
    end
    return fixed
end

function g269_load(sq::SemQuad, g::AbstractGeometry, src::Function)
    fd = make_fedata(sq.p, sq.p)
    n = sq.p + 1
    G = zeros(Float64, sq.N)
    for lf in sq.leaves
        lm = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(lm, sq, lf, fd, qx, qy)
            w = MS269.quad_measure(lm, fd, qx, qy)
            s = src(x, y)
            for j in 1:n, i in 1:n
                G[ids[(j - 1) * n + i]] +=
                    w * s * (fd.phi_v[i, qx] * fd.phi_v[j, qy])
            end
        end
    end
    return G
end

function g269_solve_energy(sq::SemQuad, qt::Quadtree, g::AbstractGeometry,
                           pr::Params, ux, uz, theta_ex::Function;
                           advection::Symbol=:galerkin)
    M, K, C, F = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                         advection=advection)
    G = g269_load(sq, g, (x, y) -> g269_energy_source(g, x, y))
    fixed = g269_dirichlet(sq, theta_ex)
    cons = semquad_constraints(sq, qt)
    u0, Cfree, _ = MS269._affine_constraint(sq.N, cons, fixed)
    A = K + C
    Ar = Cfree' * A * Cfree
    br = Cfree' * (F + G - A * u0)
    return u0 + Cfree * (Ar \ br)
end

# ---------------------------------------------------------------------------
# Error measures: independent Gauss quadratures (`nq = p + 6`) over the mapped
# elements, through the cut-1 mapped evaluator/gradient.  The exact fields are
# the functions above, never the nodal interpolant.
# ---------------------------------------------------------------------------
function g269_stokes_errors(g::AbstractGeometry, ss, qt::Quadtree, u, pv;
                            nq::Int=0)
    su = ss.su
    sp = ss.sp
    nu = su.N
    nq == 0 && (nq = su.p + 6)
    fd = make_fedata(su.p, sp.p; nq=nq)
    leaves = Set(su.leaves)
    uvec = u[1:nu]
    zvec = u[(nu + 1):(2nu)]
    lu2 = 0.0; lh1 = 0.0; area = 0.0
    pe = Float64[]; pw = Float64[]
    for lf in su.leaves
        lm = leaf_metric(g, su, lf, fd)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(lm, su, lf, fd, qx, qy)
            w = MS269.quad_measure(lm, fd, qx, qy)
            ux = semquad_eval(su, uvec, x, y)
            uz = semquad_eval(su, zvec, x, y)
            lu2 += w * ((ux - g269_ux(x, y))^2 + (uz - g269_uz(x, y))^2)
            gxu, gyu = MS269._semquad_grad_at(su, uvec, x, y, leaves)
            gxz, gyz = MS269._semquad_grad_at(su, zvec, x, y, leaves)
            # exact: d(ux)/dx = 2x, d(ux)/dy = -2y, d(uz)/dx = -2y,
            # d(uz)/dy = -2x
            lh1 += w * ((gxu - 2.0 * x)^2 + (gyu + 2.0 * y)^2 +
                        (gxz + 2.0 * y)^2 + (gyz + 2.0 * x)^2)
            push!(pe, semquad_eval(sp, pv, x, y) - g269_p(x, y))
            push!(pw, w)
            area += w
        end
    end
    shift = sum(pw .* pe) / area
    lp2 = sqrt(sum(pw .* (pe .- shift) .^ 2))
    return sqrt(lu2), sqrt(lh1), lp2
end

function g269_energy_errors(g::AbstractGeometry, sq::SemQuad, qt::Quadtree, th;
                            nq::Int=0)
    nq == 0 && (nq = sq.p + 6)
    fd = make_fedata(sq.p, sq.p; nq=nq)
    leaves = Set(sq.leaves)
    l2 = 0.0; h1 = 0.0
    for lf in sq.leaves
        lm = leaf_metric(g, sq, lf, fd)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(lm, sq, lf, fd, qx, qy)
            w = MS269.quad_measure(lm, fd, qx, qy)
            e = semquad_eval(sq, th, x, y) - g269_theta(x, y)
            l2 += w * e^2
            gx, gy = MS269._semquad_grad_at(sq, th, x, y, leaves)
            h1 += w * ((gx - g269_theta_dx(x, y))^2 +
                       (gy - g269_theta_dy(x, y))^2)
        end
    end
    return sqrt(l2), sqrt(h1)
end

# order between two levels; h is halved each level, so log2 of the ratio
g269_order(e_coarse, e_fine) = log2(e_coarse / e_fine)

# reference derivatives of the coordinate monomials in leaf reference
# coordinates, from the CurvedBox map definition:
#   (f_xi, f_eta, f_xi xi, f_xi eta, f_eta eta)
function g269_ref_derivs(g::CurvedBox, sq::SemQuad, lf, xi, eta)
    S, T = MS269.global_logical(sq, lf, xi, eta)
    dS = (1.0 / MS269.leaf_nside_x(sq, lf[1])) * 0.5
    dT = (1.0 / MS269.leaf_nside_y(sq, lf[1])) * 0.5
    d = 1.0 + g.a * sinpi(S)
    dp = g.a * pi * cospi(S)
    dpp = -g.a * pi * pi * sinpi(S)
    x = g.Lx * S
    y = T * d
    x_xi = g.Lx * dS
    y_xi = T * dp * dS
    y_xixi = T * dpp * dS * dS
    y_xie = dp * dS * dT
    y_eta = d * dT
    f1 = (x_xi, 0.0, 0.0, 0.0, 0.0)                       # x
    f2 = (y_xi, y_eta, y_xixi, y_xie, 0.0)                # y
    f3 = (2.0 * x * x_xi, 0.0, 2.0 * x_xi * x_xi, 0.0, 0.0)   # x^2
    f4 = (2.0 * y * y_xi, 2.0 * y * y_eta,
          2.0 * y_xi * y_xi + 2.0 * y * y_xixi,
          2.0 * y_xi * y_eta + 2.0 * y * y_xie,
          2.0 * y_eta * y_eta)                            # y^2
    f5 = (x_xi * y, x * y_eta,
          2.0 * x_xi * y_xi + x * y_xixi,
          x_xi * y_eta + x * y_xie, 0.0)                  # x*y
    return ((f1, 0.0, "x"), (f2, 0.0, "y"), (f3, 2.0, "x^2"),
            (f4, 2.0, "y^2"), (f5, 0.0, "x*y"))
end

# the mapped Laplacian formula, with optional deliberate omissions (the
# "wrong second-derivative term" variants the gate measures)
function g269_lap_from(mc, qx::Int, qy::Int, d;
                       drop_c::Bool=false, drop_m12::Bool=false)
    k = (qy - 1) * mc.nq + qx
    m12 = drop_m12 ? 0.0 : mc.m12[k]
    c1 = drop_c ? 0.0 : mc.c1[k]
    c2 = drop_c ? 0.0 : mc.c2[k]
    return mc.m11[k] * d[3] + 2.0 * m12 * d[4] + mc.m22[k] * d[5] +
           c1 * d[1] + c2 * d[2]
end

@testset "G269 curved energy assembly (stage 0 cut 3)" begin
    p = 2
    # Boussinesq base with H = 0: Tbar(h) = 1 - h, dTbar = -1, no refstate.
    pr0 = Params(Ra=1.0e3, H=0.0)

    # ======================================================================
    # (a) affine exactness of the mapped energy operator
    # ======================================================================
    @testset "affine energy exactness (AffineBox)" begin
        g = g269_affine()
        for (tag, qt) in (("uniform", g269_tree(2)), ("mixed", g269_mixed_tree()))
            sq = make_semquad(qt, p; geom=g)
            ux = [g269_ux(c...) for c in sq.coords]
            uz = [g269_uz(c...) for c in sq.coords]
            th = g269_solve_energy(sq, qt, g, pr0, ux, uz, g269_theta)
            # nodal and independent-quadrature errors, at solver round-off
            eth_nodal = maximum(abs(th[gi] - g269_theta(sq.coords[gi]...))
                                for gi in 1:sq.N)
            l2, h1 = g269_energy_errors(g, sq, qt, th)
            # the exact field is in the mapped Q2 space: max error is the solve
            @test eth_nodal < 1.0e-11      # measured ~1e-13 or below
            @test l2 < 1.0e-11
            @test h1 < 1.0e-11
            # the mass operator reproduces the exact P2 mass action
            M, K, C, F = assemble_semquad_energy(sq, qt, pr0, ux, uz)
            thex = [g269_theta(c...) for c in sq.coords]
            Gth = g269_load(sq, g, g269_theta)
            @test g269_rel(M * thex, Gth) < 1.0e-11
            # the hanging tree really has constraints, so the lift above was
            # not the identity
            if tag == "mixed"
                @test !isempty(semquad_constraints(sq, qt))
                @test size(MS269._affine_constraint(sq.N,
                        semquad_constraints(sq, qt), Dict{Int,Float64}())[2], 2) <
                      sq.N
            end
            # the affine exactness is meaningful only if the temperature field
            # is genuinely represented: the field has O(1) variation
            @test maximum(thex) - minimum(thex) > 0.1
            println("G269 affine(", tag, "): theta nodal=", eth_nodal,
                    " L2=", l2, " H1=", h1,
                    " mass rel=", g269_rel(M * thex, Gth))
        end
    end

    # ======================================================================
    # (b) CurvedBox manufactured solutions: measured convergence orders
    # ======================================================================
    @testset "CurvedBox manufactured solutions" begin
        g = CurvedBox(0.2, 1.0)
        # h = 1/4, 1/8, 1/16, 1/32 (nside = 2^nref, leaves = 4^nref)
        nrefs = (2, 3, 4, 5)
        hs = Float64[]; eu2 = Float64[]; eu1 = Float64[]; ep2 = Float64[]
        et2 = Float64[]; et1 = Float64[]
        for nref in nrefs
            qt = g269_tree(nref)
            hs = push!(hs, 1.0 / (2^nref))
            # --- Stokes: exact u/p, Dirichlet on the three straight logical
            # walls and the exact total traction on the curved yplus wall (see
            # `g269_top_traction` for why not all four walls are Dirichlet) ---
            ss = make_semstokes(qt, p; geom=g)
            tnf = (x, y) -> g269_top_traction(g, x, y)[1]
            ttf = (x, y) -> g269_top_traction(g, x, y)[2]
            res = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, g269_fx,
                                           g269_fz, g269_ux, g269_uz;
                                           linear=:direct, bc=:rigid,
                                           wall_bc=(:same, :same, :same, :open),
                                           wall_t=(nothing, nothing, nothing, ttf),
                                           wall_tn=(nothing, nothing, nothing, tnf))
            lu2, lh1, lp2 = g269_stokes_errors(g, ss, qt, res[1], res[2])
            push!(eu2, lu2); push!(eu1, lh1); push!(ep2, lp2)
            # the traction data drive a genuine flow (a dropped load would give
            # the zero solution, whose errors would still "converge" trivially)
            @test maximum(abs, res[1]) > 0.1
            # --- energy: exact velocity as nodal data (the design's test 2),
            # all four walls Dirichlet with the manufactured field ---
            sq = make_semquad(qt, p; geom=g)
            ux = [g269_ux(c...) for c in sq.coords]
            uz = [g269_uz(c...) for c in sq.coords]
            th = g269_solve_energy(sq, qt, g, pr0, ux, uz, g269_theta)
            l2, h1 = g269_energy_errors(g, sq, qt, th)
            push!(et2, l2); push!(et1, h1)
        end
        # measured orders between consecutive levels
        ou2 = [g269_order(eu2[k], eu2[k + 1]) for k in 1:(length(hs) - 1)]
        ou1 = [g269_order(eu1[k], eu1[k + 1]) for k in 1:(length(hs) - 1)]
        op2 = [g269_order(ep2[k], ep2[k + 1]) for k in 1:(length(hs) - 1)]
        ot2 = [g269_order(et2[k], et2[k + 1]) for k in 1:(length(hs) - 1)]
        ot1 = [g269_order(et1[k], et1[k + 1]) for k in 1:(length(hs) - 1)]
        println("G269 CurvedBox convergence (a=0.2, p=2):")
        println("   h        vel_L2      ord      vel_H1      ord      p_L2        ord" *
                "      theta_L2    ord      theta_H1    ord")
        for k in eachindex(hs)
            println("  ", rpad(hs[k], 8),
                    rpad(round(eu2[k], sigdigits=4), 11),
                    rpad(k > 1 ? round(ou2[k - 1], digits=3) : "  -", 9),
                    rpad(round(eu1[k], sigdigits=4), 11),
                    rpad(k > 1 ? round(ou1[k - 1], digits=3) : "  -", 9),
                    rpad(round(ep2[k], sigdigits=4), 12),
                    rpad(k > 1 ? round(op2[k - 1], digits=3) : "  -", 9),
                    rpad(round(et2[k], sigdigits=4), 12),
                    rpad(k > 1 ? round(ot2[k - 1], digits=3) : "  -", 9),
                    rpad(round(et1[k], sigdigits=4), 12),
                    k > 1 ? round(ot1[k - 1], digits=3) : "  -")
        end
        # monotone decrease over every level
        for (nm, e) in (("vel L2", eu2), ("vel H1", eu1), ("p L2", ep2),
                        ("theta L2", et2), ("theta H1", et1))
            for k in 1:(length(e) - 1)
                @test e[k + 1] < e[k]
            end
            @test all(isfinite, e)
        end
        # the design's thresholds on the finest pair (0.1 slack is not needed:
        # these ARE the design's numbers; the measured values are printed)
        @test ou2[end] >= 2.7     # design: velocity L2 order 3
        @test ou1[end] >= 1.8     # design: velocity H1 order 2
        @test op2[end] >= 1.7     # design: pressure L2 order 2 (Q2-Q1 caveat)
        @test ot2[end] >= 2.7     # design: theta L2 order 3
        @test ot1[end] >= 1.8     # design: theta H1 order 2
        # the energy error must be in the same asymptotic regime as the
        # velocity error: the manufactured theta field is a P2 physical
        # polynomial, which the curved mapped space does not contain, so the
        # convergence is genuine (not exactness)
        @test et2[end] > 0.0
    end

    # ======================================================================
    # (c) mapped-vs-box cross-check on the box map (the G268 analogue)
    # ======================================================================
    @testset "mapped-vs-box energy operators (box map)" begin
        for (tag, qt) in (("dyadic", g269_tree(3)),
                          ("cells73", g269_cells73()),
                          ("mixed", g269_mixed_tree()))
            gid = g269_boxmap(qt)
            sqb = make_semquad(qt, p)
            sqm = make_semquad(qt, p; geom=gid)
            @test sqm.geom === gid
            uxb = [0.3 + 0.2 * c[1] - 0.1 * c[2] + 0.05 * c[1] * c[2]
                   for c in sqb.coords]
            uzb = [-0.2 + 0.1 * c[1] + 0.25 * c[2] - 0.04 * c[1]^2
                   for c in sqb.coords]
            thb = [0.15 * sinpi(1.5 * c[2]) * (1.0 + 0.2 * cospi(2.0 * c[1]))
                   for c in sqb.coords]
            uxm = [0.3 + 0.2 * c[1] - 0.1 * c[2] + 0.05 * c[1] * c[2]
                   for c in sqm.coords]
            uzm = [-0.2 + 0.1 * c[1] + 0.25 * c[2] - 0.04 * c[1]^2
                   for c in sqm.coords]
            thm = [0.15 * sinpi(1.5 * c[2]) * (1.0 + 0.2 * cospi(2.0 * c[1]))
                   for c in sqm.coords]

            # ---- the four basic energy operators -------------------------
            Mb, Kb, Cb, Fb = assemble_semquad_energy(sqb, qt, pr0, uxb, uzb;
                                                     theta=thb)
            Mm, Km, Cm, Fm = assemble_semquad_energy(sqm, qt, pr0, uxm, uzm;
                                                     theta=thm)
            @test size(Mm) == size(Mb)
            for (nm, a, b) in (("M", Mm, Mb), ("K", Km, Kb), ("C", Cm, Cb))
                @test g269_rel(a, b) <= 1.0e-14
            end
            @test g269_rel(Fm, Fb) <= 1.0e-14
            # the dyadic tree agrees bitwise (both coordinate arrays are the
            # same floats); the assertion above is then also a bitwise one
            if tag == "dyadic"
                @test Mm == Mb && Km == Kb && Cm == Cb && Fm == Fb
            end

            # ---- entropy-viscosity coefficient --------------------------
            evgb = MS269.semquad_ev_globals(sqb, qt, pr0, thb, thb, uxb, uzb,
                                            0.0)
            evgm = MS269.semquad_ev_globals(gid, sqm, qt, pr0, thm, thm, uxm,
                                            uzm, 0.0)
            nub = [MS269.semquad_ev_nu(sqb, pr0, thb, thb, uxb, uzb, 1.0e-2,
                                       lf, evgb; ev=1.0e3, ev_cmax=0.052,
                                       ev_alpha=2, ev_timestep=10)
                   for lf in sqb.leaves]
            num = [MS269.semquad_ev_nu(gid, sqm, pr0, thm, thm, uxm, uzm,
                                       1.0e-2, lf, evgm; ev=1.0e3,
                                       ev_cmax=0.052, ev_alpha=2,
                                       ev_timestep=10)
                   for lf in sqm.leaves]
            # the EV field is a real field: some interior leaf has nu > 0 (a
            # tree with no interior row would make this comparison trivial)
            @test any(nub .> 0.0)
            @test g269_rel(num, nub) <= 1.0e-14
            # and it actually enters the stiffness: the EV-on K differs from
            # the EV-off K (a mapped body that dropped the EV factor would fail
            # the cross-check above but also this)
            evob = zeros(Float64, length(sqb.leaves))
            evom = zeros(Float64, length(sqm.leaves))
            _, Kevb, _, _ = assemble_semquad_energy(sqb, qt, pr0, uxb, uzb;
                                                    theta=thb, ev=1.0e3,
                                                    ev_cmax=0.052, ev_alpha=2,
                                                    ev_timestep=10, evg=evgb,
                                                    ev_out=evob)
            _, Kevm, _, _ = assemble_semquad_energy(sqm, qt, pr0, uxm, uzm;
                                                    theta=thm, ev=1.0e3,
                                                    ev_cmax=0.052, ev_alpha=2,
                                                    ev_timestep=10, evg=evgm,
                                                    ev_out=evom)
            @test g269_rel(evom, evob) <= 1.0e-14
            @test g269_rel(Kevm, Kevb) <= 1.0e-14
            @test norm(Kevb - Kb) > 10.0 * eps(norm(Kb))

            # ---- full physics: heating, latent phases, density factors ---
            prf = Params(Ra=1.0e3, H=6.0, rho_heating=true,
                         heating_adiabatic=true, heating_dissipation=true,
                         Di=0.3)
            laps = [PhaseTransition(0.5, 0.08, 0.0, 0.5, 100.0, 0.5)]
            Mb2, Kb2, Cb2, Fb2 = assemble_semquad_energy(
                sqb, qt, prf, uxb, uzb; theta=thb, theta_lat=thb,
                phases=laps, rho_heating=true, heating_adiabatic=true,
                heating_dissipation=true, eta_fun=(x, y) -> 1.0 + 0.1 * x)
            Mm2, Km2, Cm2, Fm2 = assemble_semquad_energy(
                sqm, qt, prf, uxm, uzm; theta=thm, theta_lat=thm,
                phases=laps, rho_heating=true, heating_adiabatic=true,
                heating_dissipation=true, eta_fun=(x, y) -> 1.0 + 0.1 * x)
            # 1e-12 rather than 1e-14: the box body evaluates T at the
            # quadrature point through `_semquad_eval_at_leaf!` (inverse map
            # then Lagrange sums) while the mapped body sums the same basis
            # products already in hand; both are the same polynomial value but
            # the summation order differs, and the density/latent factors
            # amplify the last-bit difference.  A wrong mapped term is O(1).
            for (nm, a, b) in (("M", Mm2, Mb2), ("K", Km2, Kb2),
                               ("C", Cm2, Cb2), ("F", Fm2, Fb2))
                @test g269_rel(a, b) <= 1.0e-12
            end

            # ---- one BDF2 energy step through solve_semquad_energy --------
            # EV on: exercises the mapped EV globals, the mapped fixed-dof
            # loop and the mapped leaf assembly end to end.
            r = solve_semquad_energy(sqb, qt, pr0, thb, thb, uxb, uzb,
                                     1.0e-2, 1.0e-2, true;
                                     advection=:galerkin, ev=1.0e3,
                                     ev_cmax=0.052, ev_alpha=2,
                                     ev_timestep=10)
            rm = solve_semquad_energy(sqm, qt, pr0, thm, thm, uxm, uzm,
                                      1.0e-2, 1.0e-2, true;
                                      advection=:galerkin, ev=1.0e3,
                                      ev_cmax=0.052, ev_alpha=2,
                                      ev_timestep=10)
            @test g269_rel(rm, r) <= 1.0e-11
            # a prescribed flux on the bottom wall: exercises the mapped
            # `_add_semquad_wall_flux!` (face Jacobian and Gauss weights)
            rf = solve_semquad_energy(sqb, qt, pr0, thb, thb, uxb, uzb,
                                      1.0e-2, 1.0e-2, true;
                                      advection=:galerkin, flux_bot=0.3)
            rfm = solve_semquad_energy(sqm, qt, pr0, thm, thm, uxm, uzm,
                                       1.0e-2, 1.0e-2, true;
                                       advection=:galerkin, flux_bot=0.3)
            @test g269_rel(rfm, rf) <= 1.0e-11
            # the flux really changed the step (a mapped body that dropped the
            # wall load would reproduce the Dirichlet step instead)
            @test g269_rel(rf, r) > 1.0e-3

            # ---- the mapped Laplacian fed the box map == historical box ----
            fd = make_fedata(p, p)
            d2tab = [MS269.lagrange_d2(MS269.gll_rule_cached(p), fd.gl_x[k])[3]
                     for k in 1:fd.nq]
            mcb = MS269.map_curvature(sqb.geom, sqb, sqb.leaves[1], fd)
            mcm = MS269.map_curvature(gid, sqm, sqm.leaves[1], fd)
            n = p + 1
            maxlap = 0.0; scalelap = 0.0
            for qy in 1:fd.nq, qx in 1:fd.nq, j in 1:n, i in 1:n
                vb = MS269.lap_phi(mcb, fd, d2tab, qx, qy, i, j)
                vm = MS269.lap_phi(mcm, fd, d2tab, qx, qy, i, j)
                maxlap = max(maxlap, abs(vm - vb))
                scalelap = max(scalelap, abs(vb))
            end
            @test scalelap > 0.0
            @test maxlap / scalelap <= 1.0e-14
            # the box curvature is the *historical* (2/h)^2 with h = leaf_hy;
            # on square leaves the mapped and box coefficients coincide
            hh = MS269.leaf_hy(sqb, sqb.leaves[1][1])
            @test mcb.m11 == (2.0 / hh)^2
            @test mcb.m22 == (2.0 / hh)^2
            @test mcb.m12 == 0.0 && mcb.c1 == 0.0 && mcb.c2 == 0.0
            @test mcm.m11[1] == mcb.m11
            @test mcm.m22[1] == mcb.m22
            println("G269 cross(", tag, "): M/K/C/F rel=",
                    g269_rel(Mm, Mb), "/", g269_rel(Km, Kb), "/",
                    g269_rel(Cm, Cb), "/", g269_rel(Fm, Fb),
                    " EV rel=", g269_rel(num, nub),
                    " step rel=", g269_rel(rm, r),
                    " flux step rel=", g269_rel(rfm, rf),
                    " lap rel=", maxlap / scalelap)
        end
    end

    # ======================================================================
    # (d) map-curvature second derivatives: the wrong-term catches
    # ======================================================================
    @testset "map curvature second derivatives" begin
        g = CurvedBox(0.2, 1.0)
        qt = g269_tree(3)                 # 8x8 leaves, interior quadrature pts
        sq = make_semquad(qt, p; geom=g)
        fd = make_fedata(p, p)
        d2tab = [MS269.lagrange_d2(MS269.gll_rule_cached(p), fd.gl_x[k])[3]
                 for k in 1:fd.nq]

        # ---- 1. coordinate Laplacian identities ---------------------------
        # lap x = 0, lap y = 0, lap x^2 = 2, lap y^2 = 2, lap xy = 0 exactly
        err = Dict{String,Float64}()
        errC = Dict{String,Float64}()
        errM12 = Dict{String,Float64}()
        for lf in sq.leaves
            mc = MS269.map_curvature(g, sq, lf, fd)
            for qy in 1:fd.nq, qx in 1:fd.nq
                for (d, exact, nm) in g269_ref_derivs(g, sq, lf,
                                                      fd.gl_x[qx], fd.gl_x[qy])
                    v = g269_lap_from(mc, qx, qy, d)
                    err[nm] = max(get(err, nm, 0.0), abs(v - exact))
                    vc = g269_lap_from(mc, qx, qy, d; drop_c=true)
                    errC[nm] = max(get(errC, nm, 0.0), abs(vc - exact))
                    vm = g269_lap_from(mc, qx, qy, d; drop_m12=true)
                    errM12[nm] = max(get(errM12, nm, 0.0), abs(vm - exact))
                end
            end
        end
        for nm in ("x", "y", "x^2", "y^2", "x*y")
            @test err[nm] < 1.0e-13   # measured at round-off
            println("G269 curvature id ", rpad(nm, 4), " full=", err[nm],
                    "  drop-C=", errC[nm], "  drop-M12=", errM12[nm])
        end
        # The two wrong variants must be **large** somewhere, which is why the
        # gate would catch a dropped term even though the box cannot see it.
        # On this map `c1 == 0` identically (the map Hessian is x-Hessian =
        # 0 and the curvature vector's xi-component is Jinv[y,xi]*tr(M H_y)
        # with Jinv[y,xi] = 0), so dropping `C` is invisible for functions
        # that only vary in x (x and x^2); the y-dependent functions exercise
        # `c2` and `M12`.  The per-function table above is printed so that this
        # structure is explicit rather than hidden.
        @test maximum(values(errC)) > 0.1
        @test maximum(values(errM12)) > 0.1
        println("G269 curvature identities: full err=",
                maximum(values(err)),
                "  drop-C err=", maximum(values(errC)),
                "  drop-M12 err=", maximum(values(errM12)))

        # ---- 2. high-order finite-difference reference --------------------
        # a smooth nodal field; the production lap_phi applied to the nodal
        # values is the exact Laplacian of the interpolant.  A fourth-order
        # central difference of the same interpolant (through the cut-1 mapped
        # evaluator `semquad_eval`) is an independent reference.
        fth(x, y) = sinpi(x) * cospi(2.0 * y) + 0.3 * x * y +
                    0.2 * x^2 - 0.1 * y^2
        thn = [fth(c...) for c in sq.coords]
        fthev = (x, y) -> semquad_eval(sq, thn, x, y)
        hloc = 1.0 / (2^3)                 # leaf size at nref = 3
        δ = 0.02 * hloc
        function fd4(f, x, y)
            fxx = (-f(x + 2δ, y) + 16.0 * f(x + δ, y) - 30.0 * f(x, y) +
                   16.0 * f(x - δ, y) - f(x - 2δ, y)) / (12.0 * δ * δ)
            fyy = (-f(x, y + 2δ) + 16.0 * f(x, y + δ) - 30.0 * f(x, y) +
                   16.0 * f(x, y - δ) - f(x, y - 2δ)) / (12.0 * δ * δ)
            return fxx + fyy
        end
        maxfd = 0.0; maxfdC = 0.0; scale = 0.0
        for lf in sq.leaves
            mc = MS269.map_curvature(g, sq, lf, fd)
            ids = sq.elem[lf]
            for qy in 1:fd.nq, qx in 1:fd.nq
                x, y = quad_point(leaf_metric(g, sq, lf, fd), sq, lf, fd,
                                  qx, qy)
                ld = 0.0; ldc = 0.0
                for j in 1:(p + 1), i in 1:(p + 1)
                    k = (j - 1) * (p + 1) + i
                    ld += thn[ids[k]] *
                          MS269.lap_phi(mc, fd, d2tab, qx, qy, i, j)
                    # the dropped-C variant: rebuild the same combination with
                    # c1 = c2 = 0 by using a copy of the coefficient arrays
                    ldc += thn[ids[k]] * g269_lap_from(
                        mc, qx, qy,
                        (fd.dphi_v[i, qx] * fd.phi_v[j, qy],
                         fd.phi_v[i, qx] * fd.dphi_v[j, qy],
                         d2tab[qx][i] * fd.phi_v[j, qy],
                         fd.dphi_v[i, qx] * fd.dphi_v[j, qy],
                         fd.phi_v[i, qx] * d2tab[qy][j]); drop_c=true)
                end
                lfd = fd4(fthev, x, y)
                maxfd = max(maxfd, abs(ld - lfd))
                maxfdC = max(maxfdC, abs(ldc - lfd))
                scale = max(scale, abs(lfd))
            end
        end
        @test scale > 0.0
        # measured: the production Laplacian agrees with the FD reference to
        # ~1e-9 relative; dropping C is an O(1) relative failure
        @test maxfd / scale < 1.0e-4
        @test maxfdC / scale > 0.1
        println("G269 FD Laplacian: full rel=", maxfd / scale,
                "  drop-C rel=", maxfdC / scale)

        # ---- 3. base-state second derivative: height_laplacian -----------
        maxhl = 0.0; scalehl = 0.0
        for lf in sq.leaves
            for qy in 1:fd.nq, qx in 1:fd.nq
                x, y = quad_point(leaf_metric(g, sq, lf, fd), sq, lf, fd,
                                  qx, qy)
                lh = height_laplacian(g, x, y)
                lfd = fd4((a, b) -> height(g, a, b), x, y)
                maxhl = max(maxhl, abs(lh - lfd))
                scalehl = max(scalehl, abs(lfd))
            end
        end
        @test scalehl > 0.0
        @test maxhl / scalehl < 1.0e-4
        println("G269 height_laplacian: rel=", maxhl / scalehl)

        # ---- 4. the EV field on CurvedBox is a real, wall-switched field --
        uxn = [g269_ux(c...) for c in sq.coords]
        uzn = [g269_uz(c...) for c in sq.coords]
        evg = MS269.semquad_ev_globals(g, sq, qt, pr0, thn, thn, uxn, uzn, 0.0)
        ev_out = zeros(Float64, length(sq.leaves))
        _, Knoev, _, _ = assemble_semquad_energy(sq, qt, pr0, uxn, uzn;
                                                 theta=thn)
        _, Kev, _, _ = assemble_semquad_energy(sq, qt, pr0, uxn, uzn;
                                               theta=thn, ev=1.0e3,
                                               ev_cmax=0.052, ev_alpha=2,
                                               ev_timestep=10, evg=evg,
                                               ev_out=ev_out)
        @test all(isfinite, ev_out)
        @test all(ev_out .>= 0.0)
        @test any(ev_out .> 0.0)          # interior leaves see the residual
        for (i, lf) in enumerate(sq.leaves)
            if lf[3] == 0 || lf[3] == sq.ny0 * (1 << lf[1]) - 1
                @test ev_out[i] == 0.0
            end
        end
        # the mapped EV stiffness is the base stiffness plus a real term
        @test norm(Kev - Knoev) > 10.0 * eps(norm(Knoev))
        # a mapped EV field from the standalone routine matches the assembled
        # `ev_out` (the two entry points must not drift)
        fld = MS269.semquad_ev_field(g, sq, qt, pr0, thn, uxn, uzn;
                                     dt=1.0e-2, ev=1.0e3, ev_cmax=0.052,
                                     ev_alpha=2, ev_timestep=10)
        @test g269_rel(fld, ev_out) <= 1.0e-14
        println("G269 EV field: min=", minimum(ev_out), " max=",
                maximum(ev_out), " nonzero leaves=", count(>(0.0), ev_out))
    end
end
