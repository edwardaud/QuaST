# G154: manufactured solutions for variable-viscosity Stokes and the energy
# equation (the MMS table that G151 started with constant eta / Poisson).
#
#   A. variable-viscosity Stokes, `solve_semquad_stokes_var`, bc=:rigid, :direct
#      A1  u=(2xy,-y^2), p=xy, eta=1+0.2xy            -> exact to round-off
#      A2  eta=1+0.5 sin(pi x/Lx) sin(pi y), trig u,p -> spectral in p
#   B. steady energy equation, `assemble_semquad_energy` + K+C reduced solve
#      B1  theta=y(1-y)x^2(x-Lx)^2, u=(2xy,-y^2)      -> exact to round-off
#      B2  theta=sin(pi y)cos(pi x/Lx)                -> spectral in p
#      B3  as B2 with u -> 20u (advection dominated)
#
# All norms are independent Gauss quadratures (p+6 points per direction), never
# dot(e, M e) and never the nodal interpolant.  The energy source convention is
#   S_i = int s phi_i,  s = u.grad(theta_ex) - lap(theta_ex) - src_base,
#   src_base = -u_z dTbar/dz,
# because `assemble_semquad_energy` already puts src_base into F; omitting the
# subtraction double-counts base heating.  Side walls are natural, so theta_ex
# must have zero normal derivative there.
#
# Originally prototyped in out_julia/agent/mms_g153_prototype.jl (MMS_REPORT.md).

using Test
using QuaST
using LinearAlgebra

const LX = 2.5                 # 5x2 forest (ASPECT 2b): [0,2.5] x [0,1]
const EA = 0.2                 # eta = 1 + EA x y  in [0.5, 1.5]

# ---------------------------------------------------------------------------
# quadrature / norm helpers
# ---------------------------------------------------------------------------

"""Nodal Q_p basis values and derivatives at `nq` Gauss points."""
function _basis(p::Int, nq::Int)
    rule = gll_rule(p)
    gq, gw = gauss_legendre(nq)
    n = p + 1
    V = zeros(Float64, n, nq)
    D = zeros(Float64, n, nq)
    for k in 1:nq
        L, dL = lagrange_eval(rule, gq[k])
        V[:, k] .= L
        D[:, k] .= dL
    end
    return V, D, gq, gw
end

"""
    _stokes_err(ss, u, pv, uex, pex; nq)

`(l2, p_l2, div_l2)`: true L2 velocity error, L2 pressure error with the
irrelevant constant removed (one pressure dof is pinned), and the L2 norm of the
discrete divergence (zero for a divergence-free exact solution).
"""
function _stokes_err(ss, u, pv, uex, pex; nq::Int=0)
    p = ss.p; q = ss.q
    nq == 0 && (nq = p + 6)
    su = ss.su; sp = ss.sp
    n = p + 1; m = q + 1
    Vu, Du, gq, gw = _basis(p, nq)
    Vp, _, _, _ = _basis(q, nq)
    nu = su.N
    area = 0.0; pmean = 0.0
    for lf in su.leaves
        hx = QuaST.leaf_hx(su, lf[1]); hy = QuaST.leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        pids = sp.elem[lf]
        pl = [pv[g] for g in pids]
        for qy in 1:nq, qx in 1:nq
            x = x0 + 0.5 * (gq[qx] + 1.0) * hx
            y = y0 + 0.5 * (gq[qy] + 1.0) * hy
            w = 0.25 * hx * hy * gw[qx] * gw[qy]
            ph = 0.0
            for b in 1:m, a in 1:m
                ph += pl[(b - 1) * m + a] * Vp[a, qx] * Vp[b, qy]
            end
            area += w
            pmean += w * (ph - pex(x, y))
        end
    end
    shift = pmean / area
    eu2 = 0.0; ep2 = 0.0; div2 = 0.0
    for lf in su.leaves
        hx = QuaST.leaf_hx(su, lf[1]); hy = QuaST.leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]; pids = sp.elem[lf]
        ul = [u[g] for g in uids]
        vl = [u[nu + g] for g in uids]
        pl = [pv[g] for g in pids]
        cx = 2.0 / hx; cy = 2.0 / hy
        for qy in 1:nq, qx in 1:nq
            x = x0 + 0.5 * (gq[qx] + 1.0) * hx
            y = y0 + 0.5 * (gq[qy] + 1.0) * hy
            w = 0.25 * hx * hy * gw[qx] * gw[qy]
            uhx = 0.0; uhz = 0.0; dxu = 0.0; dyv = 0.0; ph = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                uhx += ul[k] * Vu[i, qx] * Vu[j, qy]
                uhz += vl[k] * Vu[i, qx] * Vu[j, qy]
                dxu += ul[k] * Du[i, qx] * Vu[j, qy] * cx
                dyv += vl[k] * Vu[i, qx] * Du[j, qy] * cy
            end
            for b in 1:m, a in 1:m
                ph += pl[(b - 1) * m + a] * Vp[a, qx] * Vp[b, qy]
            end
            exh, eyh = uex(x, y)
            eu2 += w * ((uhx - exh)^2 + (uhz - eyh)^2)
            div2 += w * (dxu + dyv)^2
            ep2 += w * (ph - pex(x, y) - shift)^2
        end
    end
    return (l2=sqrt(eu2), p_l2=sqrt(ep2), div_l2=sqrt(div2))
end

"""L2 error and H1 seminorm error of a scalar Q_p field (exact gradient given)."""
function _scalar_err(sq, th, tex, gtex; nq::Int=0)
    p = sq.p
    nq == 0 && (nq = p + 6)
    n = p + 1
    V, D, gq, gw = _basis(p, nq)
    e2 = 0.0; eH1 = 0.0
    for lf in sq.leaves
        hx = QuaST.leaf_hx(sq, lf[1]); hy = QuaST.leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        tl = [th[g] for g in ids]
        for qy in 1:nq, qx in 1:nq
            x = x0 + 0.5 * (gq[qx] + 1.0) * hx
            y = y0 + 0.5 * (gq[qy] + 1.0) * hy
            w = 0.25 * hx * hy * gw[qx] * gw[qy]
            tq = 0.0; dx = 0.0; dy = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                tq += tl[k] * V[i, qx] * V[j, qy]
                dx += tl[k] * D[i, qx] * V[j, qy] * (2.0 / hx)
                dy += tl[k] * V[i, qx] * D[j, qy] * (2.0 / hy)
            end
            g = gtex(x, y)
            e2 += w * (tq - tex(x, y))^2
            eH1 += w * ((dx - g[1])^2 + (dy - g[2])^2)
        end
    end
    return (l2=sqrt(e2), h1=sqrt(eH1))
end

"""Load vector `S_i = int s(x,y) phi_i` with `nq` Gauss points (`p+2` = the
quadrature `assemble_semquad_energy` uses)."""
function _load(sq, s::Function; nq::Int=0)
    p = sq.p
    n = p + 1
    nq == 0 && (nq = p + 2)
    V, _, gq, gw = _basis(p, nq)
    b = zeros(Float64, sq.N)
    for lf in sq.leaves
        hx = QuaST.leaf_hx(sq, lf[1]); hy = QuaST.leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:nq, qx in 1:nq
            x = x0 + 0.5 * (gq[qx] + 1.0) * hx
            y = y0 + 0.5 * (gq[qy] + 1.0) * hy
            w = 0.25 * hx * hy * gw[qx] * gw[qy] * s(x, y)
            for j in 1:n, i in 1:n
                b[ids[(j - 1) * n + i]] += w * V[i, qx] * V[j, qy]
            end
        end
    end
    return b
end

"""Affine constraint of the energy solve: homogeneous Dirichlet on y=0,1,
natural (insulating) side walls, hanging nodes eliminated."""
function _energy_constraint(sq, qt, pr)
    fixed = Dict{Int,Float64}()
    for g in 1:sq.N
        y = sq.coords[g][2]
        y <= 1.0e-14 && (fixed[g] = pr.Twall_bot - QuaST.semquad_base_T(pr, y))
        y >= 1.0 - 1.0e-14 && (fixed[g] = pr.Twall_top - QuaST.semquad_base_T(pr, y))
    end
    cons = semquad_constraints(sq, qt)
    return QuaST._affine_constraint(sq.N, cons, fixed)
end

"""Steady energy solve `(K+C) theta = F + S` (the dt -> inf limit of
solve_semquad_energy)."""
function _energy_solve(sq, qt, pr, ux, uz, S)
    _, K, C, F = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                         advection=:galerkin)
    u0, Cfree, _ = _energy_constraint(sq, qt, pr)
    A = K + C
    rhs = F + S
    return u0 + Cfree * ((Cfree' * A * Cfree) \ (Cfree' * (rhs - A * u0)))
end

"""Relative discrete defect of a candidate field: `||Cfree'(A thI - (F+S))||`."""
function _defect(sq, qt, pr, ux, uz, S, thI)
    _, K, C, F = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                         advection=:galerkin)
    _, Cfree, _ = _energy_constraint(sq, qt, pr)
    return norm(Cfree' * ((K + C) * thI - (F + S))) / max(norm(Cfree' * (F + S)), 1.0e-300)
end

# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------

function _uniform(cx::Int, cy::Int, nref::Int)
    qt = Quadtree(nref; cells_x=cx, cells_y=cy)
    for _ in 1:nref, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    return qt
end

"""Uniform 5x2 tree with one refined (hence hanging) quarter."""
function _nonuniform(cx::Int, cy::Int, nref::Int)
    qt = Quadtree(nref + 1; cells_x=cx, cells_y=cy)
    for _ in 1:nref, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    refine!(qt, (nref, 0, 0))
    balance!(qt)
    return qt
end

# ---------------------------------------------------------------------------
# manufactured fields
# ---------------------------------------------------------------------------

# --- A1: polynomial, variable viscosity -----------------------------------
_a1_u = (x, y) -> (2.0 * x * y, -y^2)
_a1_p = (x, y) -> x * y
_a1_eta = (x, y) -> 1.0 + EA * x * y
# f = -div(2 eta eps(u)) + grad p  with  eta = 1 + EA x y
_a1_fx = (x, y) -> y - 4 * EA * y^2 - 2 * EA * x^2
_a1_fz = (x, y) -> 2.0 + 4 * EA * x * y + x

# --- A2: trig, variable viscosity -----------------------------------------
const _K = pi / LX                     # = pi / Lx
const _L = pi
const _AA = 0.5 * (_K^2 / _L - _L)     # eps_xz amplitude

_a2_u = (x, y) -> (sin(_K * x) * cos(_L * y), -(_K / _L) * cos(_K * x) * sin(_L * y))
_a2_p = (x, y) -> sin(_K * x) * sin(_L * y)
_a2_eta = (x, y) -> 1.0 + 0.5 * sin(_K * x) * sin(_L * y)
_a2_etax = (x, y) -> 0.5 * _K * cos(_K * x) * sin(_L * y)
_a2_etay = (x, y) -> 0.5 * _L * sin(_K * x) * cos(_L * y)

"""f = -div(2 eta eps(u)) + grad p for the trig family (verified against
8th-order finite differences to 1e-12)."""
function _a2_f(eta, etax, etay)
    fx = (x, y) -> begin
        s, c = sin(_K * x), cos(_K * x)
        S, C = sin(_L * y), cos(_L * y)
        e = eta(x, y); ex = etax(x, y); ey = etay(x, y)
        divs1 = 2 * _K * ex * c * C - 2 * _K^2 * e * s * C +
                2 * _AA * ey * s * S + 2 * _AA * _L * e * s * C
        -divs1 + _K * c * S
    end
    fz = (x, y) -> begin
        s, c = sin(_K * x), cos(_K * x)
        S, C = sin(_L * y), cos(_L * y)
        e = eta(x, y); ex = etax(x, y); ey = etay(x, y)
        divs2 = 2 * _AA * ex * s * S + 2 * _AA * _K * e * c * S -
                2 * _K * ey * c * C + 2 * _K * _L * e * c * S
        -divs2 + _L * s * C
    end
    return fx, fz
end

# --- B1: polynomial theta --------------------------------------------------
_b1_th = (x, y) -> y * (1.0 - y) * x^2 * (x - LX)^2
_b1_gth = (x, y) -> (y * (1.0 - y) * (4x^3 - 6LX * x^2 + 2LX^2 * x),
                     (1.0 - 2y) * x^2 * (x - LX)^2)
_b1_lap = (x, y) -> y * (1.0 - y) * (12x^2 - 12LX * x + 2LX^2) -
                    2.0 * x^2 * (x - LX)^2

# --- B2/B3: smooth theta ---------------------------------------------------
_b2_th = (x, y) -> sin(_L * y) * cos(_K * x)
_b2_gth = (x, y) -> (-_K * sin(_L * y) * sin(_K * x),
                     _L * cos(_L * y) * cos(_K * x))
_b2_lap = (x, y) -> -(_K^2 + _L^2) * _b2_th(x, y)

# divergence-free velocity used by all the energy cases (exactly representable
# in Q_p for p >= 2, so the advection matrix carries no interpolation error)


"""
    _energy_src(uxf, uzf, gth, lap) -> s

Extra source of the steady theta equation,
`s = u.grad(theta_ex) - lap(theta_ex) - src_base`, with `src_base = -u_z dTbar/dz
= u_z` for the conductive base state (H = 0, dTbar/dz = -1) that
`assemble_semquad_energy` already puts in `F`.  Subtracting it is essential:
the assembled system is `(K+C) theta = F + S`, so the *total* right-hand side
must be `int (u.grad(theta_ex) - lap(theta_ex)) phi`.
"""
_energy_src(uxf, uzf, gth, lap) = (x, y) ->
    uxf(x, y) * gth(x, y)[1] + uzf(x, y) * gth(x, y)[2] - lap(x, y) - uzf(x, y)

# ===========================================================================
@testset "G154 manufactured solutions: variable-eta Stokes and energy" begin
    pr = Params()                     # Ra=1e4, H=0, eta=1, Di=0, Twall 1/0
    uA = (x, y) -> _a1_u(x, y)[1]
    uB = (x, y) -> _a1_u(x, y)[2]

    # ---------------- A1: polynomial variable-eta Stokes -------------------
    # u in Q_2, p in Q_1, eta polynomial and the Gauss (p+2) rule exact, so the
    # discrete solution must equal the exact field to round-off on ANY 2:1 tree.
    for (label, qt, pmax) in (("uniform 5x2 L2", _uniform(5, 2, 2), 5),
                              ("uniform 5x2 L3", _uniform(5, 2, 3), 4),
                              ("nonuniform 5x2", _nonuniform(5, 2, 2), 5))
        for p in 2:pmax
            ss = make_semstokes(qt, p)
            u, pv, _, _, _ = solve_semquad_stokes_var(ss, qt, _a1_eta,
                                                      _a1_fx, _a1_fz,
                                                      uA, uB; bc=:rigid)
            nm = _stokes_err(ss, u, pv, _a1_u, _a1_p)
            @test nm.l2 < 1.0e-9
            @test nm.p_l2 < 1.0e-8
            @test nm.div_l2 < 1.0e-8
            @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-10
        end
    end

    # ---------------- A2: trig variable-eta Stokes, fixed 5x2 L2 ------------
    # Spectral decay in p; the exponential rate (not the algebraic h-rate) is
    # what the SEM buys, so require a large factor per order.
    qt2 = _uniform(5, 2, 2)
    errs = Float64[]; perrs = Float64[]
    for p in 2:5
        ss = make_semstokes(qt2, p)
        fx, fz = _a2_f(_a2_eta, _a2_etax, _a2_etay)
        u, pv, _, _, _ = solve_semquad_stokes_var(ss, qt2, _a2_eta, fx, fz,
                                                  (x, y) -> _a2_u(x, y)[1],
                                                  (x, y) -> _a2_u(x, y)[2];
                                                  bc=:rigid)
        nm = _stokes_err(ss, u, pv, _a2_u, _a2_p)
        push!(errs, nm.l2); push!(perrs, nm.p_l2)
        @test nm.div_l2 < 100 * nm.l2      # discrete divergence ~ velocity error
    end
    @test all(errs[k + 1] < errs[k] / 8 for k in 1:3)
    @test all(perrs[k + 1] < perrs[k] / 8 for k in 1:3)
    @test errs[end] < 1.0e-7
    @test perrs[end] < 1.0e-5

    # ---------------- B1: polynomial theta ---------------------------------
    # theta in Q_4, u in Q_2, s polynomial: exact to round-off.  `defect`
    # checks the assembly alone (no solve): theta_ex satisfies the discrete
    # equations, which pins the sign of both K and the base-state F.
    for (label, qt) in (("uniform 5x2 L2", _uniform(5, 2, 2)),
                        ("nonuniform 5x2", _nonuniform(5, 2, 2)))
        for p in (4, 5)
            sq = make_semquad(qt, p)
            ux = [uA(x, y) for (x, y) in sq.coords]
            uz = [uB(x, y) for (x, y) in sq.coords]
            s = _energy_src(uA, uB, _b1_gth, _b1_lap)
            S = _load(sq, s; nq=p + 2)
            th = _energy_solve(sq, qt, pr, ux, uz, S)
            nm = _scalar_err(sq, th, _b1_th, _b1_gth)
            thI = [_b1_th(x, y) for (x, y) in sq.coords]
            @test nm.l2 < 1.0e-11
            @test nm.h1 < 1.0e-10
            @test _defect(sq, qt, pr, ux, uz, S, thI) < 1.0e-9
        end
    end

    # ---------------- B2: smooth theta, p-convergence ----------------------
    for (label, qt, pmax) in (("uniform 5x2 L2", _uniform(5, 2, 2), 4),
                              ("uniform 5x2 L3", _uniform(5, 2, 3), 4))
        el2 = Float64[]; eh1 = Float64[]
        for p in 2:pmax
            sq = make_semquad(qt, p)
            ux = [uA(x, y) for (x, y) in sq.coords]
            uz = [uB(x, y) for (x, y) in sq.coords]
            s = _energy_src(uA, uB, _b2_gth, _b2_lap)
            S = _load(sq, s; nq=p + 2)
            th = _energy_solve(sq, qt, pr, ux, uz, S)
            nm = _scalar_err(sq, th, _b2_th, _b2_gth)
            push!(el2, nm.l2); push!(eh1, nm.h1)
        end
        @test all(el2[k + 1] < el2[k] / 8 for k in 1:(length(el2) - 1))
        @test all(eh1[k + 1] < eh1[k] / 5 for k in 1:(length(eh1) - 1))
        @test el2[end] < 1.0e-5
    end

    # ---------------- B3: advection dominated ------------------------------
    # |u| ~ 49 with h = 1/8 gives a cell Peclet number ~6: the Galerkin solution
    # must still converge spectrally (no oscillation-driven stall).
    qtb = _uniform(5, 2, 2)
    uxb = (x, y) -> 20.0 * uA(x, y)
    uzb = (x, y) -> 20.0 * uB(x, y)
    el2 = Float64[]
    for p in 2:4
        sq = make_semquad(qtb, p)
        ux = [uxb(x, y) for (x, y) in sq.coords]
        uz = [uzb(x, y) for (x, y) in sq.coords]
        s = _energy_src(uxb, uzb, _b2_gth, _b2_lap)
        S = _load(sq, s; nq=p + 2)
        th = _energy_solve(sq, qtb, pr, ux, uz, S)
        push!(el2, _scalar_err(sq, th, _b2_th, _b2_gth).l2)
    end
    @test all(el2[k + 1] < el2[k] / 8 for k in 1:2)
    @test el2[end] < 1.0e-5
end
