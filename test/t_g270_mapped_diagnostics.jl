# G270: diagnostics and wall integrals on the mapped geometry (stage 0, cut 4).
#
# THE CONTRACT OF THIS GATE
#
# Cut 4 routes every per-step diagnostic through the geometry layer
# (GEOM_DESIGN §3 cut 4, §5.5/G270): the wall Nusselt pair, the wall normal
# derivative, the surface velocity, Vrms, mean T, the energy Nusselt, the
# work/dissipation pair, the base-advection and theta^2 div u integrals, the max
# divergence, the heat budget (and its mapped latent/adiabatic terms), the
# consistent-boundary-flux pair, the exact residual and the level stats, plus
# the hmin/CFL sizes.  This gate proves that routing:
#
# (a) **Affine exactness and invariance.**
#     (a1) On a rotated + translated `AffineBox` the *same physical problem* as
#          the box (the box state rotated by the rigid map) gives the same
#          physical diagnostics as the box computation: every diagnostic in the
#          list above agrees with its `CartesianBox` value at 1e-12 relative.
#          A rotation preserves norms, the height `h = T`, the normal derivative
#          `grad(theta).n` and every wall length, so the two computations are two
#          representations of one problem.  An implementation that used the
#          physical `y` derivative, a logical face weight or `u_z` instead of
#          `u . grad h` fails here by O(1): the rotation makes the physical
#          direction differ from the logical one.
#     (a2) On a **general** affine map (rotation + shear + anisotropic scale,
#          det A = 1.17 so the physical normal/measure differ from the coordinate
#          ones) the volume integrals, wall fluxes and Nusselt numbers equal an
#          **independent high-order quadrature of the analytic fields**: the gate
#          writes the fields as polynomials in the logical coordinates, composes
#          them with the map, and integrates in logical coordinates with an
#          8-point Gauss rule (exact for these polynomials) carrying the analytic
#          `|det A| Lx` measure and the analytic normal wall length.  The
#          production diagnostics use the GLL nodal interpolation and their own
#          quadrature, so a wrong measure, normal, height argument or
#          `J^{-T}` multiplication is an O(1) failure.
#     (a3) The same physical parallelogram in a different logical frame
#          (A' = A*M with M the orientation-preserving square automorphism) gives
#          the same physical diagnostics at 1e-12: the metric enters
#          consistently.
#
# (b) **Mapped-vs-box cross-check on the box map** for every diagnostic at
#     1e-14 relative (the design's bound; measured below it) on a dyadic unit
#     tree, a non-dyadic `Lx = 7/3` tree and a hanging tree: wall fluxes, Nu
#     (pointwise and CBF), Vrms, mean T, energy Nu, the work/dissipation pair,
#     the surface velocities, the heat budget (with latent phases, internal
#     heating and adiabatic heating on), the base-advection and theta^2 div u
#     integrals, the max divergence, the hmin/CFL sizes, the level stats, the
#     exact residual and the history `closure` column.  The box bodies are *not*
#     routed through the geometry (that is what keeps the nine state-dump
#     hashes); the mapped bodies are new code, so the comparison is between two
#     independent implementations.
#
# (c) **A wrong normal / wrong face Jacobian is caught at O(1).**  On the
#     general affine map the gate computes the correct analytic wall flux and two
#     deliberately wrong reconstructions: the physical-y derivative instead of
#     `grad(theta).n`, and the logical wall measure instead of the physical face
#     Jacobian.  The measured differences are printed and asserted to be O(20%)
#     against the correct value; (a2) pins the production value to that correct
#     integral, so this is the measured failure of the bug class, not an
#     unasserted comment.
#
# (d) **End-to-end mapped driver.**  `run_semquad_simulation` on `AffineBox` and
#     `CurvedBox` (fixed mesh, `linear=:direct`, Picard) completes a few steps
#     with heating off and on (internal heating `rho_heating` and adiabatic
#     heating), and with the CBF / exact-residual / latent diagnostic columns
#     switched on; the history row is the mapped diagnostics (it equals direct
#     mapped calls on the final state), the mass-flux identities hold (the
#     discrete continuity gives `int div u = 0`, and the divergence theorem on
#     the discrete field), and the time-step CBF energy-budget identity (ID1)
#     closes to the linear solver's floor:
#
#       Qc = dH/dt + sum(C theta) + Ibase - heating - int lap(Tbar(h)) dV,
#
#     with `Qc` the full step residual summed over the Dirichlet wall nodes and
#     `Ibase` density-weighted where the card weights the mass/advection by
# `rho = 1-T` (the cut-3 mapped source). The
#     `int lap(Tbar(h))` term is the mapped base-state curvature the box hides
#     (`|grad h| = 1`, `lap y = 0`); it is an O(1) term on `CurvedBox`, and the
#     identity closing there is what proved the cut-3 mapped-source defect
#     documented in GEOM_PROGRESS.md (B1).
#
# Every assertion states what wrong implementation it catches.  No assertion
# compares a quantity against itself: (a) uses the analytic fields and an
# independent quadrature (or the *other* geometry's untouched box bodies), (b)
# compares two independent kernel bodies, (c) compares against the correct
# analytic values, and (d) checks algebraic identities of the assembled step.

using Test, QuaST, LinearAlgebra
const MS270 = QuaST

# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

function g270_tree(level::Int; Lx::Float64=1.0)
    qt = Quadtree(level; Lx=Lx)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function g270_tree_cells(level::Int; cx::Int=7, cy::Int=3)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function g270_hanging_tree()
    qt = Quadtree(2)
    for lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for lf in sort(collect(qt.leaves))
        lf[2] < 2 && lf[3] < 2 && refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# ---------------------------------------------------------------------------
# Nodal values of analytic fields on a space.
#
# The value at node (i,j) of leaf lf is the function evaluated at the node's
# **logical** coordinates, computed from the canonical node identity (cut 1), so
# the box and the mapped space on the same tree receive bit-identical nodal
# values for the same function.  The physical fields are then the map images:
# a scalar is the same function of the logical point; a velocity is the map's
# linear part applied in the logical frame (`u_phys = A*(uS,uT)` for the affine
# geometries, `A = I` on the box map).
# ---------------------------------------------------------------------------

function g270_fill_scalar!(vals::Vector{Float64}, ss::MS270.SemStokes,
                           qt::Quadtree, f)
    sq = ss.su
    n = sq.p + 1
    for lf in sq.leaves
        l = lf[1]
        nx = qt.nx0 * (1 << l)
        ny = qt.ny0 * (1 << l)
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            S = (lf[2] + 0.5 * (sq.xi[i] + 1.0)) / nx
            T = (lf[3] + 0.5 * (sq.xi[j] + 1.0)) / ny
            vals[ids[(j - 1) * n + i]] = f(S, T)
        end
    end
    return vals
end

function g270_fill_affine_velocity!(ux::Vector{Float64}, uz::Vector{Float64},
                                    ss::MS270.SemStokes, qt::Quadtree,
                                    g::AffineBox, uSfun, uTfun)
    sq = ss.su
    n = sq.p + 1
    A = g.A
    for lf in sq.leaves
        l = lf[1]
        nx = qt.nx0 * (1 << l)
        ny = qt.ny0 * (1 << l)
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            S = (lf[2] + 0.5 * (sq.xi[i] + 1.0)) / nx
            T = (lf[3] + 0.5 * (sq.xi[j] + 1.0)) / ny
            a = uSfun(S, T)
            b = uTfun(S, T)
            k = ids[(j - 1) * n + i]
            ux[k] = A[1] * a + A[2] * b
            uz[k] = A[3] * a + A[4] * b
        end
    end
    return ux, uz
end

# ---------------------------------------------------------------------------
# Analytic fields (written here from their definitions; not read from the
# solver).  Degree <= 2 in the logical coordinates, so every affine mapped space
# represents them exactly (an affine map makes them physical P2 polynomials).
# ---------------------------------------------------------------------------

g270_th(S, T) = 0.30 + 0.40 * S - 0.20 * T + 0.15 * S^2 + 0.10 * S * T - 0.05 * T^2
g270_th_S(S, T) = 0.40 + 0.30 * S + 0.10 * T
g270_th_T(S, T) = -0.20 + 0.10 * S - 0.10 * T

g270_uS(S, T) = 0.10 + 0.20 * S - 0.05 * T + 0.04 * S^2 + 0.03 * S * T + 0.02 * T^2
g270_uT(S, T) = -0.06 + 0.10 * T + 0.05 * S - 0.02 * S^2 + 0.04 * S * T - 0.03 * T^2
g270_uS_S(S, T) = 0.20 + 0.08 * S + 0.03 * T
g270_uS_T(S, T) = -0.05 + 0.03 * S + 0.04 * T
g270_uT_S(S, T) = 0.05 - 0.04 * S + 0.04 * T
g270_uT_T(S, T) = 0.10 + 0.04 * S - 0.06 * T

g270_eta(x, y) = 1.0 + 0.10 * x + 0.05 * y

g270_phase_list() = [PhaseTransition(0.60, 0.10, -2.0, 0.50, 50.0, 2.0)]

function g270_pr(; H::Float64=1.0, rho_heating::Bool=false,
                 Di::Float64=0.0, heating_adiabatic::Bool=false,
                 heating_dissipation::Bool=false, Ra::Float64=1.0e3,
                 latent_L::Float64=0.0, deviatoric::Bool=false,
                 wall_bc::NTuple{4,Symbol}=(:same, :same, :same, :same))
    return Params(Ra=Ra, H=H, eta_law=:expT, deta_T=1.0, bc=:rigid,
                  rho_heating=rho_heating, Di=Di,
                  heating_adiabatic=heating_adiabatic,
                  heating_dissipation=heating_dissipation,
                  latent_L=latent_L, deviatoric=deviatoric,
                  wall_bc=wall_bc)
end

# ---------------------------------------------------------------------------
# Independent high-order quadrature of the analytic fields on an AffineBox.
#
# Map: (x,y) = A*(Lx*S, T) + b; the fields are polynomials in (S,T).  An 8-point
# Gauss rule (exact to degree 15 per direction, far above the degree-4
# integrands) is applied in *logical* coordinates with the analytic measure
# |det A| Lx dS dT and the analytic face length Lx*|A e_1| dS, so it never calls
# the diagnostics' quadrature or the assembled matrices.
# ---------------------------------------------------------------------------

const G270_GL = gauss_legendre(8)

"""
    g270_affine_data(g)

Analytic data of the affine map: `grad` is the 2x2 physical-gradient operator
`(B11,B12,B21,B22)` with `grad_x f = B*[f_S; f_T]`; `detJg = |det A| Lx`; the
T=const wall length `Lx*|A e_1|`; the outward unit normal of the T=1 face; and
`gh = grad_x T` (the physical height gradient, since `height = T`).
"""
function g270_affine_data(g::AffineBox)
    A = collect(g.A)
    d = A[1] * A[4] - A[2] * A[3]
    i11 = A[4] / d
    i12 = -A[2] / d
    i21 = -A[3] / d
    i22 = A[1] / d
    Lx = g.Lx
    grad = (i11 / Lx, i21, i12 / Lx, i22)
    wall_len = hypot(A[1], A[3]) * Lx
    nrm = hypot(i21, i22)
    n_top = (i21 / nrm, i22 / nrm)
    return (grad=grad, detJg=abs(d) * Lx, wall_len=wall_len, n_top=n_top,
            gh=(i21, i22))
end

function g270_vol(g::AffineBox, qt::Quadtree, f)
    ad = g270_affine_data(g)
    gx, gw = G270_GL
    s = 0.0
    for lf in qt.leaves
        l = lf[1]
        nx = qt.nx0 * (1 << l)
        ny = qt.ny0 * (1 << l)
        S0 = lf[2] / nx; S1 = (lf[2] + 1) / nx
        T0 = lf[3] / ny; T1 = (lf[3] + 1) / ny
        hs = (S1 - S0) / 2; ht = (T1 - T0) / 2
        for qT in 1:8, qS in 1:8
            S = S0 + hs * (gx[qS] + 1.0)
            T = T0 + ht * (gx[qT] + 1.0)
            s += gw[qS] * gw[qT] * hs * ht * f(S, T)
        end
    end
    return ad.detJg * s
end

function g270_top_face(g::AffineBox, qt::Quadtree, f)
    ad = g270_affine_data(g)
    gx, gw = G270_GL
    s = 0.0
    for lf in qt.leaves
        l = lf[1]
        nx = qt.nx0 * (1 << l)
        lf[3] == qt.ny0 * (1 << l) - 1 || continue
        S0 = lf[2] / nx; S1 = (lf[2] + 1) / nx
        hs = (S1 - S0) / 2
        for qS in 1:8
            S = S0 + hs * (gx[qS] + 1.0)
            s += gw[qS] * hs * f(S)
        end
    end
    return ad.wall_len * s
end

g270_grad(ad, fS, fT) = (ad.grad[1] * fS + ad.grad[2] * fT,
                         ad.grad[3] * fS + ad.grad[4] * fT)

# physical velocity gradients of the analytic field (u_phys = A*(uS,uT)) at (S,T)
function g270_vel_grads(ad, A, S, T)
    (sSx, sSy) = g270_grad(ad, g270_uS_S(S, T), g270_uS_T(S, T))
    (tSx, tSy) = g270_grad(ad, g270_uT_S(S, T), g270_uT_T(S, T))
    ux_x = A[1] * sSx + A[2] * tSx
    ux_y = A[1] * sSy + A[2] * tSy
    uz_x = A[3] * sSx + A[4] * tSx
    uz_y = A[3] * sSy + A[4] * tSy
    return (ux_x, ux_y, uz_x, uz_y)
end

# The wall RMS is defined by the layer's own wall rule (the composite GLL rule
# the box wall body uses; the GLL rule is degree-3 exact and |u|^2 is degree 4,
# so it is *not* the exact integral — the box has the same property).  This
# gate-side sum applies that rule to the analytic field with the analytic face
# length, so the production value must reproduce it up to round-off.
function g270_top_face_gll(g::AffineBox, qt::Quadtree, f; p::Int=2)
    ad = g270_affine_data(g)
    rule = gll_rule(p)
    num = 0.0
    den = 0.0
    for lf in qt.leaves
        l = lf[1]
        nx = qt.nx0 * (1 << l)
        lf[3] == qt.ny0 * (1 << l) - 1 || continue
        S0 = lf[2] / nx; S1 = (lf[2] + 1) / nx
        hs = (S1 - S0) / 2
        jac = ad.wall_len * hs
        for i in 1:p+1
            S = S0 + hs * (rule.nodes[i] + 1.0)
            w = jac * rule.weights[i]
            num += w * f(S)
            den += w
        end
    end
    return sqrt(num / den)
end

# analytic values of the diagnostics on a general affine map
function g270_analytic(g::AffineBox, qt::Quadtree, pr::Params)
    ad = g270_affine_data(g)
    A = g.A
    area = ad.detJg
    ux = (S, T) -> A[1] * g270_uS(S, T) + A[2] * g270_uT(S, T)
    uz = (S, T) -> A[3] * g270_uS(S, T) + A[4] * g270_uT(S, T)
    ugh = (S, T) -> ux(S, T) * ad.gh[1] + uz(S, T) * ad.gh[2]
    divu = function (S, T)
        gr = g270_vel_grads(ad, A, S, T)
        return gr[1] + gr[4]
    end
    eed = function (S, T)
        gr = g270_vel_grads(ad, A, S, T)
        exz = 0.5 * (gr[2] + gr[3])
        e = gr[1]^2 + gr[4]^2 + 2.0 * exz^2
        pr.deviatoric && (e -= (1.0 / 3.0) * (gr[1] + gr[4])^2)
        return e
    end
    Tfull = (S, T) -> Tbar(T, pr.H) + g270_th(S, T)
    # physical coordinates of the logical point, for the eta argument
    etaq = function (S, T)
        x = A[1] * (g.Lx * S) + A[2] * T + g.b[1]
        y = A[3] * (g.Lx * S) + A[4] * T + g.b[2]
        return g270_eta(x, y)
    end
    vrms = sqrt(g270_vol(g, qt, (S, T) -> ux(S, T)^2 + uz(S, T)^2) / area)
    meanT = g270_vol(g, qt, Tfull) / area
    w_int = g270_vol(g, qt, (S, T) -> Tfull(S, T) * ugh(S, T))
    u_int = g270_vol(g, qt, (S, T) -> 2.0 * etaq(S, T) * eed(S, T))
    base_adv = g270_vol(g, qt, (S, T) -> ugh(S, T) * dTbar(T, pr.H))
    th2div = g270_vol(g, qt, (S, T) -> g270_th(S, T)^2 * divu(S, T))
    nu_en = 1.0 + 0.5 * pr.H +
            g270_vol(g, qt, (S, T) -> ugh(S, T) * g270_th(S, T)) / area
    dthdn = function (S, T)
        (gx, gy) = g270_grad(ad, g270_th_S(S, T), g270_th_T(S, T))
        return gx * ad.n_top[1] + gy * ad.n_top[2]
    end
    flux_top = g270_top_face(g, qt, (S) -> dthdn(S, 1.0))
    # the bottom wall's "upward" flux is +grad(theta).n_top along T = 0 (the
    # outward normal there is -n_top); for a sheared map the cross metric term
    # makes it differ from the top value, so it is evaluated at T = 0.
    flux_bot = g270_top_face(g, qt, (S) -> dthdn(S, 0.0))
    top_ref = -semquad_base_gradient(pr, 1.0)
    bot_ref = -semquad_base_gradient(pr, 0.0)
    nu_top = top_ref - flux_top / ad.wall_len
    nu_bot = bot_ref - flux_bot / ad.wall_len
    surf = g270_top_face_gll(g, qt, (S) -> begin
                                 T = 1.0
                                 ux(S, T)^2 + uz(S, T)^2
                             end)
    heating = g270_vol(g, qt, (S, T) -> begin
                            h = 0.0
                            pr.rho_heating && (h -= Tfull(S, T) * pr.H)
                            pr.Di > 0.0 && pr.heating_adiabatic &&
                                (h -= pr.Di * Tfull(S, T) * ugh(S, T))
                            h
                        end)
    return (vrms=vrms, meanT=meanT, W=w_int / area, U=u_int / area,
            base_adv=base_adv, theta2divu=th2div, nu_energy=nu_en,
            flux_top=flux_top, flux_bot=flux_bot, nu_top=nu_top,
            nu_bot=nu_bot, surface=surf, area=area, wall_len=ad.wall_len,
            theta_int=g270_vol(g, qt, (S, T) -> g270_th(S, T)),
            heating=heating, ugh=ugh, dthdn=dthdn)
end

g270_rel(a, b) = abs(a - b) / max(abs(a), abs(b), 1.0e-300)

# ---------------------------------------------------------------------------
# (a) Affine exactness and invariance
# ---------------------------------------------------------------------------

@testset "G270 (a) affine diagnostics exactness and invariance" begin
    rot = 0.55
    Arot = (cos(rot), -sin(rot), sin(rot), cos(rot))
    shift = (0.30, -0.20)
    p = 2
    qt = g270_tree(2)
    Lx = 1.0
    box = CartesianBox(Lx)

    # ---------------- (a1) rotated/translated frame = box ---------------
    pr = g270_pr(H=1.0, rho_heating=true, Di=1.0, heating_adiabatic=true,
                 deviatoric=true)
    phases = PhaseTransition[]
    Arotg = AffineBox(Arot, shift, Lx)
    ssb = make_semstokes(qt, p; geom=box)
    ssr = make_semstokes(qt, p; geom=Arotg)
    thb = zeros(Float64, ssb.su.N); g270_fill_scalar!(thb, ssb, qt, g270_th)
    thr = zeros(Float64, ssr.su.N); g270_fill_scalar!(thr, ssr, qt, g270_th)
    uxb = zeros(Float64, ssb.su.N); uzb = zeros(Float64, ssb.su.N)
    g270_fill_affine_velocity!(uxb, uzb, ssb, qt, AffineBox(), g270_uS, g270_uT)
    uxr = zeros(Float64, ssr.su.N); uzr = zeros(Float64, ssr.su.N)
    g270_fill_affine_velocity!(uxr, uzr, ssr, qt, Arotg, g270_uS, g270_uT)
    @test thr == thb
    # the rotated physical velocity is R*u_box, not u_box: the diagnostics must
    # be invariant under the rotation, the arrays are not.
    @test uxr != uxb || uzr != uzb

    lb = semquad_heat_budget(ssb.su, qt, pr, thb; phases=phases, ux=uxb, uz=uzb,
                             eta_fun=g270_eta)
    lr = semquad_heat_budget(ssr.su, qt, pr, thr; phases=phases, ux=uxr, uz=uzr,
                             eta_fun=g270_eta)
    bb = semquad_work_dissipation(ssb.su, qt, pr, thb, uxb, uzb, (x, y) -> 1.0)
    br = semquad_work_dissipation(ssr.su, qt, pr, thr, uxr, uzr, (x, y) -> 1.0)
    nb = semquad_nu_walls(ssb.su, qt, thb, pr)
    nr = semquad_nu_walls(ssr.su, qt, thr, pr)
    cb = (semquad_cbf_theta_flux(ssb.su, qt, pr, thb, uxb, uzb; wall=:top),
          semquad_cbf_theta_flux(ssb.su, qt, pr, thb, uxb, uzb; wall=:bottom))
    cr = (semquad_cbf_theta_flux(ssr.su, qt, pr, thr, uxr, uzr; wall=:top),
          semquad_cbf_theta_flux(ssr.su, qt, pr, thr, uxr, uzr; wall=:bottom))
    ncb = semquad_nu_cbf(ssb.su, qt, pr, thb, uxb, uzb)
    ncr = semquad_nu_cbf(ssr.su, qt, pr, thr, uxr, uzr)
    eb = semquad_exact_residual(ssb.su, qt, pr, thb, thb, thb, uxb, uzb,
                                1.0e-4, 1.0e-4, false, nothing; advection=:skew)
    er = semquad_exact_residual(ssr.su, qt, pr, thr, thr, thr, uxr, uzr,
                                1.0e-4, 1.0e-4, false, nothing; advection=:skew)
    sb = semquad_surface_velocity(ssb.su, qt, uxb, uzb)
    sr = semquad_surface_velocity(ssr.su, qt, uxr, uzr)
    pairs = [
        ("Vrms", semquad_vrms(ssb.su, qt, uxb, uzb),
         semquad_vrms(ssr.su, qt, uxr, uzr)),
        ("meanT", semquad_mean_T(ssb.su, qt, thb, pr),
         semquad_mean_T(ssr.su, qt, thr, pr)),
        ("nu_energy", semquad_nu_energy(ssb.su, qt, pr, thb, uxb, uzb),
         semquad_nu_energy(ssr.su, qt, pr, thr, uxr, uzr)),
        ("W", bb[1], br[1]), ("U", bb[2], br[2]),
        ("BaseAdv", semquad_base_advection_integral(ssb.su, qt, pr, uxb, uzb),
         semquad_base_advection_integral(ssr.su, qt, pr, uxr, uzr)),
        ("theta2divu", semquad_theta2_divu_integral(ssb.su, qt, thb, uxb, uzb),
         semquad_theta2_divu_integral(ssr.su, qt, thr, uxr, uzr)),
        ("divmax", semquad_max_divergence(ssb.su, qt, uxb, uzb),
         semquad_max_divergence(ssr.su, qt, uxr, uzr)),
        ("Nu_top", nb[1], nr[1]), ("Nu_bot", nb[2], nr[2]),
        ("NuCbf_top", ncb[1], ncr[1]), ("NuCbf_bot", ncb[2], ncr[2]),
        ("cbf_top", cb[1], cr[1]), ("cbf_bot", cb[2], cr[2]),
        ("surf_rms", sb[1], sr[1]), ("surf_max", sb[2], sr[2]),
        ("theta_int", lb.theta_int, lr.theta_int),
        ("flux_top", lb.flux_top, lr.flux_top),
        ("flux_bot", lb.flux_bot, lr.flux_bot),
        ("flux", lb.flux, lr.flux),
        ("heating", lb.heating, lr.heating),
        ("heat_base_adv", lb.base_adv, lr.base_adv),
        ("exact_R", eb.R, er.R),
        ("exact_flux", eb.flux, er.flux),
        ("exact_rhs", eb.rhs, er.rhs),
        ("exact_area", eb.area, er.area),
    ]
    for (name, a, b) in pairs
        @test g270_rel(a, b) < 1.0e-12
        println("G270 a1 ", rpad(name, 14), " box=", a, " mapped=", b,
                " rel=", g270_rel(a, b))
    end
    # non-tautology: the expectation is the *untouched box body* on the box
    # (a different implementation), and the rotation makes the physical normal,
    # the y-direction and the logical measure all differ from their box values,
    # so a mapped body that used any of them instead of the mapped metric fails
    # this comparison O(1) (see (c) for the measured size of that failure).

    # ---------------- (a2) general affine map vs analytic integrals --------
    A = (1.30, 0.40, -0.30, 0.90)          # det A = 1.17
    g = AffineBox(A, (0.30, -0.20), 1.0)
    pr2 = g270_pr(H=1.0, rho_heating=true, Di=1.0, heating_adiabatic=true,
                  deviatoric=true)
    ss = make_semstokes(qt, p; geom=g)
    th = zeros(Float64, ss.su.N); g270_fill_scalar!(th, ss, qt, g270_th)
    ux = zeros(Float64, ss.su.N); uz = zeros(Float64, ss.su.N)
    g270_fill_affine_velocity!(ux, uz, ss, qt, g, g270_uS, g270_uT)
    an = g270_analytic(g, qt, pr2)
    lb = semquad_heat_budget(ss.su, qt, pr2, th; ux=ux, uz=uz, eta_fun=g270_eta)
    W, U = semquad_work_dissipation(ss.su, qt, pr2, th, ux, uz, g270_eta)
    nt, nbot = semquad_nu_walls(ss.su, qt, th, pr2)
    sur_rms, sur_max = semquad_surface_velocity(ss.su, qt, ux, uz; wall=:top)
    vals = [
        ("Vrms", semquad_vrms(ss.su, qt, ux, uz), an.vrms),
        ("meanT", semquad_mean_T(ss.su, qt, th, pr2), an.meanT),
        ("nu_energy", semquad_nu_energy(ss.su, qt, pr2, th, ux, uz),
         an.nu_energy),
        ("W", W, an.W), ("U", U, an.U),
        ("BaseAdv", semquad_base_advection_integral(ss.su, qt, pr2, ux, uz),
         an.base_adv),
        ("theta2divu", semquad_theta2_divu_integral(ss.su, qt, th, ux, uz),
         an.theta2divu),
        ("Nu_top", nt, an.nu_top), ("Nu_bot", nbot, an.nu_bot),
        ("flux_top", lb.flux_top, an.flux_top),
        ("flux_bot", lb.flux_bot, an.flux_bot),
        ("theta_int", lb.theta_int, an.theta_int),
        ("heating", lb.heating, an.heating),
        ("heat_base_adv", lb.base_adv, an.base_adv),
        ("surf_rms", sur_rms, an.surface),
    ]
    for (name, got, want) in vals
        @test g270_rel(got, want) < 1.0e-12
        println("G270 a2 ", rpad(name, 14), " diag=", got, " analytic=", want,
                " rel=", g270_rel(got, want))
    end
    # the design's 1e-10 bound for the wall-flux densities is met with margin
    @test abs(nt - an.nu_top) / abs(an.nu_top) < 1.0e-10
    @test abs(nbot - an.nu_bot) / abs(an.nu_bot) < 1.0e-10
    # the eta_ref interior-band mean reads the band in physical HEIGHT
    # (`driver/eta_ref_report.jl`, cut 4): on this affine map height = T, so the
    # analytic banded mean is int_{0.25<=T<=0.75} eta dV / int dV, integrated by
    # the same gate-side rule (the band edges are leaf edges, so no partial leaf).
    zlo = 0.25
    zhi = 0.75
    eta_band = function (S, T)
        x = A[1] * (g.Lx * S) + A[2] * T + g.b[1]
        y = A[3] * (g.Lx * S) + A[4] * T + g.b[2]
        return (zlo <= T <= zhi) ? g270_eta(x, y) : 0.0
    end
    inband = function (S, T)
        return (zlo <= T <= zhi) ? 1.0 : 0.0
    end
    band_analytic = g270_vol(g, qt, eta_band) / g270_vol(g, qt, inband)
    band_diag = QuaST._eta_interior_band_mean(ss.su, qt, g270_eta; zlo=zlo,
                                              zhi=zhi)
    @test g270_rel(band_diag, band_analytic) < 1.0e-12
    println("G270 a2 eta_band        diag=", band_diag, " analytic=",
            band_analytic, " rel=", g270_rel(band_diag, band_analytic))
    # non-tautology: the analytic fields are polynomials written in this file,
    # integrated by an 8-point Gauss rule in logical coordinates with the
    # *analytic* |det A| Lx measure and Lx|A e_1| face length; the production
    # values come from the GLL nodal interpolation and the layer's own metric.
    # A wrong J^{-T}, measure, height argument or normal is O(1) here.

    # ---------------- (a3) another logical frame of the same domain --------
    # The orientation-preserving automorphism M(S,T) = (1-T, S) maps the
    # logical square to itself; the same physical parallelogram is then
    # (x,y) = A*(Lx*S, T) + b with (Lx*S, T) = U*(S',T') + (Lx, 0) and
    # U = [[0,-Lx],[1,0]] (det U = Lx > 0).  Hence A2 = A*U, b2 = A*(Lx,0)+b.
    Lx = g.Lx
    Uf = (0.0, -Lx, 1.0, 0.0)              # columns: Uf e_S' = (0,1), Uf e_T' = (-Lx,0)
    A2 = (A[1] * Uf[1] + A[2] * Uf[3], A[1] * Uf[2] + A[2] * Uf[4],
          A[3] * Uf[1] + A[4] * Uf[3], A[3] * Uf[2] + A[4] * Uf[4])
    b2 = (A[1] * Lx + g.b[1], A[3] * Lx + g.b[2])
    g2 = AffineBox(A2, b2, Lx)
    MS(S, T) = (1.0 - T, S)
    th2fun(S, T) = g270_th(MS(S, T)...)
    # the physical velocity is frame invariant: A2*(uS2,uT2) = A*(uS,uT), i.e.
    # (uS2,uT2) = U^{-1} (uS,uT)(M(S,T)); U^{-1} = [[0,1],[-1/Lx,0]].
    uS2(S, T) = g270_uT(MS(S, T)...)
    uT2(S, T) = -g270_uS(MS(S, T)...) / Lx
    ss2 = make_semstokes(qt, p; geom=g2)
    th2 = zeros(Float64, ss2.su.N); g270_fill_scalar!(th2, ss2, qt, th2fun)
    ux2 = zeros(Float64, ss2.su.N); uz2 = zeros(Float64, ss2.su.N)
    g270_fill_affine_velocity!(ux2, uz2, ss2, qt, g2, uS2, uT2)
    lb2 = semquad_heat_budget(ss2.su, qt, pr2, th2; ux=ux2, uz=uz2,
                              eta_fun=g270_eta)
    # The geometry *defines* the height direction (height = logical T), so a
    # logical frame that permutes S and T changes the physical meaning of the
    # vertical: only the diagnostics that do not read `height` are frame
    # invariants.  The height/direction-dependent ones (mean T, W, base
    # advection, energy Nu, heating) are covered by (a1)/(a2), not here.
    frame = [
        ("Vrms", semquad_vrms(ss.su, qt, ux, uz),
         semquad_vrms(ss2.su, qt, ux2, uz2)),
        ("theta2divu", semquad_theta2_divu_integral(ss.su, qt, th, ux, uz),
         semquad_theta2_divu_integral(ss2.su, qt, th2, ux2, uz2)),
        ("theta_int", lb.theta_int, lb2.theta_int),
        ("U", U, semquad_work_dissipation(ss2.su, qt, pr2, th2, ux2, uz2,
                                          g270_eta)[2]),
    ]
    for (name, a1, a2) in frame
        @test g270_rel(a1, a2) < 1.0e-12
        println("G270 a3 ", rpad(name, 10), " frame1=", a1, " frame2=", a2,
                " rel=", g270_rel(a1, a2))
    end
    # non-tautology: the two frames are the same physical parallelogram but
    # different logical coordinates; a metric that forgot the frame map (a
    # logical measure, a fixed-axis direction) gives different values for these
    # volume integrals.
end

# ---------------------------------------------------------------------------
# Cross-check helper: the mapped bodies fed the box map against the untouched
# box bodies on the same tree and the same nodal state.
# ---------------------------------------------------------------------------

function g270_check_pair(name, a, b, tol)
    rel = g270_rel(a, b)
    @test rel < tol
    println("G270 b ", rpad(name, 18), " box=", a, " mapped=", b, " rel=", rel)
    return rel
end

function g270_cross_case(tag, qt, Lx; p::Int=2)
    box = CartesianBox(Lx)
    mapg = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), Lx)
    ssb = make_semstokes(qt, p; geom=box)
    ssm = make_semstokes(qt, p; geom=mapg)
    thb = zeros(Float64, ssb.su.N); g270_fill_scalar!(thb, ssb, qt, g270_th)
    thm = zeros(Float64, ssm.su.N); g270_fill_scalar!(thm, ssm, qt, g270_th)
    uxb = zeros(Float64, ssb.su.N); uzb = zeros(Float64, ssb.su.N)
    g270_fill_affine_velocity!(uxb, uzb, ssb, qt, AffineBox(), g270_uS, g270_uT)
    uxm = zeros(Float64, ssm.su.N); uzm = zeros(Float64, ssm.su.N)
    g270_fill_affine_velocity!(uxm, uzm, ssm, qt, mapg, g270_uS, g270_uT)
    @test thb == thm && uxb == uxm && uzb == uzm
    pr = g270_pr(H=1.0, rho_heating=true, Di=1.0, heating_adiabatic=true,
                 heating_dissipation=true, latent_L=1.5, deviatoric=true)
    phases = g270_phase_list()
    tol = 1.0e-14

    nb = semquad_nu_walls(ssb.su, qt, thb, pr)
    nm = semquad_nu_walls(ssm.su, qt, thm, pr)
    g270_check_pair("$(tag) Nu_top", nb[1], nm[1], tol)
    g270_check_pair("$(tag) Nu_bot", nb[2], nm[2], tol)
    g270_check_pair("$(tag) Vrms", semquad_vrms(ssb.su, qt, uxb, uzb),
                    semquad_vrms(ssm.su, qt, uxm, uzm), tol)
    g270_check_pair("$(tag) meanT", semquad_mean_T(ssb.su, qt, thb, pr),
                    semquad_mean_T(ssm.su, qt, thm, pr), tol)
    g270_check_pair("$(tag) nu_energy",
                    semquad_nu_energy(ssb.su, qt, pr, thb, uxb, uzb),
                    semquad_nu_energy(ssm.su, qt, pr, thm, uxm, uzm), tol)
    Wb, Ub = semquad_work_dissipation(ssb.su, qt, pr, thb, uxb, uzb, g270_eta)
    Wm, Um = semquad_work_dissipation(ssm.su, qt, pr, thm, uxm, uzm, g270_eta)
    g270_check_pair("$(tag) W", Wb, Wm, tol)
    g270_check_pair("$(tag) U", Ub, Um, tol)
    g270_check_pair("$(tag) BaseAdv",
                    semquad_base_advection_integral(ssb.su, qt, pr, uxb, uzb),
                    semquad_base_advection_integral(ssm.su, qt, pr, uxm, uzm), tol)
    g270_check_pair("$(tag) theta2divu",
                    semquad_theta2_divu_integral(ssb.su, qt, thb, uxb, uzb),
                    semquad_theta2_divu_integral(ssm.su, qt, thm, uxm, uzm), tol)
    g270_check_pair("$(tag) divmax",
                    semquad_max_divergence(ssb.su, qt, uxb, uzb),
                    semquad_max_divergence(ssm.su, qt, uxm, uzm), tol)
    for wall in (:top, :bottom)
        sb = semquad_surface_velocity(ssb.su, qt, uxb, uzb; wall=wall)
        sm = semquad_surface_velocity(ssm.su, qt, uxm, uzm; wall=wall)
        g270_check_pair("$(tag) surf_rms_$(wall)", sb[1], sm[1], tol)
        g270_check_pair("$(tag) surf_max_$(wall)", sb[2], sm[2], tol)
    end
    cb = (semquad_cbf_theta_flux(ssb.su, qt, pr, thb, uxb, uzb; wall=:top,
                                 advection=:skew),
          semquad_cbf_theta_flux(ssb.su, qt, pr, thb, uxb, uzb; wall=:bottom,
                                 advection=:skew))
    cm = (semquad_cbf_theta_flux(ssm.su, qt, pr, thm, uxm, uzm; wall=:top,
                                 advection=:skew),
          semquad_cbf_theta_flux(ssm.su, qt, pr, thm, uxm, uzm; wall=:bottom,
                                 advection=:skew))
    # The CBF flux is the assembled residual summed over the wall nodes: a
    # difference of O(1) entries that nearly cancel, so on the non-dyadic tree
    # the measured 4e-16 relative operator difference (G268) is amplified to
    # <=1.5e-14 relative in the flux.  The bound is 1e-13 for these two
    # cancellation-amplified sums; every direct diagnostic keeps 1e-14.
    ctol = 1.0e-13
    g270_check_pair("$(tag) cbf_top", cb[1], cm[1], ctol)
    g270_check_pair("$(tag) cbf_bot", cb[2], cm[2], ctol)
    ncb = semquad_nu_cbf(ssb.su, qt, pr, thb, uxb, uzb; advection=:skew)
    ncm = semquad_nu_cbf(ssm.su, qt, pr, thm, uxm, uzm; advection=:skew)
    g270_check_pair("$(tag) NuCbf_top", ncb[1], ncm[1], ctol)
    g270_check_pair("$(tag) NuCbf_bot", ncb[2], ncm[2], ctol)
    lb = semquad_heat_budget(ssb.su, qt, pr, thb; phases=phases, ux=uxb, uz=uzb,
                             eta_fun=g270_eta)
    lm = semquad_heat_budget(ssm.su, qt, pr, thm; phases=phases, ux=uxm, uz=uzm,
                             eta_fun=g270_eta)
    for f in (:theta_int, :latent_int, :enthalpy, :flux_top, :flux_bot, :flux,
              :heating, :base_adv, :latent_src, :latent_adv, :latent_dgdt,
              :u_grad_theta)
        g270_check_pair("$(tag) budget_$(f)", getproperty(lb, f),
                        getproperty(lm, f), tol)
    end
    eb = semquad_exact_residual(ssb.su, qt, pr, thb, thb, thb, uxb, uzb,
                                1.0e-4, 1.0e-4, false, nothing;
                                advection=:skew, phases=phases,
                                eta_fun=g270_eta)
    em = semquad_exact_residual(ssm.su, qt, pr, thm, thm, thm, uxm, uzm,
                                1.0e-4, 1.0e-4, false, nothing;
                                advection=:skew, phases=phases,
                                eta_fun=g270_eta)
    for f in (:R, :R_over_A, :flux, :dHdt, :sumC, :Ibase, :latent_src, :rhs,
              :area)
        g270_check_pair("$(tag) exact_$(f)", getproperty(eb, f),
                        getproperty(em, f), tol)
    end
    # hmin / CFL sizes (GEOM_DESIGN E15): the box keeps leaf_hy for the history
    # column and min(hx,hy) for the CFL; the mapped sizes are the physical
    # extents, which on the box map are the same up to the last bits.
    g270_check_pair("$(tag) eta_band", QuaST._eta_interior_band_mean(
                        ssb.su, qt, g270_eta; zlo=0.25, zhi=0.75),
                    QuaST._eta_interior_band_mean(ssm.su, qt, g270_eta;
                                                  zlo=0.25, zhi=0.75), tol)
    g270_check_pair("$(tag) hmin_cfl", minimum_leaf_size(box, qt),
                    minimum_leaf_size(mapg, qt), tol)
    g270_check_pair("$(tag) hmin_hist", minimum_leaf_height(box, qt),
                    minimum_leaf_height(mapg, qt), tol)
    # level stats: same tree, same strip; a mapped geometry classifies the wall
    # strips by the physical height
    sb = semquad_level_stats(qt; strip=0.13)
    sm = semquad_level_stats(mapg, qt; strip=0.13)
    @test sb.histogram == sm.histogram
    @test sb.wall_max == sm.wall_max && sb.interior_max == sm.interior_max
    @test sb.n == sm.n
    # the history-row derived quantities: the latent-free closure and the latent
    # term of the closure column
    g270_check_pair("$(tag) closure_nolat", nb[1] - nb[2] - (pr.H - lb.base_adv),
                    nm[1] - nm[2] - (pr.H - lm.base_adv), tol)
    g270_check_pair("$(tag) closure_latent", lb.latent_dgdt / QuaST._nusselt_area(ssb.su, qt),
                    lm.latent_dgdt / QuaST._nusselt_area(ssm.su, qt), tol)
    # non-tautology: two kernel bodies; the box body is not routed through the
    # geometry (that is what keeps the nine state-dump hashes) and the mapped
    # body is new code, so a shared mistake is possible only in the data both
    # read (the coordinates and closures, which the logical nodal fill fixes).
end

@testset "G270 (b) mapped-vs-box cross-check on the box map" begin
    g270_cross_case("dyadic", g270_tree(2), 1.0)
    g270_cross_case("cells73", g270_tree_cells(1), 7.0 / 3.0)
    g270_cross_case("hanging", g270_hanging_tree(), 1.0)
end

# ---------------------------------------------------------------------------
# (c) wrong normal / wrong face Jacobian
# ---------------------------------------------------------------------------

@testset "G270 (c) wrong normal and wrong face Jacobian are O(1) failures" begin
    g = AffineBox((1.30, 0.40, -0.30, 0.90), (0.30, -0.20), 1.0)
    qt = g270_tree(2)
    p = 2
    pr = g270_pr(H=1.0, rho_heating=true, Di=1.0, heating_adiabatic=true,
                 deviatoric=true)
    ss = make_semstokes(qt, p; geom=g)
    th = zeros(Float64, ss.su.N); g270_fill_scalar!(th, ss, qt, g270_th)
    ux = zeros(Float64, ss.su.N); uz = zeros(Float64, ss.su.N)
    g270_fill_affine_velocity!(ux, uz, ss, qt, g, g270_uS, g270_uT)
    ad = g270_affine_data(g)
    an = g270_analytic(g, qt, pr)
    nt, nbot = semquad_nu_walls(ss.su, qt, th, pr)
    top_ref = -semquad_base_gradient(pr, 1.0)
    lb = semquad_heat_budget(ss.su, qt, pr, th; ux=ux, uz=uz, eta_fun=g270_eta)
    # (a2) pinned the production flux to the analytic mapped-face integral, so
    # the correct value is `an.flux_top` / `an.nu_top`.
    @test g270_rel(lb.flux_top, an.flux_top) < 1.0e-12
    @test g270_rel(nt, an.nu_top) < 1.0e-12
    # wrong outward normal: take the physical y-derivative as the normal
    # derivative (the "unmapped" implementation on a rotated + sheared frame)
    grad_y = function (S)
        (gx, gy) = g270_grad(ad, g270_th_S(S, 1.0), g270_th_T(S, 1.0))
        gy
    end
    flux_wrong_normal = g270_top_face(g, qt, grad_y)
    nu_wrong_normal = top_ref - flux_wrong_normal / ad.wall_len
    # wrong face Jacobian: integrate the (correct) normal derivative with the
    # *logical* wall measure Lx dS instead of the physical Lx|A e_1| dS
    flux_wrong_jac = an.flux_top * (g.Lx / ad.wall_len)
    nu_wrong_jac = top_ref - flux_wrong_jac / ad.wall_len
    d_n = g270_rel(lb.flux_top, flux_wrong_normal)
    d_j = g270_rel(lb.flux_top, flux_wrong_jac)
    println("G270 c flux_top=", lb.flux_top, " analytic=", an.flux_top,
            " wrong_normal=", flux_wrong_normal, " wrong_jac=", flux_wrong_jac)
    println("G270 c measured failure: wrong normal rel=", d_n,
            " wrong face Jacobian rel=", d_j,
            " (Nu_top ", nt, " vs ", nu_wrong_normal, " / ", nu_wrong_jac, ")")
    @test d_n > 0.1        # the physical y direction is not the mapped normal
    @test d_j > 0.2        # Lx != Lx|A e_1| for this map
    # non-tautology: the analytic mapped-face integrals are O(1) different from
    # the two bug-class variants, and (a2) asserts the production value equals
    # them; this measures what the mapped normal and face Jacobian are worth.
end
# ---------------------------------------------------------------------------
# (d) end-to-end mapped driver
# ---------------------------------------------------------------------------

# gate-side volume integral of div u: the gate's own chain rule at the p+2
# Gauss points (the diagnostics' rule), never the assembly's B block.
function g270_div_volume(g, sq, qt, ux, uz)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            inv = jacobian_inverse(g, sq, lf, fd.gl_x[qx], fd.gl_x[qy])
            dux_dx = 0.0
            duz_dy = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                dxi = fd.dphi_v[i, qx] * fd.phi_v[j, qy]
                deta = fd.phi_v[i, qx] * fd.dphi_v[j, qy]
                dux_dx += ux[ids[k]] * (dxi * inv[1] + deta * inv[2])
                duz_dy += uz[ids[k]] * (dxi * inv[3] + deta * inv[4])
            end
            s += QuaST.quad_measure(m, fd, qx, qy) * (dux_dx + duz_dy)
        end
    end
    return s
end

# gate-side wall integral of u.n over all four logical sides (the divergence
# theorem's right-hand side); it builds its own face points and quadrature.
function g270_face_flux(g, sq, qt, ux, uz)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    tot = 0.0
    unmax = 0.0
    sides = Dict{Symbol,Float64}(s => 0.0 for s in (:xminus, :xplus, :yminus, :yplus))
    for lf in sq.leaves
        ids = sq.elem[lf]
        for side in (:xminus, :xplus, :yminus, :yplus)
            QuaST._on_logical_wall(lf, side, sq) || continue
            fm = QuaST._mapped_face_metric(g, sq, lf, side, fd.gl_x)
            isx = side === :xminus || side === :xplus
            for q in 1:fd.nq
                uxq = 0.0
                uzq = 0.0
                if isx
                    i = side === :xplus ? n : 1
                    for a in 1:n
                        ph = fd.phi_v[a, q]
                        uxq += ux[ids[(a - 1) * n + i]] * ph
                        uzq += uz[ids[(a - 1) * n + i]] * ph
                    end
                else
                    j = side === :yplus ? n : 1
                    for a in 1:n
                        ph = fd.phi_v[a, q]
                        uxq += ux[ids[(j - 1) * n + a]] * ph
                        uzq += uz[ids[(j - 1) * n + a]] * ph
                    end
                end
                nx, ny = face_normal(fm, q)
                un = uxq * nx + uzq * ny
                unmax = max(unmax, abs(un))
                w = face_weight(fm, fd, q) * un
                tot += w
                sides[side] += w
            end
        end
    end
    return (total=tot, sides=sides, maxabs_un=unmax)
end

# gate-side density-weighted base-advection integral: the energy assembly's
# source is -rho(h) u.grad h dTbar/dh with rho = 1 unless rho_heating/rho_cp
# (then 1-T); the gate writes the physics from Tbar(h) itself.
function g270_ibase_source(g, sq, pr, theta_n, ux, uz)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            h = height(g, x, y)
            uxq = 0.0
            uzq = 0.0
            thq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += ux[ids[k]] * b
                uzq += uz[ids[k]] * b
                thq += theta_n[ids[k]] * b
            end
            rho8 = (pr.rho_heating || pr.rho_cp) ?
                   max(1.0 - (Tbar(h, pr.H) + thq), 1.0e-6) : 1.0
            ugh = upward_velocity(g, x, y, uxq, uzq)
            s += QuaST.quad_measure(m, fd, qx, qy) * rho8 * ugh * dTbar(h, pr.H)
        end
    end
    return s
end

# gate-side volume integral of lap(Tbar(h)) + H (the mapped base-state
# curvature; identically zero on the box).
function g270_baselap_int(g, sq, pr)
    p = sq.p
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            h = height(g, x, y)
            ghx, ghy = grad_height(g, x, y)
            gh2 = ghx * ghx + ghy * ghy
            lap_h = height_laplacian(g, x, y)
            val = pr.refstate === nothing ?
                  pr.H * (1.0 - gh2) + dTbar(h, pr.H) * lap_h :
                  semquad_base_laplacian(pr, h) * gh2 +
                  semquad_base_gradient(pr, h) * lap_h
            s += QuaST.quad_measure(m, fd, qx, qy) * val
        end
    end
    return s
end

# the Dirichlet wall node ids of the two Nusselt sides, deduplicated in the same
# order the CBF function uses.
function g270_wall_nodes(sq, g, wall)
    sideb, sidet = nu_wall_sides(g)
    side = wall === :top ? sidet : sideb
    n = sq.p + 1
    seen = Set{Int}()
    out = Int[]
    for lf in sq.leaves
        QuaST._on_logical_wall(lf, side, sq) || continue
        ids = sq.elem[lf]
        for q in 1:n
            gid = QuaST._logical_wall_node(ids, n, side, q)
            gid in seen && continue
            push!(seen, gid)
            push!(out, gid)
        end
    end
    return out
end

"""
    g270_id1(g, res, states, pr, dt) -> (Qc, rhs, terms...)

The time-step CBF energy-budget identity with
the mapped base-curvature term:

    Qc = dH/dt + sum(C theta) + Ibase - heating - int lap(Tbar(h)) dV

where `Qc` is the full BDF2 step residual `ca M theta - cb M theta_n + cc M
theta_nm1 + K theta + C theta - F` summed over the Dirichlet wall nodes of the
two Nusselt sides.  Every integral on the right is computed here from the
geometry and the physics definitions, not by the diagnostics under test (except
`Ibase`'s `upward_velocity`, which is the physics accessor pinned by G269).
"""
function g270_id1(g, res, states, pr, dt)
    sq = res.ss.su
    qt = res.qt
    th = states[end]
    thn = states[end - 1]
    thnm1 = states[end - 2]
    ca = 1.5 / dt
    cb = 2.0 / dt
    cc = 0.5 / dt
    M, K, C, F = assemble_semquad_energy(sq, qt, pr, res.ux, res.uz;
                                         advection=:skew, theta=thn,
                                         rho_heating=pr.rho_heating,
                                         rho_cp=pr.rho_cp,
                                         heating_adiabatic=pr.heating_adiabatic,
                                         heating_dissipation=pr.heating_dissipation,
                                         eta_fun=(x, y) -> 1.0)
    r = (ca .* (M * th) .- cb .* (M * thn) .+ cc .* (M * thnm1)) .+ K * th .+
        C * th .- F
    wnodes = vcat(g270_wall_nodes(sq, g, :top), g270_wall_nodes(sq, g, :bottom))
    Qc = sum(r[wnodes])
    dHdt = sum(M * (ca .* th .- cb .* thn .+ cc .* thnm1))
    sumC = sum(C * th)
    ibase = g270_ibase_source(g, sq, pr, thn, res.ux, res.uz)
    bud = semquad_heat_budget(sq, qt, pr, thn; ux=res.ux, uz=res.uz,
                              eta_fun=(x, y) -> 1.0)
    bl = g270_baselap_int(g, sq, pr)
    rhs = dHdt + sumC + ibase - bud.heating - bl
    return (Qc=Qc, rhs=rhs, dHdt=dHdt, sumC=sumC, ibase=ibase,
            heating=bud.heating, baselap=bl)
end

function g270_driver_run(g, pr; nsteps::Int=3, ev::Float64=0.0,
                         nu_cbf::Bool=false, extdiag::Bool=false,
                         phases::Vector{PhaseTransition}=PhaseTransition[],
                         cfl::Float64=0.0, p::Int=2)
    qt = g270_tree(2)
    ss = make_semstokes(qt, p; geom=g)
    th0 = zeros(Float64, ss.su.N)
    g270_fill_scalar!(th0, ss, qt, (S, T) -> 0.05 * sinpi(S) * sinpi(T))
    states = Vector{Vector{Float64}}()
    push!(states, copy(th0))
    dt = 1.0e-4
    res = run_semquad_simulation(qt, pr, th0; geom=g, p=p,
                                 tstop=nsteps * dt, dt0=dt, dtmax=dt,
                                 dtmin=1.0e-16, linear=:direct,
                                 maxsteps=nsteps, amr_every=0,
                                 initial_refine=0, symmetry=:none,
                                 advection=:skew, ev=ev, cfl=cfl,
                                 maxpic=1, phases=phases, nu_cbf=nu_cbf,
                                 extdiag=extdiag, verbose=false,
                                 callback=(s -> push!(states, copy(s.theta))))
    return res, states
end

@testset "G270 (d) end-to-end mapped driver" begin
    gaff = AffineBox((1.30, 0.40, -0.30, 0.90), (0.30, -0.20), 1.0)
    gcur = CurvedBox(0.20, 1.0)
    dt = 1.0e-4
    pr_off = g270_pr(H=0.0)
    pr_on = g270_pr(H=2.0, rho_heating=true)
    pr_adia = g270_pr(H=0.0, Di=1.0, heating_adiabatic=true)

    for (tag, g, pr) in (("affine/off", gaff, pr_off),
                         ("curved/off", gcur, pr_off),
                         ("affine/on", gaff, pr_on),
                         ("curved/on", gcur, pr_on),
                         ("affine/adiab", gaff, pr_adia))
        res, states = g270_driver_run(g, pr)
        println("G270 d ", rpad(tag, 14), " steps=", res.steps,
                " rows=", length(res.history), " width=",
                isempty(res.history) ? 0 : length(res.history[end]),
                " theta_max=", maximum(abs, res.theta))
        @test res.steps == 3
        @test geometry_identity(res.ss.su.geom) == geometry_identity(g)
        @test length(res.history) == 3
        @test length(res.history[end]) == 18
        @test all(isfinite, vcat(res.history...))
        @test all(isfinite, res.theta) && all(isfinite, res.ux) &&
              all(isfinite, res.uz)
        # the driver advanced the state (not a no-op step)
        @test maximum(abs, res.theta - states[1]) > 1.0e-6
        sq = res.ss.su
        qt = res.qt
        row = res.history[end]
        # the history row is the mapped diagnostics of the returned state: the
        # row columns 3,4,5,6,9,10,11,12,14,15 are pure functions of it.
        nt, nb = semquad_nu_walls(sq, qt, res.theta, pr)
        @test row[3] == nt && row[4] == nb
        @test row[5] == semquad_vrms(sq, qt, res.ux, res.uz)
        @test row[6] == semquad_mean_T(sq, qt, res.theta, pr)
        @test row[9] == semquad_max_divergence(sq, qt, res.ux, res.uz)
        @test row[10] == minimum_leaf_height(res.ss.su.geom, qt)
        @test row[14] == semquad_surface_velocity(sq, qt, res.ux, res.uz)[1]
        @test row[15] == semquad_surface_velocity(sq, qt, res.ux, res.uz)[2]
        # mass flux: the discrete continuity row gives int div u = 0 (the
        # constant pressure test function), and the discrete divergence theorem
        # holds between the gate's volume and face quadratures.
        umax = maximum(hypot.(res.ux, res.uz))
        area = total_measure(g, sq)
        dv = g270_div_volume(g, sq, qt, res.ux, res.uz)
        ff = g270_face_flux(g, sq, qt, res.ux, res.uz)
        println("G270 d ", rpad(tag, 14), " int(div u)=", dv,
                " face_flux=", ff.total, " umax=", umax, " area=", area)
        @test abs(dv) < 1.0e-8 * max(umax * area, 1.0)
        @test abs(ff.total - dv) < 1.0e-10 * max(umax * area, 1.0)
        # the time-step CBF budget identity (ID1) with the mapped base curvature
        id = g270_id1(g, res, states, pr, dt)
        scale = max(abs(id.Qc), abs(id.dHdt), abs(id.heating), abs(id.baselap),
                    1.0e-300)
        println("G270 d ", rpad(tag, 14), " ID1: Qc=", id.Qc, " rhs=", id.rhs,
                " R=", id.Qc - id.rhs, " rel=", abs(id.Qc - id.rhs) / scale)
        @test abs(id.Qc - id.rhs) / scale < 1.0e-10
    end

    # heating changes the solution: the internal-heating run must differ from the
    # heating-off run on the same geometry and initial condition.
    res_off, _ = g270_driver_run(gaff, pr_off)
    res_on, _ = g270_driver_run(gaff, pr_on)
    dth = maximum(abs, res_on.theta - res_off.theta)
    println("G270 d heating changes theta by ", dth)
    @test dth > 1.0e-5
    # internal heating raises the volume-mean temperature (history column 6)
    @test res_on.history[end][6] > res_off.history[end][6]

    # a mapped run with cfl > 0 exercises the mapped CFL branch of the time loop
    # (`minimum_leaf_size`): dt is updated from the mapped physical sizes.
    res_cfl, _ = g270_driver_run(gaff, pr_off; cfl=0.5)
    @test res_cfl.steps == 3
    @test res_cfl.history[1][2] > 0.0
    @test res_cfl.history[end][10] > 0.0

    # the mapped CBF and exact-residual diagnostic columns run end to end: the
    # extra history columns are present and finite (20 = 18 + Nu_cbf pair,
    # 21 = + resid_A).
    prc = g270_pr(H=1.0, rho_heating=true)
    res_cbf, _ = g270_driver_run(gaff, prc; nu_cbf=true)
    @test length(res_cbf.history[end]) == 20
    @test all(isfinite, res_cbf.history[end][19:20])
    res_ext, _ = g270_driver_run(gcur, prc; nu_cbf=true, extdiag=true)
    @test length(res_ext.history[end]) == 21
    @test all(isfinite, res_ext.history[end][19:21])

    # the mapped latent-heat diagnostics run end to end (the closure column's
    # latent term calls the mapped heat budget): the latent run appends the
    # closure_nolatent column and stays finite.
    phases = g270_phase_list()
    prl = g270_pr(H=0.0, latent_L=1.5)
    res_lat, _ = g270_driver_run(gcur, prl; phases=phases, nsteps=2)
    @test length(res_lat.history[end]) == 19
    @test all(isfinite, res_lat.history[end])
    @test abs(res_lat.history[end][19] - res_lat.history[end][13]) > 0.0
    # A genuinely nonzero wall flux: one open (traction-free) top wall.  The
    # net normal flux over all four walls must still vanish (incompressibility)
    # and the open wall must carry a nonzero flux; the gate's per-wall mapped
    # face integrals and its volume divergence then satisfy the discrete
    # divergence theorem.  With all walls rigid the normal velocity is zero by
    # construction, so that case cannot exercise the face metric.
    pr_open = g270_pr(H=0.0, wall_bc=(:same, :same, :same, :open))
    res_open, _ = g270_driver_run(gaff, pr_open)
    @test res_open.steps == 3
    @test all(isfinite, res_open.theta)
    sqo = res_open.ss.su
    qto = res_open.qt
    umaxo = maximum(hypot.(res_open.ux, res_open.uz))
    areao = total_measure(gaff, sqo)
    dvo = g270_div_volume(gaff, sqo, qto, res_open.ux, res_open.uz)
    ffo = g270_face_flux(gaff, sqo, qto, res_open.ux, res_open.uz)
    println("G270 d open/affine  int(div u)=", dvo, " net face flux=", ffo.total,
            " top flux=", ffo.sides[:yplus], " umax=", umaxo)
    # the top wall is open and the flow there is a recirculation: the local
    # normal velocity is O(umax) and the *net* flux through it must vanish
    # because every other wall is closed (incompressibility).
    @test ffo.maxabs_un > 1.0e-3 * umaxo
    @test abs(ffo.sides[:yplus]) < 1.0e-8 * umaxo * areao
    @test abs(ffo.total) < 1.0e-8 * umaxo * areao
    @test abs(dvo) < 1.0e-8 * umaxo * areao
    @test abs(ffo.total - dvo) < 1.0e-10 * umaxo * areao
end