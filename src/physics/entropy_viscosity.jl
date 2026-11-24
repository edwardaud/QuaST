# ---------------------------------------------------------------------------
# Entropy viscosity on the quadtree (port of assembly.jl / ASPECT)
# ---------------------------------------------------------------------------

struct SEMEVGlobals
    minT::Float64
    maxT::Float64
    avgT::Float64
    entropy_variation::Float64
    field_variation::Float64
    u_infty::Float64
end

function semquad_ev_globals(sq::SemQuad, qt::Quadtree, pr::Params,
                            theta::Vector{Float64},
                            theta_prev::Union{Nothing,Vector{Float64}},
                            ux::Vector{Float64}, uz::Vector{Float64},
                            dt_ratio::Float64)
    minT = Inf; maxT = -Inf; uinf = 0.0
    for g in 1:sq.N
        y = sq.coords[g][2]
        tb = Tbar(y, pr.H)
        tn = tb + theta[g]
        t = if dt_ratio > 0.0 && theta_prev !== nothing
            (1.0 + dt_ratio) * tn - dt_ratio * (tb + theta_prev[g])
        else
            tn
        end
        minT = min(minT, t); maxT = max(maxT, t)
        uinf = max(uinf, hypot(ux[g], uz[g]))
    end
    avgT = 0.5 * (minT + maxT)
    fd = make_fedata(sq.p, sq.p)
    eint = 0.0; area = 0.0; emin = Inf; emax = -Inf
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        x0 = lf[2] * h; y0 = lf[3] * h
        ids = sq.elem[lf]
        n = sq.p + 1
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * h
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * h
            thn = 0.0; thp = 0.0
            for j in 1:n, i in 1:n
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                k = (j - 1) * n + i
                thn += theta[ids[k]] * b
                if theta_prev !== nothing
                    thp += theta_prev[ids[k]] * b
                end
            end
            Tmid = Tbar(y, pr.H) + 0.5 * (thn + thp)
            e = (Tmid - avgT)^2
            w = 0.25 * h * h * fd.gl_w[qx] * fd.gl_w[qy]
            eint += w * e; area += w
            emin = min(emin, e); emax = max(emax, e)
        end
    end
    avge = area > 0.0 ? eint / area : 0.0
    return SEMEVGlobals(minT, maxT, avgT,
                        max(emax - avge, avge - emin),
                        maxT - minT, uinf)
end

function semquad_ev_cell_stats(sq::SemQuad, pr::Params,
                               theta::Vector{Float64},
                               theta_prev::Union{Nothing,Vector{Float64}},
                               ux::Vector{Float64}, uz::Vector{Float64},
                               dt::Float64, lf::NTuple{3,Int},
                               evg::SEMEVGlobals; alpha::Int=2,
                               fd::Union{Nothing,FEData}=nothing,
                               d2tab::Union{Nothing,Vector{Vector{Float64}}}=nothing)
    h = leaf_hy(sq, lf[1])
    x0 = lf[2] * h; y0 = lf[3] * h
    ids = sq.elem[lf]
    n = sq.p + 1
    # `fd` and `d2tab` depend only on `(p, nq)`: the energy assembler builds
    # them once per assembly and hands them in (one `eigen` + 6 matrices per
    # leaf otherwise -- 57.8 MB and 44 ms per assembly on the 10 000-leaf
    # state).  The cold callers (`semquad_ev_field`, the tests) still get the
    # historical allocation: identical values, built the same way.
    fd_used = fd === nothing ? make_fedata(sq.p, sq.p) : fd
    Rmax = 0.0; umax = 0.0
    # Loop-invariant hoist (S3 stage 4): `gll_rule` and `lagrange_d2` were
    # called *inside* the qx/qy quadrature loop (four times per point: 144 per
    # leaf at p=4), and `gll_rule` recomputes the GLL Newton iteration and
    # allocates three vectors every time.  The rule and the 1-D second
    # derivative table depend only on `p` and the fixed quadrature abscissae, so
    # one rule and `nq` table entries are enough.  Values are identical, hence
    # `Rmax`/`umax` are bitwise unchanged (G188).
    d2tab_used = if d2tab === nothing
        ev_rule = gll_rule_cached(sq.p)
        t = Vector{Vector{Float64}}(undef, fd_used.nq)
        for k in 1:fd_used.nq
            t[k] = lagrange_d2(ev_rule, fd_used.gl_x[k])[3]
        end
        t
    else
        d2tab
    end
    for qy in 1:fd_used.nq, qx in 1:fd_used.nq
        y = y0 + 0.5 * (fd_used.gl_x[qy] + 1.0) * h
        d2x = d2tab_used[qx]
        d2y = d2tab_used[qy]
        uxq = 0.0; uzq = 0.0
        thn = 0.0; thp = 0.0
        gthx = 0.0; gthz = 0.0; lapth = 0.0
        gthxp = 0.0; gthzp = 0.0; lapp = 0.0
        for j in 1:n, i in 1:n
            k = (j - 1) * n + i
            b = fd_used.phi_v[i, qx] * fd_used.phi_v[j, qy]
            gx = fd_used.dphi_v[i, qx] * fd_used.phi_v[j, qy] * (2.0 / h)
            gz = fd_used.phi_v[i, qx] * fd_used.dphi_v[j, qy] * (2.0 / h)
            uxq += ux[ids[k]] * b
            uzq += uz[ids[k]] * b
            thn += theta[ids[k]] * b
            gthx += theta[ids[k]] * gx
            gthz += theta[ids[k]] * gz
            lapth += theta[ids[k]] *
                     (d2x[i] * fd_used.phi_v[j, qy] * (2.0 / h)^2 +
                      fd_used.phi_v[i, qx] * d2y[j] * (2.0 / h)^2)
            if theta_prev !== nothing
                thp += theta_prev[ids[k]] * b
                gthxp += theta_prev[ids[k]] * gx
                gthzp += theta_prev[ids[k]] * gz
                lapp += theta_prev[ids[k]] *
                        (d2x[i] * fd_used.phi_v[j, qy] * (2.0 / h)^2 +
                         fd_used.phi_v[i, qx] * d2y[j] * (2.0 / h)^2)
            end
        end
        Tmid = Tbar(y, pr.H) + 0.5 * (thn + thp)
        dTdt = dt > 0.0 ? (thn - thp) / dt : 0.0
        adv = 0.5 * uxq * (gthx + gthxp) +
              0.5 * uzq * (gthz + gthzp) + uzq * dTbar(y, pr.H)
        lap = 0.5 * (lapth + lapp)
        R = abs(dTdt + adv - lap)
        alpha == 2 && (R *= abs(Tmid - evg.avgT))
        Rmax = max(Rmax, R)
        umax = max(umax, hypot(uxq, uzq))
    end
    return Rmax, umax
end

function semquad_ev_nu(sq::SemQuad, pr::Params,
                       theta::Vector{Float64},
                       theta_prev::Union{Nothing,Vector{Float64}},
                       ux::Vector{Float64}, uz::Vector{Float64},
                       dt::Float64, lf::NTuple{3,Int}, evg::SEMEVGlobals;
                       ev::Float64, ev_cmax::Float64, ev_alpha::Int=2,
                       ev_timestep::Int=0,
                       fd::Union{Nothing,FEData}=nothing,
                       d2tab::Union{Nothing,Vector{Vector{Float64}}}=nothing)
    # element length for the entropy-viscosity scale: the geometric mean of the
    # two extents (exactly h on square leaves, since h*h is a power of two)
    h = sqrt(leaf_hx(sq, lf[1]) * leaf_hy(sq, lf[1]))
    he = sqrt(2.0) * h
    Rmax, umax = semquad_ev_cell_stats(sq, pr, theta, theta_prev, ux, uz,
                                        dt, lf, evg; alpha=ev_alpha,
                                        fd=fd, d2tab=d2tab)
    nu_max = ev_cmax * umax * he
    if abs(evg.u_infty) < 1.0e-50 || ev_timestep <= 1 ||
       abs(evg.entropy_variation) < 1.0e-50 ||
       abs(evg.field_variation) < 1.0e-50
        nu = nu_max
    else
        nu_e = ev_alpha == 2 ?
               ev * he * he * Rmax / evg.entropy_variation :
               ev * he * Rmax / evg.field_variation
        nu = min(nu_max, nu_e)
    end
    if lf[3] == 0 || lf[3] == sq.ny0 * (1 << lf[1]) - 1
        nu = 0.0
    end
    return nu
end

"""
    semquad_ev_field(sq, qt, pr, theta, ux, uz; dt, ev, ev_cmax, ev_alpha, ev_timestep)

Elementwise entropy-viscosity field (one value per leaf, in `sq.leaves` order)
exactly as the steady operator of the energy step computes it: `theta_prev = theta` and
`dt_ratio = 0`, i.e. the steady form of the residual.  Exposed so the Newton
JVP can differentiate the field (the EV tangent, `ev_tangent=true`).

the energy step fills the same values through
`assemble_semquad_energy(...; ev_out=...)`; this routine is the standalone
version used for finite differences.
"""
function semquad_ev_field(sq::SemQuad, qt::Quadtree, pr::Params,
                          theta::Vector{Float64},
                          ux::Vector{Float64}, uz::Vector{Float64};
                          dt::Float64=1.0e-2, ev::Float64=0.0,
                          ev_cmax::Float64=0.052, ev_alpha::Int=2,
                          ev_timestep::Int=10)
    # Cut 3: the box body below is the historical arithmetic; a mapped geometry
    # has its own method (the residual, the base state and the element size are
    # all evaluated through the geometry).  The `isa` branch keeps the box call
    # a static, non-allocating call.
    geom = sq.geom
    geom isa CartesianBox ||
        return semquad_ev_field(geom, sq, qt, pr, theta, ux, uz;
                                dt=dt, ev=ev, ev_cmax=ev_cmax,
                                ev_alpha=ev_alpha, ev_timestep=ev_timestep)
    evg = semquad_ev_globals(sq, qt, pr, theta, theta, ux, uz, 0.0)
    out = zeros(Float64, length(sq.leaves))
    fd_ev = make_fedata(sq.p, sq.p)
    d2tab = [lagrange_d2(gll_rule_cached(sq.p), fd_ev.gl_x[k])[3]
             for k in 1:fd_ev.nq]
    for (i, lf) in enumerate(sq.leaves)
        out[i] = semquad_ev_nu(sq, pr, theta, theta, ux, uz, dt, lf, evg;
                               ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                               ev_timestep=ev_timestep, fd=fd_ev, d2tab=d2tab)
    end
    return out
end

# ---------------------------------------------------------------------------
# Cut 3: the mapped entropy-viscosity path.
#
# Every *physical* quantity the box body evaluates through `y` and `hx/hy` is
# evaluated here through the geometry:
#
#   * the temperature/velocity sample points are `quad_point` / `sq.coords`
#     (mapped by cut 1) and the base state is a function of the physical
#     **height** `h = height(g,x,y)` (`Tbar(h)`, `dTbar(h)`) — the annulus
#     generalisation of the Cartesian `y` (`GEOM_DESIGN` §2.7/§6.4);
#   * the gradients are the chain-rule `basis_grad`, and the **Laplacian** is
#     `lap_phi` of the map-curvature surface (the map Hessian term `C` is the
#     part the box cannot see, F4/§8 risk 10).  The residual also subtracts the
#     mapped base-state Laplacian
#     `Tbar''(h)|grad h|^2 + Tbar'(h) lap(h)`, which the energy assembly adds as
#     a source, so the residual is the residual of the equation the energy step
#     actually solves (it is not zero for a static base state on a curved map,
#     because the base state's Laplacian is part of that equation);
#   * the element length is the mapped `he = sqrt(2*|element|)` (the design's
#     cut-3 statement), where `|element|` is the physical leaf area integrated
#     with the metric.  The **box keeps `he = sqrt(2)*sqrt(hx*hy)`** (E14, a
#     different float from `sqrt(2*hx*hy)` in general) and the box path keeps
#     the physical `(2/h)^2` Laplacian; both are documented differences of the
#     historical box code, not of the mapped one;
#   * the wall-row switch `lf[3] == 0 || lf[3] == ny0*(1<<l)-1` is already a
#     **logical** side test (it reads the leaf index, not a physical row), so it
#     is the same statement on the mapped path and is kept verbatim;
#   * the velocity scale `umax` is `hypot(uxq, uzq)` on both paths (the physical
#     velocity magnitude; it does not change under the map).
# ---------------------------------------------------------------------------

function semquad_ev_globals(g::AbstractGeometry, sq::SemQuad, qt::Quadtree,
                            pr::Params, theta::Vector{Float64},
                            theta_prev::Union{Nothing,Vector{Float64}},
                            ux::Vector{Float64}, uz::Vector{Float64},
                            dt_ratio::Float64)
    minT = Inf; maxT = -Inf; uinf = 0.0
    for gi in 1:sq.N
        x, y = sq.coords[gi]
        tb = semquad_base_T(pr, g, x, y)
        tn = tb + theta[gi]
        t = if dt_ratio > 0.0 && theta_prev !== nothing
            (1.0 + dt_ratio) * tn - dt_ratio * (tb + theta_prev[gi])
        else
            tn
        end
        minT = min(minT, t); maxT = max(maxT, t)
        uinf = max(uinf, hypot(ux[gi], uz[gi]))
    end
    avgT = 0.5 * (minT + maxT)
    fd = make_fedata(sq.p, sq.p)
    eint = 0.0; area = 0.0; emin = Inf; emax = -Inf
    n = sq.p + 1
    for lf in sq.leaves
        lm = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(lm, sq, lf, fd, qx, qy)
            thn = 0.0; thp = 0.0
            for j in 1:n, i in 1:n
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                k = (j - 1) * n + i
                thn += theta[ids[k]] * b
                if theta_prev !== nothing
                    thp += theta_prev[ids[k]] * b
                end
            end
            Tmid = semquad_base_T(pr, g, x, y) + 0.5 * (thn + thp)
            e = (Tmid - avgT)^2
            w = quad_measure(lm, fd, qx, qy)
            eint += w * e; area += w
            emin = min(emin, e); emax = max(emax, e)
        end
    end
    avge = area > 0.0 ? eint / area : 0.0
    return SEMEVGlobals(minT, maxT, avgT,
                        max(emax - avge, avge - emin),
                        maxT - minT, uinf)
end

function semquad_ev_cell_stats(g::AbstractGeometry, sq::SemQuad, pr::Params,
                               theta::Vector{Float64},
                               theta_prev::Union{Nothing,Vector{Float64}},
                               ux::Vector{Float64}, uz::Vector{Float64},
                               dt::Float64, lf::NTuple{3,Int},
                               evg::SEMEVGlobals; alpha::Int=2,
                               fd::Union{Nothing,FEData}=nothing,
                               d2tab::Union{Nothing,Vector{Vector{Float64}}}=nothing)
    ids = sq.elem[lf]
    n = sq.p + 1
    fd_used = fd === nothing ? make_fedata(sq.p, sq.p) : fd
    d2tab_used = if d2tab === nothing
        ev_rule = gll_rule_cached(sq.p)
        t = Vector{Vector{Float64}}(undef, fd_used.nq)
        for k in 1:fd_used.nq
            t[k] = lagrange_d2(ev_rule, fd_used.gl_x[k])[3]
        end
        t
    else
        d2tab
    end
    lm = leaf_metric(g, sq, lf, fd_used)
    mc = map_curvature(g, sq, lf, fd_used)
    Rmax = 0.0; umax = 0.0
    for qy in 1:fd_used.nq, qx in 1:fd_used.nq
        x, y = quad_point(lm, sq, lf, fd_used, qx, qy)
        # The mapped base-state Laplacian (`lap(Tbar(h)) + H`) the residual must
        # subtract: the same helper the mapped energy assembly adds as a source
        # (cut-4 defect B1: cut 3 used `semquad_base_laplacian` alone, which is 0
        # for the Boussinesq base, so `H*(1-|grad h|^2)` was dropped on a
        # geometry with `|grad h| != 1`).
        lapT = _mapped_base_curvature(pr, g, x, y)
        uxq = 0.0; uzq = 0.0
        thn = 0.0; thp = 0.0
        gthx = 0.0; gthz = 0.0; lapth = 0.0
        gthxp = 0.0; gthzp = 0.0; lapp = 0.0
        for j in 1:n, i in 1:n
            k = (j - 1) * n + i
            b = fd_used.phi_v[i, qx] * fd_used.phi_v[j, qy]
            gx, gz = basis_grad(lm, fd_used, qx, qy, i, j)
            lphi = lap_phi(mc, fd_used, d2tab_used, qx, qy, i, j)
            uxq += ux[ids[k]] * b
            uzq += uz[ids[k]] * b
            thn += theta[ids[k]] * b
            gthx += theta[ids[k]] * gx
            gthz += theta[ids[k]] * gz
            lapth += theta[ids[k]] * lphi
            if theta_prev !== nothing
                thp += theta_prev[ids[k]] * b
                gthxp += theta_prev[ids[k]] * gx
                gthzp += theta_prev[ids[k]] * gz
                lapp += theta_prev[ids[k]] * lphi
            end
        end
        Tmid = semquad_base_T(pr, g, x, y) + 0.5 * (thn + thp)
        dTdt = dt > 0.0 ? (thn - thp) / dt : 0.0
        ugh = upward_velocity(g, x, y, uxq, uzq)
        adv = 0.5 * uxq * (gthx + gthxp) +
              0.5 * uzq * (gthz + gthzp) +
              ugh * semquad_base_gradient(pr, g, x, y)
        lap = 0.5 * (lapth + lapp) + lapT
        R = abs(dTdt + adv - lap)
        alpha == 2 && (R *= abs(Tmid - evg.avgT))
        Rmax = max(Rmax, R)
        umax = max(umax, hypot(uxq, uzq))
    end
    return Rmax, umax
end

function semquad_ev_nu(g::AbstractGeometry, sq::SemQuad, pr::Params,
                       theta::Vector{Float64},
                       theta_prev::Union{Nothing,Vector{Float64}},
                       ux::Vector{Float64}, uz::Vector{Float64},
                       dt::Float64, lf::NTuple{3,Int}, evg::SEMEVGlobals;
                       ev::Float64, ev_cmax::Float64, ev_alpha::Int=2,
                       ev_timestep::Int=0,
                       fd::Union{Nothing,FEData}=nothing,
                       d2tab::Union{Nothing,Vector{Vector{Float64}}}=nothing)
    fd_used = fd === nothing ? make_fedata(sq.p, sq.p) : fd
    lm = leaf_metric(g, sq, lf, fd_used)
    # physical element area |element| = int detJ; the mapped element length is
    # `he = sqrt(2*|element|)` (GEOM_DESIGN §3 cut 3), the mapped generalisation
    # of the box E14 value.  The integration reuses the EV rule; on a curved
    # element detJ is smooth, so the value is smooth in h and the convergence
    # order of the operator cannot be degraded by this scalar.
    area = 0.0
    for qy in 1:fd_used.nq, qx in 1:fd_used.nq
        area += quad_measure(lm, fd_used, qx, qy)
    end
    he = sqrt(2.0 * area)
    Rmax, umax = semquad_ev_cell_stats(g, sq, pr, theta, theta_prev, ux, uz,
                                       dt, lf, evg; alpha=ev_alpha,
                                       fd=fd_used, d2tab=d2tab)
    nu_max = ev_cmax * umax * he
    if abs(evg.u_infty) < 1.0e-50 || ev_timestep <= 1 ||
       abs(evg.entropy_variation) < 1.0e-50 ||
       abs(evg.field_variation) < 1.0e-50
        nu = nu_max
    else
        nu_e = ev_alpha == 2 ?
               ev * he * he * Rmax / evg.entropy_variation :
               ev * he * Rmax / evg.field_variation
        nu = min(nu_max, nu_e)
    end
    # the wall-row switch is the logical side test (leaf index, not a physical
    # coordinate), so it is the same statement as on the box path
    if lf[3] == 0 || lf[3] == sq.ny0 * (1 << lf[1]) - 1
        nu = 0.0
    end
    return nu
end

"""
    semquad_ev_field(g, sq, qt, pr, theta, ux, uz; ...)

Mapped method of [`semquad_ev_field`](@ref): the same elementwise EV field with
the mapped residual of `semquad_ev_cell_stats(g, ...)` and the mapped element
length `he = sqrt(2*|element|)`.
"""
function semquad_ev_field(g::AbstractGeometry, sq::SemQuad, qt::Quadtree,
                          pr::Params, theta::Vector{Float64},
                          ux::Vector{Float64}, uz::Vector{Float64};
                          dt::Float64=1.0e-2, ev::Float64=0.0,
                          ev_cmax::Float64=0.052, ev_alpha::Int=2,
                          ev_timestep::Int=10)
    evg = semquad_ev_globals(g, sq, qt, pr, theta, theta, ux, uz, 0.0)
    out = zeros(Float64, length(sq.leaves))
    fd_ev = make_fedata(sq.p, sq.p)
    d2tab = [lagrange_d2(gll_rule_cached(sq.p), fd_ev.gl_x[k])[3]
             for k in 1:fd_ev.nq]
    for (i, lf) in enumerate(sq.leaves)
        out[i] = semquad_ev_nu(g, sq, pr, theta, theta, ux, uz, dt, lf, evg;
                               ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                               ev_timestep=ev_timestep, fd=fd_ev, d2tab=d2tab)
    end
    return out
end




