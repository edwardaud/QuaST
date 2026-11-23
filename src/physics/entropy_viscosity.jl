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



