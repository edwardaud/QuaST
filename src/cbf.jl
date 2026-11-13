# Gresho consistent boundary flux (CBF) for the temperature equation.
# Returns the integrated outward heat flux through the bottom (z=0) and top
# (z=1) boundaries.  With GLL face quadrature the face mass matrix is diagonal,
# so the integrated flux is simply the sum of the local volume residuals over
# the boundary face nodes.

function cbf_wall_flux(m::Mesh, fd::FEData, pr::Params,
                       theta::Vector{Float64}, theta_prev::Vector{Float64},
                       ux::Vector{Float64}, uz::Vector{Float64},
                       dt::Float64;
                       theta_prev2::Union{Nothing,Vector{Float64}}=nothing,
                       dt_prev::Float64=0.0,
                       theta_ev::Union{Nothing,Vector{Float64}}=nothing,
                       theta_ev_prev::Union{Nothing,Vector{Float64}}=nothing,
                       ev::Float64=0.0, ev_cmax::Float64=0.5,
                       debug::Bool=false, debug_top::Bool=true,
                       debug_cell::Int=0)
    p = m.p
    hx = m.hx
    pv = fd.phi_v
    dpv = fd.dphi_v
    nv = (p + 1)^2
    phik = zeros(Float64, nv)
    gxk = zeros(Float64, nv)
    gzk = zeros(Float64, nv)
    top = 0.0
    bot = 0.0
    for (ez, is_top) in ((0, false), (m.nze - 1, true))
        hze = z_size(m, ez)
        z0 = z_left(m, ez)
        he = sqrt(hx * hx + hze * hze)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            # CBF only reconstructs the boundary flux in the top/bottom cell
            # row, where ASPECT sets the entropy viscosity to zero anyway
            # (Dirichlet temperature + tangential velocity, see
            # entropy_viscosity.cc:371-447), so no EV is needed here.
            nu_h = 0.0
            kap = 1.0
            rhssum = 0.0
            rhs_all = 0.0
            dbg_rhs = zeros(Float64, p + 1)
            dbg_diff = zeros(Float64, p + 1)
            dbg_adv = zeros(Float64, p + 1)
            dbg_dt = zeros(Float64, p + 1)
            dbg_src = zeros(Float64, p + 1)
            for qy in 1:fd.nq
                z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                tbz = dTbar(z, pr.H)
                for qx in 1:fd.nq
                    uxq = 0.0; uzq = 0.0
                    thq = 0.0; thp = 0.0; thp2 = 0.0; gthx = 0.0; gthz = 0.0
                    k = 0
                    for j in 1:p+1
                        py = pv[j, qy]
                        for i in 1:p+1
                            k += 1
                            b = pv[i, qx] * py
                            phik[k] = b
                            gxk[k] = dpv[i, qx] * py * (2.0 / hx)
                            gzk[k] = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                            uxq += ux[g[k]] * b
                            uzq += uz[g[k]] * b
                            thq += theta[g[k]] * b
                            thp += theta_prev[g[k]] * b
                            if theta_prev2 !== nothing
                                thp2 += theta_prev2[g[k]] * b
                            end
                            gthx += theta[g[k]] * gxk[k]
                            gthz += theta[g[k]] * gzk[k]
                        end
                    end
                    if theta_prev2 !== nothing && dt_prev > 0.0
                        r = dt / dt_prev
                        a0 = (2.0 + r) / (1.0 + r)
                        b0 = 1.0 + r
                        c0 = r * r / (1.0 + r)
                        dTdt = (a0 * thq - b0 * thp + c0 * thp2) / dt
                    else
                        dTdt = (thq - thp) / dt
                    end
                    adv = uxq * gthx + uzq * (gthz + tbz)
                    w = fd.gl_w[qx] * fd.gl_w[qy] * hx * hze / 4.0
                    for kk in 1:nv
                        # -kappa grad(T).grad(phi_i) + (-(dTdt+u.gradT)+H) phi_i
                        rdiff = -kap * (gxk[kk] * gthx + gzk[kk] * (gthz + tbz))
                        radv = -adv * phik[kk]
                        rdt = -dTdt * phik[kk]
                        rsrc = pr.H * phik[kk]
                        rhs = (rdiff + radv + rdt + rsrc) * w
                        if debug && is_top == debug_top && ex == debug_cell
                            rhs_all += rhs
                        end
                        jj = div(kk - 1, p + 1) + 1
                        if (!is_top && jj == 1) || (is_top && jj == p + 1)
                            rhssum += rhs
                            if debug && is_top == debug_top && ex == debug_cell
                                ii = is_top ? (kk - p * (p + 1)) : kk
                                dbg_rhs[ii] += rhs
                                dbg_diff[ii] += rdiff * w
                                dbg_adv[ii] += radv * w
                                dbg_dt[ii] += rdt * w
                                dbg_src[ii] += rsrc * w
                            end
                        end
                    end
                end
            end
            if is_top
                top += rhssum
            else
                bot += rhssum
            end
            if debug && is_top == debug_top && ex == debug_cell
                _, dL = lagrange_eval(fd.gll_v, is_top ? 1.0 : -1.0)
                zf = is_top ? z0 + hze : z0
                tbf = dTbar(zf, pr.H)
                println("CBF debug is_top=", is_top, " ex=", ex, " zface=", zf,
                        " sum_all_rhs=", rhs_all)
                for ii in 1:p+1
                    jf = is_top ? p + 1 : 1
                    kkf = (jf - 1) * (p + 1) + ii
                    dthdz = 0.0
                    for j in 1:p+1
                        kkj = (j - 1) * (p + 1) + ii
                        dthdz += theta[g[kkj]] * dL[j] * (2.0 / hze)
                    end
                    mi = fd.gll_v.weights[ii] * hx / 2.0
                    gi = dbg_rhs[ii] / mi
                    direct = is_top ? -(dthdz + tbf) : (dthdz + tbf)
                    println("  i=", ii, " T=", Tbar(zf, pr.H) + theta[g[kkf]],
                            " rhs=", dbg_rhs[ii], " diff=", dbg_diff[ii],
                            " adv=", dbg_adv[ii], " dt=", dbg_dt[ii],
                            " src=", dbg_src[ii], " mass=", mi,
                            " g=", gi, " direct=", direct)
                end
            end
        end
    end
    return bot, top
end
