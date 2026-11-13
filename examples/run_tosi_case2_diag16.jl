using QuaST
using Printf

function diagnostics(m, fd, res, pr)
    p = m.p
    w = fd.gll_v.weights
    avgT = 0.0
    for ez in 0:m.nze-1
        hze = z_size(m, ez); ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for j in 1:p+1, i in 1:p+1
                k = (j-1)*(p+1) + i
                z = ze0 + 0.5 * (fd.gll_v.nodes[j] + 1) * hze
                T = Tbar(z, pr.H) + res.theta[g[k]]
                avgT += T * w[i] * w[j] * (m.hx/2) * (hze/2)
            end
        end
    end
    ez = m.nze - 1; hze = z_size(m, ez)
    s2 = 0.0; smax = 0.0
    for ex in 0:m.nxe-1
        g = element_vel_gids(m, ex, ez)
        j = p + 1
        for i in 1:p+1
            u = res.ux[g[(j-1)*(p+1)+i]]
            s2 += u*u * w[i] * (m.hx/2)
            smax = max(smax, u)
        end
    end
    return avgT, sqrt(s2), smax
end

n = 16
m = make_mesh(1.0, n, n, 4; zgrading=:uniform)
fd = make_fedata(4, m.q)
xs, zs = node_coords(m, fd.gll_v)
nv = m.nvx * m.nvz
theta0 = [0.01 * cospi(xs[mod(k-1,m.nvx)+1]) * sinpi(zs[div(k-1,m.nvx)+1]) for k in 1:nv]
pr = Params(Ra=100.0, H=0.0)
tp = TosiParams(1e5, 1.0, 1e-3, 1.0, 1e-6, 10.0, 0.115, 1.0, false)
res = run_steady(m, fd, pr; theta0=theta0, maxiter=120, tol=1e-9, relax=0.5,
                 verbose=false, linear=:direct, velocity_space=:taylor_hood,
                 res_bar=1e-9, tosi=tp)
avgT, urms, umax = diagnostics(m, fd, res, pr)
@printf("TOSI Case2 n=%d diag: Nu_top=%.6f Nu_bot=%.6f Vrms=%.6f avgT=%.6f usurf_rms=%.6f usurf_max=%.6f\n",
        n, res.Nu_top_avg, res.Nu_bot_avg, res.Vrms_avg, avgT, urms, umax)

