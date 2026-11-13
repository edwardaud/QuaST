using QuaST
using Printf

m = make_mesh(1.0, 8, 8, 4; zgrading=:uniform)
fd = make_fedata(4, m.q)
xs, zs = node_coords(m, fd.gll_v)
nv = m.nvx * m.nvz
theta0 = [0.01 * cospi(xs[mod(k-1,m.nvx)+1]) * sinpi(zs[div(k-1,m.nvx)+1]) for k in 1:nv]
pr = Params(Ra=100.0, H=0.0)
tp = TosiParams(1e5, 1.0, 0.0, 0.0, 1e-6, 10.0, 1e-1, 1.0, false)
res = run_steady(m, fd, pr; theta0=theta0, maxiter=100, tol=1e-9, relax=0.5,
                 verbose=true, linear=:direct, velocity_space=:taylor_hood,
                 res_bar=1e-9, tosi=tp)
@printf("TOSI Case1: Nu_top=%.6f Nu_bot=%.6f Vrms=%.6f iters=%d\n",
        res.Nu_top_avg, res.Nu_bot_avg, res.Vrms_avg, res.steps)


