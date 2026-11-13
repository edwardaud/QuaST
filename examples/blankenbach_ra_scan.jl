using QuaST
using Printf

m = make_mesh(1.0, 8, 8, 4; zgrading=:uniform)
fd = make_fedata(4, m.q)
xs, zs = node_coords(m, fd.gll_v)
nv = m.nvx * m.nvz
theta0 = [0.01 * sinpi(xs[mod(k-1,m.nvx)+1]) * sinpi(zs[div(k-1,m.nvx)+1])
          for k in 1:nv]
for Ra in (3.0e3, 1.0e4, 3.0e4)
    pr = Params(Ra=Ra, H=0.0)
    res = run_steady(m, fd, pr; theta0=theta0, maxiter=80, tol=1e-9,
                     relax=0.5, verbose=false, linear=:direct,
                     velocity_space=:taylor_hood, res_bar=1e-9)
    @printf("Ra=%.1e Nu=%.5f Vrms=%.4f\n", Ra, res.Nu_top_avg, res.Vrms_avg)
end
