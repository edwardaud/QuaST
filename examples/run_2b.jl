# Blankenbach (1989) case 2b: 2.5 x 1 box, temperature+depth dependent viscosity.
#
# Source of every parameter:
#   benchmarks/blankenbach/base_case2b.prm (ASPECT 2.5.0-pre, main 340c0d018)
#     Geometry: box 2.5 x 1 (X repetitions 5, Y repetitions 2)
#     Ra = 1e4, Di = 0, TALA = false, Boussinesq
#     Viscosity temperature prefactor = 9.70406052783923433184 = ln(16384)
#     Viscosity depth prefactor       = 4.1588830833596718565  = ln(64)
#     eta = (1/Ra) exp(-b T' + c depth)   (material_model/nondimensional.cc:73-75)
#     Initial T = (1-z) + 0.01 cos(pi x / 2.5) sin(pi z)
#     Free slip on all walls; T = 1 bottom, T = 0 top; side walls insulating.
#   Reference values (ASPECT reference_statistics.txt / Blankenbach et al. 1989):
#     Nu = 6.9299, Vrms = 171.755
#
# The 1/Ra prefactor of the viscosity is absorbed by the body-force scaling, as
# in the validated 2a case; eta = exp(-b T + c (1-z)).
#
# Usage: julia --project=. examples/run_2b.jl [nxe] [nze] [stage...]
using QuaST
using Printf

function main()
    nxe = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 40
    nze = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
    stages = length(ARGS) >= 3 ? parse.(Float64, ARGS[3:end]) : [0.25, 0.5, 0.75, 1.0]

    b = log(16384.0)
    c = log(64.0)
    m = make_mesh(2.5, nxe, nze, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    theta = [0.01 * cospi(xs[mod(k-1,m.nvx)+1] / 2.5) * sinpi(zs[div(k-1,m.nvx)+1])
             for k in 1:m.Nscalar]
    @printf("# 2b mesh %d x %d (p4, %d x %d nodes), b=%.8f c=%.8f, contrast=%.3e\n",
            nxe, nze, m.nvx, m.nvz, b, c, exp(b) * exp(c))
    @printf("# reference: Nu=6.9299  Vrms=171.755\n")
    for s in stages
        pr = Params(Ra=1.0e4, eta_law=:expTz, eta_b=s*b, eta_c=s*c, bc=:free)
        res = run_steady(m, fd, pr; theta0=theta, maxiter=80, tol=1.0e-8,
                         relax=0.5, verbose=false, advection=:galerkin,
                         linear=:direct, res_bar=1.0e-9)
        theta = res.theta
        @printf("STAGE s=%.3f  Nu_top=%.6f Nu_bot=%.6f Vrms=%.6f it=%d  (ref 6.9299 / 171.755)\n",
                s, res.Nu_top, res.Nu_bot, res.Vrms, res.steps)
        flush(stdout)
    end
end

main()
