using MantleSEM, Printf

# Blankenbach 2b: what does the consistent boundary flux (CBF) actually say on
# the converged state, and does the entropy-viscosity (EV) field change it?
#
# This is the follow-up to the claim "2b CBF is 4.5% low because the EV term is
# missing".  `semquad_ev_nu` switches EV off on wall leaves, and a CBF wall row
# only involves the wall element, so the *physical* EV field cannot change the
# wall flux.  This script measures that directly on a state evolved with EV:
# the CBF value with the stored EV field and the CBF value without it must be
# bitwise identical.  The residual gap to the reference Nu is therefore a
# discretisation property of the CBF functional on this 40x16 mesh, not a
# missing EV term.
#
# Usage: julia --project=julia julia/scripts/blankenbach_2b_cbf_ev.jl [level] [tstop]
const B = 9.70406052783923433184
const C = 4.1588830833596718565
const REF_NU = 6.9299
const REF_VR = 171.755

function load_case2b(path::String)
    xs = Float64[]; ys = Float64[]; ts = Float64[]
    for line in eachline(path)
        startswith(strip(line), "#") && continue
        isempty(strip(line)) && continue
        parts = split(strip(line))
        length(parts) < 3 && continue
        push!(xs, parse(Float64, parts[1]))
        push!(ys, parse(Float64, parts[2]))
        push!(ts, parse(Float64, parts[3]))
    end
    ux = sort(unique(xs)); uy = sort(unique(ys))
    T = zeros(Float64, length(ux), length(uy))
    for k in eachindex(xs)
        i = searchsortedfirst(ux, xs[k]); j = searchsortedfirst(uy, ys[k])
        T[i, j] = ts[k]
    end
    return ux, uy, T
end

function interp_centres(ux, uy, T, x, y)
    i = clamp(searchsortedlast(ux, x), 1, length(ux) - 1)
    j = clamp(searchsortedlast(uy, y), 1, length(uy) - 1)
    tx = clamp((x - ux[i]) / (ux[i + 1] - ux[i]), 0.0, 1.0)
    ty = clamp((y - uy[j]) / (uy[j + 1] - uy[j]), 0.0, 1.0)
    return (1 - tx) * (1 - ty) * T[i, j] + tx * (1 - ty) * T[i + 1, j] +
           (1 - tx) * ty * T[i, j + 1] + tx * ty * T[i + 1, j + 1]
end

function main()
    level = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
    tstop = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.3
    ux, uy, T = load_case2b(joinpath(@__DIR__, "data", "initial_temperature_case2b.txt"))
    qt = Quadtree(level; cells_x=5, cells_y=2)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=B, eta_c=C,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th0 = [interp_centres(ux, uy, T, x, y) - (1.0 - y) for (x, y) in ss.su.coords]
    for (i, (x, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end
    path = joinpath(@__DIR__, "..", "..", "out_julia",
                    "bb2b_L$(level)_cbf_ck.h5")
    @printf("# level %d  nodes=%d  running to t=%.2f (EV on, no AMR)\n",
            level, ss.su.N, tstop)
    flush(stdout)
    # Benchmark reference path: pin the bitwise direct solver, not the default.
    st = run_semquad_simulation(qt, pr, th0; p=2, tstop=tstop, dt0=1.0e-3,
                                linear=:direct,
                                cfl=5.0, maxsteps=300000, verbose=false,
                                ev=0.11, ev_cmax=0.052,
                                checkpoint_path=path, checkpoint_every=100)
    ld = load_semquad_state(path)
    ss2 = make_semstokes(ld.qt, 2)
    nt, nb = semquad_nu_walls(ss2.su, ld.qt, ld.theta, pr)
    vr = semquad_vrms(ss2.su, ld.qt, ld.ux, ld.uz)
    ne = semquad_nu_energy(ss2.su, ld.qt, pr, ld.theta, ld.ux, ld.uz)
    kw = (advection=:galerkin,)
    nt0, nb0 = semquad_nu_cbf(ss2.su, ld.qt, pr, ld.theta, ld.ux, ld.uz; kw...)
    nte, nbe = if ld.ev === nothing
        (NaN, NaN)
    else
        semquad_nu_cbf(ss2.su, ld.qt, pr, ld.theta, ld.ux, ld.uz;
                       ev_field=ld.ev, kw...)
    end
    @printf("t=%.4f steps=%d\n", ld.t, ld.step)
    @printf("wall gradient : Nu_top=%.4f (%+.2f%%)  Nu_bot=%.4f  Vrms=%.2f (%+.2f%%)\n",
            nt, 100 * (nt / REF_NU - 1), nb, vr, 100 * (vr / REF_VR - 1))
    @printf("CBF (no EV)   : Nu_top=%.4f (%+.2f%%)  Nu_bot=%.4f\n",
            nt0, 100 * (nt0 / REF_NU - 1), nb0)
    @printf("CBF (stored EV): Nu_top=%.4f (%+.2f%%)  Nu_bot=%.4f\n",
            nte, 100 * (nte / REF_NU - 1), nbe)
    @printf("difference    : dNu_top=%.3e dNu_bot=%.3e  (must be 0: EV is zero on wall leaves)\n",
            nte - nt0, nbe - nb0)
    @printf("mean of gradient/CBF: %.4f (%+.2f%%)\n",
            0.5 * (nt + nt0), 100 * (0.5 * (nt + nt0) / REF_NU - 1))
    @printf("energy Nu (wall-free): %.4f (%+.2f%%)\n",
            ne, 100 * (ne / REF_NU - 1))
end
main()
