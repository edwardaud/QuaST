# G150: the numerical regression snapshot — Poisson, Stokes and a Blankenbach 1a window.
using Test, QuaST, LinearAlgebra

# G150: numerical regression snapshot.
#
# Four small, deterministic cases pin the numbers that the physics and the
# discretisation produce, so a refactor cannot silently drift.  These are not
# "manufactured solution" gates (those check convergence); they record the
# current values of integrated diagnostics, with a tolerance loose enough to
# survive BLAS/SuiteSparse variation but tight enough to catch a real change:
#
#   A. Poisson, 5x2 base (Lx=2.5), level 3, p=2: true L2 error of the
#      manufactured solution (independent Gauss quadrature, not the SEM mass
#      matrix -- the latter measures against the nodal interpolant).
#   B. variable-viscosity Stokes driver, same mesh: the exact polynomial
#      u=(2xy,-y^2), p=0 is reproduced to round-off; records the residual.
#   C. Blankenbach 1a, unit square 16x16, 10 steps: Nu_top/Nu_bot/Vrms.
#   D. Blankenbach 2b, 5x2 base level 2, 20 steps from ASPECT's IC, EV on:
#      the early transient values of Nu_top/Nu_bot/Vrms.
#
# The expected values were recorded on 2026-11-03 with Julia 1.12.7; see
# `out_julia/agent/snapshot_measure.jl` for the measuring script.
#
# Both driver calls pin `linear=:direct`: this gate records direct-path numbers,
# and case C is a 256-leaf / 2 467-dof tree, above the selector's 2 000-dof
# threshold, so the `:auto` default would resolve to `:mg`.

function _snapshot_l2_true(sq, qt, u, uex; nq::Int=30)
    xq, wq = gauss_legendre(nq)
    s = 0.0
    leaves = Set(sq.leaves)
    for lf in qt.leaves
        hx = QuaST.leaf_hx(sq, lf[1]); hy = QuaST.leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        for j in 1:nq, i in 1:nq
            x = x0 + 0.5 * (xq[i] + 1) * hx
            y = y0 + 0.5 * (xq[j] + 1) * hy
            e = QuaST._semquad_eval_at(sq, u, x, y, leaves) - uex(x, y)
            s += 0.25 * hx * hy * wq[i] * wq[j] * e * e
        end
    end
    return sqrt(s)
end

function _snapshot_refined(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _snapshot_case2b(path)
    xs = Float64[]; ys = Float64[]; ts = Float64[]
    for line in eachline(path)
        startswith(strip(line), "#") && continue
        isempty(strip(line)) && continue
        parts = split(strip(line))
        length(parts) < 3 && continue
        push!(xs, parse(Float64, parts[1])); push!(ys, parse(Float64, parts[2]))
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

function _snapshot_interp(ux, uy, T, x, y)
    i = clamp(searchsortedlast(ux, x), 1, length(ux) - 1)
    j = clamp(searchsortedlast(uy, y), 1, length(uy) - 1)
    tx = clamp((x - ux[i]) / (ux[i + 1] - ux[i]), 0.0, 1.0)
    ty = clamp((y - uy[j]) / (uy[j + 1] - uy[j]), 0.0, 1.0)
    return (1 - tx) * (1 - ty) * T[i, j] + tx * (1 - ty) * T[i + 1, j] +
           (1 - tx) * ty * T[i, j + 1] + tx * ty * T[i + 1, j + 1]
end

@testset "G150 regression snapshot" begin
    # ---- A. Poisson on the 5x2 / Lx=2.5 forest ----
    qt = _snapshot_refined(3; cx=5, cy=2)
    sq = make_semquad(qt, 2)
    k = pi / qt.Lx
    uex = (x, y) -> sin(k * x) * sin(pi * y)
    f = (x, y) -> (k^2 + pi^2) * uex(x, y)
    u, _ = solve_semquad_poisson(sq, qt, f, (x, y) -> 0.0)
    eA = _snapshot_l2_true(sq, qt, u, uex)
    @test eA ≈ 3.448761010547007e-05 rtol = 1.0e-6

    # ---- B. variable-viscosity Stokes, exact polynomial ----
    ss = make_semstokes(qt, 2)
    uxex = (x, y) -> 2.0 * x * y
    uzex = (x, y) -> -y^2
    us, _, _, _, _ = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0,
                                              (x, y) -> 0.0, (x, y) -> 2.0,
                                              uxex, uzex; bc=:rigid)
    ns = ss.su.N
    errB = 0.0
    for (g, c) in enumerate(ss.su.coords)
        errB = max(errB, abs(us[g] - uxex(c...)), abs(us[ns + g] - uzex(c...)))
    end
    @test errB < 1.0e-11
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-11

    # ---- C. Blankenbach 1a, 16x16, 10 steps ----
    qt1 = _snapshot_refined(4)
    ss1 = make_semstokes(qt1, 2)
    pr1 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                 Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th1 = [0.01 * cos(pi * x) * sin(pi * y) for (x, y) in ss1.su.coords]
    for (i, (x, y)) in enumerate(ss1.su.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th1[i] = 0.0)
    end
    st1 = run_semquad_simulation(qt1, pr1, th1; p=2, tstop=2.0e-3, dt0=2.0e-4,
                                 maxsteps=10, linear=:direct, verbose=false)
    nt1, nb1 = semquad_nu_walls(ss1.su, st1.qt, st1.theta, pr1)
    vr1 = semquad_vrms(ss1.su, st1.qt, st1.ux, st1.uz)
    @test st1.t ≈ 2.0e-3 rtol = 1.0e-12
    @test nt1 ≈ 1.000761776406918 rtol = 1.0e-6
    @test nb1 ≈ 1.000761776406919 rtol = 1.0e-6
    @test vr1 ≈ 2.677656198451545 rtol = 1.0e-6
    # at this early time the flow is still reflection-symmetric: top and bottom
    # fluxes must agree to solver accuracy
    @test nt1 ≈ nb1 rtol = 1.0e-10

    # ---- D. Blankenbach 2b, 5x2 level 2, 20 steps from ASPECT's IC ----
    data = joinpath(@__DIR__, "..", "examples", "data",
                    "initial_temperature_case2b.txt")
    uxt, uyt, Tt = _snapshot_case2b(data)
    qt2 = _snapshot_refined(2; cx=5, cy=2)
    ss2 = make_semstokes(qt2, 2)
    pr2 = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                 eta_b=9.70406052783923433184,
                 eta_c=4.1588830833596718565,
                 Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th2 = [_snapshot_interp(uxt, uyt, Tt, x, y) - (1.0 - y)
           for (x, y) in ss2.su.coords]
    for (i, (x, y)) in enumerate(ss2.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th2[i] = 0.0)
    end
    st2 = run_semquad_simulation(qt2, pr2, th2; p=2, tstop=1.0,
                                 dt0=1.0e-4, cfl=5.0, maxsteps=20,
                                 ev=0.11, ev_cmax=0.052, linear=:direct,
                                 verbose=false)
    nt2, nb2 = semquad_nu_walls(ss2.su, st2.qt, st2.theta, pr2)
    vr2 = semquad_vrms(ss2.su, st2.qt, st2.ux, st2.uz)
    @test st2.steps == 20
    @test nt2 ≈ 7.029996422723808 rtol = 1.0e-6
    @test nb2 ≈ 5.585292640587811 rtol = 1.0e-6
    @test vr2 ≈ 1.259039928721316e+02 rtol = 1.0e-6
end
