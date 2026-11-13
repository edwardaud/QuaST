using Test, MantleSEM, LinearAlgebra

# G155: analytic tangent-linear Jacobian and Newton-Krylov for the steady
# theta map of route C.
#
# The Jacobian is checked against a central finite difference of the *same*
# step map (not of a linearised residual), then the Newton iteration must reach
# 1e-9 in a handful of iterations from the Blankenbach 2b initial condition.
# The finite-difference version of this Newton stalled at ~1e-3; that is the
# regression this gate locks.

function _g155_case2b(path)
    xs = Float64[]; ys = Float64[]; ts = Float64[]
    for line in eachline(path)
        s = strip(line)
        (startswith(s, "#") || isempty(s)) && continue
        p = split(s)
        length(p) < 3 && continue
        push!(xs, parse(Float64, p[1]))
        push!(ys, parse(Float64, p[2]))
        push!(ts, parse(Float64, p[3]))
    end
    ux = sort(unique(xs)); uy = sort(unique(ys))
    T = zeros(Float64, length(ux), length(uy))
    for k in eachindex(xs)
        T[searchsortedfirst(ux, xs[k]), searchsortedfirst(uy, ys[k])] = ts[k]
    end
    return ux, uy, T
end

function _g155_interp(ux, uy, T, x, y)
    i = clamp(searchsortedlast(ux, x), 1, length(ux) - 1)
    j = clamp(searchsortedlast(uy, y), 1, length(uy) - 1)
    tx = clamp((x - ux[i]) / (ux[i + 1] - ux[i]), 0.0, 1.0)
    ty = clamp((y - uy[j]) / (uy[j + 1] - uy[j]), 0.0, 1.0)
    return (1 - tx) * (1 - ty) * T[i, j] + tx * (1 - ty) * T[i + 1, j] +
           (1 - tx) * ty * T[i, j + 1] + tx * ty * T[i + 1, j + 1]
end

@testset "G155 analytic tangent Newton" begin
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                eta_b=9.70406052783923433184,
                eta_c=4.1588830833596718565,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)

    # derivative of eta(T,z) against a central difference
    T = 0.4; z = 0.3; he = 1.0e-6
    fdeta = (eta_at(T + he, z, pr)[1] - eta_at(T - he, z, pr)[1]) / (2he)
    @test deta_dT_at(T, z, pr) ≈ fdeta rtol = 1.0e-6

    # a small 2b mesh (5x2 base, level 2: 160 leaves, N=4000 nodes)
    qt = Quadtree(2; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    data = joinpath(@__DIR__, "..", "scripts", "data",
                    "initial_temperature_case2b.txt")
    ux, uy, Tt = _g155_case2b(data)
    th = [_g155_interp(ux, uy, Tt, x, y) - (1.0 - y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[i] = 0.0)
    end

    # tangent check at the initial state: deterministic, wall-free direction
    N = sq.N
    v = [sin(2.0 * pi * x / 2.5) * sin(pi * y) for (x, y) in sq.coords]
    for g in 1:N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (v[g] = 0.0)
    end
    v ./= norm(v)
    st = semquad_steady_step(sq, qt, ss, pr, th; dt=1.0e-2)
    Jv = semquad_steady_jv(sq, qt, ss, pr, th, v, st; dt=1.0e-2)
    eps = 1.0e-6
    stp = semquad_steady_step(sq, qt, ss, pr, th .+ eps .* v; dt=1.0e-2)
    stm = semquad_steady_step(sq, qt, ss, pr, th .- eps .* v; dt=1.0e-2)
    fd = (stp.theta .- stm.theta) ./ (2.0 * eps)
    @test norm(Jv - fd) / norm(fd) < 1.0e-4

    res = semquad_newton_steady(sq, qt, ss, pr, th; dt=1.0e-2,
                                tol=1.0e-8, maxit=8, krylov=40, restart=20)
    @test res.converged
    @test res.iters <= 10
    @test norm(res.step.theta .- res.theta) < 1.0e-8
    @test length(res.history) == res.iters + 1
    @test res.history[end] < 1.0e-8

    # starting at the fixed point: no Newton step, residual already below tol
    res2 = semquad_newton_steady(sq, qt, ss, pr, res.theta; dt=1.0e-2,
                                 tol=1.0e-8, maxit=4, krylov=20, restart=10)
    @test res2.converged
    @test res2.iters <= 1

    # lagged-EV plumbing: the map recomputes the elementwise EV field and the
    # JVP freezes it (the nonlinear EV steady solve is exercised at L3/L4 by the
    # driver, not on this small gate mesh).
    st_ev = semquad_steady_step(sq, qt, ss, pr, th; dt=1.0e-2, ev=0.11)
    @test st_ev.parts.ev_field !== nothing
    @test length(st_ev.parts.ev_field) == length(qt.leaves)
    @test all(isfinite, st_ev.parts.ev_field)
    Jv_ev = semquad_steady_jv(sq, qt, ss, pr, th, v, st_ev; dt=1.0e-2)
    @test all(isfinite, Jv_ev)

    # a small hanging-node tree: the same tangent through the constrained
    # Cfree/Cu maps
    qtn = Quadtree(2; cells_x=2, cells_y=1)
    refine!(qtn, (2, 0, 0))
    refine!(qtn, (2, 0, 0))
    balance!(qtn)
    ssn = make_semstokes(qtn, 2)
    sqn = ssn.su
    thn = [0.05 * sin(pi * x / 2.0) * sin(pi * y) for (x, y) in sqn.coords]
    for (i, (x, y)) in enumerate(sqn.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (thn[i] = 0.0)
    end
    vn = [sin(pi * x / 2.0) * sin(pi * y) for (x, y) in sqn.coords]
    for g in 1:sqn.N
        y = sqn.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (vn[g] = 0.0)
    end
    vn ./= norm(vn)
    stn = semquad_steady_step(sqn, qtn, ssn, pr, thn; dt=1.0e-2)
    Jvn = semquad_steady_jv(sqn, qtn, ssn, pr, thn, vn, stn; dt=1.0e-2)
    stnp = semquad_steady_step(sqn, qtn, ssn, pr, thn .+ eps .* vn; dt=1.0e-2)
    stnm = semquad_steady_step(sqn, qtn, ssn, pr, thn .- eps .* vn; dt=1.0e-2)
    fdn = (stnp.theta .- stnm.theta) ./ (2.0 * eps)
    @test norm(Jvn - fdn) / norm(fdn) < 1.0e-4
end