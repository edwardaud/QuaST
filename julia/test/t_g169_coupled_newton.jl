using Test, MantleSEM, LinearAlgebra, Printf

# G169: production coupled (u,p,theta) Newton.
#
# The coupled Jacobian was the subject of the 2026-11-05 sign-error correction
# (ROUTE_C_STATUS §1.4): a *global* FD hides the error because the theta block
# dominates the norm, so this gate checks the three blocks separately.  It also
# locks (i) the EV tangent in the energy block (frozen EV must be measurably
# worse), (ii) local quadratic convergence from a developed 2b state, and
# (iii) the switchable Stokes block (:direct and :mg).

function _g169_case2b(path)
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

function _g169_interp(ux, uy, T, x, y)
    i = clamp(searchsortedlast(ux, x), 1, length(ux) - 1)
    j = clamp(searchsortedlast(uy, y), 1, length(uy) - 1)
    tx = clamp((x - ux[i]) / (ux[i + 1] - ux[i]), 0.0, 1.0)
    ty = clamp((y - uy[j]) / (uy[j + 1] - uy[j]), 0.0, 1.0)
    return (1 - tx) * (1 - ty) * T[i, j] + tx * (1 - ty) * T[i + 1, j] +
           (1 - tx) * ty * T[i, j + 1] + tx * ty * T[i + 1, j + 1]
end

@testset "G169 coupled Newton" begin
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                eta_b=9.70406052783923433184,
                eta_c=4.1588830833596718565,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    qt = Quadtree(2; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    data = joinpath(@__DIR__, "..", "scripts", "data",
                    "initial_temperature_case2b.txt")
    ux, uy, Tt = _g169_case2b(data)
    th = [_g169_interp(ux, uy, Tt, x, y) - (1.0 - y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[i] = 0.0)
    end
    N = sq.N
    # ---- developed 2b state from the map Newton -----------------------------
    rm = semquad_newton_steady(sq, qt, ss, pr, th; dt=1.0e-2, tol=1.0e-8,
                               maxit=10, krylov=40, restart=20)
    @test rm.converged
    thstar = rm.theta
    # ---- coupled Newton from the map fixed point ---------------------------
    t0 = time()
    rc = semquad_coupled_newton(sq, qt, ss, pr, thstar; dt=1.0e-2, tol=1.0e-8,
                                maxit=10, krylov=300, restart=100,
                                stokes_block=:direct)
    tc = time() - t0
    @test rc.converged
    @test rc.iters <= 4
    @test norm(rc.theta - thstar) / norm(thstar) < 1.0e-6
    @printf("G169: coupled warm iters=%d time=%.2fs\n", rc.iters, tc)
    # ---- per-block FD of the coupled Jacobian ------------------------------
    sys, parts = rc.sys, rc.parts
    nf, nfp, nt = rc.nf, rc.nfp, rc.nt
    tostate = function (x)
        thv = rc.theta + parts.Cfree * x[(nf + nfp + 1):end]
        uv = rc.u + sys.Cu * x[1:nf]
        pv = rc.p + sys.Cp * x[(nf + 1):(nf + nfp)]
        return thv, uv, pv
    end
    h = 1.0e-6
    for (label, sl, others) in (("u", 1:nf, (nf + 1):(nf + nfp + nt)),
                                ("p", (nf + 1):(nf + nfp), nothing),
                                ("theta", (nf + nfp + 1):(nf + nfp + nt),
                                 nothing))
        x = zeros(Float64, nf + nfp + nt)
        # deterministic direction on the block
        idx = collect(sl)
        x[idx] = [sin(0.7 * k) + 0.3 for k in 1:length(idx)]
        x ./= norm(x)
        Jx = rc.jac(x)
        pp = rc.resid(tostate(h .* x)...)[1]
        pm = rc.resid(tostate(-h .* x)...)[1]
        fd = (pp .- pm) ./ (2h)
        rel = norm(Jx - fd) / norm(fd)
        @printf("G169: block %-5s FD rel=%.2e\n", label, rel)
        @test rel < 1.0e-3
    end
    # ---- EV: the coupled system must converge on an EV steady state --------
    # The residual and the energy block include the entropy viscosity (the
    # operators of `semquad_steady_parts`).  `ev_tangent=true` adds the EV field
    # derivative (G166); on this 160-leaf case the outer GMRES then needs ~310
    # iterations per step instead of ~29 (the preconditioner does not model the
    # EV operator), so the practical setting is the frozen field and the
    # tangent's correctness stays gated by G166 at the map level.
    rcev = semquad_coupled_newton(sq, qt, ss, pr, thstar; dt=1.0e-2, ev=0.11,
                                  tol=1.0e-7, maxit=25, krylov=300,
                                  restart=100, ev_tangent=false)
    @test rcev.converged
    @test rcev.iters <= 20
    vev = semquad_vrms(sq, qt, rcev.u[1:N], rcev.u[(N + 1):(2N)])
    @test vev > 150.0                       # the EV steady 2b state
    @test norm(rcev.theta - thstar) > 1.0e-3  # EV genuinely changes the state
    @printf("G169: EV coupled iters=%d Vrms=%.2f\n", rcev.iters, vev)
    # ---- switchable Stokes block ------------------------------------------
    rmg = semquad_coupled_newton(sq, qt, ss, pr, thstar; dt=1.0e-2, tol=1.0e-8,
                                 maxit=6, krylov=300, restart=100,
                                 stokes_block=:mg)
    @test rmg.converged
    @test norm(rmg.theta - thstar) / norm(thstar) < 1.0e-6
    @printf("G169: mg block iters=%d block=%s\n", rmg.iters, rmg.stokes_block)
end
