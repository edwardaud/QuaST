# G160: the MG-accelerated steady Newton agrees with the direct path.
using Test, QuaST, LinearAlgebra

# G160: the steady Newton-Krylov solve with the geometric-MG saddle solver.
#
# `semquad_newton_steady(...; linear=:mg)` replaces the direct saddle LU of
# `semquad_steady_step` with the same block-triangular MG/BFBT preconditioner
# used by the transient `linear=:mg` path (`_semquad_newton_saddle`).  The
# Jacobian-vector product reuses that solver, so the whole Newton iteration is
# factorisation-free.
#
# The gate compares it with the historical direct-LU Newton on the 2b problem:
# same convergence, same iteration count, same steady state.  The L2 tree keeps
# the gate fast; the 8800-leaf wall case is exercised by
# `examples/benchmark_cases.jl 2b 6 ...` (documented in
# docs/dev/SOLVER.md).

const _B2B = 9.70406052783923433184
const _C2B = 4.1588830833596718565

function _g160_case(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=_B2B, eta_c=_C2B,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    data = joinpath(@__DIR__, "..", "examples", "data",
                    "initial_temperature_case2b.txt")
    xs = Float64[]; ys = Float64[]; ts = Float64[]
    for line in eachline(data)
        s = strip(line)
        (startswith(s, "#") || isempty(s)) && continue
        p = split(s); length(p) < 3 && continue
        push!(xs, parse(Float64, p[1])); push!(ys, parse(Float64, p[2]))
        push!(ts, parse(Float64, p[3]))
    end
    ux = sort(unique(xs)); uy = sort(unique(ys))
    T = zeros(Float64, length(ux), length(uy))
    for k in eachindex(xs)
        T[searchsortedfirst(ux, xs[k]), searchsortedfirst(uy, ys[k])] = ts[k]
    end
    interp(x, y) = begin
        i = clamp(searchsortedlast(ux, x), 1, length(ux) - 1)
        j = clamp(searchsortedlast(uy, y), 1, length(uy) - 1)
        tx = clamp((x - ux[i]) / (ux[i + 1] - ux[i]), 0.0, 1.0)
        ty = clamp((y - uy[j]) / (uy[j + 1] - uy[j]), 0.0, 1.0)
        (1 - tx) * (1 - ty) * T[i, j] + tx * (1 - ty) * T[i + 1, j] +
        (1 - tx) * ty * T[i, j + 1] + tx * ty * T[i + 1, j + 1]
    end
    th = [interp(x, y) - (1 - y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1 - 1.0e-12) && (th[i] = 0.0)
    end
    return ss, sq, pr, th
end

function _g160_run(qt, ss, sq, pr, th; linear)
    res = semquad_newton_steady(sq, qt, ss, pr, th; dt=1.0e-2, tol=1.0e-8,
                                maxit=15, krylov=40, restart=20, verbose=false,
                                linear=linear)
    u = res.step.u
    nt, nb = semquad_nu_walls(sq, qt, res.theta, pr)
    vr = semquad_vrms(sq, qt, u[1:sq.N], u[(sq.N + 1):(2sq.N)])
    return res, nt, nb, vr
end

@testset "G160 MG-accelerated steady Newton" begin
    qt = Quadtree(2; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss, sq, pr, th = _g160_case(qt)

    rd, ntd, nbd, vrd = _g160_run(qt, ss, sq, pr, th; linear=:direct)
    rm, ntm, nbm, vrm = _g160_run(qt, ss, sq, pr, th; linear=:mg)

    @test rd.converged && rm.converged
    @test rd.step.solver.kind === :direct
    @test rm.step.solver.kind === :mg
    @test abs(rm.iters - rd.iters) <= 1
    @test norm(rm.theta - rd.theta) / norm(rd.theta) < 1.0e-5
    @test abs(ntm - ntd) / abs(ntd) < 1.0e-5
    @test abs(nbm - nbd) / abs(nbd) < 1.0e-5
    @test abs(vrm - vrd) / abs(vrd) < 1.0e-5
end
