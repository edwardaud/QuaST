# G36: BDF2 energy step on the quadtree -- remediation batch P0 part 1,
# AUDIT_VV F3.
#
# Pure diffusion of theta = exp(-2 pi^2 t) cos(pi x) sin(pi y): zero normal
# derivative on the insulating sides, zero value on the top/bottom walls.  The
# dt study isolates the *time* order (the spatial discretisation is fixed).
#
# **Strengthened in this batch.**  The old gate compared *two* dt differences
# and asserted `log2(d12/d23) > 1.5` -- a 1.6-order time integrator passes
# although the scheme is BDF2 (expected order 2; AUDIT_VV F3).  The gate now:
#   * runs a four-level dt ladder (dt, dt/2, dt/4, dt/8 with the same horizon)
#     and computes the observed order of the error against the exact solution
#     on **every** consecutive pair;
#   * asserts every observed order is >= 1.7 (G269-style margin) and prints the
#     table;
#   * includes a **first-order control**: the same solver with the backward-
#     Euler start forced at *every* step (passing `first=true` every step).  Its
#     measured order must be <= 1.4 -- this proves the order measurement itself
#     discriminates, so the BDF2 assertion is not a vacuous "any ratio" check.
#     The control is a genuine wrong-scheme implementation, not a tautology.
#   * keeps the exact-solution tolerance of the old gate (< 0.05).
#
# Old threshold: 1.5 (one ratio).  New: 1.7 (BDF2, all pairs) with the
# first-order control <= 1.4.  Measured (this campaign): BDF2 orders 2.060,
# 2.158, 2.310; backward-Euler control orders 0.986, 0.995, 1.002.
# (A fifth dt = 1.25e-4 was measured and deliberately excluded: the spatial
# error floor is ~9.5e-6 there, so the time error stops resolving and its
# measured order collapses to 0.165 -- pinning it would pin the floor, not the
# scheme.  The four-level ladder stops where the time error is still dominant.)
using Test
using LinearAlgebra
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G36 quadtree BDF2 energy step" begin
    L = 3
    qt = uniform_tree(L)
    sq = make_semquad(qt, 2)
    pr = Params(Ra=1.0, H=0.0, eta_law=:const, bc=:free, supg=0.0)
    th0 = [cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2]) for i in 1:sq.N]
    exactf(t, x, y) = exp(-2 * pi^2 * t) * cospi(x) * sinpi(y)

    # `first_step` is the BDF2 start switch; forcing it every step turns the
    # scheme into backward Euler (the first-order control).
    function run(dt::Float64, nsteps::Int; backward_euler::Bool=false)
        th = copy(th0)
        thm1 = copy(th0)
        t = 0.0
        for s in 1:nsteps
            thn = solve_semquad_energy(sq, qt, pr, th, thm1,
                                       zeros(Float64, sq.N), zeros(Float64, sq.N),
                                       dt, dt, backward_euler || s == 1;
                                       advection=:galerkin)
            thm1 = th
            th = thn
            t += dt
        end
        return th, t
    end

    dts = [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4]
    errs = Float64[]; tfinal = Float64[]
    errs_be = Float64[]; tfinal_be = Float64[]
    for (i, dt) in enumerate(dts)
        n = 10 * 2^(i - 1)
        th, t = run(dt, n)
        ex = [exactf(t, sq.coords[j][1], sq.coords[j][2]) for j in 1:sq.N]
        push!(errs, norm(th - ex) / norm(ex)); push!(tfinal, t)
        thb, tb = run(dt, n; backward_euler=true)
        exb = [exactf(tb, sq.coords[j][1], sq.coords[j][2]) for j in 1:sq.N]
        push!(errs_be, norm(thb - exb) / norm(exb)); push!(tfinal_be, tb)
        @test abs(t - 0.02) < 1.0e-14          # same horizon for every rung
        @test abs(tb - 0.02) < 1.0e-14
    end
    # keep the arrays in sync with the printed variable names
    ords = [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
    ords_be = [log2(errs_be[k] / errs_be[k + 1]) for k in 1:(length(errs_be) - 1)]
    println("G36 BDF2 energy step, dt ladder: ", [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4])
    println("  BDF2 errors = ", errs)
    println("  BDF2 orders = ", round.(ords, digits=3))
    println("  backward-Euler control errors = ", errs_be)
    println("  backward-Euler control orders = ", round.(ords_be, digits=3))
    for k in 1:(length(errs) - 1)
        @test errs[k + 1] < errs[k]
        @test errs_be[k + 1] < errs_be[k]
    end
    @test all(isfinite, errs) && all(isfinite, errs_be)
    # BDF2: every observed order is the second-order design within the margin
    for o in ords
        @test o >= 1.7
    end
    # the first-order control is genuinely first order and below the BDF2 gate:
    # this is the non-vacuity witness of the order measurement itself.
    for o in ords_be
        @test o <= 1.4
    end
    @test minimum(ords) > maximum(ords_be) + 0.5
    # the old exact-solution check is kept (at the coarsest ladder state)
    th, t = run(5.0e-4, 40)
    ex = [exactf(t, sq.coords[i][1], sq.coords[i][2]) for i in 1:sq.N]
    @test norm(th - ex) / norm(ex) < 0.05
end
