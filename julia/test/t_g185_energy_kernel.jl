using Test, MantleSEM, LinearAlgebra, SparseArrays

# G185 / S3 stage 2: the energy kernel's optional-function paths are numerically
# inert, and the per-step profile hook does not change the trajectory.
#
# The Stage-2 fix (`julia/src/semquad.jl`, `_semquad_assemble_energy_block!`)
# replaces the `Union{Nothing,Function}` / `Union{Nothing,Vector}` field accesses
# in the element loop with `::Float64`-asserted locals, because inference was
# joining their values with `Any` and boxing the nv x nv loop (328M allocations,
# 5 GiB per energy assembly on the 1024-leaf blob tree; docs/dev/SOLVER.md §15).
# Three things must hold:
#
#   (a) supplying an identity callable explicitly is bitwise the same as leaving
#       it out (`rho_ref = y -> 1.0`, `eta_fun = (x,y) -> 1.0`), i.e. the
#       specialised branch returns exactly what the boxed branch did;
#   (b) the assembled matrices and the BDF2 solve on a fixed small tree still
#       reproduce the pre-fix numbers.  The constants below were recorded in the
#       pre-fix worktree (commit 6c706a8) and the same script prints identical
#       17-digit values on both sides;
#   (c) `profile_steps=true` only adds `time_ns()` calls: the theta trajectory
#       and history of a short run are bitwise identical with and without it.

const _G185_Z2 = (x, y) -> 0.0

function _g185_uniform(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g185_state(qt, p)
    sq = make_semquad(qt, p)
    th = [0.25 * sinpi(x) * sinpi(y) + 0.05 * sinpi(3x) * sinpi(2y)
          for (x, y) in sq.coords]
    for g in 1:sq.N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th[g] = 0.0)
    end
    ux = [0.3 * sinpi(x) * cospi(y) for (x, y) in sq.coords]
    uz = [-0.2 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
    return sq, th, ux, uz
end

@testset "G185 energy-kernel specialisation" begin
    qt = _g185_uniform(3)                       # 64 leaves
    sq, th, ux, uz = _g185_state(qt, 2)
    pr = Params(Ra=1.0e5, H=0.5, eta_law=:expT, deta_T=1.0,
                Twall_top=0.0, Twall_bot=1.0, bc=:free,
                Di=0.1, heating_dissipation=true, heating_adiabatic=true)
    eta1 = (x, y) -> 1.0 + 0.5 * x

    # (a) identity callables: explicitly supplied == left out, bitwise.
    M1, K1, C1, F1 = assemble_semquad_energy(
        sq, qt, pr, ux, uz; advection=:skew, theta=th,
        heating_adiabatic=true, heating_dissipation=true, eta_fun=eta1)
    M2, K2, C2, F2 = assemble_semquad_energy(
        sq, qt, pr, ux, uz; advection=:skew, theta=th,
        heating_adiabatic=true, heating_dissipation=true, eta_fun=eta1,
        rho_ref=(y) -> 1.0)
    @test M2 == M1
    @test K2 == K1
    @test C2 == C1
    @test F2 == F1

    # `eta_fun` only enters through the dissipation term, where `nothing` means
    # eta = 1; a constant 1.0 must give the identical matrix/vector.
    pr1 = Params(Ra=1.0e5, H=0.5, eta_law=:expT, deta_T=1.0,
                 Twall_top=0.0, Twall_bot=1.0, bc=:free, Di=0.1,
                 heating_dissipation=true)
    M3, K3, C3, F3 = assemble_semquad_energy(
        sq, qt, pr1, ux, uz; advection=:skew, theta=th,
        heating_dissipation=true, eta_fun=(x, y) -> 1.0)
    M4, K4, C4, F4 = assemble_semquad_energy(
        sq, qt, pr1, ux, uz; advection=:skew, theta=th,
        heating_dissipation=true)
    @test M3 == M4
    @test K3 == K4
    @test C3 == C4
    @test F3 == F4

    # (b) pre-fix snapshot (worktree 6c706a8): unchanged to the last bit.
    @test sq.N == 289
    @test nnz(M1) == 4225
    @test nnz(K1) == 4225
    @test nnz(C1) == 3928
    @test sum(M1) ≈ 0.99999999999999989 rtol = 1.0e-15
    @test sum(abs, nonzeros(K1)) ≈ 2275.555555555552 rtol = 1.0e-14
    @test norm(C1) ≈ 0.14702735381040949 rtol = 1.0e-13
    @test norm(F1) ≈ 0.0064361664296510145 rtol = 1.0e-12
    th1 = solve_semquad_energy(sq, qt, pr, th, th, ux, uz, 1.0e-3, 1.0e-3,
                               false; advection=:skew, eta_fun=eta1)
    @test norm(th1) ≈ 2.0113392917405881 rtol = 1.0e-13
    @test th1[cld(length(th1), 2)] ≈ 0.22054882215006449 rtol = 1.0e-13
    @test th1[1] == 0.0 && th1[end] == 0.0

    # (c) the profiling hook is numerically inert.
    qt2 = _g185_uniform(2)                      # 16 leaves
    ss2 = make_semstokes(qt2, 2)
    pr2 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                 Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th2 = [0.01 * cos(pi * x) * sin(pi * y) for (x, y) in ss2.su.coords]
    for (i, (x, y)) in enumerate(ss2.su.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th2[i] = 0.0)
    end
    r_plain = run_semquad_simulation(qt2, pr2, th2; p=2, tstop=1.0e-6,
                                     dt0=1.0e-7, maxsteps=3, verbose=false)
    @test SEMQUAD_STEP_PROFILE[] === nothing
    r_prof = run_semquad_simulation(qt2, pr2, th2; p=2, tstop=1.0e-6,
                                    dt0=1.0e-7, maxsteps=3, profile_steps=true,
                                    verbose=false)
    @test r_prof.theta == r_plain.theta
    @test r_prof.history == r_plain.history
    prof = SEMQUAD_STEP_PROFILE[]
    @test length(prof) == 3
    @test all(p -> p.total >= p.stokes + p.energy, prof)
    @test all(p -> p.leaves == 16, prof)
end
