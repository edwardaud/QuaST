using Test, MantleSEM, LinearAlgebra

# G159: `linear=:mg` with the three-factor w-BFBT pressure block must converge on
# developed 2b states, not just on the initial condition.
#
# The BFBT diagonal used to be the *row sums* of the reduced velocity mass,
# `D = diag(M_u(sqrt(eta)) * 1)`.  On free-slip boundaries those row sums are
# not guaranteed positive.  Three states matter here (all deterministic, from
# the ASPECT 2b IC on the L3 40x16 tree):
#   * step 1/2/5/20: row sums positive, the old row-sum code converges;
#   * step 10 and step 40: several rows of `Mv_red` are <= 1e-25 (and the
#     smallest is negative), so `1/max(row, 1e-30)` produces 1e30 entries,
#     ||S1|| ~ 1e26, and GMRES diverges (100-300 iterations, residual 1e13-1e17).
# Using the mass diagonal instead (`diag(Mv_red)`, strictly positive for a mass
# matrix) gives ~36 iterations and a solution that matches the direct saddle
# solve on every state.
#
# The gate therefore builds the state after 10 direct steps and requires the
# BFBT solve to agree with `:direct`.  Before the fix this fails with
# `res ~ 1e16, du ~ 1`; after the fix it passes with ~36 iterations.

function _g159_tree()
    qt = Quadtree(3; cells_x=5, cells_y=2)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g159_theta(sq)
    data = joinpath(@__DIR__, "..", "scripts", "data",
                    "initial_temperature_case2b.txt")
    xs = Float64[]; ys = Float64[]; ts = Float64[]
    for line in eachline(data)
        s = strip(line)
        (startswith(s, "#") || isempty(s)) && continue
        p = split(s)
        length(p) < 3 && continue
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
    return th
end

@testset "G159 2b L3 MG/BFBT does not diverge" begin
    qt = _g159_tree()
    ss = make_semstokes(qt, 2)
    sq = ss.su
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                eta_b=9.70406052783923433184,
                eta_c=4.1588830833596718565,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th0 = _g159_theta(sq)

    # A developed state where the old lumped D had non-positive rows (the exact
    # state is reproducible: 10 fixed-dt direct steps from the ASPECT IC).
    st = run_semquad_simulation(qt, pr, th0; p=2, tstop=1.0, dt0=1.0e-3,
                                cfl=0.0, maxsteps=10, verbose=false,
                                linear=:direct)
    th = st.theta
    ssd = st.ss
    leaves = Set(st.qt.leaves)
    Tfun = (x, y) -> MantleSEM._semquad_eval_at(ssd.su, th, x, y, leaves)
    eta = (x, y) -> semquad_eta_at(ssd.su, th, x, y, pr, leaves)
    fz = (x, y) -> pr.Ra * Tfun(x, y)
    z2 = (x, y) -> 0.0

    u0, _, _, _, _ = solve_semquad_stokes_var(ssd, st.qt, eta, z2, fz, z2, z2;
                                              bc=:free, linear=:direct)
    u1, _, _, _, _ = solve_semquad_stokes_var(ssd, st.qt, eta, z2, fz, z2, z2;
                                              bc=:free, linear=:mg,
                                              mg_pressure_mode=:bfbt,
                                              mg_maxit=300)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-8
    @test LAST_SEMSTOKES_SCHUR_ITERS[] < 150
    @test norm(u1 - u0) / norm(u0) < 1.0e-8

    # `:auto` selects BFBT for the 16384x contrast, so the default production
    # path must pass the same regression.
    u2, _, _, _, _ = solve_semquad_stokes_var(ssd, st.qt, eta, z2, fz, z2, z2;
                                              bc=:free, linear=:mg,
                                              mg_pressure_mode=:auto,
                                              mg_maxit=300)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-8
    @test LAST_SEMSTOKES_SCHUR_ITERS[] < 150
    @test norm(u2 - u0) / norm(u0) < 1.0e-8
end
