using Test, QuaST
using SparseArrays

@testset "G129 quadrature divergence matrix" begin
    # Step 1 of P3.2d: rebuild B by quadrature without changing the physics.
    # The production matrix is assembled from reference-element matrices scaled by
    # h/2, so the two routes are compared directly; a later stage multiplies the
    # quadrature one by rho(y).
    qt = Quadtree(5)
    for _ in 1:2, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    p, q = 3, 2
    ss = make_semstokes(qt, p)
    su, sp = ss.su, ss.sp
    n = p + 1
    nq1 = q + 1
    nu = su.N

    # reference-element B, assembled exactly as solve_semquad_stokes_var does
    _, _, _, Bx, Bz = QuaST._sem_stokes_elem(p, q)
    I = Int[]; J = Int[]; V = Float64[]
    for lf in su.leaves
        h = 2.0^(-lf[1])
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for b in 1:nq1, a in 1:nq1, j in 1:n, i in 1:n
            pi_ = pids[(b - 1) * nq1 + a]
            uj = uids[(j - 1) * n + i]
            push!(I, pi_); push!(J, uj)
            push!(V, 0.5 * h * Bx[(b - 1) * nq1 + a, (j - 1) * n + i])
            push!(I, pi_); push!(J, nu + uj)
            push!(V, 0.5 * h * Bz[(b - 1) * nq1 + a, (j - 1) * n + i])
        end
    end
    Bref = sparse(I, J, V, sp.N, 2 * nu)
    Bq = semquad_stokes_B(ss, qt)

    # same matrix up to round-off (different summation order, so not bit-identical)
    @test size(Bq) == size(Bref)
    @test maximum(abs, Bq - Bref) < 1.0e-12 * max(maximum(abs, Bref), 1.0)
    @test maximum(abs, Bq - Bref) / maximum(abs, Bref) < 1.0e-12

    # black box: a production Stokes solution satisfies the production
    # constraint B_ref u = 0, so it must also satisfy B_q u = 0
    theta0 = [0.1 * sinpi(x) * sinpi(z) for (x, z) in su.coords]
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0)
    leaves = Set(qt.leaves)
    eta_fun = (x, y) -> semquad_eta_at(su, theta0, x, y, pr, leaves)
    u, pp, _, _, _ = solve_semquad_stokes_var(ss, qt, eta_fun,
                                              (x, y) -> 0.0,
                                              (x, y) -> pr.Ra * 0.1 * sinpi(x) * sinpi(y),
                                              (x, y) -> 0.0, (x, y) -> 0.0,
                                              bc=:free, linear=:direct)
    @test maximum(abs, Bref * u) < 1.0e-9 * max(1.0, maximum(abs, u))
    @test maximum(abs, Bq * u) < 1.0e-9 * max(1.0, maximum(abs, u))

    # rho of one is the constant case; a stratified rho changes the matrix
    B1 = semquad_stokes_B(ss, qt; rho_fun=y -> 1.0)
    @test maximum(abs, B1 - Bq) < 1.0e-14
    Bs = semquad_stokes_B(ss, qt; rho_fun=y -> exp(0.7 * (1.0 - y)))
    @test maximum(abs, Bs - Bq) > 1.0e-3 * maximum(abs, Bq)
end
