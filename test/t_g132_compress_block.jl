# G132: the compressibility block Pcm — vertical pressure mass, no horizontal component.
using Test, QuaST
using SparseArrays

@testset "G132 compressibility block Pcm" begin
    qt = Quadtree(5)
    for _ in 1:2, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    p, q = 3, 2
    ss = make_semstokes(qt, p)
    su, sp = ss.su, ss.sp
    n = p + 1
    nq1 = q + 1
    nu = su.N

    # (1) rho = 1, Di = gamma = 1: the block is the pressure mass matrix in the
    # vertical rows and zero in the horizontal rows.  It is a density
    # perturbation, i.e. a force along gravity, so there is no x component
    # (applying it to both components was the King 2010 bug).
    P = semquad_stokes_Pcm(ss, qt)
    @test size(P) == (2 * nu, sp.N)
    I = Int[]; J = Int[]; V = Float64[]
    fd = make_fedata(p, q)
    for lf in su.leaves
        h = 2.0^(-lf[1])
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * h * h * fd.gl_w[qx] * fd.gl_w[qy]
            for b in 1:nq1, a in 1:nq1
                pq = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                pq == 0.0 && continue
                piq = pids[(b - 1) * nq1 + a]
                for j in 1:n, i in 1:n
                    uj = uids[(j - 1) * n + i]
                    ph = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                    ph == 0.0 && continue
                    push!(I, nu + uj); push!(J, piq); push!(V, w * ph * pq)
                end
            end
        end
    end
    P_ref = sparse(I, J, V, 2 * nu, sp.N)
    @test maximum(abs, P - P_ref) < 1.0e-14 * max(1.0, maximum(abs, P_ref))
    @test maximum(abs, P[1:nu, :]) == 0.0

    # (1b) the vertical block equals the (unweighted) pressure mass matrix; use
    # the independent assembly of the pressure mass to check it.
    Mfull = assemble_semquad_pressure_mass(sp)
    # P's z block is np x np only after mapping velocity -> pressure columns; with
    # rho = 1 and identical Q_p/Q_p bases it is the mass matrix sampled at the
    # pressure test functions of the *velocity* rows.  Compare the column sums
    # instead of the full matrix: sum_k P[i,k] = int phi_i * 1 dV for each
    # pressure basis function, which the pressure mass matrix also satisfies.
    @test maximum(abs, vec(sum(Matrix(P[nu+1:end, :]), dims=1)) -
                       vec(sum(Matrix(Mfull), dims=1))) <
          1.0e-12 * max(1.0, maximum(abs, Mfull))

    # (2) the legacy :both form is the same column replicated to x and z
    Pboth = semquad_stokes_Pcm(ss, qt; components=:both)
    @test maximum(abs, Pboth[1:nu, :] - Pboth[(nu + 1):(2nu), :]) == 0.0
    @test maximum(abs, Pboth[(nu + 1):(2nu), :] - P[(nu + 1):(2nu), :]) == 0.0
    @test maximum(abs, P - Pboth) > 0.0

    # (3) scaling: Di and gamma multiply the block, rho weights it pointwise
    P2 = semquad_stokes_Pcm(ss, qt; Di=3.0, gamma=0.5)
    @test maximum(abs, P2 - 1.5 * P) < 1.0e-14 * max(1.0, maximum(abs, 1.5 * P))
    P3 = semquad_stokes_Pcm(ss, qt; rho_fun=y -> exp(1.0 * (1.0 - y)))
    @test maximum(abs, P3 - P) > 1.0e-3 * maximum(abs, P)

    # (4) the column sums of B and Pcm differ (divergence vs value), so they are
    # genuinely different blocks
    B = semquad_stokes_B(ss, qt)          # np x 2nu
    Bt = Matrix(B')                      # 2nu x np, same shape as Pcm
    @test maximum(abs, Bt - Matrix(P)) > 1.0e-3 * max(1.0, maximum(abs, Bt))
end
