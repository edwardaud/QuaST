# G149: the EV field, CBF and checkpoint — the recorded field is the field used.
using Test, QuaST, LinearAlgebra, HDF5

# G149: elementwise entropy-viscosity field, CBF consistency and checkpoint I/O.
#
# The EV field is elementwise (one value per leaf) and is needed by the
# consistent-boundary-flux residual, because the diffusion operator that
# produced theta^(n+1) was built from theta^n/theta^(n-1) and cannot be
# reconstructed from the post-step state.  This gate checks:
#   1. `ev_out` records exactly the per-leaf viscosity used by the assembly,
#      and passing it back through `ev_field` reproduces M, K, C and F bitwise;
#   2. EV = 0 leaves the operator bit-identical and records all zeros;
#   3. the CBF flux is unchanged (bitwise) by the physical EV field because EV
#      is zero on wall leaves, while an artificial nonzero-wall field does
#      change it (proving the field reaches the operator), and the value equals
#      a direct wall sum of the residual assembled with that field;
#   4. checkpoints round-trip the EV field, the box width `Lx`, and old
#      checkpoints without `nx0`/`ny0`/`Lx`/`ev` (multi-root inference).

@testset "G149 EV field, CBF and checkpoint" begin
    # A `Quadtree(L; cells_x=..., cells_y=...)` has maxlevel L but only the base
    # leaves; refine L times to obtain the uniform level-L grid (otherwise every
    # leaf is a wall leaf and the EV field is identically zero).
    function refined_tree(level::Int; cx::Int=1, cy::Int=1)
        qt = Quadtree(level; cells_x=cx, cells_y=cy)
        for _ in 1:level, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        return qt
    end

    # ---- 1. the recorded field is the field used, and replay is bitwise ----
    for (cx, cy) in ((5, 2), (1, 1))
        qt = refined_tree(3; cx=cx, cy=cy)
        sq = make_semquad(qt, 2)
        pr = Params(Ra=1.0e4, H=0.0)
        Lx = qt.Lx
        th = [0.10 * sin(pi * x / Lx) * sin(pi * y) for (x, y) in sq.coords]
        thp = [0.09 * sin(pi * x / Lx) * sin(pi * y) for (x, y) in sq.coords]
        ux = [0.10 * cos(pi * x / Lx) * sin(pi * y) for (x, y) in sq.coords]
        uz = [-0.10 * sin(pi * x / Lx) * cos(pi * y) for (x, y) in sq.coords]
        dt = 1.0e-3
        evg = QuaST.semquad_ev_globals(sq, qt, pr, th, thp, ux, uz, 1.0)
        evbuf = zeros(Float64, length(sq.leaves))
        M1, K1, C1, F1 = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                                 advection=:galerkin, theta=th,
                                                 ev=0.11, ev_cmax=0.052,
                                                 ev_timestep=5, evg=evg,
                                                 theta_prev=thp, dt=dt,
                                                 ev_out=evbuf)
        M2, K2, C2, F2 = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                                 advection=:galerkin, theta=th,
                                                 ev_field=evbuf)
        @test M1 == M2
        @test K1 == K2
        @test C1 == C2
        @test F1 == F2
        # every recorded entry is the value `semquad_ev_nu` would compute
        for (i, lf) in enumerate(sq.leaves)
            nu = QuaST.semquad_ev_nu(sq, pr, th, thp, ux, uz, dt, lf, evg;
                                          ev=0.11, ev_cmax=0.052,
                                          ev_alpha=2, ev_timestep=5)
            @test evbuf[i] == nu
        end
        # wall leaves are switched off inside semquad_ev_nu
        @test all(evbuf[i] == 0.0 for (i, lf) in enumerate(sq.leaves)
                  if lf[3] == 0 || lf[3] == sq.ny0 * (1 << lf[1]) - 1)

        # ---- 2. EV = 0 is bit-identical and records zeros ----
        z = ones(Float64, length(sq.leaves))   # deliberately dirty
        M0, K0, C0, F0 = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                                 advection=:galerkin, theta=th,
                                                 ev_out=z)
        Mref, Kref, Cref, Fref = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                                         advection=:galerkin)
        @test M0 == Mref && K0 == Kref && C0 == Cref && F0 == Fref
        @test all(iszero, z)
        @test_throws ArgumentError assemble_semquad_energy(
            sq, qt, pr, ux, uz; theta=th, ev=0.11,
            evg=evg, theta_prev=thp, dt=dt,
            ev_field=zeros(Float64, length(sq.leaves) + 1))

        # ---- 3. CBF and the EV field ----
        # `semquad_ev_nu` switches EV off on the top/bottom leaves.  A CBF wall
        # row only involves the wall element, so the *physical* EV field cannot
        # change the wall flux: the two values must be bitwise identical.  This
        # is the reason the "2b CBF is low because EV is missing" hypothesis is
        # false; see G149 notes / ROUTE_C_STATUS.md.
        nt0, nb0 = semquad_nu_cbf(sq, qt, pr, th, ux, uz; advection=:galerkin)
        nt1, nb1 = semquad_nu_cbf(sq, qt, pr, th, ux, uz; advection=:galerkin,
                                  ev_field=evbuf)
        @test isfinite(nt1) && isfinite(nb1)
        @test nt1 == nt0 && nb1 == nb0
        # An artificial field with nonzero wall values must change the flux;
        # this proves that `ev_field` really reaches the assembled operator.
        evart = fill(0.1, length(sq.leaves))
        nt2, nb2 = semquad_nu_cbf(sq, qt, pr, th, ux, uz; advection=:galerkin,
                                  ev_field=evart)
        @test abs(nt2 - nt0) > 1.0e-8 || abs(nb2 - nb0) > 1.0e-8
        # direct residual sum at the top wall with the artificial field
        _, Kf, Cf, Ff = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                                advection=:galerkin, theta=th,
                                                ev_field=evart)
        r = Kf * th .+ Cf * th .- Ff
        n = sq.p + 1
        seen = Set{Int}()
        s = 0.0
        for lf in sq.leaves
            lf[3] == sq.ny0 * (1 << lf[1]) - 1 || continue
            ids = sq.elem[lf]
            for i in 1:n
                g = ids[(n - 1) * n + i]
                g in seen && continue
                push!(seen, g); s += r[g]
            end
        end
        top_ref = -semquad_base_gradient(pr, 1.0)
        A = total_area(qt)
        @test nt2 ≈ top_ref - s / A rtol = 1.0e-12
    end

    # ---- 4a. checkpoint round-trip stores ev, Lx and base cells ----
    for (cx, cy, Lx) in ((5, 2, 2.5), (1, 1, 1.0), (1, 1, 2.5))
        qt = Quadtree(2; cells_x=cx, cells_y=cy, Lx=(cx == 1 && cy == 1 ? Lx : nothing))
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        n = ss.su.N
        th = zeros(Float64, n)
        evv = [0.01 * (i + 1) for i in 1:length(qt.leaves)]
        path = joinpath(tempdir(), "g149_ck_$(cx)x$(cy)_$(Lx).h5")
        save_semquad_state(path, qt, th, th, zeros(n), zeros(n),
                           zeros(ss.sp.N); t=0.5, dt=1.0e-4, dt_prev=1.0e-4,
                           step=7, p=2, ev=evv,
                           config=Dict("ev" => 0.11))
        ld = load_semquad_state(path)
        @test ld.qt.nx0 == cx && ld.qt.ny0 == cy
        @test ld.qt.Lx ≈ qt.Lx
        @test Set(ld.qt.leaves) == Set(qt.leaves)
        @test ld.ev == evv
        @test ld.step == 7
        @test ld.config["ev"] == "0.11"
        rm(path; force=true)
    end

    # ---- 4b. an old checkpoint (no nx0/ny0/Lx/ev) is recovered ----
    qt = refined_tree(3; cx=5, cy=2)
    sq = make_semquad(qt, 2)
    ss = make_semstokes(qt, 2)
    leaves = sort(collect(qt.leaves))
    L = Matrix{Int32}(undef, 3, length(leaves))
    for (k, lf) in enumerate(leaves)
        L[1, k] = lf[1]; L[2, k] = lf[2]; L[3, k] = lf[3]
    end
    path = joinpath(tempdir(), "g149_old_ck.h5")
    h5open(path, "w") do f
        f["leaves"] = L
        f["theta"] = zeros(Float64, sq.N)
        f["theta_prev"] = zeros(Float64, sq.N)
        f["ux"] = zeros(Float64, sq.N)
        f["uz"] = zeros(Float64, sq.N)
        f["pv"] = zeros(Float64, ss.sp.N)
        HDF5.attrs(f)["t"] = 0.25
        HDF5.attrs(f)["dt"] = 1.0e-4
        HDF5.attrs(f)["dt_prev"] = 1.0e-4
        HDF5.attrs(f)["step"] = 11
        HDF5.attrs(f)["p"] = 2
        HDF5.attrs(f)["maxlevel"] = qt.maxlevel
    end
    ld = load_semquad_state(path)
    @test ld.qt.nx0 == 5 && ld.qt.ny0 == 2
    @test ld.qt.Lx ≈ 2.5
    @test Set(ld.qt.leaves) == Set(qt.leaves)
    @test ld.ev === nothing
    rm(path; force=true)

    # ---- 4c. a short EV run records a non-trivial field ----
    qt = refined_tree(2; cx=5, cy=2)
    ss = make_semstokes(qt, 2)
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=log(10.0),
                eta_c=log(2.0), Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th0 = [0.05 * sin(pi * x / qt.Lx) * sin(pi * y) for (x, y) in ss.su.coords]
    for (i, (x, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th0[i] = 0.0)
    end
    path = joinpath(tempdir(), "g149_run_ck.h5")
    st = run_semquad_simulation(qt, pr, th0; p=2, tstop=2.0e-3, dt0=1.0e-4,
                                maxsteps=4, verbose=false, ev=0.11,
                                ev_cmax=0.052, nu_cbf=true,
                                checkpoint_path=path, checkpoint_every=1)
    @test length(st.history[end]) == 20
    @test all(isfinite, st.history[end][19:20])
    ld = load_semquad_state(path)
    @test ld.ev !== nothing
    @test length(ld.ev) == length(ld.qt.leaves)
    @test maximum(abs, ld.ev) > 0.0
    rm(path; force=true)
end
