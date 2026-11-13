using Test, QuaST, LinearAlgebra, SparseArrays

# G181 / S4 stage 1: leaf-level parallel (multi-threaded) Stokes assembly.
#
# The assembler walks `su.leaves` and accumulates COO triplets + a load vector.
# `assemble_threads=N>1` cuts the leaf list into N fixed contiguous blocks,
# assembles each block into its own buffers and concatenates the block buffers
# in block-index order.  The leaf->block map and the merge order are fixed, so:
#   * a threaded assembly is deterministic run to run (checked bitwise here);
#   * the COO order is in fact the same as the serial one (block order = leaf
#     order), hence K/Kgeo/Kp/B come out bitwise equal on this machine -- the
#     gate only *requires* <=1e-15 relative, and reports the bitwise result;
#   * the load vector is a sum of N block sums instead of one running sum, so
#     `fv` differs at round-off and the resulting solution differs at a few ulp
#     of the saddle solve, not at solver tolerance.
# The default `assemble_threads=1` is the historical serial path bit for bit
# (G150 keeps pinning the direct-solver round-off).
#
# The block+merge machinery is exercised directly (`_semquad_assemble_blocks`)
# so the gate is meaningful even in a single-threaded CI process; the
# solver-level keyword path is additionally checked when Threads.nthreads()>1.

const _G181_Z2 = (x, y) -> 0.0
const _G181_ONE = (x, y) -> 1.0

function _g181_uniform(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

_relmax(a::AbstractArray, b::AbstractArray) =
    maximum(abs.(a .- b)) / max(maximum(abs.(a)), maximum(abs.(b)), 1.0e-300)

@testset "G181 threaded leaf assembly" begin
    qt = _g181_uniform(4)                       # 256 leaves, p=4, 8450 dofs
    ss = make_semstokes(qt, 4)
    @test length(qt.leaves) == 256
    fz = (x, y) -> sin(pi * x / qt.Lx) * sin(pi * y)
    asmkw = (bc=:free, linear=:direct, assemble_only=true)

    sys = solve_semquad_stokes_var(ss, qt, _G181_ONE, _G181_Z2, fz,
                                   _G181_Z2, _G181_Z2; asmkw...)
    sys2 = solve_semquad_stokes_var(ss, qt, _G181_ONE, _G181_Z2, fz,
                                    _G181_Z2, _G181_Z2; asmkw...)
    # default (assemble_threads=1) is deterministic and bitwise repeatable
    @test sys.K.nzval == sys2.K.nzval
    @test sys.B.nzval == sys2.B.nzval
    @test sys.fv == sys2.fv

    # ---- option validation -------------------------------------------------
    @test_throws ArgumentError solve_semquad_stokes_var(
        ss, qt, _G181_ONE, _G181_Z2, fz, _G181_Z2, _G181_Z2;
        asmkw..., assemble_threads=0)

    # ---- requested > available: fall back / clamp, warning exactly once ----
    nproc = Threads.nthreads()
    if nproc == 1
        @test_logs (:warn, r"falling back to serial") solve_semquad_stokes_var(
            ss, qt, _G181_ONE, _G181_Z2, fz, _G181_Z2, _G181_Z2;
            asmkw..., assemble_threads=8)
    else
        @test_logs (:warn, r"using .* blocks") solve_semquad_stokes_var(
            ss, qt, _G181_ONE, _G181_Z2, fz, _G181_Z2, _G181_Z2;
            asmkw..., assemble_threads=nproc + 1)
    end

    # ---- the block+merge assembly equals the serial solver's matrices ------
    cfg = QuaST.SemQuadAsmCfg(2.0, false, false, nothing, nothing, 0.0,
                                  nothing, nothing, nothing)
    fd = make_fedata(ss.p, ss.q)
    pl = Set(ss.sp.leaves)
    Nu2 = 2 * ss.su.N
    function blocksys(nt)
        a = QuaST._semquad_assemble_blocks(nt, cfg, ss.su, ss.sp,
                                               _G181_ONE, _G181_Z2, fz, fd, pl)
        return (sparse(a.IK, a.JK, a.VK, Nu2, Nu2),
                sparse(a.IB, a.JB, a.VB, ss.sp.N, Nu2), a.fv)
    end
    K1, B1, fv1 = blocksys(1)
    @test _relmax(K1, sys.K) <= 1.0e-15
    @test _relmax(B1, sys.B) <= 1.0e-15
    @test _relmax(fv1, sys.fv) <= 1.0e-15
    for nt in (2, 4, 8)
        Kt, Bt, fvt = blocksys(nt)
        @test _relmax(Kt, K1) <= 1.0e-15
        @test _relmax(Bt, B1) <= 1.0e-15
        @test _relmax(fvt, fv1) <= 1.0e-15
        @info "G181 block assembly" blocks = nt K_bitwise = (Kt.nzval == K1.nzval) K_rel = _relmax(Kt, K1) fv_rel = _relmax(fvt, fv1)
    end
    # deterministic run to run at the same block count (bitwise)
    Ka, Ba, fva = blocksys(4)
    Kb, Bb, fvb = blocksys(4)
    @test Ka.nzval == Kb.nzval
    @test Ba.nzval == Bb.nzval
    @test fva == fvb

    # ---- geo / dp blocks are merged too (small tree) -----------------------
    qtg = _g181_uniform(2)
    ssg = make_semstokes(qtg, 2)
    ug = zeros(Float64, 2 * ssg.su.N)
    for (g, (x, y)) in enumerate(ssg.su.coords)
        ug[g] = y
        ug[ssg.su.N + g] = 0.0
    end
    pbase = [1.5 + 0.3 * sin(pi * x) * sin(pi * y) for (x, y) in ssg.sp.coords]
    geoc = (x, y, ed) -> 1.0 / (1.0 + ed)
    dpc = (x, y, ed, p) -> 1.0e-3 * cos(p)
    z2 = _G181_Z2
    sysg = solve_semquad_stokes_var(ssg, qtg, _G181_ONE, z2, z2, z2, z2;
                                    bc=:free, assemble_only=true, geo_u=ug,
                                    geo_coef=geoc, dp_u=ug, dp_pfield=pbase,
                                    dp_coef=dpc)
    @test norm(sysg.Kgeo) > 0.0
    @test norm(sysg.Kp) > 0.0
    cfgg = QuaST.SemQuadAsmCfg(2.0, false, false, ug, geoc, 0.0,
                                   ug, pbase, dpc)
    fdg = make_fedata(ssg.p, ssg.q)
    plg = Set(ssg.sp.leaves)
    function blocksysg(nt)
        a = QuaST._semquad_assemble_blocks(nt, cfgg, ssg.su, ssg.sp,
                                               _G181_ONE, z2, z2, fdg, plg)
        return (sparse(a.IG, a.JG, a.VG, 2 * ssg.su.N, 2 * ssg.su.N),
                sparse(a.IP, a.JP, a.VP, 2 * ssg.su.N, ssg.sp.N))
    end
    Kg1, Kp1 = blocksysg(1)
    Kg4, Kp4 = blocksysg(4)
    @test _relmax(Kg4, Kg1) <= 1.0e-15
    @test _relmax(Kp4, Kp1) <= 1.0e-15
    @test _relmax(Kg1, sysg.Kgeo) <= 1.0e-15
    @test _relmax(Kp1, sysg.Kp) <= 1.0e-15
    @info "G181 geo/dp blocks" geo_rel = _relmax(Kg4, Kg1) dp_rel = _relmax(Kp4, Kp1)

    # ---- solver-level keyword path (needs a real thread pool) --------------
    if Threads.nthreads() > 1
        nt = min(4, Threads.nthreads())
        uD, _, _, _, _ = solve_semquad_stokes_var(ss, qt, _G181_ONE, _G181_Z2,
                                                  fz, _G181_Z2, _G181_Z2;
                                                  bc=:free, linear=:direct)
        resD = LAST_SEMSTOKES_RESIDUAL[]
        uT, _, _, _, _ = solve_semquad_stokes_var(ss, qt, _G181_ONE, _G181_Z2,
                                                  fz, _G181_Z2, _G181_Z2;
                                                  bc=:free, linear=:direct,
                                                  assemble_threads=nt)
        @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-11
        du = norm(uT - uD) / norm(uD)
        @test du < 1.0e-10
        # matrix-free MG path: the sparse K is not assembled at all
        uM, _, _, _, _ = solve_semquad_stokes_var(ss, qt, _G181_ONE, _G181_Z2,
                                                  fz, _G181_Z2, _G181_Z2;
                                                  bc=:free, linear=:mg,
                                                  assemble_threads=nt)
        @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
        uM1, _, _, _, _ = solve_semquad_stokes_var(ss, qt, _G181_ONE, _G181_Z2,
                                                   fz, _G181_Z2, _G181_Z2;
                                                   bc=:free, linear=:mg)
        @test norm(uM - uM1) / norm(uM1) < 1.0e-9
    end

    # ---- energy assembly: the same block+merge machinery -------------------
    # `assemble_semquad_energy` uses the same contiguous-block split; its M, K
    # and C come from a leaf-ordered COO list (bitwise equal), while F is a sum
    # of block sums (round-off only).
    qte = _g181_uniform(3; cx=5, cy=2)          # 640 leaves
    sqe = make_semquad(qte, 3)
    pre = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                 Twall_top=0.0, Twall_bot=1.0, bc=:free, Di=0.3,
                 heating_adiabatic=true, heating_dissipation=true)
    the = [0.05 * cos(pi * x / qte.Lx) * sin(pi * y) for (x, y) in sqe.coords]
    for (i, (x, y)) in enumerate(sqe.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (the[i] = 0.0)
    end
    uxe = [sin(pi * x / qte.Lx) * cos(pi * y) for (x, y) in sqe.coords]
    uze = [-cos(pi * x / qte.Lx) * sin(pi * y) for (x, y) in sqe.coords]
    etae = (x, y) -> exp(-log(100.0) * (1 - y))
    evge = QuaST.semquad_ev_globals(sqe, qte, pre, the, the, uxe, uze, 0.0)
    evoe = zeros(Float64, length(sqe.leaves))
    ekeyw = (theta=the, ev=0.5, evg=evge, ev_out=copy(evoe),
             heating_adiabatic=true, heating_dissipation=true, eta_fun=etae)
    Me1, Ke1, Ce1, Fe1 = assemble_semquad_energy(sqe, qte, pre, uxe, uze; ekeyw...)
    Me2, Ke2, Ce2, Fe2 = assemble_semquad_energy(sqe, qte, pre, uxe, uze; ekeyw...)
    @test Me1.nzval == Me2.nzval
    @test Ke1.nzval == Ke2.nzval
    @test Fe1 == Fe2
    @test_throws ArgumentError assemble_semquad_energy(
        sqe, qte, pre, uxe, uze; ekeyw..., assemble_threads=0)
    # `theta_lat` (G194) is the temperature of the latent-heat coefficients; the
    # default path leaves it `nothing` (= the same `theta` the rest uses).
    cfg_e = QuaST.SemQuadEnergyCfg(pre, uxe, uze, the, nothing, false,
                                       false, 0.5,
                                       0.052, 2, 0, evge, nothing, 0.0, nothing,
                                       copy(evoe), true, true, etae, nothing,
                                       PhaseTransition[])
    fde = make_fedata(sqe.p, sqe.p - 1)
    sete = Set(sqe.leaves)
    function energyblocks(nt)
        a = QuaST._semquad_assemble_energy_blocks(nt, cfg_e, sqe, fde, sete)
        C = sparse(a.IC, a.JC, a.VC, sqe.N, sqe.N)
        return (sparse(a.IM, a.JM, a.VM, sqe.N, sqe.N),
                sparse(a.IK, a.JK, a.VK, sqe.N, sqe.N), 0.5 * (C - C'), a.F)
    end
    for nt in (2, 4, 8)
        Mb, Kb, Cb, Fb = energyblocks(nt)
        @test _relmax(Mb, Me1) <= 1.0e-15
        @test _relmax(Kb, Ke1) <= 1.0e-15
        @test _relmax(Cb, Ce1) <= 1.0e-15
        @test _relmax(Fb, Fe1) <= 1.0e-15
        @info "G181 energy blocks" blocks = nt M_bitwise = (Mb.nzval == Me1.nzval) F_rel = _relmax(Fb, Fe1)
    end
    Ma, Ka, Ca, Fa = energyblocks(4)
    Mb, Kb, Cb, Fb = energyblocks(4)
    @test Ma.nzval == Mb.nzval
    @test Fa == Fb
    if Threads.nthreads() > 1
        Mt, Kt, Ct, Ft = assemble_semquad_energy(sqe, qte, pre, uxe, uze;
                                                 ekeyw...,
                                                 assemble_threads=min(4, Threads.nthreads()))
        @test _relmax(Mt, Me1) <= 1.0e-15
        @test _relmax(Kt, Ke1) <= 1.0e-15
        @test _relmax(Ct, Ce1) <= 1.0e-15
        @test _relmax(Ft, Fe1) <= 1.0e-15
    end
end
