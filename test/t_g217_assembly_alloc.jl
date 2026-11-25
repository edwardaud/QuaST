# G217 / assembly-allocation audit: the SEM hot-path kernels (indicator,
# nodal transfer, Lagrange basis, energy assembly) are bitwise unchanged by the
# scratch/out-parameter refactor, and the refactored per-call paths allocate
# their budget or less.  The allocating implementations are carried verbatim
# next to the workspace form as the "before" side and the bitwise oracle, so
# every 0-B assertion has a positive control next to it (the G215 pattern, not a
# re-recorded number).  The y-face fine-side index of the allocating copy was
# corrected on 2026-11-25 together with the production body: this gate tests the
# workspace refactor, not the historical arithmetic.  The one-line pre-fix index
# (the defect the box body had until 2026-11-25) is pinned independently by
# test/t_g278_indicator_oracle.jl.
#
# Measured context: `--track-allocation=user`
# over 4 AMR cycles of out_julia/s340/ab_none/resume.h5 (10 000 leaves, p=2)
# went 282.47 MB -> 44.22 MB total, gll.jl 166.02 MB -> 0.004 MB and
# semquad.jl 97.60 MB -> 25.37 MB, with the leaf/theta FNVs unchanged.
using Test, QuaST, LinearAlgebra, SparseArrays, Random, Printf

const MS = QuaST

# ---------------------------------------------------------------------------
# Allocating reference implementations, verbatim (HEAD 8501b53), as the positive
# control and the bitwise oracle.  The y-face fine-side index of the sub-face
# kernel is the production one (corrected 2026-11-25); the pre-fix line is the
# defect, not this gate's subject.
# ---------------------------------------------------------------------------
function g217_old_subface_derivs(sq::SemQuad, L, R, dir, u)
    p = sq.p
    n = p + 1
    hL = MS.leaf_hy(sq, L[1]); hR = MS.leaf_hy(sq, R[1])
    hxL = MS.leaf_hx(sq, L[1]); hxR = MS.leaf_hx(sq, R[1])
    xL = L[2] * hxL; yL = L[3] * hL
    xR = R[2] * hxR; yR = R[3] * hR
    rule = gll_rule(p)
    BL = zeros(n, n)
    vertical = dir === :xplus || dir === :xminus
    if vertical
        wq = 0.5 * hR .* sq.w
        iFL = dir === :xplus ? n : 1
        iFR = dir === :xplus ? 1 : n
        pos = yR .+ 0.5 .* (sq.xi .+ 1) .* hR
        xiL = 2.0 .* (pos .- yL) ./ hL .- 1.0
        for q in 1:n
            Lb, _ = lagrange_eval(rule, xiL[q])
            BL[q, :] .= Lb
        end
        cL = (2.0 / hxL) .* sq.D[iFL, :]
        cR = (2.0 / hxR) .* sq.D[iFR, :]
        gidL = (i, a) -> sq.elem[L][(a - 1) * n + i]
        gidR = (i, q) -> sq.elem[R][(q - 1) * n + i]
    else
        wq = 0.5 * hxR .* sq.w
        iFL = dir === :yplus ? n : 1
        iFR = dir === :yplus ? 1 : n
        pos = xR .+ 0.5 .* (sq.xi .+ 1) .* hxR
        xiL = 2.0 .* (pos .- xL) ./ hxL .- 1.0
        for q in 1:n
            Lb, _ = lagrange_eval(rule, xiL[q])
            BL[q, :] .= Lb
        end
        cL = (2.0 / hL) .* sq.D[iFL, :]
        cR = (2.0 / hR) .* sq.D[iFR, :]
        gidL = (i, a) -> sq.elem[L][(i - 1) * n + a]
        # y-face fine side: the normal derivative at fixed x-node q is a
        # contraction over the *y* node index of column q (the pre-fix line
        # contracted over the x index of row q; G278 pins that defect)
        gidR = (i, q) -> sq.elem[R][(i - 1) * n + q]
    end
    dL = zeros(Float64, n)
    dR = zeros(Float64, n)
    for q in 1:n
        for a in 1:n
            da = 0.0
            for i in 1:n
                da += cL[i] * u[gidL(i, a)]
            end
            dL[q] += BL[q, a] * da
        end
        for i in 1:n
            dR[q] += cR[i] * u[gidR(i, q)]
        end
    end
    return wq, dL, dR
end

function g217_old_transfer(old::SemQuad, uold::Vector{Float64}, new::SemQuad)
    oldleaves = Set(old.leaves)
    unew = zeros(Float64, new.N)
    n = new.p + 1
    for lf in new.leaves
        hx = MS.leaf_hx(new, lf[1]); hy = MS.leaf_hy(new, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = new.elem[lf]
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (new.xi[i] + 1.0) * hx
            y = y0 + 0.5 * (new.xi[j] + 1.0) * hy
            unew[ids[(j - 1) * n + i]] =
                MS._semquad_eval_at(old, uold, x, y, oldleaves)
        end
    end
    return unew
end

# ---------------------------------------------------------------------------
# small graded tree + smooth field
# ---------------------------------------------------------------------------
function g217_tree(rng, level::Int, nref::Int)
    qt = Quadtree(level)
    for _ in 1:nref, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function g217_field(sq)
    return [0.3 * sinpi(x) * sinpi(y) + 0.05 * sinpi(3x) * sinpi(2y)
            for (x, y) in sq.coords]
end

@testset "G217 assembly allocation" begin
    # ---- 1. out-parameter Lagrange forms are bitwise the wrappers ----------
    @testset "Lagrange out-parameter forms" begin
        rng = MersenneTwister(217)
        nchecked = 0
        for p in (1, 2, 3, 4, 7, 12)
            rule = gll_rule(p)
            n = p + 1
            xs = Float64[-1.0, 1.0, 0.123456789, 1.0 - 1e-10]
            append!(xs, rule.nodes)
            append!(xs, 2 .* rand(rng, 25) .- 1)
            L0 = zeros(n); dL0 = zeros(n); d20 = zeros(n)
            for x in xs
                L, dL = lagrange_eval(rule, x)
                fill!(L0, NaN); fill!(dL0, NaN)
                lagrange_eval!(L0, dL0, rule, x)
                @test L0 == L && dL0 == dL
                Lv = MS.lagrange_values(rule, x)
                fill!(L0, NaN)
                lagrange_values!(L0, rule, x)
                @test L0 == Lv
                @test Lv == L
                L2, dL2, d22 = lagrange_d2(rule, x)
                fill!(L0, NaN); fill!(dL0, NaN); fill!(d20, NaN)
                lagrange_d2!(L0, dL0, d20, rule, x)
                @test L0 == L2 && dL0 == dL2 && d20 == d22
                nchecked += 1
            end
        end
        @test nchecked > 200
        rule = gll_rule(4)
        L = zeros(4); dL = zeros(4)
        lagrange_eval!(L, dL, rule, 0.3)
        a = @allocated lagrange_eval!(L, dL, rule, 0.3)
        @printf("  lagrange_eval! bytes = %d\n", a)
        @test a == 0
    end

    # ---- 2. indicator workspace: bitwise + 0 B ----------------------------
    @testset "gradient-jump workspace" begin
        qt = g217_tree(MersenneTwister(1), 6, 4)     # 256 leaves, graded
        sq = make_semquad(qt, 2)
        u = g217_field(sq)
        pairs = collect(face_pairs(qt))
        @test length(pairs) > 100
        w = MS.SemQuadJumpWork(sq.p + 1)
        nbad = 0
        for pf in pairs
            wo, dLo, dRo = g217_old_subface_derivs(sq, pf.leafL, pf.leafR,
                                                   pf.dir, u)
            wn, dLn, dRn = MS._semquad_subface_derivs!(w, sq, pf.leafL,
                                                       pf.leafR, pf.dir, u)
            nbad += (wo == wn && dLo == dLn && dRo == dRn) ? 0 : 1
        end
        @test nbad == 0
        # full indicator: this gate compares the new workspace path against the
        # historical one through the same per-pair oracle, value by value.
        eta_new = semquad_gradient_jump(sq, qt, u)
        eta_old = Dict{NTuple{3,Int},Float64}(lf => 0.0 for lf in sq.leaves)
        for pf in pairs
            L, R, dir = pf.leafL, pf.leafR, pf.dir
            wq, dL, dR = g217_old_subface_derivs(sq, L, R, dir, u)
            J = 0.0
            for q in eachindex(wq)
                J += wq[q] * (dL[q] - dR[q])^2
            end
            nx = dir === :xplus || dir === :xminus
            hL = nx ? MS.leaf_hx(sq, L[1]) : MS.leaf_hy(sq, L[1])
            hR = nx ? MS.leaf_hx(sq, R[1]) : MS.leaf_hy(sq, R[1])
            eta_old[L] += 0.5 * hL * J
            eta_old[R] += 0.5 * hR * J
        end
        # boundary terms (top/bottom walls only; the sides are insulating)
        n = sq.p + 1
        for lf in sq.leaves
            h = MS.leaf_hy(sq, lf[1])
            hx = MS.leaf_hx(sq, lf[1])
            ids = sq.elem[lf]
            for dir in (:xminus, :xplus, :yminus, :yplus)
                MS.has_no_face_neighbor(qt, lf, dir) || continue
                (dir === :xminus || dir === :xplus) && continue
                jF = dir === :yminus ? 1 : n
                cB = (2.0 / h) .* sq.D[jF, :]
                for i in 1:n
                    d = 0.0
                    for j in 1:n
                        d += cB[j] * u[ids[(j - 1) * n + i]]
                    end
                    wf = 0.5 * hx * sq.w[i]
                    eta_old[lf] += 0.5 * h * wf * d^2
                end
            end
        end
        @test all(eta_old[lf] == eta_new[lf] for lf in sq.leaves)
        # budget: the per-pair kernel is 0 B, the allocating form is not
        MS._semquad_subface_derivs!(w, sq, pairs[1].leafL, pairs[1].leafR,
                                    pairs[1].dir, u)
        a_new = @allocated MS._semquad_subface_derivs!(w, sq, pairs[1].leafL,
                                                       pairs[1].leafR,
                                                       pairs[1].dir, u)
        a_old = @allocated g217_old_subface_derivs(sq, pairs[1].leafL,
                                                   pairs[1].leafR,
                                                   pairs[1].dir, u)
        @printf("  subface bytes: new=%d old=%d\n", a_new, a_old)
        @test a_new == 0
        @test a_old > 0
        @test length(face_pairs(qt)) == length(pairs)
        @test eltype(face_pairs(qt)) === MS.FacePair
    end

    # ---- 3. nodal transfer: bitwise + one-shot budget ---------------------
    @testset "nodal transfer" begin
        qt = g217_tree(MersenneTwister(2), 6, 3)     # 64 leaves
        sq = make_semquad(qt, 2)
        u = g217_field(sq)
        old = g217_old_transfer(sq, u, sq)
        new = transfer_semquad(sq, u, sq; mode=:nodal, zero_walls=true)
        @test old == new
        # refine then coarsen elsewhere: a genuinely non-uniform new mesh
        qt2 = deepcopy(qt)
        for lf in sort(collect(qt2.leaves))[1:4]
            lf in qt2.leaves && refine!(qt2, lf)
        end
        balance!(qt2)
        sq2 = make_semquad(qt2, 2)
        old2 = g217_old_transfer(sq, u, sq2)
        new2 = transfer_semquad(sq, u, sq2; mode=:nodal)
        @test old2 == new2
        g217_old_transfer(sq, u, sq2)
        transfer_semquad(sq, u, sq2; mode=:nodal)
        a_old = @allocated g217_old_transfer(sq, u, sq2)
        a_new = @allocated transfer_semquad(sq, u, sq2; mode=:nodal)
        @printf("  transfer bytes: new=%.2f MB old=%.2f MB\n",
                a_new / 2^20, a_old / 2^20)
        # the deduplicated evaluation is the bulk of the win: on this mesh the
        # output vector alone is N*8 bytes, and the old path paid 2*n vectors
        # per (leaf,node) evaluation
        @test a_new < 0.25 * a_old
        @test a_old > 0
    end

    # ---- 4. energy assembly: pooled buffers + hoisted EV data -------------
    @testset "energy assembly pool and EV hoist" begin
        qt = g217_tree(MersenneTwister(3), 6, 5)     # 1 024 leaves
        sq = make_semquad(qt, 2)
        n = sq.p + 1
        nv = n * n
        th = g217_field(sq)
        for g in 1:sq.N
            y = sq.coords[g][2]
            (y <= 1e-14 || y >= 1 - 1e-14) && (th[g] = 0.0)
        end
        ux = [0.3 * sinpi(x) * cospi(y) for (x, y) in sq.coords]
        uz = [-0.2 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
        pr = Params(Ra=1.0e5, H=10.0, eta_law=:expT, deta_T=1.0,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free,
                    latent_L=0.11, latent_mode=:global)
        ph = [PhaseTransition(0.87, 0.025, 0.032, 0.5, 180.0, 1.818)]
        evg = MS.semquad_ev_globals(sq, qt, pr, th, th, ux, uz, 0.0)
        ev_out = zeros(Float64, length(sq.leaves))
        kw = (advection=:skew, theta=th, phases=ph, ev=0.11, ev_cmax=0.052,
              ev_alpha=2, ev_timestep=20000, evg=evg, theta_prev=th,
              dt=1.0e-2, ev_out=ev_out)
        empty!(MS._SEMQUAD_ENERGY_BUF_CACHE)      # force a cold first call
        MS._SEMQUAD_ENERGY_BUF_USED[] = 0
        a1 = @allocated assemble_semquad_energy(sq, qt, pr, ux, uz; kw...)
        M1, K1, C1, F1 = assemble_semquad_energy(sq, qt, pr, ux, uz; kw...)
        M2, K2, C2, F2 = assemble_semquad_energy(sq, qt, pr, ux, uz; kw...)
        @test M1 == M2 && K1 == K2 && C1 == C2 && F1 == F2
        a2 = @allocated assemble_semquad_energy(sq, qt, pr, ux, uz; kw...)
        @printf("  energy assembly bytes: first=%.2f MB second=%.2f MB\n",
                a1 / 2^20, a2 / 2^20)
        # the first call builds the pooled COO capacity (9 arrays of
        # nv^2*nleaf entries); the second must reuse it.  Both keep the sparse
        # operators and F, so the second call is bounded well below the first.
        @test a1 > 0
        bufbytes = 9 * 8 * nv * nv * length(sq.leaves)
        @printf("  first-call buffer capacity = %.2f MB\n", bufbytes / 2^20)
        @test a1 - a2 > 0.5 * bufbytes
        # pooled acquisition is 0 B after the first, the raw constructor is not
        empty!(MS._SEMQUAD_ENERGY_BUF_CACHE)
        MS._SEMQUAD_ENERGY_BUF_USED[] = 0
        MS._semquad_energy_buf(length(sq.leaves), nv, sq.N)
        MS._SEMQUAD_ENERGY_BUF_USED[] = 0
        a_pool = @allocated MS._semquad_energy_buf(length(sq.leaves), nv, sq.N)
        a_raw = @allocated MS._semquad_energy_buf_alloc(length(sq.leaves), nv,
                                                        sq.N)
        @printf("  energy buf bytes: pooled=%d raw=%d\n", a_pool, a_raw)
        @test a_pool == 0
        @test a_raw > 0

        # EV sampling: the hoisted quadrature data is bitwise the cold path
        lf = first(sort(collect(sq.leaves)))
        fd_ev = make_fedata(sq.p, sq.p)
        d2tab = [lagrange_d2(MS.gll_rule_cached(sq.p), fd_ev.gl_x[k])[3]
                 for k in 1:fd_ev.nq]
        cold = MS.semquad_ev_nu(sq, pr, th, th, ux, uz, 1.0e-2, lf, evg;
                                ev=0.11, ev_cmax=0.052, ev_alpha=2,
                                ev_timestep=20000)
        warm = MS.semquad_ev_nu(sq, pr, th, th, ux, uz, 1.0e-2, lf, evg;
                                ev=0.11, ev_cmax=0.052, ev_alpha=2,
                                ev_timestep=20000, fd=fd_ev, d2tab=d2tab)
        @test cold == warm
        # ... and the hoisted call allocates no quadrature data per leaf
        MS.semquad_ev_nu(sq, pr, th, th, ux, uz, 1.0e-2, lf, evg;
                         ev=0.11, ev_cmax=0.052, ev_alpha=2,
                         ev_timestep=20000, fd=fd_ev, d2tab=d2tab)
        a_warm = @allocated MS.semquad_ev_nu(sq, pr, th, th, ux, uz, 1.0e-2,
                                             lf, evg; ev=0.11, ev_cmax=0.052,
                                             ev_alpha=2, ev_timestep=20000,
                                             fd=fd_ev, d2tab=d2tab)
        a_cold = @allocated MS.semquad_ev_nu(sq, pr, th, th, ux, uz, 1.0e-2,
                                             lf, evg; ev=0.11, ev_cmax=0.052,
                                             ev_alpha=2, ev_timestep=20000)
        @printf("  ev_nu bytes: hoisted=%d cold=%d\n", a_warm, a_cold)
        @test a_warm == 0
        @test a_cold > 0
    end
end
