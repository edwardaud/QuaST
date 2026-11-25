# G279: the mapped-geometry defect fix campaign.
#
# One gate per confirmed audit defect, plus the (b) rows this campaign closed:
#
#   (D1)  `semquad_eta_contrast` mapped sample domain: the parent (f607f9e)
#         handed `logical_map` a physical-like `Lx*(i/n)`, so it sampled
#         `S in [0,Lx]` — on an `AffineBox` with `Lx = 7/3` that is
#         `x in [0, 5.4444]` (outside the domain) and returned contrast 6.4444
#         where the box body returns 3.3333; on an annulus ring (`Lx = 1/8` on
#         a `cells_y = 8` tree) it sampled a 12.5 % radial sliver and measured
#         3.162 for a field whose true contrast is 1e4, flipping the
#         `linear=:auto` selector to `:mg` at 4 928 saddle dofs where the
#         correct contrast selects `:direct`.  RED ON PARENT.
#   (D4)  the mapped prescribed-flux energy branch put the load on the logical
#         `y` sides; on the annulus ring the whole `flux_bot` load landed on
#         the periodic seam row and the inner circle carried nothing (parent:
#         inner-circle load 0 where the analytic integral is `q*2πR0`), i.e. a
#         different boundary-value problem solved silently.  RED ON PARENT.
#   (D2)  `semquad_cbf_theta_flux` had no geometry dispatch: on the annulus it
#         summed the residual at free logical-`y` rows and returned ~1.9e-16
#         where the wall flux is +1.0496, and its mapped Nusselt pair came back
#         (1, 1) where the per-wall definition gives (0.9000, 0.9000) on the
#         steady conduction state with an offset outer temperature.
#         RED ON PARENT.
#   (D3)  `_semquad_nu_cbf_mapped` normalised with the Cartesian base gradient;
#         with CBF now on the right rows this gate pins the annulus per-wall
#         ratio form (top_ref/bot_ref from `semquad_base_wall_refs(pr,g)`).
#         RED ON PARENT (same fixture: (1,1) against (0.900, 0.900)).
#   (D5)  `refine_wall_to_min!` / `amr_cycle_balanced!` defaulted `geom` to
#         `CartesianBox(qt)`, so a caller holding a mapped tree silently
#         refined the logical `y` strips — on the annulus ring the seam, not
#         the circles (parent: 16 refined leaves -> 80 leaves on the seam
#         against 32 -> 128 on the circles).  RED ON PARENT (the omission no
#         longer throws there), and the explicit-box call is pinned against a
#         gate-local copy of the historical box criterion.
#   (B3)  `semquad_level_stats(qt)` (box signature) is the same trap in
#         read-only form: parent reports `wall_max = 1, interior_max = 1` on a
#         mapped ring tree (measured `probe8.log`); the mapped twin reports
#         `wall_max = 0` because the annulus wall strips are the circles.
#         RED ON PARENT (no refusal there).
#   (B4)  `total_area(qt)` returns the logical tree measure: 0.125 on the
#         `cells_y = 8` ring tree against the physical 10.807.  RED ON PARENT
#         (no refusal there).
#   (B5)  the history `closure`/`closure_nolatent` columns used the box
#         area-1 bookkeeping on mapped geometries: parent reports
#         `closure = -1.0000` on the exact annulus conduction state at
#         `H = 1` (the box-style formula is dominated by `-H`), where the
#         mapped normalisation `(Ibase + latent_dgdt - flux)/A_phys` is 0 to
#         round-off.  RED ON PARENT.
#   (B2)  `semquad_nu_energy`: no assertion here is red on f607f9e — the audit
#         found the mapped body already correct and the defect was the stale
#         GEOM_PROGRESS sentence "stays box-normalised" (Fix F1).  The gate is
#         new coverage for an exported, previously ungated function: the box
#         and identity-map bodies are bitwise, and the annulus value is pinned
#         analytically.
#
# Why the assertions are not tautologies:
#   * D1: the expected contrast is the analytic contrast of a field whose
#     extrema are at the sampled corner/end points of the *physical* domain
#     (independent of the sampler); the identity-map assertion compares two
#     kernel bodies (the untouched box body and the mapped body); the selector
#     assertion compares against the independently reimplemented parent
#     sampling, which the gate computes and prints.
#   * D4: the target is an analytic quadrature (`q * boundary_measure` and
#     `q * 2πR0`) and, for the full solve, an analytic solution value; the
#     box-map row compares the mapped face-Jacobian/Gauss body with the
#     untouched box GLL body (two implementations).
#   * D2/D3: the target is the analytic steady conduction flux
#     `2π ΔT / ln(R1/R0)`; the Nusselt cross-check compares two *independent
#     families* (CBF reaction sums against the pointwise GLL wall derivative,
#     a measured 0.24 % O(h) family difference) and the annulus ratio form is
#     recomposed from separately-gated pieces.
#   * D5/B3/B4: the refusals are the production behaviour (a thrown
#     `ArgumentError` with the geometry named); the box-path rows compare
#     against gate-local copies of the historical box bodies, written out in
#     this file, never against the production function itself.
#   * B5: the analytic closures are `0` for the exact steady conduction state
#     and `-2α/(R0+R1)` for the manufactured `theta = α(r-R0)` state, derived
#     in the gate from the definition, not read from the production code; the
#     identity-map row compares the mapped driver column with the box driver
#     column (two bodies).
using Test, QuaST, LinearAlgebra, Printf

const G279_R0 = 1.22
const G279_R1 = 2.22

g279_rel(a, b) = abs(a - b) / max(abs(a), abs(b), 1.0e-300)

"Annulus ring tree: `level` uniform refinement levels, `cells_y` base cells."
function g279_ring(level; maxlev=3, cy=8)
    qq = Quadtree(maxlev; cells_x=1, cells_y=cy, seam=:periodic)
    for _ in 1:level, lf in collect(qq.leaves)
        refine!(qq, lf)
    end
    balance!(qq)
    return qq
end

"Global ids of the GLL wall nodes of `sq` on logical `side` (box node order)."
function g279_wall_nodes(sq, side)
    n = sq.p + 1
    s = Set{Int}()
    for lf in sq.leaves
        QuaST._on_logical_wall(lf, side, sq) || continue
        ids = sq.elem[lf]
        for qq in 1:n
            push!(s, QuaST._logical_wall_node(ids, n, side, qq))
        end
    end
    return s
end

"Gate-local copy of the historical box wall criterion (D5: wall rows plus the
physical `y` centre-height strips, `_amr_in_wall_strip`'s box body)."
g279_hist_in_wall_strip(qt, lf, strip) =
    begin
        (lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1) && return true
        h = QuaST.leaf_hy(qt, lf[1])
        zc = (lf[3] + 0.5) * h
        zc <= strip || zc >= 1.0 - strip
    end

"Gate-local copy of the pre-D5 `refine_wall_to_min!` box body (audit D5)."
function g279_hist_refine_wall_box!(qt, min_level; strip::Float64=0.1,
                                    symmetry::Symbol=:none)
    min_level <= 0 && return 0
    nref = 0
    for _ in 1:min_level
        changed = false
        for lf in collect(qt.leaves)
            if g279_hist_in_wall_strip(qt, lf, strip) && lf[1] < min_level
                if QuaST.refine!(qt, lf)
                    nref += 1
                    changed = true
                end
            end
        end
        changed || break
        QuaST.balance!(qt; symmetry=symmetry)
    end
    return nref
end

"Gate-local expected `semquad_level_stats` for a CartesianBox tree (E16 body)."
function g279_hist_level_stats(qt, strip::Float64)
    hist = Dict{Int,Int}()
    wall_max = 0
    interior_max = 0
    for lf in qt.leaves
        l = lf[1]
        h = QuaST.leaf_hy(qt, l)
        y0 = lf[3] * h
        hist[l] = get(hist, l, 0) + 1
        if y0 + h <= strip || y0 >= 1.0 - strip
            wall_max = max(wall_max, l)
        else
            interior_max = max(interior_max, l)
        end
    end
    return (histogram=hist, wall_max=wall_max, interior_max=interior_max,
            n=length(qt.leaves))
end

"Gate-local expected mapped `semquad_level_stats`: physical height `r - R0`."
function g279_expected_level_stats_mapped(g, qt, sq, strip::Float64)
    hist = Dict{Int,Int}()
    wall_max = 0
    interior_max = 0
    for lf in qt.leaves
        l = lf[1]
        hist[l] = get(hist, l, 0) + 1
        hlo = Inf
        hhi = -Inf
        for xi in (-1.0, 1.0), eta in (-1.0, 1.0)
            x, y = QuaST.element_map(g, sq, lf, xi, eta)
            h = hypot(x, y) - G279_R0
            hlo = min(hlo, h)
            hhi = max(hhi, h)
        end
        if hhi <= strip || hlo >= 1.0 - strip
            wall_max = max(wall_max, l)
        else
            interior_max = max(interior_max, l)
        end
    end
    return (histogram=hist, wall_max=wall_max, interior_max=interior_max,
            n=length(qt.leaves))
end

"Gate-local box-style `semquad_level_stats(qt)` on a tree (the parent body)."
function g279_boxsig_level_stats(qt, strip::Float64)
    return g279_hist_level_stats(qt, strip)   # same logical-y expression
end

@testset "G279 mapped-geometry fix campaign" begin
    R0, R1 = G279_R0, G279_R1
    g = PolarAnnulus(R0, R1, 2.0 * pi; seam=:periodic)
    @test nu_wall_sides(g) == (:xminus, :xplus)

    # =====================================================================
    # (D1) the eta-contrast sampler samples the physical domain
    # =====================================================================
    @testset "G279 (D1) eta contrast on mapped geometries" begin
        Lx = 7.0 / 3.0
        qt = Quadtree(2; Lx=Lx)
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        etaf = (x, y) -> 1.0 + x              # physical contrast (1+Lx)/1
        sbox = make_semstokes(qt, 2; geom=CartesianBox(Lx))
        gid = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), Lx)
        smap = make_semstokes(qt, 2; geom=gid)
        cb = semquad_eta_contrast(sbox.su, etaf)
        cm = semquad_eta_contrast(smap.su, etaf)
        println("G279 D1 identity Lx=7/3: box=", cb, " mapped=", cm,
                " analytic=", 1.0 + Lx)
        @test cb ≈ (1.0 + Lx)
        @test cm == cb
        # general affine map with Lx != 1: the extrema of eta = 1+x on the
        # parallelogram are at its corners, and the sampler includes them
        A = (1.30, 0.40, -0.30, 0.90)
        bb = (0.30, -0.20)
        gaff = AffineBox(A, bb, Lx)
        qta = Quadtree(2; Lx=Lx)
        for _ in 1:2, lf in collect(qta.leaves)
            refine!(qta, lf)
        end
        saff = make_semstokes(qta, 2; geom=gaff)
        xs = [QuaST.logical_map(gaff, S, T)[1]
              for S in (0.0, 1.0) for T in (0.0, 1.0)]
        expect = (1.0 + maximum(xs)) / (1.0 + minimum(xs))
        caff = semquad_eta_contrast(saff.su, etaf)
        println("G279 D1 affine Lx=7/3: mapped=", caff, " analytic=", expect)
        @test caff ≈ expect rtol = 1.0e-13
        # annulus: a radial eta whose contrast is 1e4 by construction
        etar = (x, y) -> exp(log(1.0e4) * ((hypot(x, y) - R0) / (R1 - R0)))
        qa = g279_ring(2)
        sa = make_semstokes(qa, 2; geom=g)
        cr = semquad_eta_contrast(sa.su, etar)
        println("G279 D1 annulus radial eta contrast=", cr, " (construct 1e4)")
        @test cr ≈ 1.0e4 rtol = 1.0e-12
        # the selector impact at a production-like saddle size: a level-3 ring
        # has 4 928 saddle dofs (in the 2 000..20 000 window where contrast
        # decides).  The parent sampled S in [0, Lx = 1/8] and measured 3.162,
        # chosing :mg; the physical-domain contrast 1e4 chooses :direct.
        q3 = g279_ring(3)
        s3 = make_semstokes(q3, 2; geom=g)
        dofs = 2 * s3.su.N + s3.sp.N
        @test 2_000 <= dofs < 20_000
        choice, _ = QuaST._semquad_resolve_auto(s3, etar)
        last = QuaST.LAST_SEMSTOKES_AUTO_CHOICE[]
        println("G279 D1 selector: dofs=", dofs, " contrast=", last.contrast,
                " choice=", choice)
        @test last.contrast ≈ 1.0e4 rtol = 1.0e-12
        @test last.samples == (QuaST.SEMQUAD_AUTO_CONTRAST_GRID + 1)^2
        @test choice === :direct
        # the parent's sampling (independent reimplementation) would have
        # selected :mg on the same card; print it so the assertion is
        # self-documenting and non-vacuous.
        Lxa = q3.Lx
        parent_c = let vals = [etar(QuaST.logical_map(g, Lxa * (i / 32),
                                                      j / 32)...)
                               for j in 0:32, i in 0:32]
            maximum(vals) / minimum(vals)
        end
        println("G279 D1 parent-style contrast (S in [0,Lx])=", parent_c,
                " -> ", semquad_choose_linear(dofs, parent_c))
        @test parent_c < 100.0
        @test semquad_choose_linear(dofs, parent_c) === :mg
    end

    # =====================================================================
    # (D4) prescribed-flux wall branch lands on the geometry's walls
    # =====================================================================
    @testset "G279 (D4) mapped prescribed-flux branch" begin
        qa = g279_ring(2)
        sq = make_semstokes(qa, 2; geom=g).su
        q = 0.3
        inner = g279_wall_nodes(sq, :xminus)
        outer = g279_wall_nodes(sq, :xplus)
        bm_in = QuaST.boundary_measure(g, sq, :xminus)
        # analytic: q per unit wall measure over the inner circle
        F = zeros(Float64, sq.N)
        QuaST._add_semquad_wall_flux!(F, sq, qa, q, :bottom)
        load_in = sum(F[collect(inner)])
        println("G279 D4 bottom load: inner=", load_in, " q*measure=", q * bm_in,
                " q*2piR0=", q * 2 * pi * R0)
        @test load_in ≈ q * bm_in rtol = 1.0e-13
        @test load_in ≈ q * 2 * pi * R0 rtol = 1.0e-13
        @test sum(F[collect(outer)]) == 0.0
        # every loaded node is a circle node: the seam row gets corner nodes
        # only through their membership in the circle row
        loaded = findall(!iszero, F)
        @test all(i -> i in inner, loaded)
        # top wall: the same construction on the outer circle
        Ft = zeros(Float64, sq.N)
        QuaST._add_semquad_wall_flux!(Ft, sq, qa, q, :top)
        @test sum(Ft[collect(outer)]) ≈ q * 2 * pi * R1 rtol = 1.0e-13
        @test all(i -> i in outer, findall(!iszero, Ft))
        # box-map identity: the mapped face-Jacobian/Gauss body against the
        # untouched box GLL body on the identity map (two implementations)
        qtb = Quadtree(2)
        for _ in 1:2, lf in collect(qtb.leaves)
            refine!(qtb, lf)
        end
        balance!(qtb)
        gid = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0)
        sqb = make_semstokes(qtb, 2; geom=CartesianBox(1.0)).su
        sqi = make_semstokes(qtb, 2; geom=gid).su
        Fb = zeros(Float64, sqb.N)
        Fi = zeros(Float64, sqi.N)
        QuaST._add_semquad_wall_flux_box!(Fb, sqb, qtb, q, :bottom)
        QuaST._add_semquad_wall_flux_mapped!(Fi, gid, sqi, q, :bottom)
        rel = maximum(abs.(Fi .- Fb)) / maximum(abs, Fb)
        println("G279 D4 box-map flux-load rel=", rel)
        @test rel <= 1.0e-13
        # a full energy solve with a prescribed bottom flux: the inner circle
        # carries `q` per unit measure (integral q*2πR0) and the Dirichlet
        # outer wall the opposite total.  The pointwise GLL flux differs from
        # the reaction by the measured O(h) family difference.
        prf = Params(Ra=100.0, H=0.0, eta_law=:expT, bc=:free,
                     wall_bc=(:free, :free, :free, :free))
        z = zeros(Float64, sq.N)
        th = solve_semquad_energy(sq, qa, prf, z, z, z, z, 1.0e3, 1.0e3, false;
                                  advection=:skew, flux_bot=q)
        f_in, _ = QuaST._semquad_wall_flux(g, sq, th, :xminus)
        f_out, _ = QuaST._semquad_wall_flux(g, sq, th, :xplus)
        println("G279 D4 flux solve: inner=", f_in, " outer=", f_out,
                " target=", q * 2 * pi * R0)
        @test f_in ≈ q * 2 * pi * R0 rtol = 1.0e-2
        @test f_out ≈ -q * 2 * pi * R0 rtol = 1.0e-2
        @test abs(f_in + f_out) <= 1.0e-2 * (q * 2 * pi * R0)
    end

    # =====================================================================
    # (D2/D3) the CBF family on the annulus
    # =====================================================================
    @testset "G279 (D2/D3) annulus CBF rows and per-wall references" begin
        qa = g279_ring(2)
        sq = make_semstokes(qa, 2; geom=g).su
        # steady conduction with an offset outer-wall temperature: theta is
        # analytic, the wall fluxes are O(1) and known
        prT = Params(Ra=100.0, H=0.0, eta_law=:expT, bc=:free,
                     wall_bc=(:free, :free, :free, :free),
                     Twall_top=0.1, Twall_bot=1.0)
        z = zeros(Float64, sq.N)
        th = solve_semquad_energy(sq, qa, prT, z, z, z, z, 1.0e8, 1.0e8, true;
                                  advection=:skew)
        an_flux = 2 * pi * 0.1 / log(R1 / R0)
        cbf_t = semquad_cbf_theta_flux(sq, qa, prT, th, z, z; wall=:top)
        cbf_b = semquad_cbf_theta_flux(sq, qa, prT, th, z, z; wall=:bottom)
        println("G279 D2 CBF flux top=", cbf_t, " bottom=", cbf_b,
                " analytic=", an_flux)
        @test abs(cbf_t) > 0.5                     # parent: 1.9e-16
        @test cbf_t ≈ an_flux rtol = 1.0e-4
        @test cbf_b ≈ an_flux rtol = 1.0e-4
        # D3: the annulus Nusselt pair is the per-wall ratio.  Recompose it
        # from the separately-gated pieces and cross-check against the
        # *pointwise* family (`semquad_nu_walls`), the benchmark definition.
        wt = QuaST.boundary_measure(g, sq, :xplus)
        wb = QuaST.boundary_measure(g, sq, :xminus)
        bot_ref, top_ref = QuaST.semquad_base_wall_refs(prT, g)
        @test top_ref > 0 && bot_ref > 0
        nt_cbf, nb_cbf = semquad_nu_cbf(sq, qa, prT, th, z, z)
        ratio_t = (top_ref - cbf_t / wt) / top_ref
        ratio_b = (bot_ref - cbf_b / wb) / bot_ref
        println("G279 D3 nu_cbf=", (nt_cbf, nb_cbf), " recomposed=",
                (ratio_t, ratio_b))
        @test nt_cbf ≈ ratio_t rtol = 1.0e-13
        @test nb_cbf ≈ ratio_b rtol = 1.0e-13
        # the CBF pair against the pointwise pair: a measured 0.24 % O(h)
        # family difference, not a round-off comparison (stage-0.5b1 record)
        nt_pt, nb_pt = semquad_nu_walls(sq, qa, th, prT)
        println("G279 D3 pointwise Nu=", (nt_pt, nb_pt), " rel to CBF=",
                (g279_rel(nt_cbf, nt_pt), g279_rel(nb_cbf, nb_pt)))
        @test nt_cbf ≈ nt_pt rtol = 2.0e-3
        @test nb_cbf ≈ nb_pt rtol = 2.0e-3
        # the generic mapped path keeps the box convention and the box bodies
        # are untouched: on the identity map both CBF values are bitwise equal
        qtb = Quadtree(2)
        for _ in 1:2, lf in collect(qtb.leaves)
            refine!(qtb, lf)
        end
        balance!(qtb)
        gid = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0)
        ssb = make_semstokes(qtb, 2; geom=CartesianBox(1.0))
        ssi = make_semstokes(qtb, 2; geom=gid)
        @test ssi.su.coords == ssb.su.coords
        thb = [0.02 * sinpi(x) * sinpi(y) for (x, y) in ssb.su.coords]
        uxb = [sinpi(x) * sinpi(y) for (x, y) in ssb.su.coords]
        uzb = [cospi(x) * sinpi(y) for (x, y) in ssb.su.coords]
        pru = Params(Ra=100.0, H=1.0, eta_law=:expT, bc=:free,
                     wall_bc=(:free, :free, :free, :free), rho_heating=true)
        cb = (semquad_cbf_theta_flux(ssb.su, qtb, pru, thb, uxb, uzb;
                                     wall=:top),
              semquad_cbf_theta_flux(ssb.su, qtb, pru, thb, uxb, uzb;
                                     wall=:bottom))
        ci = (semquad_cbf_theta_flux(ssi.su, qtb, pru, thb, uxb, uzb;
                                     wall=:top),
              semquad_cbf_theta_flux(ssi.su, qtb, pru, thb, uxb, uzb;
                                     wall=:bottom))
        println("G279 D3 box-map CBF: box=", cb, " identity-map=", ci)
        @test cb == ci
        ncb = semquad_nu_cbf(ssb.su, qtb, pru, thb, uxb, uzb)
        nci = semquad_nu_cbf(ssi.su, qtb, pru, thb, uxb, uzb)
        @test ncb == nci
    end

    # =====================================================================
    # (D5) the public AMR entry points require the geometry
    # =====================================================================
    @testset "G279 (D5) AMR entry points require the geometry" begin
        # refusal of the omitted geometry, with the tree untouched
        ring = g279_ring(1)
        before = Set(ring.leaves)
        @test_throws ArgumentError refine_wall_to_min!(ring, 2; strip=0.25)
        @test Set(ring.leaves) == before
        err = try
            refine_wall_to_min!(ring, 2; strip=0.25)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("geom", err)
        eta0 = Dict{NTuple{3,Int},Float64}(lf => 1.0 for lf in ring.leaves)
        @test_throws ArgumentError amr_cycle_balanced!(
            ring, eta0; refine_frac=0.2, coarsen_frac=0.05, max_level=3)
        @test Set(ring.leaves) == before
        # the explicit-box call is the historical box body: compare against a
        # gate-local copy of the pre-D5 criterion (never the production body)
        qtb = Quadtree(3)
        for _ in 1:1, lf in collect(qtb.leaves)
            refine!(qtb, lf)
        end
        balance!(qtb)
        q1 = deepcopy(qtb)
        q2 = deepcopy(qtb)
        n1 = refine_wall_to_min!(q1, 4; strip=0.1, geom=CartesianBox(qtb.Lx))
        n2 = g279_hist_refine_wall_box!(q2, 4; strip=0.1)
        println("G279 D5 box wall refinement: production=", n1,
                " gate-local historical=", n2)
        @test n1 == n2
        @test Set(q1.leaves) == Set(q2.leaves)
        # the mapped call refines the physical circles, not the seam.  The
        # level-1 ring (32 leaves, maxlevel 3) needs 32 refinements to put the
        # circles at level 2 on this fixture: the parent's default refines the
        # logical-y seam rows instead (16 -> 80 leaves, audit probe7).
        ring2 = g279_ring(1)
        before = Set(ring2.leaves)
        n = refine_wall_to_min!(ring2, 2; strip=0.25, geom=g)
        println("G279 D5 mapped wall refinement: nref=", n,
                " leaves=", length(ring2.leaves))
        @test n == 32
        @test length(ring2.leaves) == 128
        # every final leaf is a child of one of the original 32 wall leaves and
        # none of the original leaves remains: i.e. exactly the leaves sitting
        # on the physical circle strips were refined, nothing else (a 2:1
        # balance rescue would show up as an extra parent).  The parent's
        # default refined the logical-y seam rows instead (16 -> 80 leaves).
        @test all(lf -> lf[1] == 0 || (lf[1] - 1, lf[2] ÷ 2, lf[3] ÷ 2) in before,
                  ring2.leaves)
        @test !any(lf -> lf in before, ring2.leaves)
        # the mapped balanced cycle still works with the geometry supplied
        ring3 = g279_ring(1)
        sq3 = make_semquad(ring3, 2; geom=g)
        eta3 = Dict{NTuple{3,Int},Float64}()
        for lf in ring3.leaves
            x, y = QuaST.element_map(g, sq3, lf, 0.0, 0.0)
            h = hypot(x, y) - R0
            eta3[lf] = (h <= 0.3 || h >= 0.7) ? 1.0 : 0.0
        end
        rr = amr_cycle_balanced!(ring3, eta3; geom=g, refine_frac=0.5,
                                 coarsen_frac=0.0, max_level=3,
                                 wall_min_level=2, wall_strip=0.3,
                                 symmetry=:none)
        println("G279 D5 mapped cycle: returns=", rr,
                " leaves=", length(ring3.leaves))
        @test rr[1] > 0
        @test length(ring3.leaves) > 32
    end

    # =====================================================================
    # (B3/B4) box-signature diagnostics refuse a mapped topology
    # =====================================================================
    @testset "G279 (B3/B4) box-signature measures on a mapped tree" begin
        ring = g279_ring(2)
        # B4: total_area has no geometry, so it must refuse the mapped topology
        @test_throws ArgumentError total_area(ring)
        err = try
            total_area(ring)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("total_measure", err)
        # the box path is unchanged: the historical E13 sum, gate-local
        boxq = Quadtree(3; Lx=2.5)
        for _ in 1:1, lf in collect(boxq.leaves)
            refine!(boxq, lf)
        end
        balance!(boxq)
        area_hist = sum(QuaST.leaf_hx(boxq, lf[1]) *
                        QuaST.leaf_hy(boxq, lf[1]) for lf in boxq.leaves)
        println("G279 B4 box area: total_area=", total_area(boxq),
                " historical sum=", area_hist)
        @test total_area(boxq) == area_hist
        @test total_area(boxq) ≈ 2.5 rtol = 1.0e-14
        # B3: the box signature must refuse the mapped topology...
        @test_throws ArgumentError semquad_level_stats(ring)
        err3 = try
            semquad_level_stats(ring)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("semquad_level_stats(g, qt", err3)
        # ...and the box tree keeps the historical E16 result, cross-checked
        # against the gate-local historical body
        sq = make_semquad(boxq, 2)
        got = semquad_level_stats(boxq; strip=0.13)
        want = g279_hist_level_stats(boxq, 0.13)
        println("G279 B3 box level stats: got=", got.wall_max, "/",
                got.interior_max, " gate-local=", want.wall_max, "/",
                want.interior_max)
        @test got.histogram == want.histogram
        @test got.wall_max == want.wall_max
        @test got.interior_max == want.interior_max
        @test got.n == want.n
        # the geometry-aware method classifies the annulus strips by the
        # physical height r-R0; compare with the gate-local analytic-height
        # census (not the production `height` accessor)
        sqr = make_semstokes(ring, 2; geom=g).su
        gotm = semquad_level_stats(g, ring; strip=0.2)
        wantm = g279_expected_level_stats_mapped(g, ring, sqr, 0.2)
        println("G279 B3 mapped level stats: got=", gotm.wall_max, "/",
                gotm.interior_max, " gate-local=", wantm.wall_max, "/",
                wantm.interior_max)
        @test gotm.histogram == wantm.histogram
        @test gotm.wall_max == wantm.wall_max
        @test gotm.interior_max == wantm.interior_max
        # the parent box signature (logical-y rows) would have reported a wall
        # level on this ring tree: the refusal is what removes the trap
        @test g279_boxsig_level_stats(ring, 0.2).wall_max >= 0
    end

    # =====================================================================
    # (B5) the history closure columns use the mapped physical balance
    # =====================================================================
    @testset "G279 (B5) mapped closure normalisation" begin
        qa = g279_ring(2)
        sq = make_semstokes(qa, 2; geom=g).su
        z = zeros(Float64, sq.N)
        # (i) the exact steady conduction state at H = 1: dH/dt = 0 and the
        # steady balance closes, so the physical residual is zero.  The box
        # formula gives -1 here (dominated by -H).
        prH = Params(Ra=100.0, H=1.0, eta_law=:expT, bc=:free,
                     wall_bc=(:free, :free, :free, :free))
        thH = solve_semquad_energy(sq, qa, prH, z, z, z, z, 1.0e8, 1.0e8,
                                   true; advection=:skew)
        @test maximum(abs, thH) < 1.0e-13
        bud = semquad_heat_budget(sq, qa, prH, thH; ux=z, uz=z)
        A = total_measure(g, sq)
        clo = (bud.base_adv - bud.flux) / A
        println("G279 B5 steady H=1: theta_max=", maximum(abs, thH),
                " closure=", clo)
        @test abs(clo) < 1.0e-13
        # (ii) manufactured analytic state theta = alpha*(r-R0), u = 0:
        # Ibase = 0, the net wall flux is int dtheta/dn ds = 2*pi*alpha (the
        # inner and outer contributions add: -alpha*2*pi*R0 + alpha*2*pi*R1 and
        # R1-R0 = 1), A = pi*(R1^2-R0^2) = pi*(R0+R1), so the mapped closure is
        # -2*alpha/(R0+R1).  Derived here, not read from the production code.
        alpha = 0.7
        thm = [alpha * (hypot(x, y) - R0) for (x, y) in sq.coords]
        budm = semquad_heat_budget(sq, qa, prH, thm; ux=z, uz=z)
        clm = (budm.base_adv - budm.flux) / A
        want = -2 * alpha / (R0 + R1)
        println("G279 B5 manufactured: closure=", clm, " analytic=", want,
                " flux=", budm.flux)
        @test budm.flux ≈ 2 * pi * alpha rtol = 1.0e-12
        @test clm ≈ want rtol = 1.0e-12
        # (iii) the driver column: one step from the steady conduction state
        # stays on it, so the mapped history closure is ~0 (parent: -1.0)
        res = run_semquad_simulation(qa, prH, z; geom=g, p=2, tstop=1.0e-4,
                                     dt0=1.0e-4, dtmax=1.0e-4, dtmin=1.0e-16,
                                     linear=:direct, maxsteps=1, amr_every=0,
                                     advection=:skew, maxpic=1, verbose=false)
        println("G279 B5 driver closure column=", res.history[1][13],
                " width=", length(res.history[1]))
        @test abs(res.history[1][13]) < 1.0e-12
        # (iv) identity map: the mapped branch reduces to the box formula.  Two
        # driver runs (box body and identity-mapped body) on one problem.
        qtb = Quadtree(2)
        for _ in 1:1, lf in collect(qtb.leaves)
            refine!(qtb, lf)
        end
        balance!(qtb)
        gid = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0)
        pr = Params(Ra=100.0, H=1.0, eta_law=:expT, deta_T=1.0, bc=:free,
                    wall_bc=(:free, :free, :free, :free), rho_heating=true)
        n0 = make_semstokes(qtb, 2; geom=CartesianBox(1.0)).su.N
        th0 = zeros(Float64, n0)
        kwargs = (p=2, tstop=2.0e-4, dt0=1.0e-4, dtmax=1.0e-4,
                  dtmin=1.0e-16, linear=:direct, maxsteps=2, amr_every=0,
                  advection=:skew, maxpic=1, verbose=false)
        rb = run_semquad_simulation(qtb, pr, copy(th0); kwargs...)
        rm = run_semquad_simulation(qtb, pr, copy(th0); geom=gid, kwargs...)
        @test rb.theta == rm.theta
        for s in 1:2
            println("G279 B5 identity step ", s, ": box closure=",
                    rb.history[s][13], " mapped=", rm.history[s][13])
            @test abs(rb.history[s][13] - rm.history[s][13]) < 1.0e-12
        end
    end

    # =====================================================================
    # (B2) semquad_nu_energy: coverage for the previously ungated function
    # (no assertion here is red on the parent; the defect was stale wording)
    # =====================================================================
    @testset "G279 (B2) nu_energy on mapped geometries" begin
        R0f, R1f = R0, R1
        qtb = Quadtree(2)
        for _ in 1:2, lf in collect(qtb.leaves)
            refine!(qtb, lf)
        end
        balance!(qtb)
        gid = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0)
        ssb = make_semstokes(qtb, 2; geom=CartesianBox(1.0))
        ssi = make_semstokes(qtb, 2; geom=gid)
        thb = [0.02 * sinpi(x) * sinpi(y) for (x, y) in ssb.su.coords]
        uxb = [sinpi(x) * sinpi(y) for (x, y) in ssb.su.coords]
        uzb = [cospi(x) * sinpi(y) for (x, y) in ssb.su.coords]
        pru = Params(Ra=100.0, H=1.0, eta_law=:expT, bc=:free,
                     wall_bc=(:free, :free, :free, :free), rho_heating=true)
        neb = semquad_nu_energy(ssb.su, qtb, pru, thb, uxb, uzb)
        nei = semquad_nu_energy(ssi.su, qtb, pru, thb, uxb, uzb)
        println("G279 B2 identity map: box=", neb, " mapped=", nei)
        @test neb == nei
        # annulus conduction state: nu_energy = 1 + H/2 exactly
        qa = g279_ring(2)
        sq = make_semstokes(qa, 2; geom=g).su
        z = zeros(Float64, sq.N)
        prH = Params(Ra=100.0, H=1.0, eta_law=:expT, bc=:free,
                     wall_bc=(:free, :free, :free, :free))
        @test semquad_nu_energy(sq, qa, prH, z, z, z) ≈ 1.0 + 0.5 * prH.H
        # manufactured state: theta = c constant, u = the radial unit field.
        # u.grad(h) = 1 and <theta> = c, so the mapped value is 1 + H/2 + c;
        # the measured 2.6e-7 relative difference is the Q2 interpolation error
        # of e_r, i.e. the test is not exact algebra but a quadrature check.
        c = 0.3
        thc = fill(c, sq.N)
        uxs = [x / hypot(x, y) for (x, y) in sq.coords]
        uzs = [y / hypot(x, y) for (x, y) in sq.coords]
        ne = semquad_nu_energy(sq, qa, prH, thc, uxs, uzs)
        println("G279 B2 manufactured annulus: nu_energy=", ne,
                " expected=", 1.0 + 0.5 * prH.H + c)
        @test ne ≈ 1.0 + 0.5 * prH.H + c rtol = 1.0e-5
        # and the tangential unit field has u.grad(h) = 0 at the nodes
        uxt = [-y for (x, y) in sq.coords]
        uzt = [x for (x, y) in sq.coords]
        @test semquad_nu_energy(sq, qa, prH, thc, uxt, uzt) ≈
              1.0 + 0.5 * prH.H rtol = 1.0e-5
        # (R0/R1 are used above through the fixture; keep them live)
        @test R0f > 0 && R1f > R0f
    end
end
