# G188: assembly and point-evaluation kernel specialisation is bitwise.
using Test, QuaST, LinearAlgebra, SparseArrays, SHA, Random

# G188 / S3 stage 4: kernel specialisation of the assembly and point-evaluation
# loops is bitwise inert.
#
# The stage-4 restructure touches five kernels, all of which keep the exact
# floating-point expression tree of every entry and only move *loop-invariant*
# work out of the inner loops:
#
#   1. `_semquad_assemble_block!` (Stokes K/B/fv): the (k,l) double loop is
#      transposed so `k` is innermost (unit-stride stores, was stride 2*nv) and
#      the per-(l) scalars are hoisted.  Each entry still accumulates over the
#      quadrature points in the same order.
#   2. `_semquad_assemble_energy_block!` (M/K/C/F): `cM = (1+lat_cp)*w*rho`,
#      `cK = w*(1+nu_ev)`, `ws = w*src` and `cx = uxq*gx[l]+uzq*gz[l]` are
#      hoisted, the inner index is `k`, everything else identical.
#   3. `_semquad_eval_at`: the GLL rule is cached (`gll_rule_cached`) and the
#      values-only `lagrange_values` replaces `lagrange_eval` (same barycentric
#      formula, same summation order).
#   4. `semquad_ev_cell_stats`: the 1-D second-derivative table is built once
#      per leaf instead of four `gll_rule`+`lagrange_d2` calls *per quadrature
#      point* (144 -> 7 calls per leaf at p=4).
#   5. `semquad_rho`/`semquad_work_dissipation`: `::Float64` assertions remove
#      the `Any` that the `Union{Nothing,Function}` reference-state field
#      introduced (the `W` accumulator was boxed per quadrature point).
#
# The gate pins (a) the two small kernels against explicit pre-fix reference
# implementations, (b) the three big ones against SHA-256 fingerprints recorded
# in the pre-fix worktree (commit 4bbdae9, generator
# `out_julia/agent/stage4/g188_snapshot.jl`), and (c) the inferred return types
# that item 5 restored.

const _G188_Z2 = (x, y) -> 0.0

_h188(v) = bytes2hex(sha256(collect(reinterpret(UInt8, v))))
_h188(v::SparseMatrixCSC) = _h188(v.colptr) * "/" * _h188(v.rowval) * "/" *
                            _h188(v.nzval)

# --- pre-fix reference implementations (verbatim loops) ---------------------

"""Pre-fix `_semquad_eval_at`: rule rebuilt per point, `lagrange_eval`."""
function _g188_eval_at_ref(sq::SemQuad, u::Vector{Float64}, x::Float64,
                           y::Float64, leaves::Set{NTuple{3,Int}})
    lf = QuaST._semquad_leaf_at(sq, x, y, leaves)
    n = sq.p + 1
    hx = QuaST.leaf_hx(sq, lf[1]); hy = QuaST.leaf_hy(sq, lf[1])
    x0 = lf[2] * hx; y0 = lf[3] * hy
    xi = 2.0 * (x - x0) / hx - 1.0
    eta = 2.0 * (y - y0) / hy - 1.0
    rule = gll_rule(sq.p)
    Lb, _ = lagrange_eval(rule, xi)
    Lc, _ = lagrange_eval(rule, eta)
    ids = sq.elem[lf]
    v = 0.0
    for j in 1:n, i in 1:n
        v += u[ids[(j - 1) * n + i]] * Lb[i] * Lc[j]
    end
    return v
end

"""Pre-fix `semquad_ev_cell_stats`: `gll_rule`/`lagrange_d2` inside the loop."""
function _g188_ev_cell_stats_ref(sq::SemQuad, pr::Params,
                                 theta::Vector{Float64},
                                 theta_prev::Union{Nothing,Vector{Float64}},
                                 ux::Vector{Float64}, uz::Vector{Float64},
                                 dt::Float64, lf::NTuple{3,Int},
                                 evg::QuaST.SEMEVGlobals; alpha::Int=2)
    h = QuaST.leaf_hy(sq, lf[1])
    x0 = lf[2] * h; y0 = lf[3] * h
    ids = sq.elem[lf]
    n = sq.p + 1
    fd = make_fedata(sq.p, sq.p)
    Rmax = 0.0; umax = 0.0
    for qy in 1:fd.nq, qx in 1:fd.nq
        y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * h
        _, _, d2x = lagrange_d2(gll_rule(sq.p), fd.gl_x[qx])
        _, _, d2y = lagrange_d2(gll_rule(sq.p), fd.gl_x[qy])
        uxq = 0.0; uzq = 0.0
        thn = 0.0; thp = 0.0
        gthx = 0.0; gthz = 0.0; lapth = 0.0
        gthxp = 0.0; gthzp = 0.0; lapp = 0.0
        for j in 1:n, i in 1:n
            k = (j - 1) * n + i
            b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
            gx = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / h)
            gz = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / h)
            uxq += ux[ids[k]] * b
            uzq += uz[ids[k]] * b
            thn += theta[ids[k]] * b
            gthx += theta[ids[k]] * gx
            gthz += theta[ids[k]] * gz
            lapth += theta[ids[k]] *
                     (d2x[i] * fd.phi_v[j, qy] * (2.0 / h)^2 +
                      fd.phi_v[i, qx] * d2y[j] * (2.0 / h)^2)
            if theta_prev !== nothing
                thp += theta_prev[ids[k]] * b
                gthxp += theta_prev[ids[k]] * gx
                gthzp += theta_prev[ids[k]] * gz
                lapp += theta_prev[ids[k]] *
                        (d2x[i] * fd.phi_v[j, qy] * (2.0 / h)^2 +
                         fd.phi_v[i, qx] * d2y[j] * (2.0 / h)^2)
            end
        end
        Tmid = Tbar(y, pr.H) + 0.5 * (thn + thp)
        dTdt = dt > 0.0 ? (thn - thp) / dt : 0.0
        adv = 0.5 * uxq * (gthx + gthxp) +
              0.5 * uzq * (gthz + gthzp) + uzq * dTbar(y, pr.H)
        lap = 0.5 * (lapth + lapp)
        R = abs(dTdt + adv - lap)
        alpha == 2 && (R *= abs(Tmid - evg.avgT))
        Rmax = max(Rmax, R)
        umax = max(umax, hypot(uxq, uzq))
    end
    return Rmax, umax
end

function _g188_state(qt, p)
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

@testset "G188 kernel specialisation is bitwise" begin
    # (a1) `lagrange_values` == `lagrange_eval(...)[1]`, bitwise.
    for p in 1:8
        rule = gll_rule(p)
        for x in (-1.0, -0.75, -0.5, -0.2, 0.0, 0.1, 1 / 3, 0.5, 0.9, 1.0,
                  rule.nodes[1], rule.nodes[end], rule.nodes[cld(p + 1, 2)])
            L1 = lagrange_eval(rule, x)[1]
            L2 = QuaST.lagrange_values(rule, x)
            @test L2 == L1
        end
    end

    qt = wall_graded_tree(3, 5)
    balance!(qt)
    p = 3
    ss = make_semstokes(qt, p)
    sq, th, ux, uz = _g188_state(qt, p)
    leaves = Set(sq.leaves)
    eta = (x, y) -> 1.0 + 0.5 * sinpi(x) * cospi(y)

    # (a2) `_semquad_eval_at` == pre-fix reference at 200 points.
    rng = MersenneTwister(188)
    for _ in 1:200
        x = rand(rng); y = rand(rng)
        @test QuaST._semquad_eval_at(sq, th, x, y, leaves) ==
              _g188_eval_at_ref(sq, th, x, y, leaves)
    end

    # (a3) the EV cell statistic (hoisted 1-D table) == pre-fix reference.
    pr = Params(Ra=1.0e6, H=0.5, eta_law=:expT, deta_T=1.0,
                Twall_top=0.0, Twall_bot=1.0, bc=:free, Di=0.1,
                heating_dissipation=true, heating_adiabatic=true,
                rho_heating=true, rho_cp=true)
    prev = [0.2 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
    evg = QuaST.semquad_ev_globals(sq, qt, pr, th, prev, ux, uz,
                                       (8.0e-11) / (7.0e-11))
    for lf in sq.leaves[1:min(20, length(sq.leaves))]
        r1, u1 = QuaST.semquad_ev_cell_stats(sq, pr, th, prev, ux, uz,
                                                 7.0e-11, lf, evg; alpha=2)
        r0, u0 = _g188_ev_cell_stats_ref(sq, pr, th, prev, ux, uz, 7.0e-11,
                                         lf, evg; alpha=2)
        @test r1 == r0
        @test u1 == u0
    end

    # (b1) Stokes K/B/fv fingerprints recorded *before* the restructure
    #      (worktree /tmp/ms_base at 4bbdae9, g188_snapshot.jl).
    fd = make_fedata(p, ss.q)
    pleaves = Set(ss.sp.leaves)
    for dev in (false, true)
        cfg = QuaST.SemQuadAsmCfg(dev ? 4.0 / 3.0 : 2.0, dev, false,
                                      nothing, nothing, 0.0, nothing, nothing,
                                      nothing)
        asm = QuaST._semquad_assemble_blocks(1, cfg, ss.su, ss.sp, eta,
                                                 _G188_Z2, _G188_Z2, fd,
                                                 pleaves)
        if !dev
            @test _h188(asm.IK) == "ce2755b75780aa7b695a163030bd63034a57621a4ecfaef8e68dceffb0b7805e"
            @test _h188(asm.JK) == "7a2116fce1560f2235bef76c64a39cd7f84ee875e2aea0e18115eaa5ce84f270"
            @test _h188(asm.VK) == "c050e6da98b5d54cb2dd09348b13c94317be99296a7aafeddb43468b3c091345"
            @test _h188(asm.IB) == "9fb8475e5a7a3312581b8f7746d7f66d0f6aeaba25363ee05c9ec93f48238fab"
            @test _h188(asm.JB) == "13435ccf836bb885ae6b815b10885c2ce8e71e7e434939200a4bfe1c45df94d1"
            @test _h188(asm.VB) == "411b7c1ba94a41e44dcc55bad1b2d4d64bf25caa2c1a636c9be34755590972bf"
            @test _h188(asm.fv) == "467b59659413f71b7e04e27ca263582e832e1838af0d53b8a282b9da0bc368f5"
        else
            @test _h188(asm.VK) == "a439fdb7b1ed06f1cbe5f6c4e5cdc31e891416ea9fed450857edc33c80b96195"
        end
        @test _h188(asm.IK) == "ce2755b75780aa7b695a163030bd63034a57621a4ecfaef8e68dceffb0b7805e"
    end

    # (b2) energy M/K/C/F fingerprints, every optional term on, EV field on.
    ev = zeros(Float64, length(qt.leaves))
    M, K, C, F = assemble_semquad_energy(sq, qt, pr, ux, uz; advection=:skew,
                                        theta=th, rho_heating=true,
                                        rho_cp=true, heating_adiabatic=true,
                                        heating_dissipation=true,
                                        eta_fun=eta, ev=0.44, ev_cmax=0.052,
                                        ev_alpha=2, ev_timestep=5, evg=evg,
                                        theta_prev=prev, dt=7.0e-11,
                                        ev_out=ev)
    @test _h188(M) == "c5b68d88c82ed8a49322c2bc889c3053537513b1978c58b598faf1d858459b61/d9dbe4e6684a4a3e7b46ca34c3a3701d559d1dd06cd62d0214c76a55ee227db6/d7cac3242f84e231f140b67a1ed9162aaddf8deb41f860574ac6793b41589254"
    @test _h188(K) == "c5b68d88c82ed8a49322c2bc889c3053537513b1978c58b598faf1d858459b61/d9dbe4e6684a4a3e7b46ca34c3a3701d559d1dd06cd62d0214c76a55ee227db6/524283a379979fa40a0af6b53e419e7edd030a7d8ae21b15b352cd9e0adec315"
    @test _h188(C) == "8f6b4c853685974fe38ec42f64c2fa03e3843ec392a97cc6de259e7d10a2ea6d/eb8ebeaced7f0f69a853eef0a766f1ad3b358281ecfcae70ef2e701feee1c581/1d03d775e5fbdf318be37a72ed4928342a4786886ac7f8fc689d1278f5ed2f84"
    @test _h188(F) == "7b26f3badbb3024567fe91917cbbf6fc74a85978a7cdfd9c8150bbbf7ab0a468"
    @test _h188(ev) == "71d752867da97271e0c50dbfd8c3b7f8a9f23d2cac8076748a7a834a92c694aa"

    # (b3) point evaluation on the whole tree and the work/dissipation pair.
    xs = range(0.0, 1.0; length=9)
    vals = Float64[]
    for y in xs, x in xs
        push!(vals, QuaST._semquad_eval_at(sq, th, Float64(x), Float64(y),
                                               leaves))
    end
    @test _h188(vals) == "7abc89cd2c8aa55c5fce061e9dd6757d74a350f5c82b854d889c9b894730176b"
    rmax = Float64[]; umax = Float64[]
    for lf in sq.leaves
        r, u = QuaST.semquad_ev_cell_stats(sq, pr, th, prev, ux, uz,
                                               7.0e-11, lf, evg; alpha=2)
        push!(rmax, r); push!(umax, u)
    end
    @test _h188(rmax) == "bd43b4d17c22e8ba834fe34bf02a0754ecc47b09557dd3c3e42a8d064473d06c"
    @test _h188(umax) == "2358880e2c45aa966eac5347a505a749f935586297c258df55dd37447d5a4ce6"
    W, U = semquad_work_dissipation(sq, qt, pr, th, ux, uz, eta)
    @test _h188([W, U]) == "8921584e76a0066fb13bb546b02f1267e07d17b5488e2a6f5e83eccfa8293ed9"

    # (c) the `Any` that item 5 removed is gone: these must infer concretely.
    @test (@inferred QuaST.semquad_rho(SemRefState(; rho0=2.0), 0.5)) === 2.0
    @test (@inferred semquad_work_dissipation(sq, qt, pr, th, ux, uz, eta)) isa
          Tuple{Float64,Float64}
    @test (@inferred QuaST._semquad_eval_at(sq, th, 0.31, 0.62, leaves)) isa
          Float64
end
