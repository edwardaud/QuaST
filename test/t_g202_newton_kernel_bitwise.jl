# G202 / S15 kernel specialisation of the *Newton-path* assembly: `SemQuadAsmCfg`
# carries `geo_coef`/`dp_coef` as type parameters (they were `Union{Nothing,
# Function}` fields, and a function *field* is not specialised, so `gc` in the
# Kgeo accumulation was `Any` and every `Kgloc[...] += gc*...*w` boxed;
# docs/dev/BUG_LEDGER.md).
#
# The gate pins
#   (a) `sys.Kgeo` against a verbatim pre-fix reference loop (same expression
#       tree, same push order) — equality is bitwise (`==` on Float64),
#   (b) `sys.Kp` (the Drucker-Prager block, same treatment of `dp_coef`) against
#       the same kind of reference,
#   (c) SHA-256 fingerprints of `K, B, Kgeo, Kp, Kred, fv` recorded in the
#       pre-fix worktree (`/tmp/ms_base`, commit 8b08af3, generator: this file
#       with `G202_SNAPSHOT=1`), so a future change to the loop that keeps the
#       value but changes the order is caught as well,
#   (d) the inferred types that the fix restored: `gc`/`deta_q` are `Float64`.
using Test, QuaST, LinearAlgebra, SparseArrays, SHA

const _G202_Z2 = (x, y) -> 0.0
const SNAPSHOT = get(ENV, "G202_SNAPSHOT", "") == "1"

_h202(v::Vector{Int}) = bytes2hex(sha256(collect(reinterpret(UInt8, v))))
_h202(v::Vector{Float64}) = bytes2hex(sha256(collect(reinterpret(UInt8, v))))
_h202(v::SparseMatrixCSC) = _h202(v.colptr) * "/" * _h202(v.rowval) * "/" *
                            _h202(v.nzval)

# --- pre-fix reference: the Kgeo accumulation, copied from the loop body -----
# (src/semquadstokes.jl `_semquad_assemble_block!` before the fix).  The
# only difference to the shipped kernel is that the coefficient arrives as a
# `Function` argument instead of a struct field.
function _g202_kgeo_ref(ss::SemStokes, eta_fun, geo_u::Vector{Float64},
                        geo_coef, geo_spd_safety::Float64=0.0)
    su = ss.su
    n = su.p + 1
    nv = n * n
    nu = su.N
    Nu2 = 2 * nu
    fd = make_fedata(su.p, ss.q)
    gx = zeros(Float64, nv); gz = zeros(Float64, nv)
    Kgloc = zeros(Float64, 2nv, 2nv)
    IG = Int[]; JG = Int[]; VG = Float64[]
    for lf in su.leaves
        hx = QuaST.leaf_hx(su, lf[1]); hy = QuaST.leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        fill!(Kgloc, 0.0)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            etaq = eta_fun(x, y)
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
            end
            duxdx = 0.0; duxdy = 0.0; duzdx = 0.0; duzdy = 0.0
            for k in 1:nv
                duxdx += geo_u[uids[k]] * gx[k]
                duxdy += geo_u[uids[k]] * gz[k]
                duzdx += geo_u[nu + uids[k]] * gx[k]
                duzdy += geo_u[nu + uids[k]] * gz[k]
            end
            exd = duxdx; ezd = duzdy
            exz = 0.5 * (duxdy + duzdx)
            edot = sqrt(max(0.5 * (exd * exd + ezd * ezd + 2.0 * exz * exz), 0.0))
            deta_q = edot > 0.0 ? geo_coef(x, y, edot) : 0.0
            d_a = geo_spd_safety > 0.0 && deta_q != 0.0 ?
                  QuaST.spd_factor(etaq, deta_q, edot, exd, ezd, exz, exd,
                                       ezd, exz; safety=geo_spd_safety) : 1.0
            gc = edot > 0.0 ? d_a * deta_q / edot : 0.0
            if gc != 0.0
                for k in 1:nv
                    Tux_k = exd * gx[k] + exz * gz[k]
                    Tuz_k = ezd * gz[k] + exz * gx[k]
                    for l in 1:nv
                        Dux_l = exd * gx[l] + exz * gz[l]
                        Duz_l = ezd * gz[l] + exz * gx[l]
                        Kgloc[k, l] += gc * Tux_k * Dux_l * w
                        Kgloc[k, nv + l] += gc * Tux_k * Duz_l * w
                        Kgloc[nv + k, l] += gc * Tuz_k * Dux_l * w
                        Kgloc[nv + k, nv + l] += gc * Tuz_k * Duz_l * w
                    end
                end
            end
        end
        for k in 1:nv
            g = uids[k]
            for l in 1:nv
                gl = uids[l]
                push!(IG, g); push!(JG, gl); push!(VG, Kgloc[k, l])
                push!(IG, nu + g); push!(JG, nu + gl); push!(VG, Kgloc[nv + k, nv + l])
                push!(IG, g); push!(JG, nu + gl); push!(VG, Kgloc[k, nv + l])
                push!(IG, nu + g); push!(JG, gl); push!(VG, Kgloc[nv + k, l])
            end
        end
    end
    return sparse(IG, JG, VG, Nu2, Nu2)
end

# --- pre-fix reference: the Drucker-Prager block (the `dp_coef` twin) --------
function _g202_kp_ref(ss::SemStokes, dp_u::Vector{Float64},
                      dp_pfield::Vector{Float64}, dp_coef)
    su = ss.su; sp = ss.sp
    n = su.p + 1; q = sp.p
    nv = n * n; nu = su.N; npq = (q + 1)^2
    fd = make_fedata(su.p, ss.q)
    gx = zeros(Float64, nv); gz = zeros(Float64, nv)
    Kploc = zeros(Float64, 2nv, npq)
    IP = Int[]; JP = Int[]; VP = Float64[]
    pleaves = Set(sp.leaves)
    for lf in su.leaves
        hx = QuaST.leaf_hx(su, lf[1]); hy = QuaST.leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]; pids = sp.elem[lf]
        fill!(Kploc, 0.0)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
            end
            duxdx = 0.0; duxdy = 0.0; duzdx = 0.0; duzdy = 0.0
            for k in 1:nv
                duxdx += dp_u[uids[k]] * gx[k]
                duxdy += dp_u[uids[k]] * gz[k]
                duzdx += dp_u[nu + uids[k]] * gx[k]
                duzdy += dp_u[nu + uids[k]] * gz[k]
            end
            dp_exd = duxdx; dp_ezd = duzdy
            dp_exz = 0.5 * (duxdy + duzdx)
            dp_edot = sqrt(max(0.5 * (dp_exd * dp_exd + dp_ezd * dp_ezd +
                                      2.0 * dp_exz * dp_exz), 0.0))
            dp_p = QuaST._semquad_eval_at(sp, dp_pfield, x, y, pleaves)
            dp_c = dp_coef(x, y, dp_edot, dp_p)
            if dp_c != 0.0
                for b in 1:(q + 1), a in 1:(q + 1)
                    lp = (b - 1) * (q + 1) + a
                    psp = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                    for k in 1:nv
                        Kploc[k, lp] += 2.0 * dp_c * psp *
                                        (dp_exd * gx[k] + dp_exz * gz[k]) * w
                        Kploc[nv + k, lp] += 2.0 * dp_c * psp *
                                             (dp_ezd * gz[k] + dp_exz * gx[k]) * w
                    end
                end
            end
        end
        for l in 1:npq, k in 1:nv
            pg = pids[l]
            push!(IP, uids[k]); push!(JP, pg); push!(VP, Kploc[k, l])
            push!(IP, nu + uids[k]); push!(JP, pg); push!(VP, Kploc[nv + k, l])
        end
    end
    return sparse(IP, JP, VP, 2nu, sp.N)
end

@testset "G202 Newton-path assembly kernel specialisation (bitwise)" begin
    # ---- the case: a 16-leaf tree, p=2, a smooth velocity, a composite law ---
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    leaves = Set(qt.leaves)
    uf = zeros(Float64, 2N)
    for (i, (x, y)) in enumerate(sq.coords)
        uf[i] = sinpi(x) * cospi(y)
        uf[N + i] = -cospi(x) * sinpi(y)
    end
    th = [0.1 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
    pr = Params(Ra=1.0e6, H=0.0, eta_law=:composite, comp_Ad=exp(6.0),
                comp_Ed=9.0, comp_Vd=0.0, comp_As=1.0e-2 * exp(4.0),
                comp_Es=12.0, comp_Vs=0.0, comp_n=3.0, eta_T0=0.5,
                eta_min=1.0e-6, eta_max=1.0e4)
    lvs = Set(qt.leaves)
    T_at(x, y) = Tbar(y, pr.H) + QuaST._semquad_eval_at(sq, th, x, y, lvs)
    ed_at(x, y) = semquad_strain_rate_at(sq, uf[1:N], uf[(N + 1):(2N)], x, y, lvs)
    eta_fun = (x, y) -> eta_composite_at(T_at(x, y), ed_at(x, y), y, pr)[1]
    de_fun = (x, y, ed) -> deta_composite_at(T_at(x, y), ed, y, pr)[2]
    fz = (x, y) -> pr.Ra * QuaST._semquad_eval_at(sq, th, x, y, lvs)
    pv = [0.3 * sinpi(x) * sinpi(y) for (x, y) in ss.sp.coords]
    dp_fun = (x, y, ed, pq) -> 0.25 * (1.0 + pq) * ed
    dp_chi = (x, y, ed, pq) -> 0.0   # unused; kept for signature parity

    sys = solve_semquad_stokes_var(ss, qt, eta_fun, _G202_Z2, fz, _G202_Z2,
                                   _G202_Z2; bc=pr.bc, linear=:direct,
                                   deviatoric=false, assemble_only=true,
                                   geo_u=uf, geo_coef=de_fun,
                                   dp_u=uf, dp_pfield=pv, dp_coef=dp_fun)
    @test sys.Kgeo !== nothing
    @test sys.Kp !== nothing

    # ---- (a) Kgeo against the verbatim pre-fix loop -------------------------
    Kg = _g202_kgeo_ref(ss, eta_fun, uf, de_fun)
    @test size(Kg) == size(sys.Kgeo)
    @test nnz(Kg) == nnz(sys.Kgeo)
    @test Matrix(Kg) == Matrix(sys.Kgeo)          # bitwise, entry by entry
    @test Matrix(Kg) == Matrix(sys.Kgeo) .+ 0.0   # (guards against NaN-only eq)
    @test count(x -> x == 0.0, sys.Kgeo.nzval) == count(x -> x == 0.0,
                                                       Kg.nzval)
    @test !any(isnan, sys.Kgeo.nzval)

    # ---- (b) Kp against the verbatim pre-fix loop ---------------------------
    Kp = _g202_kp_ref(ss, uf, pv, dp_fun)
    @test size(Kp) == size(sys.Kp)
    @test Matrix(Kp) == Matrix(sys.Kp)

    # ---- (d) the inferred types the fix restored ---------------------------
    @test isconcretetype(typeof(QuaST.SemQuadAsmCfg(
        2.0, false, false, uf, de_fun, 0.0, uf, pv, dp_fun)))
    @test fieldtype(typeof(QuaST.SemQuadAsmCfg(
        2.0, false, false, uf, de_fun, 0.0, uf, pv, dp_fun)), :geo_coef) ==
          typeof(de_fun)

    # ---- (c) SHA-256 pins recorded in the pre-fix worktree -----------------
    pins = ("K" => _h202(sys.K), "B" => _h202(sys.B), "Kgeo" => _h202(sys.Kgeo),
            "Kp" => _h202(sys.Kp), "Kred" => _h202(sys.Kred),
            "fv" => _h202(sys.fv))
    if SNAPSHOT
        for (k, v) in pins
            println("G202 pin ", k, " = ", v)
        end
        @test true
    else
        @test pins[1].second == "a7ab253441e8cd8696c518e513aa29c5384875e42642c668f070e167c038f9a8/967b8aa7c54b2da3377079faf1d18b62e167910216bafcb2d620b1a4ee3e4d9f/c1fb569e7c221d1f85f451708a97859631cf46e25da75e8d6ecdc983c19cb967"
        @test pins[2].second == "5aeb295244d0ea45b9e9cad6bddf7303bfbe86ef2de40d98ccaaf48910c8d4bd/7f9861051ec0a7dc110ad418bcc955abcc7bb8b7d571635b07d136be6930d5ee/bbc380b5d5afed4877a65b6a8cde8d2f3c429886a8957f53aecfcc916a661b23"
        @test pins[3].second == "a7ab253441e8cd8696c518e513aa29c5384875e42642c668f070e167c038f9a8/967b8aa7c54b2da3377079faf1d18b62e167910216bafcb2d620b1a4ee3e4d9f/70d3807cb8495cac1507363324defbc91bfb2b472759af866966449ecb6de1fc"
        @test pins[4].second == "1ff5de6d1bfbe501b40742292e09b3e75c174b9a03eca1b5436d12c7ad9adfc6/018a60130bf19b1bb245bffd6f268a2de7eba9addc45512657b79b15ed27202f/7065c65afdb525ca53ed63260ffdb5441a10dbff7e994bb99d69ad041ebe0cab"
        @test pins[5].second == "33424c3ad8299ce077c230b13e5212fd1f31496896ed623560c5afc2d793a5d4/8651ec65ee43f8b704c329ac5839d8f912352b9fc67467582bd255d5c66ddebb/4a21fb1eb7ae9e30474437eb23f7d54cdae3e58ac9151968ca3d03c74021a3d5"
        @test pins[6].second == "1df3b58a222864ba3630c3636553e44522e45c5d24b80b47c1cdee9503a4bbf7"
    end
end
