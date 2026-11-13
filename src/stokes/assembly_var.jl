# ---------------------------------------------------------------------------
# S4 stage 1: leaf-level parallel assembly
#
# The Stokes assembler walks `su.leaves` in tree order, accumulates one COO
# triplet list (K, optionally Kgeo/Kp, plus B) and one load vector.  That loop
# is embarrassingly parallel over leaves.  `assemble_threads > 1` cuts the leaf
# list into that many *fixed contiguous* blocks, assembles each block into its
# own COO buffers (one per block), and then concatenates the block buffers in
# block-index order.  The result is deterministic run to run: the leaf->block
# assignment and the merge order do not depend on the scheduler.
#
# It is deterministic run to run: the leaf->block assignment and the merge order
# do not depend on the scheduler.  Because the blocks are contiguous and merged
# in block order, the concatenated COO sequence is exactly the serial leaf
# sequence, so K/Kgeo/Kp/B come out bitwise equal to the serial path (measured;
# G181 only requires <=1e-15 relative).  The one place the summation order does
# change is the load vector: each block accumulates its own `fv` and the block
# sums are then added in order, so `fv` (and hence `fr` and the solution) can
# differ at round-off.  `assemble_threads=1` (the default) keeps the whole path
# bit for bit, so the G150 snapshot still pins direct-solver round-off.
# ---------------------------------------------------------------------------

struct SemQuadAsmScratch
    phi::Vector{Float64}
    gx::Vector{Float64}
    gz::Vector{Float64}
    Kloc::Matrix{Float64}
    Kgloc::Matrix{Float64}
    Bloc::Matrix{Float64}
    floc::Vector{Float64}
    Kploc::Matrix{Float64}
end

struct SemQuadAsmBuf
    IK::Vector{Int}
    JK::Vector{Int}
    VK::Vector{Float64}
    IB::Vector{Int}
    JB::Vector{Int}
    VB::Vector{Float64}
    IG::Vector{Int}
    JG::Vector{Int}
    VG::Vector{Float64}
    IP::Vector{Int}
    JP::Vector{Int}
    VP::Vector{Float64}
    fv::Vector{Float64}
    scr::SemQuadAsmScratch
end

# Everything the leaf kernel needs that is not a leaf or a buffer.
#
# `mf_path` is the matrix-free branch: the leaf loop then skips the local
# stiffness block **and** the viscosity closure (both only feed `K`/`Kgeo`/
# `Kp`), so the assembler produces `B`, `fv` and the constraint-free geometry
# data without paying for the operator the caller never stores or forms.
#
# `geo_coef`/`dp_coef` are **type parameters**, not `Union{Nothing,Function}`
# fields.  A function *field* is not specialised (docs/dev/SOLVER.md §17: the
# `semquad_rho` bug), so `geo_coef(x,y,edot)` inferred as `Any` and the Kgeo
# accumulation `Kgloc[k,l] += gc * Tux_k * Dux_l * w` was a boxed multiplication
# inside the innermost `(k,l)` loop: measured 13.0 of the 15.0 ms of the Kgeo
# branch and 336 MB of the 831 MB allocated by 3 Newton steps (docs/
# NEWTON_AUDIT.md).  With the parameter the call is concrete, `gc::Float64` and
# the loop is monomorphic; the arithmetic (and therefore the result) is
# unchanged, G202 pins it bitwise against the pre-fix worktree.
struct SemQuadAsmCfg{G,D}
    a2::Float64
    deviatoric::Bool
    mf_path::Bool
    geo_u::Union{Nothing,Vector{Float64}}
    geo_coef::G
    geo_spd_safety::Float64
    dp_u::Union{Nothing,Vector{Float64}}
    dp_pfield::Union{Nothing,Vector{Float64}}
    dp_coef::D
end

function _semquad_asm_buf(nleaf::Int, nv::Int, npq::Int, Nu2::Int;
                          has_K::Bool, has_geo::Bool, has_dp::Bool)
    IK = Int[]; JK = Int[]; VK = Float64[]
    if has_K
        # Exact counts (4*nv^2 velocity entries per leaf); this is purely an
        # allocation hint, the push! order is unchanged.
        sizehint!(IK, 4 * nv * nv * nleaf)
        sizehint!(JK, 4 * nv * nv * nleaf)
        sizehint!(VK, 4 * nv * nv * nleaf)
    end
    IB = Int[]; JB = Int[]; VB = Float64[]
    sizehint!(IB, 2 * npq * nv * nleaf)
    sizehint!(JB, 2 * npq * nv * nleaf)
    sizehint!(VB, 2 * npq * nv * nleaf)
    IG = Int[]; JG = Int[]; VG = Float64[]
    if has_geo
        sizehint!(IG, 4 * nv * nv * nleaf)
        sizehint!(JG, 4 * nv * nv * nleaf)
        sizehint!(VG, 4 * nv * nv * nleaf)
    end
    IP = Int[]; JP = Int[]; VP = Float64[]
    if has_dp
        sizehint!(IP, 2 * npq * nv * nleaf)
        sizehint!(JP, 2 * npq * nv * nleaf)
        sizehint!(VP, 2 * npq * nv * nleaf)
    end
    n2v = 2 * nv
    scr = SemQuadAsmScratch(zeros(Float64, nv), zeros(Float64, nv),
                            zeros(Float64, nv),
                            zeros(Float64, n2v, n2v),
                            zeros(Float64, n2v, n2v),
                            zeros(Float64, npq, n2v),
                            zeros(Float64, n2v),
                            zeros(Float64, n2v, npq))
    return SemQuadAsmBuf(IK, JK, VK, IB, JB, VB, IG, JG, VG, IP, JP, VP,
                         zeros(Float64, Nu2), scr)
end

# Assemble leaves `lo:hi` into `buf` (the buffers are per block, so this is
# safe to run concurrently on disjoint leaf ranges).
function _semquad_assemble_block!(buf::SemQuadAsmBuf, cfg::SemQuadAsmCfg,
                                  su::SemQuad, sp::SemQuad,
                                  leaves::Vector{NTuple{3,Int}}, lo::Int, hi::Int,
                                  eta_fun::Function, fx::Function, fz::Function,
                                  fd::FEData, pleaves::Set{NTuple{3,Int}})
    hi < lo && return buf
    n = su.p + 1
    q = sp.p
    nv = n * n
    npq = (q + 1)^2
    nu = su.N
    a2 = cfg.a2
    deviatoric = cfg.deviatoric
    mf_path = cfg.mf_path
    geo_u = cfg.geo_u
    geo_coef = cfg.geo_coef
    geo_spd_safety = cfg.geo_spd_safety
    dp_u = cfg.dp_u
    dp_pfield = cfg.dp_pfield
    dp_coef = cfg.dp_coef
    IK = buf.IK; JK = buf.JK; VK = buf.VK
    IB = buf.IB; JB = buf.JB; VB = buf.VB
    IG = buf.IG; JG = buf.JG; VG = buf.VG
    IP = buf.IP; JP = buf.JP; VP = buf.VP
    fv = buf.fv
    scr = buf.scr
    phi = scr.phi; gx = scr.gx; gz = scr.gz
    Kloc = scr.Kloc; Kgloc = scr.Kgloc; Bloc = scr.Bloc; floc = scr.floc
    Kploc = scr.Kploc
    for it in lo:hi
        lf = leaves[it]
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        fill!(Kloc, 0.0); fill!(Kgloc, 0.0); fill!(Bloc, 0.0); fill!(floc, 0.0)
        if dp_u !== nothing
            fill!(Kploc, 0.0)
        end
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            # The viscosity enters the leaf assembly only through `K` and the
            # two Jacobian blocks, all of which are skipped on the matrix-free
            # path (`mf_path`): the operator's own eta field
            # (`semquad_eta_field`) samples the same points.  Evaluating the
            # closure here anyway cost 0.15 s of the F setup at one thread
            # (131 008 calls, measured; docs/dev/MG.md).  `etaq` stays
            # defined so the `K`-forming branches below compile unchanged.
            etaq = 0.0
            if !mf_path
                etaq = eta_fun(x, y)
            end
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
            end
            # Performance (ported from Route D): fx/fz are spatially varying but
            # constant over the element basis loop; evaluating them once per
            # quadrature point instead of 2*nv times cuts the closure calls from
            # 1.4M to 158k on the 8800-leaf tree (measured ~0.5 s -> ~0.05 s).
            # Identical values.
            fxv = fx(x, y)
            fzv = fz(x, y)
            for k in 1:nv
                floc[k] += w * fxv * phi[k]
                floc[nv + k] += w * fzv * phi[k]
            end
            # Local stiffness block.  Four independent accumulations over the
            # (k, l) pairs.  The *inner* loop index is `k`, so every
            # `Kloc[...]` store and every `gx[k]`/`gz[k]` load is unit-stride
            # (the transposed order used to store with stride 2*nv and could
            # not vectorise).  The quadrature loop stays outermost, so each
            # entry still accumulates over q in the same order: the result is
            # bitwise identical (G188), and the per-q scalars (`etaq*w`,
            # `a2`-scaled gradients) are re-associated only where the
            # left-to-right expression tree is unchanged.
            if !mf_path
                if deviatoric
                    for l in 1:nv
                        xl = gx[l]; zl = gz[l]
                        @simd for k in 1:nv
                            Kloc[k, l] += etaq * (a2 * gx[k] * xl + gz[k] * zl) * w
                            # full 2 eta eps:eps(v) minus (2/3) eta div(u)div(v);
                            # the off-diagonal correction is a DIFFERENT product
                            # (gx_k gz_l), not a rescaling of gz_k gx_l.
                            Kloc[k, nv + l] += etaq * (gz[k] * xl -
                                                       (2.0 / 3.0) * gx[k] * zl) * w
                            Kloc[nv + k, l] += etaq * (gx[k] * zl -
                                                       (2.0 / 3.0) * gz[k] * xl) * w
                            Kloc[nv + k, nv + l] += etaq * (a2 * gz[k] * zl + gx[k] * xl) * w
                        end
                    end
                else
                    for l in 1:nv
                        xl = gx[l]; zl = gz[l]
                        @simd for k in 1:nv
                            Kloc[k, l] += etaq * (a2 * gx[k] * xl + gz[k] * zl) * w
                            Kloc[k, nv + l] += etaq * (gz[k] * xl) * w
                            Kloc[nv + k, l] += etaq * (gx[k] * zl) * w
                            Kloc[nv + k, nv + l] += etaq * (a2 * gz[k] * zl + gx[k] * xl) * w
                        end
                    end
                end
            end
            if geo_u !== nothing
                # Geometric (strain-rate) part of the Newton Jacobian:
                #   Kgeo = sum_q w (deta/dedot)/edot  T(q) T(q)'
                # with T the symmetric strain-rate operator acting on the
                # velocity basis.  It is kept separate from Kloc so the
                # residual `fv - K*u0` still uses the true (non-geometric) K.
                duxdx = 0.0; duxdy = 0.0; duzdx = 0.0; duzdy = 0.0
                for k in 1:nv
                    duxdx += geo_u[uids[k]] * gx[k]
                    duxdy += geo_u[uids[k]] * gz[k]
                    duzdx += geo_u[nu + uids[k]] * gx[k]
                    duzdy += geo_u[nu + uids[k]] * gz[k]
                end
                exd = duxdx; ezd = duzdy
                exz = 0.5 * (duxdy + duzdx)
                edot = sqrt(max(0.5 * (exd * exd + ezd * ezd +
                                       2.0 * exz * exz), 0.0))
                deta_q = edot > 0.0 ? geo_coef(x, y, edot) : 0.0
                # ASPECT `Utilities::compute_spd_factor` (stabilization): with a
                # strongly weakening law (Drucker-Prager plastic branch has
                # deta/dedot = -eta/edot) the geometric term can cancel the
                # viscous modulus and leave the velocity block singular.
                # `geo_spd_safety > 0` scales Kgeo by alpha in (0,1] so that
                # 2 eta + alpha * (deviator) stays SPD; 0 keeps G156's
                # unstabilized operator bit for bit.
                if geo_spd_safety > 0.0 && deta_q != 0.0
                    d_a = spd_factor(etaq, deta_q, edot, exd, ezd, exz,
                                     exd, ezd, exz; safety=geo_spd_safety)
                else
                    d_a = 1.0
                end
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
            if dp_u !== nothing
                # Pressure-derivative block: Kp_{i,a} = int 2 (deta/dp)
                # eps(u):eps(v_i) phi_a.  The strain rate is the one that
                # enters the viscosity, i.e. of `dp_u` at this quadrature point.
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
                dp_p = _semquad_eval_at(sp, dp_pfield, x, y, pleaves)
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
            for b in 1:(q + 1), a in 1:(q + 1)
                l = (b - 1) * (q + 1) + a
                ps = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                for k in 1:nv
                    Bloc[l, k] += ps * gx[k] * w
                    Bloc[l, nv + k] += ps * gz[k] * w
                end
            end
        end
        for k in 1:nv
            g = uids[k]
            fv[g] += floc[k]
            fv[nu + g] += floc[nv + k]
            mf_path && continue
            for l in 1:nv
                gl = uids[l]
                push!(IK, g); push!(JK, gl); push!(VK, Kloc[k, l])
                push!(IK, nu + g); push!(JK, nu + gl); push!(VK, Kloc[nv + k, nv + l])
                push!(IK, g); push!(JK, nu + gl); push!(VK, Kloc[k, nv + l])
                push!(IK, nu + g); push!(JK, gl); push!(VK, Kloc[nv + k, l])
                if geo_u !== nothing
                    push!(IG, g); push!(JG, gl); push!(VG, Kgloc[k, l])
                    push!(IG, nu + g); push!(JG, nu + gl); push!(VG, Kgloc[nv + k, nv + l])
                    push!(IG, g); push!(JG, nu + gl); push!(VG, Kgloc[k, nv + l])
                    push!(IG, nu + g); push!(JG, gl); push!(VG, Kgloc[nv + k, l])
                end
            end
        end
        for l in 1:npq
            pg = pids[l]
            for k in 1:nv
                push!(IB, pg); push!(JB, uids[k]); push!(VB, Bloc[l, k])
                push!(IB, pg); push!(JB, nu + uids[k]); push!(VB, Bloc[l, nv + k])
            end
        end
        if dp_u !== nothing
            for l in 1:npq
                pg = pids[l]
                for k in 1:nv
                    push!(IP, uids[k]); push!(JP, pg); push!(VP, Kploc[k, l])
                    push!(IP, nu + uids[k]); push!(JP, pg)
                    push!(VP, Kploc[nv + k, l])
                end
            end
        end
    end
    return buf
end

# Split the leaves into `nt` contiguous blocks, assemble them (in parallel when
# Julia has more than one thread) and merge the block buffers in block order.
# `nt == 1` is the historical serial path: one buffer over all leaves, returned
# as-is (no copy, no re-ordering).
function _semquad_assemble_blocks(nt::Int, cfg::SemQuadAsmCfg,
                                  su::SemQuad, sp::SemQuad,
                                  eta_fun::Function, fx::Function, fz::Function,
                                  fd::FEData, pleaves::Set{NTuple{3,Int}})
    nleaf = length(su.leaves)
    n = su.p + 1
    nv = n * n
    npq = (sp.p + 1)^2
    Nu2 = 2 * su.N
    has_K = !cfg.mf_path
    has_geo = cfg.geo_u !== nothing
    has_dp = cfg.dp_u !== nothing
    if nt <= 1
        buf = _semquad_asm_buf(nleaf, nv, npq, Nu2; has_K=has_K,
                               has_geo=has_geo, has_dp=has_dp)
        _semquad_assemble_block!(buf, cfg, su, sp, su.leaves, 1, nleaf,
                                 eta_fun, fx, fz, fd, pleaves)
        return (IK=buf.IK, JK=buf.JK, VK=buf.VK,
                IB=buf.IB, JB=buf.JB, VB=buf.VB,
                IG=buf.IG, JG=buf.JG, VG=buf.VG,
                IP=buf.IP, JP=buf.JP, VP=buf.VP,
                fv=buf.fv)
    end
    bufs = Vector{SemQuadAsmBuf}(undef, nt)
    los = Vector{Int}(undef, nt)
    his = Vector{Int}(undef, nt)
    base, rem = divrem(nleaf, nt)
    off = 1
    for t in 1:nt
        cnt = base + (t <= rem ? 1 : 0)
        los[t] = off
        his[t] = off + cnt - 1
        off += cnt
        bufs[t] = _semquad_asm_buf(cnt, nv, npq, Nu2; has_K=has_K,
                                   has_geo=has_geo, has_dp=has_dp)
    end
    Threads.@threads :static for t in 1:nt
        _semquad_assemble_block!(bufs[t], cfg, su, sp, su.leaves, los[t],
                                 his[t], eta_fun, fx, fz, fd, pleaves)
    end
    fv = zeros(Float64, Nu2)
    for t in 1:nt
        fv .+= bufs[t].fv
    end
    IK = has_K ? _semquad_concat_blocks([b.IK for b in bufs]) : Int[]
    JK = has_K ? _semquad_concat_blocks([b.JK for b in bufs]) : Int[]
    VK = has_K ? _semquad_concat_blocks([b.VK for b in bufs]) : Float64[]
    IB = _semquad_concat_blocks([b.IB for b in bufs])
    JB = _semquad_concat_blocks([b.JB for b in bufs])
    VB = _semquad_concat_blocks([b.VB for b in bufs])
    IG = has_geo ? _semquad_concat_blocks([b.IG for b in bufs]) : Int[]
    JG = has_geo ? _semquad_concat_blocks([b.JG for b in bufs]) : Int[]
    VG = has_geo ? _semquad_concat_blocks([b.VG for b in bufs]) : Float64[]
    IP = has_dp ? _semquad_concat_blocks([b.IP for b in bufs]) : Int[]
    JP = has_dp ? _semquad_concat_blocks([b.JP for b in bufs]) : Int[]
    VP = has_dp ? _semquad_concat_blocks([b.VP for b in bufs]) : Float64[]
    return (IK=IK, JK=JK, VK=VK, IB=IB, JB=JB, VB=VB,
            IG=IG, JG=JG, VG=VG, IP=IP, JP=JP, VP=VP, fv=fv)
end

