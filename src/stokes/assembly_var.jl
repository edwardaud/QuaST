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
#
# `elastic` is the same kind of type parameter: `nothing` (the default and every
# `shear_modulus = 0` run) makes the 2b elastic-force branch below compile out
# entirely, so the historical load vector is untouched bit for bit; when it is a
# 3-tuple of nodal Q2 stress components the branch is monomorphic too.
struct SemQuadAsmCfg{G,D,E}
    a2::Float64
    deviatoric::Bool
    mf_path::Bool
    geo_u::Union{Nothing,Vector{Float64}}
    geo_coef::G
    geo_spd_safety::Float64
    dp_u::Union{Nothing,Vector{Float64}}
    dp_pfield::Union{Nothing,Vector{Float64}}
    dp_coef::D
    elastic::E
end

# The historical 9-argument positional form (G181/G188/G202 build it): no
# elastic force term, i.e. the pre-2b-coupling assembly.
SemQuadAsmCfg(a2::Float64, deviatoric::Bool, mf_path::Bool,
              geo_u, geo_coef, geo_spd_safety::Float64, dp_u, dp_pfield,
              dp_coef) =
    SemQuadAsmCfg(a2, deviatoric, mf_path, geo_u, geo_coef, geo_spd_safety,
                  dp_u, dp_pfield, dp_coef, nothing)

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
    elastic = cfg.elastic
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
            # Performance (ported from an earlier revision): fx/fz are spatially varying but
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
            # 2b step 2: the elastic force term, `-int tau : eps(v)` with the
            # nodal previous-step stress field (`a tau^n`, scaled by the driver).
            # It reuses this quadrature loop and the same velocity basis the
            # stiffness above uses: `tau` is evaluated at the Gauss point with
            # the Q2 interpolation and contracted with the basis gradients,
            #     tau : eps(v) = tau_xx dv_x/dx + tau_yy dv_y/dy
            #                    + tau_xy (dv_x/dy + dv_y/dx),
            # so the x-test row is `-(tau_xx gx_k + tau_xy gz_k)` and the y row
            # `-(tau_yy gz_k + tau_xy gx_k)`.  A uniform stress therefore
            # produces no load (`div tau = 0`: integration by parts makes it a
            # boundary term, and the test functions vanish on the Dirichlet
            # boundary) -- G247 checks that, and the body-force equivalence that
            # fixes the sign.  `elastic === nothing` compiles this out.
            if elastic !== nothing
                txx_q = 0.0; tyy_q = 0.0; txy_q = 0.0
                @inbounds for k in 1:nv
                    p_k = phi[k]
                    txx_q += elastic[1][uids[k]] * p_k
                    tyy_q += elastic[2][uids[k]] * p_k
                    txy_q += elastic[3][uids[k]] * p_k
                end
                @inbounds for k in 1:nv
                    floc[k] -= w * (txx_q * gx[k] + txy_q * gz[k])
                    floc[nv + k] -= w * (tyy_q * gz[k] + txy_q * gx[k])
                end
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

# ---------------------------------------------------------------------------
# Cut 2: the **mapped** leaf kernel.
#
# This is a separate body, not a reroute of the box one.  The box body above is
# the historical arithmetic character for character (`2.0/hx` folded into the
# gradient, `0.25*hx*hy*gl_w`, Cartesian point map) and it is what keeps the
# nine state-dump hashes; GEOM_DESIGN §4.1/§4.3 forbid routing it through
# `J = diag(hx/2, hy/2)` (F1) or re-associating the weight tree (F3).  The mapped
# body below gets the same three quantities from the geometry metric instead:
#
#   (x, y)    = quad_point(lm, su, lf, fd, qx, qy)      # element_map at the
#                                                        # Gauss point (cached)
#   w         = quad_measure(lm, fd, qx, qy)            # detJ * gl_w * gl_w
#   (gx, gz)  = basis_grad(lm, fd, qx, qy, i, j)        # J^{-T} chain rule
#
# and everything downstream — the viscosity closure, the four stiffness
# accumulations, the deviatoric correction, the elastic/geometric branches, the
# B block, the load — is the same expression tree as the box kernel, which is
# why the two kernels agree to round-off when the mapped metric is fed the box
# map (G268).  `metrics` is the precomputed per-leaf metric vector (built once
# per assembly call, `leaf_metric(g, su, lf, fd)`), so the quadrature loop only
# indexes flat `Vector{Float64}`s (GEOM_DESIGN §2.5: no map evaluation in the
# loop).
#
# Features that are not routed through the metric in cut 2 (the strain-rate
# Jacobian `Kgeo`, the pressure-derivative block `Kp`, the 2b elastic load, the
# density-multiplier finite differences) are **refused** for a mapped geometry in
# `_solve_semquad_stokes_var_core`; they stay box-only until a cut gates them.
# ---------------------------------------------------------------------------
function _semquad_assemble_block_mapped!(buf::SemQuadAsmBuf, cfg::SemQuadAsmCfg,
                                         metrics::Vector{MappedLeafMetric},
                                         su::SemQuad, sp::SemQuad,
                                         leaves::Vector{NTuple{3,Int}},
                                         lo::Int, hi::Int,
                                         eta_fun::Function, fx::Function,
                                         fz::Function, fd::FEData)
    hi < lo && return buf
    n = su.p + 1
    q = sp.p
    nv = n * n
    npq = (q + 1)^2
    nu = su.N
    a2 = cfg.a2
    deviatoric = cfg.deviatoric
    mf_path = cfg.mf_path
    IK = buf.IK; JK = buf.JK; VK = buf.VK
    IB = buf.IB; JB = buf.JB; VB = buf.VB
    fv = buf.fv
    scr = buf.scr
    phi = scr.phi; gx = scr.gx; gz = scr.gz
    Kloc = scr.Kloc; Bloc = scr.Bloc; floc = scr.floc
    for it in lo:hi
        lf = leaves[it]
        lm = metrics[it]
        uids = su.elem[lf]
        pids = sp.elem[lf]
        fill!(Kloc, 0.0); fill!(Bloc, 0.0); fill!(floc, 0.0)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(lm, su, lf, fd, qx, qy)
            w = quad_measure(lm, fd, qx, qy)
            etaq = 0.0
            if !mf_path
                etaq = eta_fun(x, y)
            end
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k], gz[k] = basis_grad(lm, fd, qx, qy, i, j)
            end
            fxv = fx(x, y)
            fzv = fz(x, y)
            for k in 1:nv
                floc[k] += w * fxv * phi[k]
                floc[nv + k] += w * fzv * phi[k]
            end
            if !mf_path
                if deviatoric
                    for l in 1:nv
                        xl = gx[l]; zl = gz[l]
                        @simd for k in 1:nv
                            Kloc[k, l] += etaq * (a2 * gx[k] * xl + gz[k] * zl) * w
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
            end
        end
        for l in 1:npq
            pg = pids[l]
            for k in 1:nv
                push!(IB, pg); push!(JB, uids[k]); push!(VB, Bloc[l, k])
                push!(IB, pg); push!(JB, nu + uids[k]); push!(VB, Bloc[l, nv + k])
            end
        end
    end
    return buf
end

# Split the leaves into `nt` contiguous blocks, assemble them (in parallel when
# Julia has more than one thread) and merge the block buffers in block order.
# `nt == 1` is the historical serial path: one buffer over all leaves, returned
# as-is (no copy, no re-ordering).
"""
    _semquad_assemble_blocks(nt, cfg, su, sp, eta_fun, fx, fz, fd, pleaves)

Assemble the Stokes leaf blocks and merge them.  This entry point dispatches on
the geometry the space carries (`su.geom`) and forwards a **concrete** geometry
to the geometry-typed method below (GEOM_DESIGN §2.5: one dispatch at the
entry, no `sq.geom` read inside a kernel loop).  The box method is the
historical body, character for character; the mapped method precomputes the
leaf metrics and runs `_semquad_assemble_block_mapped!`.
"""
function _semquad_assemble_blocks(nt::Int, cfg::SemQuadAsmCfg,
                                  su::SemQuad, sp::SemQuad,
                                  eta_fun::Function, fx::Function, fz::Function,
                                  fd::FEData, pleaves::Set{NTuple{3,Int}})
    return _semquad_assemble_blocks(su.geom, nt, cfg, su, sp, eta_fun, fx, fz,
                                    fd, pleaves)
end

# Historical box body, verbatim (`su.geom` is a CartesianBox; the geometry
# argument is not read by any arithmetic in this method).
function _semquad_assemble_blocks(::CartesianBox, nt::Int, cfg::SemQuadAsmCfg,
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

# Mapped body of the block assembler: the leaf metric is built once per leaf
# (serial, cold) and the leaf blocks only read it, so no map evaluation happens
# in the quadrature loops.  `Kgeo`/`Kp`/`elastic` are box-only in cut 2 and are
# refused before the assembly (see `_solve_semquad_stokes_var_core`), so the
# buffers for them are not created here.
function _semquad_assemble_blocks(g::AbstractGeometry, nt::Int,
                                  cfg::SemQuadAsmCfg,
                                  su::SemQuad, sp::SemQuad,
                                  eta_fun::Function, fx::Function, fz::Function,
                                  fd::FEData, pleaves::Set{NTuple{3,Int}})
    (cfg.geo_u === nothing && cfg.dp_u === nothing && cfg.elastic === nothing) ||
        throw(ArgumentError("the mapped Stokes leaf assembly does not implement " *
                            "the strain-rate Jacobian (geo_u), the " *
                            "pressure-derivative block (dp_u) or the elastic " *
                            "force (elastic_stress) in cut 2; use a " *
                            "CartesianBox geometry for those, or drop the " *
                            "keyword."))
    nleaf = length(su.leaves)
    n = su.p + 1
    nv = n * n
    npq = (sp.p + 1)^2
    Nu2 = 2 * su.N
    has_K = !cfg.mf_path
    metrics = Vector{MappedLeafMetric}(undef, nleaf)
    @inbounds for it in 1:nleaf
        metrics[it] = leaf_metric(g, su, su.leaves[it], fd)
    end
    if nt <= 1
        buf = _semquad_asm_buf(nleaf, nv, npq, Nu2; has_K=has_K,
                               has_geo=false, has_dp=false)
        _semquad_assemble_block_mapped!(buf, cfg, metrics, su, sp, su.leaves,
                                        1, nleaf, eta_fun, fx, fz, fd)
        return (IK=buf.IK, JK=buf.JK, VK=buf.VK,
                IB=buf.IB, JB=buf.JB, VB=buf.VB,
                IG=Int[], JG=Int[], VG=Float64[],
                IP=Int[], JP=Int[], VP=Float64[],
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
                                   has_geo=false, has_dp=false)
    end
    Threads.@threads :static for t in 1:nt
        _semquad_assemble_block_mapped!(bufs[t], cfg, metrics, su, sp, su.leaves,
                                        los[t], his[t], eta_fun, fx, fz, fd)
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
    return (IK=IK, JK=JK, VK=VK, IB=IB, JB=JB, VB=VB,
            IG=Int[], JG=Int[], VG=Float64[],
            IP=Int[], JP=Int[], VP=Float64[], fv=fv)
end

# ---------------------------------------------------------------------------
# Boundary-condition feature, stage 2/3: the prescribed **traction** as a
# boundary-face load (G259/G260 tangential, G261-G263 normal/open).
#
# The Stokes assembly had no boundary-face integral at all: the natural
# (do-nothing) condition of the weak form is `sigma.n = 0`, so every
# unconstrained boundary component was silently traction-free
# (docs/dev/BC_DESIGN.md §1.1).  Integration by parts of `-div sigma = f` with
# `sigma = -p I + 2 eta eps(u)` gives
#
#     int 2 eta eps(u):eps(v) - int p div v = int f.v + int_{dOmega} (sigma.n).v dS,
#
# and the code solves exactly the first two terms (`K u - B'p = fv`).  A
# prescribed traction is therefore an **RHS** term with a **plus** sign,
#
#     fv_i += int_{Gamma_t} tbar . phi_i dS,     tbar = prescribed sigma.n,
#
# where `n` is the outward normal and `phi_i` the global velocity basis
# function.  The load is added to the *unreduced* `fv` (this file has no access
# to the constraints) and the solve forms `fr = Cu' (fv - K u0 + Bm' p0)`; the
# constraint matrix is then the exact virtual-work projection onto the
# admissible test space (a hanging master receives its share, a Dirichlet dof
# has no row and receives nothing -- see `_semquad_validate_wall_traction`).
#
# Stage 2 added the **tangential** component, only where the wall's tangential
# component is free (`:free`).  Stage 3 adds the **normal** component on an
# `:open` wall, where *both* components are free (no Dirichlet constraint), so
# `tbar = sigma.n` may be prescribed on the whole wall.  For an axis-aligned
# wall the two components of the total Cauchy traction are
#
#     y-wall:  t_x = sigma_xy  (pressure-free, tangential)
#              t_y = sigma_yy = -p + 2 eta eps_yy   (carries the pressure)
#     x-wall:  t_z = sigma_zy  (pressure-free, tangential)
#              t_x = sigma_xx = -p + 2 eta eps_xx   (carries the pressure)
#
# so the normal part is the data term that fixes the pressure level on an open
# boundary (`wall_tn`), and the tangential part is the same shear traction as
# stage 2 (`wall_t`).  The normal load is the *only* piece that differs from
# stage 2: it is added to the rows of the wall's normal component, `u_z` on a
# y-wall and `u_x` on an x-wall.
#
# The data convention is the **global component in the positive coordinate
# direction**, the same one the legacy Dirichlet closures use: `wall_t` is the
# global `t_z` on an x-wall and `t_x` on a y-wall; `wall_tn` is the global `t_x`
# on an x-wall and `t_z` on a y-wall.  The map to the `(t_x, t_z)` pair is
# therefore a pure assignment with **no outward-normal sign table** (§3.2 eq. 7):
# the physical normal sign is already contained in `tbar = sigma.n`, and a
# user-facing sign table would duplicate it and be the easiest thing in the
# feature to get wrong.  Consequence: because the traction uses the *outward*
# normal, a pressure `p0` imposed on an open boundary is `wall_tn = -p0` on the
# top wall and `wall_tn = +p0` on the bottom wall.
#
# The integral needs no new quadrature data: `fd` already holds the 1-D GLL
# velocity basis at `nq = p+2` Gauss points (`fd.phi_v`) and the Gauss rule
# (`fd.gl_x`, `fd.gl_w`), and the physical face Jacobian of a leaf face is
# `h/2`.  The weight per point is `(h/2)*gl_w[q]`; the integrand is a
# polynomial of degree `deg(tbar) + p`, so polynomial data up to the velocity
# degree is integrated exactly (the exactness of §3.2), which is what makes the
# G259/G261 gates exact to solver round-off.
# ---------------------------------------------------------------------------

"""
    _semquad_boundary_face_mask(qt, su) -> Vector{UInt8}

Serially precomputed domain-boundary flag for every leaf face: entry
`4*(it-1) + _wall_index(dir)` is `0x01` when the face of `su.leaves[it]` in
direction `dir` has no neighbour, `0x00` otherwise.

The mask exists because the quadtree neighbour table is **not thread-safe**
(its lazy construction writes the table): the leaf assembly must not query
`has_no_face_neighbor`/`neighbors` from `Threads.@threads`
(docs/dev/BC_DESIGN.md §3.3 item 1).  It is built once, serially, before the
threaded leaf assembly and reused by the fixed-dof loop, the net-flux check and
`_semquad_boundary_traction!`, so those passes never touch the table.  The
per-face booleans are exactly what those passes asked the table for before the
mask existed, so the change is a no-op for every quantity.
"""
function _semquad_boundary_face_mask(qt::Quadtree, su::SemQuad)
    leaves = su.leaves
    bmask = Vector{UInt8}(undef, 4 * length(leaves))
    @inbounds for it in eachindex(leaves)
        lf = leaves[it]
        b = 4 * (it - 1)
        bmask[b + 1] = has_no_face_neighbor(qt, lf, :xminus) ? 0x01 : 0x00
        bmask[b + 2] = has_no_face_neighbor(qt, lf, :xplus) ? 0x01 : 0x00
        bmask[b + 3] = has_no_face_neighbor(qt, lf, :yminus) ? 0x01 : 0x00
        bmask[b + 4] = has_no_face_neighbor(qt, lf, :yplus) ? 0x01 : 0x00
    end
    return bmask
end

"""
    _semquad_has_wall_t(wall_t) -> Bool

`true` when at least one wall carries a tangential-traction closure.  When all
four entries are `nothing` the load pass is skipped before it touches `fv`, so
the default path is bit-identical (G260).
"""
_semquad_has_wall_t(wall_t) = any(t -> t !== nothing, wall_t)

"""
    _semquad_validate_wall_traction(wall_t, resolved_wall_bc) -> Bool

Validate the per-wall tangential-traction data against the *resolved* wall
conditions (the four-tuple already has `:same` replaced by `bc`) and return
[`_semquad_has_wall_t`](@ref).

`wall_t` is a data term on the **free side**, so it only exists where the
tangential component is free: a `:free` wall (stage 2) or an `:open` wall
(stage 3, both components free).  A tangential traction on a `:rigid` wall
targets a Dirichlet component; the load would be assembled into `fv` but `Cu'`
would drop it, i.e. it would be a silent no-op.  The design rejects that
combination rather than letting a user believe a load was applied
(docs/dev/BC_DESIGN.md §3.2/§3.6), and the same validation runs in `Params` and
in `_solve_semquad_stokes_var_core` so a direct library call cannot bypass it.
Note that this is a *wall-level* rule: a face-integrated load also distributes
to the face's corner nodes, whose dof may be owned by an adjacent wall.  That
part is not an error -- the test function associated with a Dirichlet dof is
not in the admissible space, so `Cu'` dropping its contribution is exactly the
weak-form restriction (every tangential-traction configuration on a closed box
has such corners, and the exact Couette data agree there; G259).
"""
function _semquad_validate_wall_traction(wall_t, resolved_wall_bc)
    names = (:xminus, :xplus, :yminus, :yplus)
    for i in 1:4
        wall_t[i] === nothing && continue
        wbc = resolved_wall_bc[i]
        (wbc === :free || wbc === :open) || throw(ArgumentError(
            "wall_t[$i] (wall $(names[i])) is a tangential *traction*, but " *
            "that wall resolves to :$wbc, whose tangential " *
            "component is Dirichlet.  The boundary load would be assembled " *
            "onto a dof with no equation and silently dropped, so the request " *
            "is refused instead of being a no-op.  A tangential traction needs " *
            "a wall with a free tangential component (wall_bc[$i] = :free or " *
            ":open, or `bc = :free`); set that, or drop wall_t[$i]."))
    end
    return _semquad_has_wall_t(wall_t)
end

"""
    _semquad_has_wall_tn(wall_tn) -> Bool

`true` when at least one wall carries a normal-traction closure.  When all four
entries are `nothing` the normal part of the load pass is skipped, so a
tangential-only or default configuration is bit-identical (`fv` never sees the
normal branch).
"""
_semquad_has_wall_tn(wall_tn) = any(t -> t !== nothing, wall_tn)

"""
    _semquad_validate_wall_tnormal(wall_tn, resolved_wall_bc) -> Bool

Validate the per-wall *normal*-traction data against the resolved wall
conditions and return [`_semquad_has_wall_tn`](@ref).

The normal component of the total Cauchy traction `tbar = sigma.n` carries the
pressure (`t_n = -p + 2 eta eps_nn`), so prescribing it needs a **free normal
velocity**: exactly the `:open` wall, whose two components are unconstrained
(natural traction).  On a `:free`/`:rigid` wall the normal velocity is Dirichlet,
so the assembled load would land on a dof with no equation and be silently
dropped by `Cu'` -- refused here (and identically in `Params`), never ignored.
This is also what makes `wall_tn` the pressure data: on an open boundary the
traction level fixes the pressure constant, which is why the solve's pressure
gauge becomes derived (no nodal pin, docs/dev/BC_DESIGN.md §3.5).
"""
function _semquad_validate_wall_tnormal(wall_tn, resolved_wall_bc)
    names = (:xminus, :xplus, :yminus, :yplus)
    for i in 1:4
        wall_tn[i] === nothing && continue
        wbc = resolved_wall_bc[i]
        wbc === :open && continue
        throw(ArgumentError("wall_tn[$i] (wall $(names[i])) is a *normal " *
            "traction*, but that wall resolves to :$wbc, whose normal " *
            "component is Dirichlet.  The boundary load would be assembled onto " *
            "a dof with no equation and silently dropped, so the request is " *
            "refused instead of being a no-op.  A normal traction carries the " *
            "pressure and needs a wall with a free normal velocity " *
            "(wall_bc[$i] = :open; an :open wall has no Dirichlet component " *
            "and its pressure gauge becomes derived, docs/dev/BC_DESIGN.md " *
            "§3.5); set that, or drop wall_tn[$i]."))
    end
    return _semquad_has_wall_tn(wall_tn)
end

"""
    _semquad_boundary_traction!(fv, su, bmask, fd, wall_t, wall_tn, resolved_wall_bc)

Add the prescribed-traction load `int tbar.phi_i dS` to the **unreduced**
velocity load vector `fv` (length `2N`), in place.  `wall_t` is the tangential
component, `wall_tn` the normal component of the total Cauchy traction
`tbar = sigma.n` (both as global components in the positive coordinate
direction; see the file header for the sign/convention).

For every leaf face flagged by `bmask` and every Gauss point of the face, with
face weight `(h/2) * fd.gl_w[q] * fd.phi_v[k,q]` and **no** outward-normal sign:

* y-faces (`yminus`/`yplus`, physical face length `hx`):
  `wall_t` is the global `t_x` -> the `u_x` rows `g`;
  `wall_tn` is the global `t_z` -> the `u_z` rows `nu + g`;
* x-faces (`xminus`/`xplus`, physical face length `hy`):
  `wall_t` is the global `t_z` -> the `u_z` rows `nu + g`;
  `wall_tn` is the global `t_x` -> the `u_x` rows `g`.

The tangential part is exactly stage 2's pass (same accumulation order, bitwise
for a `wall_t`-only configuration).  The pass is serial and iterates the leaves
and the four directions in a fixed order, so the result is deterministic and
does not depend on the leaf assembly's thread count.  It is a bitwise no-op
when every entry of both tuples is `nothing`; a zero closure adds `+-0.0`,
which leaves `fv` bitwise unchanged as well (G260).

**Mapped geometry (stage 0 cut 2).**  The method above dispatches on
`su.geom`; the `CartesianBox` method is exactly the historical body documented
here, while the mapped method reads the data as the **wall-local** components
of the total traction `tbar = sigma.n` (GEOM_DESIGN §6.3): `wall_tn` is `t.n`
with the mapped outward unit normal and `wall_t` is `t.tau` with the positive
logical tangent, so the load is `int (wall_tn*n + wall_t*tau).phi dS`.  The two
conventions agree on the box map up to the outward sign that the box convention
absorbs (`wall_tn_box = +-t.n` on the four walls, `wall_t_box = t.tau`); G268
wires that sign explicitly and compares the loads.
"""
function _semquad_boundary_traction!(fv::Vector{Float64}, su::SemQuad,
                                     bmask::Vector{UInt8}, fd::FEData,
                                     wall_t::NTuple{4,Union{Nothing,Function}},
                                     wall_tn::NTuple{4,Union{Nothing,Function}},
                                     resolved_wall_bc)
    return _semquad_boundary_traction!(fv, su.geom, su, bmask, fd, wall_t,
                                       wall_tn, resolved_wall_bc)
end

# Historical box body, verbatim (global-component convention, no normal sign):
function _semquad_boundary_traction!(fv::Vector{Float64}, ::CartesianBox,
                                     su::SemQuad,
                                     bmask::Vector{UInt8}, fd::FEData,
                                     wall_t::NTuple{4,Union{Nothing,Function}},
                                     wall_tn::NTuple{4,Union{Nothing,Function}},
                                     resolved_wall_bc)
    has_t = _semquad_validate_wall_traction(wall_t, resolved_wall_bc)
    has_tn = _semquad_validate_wall_tnormal(wall_tn, resolved_wall_bc)
    (has_t || has_tn) || return fv
    n = su.p + 1
    nu = su.N
    gl_x = fd.gl_x
    gl_w = fd.gl_w
    phi = fd.phi_v
    nq = fd.nq
    leaves = su.leaves
    @inbounds for it in eachindex(leaves)
        b = 4 * (it - 1)
        bxm = bmask[b + 1] != 0x00
        bxp = bmask[b + 2] != 0x00
        bym = bmask[b + 3] != 0x00
        byp = bmask[b + 4] != 0x00
        (bxm | bxp | bym | byp) || continue
        lf = leaves[it]
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        # x-walls: the tangential direction is z (u_z rows); the normal
        # direction is x (u_x rows).
        if bxm
            if has_t && wall_t[1] !== nothing
                t_τ = wall_t[1]
                for q in 1:nq
                    yq = y0 + 0.5 * (gl_x[q] + 1.0) * hy
                    wt = 0.5 * hy * gl_w[q] * t_τ(x0, yq)
                    for j in 1:n
                        fv[nu + uids[(j - 1) * n + 1]] += wt * phi[j, q]
                    end
                end
            end
            if has_tn && wall_tn[1] !== nothing
                t_n = wall_tn[1]
                for q in 1:nq
                    yq = y0 + 0.5 * (gl_x[q] + 1.0) * hy
                    wt = 0.5 * hy * gl_w[q] * t_n(x0, yq)
                    for j in 1:n
                        fv[uids[(j - 1) * n + 1]] += wt * phi[j, q]
                    end
                end
            end
        end
        if bxp
            xf = x0 + hx
            if has_t && wall_t[2] !== nothing
                t_τ = wall_t[2]
                for q in 1:nq
                    yq = y0 + 0.5 * (gl_x[q] + 1.0) * hy
                    wt = 0.5 * hy * gl_w[q] * t_τ(xf, yq)
                    for j in 1:n
                        fv[nu + uids[(j - 1) * n + n]] += wt * phi[j, q]
                    end
                end
            end
            if has_tn && wall_tn[2] !== nothing
                t_n = wall_tn[2]
                for q in 1:nq
                    yq = y0 + 0.5 * (gl_x[q] + 1.0) * hy
                    wt = 0.5 * hy * gl_w[q] * t_n(xf, yq)
                    for j in 1:n
                        fv[uids[(j - 1) * n + n]] += wt * phi[j, q]
                    end
                end
            end
        end
        # y-walls: the tangential direction is x (u_x rows); the normal
        # direction is z (u_z rows).
        if bym
            if has_t && wall_t[3] !== nothing
                t_τ = wall_t[3]
                for q in 1:nq
                    xq = x0 + 0.5 * (gl_x[q] + 1.0) * hx
                    wt = 0.5 * hx * gl_w[q] * t_τ(xq, y0)
                    for i in 1:n
                        fv[uids[i]] += wt * phi[i, q]
                    end
                end
            end
            if has_tn && wall_tn[3] !== nothing
                t_n = wall_tn[3]
                for q in 1:nq
                    xq = x0 + 0.5 * (gl_x[q] + 1.0) * hx
                    wt = 0.5 * hx * gl_w[q] * t_n(xq, y0)
                    for i in 1:n
                        fv[nu + uids[i]] += wt * phi[i, q]
                    end
                end
            end
        end
        if byp
            yf = y0 + hy
            if has_t && wall_t[4] !== nothing
                t_τ = wall_t[4]
                for q in 1:nq
                    xq = x0 + 0.5 * (gl_x[q] + 1.0) * hx
                    wt = 0.5 * hx * gl_w[q] * t_τ(xq, yf)
                    for i in 1:n
                        fv[uids[(n - 1) * n + i]] += wt * phi[i, q]
                    end
                end
            end
            if has_tn && wall_tn[4] !== nothing
                t_n = wall_tn[4]
                for q in 1:nq
                    xq = x0 + 0.5 * (gl_x[q] + 1.0) * hx
                    wt = 0.5 * hx * gl_w[q] * t_n(xq, yf)
                    for i in 1:n
                        fv[nu + uids[(n - 1) * n + i]] += wt * phi[i, q]
                    end
                end
            end
        end
    end
    return fv
end

# Mapped body of the traction load.  The two data functions are the wall-local
# components of the **total** Cauchy traction `tbar = sigma.n` (GEOM_DESIGN
# §6.3): `wall_tn` is `t.n` (outward unit normal) and `wall_t` is `t.tau`
# (positive logical tangent), so the load in the two global velocity components
# is `int (wall_tn*n + wall_t*tau) . phi dS` and there is no hand-written sign
# table: the outward direction is the mapped unit normal.  The face weight is
# the mapped face Jacobian times the Gauss weight.
#
# The box data convention is the *global component in the positive coordinate
# direction* (see the box body and BC_DESIGN §3.2 eq. 7), because a box wall is
# axis aligned and the outward sign is already in `tbar = sigma.n`.  On the box
# map the two conventions differ only by that sign on `xminus`/`yminus`:
# `wall_tn_box = +-t.n`, `wall_t_box = t.tau`, which is what G268's traction
# comparison wires explicitly.
function _semquad_boundary_traction!(fv::Vector{Float64}, g::AbstractGeometry,
                                     su::SemQuad, bmask::Vector{UInt8},
                                     fd::FEData,
                                     wall_t::NTuple{4,Union{Nothing,Function}},
                                     wall_tn::NTuple{4,Union{Nothing,Function}},
                                     resolved_wall_bc)
    has_t = _semquad_validate_wall_traction(wall_t, resolved_wall_bc)
    has_tn = _semquad_validate_wall_tnormal(wall_tn, resolved_wall_bc)
    (has_t || has_tn) || return fv
    n = su.p + 1
    nu = su.N
    phi = fd.phi_v
    leaves = su.leaves
    @inbounds for it in eachindex(leaves)
        b = 4 * (it - 1)
        (bmask[b + 1] | bmask[b + 2] | bmask[b + 3] | bmask[b + 4]) != 0x00 ||
            continue
        lf = leaves[it]
        uids = su.elem[lf]
        for (s, side) in enumerate(LOGICAL_SIDES)
            bmask[b + s] != 0x00 || continue
            tfun = has_t ? wall_t[s] : nothing
            nfun = has_tn ? wall_tn[s] : nothing
            (tfun === nothing && nfun === nothing) && continue
            fm = face_metric(g, su, lf, side, fd)
            for q in 1:fd.nq
                xq, yq = face_point(fm, q)
                wt = face_weight(fm, fd, q)
                fnx = 0.0
                fny = 0.0
                if tfun !== nothing
                    tv = tfun(xq, yq)
                    tx_, ty_ = face_tangent(fm, q)
                    fnx += tv * tx_
                    fny += tv * ty_
                end
                if nfun !== nothing
                    nv_ = nfun(xq, yq)
                    nx_, ny_ = face_normal(fm, q)
                    fnx += nv_ * nx_
                    fny += nv_ * ny_
                end
                for j in 1:n
                    node = if side === :xminus
                        uids[(j - 1) * n + 1]
                    elseif side === :xplus
                        uids[(j - 1) * n + n]
                    elseif side === :yminus
                        uids[j]
                    else
                        uids[(n - 1) * n + j]
                    end
                    pj = wt * phi[j, q]
                    fv[node] += pj * fnx
                    fv[nu + node] += pj * fny
                end
            end
        end
    end
    return fv
end

