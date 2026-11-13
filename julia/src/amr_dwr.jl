# Goal-oriented (DWR) AMR kernel, experimental.  Kept separate from the
# legacy AMR path.  The adjoint solve is not wired in yet; this file provides
# the goal functional and the residual*adjoint leaf indicator so the next
# step can plug in a Stokes/energy adjoint without touching amr_cycle!.

"""
    goal_upper_half(sq; z0=0.5)

Nodal weight for the goal functional J = ∫_{z>z0} T dV (a smooth upper-half
heat-content goal).  Use as the RHS of the adjoint energy equation.
"""
function goal_upper_half(sq::SemQuad; z0::Float64=0.5, sharpness::Float64=20.0)
    return [0.5 * (1.0 + tanh(sharpness * (y - z0))) for (_, y) in sq.coords]
end

"""
    dwr_leaf_indicator(sq, qt, residual, adjoint)

Aggregate the dual-weighted residual `|r_g * z_g|` over the nodes of each
leaf.  `residual` and `adjoint` are nodal vectors on `sq`; the result is a
leaf => nonnegative indicator dictionary suitable for Dörfler marking.
This is the DWR analogue of `semquad_gradient_jump`.
"""
function dwr_leaf_indicator(sq::SemQuad, qt::Quadtree,
                            residual::AbstractVector{<:Real},
                            adjoint::AbstractVector{<:Real})
    length(residual) == sq.N ||
        throw(ArgumentError("residual length $(length(residual)) != N $(sq.N)"))
    length(adjoint) == sq.N ||
        throw(ArgumentError("adjoint length $(length(adjoint)) != N $(sq.N)"))
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves
        s = 0.0
        for g in sq.elem[lf]
            s += abs(residual[g] * adjoint[g])
        end
        eta[lf] = s
    end
    return eta
end
"""
    solve_stokes_adjoint_system(K, B, Cu, Cp, gu, gp)

Solve the reduced transpose saddle system
    [Kred' Bs'; Bs 0] [zf; qf] = [Cu'gu; Cp'gp]
with Kred=Cu'KCu, Bs=-Cp'B Cu, and return full (z,q).
"""
function solve_stokes_adjoint_system(K, B, Cu, Cp,
                                     gu::AbstractVector{<:Real},
                                     gp::AbstractVector{<:Real})
    Kred = Cu' * K * Cu
    Bs = -Cp' * B * Cu
    nf = size(Cu, 2); nfp = size(Cp, 2)
    S = [Kred' Bs'; Bs spzeros(nfp, nfp)]
    rhs = vcat(Cu' * gu, Cp' * gp)
    sol = S \ rhs
    zf = sol[1:nf]; qf = sol[nf+1:end]
    return Cu * zf, Cp * qf
end
"""
    stokes_residual(K, B, u, p, fv)

Full residual of the saddle system: r_u = fv - K u - B' p, r_p = -B u.
`u` has length 2N, `p` length Np.
"""
function stokes_residual(K, B, u::AbstractVector{<:Real},
                         p::AbstractVector{<:Real},
                         fv::AbstractVector{<:Real})
    ru = fv .- K * u .- B' * p
    rp = .-B * u
    return ru, rp
end

"""
    stokes_dwr_indicator(sq, qt, ru, rp, z, q)

DWR leaf indicator |ru·z| + |rp·q| aggregated over velocity/pressure nodes.
"""
function stokes_dwr_indicator(sq::SemQuad, qt::Quadtree,
                              ru::AbstractVector{<:Real},
                              rp::AbstractVector{<:Real},
                              z::AbstractVector{<:Real},
                              q::AbstractVector{<:Real};
                              sp::Union{Nothing,SemQuad}=nothing)
    n = length(ru) ÷ 2
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves
        s = 0.0
        for g in sq.elem[lf]
            s += abs(ru[g]*z[g]) + abs(ru[n+g]*z[n+g])
        end
        if sp !== nothing
            for g in sp.elem[lf]
                s += abs(rp[g]*q[g])
            end
        end
        eta[lf] = s
    end
    return eta
end
"""Reduced Stokes residual: Cu'*(fv-Ku-B'p), Cp'*(-B u)."""
function stokes_residual_reduced(K, B, Cu, Cp, u, p, fv)
    ru = Cu' * (fv .- K * u .- B' * p)
    rp = Cp' * (-(B * u))
    return ru, rp
end
"""
    stokes_reduced_residual(K, B, Cu, Cp, u0, p0, fv, u, p)

Residual of the reduced saddle system actually solved:
  fr = Cu'*(fv - K*u0 + B'*p0), gr = Cp'*(B*u0)
  ru = fr - Kred*uf - Bt*pf,  rp = gr - Bs*uf
with uf=Cu'*(u-u0), pf=Cp'*(p-p0).
"""
function stokes_reduced_residual(K, B, Cu, Cp, u0, p0, fv, u, p)
    Kred = Cu' * K * Cu
    Bs = -Cp' * B * Cu
    Bt = sparse(transpose(Bs))
    fr = Cu' * (fv .- K * u0 .+ B' * p0)
    gr = Cp' * (B * u0)
    uf = Cu' * (u .- u0)
    pf = Cp' * (p .- p0)
    ru = fr .- Kred * uf .- Bt * pf
    rp = gr .- Bs * uf
    return ru, rp
end
"""Block-diagonal velocity mass matrix [M 0; 0 M] on the Q_p velocity space."""
function assemble_semquad_velocity_mass(sq::SemQuad)
    M = semquad_mass(sq)
    return [M spzeros(sq.N, sq.N); spzeros(sq.N, sq.N) M]
end

"""Goal RHS gu = Mv * w for J = ∫ w·u dV (velocity goal)."""
function velocity_goal_rhs(sq::SemQuad, wx::AbstractVector{<:Real},
                           wz::AbstractVector{<:Real})
    M = semquad_mass(sq)
    return vcat(M * wx, M * wz)
end
"""
    hierarchical_stokes_residual(ssH, qtH, ssL, qtL, eta_fun, fx, fz, uH, pH)

Prolong a coarse Stokes solution to the fine space, assemble the fine system
without solving, and return the reduced residual
  ru = Cu'*(fv - K*uL - B'*pL), rp = Cp'*(-B*uL).
`ssL` must be the fine space on `qtL`.
"""
function hierarchical_stokes_residual(ssH, qtH, ssL, qtL, eta_fun, fx, fz, uH, pH)
    nH = ssH.su.N
    uxL = transfer_semquad(ssH.su, uH[1:nH], ssL.su;
                           mode=:nodal, zero_walls=false, qt_new=qtL)
    uzL = transfer_semquad(ssH.su, uH[nH+1:end], ssL.su;
                           mode=:nodal, zero_walls=false, qt_new=qtL)
    pL = transfer_semquad(ssH.sp, pH, ssL.sp;
                          mode=:nodal, zero_walls=false, qt_new=qtL)
    S = solve_semquad_stokes_var(ssL, qtL, eta_fun, fx, fz,
                                 (x,y)->0.0, (x,y)->0.0;
                                 bc=:free, assemble_only=true)
    u = vcat(uxL, uzL)
    ru = S.Cu' * (S.fv .- S.K * u .- S.B' * pL)
    rp = S.Cp' * (-(S.B * u))
    return ru, rp
end
"""Aggregate a leaf indicator up to a coarser level by summing children."""
function aggregate_indicator_to_level(eta::Dict{NTuple{3,Int},Float64}, level::Int)
    out = Dict{NTuple{3,Int},Float64}()
    for (lf, v) in eta
        cur = lf
        while cur[1] > level
            cur = (cur[1]-1, cur[2] >> 1, cur[3] >> 1)
        end
        out[cur] = get(out, cur, 0.0) + v
    end
    return out
end
"""Interpolate a scalar SEM field from src_sq to dst_sq at dst nodes."""
function interpolate_semquad(src_sq::SemQuad, z::AbstractVector{<:Real},
                             dst_sq::SemQuad)
    leaves = Set(src_sq.leaves)
    return [MantleSEM._semquad_eval_at(src_sq, z, x, y, leaves)
            for (x, y) in dst_sq.coords]
end
"""dual_weights = z_fine - I_h z_fine, with I_h = fine->coarse->fine."""
function dual_weights_semquad(z::AbstractVector{<:Real}, sqf::SemQuad, sqc::SemQuad)
    zc = interpolate_semquad(sqf, z, sqc)
    zback = interpolate_semquad(sqc, zc, sqf)
    return z .- zback
end
"""
    poisson_dwr_cell_term(sq, qt, u, z, f, sH)

Cell residual term of the scalar Poisson DWR residual, assembled on the
elements of the space that holds `u_h`.

`sq`/`qt` are the current (finest) space and its tree; `u` is `u_h` and `z`
the adjoint solution on that space; `sH` is the coarser space whose
interpolant `I_H` defines the dual weight `z - I_H z`.  Returns the per-leaf
integral of `(f + Delta u_h)(z - I_H z)`; the signed estimator is this minus
`poisson_dwr_face_terms`.
"""
function poisson_dwr_cell_term(sq::SemQuad, qt::Quadtree,
                               u::AbstractVector{<:Real},
                               z::AbstractVector{<:Real}, f::Function,
                               sH::SemQuad)
    length(u) == sq.N ||
        throw(ArgumentError("u length $(length(u)) != N $(sq.N)"))
    length(z) == sq.N ||
        throw(ArgumentError("z length $(length(z)) != N $(sq.N)"))
    n = sq.p + 1
    rule = gll_rule(sq.p)
    leaves = Set(sq.leaves)
    dw = dual_weights_semquad(z, sq, sH)
    out = Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2]*hx; y0 = lf[3]*hy
        ids = sq.elem[lf]; s = 0.0
        for j in 1:n, i in 1:n
            uxx = 0.0; uyy = 0.0
            # Second derivative: the two D factors act on the same coordinate,
            # i.e. (D*D)_{i,m} = sum_k D[i,k] D[k,m], not D[i,k]*D[i,m].
            for k in 1:n, m in 1:n
                uxx += sq.D[i,k]*sq.D[k,m]*u[ids[(j-1)*n+m]]
                uyy += sq.D[j,k]*sq.D[k,m]*u[ids[(m-1)*n+i]]
            end
            x = x0 + 0.5*(rule.nodes[i]+1)*hx
            y = y0 + 0.5*(rule.nodes[j]+1)*hy
            w = 0.25*hx*hy*rule.weights[i]*rule.weights[j]
            zval = MantleSEM._semquad_eval_at(sq, dw, x, y, leaves)
            # D differentiates in the reference coordinate (-1,1); the map to
            # a leaf of size (hx,hy) contributes 4/hx^2 and 4/hy^2.  Square
            # leaves keep the historical single-factor expression bitwise.
            lap = hx == hy ? (4.0/(hx*hx)) * (uxx + uyy) :
                             (4.0/(hx*hx)) * uxx + (4.0/(hy*hy)) * uyy
            s += w * (f(x,y) + lap) * zval
        end
        out[lf] = s
    end
    return out
end
"""
    poisson_dwr_face_terms(sq, qt, u, z, sH)

Interior-face flux term of the scalar Poisson DWR residual, in the
elementwise form that matches the assembled operator.

`sq`/`qt` are the current (finest) space and its tree; `u` is the discrete
solution `u_h` and `z` the adjoint solution on that space; `sH` is the coarser
space whose interpolant `I_H` defines the dual weight `z - I_H z`.

For every leaf K this returns the outward flux integral over the interior
faces of K,
    Phi_K = sum_{F in dK, F interior} int_F (dn_K u_h) (z - I_H z) ds ,
with dn_K the outward normal of K.  Each element face is integrated with that
element's own GLL nodes and weights, which is exactly the quadrature used by
the assembled operator, so summing over leaves reproduces the elementwise
integration-by-parts identity for the residual
    R(z) = sum_K int_K (f+Delta u_h) z - sum_F int_F [dn u_h] z ,
    [dn u_h] = dn u_h|_L - dn u_h|_R .
The signed leaf estimator is
`poisson_dwr_cell_term(...) - poisson_dwr_face_terms(...)`.

The faces must be those of `u_h`'s space: `z - I_H z` vanishes identically on
the skeleton of `sH`, so assembling this term on coarse-mesh faces returns
zero by construction -- a property of the dual weight, not an empty integrand.
Boundary (Nitsche) terms are not included; the identity above assumes that the
traces vanish on the boundary.
"""
function poisson_dwr_face_terms(sq::SemQuad, qt::Quadtree,
                                u::AbstractVector{<:Real},
                                z::AbstractVector{<:Real},
                                sH::SemQuad)
    length(u) == sq.N ||
        throw(ArgumentError("u length $(length(u)) != N $(sq.N)"))
    length(z) == sq.N ||
        throw(ArgumentError("z length $(length(z)) != N $(sq.N)"))
    n = sq.p + 1
    rule = gll_rule(sq.p)
    leaves = Set(sq.leaves)
    dw = dual_weights_semquad(z, sq, sH)
    out = Dict{NTuple{3,Int},Float64}(lf => 0.0 for lf in qt.leaves)
    for lf in qt.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2]*hx; y0 = lf[3]*hy
        ids = sq.elem[lf]
        s = 0.0
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) && continue   # interior faces only
            sgn = (dir === :xplus || dir === :yplus) ? 1.0 : -1.0
            iF = dir === :xminus ? 1 : dir === :xplus ? n : 0
            jF = dir === :yminus ? 1 : dir === :yplus ? n : 0
            for q in 1:n
                if iF > 0
                    der = 0.0
                    for k in 1:n
                        der += sq.D[iF,k] * u[ids[(q-1)*n+k]]
                    end
                    der *= 2.0/hx
                    x = dir === :xplus ? x0 + hx : x0
                    y = y0 + 0.5*(rule.nodes[q] + 1.0)*hy
                    w = 0.5*hy*rule.weights[q]
                else
                    der = 0.0
                    for k in 1:n
                        der += sq.D[jF,k] * u[ids[(k-1)*n+q]]
                    end
                    der *= 2.0/hy
                    y = dir === :yplus ? y0 + hy : y0
                    x = x0 + 0.5*(rule.nodes[q] + 1.0)*hx
                    w = 0.5*hx*rule.weights[q]
                end
                s += sgn * w * der * MantleSEM._semquad_eval_at(sq, dw, x, y, leaves)
            end
        end
        out[lf] = s
    end
    return out
end


"""
    restrict_indicator_to_tree(etaE, qt)

Sum an indicator defined on the leaves of a refined tree onto the leaves of
`qt`: each coarse leaf receives the sum of the finer leaves it contains.
"""
function restrict_indicator_to_tree(etaE::Dict{NTuple{3,Int},Float64}, qt::Quadtree)
    out = Dict{NTuple{3,Int},Float64}(lf => 0.0 for lf in qt.leaves)
    for (lfE, v) in etaE
        cur = lfE
        while cur[1] > 0 && !haskey(out, cur)
            cur = (cur[1] - 1, cur[2] >> 1, cur[3] >> 1)
        end
        haskey(out, cur) && (out[cur] += v)
    end
    return out
end

"""
    poisson_dwr_estimate(sq, qt, u, zhE, sqE, qtE, f)

Goal-oriented DWR estimate for the scalar Poisson problem using an *enriched*
adjoint.

`sq`/`qt` hold the primal solution `u`; `sqE`/`qtE` is an enriched space (one
or more uniform refinements, or order p+1) holding the adjoint `zhE`; `f` is
the source.  The dual weight is

    phi = zhE - I_h zhE ,

with `I_h` the nodal interpolation into the primal space, and the per-leaf
estimate is the residual functional restricted to the primal leaves,

    eta_K = int_K (f + Delta u_h) phi - int_{dK} (dn_K u_h) phi .

Summing over leaves gives `R(u_h)(phi) = J(u_h) - J(u)`, i.e. minus the goal
error of the primal solution.  Measured effectivity is `-1 +- 3e-3` for p=2,3
when the enriched adjoint uses order p+1.

The *un-enriched* construction (adjoint on the primal space, minus an
interpolation onto a coarser space) is not a DWR estimator: it measures
`a(e, z_h - I_H z_h)`, which is nearly the Galerkin-orthogonal zero and gives
effectivities of 1e-3 that degrade under refinement.  Do not use that form.
"""
function poisson_dwr_estimate(sq::SemQuad, qt::Quadtree,
                              u::AbstractVector{<:Real},
                              zhE::AbstractVector{<:Real},
                              sqE::SemQuad, qtE::Quadtree, f::Function)
    length(u) == sq.N ||
        throw(ArgumentError("u length $(length(u)) != N $(sq.N)"))
    length(zhE) == sqE.N ||
        throw(ArgumentError("zhE length $(length(zhE)) != N $(sqE.N)"))
    uE = interpolate_semquad(sq, u, sqE)
    cell = poisson_dwr_cell_term(sqE, qtE, uE, zhE, f, sq)
    face = poisson_dwr_face_terms(sqE, qtE, uE, zhE, sq)
    etaE = Dict{NTuple{3,Int},Float64}()
    for (lf, v) in cell
        etaE[lf] = v - get(face, lf, 0.0)
    end
    return restrict_indicator_to_tree(etaE, qt)
end