# Stokes boundary loads: prescribed traction (`wall_t`) and normal traction
# (`wall_tn`, including the `:open` normal-velocity compatibility checks) as
# boundary-face right-hand-side integrals, box and mapped methods.  Moved
# verbatim from the tail of assembly_var.jl; the assembly file now owns only
# the variable-viscosity leaf/block assembly.
# ---------------------------------------------------------------------------
# Boundary-condition feature, stage 2/3: the prescribed **traction** as a
# boundary-face load (G259/G260 tangential, G261-G263 normal/open).
#
# The Stokes assembly had no boundary-face integral at all: the natural
# (do-nothing) condition of the weak form is `sigma.n = 0`, so every
# unconstrained boundary component was silently traction-free
    # Integration by parts of `-div sigma = f` with
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
`has_no_face_neighbor`/`neighbors` from `Threads.@threads`.  It is built once,
serially, before the
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
combination rather than letting a user believe a load was applied, and the same
validation runs in `Params` and in `_solve_semquad_stokes_var_core` so a direct
library call cannot bypass it.
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
gauge becomes derived (no nodal pin).
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
            "and its pressure gauge becomes derived); set that, or drop " *
            "wall_tn[$i]."))
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
