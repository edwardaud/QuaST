# Stage 0.5b2: general curved-wall linear constraint rows.
#
# The box solver applies Dirichlet data as Cartesian components: on an
# axis-aligned wall the "normal component" is a single global dof, so a `:free`
# (free-slip) wall is one fixed Cartesian dof and a `:rigid` wall is two.  On a
# mapped geometry the wall normal `n` varies along the wall, so free slip is the
# **linear relation** `n . u = 0` between the two Cartesian components at every
# wall node, and a moving rigid wall prescribes the velocity in the wall frame
# `(n, tau)`:
#
#   :free    n . u = g          (g = wall_un . n or 0; tangential traction natural)
#   :rigid   u = un * n + ut * tau,  un = wall_un (or uD . n), ut = wall_u (or uD . tau)
#   :open    no velocity row; the natural traction applies with the wall_t/wall_tn
#            data in the n/tau frame (already wired by the cut-2 traction pass)
#
# where `uD = (uDx, uDz)` is the historical exact-Dirichlet closure pair, and
# `(n, tau)` is the mapped outward unit normal / positive logical tangent at the
# wall GLL node (`face_metric` at the GLL parameters; on the annulus circles
# `n = +-e_r`, `tau = +e_phi`).  The map convention is therefore: on a *mapped*
# geometry `wall_un` is the outward-normal velocity `u . n` and `wall_u` is
# `u . tau`; on the box they keep the historical global positive-component
# meaning.  With no `wall_un`/`wall_u` data the rigid formula reproduces the old
# mapped behaviour `u = uD` exactly.
#
# **Row representation.**  The solver's affine machinery (`_affine_constraint`)
# already supports general *homogeneous* linear relations `u_c = sum_m w_m u_m`
# (the hanging-node constraints) and Dirichlet values.  A free-slip row
# `n_x u_x + n_y u_y = g` is expressed by eliminating the larger-|n| component
# (`u_p = -(n_q/n_p) u_q + g/n_p`): the eliminated global dof `p` goes to
# `cons[p] = ([q], [-n_q/n_p])`, and the constant `g/n_p` to the new optional
# `offset[p]`.  The sign is chosen so the row is exact in the *outward normal*
# convention: `u . n = g` with `g` a signed normal velocity.
#
# **Corners.**  The four logical walls are processed in the historical order
# `(xminus, xplus, yminus, yplus)`; a corner node belongs to two walls and the
# later wall wins, exactly as the box's write order does:
#   * a `:rigid` wall writes both components (Dirichlet) and overwrites any
#     earlier value there;
#   * a `:free` wall imposes `n . u = g`; if one component is already Dirichlet
#     the relation determines the other component (a Dirichlet value), and if
#     both are Dirichlet it overwrites the *dominant* component of its normal
#     (`|n_p| >= |n_q|`) to satisfy its own row -- on an axis-aligned wall this
#     is exactly the box's "later wall writes that component" rule;
#   * two `:free` walls at a corner give two independent rows, solved as a 2x2
#     system: both components become Dirichlet (the corner velocity is the
#     intersection of the two slip planes);
#   * an `:open` wall writes nothing, so a corner shared with it keeps the other
#     wall's condition.
#
# **Hanging wall nodes.**  A fine wall node that lies inside a coarser leaf's
# wall face is interpolation-constrained (`semquad_constraints`).  The wall row
# there is imposed *after* the interpolation: the tangential component is the
# interpolation of the coarse values projected on the fine node's tangent,
# `u_f = (V . tau) tau + g n` with `V` the interpolated vector, and both
# components of `u_f` are expressed as affine combinations of the coarse *free*
# dofs (the coarse wall nodes' own rows are substituted through their affine
# representations).  This keeps every wall node exactly on the curved wall
# instead of only the coarse ones.
#
# **Rigid-mode analysis.**  Every wall row is a linear condition on the three 2-D
# rigid modes `(1,0)`, `(0,1)`, `(-y,x)`; the Gram matrix `A = sum r r'` of those
# coefficient rows has rank 3 iff the boundary data leave no rigid-body mode
# admissible (a unique velocity).  Rank 2 with the null vector `(0,0,1)` means a
# solid-body rotation is admissible -- the Stokes nullspace of a free-slip curved
# wall (the full ring later; a wedge with open radial cuts here): one free
# velocity dof is pinned to remove it from the linear system, and the returned
# velocity is projected to the zero-net-angular-momentum representative.  Any
# other nullspace is refused with the motion named (a truncated boundary set,
# not something the rotation gauge can fix).
#
# **Rotation gauge.**  The design's condition is `int r x u dA = 0`, equivalent
# to the L2 projection onto `(-y, x)` because the two integrals are the same
# mass-matrix inner product.  `omega = (int (x u_y - y u_x) dA) / (int r^2 dA)`
# is removed from the returned velocity; the momentum-equation residual is
# unchanged (a rigid rotation has zero strain rate and zero divergence) provided
# the load has no net torque -- which `_semquad_check_rotation_load` verifies,
# refusing an inconsistent card instead of returning a field that cannot balance
# it.

"""
    _SemQuadWallCond

One wall condition of one global node: the wall index (`1..4` in
`(xminus, xplus, yminus, yplus)` order), the physical node position and the
mapped outward normal / positive tangent there.  A corner node carries two of
these (one per wall), in wall order.
"""
struct _SemQuadWallCond
    wi::Int
    x::Float64
    y::Float64
    nx::Float64
    ny::Float64
    tx::Float64
    ty::Float64
end

"""
    SemQuadWallData

What the mapped wall-row pass produces for one solve:

* `fixed` / `cons` / `offsets` -- the affine constraint data (`cons[c] =
  (masters, weights)` with `u_c = sum w u_m`, `offsets[c]` the constant term);
* `un` -- per wall, the prescribed outward normal velocity at each wall node,
  used by the net-flux compatibility check (`nothing` on an `:open` wall);
* `umax` -- the largest prescribed normal value (the default path skips the flux
  check when it is zero, exactly like the box);
* `rows` -- the rigid-mode coefficient rows for the nullspace analysis;
* `gauge` / `gauge_dof` -- `:none` or `:rotation`; `gauge_dof` is the free
  velocity dof pinned to remove the rotation nullspace (`0` when `:none`).
"""
struct SemQuadWallData
    fixed::Dict{Int,Float64}
    cons::Dict{Int,Tuple{Vector{Int},Vector{Float64}}}
    offsets::Dict{Int,Float64}
    un::NTuple{4,Dict{Int,Float64}}
    umax::Float64
    rows::Vector{NTuple{3,Float64}}
    gauge::Symbol
    gauge_dof::Int
end

# The node id of face node `j` on logical `side` (same ordering as the box
# bodies' explicit index arithmetic).
@inline function _wall_face_node(uids::Vector{Int}, n::Int, side::Symbol, j::Int)
    side === :xminus && return uids[(j - 1) * n + 1]
    side === :xplus && return uids[(j - 1) * n + n]
    side === :yminus && return uids[j]
    return uids[(n - 1) * n + j]
end

"""
    _semquad_wall_frames(g, su, bmask) -> Dict{Int,Vector{_SemQuadWallCond}}

All domain-boundary GLL nodes with their wall condition data.  The walls are
visited in the historical order, so a corner node's vector is ordered by wall
(the write rule of the header); duplicate visits of the same node on the same
wall (shared coarse/fine leaf faces) are skipped.
"""
function _semquad_wall_frames(g::AbstractGeometry, su::SemQuad,
                              bmask::Vector{UInt8})
    n = su.p + 1
    conds = Dict{Int,Vector{_SemQuadWallCond}}()
    for (wi, side) in enumerate(LOGICAL_SIDES)
        seen = Set{Int}()
        for it in eachindex(su.leaves)
            b = 4 * (it - 1)
            bmask[b + wi] != 0x00 || continue
            lf = su.leaves[it]
            uids = su.elem[lf]
            fm = _mapped_face_metric(g, su, lf, side, su.xi)
            @inbounds for j in 1:n
                node = _wall_face_node(uids, n, side, j)
                node in seen && continue
                push!(seen, node)
                xq, yq = face_point(fm, j)
                nx_, ny_ = face_normal(fm, j)
                tx_, ty_ = face_tangent(fm, j)
                push!(get!(conds, node, _SemQuadWallCond[]),
                      _SemQuadWallCond(wi, xq, yq, nx_, ny_, tx_, ty_))
            end
        end
    end
    return conds
end

# Affine representation of one velocity dof: `u_d = sum_j ws[j] u_ms[j] + off`.
# The masters are either Dirichlet dofs or free dofs; a dof that is itself
# constrained by a wall row is returned through that row (wall rows are built
# per node, so the recursion is one level deep).
@inline function _wall_dof_rep(fixed::Dict{Int,Float64},
                               cons::Dict{Int,Tuple{Vector{Int},Vector{Float64}}},
                               offs::Dict{Int,Float64}, dof::Int)
    haskey(fixed, dof) && return (Int[], Float64[], fixed[dof])
    if haskey(cons, dof)
        ms, ws = cons[dof]
        return (ms, ws, get(offs, dof, 0.0))
    end
    return ([dof], [1.0], 0.0)
end

# Add `coef * (sum ws u_ms + off)` to an affine accumulator.
@inline function _wall_affine_add!(acc::Dict{Int,Float64}, coef::Float64,
                                   ms::Vector{Int}, ws::Vector{Float64})
    for (m, w) in zip(ms, ws)
        acc[m] = get(acc, m, 0.0) + coef * w
    end
    return nothing
end

"""
    _semquad_mapped_wall_data(g, su, bmask, resolved_wall_bc, wall_u, wall_un, uDx, uDz, cons_hang) -> SemQuadWallData

Build the general curved-wall constraint data of the header: the Dirichlet
values (in the wall frame), the free-slip linear relations (with their affine
offsets), the hanging wall-node rows, the prescribed normal data for the flux
check and the rigid-mode rows / rotation gauge decision.
"""
function _semquad_mapped_wall_data(g::AbstractGeometry, su::SemQuad,
                                   bmask::Vector{UInt8}, resolved_wall_bc,
                                   wall_u, wall_un,
                                   uDx::Function, uDz::Function,
                                   cons_hang::Dict{Int,Tuple{Vector{Int},Vector{Float64}}})
    n = su.p + 1
    nu = su.N
    conds = _semquad_wall_frames(g, su, bmask)
    fixed = Dict{Int,Float64}()
    cons = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    offs = Dict{Int,Float64}()
    un = (Dict{Int,Float64}(), Dict{Int,Float64}(),
          Dict{Int,Float64}(), Dict{Int,Float64}())
    umax = 0.0
    rows = NTuple{3,Float64}[]
    cand = Tuple{Int,Float64}[]     # (free dof, |rotation value|) candidates

    nonhang = sort!([node for node in keys(conds) if !haskey(cons_hang, node)])
    for node in nonhang
        cs = conds[node]
        elim = 0
        elimw = 0.0
        elimoff = 0.0
        for c in cs
            wbc = resolved_wall_bc[c.wi]
            wbc === :open && continue
            unv = wall_un[c.wi] === nothing ?
                  (uDx(c.x, c.y) * c.nx + uDz(c.x, c.y) * c.ny) :
                  Float64(wall_un[c.wi](c.x, c.y))
            un[c.wi][node] = unv
            umax = max(umax, abs(unv))
            if wbc === :rigid
                utv = wall_u[c.wi] === nothing ?
                      (uDx(c.x, c.y) * c.tx + uDz(c.x, c.y) * c.ty) :
                      Float64(wall_u[c.wi](c.x, c.y))
                fixed[node] = unv * c.nx + utv * c.tx
                fixed[nu + node] = unv * c.ny + utv * c.ty
                elim = 0                  # a later rigid/free wall owns the corner
            else                       # :free -- impose n . u = g
                gval = unv
                if haskey(fixed, node) && haskey(fixed, nu + node)
                    # both components Dirichlet: the free row overwrites the
                    # dominant component of its normal (box "later wall wins")
                    if abs(c.nx) >= abs(c.ny)
                        fixed[node] = (gval - c.ny * fixed[nu + node]) / c.nx
                    else
                        fixed[nu + node] = (gval - c.nx * fixed[node]) / c.ny
                    end
                    elim = 0
                elseif haskey(fixed, node)
                    fixed[nu + node] = (gval - c.nx * fixed[node]) / c.ny
                    elim = 0
                elseif haskey(fixed, nu + node)
                    fixed[node] = (gval - c.ny * fixed[nu + node]) / c.nx
                    elim = 0
                elseif elim != 0
                    # second free row at a corner: previous row is
                    # u_elim = elimw * u_other + elimoff, solve the 2x2.
                    if elim == node
                        den = c.nx * elimw + c.ny
                        uy_ = (gval - c.nx * elimoff) / den
                        fixed[nu + node] = uy_
                        fixed[node] = elimw * uy_ + elimoff
                    else
                        den = c.ny * elimw + c.nx
                        ux_ = (gval - c.ny * elimoff) / den
                        fixed[node] = ux_
                        fixed[nu + node] = elimw * ux_ + elimoff
                    end
                    elim = 0
                elseif abs(c.nx) >= abs(c.ny)
                    elim = node
                    elimw = -c.ny / c.nx
                    elimoff = gval / c.nx
                else
                    elim = nu + node
                    elimw = -c.nx / c.ny
                    elimoff = gval / c.ny
                end
            end
        end
        if elim != 0
            other = elim == node ? nu + node : node
            cons[elim] = ([other], [elimw])
            offs[elim] = elimoff
        end
        x0 = cs[1].x
        y0 = cs[1].y
        fixx = haskey(fixed, node)
        fixy = haskey(fixed, nu + node)
        if fixx
            push!(rows, (1.0, 0.0, -y0))
        end
        if fixy
            push!(rows, (0.0, 1.0, x0))
        end
        if !(fixx || fixy)
            for c in cs
                resolved_wall_bc[c.wi] === :open && continue
                push!(rows, (c.nx, c.ny, -c.nx * c.y + c.ny * c.x))
            end
        end
        # gauge-dof candidates: the node's free component(s)
        if !fixx && !haskey(cons, node)
            push!(cand, (node, abs(y0)))
        end
        if !fixy && !haskey(cons, nu + node)
            push!(cand, (nu + node, abs(x0)))
        end
    end

    # Pass 2: hanging wall nodes (fine node inside a coarse leaf's wall face).
    hang = sort!([node for node in keys(conds) if haskey(cons_hang, node)])
    for node in hang
        cs = conds[node]
        length(cs) == 1 ||
            throw(ArgumentError("a hanging wall node lies on two walls (node " *
                "$node): a hanging node is interior to one coarse face, so this " *
                "cannot be a domain corner; the constraint system is ambiguous.  " *
                "This is a bug in the wall-row pass, please report it."))
        c = cs[1]
        wbc = resolved_wall_bc[c.wi]
        wbc === :open && continue
        unv = wall_un[c.wi] === nothing ?
              (uDx(c.x, c.y) * c.nx + uDz(c.x, c.y) * c.ny) :
              Float64(wall_un[c.wi](c.x, c.y))
        un[c.wi][node] = unv
        umax = max(umax, abs(unv))
        if wbc === :rigid
            utv = wall_u[c.wi] === nothing ?
                  (uDx(c.x, c.y) * c.tx + uDz(c.x, c.y) * c.ty) :
                  Float64(wall_u[c.wi](c.x, c.y))
            # Dirichlet wins over the hanging interpolation, exactly as in the
            # historical mapped fixed-dof loop.
            fixed[node] = unv * c.nx + utv * c.tx
            fixed[nu + node] = unv * c.ny + utv * c.ty
            push!(rows, (1.0, 0.0, -c.y))
            push!(rows, (0.0, 1.0, c.x))
        else
            # free slip at a hanging node: tangential component = interpolation
            # of the coarse values, normal component = g.
            mxs, wxs = cons_hang[node]
            mys, wys = cons_hang[nu + node]
            ax = Dict{Int,Float64}(); ox = 0.0
            ay = Dict{Int,Float64}(); oy = 0.0
            for (m, w) in zip(mxs, wxs)
                ms_, ws_, off_ = _wall_dof_rep(fixed, cons, offs, m)
                _wall_affine_add!(ax, w, ms_, ws_)
                ox += w * off_
            end
            for (m, w) in zip(mys, wys)
                ms_, ws_, off_ = _wall_dof_rep(fixed, cons, offs, m)
                _wall_affine_add!(ay, w, ms_, ws_)
                oy += w * off_
            end
            # V . tau, then u_f = (V.tau) tau + g n
            at = Dict{Int,Float64}()
            for (m, w) in ax
                at[m] = get(at, m, 0.0) + c.tx * w
            end
            for (m, w) in ay
                at[m] = get(at, m, 0.0) + c.ty * w
            end
            ot = c.tx * ox + c.ty * oy
            # u_x = tx * (V.tau) + nx * g, u_y = ty * (V.tau) + ny * g
            if isempty(at)
                fixed[node] = c.tx * ot + c.nx * unv
                fixed[nu + node] = c.ty * ot + c.ny * unv
            else
                uxm = collect(keys(at)); uxw = collect(values(at))
                cons[node] = (uxm, c.tx .* uxw)
                offs[node] = c.tx * ot + c.nx * unv
                cons[nu + node] = (copy(uxm), c.ty .* uxw)
                offs[nu + node] = c.ty * ot + c.ny * unv
            end
            push!(rows, (c.nx, c.ny, -c.nx * c.y + c.ny * c.x))
        end
    end

    gauge, gauge_dof = _semquad_wall_nullspace(rows, cand)
    if gauge === :rotation
        fixed[gauge_dof] = 0.0
    end
    return SemQuadWallData(fixed, cons, offs, un, umax, rows, gauge, gauge_dof)
end

"""
    _semquad_wall_nullspace(rows, cand) -> (gauge, gauge_dof)

Rigid-mode admissibility analysis of the wall rows (the header's "Rigid-mode
analysis").  Returns `(:none, 0)` when the rows exclude all three rigid modes,
`(:rotation, dof)` when the only admissible mode is the solid-body rotation (the
free velocity dof with the largest `|rotation mode|` is returned as the gauge
pin), and throws an actionable `ArgumentError` for any other nullspace.
"""
function _semquad_wall_nullspace(rows::Vector{NTuple{3,Float64}},
                                 cand::Vector{Tuple{Int,Float64}})
    A = zeros(Float64, 3, 3)
    for r in rows
        A[1, 1] += r[1] * r[1]
        A[1, 2] += r[1] * r[2]
        A[1, 3] += r[1] * r[3]
        A[2, 2] += r[2] * r[2]
        A[2, 3] += r[2] * r[3]
        A[3, 3] += r[3] * r[3]
    end
    A[2, 1] = A[1, 2]; A[3, 1] = A[1, 3]; A[3, 2] = A[2, 3]
    E = eigen(Symmetric(A))
    lam = E.values
    lamax = lam[3]
    lamax > 0.0 || throw(ArgumentError("the mapped wall rows contain no " *
        "constraint on the rigid-body modes, so the velocity is determined only " *
        "up to an arbitrary rigid motion.  Leave at least one wall with a " *
        "Dirichlet component (`wall_bc = :free` or `:rigid`); all-`:open` is " *
        "refused before the assembly."))
    thresh = 1.0e-10 * lamax
    nulls = [k for k in 1:3 if lam[k] <= thresh]
    isempty(nulls) && return (:none, 0)
    names = ("x-translation", "y-translation", "rigid rotation")
    if length(nulls) == 1
        v = E.vectors[:, nulls[1]]
        vmax = maximum(abs, v)
        vmax > 0.0 && (v ./= vmax)
        if abs(v[1]) <= 1.0e-6 && abs(v[2]) <= 1.0e-6
            # only the solid-body rotation is admissible: velocity gauge
            isempty(cand) && throw(ArgumentError("the boundary data admit a " *
                "solid-body rotation but no free wall dof was found to pin it; " *
                "this is a bug in the wall-row pass, please report it."))
            best = cand[1]
            for c in cand
                c[2] > best[2] && (best = c)
            end
            return (:rotation, best[1])
        end
        imax = argmax(abs.(v))
        motion = join([string(round(v[k]; digits=4)) * " * " * names[k]
                       for k in 1:3 if abs(v[k]) > 1.0e-6], " + ")
        throw(ArgumentError("the mapped wall data leave a rigid-body mode " *
            "unconstrained (the admissible motion is $motion), so the velocity " *
            "is not unique.  Only the solid-body rotation can be gauged " *
            "automatically (the annulus zero-net-angular-momentum projection); " *
            "a translation needs a Dirichlet normal component on a wall that " *
            "blocks it.  Add a `:rigid` or `:free` wall (all four `:open` is " *
            "refused before the assembly), or remove the offending " *
            "per-wall condition."))
    end
    descr = join([names[k] for k in nulls], ", ")
    throw(ArgumentError("the mapped wall data leave admissible rigid-body " *
        "modes {$descr}: the coefficient rows do not constrain them, so the " *
        "velocity and the discrete saddle are non-unique.  The rotation gauge " *
        "only removes the solid-body rotation; add a Dirichlet normal " *
        "component on a wall that blocks the printed motion(s), or use a " *
        "closed `:rigid`/`:free` wall set."))
end

"""
    _semquad_merge_wall_constraints(cons_hang, wd) -> (cons, offsets)

Merge the hanging-node constraints with the wall rows.  Two composition rules:

* a hanging constraint whose *master* is wall-constrained is substituted by the
  wall row's affine representation (the hanging constraints were resolved among
  themselves, but they cannot know the wall rows);
* a hanging *wall node*'s own rows replace its hanging entries (the wall pass
  already composed the interpolation into the row, see the header).

`offsets` starts from the wall rows' constants and accumulates the constants of
substituted masters.
"""
function _semquad_merge_wall_constraints(cons_hang, wd::SemQuadWallData)
    cons = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    offs = copy(wd.offsets)
    wc = wd.cons
    for (c, (ms, ws)) in cons_hang
        nms = Int[]
        nws = Float64[]
        off = get(offs, c, 0.0)
        for (m, w) in zip(ms, ws)
            if haskey(wc, m)
                m2, w2 = wc[m]
                for (mm, ww) in zip(m2, w2)
                    push!(nms, mm)
                    push!(nws, w * ww)
                end
                off += w * get(wd.offsets, m, 0.0)
            else
                push!(nms, m)
                push!(nws, w)
            end
        end
        cons[c] = (nms, nws)
        off == 0.0 || (offs[c] = off)
    end
    # A wall row at a hanging wall node replaces the hanging row (it encodes
    # both the wall condition and the interpolation).
    for (c, row) in wc
        cons[c] = row
    end
    return cons, offs
end

"""
    _semquad_wall_normal_flux(g, su, bmask, un, fd) -> (flux, umax)

Net outward boundary flux of the **prescribed** wall data on a mapped geometry:
per wall, the nodal values `un[wall][node]` (outward normal velocity) are
interpolated with the face GLL basis and integrated with the mapped face
Jacobian.  `umax` is the largest value (the solver skips the compatibility check
when it is zero).  This is the mapped analogue of the box body's analytic
component integral; the box method is untouched and keep its historical sign
table.
"""
function _semquad_wall_normal_flux(g::AbstractGeometry, su::SemQuad,
                                   bmask::Vector{UInt8},
                                   un::NTuple{4,Dict{Int,Float64}},
                                   fd::FEData)
    n = su.p + 1
    phi = fd.phi_v
    flux = 0.0
    umax = 0.0
    @inbounds for it in eachindex(su.leaves)
        lf = su.leaves[it]
        uids = su.elem[lf]
        b = 4 * (it - 1)
        for (s, side) in enumerate(LOGICAL_SIDES)
            bmask[b + s] != 0x00 || continue
            d = un[s]
            isempty(d) && continue
            fm = face_metric(g, su, lf, side, fd)
            for q in 1:fd.nq
                v = 0.0
                for j in 1:n
                    v += get(d, _wall_face_node(uids, n, side, j), 0.0) *
                         phi[j, q]
                end
                flux += face_weight(fm, fd, q) * v
            end
        end
    end
    for d in un
        for v in values(d)
            umax = max(umax, abs(v))
        end
    end
    return flux, umax
end

"""
    _semquad_rotation_moment(g, sq, u) -> (S, Q)

Discrete angular momentum `S = int (x u_y - y u_x) dA` of the velocity field
`u = [u_x; u_y]` and the rotation-mode norm `Q = int (x^2 + y^2) dA`, both with
the mapped measure and the velocity space's Gauss quadrature (exact for the
polynomial integrands: `u` is degree `p` per leaf, the rotation mode linear).
"""
function _semquad_rotation_moment(g::AbstractGeometry, sq::SemQuad,
                                  u::Vector{Float64})
    p = sq.p
    n = p + 1
    nu = sq.N
    fd = make_fedata(p, p - 1)
    S = 0.0
    Q = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0
            uyq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += u[ids[k]] * b
                uyq += u[nu + ids[k]] * b
            end
            S += w * (x * uyq - y * uxq)
            Q += w * (x * x + y * y)
        end
    end
    return S, Q
end

"""
    _semquad_project_rotation!(u, g, sq) -> omega

Remove the solid-body rotation from the velocity field in place
(`u_x += omega y`, `u_y -= omega x`), leaving the zero-net-angular-momentum
representative `int r x u dA = 0` in the **discrete** functional: `omega` is
`S(u)/S(Z_h)` where `Z_h` is the represented rotation mode (the nodal `(-y,x)`
interpolated by the velocity space), not `S(u)/int r^2 dA`.  On the mapped
annulus the nodal interpolation of a linear field is not exactly linear, so
`S(Z_h) = int (x Z_y - y Z_x) dA` differs from `int r^2 dA` by `O(h^(p+1))`;
normalising by `S(Z_h)` is what makes the *reported* discrete angular momentum
zero to round-off (the design's condition), and in the continuum limit both
normalisations are the L2 projection onto `(-y,x)` (the gate measures the
`O(h^(p+1))` difference).  Returns `omega` so the caller records the removed
rotation.
"""
function _semquad_project_rotation!(u::Vector{Float64}, g::AbstractGeometry,
                                    sq::SemQuad)
    p = sq.p
    n = p + 1
    nu = sq.N
    fd = make_fedata(p, p - 1)
    S = 0.0
    SZ = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0
            uyq = 0.0
            zxq = 0.0
            zyq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += u[ids[k]] * b
                uyq += u[nu + ids[k]] * b
                cx, cy = sq.coords[ids[k]]
                zxq += (-cy) * b
                zyq += cx * b
            end
            S += w * (x * uyq - y * uxq)
            SZ += w * (x * zyq - y * zxq)
        end
    end
    omega = S / SZ
    for (gid, pp) in enumerate(sq.coords)
        x = pp[1]
        y = pp[2]
        u[gid] += omega * y
        u[nu + gid] -= omega * x
    end
    return omega
end

"""
    _semquad_check_rotation_load(su, fv, gauge)

Solvability check of a rotation-gauge configuration: the assembled load must
have zero net torque about the origin, `int Z . f dA = 0` for the nodal rotation
mode `Z = (-y, x)`, because the rotation mode is in the kernel of the operator
and no wall in a free-slip/open configuration can exert a torque.  An
inconsistent load has no steady Stokes solution; the pinned solve would return a
field that does not balance it, so the card is refused instead with the measured
torque.
"""
function _semquad_check_rotation_load(su::SemQuad, fv::Vector{Float64},
                                      gauge::Symbol)
    gauge === :rotation || return nothing
    nu = su.N
    ztf = 0.0
    nz = 0.0
    nf = 0.0
    for (gid, p) in enumerate(su.coords)
        zx = -p[2]
        zy = p[1]
        fx = fv[gid]
        fy = fv[nu + gid]
        ztf += fx * zx + fy * zy
        nz += zx * zx + zy * zy
        nf += fx * fx + fy * fy
    end
    tol = 1.0e-6 * sqrt(nz) * sqrt(nf)
    abs(ztf) <= tol && return nothing
    throw(ArgumentError("the boundary configuration admits a solid-body " *
        "rotation (a free-slip curved wall with an `:open` wall; the solver " *
        "removes it with the zero-net-angular-momentum gauge), but the " *
        "assembled load has a net torque about the origin of $ztf (threshold " *
        "$tol = 1e-6 * |Z| * |f|; the mapped rotation mode is only " *
        "approximated by the nodal field, so a smooth torque-free load has an " *
        "O(h^(p+1)) consistency defect).  A rigid rotation has zero strain rate and " *
        "zero divergence, so a torque imbalance cannot be balanced by a " *
        "steady Stokes solution and the projected velocity would not satisfy " *
        "the momentum equation.  Either make the load torque-free (the " *
        "annulus buoyancy `Ra theta e_r` is), or leave a rigid wall that can " *
        "exert a torque.  Test: _semquad_check_rotation_load, G275."))
end

"""
    _SEMQUAD_NO_WALLS

The no-wall sentinel of [`_semquad_prepare_wall_rows!`](@ref): an empty
`SemQuadWallData` for the `CartesianBox` method.  It is a module constant and is
never mutated (the box path only reads `gauge === :none`); using it instead of
`nothing` keeps the prepared-data local concretely typed on the box path, which
the stage-0.5b2 allocation A/B caught as +208 B per assembly when the union was
there.
"""
const _SEMQUAD_NO_WALLS = SemQuadWallData(Dict{Int,Float64}(),
    Dict{Int,Tuple{Vector{Int},Vector{Float64}}}(),
    Dict{Int,Float64}(),
    (Dict{Int,Float64}(), Dict{Int,Float64}(),
     Dict{Int,Float64}(), Dict{Int,Float64}()),
    0.0, NTuple{3,Float64}[], :none, 0)

"""
    _semquad_prepare_wall_rows!(fixed, cons2, geom, su, qt, bmask, resolved_wall_bc, wall_u, wall_un, uDx, uDz) -> SemQuadWallData

Boundary-dof preparation as a **function barrier dispatched on the concrete
geometry**.  The `CartesianBox` method runs the historical fixed-dof body into
`fixed` and returns the no-wall sentinel (nothing else is written; `cons2` is
filled later by the caller in its historical order).  The mapped method builds
the hanging constraints into `cons2`, runs the general curved-wall row pass
([`_semquad_mapped_wall_data`](@ref)), copies its Dirichlet values into `fixed`
and returns its `SemQuadWallData`.

Both methods return a concrete `SemQuadWallData`, so the core solve's local is
never a `Union{Nothing,...}` and the box's `_affine_constraint` call stays
statically dispatched (the allocation contract of the stage-0.5b2 A/B).
"""
function _semquad_prepare_wall_rows!(fixed::Dict{Int,Float64},
                                     cons2::Dict{Int,Tuple{Vector{Int},Vector{Float64}}},
                                     geom::CartesianBox, su::SemQuad,
                                     qt::Quadtree, bmask::Vector{UInt8},
                                     resolved_wall_bc, wall_u, wall_un,
                                     uDx::Function, uDz::Function)
    _semquad_fill_fixed!(fixed, geom, su, bmask, resolved_wall_bc, wall_u,
                         wall_un, uDx, uDz)
    return _SEMQUAD_NO_WALLS
end

function _semquad_prepare_wall_rows!(fixed::Dict{Int,Float64},
                                     cons2::Dict{Int,Tuple{Vector{Int},Vector{Float64}}},
                                     geom::AbstractGeometry, su::SemQuad,
                                     qt::Quadtree, bmask::Vector{UInt8},
                                     resolved_wall_bc, wall_u, wall_un,
                                     uDx::Function, uDz::Function)
    nu = su.N
    # The wall-row pass needs the hanging constraints (a fine wall node is
    # interpolation-constrained and its row is composed with the interpolation).
    for (c, (ms, ws)) in semquad_constraints(su, qt)
        cons2[c] = (ms, ws)
        cons2[nu + c] = (nu .+ ms, ws)
    end
    wd = _semquad_mapped_wall_data(geom, su, bmask, resolved_wall_bc, wall_u,
                                   wall_un, uDx, uDz, cons2)
    for (k, v) in wd.fixed
        fixed[k] = v
    end
    return wd
end