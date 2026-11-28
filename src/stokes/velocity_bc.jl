# Stokes velocity boundary conditions: wall-index mapping, wall normal flux,
# rigid-mode well-posedness, fixed-DOF lifting, perimeter and the composed
# `_semquad_velocity_boundary` barrier.  Moved verbatim from the head of
# solve_var_core.jl; the solve body now starts at the residual solver.
# The core variable-viscosity Stokes solve.
#
# `_solve_semquad_stokes_var_core` is the body of every `linear` setting: it
# assembles the operators for the viscosity closure it is handed and runs the
# path the selector chose (`:direct`, `:mg`), with the iteration diagnostics and
# the acceptance test.  The public entry point in `solve_var.jl` wraps it with the
# `:auto` selection and the fallback.

"""
    _wall_index(dir)

Position of a wall in `Params.wall_bc`, ordered `(xminus, xplus, yminus, yplus)`.
"""
_wall_index(dir::Symbol) = dir === :xminus ? 1 :
                           dir === :xplus ? 2 :
                           dir === :yminus ? 3 : 4

"""
    _semquad_wall_normal_flux(ss, fixed, fd, n, nu, bmask) -> (flux, umax)

Net boundary flux `∫∂Ω u·n dS` of the prescribed normal Dirichlet data, in the
**outward-normal** convention (negative on `xminus`/`yminus`, positive on
`xplus`/`yplus`), plus the largest prescribed normal value `umax`.

The data are read from the final `fixed` dict and integrated with the same 1-D
Gauss rule the leaf assembly uses (`fd.gl_w`/`fd.phi_v`, exact for the
polynomial interpolant of the face data).  A corner node belongs to two walls;
using the final dict is deliberate, because the wall write order
`(xminus, xplus, yminus, yplus)` lets a later rigid y-wall's *tangential* value
overwrite the x-wall's normal one at the corner, and that is the data the solve
actually imposes. The pass is serial and runs
after the fixed-dof loop, so it changes no assembly order; `bmask` is the
precomputed serial boundary mask (`_semquad_boundary_face_mask`), so this pass
never queries the quadtree neighbour table either.
"""
function _semquad_wall_normal_flux(ss::SemStokes,
                                   fixed::Dict{Int,Float64}, fd::FEData,
                                   n::Int, nu::Int, bmask::Vector{UInt8})
    return _semquad_wall_normal_flux(ss, ss.su.geom, fixed, fd, n, nu, bmask)
end

# Historical box body, verbatim (no coordinate is read: the data live in
# `fixed`, and the outward sign is the wall's, see the docstring above).
function _semquad_wall_normal_flux(ss::SemStokes, ::CartesianBox,
                                   fixed::Dict{Int,Float64}, fd::FEData,
                                   n::Int, nu::Int, bmask::Vector{UInt8})
    su = ss.su
    flux = 0.0
    umax = 0.0
    for it in eachindex(su.leaves)
        lf = su.leaves[it]
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        uids = su.elem[lf]
        b = 4 * (it - 1)
        for dir in (:xminus, :xplus, :yminus, :yplus)
            bmask[b + _wall_index(dir)] != 0x00 || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                sgn = dir === :xplus ? 1.0 : -1.0
                for j in 1:n
                    umax = max(umax, abs(get(fixed, uids[(j - 1) * n + iF], 0.0)))
                end
                for q in 1:fd.nq
                    val = 0.0
                    for j in 1:n
                        val += get(fixed, uids[(j - 1) * n + iF], 0.0) *
                               fd.phi_v[j, q]
                    end
                    flux += sgn * (0.5 * hy * fd.gl_w[q]) * val
                end
            else
                jF = dir === :yminus ? 1 : n
                sgn = dir === :yplus ? 1.0 : -1.0
                for i in 1:n
                    umax = max(umax, abs(get(fixed, nu + uids[(jF - 1) * n + i], 0.0)))
                end
                for q in 1:fd.nq
                    val = 0.0
                    for i in 1:n
                        val += get(fixed, nu + uids[(jF - 1) * n + i], 0.0) *
                               fd.phi_v[i, q]
                    end
                    flux += sgn * (0.5 * hx * fd.gl_w[q]) * val
                end
            end
        end
    end
    return flux, umax
end

# The node index of face node `j` in the leaf's element id table, for the four
# logical sides (the same ordering the box body enumerates explicitly).
@inline function _face_node(uids::Vector{Int}, n::Int, side::Symbol, j::Int)
    side === :xminus && return uids[(j - 1) * n + 1]
    side === :xplus && return uids[(j - 1) * n + n]
    side === :yminus && return uids[j]
    return uids[(n - 1) * n + j]
end

# Mapped body: the prescribed data are global Cartesian components on the
# Dirichlet walls, so the outward flux is `int (u . n) dS` with the mapped
# outward unit normal and the mapped face Jacobian.  The interpolation uses the
# same 1-D face basis/quadrature as the box body (`fd.phi_v`, `fd.gl_w`).  On the
# box map this equals the box expression, but it is a separate body: the box
# path keeps its historical sign table.
function _semquad_wall_normal_flux(ss::SemStokes, g::AbstractGeometry,
                                   fixed::Dict{Int,Float64}, fd::FEData,
                                   n::Int, nu::Int, bmask::Vector{UInt8})
    su = ss.su
    flux = 0.0
    umax = 0.0
    phi = fd.phi_v
    for it in eachindex(su.leaves)
        lf = su.leaves[it]
        uids = su.elem[lf]
        b = 4 * (it - 1)
        for (s, side) in enumerate(LOGICAL_SIDES)
            bmask[b + s] != 0x00 || continue
            fm = face_metric(g, su, lf, side, fd)
            for q in 1:fd.nq
                vx = 0.0
                vy = 0.0
                for j in 1:n
                    node = _face_node(uids, n, side, j)
                    vx += get(fixed, node, 0.0) * phi[j, q]
                    vy += get(fixed, nu + node, 0.0) * phi[j, q]
                end
                nx_, ny_ = face_normal(fm, q)
                un = vx * nx_ + vy * ny_
                flux += face_weight(fm, fd, q) * un
                umax = max(umax, abs(un))
            end
        end
    end
    return flux, umax
end

"""
    _semquad_rigid_mode_gram(ss, fixed) -> Matrix{Float64}

3x3 Gram matrix `G_ij = sum_d m_i(d) m_j(d)` of the three 2-D rigid-body modes
restricted to the Dirichlet dofs `d in keys(fixed)`:

```text
 r_1 = (1, 0)          x-translation
 r_2 = (0, 1)          y-translation
 r_3 = (-y, x)         rotation about the origin
```

so an x-component dof at `(x,y)` contributes `(1, 0, -y)` and a z-component dof
contributes `(0, 1, x)`.  A rigid mode is admissible (it vanishes on all Dirichlet
dofs, hence is in the range of `C_u`) **iff** its coefficient vector lies in the
kernel of `G`; the velocity of a solution is therefore unique iff `G` has rank 3.
`G` is positive semidefinite, so the solver's test is
`eigmin(G) > 1e-10 * eigmax(G)` (a relative threshold; the rank-deficient cases
of an axis-aligned box produce an eigenvalue at round-off, the well-posed ones
one of order the number of Dirichlet dofs).  `fixed` carries the *values* too,
but the uniqueness question is about the homogeneous linearised problem, so only
the dof set is used -- the rigid-mode rank test
"""
function _semquad_rigid_mode_gram(ss::SemStokes, fixed::Dict{Int,Float64})
    su = ss.su
    nu = su.N
    G = zeros(Float64, 3, 3)
    for k in keys(fixed)
        if k <= nu
            y = su.coords[k][2]
            m1 = 1.0; m2 = 0.0; m3 = -y
        else
            x = su.coords[k - nu][1]
            m1 = 0.0; m2 = 1.0; m3 = x
        end
        G[1, 1] += m1 * m1; G[1, 2] += m1 * m2; G[1, 3] += m1 * m3
        G[2, 1] += m2 * m1; G[2, 2] += m2 * m2; G[2, 3] += m2 * m3
        G[3, 1] += m3 * m1; G[3, 2] += m3 * m2; G[3, 3] += m3 * m3
    end
    return G
end

"""
    _semquad_check_rigid_modes(ss, fixed, resolved_wall_bc)

Refuse a wall configuration whose Dirichlet dof set leaves a rigid-body mode
admissible, with an actionable message naming the missing motion.  Called by
`_solve_semquad_stokes_var_core` exactly when some wall is `:open` (with every
wall closed the historical normal-Dirichlet data remove all three modes, so the
default path does not pay for this check or change its answer).

The test is the **numeric rank test** of [`_semquad_rigid_mode_gram`](@ref) on
the actual `fixed` dict: `eigmin(G) > 1e-10 * eigmax(G)`.  The eigenvector of
the smallest eigenvalue is the admissible rigid motion and is what the error
message reports, so the diagnosis cannot drift from the matrix it is based on.
(The same condition is checked combinatorially and earlier in `Params`, where a
non-`:open` x-wall blocks the x-translation and a non-`:open` y-wall the
y-translation, and either blocks the rotation; that cheap layer gives a card
user the error before any solve is attempted.)

The eigenvalue test is used rather than a Cholesky factorisation because a
singular `G` may come out with a tiny positive eigenvalue from round-off; the
relative threshold `1e-10` is far above that round-off and far below the
well-posed eigenvalues (order the number of Dirichlet dofs, measured for the
G262 configurations).  The null vector is reported as the admissible rigid motion
(normalised to its largest coefficient), so the message names the direction even
if a future mesh rule breaks the simple coverage argument.
"""
function _semquad_check_rigid_modes(ss::SemStokes,
                                    fixed::Dict{Int,Float64},
                                    resolved_wall_bc)
    G = _semquad_rigid_mode_gram(ss, fixed)
    E = eigen(Symmetric(G))
    lam = E.values
    lamax = lam[3]
    (isfinite(lamax) && lamax > 0.0 && lam[1] > 1.0e-10 * lamax) && return nothing
    # Rank-deficient: the eigenvector of the (near-)zero eigenvalue is the
    # admissible rigid motion.  Report it honestly instead of guessing from the
    # wall bookkeeping.
    v = E.vectors[:, 1]
    vmax = maximum(abs, v)
    if vmax > 0.0
        v ./= vmax
    else
        v .= (1.0, 0.0, 0.0)
    end
    names = ("x-translation", "y-translation", "rigid rotation")
    imax = argmax(abs.(v))
    dominant = names[imax]
    throw(ArgumentError("the wall configuration leaves a rigid-body mode " *
        "unconstrained: the 3x3 Gram matrix of the rigid modes " *
        "(x-translation, y-translation, rotation about the origin) restricted " *
        "to the Dirichlet dofs is singular (eigenvalues $(lam[1]), $(lam[2]), " *
        "$(lam[3]); the admissible rigid motion is " *
        "$(v[1]) * x-translation + $(v[2]) * y-translation + " *
        "$(v[3]) * rotation, i.e. the $dominant).  The velocity of a solution " *
        "is determined only up to that motion and the discrete saddle is " *
        "singular.  An :open wall is well posed only when the normal velocity " *
        "is Dirichlet on at least one x-wall and one y-wall (a non-:open wall " *
        "fixes that component along a whole line): set wall_bc = :free (or " *
        ":rigid) on enough walls to remove the printed motion.  Resolved " *
        "wall_bc = ($(join(String.(resolved_wall_bc), ","))).  Test: " *
        "_semquad_rigid_mode_gram, G262."))
end

# Historical box body of the Dirichlet dof selection, moved verbatim out of the
# core solve so the mapped geometry can have its own path (stage 0.5b2: the
# general curved-wall rows of `stokes/wall_constraints.jl`; the box statements,
# write order and corner overwrites are unchanged).  The statements, the write
# order and the corner overwrites are unchanged; only `n`/`nu` are recomputed
# from `su` (the same values the caller had).  This is the *only* structural
# change the box path receives in cut 2.
function _semquad_fill_fixed!(fixed::Dict{Int,Float64}, ::CartesianBox,
                              su::SemQuad, bmask::Vector{UInt8},
                              resolved_wall_bc, wall_u, wall_un,
                              uDx::Function, uDz::Function)
    n = su.p + 1
    nu = su.N
    for it in eachindex(su.leaves)
        lf = su.leaves[it]
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        fb = 4 * (it - 1)
        for dir in (:xminus, :xplus, :yminus, :yplus)
            # per-wall condition: :same follows the global bc; this is what
            # keeps the historical single-bc runs bit-identical
            wi = _wall_index(dir)
            bmask[fb + wi] != 0x00 || continue
            wbc = resolved_wall_bc[wi]
            # BC stage 3: an `:open` wall has **no** Dirichlet component -- both
            # components are free and the natural traction condition applies,
            # with `wall_t`/`wall_tn` as data.  It writes nothing, so every
            # legacy (:free/:rigid) `fixed` dict stays bit-identical.
            wbc === :open && continue
            rigid_wall = wbc === :rigid
            # per-wall normal value: `wall_un[wi] === nothing` reproduces the
            # historical global closure for that wall bit for bit (G258);
            # `wall_un[wi]` is the normal *component* in the positive
            # coordinate direction, exactly like `uDx`/`uDz`, not the outward
            # flux.  The write order (x walls, then y walls, normal before
            # tangential) is unchanged, so every corner overwrite of the legacy
            # paths is preserved.
            un = wall_un[wi]
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                xf = dir === :xplus ? x0 + hx : x0
                for j in 1:n
                    yv = y0 + 0.5 * (su.xi[j] + 1.0) * hy
                    g = uids[(j - 1) * n + iF]
                    fixed[g] = un === nothing ? uDx(xf, yv) : un(xf, yv)
                    tang = wall_u[wi]
                    rigid_wall && (fixed[nu + g] =
                                   tang === nothing ? uDz(xf, yv) : tang(xf, yv))
                end
            else
                jF = dir === :yminus ? 1 : n
                yf = dir === :yplus ? y0 + hy : y0
                for i in 1:n
                    xv = x0 + 0.5 * (su.xi[i] + 1.0) * hx
                    g = uids[(jF - 1) * n + i]
                    fixed[nu + g] = un === nothing ? uDz(xv, yf) : un(xv, yf)
                    tang = wall_u[wi]
                    rigid_wall && (fixed[g] =
                                   tang === nothing ? uDx(xv, yf) : tang(xv, yf))
                end
            end
        end
    end
    return fixed
end

"""
    _semquad_perimeter(su, bmask) -> Float64

Scale of the domain boundary used by the net-flux compatibility threshold.  The
box is the historical analytic expression `2*(Lx+1)` (GEOM_DESIGN E12) bit for
bit; a mapped geometry integrates its face Jacobians with [`perimeter`](@ref).
This is a cold path (only evaluated when some prescribed wall velocity is
non-zero) and only scales a round-off-level threshold, never an operator.
"""
@inline _semquad_perimeter(su::SemQuad, bmask::Vector{UInt8}) =
    _semquad_perimeter(su.geom, su, bmask)
@inline _semquad_perimeter(g::CartesianBox, su::SemQuad,
                           bmask::Vector{UInt8}) = perimeter(g)
@inline _semquad_perimeter(g::AbstractGeometry, su::SemQuad,
                           bmask::Vector{UInt8}) = perimeter(g, su, bmask)

# ---------------------------------------------------------------------------
# The velocities boundary preparation (stage 0.5b2)
# ---------------------------------------------------------------------------
# One function barrier, dispatched on the concrete geometry, replacing the
# cut-2 boundary block of the core solve.  The `CartesianBox` method is the
# historical body **verbatim** (the fixed-dof loop, the open-wall rigid-mode
# check, the net-flux compatibility check, the hanging constraints and the
# affine system) plus the no-wall sentinel; the mapped method builds the
# stage-0.5b2 general curved-wall rows (`stokes/wall_constraints.jl`), the
# mapped net-flux data integral and the merged affine system with offsets.
# The barrier exists for the box allocation contract: geometry-dependent
# branches and union-typed locals inside the 1 300-line core changed its
# codegen by +208 B per assembly in the A/B, while moving them behind a typed
# method boundary keeps the box path's compiled body identical in shape to the
# cut-2 code.  Both methods return the same concrete tuple
# `(fixed, cons2, u0, Cu, wd)`; `wd` is `_SEMQUAD_NO_WALLS` for the box.
function _semquad_velocity_boundary(ss::SemStokes, geom::CartesianBox,
                                    su::SemQuad, qt::Quadtree,
                                    bmask::Vector{UInt8}, resolved_wall_bc,
                                    wall_u, wall_un, uDx::Function,
                                    uDz::Function, fd::FEData, n::Int,
                                    nu::Int, Nu2::Int)
    fixed = Dict{Int,Float64}()
    _semquad_fill_fixed!(fixed, geom, su, bmask, resolved_wall_bc, wall_u,
                         wall_un, uDx, uDz)
    # BC stage 3: is there an `:open` wall?  It decides the pressure gauge (pin
    # only when *no* wall is open) and the well-posedness / net-flux checks.
    if any(w -> w === :open, resolved_wall_bc)
        # with an `:open` wall the Dirichlet set can shrink enough to leave a
        # rigid-body mode admissible; the numeric 3x3 Gram test runs on exactly
        # the dofs the solve will impose; the
        # all-`:open` pure-traction case was refused before the assembly).  With
        # every wall closed the historical normal Dirichlet data remove the
        # three modes, so the default path does not run this check.
        _semquad_check_rigid_modes(ss, fixed, resolved_wall_bc)
    else
    # Net-flux compatibility of the prescribed normal data (eq. 8).  When the
    # normal component is Dirichlet on
        # *every* wall (i.e. no wall is `:open`), the constant pressure is in the
        # null space of the reduced gradient and the pinned pressure gauge is
        # exact only when the boundary data satisfy 鈭垈惟 u路n dS = 0.  The default
        # card (driver closures `(x,y)->0.0`, every `wall_un` nothing) integrates
        # to zero exactly, so the `umax == 0` shortcut keeps the default path
        # free of extra arithmetic.
        netflux, wumax = _semquad_wall_normal_flux(ss, geom, fixed, fd, n, nu,
                                                   bmask)
        if wumax > 0.0
            # The box keeps the historical analytic `2*(Lx+1)` perimeter.
            perimeter = _semquad_perimeter(su, bmask)
            # The relative threshold is the design's `1e-12 * u_scale *
            # perimeter`, with one necessary correction: an **absolute floor** of
            # `1e-12` (in the solver's nondimensional velocity unit).  A
            # manufactured-solution gate (G33/G44/G150/G151/G154) prescribes the
            # exact field as its Dirichlet data; the *mathematical* net flux of
            # that data is zero, but its floating-point evaluation at the
            # boundary nodes is not (e.g. `sin(pi) = 1.2e-16`), so a strictly
            # data-scaled test would reject legitimate reference gates whose
            # imbalance is a round-off artefact and physically irrelevant.  The
            # floor only decides cases whose flux inconsistency is below 1e-12
            # of a unit velocity -- far below any accuracy gate in the suite --
            # and it does not weaken the user-facing check (a card's wall_un
            # values are O(1..1e3)).
            tol = 1.0e-12 * max(wumax * perimeter, 1.0)
            if abs(netflux) > tol
                throw(ArgumentError("the prescribed normal wall velocities have a " *
                    "net outward boundary flux of $(netflux) (threshold $(tol) = " *
                    "1e-12 * max(umax * perimeter, 1), umax = $(wumax)).  Every wall " *
                    "keeps a Dirichlet normal component, and the discrete continuity " *
                    "equations are consistent only when 鈭垈惟 u路n dS = 0.  " *
                    "`wall_un[i]` (and the legacy `uDx`/`uDz` closures) is the wall's " *
                    "normal *component* in the positive coordinate direction, so the " *
                    "outward flux is its face integral with the sign -1 on " *
                    "xminus/yminus and +1 on xplus/yplus; balance the inflow and the " *
                    "outflow (e.g. `wall_un = (U, U, 0, 0)` for a uniform " *
                    "through-flow)."))
            end
        end
    end
    cons2 = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in semquad_constraints(su, qt)
        cons2[c] = (ms, ws)
        cons2[nu + c] = (nu .+ ms, ws)
    end
    u0, Cu, free = _affine_constraint(Nu2, cons2, fixed)
    return fixed, cons2, u0, Cu, free, _SEMQUAD_NO_WALLS
end

function _semquad_velocity_boundary(ss::SemStokes, geom::AbstractGeometry,
                                    su::SemQuad, qt::Quadtree,
                                    bmask::Vector{UInt8}, resolved_wall_bc,
                                    wall_u, wall_un, uDx::Function,
                                    uDz::Function, fd::FEData, n::Int,
                                    nu::Int, Nu2::Int)
    fixed = Dict{Int,Float64}()
    cons_hang = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in semquad_constraints(su, qt)
        cons_hang[c] = (ms, ws)
        cons_hang[nu + c] = (nu .+ ms, ws)
    end
    wd = _semquad_mapped_wall_data(geom, su, bmask, resolved_wall_bc, wall_u,
                                   wall_un, uDx, uDz, cons_hang)
    for (k, v) in wd.fixed
        fixed[k] = v
    end
    if wd.gauge === :rotation
        LAST_SEMSTOKES_ROTATION_GAUGE[] = :rotation
    end
    if any(w -> w === :open, resolved_wall_bc)
        # The rigid-mode analysis ran inside the wall-row pass, which knows the
        # free-slip rows; the Cartesian-dof Gram test would false-refuse them.
    else
        # Net-flux compatibility of the *prescribed outward normal data*; on a
        # curved free-slip wall the nodal normal components are the data and the
        # tangential components are free, so the box's Cartesian component
        # integral is not the mapped data's flux.
        netflux, wumax = _semquad_wall_normal_flux(geom, su, bmask, wd.un, fd)
        if wumax > 0.0
            perimeter = _semquad_perimeter(su, bmask)
            tol = 1.0e-12 * max(wumax * perimeter, 1.0)
            if abs(netflux) > tol
                throw(ArgumentError("the prescribed normal wall velocities have a " *
                    "net outward boundary flux of $(netflux) (threshold $(tol) = " *
                    "1e-12 * max(umax * perimeter, 1), umax = $(wumax)).  Every wall " *
                    "keeps a Dirichlet normal component, and the discrete continuity " *
                    "equations are consistent only when 鈭垈惟 u路n dS = 0.  " *
                    "`wall_un[i]` is the wall's normal *velocity* (on a mapped " *
                    "geometry the outward-normal component `u.n`; on the box the " *
                    "normal component in the positive coordinate direction), and the " *
                    "flux above is its outward face integral; balance the inflow and " *
                    "the outflow (e.g. `wall_un = (U, U, 0, 0)` for a uniform " *
                    "through-flow, or the wall-frame data of a manufactured " *
                    "solution)."))
            end
        end
    end
    # Compose the hanging constraints with the wall rows (masters that the wall
    # rows constrain, and the wall rows of hanging wall nodes, are already
    # folded into the affine representation by the wall pass) and build
    # u = u0 + Cu u_free with the affine offsets of the wall rows.
    cons2, offsets = _semquad_merge_wall_constraints(cons_hang, wd)
    u0, Cu, free = _affine_constraint(Nu2, cons2, fixed, offsets)
    return fixed, cons2, u0, Cu, free, wd
end
