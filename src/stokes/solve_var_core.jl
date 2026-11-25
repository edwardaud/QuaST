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
actually imposes (docs/dev/BC_DESIGN.md §1.4).  The pass is serial and runs
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
the dof set is used -- the rigid-mode rank test of docs/dev/BC_DESIGN.md §3.5.
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
        "_semquad_rigid_mode_gram, docs/dev/BC_DESIGN.md §3.5, G262."))
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
            # paths is preserved (docs/dev/BC_DESIGN.md §1.4/§3.4).
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
        # the dofs the solve will impose (docs/dev/BC_DESIGN.md 搂3.5; the
        # all-`:open` pure-traction case was refused before the assembly).  With
        # every wall closed the historical normal Dirichlet data remove the
        # three modes, so the default path does not run this check.
        _semquad_check_rigid_modes(ss, fixed, resolved_wall_bc)
    else
        # Net-flux compatibility of the prescribed normal data (docs/dev/
        # BC_DESIGN.md 搂3.5 eq. 8).  When the normal component is Dirichlet on
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
    u0, Cu, _ = _affine_constraint(Nu2, cons2, fixed)
    return fixed, cons2, u0, Cu, _SEMQUAD_NO_WALLS
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
    u0, Cu, _ = _affine_constraint(Nu2, cons2, fixed, offsets)
    return fixed, cons2, u0, Cu, wd
end

function _solve_semquad_stokes_var_core(ss::SemStokes, qt::Quadtree,
                                  eta_fun::Function,
                                  fx::Function, fz::Function,
                                  uDx::Function, uDz::Function;
                                  mass_rho::Union{Nothing,Function}=nothing,
                                  mass_form::Symbol=:anelastic,
                                  mass_drho::Union{Nothing,Function}=nothing,
                                  # Density hook (`Params.rho_multiplier`, G248)
                                  # on the anelastic mass constraint: the
                                  # assembled density becomes
                                  # `mass_rho(y)*mass_rho_mult(x,y)` and its
                                  # gradient is differentiated consistently, so
                                  # `Bc` is the Galerkin form of
                                  # `div(mass_rho*mass_rho_mult*u) = 0`.
                                  # `nothing` (default) leaves `Bc` bit-identical.
                                  mass_rho_mult::Union{Nothing,Function}=nothing,
                                  compress_block::Union{Nothing,AbstractMatrix}=nothing,
                                  deviatoric::Bool=false,
                                  geo_u::Union{Nothing,Vector{Float64}}=nothing,
                                  geo_coef::Union{Nothing,Function}=nothing,
                                  geo_spd_safety::Float64=0.0,
                                  dp_u::Union{Nothing,Vector{Float64}}=nothing,
                                  dp_pfield::Union{Nothing,Vector{Float64}}=nothing,
                                  dp_coef::Union{Nothing,Function}=nothing,
                                  elastic_stress::Union{Nothing,NTuple{3,Vector{Float64}}}=nothing,
                                  bc::Symbol=:rigid, linear::Symbol=:direct,
                                  wall_bc::NTuple{4,Symbol}=(:same, :same, :same, :same),
                                  wall_u::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
                                  wall_un::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
                                  # BC stage 2: per-wall tangential *traction*,
                                  # in the same `(xminus, xplus, yminus, yplus)`
                                  # order; `nothing` = zero (no load).  The
                                  # value is the global traction component in
                                  # the positive coordinate direction (`t_z` on
                                  # an x-wall, `t_x` on a y-wall) and it is a
                                  # data term on the free side, so it needs a
                                  # wall with a free tangential component
                                  # (`:free`; `:open` in stage 3).
                                  wall_t::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
                                  # BC stage 3: per-wall *normal* traction, same
                                  # order; `nothing` = zero (the natural
                                  # condition of an `:open` wall).  The value is
                                  # the global traction component in the positive
                                  # coordinate direction (`t_x` on an x-wall,
                                  # `t_z` on a y-wall), i.e. the **outward**
                                  # normal is already in `tbar = sigma.n`: a
                                  # pressure p0 on an open boundary is
                                  # `-p0` on the top wall and `+p0` on the
                                  # bottom.  It carries the pressure, so it
                                  # needs an `:open` wall (free normal velocity)
                                  # and it forces the derived pressure gauge.
                                  wall_tn::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
                                  stokes_accept_tol::Float64=Inf,
                                  stokes_lost_tol::Float64=SEMSTOKES_LOST_TOL,
                                  stokes_scaling::Symbol=:none,
                                  schur_maxiter::Int=200, schur_precond::Symbol=:mass,
                                  mg_cycles::Int=1, mg_tol::Float64=1.0e-10,
                                  mg_pressure_scale::Float64=0.0,
                                  mg_pressure_weight::Symbol=:inv_eta,
                                  mg_pressure_mode::Symbol=:auto,
                                  mg_pressure_precond::Symbol=:auto,
                                  mg_pressure_cache::Bool=true,
                                  mg_pressure_eta_tol::Float64=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
                                  mg_pressure_bfbt_d::Symbol=:auto,
                                  mg_pressure_bfbt_s2::Symbol=:free,
                                  mg_pressure_bfbt_s1::Symbol=:chol,
                                  mg_pressure_bfbt_s1_solve::Symbol=:chol,
                                  mg_pressure_bfbt_s1_tol::Float64=1.0e-2,
                                  mg_pressure_bfbt_s1_maxit::Int=50,
                                  mg_pressure_bfbt_s1_minwork::Int=0,
                                  mg_pressure_bfbt_s1_chunk::Int=0,
                                  mg_smoother::Symbol=:auto,
                                  mg_coarse::Symbol=:auto,
                                  mg_coarsen_layers::Int=1,
                                  mg_coarsen_factor::Union{Nothing,Float64}=nothing,
                                  mg_coarse_direct_max::Int=0,
                                  mg_smoothing::Symbol=:auto,
                                  mg_pre::Union{Integer,AbstractVector{<:Integer}}=2,
                                  mg_post::Union{Integer,AbstractVector{<:Integer}}=2,
                                  mg_omega::Union{Nothing,Float64}=nothing,
                                  mg_operator::Symbol=:matrixfree,
                                  mg_vel_solver::Symbol=:vcycle,
                                  mg_vel_tol::Float64=1.0e-1,
                                  mg_vel_maxit::Int=10,
                                  mg_restart::Int=100, mg_maxit::Int=500,
                                  mg_threads::Int=1,
                                  mg_diag::Symbol=:constrained,
                                  assemble_threads::Int=1,
                                  return_system::Bool=false,
                                  assemble_only::Union{Bool,Symbol}=false)
    bc in (:free, :rigid) || throw(ArgumentError("bc must be :free or :rigid"))
    all(w -> w in (:same, :free, :rigid, :open), wall_bc) ||
        throw(ArgumentError("wall_bc entries must be :same, :free, :rigid or " *
                            ":open"))
    # Stage 1 validation (G258 (e)): `wall_u` is a *tangential* value and only a
    # `:rigid` wall applies it; on a `:free` wall the tangential component is
    # traction-free and on an `:open` wall there is no Dirichlet component at
    # all, so the value would be a silent no-op.  `Params` refuses the same
    # configuration, but a direct library call bypasses it -- hence the
    # duplicate check here.
    #
    # BC stage 3: `wall_un` is a *normal velocity* (Dirichlet data) and does not
    # exist on an `:open` wall (its normal component is free); `wall_t` needs a
    # free tangential component (`:free`/`:open`) and `wall_tn` needs the free
    # normal component of an `:open` wall (see `_semquad_validate_wall_tnormal`).
    #
    # The *resolved* wall tuple (`:same` -> `bc`) is built once and reused by the
    # fixed-dof loop, the validation, the well-posedness test and the load pass;
    # the resolution is pure, so every legacy `fixed` dict is unchanged.
    resolved_wall_bc = ntuple(i -> wall_bc[i] === :same ? bc : wall_bc[i], 4)
    for i in 1:4
        wbc = resolved_wall_bc[i]
        wall_u[i] === nothing || wbc === :rigid ||
            throw(ArgumentError("wall_u[$i] sets the tangential velocity of a " *
                                "wall that resolves to :$wbc; " *
                                (wbc === :open ?
                                 "an :open wall has no Dirichlet component (the " *
                                 "natural traction condition applies to both " *
                                 "components)" :
                                 "only a :rigid wall has a Dirichlet " *
                                 "tangential component") *
                                ", so the value would be silently ignored.  Set " *
                                "wall_bc[$i] = :rigid (or `bc = :rigid`), or " *
                                "drop wall_u[$i]."))
        wall_un[i] === nothing || wbc !== :open ||
            throw(ArgumentError("wall_un[$i] sets the normal velocity of a " *
                                "wall that resolves to :open, which has no " *
                                "Dirichlet component: its normal velocity is an " *
                                "unknown and the natural (zero unless wall_tn) " *
                                "traction applies, so the value would be " *
                                "silently ignored.  Set wall_bc[$i] = :free (or " *
                                ":rigid) to prescribe a normal velocity, or use " *
                                "wall_tn[$i] for a normal traction on the open " *
                                "wall, or drop wall_un[$i]."))
    end
    # Stage 2/3: `wall_t` is a tangential *traction*, i.e. a data term on the free
    # side, so its wall's tangential component must be free (`:free` or `:open`).
    # On a `:rigid` wall the tangential component is Dirichlet and `Cu'` would
    # drop the assembled load: a silent no-op, refused here and in `Params`.
    # `wall_tn` is the *normal* traction and needs the free normal component of
    # an `:open` wall; both validators are shared with `Params`.
    _semquad_validate_wall_traction(wall_t, resolved_wall_bc)
    _semquad_validate_wall_tnormal(wall_tn, resolved_wall_bc)
    # BC stage 3: the well-posedness refusal that needs no mesh -- all four walls
    # `:open` is the pure traction problem (velocity determined only up to a
    # rigid motion, pressure not fixed by the boundary, and a net force/torque
    # compatibility condition the data generally violate).  The *other* refused
    # configuration (a rigid mode left admissible by an open pair) is diagnosed
    # after the fixed-dof set is built, where the actual Dirichlet dofs are
    # known; `_semquad_check_rigid_modes`.
    all(w -> w === :open, resolved_wall_bc) &&
        throw(ArgumentError("all four walls are :open.  That is the pure " *
                            "traction problem: with the natural condition " *
                            "sigma.n = 0 on the whole boundary the velocity is " *
                            "determined only up to a rigid motion (and the data " *
                            "must satisfy zero net force and torque " *
                            "compatibility), the discrete saddle is singular and " *
                            "the pressure is not fixed by the boundary.  Leave " *
                            "at least one x-wall (xminus/xplus) and one y-wall " *
                            "(yminus/yplus) with a Dirichlet normal component " *
                            "(wall_bc = :free or :rigid on those walls)."))
    stokes_scaling in SEMSTOKES_SCALINGS ||
        throw(ArgumentError("stokes_scaling must be :none or :equilibrated"))
    (geo_u === nothing) == (geo_coef === nothing) ||
        throw(ArgumentError("geo_u and geo_coef must be given together"))
    if elastic_stress !== nothing
        # 2b step 2: `elastic_stress` is the previous step's deviatoric stress
        # (already relaxed by `a = eta/(eta+G dt)`), a nodal Q2 tensor field on
        # the velocity space.  It enters the load vector as `-int tau:eps(v)`;
        # all three components must live on the velocity nodes.
        for i in 1:3
            length(elastic_stress[i]) == ss.su.N ||
                throw(ArgumentError("elastic_stress component $i has " *
                                    "$(length(elastic_stress[i])) entries, " *
                                    "the velocity space has $(ss.su.N)"))
        end
    end
    if geo_u !== nothing
        length(geo_u) == 2 * ss.su.N ||
            throw(ArgumentError("geo_u has $(length(geo_u)) entries, expected $(2 * ss.su.N)"))
        deviatoric &&
            throw(ArgumentError("the geometric strain-rate Jacobian is only " *
                                "implemented for deviatoric=false"))
    end
    # Pressure-derivative (Drucker-Prager) Jacobian block `Kp`: the momentum
    # residual is `K(eta(T, edot(u), p)) u - B'p - f`, so its pressure
    # derivative is
    #     (Kp)_{i,a} = int 2 (deta/dp) eps(u):eps(v_i) phi_a
    # in the velocity rows and nothing in the constraint rows.  The resulting
    # saddle system is non-symmetric (velocity rows <-> pressure columns), which
    # the direct LU handles.  `dp_u` is the velocity whose strain rate enters
    # `deta/dp`, `dp_pfield` the nodal pressure field, `dp_coef(x,y,edot,p)` the
    # derivative itself (e.g. `semquad_deta_dp` wrapped with the temperature).
    if dp_u !== nothing || dp_pfield !== nothing || dp_coef !== nothing
        (dp_u !== nothing && dp_pfield !== nothing && dp_coef !== nothing) ||
            throw(ArgumentError("dp_u, dp_pfield and dp_coef must be given " *
                                "together"))
        length(dp_u) == 2 * ss.su.N ||
            throw(ArgumentError("dp_u has $(length(dp_u)) entries, expected $(2 * ss.su.N)"))
        length(dp_pfield) == ss.sp.N ||
            throw(ArgumentError("dp_pfield has $(length(dp_pfield)) entries, " *
                                "expected $(ss.sp.N)"))
        deviatoric &&
            throw(ArgumentError("the pressure-derivative Jacobian block is " *
                                "only implemented for deviatoric=false"))
    end
    linear in (:direct, :schur_cg, :mg, :lagged, :auto) ||
        throw(ArgumentError("linear must be :direct, :schur_cg, :mg, :lagged " *
                            "or :auto"))
    # BC stage 3: an `:open` wall removes the nodal pressure pin (the constant
    # pressure is no longer in the null space of the reduced gradient, so pinning
    # one dof would replace a continuity equation by `p_a = 0`, i.e. solve a
    # different problem).  The direct path (with its shift ladder), the
    # multigrid path and the `:auto` selector are verified without the pin;
    # `:schur_cg` and `:lagged` carry their own gauge/scaling assumptions and are
    # refused on an open-wall configuration rather than answered wrongly.
    if any(w -> w === :open, resolved_wall_bc) &&
       (linear === :schur_cg || linear === :lagged)
        throw(ArgumentError("linear=:$linear is not supported with an :open " *
                            "wall: the derived pressure gauge removes the nodal " *
                            "pin, so the reduced saddle is a different " *
                            "(unpinned) problem and the :schur_cg/:lagged paths " *
                            "assume the closed-boundary gauge.  Use " *
                            "linear=:direct (with its shift ladder), linear=:mg " *
                            "or linear=:auto -- all three are verified for open " *
                            "walls (G263)."))
    end
    (assemble_only === false || assemble_only === true || assemble_only === :mg) ||
        throw(ArgumentError("assemble_only must be false (solve), true (assemble " *
                            "everything) or :mg (the reduced matrix-free MG " *
                            "geometry: no velocity stiffness K, no Kred)"))
    mg_cycles >= 1 || throw(ArgumentError("mg_cycles must be >= 1"))
    mg_tol > 0.0 || throw(ArgumentError("mg_tol must be > 0"))
    mg_pressure_scale >= 0.0 ||
        throw(ArgumentError("mg_pressure_scale must be >= 0 (0 = auto)"))
    mg_pressure_weight in (:inv_eta, :eta, :none) ||
        throw(ArgumentError("mg_pressure_weight must be :inv_eta, :eta or :none"))
    mg_pressure_mode in (:auto, :mass, :stiffness, :lsc, :bfbt) ||
        throw(ArgumentError("mg_pressure_mode must be :auto, :mass, :stiffness, :lsc or :bfbt"))
    mg_pressure_precond in (:auto, :bfbt, :mass, :scaled_mass, :pcd, :lsc) ||
        throw(ArgumentError("mg_pressure_precond must be :auto, :bfbt, :mass, " *
                            ":scaled_mass, :pcd or :lsc"))
    mg_pressure_eta_tol > 0.0 ||
        throw(ArgumentError("mg_pressure_eta_tol must be > 0 (Inf = never " *
                            "rebuild on eta drift; a tree change still rebuilds)"))
    mg_pressure_bfbt_d in (:auto, :leaf, :exact, :matrix) ||
        throw(ArgumentError("mg_pressure_bfbt_d must be :auto, :leaf, :exact " *
                            "or :matrix"))
    mg_pressure_bfbt_s2 in (:free, :matrix) ||
        throw(ArgumentError("mg_pressure_bfbt_s2 must be :free or :matrix"))
    mg_pressure_bfbt_s1 in (:chol, :mass) ||
        throw(ArgumentError("mg_pressure_bfbt_s1 must be :chol or :mass"))
    mg_pressure_bfbt_s1_solve in (:chol, :level, :cg_diag, :cg_ichol) ||
        throw(ArgumentError("mg_pressure_bfbt_s1_solve must be :chol, :level, " *
                            ":cg_diag or :cg_ichol"))
    mg_pressure_bfbt_s1_tol > 0.0 ||
        throw(ArgumentError("mg_pressure_bfbt_s1_tol must be > 0"))
    mg_pressure_bfbt_s1_maxit >= 1 ||
        throw(ArgumentError("mg_pressure_bfbt_s1_maxit must be >= 1"))
    mg_smoother in (:auto, :jacobi, :sgs) ||
        throw(ArgumentError("mg_smoother must be :auto, :jacobi or :sgs"))
    mg_coarse in (:auto, :rediscretize, :galerkin) ||
        throw(ArgumentError("mg_coarse must be :auto, :rediscretize or :galerkin"))
    mg_coarsen_layers >= 1 ||
        throw(ArgumentError("mg_coarsen_layers must be >= 1"))
    (mg_coarsen_factor === nothing ||
     (mg_coarsen_factor >= 1.0 && isfinite(mg_coarsen_factor))) ||
        throw(ArgumentError("mg_coarsen_factor must be >= 1 (or `nothing` = " *
                            "the matrix-free production default 1.5)"))
    mg_coarse_direct_max >= 0 ||
        throw(ArgumentError("mg_coarse_direct_max must be >= 0"))
    mg_smoothing in (:auto, :uniform, :taper) ||
        throw(ArgumentError("mg_smoothing must be :auto, :uniform or :taper"))
    _mg_sw_ok(v) = v isa Integer ? v >= 0 : all(x -> x >= 0, v)
    _mg_sw_ok(mg_pre) || throw(ArgumentError("mg_pre must be >= 0"))
    _mg_sw_ok(mg_post) || throw(ArgumentError("mg_post must be >= 0"))
    (mg_omega === nothing || 0.0 < mg_omega < 1.0) ||
        throw(ArgumentError("mg_omega must be in (0,1) or `nothing`"))
    mg_operator in (:sparse, :matrixfree) ||
        throw(ArgumentError("mg_operator must be :sparse or :matrixfree"))
    # Production default of the h-coarsening step.  `coarsen_factor=2.0` was
    # measured on the F tree (1.5x end to end, docs/dev/MG.md) but on
    # P1 it collapses the hierarchy from 8 to 6 levels and costs 177 outer
    # iterations instead of 61 (8.03 s against 5.45 s, measured at 1e-6).  The
    # shipped default is therefore 1.5: it still removes F's 1.7 %-coarsening
    # near-duplicate level (F: 8 -> 6 levels) but keeps P1's 40 776-dof level
    # (P1: 8 -> 7 levels), and wins on both states (F 5.42 -> 3.36 s,
    # P1 5.45 -> 5.12 s at one thread).  `mg_coarsen_factor=1.0` restores the
    # historical rule and `2.0` remains selectable; uniform/blob trees are a
    # bitwise no-op at 1.5 (their steps already coarsen by >=2x).
    coarsen_factor = mg_coarsen_factor === nothing ?
                     (mg_operator === :matrixfree ? 1.5 : 1.0) :
                     mg_coarsen_factor
    mg_vel_solver in (:cg, :vcycle) ||
        throw(ArgumentError("mg_vel_solver must be :cg or :vcycle"))
    mg_vel_tol > 0.0 || throw(ArgumentError("mg_vel_tol must be > 0"))
    mg_vel_maxit >= 1 || throw(ArgumentError("mg_vel_maxit must be >= 1"))
    mg_restart >= 5 || throw(ArgumentError("mg_restart must be >= 5"))
    mg_maxit >= 1 || throw(ArgumentError("mg_maxit must be >= 1"))
    mg_threads >= 1 || throw(ArgumentError("mg_threads must be >= 1"))
    mg_diag in (:constrained, :lumped) ||
        throw(ArgumentError("mg_diag must be :constrained or :lumped"))
    mass_form in (:anelastic, :boussinesq) ||
        throw(ArgumentError("mass_form must be :anelastic or :boussinesq"))
    schur_precond in (:none, :mass) ||
        throw(ArgumentError("schur_precond must be :none or :mass"))
    assemble_threads >= 1 ||
        throw(ArgumentError("assemble_threads must be >= 1"))
    # S6: resolve `linear=:auto` **before** the assembly, so an MG decision
    # never pays for or stores the sparse velocity operator.  The decision uses
    # only O(1) node counts and the fixed 33x33-grid eta sample (measured cost
    # in docs/dev/SOLVER.md §20); an explicit `:direct`/`:mg` call skips this
    # block entirely and runs exactly as before (bitwise for `:direct`).
    if linear === :auto
        auto_choice, auto_threads = _semquad_resolve_auto(ss, eta_fun;
                                                          schur_precond=schur_precond)
        linear = auto_choice
        if linear === :mg
            # The selector is calibrated on the matrix-free operator, and
            # `mg_threads=min(8, Threads.nthreads())` (S5-2 measured the gain).
            mg_operator = :matrixfree
            mg_threads = auto_threads
        end
    else
        # An `assemble_only` call never solves, so it must not overwrite the
        # `:auto` decision of the last real solve: the MG hierarchy builder and
        # the caller/Newton probes assemble with `linear=:direct, assemble_only=true`
        # *inside* an auto MG solve, and recording `:none` there would hide the
        # selector's decision (it broke the auto diagnostics until S7).
        assemble_only !== false ||
            (LAST_SEMSTOKES_AUTO_CHOICE[] =
                 (choice=:none,
                  reason=get(_SEMSTOKES_AUTO_OFF_REASONS, linear,
                             "linear not :auto (auto not requested)"),
                  saddle_dofs=0, contrast=1.0, samples=0, sample_ms=0.0,
                  mg_threads=0))
    end
    # ------------------------------------------------------------------
    # Stage 0 cut 5: the mapped-geometry linear-path restriction of the solve.
    #
    # Cut 1 created the layer, cut 2 routed the assembled Stokes operator, cuts 3
    # and 4 the energy/step/diagnostics; cut 5 routes the matrix-free operator
    # and the multigrid hierarchy.  A mapped geometry is therefore accepted for
    # the **linear** paths `:direct`, `:auto` and `:mg` (`:auto` is resolved above
    # and becomes one of the two; on a mapped space it decides from the mapped
    # eta contrast).  `:schur_cg` and `:lagged` still carry box reference-matrix
    # / gauge assumptions and are refused with an actionable message.
    #
    # The remaining mapped restrictions below are *feature* restrictions owned by
    # later cuts (strain-rate Jacobian, wall BC kinds), not by the linear solver;
    # they are kept because their elements still run the box-only leaf kernel.
    # ------------------------------------------------------------------
    geom = ss.su.geom
    if !(geom isa CartesianBox)
        gname = geometry_name(geom)
        if !(linear === :direct || linear === :mg)
            throw(ArgumentError("a mapped geometry ($gname) supports " *
                "linear=:direct, :auto and :mg (resolved linear = :$linear): " *
                ":schur_cg carries the box reference-matrix Schur assumption " *
                "and :lagged the box stale-LU gauge; both stay box-only in " *
                "stage 0.  Pass linear=:direct, linear=:mg or linear=:auto."))
        end
        if linear === :mg && schur_precond !== :mass
            throw(ArgumentError("a mapped geometry ($gname) with linear=:mg " *
                "needs schur_precond=:mass: the pressure-block preconditioner " *
                "is built from the mapped pressure mass / w-BFBT factors, and " *
                "the :none Schur path has no mapped equivalent.  Pass " *
                "schur_precond=:mass (the default)."))
        end
        (geo_u === nothing && dp_u === nothing && elastic_stress === nothing) ||
            throw(ArgumentError("a mapped geometry ($gname) does not support " *
                "the strain-rate Jacobian (geo_u), the pressure-derivative " *
                "block (dp_u) or the elastic force (elastic_stress) in cut 2: " *
                "their elements still run the box-only leaf kernel.  Use a " *
                "CartesianBox geometry for those."))
        # Stage 0.5b2: `:free` walls and `wall_un` are supported on a mapped
        # geometry (the general curved-wall rows of
        # `stokes/wall_constraints.jl`); the wall-frame semantics are
        # `wall_un = u.n` and `wall_u = u.tau`, documented in that file.  No
        # per-wall-kind restriction remains here.
        mass_rho_mult === nothing ||
            throw(ArgumentError("a mapped geometry ($gname) does not support " *
                "the Cartesian density multiplier (mass_rho_mult): its " *
                "gradient is a finite-difference hook clamped to [0,1]^2, " *
                "which has no mapped meaning.  Use a height-dependent " *
                "reference state (mass_rho/mass_drho) instead."))
    end
    # BFBT weight: the exact restricted diagonal
    # `diag(Cu' M_u(sqrt(eta)) Cu)` is the shipped default on the matrix-free
    # path (docs/dev/MG.md).  It is a zero-per-apply-cost change
    # that removes 10 outer iterations on `F` (81 -> 71) and leaves `P1` at 65;
    # its leaf-local construction is cheaper than the historical lumped `:leaf`
    # diagonal at 8 threads and the same at 1 (both are dominated by the
    # leaf/quadrature-point eta evaluations).  `:leaf` is kept as the A/B and
    # `:matrix` as the sparse reference for G226; the sparse-operator MG path
    # keeps `:leaf` because it is not the measured production path.  Resolved
    # after the `linear=:auto` block, which may switch `mg_operator`.
    bfbt_d = mg_pressure_bfbt_d === :auto ?
             (mg_operator === :matrixfree ? :exact : :leaf) :
             mg_pressure_bfbt_d
    # `assemble_only=:mg` is the *reduced* assembly of the matrix-free MG
    # geometry: the caller (the hierarchy builder) only reads `Cu`, so the
    # velocity stiffness `K`, the COO triplets behind it and `Kred` are not
    # formed at all.  It is defined only for the matrix-free MG path (the sparse
    # operator path and the direct path need `K`/`Kred`); it is opt-in and
    # returns the same `Cu`/`Cp`/`B`/`fv` bit for bit.
    kfree = assemble_only === :mg
    kfree && return_system &&
        throw(ArgumentError("assemble_only=:mg does not form K/Kred, so it is " *
                            "incompatible with return_system=true"))
    kfree && !(linear === :mg && mg_operator === :matrixfree) &&
        throw(ArgumentError("assemble_only=:mg is the reduced matrix-free MG " *
                            "assembly; it requires linear=:mg with " *
                            "mg_operator=:matrixfree (got linear=$linear, " *
                            "mg_operator=$mg_operator)"))
    LAST_SEMSTOKES_VEL_CG_ITERS[] = 0
    p = ss.p; q = ss.q
    su = ss.su; sp = ss.sp
    nu = su.N; np = sp.N
    Nu2 = 2 * nu
    fd = make_fedata(p, q)
    n = p + 1
    nv = n * n
    npq = (q + 1)^2
    # S3 stage C: `linear=:mg, mg_operator=:matrixfree` never needs the sparse
    # velocity operator.  The right-hand side and the outer FGMRES operator are
    # built from the leaf-local apply (`semquad_apply_K!` / `_apply_Kred!`), so
    # the `2n^2 x 2n^2` dense leaf blocks and their `4 nv^2` triplets are never
    # formed.  Everything that still needs `K` (return_system, the Newton
    # geometric/pressure-derivative blocks) keeps it, and so does a plain
    # `assemble_only=true`; only the reduced `assemble_only=:mg` geometry call
    # joins the solve path in the `K`-free branch.  On that branch the leaf
    # assembly also skips the viscosity closure (its `etaq` only enters `K` and
    # the two Jacobian blocks), which is 0.15 s of the F setup at one thread --
    # see `SemQuadAsmCfg.mf_path`.
    mf_path = linear === :mg && mg_operator === :matrixfree &&
              (assemble_only === false || kfree) &&
              !return_system && geo_u === nothing && dp_u === nothing
    pleaves = Set(sp.leaves)
    # BC stage 2: the domain-boundary flag of every leaf face, precomputed
    # **serially** (the quadtree neighbour table is not thread-safe and must not
    # be queried from the threaded leaf assembly).  The fixed-dof loop, the
    # net-flux check and the traction load pass all read this mask instead of
    # asking `has_no_face_neighbor`, which is exactly the same per-face boolean
    # they asked for before, so no quantity changes (G259/G260).
    bmask = _semquad_boundary_face_mask(qt, su)
    # Deviatoric stress 2 eta (eps - (1/3) div(u) I): the volumetric part of
    # 2 eta eps:eps(v) loses 2/3 eta div(u) div(v).  ASPECT's ALA always uses
    # this form (manual eq. 20); without it a compressible (div u = Di u_z)
    # flow is too stiff.  `deviatoric=false` keeps the historical operator.
    a2 = deviatoric ? 4.0 / 3.0 : 2.0
    # S4 stage 1: leaf assembly.  `assemble_threads=1` (default) is the
    # historical serial path, bit for bit; >1 cuts the leaves into fixed
    # contiguous blocks, assembles them in parallel and merges the block COO
    # buffers in block order.  The threaded result is deterministic run to run
    # but is *not* bitwise equal to the serial one (different summation order),
    # which is why the default stays 1.
    asm_cfg = SemQuadAsmCfg(a2, deviatoric, mf_path, geo_u, geo_coef,
                            geo_spd_safety, dp_u, dp_pfield, dp_coef,
                            elastic_stress)
    # Cut 2: dispatch on the concrete geometry here (the guard block above has
    # already read it into `geom`).  No forwarding method is in this hot path:
    # the 9-argument `_semquad_assemble_blocks` shim exists for the G181/G188
    # call sites, and going through it costs one constant 560 B dispatch frame
    # per assembly (measured; 0 B per leaf) — avoided on the production path.
    asm = _semquad_assemble_blocks(
        geom, _semquad_asm_thread_count(assemble_threads), asm_cfg, su, sp,
        eta_fun, fx, fz, fd, pleaves)
    IK = asm.IK; JK = asm.JK; VK = asm.VK
    IB = asm.IB; JB = asm.JB; VB = asm.VB
    IG = asm.IG; JG = asm.JG; VG = asm.VG
    IP = asm.IP; JP = asm.JP; VP = asm.VP
    fv = asm.fv
    # BC stage 2/3: prescribed traction (tangential `wall_t` and, on `:open`
    # walls, normal `wall_tn`).  RHS only: the load is added
    # to the **unreduced** `fv` (never to `fr`, and never to `u0`, which is
    # Dirichlet data) so that `fr = Cu' (fv - K u0 + Bm' p0)` below is the exact
    # virtual work on the admissible test space; `Cu'` redistributes a loaded
    # hanging-master node and drops a loaded Dirichlet dof, which is the
    # weak-form restriction (docs/dev/BC_DESIGN.md §3.2/§3.3).  With every
    # `wall_t`/`wall_tn` entry `nothing` this call returns before touching `fv`,
    # which is what keeps the default path bit-identical (G260).  The pass runs
    # after the threaded leaf assembly and touches no operator (`K`, `Kred`,
    # `B`, the matrix-free path and the multigrid hierarchy never see it).
    _semquad_boundary_traction!(fv, geom, su, bmask, fd, wall_t, wall_tn,
                                resolved_wall_bc)

    if mf_path
        K = nothing
    else
        SEMSTOKES_K_ASSEMBLIES[] += 1
        K = sparse(IK, JK, VK, Nu2, Nu2)
    end
    Kgeo = geo_u === nothing ? nothing : sparse(IG, JG, VG, Nu2, Nu2)
    Kp = dp_u === nothing ? nothing : sparse(IP, JP, VP, Nu2, np)
    Bref = sparse(IB, JB, VB, np, Nu2)
    # Mass constraint with a stratified reference density.  ASPECT's ALA mass
    # equation is div(rho u) = 0, whose Galerkin form is
    #     int q (rho div u + u_z drho/dy) dV = 0,
    # **not** `int q rho div u = 0`: the latter only enforces div u = 0 and, with
    # upwellings, breaks the exact cancellation between the advection of the
    # reference state and the adiabatic heating term (King 2010 then cools and
    # its energy budget does not close).  The momentum pressure term stays the
    # unweighted -int p div v, so Bm = Bref while Bc carries the density.
    # `mass_form=:boussinesq` is the **TALA** continuity equation `div u = 0`:
    # the reference density enters the buoyancy only, not the mass conservation
    # (ASPECT's `truncated anelastic liquid approximation`).  It is the
    # apples-to-apples TALA counterpart of the ALA form for the same reference
    # state, which is what docs/dev/PHYSICS.md §10.3 measures; it is off unless
    # requested, so the default path is unchanged.
    Bm = Bref
    Bc = Bref
    if mass_rho !== nothing && mass_form === :anelastic
        # derivative of the density profile at the quadrature points
        drho = mass_drho === nothing ? (y -> begin
            hd = 1.0e-6
            yp = min(y + hd, 1.0); ym = max(y - hd, 0.0)
            (mass_rho(yp) - mass_rho(ym)) / (yp - ym)
        end) : mass_drho
        Bc = semquad_stokes_B(ss, qt; rho_fun=mass_rho, drho_fun=drho,
                              rho_mult=mass_rho_mult)
        linear === :schur_cg &&
            throw(ArgumentError("mass_form=:anelastic makes the saddle system " *
                                "nonsymmetric (Bm != Bc'); use linear=:direct"))
    end

    # Stage 0.5b2: the velocity gauge of the previous solve is stale; every
    # solve resolves it again (`:none` on the box and on every configuration
    # whose wall rows exclude the rigid-body modes).  Two Ref writes, no
    # allocation and no arithmetic on the box path.
    LAST_SEMSTOKES_ROTATION_GAUGE[] = :none
    LAST_SEMSTOKES_ROTATION_SHIFT[] = 0.0
    # The boundary preparation is a **function barrier dispatched on the
    # concrete geometry** (see `_semquad_velocity_boundary`): the box method is
    # the cut-2 boundary block verbatim and the mapped method builds the general
    # curved-wall rows (`stokes/wall_constraints.jl`: the free-slip `n.u = g`
    # lines, the wall-frame Dirichlet data, the corner order, the rigid-mode
    # analysis and the rotation gauge).  Keeping the geometry branches behind a
    # typed method boundary keeps the box path's codegen and allocation at the
    # cut-2 values (the A/B measured +208 B per assembly when the branches were
    # in the core body).
    fixed, cons2, u0, Cu, wd = _semquad_velocity_boundary(
        ss, geom, su, qt, bmask, resolved_wall_bc, wall_u, wall_un, uDx, uDz,
        fd, n, nu, Nu2)
    # BC stage 3: is there an `:open` wall?  It decides the pressure gauge (pin
    # only when *no* wall is open; the net-flux/well-posedness checks ran in the
    # boundary preparation above).
    any_open = any(w -> w === :open, resolved_wall_bc)
    cons_p = semquad_constraints(sp, qt)
    # BC stage 3: the pressure gauge is **derived**, not unconditional.  The nodal
    # pin (one pressure dof fixed to zero, its continuity equation dropped) is
    # exact only when the constant pressure is in the null space of the reduced
    # gradient, i.e. when ∫_Ω div v = ∫_∂Ω v·n = 0 for every admissible test
    # function -- true **iff** the normal velocity is Dirichlet on the whole
    # boundary.  With any `:open` wall a boundary test function with non-zero
    # normal trace exists, the constant is NOT in the null space, and pinning one
    # dof plus dropping its continuity row would solve a different problem
    # (p_a = 0 in place of one continuity equation).  The rule is therefore
    # mechanical: pin iff no wall is `:open`; with every wall closed this is
    # bit-identical to the historical code.  `LAST_SEMSTOKES_PRESSURE_GAUGE`
    # records the resolved decision (`:pin` / `:none`) for G262, and the returned
    # `C_p` carries `np - 1` columns when pinned and `np` when not (its column
    # count is the second, structural observation of the same decision).
    pf = Dict{Int,Float64}()
    if any_open
        LAST_SEMSTOKES_PRESSURE_GAUGE[] = :none
    else
        pfree0 = setdiff(1:np, collect(keys(cons_p)))
        isempty(pfree0) || (pf[first(pfree0)] = 0.0)
        LAST_SEMSTOKES_PRESSURE_GAUGE[] = :pin
    end
    p0, Cp, _ = _affine_constraint(np, cons_p, pf)

    CpB = Cp' * Bc * Cu
    # S3 stage C: on the fully matrix-free MG path there is no sparse `K` and no
    # `Kred = Cu' K Cu`.  `fr = Cu' (fv - K u0 + Bm' p0)` needs `K u0` on the
    # *full* velocity vector, which the leaf-local apply gives directly, and the
    # outer operator becomes `Cu' K (Cu v)` (`_semquad_apply_Kred!`).
    mfop = nothing
    mfapplyK = nothing
    mfpar = nothing
    if mf_path
        mfop = semquad_kop(ss, qt; deviatoric=deviatoric)
        # sampled once, then laid out for the lane kernel (docs/dev/MG.md
        # §15): the leaf-major field plus its block-major lane table
        mfetav = semquad_eta_lanes(mfop, semquad_eta_field(mfop, eta_fun))
        mfwork = SemQuadKWork(mfop)
        Ku0 = zeros(Float64, Nu2)
        _semquad_apply_K_raw!(Ku0, mfop, mfetav, u0, mfwork)
        fr = Cu' * (fv - Ku0 + Bm' * p0)
        # S5-2: the outer operator (FGMRES residual and the matrix-free BFBT
        # `S2`) uses the same threaded leaf blocks as the V-cycle.
        nt_outer = _semquad_mg_thread_count(mg_threads)
        if nt_outer > 1
            par_outer = SemQuadKPar(mfop, nt_outer)
            mfpar = par_outer
            mfapplyK = (out, v) -> _semquad_apply_Kred_threaded!(out, mfop, Cu,
                                                                 mfetav, v,
                                                                 mfwork,
                                                                 par_outer)
        else
            mfapplyK = (out, v) -> _semquad_apply_Kred!(out, mfop, Cu, mfetav,
                                                        v, mfwork)
        end
    else
        fr = Cu' * (fv - K * u0 + Bm' * p0)
    end
    # Stage 0.5b2: a rotation-gauge configuration pins one free velocity dof to
    # remove the solid-body rotation from the linear system, but the pinned
    # saddle is still unsolvable for a load with a net torque: no wall of a
    # free-slip curved wall with an `:open` wall can exert one, and the
    # projected velocity would not satisfy the momentum equation.  Refuse the
    # card with the measured torque instead of returning a wrong field.
# Stage 0.5b2 rotation-gauge load solvability (see the file header)
    # Stage 0.5b2: a rotation-gauge configuration pins one free velocity dof to
    # remove the solid-body rotation from the linear system, but the pinned
    # saddle is still unsolvable for a load with a net torque: no wall of a
    # free-slip curved wall with an `:open` wall can exert one, and the
    # projected velocity would not satisfy the momentum equation.  Refuse the
    # card with the measured torque instead of returning a wrong field.
    if wd.gauge === :rotation
        _semquad_check_rotation_load(su, fv, wd.gauge)
    end
    Kred = mf_path ? nothing :
           Kgeo === nothing ? Cu' * K * Cu : Cu' * (K + Kgeo) * Cu
    Kp_red = Kp === nothing ? nothing : Cu' * Kp * Cp
    gr = Cp' * (Bc * u0)
    nf = size(Cu, 2); nfp = size(Cp, 2)
    Bs = -CpB
    Bt = Bm === Bc ? sparse(transpose(Bs)) : -(Cu' * Bm' * Cp)
    # Compressibility: in the multigrid path's (dynamic) pressure gauge the ALA density
    # perturbation -rho_ad Di gamma (p - p_ad) contributes +rho_ad Di gamma p
    # to the *upward* momentum.  The sign is not free: with ASPECT's own
    # temperature field this sign reproduces ASPECT's velocity to six digits
    # (Vrms 24.7657 vs 24.7657, max|u| 46.858 vs 46.858), while the opposite
    # sign gives Vrms 26.94 (the historical 8% King error).  See
    # docs/dev/SOLVER.md for the operator-agreement test.
    if compress_block !== nothing
        Bmm = sparse(transpose(Bm)) - compress_block
        Bt = -(Cu' * Bmm * Cp)
    end
    # Pressure-block preconditioner for the saddle solve.  It is built after
    # Kred/Bs/Bt (and after the compressibility update of Bt) because the LSC
    # variant needs diag(Kred) and the final Bt.
    Fp = nothing
    if linear === :schur_cg && schur_precond === :mass
        Mfull = assemble_semquad_pressure_mass(sp)
        Mred = Cp' * Mfull * Cp
        Fp = cholesky(Symmetric(Mred))
        issuccess(Fp) ||
            throw(ArgumentError("pressure mass preconditioner is not SPD"))
    elseif linear === :mg && !kfree
        # Pressure-block approximate inverse (mass / LSC / PCD / three-factor
        # w-BFBT); shared with the MG-accelerated Newton solver, see
        # mg/pressure.jl.  `mg_pressure_precond` is the public selector
        # (:bfbt | :mass | :scaled_mass | :pcd | :lsc); its default `:auto`
        # keeps the historical `mg_pressure_mode` resolution, which picks the
        # three-factor w-BFBT above 100x viscosity contrast -- the measured
        # winner on the blob tree (docs/dev/SOLVER.md §13).  Not built for the
        # reduced `assemble_only=:mg` geometry call, which never uses it.
        pmode_arg = mg_pressure_mode
        pweight_arg = mg_pressure_weight
        if mg_pressure_precond !== :auto
            pmode_arg, pweight_arg =
                _semquad_pressure_precond_mode(mg_pressure_precond,
                                               mg_pressure_weight)
        end
        # S3 stage A: cached by (tree fingerprint, nf, nfp) and reused while the
        # sampled viscosity moves by at most `mg_pressure_eta_tol` (the outer
        # FGMRES is flexible, so a stale Shat^{-1} is a valid preconditioner).
        Fp, pscale = semquad_mg_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred,
                                                 eta_fun;
                                                 mode=pmode_arg,
                                                 scale=mg_pressure_scale,
                                                 weight=pweight_arg,
                                                 key=_saddle_key(qt, geom, p, bc, nf, nfp),
                                                 eta_tol=mg_pressure_eta_tol,
                                                 cache=mg_pressure_cache,
                                                 bfbt_d=bfbt_d,
                                                 bfbt_s2=mg_pressure_bfbt_s2,
                                                 bfbt_s1=mg_pressure_bfbt_s1,
                                                 bfbt_s1_solve=mg_pressure_bfbt_s1_solve,
                                                 bfbt_s1_tol=mg_pressure_bfbt_s1_tol,
                                                 bfbt_s1_maxit=mg_pressure_bfbt_s1_maxit,
                                                 bfbt_s1_minwork=mg_pressure_bfbt_s1_minwork,
                                                 bfbt_s1_chunk=mg_pressure_bfbt_s1_chunk,
                                                 nthreads=_semquad_mg_thread_count(mg_threads),
                                                 applyK=mfapplyK)
    end
    if assemble_only !== false
        return (K=K, B=Bref, Bc=Bc, Bm=Bm, Cu=Cu, Cp=Cp, u0=u0, p0=p0, fv=fv,
                fr=fr, gr=gr, Kred=Kred, Bs=Bs, Bt=Bt, Kgeo=Kgeo, Kp=Kp,
                Kp_red=Kp_red)
    end
    nit = 0
    # Every solve resets the direct-path shift record, so it describes this
    # solve only (a `:mg`/`:schur_cg`/`:lagged` solve reports "no shift").
    LAST_SEMSTOKES_SHIFT[] = 0.0
    LAST_SEMSTOKES_REFINE_ITERS[] = 0
    LAST_SEMSTOKES_LADDER[] = nothing
    LAST_SEMSTOKES_LADDER_PHASE[] = nothing
    if linear === :schur_cg
        # Saddle [Kred (-CpB'); (-CpB) 0] with Bs = -CpB, Bt = Bs':
        #   Kred u + Bt p = fr,   Bs u = gr,
        # solved by a matrix-free Schur CG on S = Bs Kred^{-1} Bt.
        # A constrained/hanging velocity mode can leave a zero eigenvalue in
        # Kred after AMR; regularize with a 1e-12 relative diagonal shift so the
        # sparse LU is defined.  The shift is below the G44 exactness tolerance.
        dmax = maximum(abs, diag(Kred))
        Kred = Kred + (1.0e-12 * max(dmax, 1.0)) * I
        # Row/column equilibration: wall-refined AMR trees have h_min/h_max
        # ~1e-4 and Kred condition numbers ~1e12; the scaled LU is much better
        # conditioned and removes the artificial large velocity from the shift.
        d = sqrt.(max.(diag(Kred), 1.0e-14 * dmax))
        Dinv = spdiagm(0 => 1.0 ./ d)
        Kscaled = Dinv * Kred * Dinv
        Fv = lu(Kscaled)
        ufree = Dinv * (Fv \ (Dinv * fr))
        r = Bs * ufree - gr
        pfree = zeros(Float64, nfp)
        z = Fp === nothing ? copy(r) : Fp \ r
        pp = copy(z)
        rho = dot(r, z)
        bnorm = max(dot(r, r), 1.0e-300)
        for it in 1:schur_maxiter
            nit = it
            Ap = Bs * (Dinv * (Fv \ (Dinv * (Bt * pp))))
            den = dot(pp, Ap)
            den == 0.0 && break
            alpha = rho / den
            pfree .+= alpha .* pp
            r .-= alpha .* Ap
            rs = dot(r, r)
            rs <= 1.0e-18 * bnorm && break
            z = Fp === nothing ? copy(r) : Fp \ r
            rho_new = dot(r, z)
            pp .= z .+ (rho_new / rho) .* pp
            rho = rho_new
        end
        ufree = Dinv * (Fv \ (Dinv * (fr - Bt * pfree)))
    elseif linear === :mg
        # Scalable path: geometric MG for the velocity block plus a pressure
        # Schur block, used as a block-triangular right preconditioner for
        # flexible GMRES on the reduced saddle (S3/MF-4).  The velocity block
        # is solved by the fixed `mg_cycles` V-cycles
        # (`mg_vel_solver=:vcycle`, the default since MF-6, 33% faster than the
        # inner-CG variant on the 1024-leaf blob tree); `mg_vel_solver=:cg`
        # keeps the MF-4 scheme, CG on the SPD `Kred` preconditioned by the
        # hierarchy (the deal.II/ASPECT FGMRES+CGMG form).  The criterion is
        # the *true* reduced-saddle residual (see `fgmres_restarted`), never
        # the preconditioned one.
        uniform = length(unique(lf[1] for lf in qt.leaves)) == 1
        smoother = mg_smoother === :auto ? :jacobi : mg_smoother
        coarse = mg_coarse === :auto ? (uniform ? :rediscretize : :galerkin) :
                 mg_coarse
        # S3/MF-2: the matrix-free hierarchy always coarsens by
        # rediscretisation (see make_semquad_mg).
        mg_operator === :matrixfree && (coarse = :rediscretize)
        mg = make_semquad_mg(ss, qt, p, bc, deviatoric, eta_fun, Kred;
                             wall_bc=wall_bc,
                             cycles=mg_cycles, smoother=smoother,
                             coarse_operator=coarse, operator=mg_operator,
                             coarsen_layers=mg_coarsen_layers,
                             coarsen_factor=coarsen_factor,
                             coarse_direct_max=mg_coarse_direct_max,
                             smoothing=mg_smoothing, pre=mg_pre,
                             post=mg_post, omega=mg_omega,
                             mg_threads=mg_threads, mg_diag=mg_diag,
                             fine_Cu=Cu,
                             # On the matrix-free path the solve already built
                             # the finest level's leaf-local operator and its
                             # eta field; reuse both instead of resampling the
                             # viscosity a third time (0.17 s at one thread).
                             fine_mfop=mf_path ? mfop : nothing,
                             fine_mfeta=mf_path ? mfetav : nothing)
        Fp === nothing &&
            throw(ArgumentError("linear=:mg requires the pressure mass " *
                                "preconditioner (schur_precond=:mass)"))
        nftot = nf + nfp
        sol, nit, relres, cgiters =
            semquad_mg_saddle_solve(Kred, Bs, Bt, Fp, pscale, vcat(fr, gr), mg;
                                    tol=mg_tol, maxit=mg_maxit,
                                    restart=mg_restart,
                                    vel_solver=mg_vel_solver,
                                    vel_tol=mg_vel_tol,
                                    vel_maxit=mg_vel_maxit,
                                    applyK=mfapplyK)
        LAST_SEMSTOKES_VEL_CG_ITERS[] = cgiters
        ufree = sol[1:nf]
        pfree = sol[(nf + 1):nftot]
        relres > 1.0e-5 &&
            @warn "MG-FGMRES true relres=$relres exceeds 1e-5 at level " *
                  "$(mg.geo.L)"
    elseif linear === :lagged
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        # S2: reuse the previous LU as a preconditioner and polish with iterative
        # refinement; refactor only on a cache miss / stall.  The saddle depends
        # on the viscosity field and the tree, not on dt, so between two AMR
        # cycles the stale factor is a good preconditioner.
        key = _saddle_key(qt, geom, ss.su.p, bc, nf, nfp)
        sol, rn = _lagged_solve!(S, rhs, key; tol=1.0e-8, maxit=25)
        LAST_SEMSTOKES_REGULARIZED[] = false
        if rn > 1.0e-8
            dmax = maximum(abs, diag(S)) + 1.0
            Ssh = S + (1.0e-12 * dmax) * I
            sol, rn = _lagged_solve!(Ssh, rhs, key; tol=1.0e-8, maxit=6,
                                     force_refactor=true)
            LAST_SEMSTOKES_REGULARIZED[] = true
        end
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    else
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        # Wall-refined trees with free-slip walls make the *unshifted* sparse LU
        # unstable: on the 2b base=3/4 wall trees the factorisation has a growth
        # factor ~1e15 (max |U_ii| ~ 1e13-1e16 against a 2.9e-5 smallest pivot)
        # and the backward residual is ~1e3, while the same system solved with a
        # 1e-12 relative diagonal shift -- exactly what `_semquad_newton_saddle`
        # and the :schur_cg path already use -- is exact (residual 3e-11) and
        # gives the same velocity to 1e-6.  The near-null mode is a pressure mode
        # of the level-6 wall strip (|Bs'q|/|q| ~ 5e-3, velocity part 2e-4).
        #
        # `_shift_ladder_solve` keeps that single retry as its first rung and
        # only escalates (1e-10, 1e-8) while the residual misses
        # `stokes_accept_tol`, the driver's Stokes gate; the last resort is a few
        # iterative-refinement steps with the same factorisation.  The default
        # `accept_tol = Inf` stops after the first rung, so the well-conditioned
        # systems (uniform meshes, L2-L4 2b, the blob) stay bit-identical.
        # docs/dev/CASE_REALISTIC.md §13: on the 8192-leaf realistic tree the 1e-12
        # rung alone leaves 1.3e-3 (1.3x the card's 1e-3) and the driver aborted
        # the long run at step 6130.  §13.10: when the unshifted solve is *lost*
        # (`stokes_lost_tol`, default 1.0) the ladder tries the tiny shifts
        # first, because the shift's bias is `shift*|x|` and the near-null
        # cluster of the level-5 wall strip sits ~170x below the 1e-12 rung
        # (measured: 4.7e7 unshifted -> 5.2e-7 at 1e-14 on the step-6600 state,
        # while the 1e-12 rung leaves 1.1e-4).
        rden0 = max(norm(fr), norm(gr), 1.0)
        rel = function (x)
            rv = norm(Kred * x[1:nf] + Bt * x[(nf + 1):(nf + nfp)] - fr) / rden0
            rd = norm(Bs * x[1:nf] - gr) / rden0
            return max(rv, rd)
        end
        if stokes_scaling === :equilibrated
            # docs/dev/BUG_LEDGER.md: solver-side symmetric equilibration
            # of the same saddle.  First the historical unscaled unshifted solve,
            # exactly the ladder's rule 1: a solve that is healthy there is
            # returned untouched, so healthy paths stay bitwise identical even
            # with the option on, and the scaling is only paid for on a solve
            # that would otherwise enter the shift ladder.
            x_un = S \ rhs
            r_un = rel(x_un)
            r_un = isfinite(r_un) ? r_un : Inf
            if !(r_un > 1.0e-8) && !(r_un > stokes_accept_tol)
                sol, sh, nref, rungs = x_un, 0.0, 0, nothing
                LAST_SEMSTOKES_LADDER_PHASE[] = :healthy
            else
                # The congruence S -> D*S*D, rhs -> D*rhs leaves the solution
                # unchanged (`x = D*y`); it changes only the arithmetic of the
                # factorisation.  In the scaled operator the wall-strip pressure
                # family is lifted far above the round-off floor (the measured
                # smallest eigenvalue is a momentum mode at 4.2e-11 relative,
                # not the unscaled 1.7e-14 pressure mode), so the *unshifted* LU
                # resolves it: 6.0e-10 (AMR 7641) and 3.6e-8 (uniform L5,
                # round-off limited by the null component) against 3.7e-7 /
                # 1.2e-6 for the unscaled tiny-shift stopgap.  The 1e-6 gate is
                # the stated accuracy goal; if the scaled solve misses it, the
                # historical unscaled ladder is the unchanged fallback.
                sol_scaled, _ = _semquad_stokes_equil_solve(S, Kred, Bt, Bs, rhs)
                r_sc = rel(sol_scaled)
                r_sc = isfinite(r_sc) ? r_sc : Inf
                if r_sc <= 1.0e-6 && r_sc <= stokes_accept_tol
                    sol, sh, nref = sol_scaled, 0.0, 0
                    rungs = Any[(0.0, r_un), (:equilibrated, r_sc)]
                    LAST_SEMSTOKES_LADDER_PHASE[] = :equilibrated
                    LAST_SEMSTOKES_SCALED[] += 1
                else
                    sol, _, sh, nref, rungs =
                        _shift_ladder_solve(S, rhs; accept_tol=stokes_accept_tol,
                                            lost_tol=stokes_lost_tol,
                                            bad_tol=1.0e-8, resid=rel)
                end
            end
        else
            sol, _, sh, nref, rungs =
                _shift_ladder_solve(S, rhs; accept_tol=stokes_accept_tol,
                                    lost_tol=stokes_lost_tol,
                                    bad_tol=1.0e-8, resid=rel)
        end
        LAST_SEMSTOKES_REGULARIZED[] = sh > 0.0
        LAST_SEMSTOKES_SHIFT[] = sh
        LAST_SEMSTOKES_REFINE_ITERS[] = nref
        LAST_SEMSTOKES_LADDER[] = rungs
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    end
    # True reduced-saddle residual of the system that was actually solved
    # (Schur uses the regularized Kred).  This is what distinguishes a
    # converged solve from a singular/ill-conditioned direct solve.
    r1 = if mf_path
        Ku = zeros(Float64, nf)
        mfapplyK(Ku, ufree)
        Ku .+ Bt * pfree .- fr
    else
        Kred * ufree + Bt * pfree - fr
    end
    r2 = Bs * ufree - gr
    rden = max(norm(fr), norm(gr), 1.0)
    LAST_SEMSTOKES_RESIDUAL[] = max(norm(r1), norm(r2)) / rden
    LAST_SEMSTOKES_RESID_SPLIT[] = (vel=norm(r1) / rden, div=norm(r2) / rden)
    LAST_SEMSTOKES_SCHUR_ITERS[] = nit
    u = u0 + Cu * ufree
    pv = p0 + Cp * pfree
    if return_system
        return u, pv, fixed, cons2, cons_p, K, Bref, Cu, Cp, u0, p0, fv
    end
    # Stage 0.5b2 rotation gauge: the solve used a pinned free dof to remove the
    # solid-body-rotation nullspace; the returned field is the canonical
    # zero-net-angular-momentum representative (`int r x u dA = 0`), i.e. the L2
    # projection onto the rotation mode has been removed.  The pressure is
    # unchanged (a rigid rotation has zero strain rate and zero divergence) and
    # the momentum residual is unchanged provided the load had no net torque,
    # which `_semquad_check_rotation_load` verified above.
    if wd.gauge === :rotation
        LAST_SEMSTOKES_ROTATION_SHIFT[] =
            _semquad_project_rotation!(u, geom, su)
    end
    return u, pv, fixed, cons2, cons_p
end
