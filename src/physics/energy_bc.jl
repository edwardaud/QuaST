# Energy-step boundary terms: the Dirichlet lifting map and the prescribed
# wall-flux right-hand side, box and mapped methods.  Included after the energy
# step; the step resolves these functions at runtime.
"""
    _semquad_energy_fixed!(fixed, sq, pr, flux_top, flux_bot, side_fixed)

Fill the Dirichlet map of the energy step.  The `CartesianBox` method is the
historical physical-row loop, verbatim (GEOM_DESIGN E16: the box keeps its
`y <= 1e-14`/`x >= Lx-1e-14` tests).  A mapped geometry detects the four walls
from the **logical leaf index** (`lf[2] == 0`/`last`, `lf[3] == 0`/`last`, node
at the face) and evaluates the base temperature at the mapped node's physical
height; the assignment order (bottom, top, left, right) is the box body's, so a
corner keeps the same value as on the box path.  Nodes shared by several leaves
are assigned the same value once per leaf, so the repeated writes are
idempotent.
"""
function _semquad_energy_fixed!(fixed::Dict{Int,Float64}, sq::SemQuad,
                                pr::Params,
                                flux_top::Union{Nothing,Float64},
                                flux_bot::Union{Nothing,Float64},
                                side_fixed::Bool)
    geom = sq.geom
    if geom isa CartesianBox
        return _semquad_energy_fixed_box!(fixed, sq, pr, flux_top, flux_bot,
                                          side_fixed)
    end
    return _semquad_energy_fixed_mapped!(fixed, geom, sq, pr, flux_top,
                                         flux_bot, side_fixed)
end

# Historical box body, verbatim: physical coordinate row tests (E16).
function _semquad_energy_fixed_box!(fixed::Dict{Int,Float64}, sq::SemQuad,
                                    pr::Params,
                                    flux_top::Union{Nothing,Float64},
                                    flux_bot::Union{Nothing,Float64},
                                    side_fixed::Bool)
    for g in 1:sq.N
        x = sq.coords[g][1]
        y = sq.coords[g][2]
        # theta_wall = Twall - Tad: zero for the Boussinesq base (Tad = Tbar,
        # Twall = 1 at the bottom and 0 at the top), non-zero once a reference
        # state shifts the base off the physical wall temperature.  A wall with a
        # prescribed flux is left free; the flux enters the right-hand side.
        y <= 1.0e-14 && flux_bot === nothing &&
            (fixed[g] = pr.Twall_bot - semquad_base_T(pr, y))
        y >= 1.0 - 1.0e-14 && flux_top === nothing &&
            (fixed[g] = pr.Twall_top - semquad_base_T(pr, y))
        if side_fixed
            x <= 1.0e-14 &&
                (fixed[g] = pr.Twall_left - semquad_base_T(pr, y))
            x >= sq.Lx - 1.0e-14 &&
                (fixed[g] = pr.Twall_right - semquad_base_T(pr, y))
        end
    end
    return fixed
end

function _semquad_energy_fixed_mapped!(fixed::Dict{Int,Float64},
                                       g::AbstractGeometry, sq::SemQuad,
                                       pr::Params,
                                       flux_top::Union{Nothing,Float64},
                                       flux_bot::Union{Nothing,Float64},
                                       side_fixed::Bool)
    n = sq.p + 1
    # Stage 0.5b1: the two thermal Dirichlet walls are the geometry's Nusselt
    # pair, not the logical `y` sides.  On the box-in-height validation
    # geometries that pair is `(:yminus,:yplus)`, so the loop below is the
    # historical bottom/top body verbatim; on the annulus it is
    # `(:xminus,:xplus)` (the inner and outer circles), which is where the
    # physical `T = 1/0` boundary data live.  `side_thermal=:fixed` pins the
    # *other* logical pair (the Cartesian x sides on a validation geometry; the
    # wedge cuts on the annulus, which the driver refuses).
    sideb, sidet = nu_wall_sides(g)
    sides_left = (sideb === :xminus || sidet === :xminus) ?
                 (:yminus, :yplus) : (:xminus, :xplus)
    for lf in sq.leaves
        nsidex = leaf_nside_x(sq, lf[1])
        nsidey = leaf_nside_y(sq, lf[1])
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            on_bot = _energy_node_on_side(sideb, i, j, n, lf, nsidex, nsidey)
            on_top = _energy_node_on_side(sidet, i, j, n, lf, nsidex, nsidey)
            on_left = side_fixed &&
                      _energy_node_on_side(sides_left[1], i, j, n, lf,
                                           nsidex, nsidey)
            on_right = side_fixed &&
                       _energy_node_on_side(sides_left[2], i, j, n, lf,
                                            nsidex, nsidey)
            (on_bot || on_top || on_left || on_right) || continue
            gid = ids[(j - 1) * n + i]
            x, y = sq.coords[gid]
            # Stage 0.5b1: the base state is the geometry's conductive profile
            # (the annulus's radial log/H state); on the box-in-height
            # validation geometries this is exactly `semquad_base_T(pr, h)`.
            base = semquad_base_T(pr, g, x, y)
            on_bot && flux_bot === nothing &&
                (fixed[gid] = pr.Twall_bot - base)
            on_top && flux_top === nothing &&
                (fixed[gid] = pr.Twall_top - base)
            if side_fixed
                on_left && (fixed[gid] = pr.Twall_left - base)
                on_right && (fixed[gid] = pr.Twall_right - base)
            end
        end
    end
    return fixed
end

# Is node `(i,j)` of leaf `lf` on logical side `side`?  The same leaf-index test
# the mapped diagnostics use (`_on_logical_wall`), at node granularity.
@inline function _energy_node_on_side(side::Symbol, i::Int, j::Int, n::Int,
                                      lf::NTuple{3,Int}, nsidex::Int,
                                      nsidey::Int)
    side === :xminus && return i == 1 && lf[2] == 0
    side === :xplus && return i == n && lf[2] == nsidex - 1
    side === :yminus && return j == 1 && lf[3] == 0
    return j == n && lf[3] == nsidey - 1
end

"""
    _add_semquad_wall_flux!(F, sq, qt, q, wall)

Add the boundary term `int_wall q phi ds` to the energy right-hand side.  `q` is
the prescribed *outward normal* derivative `dtheta/dn` on that wall (for the
bottom wall, where the outward normal is -z, a positive `q` therefore means
`-dtheta/dz > 0`, i.e. heat entering from below).

The `CartesianBox` method is the historical body, verbatim: the GLL weights
`0.5*hx*sq.w[i]` integrate the wall basis exactly.  The mapped method integrates
the same form with the leaf's mapped face Jacobian and Gauss face weights on the
geometry's own wall sides — `nu_wall_sides(g)`, so on the validation geometries
the logical `y` sides (the box body's walls) and on the annulus the two circles.
A stage-0.5 audit defect (D4) had this method hardcode the logical `y` sides, so
a prescribed flux landed on the annulus's insulating radial cuts while the
circle kept its natural zero-flux condition; the Dirichlet side of the same
branch was already routed (`_semquad_energy_fixed_mapped!`, stage 0.5b1).
On the box map the two methods agree to round-off (they are the same integral,
`t_g269_curved_energy.jl` part (c)); the box body stays untouched.
"""
function _add_semquad_wall_flux!(F::Vector{Float64}, sq::SemQuad, qt::Quadtree,
                                 q::Float64, wall::Symbol)
    wall in (:top, :bottom) || throw(ArgumentError("wall must be :top or :bottom"))
    geom = sq.geom
    geom isa CartesianBox &&
        return _add_semquad_wall_flux_box!(F, sq, qt, q, wall)
    return _add_semquad_wall_flux_mapped!(F, geom, sq, q, wall)
end

# Historical box body, verbatim.
function _add_semquad_wall_flux_box!(F::Vector{Float64}, sq::SemQuad,
                                     qt::Quadtree, q::Float64, wall::Symbol)
    wall in (:top, :bottom) || throw(ArgumentError("wall must be :top or :bottom"))
    n = sq.p + 1
    for lf in sq.leaves
        (wall === :top ? lf[3] == sq.ny0 * (1 << lf[1]) - 1 : lf[3] == 0) || continue
        hx = leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        row = wall === :top ? n : 1
        for i in 1:n
            F[ids[(row - 1) * n + i]] += 0.5 * hx * sq.w[i] * q
        end
    end
    return F
end

function _add_semquad_wall_flux_mapped!(F::Vector{Float64}, g::AbstractGeometry,
                                        sq::SemQuad, q::Float64,
                                        wall::Symbol)
    wall in (:top, :bottom) || throw(ArgumentError("wall must be :top or :bottom"))
    n = sq.p + 1
    # Stage-0.5 audit defect D4: the wall is the geometry's own Nusselt side
    # (`nu_wall_sides`, the same pair the Dirichlet branch uses), not the
    # logical `y` rows.  On the box-in-height validation geometries that pair is
    # `(:yminus,:yplus)`, so this body is the historical one there.
    side = wall === :top ? nu_wall_sides(g)[2] : nu_wall_sides(g)[1]
    fd = make_fedata(sq.p, sq.p)
    for lf in sq.leaves
        _on_logical_wall(lf, side, sq) || continue
        ids = sq.elem[lf]
        fm = face_metric(g, sq, lf, side, fd)
        for qq in 1:fd.nq
            wq = face_weight(fm, fd, qq) * q
            for ii in 1:n
                gid = _logical_wall_node(ids, n, side, ii)
                F[gid] += wq * fd.phi_v[ii, qq]
            end
        end
    end
    return F
end
