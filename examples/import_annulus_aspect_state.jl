#!/usr/bin/env julia
# =============================================================================
# import_annulus_aspect_state.jl — build a QuaST annulus checkpoint from an
# ASPECT state.
#
#   julia --project=. examples/import_annulus_aspect_state.jl \
#       --state  out_julia/annulus_import/asp_t3_state.h5 \
#       --template out_julia/annulus_video_v3/quast_ckpt/ann_16x192.h5 \
#       --out    out_julia/annulus_import/quast_t3/ann_16x192.h5
#
# This is the annulus analogue of `examples/import_aspect_state.jl` (the box
# importer).  `--state` is written by
# `aspect_runs/davies_et_al/extract_annulus_state.py`, which recovers the exact
# Q2 field of one ASPECT VTU frame on the structured polar node grid
# `r (nr)`, `phi (nphi)`, `T (nr, nphi)`, `ux`, `uy` (the Davies et al. (2022)
# card is nondimensional: the gap d = 1 is the length unit, kappa = 1, dT = 1,
# so no unit conversion is applied; contrast the box importer, whose card is
# dimensional).
#
# What this script does:
#
#  1. builds the uniform periodic ring (the harness's own tree rule,
#     `annulus_benchmark_tree(nx, ny)` — Q2-Q1-Q2, seam shared) on the benchmark
#     geometry R0 = 1.22, R1 = 2.22, phi = 2pi;
#  2. evaluates the ASPECT Q2 field at every QuaST GLL node by the exact
#     tensor-product Q2 interpolation: the radial cell is found from the stored
#     r ladder (Q2 nodes 2k, 2k+1, 2k+2), the angular cell from the uniform
#     phi ladder with the periodic wrap (node index modulo nphi, so the last
#     cell's right node is column 0 = the seam).  The evaluation is checked to
#     be exact at the ASPECT nodes themselves, and the interpolation into our
#     nodes is reported (on this card the two grids coincide node for node —
#     16x192 ring = 33 x 384 nodes = the ref-4 ASPECT Q2 node cloud — so the
#     transfer is a nodal copy: no O(h^2) interpolation error enters the real
#     import; on a state grid that differs it does, and the measured bound is
#     printed);
#  3. forms the perturbation theta = T_total - semquad_base_T(pr, g, x, y) (the
#     radial log conduction base state, exactly the solver's `_annulus_Tbar`),
#     zeroes theta on both walls (r = R0, R1: the Dirichlet nodes of both
#     codes) and asserts the periodic seam trace is continuous (the same
#     physical nodes evaluated from phi = 0 and from phi = 2pi agree to
#     round-off);
#  4. computes and prints the imported state's diagnostics — family-A
#     `Nu_top`/`Nu_bot` (one-sided GLL wall gradient / base gradient),
#     `V_rms`, mean T — and compares them with the ASPECT statistics row and
#     with the family-A diagnostics of the extracted Q2 field carried in the
#     state file.  `V_rms` and mean T are the interpolation check (they must
#     match the statistics row to the quadrature accuracy); the Nusselt pair
#     is compared to ASPECT's *family-C* consistent-boundary-flux value only
#     with the family caveat stated (QuaST's family-A coarse-mesh offset is
#     +8 % on this card at this resolution, a measurement definition, not a
#     different state);
#  5. writes a checkpoint with `save_semquad_state`, taking the `run_config`
#     from `--template` (an existing checkpoint of the target card) so the
#     restart guard accepts the resume: `t` = the state's time, `step = 0`,
#     `theta_prev = theta` (the import has no time history; a faked BDF2
#     history would be a larger error than the O(dt) first step — the box
#     importer's measured argument, unchanged here), pressure zeros (not an
#     initial condition of the Boussinesq system), `dt`/`dt_prev` from the
#     template, `velocity_stale = false` and the imported velocity as
#     `ux`/`uz`.  A same-card continuation of the imported state is then an
#     ordinary restart; continuing an ASPECT case-2.2 state (Ra = 1e5) from a
#     case-2.1 template (Ra = 1e4) is the documented `restart_physics=:allow`
#     protocol.
#
# The checkpoint is reloaded after writing and its config, time, geometry
# record and field sizes are checked again: loading must succeed and give back
# the same state.
#
# `--print-only` runs the whole import and prints the diagnostics without
# writing a checkpoint.
# =============================================================================

using QuaST, Printf
import HDF5

# The annulus benchmark harness (card / geometry / uniform ring tree).  Its
# `main` is guarded by `abspath(PROGRAM_FILE) == @__FILE__`, so including it
# only defines functions; this file's `main_import` is the entry point here.
include(joinpath(@__DIR__, "annulus_benchmark.jl"))

const IMPORT_FORMAT = "annulus_import_v1"

# `@printf` needs a literal format string; the multi-line messages here are
# assembled from adjacent literals for readability, so use the runtime
# formatter (same output, negligible cost in this one-shot import).
iprintf(fmt::AbstractString, args...) =
    print(Printf.format(Printf.Format(fmt), args...))

# ---------------------------------------------------------------------------
# the extracted state file
# ---------------------------------------------------------------------------
"""
    AspAnnulusState

One ASPECT frame on the structured polar node grid: `r` (nr), `phi` (nphi),
`T`, `ux`, `uy` (nr, nphi) and the provenance/diagnostics attributes of
`extract_annulus_state.py` (`R0`, `R1`, `cells_r`, `cells_t`, `t`, `frame`,
`source_vtu_sha256`, the family-A field diagnostics and -- when the extractor
was given the statistics table -- the matching ASPECT statistics row).
"""
struct AspAnnulusState
    r::Vector{Float64}
    phi::Vector{Float64}
    T::Matrix{Float64}
    ux::Matrix{Float64}
    uy::Matrix{Float64}
    R0::Float64
    R1::Float64
    cells_r::Int
    cells_t::Int
    nr::Int
    nphi::Int
    t::Float64
    frame::Int
    source_sha::String
    attrs::Dict{String,Any}
end

"""
    _read_h5_matrix(f, name, nr, nphi, path)

Read an h5py-written `(nr, nphi)` dataset.  HDF5.jl reads the same bytes
column-major, so a C-ordered h5py matrix arrives transposed; accept both
orientations (the box importer's convention, `read_asp`'s `read_mat`).
"""
function _read_h5_matrix(f, name::AbstractString, nr::Int, nphi::Int,
                         path::AbstractString)
    A = read(f[name])
    size(A) == (nr, nphi) && return Matrix{Float64}(A)
    size(A) == (nphi, nr) && return Matrix{Float64}(permutedims(A))
    throw(ArgumentError("state file $path: dataset `$name` has size " *
                        "$(size(A)) but r/phi imply ($nr, $nphi) (or its " *
                        "transpose): the file is not an annulus state from " *
                        "aspect_runs/davies_et_al/extract_annulus_state.py"))
end

"""
    read_asp_annulus_state(path; r0=ANN_R0, r1=ANN_R1) -> AspAnnulusState

Load and validate one extracted ASPECT annulus state.  Every validation is an
actionable `ArgumentError`: a non-finite field, a grid that is not the uniform
periodic polar ladder (odd node count, duplicate seam column, non-uniform
spacing), a radius span that does not match its own `R0`/`R1` attributes or the
target annulus geometry, and a wall temperature that is not the benchmark's
inner 1 / outer 0 are all refused here, before any checkpoint is written.
"""
function read_asp_annulus_state(path::AbstractString;
                                r0::Float64=ANN_R0, r1::Float64=ANN_R1)
    isfile(path) || throw(ArgumentError("state file not found: $path (run " *
        "aspect_runs/davies_et_al/extract_annulus_state.py first)"))
    return HDF5.h5open(path, "r") do f
        for k in ("r", "phi", "T", "ux", "uy")
            haskey(f, k) || throw(ArgumentError("state file $path has no " *
                "`$k` dataset: it is not an annulus state written by " *
                "aspect_runs/davies_et_al/extract_annulus_state.py"))
        end
        r = Vector{Float64}(read(f["r"]))
        phi = Vector{Float64}(read(f["phi"]))
        nr, nphi = length(r), length(phi)
        nr >= 3 || throw(ArgumentError("state file $path: only $nr radial " *
            "nodes; an annulus Q2 grid has 2*cells_r+1 >= 3"))
        nphi >= 8 || throw(ArgumentError("state file $path: only $nphi " *
            "angular nodes; the periodic ring needs an even count >= 8"))
        T = _read_h5_matrix(f, "T", nr, nphi, path)
        ux = _read_h5_matrix(f, "ux", nr, nphi, path)
        uy = _read_h5_matrix(f, "uy", nr, nphi, path)
        attrs = Dict{String,Any}(String(k) => v for (k, v) in HDF5.attrs(f))
        for (name, A) in (("T", T), ("ux", ux), ("uy", uy))
            all(isfinite, A) || throw(ArgumentError("state file $path: the " *
                "`$name` field has non-finite values; refusing to import a " *
                "state with NaN/Inf (fix the extraction or the VTU frame)"))
        end
        # --- the node ladders ------------------------------------------------
        # radial: strictly increasing and uniform, spanned by its own attrs
        for i in 2:nr
            r[i] > r[i - 1] || throw(ArgumentError("state file $path: the r " *
                "ladder is not strictly increasing at index $i " *
                "($(r[i-1]) -> $(r[i])); it must be the uniform polar grid " *
                "r = R0 + i*(R1-R0)/(nr-1)"))
        end
        for k in ("R0", "R1", "cells_r", "cells_t", "t", "frame")
            haskey(attrs, k) || throw(ArgumentError("state file $path has no " *
                "`$k` attribute: it is not an annulus state from " *
                "aspect_runs/davies_et_al/extract_annulus_state.py"))
        end
        R0 = Float64(attrs["R0"])
        R1 = Float64(attrs["R1"])
        cells_r = Int(attrs["cells_r"])
        cells_t = Int(attrs["cells_t"])
        t = Float64(attrs["t"])
        frame = Int(attrs["frame"])
        nr == 2 * cells_r + 1 || throw(ArgumentError("state file $path: nr=" *
            "$nr but cells_r=$cells_r; the Q2 node grid must have " *
            "nr = 2*cells_r + 1. The file's grid metadata is inconsistent."))
        nphi == 2 * cells_t || throw(ArgumentError("state file $path: nphi=" *
            "$nphi but cells_t=$cells_t; the periodic ring counts the seam " *
            "node once, so the angular node grid must have nphi = 2*cells_t " *
            "(an odd nphi or a duplicated phi = 2pi column is not a periodic " *
            "grid). Fix the extractor or the state file."))
        dr = (R1 - R0) / (nr - 1)
        for i in 1:nr
            target = R0 + (i - 1) * dr
            abs(r[i] - target) <= 1.0e-9 * max(1.0, abs(dr)) ||
                throw(ArgumentError("state file $path: r[$i] = $(r[i]) is " *
                    "off the uniform ladder R0 + (i-1)*(R1-R0)/(nr-1) = " *
                    "$target; the radial grid is not the uniform polar grid"))
        end
        # the target geometry is the benchmark card; a state of another
        # annulus (or a corrupted span) must not be imported silently
        for (name, val, want) in (("R0", R0, r0), ("R1", R1, r1))
            abs(val - want) <= 1.0e-6 * (r1 - r0) ||
                throw(ArgumentError("state file $path: $name = $val does not " *
                    "match the target annulus geometry $name = $want (the " *
                    "Davies et al. card R0=1.22, R1=2.22). Import the state " *
                    "on the geometry it was extracted from, or fix the " *
                    "extractor's --r0/--r1."))
        end
        # angular: start at 0, uniform in 2pi/nphi (the last node is one
        # spacing below 2pi; phi = 2pi is the same node as phi = 0)
        abs(phi[1]) <= 1.0e-12 || throw(ArgumentError("state file $path: " *
            "phi[1] = $(phi[1]) != 0; the angular ladder must start at " *
            "phi = 0 with the seam counted once"))
        dphi = 2.0 * pi / nphi
        for j in 1:nphi
            target = (j - 1) * dphi
            abs(phi[j] - target) <= 1.0e-9 * max(1.0, dphi) ||
                throw(ArgumentError("state file $path: phi[$j] = $(phi[j]) " *
                    "is off the uniform periodic ladder j*2pi/nphi = " *
                    "$target (a duplicated phi = 2pi column or a non-periodic " *
                    "angular grid cannot be imported as the closed ring)"))
        end
        # the wall values of the benchmark state (theta = T - Tbar is zero there
        # because Tbar(R0) = 1, Tbar(R1) = 0): a different wall data state would
        # make the theta normalisation below wrong, so refuse it up front
        max_in = maximum(abs.(T[1, :] .- 1.0))
        max_out = maximum(abs.(T[nr, :]))
        (max_in <= 1.0e-6 && max_out <= 1.0e-6) ||
            throw(ArgumentError("state file $path: wall temperatures are " *
                "inner 1 + O($max_in), outer O($max_out); the benchmark's " *
                "Dirichlet data is T = 1 at r = R0 and T = 0 at r = R1, which " *
                "is what the theta = T - Tbar normalisation (Tbar(R0)=1, " *
                "Tbar(R1)=0) assumes. Refusing to import other wall data."))
        stats = Dict{String,Float64}()
        field = Dict{String,Float64}()
        for (k, v) in attrs
            v isa Real || continue
            startswith(k, "aspect_") && (stats[k] = Float64(v))
            startswith(k, "field_") && (field[k] = Float64(v))
        end
        return AspAnnulusState(r, phi, T, ux, uy, R0, R1, cells_r, cells_t,
                               nr, nphi, t, frame,
                               string(get(attrs, "source_vtu_sha256", "")),
                               attrs)
    end
end

# ---------------------------------------------------------------------------
# exact tensor-product Q2 evaluation on the (r, phi) node grid
# ---------------------------------------------------------------------------
# The nine Q2 nodes of a cell are (2k, 2k+1, 2k+2) radially and
# (2m, 2m+1, 2m+2) angularly (the last wraps to column 0: the seam is one
# column).  `_lag` is the Q2 Lagrange basis on the nodes -1, 0, 1 (the same
# three-point basis as the box importer's `_lag`).
@inline function _lag(t::Float64, i::Int)
    i == 0 && return 0.5 * t * (t - 1.0)
    i == 1 && return 1.0 - t * t
    return 0.5 * t * (t + 1.0)
end

"""
    _q2_cell_indices(st, r, phi) -> (k_radial, k_angular, xi, eta)

Locate `(r, phi)` in the structured grid and return the zero-based radial and
angular cell indices plus the reference coordinates `xi`, `eta` in [-1, 1].
`phi` is used modulo the physical circle only through the caller; `phi = 2pi`
lands in the last angular cell at `eta = +1` (whose right node is column 0).
"""
@inline function _q2_cell_indices(st::AspAnnulusState, r::Float64, phi::Float64)
    dr = (st.r[end] - st.r[1]) / (st.nr - 1)
    dphi = 2.0 * pi / st.nphi
    k = clamp(floor(Int, (r - st.r[1]) / (2.0 * dr)), 0, st.cells_r - 1)
    kk = clamp(floor(Int, phi / (2.0 * dphi)), 0, st.cells_t - 1)
    xi = (r - st.r[1] - 2.0 * k * dr) / dr - 1.0
    eta = (phi - 2.0 * kk * dphi) / dphi - 1.0
    return k, kk, xi, eta
end

"""
    eval_asp_q2(st, r, phi) -> (T, ux, uy)

Exact tensor-product Q2 evaluation of the extracted ASPECT field at `(r, phi)`.
"""
@inline function eval_asp_q2(st::AspAnnulusState, r::Float64, phi::Float64)
    k, kk, xi, eta = _q2_cell_indices(st, r, phi)
    sT = 0.0
    sx = 0.0
    sy = 0.0
    @inbounds for b in 0:2
        mb = _lag(eta, b)
        jj = mod(2 * kk + b, st.nphi) + 1
        for a in 0:2
            w = _lag(xi, a) * mb
            ii = 2 * k + a + 1
            sT += w * st.T[ii, jj]
            sx += w * st.ux[ii, jj]
            sy += w * st.uy[ii, jj]
        end
    end
    return sT, sx, sy
end

# (end of the Q2 evaluation helpers)

# ---------------------------------------------------------------------------
# the import
# ---------------------------------------------------------------------------
"""
    import_params(cfg) -> Params

The target card's `Params`, taken from the template checkpoint's `run_config`
(the annulus card is isoviscous Boussinesq: only `H` and the wall data enter
the base state and the diagnostics used by the import; `Ra`, the viscosity law
and the boundary-condition fields are carried for the card printout and for a
potential continuation, not for the import arithmetic).
"""
function import_params(cfg::AbstractDict)
    ra = parse(Float64, get(cfg, "Ra", "1.0e5"))
    H = parse(Float64, get(cfg, "H", "0.0"))
    ttop = parse(Float64, get(cfg, "Twall_top", "0.0"))
    tbot = parse(Float64, get(cfg, "Twall_bot", "1.0"))
    return Params(Ra=ra, H=H, eta_law=:expT, deta_T=1.0,
                  Twall_top=ttop, Twall_bot=tbot, bc=:free,
                  wall_bc=(:free, :free, :free, :free))
end

"""
    interp_ring!(theta, ux, uz, sq, st, pr, g) -> NamedTuple

Fill the QuaST ring state from the ASPECT frame.  `theta = T_total - Tbar(r)`
with the base state evaluated at the geometry's own point, the wall nodes are
set to exactly zero (the Dirichlet nodes of both codes), and the periodic seam
trace is checked: every ring node on the `phi = 0` ray is evaluated from both
`phi = 0` (first angular cell) and `phi = 2pi` (last cell, wrapped) and the two
values must agree to round-off.  Returns the measured counts and errors.
"""
function interp_ring!(theta::Vector{Float64}, ux::Vector{Float64},
                      uz::Vector{Float64}, sq::SemQuad, st::AspAnnulusState,
                      pr::Params, g::PolarAnnulus)
    nwall = 0
    n_wall_nodes = 0
    max_wall = 0.0
    n_seam = 0
    max_seam = 0.0
    Tscale = max(1.0, maximum(abs.(st.T)))
    for gi in eachindex(sq.coords)
        x, y = sq.coords[gi]
        r = hypot(x, y)
        phi = mod(atan(y, x), 2.0 * pi)
        Tt, vx, vy = eval_asp_q2(st, r, phi)
        th = Tt - semquad_base_T(pr, g, x, y)
        ux[gi] = vx
        uz[gi] = vy
        # the seam nodes (phi = 0 ray): evaluate the same physical point from
        # the other side of the periodic boundary; the tensor-product basis
        # must select the wrapped column (index modulo nphi) and give exactly
        # the same node values.
        if phi == 0.0
            n_seam += 1
            T2, _, _ = eval_asp_q2(st, r, 2.0 * pi)
            max_seam = max(max_seam, abs(Tt - T2))
        end
        # the walls are level sets of the physical height; the driver's wall
        # guard uses the same test (`height(g,x,y) <= 1e-14` / `>= 1-1e-14`)
        h = height(g, x, y)
        if h <= 1.0e-14 || h >= 1.0 - 1.0e-14
            n_wall_nodes += 1
            max_wall = max(max_wall, abs(th))
            abs(th) > 1.0e-8 && (nwall += 1)
            th = 0.0
        end
        theta[gi] = th
    end
    return (nwall=nwall, n_wall_nodes=n_wall_nodes, max_wall=max_wall,
            n_seam=n_seam, max_seam=max_seam / Tscale, n_coords=sq.N)
end

"""
    check_q2_exactness(st; tol=1e-12) -> NamedTuple

Evaluate the Q2 interpolant at every one of the state's own nodes and compare
with the stored nodal values.  A mismatch would mean the interpolation or the
node ordering is wrong (a transposed or off-by-one index shows up here); the
exact tensor-product basis must return the stored value to round-off.
"""
function check_q2_exactness(st::AspAnnulusState; tol::Float64=1.0e-12)
    maxT = 0.0
    maxu = 0.0
    Tscale = max(1.0, maximum(abs.(st.T)))
    uscale = max(1.0, maximum(hypot.(st.ux, st.uy)))
    for i in 1:st.nr, j in 1:st.nphi
        Tt, vx, vy = eval_asp_q2(st, st.r[i], st.phi[j])
        maxT = max(maxT, abs(Tt - st.T[i, j]))
        maxu = max(maxu, abs(vx - st.ux[i, j]), abs(vy - st.uy[i, j]))
    end
    (max_T=maxT / Tscale, max_u=maxu / uscale,
     max_T_abs=maxT, max_u_abs=maxu)
end

"""
    interp_report(sq, st) -> NamedTuple

How far are the QuaST GLL nodes from the state's node cloud, and how much
interpolation does the transfer actually use?  For every ring node the nearest
state node is found by rounding the (r, phi) indices; `n_coincident` counts the
nodes that ARE state nodes (distance <= 1e-9 in r and phi), `max_node_offset`
the worst distance, and `max_node_value_gap` the largest difference between the
Q2 evaluation at the ring node and the nearest state node's stored value (zero
for a coincident node, O(h^2) interpolation error otherwise).
"""
function interp_report(sq::SemQuad, st::AspAnnulusState)
    dr = (st.r[end] - st.r[1]) / (st.nr - 1)
    dphi = 2.0 * pi / st.nphi
    nco = 0
    max_off = 0.0
    max_gap = 0.0
    Tscale = max(1.0, maximum(abs.(st.T)))
    for (x, y) in sq.coords
        r = hypot(x, y)
        phi = mod(atan(y, x), 2.0 * pi)
        i = clamp(round(Int, (r - st.r[1]) / dr) + 1, 1, st.nr)
        j = mod(round(Int, phi / dphi), st.nphi) + 1
        off = max(abs(r - st.r[i]), abs(mod(phi - st.phi[j] + pi, 2.0 * pi) - pi))
        max_off = max(max_off, off)
        (abs(r - st.r[i]) <= 1.0e-9 * max(1.0, dr) &&
         abs(mod(phi - st.phi[j] + pi, 2.0 * pi) - pi) <= 1.0e-9 * max(1.0, dphi)) &&
            (nco += 1)
        Tt, _, _ = eval_asp_q2(st, r, phi)
        max_gap = max(max_gap, abs(Tt - st.T[i, j]) / Tscale)
    end
    (n_nodes=sq.N, n_coincident=nco, max_node_offset=max_off,
     max_node_value_gap=max_gap)
end

"""
    state_diagnostics(qt, sq, theta, ux, uz, pr) -> NamedTuple

The imported state's diagnostics on the QuaST ring: family-A wall Nusselt pair
(`semquad_nu_walls`), `V_rms` (`semquad_vrms`), the area mean of the full
T = Tbar + theta and its maximum.
"""
function state_diagnostics(qt::Quadtree, sq::SemQuad, theta::Vector{Float64},
                           ux::Vector{Float64}, uz::Vector{Float64},
                           pr::Params)
    nut, nub = semquad_nu_walls(sq, qt, theta, pr)
    vr = semquad_vrms(sq, qt, ux, uz)
    tm = semquad_mean_T(sq, qt, theta, pr)
    tmax = maximum(semquad_base_T(pr, sq.geom, x, y) + theta[gi]
                   for (gi, (x, y)) in enumerate(sq.coords))
    return (Nu_top=nut, Nu_bot=nub, Vrms=vr, Tmean=tm, Tmax=tmax)
end

"""Relative difference `(a - b)/b` (NaN when `b == 0`)."""
reldiff(a, b) = b == 0.0 ? NaN : (a - b) / b

"""
    import_annulus_state(; state, template, out=nothing, nx, ny,
                          print_only=false, verbose=true) -> NamedTuple

The whole import (load + validate the state, build the ring, evaluate + transfer
the Q2 field, wall/seam handling, diagnostics + ASPECT comparison, optional
checkpoint write + reload).  This is the function the gate
`test/t_g292_annulus_import.jl` calls for every arm; `main_import` is the CLI
wrapper.
"""
function import_annulus_state(; state::AbstractString,
                              template::AbstractString,
                              out::Union{Nothing,AbstractString}=nothing,
                              nx::Int=16, ny::Int=192,
                              print_only::Bool=false, verbose::Bool=true)
    st = read_asp_annulus_state(state)
    g = annulus_benchmark_geometry()
    tpl = load_semquad_state(template)
    cfg = copy(tpl.config)
    # --- the template must be a checkpoint of the annulus target card ------
    isempty(cfg) && throw(ArgumentError("template checkpoint $template has " *
        "an empty run_config: it cannot establish the target card's " *
        "configuration. Pass a checkpoint written by an annulus run (e.g. " *
        "out_julia/annulus_video_v3/quast_ckpt/ann_16x192.h5)."))
    gid = geometry_identity(g)
    stored_geom = get(cfg, "geometry", nothing)
    stored_geom === nothing && throw(ArgumentError("template checkpoint " *
        "$template has no `geometry` record in its run_config (it predates " *
        "the geometry key): refusing to write an annulus checkpoint with the " *
        "configuration of an unnamed geometry. Pass an annulus checkpoint " *
        "written by the current code."))
    stored_geom == gid || throw(ArgumentError("template checkpoint $template " *
        "is not the annulus target card: its geometry record is " *
        "\"$stored_geom\" but the import builds \"$gid\". Pass an annulus " *
        "checkpoint (R0=1.22, R1=2.22, Phi=2pi, seam=:periodic)."))
    pr = import_params(cfg)
    # the wall zeroing below is only consistent with the benchmark's Dirichlet
    # data (inner 1, outer 0); a card with other wall temperatures would expect
    # a nonzero theta at the walls and the driver's wall guard would refuse it.
    (pr.Twall_bot == 1.0 && pr.Twall_top == 0.0) ||
        throw(ArgumentError("template checkpoint $template records " *
            "Twall_bot=$(pr.Twall_bot), Twall_top=$(pr.Twall_top); the " *
            "annulus import normalises theta against the benchmark's " *
            "Tbar(R0)=1/Tbar(R1)=0 walls (inner 1, outer 0). Pass a " *
            "benchmark-card checkpoint."))
    qt = annulus_benchmark_tree(nx, ny)
    ss = make_semstokes(qt, 2; geom=g)
    sq = ss.su
    theta = zeros(Float64, sq.N)
    ux = zeros(Float64, sq.N)
    uz = zeros(Float64, sq.N)
    if verbose
        iprintf("[import] state: %s\n", abspath(state))
        iprintf("[import]   frame %d, t = %.9f (nondimensional card: no unit " *
                "conversion), source VTU sha256 = %s\n",
                st.frame, st.t, isempty(st.source_sha) ? "n/a" :
                                    st.source_sha[1:min(end, 16)] * "...")
        iprintf("[import]   grid: %d x %d nodes (%d x %d Q2 cells, r in " *
                "[%.6f, %.6f])\n", st.nr, st.nphi, st.cells_r, st.cells_t,
                st.R0, st.R1)
        iprintf("[import] ring: nx=%d ny=%d, %d leaves, %d Q2 nodes, " *
                "geometry %s\n", nx, ny, length(qt.leaves), sq.N, gid)
        iprintf("[import] template: %s (%d run_config keys, dt = %.3e, " *
                "dt_prev = %.3e, Ra = %g, H = %g)\n", abspath(template),
                length(cfg), tpl.dt, tpl.dt_prev, pr.Ra, pr.H)
    end
    # --- Q2 evaluation exactness at the state's own nodes ------------------
    ex = check_q2_exactness(st)
    (ex.max_T <= 1.0e-12 && ex.max_u <= 1.0e-12) ||
        error("Q2 evaluation at the ASPECT nodes is not exact: max|dT| = " *
              "$(ex.max_T_abs) (relative $(ex.max_T)), max|du| = " *
              "$(ex.max_u_abs) (relative $(ex.max_u)). The interpolation or " *
              "the node ordering is wrong.")
    if verbose
        iprintf("[import] Q2 evaluation at the state's own nodes: max|dT| = " *
                "%.3e (rel %.3e), max|du| = %.3e (rel %.3e) -- exact to " *
                "round-off\n", ex.max_T_abs, ex.max_T, ex.max_u_abs,
                ex.max_u)
    end
    # --- transfer into the ring -------------------------------------------
    ir = interp_ring!(theta, ux, uz, sq, st, pr, g)
    ir.n_seam == nx * 2 + 1 || error("seam check: found $(ir.n_seam) ring " *
        "nodes on the phi = 0 ray, expected 2*nx+1 = $(2 * nx + 1)")
    ir.max_seam <= 1.0e-12 || error("the periodic seam trace is not " *
        "continuous: the same physical nodes evaluated from phi = 0 and " *
        "phi = 2pi differ by $(ir.max_seam) (relative); the angular node " *
        "index wrap (modulo nphi) is wrong.")
    irep = interp_report(sq, st)
    if verbose
        iprintf("[import] transfer into the %d GLL nodes: %d coincide with " *
                "state nodes, max node offset %.3e (r/phi), max |Q2(ring " *
                "node) - nearest state node| = %.3e (relative)\n",
                irep.n_nodes, irep.n_coincident, irep.max_node_offset,
                irep.max_node_value_gap)
        iprintf("[import] walls: %d nodes, %d had |theta| > 1e-8 before " *
                "zeroing (max pre-zero |theta| = %.3e); seam: %d nodes, " *
                "trace jump %.3e (relative)\n", ir.n_wall_nodes,
                ir.nwall, ir.max_wall, ir.n_seam, ir.max_seam)
        # wall free-slip normal velocity: the ASPECT state should already
        # satisfy u . e_r = 0 to its Float32 precision
        mur = 0.0
        for gi in eachindex(sq.coords)
            x, y = sq.coords[gi]
            r = hypot(x, y)
            (r <= ANN_R0 + 1.0e-14 || r >= ANN_R1 - 1.0e-14) || continue
            mur = max(mur, abs((ux[gi] * x + uz[gi] * y) / r))
        end
        iprintf("[import] wall normal velocity of the imported state: " *
                "max|u_r| = %.3e (free-slip data, Float32-derived)\n", mur)
    end
    # --- diagnostics -------------------------------------------------------
    di = state_diagnostics(qt, sq, theta, ux, uz, pr)
    if verbose
        iprintf("[import] imported state: Nu_top = %.9f  Nu_bot = %.9f  " *
                "Vrms = %.9f  Tmax = %.9f  Tmean = %.12f\n",
                di.Nu_top, di.Nu_bot, di.Vrms, di.Tmax, di.Tmean)
    end
    stats = st.attrs
    has_asp = haskey(stats, "aspect_vrms") && haskey(stats, "aspect_meanT")
    if verbose
        if has_asp
            iprintf("[import] ASPECT statistics row at t = %.9f: Nu_top = " *
                    "%.9f  Nu_bot = %.9f  Vrms = %.9f  Tmean = %.12f\n",
                    Float64(stats["aspect_t"]), Float64(stats["aspect_nu_top"]),
                    Float64(stats["aspect_nu_bot"]),
                    Float64(stats["aspect_vrms"]),
                    Float64(stats["aspect_meanT"]))
            iprintf("[import]   vs ASPECT statistics (family C Nu): " *
                    "Nu_top %+.6f%%  Nu_bot %+.6f%%  Vrms %+.3e  Tmean " *
                    "%+.3e\n",
                    100 * reldiff(di.Nu_top, Float64(stats["aspect_nu_top"])),
                    100 * reldiff(di.Nu_bot, Float64(stats["aspect_nu_bot"])),
                    reldiff(di.Vrms, Float64(stats["aspect_vrms"])),
                    reldiff(di.Tmean, Float64(stats["aspect_meanT"])))
            iprintf("[import]   Vrms/Tmean are the interpolation check (they " *
                    "must match to the quadrature accuracy); the Nu columns " *
                    "are different wall-flux families (QuaST family A " *
                    "one-sided GLL gradient vs ASPECT family C " *
                    "consistent-boundary flux) -- the several-percent gap is " *
                    "the family difference, not the state.\n")
        else
            iprintf("[import] state carries no ASPECT statistics row (the " *
                    "extractor had no --stats file): skipping the " *
                    "statistics comparison\n")
        end
        if haskey(stats, "field_vrms")
            iprintf("[import]   vs the extracted Q2 field (family A both): " *
                    "Nu_top %+.3e  Nu_bot %+.3e  Vrms %+.3e  Tmean %+.3e\n",
                    reldiff(di.Nu_top, Float64(stats["field_nu_top"])),
                    reldiff(di.Nu_bot, Float64(stats["field_nu_bot"])),
                    reldiff(di.Vrms, Float64(stats["field_vrms"])),
                    reldiff(di.Tmean, Float64(stats["field_meanT"])))
        end
    end
    result = (state=st, qt=qt, theta=theta, ux=ux, uz=uz, diag=di,
              interp=ir, interp_report=irep, exactness=ex, template=tpl,
              config=cfg, params=pr, checkpoint=nothing)
    (print_only || out === nothing) && return result
    # --- the checkpoint ----------------------------------------------------
    pv = zeros(Float64, ss.sp.N)
    meta = Dict{String,Any}(
        "imported_from" => abspath(state),
        "import_format" => IMPORT_FORMAT,
        "aspect_frame" => st.frame,
        "aspect_t" => st.t,
        "asp_source_vtu_sha256" => st.source_sha,
        "import_nx" => nx, "import_ny" => ny,
        "import_notes" => "theta = T_ASPECT - Tbar(r); walls zeroed; " *
                          "theta_prev = theta; step = 0; pv = 0; run_config " *
                          "from the template checkpoint")
    mkpath(dirname(abspath(out)))
    save_semquad_state(out, qt, theta, copy(theta), ux, uz, pv;
                       t=st.t, dt=tpl.dt, dt_prev=tpl.dt_prev, step=0, p=2,
                       config=cfg, meta=meta, velocity_stale=false)
    # the checkpoint must load again and carry the template's configuration
    chk = load_semquad_state(out)
    chk.config == cfg || error("the written checkpoint's run_config does not " *
        "round-trip the template's: $(length(chk.config)) vs $(length(cfg)) keys")
    chk.t == st.t || error("the written checkpoint's t = $(chk.t) != " *
        "state t = $(st.t)")
    chk.step == 0 || error("the written checkpoint's step = $(chk.step) != 0")
    length(chk.theta) == sq.N || error("the written checkpoint has " *
        "$(length(chk.theta)) theta entries, ring needs $(sq.N)")
    chk.theta == theta || error("the written checkpoint's theta does not " *
        "round-trip the imported field")
    chk.velocity_stale == false || error("the written checkpoint is marked " *
        "velocity_stale")
    if verbose
        iprintf("[import] wrote %s: t = %.9f, step = 0, dt = %.3e, " *
                "dt_prev = %.3e, %d leaves, %d dofs, %d run_config keys\n",
                abspath(out), st.t, tpl.dt, tpl.dt_prev, length(qt.leaves),
                sq.N, length(cfg))
        iprintf("[import] reload check: config %d keys, geometry %s, t = " *
                "%.9f, theta round-trip exact, velocity_stale = false\n",
                length(chk.config), get(chk.config, "geometry", "?"), chk.t)
    end
    return merge(result, (checkpoint=abspath(out), reloaded=chk))
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
function parse_import_args(argv)
    d = Dict{String,String}()
    flags = Set(["print-only", "quiet", "help"])
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("unexpected argument `$a` (arguments are " *
                                     "--flag or --key value pairs)")
        k = a[3:end]
        if k in flags
            d[k] = "true"
            i += 1
        else
            i + 1 <= length(argv) || error("missing value for `$a`")
            d[k] = argv[i+1]
            i += 2
        end
    end
    return d
end

function main_import()
    args = parse_import_args(ARGS)
    if haskey(args, "help")
        println("usage: julia --project=. examples/import_annulus_aspect_state.jl " *
                "--state <asp_state.h5> --template <annulus_checkpoint.h5> " *
                "[--out <checkpoint.h5>] [--nx 16] [--ny 192] [--print-only]")
        return
    end
    for k in ("state", "template")
        haskey(args, k) || error("--$k is required")
    end
    print_only = haskey(args, "print-only")
    out = get(args, "out", nothing)
    (print_only || out !== nothing) ||
        error("--out <checkpoint.h5> is required unless --print-only is given")
    res = import_annulus_state(;
        state=args["state"], template=args["template"], out=out,
        nx=haskey(args, "nx") ? parse(Int, args["nx"]) : 16,
        ny=haskey(args, "ny") ? parse(Int, args["ny"]) : 192,
        print_only=print_only, verbose=!haskey(args, "quiet"))
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_import()
end
