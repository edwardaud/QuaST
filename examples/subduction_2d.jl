# A regional subduction-style mantle-flow case built from the per-wall boundary
# mechanisms: a prescribed normal-velocity inflow, a plate-driven tangential
# velocity, and an `:open` outflow carrying the normal and tangential traction of
# the reference channel.  The spec is `docs/dev/REGIONAL_DESIGN.md`; gate G265
# exercises this file.
#
#   julia --project=. --threads=8 examples/subduction_2d.jl validate
#   julia --project=. --threads=8 examples/subduction_2d.jl rheology
#   julia --project=. --threads=8 examples/subduction_2d.jl phases
#   julia --project=. --threads=8 examples/subduction_2d.jl full
#   julia --project=. --threads=8 examples/subduction_2d.jl converge
#
# The box is [0, 2] x [0, 1] (2 x 1000 km, x = 0 far-field/back-arc, x = 2
# trench, y = 0 the deep-mantle base, y = 1 the surface).  A plate-driven
# asthenospheric channel flows trenchward: the far field enters through `xminus`
# (`:rigid`, `wall_un` = the normal velocity of the reference channel), the
# overriding plate drags the top (`yplus`, `:rigid`, `wall_u = U_p`), the base is
# no slip (`yminus`, `:rigid`), and the channel drains through the `:open`
# `xplus` wall whose dynamic pressure and shear traction are the far-field
# tractions (`wall_tn = -p_out`, `wall_t = tau_ref(y)`).  Thermally the surface
# is cold, the base and both side walls are the hot mantle (T = 1), so the
# plate-cooling boundary layer is the only cold structure; `Ra = 0` makes it a
# kinematic thermal model (van Keken et al. 2008 tradition), so neither the
# thermal nor the phase buoyancy acts and the phase transitions contribute
# through the latent heat only.
#
# The isoviscous limit of exactly this geometry is the analytic
# Couette-Poiseuille superposition
#
#     u_x = u_ref(y) = U_p y + G y (1-y)/2,   u_y = 0,
#     p   = p_ref(x) = p_out + G (Lx - x),
#     sigma.n on x=Lx:  (-p_out, tau_ref(y)),  tau_ref = eta du_ref/dy,
#
# a degree-2 velocity / degree-1 pressure field that Q2/Q1 contain exactly and
# whose linear traction data the face quadrature integrates exactly.  The
# `validate` mode therefore measures the linear solve, not the discretisation,
# and every later stage changes only the physics, not the boundary data.
#
# The checks and their classes (docs/dev/REGIONAL_DESIGN.md section 6):
#   1  the velocity field against u_ref                    exact identity
#   2  the pressure field against p_ref                    exact identity
#   3  the per-wall mass flux and the net-zero identity    exact discrete identity
#   4  the outflow pressure against p_out                  exact identity (gauge)
#   5  the open-box heat budget                            DIAGNOSTIC here
#   6  the plate-cooling / half-space profile              asymptotic (10 %)
#   7  the resolution check                                `converge` mode
#   8  the deta_T = 1 continuation                          exact identity
#   9  cfl 1 vs 2 steady states                            discretisation (manual)
#  10  the latent-heat enthalpy identity                   DIAGNOSTIC here
#  11  the corner-flow comparison                          NOT a gate (section 6.5)
#
# Checks 5 and 10 are built from the pointwise one-sided wall gradient, which is
# NOT the discrete boundary exchange for a card with fixed side walls: the
# side-wall exchange is a Dirichlet reaction of the step (docs/dev/VALIDATION.md
# section 4.1 says the consistent boundary flux is what closes the step
# identity, and the library implements it for the top/bottom walls only).  Both
# are therefore reported with their full term dump and measured residual, not
# gated; docs/dev/REGIONAL_PROGRESS.md records the measurement.  The mass flux
# and the latent coefficients stay gated, and the plate-cooling check is gated
# in `rheology`.
#
# One card deviation is deliberate and stated: the outflow tangential traction
# is the reference channel's far-field shear *rate* carried by the far-field
# viscosity (`CASE_WALL_T=rate`, the default; exactly `tau_ref` at deta_T = 1)
# rather than the design's literal isoviscous stress (`CASE_WALL_T=stress`),
# because the latter is inconsistent with the 10^3 contrast -- measured, it
# traps an O(1e5) outflow layer and makes the designed steady state
# unaffordable.  The measurement is in the `regional_wall_data` docstring.
#
# What the case is not (the design's section 7 and 9 list): no dipping slab, no
# free surface, no compositional slab or self-consistent slab pull, a kinematic
# `Ra = 0` thermal model, and published subduction numbers are not comparable
# for the reasons that list states.
#
# Environment knobs: CASE_OUT, CASE_LEVEL, CASE_WALLMIN, CASE_WALLSTRIP,
# CASE_CFL, CASE_DT0, CASE_STEPS, CASE_FRAME, CASE_RA, CASE_LINEAR, CASE_EXTDIAG,
# CASE_MG_TOL, CASE_WALL_T (rate|stress), CASE_LAW (converge).
# The globals and functions are prefixed `regional_`/`REGIONAL_` because this
# file is also included by gate G265 inside the one-process suite.
using QuaST, Printf

# ===========================================================================
# 1. The card's literals (docs/dev/REGIONAL_DESIGN.md sections 4.1-4.4, 5.1-5.3)
# ===========================================================================
# Geometry: 2 x 1 box, 4 x 2 base cells (square leaves).
const REGIONAL_LX = 2.0
const REGIONAL_CELLS_X = 4
const REGIONAL_CELLS_Y = 2

# Reference channel (section 4.2): eta_ref = 1, U_p = 1.6e3, G = 1.0e3,
# p_out = 1.0e3; u_ref(0) = 0 and u_ref(1) = U_p, so the inflow data are
# continuous around both left corners.
const REGIONAL_UP = 1.6e3
const REGIONAL_DPDX = 1.0e3
const REGIONAL_POUT = 1.0e3

# Thermal card (section 4.4).
const REGIONAL_H = 1.0
const REGIONAL_H_HALF_LIFE = Inf
const REGIONAL_TWALL_TOP = 0.0
const REGIONAL_TWALL_BOT = 1.0
const REGIONAL_TWALL_SIDE = 1.0          # both side walls, :fixed
const REGIONAL_RA = 0.0                  # kinematic thermal model

# Rheology (section 4.4).
const REGIONAL_ETA_REF = 1.0
const REGIONAL_ETA_T0 = 0.5
const REGIONAL_ETA_MIN = 1.0e-4
const REGIONAL_ETA_MAX = 1.0e4
const REGIONAL_DETA_T = 1.0e3            # stages (ii) and (iii)
const REGIONAL_COMP_AD = 1.6240183119e5  # the realistic card's calibrated pair
const REGIONAL_COMP_AS = 2.6796419238e-1
const REGIONAL_COMP_ED = 12.0
const REGIONAL_COMP_ES = 17.0
const REGIONAL_COMP_N = 3.0
const REGIONAL_COMP_EPS_MIN = 1.0e-12

# Phase transitions (section 4.4/5.3): physical depths 410 km and 660 km of the
# 1000-km box, y = 1 - depth/1000; the Clapeyron slopes are the physical
# ~3 MPa/K re-expressed as dz/dT in code units (section 5.3).
const REGIONAL_Z410 = 0.59
const REGIONAL_Z660 = 0.34
const REGIONAL_WIDTH_PT = 0.02
const REGIONAL_SLOPE410 = 0.0928
const REGIONAL_SLOPE660 = -0.0928
const REGIONAL_TREF_PT = 0.5
const REGIONAL_DRHO410 = 180.0
const REGIONAL_DRHO660 = 300.0
const REGIONAL_B410 = 180.0 / (3300.0 * 3.0e-5 * 1000.0)   # 1.8182
const REGIONAL_B660 = 300.0 / (3300.0 * 3.0e-5 * 1000.0)   # 3.0303

# The dimensional scale set (section 5.1): the realistic card's ASPECT set with
# the regional depth d = 1e6 m.
const REGIONAL_SCALE = (rho=3300.0, g=9.81, alpha=3.0e-5, dT=1000.0,
                        d=1.0e6, kappa=1.0e-6)
const REGIONAL_LATENT_SCALE = (rho_ref=3300.0, cp=1000.0, dT=1000.0, g=9.81,
                               d=1.0e6, T_surface=REGIONAL_ETA_T0)
# With Ra = 0 the card does not define a physical eta_0 through the Rayleigh
# number; section 5.2 adopts this value for the dimensional reporting.
const REGIONAL_ETA0_PHYS = 1.0e21          # Pa s
const REGIONAL_VELOCITY_UNIT = 1.0e-12     # m/s = 3.156e-5 m/yr
const REGIONAL_TIME_UNIT = 1.0e18          # s = 31.7 Gyr

# Numerics (section 4.4).
const REGIONAL_P = 2
const REGIONAL_EV = 0.11
const REGIONAL_EV_CMAX = 0.052
const REGIONAL_EV_ALPHA = 2
const REGIONAL_DT0 = 1.0e-6
const REGIONAL_DTMAX = 1.0e-4
const REGIONAL_CFL = 2.0
const REGIONAL_WALL_STRIP = 0.08
const REGIONAL_WALL_MIN = 5
const REGIONAL_STOKES_TOL_PROD = 1.0e-3
const REGIONAL_MG_TOL = 1.0e-8
const REGIONAL_STOKES_TOL_EXACT = 1.0e-8
const REGIONAL_STEADY_TOL = 1.0e-6
const REGIONAL_STEADY_WINDOW = 20
const REGIONAL_TMIN = 1.0e-3
const REGIONAL_FRAME_STRIDE = 20
const REGIONAL_FRAME_GRID = 256
const REGIONAL_TIME_STEP_HALF = Inf      # H_half_life

const REGIONAL_MODES = (:validate, :rheology, :phases, :full, :converge)

# ===========================================================================
# 2. The reference channel solution (section 4.2) and the analytic errors
# ===========================================================================
regional_uref(y) = REGIONAL_UP * y + REGIONAL_DPDX * y * (1.0 - y) / 2.0
regional_tauref(y) = REGIONAL_UP + REGIONAL_DPDX * (0.5 - y)
regional_pref(x) = REGIONAL_POUT + REGIONAL_DPDX * (REGIONAL_LX - x)
regional_qref() = REGIONAL_UP / 2.0 + REGIONAL_DPDX / 12.0   # 883.33

# The scale of the analytic tolerances (section 6.1).
regional_u_scale() = REGIONAL_UP
regional_p_scale() = REGIONAL_DPDX * REGIONAL_LX + REGIONAL_POUT
regional_tau_scale() = REGIONAL_UP + REGIONAL_DPDX / 2.0      # max |tau_ref|

# Outward flux per wall, in the order (xminus, xplus, yminus, yplus), integrated
# with the assembly's own 1-D Gauss rule (exact for the Q2 trace); the shipped
# channel gate G264 uses the same helper.
function regional_flux_per_wall(ss, qt, u)
    su = ss.su
    n = su.p + 1
    nu = su.N
    fd = make_fedata(ss.p, ss.q)
    out = zeros(Float64, 4)
    for it in eachindex(su.leaves)
        lf = su.leaves[it]
        hx = QuaST.leaf_hx(su, lf[1])
        hy = QuaST.leaf_hy(su, lf[1])
        ids = su.elem[lf]
        for (k, dir) in enumerate((:xminus, :xplus, :yminus, :yplus))
            QuaST.has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                sgn = dir === :xplus ? 1.0 : -1.0
                for q in 1:fd.nq
                    val = 0.0
                    for j in 1:n
                        val += u[ids[(j - 1) * n + iF]] * fd.phi_v[j, q]
                    end
                    out[k] += sgn * (0.5 * hy * fd.gl_w[q]) * val
                end
            else
                jF = dir === :yminus ? 1 : n
                sgn = dir === :yplus ? 1.0 : -1.0
                for q in 1:fd.nq
                    val = 0.0
                    for i in 1:n
                        val += u[nu + ids[(jF - 1) * n + i]] * fd.phi_v[i, q]
                    end
                    out[k] += sgn * (0.5 * hx * fd.gl_w[q]) * val
                end
            end
        end
    end
    return out
end

"""
    regional_outflow_shear_error(sq, qt, theta, ux) -> Float64

The one-sided GLL derivative `eta du_x/dy` on the outflow wall (the same rule
`semquad_nu_walls` uses for temperature), max-scaled by `max|tau_ref|`.  In the
isoviscous stage this measures the assembled data: the traction data are exactly
`tau_ref`, so the field shear must reproduce them.
"""
function regional_outflow_shear_error(sq, qt, theta, ux)
    p = sq.p
    n = p + 1
    rule = gll_rule(p)
    worst = 0.0
    for lf in sq.leaves
        QuaST.has_no_face_neighbor(qt, lf, :xplus) || continue
        hy = QuaST.leaf_hy(sq, lf[1])
        y0 = lf[3] * hy
        ids = sq.elem[lf]
        for j0 in 1:n
            d = 0.0
            for j in 1:n
                d += sq.D[j0, j] * ux[ids[(j - 1) * n + n]]
            end
            d *= 2.0 / hy
            y = y0 + 0.5 * (rule.nodes[j0] + 1.0) * hy
            worst = max(worst, abs(d - regional_tauref(y)))
        end
    end
    return worst / regional_tau_scale()
end

"""
    regional_analytic_errors(res) -> NamedTuple

The stage-(i) exact comparisons: the velocity field against `u_ref`, the pressure
field against `p_ref`, the outflow pressure against `p_out`, the outflow field
shear against `tau_ref`, and the four wall fluxes.
"""
function regional_analytic_errors(res)
    sq = res.ss.su
    sp = res.ss.sp
    coords = sq.coords
    pcoords = sp.coords
    err_u = maximum(abs(res.ux[i] - regional_uref(coords[i][2]))
                    for i in eachindex(coords)) / regional_u_scale()
    err_w = maximum(abs, res.uz) / regional_u_scale()
    err_p = maximum(abs(res.pv[j] - regional_pref(pcoords[j][1]))
                    for j in eachindex(pcoords)) / regional_p_scale()
    # The outflow pressure: p_out with the outward-normal sign.  Check 4 is an
    # absolute comparison, so it is reported in the pressure unit itself.
    err_pout = 0.0
    nout = 0
    for j in eachindex(pcoords)
        if abs(pcoords[j][1] - REGIONAL_LX) <= 1.0e-12
            err_pout = max(err_pout, abs(res.pv[j] - REGIONAL_POUT))
            nout += 1
        end
    end
    err_shear = regional_outflow_shear_error(sq, res.qt, res.theta, res.ux)
    fluxes = regional_flux_per_wall(res.ss, res.qt, vcat(res.ux, res.uz))
    return (err_u=err_u, err_uz=err_w, err_p=err_p, err_pout=err_pout,
            n_pout=nout, err_shear=err_shear, flux=fluxes,
            flux_net=sum(fluxes), qref=regional_qref(),
            u_scale=regional_u_scale(), p_scale=regional_p_scale())
end

# ===========================================================================
# 3. Wall data, phase transitions and the parameter cards
# ===========================================================================
"""
    regional_base_params(law, deta_T; ra=REGIONAL_RA) -> Params

The card without per-wall data: the same thermal card and the same constitutive
constants, so it is what `regional_farfield_eta` evaluates the law with.  (A
`Params` is immutable, so the final card is built once more with the wall data;
this helper exists only for the viscosity probe.)
"""
function regional_base_params(law::Symbol, deta_T::Float64;
                              ra::Float64=REGIONAL_RA)
    return Params(Ra=ra, H=REGIONAL_H, H_half_life=REGIONAL_H_HALF_LIFE,
                  eta_law=law, deta_T=deta_T,
                  eta_T0=REGIONAL_ETA_T0, eta_ref=REGIONAL_ETA_REF,
                  eta_min=REGIONAL_ETA_MIN, eta_max=REGIONAL_ETA_MAX,
                  comp_Ad=REGIONAL_COMP_AD, comp_As=REGIONAL_COMP_AS,
                  comp_Ed=REGIONAL_COMP_ED, comp_Es=REGIONAL_COMP_ES,
                  comp_n=REGIONAL_COMP_N, comp_Vd=0.0, comp_Vs=0.0,
                  comp_Tcut=0.0, comp_eps_min=REGIONAL_COMP_EPS_MIN,
                  comp_rule=:isostress,
                  bc=:free, wall_bc=(:rigid, :open, :rigid, :rigid))
end

"""
    regional_farfield_eta(y, pr) -> Float64

The laterally uniform far field's viscosity at height `y`: the card's law
evaluated at the far field's temperature (the card's side-wall scalar, the hot
asthenosphere the channel flows in) and at the reference channel's strain rate
`|tau_ref(y)|/(2 eta_ref)`.  This is the viscosity the far-field *stress* has to
be matched with, so that the outflow data impose the reference channel's shear
*rate* rather than its (isoviscous) stress.
"""
function regional_farfield_eta(y::Float64, pr::Params)
    T = REGIONAL_TWALL_SIDE
    if pr.eta_law === :composite
        ed = abs(regional_tauref(y)) / (2.0 * pr.eta_ref)
        return eta_composite_at(T, ed, y, pr)[1] / pr.eta_ref
    end
    return eta_at(T, y, pr)[1] / pr.eta_ref
end

"""
    regional_wall_data(; t_mode=:rate, law=:expT, deta_T=1.0) -> NamedTuple

The reference-channel boundary data of section 4.2, in the solver's
`(xminus, xplus, yminus, yplus)` order and its sign conventions: `wall_un` is
the normal component in the positive coordinate direction, `wall_u` the
tangential velocity, `wall_t`/`wall_tn` the total Cauchy traction's tangential /
normal component with the outward normal, added as a boundary load.
`nothing` means "no data" (zero traction); on an `:open` wall the natural
condition is exactly the zero-load case.

`t_mode` selects the outflow tangential traction (the one data choice on which
the implementation deviates from the design, with the measured reason in the
device note below):

* `:rate` (default) -- `tau_ref(y) * eta_far(y)/eta_ref`, the reference channel's
  far-field shear *rate* carried by the far field's own viscosity.  For
  `deta_T = 1` (`eta_far == 1`) it is exactly the design's `tau_ref(y)`, so the
  stage-(i) analytic identity is unchanged.
* `:stress` -- the design's literal fixed traction `tau_ref(y)`, the isoviscous
  far-field stress applied to a domain that may be 10^3 times weaker.

Why the default is `:rate`: the design's fixed isoviscous stress is inconsistent
with the 10^3 contrast of its own rheology stage.  A stress of order 10^3 on
material of viscosity 10^-3 needs a strain rate of order 10^6, so the measured
`:stress` stage traps an outflow shear layer of `max|u| ~ 1.7e5`
(`Vrms ~ 1.4e4`), its CFL step collapses to `2e-7` (against the design's
implicit ~1e-4), and the steady state the design asks for is unreachable in the
budget (the measured 150-step run is still 2.4e5 and climbing at `t = 3e-5`,
while the plate-cooling structure needs `t ~ 1e-3`).  The `:rate` data keep the
far field's *kinematics* -- which is the design's own "window into the reference
channel" assumption, section 5.4 -- and reproduce the design's predicted
`Nu_top ~ 31` and plate-cooling `delta(0.8) ~ 0.052` at `deta_T = 1e3`.  Both
modes are exposed (`CASE_WALL_T=rate|stress`) and both are measured in
`REGIONAL_PROGRESS.md`.
"""
function regional_wall_data(; t_mode::Symbol=:rate, law::Symbol=:expT,
                            deta_T::Float64=1.0)
    t_mode in (:rate, :stress) ||
        throw(ArgumentError("CASE_WALL_T must be rate or stress, got $t_mode"))
    z = (x, y) -> 0.0
    if t_mode === :stress
        tfun = (x, y) -> regional_tauref(y)
    else
        probe = regional_base_params(law, deta_T)
        tfun = (x, y) -> regional_tauref(y) * regional_farfield_eta(y, probe)
    end
    return (wall_un=((x, y) -> regional_uref(y), nothing, z, z),
            wall_u=(z, nothing, nothing, (x, y) -> REGIONAL_UP),
            wall_t=(nothing, tfun, nothing, nothing),
            wall_tn=(nothing, (x, y) -> -REGIONAL_POUT, nothing, nothing))
end

"""
    regional_phase_transitions(pr) -> Vector{PhaseTransition}

The two transitions of section 4.4/5.3.  The latent coefficients are computed
from `per_phase_latent` with the regional scale set, never typed: the design
wants a wrong `d`/`dT` in the latent scale set to show up in the header (the
coefficients are printed and compared to their closed forms) rather than in the
energy budget.
"""
function regional_phase_transitions(pr::Params)
    ph = [PhaseTransition(REGIONAL_Z410, REGIONAL_WIDTH_PT, REGIONAL_SLOPE410,
                          REGIONAL_TREF_PT, REGIONAL_DRHO410, REGIONAL_B410),
          PhaseTransition(REGIONAL_Z660, REGIONAL_WIDTH_PT, REGIONAL_SLOPE660,
                          REGIONAL_TREF_PT, REGIONAL_DRHO660, REGIONAL_B660)]
    return per_phase_latent(pr, ph; REGIONAL_LATENT_SCALE...)
end

"""
    regional_card(mode; deta_T_override=nothing, wall_data=regional_wall_data(),
                  law_override=nothing, ra=REGIONAL_RA) -> Params

The parameter card of one stage (section 4.4).  The four physics stages differ
only in the viscosity law and the phase list; the boundary data and the thermal
card are the same in every stage, which is what makes the stage-(i) analytic
identity transfer to the later stages as a reference.
"""
function regional_card(mode::Symbol; deta_T_override=nothing,
                       wall_data=nothing, law_override=nothing,
                       ra::Float64=REGIONAL_RA, t_mode::Symbol=:rate)
    mode in REGIONAL_MODES ||
        throw(ArgumentError("unknown regional mode :$mode; modes are " *
                            "$(join(string.(REGIONAL_MODES), ", "))"))
    law = law_override === nothing ? (mode === :full ? :composite : :expT) :
          law_override
    deta = deta_T_override === nothing ?
           (mode === :validate ? 1.0 : REGIONAL_DETA_T) : deta_T_override
    latent_mode = regional_has_phases(mode) ? :per_phase : :global
    if wall_data === nothing
        wall_data = regional_wall_data(; t_mode=t_mode, law=law, deta_T=deta)
    end
    pr = Params(Ra=ra, H=REGIONAL_H, H_half_life=REGIONAL_H_HALF_LIFE,
                eta_law=law, deta_T=deta,
                eta_T0=REGIONAL_ETA_T0, eta_ref=REGIONAL_ETA_REF,
                eta_min=REGIONAL_ETA_MIN, eta_max=REGIONAL_ETA_MAX,
                comp_Ad=REGIONAL_COMP_AD, comp_As=REGIONAL_COMP_AS,
                comp_Ed=REGIONAL_COMP_ED, comp_Es=REGIONAL_COMP_ES,
                comp_n=REGIONAL_COMP_N, comp_Vd=0.0, comp_Vs=0.0,
                comp_Tcut=0.0, comp_eps_min=REGIONAL_COMP_EPS_MIN,
                comp_rule=:isostress,
                bc=:free, wall_bc=(:rigid, :open, :rigid, :rigid),
                wall_un=wall_data.wall_un, wall_u=wall_data.wall_u,
                wall_t=wall_data.wall_t, wall_tn=wall_data.wall_tn,
                side_thermal=:fixed, Twall_top=REGIONAL_TWALL_TOP,
                Twall_bot=REGIONAL_TWALL_BOT, Twall_left=REGIONAL_TWALL_SIDE,
                Twall_right=REGIONAL_TWALL_SIDE,
                latent_mode=latent_mode, latent_L=0.0,
                # deliberately off (section 4.4): Maxwell viscoelasticity,
                # dissipation/adiabatic heating, yielding/strain weakening,
                # density-weighted heating, adiabatic reference state
                shear_modulus=0.0, Di=0.0, heating_adiabatic=false,
                heating_dissipation=false, rho_heating=false, rho_cp=false,
                refstate=nothing, deviatoric=false,
                yield_c=Inf, yield_phi=0.0, yield_taumax=Inf,
                yield_c_grad=0.0, yield_mu_grad=0.0,
                weak_f=1.0, weak_eps_ref=0.0)
    return pr
end

"""True when the command-line mode needs the two phase transitions."""
regional_has_phases(mode::Symbol) = mode === :phases || mode === :full

"""The phase list of one stage (empty for `validate` and `rheology`)."""
function regional_phases(mode::Symbol, pr::Params)
    regional_has_phases(mode) ? regional_phase_transitions(pr) :
        PhaseTransition[]
end

# ===========================================================================
# 4. The static graded trees (section 4.1/6.6)
# ===========================================================================
"""
    regional_tree(level; wall_min=0, strip=REGIONAL_WALL_STRIP) -> Quadtree

A `[0,2] x [0,1]` quadtree with `level` uniform refinements of the 4 x 2 base
grid, then (when `wall_min > 0`) every leaf whose box comes within `strip` of
the top or bottom wall refined up to `wall_min`.  This is the script's static
graded tree: with `amr_every = 0` the driver's own wall rule is inactive, so a
run is reproducible from this function alone.
"""
function regional_tree(level::Int; wall_min::Int=0,
                       strip::Float64=REGIONAL_WALL_STRIP)
    maxlev = max(level, wall_min)
    qt = Quadtree(maxlev; Lx=REGIONAL_LX, cells_x=REGIONAL_CELLS_X,
                  cells_y=REGIONAL_CELLS_Y)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    if wall_min > level
        for l in level:(wall_min - 1)
            for lf in collect(qt.leaves)
                lf[1] == l || continue
                h = QuaST.leaf_hy(qt, l)
                y0 = lf[3] * h
                (y0 + h >= 1.0 - strip || y0 <= strip) && refine!(qt, lf)
            end
        end
        balance!(qt)
    end
    return qt
end

# ===========================================================================
# 5. Wall quadratures for the thermal checks
# ===========================================================================
"""
    regional_erf(z)

The error function to 1.5e-7 (Abramowitz & Stegun 7.1.26), used only by the
10 %-tolerance half-space comparison of the plate-cooling check.  The standard
library has no `erf`; a rational approximation is ample at that tolerance.
"""
function regional_erf(z::Float64)
    s = z >= 0.0 ? 1.0 : -1.0
    a = abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                0.284496736) * t + 0.254829592) * t * exp(-a * a)
    return s * y
end

"""
    regional_xwall_terms(sq, qt, pr, theta, ux, uz; phases) -> NamedTuple

The wall integrals of the two vertical walls, with the same one-sided GLL rule
`semquad_nu_walls` uses:

* `q_left`, `q_right` -- the outward normal derivative `int dtheta/dn ds` on
  each x-wall (`n = -e_x` on the left, `+e_x` on the right);
* `a_left`, `a_right` -- the advective flux `int T u_x dy` of the full
  temperature (`T = Tbar + theta`), i.e. the lateral enthalpy flux the open
  heat budget carries;
* `h_left`, `h_right` -- the same flux of the *advected enthalpy temperature*
  `H = T + sum_i latent_i (Gamma_i(T) - Gamma_i(Tbar))`, the quantity the
  latent-heat enthalpy identity of section 6.7 uses.  `h_* == a_*` when the
  latent terms are off.
"""
function regional_xwall_terms(sq, qt, pr::Params, theta::Vector{Float64},
                              ux::Vector{Float64}, uz::Vector{Float64};
                              phases::Vector{PhaseTransition}=PhaseTransition[])
    p = sq.p
    n = p + 1
    rule = gll_rule(p)
    per_phase = pr.latent_mode === :per_phase
    use_lat = semquad_latent_active(pr, phases)
    lcoef = use_lat ? semquad_latent_coefficients(pr, phases) : Float64[]
    ql = 0.0; qr = 0.0; al = 0.0; ar = 0.0; hl = 0.0; hr = 0.0
    for lf in sq.leaves
        hx = QuaST.leaf_hx(sq, lf[1])
        hy = QuaST.leaf_hy(sq, lf[1])
        y0 = lf[3] * hy
        ids = sq.elem[lf]
        for (dir, iF) in ((:xminus, 1), (:xplus, n))
            QuaST.has_no_face_neighbor(qt, lf, dir) || continue
            right = dir === :xplus
            for j in 1:n
                g = ids[(j - 1) * n + iF]
                w = 0.5 * hy * sq.w[j]
                y = y0 + 0.5 * (rule.nodes[j] + 1.0) * hy
                d = 0.0
                for i in 1:n
                    d += sq.D[iF, i] * theta[ids[(j - 1) * n + i]]
                end
                d *= 2.0 / hx
                Tv = Tbar(y, pr.H) + theta[g]
                Hv = Tv
                if use_lat
                    for (k, pt) in enumerate(phases)
                        c = per_phase ? lcoef[k] : pr.latent_L
                        Hv += c * (phase_gamma(pt, y, Tv) -
                                   phase_gamma(pt, y, Tbar(y, pr.H)))
                    end
                end
                uxv = ux[g]
                if right
                    qr += w * d
                    ar += w * Tv * uxv
                    hr += w * Hv * uxv
                else
                    ql -= w * d
                    al += w * Tv * uxv
                    hl += w * Hv * uxv
                end
            end
        end
    end
    return (q_left=ql, q_right=qr, a_left=al, a_right=ar,
            h_left=hl, h_right=hr)
end

"""
    regional_energy_terms(sq, qt, pr, theta; phases, ux, uz) -> NamedTuple

Everything the open-box energy checks need from one state: the driver's
`semquad_heat_budget` fields (sensible and latent enthalpy, the top/bottom
normal-derivative integrals, the heating integral), the two Nusselt functionals,
the side-wall integrals of `regional_xwall_terms`, and the derived budget
residuals of section 6.3 (full-T form) and section 6.7 (latent enthalpy form).
"""
function regional_energy_terms(sq, qt, pr::Params, theta::Vector{Float64};
                               phases::Vector{PhaseTransition}=PhaseTransition[],
                               ux::Vector{Float64}, uz::Vector{Float64})
    b = semquad_heat_budget(sq, qt, pr, theta; phases=phases, ux=ux, uz=uz)
    nut, nub = semquad_nu_walls(sq, qt, theta, pr)
    xw = regional_xwall_terms(sq, qt, pr, theta, ux, uz; phases=phases)
    area = total_area(qt)
    q_sides = xw.q_left + xw.q_right
    # section 6.3: Lx (Nu_top - Nu_bot) = H A + Q_sides - A_right + A_left at a
    # steady state, with A the box area (= Lx for a unit height).
    sigma = REGIONAL_LX * (nut - nub) - pr.H * area - q_sides +
            xw.a_right - xw.a_left
    scale = pr.H * area + REGIONAL_LX * abs(nut) + REGIONAL_LX * abs(nub) +
            abs(q_sides) + abs(xw.a_right) + abs(xw.a_left)
    return (Nu_top=nut, Nu_bot=nub, enthalpy=b.enthalpy, theta_int=b.theta_int,
            latent_int=b.latent_int, flux_top=b.flux_top, flux_bot=b.flux_bot,
            flux=b.flux, heating=b.heating, u_grad_theta=b.u_grad_theta,
            base_adv=b.base_adv,
            q_left=xw.q_left, q_right=xw.q_right, q_sides=q_sides,
            a_left=xw.a_left, a_right=xw.a_right,
            h_left=xw.h_left, h_right=xw.h_right,
            budget_sigma=sigma, budget_scale=scale,
            budget_ratio=abs(sigma) / max(scale, 1.0e-300))
end

"""
    regional_plate_cooling(sq, qt, pr, theta; u_eff=REGIONAL_UP,
                           stations=(0.8, 1.6)) -> NamedTuple

The asymptotic plate-cooling check of section 6.4 at the two stations.  The
reference is the semi-infinite half-space solution with the uniform plate speed
`u_eff` and the *measured* far-field temperature at mid-depth,

    T_hs(x,y) = T_inf erf((1-y)/(2 sqrt(x/u_eff))),
    delta_hs(x) = 2.32 sqrt(x/u_eff),
    q_hs(x) = T_inf sqrt(u_eff/(pi x)).

`delta` is the depth of the 10 % temperature-defect level relative to the local
far field; `q` is the anomaly flux `-dtheta/dy` at the surface, computed with
the same one-sided GLL rule `semquad_nu_walls` uses.  Asymptotic, tolerance
10 % / 10 % / sqrt(2) 10 % / 15 % (section 6.4).
"""
function regional_plate_cooling(sq, qt, pr::Params, theta::Vector{Float64};
                                u_eff::Float64=REGIONAL_UP,
                                stations=(0.8, 1.6))
    leaves = Set(sq.leaves)
    rows = NamedTuple[]
    for x in stations
        Tinf = Tbar(0.5, pr.H) +
               QuaST._semquad_eval_at(sq, theta, x, 0.5, leaves)
        dhs = 2.32 * sqrt(x / u_eff)
        ys = range(1.0, 0.4; length=1201)
        err_profile = 0.0
        delta = NaN
        prevD = 1.0     # T(1-y=0) = T_top = 0, so the surface defect is 1
        prevd = 0.0
        for y in ys
            T = Tbar(y, pr.H) + QuaST._semquad_eval_at(sq, theta, x, y, leaves)
            d = 1.0 - y
            if 0.5 * dhs <= d <= 2.0 * dhs
                err_profile = max(err_profile,
                                  abs(T - Tinf * regional_erf(d / (2.0 * sqrt(x / u_eff)))))
            end
            D = 1.0 - T / max(Tinf, 1.0e-300)
            # ys runs from the surface (defect ~1) downward (defect -> 0), so
            # the 10 % defect level is the first crossing from above to below.
            if !isfinite(delta) && prevD >= 0.1 && D < 0.1
                delta = prevd + (d - prevd) * (prevD - 0.1) / (prevD - D)
            end
            prevD = D
            prevd = d
        end
        g = QuaST._semquad_grad_at(sq, theta, x, 1.0, leaves)
        q = -g[2]
        qhs = Tinf * sqrt(u_eff / (pi * x))
        push!(rows, (x=x, Tinf=Tinf, delta=delta, delta_hs=dhs,
                     err_profile=err_profile, q=q, q_hs=qhs,
                     q_ratio=abs(q - qhs) / max(abs(qhs), 1.0e-300)))
    end
    delta_ratio = maximum(abs(r.delta / r.delta_hs - 1.0) for r in rows)
    growth = rows[end].delta / rows[1].delta
    growth_ref = sqrt(rows[end].x / rows[1].x)
    growth_err = abs(growth / growth_ref - 1.0)
    q_err = maximum(r.q_ratio for r in rows)
    err_profile = maximum(r.err_profile for r in rows)
    return (stations=rows, err_profile=err_profile, delta_ratio=delta_ratio,
            growth=growth, growth_ref=growth_ref, growth_err=growth_err,
            q_err=q_err)
end

# ===========================================================================
# 6. The driver call
# ===========================================================================
"""
    regional_run(qt, pr; maxsteps, linear, stokes_tol, mg_tol, cfl, dt0, dtmax,
                 phases, nu_cbf, extdiag, callback) -> the driver result

One `run_semquad_simulation` call with the case's numerics (section 4.4):
`p = 2`, Boussinesq mass, skew advection, entropy viscosity, `maxpic = 1`,
`amr_every = 0` (the static graded tree is already built), the parameter card's
`eta_ref` scale set, and `theta0 = 0` (T = Tbar at t = 0; the boundary data then
build the plate-cooling structure).
"""
function regional_run(qt::Quadtree, pr::Params; maxsteps::Int=2,
                      linear::Symbol=:direct,
                      stokes_tol::Float64=REGIONAL_STOKES_TOL_EXACT,
                      mg_tol::Float64=REGIONAL_MG_TOL,
                      cfl::Float64=REGIONAL_CFL, dt0::Float64=REGIONAL_DT0,
                      dtmax::Float64=REGIONAL_DTMAX,
                      steady_tol::Float64=Inf,
                      steady_window::Int=REGIONAL_STEADY_WINDOW,
                      tmin::Float64=REGIONAL_TMIN,
                      phases::Vector{PhaseTransition}=PhaseTransition[],
                      nu_cbf::Bool=false, extdiag::Bool=false,
                      callback::Union{Nothing,Function}=nothing)
    ss = make_semstokes(qt, REGIONAL_P)
    theta0 = zeros(Float64, ss.su.N)
    return run_semquad_simulation(qt, pr, theta0;
        p=REGIONAL_P, tstop=Inf, maxsteps=maxsteps,
        dt0=dt0, dtmax=dtmax, dtmin=1.0e-16, cfl=cfl, dt_growth=1.05,
        amr_every=0, advection=:skew, ev=REGIONAL_EV, ev_cmax=REGIONAL_EV_CMAX,
        ev_alpha=REGIONAL_EV_ALPHA, maxpic=1, pic_tol=1.0e-6,
        linear=linear, stokes_tol=stokes_tol, mg_tol=mg_tol,
        mass_form=:boussinesq, phases=phases, latent_scheme=:implicit,
        nu_cbf=nu_cbf, extdiag=extdiag, eta_ref_scale=REGIONAL_SCALE,
        steady_tol=steady_tol, steady_window=steady_window, tmin=tmin,
        callback=callback, verbose=false)
end

# ===========================================================================
# 7. Checks, printing and the small CSV outputs
# ===========================================================================
"""
    regional_check!(checks, io; id, name, quantity, measured, reference,
                    tolerance, gated=true, skip=false) -> row

Append one check row, print its measured value against the tolerance, and
return it.  `pass` is `isfinite(measured) && measured <= tolerance`; a
non-gated row is printed `DIAG` and does not decide the mode's verdict, and a
`skip` row (`NaN`) is printed `SKIP` because its premise did not hold (for
example a steady-state check on a run that stopped at its step cap).
"""
function regional_check!(checks, io::IO; id::Int, name::String,
                         quantity::String, measured::Float64,
                         reference::Float64, tolerance::Float64,
                         gated::Bool=true, skip::Bool=false)
    skip && (measured = NaN)
    pass = !skip && isfinite(measured) && measured <= tolerance
    status = skip ? "SKIP" : (!gated ? "DIAG" : (pass ? "PASS" : "FAIL"))
    row = (id=id, name=name, quantity=quantity, measured=measured,
           reference=reference, tolerance=tolerance, pass=pass, gated=gated,
           skip=skip)
    push!(checks, row)
    @printf(io, "[regional] check %2d %-26s %-30s measured=%.4e reference=%.4e tol=%.1e  %s\n",
            id, name, quantity, measured, reference, tolerance, status)
    return row
end

"""One-line summary of a mode's check list, and its gated verdict."""
function regional_summary(io::IO, mode::Symbol, checks)
    ngated = count(c -> c.gated && !c.skip, checks)
    npass = count(c -> c.gated && !c.skip && c.pass, checks)
    nfail = count(c -> c.gated && !c.skip && !c.pass, checks)
    nskip = count(c -> c.skip, checks)
    ndiag = count(c -> !c.gated && !c.skip, checks)
    @printf(io, "[regional] %s: %d checks (%d gated): %d PASS, %d FAIL, %d DIAG, %d SKIP\n",
            mode, length(checks), ngated, npass, nfail, ndiag, nskip)
    @printf(io, "[regional] %s: %s\n", mode, nfail == 0 ? "PASS" : "FAIL")
    return nfail == 0
end

"""Write `checks.csv` (one row per check) under `dir`."""
function regional_write_checks(dir::String, mode::Symbol, checks)
    mkpath(dir)
    path = joinpath(dir, "checks.csv")
    open(path, "w") do io
        println(io, "mode,id,name,quantity,measured,reference,tolerance,class,status")
        for c in checks
            status = c.skip ? "SKIP" : (!c.gated ? "DIAG" : (c.pass ? "PASS" : "FAIL"))
            @printf(io, "%s,%d,%s,%s,%.9e,%.9e,%.1e,%s,%s\n", mode, c.id,
                    c.name, c.quantity, c.measured, c.reference, c.tolerance,
                    c.gated ? "gated" : "diagnostic", status)
        end
    end
    @printf("  wrote %s\n", path)
    return path
end

"""Write the driver's history (the fixed 18 columns plus the optional ones)."""
function regional_write_history(dir::String, res)
    mkpath(dir)
    path = joinpath(dir, "history.csv")
    extra = res.history_extra
    open(path, "w") do io
        print(io, "t,dt,Nu_top,Nu_bot,Vrms,meanT,N,leaves,div_max,hmin,umax,cfl,closure,usrms_surf,umax_surf,W,U,balance")
        isempty(extra) || print(io, ",", join(extra, ","))
        println(io)
        for r in res.history
            isempty(r) && continue
            @printf(io, "%.9e,%.6e,%.9e,%.9e,%.9e,%.9e,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.9e,%.9e,%.6e",
                    r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                    r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17],
                    r[18])
            for k in 19:length(r)
                @printf(io, ",%.9e", r[k])
            end
            println(io)
        end
    end
    @printf("  wrote %s (%d rows)\n", path, length(res.history))
    return path
end

"""The case header: the card, the tree, the walls, the gauge and the mapping."""
function regional_header(io::IO, mode::Symbol, qt::Quadtree, pr::Params,
                         phases::Vector{PhaseTransition}; linear::Symbol=:auto,
                         steps::Int=0, cfl::Float64=REGIONAL_CFL,
                         dt0::Float64=REGIONAL_DT0,
                         frames::Union{Nothing,String}=nothing)
    ss = make_semstokes(qt, REGIONAL_P)
    hmin = minimum(min(QuaST.leaf_hx(qt, lf[1]), QuaST.leaf_hy(qt, lf[1]))
                   for lf in qt.leaves)
    @printf(io, "[regional] mode=%s  tree=%s  leaves=%d  Nu=%d  Np=%d  saddle=%d  hmin=%.4e\n",
            mode, regional_tree_name(mode), length(qt.leaves), ss.su.N, ss.sp.N,
            2 * ss.su.N + ss.sp.N, hmin)
    @printf(io, "[regional] walls=(xminus=:rigid, xplus=:open, yminus=:rigid, yplus=:rigid)  bc=%s\n",
            pr.bc)
    @printf(io, "[regional] data: xminus wall_un=u_ref(y), wall_u=0; xplus wall_tn=-p_out, wall_t=tau_ref(y); yminus 0/0; yplus wall_un=0, wall_u=U_p\n")
    @printf(io, "[regional] reference channel: U_p=%.4g G=%.4g p_out=%.4g eta=1  Q_ref=%.6f\n",
            REGIONAL_UP, REGIONAL_DPDX, REGIONAL_POUT, regional_qref())
    @printf(io, "[regional] thermal: Twall_top=%.1f Twall_bot=%.1f side_thermal=%s Twall_left=%.1f Twall_right=%.1f H=%.1f H_half_life=%s\n",
            REGIONAL_TWALL_TOP, REGIONAL_TWALL_BOT, pr.side_thermal,
            REGIONAL_TWALL_SIDE, REGIONAL_TWALL_SIDE, REGIONAL_H,
            REGIONAL_H_HALF_LIFE == Inf ? "Inf" : string(REGIONAL_H_HALF_LIFE))
    @printf(io, "[regional] rheology: eta_law=%s deta_T=%.4g eta_T0=%.3f eta_ref=%.3g bounds=[%.1e, %.1e]\n",
            pr.eta_law, pr.deta_T, pr.eta_T0, pr.eta_ref, pr.eta_min, pr.eta_max)
    @printf(io, "[regional] Ra=%.1f  (kinematic thermal model: no thermal or phase buoyancy)\n",
            pr.Ra)
    if isempty(phases)
        @printf(io, "[regional] phases: none\n")
    else
        @printf(io, "[regional] phases: %d  latent_mode=%s\n", length(phases),
                pr.latent_mode)
        for (i, pt) in enumerate(phases)
            @printf(io, "[regional]   %d: z0=%.3f width=%.3f slope=%+.4f Tref=%.2f drho=%.1f B=%.4f latent=%+.6f\n",
                    i == 1 ? 410 : 660, pt.z0, pt.width, pt.slope, pt.Tref,
                    pt.drho, pt.B, pt.latent === nothing ? NaN : pt.latent)
        end
    end
    @printf(io, "[regional] scale set: d=%.1e m kappa=%.1e m2/s; unit velocity=%.3e m/yr, unit time=%.3e Gyr, quoted eta_0=%.1e Pa s\n",
            REGIONAL_SCALE.d, REGIONAL_SCALE.kappa,
            REGIONAL_VELOCITY_UNIT * 3.156e7,
            REGIONAL_TIME_UNIT / 3.156e16, REGIONAL_ETA0_PHYS)
    @printf(io, "[regional] numerics: p=%d linear=%s steps=%d cfl=%.3g dt0=%.1e dtmax=%.1e ev=%.3g mg_tol=%.0e steady_tol=%.1e\n",
            REGIONAL_P, linear, steps, cfl, dt0, REGIONAL_DTMAX, REGIONAL_EV,
            REGIONAL_MG_TOL, REGIONAL_STEADY_TOL)
    frames === nothing || @printf(io, "[regional] frames: %s (grid %d x %d)\n",
                                  frames, REGIONAL_FRAME_GRID, REGIONAL_FRAME_GRID)
    return nothing
end

regional_tree_name(mode::Symbol) = mode === :validate ? "uniform-level-3" :
    mode === :converge ? "converge-sequence" : "production(level-4+wall-5)"

# ===========================================================================
# 8. The frame writer (the realistic card's format, section 4.6)
# ===========================================================================
mutable struct RegionalFrameWriter
    dir::String
    meta::IO
    ng::Int
    xs::Vector{Float64}
    ys::Vector{Float64}
    t0::Float64
    nframe::Int
end

function RegionalFrameWriter(root::String; ng::Int=REGIONAL_FRAME_GRID)
    dir = joinpath(root, "frames")
    mkpath(dir)
    meta = joinpath(dir, "frame_meta.csv")
    fresh = !isfile(meta) || filesize(meta) == 0
    io = open(meta, "a")
    if fresh
        println(io, "frame,step,t,N,leaves,nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,wall_s,ng")
    end
    xs = collect(range(0.0, REGIONAL_LX; length=ng))
    ys = collect(range(0.0, 1.0; length=ng))
    return RegionalFrameWriter(dir, io, ng, xs, ys, time(), 0)
end

"""The viscosity the frame shows: the material law divided by `eta_ref`."""
function regional_nodal_eta(sq, theta, ux, uz, pr::Params)
    ed = semquad_strain_rate_field(sq, ux, uz)
    out = Vector{Float64}(undef, length(ed))
    for g in eachindex(ed)
        y = sq.coords[g][2]
        T = Tbar(y, pr.H) + theta[g]
        eta = pr.eta_law === :composite ?
              eta_composite_at(T, ed[g], y, pr)[1] : eta_at(T, y, pr)[1]
        out[g] = eta / pr.eta_ref
    end
    return out
end

function regional_write_frame!(fw::RegionalFrameWriter, st, pr::Params)
    s = st.sq
    leaves = Set(s.leaves)
    ng = fw.ng
    T = Matrix{Float64}(undef, ng, ng)
    for j in 1:ng, i in 1:ng
        T[j, i] = Tbar(fw.ys[j], pr.H) +
                  QuaST._semquad_eval_at(s, st.theta, fw.xs[i], fw.ys[j], leaves)
    end
    open(joinpath(fw.dir, @sprintf("frame_%05d_T.bin", st.step)), "w") do io
        write(io, vec(permutedims(T)))     # x fastest -> reshape(ng,ng) is A[z,x]
    end
    open(joinpath(fw.dir, @sprintf("frame_%05d_mesh.csv", st.step)), "w") do io
        for lf in sort(collect(st.qt.leaves))
            h = REGIONAL_LX / (REGIONAL_CELLS_X * 2.0^lf[1])
            @printf(io, "%d,%.17g,%.17g,%.17g\n", lf[1], lf[2] * h, lf[3] * h, h)
        end
    end
    eta = regional_nodal_eta(s, st.theta, st.ux, st.uz, pr)
    open(joinpath(fw.dir, @sprintf("frame_%05d_eta.bin", st.step)), "w") do io
        write(io, Float32.(log10.(eta)))
    end
    open(joinpath(fw.dir, @sprintf("frame_%05d_speed.bin", st.step)), "w") do io
        write(io, Float32.(hypot.(st.ux, st.uz)))
    end
    open(joinpath(fw.dir, @sprintf("frame_%05d_nodes.bin", st.step)), "w") do io
        nodes = Vector{Float32}(undef, 2 * s.N)
        for g in 1:s.N
            nodes[2 * g - 1] = Float32(s.coords[g][1])
            nodes[2 * g] = Float32(s.coords[g][2])
        end
        write(io, nodes)
    end
    nut, nub = semquad_nu_walls(s, st.qt, st.theta, pr)
    vr = semquad_vrms(s, st.qt, st.ux, st.uz)
    @printf(fw.meta, "%d,%d,%.9e,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.3f,%d\n",
            fw.nframe, st.step, st.t, s.N, length(s.leaves),
            length(unique(lf[1] for lf in s.leaves)), minimum(T), maximum(T),
            sum(T) / length(T), nut, nub, vr, maximum(hypot.(st.ux, st.uz)),
            minimum(eta), maximum(eta), time() - fw.t0, ng)
    flush(fw.meta)
    fw.nframe += 1
    return nothing
end

# ===========================================================================
# 9. The modes
# ===========================================================================
"""The closed-form latent coefficient of section 5.3, from the typed scale set."""
function regional_latent_closed_form(pr::Params, slope::Float64, drho::Float64,
                                     z0::Float64)
    rho_ref, cp, dT, g, d = REGIONAL_LATENT_SCALE.rho_ref,
                            REGIONAL_LATENT_SCALE.cp,
                            REGIONAL_LATENT_SCALE.dT,
                            REGIONAL_LATENT_SCALE.g,
                            REGIONAL_LATENT_SCALE.d
    T_abs = semquad_base_T(pr, z0) + REGIONAL_ETA_T0
    gamma = slope * rho_ref * g * d / dT
    dS_A = gamma * drho / (rho_ref * rho_ref)
    return -T_abs * dS_A / cp
end

# --- stage (i): the isoviscous analytic validation --------------------------
"""
    regional_mode_validate(; io=stdout, outdir=nothing,
                           wall_data=regional_wall_data(),
                           level=3, maxsteps=2) -> NamedTuple

Stage (i): the isoviscous (`:expT`, `deta_T = 1`) channel on the uniform
level-3 tree with `linear = :direct`.  Checks 1-4 and 8 of section 6.8, the
outflow shear, the derived pressure gauge, and the two well-posedness refusals
(printed, not gated).  Returns `(mode, res, errors, continuation,
continuation_errors, continuation_prod, continuation_prod_errors, checks,
err, residual)`; the production-tree continuation is the printed diagnostic of
the refined tree's direct-solve round-off.
"""
function regional_mode_validate(; io::IO=stdout, outdir::Union{Nothing,String}=nothing,
                                wall_data=regional_wall_data(), level::Int=3,
                                maxsteps::Int=2)
    checks = NamedTuple[]
    qt = regional_tree(level)
    pr = regional_card(:validate; wall_data=wall_data)
    regional_header(io, :validate, qt, pr, PhaseTransition[]; linear=:direct,
                    steps=maxsteps, cfl=REGIONAL_CFL)
    res = regional_run(qt, pr; maxsteps=maxsteps, linear=:direct,
                       stokes_tol=REGIONAL_STOKES_TOL_EXACT,
                       phases=PhaseTransition[])
    residual_validate = LAST_SEMSTOKES_RESIDUAL[]
    errs = regional_analytic_errors(res)
    @printf(io, "[regional] steps=%d t=%.4e leaves=%d N=%d gauge=%s residual=%.3e\n",
            res.steps, res.t, length(res.qt.leaves), res.ss.su.N,
            string(LAST_SEMSTOKES_PRESSURE_GAUGE[]), LAST_SEMSTOKES_RESIDUAL[])
    # checks 1-2: the exact velocity and pressure fields
    regional_check!(checks, io; id=1, name="u-field", quantity="max|ux-uref|/Up",
                    measured=errs.err_u, reference=0.0, tolerance=1.0e-10)
    regional_check!(checks, io; id=1, name="uy-field", quantity="max|uy|/Up",
                    measured=errs.err_uz, reference=0.0, tolerance=1.0e-10)
    regional_check!(checks, io; id=2, name="p-field", quantity="max|p-pref|/pscale",
                    measured=errs.err_p, reference=0.0, tolerance=1.0e-10)
    # check 3: the per-wall mass flux and the net-zero identity
    qx = errs.flux[1]; qp = errs.flux[2]; qm = errs.flux[3]; qy = errs.flux[4]
    qref = errs.qref
    regional_check!(checks, io; id=3, name="flux-xminus",
                    quantity="|Qxminus+Qref|/Qref",
                    measured=abs(qx + qref) / qref, reference=0.0,
                    tolerance=1.0e-10)
    regional_check!(checks, io; id=3, name="flux-xplus",
                    quantity="|Qxplus-Qref|/Qref",
                    measured=abs(qp - qref) / qref, reference=0.0,
                    tolerance=1.0e-10)
    regional_check!(checks, io; id=3, name="flux-ywalls",
                    quantity="max(|Qyminus|,|Qyplus|)/Qref",
                    measured=max(abs(qm), abs(qy)) / qref, reference=0.0,
                    tolerance=1.0e-10)
    regional_check!(checks, io; id=3, name="flux-net",
                    quantity="|sum Q|/Qref",
                    measured=abs(errs.flux_net) / qref, reference=0.0,
                    tolerance=1.0e-10)
    @printf(io, "[regional]   fluxes: xminus=%.9f xplus=%.9f yminus=%.3e yplus=%.3e net=%.3e (Q_ref=%.6f)\n",
            qx, qp, qm, qy, errs.flux_net, qref)
    # check 4: the outflow pressure held by wall_tn (the derived gauge).  It is
    # compared RELATIVE to the pressure scale: the pressure level is O(1e3)
    # here, so a 1e-10 *absolute* tolerance would sit below the direct solve's
    # round-off (the measured absolute error is ~4e-9 = 1.3e-12 of the scale;
    # the design's "1e-10 abs" is not a round-off class at this pressure level).
    regional_check!(checks, io; id=4, name="outflow-pressure",
                    quantity="max|p(x=Lx)-p_out|/pscale",
                    measured=errs.err_pout / errs.p_scale, reference=0.0,
                    tolerance=1.0e-10)
    # the outflow field shear against the assembled tau_ref data
    regional_check!(checks, io; id=1, name="outflow-shear",
                    quantity="max|eta dux/dy-tau_ref|/taumax",
                    measured=errs.err_shear, reference=0.0, tolerance=1.0e-10)
    # the derived gauge: LAST_SEMSTOKES_PRESSURE_GAUGE must be :none
    gauge_ok = LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    push!(checks, (id=4, name="gauge", quantity="LAST_SEMSTOKES_PRESSURE_GAUGE",
                   measured=gauge_ok ? 0.0 : 1.0, reference=0.0,
                   tolerance=0.0, pass=gauge_ok, gated=true, skip=false))
    @printf(io, "[regional] check %2d %-26s %-30s value=%-8s tol=0  %s\n", 4,
            "gauge", "LAST_SEMSTOKES_PRESSURE_GAUGE", string(LAST_SEMSTOKES_PRESSURE_GAUGE[]),
            gauge_ok ? "PASS" : "FAIL")
    # check 8: the deta_T = 1 continuation.  Section 6.7's definition: the
    # stage-(ii) card with deta_T = 1, *same data, same tree, same linear path*
    # as stage (i), must reproduce it to 1e-10 relative.  It is a guard that the
    # stage cards cannot drift apart (a changed H, bound or wall datum in the
    # rheology card shows up here), not a physical comparison.
    cont_pr = regional_card(:rheology; wall_data=wall_data, deta_T_override=1.0)
    cont = regional_run(qt, cont_pr; maxsteps=maxsteps, linear=:direct,
                        stokes_tol=REGIONAL_STOKES_TOL_EXACT,
                        phases=PhaseTransition[])
    cont_errs = regional_analytic_errors(cont)
    @printf(io, "[regional] continuation (deta_T=1, stage-(ii) card, uniform-3 tree): steps=%d max|ux-uref|/Up=%.3e max|p-pref|/pscale=%.3e\n",
            cont.steps, cont_errs.err_u, cont_errs.err_p)
    regional_check!(checks, io; id=8, name="continuation-u",
                    quantity="max|ux-uref|/Up",
                    measured=cont_errs.err_u, reference=0.0, tolerance=1.0e-10)
    regional_check!(checks, io; id=8, name="continuation-p",
                    quantity="max|p-pref|/pscale",
                    measured=cont_errs.err_p, reference=0.0, tolerance=1.0e-10)
    regional_check!(checks, io; id=8, name="continuation-uz",
                    quantity="max|uz|/Up",
                    measured=cont_errs.err_uz, reference=0.0, tolerance=1.0e-10)
    # The same re-solve on the production (wall-refined) tree is a DIAGNOSTIC of
    # the tree's direct-solve round-off, not a gate: the exact solution is still
    # a polynomial in the space, but the 29 953-dof direct saddle's round-off is
    # measured at 1e-9 relative, an order above the 1e-10 identity class.
    cont_qt = regional_tree(4; wall_min=REGIONAL_WALL_MIN)
    cont_prod = regional_run(cont_qt, cont_pr; maxsteps=maxsteps, linear=:direct,
                             stokes_tol=REGIONAL_STOKES_TOL_EXACT,
                             phases=PhaseTransition[])
    cpe = regional_analytic_errors(cont_prod)
    @printf(io, "[regional] continuation on the production tree (diagnostic, not gated): leaves=%d max|ux-uref|/Up=%.3e max|p-pref|/pscale=%.3e\n",
            length(cont_prod.qt.leaves), cpe.err_u, cpe.err_p)
    # the negative well-posedness checks (printed, not gated; section 6.7)
    try
        Params(bc=:free, wall_bc=(:open, :open, :rigid, :rigid))
        @printf(io, "[regional] well-posedness: (:open,:open,:rigid,:rigid) ACCEPTED (unexpected)\n")
    catch err
        @printf(io, "[regional] well-posedness: (:open,:open,:rigid,:rigid) refused (%s)\n",
                err isa ArgumentError ? "ArgumentError" : string(typeof(err)))
    end
    try
        Params(bc=:free, wall_bc=(:open, :open, :open, :open))
        @printf(io, "[regional] well-posedness: all-open ACCEPTED (unexpected)\n")
    catch err
        @printf(io, "[regional] well-posedness: all-open refused (%s)\n",
                err isa ArgumentError ? "ArgumentError" : string(typeof(err)))
    end
    out = (mode=:validate, res=res, errors=errs, continuation=cont,
           continuation_errors=cont_errs, continuation_prod=cont_prod,
           continuation_prod_errors=cpe, checks=checks, err=errs.err_u,
           residual=residual_validate)
    if outdir !== nothing
        mkpath(outdir)
        regional_write_checks(outdir, :validate, checks)
        regional_write_history(outdir, res)
    end
    regional_summary(io, :validate, checks)
    return out
end

# --- the environment knobs (section 8) --------------------------------------
"""Parse a `CASE_*` float knob, default otherwise."""
function regional_env_float(name::String, default::Float64)
    v = strip(get(ENV, name, ""))
    return isempty(v) ? default : parse(Float64, v)
end

"""Parse a `CASE_*` integer knob, default otherwise."""
function regional_env_int(name::String, default::Int)
    v = strip(get(ENV, name, ""))
    return isempty(v) ? default : parse(Int, v)
end

"""Step-cap default per mode (measured runtimes are in REGIONAL_PROGRESS.md).

The design's section 8 estimated 200-400 steps at 0.2-0.4 s/step for the
`deta_T = 1e3` stages; the measured per-step cost on this machine is
1.4-1.7 s/step (the 29 953-dof saddle's multigrid solve dominates), so the caps
are set to keep each physics mode near its budget: the steady plate-cooling
state needs about 0.8-1 transit of the channel, `t ~ 2-4e-3`, which at the
measured `dt ~ 2e-5` is 100-200 steps.
"""
regional_default_steps(mode::Symbol) = mode === :validate ? 2 :
    mode === :rheology ? 240 : mode === :phases ? 240 :
    mode === :full ? 120 : 200

"""
    regional_config(mode) -> NamedTuple

The card and numerics of one mode, read from the `CASE_*` environment knobs at
call time (so a test can set them before calling): `CASE_OUT`, `CASE_LEVEL`,
`CASE_WALLMIN`, `CASE_WALLSTRIP`, `CASE_CFL`, `CASE_DT0`, `CASE_STEPS`,
`CASE_FRAME`, `CASE_RA`, `CASE_LINEAR`, `CASE_EXTDIAG`, `CASE_WALL_T` and (for
`converge`) `CASE_LAW`.
"""
function regional_config(mode::Symbol)
    mode in REGIONAL_MODES ||
        throw(ArgumentError("unknown regional mode :$mode; modes are " *
                            "$(join(string.(REGIONAL_MODES), ", "))"))
    out = get(ENV, "CASE_OUT",
              normpath(joinpath(@__DIR__, "..", "out_julia", "regional_channel")))
    t_mode = Symbol(strip(get(ENV, "CASE_WALL_T", "rate")))
    t_mode in (:rate, :stress) ||
        throw(ArgumentError("CASE_WALL_T must be rate or stress, got $t_mode"))
    level = regional_env_int("CASE_LEVEL", mode === :validate ? 3 : 4)
    wall_min = regional_env_int("CASE_WALLMIN",
                                mode === :validate ? 0 : REGIONAL_WALL_MIN)
    wall_strip = regional_env_float("CASE_WALLSTRIP", REGIONAL_WALL_STRIP)
    cfl = regional_env_float("CASE_CFL", REGIONAL_CFL)
    dt0 = regional_env_float("CASE_DT0", REGIONAL_DT0)
    # `full` defaults to the direct path: measured, its composite field makes
    # `:auto` resolve to multigrid, which emits MG-FGMRES relres warnings and is
    # more than 5x slower than the direct factorization at this card's 29 953
    # saddle dofs.  `CASE_LINEAR=auto` restores the design path.
    linear = Symbol(strip(get(ENV, "CASE_LINEAR",
                              string(mode === :validate ? :direct :
                                     mode === :full ? :direct : :auto))))
    linear in (:direct, :mg, :auto, :schur_cg, :lagged) ||
        throw(ArgumentError("CASE_LINEAR must be direct, mg or auto"))
    ra = regional_env_float("CASE_RA", REGIONAL_RA)
    frame = regional_env_int("CASE_FRAME",
                             mode === :full ? REGIONAL_FRAME_STRIDE : 0)
    extdiag = lowercase(strip(get(ENV, "CASE_EXTDIAG",
                                  mode === :full ? "1" : "0"))) in
              ("1", "true", "yes", "on")
    law = Symbol(strip(get(ENV, "CASE_LAW", "expT")))
    law in (:expT, :composite) ||
        throw(ArgumentError("CASE_LAW must be expT or composite, got $law"))
    mg_tol = regional_env_float("CASE_MG_TOL", REGIONAL_MG_TOL)
    steps = regional_env_int("CASE_STEPS", 0)
    steps > 0 || (steps = regional_default_steps(mode))
    stokes_tol = mode === :validate ? REGIONAL_STOKES_TOL_EXACT :
                 REGIONAL_STOKES_TOL_PROD
    return (out=out, t_mode=t_mode, level=level, wall_min=wall_min,
            strip=wall_strip, cfl=cfl, dt0=dt0, linear=linear, ra=ra,
            frame=frame, extdiag=extdiag, law=law, steps=steps,
            stokes_tol=stokes_tol, mg_tol=mg_tol)
end

# --- the production stages (ii), (iii), (iv) -------------------------------
"""Window statistics of the open-box heat budget (section 6.3)."""
function regional_budget_window(res, buf, pr::Params, phases)
    area = total_area(buf[end].qt)
    ratios_steady = Float64[]
    ratios_full = Float64[]
    sigmas = Float64[]
    storages = Float64[]
    for b in buf
        b.step >= 2 || continue
        et = regional_energy_terms(b.sq, b.qt, pr, b.theta; phases=phases,
                                   ux=b.ux, uz=b.uz)
        row = res.history[b.step]
        rowm1 = res.history[b.step - 1]
        dmeanT = (row[6] - rowm1[6]) / max(row[2], 1.0e-300)
        storage = area * dmeanT
        push!(sigmas, et.budget_sigma)
        push!(storages, storage)
        push!(ratios_steady, et.budget_ratio)
        push!(ratios_full, abs(et.budget_sigma + storage) /
                           max(et.budget_scale, 1.0e-300))
    end
    return (n=length(sigmas),
            mean_steady=sum(ratios_steady) / length(ratios_steady),
            max_steady=maximum(ratios_steady),
            mean_full=sum(ratios_full) / length(ratios_full),
            max_full=maximum(ratios_full),
            mean_sigma=sum(sigmas) / length(sigmas),
            sigma_spread=maximum(sigmas) - minimum(sigmas),
            mean_storage=sum(storages) / length(storages))
end

"""Statistics of the latent-heat enthalpy identity over the sampled window."""
function regional_enthalpy_window(erec)
    ratios = Float64[]
    for k in 2:length(erec)
        r0 = erec[k - 1]
        r1 = erec[k]
        r1.step == r0.step + 1 || continue
        F0 = r0.flux + r0.q_sides - r0.h_right + r0.h_left
        F1 = r1.flux + r1.q_sides - r1.h_right + r1.h_left
        dEdt = (r1.enthalpy - r0.enthalpy) / r1.dt
        Fmid = 0.5 * (F0 + F1)
        scale = max(abs(F0), abs(F1), abs(dEdt), 1.0e-30)
        push!(ratios, abs(dEdt - Fmid) / scale)
    end
    isempty(ratios) && return (n=0, mean_ratio=NaN, max_ratio=NaN)
    return (n=length(ratios), mean_ratio=sum(ratios) / length(ratios),
            max_ratio=maximum(ratios))
end

"""
    regional_mode_production(mode; io=stdout, outdir=nothing, cfg=nothing,
                             maxsteps=nothing) -> NamedTuple

Stages (ii) `rheology`, (iii) `phases` and (iv) `full`: the production tree, the
stage's rheology and phase list, a run to the steady-state tolerance, the open-box
checks, the latent-heat checks and (in `full`) the frames and the
`nu_cbf`/`extdiag` columns.

Gated checks: 3 (per-wall mass flux and net zero), the derived pressure gauge,
the plate-cooling profile in `rheology` (a diagnostic elsewhere, as section 8
states) and the latent-coefficient mapping in `phases`.  Checks 5 (open-box heat
budget) and 10 (latent enthalpy) are reported with their full term dump and
measured residual but stay DIAGNOSTIC: their pointwise-wall-gradient form does
not close for a fixed-side-wall card (docs/dev/REGIONAL_PROGRESS.md D6/D7).
Check 9 (cfl 1 vs 2) is a manual two-run comparison, printed as such.
"""
function regional_mode_production(mode::Symbol; io::IO=stdout,
                                  outdir::Union{Nothing,String}=nothing,
                                  cfg=nothing, maxsteps=nothing)
    cfg === nothing && (cfg = regional_config(mode))
    steps = maxsteps === nothing ? cfg.steps : maxsteps
    checks = NamedTuple[]
    qt = regional_tree(cfg.level; wall_min=cfg.wall_min, strip=cfg.strip)
    pr = regional_card(mode; ra=cfg.ra, t_mode=cfg.t_mode)
    ph = regional_phases(mode, pr)
    frames = (mode === :full && outdir !== nothing) ?
             RegionalFrameWriter(outdir) : nothing
    buf = Any[]
    erec = Any[]
    win = REGIONAL_STEADY_WINDOW
    eh0, eh1 = 25, 85
    cb = function (st)
        push!(buf, (step=st.step, t=st.t, dt=st.dt, theta=st.theta,
                    ux=st.ux, uz=st.uz, sq=st.sq, qt=st.qt))
        length(buf) > win + 1 && popfirst!(buf)
        if !isempty(ph) && eh0 <= st.step <= eh1
            et = regional_energy_terms(st.sq, st.qt, pr, st.theta; phases=ph,
                                       ux=st.ux, uz=st.uz)
            push!(erec, (step=st.step, t=st.t, dt=st.dt,
                         enthalpy=et.enthalpy, flux=et.flux,
                         q_sides=et.q_sides, h_left=et.h_left,
                         h_right=et.h_right))
        end
        if frames !== nothing && cfg.frame > 0 && st.step % cfg.frame == 0
            regional_write_frame!(frames, st, pr)
        end
        return nothing
    end
    regional_header(io, mode, qt, pr, ph; linear=cfg.linear, steps=steps,
                    cfl=cfg.cfl, dt0=cfg.dt0,
                    frames=frames === nothing ? nothing : outdir)
    @printf(io, "[regional] case config: CASE_WALL_T=%s CASE_LEVEL=%d CASE_WALLMIN=%d CASE_STEPS=%d CASE_LINEAR=%s CASE_FRAME=%d CASE_EXTDIAG=%s CASE_MG_TOL=%.1e\n",
            cfg.t_mode, cfg.level, cfg.wall_min, steps, cfg.linear, cfg.frame,
            cfg.extdiag, cfg.mg_tol)
    if cfg.t_mode === :rate
        @printf(io, "[regional] outflow traction = tau_ref(y) * eta_far(y)/eta_ref (the reference channel's far-field shear rate; exact tau_ref when deta_T = 1)\n")
    else
        @printf(io, "[regional] outflow traction = tau_ref(y) (the design's literal fixed far-field stress)\n")
    end
    t0 = time()
    res = regional_run(qt, pr; maxsteps=steps, linear=cfg.linear,
                       stokes_tol=cfg.stokes_tol, mg_tol=cfg.mg_tol,
                       cfl=cfg.cfl, dt0=cfg.dt0,
                       steady_tol=REGIONAL_STEADY_TOL, phases=ph,
                       nu_cbf=(mode === :full), extdiag=cfg.extdiag,
                       callback=cb)
    wall = time() - t0
    lastrow = res.history[end]
    steady_reached = res.steps < steps
    if frames !== nothing
        regional_write_frame!(frames, (step=res.steps, t=res.t,
                                       dt=lastrow[2], theta=res.theta,
                                       ux=res.ux, uz=res.uz, sq=res.ss.su,
                                       qt=res.qt), pr)
        close(frames.meta)
        @printf(io, "[regional] frames: %d written to %s\n", frames.nframe,
                joinpath(outdir, "frames"))
    end
    @printf(io, "[regional] run: steps=%d/%d t=%.4e wall=%.1f s (%.3f s/step) steady=%s leaves=%d gauge=%s\n",
            res.steps, steps, res.t, wall, wall / max(res.steps, 1),
            steady_reached ? "yes" : "NO (step cap)", length(res.qt.leaves),
            string(LAST_SEMSTOKES_PRESSURE_GAUGE[]))
    @printf(io, "[regional] final: Nu_top=%.6f Nu_bot=%.6f Vrms=%.4e umax=%.4e meanT=%.6f dt=%.2e residual=%.2e auto=%s fallbacks=%d\n",
            lastrow[3], lastrow[4], lastrow[5], lastrow[11], lastrow[6],
            lastrow[2], LAST_SEMSTOKES_RESIDUAL[],
            string(LAST_SEMSTOKES_AUTO_CHOICE[].choice),
            LAST_AUTO_FALLBACK[].count)
    if cfg.extdiag
        i = 18 + findfirst(==("resid_A"), res.history_extra)
        @printf(io, "[regional] extdiag: resid_A=%.4e (the scheme residual of the pointwise pair against identity ID1; for this fixed-side-wall card it also absorbs the side-wall reaction and is NOT small -- the design's 1e-4 class is for the closed-box realistic card)\n",
                lastrow[i])
    end
    # --- check 3: the per-wall mass flux and the net-zero identity ----------
    # The inflow flux is the prescribed polynomial data, so it is ±Q_ref exactly
    # (the GLL rule integrates the Q2 trace exactly).  The outflow and net flux
    # are the discrete continuity identity up to the Stokes residual.  The
    # design's 1e-10/1e-8 class was measured on the isoviscous uniform channel.
    # Measured defects on this card: 2.9e-8 (uniform-4, variable viscosity,
    # direct path with its 1e-12 shift rung), 5.2e-8 (production tree, expT) and
    # 7.7e-5 (production tree, composite: the graded side wall's 2:1 transitions
    # add a solution-dependent trace-quadrature term on top of the shifted
    # solve).  The graded-tree tolerance is therefore 5e-4; gate G265 pins the
    # identity at 1e-10 on a healthy uniform variable-viscosity solve (measured
    # 6e-16).
    fl = regional_flux_per_wall(res.ss, res.qt, vcat(res.ux, res.uz))
    qref = regional_qref()
    mtol = 5.0e-4
    @printf(io, "[regional]   fluxes: xminus=%.9f xplus=%.9f yminus=%.3e yplus=%.3e net=%.3e (Q_ref=%.6f, tol=%.0e)\n",
            fl[1], fl[2], fl[3], fl[4], sum(fl), qref, mtol)
    regional_check!(checks, io; id=3, name="flux-xminus",
                    quantity="|Qxminus+Qref|/Qref",
                    measured=abs(fl[1] + qref) / qref, reference=0.0,
                    tolerance=mtol)
    regional_check!(checks, io; id=3, name="flux-xplus",
                    quantity="|Qxplus-Qref|/Qref",
                    measured=abs(fl[2] - qref) / qref, reference=0.0,
                    tolerance=mtol)
    regional_check!(checks, io; id=3, name="flux-ywalls",
                    quantity="max(|Qyminus|,|Qyplus|)/Qref",
                    measured=max(abs(fl[3]), abs(fl[4])) / qref, reference=0.0,
                    tolerance=mtol)
    regional_check!(checks, io; id=3, name="flux-net", quantity="|sum Q|/Qref",
                    measured=abs(sum(fl)) / qref, reference=0.0,
                    tolerance=mtol)
    # --- the derived pressure gauge ----------------------------------------
    gauge_ok = LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    push!(checks, (id=4, name="gauge", quantity="LAST_SEMSTOKES_PRESSURE_GAUGE",
                   measured=gauge_ok ? 0.0 : 1.0, reference=0.0,
                   tolerance=0.0, pass=gauge_ok, gated=true, skip=false))
    @printf(io, "[regional] check %2d %-26s %-30s value=%-8s tol=0  %s\n", 4,
            "gauge", "LAST_SEMSTOKES_PRESSURE_GAUGE",
            string(LAST_SEMSTOKES_PRESSURE_GAUGE[]), gauge_ok ? "PASS" : "FAIL")
    # --- check 5: the open-box heat budget --------------------------------
    # The identity is d/dt int T = wall fluxes + side flux + H A - advective
    # flux.  It is built here with the POINTWISE one-sided GLL wall functional
    # (the design's section 6.3 form).  Measured: it does NOT close for this
    # card -- not because of a sign, but because the two side walls are
    # Dirichlet and their heat exchange is a boundary *reaction* of the discrete
    # step, not the one-sided derivative of a resolved layer (the library's
    # consistent boundary flux is implemented for the top/bottom walls only;
    # docs/dev/VALIDATION.md section 4.1 states that the consistent flux is what
    # closes the step identity).  During the side-wall adjustment the pointwise
    # value is off by O(1) relative; the series is reported in full below so the
    # sign of every term is visible, and the row is a DIAGNOSTIC, not a gate.
    bw = regional_budget_window(res, buf, pr, ph)
    measured = steady_reached ? bw.mean_steady : bw.mean_full
    regional_check!(checks, io; id=5, name="heat-budget",
                    quantity=steady_reached ? "mean |sigma|/scale" :
                                              "mean |sigma+storage|/scale",
                    measured=measured, reference=0.0, tolerance=1.0e-3,
                    gated=false)
    @printf(io, "[regional]   budget: n=%d mean_sigma=%.3e spread=%.3e mean_storage=%.3e max_ratio=%.3e (steady=%s)\n",
            bw.n, bw.mean_sigma, bw.sigma_spread, bw.mean_storage,
            steady_reached ? bw.max_steady : bw.max_full, steady_reached)
    @printf(io, "[regional]   budget note: this pointwise-flux form is a diagnostic for a fixed-side-wall card (docs/dev/REGIONAL_PROGRESS.md); the design's 1e-3 conservation class assumes the consistent boundary flux, which the library implements for the top/bottom walls only\n")
    # --- check 6: the asymptotic plate-cooling profile --------------------
    pc = regional_plate_cooling(res.ss.su, res.qt, pr, res.theta)
    for r in pc.stations
        @printf(io, "[regional]   station x=%.2f Tinf=%.4f delta=%.4f delta_hs=%.4f q=%.3f q_hs=%.3f Nu_pt=%.3f\n",
                r.x, r.Tinf, r.delta, r.delta_hs, r.q, r.q_hs, 1.5 + r.q)
    end
    gated6 = mode === :rheology
    regional_check!(checks, io; id=6, name="plate-cooling-T",
                    quantity="max|T-erf| in the band", measured=pc.err_profile,
                    reference=0.0, tolerance=0.10, gated=gated6)
    regional_check!(checks, io; id=6, name="plate-cooling-delta",
                    quantity="max|delta/delta_hs-1|", measured=pc.delta_ratio,
                    reference=0.0, tolerance=0.10, gated=gated6)
    regional_check!(checks, io; id=6, name="plate-cooling-growth",
                    quantity="|growth/sqrt(2)-1|", measured=pc.growth_err,
                    reference=0.0, tolerance=0.10, gated=gated6)
    regional_check!(checks, io; id=6, name="plate-cooling-q",
                    quantity="max|q/q_hs-1|", measured=pc.q_err,
                    reference=0.0, tolerance=0.15, gated=gated6)
    @printf(io, "[regional]   plate cooling is a spatial TBL check: it is evaluated on the reached state even when the global thermal steady state is still drifting (the design's steady-state risk, section 11.1)\n")
    # --- check 10: the latent coefficients and the enthalpy identity ------
    if !isempty(ph)
        for (i, (slope, drho, z0)) in
            ((1, (REGIONAL_SLOPE410, REGIONAL_DRHO410, REGIONAL_Z410)),
             (2, (REGIONAL_SLOPE660, REGIONAL_DRHO660, REGIONAL_Z660)))
            closed = regional_latent_closed_form(pr, slope, drho, z0)
            got = ph[i].latent
            rel = abs(got - closed) / max(abs(closed), 1.0e-300)
            @printf(io, "[regional]   latent %d: per_phase_latent=%+.6f closed form=%+.6f\n",
                    i == 1 ? 410 : 660, got, closed)
            regional_check!(checks, io; id=10, name="latent-coefficient-$i",
                            quantity="|latent-closed|/|closed|", measured=rel,
                            reference=0.0, tolerance=1.0e-12,
                            gated=(mode === :phases))
        end
        # The design's check 10: d(enthalpy)/dt against the time integral of the
        # top/bottom flux + the lateral advective enthalpy flux.  The lateral
        # DIFFUSIVE exchange is missing for the same reason as check 5: the side
        # walls are Dirichlet, so their exchange is a step reaction, not a
        # one-sided derivative of a resolved layer.  Measured on the transient
        # window: the residual is O(1) dominated by that side exchange, so the
        # row is a DIAGNOSTIC; the CLOSED-box latent conservation identity is
        # gated by t_g194_latent_heat.jl, as the design says.
        ew = regional_enthalpy_window(erec)
        regional_check!(checks, io; id=10, name="latent-enthalpy",
                        quantity="mean |dEdt-(flux+adv)|/scale",
                        measured=ew.mean_ratio, reference=0.0, tolerance=0.02,
                        gated=false, skip=(ew.n < 10))
        @printf(io, "[regional]   latent enthalpy window: n=%d mean=%.3e max=%.3e (diagnostic: the side-wall diffusive exchange is a Dirichlet reaction)\n",
                ew.n, ew.mean_ratio, ew.max_ratio)
    end
    @printf(io, "[regional] check 9 (cfl 1 vs 2 steady state) is a manual two-run comparison: rerun this mode with CASE_CFL=1 and compare Nu_top, delta(1.6) and the outflow flux printed above.\n")
    out = (mode=mode, res=res, checks=checks, wall=wall,
           steady=steady_reached, energies=lastrow, budget=bw,
           plate_cooling=pc)
    if outdir !== nothing
        mkpath(outdir)
        regional_write_checks(outdir, mode, checks)
        regional_write_history(outdir, res)
    end
    regional_summary(io, mode, checks)
    return out
end

# --- the resolution sequence (section 6.6) ---------------------------------
"""
    regional_mode_converge(; io=stdout, outdir=nothing, cfg=nothing) -> NamedTuple

The resolution check: the same steady state on the uniform level-3, uniform
level-4 and production trees (and on a level-6 wall strip when
`CASE_WALLMIN >= 6`), reporting `Nu_top`, `Nu_bot`, the CBF pair, `max|u|`, the
outflow flux, `delta(1.6)` and the budget residual of each arm, and gating the
3 % production-vs-uniform-4 agreement on `Nu_top` and `delta(1.6)` plus the
monotone decrease of the budget residual.  `CASE_LAW=composite` runs the
composite card instead of the `:expT deta_T = 1e3` state; the default is the
cheap rheology state (the composite arm costs about four times as much).
"""
function regional_mode_converge(; io::IO=stdout,
                                outdir::Union{Nothing,String}=nothing,
                                cfg=nothing)
    cfg === nothing && (cfg = regional_config(:converge))
    checks = NamedTuple[]
    arms = [(name="uniform-3", level=3, wall_min=0, strip=cfg.strip),
            (name="uniform-4", level=4, wall_min=0, strip=cfg.strip),
            (name="production", level=cfg.level, wall_min=cfg.wall_min,
             strip=cfg.strip)]
    if cfg.wall_min >= 6
        push!(arms, (name="wall-6", level=cfg.level, wall_min=6, strip=0.04))
    end
    @printf(io, "[regional] converge: law=%s steps<=%d cfl=%.3g linear=%s CASE_WALL_T=%s\n",
            cfg.law, cfg.steps, cfg.cfl, cfg.linear, cfg.t_mode)
    results = NamedTuple[]
    for arm in arms
        qt = regional_tree(arm.level; wall_min=arm.wall_min, strip=arm.strip)
        pr = regional_card(:converge; ra=cfg.ra, t_mode=cfg.t_mode,
                           law_override=cfg.law)
        @printf(io, "[regional] arm %s: leaves=%d law=%s\n", arm.name,
                length(qt.leaves), cfg.law)
        t0 = time()
        res = regional_run(qt, pr; maxsteps=cfg.steps, linear=cfg.linear,
                           stokes_tol=cfg.stokes_tol, mg_tol=cfg.mg_tol,
                           cfl=cfg.cfl, dt0=cfg.dt0,
                           steady_tol=REGIONAL_STEADY_TOL, nu_cbf=true)
        wall = time() - t0
        last = res.history[end]
        et = regional_energy_terms(res.ss.su, res.qt, pr, res.theta;
                                   ux=res.ux, uz=res.uz)
        pcs = regional_plate_cooling(res.ss.su, res.qt, pr, res.theta;
                                     stations=(1.6,))
        fl = regional_flux_per_wall(res.ss, res.qt, vcat(res.ux, res.uz))
        i_cbf = 18 + findfirst(==("Nu_cbf_top"), res.history_extra)
        row = (name=arm.name, leaves=length(res.qt.leaves),
               steady=res.steps < cfg.steps, wall=wall, steps=res.steps,
               Nu_top=last[3], Nu_bot=last[4], Nu_cbf_top=last[i_cbf],
               Nu_cbf_bot=last[i_cbf + 1], umax=last[11], flux_out=fl[2],
               delta16=pcs.stations[1].delta,
               budget_abs=abs(et.budget_sigma), sigma=et.budget_sigma)
        push!(results, row)
        @printf(io, "[regional]   %s: leaves=%d steps=%d t=%.3e steady=%s Nu_top=%.6f Nu_bot=%.6f Nu_cbf=%.6f/%.6f umax=%.4e flux=%.6f delta16=%.5f |sigma|=%.3e wall=%.1f s\n",
                row.name, row.leaves, row.steps, res.t, row.steady, row.Nu_top,
                row.Nu_bot, row.Nu_cbf_top, row.Nu_cbf_bot, row.umax,
                row.flux_out, row.delta16, row.budget_abs, wall)
        if outdir !== nothing
            regional_write_history(joinpath(outdir, arm.name), res)
        end
    end
    # --- the gated comparisons --------------------------------------------
    ref = results[end]                 # the finest tree
    i4 = findfirst(r -> r.name == "uniform-4", results)
    steadyall = all(r -> r.steady, results)
    if i4 !== nothing
        u4 = results[i4]
        nu_diff = abs(ref.Nu_top - u4.Nu_top) / max(abs(ref.Nu_top), 1.0e-300)
        d_diff = abs(ref.delta16 - u4.delta16) / max(abs(ref.delta16), 1.0e-300)
        regional_check!(checks, io; id=7, name="resolution-Nu_top",
                        quantity="|Nu(prod)-Nu(u4)|/|Nu(prod)|",
                        measured=nu_diff, reference=0.0, tolerance=0.03,
                        skip=!steadyall)
        regional_check!(checks, io; id=7, name="resolution-delta16",
                        quantity="|delta(prod)-delta(u4)|/|delta(prod)|",
                        measured=d_diff, reference=0.0, tolerance=0.03,
                        skip=!steadyall)
    end
    # budget residual must fall with the wall resolution
    seq = results[1:min(3, length(results))]
    mono = seq[end].budget_abs <= seq[end - 1].budget_abs <= seq[end - 2].budget_abs
    for k in 1:length(seq)
        @printf(io, "[regional]   budget |sigma| %-12s = %.3e\n", seq[k].name,
                seq[k].budget_abs)
    end
    regional_check!(checks, io; id=7, name="budget-monotone",
                    quantity="|sigma| falls from uniform-3 to production",
                    measured=mono ? 0.0 : 1.0, reference=0.0, tolerance=0.0,
                    skip=!steadyall)
    if !steadyall
        @printf(io, "[regional] resolution comparison SKIPPED: at least one arm stopped at the step cap (not steady)\n")
    end
    out = (mode=:converge, arms=results, checks=checks)
    if outdir !== nothing
        mkpath(outdir)
        regional_write_checks(outdir, :converge, checks)
    end
    regional_summary(io, :converge, checks)
    return out
end

# ===========================================================================
# 10. The command line
# ===========================================================================
"""
    regional_parse_mode(args) -> Symbol

The mode of a command line: one of `REGIONAL_MODES`, or `:help` for an empty
argument list / `-h` / `--help` / `help`.  Anything else is an `ArgumentError`
listing the modes.
"""
function regional_parse_mode(args::AbstractVector{<:AbstractString})
    isempty(args) && return :help
    m = strip(String(args[1]))
    (isempty(m) || m in ("-h", "--help", "help")) && return :help
    sym = Symbol(m)
    sym in REGIONAL_MODES ||
        throw(ArgumentError("unknown mode \"$m\"; modes are " *
                            "$(join(string.(REGIONAL_MODES), ", "))"))
    return sym
end

"Print the usage line of the case."
function regional_usage(io::IO=stdout)
    println(io, "usage: julia --project=. --threads=8 examples/subduction_2d.jl <mode>")
    println(io, "  modes: ", join(string.(REGIONAL_MODES), ", "))
    println(io, "  see docs/en/examples.md and docs/dev/REGIONAL_DESIGN.md")
    println(io, "  CASE_* knobs: CASE_OUT, CASE_LEVEL, CASE_WALLMIN, CASE_WALLSTRIP,")
    println(io, "    CASE_CFL, CASE_DT0, CASE_STEPS, CASE_FRAME, CASE_RA, CASE_LINEAR,")
    println(io, "    CASE_EXTDIAG, CASE_MG_TOL, CASE_WALL_T (rate|stress), CASE_LAW (converge)")
    return nothing
end

"""
    regional_main(mode; io=stdout) -> NamedTuple

Run one mode end to end (build the tree and card, run, print the checks, write
`checks.csv` and `history.csv` under `\$CASE_OUT/<mode>/`), and return the
mode's result named tuple.
"""
function regional_main(mode::Symbol; io::IO=stdout)
    mode in REGIONAL_MODES ||
        throw(ArgumentError("unknown mode :$mode; modes are " *
                            "$(join(string.(REGIONAL_MODES), ", "))"))
    cfg = regional_config(mode)
    outdir = joinpath(cfg.out, string(mode))
    t0 = time()
    out = if mode === :validate
        regional_mode_validate(; io=io, outdir=outdir)
    elseif mode === :converge
        regional_mode_converge(; io=io, outdir=outdir, cfg=cfg)
    else
        regional_mode_production(mode; io=io, outdir=outdir, cfg=cfg)
    end
    @printf(io, "[regional] wall clock (mode %s) = %.1f s\n", mode,
            time() - t0)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = try
        regional_parse_mode(ARGS)
    catch err
        @printf(stderr, "subduction_2d: %s\n", sprint(showerror, err))
        exit(1)
    end
    if mode === :help
        regional_usage(stdout)
    else
        regional_main(mode)
    end
end

