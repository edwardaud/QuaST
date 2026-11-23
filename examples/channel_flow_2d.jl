# A pressure-driven plane channel built from the three per-wall boundary
# mechanisms of the boundary-condition feature: a prescribed *normal velocity*
# on the inflow, a prescribed *tangential traction* on the open outflow, and the
# `:open` wall itself (whose normal-traction data hold the outflow pressure).
# Gate G264 exercises this file.
#
#   julia --project=. examples/channel_flow_2d.jl
#
# Domain `[0, Lx] x [0, H]`.  Top and bottom are no-slip (`:rigid`), the upstream
# x-wall is `:rigid` with `wall_un` = the normal velocity of plane Poiseuille,
# and the downstream x-wall is `:open` with the exact total traction of that
# solution.  The driver's fluid solve has no horizontal body force, so the flow
# is driven by the pressure gradient alone: with `dp/dx = -G`,
#
#     u = (G y (H - y) / (2 eta), 0),      p = p_out + G (Lx - x),
#
# and on the outflow (`n = +e_x`) the total Cauchy traction is
#
#     sigma.n = (-p, eta du/dy) = (-p_out, G (H/2 - y)),
#
# so the card carries `wall_tn = -p_out` and `wall_t = G (H/2 - y)`.  Q2 holds
# the quadratic velocity and Q1 the linear pressure exactly, and the face
# quadrature is exact for the linear traction, so the printed profile error is
# the linear solve's round-off, not a discretisation error.
#
# The run is one time step; with no buoyancy and `u_y = 0` the temperature
# anomaly stays zero, so the same channel is re-solved every step.  It is not a
# convergence study: the exact solution is compared as a manufactured solution.
#
# The globals are prefixed `channel_`/`CHANNEL_` because this file is also
# included by gate G264 inside the one-process suite, where earlier gates have
# already defined globals of their own.
using QuaST, Printf

const CHANNEL_LX = 2.0
const CHANNEL_H = 1.0
const CHANNEL_G = 2.0          # pressure gradient magnitude, dp/dx = -G
const CHANNEL_ETA = 1.0
const CHANNEL_POUT = 2.0       # dynamic pressure held at the open outflow
const CHANNEL_UMAX = CHANNEL_G * CHANNEL_H^2 / (8.0 * CHANNEL_ETA)

# Exact plane-Poiseuille velocity and its shear traction eta du/dy.
channel_profile(y) = CHANNEL_G * y * (CHANNEL_H - y) / (2.0 * CHANNEL_ETA)
channel_shear(y) = CHANNEL_G * (CHANNEL_H / 2.0 - y)

# `Lx x H` box of square leaves: a 4 x 2 base grid refined twice (128 leaves).
function channel_tree()
    qt = Quadtree(2; Lx=CHANNEL_LX, cells_x=4, cells_y=2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# The card: no buoyancy (Ra = 0, H = 0) and constant viscosity (`deta_T = 1` is
# the constant-viscosity limit of the :expT law).
function channel_card()
    return Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free,
                  wall_bc=(:rigid, :open, :rigid, :rigid),
                  wall_un=((x, y) -> channel_profile(y), nothing, nothing,
                           nothing),
                  wall_t=(nothing, (x, y) -> channel_shear(y), nothing, nothing),
                  wall_tn=(nothing, (x, y) -> -CHANNEL_POUT, nothing, nothing))
end

"""
    channel_demo(; io=stdout) -> NamedTuple

Run the channel card through [`run_semquad_simulation`](@ref) for one step,
print the measured errors against the exact solution, and return the errors and
the run result.  `io` lets a test capture the printed line instead of the
terminal.
"""
function channel_demo(; io::IO=stdout)
    qt = channel_tree()
    ss = make_semstokes(qt, 2)
    res = run_semquad_simulation(qt, channel_card(), zeros(Float64, ss.su.N);
                                 p=2, tstop=1.0e-3, dt0=1.0e-3,
                                 dtmax=1.0e-3, dtmin=1.0e-16, amr_every=0,
                                 ev=0.0, linear=:direct, maxsteps=1,
                                 mass_form=:boussinesq, verbose=false)
    coords = res.ss.su.coords
    pcoords = res.ss.sp.coords
    err_u = maximum(abs(res.ux[i] - channel_profile(coords[i][2]))
                    for i in eachindex(coords)) / CHANNEL_UMAX
    err_w = maximum(abs, res.uz) / CHANNEL_UMAX
    err_p = maximum(abs(res.pv[j] -
                        (CHANNEL_POUT + CHANNEL_G * (CHANNEL_LX - pcoords[j][1])))
                    for j in eachindex(pcoords))
    @printf(io, "[channel] steps=%d leaves=%d err_u=%.3e err_uz=%.3e err_p=%.3e max|u|=%.6f\n",
            res.steps, length(res.qt.leaves), err_u, err_w, err_p,
            maximum(abs, res.ux))
    return (res=res, err_u=err_u, err_uz=err_w, err_p=err_p,
            umax=maximum(abs, res.ux))
end

if abspath(PROGRAM_FILE) == @__FILE__
    channel_demo()
end
