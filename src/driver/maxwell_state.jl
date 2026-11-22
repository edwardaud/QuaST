# The driver-side Maxwell stress state.
#
# `_nodal_eta` samples the material viscosity closure at the scalar nodes, which
# is what the Maxwell time `eta/G` must be built from; `_advance_maxwell_stress!`
# advances the three nodal tensor components by one backward-Euler step of the
# objective (co-rotational, Jaumann) Maxwell equation and returns the per-step
# rotation-band maxima the report in `maxwell_reports.jl` accumulates.  The
# kernels themselves are in `src/physics/viscoelastic.jl`; this file is the
# driver-side application of them to one step's state.

"""
    _nodal_eta(eta_fun, sq)

Sample the viscosity closure `eta_fun(x, y)` at the `sq` nodes, one value per
scalar dof.  Used by the Maxwell stress-history update to give every node the
*material* viscosity (the law before the 2a effective-viscosity softening), so
its Maxwell time `eta/G` is the material one.
"""
function _nodal_eta(eta_fun::Function, sq::SemQuad)
    eta = Vector{Float64}(undef, sq.N)
    @inbounds for (g, (x, y)) in enumerate(sq.coords)
        eta[g] = eta_fun(x, y)
    end
    return eta
end

"""
    _advance_maxwell_stress!(txx, tyy, txy, sq, ux, uz, dt, eta, G)

One backward-Euler step of the **objective** (co-rotational, Jaumann) Maxwell
stress for the three nodal tensor components of the viscoelastic stress state
(docs/dev/PHASE2.md "2b: the design").  The equation is

    dtau/dt = 2 G ehat + (W tau - tau W) - tau / tau_M,     tau_M = eta / G,

and its backward-Euler step with the step's velocity `(ux, uz)` is

    tau^{n+1} = (tau^n + 2 G dt ehat + dt (W tau^n - tau^n W)) / (1 + dt/tau_M)
              = maxwell_stress_step(maxwell_stress_corotate(tau^n, wxy, dt),
                                    ehat, dt, eta, G),

i.e. the already-gated scalar kernel [`maxwell_stress_step`](@ref) applied to the
co-rotated old stress [`maxwell_stress_corotate`](@ref).  The `W` term is what
makes the stress objective: without it the tensor does not rotate with the
material, which is wrong for every large-strain flow (all mantle convection).
`W` is evaluated at the current just-solved velocity, with the stress lagged at
`tau^n` -- the same lagged numerator as ASPECT's `viscoelastic` update.

The driving strain rate is the **deviatoric** one,

    ehat_ij = eps_ij - (1/3) tr(eps) delta_ij,     tr(eps) = exx + ezz,

i.e. the three components advanced are

    txx <- step(ehat_xx = exx - tr/3),
    tyy <- step(ehat_zz = ezz - tr/3),
    txy <- step(ehat_xz = exz),

with the nodal material viscosity `eta`.  In 2-D plane strain the out-of-plane
normal strain is exactly zero, but it still enters the trace: the tensor being
updated is the full 3-D one restricted to the plane, whose third normal
deviatoric strain rate is `-tr/3`.  The projection belongs to this caller, not
to the kernel: [`maxwell_stress_step`](@ref) is a scalar backward-Euler step
gated on its own (G244), and the Maxwell stress is the *deviatoric* stress, so
`ehat` is its input (G249).

This is not a cosmetic correction.  The discrete trace of the solver's velocity
is not zero even when the flow is incompressible, so feeding the raw tensor
would accumulate a spurious isotropic stress; `div(a tau)` then realises that
trace as a pressure-like load and every stress diagnostic inherits it.  The
projection is also the strain-rate measure the viscous operator uses when
`Params.deviatoric = true` (`K = int 2 eta ehat : eps(v)`), so both halves of
the backward-Euler split then describe the same tensor; G249 pins that, and the
G246 stress-state gate -- written with the pre-fix raw mapping -- had to move
its expected components to `ehat`.

The spin component `wxy = W_xy = (du_x/dy - du_z/dx)/2` is sampled from the same
step velocity by [`semquad_spin_component`](@ref), so the strain rate and the
spin of a step can never come from different velocity fields.  With
`W = [0 wxy; -wxy 0]` the co-rotated components are

    xx:  txx + 2 dt wxy txy,
    yy:  tyy - 2 dt wxy txy,
    xy:  txy + dt wxy (tyy - txx),

and the shear sign is the one that rotates the material the right way (the
derivation is in [`maxwell_stress_corotate`](@ref); G250 pins it against
`W*T - T*W` and against the analytic rigid-rotation solution).  The in-plane
trace and the implied `tau_zz` are invariant under the term, so no deviatoric
projection is applied to it and the stored triple stays closed.

The fixed point of the recursion moves with the term: it is

    tau = 2 eta ehat + tau_M (W tau - tau W),

the Jaumann steady stress, not `2 eta ehat`.  That is real co-rotational
physics -- in a steady rotational flow the principal axes of the stress are
rotated away from those of the strain rate -- and the shift is `O(tau_M |W|)`,
which is why the former `2 eta ehat` fixed-point assertions of G247/G249 had to
be re-expressed when the term landed while their *velocity* assertions (the
material-viscosity fixed point, `eta_eff + a eta = eta`) stayed.  For a
rotation-free flow the discrete spin is round-off only, so the update agrees
with the spin-free one to round-off; if `wxy` is exactly zero it is the same
operation for operation (G250).

The out-of-plane normal stress `tau_zz` is **not stored**.  Trace zero makes it
implied,

    tau_zz = -(tau_xx + tau_yy),

which is what the deviatoric update driven by `-tr/3` produces (G249 checks the
two agree; it is not defined as the negation of the stored sum).  The 2-D
elastic force term never needs it: `div(a tau)` is

    (d/dx(a tau_xx) + d/dy(a tau_xy),  d/dx(a tau_xy) + d/dy(a tau_yy)),

and `src/stokes/assembly_var.jl` assembles exactly those two rows from the
three in-plane components (the out-of-plane normal stress would contract with
the identically zero out-of-plane velocity gradient); the Newton path uses the
same assembler.  A fourth field would therefore be state no assembly reads.

The state is advanced after the momentum solve of the step (so the next step's
Stokes solve sees `tau^n`, never a same-step update): the driver adds
`div(a tau^n)` to that solve's right-hand side (2b step 2), where
`a = eta/(eta + G dt)` is the part of the backward-Euler stress the effective
viscosity does not already carry, see [`maxwell_force_stress`](@ref).  That
lagged load is the one place the update's rotation increment is not carried
(the exact split would load `a (tau^n + dt (W tau^n - tau^n W))`); the scope
choice and its order are documented on [`maxwell_force_stress`](@ref).

The step `dt` is the driver's **actual numerical step for that step** -- the one
the CFL/growth rule selected, which the energy solve and the returned history
use too -- read from the argument on every call; there is no fixed elastic
timescale in this update.  That is consistent: backward Euler has `O(dt^2)`
local truncation error, so on any sequence of steps summing to the same time
the stress converges to the analytic Maxwell curve `O(max dt)` and halving all
steps halves the error, however the steps vary.  G251 integrates a uniform, a
factor-10 alternating and a band-random step sequence to that analytic value
(both the spin-free kernel and the composed update above) and measures the
order, and its driver-level half replays a forced-varying-`dt` run's captured
`(dt, velocity)` sequence through this function and reproduces the driver's
stress bit for bit.  ASPECT's `Use fixed elastic time step`, `Fixed elastic
time step` and `Use stress averaging` therefore have no counterpart here: they
replace the numerical step by a chosen elastic timescale, which is a modelling
decision (a seismic cycle, say) rather than a consistency requirement for an
adaptive-step run.  We do not expose them because this update needs no such
rescaling to be trustworthy, and a parameter that changes the elastic response
without a case that needs it would be an unvalidated choice.

`G <= 0` is the disabled convention (`shear_modulus = 0`): the state is left
untouched and nothing is computed, not even the rotation.

The **return value** is the co-rotation validity bookkeeping (G254): the
node-wise maxima over the mesh of
[`maxwell_corotation_band`](@ref)'s three per-step rotations, as
`(two_w_dt, two_taum_w, resolved_rotation)`.  They are accumulated in the node
loop that already reads `wxy` and `eta`, so the measurement costs a few flops
per node and allocates nothing, and it changes no stress component.
`run_semquad_simulation` accumulates them over the run and reports/warns
against [`MAXWELL_COROTATION_BOUND`](@ref).  With `G <= 0` the function returns
`nothing` before sampling anything, so a `shear_modulus = 0` run pays nothing.
"""
function _advance_maxwell_stress!(txx::Vector{Float64}, tyy::Vector{Float64},
                                  txy::Vector{Float64}, sq::SemQuad,
                                  ux::Vector{Float64}, uz::Vector{Float64},
                                  dt::Float64, eta::Vector{Float64},
                                  G::Float64)
    length(eta) == sq.N ||
        throw(ArgumentError("eta node field has $(length(eta)) entries, mesh " *
                            "needs $(sq.N)"))
    (length(txx) == sq.N && length(tyy) == sq.N && length(txy) == sq.N) ||
        throw(ArgumentError("Maxwell stress fields have lengths " *
                            "($(length(txx)), $(length(tyy)), " *
                            "$(length(txy))), mesh needs $(sq.N)"))
    # `G <= 0` is off, not "elastic with zero modulus": the scalar kernel would
    # return its input, but the co-rotational increment is not scaled by G and
    # would still rotate the state.  Keep the disabled path from touching it.
    # (This is also what makes the co-rotation guard free when G = 0: nothing is
    # sampled, so nothing beyond the two length checks is paid.)
    G <= 0.0 && return nothing
    eps = semquad_strain_rate_components(sq, ux, uz)
    wxy = semquad_spin_component(sq, ux, uz)
    # Co-rotation validity band (G254).  Maxima over the domain of the three
    # per-step rotations; `resolved_rotation = 2|W| min(dt, tau_M)` is the one
    # compared against the bound (see `MAXWELL_COROTATION_BOUND`).
    two_w_dt_max = 0.0
    two_taum_w_max = 0.0
    resolved_max = 0.0
    @inbounds for g in eachindex(txx)
        # Deviatoric projection (the Maxwell stress has zero trace).  The
        # out-of-plane normal strain is zero in plane strain but contributes its
        # 0 to the trace, so tr = exx + ezz and the implied third deviatoric
        # normal strain rate is -tr/3 (not stored; the docstring has the
        # tau_zz argument and the code evidence that the force term never needs
        # it).
        tr = eps.exx[g] + eps.ezz[g]
        # Objective part: rotate the old stress with the material before the
        # backward-Euler relaxation (`maxwell_stress_corotate` has the component
        # derivation; the xy sign is the one that is easy to get backwards).
        rxx, ryy, rxy = maxwell_stress_corotate(txx[g], tyy[g], txy[g],
                                                wxy[g], dt)
        txx[g] = maxwell_stress_step(rxx, eps.exx[g] - tr / 3.0, dt, eta[g], G)
        tyy[g] = maxwell_stress_step(ryy, eps.ezz[g] - tr / 3.0, dt, eta[g], G)
        txy[g] = maxwell_stress_step(rxy, eps.exz[g], dt, eta[g], G)
        band = maxwell_corotation_band(wxy[g], dt, eta[g] / G)
        two_w_dt_max = max(two_w_dt_max, band.two_w_dt)
        two_taum_w_max = max(two_taum_w_max, band.two_taum_w)
        resolved_max = max(resolved_max, band.resolved_rotation)
    end
    return (two_w_dt=two_w_dt_max, two_taum_w=two_taum_w_max,
            resolved_rotation=resolved_max)
end
