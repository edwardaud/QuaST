# Diagnostics

The acceptance quantities and the auxiliary fields: wall Nusselt numbers from
the pointwise GLL gradient, Vrms, mean temperature, the work and dissipation
integrals, the consistent boundary flux (CBF) that replaces the wall derivative
with the discrete reaction, and the entropy-viscosity field the energy assembly
uses on the adaptive mesh. They read a solved state; none of them changes it.

Every entry point dispatches once on the stored geometry (stage 0 cut 4): the
`CartesianBox` body is the historical statement, character for character (that
is what keeps the nine state-dump hashes), and the mapped body reads the geometry
layer's leaf/face metric and physical height.  The mapped bodies use
`quad_point`/`quad_measure`/`basis_grad` for the volume integrals, the mapped
outward normal and face Jacobian for the wall integrals (`grad(theta) . n` at the
wall GLL nodes, in the box's "positive upward" sign convention), `height(g,x,y)`
for every base-state closure, `upward_velocity` for every directional physics
term (`u . grad h` instead of `u_z`), and `total_measure`/`boundary_measure` for
the normalisations instead of `total_area(qt)`.  The box bodies stay: the box's
GLL wall weights, one-sided `2/h` wall derivative and sign table are not routed
through the mapped code (`GEOM_DESIGN` §4.2 E8/E13/E16).

The CBF and exact-residual paths inherit the mapped energy operator of cut 3 and
are geometry-generic: `semquad_cbf_theta_flux` selects the wall nodes from the
geometry's Nusselt sides (`nu_wall_sides`; `:yplus`/`:yminus` on the box and the
box-in-height validation geometries, the two circles on the annulus), and
`semquad_nu_cbf` normalises with each wall's physical measure
(`boundary_measure`) and the geometry's base wall references
(`semquad_base_wall_refs`), adding a per-wall-ratio `PolarAnnulus` method;
`semquad_exact_residual` subtracts the mapped base-state curvature
`int lap(Tbar(h)) dV` that the box hides (`_mapped_base_curvature`; see the cut-4
defect D2 in `docs/dev/GEOM_PROGRESS.md`).  G270
(`test/t_g270_mapped_diagnostics.jl`) proves the routing: affine exactness
against analytic integrals, rotation invariance against the box bodies, a 1e-14
mapped-vs-box cross-check on the box map, an O(20 %) measured failure for a
wrong normal or face Jacobian, and the end-to-end mapped driver with the (ID1)
energy-budget identity.  **Stage-0.5 fix campaign (audit defects D2/D3, gate
G279):** before it, `semquad_cbf_theta_flux` had no geometry dispatch and summed
the residual on the logical `y` rows — on an annulus ring that is the free
periodic seam, so it returned ~1.9e-16 where the wall flux is +1.0496 — and the
mapped Nusselt pair used the Cartesian base gradient, returning `(1,1)` where
the per-wall normalisation gives `(0.900, 0.901)` on the audit's conduction
state.  G279 pins the corrected annulus flux against the analytic
`2π ΔT/ln(R1/R0)` and the CBF pair against the pointwise `semquad_nu_walls`
pair (a measured 0.24 % O(h) family difference).

**Stage 0.5b1: the annulus Nusselt pair and the geometry base state.**  The
mapped bodies evaluate the base temperature/gradient through
`semquad_base_T(pr, g, x, y)`/`semquad_base_gradient(pr, g, x, y)`, so the
annulus's radial log/H conduction state (not the Cartesian quadratic in `h`)
drives `mean_T`, the work integral and the base-advection diagnostic.  The wall
Nusselt pair has an annulus method: the arc-length wall flux is normalised by
the **per-wall** radial conduction flux, `Nu_wall = <−dT/dr>_wall /
(−dTbar/dr|_{r_wall})`, i.e. `Nu_wall = r_wall |ln f| <−dT/dr>_wall` at
`H = 0` (the ANNULUS_REFS §6.2 multiplier form) and the H-inclusive
`Q_wall/Q_ref(r_wall)` in general.  The generic mapped body keeps the box's
single-gradient convention for the box-in-height validation geometries.  The
annulus driver keeps the `nu_cbf`/`extdiag` **history columns** out of the
benchmark history as a definition/units choice (the CBF family is mapped now,
but the published annulus Nusselt definition is the per-wall pointwise pair that
G274/G277 gate); the refusal text says so.

**Stage-0.5 fix campaign: box-signature measures and the history closure.**
`total_area(qt)` is the logical box-tree measure and now refuses a
periodic-seam tree (`ArgumentError` pointing at `total_measure(g, sq)`);
`semquad_level_stats(qt)` (the box wall-strip signature) refuses the same tree
and points at the geometry-aware `semquad_level_stats(g, qt)` method.  A
bounded mapped tree (a wedge) is topologically indistinguishable from a box
tree, so those two signatures stay the documented logical-`y` claim for it.
The history `closure`/`closure_nolatent` columns use
`(Ibase + latent_dgdt − flux)/A_phys` on a mapped geometry — the same physical
steady-balance residual as the box's `Nut − Nub − (H − Ibase)` form, which is
its area-1 special case (`A_phys = total_measure(g, sq)`; the box branch keeps
its historical formula bit for bit).  `semquad_nu_energy` is mapped
(`1 + H/2 + <u·grad h θ>`, normalised by `total_measure`) and is gated on the
annulus by G279; it is still not a history column, and its annulus value is not
the benchmark Nusselt number.

| File | What it is |
|---|---|
| `flux.jl` | Consistent boundary flux for the temperature equation and `semquad_nu_cbf`. |
| `global.jl` | Wall Nusselt, Vrms, mean T, the energy Nusselt number, the work/dissipation pair, the heat budget, the exact residual and the level stats. |
