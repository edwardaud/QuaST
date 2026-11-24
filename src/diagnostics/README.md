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

The CBF and exact-residual paths inherit the mapped energy operator of cut 3, so
they are geometry-generic apart from their normalisations: `semquad_nu_cbf`
divides by the wall's physical measure and `semquad_exact_residual` subtracts the
mapped base-state curvature `int lap(Tbar(h)) dV` that the box hides
(`_mapped_base_curvature`; see the cut-4 defect D2 in
`docs/dev/GEOM_PROGRESS.md`).  G270 (`test/t_g270_mapped_diagnostics.jl`) proves
the routing: affine exactness against analytic integrals, rotation invariance
against the box bodies, a 1e-14 mapped-vs-box cross-check on the box map, an
O(20 %) measured failure for a wrong normal or face Jacobian, and the end-to-end
mapped driver with the (ID1) energy-budget identity.

| File | What it is |
|---|---|
| `flux.jl` | Consistent boundary flux for the temperature equation and `semquad_nu_cbf`. |
| `global.jl` | Wall Nusselt, Vrms, mean T, the energy Nusselt number, the work/dissipation pair, the heat budget, the exact residual and the level stats. |
