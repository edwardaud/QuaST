# Diagnostics

The acceptance quantities and the auxiliary fields: wall Nusselt numbers from
the pointwise GLL gradient, Vrms, mean temperature, the work and dissipation
integrals, the consistent boundary flux (CBF) that replaces the wall derivative
with the discrete reaction, and the entropy-viscosity field the energy assembly
uses on the adaptive mesh. They read a solved state; none of them changes it.

| File | What it is |
|---|---|
| `flux.jl` | Consistent boundary flux for the temperature equation and `semquad_nu_cbf`. |
| `global.jl` | Wall Nusselt, Vrms, mean T and the energy Nusselt number. |
