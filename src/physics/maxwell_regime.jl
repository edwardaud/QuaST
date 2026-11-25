# The viscosity regime of a viscoelastic run.
#
# The 2a closure replaces the material viscosity by
# `eta_eff = 1/(1/eta + 1/(G*dt))`, so `G*dt` caps `eta_eff` from above and the
# contrast the solve sees is not the material contrast.  `maxwell_regime`
# measures the material range and contrast, the effective range and contrast at
# one step's `dt`, and the range of `G*dt/eta`; the driver reports it once at the
# first stress step and once at the end of the run.  Pure measurement: it reads
# the nodal material field and returns a NamedTuple.

# --- the viscosity regime of a viscoelastic run (G255) ----------------------
#
# The 2a closure replaces the material viscosity by
#
#     eta_eff = 1/(1/eta + 1/(G dt)),
#
# which *caps* eta from above at the elastic viscosity `G dt`.  That is correct
# physics, but it means a viscoelastic run does not solve "the same problem plus
# memory": every viscosity well above `G dt` is flattened to `G dt`, so a
# material contrast carried by high-viscosity material (a cold lid, say) can be
# removed from the problem the Stokes solve sees by several decades, and the
# solve becomes *softer*.  Measured on the developed production state: material
# contrast 3.3e6 against an effective ~5e2 at `G dt = 1.485` (the second
# step of the case).  `maxwell_regime` measures both ranges on one viscosity
# field and at one step so the driver can report the difference rather than
# leave it implicit.

"""
    maxwell_regime(eta, dt, G)

The material and effective viscosity ranges of one step, and the ratio
`G*dt/eta`, as a named tuple:

    material_min, material_max, material_contrast
    effective_min, effective_max, effective_contrast
    gdt_over_eta_min, gdt_over_eta_max

`eta` is the **material** (unsoftened) nodal viscosity field the Maxwell stress
update uses, `dt` the step actually taken, and `G = Params.shear_modulus`; the
effective viscosity is the 2a closure `1/(1/eta + 1/(G dt))` per node.  Each
contrast is `max/min` over the field.

`eta_eff` is the harmonic mean of `eta` and `G*dt`, so it never exceeds either:
the effective contrast never exceeds the material one, and it is smaller
whenever the field straddles the cap `G*dt`.  The run-time report that uses this
helper states the resulting collapse in decades (G255).

The function is read-only (it does not modify `eta`) and pure.  `G <= 0` and
`dt <= 0` are rejected: with the viscoelastic terms off there is no
effective-viscosity regime to report (the `shear_modulus = 0` convention), and
the driver only calls it for `G > 0`.
"""
function maxwell_regime(eta::Vector{Float64}, dt::Float64, G::Float64)
    G > 0.0 || throw(ArgumentError("maxwell_regime requires G > 0; with " *
                                   "shear_modulus = 0 there is no " *
                                   "effective-viscosity regime to report"))
    dt > 0.0 || throw(ArgumentError("maxwell_regime requires dt > 0"))
    isempty(eta) &&
        throw(ArgumentError("maxwell_regime requires a non-empty viscosity " *
                            "field"))
    Gdt = G * dt
    material_min = Inf
    material_max = -Inf
    effective_min = Inf
    effective_max = -Inf
    ratio_min = Inf
    ratio_max = -Inf
    @inbounds for i in eachindex(eta)
        e = eta[i]
        # The viscosity laws and their bounds are positive by construction; a
        # non-positive value here would be a closure bug, and the arithmetic
        # below would report it as an infinite contrast rather than hide it.
        eff = 1.0 / (1.0 / e + 1.0 / Gdt)
        material_min = min(material_min, e)
        material_max = max(material_max, e)
        effective_min = min(effective_min, eff)
        effective_max = max(effective_max, eff)
        ratio = Gdt / e
        ratio_min = min(ratio_min, ratio)
        ratio_max = max(ratio_max, ratio)
    end
    return (material_min=material_min, material_max=material_max,
            material_contrast=material_max / material_min,
            effective_min=effective_min, effective_max=effective_max,
            effective_contrast=effective_max / effective_min,
            gdt_over_eta_min=ratio_min, gdt_over_eta_max=ratio_max)
end
