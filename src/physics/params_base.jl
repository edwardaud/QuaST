# Base-state profiles Tbar/dTbar used by the energy assembly and diagnostics.
@inline Tbar(z::Float64, H::Float64) = (1.0 - z) + 0.5 * H * z * (1.0 - z)
@inline dTbar(z::Float64, H::Float64) = -1.0 + 0.5 * H * (1.0 - 2.0 * z)

