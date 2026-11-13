# Validate the dimensional mapping of the MantleSEM `:composite` rheology onto
# ASPECT's `diffusion dislocation` model (docs/dev/VALIDATION.md §2).
#
# Our law (julia/src/physics.jl, eta_law=:composite, comp_rule=:isostress):
#
#   eta_diff  = 1/(comp_Ad * exp(-comp_Ed/T_nd))
#   eta_disl  = 1/(comp_As * exp(-comp_Es/(n*T_nd)) * edot_nd^((n-1)/n))
#   isostress: eta = x*eta_disl(edot),  k*x + x^n = 1,  k = eta_disl(edot)/eta_diff
#
# ASPECT's law (source/material_model/diffusion_dislocation.cc):
#
#   eta_diff = 0.5*A_d^-1*d_g^m*exp(E_d/(R*T))
#   eta_disl = 0.5*A_s^(-1/n)*edot^((1-n)/n)*exp(E_s/(n*R*T))
#   same isostress Newton solve and the same [eta_min, eta_max] clamp.
#
# The script computes both at a grid of (T_nd, edot_nd, z) and prints the
# relative difference.  T_nd = T + 0.5 is the absolute nondimensional
# temperature of our model (T is the field), T_ASPECT = T_nd*dT.

using MantleSEM
using Printf

const R_GAS = 8.3144621

# ---- our case card -------------------------------------------------------
const COMP_AD = 1.6240183119e5
const COMP_AS = 2.6796419238e-1
const COMP_ED = 12.0
const COMP_ES = 17.0
const COMP_N  = 3.0
const ETA_T0  = 0.5
const EPS_MIN = 1.0e-12
const ETA_MIN = 1.0e-4
const ETA_MAX = 1.0e4

# ---- dimensional scale set (doc section 2) ------------------------------
const D_BOX = 2.9e6       # m
const KAPPA = 1.0e-6      # m^2/s
const DT    = 1000.0      # K
const RHO0  = 3300.0      # kg/m^3
const GRAV  = 9.81        # m/s^2
const ALPHA = 3.0e-5      # 1/K
const CP    = 1000.0      # J/(kg K)
const RA    = 1.0e5
const H_NUM = 10.0
const ETA0  = RHO0*GRAV*ALPHA*DT*D_BOX^3/(RA*KAPPA)
const S_RATE = KAPPA/D_BOX^2
const D_GRAIN = 1.0e-3    # m (ASPECT's `Grain size`)

const A_D = COMP_AD*D_GRAIN^COMP_N/(2*ETA0)
const A_S = (COMP_AS/(2*ETA0))^COMP_N * S_RATE^(-(COMP_N-1))
const E_D = COMP_ED*R_GAS*DT
const E_S = COMP_ES*R_GAS*DT

@printf("eta0 = %.6e Pa s, S = %.6e 1/s\n", ETA0, S_RATE)
@printf("A_d  = %.6e, E_d = %.6f J/mol\n", A_D, E_D)
@printf("A_s  = %.6e, E_s = %.6f J/mol\n", A_S, E_S)

"""ASPECT's diffusion-dislocation law, in units of eta0, for absolute
nondimensional temperature T_nd and nondimensional strain rate edot_nd."""
function aspect_eta(T_nd::Float64, edot_nd::Float64, z::Float64)
    T_K = T_nd*DT
    ed  = max(edot_nd*S_RATE, 1.4e-20)          # ASPECT's Minimum strain rate
    # branch viscosities exactly as in calculate_isostrain_viscosities()
    pref_d = A_D * D_GRAIN^(-COMP_N) * exp(-(E_D)/(R_GAS*T_K))
    # start from the diffusion branch and solve tau/(2 eta_d) + A_s tau^n
    # exp(-E_s/(R T)) = ed  by Newton on tau
    eta_d = 0.5/pref_d
    B = A_S*exp(-E_S/(R_GAS*T_K))
    tau = 2*eta_d*ed/(1 + 2*eta_d*B*(2*eta_d*ed)^(COMP_N-1))
    for it in 1:200
        f  = tau/(2*eta_d) + B*tau^COMP_N - ed
        df = 1/(2*eta_d) + COMP_N*B*tau^(COMP_N-1)
        dtau = f/df
        tau -= dtau
        abs(dtau) <= 1e-15*abs(tau) && break
    end
    eta = tau/(2*ed)
    return min(max(eta/ETA0, ETA_MIN), ETA_MAX), eta/ETA0
end

pr = Params(Ra=RA, eta_law=:composite,
            comp_Ad=COMP_AD, comp_As=COMP_AS,
            comp_Ed=COMP_ED, comp_Es=COMP_ES, comp_n=COMP_N,
            comp_eps_min=EPS_MIN, eta_T0=ETA_T0,
            eta_min=ETA_MIN, eta_max=ETA_MAX, latent_L=0.11)

Ts = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 2.5]
eds = [1.0e-4, 1.0e-2, 1.0, 1.0e2, 1.0e3, 1.0e4]
zs = [0.0, 0.5, 0.9]

worst = 0.0
worst_at = (0.0, 0.0, 0.0)
rows = 0
for T in Ts, ed in eds, z in zs
    eta_ours, clipped = eta_composite_at(T, ed, z, pr)
    T_nd = T + ETA_T0
    eta_asp, raw = aspect_eta(T_nd, ed, z)
    rel = abs(eta_asp - eta_ours)/max(abs(eta_ours), 1e-300)
    global rows += 1
    if rel > worst
        global worst = rel; global worst_at = (T, ed, z)
    end
end
@printf("\n%d points: worst relative difference = %.3e at (T=%.2f, edot=%.1e, z=%.2f)\n",
        rows, worst, worst_at...)

# a few explicit rows
@printf("\n%8s %10s %6s %14s %14s %12s\n", "T", "edot", "z", "eta_ours", "eta_ASPECT", "rel.diff")
for (T, ed, z) in [(0.0, 1e3, 0.5), (0.5, 1e3, 0.5), (0.9, 1e3, 0.5),
                   (1.5, 1e3, 0.5), (0.5, 1.0, 0.5), (0.5, 1e4, 0.5),
                   (0.5, 1e3, 0.0), (0.5, 1e3, 0.9)]
    eta_ours, _ = eta_composite_at(T, ed, z, pr)
    eta_asp, _ = aspect_eta(T + ETA_T0, ed, z)
    @printf("%8.2f %10.1e %6.2f %14.6e %14.6e %12.2e\n",
            T, ed, z, eta_ours, eta_asp, abs(eta_asp-eta_ours)/eta_ours)
end
