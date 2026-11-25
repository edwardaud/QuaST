#!/usr/bin/env julia
# verify100_ic_check.jl -- is the ASPECT card's t=0 state the same function as
# the QuaST case card's t=0 state?
#
# QuaST card (examples/case_realistic_2d.jl, literals):
#
#   Tbar(y) = (1 - y) + (H/2) y (1 - y)                  y = height, H = 10
#   theta0  = 0.05 sin(pi y) cos(2 pi x / Lx)
#           + 0.02 sin(2 pi y) cos(4 pi x / Lx)          Lx = 2.5
#   T       = Tbar + theta0
#
# ASPECT card (aspect_runs/realistic/realistic.prm, `Initial temperature model =
# function`), with ASPECT's z the *height* (z = 0 at the bottom, z = d at the
# top; the box geometry's `depth` is d - z):
#
#   T_K = 500 + 1000 [ (1 - z/d) + 0.5 H (z/d)(1 - z/d)
#                      + 0.05 sin(pi z/d) cos(2 pi x / Lx)
#                      + 0.02 sin(2 pi z/d) cos(4 pi x / Lx) ]
#   T_nd = T_K / 1000 - 0.5
#
# With y = z/d the two expressions are identical term by term.  This script
# evaluates them independently (the ASPECT one from the literals of the prm,
# x converted from metres) on a grid and then checks every constant that the
# two cards must share: geometry, H, Ra/eta0, the composite rheology, the two
# phase transitions and their latent coefficients, and the boundary conditions.
#
# Usage: julia verify100_ic_check.jl [prm path]
# Exit code 0 if every check passes.

using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PRM  = length(ARGS) >= 1 ? ARGS[1] :
             joinpath(ROOT, "aspect_runs", "realistic", "realistic.prm")

# --------------------------------------------------------------- card ------
const LX     = 2.5
const HHEAT  = 10.0
const DEPTH  = 1.0                      # our box height (nondimensional)
const RA     = 1.0e5
const COMP_AD = 1.6240183119e5
const COMP_AS = 2.6796419238e-1
const COMP_ED = 12.0
const COMP_ES = 17.0
const COMP_N  = 3.0
const ETA_T0  = 0.5
const ETA_MIN = 1.0e-4
const ETA_MAX = 1.0e4
const SLOPE410 = 0.032
const SLOPE660 = -0.032
const WIDTH_PT = 0.025
const Z410 = 0.87
const Z660 = 0.78
const DRHO410 = 180.0
const DRHO660 = 300.0
const LAT410 = -0.059364
const LAT660 =  0.130596
const RHO0 = 3300.0
const ALPHA0 = 3.0e-5
const DT = 1000.0
const CP = 1000.0
const GRAV = 9.81
const D_SI = 2.9e6
const KAPPA = 1.0e-6
const R_GAS = 8.3144621

Tbar(y)  = (1.0 - y) + 0.5 * HHEAT * y * (1.0 - y)
seed(x, y) = 0.05 * sinpi(y) * cospi(2.0 * x / LX) +
             0.02 * sinpi(2.0 * y) * cospi(4.0 * x / LX)
card_T(x, y) = Tbar(y) + seed(x, y)

"""The prm's `Function expression`, transcribed literally with x in metres and
z in metres, returned in the card's nondimensional T = T_K/dT - 0.5."""
function prm_T(x_nd, y)
    x = x_nd * D_SI
    z = y * D_SI
    T_K = 500.0 + 1000.0 * ((1.0 - z / D_SI) +
                            0.5 * 10.0 * (z / D_SI) * (1.0 - z / D_SI) +
                            0.05 * sin(pi * z / D_SI) * cos(2.0 * pi * x / 7.25e6) +
                            0.02 * sin(2.0 * pi * z / D_SI) * cos(4.0 * pi * x / 7.25e6))
    return T_K / 1000.0 - 0.5
end

# ---------------------------------------------------------------- prm ------
ispath(PRM) || error("no prm at $PRM")
const PRM_TXT = read(PRM, String)

function prm_key(key::AbstractString)
    m = match(Regex("^\\s*set " * key * "\\s*=\\s*(.+?)\\s*\$", "m"), PRM_TXT)
    m === nothing && error("realistic.prm has no `set $key`")
    return strip(m.captures[1])
end

function prm_num(key::AbstractString)
    s = prm_key(key)
    v = tryparse(Float64, s)
    v === nothing && error("prm key $key is not a number: $s")
    return v
end

function phase_list(key::AbstractString)
    s = prm_key(key)
    s = replace(s, r"^background\s*:\s*" => "")
    # the prm spells the two transitions either with `|` (depths, widths,
    # slopes, temperatures) or with a comma (density jumps)
    parts = split(replace(s, "|" => ","), ',')
    return [parse(Float64, strip(p)) for p in parts]
end

const D_PRM    = prm_num("Y extent")
const LX_PRM   = prm_num("X extent")
const H_R      = prm_num("Radiogenic heating rate")
const A_D_PRM  = prm_num("Prefactors for diffusion creep")
const E_D_PRM  = prm_num("Activation energies for diffusion creep")
const A_S_PRM  = prm_num("Prefactors for dislocation creep")
const E_S_PRM  = prm_num("Activation energies for dislocation creep")
const N_PRM    = prm_num("Stress exponents for dislocation creep")
const ETAMIN_P = prm_num("Minimum viscosity")
const ETAMAX_P = prm_num("Maximum viscosity")
const T_BOT    = prm_num("Bottom temperature")
const T_TOP    = prm_num("Top temperature")
const CFL_PRM  = prm_num("CFL number")
const RDEPTH   = phase_list("Phase transition depths")
const RWIDTH   = phase_list("Phase transition widths")
const RSLOPE   = phase_list("Phase transition Clapeyron slopes")
const RDRHO    = phase_list("Phase transition density jumps")
const RTEMP    = phase_list("Phase transition temperatures")

# eta0 = rho0 g alpha dT d^3 / (Ra kappa);  H = H_r d^2/(cp kappa dT)
const ETA0 = RHO0 * GRAV * ALPHA0 * DT * D_PRM^3 / (RA * KAPPA)
const H_FROM_PR = H_R * D_PRM^2 / (CP * KAPPA * DT)
# branch-law mapping
const A_D_CARD = COMP_AD * 1.0e-9 / (2.0 * ETA0)          # d_grain = 1e-3 m
const E_D_CARD = COMP_ED * R_GAS * DT
const S_SCALE  = KAPPA / D_PRM^2
const A_S_CARD = (COMP_AS / (2.0 * ETA0))^3 * S_SCALE^(-2)
const E_S_CARD = COMP_ES * R_GAS * DT
# transition mapping: depth_frac = 1 - z0;  width_m = width * d;
# gamma = slope rho0 g d / dT ; B = drho/(rho0 alpha dT)
const Z410_M = (1.0 - Z410) * D_PRM
const Z660_M = (1.0 - Z660) * D_PRM
const WIDTH_M = WIDTH_PT * D_PRM
const GAMMA410 = SLOPE410 * RHO0 * GRAV * D_PRM / DT
const GAMMA660 = SLOPE660 * RHO0 * GRAV * D_PRM / DT
const B410 = DRHO410 / (RHO0 * ALPHA0 * DT)
const B660 = DRHO660 / (RHO0 * ALPHA0 * DT)
# per-phase latent coefficients, the card's formula: -T_nd dS_A/cp,
# dS_A = gamma drho/rho^2.  The card's `z0` is the transition's *height* y0
# (the prm's depth is (1 - z0) d = 377/638 km), so the base state is Tbar(z0);
# Tbar(0.87) = 0.6955 and Tbar(0.78) = 1.078, i.e. T_nd = 1.1955 / 1.578.
dS(gamma, drho) = gamma * drho / RHO0^2
lat410_card = -((Tbar(Z410) + ETA_T0) * dS(GAMMA410, DRHO410) / CP)
lat660_card = -((Tbar(Z660) + ETA_T0) * dS(GAMMA660, DRHO660) / CP)

# -------------------------------------------------------------- checks -----
const PASS = Bool[]
function check(name::AbstractString, have, want; rtol=1e-6, atol=0.0)
    ok = isapprox(float(have), float(want); rtol=rtol, atol=atol)
    push!(PASS, ok)
    @printf("  %-46s prm/derived = %-18.10g card = %-18.10g %s\n",
            name, float(have), float(want), ok ? "ok" : "MISMATCH")
end
function checktrue(name::AbstractString, ok::Bool, detail::AbstractString="")
    push!(PASS, ok)
    @printf("  %-46s %s %s\n", name, ok ? "ok" : "MISMATCH", detail)
end

println("verify100_ic_check.jl")
println("  prm: $PRM")

println("\n[1] t=0 temperature field: card formula against prm formula")
function ic_grid_stats()
    worst = 0.0; wx = 0.0; wy = 0.0; tmax = -Inf
    for j in 0:400, i in 0:600
        x = LX * i / 600
        y = j / 400
        d = abs(card_T(x, y) - prm_T(x, y))
        if d > worst
            worst = d; wx = x; wy = y
        end
        tmax = max(tmax, card_T(x, y))
    end
    # trapezoid on the 601x401 grid: the interior point average is not the
    # volume mean (its weight would be 1/601 rather than h), so use the weights
    hx = LX / 600; hy = 1.0 / 400
    tmean = sum((i in (0, 600) ? 0.5 : 1.0) * (j in (0, 400) ? 0.5 : 1.0) *
                card_T(LX * i / 600, j / 400)
                for i in 0:600, j in 0:400) * hx * hy / LX
    return (worst=worst, wx=wx, wy=wy, tmax=tmax, tmean=tmean)
end
const ICS = ic_grid_stats()
@printf("  max |T_card - T_prm| over 601x401 points = %.3e  (at x=%.4f y=%.4f)\n",
        ICS.worst, ICS.wx, ICS.wy)
checktrue("analytic t=0 fields identical", ICS.worst < 1.0e-12,
          @sprintf("(max diff %.1e)", ICS.worst))
# orientation: ASPECT's z is the height, so z/d = y; the terms map as
# (1 - z/d) -> 1 - y, sin(pi z/d) -> sin(pi y), sin(2 pi z/d) -> sin(2 pi y).
# Read as a depth instead (z/d = 1 - y) the 0.02 harmonic would flip sign and
# the two cards would differ by 0.04 * |sin(2 pi y) cos(4 pi x / Lx)|, i.e. up
# to 0.04 in T_nd.  This is the magnitude the equality check rules out.
wrong = maximum(0.04 * abs(sinpi(2.0 * j / 400) * cospi(4.0 * (LX * i / 600) / LX))
                for i in 0:600, j in 0:400)
@printf("  a depth-z reading would differ by %.3f in T_nd (orientation sanity)\n", wrong)
@printf("  analytic t=0 extent: Tmin=%.6f Tmax=%.6f mean=%.6f (base-state mean %.6f)\n",
        0.0, ICS.tmax, ICS.tmean, 1.0 + HHEAT / 12.0)

println("\n[2] geometry and heating")
check("X extent / Y extent = 2.5", LX_PRM / D_PRM, 2.5)
check("H from Radiogenic heating rate", H_FROM_PR, HHEAT)
checktrue("ASPECT z is height (T at z=0 is T_bot)", T_BOT == 1500.0 && T_TOP == 500.0,
          @sprintf("(T_bot=%.0f K, T_top=%.0f K, T_nd top/bottom = %.1f/%.1f)",
                  T_BOT, T_TOP, T_TOP / DT - 0.5, T_BOT / DT - 0.5))

println("\n[3] Rayleigh number and viscosity scale")
@printf("  eta0 = rho0 g alpha dT d^3/(Ra kappa) = %.6e Pa s\n", ETA0)
check("Minimum viscosity / eta0", ETAMIN_P / ETA0, ETA_MIN)
check("Maximum viscosity / eta0", ETAMAX_P / ETA0, ETA_MAX)

println("\n[4] composite rheology (card comp_* -> ASPECT diffusion dislocation)")
check("A_d", A_D_PRM, A_D_CARD; rtol=2e-6)
check("E_d [J/mol]", E_D_PRM, E_D_CARD; rtol=1e-8)
check("A_s", A_S_PRM, A_S_CARD; rtol=2e-6)
check("E_s [J/mol]", E_S_PRM, E_S_CARD; rtol=1e-8)
check("stress exponent n", N_PRM, COMP_N)

println("\n[5] the two phase transitions (410 at z0=0.87, 660 at z0=0.78)")
check("410 depth [m]  (1-z0) d", RDEPTH[1], Z410_M; rtol=1e-6)
check("660 depth [m]  (1-z0) d", RDEPTH[2], Z660_M; rtol=1e-6)
check("410 width [m]  w d", RWIDTH[1], WIDTH_M; rtol=1e-6)
check("660 width [m]  w d", RWIDTH[2], WIDTH_M; rtol=1e-6)
check("410 Clapeyron [Pa/K]  slope rho g d/dT", RSLOPE[1], GAMMA410; rtol=2e-6)
check("660 Clapeyron [Pa/K]", RSLOPE[2], GAMMA660; rtol=2e-6)
check("410 density jump [kg/m3]", RDRHO[1], DRHO410; rtol=2e-5)
check("660 density jump [kg/m3]", RDRHO[2], DRHO660; rtol=2e-5)
check("410 reference T [K] = (0.5+0.5) dT", RTEMP[1], (ETA_T0 + 0.5) * DT)
check("660 reference T [K]", RTEMP[2], (ETA_T0 + 0.5) * DT)
check("410 buoyancy number B from prm drho", RDRHO[1] / (RHO0 * ALPHA0 * DT), B410; rtol=2e-5)
check("660 buoyancy number B from prm drho", RDRHO[2] / (RHO0 * ALPHA0 * DT), B660; rtol=2e-5)
println(@sprintf("  latent coefficient 410 = %+.6f (card %+.6f), 660 = %+.6f (card %+.6f)",
                 lat410_card, LAT410, lat660_card, LAT660))
checktrue("printed latent coefficients match the card",
          isapprox(lat410_card, LAT410; atol=5e-7) &&
          isapprox(lat660_card, LAT660; atol=5e-7))

println("\n[6] boundary conditions and formulation")
checktrue("top/bottom temperatures fixed", occursin("Fixed temperature boundary indicators = bottom, top", PRM_TXT))
checktrue("four free-slip walls",
          occursin("Tangential velocity boundary indicators = left, right, bottom, top", PRM_TXT))
checktrue("Boussinesq formulation", occursin("Formulation = Boussinesq approximation", PRM_TXT))
check("CFL number", CFL_PRM, 0.5)

println("\n[7] mesh (deliberately different)")
@printf("  card: max_level 7 on a 5x2 base = leaf 1/256; production budget 10000 leaves\n")
@printf("  prm : initial global refinement %s + initial adaptive %s (min level %s)\n",
        prm_key("Initial global refinement"), prm_key("Initial adaptive refinement"),
        prm_key("Minimum refinement level"))

nbad = count(!, PASS)
println(@sprintf("\nRESULT: %d checks, %d failed", length(PASS), nbad))
exit(nbad == 0 ? 0 : 1)
