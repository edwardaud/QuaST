#!/usr/bin/env julia
# =============================================================================
# case_realistic_2d.jl — "realistic 2-D mantle convection", statistical case,
# round 1 (no yielding, no strain weakening).  Parameter card + run.
#
#   julia --project=. examples/case_realistic_2d.jl <mode> [args]
#
#   modes
#     smoke             10 steps, one frame            (sanity, ~2 min)
#     short             `SHORT_STEPS` (=150) steps      (stability check)
#     spinup            production spin-up WITH AMR, 10^4-step feasibility run
#     freeze <ckpt>     restart that checkpoint with the mesh frozen
#                       (amr_every = 0) and run the statistics phase
#     wallrefine <ckpt> restart that checkpoint under the wall-refinement rule
#                       CASE_WALLMIN / CASE_WALLSTRIP: the seed copy gets the
#                       wall strips refined and the leaf budget redistributed
#                       (see the wall_checkpoint docstring, §14)
#     symseed <ckpt>    write a copy of that checkpoint whose stored `symmetry`
#                       is `CASE_SYMMETRY`, so a switch of the x-mirror rule can
#                       be resumed (the restart guard compares `symmetry`; the
#                       state is copied verbatim, only that one string changes)
#
#   environment
#     CASE_OUT   output root            (default out_julia/case_realistic)
#     CASE_STEPS step budget            (spinup default 10000; freeze 20000)
#     CASE_FRAME stride in steps        (0 = the mode default)
#     CASE_HYST                         hysteresis cooldown of the balanced AMR
#                                       cycle, in AMR cycles (default 1).
#                                       The balanced policy of docs/dev/AMR.md §3
#                                       (forward coarsening + budget exchange +
#                                       hysteresis) is the only adaptive policy.
#     CASE_AMR_GATE                     1 (default) = the 2:1 `coarsening_allowed`
#                                       pre-check of the balanced policy
#                                       (docs/dev/AMR.md §4); 0 turns it off
#     CASE_COARSEN                      AMR coarsening fraction: 0.05 (default,
#                                       the balanced policy's production value;
#                                       the historical A/B pair used 0.005)
#     CASE_LINEAR linear solver         (default auto; resolves to MG at the
#                                       production size, see the LIN_SOLVE
#                                       comment below)
#     CASE_WALLMIN / CASE_WALLSTRIP     force the top/bottom wall strips to at
#                                       least this level (default 5 / 0.1, the
#                                       production wall rule; CASE_WALLMIN=0
#                                       turns it off)
#     CASE_SYMMETRY                     x-mirror closure of the AMR passes:
#                                       `none` (default, the production card:
#                                       the pair-strict mirror rule off, the
#                                       field is not mirror-symmetric,
#                                       docs/dev/BUG_LEDGER.md §5.6/§6) or `x` (the
#                                       historical card and the E arm).  The
#                                       value is part of `run_config` and is
#                                       compared on restart, so an `x` resume of
#                                       the current default needs
#                                       CASE_SYMMETRY=x (or a `symseed` copy).
#     CASE_EXTDIAG                     1 = append the exact energy-identity
#                                       residual `resid_A` (R/A, docs/
#                                       NUSSELT_AUDIT.md §3.2) to history.csv.
#                                       Default 0: the 18-column layout (plus
#                                       closure_nolatent when latent heat is on)
#                                       is unchanged byte for byte.  Because the
#                                       column count changes, CASE_EXTDIAG=1 is
#                                       accepted only in a new CASE_OUT
#                                       directory (checked in main()).
#     CASE_STOKES_SCALING               solver-side equilibration of the direct
#                                       saddle: `none` (default, the historical
#                                       path bit for bit) or `equilibrated`
#                                       (docs/dev/BUG_LEDGER.md; engages
#                                       only on a solve whose unscaled unshifted
#                                       answer fails its health check).  It is
#                                       part of `run_config` and compared on
#                                       restart: a checkpoint without the key
#                                       predates the option and is read as
#                                       `none`, so switching it on for such a
#                                       checkpoint is refused.
#
# Everything this file writes lives under CASE_OUT:
#     run.log (created by the shell redirect), history.csv (at exit),
#     spinup.h5 / frozen.h5 (checkpoints), frames/frame_%05d_T.bin,
#     frames/frame_%05d_mesh.csv, frames/frame_meta.csv
#
# =============================================================================
# PARAMETER CARD  (all numbers are literals: the run does not depend on
# calib_realistic_2d.jl, which is the derivation of the two prefactors)
# -----------------------------------------------------------------------------
# Geometry / BCs
#   box 2.5 x 1 (5 x 2 base cells, square leaves), four free-slip walls,
#   T_top = 0, T_bot = 1, internal heating H = 10.0  (internal-heating number,
#   i.e. the source of the nondimensional theta equation in this code:
#   Tbar(z) = (1-z) + H/2 z(1-z), lap(Tbar) = -H, so H is the volumetric
#   heating and the conductive base state carries it exactly).
# Rheology  eta_law = :composite, comp_rule = :isostress (ASPECT's partition),
#   comp_n = 3, eta_T0 = 0.5, comp_Ed = 12, comp_Es = 17, comp_Vd = comp_Vs = 0,
#   comp_eps_min = 1e-12, eta in [1e-4, 1e6].
#   Prefactors calibrated (docs/dev/CASE_REALISTIC.md §3, calib_realistic_2d.jl) at
#   z = 0.5, edot_ref = 1.65e3 (the RMS strain rate of the constant-viscosity
#   precursor of this case) so that
#       (i)  eta(T=0.5, z=0.5, edot_ref) = 1
#       (ii) eta(T=0.1)/eta(T=0.9) at z=0.5, edot_ref = 1e4
#   -> comp_Ad = 1.6240183119e5, comp_As = 2.6796419238e-1
# Phase transitions (410 / 660 km equivalents, depth = 1-y)
#   y0 = 0.87 : width 0.025, Clapeyron slope +0.032 (dz/dT), drho = +180,
#               B = drho/(rho0 alpha0 dT) = 1.818
#   y0 = 0.78 : width 0.025, Clapeyron slope -0.032,           drho = +300,
#               B = 3.030
#   latent heat latent_L = L/(cp dT) = 0.11 (from L = T (dP/dT) (drho/rho^2):
#   0.094 for the 410, 0.131 for the 660; the code has a single L).
# Numerics
#   Ra = 1e5 (CASE_RA overrides), p = 2, max_level = 7 (leaf 1/256),
#   max_leaves = 10000, dt0/dtmax/dtmin = 1e-6/5e-5/1e-9 (CASE_DT* override),
#   comp_Ad/comp_As = 1.6240183119e5 / 2.6796419238e-1 (CASE_COMP_* override),
#   eta_min/eta_max = 1e-4 / 1e4 (CASE_ETA_MIN/MAX override),
#   amr_every = 5, refine_frac = 0.2,
#   amr_coarsen_gate = true, amr_hysteresis = 1,
#   coarsen_frac = 0.05, symmetry = :none
#   (the balanced policy's production set, i.e. the G card of
#   docs/dev/CASE_REALISTIC.md §1.5; the balanced AMR policy is the only one
#   since the classic policy was removed),
#   advection = :skew, ev = 0.11 / ev_cmax = 0.052, cfl = 0.5,
#   pic_mode = :picard, maxpic = 1, linear = :auto, stokes_tol = 1e-3,
#   mg_tol = 1e-8 (docs/dev/CASE_REALISTIC.md §1.6),
#   wall_min_level = 5, wall_strip = 0.1 (the production wall rule; the AMR
#   otherwise never refines the top wall, it sat at level 3/4
#   (h = 1/16..1/32) while the interior is L5/6, and the pointwise Nu_top there
#   differs from the wall-refined value by 1.2 % while the consistent boundary
#   flux differs by 9.6 % (docs/dev/CASE_REALISTIC.md §7.2, docs/dev/PHYSICS.md
#   §7.3.5).  CASE_WALLMIN=0 turns it off).
#   eta in [1e-4, 1e4]: the calibration needs eta(0.5)=1 with a 1e4 drop over
#   [0.1,0.9], so the cold lid would be eta ~ 2.2e3 at T=0 and eta_max = 1e6
#   produced a 1.6e8 contrast, at which the *direct* reduced-saddle residual is
#   round-off-limited at 4.1e-4 (measured, probe0.jl).  With eta_max = 1e4 the
#   measured contrast is 1.0e7 and the residual 8.5e-10, so 1e4 is used and
#   stokes_tol = 1e-3 only guards against a genuine failure.
#   (docs/dev/CASE_REALISTIC.md §2 records why p=2/max_level=7 rather than
#   p=4/max_level=6, and why the anelastic/EOS reference state of the brief is
#   NOT expressible together with internal heating in this driver.)
# No yielding (yield_c / yield_taumax = Inf, yield_phi = 0) and no strain
# weakening (weak_eps_ref = 0): round 2 adds them.
# =============================================================================

using QuaST, Printf, LinearAlgebra

# ------------------------------------------------------------------ card ----
# Box geometry.  The card is the 2.5 x 1 realistic case; `CASE_LX/CX/CY` exist so a
# square-box variant (the 1 x 1 probe at Ra = 1e8) can be run without editing it.
# Keep `LX/CX == 1/CY` or the leaves stop being square, which the AMR rules and the
# wall strip assume.
const LX           = parse(Float64, get(ENV, "CASE_LX", "2.5"))
const CX           = parse(Int, get(ENV, "CASE_CX", "5"))
const CY           = parse(Int, get(ENV, "CASE_CY", "2"))
# max_level / max_leaves are overridable only so the resolution check of
# docs/dev/CASE_REALISTIC.md §7 is a one-line command; the defaults are the card.
const MAXLEV       = parse(Int, get(ENV, "CASE_MAXLEV", "7"))   # leaf 1/256
const MAXLEAVES    = parse(Int, get(ENV, "CASE_MAXLEAVES", "10000"))
const DT0          = parse(Float64, get(ENV, "CASE_DT0", "1.0e-6"))
const DTMAX        = parse(Float64, get(ENV, "CASE_DTMAX", "5.0e-5"))
const DTMIN        = parse(Float64, get(ENV, "CASE_DTMIN", "1.0e-9"))
const PORD         = 2
# `CASE_RA` exists so the same card can be re-run at another Rayleigh number
# without editing it (the Ra = 1e7 pilot).  The default is the card's own value.
const RA           = parse(Float64, get(ENV, "CASE_RA", "1.0e5"))
const HHEAT        = 10.0
# The two prefactors carry the units of 1/eta0, and eta0 is fixed by Ra
# (`Ra = rho g alpha dT d^3 / (eta0 kappa)`).  A run at another Ra must therefore
# rescale them by `Ra / Ra_new` to keep the *dimensional* material constants
# fixed; `CASE_COMP_AD` / `CASE_COMP_AS` are that hook, and the defaults are the
# card's own values.
const COMP_AD      = parse(Float64, get(ENV, "CASE_COMP_AD", "1.6240183119e5"))
const COMP_AS      = parse(Float64, get(ENV, "CASE_COMP_AS", "2.6796419238e-1"))
const COMP_ED      = 12.0
const COMP_ES      = 17.0
const COMP_N       = 3.0
const ETA_T0       = 0.5
const COMP_EPS_MIN = 1.0e-12
# `eta_min` is a *physical* bound (rocks have a floor), not only a numerical
# guard: at high Ra the composite law drives the dislocation branch toward it and
# the material turns effectively isoviscous, which removes the feature the card
# is about.  `CASE_ETA_MIN/MAX` let a high-Ra variant state that bound explicitly.
const ETA_MIN      = parse(Float64, get(ENV, "CASE_ETA_MIN", "1.0e-4"))
const ETA_MAX      = parse(Float64, get(ENV, "CASE_ETA_MAX", "1.0e4"))
const SLOPE410     = 0.032
const SLOPE660     = -0.032
const WIDTH_PT     = 0.025
const B410         = 180.0 / (3300.0 * 3.0e-5 * 1000.0)   # 1.818
const B660         = 300.0 / (3300.0 * 3.0e-5 * 1000.0)   # 3.030
const LATENT_L     = 0.11
# Latent heat: `per_phase` (default) derives one coefficient per transition from
# its Clapeyron slope and density jump (docs/dev/PHYSICS.md §7.8), which is the
# physics: ASPECT's `+T dS_A DGamma/Dt` with `dS_A = gamma drho/rho^2`, i.e.
#   latent_410 = -0.059364 (negative: the 410 HEATS a downwelling),
#   latent_660 = +0.130596 (positive: the 660 COOLS it).
# `global` reproduces the pre-2026-11-08 runs bit for bit (one `latent_L = 0.11`
# for both transitions, which is the wrong sign at the 410); the committed
# reference run in out_julia/case_realistic was made with it, so a restart of
# that checkpoint needs CASE_LATENT=global (the checkpoint stores latent_mode
# and a mismatch raises).
const LATENT_MODE  = Symbol(get(ENV, "CASE_LATENT", "per_phase"))
const LATENT_SCALE = (rho_ref=3300.0, cp=1000.0, dT=1000.0, g=9.81, d=2.9e6,
                      T_surface=0.5)   # ASPECT scale set, §14.2
const NG           = 256        # frame grid (x and y)
const AMR_EVERY    = 5
# Coarsening knob of the card.  The default is the balanced policy's production
# value (docs/dev/CASE_REALISTIC.md §1.5): `coarsen_frac` is the per-cycle cell
# cap of the budget exchange
# (`capcells = round(coarsen_frac*n0)`, docs/dev/AMR.md §3.3).  The historical
# A/B pair used `0.005`.
const COARSEN_FRAC = parse(Float64, get(ENV, "CASE_COARSEN", "0.05"))
const EV_COEF      = parse(Float64, get(ENV, "CASE_EV", "0.11"))
# Wall refinement (`refine_wall_to_min!` after every AMR cycle, src/
# src/driver/run.jl:401/1089): force every leaf within `wall_strip` of the
# bottom/top wall, and the wall rows themselves, back to at least
# `wall_min_level`.  The default is the production rule (5 / 0.1): without it
# the top wall stays at level 3/4 while the interior is L5/6, and that
# under-resolved wall is what puts the consistent boundary flux 14 % above the
# pointwise Nu_top (docs/dev/CASE_REALISTIC.md §7.2, §14; docs/dev/PHYSICS.md §7.3.5).
# `CASE_WALLMIN=0` turns it off for a historical-arm replay; the wall keys are
# part of `run_config`, so an existing checkpoint written with the old default
# needs the same explicit value (or a `wallrefine` seed).
const WALL_MIN     = parse(Int, get(ENV, "CASE_WALLMIN", "5"))
const WALL_STRIP   = parse(Float64, get(ENV, "CASE_WALLSTRIP", "0.1"))
# AMR cycle (docs/dev/AMR.md §3-§4): the balanced policy is the only adaptive
# policy (the classic fraction+protect+hard-budget path was removed), so a new
# run always gets forward coarsening, budget exchange, hysteresis and the 2:1
# gate.  The hysteresis cooldown is `CASE_HYST` AMR cycles (default 1).
const AMR_HYST     = parse(Int, get(ENV, "CASE_HYST", "1"))
# :theta (default) or :velocity; see docs/dev/PHASE1.md 1.1
const AMR_IND      = Symbol(get(ENV, "CASE_INDICATOR", "theta"))
# Per-wall velocity condition, `(xminus, xplus, yminus, yplus)`.  `CASE_WALL_BC` is
# a comma-separated list of `same|free|rigid`; `same` follows `CASE_BC` (the card's
# single `bc`).  Empty means all `:same`, i.e. the historical behaviour.
const WALL_BC     = begin
    raw = strip(get(ENV, "CASE_WALL_BC", ""))
    if isempty(raw)
        (:same, :same, :same, :same)
    else
        parts = Tuple(Symbol(strip(v)) for v in split(raw, ','))
        length(parts) == 4 ||
            error("CASE_WALL_BC needs four comma-separated entries " *
                  "(xminus,xplus,yminus,yplus), got \"$raw\"")
        all(w -> w in (:same, :free, :rigid), parts) ||
            error("CASE_WALL_BC entries must be same, free or rigid, got \"$raw\"")
        parts
    end
end
# x-mirror closure of the AMR passes (`symmetry=:x`).  The default is `none`,
# the production card (docs/dev/CASE_REALISTIC.md §1.5): `none` switches the
# pair-strict mirror rule off.  Under `:x` a quartet is merged only together
# with its mirror quartet and a leaf is split only together with its mirror
# partner, and the measured production field is not mirror-symmetric, so the
# rule is what limits the coarsening supply -- 241 eligible quartets -> 4, a
# 60x cap on the exchange (docs/dev/BUG_LEDGER.md §5.6/§6).  `symmetry` is part of
# `run_config` and IS compared on restart (the guard exists so a resume cannot
# silently change the AMR/mirror rule), so an `x` restart of a checkpoint
# written under this default has to set `CASE_SYMMETRY=x`, and a `none` run
# over an `x` checkpoint starts from a `symseed` copy -- the same mechanism the
# wall keys use in `wall_checkpoint`.
const SYMMETRY     = Symbol(get(ENV, "CASE_SYMMETRY", "none"))
# Coarsening pre-check (`coarsening_allowed`, docs/dev/AMR.md,
# docs/dev/AMR.md §4).  Only `:balanced` uses it.  The default is ON (1), the
# production card: the gate is what stops the merge/refine recirculation
# (docs/dev/BUG_LEDGER.md §5.7), and a new arm should have it.  It is part of
# `run_config`, so resuming a checkpoint written *without* the gate needs
# `CASE_AMR_GATE=0` explicitly: an old checkpoint does not carry the key, and
# the guard iterates over the checkpoint's keys, so an absent key is not
# compared -- the explicit pin is what keeps that resume honest.
const AMR_GATE     = parse(Int, get(ENV, "CASE_AMR_GATE", "1"))
# Opt-in exact-residual column (`resid_A`): the scheme's own identity (ID1)
# evaluated for the quoted pointwise Nu pair, per unit area (docs/
# NUSSELT_AUDIT.md §2.3/§3.2).  OFF by default, so every existing run keeps its
# 18/19-column history byte for byte.  The extra column changes the layout, so
# main() refuses it on an output directory that already holds output.
const EXTDIAG      = lowercase(strip(get(ENV, "CASE_EXTDIAG", "0"))) in
                     ("1", "true", "yes", "on")
# Linear solver: the library default is `:auto` since 2026-11-13; this card
# passes `:auto` explicitly, so its behaviour is unchanged.  The path is
# resolved before assembly from `saddle_dofs = 2*Nu + Np` and the sampled eta
# contrast (`semquad_choose_linear`).  Both production states are above the
# 20_000 rule and run the matrix-free MG with `mg_threads=min(8,nthreads)`:
# 78 720 selector dofs on F (8 188 leaves),
# 95 308 on P1 (10 000 leaves).  On read-only copies with the driver's kwargs
# (`stokes_tol=1e-3`) the library-default `mg_tol=1e-10` reaches 92 iterations /
# relres 7.8e-11 on F and 108 / 8.7e-11 on P1 with zero `LAST_AUTO_FALLBACK`
# events; warm 1.61 s and 2.68 s (best of three) against 3.47 s and 3.92 s for
# `linear=:direct` with `stokes_scaling=:equilibrated`.  That tolerance sits
# seven orders below the `1e-3` residual gate, so the card passes
# `MG_TOL=1e-8` explicitly (docs/dev/CASE_REALISTIC.md §1.6): F/P1/C take
# 81/91/67 iterations, true residual 8.6e-9/9.4e-9/7.0e-9 and 1.41/2.21/1.22 s
# warm, and the fields move by at most 2.7e-8 (|du|/|u|) and 7.9e-9
# (gauge-removed |dp|/|p|) against the 1e-10 run.  The benchmark's `mg_tol=1e-6`
# is cheaper still (71/65/57 iterations, 1.26/1.70/1.08 s) but moves P1's
# gauge-removed pressure by 1.28e-6, above the documented 8.8e-7 class, so it is
# not used here.
#
# History: at `max_leaves=8192` (7 498 leaves, ~72k selector dofs) the old
# rule did switch to MG and failed -- MG-FGMRES relres=4.33, driver abort
# `Stokes residual 7.28 exceeds stokes_tol=1.0e-3` -- because the matrix-free
# smoother clipped `Cu' diag(K)`, which is not `diag(Cu' K Cu)` on hanging
# rows, and the BFBT diagonal was the lumped approximation.  The exact
# constrained Jacobi diagonal (9250a3c) removed the defect behind that failure;
# the exact leaf-local BFBT diagonal (503fd87) then cut the iteration count and
# improved the accuracy.  The 8192-leaf C-arm checkpoint (8 260 leaves,
# 79 308 selector dofs) now resolves to MG and converges: 90 iterations,
# relres 8.1e-11, no fallback at the old default, 67 iterations / 7.0e-9 at the
# card's 1e-8.  The card keeps `auto`, so the restart can use it;
# `CASE_LINEAR=direct` still forces the bitwise direct path.
const LIN_SOLVE    = Symbol(get(ENV, "CASE_LINEAR", "auto"))
# Stokes acceptance gate of the driver: `run_semquad_simulation` forwards it as
# both `auto_fallback_tol` (the auto-MG safety net) and `stokes_accept_tol` (the
# direct shift ladder), and fails the step above it (src/driver/run.jl:944).
const STOKES_TOL   = 1.0e-3
# The MG outer tolerance, passed explicitly so a future change of the library
# `mg_tol` default cannot silently over-solve the card (that default is
# deliberately unchanged).  It is `stokes_tol/1e5` = 1e-8: five orders below the gate, and
# the measured movement against the 1e-10 solve is <=2.7e-8, well inside the
# documented MG accuracy class (|du|/|u| <= 2.1e-6, gauge-removed
# |dp|/|p| <= 8.8e-7; docs/dev/CASE_REALISTIC.md §1.6).
const MG_TOL       = 1.0e-8

# Solver-side equilibration of the direct saddle (docs/dev/BUG_LEDGER.md).
# `none` is the historical path bit for bit (and the default here, so every
# existing card is unchanged); `equilibrated` forwards
# `run_semquad_simulation(stokes_scaling=:equilibrated)`, which engages only on
# a solve whose historical unscaled unshifted answer fails its health check --
# the failing G step-7641 class.  The run_config/restart-guard behaviour (old
# checkpoint = none) is described in the header comment.
const STOKES_SCALING = Symbol(get(ENV, "CASE_STOKES_SCALING", "none"))
const FRAME_STRIDE = 5

const OUT = get(ENV, "CASE_OUT",
                normpath(joinpath(@__DIR__, "..", "out_julia",
                                  "case_realistic")))
const THE_STEPS = parse(Int, get(ENV, "CASE_STEPS", "0"))   # 0 = mode default
const FRAME_EVERY = parse(Int, get(ENV, "CASE_FRAME", "0")) # 0 = default

# --------------------------------------------------------------- geometry ----
function make_initial_tree()
    qt = Quadtree(MAXLEV; Lx=LX, cells_x=CX, cells_y=CY)
    for _ in 1:3, lf in collect(qt.leaves)       # uniform level 3 = 640 leaves
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function phases()
    ph = [PhaseTransition(0.87, WIDTH_PT, SLOPE410, 0.5, 180.0, B410),
          PhaseTransition(0.78, WIDTH_PT, SLOPE660, 0.5, 300.0, B660)]
    # `per_phase`: fill `latent` from the slope/drho of each transition at its
    # own reference temperature (docs/dev/PHYSICS.md §7.8).  `global` leaves the
    # coefficients unset; `Params.latent_mode` then selects `latent_L`.
    return LATENT_MODE === :per_phase ?
           per_phase_latent(Params(Ra=RA, H=HHEAT), ph; LATENT_SCALE...) : ph
end

function make_params()
    return Params(Ra=RA, H=HHEAT,
                  eta_law=:composite, comp_rule=:isostress,
                  comp_Ad=COMP_AD, comp_As=COMP_AS,
                  comp_Ed=COMP_ED, comp_Es=COMP_ES,
                  comp_Vd=0.0, comp_Vs=0.0, comp_n=COMP_N,
                  comp_Tcut=0.0, comp_eps_min=COMP_EPS_MIN,
                  eta_T0=ETA_T0, eta_min=ETA_MIN, eta_max=ETA_MAX,
                  bc=:free, wall_bc=WALL_BC,
                  latent_L=LATENT_L, latent_mode=LATENT_MODE,
                  # everything below is the round-1 "off" default, written out
                  # so the card is explicit:
                  yield_c=Inf, yield_phi=0.0, yield_taumax=Inf,
                  yield_c_grad=0.0, yield_mu_grad=0.0,
                  weak_f=1.0, weak_eps_ref=0.0,
                  rho_heating=false, rho_cp=false, Di=0.0,
                  heating_adiabatic=false, heating_dissipation=false,
                  refstate=nothing, deviatoric=false)
end

# theta = T - Tbar must vanish on the top/bottom walls (Tbar(0)=1, Tbar(1)=0)
# and be x-symmetric about Lx/2 (symmetry=:x).
seed(x, y) = 0.05 * sinpi(y) * cospi(2.0 * x / LX) +
             0.02 * sinpi(2.0 * y) * cospi(4.0 * x / LX)

# ------------------------------------------------------------ frame writer ----
mutable struct FrameWriter
    dir::String
    meta::IO
    ng::Int
    xs::Vector{Float64}
    ys::Vector{Float64}
    t0::Float64
    nframe::Int
    nu::Bool
end

function FrameWriter(root::String; ng::Int=NG)
    dir = joinpath(root, "frames")
    mkpath(dir)
    metapath = joinpath(dir, "frame_meta.csv")
    fresh = !isfile(metapath) || filesize(metapath) == 0
    io = open(metapath, "a")
    if fresh
        println(io, "frame,step,t,dt,N,leaves,nlev,Tmin,Tmax,Tmean," *
                    "Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac," *
                    "wall_s,ng")
    end
    xs = collect(range(0.0, LX; length=ng))
    ys = collect(range(0.0, 1.0; length=ng))
    return FrameWriter(dir, io, ng, xs, ys, time(), 0, true)
end

function write_frame!(fw::FrameWriter, st, pr)
    s = st.sq
    L = Set(s.leaves)
    ng = fw.ng
    T = Matrix{Float64}(undef, ng, ng)
    for j in 1:ng, i in 1:ng
        T[j, i] = Tbar(fw.ys[j], pr.H) +
                  QuaST._semquad_eval_at(s, st.theta, fw.xs[i], fw.ys[j], L)
    end
    fn = @sprintf("frame_%05d_T.bin", st.step)
    open(joinpath(fw.dir, fn), "w") do io
        # x varies fastest -> numpy reshape(ng,ng) is A[z,x]; imshow(origin=lower)
        write(io, vec(permutedims(T)))
    end
    open(joinpath(fw.dir, @sprintf("frame_%05d_mesh.csv", st.step)), "w") do io
        for lf in sort(collect(st.qt.leaves))
            h = LX / (CX * 2.0^lf[1])            # square leaves: Lx/Cx = 1/Cy
            @printf(io, "%d,%.17g,%.17g,%.17g\n", lf[1], lf[2] * h, lf[3] * h, h)
        end
    end
    # diagnostics on the same state the frame shows (cheap: one O(N) sweep each)
    Nut, Nub = semquad_nu_walls(s, st.qt, st.theta, pr)
    Vr = semquad_vrms(s, st.qt, st.ux, st.uz)
    Tmean = semquad_mean_T(s, st.qt, st.theta, pr)
    # viscosity census at the velocity nodes (the "is eta clamped everywhere"
    # health check of the brief)
    ed = semquad_strain_rate_field(s, st.ux, st.uz)
    emin = Inf; emax = -Inf; nclip = 0
    # log10(eta) per velocity node, kept for the renderer: the viscosity field is
    # the one thing in this case that the temperature does not show
    logeta = Vector{Float32}(undef, length(ed))
    for g in eachindex(ed)
        y = s.coords[g][2]
        Tn = Tbar(y, pr.H) + st.theta[g]
        v, cl = eta_composite_at(Tn, ed[g], y, pr)
        logeta[g] = Float32(log10(v))
        emin = min(emin, v); emax = max(emax, v)
        nclip += cl ? 1 : 0
    end
    open(joinpath(fw.dir, @sprintf("frame_%05d_eta.bin", st.step)), "w") do io
        write(io, logeta)
    end
    open(joinpath(fw.dir, @sprintf("frame_%05d_speed.bin", st.step)), "w") do io
        speed = Vector{Float32}(undef, length(ed))
        for g in eachindex(ed)
            speed[g] = Float32(hypot(st.ux[g], st.uz[g]))
        end
        write(io, speed)
    end
    open(joinpath(fw.dir, @sprintf("frame_%05d_nodes.bin", st.step)), "w") do io
        nodes = Vector{Float32}(undef, 2 * length(ed))
        for g in eachindex(ed)
            nodes[2 * g - 1] = Float32(s.coords[g][1])
            nodes[2 * g] = Float32(s.coords[g][2])
        end
        write(io, nodes)
    end
    nlev = length(unique(lf[1] for lf in s.leaves))
    umax = maximum(hypot.(st.ux, st.uz))
    @printf(fw.meta, "%d,%d,%.9e,%.6e,%d,%d,%d,%.6e,%.6e,%.6e,%.9e,%.9e,%.9e,%.6e,%.6e,%.6e,%.6f,%.3f,%d\n", fw.nframe, st.step, st.t, st.dt, s.N, length(s.leaves), nlev, minimum(T), maximum(T), Tmean, Nut, Nub, Vr, umax, emin, emax, nclip / length(ed), time() - fw.t0, ng)
    flush(fw.meta)
    fw.nframe += 1
    return nothing
end

# ------------------------------------------------------------------ driver ----
function run_case(mode::String; restart_from::String="", amr_every::Int=AMR_EVERY,
                  maxsteps::Int=10000, checkpoint::String="")
    pr = make_params()
    ph = phases()
    qt = make_initial_tree()
    ss = make_semstokes(qt, PORD)
    theta0 = [seed(x, y) for (x, y) in ss.su.coords]
    if !isempty(restart_from)
        # theta comes from the checkpoint; run_semquad_simulation replaces it.
        theta0 = zeros(ss.su.N)
    end
    fw = FrameWriter(OUT)
    stride = FRAME_EVERY > 0 ? FRAME_EVERY : FRAME_STRIDE
    cb = function (st)
        # `amr_every` zeroes ux/uz right before the callback, so a frame written
        # on an AMR step would carry a zero velocity in its Nu/Vrms columns:
        # offset the cadence by one step past every AMR cycle.
        st.step % stride == 1 && write_frame!(fw, st, pr)
        if st.step % 100 == 0
            @printf("  [case] step=%d t=%.4e leaves=%d N=%d auto_fallbacks=%d\n",
                    st.step, st.t, length(st.qt.leaves), st.sq.N,
                    LAST_AUTO_FALLBACK[].count)
            flush(stdout)
        end
    end
    meta = Dict{String,Any}(
        "case" => "realistic_2d_round1", "Ra" => RA, "H" => HHEAT,
        "comp_Ad" => COMP_AD, "comp_As" => COMP_AS, "comp_Ed" => COMP_ED,
        "comp_Es" => COMP_ES, "comp_n" => COMP_N, "eta_T0" => ETA_T0,
        "eta_min" => ETA_MIN, "eta_max" => ETA_MAX, "latent_L" => LATENT_L,
        "latent_mode" => string(LATENT_MODE),
        "latent410" => ph[1].latent, "latent660" => ph[2].latent,
        "B410" => B410, "B660" => B660, "p" => PORD, "max_level" => MAXLEV,
        "max_leaves" => MAXLEAVES, "mode" => mode, "linear" => string(LIN_SOLVE),
        "wall_min_level" => WALL_MIN, "wall_strip" => WALL_STRIP,
        "amr_policy" => "balanced", "amr_hysteresis" => AMR_HYST,
        "amr_coarsen_gate" => AMR_GATE,
        "edot_ref" => 1.65e3, "cfl" => 0.5)
    # Only an opt-in run records it, so the checkpoints of the default card stay
    # byte-identical (the key is diagnostic, not part of the restart guard).
    EXTDIAG && (meta["extdiag"] = true)
    @printf("[case] mode=%s Ra=%.1e H=%.1f p=%d max_level=%d max_leaves=%d amr_every=%d steps=%d coarsen=%.4g linear=%s stokes_tol=%.1e mg_tol=%.1e wall_min=%d wall_strip=%.4g ev=%.3g hyst=%d symmetry=%s gate=%d stokes_scaling=%s indicator=%s wall_bc=%s\n", mode, RA, HHEAT, PORD, MAXLEV, MAXLEAVES, amr_every, maxsteps, COARSEN_FRAC, LIN_SOLVE, STOKES_TOL, MG_TOL, WALL_MIN, WALL_STRIP, EV_COEF, AMR_HYST, SYMMETRY, AMR_GATE, STOKES_SCALING, AMR_IND, join(string.(WALL_BC), ","))
    @printf("[case] comp_Ad=%.10e comp_As=%.10e Ed=%.1f Es=%.1f n=%.1f T0=%.2f eta in [%.0e,%.0e]\n", COMP_AD, COMP_AS, COMP_ED, COMP_ES, COMP_N, ETA_T0, ETA_MIN, ETA_MAX)
    @printf("[case] phases: z0=0.87 w=%.3f slope=%+.3f B=%.4f | z0=0.78 w=%.3f slope=%+.3f B=%.4f | latent_L=%.3f latent_mode=%s\n", WIDTH_PT, SLOPE410, B410,
            WIDTH_PT, SLOPE660, B660, LATENT_L, LATENT_MODE)
    @printf("[case] latent coefficients: 410 %s 660 %s (negative = heats a downwelling)\n",
            ph[1].latent === nothing ? "unset" : @sprintf("%+.6f", ph[1].latent),
            ph[2].latent === nothing ? "unset" : @sprintf("%+.6f", ph[2].latent))
    flush(stdout)
    t0 = time()
    res = run_semquad_simulation(qt, pr, theta0;
        p=PORD, tstop=Inf, maxsteps=maxsteps,
        dt0=DT0, dtmax=DTMAX, dtmin=DTMIN, cfl=0.5, dt_growth=1.05,
        maxpic=1, pic_tol=1.0e-6, pic_mode=:picard,
        amr_every=amr_every,
        refine_frac=0.2, coarsen_frac=COARSEN_FRAC,
        max_leaves=MAXLEAVES, max_level=MAXLEV, symmetry=SYMMETRY,
        wall_min_level=WALL_MIN, wall_strip=WALL_STRIP,
        amr_hysteresis=AMR_HYST, amr_indicator=AMR_IND,
        amr_coarsen_gate=AMR_GATE == 1,
        transfer_mode=:nodal,
        advection=:skew, ev=EV_COEF, ev_cmax=0.052, ev_alpha=2,
        linear=LIN_SOLVE, stokes_tol=STOKES_TOL, mg_tol=MG_TOL,
        stokes_scaling=STOKES_SCALING,
        mass_form=:boussinesq, phases=ph, latent_scheme=:implicit,
        callback=cb, verbose=true, profile_steps=false,
        restart_from=restart_from, checkpoint_path=checkpoint,
        checkpoint_every=200, checkpoint_meta=meta, extdiag=EXTDIAG)
    wall = time() - t0
    # A resume whose `maxsteps` does not exceed the checkpoint's step number
    # runs zero steps and `res.history` is empty; the final-frame and final-
    # summary paths below index `res.history[end]`, which used to throw a
    # BoundsError.  Fail informatively instead (and write nothing, so the
    # checkpoint/frame stream is untouched).
    if isempty(res.history)
        close(fw.meta)
        @printf("[case] DONE mode=%s steps=%d (no steps were run: maxsteps=%d <= checkpoint step); nothing written\n",
                mode, res.steps, maxsteps)
        return res
    end
    # final frame + history dump
    write_frame!(fw, (step=res.steps, t=res.t, dt=res.history[end][2],
                      theta=res.theta, ux=res.ux, uz=res.uz, sq=res.ss.su,
                      qt=res.qt), pr)
    close(fw.meta)
    # Column 13 `closure` is the **historical** formula, not a flux error: by
    # `(DEC)` of docs/dev/VALIDATION.md §4 it equals `-dH/dt` plus three
    # bookkeeping terms at a transient state and does not close for any Nu
    # definition (docs/dev/VALIDATION.md §3).  Column 19 `closure_nolatent` is
    # its latent-free form.  The opt-in `resid_A` (CASE_EXTDIAG=1) is the exact
    # residual of the quoted pointwise pair against the scheme's identity
    # (`(ID1)`, NUSSELT_AUDIT §2.3/§3.2).  The extra columns are named by the
    # driver (`res.history_extra`), so the header cannot disagree with the row
    # layout: the default run writes exactly 18 columns and its bitwise hash is
    # unchanged, a latent run adds `closure_nolatent` (19), and
    # `CASE_EXTDIAG=1` adds `resid_A`.
    extra = res.history_extra
    ncol = isempty(res.history) ? 18 + length(extra) : length(res.history[1])
    ncol == 18 + length(extra) ||
        error("history row has $ncol columns but the driver named " *
              "$(18 + length(extra)) appended columns; the layout changed " *
              "under the writer")
    open(joinpath(OUT, "history.csv"), "w") do io
        print(io, "t,dt,Nu_top,Nu_bot,Vrms,meanT,N,leaves,div_max,hmin,umax,cfl,closure,usrms_surf,umax_surf,W,U,balance")
        isempty(extra) || print(io, ",", join(extra, ","))
        println(io)
        for r in res.history
            isempty(r) && continue
            @printf(io, "%.9e,%.6e,%.9e,%.9e,%.9e,%.9e,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.9e,%.9e,%.6e", r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18])
            for k in 19:ncol
                @printf(io, ",%.9e", r[k])
            end
            println(io)
        end
    end
    # `res.steps` is the absolute step number, so a resumed run must not divide
    # the wall time by it (a 200-step resume of step 18800 would print
    # 0.06 s/step); `res.history` holds one row per step *actually run*.
    nrun = max(length(res.history), 1)
    @printf("[case] DONE mode=%s steps=%d t=%.6e wall=%.1f s (%.3f s/step) linear=%s scaled=%d auto_fallbacks=%d\n",
            mode, res.steps, res.t, wall, wall / nrun, LIN_SOLVE,
            LAST_SEMSTOKES_SCALED[], LAST_AUTO_FALLBACK[].count)
    last = res.history[end]
    @printf("[case] final: Nu_top=%.6f Nu_bot=%.6f Vrms=%.6e meanT=%.6f closure=%.3e leaves=%d N=%d\n",
            last[3], last[4], last[5], last[6], last[13], last[8], last[7])
    return res
end

# ------------------------------------------------------------------- freeze ----
"""Rewrite a checkpoint's `amr_every` to 0 so it can be restarted frozen.

`amr_every` is part of the driver's `run_config` (it changes the trajectory) and
is compared on restart, so a frozen restart needs the stored value to be 0 as
well.  Only that one string is changed; the state (tree/theta/ux/uz/pv/dt/t/
step, and the optional `:balanced` cooldown) is copied verbatim.  No solver code
is touched — this uses the public `load_semquad_state` / `save_semquad_state`
API."""
function freeze_checkpoint(src::String, dst::String)
    st = load_semquad_state(src)
    cfg = copy(st.config)
    cfg["amr_every"] = "0.0"
    save_semquad_state(dst, st.qt, st.theta, st.theta_prev, st.ux, st.uz, st.pv;
                       t=st.t, dt=st.dt, dt_prev=st.dt_prev, step=st.step,
                       p=st.p, config=cfg, meta=Dict("frozen_from" => src),
                       cooldown=st.cooldown,
                       velocity_stale=st.velocity_stale)
    @printf("[case] froze %s -> %s (leaves=%d t=%.6e step=%d)\n", src, dst,
            length(st.qt.leaves), st.t, st.step)
    return dst
end

# ------------------------------------------------------- wall-refinement seed ----
"""Build a wall-refined seed checkpoint from `src`, at the card's leaf budget.

Two things have to be done here, both with the public API only:

1. **The restart guard.**  `wall_min_level`/`wall_strip` are `run_config` keys
   and are compared on restart, so a checkpoint written with the walls
   unrefined cannot be resumed with `CASE_WALLMIN > 0`:

       restart configuration mismatch ...: wall_min_level: checkpoint=0.0, current=5.0

   The guard is not bypassed: the seed carries the new values (only those two
   strings are rewritten) and the source is never touched.
2. **The leaf budget.**  At `max_leaves = 8192` a wall refinement on the B2 tree
   is a no-op, measured: `refine_wall_to_min!` takes 8188 -> 9682 leaves, then
   the driver's hard budget pass merges the new wall children straight back
   (their indicator is zero, and only the *pre-existing* wall leaves are in its
   `protect` set) -> 8314 leaves, top row back at level 3/4.  That is the
   budget's lowest-error merge rule acting on the refinement it was supposed to
   keep, not a wall-criterion error.  Since `src` is not touched here, the
   seed does the redistribution explicitly instead: refine the wall strips, put
   every leaf of the strip (the new children included) in the protect set, and
   merge the lowest-error **interior** quartets back down to `max_leaves`.  The
   card's budget is then respected exactly and every AMR cycle that follows
   protects the (already refined) wall strip.

The state (tree, theta, theta_prev, ux, uz, pv, t, dt, dt_prev, step, p) is
transferred to the new tree; dt, t and the step are copied verbatim.  Nothing
in `src` is touched."""
function wall_checkpoint(src::String; wall_min_level::Int=WALL_MIN,
                         wall_strip::Float64=WALL_STRIP,
                         max_leaves::Int=MAXLEAVES,
                         ev::Float64=EV_COEF)
    wall_min_level >= 0 ||
        throw(ArgumentError("CASE_WALLMIN must be >= 0, got $wall_min_level"))
    0.0 < wall_strip < 0.5 ||
        throw(ArgumentError("CASE_WALLSTRIP must be in (0, 0.5), got $wall_strip"))
    st = load_semquad_state(src)
    cfg = copy(st.config)
    old_min = get(cfg, "wall_min_level", "unset")
    old_strip = get(cfg, "wall_strip", "unset")
    old_cap = get(cfg, "max_leaves", "unset")
    cfg["wall_min_level"] = string(Float64(wall_min_level))
    cfg["wall_strip"] = string(Float64(wall_strip))
    # the leaf budget is part of run_config and the restart guard compares it;
    # a wall-refined seed that is meant for a larger budget must say so, or the
    # resume is refused with "max_leaves: checkpoint=8192.0, current=10000.0"
    cfg["max_leaves"] = string(Float64(max_leaves))
    # `ev` likewise: the restart guard compares it, and an EV sensitivity test
    # must be able to start from a checkpoint written with another value
    old_ev = get(cfg, "ev", "unset")
    cfg["ev"] = string(Float64(ev))
    # `symmetry` is a `run_config` key too (the guard must not let a resume switch
    # the x-mirror rule silently), and a wall seed prepared for a
    # `CASE_SYMMETRY=none` run has to carry that value.  Rewriting the string is
    # the only thing that changes; the tree work below then follows `SYMMETRY`
    # (no mirror repairs when the mirror rule is off).
    old_sym = get(cfg, "symmetry", "unset")
    cfg["symmetry"] = string(SYMMETRY)
    qt = st.qt
    ss_old = make_semstokes(qt, st.p)
    leaves0 = length(qt.leaves)
    # the gradient-jump indicator of the *current* tree: the interior quartets
    # that get merged are the lowest-error ones, exactly the budget's rule
    eta = semquad_gradient_jump(ss_old.su, qt, st.theta)
    nref = refine_wall_to_min!(qt, wall_min_level; strip=wall_strip,
                               symmetry=SYMMETRY)
    leaves_wall = length(qt.leaves)
    # The source checkpoint is not mirror-closed (120 non-mirrored leaves at
    # step 6130, §15), so close it here: otherwise the x-mirror invariant is
    # inherited by every following cycle and the "symmetric mesh" claim stays
    # false for the whole run.  `mirror_close!` only refines, and the paired
    # budget merge below is what pays for it.  Under `CASE_SYMMETRY=none` there
    # is no invariant to repair and the pass is skipped.
    defect0 = mirror_defect(qt)
    closed, closed_ok = SYMMETRY === :x ? mirror_close!(qt; max_level=MAXLEV) :
                                          (0, true)
    defect1 = mirror_defect(qt)
    closed_ok || @warn "wall seed: mirror_close! could not close the tree " *
                       "(max_level=$(MAXLEV)): $defect1 non-mirrored leaves"
    rows = wall_rows(qt)
    leaves1 = length(qt.leaves)
    # protect every leaf of the wall strip, the new children included
    protect = Set{NTuple{3,Int}}()
    for lf in qt.leaves
        h = 1.0 / (qt.ny0 * (1 << lf[1]))        # = leaf_hy(qt, lf[1])
        zc = (lf[3] + 0.5) * h
        if lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1 ||
           zc <= wall_strip || zc >= 1.0 - wall_strip
            push!(protect, lf)
        end
    end
    merged, budget_ok = semquad_enforce_leaf_budget!(qt, copy(eta);
                            max_leaves=max_leaves, protect=protect,
                            symmetry=SYMMETRY)
    budget_ok || @warn "wall seed: leaf budget not reachable: " *
                       "$(length(qt.leaves)) leaves > max_leaves=$max_leaves"
    # transfer the state onto the new tree (theta/velocity on the scalar space,
    # the pressure on its own space; dt/t/step are copied verbatim)
    ss = make_semstokes(qt, st.p)
    theta = transfer_semquad(ss_old.su, st.theta, ss.su; mode=:nodal,
                             zero_walls=true, qt_new=qt)
    theta_prev = transfer_semquad(ss_old.su, st.theta_prev, ss.su; mode=:nodal,
                                  zero_walls=true, qt_new=qt)
    ux = transfer_semquad(ss_old.su, st.ux, ss.su; mode=:nodal,
                          zero_walls=false, qt_new=qt)
    uz = transfer_semquad(ss_old.su, st.uz, ss.su; mode=:nodal,
                          zero_walls=false, qt_new=qt)
    pv = transfer_semquad(ss_old.sp, st.pv, ss.sp; mode=:nodal,
                          zero_walls=false, qt_new=qt)
    dst = joinpath(OUT, "seed",
                   @sprintf("wall_min%d_strip%g_from_step%06d.h5",
                            wall_min_level, wall_strip, st.step))
    mkpath(dirname(dst))
    # The `:balanced` cooldown is deliberately not copied: its keys refer to the
    # source leaf set, and this seed changes the tree (wall refinement, mirror
    # closure and budget merges).  The new branch starts with an empty cooldown,
    # exactly like a fresh run; a source checkpoint without cooldown data would
    # load an empty one anyway.
    save_semquad_state(dst, qt, theta, theta_prev, ux, uz, pv;
                       t=st.t, dt=st.dt, dt_prev=st.dt_prev, step=st.step,
                       p=st.p, config=cfg,
                       meta=Dict("wall_retuned_from" => src,
                                 "wall_seed_refined" => nref,
                                 "wall_seed_merged" => merged))
    @printf("[case] wall seed: %s (step %d) -> %s\n", src, st.step, dst)
    @printf("[case] wall seed: wall_min_level %s -> %.1f, wall_strip %s -> %.4g, max_leaves %s -> %d, ev %s -> %.3g, symmetry %s -> %s (config keys rewritten: these five)\n",
            old_min, Float64(wall_min_level), old_strip, wall_strip, old_cap,
            max_leaves, old_ev, ev, old_sym, SYMMETRY)
    @printf("[case] wall seed: %d -> %d leaves by the wall rule (refined %d), %d mirror repairs (nonmirrored %d -> %d) -> %d leaves, then %d interior quartets merged -> %d leaves (budget %d, reached=%s, nonmirrored=%d)\n",
            leaves0, leaves_wall, nref, closed, defect0, defect1, leaves1,
            merged, length(qt.leaves), max_leaves, budget_ok,
            mirror_defect(qt))
    @printf("[case] wall seed: bottom row %s | top row %s\n", rows.bottom, rows.top)
    return dst
end

"""Level census of the bottom/top wall rows, e.g. `L5=160` (for the log)."""
function wall_rows(qt::Quadtree)
    top = [lf[1] for lf in qt.leaves if lf[3] == qt.ny0 * (1 << lf[1]) - 1]
    bot = [lf[1] for lf in qt.leaves if lf[3] == 0]
    pk(v) = isempty(v) ? "-" :
            join(["L$(l)=$(count(==(l), v))" for l in sort(unique(v))], " ")
    return (bottom=pk(bot), top=pk(top))
end

# ------------------------------------------------------- symmetry seed ----
"""Copy a checkpoint with its stored `symmetry` rewritten to `SYMMETRY`.

`symmetry` is part of the driver's `run_config` and is compared on restart (the
guard exists so a resume cannot silently change the AMR/mirror rule), so a
`CASE_SYMMETRY=none` run cannot start from a checkpoint written with `:x`:

    restart configuration mismatch ...: symmetry: checkpoint=x, current=none

The guard is not bypassed here either: the seed carries the new string and the
source is never touched.  Unlike `wall_checkpoint` **nothing else is changed** --
the tree, theta/theta_prev, ux/uz, pv, t, dt, dt_prev, step and maxlevel are
copied verbatim, and so is the optional `:balanced` AMR cooldown (the tree is
unchanged, so its leaf keys remain valid).  A :x tree resumed as `:none` still
has `mirror_defect == 0`; it simply stops being *repaired*, and from the first
cycle on only the actual field decides the leaf set."""
function symmetry_checkpoint(src::String, dst::String)
    st = load_semquad_state(src)
    cfg = copy(st.config)
    old = get(cfg, "symmetry", "unset")
    old == string(SYMMETRY) && @warn "symmetry seed: the checkpoint already " *
                                     "says symmetry=$old, the copy is a no-op"
    cfg["symmetry"] = string(SYMMETRY)
    mkpath(dirname(dst))
    save_semquad_state(dst, st.qt, st.theta, st.theta_prev, st.ux, st.uz, st.pv;
                       t=st.t, dt=st.dt, dt_prev=st.dt_prev, step=st.step,
                       p=st.p, config=cfg, ev=st.ev,
                       cooldown=st.cooldown,
                       velocity_stale=st.velocity_stale,
                       meta=Dict("symmetry_retuned_from" => src,
                                 "symmetry_from" => old))
    @printf("[case] symmetry seed: %s (step %d, %d leaves, mirror_defect=%d) -> %s (symmetry %s -> %s)\n",
            src, st.step, length(st.qt.leaves), mirror_defect(st.qt), dst,
            old, SYMMETRY)
    return dst
end

# --------------------------------------------------------------------- main ----
function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "short"
    AMR_GATE in (0, 1) ||
        error("CASE_AMR_GATE must be 0 or 1, got $AMR_GATE")
    SYMMETRY in (:none, :x) ||
        error("CASE_SYMMETRY must be none or x, got $SYMMETRY")
    STOKES_SCALING in (:none, :equilibrated) ||
        error("CASE_STOKES_SCALING must be none or equilibrated, got " *
              "$STOKES_SCALING")
    if EXTDIAG
        # The extra `resid_A` column changes history.csv's layout, and every
        # reader of an existing directory (watchdog, live_render, the analysis
        # snippets of docs/dev/CASE_REALISTIC.md §7.2) indexes columns by name.  A
        # mid-run layout change would make the new rows silently disagree with
        # the old ones, so the opt-in is only accepted in a *new* output
        # directory (spelled out in docs/dev/CASE_REALISTIC.md §7.2).
        for marker in ("history.csv", "spinup.h5", "frozen.h5", "resume.h5",
                       "short.h5", "frames")
            p = joinpath(OUT, marker)
            ispath(p) && error("CASE_EXTDIAG=1 appends a history.csv column " *
                               "and is only allowed in a new CASE_OUT " *
                               "directory; $p already exists")
        end
    end
    mkpath(OUT)
    if mode == "smoke"
        run_case("smoke"; maxsteps=THE_STEPS > 0 ? THE_STEPS : 10)
    elseif mode == "short"
        run_case("short"; maxsteps=THE_STEPS > 0 ? THE_STEPS : 150,
                 checkpoint=joinpath(OUT, "short.h5"))
    elseif mode == "spinup"
        run_case("spinup"; maxsteps=THE_STEPS > 0 ? THE_STEPS : 10000,
                 checkpoint=joinpath(OUT, "spinup.h5"))
    elseif mode == "freeze"
        src = length(ARGS) >= 2 ? ARGS[2] : joinpath(OUT, "spinup.h5")
        dst = joinpath(OUT, "frozen.h5")
        freeze_checkpoint(src, dst)
        run_case("freeze"; restart_from=dst, amr_every=0,
                 maxsteps=THE_STEPS > 0 ? THE_STEPS : 20000,
                 checkpoint=joinpath(OUT, "frozen.h5"))
    elseif mode == "wallrefine"
        # Resume a checkpoint under a *changed* wall-refinement rule: the seed
        # copy carries the new wall keys, the rest of the configuration is the
        # checkpoint's, so the restart guard still checks everything else.
        src = length(ARGS) >= 2 ? ARGS[2] : joinpath(OUT, "spinup.h5")
        seed = wall_checkpoint(src)
        run_case("wallrefine"; restart_from=seed, amr_every=AMR_EVERY,
                 maxsteps=THE_STEPS > 0 ? THE_STEPS : 2000,
                 checkpoint=joinpath(OUT, "resume.h5"))
    elseif mode == "resume"
        # continue any checkpoint with the spin-up settings (AMR on)
        src = length(ARGS) >= 2 ? ARGS[2] : joinpath(OUT, "spinup.h5")
        run_case("resume"; restart_from=src, amr_every=AMR_EVERY,
                 maxsteps=THE_STEPS > 0 ? THE_STEPS : 2000,
                 checkpoint=joinpath(OUT, "resume.h5"))
    elseif mode == "symseed"
        # Prepare a `CASE_SYMMETRY` switch only: one config string is rewritten,
        # the state is copied verbatim.  Resume the printed seed with
        # `CASE_SYMMETRY` set to the same value.
        src = length(ARGS) >= 2 ? ARGS[2] : joinpath(OUT, "spinup.h5")
        symmetry_checkpoint(src,
            joinpath(OUT, "seed", "symmetry_$(SYMMETRY)_from_" * basename(src)))
    else
        error("unknown mode `$mode`; use smoke|short|spinup|freeze|wallrefine|resume|symseed")
    end
end

main()
