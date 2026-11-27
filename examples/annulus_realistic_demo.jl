#!/usr/bin/env julia
# =============================================================================
# annulus_realistic_demo.jl -- the "realistic 2-D mantle convection" card
# (examples/case_realistic_2d.jl) ported to the closed 2-D ring, for the
# QuaST-vs-ASPECT annulus demonstration clip.
#
#   julia --project=. --threads=8 examples/annulus_realistic_demo.jl
#
# The card (all of it is the original box card's, re-used unchanged):
#
#   geometry    full periodic ring PolarAnnulus(1.22, 2.22, 2*pi; seam=:periodic),
#               gap d = R1 - R0 = 1 (the physics length unit), free slip on both
#               circles, inner T = 1 / outer T = 0 (nondimensional);
#   physics     H = 10, nominal Ra = 1e5 at the composite calibration point,
#               composite isostress diffusion+dislocation rheology with the
#               card's prefactors (comp_Ad = 1.6240183119e5,
#               comp_As = 2.6796419238e-1, comp_Ed = 12, comp_Es = 17,
#               comp_n = 3, comp_Tcut = 0, comp_eps_min = 1e-12, eta_T0 = 0.5,
#               eta_min = 1e-4, eta_max = 1e4), two phase transitions at
#               z0 = 0.87 (slope +0.032, B = 1.8182) and z0 = 0.78
#               (slope -0.032, B = 3.0303), per-phase latent heat, Boussinesq
#               (mass_form = :boussinesq), ev = 0.11, no yielding/weakening;
#   initial IC  theta0 = 0.05 sin(pi h) cos(2 phi) + 0.02 sin(2 pi h) cos(4 phi),
#               h = (r - R0)/d, phi = atan2(y,x), a unit-total-temperature
#               perturbation on top of the annular H-inclusive conductive base
#               (the annular analogue of the box card's seed);
#   AMR         the original card's settings: balanced policy, amr_every = 5,
#               refine_frac = 0.2, coarsen_frac = 0.05, max_leaves = 10000,
#               max_level = 7, wall_min_level = 5, wall_strip = 0.1,
#               hysteresis = 1, coarsen gate on, symmetry = :none,
#               indicator = :theta by default; `ANN_REAL_INDICATOR=both` or
#               `velocity` selects another indicator, `ANN_REAL_REFINE_FRAC`
#               overrides the 0.2 Dörfler fraction; both actual values are
#               printed and recorded in the card meta; start from the
#               640-leaf level-3 ring tree
#               (radial 8 x angular 80, i.e. cells_y = 10) that mirrors the box
#               card's 640-leaf level-3 start; no initial_refine (the box card
#               starts from its 640-leaf tree and lets the time loop refine);
#   time        CFL 0.5, dt0 = 1e-6, dtmax = 5e-5, dtmin = 1e-9, growth 1.05,
#               linear = :mg (matrix-free multigrid, the GIF's solver arm),
#               stokes_tol = 1e-3, mg_tol = 1e-8, skew advection,
#               ev_cmax = 0.052, ev_alpha = 2, maxpic = 1 (Picard).
#
# ## The Nusselt pair is undefined on this card, and that is fine here
#
# The per-wall Nusselt normalisation needs the H-inclusive base conductive flux
# -dTbar/dr to be positive at both circles; for H = 10 it is -4.3058 at the
# inner circle (and +5.3815 at the outer one), so `semquad_nu_walls` refuses by
# design.  The demonstration does not use the Nusselt diagnostics (the evidence
# is Tmax / Vrms / meanT), so the run passes `nu_undefined = :nan`: the two Nu
# history columns are NaN, the driver warns once, and the state/trajectory are
# untouched (G294 pins the default refusal and the opt-in mode).  The frame
# writer below has no Nusselt column at all.
#
# ## Latent-heat convention on the ring (stated, not hidden)
#
# `latent_mode = :per_phase` and each transition's coefficient is computed by
# `phase_latent_coefficient` from the **annular** base temperature at that
# transition's height, i.e. `Tbar(R0 + z0) + T_surface` with the ring's radial
# H-inclusive profile.  This is the same formula the box card uses, evaluated on
# the base state the run actually has; it gives (-0.056345, +0.123822) against
# the box card's (-0.059364, +0.130596).  The box values come from
# `per_phase_latent`, which evaluates the *Cartesian* `Tbar(z0, H)` (a box-only
# helper); using them on the ring would be a 5% latent-coefficient mismatch to
# the card's own AV (ASPECT's latent heat uses the state's own temperature), so
# the ring-consistent set is used and both sets are printed.  See the pilot
# record (docs/dev/BUG_FIX_LOG.md) for the audit note.
#
# ## Frame stream (fed to the renderer)
#
# One frame every ANN_FRAME steps, written one step *past* each AMR cycle (the
# AMR block zeroes the flow, so a frame on the AMR step would carry a zero
# velocity), plus the final state.  The writer is the annulus harness's
# `AnnulusFrameWriter` with `profile = :state` (reused from
# examples/annulus_benchmark.jl): per frame the solver's own node cloud
# (x, y, theta, T, ux, uy float64), a companion log10(eta/eta_ref) field, the
# adaptive mesh dump, and a meta row
# frame,step,t,Vrms,meanT,Tmin,Tmax,nnodes,leaves,level_min,level_max,dt,umax,eta_interior.
# `eta_interior` there is the arithmetic node mean of eta/eta_ref over the
# interior band 0.25 <= h <= 0.75; the final summary prints the volume-weighted
# band mean (the guide's section 1.9 convention) as well.
#
# ## ENV knobs (pilot and production)
#
#   ANN_REAL_OUT      output root (default out_julia/annulus_realistic/quast_pilot)
#   ANN_REAL_STEPS    step budget (default 200)
#   ANN_REAL_TSTOP    stop time, nondimensional (default 1e-5)
#   ANN_REAL_FRAME    frame cadence in steps (default 5)
#   ANN_REAL_LINEAR   :mg (default; the GIF's arm), :auto or :direct
#   ANN_REAL_MAXLEAVES  leaf budget override (default 10000, the card's value)
#   ANN_REAL_MG_THREADS  MG threads (default 8)
#   ANN_REAL_REPORT   progress interval in steps (default 25)
#   ANN_REAL_CHECKPOINT        HDF5 checkpoint path (empty = no checkpoint)
#   ANN_REAL_CHECKPOINT_EVERY  steps between checkpoints (default 100)
#   ANN_REAL_RESTART           checkpoint to resume from (empty = from scratch;
#                              the frame stream then appends, frames already written
#                              before the resume step are not rewritten)
#
# The tracked figure's QuaST command (balanced AMR, normalised theta+velocity
# indicator, refine_frac 0.3, 25 000 leaves; full active window t_nd = 1.2e-4 in
# about 4.7 h on 8 threads including the pilot segment).  This is a
# demonstration setting: the card defaults stay theta/0.2, but on this ring the
# theta/0.2 arm was measured to stall at level 7 (28 identical mesh dumps) while
# the budget was below its cap, so the tracked clip uses:
#
#   ANN_REAL_OUT=out_julia/annulus_realistic/quast_pilot_both25_f03 \
#   ANN_REAL_STEPS=6000 ANN_REAL_TSTOP=1.2e-4 ANN_REAL_FRAME=5 \
#   ANN_REAL_MAXLEAVES=25000 ANN_REAL_LINEAR=direct \
#   ANN_REAL_INDICATOR=both ANN_REAL_REFINE_FRAC=0.3 \
#   ANN_REAL_CHECKPOINT=out_julia/annulus_realistic/quast_both03_final.h5 \
#   ANN_REAL_CHECKPOINT_EVERY=100 \
#   julia --project=. --threads=8 examples/annulus_realistic_demo.jl
#
# To continue a frame stream, set ANN_REAL_RESTART to the checkpoint: the
# driver restores the tree, fields, t, dt, step and cooldown, and the frame
# writer appends without rewriting frames.
#
# The default benchmark path of examples/annulus_benchmark.jl is untouched
# (G277/G293): this file only includes it for the tree helper and the frame
# writer.
# =============================================================================
using QuaST, Printf

# Reuse the annulus harness's uniform-ring tree builder, level-range helper and
# frame writer (its `main` is guarded by PROGRAM_FILE, so including is inert).
include(joinpath(@__DIR__, "annulus_benchmark.jl"))

# ------------------------------------------------------------------- card ----
const R0 = ANN_R0                       # 1.22
const R1 = ANN_R1                       # 2.22
const PHI = ANN_PHI                     # 2 pi
const D = R1 - R0                       # the gap d = 1
const RA = 1.0e5
const HHEAT = 10.0
const COMP_AD = 1.6240183119e5
const COMP_AS = 2.6796419238e-1
const COMP_ED = 12.0
const COMP_ES = 17.0
const COMP_N = 3.0
const ETA_T0 = 0.5
const COMP_EPS_MIN = 1.0e-12
const ETA_MIN = 1.0e-4
const ETA_MAX = 1.0e4
const ETA_REF = 1.0
const SLOPE410 = 0.032
const SLOPE660 = -0.032
const WIDTH_PT = 0.025
const B410 = 180.0 / (3300.0 * 3.0e-5 * 1000.0)
const B660 = 300.0 / (3300.0 * 3.0e-5 * 1000.0)
const LATENT_L = 0.11
const LATENT_SCALE = (rho_ref=3300.0, cp=1000.0, dT=1000.0, g=9.81, d=2.9e6,
                      T_surface=0.5)
const SCALE_SET = (rho=3300.0, g=9.81, alpha=3.0e-5, dT=1000.0, d=2.9e6,
                   kappa=1.0e-6)
# AMR card (the original box card's production settings)
const AMR_EVERY = 5
# The box card's 0.2 Dörfler fraction; `ANN_REAL_REFINE_FRAC` is an experiment
# knob and the actual value is recorded in the card meta.
const REFINE_FRAC = parse(Float64, get(ENV, "ANN_REAL_REFINE_FRAC", "0.2"))
0.0 < REFINE_FRAC <= 1.0 ||
    error("ANN_REAL_REFINE_FRAC must be in (0, 1] (got $REFINE_FRAC)")
const COARSEN_FRAC = 0.05
const MAXLEAVES = parse(Int, get(ENV, "ANN_REAL_MAXLEAVES", "10000"))
const MAXLEV = 7
const WALL_MIN = 5
const WALL_STRIP = 0.1
const AMR_HYST = 1
const AMR_GATE = true
const SYMMETRY = :none
# The box card's indicator is :theta.  `ANN_REAL_INDICATOR=both|velocity`
# switches it for experiments; the actual value is printed and recorded in the
# card meta, so a rendered sidecar cannot claim the default when it was not.
const AMR_IND = Symbol(get(ENV, "ANN_REAL_INDICATOR", "theta"))
AMR_IND in (:theta, :velocity, :both) ||
    error("ANN_REAL_INDICATOR must be theta, velocity or both (got $AMR_IND)")
# time-step card
const CFL = 0.5
const DT0 = 1.0e-6
const DTMAX = 5.0e-5
const DTMIN = 1.0e-9
const EV = 0.11
const EV_CMAX = 0.052
const EV_ALPHA = 2
const NG = 80                             # angular cells of the 640-leaf tree
const NX = 8                              # radial cells at level 3

const OUT = get(ENV, "ANN_REAL_OUT",
                normpath(joinpath(@__DIR__, "..", "out_julia",
                                  "annulus_realistic", "quast_pilot")))
const STEPS = parse(Int, get(ENV, "ANN_REAL_STEPS", "200"))
const TSTOP = parse(Float64, get(ENV, "ANN_REAL_TSTOP", "1e-5"))
const FRAME_STRIDE = parse(Int, get(ENV, "ANN_REAL_FRAME", "5"))
const REPORT = parse(Int, get(ENV, "ANN_REAL_REPORT", "25"))
const LINEAR = Symbol(get(ENV, "ANN_REAL_LINEAR", "mg"))
const MG_THREADS = parse(Int, get(ENV, "ANN_REAL_MG_THREADS", "8"))
# Solver-side probe knobs.  The card's box arms ran `stokes_tol = 1e-3` as a
# failure guard (the direct path solves to ~1e-9, so the guard is never active
# there); the ring's matrix-free MG path stalls at a true residual just above it
# (measured in the pilot), so the MG probe needs an explicit, reported value.
const STOKES_TOL = parse(Float64, get(ENV, "ANN_REAL_STOKES_TOL", "1.0e-3"))
const MG_TOL = parse(Float64, get(ENV, "ANN_REAL_MG_TOL", "1.0e-8"))
# Production safety net: the full clip is O(5-7 h), so the run writes an HDF5
# checkpoint every `ANN_REAL_CHECKPOINT_EVERY` steps (and at the end) and can be
# resumed with `ANN_REAL_RESTART=<file>`: the driver restores the tree, theta,
# theta_prev, t, dt, step and pressure, the frame stream continues its numbering
# (the writer appends) and frames already written before the resume step are not
# rewritten.  Empty values keep the historical no-checkpoint pilot path.
const CHECKPOINT = get(ENV, "ANN_REAL_CHECKPOINT", "")
const CHECKPOINT_EVERY = parse(Int, get(ENV, "ANN_REAL_CHECKPOINT_EVERY", "100"))
const RESTART = get(ENV, "ANN_REAL_RESTART", "")

# --------------------------------------------------------------- physics ----
"""
    realistic_phases(pr0, g) -> (ring_phases, box_phases)

The two transitions of the card, with the per-phase latent coefficient evaluated
on the **annular** base state for the pair the run uses, and the box card's
`per_phase_latent` pair returned beside it for the record.
"""
function realistic_phases(pr0::Params, g::PolarAnnulus)
    ph = [PhaseTransition(0.87, WIDTH_PT, SLOPE410, 0.5, 180.0, B410),
          PhaseTransition(0.78, WIDTH_PT, SLOPE660, 0.5, 300.0, B660)]
    box = per_phase_latent(pr0, deepcopy(ph); LATENT_SCALE...)
    ring = [PhaseTransition(pt.z0, pt.width, pt.slope, pt.Tref, pt.drho, pt.B,
                            phase_latent_coefficient(pt,
                                semquad_base_T(pr0, g, R0 + pt.z0, 0.0) +
                                LATENT_SCALE.T_surface;
                                rho_ref=LATENT_SCALE.rho_ref,
                                cp=LATENT_SCALE.cp,
                                dT=LATENT_SCALE.dT,
                                g=LATENT_SCALE.g,
                                d=LATENT_SCALE.d))
            for pt in box]
    return ring, box
end

function realistic_params()
    return Params(Ra=RA, H=HHEAT,
                  eta_law=:composite, comp_rule=:isostress,
                  comp_Ad=COMP_AD, comp_As=COMP_AS,
                  comp_Ed=COMP_ED, comp_Es=COMP_ES,
                  comp_Vd=0.0, comp_Vs=0.0, comp_n=COMP_N,
                  comp_Tcut=0.0, comp_eps_min=COMP_EPS_MIN,
                  eta_T0=ETA_T0, eta_min=ETA_MIN, eta_max=ETA_MAX,
                  eta_ref=ETA_REF,
                  bc=:free,
                  latent_L=LATENT_L, latent_mode=:per_phase,
                  yield_c=Inf, yield_phi=0.0, yield_taumax=Inf,
                  yield_c_grad=0.0, yield_mu_grad=0.0,
                  weak_f=1.0, weak_eps_ref=0.0,
                  rho_heating=false, rho_cp=false, Di=0.0,
                  heating_adiabatic=false, heating_dissipation=false,
                  refstate=nothing, deviatoric=false)
end

"""The card's seed: a unit-total-temperature perturbation on the annular
H-inclusive conductive base; it vanishes on both circles."""
function seed(x, y)
    r = hypot(x, y)
    h = (r - R0) / D
    phi = atan(y, x)
    return 0.05 * sinpi(h) * cos(2.0 * phi) +
           0.02 * sinpi(2.0 * h) * cos(4.0 * phi)
end

"""Volume-weighted interior-band mean of eta/eta_ref (the guide's section 1.9
convention) on the final state, through the same closure the flow solve used."""
function eta_ref_stats(ss, qt, theta, ux, uz, pr, g)
    sq = ss.su
    leaves = Set(qt.leaves)
    eta_fun = (x, y) -> begin
        T = semquad_base_T(pr, g, x, y) +
            QuaST._semquad_eval_at(sq, theta, x, y, leaves)
        ed = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
        eta_composite_at(T, ed, height(g, x, y), pr)[1] / pr.eta_ref
    end
    return QuaST._eta_ref_field_stats(sq, qt, eta_fun, pr)
end

function main()
    g = PolarAnnulus(R0, R1, PHI; seam=:periodic)
    qt = annulus_benchmark_tree(NX, NG; maxlevel=MAXLEV)   # 640 leaves, level 3
    ss = make_semstokes(qt, 2; geom=g)
    pr = realistic_params()
    pr0 = Params(Ra=RA, H=HHEAT)
    ph, ph_box = realistic_phases(pr0, g)
    theta0 = [seed(c[1], c[2]) for c in ss.su.coords]
    for (i, c) in enumerate(ss.su.coords)
        h = (hypot(c[1], c[2]) - R0) / D
        (h <= 1.0e-12 || h >= 1.0 - 1.0e-12) && (theta0[i] = 0.0)
    end
    all(isfinite, theta0) ||
        error("the card's seed is not finite on the 640-leaf ring")
    resuming = !isempty(RESTART)
    resuming && (theta0 = zeros(ss.su.N))   # the driver replaces it

    mkpath(OUT)
    println("# annulus realistic card: ring R0=$R0 R1=$R1 phi=2pi ",
            "d=$D free-slip; Ra=$RA H=$HHEAT composite isostress; ",
            "phases z0=0.87/0.78 slope=+$SLOPE410/$SLOPE660 B=$B410/$B660; ",
            "latent per_phase ring-consistent")
    println("# tree: nx=$NX ny=$NG leaves=", length(qt.leaves), " dofs=",
            2 * ss.su.N + ss.sp.N, " (u=", 2 * ss.su.N, " p=", ss.sp.N, " T=",
            ss.su.N, ") max_level=$MAXLEV")
    println("# latent coefficients: ring [",
            join([@sprintf("%+.6f", pt.latent) for pt in ph], ", "),
            "]  box per_phase_latent [",
            join([@sprintf("%+.6f", pt.latent) for pt in ph_box], ", "), "]")
    println("# AMR: balanced every=$AMR_EVERY steps refine_frac=$REFINE_FRAC ",
            "coarsen_frac=$COARSEN_FRAC max_leaves=$MAXLEAVES max_level=$MAXLEV ",
            "wall_min_level=$WALL_MIN wall_strip=$WALL_STRIP hysteresis=",
            "$AMR_HYST gate=$AMR_GATE symmetry=$SYMMETRY indicator=$AMR_IND ",
            "initial_refine=0")
    println("# time: cfl=$CFL dt0=$DT0 dtmax=$DTMAX dtmin=$DTMIN ev=$EV ",
            "linear=$LINEAR mg_tol=$MG_TOL stokes_tol=$STOKES_TOL ",
            "nu_undefined=:nan (the H=10 ring's Nusselt pair is undefined ",
            "by construction)")
    println("# run window: maxsteps=$STEPS tstop=$TSTOP frames every ",
            "$FRAME_STRIDE steps -> $OUT")
    if !isempty(CHECKPOINT)
        println("# checkpoint: $CHECKPOINT every $CHECKPOINT_EVERY steps")
    end
    if resuming
        println("# resume: $RESTART (frames append to $OUT, none rewritten)")
    end
    flush(stdout)

    resume_step = 0
    leaves_last = length(qt.leaves)
    if resuming
        metapath = joinpath(OUT, "frame_meta.csv")
        if isfile(metapath)
            rows = [l for l in readlines(metapath) if !startswith(l, "#") &&
                    !startswith(l, "frame,") && !isempty(strip(l))]
            if !isempty(rows)
                fields = split(rows[end], ',')
                resume_step = parse(Int, fields[2])
                leaves_last = parse(Int, fields[9])
            end
        end
        println("# resume: last written frame is step $resume_step ",
                "(tree leaves $leaves_last); later frames are appended")
        flush(stdout)
    end
    fw = AnnulusFrameWriter(OUT, NX, NG, g, pr; profile=:state,
                            mesh_dump=true, defer_stale=true,
                            resume=resuming && isfile(joinpath(OUT,
                                                             "frame_meta.csv")))
    leaves_max = leaves_last
    level_seen_min, level_seen_max = annulus_level_range(qt)
    amr_cycles = 0
    stale_n = 0
    cb = (st) -> begin
        nl = length(st.qt.leaves)
        stale = AMR_EVERY > 0 && all(iszero, st.ux) && all(iszero, st.uz)
        if nl != leaves_last || stale
            # a mesh change happened in the AMR block of this step
            if nl != leaves_last
                amr_cycles += 1
                lm, lM = annulus_level_range(st.qt)
                level_seen_min = min(level_seen_min, lm)
                level_seen_max = max(level_seen_max, lM)
                leaves_max = max(leaves_max, nl)
                @printf("  AMR  step %6d t=%.6e leaves=%d level=%d-%d dt=%.3e\n",
                        st.step, st.t, nl, lm, lM, st.dt)
                flush(stdout)
            end
            stale_n += stale ? 1 : 0
            leaves_last = nl
        end
        # frames go one step past every AMR cycle (the AMR block zeroes the
        # flow fields for the next solve, so a frame on the AMR step would
        # report a zero velocity); a resumed run never rewrites a frame that
        # was already written before the checkpoint
        if st.step % FRAME_STRIDE == 1 && st.step > resume_step
            write_annulus_frame!(fw, st, pr, g)
        end
        if st.step % REPORT == 0
            vr = semquad_vrms(st.sq, st.qt, st.ux, st.uz)
            tm = semquad_mean_T(st.sq, st.qt, st.theta, pr)
            umax = maximum(hypot.(st.ux, st.uz))
            Tmax = maximum(semquad_base_T(pr, g, c[1], c[2]) + st.theta[i]
                           for (i, c) in enumerate(st.sq.coords))
            lm, lM = annulus_level_range(st.qt)
            @printf("  step %6d t=%.6e dt=%.3e leaves=%d level=%d-%d Vrms=%.5e meanT=%.6f Tmax=%.6f u_max=%.5e\n",
                    st.step, st.t, st.dt, length(st.qt.leaves), lm, lM, vr, tm,
                    Tmax, umax)
            flush(stdout)
        end
        return nothing
    end

    ckpt_meta = Dict{String,Any}(
        "demo" => "annulus_realistic", "R0" => R0, "R1" => R1,
        "Phi" => PHI, "Ra" => RA, "H" => HHEAT,
        "comp_Ad" => COMP_AD, "comp_As" => COMP_AS, "comp_Ed" => COMP_ED,
        "comp_Es" => COMP_ES, "comp_n" => COMP_N, "comp_Tcut" => 0.0,
        "comp_eps_min" => COMP_EPS_MIN, "comp_rule" => "isostress",
        "eta_T0" => ETA_T0, "eta_min" => ETA_MIN, "eta_max" => ETA_MAX,
        "eta_ref" => ETA_REF, "latent_mode" => "per_phase",
        "latent410" => ph[1].latent, "latent660" => ph[2].latent,
        "B410" => B410, "B660" => B660, "slope410" => SLOPE410,
        "slope660" => SLOPE660, "width_pt" => WIDTH_PT, "L" => LATENT_L,
        "p" => 2, "max_level" => MAXLEV, "max_leaves" => MAXLEAVES,
        "amr_policy" => "balanced", "amr_every" => AMR_EVERY,
        "refine_frac" => REFINE_FRAC, "coarsen_frac" => COARSEN_FRAC,
        "wall_min_level" => WALL_MIN, "wall_strip" => WALL_STRIP,
        "amr_hysteresis" => AMR_HYST, "amr_coarsen_gate" => AMR_GATE,
        "symmetry" => "none", "indicator" => string(AMR_IND),
        "linear" => string(LINEAR), "cfl" => CFL, "ev" => EV,
        "nu_undefined" => "nan",
        "seed" => "0.05 sin(pi h) cos(2 phi) + 0.02 sin(2 pi h) cos(4 phi)")

    wall0 = time()
    res = run_semquad_simulation(qt, pr, theta0;
        geom=g, p=2, tstop=TSTOP, maxsteps=STEPS,
        dt0=DT0, dtmax=DTMAX, dtmin=DTMIN, cfl=CFL, dt_growth=1.05,
        amr_every=AMR_EVERY, refine_frac=REFINE_FRAC,
        coarsen_frac=COARSEN_FRAC, max_leaves=MAXLEAVES,
        max_level=MAXLEV, wall_min_level=WALL_MIN, wall_strip=WALL_STRIP,
        amr_hysteresis=AMR_HYST, amr_coarsen_gate=AMR_GATE,
        symmetry=SYMMETRY, amr_indicator=AMR_IND, amr_passes=1,
        initial_refine=0, transfer_mode=:nodal,
        advection=:skew, ev=EV, ev_cmax=EV_CMAX, ev_alpha=EV_ALPHA,
        maxpic=1, pic_tol=1.0e-6, pic_mode=:picard,
        linear=LINEAR, mg_threads=MG_THREADS, mg_tol=MG_TOL,
        stokes_tol=STOKES_TOL, stokes_scaling=:equilibrated,
        mass_form=:boussinesq, phases=ph, latent_scheme=:implicit,
        nu_undefined=:nan,
        eta_ref_scale=SCALE_SET,
        restart_from=RESTART, checkpoint_path=CHECKPOINT,
        checkpoint_every=CHECKPOINT_EVERY, checkpoint_meta=ckpt_meta,
        profile_steps=true, verbose=true, callback=cb)
    wall = time() - wall0
    isempty(res.history) && error("the run did no step; nothing to report")

    # final frame (the run's last state, on its final tree); a resumed run whose
    # last written frame already shows this step does not duplicate it
    if res.steps > resume_step
        write_annulus_frame!(fw, (step=res.steps, t=res.t,
                                  dt=res.history[end][2], theta=res.theta,
                                  ux=res.ux, uz=res.uz, sq=res.ss.su,
                                  qt=res.qt), pr, g)
    end
    close(fw.meta)

    # final measurements
    last = res.history[end]
    vr = semquad_vrms(res.ss.su, res.qt, res.ux, res.uz)
    tm = semquad_mean_T(res.ss.su, res.qt, res.theta, pr)
    umax = maximum(hypot.(res.ux, res.uz))
    Tmax = maximum(semquad_base_T(pr, g, c[1], c[2]) + res.theta[i]
                   for (i, c) in enumerate(res.ss.su.coords))
    Tmin = minimum(semquad_base_T(pr, g, c[1], c[2]) + res.theta[i]
                   for (i, c) in enumerate(res.ss.su.coords))
    # nodal interior mean of eta/eta_ref (the frame meta's definition)
    ed = semquad_strain_rate_field(res.ss.su, res.ux, res.uz)
    num = 0.0; cnt = 0
    for (i, c) in enumerate(res.ss.su.coords)
        h = height(g, c[1], c[2])
        if 0.25 <= h <= 0.75
            T = semquad_base_T(pr, g, c[1], c[2]) + res.theta[i]
            num += eta_composite_at(T, ed[i], h, pr)[1] / pr.eta_ref
            cnt += 1
        end
    end
    eta_int_node = num / max(cnt, 1)
    estats = eta_ref_stats(res.ss, res.qt, res.theta, res.ux, res.uz, pr, g)
    tprof = SEMQUAD_STEP_PROFILE[]
    warm = length(tprof) > 3 ? tprof[4:end] : tprof
    s_warm = isempty(warm) ? NaN : sum(r.total for r in warm) / length(warm)
    lfin_min, lfin_max = annulus_level_range(res.qt)
    @printf("ANNULUS_REALISTIC_RESULT steps=%d t=%.6e dt=%.3e wall=%.1f s_per_step=%.4f s_step_warm=%.4f linear=%s leaves_final=%d leaves_max=%d level_final=%d-%d level_seen=%d-%d amr_cycles=%d Vrms=%.6e meanT=%.6f Tmax=%.6f Tmin=%.6f u_max=%.6e eta_interior_node=%.6e eta_interior_vol=%.6e eta_min=%.6e eta_max=%.6e eta_contrast=%.3e Ra_interior=%.6e frames=%d stale_callbacks=%d\n",
            res.steps, res.t, last[2], wall, wall / max(length(res.history), 1),
            s_warm, LINEAR, length(res.qt.leaves), leaves_max, lfin_min,
            lfin_max, level_seen_min, level_seen_max, amr_cycles, vr, tm, Tmax,
            Tmin, umax, eta_int_node, estats.interior, estats.min, estats.max,
            estats.contrast, RA / estats.interior, fw.frame, stale_n)
    @printf("ANNULUS_REALISTIC_LATENT ring=%s box=%s latent_L=%.3f mode=%s\n",
            join([@sprintf("%+.6f", pt.latent) for pt in ph], ","),
            join([@sprintf("%+.6f", pt.latent) for pt in ph_box], ","),
            LATENT_L, "per_phase")
    @printf("ANNULUS_REALISTIC_NU nu_undefined=:nan history_columns_3_4=NaN steady_criterion=disabled (the H=10 ring's Nusselt pair is undefined: -dTbar/dr(R0) = %.6f)\n",
            QuaST.semquad_base_wall_refs(pr, g)[1])
    flush(stdout)
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
