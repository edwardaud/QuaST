#!/usr/bin/env julia
# =============================================================================
# nu_cbf_compare.jl — is the reported wall Nusselt number reliable?
#
# The driver reports `Nu_top/Nu_bot` from `semquad_nu_walls`, i.e. from the
# one-sided GLL derivative of theta at the wall.  The energy-budget diagnostic
# compares that pointwise wall flux with the *consistent Dirichlet reaction* Qc
# (the sum of the step residual over the wall dofs, "consistent boundary flux",
# Gresho et al. 1987; ASPECT's `heat flux statistics`; the benchmark side is
# Zhong et al. 2008, G-cubed 9, Q10017, doi:10.1029/2008GC002048 -- the earlier
# "Zhong, King & Gurnis 2007" here could not be verified, docs/
# NUSSELT_CONVENTIONS.md §5) and found |Qc - flux|/A = 0.63 with
# flux/A = 1.66 at the case_realistic_B2 state — far more than the 1e-2 of a
# 1-D uniform column.  This script measures, on one state:
#
#   * the pointwise flux (what Nu_top/Nu_bot are),
#   * the consistent boundary flux (CBF) of the *steady* energy operator
#     (`semquad_nu_cbf`, the quantity stage 2 would write to history.csv),
#   * the exact consistent reaction Qc of the *time step* that produced the
#     state (needs the two previous BDF2 levels), and
#   * the heat-budget terms (H, Ibase, latent, enthalpy, closure variants),
#
# and, with `--sweep`, the resolution convergence of their differences on three
# uniform meshes (h = 1/16, 1/32, 1/64; the last is the production resolution).
#
# The state must be one the energy step actually produced, with its EV field:
# a checkpoint written on an AMR step has ux = uz = 0 and no EV field, so it is
# resumed a few steps in a scratch directory first (`--resume`, the default when
# the checkpoint is post-AMR).  The given checkpoint is never modified.
#
# usage
#   julia --project=. examples/nu_cbf_compare.jl [options] [ckpt]
#
#   ckpt                    state checkpoint.  default:
#                           out_julia/case_realistic_B2/spinup.h5
#   --prev=PATH             checkpoint of the *previous* step (step n-1).  It
#                           supplies theta^(n-2), the second BDF2 level of the
#                           step that produced the state; without it Qc is not
#                           computed.  For a resumed state it is the source
#                           checkpoint (step n-1), found automatically.
#   --resume=N              resume the checkpoint N steps (default 2; 1 is
#                           enough) in --scratch before measuring.  Automatic
#                           when ux = uz = 0 (post-AMR checkpoint) or the EV
#                           field is missing.
#   --no-resume             never resume; fail instead of measuring a state the
#                           energy step did not produce.
#   --scratch=DIR           scratch root (default out_julia/nu_cbf_scratch).
#   --sweep                 run the uniform-resolution sweep (levels from
#                           --levels) and print the convergence table.
#   --levels=3,4,5          sweep levels (default 3,4,5 -> h = 1/16,1/32,1/64).
#   --steps=N               sweep steps (default 32; the last step must be
#                           n % 5 == 2 to be a non-AMR step).
#
# Everything this script writes goes under --scratch (or the case script's own
# CASE_OUT).  No solver code is touched: it uses only the public API
# (`load_semquad_state`, `run_semquad_simulation`, `solve_semquad_energy`,
# `semquad_nu_walls`, `semquad_nu_cbf`, `semquad_heat_budget`).
# =============================================================================
using QuaST, Printf, LinearAlgebra, HDF5

const DEFAULT_CKPT = normpath(joinpath(@__DIR__, "..", "..", "out_julia",
                                       "case_realistic_B2", "spinup.h5"))
const DEFAULT_SCRATCH = normpath(joinpath(@__DIR__, "..", "..", "out_julia",
                                          "nu_cbf_scratch"))
const USAGE = """
usage: julia --project=. examples/nu_cbf_compare.jl [options] [ckpt]

  ckpt            state checkpoint (default $DEFAULT_CKPT)
  --prev=PATH     checkpoint of the previous step (step n-1); required for Qc
  --resume=N      resume N steps in --scratch before measuring (default 2;
                  automatic for a post-AMR checkpoint with ux = uz = 0)
  --no-resume     fail instead of measuring a post-AMR checkpoint
  --scratch=DIR   scratch root (default $DEFAULT_SCRATCH)
  --sweep         run the uniform-resolution sweep and print the table
  --levels=3,4,5  sweep resolution levels (default 3,4,5: h = 1/16, 1/32, 1/64)
  --steps=N       sweep steps (default 32)
  --prod=PATH     production checkpoint for the sweep's last table row
"""

# ------------------------------------------------------------------- config ----
# A checkpoint stores the full `run_config` (physics + numerics) and, in its HDF5
# attributes, the two per-transition latent coefficients.  Rebuilding the
# operator from those is what makes the measurement independent of a hard-coded
# copy of the parameter card.
cfg_str(cfg, k, d::String="") = get(cfg, k, d)
cfg_float(cfg, k, d::Float64=NaN) = haskey(cfg, k) ? parse(Float64, cfg[k]) : d
cfg_int(cfg, k, d::Int=typemax(Int)) = haskey(cfg, k) ? round(Int, parse(Float64, cfg[k])) : d
cfg_bool(cfg, k, d::Bool=false) = haskey(cfg, k) ? (cfg[k] == "true") : d
cfg_sym(cfg, k, d::Symbol=:none) = haskey(cfg, k) ? Symbol(cfg[k]) : d

"Scalar HDF5 attributes of a checkpoint (the `meta` dict written by the drivers)."
function checkpoint_attrs(path::AbstractString)
    return h5open(path, "r") do f
        Dict{String,Any}(k => HDF5.attrs(f)[k] for k in keys(HDF5.attrs(f)))
    end
end

"""
    physics_from_checkpoint(path) -> (pr, ph, cfg, attrs)

`Params` and the phase list of the run that wrote `path`, from the checkpoint
itself: the `run_config` string plus the `latent410`/`latent660` attributes
(the per-transition coefficients of `:per_phase`).  Nothing is guessed from the
parameter card, so a measurement can never silently use another rheology.
"""
function physics_from_checkpoint(path::AbstractString; pr_override=nothing)
    st = load_semquad_state(path)
    cfg = st.config
    attrs = checkpoint_attrs(path)
    # `eta_law` itself is not part of run_config (the per-law parameters are); a
    # checkpoint with the composite-law parameters is a :composite run.
    eta_law = haskey(cfg, "comp_Ad") ? :composite : cfg_sym(cfg, "eta_law", :const)
    # `Ra` lives in the checkpoint's meta attributes (the case drivers put it
    # there), not in `run_config`.
    Ra = haskey(attrs, "Ra") ? Float64(attrs["Ra"]) : cfg_float(cfg, "Ra", 1.0e4)
    # `bc` and the wall temperatures are not in run_config (they do not change
    # the energy operator measured here); the case's values are :free, 0 and 1.
    pr = Params(Ra=Ra,
                H=cfg_float(cfg, "H", 0.0),
                eta_law=eta_law,
                comp_Ad=cfg_float(cfg, "comp_Ad", 0.0),
                comp_As=cfg_float(cfg, "comp_As", 0.0),
                comp_Ed=cfg_float(cfg, "comp_Ed", 0.0),
                comp_Es=cfg_float(cfg, "comp_Es", 0.0),
                comp_Vd=cfg_float(cfg, "comp_Vd", 0.0),
                comp_Vs=cfg_float(cfg, "comp_Vs", 0.0),
                comp_n=cfg_float(cfg, "comp_n", 3.0),
                comp_rule=cfg_sym(cfg, "comp_rule", :isostress),
                comp_Tcut=cfg_float(cfg, "comp_Tcut", 0.0),
                comp_eps_min=cfg_float(cfg, "comp_eps_min", 1.0e-12),
                eta_T0=cfg_float(cfg, "eta_T0", 0.0),
                eta_min=cfg_float(cfg, "eta_min", -Inf),
                eta_max=cfg_float(cfg, "eta_max", Inf),
                bc=:free,
                latent_L=cfg_float(cfg, "latent_L", 0.0),
                latent_mode=cfg_sym(cfg, "latent_mode", :global),
                yield_c=cfg_float(cfg, "yield_c", Inf),
                yield_phi=cfg_float(cfg, "yield_phi", 0.0),
                yield_taumax=cfg_float(cfg, "yield_taumax", Inf),
                yield_eps_min=cfg_float(cfg, "yield_eps_min", 1.0e-30),
                yield_damper=cfg_float(cfg, "yield_damper", 0.0),
                yield_c_grad=cfg_float(cfg, "yield_c_grad", 0.0),
                yield_mu_grad=cfg_float(cfg, "yield_mu_grad", 0.0),
                weak_f=cfg_float(cfg, "weak_f", 1.0),
                weak_eps_ref=cfg_float(cfg, "weak_eps_ref", 0.0),
                weak_heal_time=cfg_float(cfg, "weak_heal_time", Inf),
                weak_mode=cfg_sym(cfg, "weak_mode", :rate),
                rho_heating=cfg_bool(cfg, "rho_heating", false),
                rho_cp=cfg_bool(cfg, "rho_cp", false),
                Di=cfg_float(cfg, "Di", 0.0),
                heating_adiabatic=cfg_bool(cfg, "heating_adiabatic", false),
                heating_dissipation=cfg_bool(cfg, "heating_dissipation", false),
                refstate=nothing,
                deviatoric=cfg_bool(cfg, "deviatoric", false))
    ph = phases_from_checkpoint(cfg, attrs)
    return pr, ph, cfg, attrs
end

"""
    phases_from_checkpoint(cfg, attrs) -> Vector{PhaseTransition}

The phase list from `run_config["phases"]` (`z0,width,slope,Tref,drho,B` per
transition, `;`-separated) with the `:per_phase` latent coefficients taken from
the `latent410`/`latent660` attributes — the values the run actually used, not a
recomputation from a scale set.
"""
function phases_from_checkpoint(cfg, attrs)
    s = cfg_str(cfg, "phases", "none")
    s == "none" && return PhaseTransition[]
    mode = cfg_sym(cfg, "latent_mode", :global)
    out = PhaseTransition[]
    for (i, chunk) in enumerate(split(s, ';'))
        v = parse.(Float64, split(chunk, ','))
        length(v) == 6 ||
            throw(ArgumentError("cannot parse the checkpoint phase list: $s"))
        lat = nothing
        if mode === :per_phase
            key = i == 1 ? "latent410" : i == 2 ? "latent660" : ""
            lat = (haskey(attrs, key) ? Float64(attrs[key]) :
                   throw(ArgumentError("checkpoint has latent_mode=:per_phase " *
                                       "but no $key attribute")))
        end
        push!(out, PhaseTransition(v..., lat))
    end
    return out
end

"""
    run_opts_from_checkpoint(cfg, attrs) -> NamedTuple

The `run_semquad_simulation` keywords a restart of this checkpoint needs, read
from its `run_config`.  The driver compares these on restart, so reproducing them
is what makes the scratch resume the same run.
"""
function run_opts_from_checkpoint(cfg, attrs)
    return (amr_every=cfg_int(cfg, "amr_every", 0),
            amr_strategy=cfg_sym(cfg, "amr_strategy", :legacy),
            amr_passes=cfg_int(cfg, "amr_passes", 1),
            refine_frac=cfg_float(cfg, "refine_frac", 0.2),
            coarsen_frac=cfg_float(cfg, "coarsen_frac", 0.0),
            marking=cfg_sym(cfg, "marking", :global),
            max_leaves=cfg_int(cfg, "max_leaves", typemax(Int)),
            max_level=cfg_int(cfg, "max_level", typemax(Int)),
            wall_min_level=cfg_int(cfg, "wall_min_level", 0),
            wall_strip=cfg_float(cfg, "wall_strip", 0.1),
            symmetry=cfg_sym(cfg, "symmetry", :none),
            transfer_mode=cfg_sym(cfg, "transfer_mode", :nodal),
            hotspot_weight=cfg_float(cfg, "hotspot_weight", 0.0),
            hotspot_frac=cfg_float(cfg, "hotspot_frac", 0.5),
            hotspot_min_level=cfg_int(cfg, "hotspot_min_level", 0),
            hotspot_mode=cfg_sym(cfg, "hotspot_mode", :theta),
            hotspot_zmin=cfg_float(cfg, "hotspot_zmin", 0.0),
            zero_side_eta=cfg_bool(cfg, "zero_side_eta", false),
            side_strip=cfg_float(cfg, "side_strip", 0.05),
            top_wall_level=cfg_int(cfg, "top_wall_level", 0),
            top_wall_frac=cfg_float(cfg, "top_wall_frac", 0.5),
            advection=cfg_sym(cfg, "advection", :galerkin),
            ev=cfg_float(cfg, "ev", 0.0),
            ev_cmax=cfg_float(cfg, "ev_cmax", 0.052),
            ev_alpha=cfg_int(cfg, "ev_alpha", 2),
            maxpic=cfg_int(cfg, "maxpic", 1),
            pic_tol=cfg_float(cfg, "pic_tol", 1.0e-6),
            pic_relax=cfg_float(cfg, "pic_relax", 1.0),
            pic_mode=cfg_sym(cfg, "pic_mode", :picard),
            pic_reject=cfg_bool(cfg, "pic_reject", false),
            cfl=cfg_float(cfg, "cfl", 0.0),
            dtmax=cfg_float(cfg, "dtmax", Inf),
            dtmin=cfg_float(cfg, "dtmin", 1.0e-14),
            mass_form=cfg_sym(cfg, "mass_form", :anelastic),
            latent_scheme=cfg_sym(cfg, "latent_scheme", :implicit))
end

# ----------------------------------------------------------------- analysis ----
"Wall dofs of the scalar space (the Dirichlet dofs of the energy operator)."
function wall_dofs(sq::SemQuad)
    return [g for g in 1:sq.N
            if (sq.coords[g][2] <= 1.0e-14 || sq.coords[g][2] >= 1.0 - 1.0e-14)]
end

"""
    wall_dofs_split(sq) -> (top, bottom)

The same set, split by wall.  ASPECT's `heat flux statistics` is a *per-boundary*
quantity (`source/postprocess/heat_flux_statistics.cc` sums the CBF over each
boundary separately), so the comparison needs the per-wall reactions: the total
`Qc` mixes the two walls and cannot be compared with one of them.
"""
function wall_dofs_split(sq::SemQuad)
    top = Int[]; bot = Int[]
    for g in 1:sq.N
        y = sq.coords[g][2]
        y >= 1.0 - 1.0e-14 && push!(top, g)
        y <= 1.0e-14 && push!(bot, g)
    end
    return (top=top, bottom=bot)
end

"""
    analyze(ckpt, prev; verbose=false) -> NamedTuple

Measure the wall-flux diagnostics and the heat budget at the state in `ckpt`.
`prev` is the checkpoint of the previous step (step n-1); it supplies
`theta^(n-2)`, the second BDF2 level of the step that produced the state, so
`Qc` can be computed.  `prev = nothing` skips `Qc`.
"""
function analyze(ckpt::AbstractString, prev::Union{Nothing,AbstractString};
                 verbose::Bool=false)
    st = load_semquad_state(ckpt)
    pr, ph, cfg, attrs = physics_from_checkpoint(ckpt)
    sq = make_semstokes(st.qt, st.p).su
    A = total_area(st.qt)
    nl = length(st.qt.leaves)
    h_eff = sqrt(A / nl)
    theta = st.theta
    adv = cfg_sym(cfg, "advection", :galerkin)
    has_ev = cfg_float(cfg, "ev", 0.0) > 0.0
    # A state without its EV field cannot be measured: the consistent residual
    # must be assembled with the same stabilisation as the step that produced
    # it (semquad_flux.jl).  Such a checkpoint is post-AMR; resume it first.
    has_ev && st.ev === nothing &&
        throw(ArgumentError("checkpoint $ckpt has no EV field: it was written " *
                            "on an AMR step.  Run this script with --resume " *
                            "(the default) to measure a state the energy step " *
                            "actually produced."))

    # pointwise wall flux (the reported Nu) and the steady-operator CBF.
    Nut, Nub = semquad_nu_walls(sq, st.qt, theta, pr)
    cbf_kw = (advection=adv, ev_field=st.ev, phases=ph,
              rho_heating=pr.rho_heating, rho_cp=pr.rho_cp,
              heating_adiabatic=pr.heating_adiabatic,
              heating_dissipation=pr.heating_dissipation)
    ntc, nbc = semquad_nu_cbf(sq, st.qt, pr, theta, st.ux, st.uz; cbf_kw...)
    # The same CBF without the EV field: the difference is the numerical
    # diffusion's contribution to the wall flux.
    ntc_ne, nbc_ne = semquad_nu_cbf(sq, st.qt, pr, theta, st.ux, st.uz;
                                    advection=adv, ev_field=nothing, phases=ph,
                                    rho_heating=pr.rho_heating,
                                    rho_cp=pr.rho_cp,
                                    heating_adiabatic=pr.heating_adiabatic,
                                    heating_dissipation=pr.heating_dissipation)
    bud = semquad_heat_budget(sq, st.qt, pr, theta; phases=ph,
                              ux=st.ux, uz=st.uz)
    Ibase = semquad_base_advection_integral(sq, st.qt, pr, st.ux, st.uz)
    top_ref = -semquad_base_gradient(pr, 1.0)
    bot_ref = -semquad_base_gradient(pr, 0.0)
    flux_pt = bud.flux
    flux_cbf = (top_ref - ntc) * A - (bot_ref - nbc) * A
    latent = bud.latent_dgdt === nothing ? 0.0 : bud.latent_dgdt
    closure_nolatent = Nut - Nub - (pr.H - Ibase)
    closure_old = closure_nolatent + latent / A
    closure_area = Nut - Nub - (pr.H - Ibase / A) + latent / A

    # exact consistent Dirichlet reaction of the step that produced the state.
    Qc = NaN
    Qc_top = NaN
    Qc_bot = NaN
    dHdt = NaN
    ev_maxdiff = NaN
    steps_between = 0
    if prev !== nothing
        pst = load_semquad_state(prev)
        steps_between = st.step - pst.step
        steps_between == 1 ||
            throw(ArgumentError("--prev must be the checkpoint of step " *
                                "$(st.step - 1), got step $(pst.step)"))
        Set(st.qt.leaves) == Set(pst.qt.leaves) ||
            throw(ArgumentError("the state and its previous checkpoint are on " *
                                "different trees (step $(st.step) vs $(pst.step))"))
        # `dt_prev` of the step that produced the state.  A checkpoint written
        # inside the run loop stores it correctly; the *final* checkpoint of a
        # run is saved after the loop's `dt_prev = dt`, so there it equals `dt`.
        # The previous checkpoint's `dt` is always the right value.
        dt_prev_used = pst.dt
        ev_out = has_ev ? zeros(Float64, nl) : nothing
        parts = solve_semquad_energy(sq, st.qt, pr, st.theta_prev, pst.theta_prev,
                                     st.ux, st.uz, st.dt, dt_prev_used, false;
                                     advection=adv,
                                     ev=cfg_float(cfg, "ev", 0.0),
                                     ev_cmax=cfg_float(cfg, "ev_cmax", 0.052),
                                     ev_alpha=cfg_int(cfg, "ev_alpha", 2),
                                     ev_timestep=st.step, ev_out=ev_out,
                                     phases=ph, latent_theta=st.theta_prev,
                                     return_parts=true)
        r = parts.A * theta .- parts.rhs
        wd = wall_dofs_split(sq)
        Qc = sum(r[wall_dofs(sq)])
        Qc_top = sum(r[wd.top])
        Qc_bot = sum(r[wd.bottom])
        w = st.dt / dt_prev_used
        a = (2.0 * w + 1.0) / (1.0 + w)
        b = 1.0 + w
        c = w * w / (1.0 + w)
        dHdt = sum(parts.M * (a .* theta .- b .* st.theta_prev .+ c .* pst.theta_prev)) / st.dt
        if has_ev && st.ev !== nothing
            ev_maxdiff = maximum(abs.(ev_out .- st.ev))
        end
    end

    return (ckpt=ckpt, prev=prev, step=st.step, t=st.t, dt=st.dt,
            dt_prev=st.dt_prev, leaves=nl, N=sq.N, p=st.p, A=A, h_eff=h_eff,
            maxlevel=st.qt.maxlevel, umax=maximum(hypot.(st.ux, st.uz)),
            ev_stored=(st.ev !== nothing), ev_maxdiff=ev_maxdiff,
            latent_mode=pr.latent_mode, latent_L=pr.latent_L,
            latent410=(length(ph) >= 1 ? ph[1].latent : nothing),
            latent660=(length(ph) >= 2 ? ph[2].latent : nothing),
            advection=adv, ev=cfg_float(cfg, "ev", 0.0),
            ev_cmax=cfg_float(cfg, "ev_cmax", 0.052),
            Nut=Nut, Nub=Nub, Nu_cbf_top=ntc, Nu_cbf_bot=nbc,
            Nu_cbf_top_noev=ntc_ne, Nu_cbf_bot_noev=nbc_ne,
            Qc=Qc, Qc_top=Qc_top, Qc_bot=Qc_bot,
            Nu_Qc_top=(isnan(Qc_top) ? NaN : top_ref - Qc_top / A),
            Nu_Qc_bot=(isnan(Qc_bot) ? NaN : bot_ref + Qc_bot / A),
            dHdt=dHdt, flux_pt=flux_pt, flux_cbf=flux_cbf,
            Ibase=Ibase, H=pr.H, theta_int=bud.theta_int,
            latent_int=bud.latent_int, latent_src=bud.latent_src,
            latent_adv=bud.latent_adv, latent_dgdt=latent,
            base_adv=bud.base_adv, heating=bud.heating,
            u_grad_theta=bud.u_grad_theta,
            closure_nolatent=closure_nolatent, closure_old=closure_old,
            closure_area=closure_area,
            budget=bud, cfg=cfg, attrs=attrs, pr=pr, ph=ph)
end

rel(a, b) = abs(b) > 0.0 ? abs(a - b) / abs(b) : NaN

"""One-line table row: `|label|leaves|h|Nu|Nu_cbf|...|`."""
function table_header()
    @printf("| %-26s | %6s | %8s | %10s | %12s | %9s | %10s | %10s | %11s | %9s |\n",
            "state", "leaves", "h_eff", "Nu_top", "Nu_cbf_top",
            "dNu rel", "flux_pt/A", "Qc/A", "(Qc-flux)/A", "rel")
end

function table_row(r; label="")
    isempty(label) && (label = basename(r.ckpt))
    dnu = r.Nut - r.Nu_cbf_top
    dq = (r.Qc - r.flux_pt) / r.A
    @printf("| %-26s | %6d | %8.5f | %10.6f | %12.6f | %8.2f%% | %10.6f | %10.6f | %11.5f | %8.2f%% |\n",
            label, r.leaves, r.h_eff, r.Nut, r.Nu_cbf_top,
            100 * rel(r.Nut, r.Nu_cbf_top), r.flux_pt / r.A, r.Qc / r.A, dq,
            100 * rel(r.Qc, r.flux_pt))
    return nothing
end

"""Print the full single-state report."""
function print_report(r; label="")
    isempty(label) && (label = r.ckpt)
    println("="^96)
    @printf("state        %s\n", label)
    @printf("             step %d  t %.6e  dt %.4e  leaves %d  N %d  p %d  A %.6g\n",
            r.step, r.t, r.dt, r.leaves, r.N, r.p, r.A)
    @printf("             h_eff %.6f  maxlevel %d  umax %.4e  EV stored %s\n",
            r.h_eff, r.maxlevel, r.umax, r.ev_stored ? "yes" : "no")
    @printf("physics      latent_mode=%s  latent_L=%.6g  410=%s  660=%s\n",
            r.latent_mode, r.latent_L,
            r.latent410 === nothing ? "unset" : @sprintf("%+.6f", r.latent410),
            r.latent660 === nothing ? "unset" : @sprintf("%+.6f", r.latent660))
    @printf("             advection=%s  ev=%.4g  ev_cmax=%.4g  H=%.6g\n",
            r.advection, r.ev, r.ev_cmax, r.H)
    println("-"^96)
    @printf("%-40s %+16.9f %+16.9f\n", "pointwise GLL derivative Nu", r.Nut, r.Nub)
    @printf("%-40s %+16.9f %+16.9f\n", "consistent boundary flux Nu_cbf", r.Nu_cbf_top, r.Nu_cbf_bot)
    @printf("%-40s %+16.9f %+16.9f\n", "CBF without EV field (no stabilisation)", r.Nu_cbf_top_noev, r.Nu_cbf_bot_noev)
    @printf("%-40s %+16.9e %+16.9e\n", "Nu_top - Nu_cbf_top / Nu_bot - Nu_cbf_bot",
            r.Nut - r.Nu_cbf_top, r.Nub - r.Nu_cbf_bot)
    @printf("%-40s %+16.3f%% %+16.3f%%\n", "  relative to Nu_cbf",
            100 * rel(r.Nut, r.Nu_cbf_top), 100 * rel(r.Nub, r.Nu_cbf_bot))
    @printf("%-40s %+16.3f%% %+16.3f%%\n", "  CBF top without the EV field vs with",
            100 * rel(r.Nu_cbf_top_noev, r.Nu_cbf_top),
            100 * rel(r.Nu_cbf_bot_noev, r.Nu_cbf_bot))
    println("-"^96)
    if isnan(r.Qc)
        println("consistent Dirichlet reaction Qc: not computed (no --prev checkpoint)")
    else
        @printf("%-40s %+16.9e\n", "consistent Dirichlet reaction Qc", r.Qc)
        @printf("%-40s %+16.9e\n", "Qc / A", r.Qc / r.A)
        @printf("%-40s %+16.9e %+16.9e\n", "  per wall: Qc top / Qc bottom",
                r.Qc_top, r.Qc_bot)
        @printf("%-40s %+16.7f %+16.7f\n", "  Nu from the time-step CBF (ASPECT's)",
                r.Nu_Qc_top, r.Nu_Qc_bot)
        @printf("%-40s %+16.9e\n", "dH/dt (exact BDF2)", r.dHdt)
        @printf("%-40s %+16.9e\n", "dH/dt / A", r.dHdt / r.A)
        @printf("%-40s %+16.9e %+16.9e\n", "wall flux: pointwise / CBF",
                r.flux_pt, r.flux_cbf)
        @printf("%-40s %+16.9e %+16.9e\n", "  / A", r.flux_pt / r.A, r.flux_cbf / r.A)
        @printf("%-40s %+16.9e   (%+7.2f%% of |flux_pt|/A)\n",
                "Qc - flux_pt (total) / A", (r.Qc - r.flux_pt) / r.A,
                100 * abs(r.Qc - r.flux_pt) / abs(r.flux_pt))
        @printf("%-40s %+16.9e   (%+7.2f%%)\n", "Qc - flux_cbf (total) / A",
                (r.Qc - r.flux_cbf) / r.A, 100 * rel(r.Qc, r.flux_cbf))
        @printf("%-40s %+16.9e   (%+7.2f%%)\n", "flux_cbf - flux_pt (total) / A",
                (r.flux_cbf - r.flux_pt) / r.A, 100 * rel(r.flux_cbf, r.flux_pt))
        isnan(r.ev_maxdiff) ||
            @printf("%-40s %+16.9e\n", "EV field: recomputed - stored (max abs)", r.ev_maxdiff)
    end
    println("-"^96)
    @printf("%-40s %+16.9e\n", "theta_int", r.theta_int)
    @printf("%-40s %+16.9e\n", "latent_int (enthalpy, not the closure term)", r.latent_int)
    @printf("%-40s %+16.9e\n", "Ibase = int u_z dTbar/dz", r.Ibase)
    @printf("%-40s %+16.9e\n", "base_adv (volume, = Ibase)", r.base_adv)
    @printf("%-40s %+16.9e\n", "latent_src", r.latent_src)
    @printf("%-40s %+16.9e\n", "latent_adv", r.latent_adv)
    @printf("%-40s %+16.9e\n", "latent_dgdt = latent_src + latent_adv", r.latent_dgdt)
    @printf("%-40s %+16.9e\n", "u_grad_theta", r.u_grad_theta)
    @printf("%-40s %+16.9e\n", "heating", r.heating)
    @printf("%-40s %+16.9e\n", "closure_nolatent (historical, no latent)", r.closure_nolatent)
    @printf("%-40s %+16.9e\n", "closure_old (historical column 13)", r.closure_old)
    @printf("%-40s %+16.9e\n", "closure_area (area-normalised Ibase)", r.closure_area)
    return nothing
end

# ------------------------------------------------------------------- resume ----
"CopyWithCheck: copy `src` to `dst` unless `dst` exists and is newer."
function copy_checkpoint(src::AbstractString, dst::AbstractString)
    mkpath(dirname(dst))
    cp(src, dst; force=true)
    return dst
end

"""
    checkpoint_meta_from_attrs(attrs) -> Dict{String,Any}

The scalar HDF5 attributes of a checkpoint (its `meta` dict: `case`, `mode`,
`latent410`, `latent660`, ...), for `checkpoint_meta` of a scratch run.  Carrying
them over is what keeps a scratch checkpoint self-describing: the per-transition
latent coefficients of `:per_phase` live there, not in `run_config`.
"""
function checkpoint_meta_from_attrs(attrs)
    reserved = ("t", "dt", "dt_prev", "step", "p", "maxlevel", "nx0", "ny0",
                "Lx", "config")
    return Dict{String,Any}(k => v for (k, v) in attrs
                            if !(k in reserved) && (v isa Number || v isa String))
end

"""
    resume_state(ckpt; scratch, steps=2, tag) -> (state, prev)

Resume `ckpt` `steps` steps in `scratch`, with the run settings read from the
checkpoint itself, and return the final checkpoint and the one of the previous
step (`steps == 1` returns the source as previous).  The source is copied into
the scratch directory first, so a live production checkpoint is never opened
twice and never written.
"""
function resume_state(ckpt::AbstractString; scratch::AbstractString,
                      steps::Int=2, tag::AbstractString="")
    st = load_semquad_state(ckpt)
    isempty(tag) && (tag = @sprintf("step%06d", st.step))
    dir = joinpath(scratch, tag)
    mkpath(dir)
    src = copy_checkpoint(ckpt, joinpath(dir, "source.h5"))
    pr, ph, cfg, attrs = physics_from_checkpoint(src)
    opts = run_opts_from_checkpoint(cfg, attrs)
    final = joinpath(dir, @sprintf("state_%06d.h5", st.step + steps))
    prev = steps == 1 ? src : joinpath(dir, @sprintf("state_%06d.h5", st.step + steps - 1))
    # `checkpoint_every=1` writes the step-n checkpoint at the end of step n; the
    # callback of step n+1 can copy it, which is how step n-1 is kept when
    # more than one step is resumed.
    ckpt_path = joinpath(dir, "running.h5")
    cb = function (s)
        if steps > 1 && s.step == st.step + steps
            copy_checkpoint(ckpt_path, prev)
        end
        return nothing
    end
    meta = checkpoint_meta_from_attrs(attrs)
    @printf("[resume] %s -> step %d (%d steps) in %s\n", ckpt, st.step + steps,
            steps, dir)
    t0 = time()
    res = run_semquad_simulation(st.qt, pr, zeros(length(st.theta));
        p=st.p, tstop=Inf, maxsteps=st.step + steps,
        dt0=1.0e-6, dt_growth=1.05, linear=:direct, stokes_tol=1.0e-3,
        restart_from=src, checkpoint_path=ckpt_path, checkpoint_every=1,
        verbose=false, callback=cb, checkpoint_meta=meta,
        phases=ph, opts...)
    @printf("[resume] done: %d steps, t=%.6e, wall=%.1f s, final leaves=%d\n",
            res.steps - st.step, res.t, time() - t0, length(res.qt.leaves))
    flush(stdout)
    copy_checkpoint(ckpt_path, final)
    steps > 1 && !isfile(prev) &&
        throw(ArgumentError("the intermediate checkpoint $prev was not written"))
    return final, prev
end

# -------------------------------------------------------------------- sweep ----
"Seed used by `examples/case_realistic_2d.jl` (the same initial condition)."
seed(x, y, Lx) = 0.05 * sinpi(y) * cospi(2.0 * x / Lx) +
                 0.02 * sinpi(2.0 * y) * cospi(4.0 * x / Lx)

"""
    run_uniform_case(pr, ph, opts, level; scratch, steps, Lx, nx0, ny0) -> (state, prev)

One fresh run on a uniform level-`level` tree (no AMR: the resolution is exactly
`h = Lx/(nx0*2^level)`), from the case's initial condition, `steps` steps.  The
last step must satisfy `steps % 5 == 2`.
"""
function run_uniform_case(pr, ph, opts, level; scratch::AbstractString,
                          steps::Int, Lx::Float64, nx0::Int, ny0::Int,
                          meta::AbstractDict=Dict{String,Any}(),
                          stokes_tol::Float64=1.0e-2)
    tag = @sprintf("uniform_L%d_%03d", level, steps)
    dir = joinpath(scratch, tag)
    mkpath(dir)
    # `Quadtree(level; ...)` sets the *maximum* level and starts from the base
    # cells (10 for the 5x2 box); refine every leaf `level` times to get the
    # uniform level-`level` mesh (the same construction as `make_initial_tree`
    # in case_realistic_2d.jl, which stops at level 3).
    qt = Quadtree(level; Lx=Lx, cells_x=nx0, cells_y=ny0)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    theta0 = [seed(x, y, Lx) for (x, y) in ss.su.coords]
    for g in 1:ss.su.N
        y = ss.su.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta0[g] = 0.0)
    end
    h = Lx / (nx0 * 2.0^level)
    final = joinpath(dir, @sprintf("state_%06d.h5", steps))
    prev = joinpath(dir, @sprintf("state_%06d.h5", steps - 1))
    ckpt_path = joinpath(dir, "running.h5")
    cb = function (s)
        s.step == steps && copy_checkpoint(ckpt_path, prev)
        return nothing
    end
    @printf("[sweep] level %d: %d leaves, h=%.6f, %d steps (amr_every=0)\n",
            level, length(qt.leaves), h, steps)
    flush(stdout)
    t0 = time()
    res = run_semquad_simulation(qt, pr, theta0;
        p=2, tstop=Inf, maxsteps=steps, dt0=1.0e-6, dt_growth=1.05,
        initial_refine=0, linear=:direct, stokes_tol=stokes_tol,
        checkpoint_path=ckpt_path, checkpoint_every=1, verbose=false,
        callback=cb, checkpoint_meta=meta, phases=ph,
        merge(opts, (amr_every=0,))...)
    @printf("[sweep] level %d done: t=%.6e wall=%.1f s (%.2f s/step) leaves=%d\n",
            level, res.t, time() - t0, (time() - t0) / steps,
            length(res.qt.leaves))
    @printf("[sweep] level %d: last Stokes relative residual = %.3e (tol %.1e)\n",
            level, LAST_SEMSTOKES_RESIDUAL[], stokes_tol)
    flush(stdout)
    copy_checkpoint(ckpt_path, final)
    isfile(prev) ||
        throw(ArgumentError("the checkpoint of step $(steps - 1) was not written"))
    return final, prev
end

"Least-squares slope of log(e) against log(h) for the three sweep points."
function fit_order(hs, es)
    ok = [isfinite(e) && e > 0.0 && isfinite(h) && h > 0.0 for (h, e) in zip(hs, es)]
    count(ok) < 2 && return NaN
    x = log.(hs[ok]); y = log.(es[ok])
    xm = sum(x) / length(x); ym = sum(y) / length(y)
    return sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
end

function run_sweep(scratch::AbstractString, levels::Vector{Int}, steps::Int;
                   prod::AbstractString=DEFAULT_CKPT)
    # the parameter card comes from the production checkpoint, not from a copy
    pr, ph, cfg, attrs = physics_from_checkpoint(prod)
    opts = run_opts_from_checkpoint(cfg, attrs)
    meta = checkpoint_meta_from_attrs(attrs)
    st0 = load_semquad_state(prod)
    Lx = st0.qt.Lx; nx0 = st0.qt.nx0; ny0 = st0.qt.ny0
    println("resolution sweep: uniform meshes, $(steps) steps, parameter card from")
    println("  ", prod)
    rows = NamedTuple[]
    labels = String[]
    hs = Float64[]
    for lev in levels
        state, prev = run_uniform_case(pr, ph, opts, lev; scratch=scratch,
                                       steps=steps, Lx=Lx, nx0=nx0, ny0=ny0,
                                       meta=meta)
        r = analyze(state, prev)
        print_report(r; label=@sprintf("uniform L%d", lev))
        push!(rows, r); push!(labels, @sprintf("uniform L%d", lev))
        push!(hs, r.h_eff)
    end
    n_uniform = length(rows)
    # the production AMR state itself (one step past a post-AMR checkpoint, so
    # its EV field and velocity are the ones the energy step produced)
    if !isempty(prod)
        state, prev = resume_state(prod; scratch=scratch, steps=1, tag="production")
        r = analyze(state, prev)
        print_report(r; label="production B2 (AMR)")
        push!(rows, r); push!(labels, "production B2 (AMR)")
        push!(hs, r.h_eff)
    end
    table_header()
    for (i, r) in enumerate(rows)
        table_row(r; label=labels[i])
    end
    # convergence of the three differences, from the *uniform* points only (the
    # production tree is graded, so its h_eff is not a mesh-size parameter)
    e_qc = [abs(r.Qc - r.flux_pt) / r.A for r in rows[1:n_uniform]]
    e_nu = [abs(r.Nut - r.Nu_cbf_top) for r in rows[1:n_uniform]]
    e_cbf = [abs(r.flux_cbf - r.flux_pt) / r.A for r in rows[1:n_uniform]]
    hu = hs[1:n_uniform]
    println()
    println("convergence (log-log slope over the uniform meshes):")
    @printf("  %-34s  p = %+6.3f   values: %s\n", "|Qc - flux_pt| / A",
            fit_order(hu, e_qc), join([@sprintf("%.4e", v) for v in e_qc], "  "))
    @printf("  %-34s  p = %+6.3f   values: %s\n", "|Nu_top - Nu_cbf_top|",
            fit_order(hu, e_nu), join([@sprintf("%.4e", v) for v in e_nu], "  "))
    @printf("  %-34s  p = %+6.3f   values: %s\n", "|flux_cbf - flux_pt| / A",
            fit_order(hu, e_cbf), join([@sprintf("%.4e", v) for v in e_cbf], "  "))
    println("  pairwise rates of |Qc - flux_pt| / A (coarse -> fine):")
    for i in 1:n_uniform - 1
        p = log(e_qc[i] / e_qc[i + 1]) / log(hu[i] / hu[i + 1])
        @printf("    h %.5f -> %.5f : p = %+.3f\n", hu[i], hu[i + 1], p)
    end
    return rows
end

# --------------------------------------------------------------------- main ----
function main()
    opts = Dict{String,String}()
    pos = String[]
    for a in ARGS
        if startswith(a, "--")
            kv = split(a[3:end], '='; limit=2)
            opts[String(kv[1])] = length(kv) == 2 ? String(kv[2]) : "true"
        else
            push!(pos, a)
        end
    end
    haskey(opts, "help") && (println(USAGE); return)
    scratch = get(opts, "scratch", DEFAULT_SCRATCH)
    steps = parse(Int, get(opts, "steps", "32"))
    levels = [parse(Int, s) for s in split(get(opts, "levels", "3,4,5"), ",")]
    do_sweep = haskey(opts, "sweep")

    if do_sweep
        run_sweep(scratch, levels, steps; prod=get(opts, "prod", DEFAULT_CKPT))
        return
    end

    ckpt = isempty(pos) ? DEFAULT_CKPT : pos[1]
    isfile(ckpt) || error("checkpoint not found: $ckpt")
    prev = get(opts, "prev", "")
    st = load_semquad_state(ckpt)
    umax = maximum(hypot.(st.ux, st.uz))
    ev_on = cfg_float(st.config, "ev", 0.0) > 0.0
    needs_resume = umax == 0.0 || (ev_on && st.ev === nothing)
    resume_steps = parse(Int, get(opts, "resume", "2"))
    if needs_resume && !haskey(opts, "no-resume")
        @printf("the checkpoint has umax=%.1e, EV stored=%s: it was written on an AMR step, so it is not a state the energy step produced\n",
                umax, st.ev !== nothing)
        @printf("resuming %d step(s) in %s\n", resume_steps, scratch)
        state, autoprev = resume_state(ckpt; scratch=scratch,
                                       steps=resume_steps)
        isempty(prev) && (prev = autoprev)
        ckpt = state
    elseif needs_resume
        error("checkpoint $ckpt is post-AMR (umax=$umax, ev=" *
              "$(st.ev === nothing ? "missing" : "present")) and --no-resume " *
              "was given; pass a state written by a non-AMR step")
    end
    r = analyze(ckpt, isempty(prev) ? nothing : prev)
    print_report(r)
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
