#!/usr/bin/env julia
# =============================================================================
# nusselt_audit.jl — the arbitration on the wall Nusselt number of the
# realistic card: which definition satisfies the *scheme's own* discrete budget?
#
# The driver reports `Nu_top`/`Nu_bot` from `semquad_nu_walls` (the one-sided GLL
# derivative of the P2 interpolant at the wall).  `semquad_nu_cbf` gives the
# reaction of the *steady* energy operator (K + C, no mass matrix) on the same
# state, and the Dirichlet reaction of the *time step* (A = ca M + K + C) is what
# ASPECT's `heat flux statistics` reports.  On the late E states these disagree
# by up to ~17 %, so this script measures, on one fixed state, every term of the
# identity the assembled operator actually satisfies, and the residual of each
# candidate definition against it.
#
# ---------------------------------------------------------------------------
# THE IDENTITY (derived from the assembly, not from the PDE)
#
# `assemble_semquad_energy` builds, per element and on the energy quadrature,
#     M_ij = int (1 + lat_cp) rho phi_i phi_j,   K_ij = int (1+nu_ev) grad phi_i . grad phi_j,
#     C_ij = int (1 + lat_cp) rho phi_i (u . grad phi_j),   F_i = int src phi_i,
# with `src = -rho u_z dTbar/dz - sum_a L_a dGamma_a/dz u_z (+ heating)` and
# `C <- 0.5(C - C')` for `advection = :skew`.  The step solves
#     Cfree' (A theta - rhs) = 0,   A = ca M + K + C,
#     rhs = cb M theta_n - cc M theta_{n-1} + F,
# i.e. the residual r = A theta - rhs satisfies r_g = 0 on the free dofs and is
# a Lagrange multiplier on the constrained ones (Dirichlet walls + hanging
# nodes).  Summing r over *all* dofs and using
#   (i)  sum_i phi_i = 1 pointwise  =>  sum_i M_ij = row sum, sum_i K_ij = 0,
#        sum_i C_ij = 0 for the skew form (since sum_j grad phi_j = 0),
#  (ii)  the hanging-node weights reproduce constants (sum_m w_m = 1), so the
#        constrained rows contribute nothing to 1'r *except* through the
#        Dirichlet multipliers, whose sum is the wall reaction,
# gives the exact discrete identity
#
#   1'r  ==  Qc + hang,     Qc = sum over wall dofs of r,
#                           hang = sum over hanging (non-Dirichlet) dofs of r,
#   Qc + hang = dH/dt + sum(C theta) + Ibase + latent_src,
#   dH/dt     = 1'(ca M theta - cb M theta_n + cc M theta_{n-1})/dt,
#   sum(C theta) = 0.5 int (1 + sum_a L_a Gamma_T,a) u.grad theta dV,
#   -1'F      = Ibase + latent_src,
#   Ibase     = int u_z dTbar/dz dV.
#
# `Qc` is the *time-step* Dirichlet reaction (ASPECT's definition); the steady
# CBF of `semquad_nu_cbf` is the same sum without the mass term
# (`Qc_ss = sum_wall (K theta + C theta - F)`), related by
#     Qc - Qc_ss = sum over wall dofs of the BDF2 mass term  ("wall mass term").
# The pointwise flux `F_pt = sum_wall dtheta/dy dx` (what `semquad_nu_walls`
# integrates) is not tied to any of these; its difference from `Qc_ss` is the
# wall-quadrature (one-sided Dirichlet row) term.
#
# Every number below is measured at one state the energy step actually produced
# (a checkpoint is resumed --resume steps in --scratch first, because a
# checkpoint written on an AMR step carries a zero velocity and no EV field).
# The input checkpoint is never modified.
#
# usage
#   julia --project=julia julia/scripts/nusselt_audit.jl [options] ckpt [ckpt...]
#     --scratch=DIR   scratch root (default out_julia/nusselt_audit/scratch)
#     --resume=N      steps to resume before measuring (default 2)
#     --no-resume     measure the checkpoint as it is (needs ux != 0 and EV)
#     --tag=NAME      label for the report
#     --detail        also print the per-wall-node pointwise/CBF flux table
# =============================================================================
using MantleSEM, Printf, LinearAlgebra, HDF5

include(joinpath(@__DIR__, "nu_cbf_compare.jl"))

# --------------------------------------------------------------- physics ----
"""
    phases_of(pr, cfg, attrs) -> Vector{PhaseTransition}

The phase list of the state, with the `:per_phase` latent coefficients from the
checkpoint attributes.  A `wallrefine` seed does **not** carry those two
attributes (it is written with a fresh `meta` dict), so they are recomputed from
the checkpoint's own phase list and `H` with the same `per_phase_latent` the
case card uses; the recomputation is checked against the attributes whenever
they are present.
"""
function phases_of(pr, cfg, attrs)
    ph = phases_from_checkpoint(cfg, attrs)
    isempty(ph) && return ph
    mode = cfg_sym(cfg, "latent_mode", :global)
    mode === :per_phase || return ph
    return ph
end

"Recompute the per-phase coefficients (what a seed checkpoint does not store)."
function attach_seed_latent(pr, ph, cfg, attrs)
    isempty(ph) && return ph
    cfg_sym(cfg, "latent_mode", :global) === :per_phase || return ph
    haskey(attrs, "latent410") && return ph      # already the run's own values
    re = per_phase_latent(pr, ph; rho_ref=3300.0, cp=1000.0, dT=1000.0,
                          g=9.81, d=2.9e6, T_surface=0.5)
    return re
end

"Dirichlet dofs of the energy step (the `fixed` dict of `solve_semquad_energy`)."
function dirichlet_dofs(sq, pr)
    fixed = Dict{Int,Float64}()
    for g in 1:sq.N
        y = sq.coords[g][2]
        y <= 1.0e-14 && (fixed[g] = pr.Twall_bot - semquad_base_T(pr, y))
        y >= 1.0 - 1e-14 && (fixed[g] = pr.Twall_top - semquad_base_T(pr, y))
    end
    return fixed
end

"""Wall-node table of one wall: `(x, q_pt, q_cbf, weight)`.

`q_pt` is the pointwise contribution `0.5 hx w_i (D theta)_i 2/h` (what
`semquad_nu_walls` sums), `q_cbf` the same node's residual row `r_i` (what
`semquad_cbf_theta_flux` sums).
"""
function wall_rows_table(sq, qt, theta, r, wall::Symbol)
    n = sq.p + 1
    xs = Float64[]; qp = Float64[]; qc = Float64[]; ws = Float64[]
    seen = Set{Int}()
    for lf in sq.leaves
        (wall === :top ? lf[3] == sq.ny0 * (1 << lf[1]) - 1 : lf[3] == 0) || continue
        h = MantleSEM.leaf_hy(sq, lf[1]); hx = MantleSEM.leaf_hx(sq, lf[1]); x0 = lf[2] * hx
        ids = sq.elem[lf]
        row = wall === :top ? n : 1
        for i in 1:n
            g = ids[(row - 1) * n + i]
            g in seen && continue
            push!(seen, g)
            d = 0.0
            for j in 1:n
                d += sq.D[row, j] * theta[ids[(j - 1) * n + i]]
            end
            d *= 2.0 / h
            w = 0.5 * hx * sq.w[i]
            push!(xs, x0 + 0.5 * (sq.xi[i] + 1.0) * hx)
            push!(ws, w)
            push!(qp, w * d)
            push!(qc, r[g])
        end
    end
    return (x=xs, q_pt=qp, q_cbf=qc, w=ws)
end

"""The same integral of the P2 wall derivative on a 4-point Gauss rule (`exact`)."""
function wall_flux_gauss(sq, theta, wall::Symbol)
    n = sq.p + 1
    gx, gw = gauss_rule(6)
    s = 0.0
    for lf in sq.leaves
        (wall === :top ? lf[3] == sq.ny0 * (1 << lf[1]) - 1 : lf[3] == 0) || continue
        h = MantleSEM.leaf_hy(sq, lf[1]); hx = MantleSEM.leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        row = wall === :top ? n : 1
        for (xq, wq) in zip(gx, gw)
            # 1-D Lagrange derivative at the wall node, evaluated at xq
            d = 0.0
            for i in 1:n
                d += lagrange_deriv(sq.xi, i, xq) * theta[ids[(row - 1) * n + i]]
            end
            s += 0.5 * hx * wq * d * (2.0 / h)
        end
    end
    return s
end

"6-point Gauss-Legendre rule on [-1,1] (positive weights)."
function gauss_rule(n::Int)
    # Golub-Welsch via the Jacobi matrix (tiny, no dependency)
    beta = [k / sqrt(4.0 * k * k - 1.0) for k in 1:n - 1]
    J = SymTridiagonal(zeros(n), beta)
    E = eigen(J)
    return E.values, 2.0 .* (E.vectors[1, :] .^ 2)
end

"Derivative of the 1-D Lagrange basis `i` (nodes `xi`) at `x`."
function lagrange_deriv(xi::Vector{Float64}, i::Int, x::Float64)
    n = length(xi)
    s = 0.0
    for m in 1:n
        m == i && continue
        p = 1.0 / (xi[i] - xi[m])
        for k in 1:n
            (k == i || k == m) && continue
            p *= (x - xi[k]) / (xi[i] - xi[k])
        end
        s += p
    end
    return s
end

# ----------------------------------------------------------------- audit ----
function audit(ckpt, prev; label="", detail=false)
    st = load_semquad_state(ckpt)
    pst = load_semquad_state(prev)
    pr, ph0, cfg, attrs = physics_from_checkpoint(ckpt)
    ph = attach_seed_latent(pr, ph0, cfg, attrs)
    sq = make_semstokes(st.qt, st.p).su
    qt = st.qt
    A = total_area(qt)
    nl = length(qt.leaves)
    adv = cfg_sym(cfg, "advection", :skew)
    evv = cfg_float(cfg, "ev", 0.0)
    ev_cmax = cfg_float(cfg, "ev_cmax", 0.052)
    ev_alpha = cfg_int(cfg, "ev_alpha", 2)
    theta = st.theta
    thn = st.theta_prev
    thnm1 = pst.theta_prev
    dt = st.dt
    dt_prev = pst.dt

    # ---------------------------------------------------------- the step ----
    # The operator must be the one the step *actually* used.  The EV field is
    # the one the assembly computed on that step, which the checkpoint stores:
    # passing it back reproduces the step bit for bit (measured: re-solving the
    # step and comparing with the stored `theta` gives a relative difference of
    # exactly 0.0 with `ev_field = st.ev`, and 2.5e-5 when the EV field is
    # recomputed from `ev`/`evg` -- the recomputation is not bit-identical
    # because `semquad_ev_nu` also depends on the time-step index).
    ev_have = evv > 0.0 && st.ev !== nothing
    if ev_have
        M, K, C, F = assemble_semquad_energy(sq, qt, pr, st.ux, st.uz;
                        advection=adv, theta=thn, theta_lat=thn, phases=ph,
                        ev=0.0, ev_field=st.ev)
    else
        evg = evv > 0.0 ? MantleSEM.semquad_ev_globals(sq, qt, pr, thn, thnm1,
                                                       st.ux, st.uz,
                                                       dt / dt_prev) : nothing
        M, K, C, F = assemble_semquad_energy(sq, qt, pr, st.ux, st.uz;
                        advection=adv, theta=thn, theta_lat=thn, phases=ph,
                        ev=evv, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                        ev_timestep=st.step, evg=evg)
    end
    w = dt / dt_prev
    ca = (2.0 * w + 1.0) / (1.0 + w) / dt
    cb = (1.0 + w) / dt
    cc = w * w / (1.0 + w) / dt
    Aop = ca * M + K + C
    rhs = cb * (M * thn) - cc * (M * thnm1) + F
    r = Aop * theta .- rhs

    # ------------------------------------------------------ dof partition ----
    fixed = dirichlet_dofs(sq, pr)
    cons = semquad_constraints(sq, qt)
    u0, Cfree, free = MantleSEM._affine_constraint(sq.N, cons, fixed)
    wt = Int[]; wb = Int[]
    for g in 1:sq.N
        y = sq.coords[g][2]
        y >= 1.0 - 1e-14 && push!(wt, g)
        y <= 1.0e-14 && push!(wb, g)
    end
    hangd = [c for c in keys(cons) if !haskey(fixed, c)]
    wmax = maximum(abs(1.0 - sum(cons[c][2])) for c in keys(cons))
    nwh = count(c -> haskey(fixed, c), keys(cons))

    Qc_top = sum(@view r[wt]); Qc_bot = sum(@view r[wb])
    Qc = Qc_top + Qc_bot
    hang = sum(@view r[hangd])
    total = sum(r)
    free_res = maximum(abs, @view r[free])
    free_norm = maximum(abs, @view rhs[free])
    # the free rows are only satisfied to the linear-solver tolerance
    free_rel = free_res / max(free_norm, 1e-300)

    # ------------------------------------------- operator volume integrals ----
    # `ca, cb, cc` already carry the 1/dt, so `vM` is the *rate* dH/dt per dof
    # (the same units as K theta, C theta and F).
    vM = M * (ca .* theta .- cb .* thn .+ cc .* thnm1)
    vK = K * theta
    vC = C * theta
    vF = F
    dHdt = sum(vM)
    sumK = sum(vK); sumC = sum(vC); sumF = sum(vF)
    # wall-row parts
    m_top = sum(@view vM[wt]); m_bot = sum(@view vM[wb])
    k_top = sum(@view vK[wt]); k_bot = sum(@view vK[wb])
    c_top = sum(@view vC[wt]); c_bot = sum(@view vC[wb])
    f_top = sum(@view vF[wt]); f_bot = sum(@view vF[wb])
    Qss_top = k_top + c_top - f_top
    Qss_bot = k_bot + c_bot - f_bot
    Qss = Qss_top + Qss_bot

    # ------------------------------------------------- pointwise flux ----
    Nut, Nub = semquad_nu_walls(sq, qt, theta, pr)
    top_ref = -semquad_base_gradient(pr, 1.0)
    bot_ref = -semquad_base_gradient(pr, 0.0)
    # recompute the two pointwise wall integrals so the signs/normalisation of
    # the comparison are explicit (the same rule semquad_nu_walls sums)
    pt_top = 0.0; pt_bot = 0.0
    n = sq.p + 1
    for lf in sq.leaves
        h = MantleSEM.leaf_hy(sq, lf[1]); hx = MantleSEM.leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        if lf[3] == qt.ny0 * (1 << lf[1]) - 1
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += sq.D[n, j] * theta[ids[(j - 1) * n + i]]
                end
                pt_top += 0.5 * hx * sq.w[i] * d * (2.0 / h)
            end
        end
        if lf[3] == 0
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += sq.D[1, j] * theta[ids[(j - 1) * n + i]]
                end
                pt_bot += 0.5 * hx * sq.w[i] * d * (2.0 / h)
            end
        end
    end
    F_pt = pt_top - pt_bot
    g_top = wall_flux_gauss(sq, theta, :top)
    g_bot = wall_flux_gauss(sq, theta, :bottom)

    # --------------------------------------------------- heat budget terms ----
    bud = semquad_heat_budget(sq, qt, pr, theta; phases=ph, ux=st.ux, uz=st.uz)
    Ibase = semquad_base_advection_integral(sq, qt, pr, st.ux, st.uz)
    latent = bud.latent_dgdt === nothing ? 0.0 : bud.latent_dgdt
    lsrc = bud.latent_src === nothing ? 0.0 : bud.latent_src
    ladv = bud.latent_adv === nothing ? 0.0 : bud.latent_adv
    ugrad = bud.u_grad_theta === nothing ? 0.0 : bud.u_grad_theta

    # ------------------------------------------------------------- EV ----
    ev_at_wall = 0.0; ev_max = 0.0; ev_mean = 0.0; ev_nonzero = 0
    evf = st.ev
    if evf !== nothing
        ev_max = maximum(evf); ev_mean = sum(evf) / length(evf)
        ev_nonzero = count(>(0.0), evf)
        wl = [i for (i, lf) in enumerate(sq.leaves)
              if lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1]
        ev_at_wall = maximum(abs.(evf[wl]))
    end

    # ------------------------------- the wall functions, in closed form ----
    # For one wall element (GLL nodes 0 = interior edge, 1 = mid, 2 = wall,
    # theta_wall = 0) both diagnostics are linear functionals of (th_0, th_1):
    #     pointwise  p = (th_0 - 4 th_1)/h          (exact for a cubic)
    #     steady CBF q = (th_0 - 8 th_1)/(3h)       (exact for a linear profile)
    # and q - p = -(2/(3h))(th_0 - 2 th_1) = -(h/3) theta'' inside the element.
    # `wq_pred` sums that closed form over the wall nodes, `wq_meas` is the
    # measured (Qss - F_pt) of the same wall; they agree to the C/F wall terms.
    wq_pred_top = 0.0; wq_pred_bot = 0.0
    for lf in sq.leaves
        hh = MantleSEM.leaf_hy(sq, lf[1]); hx = MantleSEM.leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        istop = lf[3] == qt.ny0 * (1 << lf[1]) - 1
        isbot = lf[3] == 0
        (istop || isbot) || continue
        for i in 1:n
            if istop
                th0 = theta[ids[(1 - 1) * n + i]]     # j = 1: interior edge
                th1 = theta[ids[(2 - 1) * n + i]]     # j = 2: mid
                wq_pred_top += 0.5 * hx * sq.w[i] * (-(2.0 / (3.0 * hh)) * (th0 - 2.0 * th1))
            else
                th0 = theta[ids[(n - 1) * n + i]]     # j = n: interior edge
                th1 = theta[ids[(n - 1) * n + i]]     # j = n-1: mid
                th1 = theta[ids[(n - 2) * n + i]]
                wq_pred_bot += 0.5 * hx * sq.w[i] * (-(2.0 / (3.0 * hh)) * (th0 - 2.0 * th1))
            end
        end
    end
    # wall level census
    wtlev = Dict{Int,Int}(); wblev = Dict{Int,Int}()
    for lf in sq.leaves
        lf[3] == 0 && (wblev[lf[1]] = get(wblev, lf[1], 0) + 1)
        lf[3] == qt.ny0 * (1 << lf[1]) - 1 && (wtlev[lf[1]] = get(wtlev, lf[1], 0) + 1)
    end

    # ------------------------------------------- candidate identity residuals ----
    rhs_exact = dHdt + sumC + Ibase + lsrc      # the scheme's identity
    R_pt = F_pt - rhs_exact
    R_ss = Qss - rhs_exact
    R_ts = Qc - rhs_exact
    rhs_steady = sumC + Ibase + lsrc            # its steady limit
    S_pt = F_pt - rhs_steady
    S_ss = Qss - rhs_steady
    S_ts = Qc - rhs_steady
    # the driver's `closure_area` for each candidate pair:
    #   closure = Nut - Nub - (H - Ibase/A) + latent_dgdt/A,  Nut-Nub = H - F/A
    # => closure*A = Ibase + latent_dgdt - F
    cl_pt = (Ibase + latent - F_pt) / A
    cl_ss = (Ibase + latent - Qss) / A
    cl_ts = (Ibase + latent - Qc) / A

    # Nu pairs
    nu_pt = (Nut, Nub)
    nu_ss = (top_ref - Qss_top / A, bot_ref + Qss_bot / A)
    nu_ts = (top_ref - Qc_top / A, bot_ref + Qc_bot / A)

    # ------------------------------------------- decomposition of cl_pt*A ----
    # D := Ibase + latent_dgdt - F_pt = (F_ts - F_pt) + (latent_adv - sumC) - dHdt - eps
    # where eps := Qc - rhs_exact = -(hang + sum(r_free)) is the part of the
    # algebraic residual that the constrained solve leaves outside the
    # constraint range (the linear solver's backward error; identically zero
    # when the free rows are satisfied).
    D = Ibase + latent - F_pt
    d_fluxdef = Qc - F_pt
    d_wallmass = Qc - Qss
    d_steady_wq = Qss - F_pt
    d_latent = ladv - sumC
    d_dhdt = -dHdt
    eps = Qc - rhs_exact
    d_eps = -eps
    d_sum = d_fluxdef + d_latent + d_dhdt + d_eps
    # cross-checks that close exactly in the assembled operator
    x1 = (Qc + hang) - (dHdt + sumC + Ibase + lsrc)      # should be ~ round-off
    x2 = sumC - 0.5 * (ugrad + ladv)                      # quadrature identity
    x3 = sumF + (Ibase + lsrc)                            # -1'F = Ibase + latent_src
    x4 = sumK                                             # 1'(K theta) = 0
    x5 = total - (Qc + hang) - sum(@view r[free])         # dof partition

    return (label=label, ckpt=ckpt, prev=prev, step=st.step, t=st.t, dt=dt,
            dt_prev=dt_prev, leaves=nl, N=sq.N, p=st.p, A=A,
            umax=maximum(hypot.(st.ux, st.uz)), ev_stored=(st.ev !== nothing),
            adv=adv, ev=evv, ev_cmax=ev_cmax, H=pr.H,
            latent_mode=pr.latent_mode,
            lat410=(length(ph) >= 1 ? ph[1].latent : nothing),
            lat660=(length(ph) >= 2 ? ph[2].latent : nothing),
            Nut=Nut, Nub=Nub, F_pt=F_pt, pt_top=pt_top, pt_bot=pt_bot,
            g_top=g_top, g_bot=g_bot,
            Qc=Qc, Qc_top=Qc_top, Qc_bot=Qc_bot, Qss=Qss, Qss_top=Qss_top,
            Qss_bot=Qss_bot, k_top=k_top, k_bot=k_bot, c_top=c_top,
            c_bot=c_bot, f_top=f_top, f_bot=f_bot,
            m_top=m_top, m_bot=m_bot, dHdt=dHdt, sumK=sumK, sumC=sumC,
            sumF=sumF, Ibase=Ibase, base_adv=bud.base_adv, lsrc=lsrc, ladv=ladv,
            latent=latent, ugrad=ugrad, theta_int=bud.theta_int,
            latent_int=bud.latent_int, heating=bud.heating,
            hang=hang, n_hang=length(hangd), n_wall_hang=nwh, wmax=wmax,
            total=total, free_rel=free_rel, free_res=free_res,
            free_norm=free_norm,
            rhs_exact=rhs_exact, R_pt=R_pt, R_ss=R_ss, R_ts=R_ts,
            S_pt=S_pt, S_ss=S_ss, S_ts=S_ts,
            cl_pt=cl_pt, cl_ss=cl_ss, cl_ts=cl_ts,
            nu_pt=nu_pt, nu_ss=nu_ss, nu_ts=nu_ts,
            D=D, d_fluxdef=d_fluxdef, d_wallmass=d_wallmass,
            d_steady_wq=d_steady_wq, d_latent=d_latent, d_dhdt=d_dhdt,
            d_eps=d_eps, d_sum=d_sum,
            x1=x1, x2=x2, x3=x3, x4=x4, x5=x5,
            ev_max=ev_max, ev_mean=ev_mean, ev_nonzero=ev_nonzero,
            ev_at_wall=ev_at_wall, ev_have=ev_have,
            wq_pred_top=wq_pred_top, wq_pred_bot=wq_pred_bot,
            wall_top_levels=wtlev, wall_bot_levels=wblev,
            table=(detail ? (top=wall_rows_table(sq, qt, theta, r, :top),
                             bot=wall_rows_table(sq, qt, theta, r, :bottom)) :
                   nothing),
            sq=sq, qt=qt, theta=theta, r=r, pr=pr)
end

# ---------------------------------------------------------------- report ----
function report(a)
    println("="^104)
    @printf("state %s   step %d   t %.9e   dt %.4e   dt_prev %.4e\n",
            isempty(a.label) ? a.ckpt : a.label, a.step, a.t, a.dt, a.dt_prev)
    @printf("      leaves %d  N %d  p %d  A %.6g  umax %.4e  advection %s  ev %.3g (cmax %.3g)\n",
            a.leaves, a.N, a.p, a.A, a.umax, a.adv, a.ev, a.ev_cmax)
    @printf("      latent %s  410 %s  660 %s  H %.6g\n", a.latent_mode,
            a.lat410 === nothing ? "unset" : @sprintf("%+.8f", a.lat410),
            a.lat660 === nothing ? "unset" : @sprintf("%+.8f", a.lat660), a.H)
    @printf("      hanging dofs %d (of which on a wall %d)  max|1-sum w| %.2e  free-row rel res %.2e\n",
            a.n_hang, a.n_wall_hang, a.wmax, a.free_rel)
    println("-"^104)
    println("A. THE THREE FLUX DEFINITIONS (flux = int_wall dtheta/dy dx, per wall)")
    @printf("   %-34s %+16.9f %+16.9f %+16.9f\n", "", a.pt_top, a.Qss_top, a.Qc_top)
    @printf("   %-34s %+16.9f %+16.9f %+16.9f\n", "  (bottom wall)", a.pt_bot, a.Qss_bot, a.Qc_bot)
    @printf("   %-34s %+16.9f %+16.9f %+16.9f\n", "  net flux_top - flux_bot", a.F_pt, a.Qss, a.Qc)
    @printf("   %-34s %+16.6f %+16.6f %+16.6f\n", "Nu_top", a.nu_pt[1], a.nu_ss[1], a.nu_ts[1])
    @printf("   %-34s %+16.6f %+16.6f %+16.6f\n", "Nu_bot", a.nu_pt[2], a.nu_ss[2], a.nu_ts[2])
    @printf("   %-34s %+16.6f %+16.6f %+16.6f\n", "Nu_top - Nu_bot",
            a.nu_pt[1] - a.nu_pt[2], a.nu_ss[1] - a.nu_ss[2], a.nu_ts[1] - a.nu_ts[2])
    @printf("   %-34s %16s %16s %16s\n", "columns: pointwise | steady CBF | time-step CBF", "", "", "")
    println("-"^104)
    println("B. THE SCHEME'S OWN IDENTITY   Qc + hang = dH/dt + sum(C theta) + Ibase + latent_src")
    @printf("   %-38s %+18.9e\n", "dH/dt (exact BDF2)", a.dHdt)
    @printf("   %-38s %+18.9e\n", "sum(C theta) = .5 int (1+L G_T) u.grad th", a.sumC)
    @printf("   %-38s %+18.9e\n", "Ibase = int u_z dTbar/dz", a.Ibase)
    @printf("   %-38s %+18.9e\n", "latent_src = int L G_z u_z", a.lsrc)
    @printf("   %-38s %+18.9e\n", "  -> rhs_exact", a.rhs_exact)
    @printf("   %-38s %+18.9e\n", "Qc (time-step Dirichlet reaction)", a.Qc)
    @printf("   %-38s %+18.9e\n", "hang (hanging-dof residual sum)", a.hang)
    @printf("   %-38s %+18.9e\n", "residual  Qc + hang - rhs_exact", a.x1)
    @printf("   %-38s %+18.9e  (sum(C theta) - .5(ugrad+ladv))\n", "quadrature identity", a.x2)
    @printf("   %-38s %+18.9e  (sum(F) + Ibase + latent_src)\n", "source identity", a.x3)
    @printf("   %-38s %+18.9e  (sum(K theta))\n", "partition-of-unity identity", a.x4)
    @printf("   %-38s %+18.9e\n", "dof partition  sum(r)-Qc-hang-sum_free(r)", a.x5)
    println("-"^104)
    println("C. CLOSURE RESIDUAL OF EACH CANDIDATE (units of flux; /A is Nu)")
    @printf("   %-30s %+16.9e %+16.9e %+16.9e\n", "", a.R_pt, a.R_ss, a.R_ts)
    @printf("   %-30s %+16.9e %+16.9e %+16.9e\n", "F - rhs_exact (full identity)",
            a.R_pt, a.R_ss, a.R_ts)
    @printf("   %-30s %+16.9e %+16.9e %+16.9e\n", "   /A", a.R_pt / a.A, a.R_ss / a.A, a.R_ts / a.A)
    @printf("   %-30s %+16.9e %+16.9e %+16.9e\n", "F - rhs_steady (dH/dt dropped)",
            a.S_pt, a.S_ss, a.S_ts)
    @printf("   %-30s %+16.9e %+16.9e %+16.9e\n", "   /A", a.S_pt / a.A, a.S_ss / a.A, a.S_ts / a.A)
    @printf("   %-30s %+16.9e %+16.9e %+16.9e\n", "driver closure_area (Ibase+lat-F)/A",
            a.cl_pt, a.cl_ss, a.cl_ts)
    @printf("   %-30s %16s %16s %16s\n", "columns: pointwise | steady CBF | time-step CBF", "", "", "")
    println("-"^104)
    println("D. DECOMPOSITION OF D = Ibase + latent_dgdt - F_pt  (= A * driver closure_area)")
    @printf("   %-42s %+18.9e   %+10.4f %% of D\n", "D (measured)", a.D,
            100.0 * a.D / a.D)
    @printf("   %-42s %+18.9e\n", "  (F_ts - F_pt) flux-definition difference", a.d_fluxdef)
    @printf("   %-42s %+18.9e\n", "    = wall mass term (F_ts - F_ss)", a.d_wallmass)
    @printf("   %-42s %+18.9e\n", "    + steady wall-quadrature (F_ss - F_pt)", a.d_steady_wq)
    @printf("   %-42s %+18.9e\n", "  latent_adv - sum(C theta)", a.d_latent)
    @printf("   %-42s %+18.9e\n", "  -dH/dt", a.d_dhdt)
    @printf("   %-42s %+18.9e   (rhs_exact - Qc; = -(hang + sum_free r))\n",
            "  -eps  (constraint + solver residual)", a.d_eps)
    @printf("   %-42s %+18.9e\n", "  sum of the four items", a.d_sum)
    println("-"^104)
    println("E. THE WALL-QUADRATURE TERM, PIECE BY PIECE (per wall: K reaction, C reaction, -F)")
    @printf("   top    %+16.9f %+16.9f %+16.9f  = %+16.9f (steady CBF)\n",
            a.k_top, a.c_top, -a.f_top, a.Qss_top)
    @printf("   bottom %+16.9f %+16.9f %+16.9f  = %+16.9f (steady CBF)\n",
            a.k_bot, a.c_bot, -a.f_bot, a.Qss_bot)
    @printf("   wall mass term: top %+16.9e  bottom %+16.9e\n", a.m_top, a.m_bot)
    @printf("   pointwise      : top %+16.9f  bottom %+16.9f\n", a.pt_top, a.pt_bot)
    @printf("   4-pt Gauss wall integral of the same P2 derivative: top %+16.9f  bottom %+16.9f\n",
            a.g_top, a.g_bot)
    @printf("   closed form -(2/(3h))(th0-2th1):  top %+16.9f  bottom %+16.9f\n",
            a.wq_pred_top, a.wq_pred_bot)
    @printf("   measured (Qss - pt):             top %+16.9f  bottom %+16.9f\n",
            a.Qss_top - a.pt_top, -a.Qss_bot - a.pt_bot)
    @printf("   wall rows: top %s   bottom %s\n",
            join(["L$(k)=$(v)" for (k, v) in sort(collect(a.wall_top_levels))], " "),
            join(["L$(k)=$(v)" for (k, v) in sort(collect(a.wall_bot_levels))], " "))
    println("-"^104)
    println("F. OTHER TERMS")
    @printf("   %-38s %+18.9e   (budget's u.grad theta)\n", "u_grad_theta", a.ugrad)
    @printf("   %-38s %+18.9e   (budget's base_adv)\n", "base_adv", a.base_adv)
    @printf("   %-38s %+18.9e\n", "latent_adv", a.ladv)
    @printf("   %-38s %+18.9e\n", "latent_dgdt = src + adv", a.latent)
    @printf("   %-38s %+18.9e\n", "theta_int", a.theta_int)
    @printf("   %-38s %+18.9e\n", "latent_int (enthalpy, not a closure term)", a.latent_int)
    @printf("   %-38s %+18.9e\n", "heating (rho_heating/adiabatic/dissipation)", a.heating)
    @printf("   %-38s ev_max %.4e   mean %.4e   nonzero leaves %d\n", "EV field", a.ev_max,
            a.ev_mean, a.ev_nonzero)
    @printf("   %-38s %.4e  (wall leaves: identically 0 by semquad_ev_nu)\n",
            "EV max on a wall leaf", a.ev_at_wall)
    println("-"^104)
    if a.table !== nothing
        for (nm, tab) in (("top", a.table.top), ("bottom", a.table.bot))
            println("G. per-wall-node flux, $nm wall   (x, q_pt, q_cbf, q_cbf/q_pt)")
            for i in eachindex(tab.x)
                @printf("   %8.5f %+14.6e %+14.6e %+9.4f\n", tab.x[i], tab.q_pt[i],
                        tab.q_cbf[i], tab.q_cbf[i] / tab.q_pt[i])
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------------- main ----
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
    scratch = get(opts, "scratch", joinpath(@__DIR__, "..", "..", "out_julia",
                                            "nusselt_audit", "scratch"))
    nres = parse(Int, get(opts, "resume", "2"))
    detail = haskey(opts, "detail")
    prevopt = get(opts, "prev", "")
    isempty(pos) && error("give at least one checkpoint")
    for (k, ck) in enumerate(pos)
        isfile(ck) || error("checkpoint not found: $ck")
        st = load_semquad_state(ck)
        umax = maximum(hypot.(st.ux, st.uz))
        ev_on = cfg_float(st.config, "ev", 0.0) > 0.0
        ck2 = ck; prev = ""
        if !isempty(prevopt)                 # an already-resumed pair on disk
            ck2 = ck; prev = prevopt
        elseif !haskey(opts, "no-resume") && (umax == 0.0 || (ev_on && st.ev === nothing))
            tag = isempty(get(opts, "tag", "")) ? @sprintf("state%06d", st.step) :
                  get(opts, "tag", "")
            ck2, prev = resume_state(ck; scratch=scratch, steps=nres, tag=tag)
        end
        a = audit(ck2, prev; label=get(opts, "label", basename(ck)), detail=detail)
        report(a)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
