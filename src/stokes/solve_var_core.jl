# The core variable-viscosity Stokes solve.
#
# `_solve_semquad_stokes_var_core` is the body of every `linear` setting: it
# assembles the operators for the viscosity closure it is handed and runs the
# path the selector chose (`:direct`, `:mg`), with the iteration diagnostics and
# the acceptance test.  The public entry point in `solve_var.jl` wraps it with the
# `:auto` selection and the fallback.

"""
    _wall_index(dir)

Position of a wall in `Params.wall_bc`, ordered `(xminus, xplus, yminus, yplus)`.
"""
_wall_index(dir::Symbol) = dir === :xminus ? 1 :
                           dir === :xplus ? 2 :
                           dir === :yminus ? 3 : 4

"""
    _semquad_wall_normal_flux(ss, qt, fixed, fd, n, nu) -> (flux, umax)

Net boundary flux `∫∂Ω u·n dS` of the prescribed normal Dirichlet data, in the
**outward-normal** convention (negative on `xminus`/`yminus`, positive on
`xplus`/`yplus`), plus the largest prescribed normal value `umax`.

The data are read from the final `fixed` dict and integrated with the same 1-D
Gauss rule the leaf assembly uses (`fd.gl_w`/`fd.phi_v`, exact for the
polynomial interpolant of the face data).  A corner node belongs to two walls;
using the final dict is deliberate, because the wall write order
`(xminus, xplus, yminus, yplus)` lets a later rigid y-wall's *tangential* value
overwrite the x-wall's normal one at the corner, and that is the data the solve
actually imposes (docs/dev/BC_DESIGN.md §1.4).  The pass is serial and runs
after the fixed-dof loop, so it changes no assembly order.
"""
function _semquad_wall_normal_flux(ss::SemStokes, qt::Quadtree,
                                   fixed::Dict{Int,Float64}, fd::FEData,
                                   n::Int, nu::Int)
    su = ss.su
    flux = 0.0
    umax = 0.0
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        uids = su.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                sgn = dir === :xplus ? 1.0 : -1.0
                for j in 1:n
                    umax = max(umax, abs(get(fixed, uids[(j - 1) * n + iF], 0.0)))
                end
                for q in 1:fd.nq
                    val = 0.0
                    for j in 1:n
                        val += get(fixed, uids[(j - 1) * n + iF], 0.0) *
                               fd.phi_v[j, q]
                    end
                    flux += sgn * (0.5 * hy * fd.gl_w[q]) * val
                end
            else
                jF = dir === :yminus ? 1 : n
                sgn = dir === :yplus ? 1.0 : -1.0
                for i in 1:n
                    umax = max(umax, abs(get(fixed, nu + uids[(jF - 1) * n + i], 0.0)))
                end
                for q in 1:fd.nq
                    val = 0.0
                    for i in 1:n
                        val += get(fixed, nu + uids[(jF - 1) * n + i], 0.0) *
                               fd.phi_v[i, q]
                    end
                    flux += sgn * (0.5 * hx * fd.gl_w[q]) * val
                end
            end
        end
    end
    return flux, umax
end


function _solve_semquad_stokes_var_core(ss::SemStokes, qt::Quadtree,
                                  eta_fun::Function,
                                  fx::Function, fz::Function,
                                  uDx::Function, uDz::Function;
                                  mass_rho::Union{Nothing,Function}=nothing,
                                  mass_form::Symbol=:anelastic,
                                  mass_drho::Union{Nothing,Function}=nothing,
                                  # Density hook (`Params.rho_multiplier`, G248)
                                  # on the anelastic mass constraint: the
                                  # assembled density becomes
                                  # `mass_rho(y)*mass_rho_mult(x,y)` and its
                                  # gradient is differentiated consistently, so
                                  # `Bc` is the Galerkin form of
                                  # `div(mass_rho*mass_rho_mult*u) = 0`.
                                  # `nothing` (default) leaves `Bc` bit-identical.
                                  mass_rho_mult::Union{Nothing,Function}=nothing,
                                  compress_block::Union{Nothing,AbstractMatrix}=nothing,
                                  deviatoric::Bool=false,
                                  geo_u::Union{Nothing,Vector{Float64}}=nothing,
                                  geo_coef::Union{Nothing,Function}=nothing,
                                  geo_spd_safety::Float64=0.0,
                                  dp_u::Union{Nothing,Vector{Float64}}=nothing,
                                  dp_pfield::Union{Nothing,Vector{Float64}}=nothing,
                                  dp_coef::Union{Nothing,Function}=nothing,
                                  elastic_stress::Union{Nothing,NTuple{3,Vector{Float64}}}=nothing,
                                  bc::Symbol=:rigid, linear::Symbol=:direct,
                                  wall_bc::NTuple{4,Symbol}=(:same, :same, :same, :same),
                                  wall_u::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
                                  wall_un::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
                                  stokes_accept_tol::Float64=Inf,
                                  stokes_lost_tol::Float64=SEMSTOKES_LOST_TOL,
                                  stokes_scaling::Symbol=:none,
                                  schur_maxiter::Int=200, schur_precond::Symbol=:mass,
                                  mg_cycles::Int=1, mg_tol::Float64=1.0e-10,
                                  mg_pressure_scale::Float64=0.0,
                                  mg_pressure_weight::Symbol=:inv_eta,
                                  mg_pressure_mode::Symbol=:auto,
                                  mg_pressure_precond::Symbol=:auto,
                                  mg_pressure_cache::Bool=true,
                                  mg_pressure_eta_tol::Float64=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
                                  mg_pressure_bfbt_d::Symbol=:auto,
                                  mg_pressure_bfbt_s2::Symbol=:free,
                                  mg_pressure_bfbt_s1::Symbol=:chol,
                                  mg_pressure_bfbt_s1_solve::Symbol=:chol,
                                  mg_pressure_bfbt_s1_tol::Float64=1.0e-2,
                                  mg_pressure_bfbt_s1_maxit::Int=50,
                                  mg_pressure_bfbt_s1_minwork::Int=0,
                                  mg_pressure_bfbt_s1_chunk::Int=0,
                                  mg_smoother::Symbol=:auto,
                                  mg_coarse::Symbol=:auto,
                                  mg_coarsen_layers::Int=1,
                                  mg_coarsen_factor::Union{Nothing,Float64}=nothing,
                                  mg_coarse_direct_max::Int=0,
                                  mg_smoothing::Symbol=:auto,
                                  mg_pre::Union{Integer,AbstractVector{<:Integer}}=2,
                                  mg_post::Union{Integer,AbstractVector{<:Integer}}=2,
                                  mg_omega::Union{Nothing,Float64}=nothing,
                                  mg_operator::Symbol=:matrixfree,
                                  mg_vel_solver::Symbol=:vcycle,
                                  mg_vel_tol::Float64=1.0e-1,
                                  mg_vel_maxit::Int=10,
                                  mg_restart::Int=100, mg_maxit::Int=500,
                                  mg_threads::Int=1,
                                  mg_diag::Symbol=:constrained,
                                  assemble_threads::Int=1,
                                  return_system::Bool=false,
                                  assemble_only::Union{Bool,Symbol}=false)
    bc in (:free, :rigid) || throw(ArgumentError("bc must be :free or :rigid"))
    all(w -> w in (:same, :free, :rigid), wall_bc) ||
        throw(ArgumentError("wall_bc entries must be :same, :free or :rigid"))
    # Stage 1 validation (G258 (e)): `wall_u` is a *tangential* value and only a
    # `:rigid` wall applies it; on a `:free` wall the tangential component is
    # traction-free, so the value would be a silent no-op.  `Params` refuses the
    # same configuration, but a direct library call bypasses it -- hence the
    # duplicate check here.  `wall_un` needs no such check: the normal component
    # is Dirichlet on every `:free`/`:rigid` wall.
    for i in 1:4
        wbc = wall_bc[i] === :same ? bc : wall_bc[i]
        wall_u[i] === nothing || wbc === :rigid ||
            throw(ArgumentError("wall_u[$i] sets the tangential velocity of a " *
                                "wall that resolves to :$wbc; only a :rigid wall " *
                                "has a Dirichlet tangential component, so the " *
                                "value would be silently ignored.  Set " *
                                "wall_bc[$i] = :rigid (or `bc = :rigid`), or " *
                                "drop wall_u[$i]."))
    end
    stokes_scaling in SEMSTOKES_SCALINGS ||
        throw(ArgumentError("stokes_scaling must be :none or :equilibrated"))
    (geo_u === nothing) == (geo_coef === nothing) ||
        throw(ArgumentError("geo_u and geo_coef must be given together"))
    if elastic_stress !== nothing
        # 2b step 2: `elastic_stress` is the previous step's deviatoric stress
        # (already relaxed by `a = eta/(eta+G dt)`), a nodal Q2 tensor field on
        # the velocity space.  It enters the load vector as `-int tau:eps(v)`;
        # all three components must live on the velocity nodes.
        for i in 1:3
            length(elastic_stress[i]) == ss.su.N ||
                throw(ArgumentError("elastic_stress component $i has " *
                                    "$(length(elastic_stress[i])) entries, " *
                                    "the velocity space has $(ss.su.N)"))
        end
    end
    if geo_u !== nothing
        length(geo_u) == 2 * ss.su.N ||
            throw(ArgumentError("geo_u has $(length(geo_u)) entries, expected $(2 * ss.su.N)"))
        deviatoric &&
            throw(ArgumentError("the geometric strain-rate Jacobian is only " *
                                "implemented for deviatoric=false"))
    end
    # Pressure-derivative (Drucker-Prager) Jacobian block `Kp`: the momentum
    # residual is `K(eta(T, edot(u), p)) u - B'p - f`, so its pressure
    # derivative is
    #     (Kp)_{i,a} = int 2 (deta/dp) eps(u):eps(v_i) phi_a
    # in the velocity rows and nothing in the constraint rows.  The resulting
    # saddle system is non-symmetric (velocity rows <-> pressure columns), which
    # the direct LU handles.  `dp_u` is the velocity whose strain rate enters
    # `deta/dp`, `dp_pfield` the nodal pressure field, `dp_coef(x,y,edot,p)` the
    # derivative itself (e.g. `semquad_deta_dp` wrapped with the temperature).
    if dp_u !== nothing || dp_pfield !== nothing || dp_coef !== nothing
        (dp_u !== nothing && dp_pfield !== nothing && dp_coef !== nothing) ||
            throw(ArgumentError("dp_u, dp_pfield and dp_coef must be given " *
                                "together"))
        length(dp_u) == 2 * ss.su.N ||
            throw(ArgumentError("dp_u has $(length(dp_u)) entries, expected $(2 * ss.su.N)"))
        length(dp_pfield) == ss.sp.N ||
            throw(ArgumentError("dp_pfield has $(length(dp_pfield)) entries, " *
                                "expected $(ss.sp.N)"))
        deviatoric &&
            throw(ArgumentError("the pressure-derivative Jacobian block is " *
                                "only implemented for deviatoric=false"))
    end
    linear in (:direct, :schur_cg, :mg, :lagged, :auto) ||
        throw(ArgumentError("linear must be :direct, :schur_cg, :mg, :lagged " *
                            "or :auto"))
    (assemble_only === false || assemble_only === true || assemble_only === :mg) ||
        throw(ArgumentError("assemble_only must be false (solve), true (assemble " *
                            "everything) or :mg (the reduced matrix-free MG " *
                            "geometry: no velocity stiffness K, no Kred)"))
    mg_cycles >= 1 || throw(ArgumentError("mg_cycles must be >= 1"))
    mg_tol > 0.0 || throw(ArgumentError("mg_tol must be > 0"))
    mg_pressure_scale >= 0.0 ||
        throw(ArgumentError("mg_pressure_scale must be >= 0 (0 = auto)"))
    mg_pressure_weight in (:inv_eta, :eta, :none) ||
        throw(ArgumentError("mg_pressure_weight must be :inv_eta, :eta or :none"))
    mg_pressure_mode in (:auto, :mass, :stiffness, :lsc, :bfbt) ||
        throw(ArgumentError("mg_pressure_mode must be :auto, :mass, :stiffness, :lsc or :bfbt"))
    mg_pressure_precond in (:auto, :bfbt, :mass, :scaled_mass, :pcd, :lsc) ||
        throw(ArgumentError("mg_pressure_precond must be :auto, :bfbt, :mass, " *
                            ":scaled_mass, :pcd or :lsc"))
    mg_pressure_eta_tol > 0.0 ||
        throw(ArgumentError("mg_pressure_eta_tol must be > 0 (Inf = never " *
                            "rebuild on eta drift; a tree change still rebuilds)"))
    mg_pressure_bfbt_d in (:auto, :leaf, :exact, :matrix) ||
        throw(ArgumentError("mg_pressure_bfbt_d must be :auto, :leaf, :exact " *
                            "or :matrix"))
    mg_pressure_bfbt_s2 in (:free, :matrix) ||
        throw(ArgumentError("mg_pressure_bfbt_s2 must be :free or :matrix"))
    mg_pressure_bfbt_s1 in (:chol, :mass) ||
        throw(ArgumentError("mg_pressure_bfbt_s1 must be :chol or :mass"))
    mg_pressure_bfbt_s1_solve in (:chol, :level, :cg_diag, :cg_ichol) ||
        throw(ArgumentError("mg_pressure_bfbt_s1_solve must be :chol, :level, " *
                            ":cg_diag or :cg_ichol"))
    mg_pressure_bfbt_s1_tol > 0.0 ||
        throw(ArgumentError("mg_pressure_bfbt_s1_tol must be > 0"))
    mg_pressure_bfbt_s1_maxit >= 1 ||
        throw(ArgumentError("mg_pressure_bfbt_s1_maxit must be >= 1"))
    mg_smoother in (:auto, :jacobi, :sgs) ||
        throw(ArgumentError("mg_smoother must be :auto, :jacobi or :sgs"))
    mg_coarse in (:auto, :rediscretize, :galerkin) ||
        throw(ArgumentError("mg_coarse must be :auto, :rediscretize or :galerkin"))
    mg_coarsen_layers >= 1 ||
        throw(ArgumentError("mg_coarsen_layers must be >= 1"))
    (mg_coarsen_factor === nothing ||
     (mg_coarsen_factor >= 1.0 && isfinite(mg_coarsen_factor))) ||
        throw(ArgumentError("mg_coarsen_factor must be >= 1 (or `nothing` = " *
                            "the matrix-free production default 1.5)"))
    mg_coarse_direct_max >= 0 ||
        throw(ArgumentError("mg_coarse_direct_max must be >= 0"))
    mg_smoothing in (:auto, :uniform, :taper) ||
        throw(ArgumentError("mg_smoothing must be :auto, :uniform or :taper"))
    _mg_sw_ok(v) = v isa Integer ? v >= 0 : all(x -> x >= 0, v)
    _mg_sw_ok(mg_pre) || throw(ArgumentError("mg_pre must be >= 0"))
    _mg_sw_ok(mg_post) || throw(ArgumentError("mg_post must be >= 0"))
    (mg_omega === nothing || 0.0 < mg_omega < 1.0) ||
        throw(ArgumentError("mg_omega must be in (0,1) or `nothing`"))
    mg_operator in (:sparse, :matrixfree) ||
        throw(ArgumentError("mg_operator must be :sparse or :matrixfree"))
    # Production default of the h-coarsening step.  `coarsen_factor=2.0` was
    # measured on the F tree (1.5x end to end, docs/dev/MG.md) but on
    # P1 it collapses the hierarchy from 8 to 6 levels and costs 177 outer
    # iterations instead of 61 (8.03 s against 5.45 s, measured at 1e-6).  The
    # shipped default is therefore 1.5: it still removes F's 1.7 %-coarsening
    # near-duplicate level (F: 8 -> 6 levels) but keeps P1's 40 776-dof level
    # (P1: 8 -> 7 levels), and wins on both states (F 5.42 -> 3.36 s,
    # P1 5.45 -> 5.12 s at one thread).  `mg_coarsen_factor=1.0` restores the
    # historical rule and `2.0` remains selectable; uniform/blob trees are a
    # bitwise no-op at 1.5 (their steps already coarsen by >=2x).
    coarsen_factor = mg_coarsen_factor === nothing ?
                     (mg_operator === :matrixfree ? 1.5 : 1.0) :
                     mg_coarsen_factor
    mg_vel_solver in (:cg, :vcycle) ||
        throw(ArgumentError("mg_vel_solver must be :cg or :vcycle"))
    mg_vel_tol > 0.0 || throw(ArgumentError("mg_vel_tol must be > 0"))
    mg_vel_maxit >= 1 || throw(ArgumentError("mg_vel_maxit must be >= 1"))
    mg_restart >= 5 || throw(ArgumentError("mg_restart must be >= 5"))
    mg_maxit >= 1 || throw(ArgumentError("mg_maxit must be >= 1"))
    mg_threads >= 1 || throw(ArgumentError("mg_threads must be >= 1"))
    mg_diag in (:constrained, :lumped) ||
        throw(ArgumentError("mg_diag must be :constrained or :lumped"))
    mass_form in (:anelastic, :boussinesq) ||
        throw(ArgumentError("mass_form must be :anelastic or :boussinesq"))
    schur_precond in (:none, :mass) ||
        throw(ArgumentError("schur_precond must be :none or :mass"))
    assemble_threads >= 1 ||
        throw(ArgumentError("assemble_threads must be >= 1"))
    # S6: resolve `linear=:auto` **before** the assembly, so an MG decision
    # never pays for or stores the sparse velocity operator.  The decision uses
    # only O(1) node counts and the fixed 33x33-grid eta sample (measured cost
    # in docs/dev/SOLVER.md §20); an explicit `:direct`/`:mg` call skips this
    # block entirely and runs exactly as before (bitwise for `:direct`).
    if linear === :auto
        auto_choice, auto_threads = _semquad_resolve_auto(ss, eta_fun;
                                                          schur_precond=schur_precond)
        linear = auto_choice
        if linear === :mg
            # The selector is calibrated on the matrix-free operator, and
            # `mg_threads=min(8, Threads.nthreads())` (S5-2 measured the gain).
            mg_operator = :matrixfree
            mg_threads = auto_threads
        end
    else
        # An `assemble_only` call never solves, so it must not overwrite the
        # `:auto` decision of the last real solve: the MG hierarchy builder and
        # the caller/Newton probes assemble with `linear=:direct, assemble_only=true`
        # *inside* an auto MG solve, and recording `:none` there would hide the
        # selector's decision (it broke the auto diagnostics until S7).
        assemble_only !== false ||
            (LAST_SEMSTOKES_AUTO_CHOICE[] =
                 (choice=:none,
                  reason=get(_SEMSTOKES_AUTO_OFF_REASONS, linear,
                             "linear not :auto (auto not requested)"),
                  saddle_dofs=0, contrast=1.0, samples=0, sample_ms=0.0,
                  mg_threads=0))
    end
    # BFBT weight: the exact restricted diagonal
    # `diag(Cu' M_u(sqrt(eta)) Cu)` is the shipped default on the matrix-free
    # path (docs/dev/MG.md).  It is a zero-per-apply-cost change
    # that removes 10 outer iterations on `F` (81 -> 71) and leaves `P1` at 65;
    # its leaf-local construction is cheaper than the historical lumped `:leaf`
    # diagonal at 8 threads and the same at 1 (both are dominated by the
    # leaf/quadrature-point eta evaluations).  `:leaf` is kept as the A/B and
    # `:matrix` as the sparse reference for G226; the sparse-operator MG path
    # keeps `:leaf` because it is not the measured production path.  Resolved
    # after the `linear=:auto` block, which may switch `mg_operator`.
    bfbt_d = mg_pressure_bfbt_d === :auto ?
             (mg_operator === :matrixfree ? :exact : :leaf) :
             mg_pressure_bfbt_d
    # `assemble_only=:mg` is the *reduced* assembly of the matrix-free MG
    # geometry: the caller (the hierarchy builder) only reads `Cu`, so the
    # velocity stiffness `K`, the COO triplets behind it and `Kred` are not
    # formed at all.  It is defined only for the matrix-free MG path (the sparse
    # operator path and the direct path need `K`/`Kred`); it is opt-in and
    # returns the same `Cu`/`Cp`/`B`/`fv` bit for bit.
    kfree = assemble_only === :mg
    kfree && return_system &&
        throw(ArgumentError("assemble_only=:mg does not form K/Kred, so it is " *
                            "incompatible with return_system=true"))
    kfree && !(linear === :mg && mg_operator === :matrixfree) &&
        throw(ArgumentError("assemble_only=:mg is the reduced matrix-free MG " *
                            "assembly; it requires linear=:mg with " *
                            "mg_operator=:matrixfree (got linear=$linear, " *
                            "mg_operator=$mg_operator)"))
    LAST_SEMSTOKES_VEL_CG_ITERS[] = 0
    p = ss.p; q = ss.q
    su = ss.su; sp = ss.sp
    nu = su.N; np = sp.N
    Nu2 = 2 * nu
    fd = make_fedata(p, q)
    n = p + 1
    nv = n * n
    npq = (q + 1)^2
    # S3 stage C: `linear=:mg, mg_operator=:matrixfree` never needs the sparse
    # velocity operator.  The right-hand side and the outer FGMRES operator are
    # built from the leaf-local apply (`semquad_apply_K!` / `_apply_Kred!`), so
    # the `2n^2 x 2n^2` dense leaf blocks and their `4 nv^2` triplets are never
    # formed.  Everything that still needs `K` (return_system, the Newton
    # geometric/pressure-derivative blocks) keeps it, and so does a plain
    # `assemble_only=true`; only the reduced `assemble_only=:mg` geometry call
    # joins the solve path in the `K`-free branch.  On that branch the leaf
    # assembly also skips the viscosity closure (its `etaq` only enters `K` and
    # the two Jacobian blocks), which is 0.15 s of the F setup at one thread --
    # see `SemQuadAsmCfg.mf_path`.
    mf_path = linear === :mg && mg_operator === :matrixfree &&
              (assemble_only === false || kfree) &&
              !return_system && geo_u === nothing && dp_u === nothing
    pleaves = Set(sp.leaves)
    # Deviatoric stress 2 eta (eps - (1/3) div(u) I): the volumetric part of
    # 2 eta eps:eps(v) loses 2/3 eta div(u) div(v).  ASPECT's ALA always uses
    # this form (manual eq. 20); without it a compressible (div u = Di u_z)
    # flow is too stiff.  `deviatoric=false` keeps the historical operator.
    a2 = deviatoric ? 4.0 / 3.0 : 2.0
    # S4 stage 1: leaf assembly.  `assemble_threads=1` (default) is the
    # historical serial path, bit for bit; >1 cuts the leaves into fixed
    # contiguous blocks, assembles them in parallel and merges the block COO
    # buffers in block order.  The threaded result is deterministic run to run
    # but is *not* bitwise equal to the serial one (different summation order),
    # which is why the default stays 1.
    asm_cfg = SemQuadAsmCfg(a2, deviatoric, mf_path, geo_u, geo_coef,
                            geo_spd_safety, dp_u, dp_pfield, dp_coef,
                            elastic_stress)
    asm = _semquad_assemble_blocks(
        _semquad_asm_thread_count(assemble_threads), asm_cfg, su, sp,
        eta_fun, fx, fz, fd, pleaves)
    IK = asm.IK; JK = asm.JK; VK = asm.VK
    IB = asm.IB; JB = asm.JB; VB = asm.VB
    IG = asm.IG; JG = asm.JG; VG = asm.VG
    IP = asm.IP; JP = asm.JP; VP = asm.VP
    fv = asm.fv

    if mf_path
        K = nothing
    else
        SEMSTOKES_K_ASSEMBLIES[] += 1
        K = sparse(IK, JK, VK, Nu2, Nu2)
    end
    Kgeo = geo_u === nothing ? nothing : sparse(IG, JG, VG, Nu2, Nu2)
    Kp = dp_u === nothing ? nothing : sparse(IP, JP, VP, Nu2, np)
    Bref = sparse(IB, JB, VB, np, Nu2)
    # Mass constraint with a stratified reference density.  ASPECT's ALA mass
    # equation is div(rho u) = 0, whose Galerkin form is
    #     int q (rho div u + u_z drho/dy) dV = 0,
    # **not** `int q rho div u = 0`: the latter only enforces div u = 0 and, with
    # upwellings, breaks the exact cancellation between the advection of the
    # reference state and the adiabatic heating term (King 2010 then cools and
    # its energy budget does not close).  The momentum pressure term stays the
    # unweighted -int p div v, so Bm = Bref while Bc carries the density.
    # `mass_form=:boussinesq` is the **TALA** continuity equation `div u = 0`:
    # the reference density enters the buoyancy only, not the mass conservation
    # (ASPECT's `truncated anelastic liquid approximation`).  It is the
    # apples-to-apples TALA counterpart of the ALA form for the same reference
    # state, which is what docs/dev/PHYSICS.md §10.3 measures; it is off unless
    # requested, so the default path is unchanged.
    Bm = Bref
    Bc = Bref
    if mass_rho !== nothing && mass_form === :anelastic
        # derivative of the density profile at the quadrature points
        drho = mass_drho === nothing ? (y -> begin
            hd = 1.0e-6
            yp = min(y + hd, 1.0); ym = max(y - hd, 0.0)
            (mass_rho(yp) - mass_rho(ym)) / (yp - ym)
        end) : mass_drho
        Bc = semquad_stokes_B(ss, qt; rho_fun=mass_rho, drho_fun=drho,
                              rho_mult=mass_rho_mult)
        linear === :schur_cg &&
            throw(ArgumentError("mass_form=:anelastic makes the saddle system " *
                                "nonsymmetric (Bm != Bc'); use linear=:direct"))
    end

    # affine Dirichlet data on the domain boundary
    fixed = Dict{Int,Float64}()
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            # per-wall condition: :same follows the global c; this is what
            # keeps the historical single-c runs bit-identical
            wi = _wall_index(dir)
            wbc = (wall_bc[wi] === :same ? bc : wall_bc[wi])
            rigid_wall = wbc === :rigid
            # per-wall normal value: `wall_un[wi] === nothing` reproduces the
            # historical global closure for that wall bit for bit (G258);
            # `wall_un[wi]` is the normal *component* in the positive
            # coordinate direction, exactly like `uDx`/`uDz`, not the outward
            # flux.  The write order (x walls, then y walls, normal before
            # tangential) is unchanged, so every corner overwrite of the legacy
            # paths is preserved (docs/dev/BC_DESIGN.md §1.4/§3.4).
            un = wall_un[wi]
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                xf = dir === :xplus ? x0 + hx : x0
                for j in 1:n
                    yv = y0 + 0.5 * (su.xi[j] + 1.0) * hy
                    g = uids[(j - 1) * n + iF]
                    fixed[g] = un === nothing ? uDx(xf, yv) : un(xf, yv)
                    tang = wall_u[wi]
                    rigid_wall && (fixed[nu + g] =
                                   tang === nothing ? uDz(xf, yv) : tang(xf, yv))
                end
            else
                jF = dir === :yminus ? 1 : n
                yf = dir === :yplus ? y0 + hy : y0
                for i in 1:n
                    xv = x0 + 0.5 * (su.xi[i] + 1.0) * hx
                    g = uids[(jF - 1) * n + i]
                    fixed[nu + g] = un === nothing ? uDz(xv, yf) : un(xv, yf)
                    tang = wall_u[wi]
                    rigid_wall && (fixed[g] =
                                   tang === nothing ? uDx(xv, yf) : tang(xv, yf))
                end
            end
        end
    end
    # Net-flux compatibility of the prescribed normal data (docs/dev/BC_DESIGN.md
    # §3.5 eq. 8).  In stage 1 the normal component is Dirichlet on *every* wall,
    # so the constant pressure is in the null space of the reduced gradient and
    # the pinned pressure gauge is exact only when the boundary data satisfy
    # ∫∂Ω u·n dS = 0.  The default card (driver closures `(x,y)->0.0`, every
    # `wall_un` nothing) integrates to zero exactly, so the `umax == 0` shortcut
    # keeps the default path free of extra arithmetic.  When `:open` walls land
    # (design commit 3) this check must skip the walls with a free normal
    # component, which absorb the imbalance by construction.
    netflux, wumax = _semquad_wall_normal_flux(ss, qt, fixed, fd, n, nu)
    if wumax > 0.0
        perimeter = 2.0 * (qt.Lx + 1.0)
        # The relative threshold is the design's `1e-12 * u_scale * perimeter`,
        # with one necessary correction: an **absolute floor** of `1e-12`
        # (in the solver's nondimensional velocity unit).  A manufactured-
        # solution gate (G33/G44/G150/G151/G154) prescribes the exact field as
        # its Dirichlet data; the *mathematical* net flux of that data is zero,
        # but its floating-point evaluation at the boundary nodes is not (e.g.
        # `sin(pi) = 1.2e-16`), so a strictly data-scaled test would reject
        # legitimate reference gates whose imbalance is a round-off artefact
        # and physically irrelevant.  The floor only decides cases whose flux
        # inconsistency is below 1e-12 of a unit velocity -- far below any
        # accuracy gate in the suite -- and it does not weaken the user-facing
        # check (a card's wall_un values are O(1..1e3)).
        tol = 1.0e-12 * max(wumax * perimeter, 1.0)
        if abs(netflux) > tol
            throw(ArgumentError("the prescribed normal wall velocities have a " *
                "net outward boundary flux of $(netflux) (threshold $(tol) = " *
                "1e-12 * max(umax * perimeter, 1), umax = $(wumax)).  Every wall " *
                "keeps a Dirichlet normal component, and the discrete continuity " *
                "equations are consistent only when ∫∂Ω u·n dS = 0.  " *
                "`wall_un[i]` (and the legacy `uDx`/`uDz` closures) is the wall's " *
                "normal *component* in the positive coordinate direction, so the " *
                "outward flux is its face integral with the sign -1 on " *
                "xminus/yminus and +1 on xplus/yplus; balance the inflow and the " *
                "outflow (e.g. `wall_un = (U, U, 0, 0)` for a uniform " *
                "through-flow)."))
        end
    end
    cons_u = semquad_constraints(su, qt)
    cons2 = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in cons_u
        cons2[c] = (ms, ws)
        cons2[nu + c] = (nu .+ ms, ws)
    end
    u0, Cu, _ = _affine_constraint(Nu2, cons2, fixed)
    cons_p = semquad_constraints(sp, qt)
    pf = Dict{Int,Float64}()
    pfree0 = setdiff(1:np, collect(keys(cons_p)))
    isempty(pfree0) || (pf[first(pfree0)] = 0.0)
    p0, Cp, _ = _affine_constraint(np, cons_p, pf)

    CpB = Cp' * Bc * Cu
    # S3 stage C: on the fully matrix-free MG path there is no sparse `K` and no
    # `Kred = Cu' K Cu`.  `fr = Cu' (fv - K u0 + Bm' p0)` needs `K u0` on the
    # *full* velocity vector, which the leaf-local apply gives directly, and the
    # outer operator becomes `Cu' K (Cu v)` (`_semquad_apply_Kred!`).
    mfop = nothing
    mfapplyK = nothing
    mfpar = nothing
    if mf_path
        mfop = semquad_kop(ss, qt; deviatoric=deviatoric)
        # sampled once, then laid out for the lane kernel (docs/dev/MG.md
        # §15): the leaf-major field plus its block-major lane table
        mfetav = semquad_eta_lanes(mfop, semquad_eta_field(mfop, eta_fun))
        mfwork = SemQuadKWork(mfop.n, mfop.nq, Nu2)
        Ku0 = zeros(Float64, Nu2)
        _semquad_apply_K_raw!(Ku0, mfop, mfetav, u0, mfwork)
        fr = Cu' * (fv - Ku0 + Bm' * p0)
        # S5-2: the outer operator (FGMRES residual and the matrix-free BFBT
        # `S2`) uses the same threaded leaf blocks as the V-cycle.
        nt_outer = _semquad_mg_thread_count(mg_threads)
        if nt_outer > 1
            par_outer = SemQuadKPar(mfop, nt_outer)
            mfpar = par_outer
            mfapplyK = (out, v) -> _semquad_apply_Kred_threaded!(out, mfop, Cu,
                                                                 mfetav, v,
                                                                 mfwork,
                                                                 par_outer)
        else
            mfapplyK = (out, v) -> _semquad_apply_Kred!(out, mfop, Cu, mfetav,
                                                        v, mfwork)
        end
    else
        fr = Cu' * (fv - K * u0 + Bm' * p0)
    end
    Kred = mf_path ? nothing :
           Kgeo === nothing ? Cu' * K * Cu : Cu' * (K + Kgeo) * Cu
    Kp_red = Kp === nothing ? nothing : Cu' * Kp * Cp
    gr = Cp' * (Bc * u0)
    nf = size(Cu, 2); nfp = size(Cp, 2)
    Bs = -CpB
    Bt = Bm === Bc ? sparse(transpose(Bs)) : -(Cu' * Bm' * Cp)
    # Compressibility: in the multigrid path's (dynamic) pressure gauge the ALA density
    # perturbation -rho_ad Di gamma (p - p_ad) contributes +rho_ad Di gamma p
    # to the *upward* momentum.  The sign is not free: with ASPECT's own
    # temperature field this sign reproduces ASPECT's velocity to six digits
    # (Vrms 24.7657 vs 24.7657, max|u| 46.858 vs 46.858), while the opposite
    # sign gives Vrms 26.94 (the historical 8% King error).  See
    # docs/dev/SOLVER.md for the operator-agreement test.
    if compress_block !== nothing
        Bmm = sparse(transpose(Bm)) - compress_block
        Bt = -(Cu' * Bmm * Cp)
    end
    # Pressure-block preconditioner for the saddle solve.  It is built after
    # Kred/Bs/Bt (and after the compressibility update of Bt) because the LSC
    # variant needs diag(Kred) and the final Bt.
    Fp = nothing
    if linear === :schur_cg && schur_precond === :mass
        Mfull = assemble_semquad_pressure_mass(sp)
        Mred = Cp' * Mfull * Cp
        Fp = cholesky(Symmetric(Mred))
        issuccess(Fp) ||
            throw(ArgumentError("pressure mass preconditioner is not SPD"))
    elseif linear === :mg && !kfree
        # Pressure-block approximate inverse (mass / LSC / PCD / three-factor
        # w-BFBT); shared with the MG-accelerated Newton solver, see
        # mg/pressure.jl.  `mg_pressure_precond` is the public selector
        # (:bfbt | :mass | :scaled_mass | :pcd | :lsc); its default `:auto`
        # keeps the historical `mg_pressure_mode` resolution, which picks the
        # three-factor w-BFBT above 100x viscosity contrast -- the measured
        # winner on the blob tree (docs/dev/SOLVER.md §13).  Not built for the
        # reduced `assemble_only=:mg` geometry call, which never uses it.
        pmode_arg = mg_pressure_mode
        pweight_arg = mg_pressure_weight
        if mg_pressure_precond !== :auto
            pmode_arg, pweight_arg =
                _semquad_pressure_precond_mode(mg_pressure_precond,
                                               mg_pressure_weight)
        end
        # S3 stage A: cached by (tree fingerprint, nf, nfp) and reused while the
        # sampled viscosity moves by at most `mg_pressure_eta_tol` (the outer
        # FGMRES is flexible, so a stale Shat^{-1} is a valid preconditioner).
        Fp, pscale = semquad_mg_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred,
                                                 eta_fun;
                                                 mode=pmode_arg,
                                                 scale=mg_pressure_scale,
                                                 weight=pweight_arg,
                                                 key=_saddle_key(qt, p, bc, nf, nfp),
                                                 eta_tol=mg_pressure_eta_tol,
                                                 cache=mg_pressure_cache,
                                                 bfbt_d=bfbt_d,
                                                 bfbt_s2=mg_pressure_bfbt_s2,
                                                 bfbt_s1=mg_pressure_bfbt_s1,
                                                 bfbt_s1_solve=mg_pressure_bfbt_s1_solve,
                                                 bfbt_s1_tol=mg_pressure_bfbt_s1_tol,
                                                 bfbt_s1_maxit=mg_pressure_bfbt_s1_maxit,
                                                 bfbt_s1_minwork=mg_pressure_bfbt_s1_minwork,
                                                 bfbt_s1_chunk=mg_pressure_bfbt_s1_chunk,
                                                 nthreads=_semquad_mg_thread_count(mg_threads),
                                                 applyK=mfapplyK)
    end
    if assemble_only !== false
        return (K=K, B=Bref, Bc=Bc, Bm=Bm, Cu=Cu, Cp=Cp, u0=u0, p0=p0, fv=fv,
                fr=fr, gr=gr, Kred=Kred, Bs=Bs, Bt=Bt, Kgeo=Kgeo, Kp=Kp,
                Kp_red=Kp_red)
    end
    nit = 0
    # Every solve resets the direct-path shift record, so it describes this
    # solve only (a `:mg`/`:schur_cg`/`:lagged` solve reports "no shift").
    LAST_SEMSTOKES_SHIFT[] = 0.0
    LAST_SEMSTOKES_REFINE_ITERS[] = 0
    LAST_SEMSTOKES_LADDER[] = nothing
    LAST_SEMSTOKES_LADDER_PHASE[] = nothing
    if linear === :schur_cg
        # Saddle [Kred (-CpB'); (-CpB) 0] with Bs = -CpB, Bt = Bs':
        #   Kred u + Bt p = fr,   Bs u = gr,
        # solved by a matrix-free Schur CG on S = Bs Kred^{-1} Bt.
        # A constrained/hanging velocity mode can leave a zero eigenvalue in
        # Kred after AMR; regularize with a 1e-12 relative diagonal shift so the
        # sparse LU is defined.  The shift is below the G44 exactness tolerance.
        dmax = maximum(abs, diag(Kred))
        Kred = Kred + (1.0e-12 * max(dmax, 1.0)) * I
        # Row/column equilibration: wall-refined AMR trees have h_min/h_max
        # ~1e-4 and Kred condition numbers ~1e12; the scaled LU is much better
        # conditioned and removes the artificial large velocity from the shift.
        d = sqrt.(max.(diag(Kred), 1.0e-14 * dmax))
        Dinv = spdiagm(0 => 1.0 ./ d)
        Kscaled = Dinv * Kred * Dinv
        Fv = lu(Kscaled)
        ufree = Dinv * (Fv \ (Dinv * fr))
        r = Bs * ufree - gr
        pfree = zeros(Float64, nfp)
        z = Fp === nothing ? copy(r) : Fp \ r
        pp = copy(z)
        rho = dot(r, z)
        bnorm = max(dot(r, r), 1.0e-300)
        for it in 1:schur_maxiter
            nit = it
            Ap = Bs * (Dinv * (Fv \ (Dinv * (Bt * pp))))
            den = dot(pp, Ap)
            den == 0.0 && break
            alpha = rho / den
            pfree .+= alpha .* pp
            r .-= alpha .* Ap
            rs = dot(r, r)
            rs <= 1.0e-18 * bnorm && break
            z = Fp === nothing ? copy(r) : Fp \ r
            rho_new = dot(r, z)
            pp .= z .+ (rho_new / rho) .* pp
            rho = rho_new
        end
        ufree = Dinv * (Fv \ (Dinv * (fr - Bt * pfree)))
    elseif linear === :mg
        # Scalable path: geometric MG for the velocity block plus a pressure
        # Schur block, used as a block-triangular right preconditioner for
        # flexible GMRES on the reduced saddle (S3/MF-4).  The velocity block
        # is solved by the fixed `mg_cycles` V-cycles
        # (`mg_vel_solver=:vcycle`, the default since MF-6, 33% faster than the
        # inner-CG variant on the 1024-leaf blob tree); `mg_vel_solver=:cg`
        # keeps the MF-4 scheme, CG on the SPD `Kred` preconditioned by the
        # hierarchy (the deal.II/ASPECT FGMRES+CGMG form).  The criterion is
        # the *true* reduced-saddle residual (see `fgmres_restarted`), never
        # the preconditioned one.
        uniform = length(unique(lf[1] for lf in qt.leaves)) == 1
        smoother = mg_smoother === :auto ? :jacobi : mg_smoother
        coarse = mg_coarse === :auto ? (uniform ? :rediscretize : :galerkin) :
                 mg_coarse
        # S3/MF-2: the matrix-free hierarchy always coarsens by
        # rediscretisation (see make_semquad_mg).
        mg_operator === :matrixfree && (coarse = :rediscretize)
        mg = make_semquad_mg(ss, qt, p, bc, deviatoric, eta_fun, Kred;
                             wall_bc=wall_bc,
                             cycles=mg_cycles, smoother=smoother,
                             coarse_operator=coarse, operator=mg_operator,
                             coarsen_layers=mg_coarsen_layers,
                             coarsen_factor=coarsen_factor,
                             coarse_direct_max=mg_coarse_direct_max,
                             smoothing=mg_smoothing, pre=mg_pre,
                             post=mg_post, omega=mg_omega,
                             mg_threads=mg_threads, mg_diag=mg_diag,
                             fine_Cu=Cu,
                             # On the matrix-free path the solve already built
                             # the finest level's leaf-local operator and its
                             # eta field; reuse both instead of resampling the
                             # viscosity a third time (0.17 s at one thread).
                             fine_mfop=mf_path ? mfop : nothing,
                             fine_mfeta=mf_path ? mfetav : nothing)
        Fp === nothing &&
            throw(ArgumentError("linear=:mg requires the pressure mass " *
                                "preconditioner (schur_precond=:mass)"))
        nftot = nf + nfp
        sol, nit, relres, cgiters =
            semquad_mg_saddle_solve(Kred, Bs, Bt, Fp, pscale, vcat(fr, gr), mg;
                                    tol=mg_tol, maxit=mg_maxit,
                                    restart=mg_restart,
                                    vel_solver=mg_vel_solver,
                                    vel_tol=mg_vel_tol,
                                    vel_maxit=mg_vel_maxit,
                                    applyK=mfapplyK)
        LAST_SEMSTOKES_VEL_CG_ITERS[] = cgiters
        ufree = sol[1:nf]
        pfree = sol[(nf + 1):nftot]
        relres > 1.0e-5 &&
            @warn "MG-FGMRES true relres=$relres exceeds 1e-5 at level " *
                  "$(mg.geo.L)"
    elseif linear === :lagged
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        # S2: reuse the previous LU as a preconditioner and polish with iterative
        # refinement; refactor only on a cache miss / stall.  The saddle depends
        # on the viscosity field and the tree, not on dt, so between two AMR
        # cycles the stale factor is a good preconditioner.
        key = _saddle_key(qt, ss.su.p, bc, nf, nfp)
        sol, rn = _lagged_solve!(S, rhs, key; tol=1.0e-8, maxit=25)
        LAST_SEMSTOKES_REGULARIZED[] = false
        if rn > 1.0e-8
            dmax = maximum(abs, diag(S)) + 1.0
            Ssh = S + (1.0e-12 * dmax) * I
            sol, rn = _lagged_solve!(Ssh, rhs, key; tol=1.0e-8, maxit=6,
                                     force_refactor=true)
            LAST_SEMSTOKES_REGULARIZED[] = true
        end
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    else
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        # Wall-refined trees with free-slip walls make the *unshifted* sparse LU
        # unstable: on the 2b base=3/4 wall trees the factorisation has a growth
        # factor ~1e15 (max |U_ii| ~ 1e13-1e16 against a 2.9e-5 smallest pivot)
        # and the backward residual is ~1e3, while the same system solved with a
        # 1e-12 relative diagonal shift -- exactly what `_semquad_newton_saddle`
        # and the :schur_cg path already use -- is exact (residual 3e-11) and
        # gives the same velocity to 1e-6.  The near-null mode is a pressure mode
        # of the level-6 wall strip (|Bs'q|/|q| ~ 5e-3, velocity part 2e-4).
        #
        # `_shift_ladder_solve` keeps that single retry as its first rung and
        # only escalates (1e-10, 1e-8) while the residual misses
        # `stokes_accept_tol`, the driver's Stokes gate; the last resort is a few
        # iterative-refinement steps with the same factorisation.  The default
        # `accept_tol = Inf` stops after the first rung, so the well-conditioned
        # systems (uniform meshes, L2-L4 2b, the blob) stay bit-identical.
        # docs/dev/CASE_REALISTIC.md §13: on the 8192-leaf realistic tree the 1e-12
        # rung alone leaves 1.3e-3 (1.3x the card's 1e-3) and the driver aborted
        # the long run at step 6130.  §13.10: when the unshifted solve is *lost*
        # (`stokes_lost_tol`, default 1.0) the ladder tries the tiny shifts
        # first, because the shift's bias is `shift*|x|` and the near-null
        # cluster of the level-5 wall strip sits ~170x below the 1e-12 rung
        # (measured: 4.7e7 unshifted -> 5.2e-7 at 1e-14 on the step-6600 state,
        # while the 1e-12 rung leaves 1.1e-4).
        rden0 = max(norm(fr), norm(gr), 1.0)
        rel = function (x)
            rv = norm(Kred * x[1:nf] + Bt * x[(nf + 1):(nf + nfp)] - fr) / rden0
            rd = norm(Bs * x[1:nf] - gr) / rden0
            return max(rv, rd)
        end
        if stokes_scaling === :equilibrated
            # docs/dev/BUG_LEDGER.md: solver-side symmetric equilibration
            # of the same saddle.  First the historical unscaled unshifted solve,
            # exactly the ladder's rule 1: a solve that is healthy there is
            # returned untouched, so healthy paths stay bitwise identical even
            # with the option on, and the scaling is only paid for on a solve
            # that would otherwise enter the shift ladder.
            x_un = S \ rhs
            r_un = rel(x_un)
            r_un = isfinite(r_un) ? r_un : Inf
            if !(r_un > 1.0e-8) && !(r_un > stokes_accept_tol)
                sol, sh, nref, rungs = x_un, 0.0, 0, nothing
                LAST_SEMSTOKES_LADDER_PHASE[] = :healthy
            else
                # The congruence S -> D*S*D, rhs -> D*rhs leaves the solution
                # unchanged (`x = D*y`); it changes only the arithmetic of the
                # factorisation.  In the scaled operator the wall-strip pressure
                # family is lifted far above the round-off floor (the measured
                # smallest eigenvalue is a momentum mode at 4.2e-11 relative,
                # not the unscaled 1.7e-14 pressure mode), so the *unshifted* LU
                # resolves it: 6.0e-10 (AMR 7641) and 3.6e-8 (uniform L5,
                # round-off limited by the null component) against 3.7e-7 /
                # 1.2e-6 for the unscaled tiny-shift stopgap.  The 1e-6 gate is
                # the stated accuracy goal; if the scaled solve misses it, the
                # historical unscaled ladder is the unchanged fallback.
                sol_scaled, _ = _semquad_stokes_equil_solve(S, Kred, Bt, Bs, rhs)
                r_sc = rel(sol_scaled)
                r_sc = isfinite(r_sc) ? r_sc : Inf
                if r_sc <= 1.0e-6 && r_sc <= stokes_accept_tol
                    sol, sh, nref = sol_scaled, 0.0, 0
                    rungs = Any[(0.0, r_un), (:equilibrated, r_sc)]
                    LAST_SEMSTOKES_LADDER_PHASE[] = :equilibrated
                    LAST_SEMSTOKES_SCALED[] += 1
                else
                    sol, _, sh, nref, rungs =
                        _shift_ladder_solve(S, rhs; accept_tol=stokes_accept_tol,
                                            lost_tol=stokes_lost_tol,
                                            bad_tol=1.0e-8, resid=rel)
                end
            end
        else
            sol, _, sh, nref, rungs =
                _shift_ladder_solve(S, rhs; accept_tol=stokes_accept_tol,
                                    lost_tol=stokes_lost_tol,
                                    bad_tol=1.0e-8, resid=rel)
        end
        LAST_SEMSTOKES_REGULARIZED[] = sh > 0.0
        LAST_SEMSTOKES_SHIFT[] = sh
        LAST_SEMSTOKES_REFINE_ITERS[] = nref
        LAST_SEMSTOKES_LADDER[] = rungs
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    end
    # True reduced-saddle residual of the system that was actually solved
    # (Schur uses the regularized Kred).  This is what distinguishes a
    # converged solve from a singular/ill-conditioned direct solve.
    r1 = if mf_path
        Ku = zeros(Float64, nf)
        mfapplyK(Ku, ufree)
        Ku .+ Bt * pfree .- fr
    else
        Kred * ufree + Bt * pfree - fr
    end
    r2 = Bs * ufree - gr
    rden = max(norm(fr), norm(gr), 1.0)
    LAST_SEMSTOKES_RESIDUAL[] = max(norm(r1), norm(r2)) / rden
    LAST_SEMSTOKES_RESID_SPLIT[] = (vel=norm(r1) / rden, div=norm(r2) / rden)
    LAST_SEMSTOKES_SCHUR_ITERS[] = nit
    u = u0 + Cu * ufree
    pv = p0 + Cp * pfree
    if return_system
        return u, pv, fixed, cons2, cons_p, K, Bref, Cu, Cp, u0, p0, fv
    end
    return u, pv, fixed, cons2, cons_p
end
