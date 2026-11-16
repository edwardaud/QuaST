"""
    SemQuadPressureCache

Cached pressure-block approximate inverse of the `linear=:mg` saddle solve
(S3 stage A).  The three-factor w-BFBT costs ~1.1 s to build on the 1024-leaf
blob tree, but it depends only on the tree, the pressure space and the viscosity
field: the tree changes every few AMR cycles and `eta` moves a fraction of a
percent per step.  The outer solver is *flexible* GMRES, so a slightly stale
`Shat^{-1}` is a perfectly valid preconditioner -- exactly the S2 stale-factor
idea (`StaleSaddle`) applied to the pressure block instead of the whole saddle.

`eta` is the sampled viscosity vector the factors were built from; a rebuild is
triggered when `max|eta_new/eta_old - 1|` exceeds the caller's threshold.
"""
mutable struct SemQuadPressureCache
    key::UInt64              # tree + geometry fingerprint (`_saddle_key`)
    mode::Symbol             # resolved mode (:mass, :bfbt, :lsc, :stiffness)
    pscale::Float64
    weight::Symbol
    dweight::Symbol          # BFBT weight construction (:leaf, :exact, :matrix)
    s1solve::Symbol          # BFBT inner solve (:chol, :level, :cg_diag, :cg_ichol)
    s1tol::Float64           # its relative tolerance (CG kinds)
    eta::Vector{Float64}     # eta sample the factors were built from
    Fp::Any                  # (kind=:bfbt, F1, S1solve, S2) or a Cholesky factorisation
    builds::Int
end

const SEMQUAD_PRESSURE_CACHE = Ref{Union{Nothing,SemQuadPressureCache}}(nothing)
# Default cache threshold of the pressure block (S3-press A3, retuned in S5-1).
# The blob's viscosity moves 5.7-8.9% per *step* (25-37% between stored frames),
# so the old 1e-2 threshold almost never hit.  The measured threshold curve
# (docs/dev/SOLVER.md §13.3) shows the outer iteration count is *flat* (39.7 at
# 1e-12, 39.9 even at Inf), so the threshold only trades build time against
# staleness; the AMR tree changes every 5 steps and forces a rebuild anyway, so
# 1.0 buys one build per tree block (21 builds instead of 41 in the 25-frame
# blob run, §19.1).  `Inf` is a *supported but not default* value: it never
# rebuilds on eta drift (the tree fingerprint still wins) and is only safe for
# smoothly evolving viscosity -- no phase change and no yield onset inside a
# tree block (docs/dev/SOLVER.md §19.1).
const SEMQUAD_PRESSURE_DEFAULT_ETA_TOL = 1.0
# Diagnostics for gates/probes: cumulative builds and reuses, and what the last
# call did.
const SEMSTOKES_PRESSURE_BUILDS = Ref(0)
const SEMSTOKES_PRESSURE_REUSES = Ref(0)
const LAST_SEMSTOKES_PRESSURE_REUSED = Ref(false)
const LAST_SEMSTOKES_PRESSURE_JUMP = Ref(0.0)
# Resolved mode of the last pressure block (diagnostic for gates/probes): one of
# :mass, :stiffness, :lsc, :bfbt, :bfbt_m1, :pcd.
const LAST_SEMSTOKES_PRESSURE_MODE = Ref(:none)

"""
    _semquad_pressure_precond_mode(pre, weight) -> (mode, weight)

Resolve the public `mg_pressure_precond` keyword (`:bfbt`, `:mass`,
`:scaled_mass`, `:pcd`, `:lsc`) into the internal `(mode, weight)` pair of the
factor builder:

| `mg_pressure_precond` | internal mode | weight |
|---|---|---|
| `:bfbt` | `:bfbt` | (unused) |
| `:mass` | `:mass` | `:none` (unweighted `M_p`) |
| `:scaled_mass` | `:mass` | `mg_pressure_weight` (`:inv_eta` by default) |
| `:pcd` | `:pcd` | (unused) |
| `:lsc` | `:lsc` | (unused) |

`:mass` is the *unweighted* pressure mass matrix on purpose: it is the variant
the spectral-equivalence theory names, and keeping it distinct from
`:scaled_mass` makes the two measurable side by side (docs/dev/SOLVER.md §13).
"""
function _semquad_pressure_precond_mode(pre::Symbol, weight::Symbol)
    pre === :bfbt && return (:bfbt, weight)
    pre === :lsc && return (:lsc, weight)
    pre === :pcd && return (:pcd, weight)
    pre === :mass && return (:mass, :none)
    pre === :scaled_mass && return (:mass, weight)
    throw(ArgumentError("mg_pressure_precond must be :auto, :bfbt, :mass, " *
                        ":scaled_mass, :pcd or :lsc"))
end

"""
    _semquad_eta_sample(su, eta_fun; nsamples=256)

Viscosity sampled on a fixed, evenly strided set of velocity nodes (`<= 256`
samples, the same set the `:auto` pressure-mode decision uses).  Used as the
change metric of the pressure-cache: the sample is deterministic for a given
tree, so `max|eta_new/eta_old - 1|` measures how much the viscosity field moved
between two Stokes solves.
"""
function _semquad_eta_sample(su::SemQuad, eta_fun::Function; nsamples::Int=256)
    N = su.N
    stride = max(1, N ÷ nsamples)
    vals = Vector{Float64}(undef, length(1:stride:N))
    i = 0
    @inbounds for g in 1:stride:N
        e = eta_fun(su.coords[g]...)
        e > 0.0 || throw(ArgumentError("eta_fun returned $e <= 0"))
        vals[i += 1] = e
    end
    return vals
end

"""Number of velocity nodes `_semquad_eta_sample` visits on a space of `N` nodes."""
_semquad_eta_sample_count(N::Integer; nsamples::Int=256) =
    length(1:max(1, Int(N) ÷ nsamples):Int(N))

"""
    semquad_eta_contrast(su, eta_fun; npoints=SEMQUAD_AUTO_CONTRAST_GRID) -> Float64

Viscosity contrast `max(eta)/min(eta)` of the **actual** eta field the solve
uses, sampled on a fixed Cartesian grid of `(npoints+1)^2` points spanning
`[0, Lx] x [0, 1]` (default 32 intervals -> 33x33 = 1 089 points).  This is the
`contrast` input of `semquad_choose_linear` / `linear=:auto`.

Why a global grid instead of the <=256 strided velocity nodes used by the
pressure-cache metric (`_semquad_eta_sample`): measured on the production blob
frame-19 eta field, the node sample under-estimates the full quadrature-field
contrast by up to 5x (985x vs 4 926x on the 1024-leaf tree, 995x vs 5 299x on
the 4096-leaf tree), while 33x33 grid points give 5 087x (-3% / -9% vs the two
trees) with the same value on every tree (docs/dev/SOLVER.md §20).  Sampling a
strided *subset of leaves* is worse still (808-892x on the refined trees) for
the same reason: the viscosity extrema are spatially localised.  The grid cost
is fixed and independent of the tree: measured ~20 us on the production blob
closure (1 089 evaluations, ~18 ns each; a constant eta closure is folded to
nothing), vs 0.7-2.6 ms for a full O(N) quadrature-field pass on the
1024-4096-leaf trees.

The grid sample can still miss a sub-grid-scale extremum, but the selector's
contrast cut-offs are 10x/100x and the measured fields sit at 1x and ~5 000x;
grid resolution is reported in docs/dev/SOLVER.md §20.
"""
function semquad_eta_contrast(su::SemQuad, eta_fun::Function;
                              npoints::Int=SEMQUAD_AUTO_CONTRAST_GRID)
    n = max(1, npoints)
    Lx = su.Lx
    lo = Inf; hi = 0.0
    @inbounds for j in 0:n, i in 0:n
        e = eta_fun(Lx * (i / n), j / n)
        e > 0.0 || throw(ArgumentError("eta_fun returned $e <= 0 at " *
                                       "($(Lx * i / n), $(j / n))"))
        lo = min(lo, e); hi = max(hi, e)
    end
    return hi / lo
end

"""
    semquad_mg_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred, eta_fun; mode=:auto,
                                scale=0.0, weight=:inv_eta, key=nothing,
                                eta_tol=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
                                cache=true)

Pressure-block approximate inverse for the `linear=:mg` saddle solve (and the
MG-accelerated Newton).  Returns `(Fp, pscale)`:

* `mode=:auto` picks `:mass` for viscosity contrast <= 100 and the three-factor
  w-BFBT for stronger contrast (sampled on <= 256 velocity nodes);
* `:mass` / `:stiffness` build `Cp' M Cp` with the `weight`-scaled pressure
  mass/stiffness; `:lsc` uses `Bs diag(Kred)^{-1} Bt`; `:pcd` builds the
  pressure convection-diffusion factors `A_p^{-1} F_p M_p^{-1}` (see
  `_semquad_build_pressure_precond`);
* `:bfbt` builds `S1 = Bs D^{-1} Bt`, `S2 = Bs D^{-1} Kred D^{-1} Bt` with
  `D = diag(Cu' M_u(sqrt(eta)) Cu)` (the *restricted* diagonal, not the lumped
  row sums; `bfbt_d` selects the construction -- see
  [`_semquad_bfbt_diag`](@ref)) and `F1 = chol(S1)`.  `bfbt_d=:auto` resolves
  to the exact leaf-local `:exact` weight; `:leaf` (lumped) and `:matrix`
  (sparse reference) stay selectable for the A/B and the gate.

`Fp` is either a `NamedTuple` `(kind=:bfbt, F1, S1solve, S2=...)` or
`(kind=:mass, F=...)`; `pscale` is the block scaling used by the caller.

**The `S1` inner solve.**  `bfbt_s1_solve` selects how the two `F1` solves of the
w-BFBT apply are carried out (see `src/mg/s1solve.jl` and
docs/dev/MG.md):

| value | the solve | measured |
|---|---|---|
| `:chol` (default) | `ldiv!(x, F1, b)`, the exact CHOLMOD solve | 0.42-0.46 ms per solve on `F` |
| `:level` | the same factor, level-scheduled parallel substitutions | 0.43x of `:chol`: no level can pay for a barrier on this factor |
| `:cg_diag` / `:cg_ichol` | CG to `bfbt_s1_tol` with the diagonal / an IC(0) | 51-75 CG iterations per solve, 17-22 ms: 40x `:chol` |

The default is the exact solve; the other kinds exist because the measurement is
reproducible, and docs/dev/MG.md records why neither of them ships.
`bfbt_s1_tol` / `bfbt_s1_maxit` configure the CG kinds, and
`bfbt_s1_minwork` / `bfbt_s1_chunk` the level schedule.  The pressure cache is
keyed on the resolved kind and tolerance, so switching them rebuilds.

**Caching (S3 stage A).**  When `key` is given (the tree/geometry fingerprint
from `_saddle_key`) and `cache=true`, the factors are reused as long as the key,
the resolved mode, the scaling and the weight match and the sampled viscosity
has moved by at most `eta_tol` (relative, max norm).  Otherwise they are
rebuilt.  `key=nothing` disables the cache, which keeps the Newton path (one
factorisation per nonlinear solve) bit-identical to before.

`eta_tol` semantics (S5-1 retune, docs/dev/SOLVER.md §19.1):

| `eta_tol` | rebuilds when |
|---|---|
| `1e-12` | the sampled viscosity changes at all (the pre-cache behaviour) |
| `1.0` (default) | the tree fingerprint changes, or any sampled `eta` moves by >100% (a phase change / yield onset inside one tree block) |
| `Inf` | **only** when the tree fingerprint changes; eta drift is always accepted |

The outer FGMRES is flexible, so a stale `Shat^{-1}` is a valid preconditioner;
§13.3 measured the outer iteration count to be flat (39.7 at `1e-12`, 39.9 at
`Inf`) because BFBT's mass is set by the geometry and `eta` only enters through
`D = diag(M_u(sqrt(eta)))`.  `Inf` is therefore *available* but **not the
default**: it is only for smoothly evolving viscosity (no phase change, no yield
onset) -- otherwise the factors can go stale by orders of magnitude and the
outer iteration count must be re-measured.
"""
function semquad_mg_pressure_precond(ss::SemStokes, Cu, Cp, Bs, Bt, Kred,
                                     eta_fun::Function;
                                     mode::Symbol=:auto, scale::Float64=0.0,
                                     weight::Symbol=:inv_eta,
                                     key::Union{Nothing,UInt64}=nothing,
                                     eta_tol::Float64=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
                                     cache::Bool=true,
                                     bfbt_d::Symbol=:leaf,
                                     bfbt_s2::Symbol=:free,
                                     bfbt_s1::Symbol=:chol,
                                     bfbt_s1_solve::Symbol=:chol,
                                     bfbt_s1_tol::Float64=1.0e-2,
                                     bfbt_s1_maxit::Int=50,
                                     bfbt_s1_minwork::Int=0,
                                     bfbt_s1_chunk::Int=0,
                                     nthreads::Int=1,
                                     applyK::Union{Nothing,Function}=nothing)
    bfbt_d === :auto && (bfbt_d = :exact)
    bfbt_d in (:leaf, :exact, :matrix) ||
        throw(ArgumentError("bfbt_d must be :auto, :leaf, :exact or :matrix"))
    bfbt_s1_solve in (:chol, :level, :cg_diag, :cg_ichol) ||
        throw(ArgumentError("bfbt_s1_solve must be :chol, :level, :cg_diag or " *
                            ":cg_ichol"))
    LAST_SEMSTOKES_PRESSURE_REUSED[] = false
    LAST_SEMSTOKES_PRESSURE_JUMP[] = 0.0
    use_cache = cache && key !== nothing
    # One sampling pass serves both the `:auto` decision and the cache metric.
    esamp = (use_cache || mode === :auto) ?
            _semquad_eta_sample(ss.su, eta_fun) : Float64[]
    pmode = mode
    if pmode === :auto
        pmode = (maximum(esamp) / minimum(esamp) > 100.0) ? :bfbt : :mass
    end
    pscale = scale == 0.0 ? (pmode === :lsc ? 0.1 : pmode === :bfbt ? 0.4 : 1.0) :
             scale
    st = SEMQUAD_PRESSURE_CACHE[]
    if use_cache && st !== nothing && st.key == key && st.mode === pmode &&
       st.weight === weight && st.dweight === bfbt_d && st.pscale == pscale &&
       st.s1solve === bfbt_s1_solve && st.s1tol == bfbt_s1_tol &&
       length(st.eta) == length(esamp)
        jump = 0.0
        @inbounds for i in eachindex(esamp)
            jump = max(jump, abs(esamp[i] / st.eta[i] - 1.0))
        end
        LAST_SEMSTOKES_PRESSURE_JUMP[] = jump
        if jump <= eta_tol
            SEMSTOKES_PRESSURE_REUSES[] += 1
            LAST_SEMSTOKES_PRESSURE_REUSED[] = true
            Fp = st.Fp
            # A cached matrix-free S2 holds the *previous* `Kred`; rebind it to
            # the current physics so a stale cache does not also hold a stale
            # operator alive.
            if Fp isa NamedTuple && Fp.S2 === nothing && applyK !== nothing
                Fp = merge(Fp, (applyK=applyK, Kred=nothing))
            end
            LAST_SEMSTOKES_PRESSURE_MODE[] =
                Fp isa NamedTuple ? Fp.kind : st.mode
            return Fp, st.pscale
        end
    end
    Fp, pscale = _semquad_build_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred,
                                                 eta_fun, pmode, pscale, weight;
                                                 bfbt_d=bfbt_d, bfbt_s2=bfbt_s2,
                                                 bfbt_s1=bfbt_s1,
                                                 bfbt_s1_solve=bfbt_s1_solve,
                                                 bfbt_s1_tol=bfbt_s1_tol,
                                                 bfbt_s1_maxit=bfbt_s1_maxit,
                                                 bfbt_s1_minwork=bfbt_s1_minwork,
                                                 bfbt_s1_chunk=bfbt_s1_chunk,
                                                 nthreads=nthreads,
                                                 applyK=applyK)
    if use_cache
        n = st === nothing ? 1 : st.builds + 1
        SEMQUAD_PRESSURE_CACHE[] = SemQuadPressureCache(key, pmode, pscale,
                                                        weight, bfbt_d,
                                                        bfbt_s1_solve, bfbt_s1_tol,
                                                        esamp, Fp, n)
    end
    SEMSTOKES_PRESSURE_BUILDS[] += 1
    LAST_SEMSTOKES_PRESSURE_MODE[] = Fp isa NamedTuple ? Fp.kind : pmode
    return Fp, pscale
end

"""
    _semquad_build_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred, eta_fun, pmode,
                                    pscale, weight)

Build the pressure-block approximate inverse selected by the resolved `pmode`:
the uncached body of `semquad_mg_pressure_precond`.
"""
function _semquad_build_pressure_precond(ss::SemStokes, Cu, Cp, Bs, Bt, Kred,
                                         eta_fun::Function, pmode::Symbol,
                                         pscale::Float64, weight::Symbol;
                                         bfbt_d::Symbol=:leaf,
                                         bfbt_s2::Symbol=:free,
                                         bfbt_s1::Symbol=:chol,
                                         bfbt_s1_solve::Symbol=:chol,
                                         bfbt_s1_tol::Float64=1.0e-2,
                                         bfbt_s1_maxit::Int=50,
                                         bfbt_s1_minwork::Int=0,
                                         bfbt_s1_chunk::Int=0,
                                         nthreads::Int=1,
                                         applyK::Union{Nothing,Function}=nothing)
    su = ss.su
    sp = ss.sp
    # The Schur complement S = B K^{-1} B' scales like 1/eta, so the pressure
    # block of the block-triangular preconditioner uses the *weighted* mass
    # matrix int q_i q_j / eta, not the unweighted mass (which is only
    # spectrally equivalent for constant viscosity).  `:lsc` uses the
    # least-squares commutator B diag(K)^{-1} B' and `:stiffness` the weighted
    # pressure Laplacian; both are diagnostic options.
    pweight = (x, y) -> begin
        e = eta_fun(x, y)
        e > 0.0 || throw(ArgumentError("eta_fun returned $e <= 0"))
        weight === :eta ? e : weight === :none ? 1.0 : 1.0 / e
    end
    if pmode === :pcd
        # Pressure convection-diffusion (PCD, Elman-Silvester-Wathen ch. 9):
        #   Shat^{-1} = A_p^{-1} F_p M_p^{-1},   F_p = A_p + C_p,
        # with A_p the pressure-space version of the momentum operator's
        # Laplacian (here -div(eta grad q), Neumann on the whole boundary, made
        # definite by the caller's pinned pressure dof `Cp`), M_p the
        # *unweighted* pressure mass matrix, and C_p the pressure convection
        # operator built from the advecting velocity.
        #
        # The Stokes momentum equation of this code has **no advection term**
        # (infinite-Prandtl convection), so C_p = 0 and F_p = A_p identically;
        # the three factors then collapse to M_p^{-1} in exact arithmetic and
        # the iteration count equals that of `mg_pressure_precond=:mass`.  The
        # mode is kept (and wired) because the measurement is the point: on the
        # 1024-leaf blob tree it costs 2.7x more per apply than `:mass` for the
        # same outer iteration count (docs/dev/SOLVER.md §13).
        Ap = assemble_semquad_pressure_stiffness(sp;
                                                 weight=(x, y) -> eta_fun(x, y))
        Mp = assemble_semquad_pressure_mass(sp)
        Ared = Cp' * Ap * Cp
        Ared = 0.5 * (Ared + sparse(transpose(Ared)))
        Mred = Cp' * Mp * Cp
        Mred = 0.5 * (Mred + sparse(transpose(Mred)))
        FA = cholesky(Symmetric(Ared))
        issuccess(FA) ||
            throw(ArgumentError("PCD pressure Laplacian factorisation failed"))
        FM = cholesky(Symmetric(Mred))
        issuccess(FM) || throw(ArgumentError("PCD pressure mass is not SPD"))
        # `FF` is the *matrix* F_p = A_p + C_p (C_p = 0): the middle factor is a
        # matvec, only the two outer ones are triangular solves.
        return (kind=:pcd, FA=FA, FF=Ared, FM=FM), pscale
    end
    if pmode === :bfbt
        # Three-factor w-BFBT (Rudi-Stadler-Ghattas 2017):
        #   D = diag(M_u(sqrt(eta))),
        #   Shat^{-1} = S1^{-1} S2 S1^{-1},
        #   S1 = B D^{-1} B',  S2 = B D^{-1} K D^{-1} B'.
        wfun = (x, y) -> sqrt(max(eta_fun(x, y), 1.0e-300))
        d = _semquad_bfbt_diag(su, Cu, eta_fun, wfun; how=bfbt_d)
        Dinvv = 1.0 ./ d
        Dinv = spdiagm(0 => Dinvv)
        S1 = Bs * Dinv * Bt
        S1 = 0.5 * (S1 + sparse(transpose(S1)))
        dS1 = maximum(abs, diag(S1))
        S1 = S1 + (1.0e-14 * max(dS1, 1.0)) * I
        # `S2 = Bs D^{-1} Kred D^{-1} Bt` (327 nnz/row, 0.42 s to form on the
        # 1024-leaf tree) or its matrix-free application
        # `Bs (D^{-1} (Kred (D^{-1} (Bt r))))`, which costs the same per apply
        # (3.0 vs 3.1 ms) and no build time.  Forming it needs a sparse `Kred`;
        # on the fully matrix-free path (stage C) only the apply exists.
        S2 = (bfbt_s2 === :free || Kred === nothing) ? nothing :
             begin
                 S2m = Bs * Dinv * Kred * Dinv * Bt
                 0.5 * (S2m + sparse(transpose(S2m)))
             end
        # When S2 is applied matrix-free, keep whichever velocity operator the
        # caller can provide: the leaf-local apply of the fully matrix-free path
        # (stage C), or else the sparse `Kred`.
        s2Kred = (S2 === nothing && applyK === nothing) ? Kred : nothing
        s2apply = S2 === nothing ? applyK : nothing
        if bfbt_s1 === :mass
            # (b2) spectral equivalence: S1 = B D^{-1} B' is spectrally
            # equivalent to the 1/eta-weighted pressure mass matrix, so its
            # inverse can be replaced by the (much cheaper) mass inverse.
            Mfull = assemble_semquad_pressure_mass(sp; weight=pweight)
            Mred = Cp' * Mfull * Cp
            Mred = 0.5 * (Mred + sparse(transpose(Mred)))
            F1 = cholesky(Symmetric(Mred))
            issuccess(F1) || throw(ArgumentError("BFBT S1 (mass) is not SPD"))
            return (kind=:bfbt_m1, F1=F1, S2=S2, Dinvv=Dinvv, Bs=Bs, Bt=Bt,
                    Kred=s2Kred, applyK=s2apply), pscale
        end
        F1 = cholesky(Symmetric(S1))
        issuccess(F1) || throw(ArgumentError("BFBT S1 is not SPD"))
        # The inner solve is a first-class object (`semquad_s1solve.jl`): the
        # exact CHOLMOD solve by default, a level-scheduled solve over the same
        # factor or a loose-tolerance CG behind `bfbt_s1_solve`.
        S1i = _semquad_s1_inner(bfbt_s1_solve, F1, S1, nthreads; tol=bfbt_s1_tol,
                                maxit=bfbt_s1_maxit, minwork=bfbt_s1_minwork,
                                chunkwork=bfbt_s1_chunk)
        return (kind=:bfbt, F1=F1, S1solve=S1i, S2=S2, Dinvv=Dinvv, Bs=Bs, Bt=Bt,
                Kred=s2Kred, applyK=s2apply), pscale
    end
    Mfull = pmode === :mass ?
            assemble_semquad_pressure_mass(sp; weight=pweight) :
            pmode === :stiffness ?
            assemble_semquad_pressure_stiffness(sp; weight=pweight) :
            nothing
    Mred = if Mfull === nothing
        Kred === nothing &&
            throw(ArgumentError("mg_pressure_precond=:lsc (and " *
                                "mg_pressure_mode=:lsc) needs `diag(Kred)`, i.e. " *
                                "the assembled velocity operator; use " *
                                "mg_operator=:sparse with :lsc"))
        dK = max.(diag(Kred), 1.0e-14 * maximum(abs, diag(Kred)))
        DinvK = spdiagm(0 => 1.0 ./ dK)
        Bs * DinvK * Bt
    else
        Cp' * Mfull * Cp
    end
    Mred = 0.5 * (Mred + sparse(transpose(Mred)))
    F = cholesky(Symmetric(Mred))
    issuccess(F) || throw(ArgumentError("pressure Schur preconditioner is not SPD"))
    return F, pscale
end

"""
    _semquad_mass_diag!(d, su, w)

Diagonal of the velocity mass matrix `int w(x,y) u_i u_j` on the velocity space
`su`, computed leaf by leaf and scatter-added (`d` has length `su.N`).  The
assembled matrix has `(p+1)^4` entries per quadrature point; the diagonal has
`(p+1)^2`, so this is ~25x cheaper at p=4 and needs no sparse matrix at all.
"""
function _semquad_mass_diag!(d::Vector{Float64}, su::SemQuad, w::Function)
    p = su.p
    n = p + 1
    # `assemble_semquad_pressure_mass(su)` builds `make_fedata(2, su.p)` and uses
    # `fd.phi_p`, i.e. the degree-`p` nodal basis on the degree-`p` GLL nodes --
    # which is exactly the velocity basis of `su`.  Use the same table so the
    # leaf-wise diagonal is the assembled matrix's diagonal, not a different
    # quadrature of it.
    fd = make_fedata(2, p)
    dloc = zeros(Float64, n * n)
    fill!(d, 0.0)
    @inbounds for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        fill!(dloc, 0.0)
        for qy in 1:fd.nq, qx in 1:fd.nq
            wq = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy] *
                 w(x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx,
                   y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy)
            for j in 1:n, i in 1:n
                ph = fd.phi_p[i, qx] * fd.phi_p[j, qy]
                dloc[(j - 1) * n + i] += wq * ph * ph
            end
        end
        for idx in 1:(n * n)
            d[uids[idx]] += dloc[idx]
        end
    end
    return d
end

