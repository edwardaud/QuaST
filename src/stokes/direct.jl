# --- stale-factor cache for the direct saddle (S2: lagged factorization) -----
#
# The reduced saddle A = [Kred Bt; Bs 0] depends on the viscosity field and on
# the tree, not on dt.  With AMR every few steps the tree changes rarely, and
# between two AMR cycles the viscosity changes only through the temperature
# update.  So the LU of the previous solve is a good preconditioner for the
# current one: we reuse it and apply iterative refinement (Wilkinson), and only
# refactor when the refinement stalls or the tree changes.  This is the same
# idea as a frozen/stale preconditioner in PETSc/Trilinos (`-ksp_reuse_preconditioner`,
# ASPECT rebuilding its AMG only every few steps); iterative refinement with an
# LU factor is the cheapest form of it.
mutable struct StaleSaddle
    key::UInt64
    F::Any
    nfac::Int
end

const STALE_SADDLE = Ref{Union{Nothing,StaleSaddle}}(nothing)
const LAST_SEMSTOKES_LAGGED_ITERS = Ref(0)
const LAST_SEMSTOKES_LAGGED_REFACTOR = Ref(false)
const SEMSTOKES_LAGGED_REFACTORS = Ref(0)

"""Order-independent fingerprint of the tree plus the solver geometry."""
function _saddle_key(qt::Quadtree, p::Int, bc::Symbol, nf::Int, nfp::Int)
    h = hash((p, bc, nf, nfp))
    lvs = collect(qt.leaves)
    sort!(lvs)
    for lf in lvs
        h = hash(lf, h)
    end
    return h
end

"""
    _lagged_solve!(S, rhs, key; tol=1e-8, maxit=6, force_refactor=false)

Solve `S x = rhs` reusing the cached factorization of the previous saddle when
the tree fingerprint still matches, and polishing with iterative refinement.
Returns `(x, relative_residual)`.  A refactor happens when the cache misses,
when refinement did not reach `tol`, or when `force_refactor` is set.
"""
function _lagged_solve!(S::SparseMatrixCSC, rhs::Vector{Float64}, key::UInt64;
                        tol::Float64=1.0e-8, maxit::Int=6, force_refactor::Bool=false)
    # Symmetric diagonal equilibration: the blob saddle at Ra=1e8 has a condition
    # number ~1e12, and a stale factorisation only behaves as a preconditioner
    # after scaling (same trick as the :schur_cg path uses on Kred).
    dg = abs.(diag(S))
    dg .= max.(dg, 1.0e-14 * max(maximum(dg), 1.0))
    d = sqrt.(dg)
    Ds = spdiagm(0 => 1.0 ./ d)
    Ss = Ds * S * Ds
    brhs = Ds * rhs

    st = STALE_SADDLE[]
    refactored = false
    if force_refactor || st === nothing || st.key != key
        st = StaleSaddle(key, lu(Ss), st === nothing ? 1 : st.nfac + 1)
        STALE_SADDLE[] = st
        SEMSTOKES_LAGGED_REFACTORS[] += 1
        refactored = true
    end
    F = st.F
    applyA = v -> Ss * v
    precond = r -> F \ r
    y0 = F \ brhs
    y, _, nit = gmres_restarted(applyA, brhs; precond=precond, tol=tol,
                                maxit=maxit, restart=maxit, x0=y0)
    x = Ds * y
    r = rhs - S * x
    rn = norm(r) / max(norm(rhs), 1.0)
    # GMRES minimise the preconditioned residual; accept on the true one, at the
    # Stokes tolerance (1e-6), otherwise refactor.
    if !(rn <= 1.0e-6)
        st = StaleSaddle(key, lu(Ss), st.nfac + 1)
        STALE_SADDLE[] = st
        SEMSTOKES_LAGGED_REFACTORS[] += 1
        refactored = true
        x = Ds * (st.F \ brhs)
        r = rhs - S * x
        rn = norm(r) / max(norm(rhs), 1.0)
        nit = -1
    end
    LAST_SEMSTOKES_LAGGED_ITERS[] = nit
    LAST_SEMSTOKES_LAGGED_REFACTOR[] = refactored
    return x, rn
end

# ---------------------------------------------------------------------------
# The direct path's diagonal-shift ladder (docs/dev/CASE_REALISTIC.md §13)
# ---------------------------------------------------------------------------

# Relative diagonal shifts tried, in order, by `_shift_ladder_solve`.  The first
# rung is the historical single retry of the direct branch (a wall-refined tree
# can make the *unshifted* sparse LU unstable, G164); the larger rungs trade a
# larger perturbation of the operator for a better-conditioned factorisation and
# are only reached when the previous rung still misses the caller's acceptance
# tolerance.
const SEMSTOKES_SHIFT_LADDER = (1.0e-12, 1.0e-10, 1.0e-8)

# Relative diagonal shifts tried *before* the ladder when the unshifted LU has
# lost the solution outright (docs/dev/CASE_REALISTIC.md §13.10).  The wall-refined
# realistic tree carries a cluster of near-null pressure modes on its level-5
# wall strips (a discrete inf-sup deficiency of that strip: measured smallest
# |eigenvalue| 4.6e-10 against sigma_max 1.2e5, the three smallest singular
# values of the 1e-12-shift operator being 4.6e-10, 5.4e-10 and 9.3e-10).  A
# shift has two jobs there -- it makes the (structurally zero) pressure block
# pivottable and it biases the answer by `shift*|x|` -- and the historical
# 1e-12 rung sits ~170x above the cluster, so its bias (1e-4 .. 1e-5 relative on
# the measured states) is what limits the residual.  A shift at the cluster's
# own scale keeps the factorization stable (measured growth 1e-4 against 1e15
# unshifted) and leaves a bias two orders smaller.  The residual is measured, so
# an unlucky shift is simply rejected; the ladder then continues as before.
const SEMSTOKES_TINY_SHIFTS = (1.0e-14, 1.0e-13)
# An unshifted relative residual above this means the factorization did not
# solve the system at all (the answer is worse than `x = 0`).  Only then does the
# *priority* tiny-shift phase (rule 2) run, so a solve that already produced a
# usable answer -- however marginal -- is bitwise unchanged on the historical
# path.  The threshold is deliberately not lowered: the last-resort phase (rule
# 7) covers the states just *below* it, where the historical rungs can still sit
# inside the wall-strip cluster (G step 7641 measured 0.9654, 3.5 % below 1.0;
# docs/dev/CASE_REALISTIC.md §17.3-§17.4).
const SEMSTOKES_LOST_TOL = 1.0
# Allowed `stokes_scaling` values of the direct path
# (docs/dev/BUG_LEDGER.md): `:none` is the historical unscaled saddle,
# bit for bit; `:equilibrated` is the symmetric congruence below.  No other
# value changes an existing path.
const SEMSTOKES_SCALINGS = (:none, :equilibrated)

"""
    _semquad_stokes_equil_diag(Kred, Bt, Bs) -> Vector{Float64}

Symmetric equilibration diagonal `d = [d_u; d_p]` for the reduced saddle
`S = [Kred Bt; Bs 0]` (docs/dev/BUG_LEDGER.md):

* `d_u = 1/sqrt(diag Kred)` removes the viscosity scale from the momentum block
  (the lid's `eta = 1e4` against the interior's `1e-3` is the whole reason the
  wall checkerboard is weak: `beta ~ 0.696/sqrt(eta_lid)`,
  docs/dev/BUG_LEDGER.md §7);
* `d_p = 2 max|Kred| / max(|Bt|, |Bs|)` is a single scalar that balances the
  constraint block against the momentum block: it makes `d_p |B|` comparable to
  `|K|`.  This is the max-norm equilibration of the two blocks, i.e. the
  matrix-derived form of the `eta/h` pressure scale, with a factor 2 of margin
  (the measured plateau starts at ~0.3 of `max|K|/max|B|` and extends over four
  decades, docs/dev/BUG_LEDGER.md §2.1).

The congruence `S -> D S D`, `rhs -> D rhs` has the exact same solution
(`x = D y`); it changes only the arithmetic of the factorisation.  In the
scaled operator the wall-strip pressure family is lifted far above the
round-off floor: the measured smallest eigenvalue of the scaled operator is a
*momentum* mode at `4.2e-11` relative (AMR 7641), not the `1.7e-14` pressure
mode of the unscaled system.
"""
function _semquad_stokes_equil_diag(Kred::SparseMatrixCSC, Bt::SparseMatrixCSC,
                                   Bs::SparseMatrixCSC)
    dk = diag(Kred)
    du = 1.0 ./ sqrt.(max.(dk, 1.0e-14 * maximum(dk)))
    bmax = max(maximum(abs, Bt), maximum(abs, Bs), 1.0e-300)
    dp = fill(2.0 * maximum(abs, Kred) / bmax, size(Bs, 1))
    return vcat(du, dp)
end

"""
    _semquad_stokes_equil_solve(S, Kred, Bt, Bs, rhs) -> (x, d)

One unshifted LU of the equilibrated saddle: `D = diag(d)`,
`x = D * ((D*S*D) \\ (D*rhs))`.  `x` solves `S x = rhs` to the accuracy of the
scaled factorisation (no shift of any kind); `d` is the equilibration diagonal
of [`_semquad_stokes_equil_diag`](@ref).  Exposed for the G214 gate, which
checks the diagonal and the congruence on a real saddle.
"""
function _semquad_stokes_equil_solve(S::SparseMatrixCSC, Kred::SparseMatrixCSC,
                                    Bt::SparseMatrixCSC, Bs::SparseMatrixCSC,
                                    rhs::Vector{Float64})
    d = _semquad_stokes_equil_diag(Kred, Bt, Bs)
    D = spdiagm(0 => d)
    return D * ((D * S * D) \ (D * rhs)), d
end

"""
    _shift_ladder_solve(S, rhs; accept_tol=Inf, bad_tol=1e-8, resid=nothing,
                        shifts=SEMSTOKES_SHIFT_LADDER,
                        tiny_shifts=SEMSTOKES_TINY_SHIFTS,
                        lost_tol=SEMSTOKES_LOST_TOL,
                        refine_maxit=4, refine_tol=1e-10)

Sparse direct solve of the reduced saddle `S x = rhs` with the diagonal-shift
ladder.  Returns `(x, r, shift, refine_iters, rungs)`:

* `x`, `r` — the accepted solution and its relative residual,
* `shift` — the relative diagonal shift of the factorisation behind it: `0.0`
  for the unshifted LU, otherwise one of `tiny_shifts`/`shifts`.  When the
  returned iterate comes from the refinement, it is the shift of the
  factorisation the refinement was preconditioned by,
* `refine_iters` — GMRES iterative-refinement iterations spent after the ladder
  (0 when the ladder's own rung was accepted),
* `rungs` — every rung tried, in order, as `(shift, residual)`; the refinement,
  when it ran, appends `(:refine, residual)`.  `nothing` for a healthy solve.

The phase that produced the returned answer is recorded in
`LAST_SEMSTOKES_LADDER_PHASE` (`:healthy`, `:tiny`, `:ladder`, `:refine`,
`:fallback` or `:fail`; the direct branch also records `:equilibrated` for a
`stokes_scaling = :equilibrated` answer -- see that keyword).

The rules, in the order they apply:

1. **Healthy solve.**  If the unshifted factorisation already satisfies
   `r <= bad_tol` and `r <= accept_tol`, it is returned untouched — one
   factorisation, no shift, `rungs = nothing`.  This is the historical
   behaviour of the direct branch and is what keeps healthy systems bitwise
   identical.
2. **Lost solution** (docs/dev/CASE_REALISTIC.md §13.10).  If `r > lost_tol` the
   unshifted LU did not solve the system at all (its answer is worse than
   `x = 0`): the near-null cluster has swallowed the factorisation, and the
   diagonal shift then has to be chosen at the *cluster's* scale rather than
   far above it, because the shift biases the answer by `shift*|x|`.  The
   `tiny_shifts` are tried in order and the first one that both improves on `r`
   and meets `accept_tol` is accepted; a rung that does not improve is rejected,
   so this phase can never make an answer worse, and if none improves the
   ladder continues exactly as before (`rungs` records every attempt).
3. **First rung.**  Otherwise the shifts are tried in order, *refactoring* each
   time (`lu(S + shift*dmax*I)`, `dmax = max|diag S| + 1` as in the historical
   retry).  The first rung whose residual is `<= accept_tol` is accepted and the
   ladder stops.  With the default `accept_tol = Inf` that is the first rung,
   i.e. exactly the old "retry once with 1e-12" behaviour, including its answer
   for a system whose shifted residual is *worse* than the unshifted one.
4. **Escalation.**  With a finite `accept_tol` a rung that still misses it
   escalates to the next shift.  The best solve seen -- *including the
   unshifted one* -- is remembered, and escalation stops as soon as that best
   meets the gate, so a larger shift (which perturbs the operator more) can
   never throw away an answer that already passed.
5. **Iterative refinement.**  If nothing meets `accept_tol`, a few GMRES
   iterations against the *unshifted* `S`, preconditioned by the best (else the
   last) rung's factorisation, polish the iterate (the `:lagged` machinery: the
   shift makes the factorisation stable, the refinement removes its
   perturbation).  Only an improvement is kept.
6. **Return the best.**  If even that misses `accept_tol` the best iterate is
   returned with its residual; the caller decides (the driver keeps the fatal
   error as the last resort).
7. **Last-resort tiny shifts** (docs/dev/CASE_REALISTIC.md §17.4, G212).  A finite
   unshifted residual at or below `lost_tol` skips rule 2, and the historical
   rungs can then sit inside the wall-strip cluster: G step 7641 measured
   `r = 0.9654` (3.5 % below the 1.0 trigger), `1e-12 -> 1.56e-2`, and four
   refinements only reached 4.47e-3.  If the ladder and its refinement (and
   rule 2, when it ran) all missed `accept_tol`, the unused `tiny_shifts` are
   tried here; the best answer seen is kept -- including the refinement's, so a
   tiny rung only wins by improving on it -- and the solve still fails if they
   all miss.  This costs at most `length(tiny_shifts)` factorisations and only
   runs on a solve that was about to abort: a healthy solve returns at rule 1,
   an accepted rung at rule 3/4, and `accept_tol = Inf` callers never enter the
   phase, so the historical numerics are bitwise unchanged.  `tiny_shifts = ()`
   disables both tiny phases; `lost_tol` only gates rule 2.

`resid(x)` measures the relative residual; the default is
`norm(S*x - rhs)/max(norm(rhs), 1)`.  The driver passes the block-split saddle
residual (momentum rows / continuity rows), which is the number its Stokes gate
checks.  A non-finite `resid` (NaN/Inf from a singular factorisation) is treated
as a failure and recorded as `Inf`, so the ladder fires and a finite rung can
still win; the old path accepted a NaN as "not bad" because `NaN > tol` is
false.
"""
function _shift_ladder_solve(S::SparseMatrixCSC, rhs::Vector{Float64};
                             accept_tol::Float64=Inf,
                             bad_tol::Float64=1.0e-8,
                             resid::Union{Nothing,Function}=nothing,
                             shifts=SEMSTOKES_SHIFT_LADDER,
                             tiny_shifts=SEMSTOKES_TINY_SHIFTS,
                             lost_tol::Float64=SEMSTOKES_LOST_TOL,
                             refine_maxit::Int=4,
                             refine_tol::Float64=1.0e-10)
    rel = resid === nothing ?
          (x -> norm(S * x - rhs) / max(norm(rhs), 1.0)) : resid
    x = S \ rhs
    r = rel(x)
    # A non-finite residual (NaN/Inf from a singular factorisation) is a
    # failure, not "healthy": record it as `Inf` so the ladder fires, a finite
    # rung can win, and the returned residual is then finite.  Before the ladder
    # a NaN was silently accepted as "not bad" (`NaN > tol` is false).
    r = isfinite(r) ? r : Inf
    if !(r > bad_tol) && !(r > accept_tol)
        LAST_SEMSTOKES_LADDER_PHASE[] = :healthy
        return x, r, 0.0, 0, nothing
    end
    dmax = maximum(abs, diag(S)) + 1.0
    rungs = Any[(0.0, r)]
    xbest, rbest, shbest = x, r, 0.0
    Fbest = nothing
    Fshift = nothing
    rshift = Inf
    shshift = 0.0
    # Set when rule 2 has tried the tiny shifts, so rule 7 does not repeat them.
    tiny_used = false
    # --- rule 2: the unshifted LU lost the solution --------------------------
    # A *non-finite* residual is not this case: that is a broken factorisation
    # (a NaN/Inf pivot), not a biased one, and the historical shifted rungs are
    # what recovers it (§13.9/G207).  The tiny shifts are for the finite-but-lost
    # regime the wall-refined realistic tree produces.
    if !isempty(tiny_shifts) && isfinite(r) && r > lost_tol
        tiny_used = true
        for sh in tiny_shifts
            F = lu(S + (sh * dmax) * I)
            xs = F \ rhs
            rs = rel(xs)
            push!(rungs, (Float64(sh), rs))
            if rs < rshift
                Fshift, rshift, shshift = F, rs, Float64(sh)
            end
            if isfinite(rs) && rs <= accept_tol && rs < r
                # A genuine improvement on a lost solve: accept it.  `rs < r`
                # is what keeps this phase monotone -- it can never hand back
                # something worse than the unshifted answer it replaced.
                xbest, rbest, shbest, Fbest = xs, rs, Float64(sh), F
                break
            end
            if isfinite(rs) && rs < rbest
                xbest, rbest, shbest, Fbest = xs, rs, Float64(sh), F
            end
        end
        if rbest < r && rbest <= accept_tol
            LAST_SEMSTOKES_LADDER_PHASE[] = :tiny
            return xbest, rbest, shbest, 0, rungs
        end
    end
    for sh in shifts
        F = lu(S + (sh * dmax) * I)
        xs = F \ rhs
        rs = rel(xs)
        push!(rungs, (Float64(sh), rs))
        if rs < rshift
            Fshift, rshift, shshift = F, rs, Float64(sh)
        end
        if rs <= accept_tol
            # This rung meets the gate: accept it.  With the default
            # `accept_tol = Inf` this is the historical "retry once with 1e-12
            # and take the answer", including for a system whose shifted
            # residual is worse than the unshifted one.
            xbest, rbest, shbest, Fbest = xs, rs, Float64(sh), F
            break
        end
        if rs < rbest
            xbest, rbest, shbest, Fbest = xs, rs, Float64(sh), F
        end
        # Stop escalating when what we already have passes: a larger shift
        # perturbs the operator more, so a rung that is *worse* than the
        # unshifted solve must not cost us the unshifted answer.  Measured on
        # the step-6130 realistic state: the unshifted residual is 4.0e-4 (it
        # passes the 1e-3 card gate) while the 1e-12 rung is 1.3e-3 (it does
        # not), so the ladder returns the unshifted solve instead of refining.
        rbest <= accept_tol && break
    end
    nref = 0
    phase = :ladder
    if rbest > accept_tol
        # The preconditioner of the refinement is the factorisation of the best
        # *shifted* rung: a shifted factorisation is the well-conditioned one
        # (that is the whole point of the rung), and the refinement then
        # iterates against the unshifted operator to remove the shift's bias.
        F = Fbest === nothing ? Fshift : Fbest
        shF = Fbest === nothing ? shshift : shbest
        if F !== nothing
            xtol = min(refine_tol, 0.01 * accept_tol)
            xr, _, nref = gmres_restarted(v -> S * v, rhs;
                                          precond=r -> F \ r,
                                          tol=xtol, maxit=refine_maxit,
                                          restart=refine_maxit, x0=xbest)
            rr = rel(xr)
            push!(rungs, (:refine, rr))
            if rr < rbest
                # The returned iterate came from the refinement, so report the
                # shift of the factorisation it was preconditioned by (its
                # solve is no longer what we return, but its conditioning is
                # what made the polish possible).
                xbest, rbest, shbest = xr, rr, shF
                phase = :refine
            end
        end
    end
    # --- rule 7: the last-resort tiny shifts ---------------------------------
    # A finite unshifted residual at or below `lost_tol` skips rule 2, and the
    # historical rungs can then sit inside the wall-strip cluster (G step 7641:
    # r = 0.9654, 1e-12 -> 1.56e-2, refinement -> 4.47e-3).  If the ladder has
    # already failed, the tiny shifts are the only untried rungs.  The best
    # answer seen so far is kept, so this phase can never make the answer worse;
    # only an improvement is returned.  It runs only on a solve that is about to
    # abort, and never when rule 2 already tried these shifts.
    if rbest > accept_tol && !isempty(tiny_shifts) && isfinite(r) && !tiny_used
        for sh in tiny_shifts
            F = lu(S + (sh * dmax) * I)
            xs = F \ rhs
            rs = rel(xs)
            rs = isfinite(rs) ? rs : Inf
            push!(rungs, (Float64(sh), rs))
            if rs < rbest
                xbest, rbest, shbest = xs, rs, Float64(sh)
                phase = :fallback
            end
            rbest <= accept_tol && break
        end
    end
    phase = rbest <= accept_tol ? phase : :fail
    LAST_SEMSTOKES_LADDER_PHASE[] = phase
    return xbest, rbest, shbest, nref, rungs
end

