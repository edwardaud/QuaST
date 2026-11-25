# Constant-viscosity quadtree Stokes solve and point evaluation (split from stokes/solve.jl).
"""Assemble and solve the constant-viscosity quadtree Stokes system."""
function solve_semquad_stokes(ss::SemStokes, qt::Quadtree,
                              fx::Function, fz::Function,
                              uDx::Function, uDz::Function;
                              linear::Symbol=:direct, schur_maxiter::Int=200,
                              schur_precond::Symbol=:mass)
    # This entry point is the legacy uniform-mesh path: its element matrices are
    # the *reference* matrices, which are the physical ones only because the
    # area Jacobian cancels the derivative factors on an axis-aligned map
    # (GEOM_DESIGN §1.6 item 2).  A mapped geometry must not silently take that
    # shortcut: refuse instead (cut 2 routes the variable-viscosity assembly
    # through the metric).
    ss.su.geom isa CartesianBox ||
        throw(ArgumentError("solve_semquad_stokes is the box-only " *
                            "reference-matrix path; a mapped geometry needs " *
                            "the metric-based assembly (GEOM_DESIGN cut 2)"))
    linear in (:direct, :schur_cg) ||
        throw(ArgumentError("linear must be :direct or :schur_cg"))
    schur_precond in (:none, :mass) ||
        throw(ArgumentError("schur_precond must be :none or :mass"))
    p = ss.p; q = ss.q
    su = ss.su; sp = ss.sp
    nu = su.N; np = sp.N
    Nu2 = 2 * nu
    IK = Int[]; JK = Int[]; VK = Float64[]
    IB = Int[]; JB = Int[]; VB = Float64[]
    fv = zeros(Float64, Nu2)
    Axx, Azz, Axz, Bx, Bz = _sem_stokes_elem(p, q)
    n = p + 1
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        # The reference-space element matrices are already the physical ones for
        # square leaves: the area jacobian (hx*hy/4) cancels the derivative
        # factors (2/hx)(2/hy) for the second-order terms, and leaves the aspect
        # ratios hy/hx and hx/hy (both 1 for the unit square).
        axx = hy / hx
        azz = hx / hy
        # scatter the velocity block (local order: (j-1)n+i for each component)
        for jb in 1:n, ib in 1:n, ja in 1:n, ia in 1:n
            ki = uids[(jb - 1) * n + ib]; kj = uids[(ja - 1) * n + ia]
            push!(IK, ki); push!(JK, kj)
            push!(VK, axx * Axx[(jb - 1) * n + ib, (ja - 1) * n + ia])
            push!(IK, nu + ki); push!(JK, nu + kj)
            push!(VK, azz * Azz[(jb - 1) * n + ib, (ja - 1) * n + ia])
            push!(IK, ki); push!(JK, nu + kj)
            push!(VK, Axz[(jb - 1) * n + ib, (ja - 1) * n + ia])
            push!(IK, nu + ki); push!(JK, kj)
            push!(VK, Axz[(ja - 1) * n + ia, (jb - 1) * n + ib])
        end
        # divergence coupling: int q d(ux)/dx = (hy/2) int_ref q d/dxi,
        #                      int q d(uz)/dy = (hx/2) int_ref q d/deta
        for b in 1:(q + 1), a in 1:(q + 1), j in 1:n, i in 1:n
            pi_ = pids[(b - 1) * (q + 1) + a]
            uj = uids[(j - 1) * n + i]
            push!(IB, pi_); push!(JB, uj)
            push!(VB, 0.5 * hy * Bx[(b - 1) * (q + 1) + a, (j - 1) * n + i])
            push!(IB, pi_); push!(JB, nu + uj)
            push!(VB, 0.5 * hx * Bz[(b - 1) * (q + 1) + a, (j - 1) * n + i])
        end
        # loads
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (su.xi[i] + 1.0) * hx
            y = y0 + 0.5 * (su.xi[j] + 1.0) * hy
            w = 0.25 * hx * hy * su.w[i] * su.w[j]
            g = uids[(j - 1) * n + i]
            fxv = fx(x, y)
            fzv = fz(x, y)
            # F5 guard: name a non-finite body force at the sample point (cold
            # throw path; the constant-viscosity path is verification-only, but
            # a silent NaN here is the same defect).  Coordinates are numbers,
            # not an interpolated string, so the success path does not allocate.
            _require_load!(fxv, "the Stokes body force fx(x, y)", x, y)
            _require_load!(fzv, "the Stokes body force fz(x, y)", x, y)
            fv[g] += w * fxv
            fv[nu + g] += w * fzv
        end
    end
    K = sparse(IK, JK, VK, Nu2, Nu2)
    B = sparse(IB, JB, VB, np, Nu2)
    # velocity Dirichlet on the domain boundary
    fixed = Dict{Int,Float64}()
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                xf = dir === :xplus ? x0 + hx : x0
                for j in 1:n
                    yv = y0 + 0.5 * (su.xi[j] + 1.0) * hy
                    g = uids[(j - 1) * n + iF]
                    fixed[g] = uDx(xf, yv)
                    fixed[nu + g] = uDz(xf, yv)
                end
            else
                jF = dir === :yminus ? 1 : n
                yf = dir === :yplus ? y0 + hy : y0
                for i in 1:n
                    xv = x0 + 0.5 * (su.xi[i] + 1.0) * hx
                    g = uids[(jF - 1) * n + i]
                    fixed[g] = uDx(xv, yf)
                    fixed[nu + g] = uDz(xv, yf)
                end
            end
        end
    end
    cons_u = semquad_constraints(su, qt)
    # lift to the two velocity components
    cons2 = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in cons_u
        cons2[c] = (ms, ws)
        cons2[nu + c] = (nu .+ ms, ws)
    end
    u0, Cu, _ = _affine_constraint(Nu2, cons2, fixed)
    # pressure: hanging constraints and one pin
    cons_p = semquad_constraints(sp, qt)
    pf = Dict{Int,Float64}()
    pfree0 = setdiff(1:np, collect(keys(cons_p)))
    isempty(pfree0) || (pf[first(pfree0)] = 0.0)
    p0, Cp, _ = _affine_constraint(np, cons_p, pf)
    # reduced saddle system: K u - B' p = f, -B u = 0 (the weak form has
    # -int p div v - int q div u, so the pressure blocks carry a minus sign)
    Fp = nothing
    if linear === :schur_cg && schur_precond === :mass
        Mfull = assemble_semquad_pressure_mass(sp)
        Mred = Cp' * Mfull * Cp
        Fp = cholesky(Symmetric(Mred))
        issuccess(Fp) ||
            throw(ArgumentError("pressure mass preconditioner is not SPD"))
    end
    CpB = Cp' * B * Cu
    Kred = Cu' * K * Cu
    fr = Cu' * (fv - K * u0 + B' * p0)
    # F5 guard: the reduced RHS is what the solve sees; it also carries the
    # Dirichlet data uDx/uDz.
    _require_finite!(fr, "the reduced Stokes right-hand side fr",
                     "the boundary data (uDx/uDz) of this solve")
    gr = Cp' * (B * u0)
    nf = size(Cu, 2); nfp = size(Cp, 2)
    Bs = -CpB
    Bt = sparse(transpose(Bs))
    nit = 0
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
    else
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        if linear === :lagged
            # S2: reuse the previous LU as a preconditioner and polish with
            # iterative refinement; refactor only on a cache miss / stall.
            key = _saddle_key(qt, ss.su.p, bc, nf, nfp)
            sol, rn = _lagged_solve!(S, rhs, key; tol=1.0e-8, maxit=25)
            LAST_SEMSTOKES_REGULARIZED[] = false
            if rn > 1.0e-6
                dmax = maximum(abs, diag(S)) + 1.0
                Ssh = S + (1.0e-12 * dmax) * I
                sol, rn = _lagged_solve!(Ssh, rhs, key; tol=1.0e-8, maxit=6,
                                         force_refactor=true)
                LAST_SEMSTOKES_REGULARIZED[] = true
            end
        else
            sol = S \ rhs
            # See `solve_semquad_stokes_var`: the unshifted LU can be unstable on
            # free-slip wall trees; retry with a 1e-12 relative diagonal shift only
            # when the first solve leaves a bad residual.
            rden0 = max(norm(fr), norm(gr), 1.0)
            bad = norm(Kred * sol[1:nf] + Bt * sol[(nf + 1):(nf + nfp)] - fr) >
                  1.0e-8 * rden0 ||
                  norm(Bs * sol[1:nf] - gr) > 1.0e-8 * rden0
            LAST_SEMSTOKES_REGULARIZED[] = bad
            if bad
                dmax = maximum(abs, diag(S)) + 1.0
                sol = (S + (1.0e-12 * dmax) * I) \ rhs
            end
        end
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    end
    # True reduced-saddle residual of the system that was actually solved
    # (Schur uses the regularized Kred).  This is what distinguishes a
    # converged solve from a singular/ill-conditioned direct solve.
    r1 = Kred * ufree + Bt * pfree - fr
    r2 = Bs * ufree - gr
    rden = max(norm(fr), norm(gr), 1.0)
    LAST_SEMSTOKES_RESIDUAL[] = max(norm(r1), norm(r2)) / rden
    LAST_SEMSTOKES_SCHUR_ITERS[] = nit
    u = u0 + Cu * ufree
    pv = p0 + Cp * pfree
    return u, pv, fixed, cons2, cons_p
end

"""Evaluate a quadtree velocity/pressure field at (x,y)."""
function semstokes_eval(ss::SemStokes, u::Vector{Float64}, pv::Vector{Float64},
                        x::Float64, y::Float64)
    # Legacy box-only point search (physical coordinate interval test); the
    # mapped path uses `core/eval.jl`'s geometry-dispatched evaluation.
    ss.su.geom isa CartesianBox ||
        throw(ArgumentError("semstokes_eval is the box-only physical-point " *
                            "search; use the geometry-aware evaluation of " *
                            "core/eval.jl for a mapped geometry"))
    nu = ss.su.N
    n = ss.p + 1
    rule = gll_rule(ss.p)
    for lf in ss.su.leaves
        hx = leaf_hx(ss.su, lf[1]); hy = leaf_hy(ss.su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        if x >= x0 - 1.0e-14 && x <= x0 + hx + 1.0e-14 &&
           y >= y0 - 1.0e-14 && y <= y0 + hy + 1.0e-14
            xi = 2.0 * (x - x0) / hx - 1.0
            eta = 2.0 * (y - y0) / hy - 1.0
            Lx, _ = lagrange_eval(rule, xi)
            Ly, _ = lagrange_eval(rule, eta)
            ids = ss.su.elem[lf]
            ux = 0.0; uz = 0.0
            for j in 1:n, i in 1:n
                g = ids[(j - 1) * n + i]
                ux += u[g] * Lx[i] * Ly[j]
                uz += u[nu + g] * Lx[i] * Ly[j]
            end
            q = ss.q
            nq = q + 1
            rq = gll_rule(q)
            Lxq, _ = lagrange_eval(rq, xi)
            Lyq, _ = lagrange_eval(rq, eta)
            pids = ss.sp.elem[lf]
            pp = 0.0
            for b in 1:nq, a in 1:nq
                pp += pv[pids[(b - 1) * nq + a]] * Lxq[a] * Lyq[b]
            end
            return ux, uz, pp
        end
    end
    throw(ArgumentError("point ($x,$y) not in the tree"))
end

