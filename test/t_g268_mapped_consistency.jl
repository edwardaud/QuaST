# G268: cross-path consistency of the _mapped_ Stokes kernel fed the box map
# against the untouched box kernel (stage 0, cut 2).
#
# The two kernels are different code bodies: the box kernel is the historical
# arithmetic (`2.0/hx` folded into the gradient, `dphi*phi*ihx`, `0.25*hx*hy`,
# `0.5*h*gl_w`, the Cartesian face assignment tables), while the mapped kernel
# reads the leaf metric (`quad_point`, `Q.quad_measure`, `basis_grad` chain rule)
# and the face metric (`face_point`, `face_weight`, `face_normal`,
# `face_tangent`).  Fed the **box map** (`AffineBox` with `A = I`, `b = 0`,
# `Lx = qt.Lx`), the mapped kernel is mathematically the same operator, so every
# assembled entry and every direct solve must agree to round-off.  They are not
# required to be bitwise (GEOM_DESIGN §5.3); on the dyadic trees they actually
# are bitwise, on a non-dyadic leaf layout the sampled coordinates differ by an
# ulp, which is the interesting case for the 1e-14 bounds.
#
# What each block catches (all are failures of the *mapped* body, which the box
# suite cannot see):
#   * `K`, `Kred`: a wrong volume weight `quad_measure`, a wrong gradient
#     scale/association in `basis_grad`, or a swapped x/z row in the stiffness
#     accumulations;
#   * `B`, `Bc`, `Bm`, `Bs`, `Bt`: the divergence block's gradient components and
#     the transpose/sign relations of the saddle (`Bt = Bs'` when `Bm == Bc`);
#   * pressure mass and stiffness: the scalar mapped measure and the pressure
#     basis gradients (a different basis degree from the velocity one — the
#     first implementation of the mapped stiffness reused the velocity basis and
#     failed here by 2.8x, see GEOM_PROGRESS.md);
#   * `B` with a density (`rho_fun`/`drho_fun`) and the compressibility block
#     `Pcm`: the mapped `div(rho u)` term `u.grad rho` and the component signs of
#     the gravity-direction block;
#   * the traction load `fv`: the mapped face Jacobian, outward normal and
#     positive tangent, compared against the historical global-component
#     assignment with the convention conversion `wall_tn_box = +-t.n`,
#     `wall_t_box = t.tau` written explicitly in this file;
#   * the direct-solve result: the mapped Dirichlet closure positions (a
#     closure evaluated at *logical* instead of physical positions changes the
#     prescribed data and therefore the solution), the fixed-dof set, the
#     net-flux check (`_semquad_wall_normal_flux` with the mapped normal) and
#     the mapped `perimeter`.
#
# Why this is not tautological: the comparison is between two independent
# implementations — the box kernel is not routed through the geometry (that is
# what keeps the nine state-dump hashes) and the mapped kernel contains no box
# formula.  A shared mistake is only possible where both read the same data
# (`su.coords`, the closures); every operator is rebuilt from the leaf metric
# on one side and from `hx`/`hy` on the other.

using Test, QuaST, LinearAlgebra, SparseArrays

const MS268 = QuaST

# ---------------------------------------------------------------------------
# trees: a dyadic unit square (mapped coordinates happen to be bitwise the box
# ones, so the agreement is bitwise there), a non-dyadic layout where the mapped
# coordinates are NOT bitwise the box ones (`Lx = 7/3`, base cells 7x3: measured
# max coordinate difference 4.4e-16, K relative difference 2.1e-16), and a
# mixed-level hanging tree (constraints active)
# ---------------------------------------------------------------------------
function g268_trees()
    out = Pair{String,Quadtree}[]
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    push!(out, "unit" => qt)
    q73 = Quadtree(2; cells_x=7, cells_y=3)
    for _ in 1:2, lf in collect(q73.leaves)
        refine!(q73, lf)
    end
    balance!(q73)
    push!(out, "cells73" => q73)
    qm = Quadtree(3)
    for lf in collect(qm.leaves)
        refine!(qm, lf)
    end
    for lf in sort(collect(qm.leaves))[1:3]
        refine!(qm, lf)
    end
    balance!(qm)
    push!(out, "mixed" => qm)
    return out
end

g268_boxmap(qt) = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), qt.Lx)

# relative disagreement: max |a-b| / max |b| (0 when equal, including both zero)
function g268_rel(a, b)
    s = maximum(abs, b)
    d = maximum(abs, a - b)
    return s == 0.0 ? d : d / s
end

g268_eta(x, y) = 1.0 + 0.2 * x + 0.1 * y
g268_fx(x, y) = 2.0
g268_fz(x, y) = -0.5
g268_ux(x, y) = x^2 - y^2
g268_uz(x, y) = -2.0 * x * y
g268_pex(x, y) = 1.0 + 2.0 * x - 0.5 * y

function g268_assemble(ss, qt; wall_bc=(:same, :same, :same, :same),
                       wall_t=(nothing, nothing, nothing, nothing),
                       wall_tn=(nothing, nothing, nothing, nothing))
    return solve_semquad_stokes_var(ss, qt, g268_eta, g268_fx, g268_fz,
                                    g268_ux, g268_uz;
                                    linear=:direct, bc=:rigid,
                                    wall_bc=wall_bc, wall_t=wall_t,
                                    wall_tn=wall_tn, assemble_only=true)
end

# one traction load on wall `wi`, in the geometry's own convention:
#   box:  data are global components in the positive coordinate direction
#         (wall_tn = global normal-direction component, wall_t = global tangent)
#   mapped: data are t.n and t.tau
# with the box->mapped conversion tn_mapped = s_i * tn_box (s = -1 on
# xminus/yminus because the box convention absorbs the outward sign there) and
# t_mapped = t_box (the positive tangent is +e_y / +e_x on the box).
function g268_traction_fv(ss, qt, wi::Int, tn_box::Function, t_box::Function;
                          mapped::Bool)
    wbc = ntuple(i -> i == wi ? :open : :same, 4)
    s = (wi == 1 || wi == 3) ? -1.0 : 1.0
    tn = mapped ? ((x, y) -> s * tn_box(x, y)) : tn_box
    tt = t_box
    wt = ntuple(i -> i == wi ? tt : nothing, 4)
    wtn = ntuple(i -> i == wi ? tn : nothing, 4)
    asm = g268_assemble(ss, qt; wall_bc=wbc, wall_t=wt, wall_tn=wtn)
    return asm.fv
end

@testset "G268 mapped-vs-box consistency (box map through the mapped kernel)" begin
    p = 2
    for (tag, qt) in g268_trees()
        @testset "$tag" begin
            gid = g268_boxmap(qt)
            ssb = make_semstokes(qt, p)
            ssm = make_semstokes(qt, p; geom=gid)
            @test ssm.su.geom === gid
            @test ssm.sp.geom === gid

            asb = g268_assemble(ssb, qt)
            asm = g268_assemble(ssm, qt)

            # ---- assembled operator entries ----------------------------------
            maxop = 0.0
            maxrhs = 0.0
            # (b is the box/historical value, a the mapped one; every sparse
            # matrix is compared entry by entry through the max-abs difference)
            for (nm, a, b) in (("K", asm.K, asb.K),
                               ("Kred", asm.Kred, asb.Kred),
                               ("B", asm.B, asb.B),
                               ("Bc", asm.Bc, asb.Bc),
                               ("Bm", asm.Bm, asb.Bm),
                               ("Bs", asm.Bs, asb.Bs),
                               ("Bt", asm.Bt, asb.Bt),
                               ("Cu", asm.Cu, asb.Cu),
                               ("Cp", asm.Cp, asb.Cp))
                @test size(a) == size(b)
                r = g268_rel(a, b)
                maxop = max(maxop, r)
                @test r <= 1.0e-14
            end
            for (nm, a, b) in (("fv", asm.fv, asb.fv),
                               ("fr", asm.fr, asb.fr),
                               ("gr", asm.gr, asb.gr),
                               ("u0", asm.u0, asb.u0),
                               ("p0", asm.p0, asb.p0))
                @test size(a) == size(b)
                maxrhs = max(maxrhs, g268_rel(a, b))
                @test maxrhs <= 1.0e-14
            end
            # the hanging tree really has constraints, so `Cu`/`Cp` above are
            # not trivially the identity
            if tag == "mixed"
                @test size(asb.Cu, 2) < 2 * ssb.su.N
                @test !isempty(semquad_constraints(ssb.su, qt))
                @test !isempty(semquad_constraints(ssm.su, qt))
            end

            # ---- pressure mass and stiffness (scalar mapped measure) --------
            for (nm, Mfun) in (("mass", MS268.assemble_semquad_pressure_mass),
                               ("stiffness",
                                MS268.assemble_semquad_pressure_stiffness))
                Mb = Mfun(ssb.sp)
                Mm = Mfun(ssm.sp)
                @test size(Mm) == size(Mb)
                @test g268_rel(Mm, Mb) <= 1.0e-14
                w = (x, y) -> 1.0 + 0.3 * x - 0.2 * y
                Mbw = Mfun(ssb.sp; weight=w)
                Mmw = Mfun(ssm.sp; weight=w)
                @test g268_rel(Mmw, Mbw) <= 1.0e-14
            end

            # ---- anelastic mass and compressibility block ---------------------
            # A height-dependent density with its analytic derivative: no
            # finite-difference fallback (a FD of a 1-ulp coordinate difference
            # would be visible at 1e-10 and is not the object of this gate).
            rhof = (h) -> exp(0.3 * h)
            drhof = (h) -> 0.3 * exp(0.3 * h)
            Bb = semquad_stokes_B(ssb, qt; rho_fun=rhof, drho_fun=drhof)
            Bm = semquad_stokes_B(ssm, qt; rho_fun=rhof, drho_fun=drhof)
            @test size(Bm) == size(Bb)
            @test g268_rel(Bm, Bb) <= 1.0e-14
            Pb = semquad_stokes_Pcm(ssb, qt; rho_fun=rhof, gamma=0.7, Di=0.4)
            Pm = semquad_stokes_Pcm(ssm, qt; rho_fun=rhof, gamma=0.7, Di=0.4)
            @test size(Pm) == size(Pb)
            @test g268_rel(Pm, Pb) <= 1.0e-14

            # ---- traction load on each logical wall --------------------------
            tnfun = (x, y) -> 1.0 + 0.2 * x - 0.3 * y
            tfun = (x, y) -> -0.4 + 0.1 * x + 0.25 * y
            for wi in 1:4
                fvb = g268_traction_fv(ssb, qt, wi, tnfun, tfun; mapped=false)
                fvm = g268_traction_fv(ssm, qt, wi, tnfun, tfun; mapped=true)
                @test g268_rel(fvm, fvb) <= 1.0e-14
            end

            # ---- mapped boundary measure on the box map ----------------------
            bmask = MS268._semquad_boundary_face_mask(qt, ssb.su)
            per = perimeter(gid, ssb.su, bmask)
            @test abs(per - (2.0 * (qt.Lx + 1.0))) <= 1.0e-14 * (qt.Lx + 1.0)

            # ---- direct solve result ----------------------------------------
            ub, pbv, _, _, _ = solve_semquad_stokes_var(ssb, qt, g268_eta,
                                                        g268_fx, g268_fz,
                                                        g268_ux, g268_uz;
                                                        linear=:direct, bc=:rigid)
            um, pmv, _, _, _ = solve_semquad_stokes_var(ssm, qt, g268_eta,
                                                        g268_fx, g268_fz,
                                                        g268_ux, g268_uz;
                                                        linear=:direct, bc=:rigid)
            @test g268_rel(um, ub) <= 1.0e-11
            # pressure up to its one free constant (the pinned dof is the same
            # dof for both spaces, so no constant remains; the threshold is
            # looser than the velocity one because the saddle solve amplifies
            # the last-bit operator difference: measured 1.4e-11 on the
            # non-dyadic tree against 4e-16 on the operator entries)
            prel = g268_rel(pmv .- sum(pmv) / length(pmv),
                            pbv .- sum(pbv) / length(pbv))
            @test prel <= 1.0e-10
            # exactness sanity on the box map with CONSTANT viscosity: the mapped
            # space contains the physical P2/P1 fields, so both paths reproduce
            # them to round-off.  (The variable-eta solve above is only an
            # operator comparison: a polynomial field is not the exact solution
            # of the variable-viscosity problem, that is G154/G267's separate
            # constant-eta statement.)
            eta1 = (x, y) -> 1.0
            ub1, _, _, _, _ = solve_semquad_stokes_var(ssb, qt, eta1,
                                                       g268_fx, g268_fz,
                                                       g268_ux, g268_uz;
                                                       linear=:direct, bc=:rigid)
            um1, _, _, _, _ = solve_semquad_stokes_var(ssm, qt, eta1,
                                                       g268_fx, g268_fz,
                                                       g268_ux, g268_uz;
                                                       linear=:direct, bc=:rigid)
            @test g268_rel(um1, ub1) <= 1.0e-11
            nu = ssm.su.N
            eu = 0.0
            for (gi, c) in enumerate(ssm.su.coords)
                eu = max(eu, abs(um1[gi] - g268_ux(c...)),
                             abs(um1[nu + gi] - g268_uz(c...)))
            end
            @test eu < 1.0e-10
            println("G268 ", rpad(tag, 8), " max operator rel=", maxop,
                    "  max RHS rel=", maxrhs,
                    "  solve u rel=", g268_rel(um, ub), "  p rel=", prel,
                    "  traction rel=",
                    g268_rel(g268_traction_fv(ssm, qt, 1, tnfun, tfun;
                                              mapped=true),
                             g268_traction_fv(ssb, qt, 1, tnfun, tfun;
                                              mapped=false)))
        end
    end
end
