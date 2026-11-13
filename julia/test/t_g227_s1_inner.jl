#!/usr/bin/env julia
# smoke test: the pressure block's inner-solve switch inside the production
# apply path, all three kinds against the exact factor.
#
#   julia --project=julia --threads=8 julia/test/t_g227_s1_inner.jl
#
# Builds the w-BFBT pressure block on a small hanging-node tree with each
# `bfbt_s1_solve` kind, applies it to the same right-hand side and checks that
#   * `:chol`   reproduces the shipped apply bitwise,
#   * `:level`  matches the exact solve to round-off and is independent of the
#     thread count,
#   * `:cg_*`   matches the exact solve to their tolerance.
# It also pins that the pressure cache is keyed on the inner solve (switching
# the kind rebuilds instead of silently reusing the other factors).
using MantleSEM, LinearAlgebra, SparseArrays, Test
const MS = MantleSEM

BLAS.set_num_threads(1)

# a small hanging-node tree (the G218/G226 tree): 2x2 base cells, one uniform
# refinement, the lower-left 2x2 block refined once more, 2:1-balanced
function _g227_tree()
    qt = Quadtree(3; cells_x=2, cells_y=2)
    for lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    for lf in collect(qt.leaves)
        lf[1] == 1 && lf[2] < 2 && lf[3] < 2 && refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

qt = _g227_tree()
p = 4
ss = make_semstokes(qt, p)
Z2 = (x, y) -> 0.0
eta_fun = (x, y) -> 1.0 + 4.0 * (x + y)
fz = (x, y) -> 1.0
sys = solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2;
                               bc=:free, deviatoric=false,
                               mass_form=:boussinesq, linear=:mg,
                               mg_operator=:matrixfree, assemble_only=:mg)
key = MS._saddle_key(qt, p, :free, size(sys.Bs, 2), size(sys.Bs, 1))
nthreads = max(Threads.nthreads(), 2)

# the matrix-free velocity apply of the production path (the `S2` middle factor
# is applied matrix-free, so the block needs a callable `K`)
mfop = MS.semquad_kop(ss, qt; deviatoric=false)
mfetav = MS.semquad_eta_lanes(mfop, MS.semquad_eta_field(mfop, eta_fun))
mfwork = MS.SemQuadKWork(mfop.n, mfop.nq, 2 * ss.su.N)
mfapplyK! = (out, v) -> MS._semquad_apply_Kred!(out, mfop, sys.Cu, mfetav, v,
                                                 mfwork)

function block(kind; tol=1.0e-10, nthr=nthreads)
    MS.SEMQUAD_PRESSURE_CACHE[] = nothing
    return MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt,
                                          nothing, eta_fun;
                                          mode=:bfbt, scale=0.0,
                                          weight=:inv_eta, key=key, cache=false,
                                          bfbt_d=:exact, bfbt_s1_solve=kind,
                                          bfbt_s1_tol=tol, bfbt_s1_maxit=400,
                                          nthreads=nthr, applyK=mfapplyK!)
end

nf = size(sys.Bs, 2)
nfp = size(sys.Bs, 1)
rng = 12345
rp = zeros(Float64, nfp)
for i in eachindex(rp)
    rp[i] = sin(0.7 * i)          # deterministic, zero-free right-hand side
end

function apply_block(Fp, pscale, rp)
    wP = zeros(Float64, nf + nfp)
    rp2 = zeros(Float64, nfp)
    s1 = zeros(Float64, nf)
    s2 = zeros(Float64, nf)
    MS._semquad_pressure_block!(wP, copy(rp), rp2, s1, s2, Fp, pscale, nf)
    return wP[(nf + 1):end]
end

Fp_c, ps_c = block(:chol)
ref = apply_block(Fp_c, ps_c, rp)

@testset "G227 S1 inner solve" begin
    # :chol must reproduce the shipped form bitwise
    Fp_c2, ps_c2 = block(:chol)
    @test apply_block(Fp_c2, ps_c2, rp) == ref

    # :level: same factor, level-scheduled substitutions
    Fp_l, ps_l = block(:level)
    xl = apply_block(Fp_l, ps_l, rp)
    @test isapprox(xl, ref; rtol=1.0e-12, norm=norm)
    xl_again = apply_block(Fp_l, ps_l, rp)
    @test xl_again == xl                     # repeatable
    @test Fp_l.S1solve isa MS.SemQuadS1Level
    @test MS._semquad_s1_iters(Fp_l.S1solve) > 0
    # bitwise independent of the thread count: the same block built with a
    # one-worker region and with `nthreads` workers gives the same bits
    Fp_l1, ps_l1 = block(:level; nthr=1)
    @test Fp_l1.S1solve.region === nothing
    @test apply_block(Fp_l1, ps_l1, rp) == xl
    if nthreads > 1
        @test Fp_l.S1solve.region !== nothing
    end

    # :cg_ichol to a tight tolerance
    Fp_i, ps_i = block(:cg_ichol; tol=1.0e-12)
    xi = apply_block(Fp_i, ps_i, rp)
    @test isapprox(xi, ref; rtol=1.0e-8, norm=norm)
    @test MS._semquad_s1_iters(Fp_i.S1solve) > 0

    # :cg_diag looser (the diagonal preconditioner is weak by construction)
    Fp_d, ps_d = block(:cg_diag; tol=1.0e-12)
    xd = apply_block(Fp_d, ps_d, rp)
    @test isapprox(xd, ref; rtol=1.0e-5, norm=norm)

    # the cache key carries the inner solve
    MS.SEMQUAD_PRESSURE_CACHE[] = nothing
    nb0 = MS.SEMSTOKES_PRESSURE_BUILDS[]
    MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt, nothing,
                                   eta_fun; mode=:bfbt, weight=:inv_eta,
                                   key=key, cache=true, bfbt_d=:exact,
                                   bfbt_s1_solve=:chol, nthreads=nthreads, applyK=mfapplyK!)
    MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt, nothing,
                                   eta_fun; mode=:bfbt, weight=:inv_eta,
                                   key=key, cache=true, bfbt_d=:exact,
                                   bfbt_s1_solve=:cg_ichol, bfbt_s1_tol=1.0e-8,
                                   nthreads=nthreads, applyK=mfapplyK!)
    @test MS.SEMSTOKES_PRESSURE_BUILDS[] - nb0 == 2   # a rebuild, not a reuse
    @test MS.LAST_SEMSTOKES_PRESSURE_REUSED[] == false
    # ... and the same kind with the same tol *does* reuse
    MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt, nothing,
                                   eta_fun; mode=:bfbt, weight=:inv_eta,
                                   key=key, cache=true, bfbt_d=:exact,
                                   bfbt_s1_solve=:cg_ichol, bfbt_s1_tol=1.0e-8,
                                   nthreads=nthreads, applyK=mfapplyK!)
    @test MS.SEMSTOKES_PRESSURE_BUILDS[] - nb0 == 2
    @test MS.LAST_SEMSTOKES_PRESSURE_REUSED[] == true
end
