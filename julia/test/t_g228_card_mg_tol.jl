using Test, MantleSEM, LinearAlgebra, Printf

# G228: the realistic card's explicit MG outer tolerance.
#
# `CASE_LINEAR=auto` resolves to matrix-free MG at both production states, so
# the card runs the MG outer solve with whatever `mg_tol` it passes (or, before
# this gate existed, with the library default).  The library default is 1e-10,
# five to seven orders below the driver's `stokes_tol=1e-3` acceptance gate; the
# measured production cost of that excess is 92/108/90 outer iterations and
# 1.68/2.63/1.54 s warm on `F`/`P1`/`C`, against 81/91/67 and 1.44/2.27/1.30 s
# at `mg_tol=1e-8` (docs/dev/CASE_REALISTIC.md §1.6).  The card therefore passes
# `MG_TOL=1e-8` explicitly.
#
# This gate pins three things, so a future change cannot silently undo it:
#   1. the card source still declares `STOKES_TOL=1e-3`, `MG_TOL=1e-8` and
#      forwards them as `stokes_tol=STOKES_TOL, mg_tol=MG_TOL` (removing the
#      explicit `mg_tol` would hand the card back to the library default);
#   2. through the actual `linear=:auto` path on a uniform L4 tree (10 851
#      selector dofs, contrast 1, so the selector picks MG as at production),
#      the card's tolerance leaves the true residual at least four orders below
#      the driver gate, needs fewer outer iterations than the 1e-10 solve, and
#      moves u/p against that tight solve by less than the documented MG
#      accuracy class `|du|/|u| <= 2.1e-6`, gauge-removed
#      `|dp|/|p| <= 8.8e-7` (docs/dev/MG.md);
#   3. the same at production-like contrast (1e4) on a small hanging-node tree,
#      where `mg_tol=1e-6` is measured to exceed the pressure class
#      (1.28e-6 on P1), i.e. the looser benchmark tolerance is not a
#      substitute.

const _G228_CARD = normpath(joinpath(@__DIR__, "..", "scripts",
                                     "case_realistic_2d.jl"))
const _G228_Z2 = (x, y) -> 0.0
const _G228_GATE = 1.0e-3                 # the card's driver acceptance gate
const _G228_CLASS_U = 2.1e-6              # documented MG class (MG_TUNING §0)
const _G228_CLASS_P = 8.8e-7

# Read a `const NAME = <literal Float64>` from the card.  Only a decimal
# literal is accepted: a functional form (`min(...)`) must be reflected here
# deliberately instead of being silently evaluated.
function _g228_const(src::AbstractString, name::AbstractString)::Float64
    pat = Regex("const\\s+" * name *
                "\\s*=\\s*([0-9]+(?:\\.[0-9]*)?[eE][+-]?[0-9]+)")
    m = match(pat, src)
    m === nothing && error("G228: no `const $name = <literal>` in the card; " *
                           "update this gate together with the card")
    return parse(Float64, m.captures[1])
end

# 28-leaf hanging-node tree (the G218/G226 tree), p=2.
function _g228_tree()
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

# Uniform L4, p=4: 10 851 selector dofs, contrast 1.  This is the G191 tree and
# the selector's first MG row; the card's kwargs go through `linear=:auto` here.
function _g228_uniform(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g228_gauge(v)
    return v .- sum(v) / length(v)
end

function _g228_rel(a, b)
    return norm(a - b) / norm(b)
end

@testset "G228 realistic card mg_tol" begin
    # ---- 1. the card declares and forwards the documented tolerance --------
    src = read(_G228_CARD, String)
    stokes_tol = _g228_const(src, "STOKES_TOL")
    mg_tol = _g228_const(src, "MG_TOL")
    @test stokes_tol == _G228_GATE
    @test mg_tol == 1.0e-8
    @test mg_tol <= stokes_tol / 1.0e4        # >= 4 orders below the gate
    @test occursin(r"mg_tol\s*=\s*MG_TOL", src)
    @test occursin(r"stokes_tol\s*=\s*STOKES_TOL", src)
    @info "G228 card constants" stokes_tol = stokes_tol mg_tol = mg_tol

    # ---- 2. the card's kwargs through the real :auto path -----------------
    # Same tolerance pair as the card: auto_fallback_tol/stokes_accept_tol get
    # the driver's gate, mg_tol the card's value.
    function _g228_auto(mgtol)
        qt = _g228_uniform(4)
        p = 4
        ss = make_semstokes(qt, p)
        @test 2 * ss.su.N + ss.sp.N == 10_851
        eta = (x, y) -> 1.0
        fz = (x, y) -> sinpi(x) * sinpi(y)
        empty!(MantleSEM.SEMQUAD_MG_CACHE)
        MantleSEM.SEMQUAD_PRESSURE_CACHE[] = nothing
        MantleSEM.LAST_AUTO_FALLBACK[] = (count=0, last_reason="")
        u, pv = solve_semquad_stokes_var(ss, qt, eta, _G228_Z2, fz, _G228_Z2,
                                         _G228_Z2; bc=:free, deviatoric=false,
                                         mass_form=:boussinesq, linear=:auto,
                                         auto_fallback_tol=stokes_tol,
                                         stokes_accept_tol=stokes_tol,
                                         mg_tol=mgtol)
        return (; u, pv, res=MantleSEM.LAST_SEMSTOKES_RESIDUAL[],
                nit=MantleSEM.LAST_SEMSTOKES_SCHUR_ITERS[],
                choice=MantleSEM.LAST_SEMSTOKES_AUTO_CHOICE[].choice,
                fb=MantleSEM.LAST_AUTO_FALLBACK[].count)
    end
    tight = _g228_auto(1.0e-10)
    card = _g228_auto(mg_tol)
    @test tight.choice === :mg
    @test card.choice === :mg
    @test tight.fb == 0 && card.fb == 0
    @test tight.res <= 1.0e-9
    @test card.res <= stokes_tol / 1.0e4
    @test card.nit < tight.nit                       # no over-solve
    du = _g228_rel(card.u, tight.u)
    dpg = _g228_rel(_g228_gauge(card.pv), _g228_gauge(tight.pv))
    @test du <= _G228_CLASS_U
    @test dpg <= _G228_CLASS_P
    @info "G228 auto uniform L4" it_tight = tight.nit it_card = card.nit res = card.res du = du dpg = dpg

    # ---- 3. production-like contrast on a hanging-node tree ---------------
    # Explicit `:mg` (a 28-leaf tree is below the selector's MG threshold) with
    # the card's tolerance, against the same solve at the library default.  The
    # tree has hanging nodes, so this also keeps the hanging-row operator in
    # the comparison; contrast 1e4 is closer to the card's 1e7 than the
    # uniform row.
    function _g228_hc(mgtol)
        qt = _g228_tree()
        ss = make_semstokes(qt, 2)
        eta = (x, y) -> exp(-log(1.0e4) * (1 - y))
        fz = (x, y) -> sinpi(x) * sinpi(y)
        empty!(MantleSEM.SEMQUAD_MG_CACHE)
        MantleSEM.SEMQUAD_PRESSURE_CACHE[] = nothing
        u, pv = solve_semquad_stokes_var(ss, qt, eta, _G228_Z2, fz, _G228_Z2,
                                         _G228_Z2; bc=:free, deviatoric=false,
                                         mass_form=:boussinesq, linear=:mg,
                                         mg_operator=:matrixfree,
                                         mg_vel_solver=:vcycle, mg_threads=1,
                                         mg_tol=mgtol)
        return (; u, pv, res=MantleSEM.LAST_SEMSTOKES_RESIDUAL[],
                nit=MantleSEM.LAST_SEMSTOKES_SCHUR_ITERS[])
    end
    tight = _g228_hc(1.0e-10)
    card = _g228_hc(mg_tol)
    @test tight.res <= 1.0e-9
    @test card.res <= stokes_tol / 1.0e4
    @test card.nit < tight.nit
    du = _g228_rel(card.u, tight.u)
    dpg = _g228_rel(_g228_gauge(card.pv), _g228_gauge(tight.pv))
    @test du <= _G228_CLASS_U
    @test dpg <= _G228_CLASS_P
    @info "G228 hanging-node contrast 1e4" it_tight = tight.nit it_card = card.nit res = card.res du = du dpg = dpg
end
