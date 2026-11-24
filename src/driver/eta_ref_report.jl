# The Rayleigh-number reference viscosity report.
#
# `Params.eta_ref` is the viscosity `Ra` is defined with
# (see `src/physics/params.jl`), so the flow solve uses `eta_law/eta_ref`.  This
# file measures one viscosity closure and states the convention: the normalised
# field range, the interior-band mean that defines `Ra_interior`, and -- when the
# card supplies its dimensional scale set -- the physical reference viscosity in
# Pa s.  Diagnostic only: no arithmetic result of the solve reads it, and a run
# at the default `eta_ref = 1` prints nothing.

# ---------------------------------------------------------------------------
# G257: the eta_ref convention.  `Params.eta_ref` is the viscosity the Rayleigh
# number is defined with, in the units the material law returns: the driver
# solves with `eta_law/eta_ref`, so a law rescaled by a constant `c` is the same
# physical problem only when `eta_ref` is rescaled by `c` as well.  The helpers
# below only *measure* one viscosity closure and build the run-start/end report;
# no arithmetic result of the solve reads them.
# ---------------------------------------------------------------------------

"""
    _eta_interior_band_mean(sq, qt, eta_fun; zlo=0.25, zhi=0.75)

Arithmetic volume mean of the normalised viscosity closure `eta_fun` over the
interior band `zlo <= z <= zhi` -- by default the middle half of the layer,
which excludes both thermal boundary layers.  Gauss quadrature on the quadtree
(the rule the volume diagnostics use), values outside the band skipped and the
mean renormalised by the band volume.  `eta_fun` must be the closure the flow
solve used, i.e. already divided by `Params.eta_ref`.
"""
function _eta_interior_band_mean(sq::SemQuad, qt::Quadtree, eta_fun::Function;
                                 zlo::Float64=0.25, zhi::Float64=0.75)
    g = sq.geom
    g isa CartesianBox ||
        return _eta_interior_band_mean_mapped(g, sq, qt, eta_fun; zlo=zlo,
                                              zhi=zhi)
    p = sq.p
    fd = make_fedata(p, p - 1)
    num = 0.0; den = 0.0
    for lf in qt.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx
        y0 = lf[3] * hy
        for qy in 1:fd.nq, qx in 1:fd.nq
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            (zlo <= y <= zhi) || continue
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            num += w * eta_fun(x, y)
            den += w
        end
    end
    return num / den
end

"""
    _eta_interior_band_mean_mapped(g, sq, qt, eta_fun; zlo=0.25, zhi=0.75)

Mapped method of [`_eta_interior_band_mean`](@ref) (stage 0 cut 4): the interior
band is the physical **height** band `zlo <= height(g,x,y) <= zhi` — the
depth-coordinate generalisation of "the middle of the layer" — and the measure
is the mapped `quad_measure` (GEOM_DESIGN §3 cut 4: "interior band in
`height`").  The box body above is unchanged.
"""
function _eta_interior_band_mean_mapped(g::AbstractGeometry, sq::SemQuad,
                                        qt::Quadtree, eta_fun::Function;
                                        zlo::Float64=0.25, zhi::Float64=0.75)
    p = sq.p
    fd = make_fedata(p, p - 1)
    num = 0.0; den = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            h = height(g, x, y)
            (zlo <= h <= zhi) || continue
            w = quad_measure(m, fd, qx, qy)
            num += w * eta_fun(x, y)
            den += w
        end
    end
    return num / den
end

"""
    _eta_ref_field_stats(sq, qt, eta_fun, pr; zlo=0.25, zhi=0.75)

The eta_ref-convention measurements of one viscosity closure: `min`/`max`/`nclip`
of the nodal field (`semquad_eta_range`, in `eta_ref` units), the field contrast,
the interior-band volume mean of [`_eta_interior_band_mean`](@ref) and the band
bounds.  Diagnostic only: nothing here writes solver state.
"""
function _eta_ref_field_stats(sq::SemQuad, qt::Quadtree, eta_fun::Function,
                              pr::Params; zlo::Float64=0.25, zhi::Float64=0.75)
    r = semquad_eta_range(sq, eta_fun, pr)
    interior = _eta_interior_band_mean(sq, qt, eta_fun; zlo=zlo, zhi=zhi)
    return (min=r.min, max=r.max, nclip=r.nclip, n=r.n,
            contrast=r.max / r.min, interior=interior, zlo=zlo, zhi=zhi)
end

"""
    _eta_ref_message(pr, st, step, t, final, scale) -> String

The G257 run report: what Rayleigh number the run solves, what reference
viscosity it is defined with, the realised normalised field range, the interior
Rayleigh number and -- when the card's dimensional scale set `scale` is
available -- the physical reference viscosity in Pa s.  A pure string builder
(every `@sprintf` format is a literal), so the gate can pin its wording and its
numbers.  `scale === nothing` is the generic (dimensionless) run.
"""
function _eta_ref_message(pr::Params, st::NamedTuple, step::Int, t::Real,
                          final::Bool, scale)
    head = final ? "end of run" : "first flow solve"
    body = string("Rayleigh-number reference viscosity (eta_ref convention; ",
                  head, ", step ", step, ", t=", @sprintf("%.6e", t), "): ",
                  "Ra = ", @sprintf("%.6e", pr.Ra),
                  " is defined with eta_ref = ", @sprintf("%.6e", pr.eta_ref),
                  " in the units the material law returns, i.e. ",
                  "Ra = rho g alpha dT d^3/(kappa eta_ref); the flow solve ",
                  "used eta/eta_ref wherever the law's viscosity enters.  ",
                  "The normalised viscosity field of this state is eta/eta_ref ",
                  "in [", @sprintf("%.6e", st.min), ", ",
                  @sprintf("%.6e", st.max), "], contrast ",
                  @sprintf("%.6e", st.contrast), ".  eta_interior = ",
                  @sprintf("%.6e", st.interior),
                  " is the arithmetic volume mean of eta/eta_ref over the ",
                  "interior band ", @sprintf("%.3g", st.zlo), " <= z <= ",
                  @sprintf("%.3g", st.zhi),
                  " (the middle half of the layer, which excludes both wall ",
                  "boundary layers), so Ra_interior = Ra/eta_interior = ",
                  @sprintf("%.6e", pr.Ra / st.interior), ".  ")
    if scale !== nothing
        eta0 = scale.rho * scale.g * scale.alpha * scale.dT * scale.d^3 /
               (pr.Ra * scale.kappa)
        if isfinite(eta0) && eta0 > 0.0
            unit = eta0 / pr.eta_ref
            body = string(body,
                          "With this card's scale set eta_0 = rho g alpha dT d^3/",
                          "(Ra kappa) = ", @sprintf("%.6e", eta0),
                          " Pa s is the physical viscosity Ra is defined with; ",
                          "the law returns it at eta = eta_ref = ",
                          @sprintf("%.6e", pr.eta_ref),
                          " (one material-law unit = ", @sprintf("%.6e", unit),
                          " Pa s, and the interior viscosity is ",
                          @sprintf("%.6e", st.interior * unit), " Pa s).  ")
        end
    end
    return string(body,
                  "Quote Ra together with eta_ref: rescaling the material law ",
                  "must be paired with the same rescaling of eta_ref, " *
                  "otherwise Ra keeps its value while the physical Rayleigh ",
                  "number moves.")
end
