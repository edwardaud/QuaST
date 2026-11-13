"""
    run_production(n=16; tstop=1.5e-6, dt=nothing, tag="", save_dt=1e-8,
                   save_path="", log_path="", verbose=false, kwargs...)

Run the validated production configuration in one call:

- analytic H=5 base initial condition
  ``T0 = 1 - z + 2.5 z (1-z) + 0.05 sin(pi z) sin(pi x)`` evaluated at the GLL nodes;
- fixed time step (no adaptive controller) chosen as ``dt = 5e-10 * 16/n`` when
  `dt` is not given, so the nodal Courant number stays below one for the corrected
  GLL hmin (n16: 5e-10, n32: 2.5e-10, n64: 1.25e-10);
- `ev = 0.11`, `ev_cmax = 0.052`, `maxpic = 1`, `advection = :skew`,
  `velocity_space = :taylor_hood`, `ev_bnd_own = true`, corrected GLL hmin;
- saves every `save_dt` and logs every 20 steps.

This is a convenience wrapper around [`run_simulation`](@ref); the generic defaults
of `run_simulation` are unchanged. Returns the `SimulationResult`.

# Example
```julia
res = run_production(16; tag = "prod_n16")
```
"""
function run_production(n::Int=16; tstop::Float64=1.5e-6,
                        dt::Union{Nothing,Float64}=nothing,
                        tag::String="", save_dt::Float64=1e-8,
                        save_path::String="", log_path::String="",
                        verbose::Bool=false, kwargs...)
    n >= 4 || throw(ArgumentError("n must be >= 4"))
    dtfix = dt === nothing ? 5.0e-10 * (16.0 / n) : dt
    base = isempty(tag) ? "prod_n$(n)" : tag
    sp = isempty(save_path) ? joinpath("out_julia", base * ".h5") : save_path
    lp = isempty(log_path) ? joinpath("out_julia", base * ".log") : log_path

    m = make_mesh(1.0, n, n, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    theta0 = Vector{Float64}(undef, m.nvx * m.nvz)
    for iz in eachindex(zs), ix in eachindex(xs)
        g = (iz - 1) * m.nvx + ix
        theta0[g] = 0.05 * sinpi(zs[iz]) * sinpi(xs[ix])
    end
    pr = Params(Ra=1e7, H=5.0, eta_law=:expT, deta_T=1000.0, bc=:free)
    mkpath(dirname(sp)); mkpath(dirname(lp))
    maxsteps = ceil(Int, 1.2 * tstop / dtfix)
    return run_simulation(m, fd, pr; theta0=theta0, tstop=tstop,
                          dt0=dtfix, dtmax=dtfix, dtmin=1e-15, cfl=1e9,
                          r_min=0.95, r_max=1.05,
                          maxsteps=maxsteps, maxpic=1, pic_tol=1e-5, pic_relax=1.0,
                          steady_tol=1e-9, steady_window=10, tmin=1e-6,
                          report_every=100, verbose=verbose, linear=:schur_cg,
                          advection=:skew, velocity_space=:taylor_hood,
                          ev=0.11, ev_cmax=0.052, ev_bnd_neighbor=false,
                          ev_bnd_own=true, save_dt=save_dt, save_path=sp,
                          log_path=lp, log_every=20, kwargs...)
end


