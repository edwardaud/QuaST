# What are the two wall functionals, exactly?
#
#   pointwise (semquad_nu_walls):  p = (1/h) * (th_0 - 4 th_1 + 3 th_wall)
#   steady CBF (semquad_nu_cbf):   q = (1/(3h)) * (th_0 - 8 th_1)      [th_wall = 0]
#
# for ONE wall element with GLL nodes th_0 (interior edge), th_1 (mid), th_wall=0.
# Both are linear functionals of the three nodal values; q - p =
# -(2/(3h)) (th_0 - 2 th_1), i.e. exactly the *quadratic* part of the profile
# inside the wall element.
#
# This script applies both to exact polynomials on the real E tree (u = 1-y, so
# the top wall is u = 0) and compares with the exact int_wall dtheta/dy dx:
#     theta = u    : exact flux = -Lx        (both should be exact)
#     theta = u^2  : exact flux = 0          (pointwise exact, CBF = -h/3 Lx)
#     theta = u^3  : exact flux = 0          (CBF exact, pointwise = +h^2/2 Lx)
using QuaST, Printf, LinearAlgebra
include(joinpath(@__DIR__, "nu_cbf_compare.jl"))
include(joinpath(@__DIR__, "nusselt_audit.jl"))

ck = ARGS[1]
st = load_semquad_state(ck)
pr, ph0, cfg, attrs = physics_from_checkpoint(ck)
ph = attach_seed_latent(pr, ph0, cfg, attrs)
sq = make_semstokes(st.qt, st.p).su
A = total_area(st.qt)
h = 1.0 / 64.0
top_ref = -semquad_base_gradient(pr, 1.0)
bot_ref = -semquad_base_gradient(pr, 0.0)
@printf("wall leaf level 5: h = 1/64 = %.9f, Lx = %.6f, A = %.6f\n", h, st.qt.Lx, A)
println("theta = f(u), u = 1-y (top wall at u=0); both walls are measured, but the")
println("bottom wall value of u^2/u^3 is not zero, so only its top row is meaningful.")
println()
@printf("%-10s %-22s %-22s %-22s\n", "theta", "pointwise-flux", "steady-CBF-flux", "CBF - pointwise")
res = Dict{String,NTuple{2,Float64}}()
for (name, f) in (("u", uu -> uu), ("u^2", uu -> uu^2), ("u^3", uu -> uu^3))
    theta = zeros(Float64, sq.N)
    for g in 1:sq.N
        theta[g] = f(1.0 - sq.coords[g][2])
    end
    Nut, _ = semquad_nu_walls(sq, st.qt, theta, pr)
    pt = top_ref - Nut
    ntc, _ = semquad_nu_cbf(sq, st.qt, pr, theta, zeros(sq.N), zeros(sq.N);
                            advection=:skew, ev_field=nothing, phases=ph)
    cbf = top_ref - ntc
    res[name] = (pt, cbf)
    @printf("%-10s %+22.12e %+22.12e %+22.12e\n", name, pt, cbf, cbf - pt)
end
println()
println("closed-form values of int_wall dtheta/dy dx (top wall, outward +y):")
@printf("  u    : exact %+22.12e\n", -1.0 * st.qt.Lx)
@printf("  u^2  : exact %+22.12e\n", 0.0)
@printf("  u^3  : exact %+22.12e\n", 0.0)
println()
println("errors against the closed forms:")
@printf("  pointwise: u^2 error %+12.3e   u^3 error %+12.3e   (predicted +h^2/2 Lx = %+12.3e)\n",
        res["u^2"][1] - 0.0, res["u^3"][1] - 0.0, h^2 / 2 * st.qt.Lx)
@printf("  steady CBF: u^2 error %+12.3e (predicted -h/3 Lx = %+12.3e)   u^3 error %+12.3e\n",
        res["u^2"][2] - 0.0, -h / 3 * st.qt.Lx, res["u^3"][2] - 0.0)
