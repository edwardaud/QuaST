using Test, QuaST

# G162: Drucker-Prager pressure derivative of the viscosity, deta/dp.
#
# The plastic branch is eta_pl = (c cos(phi) + p sin(phi)) / (2 edot), so
# deta/dp = sin(phi)/(2 edot) when that branch is active and neither capped by
# tau_max nor clipped to [eta_min, eta_max].  This is the term that makes the
# Drucker-Prager Newton Jacobian non-symmetric (velocity rows coupling to the
# pressure column); the derivative itself is verified here against central
# differences of the production viscosity.

@testset "G162 Drucker-Prager deta/dp" begin
    pr = Params(Ra=1.0, H=0.0, eta_law=:const, eta_const=1.0,
                yield_c=1.0, yield_phi=0.5, yield_taumax=Inf,
                eta_min=1.0e-4, eta_max=1.0e4)
    T = 0.5
    y = 0.5
    # plastic branch active (fast strain rate makes eta_pl < eta_visc), no
    # cap, no clip
    ed = 1.0
    p = 1.0
    h = 1.0e-6
    ep = semquad_visco_plastic_eta(T, y, ed, p + h, pr)
    em = semquad_visco_plastic_eta(T, y, ed, p - h, pr)
    fd = (ep - em) / (2h)
    an = semquad_deta_dp(T, y, ed, p, pr)
    @test abs(an - fd) <= 1.0e-6 * max(abs(fd), 1.0e-12)
    @test an > 0.0

    # viscous branch: very slow strain rate -> eta_pl > eta_visc -> derivative 0
    @test semquad_deta_dp(T, y, 1.0e-8, p, pr) == 0.0

    # tau_max cap
    pr_cap = Params(Ra=1.0, H=0.0, eta_law=:const, eta_const=1.0,
                    yield_c=1.0, yield_phi=0.5, yield_taumax=1.0,
                    eta_min=1.0e-4, eta_max=1.0e4)
    epc = semquad_visco_plastic_eta(T, y, ed, p + h, pr_cap)
    emc = semquad_visco_plastic_eta(T, y, ed, p - h, pr_cap)
    @test abs((epc - emc) / (2h)) < 1.0e-12
    @test semquad_deta_dp(T, y, ed, p, pr_cap) == 0.0

    # phi = 0: no pressure dependence at all
    pr0 = Params(Ra=1.0, H=0.0, eta_law=:const, eta_const=1.0,
                 yield_c=1.0, yield_phi=0.0, yield_taumax=Inf,
                 eta_min=1.0e-4, eta_max=1.0e4)
    @test semquad_deta_dp(T, y, ed, p, pr0) == 0.0
    @test semquad_visco_plastic_eta(T, y, ed, p + h, pr0) ==
          semquad_visco_plastic_eta(T, y, ed, p - h, pr0)
end
