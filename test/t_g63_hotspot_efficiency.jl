# G63: local-refinement efficiency on a thin Gaussian hotspot.
#
# Pure advection-diffusion: u=(1,0), H=0, so the base term is zero.  A thin
# sigma=0.02 Gaussian is advected/diffused for 10 steps.  The fixed AMR tree
# (level 3 base, central refinement to level 5) is compared with uniform level 4
# and a level-6 reference.  This is the efficiency gate that the smooth low-Ra
# H=5 flow could not provide.
using Test
using LinearAlgebra
using Printf
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

function h63_blob(sq, sigma=0.02)
    return [exp(-((sq.coords[i][1] - 0.5)^2 + (sq.coords[i][2] - 0.5)^2) / sigma^2)
            for i in 1:sq.N]
end

function h63_run(qt, p, nsteps, dt)
    sq = make_semquad(qt, p)
    theta = h63_blob(sq)
    theta_prev = copy(theta)
    dt_prev = dt
    first = true
    ux = fill(1.0, sq.N)
    uz = zeros(sq.N)
    pr = Params(Ra=1.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free)
    for _ in 1:nsteps
        th = solve_semquad_energy(sq, qt, pr, theta, theta_prev, ux, uz,
                                  dt, dt_prev, first; advection=:skew)
        theta_prev = theta
        theta = th
        dt_prev = dt
        first = false
    end
    return sq, qt, theta
end

function h63_run_dynamic(qt, p, nsteps, dt; pre=2, pre_frac=0.35,
                         dyn_frac=0.05, cap=160)
    sq = make_semquad(qt, p)
    theta = h63_blob(sq)
    for _ in 1:pre
        eta = semquad_gradient_jump(sq, qt, theta)
        if length(qt.leaves) < cap
            amr_cycle_balanced!(qt, eta; geom=CartesianBox(qt), refine_frac=pre_frac, coarsen_frac=0.0)
            sq2 = make_semquad(qt, p)
            theta = h63_blob(sq2)
            sq = sq2
        end
    end
    theta_prev = copy(theta)
    dt_prev = dt
    first = true
    ux = fill(1.0, sq.N)
    uz = zeros(sq.N)
    pr = Params(Ra=1.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free)
    for step in 1:nsteps
        th = solve_semquad_energy(sq, qt, pr, theta, theta_prev, ux, uz,
                                  dt, dt_prev, first; advection=:skew)
        theta_prev = theta
        theta = th
        dt_prev = dt
        first = false
        if step % 2 == 0 && step < nsteps
            eta = semquad_gradient_jump(sq, qt, theta)
            if length(qt.leaves) < cap
                amr_cycle_balanced!(qt, eta; geom=CartesianBox(qt), refine_frac=dyn_frac,
                                    coarsen_frac=0.1)
            else
                amr_cycle_balanced!(qt, eta; geom=CartesianBox(qt), refine_frac=0.0, coarsen_frac=0.2)
            end
            sq2 = make_semquad(qt, p)
            theta = transfer_semquad(sq, theta, sq2)
            theta_prev = transfer_semquad(sq, theta_prev, sq2)
            sq = sq2
            ux = fill(1.0, sq.N)
            uz = zeros(sq.N)
        end
    end
    return sq, qt, theta
end

@testset "G63 local hotspot efficiency" begin
    p = 4
    nsteps = 10
    dt = 1.0e-3

    qt = uniform_tree(3)
    for _ in 1:3
        for lf in collect(qt.leaves)
            h = 2.0^(-lf[1])
            xc = (lf[2] + 0.5) * h
            yc = (lf[3] + 0.5) * h
            if abs(xc - 0.5) < 0.10 && abs(yc - 0.5) < 0.10 && lf[1] < 6
                refine!(qt, lf)
            end
        end
    end
    balance!(qt)
    sa, _, tha = h63_run(qt, p, nsteps, dt)
    ss, _, ths = h63_run(uniform_tree(4), p, nsteps, dt)
    sf, _, thf = h63_run(uniform_tree(6), p, nsteps, dt)

    es = norm(transfer_semquad(ss, ths, sf) - thf) / norm(thf)
    ea = norm(transfer_semquad(sa, tha, sf) - thf) / norm(thf)
    @printf("G63 static N=%d err=%.6e | AMR N=%d err=%.6e\n",
            ss.N, es, sa.N, ea)

    @test sa.N < ss.N
    @test ea < es / 10.0
    @test max_face_level_jump(qt) <= 1

    qtd = uniform_tree(3)
    # The balanced cycle's refine_frac is a cumulative-error share, not a cell
    # fraction, so the dynamic arm needs a wider cut than the classic 0.05/0.35
    # to resolve the same hotspot within the 160-leaf cap.
    sd, qtd2, thd = h63_run_dynamic(qtd, p, nsteps, dt;
                                    pre=2, pre_frac=0.9, dyn_frac=0.5,
                                    cap=160)
    ed = norm(transfer_semquad(sd, thd, sf) - thf) / norm(thf)
    @printf("G63 dynamic N=%d err=%.6e | static N=%d err=%.6e\n",
            sd.N, ed, ss.N, es)
    @test sd.N < ss.N
    @test ed < es / 10.0
    @test max_face_level_jump(qtd2) <= 1
end
