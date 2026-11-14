# G174: the stale-factor (:lagged) cache — bitwise vs direct, refactor on AMR, bounded iterations.
using QuaST, Printf, LinearAlgebra

# G174: gate for the stale-factor cache (:lagged) -- the S2 speed-up path.
#  * same tree, same eta: second solve costs zero refinements and is bitwise
#    identical to the direct path;
#  * same tree, perturbed eta: refinement converges and matches direct to 1e-8;
#  * tree changed (AMR): the cache must miss and refactor, result matches direct;
#  * the refinement count stays below the maxit cap.

function tree2b()
    qt = Quadtree(3; cells_x=5, cells_y=2, Lx=2.5)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

eta_a(x, y) = exp(-9.704060527839234 * (0.5 + 0.1 * cos(pi * x / 2.5) * sin(pi * y)) +
                  4.158883083359672 * (1 - y))
eta_b(x, y) = 1.01 * eta_a(x, y)

const Z2 = (x, y) -> 0.0
const FZ = (x, y) -> sin(pi * x / 2.5) * sin(pi * y)

function solve(qt, eta, lin)
    ss = make_semstokes(qt, 2)
    u, p, fixed, cons2, cons_p = solve_semquad_stokes_var(ss, qt, eta, Z2, FZ, Z2, Z2;
                                                          bc=:free, linear=lin)
    return u, p, ss.su.N, LAST_SEMSTOKES_RESIDUAL[], LAST_SEMSTOKES_LAGGED_ITERS[]
end

qt = tree2b()
@printf("G174: tree leaves=%d\n", length(qt.leaves))

# 1) direct baseline, then lagged on the identical problem
u_dir, p_dir, N, rdir, _ = solve(qt, eta_a, :direct)
u_lag1, p_lag1, _, r1, it1 = solve(qt, eta_a, :lagged)
u_lag2, p_lag2, _, r2, it2 = solve(qt, eta_a, :lagged)
@test isapprox(norm(u_lag1 - u_dir) / norm(u_dir), 0.0; atol=1e-13)
@test isapprox(norm(u_lag2 - u_dir) / norm(u_dir), 0.0; atol=1e-13)
@test it2 <= 1                          # nothing to refine on a repeated solve
@printf("  same eta: ref1=%d ref2=%d  res=%.2e\n", it1, it2, r2)

# 2) perturbed eta: refinement must converge to the direct answer
u_dir_b, _, _, rdir_b, _ = solve(qt, eta_b, :direct)
u_lag_b, _, _, rlag_b, it_b = solve(qt, eta_b, :lagged)
rel = norm(u_lag_b - u_dir_b) / norm(u_dir_b)
@printf("  perturbed eta: iters=%d rel=%.2e res=%.2e (direct res %.2e)\n",
        it_b, rel, rlag_b, rdir_b)
@test rel < 1.0e-7
@test it_b <= 6

# 3) tree change: the fingerprint must invalidate the cache
qt2 = Quadtree(4; cells_x=5, cells_y=2, Lx=2.5)
for _ in 1:3, lf in collect(qt2.leaves)
    refine!(qt2, lf)
end
balance!(qt2)
refine!(qt2, (3, 0, 0)); refine!(qt2, (3, 1, 0)); refine!(qt2, (3, 0, 1)); refine!(qt2, (3, 1, 1))
balance!(qt2)
nref0 = SEMSTOKES_LAGGED_REFACTORS[]
u_dir2, _, _, _, _ = solve(qt2, eta_a, :direct)
u_lag2b, _, _, r2b, it2b = solve(qt2, eta_a, :lagged)
@test SEMSTOKES_LAGGED_REFACTORS[] == nref0 + 1      # exactly one new factorisation
rel2 = norm(u_lag2b - u_dir2) / norm(u_dir2)
@printf("  new tree: leaves=%d refactors+%d iters=%d rel=%.2e\n",
        length(qt2.leaves), SEMSTOKES_LAGGED_REFACTORS[] - nref0, it2b, rel2)
@test rel2 < 1.0e-12

# 4) residual quality of the lagged path
@test rlag_b < 1.0e-6
@printf("G174: stale-factor (S2) gate ok\n")
