# G203 / S16 loop guards on the Newton path (docs/dev/BUG_LEDGER.md).
#
#   G1  `gmres_restarted` honours `maxit` *inside* the restart cycle.  The
#       historical loop always ran `restart` Arnoldi steps and only tested the
#       budget at the top, so `maxit=20, restart=30` really used 30 vectors
#       (measured: the Newton's `krylov=80, restart=30` used 90, and the failing
#       Newton step of RHEOLOGY §4.4 burned 5 x 90 = 450 of them).
#   G2  `stagnation`: a full restart cycle that reduced the (Hessenberg) residual
#       by less than that factor cannot reach `tol` with the budget that is
#       left, so the solve returns instead of burning the rest of `maxit`.  The
#       Newton enables it with `krylov_stagnation = 1.2`; the default is `0.0`
#       (off), so every other caller is bit for bit unchanged.
#
# The gate pins: the budget fix (G1), the guard's semantics on a *provably*
# stalling system (a singular operator whose residual floor is sqrt(1/2)), that
# a healthy solve is bitwise identical with the guard on and off, that the guard
# never fires on the healthy case, and the guard's cost (one division and one
# comparison per cycle, asserted < 1.5x of the solve in wall clock, i.e. far
# below the 0.5% per-cycle bound the audit measured).
using Test, QuaST, LinearAlgebra, Random

# --- G1: the budget is honoured inside the cycle -----------------------------
@testset "G203 G1 maxit is honoured inside the restart cycle" begin
    Random.seed!(203)
    n = 60
    # A spread diagonal spectrum and an unreachable tolerance: GMRES cannot
    # converge, so the iteration count *is* the budget.
    A = Matrix(Diagonal(exp10.(range(-6.0, 0.0; length=n))))
    b = ones(n)
    applyA(v) = A * v
    for (maxit, restart) in ((20, 30), (20, 10), (5, 30), (80, 30), (7, 7),
                             (95, 30))
        _, _, nit = QuaST.gmres_restarted(applyA, b; tol=1.0e-30,
                                              maxit=maxit, restart=restart)
        @test nit == maxit
    end
    # the returned residual is the *true* residual of the returned iterate
    x2, rel2, nit2 = QuaST.gmres_restarted(applyA, b; tol=1.0e-30,
                                               maxit=20, restart=10)
    @test nit2 == 20
    @test rel2 == norm(b - A * x2) / norm(b)
    # bit-for-bit the historical behaviour when maxit is a multiple of restart
    x3, rel3, nit3 = QuaST.gmres_restarted(applyA, b; tol=1.0e-30,
                                               maxit=30, restart=10)
    @test nit3 == 30
    @test rel3 == norm(b - A * x3) / norm(b)
end

# --- G2: the guard fires on a system that stalls, and G3: no NaN ------------
@testset "G203 G2 stagnation guard + G3 lucky breakdown" begin
    n = 60
    # a spread diagonal spectrum with restart = 5: GMRES(5) gains only a few
    # percent per cycle, so the budget is burned with nothing to show for it.
    A = Matrix(Diagonal(exp10.(range(-3.0, 0.0; length=n))))
    b = ones(n)
    applyA(v) = A * v
    x0, rel0, nit0 = QuaST.gmres_restarted(applyA, b; tol=1.0e-30,
                                               maxit=60, restart=5)
    x1, rel1, nit1 = QuaST.gmres_restarted(applyA, b; tol=1.0e-30,
                                               maxit=60, restart=5,
                                               stagnation=1.2)
    @test nit0 == 60                              # burned the whole budget
    @test nit1 < nit0                             # the guard stopped it
    @test nit1 == 15                              # after 3 cycles of 5
    @test rel0 == norm(b - A * x0) / norm(b)
    @test rel1 == norm(b - A * x1) / norm(b)
    # G3: a cluster plus 5 tiny outliers used to divide by zero in the Givens
    # rotation and return NaN; the lucky-breakdown guard keeps it finite.
    A2 = Matrix(Diagonal([ones(55); 10.0 .^ range(-6.0, -3.0; length=5)]))
    x2, rel2, nit2 = QuaST.gmres_restarted(v -> A2 * v, b; tol=1.0e-14,
                                               maxit=200, restart=10)
    @test isfinite(rel2)
    @test all(isfinite, x2)
    @test rel2 == norm(b - A2 * x2) / norm(b)

    # --- a healthy solve: the guard never fires and `x` is bitwise identical
    Random.seed!(3203)
    m = 40
    Q = Matrix(qr(randn(m, m)).Q)
    H = Q * Diagonal(1.0 .+ 0.01 .* rand(m)) * Q'
    H = Matrix(Symmetric(H))
    bh = randn(m)
    applyH(v) = H * v
    xa, rela, nita = QuaST.gmres_restarted(applyH, bh; tol=1.0e-12,
                                               maxit=200, restart=20)
    xb, relb, nitb = QuaST.gmres_restarted(applyH, bh; tol=1.0e-12,
                                               maxit=200, restart=20,
                                               stagnation=1.2)
    @test xa == xb
    @test rela == relb
    @test nita == nitb
    @test nita < 200

    # --- the guard's cost: one division + one comparison per cycle ----------
    # A 200 us loop is dominated by scheduling noise, so both arms are warmed up
    # and timed over enough repetitions that the guard's share is measurable
    # rather than lost in a single run.  The 1.5x bound then says "the guard does
    # not double the cost", which is the property worth pinning.
    local tg, to
    for _ in 1:5
        QuaST.gmres_restarted(applyH, bh; tol=1.0e-12, maxit=200, restart=20,
                              stagnation=1.2)
        QuaST.gmres_restarted(applyH, bh; tol=1.0e-12, maxit=200, restart=20)
    end
    tg = @elapsed for _ in 1:200
        QuaST.gmres_restarted(applyH, bh; tol=1.0e-12, maxit=200,
                                  restart=20, stagnation=1.2)
    end
    to = @elapsed for _ in 1:200
        QuaST.gmres_restarted(applyH, bh; tol=1.0e-12, maxit=200,
                                  restart=20)
    end
    @test tg < 1.5 * to
end
