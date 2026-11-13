using QuaST, Printf, SHA

# Produce the 25-frame blob baseline of docs/dev/VALIDATION.md §2.3.
#
# Promoted 2026-11-13 from the untracked out_julia/agent/run_blob_kelly_1024.jl
# (docs/dev/AMR.md): a clean checkout could not regenerate
# the 25-frame digest.  The protocol is unchanged -- same Params, tree seed,
# fixed dt, Kelly marking every 5 steps, p=4, direct solve and 256^2 frames --
# and the only additions are the per-frame SHA-256 rows and the policy pin.
#
# The classic-era baseline (final theta digest 868a79d9...458afe88, frame-25
# Tmax 1.264427445) was produced under `amr_policy=:classic`; after the
# 2026-11-13 default flip the balanced path is the default baseline.  Both
# remain reproducible:
#
#   BLOB_POLICY=classic BLOB_OUT=out_julia/fresh_blob_kelly_1024_classic \
#       julia --project=. examples/run_blob_kelly_1024.jl
#   julia --project=. examples/run_blob_kelly_1024.jl   # balanced
#
# `FINAL theta_sha256=...` and `FINAL Tmax25=...` are printed for the digest
# quoted in docs/dev/VALIDATION.md; frame_sha256.csv holds all 25 per-frame digests.

const PR = Params(Ra=1.0e8, H=5.0, eta_law=:expTz,
                  eta_b=log(16384.0), eta_c=log(64.0), bc=:free)
const NG = 256
const TOUT = 4.0e-10
const DT = 8.0e-11
const MAXLEAVES = 1024
const POLICY = Symbol(get(ENV, "BLOB_POLICY", "balanced"))
const DIR = normpath(get(ENV, "BLOB_OUT",
    joinpath(@__DIR__, "..", "out_julia",
             "fresh_blob_kelly_1024_" * string(POLICY))))

_hash(v) = bytes2hex(sha256(collect(reinterpret(UInt8, v))))

function main()
    POLICY in (:classic, :balanced) ||
        error("BLOB_POLICY must be classic or balanced, got $POLICY")
    if isdir(DIR)
        # Only ever clear a directory that holds nothing but this producer's own
        # output: an unset BLOB_OUT used to be able to point outside the tree.
        ours = Set(["frame_meta.csv", "frame_sha256.csv", "run_summary.csv"])
        extra = setdiff(Set(readdir(DIR)), ours)
        isempty(extra) ||
            error("refusing to overwrite $(DIR): it holds files this script did not write: " *
                  join(sort(collect(extra)), ", "))
        rm(DIR; recursive=true, force=true)
    end
    mkdir(DIR)
    open(joinpath(DIR, "frame_meta.csv"), "w") do io
        println(io, "frame,t,N,leaves,Tmax,nlev")
    end
    open(joinpath(DIR, "frame_sha256.csv"), "w") do io
        println(io, "frame,t,theta_sha256,u_sha256,leaves,Tmax")
    end
    qt = Quadtree(6)
    for _ in 1:4, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    p = 4
    ss = make_semstokes(qt, p)
    theta0 = [0.12 * exp(-((x - 0.5) / 0.12)^2 - ((z - 0.35) / 0.10)^2) * sinpi(z)
              for (x, z) in ss.su.coords]
    xs = range(0.0, 1.0; length=NG); zs = range(0.0, 1.0; length=NG)
    knot = Ref(0); nextt = Ref(TOUT); last_tmax = Ref(NaN); final_sha = Ref("")
    cb = function (st)
        while st.t >= nextt[] - 1.0e-18
            knot[] += 1
            leaves = Set(st.sq.leaves)
            T = Array{Float64}(undef, NG, NG)
            for j in 1:NG, i in 1:NG
                T[j, i] = Tbar(zs[j], PR.H) +
                          QuaST._semquad_eval_at(st.sq, st.theta, xs[i], zs[j], leaves)
            end
            open(joinpath(DIR, "frame_$(knot[])_T.bin"), "w") do io; write(io, T); end
            open(joinpath(DIR, "frame_$(knot[])_mesh.csv"), "w") do io
                for lf in sort(collect(st.sq.leaves))
                    h = 2.0^(-lf[1])
                    @printf(io, "%d,%.17g,%.17g,%.17g\n", lf[1], lf[2] * h, lf[3] * h, h)
                end
            end
            last_tmax[] = maximum(T)
            final_sha[] = _hash(st.theta)
            open(joinpath(DIR, "frame_meta.csv"), "a") do io
                @printf(io, "%d,%.17g,%d,%d,%.10g,%d\n", knot[], st.t, st.sq.N,
                        length(st.sq.leaves), last_tmax[],
                        length(unique(lf[1] for lf in st.sq.leaves)))
            end
            open(joinpath(DIR, "frame_sha256.csv"), "a") do io
                @printf(io, "%d,%.17g,%s,%s,%d,%.10g\n", knot[], st.t, final_sha[],
                        _hash(vcat(st.ux, st.uz)), length(st.sq.leaves), last_tmax[])
            end
            nextt[] = (knot[] + 1) * TOUT
        end
    end
    t0 = time()
    res = run_semquad_simulation(qt, PR, theta0; p=p, tstop=1.0e-8, dt0=DT, dtmax=DT, dtmin=1.0e-16,
                                 maxsteps=200, maxpic=1,
                                 amr_every=5, refine_frac=0.2, coarsen_frac=0.005,
                                 coarsen_theta_tol=1.0e-3, amr_policy=POLICY,
                                 marking=:error_fraction, symmetry=:x,
                                 max_leaves=MAXLEAVES, max_level=6,
                                 ev=0.11, ev_cmax=0.052, advection=:skew, linear=:direct,
                                 stokes_tol=1.0e-6, callback=cb, verbose=false)
    @printf("done t=%.3e steps=%d leaves=%d N=%d frames=%d in %.1f s\n",
            res.t, res.steps, length(res.qt.leaves), res.ss.su.N, knot[], time() - t0)
    # The digest quoted in docs/dev/VALIDATION.md is the *end-of-run* field (step
    # 126), which is what bench_blob_production_ab.jl printed as "theta sha";
    # frame 25's field is one step earlier and is in frame_sha256.csv.
    end_th = _hash(res.theta)
    end_u = _hash(vcat(res.ux, res.uz))
    open(joinpath(DIR, "run_summary.csv"), "w") do io
        println(io, "policy,frames,steps,end_theta_sha256,end_u_sha256,Tmax25,wall_s")
        @printf(io, "%s,%d,%d,%s,%s,%.10g,%.1f\n", POLICY, knot[], res.steps,
                end_th, end_u, last_tmax[], time() - t0)
    end
    @printf("FINAL policy=%s frames=%d end_theta_sha256=%s end_u_sha256=%s Tmax25=%.10g\n",
            POLICY, knot[], end_th, end_u, last_tmax[])
end
main()
