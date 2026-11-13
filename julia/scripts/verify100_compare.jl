#!/usr/bin/env julia
# verify100_compare.jl -- end-state comparison of the direct and MG arms, and
# (optionally) of the direct arm against the production G run's early frames.
#
#   julia --project=julia julia/scripts/verify100_compare.jl \
#       out_julia/verify100_direct out_julia/verify100_mg [out_julia/case_realistic_G]
#
# Every arm is the same card from the same t=0 initial condition; only
# CASE_LINEAR differs, so the two velocity/temperature fields should agree to
# the linear solver tolerance at every step until the AMR error indicators
# (which are discrete) pick different leaves.  The script reports the end-state
# diagnostics, the field difference on the common 256x256 frame grid, and the
# step at which the fields start to separate.
using Printf, Statistics
using MantleSEM

const NG = 256
const LX = 2.5
const HHEAT = 10.0
Tbar(y) = (1.0 - y) + 0.5 * HHEAT * y * (1.0 - y)
const YS = collect(range(0.0, 1.0; length=NG))
const TBAR2 = Tbar.(reshape(YS, 1, NG))          # [x-index 1, y-index 2]
rms(v) = sqrt(mean(abs2, v))

function read_table(path::AbstractString)
    lines = readlines(path)
    header = split(lines[1], ',')
    rows = Vector{Vector{Float64}}()
    for l in lines[2:end]
        isempty(strip(l)) && continue
        push!(rows, parse.(Float64, split(l, ',')))
    end
    return header, rows
end

function col(header, rows, name::AbstractString)
    j = findfirst(==(String(name)), header)
    j === nothing && error("no column $name in $header")
    return [r[j] for r in rows]
end

read_T(path) = reshape(collect(reinterpret(Float64, read(path))), NG, NG)

"""max|dT|, rms|dT| and rms|dT| / rms(theta) on the frame grid (theta = T - Tbar)."""
function field_diff(a, b)
    d = a .- b
    th = a .- TBAR2
    return (maxabs=maximum(abs, d), rms=rms(d),
            rel_anom=rms(d) / rms(th),
            span=maximum(abs, d) / (maximum(a) - minimum(a)))
end

function wall_seconds(log::AbstractString)
    isfile(log) || return NaN
    m = match(r"wall=([0-9.]+) s", read(log, String))
    return m === nothing ? NaN : parse(Float64, m.captures[1])
end

"""Arm summary from history.csv, frame_meta.csv and the log.

The end-state diagnostics are taken from the **last frame** (`frame_meta.csv`),
because that is the state the video shows; on an AMR step the history row is
recorded before the AMR pass while the final frame is written after it, so
`Vrms` can differ in the third digit there (the temperature diagnostics do
not).  The history row is kept as `hist` for the log comparison."""
function arm_summary(dir::AbstractString)
    hdr, rows = read_table(joinpath(dir, "history.csv"))
    fhdr, frows = read_table(joinpath(dir, "frames", "frame_meta.csv"))
    last = rows[end]
    getc(name) = last[findfirst(==(String(name)), hdr)]
    f = frows[end]
    frames = count(fn -> occursin(r"frame_\d+_T\.bin$", fn),
                   readdir(joinpath(dir, "frames")))
    return (dir=dir, wall=wall_seconds(dir * ".log"), steps=Int(f[2]),
            t=f[3], Nut=f[11], Nub=f[12], Vrms=f[13], meanT=f[10],
            N=Int(f[5]), leaves=Int(f[6]),
            frames=frames, meta_rows=length(frows),
            fstep=Int(f[2]), ft=f[3],
            hist=(t=last[1], Nut=getc("Nu_top"), Nub=getc("Nu_bot"),
                  Vrms=getc("Vrms"), meanT=getc("meanT"),
                  N=Int(getc("N")), leaves=Int(getc("leaves"))),
            T=read_T(joinpath(dir, "frames", @sprintf("frame_%05d_T.bin", Int(f[2])))))
end

function frame_diff_series(a::AbstractString, b::AbstractString)
    fa = Dict{Int,String}(); fb = Dict{Int,String}()
    for f in readdir(joinpath(a, "frames"))
        m = match(r"^frame_(\d+)_T\.bin$", f)
        m === nothing || (fa[parse(Int, m.captures[1])] = f)
    end
    for f in readdir(joinpath(b, "frames"))
        m = match(r"^frame_(\d+)_T\.bin$", f)
        m === nothing || (fb[parse(Int, m.captures[1])] = f)
    end
    steps = sort(collect(intersect(keys(fa), keys(fb))))
    out = NamedTuple[]
    for s in steps
        A = read_T(joinpath(a, "frames", fa[s]))
        B = read_T(joinpath(b, "frames", fb[s]))
        push!(out, (step=s, diff=field_diff(A, B)))
    end
    return out
end

DIRECT = length(ARGS) >= 1 ? ARGS[1] : "out_julia/verify100_direct"
MG     = length(ARGS) >= 2 ? ARGS[2] : "out_julia/verify100_mg"
GREF   = length(ARGS) >= 3 ? ARGS[3] : ""

d = arm_summary(DIRECT)
m = arm_summary(MG)

println("verify100_compare.jl")
@printf("  direct: %s\n  mg    : %s\n", DIRECT, MG)

println("\n[0] initial state (t=0) and the saddle size")
qt0 = Quadtree(7; Lx=LX, cells_x=5, cells_y=2)
for _ in 1:3, lf in collect(qt0.leaves)
    refine!(qt0, lf)
end
balance!(qt0)
ss0 = make_semstokes(qt0, 2)
@printf("  card initial tree: %d leaves, N=%d, Np=%d, saddle (2N+Np) = %d dofs\n",
        length(qt0.leaves), ss0.su.N, ss0.sp.N, 2 * ss0.su.N + ss0.sp.N)
for (name, dir) in (("direct", DIRECT), ("mg", MG))
    ck = joinpath(dir, "spinup.h5")
    if isfile(ck)
        st = load_semquad_state(ck)
        ssf = make_semstokes(st.qt, st.p)
        @printf("  %s checkpoint step %6d: %5d leaves, N=%d, Np=%d, saddle = %d dofs\n",
                name, st.step, length(st.qt.leaves), ssf.su.N, ssf.sp.N,
                2 * ssf.su.N + ssf.sp.N)
    end
end

println("\n[1] end-state diagnostics")
@printf("  %-10s %14s %14s %14s %14s\n", "quantity", "direct", "mg", "mg-direct", "rel")
for (name, a, b) in (("t", d.t, m.t), ("Nu_top", d.Nut, m.Nut),
                     ("Nu_bot", d.Nub, m.Nub), ("Vrms", d.Vrms, m.Vrms),
                     ("meanT", d.meanT, m.meanT))
    rel = a == 0 ? NaN : (b - a) / abs(a)
    @printf("  %-10s %14.9g %14.9g %+14.3e %+14.3e\n", name, a, b, b - a, rel)
end
@printf("  %-10s %14d %14d %+14d\n", "leaves", d.leaves, m.leaves, m.leaves - d.leaves)
@printf("  %-10s %14d %14d %+14d\n", "N", d.N, m.N, m.N - d.N)
@printf("  %-10s %14d %14d %+14d\n", "steps", d.steps, m.steps, m.steps - d.steps)
@printf("  %-10s %14d %14d %+14d\n", "frames", d.frames, m.frames, m.frames - d.frames)
@printf("  %-10s %14.1f %14.1f %+14.1f\n", "wall_s", d.wall, m.wall, m.wall - d.wall)

println("\n[2] final field difference (step $(d.fstep) against step $(m.fstep), 256x256 grid)")
if d.fstep == m.fstep
    fd = field_diff(d.T, m.T)
    @printf("  max |dT|                 = %.6e\n", fd.maxabs)
    @printf("  rms |dT|                 = %.6e\n", fd.rms)
    @printf("  rms |dT| / rms(theta)    = %.6e\n", fd.rel_anom)
    @printf("  max |dT| / (Tmax - Tmin) = %.6e\n", fd.span)
else
    println("  final frames are at different steps; skipped")
end

println("\n[3] field difference over the common frames")
ser = frame_diff_series(DIRECT, MG)
if isempty(ser)
    println("  no common frames")
else
    worst = argmax([x.diff.maxabs for x in ser])
    w = ser[worst]
    @printf("  common frames: %d (steps %d .. %d)\n", length(ser), ser[1].step, ser[end].step)
    @printf("  worst max|dT| = %.6e at step %d\n", w.diff.maxabs, w.step)
    for thr in (1e-8, 1e-6, 1e-4, 1e-2)
        hit = findfirst(x -> x.diff.maxabs > thr, ser)
        @printf("  first frame with max|dT| > %.0e: %s\n", thr,
                hit === nothing ? "none" : "step $(ser[hit].step) (max|dT| = $(@sprintf("%.2e", ser[hit].diff.maxabs)))")
    end
    @printf("  first frame: step %d max|dT| = %.3e rms = %.3e\n",
            ser[1].step, ser[1].diff.maxabs, ser[1].diff.rms)
end

println("\n[3b] mesh evolution: leaf count per frame, direct against mg")
_, fd_rows = read_table(joinpath(DIRECT, "frames", "frame_meta.csv"))
_, fm_rows = read_table(joinpath(MG, "frames", "frame_meta.csv"))
# frame_meta columns: frame,step,t,dt,N,leaves,nlev,...

function mesh_evolution(fd_rows, fm_rows)
    fs_d = Dict(Int(r[2]) => r for r in fd_rows)
    fs_m = Dict(Int(r[2]) => r for r in fm_rows)
    common = sort(collect(intersect(keys(fs_d), keys(fs_m))))
    first_diff = nothing
    max_dl = 0
    max_dn = 0
    ndiff = 0
    for s in common
        dl = Int(fs_d[s][6]) - Int(fs_m[s][6])
        dn = Int(fs_d[s][5]) - Int(fs_m[s][5])
        if dl != 0 || dn != 0
            ndiff += 1
            first_diff === nothing && (first_diff = (s, dl, dn, Int(fs_d[s][6]), Int(fs_m[s][6])))
            max_dl = max(max_dl, abs(dl))
            max_dn = max(max_dn, abs(dn))
        end
    end
    return (common=common, first_diff=first_diff, ndiff=ndiff,
            max_dl=max_dl, max_dn=max_dn, fs_d=fs_d, fs_m=fs_m)
end

mev = mesh_evolution(fd_rows, fm_rows)
common = mev.common
if isempty(common)
    println("  no common steps")
else
    @printf("  common frame steps: %d\n", length(common))
    if mev.first_diff === nothing
        println("  leaf count and N are IDENTICAL at every common frame")
    else
        f = mev.first_diff
        @printf("  first divergence: step %d, leaves direct-mg = %+d (direct %d, mg %d), N delta %+d\n",
                f[1], f[2], f[4], f[5], f[3])
        @printf("  frames with any mesh difference: %d of %d; max |d leaves| = %d, max |dN| = %d\n",
                mev.ndiff, length(common), mev.max_dl, mev.max_dn)
    end
    # the last five common frames make the end-state mesh difference readable
    k0 = max(1, length(common) - 4)
    for s in common[k0:end]
        @printf("  step %3d: direct leaves=%5d N=%6d | mg leaves=%5d N=%6d | dleaves=%+d\n",
                s, Int(mev.fs_d[s][6]), Int(mev.fs_d[s][5]),
                Int(mev.fs_m[s][6]), Int(mev.fs_m[s][5]),
                Int(mev.fs_d[s][6]) - Int(mev.fs_m[s][6]))
    end
end

function compare_with_g(dir::AbstractString, gref::AbstractString)
    println("\n[4] the same card against the production G run (direct, 1 thread)")
    for step in (1, 496, 500)
        fa = joinpath(dir, "frames", @sprintf("frame_%05d_T.bin", step))
        fg = joinpath(gref, "frames", @sprintf("frame_%05d_T.bin", step))
        if isfile(fa) && isfile(fg)
            fd = field_diff(read_T(fa), read_T(fg))
            @printf("  step %3d: max|dT| = %.3e rms = %.3e (rms/ana = %.2e)\n",
                    step, fd.maxabs, fd.rms, fd.rel_anom)
        end
    end
end
isempty(GREF) || compare_with_g(DIRECT, GREF)

println("\n[5] wall clock")
@printf("  direct %.1f s (%.1f min, %.3f s/step)\n", d.wall, d.wall / 60,
        d.wall / d.steps)
@printf("  mg     %.1f s (%.1f min, %.3f s/step)\n", m.wall, m.wall / 60,
        m.wall / m.steps)
