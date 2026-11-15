#!/usr/bin/env julia
# Turn the benchmark logs into the rows of the validation table. Run from the
# repository root after `scripts/run_benchmarks.sh` has produced results:
#
#     julia --project=. scripts/benchmark_table.jl
#
# For each log in out_julia/benchmarks it prints one Markdown row: the case, its
# parameters, the computed and reference values with the relative difference, and
# a status. A log without a final line is a run still in flight, and the row says
# so rather than quoting a transient number.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const LOGDIR = joinpath(ROOT, "out_julia", "benchmarks")

# case -> (Ra, heating, viscosity, reference Nu, reference Vrms)
# The case definitions are ASPECT's benchmarks/blankenbach inputs: 1a/1b/1c are
# isoviscous at Ra = 1e4/1e5/1e6 with no internal heating, 2a adds eta(T) with a
# contrast of 1000, and 2b is the joint temperature and depth law.
const CASES = Dict(
    "1a" => ("1e4", "0", "constant", 4.8844093, 42.864956),
    "1b" => ("1e5", "0", "constant", 10.534095, 193.21454),
    "1c" => ("1e6", "0", "constant", 21.972465, 833.98977),
    "2a" => ("1e4", "0", "Δη_T = 1e3", 10.066, 480.4334),
    "2b" => ("1e4", "0", "temperature and depth", 6.9299, 171.755),
)

fmt(x, n = 6) = string(round(x, digits = n))
rel(a, b) = b == 0 ? "n/a" : string(round(100 * (a - b) / b, digits = 2), "%")

function parse_log(path)
    final = nothing
    last_step = nothing
    for line in eachline(path)
        if startswith(line, "[") && occursin("] steps=", line)
            final = line
        elseif occursin("step ", line) && occursin("Nu_top=", line)
            last_step = line
        end
    end
    return final, last_step
end

println("| case | `Ra` | heating | viscosity | `Nu` | ref. | Δ | `Vrms` | ref. | Δ | status |")
println("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")

for case in ("1a", "1b", "1c", "2a", "2b")
    logs = filter(f -> startswith(f, case * "_L") && endswith(f, ".log"), readdir(LOGDIR))
    isempty(logs) && continue
    path = joinpath(LOGDIR, sort(logs)[end])
    ra, heat, visc, refnu, refvr = CASES[case]
    final, last_step = parse_log(path)
    level = match(r"_L(\d+)", basename(path))
    lvl = level === nothing ? "?" : level.captures[1]

    if final === nothing
        step_txt = last_step === nothing ? "no output yet" :
                   "at " * strip(split(last_step, r"\s+")[end - 4]) * ", still transient"
        # A log with the `aborted` marker was stopped on purpose, so it must not
        # read as a run still in flight.
        state = occursin("aborted", basename(path)) ? "aborted, $step_txt" : "running, $step_txt"
        println("| $case (L$lvl) | $ra | $heat | $visc | — | $refnu | — | — | $refvr | — | $state |")
        continue
    end
    m = match(r"Nu_top=([-\d.eE+]+) Nu_bot=([-\d.eE+]+) Vrms=([-\d.eE+]+)", final)
    if m === nothing
        println("| $case (L$lvl) | $ra | $heat | $visc | — | $refnu | — | — | $refvr | — | unparsable log |")
        continue
    end
    nu = parse(Float64, m.captures[1])
    nb = parse(Float64, m.captures[2])
    vr = parse(Float64, m.captures[3])
    balanced = abs(nu - nb) <= 0.005 * abs(nu)
    status = balanced ? "steady (top and bottom agree)" : "not steady (ΔNu = " * rel(nu, nb) * ")"
    println("| $case (L$lvl) | $ra | $heat | $visc | $(fmt(nu)) | $refnu | $(rel(nu, refnu)) | " *
            "$(fmt(vr)) | $refvr | $(rel(vr, refvr)) | $status |")
end
