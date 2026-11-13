using Test
using QuaST

# Run one or more gate files in a single Julia process:
#
#     julia --project=. test/run_gates.jl t_g136_box_aspect.jl t_g149_ev_cbf.jl
#
# Paths may be bare file names (resolved against `test/`) or paths
# relative to the repository root / absolute paths.  This is the runner used by
# `.github/workflows/gates.yml`; it is tracked (unlike the scratch
# `out_julia/agent/run_gates.jl`).

const _ROOT = normpath(joinpath(@__DIR__, ".."))

function _gate_path(f::AbstractString)
    isabspath(f) && return f
    isfile(f) && return abspath(f)
    p = joinpath(_ROOT, "test", f)
    isfile(p) || error("gate file not found: $f (also tried $p)")
    return p
end

isempty(ARGS) && error("usage: run_gates.jl <gate file> [gate file ...]")
for f in ARGS
    path = _gate_path(f)
    @info "including $(relpath(path, _ROOT))"
    include(path)
end
