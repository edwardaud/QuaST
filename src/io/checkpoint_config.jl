# The `key=value` configuration codec of a checkpoint.
#
# A checkpoint stores the run settings as one canonical string, and the restart
# guard compares each stored value against the one the new run would use.  Both
# sides of that comparison go through this file, so how a value is spelled is
# defined in one place (docs/dev/HANDOVER.md section 7).

"""Canonical string form of a run setting, used to compare checkpoints."""
config_value_string(v::Symbol) = string(v)
config_value_string(v::Bool) = v ? "true" : "false"
config_value_string(v::Real) = string(Float64(v))
config_value_string(v) = string(v)

"""Parse the `key=value` config string stored in a checkpoint."""
function parse_config_string(s::AbstractString)
    out = Dict{String,String}()
    for line in split(s, '\n')
        isempty(strip(line)) && continue
        i = findfirst('=', line)
        i === nothing && continue
        out[String(line[1:prevind(line, i)])] = String(line[nextind(line, i):end])
    end
    return out
end
