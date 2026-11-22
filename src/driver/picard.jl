# The Picard iteration helpers, split by concern:
#
#   `strain_rate.jl`      the nodal strain-rate and spin fields;
#   `picard_control.jl`   the field blend, the divergence test and the
#                         adaptive relaxation schedule.
#
# The include order in `src/QuaST.jl` follows that list.
