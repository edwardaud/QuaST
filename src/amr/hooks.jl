# The AMR hooks the driver calls.
#
# This subject is split by concern:
#
#   `wall_refine.jl`  refining the wall strips to a minimum level before a run;
#   `amr_pass.jl`     one adaptive pass (indicator, marking, coarsening, balance,
#                     solution transfer) during a run.
#
# The include order in `src/QuaST.jl` follows that list.
