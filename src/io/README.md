# I/O

Checkpoint and restart. `save_semquad_state` and `load_semquad_state` write and
read the complete quadtree state as one deterministic HDF5 file: the quadtree
(leaves and `maxlevel`), both BDF2 temperature levels, velocity and pressure,
the elementwise entropy-viscosity field used by the last energy assembly, the
optional AMR cooldown, and the scalars `t, dt, dt_prev, step, p`. Everything
needed for a bit-identical resume is here; visualisation frames and
`history.csv` are written by the case scripts, not by this directory.

| File | What it is |
|---|---|
| `checkpoint.jl` | HDF5 checkpoint/restart of the full quadtree SEM state. |
| `checkpoint_cooldown.jl` | The optional `:balanced` AMR cooldown datasets. |
| `checkpoint_config.jl` | The `key=value` configuration codec of a checkpoint. |
| `checkpoint_tree.jl` | Rebuilding the quadtree a checkpoint describes, including its `seam` topology. |
