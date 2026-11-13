#!/usr/bin/env python3
"""Generate `realistic_E6_long.prm`: carry the E6 configuration (bottom band
y_nd <= 0.05 at level 6, h = 1/64 -- the same wall-normal resolution as our
card) from its own checkpoint out to

    End time   8.41e15 s (t_nd = 1e-3)  ->  2.523e16 s (t_nd = 3e-3)

which is the window the "is ASPECT's bottom-flux variability an under-resolution
effect of the L5 wall?" question needs (section 18).

Only these three lines differ from `realistic_E6.prm`:

    End time                        8.41e15 s  ->  2.523e16 s   (1e-3 -> 3e-3)
    Output directory                run_E6     ->  run_E6_long
    Time between graphical output    5e14 s    ->  5e13 s       (5.9e-6 t_nd/frame)

Everything else -- the plugin, the phase transitions, the minimum-refinement
function that holds the bottom band at level 6, the AMR fractions/levels, the
solver tolerances, the checkpoint interval, `Resume computation = auto` -- is
`realistic_E6.prm` verbatim, so the physics is identical to run_E6 (same three
functional knobs as E6, only the end time / output bookkeeping differ).

`Resume computation = auto` looks for the checkpoint in the *output* directory,
so `run_E6/restart.*` is copied into `run_E6_long/` (source is read-only here).
The E6 checkpoint is from step 3716, t = 6.38887e15 s (t_nd = 7.5968e-4), i.e.
ASPECT re-walks 7.5968e-4 .. 1e-3 before entering new territory; that segment is
a deterministic replay (section 17.1 measured bit-identical reproduction on the
same kind of resume), so it is not an error source.

Usage:  python make_e6_long.py
"""
import difflib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "realistic_E6.prm")
DST = os.path.join(HERE, "realistic_E6_long.prm")
OUT = os.path.join(HERE, "run_E6_long")
SRC_OUT = os.path.join(HERE, "run_E6")

SUBS = [
    ("set End time                               = 8.41e15",
     "set End time                               = 2.523e16"),
    ("set Output directory                       = run_E6",
     "set Output directory                       = run_E6_long"),
    ("set Time between graphical output = 5e14",
     "set Time between graphical output = 5e13"),
]


def main():
    txt = open(SRC).read()
    for old, new in SUBS:
        n = txt.count(old)
        if n != 1:
            sys.exit(f"expected exactly one `{old}`, found {n}")
        txt = txt.replace(old, new)
    d = list(difflib.unified_diff(open(SRC).read().splitlines(),
                                  txt.splitlines(), lineterm="", n=0))
    changes = [l for l in d if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    assert len(changes) == 6, changes            # exactly 3 pairs
    open(DST, "w").write(txt)
    print(f"wrote {DST}  ({len(changes)//2} substitutions)")

    os.makedirs(OUT, exist_ok=True)
    for name in ("restart.mesh", "restart.mesh.info", "restart.mesh_fixed.data",
                 "restart.resume.z"):
        s = os.path.join(SRC_OUT, name)
        if not os.path.isfile(s):
            sys.exit(f"missing {s}")
        shutil.copy2(s, os.path.join(OUT, name))
        print(f"  copied {name} ({os.path.getsize(s)} bytes)")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
