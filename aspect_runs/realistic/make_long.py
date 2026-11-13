#!/usr/bin/env python3
"""Generate `realistic_E_long.prm`: resume the run from the checkpoint that
`run_E_cont` inherited from `run_E` (t = 8.05918e15 s, step 4759) and carry the
same case out to

    End time   8.41e15 s (t_nd = 1e-3)  ->  4.205e16 s (t_nd = 5e-3)

which is the long-time window the cross-code Nu_bot question needs (section 17).
Only these three lines differ from `realistic_E.prm`:

    End time                       8.41e15 s  ->  4.205e16 s   (1e-3 -> 5e-3)
    Output directory               run_E      ->  run_E_long
    Time between graphical output   5e14 s    ->  5e13 s       (5.9e-6 t_nd/frame)

Everything else -- the plugin, the phase transitions, AMR fractions/levels, the
solver tolerances, the checkpoint interval -- is `realistic_E.prm` verbatim, so
the physics is identical to run_E / run_E_cont.

`Resume computation = auto` looks for the checkpoint in the *output* directory,
so `run_E_cont/restart.*` is copied into `run_E_long/` (both read-only here).

Cost, measured on run_E_cont (same mesh, same ranks): 0.90 s/step, 1894 steps
per 5.4e-4 of t_nd, i.e. ~3.5e-3 of t_nd needs ~1.2e4 steps / ~3 h.

Usage:  python make_long.py
"""
import difflib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "realistic_E.prm")
DST = os.path.join(HERE, "realistic_E_long.prm")
OUT = os.path.join(HERE, "run_E_long")
SRC_OUT = os.path.join(HERE, "run_E_cont")

SUBS = [
    ("set End time                               = 8.41e15",
     "set End time                               = 4.205e16"),
    ("set Output directory                       = run_E",
     "set Output directory                       = run_E_long"),
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
