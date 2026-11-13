#!/usr/bin/env python3
"""Generate `realistic_E_cont.prm`: resume `run_E` from its own checkpoint and
continue the same case, same card, with only these three changes:

    End time                     8.41e15 s  ->  1.2615e16 s   (t_nd 1e-3 -> 1.5e-3)
    Output directory             run_E      ->  run_E_cont
    Time between graphical output  5e14 s   ->  2.5e13 s       (3e-6 t_nd per frame)

Everything else is `realistic_E.prm` verbatim (`Resume computation = auto` is
already in it).  `run_E/restart.*` is copied into `run_E_cont/` first, because
`Resume computation = auto` looks for the checkpoint in the *output* directory.

Measured 2026-11-10: the resumed run started at t = 8.2419e15 s (t_nd 9.80e-4),
i.e. the checkpoint lags the run's end (8.41e15 s) by 2.0e-4 in t_nd because the
checkpoint interval is 1800 s of wall time and the run finished 104 s after its
last checkpoint.  The comparison therefore uses the *first VTU frame* of the
continuation as the common state (its time is carried in the frame's TIME field
and in `run_E_cont/statistics`).

Usage:  python make_cont.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "realistic_E.prm")
DST = os.path.join(HERE, "realistic_E_cont.prm")
OUT = os.path.join(HERE, "run_E_cont")
SRC_OUT = os.path.join(HERE, "run_E")

SUBS = [
    ("set End time                               = 8.41e15",
     "set End time                               = 1.2615e16"),
    ("set Output directory                       = run_E",
     "set Output directory                       = run_E_cont"),
    ("set Time between graphical output = 5e14",
     "set Time between graphical output = 2.5e13"),
]


def main():
    txt = open(SRC).read()
    head, sep, body = txt.partition("set End time")
    for old, new in SUBS:
        n = txt.count(old)
        if n != 1:
            sys.exit(f"expected exactly one `{old}`, found {n}")
        txt = txt.replace(old, new)
    # nothing else may differ
    import difflib
    d = list(difflib.unified_diff(open(SRC).read().splitlines(),
                                  txt.splitlines(), lineterm="", n=0))
    changes = [l for l in d if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    assert len(changes) == 6, changes            # 3 pairs
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
