#!/usr/bin/env python3
"""Direct boundary-layer profile comparison: ASPECT `run_E` vs `case_realistic_E`.

Why this script exists
----------------------
The window-mean `Nu_bot` differs by -6.1 % (ASPECT -3.6368, ours -3.4274) over
`t_nd = 1e-6 .. 1e-3`, and `docs/dev/VALIDATION.md` section 10 leaves
the late-window part (-9.7 % in `t_nd = 3e-4 .. 1e-3`) attributed to "the two
solutions' bottom boundary layers" without decomposing it.  This script does the
decomposition on *measured profiles*: it extracts the x-averaged temperature
profile `T(y)` near the bottom wall from both codes at matched `t_nd`, computes
the one-sided wall gradient (which is `Nu_bot` by our definition), evaluates the
composite isostress viscosity on both profiles, and tests whether the
temperature difference can carry the viscosity difference that the `Nu_bot`
difference would require.

Data (read-only; the running `case_realistic_E` is never touched)
----------------------------------------------------------------
ASPECT : `run_E/solution/solution-NNNNN.RRRR.vtu`, 6 MPI pieces per frame,
         `Time between graphical output = 5e14 s`.  deal.II's `DataOut` writes
         each Q2 cell as a 2x2 patch of *bilinear* quads whose vertices are the
         Q2 support points, so every field is available exactly at the Q2 nodes
         (y = 0, h/2, h, ... with h = 1/32 for the bottom row).  Fields used:
         `T`, `velocity`, `viscosity`.
ours   : `out_julia/case_realistic_E/frames/frame_NNNNN_T.bin` - 256x256
         Float64, `vec(permutedims(T))` with `T[j,i] = T(y_j, x_i)`,
         `x in [0, 2.5]`, `y in [0, 1]` (both endpoints included), i.e.
         `numpy.fromfile(...).reshape(256, 256)` is `A[y, x]` and the file holds
         the *full* temperature `Tbar(y; H) + theta`, not theta
         (`julia/scripts/case_realistic_2d.jl`, `write_frame!`).

Matched times are the exact ASPECT visualization times (from `solution.pvd`),
so the ASPECT row is a snapshot and our side is linearly interpolated in time
between the two bracketing frames (both frames and the size of the temporal
interpolation are reported).

Units: everything is converted to our nondimensional units with the section-2.1
scale set (`T_nd = T_K/1000 - 0.5`, `y_nd = y_m/d`, `t_nd = t*kappa/d^2`,
`u_nd = u*d/kappa`).  The viscosity is compared in units of `eta0` (ASPECT's
`viscosity` column is in Pa s, `eta0 = 2.368635e23 Pa s`).

Usage
-----
  python3 bl_profile_compare.py                  # all late-window VTU frames
  python3 bl_profile_compare.py --times 5.35e-4 7.14e-4 8.92e-4
  python3 bl_profile_compare.py --frames 8 12 15 16
  python3 bl_profile_compare.py --eta-check /tmp/eta_julia.csv
"""

import glob
import math
import os
import re
import sys

import numpy as np

# --- the scale set of docs/dev/VALIDATION.md section 2.1 ----------
D = 2.9e6                 # box depth [m]
KAPPA = 1.0e-6            # thermal diffusivity [m^2/s]
DT = 1000.0               # temperature drop [K]
L_X = 2.5 * D             # box length [m]
K_COND = 3.3              # kappa cp rho0 [W/(m K)]
NORM = L_X * K_COND * DT / D      # W per unit of Nu (= 8250)
T0 = D * D / KAPPA        # 8.41e18 s, the time scale
ETA0 = 2.368635e23        # Pa s

# --- the composite law of docs/dev/PHYSICS.md section 1 (case card) -----------
COMP_AD = 1.6240183119e5
COMP_AS = 2.6796419238e-1
COMP_ED = 12.0
COMP_ES = 17.0
COMP_N = 3.0
ETA_T0 = 0.5
COMP_EPS_MIN = 1.0e-12
ETA_MIN, ETA_MAX = 1.0e-4, 1.0e4

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SOLUTION = os.path.join(HERE, "run_E", "solution")
PVD = os.path.join(HERE, "run_E", "solution.pvd")
STATS = os.path.join(HERE, "run_E", "statistics")
FRAMES = os.path.join(REPO, "out_julia", "case_realistic_E", "frames")

trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------- viscosity --
def eta_branches(T):
    """Inverse branch viscosities `inv_diff`, `inv_disl_unit` (edot included)."""
    Tnd = np.maximum(np.asarray(T, dtype=float) + ETA_T0, 0.0)
    inv_diff = COMP_AD * np.exp(-COMP_ED / Tnd)
    inv_disl_unit = COMP_AS * np.exp(-COMP_ES / (COMP_N * Tnd))
    return inv_diff, inv_disl_unit, Tnd


def eta_isostress(T, edot, z=0.0):
    """Our `:composite` / `:isostress` viscosity (docs/dev/PHYSICS.md section 1.1).

    `eta_diff = Ad^-1 exp(Ed/T_nd)`, `eta_disl = As^-1 exp(Es/(n T_nd)) ed^((n-1)/n)`;
    isostress: solve `k x + x^n = 1` with `k = inv_diff/inv_disl`, `eta = x/inv_disl`.
    Vectorised; returns `(eta, x)` in code units (`eta0 = 1`).  `z` is unused
    (`comp_Vd = comp_Vs = 0` on this card) but kept for signature parity with
    `eta_composite_at`.
    """
    edot = np.maximum(np.asarray(edot, dtype=float), COMP_EPS_MIN)
    inv_diff, inv_disl_unit, _ = eta_branches(T)
    inv_disl = inv_disl_unit * edot ** ((COMP_N - 1.0) / COMP_N)
    k = inv_diff / inv_disl
    x = 1.0 / (1.0 + k)          # exact for n = 1, the Newton start otherwise
    for _ in range(80):
        xn = x ** COMP_N
        g = k * x + xn - 1.0
        dg = k + COMP_N * xn / x
        dx = g / dg
        x = np.clip(x - dx, 1e-300, 1.0)
        if np.all(np.abs(dx) <= 4e-16 * np.maximum(x, 1e-300)):
            break
    return np.clip(x / inv_disl, ETA_MIN, ETA_MAX), x


def eta_diff_only(T):
    inv_diff, _, _ = eta_branches(T)
    return 1.0 / inv_diff


def edot_from_eta(T, eta_target):
    """Invert `eta_isostress` for the strain rate (bisection in log10 edot)."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    eta_target = np.atleast_1d(np.asarray(eta_target, dtype=float))
    lo = np.full(T.shape, -6.0)
    hi = np.full(T.shape, 8.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        eta = eta_isostress(T, 10.0 ** mid)[0]
        need_more = eta > eta_target     # too viscous -> more strain rate
        lo = np.where(need_more, mid, lo)
        hi = np.where(need_more, hi, mid)
    return 10.0 ** (0.5 * (lo + hi))


# ------------------------------------------------------------------- ASPECT --
def read_pvd(path=PVD):
    """[(index, time_s, file)] from the .pvd collection."""
    out = []
    txt = open(path, errors="replace").read()
    for m in re.finditer(r'<DataSet\s+timestep="([^"]+)"[^>]*file="([^"]+)"', txt):
        idx = int(re.search(r"solution-(\d+)\.pvtu", m.group(2)).group(1))
        out.append((idx, float(m.group(1)), os.path.join(os.path.dirname(path), m.group(2))))
    out.sort()
    return out


def _meshio():
    import meshio
    return meshio


def aspect_frame(idx):
    """All pieces of ASPECT frame `idx`.

    Returns dict with 1-D arrays `x, y, T, ux, uy, eta` (Q2 support points, all
    pieces concatenated) and the cell connectivity (needed for the pointwise
    strain rate).
    """
    files = sorted(glob.glob(os.path.join(SOLUTION, "solution-%05d.*.vtu" % idx)))
    if not files:
        raise SystemExit("no VTU pieces for frame %d" % idx)
    meshio = _meshio()
    x, y, T, ux, uy, et = [], [], [], [], [], []
    cells = []
    off = 0
    t_s = None
    for f in files:
        m = meshio.read(f)
        if t_s is None:
            t_s = float(m.field_data["TIME"][0])
        p = m.points
        n = p.shape[0]
        quad = [c for c in m.cells if c.type == "quad"]
        if len(quad) != 1:
            raise SystemExit("expected one quad block in %s" % f)
        cells.append(quad[0].data + off)
        off += n
        x.append(p[:, 0] / L_X)
        y.append(p[:, 1] / D)
        T.append(m.point_data["T"].astype(np.float64) / DT - 0.5)
        v = m.point_data["velocity"]
        ux.append(v[:, 0].astype(np.float64) * D / KAPPA)
        uy.append(v[:, 1].astype(np.float64) * D / KAPPA)
        et.append(m.point_data["viscosity"].astype(np.float64) / ETA0)
    return dict(t_s=t_s, x=np.concatenate(x), y=np.concatenate(y),
                T=np.concatenate(T), ux=np.concatenate(ux), uy=np.concatenate(uy),
                eta=np.concatenate(et), cells=np.concatenate(cells))


def strain_rate_cells(fr):
    """Pointwise `edot = sqrt(0.5 e:e)` at the cell centres of the Q2 patches.

    Each output quad is a bilinear patch with the Q2 support points at its
    corners, so a least-squares plane through the four corner velocities gives
    the gradient at the patch centre (exact for a rectangle).  Returns
    `(xc, yc, edot, dx, dy)`; the patches are h/2 tall, so `yc` sits between the
    Q2 levels (y = 1/128, 3/128, ... for the bottom row).
    """
    c = fr["cells"]
    x = fr["x"][c]           # (ncells, 4)
    y = fr["y"][c]
    ux = fr["ux"][c]
    uy = fr["uy"][c]
    # normal equations of a plane fit for each patch
    A = np.stack([np.ones_like(x), x, y], axis=-1)          # (ncells,4,3)
    AtA = np.einsum("nki,nkj->nij", A, A)
    Atb = np.stack([np.einsum("nki,nk->ni", A, np.stack([ux, uy], -1)[..., j])
                    for j in range(2)], -1)
    sol = np.linalg.solve(AtA, Atb)                          # (ncells,3,2)
    dux_dx, dux_dy = sol[:, 1, 0], sol[:, 2, 0]
    duy_dx, duy_dy = sol[:, 1, 1], sol[:, 2, 1]
    exy = 0.5 * (dux_dy + duy_dx)
    edot = np.sqrt(0.5 * (dux_dx ** 2 + 2 * exy ** 2 + duy_dy ** 2))
    return x.mean(1), y.mean(1), edot, dux_dy, duy_dy


def levels(x, y, fields, key=7, cover_min=0.999):
    """x-average every field on every y level of the (unstructured) support set.

    Each y level is integrated in x with the trapezoid rule over the covered
    range and divided by the covered length.  Levels whose x coverage is below
    `cover_min` (an AMR patch that spans only part of the box) are returned but
    flagged `valid = False` - they must not be used for a profile.
    """
    yy = np.round(y, key)
    out = dict(y=[], cover=[], n=[], mean=[], med=[], p10=[], p90=[], lo=[], hi=[])
    for lev in np.unique(yy):
        sel = yy == lev
        xs = x[sel]
        o = np.argsort(xs)
        xs = xs[o]
        keep = np.concatenate([[True], np.diff(xs) > 1e-12])
        xs = xs[keep]
        if xs.size < 2:
            continue
        cov = xs[-1] - xs[0]
        out["y"].append(lev)
        out["cover"].append(cov)
        out["n"].append(int(xs.size))
        for k in range(len(fields)):
            fs = np.asarray(fields[k])[sel][o][keep]
            out["mean"].append(trapz(fs, xs) / cov)
        for k in range(len(fields)):
            fs = np.sort(np.asarray(fields[k])[sel])
            out["med"].append(np.median(fs))
            out["p10"].append(np.percentile(fs, 10))
            out["p90"].append(np.percentile(fs, 90))
            out["lo"].append(fs[0])
            out["hi"].append(fs[-1])
    res = {k: np.array(v) for k, v in out.items()}
    res["valid"] = res["cover"] >= cover_min
    res["mean"] = np.array([np.array(m, dtype=float) for m in out["mean"]]).reshape(
        len(out["y"]), len(fields), -1)
    for tag in ("med", "p10", "p90", "lo", "hi"):
        res[tag] = np.array([np.array(m, dtype=float) for m in out[tag]]).reshape(
            len(out["y"]), len(fields), -1)
    return res


def aspect_profile(idx):
    fr = aspect_frame(idx)
    # pointwise "how far below the pure diffusion branch": 1 - eta/eta_diff(T).
    # Done pointwise and only then averaged - the x-mean of eta is *not* eta of
    # the x-mean T (eta_diff is convex in T), so averaging first would bias it.
    red = 1.0 - fr["eta"] / eta_diff_only(fr["T"])
    lv = levels(fr["x"], fr["y"], [fr["T"], fr["eta"], fr["ux"], red])
    p = dict(t_s=fr["t_s"], t=fr["t_s"] / T0, frame=idx,
             y=lv["y"], cover=lv["cover"], valid=lv["valid"],
             T=lv["mean"][:, 0, 0], eta=lv["mean"][:, 1, 0],
             eta_med=lv["med"][:, 1, 0], eta_lo=lv["lo"][:, 1, 0], eta_hi=lv["hi"][:, 1, 0],
             ux=lv["mean"][:, 2, 0],
             red_med=lv["med"][:, 3, 0], red_mean=lv["mean"][:, 3, 0],
             red_p90=lv["p90"][:, 3, 0], red_glob=float(np.median(red)),
             red_glob_p90=float(np.percentile(red, 90)),
             red_glob_max=float(red.max()),
             red_glob_frac=float(np.mean(red > 0.01)))
    xc, yc, ed, dux_dy, duy_dy = strain_rate_cells(fr)
    yy = np.round(yc, 7)
    ey, emed, ep90, ep10, emean = [], [], [], [], []
    for lev in np.unique(yy):
        sel = yy == lev
        ey.append(lev)
        emed.append(np.median(ed[sel]))
        ep90.append(np.percentile(ed[sel], 90))
        ep10.append(np.percentile(ed[sel], 10))
        emean.append(ed[sel].mean())
    p["ey"] = np.array(ey)
    p["edot_med"] = np.array(emed)
    p["edot_p90"] = np.array(ep90)
    p["edot_p10"] = np.array(ep10)
    p["edot_mean"] = np.array(emean)
    return p


# --------------------------------------------------------------------- ours --
def read_meta(path=os.path.join(FRAMES, "frame_meta.csv")):
    out = []
    with open(path, errors="replace") as f:
        header = f.readline().strip().split(",")
        for line in f:
            p = line.strip().split(",")
            if len(p) != len(header):
                continue
            r = dict(zip(header, p))
            try:
                out.append(dict(step=int(r["step"]), frame=int(r["frame"]),
                                t=float(r["t"]), tmax=float(r["Tmax"]),
                                tmean=float(r["Tmean"]), nu_bot=float(r["Nu_bot"]),
                                nu_top=float(r["Nu_top"]), vrms=float(r["Vrms"]),
                                umax=float(r["umax"]), leaves=int(r["leaves"])))
            except (ValueError, KeyError):
                continue
    return out


_NG = 256
_YS = np.linspace(0.0, 1.0, _NG)
_XS = np.linspace(0.0, 1.0, _NG)


def our_profile(step):
    """x-averaged T(y) of one of our frames (row index = y, 0..1)."""
    A = np.fromfile(os.path.join(FRAMES, "frame_%05d_T.bin" % step),
                    dtype="<f8").reshape(_NG, _NG)
    return _YS, np.array([trapz(A[j], _XS) / (_XS[-1] - _XS[0]) for j in range(_NG)])


def bracket(series, t):
    lo = None
    for i, r in enumerate(series):
        if r["t"] <= t:
            lo = i
    if lo is None or lo == len(series) - 1:
        return None, None
    return lo, lo + 1


def interp_meta(meta, key, t):
    lo, hi = bracket(meta, t)
    if lo is None:
        return float("nan")
    a, b = meta[lo], meta[hi]
    if b["t"] == a["t"]:
        return a[key]
    w = (t - a["t"]) / (b["t"] - a["t"])
    return a[key] + w * (b[key] - a[key])


def our_profile_at(t, meta):
    """Our x-averaged T(y) linearly interpolated in time to `t`."""
    lo, hi = bracket(meta, t)
    if lo is None:
        return None
    if abs(meta[lo]["t"] - t) < 1e-13:
        y, T = our_profile(meta[lo]["step"])
        return dict(t=t, y=y, T=T, steps=(meta[lo]["step"], 0), dT_all=0.0, dT_bl=0.0)
    y0, T0p = our_profile(meta[lo]["step"])
    y1, T1p = our_profile(meta[hi]["step"])
    w = (t - meta[lo]["t"]) / (meta[hi]["t"] - meta[lo]["t"])
    d = np.abs(T1p - T0p)
    sub = y0 <= 0.0501
    return dict(t=t, y=y0, T=(1 - w) * T0p + w * T1p,
                steps=(meta[lo]["step"], meta[hi]["step"]),
                dT_all=float(d.max()), dT_bl=float(d[sub].max()))


# ------------------------------------------------------- wall gradient (1D) --
def wall_gradient(y, T, k=3):
    """One-sided `dT/dy` at y=0 from the first `k` profile points.

    The polynomial of degree `k-1` through them is differentiated at y=0
    (non-uniform abscissae).  `k = 3` is the unique quadratic through
    (0, h1, h2) - the same family as the codes' own one-sided GLL/Q2 wall
    derivative (docs/dev/VALIDATION.md section 10.4).
    """
    y = np.asarray(y[:k], dtype=float)
    T = np.asarray(T[:k], dtype=float)
    return np.linalg.solve(np.vander(y, k, increasing=True), T)[1]


def nu_bot_from_profile(y, T, k=3):
    return -wall_gradient(y, T, k)


# -------------------------------------------------------------- statistics --
def read_stats(path=STATS):
    names, rows = {}, []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                m = re.match(r"#\s*(\d+):\s*(.*)", line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip()
                continue
            if line:
                rows.append(line.split())

    def col(tag):
        for i, n in names.items():
            if n.startswith(tag):
                return i - 1
        raise KeyError(tag)

    c_t = col("Time (seconds)")
    c_tmax = col("Maximal temperature")
    c_vrms = col("RMS velocity")
    c_cells = col("Number of mesh cells")
    c_qbot = col('Outward heat flux through boundary with indicator 2 ("bottom")')
    c_qtop = col('Outward heat flux through boundary with indicator 3 ("top")')
    out = []
    for v in rows:
        try:
            t_s = float(v[c_t])
            out.append(dict(t_s=t_s, t=t_s / T0,
                            nu_bot=-float(v[c_qbot]) / NORM,
                            nu_top=float(v[c_qtop]) / NORM,
                            tmax=float(v[c_tmax]) / DT - 0.5,
                            vrms=float(v[c_vrms]) * D / KAPPA,
                            cells=int(float(v[c_cells]))))
        except (ValueError, IndexError):
            continue
    return out


def nearest_stats(stats, t_s):
    return min(stats, key=lambda r: abs(r["t_s"] - t_s))


def pct(a, b):
    return 100.0 * (a - b) / abs(b) if b else float("nan")


def at_y(y, f, y0):
    """Our (fine, uniform) profile interpolated linearly to y0 (error ~1e-5)."""
    j = np.searchsorted(y, y0)
    j = min(max(j, 1), len(y) - 1)
    w = (y0 - y[j - 1]) / (y[j] - y[j - 1])
    return f[j - 1] + w * (f[j] - f[j - 1])


# ------------------------------------------------------------------- report --
def eta_check(path):
    """Compare this script's law with `eta_law_dump.jl` (Julia's own function)."""
    ref = np.atleast_2d(np.genfromtxt(path, delimiter=",", comments="#"))
    T, ed, eta_jl, clip = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
    eta_py, _ = eta_isostress(T, ed)
    rel = np.abs(eta_py - eta_jl) / np.abs(eta_jl)
    free = (eta_jl > ETA_MIN * 1.001) & (eta_jl < ETA_MAX * 0.999)
    print("eta law check against Julia's `eta_composite_at`: %d points, "
          "T %.2f..%.2f, edot %.1e..%.1e" % (T.size, T.min(), T.max(), ed.min(), ed.max()))
    print("  worst relative difference = %.3e  (all points)" % rel.max())
    print("  worst off the clamps       = %.3e" % rel[free].max())
    print("  eta = eta_max (clamped) at %d points" % int(clip.sum()))
    return 0


def main(argv):
    times, frame_ids, eta_path = [], [], None
    ys_max = 0.05
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--eta-check":
            return eta_check(argv[i + 1])
        elif a == "--times":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                times.append(float(argv[i])); i += 1
        elif a == "--frames":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                frame_ids.append(int(argv[i])); i += 1
        elif a == "--ymax":
            ys_max = float(argv[i + 1]); i += 2
        elif a in ("-h", "--help"):
            print(__doc__); return 0
        else:
            raise SystemExit("unknown argument: " + a)

    pvd = read_pvd()
    meta = read_meta()
    stats = read_stats()
    print("ASPECT : %d VTU frames, t_nd %.3e .. %.3e, %d statistics rows"
          % (len(pvd), pvd[0][1] / T0, pvd[-1][1] / T0, len(stats)))
    print("ours   : %d frames, t_nd %.3e .. %.3e, ng = %d, dt_nd ~ %.2e"
          % (len(meta), meta[0]["t"], meta[-1]["t"], _NG,
             np.median(np.diff([m["t"] for m in meta]))))

    if frame_ids:
        todo = [p for p in pvd if p[0] in frame_ids]
    elif times:
        todo = sorted({min(pvd, key=lambda p: abs(p[1] / T0 - t)) for t in times},
                      key=lambda p: p[0])
    else:
        todo = [p for p in pvd if 3.0e-4 <= p[1] / T0 <= 1.0e-3]

    # ------------------------------------------------ 1. matched-time table --
    print("\n== 1. matched times: the diagnostics (A = ASPECT, O = ours) ==")
    head = ("%-10s %-10s %-11s %9s %9s %7s %8s %8s %7s %9s %9s %7s"
            % ("t_nd", "t_nd(O)", "frames_O", "Nu_bot_A", "Nu_bot_O", "d%",
               "Tmax_A", "Tmax_O", "d%", "Vrms_A", "Vrms_O", "yTmax_A"))
    print(head); print("-" * len(head))
    payload = []
    for (idx, t_s, f) in todo:
        row = nearest_stats(stats, t_s)
        prof = aspect_profile(idx)
        our = our_profile_at(prof["t"], meta)
        if our is None:
            print("%-10.3e  no bracketing frames" % prof["t"])
            continue
        payload.append((prof, our, row))
        jt = np.argmax(prof["T"][prof["valid"]])
        print("%-10.3e %-10.3e %-11s %9.4f %9.4f %7.2f %8.4f %8.4f %7.2f %9.3e %9.3e %7.4f"
              % (prof["t"], our["t"], "%d/%d" % our["steps"],
                 row["nu_bot"], interp_meta(meta, "nu_bot", prof["t"]),
                 pct(row["nu_bot"], interp_meta(meta, "nu_bot", prof["t"])),
                 row["tmax"], interp_meta(meta, "tmax", prof["t"]),
                 pct(row["tmax"], interp_meta(meta, "tmax", prof["t"])),
                 row["vrms"], interp_meta(meta, "vrms", prof["t"]),
                 prof["y"][prof["valid"]][jt]))
    print("(Nu_bot = ASPECT's CBF column / our point value; Tmax = the reported")
    print(" values; our side is interpolated between the two bracketing frames.)")
    if not payload:
        return 1

    # ---------------------------------------- 2. wall gradient extraction ----
    print("\n== 2. one-sided wall gradient from the profile vs the codes' own Nu_bot ==")
    head = ("%-10s %10s %10s %8s %10s %10s %9s %8s %9s"
            % ("t_nd", "A k=3", "A CBF", "dA%", "O k=3", "O k=4", "O point", "dO%",
               "dT_bl"))
    print(head); print("-" * len(head))
    for prof, our, row in payload:
        m = prof["valid"]
        yv, Tv = prof["y"][m], prof["T"][m]
        gA = nu_bot_from_profile(yv, Tv, 3)
        gO3 = nu_bot_from_profile(our["y"], our["T"], 3)
        gO4 = nu_bot_from_profile(our["y"], our["T"], 4)
        nuO = interp_meta(meta, "nu_bot", prof["t"])
        print("%-10.3e %10.4f %10.4f %8.2f %10.4f %10.4f %9.4f %8.3f %9.2e"
              % (prof["t"], gA, row["nu_bot"], pct(gA, row["nu_bot"]),
                 gO3, gO4, nuO, pct(gO3, nuO), our["dT_bl"]))
    print("(A k=3 = the 3-point one-sided quadratic through (0, h/2, h) on the ASPECT")
    print(" profile; that is the codes' own wall-derivative family.  A CBF is ASPECT's")
    print(" reported column, so dA% is the extraction error (0.1-1.5%).  O k=3 and")
    print(" O k=4 are the same formula on our 256-point profile, where they are exact;")
    print(" dT_bl = the temporal drift between the two bracketing frames, y <= %.3f.)"
          % ys_max)

    # -------------------------------------------------- 3. profile table ----
    print("\n== 3. the near-bottom temperature profile T(y), y <= %.3f ==" % ys_max)
    for prof, our, row in payload:
        m = prof["valid"]
        yv, Tv = prof["y"][m], prof["T"][m]
        print("\n-- t_nd = %.4e (ASPECT frame %d; ours = step %d/%d, |dT|max in the BL = %.2e)"
              % (prof["t"], prof["frame"], our["steps"][0], our["steps"][1], our["dT_bl"]))
        print("%-9s %-12s %-12s %-11s %-9s %-9s %-9s"
              % ("y", "T_A", "T_O", "dT (A-O)", "100dT/T_nd", "dT/dy_A", "dT/dy_O"))
        for j in range(yv.size):
            if yv[j] > ys_max + 1e-9:
                break
            TO = at_y(our["y"], our["T"], yv[j])
            dTdy_A = gA_local(yv, Tv, j)
            dTdy_O = gA_local(our["y"], our["T"], np.argmin(np.abs(our["y"] - yv[j])))
            print("%-9.5f %-12.6f %-12.6f %-+11.3e %-9.3f %-9.4f %-9.4f"
                  % (yv[j], Tv[j], TO, Tv[j] - TO, 100 * (Tv[j] - TO) / (Tv[j] + ETA_T0),
                     dTdy_A, dTdy_O))
    print("(T in our units, T_nd = T + %.1f; our profile is linearly interpolated in y"
          % ETA_T0)
    print(" (max error 1e-5) and in time between the bracketing frames.  dT/dy is the")
    print(" derivative of the quadratic through the three nearest profile points, i.e.")
    print(" the same rule as the wall gradient of section 2.)")

    # ------------------------------------- 3b. how dT grows into the interior --
    print("\n== 3b. the same difference out to mid-depth, and the phase functions ==")
    print("(Gamma_i = 0.5(1 + tanh(((y0_i - y) - slope_i (T - 0.5))/w)) with")
    print(" y0 = 0.87 / 0.78, w = 0.025, slope = +0.032 / -0.032, evaluated on T_A)")
    picks = (0.0, 1 / 64, 1 / 32, 3 / 64, 1 / 16, 0.10, 0.15, 0.20, 0.30, 0.50)
    for prof, our, row in payload:
        m = prof["valid"]
        yv, Tv = prof["y"][m], prof["T"][m]
        print("\n-- t_nd = %.4e" % prof["t"])
        print("%-9s %-11s %-11s %-11s %-9s %-10s %-10s"
              % ("y", "T_A", "T_O", "dT (A-O)", "100dT/T_nd", "Gamma_410", "Gamma_660"))
        for ystar in picks:
            below = np.where(np.abs(yv - ystar) <= 0.004)[0]
            if below.size == 0:
                continue
            j = below[np.argmin(np.abs(yv[below] - ystar))]
            TO = at_y(our["y"], our["T"], yv[j])
            g410 = gamma(yv[j], Tv[j], 0.87, +0.032)
            g660 = gamma(yv[j], Tv[j], 0.78, -0.032)
            print("%-9.5f %-11.6f %-11.6f %-+11.3e %-9.3f %-10.6f %-10.6f"
                  % (yv[j], Tv[j], TO, Tv[j] - TO, 100 * (Tv[j] - TO) / (Tv[j] + ETA_T0),
                     g410, g660))
    print("(the transitions live at y = 0.78 / 0.87 with a half-width of 0.025, so")
    print(" inside y <= 0.05 both Gamma are 1 to 1e-20 and carry no gradient there.)")

    # ------------------------------------------------------ 4. viscosity ----
    print("\n== 4. viscosity in the bottom BL: measured (ASPECT VTU) vs the T-only law ==")
    print("(red% = the pointwise 1 - eta/eta_diff, i.e. how far the dislocation")
    print(" branch pushes the viscosity below the pure diffusion value; median over x.")
    print(" ratio = eta_diff(T_O)/eta_A(VTU); ratio_full uses the measured median")
    print(" strain rate at that y for BOTH temperatures, and dln(eta) = ln(ratio).)")
    for prof, our, row in payload:
        m = prof["valid"]
        yv, Tv, etaV = prof["y"][m], prof["T"][m], prof["eta"][m]
        print("\n-- t_nd = %.4e   (red over the whole box: median %.2e, p90 %.2e,"
              % (prof["t"], prof["red_glob"], prof["red_glob_p90"]))
        print("   max %.2f (= eta_max clamp), fraction with red > 1%%: %.3f)"
              % (prof["red_glob_max"], prof["red_glob_frac"]))
        print("%-9s %-11s %-11s %-10s %-11s %-9s %-9s %-11s"
              % ("y", "eta_A(VTU)", "eta_diff(T_A)", "red", "eta_diff(T_O)",
                 "ratio", "dln(eta)", "ratio_full"))
        for j in range(yv.size):
            if yv[j] > ys_max + 1e-9:
                break
            TO = at_y(our["y"], our["T"], yv[j])
            eAd = eta_diff_only(Tv[j])
            eOd = eta_diff_only(TO)
            ed = edot_at(prof, yv[j])
            ef = (eta_isostress(TO, ed)[0] / eta_isostress(Tv[j], ed)[0]
                  if np.isfinite(ed) else float("nan"))
            print("%-9.5f %-11.4e %-11.4e %-10.2e %-11.4e %-9.4f %-+9.4f %-11.4f"
                  % (yv[j], etaV[j], eAd, prof["red_med"][m][j], eOd,
                     eOd / etaV[j], math.log(eOd / etaV[j]), ef))
        print("   wall: eta_A = %.6e, eta_diff(T_A = %.6f) = %.6e"
              % (etaV[0], Tv[0], eta_diff_only(Tv[0])))
    print("(ratio_full ~ ratio: red < 1e-4 in the BL means the dislocation branch")
    print(" carries ~1e-4 of the strain rate there, so eta is a function of T only.)")

    # ------------------------------------------------ 5. measured strain rate --
    print("\n== 5. measured pointwise strain rate in the bottom BL (ASPECT VTU) ==")
    print("(edot = sqrt(0.5 e:e) of the Q2 velocity field at the patch centres, one")
    print(" value per patch; the statistics below are over all patches with y<0.05)")
    head = ("%-10s %9s %9s %9s %9s %9s %9s"
            % ("t_nd", "edot_p10", "edot_med", "edot_p90", "n", "Vrms_A", "Vrms_O"))
    print(head); print("-" * len(head))
    for prof, our, row in payload:
        sel = (prof["ey"] < ys_max) & (prof["ey"] > 0)
        if not sel.any():
            continue
        e = prof["edot_med"][sel]
        p10, p90 = prof["edot_p10"][sel], prof["edot_p90"][sel]
        print("%-10.3e %9.2e %9.2e %9.2e %9d %9.3e %9.3e"
              % (prof["t"], np.median(p10), np.median(e), np.median(p90),
                 int(sel.sum()), row["vrms"], interp_meta(meta, "vrms", prof["t"])))
    print("(p10 / med / p90 are the medians over the levels of each level's own")
    print(" percentile, i.e. a typical spread of the strain rate inside the BL.)")

    print("\n  sensitivity of the eta ratio to edot (ASPECT's measured median edot at")
    print("  y = 1/64, both temperatures evaluated at the same edot):")
    for prof, our, row in payload:
        m = prof["valid"]
        yv, Tv = prof["y"][m], prof["T"][m]
        j = int(np.argmin(np.abs(yv - 1.0 / 64)))
        TO = at_y(our["y"], our["T"], yv[j])
        ed0 = edot_at(prof, yv[j])
        rs = [eta_isostress(TO, ed0 * f)[0] / eta_isostress(Tv[j], ed0 * f)[0]
              for f in (0.0625, 0.25, 1.0, 4.0, 16.0)]
        print("    t_nd %.3e  y = %.5f  edot0 = %.2e:  ratio = %s"
              % (prof["t"], yv[j], ed0, "  ".join("%.4f" % r for r in rs)))
    print("    (edot / edot0 = 0.06x 0.25x 1x 4x 16x: a 256-fold change in the strain")
    print("     rate moves the ratio by < 1e-3, because red < 1e-4 in the BL.)")

    # ------------------------------------------------ 6. the chain test -----
    print("\n== 6. the chain: dT -> deta -> dNu ==")
    print("Conductive-BL flux: with delta(y*) = dT(y*) / Nu  (the 'conductive")
    print("thickness' of the layer below y*)")
    print("   dln(Nu) = dln(dT(y*)) - dln(delta(y*)),   and delta ~ eta^(1/3)")
    print("so the observed dln(Nu) requires dln(eta)_req = 3*dln(delta).")
    print()
    head = ("%-10s %8s %8s %8s %9s %9s %9s %9s %9s"
            % ("t_nd", "dlnNu", "Nu_A", "Nu_O", "dlnNu_req", "dlnEta_meas", "dT_meas", "dT_req",
               "dTmax"))
    print(head); print("-" * len(head))
    acc = []
    for prof, our, row in payload:
        m = prof["valid"]
        yv, Tv, etaV = prof["y"][m], prof["T"][m], prof["eta"][m]
        nuA, nuO = row["nu_bot"], interp_meta(meta, "nu_bot", prof["t"])
        dln_nu = math.log(nuA / nuO)
        dln, dTs, Ts = [], [], []
        for j in range(yv.size):
            if yv[j] > ys_max + 1e-9:
                break
            TO = at_y(our["y"], our["T"], yv[j])
            dln.append(math.log(eta_diff_only(TO) / etaV[j]))
            dTs.append(Tv[j] - TO)
            Ts.append(Tv[j] + ETA_T0)
        dln_m, dT_m = float(np.mean(dln)), float(np.mean(dTs))
        Tmean = float(np.mean(Ts))
        # dT(y*) and delta(y*) for the *same* y*, so the split below is exact
        dT_bl_A = Tv[-1] - Tv[0] if yv[-1] <= ys_max + 1e-9 else None
        acc.append((dln_nu, dln_m, dT_m, Tmean, nuA, nuO,
                    float(np.mean(Ts)), prof, our, yv, Tv, etaV))
        print("%-10.3e %8.4f %8.4f %8.4f %9.3f %9.3f %+9.4f %9.3f %+9.4f"
              % (prof["t"], dln_nu, nuA, nuO, 3 * dln_nu, dln_m, dT_m,
                 (3 * dln_nu) * Tmean * Tmean / COMP_ED,
                 pct(row["tmax"], interp_meta(meta, "tmax", prof["t"]))))
    a = np.array([[c[0], c[1], c[2]] for c in acc])
    print("%-10s %8.4f %8s %8s %9.3f %9.3f %+9.4f %9s %9s"
          % ("mean", a[:, 0].mean(), "-", "-", 3 * a[:, 0].mean(), a[:, 1].mean(),
             a[:, 2].mean(), "-", "-"))
    print()
    print("dlnEta_meas = <ln(eta_O/eta_A)> over y <= %.2f, eta_O = eta_diff(T_O) and" % ys_max)
    print(" eta_A the measured VTU value; dT_meas = <T_A - T_O>; dT_req = dlnEta_req*T^2/Ed.")

    # the exact flux/thickness decomposition at two yardsticks
    print("\n  flux decomposition  dlnNu = dln(dT(y*)) - dln(delta(y*)):")
    print("  %-10s %-8s %10s %10s %10s %10s %10s %9s"
          % ("t_nd", "y*", "dlnNu", "dln dT", "dln delta", "del_A", "del_O", "eta^(1/3)"))
    for (dln_nu, dln_m, dT_m, Tmean, nuA, nuO, _g, prof, our, yv, Tv, etaV) in acc:
        for ystar in (0.05, 0.1, 0.2):
            below = np.where(yv <= ystar + 1e-12)[0]
            if below.size == 0:
                continue
            jA = below[-1]
            TO = at_y(our["y"], our["T"], yv[jA])
            dTA, dTO = Tv[jA] - Tv[0], TO - our["T"][0]
            delA, delO = dTA / abs(nuA), dTO / abs(nuO)
            print("  %-10.3e %-8.3f %10.4f %10.4f %10.4f %10.5f %10.5f %9.4f"
                  % (prof["t"], yv[jA], dln_nu, math.log(dTA / dTO), math.log(delO / delA),
                     delA, delO, math.exp(dln_m) ** (1.0 / 3.0)))
    print("  (dln dT + (-dln delta) = dlnNu exactly at each y*; delta(y*) = dT(y*)/|Nu|.")
    print("   eta^(1/3) is the delta ratio that the measured viscosity ratio predicts.)")

    # ------------------------------------------------- 7. top-wall control --
    print("\n== 7. control: the top wall (thick BL, where Nu_top agrees to 0.03%) ==")
    for prof, our, row in payload:
        m = prof["valid"]
        yv, Tv = prof["y"][m], prof["T"][m]
        top = yv > 1.0 - ys_max - 1e-9
        if not top.any():
            continue
        nuTA = row["nu_top"]
        nuTO = interp_meta(meta, "nu_top", prof["t"])
        print("\n-- t_nd = %.4e  Nu_top A = %.5f  O = %.5f  d = %.3f%%  (max |dT| = %.2e)"
              % (prof["t"], nuTA, nuTO, pct(nuTA, nuTO),
                 float(np.max(np.abs([Tv[j] - at_y(our["y"], our["T"], yv[j])
                                      for j in range(yv.size) if top[j]])))))
        print("%-9s %-12s %-12s %-11s %-9s" % ("y", "T_A", "T_O", "dT (A-O)", "100dT"))
        for j in range(yv.size):
            if not top[j]:
                continue
            TO = at_y(our["y"], our["T"], yv[j])
            print("%-9.5f %-12.6f %-12.6f %-+11.3e %-9.3f"
                  % (yv[j], Tv[j], TO, Tv[j] - TO, 100 * (Tv[j] - TO)))
    return 0


def edot_at(prof, y0):
    """Measured median strain rate at height `y0` (linear in y, nan off the grid)."""
    if not prof["ey"].size:
        return float("nan")
    return float(np.interp(y0, prof["ey"], prof["edot_med"]))


def gamma(y, T, y0, slope, width=0.025, Tref=0.5):
    """The case's phase function Gamma_i(y, T) (docs/dev/PHYSICS.md, case card)."""
    return 0.5 * (1.0 + math.tanh(((y0 - y) - slope * (T - Tref)) / width))


def gA_local(y, f, j):
    """`df/dy` at profile index `j` from the unique quadratic through the three
    nearest points (one-sided at the ends) - the same rule as `wall_gradient`."""
    n = len(y)
    if j == 0:
        idx = [0, 1, 2]
    elif j >= n - 1:
        idx = [n - 3, n - 2, n - 1]
    else:
        idx = [j - 1, j, j + 1]
    idx = [k for k in idx if 0 <= k < n]
    c = np.linalg.solve(np.vander(np.asarray(y)[idx], len(idx), increasing=True),
                        np.asarray(f)[idx])
    return c[1] + 2.0 * c[2] * y[j] if len(idx) == 3 else c[1]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
