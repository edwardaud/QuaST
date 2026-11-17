"""log10(viscosity) images for the two arms of the verification clip.

Two sources, one output convention: an (nz, nx) array of log10(eta / eta0) on the
same regular grid the temperature panels use, so a single fixed colour range
serves both rows.

Our arm reads what the solver wrote per frame: `frame_NNNNN_eta.bin` holds
log10(eta) at every velocity node and `frame_NNNNN_nodes.bin` the node
coordinates.  The nodes are scattered over the quadtree, so they are binned onto
the grid and the holes filled from the neighbours.

ASPECT's arm rebuilds the field from ASPECT's own temperature and velocity,
because ASPECT's saved `viscosity` is the unclipped value and runs to 1e141 Pa s.
The law is the shared `:composite` one, ported from `src/physics/constitutive.jl`
and checked against it to 5e-15 relative over 245 (T, edot, z) points.
"""
import base64
import glob
import io
import os
import re
import struct
import zlib

import numpy as np

# --- the law, from src/physics/constitutive.jl -------------------------------
A_D = 1.6240183119e5
A_S = 2.6796419238e-1
E_D = 12.0
E_S = 17.0
V_D = 0.0
V_S = 0.0
N = 3.0
ETA_T0 = 0.5
EPS_MIN = 1.0e-12
ETA_MIN = 1.0e-4
ETA_MAX = 1.0e4
_TOL = 1.0e-13
_MAXIT = 200

# ASPECT's units: metres, kelvin, 1/s
D_LEN, KAPPA, T_TOP, DT = 2.9e6, 1.0e-6, 500.0, 1000.0


def eta_composite(T, edot, z):
    """Effective viscosity at non-dimensional T, the strain-rate invariant and z."""
    T = np.asarray(T, dtype="float64")
    edot = np.asarray(edot, dtype="float64")
    z = np.asarray(z, dtype="float64")
    Tnd = T + ETA_T0
    depth = 1.0 - z
    inv_diff, inv_disl = np.broadcast_arrays(
        A_D * np.exp(-(E_D + V_D * depth) / Tnd),
        A_S * np.exp(-(E_S + V_S * depth) / (N * Tnd))
        * np.maximum(edot, EPS_MIN) ** ((N - 1.0) / N))
    k = inv_diff / inv_disl
    out = np.empty(k.shape[0], dtype="float64")
    kf, idf, idl = k.reshape(-1), inv_diff.reshape(-1), inv_disl.reshape(-1)
    for i in range(out.size):
        kk = kf[i]
        if not np.isfinite(kk):
            out[i] = 1.0 / idf[i]
            continue
        x = 1.0 / (1.0 + kk)
        lo, hi = (x if N >= 1.0 else 0.0), 1.0
        x = x if x > 0.0 else 0.5
        for _ in range(_MAXIT):
            xn = x ** N
            g = kk * x + xn - 1.0
            if abs(g) <= _TOL:
                break
            if g < 0.0:
                lo = x
            else:
                hi = x
            dg = kk + N * xn / x
            xnew = x - g / dg if (np.isfinite(dg) and dg > 0.0) else np.nan
            if not np.isfinite(xnew) or xnew <= lo or xnew >= hi:
                xnew = 0.5 * (lo + hi)
            if abs(xnew - x) <= 4.0 * np.finfo(float).eps:
                x = xnew
                break
            x = xnew
        out[i] = x / idl[i]
    return np.clip(out, ETA_MIN, ETA_MAX)


def strain_rate_invariant(exx, ezz, exz):
    return np.sqrt(0.5 * (exx ** 2 + ezz ** 2 + 2.0 * exz ** 2))


# --- binned images -----------------------------------------------------------
def _fill_holes(img, mask):
    out = img.copy()
    have = mask.copy()
    for _ in range(600):
        if have.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            sh = np.roll(np.roll(out, dz, axis=0), dx, axis=1)
            sm = np.roll(np.roll(have, dz, axis=0), dx, axis=1)
            acc += np.where(sm, sh, 0.0)
            cnt += sm
        grow = (~have) & (cnt > 0)
        out[grow] = acc[grow] / cnt[grow]
        have |= grow
    return out


def bin_nodes(x, z, values, lx, nx, nz):
    ix = np.clip(((x / lx) * (nx - 1)).round().astype(int), 0, nx - 1)
    iz = np.clip((z * (nz - 1)).round().astype(int), 0, nz - 1)
    acc = np.zeros((nz, nx))
    cnt = np.zeros((nz, nx))
    np.add.at(acc, (iz, ix), values)
    np.add.at(cnt, (iz, ix), 1.0)
    ok = cnt > 0
    img = np.where(ok, acc / np.maximum(cnt, 1.0), 0.0)
    if not ok.all():
        img = _fill_holes(img, ok)
    return img


def rasterize_cells(corners_x, corners_z, values, lx, nx, nz):
    """Write each cell's value into every pixel whose centre lies inside it.

    Rounding cell centres onto the grid leaves the boundary rows empty, and
    filling those from their neighbours is what produced the triangular wedges
    along the top and bottom of the ASPECT panel.  A pixel belongs to exactly
    one cell here, so the image has no holes and no staircase.
    """
    img = np.full((nz, nx), np.nan)
    for k in range(values.size):
        xs = corners_x[k]
        zs = corners_z[k]
        i0 = int(np.ceil(xs.min() / lx * (nx - 1)))
        i1 = int(np.floor(xs.max() / lx * (nx - 1)))
        j0 = int(np.ceil(zs.min() * (nz - 1)))
        j1 = int(np.floor(zs.max() * (nz - 1)))
        i0 = max(i0, 0); j0 = max(j0, 0)
        i1 = min(i1, nx - 1); j1 = min(j1, nz - 1)
        if i1 >= i0 and j1 >= j0:
            img[j0:j1 + 1, i0:i1 + 1] = values[k]
    holes = np.isnan(img)
    if holes.any():                      # only the outermost pixels can be left
        img = _fill_holes(np.where(holes, 0.0, img), ~holes)
    return img


def our_eta_grid(frames_dir, step, lx, nx, nz):
    e = np.fromfile(os.path.join(frames_dir, "frame_%05d_eta.bin" % step),
                    dtype=np.float32)
    n = np.fromfile(os.path.join(frames_dir, "frame_%05d_nodes.bin" % step),
                    dtype=np.float32)
    return bin_nodes(n[0::2].astype("float64"), n[1::2].astype("float64"),
                     e.astype("float64"), lx, nx, nz)


# --- ASPECT ------------------------------------------------------------------
_PAT = re.compile(r"<DataArray([^>]*)>(.*?)</DataArray>", re.S)
_POINTS = re.compile(r"<Points>\s*<DataArray([^>]*)>(.*?)</DataArray>", re.S)
_DTYPE = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8",
          "UInt8": "u1", "UInt32": "<u4", "UInt64": "<u8"}


def _decode(b):
    i = b.index("==") + 2
    head = base64.b64decode(b[:i])
    rest = base64.b64decode(b[i:])
    nb, us, bs, cs = struct.unpack_from("<IIII", head, 0)
    return zlib.decompress(rest[:cs])


def _array(attrs, payload):
    m = re.search(r'type="(\w+)"', attrs)
    dt = np.dtype(_DTYPE[m.group(1)])
    b = _decode(payload.strip())
    n = len(b) // dt.itemsize
    return np.frombuffer(b[:n * dt.itemsize], dtype=dt)


def _field(text, name):
    if name == "Points":
        m = _POINTS.search(text)
        return None if m is None else _array(m.group(1), m.group(2))
    for m in _PAT.finditer(text):
        if 'Name="%s"' % name in m.group(1) and 'format="binary"' in m.group(1):
            return _array(m.group(1), m.group(2))
    return None


def aspect_eta_grid(solution_dir, index, lx, nx, nz):
    """Rebuild ASPECT's viscosity at one output and bin it onto the grid."""
    pieces = sorted(glob.glob(os.path.join(solution_dir,
                                           "solution-%05d.*.vtu" % index)))
    if not pieces:
        raise SystemExit("no pieces for ASPECT output %05d" % index)
    P, C, T, V, off = [], [], [], [], 0
    for f in pieces:
        text = io.open(f, "rb").read().decode("latin-1")
        pts = _field(text, "Points").reshape(-1, 3)
        P.append(pts)
        C.append(_field(text, "connectivity") + off)
        T.append(_field(text, "T"))
        V.append(_field(text, "velocity").reshape(-1, 3)[:, :2])
        off += pts.shape[0]
    P = np.vstack(P)
    C = np.concatenate(C).reshape(-1, 4)
    T = np.concatenate(T)
    V = np.vstack(V)

    xy = P[C][:, :, :2]
    uv = V[C]
    tq = T[C]
    a = np.array([-1.0, 1.0, 1.0, -1.0])
    b = np.array([-1.0, -1.0, 1.0, 1.0])
    dxi, det = 0.25 * a, 0.25 * b
    dx_dxi = np.einsum("i,ni->n", dxi, xy[:, :, 0])
    dz_dxi = np.einsum("i,ni->n", dxi, xy[:, :, 1])
    dx_det = np.einsum("i,ni->n", det, xy[:, :, 0])
    dz_det = np.einsum("i,ni->n", det, xy[:, :, 1])
    jac = dx_dxi * dz_det - dx_det * dz_dxi
    dux_dxi = np.einsum("i,ni->n", dxi, uv[:, :, 0])
    dux_det = np.einsum("i,ni->n", det, uv[:, :, 0])
    duz_dxi = np.einsum("i,ni->n", dxi, uv[:, :, 1])
    duz_det = np.einsum("i,ni->n", det, uv[:, :, 1])
    dux_dx = (dz_det * dux_dxi - dx_det * dux_det) / jac
    dux_dz = (-dz_dxi * dux_dxi + dx_dxi * dux_det) / jac
    duz_dx = (dz_det * duz_dxi - dx_det * duz_det) / jac
    duz_dz = (-dz_dxi * duz_dxi + dx_dxi * duz_det) / jac
    edot = strain_rate_invariant(dux_dx, duz_dz, 0.5 * (dux_dz + duz_dx))

    xc = xy[:, :, 0].mean(axis=1) / D_LEN
    zc = xy[:, :, 1].mean(axis=1) / D_LEN
    tc = (tq.mean(axis=1) - T_TOP) / DT
    eta = eta_composite(tc, edot * D_LEN ** 2 / KAPPA, zc)
    return rasterize_cells(xy[:, :, 0] / D_LEN, xy[:, :, 1] / D_LEN,
                           np.log10(eta), lx, nx, nz)
