"""CrossViewWarp 360 — the CrossView-Warp conditioning for equirectangular (360) video.

The same warp as the perspective CrossViewWarp node (and the training-time builder): every source pixel is lifted
to 3D with its metric distance, moved into the target camera, painted far-to-near with the same splat passes, and
whatever no pixel lands on stays magenta. Only the projection differs: both cameras are equirectangular (x right,
y up, z forward = the image centre), the target moves along a path of positions (no rotation: a 360 frame holds
every direction), the image wraps around horizontally, and the splat widens toward the poles, where an equirect
pixel covers less of the sphere horizontally.

Several source frames can be given: the FIRST is the primary - frame 0 of a path starting at its camera reproduces
it exactly - and the others only fill what the primary leaves magenta (rendered together, the other frames'
surfaces poked through the primary's, so a chained video would not continue seamlessly).
"""

import json
import logging

import numpy as np
import torch

from .crossview_warp_node import MAGENTA, NODE_SUFFIX

try:
    from numba import njit

    @njit(cache=False)
    def _splat360_kernel(tu0, tv0, rx, ry, cols, zs, warp, zbuf, offs, H, W, ztest):
        # offset passes from the outermost ring to the centre (see _offsets), points far-to-near inside each; with
        # ztest a true z-buffer (the nearest point wins every pixel it reaches): the perspective kernel's painter passes
        # (offsets outer, points inner) let a far point's outer splat pixels overwrite a near point's centre - with
        # its fixed 2 px splat that is the training format, with a splat grown by magnification it punched the floor
        # through a near chair
        n = tu0.shape[0]
        for k in range(offs.shape[0]):
            dy = offs[k, 0]
            dx = offs[k, 1]
            for i in range(n):
                if dx < -rx[i] or dx > rx[i] or dy < -ry[i] or dy > ry[i]:
                    continue
                y = tv0[i] + dy
                if 0 <= y < H:
                    x = (tu0[i] + dx) % W
                    if ztest and zs[i] > zbuf[y, x]:
                        continue
                    warp[y, x, 0] = cols[i, 0]
                    warp[y, x, 1] = cols[i, 1]
                    warp[y, x, 2] = cols[i, 2]
                    zbuf[y, x] = zs[i]
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _dirs(W, H):
    lon = ((np.arange(W) + 0.5) / W - 0.5) * 2 * np.pi
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)], -1)


def warp360(rgb, dist, src_pos, tgt_pos, splat=2, see_through=False, mode="training", return_depth=False):
    """mode "training": the perspective node's splat exactly - fixed radius, painter passes, no depth test - only widened
    in x toward the poles (an equirect pixel there is narrower on the sphere); "zbuffer": the radius grows with each
    point's magnification and the nearest point wins every pixel."""
    """One equirect warp. rgb (H, W, 3) uint8, dist (H, W) metres along each ray (NaN / <= 0 = unknown),
    src_pos / tgt_pos (3,) camera positions. Returns (H, W, 3) uint8, magenta where nothing landed."""
    H, W = dist.shape
    fin = np.isfinite(dist) & (dist > 0)
    P = _dirs(W, H) * np.where(fin, dist, 0)[..., None] + np.asarray(src_pos, np.float64)
    D = P.reshape(-1, 3) - np.asarray(tgt_pos, np.float64)
    r = np.linalg.norm(D, axis=1)
    ok = fin.ravel() & (r > 1e-6)
    with np.errstate(invalid="ignore", divide="ignore"):
        lon = np.arctan2(D[:, 0], D[:, 2])
        lat = np.arcsin(np.clip(D[:, 1] / r, -1, 1))
    u = np.round((lon / (2 * np.pi) + 0.5) * W - 0.5)
    v = np.round((0.5 - lat / np.pi) * H - 0.5)
    ok &= np.isfinite(u) & np.isfinite(v)
    order = np.argsort(-np.where(ok, r, -1))
    sel = order[ok[order]]
    tu0 = (u[sel].astype(np.int64)) % W
    tv0 = np.clip(v[sel].astype(np.int64), 0, H - 1)
    # the splat grows with the point's magnification (its source distance / its target distance: walking up to a
    # chair spreads it over more pixels, a fixed splat left it a sieve with the floor showing through) and, in x,
    # toward the poles
    rmax = 4 * max(splat, 1)
    src_r = dist.ravel()[sel]
    mag = np.maximum(src_r / np.maximum(r[sel], 1e-6), 1.0) if mode == "zbuffer" else np.ones(len(sel))
    ry = np.minimum(np.ceil(splat * mag), rmax).astype(np.int64)
    rx = np.minimum(np.ceil(splat * mag / np.maximum(np.cos(lat[sel]), 1e-3)), rmax).astype(np.int64)
    cols = np.ascontiguousarray(rgb.reshape(-1, 3)[sel])
    zs = np.ascontiguousarray(r[sel])
    offs = _offsets(rmax)
    warp = np.tile(MAGENTA, (H, W, 1))
    zbuf = np.full((H, W), np.inf)
    done = False
    if _HAVE_NUMBA:
        try:
            _splat360_kernel(tu0, tv0, rx, ry, cols, zs, warp, zbuf, offs, H, W, mode == "zbuffer")
            done = True
        except Exception:
            logging.exception("CrossViewWarp360: numba kernel failed, numpy fallback")
            warp = np.tile(MAGENTA, (H, W, 1))
            zbuf = np.full((H, W), np.inf)
    if not done:
        flat, zf = warp.reshape(-1, 3), zbuf.reshape(-1)
        for dy, dx in offs:
            if True:
                m = (np.abs(dx) <= rx) & (np.abs(dy) <= ry)
                tv = tv0[m] + dy
                ok2 = (tv >= 0) & (tv < H)
                idx = tv[ok2] * W + (tu0[m][ok2] + dx) % W
                z_ = zs[m][ok2]
                win = (z_ <= zf[idx]) if mode == "zbuffer" else np.ones(len(z_), bool)
                flat[idx[win]] = cols[m][ok2][win]
                zf[idx[win]] = z_[win]
    if see_through:
        # a surface magnified by the move (the camera walking up to a chair) splats into a sieve, and the far floor
        # behind it shows through its gaps - in the perspective node's dolly / orbit range that stays rare, on a
        # walk through a room it is everywhere near the path. A pixel mostly surrounded (> half of a window as wide
        # as the splat reach) by points >10 % nearer is such a gap: it becomes a hole, the model fills it.
        zf = np.where(np.isfinite(zbuf), zbuf, 1e9)
        warp[_near_share(zf, 2 * rmax + 1, splat) > 0.5] = MAGENTA
    if return_depth:   # the distance each painted pixel got from the target camera (inf in the holes)
        return warp, np.where((warp == MAGENTA).all(-1), np.inf, zbuf)
    return warp


def _near_share(zf, k, splat):
    """Per pixel, the share of its window (rows 2*splat+3, columns k, wrapping) whose depth is >10 % nearer."""
    out = np.zeros(zf.shape, np.float32)
    h = splat + 1
    for dy in range(-h, h + 1):
        for dx in range(-(k // 2), k // 2 + 1):
            out += np.roll(np.roll(zf, dy, 0), dx, 1) < zf / 1.1
    return out / ((2 * h + 1) * k)


def _offsets(rmax):
    """The splat's offset passes, outermost ring first, the centre last. The perspective node (and the training
    builder) runs them in raster order, dy then dx ascending, so the last write to a pixel is the (+r, +r) offset and
    the whole warp lands r px right and down of the geometry (2 px there, a frame at its source camera came out
    shifted 3.2 / 2.0 px here). In one shot that is invisible; along a chain of warps each segment added it again -
    0.7 deg of yaw per segment measured by COLMAP (2026-10-20). Rings to the centre keep every point's own pixel."""
    o = [(dy, dx) for dy in range(-rmax, rmax + 1) for dx in range(-rmax, rmax + 1)]
    o.sort(key=lambda t: (-max(abs(t[0]), abs(t[1])), -(abs(t[0]) + abs(t[1]))))
    return np.array(o, np.int64)


def warp360_path(sources, path, splat=2, mode="training", return_depth=False):
    """sources: [(rgb, dist, pos)], the first the primary; path: target positions. The others fill only the
    primary's magenta. Returns a list of (H, W, 3) uint8 warps (with return_depth: (warp, distance) pairs)."""
    out = []
    for p in path:
        w, z = warp360(*sources[0], p, splat, mode=mode, return_depth=True)
        for rgb, dist, pos in sources[1:]:
            hole = (w == MAGENTA).all(-1)
            if not hole.any():
                break
            w2, z2 = warp360(rgb, dist, pos, p, splat, mode=mode, return_depth=True)
            fill = hole & ~(w2 == MAGENTA).all(-1)
            w[fill] = w2[fill]
            z[fill] = z2[fill]
        out.append((w, z) if return_depth else w)
    return out


_EASE = {"linear": lambda u: u, "in": lambda u: u * u, "out": lambda u: 1.0 - (1.0 - u) ** 2,
         "in_out": lambda u: u * u * (3.0 - 2.0 * u)}


def drop_edges(dist, k, ratio=1.15):
    """Unknown (NaN) for every pixel within k px of a surface more than `ratio` nearer: the soft depth at an object's
    outline gives its edge pixels the background's distance, so they stayed put while the object moved and left a
    ghost outline (a second lamp post, floating leaves) that the model turned into a second object. Dropped, they
    become holes the model fills. The window wraps around horizontally."""
    if k <= 0:
        return dist
    d = torch.from_numpy(np.where(np.isfinite(dist) & (dist > 0), dist, np.inf).astype(np.float32))[None, None]
    d = torch.cat([d[..., -k:], d, d[..., :k]], -1)
    near = -torch.nn.functional.max_pool2d(-d, 2 * k + 1, stride=1, padding=(k, 0))[0, 0].numpy()
    return np.where(dist > near * ratio, np.nan, dist)


def _parse_path(raw, frame_count, interpolation="linear"):
    """Keyframes [{"f": 1-based frame, "x", "y", "z", "ease"?}] -> one position per frame. The curve through the keys is
    linear or smooth (uniform Catmull-Rom); "ease" (linear | in | out | in_out) times the segment that ends at that
    key (the motion into it). Before the first key and after the last the camera holds."""
    kfs = sorted(json.loads(raw) if raw.strip() else [{"f": 1, "x": 0, "y": 0, "z": 0}], key=lambda k: k["f"])
    fs = np.array([k["f"] for k in kfs], np.float64)
    p = np.array([[k.get("x", 0.0), k.get("y", 0.0), k.get("z", 0.0)] for k in kfs], np.float64)
    q = np.concatenate([p[:1], p, p[-1:]])          # end points repeated: the curve starts and ends at a key
    out = []
    for t in np.arange(1, frame_count + 1, dtype=np.float64):
        if len(kfs) == 1 or t <= fs[0]:
            out.append(p[0])
            continue
        if t >= fs[-1]:
            out.append(p[-1])
            continue
        i = int(np.searchsorted(fs, t, side="right")) - 1
        u = _EASE[kfs[i + 1].get("ease", "linear")]((t - fs[i]) / (fs[i + 1] - fs[i]))
        if interpolation == "smooth":
            p0, p1, p2, p3 = q[i], q[i + 1], q[i + 2], q[i + 3]
            out.append(0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                              + (3 * p1 - p0 - 3 * p2 + p3) * u ** 3))
        else:
            out.append(p[i] + (p[i + 1] - p[i]) * u)
    return np.stack(out)


class CrossViewWarp360:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE", {"tooltip": "Equirect source frame(s), 2:1. The first is the primary; the others only "
                                             "fill what it leaves magenta."}),
            "moge_geometry": ("MOGE_GEOMETRY", {"tooltip": "Metric DISTANCE along each ray per source frame "
                                                            "(MoGe panorama inference, or a loaded distance map)."}),
            "frame_count": ("INT", {"default": 39, "min": 1, "max": 1000}),
            "path": ("STRING", {"multiline": True, "default": '[{"f": 1, "x": 0, "y": 0, "z": 0}, {"f": 39, "x": 0, "y": 0, "z": 1.0}]',
                                "tooltip": "Target camera keyframes in metres in the primary source camera's frame (x "
                                           "right, y up, z forward = the image centre); optional \"ease\" per key "
                                           "(linear | in | out | in_out) times the motion into that key."}),
            "source_positions": ("STRING", {"default": "[[0, 0, 0]]",
                                            "tooltip": "Each source frame's camera position (same frame), one per frame."}),
            "distance_scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.001,
                                         "tooltip": "Multiplies every distance (a relative map -> metres)."}),
            "splat": ("INT", {"default": 2, "min": 0, "max": 4,
                              "tooltip": "Splat radius in pixels, as in the perspective node (2 = its training format)."}),
            "interpolation": (["linear", "smooth"], {"default": "linear",
                                                     "tooltip": "The camera curve through the keyframes: straight "
                                                                "segments or a smooth curve (Catmull-Rom)."}),
            "mode": (["training", "zbuffer"], {"default": "training",
                                              "tooltip": "training: the perspective node's splat exactly (the LoRA's "
                                                         "training format); zbuffer: the radius grows with "
                                                         "magnification and the nearest point wins."}),
        }, "optional": {
            "edge_drop": ("INT", {"default": 0, "min": 0, "max": 8,
                                  "tooltip": "Drop the pixels within this many px behind a depth edge (their soft "
                                             "depth left a ghost outline of near objects); 0 = off, 2 = typical."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warp",)
    FUNCTION = "build"
    CATEGORY = "CrossView"

    def build(self, frames, moge_geometry, frame_count, path, source_positions, distance_scale, splat, mode="training",
              interpolation="linear", edge_drop=0):
        rgb = (frames.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        d = moge_geometry["depth"].detach().float().cpu().numpy() * float(distance_scale)
        m = moge_geometry.get("mask")
        if m is not None:
            d = np.where(m.detach().cpu().numpy(), d, np.nan)
        if d.ndim == 2:
            d = d[None]
        H, W = rgb.shape[1:3]
        pos = np.asarray(json.loads(source_positions), np.float64).reshape(-1, 3)
        srcs = []
        for i in range(rgb.shape[0]):
            di = d[min(i, d.shape[0] - 1)]
            if di.shape != (H, W):
                di = torch.nn.functional.interpolate(torch.from_numpy(di)[None, None], size=(H, W),
                                                     mode="nearest")[0, 0].numpy()
            srcs.append((rgb[i], drop_edges(di, int(edge_drop)), pos[min(i, len(pos) - 1)]))
        warps = warp360_path(srcs, _parse_path(path, frame_count, interpolation), int(splat), mode)
        return (torch.from_numpy(np.stack(warps).astype(np.float32) / 255.0),)


_ID = f"CrossViewWarp360{NODE_SUFFIX}"
NODE_CLASS_MAPPINGS = {_ID: CrossViewWarp360}
NODE_DISPLAY_NAME_MAPPINGS = {_ID: "CrossView Warp 360 (equirect)" + (f" [{NODE_SUFFIX}]" if NODE_SUFFIX else "")}
