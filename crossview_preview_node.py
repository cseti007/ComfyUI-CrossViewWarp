"""Live viewfinder for the CrossView Warp node.

Registers no node of its own: it hangs a cache off the end of build() and serves
a route that re-warps frames from it, so the widget can orbit the camera and
scrub the clip without re-running the graph. A node's inputs only exist inside
its execute call, and ComfyUI has no API for reading them afterwards - hence the
cache. Run once, scrub afterwards.

Two details keep the preview equal to the real run rather than merely close:

  * build() derives the pivot from frame 0 of whatever batch it gets, so a
    request renders [frame 0, frame i] and keeps warp[1]. Frame i alone
    measured up to 14.7/255 of error.
  * _depth_to_z normalises against the whole stack's 1st/99th percentiles, which
    two frames cannot reproduce, so they are taken once over the full clip and
    the resulting z goes in through build()'s metric-geometry path.

The camera maths is never reimplemented here; fx follows the frame width, so a
downscaled preview is geometrically identical rather than approximate.
"""

import asyncio
import io
import json
import logging
import re
import threading
from collections import OrderedDict

import numpy as np
import torch

from . import crossview_warp_node as cvw
from .crossview_warp_node import NODE_SUFFIX

# Roughly 1.8 MB per frame at the default preview size.
_CACHE = OrderedDict()
_CACHE_MAX = 2
_CACHE_LOCK = threading.Lock()

# Renders are serialised: they are short, the browser only ever has one request
# in flight, and the lock also scopes the ProgressBar swap in _render_sync.
_RENDER_LOCK = threading.Lock()

# Half-width of the pivot depth-plane highlight, as a fraction of pivot z.
_PIVOT_BAND = 0.06
_ISO_COLOUR = np.array([80.0, 220.0, 200.0], dtype=np.float32)

# build() logs once per run, which a drag turns into twenty lines a second.
# Thread-local, so a concurrent graph execution keeps its logging.
_quiet = threading.local()


class _QuietDuringPreview(logging.Filter):
    def filter(self, record):
        return not getattr(_quiet, "on", False)


# Guarded because two installed copies of this package would otherwise each add
# their own (harmless, but there is no reason to stack them).
if not getattr(logging.root, "_crossview_preview_filter", False):
    logging.root.addFilter(_QuietDuringPreview())
    logging.root._crossview_preview_filter = True

# Build kwargs the browser may set, with fallbacks. Keyframes and camera_info
# are handled separately: one needs a timeline, the other is a link.
_PARAMS = {
    "azimuth": (float, -30.0),
    "elevation": (float, 20.0),
    "distance": (float, 1.0),
    "hfov": (float, 50.0),
    "vertical_shift": (float, 0.0),
    "depth_ratio": (float, 6.0),
    "smooth_depth": (bool, False),
    "invert_depth": (bool, False),
    "pivot_override": (bool, True),
    "pivot_x": (float, 0.0),
    "pivot_y": (float, 0.0),
    "pivot_z": (float, 1.05),
    "keep_source_aim": (bool, False),
}


def _fit(t, size, mode="bilinear"):
    """[B,H,W,C] float -> longest side at most `size`, aspect preserved."""
    h, w = t.shape[1], t.shape[2]
    s = size / float(max(h, w))
    if s >= 1.0:
        return t
    nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    x = t.permute(0, 3, 1, 2)
    kw = {"align_corners": False} if mode == "bilinear" else {}
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode=mode, **kw)
    return x.permute(0, 2, 3, 1).contiguous()


def _smooth_inplace(d32, rgb):
    """build()'s edge-aware depth smoothing, on the frames given.

    Mirrors the loop in build() exactly, including the bilateral fallback when
    opencv-contrib is missing. Applied per frame there, so applying it to a
    subset here gives identical values.
    """
    import cv2
    for i in range(d32.shape[0]):
        x = cv2.medianBlur(d32[i].astype(np.float32), 3)
        try:
            x = cv2.ximgproc.guidedFilter(rgb[i], x, radius=8, eps=1e-3)
        except Exception:
            x = cv2.bilateralFilter(x, 9, 0.1, 9.0)
        d32[i] = x
    return d32


def _lohi(d32, invert):
    """The percentile pair _depth_to_z would derive from this whole stack."""
    d = d32.astype(np.float64)
    if invert:
        d = -d
    return float(np.percentile(d, 1)), float(np.percentile(d, 99))


def _z_from(d32, invert, ratio, lo, hi):
    """_depth_to_z's body with the percentiles supplied instead of measured.

    Same arithmetic, same order, same float64 - the only difference is that lo
    and hi come from the full clip rather than from the frames in hand, which
    is the whole point: they are a property of the clip, not of the request.
    """
    d = d32.astype(np.float64)
    if invert:
        d = -d
    dn = np.clip((d - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    r = max(float(ratio), 1.01)
    return 1.0 / (1.0 / r + (1.0 - 1.0 / r) * dn)


def _coerce(params):
    out = {}
    for k, (typ, dflt) in _PARAMS.items():
        v = params.get(k, dflt)
        try:
            out[k] = bool(v) if typ is bool else float(v)
        except (TypeError, ValueError):
            out[k] = dflt
    return out


class _Capture:
    """Record the pose build() actually used, by intercepting its own calls.

    The pivot marker needs the pivot and the target camera, and the pivot may be
    estimated from the frame rather than given. Copying that derivation here
    would be a second implementation that drifts, so the two functions are
    wrapped for the duration of the call and asked what they were handed.

    The wrappers are transparent - they record and delegate, changing nothing -
    so a real graph execution running concurrently in another thread is
    unaffected beyond overwriting a dict this request then ignores.
    """

    def __enter__(self):
        self.data = {}
        self._orbit, self._warp = cvw._orbit_C_tgt, cvw._warp_frame

        def orbit(az, el, dist, pivot, aim=None):
            self.data["pivot"] = np.asarray(pivot, dtype=np.float64).copy()
            return self._orbit(az, el, dist, pivot, aim)

        def warp_frame(rgb_ref, depth_ref, C_ref, C_tgt, fx_pix, splat, cx, cy):
            self.data.update(C_tgt=np.array(C_tgt, dtype=np.float64), fx=float(fx_pix),
                             cx=float(cx), cy=float(cy))
            return self._warp(rgb_ref, depth_ref, C_ref, C_tgt, fx_pix, splat, cx, cy)

        cvw._orbit_C_tgt, cvw._warp_frame = orbit, warp_frame
        return self.data

    def __exit__(self, *exc):
        cvw._orbit_C_tgt, cvw._warp_frame = self._orbit, self._warp
        return False


def _pivot_info(kw, cap):
    """Where the pivot lands in the warped view, for the crosshair."""
    pivot = cap.get("pivot")
    if pivot is None:                       # camera_info: _orbit_C_tgt never ran
        pivot = np.array([kw["pivot_x"], kw["pivot_y"], kw["pivot_z"]], dtype=np.float64)
    out = {"auto": not bool(kw["pivot_override"])}
    C = cap.get("C_tgt")
    if C is None:
        return out
    xd = np.linalg.inv(C)[:3, :3] @ pivot + np.linalg.inv(C)[:3, 3]
    if xd[2] > 1e-6:
        out["u"] = float(xd[0] / xd[2] * cap["fx"] + cap["cx"])
        out["v"] = float(xd[1] / xd[2] * cap["fx"] + cap["cy"])
    return out


def _iso_tint(rgb_i, zi, pivot_z):
    """Tint the pixels lying in the pivot's depth plane.

    This is the question a marker cannot answer: not where the pivot is on
    screen, but WHAT is at that depth - the subject, or the wall behind it.
    """
    img = rgb_i.copy()
    if zi is None or not np.isfinite(pivot_z) or pivot_z <= 0:
        return img
    m = np.abs(zi - pivot_z) <= _PIVOT_BAND * pivot_z
    if m.any():
        img[m] = (0.45 * _ISO_COLOUR + 0.55 * img[m].astype(np.float32)).astype(np.uint8)
    return img


def _adopt(kw, sample):
    """Copy a sampled pose into the build kwargs.

    The pivot too: it is part of the pose now, and the pivot readout and the
    depth-plane overlay both read it from here.
    """
    for dst, src in (("azimuth", "az"), ("elevation", "el"), ("distance", "dist"),
                     ("vertical_shift", "vs"), ("pivot_x", "px"), ("pivot_y", "py"),
                     ("pivot_z", "pz")):
        if src in sample:
            kw[dst] = sample[src]


def _render_array(entry, frame, params, iso=False):
    """Render one frame of the cached clip -> (uint8 HxWx3, frame no, pivot info).

    Split out from _render_sync so the fidelity test can compare raw pixels:
    through JPEG a single displaced pixel rewrites its whole 8x8 block, which
    turns a handful of differences into tens of thousands of changed bytes.
    """
    kw = _coerce(params)
    n = entry["n"]
    i = int(np.clip(int(frame) - 1, 0, n - 1))

    # build() interpolates by buffer index and rejects frames past the batch, so
    # the real path cannot go to a 2-frame request. Sample it here instead and
    # hand over a synthetic path: f=1 the move's midpoint, f=2 the playhead.
    # build() reads its base pose at (B+1)//2 = 1, the true midpoint, and renders
    # the kept frame at the playhead pose. Two keyframes sampled at both ends
    # make easing and the spline no-ops.
    kpath = None
    if params.get("use_keyframes"):
        kfs = cvw._parse_keyframes(params.get("keyframes", ""), n, kw["vertical_shift"],
                                   (kw["pivot_x"], kw["pivot_y"], kw["pivot_z"]))
        if len(kfs) >= 2:
            path = cvw._prepare_path(kfs)
            motion = str(params.get("interp_motion") or "linear")
            smooth_path = str(params.get("interpolation")) == "smooth"
            mid = cvw._sample_path(path, (n + 1) // 2, motion, smooth_path)
            cur = cvw._sample_path(path, i + 1, motion, smooth_path)
            kpath = json.dumps([{"f": 1, **mid}, {"f": 2, **cur}])
            # the readout should report the pose actually rendered, not the
            # static widgets the path has overridden
            _adopt(kw, cur)
        elif len(kfs) == 1:
            # build() treats a lone keyframe as the static pose; matching that
            # here keeps the preview from showing the widgets instead.
            _adopt(kw, kfs[0])

    # build() estimates the pivot from the batch's FIRST frame, so frame 0 rides
    # along. A keyframed request always needs two, even at frame 1, to carry the
    # midpoint pose.
    idx = [0] if (i == 0 and kpath is None) else [0, i]

    rgb = entry["rgb"][idx]                                  # uint8 [k,H,W,3]
    frames = torch.from_numpy(rgb.astype(np.float32) / 255.0)

    if entry["moge_z"] is not None:
        # Metric geometry is absolute, so a subset needs no clip-wide fixing up.
        moge = {"depth": torch.from_numpy(entry["moge_z"][idx])}
        if entry["moge_mask"] is not None:
            moge["mask"] = torch.from_numpy(entry["moge_mask"][idx])
        if entry["moge_K"] is not None:
            moge["intrinsics"] = entry["moge_K"]
    else:
        smooth = bool(kw["smooth_depth"])
        invert = bool(kw["invert_depth"])
        d32 = entry["depth"][idx].copy()
        if smooth:
            _smooth_inplace(d32, rgb)
        lo, hi = entry["lohi"][(smooth, invert)]
        z = _z_from(d32, invert, kw["depth_ratio"], lo, hi)
        # As metric geometry, so build() takes z as given. Without `intrinsics`
        # hfov=0 falls back to 50, the same as the depth path.
        moge = {"depth": torch.from_numpy(z)}

    zb = entry["moge_z"][idx] if entry["moge_z"] is not None else z

    if iso:
        # Tinted into the source pixels BEFORE warping, so the highlight lands
        # where those surfaces end up. Cosmetic only: z is already computed and
        # build() skips smoothing on the metric path, so rgb feeds nothing but
        # the warp colours. The preview is an annotated warp while this is on.
        pz = kw["pivot_z"]
        if not kw["pivot_override"]:
            # the estimate only exists inside build(), so it costs one extra pass
            with _RENDER_LOCK, _Capture() as cap0:
                saved0 = cvw.ProgressBar
                cvw.ProgressBar = None
                _quiet.on = True
                try:
                    cvw.CrossViewWarp().build(frames=frames, depth=None,
                                              moge_geometry=moge, **kw)
                finally:
                    cvw.ProgressBar = saved0
                    _quiet.on = False
            if cap0.get("pivot") is not None:
                pz = float(cap0["pivot"][2])
        tinted = rgb.copy()
        for k in range(tinted.shape[0]):
            tinted[k] = _iso_tint(tinted[k], zb[k], pz)
        frames = torch.from_numpy(tinted.astype(np.float32) / 255.0)

    cap = {}
    with _RENDER_LOCK, _Capture() as cap:
        # build() drives the node progress bar, which has no node to attribute
        # itself to out here and would spam the websocket at drag rates. A real
        # run starting inside this window loses its bar for that run only.
        saved = cvw.ProgressBar
        cvw.ProgressBar = None
        _quiet.on = True
        bkw = dict(kw)
        # Replayed from the cache. With it connected build() takes the pose from
        # there and ignores the angles, the pivot and the path.
        if entry["cam"] is not None:
            bkw["camera_info"] = entry["cam"]
        if kpath is not None:
            bkw.update(use_keyframes=True, keyframes=kpath, frame_count=0,
                       interp_motion="linear", interpolation="linear")
        try:
            warp, _orbit = cvw.CrossViewWarp().build(
                frames=frames, depth=None, moge_geometry=moge, **bkw)
        finally:
            cvw.ProgressBar = saved
            _quiet.on = False
    arr = (warp[-1].clamp(0, 1).numpy() * 255.0).astype(np.uint8)

    info = _pivot_info(kw, cap)
    # During a keyframed move the static widgets no longer describe what is on
    # screen, so the pose that was actually rendered travels back with it.
    info["pose"] = {"az": kw["azimuth"], "el": kw["elevation"], "dist": kw["distance"],
                    "vs": kw["vertical_shift"], "px": kw["pivot_x"], "py": kw["pivot_y"],
                    "pz": kw["pivot_z"]}
    info["iso"] = bool(iso)
    return arr, i + 1, info


def _render_sync(entry, frame, params, iso=False):
    """As _render_array, JPEG-encoded for the wire."""
    arr, no, info = _render_array(entry, frame, params, iso)
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=88)
    return buf.getvalue(), no, info


def seed_cache(unique_id, frames, depth=None, moge_geometry=None, camera_info=None,
               preview_size=384):
    """Keep a downscaled copy of the clip so the browser can re-warp it.

    Called from the end of CrossViewWarp.build() through the _PREVIEW_SINK slot,
    so the preview is a layer over the node rather than something the node has
    to know about. Only the inputs are cached - every pose parameter arrives per
    request from the widget.
    """
    if depth is None and moge_geometry is None:
        return
    size = int(preview_size)
    small = _fit(frames.detach().float().cpu(), size)
    # build() converts frames to uint8 anyway and the round trip through /255 is
    # exact, so this costs no fidelity and a quarter of the memory.
    rgb = (small.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    n, h, w = rgb.shape[:3]

    entry = {"rgb": rgb, "n": n, "depth": None, "lohi": {},
             "moge_z": None, "moge_mask": None, "moge_K": None,
             # a camera_info is a plain dict of numbers, so unlike the other
             # links it can simply be kept and replayed
             "cam": camera_info}

    if moge_geometry is not None:
        md = moge_geometry.get("depth")
        z = _fit(md[..., None].detach().float().cpu(), size).squeeze(-1).numpy()
        mm = moge_geometry.get("mask")
        if mm is not None:
            m = _fit(mm[..., None].detach().float().cpu(), size, "nearest").squeeze(-1)
            entry["moge_mask"] = (m > 0.5).numpy()
        entry["moge_z"] = z
        # normalised by width, so it survives the resize untouched
        entry["moge_K"] = moge_geometry.get("intrinsics")
    else:
        # Exactly build()'s depth_bhw, at preview scale: clamp and channel mean
        # AFTER the resize, in that order, or the values drift.
        d32 = _fit(depth.detach().float().cpu(), size).clamp(0, 1).mean(dim=-1).numpy()
        entry["depth"] = d32
        # The percentiles _depth_to_z would measure over the whole clip, per
        # switch combination. Smoothing is per frame, but its percentiles still
        # have to come from a smoothed full clip.
        for sm in (False, True):
            try:
                ds = _smooth_inplace(d32.copy(), rgb) if sm else d32
            except ImportError:
                continue     # no cv2: a real run with smooth_depth would fail too
            for inv in (False, True):
                entry["lohi"][(sm, inv)] = _lohi(ds, inv)

    entry["mb"] = round((rgb.nbytes
                         + (entry["depth"].nbytes if entry["depth"] is not None else 0)
                         + (entry["moge_z"].nbytes if entry["moge_z"] is not None else 0)) / 1e6, 1)
    key = str(unique_id)
    with _CACHE_LOCK:
        _CACHE[key] = entry
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    logging.info("CrossViewWarp: cached %d preview frames at %dx%d (%.1f MB) for node %s",
                 n, w, h, entry["mb"], key)
    return {"of": n, "w": w, "h": h, "mb": entry["mb"],
            "source": "moge" if entry["moge_z"] is not None else "depth"}


# URL-safe form of the local dev suffix, so two installed copies cannot fight
# over one route path (see .node_suffix in crossview_warp_node.py).
_SLUG = re.sub(r"[^A-Za-z0-9_-]", "", NODE_SUFFIX)


def _register_routes():
    """Add the render endpoint. Never allowed to break ComfyUI startup."""
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return  # standalone import (tests, non-ComfyUI use): no server to serve from
    srv = getattr(PromptServer, "instance", None)
    if srv is None:
        return
    # A duplicate aiohttp path raises at startup, and PromptServer is the one
    # object both installed copies share to flag it.
    flag = f"_crossview_preview_route{_SLUG}"
    if getattr(srv, flag, False):
        return
    setattr(srv, flag, True)

    @srv.routes.post(f"/crossview_preview{_SLUG}/render")
    async def _render(request):
        try:
            data = await request.json()
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"error": "bad request body"}, status=400)
        key = str(data.get("node_id"))
        with _CACHE_LOCK:
            entry = _CACHE.get(key)
        if entry is None:
            # 409, not 404: the endpoint exists, the data does not YET - the
            # browser turns this into "run the graph once", not into an error.
            return web.json_response({"error": "no cached clip - run the graph once"}, status=409)
        iso = bool(data.get("iso"))
        loop = asyncio.get_running_loop()
        try:
            # Off the event loop: a ~50 ms render inline would stall every other
            # ComfyUI request for as long as the user keeps dragging.
            jpeg, frame, pivot = await loop.run_in_executor(
                None, _render_sync, entry, data.get("frame", 1),
                data.get("params") or {}, iso)
        except Exception as e:
            logging.exception("CrossView Preview: render failed")
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
        # In a header, not drawn into the JPEG: the marker can be restyled
        # without a re-render and the pixels stay the plain warp.
        info = json.dumps({"frame": frame, "count": entry["n"], "pivot": pivot})
        return web.Response(body=jpeg, content_type="image/jpeg",
                            headers={"Cache-Control": "no-store",
                                     "X-Preview-Frame": str(frame),
                                     "X-Preview-Count": str(entry["n"]),
                                     "X-Preview-Info": info})


try:
    _register_routes()
except Exception:
    logging.exception("CrossView Preview: could not register the render route; "
                      "the node still loads, the live preview will not work")


# Hand the cache to the node. Done here, not by an import in the warp module, so
# the dependency stays one-way and that module still runs on its own.
cvw._PREVIEW_SINK = seed_cache
