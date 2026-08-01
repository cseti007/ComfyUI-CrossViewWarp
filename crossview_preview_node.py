"""CrossView Warp with a live viewfinder.

This node produces the same `warp` and `orbit_view` outputs as the plain warp
node - it calls the same build() - and on top of that caches the clip so the
browser can re-warp it interactively. One node instead of two: the pose dialled
in on the widget is the pose that reaches the IC-LoRA guide, with no numbers
copied across. Keyframed moves are authored on the playhead - scrub, aim, press
KEY - which needs none of the sphere's even-spreading machinery, because the
playhead already knows which frame was meant. camera_info stays on the plain node.

The warp is fast enough to be interactive - a single frame at 384 px costs
~29 ms through the full unmodified build() - so the only real obstacle to a
draggable preview is DATA: a node's inputs exist solely inside its execute
call, and ComfyUI has no API for "hand me node X's current input" without
running the graph. So this node caches the clip when the graph runs, and an
HTTP route re-warps frames from that cache while the browser drags the camera
and scrubs the playhead. Run once, scrub afterwards.

Two things make the preview agree with the real run rather than merely
resemble it:

  * build() derives the pivot, the roll-lock correction and the subject cloud
    from frame 0 of whatever batch it is handed, so a request renders the
    2-frame batch [frame 0, frame i] and keeps warp[1]. Handing it frame i
    alone silently re-estimates all three from the wrong frame - measured at up
    to 14.7/255 mean error with an automatic pivot.
  * _depth_to_z normalises against the 1st/99th percentile of the WHOLE stack,
    which a 2-frame batch cannot reproduce. The percentiles are therefore taken
    once over the full clip at cache time and applied here, with the resulting
    z handed to build() through its metric-geometry path (which uses z as given
    instead of re-normalising it).

The camera maths itself is never reimplemented: a preview whose only job is to
predict the real output must not become a second, separately-drifting copy of
it. Since fx is derived from the frame width, downscaling also scales the
focal, which makes a small preview geometrically identical to the full-size
warp rather than approximately so.
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

# Whole clips now, not single frames - roughly 1 MB per frame at the default
# size, so a couple of entries is already a lot of memory and more would be
# hoarding. The cached size is logged and shown in the widget.
_CACHE = OrderedDict()
_CACHE_MAX = 2
_CACHE_LOCK = threading.Lock()

# Renders are serialised: they are short, the browser only ever has one request
# in flight, and the lock also scopes the ProgressBar swap in _render_sync.
_RENDER_LOCK = threading.Lock()

# Half-width of the "pivot depth plane" highlight, as a fraction of the pivot's
# own depth. Wide enough to catch a whole subject, narrow enough that it does
# not swallow the background.
_PIVOT_BAND = 0.06
_ISO_COLOUR = np.array([80.0, 220.0, 200.0], dtype=np.float32)

# build() logs things worth saying once per run - which becomes a flood when a
# drag replays that same call twenty times a second. The flag is THREAD-LOCAL,
# so a real graph execution running at the same time keeps every line.
_quiet = threading.local()


class _QuietDuringPreview(logging.Filter):
    def filter(self, record):
        return not getattr(_quiet, "on", False)


# Guarded because two installed copies of this package would otherwise each add
# their own (harmless, but there is no reason to stack them).
if not getattr(logging.root, "_crossview_preview_filter", False):
    logging.root.addFilter(_QuietDuringPreview())
    logging.root._crossview_preview_filter = True

# Scrubbable parameters, with the defaults used when the browser omits one.
# Keyframes are excluded on purpose - this previews one pose at a time, and a
# path needs a timeline to be worth previewing. camera_info likewise: it is a
# link, so it has no value outside execution.
_PARAMS = {
    "azimuth": (float, -30.0),
    "elevation": (float, 20.0),
    "distance": (float, 1.0),
    "hfov": (float, 50.0),
    "vertical_shift": (float, 0.0),
    "depth_ratio": (float, 6.0),
    "smooth_depth": (bool, False),
    "invert_depth": (bool, False),
    "roll_lock": (bool, True),
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

    The pivot marker needs the pivot and the final target camera - and both are
    build() internals: the pivot may be estimated from the frame rather than
    given, and roll-lock rotates C_tgt after _orbit_C_tgt returns it. Copying
    that derivation here would be a second implementation that drifts, so
    instead the two functions are wrapped for the duration of the call and asked
    what they were handed.

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


def _pivot_info(entry, zi, kw, cap):
    """Where the pivot is, in both views, and how it sits against the scene.

    The image position is only half the story and usually the boring half: the
    camera looks AT the pivot, so in the warped view it lands dead centre unless
    keep_source_aim is on with an off-axis pivot. The useful part is the depth
    comparison - whether the point you are orbiting sits on the subject or
    floats in front of or behind it - and the orbit radius it implies.
    """
    h, w = entry["rgb"].shape[1:3]
    pivot = cap.get("pivot")
    if pivot is None:                       # source view never runs build()
        pivot = np.array([kw["pivot_x"], kw["pivot_y"], kw["pivot_z"]], dtype=np.float64)
    fx = cap.get("fx")
    if fx is None:
        deg = kw["hfov"] if kw["hfov"] > 0 else 50.0
        fx = w / (2.0 * np.tan(np.radians(deg) / 2.0))

    out = {"auto": not bool(kw["pivot_override"]),
           "z": float(pivot[2]),
           # eye = pivot + dist * (R @ -pivot), so the arc the camera sweeps has
           # radius dist*|pivot| - which is why pushing pivot_z back also widens
           # the orbit instead of only moving its centre.
           "radius": float(np.linalg.norm(pivot)) * float(kw["distance"])}

    # Source view: _warp_frame unprojects about the image centre, so that is the
    # principal point the pivot has to be projected back through.
    if pivot[2] > 1e-6:
        su = float(pivot[0] / pivot[2] * fx + w / 2.0)
        sv = float(pivot[1] / pivot[2] * fx + h / 2.0)
        out["src_u"], out["src_v"] = su, sv
        iu, iv = int(round(su)), int(round(sv))
        if 0 <= iu < w and 0 <= iv < h and zi is not None:
            sz = float(zi[iv, iu])
            if np.isfinite(sz):
                out["scene_z"] = sz

    C = cap.get("C_tgt")
    if C is not None:
        Ci = np.linalg.inv(C)
        xd = Ci[:3, :3] @ pivot + Ci[:3, 3]
        if xd[2] > 1e-6:
            out["u"] = float(xd[0] / xd[2] * fx + cap["cx"])
            out["v"] = float(xd[1] / xd[2] * fx + cap["cy"])
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


def _render_array(entry, frame, params, mode="warp"):
    """Render one frame of the cached clip -> (uint8 HxWx3, frame no, pivot info).

    Split out from _render_sync so the fidelity test can compare raw pixels:
    through JPEG a single displaced pixel rewrites its whole 8x8 block, which
    turns a handful of differences into tens of thousands of changed bytes.
    """
    kw = _coerce(params)
    n = entry["n"]
    i = int(np.clip(int(frame) - 1, 0, n - 1))

    # A keyframed move cannot simply be forwarded: build() interpolates by
    # BUFFER INDEX (_sample_path(path, i + 1) over the batch it was handed), and
    # _parse_keyframes rejects any frame number past that batch - so a keyframe
    # at frame 37 would raise on a 2-frame request. The path is therefore
    # sampled here with the node's own _sample_path and handed over as a
    # SYNTHETIC 2-point path: f=1 carrying the move's midpoint pose, f=2 the
    # playhead's. build() then estimates roll-lock at (B+1)//2 = 1, i.e. at the
    # true midpoint exactly as the full run does, and the frame that is kept
    # renders at the playhead pose. Frame 1's warp is discarded, and the pivot,
    # roll and subject-cloud estimates come from its depth and image rather than
    # its pose, so nothing else moves. With only two keyframes, sampled exactly
    # at both ends, easing and the spline are no-ops - hence linear/linear.
    kpath = None
    if params.get("use_keyframes"):
        kfs = cvw._parse_keyframes(params.get("keyframes", ""), n, kw["vertical_shift"])
        if len(kfs) >= 2:
            path = cvw._prepare_path(kfs)
            motion = str(params.get("interp_motion") or "linear")
            smooth_path = str(params.get("interpolation")) == "smooth"
            mid = cvw._sample_path(path, (n + 1) // 2, motion, smooth_path)
            cur = cvw._sample_path(path, i + 1, motion, smooth_path)
            kpath = json.dumps([
                {"f": 1, "az": mid[0], "el": mid[1], "dist": mid[2], "vs": mid[3]},
                {"f": 2, "az": cur[0], "el": cur[1], "dist": cur[2], "vs": cur[3]}])
            # the readout should report the pose actually rendered, not the
            # static widgets the path has overridden
            (kw["azimuth"], kw["elevation"], kw["distance"], kw["vertical_shift"]) = cur
        elif len(kfs) == 1:
            # build() treats a lone keyframe as the static pose; matching that
            # here keeps the preview from showing the widgets instead.
            (kw["azimuth"], kw["elevation"], kw["distance"],
             kw["vertical_shift"]) = kfs[0][1], kfs[0][2], kfs[0][3], kfs[0][4]

    # Frame 0 rides along because build() estimates the pivot, the roll-lock
    # correction and the subject cloud from the batch's FIRST frame. Dropping it
    # would re-estimate all three from frame i and quietly disagree with the run.
    # A keyframed request always needs two, even at the playhead's frame 1, so
    # the midpoint pose has a slot to sit in.
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
        # Handed over as metric geometry so build() takes z as given. Without
        # `intrinsics`, hfov=0 falls back to 50 degrees there exactly as it does
        # on the depth path with nothing else connected.
        moge = {"depth": torch.from_numpy(z)}

    # Depth for the requested frame: the LAST of the batch, matching warp[-1].
    zi = entry["moge_z"][i] if entry["moge_z"] is not None else z[-1]

    # The source view needs no warp. It still needs build() when the pivot is
    # estimated rather than given, because that estimate only exists in there.
    arr, cap = None, {}
    if mode != "source" or not kw["pivot_override"]:
        with _RENDER_LOCK, _Capture() as cap:
            # build() drives the node progress bar, which outside an execution
            # has no node to attribute itself to - and at drag rates it would
            # spam the websocket. Swapped rather than parameterised so the
            # released node needs no edit; the cost is that a real run starting
            # inside this window loses its progress bar for that one run, which
            # self-heals on the next.
            saved = cvw.ProgressBar
            cvw.ProgressBar = None
            _quiet.on = True
            bkw = dict(kw)
            if kpath is not None:
                bkw.update(use_keyframes=True, keyframes=kpath, frame_count=0,
                           interp_motion="linear", interpolation="linear")
            try:
                warp, _orbit = cvw.CrossViewWarp().build(
                    frames=frames, depth=None, moge_geometry=moge, **bkw)
            finally:
                cvw.ProgressBar = saved
                _quiet.on = False
        if mode != "source":
            arr = (warp[-1].clamp(0, 1).numpy() * 255.0).astype(np.uint8)

    info = _pivot_info(entry, zi, kw, cap)
    # During a keyframed move the static widgets no longer describe what is on
    # screen, so the pose that was actually rendered travels back with it.
    info["pose"] = {"az": kw["azimuth"], "el": kw["elevation"], "dist": kw["distance"],
                    "vs": kw["vertical_shift"]}
    info["keyed"] = kpath is not None
    if arr is None:
        arr = _iso_tint(entry["rgb"][i], zi, info["z"])
    return arr, i + 1, info


def _render_sync(entry, frame, params, mode="warp"):
    """As _render_array, JPEG-encoded for the wire."""
    arr, no, info = _render_array(entry, frame, params, mode)
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=88)
    return buf.getvalue(), no, info


class CrossViewPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames - the same ones you feed the CrossView Warp "
                    "node. The whole clip is cached, downscaled, so you can scrub it."}),
                "preview_size": ("INT", {"default": 384, "min": 128, "max": 768, "step": 32,
                    "tooltip": "Longest side of the preview, in pixels. The warp is scale-"
                    "invariant, so a small preview shows the same geometry as the full-size run "
                    "- this trades sharpness and memory for frame rate. Changing it needs a "
                    "re-run, since it is what the clip is cached at."}),
                "frame_index": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "Playhead, counted from 1. Scrub the bar under the preview or "
                    "press play; this widget follows along and can also be typed into. Changing "
                    "it does NOT need a re-run."}),
                "azimuth": ("FLOAT", {"default": -30.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Horizontal orbit angle (deg). Drag the preview left/right to set "
                    "it. Negative = camera orbits LEFT."}),
                "elevation": ("FLOAT", {"default": 20.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Vertical orbit angle (deg). Drag the preview up/down to set it. "
                    "Positive = camera rises."}),
                "distance": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Camera distance (1.0 = same as source). Mouse wheel over the "
                    "preview."}),
                "hfov": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Assumed horizontal field of view (deg). Same meaning as on the "
                    "warp node; 0 reads it from moge_geometry."}),
                "depth_ratio": ("FLOAT", {"default": 6.0, "min": 1.5, "max": 1000.0, "step": 0.5,
                    "tooltip": "Max far/near depth ratio. Ignored with moge_geometry connected. "
                    "One of the settings this preview is most useful for - scrub it and watch "
                    "the relief change."}),
            },
            "optional": {
                "vertical_shift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.02, "tooltip": "Vertical lens shift as a fraction of image height: slides the rendered frame, positive moves the picture DOWN. Not a camera move, so it cannot change parallax."}),
                "smooth_depth": ("BOOLEAN", {"default": False,
                    "tooltip": "Edge-aware depth smoothing before warping."}),
                "invert_depth": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip depth polarity. Leave FALSE for the standard DA-V2 node."}),
                "roll_lock": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep the subject's in-image lean the same as the source."}),
                "pivot_override": ("BOOLEAN", {"default": True,
                    "tooltip": "Orbit around the explicit pivot below instead of an estimated "
                    "one."}),
                "pivot_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Pivot X (+ = right), in depth units."}),
                "pivot_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Pivot Y (+ = down), in depth units."}),
                "pivot_z": ("FLOAT", {"default": 1.05, "min": 0.01, "max": 1000.0, "step": 0.01,
                    "tooltip": "Pivot depth. Note that the orbit radius is the pivot's distance "
                    "from the camera, so raising this widens the arc as well as moving it back."}),
                "keep_source_aim": ("BOOLEAN", {"default": False,
                    "tooltip": "Keep the source camera's aim while orbiting instead of turning to "
                    "face the pivot. Only bites with an off-axis pivot."}),
                "use_keyframes": ("BOOLEAN", {"default": False,
                    "tooltip": "Animate the camera along the 'keyframes' path instead of holding "
                    "one pose. Turned on for you once a second keyframe exists."}),
                "keyframes": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Camera path as JSON - you normally never type in here. Scrub to a "
                    "frame, aim the camera by dragging the preview, and press the KEY button under "
                    "it to drop a keyframe at that exact frame; press it again on the same frame "
                    "to remove it. Unlike the orbit sphere, nothing has to be spread out "
                    "afterwards, because the playhead already knows which frame you meant. "
                    'Example: [{"f":1,"az":-30,"el":20,"dist":1.0,"vs":0.0},{"f":49,"az":45,'
                    '"el":10,"dist":1.2,"vs":0.15}] . "vs" is the vertical lens shift and is '
                    "OPTIONAL - a keyframe without one inherits the static vertical_shift widget, "
                    "so paths written before it existed behave exactly as they did. Before the "
                    "first and after the last keyframe the pose is held."}),
                "interp_motion": (["linear", "ease_in_out", "ease_in", "ease_out"],
                                  {"default": "linear",
                    "tooltip": "Timing between consecutive keyframes. Applied per segment, so "
                    "ease_in_out settles the camera into every keyframe. Pair easing with "
                    "interpolation='linear'; with 'smooth' it cancels the continuity the spline "
                    "is there to provide."}),
                "interpolation": (["linear", "smooth"], {"default": "linear",
                    "tooltip": "Shape of the path through the keyframes. linear = straight legs "
                    "with a corner at each one; smooth = Catmull-Rom spline gliding through them. "
                    "With only two keyframes the two are identical."}),
                "depth": ("IMAGE", {
                    "tooltip": "Depth map for the same frames, from Depth Anything V2."}),
                "moge_geometry": ("MOGE_GEOMETRY", {
                    "tooltip": "Metric geometry from Run MoGe Inference, used INSTEAD of the "
                    "depth image."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("warp", "orbit_view")
    OUTPUT_TOOLTIPS = (
        "The warp control video, identical to the CrossView Warp node's - connect to the "
        "IC-LoRA reference guide for the WARP stream.",
        "Front view of the orbit globe with the camera marker at the requested az/el/dist.",
    )
    FUNCTION = "seed"
    CATEGORY = "CrossView"
    # Both an output node AND a source of outputs: OUTPUT_NODE keeps it running
    # when nothing downstream needs it, which is what a session spent only
    # scrubbing the preview looks like.
    OUTPUT_NODE = True
    DESCRIPTION = ("CrossView Warp with a live viewfinder. Produces the same warp and orbit_view "
                   "outputs as the plain node, and additionally caches the clip so you can drag "
                   "the preview to orbit the camera and scrub or play the video underneath it. "
                   "Does not do keyframed camera moves or camera_info - use the plain node for "
                   "those.")

    def seed(self, frames, preview_size, frame_index, azimuth, elevation, distance, hfov,
             depth_ratio, vertical_shift=0.0, smooth_depth=False, invert_depth=False, roll_lock=True,
             pivot_override=True, pivot_x=0.0, pivot_y=0.0, pivot_z=1.05, keep_source_aim=False,
             use_keyframes=False, keyframes="", interp_motion="linear",
             interpolation="linear", depth=None, moge_geometry=None, unique_id=None):
        if depth is None and moge_geometry is None:
            raise ValueError("CrossView Preview: connect either `depth` (a depth image) or "
                             "`moge_geometry` (metric geometry from Run MoGe Inference).")
        size = int(preview_size)
        small = _fit(frames.detach().float().cpu(), size)
        # uint8 is what build() converts frames to anyway, and the round trip
        # through /255 back to uint8 is exact - so this costs nothing in
        # fidelity and a quarter of the memory of keeping floats.
        rgb = (small.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        n, h, w = rgb.shape[:3]

        entry = {"rgb": rgb, "n": n, "depth": None, "lohi": {},
                 "moge_z": None, "moge_mask": None, "moge_K": None}

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
            if depth is not None:
                logging.warning("CrossView Preview: both depth and moge_geometry are connected; "
                                "using moge_geometry, as the warp node does.")
        else:
            # Exactly build()'s depth_bhw, at preview scale: clamp and channel
            # mean AFTER the resize, in that order, or the values drift.
            d32 = _fit(depth.detach().float().cpu(), size).clamp(0, 1).mean(dim=-1).numpy()
            entry["depth"] = d32
            # The percentiles _depth_to_z would measure over the whole clip, for
            # every switch combination a request can ask for. Smoothing is per
            # frame, so its variant has to be measured on a smoothed full clip
            # even though only one frame gets smoothed per request.
            for sm in (False, True):
                try:
                    ds = _smooth_inplace(d32.copy(), rgb) if sm else d32
                except ImportError:
                    continue     # no cv2: a real run with smooth_depth would fail too
                for inv in (False, True):
                    entry["lohi"][(sm, inv)] = _lohi(ds, inv)

        mb = (rgb.nbytes + (entry["depth"].nbytes if entry["depth"] is not None else 0)
              + (entry["moge_z"].nbytes if entry["moge_z"] is not None else 0)) / 1e6
        entry["mb"] = round(mb, 1)

        key = str(unique_id)
        with _CACHE_LOCK:
            _CACHE[key] = entry
            _CACHE.move_to_end(key)
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
        logging.info("CrossView Preview: cached %d frames at %dx%d (%.1f MB) for node %s",
                     n, w, h, mb, key)

        # The real output, at full resolution, from the same build() the preview
        # is a viewfinder on. Producing it here is what lets one node do the job
        # of two: the pose you dial in on the widget is the pose that reaches the
        # IC-LoRA guide, with no numbers copied between nodes. Cached first, so a
        # clip that makes build() fail still leaves something to look at.
        warp, orbit = cvw.CrossViewWarp().build(
            frames=frames, depth=depth, moge_geometry=moge_geometry,
            azimuth=azimuth, elevation=elevation, distance=distance, hfov=hfov,
            vertical_shift=vertical_shift, depth_ratio=depth_ratio,
            smooth_depth=smooth_depth, invert_depth=invert_depth, roll_lock=roll_lock,
            pivot_override=pivot_override, pivot_x=pivot_x, pivot_y=pivot_y,
            pivot_z=pivot_z, keep_source_aim=keep_source_aim,
            # frame_count is a UI aid for the orbit sphere's even spreading; the
            # playhead already puts each keyframe on the frame the user chose, so
            # there is nothing to spread and 0 keeps build() from warning about it.
            use_keyframes=use_keyframes, frame_count=0, keyframes=keyframes,
            interp_motion=interp_motion, interpolation=interpolation)

        # `ui` tells the browser the clip length so the widget can stop claiming
        # to be current once the source changes underneath it; `result` is the
        # ordinary node output.
        return {"ui": {"crossview_preview": [{"of": n, "w": w, "h": h, "mb": entry["mb"],
                                              "source": "moge" if entry["moge_z"] is not None
                                              else "depth"}]},
                "result": (warp, orbit)}


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
    # PromptServer is a singleton shared by both installed copies, so it is the
    # one place an "already registered" flag is visible to both. A duplicate
    # aiohttp path raises at app finalisation, i.e. at startup.
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
        mode = "source" if data.get("mode") == "source" else "warp"
        loop = asyncio.get_running_loop()
        try:
            # Off the event loop: a ~50 ms render inline would stall every other
            # ComfyUI request for as long as the user keeps dragging.
            jpeg, frame, pivot = await loop.run_in_executor(
                None, _render_sync, entry, data.get("frame", 1),
                data.get("params") or {}, mode)
        except Exception as e:
            logging.exception("CrossView Preview: render failed")
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
        # The pivot rides along in a header rather than being drawn into the
        # JPEG, so the browser can restyle or hide the marker without asking for
        # another render - and the returned pixels stay the warp, unannotated.
        info = json.dumps({"frame": frame, "count": entry["n"], "mode": mode, "pivot": pivot})
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


_PID = f"CrossViewPreview{NODE_SUFFIX}"
_PNAME = "CrossView Live Preview" + (f" [{NODE_SUFFIX}]" if NODE_SUFFIX else "")

NODE_CLASS_MAPPINGS = {_PID: CrossViewPreview}
NODE_DISPLAY_NAME_MAPPINGS = {_PID: _PNAME}
