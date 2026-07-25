"""CrossViewWarp — build the depth-warp conditioning for the CrossView-Warp IC-LoRA.

Given the input video frames (cam A) + a depth map (from an external Depth Anything V2
node) + a requested relative camera pose, this reprojects each frame into the target
viewpoint (magenta disocclusion holes) and returns the warp control video plus an
orbit-view diagram of the camera setup.

The warp math mirrors the training-time builder (build_warp_dataset.py /
monocular_warp.py) EXACTLY -- same magenta hole fill, painter z-buffer, splat,
intrinsics (fx = W/(2 tan(hfov/2)), cx=cy=W/2) -- so the model sees the same signal
it was trained on. Depth is normalised globally across the clip for temporal
consistency and re-centred so an off-centre subject stays framed.
"""

import numpy as np
import torch

try:
    from comfy.utils import ProgressBar   # ComfyUI runtime: drives the node's progress bar
except Exception:  # allow standalone import (tests / non-ComfyUI use)
    ProgressBar = None

MAGENTA = np.array([255, 0, 255], dtype=np.uint8)


def _dist_scale(d):
    """Map source-distance ratio (0.2..3.0) to a canvas radius multiplier.

    Piecewise so dist=1.0 lands exactly on the shell (multiplier 1.0):
    dist<=1 spreads 0.45..1.0 (inside the shell = closer), dist>1 spreads
    1.0..1.5 so the marker keeps moving visibly all the way up to 3.0
    (the previous clamp at 1.3 froze it past dist~=1.55)."""
    d = float(d)
    if d <= 1.0:
        return 0.45 + 0.55 * d
    return 1.0 + 0.25 * (d - 1.0)

try:
    from numba import njit

    @njit(cache=False)   # disk cache pickles the module path -> breaks when
    # the module name differs between ComfyUI and standalone use; JIT-once per
    # process (~2s at first node run) is the safe trade
    def _splat_kernel(tu0, tv0, cols, warp, splat, H, W):
        # z-sorted painter order, offset passes in dy/dx order - must stay
        # identical to the numpy fallback below (training warp format)
        n = tu0.shape[0]
        for dy in range(-splat, splat + 1):
            for dx in range(-splat, splat + 1):
                for i in range(n):
                    x = tu0[i] + dx
                    y = tv0[i] + dy
                    if 0 <= x < W and 0 <= y < H:
                        warp[y, x, 0] = cols[i, 0]
                        warp[y, x, 1] = cols[i, 1]
                        warp[y, x, 2] = cols[i, 2]
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _orbit_view_image(azimuth, elevation, distance, size=512):
    """1:1 port of the crossview_orbit.js widget's default view (gizmo style):
    coverage zone hint, equator/meridian rings, snap dots (F/L/R/+-90/H/Lo),
    dolly ray + 1.0x tick, camera handle with viewfinder lines."""
    from PIL import Image, ImageDraw, ImageFont
    W = H = size
    k = size / 300.0                     # widget geometry is authored at ~300px height
    S = min(W - 16 * k, H - 12 * k)
    cx, cy = W / 2.0, H / 2.0
    R = (S / 2.0 - 4 * k) * 0.76
    yaw, tilt = 0.24, 0.20               # widget default view

    ZONE_GREEN, ZONE_YELLOW = (45, 30, 15), (65, 40, 25)
    C_GREEN, C_YELLOW = (80, 200, 120), (230, 200, 90)

    def in_ellipse(az, el, zone):
        A, Eup, Edn = zone
        an = az / A
        en = el / Eup if el >= 0 else el / Edn
        return an * an + en * en <= 1

    def zone_color(az, el):
        if in_ellipse(az, el, ZONE_GREEN):
            return C_GREEN
        if in_ellipse(az, el, ZONE_YELLOW):
            return C_YELLOW
        return None                       # red zone: no hint drawn (widget skips it)

    def rot(p):
        x, y, z = p
        cyw, syw = np.cos(yaw), np.sin(yaw)
        ct, st = np.cos(tilt), np.sin(tilt)
        xr = x * cyw + y * syw
        yr = -x * syw + y * cyw
        yv = yr * ct - z * st
        zv = yr * st + z * ct
        return xr, yv, zv                 # yv < 0 => front

    def pt(a_deg, e_deg):
        a, e = np.radians(a_deg), np.radians(e_deg)
        xr, yv, zv = rot((np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)))
        return cx + R * xr, cy - R * zv, yv

    img = Image.new("RGBA", (W, H), (27, 27, 31, 255))

    def layer():
        return Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # sphere disc (widget: rgba(70,74,86,0.25))
    lay = layer(); d = ImageDraw.Draw(lay)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(70, 74, 86, 64))
    img = Image.alpha_composite(img, lay)

    # coverage hint band (front hemisphere only, alpha 0.13)
    lay = layer(); d = ImageDraw.Draw(lay)
    STEP = 15
    for a0 in range(-70, 70, STEP):
        for e0 in range(-30, 45, STEP):
            col = zone_color(a0 + STEP / 2, e0 + STEP / 2)
            if col is None:
                continue
            quad, dep = [], 0
            for aa, ee in ((a0, e0), (a0 + STEP, e0), (a0 + STEP, e0 + STEP), (a0, e0 + STEP)):
                x, y, f = pt(aa, ee)
                quad.append((x, y)); dep = f
            if dep >= 0:
                continue
            d.polygon(quad, fill=col + (33,))
    img = Image.alpha_composite(img, lay)

    d = ImageDraw.Draw(img)
    lw = max(1, round(1.5 * k))
    # equator + meridian rings (front bright, back faint)
    for ring in ([(a, 0) for a in range(-180, 181, 8)],
                 [(0, e) for e in range(-90, 91, 8)]):
        for (a1, e1), (a2, e2) in zip(ring[:-1], ring[1:]):
            x1, y1, f1 = pt(a1, e1)
            x2, y2, f2 = pt(a2, e2)
            col = (200, 204, 216, 128) if (f1 < 0 and f2 < 0) else (120, 124, 138, 38)
            d.line([x1, y1, x2, y2], fill=col, width=lw)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(190, 190, 205, 140), width=lw)

    # subject pin (widget: 4px line + r5 head at 300px scale)
    d.line([cx, cy + 12 * k, cx, cy - 2 * k], fill=(216, 216, 224, 255), width=max(2, round(4 * k)))
    r5 = 5 * k
    d.ellipse([cx - r5, cy - 8 * k - r5, cx + r5, cy - 8 * k + r5], fill=(216, 216, 224, 255))

    # snap dots
    try:
        f_dot = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", round(9 * k))
        f_txt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", round(11 * k))
    except Exception:
        f_dot = f_txt = ImageFont.load_default()
    SNAPS = [(0, 0, "F"), (-45, 0, "L"), (45, 0, "R"), (-90, 0, ""), (90, 0, ""), (0, 30, "H"), (0, -15, "Lo")]
    for a, e, lab in SNAPS:
        sxp, syp, f = pt(a, e)
        rr = (9 if lab else 6) * k
        al = 255 if f < 0 else 90
        fill = (255, 255, 255, al) if (a == 0 and e == 0) else (58, 65, 80, al)
        d.ellipse([sxp - rr, syp - rr, sxp + rr, syp + rr], fill=fill,
                  outline=(154, 162, 181, al), width=max(1, round(1.5 * k)))
        if lab:
            tcol = (20, 22, 27, al) if (a == 0 and e == 0) else (217, 220, 227, al)
            bb = d.textbbox((0, 0), lab, font=f_dot)
            d.text((sxp - (bb[2] - bb[0]) / 2, syp - (bb[3] - bb[1]) / 2 - bb[1]), lab, fill=tcol, font=f_dot)

    # arc home -> camera (width 2.5 @300px)
    for i in range(14):
        x1, y1, _ = pt(azimuth * i / 14.0, elevation * i / 14.0)
        x2, y2, _ = pt(azimuth * (i + 1) / 14.0, elevation * (i + 1) / 14.0)
        d.line([x1, y1, x2, y2], fill=(120, 190, 255, 255), width=max(2, round(2.5 * k)))

    # dolly ray (dashed, subject -> 1.32x) + 1.0x tick
    sx, sy, front = pt(azimuth, elevation)
    vx, vy = sx - cx, sy - cy
    L = (vx * vx + vy * vy) ** 0.5 + 1e-9
    dash, gap = 4 * k, 4 * k
    t = 0.0
    while t < 1.32 * L:
        t2 = min(t + dash, 1.32 * L)
        d.line([cx + vx / L * t, cy + vy / L * t, cx + vx / L * t2, cy + vy / L * t2],
               fill=(120, 190, 255, 90), width=max(1, round(k)))
        t = t2 + gap
    r3 = 3 * k
    d.ellipse([sx - r3, sy - r3, sx + r3, sy + r3], outline=(255, 255, 255, 153), width=max(1, round(k)))

    # camera handle at distance-scaled radius, with viewfinder lines
    distF = _dist_scale(distance)
    px = cx + vx * distF
    py = cy + vy * distF
    al = 255 if front < 0 else 115
    for dx, dy in ((-8, -6), (8, -6), (-8, 6), (8, 6)):
        d.line([px, py, cx + dx * k, cy + (dy - 6) * k], fill=(120, 190, 255, al), width=max(1, round(k)))
    d.rounded_rectangle([px - 13 * k, py - 9 * k, px + 13 * k, py + 9 * k], radius=4 * k,
                        fill=(120, 190, 255, al), outline=(255, 255, 255, al), width=max(2, round(2 * k)))
    r35 = 3.5 * k
    d.ellipse([px + 5 * k - r35, py - r35, px + 5 * k + r35, py + r35], fill=(32, 36, 44, al))

    d.text((8 * k, 6 * k), f"az {azimuth:+.0f}  el {elevation:+.0f}  dist {distance:.2f}x",
           fill=(255, 255, 100, 255), font=f_txt)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


# --- warp math (mirrors monocular_warp.py) -----------------------------------

def _warp_frame(rgb_ref, depth_ref, C_ref, C_tgt, fx_pix, splat, cx, cy):
    H, W = depth_ref.shape
    fy_pix = fx_pix
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_ref
    fin = np.isfinite(z) & (z > 0)
    thr = np.percentile(z[fin], 99.5) if fin.any() else 0
    fin = fin & (z < thr)
    Xc = np.stack([(u - W / 2.0) / fx_pix * z, (v - H / 2.0) / fy_pix * z, z], -1).reshape(-1, 3)
    Xw = (C_ref[:3, :3] @ Xc.T).T + C_ref[:3, 3]
    Ci = np.linalg.inv(C_tgt)
    Xd = (Ci[:3, :3] @ Xw.T).T + Ci[:3, 3]
    zt = Xd[:, 2]
    ui = np.round(Xd[:, 0] / zt * fx_pix + cx).astype(int)
    vi = np.round(Xd[:, 1] / zt * fy_pix + cy).astype(int)
    val = fin.ravel() & (zt > 0)
    order = np.argsort(-zt)
    sel = order[val[order]]
    # gather indices/colors once; the splat passes only offset them
    tu0 = ui[sel]
    tv0 = vi[sel]
    cols = np.ascontiguousarray(rgb_ref.reshape(-1, 3)[sel])
    warp = np.tile(MAGENTA, (H, W, 1))
    done = False
    if _HAVE_NUMBA:
        try:
            _splat_kernel(tu0.astype(np.int64), tv0.astype(np.int64), cols, warp, int(splat), H, W)
            done = True
        except Exception:
            done = False   # any numba runtime issue -> numpy fallback below
    if not done:
        flat = warp.reshape(-1, 3)
        for dy in range(-splat, splat + 1):
            for dx in range(-splat, splat + 1):
                tu = tu0 + dx
                tv = tv0 + dy
                ok = (tu >= 0) & (tu < W) & (tv >= 0) & (tv < H)
                flat[tv[ok] * W + tu[ok]] = cols[ok]
    return warp.astype(np.uint8)


def _look_at(eye, target, world_down=np.array([0.0, 1.0, 0.0])):
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    right = np.cross(world_down, f)
    right = right / (np.linalg.norm(right) + 1e-9)
    down = np.cross(f, right)
    C = np.eye(4)
    C[:3, 0], C[:3, 1], C[:3, 2], C[:3, 3] = right, down, f, eye
    return C


def _rodrigues(axis, ang):
    a = axis / (np.linalg.norm(axis) + 1e-9)
    c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _depth_to_z(depth_bhw, invert, ratio=4.0):
    """Normalise a brightness-like depth stack globally -> z (near small, far large).

    `ratio` bounds the far/near depth ratio so close-ups don't get exaggerated
    relief (which smears the warp and makes a small orbit look much larger):
    z = 1/(1/ratio + (1-1/ratio)*dn).
    """
    d = depth_bhw.astype(np.float64)
    if invert:
        d = -d
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    dn = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    r = max(float(ratio), 1.01)
    return 1.0 / (1.0 / r + (1.0 - 1.0 / r) * dn)


# --- keyframe interpolation --------------------------------------------------

def _wrap_deg(a):
    """Wrap degrees to (-180, 180]."""
    return ((a + 180.0) % 360.0) - 180.0


def _ease(t, mode):
    """Map linear t in [0,1] to eased t in [0,1] per the chosen curve."""
    if mode == "ease_in_out":
        return 0.5 - 0.5 * np.cos(np.pi * t)
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    return t   # "linear" / unknown -> identity


def _interp_value(a, b, t, mode):
    return a + (b - a) * _ease(t, mode)


def _interp_angle(a_deg, b_deg, t, mode):
    """Shortest-path interpolation on the azimuth circle (handles the seam at +-180)."""
    diff = _wrap_deg(b_deg - a_deg)
    return _wrap_deg(a_deg + diff * _ease(t, mode))


def _interp_abc(a, b, c, t, smooth):
    """Three-point scalar interpolation. smooth=True uses a quadratic Bezier
    whose control point pc = 2*b - 0.5*(a+c) makes the curve pass through b
    at t=0.5; smooth=False is piecewise linear a->b->c. Both hit a/b/c at
    t=0/0.5/1."""
    if smooth:
        pc = 2.0 * b - 0.5 * (a + c)
        mt = 1.0 - t
        return mt * mt * a + 2.0 * mt * t * pc + t * t * c
    if t <= 0.5:
        return a + (b - a) * (t / 0.5)
    return b + (c - b) * ((t - 0.5) / 0.5)


def _interp_abc_angle(a_deg, b_deg, c_deg, t, smooth):
    """Three-point angle interp: unwrap b then c relative to the running
    tangent so the Bezier control-point math stays continuous across the
    +-180 seam, then wrap the result back to (-180, 180]."""
    b_unwrap = a_deg + _wrap_deg(b_deg - a_deg)
    c_unwrap = b_unwrap + _wrap_deg(c_deg - b_unwrap)
    return _wrap_deg(_interp_abc(a_deg, b_unwrap, c_unwrap, t, smooth))


def _orbit_C_tgt(az_deg, el_deg, dist, pivot):
    """Camera pose orbiting `pivot` at the requested az/el/dist.

    Mirrors the inline math that was in build(): angle signs are negated so
    +azimuth = camera orbits RIGHT, +elevation = camera RISES (OpenCV frame)."""
    R_orbit = _rot_y(np.radians(-az_deg)) @ _rot_x(np.radians(-el_deg))
    eye = pivot + dist * (R_orbit @ (-pivot))
    return _look_at(eye, pivot)


# --- ComfyUI node ------------------------------------------------------------

class CrossViewWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames (the source view). Feed the same "
                    "frames you send to the Depth Anything V2 node."}),
                "depth": ("IMAGE", {
                    "tooltip": "Depth map for the SAME frames, from a Depth Anything V2 node "
                    "(DA-V2 Large). Brightness = depth; polarity is handled by invert_depth."}),
                "azimuth": ("FLOAT", {"default": -30.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Horizontal orbit angle (deg). Negative = camera orbits LEFT, "
                    "positive = RIGHT. The strongest control. Reliable up to about +-45, usable "
                    "to +-65; beyond that the hidden side is mostly invented."}),
                "elevation": ("FLOAT", {"default": 20.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Vertical orbit angle (deg). Positive = camera rises (looks down "
                    "on the subject), negative = looks up. NOTE: the effect is weaker/subtler "
                    "than azimuth - the orbit is stronger horizontally."}),
                "distance": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05,
                    "tooltip": "Camera distance from the subject (1.0 = same as source). "
                    "Below 1 = move closer / zoom in; above 1 = pull back. Extreme values "
                    "enlarge the disoccluded (magenta) holes."}),
                "hfov": ("FLOAT", {"default": 50.0, "min": 20.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Assumed camera field of view (deg). ~50 is a normal lens. Only "
                    "change it if your source clip is clearly wide-angle or telephoto."}),
                "head_bias": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.02,
                    "tooltip": "Manual vertical framing shift (fraction of height; + = shift the "
                    "view up). Leave at 0 unless the subject's head gets clipped."}),
                "depth_ratio": ("FLOAT", {"default": 6.0, "min": 1.5, "max": 1000.0, "step": 0.5,
                    "tooltip": "Max far/near depth ratio of the scene. Lower = flatter relief and "
                    "a cleaner warp; higher = more parallax but messier on cluttered scenes. "
                    "Close-up faces: 2.5-4; mid shots: 4-8; deep/wide scenes: 8-16."}),
                "smooth_depth": ("BOOLEAN", {"default": False,
                    "tooltip": "Edge-aware depth smoothing before warping. Turn ON if the warp "
                    "has too many speckle holes; it trades a little sharpness for cleaner, more "
                    "coherent disocclusion."}),
                "invert_depth": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip depth polarity (near<->far). Leave FALSE for the standard "
                    "ComfyUI DA-V2 node. Turn ON only if the warp looks inside-out (background "
                    "moves like foreground / the subject caves in)."}),
            },
            "optional": {
                "roll_lock": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep the subject upright: matches its in-image lean to the source "
                    "so a tilted source shot doesn't tip the subject over at large angles. "
                    "Leave ON."}),
                "pivot_override": ("BOOLEAN", {"default": True,
                    "tooltip": "Orbit around an explicit pivot point (set below) instead of an "
                    "auto-estimated one. ON by default with a pivot near the subject, which is "
                    "the recommended setup."}),
                "pivot_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Pivot X in the source camera frame (+ = right), in depth units. "
                    "0 = image centre."}),
                "pivot_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Pivot Y (+ = down), in depth units. 0 = image centre."}),
                "pivot_z": ("FLOAT", {"default": 1.05, "min": 0.01, "max": 1000.0, "step": 0.01,
                    "tooltip": "Pivot depth (how far in front of the camera). ~1.05 sits on the "
                    "nearest subject in the middle of the frame; raise it to orbit around "
                    "something further back."}),
                # NOTE: new widgets MUST come after the original five optional
                # ones (roll_lock..pivot_z). ComfyUI stores widget_values
                # positionally in saved workflows, so inserting a widget in the
                # middle shifts every downstream value into the wrong slot and
                # breaks previously saved nodes on load.
                "use_keyframes": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable the two-point camera move (A at frame 0 -> B at the last "
                    "frame), interpolated per-frame following 'interp'. When OFF, the single "
                    "azimuth/elevation/distance above is applied to every frame. Use S+click on "
                    "the orbit sphere to set A and B visually."}),
                "A_azimuth": ("FLOAT", {"default": -30.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Keyframe A: camera azimuth at frame 0 (deg). Hidden in the UI - "
                    "set it with S+click on the sphere. (Right-click the node to unhide.)"}),
                "A_elevation": ("FLOAT", {"default": 20.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Keyframe A: camera elevation at frame 0 (deg). Hidden in the UI - "
                    "set it with S+click on the sphere. (Right-click the node to unhide.)"}),
                "A_distance": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05,
                    "tooltip": "Keyframe A: camera distance at frame 0 (1.0 = source distance). "
                    "Drag this slider to push/pull the green A marker along its current direction."}),
                "B_azimuth": ("FLOAT", {"default": 30.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Keyframe B: camera azimuth at the LAST frame (deg). Hidden in the "
                    " UI - set it with S+click on the sphere. (Right-click the node to unhide.)"}),
                "B_elevation": ("FLOAT", {"default": 20.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Keyframe B: camera elevation at the last frame (deg). Hidden in "
                    "the UI - set it with S+click on the sphere. (Right-click the node to unhide.)"}),
                "B_distance": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05,
                    "tooltip": "Keyframe B: camera distance at the last frame (1.0 = source "
                    "distance). Drag this slider to push/pull the green B marker along its "
                    "current direction."}),
                "C_azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 999.0, "step": 1.0,
                    "tooltip": "Keyframe C: camera azimuth at the midpoint frame (deg). Hidden "
                    "in the UI - set it with S+click on the sphere. (Right-click the node to "
                    "unhide.) 999.0 is the 'no C placed' sentinel - max is 999 instead of 180 "
                    "only so this sentinel passes ComfyUI's input validation; the runtime clamps "
                    "real values back to the +-180 legal range."}),
                "C_elevation": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Keyframe C: camera elevation at the midpoint frame (deg). Hidden "
                    "in the UI - set it with S+click on the sphere. (Right-click the node to "
                    "unhide.)"}),
                "C_distance": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05,
                    "tooltip": "Keyframe C: camera distance at the midpoint frame (1.0 = source "
                    "distance). Drag this slider to push/pull the green C marker along its "
                    "current direction."}),
                "interp_motion": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear",
                    "tooltip": "Curve used to interpolate between keyframe A and B across the "
                    "video frames. linear = constant angular speed; ease_in_out = slow start & "
                    "end (cinematic); ease_in / ease_out = one-sided. Only used when 'use_keyframes' "
                    "is ON."}),
                "abc_smooth": ("BOOLEAN", {"default": False,
                    "tooltip": "Path shape through keyframes A/B/C. OFF (default) = "
                    "piecewise linear A->B->C (constant angular speed per leg, corner at "
                    "B). ON = quadratic Bezier that glides through B with no corner. Only "
                    "used when 'use_keyframes' is ON and C is placed."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("warp", "orbit_view")
    OUTPUT_TOOLTIPS = (
        "The warp control video (magenta = disoccluded holes) -> connect to the "
        "IC-LoRA reference guide for the WARP stream.",
        "Front view of the orbit globe with the camera marker at the requested "
        "az/el/dist - wire to a PreviewImage/SaveImage to document the camera setup.",
    )
    FUNCTION = "build"
    CATEGORY = "CrossView"

    def build(self, frames, depth, azimuth, elevation, distance, hfov, head_bias, depth_ratio, smooth_depth, invert_depth,
              roll_lock=True, pivot_override=True, pivot_x=0.0, pivot_y=0.0, pivot_z=1.05,
              interp_motion="linear", use_keyframes=False,
              A_azimuth=-30.0, A_elevation=20.0, A_distance=1.0,
              B_azimuth=30.0, B_elevation=20.0, B_distance=1.0,
              C_azimuth=0.0, C_elevation=0.0, C_distance=1.0,
              abc_smooth=False):
        # frames: [B,H,W,C] float [0,1]; depth: [B,H,W,C] brightness [0,1]
        rgb = (frames.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)   # [B,H,W,3]
        B, H, W = rgb.shape[:3]
        depth_bhw = depth.clamp(0, 1).mean(dim=-1).cpu().numpy()            # brightness
        if smooth_depth:
            # edge-aware smoothing: median kills salt noise, the RGB-guided
            # filter re-aligns depth edges to image edges -> neighbouring
            # pixels land adjacently in the warp = coherent holes (see tooltip)
            import cv2
            for i in range(B):
                d32 = cv2.medianBlur(depth_bhw[i].astype(np.float32), 3)
                try:
                    d32 = cv2.ximgproc.guidedFilter(rgb[i], d32, radius=8, eps=1e-3)
                except Exception:
                    d32 = cv2.bilateralFilter(d32, 9, 0.1, 9.0)
                depth_bhw[i] = d32
        z = _depth_to_z(depth_bhw, invert_depth, depth_ratio)              # [B,H,W]

        fx = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
        cc = W / 2.0
        cch = H / 2.0

        # subject pivot = central-region foreground centroid on frame 0
        z0 = z[0]
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        central = np.zeros_like(z0, bool)
        central[H // 8: 4 * H // 5, W // 5: 4 * W // 5] = True
        fin = np.isfinite(z0) & (z0 > 0) & central
        fin &= z0 < np.percentile(z0[fin], 95.0)
        fg = fin & (z0 < np.percentile(z0[fin], 50.0))
        Xw0 = np.stack([(uu - cc) / fx * z0, (vv - cch) / fx * z0, z0], -1)
        # pivot = median 3D point of the trimmed central region (scene-level
        # aim, matching the training warps' real-camera framing; a nearest-
        # cluster pivot would zoom the orbit camera into the subject)
        pivot = np.median(Xw0[fin], axis=0)
        if pivot_override:
            pivot = np.array([pivot_x, pivot_y, pivot_z], dtype=np.float64)

        C_ref = np.eye(4)

        # "Base" pose drives roll-lock estimation AND the orbit_view preview.
        # For a single-pose run this is just (az, el, dist); for a keyframed
        # run we use the move's midpoint so the upright correction is the
        # average of where the camera actually lives across the clip.
        keyframing = bool(use_keyframes and B > 1)
        # "C exists" is signalled by a sentinel value the JS writes into
        # C_azimuth whenever kfC is null: 999.0 is outside the legal +-180
        # range, so an in-range C_azimuth means C has been placed.
        use_c_path = keyframing and abs(C_azimuth) <= 180.0
        smooth_abc = bool(abc_smooth)
        if use_c_path:
            mid_az = _interp_abc_angle(A_azimuth, B_azimuth, C_azimuth, 0.5, smooth_abc)
            mid_el = _interp_abc(A_elevation, B_elevation, C_elevation, 0.5, smooth_abc)
            mid_dist = _interp_abc(A_distance, B_distance, C_distance, 0.5, smooth_abc)
        elif keyframing:
            mid_az = _interp_angle(A_azimuth, B_azimuth, 0.5, interp_motion)
            mid_el = _interp_value(A_elevation, B_elevation, 0.5, interp_motion)
            mid_dist = _interp_value(A_distance, B_distance, 0.5, interp_motion)
        else:
            mid_az, mid_el, mid_dist = azimuth, elevation, distance
        C_tgt = _orbit_C_tgt(mid_az, mid_el, mid_dist, pivot)

        # roll lock: keep the subject's projected in-image lean identical to
        # the source by rolling the camera about its optical axis (a pitched
        # source shot would otherwise tip the character at large azimuths).
        # The subject cloud = depth band around the pivot + horizontal window
        # around its projected column, so receding walls can't hijack the axis.
        fgc = Xw0[fg].mean(0)                      # subject (nearest cluster) centre
        pu = cc + fgc[0] / fgc[2] * fx
        subj = fin & (np.abs(z0 - fgc[2]) < 0.3 * fgc[2]) & (np.abs(uu - pu) < 0.3 * W)
        if subj.sum() < 500:
            subj = fg
        def _lean(P2):
            """(angle, dominance) of the cloud's major axis; angle 0 = vertical.
            Returns angle=None when the axis is ambiguous: near-isotropic cloud
            or a near-HORIZONTAL axis, where the up-sign disambiguation flips
            arbitrarily and produced 90-180 deg phantom rolls."""
            P2 = P2 - P2.mean(0)
            _ev, V2 = np.linalg.eigh(P2.T @ P2)
            v2 = V2[:, -1]
            if v2[1] > 0:
                v2 = -v2                      # point up (image y is down)
            dom = np.sqrt(_ev[-1] / max(_ev[-2], 1e-9))
            ang = np.arctan2(v2[0], -v2[1])   # 0 = vertical, + = leaning right
            if dom < 1.3 or abs(ang) > np.radians(45):
                return None
            return ang

        def _lean_in(C):
            Ci2 = np.linalg.inv(C)
            Xd2 = (Ci2[:3, :3] @ Xw0[subj].T).T + Ci2[:3, 3]
            m2 = Xd2[:, 2] > 0
            return _lean(np.stack([Xd2[m2, 0] / Xd2[m2, 2], Xd2[m2, 1] / Xd2[m2, 2]], -1))

        def _wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        th_src = _lean(np.stack([uu[subj].astype(np.float64), vv[subj].astype(np.float64)], -1))
        th_tgt = _lean_in(C_tgt)
        # roll-lock correction is estimated ONCE at the midpoint pose and the
        # same scalar `applied_droll` is then applied to every per-frame C_tgt
        # so the subject stays upright consistently across the whole move.
        applied_droll = 0.0
        if roll_lock and th_src is not None and th_tgt is not None:
            droll = float(np.clip(_wrap(th_tgt - th_src), -np.radians(35), np.radians(35)))
            err0 = abs(_wrap(th_tgt - th_src))
            for cand in (droll, -droll):
                C_try = C_tgt.copy()
                C_try[:3, :3] = _rodrigues(C_tgt[:3, 2], cand) @ C_tgt[:3, :3]
                th_try = _lean_in(C_try)
                if th_try is not None and abs(_wrap(th_try - th_src)) < err0 - 1e-6:
                    applied_droll = cand
                    C_tgt = C_try
                    break

        # principal point stays at the image centre (training warps use real
        # camera intrinsics, no lens-shift); head_bias is a manual vertical
        # shift only (default 0 = off)
        cx_eff = cc
        cy_eff = cch + head_bias * H

        pbar = ProgressBar(B) if ProgressBar is not None else None
        warp_frames = []
        for i in range(B):
            if use_c_path:
                t = i / (B - 1)
                fr_az = _interp_abc_angle(A_azimuth, B_azimuth, C_azimuth, t, smooth_abc)
                fr_el = _interp_abc(A_elevation, B_elevation, C_elevation, t, smooth_abc)
                fr_dist = _interp_abc(A_distance, B_distance, C_distance, t, smooth_abc)
                C_tgt_i = _orbit_C_tgt(fr_az, fr_el, fr_dist, pivot)
                if applied_droll != 0.0:
                    C_tgt_i = C_tgt_i.copy()
                    C_tgt_i[:3, :3] = _rodrigues(C_tgt_i[:3, 2], applied_droll) @ C_tgt_i[:3, :3]
            elif keyframing:
                t = i / (B - 1)
                fr_az = _interp_angle(A_azimuth, B_azimuth, t, interp_motion)
                fr_el = _interp_value(A_elevation, B_elevation, t, interp_motion)
                fr_dist = _interp_value(A_distance, B_distance, t, interp_motion)
                C_tgt_i = _orbit_C_tgt(fr_az, fr_el, fr_dist, pivot)
                if applied_droll != 0.0:
                    C_tgt_i = C_tgt_i.copy()
                    C_tgt_i[:3, :3] = _rodrigues(C_tgt_i[:3, 2], applied_droll) @ C_tgt_i[:3, :3]
            else:
                C_tgt_i = C_tgt
            warp_frames.append(_warp_frame(rgb[i], z[i], C_ref, C_tgt_i, fx, 2, cx_eff, cy_eff))
            if pbar is not None:
                pbar.update(1)   # advance the node's progress bar one frame
        warp = np.stack(warp_frames, 0)  # [B,H,W,3] uint8

        warp_t = torch.from_numpy(warp.astype(np.float32) / 255.0)
        orbit = _orbit_view_image(mid_az, mid_el, mid_dist)
        orbit_t = torch.from_numpy(orbit.astype(np.float32) / 255.0)[None]  # [1,H,W,3]
        return (warp_t, orbit_t)


NODE_CLASS_MAPPINGS = {"CrossViewWarp": CrossViewWarp}
NODE_DISPLAY_NAME_MAPPINGS = {"CrossViewWarp": "CrossView Warp (video -> warp)"}
