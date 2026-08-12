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

import json
import logging
from pathlib import Path

import numpy as np
import torch

# Optional local suffix for the node id, read from an untracked `.node_suffix`
# file next to this module. It lets a second copy of this package (a dev
# worktree symlinked into custom_nodes beside the released one) register under
# its own id, so both can be dropped into one workflow and compared. Users never
# have the file, so the id is unchanged for them and nothing needs reverting
# before a release - which is the point: an edit that must be undone by hand
# eventually ships by accident.
_sfx = Path(__file__).parent / ".node_suffix"
NODE_SUFFIX = _sfx.read_text().strip() if _sfx.exists() else ""

try:
    from comfy.utils import ProgressBar   # ComfyUI runtime: drives the node's progress bar
except Exception:  # allow standalone import (tests / non-ComfyUI use)
    ProgressBar = None

MAGENTA = np.array([255, 0, 255], dtype=np.uint8)

# Filled in by crossview_preview_node on import; called at the end of build().
# A slot rather than an import keeps the dependency one-way.
_PREVIEW_SINK = None


def _dist_scale(d):
    """Map source-distance ratio (0.1..3.0) to a canvas radius multiplier.

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


def _orbit_view_image(azimuth, elevation, distance, size=512, kfs=None, smooth=False):
    """1:1 port of the crossview_orbit.js widget's default view (gizmo style):
    coverage zone hint, equator/meridian rings, snap dots (F/L/R/+-90/H/Lo),
    dolly ray + 1.0x tick, camera handle with viewfinder lines.

    Pass `kfs` (the parsed keyframe list) to draw the camera PATH instead of the
    single pose -- green arc plus frame-numbered markers, matching the widget."""
    from PIL import Image, ImageDraw, ImageFont
    W = H = size
    k = size / 300.0                     # widget geometry is authored at ~300px height
    S = min(W - 16 * k, H - 12 * k)
    cx, cy = W / 2.0, H / 2.0
    # 0.62 (not 0.76) so the camera handle still fits when _dist_scale reaches
    # 1.5R at dist=3.0 -- must stay in step with geom() in web/crossview_orbit.js,
    # which this function is a 1:1 port of.
    R = (S / 2.0 - 4 * k) * 0.62
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

    def dashed(x1, y1, x2, y2, col, w):
        seg = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 + 1e-9
        dash, gap = 4 * k, 4 * k
        t = 0.0
        while t < seg:
            t2 = min(t + dash, seg)
            d.line([x1 + (x2 - x1) * t / seg, y1 + (y2 - y1) * t / seg,
                    x1 + (x2 - x1) * t2 / seg, y1 + (y2 - y1) * t2 / seg], fill=col, width=w)
            t = t2 + gap

    def dim(col, front):
        """Fade a colour toward the background for markers on the far side.

        Done by mixing rather than by an alpha value: this ImageDraw writes
        straight onto an opaque image, so an alpha in the fill would be dropped
        by the final convert("RGB") and the marker would come out full strength."""
        if front < 0:
            return col + (255,)
        return tuple(round(c + (b - c) * 0.55) for c, b in zip(col, (27, 27, 31))) + (255,)

    if not kfs:
        # Static pose: the blue camera overlay. Skipped entirely during a
        # keyframed move (as the widget does), since azimuth/elevation/distance
        # do not drive the render then and drawing both would show two cameras.

        # arc home -> camera (width 2.5 @300px)
        for i in range(14):
            x1, y1, _ = pt(azimuth * i / 14.0, elevation * i / 14.0)
            x2, y2, _ = pt(azimuth * (i + 1) / 14.0, elevation * (i + 1) / 14.0)
            d.line([x1, y1, x2, y2], fill=(120, 190, 255, 255), width=max(2, round(2.5 * k)))

        # dolly ray (dashed, subject -> 1.32x) + 1.0x tick
        sx, sy, front = pt(azimuth, elevation)
        vx, vy = sx - cx, sy - cy
        L = (vx * vx + vy * vy) ** 0.5 + 1e-9
        dashed(cx, cy, cx + vx / L * 1.32 * L, cy + vy / L * 1.32 * L,
               (120, 190, 255, 90), max(1, round(k)))
        r3 = 3 * k
        d.ellipse([sx - r3, sy - r3, sx + r3, sy + r3], outline=(255, 255, 255, 153),
                  width=max(1, round(k)))

        # camera handle at distance-scaled radius, with viewfinder lines
        distF = _dist_scale(distance)
        px = cx + vx * distF
        py = cy + vy * distF
        al = 255 if front < 0 else 115
        for dx, dy in ((-8, -6), (8, -6), (-8, 6), (8, 6)):
            d.line([px, py, cx + dx * k, cy + (dy - 6) * k], fill=(120, 190, 255, al),
                   width=max(1, round(k)))
        d.rounded_rectangle([px - 13 * k, py - 9 * k, px + 13 * k, py + 9 * k], radius=4 * k,
                            fill=(120, 190, 255, al), outline=(255, 255, 255, al),
                            width=max(2, round(2 * k)))
        r35 = 3.5 * k
        d.ellipse([px + 5 * k - r35, py - r35, px + 5 * k + r35, py + r35], fill=(32, 36, 44, al))

        label = f"az {azimuth:+.0f}  el {elevation:+.0f}  dist {distance:.2f}x"
    else:
        # Keyframed move: the same green path the orbit widget draws, so this
        # output documents the actual camera move rather than one pose from it.
        # Walked segment by segment with the shared unwrap + Catmull-Rom helpers,
        # which is what keeps it identical to the widget and to the render.
        if len(kfs) >= 2:
            az_un = _unwrap_seq([kf["az"] for kf in kfs])
            els = [kf["el"] for kf in kfs]
            SUB = 24
            prev = None
            for seg in range(len(kfs) - 1):
                for s in range(SUB + 1):
                    u = s / SUB
                    a = _wrap_deg(_seg_value(az_un, seg, u, smooth))
                    e = _seg_value(els, seg, u, smooth)
                    x1, y1, _ = pt(a, e)
                    if prev is not None:
                        d.line([prev[0], prev[1], x1, y1], fill=(95, 206, 128, 255),
                               width=max(2, round(2.5 * k)))
                    prev = (x1, y1)

        for kf in kfs:
            f_no, kaz, kel, kdist = kf["f"], kf["az"], kf["el"], kf["dist"]
            kx, ky, kfront = pt(kaz, kel)
            distFk = _dist_scale(kdist)
            kpx, kpy = cx + (kx - cx) * distFk, cy + (ky - cy) * distFk
            dashed(kpx, kpy, kx, ky, dim((95, 206, 128), kfront), max(1, round(k)))
            r3 = 3 * k
            d.ellipse([kx - r3, ky - r3, kx + r3, ky + r3],
                      outline=dim((190, 190, 200), kfront), width=max(1, round(k)))
            rr = 10 * k
            d.ellipse([kpx - rr, kpy - rr, kpx + rr, kpy + rr], fill=dim((95, 206, 128), kfront),
                      outline=dim((255, 255, 255), kfront), width=max(2, round(2 * k)))
            lab = str(f_no)
            bb = d.textbbox((0, 0), lab, font=f_dot)
            d.text((kpx - (bb[2] - bb[0]) / 2, kpy - (bb[3] - bb[1]) / 2 - bb[1]), lab,
                   fill=(14, 20, 16, 255), font=f_dot)

        label = (f"{len(kfs)} keyframes  f{kfs[0]['f']}-{kfs[-1]['f']}  "
                 f"{'smooth' if smooth else 'linear'}")

    d.text((8 * k, 6 * k), label, fill=(255, 255, 100, 255), font=f_txt)
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
    # Metric geometry carries NaN where the model marked sky/invalid, so this
    # division produces non-finite values that numpy refuses to cast quietly.
    # They are dropped by `val` a line below either way, but casting them raises
    # a RuntimeWarning per frame; parking them far outside the frame keeps the
    # console clean without touching a single valid pixel.
    with np.errstate(invalid="ignore", divide="ignore"):
        uf = Xd[:, 0] / zt * fx_pix + cx
        vf = Xd[:, 1] / zt * fy_pix + cy
    ui = np.round(np.nan_to_num(uf, nan=-1e6, posinf=1e6, neginf=-1e6)).astype(int)
    vi = np.round(np.nan_to_num(vf, nan=-1e6, posinf=1e6, neginf=-1e6)).astype(int)
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
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        # Looking straight up or down: the view direction is parallel to
        # world_down, so their cross product carries no direction and the basis
        # collapses to a singular matrix (which then kills the warp with
        # "Singular matrix" when it is inverted). Any axis perpendicular to f
        # gives a valid frame; the world forward axis is the natural pick.
        right = np.cross(np.array([0.0, 0.0, 1.0]), f)
        rn = np.linalg.norm(right)
    right = right / (rn + 1e-9)
    down = np.cross(f, right)
    C = np.eye(4)
    C[:3, 0], C[:3, 1], C[:3, 2], C[:3, 3] = right, down, f, eye
    return C


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


def _unwrap_seq(degs):
    """Unwrap a sequence of angles so each step takes the short way round.

    Interpolating in unwrapped space is what keeps a move across the +-180 seam
    continuous (170 -> -170 becomes 170 -> 190); the result is wrapped back at
    the end."""
    out = [float(degs[0])]
    for d in degs[1:]:
        out.append(out[-1] + _wrap_deg(float(d) - out[-1]))
    return out


def _catmull(p0, p1, p2, p3, u):
    """Uniform Catmull-Rom segment: passes through p1 at u=0 and p2 at u=1.

    Chosen over a Bezier because it INTERPOLATES its control points -- every
    keyframe is hit exactly -- and it generalises to any number of points,
    unlike a quadratic Bezier which only lands on the middle point via a
    hand-fitted control point."""
    return 0.5 * ((2.0 * p1)
                  + (-p0 + p2) * u
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u * u * u)


def _seg_value(vals, seg, u, smooth):
    """Value between vals[seg] and vals[seg+1] at local position u in [0,1].

    smooth=False is a straight lerp. smooth=True is Catmull-Rom, which needs the
    two neighbouring points; at the ends they are reflected so the curve keeps
    its tangent instead of flattening. With only two keyframes Catmull-Rom is
    identical to the lerp, so the simple case is unaffected."""
    p1, p2 = vals[seg], vals[seg + 1]
    if not smooth or len(vals) < 3:
        return p1 + (p2 - p1) * u
    p0 = vals[seg - 1] if seg > 0 else p1 + (p1 - p2)
    p3 = vals[seg + 2] if seg + 2 < len(vals) else p2 + (p2 - p1)
    return _catmull(p0, p1, p2, p3, u)


def _parse_keyframes(raw, frame_count, default_vs=0.0, default_pivot=(0.0, 0.0, 1.05)):
    """Parse the `keyframes` widget into a sorted list of keyframe dicts.

    "vs" (vertical lens shift) and "px"/"py"/"pz" (the pivot) are OPTIONAL: a
    keyframe without them inherits the node's static widgets. A path written
    before those fields existed therefore gives every keyframe the same values,
    which is exactly what it used to do.

    Frame numbers are 1-based, matching how the clip reads to a user (frame 1 is
    the first frame, frame `frame_count` the last); the loop in build() converts.

    Empty input means "no camera move". Anything malformed raises ValueError with
    a message written for the user: a silently truncated or misread camera move is
    worse than a failed queue, and every bug class this format replaced was of the
    silently-wrong kind."""
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw or raw in ("[]", "null"):
        return []
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            "CrossViewWarp: 'keyframes' is not valid JSON (%s). Expected something like "
            '[{"f":1,"az":-30,"el":20,"dist":1.0},{"f":49,"az":45,"el":10,"dist":1.2}]' % exc
        ) from None
    if not isinstance(data, list):
        raise ValueError("CrossViewWarp: 'keyframes' must be a JSON list of keyframe objects.")

    out = []
    for i, kf in enumerate(data):
        if not isinstance(kf, dict):
            raise ValueError("CrossViewWarp: keyframe #%d is not an object." % i)
        try:
            f = int(round(float(kf["f"])))
            az, el, dist = float(kf["az"]), float(kf["el"]), float(kf["dist"])
            vs = float(kf.get("vs", default_vs))
            px = float(kf.get("px", default_pivot[0]))
            py = float(kf.get("py", default_pivot[1]))
            pz = float(kf.get("pz", default_pivot[2]))
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "CrossViewWarp: keyframe #%d needs numeric 'f', 'az', 'el' and 'dist'." % i
            ) from None
        if f < 1:
            raise ValueError(
                "CrossViewWarp: keyframe #%d sits at frame %d - frame numbers start at 1 "
                "(frame 1 is the first frame of the clip)." % (i, f))
        if f > frame_count:
            raise ValueError(
                "CrossViewWarp: keyframe #%d sits at frame %d, but this clip only has %d frames. "
                "The camera path was authored for a longer clip -- move that keyframe, or feed a "
                "longer clip." % (i, f, frame_count))
        out.append({"f": f, "az": az, "el": el, "dist": dist, "vs": vs,
                    "px": px, "py": py, "pz": pz})

    out.sort(key=lambda k: k["f"])
    seen = [k["f"] for k in out]
    if len(set(seen)) != len(seen):
        raise ValueError("CrossViewWarp: two keyframes share the same frame number.")
    return out


def _prepare_path(kfs):
    """Pre-split the keyframe list into the arrays the sampler reads (done once,
    not per frame). Azimuth is unwrapped here so every segment lookup stays cheap."""
    out = {"f": [k["f"] for k in kfs], "az": _unwrap_seq([k["az"] for k in kfs])}
    for key in ("el", "dist", "vs", "px", "py", "pz"):
        out[key] = [k[key] for k in kfs]
    return out


def _sample_path(path, frame, easing, smooth):
    """Camera pose at a 1-based frame number, as a dict.

    Returns az / el / dist / vs / px / py / pz - a dict rather than a tuple
    because the channel list has grown twice now, and positional unpacking at
    four call sites is how one of them quietly ends up reading the wrong one.

    Outside the keyframed span the pose is HELD rather than extrapolated, so a
    path covering only part of the clip simply stops moving -- that is what makes
    "swing for the first half, then hold" expressible at all.

    Easing is applied per segment, so it reads as a deceleration into each
    keyframe; combine it with interpolation='linear'. For one continuous flowing
    move use interpolation='smooth' with easing 'linear', otherwise the eased
    stop at every knot cancels the smoothing the spline exists to provide."""
    fs = path["f"]
    KEYS = ("az", "el", "dist", "vs", "px", "py", "pz")
    if frame <= fs[0]:
        return {k: path[k][0] for k in KEYS}
    if frame >= fs[-1]:
        return {k: path[k][-1] for k in KEYS}

    seg = 0
    for i in range(len(fs) - 1):
        if fs[i] <= frame <= fs[i + 1]:
            seg = i
            break
    u = _ease((frame - fs[seg]) / float(fs[seg + 1] - fs[seg]), easing)

    # Catmull-Rom overshoots between points, so clamp what has a physical limit:
    # past +-90 elevation the camera tips over the pole, and a distance <= 0 puts
    # the eye on the far side of the pivot.
    az = _wrap_deg(_seg_value(path["az"], seg, u, smooth))
    out = {
        "az": az,
        "el": float(np.clip(_seg_value(path["el"], seg, u, smooth), -90.0, 90.0)),
        "dist": float(np.clip(_seg_value(path["dist"], seg, u, smooth), 0.1, 3.0)),
        "vs": float(np.clip(_seg_value(path["vs"], seg, u, smooth), -1.0, 1.0)),
        "px": float(_seg_value(path["px"], seg, u, smooth)),
        "py": float(_seg_value(path["py"], seg, u, smooth)),
        # a pivot at or behind the camera has no orbit to define
        "pz": float(max(_seg_value(path["pz"], seg, u, smooth), 0.01)),
    }
    return out


def _quat_to_mat_xyzw(q):
    """Unit quaternion (x, y, z, w) -> rotation matrix, columns = rotated axes."""
    x, y, z, w = (float(c) for c in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# three.js world/camera (Y up, camera looks down -Z) -> this node's frame, which
# is OpenCV-style (y down, +z forward) because _warp_frame unprojects with
# [(u-cx)/fx*z, (v-cy)/fx*z, z]. Flipping y and z converts both the point and
# the camera basis; it is a rotation (det +1), not a mirror.
_T3 = np.diag([1.0, -1.0, -1.0])


def _fx(ci, height):
    """Focal in pixels from a camera_info's VERTICAL fov and zoom (square pixels)."""
    vfov = float(ci.get("fov") or 0.0) or 35.0
    zoom = float(ci.get("zoom") or 1.0) or 1.0
    return height / (2.0 * np.tan(np.radians(vfov) / 2.0)) * zoom


def _camera_info_pose(ci, height):
    """Load3DCamera dict -> (C_tgt 4x4 in this node's frame, fx in pixels).

    Removes the pose ESTIMATION entirely: instead of orbiting an guessed pivot,
    the camera goes exactly where the caller put it. `fov` in a camera_info is
    VERTICAL and `zoom` multiplies the focal length, unlike this node's
    horizontal `hfov`.
    """
    g = lambda d, k: float((d or {}).get(k, 0.0) or 0.0)
    eye3 = np.array([g(ci.get("position"), "x"), g(ci.get("position"), "y"),
                     g(ci.get("position"), "z")])
    q = ci.get("quaternion")
    if q:                                   # exact world rotation, roll included
        R3 = _quat_to_mat_xyzw((g(q, "x"), g(q, "y"), g(q, "z"),
                               float((q or {}).get("w", 1.0) or 1.0)))
    else:                                   # only a look-at target: no roll info
        tgt3 = np.array([g(ci.get("target"), "x"), g(ci.get("target"), "y"),
                         g(ci.get("target"), "z")])
        return _look_at(_T3 @ eye3, _T3 @ tgt3), _fx(ci, height)
    C = np.eye(4)
    # columns are (right, up, backward) in three.js; this node wants
    # (right, down, forward), hence the second flip on the right
    C[:3, :3] = _T3 @ R3 @ np.diag([1.0, -1.0, -1.0])
    C[:3, 3] = _T3 @ eye3
    return C, _fx(ci, height)


def _pose_to_orbit_angles(C, pivot):
    """Inverse of _orbit_C_tgt's placement: pose -> (az, el, dist) about `pivot`.

    Only used to draw the orbit_view diagram for a pose that did not come from
    an orbit (an explicit camera_info), so the picture shows where the camera
    ended up instead of angles nobody asked for. Round-trips exactly against
    _orbit_C_tgt for a pivot on the optical axis.
    """
    v = C[:3, 3] - pivot
    r = float(np.linalg.norm(v))
    p = float(np.linalg.norm(pivot))
    if r < 1e-9 or p < 1e-9:
        return 0.0, 0.0, 1.0
    d = v / r
    el = float(np.degrees(np.arcsin(np.clip(-d[1], -1.0, 1.0))))
    az = float(np.degrees(np.arctan2(d[0], -d[2])))
    return az, el, r / p


def _orbit_C_tgt(az_deg, el_deg, dist, pivot, aim=None):
    """Camera pose orbiting `pivot` at the requested az/el/dist, looking at `aim`.

    Mirrors the inline math that was in build(): angle signs are negated so
    +azimuth = camera orbits RIGHT, +elevation = camera RISES (OpenCV frame).

    `aim` defaults to the pivot, which is what this node always did - and which
    is measurably NOT what the training data does. The dataset builder orbits
    the camera about the subject but keeps it pointed at whatever camera A was
    pointed at, so the framing carries over instead of the subject snapping to
    the middle. Only 25% of the dataset (the `standard` family) has those two
    points coincide; across the rest camera B is off the pivot by a median 4 to
    31 degrees. Measured against the training warp on 12 scenes: aiming at the
    pivot scores 42.3, keeping the source aim scores 28.9, and the exact pose
    ceiling is 26.8 - so the aim is worth ~14 points and the pivot itself only
    ~3.
    """
    R_orbit = _rot_y(np.radians(-az_deg)) @ _rot_x(np.radians(-el_deg))
    eye = pivot + dist * (R_orbit @ (-pivot))
    return _look_at(eye, pivot if aim is None else aim)


# --- ComfyUI node ------------------------------------------------------------

class CrossViewWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames. Feed the same frames you send to the "
                    "depth node."}),
                "azimuth": ("FLOAT", {"default": -30.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Horizontal orbit angle (deg). Negative orbits LEFT, "
                    "positive RIGHT. The strongest control: reliable to about "
                    "+-45, usable to +-65, beyond that the hidden side is "
                    "mostly invented."}),
                "elevation": ("FLOAT", {"default": 20.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Vertical orbit angle (deg). Positive rises above the "
                    "subject, negative looks up from below. Weaker than azimuth "
                    "- the orbit is stronger horizontally."}),
                "distance": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Camera distance (1.0 = same as the source). Below 1 moves "
                    "closer, above 1 pulls back - and pulling back reveals area "
                    "the source never framed, so the magenta holes grow. The "
                    "current LoRA does not follow this reliably."}),
                "hfov": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Assumed horizontal field of view (deg). ~50 is a normal "
                    "lens; change it if the clip is clearly wide-angle or "
                    "telephoto. 0 takes it from moge_geometry, which runs about "
                    "10% short - set it yourself if you know the lens."}),
                # Renamed from head_bias, which named the use case rather than the
                # effect. Kept in the SAME widget position: widget values are
                # stored positionally, so a rename in place leaves saved
                # workflows intact (a converted-to-input link, which is stored by
                # name, is the one thing that does not survive).
                "vertical_shift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.02,
                    "tooltip": "Vertical lens shift, as a fraction of image height. "
                    "Positive moves the picture DOWN, i.e. the framing up, "
                    "which rescues a clipped head. The camera does not move, so "
                    "parallax is unchanged. Leave at 0 unless the framing needs "
                    "it."}),
                "depth_ratio": ("FLOAT", {"default": 6.0, "min": 1.5, "max": 1000.0, "step": 0.5,
                    "tooltip": "Max far/near depth ratio of the scene. Lower = flatter "
                    "relief and a cleaner warp; higher = more parallax but "
                    "messier on cluttered scenes. Close-up faces: 2.5-4; mid "
                    "shots: 4-8; deep/wide scenes: 8-16."}),
                "smooth_depth": ("BOOLEAN", {"default": False,
                    "tooltip": "Edge-aware depth smoothing before warping. Turn ON if the "
                    "warp has too many speckle holes; it trades a little "
                    "sharpness for cleaner disocclusion."}),
                "invert_depth": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip depth polarity (near<->far). Leave OFF for the "
                    "standard DA-V2 node. Turn ON only if the warp looks "
                    "inside-out - background moving like foreground, or the "
                    "subject caving in."}),
            },
            "optional": {
                "roll_lock": ("BOOLEAN", {"default": True,
                    "tooltip": "Retired and ignored. It rolled the camera to hold the subject's "
                    "projected lean, but the orbit introduces no roll to correct - a level "
                    "source gives a level target at every angle - and what it was reacting to is "
                    "keystone from elevation, which a roll cannot fix. Both training datasets "
                    "have exactly 0 camera roll."}),
                "pivot_override": ("BOOLEAN", {"default": True,
                    "tooltip": "Orbit around the explicit pivot below instead of an auto- "
                    "estimated one. ON by default, which is the recommended "
                    "setup."}),
                "pivot_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Pivot X in the source camera frame (+ = right), in depth "
                    "units. 0 = image centre. The camera aims at the pivot, so "
                    "moving this also reframes the shot - unless "
                    "keep_source_aim is ON, where it only moves the orbit "
                    "centre."}),
                "pivot_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Pivot Y (+ = down), in depth units. 0 = image centre. "
                    "Doubles as a reframe like pivot_x. Use vertical_shift to "
                    "reframe without moving the camera."}),
                "pivot_z": ("FLOAT", {"default": 1.05, "min": 0.01, "max": 1000.0, "step": 0.01,
                    "tooltip": "Pivot depth. ~1.05 sits on the nearest subject in the "
                    "middle of the frame; raise it to orbit something further "
                    "back. The orbit radius is distance * |pivot|, so this "
                    "widens the arc as well as moving its centre."}),
                # NOTE: new widgets MUST come after the original five optional
                # ones (roll_lock..pivot_z). ComfyUI stores widget_values
                # positionally in saved workflows, so inserting a widget in the
                # middle shifts every downstream value into the wrong slot and
                # breaks previously saved nodes on load.
                "use_keyframes": ("BOOLEAN", {"default": False,
                    "tooltip": "Animate the camera along the keyframes path instead of "
                    "holding one pose. Switched on for you when the first "
                    "keyframe is placed."}),
                "frame_count": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Retired and ignored. The clip length comes from the cached "
                    "preview."}),
                "keyframes": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Camera path as JSON - normally written for you by the KEY button on the "
                    "preview, or by right-clicking the orbit sphere. Each entry: 'f' is the "
                    "frame number counted from 1, 'az'/'el' are degrees, 'dist' is the "
                    "distance. Optional 'vs' (vertical shift) and 'px'/'py'/'pz' (the pivot) "
                    "inherit the static widgets when left out; px/py/pz need pivot_override ON. "
                    'Example: [{"f":1,"az":-30,"el":20,"dist":1.0},{"f":49,"az":45,"el":10,'
                    '"dist":1.2}] . '
                    "The pose is held before the first and after the last keyframe, so a path "
                    "may cover only part of the clip. Empty = no move."}),
                # Shape AND timing in one control. "smooth" is not an easing -
                # _ease falls through to identity for it - it is the value that
                # drives the hidden `interpolation` widget, which keeps its slot
                # because widget values are positional.
                "interp_motion": (["linear", "ease_in", "ease_out", "ease_in_out", "smooth"],
                                  {"default": "linear",
                    "tooltip": "Shape and timing of the camera path. linear = straight "
                    "legs at constant speed. ease_in / ease_out / ease_in_out = "
                    "the same legs with the camera settling into every "
                    "keyframe. smooth = a Catmull-Rom spline gliding through "
                    "them with no corner. Only used while use_keyframes is ON."}),
                "interpolation": (["linear", "smooth"], {"default": "linear",
                    "tooltip": "Driven by interp_motion."}),
                "keep_source_aim": ("BOOLEAN", {"default": False,
                    "tooltip": "Which point the orbiting camera looks at. OFF: it looks at the "
                    "pivot, so the pivot sits dead centre and moving it reframes the shot. ON: it "
                    "looks where the SOURCE camera looked, so your original framing is kept and "
                    "the pivot is only the orbit centre. Needs pivot_x or pivot_y off zero to do "
                    "anything - on the optical axis both aims are the same point. ON is what the "
                    "training data does and scores 28.9 against 42.3 measured on the training "
                    "warps; OFF by default for compatibility, turn it ON for new work."}),
                # Sockets, not widgets, so they take no widget_values slot and
                # cannot shift the ones above in saved workflows. `depth` moved
                # here from `required` so a metric-geometry graph does not have
                # to wire a depth image it will not use; saved workflows already
                # have it connected, so nothing breaks.
                "depth": ("IMAGE", {
                    "tooltip": "Depth map for the same frames, from a Depth Anything V2 "
                    "node. Brightness = depth; polarity is handled by "
                    "invert_depth. Leave unconnected if you are feeding "
                    "moge_geometry instead."}),
                "moge_geometry": ("MOGE_GEOMETRY", {
                    "tooltip": "Metric geometry from Run MoGe Inference, used instead of "
                    "the depth image. Depth then arrives in real metres, so "
                    "depth_ratio, invert_depth and smooth_depth no longer "
                    "apply, and pivot and distance become real distances. "
                    "Measured against ground truth it roughly halves the warp "
                    "error. hfov comes from its intrinsics when hfov is 0."}),
                "camera_info": ("LOAD3D_CAMERA", {
                    "tooltip": "Exact target camera, from Create Camera Info or a Load 3D "
                    "viewport. When connected the pose comes from it, and the "
                    "orbit controls - angles, distance, hfov, the "
                    "pivot and the keyframe path - are all ignored."}),
                # APPENDED, never inserted: widget values are positional. The
                # sockets above take no slot, so these land after keep_source_aim.
                "preview_size": ("INT", {"default": 512, "min": 128, "max": 768, "step": 32,
                    "tooltip": "Longest side the clip is cached at for the live preview. "
                    "The warp is scale-invariant, so this trades sharpness and "
                    "memory for frame rate: about 1 MB per frame and 18 fps at "
                    "384, 1.8 MB and 11 fps at 512. Takes effect on the next "
                    "run."}),
                "frame_index": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "Playhead for the live preview, counted from 1. The scrub "
                    "bar under the preview drives it; it does not affect the "
                    "output and needs no re-run."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
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
    # An execution ends at an output node, which is what lets the frontend offer
    # "Run Branch" on hover - run the warp alone, look at the preview, skip the
    # generation. The cost is that queueing the graph always runs this node, but
    # ComfyUI skips it while the inputs are unchanged.
    OUTPUT_NODE = True

    def build(self, frames, azimuth, elevation, distance, hfov, vertical_shift, depth_ratio, smooth_depth, invert_depth,
              roll_lock=True, pivot_override=True, pivot_x=0.0, pivot_y=0.0, pivot_z=1.05,
              use_keyframes=False, frame_count=0, keyframes="", interp_motion="linear",
              interpolation="linear", keep_source_aim=False,
              camera_info=None, moge_geometry=None, depth=None,
              preview_size=384, frame_index=1, unique_id=None):
        # frames: [B,H,W,C] float [0,1]; depth: [B,H,W,C] brightness [0,1]
        rgb = (frames.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)   # [B,H,W,3]
        B, H, W = rgb.shape[:3]
        if moge_geometry is None and depth is None:
            raise ValueError("CrossViewWarp: connect either `depth` (a depth image) or "
                             "`moge_geometry` (metric geometry from Run MoGe Inference).")
        metric_z = None
        if moge_geometry is not None:
            # Metric metres straight from the model - no normalisation, so no
            # scale to guess. `mask` marks sky/invalid, which becomes a hole
            # rather than being clamped into the far plane by the percentile cut.
            md = moge_geometry.get("depth")
            metric_z = md.detach().float().cpu().numpy()
            mm = moge_geometry.get("mask")
            if mm is not None:
                metric_z = np.where(mm.detach().cpu().numpy(), metric_z, np.nan)
            metric_z = np.where(np.isfinite(metric_z) & (metric_z > 0), metric_z, np.nan)
            if depth is not None:
                logging.warning("CrossViewWarp: both depth and moge_geometry are connected; "
                                "using moge_geometry (metric) and ignoring the depth image.")
        depth_bhw = (depth.clamp(0, 1).mean(dim=-1).cpu().numpy()
                     if metric_z is None else metric_z)
        if smooth_depth and metric_z is None:
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
        # metric depth IS z; the mapping exists only to invent a scale for a
        # relative map, so applying it here would throw the metres away
        z = metric_z if metric_z is not None else _depth_to_z(depth_bhw, invert_depth, depth_ratio)

        if 0.0 < hfov < 20.0:
            raise ValueError(
                "CrossViewWarp: hfov=%.1f is not a usable field of view. Use 0 to read the "
                "focal length from moge_geometry, or a real lens angle of 20 degrees or more."
                % hfov)
        fx = W / (2.0 * np.tan(np.radians(hfov) / 2.0)) if hfov > 0 else None
        if fx is None:
            # hfov 0 = take it from MoGe's own intrinsics. Measured: MoGe runs
            # about 10% short on focal, so this is the fallback, not the default
            # choice - a known lens beats it.
            K = (moge_geometry or {}).get("intrinsics")
            fx = (float(K[0][0, 0]) * W if K is not None
                  else W / (2.0 * np.tan(np.radians(50.0) / 2.0)))
            logging.info("CrossViewWarp: hfov=0, using fx=%.1f px (%.1f deg)", fx,
                         np.degrees(2 * np.arctan(W / (2 * fx))))
        cc = W / 2.0
        cch = H / 2.0

        # An explicit camera_info replaces the whole pose ESTIMATION - pivot,
        # orbit and roll lock all exist only to guess a pose from angles, and
        # measured against ground truth that guess is what costs this node most
        # of its accuracy. When one is supplied the camera simply goes where it
        # says.
        ci_C = ci_fx = None
        if camera_info:
            ci_C, ci_fx = _camera_info_pose(camera_info, H)
            if ci_fx:
                fx = ci_fx
            if use_keyframes:
                logging.warning("CrossViewWarp: camera_info is connected, so the keyframe "
                                "path is ignored - the pose comes from camera_info.")

        # subject pivot = central-region foreground centroid on frame 0
        z0 = z[0]
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        central = np.zeros_like(z0, bool)
        central[H // 8: 4 * H // 5, W // 5: 4 * W // 5] = True
        fin = np.isfinite(z0) & (z0 > 0) & central
        fin &= z0 < np.percentile(z0[fin], 95.0)
        Xw0 = np.stack([(uu - cc) / fx * z0, (vv - cch) / fx * z0, z0], -1)
        # pivot = median 3D point of the trimmed central region (scene-level
        # aim, matching the training warps' real-camera framing; a nearest-
        # cluster pivot would zoom the orbit camera into the subject)
        pivot = np.median(Xw0[fin], axis=0)
        if pivot_override:
            pivot = np.array([pivot_x, pivot_y, pivot_z], dtype=np.float64)

        # Where the orbiting camera LOOKS. The source camera sits at the origin
        # looking down +Z in this frame, so its optical axis IS +Z and keeping
        # its aim means targeting a point on that axis; the only choice left is
        # how far along.
        #
        # The pivot's own distance, and NOT the metric depth at the frame
        # centre. Taking the centre depth was the obvious refinement and it
        # measured worse - 34.2 against 28.9 on the same twelve scenes - because
        # the middle of the frame is usually the subject, which sits nearer than
        # the point a real camera is actually aimed at.
        def _aim_of(p):
            return (np.array([0.0, 0.0, max(float(np.linalg.norm(p)), 1e-3)])
                    if keep_source_aim else None)

        def _pivot_of(sample):
            """The pivot a sampled pose asks for.

            Per-keyframe pivots only mean anything while the pivot is being
            GIVEN: with pivot_override off the node estimates one from frame 0,
            and there is nothing for a path to animate."""
            if not pivot_override:
                return pivot
            return np.array([sample["px"], sample["py"], sample["pz"]], dtype=np.float64)

        C_ref = np.eye(4)

        # "Base" pose drives roll-lock estimation AND the orbit_view preview.
        # For a single-pose run this is just (az, el, dist); for a keyframed
        # run we use the move's midpoint so the upright correction is the
        # average of where the camera actually lives across the clip.
        if use_keyframes and frame_count and frame_count != B:
            # `frame_count` only tells the widget where to space keyframes; the clip
            # in hand is the authority. Worth saying out loud, because a stale value
            # here means the path was laid out for a different length.
            logging.warning(
                "CrossViewWarp: frame_count is %d but this clip has %d frames - the camera path "
                "was spaced for a different length. Update frame_count to re-space new keyframes.",
                frame_count, B)
        # A keyframe with no px/py/pz inherits this, so older paths are unchanged.
        kfs = (_parse_keyframes(keyframes, B, vertical_shift, tuple(float(c) for c in pivot))
               if use_keyframes else [])
        path = _prepare_path(kfs) if kfs else None
        keyframing = bool(len(kfs) >= 2 and B > 1) and ci_C is None
        # "smooth" on either: the widget carries it, and interp_motion is what
        # the UI actually offers, so a headless call using only that still works.
        smooth_path = "smooth" in (interpolation, interp_motion)
        if keyframing:
            # middle frame, 1-based: frames run 1..B
            mid = _sample_path(path, (B + 1) // 2, interp_motion, smooth_path)
        elif kfs:
            # a single keyframe is not a move -- treat it as the static pose
            mid = kfs[0]
        else:
            mid = {"az": azimuth, "el": elevation, "dist": distance, "vs": vertical_shift,
                   "px": float(pivot[0]), "py": float(pivot[1]), "pz": float(pivot[2])}
        mid_az, mid_el, mid_dist, mid_vs = mid["az"], mid["el"], mid["dist"], mid["vs"]
        mid_pivot = _pivot_of(mid)
        C_tgt = (ci_C if ci_C is not None
                 else _orbit_C_tgt(mid_az, mid_el, mid_dist, mid_pivot, _aim_of(mid_pivot)))

        # principal point stays at the image centre (training warps use real
        # camera intrinsics, no lens shift); vertical_shift is a manual lens
        # shift only (default 0 = off) - it translates the rendered frame and
        # leaves the camera where it is, so it cannot change parallax
        cx_eff = cc
        # mid_vs is the static widget when no path is running, so this is
        # unchanged for an ordinary run; a keyframed one overrides it per frame.
        cy_eff = cch + mid_vs * H

        pbar = ProgressBar(B) if ProgressBar is not None else None
        warp_frames = []
        for i in range(B):
            if keyframing:
                # i is a 0-based buffer index; keyframes are numbered from 1
                fr = _sample_path(path, i + 1, interp_motion, smooth_path)
                fr_vs = fr["vs"]
                # The pivot is part of the pose, so a move can swing around one
                # thing then another. Radius is dist * |pivot|, so this animates
                # the parallax too.
                fr_pivot = _pivot_of(fr)
                C_tgt_i = _orbit_C_tgt(fr["az"], fr["el"], fr["dist"], fr_pivot,
                                       _aim_of(fr_pivot))
                # the lens shift is part of the framing, so it keyframes with the
                # rest of the pose rather than staying pinned to the widget
                cy_i = cch + fr_vs * H
            else:
                C_tgt_i = C_tgt
                cy_i = cy_eff
            warp_frames.append(_warp_frame(rgb[i], z[i], C_ref, C_tgt_i, fx, 2, cx_eff, cy_i))
            if pbar is not None:
                pbar.update(1)   # advance the node's progress bar one frame
        warp = np.stack(warp_frames, 0)  # [B,H,W,3] uint8

        warp_t = torch.from_numpy(warp.astype(np.float32) / 255.0)
        # keyframed runs document the whole path; a static run documents the pose.
        # A camera_info pose never came from an orbit, so read its angles back out
        # rather than drawing the az/el widgets it overrode.
        if ci_C is not None:
            mid_az, mid_el, mid_dist = _pose_to_orbit_angles(ci_C, mid_pivot)
        orbit = _orbit_view_image(mid_az, mid_el, mid_dist,
                                  kfs=kfs if keyframing else None, smooth=smooth_path)
        orbit_t = torch.from_numpy(orbit.astype(np.float32) / 255.0)[None]  # [1,H,W,3]
        # The live preview keeps a downscaled copy so the browser can re-warp
        # without another run. Left None, the node behaves exactly as before.
        info = None
        if _PREVIEW_SINK is not None and unique_id is not None:
            try:
                info = _PREVIEW_SINK(unique_id, frames, depth, moge_geometry, camera_info,
                                     int(preview_size))
            except Exception:
                # a preview that cannot be cached must never fail the render
                logging.exception("CrossViewWarp: could not cache the live preview")

        if info is None:
            return (warp_t, orbit_t)
        # A `ui` payload is what makes ComfyUI emit an "executed" message, which
        # is the widget's only signal that the cache is fresh.
        return {"ui": {"crossview_preview": [info]}, "result": (warp_t, orbit_t)}


_ID = f"CrossViewWarp{NODE_SUFFIX}"
_NAME = "CrossView Warp" + (f" [{NODE_SUFFIX}]" if NODE_SUFFIX else "")
NODE_CLASS_MAPPINGS = {_ID: CrossViewWarp}
NODE_DISPLAY_NAME_MAPPINGS = {_ID: _NAME}
