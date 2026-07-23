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
    distF = float(np.clip(0.45 + 0.55 * distance, 0.45, 1.3))
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

    `ratio` bounds the far/near depth ratio. The old unbounded mapping
    (1/(dn+1e-3), ~1000x) exaggerated relative-depth relief on close-ups
    (a face got ~30% relief vs a real ~10%), turning a 30-deg orbit into a
    90-deg-looking, smeared warp. Bounded: z = 1/(1/ratio + (1-1/ratio)*dn).
    """
    d = depth_bhw.astype(np.float64)
    if invert:
        d = -d
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    dn = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    r = max(float(ratio), 1.01)
    return 1.0 / (1.0 / r + (1.0 - 1.0 / r) * dn)


# --- ComfyUI node ------------------------------------------------------------

class CrossViewWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames = camera A (the source view). "
                    "Feed the same frames you send to the Depth Anything V2 node."}),
                "depth": ("IMAGE", {
                    "tooltip": "Depth map for the SAME frames, from a Depth Anything V2 node "
                    "(DA-V2 Large). Brightness = depth; polarity is handled by invert_depth."}),
                "azimuth": ("FLOAT", {"default": 20.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Horizontal orbit angle (deg). Negative = orbit LEFT, positive = "
                    "RIGHT. This is the strongest control. For close-up faces keep it small "
                    "(15-25); wide/full-body shots take larger angles (up to ~90)."}),
                "elevation": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Vertical orbit angle (deg). Positive = camera rises (looks down "
                    "on the subject), negative = looks up. NOTE: the effect is weaker/subtler "
                    "than azimuth - the orbit is stronger horizontally."}),
                "distance": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05,
                    "tooltip": "Camera distance from the subject (1.0 = same as source). "
                    "Below 1 = move closer / zoom in; above 1 = pull back. Extreme values "
                    "enlarge the disoccluded (magenta) holes."}),
                "hfov": ("FLOAT", {"default": 50.0, "min": 20.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Horizontal field of view (deg) = the assumed lens. ~50 suits a "
                    "normal lens; lower = more telephoto (flatter), higher = wider (more "
                    "perspective distortion). Keep near the source clip's real FOV."}),
                "splat": ("INT", {"default": 2, "min": 0, "max": 4,
                    "tooltip": "Point-splat radius in pixels when scattering the warp. Higher = "
                    "fewer speckle holes but a slightly blurrier/thicker warp; 0 = raw points "
                    "(most holes). 2 is a good default."}),
                "head_bias": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.02,
                    "tooltip": "Manual vertical framing shift (fraction of height; + = shift view "
                    "up). Default 0 = OFF, matching the training-data warps (real cameras, no "
                    "recentring). Use only if the subject's head gets clipped."}),
                "depth_ratio": ("FLOAT", {"default": 1.5, "min": 1.5, "max": 1000.0, "step": 0.5,
                    "tooltip": "Max far/near depth ratio of the scene. Lower = flatter relief, "
                    "cleaner warp. Close-up faces: 2.5-4; mid shots: 4-8; deep/wide scenes: "
                    "8-16. Very high values reproduce the old unbounded behaviour (smeared, "
                    "over-rotated-looking warps on close-ups)."}),
                "metric_depth": ("BOOLEAN", {"default": False,
                    "tooltip": "The depth input is RAW METRIC depth in meters (larger = farther), "
                    "e.g. from the CrossView Metric Depth node with a metric DA-V2 checkpoint. "
                    "Uses true 1/z geometry directly - depth_ratio and invert_depth are ignored. "
                    "This matches the scale-fitted depth of the training data best: the subject "
                    "keeps its real (thin) depth slab AND the scene keeps its real parallax."}),
                "smooth_depth": ("BOOLEAN", {"default": True,
                    "tooltip": "EXPERIMENTAL: edge-aware depth smoothing (RGB-guided filter + "
                    "median) before warping. Goal: coherent training-like disocclusion holes "
                    "instead of shredded speckle (dataset warps: ~55 large magenta regions, 0% "
                    "pinholes; unsmoothed ours: ~586 fragments). OFF = original behaviour."}),
                "invert_depth": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip depth polarity (near<->far). Leave FALSE for the standard "
                    "ComfyUI DA-V2 node. Turn ON only if the warp looks inside-out (background "
                    "moves like foreground / the subject caves in)."}),
            },
            "optional": {
                "roll_lock": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep the subject's in-image lean identical to the source (default). "
                    "Turn OFF for exact-pose replication experiments - real camera pairs have "
                    "their own roll."}),
                "pivot_override": ("BOOLEAN", {"default": True,
                    "tooltip": "EXPERT: orbit around an explicit 3D pivot instead of the "
                    "auto-estimated scene point. Coordinates are in the SOURCE camera's frame "
                    "(x right, y down, z forward), in the depth field's units."}),
                "pivot_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "pivot_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "pivot_z": ("FLOAT", {"default": 1.05, "min": 0.01, "max": 1000.0, "step": 0.01}),
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

    def build(self, frames, depth, azimuth, elevation, distance, hfov, splat, head_bias, depth_ratio, metric_depth, smooth_depth, invert_depth,
              roll_lock=True, pivot_override=False, pivot_x=0.0, pivot_y=0.0, pivot_z=2.5):
        # frames: [B,H,W,C] float [0,1]; depth: [B,H,W,C] - [0,1] brightness for
        # the relative path, raw meters for the metric path (no clamp there!)
        rgb = (frames.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)   # [B,H,W,3]
        B, H, W = rgb.shape[:3]
        if metric_depth:
            depth_bhw = depth.mean(dim=-1).cpu().numpy()                    # meters
        else:
            depth_bhw = depth.clamp(0, 1).mean(dim=-1).cpu().numpy()        # brightness
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
        if metric_depth:
            # true 1/z geometry, no normalisation heuristics. metric+invert:
            # the input is raw DISPARITY (1/depth) - useful because 8-bit
            # encodings keep near-depth precision in disparity space.
            if invert_depth:
                z = 1.0 / np.clip(depth_bhw.astype(np.float64), 1e-4, None)
            else:
                z = np.clip(depth_bhw.astype(np.float64), 1e-2, None)
        else:
            z = _depth_to_z(depth_bhw, invert_depth, depth_ratio)          # [B,H,W]

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
        # angles are negated so that + azimuth = camera orbits RIGHT and
        # + elevation = camera RISES (OpenCV frame: x right, y down, z forward)
        R_orbit = _rot_y(np.radians(-azimuth)) @ _rot_x(np.radians(-elevation))
        eye = pivot + distance * (R_orbit @ (-pivot))
        C_tgt = _look_at(eye, pivot)

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
        if roll_lock and th_src is not None and th_tgt is not None:
            droll = float(np.clip(_wrap(th_tgt - th_src), -np.radians(35), np.radians(35)))
            err0 = abs(_wrap(th_tgt - th_src))
            for cand in (droll, -droll):
                C_try = C_tgt.copy()
                C_try[:3, :3] = _rodrigues(C_tgt[:3, 2], cand) @ C_tgt[:3, :3]
                th_try = _lean_in(C_try)
                if th_try is not None and abs(_wrap(th_try - th_src)) < err0 - 1e-6:
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
            warp_frames.append(_warp_frame(rgb[i], z[i], C_ref, C_tgt, fx, int(splat), cx_eff, cy_eff))
            if pbar is not None:
                pbar.update(1)   # advance the node's progress bar one frame
        warp = np.stack(warp_frames, 0)  # [B,H,W,3] uint8

        warp_t = torch.from_numpy(warp.astype(np.float32) / 255.0)
        orbit = _orbit_view_image(azimuth, elevation, distance)
        orbit_t = torch.from_numpy(orbit.astype(np.float32) / 255.0)[None]  # [1,H,W,3]
        return (warp_t, orbit_t)


class CrossViewMetricDepth:
    """Raw (un-normalised) DA-V2 depth. The stock DepthAnything_V2 node min-max
    normalises PER FRAME, which destroys metric meters AND makes the scale
    wobble frame to frame. This node reuses the same loaded model (wire the
    DownloadAndLoadDepthAnythingV2Model output in, pick a *metric* checkpoint,
    e.g. depth_anything_v2_metric_hypersim_vitl) but returns the raw output:
    METERS for metric checkpoints. Feed into CrossViewWarp with metric_depth=ON.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "da_model": ("DAMODEL", {"tooltip": "From DownloadAndLoadDepthAnythingV2Model - "
                                     "use a *metric* checkpoint (hypersim = indoor/general 20m, "
                                     "vkitti = outdoor/driving 80m)."}),
            "images": ("IMAGE", {"tooltip": "Same frames you feed to CrossViewWarp."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("raw_depth",)
    OUTPUT_TOOLTIPS = ("Raw depth, meters (larger = farther), NOT normalised - values exceed 1. "
                       "Connect to CrossViewWarp.depth with metric_depth=ON.",)
    FUNCTION = "process"
    CATEGORY = "CrossView"

    def process(self, da_model, images):
        import torch.nn.functional as F
        from contextlib import nullcontext
        from torchvision import transforms
        import comfy.model_management as mm
        model = da_model["model"]
        dtype = da_model.get("dtype", torch.float32)
        device = mm.get_torch_device()
        B, H, W, _ = images.shape
        imgs = images.permute(0, 3, 1, 2)
        H14, W14 = H - (H % 14), W - (W % 14)
        if H14 != H or W14 != W:
            imgs = F.interpolate(imgs, size=(H14, W14), mode="bilinear")
        imgs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imgs)
        pbar = ProgressBar(B) if ProgressBar is not None else None
        out = []
        model.to(device)
        autocast = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.no_grad():
            with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast else nullcontext():
                for img in imgs:
                    d = model(img.unsqueeze(0).to(device))     # raw: meters (metric ckpt)
                    out.append(d.cpu().float())
                    if pbar is not None:
                        pbar.update(1)
        model.to(mm.unet_offload_device())
        d = torch.cat(out, dim=0)                              # [B,H',W']
        if d.shape[-2] != H or d.shape[-1] != W:
            d = F.interpolate(d.unsqueeze(1), size=(H, W), mode="bilinear")[:, 0]
        return (d.unsqueeze(-1).repeat(1, 1, 1, 3),)


NODE_CLASS_MAPPINGS = {"CrossViewWarp": CrossViewWarp,
                       "CrossViewMetricDepth": CrossViewMetricDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"CrossViewWarp": "CrossView Warp (video -> warp)",
                              "CrossViewMetricDepth": "CrossView Metric Depth (raw DA-V2)"}
