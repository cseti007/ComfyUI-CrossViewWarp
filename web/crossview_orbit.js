// CrossView Orbit — 3D sphere picker widget for the CrossViewWarp node.
//
// DOM-widget architecture (KJNodes editor pattern): a real <canvas> element is
// added via node.addDOMWidget and receives NATIVE pointer events — no reliance
// on LiteGraph's legacy widget-mouse routing.
//
// Full sphere around the subject; the HOME point (the source camera = the
// viewer) faces the screen. Coverage zones are concentric around home by
// angular distance g = acos(cos(el)*cos(az)) — the same forward-angle metric
// the training dataset uses. Interactions:
//   - drag the camera handle  -> set azimuth/elevation (snaps to 5 deg)
//   - drag empty space        -> rotate the VIEW (arcball)
//   - mouse wheel             -> distance (0.2..3.0)
//   - double click            -> reset the view rotation
// The azimuth/elevation/distance number widgets stay the source of truth.

import { app } from "../../scripts/app.js";

console.log("[CrossView Orbit] module loaded");

// Anisotropic coverage zones from MEASURED dataset stats: azimuth is wide
// (+-45 well trained), elevation is narrow, and DOWNWARD (camera below the
// subject, looking up) is the weakest direction -> red kicks in sooner there.
// Ellipse half-axes: [azimuth, elevation-up, elevation-down] in degrees.
const ZONE_GREEN = [45, 30, 15];
const ZONE_YELLOW = [65, 40, 25];
const C_GREEN = [80, 200, 120], C_YELLOW = [230, 200, 90], C_RED = [225, 95, 95];
const SNAP_DEG = 5;
const HANDLE_R = 14;
const WIDGET_H = 304;

function inEllipse(az, el, [A, Eup, Edn]) {
  const an = az / A;
  const en = el >= 0 ? el / Eup : el / Edn;
  return an * an + en * en <= 1;
}
function zoneColor(az, el) {
  if (inEllipse(az, el, ZONE_GREEN)) return C_GREEN;
  if (inEllipse(az, el, ZONE_YELLOW)) return C_YELLOW;
  return C_RED;
}
function d2r(x) { return (x * Math.PI) / 180; }
function r2d(x) { return (x * 180) / Math.PI; }

// Shortest-path (wrap-aware) linear interpolation between two angles in degrees.
function lerpAngle(a, b, t) {
  let diff = ((b - a + 540) % 360) - 180;
  return a + diff * t;
}

// Distance ratio -> canvas radius multiplier. Piecewise so dist=1.0 lands on
// the shell: dist<=1 spreads 0.45..1.0 (inside = closer), dist>1 spreads
// 1.0..1.5 so the marker keeps moving all the way up to dist=3.0 (the old
// 1.3 clamp froze it past ~1.55).
function distScale(dist) {
  const d = Number(dist);
  if (d <= 1) return 0.45 + 0.55 * d;
  return 1.0 + 0.25 * (d - 1.0);
}

// Quadratic Bezier through 3 scalar values; pc = 2*b - 0.5*(a+c) makes the
// curve pass through b at t=0.5 (otherwise b would be just the off-curve
// control point).
function bezier(a, b, c, t) {
  const pc = 2 * b - 0.5 * (a + c);
  const mt = 1 - t;
  return mt * mt * a + 2 * mt * t * pc + t * t * c;
}
// Bezier for azimuth angles: unwrap b and c relative to the running tangent
// so the control-point math stays continuous across the +-180 seam, then wrap
// the result back to (-180, 180].
function bezierAngle(aDeg, bDeg, cDeg, t) {
  const bUnwrap = aDeg + (((bDeg - aDeg + 540) % 360) - 180);
  const cUnwrap = bUnwrap + (((cDeg - bUnwrap + 540) % 360) - 180);
  const val = bezier(aDeg, bUnwrap, cUnwrap, t);
  return ((val + 180) % 360 + 360) % 360 - 180;
}

// sphere point for (az, el); home (0,0) faces the viewer (-y toward screen)
function spherePt(azDeg, elDeg) {
  const a = d2r(azDeg), e = d2r(elDeg);
  return [Math.cos(e) * Math.sin(a), -Math.cos(e) * Math.cos(a), Math.sin(e)];
}

class OrbitView {
  constructor() {
    this.viewYaw = 0.24;
    this.viewTilt = 0.20;
  }
  rot(p) {
    const [x, y, z] = p;
    const cyw = Math.cos(this.viewYaw), syw = Math.sin(this.viewYaw);
    const ct = Math.cos(this.viewTilt), st = Math.sin(this.viewTilt);
    const xr = x * cyw + y * syw;
    const yr = -x * syw + y * cyw;
    const yv = yr * ct - z * st;
    const zv = yr * st + z * ct;
    return [xr, yv, zv]; // yv < 0 => front (toward viewer)
  }
  unproject(px, py, R) {
    const xr = px / R, zv = -py / R;
    const rr = xr * xr + zv * zv;
    if (rr > 1) return null;
    const yv = -Math.sqrt(1 - rr);
    const ct = Math.cos(this.viewTilt), st = Math.sin(this.viewTilt);
    const yr = yv * ct + zv * st;
    const z = -yv * st + zv * ct;
    const cyw = Math.cos(this.viewYaw), syw = Math.sin(this.viewYaw);
    const x = xr * cyw - yr * syw;
    const y = xr * syw + yr * cyw;
    const el = r2d(Math.asin(Math.max(-1, Math.min(1, z))));
    const az = r2d(Math.atan2(x, -y));
    return [az, el];
  }
}

// Static sphere, GIZMO STYLE (user-picked variant B): flat disc, a subtle
// coverage hint only in the trained band, equator + meridian rings, outline.
function renderSphere(ctx, view, cx, cy, R) {
  ctx.fillStyle = "rgba(70,74,86,0.25)";
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill();
  const STEP = 15;
  for (let a0 = -70; a0 < 70; a0 += STEP) {
    for (let e0 = -30; e0 < 45; e0 += STEP) {
      const col = zoneColor(a0 + STEP / 2, e0 + STEP / 2);
      if (col === C_RED) continue;
      const quad = [];
      let dep = 0;
      for (const [aa, ee] of [[a0, e0], [a0 + STEP, e0], [a0 + STEP, e0 + STEP], [a0, e0 + STEP]]) {
        const [xr, yv, zv] = view.rot(spherePt(aa, ee));
        quad.push([cx + R * xr, cy - R * zv]);
        dep = yv;
      }
      if (dep >= 0) continue;   // front hemisphere only for the hint
      ctx.beginPath();
      ctx.moveTo(quad[0][0], quad[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(quad[i][0], quad[i][1]);
      ctx.closePath();
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},0.13)`;
      ctx.fill();
    }
  }
  ctx.lineWidth = 1.5;
  const P = (a, e) => {
    const [xr, yv, zv] = view.rot(spherePt(a, e));
    return [cx + R * xr, cy - R * zv, yv];
  };
  const ring = (pts) => {
    for (let i = 0; i < pts.length - 1; i++) {
      const [x1, y1, f1] = pts[i], [x2, y2, f2] = pts[i + 1];
      ctx.strokeStyle = f1 < 0 && f2 < 0 ? "rgba(200,204,216,0.5)" : "rgba(120,124,138,0.15)";
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
  };
  const eq = []; for (let a = -180; a <= 180; a += 8) eq.push(P(a, 0)); ring(eq);
  const mer = []; for (let e = -90; e <= 90; e += 8) mer.push(P(0, e)); ring(mer);
  ctx.strokeStyle = "rgba(190,190,205,0.55)";
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
}

// clickable snap points on the sphere: [az, el, label]
const SNAPS = [[0, 0, "F"], [-45, 0, "L"], [45, 0, "R"], [-90, 0, ""], [90, 0, ""], [0, 30, "H"], [0, -15, "Lo"]];

function getW(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

class OrbitEditor {
  constructor(node, canvas) {
    this.node = node;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.view = new OrbitView();
    this.drag = null;      // "cam" | "view"
    this.lastPos = null;
    this.handle = null;    // [x, y] canvas coords
    this.cache = null;
    this.cacheKey = "";
    this._renderKey = "";

    // S+click keyframe state: kfA / kfB / kfC are {az, el, dist} or null.
    // kf*Pos are canvas-space centres written by render() and read by
    // onKeyframeClick / onWheel / drag hit detection.
    this.kfA = null;
    this.kfB = null;
    this.kfC = null;
    this.kfAPos = null;
    this.kfBPos = null;
    this.kfCPos = null;
    // abc_smooth mirror of the Python widget: drives the A->B->C arc shape
    // (false = piecewise linear, true = quadratic Bezier through B).
    this.abcSmooth = false;
    // use_keyframes mirror: when OFF the markers are rendered at 20% alpha so
    // the user can see the keyframed move is disabled without losing the
    // placements they made.
    this.useKeyframes = false;
    this.sHeld = false;    // 'S' modifier currently down (only counts when
    // the pointer is over our canvas, so ComfyUI's own 'S' shortcut still
    // works elsewhere)
    this._mouseOver = false;

    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture?.(e.pointerId);
      this.onDown(e);
    });
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener("dblclick", () => {
      this.view.viewYaw = 0.24; this.view.viewTilt = 0.20;
      this.render(true);
    });
    canvas.addEventListener("mouseenter", () => { this._mouseOver = true; });
    canvas.addEventListener("mouseleave", () => { this._mouseOver = false; this.sHeld = false; });
    this._onKeyDown = (e) => {
      if (!this.canvas.isConnected) { document.removeEventListener("keydown", this._onKeyDown); return; }
      if (!this._mouseOver) return;
      if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
      if (e.key === "s" || e.key === "S") {
        this.sHeld = true;
        e.preventDefault();
      }
    };
    this._onKeyUp = (e) => {
      if (e.key === "s" || e.key === "S") this.sHeld = false;
    };
    document.addEventListener("keydown", this._onKeyDown);
    document.addEventListener("keyup", this._onKeyUp);

    this._onMove = (e) => this.onMove(e);
    this._onUp = () => this.onUp();

    // _writeKfWidgets runs now so the C sentinel is in place on a fresh node
    // before any prompt can be queued. _restoreKeyframes is deferred to the
    // first render — ComfyUI runs onNodeCreated (this constructor) BEFORE
    // configure() loads the saved widget values, so reading widgets here
    // would see defaults and miss the saved markers. By render time (next
    // animation frame) configure has run and the saved values are visible.
    this._writeKfWidgets();
    this._needsRestore = true;

    // repaint when the number widgets change externally (cheap change-detect)
    this._raf = () => {
      if (!this.canvas.isConnected) return;   // node removed -> stop
      this.render(false);
      requestAnimationFrame(this._raf);
    };
    requestAnimationFrame(this._raf);
  }
  _restoreKeyframes() {
    // Restoring A/B/C is independent of use_keyframes: the markers must
    // come back after a tab-switch / page reload even when the toggle is
    // OFF (they just render dimmed - see markerAlpha in render()).
    const aAz = getW(this.node, "A_azimuth")?.value;
    const aEl = getW(this.node, "A_elevation")?.value;
    const aDist = getW(this.node, "A_distance")?.value;
    const bAz = getW(this.node, "B_azimuth")?.value;
    const bEl = getW(this.node, "B_elevation")?.value;
    const bDist = getW(this.node, "B_distance")?.value;
    if (aAz != null && aEl != null && aDist != null) this.kfA = { az: aAz, el: aEl, dist: aDist };
    if (bAz != null && bEl != null && bDist != null) this.kfB = { az: bAz, el: bEl, dist: bDist };
    // C exists iff C_azimuth holds a real angle (the sentinel 999.0 means
    // "never placed"; Python uses the same check on its side).
    const cAz = getW(this.node, "C_azimuth")?.value;
    if (typeof cAz === "number" && Math.abs(cAz) <= 180) {
      const cEl = getW(this.node, "C_elevation")?.value;
      const cDist = getW(this.node, "C_distance")?.value;
      if (cEl != null && cDist != null) this.kfC = { az: cAz, el: cEl, dist: cDist };
    }
  }
  // Pull A/B/C widget values back into the in-memory keyframe objects every
  // frame so the user editing A_distance / B_distance / C_distance (or the
  // az/el widgets after un-hiding them) moves the markers on the sphere.
  _syncFromWidgets() {
    const pull = (kf, names) => {
      if (!kf) return;
      const az = getW(this.node, names.az)?.value;
      const el = getW(this.node, names.el)?.value;
      const dist = getW(this.node, names.dist)?.value;
      if (az != null) kf.az = az;
      if (el != null) kf.el = el;
      if (dist != null) kf.dist = dist;
    };
    pull(this.kfA, { az: "A_azimuth", el: "A_elevation", dist: "A_distance" });
    pull(this.kfB, { az: "B_azimuth", el: "B_elevation", dist: "B_distance" });
    pull(this.kfC, { az: "C_azimuth", el: "C_elevation", dist: "C_distance" });
    const sm = getW(this.node, "abc_smooth")?.value;
    if (typeof sm === "boolean") this.abcSmooth = sm;
    const uk = getW(this.node, "use_keyframes")?.value;
    if (typeof uk === "boolean") this.useKeyframes = uk;
  }
  _writeKfWidgets() {
    const setW = (name, v) => {
      const w = getW(this.node, name);
      if (w) w.value = v;
    };
    if (this.kfA) {
      setW("A_azimuth", this.kfA.az);
      setW("A_elevation", this.kfA.el);
      setW("A_distance", this.kfA.dist);
    }
    if (this.kfB) {
      setW("B_azimuth", this.kfB.az);
      setW("B_elevation", this.kfB.el);
      setW("B_distance", this.kfB.dist);
    }
    if (this.kfC) {
      setW("C_azimuth", this.kfC.az);
      setW("C_elevation", this.kfC.el);
      setW("C_distance", this.kfC.dist);
    } else {
      // kfC is null -> write the sentinel so Python's `abs(C_azimuth) <= 180`
      // check evaluates False and the node falls back to 2-point interp.
      setW("C_azimuth", 999.0);
    }
    // auto-flip use_keyframes ON once both A and B exist, so the user does
    // not have to hunt for the toggle after placing the markers
    if (this.kfA && this.kfB) {
      const w = getW(this.node, "use_keyframes");
      if (w && w.value === false) w.value = true;
    }
    this.node.setDirtyCanvas(true, true);
  }
  onKeyframeClick(x, y) {
    const g = this.geom();
    const ae = this.view.unproject(x - g.cx, y - g.cy, g.R);
    if (!ae) return;                       // missed the sphere disc entirely
    const az = Math.round(ae[0] / SNAP_DEG) * SNAP_DEG;
    const el = Math.round(ae[1] / SNAP_DEG) * SNAP_DEG;
    const dist = getW(this.node, "distance")?.value ?? 1;

    const HIT = 16;
    const onA = this.kfAPos && Math.hypot(x - this.kfAPos[0], y - this.kfAPos[1]) <= HIT;
    const onB = this.kfBPos && Math.hypot(x - this.kfBPos[0], y - this.kfBPos[1]) <= HIT;
    const onC = this.kfCPos && Math.hypot(x - this.kfCPos[0], y - this.kfCPos[1]) <= HIT;

    // Placement order is A -> B -> C. Clicking on an existing marker deletes
    // it and shifts everyone behind it up one slot (B->A, C->B) so the chain
    // stays contiguous with no gaps. When all three are placed, an S+click
    // on empty space repositions C.
    if (onA) {
      this.kfA = this.kfB;
      this.kfB = this.kfC;
      this.kfC = null;
    } else if (onB) {
      this.kfB = this.kfC;
      this.kfC = null;
    } else if (onC) {
      this.kfC = null;
    } else if (!this.kfA) {
      this.kfA = { az, el, dist };
    } else if (!this.kfB) {
      this.kfB = { az, el, dist };
    } else if (!this.kfC) {
      this.kfC = { az, el, dist };
    } else {
      this.kfC = { az, el, dist };
    }
    this._writeKfWidgets();
    this.render(true);
  }
  vals() {
    return {
      az: getW(this.node, "azimuth")?.value ?? 0,
      el: getW(this.node, "elevation")?.value ?? 0,
      dist: getW(this.node, "distance")?.value ?? 1,
    };
  }
  canvasPos(e) {
    const r = this.canvas.getBoundingClientRect();
    return [
      (e.clientX - r.left) * (this.canvas.width / r.width),
      (e.clientY - r.top) * (this.canvas.height / r.height),
    ];
  }
  geom() {
    const W = this.canvas.width, H = this.canvas.height;
    const S = Math.min(W - 16, H - 12);
    // sphere = the 1.0x reference shell; leave margin so the camera handle
    // and keyframe markers can sit OUTSIDE it (dist > 1) up to ~1.5R without
    // clipping (distScale now goes to 1.5 at dist=3.0).
    return { cx: W / 2, cy: H / 2, R: (S / 2 - 4) * 0.62 };
  }
  onDown(e) {
    e.preventDefault(); e.stopPropagation();
    const [x, y] = this.canvasPos(e);
    // S + click = keyframe placement (intercept before any drag begins)
    if (this.sHeld) {
      this.onKeyframeClick(x, y);
      return;
    }
    // ORDER MATTERS: camera handle first — if it sits ON a snap dot, the dot
    // must not steal the grab (that made the handle feel "stuck").
    if (this._resetBtn && Math.hypot(x - this._resetBtn[0], y - this._resetBtn[1]) < 14) {
      this.view.viewYaw = 0.24; this.view.viewTilt = 0.20;
      this.render(true);
      return;
    }
    if (this.handle && Math.hypot(x - this.handle[0], y - this.handle[1]) <= HANDLE_R + 10) {
      this.drag = "cam";
    } else {
      // grab a keyframe marker (A/B/C) to drag it — reuses the closest-match
      // pattern from onWheel so targeting feels consistent. Plain click only;
      // S+click still routes through onKeyframeClick (place/delete).
      const HIT = 20;
      let kfDrag = null;
      let best = HIT;
      if (this.kfA && this.kfAPos) {
        const d = Math.hypot(x - this.kfAPos[0], y - this.kfAPos[1]);
        if (d < best) { best = d; kfDrag = "kfA"; }
      }
      if (this.kfB && this.kfBPos) {
        const d = Math.hypot(x - this.kfBPos[0], y - this.kfBPos[1]);
        if (d < best) { best = d; kfDrag = "kfB"; }
      }
      if (this.kfC && this.kfCPos) {
        const d = Math.hypot(x - this.kfCPos[0], y - this.kfCPos[1]);
        if (d < best) { best = d; kfDrag = "kfC"; }
      }
      if (kfDrag) {
        this.drag = kfDrag;
      } else if (this.snapPts && this.snapPts.some(([sxp, syp]) => Math.hypot(x - sxp, y - syp) < 12)) {
        const [sxp, syp, a, e] = this.snapPts.find(([sxp2, syp2]) => Math.hypot(x - sxp2, y - syp2) < 12);
        const wA = getW(this.node, "azimuth"), wE = getW(this.node, "elevation");
        if (wA) wA.value = a;
        if (wE) wE.value = e;
        this.node.setDirtyCanvas(true, true);
        this.render(true);
        return;
      } else {
        this.drag = "view";
      }
    }
    this.lastPos = [x, y];
    document.addEventListener("pointermove", this._onMove);
    document.addEventListener("pointerup", this._onUp);
  }
  onMove(e) {
    if (!this.drag) return;
    const [x, y] = this.canvasPos(e);
    const g = this.geom();
    if (this.drag === "view") {
      this.view.viewYaw -= (x - this.lastPos[0]) * 0.01;   // grab metaphor: drag left -> globe turns left
      this.view.viewTilt += (y - this.lastPos[1]) * 0.01;
      this.view.viewTilt = Math.max(-1.4, Math.min(1.4, this.view.viewTilt));
      this.lastPos = [x, y];
    } else if (this.drag === "kfA" || this.drag === "kfB" || this.drag === "kfC") {
      // drag the picked keyframe on the sphere — same unproject+snap math as
      // the camera handle, but writes az/el into the kf and its widgets.
      // dist is left alone (the wheel owns that axis).
      const ae = this.view.unproject(x - g.cx, y - g.cy, g.R);
      if (ae) {
        const kf = this[this.drag];
        if (kf) {
          kf.az = Math.round(ae[0] / SNAP_DEG) * SNAP_DEG;
          kf.el = Math.round(ae[1] / SNAP_DEG) * SNAP_DEG;
          this._writeKfWidgets();
        }
      }
    } else {
      const ae = this.view.unproject(x - g.cx, y - g.cy, g.R);
      if (ae) {
        const wA = getW(this.node, "azimuth"), wE = getW(this.node, "elevation");
        if (wA) wA.value = Math.round(ae[0] / SNAP_DEG) * SNAP_DEG;
        if (wE) wE.value = Math.round(ae[1] / SNAP_DEG) * SNAP_DEG;
        this.node.setDirtyCanvas(true, true);
      }
    }
    this.render(true);
  }
  onUp() {
    this.drag = null;
    document.removeEventListener("pointermove", this._onMove);
    document.removeEventListener("pointerup", this._onUp);
  }
  onWheel(e) {
    e.preventDefault(); e.stopPropagation();
    const [mx, my] = this.canvasPos(e);
    const g = this.geom();
    // On-shell projection of a keyframe: depends only on (az, el), NOT on
    // dist -> stable while the dolly wheel actually moves the marker along
    // its ray, so the targeted keyframe can't flip mid-scroll.
    const shellPos = (kf) => {
      const [xr, , zv] = this.view.rot(spherePt(kf.az, kf.el));
      return [g.cx + g.R * xr, g.cy - g.R * zv];
    };
    const HIT = 22;
    let target = "distance";
    let best = HIT;
    if (this.kfA) {
      const [ax, ay] = shellPos(this.kfA);
      const d = Math.hypot(mx - ax, my - ay);
      if (d < best) { best = d; target = "A_distance"; }
    }
    if (this.kfB) {
      const [bx, by] = shellPos(this.kfB);
      const d = Math.hypot(mx - bx, my - by);
      if (d < best) { best = d; target = "B_distance"; }
    }
    if (this.kfC) {
      const [cx, cy] = shellPos(this.kfC);
      const d = Math.hypot(mx - cx, my - cy);
      if (d < best) { best = d; target = "C_distance"; }
    }
    const wD = getW(this.node, target);
    if (wD) {
      wD.value = Math.max(0.2, Math.min(3.0,
        Math.round((wD.value - Math.sign(e.deltaY) * 0.05) * 100) / 100));
      this.node.setDirtyCanvas(true, true);
      this.render(true);
    }
  }
  render(force) {
    // One-shot deferred restore: runs on the first render call, which fires
    // after ComfyUI's configure() has populated widget values from the saved
    // workflow. Reading them here (instead of in the constructor) is what
    // brings the A/B/C markers back after a page refresh.
    if (this._needsRestore) {
      this._restoreKeyframes();
      this._needsRestore = false;
    }
    // keep the backing resolution in sync with the element's layout size so
    // resizing the node doesn't stretch (deform) the globe; geom() uses
    // min(W,H) so the sphere stays circular at any aspect ratio
    const cw = this.canvas.clientWidth | 0, chh = this.canvas.clientHeight | 0;
    if (cw > 0 && chh > 0 && (this.canvas.width !== cw || this.canvas.height !== chh)) {
      this.canvas.width = cw;
      this.canvas.height = chh;
      this.cacheKey = "";   // sphere cache is size-dependent -> rebuild
    }
    // pick up A/B widget edits (e.g. user dragging the A_distance slider)
    // before computing the render cache key, so changing them forces a redraw
    this._syncFromWidgets();
    const { az, el, dist } = this.vals();
    const g = this.geom();
    const kfKey = `${this.kfA ? `${this.kfA.az.toFixed(1)},${this.kfA.el.toFixed(1)},${this.kfA.dist.toFixed(2)}` : "x"}|${this.kfB ? `${this.kfB.az.toFixed(1)},${this.kfB.el.toFixed(1)},${this.kfB.dist.toFixed(2)}` : "x"}|${this.kfC ? `${this.kfC.az.toFixed(1)},${this.kfC.el.toFixed(1)},${this.kfC.dist.toFixed(2)}` : "x"}|${this.abcSmooth ? 1 : 0}|${this.useKeyframes ? 1 : 0}`;
    const key = `${az}|${el}|${dist}|${this.view.viewYaw.toFixed(3)}|${this.view.viewTilt.toFixed(3)}|${kfKey}|${this.canvas.width}x${this.canvas.height}`;
    if (!force && key === this._renderKey) return;
    this._renderKey = key;

    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#1b1b1f";
    ctx.fillRect(0, 0, W, H);

    // cached static sphere
    const ck = `${this.view.viewYaw.toFixed(3)}|${this.view.viewTilt.toFixed(3)}|${g.R.toFixed(0)}`;
    if (this.cacheKey !== ck || !this.cache) {
      this.cache = document.createElement("canvas");
      const D = Math.ceil(g.R * 2) + 8;
      this.cache.width = D; this.cache.height = D;
      renderSphere(this.cache.getContext("2d"), this.view, D / 2, D / 2, g.R);
      this.cacheKey = ck;
    }
    ctx.drawImage(this.cache, g.cx - this.cache.width / 2, g.cy - this.cache.height / 2);

    const P = (a, e) => {
      const [xr, yv, zv] = this.view.rot(spherePt(a, e));
      return [g.cx + g.R * xr, g.cy - g.R * zv, yv];
    };

    // subject
    ctx.strokeStyle = "#d8d8e0"; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(g.cx, g.cy + 12); ctx.lineTo(g.cx, g.cy - 2); ctx.stroke();
    ctx.fillStyle = "#d8d8e0";
    ctx.beginPath(); ctx.arc(g.cx, g.cy - 8, 5, 0, Math.PI * 2); ctx.fill();

    // snap dots (gizmo style; F = home / source camera)
    this.snapPts = [];
    for (const [a, e, lab] of SNAPS) {
      const [sxp, syp, f] = P(a, e);
      this.snapPts.push([sxp, syp, a, e]);
      const rr = lab ? 9 : 6;
      ctx.globalAlpha = f < 0 ? 1.0 : 0.35;
      ctx.fillStyle = (a === 0 && e === 0) ? "#ffffff" : "#3a4150";
      ctx.strokeStyle = "#9aa2b5"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(sxp, syp, rr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      if (lab) {
        ctx.fillStyle = (a === 0 && e === 0) ? "#14161b" : "#d9dce3";
        ctx.font = "bold 9px monospace";
        const lw = ctx.measureText(lab).width;
        ctx.fillText(lab, sxp - lw / 2, syp + 3);
      }
      ctx.globalAlpha = 1.0;
    }

    // arc home -> camera
    ctx.strokeStyle = "#78beff"; ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i <= 14; i++) {
      const [ax, ay] = P((az * i) / 14, (el * i) / 14);
      i ? ctx.lineTo(ax, ay) : ctx.moveTo(ax, ay);
    }
    ctx.stroke();

    // camera handle at distance-scaled radius: the sphere is the 1.0x reference
    // shell; dist<1 pulls the camera INSIDE (closer to the subject), dist>1
    // pushes it OUTSIDE. Mouse wheel = dolly (industry standard).
    const distF = distScale(dist);
    const [sx, sy, pf] = P(az, el);           // on-shell point (direction)
    const px = g.cx + (sx - g.cx) * distF;
    const py = g.cy + (sy - g.cy) * distF;
    // faint dolly ray: subject -> out past the shell, with a tick at 1.0x
    ctx.strokeStyle = "rgba(120,190,255,0.35)"; ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(g.cx, g.cy);
    ctx.lineTo(g.cx + (sx - g.cx) * 1.32, g.cy + (sy - g.cy) * 1.32); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.beginPath(); ctx.arc(sx, sy, 3, 0, Math.PI * 2); ctx.stroke();  // 1.0x tick
    this.handle = [px, py];
    ctx.globalAlpha = pf < 0 ? 1.0 : 0.45;
    ctx.strokeStyle = "#78beff"; ctx.lineWidth = 1;
    for (const [dx, dy] of [[-8, -6], [8, -6], [-8, 6], [8, 6]]) {
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(g.cx + dx, g.cy + dy - 6); ctx.stroke();
    }
    ctx.fillStyle = "#78beff"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.roundRect(px - 13, py - 9, 26, 18, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#20242c";
    ctx.beginPath(); ctx.arc(px + 5, py, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1.0;

    // --- keyframe markers (S+click): green dots A/B/C joined by a green arc
    // mirroring the blue home->camera arc above. With only A+B the arc is a
    // straight wrap-aware lerp. With A+B+C it follows `abc_smooth`: piecewise
    // linear (corner at B) when OFF, or a quadratic Bezier whose control point
    // is 2*B - 0.5*(A+C) so the curve still passes through B at t=0.5 when ON.
    // markerAlpha dims the whole keyframe overlay to 20% when use_keyframes
    // is OFF, so placements persist visually but read as disabled.
    const markerAlpha = this.useKeyframes ? 1.0 : 0.2;
    this.kfAPos = null;
    this.kfBPos = null;
    this.kfCPos = null;
    if (this.kfA && this.kfB) {
      ctx.globalAlpha = markerAlpha;
      ctx.strokeStyle = "#5fce80"; ctx.lineWidth = 2.5;
      ctx.beginPath();
      const STEPS = this.kfC ? 42 : 28;
      for (let i = 0; i <= STEPS; i++) {
        const t = i / STEPS;
        let a, e;
        if (this.kfC && this.abcSmooth) {
          a = bezierAngle(this.kfA.az, this.kfB.az, this.kfC.az, t);
          e = bezier(this.kfA.el, this.kfB.el, this.kfC.el, t);
        } else if (this.kfC) {
          // piecewise linear: A->B->C with B exactly at t=0.5
          if (t <= 0.5) {
            const tt = t / 0.5;
            a = lerpAngle(this.kfA.az, this.kfB.az, tt);
            e = this.kfA.el + (this.kfB.el - this.kfA.el) * tt;
          } else {
            const tt = (t - 0.5) / 0.5;
            a = lerpAngle(this.kfB.az, this.kfC.az, tt);
            e = this.kfB.el + (this.kfC.el - this.kfB.el) * tt;
          }
        } else {
          a = lerpAngle(this.kfA.az, this.kfB.az, t);
          e = this.kfA.el + (this.kfB.el - this.kfA.el) * t;
        }
        const [ax, ay] = P(a, e);
        i ? ctx.lineTo(ax, ay) : ctx.moveTo(ax, ay);
      }
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }
    const drawKf = (kf, label) => {
      const [kx, ky, kfront] = P(kf.az, kf.el);
      const distFk = distScale(kf.dist);
      const kpx = g.cx + (kx - g.cx) * distFk;
      const kpy = g.cy + (ky - g.cy) * distFk;
      ctx.globalAlpha = markerAlpha * (kfront < 0 ? 1.0 : 0.55);
      // dotted dolly from marker -> on-shell point (the 1.0x tick), so the
      // distance is readable at a glance: marker inside the shell = closer
      // than source, outside = further back.
      ctx.strokeStyle = "rgba(95,206,128,0.7)"; ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(kpx, kpy); ctx.lineTo(kx, ky); ctx.stroke();
      ctx.setLineDash([]);
      // 1.0x tick on the shell (matches the camera handle's white tick)
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.beginPath(); ctx.arc(kx, ky, 3, 0, Math.PI * 2); ctx.stroke();
      // halo + solid disc so the marker stays visible over the blue handle
      ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(kpx, kpy, 11, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#5fce80";
      ctx.beginPath(); ctx.arc(kpx, kpy, 10, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#0e1410";
      ctx.font = "bold 12px monospace";
      const lw = ctx.measureText(label).width;
      ctx.fillText(label, kpx - lw / 2, kpy + 4);
      ctx.globalAlpha = 1.0;
      return [kpx, kpy];
    };
    if (this.kfA) this.kfAPos = drawKf(this.kfA, "A");
    if (this.kfB) this.kfBPos = drawKf(this.kfB, "B");
    if (this.kfC) this.kfCPos = drawKf(this.kfC, "C");

  }
}

app.registerExtension({
  name: "crossview.orbitPicker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CrossViewWarp") return;
    console.log("[CrossView Orbit] registering widget on CrossViewWarp");

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try {
        const node = this;
        const container = document.createElement("div");
        container.style.width = "100%";
        const canvas = document.createElement("canvas");
        canvas.width = 360;
        canvas.height = 300;
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        canvas.style.borderRadius = "8px";
        canvas.style.cursor = "grab";
        container.appendChild(canvas);

        node.addDOMWidget("orbit", "crossviewOrbitWidget", container, {
          serialize: false,
          hideOnZoom: false,
          getMinHeight: () => WIDGET_H,
          getMaxHeight: () => WIDGET_H,
          getHeight: () => WIDGET_H,
        });
        node._crossviewOrbit = new OrbitEditor(node, canvas);
        // hide the A/B/C az/el widgets: their values are driven by S+click on
        // the sphere, so showing six extra number fields would be noise.
        // User can re-enable via right-click -> show hidden widget if needed.
        for (const name of ["A_azimuth", "A_elevation", "B_azimuth", "B_elevation", "C_azimuth", "C_elevation"]) {
          const w = node.widgets?.find((w) => w.name === name);
          if (w) w.hidden = true;
        }
        node.setSize(node.computeSize());
        console.log("[CrossView Orbit] DOM widget attached to node", node.id);
      } catch (e) {
        console.error("[CrossView Orbit] widget attach FAILED:", e);
      }
      return r;
    };
  },
});
