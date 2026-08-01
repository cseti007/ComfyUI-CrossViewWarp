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
//   - mouse wheel             -> distance (0.1..3.0), or a hovered keyframe's
//   - double click            -> reset the view rotation
//   - RIGHT-CLICK             -> place a keyframe, or delete the one clicked
// For a static pose the azimuth/elevation/distance number widgets stay the
// source of truth; once keyframes exist the `keyframes` JSON widget is, and the
// static three are hidden because they no longer feed the render.

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
// Upper bound for the auto-grown globe (see globeH below) so a very wide node
// cannot turn the widget into a full-screen canvas.
const GLOBE_H_MAX = 700;
// Fallback spacing for a new keyframe when `frame_count` is 0 (i.e. the clip
// length was not supplied): one second apart at 24fps. With frame_count set, the
// keyframes are spread evenly across the clip instead — see evenFrames().
const KF_FRAME_STEP = 24;

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

// Distance ratio -> canvas radius multiplier. Piecewise so dist=1.0 lands on
// the shell: dist<=1 spreads 0.45..1.0 (inside = closer), dist>1 spreads
// 1.0..1.5 so the marker keeps moving all the way up to dist=3.0 (the old
// 1.3 clamp froze it past ~1.55).
function distScale(dist) {
  const d = Number(dist);
  if (d <= 1) return 0.45 + 0.55 * d;
  return 1.0 + 0.25 * (d - 1.0);
}

// --- keyframe path math ------------------------------------------------------
// Deliberate 1:1 mirror of _wrap_deg / _unwrap_seq / _catmull / _seg_value in
// crossview_warp_node.py: the arc drawn here must be the path the node renders.

function wrapDeg(a) { return (((a + 180) % 360) + 360) % 360 - 180; }

// Unwrap a sequence of angles so each step takes the short way round; this is
// what keeps a move across the +-180 seam continuous (170 -> -170 becomes
// 170 -> 190) instead of sweeping the long way through 0.
function unwrapSeq(degs) {
  const out = [Number(degs[0])];
  for (let i = 1; i < degs.length; i++) {
    out.push(out[i - 1] + wrapDeg(Number(degs[i]) - out[i - 1]));
  }
  return out;
}

// Uniform Catmull-Rom segment: passes through p1 at u=0 and p2 at u=1. Used
// instead of a Bezier because it interpolates its control points, so every
// keyframe is hit exactly and it generalises to any number of them.
function catmull(p0, p1, p2, p3, u) {
  return 0.5 * ((2 * p1)
    + (-p0 + p2) * u
    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
    + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u);
}

// Value between vals[seg] and vals[seg+1] at local u. Ends reflect their
// neighbour so the curve keeps its tangent; with two points Catmull-Rom is
// identical to the lerp, so the simple case is unaffected.
function segValue(vals, seg, u, smooth) {
  const p1 = vals[seg], p2 = vals[seg + 1];
  if (!smooth || vals.length < 3) return p1 + (p2 - p1) * u;
  const p0 = seg > 0 ? vals[seg - 1] : p1 + (p1 - p2);
  const p3 = seg + 2 < vals.length ? vals[seg + 2] : p2 + (p2 - p1);
  return catmull(p0, p1, p2, p3, u);
}

// Frame numbers an even spread over a `count`-frame clip would give to n
// keyframes. 1-based: frame 1 is the first frame, frame `count` the last.
function evenFrames(n, count) {
  if (n <= 1) return n === 1 ? [1] : [];
  return Array.from({ length: n }, (_, i) => 1 + Math.round((i * (count - 1)) / (n - 1)));
}

// True while the path still carries exactly the automatic even spread. This is
// what lets the spread be re-applied when keyframes are added or removed, yet
// stop the moment the user hand-edits a frame number in the widget — no extra
// "is this one locked?" state to store or keep in sync.
function isAutoTimed(kfs, count) {
  const want = evenFrames(kfs.length, count);
  return kfs.every((k, i) => k.f === want[i]);
}

// Refit an existing path onto a clip of `count` frames: the first keyframe moves
// to 1, the last to `count`, and the ones between keep their relative spacing.
//
// This runs when frame_count CHANGES, and it is what makes the setting usable at
// all. isAutoTimed() alone cannot do it: any path that does not match the current
// formula reads as hand-edited, so keyframes placed before frame_count was set
// (or by an older version numbering from 0) would never line up with the clip
// again. Rescaling is also the honest answer for a hand-timed path — the rhythm
// the user authored is preserved, just fitted to the new length.
function rescaleFrames(kfs, count) {
  if (kfs.length < 2 || count < 2) return false;
  const first = kfs[0].f, span = kfs[kfs.length - 1].f - first;
  const want = kfs.map((k, i) => (span > 0
    ? 1 + Math.round(((k.f - first) * (count - 1)) / span)
    : 1 + Math.round((i * (count - 1)) / (kfs.length - 1))));
  // a clip too short to hold them all as distinct frames is left alone
  if (new Set(want).size !== want.length) return false;
  kfs.forEach((k, i) => { k.f = want[i]; });
  return true;
}

function round1(v) { return Math.round(Number(v) * 10) / 10; }
function round2(v) { return Math.round(Number(v) * 100) / 100; }

// Read the `keyframes` widget into an ordered list of {f, az, el, dist}.
// Anything unparseable yields [] here rather than throwing: the widget is
// user-editable text, and a half-typed edit must not break the whole canvas.
// The node itself re-validates and reports a real error at execution time.
function parseKfs(raw) {
  if (typeof raw !== "string" || !raw.trim()) return [];
  let data;
  try { data = JSON.parse(raw); } catch (e) { return []; }
  if (!Array.isArray(data)) return [];
  const out = [];
  for (const kf of data) {
    if (!kf || typeof kf !== "object") continue;
    const f = Number(kf.f), az = Number(kf.az), el = Number(kf.el), dist = Number(kf.dist);
    if (![f, az, el, dist].every(Number.isFinite)) continue;
    out.push({ f: Math.round(f), az, el, dist });
  }
  out.sort((a, b) => a.f - b.f);
  return out;
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
    const self = this;
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

    // Keyframe state. this.kfs is the single source of truth: an
    // ordered list of {f, az, el, dist}, serialised verbatim into the
    // `keyframes` widget. Keeping every keyframe in one list (rather than
    // fixed A/B/C slots) is what makes deletion a plain splice and removes
    // the need for any "was this one placed?" sentinel.
    this.kfs = [];
    // canvas-space marker centres, written by render() and read by the drag
    // and wheel hit-tests so both target what is actually drawn.
    this.kfPos = [];
    this._lastKfRaw = undefined;   // last `keyframes` string we read or wrote
    this._lastKfCount = 0;         // to detect crossing the "is there a move" line
    // `interpolation` mirror: false = straight legs, true = Catmull-Rom.
    this.smoothPath = false;
    // use_keyframes mirror: when OFF the markers are rendered at 20% alpha so
    // the user can see the keyframed move is disabled without losing the
    // placements they made.
    this.useKeyframes = false;

    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture?.(e.pointerId);
      this.onDown(e);
    });
    // Right-click places (or, on a marker, deletes) a keyframe. Deliberately a
    // pure pointer gesture with no keyboard modifier: a modifier key can always
    // be claimed by another node pack through ComfyUI's keybinding system, and
    // that is exactly what bit this widget before -- KJNodes binds S to its
    // node-swap gesture, whose pointerup handler called disconnectAll() on the
    // node under the cursor, tearing every link off CrossViewWarp and swapping
    // its position. A mouse button on our own canvas cannot be hijacked that way.
    canvas.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const [x, y] = this.canvasPos(e);
      this.onKeyframeClick(x, y);
    });
    // Second line of defence against document-level gesture handlers from other
    // packs (KJNodes' node-swap reacts on `document` pointerup): a release that
    // belongs to this widget must not bubble out of it. Because pointerdown sets
    // pointer capture, every pointerup of our gesture retargets here -- including
    // releases outside the canvas -- so onUp() has to be driven from here rather
    // than relying on the document listener that stopPropagation now blocks.
    canvas.addEventListener("pointerup", (e) => {
      e.stopPropagation();
      if (this.drag) this.onUp();
    });
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener("dblclick", () => {
      this.view.viewYaw = 0.24; this.view.viewTilt = 0.20;
      this.render(true);
    });
    this._onMove = (e) => this.onMove(e);
    this._onUp = () => this.onUp();

    // Tear the document-level listeners down when the node goes away. Chained
    // rather than assigned: ComfyUI installs its own onRemoved for DOM widgets,
    // and overwriting it would leak the widget registration instead.
    const prevRemoved = node.onRemoved;
    node.onRemoved = function () {
      document.removeEventListener("pointermove", self._onMove);
      document.removeEventListener("pointerup", self._onUp);
      return prevRemoved?.apply(this, arguments);
    };

    // Reading the widgets is deferred to the first render — ComfyUI runs
    // onNodeCreated (this constructor) BEFORE configure() loads saved widget
    // values, so reading here would see defaults and miss a saved path. By
    // render time (next animation frame) configure has run.
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
    // Restoring is independent of use_keyframes: the markers must come back
    // after a tab-switch / page reload even when the toggle is OFF (they just
    // render dimmed - see markerAlpha in render()). An unreadable string is
    // left alone rather than overwritten, so a hand-edit typo is not silently
    // destroyed by the widget being repainted.
    // A workflow saved before the orbit canvas stopped serialising itself carries
    // one extra trailing value, which now lands on use_keyframes and leaves a
    // boolean toggle holding "". ComfyUI coerces it with bool() at execution, so
    // mirroring that here keeps the widget showing what the node will actually do
    // instead of a stray string, until the workflow is re-saved.
    const wu = getW(this.node, "use_keyframes");
    if (wu && typeof wu.value !== "boolean") wu.value = !!wu.value;

    this._lastKfRaw = getW(this.node, "keyframes")?.value;
    this.kfs = parseKfs(this._lastKfRaw);
    this._lastKfCount = this.kfs.length;
    // Remember the starting frame_count without acting on it: rescaling belongs
    // to a deliberate change during the session, not to merely opening a
    // workflow, which would rewrite a saved path every time it is loaded.
    this._lastFrames = Math.round(getW(this.node, "frame_count")?.value ?? 0);
  }
  // Where the `keyframes` value comes from. Once the widget is converted to an
  // input, the node executes with whatever the upstream sends and our local
  // widget value is a leftover — editing the sphere would then show a path that
  // is not the one rendered. So: report that it is linked, and resolve the value
  // when it is knowable. A literal upstream (primitive / string constant) carries
  // the text in one of its own widgets; try each and take the first that parses.
  // A computed upstream only produces its string on the server at execution time,
  // so there is genuinely nothing to preview — say so rather than show a stale
  // local value.
  // camera_info is a plain socket rather than a converted widget, so it is
  // matched by input name. When it is wired the node takes its pose from there
  // and ignores azimuth/elevation/distance, the pivot and the keyframe path
  // entirely - the sphere would otherwise keep drawing a camera that has no
  // bearing on the render. Its value only exists at execution time, so this can
  // report THAT it is driven from outside but never where the camera ended up;
  // the orbit_view output image shows that.
  _cameraInfoLinked() {
    return this.node.inputs?.some((i) => i.name === "camera_info" && i.link != null) === true;
  }
  _keyframeSource() {
    const inp = this.node.inputs?.find((i) => i.widget?.name === "keyframes");
    if (inp?.link == null) return { linked: false, kfs: null };
    const graph = this.node.graph || app.graph;
    const link = graph?.links?.[inp.link];
    const src = link ? graph.getNodeById?.(link.origin_id) : null;
    for (const w of src?.widgets ?? []) {
      if (typeof w.value !== "string") continue;
      const kfs = parseKfs(w.value);
      if (kfs.length) return { linked: true, kfs };
    }
    return { linked: true, kfs: null };
  }
  // Re-read the widgets every frame so hand-edits to the `keyframes` string and
  // the interpolation / use_keyframes toggles show up on the sphere.
  _syncFromWidgets() {
    this.camLinked = this._cameraInfoLinked();
    const src = this._keyframeSource();
    this.linked = src.linked;
    this.linkedUnknown = src.linked && src.kfs === null;
    if (src.linked) {
      // driven from outside: mirror what we can resolve, never our own leftover
      this.kfs = src.kfs ?? [];
    } else {
      const raw = getW(this.node, "keyframes")?.value;
      if (raw !== this._lastKfRaw) {
        this._lastKfRaw = raw;
        this.kfs = parseKfs(raw);
        // a hand-edit is as authoritative as a click, so the threshold tracker
        // must follow it too
        this._lastKfCount = this.kfs.length;
      }
      // A changed frame_count refits the path onto the new clip length. Only on
      // an actual change, so it never fights a hand-edited path that the user is
      // not currently retargeting.
      const frames = Math.round(getW(this.node, "frame_count")?.value ?? 0);
      if (frames !== this._lastFrames) {
        this._lastFrames = frames;
        if (frames > 1 && rescaleFrames(this.kfs, frames)) this._writeKfWidgets();
      }
    }
    const sm = getW(this.node, "interpolation")?.value;
    if (typeof sm === "string") this.smoothPath = sm === "smooth";
    const uk = getW(this.node, "use_keyframes")?.value;
    if (typeof uk === "boolean" && uk !== this.useKeyframes) {
      this.useKeyframes = uk;
      // the user can flip the toggle directly, not just via _writeKfWidgets
      this.node._crossviewSyncHidden?.();
    }
  }
  _writeKfWidgets() {
    const w = getW(this.node, "keyframes");
    if (w) {
      w.value = this.kfs.length
        ? JSON.stringify(this.kfs.map((k) => ({
            f: k.f, az: round1(k.az), el: round1(k.el), dist: round2(k.dist),
          })))
        : "";
      this._lastKfRaw = w.value;   // our own write must not look like a hand-edit
    }
    // Follow the path only when it CROSSES the "is there a move at all"
    // threshold: gaining a second keyframe turns the move on, dropping below
    // two turns it off. Deliberately not set on every write — otherwise
    // nudging a marker would override a user who had just switched the move
    // off to compare against the static pose.
    const usable = this.kfs.length >= 2;
    if (usable !== (this._lastKfCount >= 2)) {
      const wu = getW(this.node, "use_keyframes");
      if (wu) wu.value = usable;
    }
    this._lastKfCount = this.kfs.length;
    this.node._crossviewSyncHidden?.();
    this.node.setDirtyCanvas(true, true);
  }
  onKeyframeClick(x, y) {
    // driven by an upstream node -> the sphere is a preview, not an editor;
    // writing here would show a path the node will not actually render
    if (this.linked) return;
    const g = this.geom();
    const ae = this.view.unproject(x - g.cx, y - g.cy, g.R);
    if (!ae) return;                       // missed the sphere disc entirely
    const az = Math.round(ae[0] / SNAP_DEG) * SNAP_DEG;
    const el = Math.round(ae[1] / SNAP_DEG) * SNAP_DEG;
    const dist = getW(this.node, "distance")?.value ?? 1;

    // With frame_count set, the path spans the whole clip and keyframes are
    // re-spread on every add/remove — but only while the timing is still the
    // automatic one. Checked BEFORE the list changes, since a freshly appended
    // keyframe would never match the spread.
    const count = Math.round(getW(this.node, "frame_count")?.value ?? 0);
    const respread = count > 1 && isAutoTimed(this.kfs, count);

    // Right-click on an existing marker deletes just that one.
    const HIT = 16;
    const hit = this.kfs.findIndex((_, i) => this.kfPos[i] &&
      Math.hypot(x - this.kfPos[i][0], y - this.kfPos[i][1]) <= HIT);
    if (hit >= 0) {
      this.kfs.splice(hit, 1);
    } else {
      if (!this.kfs.length) {
        // Seed the path with the pose the user already dialled in, so the move
        // starts where the camera currently is instead of discarding that work.
        const v = this.vals();
        this.kfs.push({ f: 1, az: v.az, el: v.el, dist: v.dist });
      }
      const lastF = this.kfs[this.kfs.length - 1].f;
      this.kfs.push({ f: lastF + KF_FRAME_STEP, az, el, dist });
    }

    if (respread) {
      const want = evenFrames(this.kfs.length, count);
      // A clip too short to hold this many distinct frames would round several
      // keyframes onto the same one, which the node rejects. Leave the existing
      // timing alone in that case rather than generating an invalid path.
      if (new Set(want).size === want.length) {
        this.kfs.forEach((k, i) => { k.f = want[i]; });
      }
    }
    this._writeKfWidgets();
    this.render(true);
  }
  // Index of the keyframe marker nearest to (x, y) within `hit` px, or -1.
  // Uses kfPos — where the marker is actually DRAWN — so drag, delete and the
  // dolly wheel all target the thing under the cursor. (The wheel previously
  // tested against the on-shell position, which for dist=0.2 is ~38px away
  // from the visible marker: you could not scroll the marker you were pointing
  // at.)
  pickKf(x, y, hit) {
    let best = hit, idx = -1;
    for (let i = 0; i < this.kfs.length; i++) {
      const p = this.kfPos[i];
      if (!p) continue;
      const d = Math.hypot(x - p[0], y - p[1]);
      if (d < best) { best = d; idx = i; }
    }
    return idx;
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
    // Left button only. A right-click also fires pointerdown, and without this
    // the keyframe gesture would start a view drag underneath itself.
    if (e.button !== 0) return;
    const [x, y] = this.canvasPos(e);
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
      // grab the nearest keyframe marker to drag it. Left button only; right-click
      // still routes through onKeyframeClick (place/delete).
      const kfIdx = this.linked ? -1 : this.pickKf(x, y, 20);
      if (kfIdx >= 0) {
        this.drag = "kf";
        this.dragKf = kfIdx;
      } else if (!this.useKeyframes && this.snapPts &&
                 this.snapPts.some(([sxp, syp]) => Math.hypot(x - sxp, y - syp) < 12)) {
        // snap dots write azimuth/elevation, which are inert while a keyframed
        // move is running — so they only respond in static mode
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
    } else if (this.drag === "kf") {
      // drag the picked keyframe on the sphere — same unproject+snap math as
      // the camera handle, but writes az/el into the keyframe. dist is left
      // alone (the wheel owns that axis) and so is f (the timing).
      const ae = this.view.unproject(x - g.cx, y - g.cy, g.R);
      const kf = this.kfs[this.dragKf];
      if (ae && kf) {
        kf.az = Math.round(ae[0] / SNAP_DEG) * SNAP_DEG;
        kf.el = Math.round(ae[1] / SNAP_DEG) * SNAP_DEG;
        this._writeKfWidgets();
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
    // Wheel back (deltaY > 0) raises the number, wheel forward lowers it - the
    // same direction as every other numeric widget in the graph.
    const step = Math.sign(e.deltaY) * 0.05;
    const clampDist = (v) => Math.max(0.1, Math.min(3.0, Math.round(v * 100) / 100));

    // Hovering a keyframe dollies THAT keyframe; anywhere else dollies the
    // static camera. Both use the drawn marker position (see pickKf).
    const idx = this.linked ? -1 : this.pickKf(mx, my, 22);
    if (idx >= 0) {
      const kf = this.kfs[idx];
      kf.dist = clampDist(kf.dist + step);
      this._writeKfWidgets();
      this.render(true);
      return;
    }
    const wD = getW(this.node, "distance");
    if (wD) {
      wD.value = clampDist(wD.value + step);
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
    // pick up widget edits (e.g. a hand-edited keyframes string) before
    // computing the render cache key, so changing them forces a redraw
    this._syncFromWidgets();
    const { az, el, dist } = this.vals();
    const g = this.geom();
    const kfKey = `${this.kfs.map((k) => `${k.f},${k.az.toFixed(1)},${k.el.toFixed(1)},${k.dist.toFixed(2)}`).join(";")}|${this.smoothPath ? 1 : 0}|${this.useKeyframes ? 1 : 0}|${this.linked ? 1 : 0}${this.linkedUnknown ? "?" : ""}|${this.camLinked ? 1 : 0}`;
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

    // The blue camera handle visualises the STATIC azimuth/elevation/distance,
    // which do nothing once a keyframed move is running (those widgets are
    // hidden then). Drawing it anyway left two markers for one camera, with the
    // blue one sitting under the green frame-0 marker it was seeded from. So in
    // keyframe mode the whole static-camera overlay is skipped and the green
    // path is the only camera indicator.
    // A camera_info pose overrides everything the sphere shows, so the overlay
    // drops to a ghost: still legible as a setup, clearly not in charge. Applied
    // as a factor because the block below assigns globalAlpha itself.
    const camDim = this.camLinked ? 0.25 : 1.0;
    ctx.globalAlpha = camDim;
    if (!this.useKeyframes) {
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
      ctx.globalAlpha = (pf < 0 ? 1.0 : 0.45) * camDim;
      ctx.strokeStyle = "#78beff"; ctx.lineWidth = 1;
      for (const [dx, dy] of [[-8, -6], [8, -6], [-8, 6], [8, 6]]) {
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(g.cx + dx, g.cy + dy - 6); ctx.stroke();
      }
      ctx.fillStyle = "#78beff"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.roundRect(px - 13, py - 9, 26, 18, 4); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#20242c";
      ctx.beginPath(); ctx.arc(px + 5, py, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1.0;
    } else {
      // no static handle on screen -> nothing for onDown() to grab, otherwise
      // the last drawn position would stay draggable while invisible
      this.handle = null;
    }

    // --- keyframe markers: green dots joined by a green arc that
    // mirrors the blue home->camera arc above. The arc is walked segment by
    // segment with the same unwrap + Catmull-Rom math the node uses, so what
    // is drawn here is the path that will actually be rendered. Segment
    // spacing on screen shows shape only — the timing lives in each keyframe's
    // frame number. markerAlpha dims the overlay to 20% when use_keyframes is
    // OFF, so placements persist visually but read as disabled.
    const markerAlpha = (this.useKeyframes ? 1.0 : 0.2) * camDim;
    this.kfPos = [];
    if (this.kfs.length >= 2) {
      const azU = unwrapSeq(this.kfs.map((k) => k.az));
      const els = this.kfs.map((k) => k.el);
      ctx.globalAlpha = markerAlpha;
      ctx.strokeStyle = "#5fce80"; ctx.lineWidth = 2.5;
      ctx.beginPath();
      const SUB = 24;
      let started = false;
      for (let seg = 0; seg < this.kfs.length - 1; seg++) {
        for (let s = 0; s <= SUB; s++) {
          const u = s / SUB;
          const a = wrapDeg(segValue(azU, seg, u, this.smoothPath));
          const e = segValue(els, seg, u, this.smoothPath);
          const [ax, ay] = P(a, e);
          if (started) ctx.lineTo(ax, ay); else { ctx.moveTo(ax, ay); started = true; }
        }
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
      ctx.font = "bold 11px monospace";
      const lw = ctx.measureText(label).width;
      ctx.fillText(label, kpx - lw / 2, kpy + 4);
      ctx.globalAlpha = 1.0;
      return [kpx, kpy];
    };
    // Label each marker with its FRAME number, not a letter: the timing is the
    // one thing the sphere cannot show geometrically, and it is what the user
    // edits in the keyframes widget.
    this.kfPos = this.kfs.map((kf) => drawKf(kf, String(kf.f)));

    // Status line. When `keyframes` is wired from another node the sphere cannot
    // be edited, and if that node computes its string at execution time there is
    // nothing to preview either — better to admit that than to leave a confident
    // but wrong picture on screen. Otherwise the right-click hint, which retires
    // once the first keyframe exists.
    ctx.globalAlpha = 1.0;
    let hint = null;
    if (this.camLinked) hint = "camera from camera_info input - these controls are ignored";
    else if (this.linkedUnknown) hint = "keyframes: driven by input (value known at run time)";
    else if (this.linked) hint = "keyframes: driven by input - preview only";
    else if (!this.kfs.length) hint = "right-click: add camera keyframe";
    if (hint) {
      ctx.fillStyle = (this.camLinked || this.linkedUnknown)
        ? "rgba(230,200,90,0.75)" : "rgba(200,204,216,0.45)";
      ctx.font = "11px monospace";
      ctx.fillText(hint, (W - ctx.measureText(hint).width) / 2, H - 8);
    }
  }
}

// Two copies of this package can be installed side by side (a dev worktree next
// to the release, see .node_suffix in crossview_warp_node.py). Extension names
// must be unique or the second registration is dropped, so derive one from the
// directory this module was served from; and match the node by PREFIX so the
// widget attaches to a suffixed dev id as well.
const PKG = (import.meta.url.match(/\/extensions\/([^/]+)\//) || [, "core"])[1];

app.registerExtension({
  name: `crossview.orbitPicker.${PKG}`,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name.startsWith("CrossViewWarp")) return;
    console.log(`[CrossView Orbit] registering widget on ${nodeData.name}`);

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try {
        const node = this;
        // With two copies of the package installed, both register an extension
        // and both match this node by prefix, so onNodeCreated is wrapped twice
        // and the canvas would be added twice. Cheap guard, and it makes the
        // widget idempotent in general.
        if (node.widgets?.some((w) => w.name === "orbit")) return r;
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

        // geom() sizes the sphere from min(canvasW, canvasH). The canvas width
        // already follows the node, but with the height pinned at WIDGET_H the
        // globe stopped growing as soon as the node was wider than that. Letting
        // the height track the node width keeps the canvas roughly square, so
        // widening the node actually enlarges the globe; clamped at both ends so
        // it can neither collapse nor run away.
        const globeH = () => Math.round(
          Math.max(WIDGET_H, Math.min(GLOBE_H_MAX, (node.size?.[0] ?? 360) * 0.85)));

        const orbitWidget = node.addDOMWidget("orbit", "crossviewOrbitWidget", container, {
          serialize: false,
          hideOnZoom: false,
          getMinHeight: globeH,
          getMaxHeight: globeH,
          getHeight: globeH,
        });
        // The options object above only reaches widget.options.serialize, but
        // litegraph's serialize()/configure() check widget.serialize — so this
        // canvas was being written into widgets_values as a stray "". Harmless
        // on its own, but it shifts every value added after it: workflows saved
        // before this fix carry one extra trailing entry.
        if (orbitWidget) orbitWidget.serialize = false;
        node._crossviewOrbit = new OrbitEditor(node, canvas);

        // While a keyframed move is active the static azimuth/elevation/distance
        // do nothing, so hide them rather than leave three dead controls that
        // still draw a camera handle. Their values are kept (the first keyframe
        // is seeded from them) and they come back when the path is cleared.
        //
        // Both flags are set on purpose: the canvas renderer reads widget.hidden,
        // the Vue node renderer reads widget.options.hidden and does NOT fall
        // back to the former. Setting only one leaves them visible in one of the
        // two frontends.
        node._crossviewSyncHidden = () => {
          const active = node.widgets?.find((w) => w.name === "use_keyframes")?.value === true;
          let changed = false;
          for (const name of ["azimuth", "elevation", "distance"]) {
            const w = node.widgets?.find((x) => x.name === name);
            if (!w || w.hidden === active) continue;
            w.hidden = active;
            if (w.options) w.options.hidden = active;
            changed = true;
          }
          // Deliberately NO setSize/computeSize here. computeSize() returns the
          // node's minimum layout size, so calling it on a visibility change threw
          // away a manual resize: the node snapped smaller and jumped, which looked
          // like it was tearing its own links off. Repainting is enough — the
          // frontend lays hidden widgets out on its own; worst case a small gap is
          // left where they were.
          if (changed) node.setDirtyCanvas(true, true);
        };
        node._crossviewSyncHidden();
        node.setSize(node.computeSize());
        console.log("[CrossView Orbit] DOM widget attached to node", node.id);
      } catch (e) {
        console.error("[CrossView Orbit] widget attach FAILED:", e);
      }
      return r;
    };
  },
});
