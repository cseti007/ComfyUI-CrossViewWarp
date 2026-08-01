// CrossView Live Preview — draggable, playable warp preview for the
// CrossViewPreview node.
//
// The node caches the whole clip when the graph runs; this widget then POSTs a
// camera pose plus a frame number at it and paints the returned warp. Dragging
// happens ON THE IMAGE (left/right = azimuth, up/down = elevation, wheel =
// distance) rather than on a separate globe: the thing you are aiming is the
// picture, so that is what you grab. A scrub bar underneath moves the playhead.
//
// Playback is self-clocked: each frame is rendered on demand and the next is
// requested only once the previous one has arrived, so the clip plays at
// whatever rate the server can warp (about 20 fps at 384 px). It therefore does
// not run at the clip's real frame rate — it is for judging the warp, not the
// timing.
//
// Exactly one request is ever in flight. Intermediate drag positions are
// dropped rather than queued: a queue would turn a fast drag into a growing
// backlog of renders nobody wants to see any more, which reads as lag.

import { app } from "../../scripts/app.js";

const WIDGET_H = 220;
const H_MAX = 560;
const CTRL_H = 34;               // control strip: play button, scrub bar, readout
const DRAG_SENS = 0.35;          // degrees per pixel
const BTN_W = 24;
const MODE_W = 46;               // "warp" / "source" toggle at the right of the strip
const KEY_W = 26;                // keyframe drop/remove button
const RST_W = 28;                // reset-to-defaults button
const DIM = "rgba(200,204,216,0.45)";

// Parameters forwarded to the render endpoint. Kept in sync with _PARAMS in
// crossview_preview_node.py; anything missing here falls back server-side.
// frame_index is deliberately absent — it is the playhead, sent separately,
// and it is not an argument to the warp.
const PARAM_NAMES = [
  "azimuth", "elevation", "distance", "hfov", "vertical_shift", "depth_ratio",
  "smooth_depth", "invert_depth", "roll_lock", "pivot_override",
  "pivot_x", "pivot_y", "pivot_z", "keep_source_aim",
  // The path travels with the request: the server samples it at the playhead
  // with the node's own interpolation, so the preview shows the pose the move
  // actually holds there rather than the static widgets it overrides.
  "use_keyframes", "keyframes", "interp_motion", "interpolation",
];

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

class PreviewEditor {
  constructor(node, canvas, endpoint, defaults) {
    this.node = node;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.endpoint = endpoint;
    // Straight from the node definition, so the button can never drift from
    // the Python defaults the way a hand-kept copy would.
    this.defaults = defaults || {};
    this.armed = false;          // reset asked for once, waiting for confirmation
    this.img = null;
    this.status = "run the graph once to load the clip";
    this.ms = null;
    this.info = null;            // {of, w, h, mb, source} from the last execution
    this.stale = false;          // cached clip may no longer match the inputs
    this.playing = false;
    // Auditioning: a pose being tried out at a frame that has no keyframe.
    // The path is suspended for the render so you can see what you are
    // aiming at, and KEY commits it. Nothing is written until you do.
    this.auditioning = false;
    this.mode = "warp";          // "warp" = the render, "source" = the input frame
    this.pivot = null;           // {u,v,src_u,src_v,z,scene_z,radius,auto} per render
    this.rect = null;            // where the image was last drawn, for the marker
    this.inflight = false;
    this.shownSig = null;
    this.drag = null;            // {mode: "orbit"|"scrub", ...}

    canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerup", (e) => this.onUp(e));
    canvas.addEventListener("pointercancel", (e) => this.onUp(e));
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    // The image is not a context menu surface, and a stray menu mid-drag is
    // just noise; the node's title and body still open it.
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    this.hookWidgets();
    this.draw();
    // The server cache outlives the browser, so after a page reload the clip is
    // usually still there and the preview can paint without another run.
    // Deferred by a tick because a node restored from a saved workflow does not
    // have its final id yet while onNodeCreated is running.
    setTimeout(() => this.request(), 0);
  }

  w(name) {
    return this.node.widgets?.find((x) => x.name === name);
  }

  frames() {
    return this.info?.of ?? 1;
  }

  frame() {
    return clamp(Math.round(this.w("frame_index")?.value ?? 1), 1, this.frames());
  }

  // Re-render when any parameter changes, including from the number widgets
  // themselves — otherwise typing a value would leave the picture behind.
  hookWidgets() {
    const POSE = ["azimuth", "elevation", "distance", "vertical_shift"];
    for (const name of [...PARAM_NAMES, "frame_index"]) {
      const wd = this.w(name);
      if (!wd) continue;
      const prev = wd.callback;
      wd.callback = (...args) => {
        const r = prev?.apply(wd, args);
        if (POSE.includes(name)) this.syncKey();
        this.request();
        return r;
      };
    }
  }

  params() {
    const out = {};
    for (const name of PARAM_NAMES) {
      const wd = this.w(name);
      if (wd) out[name] = wd.value;
    }
    // Suspend the path so the pose being auditioned is what gets rendered. The
    // widget itself is untouched — nothing is committed until KEY is pressed.
    if (this.auditioning) out.use_keyframes = false;
    return out;
  }

  sig() {
    return `${this.mode}|${this.frame()}|${JSON.stringify(this.params())}`;
  }

  request() {
    this.pump();
  }

  async pump() {
    if (this.inflight) return;
    const sig = this.sig();
    if (sig === this.shownSig) return;
    this.inflight = true;
    const t0 = performance.now();
    let ok = false;
    try {
      const res = await fetch(this.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          node_id: String(this.node.id),
          frame: this.frame(),
          mode: this.mode,
          params: this.params(),
        }),
      });
      if (res.status === 409) {
        this.img = null;
        this.status = "run the graph once to load the clip";
      } else if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const j = await res.json();
          if (j?.error) detail = j.error;
        } catch (e) { /* non-JSON error body: the status code is all we get */ }
        this.status = detail;
      } else {
        // Pivot geometry rides in a header, so the marker can be drawn, moved
        // or hidden without the server re-rendering anything.
        try {
          const j = JSON.parse(res.headers.get("X-Preview-Info") || "null");
          if (j) {
            this.pivot = j.pivot || null;
            if (Number.isFinite(j.count)) this.setCount(j.count);
          }
        } catch (e) { this.pivot = null; }
        this.img = await createImageBitmap(await res.blob());
        this.status = null;
        this.ms = Math.round(performance.now() - t0);
        ok = true;
      }
      this.shownSig = sig;
    } catch (e) {
      this.status = String(e?.message || e);
    } finally {
      this.inflight = false;
      // Playback advances only after a frame actually arrives, so an error
      // stops the clip instead of hammering the server at full speed.
      if (!ok) this.playing = false;
      this.draw();
      if (this.playing) this.step(1, true);
      else if (this.sig() !== this.shownSig) this.pump();
    }
  }

  // The clip length is only known once the node has run; keep the widget's own
  // limit in step so typing into it cannot point past the end of the cache.
  setCount(n) {
    if (!this.info) this.info = {};
    this.info.of = n;
    const wd = this.w("frame_index");
    if (wd) {
      if (wd.options) wd.options.max = n;
      if (wd.value > n) this.setFrame(n);
    }
  }

  setNum(name, v, dec = 1) {
    const wd = this.w(name);
    if (!wd) return;
    const f = Math.pow(10, dec);
    wd.value = Math.round(v * f) / f;
    wd.callback?.(wd.value);          // hooked above, so this also re-renders
    app.graph?.setDirtyCanvas(true, false);
  }

  setFrame(f) {
    const cur = this.frame();
    const next = clamp(Math.round(f), 1, this.frames());
    // Moving off the frame abandons an uncommitted pose rather than dragging it
    // along, which would silently apply it somewhere it was never aimed.
    if (next !== cur) this.auditioning = false;
    this.setNum("frame_index", next, 0);
  }

  step(d, wrap) {
    const n = this.frames();
    let f = this.frame() + d;
    if (wrap) f = ((f - 1 + n) % n) + 1;
    this.setFrame(f);
  }

  // Canvas-space coordinates: the element is stretched by CSS, so client pixels
  // are not backing-store pixels and hit testing needs the conversion.
  pos(e) {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (this.canvas.width / (r.width || 1)),
      y: (e.clientY - r.top) * (this.canvas.height / (r.height || 1)),
    };
  }

  barGeom() {
    const W = this.canvas.width, H = this.canvas.height;
    return { x0: BTN_W + KEY_W + 14, x1: W - MODE_W - 14, y: H - CTRL_H + 12, top: H - CTRL_H };
  }

  keyGeom() {
    const H = this.canvas.height;
    return { x0: BTN_W + 6, x1: BTN_W + 6 + KEY_W, y0: H - CTRL_H + 3, y1: H - CTRL_H + 21 };
  }

  rstGeom() {
    const W = this.canvas.width, H = this.canvas.height;
    return { x0: W - MODE_W - RST_W - 12, x1: W - MODE_W - 12,
             y0: H - CTRL_H + 3, y1: H - CTRL_H + 21 };
  }

  // Back to the node's own defaults. A camera path is real work, so when one
  // exists the first click only arms the button - losing a move to a stray
  // click is not something an undo would get back here.
  reset() {
    if (this.kfs().length && !this.armed) { this.armed = true; this.draw(); return; }
    this.armed = false;
    for (const [name, val] of Object.entries(this.defaults)) {
      // preview_size is what the clip was CACHED at; changing it without a
      // re-run would leave the widget disagreeing with the frames on hand.
      if (name === "preview_size") continue;
      const wd = this.w(name);
      if (wd) wd.value = val;
    }
    this.playing = false;
    this.auditioning = false;
    this.mode = "warp";
    this.shownSig = null;
    app.graph?.setDirtyCanvas(true, false);
    this.request();
  }

  // --- keyframes -----------------------------------------------------------
  // The playhead IS the frame number, so a keyframe needs no spreading after
  // the fact — which is the whole reason this is nicer than dropping them on
  // the orbit sphere, where the widget has to guess the timing.
  kfs() {
    const raw = this.w("keyframes")?.value;
    if (!raw) return [];
    try {
      const a = JSON.parse(raw);
      return Array.isArray(a) ? a.filter((k) => k && Number.isFinite(+k.f)) : [];
    } catch (e) { return []; }      // hand-edited into invalid JSON: the server will say so
  }

  keyIndex() {
    const f = this.frame();
    return this.kfs().findIndex((k) => Math.round(+k.f) === f);
  }

  writeKfs(list) {
    list.sort((a, b) => a.f - b.f);
    const wd = this.w("keyframes");
    if (!wd) return;
    wd.value = JSON.stringify(list);
    // A single keyframe is not a move; the node treats it as a static pose, so
    // the switch follows the path rather than leaving a dead flag set.
    const uk = this.w("use_keyframes");
    if (uk) uk.value = list.length >= 2;
    wd.callback?.(wd.value);
    app.graph?.setDirtyCanvas(true, false);
  }

  poseNow() {
    return {
      az: this.w("azimuth")?.value ?? 0,
      el: this.w("elevation")?.value ?? 0,
      dist: this.w("distance")?.value ?? 1,
      // the lens shift is part of the framing, so it belongs to the keyframe
      vs: this.w("vertical_shift")?.value ?? 0,
    };
  }

  keying() {
    return !!this.w("use_keyframes")?.value && this.kfs().length >= 2;
  }

  // Called before a drag or a wheel step. On a keyframe, the drag edits it. Off
  // one, the path is what drives the camera, so the widgets are stale — they get
  // seeded from the pose actually on screen first, or the first drag would snap
  // the camera to wherever the widgets were left.
  beginAudition() {
    if (this.auditioning || !this.keying() || this.keyIndex() >= 0) return;
    const ps = this.pivot?.pose;
    if (ps) {
      const set = (n, v, d) => {
        const wd = this.w(n);
        if (wd && Number.isFinite(v)) wd.value = Math.round(v * 10 ** d) / 10 ** d;
      };
      set("azimuth", ps.az, 1); set("elevation", ps.el, 1);
      set("distance", ps.dist, 2); set("vertical_shift", ps.vs, 2);
    }
    this.auditioning = true;
  }

  toggleKey() {
    const list = this.kfs();
    const i = this.keyIndex();
    if (i >= 0) list.splice(i, 1);
    else list.push({ f: this.frame(), ...this.poseNow() });
    this.auditioning = false;      // committed (or removed): back on the path
    this.writeKfs(list);
  }

  // Dragging edits the keyframe under the playhead, if there is one. Between
  // keyframes the path is what drives the render, so a drag there would write
  // into widgets that no longer feed anything — the readout says so instead.
  syncKey() {
    if (!this.w("use_keyframes")?.value) return;
    const i = this.keyIndex();
    if (i < 0) return;
    const list = this.kfs();
    list[i] = { f: this.frame(), ...this.poseNow() };
    this.writeKfs(list);
  }

  modeGeom() {
    const W = this.canvas.width, H = this.canvas.height;
    return { x0: W - MODE_W - 6, x1: W - 6, y0: H - CTRL_H + 3, y1: H - CTRL_H + 21 };
  }

  scrubTo(x) {
    const g = this.barGeom();
    const t = clamp((x - g.x0) / Math.max(1, g.x1 - g.x0), 0, 1);
    this.setFrame(1 + t * (this.frames() - 1));
  }

  onDown(e) {
    if (e.button !== 0) return;
    const p = this.pos(e);
    const g = this.barGeom();
    const m = this.modeGeom();
    const k = this.keyGeom();
    const r = this.rstGeom();
    const hitRst = p.y >= g.top && p.x >= r.x0 && p.x <= r.x1;
    if (!hitRst && this.armed) { this.armed = false; this.draw(); }   // anything else disarms
    if (p.y >= g.top) {
      if (hitRst) {
        this.reset();
      } else if (p.x >= k.x0 && p.x <= k.x1) {
        this.toggleKey();
      } else if (p.x >= m.x0) {
        this.mode = this.mode === "warp" ? "source" : "warp";
        this.playing = false;
        this.request();
      } else if (p.x <= BTN_W + 4) {
        this.playing = !this.playing;
        this.draw();
        if (this.playing) this.step(1, true);
      } else {
        this.playing = false;
        this.drag = { mode: "scrub" };
        this.scrubTo(p.x);
      }
    } else {
      this.beginAudition();
      this.drag = {
        mode: "orbit", x: e.clientX, y: e.clientY,
        az: this.w("azimuth")?.value ?? 0,
        el: this.w("elevation")?.value ?? 0,
      };
      this.canvas.style.cursor = "grabbing";
    }
    this.canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  }

  onMove(e) {
    if (!this.drag) return;
    if (this.drag.mode === "scrub") {
      this.scrubTo(this.pos(e).x);
    } else {
      const az = clamp(this.drag.az + (e.clientX - this.drag.x) * DRAG_SENS, -180, 180);
      const el = clamp(this.drag.el - (e.clientY - this.drag.y) * DRAG_SENS, -90, 90);
      this.setNum("azimuth", az);
      this.setNum("elevation", el);
      this.syncKey();
    }
    e.preventDefault();
  }

  onUp(e) {
    if (!this.drag) return;
    this.drag = null;
    try { this.canvas.releasePointerCapture(e.pointerId); } catch (err) { /* already gone */ }
    this.canvas.style.cursor = "grab";
  }

  onWheel(e) {
    // Over the control strip the wheel steps the playhead, over the image it
    // dollies — the surface under the cursor says which one you meant. Wheel
    // back (deltaY > 0) raises the number in both cases, matching the orbit
    // widget and the graph's own numeric widgets.
    if (this.pos(e).y >= this.barGeom().top) {
      this.playing = false;
      this.step(Math.sign(e.deltaY), false);
    } else {
      this.beginAudition();
      const d = this.w("distance");
      if (d) {
        this.setNum("distance", clamp(d.value + Math.sign(e.deltaY) * 0.05, 0.1, 3.0), 2);
        this.syncKey();
      }
    }
    e.preventDefault();
    e.stopPropagation();
  }

  // The pivot marker, plus the numbers that actually carry the information.
  // Its screen position is the boring half — the camera looks AT the pivot, so
  // in the warped view it sits dead centre unless keep_source_aim is on with an
  // off-axis pivot. The depth comparison is the part worth reading.
  drawPivot(ctx) {
    const p = this.pivot;
    if (!p || !this.rect || !this.img) return;
    const src = this.mode === "source";
    const u = src ? p.src_u : p.u;
    const v = src ? p.src_v : p.v;
    const col = p.auto ? "230,200,90" : "120,190,255";

    if (Number.isFinite(u) && Number.isFinite(v)) {
      const x = this.rect.x + u * (this.rect.w / this.img.width);
      const y = this.rect.y + v * (this.rect.h / this.img.height);
      const inside = x >= this.rect.x && x <= this.rect.x + this.rect.w &&
                     y >= this.rect.y && y <= this.rect.y + this.rect.h;
      if (inside) {
        ctx.save();
        ctx.strokeStyle = `rgba(${col},0.95)`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x - 11, y); ctx.lineTo(x - 4, y);
        ctx.moveTo(x + 4, y);  ctx.lineTo(x + 11, y);
        ctx.moveTo(x, y - 11); ctx.lineTo(x, y - 4);
        ctx.moveTo(x, y + 4);  ctx.lineTo(x, y + 11);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }

    const bits = [`pivot z ${p.z.toFixed(2)}`];
    if (Number.isFinite(p.scene_z)) {
      const d = p.z - p.scene_z;
      bits.push(`scene ${p.scene_z.toFixed(2)}`);
      // positive = the pivot sits deeper than the surface under it, i.e. you are
      // orbiting a point buried behind the thing you are looking at
      bits.push(`${d >= 0 ? "+" : ""}${d.toFixed(2)} ${d >= 0 ? "behind" : "in front"}`);
    }
    bits.push(`r ${p.radius.toFixed(2)}`);
    if (p.auto) bits.push("auto");
    ctx.font = "10px monospace";
    const t = bits.join("  ");
    const tw = ctx.measureText(t).width;
    ctx.fillStyle = "rgba(20,22,28,0.65)";
    ctx.fillRect(4, 4, tw + 8, 14);
    ctx.fillStyle = `rgba(${col},0.9)`;
    ctx.fillText(t, 8, 14);
  }

  draw() {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    const boxH = H - CTRL_H;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#1a1c22";
    ctx.fillRect(0, 0, W, H);

    const pad = 6;
    this.rect = null;
    if (this.img) {
      const s = Math.min((W - pad * 2) / this.img.width, (boxH - pad * 2) / this.img.height);
      const dw = this.img.width * s, dh = this.img.height * s;
      this.rect = { x: (W - dw) / 2, y: (boxH - dh) / 2, w: dw, h: dh };
      ctx.globalAlpha = this.stale ? 0.5 : 1.0;
      ctx.drawImage(this.img, this.rect.x, this.rect.y, dw, dh);
      ctx.globalAlpha = 1.0;
      this.drawPivot(ctx);
    }

    ctx.font = "11px monospace";
    if (this.status) {
      ctx.fillStyle = "rgba(230,200,90,0.85)";
      ctx.fillText(this.status, (W - ctx.measureText(this.status).width) / 2, boxH / 2);
    }

    const g = this.barGeom();
    const m = this.modeGeom();
    const k = this.keyGeom();
    const r = this.rstGeom();
    const n = this.frames(), f = this.frame();
    const kfs = this.kfs();
    const onKey = this.keyIndex() >= 0;
    const keying = !!this.w("use_keyframes")?.value;

    // keyframe drop / remove, lit when the playhead sits on one
    ctx.fillStyle = onKey ? "rgba(120,190,255,0.28)" : "rgba(255,255,255,0.07)";
    ctx.fillRect(k.x0, k.y0, k.x1 - k.x0, k.y1 - k.y0);
    ctx.font = "10px monospace";
    ctx.fillStyle = onKey ? "rgba(170,210,255,0.95)" : DIM;
    ctx.fillText("key", k.x0 + (k.x1 - k.x0 - ctx.measureText("key").width) / 2, k.y1 - 6);

    // source / warp toggle
    ctx.fillStyle = this.mode === "source" ? "rgba(120,190,255,0.22)" : "rgba(255,255,255,0.07)";
    ctx.fillRect(m.x0, m.y0, m.x1 - m.x0, m.y1 - m.y0);
    ctx.font = "10px monospace";
    ctx.fillStyle = this.mode === "source" ? "rgba(170,210,255,0.95)" : DIM;
    const mt = this.mode === "source" ? "source" : "warp";
    ctx.fillText(mt, m.x0 + (m.x1 - m.x0 - ctx.measureText(mt).width) / 2, m.y1 - 6);

    // play / pause
    ctx.fillStyle = this.img ? "rgba(225,229,240,0.85)" : DIM;
    const bx = 8, by = g.y - 7;
    if (this.playing) {
      ctx.fillRect(bx, by, 4, 14);
      ctx.fillRect(bx + 7, by, 4, 14);
    } else {
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(bx + 12, by + 7);
      ctx.lineTo(bx, by + 14);
      ctx.closePath();
      ctx.fill();
    }

    // reset to defaults
    ctx.fillStyle = this.armed ? "rgba(230,200,90,0.30)" : "rgba(255,255,255,0.07)";
    ctx.fillRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
    ctx.font = "10px monospace";
    ctx.fillStyle = this.armed ? "rgba(240,215,120,0.95)" : DIM;
    const rt = this.armed ? "sure?" : "reset";
    ctx.fillText(rt, r.x0 + (r.x1 - r.x0 - ctx.measureText(rt).width) / 2, r.y1 - 6);

    // scrub bar
    ctx.fillStyle = "rgba(255,255,255,0.10)";
    ctx.fillRect(g.x0, g.y - 2, g.x1 - g.x0, 4);
    if (n > 1) {
      const t = (f - 1) / (n - 1);
      ctx.fillStyle = "rgba(120,170,255,0.55)";
      ctx.fillRect(g.x0, g.y - 2, (g.x1 - g.x0) * t, 4);
      // keyframe ticks, so the timing of the move is readable at a glance
      ctx.fillStyle = keying ? "rgba(130,200,255,0.9)" : "rgba(130,200,255,0.35)";
      for (const kf of kfs) {
        const tt = clamp((Math.round(+kf.f) - 1) / (n - 1), 0, 1);
        ctx.fillRect(g.x0 + (g.x1 - g.x0) * tt - 1, g.y - 7, 2, 14);
      }
      ctx.beginPath();
      ctx.arc(g.x0 + (g.x1 - g.x0) * t, g.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = onKey ? "rgba(210,235,255,1.0)" : "rgba(150,190,255,0.95)";
      ctx.fill();
    }

    // The readout. These numbers are the point of the node for now: find the
    // pose here, then type it into the CrossView Warp node.
    // While a move is running the widgets are not what is on screen, so the
    // server sends back the pose it actually rendered.
    const ps = this.pivot?.pose;
    const v = (key, name, d = 1) =>
      (ps ? ps[key] : (this.w(name)?.value ?? 0)).toFixed(d);
    ctx.font = "12px monospace";
    ctx.fillStyle = "rgba(225,229,240,0.92)";
    ctx.fillText(`az ${v("az", "azimuth")}   el ${v("el", "elevation")}   ` +
                 `dist ${v("dist", "distance", 2)}`, pad, H - 5);

    ctx.font = "10px monospace";
    const right = [`f${f}/${n}`];
    // Between keyframes the path drives the camera, so dragging would write
    // into widgets that no longer feed the render. Say so rather than let it
    // look broken.
    if (this.auditioning) right.push(`kf ${kfs.length} aiming - KEY to commit`);
    else if (keying && kfs.length >= 2) right.push(onKey ? `kf ${kfs.length} on key`
                                                         : `kf ${kfs.length} between`);
    else if (kfs.length) right.push(`kf ${kfs.length}`);
    if (this.ms != null) right.push(`${this.ms} ms`);
    if (this.info?.mb) right.push(`${this.info.mb} MB`);
    if (this.stale) right.push("stale");
    const t = right.join("  ");
    ctx.fillStyle = this.stale ? "rgba(230,200,90,0.8)" : DIM;
    ctx.fillText(t, W - pad - ctx.measureText(t).width, H - 5);
  }
}

// Two copies of this package can be installed side by side (a dev worktree next
// to the released one), so the extension name must be unique per directory.
const PKG = (import.meta.url.match(/\/extensions\/([^/]+)\//) || [, "core"])[1];
const BASE = "CrossViewPreview";

app.registerExtension({
  name: `crossview.livePreview.${PKG}`,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name.startsWith(BASE)) return;
    // The node id carries the same local suffix the Python side puts on the
    // route path, so deriving the endpoint from it keeps the two copies apart
    // without any extra plumbing. Same sanitising as _SLUG in the node module.
    const slug = nodeData.name.slice(BASE.length).replace(/[^A-Za-z0-9_-]/g, "");
    const endpoint = `/crossview_preview${slug}/render`;
    // Read the defaults out of the node definition itself rather than keeping a
    // second copy here that would quietly fall behind the Python.
    const DEFAULTS = {};
    for (const sec of ["required", "optional"]) {
      for (const [name, spec] of Object.entries(nodeData?.input?.[sec] || {})) {
        if (!Array.isArray(spec)) continue;
        if (spec[1] && typeof spec[1] === "object" && "default" in spec[1]) DEFAULTS[name] = spec[1].default;
        else if (Array.isArray(spec[0])) DEFAULTS[name] = spec[0][0];   // combo, no explicit default
      }
    }
    console.log(`[CrossView Preview] registering widget on ${nodeData.name} -> ${endpoint}`);

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try {
        const node = this;
        // Both installed copies match this node by prefix and wrap
        // onNodeCreated, so without a guard the canvas would be added twice.
        if (node.widgets?.some((w) => w.name === "preview")) return r;

        const container = document.createElement("div");
        container.style.width = "100%";
        const canvas = document.createElement("canvas");
        canvas.width = 384;
        canvas.height = 340;
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        canvas.style.borderRadius = "8px";
        canvas.style.cursor = "grab";
        canvas.style.touchAction = "none";     // pointer events, not scrolling
        container.appendChild(canvas);

        const h = () => Math.round(
          Math.max(WIDGET_H, Math.min(H_MAX, (node.size?.[0] ?? 384) * 0.85)));

        const wd = node.addDOMWidget("preview", "crossviewPreviewWidget", container, {
          serialize: false,
          hideOnZoom: false,
          getMinHeight: h,
          getMaxHeight: h,
          getHeight: h,
        });
        // options.serialize is not the flag litegraph's serialize() reads —
        // without this the canvas is written into widgets_values as a stray "",
        // shifting every value stored after it.
        if (wd) wd.serialize = false;

        node._crossviewPreview = new PreviewEditor(node, canvas, endpoint, DEFAULTS);

        const onRemoved = node.onRemoved;
        node.onRemoved = function () {
          if (node._crossviewPreview) node._crossviewPreview.playing = false;
          node._crossviewPreview = null;
          return onRemoved?.apply(this, arguments);
        };
      } catch (e) {
        console.error("[CrossView Preview] widget setup failed", e);
      }
      return r;
    };

    // A run has just re-cached the clip, so whatever is on screen is current
    // again and a fresh render is worth asking for immediately.
    const onExec = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExec?.apply(this, arguments);
      const ed = this._crossviewPreview;
      if (ed) {
        const info = message?.crossview_preview?.[0];
        if (info) {
          ed.info = info;
          ed.setCount(info.of);
        }
        ed.stale = false;
        ed.shownSig = null;        // force a re-render against the new clip
        ed.request();
      }
      return r;
    };
  },

  // Rewiring the inputs means the cached clip may no longer be what the graph
  // would produce. The preview cannot know the new frames without a run, so it
  // says so rather than presenting a confident picture of the old ones.
  nodeCreated(node) {
    if (!node.comfyClass?.startsWith(BASE)) return;
    const onConn = node.onConnectionsChange;
    node.onConnectionsChange = function () {   // args forwarded via `arguments`
      const r = onConn?.apply(this, arguments);
      const ed = this._crossviewPreview;
      if (ed && ed.info) {
        ed.stale = true;
        ed.playing = false;
        ed.draw();
      }
      return r;
    };
  },
});
