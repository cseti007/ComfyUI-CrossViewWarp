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

    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture?.(e.pointerId);
      this.onDown(e);
    });
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener("dblclick", () => {
      this.view.viewYaw = 0.24; this.view.viewTilt = 0.20;
      this.render(true);
    });
    this._onMove = (e) => this.onMove(e);
    this._onUp = () => this.onUp();

    // repaint when the number widgets change externally (cheap change-detect)
    this._raf = () => {
      if (!this.canvas.isConnected) return;   // node removed -> stop
      this.render(false);
      requestAnimationFrame(this._raf);
    };
    requestAnimationFrame(this._raf);
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
    // sphere = the 1.0x reference shell; leave margin so the camera can sit
    // OUTSIDE it (dist > 1) up to ~1.3R without clipping
    return { cx: W / 2, cy: H / 2, R: (S / 2 - 4) * 0.76 };
  }
  onDown(e) {
    e.preventDefault(); e.stopPropagation();
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
    const wD = getW(this.node, "distance");
    if (wD) {
      wD.value = Math.max(0.2, Math.min(3.0,
        Math.round((wD.value - Math.sign(e.deltaY) * 0.05) * 100) / 100));
      this.node.setDirtyCanvas(true, true);
      this.render(true);
    }
  }
  render(force) {
    // keep the backing resolution in sync with the element's layout size so
    // resizing the node doesn't stretch (deform) the globe; geom() uses
    // min(W,H) so the sphere stays circular at any aspect ratio
    const cw = this.canvas.clientWidth | 0, chh = this.canvas.clientHeight | 0;
    if (cw > 0 && chh > 0 && (this.canvas.width !== cw || this.canvas.height !== chh)) {
      this.canvas.width = cw;
      this.canvas.height = chh;
      this.cacheKey = "";   // sphere cache is size-dependent -> rebuild
    }
    const { az, el, dist } = this.vals();
    const g = this.geom();
    const key = `${az}|${el}|${dist}|${this.view.viewYaw.toFixed(3)}|${this.view.viewTilt.toFixed(3)}|${this.canvas.width}x${this.canvas.height}`;
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
    const distF = Math.max(0.45, Math.min(1.3, 0.45 + 0.55 * Number(dist)));
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
        node.setSize(node.computeSize());
        console.log("[CrossView Orbit] DOM widget attached to node", node.id);
      } catch (e) {
        console.error("[CrossView Orbit] widget attach FAILED:", e);
      }
      return r;
    };
  },
});
