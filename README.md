# ComfyUI-CrossViewWarp

The companion ComfyUI node for my
[CrossView-Warp IC-LoRA](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp)
(LTX-Video 2.3, 22B). You give it a video and a camera offset, and it builds
the depth-warp conditioning the LoRA expects: the input reprojected into the
new viewpoint, with magenta holes where the original camera never saw
anything.

The node has a built-in 3D orbit picker widget — a sphere around the subject
where you drag the camera marker instead of typing angles. The green/yellow
shading on it shows the ranges the LoRA was trained for.

It also does camera *moves*: right-click the sphere to drop keyframes, and the
node interpolates a pose per frame instead of holding one for the whole clip.

This whole thing is a proof of concept. The LoRA card lists what works and
what doesn't — read it before expecting magic:
https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/cseti007/ComfyUI-CrossViewWarp
```

Then restart ComfyUI and refresh the browser. There are no required packages to
install — the node runs on NumPy + PyTorch, which ComfyUI already provides.

### Optional: faster warp (numba)

The warp step splats every source pixel into the new view in a tight loop. With
[`numba`](https://numba.pydata.org/) installed, that loop is JIT-compiled and the
warp runs ~10x faster. Without it the node uses a pure-NumPy fallback and produces
the exact same result, just slower — so numba is a pure speed-up, never required.

It is deliberately **not** in `requirements.txt` / `pyproject.toml`, so
ComfyUI-Manager won't pull it in automatically. The reason: `numba` depends on
`llvmlite` and pins a bounded NumPy version range, so installing it into an
existing ComfyUI environment can force a NumPy up-/down-grade that breaks other
custom nodes. Only add it if you want the speed-up and know your environment can
take it:

```
pip install numba
```

## Prerequisites

- A recent ComfyUI with LTX-2 support (the `LTXAddVideoICLoRAGuide` node)
- A Depth Anything V2 node pack for the depth input — I use
  [kijai/ComfyUI-DepthAnythingV2](https://github.com/kijai/ComfyUI-DepthAnythingV2)
- [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
  for video load/save
- The [CrossView-Warp LoRA](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp)
  itself — the node only builds the conditioning, the LoRA does the generation
- Optional: `opencv-contrib-python` — the `smooth_depth` option uses its
  guided filter when available, and falls back to a bilateral filter when not

## Nodes

**CrossView Warp (video -> warp)** — the main node.

| Output | What it is |
|---|---|
| `warp` | the warp control video — wire to an IC-LoRA reference guide (`latent_downscale_factor = 1`) |
| `orbit_view` | a rendered image of the orbit sphere with your camera setup — the single pose, or the whole keyframed path with its frame-numbered markers — wire to a PreviewImage/SaveImage to document what you asked for |

Key inputs (all have tooltips in the UI):

- `azimuth` / `elevation` / `distance` — the camera offset. Stay in the green
  zone of the picker: azimuth up to ±45°, elevation +30°/−15° (yellow up to
  ±65° works with degrading quality). Avoid near-zero angles — the LoRA
  misbehaves when the warp is almost identical to the source. **Important: Unfortunately distance doesn't work as expected due to some dataset problems which will be solved in the next release.**
- `depth_ratio` (default 6.0) — how much depth relief the warp gets. The low
  default keeps the subject readable in the warp; raising it gives more
  parallax but shreds cluttered scenes.
- `pivot_override` + `pivot_x/y/z` (default on, z=1.05) The default puts the pivot on the
  nearest thing in the middle of the frame, which is usually your subject.
- `smooth_depth` — (default off) edge-aware depth smoothing, fewer speckle holes in the warp.
- `roll_lock` (default on) — keeps the subject's in-image lean the same as the
  source, so a tilted source shot doesn't tip the character over at large
  angles
- `use_keyframes` / `keyframes` / `frame_count` / `interp_motion` /
  `interpolation` — a camera move instead of a single pose. See
  [Keyframed camera move](#keyframed-camera-move-right-click) below.

### Keyframed camera move (right-click)

**Right-click** the orbit sphere to place a camera keyframe. The first one also
captures your current azimuth/elevation/distance as the starting keyframe, so a
single right-click gives you a complete move. Left-drag a marker to move it,
hover it and scroll the wheel to dolly it, right-click it again to delete it.
Each marker is labelled with its **frame number**.

The gesture is deliberately mouse-only. A modifier key can be claimed by any
other node pack through ComfyUI's keybinding system — KJNodes, for one, binds S
to a node-swap gesture that disconnects and repositions whatever node is under
the cursor — so the sphere uses a plain mouse button that nothing else can
intercept. Note that right-clicking inside the sphere therefore does not open the
node's context menu; right-click the node's title or body for that.

Placing a second keyframe enables `use_keyframes` and hides the static
azimuth/elevation/distance (they do nothing while a move is running, and come
back when you clear the path). Toggling `use_keyframes` off leaves the markers
visible but dimmed, so you can compare against the static pose without losing
the setup.

The path itself lives in the `keyframes` widget as JSON, and you can edit it by
hand — which is currently the way to fine-tune the timing:

```json
[{"f":1,"az":-30,"el":20,"dist":1.0},{"f":49,"az":45,"el":10,"dist":1.2}]
```

`f` is an absolute frame number, counted from 1. Before the first and after the
last keyframe the camera **holds**, so a path may cover only part of the clip —
"swing for the first two seconds, then sit still" is just a path that ends early.
A keyframe past the end of the clip is an error rather than a silently truncated
move, so a path authored for a longer clip tells you instead of quietly doing the
wrong thing.

`dist` carries the same caveat as the static `distance` above: the current LoRA
does not follow it reliably, so keyframing a dolly move will disappoint until the
next release. Angles are what this is good at.

Set `frame_count` to your clip's length and the sphere will spread keyframes
evenly across it — the last one always lands on the final frame, and adding or
deleting a keyframe re-spreads the rest:

```
frame_count = 97,  2 keyframes -> f = 1, 97
                   3 keyframes -> f = 1, 49, 97
                   5 keyframes -> f = 1, 25, 49, 73, 97
```

Hand-edit any frame number in the widget and the even spread stops being
re-applied, so your timing survives further edits on the sphere. Leave
`frame_count` at 0 and new keyframes are simply placed 24 frames apart, as the
widget has no way of knowing the clip length on its own.

Changing `frame_count` refits the existing path onto the new length: the first
keyframe moves to 1, the last to the new final frame, and the ones between keep
their relative spacing. So it also works the other way round — place keyframes
first, then type the clip length, and they stretch to fit.

### Feeding `keyframes` from another node

Convert `keyframes` to an input and the sphere becomes read-only: the node will
render whatever the upstream sends, so letting you edit markers here would show a
path that is not the one produced. If the upstream is a literal (a primitive or
string-constant node) the sphere previews its path; if the string is computed at
execution time there is nothing to preview, and the sphere says so rather than
leaving a stale local path on screen.

Two options shape the result:

- `interpolation` — `linear` gives straight legs with a corner at each keyframe;
  `smooth` runs a Catmull-Rom spline through them (still passing exactly through
  every keyframe). With only two keyframes the two are identical.
- `interp_motion` — timing between consecutive keyframes. Applied per segment,
  so `ease_in_out` settles the camera into every keyframe. Pair easing with
  `linear`; combining it with `smooth` cancels the very continuity the spline
  is there to provide.

## Generation settings that worked for me

- IC-LoRA strength **1.3** for characters, **1.0–1.15** for cars and other
  content the base model has strong opinions about
- Both IC-LoRA reference guides at `latent_downscale_factor = 1`
- For `distance > 1`: describe the newly revealed content in the text prompt —
  the warp can't know what's outside the original frame. 

The full settings story and all the limitations are on the
[LoRA card](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp).

## Support

If this is useful and you want me to keep doing open work like this:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/chetyart)

## License

Apache-2.0
