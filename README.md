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

This whole thing is a proof of concept. The LoRA card lists what works and
what doesn't — read it before expecting magic:
https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/cseti007/ComfyUI-CrossViewWarp
pip install -r ComfyUI-CrossViewWarp/requirements.txt
```

Then restart ComfyUI and refresh the browser.

`numba` is the only dependency and it's optional in practice — without it the
node still works, just ~10x slower on the warp step.

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
| `orbit_view` | a rendered image of the orbit sphere with your camera setup — wire to a PreviewImage/SaveImage to document what you asked for |

Key inputs (all have tooltips in the UI):

- `azimuth` / `elevation` / `distance` — the camera offset. Stay in the green
  zone of the picker: azimuth up to ±45°, elevation +30°/−15° (yellow up to
  ±65° works with degrading quality). Avoid near-zero angles — the LoRA
  misbehaves when the warp is almost identical to the source.
- `depth_ratio` (default 1.5) — how much depth relief the warp gets. The low
  default keeps the subject readable in the warp; raising it gives more
  parallax but shreds cluttered scenes.
- `pivot_override` + `pivot_x/y/z` (default on, z=1.05) — orbit around the
  subject instead of the scene center. The default puts the pivot on the
  nearest thing in the middle of the frame, which is usually your subject.
- `smooth_depth` — edge-aware depth smoothing, fewer speckle holes in the warp
- `roll_lock` (default on) — keeps the subject's in-image lean the same as the
  source, so a tilted source shot doesn't tip the character over at large
  angles
- `metric_depth` + the **CrossView Metric Depth** helper node — feed raw
  metric depth (meters) instead of normalized depth, for experiments

## Generation settings that worked for me

- IC-LoRA strength **1.3** for characters, **1.0–1.15** for cars and other
  content the base model has strong opinions about
- Both IC-LoRA reference guides at `latent_downscale_factor = 1`
- Distilled-base LoRA at 0.6
- For `distance > 1`: describe the newly revealed content in the text prompt —
  the warp can't know what's outside the original frame

The full settings story and all the limitations are on the
[LoRA card](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp).

## Support

If this is useful and you want me to keep doing open work like this:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/chetyart)

## License

Apache-2.0
