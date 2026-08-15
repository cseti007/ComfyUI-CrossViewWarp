# Example workflows

## `ltx2.3-ic-lora-crossview-warp-v2.json`

End-to-end Video-to-Video workflow for the
[CrossView Warp IC-LoRA v2](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2)
(LTX-Video 2.3, 22B). Loads an input clip, builds the warp with the
**CrossViewWarp** node from MoGe-2 geometry, feeds both the warp and the source
clip to IC-LoRA reference guides, and runs an 8-step base pass plus a 2x latent
spatial upscale pass with audio. Ships with `use_keyframes` on and a 3-point
camera path.

Drag-and-drop the `.json` into ComfyUI to load it.

Full requirements (models, other custom nodes) and usage notes:
[ComfyUI-Workflows / ic-lora-crossview-warp-v2](https://huggingface.co/datasets/Cseti/ComfyUI-Workflows/blob/main/ltx/2.3/ic-lora-crossview-warp-v2/README.md)

Key settings: IC-LoRA strength `1`, both IC-LoRA guides
`latent_downscale_factor = 1`, distilled speed LoRA `0.6`. See the
[LoRA card](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2)
for the angle ranges and limitations.

## `ltx2.3-ic-lora-crossview-warp.json`

End-to-end Video-to-Video workflow for the
[CrossView Warp IC-LoRA](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp)
(LTX-Video 2.3, 22B). Loads an input clip, builds the depth-warp with the
**CrossViewWarp** node (Depth-Anything-V2 depth), feeds both the warp and the
source clip to IC-LoRA reference guides, and runs an 8-step base pass plus a 2x
latent spatial upscale pass with audio.

Drag-and-drop the `.json` into ComfyUI to load it.

Full requirements (models, other custom nodes) and usage notes:
[ComfyUI-Workflows / ic-lora-crossview-warp](https://huggingface.co/datasets/Cseti/ComfyUI-Workflows/blob/main/ltx/2.3/ic-lora-crossview-warp/README.md)

Key settings: IC-LoRA strength `1.3`, both IC-LoRA guides
`latent_downscale_factor = 1`, distilled speed LoRA `0.6`. See the
[LoRA card](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp) for
the angle ranges and limitations.
