"""CrossView-Warp ComfyUI node package.

Builds the cross-view WARP conditioning that the CrossView-Warp IC-LoRA
expects, from an input video + a requested relative camera pose. Feed it a
depth map from an external Depth Anything V2 node.

Install: symlink/copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
"""

from .crossview_warp_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# serves web/crossview_orbit.js -> the 3D sphere orbit picker widget
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
