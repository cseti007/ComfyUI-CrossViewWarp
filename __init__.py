"""CrossView-Warp ComfyUI node package.

Builds the cross-view WARP conditioning that the CrossView-Warp IC-LoRA
expects, from an input video + a requested relative camera pose. Feed it a
depth map from an external Depth Anything V2 node.

Install: symlink/copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
"""

from .crossview_warp_node import NODE_CLASS_MAPPINGS as _WARP_CLASSES
from .crossview_warp_node import NODE_DISPLAY_NAME_MAPPINGS as _WARP_NAMES
from .crossview_preview_node import NODE_CLASS_MAPPINGS as _PREVIEW_CLASSES
from .crossview_preview_node import NODE_DISPLAY_NAME_MAPPINGS as _PREVIEW_NAMES

NODE_CLASS_MAPPINGS = {**_WARP_CLASSES, **_PREVIEW_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**_WARP_NAMES, **_PREVIEW_NAMES}

# serves web/crossview_orbit.js (the 3D sphere orbit picker) and
# web/crossview_preview.js (the draggable live warp preview)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
