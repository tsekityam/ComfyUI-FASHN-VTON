from .model_paths import register_fashn_vton_model_paths
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Register models/fashn-vton with ComfyUI Model Library / folder_paths on import.
register_fashn_vton_model_paths()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
