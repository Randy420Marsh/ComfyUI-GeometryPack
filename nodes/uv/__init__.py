"""
UV nodes aggregation.

The unified UV Unwrap node consolidates all UV unwrapping methods into a single node.
Old separate nodes are kept in the codebase for reference but are no longer registered.
"""

# New unified UV unwrap node
from .uv_unwrap import UVUnwrapNode, NODE_CLASS_MAPPINGS as UV_UNWRAP_MAPS, NODE_DISPLAY_NAME_MAPPINGS as UV_UNWRAP_DISP

# Old separate nodes (kept for reference, not registered)
# from .xatlas_unwrap import XAtlasUVUnwrapNode, NODE_CLASS_MAPPINGS as XATLAS_UNWRAP_MAPS, NODE_DISPLAY_NAME_MAPPINGS as XATLAS_UNWRAP_DISP
# from .libigl_lscm import LibiglLSCMNode, NODE_CLASS_MAPPINGS as LIBIGL_LSCM_MAPS, NODE_DISPLAY_NAME_MAPPINGS as LIBIGL_LSCM_DISP
# from .libigl_harmonic import LibiglHarmonicNode, NODE_CLASS_MAPPINGS as LIBIGL_HARMONIC_MAPS, NODE_DISPLAY_NAME_MAPPINGS as LIBIGL_HARMONIC_DISP
# from .libigl_arap import LibiglARAPNode, NODE_CLASS_MAPPINGS as LIBIGL_ARAP_MAPS, NODE_DISPLAY_NAME_MAPPINGS as LIBIGL_ARAP_DISP
# from .blender_uv import BlenderUVNode, NODE_CLASS_MAPPINGS as BLENDER_UV_MAPS, NODE_DISPLAY_NAME_MAPPINGS as BLENDER_UV_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register only the unified node
NODE_CLASS_MAPPINGS.update(UV_UNWRAP_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(UV_UNWRAP_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
