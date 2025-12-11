"""
Wrapper module that imports from the meshoptimizer package.
Meshoptimizer is required - no fallback to trimesh.
"""
import sys
from pathlib import Path

# Try to import from installed meshoptimizer package
try:
    from meshoptimizer import simplify_with_screen_error
except ImportError:
    # Try to import from local path
    meshopt_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "meshoptimizer"
    if meshopt_dir.exists():
        sys.path.insert(0, str(meshopt_dir))
        try:
            from meshoptimizer import simplify_with_screen_error
        except ImportError as e:
            raise ImportError(
                f"meshoptimizer package is required but not found. "
                f"Please ensure it is installed via uv_init. "
                f"Error: {e}"
            ) from e
    else:
        raise ImportError(
            f"meshoptimizer package is required but not found. "
            f"Expected location: {meshopt_dir}. "
            f"Please ensure it is installed via uv_init."
        )

