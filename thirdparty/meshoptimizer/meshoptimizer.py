"""
Python bindings for meshoptimizer using ctypes.
Provides error-aware mesh simplification with screen-space arc angle metrics.
"""
import ctypes
import numpy as np
import os
from pathlib import Path

# Try to load meshoptimizer library
_meshopt_lib = None

def _load_meshoptimizer_lib():
    """Load meshoptimizer shared library"""
    global _meshopt_lib
    
    if _meshopt_lib is not None:
        return _meshopt_lib
    
    # Try to find the library in common locations
    # First, try to find it relative to this package
    package_dir = Path(__file__).parent
    
    # Try different library names and locations
    lib_paths = [
        package_dir / "build" / "libmeshoptimizer.so",
        package_dir / "build" / "libmeshoptimizer.dylib",
        package_dir / "build" / "meshoptimizer.dll",
        package_dir / "libmeshoptimizer.so",
        package_dir / "libmeshoptimizer.dylib",
        package_dir / "meshoptimizer.dll",
    ]
    
    for lib_path in lib_paths:
        if lib_path.exists():
            try:
                _meshopt_lib = ctypes.CDLL(str(lib_path))
                _setup_function_signatures(_meshopt_lib)
                return _meshopt_lib
            except OSError:
                continue
    
    # If not found, try to load from system
    lib_names = [
        "libmeshoptimizer.so",
        "libmeshoptimizer.dylib",
        "meshoptimizer.dll",
    ]
    for lib_name in lib_names:
        try:
            _meshopt_lib = ctypes.CDLL(lib_name)
            _setup_function_signatures(_meshopt_lib)
            return _meshopt_lib
        except OSError:
            continue
    
    raise ImportError(
        "Could not load meshoptimizer library. "
        "Please build it first using: cd thirdparty/meshoptimizer && ./build.sh"
    )


def _setup_function_signatures(lib):
    """Setup function signatures for meshoptimizer C API"""
    # meshopt_simplify signature
    lib.meshopt_simplify.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # destination
        ctypes.POINTER(ctypes.c_uint32),   # indices
        ctypes.c_size_t,                    # index_count
        ctypes.POINTER(ctypes.c_float),    # vertex_positions
        ctypes.c_size_t,                    # vertex_count
        ctypes.c_size_t,                    # vertex_positions_stride
        ctypes.c_size_t,                    # target_index_count
        ctypes.c_float,                     # target_error
        ctypes.c_uint32,                    # options
        ctypes.POINTER(ctypes.c_float),     # result_error
    ]
    lib.meshopt_simplify.restype = ctypes.c_size_t
    
    # meshopt_simplifyScale signature
    lib.meshopt_simplifyScale.argtypes = [
        ctypes.POINTER(ctypes.c_float),    # vertex_positions
        ctypes.c_size_t,                    # vertex_count
        ctypes.c_size_t,                    # vertex_positions_stride
    ]
    lib.meshopt_simplifyScale.restype = ctypes.c_float


def simplify(
    vertices: np.ndarray,
    indices: np.ndarray,
    target_index_count: int,
    target_error: float = 0.01,
    options: int = 0,
) -> tuple:
    """
    Simplify mesh using meshoptimizer.
    
    Args:
        vertices: (N, 3) float32 array of vertex positions
        indices: (M,) uint32 array of triangle indices
        target_index_count: Target number of indices after simplification
        target_error: Maximum relative error (0.01 = 1%)
        options: meshoptimizer options bitmask (0 = default)
    
    Returns:
        new_indices: Simplified index buffer
        result_error: Actual error achieved
    """
    lib = _load_meshoptimizer_lib()
    
    # Ensure correct types
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.uint32)
    
    # Allocate output buffer (worst case is same size as input)
    output_indices = np.empty_like(indices)
    result_error = ctypes.c_float(0.0)
    
    # Get pointers
    vertices_ptr = vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    output_ptr = output_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    error_ptr = ctypes.byref(result_error)
    
    # Call meshoptimizer
    new_index_count = lib.meshopt_simplify(
        output_ptr,
        indices_ptr,
        len(indices),
        vertices_ptr,
        len(vertices),
        12,  # stride = 3 floats * 4 bytes
        target_index_count,
        target_error,
        options,
        error_ptr,
    )
    
    # Return simplified indices and error
    new_indices = output_indices[:new_index_count]
    return new_indices, float(result_error.value)


def simplify_with_screen_error(
    vertices: np.ndarray,
    indices: np.ndarray,
    target_index_count: int,
    target_error: float = 0.01,
    camera_position: np.ndarray = None,
    camera_direction: np.ndarray = None,
    fov: float = 40.0,
    screen_width: int = 1920,
    screen_height: int = 1080,
    options: int = 0,
) -> tuple:
    """
    Simplify mesh using meshoptimizer with screen-space arc angle error metric.
    
    Args:
        vertices: (N, 3) float32 array of vertex positions
        indices: (M,) uint32 array of triangle indices
        target_index_count: Target number of indices after simplification
        target_error: Maximum relative error (0.01 = 1%)
        camera_position: (3,) camera position in world space. If None, uses origin.
        camera_direction: (3,) camera forward direction. If None, uses [0, 0, -1].
        fov: Field of view in degrees
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        options: meshoptimizer options bitmask (0 = default)
    
    Returns:
        new_indices: Simplified index buffer
        result_error: Actual error achieved
    """
    # Calculate screen-space error scale
    # Convert geometric error to screen-space arc angle error
    if camera_position is not None:
        # Calculate distance from camera to mesh center
        mesh_center = vertices.mean(axis=0)
        camera_pos = np.array(camera_position, dtype=np.float32)
        distance = np.linalg.norm(mesh_center - camera_pos)
        
        # Calculate pixel size at mesh center
        # Screen arc angle = geometric_error / distance * (screen_size / tan(fov/2))
        fov_rad = np.deg2rad(fov)
        screen_scale = min(screen_width, screen_height) / (2.0 * np.tan(fov_rad / 2.0))
        
        # Adjust target_error based on screen-space projection
        # Smaller distance = smaller geometric error for same screen error
        if distance > 0:
            screen_error_scale = distance / screen_scale
            target_error = target_error * screen_error_scale
    
    # Call base simplify function
    return simplify(vertices, indices, target_index_count, target_error, options)

