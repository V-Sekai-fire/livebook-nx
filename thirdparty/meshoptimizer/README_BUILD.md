# meshoptimizer Python Package

This directory contains meshoptimizer, a library for mesh optimization and simplification, packaged as a Python module.

## Installation

The package is automatically built and installed when you run `uv_init` in `omnipart_generation.exs`. The build process:

1. Automatically builds the C++ shared library using CMake during package installation
2. Copies the library to the package directory
3. Makes it available for import as `meshoptimizer`

## Using in Python

The package can be imported directly:

```python
from meshoptimizer import simplify_with_screen_error
```

The Python wrapper (`thirdparty/OmniPart/modules/part_synthesis/utils/meshoptimizer_wrapper.py`) will automatically:
1. Try to import from the installed `meshoptimizer` package
2. Fall back to local path if not installed
3. Fall back to trimesh quadric decimation if meshoptimizer is not available

## Features

The integration uses meshoptimizer's error-aware simplification with screen-space arc angle metrics:
- Calculates screen-space error based on camera position and field of view
- Adjusts geometric error tolerance based on viewing distance
- Preserves mesh topology and appearance
- Falls back gracefully if library is not available

## Build Requirements

When installing via `uv_init`, the following are required:
- CMake 3.10 or later (automatically installed as a build dependency)
- C++11 compatible compiler
- Python ctypes (included in standard library)

The build happens automatically during `pip install` or `uv pip install`.

