# Mesh Decimator (meshopt)

## Overview

**Mesh Decimator (meshopt)** is a Blender add-on that reduces the number of triangles in a mesh using the [meshoptimizer](https://github.com/zeux/meshoptimizer) library. This add-on provides high-quality mesh simplification while preserving mesh appearance and topology as much as possible.

## Prerequisite

To use this add-on, you need to have the meshoptimizer library built as a shared library (`.dll` on Windows, `.so` on Linux, `.dylib` on macOS). The add-on will attempt to load the library from:
1. The add-on directory
2. A `lib` subdirectory in the add-on directory
3. System library paths

## Features

- **High-quality mesh decimation** using the meshoptimizer library
- **Error-based simplification** (like Godot Engine) - error threshold based on scene area
- **Border locking** option to preserve mesh boundaries
- **Simple interface** integrated into Blender's mesh editing workflow
- **Batch processing** of multiple selected objects

## Installation

1. Build the meshoptimizer library as a shared library (see Building section below)
2. Place the shared library in the add-on directory or system library path
3. Download the ZIP file containing the `__init__.py` script
4. Open Blender and go to `Edit > Preferences > Add-ons`
5. Click `Install` and select the downloaded ZIP file
6. Enable the add-on from the list

## Building meshoptimizer

### Quick Build (Recommended)

Use the provided build scripts:

- **Windows**: Run `build_windows.bat`
- **Linux**: Run `./build_linux.sh`
- **macOS**: Run `./build_macos.sh`

The scripts will automatically build the library and copy it to the add-on directory.

### Manual Build

See [BUILD.md](BUILD.md) for detailed manual build instructions.

**Note**: The CMake option is `MESHOPT_BUILD_SHARED_LIBS=ON` (not `BUILD_SHARED_LIBS`).

## Usage

### Decimating Meshes

1. Select one or more mesh objects in Object Mode
2. Go to `Edit > Mesh > Decimate Mesh (meshopt)` in the 3D Viewport menu
3. Configure the decimation settings:
   - **Target Error**: Maximum allowed error relative to mesh bounding box size (scene area)
   - **Lock Border**: Preserve vertices on mesh borders
4. Click `OK` to apply decimation

### Settings

- **Target Error**: Maximum geometric error allowed, relative to the mesh bounding box size (scene area). For example:
  - `0.01` = 1% of the mesh's bounding box diagonal
  - `0.001` = 0.1% (higher quality, more triangles)
  - `0.1` = 10% (lower quality, fewer triangles)
  
  This works like Godot Engine's mesh simplification - the error is calculated based on the scene area (mesh bounding box), ensuring consistent quality regardless of mesh size.

- **Lock Border**: When enabled, vertices on mesh borders (edges with only one adjacent face) will not be moved during simplification.

## Why Use This Add-on?

- **High Quality**: meshoptimizer uses advanced algorithms to preserve mesh appearance during simplification
- **Fast**: Optimized C++ implementation provides excellent performance
- **Error-Based**: Like Godot Engine, uses error threshold based on scene area for consistent quality
- **Topology Preservation**: Maintains mesh topology when possible
- **Professional Tool**: Used in production pipelines for game development and 3D content creation

## Credits

- **meshoptimizer library**: [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer) by Arseny Kapoulkine
- **License**: MIT (see LICENSE file)
