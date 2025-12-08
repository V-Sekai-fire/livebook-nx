#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Tris-to-Quads Converter Script
# Converts triangles to quads in 3D model files using Optimized-Tris-to-Quads-Converter
# Based on: https://github.com/Rulesobeyer/Tris-Quads-Ex
#
# Usage:
#   elixir tris_to_quads_converter.exs <input_file> [options]
#
# Supported Input Formats: GLB, GLTF, USD (usd, usda, usdc), FBX
# Supported Output Format: USDC (binary only)
#
# Note: Output is always binary USDC format to preserve quads and materials
#       with optimal performance and file size. Embedded images are supported.
#
# Options:
#   --output, -o <path>        Output USDC file path (default: input file with _quads.usdc suffix)
#   --help, -h                 Show this help message

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"}
])

# Initialize Python environment with required dependencies
Pythonx.uv_init("""
[project]
name = "tris-to-quads-converter"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "bpy==4.5.*",
  "pulp",
]
""")

# Parse command-line arguments
defmodule ArgsParser do
  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        output: :string,
        help: :boolean
      ],
      aliases: [
        o: :output,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      IO.puts("""
      Tris-to-Quads Converter

      Usage:
        elixir tris_to_quads_converter.exs <input_file> [options]

      Supported Input Formats: GLB, GLTF, USD (usd, usda, usdc), FBX
      Supported Output Format: USDC (binary only)

      Note: Output is always binary USDC format to preserve quads and materials
            with optimal performance and file size. Embedded images are supported.
            FBX export is not supported (materials are lost).

      Options:
        --output, -o <path>        Output USDC file path (default: input file with _quads.usdc suffix)
        --help, -h                 Show this help message

      Example:
        elixir tris_to_quads_converter.exs model.glb -o model_quads.usdc
        elixir tris_to_quads_converter.exs model.fbx
        elixir tris_to_quads_converter.exs model.usda
        elixir tris_to_quads_converter.exs model.usd
      """)
      System.halt(0)
    end

    input_path = List.first(args)

    if !input_path do
      IO.puts("""
      Error: Input file path is required.

      Usage:
        elixir tris_to_quads_converter.exs <input_file> [options]

      Supported formats: GLB, GLTF, USD, FBX
      Use --help for more information.
      """)
      System.halt(1)
    end

    # Validate input format
    input_ext = String.downcase(Path.extname(input_path))
    valid_input_exts = [".glb", ".gltf", ".usd", ".usda", ".usdc", ".fbx"]
    if !Enum.member?(valid_input_exts, input_ext) do
      IO.puts("Error: Unsupported input format: #{input_ext}")
      IO.puts("Supported formats: #{Enum.join(valid_input_exts, ", ")}")
      System.halt(1)
    end

    # Generate output path if not provided
    output_path = Keyword.get(opts, :output)
    use_timestamped_folder = !output_path || output_path == ""

    output_path = if output_path && output_path != "" do
      # Validate output format - only USDC allowed
      output_ext = String.downcase(Path.extname(output_path))
      if output_ext != ".usdc" do
        IO.puts("Error: Only USDC output format is supported: #{output_ext}")
        IO.puts("Please use .usdc extension for output file")
        System.halt(1)
      end
      output_path
    else
      # Will be generated in Python with timestamped folder
      nil
    end

    # Check if input file exists
    if !File.exists?(input_path) do
      IO.puts("Error: Input file not found: #{input_path}")
      System.halt(1)
    end

    %{
      input_path: input_path,
      output_path: output_path,
      use_timestamped_folder: use_timestamped_folder
    }
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

output_display = if config.output_path, do: config.output_path, else: "output/<timestamp>/<filename>_quads.usdc"
IO.puts("""
=== Tris-to-Quads Converter ===
Input: #{config.input_path}
Output: #{output_display}
""")

# Convert JSON config to string for Python
config_json = Jason.encode!(config)
File.write!("config.json", config_json)

# Run conversion
{_, _python_globals} = Pythonx.eval("""
import json
import sys
import os
from pathlib import Path
import bpy
import bmesh
from pulp import PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum, value

# Get configuration from JSON file
with open("config.json", 'r') as f:
    config = json.load(f)

input_path = config['input_path']
output_path = config.get('output_path')
use_timestamped_folder = config.get('use_timestamped_folder', True)

# Create output directory structure
output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)

# Generate output path with timestamped folder if needed
if use_timestamped_folder:
    import time
    tag = time.strftime("%Y%m%d_%H_%M_%S")
    export_dir = output_dir / tag
    export_dir.mkdir(exist_ok=True, parents=True)

    # Generate output filename from input
    input_stem = Path(input_path).stem
    output_filename = f"{input_stem}_quads.usdc"
    output_path = str(export_dir / output_filename)
    print(f"Using timestamped output folder: {export_dir.name}/")
else:
    # Use provided output path, ensure directory exists
    export_dir = Path(output_path).parent
    export_dir.mkdir(exist_ok=True, parents=True)

# Detect input format
input_ext = Path(input_path).suffix.lower()
print(f"\\n=== Loading {input_ext.upper()}: {input_path} ===")

# Clear Blender scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import based on file format
try:
    if input_ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(
            filepath=input_path,
            import_pack_images=True,  # Preserve embedded textures
            import_shading='NORMALS'  # Preserve material shading
        )
        print("✓ GLB/GLTF loaded successfully")
    elif input_ext in ['.usd', '.usda', '.usdc']:
        # USD import - using only essential parameters that work in Blender 4.5
        bpy.ops.wm.usd_import(
            filepath=input_path,
            import_materials=True
        )
        print("✓ USD loaded successfully")
    elif input_ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=input_path)
        print("✓ FBX loaded successfully")
    else:
        raise ValueError(f"Unsupported input format: {input_ext}")
except Exception as e:
    print(f"✗ Error loading file: {e}")
    import traceback
    traceback.print_exc()
    raise

# Find mesh objects
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if len(mesh_objects) == 0:
    print("✗ Error: No mesh objects found in file")
    raise ValueError("No mesh objects found")

print(f"Found {len(mesh_objects)} mesh object(s)")

# Convert tris to quads for each mesh
def is_valid_edge_for_quad_conversion(edge):
    '''Check if an edge is valid for tris-to-quads conversion'''
    return (len(edge.link_faces) == 2 and
            len(edge.link_faces[0].edges) == 3 and
            len(edge.link_faces[1].edges) == 3)

def convert_tris_to_quads(obj):
    '''Convert triangles to quads using Optimized-Tris-to-Quads-Converter algorithm
    Returns: number of edges converted (0 if none)'''
    print(f"\\nProcessing mesh: {obj.name}")

    # Ensure we're in object mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')

    # Get bmesh
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Select all faces
    for face in bm.faces:
        face.select_set(True)
    for edge in bm.edges:
        edge.select_set(True)

    # Create optimization problem
    m = LpProblem(sense=LpMaximize)
    edges = {}

    # Find valid edges (edges shared by two triangular faces)
    for edge in bm.edges:
        if not is_valid_edge_for_quad_conversion(edge):
            continue
        ln = edge.calc_length()
        edges[edge] = LpVariable(f"v{len(edges):03}", cat="Binary"), ln

    if len(edges) == 0:
        print(f"  No valid edges found for conversion in {obj.name}")
        bm.free()
        bpy.ops.object.mode_set(mode='OBJECT')
        return 0

    print(f"  Found {len(edges)} candidate edges for conversion")

    # Set objective: maximize edge length with preference for longer edges
    mx = max([i[1] for i in edges.values()], default=1)
    m.setObjective(lpSum(v * (1 + 0.1 * ln / mx) for edge, (v, ln) in edges.items()))

    # Add constraints: each triangular face can have at most one edge dissolved
    for face in bm.faces:
        if len(face.edges) != 3:
            continue
        vv = [vln[0] for edge in face.edges if (vln := edges.get(edge)) is not None]
        if len(vv) > 1:
            m += lpSum(vv) <= 1

    # Solve the problem
    print("  Solving optimization problem...")
    solver = PULP_CBC_CMD(gapRel=0.01, timeLimit=60, msg=False)
    m.solve(solver)

    if m.status != 1:
        print(f"  Warning: Optimization did not solve for {obj.name}")
        bm.free()
        bpy.ops.object.mode_set(mode='OBJECT')
        return 0

    # Apply the solution: dissolve selected edges
    bpy.ops.mesh.select_all(action="DESELECT")
    n = 0
    for edge, (v, _) in edges.items():
        if value(v) > 0.5:
            edge.select_set(True)
            n += 1

    if n > 0:
        bpy.ops.mesh.dissolve_edges(use_verts=False)
        print(f"  ✓ Converted {n} edge pairs to quads")
    else:
        print(f"  No edges selected for conversion")

    bm.free()
    bpy.ops.object.mode_set(mode='OBJECT')
    return n

# Process each mesh object and track if quads were converted
total_converted = 0
total_edges_converted = 0
quads_were_converted = False

for obj in mesh_objects:
    try:
        edges_converted = convert_tris_to_quads(obj)
        if edges_converted > 0:
            quads_were_converted = True
            total_edges_converted += edges_converted
        total_converted += 1
    except Exception as e:
        print(f"✗ Error processing {obj.name}: {e}")
        import traceback
        traceback.print_exc()

if total_converted == 0:
    print("\\n✗ Error: No meshes were processed")
    raise ValueError("No meshes processed")

# Force USDC output format (only format supported)
output_ext = Path(output_path).suffix.lower()
if output_ext != '.usdc':
    # Change output to binary USDC format
    base = Path(output_path).stem
    output_path = str(Path(output_path).parent / f"{base}.usdc")
    if quads_were_converted:
        print(f"\\n⚠ Quads were converted - switching output to binary USDC format: {output_path}")
    else:
        print(f"\\n⚠ Output format changed to binary USDC: {output_path}")

# Validate output format - only USDC allowed
output_ext = Path(output_path).suffix.lower()
if output_ext != '.usdc':
    print(f"\\n✗ Error: Only USDC output format is supported, got: {output_ext}")
    print("  Please use .usdc extension for output file")
    raise ValueError(f"Only USDC output format supported, got: {output_ext}")

print(f"\\n=== Exporting USDC: {output_path} ===")

# Ensure output directory exists
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Export as binary USDC
try:
    # Binary USDC export with embedded images and material preservation
    # Using only essential parameters that are valid in Blender 4.5
    # Note: Texture embedding is controlled by relative_paths=False
    bpy.ops.wm.usd_export(
        filepath=output_path,
        export_materials=True,
        export_textures=True,
        relative_paths=False,  # False = embed textures, True = use relative paths
        export_uvmaps=True,
        export_armatures=False,
        selected_objects_only=False,
        visible_objects_only=False,
        use_instancing=False,
        evaluation_mode='RENDER'
    )
    print(f"✓ Binary USDC exported successfully: {output_path}")
except Exception as e:
    print(f"✗ Error exporting: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\\n=== Complete ===")
if quads_were_converted:
    print(f"Converted {total_edges_converted} edge pairs to quads in {total_converted} mesh object(s)")
    print(f"Saved to: {output_path} (binary USDC format preserves quads and materials)")
else:
    print(f"Processed {total_converted} mesh object(s) (no quads converted)")
    print(f"Saved to: {output_path} (binary USDC format)")

if use_timestamped_folder:
    export_dir_name = Path(output_path).parent.name
    print(f"\\nOutput files in {export_dir_name}/:")
    print(f"  - {Path(output_path).name} (Converted model)")
""", %{})

IO.puts("\n=== Complete ===")
IO.puts("Tris-to-quads conversion completed successfully!")
