#!/usr/bin/env elixir

# Tris-to-Quads Converter Script
# Converts triangles to quads in FBX files using Optimized-Tris-to-Quads-Converter
# Based on: https://github.com/Rulesobeyer/Tris-Quads-Ex
#
# Usage:
#   elixir tris_to_quads_converter.exs <input_fbx> [options]
#
# Options:
#   --output, -o <path>        Output FBX file path (default: input file with _quads suffix)
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
  "bpy==4.*",
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
        elixir tris_to_quads_converter.exs <input_fbx> [options]

      Options:
        --output, -o <path>        Output FBX file path (default: input file with _quads suffix)
        --help, -h                 Show this help message

      Example:
        elixir tris_to_quads_converter.exs model.fbx -o model_quads.fbx
      """)
      System.halt(0)
    end

    input_path = List.first(args)

    if !input_path do
      IO.puts("""
      Error: Input FBX file path is required.

      Usage:
        elixir tris_to_quads_converter.exs <input_fbx> [options]

      Use --help for more information.
      """)
      System.halt(1)
    end

    # Generate output path if not provided
    output_path = Keyword.get(opts, :output)
    output_path = if output_path && output_path != "" do
      output_path
    else
      base = Path.rootname(input_path)
      ext = Path.extname(input_path)
      "#{base}_quads#{ext}"
    end

    # Check if input file exists
    if !File.exists?(input_path) do
      IO.puts("Error: Input file not found: #{input_path}")
      System.halt(1)
    end

    %{
      input_path: input_path,
      output_path: output_path
    }
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== Tris-to-Quads Converter ===
Input: #{config.input_path}
Output: #{config.output_path}
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
output_path = config['output_path']

print(f"\\n=== Loading FBX: {input_path} ===")

# Clear Blender scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import FBX
try:
    bpy.ops.import_scene.fbx(filepath=input_path)
    print("✓ FBX loaded successfully")
except Exception as e:
    print(f"✗ Error loading FBX: {e}")
    import traceback
    traceback.print_exc()
    raise

# Find mesh objects
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if len(mesh_objects) == 0:
    print("✗ Error: No mesh objects found in FBX file")
    raise ValueError("No mesh objects found")

print(f"Found {len(mesh_objects)} mesh object(s)")

# Convert tris to quads for each mesh
def is_valid_edge_for_quad_conversion(edge):
    '''Check if an edge is valid for tris-to-quads conversion'''
    return (len(edge.link_faces) == 2 and
            len(edge.link_faces[0].edges) == 3 and
            len(edge.link_faces[1].edges) == 3)

def convert_tris_to_quads(obj):
    '''Convert triangles to quads using Optimized-Tris-to-Quads-Converter algorithm'''
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
        return

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
        return

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

# Process each mesh object
total_converted = 0
for obj in mesh_objects:
    try:
        convert_tris_to_quads(obj)
        total_converted += 1
    except Exception as e:
        print(f"✗ Error processing {obj.name}: {e}")
        import traceback
        traceback.print_exc()

if total_converted == 0:
    print("\\n✗ Error: No meshes were processed")
    raise ValueError("No meshes processed")

print(f"\\n=== Exporting FBX: {output_path} ===")

# Ensure output directory exists
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Export FBX
try:
    bpy.ops.export_scene.fbx(filepath=output_path, check_existing=False, add_leaf_bones=False)
    print(f"✓ FBX exported successfully: {output_path}")
except Exception as e:
    print(f"✗ Error exporting FBX: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\\n=== Complete ===")
print(f"Converted {total_converted} mesh object(s) and saved to: {output_path}")
""", %{})

IO.puts("\n=== Complete ===")
IO.puts("Tris-to-quads conversion completed successfully!")
