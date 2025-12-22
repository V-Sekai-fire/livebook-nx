#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# VRM Bone Renaming Script
# Renames bones in 3D models to VRM specification using Qwen3VL vision analysis
# Uses Blender screenshots with annotated bone labels and foci markers
#
# Usage:
#   elixir vrm_bone_renamer.exs <input_file> [options]
#
# Supported Input Formats: GLB, GLTF, USD (usd, usda, usdc), FBX
#
# Options:
#   --help, -h                 Show this help message
#
# Example:
#   elixir vrm_bone_renamer.exs model.glb
#   elixir vrm_bone_renamer.exs model.fbx

# Configure OpenTelemetry for console-only logging
Application.put_env(:opentelemetry, :span_processor, :batch)
Application.put_env(:opentelemetry, :traces_exporter, :none)
Application.put_env(:opentelemetry, :metrics_exporter, :none)
Application.put_env(:opentelemetry, :logs_exporter, :none)

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
  {:req, "~> 0.5.0"},
  {:opentelemetry_api, "~> 1.3"},
  {:opentelemetry, "~> 1.3"},
  {:opentelemetry_exporter, "~> 1.0"},
])

Logger.configure(level: :info)

# Load shared utilities
Code.eval_file("shared_utils.exs")

# Initialize OpenTelemetry
OtelSetup.configure()

# Initialize Python environment with required dependencies
# bpy for Blender, Qwen3VL for vision analysis
Pythonx.uv_init("""
[project]
name = "vrm-bone-renamer"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "bpy==4.5.*",
  "transformers",
  "accelerate",
  "pillow",
  "torch>=2.0.0,<2.5.0",
  "torchvision>=0.15.0,<0.20.0",
  "numpy",
  "huggingface-hub",
  "bitsandbytes",
]

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
""")

# Parse command-line arguments
defmodule ArgsParser do
  def show_help do
    IO.puts("""
    VRM Bone Renaming Script
    Renames bones in 3D models to VRM specification using Qwen3VL vision analysis
    Uses Blender screenshots with annotated bone labels and foci markers

    Usage:
      elixir vrm_bone_renamer.exs <input_file> [options]

    Supported Input Formats: GLB, GLTF, USD (usd, usda, usdc), FBX

    Options:
      --help, -h                 Show this help message

    Example:
      elixir vrm_bone_renamer.exs model.glb
      elixir vrm_bone_renamer.exs model.fbx
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        help: :boolean
      ],
      aliases: [
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    input_path = List.first(args)

    if !input_path do
      IO.puts("""
      Error: Input file path is required.

      Usage:
        elixir vrm_bone_renamer.exs <input_file> [options]

      Supported formats: GLB, GLTF, USD, FBX
      Use --help or -h for more information.
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

    # Check if input file exists
    if !File.exists?(input_path) do
      IO.puts("Error: Input file not found: #{input_path}")
      System.halt(1)
    end

    %{
      input_path: input_path
    }
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

# Convert input path to absolute path to avoid path resolution issues
# Normalize to forward slashes for cross-platform compatibility with Blender
absolute_input_path = Path.expand(config.input_path) |> String.replace("\\", "/")

# Create timestamped output directory
output_dir = OutputDir.create()
IO.puts("Output directory: #{output_dir}")

# Generate output path (replace extension with .usdc) in output directory
input_basename = Path.basename(absolute_input_path, Path.extname(absolute_input_path))
output_path = Path.join(output_dir, "#{input_basename}_vrm.usdc") |> String.replace("\\", "/")

config = Map.put(config, :input_path, absolute_input_path)
config = Map.put(config, :output_path, output_path)
config = Map.put(config, :output_dir, output_dir)

IO.puts("""
=== VRM Bone Renaming ===
Input: #{config.input_path}
""")

# Convert JSON config to string for Python (use temp file to avoid conflicts)
config_json = Jason.encode!(config)
# Use cross-platform temp directory
tmp_dir = System.tmp_dir!()
File.mkdir_p!(tmp_dir)
config_file = Path.join(tmp_dir, "vrm_bone_renamer_config_#{System.system_time(:millisecond)}.json")
File.write!(config_file, config_json)
config_file_normalized = String.replace(config_file, "\\", "/")

# Run bone renaming with proper cleanup
config_file_for_python = String.replace(config_file_normalized, "\\", "\\\\")
SpanCollector.track_span("vrm_bone_renamer.process", fn ->
try do
  {_, _python_globals} = Pythonx.eval("""
import json
import sys
import os
import tempfile
from pathlib import Path
import bpy
import bmesh
from mathutils import Vector
import bpy_extras.object_utils

# Get configuration from JSON file
config_file_path = r"#{config_file_for_python}"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Normalize paths to use forward slashes for cross-platform compatibility
input_path = str(Path(config['input_path'])).replace("\\\\", "/")

# Detect input format
input_ext = Path(input_path).suffix.lower()
print("")
print(f"=== Loading {input_ext.upper()}: {input_path} ===")

# Clear Blender scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import based on file format
try:
    if input_ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(
            filepath=input_path,
            import_pack_images=True,
            import_shading='NORMALS'
        )
        print("[OK] GLB/GLTF loaded successfully")
    elif input_ext in ['.usd', '.usda', '.usdc']:
        bpy.ops.wm.usd_import(
            filepath=input_path,
            import_materials=True
        )
        print("[OK] USD loaded successfully")
    elif input_ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=input_path)
        print("[OK] FBX loaded successfully")
    else:
        raise ValueError(f"Unsupported input format: {input_ext}")
except Exception as e:
    print(f"[ERROR] Error loading file: {e}")
    import traceback
    traceback.print_exc()
    raise

# Find armature objects
armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
if len(armatures) == 0:
    print("[ERROR] No armature objects found in file")
    raise ValueError("No armature objects found")

print(f"Found {len(armatures)} armature object(s)")

# Use the first armature (or handle multiple if needed)
armature = armatures[0]
print(f"Using armature: {armature.name}")

# Extract bone names and positions
print("")
print("=== Extracting Bone Information ===")
bone_data = []
for bone in armature.data.bones:
    bone_data.append({
        "name": bone.name,
        "head": list(bone.head_local),
        "tail": list(bone.tail_local),
        "parent": bone.parent.name if bone.parent else None
    })
    print(f"  Bone: {bone.name} (parent: {bone.parent.name if bone.parent else 'None'})")

print(f"[OK] Extracted {len(bone_data)} bones")

# Export as GLTF to extract hierarchy (works for all input formats)
print("")
print("=== Exporting to GLTF for Hierarchy Extraction ===")
gltf_json_hierarchy = None
try:
    # Export to temporary GLTF file
    temp_gltf_path = output_dir / "temp_hierarchy.gltf"
    temp_gltf_path_str = str(temp_gltf_path).replace("\\\\", "/")

    # Select all objects for export
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in ['MESH', 'ARMATURE']:
            obj.select_set(True)

    # Export as GLTF
    bpy.ops.export_scene.gltf(
        filepath=temp_gltf_path_str,
        export_format='GLTF_SEPARATE',
        export_selected=False,
        export_materials='EXPORT',
        export_colors=True,
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_skins=True,
        export_all_influences=False,
        export_morph=False,
        export_yup=True
    )
    print(f"[OK] Exported to temporary GLTF: {temp_gltf_path}")

    # Extract GLTF JSON hierarchy
    print("=== Extracting GLTF JSON Hierarchy ===")
    import struct

    # Read GLTF JSON file
    with open(temp_gltf_path, 'r', encoding='utf-8') as f:
        gltf_json = json.load(f)

    # Extract relevant hierarchy information (nodes, skins, scenes)
    # Remove binary data references to keep it lightweight
    gltf_hierarchy = {
        "nodes": [],
        "scenes": gltf_json.get("scenes", []),
        "skins": []
    }

    # Extract node hierarchy with indices
    if "nodes" in gltf_json:
        for idx, node in enumerate(gltf_json["nodes"]):
            node_info = {
                "index": idx,
                "name": node.get("name", f"node_{idx}"),
                "children": node.get("children", []),
                "skin": node.get("skin")
            }
            # Only include transform if present (not binary data)
            if "matrix" in node:
                node_info["matrix"] = node["matrix"]
            if "translation" in node:
                node_info["translation"] = node["translation"]
            if "rotation" in node:
                node_info["rotation"] = node["rotation"]
            if "scale" in node:
                node_info["scale"] = node["scale"]
            gltf_hierarchy["nodes"].append(node_info)

    # Extract skin information (skeleton references)
    if "skins" in gltf_json:
        for idx, skin in enumerate(gltf_json["skins"]):
            skin_info = {
                "index": idx,
                "joints": skin.get("joints", []),
                "skeleton": skin.get("skeleton")
            }
            gltf_hierarchy["skins"].append(skin_info)

    gltf_json_hierarchy = json.dumps(gltf_hierarchy, indent=2, ensure_ascii=False)
    print(f"[OK] Extracted GLTF hierarchy: {len(gltf_hierarchy['nodes'])} nodes, {len(gltf_hierarchy['skins'])} skins")

    # Save GLTF hierarchy to output directory
    gltf_hierarchy_file = output_dir / "gltf_hierarchy.json"
    with open(gltf_hierarchy_file, 'w', encoding='utf-8') as f:
        f.write(gltf_json_hierarchy)
    print(f"  Saved GLTF hierarchy to: {gltf_hierarchy_file}")

except Exception as e:
    print(f"[WARN] Could not extract GLTF hierarchy: {e}")
    import traceback
    traceback.print_exc()
    gltf_json_hierarchy = None

# Create annotated visualizations
print("")
print("=== Creating Annotated Visualizations ===")

# Set up viewport to show armature
bpy.context.view_layer.objects.active = armature

# Hide mesh objects to show only armature
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.hide_set(True)

# Set up armature display
armature.data.display_type = 'STICK'
armature.show_in_front = True

# Create output directory structure for intermediate files
output_dir = Path(config.get('output_dir', 'output'))
output_dir.mkdir(exist_ok=True, parents=True)

# Create subdirectories for intermediate files
normal_maps_dir = output_dir / "normal_maps"
normal_maps_dir.mkdir(exist_ok=True, parents=True)
annotated_images_dir = output_dir / "annotated_images"
annotated_images_dir.mkdir(exist_ok=True, parents=True)

# Capture screenshots from different camera angles (4 views using Fibonacci sphere)
# This maximizes viewing angle coverage using golden angle spiral distribution
# Based on HKU/SAMPart3D camera trajectory methods
# Run analysis 2 times and merge results for better accuracy
import math
num_views = 4
views = [f'VIEW_{i}' for i in range(num_views)]
annotated_images = []
view_bone_data = {}  # Store bone data for each view for re-annotation

# Set up camera and viewport
scene = bpy.context.scene
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

# Use Eevee Next render engine (works in headless mode, Blender 4.5+)
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.image_settings.file_format = 'PNG'

# Get camera (or create one)
camera = None
for obj in bpy.context.scene.objects:
    if obj.type == 'CAMERA':
        camera = obj
        break

if camera is None:
    # Create camera using data API (works in headless mode)
    camera_data = bpy.data.cameras.new(name="Camera")
    camera = bpy.data.objects.new(name="Camera", object_data=camera_data)
    bpy.context.scene.collection.objects.link(camera)

# Set camera as active
bpy.context.scene.camera = camera

# Calculate camera positioning from mesh extrema (armature bone positions)
import mathutils
bbox_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
bbox_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
has_bones = False

# Calculate bounding box from visible armature bone positions
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE' and not obj.hide_get():
        # Use bone head and tail positions for accurate extrema
        for bone in obj.data.bones:
            # Transform bone head and tail to world space
            head_world = obj.matrix_world @ mathutils.Vector(bone.head_local)
            tail_world = obj.matrix_world @ mathutils.Vector(bone.tail_local)

            # Update bounding box extrema
            bbox_min = mathutils.Vector((min(bbox_min.x, head_world.x, tail_world.x),
                                       min(bbox_min.y, head_world.y, tail_world.y),
                                       min(bbox_min.z, head_world.z, tail_world.z)))
            bbox_max = mathutils.Vector((max(bbox_max.x, head_world.x, tail_world.x),
                                       max(bbox_max.y, head_world.y, tail_world.y),
                                       max(bbox_max.z, head_world.z, tail_world.z)))
            has_bones = True

if has_bones:
    center = (bbox_min + bbox_max) / 2
    size = bbox_max - bbox_min
    # Calculate distance based on mesh extrema
    # Use diagonal of bounding box to ensure mesh fits in frame
    diagonal = size.length
    # For perspective camera, distance = diagonal / (2 * tan(FOV/2))
    # Using 50 degree FOV, we want some padding (1.2x factor)
    fov_rad = math.radians(50.0)
    distance = (diagonal * 1.2) / (2 * math.tan(fov_rad / 2))
    # Ensure minimum distance based on largest dimension
    min_distance = max(size.x, size.y, size.z) * 1.5
    distance = max(distance, min_distance)
else:
    center = mathutils.Vector((0, 0, 0))
    size = mathutils.Vector((10, 10, 10))  # Default size
    distance = 5

# Fibonacci sphere (golden angle spiral) for evenly distributed views
# This maximizes viewing angle coverage - commonly used in HKU/SAMPart3D papers
def fibonacci_sphere(n, offset=0.5):
    # Generate n points evenly distributed on a sphere using Fibonacci spiral.
    # Returns list of (x, y, z) unit vectors.
    points = []
    golden_angle = math.pi * (3 - math.sqrt(5))  # Golden angle in radians

    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = golden_angle * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

def look_at(camera, point):
    direction = point - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

# Generate camera positions using Fibonacci sphere
sphere_points = fibonacci_sphere(num_views)

for i, view_name in enumerate(views):
    print(f"  Capturing {view_name} view...")

    # Ensure correct scene state at start of each frame: meshes hidden, armature visible
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.hide_set(True)
    armature.hide_set(False)

    # Force scene update after visibility changes
    bpy.context.view_layer.update()

    # Recalculate bounding box for each frame based on currently visible objects
    bbox_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    has_bones = False

    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE' and not obj.hide_get():
            for bone in obj.data.bones:
                head_world = obj.matrix_world @ mathutils.Vector(bone.head_local)
                tail_world = obj.matrix_world @ mathutils.Vector(bone.tail_local)

                bbox_min = mathutils.Vector((min(bbox_min.x, head_world.x, tail_world.x),
                                           min(bbox_min.y, head_world.y, tail_world.y),
                                           min(bbox_min.z, head_world.z, tail_world.z)))
                bbox_max = mathutils.Vector((max(bbox_max.x, head_world.x, tail_world.x),
                                           max(bbox_max.y, head_world.y, tail_world.y),
                                           max(bbox_max.z, head_world.z, tail_world.z)))
                has_bones = True

    if has_bones:
        frame_center = (bbox_min + bbox_max) / 2
        frame_size = bbox_max - bbox_min
        diagonal = frame_size.length
        fov_rad = math.radians(50.0)
        frame_distance = (diagonal * 1.2) / (2 * math.tan(fov_rad / 2))
        min_distance = max(frame_size.x, frame_size.y, frame_size.z) * 1.5
        frame_distance = max(frame_distance, min_distance)
    else:
        frame_center = center
        frame_distance = distance

    # Get point on unit sphere and scale by distance
    x, y, z = sphere_points[i]
    camera_pos = mathutils.Vector((x * frame_distance, y * frame_distance, z * frame_distance))

    # Position camera
    camera.location = frame_center + camera_pos

    # Make camera look at center
    look_at(camera, frame_center)

    # Use perspective camera
    camera.data.type = 'PERSP'
    camera.data.angle = math.radians(50.0)  # Field of view

    # Step 1: Render normal map (with mesh visible, armature hidden)
    normal_map_path = normal_maps_dir / f"normal_map_{view_name.lower()}.png"
    normal_map_rendered = False
    mesh_objects = []  # Initialize before try block
    try:
        # Show mesh objects for normal map rendering
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                if obj.hide_get():
                    mesh_objects.append(obj)
                    obj.hide_set(False)

        # Hide armature for normal map
        armature.hide_set(True)

        # Use material-based normal visualization (most reliable approach)
        # Create a material that outputs normals as colors for all meshes
        normal_mat = bpy.data.materials.new(name="NormalMapMaterial")
        normal_mat.use_nodes = True
        nodes_mat = normal_mat.node_tree.nodes
        links_mat = normal_mat.node_tree.links

        # Clear default nodes
        for node in nodes_mat:
            nodes_mat.remove(node)

        # Create Geometry node for normal
        geometry = nodes_mat.new(type='ShaderNodeNewGeometry')

        # Convert world-space normal to object-space for consistent visualization
        # Object-space normals are relative to the object's orientation
        vector_transform = nodes_mat.new(type='ShaderNodeVectorTransform')
        vector_transform.convert_from = 'WORLD'
        vector_transform.convert_to = 'OBJECT'

        # Create Vector Math to convert normal to color (normalize from -1,1 to 0,1)
        # Normal map convention: (normal + 1) * 0.5
        vector_add = nodes_mat.new(type='ShaderNodeVectorMath')
        vector_add.operation = 'ADD'
        vector_add.inputs[1].default_value = (1.0, 1.0, 1.0)
        vector_scale = nodes_mat.new(type='ShaderNodeVectorMath')
        vector_scale.operation = 'SCALE'
        vector_scale.inputs[1].default_value = (0.5, 0.5, 0.5)

        # Create Emission shader
        emission = nodes_mat.new(type='ShaderNodeEmission')
        output = nodes_mat.new(type='ShaderNodeOutputMaterial')

        # Link: Geometry Normal (world) -> Transform to Object -> Add 1.0 -> Scale 0.5 -> Emission -> Output
        links_mat.new(geometry.outputs['Normal'], vector_transform.inputs['Vector'])
        links_mat.new(vector_transform.outputs['Vector'], vector_add.inputs[0])
        links_mat.new(vector_add.outputs['Vector'], vector_scale.inputs[0])
        links_mat.new(vector_scale.outputs['Vector'], emission.inputs['Color'])
        links_mat.new(emission.outputs['Emission'], output.inputs['Surface'])

        # Apply material to all mesh objects
        for obj in mesh_objects:
            if len(obj.data.materials) == 0:
                obj.data.materials.append(normal_mat)
            else:
                obj.data.materials[0] = normal_mat

        # Disable compositor for simple rendering
        scene.use_nodes = False

        # Set output path
        scene.render.filepath = str(normal_map_path)

        # Render normal map
        bpy.ops.render.render(write_still=True)
        print(f"    Normal map rendered: {normal_map_path}")
        normal_map_rendered = True

        # Restore scene state: re-hide meshes and re-show armature
        for obj in mesh_objects:
            obj.hide_set(True)
        armature.hide_set(False)

    except Exception as e:
        print(f"    Error rendering normal map: {e}")
        import traceback
        traceback.print_exc()
        # Restore scene state even on error
        try:
            for obj in mesh_objects:
                obj.hide_set(True)
            armature.hide_set(False)
        except:
            pass

    # Step 2: Create base image (normal map or white background)
    screenshot_path = annotated_images_dir / f"bone_view_{view_name.lower()}.png"
    try:
        from PIL import Image

        if normal_map_rendered:
            # Use normal map as background
            base_img = Image.open(normal_map_path).convert("RGBA")
            base_img.save(screenshot_path)
            print(f"    Base image saved: {screenshot_path}")
        else:
            # Create white background if normal map failed
            base_img = Image.new("RGBA", (scene.render.resolution_x, scene.render.resolution_y), (255, 255, 255, 255))
            base_img.save(screenshot_path)
            print(f"    Base image saved (white background) - {screenshot_path}")

    except Exception as e:
        print(f"    Error creating base image: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Step 3: Annotate with bone labels and foci (draw skeleton directly on image)
    try:
        from PIL import Image, ImageDraw, ImageFont

        img_pil = Image.open(screenshot_path).convert("RGBA")
        draw = ImageDraw.Draw(img_pil)

        # Try to load fonts, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_large = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()
                font_large = ImageFont.load_default()

        res_x, res_y = scene.render.resolution_x, scene.render.resolution_y

        # Store bone positions for connecting dots
        bone_positions = {}
        # Track label positions to avoid overlaps
        label_positions = []

        # First pass: collect all bone positions
        for bone in armature.data.bones:
            try:
                # Project 3D bone positions to 2D screen space
                # Convert local to world coordinates
                # bone.head_local and bone.tail_local are in armature local space
                # Transform through armature's world matrix
                head_world = armature.matrix_world @ Vector(bone.head_local)
                tail_world = armature.matrix_world @ Vector(bone.tail_local)

                # Get bone's local transformation matrix for axis calculations
                bone_matrix = armature.matrix_world @ bone.matrix_local

                # Project to camera view
                head_2d = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, head_world)
                tail_2d = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, tail_world)

                head_pix = (int(head_2d.x * res_x), int((1 - head_2d.y) * res_y))
                tail_pix = (int(tail_2d.x * res_x), int((1 - tail_2d.y) * res_y))

                # Store positions
                bone_positions[bone.name] = {
                    'head': head_pix,
                    'tail': tail_pix,
                    'head_world': head_world,
                    'tail_world': tail_world,
                    'matrix': bone_matrix,
                    'bone': bone
                }
            except Exception as e:
                continue

        # Second pass: Draw skeleton connections (parent to child)
        for bone in armature.data.bones:
            if bone.name not in bone_positions:
                continue

            if bone.parent and bone.parent.name in bone_positions:
                try:
                    parent_pos = bone_positions[bone.parent.name]
                    child_pos = bone_positions[bone.name]

                    # Draw line from parent tail to child head (skeleton connection)
                    parent_tail = parent_pos['tail']
                    child_head = child_pos['head']

                    if (0 <= parent_tail[0] < res_x and 0 <= parent_tail[1] < res_y and
                        0 <= child_head[0] < res_x and 0 <= child_head[1] < res_y):
                        # Draw skeleton connection (cyan)
                        draw.line([parent_tail, child_head], fill=(0, 255, 255, 255), width=2)
                except:
                    pass

        # Third pass: Collect bone data and initial label positions for force-directed layout
        bone_data_list = []
        for bone in armature.data.bones:
            if bone.name not in bone_positions:
                continue

            try:
                pos = bone_positions[bone.name]
                head_pix = pos['head']
                tail_pix = pos['tail']
                bone_matrix = pos['matrix']
                head_world = pos['head_world']
                tail_world = pos['tail_world']

                # Only process if within image bounds
                if (0 <= head_pix[0] < res_x and 0 <= head_pix[1] < res_y and
                    0 <= tail_pix[0] < res_x and 0 <= tail_pix[1] < res_y):

                    # Get text size for label
                    try:
                        bbox = draw.textbbox((0, 0), bone.name, font=font_large)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    except:
                        text_width = len(bone.name) * 12
                        text_height = 22

                    # Initial label position (offset from bone head)
                    initial_x = head_pix[0] + 15
                    initial_y = head_pix[1] - 15

                    bone_data_list.append({
                        'bone': bone,
                        'head_pix': head_pix,
                        'tail_pix': tail_pix,
                        'bone_matrix': bone_matrix,
                        'head_world': head_world,
                        'tail_world': tail_world,
                        'label_x': float(initial_x),
                        'label_y': float(initial_y),
                        'text_width': text_width,
                        'text_height': text_height
                    })
            except Exception as e:
                continue

        # Apply force-directed graph layout to labels
        if len(bone_data_list) > 0:
            # Force-directed layout parameters
            repulsion_strength = 5000.0   # Reduced repulsion to prevent pushing to edges
            attraction_strength = 0.3    # Increased attraction to keep labels near bone heads
            damping = 0.85               # Velocity damping
            iterations = 150             # Increased iterations for better convergence
            padding = 5                 # Label padding for overlap detection

            # Helper function to check if two labels overlap
            def labels_overlap(data1, data2):
                x1, y1 = data1['label_x'], data1['label_y']
                w1, h1 = data1['text_width'], data1['text_height']
                x2, y2 = data2['label_x'], data2['label_y']
                w2, h2 = data2['text_width'], data2['text_height']

                # Check bounding box overlap
                return not (x1 + w1 + padding < x2 - padding or
                           x2 + w2 + padding < x1 - padding or
                           y1 + h1 + padding < y2 - padding or
                           y2 + h2 + padding < y1 - padding)

            # Run force-directed layout iterations
            for iteration in range(iterations):
                # Calculate forces for each label
                forces_x = [0.0] * len(bone_data_list)
                forces_y = [0.0] * len(bone_data_list)

                for i, bone_data in enumerate(bone_data_list):
                    label_x = bone_data['label_x']
                    label_y = bone_data['label_y']
                    head_pix = bone_data['head_pix']

                    # Attraction force to bone head (spring-like)
                    dx_attract = head_pix[0] - label_x
                    dy_attract = head_pix[1] - label_y
                    dist_attract = math.sqrt(dx_attract*dx_attract + dy_attract*dy_attract)
                    if dist_attract > 0.1:
                        forces_x[i] += attraction_strength * dx_attract
                        forces_y[i] += attraction_strength * dy_attract

                    # Repulsion forces from other labels
                    for j, other_data in enumerate(bone_data_list):
                        if i == j:
                            continue
                        other_x = other_data['label_x']
                        other_y = other_data['label_y']

                        dx = label_x - other_x
                        dy = label_y - other_y
                        dist = math.sqrt(dx*dx + dy*dy)

                        # Check for overlap using bounding boxes
                        is_overlapping = labels_overlap(bone_data, other_data)

                        if is_overlapping:
                            if dist > 0.1:
                                # Strong repulsion when overlapping
                                force_magnitude = (repulsion_strength * 3.0) / (dist * dist + 1.0)
                                forces_x[i] += force_magnitude * (dx / dist)
                                forces_y[i] += force_magnitude * (dy / dist)
                            else:
                                # Very close - push apart in a random direction to avoid division by zero
                                angle = (i * 2.0 * math.pi) / len(bone_data_list)
                                forces_x[i] += repulsion_strength * math.cos(angle)
                                forces_y[i] += repulsion_strength * math.sin(angle)

                # Add edge repulsion forces to keep labels away from screen edges
                edge_padding = 100  # Larger zone to keep labels away from edges
                edge_repulsion = 2000.0  # Much stronger repulsion from edges
                for i, bone_data in enumerate(bone_data_list):
                    label_x = bone_data['label_x']
                    label_y = bone_data['label_y']
                    text_width = bone_data['text_width']
                    text_height = bone_data['text_height']

                    # Repulsion from left edge (starts at edge_padding distance)
                    if label_x < edge_padding:
                        distance_from_edge = edge_padding - label_x
                        forces_x[i] += edge_repulsion * (distance_from_edge / edge_padding) ** 2
                    # Repulsion from right edge
                    if label_x + text_width > res_x - edge_padding:
                        distance_from_edge = (label_x + text_width) - (res_x - edge_padding)
                        forces_x[i] -= edge_repulsion * (distance_from_edge / edge_padding) ** 2
                    # Repulsion from top edge
                    if label_y < edge_padding:
                        distance_from_edge = edge_padding - label_y
                        forces_y[i] += edge_repulsion * (distance_from_edge / edge_padding) ** 2
                    # Repulsion from bottom edge
                    if label_y + text_height > res_y - edge_padding:
                        distance_from_edge = (label_y + text_height) - (res_y - edge_padding)
                        forces_y[i] -= edge_repulsion * (distance_from_edge / edge_padding) ** 2

                # Update positions with damping
                for i, bone_data in enumerate(bone_data_list):
                    bone_data['label_x'] += forces_x[i] * damping
                    bone_data['label_y'] += forces_y[i] * damping

                    # Boundary constraints (keep labels within image bounds with padding)
                    # Use larger padding to keep labels away from edges
                    boundary_padding = 50
                    text_width = bone_data['text_width']
                    text_height = bone_data['text_height']
                    bone_data['label_x'] = max(boundary_padding, min(res_x - text_width - boundary_padding, bone_data['label_x']))
                    bone_data['label_y'] = max(boundary_padding, min(res_y - text_height - boundary_padding, bone_data['label_y']))

        # Fourth pass: Draw bones, labels, and connecting lines with force-directed positions
        for bone_data in bone_data_list:
            try:
                bone = bone_data['bone']
                head_pix = bone_data['head_pix']
                tail_pix = bone_data['tail_pix']
                label_x = int(bone_data['label_x'])
                label_y = int(bone_data['label_y'])
                text_width = bone_data['text_width']
                text_height = bone_data['text_height']

                # Draw bone line (red) - from head to tail
                draw.line([head_pix, tail_pix], fill=(255, 0, 0, 255), width=3)

                # Draw numbered marker circle (foci) at bone head (green with yellow outline)
                marker_radius = 10
                draw.ellipse([head_pix[0]-marker_radius, head_pix[1]-marker_radius,
                             head_pix[0]+marker_radius, head_pix[1]+marker_radius],
                            fill=(0, 255, 0, 220), outline=(255, 255, 0, 255), width=3)

                # Draw semi-transparent background for label
                padding = 5
                draw.rectangle([label_x - padding, label_y - padding,
                               label_x + text_width + padding, label_y + text_height + padding],
                              fill=(0, 0, 0, 200), outline=(255, 255, 0, 255), width=2)

                # Draw connecting line from label to bone head (foci)
                # Use a dashed or solid line to clearly connect label to marker
                draw.line([head_pix, (label_x, label_y + text_height // 2)],
                         fill=(255, 255, 255, 180), width=2)

                # Add text label (white text on dark background for maximum readability)
                draw.text((label_x, label_y), bone.name,
                         fill=(255, 255, 255, 255), font=font_large)
            except Exception as e:
                # Skip bones that can't be projected
                continue

        # Save initial annotated image with iteration 0 suffix
        iter0_path = annotated_images_dir / f"bone_view_{view_name.lower()}_iter0.png"
        img_pil.save(iter0_path)
        # Also save without suffix for use in Qwen3VL
        img_pil.save(screenshot_path)
        annotated_images.append(str(screenshot_path))
        # Store bone data for this view for potential re-annotation
        view_bone_data[view_name] = bone_data_list
        print(f"    Annotated image saved: {screenshot_path} (also saved as {iter0_path.name})")
    except Exception as e:
        print(f"    Error annotating image: {e}")
        import traceback
        traceback.print_exc()
        # Continue with unannotated image
        annotated_images.append(str(screenshot_path))

if len(annotated_images) == 0:
    raise ValueError("Failed to create any annotated images")

print(f"Created {len(annotated_images)} annotated images")

# Qwen3VL Vision Analysis
print("=== Qwen3VL Vision Analysis ===")

# Load Qwen3VL model (reuse logic from qwen3vl_inference.exs)
print("Loading Qwen3VL model...")
try:
    # Redirect stderr to avoid Pythonx conflicts with logging
    import sys
    import os
    import logging
    import warnings

    # Save original stderr
    original_stderr = sys.stderr

    # Redirect stderr to devnull to avoid Pythonx write conflicts
    try:
        sys.stderr = open(os.devnull, 'w')
    except:
        # Fallback: redirect to a temp file
        import tempfile
        sys.stderr = open(tempfile.NamedTemporaryFile(delete=False).name, 'w')

    # Suppress logging
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    logging.getLogger("tqdm").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    import torch

    four_str = "4"
    MODEL_ID = f"huihui-ai/Huihui-Qwen3-VL-{four_str}B-Instruct-abliterated"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if device == "cuda" else None

    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }

    if quantization_config:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["dtype"] = dtype

    # Try loading from local cache first
    four_str = "4"
    model_weights_dir = Path("pretrained_weights") / f"Huihui-Qwen3-VL-{four_str}B-Instruct-abliterated"
    if model_weights_dir.exists() and (model_weights_dir / "config.json").exists():
        print(f"Loading from local directory: {model_weights_dir}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_weights_dir),
            **load_kwargs
        )
    else:
        print(f"Loading from Hugging Face Hub: {MODEL_ID}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            **load_kwargs
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Restore stderr after model loading
    sys.stderr.close()
    sys.stderr = original_stderr

    print("[OK] Qwen3VL model loaded")

except Exception as e:
    # Restore stderr in case of error
    try:
        if 'original_stderr' in locals():
            try:
                sys.stderr.close()
            except:
                pass
            sys.stderr = original_stderr
    except:
        pass
    print(f"[ERROR] Error loading Qwen3VL model: {e}")
    import traceback
    traceback.print_exc()
    raise

# Create VRM specification prompt
# Build prompt with GLTF hierarchy if available
gltf_hierarchy_section = ""
if gltf_json_hierarchy is not None and gltf_json_hierarchy:
    gltf_hierarchy_section = ("\\n\\nGLTF/GLB STRUCTURE HIERARCHY:\\n"
        "The following JSON structure shows the node hierarchy and skeleton references from the GLTF/GLB file.\\n"
        "Use the node indices and names to cross-reference with the bone names in the images.\\n\\n"
        + str(gltf_json_hierarchy) + "\\n\\n"
        'The "nodes" array contains node information with indices. The "skins" array contains skeleton joint references.\\n'
        "Match the bone names in the images (bone_0, bone_1, etc.) to the node names and indices in this hierarchy.\\n")

vrm_spec_prompt = ("Analyze the annotated bone structure images and identify VRM bone mappings.\\n" + gltf_hierarchy_section +
    "VRM Humanoid Bone Specification (VRMC_vrm-1.0):\\n\\n"
    "REQUIRED BONES (must be mapped):\\n"
    "- Torso: hips, spine\\n"
    "- Head: head\\n"
    "- Legs: leftUpperLeg, leftLowerLeg, leftFoot, rightUpperLeg, rightLowerLeg, rightFoot\\n"
    "- Arms: leftUpperArm, leftLowerArm, leftHand, rightUpperArm, rightLowerArm, rightHand\\n\\n"
    "OPTIONAL BONES (map if present):\\n"
    "- Torso: chest, upperChest, neck\\n"
    "- Head: leftEye, rightEye, jaw\\n"
    "- Legs: leftToes, rightToes\\n"
    "- Arms: leftShoulder, rightShoulder\\n"
    "- Fingers: left/right thumb, index, middle, ring, little (proximal, intermediate, distal)\\n\\n"
    "PARENT-CHILD RELATIONSHIPS:\\n"
    "- hips (root) → spine → chest → upperChest → neck → head\\n"
    "- upperChest → leftShoulder → leftUpperArm → leftLowerArm → leftHand → fingers\\n"
    "- upperChest → rightShoulder → rightUpperArm → rightLowerArm → rightHand → fingers\\n"
    "- hips → leftUpperLeg → leftLowerLeg → leftFoot → leftToes\\n"
    "- hips → rightUpperLeg → rightLowerLeg → rightFoot → rightToes\\n\\n"
    "ESTIMATED POSITIONS:\\n"
    "- hips: Crotch area\\n"
    "- spine: Top of pelvis\\n"
    "- chest: Bottom of rib cage\\n"
    "- neck: Base of neck\\n"
    "- head: Top of neck\\n"
    "- leftUpperLeg/rightUpperLeg: Groin area\\n"
    "- leftLowerLeg/rightLowerLeg: Knee area\\n"
    "- leftFoot/rightFoot: Ankle area\\n"
    "- leftUpperArm/rightUpperArm: Base of upper arm\\n"
    "- leftLowerArm/rightLowerArm: Elbow area\\n"
    "- leftHand/rightHand: Wrist area\\n\\n"
    "The images show bone structures with:\\n"
    "- Red lines: bone connections\\n"
    "- Green circles: bone head markers (foci)\\n"
    "- Yellow text: bone names (bone_0, bone_1, etc.)\\n\\n"
    "Based on the bone positions, structure, and parent-child relationships visible in the images, identify which numbered bone (bone_0, bone_1, etc.) corresponds to which VRM bone name.\\n\\n"
    "Return ONLY a valid JSON object mapping bone names to VRM names, in this exact format:\\n"
    '{"bone_0": "hips", "bone_1": "spine", "bone_2": "chest", ...}\\n\\n'
    "Do not include any explanation or text outside the JSON object.")

# Prepare images for Qwen3VL
print("Preparing images for Qwen3VL analysis...")
images = []
for img_path in annotated_images:
    img = Image.open(img_path).convert("RGB")
    images.append(img)

print(f"  Loaded {len(images)} images")

# Run Qwen3VL analysis 2 times and merge results
print("")
print("=== Running Qwen3VL Analysis (2 iterations) ===")
all_mappings = []  # Store mappings from each iteration

for iteration in range(2):
    print("")
    print(f"--- Iteration {iteration + 1}/2 ---")

    # Prepare messages with multiple images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": vrm_spec_prompt}
            ] + [{"type": "image", "image": img} for img in images]
        }
    ]

    print("Running Qwen3VL inference...")
    print("  Preparing inputs...")
    import time
    start_time = time.time()

    try:
        # Prepare inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        input_length = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
        prep_time = time.time() - start_time
        prep_time_str = format(prep_time, '.1f')
        print(f"  Input prepared: {input_length} tokens (took {prep_time_str}s)")
        print("  Generating response...")
        print("  NOTE: This may take 2-5 minutes with 4 images and a 4 billion parameter model.")
        print("  The model is processing - please wait...")
        sys.stdout.flush()  # Ensure output is visible

        gen_start = time.time()

        # Generate response
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else None,
        )

        gen_time = time.time() - gen_start
        generated_length = generated_ids.shape[1] if hasattr(generated_ids, 'shape') else 0
        gen_time_str = format(gen_time, '.1f')
        print(f"  Generation complete: {generated_length} total tokens (took {gen_time_str}s)")

        # Extract only the newly generated tokens
        import itertools
        generated_ids_trimmed = []
        zipped_pairs = list(zip(inputs.input_ids, generated_ids))
        for pair in zipped_pairs:
            in_ids, out_ids = pair
            start_idx = len(in_ids)
            trimmed = list(itertools.islice(out_ids, start_idx, None))
            generated_ids_trimmed.append(trimmed)

        # Decode the response
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        response = output_text[0] if output_text else ""
        print("[OK] Qwen3VL response received")
        response_preview = response[0:200] if len(response) > 200 else response
        print(f"Response preview: {response_preview}...")

        # Save Qwen3VL response to file
        response_dir = output_dir / "qwen3vl_responses"
        response_dir.mkdir(exist_ok=True, parents=True)
        response_file = response_dir / f"response_iter{iteration + 1}.txt"
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"  Response saved to: {response_file}")

    except Exception as e:
        print(f"[ERROR] Error during Qwen3VL inference (iteration {iteration + 1}): {e}")
        import traceback
        traceback.print_exc()
        continue  # Continue to next iteration instead of raising

    # Parse JSON from response
    print(f"=== Parsing Bone Mappings (Iteration {iteration + 1}) ===")
    try:
        # Extract JSON from response (may have extra text)
        import re

        # Try to find JSON object - look for opening and closing braces
        json_start = response.find('{')
        if json_start < 0:
            print(f"  [WARN] No JSON object found in response (iteration {iteration + 1})")
            continue

        # Find matching closing brace by counting braces
        brace_count = 0
        json_end = json_start
        for i in range(json_start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i
                    break

        if brace_count != 0:
            print(f"  [WARN] No complete JSON object found in response (iteration {iteration + 1})")
            continue

        json_str = response[json_start:json_end+1]

        # Try to parse JSON
        iteration_mapping = json.loads(json_str)

        # Validate it's a dictionary with string keys and values
        if not isinstance(iteration_mapping, dict):
            print(f"  [WARN] Expected dictionary, got {type(iteration_mapping)} (iteration {iteration + 1})")
            continue

        # Validate all keys are bone names and values are strings
        valid = True
        for key, value in iteration_mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                print(f"  [WARN] Invalid mapping entry: {key} -> {value} (iteration {iteration + 1})")
                valid = False
                break

        if valid:
            all_mappings.append(iteration_mapping)
            print(f"[OK] Parsed {len(iteration_mapping)} bone mappings (iteration {iteration + 1})")
            for old_name, new_name in sorted(iteration_mapping.items()):
                print(f"  {old_name} -> {new_name}")

            # After first iteration, re-annotate images with guessed bone names
            if iteration == 0:
                print("")
                print("=== Re-annotating images with guessed bone names ===")
                try:
                    from PIL import Image, ImageDraw, ImageFont

                    # Try to load fonts
                    try:
                        font_large = ImageFont.truetype("arial.ttf", 20)
                    except:
                        try:
                            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
                        except:
                            font_large = ImageFont.load_default()

                    # Re-annotate each view
                    for view_idx, view_name in enumerate(views):
                        if view_name not in view_bone_data:
                            continue

                        img_path = annotated_images[view_idx]
                        bone_data_list = view_bone_data[view_name]

                        # Load the base image (normal map or white background)
                        try:
                            img_pil = Image.open(img_path).convert("RGBA")
                            draw = ImageDraw.Draw(img_pil)

                            # Redraw labels with VRM names
                            for bone_data in bone_data_list:
                                try:
                                    bone = bone_data['bone']
                                    head_pix = bone_data['head_pix']
                                    label_x = int(bone_data['label_x'])
                                    label_y = int(bone_data['label_y'])
                                    text_width = bone_data['text_width']
                                    text_height = bone_data['text_height']

                                    # Get VRM name from mapping, or keep original name
                                    display_name = iteration_mapping.get(bone.name, bone.name)

                                    # Get updated text size
                                    try:
                                        bbox = draw.textbbox((0, 0), display_name, font=font_large)
                                        new_text_width = bbox[2] - bbox[0]
                                        new_text_height = bbox[3] - bbox[1]
                                    except:
                                        new_text_width = len(display_name) * 12
                                        new_text_height = 22

                                    # Draw semi-transparent background for label
                                    padding = 5
                                    draw.rectangle([label_x - padding, label_y - padding,
                                                   label_x + new_text_width + padding, label_y + new_text_height + padding],
                                                  fill=(0, 0, 0, 200), outline=(255, 255, 0, 255), width=2)

                                    # Draw connecting line from label to bone head
                                    draw.line([head_pix, (label_x, label_y + new_text_height // 2)],
                                             fill=(255, 255, 255, 180), width=2)

                                    # Add text label with VRM name
                                    draw.text((label_x, label_y), display_name,
                                             fill=(255, 255, 255, 255), font=font_large)

                                    # Update bone_data with new text dimensions
                                    bone_data['text_width'] = new_text_width
                                    bone_data['text_height'] = new_text_height

                                except Exception as e:
                                    continue

                            # Save updated image (overwrite original)
                            img_pil.save(img_path)
                            # Also save a copy with iteration number for reference
                            iter_path = annotated_images_dir / f"bone_view_{view_name.lower()}_iter1.png"
                            img_pil.save(iter_path)
                            print(f"  Re-annotated {view_name} with VRM names")

                        except Exception as e:
                            print(f"  [WARN] Error re-annotating {view_name}: {e}")
                            continue

                    # Reload images for next iteration
                    images = []
                    for img_path in annotated_images:
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                    print("[OK] Images re-annotated and reloaded for next iteration")

                except Exception as e:
                    print(f"[WARN] Error during re-annotation: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with original images

    except Exception as e:
        print(f"[ERROR] Error parsing JSON from Qwen3VL response (iteration {iteration + 1}): {e}")
        print(f"Response was: {response}")
        import traceback
        traceback.print_exc()
        continue  # Continue to next iteration

# Merge mappings from all iterations using voting
print("")
print("=== Merging Results from 2 Iterations ===")
if len(all_mappings) == 0:
    raise ValueError("No valid mappings found in any iteration")

# Count votes for each bone->VRM mapping
from collections import defaultdict, Counter
vote_counts = defaultdict(Counter)

for mapping in all_mappings:
    for bone_name, vrm_name in mapping.items():
        vote_counts[bone_name][vrm_name] += 1

# Create final mapping using majority vote
bone_mapping = {}
for bone_name, votes in vote_counts.items():
    # Get the most common VRM name for this bone
    most_common = votes.most_common(1)[0]
    vrm_name, count = most_common
    bone_mapping[bone_name] = vrm_name
    print(f"  {bone_name} -> {vrm_name} (voted {count}/{len(all_mappings)} times)")

print(f"[OK] Merged {len(bone_mapping)} bone mappings from {len(all_mappings)} iterations")

# Save final merged mapping to file
mapping_file = output_dir / "bone_mapping_final.json"
with open(mapping_file, 'w', encoding='utf-8') as f:
    json.dump(bone_mapping, f, indent=2, ensure_ascii=False)
print(f"Final bone mapping saved to: {mapping_file}")

# Validate VRM compliance
print("=== Validating VRM Compliance ===")
required_bones = ["hips", "spine", "head",
                  "leftUpperLeg", "leftLowerLeg", "leftFoot",
                  "rightUpperLeg", "rightLowerLeg", "rightFoot",
                  "leftUpperArm", "leftLowerArm", "leftHand",
                  "rightUpperArm", "rightLowerArm", "rightHand"]

mapped_bones = set(bone_mapping.values())
missing_required = [b for b in required_bones if b not in mapped_bones]
if missing_required:
    print(f"[WARN] Missing required VRM bones: {missing_required}")

# Check for duplicate mappings
if len(mapped_bones) != len(bone_mapping.values()):
    print("[WARN] Duplicate VRM bone mappings detected")
    # Find duplicates
    from collections import Counter
    counts = Counter(bone_mapping.values())
    duplicates = [bone for bone, count in counts.items() if count > 1]
    print(f"  Duplicates: {duplicates}")

print("[OK] VRM compliance check completed")

# Rename bones in Blender
print("")
print("=== Renaming Bones ===")
bpy.context.view_layer.objects.active = armature

# Ensure we're in object mode first
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

# Switch to edit mode for renaming
bpy.ops.object.mode_set(mode='EDIT')

renamed_count = 0
skipped_count = 0
for bone in armature.data.edit_bones:
    if bone.name in bone_mapping:
        old_name = bone.name
        new_name = bone_mapping[bone.name]

        # Check if target name already exists
        if new_name in [b.name for b in armature.data.edit_bones if b != bone]:
            print(f"  [WARN] Skipping {old_name} -> {new_name} (target name already exists)")
            skipped_count += 1
            continue

        try:
            bone.name = new_name
            print(f"  {old_name} -> {new_name}")
            renamed_count += 1
        except Exception as e:
            error_msg = f"  [ERROR] Error renaming {old_name} -> {new_name}"
            print(error_msg + ": " + str(e))
            skipped_count += 1

# Switch back to object mode
bpy.ops.object.mode_set(mode='OBJECT')
print(f"[OK] Renamed {renamed_count} bones")
if skipped_count > 0:
    print(f"[WARN] Skipped {skipped_count} bones (conflicts or errors)")

# Export to USDC
print("")
print("=== Exporting to USDC ===")
output_path = config.get('output_path', str(Path(input_path).with_suffix('.usdc')))
output_path = str(Path(output_path)).replace("\\\\", "/")

# Ensure output directory exists
output_dir = Path(output_path).parent
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Exporting to: {output_path}")

try:
    # Select all objects for export
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in ['MESH', 'ARMATURE']:
            obj.select_set(True)

    # Export as USD
    bpy.ops.wm.usd_export(
        filepath=output_path,
        selected_objects_only=False,
        export_materials=True,
        export_uvmaps=True,
        export_normals=True,
        root_prim_path="",
        material_prim_path="materials",
        preview_surface_prim_path="preview",
        shader_prim_path="shaders",
        export_armatures=True,
        export_skins=True,
        relative_paths=True
    )
    print(f"[OK] Exported to: {output_path}")
except Exception as e:
    print(f"[ERROR] Error exporting to USDC: {e}")
    import traceback
    traceback.print_exc()
    raise

print("")
print("=== Complete ===")
print(f"Renamed {renamed_count} bones to VRM specification")
print(f"Exported to: {output_path}")
print(f"Intermediate files saved to: {output_dir}")
print(f"  - Normal maps: {normal_maps_dir}")
print(f"  - Annotated images: {annotated_images_dir}")
""", %{})
rescue
  e ->
    IO.puts("\n[ERROR] Error during processing: #{inspect(e)}")
    reraise e, __STACKTRACE__
after
  # Clean up temp file
  case File.rm(config_file) do
    :ok -> :ok
    {:error, :enoent} -> :ok  # File doesn't exist, that's fine
    {:error, reason} -> IO.puts("Warning: Could not delete temp file: #{inspect(reason)}")
  end
end
end)

IO.puts("\n=== Complete ===")
IO.puts("VRM bone renaming completed successfully!")

# Display OpenTelemetry trace (save to output directory)
SpanCollector.display_trace("output")
