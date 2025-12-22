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
# bpy for Blender, pillow for image processing
Pythonx.uv_init("""
[project]
name = "vrm-bone-renamer"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "bpy==4.5.*",
  "pillow",
]
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
output_dir = Path(config.get('output_dir', 'output'))
output_path = config.get('output_path', str(Path(input_path).with_suffix('.usdc')))
output_dir.mkdir(exist_ok=True, parents=True)

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

# Save extracted bone data to output directory
bone_data_file = output_dir / "extracted_bone_data.json"
with open(bone_data_file, 'w', encoding='utf-8') as f:
    json.dump(bone_data, f, indent=2, ensure_ascii=False)
print(f"  Extracted bone data saved to: {bone_data_file}")

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
        export_materials='EXPORT',
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

# Create output directory structure for intermediate files (output_dir already defined above)
output_dir.mkdir(exist_ok=True, parents=True)

# Create subdirectories for intermediate files
annotated_images_dir = output_dir / "annotated_images"
annotated_images_dir.mkdir(exist_ok=True, parents=True)

# Store original materials BEFORE any modifications (for PBR map extraction)
original_materials = {}
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj.data.materials:
        original_materials[obj.name] = [mat.name for mat in obj.data.materials]

# Capture screenshots from different camera angles (8 views using Fibonacci sphere)
# This maximizes viewing angle coverage using golden angle spiral distribution
# Based on HKU/SAMPart3D camera trajectory methods
import math
num_views = 4
views = [f'VIEW_{i}' for i in range(num_views)]
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

# Calculate camera positioning from mesh extrema (mesh bounding boxes)
import mathutils
bbox_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
bbox_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
has_meshes = False

# Calculate bounding box from visible mesh objects
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and not obj.hide_get():
        # Get mesh bounding box
        for vertex in obj.bound_box:
            world_vertex = obj.matrix_world @ mathutils.Vector(vertex)
            bbox_min = mathutils.Vector((min(bbox_min.x, world_vertex.x),
                                       min(bbox_min.y, world_vertex.y),
                                       min(bbox_min.z, world_vertex.z)))
            bbox_max = mathutils.Vector((max(bbox_max.x, world_vertex.x),
                                       max(bbox_max.y, world_vertex.y),
                                       max(bbox_max.z, world_vertex.z)))
            has_meshes = True

if has_meshes:
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

    # Recalculate bounding box for each frame based on currently visible mesh objects
    bbox_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    has_meshes = False

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and not obj.hide_get():
            # Get mesh bounding box
            for vertex in obj.bound_box:
                world_vertex = obj.matrix_world @ mathutils.Vector(vertex)
                bbox_min = mathutils.Vector((min(bbox_min.x, world_vertex.x),
                                           min(bbox_min.y, world_vertex.y),
                                           min(bbox_min.z, world_vertex.z)))
                bbox_max = mathutils.Vector((max(bbox_max.x, world_vertex.x),
                                           max(bbox_max.y, world_vertex.y),
                                           max(bbox_max.z, world_vertex.z)))
                has_meshes = True

    if has_meshes:
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

    # Step 1: Render normal map (with mesh visible, armature hidden) - NO LABELS
    normal_map_path = annotated_images_dir / f"bone_view_{view_name.lower()}_normal.png"
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

        # Restore scene state: re-hide meshes and re-show armature for label projection
        for obj in mesh_objects:
            obj.hide_set(True)
        armature.hide_set(False)

        # Add SAM-style labels to normal map using foci points (bone heads)
        try:
            from PIL import Image, ImageDraw, ImageFont
            normal_img = Image.open(normal_map_path).convert("RGBA")
            draw = ImageDraw.Draw(normal_img)

            # Try to load fonts
            try:
                font_large = ImageFont.truetype("arial.ttf", 18)
            except:
                try:
                    font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
                except:
                    font_large = ImageFont.load_default()

            res_x, res_y = scene.render.resolution_x, scene.render.resolution_y
            bone_data_list = []

            # Collect bone foci points (head positions) - SAM strategy
            for bone in armature.data.bones:
                try:
                    head_world = armature.matrix_world @ Vector(bone.head_local)
                    head_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, head_world)
                    head_pix = (int(head_2d.x * res_x), int((1 - head_2d.y) * res_y))

                    # Only process if within image bounds
                    if (0 <= head_pix[0] < res_x and 0 <= head_pix[1] < res_y):
                        # Get text size
                        try:
                            bbox = draw.textbbox((0, 0), bone.name, font=font_large)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        except:
                            text_width = len(bone.name) * 10
                            text_height = 18

                        # Initial label position offset from foci point
                        offset_x = 20
                        offset_y = -20
                        label_x = float(head_pix[0] + offset_x)
                        label_y = float(head_pix[1] + offset_y)

                        bone_data_list.append({
                            'bone': bone,
                            'foci': head_pix,  # Foci point (bone head)
                            'label_x': label_x,
                            'label_y': label_y,
                            'text_width': text_width,
                            'text_height': text_height
                        })
                except:
                    continue

            # Simple overlap avoidance: adjust positions to avoid overlaps
            padding = 8
            for i, bone_data in enumerate(bone_data_list):
                for j, other_data in enumerate(bone_data_list):
                    if i == j:
                        continue

                    # Check if labels overlap
                    x1, y1 = bone_data['label_x'], bone_data['label_y']
                    w1, h1 = bone_data['text_width'], bone_data['text_height']
                    x2, y2 = other_data['label_x'], other_data['label_y']
                    w2, h2 = other_data['text_width'], other_data['text_height']

                    if not (x1 + w1 + padding < x2 or x2 + w2 + padding < x1 or
                           y1 + h1 + padding < y2 or y2 + h2 + padding < y1):
                        # Overlap detected - shift this label
                        dx = x1 - x2
                        dy = y1 - y2
                        dist = math.sqrt(dx*dx + dy*dy) if (dx != 0 or dy != 0) else 1.0
                        if dist > 0:
                            shift_x = (dx / dist) * (w1 + w2) / 2
                            shift_y = (dy / dist) * (h1 + h2) / 2
                            bone_data['label_x'] += shift_x * 0.3
                            bone_data['label_y'] += shift_y * 0.3

            # Draw labels with translucent backgrounds (SAM-style)
            for bone_data in bone_data_list:
                try:
                    bone = bone_data['bone']
                    foci = bone_data['foci']
                    label_x = int(bone_data['label_x'])
                    label_y = int(bone_data['label_y'])
                    text_width = bone_data['text_width']
                    text_height = bone_data['text_height']

                    # Ensure label is within bounds
                    label_x = max(10, min(res_x - text_width - 10, label_x))
                    label_y = max(10, min(res_y - text_height - 10, label_y))

                    # Draw translucent connecting line from foci to label
                    label_center_y = label_y + text_height // 2
                    draw.line([foci, (label_x, label_center_y)],
                             fill=(255, 255, 255, 120), width=1)

                    # Draw foci point (small circle)
                    foci_radius = 4
                    draw.ellipse([foci[0]-foci_radius, foci[1]-foci_radius,
                                 foci[0]+foci_radius, foci[1]+foci_radius],
                                fill=(255, 255, 0, 180), outline=(255, 255, 255, 150), width=1)

                    # Draw text (translucent white)
                    draw.text((label_x, label_y), bone.name,
                             fill=(255, 255, 255, 200), font=font_large)
                except:
                    continue

            # Save annotated normal map
            normal_img.save(normal_map_path)
            # Store bone data for re-annotation
            view_bone_data[view_name] = bone_data_list
            print(f"    Normal map annotated with labels: {normal_map_path}")
        except Exception as e:
            print(f"    Error adding labels to normal map: {e}")
            import traceback
            traceback.print_exc()

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

    # Store bone data for this view for potential re-annotation with VRM names
    # (bone positions collected during normal map labeling)

print(f"Created {len(views)} normal maps")

# Create base color maps
print("")
print("=== Creating Base Color Maps ===")

# Generate base color maps for each view (original_materials already stored above)
pbr_images = {"basecolor": []}

for i, view_name in enumerate(views):
    print(f"  Capturing {view_name} PBR maps...")

    # Ensure meshes are visible
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.hide_set(False)
    armature.hide_set(False)

    # Force scene update
    bpy.context.view_layer.update()

    # Recalculate bounding box for camera positioning
    bbox_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    has_bones = False

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and not obj.hide_get():
            # Get mesh bounding box
            for vertex in obj.bound_box:
                world_vertex = obj.matrix_world @ mathutils.Vector(vertex)
                bbox_min = mathutils.Vector((min(bbox_min.x, world_vertex.x),
                                           min(bbox_min.y, world_vertex.y),
                                           min(bbox_min.z, world_vertex.z)))
                bbox_max = mathutils.Vector((max(bbox_max.x, world_vertex.x),
                                           max(bbox_max.y, world_vertex.y),
                                           max(bbox_max.z, world_vertex.z)))
                has_meshes = True

    if has_meshes:
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

    # Position camera
    x, y, z = sphere_points[i]
    camera_pos = mathutils.Vector((x * frame_distance, y * frame_distance, z * frame_distance))
    camera.location = frame_center + camera_pos
    look_at(camera, frame_center)

    # Disable compositor for simple rendering
    scene.use_nodes = False

    # Generate Base Color map
    try:
        # Create base color materials for each object that sample from their original materials
        # This ensures we get the actual base colors (including textures) from each object
        basecolor_materials = {}

        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue

            # Create a material that outputs base color from original material
            basecolor_mat = bpy.data.materials.new(name=f"BaseColorMaterial_{obj.name}")
            basecolor_mat.use_nodes = True
            nodes_bc = basecolor_mat.node_tree.nodes
            links_bc = basecolor_mat.node_tree.links

            # Clear default nodes
            for node in nodes_bc:
                nodes_bc.remove(node)

            # Try to get base color from original material
            base_color_input = None
            if obj.name in original_materials and len(original_materials[obj.name]) > 0:
                orig_mat_name = original_materials[obj.name][0]
                if orig_mat_name in bpy.data.materials:
                    orig_mat = bpy.data.materials[orig_mat_name]
                    if orig_mat.use_nodes:
                        # Find Principled BSDF in original material
                        for node in orig_mat.node_tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                base_color_input = node.inputs['Base Color']
                                break

                        # If we found a base color input, try to copy its connection
                        if base_color_input and base_color_input.is_linked:
                            # Get the connected node
                            connected_node = base_color_input.links[0].from_node
                            connected_socket = base_color_input.links[0].from_socket

                            # Copy the connected node to our new material
                            # Create a shader node that outputs the base color
                            # Use a Material Output node to sample from original material
                            # Actually, we need to copy the node tree structure
                            # For simplicity, let's use the base color value or texture

                            # Create an emission shader
                            emission_bc = nodes_bc.new(type='ShaderNodeEmission')
                            output_bc = nodes_bc.new(type='ShaderNodeOutputMaterial')

                            # Try to get the actual color/texture value
                            if connected_node.type == 'TEX_IMAGE':
                                # Copy the image texture
                                tex_node = nodes_bc.new(type='ShaderNodeTexImage')
                                if connected_node.image:
                                    tex_node.image = connected_node.image
                                    links_bc.new(tex_node.outputs['Color'], emission_bc.inputs['Color'])
                                else:
                                    # Fallback to default color
                                    emission_bc.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)
                            elif connected_node.type in ['RGB', 'VALUE']:
                                # Use the color value
                                if hasattr(connected_node, 'outputs') and len(connected_node.outputs) > 0:
                                    links_bc.new(connected_node.outputs[0], emission_bc.inputs['Color'])
                                else:
                                    emission_bc.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)
                            else:
                                # For other node types, try to get the output
                                if hasattr(connected_node, 'outputs') and 'Color' in connected_node.outputs:
                                    links_bc.new(connected_node.outputs['Color'], emission_bc.inputs['Color'])
                                else:
                                    emission_bc.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)

                            emission_bc.inputs['Strength'].default_value = 1.0
                            links_bc.new(emission_bc.outputs['Emission'], output_bc.inputs['Surface'])
                            basecolor_materials[obj.name] = basecolor_mat
                            continue

            # Fallback: use default color or extract from Principled BSDF default value
            base_color = (0.8, 0.8, 0.8, 1.0)  # Default gray
            if base_color_input and not base_color_input.is_linked:
                base_color = base_color_input.default_value
                if len(base_color) == 3:
                    base_color = base_color + (1.0,)

            # Create Emission shader to output base color directly (no lighting needed)
            emission_bc = nodes_bc.new(type='ShaderNodeEmission')
            output_bc = nodes_bc.new(type='ShaderNodeOutputMaterial')

            # Set emission color to base color
            emission_bc.inputs['Color'].default_value = base_color[:3] + (1.0,)
            emission_bc.inputs['Strength'].default_value = 1.0

            # Link to output
            links_bc.new(emission_bc.outputs['Emission'], output_bc.inputs['Surface'])
            basecolor_materials[obj.name] = basecolor_mat

        # Apply base color materials to objects
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH' and obj.name in basecolor_materials:
                if len(obj.data.materials) == 0:
                    obj.data.materials.append(basecolor_materials[obj.name])
                else:
                    obj.data.materials[0] = basecolor_materials[obj.name]

        # Render base color map
        basecolor_path = annotated_images_dir / f"bone_view_{view_name.lower()}_basecolor.png"
        scene.render.filepath = str(basecolor_path)
        bpy.ops.render.render(write_still=True)
        print(f"    Base color map rendered: {basecolor_path}")

        # Add SAM-style labels to base color map using foci points
        try:
            from PIL import Image, ImageDraw, ImageFont
            basecolor_img = Image.open(basecolor_path).convert("RGBA")
            draw = ImageDraw.Draw(basecolor_img)

            # Try to load fonts
            try:
                font_large = ImageFont.truetype("arial.ttf", 18)
            except:
                try:
                    font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
                except:
                    font_large = ImageFont.load_default()

            res_x, res_y = scene.render.resolution_x, scene.render.resolution_y
            bone_data_list = []

            # Collect bone foci points (head positions) - SAM strategy
            for bone in armature.data.bones:
                try:
                    head_world = armature.matrix_world @ Vector(bone.head_local)
                    head_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, head_world)
                    head_pix = (int(head_2d.x * res_x), int((1 - head_2d.y) * res_y))

                    # Only process if within image bounds
                    if (0 <= head_pix[0] < res_x and 0 <= head_pix[1] < res_y):
                        # Get text size
                        try:
                            bbox = draw.textbbox((0, 0), bone.name, font=font_large)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        except:
                            text_width = len(bone.name) * 10
                            text_height = 18

                        # Initial label position offset from foci point
                        offset_x = 20
                        offset_y = -20
                        label_x = float(head_pix[0] + offset_x)
                        label_y = float(head_pix[1] + offset_y)

                        bone_data_list.append({
                            'bone': bone,
                            'foci': head_pix,  # Foci point (bone head)
                            'label_x': label_x,
                            'label_y': label_y,
                            'text_width': text_width,
                            'text_height': text_height
                        })
                except:
                    continue

            # Simple overlap avoidance: adjust positions to avoid overlaps
            padding = 8
            for i, bone_data in enumerate(bone_data_list):
                for j, other_data in enumerate(bone_data_list):
                    if i == j:
                        continue

                    # Check if labels overlap
                    x1, y1 = bone_data['label_x'], bone_data['label_y']
                    w1, h1 = bone_data['text_width'], bone_data['text_height']
                    x2, y2 = other_data['label_x'], other_data['label_y']
                    w2, h2 = other_data['text_width'], other_data['text_height']

                    if not (x1 + w1 + padding < x2 or x2 + w2 + padding < x1 or
                           y1 + h1 + padding < y2 or y2 + h2 + padding < y1):
                        # Overlap detected - shift this label
                        dx = x1 - x2
                        dy = y1 - y2
                        dist = math.sqrt(dx*dx + dy*dy) if (dx != 0 or dy != 0) else 1.0
                        if dist > 0:
                            shift_x = (dx / dist) * (w1 + w2) / 2
                            shift_y = (dy / dist) * (h1 + h2) / 2
                            bone_data['label_x'] += shift_x * 0.3
                            bone_data['label_y'] += shift_y * 0.3

            # Draw labels with translucent backgrounds (SAM-style)
            for bone_data in bone_data_list:
                try:
                    bone = bone_data['bone']
                    foci = bone_data['foci']
                    label_x = int(bone_data['label_x'])
                    label_y = int(bone_data['label_y'])
                    text_width = bone_data['text_width']
                    text_height = bone_data['text_height']

                    # Ensure label is within bounds
                    label_x = max(10, min(res_x - text_width - 10, label_x))
                    label_y = max(10, min(res_y - text_height - 10, label_y))

                    # Draw translucent connecting line from foci to label
                    label_center_y = label_y + text_height // 2
                    draw.line([foci, (label_x, label_center_y)],
                             fill=(255, 255, 255, 120), width=1)

                    # Draw foci point (small circle)
                    foci_radius = 4
                    draw.ellipse([foci[0]-foci_radius, foci[1]-foci_radius,
                                 foci[0]+foci_radius, foci[1]+foci_radius],
                                fill=(255, 255, 0, 180), outline=(255, 255, 255, 150), width=1)

                    # Draw text (translucent white)
                    draw.text((label_x, label_y), bone.name,
                             fill=(255, 255, 255, 200), font=font_large)
                except:
                    continue

            # Save annotated base color map
            basecolor_img.save(basecolor_path)
            print(f"    Base color map annotated with labels: {basecolor_path}")
        except Exception as e:
            print(f"    Error adding labels to base color map: {e}")
            import traceback
            traceback.print_exc()

        pbr_images["basecolor"].append(str(basecolor_path))

        # Clean up temporary materials
        for mat in basecolor_materials.values():
            if mat.name in bpy.data.materials:
                bpy.data.materials.remove(mat)

    except Exception as e:
        print(f"    Error rendering base color map: {e}")
        import traceback
        traceback.print_exc()

print(f"Created {len(pbr_images['basecolor'])} base color maps")

# Restore original materials
for obj_name, mat_names in original_materials.items():
    obj = bpy.data.objects.get(obj_name)
    if obj and obj.type == 'MESH':
        obj.data.materials.clear()
        for mat_name in mat_names:
            if mat_name in bpy.data.materials:
                obj.data.materials.append(bpy.data.materials[mat_name])

# Geometric Bone Mapping Analysis
print("=== Geometric Bone Mapping Analysis ===")
print("Using geometric decision tree based on bone positions, hierarchy, and GLTF data")

import math

# Helper function to calculate bone length
def bone_length(bone):
    head = Vector(bone.head_local)
    tail = Vector(bone.tail_local)
    return (tail - head).length

# Helper function to get bone depth in hierarchy
def get_bone_depth(bone, armature_data):
    depth = 0
    current = bone
    while current.parent:
        depth += 1
        current = current.parent
    return depth

# Helper function to calculate tapered capsule properties for a bone
# Returns a dictionary with:
# - direction: normalized direction vector
# - length: bone length
# - head_radius: estimated radius at head (based on children or default)
# - tail_radius: estimated radius at tail (based on children or default)
# - orientation: classification of bone orientation (vertical_up, vertical_down, horizontal, etc.)
def calculate_tapered_capsule(bone, bone_info_dict=None):
    head = Vector(bone.head_local)
    tail = Vector(bone.tail_local)
    direction = tail - head
    length = direction.length

    if length < 1e-6:
        # Degenerate bone
        return {
            'direction': Vector((0, 1, 0)),
            'length': 0,
            'head_radius': 0.01,
            'tail_radius': 0.01,
            'orientation': 'degenerate'
        }

    direction_normalized = direction / length

    # Estimate radii based on children or use default proportional to length
    # Typical bone radius is about 1-5% of length
    default_radius = max(length * 0.02, 0.005)

    head_radius = default_radius
    tail_radius = default_radius

    # If we have bone info, check children to estimate radii
    if bone_info_dict and bone.name in bone_info_dict:
        children = bone_info_dict[bone.name].get('children', [])
        if len(children) > 0:
            # Estimate radius based on distance to nearest child
            # This is a heuristic - actual bone radius would need mesh data
            tail_radius = default_radius * 1.2  # Slightly larger at tail where children attach

    # Classify orientation based on direction vector
    # Y is typically vertical (up), X is lateral (left/right), Z is forward/back
    y_component = abs(direction_normalized.y)
    x_component = abs(direction_normalized.x)
    z_component = abs(direction_normalized.z)

    # Determine primary orientation
    if y_component > max(x_component, z_component) * 1.5:
        # Primarily vertical
        if direction_normalized.y > 0:
            orientation = 'vertical_up'
        else:
            orientation = 'vertical_down'
    elif x_component > max(y_component, z_component) * 1.5:
        # Primarily horizontal (lateral)
        orientation = 'horizontal_lateral'
    elif z_component > max(x_component, y_component) * 1.5:
        # Primarily forward/back
        orientation = 'horizontal_forward'
    else:
        # Mixed orientation
        orientation = 'mixed'

    return {
        'direction': direction_normalized,
        'length': length,
        'head_radius': head_radius,
        'tail_radius': tail_radius,
        'orientation': orientation,
        'y_component': y_component,
        'x_component': x_component,
        'z_component': z_component
    }

# Helper function to calculate centroid of bone structure
def calculate_centroid(armature_data):
    positions = []
    for bone in armature_data.bones:
        positions.append(bone.head_local)
        positions.append(bone.tail_local)
    if len(positions) == 0:
        return Vector((0, 0, 0))
    centroid = Vector((0, 0, 0))
    for pos in positions:
        centroid += Vector(pos)
    return centroid / len(positions)

# Build bone information dictionary with geometric properties
bone_info = {}
centroid = calculate_centroid(armature.data)

# First pass: build basic bone info
for bone in armature.data.bones:
    head = Vector(bone.head_local)
    tail = Vector(bone.tail_local)
    length = bone_length(bone)
    depth = get_bone_depth(bone, armature.data)

    # Calculate relative position to centroid
    head_rel = head - centroid
    tail_rel = tail - centroid

    # Determine left/right (negative X = left, positive X = right in typical VRM)
    is_left = head_rel.x < 0 or tail_rel.x < 0
    is_right = head_rel.x > 0 or tail_rel.x > 0

    # Determine vertical position (Y axis - up/down)
    vertical_pos = (head.y + tail.y) / 2

    # Determine if bone is in upper body (above hips) or lower body
    is_upper_body = vertical_pos > centroid.y

    bone_info[bone.name] = {
        'bone': bone,
        'head': head,
        'tail': tail,
        'head_local': list(bone.head_local),
        'tail_local': list(bone.tail_local),
        'length': length,
        'depth': depth,
        'parent': bone.parent.name if bone.parent else None,
        'children': [child.name for child in bone.children],
        'head_rel': head_rel,
        'tail_rel': tail_rel,
        'is_left': is_left,
        'is_right': is_right,
        'vertical_pos': vertical_pos,
        'is_upper_body': is_upper_body,
        'center': (head + tail) / 2
    }

# Second pass: add tapered capsule properties (needs bone_info for radius estimation)
for bone in armature.data.bones:
    capsule = calculate_tapered_capsule(bone, bone_info)
    bone_info[bone.name]['capsule'] = capsule
    bone_info[bone.name]['orientation'] = capsule['orientation']
    bone_info[bone.name]['direction'] = capsule['direction']

# Find root bone (no parent)
root_bones = [name for name, info in bone_info.items() if info['parent'] is None]
if len(root_bones) == 0:
    raise ValueError("No root bone found")
root_bone_name = root_bones[0]  # Use first root bone
print(f"Root bone identified: {root_bone_name}")

# Geometric Decision Tree for VRM Bone Mapping
bone_mapping = {}

# Step 1: Identify hips (root bone, typically lowest Y position among roots)
hips_candidates = root_bones
if len(hips_candidates) > 1:
    # Choose the one with lowest Y position
    hips_candidate = min(hips_candidates, key=lambda name: bone_info[name]['head'].y)
else:
    hips_candidate = hips_candidates[0]
bone_mapping[hips_candidate] = "hips"
print(f"  {hips_candidate} -> hips (root bone)")

# Step 2: Build spine chain (hips -> spine -> chest -> upperChest -> neck -> head)
# Look for the most vertical upward chain (highest Y)
def find_spine_chain(start_bone_name):
    chain = [start_bone_name]
    current = bone_info[start_bone_name]['bone']

    # Get hips position for reference
    hips_pos = bone_info[start_bone_name]['head']

    max_depth = 10
    for _ in range(max_depth):
        children = [child for child in current.children]
        if len(children) == 0:
            break

        # Filter out legs and arms using tapered capsule properties
        # Spine bones: vertical_up orientation, going upward
        # Legs: vertical_down orientation, going downward
        # Arms: horizontal_lateral orientation
        spine_candidates = []
        for child in children:
            if child.name not in bone_info:
                continue

            child_capsule = bone_info[child.name]['capsule']
            child_head = Vector(child.head_local)
            child_tail = Vector(child.tail_local)

            # Use tapered capsule orientation to classify
            orientation = child_capsule['orientation']
            y_component = child_capsule['y_component']

            # Spine bones: primarily vertical and going upward
            # Check both orientation and actual Y position
            goes_upward = child_tail.y > child_head.y and child_tail.y > hips_pos.y - 0.05
            is_vertical = orientation == 'vertical_up' or (y_component > 0.7 and goes_upward)

            # Exclude horizontal bones (arms/shoulders) and downward bones (legs)
            is_not_horizontal = orientation != 'horizontal_lateral' and orientation != 'horizontal_forward'
            is_not_downward = orientation != 'vertical_down' and child_tail.y >= hips_pos.y - 0.05

            if is_vertical and goes_upward and is_not_horizontal and is_not_downward:
                spine_candidates.append(child)

        if len(spine_candidates) == 0:
            # If no clear spine candidates, use the one with highest Y (but still going upward)
            upward_children = [c for c in children if Vector(c.tail_local).y > Vector(c.head_local).y]
            if len(upward_children) > 0:
                next_child = max(upward_children, key=lambda b: b.head_local[1])
            else:
                # Last resort: use highest Y child
                next_child = max(children, key=lambda b: b.head_local[1])
        else:
            # Choose the most vertical upward child
            next_child = max(spine_candidates, key=lambda b: b.head_local[1])

        chain.append(next_child.name)
        current = next_child

        # Stop if we've reached a bone with no children (likely head)
        if len(current.children) == 0:
            break
        if len(chain) > 6:
            break

    return chain

spine_chain = find_spine_chain(hips_candidate)
print(f"  Spine chain: {' -> '.join(spine_chain)}")

# Map spine chain to VRM bones
vrm_spine_names = ["hips", "spine", "chest", "upperChest", "neck", "head"]
for i, bone_name in enumerate(spine_chain[:len(vrm_spine_names)]):
    if bone_name not in bone_mapping:
        bone_mapping[bone_name] = vrm_spine_names[i]
        print(f"  {bone_name} -> {vrm_spine_names[i]}")

# Step 3: Find upperChest or chest for arm attachment
upper_chest_name = None
for name, vrm_name in bone_mapping.items():
    if vrm_name in ["upperChest", "chest"]:
        upper_chest_name = name
        break

# If no upperChest/chest found, use the highest spine bone before head
if upper_chest_name is None:
    # Find the highest bone in spine chain that's not head
    for bone_name in reversed(spine_chain):
        if bone_name in bone_mapping and bone_mapping[bone_name] != "hips" and bone_mapping[bone_name] != "head":
            upper_chest_name = bone_name
            break

# Step 4: Find shoulders and arms (left/right from upper spine bones)
# Shoulders are direct children of chest/upperChest that extend horizontally
# Arms typically branch from shoulders or directly from chest/upperChest
arm_attachment_bones = []
if upper_chest_name:
    arm_attachment_bones.append(upper_chest_name)
# Also check chest if it exists
for name, vrm_name in bone_mapping.items():
    if vrm_name == "chest" and name not in arm_attachment_bones:
        arm_attachment_bones.append(name)

# First, detect shoulders as direct children of chest/upperChest
# Shoulders are characterized by: horizontal_lateral orientation, short length, branching from upperChest/chest
left_shoulder = None
right_shoulder = None

for attach_bone_name in arm_attachment_bones:
    attach_bone = bone_info[attach_bone_name]['bone']
    for child in attach_bone.children:
        if child.name in bone_mapping:
            continue
        if child.name not in bone_info:
            continue

        child_info = bone_info[child.name]
        child_capsule = child_info['capsule']
        child_center = child_info['center']

        # Use tapered capsule properties to identify shoulders
        orientation = child_capsule['orientation']
        length = child_capsule['length']
        x_component = child_capsule['x_component']

        # Shoulders are:
        # 1. Horizontally oriented (lateral, not vertical)
        # 2. Relatively short (shorter than typical arm bones)
        # 3. Have significant X component (extend left/right)
        is_horizontal = orientation == 'horizontal_lateral' or x_component > 0.6
        is_short = length < 0.15  # Shoulders are typically shorter than upper arms

        # Additional check: horizontal extent vs vertical extent
        child_head = child_info['head']
        child_tail = child_info['tail']
        horizontal_extent = math.sqrt((child_tail.x - child_head.x)**2 + (child_tail.z - child_head.z)**2)
        vertical_extent = abs(child_tail.y - child_head.y)
        is_more_horizontal = horizontal_extent > vertical_extent * 1.2

        if (is_horizontal or is_more_horizontal) and is_short:
            # Check X position to determine left/right
            if child_center.x > 0.01 and right_shoulder is None:  # Right side (positive X)
                right_shoulder = child
            elif child_center.x < -0.01 and left_shoulder is None:  # Left side (negative X)
                left_shoulder = child

# Map shoulders if found
if left_shoulder and left_shoulder.name not in bone_mapping:
    bone_mapping[left_shoulder.name] = "leftShoulder"
    print(f"  {left_shoulder.name} -> leftShoulder")

if right_shoulder and right_shoulder.name not in bone_mapping:
    bone_mapping[right_shoulder.name] = "rightShoulder"
    print(f"  {right_shoulder.name} -> rightShoulder")

# Find arm start points (shoulders if found, otherwise direct children of attachment bones)
left_arm_start = left_shoulder
right_arm_start = right_shoulder

# If shoulders not found, look for arm branches directly from attachment points
# Use tapered capsule properties to identify arm bones (horizontal orientation)
if left_arm_start is None or right_arm_start is None:
    for attach_bone_name in arm_attachment_bones:
        attach_bone = bone_info[attach_bone_name]['bone']
        for child in attach_bone.children:
            if child.name in bone_mapping:
                continue
            if child.name not in bone_info:
                continue

            child_info = bone_info[child.name]
            child_capsule = child_info['capsule']
            child_center = child_info['center']

            # Use tapered capsule to identify arm-like bones
            orientation = child_capsule['orientation']
            x_component = child_capsule['x_component']

            # Arms are horizontally oriented (lateral) or have significant X component
            is_arm_like = (orientation == 'horizontal_lateral' or
                          orientation == 'horizontal_forward' or
                          x_component > 0.5)

            # Check if it's clearly on left or right side and arm-like
            if is_arm_like:
                if child_center.x > 0.01 and right_arm_start is None:
                    right_arm_start = child
                elif child_center.x < -0.01 and left_arm_start is None:
                    left_arm_start = child

# Map arms
# If shoulders are already mapped, start from the first child of the shoulder
# Otherwise, the first bone in the chain might be the shoulder or upper arm
if left_arm_start:
    # If shoulder is already mapped, start arm chain from its first child
    if left_arm_start.name in bone_mapping and bone_mapping[left_arm_start.name] == "leftShoulder":
        if len(left_arm_start.children) > 0:
            arm_chain_start = max(left_arm_start.children, key=lambda b: bone_length(b))
        else:
            arm_chain_start = None
    else:
        arm_chain_start = left_arm_start

    if arm_chain_start:
        # Follow left arm chain
        arm_chain = [arm_chain_start]
        current = arm_chain_start
        for _ in range(3):  # upperArm -> lowerArm -> hand (shoulder already mapped if present)
            if len(current.children) > 0:
                # Choose child with longest bone (main arm chain)
                next_bone = max(current.children, key=lambda b: bone_length(b))
                arm_chain.append(next_bone)
                current = next_bone
            else:
                break

        # Map: upperArm -> lowerArm -> hand
        vrm_arm_names = ["leftUpperArm", "leftLowerArm", "leftHand"]
        for i, bone in enumerate(arm_chain[:len(vrm_arm_names)]):
            if bone.name not in bone_mapping:
                bone_mapping[bone.name] = vrm_arm_names[i]
                print(f"  {bone.name} -> {vrm_arm_names[i]}")

if right_arm_start:
    # If shoulder is already mapped, start arm chain from its first child
    if right_arm_start.name in bone_mapping and bone_mapping[right_arm_start.name] == "rightShoulder":
        if len(right_arm_start.children) > 0:
            arm_chain_start = max(right_arm_start.children, key=lambda b: bone_length(b))
        else:
            arm_chain_start = None
    else:
        arm_chain_start = right_arm_start

    if arm_chain_start:
        # Follow right arm chain
        arm_chain = [arm_chain_start]
        current = arm_chain_start
        for _ in range(3):  # upperArm -> lowerArm -> hand (shoulder already mapped if present)
            if len(current.children) > 0:
                next_bone = max(current.children, key=lambda b: bone_length(b))
                arm_chain.append(next_bone)
                current = next_bone
            else:
                break

        # Map: upperArm -> lowerArm -> hand
        vrm_arm_names = ["rightUpperArm", "rightLowerArm", "rightHand"]
        for i, bone in enumerate(arm_chain[:len(vrm_arm_names)]):
            if bone.name not in bone_mapping:
                bone_mapping[bone.name] = vrm_arm_names[i]
                print(f"  {bone.name} -> {vrm_arm_names[i]}")

# Step 5: Find legs (left/right from hips)
# Legs go downward (negative Z or decreasing Z)
hips_bone = bone_info[hips_candidate]['bone']
leg_children = [child for child in hips_bone.children
                if child.name not in bone_mapping]

left_legs = []
right_legs = []

for child in leg_children:
    child_info = bone_info[child.name]
    child_center = child_info['center']
    child_head = child_info['head']
    child_tail = child_info['tail']

    # Legs go downward (tail Z < head Z, or negative Z)
    goes_downward = child_tail.z < child_head.z or child_tail.z < 0

    if goes_downward:
        # Determine left/right by X position
        if child_center.x < -0.01:  # Left side
            left_legs.append(child)
        elif child_center.x > 0.01:  # Right side
            right_legs.append(child)
        else:
            # If unclear, use the one that's more to the left/right
            if abs(child_center.x) > 0.001:
                if child_center.x < 0:
                    left_legs.append(child)
                else:
                    right_legs.append(child)

# Map legs
if len(left_legs) > 0:
    # Choose the leftmost leg
    left_upper_leg = min(left_legs, key=lambda b: bone_info[b.name]['center'].x)
    bone_mapping[left_upper_leg.name] = "leftUpperLeg"
    print(f"  {left_upper_leg.name} -> leftUpperLeg")

    # Follow leg chain: upperLeg -> lowerLeg -> foot -> toes
    leg_chain = [left_upper_leg]
    current = left_upper_leg
    for _ in range(3):
        if len(current.children) > 0:
            # Choose child that continues downward
            next_bone = max(current.children, key=lambda b: bone_length(b))
            leg_chain.append(next_bone)
            current = next_bone
        else:
            break

    vrm_leg_names = ["leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes"]
    for i, bone in enumerate(leg_chain[:len(vrm_leg_names)]):
        if bone.name not in bone_mapping:
            bone_mapping[bone.name] = vrm_leg_names[i]
            print(f"  {bone.name} -> {vrm_leg_names[i]}")

if len(right_legs) > 0:
    # Choose the rightmost leg
    right_upper_leg = max(right_legs, key=lambda b: bone_info[b.name]['center'].x)
    bone_mapping[right_upper_leg.name] = "rightUpperLeg"
    print(f"  {right_upper_leg.name} -> rightUpperLeg")

    # Follow leg chain
    leg_chain = [right_upper_leg]
    current = right_upper_leg
    for _ in range(3):
        if len(current.children) > 0:
            next_bone = max(current.children, key=lambda b: bone_length(b))
            leg_chain.append(next_bone)
            current = next_bone
        else:
            break

    vrm_leg_names = ["rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes"]
    for i, bone in enumerate(leg_chain[:len(vrm_leg_names)]):
        if bone.name not in bone_mapping:
            bone_mapping[bone.name] = vrm_leg_names[i]
            print(f"  {bone.name} -> {vrm_leg_names[i]}")

# Step 6: Map remaining bones (fingers, etc.) based on hierarchy and position
# Find hand bones and map fingers
for bone_name, vrm_name in list(bone_mapping.items()):
    if vrm_name in ["leftHand", "rightHand"]:
        hand_bone = bone_info[bone_name]['bone']
        side = "left" if "left" in vrm_name else "right"
        hand_center = bone_info[bone_name]['center']

        # Get unmapped finger children
        finger_children = [child for child in hand_bone.children
                          if child.name not in bone_mapping]

        if len(finger_children) == 0:
            continue

        # Identify thumb first (it's typically at a different angle/position)
        # Thumb is usually the one that's most separated from the other fingers
        thumb_bone = None
        if len(finger_children) >= 5:
            # Calculate average Y position of all fingers
            avg_y = sum(bone_info[child.name]['center'].y for child in finger_children) / len(finger_children)
            
            # Find the finger that's most different from the average
            # Thumb is typically at a different Y position (often higher for left hand, lower for right hand)
            thumb_candidates = []
            for child in finger_children:
                child_center = bone_info[child.name]['center']
                child_head = bone_info[child.name]['head']
                child_tail = bone_info[child.name]['tail']
                
                # Distance from average Y position
                y_diff_from_avg = abs(child_center.y - avg_y)
                
                # Angle from horizontal (thumb is typically more horizontal)
                finger_vec = child_tail - child_head
                horizontal_component = math.sqrt(finger_vec.x**2 + finger_vec.z**2)
                vertical_component = abs(finger_vec.y)
                angle_score = horizontal_component / (vertical_component + 0.001) if vertical_component > 0 else 100
                
                # For left hand, thumb is typically on the left (more negative X)
                # For right hand, thumb is typically on the right (more positive X)
                x_pos = child_center.x
                if side == "left":
                    x_score = -x_pos  # More negative = more likely thumb
                else:
                    x_score = x_pos  # More positive = more likely thumb
                
                # Combined score: thumb is the outlier
                # Higher Y difference from average + angle + X position
                score = y_diff_from_avg * 20.0 + angle_score * 0.3 + x_score * 3.0
                thumb_candidates.append((score, child))
            
            # The thumb is the one with highest score (most different)
            thumb_candidates.sort(key=lambda x: x[0], reverse=True)
            thumb_bone = thumb_candidates[0][1]
            finger_children.remove(thumb_bone)

        # Sort remaining fingers by Y position (vertical), then by X for tie-breaking
        # For right hand: index (highest Y) -> middle -> ring -> little (lowest Y)
        # For left hand: same order (index highest, little lowest)
        # When Y positions are very close, use X position to break ties
        remaining_fingers = sorted(finger_children,
                                  key=lambda b: (
                                      bone_info[b.name]['center'].y,  # Primary: Y position
                                      -bone_info[b.name]['center'].x if side == "left" 
                                      else bone_info[b.name]['center'].x  # Secondary: X position
                                  ),
                                  reverse=True)  # Highest Y first (index)

        # Map thumb first if found
        finger_mappings = []
        if thumb_bone:
            finger_mappings.append(("Thumb", thumb_bone))

        # Map remaining fingers: Index, Middle, Ring, Little
        finger_names = ["Index", "Middle", "Ring", "Little"]
        for i, finger_bone in enumerate(remaining_fingers[:len(finger_names)]):
            finger_mappings.append((finger_names[i], finger_bone))

        # Apply mappings
        for finger_name, finger_bone in finger_mappings:
            if finger_bone.name not in bone_mapping:
                finger_vrm_name = f"{side}{finger_name}"
                bone_mapping[finger_bone.name] = finger_vrm_name
                print(f"  {finger_bone.name} -> {finger_vrm_name}")

print(f"[OK] Geometric mapping complete: {len(bone_mapping)} bones mapped")

# Re-annotate images with VRM names
print("")
print("=== Re-annotating images with VRM bone names ===")
try:
    from PIL import Image, ImageDraw, ImageFont

    # Try to load fonts
    try:
        font_large = ImageFont.truetype("arial.ttf", 18)
    except:
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            font_large = ImageFont.load_default()

    # Re-annotate each view (update normal and base color maps with VRM names)
    for view_idx, view_name in enumerate(views):
        if view_name not in view_bone_data:
            continue

        bone_data_list = view_bone_data[view_name]

        # Update normal map
        normal_path = annotated_images_dir / f"bone_view_{view_name.lower()}_normal.png"
        if normal_path.exists():
            try:
                img_pil = Image.open(normal_path).convert("RGBA")
                draw = ImageDraw.Draw(img_pil)

                # Redraw labels with VRM names
                for bone_data in bone_data_list:
                    try:
                        bone = bone_data['bone']
                        foci = bone_data['foci']
                        label_x = int(bone_data['label_x'])
                        label_y = int(bone_data['label_y'])

                        # Get VRM name from mapping, or keep original name
                        display_name = bone_mapping.get(bone.name, bone.name)

                        # Get updated text size
                        try:
                            bbox = draw.textbbox((0, 0), display_name, font=font_large)
                            new_text_width = bbox[2] - bbox[0]
                            new_text_height = bbox[3] - bbox[1]
                        except:
                            new_text_width = len(display_name) * 10
                            new_text_height = 18

                        # Draw connecting line from foci to label
                        label_center_y = label_y + new_text_height // 2
                        draw.line([foci, (label_x, label_center_y)],
                                 fill=(255, 255, 255, 120), width=1)

                        # Draw foci point
                        foci_radius = 4
                        draw.ellipse([foci[0]-foci_radius, foci[1]-foci_radius,
                                     foci[0]+foci_radius, foci[1]+foci_radius],
                                    fill=(255, 255, 0, 180), outline=(255, 255, 255, 150), width=1)

                        # Add text label with VRM name (translucent)
                        draw.text((label_x, label_y), display_name,
                                 fill=(255, 255, 255, 200), font=font_large)

                    except Exception as e:
                        continue

                # Save updated normal map with VRM names
                img_pil.save(normal_path)
                print(f"  Re-annotated {view_name} normal map with VRM names")

            except Exception as e:
                print(f"  [WARN] Error re-annotating {view_name} normal map: {e}")

        # Update base color map
        basecolor_path = annotated_images_dir / f"bone_view_{view_name.lower()}_basecolor.png"
        if basecolor_path.exists():
            try:
                img_pil = Image.open(basecolor_path).convert("RGBA")
                draw = ImageDraw.Draw(img_pil)

                # Redraw labels with VRM names
                for bone_data in bone_data_list:
                    try:
                        bone = bone_data['bone']
                        foci = bone_data['foci']
                        label_x = int(bone_data['label_x'])
                        label_y = int(bone_data['label_y'])

                        # Get VRM name from mapping, or keep original name
                        display_name = bone_mapping.get(bone.name, bone.name)

                        # Get updated text size
                        try:
                            bbox = draw.textbbox((0, 0), display_name, font=font_large)
                            new_text_width = bbox[2] - bbox[0]
                            new_text_height = bbox[3] - bbox[1]
                        except:
                            new_text_width = len(display_name) * 10
                            new_text_height = 18

                        # Draw connecting line from foci to label
                        label_center_y = label_y + new_text_height // 2
                        draw.line([foci, (label_x, label_center_y)],
                                 fill=(255, 255, 255, 120), width=1)

                        # Draw foci point
                        foci_radius = 4
                        draw.ellipse([foci[0]-foci_radius, foci[1]-foci_radius,
                                     foci[0]+foci_radius, foci[1]+foci_radius],
                                    fill=(255, 255, 0, 180), outline=(255, 255, 255, 150), width=1)

                        # Add text label with VRM name (translucent)
                        draw.text((label_x, label_y), display_name,
                                 fill=(255, 255, 255, 200), font=font_large)

                    except Exception as e:
                        continue

                # Save updated base color map with VRM names
                img_pil.save(basecolor_path)
                print(f"  Re-annotated {view_name} base color map with VRM names")

            except Exception as e:
                print(f"  [WARN] Error re-annotating {view_name} base color map: {e}")

    print("[OK] Images re-annotated with geometric mapping results")

except Exception as e:
    print(f"[WARN] Error during re-annotation: {e}")
    import traceback
    traceback.print_exc()

# Create single mapping result (no voting needed for geometric approach)
all_mappings = [bone_mapping]  # For compatibility with existing code

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
duplicates = []
if len(mapped_bones) != len(bone_mapping.values()):
    print("[WARN] Duplicate VRM bone mappings detected")
    # Find duplicates
    from collections import Counter
    counts = Counter(bone_mapping.values())
    duplicates = [bone for bone, count in counts.items() if count > 1]
    print(f"  Duplicates: {duplicates}")

print("[OK] VRM compliance check completed")

# Save geometric mapping results
print("")
print("=== Saving Analysis Results ===")
geometric_results_file = output_dir / "geometric_mapping_results.json"
geometric_results = {
    "method": "geometric_decision_tree",
    "mapping": bone_mapping,
    "bone_info": {name: {
        "head": info['head_local'],
        "tail": info['tail_local'],
        "length": info['length'],
        "depth": info['depth'],
        "parent": info['parent'],
        "children": info['children'],
        "is_left": info['is_left'],
        "is_right": info['is_right'],
        "vertical_pos": info['vertical_pos'],
        "is_upper_body": info['is_upper_body']
    } for name, info in bone_info.items()}
}
with open(geometric_results_file, 'w', encoding='utf-8') as f:
    json.dump(geometric_results, f, indent=2, ensure_ascii=False)
print(f"  Geometric mapping results saved to: {geometric_results_file}")

# Save mapping for reference
all_iterations_file = output_dir / "geometric_mapping.json"
with open(all_iterations_file, 'w', encoding='utf-8') as f:
    json.dump(bone_mapping, f, indent=2, ensure_ascii=False)
print(f"  Geometric mapping saved to: {all_iterations_file}")

# Save validation results
validation_results = {
    "required_bones": required_bones,
    "mapped_bones": list(mapped_bones),
    "missing_required": missing_required,
    "duplicate_mappings": duplicates,
    "total_bones_mapped": len(bone_mapping),
    "total_required_bones": len(required_bones),
    "compliance_percentage": (len(required_bones) - len(missing_required)) / len(required_bones) * 100 if len(required_bones) > 0 else 0.0
}
validation_file = output_dir / "validation_results.json"
with open(validation_file, 'w', encoding='utf-8') as f:
    json.dump(validation_results, f, indent=2, ensure_ascii=False)
print(f"  Validation results saved to: {validation_file}")

# Save comprehensive analysis summary
analysis_summary = {
    "input_file": input_path,
    "output_file": str(output_path),
    "total_bones_extracted": len(bone_data),
    "total_bones_mapped": len(bone_mapping),
    "mapping_method": "geometric_decision_tree",
    "final_mapping": bone_mapping,
    "validation": validation_results,
    "views_captured": len(views),
    "annotated_images_dir": str(annotated_images_dir),
    "image_types": {
        "normal_maps": f"{len(views)} views (with translucent labels)",
        "basecolor_maps": f"{len(pbr_images['basecolor'])} views (with translucent labels)"
    },
    "gltf_hierarchy_available": gltf_json_hierarchy is not None
}
summary_file = output_dir / "analysis_summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
print(f"  Analysis summary saved to: {summary_file}")

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
renaming_log_entries = []

for bone in armature.data.edit_bones:
    if bone.name in bone_mapping:
        old_name = bone.name
        new_name = bone_mapping[bone.name]

        # Check if target name already exists
        if new_name in [b.name for b in armature.data.edit_bones if b != bone]:
            print(f"  [WARN] Skipping {old_name} -> {new_name} (target name already exists)")
            skipped_count += 1
            renaming_log_entries.append({
                "original_name": old_name,
                "new_name": new_name,
                "status": "skipped",
                "reason": "target name already exists"
            })
            continue

        try:
            bone.name = new_name
            print(f"  {old_name} -> {new_name}")
            renamed_count += 1
            renaming_log_entries.append({
                "original_name": old_name,
                "new_name": new_name,
                "status": "renamed",
                "vrm_spec": new_name
            })
        except Exception as e:
            error_msg = f"  [ERROR] Error renaming {old_name} -> {new_name}"
            print(error_msg + ": " + str(e))
            skipped_count += 1
            renaming_log_entries.append({
                "original_name": old_name,
                "new_name": new_name,
                "status": "error",
                "error": str(e)
            })

# Switch back to object mode
bpy.ops.object.mode_set(mode='OBJECT')
print(f"[OK] Renamed {renamed_count} bones")
if skipped_count > 0:
    print(f"[WARN] Skipped {skipped_count} bones (conflicts or errors)")

# Save renaming log
renaming_log = {
    "total_bones_in_mapping": len(bone_mapping),
    "successfully_renamed": renamed_count,
    "skipped": skipped_count,
    "renamed_bones": renaming_log_entries
}

renaming_log_file = output_dir / "renaming_log.json"
with open(renaming_log_file, 'w', encoding='utf-8') as f:
    json.dump(renaming_log, f, indent=2, ensure_ascii=False)
print(f"  Renaming log saved to: {renaming_log_file}")

# Export to USDC
print("")
print("=== Exporting to USDC ===")
output_path = config.get('output_path', str(Path(input_path).with_suffix('.usdc')))
output_path = str(Path(output_path)).replace("\\\\", "/")

# Ensure output directory exists (output_dir already defined from config above)
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
        export_armatures=True,
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
print(f"")
print("=== Analysis Files Saved ===")
print(f"All analysis results saved to: {output_dir}")
print(f"  - Extracted bone data: extracted_bone_data.json")
print(f"  - GLTF hierarchy: gltf_hierarchy.json")
print(f"  - Final bone mapping: bone_mapping_final.json")
print(f"  - Geometric mapping results: geometric_mapping_results.json")
print(f"  - Geometric mapping: geometric_mapping.json")
print(f"  - Validation results: validation_results.json")
print(f"  - Analysis summary: analysis_summary.json")
print(f"  - Renaming log: renaming_log.json")
print(f"  - Normal maps: {annotated_images_dir} (with translucent labels)")
print(f"  - Base color maps: {annotated_images_dir} (with translucent labels)")
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
