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

# Generate output path (replace extension with .usdc)
input_dir = Path.dirname(absolute_input_path)
input_basename = Path.basename(absolute_input_path, Path.extname(absolute_input_path))
output_path = Path.join(input_dir, "#{input_basename}_vrm.usdc") |> String.replace("\\", "/")

config = Map.put(config, :input_path, absolute_input_path)
config = Map.put(config, :output_path, output_path)

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

# Create annotated visualizations
print("")
print("=== Creating Annotated Visualizations ===")

# Set up viewport to show armature
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='POSE')

# Hide mesh objects to show only armature
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.hide_set(True)

# Set up armature display
armature.data.display_type = 'STICK'
armature.show_in_front = True

# Create temp directory for screenshots
temp_dir = Path(tempfile.gettempdir()) / f"vrm_bone_renamer_{os.getpid()}"
temp_dir.mkdir(exist_ok=True, parents=True)

# Capture screenshots from different camera angles (4 views using Fibonacci sphere)
# This maximizes viewing angle coverage using golden angle spiral distribution
# Based on HKU/SAMPart3D camera trajectory methods
# More views provide better coverage for vision analysis
import math
num_views = 4
views = [f'VIEW_{i}' for i in range(num_views)]
annotated_images = []

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

# Calculate bounding box center for camera positioning
import mathutils
all_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE' or obj.type == 'MESH']
if all_objects:
    # Get bounding box center
    bbox_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    for obj in all_objects:
        for vertex in obj.bound_box:
            world_vertex = obj.matrix_world @ mathutils.Vector(vertex)
            bbox_min = mathutils.Vector((min(bbox_min.x, world_vertex.x), min(bbox_min.y, world_vertex.y), min(bbox_min.z, world_vertex.z)))
            bbox_max = mathutils.Vector((max(bbox_max.x, world_vertex.x), max(bbox_max.y, world_vertex.y), max(bbox_max.z, world_vertex.z)))
    center = (bbox_min + bbox_max) / 2
    size = bbox_max - bbox_min
    distance = max(size.x, size.y, size.z) * 1.5
else:
    center = mathutils.Vector((0, 0, 0))
    size = mathutils.Vector((10, 10, 10))  # Default size
    distance = 5

# Fibonacci sphere (golden angle spiral) for evenly distributed views
# This maximizes viewing angle coverage - commonly used in HKU/SAMPart3D papers
def fibonacci_sphere(n, offset=0.5):
    \"""
    Generate n points evenly distributed on a sphere using Fibonacci spiral.
    Returns list of (x, y, z) unit vectors.
    \"""
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

    # Get point on unit sphere and scale by distance
    x, y, z = sphere_points[i]
    camera_pos = mathutils.Vector((x * distance, y * distance, z * distance))

    # Position camera
    camera.location = center + camera_pos

    # Make camera look at center
    look_at(camera, center)

    # Use perspective camera
    camera.data.type = 'PERSP'
    camera.data.angle = math.radians(50.0)  # Field of view

    # Step 1: Render normal map (with mesh visible, armature hidden)
    normal_map_path = temp_dir / f"normal_map_{view_name.lower()}.png"
    normal_map_rendered = False
    try:
        # Show mesh objects for normal map rendering
        mesh_objects = []
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

    except Exception as e:
        print(f"    Error rendering normal map: {e}")
        import traceback
        traceback.print_exc()

    # Step 2: Render bone visualization (with armature visible, mesh hidden)
    bone_vis_path = temp_dir / f"bone_vis_{view_name.lower()}.png"
    bone_vis_rendered = False

    # Restore: hide mesh, show armature, disable compositor
    try:
        for obj in mesh_objects:
            obj.hide_set(True)
        armature.hide_set(False)
        scene.use_nodes = False
    except:
        pass

    try:
        scene.render.filepath = str(bone_vis_path)
        # Render using Eevee (works in headless mode)
        bpy.ops.render.render(write_still=True)
        print(f"    Bone visualization rendered: {bone_vis_path}")
        bone_vis_rendered = True
    except Exception as e:
        print(f"    Error rendering bone visualization: {e}")
        import traceback
        traceback.print_exc()

    # Step 3: Composite normal map (background) with bone visualization (foreground)
    screenshot_path = temp_dir / f"bone_view_{view_name.lower()}.png"
    try:
        from PIL import Image

        if normal_map_rendered and bone_vis_rendered:
            # Load both images
            normal_img = Image.open(normal_map_path).convert("RGBA")
            bone_img = Image.open(bone_vis_path).convert("RGBA")

            # Composite: normal map as background, bone visualization on top
            composite = Image.alpha_composite(normal_img, bone_img)
            composite.save(screenshot_path)
            print(f"    Composite saved: {screenshot_path}")
        elif bone_vis_rendered:
            # Fallback: use only bone visualization if normal map failed
            bone_img = Image.open(bone_vis_path).convert("RGBA")
            # Create white background
            composite = Image.new("RGBA", bone_img.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(composite, bone_img)
            composite.save(screenshot_path)
            print(f"    Composite saved (bone only): {screenshot_path}")
        elif normal_map_rendered:
            # Fallback: use only normal map if bone visualization failed
            normal_img = Image.open(normal_map_path).convert("RGBA")
            normal_img.save(screenshot_path)
            print(f"    Composite saved (normal map only): {screenshot_path}")
        else:
            print(f"    Error: Both renders failed, skipping composite")
            continue

    except Exception as e:
        print(f"    Error compositing images: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Annotate with bone labels and foci (reuse UniRig pattern)
    try:
        from PIL import Image, ImageDraw, ImageFont

        img_pil = Image.open(screenshot_path).convert("RGBA")
        draw = ImageDraw.Draw(img_pil)

        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except:
                font = ImageFont.load_default()

        res_x, res_y = scene.render.resolution_x, scene.render.resolution_y

        # Store bone positions for connecting dots
        bone_positions = {}

        # First pass: collect all bone positions
        for bone in armature.data.bones:
            try:
                # Project 3D bone positions to 2D screen space
                # Convert local to world coordinates
                bone_matrix = armature.matrix_world @ bone.matrix_local
                head_world = bone_matrix @ Vector(bone.head_local)
                tail_world = bone_matrix @ Vector(bone.tail_local)

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

        # Third pass: Draw bones and roll axes
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

                # Only draw if within image bounds
                if (0 <= head_pix[0] < res_x and 0 <= head_pix[1] < res_y and
                    0 <= tail_pix[0] < res_x and 0 <= tail_pix[1] < res_y):

                    # Draw bone line (red) - from head to tail
                    draw.line([head_pix, tail_pix], fill=(255, 0, 0, 255), width=3)

                    # Calculate all 3 axes of bone's local coordinate system
                    # Bone coordinate system origin is at the bone head
                    bone_matrix_3x3 = bone_matrix.to_3x3()
                    # Get X, Y, Z axes from the rotation matrix
                    x_axis_world = bone_matrix_3x3 @ Vector((1, 0, 0))
                    y_axis_world = bone_matrix_3x3 @ Vector((0, 1, 0))
                    z_axis_world = bone_matrix_3x3 @ Vector((0, 0, 1))

                    # Scale the axes for visibility (30% of bone length)
                    axis_length = (tail_world - head_world).length * 0.3

                    # Helper function to draw an axis from bone head
                    def draw_axis(axis_world, color, label):
                        axis_end_world = head_world + axis_world * axis_length
                        axis_end_2d = bpy_extras.object_utils.world_to_camera_view(
                            scene, camera, axis_end_world)
                        axis_end_pix = (int(axis_end_2d.x * res_x), int((1 - axis_end_2d.y) * res_y))

                        if (0 <= axis_end_pix[0] < res_x and 0 <= axis_end_pix[1] < res_y and
                            0 <= head_pix[0] < res_x and 0 <= head_pix[1] < res_y):
                            # Draw axis line from bone head
                            draw.line([head_pix, axis_end_pix], fill=color, width=2)

                            # Draw small arrow at end
                            arrow_size = 5
                            dx = axis_end_pix[0] - head_pix[0]
                            dy = axis_end_pix[1] - head_pix[1]
                            arrow_angle = math.atan2(dy, dx)
                            arrow1 = (
                                int(axis_end_pix[0] - arrow_size * math.cos(arrow_angle - math.pi/6)),
                                int(axis_end_pix[1] - arrow_size * math.sin(arrow_angle - math.pi/6))
                            )
                            arrow2 = (
                                int(axis_end_pix[0] - arrow_size * math.cos(arrow_angle + math.pi/6)),
                                int(axis_end_pix[1] - arrow_size * math.sin(arrow_angle + math.pi/6))
                            )
                            draw.line([axis_end_pix, arrow1], fill=color, width=2)
                            draw.line([axis_end_pix, arrow2], fill=color, width=2)

                            # Draw axis label
                            draw.text((axis_end_pix[0]+5, axis_end_pix[1]-5), label,
                                     fill=color, font=font)

                    # Draw X-axis (red)
                    draw_axis(x_axis_world, (255, 0, 0, 255), 'X')
                    # Draw Y-axis (green) - roll axis
                    draw_axis(y_axis_world, (0, 255, 0, 255), 'Y')
                    # Draw Z-axis (blue)
                    draw_axis(z_axis_world, (0, 0, 255, 255), 'Z')

                    # Draw numbered marker circle (foci) at bone head (green)
                    marker_radius = 8
                    draw.ellipse([head_pix[0]-marker_radius, head_pix[1]-marker_radius,
                                 head_pix[0]+marker_radius, head_pix[1]+marker_radius],
                                fill=(0, 255, 0, 200), outline=(255, 255, 0, 255), width=2)

                    # Add text label (yellow)
                    draw.text((head_pix[0]+10, head_pix[1]-10), bone.name,
                             fill=(255, 255, 0, 255), font=font)
            except Exception as e:
                # Skip bones that can't be projected
                continue

        img_pil.save(screenshot_path)
        annotated_images.append(str(screenshot_path))
        print(f"    Annotated image saved: {screenshot_path}")
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
vrm_spec_prompt = '''
Analyze the annotated bone structure images and identify VRM bone mappings.

VRM Humanoid Bone Specification (VRMC_vrm-1.0):

REQUIRED BONES (must be mapped):
- Torso: hips, spine
- Head: head
- Legs: leftUpperLeg, leftLowerLeg, leftFoot, rightUpperLeg, rightLowerLeg, rightFoot
- Arms: leftUpperArm, leftLowerArm, leftHand, rightUpperArm, rightLowerArm, rightHand

OPTIONAL BONES (map if present):
- Torso: chest, upperChest, neck
- Head: leftEye, rightEye, jaw
- Legs: leftToes, rightToes
- Arms: leftShoulder, rightShoulder
- Fingers: left/right thumb, index, middle, ring, little (proximal, intermediate, distal)

PARENT-CHILD RELATIONSHIPS:
- hips (root) → spine → chest → upperChest → neck → head
- upperChest → leftShoulder → leftUpperArm → leftLowerArm → leftHand → fingers
- upperChest → rightShoulder → rightUpperArm → rightLowerArm → rightHand → fingers
- hips → leftUpperLeg → leftLowerLeg → leftFoot → leftToes
- hips → rightUpperLeg → rightLowerLeg → rightFoot → rightToes

ESTIMATED POSITIONS:
- hips: Crotch area
- spine: Top of pelvis
- chest: Bottom of rib cage
- neck: Base of neck
- head: Top of neck
- leftUpperLeg/rightUpperLeg: Groin area
- leftLowerLeg/rightLowerLeg: Knee area
- leftFoot/rightFoot: Ankle area
- leftUpperArm/rightUpperArm: Base of upper arm
- leftLowerArm/rightLowerArm: Elbow area
- leftHand/rightHand: Wrist area

The images show bone structures with:
- Red lines: bone connections
- Green circles: bone head markers (foci)
- Yellow text: bone names (bone_0, bone_1, etc.)

Based on the bone positions, structure, and parent-child relationships visible in the images, identify which numbered bone (bone_0, bone_1, etc.) corresponds to which VRM bone name.

Return ONLY a valid JSON object mapping bone names to VRM names, in this exact format:
{"bone_0": "hips", "bone_1": "spine", "bone_2": "chest", ...}

Do not include any explanation or text outside the JSON object.
'''

# Prepare images for Qwen3VL
print("Preparing images for Qwen3VL analysis...")
images = []
for img_path in annotated_images:
    img = Image.open(img_path).convert("RGB")
    images.append(img)

print(f"  Loaded {len(images)} images")

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
try:
    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

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

except Exception as e:
    print(f"[ERROR] Error during Qwen3VL inference: {e}")
    import traceback
    traceback.print_exc()
    raise

# Parse JSON from response
print("=== Parsing Bone Mappings ===")
try:
    # Extract JSON from response (may have extra text)
    import re

    # Try to find JSON object - look for opening and closing braces
    json_start = response.find('{')
    if json_start < 0:
        raise ValueError("No JSON object found in response (no opening brace)")

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
        raise ValueError("No complete JSON object found in response (unmatched braces)")

    json_str = response[json_start:json_end+1]

    # Try to parse JSON
    bone_mapping = json.loads(json_str)

    # Validate it's a dictionary with string keys and values
    if not isinstance(bone_mapping, dict):
        raise ValueError(f"Expected dictionary, got {type(bone_mapping)}")

    # Validate all keys are bone names and values are strings
    for key, value in bone_mapping.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Invalid mapping entry: {key} -> {value}")

    print(f"[OK] Parsed {len(bone_mapping)} bone mappings")
    for old_name, new_name in sorted(bone_mapping.items()):
        print(f"  {old_name} -> {new_name}")

except Exception as e:
    print(f"[ERROR] Error parsing JSON from Qwen3VL response: {e}")
    print(f"Response was: {response}")
    import traceback
    traceback.print_exc()
    raise

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
        export_colors=True,
        default_prim_path="",
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

# Cleanup temp files
try:
    import shutil
    shutil.rmtree(temp_dir)
except:
    pass
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
