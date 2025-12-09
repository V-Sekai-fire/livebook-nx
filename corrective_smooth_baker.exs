#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Corrective Smooth Baker Script
# Improves deformation quality by baking Corrective Smooth modifier effects into vertex weights
# Based on: https://superhivemarket.com/products/corrective-smooth-baker
#
# Usage:
#   elixir corrective_smooth_baker.exs <rigged_model> [options]
#
# Supported Input Formats: GLB, GLTF, USD (usd, usda, usdc), FBX
# Supported Output Format: USDC (binary only)
#
# Options:
#   --output, -o <path>              Output USDC file path (default: input file with _corrected.usdc suffix)
#   --bake-range <mode>              Bake range: All, Selected, Deviation (default: Deviation)
#   --deviation-threshold <float>    Deviation threshold in cm (default: 0.01)
#   --bake-quality <quality>         Bake quality: 0.5, 0.75, 1.0, 2.0, 3.0 (default: 1.0)
#   --twist-angle <degrees>          Maximum twist angle (default: 45.0)
#   --influence-bones <count>        Max influence bones per vertex (default: 4)
#   --prune-threshold <float>        Prune threshold (default: 0.01)
#   --solver <solver>                Linear system solver: STD, CHOLESKY, QR, INV, PINV, LSTSQ, SVD (default: SVD)
#   --refresh-frequency <fps>        Refresh frequency (default: 15.0)
#   --help, -h                       Show this help message

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"}
])

# Initialize Python environment with required dependencies
Pythonx.uv_init("""
[project]
name = "corrective-smooth-baker"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "bpy==4.5.*",
  "numpy",
]
""")

# Parse command-line arguments
defmodule ArgsParser do
  def show_help do
    IO.puts("""
    Corrective Smooth Baker
    Improves deformation quality by baking Corrective Smooth modifier effects into vertex weights
    Based on: https://superhivemarket.com/products/corrective-smooth-baker

    Usage:
      elixir corrective_smooth_baker.exs <rigged_model> [options]

    Supported Input Formats: GLB, GLTF, USD (usd, usda, usdc), FBX
    Supported Output Format: USDC (binary only)

    The script will:
    1. Load the rigged model (must have armature and mesh)
    2. Add a Corrective Smooth modifier to the mesh
    3. Bake the modifier effects into vertex weights
    4. Remove the modifier
    5. Export the corrected model

    Options:
      --output, -o <path>              Output USDC file path (default: input file with _corrected.usdc suffix)
      --bake-range <mode>              Bake range: All, Selected, Deviation (default: Deviation)
      --deviation-threshold <float>    Deviation threshold in cm (default: 0.01)
      --bake-quality <quality>         Bake quality: 0.5, 0.75, 1.0, 2.0, 3.0 (default: 1.0)
      --twist-angle <degrees>          Maximum twist angle (default: 45.0)
      --influence-bones <count>        Max influence bones per vertex (default: 4)
      --prune-threshold <float>        Prune threshold (default: 0.01)
      --solver <solver>                Linear system solver: STD, CHOLESKY, QR, INV, PINV, LSTSQ, SVD (default: SVD)
      --refresh-frequency <fps>        Refresh frequency (default: 15.0)
      --help, -h                       Show this help message

    Example:
      elixir corrective_smooth_baker.exs rigged.usdc
      elixir corrective_smooth_baker.exs rigged.usdc --bake-quality 2.0 --output corrected.usdc
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        output: :string,
        bake_range: :string,
        deviation_threshold: :float,
        bake_quality: :string,
        twist_angle: :float,
        influence_bones: :integer,
        prune_threshold: :float,
        solver: :string,
        refresh_frequency: :float,
        help: :boolean
      ],
      aliases: [
        o: :output,
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
        elixir corrective_smooth_baker.exs <rigged_model> [options]

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

    # Generate output path if not provided
    output_path = Keyword.get(opts, :output)
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
      # Generate default output path
      input_dir = Path.dirname(input_path)
      input_stem = Path.basename(input_path, input_ext)
      Path.join(input_dir, "#{input_stem}_corrected.usdc")
    end

    # Check if input file exists
    if !File.exists?(input_path) do
      IO.puts("Error: Input file not found: #{input_path}")
      System.halt(1)
    end

    # Validate bake range
    bake_range = Keyword.get(opts, :bake_range, "Deviation")
    valid_bake_ranges = ["All", "Selected", "Deviation"]
    if !Enum.member?(valid_bake_ranges, bake_range) do
      IO.puts("Error: Invalid bake range: #{bake_range}")
      IO.puts("Valid options: #{Enum.join(valid_bake_ranges, ", ")}")
      System.halt(1)
    end

    # Validate bake quality
    bake_quality = Keyword.get(opts, :bake_quality, "1.0")
    valid_bake_qualities = ["0.5", "0.75", "1.0", "2.0", "3.0"]
    if !Enum.member?(valid_bake_qualities, bake_quality) do
      IO.puts("Error: Invalid bake quality: #{bake_quality}")
      IO.puts("Valid options: #{Enum.join(valid_bake_qualities, ", ")}")
      System.halt(1)
    end

    # Validate solver
    solver = Keyword.get(opts, :solver, "SVD")
    valid_solvers = ["STD", "CHOLESKY", "QR", "INV", "PINV", "LSTSQ", "SVD"]
    if !Enum.member?(valid_solvers, solver) do
      IO.puts("Error: Invalid solver: #{solver}")
      IO.puts("Valid options: #{Enum.join(valid_solvers, ", ")}")
      System.halt(1)
    end

    deviation_threshold = Keyword.get(opts, :deviation_threshold, 0.01)
    if deviation_threshold < 0.0 do
      IO.puts("Error: deviation_threshold must be non-negative")
      System.halt(1)
    end

    twist_angle = Keyword.get(opts, :twist_angle, 45.0)
    if twist_angle < 0.0 do
      IO.puts("Error: twist_angle must be non-negative")
      System.halt(1)
    end

    influence_bones = Keyword.get(opts, :influence_bones, 4)
    if influence_bones < 1 do
      IO.puts("Error: influence_bones must be at least 1")
      System.halt(1)
    end

    prune_threshold = Keyword.get(opts, :prune_threshold, 0.01)
    if prune_threshold < 0.0 or prune_threshold > 1.0 do
      IO.puts("Error: prune_threshold must be between 0.0 and 1.0")
      System.halt(1)
    end

    refresh_frequency = Keyword.get(opts, :refresh_frequency, 15.0)
    if refresh_frequency <= 0.0 do
      IO.puts("Error: refresh_frequency must be greater than 0.0")
      System.halt(1)
    end

    %{
      input_path: input_path,
      output_path: output_path,
      workspace_root: File.cwd!(),
      bake_range: bake_range,
      deviation_threshold: deviation_threshold,
      bake_quality: bake_quality,
      twist_angle: twist_angle,
      influence_bones: influence_bones,
      prune_threshold: prune_threshold,
      solver: solver,
      refresh_frequency: refresh_frequency
    }
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== Corrective Smooth Baker ===
Input: #{config.input_path}
Output: #{config.output_path}
Bake Range: #{config.bake_range}
Deviation Threshold: #{config.deviation_threshold} cm
Bake Quality: #{config.bake_quality}
Twist Angle: #{config.twist_angle}°
Influence Bones: #{config.influence_bones}
Prune Threshold: #{config.prune_threshold}
Solver: #{config.solver}
Refresh Frequency: #{config.refresh_frequency} fps

Note: This process may take a long time depending on mesh complexity and bake quality.
Press ESC in Blender to cancel if needed.
""")

# Convert JSON config to string for Python (use temp file to avoid conflicts)
config_json = Jason.encode!(config)
# Use cross-platform temp directory
tmp_dir = System.tmp_dir!()
File.mkdir_p!(tmp_dir)
config_file = Path.join(tmp_dir, "corrective_smooth_config_#{System.system_time(:millisecond)}.json")
File.write!(config_file, config_json)
config_file_normalized = String.replace(config_file, "\\", "/")

# Run corrective smooth baking
try do
  {_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
from pathlib import Path

# Load config
""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
""" <> ~S"""

input_path = config['input_path']
output_path = config['output_path']
workspace_root = config.get('workspace_root', os.getcwd())

# Convert paths to absolute paths
if not os.path.isabs(input_path):
    input_path = os.path.abspath(os.path.join(workspace_root, input_path))
if not os.path.isabs(output_path):
    output_path = os.path.abspath(os.path.join(workspace_root, output_path))

# Normalize path separators for Windows compatibility
input_path = os.path.normpath(input_path)
output_path = os.path.normpath(output_path)

# Add corrective smooth baker to path
corrective_smooth_baker_path = os.path.join(workspace_root, 'thirdparty', 'corrective_smooth_baker')
if os.path.exists(corrective_smooth_baker_path):
    sys.path.insert(0, os.path.join(workspace_root, 'thirdparty'))
else:
    raise ImportError(f"Could not find corrective_smooth_baker at {corrective_smooth_baker_path}")

# Import Blender and corrective smooth baker
import bpy
import math
from corrective_smooth_baker.corrective_smooth_baker import (
    register_corrective_smooth_baker,
    CSB_OT_ModalTimerOperator
)

# Clean Blender scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Load input file
print(f"Loading: {input_path}")
if input_path.endswith('.usd') or input_path.endswith('.usda') or input_path.endswith('.usdc'):
    bpy.ops.wm.usd_import(filepath=input_path)
elif input_path.endswith('.glb') or input_path.endswith('.gltf'):
    bpy.ops.import_scene.gltf(filepath=input_path)
elif input_path.endswith('.fbx'):
    bpy.ops.import_scene.fbx(filepath=input_path)
else:
    raise ValueError(f"Unsupported format: {input_path}")

# Find armature and mesh objects
armature_objects = [ob for ob in bpy.context.scene.objects if ob.type == 'ARMATURE']
mesh_objects = [ob for ob in bpy.context.scene.objects if ob.type == 'MESH']

if not armature_objects:
    raise ValueError("No armature found in the model. Please provide a rigged model with an armature.")
if not mesh_objects:
    raise ValueError("No mesh found in the model. Please provide a rigged model with a mesh.")

armature = armature_objects[0]
print(f"Found armature: {armature.name}")
print(f"Found {len(mesh_objects)} mesh object(s)")

# Register corrective smooth baker
register_corrective_smooth_baker()

# Add Corrective Smooth modifier to each mesh and prepare for baking
for mesh_obj in mesh_objects:
    # Check if mesh is connected to armature
    has_armature_modifier = any(mod.type == 'ARMATURE' for mod in mesh_obj.modifiers)
    is_parented = mesh_obj.parent and mesh_obj.parent.type == 'ARMATURE'

    if not (has_armature_modifier or is_parented):
        print(f"Warning: Mesh {mesh_obj.name} is not connected to armature, skipping")
        continue

    # Add Corrective Smooth modifier if it doesn't exist
    corrective_smooth_mod = None
    for mod in mesh_obj.modifiers:
        if mod.type == 'CORRECTIVE_SMOOTH':
            corrective_smooth_mod = mod
            break

    if corrective_smooth_mod is None:
        corrective_smooth_mod = mesh_obj.modifiers.new(name="CorrectiveSmooth", type='CORRECTIVE_SMOOTH')
        # Use default settings (typically sufficient per documentation)
        corrective_smooth_mod.factor = 1.0
        corrective_smooth_mod.iterations = 5
        corrective_smooth_mod.smooth_type = 'LENGTH_WEIGHTED'
        print(f"Added Corrective Smooth modifier to {mesh_obj.name}")
    else:
        print(f"Found existing Corrective Smooth modifier on {mesh_obj.name}")

# Set baking parameters
bpy.context.scene.bake_range = config['bake_range']
bpy.context.scene.deviation_threshold = config['deviation_threshold']
bpy.context.scene.bake_quality = config['bake_quality']
bpy.context.scene.twist_angle = config['twist_angle']
bpy.context.scene.influence_bones = config['influence_bones']
bpy.context.scene.prune_threshold = config['prune_threshold']
bpy.context.scene.linear_system_solver = config['solver']
bpy.context.scene.refresh_frequency = config['refresh_frequency']

# Select armature and all meshes
bpy.ops.object.select_all(action='DESELECT')
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
for mesh_obj in mesh_objects:
    mesh_obj.select_set(True)

# Ensure we're in OBJECT mode
bpy.ops.object.mode_set(mode='OBJECT')

# Run the corrective smooth baker
print("\nStarting corrective smooth baking...")
print("This may take a long time. Press ESC in Blender to cancel if needed.")
print("Progress will be shown in Blender's status bar.")

# Create a wrapper class to run baking stages programmatically
# We can't instantiate Blender operators directly, so we'll create a simple wrapper
class BakingRunner:
    def __init__(self, arm, objs):
        self._arm = arm
        self._objs = objs
        self._should_terminate = False
        self._inverse_bind_pose_dic = None
        self._static_positions = None
        self._sharp_vertex_index_dics = None
        self._transform_matrix_dics = None
        self._dynamic_positions = None
        self._mesh_index = 0
        self._pose_index = 0
        self._vertex_index = 0
        self._optimize_stage_index = 0
        self._current_pose_dic = None
        self._transform_matrix_stack = None

    def save_current_poses(self):
        self._current_pose_dic = {}
        for bone in self._arm.pose.bones:
            self._current_pose_dic[bone] = copy.deepcopy(bone.matrix_basis)

    def restore_current_poses(self):
        for bone in self._arm.pose.bones:
            bone.matrix_basis = self._current_pose_dic[bone]

    def reset_armature(self, ob):
        ob.hide_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.mode_set(mode='POSE', toggle=False)
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        bpy.ops.object.mode_set(mode='OBJECT')

    def twist_armature(self, ob, twist_angle):
        bpy.ops.object.mode_set(mode='POSE', toggle=False)
        for bone in ob.data.bones:
            pose_bone = ob.pose.bones[bone.name]
            save_rotation_mode = pose_bone.rotation_mode
            pose_bone.rotation_mode = 'XYZ'
            axis = random.choice(['X', 'Y', 'Z'])
            angle = random.uniform(-twist_angle, twist_angle)
            pose_bone.rotation_euler.rotate_axis(axis, math.radians(angle))
            pose_bone.rotation_mode = save_rotation_mode
        bpy.ops.object.mode_set(mode='OBJECT')

    def get_inverse_bind_pose_dic(self, ob):
        pose_dic = {}
        for bone in ob.data.bones:
            if bone.use_deform:
                pose_dic[bone.name] = (ob.matrix_world @ bone.matrix_local).inverted()
        return pose_dic

    def get_transform_matrix_dic(self, ob, inverse_bind_pose_dic):
        pose_dic = {}
        for bone in ob.data.bones:
            if bone.use_deform:
                pose_dic[bone.name] = ob.matrix_world @ ob.pose.bones[bone.name].matrix @ inverse_bind_pose_dic[bone.name]
        return pose_dic

    def get_static_vertex_positions(self, ob):
        positions = []
        me = ob.data
        vertex_matrix = ob.matrix_world
        for vertex in me.vertices:
            positions.append(vertex_matrix @ vertex.co)
        return positions

    def get_dynamic_vertex_positions(self, ob, mod, show_viewport):
        positions = []
        mod.show_viewport = show_viewport
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ob_eval = ob.evaluated_get(depsgraph)
        me = ob_eval.to_mesh()
        vertex_matrix = ob.matrix_world
        for vertex in me.vertices:
            positions.append(vertex_matrix @ vertex.co)
        return positions

    def make_transform_matrix_stack(self):
        bone_names = list(self._inverse_bind_pose_dic.keys())
        transform_matrix_list = []
        for pose_index in range(len(self._dynamic_positions)):
            for bone_name in bone_names:
                transform_matrix_list.append(self._transform_matrix_dics[pose_index][bone_name])
        self._transform_matrix_stack = np.vstack(transform_matrix_list)

    def optimize_stage_0(self, objs):
        exist_any_corrective_smooth_modifiers = False
        for ob in objs:
            for mod in ob.modifiers:
                if type(mod) == bpy.types.CorrectiveSmoothModifier:
                    exist_any_corrective_smooth_modifiers = True
                    break
            if exist_any_corrective_smooth_modifiers:
                break
        return exist_any_corrective_smooth_modifiers

    def optimize_stage_1(self, arm):
        self.reset_armature(arm)
        bpy.context.view_layer.update()
        self._inverse_bind_pose_dic = self.get_inverse_bind_pose_dic(arm)

    def optimize_stage_2(self, objs):
        self._static_positions = []
        for ob in objs:
            self._static_positions.append([])
            for mod in ob.modifiers:
                if type(mod) == bpy.types.CorrectiveSmoothModifier:
                    self._static_positions[-1] = self.get_static_vertex_positions(ob)
                    break

    def optimize_stage_3(self, objs):
        self._sharp_vertex_index_dics = []
        for ob in objs:
            self._sharp_vertex_index_dics.append({})
        self._transform_matrix_dics = []
        self._dynamic_positions = []

    def optimize_stage_4(self, arm, objs, twist_angle, deviation_threshold, bake_quality, bake_range):
        self.reset_armature(arm)
        bpy.context.view_layer.update()
        self.twist_armature(arm, twist_angle)
        bpy.context.view_layer.update()
        self._transform_matrix_dics.append(self.get_transform_matrix_dic(arm, self._inverse_bind_pose_dic))
        self._dynamic_positions.append([])
        for (j, ob) in enumerate(objs):
            self._dynamic_positions[-1].append([])
            for mod in ob.modifiers:
                if type(mod) == bpy.types.CorrectiveSmoothModifier:
                    smooth_positions = self.get_dynamic_vertex_positions(ob, mod, True)
                    origin_positions = self.get_dynamic_vertex_positions(ob, mod, False)
                    self._dynamic_positions[-1][-1] = smooth_positions
                    for k in range(len(origin_positions)):
                        deviation = (origin_positions[k] - smooth_positions[k]).length
                        if bake_range == 'All' or (bake_range == 'Selected' and ob.data.vertices[k].select) or (bake_range == 'Deviation' and deviation > deviation_threshold):
                            if k not in self._sharp_vertex_index_dics[j]:
                                self._sharp_vertex_index_dics[j][k] = deviation
                            elif deviation > self._sharp_vertex_index_dics[j][k]:
                                self._sharp_vertex_index_dics[j][k] = deviation
                    break
        self._pose_index += 1

    def optimize_stage_5(self):
        self.make_transform_matrix_stack()
        self._transform_matrix_dics = None

    def optimize_stage_6(self, objs, linear_system_solver, influence_count, prune_threshold, time_out):
        import time
        if self._mesh_index == len(objs):
            self._should_terminate = True
            return 0

        sharp_vertex_index_count = len(self._sharp_vertex_index_dics[self._mesh_index])
        if sharp_vertex_index_count == 0:
            self._mesh_index += 1
            return 0

        start_time = time.time()
        sharp_vertex_indices = list(self._sharp_vertex_index_dics[self._mesh_index].items())
        sharp_vertex_indices.sort(key=lambda x:x[1], reverse=True)
        pose_count = len(self._dynamic_positions)
        bone_names = list(self._inverse_bind_pose_dic.keys())
        bone_count = len(bone_names)
        A = np.empty((pose_count*3, bone_count))
        b = np.empty((pose_count*3, 1))
        block_size = 0

        while time.time() - start_time < time_out:
            sharp_vertex_index = sharp_vertex_indices[self._vertex_index][0]
            static_position = self._static_positions[self._mesh_index][sharp_vertex_index]
            # Calculate transform: result is (N, 1) array, flatten to 1D for indexing
            transform_position_stack = np.dot(self._transform_matrix_stack, np.array([[static_position[0]],[static_position[1]],[static_position[2]],[1.0]])).flatten()

            for pose_index in range(pose_count):
                dynamic_position = self._dynamic_positions[pose_index][self._mesh_index][sharp_vertex_index]
                pose_index_per_coordinate = pose_index * 3
                b[pose_index_per_coordinate  ][0] = float(dynamic_position[0])
                b[pose_index_per_coordinate+1][0] = float(dynamic_position[1])
                b[pose_index_per_coordinate+2][0] = float(dynamic_position[2])
                bone_count_per_pose = pose_index * bone_count
                for bone_index in range(bone_count):
                    position_index_per_coordinate = (bone_count_per_pose + bone_index) * 4
                    A[pose_index_per_coordinate  ][bone_index] = float(transform_position_stack[position_index_per_coordinate  ])
                    A[pose_index_per_coordinate+1][bone_index] = float(transform_position_stack[position_index_per_coordinate+1])
                    A[pose_index_per_coordinate+2][bone_index] = float(transform_position_stack[position_index_per_coordinate+2])

            if linear_system_solver == 'SVD':
                u, s, vh = np.linalg.svd(A, full_matrices=False)
                B = np.dot(u.T, b)
                X = B
                for i in range(s.shape[0]):
                    X[i][0] /= s[i]
                x = np.dot(vh.T, X)
            elif linear_system_solver == 'STD':
                A_T = A.T
                x = np.linalg.solve(np.dot(A_T, A), np.dot(A_T, b))
            elif linear_system_solver == 'LSTSQ':
                A_T = A.T
                (x, residuals, rank, s) = np.linalg.lstsq(np.dot(A_T, A), np.dot(A_T, b), rcond=None)
            else:
                # Default to SVD
                u, s, vh = np.linalg.svd(A, full_matrices=False)
                B = np.dot(u.T, b)
                X = B
                for i in range(s.shape[0]):
                    X[i][0] /= s[i]
                x = np.dot(vh.T, X)

            bone_weights = x.tolist()
            bone_weight_list = list(zip(bone_names, bone_weights))
            bone_weight_list.sort(key=lambda x:x[1][0], reverse=True)
            bone_weight_pairs = bone_weight_list[:influence_count]

            weight_sum = 0.0
            for i in range(len(bone_weight_pairs)):
                weight_sum += bone_weight_pairs[i][1][0]

            if weight_sum != 0.0:
                weight_accumulate = 0.0
                for i in range(len(bone_weight_pairs)):
                    if weight_accumulate / weight_sum > (1.0 - prune_threshold):
                        bone_weight_pairs[i][1][0] = 0.0
                    else:
                        weight_accumulate += bone_weight_pairs[i][1][0]
                weight_sum = 0.0
                for i in range(len(bone_weight_pairs)):
                    weight_sum += bone_weight_pairs[i][1][0]

            if weight_sum != 0.0:
                for i in range(len(bone_weight_pairs)):
                    bone_weight_pairs[i][1][0] /= weight_sum

            for vertex_group in objs[self._mesh_index].vertex_groups:
                vertex_group.remove([sharp_vertex_index])

            for bone_weight_pair in bone_weight_pairs:
                (group_name, group_weight) = bone_weight_pair
                if objs[self._mesh_index].vertex_groups.get(group_name) == None:
                    objs[self._mesh_index].vertex_groups.new(name = group_name)
                objs[self._mesh_index].vertex_groups[group_name].add([sharp_vertex_index], group_weight[0], 'REPLACE')

            block_size += 1
            self._vertex_index += 1
            if self._vertex_index == len(self._sharp_vertex_index_dics[self._mesh_index]):
                self._mesh_index += 1
                self._vertex_index = 0
                break

        return block_size

import copy
import random
import numpy as np
import time
from collections import deque

# Create baking runner
operator = BakingRunner(armature, mesh_objects)

# Save current poses
operator.save_current_poses()

# Run baking stages
print("\nStage 0: Checking for Corrective Smooth modifiers...")
if not operator.optimize_stage_0(mesh_objects):
    raise ValueError("No corrective smooth modifier found. Please ensure the modifier was added.")

print("Stage 1: Getting inverse bind poses...")
operator.optimize_stage_1(armature)

print("Stage 2: Getting static positions...")
operator.optimize_stage_2(mesh_objects)

print("Stage 3: Initializing sharp vertex detection...")
operator.optimize_stage_3(mesh_objects)

print("Stage 4: Generating poses and detecting sharp vertices...")
max_poses = max(math.ceil(len(operator._inverse_bind_pose_dic) * float(config['bake_quality'])), 80)
for pose_idx in range(max_poses):
    operator.optimize_stage_4(
        armature,
        mesh_objects,
        config['twist_angle'],
        config['deviation_threshold'] * 0.01,  # Convert cm to meters
        float(config['bake_quality']),
        config['bake_range']
    )
    if (pose_idx + 1) % 10 == 0:
        print(f"  Generated {pose_idx + 1}/{max_poses} poses...")

# Restore poses
operator.restore_current_poses()

print("Stage 5: Making transform matrix stack...")
operator.optimize_stage_5()

print("Stage 6: Optimizing vertex weights...")
# Run optimization for each mesh
total_vertices = sum(len(dic) for dic in operator._sharp_vertex_index_dics)
processed_vertices = 0

# Initialize ETA estimation
# Use exponential moving average for better predictions
time_observations = deque(maxlen=50)  # Keep last 50 observations
eta_alpha = 0.3  # Exponential smoothing factor (0-1, higher = more weight to recent)
estimated_time_per_vertex = None

stage6_start_time = time.time()
last_eta_update_time = time.time()
eta_update_interval = 5.0  # Update ETA every 5 seconds

for mesh_idx in range(len(mesh_objects)):
    sharp_count = len(operator._sharp_vertex_index_dics[mesh_idx])
    if sharp_count == 0:
        print(f"  Mesh {mesh_idx + 1}/{len(mesh_objects)}: No sharp vertices, skipping")
        continue

    print(f"  Mesh {mesh_idx + 1}/{len(mesh_objects)}: Processing {sharp_count} sharp vertices...")
    operator._mesh_index = mesh_idx
    operator._vertex_index = 0
    mesh_start_vertex_idx = processed_vertices

    # Process vertices in blocks (with timeout for UI responsiveness)
    while operator._vertex_index < sharp_count:
        block_start_time = time.time()
        block_start_vertex_idx = processed_vertices

        block_size = operator.optimize_stage_6(
            mesh_objects,
            config['solver'],
            config['influence_bones'],
            config['prune_threshold'],
            1.0 / config['refresh_frequency']
        )

        block_end_time = time.time()
        block_time = block_end_time - block_start_time
        processed_vertices += block_size

        # Update time observations for ETA estimation
        if block_size > 0:
            time_per_vertex = block_time / block_size
            time_observations.append(time_per_vertex)

            # Update exponential moving average
            if estimated_time_per_vertex is None:
                estimated_time_per_vertex = time_per_vertex
            else:
                # Exponential moving average: more weight to recent observations
                estimated_time_per_vertex = eta_alpha * time_per_vertex + (1 - eta_alpha) * estimated_time_per_vertex

        # Print progress with ETA
        if block_size > 0:
            progress = 100.0 * processed_vertices / total_vertices if total_vertices > 0 else 0
            current_time = time.time()

            # Calculate ETA using exponential moving average
            eta_seconds = None
            if processed_vertices > 0 and total_vertices > processed_vertices:
                remaining_vertices = total_vertices - processed_vertices

                # Use exponential moving average if available
                if estimated_time_per_vertex is not None and estimated_time_per_vertex > 0:
                    eta_seconds = estimated_time_per_vertex * remaining_vertices
                elif len(time_observations) > 0:
                    # Fall back to recent average
                    recent_avg = np.mean(list(time_observations)[-min(10, len(time_observations)):])
                    eta_seconds = recent_avg * remaining_vertices
                else:
                    # Simple linear extrapolation as last resort
                    elapsed_time = current_time - stage6_start_time
                    avg_time_per_vertex = elapsed_time / processed_vertices
                    eta_seconds = avg_time_per_vertex * remaining_vertices

            # Format ETA
            if eta_seconds is not None:
                eta_hours = int(eta_seconds / 3600)
                eta_minutes = int((eta_seconds % 3600) / 60)
                eta_secs = int(eta_seconds % 60)
                if eta_hours > 0:
                    eta_str = f"{eta_hours}h {eta_minutes}m {eta_secs}s"
                elif eta_minutes > 0:
                    eta_str = f"{eta_minutes}m {eta_secs}s"
                else:
                    eta_str = f"{eta_secs}s"
                print(f"    Progress: {progress:.1f}% ({processed_vertices}/{total_vertices} vertices) | ETA: {eta_str}")
            else:
                print(f"    Progress: {progress:.1f}% ({processed_vertices}/{total_vertices} vertices)")

        if operator._should_terminate:
            break

print("\nBaking completed successfully!")

# Remove Corrective Smooth modifiers after baking
print("\nRemoving Corrective Smooth modifiers...")
for mesh_obj in mesh_objects:
    for mod in list(mesh_obj.modifiers):
        if mod.type == 'CORRECTIVE_SMOOTH':
            mesh_obj.modifiers.remove(mod)
            print(f"  Removed modifier from {mesh_obj.name}")

# Export result
print(f"\nExporting to: {output_path}")
os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

if output_path.endswith('.usdc') or output_path.endswith('.usda') or output_path.endswith('.usd'):
    # USD export with embedded images and material preservation
    # Using only essential parameters that are valid in Blender 4.5
    bpy.ops.wm.usd_export(
        filepath=output_path,
        export_materials=True,
        export_textures=True,
        relative_paths=False,  # False = embed textures, True = use relative paths
        export_uvmaps=True,
        export_armatures=True,  # Preserve armatures for rigged models
        selected_objects_only=False,
        visible_objects_only=False,
        use_instancing=False,
        evaluation_mode='RENDER'
    )
else:
    raise ValueError(f"Unsupported output format: {output_path}")

print(f"✓ Corrected model saved: {output_path}")

# Cleanup
try:
    if os.path.exists('config.json'):
        os.remove('config.json')
except Exception as e:
    # Ignore cleanup errors
    pass
""", %{"config_file_normalized" => config_file_normalized})
rescue
  e ->
    # Clean up temp file on error
    if File.exists?(config_file) do
      File.rm(config_file)
    end
    reraise e, __STACKTRACE__
after
  # Clean up temp file
  if File.exists?(config_file) do
    File.rm(config_file)
  end
end

IO.puts("""
=== Complete ===
Corrective smooth baking completed successfully!
Output: #{config.output_path}

The Corrective Smooth modifier effects have been baked into vertex weights,
improving deformation quality. The modifier has been removed from the exported model.
""")
