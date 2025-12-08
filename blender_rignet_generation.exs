#!/usr/bin/env elixir

# Blender RigNet Generation Script
# Automatically rig 3D models with skeleton and skinning weights using RigNet
# Repository: https://github.com/zhan-xu/RigNet
# Blender Addon: https://github.com/V-Sekai/blender-rignet
#
# Usage:
#   elixir blender_rignet_generation.exs <mesh_path> [options]
#
# Options:
#   --output-format "txt"        Output format: txt (rig file) or fbx (default: "txt")
#   --bandwidth <float>          Bandwidth for meanshift clustering (default: auto)
#   --threshold <float>          Density threshold for joint filtering (default: 1e-5)
#   --target-vertices <int>      Target vertex count for mesh optimization (2000-5000, default: 2000)
#   --checkpoint-dir <path>      Path to RigNet checkpoints (default: thirdparty/blender-rignet/RigNet/checkpoints)
#   --device "cuda"              Device: cuda or cpu (default: auto-detect)
#   --skip-optimization          Skip mesh optimization (use original mesh)

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"}
])

# Initialize Python environment with required dependencies
# RigNet uses PyTorch, torch-geometric, and various 3D processing libraries
Pythonx.uv_init("""
[project]
name = "blender-rignet-generation"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "numpy",
  "scipy",
  "torch",
  "torchvision",
  "tensorboard",
  "torch-geometric",
  "torch-cluster @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_cluster-1.6.3%2Bpt27cu118-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
  "torch-cluster @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_cluster-1.6.3%2Bpt27cu118-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux'",
  "torch_scatter @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_scatter-2.1.2%2Bpt27cu118-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
  "torch_scatter @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_scatter-2.1.2%2Bpt27cu118-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux'",
  "torch-sparse @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_sparse-0.6.18%2Bpt27cu118-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
  "torch-sparse @ https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_sparse-0.6.18%2Bpt27cu118-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux'",
  "trimesh",
  "open3d",
  "opencv-python",
  "tqdm",
  "scikit-learn",
  "fast-simplification",
  "bpy==4.5.*",
  "rtree",
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
  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        bandwidth: :float,
        threshold: :float,
        target_vertices: :integer,
        checkpoint_dir: :string,
        device: :string,
        skip_optimization: :boolean
      ],
      aliases: [
        b: :bandwidth,
        t: :threshold,
        v: :target_vertices,
        c: :checkpoint_dir,
        d: :device
      ]
    )

    mesh_path = List.first(args)

    if mesh_path == nil do
      IO.puts("""
      Error: Mesh path is required.

      Usage:
        elixir blender_rignet_generation.exs <mesh_path> [options]

      Options:
        --output-format, -f "txt"         Output format: txt (rig file) or fbx (default: "txt")
        --bandwidth, -b <float>           Bandwidth for meanshift clustering (default: auto)
        --threshold, -t <float>           Density threshold for joint filtering (default: 1e-5)
        --target-vertices, -v <int>       Target vertex count for mesh optimization (2000-5000, default: 2000)
        --skip-optimization               Skip mesh optimization (use original mesh)
        --checkpoint-dir, -c <path>       Path to RigNet checkpoints (default: thirdparty/blender-rignet/RigNet/checkpoints)
        --device, -d "cuda"               Device: cuda or cpu (default: auto-detect)
      """)
      System.halt(1)
    end

    if not File.exists?(mesh_path) do
      IO.puts("Error: Mesh file not found: #{mesh_path}")
      System.halt(1)
    end

    %{
      mesh_path: mesh_path,
      bandwidth: Keyword.get(opts, :bandwidth),
      threshold: Keyword.get(opts, :threshold, 1.0e-5),
      target_vertices: Keyword.get(opts, :target_vertices, 2000),
      checkpoint_dir: Keyword.get(opts, :checkpoint_dir),
      device: Keyword.get(opts, :device),
      skip_optimization: Keyword.get(opts, :skip_optimization, false)
    }
  end
end

# Parse arguments
config = ArgsParser.parse(System.argv())

IO.puts("""
=== Blender RigNet Generation (USDC Pipeline) ===
Input Mesh: #{config.mesh_path}
Output Format: USDC
Threshold: #{config.threshold}
Bandwidth: #{if config.bandwidth, do: Float.to_string(config.bandwidth), else: "auto"}
Target Vertices: #{config.target_vertices}
Skip Optimization: #{config.skip_optimization}
Device: #{config.device || "auto-detect"}
""")

# Determine checkpoint directory
checkpoint_dir = config.checkpoint_dir || Path.join(["thirdparty", "blender-rignet", "RigNet", "checkpoints"])

if not File.exists?(checkpoint_dir) do
  IO.puts("""
  ⚠ Warning: Checkpoint directory not found: #{checkpoint_dir}
  Please download RigNet checkpoints or specify --checkpoint-dir
  """)
end

# Determine output path (always USDC)
mesh_name = Path.basename(config.mesh_path, Path.extname(config.mesh_path))
{{year, month, day}, {hour, minute, second}} = :calendar.universal_time()
timestamp = "#{year}_#{String.pad_leading(Integer.to_string(month), 2, "0")}_#{String.pad_leading(Integer.to_string(day), 2, "0")}_#{String.pad_leading(Integer.to_string(hour), 2, "0")}_#{String.pad_leading(Integer.to_string(minute), 2, "0")}_#{String.pad_leading(Integer.to_string(second), 2, "0")}"
output_dir = Path.join(["output", timestamp])
File.mkdir_p!(output_dir)

output_path = Path.join(output_dir, "#{mesh_name}_rigged.usdc")

# Save config to JSON for Python to read
config_json = Jason.encode!(%{
  mesh_path: config.mesh_path,
  output_path: output_path,
  bandwidth: config.bandwidth,
  threshold: config.threshold,
  target_vertices: config.target_vertices,
  skip_optimization: config.skip_optimization,
  checkpoint_dir: checkpoint_dir,
  device: config.device
})
File.write!("config.json", config_json)

# Run RigNet prediction
IO.puts("\n=== Running RigNet Prediction ===")

{_, _python_globals} = Pythonx.eval("""
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
import bpy

# Prevent Python popups/windows on Windows
if sys.platform == "win32":
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Prevent GUI dialogs
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # Prevent Open3D GUI windows
    os.environ["OPEN3D_HEADLESS"] = "1"
    # Prevent subprocess windows
    import subprocess
    # Set CREATE_NO_WINDOW flag for subprocess calls
    subprocess.CREATE_NO_WINDOW = 0x08000000

# Prevent Open3D visualization windows globally
os.environ["OPEN3D_HEADLESS"] = "1"

# Add RigNet to path
riget_path = Path.cwd() / "thirdparty" / "blender-rignet"
riget_rig_path = riget_path / "RigNet"
sys.path.insert(0, str(riget_path))
sys.path.insert(0, str(riget_rig_path))

with open("config.json", 'r') as f:
    config = json.load(f)

mesh_path = config['mesh_path']
output_path = config['output_path']
checkpoint_dir = config['checkpoint_dir']
device_str = config.get('device')
threshold = config.get('threshold', 1e-5)
bandwidth = config.get('bandwidth')
target_vertices = config.get('target_vertices', 2000)
skip_optimization = config.get('skip_optimization', False)

# Determine device
if device_str:
    device = torch.device(device_str)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Input USDC: {mesh_path}")
print(f"Output USDC: {output_path}")
print(f"Checkpoints: {checkpoint_dir}")

# Step 1: Load USDC in Blender and export as OBJ for RigNet
print("\\n=== Step 1: Loading USDC Mesh ===")
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import USDC
try:
    bpy.ops.wm.usd_import(
        filepath=str(Path(mesh_path).resolve()),
        import_materials=True
    )
    print("✓ USDC loaded successfully")
except Exception as e:
    print(f"Error loading USDC: {e}")
    import traceback
    traceback.print_exc()
    raise

# Find mesh objects
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if len(mesh_objects) == 0:
    raise ValueError("No mesh objects found in USDC file")

print(f"Found {len(mesh_objects)} mesh object(s)")

# Select all meshes and join them if multiple
if len(mesh_objects) > 1:
    print("Joining multiple meshes into one...")
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    mesh_obj = bpy.context.active_object
else:
    mesh_obj = mesh_objects[0]

# Optimize mesh if needed using meshoptimizer
if not skip_optimization:
    print(f"\\n=== Optimizing Mesh to ~{target_vertices} vertices using meshoptimizer ===")
    original_vertices = len(mesh_obj.data.vertices)
    original_faces = len(mesh_obj.data.polygons)
    print(f"Original: {original_vertices} vertices, {original_faces} faces")

    if original_vertices > target_vertices:
        try:
            import ctypes
            import platform
            import os

            # Load meshoptimizer library from thirdparty/blender-meshoptimizer
            meshopt_lib = None
            system = platform.system()
            if system == "Windows":
                lib_name = "meshoptimizer.dll"
            elif system == "Darwin":  # macOS
                lib_name = "libmeshoptimizer.dylib"
            else:  # Linux
                lib_name = "libmeshoptimizer.so"

            # Try to load from thirdparty/blender-meshoptimizer
            meshopt_path = Path.cwd() / "thirdparty" / "blender-meshoptimizer"
            search_paths = [
                str(meshopt_path),
                str(meshopt_path / "lib"),
            ]

            for path in search_paths:
                lib_path = os.path.join(path, lib_name)
                if os.path.exists(lib_path):
                    try:
                        meshopt_lib = ctypes.CDLL(lib_path)
                        break
                    except OSError:
                        continue

            # Try system path as fallback
            if meshopt_lib is None:
                try:
                    meshopt_lib = ctypes.CDLL(lib_name)
                except OSError:
                    pass

            if meshopt_lib is None:
                raise ImportError(f"Could not load meshoptimizer library ({lib_name}). Please ensure it's built and available.")

            # Setup function signatures
            meshopt_lib.meshopt_simplify.argtypes = [
                ctypes.POINTER(ctypes.c_uint32),  # destination
                ctypes.POINTER(ctypes.c_uint32),  # indices
                ctypes.c_size_t,  # index_count
                ctypes.POINTER(ctypes.c_float),  # vertex_positions
                ctypes.c_size_t,  # vertex_count
                ctypes.c_size_t,  # vertex_positions_stride
                ctypes.c_size_t,  # target_index_count
                ctypes.c_float,  # target_error
                ctypes.c_uint32,  # options
                ctypes.POINTER(ctypes.c_float),  # result_error (can be NULL)
            ]
            meshopt_lib.meshopt_simplify.restype = ctypes.c_size_t

            meshopt_lib.meshopt_simplifyScale.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # vertex_positions
                ctypes.c_size_t,  # vertex_count
                ctypes.c_size_t,  # vertex_positions_stride
            ]
            meshopt_lib.meshopt_simplifyScale.restype = ctypes.c_float

            # Extract mesh data from Blender
            mesh_data = mesh_obj.data
            vertices = np.array([v.co[:] for v in mesh_data.vertices], dtype=np.float32)

            # Get face indices
            faces = []
            for poly in mesh_data.polygons:
                face_verts = [mesh_data.loops[i].vertex_index for i in range(poly.loop_start, poly.loop_start + poly.loop_total)]
                # Triangulate if needed (meshoptimizer works with triangles)
                if len(face_verts) == 3:
                    faces.append(face_verts)
                elif len(face_verts) == 4:
                    # Split quad into two triangles
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])
                else:
                    # Triangulate n-gon (simple fan triangulation)
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

            indices = np.array(faces, dtype=np.uint32).flatten()

            # Calculate target index count (approximately 3x target_vertices for triangles)
            target_indices = target_vertices * 3

            print(f"Simplifying mesh using meshoptimizer...")
            print(f"  Input: {len(vertices)} vertices, {len(indices) // 3} triangles")
            print(f"  Target: ~{target_vertices} vertices, ~{target_indices // 3} triangles")

            # Prepare arrays for ctypes
            vertex_positions = np.ascontiguousarray(vertices.flatten(), dtype=np.float32)
            indices_uint32 = np.ascontiguousarray(indices.astype(np.uint32), dtype=np.uint32)
            index_count = len(indices_uint32)
            vertex_count = len(vertices)
            target_indices_int = int(target_indices)
            vertex_positions_stride = ctypes.sizeof(ctypes.c_float) * 3  # 12 bytes

            # Calculate error scale based on mesh bounding box
            vertex_positions_ptr = vertex_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            error_scale = meshopt_lib.meshopt_simplifyScale(
                vertex_positions_ptr,
                vertex_count,
                vertex_positions_stride
            )

            # Use error-based simplification with target index count
            target_error = 0.01 * error_scale  # Scale error relative to mesh size
            options = 0  # No special options

            # Create destination array for simplified indices
            destination = np.zeros(index_count, dtype=np.uint32)
            result_error = ctypes.c_float()

            # Call meshopt_simplify
            indices_ptr = indices_uint32.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            destination_ptr = destination.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

            result_count = meshopt_lib.meshopt_simplify(
                destination_ptr,
                indices_ptr,
                index_count,
                vertex_positions_ptr,
                vertex_count,
                vertex_positions_stride,
                target_indices_int,
                target_error,
                options,
                ctypes.byref(result_error)
            )

            if result_count == 0:
                raise ValueError("meshoptimizer returned 0 indices")

            # Extract simplified indices
            simplified_indices = destination[:result_count]
            print(f"  meshoptimizer returned {result_count} indices ({result_count // 3} triangles)")

            # Validate that we actually got simplified indices
            if len(simplified_indices) >= len(indices_uint32):
                print(f"  Warning: Simplified indices ({len(simplified_indices)}) >= original ({len(indices_uint32)}), meshoptimizer may not have simplified")
                raise ValueError("meshoptimizer did not reduce mesh complexity")

            # Get unique vertices from simplified indices (these reference original vertex array)
            unique_vertex_indices = np.unique(simplified_indices)
            simplified_vertices = vertices[unique_vertex_indices]

            print(f"  Simplified to {len(unique_vertex_indices)} unique vertices, {len(simplified_indices) // 3} triangles")
        except (ImportError, Exception) as e:
            print(f"Warning: meshoptimizer failed: {e}")
            print("Falling back to Blender's decimate modifier...")
            # Fallback to Blender decimate
            bpy.context.view_layer.objects.active = mesh_obj
            decimate = mesh_obj.modifiers.new(name='Decimate', type='DECIMATE')
            decimate.ratio = target_vertices / original_vertices
            decimate.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier='Decimate')
            print(f"Decimated to {len(mesh_obj.data.vertices)} vertices (fallback)")
        else:
            # Remap indices to new vertex array
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
            remapped_indices = np.array([vertex_map[idx] for idx in simplified_indices], dtype=np.uint32)

            # Build faces list for from_pydata (list of lists, each inner list is vertex indices for a face)
            faces_list = []
            for i in range(0, len(remapped_indices), 3):
                if i + 2 < len(remapped_indices):
                    faces_list.append([
                        int(remapped_indices[i]),
                        int(remapped_indices[i + 1]),
                        int(remapped_indices[i + 2])
                    ])

            # Update Blender mesh with optimized data
            # from_pydata(vertices, edges, faces) - faces is list of vertex index lists
            mesh_data.clear_geometry()
            mesh_data.from_pydata(simplified_vertices.tolist(), [], faces_list)

            mesh_data.update()
            print(f"✓ Mesh optimized to {len(mesh_obj.data.vertices)} vertices using meshoptimizer")
    else:
        print(f"Mesh already has {original_vertices} vertices (target: {target_vertices})")

# Export as OBJ for RigNet processing
# Make path absolute to avoid issues when changing directories
temp_obj_path_base = Path(mesh_path).with_suffix('').resolve()
temp_obj_path = str(temp_obj_path_base) + '_temp_rignet.obj'
bpy.ops.wm.obj_export(
    filepath=temp_obj_path,
    export_selected_objects=True,
    export_uv=True,
    export_normals=True
)
print(f"✓ Exported OBJ for RigNet: {temp_obj_path}")
# Verify file exists
if not os.path.exists(temp_obj_path):
    raise FileNotFoundError(f"Failed to export OBJ file: {temp_obj_path}")

# Step 2: Run RigNet prediction
print("\\n=== Step 2: Running RigNet Prediction ===")

# Try to handle torch_scatter import errors gracefully for CUDA
# If CUDA version fails, the error will propagate but Erlang will handle it
try:
    import torch_scatter
    print("✓ torch_scatter loaded")
except Exception as e:
    print(f"Warning: torch_scatter import failed: {e}")
    print("  This may be due to missing CUDA dependencies")
    print("  Attempting to continue...")
    # Let the error propagate - Erlang will handle it
    raise

# Import RigNet modules
try:
    # Import the module first so we can set its global variables
    import RigNet.quick_start as quick_start_module
    from RigNet.quick_start import (
        create_single_data,
        predict_joints,
        predict_skeleton,
        predict_skinning,
    )
    from RigNet.models.GCN import JOINTNET_MASKNET_MEANSHIFT as JOINTNET
    from RigNet.models.ROOT_GCN import ROOTNET
    from RigNet.models.PairCls_GCN import PairCls as BONENET
    from RigNet.models.SKINNING import SKINNET
    print("✓ RigNet modules imported")
except Exception as e:
    print(f"Error importing RigNet modules: {e}")
    import traceback
    traceback.print_exc()
    # Let Erlang handle the error
    raise

# Load networks
print("\\n=== Loading Networks ===")
# Save original working directory
original_cwd = os.getcwd()
# Change to checkpoint directory for loading models
checkpoint_parent = Path(checkpoint_dir).parent.resolve()
os.chdir(str(checkpoint_parent))

# Helper function to load state dict with error handling
def load_state_dict_safe(model, checkpoint_path, model_name):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Try strict loading first
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ {model_name} loaded (strict)")
        except RuntimeError as e:
            # If strict fails, try non-strict and filter out unexpected keys
            print(f"Warning: {model_name} strict loading failed")
            print(f"  Attempting non-strict loading...")

            # Get model's expected keys
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())

            # Filter state_dict to only include keys that exist in model
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys

            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)} (will use random initialization)")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)} (will be ignored)")

            model.load_state_dict(filtered_dict, strict=False)
            print(f"✓ {model_name} loaded (non-strict)")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        import traceback
        traceback.print_exc()
        raise

# Joint network - use gcn_meanshift (finetuned) instead of pretrain_jointnet
jointNet = JOINTNET()
jointNet.to(device)
jointNet.eval()
joint_checkpoint_path = 'checkpoints/gcn_meanshift/model_best.pth.tar'
if not os.path.exists(joint_checkpoint_path):
    # Fallback to pretrain_jointnet if gcn_meanshift doesn't exist
    joint_checkpoint_path = 'checkpoints/pretrain_jointnet/model_best.pth.tar'
load_state_dict_safe(jointNet, joint_checkpoint_path, "Joint prediction network")
print("✓ Joint prediction network loaded")

# Root network
rootNet = ROOTNET()
rootNet.to(device)
rootNet.eval()
load_state_dict_safe(rootNet, 'checkpoints/rootnet/model_best.pth.tar', "Root prediction network")
print("✓ Root prediction network loaded")

# Bone network
boneNet = BONENET()
boneNet.to(device)
boneNet.eval()
load_state_dict_safe(boneNet, 'checkpoints/bonenet/model_best.pth.tar', "Bone connectivity network")
print("✓ Bone connectivity network loaded")

# Skin network
skinNet = SKINNET(nearest_bone=5, use_Dg=True, use_Lf=True)
skinNet.to(device)
skinNet.eval()
load_state_dict_safe(skinNet, 'checkpoints/skinnet/model_best.pth.tar', "Skinning network")
print("✓ Skinning network loaded")

# Create input data from OBJ
print("\\n=== Processing Mesh for RigNet ===")
# Use absolute path (already resolved before chdir)
# Convert to string and normalize path separators for cross-platform compatibility
temp_obj_absolute = os.path.normpath(temp_obj_path)
print(f"Loading OBJ from: {temp_obj_absolute}")
if not os.path.exists(temp_obj_absolute):
    raise FileNotFoundError(f"OBJ file not found: {temp_obj_absolute}")
print(f"OBJ file exists: {os.path.exists(temp_obj_absolute)}, size: {os.path.getsize(temp_obj_absolute)} bytes")
data, vox, surface_geodesic, translation_normalize, scale_normalize = create_single_data(temp_obj_absolute)
data.to(device)

# Predict joints
print("\\n=== Predicting Joints ===")
data = predict_joints(data, vox, jointNet, threshold, bandwidth=bandwidth)
data.to(device)

# Predict skeleton
print("\\n=== Predicting Skeleton ===")
# Construct normalized OBJ path for predict_skeleton (required parameter)
normalized_obj_path = temp_obj_absolute.replace('.obj', '_normalized.obj')
pred_skeleton = predict_skeleton(data, vox, rootNet, boneNet, normalized_obj_path)

# Predict skinning
print("\\n=== Predicting Skinning Weights ===")
# Set device as global variable in RigNet module for predict_skinning
quick_start_module.device = device
pred_rig = predict_skinning(data, pred_skeleton, skinNet, surface_geodesic, normalized_obj_path)

# Reverse normalization
pred_rig.normalize(scale_normalize, -translation_normalize)

# Save rig file temporarily
# Use absolute path for rig file
temp_rig_path = str(Path(temp_obj_path).with_suffix('').resolve()) + '_rig.txt'
pred_rig.save(temp_rig_path)
print(f"✓ Rig saved to: {temp_rig_path}")

# Clear GPU cache
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Restore original working directory before returning to Blender
os.chdir(original_cwd)

# Step 3: Load rig back into Blender and apply to mesh
print("\\n=== Step 3: Applying Rig to Mesh ===")

# Use blender-rignet's rig parser and armature generator
try:
    from RigNet.utils.rig_parser import Info
    from ob_utils.objects import ArmatureGenerator

    # Parse rig file - use absolute path
    skel_info = Info(filename=str(Path(temp_rig_path).resolve()))
    print("✓ Rig file parsed")

    # Create armature and apply to mesh
    # The mesh_obj is already in the scene from Step 1
    bpy.context.view_layer.objects.active = mesh_obj
    ArmatureGenerator(skel_info, mesh_obj).generate()
    print("✓ Armature created and applied to mesh")
except Exception as e:
    print(f"Error applying rig: {e}")
    import traceback
    traceback.print_exc()
    raise

# Step 4: Export final rigged model as USDC
print("\\n=== Step 4: Exporting Rigged Model as USDC ===")
bpy.ops.wm.usd_export(
    filepath=output_path,
    export_materials=True,
    export_textures=True,
    relative_paths=False,
    export_uvmaps=True,
    export_armatures=True,
    selected_objects_only=False,
    visible_objects_only=False,
    use_instancing=False,
    evaluation_mode='RENDER'
)
print(f"✓ Rigged USDC exported: {output_path}")

# Cleanup temp files
try:
    os.remove(temp_obj_path)
    os.remove(temp_rig_path)
    print("✓ Cleaned up temporary files")
except:
    pass

print("\\n✓ USDC pipeline complete!")

print("\\n✓ RigNet generation complete!")

print("Output saved to: #{output_path}")
import torch
import numpy as np
import bpy

# Add RigNet to path
riget_path = Path.cwd() / "thirdparty" / "blender-rignet"
riget_rig_path = riget_path / "RigNet"
sys.path.insert(0, str(riget_path))
sys.path.insert(0, str(riget_rig_path))

with open("config.json", 'r') as f:
    config = json.load(f)

mesh_path = config['mesh_path']
output_path = config['output_path']
checkpoint_dir = config['checkpoint_dir']
device_str = config.get('device')
threshold = config.get('threshold', 1e-5)
bandwidth = config.get('bandwidth')
target_vertices = config.get('target_vertices', 2000)
skip_optimization = config.get('skip_optimization', False)

# Determine device
if device_str:
    device = torch.device(device_str)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Input USDC: {mesh_path}")
print(f"Output USDC: {output_path}")
print(f"Checkpoints: {checkpoint_dir}")

# Step 1: Load USDC in Blender and export as OBJ for RigNet
print("\\n=== Step 1: Loading USDC Mesh ===")
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import USDC
try:
    bpy.ops.wm.usd_import(
        filepath=str(Path(mesh_path).resolve()),
        import_materials=True
    )
    print("✓ USDC loaded successfully")
except Exception as e:
    print(f"Error loading USDC: {e}")
    import traceback
    traceback.print_exc()
    raise

# Find mesh objects
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if len(mesh_objects) == 0:
    raise ValueError("No mesh objects found in USDC file")

print(f"Found {len(mesh_objects)} mesh object(s)")

# Select all meshes and join them if multiple
if len(mesh_objects) > 1:
    print("Joining multiple meshes into one...")
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    mesh_obj = bpy.context.active_object
else:
    mesh_obj = mesh_objects[0]

# Optimize mesh if needed using meshoptimizer
if not skip_optimization:
    print(f"\\n=== Optimizing Mesh to ~{target_vertices} vertices using meshoptimizer ===")
    original_vertices = len(mesh_obj.data.vertices)
    original_faces = len(mesh_obj.data.polygons)
    print(f"Original: {original_vertices} vertices, {original_faces} faces")

    if original_vertices > target_vertices:
        try:
            import ctypes
            import platform
            import os

            # Load meshoptimizer library from thirdparty/blender-meshoptimizer
            meshopt_lib = None
            system = platform.system()
            if system == "Windows":
                lib_name = "meshoptimizer.dll"
            elif system == "Darwin":  # macOS
                lib_name = "libmeshoptimizer.dylib"
            else:  # Linux
                lib_name = "libmeshoptimizer.so"

            # Try to load from thirdparty/blender-meshoptimizer
            meshopt_path = Path.cwd() / "thirdparty" / "blender-meshoptimizer"
            search_paths = [
                str(meshopt_path),
                str(meshopt_path / "lib"),
            ]

            for path in search_paths:
                lib_path = os.path.join(path, lib_name)
                if os.path.exists(lib_path):
                    try:
                        meshopt_lib = ctypes.CDLL(lib_path)
                        break
                    except OSError:
                        continue

            # Try system path as fallback
            if meshopt_lib is None:
                try:
                    meshopt_lib = ctypes.CDLL(lib_name)
                except OSError:
                    pass

            if meshopt_lib is None:
                raise ImportError(f"Could not load meshoptimizer library ({lib_name}). Please ensure it's built and available.")

            # Setup function signatures
            meshopt_lib.meshopt_simplify.argtypes = [
                ctypes.POINTER(ctypes.c_uint32),  # destination
                ctypes.POINTER(ctypes.c_uint32),  # indices
                ctypes.c_size_t,  # index_count
                ctypes.POINTER(ctypes.c_float),  # vertex_positions
                ctypes.c_size_t,  # vertex_count
                ctypes.c_size_t,  # vertex_positions_stride
                ctypes.c_size_t,  # target_index_count
                ctypes.c_float,  # target_error
                ctypes.c_uint32,  # options
                ctypes.POINTER(ctypes.c_float),  # result_error (can be NULL)
            ]
            meshopt_lib.meshopt_simplify.restype = ctypes.c_size_t

            meshopt_lib.meshopt_simplifyScale.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # vertex_positions
                ctypes.c_size_t,  # vertex_count
                ctypes.c_size_t,  # vertex_positions_stride
            ]
            meshopt_lib.meshopt_simplifyScale.restype = ctypes.c_float

            # Extract mesh data from Blender
            mesh_data = mesh_obj.data
            vertices = np.array([v.co[:] for v in mesh_data.vertices], dtype=np.float32)

            # Get face indices
            faces = []
            for poly in mesh_data.polygons:
                face_verts = [mesh_data.loops[i].vertex_index for i in range(poly.loop_start, poly.loop_start + poly.loop_total)]
                # Triangulate if needed (meshoptimizer works with triangles)
                if len(face_verts) == 3:
                    faces.append(face_verts)
                elif len(face_verts) == 4:
                    # Split quad into two triangles
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])
                else:
                    # Triangulate n-gon (simple fan triangulation)
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

            indices = np.array(faces, dtype=np.uint32).flatten()

            # Calculate target index count (approximately 3x target_vertices for triangles)
            target_indices = target_vertices * 3

            print(f"Simplifying mesh using meshoptimizer...")
            print(f"  Input: {len(vertices)} vertices, {len(indices) // 3} triangles")
            print(f"  Target: ~{target_vertices} vertices, ~{target_indices // 3} triangles")

            # Prepare arrays for ctypes
            vertex_positions = np.ascontiguousarray(vertices.flatten(), dtype=np.float32)
            indices_uint32 = np.ascontiguousarray(indices.astype(np.uint32), dtype=np.uint32)
            index_count = len(indices_uint32)
            vertex_count = len(vertices)
            target_indices_int = int(target_indices)
            vertex_positions_stride = ctypes.sizeof(ctypes.c_float) * 3  # 12 bytes

            # Calculate error scale based on mesh bounding box
            vertex_positions_ptr = vertex_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            error_scale = meshopt_lib.meshopt_simplifyScale(
                vertex_positions_ptr,
                vertex_count,
                vertex_positions_stride
            )

            # Use error-based simplification with target index count
            target_error = 0.01 * error_scale  # Scale error relative to mesh size
            options = 0  # No special options

            # Create destination array for simplified indices
            destination = np.zeros(index_count, dtype=np.uint32)
            result_error = ctypes.c_float()

            # Call meshopt_simplify
            indices_ptr = indices_uint32.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            destination_ptr = destination.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

            result_count = meshopt_lib.meshopt_simplify(
                destination_ptr,
                indices_ptr,
                index_count,
                vertex_positions_ptr,
                vertex_count,
                vertex_positions_stride,
                target_indices_int,
                target_error,
                options,
                ctypes.byref(result_error)
            )

            if result_count == 0:
                raise ValueError("meshoptimizer returned 0 indices")

            # Extract simplified indices
            simplified_indices = destination[:result_count]
            print(f"  meshoptimizer returned {result_count} indices ({result_count // 3} triangles)")

            # Validate that we actually got simplified indices
            if len(simplified_indices) >= len(indices_uint32):
                print(f"  Warning: Simplified indices ({len(simplified_indices)}) >= original ({len(indices_uint32)}), meshoptimizer may not have simplified")
                raise ValueError("meshoptimizer did not reduce mesh complexity")

            # Get unique vertices from simplified indices (these reference original vertex array)
            unique_vertex_indices = np.unique(simplified_indices)
            simplified_vertices = vertices[unique_vertex_indices]

            print(f"  Simplified to {len(unique_vertex_indices)} unique vertices, {len(simplified_indices) // 3} triangles")
        except (ImportError, Exception) as e:
            print(f"Warning: meshoptimizer failed: {e}")
            print("Falling back to Blender's decimate modifier...")
            # Fallback to Blender decimate
            bpy.context.view_layer.objects.active = mesh_obj
            decimate = mesh_obj.modifiers.new(name='Decimate', type='DECIMATE')
            decimate.ratio = target_vertices / original_vertices
            decimate.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier='Decimate')
            print(f"Decimated to {len(mesh_obj.data.vertices)} vertices (fallback)")
        else:
            # Remap indices to new vertex array
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
            remapped_indices = np.array([vertex_map[idx] for idx in simplified_indices], dtype=np.uint32)

            # Build faces list for from_pydata (list of lists, each inner list is vertex indices for a face)
            faces_list = []
            for i in range(0, len(remapped_indices), 3):
                if i + 2 < len(remapped_indices):
                    faces_list.append([
                        int(remapped_indices[i]),
                        int(remapped_indices[i + 1]),
                        int(remapped_indices[i + 2])
                    ])

            # Update Blender mesh with optimized data
            # from_pydata(vertices, edges, faces) - faces is list of vertex index lists
            mesh_data.clear_geometry()
            mesh_data.from_pydata(simplified_vertices.tolist(), [], faces_list)

            mesh_data.update()
            print(f"✓ Mesh optimized to {len(mesh_obj.data.vertices)} vertices using meshoptimizer")
    else:
        print(f"Mesh already has {original_vertices} vertices (target: {target_vertices})")

# Export as OBJ for RigNet processing
# Make path absolute to avoid issues when changing directories
temp_obj_path_base = Path(mesh_path).with_suffix('').resolve()
temp_obj_path = str(temp_obj_path_base) + '_temp_rignet.obj'
bpy.ops.wm.obj_export(
    filepath=temp_obj_path,
    export_selected_objects=True,
    export_uv=True,
    export_normals=True
)
print(f"✓ Exported OBJ for RigNet: {temp_obj_path}")
# Verify file exists
if not os.path.exists(temp_obj_path):
    raise FileNotFoundError(f"Failed to export OBJ file: {temp_obj_path}")

# Step 2: Run RigNet prediction
print("\\n=== Step 2: Running RigNet Prediction ===")

# Try to handle torch_scatter import errors gracefully for CUDA
# If CUDA version fails, the error will propagate but Erlang will handle it
try:
    import torch_scatter
    print("✓ torch_scatter loaded")
except Exception as e:
    print(f"Warning: torch_scatter import failed: {e}")
    print("  This may be due to missing CUDA dependencies")
    print("  Attempting to continue...")
    # Let the error propagate - Erlang will handle it
    raise

# Import RigNet modules
try:
    # Import the module first so we can set its global variables
    import RigNet.quick_start as quick_start_module
    from RigNet.quick_start import (
        create_single_data,
        predict_joints,
        predict_skeleton,
        predict_skinning,
    )
    from RigNet.models.GCN import JOINTNET_MASKNET_MEANSHIFT as JOINTNET
    from RigNet.models.ROOT_GCN import ROOTNET
    from RigNet.models.PairCls_GCN import PairCls as BONENET
    from RigNet.models.SKINNING import SKINNET
    print("✓ RigNet modules imported")
except Exception as e:
    print(f"Error importing RigNet modules: {e}")
    import traceback
    traceback.print_exc()
    # Let Erlang handle the error
    raise

# Load networks
print("\\n=== Loading Networks ===")
# Save original working directory
original_cwd = os.getcwd()
# Change to checkpoint directory for loading models
checkpoint_parent = Path(checkpoint_dir).parent.resolve()
os.chdir(str(checkpoint_parent))

# Helper function to load state dict with error handling
def load_state_dict_safe(model, checkpoint_path, model_name):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Try strict loading first
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ {model_name} loaded (strict)")
        except RuntimeError as e:
            # If strict fails, try non-strict and filter out unexpected keys
            print(f"Warning: {model_name} strict loading failed")
            print(f"  Attempting non-strict loading...")

            # Get model's expected keys
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())

            # Filter state_dict to only include keys that exist in model
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys

            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)} (will use random initialization)")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)} (will be ignored)")

            model.load_state_dict(filtered_dict, strict=False)
            print(f"✓ {model_name} loaded (non-strict)")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        import traceback
        traceback.print_exc()
        raise

# Joint network - use gcn_meanshift (finetuned) instead of pretrain_jointnet
jointNet = JOINTNET()
jointNet.to(device)
jointNet.eval()
joint_checkpoint_path = 'checkpoints/gcn_meanshift/model_best.pth.tar'
if not os.path.exists(joint_checkpoint_path):
    # Fallback to pretrain_jointnet if gcn_meanshift doesn't exist
    joint_checkpoint_path = 'checkpoints/pretrain_jointnet/model_best.pth.tar'
load_state_dict_safe(jointNet, joint_checkpoint_path, "Joint prediction network")
print("✓ Joint prediction network loaded")

# Root network
rootNet = ROOTNET()
rootNet.to(device)
rootNet.eval()
load_state_dict_safe(rootNet, 'checkpoints/rootnet/model_best.pth.tar', "Root prediction network")
print("✓ Root prediction network loaded")

# Bone network
boneNet = BONENET()
boneNet.to(device)
boneNet.eval()
load_state_dict_safe(boneNet, 'checkpoints/bonenet/model_best.pth.tar', "Bone connectivity network")
print("✓ Bone connectivity network loaded")

# Skin network
skinNet = SKINNET(nearest_bone=5, use_Dg=True, use_Lf=True)
skinNet.to(device)
skinNet.eval()
load_state_dict_safe(skinNet, 'checkpoints/skinnet/model_best.pth.tar', "Skinning network")
print("✓ Skinning network loaded")

# Create input data from OBJ
print("\\n=== Processing Mesh for RigNet ===")
# Use absolute path (already resolved before chdir)
# Convert to string and normalize path separators for cross-platform compatibility
temp_obj_absolute = os.path.normpath(temp_obj_path)
print(f"Loading OBJ from: {temp_obj_absolute}")
if not os.path.exists(temp_obj_absolute):
    raise FileNotFoundError(f"OBJ file not found: {temp_obj_absolute}")
print(f"OBJ file exists: {os.path.exists(temp_obj_absolute)}, size: {os.path.getsize(temp_obj_absolute)} bytes")
data, vox, surface_geodesic, translation_normalize, scale_normalize = create_single_data(temp_obj_absolute)
data.to(device)

# Predict joints
print("\\n=== Predicting Joints ===")
data = predict_joints(data, vox, jointNet, threshold, bandwidth=bandwidth)
data.to(device)

# Predict skeleton
print("\\n=== Predicting Skeleton ===")
# Construct normalized OBJ path for predict_skeleton (required parameter)
normalized_obj_path = temp_obj_absolute.replace('.obj', '_normalized.obj')
pred_skeleton = predict_skeleton(data, vox, rootNet, boneNet, normalized_obj_path)

# Predict skinning
print("\\n=== Predicting Skinning Weights ===")
# Set device as global variable in RigNet module for predict_skinning
quick_start_module.device = device
pred_rig = predict_skinning(data, pred_skeleton, skinNet, surface_geodesic, normalized_obj_path)

# Reverse normalization
pred_rig.normalize(scale_normalize, -translation_normalize)

# Save rig file temporarily
# Use absolute path for rig file
temp_rig_path = str(Path(temp_obj_path).with_suffix('').resolve()) + '_rig.txt'
pred_rig.save(temp_rig_path)
print(f"✓ Rig saved to: {temp_rig_path}")

# Clear GPU cache
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Restore original working directory before returning to Blender
os.chdir(original_cwd)

# Step 3: Load rig back into Blender and apply to mesh
print("\\n=== Step 3: Applying Rig to Mesh ===")

# Use blender-rignet's rig parser and armature generator
try:
    from RigNet.utils.rig_parser import Info
    from ob_utils.objects import ArmatureGenerator

    # Parse rig file - use absolute path
    skel_info = Info(filename=str(Path(temp_rig_path).resolve()))
    print("✓ Rig file parsed")

    # Create armature and apply to mesh
    # The mesh_obj is already in the scene from Step 1
    bpy.context.view_layer.objects.active = mesh_obj
    ArmatureGenerator(skel_info, mesh_obj).generate()
    print("✓ Armature created and applied to mesh")
except Exception as e:
    print(f"Error applying rig: {e}")
    import traceback
    traceback.print_exc()
    raise

# Step 4: Export final rigged model as USDC
print("\\n=== Step 4: Exporting Rigged Model as USDC ===")
bpy.ops.wm.usd_export(
    filepath=output_path,
    export_materials=True,
    export_textures=True,
    relative_paths=False,
    export_uvmaps=True,
    export_armatures=True,
    selected_objects_only=False,
    visible_objects_only=False,
    use_instancing=False,
    evaluation_mode='RENDER'
)
print(f"✓ Rigged USDC exported: {output_path}")

# Cleanup temp files
try:
    os.remove(temp_obj_path)
    os.remove(temp_rig_path)
    print("✓ Cleaned up temporary files")
except:
    pass

print("\\n✓ USDC pipeline complete!")
""", %{})

IO.puts("\n✓ RigNet generation complete!")
IO.puts("Output saved to: #{output_path}")
