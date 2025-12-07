#!/usr/bin/env elixir

# SAM 3 Object Generation Script
# Generate 3D objects from pre-created mask videos
#
# Usage:
#   elixir sam3_object_generation.exs <mask_video_path> [options]
#
# Options:
#   --output-format "ply"          Output format: glb, ply, obj (default: "ply")

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"}
])

# Initialize Python environment with required dependencies
Pythonx.uv_init("""
[project]
name = "sam3-object-generation"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "numpy<2.0",
  "pillow",
  "opencv-python",
  "open3d>=0.18.0",
  "scipy>=1.10.0",
  "scikit-image>=0.21.0",
  "trimesh>=3.23.0",
]
""")

# Parse command-line arguments
defmodule ArgsParser do
  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        output_format: :string
      ],
      aliases: [
        f: :output_format
      ]
    )

    input_path = List.first(args)

    if !input_path do
      IO.puts("""
      Error: Input video or mask video path is required.

      Usage:
        elixir sam3_object_generation.exs <input_video_path> [options]

      Options:
        --output-format, -f "ply"          Output format: glb, ply, obj (default: "ply")
      
      Note: If input_video_path doesn't end with '_mask.mp4', the script will
            automatically look for a file with '_mask.mp4' suffix.
      """)
      System.halt(1)
    end

    # Auto-detect mask video path
    mask_video_path = if String.ends_with?(input_path, "_mask.mp4") do
      input_path
    else
      # Try to find the mask video by appending _mask before the extension
      base = Path.rootname(input_path)
      ext = Path.extname(input_path)
      candidate = "#{base}_mask#{ext}"
      
      if File.exists?(candidate) do
        candidate
      else
        # If not found, use the input path directly (assume it's already a mask video)
        input_path
      end
    end

    config = %{
      input_path: input_path,
      mask_video_path: mask_video_path,
      output_format: Keyword.get(opts, :output_format, "ply")
    }

    # Validate output_format
    valid_formats = ["glb", "ply", "obj"]
    if config.output_format not in valid_formats do
      IO.puts("Error: Invalid output format. Must be one of: #{Enum.join(valid_formats, ", ")}")
      System.halt(1)
    end

    # Check if mask video file exists
    if !File.exists?(config.mask_video_path) do
      IO.puts("Error: Mask video file not found: #{config.mask_video_path}")
      IO.puts("       Tried to find mask video from input: #{input_path}")
      System.halt(1)
    end

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== 3D Object Generation from Mask Video ===
Input: #{config.input_path}
Mask Video: #{config.mask_video_path}
Output Format: #{config.output_format}
""")

# Save config to JSON for Python to read
config_json = Jason.encode!(config)
File.write!("config.json", config_json)

# Import libraries and process mask video
{_, python_globals} = Pythonx.eval("""
import json
import cv2
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom
from skimage import measure as sk_measure

# Get configuration from JSON file
with open("config.json", 'r') as f:
    config = json.load(f)

mask_video_path = config.get('mask_video_path')
output_format = config.get('output_format', 'ply')

# Resolve input path to absolute path
mask_video_path = str(Path(mask_video_path).resolve())

if not Path(mask_video_path).exists():
    raise FileNotFoundError(f"Mask video file not found: {mask_video_path}")

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

print("\\n=== Step 1: Load Mask Video ===")
print(f"Loading mask video: {mask_video_path}")

# Load mask video frames
cap = cv2.VideoCapture(mask_video_path)
if not cap.isOpened():
    raise ValueError(f"Could not open mask video file: {mask_video_path}")

mask_frames = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to grayscale (masks are typically grayscale)
    if len(frame.shape) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame
    
    mask_frames.append(frame_gray)
    frame_count += 1

cap.release()

if not mask_frames:
    raise ValueError(f"Could not load frames from mask video: {mask_video_path}")

print(f"✓ Loaded {len(mask_frames)} mask frames")

# Convert frames to numpy arrays and normalize to 0-1
all_masks = []
for idx, mask_frame in enumerate(mask_frames):
    # Normalize mask to 0-1 range
    mask = mask_frame.astype(np.float32) / 255.0
    # Threshold to binary mask
    mask = (mask > 0.5).astype(np.float32)
    all_masks.append((idx, [mask]))

print(f"✓ Processed {len(all_masks)} mask frames")

print("\\n=== Step 2: Generate 3D Objects ===")

# Extract masks from all frames
print("Extracting masks from mask video...")

if not all_masks:
    print("⚠ No masks found in mask video.")
    raise ValueError("No masks available for 3D reconstruction")

print(f"✓ Extracted masks from {len(all_masks)} frames")

# Get the first mask to determine dimensions
_, first_masks = all_masks[0]
if len(first_masks) == 0:
    raise ValueError("No masks found in first frame")

# Use the first mask
first_mask = first_masks[0]
if first_mask.ndim > 2:
    first_mask = first_mask.squeeze()

mask_height, mask_width = first_mask.shape
print(f"Mask dimensions: {mask_height}x{mask_width}")

# Create 3D reconstruction using Gaussian Splatting-inspired approach
print("\\nReconstructing 3D object from masks using point cloud to mesh conversion...")

# Step 1: Convert masks to point cloud with Gaussian-like distribution
print("Converting masks to point cloud...")
points_3d = []

# Scale factor for 3D coordinates
scale_factor = 0.01  # 1cm per pixel unit

# Video: multi-view point cloud reconstruction
print(f"Processing {len(all_masks)} frames for multi-view reconstruction...")
for layer_idx, (frame_idx, masks) in enumerate(all_masks):
    if len(masks) > 0:
        mask = masks[0]
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Sample points from mask
        y_coords, x_coords = np.where(mask_binary > 0)
        num_points = min(1000, len(y_coords))
        if num_points > 0:
            indices = np.random.choice(len(y_coords), num_points, replace=False)
            for idx in indices:
                y, x = y_coords[idx], x_coords[idx]
                # Z position based on frame index
                z = (layer_idx - len(all_masks) / 2) * scale_factor * mask_height * 0.5
                x_norm = (x / mask.shape[1] - 0.5) * mask_width * scale_factor
                y_norm = (y / mask.shape[0] - 0.5) * mask_height * scale_factor
                points_3d.append([x_norm, y_norm, z])

if not points_3d:
    raise ValueError("Could not generate 3D points from masks")

points_array = np.array(points_3d, dtype=np.float32)
print(f"✓ Generated {len(points_array)} points")

# Step 2: Create Open3D point cloud and estimate normals
print("Estimating normals for point cloud...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_array)

# Estimate normals using KNN
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(k=15)

print(f"✓ Estimated normals for {len(pcd.points)} points")

# Step 3: Poisson surface reconstruction (best method for point clouds with normals)
print("\\nReconstructing mesh using Poisson surface reconstruction...")
print("This is the industry-standard method for converting point clouds to meshes.")

try:
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False
    )
    
    # Remove low density vertices (outliers)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
    
    # Clean up mesh
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()
    
    print(f"✓ Generated mesh with {len(mesh_o3d.vertices)} vertices and {len(mesh_o3d.triangles)} faces")
    
    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vertex_normals = np.asarray(mesh_o3d.vertex_normals) if mesh_o3d.has_vertex_normals() else None
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vertex_normals)
    
except Exception as e:
    print(f"⚠ Poisson reconstruction failed: {e}")
    print("Falling back to Alpha Shapes (convex hull alternative)...")
    
    try:
        # Fallback: Alpha shapes (better than convex hull for concave shapes)
        alpha = 0.03
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        
        if len(mesh_o3d.vertices) == 0:
            raise ValueError("Alpha shape produced empty mesh")
        
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print(f"✓ Generated mesh with {len(vertices)} vertices and {len(faces)} faces using Alpha Shapes")
        
    except Exception as e2:
        print(f"⚠ Alpha shapes failed: {e2}")
        print("Falling back to Ball Pivoting Algorithm...")
        
        try:
            # Another fallback: Ball Pivoting Algorithm
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector([radius, radius * 2])
            )
            
            vertices = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            print(f"✓ Generated mesh with {len(vertices)} vertices and {len(faces)} faces using Ball Pivoting")
            
        except Exception as e3:
            print(f"⚠ All mesh reconstruction methods failed: {e3}")
            print("Creating point cloud mesh as last resort...")
            # Last resort: point cloud
            mesh = trimesh.PointCloud(vertices=points_array)

# Clean up mesh (only for triangle meshes, not point clouds)
if isinstance(mesh, trimesh.Trimesh):
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_unreferenced_vertices'):
        mesh.remove_unreferenced_vertices()
    if hasattr(mesh, 'fix_normals'):
        mesh.fix_normals()

# Export to PLY
output_filename = f"output_3d.{output_format}"
output_path = output_dir / output_filename

print(f"\\nExporting 3D object to {output_format.upper()} format...")
mesh.export(str(output_path))
print(f"✓ Saved 3D object to: {output_path}")
print(f"  - Vertices: {len(mesh.vertices)}")
if isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, 'faces'):
    print(f"  - Faces: {len(mesh.faces)}")
else:
    print(f"  - Type: Point Cloud")
print(f"  - Format: {output_format.upper()}")
print(f"\\n✓ 3D reconstruction complete using Gaussian Splatting-inspired point cloud to mesh conversion!")
""", %{})

IO.puts("\n=== Complete ===")
IO.puts("3D object generation completed successfully!")
