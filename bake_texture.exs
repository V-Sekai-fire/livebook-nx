#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Texture Baking Script
# Bakes textures from Gaussian splats onto mesh GLB files
#
# Usage:
#   elixir bake_texture.exs --mesh <mesh_glb> --gaussian <gaussian_ply> [--output <output_dir>]
#   elixir bake_texture.exs <mesh_glb> <gaussian_ply> [output_dir]
#   elixir bake_texture.exs output/20251210_13_29_36/replicate-prediction-ybg1dpnme1rm80cv15stz3qct8_processed

# Configure OpenTelemetry for console-only logging
Application.put_env(:opentelemetry, :span_processor, :batch)
Application.put_env(:opentelemetry, :traces_exporter, :none)
Application.put_env(:opentelemetry, :metrics_exporter, :none)
Application.put_env(:opentelemetry, :logs_exporter, :none)

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
])

Logger.configure(level: :info)

# Load shared utilities
Code.eval_file("shared_utils.exs")

# Initialize OpenTelemetry
OtelSetup.configure()

defmodule TextureBaking do
  def show_help do
    IO.puts("""
    Texture Baking Script
    Bakes textures from Gaussian splats onto mesh GLB files
    
    Usage:
      elixir bake_texture.exs --mesh <mesh_glb> --gaussian <gaussian_ply> [--output <output_dir>]
      elixir bake_texture.exs <mesh_glb> <gaussian_ply> [output_dir]
      elixir bake_texture.exs <output_dir>  # Auto-detect mesh_segment.glb and merged_gs.ply
    
    Options:
      --mesh, -m <path>        Path to mesh GLB file
      --gaussian, -g <path>    Path to Gaussian PLY file
      --output, -o <path>      Output directory (default: directory of mesh file)
      --texture-size <int>     Texture size for baking (default: 1024)
      --simplify <float>       Mesh simplification ratio (default: 0.3)
      --help, -h               Show this help message
    
    Examples:
      elixir bake_texture.exs mesh_segment.glb merged_gs.ply
      elixir bake_texture.exs --mesh mesh.glb --gaussian gs.ply --output results/
      elixir bake_texture.exs output/20251210_13_29_36/replicate-prediction-ybg1dpnme1rm80cv15stz3qct8_processed
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        mesh: :string,
        gaussian: :string,
        output: :string,
        texture_size: :integer,
        simplify: :float,
        help: :boolean
      ],
      aliases: [
        m: :mesh,
        g: :gaussian,
        o: :output,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    # Get paths from options or positional arguments
    mesh_path_raw = Keyword.get(opts, :mesh) || Enum.at(args, 0)
    gaussian_path_raw = Keyword.get(opts, :gaussian) || Enum.at(args, 1)
    output_dir_raw = Keyword.get(opts, :output) || Enum.at(args, 2)

    # Handle backward compatibility: single argument as output directory
    {mesh_path, gaussian_path, output_dir} = if mesh_path_raw && !gaussian_path_raw && !output_dir_raw && File.dir?(mesh_path_raw) do
      output = Path.expand(mesh_path_raw)
      {
        Path.join(output, "mesh_segment.glb"),
        Path.join(output, "merged_gs.ply"),
        output
      }
    else
      # Expand paths
      mesh_expanded = if mesh_path_raw, do: Path.expand(mesh_path_raw), else: nil
      gaussian_expanded = if gaussian_path_raw, do: Path.expand(gaussian_path_raw), else: nil
      
      # Use directory of mesh file as output directory if not specified
      output_expanded = if output_dir_raw do
        Path.expand(output_dir_raw)
      else
        if mesh_expanded, do: Path.dirname(mesh_expanded), else: nil
      end
      
      {mesh_expanded, gaussian_expanded, output_expanded}
    end

    if !mesh_path do
      IO.puts("""
      Error: Mesh file path is required.
      
      Usage:
        elixir bake_texture.exs --mesh <mesh_glb> --gaussian <gaussian_ply> [--output <output_dir>]
        elixir bake_texture.exs <mesh_glb> <gaussian_ply> [output_dir]
      
      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    if !gaussian_path do
      IO.puts("""
      Error: Gaussian file path is required.
      
      Usage:
        elixir bake_texture.exs --mesh <mesh_glb> --gaussian <gaussian_ply> [--output <output_dir>]
        elixir bake_texture.exs <mesh_glb> <gaussian_ply> [output_dir]
      
      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    # Check if files exist
    if !File.exists?(mesh_path) do
      IO.puts("Error: Mesh file not found: #{mesh_path}")
      System.halt(1)
    end

    if !File.exists?(gaussian_path) do
      IO.puts("Error: Gaussian file not found: #{gaussian_path}")
      System.halt(1)
    end

    # Ensure output directory exists
    if output_dir do
      File.mkdir_p!(output_dir)
    end

    texture_size = Keyword.get(opts, :texture_size, 1024)
    simplify = Keyword.get(opts, :simplify, 0.3)

    {mesh_path, gaussian_path, output_dir, texture_size, simplify}
  end
end

# Initialize Python environment with required dependencies
Pythonx.uv_init("""
[project]
name = "omnipart-generation"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "fast_simplification",
  "torch==2.4.0",
  "torchvision==0.19.0",
  "pillow==10.4.0",
  "imageio==2.36.1",
  "imageio-ffmpeg==0.5.1",
  "tqdm==4.67.1",
  "easydict==1.13",
  "opencv-python-headless==4.10.0.84",
  "scipy==1.14.1",
  "onnxruntime==1.20.1",
  "trimesh==4.5.3",
  "xatlas==0.0.9",
  "pyvista==0.44.2",
  "pymeshfix==0.17.0",
  "igraph==0.11.8",
  "xformers==0.0.27.post2",
  "numpy==1.26.4",  # Pin to 1.26.4 to fix spconv SIGFPE with CUDA 12.1 (NumPy 2.0+ incompatible)
  "spconv-cu120==2.3.6",
  "transformers @ git+https://github.com/huggingface/transformers.git@ff13eb668aa03f151ded71636d723f2e490ad967",
  "pydantic==2.10.6",
  "diffusers==0.32.0",
  "lightning==2.2",
  "mesh2sdf",
  "loguru",
  "tetgen==0.6.3",
  "omegaconf",
  "pycocotools",
  "kornia",
  "timm",
  "h5py",
  "boto3",
  "einops",
  "pytz",
  "scikit-image",
  "plyfile",
  "psutil",
  "transparent-background>=1.3.4",  # Free open-source background removal using InSPyReNet (ACCV 2022)
  "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux'",
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
  "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
  "torch_scatter @ https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_scatter-2.1.2%2Bpt24cu121-cp310-cp310-win_amd64.whl ; sys_platform == 'win32'",
  "torch_scatter @ https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_scatter-2.1.2%2Bpt24cu121-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux'",
  "detectron2 @ https://github.com/MiroPsota/torch_packages_builder/releases/download/detectron2-0.6%2Bfd27788/detectron2-0.6%2Bfd27788pt2.3.0cu118-cp310-cp310-win_amd64.whl  ; sys_platform == 'win32'",
  "detectron2 @ https://github.com/MiroPsota/torch_packages_builder/releases/download/detectron2-0.6%2Bfd27788/detectron2-0.6%2Bfd27788pt2.3.0cu118-cp310-cp310-linux_x86_64.whl  ; sys_platform == 'linux'",
  "diff_gaussian_rasterization @ https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl?download=true",
  "pytorch3d @ https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9+pt2.4.0cu121-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux'",
  "pytorch3d @ https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9+pt2.4.0cu121-cp310-cp310-win_amd64.whl ; sys_platform == 'win32'",
  "nvdiffrast @ https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl",
]

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[[tool.uv.index]]
name = "pyg"
url = "https://data.pyg.org/whl/torch-2.4.0+cu121.html"
explicit = true
""")

# Parse arguments
{mesh_glb, gaussian_ply, output_dir, texture_size, simplify} = TextureBaking.parse(System.argv())

IO.puts("=== Texture Baking ===")
IO.puts("Mesh file: #{mesh_glb}")
IO.puts("Gaussian file: #{gaussian_ply}")
IO.puts("Output directory: #{output_dir}")
IO.puts("Texture size: #{texture_size}")
IO.puts("Simplify ratio: #{simplify}")

# Create configuration file for Python
config_with_paths = %{
  "mesh_glb" => mesh_glb,
  "gaussian_ply" => gaussian_ply,
  "output_dir" => output_dir,
  "texture_size" => texture_size,
  "simplify" => simplify,
  "omnipart_dir" => Path.expand("thirdparty/OmniPart")
}

{config_file, config_file_normalized} = ConfigFile.create(config_with_paths, "texture_baking_config")

# Run texture baking test
# Execute the baking function directly (SpanCollector is optional and handled separately)
try do
  {_, _python_globals} = Pythonx.eval(~S"""
import sys
import os
import torch
import numpy as np
import trimesh
import json
from pathlib import Path

# Get configuration from JSON file
""" <> ConfigFile.python_path_string(config_file_normalized) <> ~S"""

mesh_glb = config.get('mesh_glb')
gaussian_ply = config.get('gaussian_ply')
output_dir = config.get('output_dir')
texture_size = config.get('texture_size')
simplify = config.get('simplify')
omnipart_dir = config.get('omnipart_dir')

# Add OmniPart to path
omnipart_path = Path(omnipart_dir)
sys.path.insert(0, str(omnipart_path))

from modules.part_synthesis.representations.mesh import MeshExtractResult
from modules.part_synthesis.representations.gaussian.gaussian_model import Gaussian
from modules.part_synthesis.utils.postprocessing_utils import to_glb

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Resolve paths to absolute
mesh_glb = str(Path(mesh_glb).resolve())
gaussian_ply = str(Path(gaussian_ply).resolve())
output_dir = str(Path(output_dir).resolve())

print(f"Loading mesh from: {mesh_glb}")
trimesh_mesh = trimesh.load(mesh_glb, force='mesh', process=False)
if isinstance(trimesh_mesh, trimesh.Scene):
    # If it's a scene, get the first mesh
    trimesh_mesh = list(trimesh_mesh.geometry.values())[0]

vertices = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(trimesh_mesh.faces, dtype=torch.long, device=device)

# Create MeshExtractResult
mesh_result = MeshExtractResult(
    vertices=vertices,
    faces=faces,
    vertex_attrs=None,
    res=64
)

print(f"Loaded mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

# Load Gaussian from PLY
print(f"Loading Gaussian from: {gaussian_ply}")
# Calculate AABB from mesh - aabb should be flat: [min_x, min_y, min_z, max_x, max_y, max_z]
aabb_min = vertices.min(dim=0)[0].cpu().numpy()
aabb_max = vertices.max(dim=0)[0].cpu().numpy()
# Flatten to [min_x, min_y, min_z, max_x, max_y, max_z]
aabb = np.concatenate([aabb_min, aabb_max]).tolist()

gaussian = Gaussian(
    aabb=aabb,
    sh_degree=0,
    device=device
)

# Load without transform to avoid tensor shape issues
gaussian.load_ply(gaussian_ply, transform=None)
print(f"Loaded Gaussian: {gaussian.get_xyz.shape[0]} points")

# Test texture baking
output_glb = os.path.join(output_dir, "mesh_textured_test.glb")
print(f"\n=== Baking Texture ===")
print(f"Output will be saved to: {output_glb}")

try:
    textured_mesh = to_glb(
        app_rep=gaussian,
        mesh=mesh_result,
        simplify=simplify,
        fill_holes=False,  # Disable hole filling - it's causing issues with visibility detection
        fill_holes_max_size=0.04,
        texture_size=texture_size,
        debug=False,
        verbose=True,
        textured=True,  # Enable texture baking
    )
    
    if textured_mesh is not None:
        # Remove existing file if it exists
        if os.path.exists(output_glb):
            os.remove(output_glb)
        
        # Export textured mesh
        textured_mesh.export(output_glb)
        print(f"\n[SUCCESS] Textured mesh saved to: {output_glb}")
        print(f"Mesh has texture: {hasattr(textured_mesh.visual, 'material')}")
    else:
        print("\n[ERROR] Texture baking returned None")
        sys.exit(1)
        
except Exception as e:
    print(f"\n[ERROR] Texture baking failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
""", %{})
rescue
  e ->
    ConfigFile.cleanup(config_file)
    reraise e, __STACKTRACE__
after
  # Clean up temp file
  ConfigFile.cleanup(config_file)
end

IO.puts("\n=== Complete ===")
