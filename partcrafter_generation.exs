#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# PartCrafter Generation Script
# Generate structured 3D mesh parts using PartCrafter
# Repository: https://github.com/wgsxm/PartCrafter
# Hugging Face: https://huggingface.co/wgsxm/PartCrafter
#
# Usage:
#   elixir partcrafter_generation.exs <image_path> [options]
#
# Options:
#   --output-format "glb"        Output format: glb (default: "glb")
#   --num-parts <int>            Number of parts to generate (1-16, default: 6)
#   --seed <int>                 Random seed for generation (default: 0)
#   --num-tokens <int>           Number of tokens (256/512/1024/1536/2048, default: 1024)
#   --num-steps <int>            Number of inference steps (default: 50)
#   --guidance-scale <float>     Guidance scale (default: 7.0)
#   --use-flash-decoder          Use flash decoder for faster inference (default: true)

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
  {:req, "~> 0.5.0"}
])

# Suppress debug logs from Req to avoid showing long URLs
Logger.configure(level: :info)

# Initialize Python environment with required dependencies
# PartCrafter uses Hugging Face models and diffusers
# All dependencies managed by uv (no pip)
# Based on official requirements: https://github.com/wgsxm/PartCrafter
Pythonx.uv_init("""
[project]
name = "partcrafter-generation"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "scikit-learn",
  "gpustat",
  "nvitop",
  "diffusers",
  "transformers",
  "einops",
  "huggingface-hub",
  "opencv-python",
  "trimesh",
  "omegaconf",
  "scikit-image",
  "numpy==1.26.4",
  "peft",
  "jaxtyping",
  "typeguard",
  "matplotlib",
  "imageio-ffmpeg",
  "pyrender",
  "wandb[media]",
  "colormaps",
  "gradio==5.35.0",
  "accelerate",
  "pillow",
  "torch",
  "torchvision",
  # torch-cluster: platform-specific wheel URLs using environment markers
  # Match torch 2.4.x version (pytorch-cu118 index installs 2.4.1)
  # Windows: win_amd64 wheel for torch 2.4.0+cu118
  "torch-cluster @ https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_cluster-1.6.3%2Bpt24cu118-cp310-cp310-win_amd64.whl ; sys_platform == 'win32'",
  # Linux: linux_x86_64 wheel for torch 2.4.0+cu118
  "torch-cluster @ https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_cluster-1.6.3%2Bpt24cu118-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux'",
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
    PartCrafter Generation Script
    Generate structured 3D mesh parts using PartCrafter
    Repository: https://github.com/wgsxm/PartCrafter
    Hugging Face: https://huggingface.co/wgsxm/PartCrafter

    Usage:
      elixir partcrafter_generation.exs <image_path> [options]

    Options:
      --output-format, -f "glb"      Output format: glb (default: "glb")
      --num-parts, -n <int>           Number of parts to generate (1-16, default: 6)
      --seed, -s <int>                Random seed for generation (default: 0)
      --num-tokens, -t <int>          Number of tokens: 256, 512, 1024, 1536, 2048 (default: 1024)
      --num-steps, --steps <int>      Number of inference steps (default: 50)
      --guidance-scale, -g <float>    Guidance scale (default: 7.0)
      --use-flash-decoder             Use flash decoder for faster inference (default: true)
      --help, -h                       Show this help message

    Example:
      elixir partcrafter_generation.exs image.jpg --num-parts 8 --seed 42
      elixir partcrafter_generation.exs image.png -n 6 -s 0 -t 1024
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        output_format: :string,
        num_parts: :integer,
        seed: :integer,
        num_tokens: :integer,
        num_steps: :integer,
        guidance_scale: :float,
        use_flash_decoder: :boolean,
        help: :boolean
      ],
      aliases: [
        f: :output_format,
        n: :num_parts,
        s: :seed,
        t: :num_tokens,
        steps: :num_steps,
        g: :guidance_scale,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    image_path = List.first(args)

    if !image_path do
      IO.puts("""
      Error: Image path is required.

      Usage:
        elixir partcrafter_generation.exs <image_path> [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    num_parts = Keyword.get(opts, :num_parts, 6)
    if num_parts < 1 or num_parts > 16 do
      IO.puts("Error: num_parts must be between 1 and 16")
      System.halt(1)
    end

    num_tokens = Keyword.get(opts, :num_tokens, 1024)
    valid_tokens = [256, 512, 1024, 1536, 2048]
    if num_tokens not in valid_tokens do
      IO.puts("Error: num_tokens must be one of: #{Enum.join(valid_tokens, ", ")}")
      System.halt(1)
    end

    num_steps = Keyword.get(opts, :num_steps, 50)
    if num_steps < 1 do
      IO.puts("Error: num_steps must be at least 1")
      System.halt(1)
    end

    guidance_scale = Keyword.get(opts, :guidance_scale, 7.0)
    if guidance_scale < 0.0 do
      IO.puts("Error: guidance_scale must be non-negative")
      System.halt(1)
    end

    config = %{
      image_path: image_path,
      output_format: Keyword.get(opts, :output_format, "glb"),
      num_parts: num_parts,
      seed: Keyword.get(opts, :seed, 0),
      num_tokens: num_tokens,
      num_steps: num_steps,
      guidance_scale: guidance_scale,
      use_flash_decoder: Keyword.get(opts, :use_flash_decoder, true)
    }

    # Validate output_format
    valid_formats = ["glb"]
    if config.output_format not in valid_formats do
      IO.puts("Error: Invalid output format. Must be: #{Enum.join(valid_formats, ", ")}")
      System.halt(1)
    end

    # Check if file exists
    if !File.exists?(config.image_path) do
      IO.puts("Error: Image file not found: #{config.image_path}")
      System.halt(1)
    end

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== PartCrafter Generation ===
Image: #{config.image_path}
Output Format: #{config.output_format}
Number of Parts: #{config.num_parts}
Seed: #{config.seed}
Number of Tokens: #{config.num_tokens}
Inference Steps: #{config.num_steps}
Guidance Scale: #{config.guidance_scale}
Use Flash Decoder: #{config.use_flash_decoder}
""")

# Add weights directories to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  partcrafter_weights_dir: Path.join([base_dir, "pretrained_weights", "PartCrafter"])
})

# Save config to JSON for Python to read (use temp file to avoid conflicts)
config_json = Jason.encode!(config_with_paths)
# Use cross-platform temp directory
tmp_dir = System.tmp_dir!()
File.mkdir_p!(tmp_dir)
config_file = Path.join(tmp_dir, "partcrafter_config_#{System.system_time(:millisecond)}.json")
File.write!(config_file, config_json)
config_file_normalized = String.replace(config_file, "\\", "/")

# Elixir-native Hugging Face download function
defmodule HuggingFaceDownloader do
  @base_url "https://huggingface.co"
  @api_base "https://huggingface.co/api"

  def download_repo(repo_id, local_dir, repo_name \\ "model") do
    IO.puts("Downloading #{repo_name}...")

    # Create directory
    File.mkdir_p!(local_dir)

    # Get file tree from Hugging Face API
    case get_file_tree(repo_id) do
      {:ok, files} ->
        # files is a map, convert to list for counting and iteration
        files_list = Map.to_list(files)
        total = length(files_list)
        IO.puts("Found #{total} files to download")

        files_list
        |> Enum.with_index(1)
        |> Enum.each(fn {{path, info}, index} ->
          download_file(repo_id, path, local_dir, info, index, total)
        end)

        IO.puts("\n[OK] #{repo_name} downloaded")
        {:ok, local_dir}

      {:error, reason} ->
        IO.puts("[ERROR] #{repo_name} download failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp get_file_tree(repo_id, revision \\ "main") do
    # Recursively get all files
    case get_files_recursive(repo_id, revision, "") do
      {:ok, files} ->
        file_map =
          files
          |> Enum.map(fn file ->
            {file["path"], file}
          end)
          |> Map.new()

        {:ok, file_map}

      error ->
        error
    end
  end

  defp get_files_recursive(repo_id, revision, path) do
    # Build URL - handle empty path correctly
    url = if path == "" do
      "#{@api_base}/models/#{repo_id}/tree/#{revision}"
    else
      "#{@api_base}/models/#{repo_id}/tree/#{revision}/#{path}"
    end

    try do
      response = Req.get(url)

      # Req.get returns response directly or wrapped in tuple
      items = case response do
        {:ok, %{status: 200, body: body}} when is_list(body) -> body
        %{status: 200, body: body} when is_list(body) -> body
        {:ok, %{status: status}} ->
          raise "API returned status #{status}"
        %{status: status} ->
          raise "API returned status #{status}"
        {:error, reason} ->
          raise inspect(reason)
        other ->
          raise "Unexpected response: #{inspect(other)}"
      end

      files = Enum.filter(items, &(&1["type"] == "file"))
      dirs = Enum.filter(items, &(&1["type"] == "directory"))

      # Recursively get files from subdirectories
      subdir_files =
        dirs
        |> Enum.flat_map(fn dir ->
          case get_files_recursive(repo_id, revision, dir["path"]) do
            {:ok, subfiles} -> subfiles
            _ -> []
          end
        end)

      {:ok, files ++ subdir_files}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp download_file(repo_id, path, local_dir, info, current, total) do
    # Construct download URL (using resolve endpoint for LFS files)
    url = "#{@base_url}/#{repo_id}/resolve/main/#{path}"
    local_path = Path.join(local_dir, path)

    # Get file size for progress display
    file_size = info["size"] || 0
    size_mb = if file_size > 0, do: Float.round(file_size / 1024 / 1024, 1), else: 0

    # Show current file being downloaded
    filename = Path.basename(path)
    IO.write("\r  [#{current}/#{total}] Downloading: #{filename} (#{size_mb} MB)")

    # Skip if file already exists
    if File.exists?(local_path) do
      IO.write("\r  [#{current}/#{total}] Skipped (exists): #{filename}")
    else
      # Create parent directories
      local_path
      |> Path.dirname()
      |> File.mkdir_p!()

      # Download file with streaming, suppress debug logs
      result = Req.get(url,
        into: File.stream!(local_path, [], 65536),
        retry: :transient,
        max_redirects: 10
      )

      case result do
        {:ok, %{status: 200}} ->
          IO.write("\r  [#{current}/#{total}] ✓ #{filename}")

        %{status: 200} ->
          IO.write("\r  [#{current}/#{total}] ✓ #{filename}")

        {:ok, %{status: status}} ->
          IO.puts("\n[WARN] Failed to download #{path}: status #{status}")

        %{status: status} ->
          IO.puts("\n[WARN] Failed to download #{path}: status #{status}")

        {:error, reason} ->
          IO.puts("\n[WARN] Failed to download #{path}: #{inspect(reason)}")
      end
    end
  end
end

# Download models using Elixir-native approach
IO.puts("\n=== Step 2: Download Pretrained Weights ===")
IO.puts("Downloading PartCrafter models from Hugging Face...")

base_dir = Path.expand(".")
partcrafter_weights_dir = Path.join([base_dir, "pretrained_weights", "PartCrafter"])

IO.puts("Using weights directory: #{partcrafter_weights_dir}")

# Download PartCrafter weights
case HuggingFaceDownloader.download_repo("wgsxm/PartCrafter", partcrafter_weights_dir, "PartCrafter") do
  {:ok, _} -> :ok
  {:error, _} -> IO.puts("[WARN] PartCrafter download had errors, but continuing...")
end

# Import libraries and process using PartCrafter
try do
  {_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
import tempfile
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from accelerate.utils import set_seed
import trimesh

# Add PartCrafter to path for imports
# Use local PartCrafter from thirdparty directory
# Need to add parent directory so "from src.pipelines..." imports work
partcrafter_path = Path.cwd() / "thirdparty" / "PartCrafter"
if (partcrafter_path / "src").exists():
    sys.path.insert(0, str(partcrafter_path))
    print(f"[OK] Using PartCrafter from: {partcrafter_path}")
else:
    raise FileNotFoundError(
        f"PartCrafter not found at {partcrafter_path / 'src'}. "
        "Please ensure PartCrafter is cloned to thirdparty/PartCrafter"
    )

# Get configuration from JSON file
""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
""" <> ~S"""

image_path = config.get('image_path')
output_format = config.get('output_format', 'glb')
num_parts = config.get('num_parts', 6)
seed = config.get('seed', 0)
num_tokens = config.get('num_tokens', 1024)
num_steps = config.get('num_steps', 50)
guidance_scale = config.get('guidance_scale', 7.0)
use_flash_decoder = config.get('use_flash_decoder', True)

# Get weights directories from config
partcrafter_weights_dir = config.get('partcrafter_weights_dir')

# Fallback to default paths if not in config
if not partcrafter_weights_dir:
    base_dir = Path.cwd()
    partcrafter_weights_dir = str(base_dir / "pretrained_weights" / "PartCrafter")

# Ensure paths are strings
partcrafter_weights_dir = str(Path(partcrafter_weights_dir).resolve())

# Resolve paths to absolute
input_image_path = str(Path(image_path).resolve())

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

print("\n=== Step 1: Load Image ===")

# Check if input_image_path is a video and extract first frame
image_path_str = str(input_image_path)
if image_path_str.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
    print(f"Image path is a video, extracting first frame: {input_image_path}")
    cap = cv2.VideoCapture(image_path_str)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {image_path_str}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame from video: {image_path_str}")

    # Convert BGR to RGB and save as temporary image file
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    temp_image_path = str(output_dir / "temp_extracted_frame.png")
    Image.fromarray(frame_rgb).save(temp_image_path)
    input_image_path = temp_image_path
    print(f"[OK] Extracted frame from video and saved to {input_image_path}")
else:
    print(f"Using image: {input_image_path}")
    # Verify image exists
    if not Path(input_image_path).exists():
        raise FileNotFoundError(f"Image file not found: {input_image_path}")
    print(f"[OK] Image file found: {input_image_path}")

print("\n=== Step 3 Initialize Models ===")
print("Loading PartCrafter pipeline...")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

try:
    # Import PartCrafter components first to ensure custom schedulers are available
    from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
    from src.utils.data_utils import get_colored_mesh_composition
    # Import custom scheduler so it's registered with diffusers
    from src.schedulers import RectifiedFlowScheduler

    # Initialize PartCrafter pipeline
    # Load from local directory (matching PartCrafter's predict.py and app.py)
    print("Loading PartCrafter pipeline from local directory...")
    pipe = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(device, dtype)
    print(f"[OK] PartCrafter pipeline loaded on {device}")

except Exception as e:
    print(f"[ERROR] Error loading models: {e}")
    import traceback
    traceback.print_exc()
    print("\nMake sure you have")
    print("  1. All dependencies installed via uv")
    print("  2. Hugging Face models will be downloaded automatically")
    print("  3. Sufficient GPU memory")
    raise

print("\n=== Step 5 Generate Parts ===")
print(f"Running PartCrafter inference with {num_parts} parts, seed={seed}, guidance_scale={guidance_scale}, steps={num_steps}...")

# Generate random seed if seed is 0
if seed == 0:
    import secrets
    seed = secrets.randbelow(9999) + 1
    print(f"Generated random seed: {seed}")

set_seed(seed)

try:
    # Load image
    img_pil = Image.open(input_image_path)

    # Run inference
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=generator,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=int(1e9),
        use_flash_decoder=use_flash_decoder,
    ).meshes

    # Handle None outputs (decoding errors)
    for i in range(len(outputs)):
        if outputs[i] is None:
            outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])

    print(f"[OK] Generated {len(outputs)} parts successfully")

except Exception as e:
    print(f"[ERROR] Error during generation: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n=== Step 6 Export Meshes ===")

# Create output directory with timestamp
import time
tag = time.strftime("%Y%m%d_%H_%M_%S")
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True)

# Export individual parts
for i, mesh in enumerate(outputs):
    part_num = str(i).zfill(2)
    part_path = export_dir / f"part_{part_num}.{output_format}"
    try:
        mesh.export(str(part_path))
        print(f"[OK] Saved part {i} to {part_path}")
    except Exception as e:
        print(f"[WARN] Could not export part {i} - {e}")

# Create and export merged mesh
try:
    merged_mesh = get_colored_mesh_composition(outputs)
    merged_path = export_dir / f"object.{output_format}"
    merged_mesh.export(str(merged_path))
    print(f"[OK] Saved merged mesh to {merged_path}")
except Exception as e:
    print(f"[WARN] Could not export merged mesh - {e}")

print("\n=== Complete ===")
print(f"Generated {len(outputs)} parts and saved to {export_dir}")
print(f"\nOutput files")
print(f"  - {export_dir}/object.{output_format} (Merged mesh)")
for i in range(len(outputs)):
    part_num = str(i).zfill(2)
    print(f"  - {export_dir}/part_{part_num}.{output_format} (Part {i})")
""", %{})
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

IO.puts("\n=== Complete ===")
IO.puts("3D parts generation completed successfully!")
