#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# OmniPart Generation Script
# Generate part-aware 3D shapes with semantic decoupling and structural cohesion
# Repository: https://github.com/HKU-MMLab/OmniPart
# Hugging Face Space: https://huggingface.co/spaces/omnipart/OmniPart
# Project Page: https://omnipart.github.io/
#
# Usage:
#   elixir omnipart_generation.exs <image_path> <mask_path> [options]
#   elixir omnipart_generation.exs --image <path> --mask <path> [options]
#
# Options:
#   --image, -i <path>            Input image path (required)
#   --mask, -m <path>             Segmentation mask path (.exr file with 2D part IDs) (required)
#   --output-dir, -o <path>       Output directory (default: outputs)
#   --output-format <format>      Output format: glb, obj, ply (default: glb)
#   --gpu <int>                   GPU ID to use (default: 0)
#   --seed <int>                  Random seed (default: 42)

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
# OmniPart uses TRELLIS framework and various 3D processing libraries
# All dependencies managed by uv (no pip)
# Based on official requirements: https://github.com/HKU-MMLab/OmniPart
Pythonx.uv_init("""
[project]
name = "omnipart-generation"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "torch",
  "torchvision",
  "numpy",
  "pillow",
  "opencv-python",
  "trimesh",
  "imageio",
  "imageio-ffmpeg",
  "omegaconf",
  "einops",
  "huggingface-hub",
  "transformers",
  "diffusers",
  "accelerate",
  "tqdm",
  "scipy",
  "scikit-image",
  "gradio",
  # OpenEXR support for .exr mask files
  "OpenEXR",
  "Imath",
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
    OmniPart Generation Script
    Generate part-aware 3D shapes with semantic decoupling and structural cohesion
    
    Repository: https://github.com/HKU-MMLab/OmniPart
    Project Page: https://omnipart.github.io/
    Hugging Face Space: https://huggingface.co/spaces/omnipart/OmniPart
    
    Usage:
      elixir omnipart_generation.exs <image_path> <mask_path> [options]
      elixir omnipart_generation.exs --image <path> --mask <path> [options]
    
    Arguments:
      <image_path>                 Input image file
      <mask_path>                  Segmentation mask file (.exr with 2D part IDs)
    
    Options:
      --image, -i <path>            Input image path (required if not positional)
      --mask, -m <path>             Segmentation mask path (.exr file) (required if not positional)
      --output-dir, -o <path>       Output directory (default: outputs)
      --output-format <format>      Output format: glb, obj, ply (default: glb)
      --gpu <int>                   GPU ID to use (default: 0)
      --seed <int>                  Random seed (default: 42)
      --help, -h                     Show this help message
    
    Note:
      The mask file should be a .exr file with shape [h, w, 3], where the last
      dimension contains the 2D part_id replicated across all three channels.
    
    Example:
      elixir omnipart_generation.exs image.jpg mask.exr
      elixir omnipart_generation.exs --image photo.png --mask segmentation.exr --output-format glb
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        image: :string,
        mask: :string,
        output_dir: :string,
        output_format: :string,
        gpu: :integer,
        seed: :integer,
        help: :boolean
      ],
      aliases: [
        i: :image,
        m: :mask,
        o: :output_dir,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    # Get image and mask paths (positional or from options)
    image_path = Keyword.get(opts, :image) || Enum.at(args, 0)
    mask_path = Keyword.get(opts, :mask) || Enum.at(args, 1)

    if !image_path do
      IO.puts("""
      Error: Image path is required.
      
      Usage:
        elixir omnipart_generation.exs <image_path> <mask_path> [options]
        elixir omnipart_generation.exs --image <path> --mask <path> [options]
      
      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    if !mask_path do
      IO.puts("""
      Error: Mask path is required.
      
      Usage:
        elixir omnipart_generation.exs <image_path> <mask_path> [options]
        elixir omnipart_generation.exs --image <path> --mask <path> [options]
      
      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    # Check if files exist
    if !File.exists?(image_path) do
      IO.puts("Error: Image file not found: #{image_path}")
      System.halt(1)
    end

    if !File.exists?(mask_path) do
      IO.puts("Error: Mask file not found: #{mask_path}")
      System.halt(1)
    end

    # Validate output format
    output_format = Keyword.get(opts, :output_format, "glb")
    valid_formats = ["glb", "obj", "ply"]
    if output_format not in valid_formats do
      IO.puts("Error: Invalid output format. Must be one of: #{Enum.join(valid_formats, ", ")}")
      System.halt(1)
    end

    %{
      image_path: image_path,
      mask_path: mask_path,
      output_dir: Keyword.get(opts, :output_dir, "outputs"),
      output_format: output_format,
      gpu: Keyword.get(opts, :gpu, 0),
      seed: Keyword.get(opts, :seed, 42)
    }
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== OmniPart Generation ===
Image: #{config.image_path}
Mask: #{config.mask_path}
Output Format: #{config.output_format}
Output Directory: #{config.output_dir}
GPU: #{config.gpu}
Seed: #{config.seed}
""")

# Add paths to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  omnipart_dir: Path.join([base_dir, "thirdparty", "OmniPart"]),
  checkpoint_dir: Path.join([base_dir, "pretrained_weights", "OmniPart"])
})

# Save config to JSON for Python to read
{config_file, config_file_normalized} = ConfigFile.create(config_with_paths, "omnipart_config")

# Download checkpoint if needed
SpanCollector.track_span("omnipart.download_weights", fn ->
  IO.puts("\n=== Step 1: Download Pretrained Weights ===")
  IO.puts("Checking for OmniPart checkpoint...")

  checkpoint_dir = config_with_paths.checkpoint_dir
  File.mkdir_p!(checkpoint_dir)

  # Check if checkpoint exists
  checkpoint_files = Path.wildcard(Path.join(checkpoint_dir, "*.pt")) ++
                      Path.wildcard(Path.join(checkpoint_dir, "*.pth")) ++
                      Path.wildcard(Path.join(checkpoint_dir, "*.ckpt"))

  if Enum.empty?(checkpoint_files) do
    IO.puts("Checkpoint not found. Please download from:")
    IO.puts("  https://github.com/HKU-MMLab/OmniPart")
    IO.puts("  Place checkpoint files in: #{checkpoint_dir}")
    IO.puts("[WARN] Continuing without checkpoint - model will download automatically if available")
  else
    IO.puts("[OK] Checkpoint found: #{List.first(checkpoint_files)}")
  end
end)

# Process using OmniPart
SpanCollector.track_span("omnipart.generation", fn ->
try do
  {_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
import subprocess
from pathlib import Path
import shutil

# Verify torch is available
try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    raise ImportError(
        f"torch is not available: {e}. "
        "This should have been installed by uv_init. "
        "Please check that uv_init completed successfully."
    )

# Get configuration from JSON file
""" <> ConfigFile.python_path_string(config_file_normalized) <> ~S"""

image_path = config.get('image_path')
mask_path = config.get('mask_path')
output_dir = config.get('output_dir', 'outputs')
output_format = config.get('output_format', 'glb')
gpu = config.get('gpu', 0)
seed = config.get('seed', 42)
omnipart_dir = config.get('omnipart_dir')
checkpoint_dir = config.get('checkpoint_dir')

# Resolve paths to absolute
image_path = str(Path(image_path).resolve())
mask_path = str(Path(mask_path).resolve())
output_dir = str(Path(output_dir).resolve())
omnipart_dir = str(Path(omnipart_dir).resolve())
checkpoint_dir = str(Path(checkpoint_dir).resolve())

# Verify input files exist
if not Path(image_path).exists():
    raise FileNotFoundError(f"Image file not found: {image_path}")

if not Path(mask_path).exists():
    raise FileNotFoundError(f"Mask file not found: {mask_path}")

# Verify mask is .exr file
if not mask_path.lower().endswith('.exr'):
    print(f"[WARN] Mask file is not .exr format: {mask_path}")
    print("OmniPart expects .exr files with shape [h, w, 3] containing 2D part IDs")

# Verify OmniPart directory exists (optional - can use from Hugging Face)
if not Path(omnipart_dir).exists():
    print(f"[WARN] OmniPart directory not found at {omnipart_dir}")
    print("Will attempt to use OmniPart from Python package or Hugging Face")

print("\n=== Step 2: Prepare Input Data ===")
print(f"Image: {image_path}")
print(f"Mask: {mask_path}")

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

print("\n=== Step 3: Run OmniPart Inference ===")
print(f"Generating 3D shape with part-aware control...")

# Set up environment
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu)
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Verify CUDA is available
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Using GPU {gpu}: {torch.cuda.get_device_name(0)}")
else:
    print("[WARN] CUDA is not available. Inference will be slower on CPU.")

# Build inference command
# OmniPart inference script: python -m scripts.inference_omnipart
inference_script = "scripts.inference_omnipart"
if Path(omnipart_dir).exists():
    # Use local OmniPart installation
    sys.path.insert(0, str(omnipart_dir))
    inference_cmd = [
        sys.executable,
        "-m", inference_script,
        "--image_input", image_path,
        "--mask_input", mask_path,
    ]
    cwd = omnipart_dir
else:
    # Try to use OmniPart as installed package
    try:
        import omnipart
        omnipart_path = Path(omnipart.__file__).parent.parent
        inference_cmd = [
            sys.executable,
            "-m", inference_script,
            "--image_input", image_path,
            "--mask_input", mask_path,
        ]
        cwd = str(omnipart_path)
    except ImportError:
        # Fallback: try to clone and use OmniPart
        print("[INFO] OmniPart not found locally. Attempting to use from GitHub...")
        temp_omnipart = Path(output_dir) / "omnipart_temp"
        if not temp_omnipart.exists():
            print("Cloning OmniPart repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/HKU-MMLab/OmniPart.git", str(temp_omnipart)],
                check=True
            )
        inference_cmd = [
            sys.executable,
            "-m", inference_script,
            "--image_input", image_path,
            "--mask_input", mask_path,
        ]
        cwd = str(temp_omnipart)
        sys.path.insert(0, str(temp_omnipart))

# Add output directory if supported
# Note: Check actual inference script for available options
inference_cmd.extend([
    "--output_dir", output_dir,
])

print(f"Command: {' '.join(inference_cmd)}")
print(f"Working directory: {cwd}")

# Run OmniPart inference
try:
    result = subprocess.run(
        inference_cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True
    )
    
    if result.stdout:
        print("\n=== Inference Output ===")
        print(result.stdout)
    
    print("[OK] OmniPart inference completed successfully")
    
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] OmniPart inference failed:")
    print(f"Return code: {e.returncode}")
    if e.stdout:
        print(f"STDOUT:\n{e.stdout}")
    if e.stderr:
        print(f"STDERR:\n{e.stderr}")
    raise

# Find output files
print("\n=== Step 4: Locate Output Files ===")
output_files = list(Path(output_dir).rglob(f"*.{output_format}"))
if not output_files:
    # Try other formats
    for fmt in ["glb", "obj", "ply", "usd", "usdc"]:
        output_files = list(Path(output_dir).rglob(f"*.{fmt}"))
        if output_files:
            break

if output_files:
    print(f"\n=== Complete ===")
    print(f"Generated 3D shape(s) saved to:")
    for output_file in output_files:
        print(f"  - {output_file}")
    
    # If multiple files, show the main one (usually the largest or most recent)
    if len(output_files) > 1:
        main_output = max(output_files, key=lambda p: p.stat().st_size)
        print(f"\nMain output file: {main_output}")
else:
    print(f"\n[WARN] No output files found in {output_dir}")
    print("Please check the inference output above for details.")

""", %{})
rescue
  e ->
    # Clean up temp file on error
    ConfigFile.cleanup(config_file)
    reraise e, __STACKTRACE__
after
  # Clean up temp file
  ConfigFile.cleanup(config_file)
end
end)

IO.puts("\n=== Complete ===")
IO.puts("3D shape generation completed successfully!")

# Display OpenTelemetry trace
SpanCollector.display_trace()

