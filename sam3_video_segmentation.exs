#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# SAM 3 Video Segmentation Script
# Segment objects in videos using SAM 3 (Segment Anything Model 3) from Meta Research
#
# Usage:
#   elixir sam3_video_segmentation.exs <video_path> [options]
#
# Options:
#   --prompt "person"              Text prompt for segmentation (default: "person")
#   --mask-color "green"          Mask color: green, red, blue, yellow, cyan, magenta, orange, purple, pink, lime, white, black (default: "green")
#   --mask-opacity 0.5             Mask opacity 0.0-1.0 (default: 0.5)
#   --mask-only                    Output original video on white background (masked areas only)
#   --mask-video                   Output a separate mask video (black and white masks)
#   --return-zip                   Create ZIP file with video and masks

# Configure OpenTelemetry for console-only logging
Application.put_env(:opentelemetry, :span_processor, :batch)
Application.put_env(:opentelemetry, :traces_exporter, :none)
Application.put_env(:opentelemetry, :metrics_exporter, :none)
Application.put_env(:opentelemetry, :logs_exporter, :none)

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
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
# Note: This installs PyTorch with CUDA 11.8 support by default
# To use CPU-only version, remove the [[tool.uv.index]] section and change torch/torchvision to regular dependencies
Pythonx.uv_init("""
[project]
name = "sam3-video-segmentation"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "torch",
  "torchvision",
  "transformers @ git+https://github.com/huggingface/transformers.git@ff13eb668aa03f151ded71636d723f2e490ad967",
  "numpy<2.0",
  "pillow",
  "imageio[ffmpeg]",
  "opencv-python",
  "decord",
  "scipy>=1.10.0",
  "scikit-image>=0.20.0",
  "requests",
  "kernels",
]

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
""")

# Define available mask colors
defmodule MaskColors do
  @colors %{
    "green" => [0, 255, 0],
    "red" => [255, 0, 0],
    "blue" => [0, 0, 255],
    "yellow" => [255, 255, 0],
    "cyan" => [0, 255, 255],
    "magenta" => [255, 0, 255],
    "orange" => [255, 165, 0],
    "purple" => [128, 0, 128],
    "pink" => [255, 192, 203],
    "lime" => [50, 205, 50],
    "white" => [255, 255, 255],
    "black" => [0, 0, 0]
  }

  def valid_colors, do: Map.keys(@colors)
  def get_rgb(color), do: Map.get(@colors, color)
  def valid?(color), do: Map.has_key?(@colors, color)
end

# Parse command-line arguments
defmodule ArgsParser do
  def show_help do
    valid_colors_str = Enum.join(MaskColors.valid_colors(), ", ")
    IO.puts("""
    SAM 3 Video Segmentation Script
    Segment objects in videos using SAM 3 (Segment Anything Model 3) from Meta Research

    Usage:
      elixir sam3_video_segmentation.exs <video_path> [options]

    Options:
      --prompt, -p "person"              Text prompt for segmentation (default: "person")
      --mask-color, -c "green"           Mask color (default: "green")
                                        Available colors: #{valid_colors_str}
      --mask-opacity <float>             Mask opacity 0.0-1.0 (default: 0.5)
      --mask-only                        Output original video on white background (masked areas only)
      --mask-video                       Output a separate mask video (black and white masks)
      --return-zip                       Create ZIP file with video and masks
      --help, -h                         Show this help message

    Example:
      elixir sam3_video_segmentation.exs video.mp4 --prompt "person" --mask-color green
      elixir sam3_video_segmentation.exs video.mp4 -p "car" -c red --mask-opacity 0.7
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        prompt: :string,
        mask_color: :string,
        mask_opacity: :float,
        mask_only: :boolean,
        mask_video: :boolean,
        return_zip: :boolean,
        help: :boolean
      ],
      aliases: [
        p: :prompt,
        c: :mask_color,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    video_filename = List.first(args)

    if !video_filename do
      IO.puts("""
      Error: Video file path is required.

      Usage:
        elixir sam3_video_segmentation.exs <video_path> [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    # Check if video file exists
    if !File.exists?(video_filename) do
      IO.puts("Error: Video file not found: #{video_filename}")
      System.halt(1)
    end

    mask_color_name = Keyword.get(opts, :mask_color, "green")

    # Validate mask_color using the enum
    if !MaskColors.valid?(mask_color_name) do
      valid_colors_str = Enum.join(MaskColors.valid_colors(), ", ")
      IO.puts("Error: Invalid mask color '#{mask_color_name}'")
      IO.puts("       Valid colors are: #{valid_colors_str}")
      System.halt(1)
    end

    # Get RGB values from the enum
    mask_color_rgb = MaskColors.get_rgb(mask_color_name)

    mask_opacity = Keyword.get(opts, :mask_opacity, 0.5)
    # Validate mask_opacity
    if mask_opacity < 0.0 or mask_opacity > 1.0 do
      IO.puts("Error: mask_opacity must be between 0.0 and 1.0")
      System.halt(1)
    end

    config = %{
      video_filename: video_filename,
      prompt: Keyword.get(opts, :prompt, "person"),
      mask_color: mask_color_name,
      mask_color_rgb: mask_color_rgb,
      mask_opacity: mask_opacity,
      mask_only: Keyword.get(opts, :mask_only, false),
      mask_video: Keyword.get(opts, :mask_video, false),
      return_zip: Keyword.get(opts, :return_zip, false),
      use_gpu: true
    }

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== SAM 3 Video Segmentation ===
Video: #{config.video_filename}
Prompt: #{config.prompt}
Mask Color: #{config.mask_color}
Mask Opacity: #{config.mask_opacity}
Mask Only: #{config.mask_only}
Mask Video: #{config.mask_video}
Return ZIP: #{config.return_zip}
""")

# Save config to JSON for Python to read (use temp file to avoid conflicts)
config_json = Jason.encode!(config)
# Use cross-platform temp directory
tmp_dir = System.tmp_dir!()
File.mkdir_p!(tmp_dir)
config_file = Path.join(tmp_dir, "sam3_config_#{System.system_time(:millisecond)}.json")
File.write!(config_file, config_json)
config_file_normalized = String.replace(config_file, "\\", "/")

# Import libraries and define constants using Pythonx
SpanCollector.track_span("sam3.segmentation", fn ->
try do
  {_, python_globals} = Pythonx.eval(~S"""
import os
import cv2
import time
import torch
import imageio
import subprocess
import numpy as np
import zipfile
import shutil
import tarfile
from pathlib import Path
from PIL import Image
from typing import Optional

# Try to import requests, fallback to urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False

from transformers import Sam3VideoModel, Sam3VideoProcessor
import warnings
import sys

MODEL_PATH = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/facebook/sam3/model.tar"

# Suppress general warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Pre-install cv_utils kernel to avoid warnings during inference
print("Pre-installing cv_utils kernel (if available)...")
try:
    from kernels import get_kernel
    import os

    # Set environment variable to allow kernel installation
    os.environ.setdefault("KERNELS_CACHE_DIR", os.path.expanduser("~/.cache/kernels"))

    try:
        # Try to get/install the kernel with explicit revision
        print("  Attempting to install kernels-community/cv_utils...")
        cv_utils_kernel = get_kernel("kernels-community/cv_utils", revision="main")
        print("✓ cv_utils kernel installed successfully")
    except Exception as e:
        error_msg = str(e)
        if "Cannot install kernel" in error_msg or "not supported" in error_msg.lower():
            print(f"⚠ cv_utils kernel installation failed: {error_msg}")
            print("  This may be due to CUDA/PyTorch compatibility or missing build tools")
            print("  Post-processing will use CPU fallback methods (this is normal and works fine)")
        else:
            print(f"⚠ cv_utils kernel installation error: {error_msg}")
            print("  Post-processing will use fallback methods")
except ImportError:
    print("⚠ kernels library not available - installing...")
    print("  Run: pip install kernels")
except Exception as e:
    print(f"⚠ Unexpected error during kernel setup: {e}")

def download_weights(url, dest):
    # Download model weights
    start = time.time()
    print(f"Downloading model weights from: {url}")
    print(f"Destination: {dest}")

    os.makedirs(dest, exist_ok=True)
    temp_file = os.path.join(dest, "model.tar")

    # Download using requests or urllib
    if HAS_REQUESTS:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        urllib.request.urlretrieve(url, temp_file)

    # Extract tar file
    with tarfile.open(temp_file, 'r:*') as tar:
        tar.extractall(dest)

    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)

    print(f"✓ Download and extraction completed in {time.time() - start:.2f} seconds")
""", %{})

# Store globals for use in subsequent steps
Process.put(:python_globals, python_globals)

IO.puts("\n=== Step 1: Download and Load SAM 3 Model ===")

# Setup device, download weights, and load model
python_globals = Process.get(:python_globals) || %{}

{_, python_globals} = Pythonx.eval(~S"""
# Check if PyTorch is installed with CUDA support
# First check if torch is available
try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed")
    raise

# Check CUDA availability
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
else:
    print("CUDA not available. Install PyTorch with CUDA support for GPU acceleration.")
    print("Visit: https://pytorch.org/get-started/locally/")

# Setup device and dtype
# Always prefer GPU if available (default behavior)
if cuda_available:
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using GPU: {device}, dtype: {dtype}")
else:
    device = "cpu"
    dtype = torch.float16
    print(f"Using CPU: {device}, dtype: {dtype}")
    print("\n" + "="*60)
    print("WARNING: CUDA is not available! Using CPU instead.")
    print("="*60)
    print("\nPyTorch was installed without CUDA support.")
    print("To enable GPU acceleration, you need to reinstall PyTorch with CUDA:")
    print("\n1. First, uninstall the current CPU-only version:")
    print("   pip uninstall torch torchvision -y")
    print("\n2. Then install PyTorch with CUDA 11.8 support:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\n   Or for CUDA 12.1:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("\n3. Verify CUDA is available:")
    print("   python -c 'import torch; print(torch.cuda.is_available())'")
    print("\nNote: Make sure you have NVIDIA GPU drivers and CUDA toolkit installed.")
    print("="*60 + "\n")

# Download weights if they don't exist
if not os.path.exists(MODEL_PATH):
    download_weights(MODEL_URL, MODEL_PATH)

# Load model
print(f"\nLoading SAM 3 model from {MODEL_PATH}...")

model = Sam3VideoModel.from_pretrained(MODEL_PATH).to(device, dtype=dtype).eval()
processor = Sam3VideoProcessor.from_pretrained(MODEL_PATH)


print("✓ Model loaded successfully!")
""", python_globals)

Process.put(:python_globals, python_globals)

IO.puts("\n=== Step 2: Process Video ===")

# Get configuration from JSON file and process video
{_, python_globals} = Pythonx.eval("""
# Get configuration from JSON file (created by Elixir)
import json
import os

# Read from JSON file created by Elixir
""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
""" <> ~S"""
if not os.path.exists(config_file_path):
    raise ValueError(f"Config file not found: {config_file_path}")

with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

prompt = config.get('prompt', 'person')
mask_color = config.get('mask_color', 'green')
mask_opacity = config.get('mask_opacity', 0.5)
mask_only = config.get('mask_only', False)
return_zip = config.get('return_zip', False)
video_filename = config.get('video_filename', None)

# Check if video_filename is set
if not video_filename or not os.path.exists(video_filename):
    raise ValueError(f"Video file not found: {video_filename}")

# Load video frames
print(f"Processing video: {video_filename}")
cap = cv2.VideoCapture(video_filename)
frames = []
original_fps = cap.get(cv2.CAP_PROP_FPS)
if original_fps <= 0:
    original_fps = 30.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(Image.fromarray(frame))
cap.release()

if not frames:
    raise ValueError("Could not load frames from video")

print(f"✓ Loaded {len(frames)} frames. FPS: {original_fps}")

# Initialize inference session
print("Initializing inference session...")
inference_session = processor.init_video_session(
    video=frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=dtype
)

# Add text prompt
if prompt and prompt != "":
    print(f"Adding text prompt: '{prompt}'")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt
    )

# Run inference
print("Running segmentation inference...")
output_frames_data = {}

for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session,
    max_frame_num_to_track=len(frames)
):
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    output_frames_data[model_outputs.frame_idx] = processed_outputs

print(f"✓ Processed {len(output_frames_data)} frames")
""", Process.get(:python_globals) || %{"config_file_normalized" => config_file_normalized})

Process.put(:python_globals, python_globals)

IO.puts("\n=== Step 3: Generate Output Video ===")

# Generate output video
{_, python_globals} = Pythonx.eval(~S"""
# Get configuration from JSON file
import json

""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
""" <> ~S"""
    config = json.load(f)

mask_color = config.get('mask_color', 'green')
mask_color_rgb = config.get('mask_color_rgb', [0, 255, 0])  # RGB values from Elixir enum (green default)
mask_opacity = config.get('mask_opacity', 0.5)
mask_only = config.get('mask_only', False)
mask_video = config.get('mask_video', False)
return_zip = config.get('return_zip', False)

# Use RGB values directly from config (set by Elixir MaskColors enum)
color_rgb = np.array(mask_color_rgb, dtype=np.uint8)

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Save video
output_video_path = output_dir / "output.mp4"
print(f"Saving output video to {output_video_path}...")

height, width = np.array(frames[0]).shape[:2]
writer = imageio.get_writer(str(output_video_path), fps=original_fps, codec='libx264', quality=None, pixelformat='yuv420p')

for idx, frame_pil in enumerate(frames):
    frame_np = np.array(frame_pil)

    # Start with white background for clean mask visualization
    output_frame = np.ones_like(frame_np) * 255

    if idx in output_frames_data:
        results = output_frames_data[idx]
        masks = results.get('masks', None)

        if masks is not None:
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()

            if len(masks) > 0:
                combined_mask = np.zeros((height, width), dtype=bool)
                for mask in masks:
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)
                    elif mask.ndim == 2:
                        pass
                    else:
                        mask = mask.squeeze()

                    if mask.shape != (height, width):
                        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

                    combined_mask = np.logical_or(combined_mask, mask > 0.0)

                mask_indices = combined_mask

                if mask_only:
                    # Show original video only in masked regions, rest is white
                    output_frame[mask_indices] = frame_np[mask_indices]
                else:
                    # Show original video in masked regions with optional color overlay
                    masked_original = frame_np[mask_indices].copy()
                    if mask_opacity > 0.0:
                        # Apply color overlay with opacity
                        overlay_color = np.tile(color_rgb, (masked_original.shape[0], 1))
                        masked_colored = (masked_original * (1 - mask_opacity) + overlay_color * mask_opacity).astype(np.uint8)
                        output_frame[mask_indices] = masked_colored
                    else:
                        # No color overlay, just show original in masked regions
                        output_frame[mask_indices] = masked_original
    else:
        # If no mask data for this frame, keep it white
        pass

    writer.append_data(output_frame)

writer.close()
print(f"✓ Video saved: {output_video_path}")

# Generate mask video if requested
if mask_video:
    mask_video_path = output_dir / "output_mask.mp4"
    print(f"\nGenerating mask video: {mask_video_path}...")

    mask_writer = imageio.get_writer(str(mask_video_path), fps=original_fps, codec='libx264', quality=None, pixelformat='yuv420p')

    for idx, frame_pil in enumerate(frames):
        # Create black background
        mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

        if idx in output_frames_data:
            results = output_frames_data[idx]
            masks = results.get('masks', None)

            if masks is not None:
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()

                if len(masks) > 0:
                    combined_mask = np.zeros((height, width), dtype=bool)
                    for mask in masks:
                        if mask.ndim == 3 and mask.shape[0] == 1:
                            mask = mask.squeeze(0)
                        elif mask.ndim == 2:
                            pass
                        else:
                            mask = mask.squeeze()

                        if mask.shape != (height, width):
                            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

                        combined_mask = np.logical_or(combined_mask, mask > 0.0)

                    # Set mask areas to white (255)
                    mask_frame[combined_mask] = [255, 255, 255]

        mask_writer.append_data(mask_frame)

    mask_writer.close()
    print(f"✓ Mask video saved: {mask_video_path}")
""", python_globals)

Process.put(:python_globals, python_globals)

# Step 4: Save individual frame masks if return_zip is True
if config.return_zip do
  IO.puts("\n=== Step 4: Save Individual Frame Masks ===")

  {_, python_globals} = Pythonx.eval(~S"""
  # Get configuration from JSON file
  import json

  with open("config.json", 'r') as f:
      config = json.load(f)

  return_zip = config.get('return_zip', False)

  if return_zip:
      masks_dir = output_dir / "masks"
      masks_dir.mkdir(exist_ok=True)

      print("Saving individual frame masks...")
      for frame_idx, results in output_frames_data.items():
          masks = results.get('masks', None)
          if masks is not None:
              if isinstance(masks, torch.Tensor):
                  masks = masks.cpu().numpy()

              if len(masks) > 0:
                  height, width = np.array(frames[0]).shape[:2]
                  combined_mask = np.zeros((height, width), dtype=np.uint8)

                  for mask in masks:
                      if mask.ndim == 3 and mask.shape[0] == 1:
                          mask = mask.squeeze(0)
                      elif mask.ndim > 2:
                          mask = mask.squeeze()

                      if mask.shape != (height, width):
                          mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

                      mask_bool = mask > 0.0
                      combined_mask = np.logical_or(combined_mask, mask_bool)

                  mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
                  mask_img.save(masks_dir / f"mask_{frame_idx:05d}.png")

      print(f"✓ Saved {len(list(masks_dir.glob('*.png')))} mask images")
  """, python_globals)

  Process.put(:python_globals, python_globals)
end

IO.puts("\n=== Step 5: Create Output Files ===")

# Create ZIP if return_zip is True
{_, _python_globals} = Pythonx.eval(~S"""
# Create ZIP if return_zip is True, otherwise just provide video download
import json

""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
""" <> ~S"""
    config = json.load(f)

return_zip = config.get('return_zip', False)

# Create ZIP if return_zip is True, otherwise just provide video download
if return_zip:
    zip_path = output_dir / "output.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add video
        zipf.write(output_video_path, "output.mp4")

        # Add mask video if it exists
        mask_video_path = output_dir / "output_mask.mp4"
        if mask_video_path.exists():
            zipf.write(mask_video_path, "output_mask.mp4")

        # Add masks
        masks_dir = output_dir / "masks"
        if masks_dir.exists():
            for mask_file in masks_dir.glob("*.png"):
                zipf.write(mask_file, f"masks/{mask_file.name}")

    print(f"✓ Created ZIP: {zip_path}")
    print(f"📥 File location: {zip_path.absolute()}")
else:
    print(f"✓ Video saved: {output_video_path}")
    print(f"📥 File location: {output_video_path.absolute()}")

    # Show mask video location if it was created
    mask_video_path = output_dir / "output_mask.mp4"
    if mask_video_path.exists():
        print(f"✓ Mask video saved: {mask_video_path}")
        print(f"📥 Mask video location: {mask_video_path.absolute()}")

print("\n✓ All done!")
""", Process.get(:python_globals) || %{"config_file_normalized" => config_file_normalized})
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
end)

IO.puts("\n=== Complete ===")

# Display OpenTelemetry trace - save to output directory
SpanCollector.display_trace("output")
