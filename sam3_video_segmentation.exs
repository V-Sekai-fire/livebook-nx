#!/usr/bin/env elixir

# SAM 3 Video Segmentation Script
# Segment objects in videos using SAM 3 (Segment Anything Model 3) from Meta Research
#
# Usage:
#   elixir sam3_video_segmentation.exs <video_path> [options]
#
# Options:
#   --prompt "person"              Text prompt for segmentation (default: "person")
#   --mask-color "green"           Mask color: green, red, blue, yellow, cyan, magenta (default: "green")
#   --mask-opacity 0.5             Mask opacity 0.0-1.0 (default: 0.5)
#   --mask-only                    Output black and white mask only
#   --return-zip                   Create ZIP file with video and masks
#   --generate-3d                  Generate 3D models (placeholder)
#   --gpu                          Force GPU usage (will use CPU if GPU unavailable)

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"}
])

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
  "transformers @ git+https://github.com/huggingface/transformers.git@refs/pull/42407/head",
  "numpy<2.0",
  "pillow",
  "imageio[ffmpeg]",
  "opencv-python",
  "decord",
  "open3d>=0.18.0",
  "pygltflib>=1.15.0",
  "scipy>=1.10.0",
  "gsply>=0.1.0",
  "requests"
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
        prompt: :string,
        mask_color: :string,
        mask_opacity: :float,
        mask_only: :boolean,
        return_zip: :boolean,
        generate_3d: :boolean,
        gpu: :boolean
      ],
      aliases: [
        p: :prompt,
        c: :mask_color,
        o: :mask_opacity
      ]
    )

    video_filename = List.first(args)

    if !video_filename do
      IO.puts("""
      Error: Video file path is required.

      Usage:
        elixir sam3_video_segmentation.exs <video_path> [options]

      Options:
        --prompt, -p "person"              Text prompt for segmentation (default: "person")
        --mask-color, -c "green"           Mask color: green, red, blue, yellow, cyan, magenta (default: "green")
        --mask-opacity, -o 0.5             Mask opacity 0.0-1.0 (default: 0.5)
        --mask-only                        Output black and white mask only
        --return-zip                       Create ZIP file with video and masks
        --generate-3d                      Generate 3D models (placeholder)
        --gpu                              Force GPU usage (will use CPU if GPU unavailable)
      """)
      System.halt(1)
    end

    config = %{
      video_filename: video_filename,
      prompt: Keyword.get(opts, :prompt, "person"),
      mask_color: Keyword.get(opts, :mask_color, "green"),
      mask_opacity: Keyword.get(opts, :mask_opacity, 0.5),
      mask_only: Keyword.get(opts, :mask_only, false),
      return_zip: Keyword.get(opts, :return_zip, false),
      generate_3d: Keyword.get(opts, :generate_3d, false),
      use_gpu: Keyword.get(opts, :gpu, false)
    }

    # Validate mask_color
    valid_colors = ["green", "red", "blue", "yellow", "cyan", "magenta"]
    if config.mask_color not in valid_colors do
      IO.puts("Error: Invalid mask color. Must be one of: #{Enum.join(valid_colors, ", ")}")
      System.halt(1)
    end

    # Validate mask_opacity
    if config.mask_opacity < 0.0 or config.mask_opacity > 1.0 do
      IO.puts("Error: mask_opacity must be between 0.0 and 1.0")
      System.halt(1)
    end

    # Check if video file exists
    if !File.exists?(config.video_filename) do
      IO.puts("Error: Video file not found: #{config.video_filename}")
      System.halt(1)
    end

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
Return ZIP: #{config.return_zip}
Generate 3D: #{config.generate_3d}
Use GPU: #{config.use_gpu}
""")

# Save config to JSON for Python to read
config_json = Jason.encode!(config)
File.write!("config.json", config_json)

# Import libraries and define constants using Pythonx
{_, python_globals} = Pythonx.eval("""
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

MODEL_PATH = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/facebook/sam3/model.tar"

def download_weights(url, dest):
    \"\"\"Download model weights\"\"\"
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
use_gpu_flag = if config.use_gpu, do: "True", else: "False"

{_, python_globals} = Pythonx.eval("""
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

# Get use_gpu flag from Elixir
use_gpu_requested = #{use_gpu_flag}

# Setup device and dtype
# Always prefer GPU if available (default behavior)
if cuda_available:
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using GPU: {device}, dtype: {dtype}")
    if use_gpu_requested:
        print("GPU usage explicitly requested via --gpu flag")
else:
    device = "cpu"
    dtype = torch.float16
    print(f"Using CPU: {device}, dtype: {dtype}")
    if use_gpu_requested:
        print("\\n" + "="*60)
        print("WARNING: --gpu flag was set but CUDA is not available!")
        print("="*60)
        print("\\nPyTorch was installed without CUDA support.")
        print("To enable GPU acceleration, you need to reinstall PyTorch with CUDA:")
        print("\\n1. First, uninstall the current CPU-only version:")
        print("   pip uninstall torch torchvision -y")
        print("\\n2. Then install PyTorch with CUDA 11.8 support:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("\\n   Or for CUDA 12.1:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("\\n3. Verify CUDA is available:")
        print("   python -c 'import torch; print(torch.cuda.is_available())'")
        print("\\nNote: Make sure you have NVIDIA GPU drivers and CUDA toolkit installed.")
        print("="*60 + "\\n")

# Download weights if they don't exist
if not os.path.exists(MODEL_PATH):
    download_weights(MODEL_URL, MODEL_PATH)

# Load model
print(f"\\nLoading SAM 3 model from {MODEL_PATH}...")
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
config_file = "config.json"
if not os.path.exists(config_file):
    raise ValueError("Config file not found")

with open(config_file, 'r') as f:
    config = json.load(f)

prompt = config.get('prompt', 'person')
mask_color = config.get('mask_color', 'green')
mask_opacity = config.get('mask_opacity', 0.5)
mask_only = config.get('mask_only', False)
return_zip = config.get('return_zip', False)
generate_3d = config.get('generate_3d', False)
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
""", python_globals)

Process.put(:python_globals, python_globals)

IO.puts("\n=== Step 3: Generate Output Video ===")

# Generate output video
{_, python_globals} = Pythonx.eval("""
# Get configuration from JSON file
import json

with open("config.json", 'r') as f:
    config = json.load(f)

mask_color = config.get('mask_color', 'green')
mask_opacity = config.get('mask_opacity', 0.5)
mask_only = config.get('mask_only', False)
return_zip = config.get('return_zip', False)

# Define colors
colors = {
    "green": [0, 255, 0],
    "red": [255, 0, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255]
}
color_rgb = np.array(colors.get(mask_color.lower(), [0, 255, 0]), dtype=np.uint8)

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
    
    if mask_only:
        output_frame = np.zeros_like(frame_np)
    else:
        output_frame = frame_np.copy()

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
                
                overlay_indices = combined_mask
                
                if mask_only:
                    output_frame[overlay_indices] = [255, 255, 255]
                else:
                    output_frame[overlay_indices] = (output_frame[overlay_indices] * (1 - mask_opacity) + color_rgb * mask_opacity).astype(np.uint8)
    
    writer.append_data(output_frame)

writer.close()
print(f"✓ Video saved: {output_video_path}")
""", python_globals)

Process.put(:python_globals, python_globals)

# Step 4: Save individual frame masks if return_zip is True
if config.return_zip do
  IO.puts("\n=== Step 4: Save Individual Frame Masks ===")

  {_, python_globals} = Pythonx.eval("""
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

# Step 5: Generate 3D models (placeholder)
if config.generate_3d do
  IO.puts("\n=== Step 5: Generate 3D Models ===")

  {_, python_globals} = Pythonx.eval("""
  # Get configuration from JSON file
  import json

  with open("config.json", 'r') as f:
      config = json.load(f)

  generate_3d = config.get('generate_3d', False)

  # Note: Full 3D generation with SAM 3D Objects requires complex setup
  # This is a placeholder - you would need to integrate SAM 3D Objects separately
  if generate_3d:
      print("⚠ Full 3D generation requires SAM 3D Objects setup")
      print("For now, you can use the PLY to GLB converter notebook separately")
      print("This feature will be available in a future update")
  """, python_globals)

  Process.put(:python_globals, python_globals)
end

IO.puts("\n=== Step 6: Create Output Files ===")

# Create ZIP if return_zip is True
{_, _python_globals} = Pythonx.eval("""
# Create ZIP if return_zip is True, otherwise just provide video download
import json

with open("config.json", 'r') as f:
    config = json.load(f)

return_zip = config.get('return_zip', False)

# Create ZIP if return_zip is True, otherwise just provide video download
if return_zip:
    zip_path = output_dir / "output.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add video
        zipf.write(output_video_path, "output.mp4")
        
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

print("\\n✓ All done!")
""", Process.get(:python_globals) || %{})

IO.puts("\n=== Complete ===")

