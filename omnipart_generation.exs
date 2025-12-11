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
#   elixir omnipart_generation.exs <image_path> [image_path2 ...] [mask_path ...] [options]
#   elixir omnipart_generation.exs --image <path1>,<path2>,<path3> [--mask <path1>,<path2>] [options]
#
# Multiview Support:
#   Multiple images can be provided for multiview 3D generation using comma-separated values.
#   Each image will be processed separately for segmentation, with independent mask indices 
#   (starting from 0 for each image). Each image can have its own merge groups specified 
#   using || separator.
#
#   Example (multiview with merge groups):
#     elixir omnipart_generation.exs \
#       --image img1.png,img2.png,img3.png \
#       --merge-groups "0,1;3,4||2,3||1,2" \
#       --segment-only
#
# Options:
#   --image, -i <paths>           Input image path(s) - comma-separated for multiple images (required)
#   --mask, -m <paths>            Segmentation mask path(s) - comma-separated for multiple masks (.exr file with 2D part IDs) (optional)
#   --output-dir, -o <path>       Output directory (default: output)
#   --segment-only                 Only generate segmentation, don't generate 3D model
#   --apply-merge                  Apply merge groups to existing segmentation state
#   --size-threshold <int>        Minimum segment size in pixels for mask generation (default: 2000)
#   --merge-groups <string>       Merge groups for mask(s). For multiple images, separate with || 
#                                  (e.g., "0,1;3,4||2,3" merges 0&1 and 3&4 for first image, 2&3 for second)
#                                  Example for 4 images: "5,6;0,3||5,6;2,4||5,6;0,2,3,4||5,6;1,3,4"
#   --num-inference-steps <int>   Number of inference steps (default: 25)
#   --guidance-scale <float>      Guidance scale for SLat sampler (default: 7.5)
#   --simplify-ratio <float>      Mesh simplification ratio (default: 0.15)
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
# Based on official requirements: https://github.com/HKU-MMLab/OmniPart
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
      elixir omnipart_generation.exs <image_path> [mask_path] [options]
      elixir omnipart_generation.exs --image <path1>,<path2> [--mask <path1>,<path2>] [options]
    
    Arguments:
      <image_path>                 Input image file (required)
      <mask_path>                  Segmentation mask file (.exr with 2D part IDs) (optional)
    
    Options:
      --image, -i <path>            Input image path(s) (required, can specify multiple times or use positional args)
      --mask, -m <path>             Segmentation mask path(s) (.exr file, one per image, optional)
      --output-dir, -o <path>       Output directory (default: output)
      --segment-only                 Only generate segmentation, don't generate 3D model
      --apply-merge                  Apply merge groups to existing segmentation state
      --size-threshold <int>        Minimum segment size in pixels for mask generation (default: 2000)
      --merge-groups <string>       Merge groups for mask(s) (use || to separate per image, e.g., "0,1;3,4||2,3||1,2")
      --num-inference-steps <int>   Number of inference steps (default: 25)
      --guidance-scale <float>      Guidance scale for SLat sampler (default: 7.5)
      --simplify-ratio <float>      Mesh simplification ratio (default: 0.15)
      --gpu <int>                   GPU ID to use (default: 0)
      --seed <int>                  Random seed (default: 42)
      --help, -h                     Show this help message
    
    Workflow (matching Hugging Face Space):
      1. Segment: Generate initial segmentation from image
      2. Edit: Apply merge groups iteratively to refine segmentation
      3. Generate: Create 3D model from final segmentation
    
    Note:
      - The image should ideally be RGBA PNG format (alpha channel used for bounding box)
      - If mask is not provided, it will be automatically generated using SAM
      - The mask file should be a .exr file with shape [h, w, 3], where the last
        dimension contains the 2D part_id replicated across all three channels.
      - Model weights will be automatically downloaded from Hugging Face on first run
    
    Examples:
      # Step 1: Generate initial segmentation (shows segment IDs)
      elixir omnipart_generation.exs image.png --segment-only
      
      # Step 2: Apply merge groups (can be run multiple times with different groups)
      elixir omnipart_generation.exs image.png --apply-merge --merge-groups "0,1;3,4"
      elixir omnipart_generation.exs image.png --apply-merge --merge-groups "2,5;6,7"
      
      # Step 3: Generate 3D model (uses final mask from previous step)
      elixir omnipart_generation.exs image.png
      
      # Or do it all in one command
      elixir omnipart_generation.exs image.png --merge-groups "0,1;3,4"
      
      # Use existing mask file
      elixir omnipart_generation.exs image.png --mask segmentation.exr
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        image: :string,
        mask: :string,
        output_dir: :string,
        segment_only: :boolean,
        apply_merge: :boolean,
        size_threshold: :integer,
        merge_groups: :string,
        num_inference_steps: :integer,
        guidance_scale: :float,
        simplify_ratio: :float,
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

    # Get image paths - support comma-separated values
    image_paths = case Keyword.get(opts, :image) do
      nil ->
        # No --image flag, use positional args
        args
      image_val when is_binary(image_val) ->
        # Split comma-separated values
        image_val
        |> String.split(",")
        |> Enum.map(&String.trim/1)
        |> Enum.filter(&(&1 != ""))
      image_val ->
        # Convert to string and split
        to_string(image_val)
        |> String.split(",")
        |> Enum.map(&String.trim/1)
        |> Enum.filter(&(&1 != ""))
    end
    
    # Debug: Verify all images are parsed
    if length(image_paths) > 1 do
      IO.puts("[DEBUG] Parsed #{length(image_paths)} images: #{inspect(Enum.map(image_paths, &Path.basename/1))}")
      OtelLogger.info("Parsed multiple images", [{"image.count", length(image_paths)}])
    end

    # Get mask paths - support comma-separated values
    mask_paths = case Keyword.get(opts, :mask) do
      nil ->
        # No --mask flag, try positional args after images
        if length(image_paths) > 0 do
          Enum.drop(args, length(image_paths))
        else
          []
        end
      mask_val when is_binary(mask_val) ->
        # Split comma-separated values
        mask_val
        |> String.split(",")
        |> Enum.map(&String.trim/1)
        |> Enum.filter(&(&1 != ""))
      mask_val ->
        # Convert to string and split
        to_string(mask_val)
        |> String.split(",")
        |> Enum.map(&String.trim/1)
        |> Enum.filter(&(&1 != ""))
    end

    if length(image_paths) == 0 do
      IO.puts("""
      Error: At least one image path is required.
      
      Usage:
        elixir omnipart_generation.exs <image_path> [image_path2 ...] [mask_path ...] [options]
        elixir omnipart_generation.exs --image <path1>,<path2>,<path3> [--mask <path1>,<path2>] [options]
      
      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    # Check if all image files exist
    Enum.each(image_paths, fn image_path ->
      if !File.exists?(image_path) do
        OtelLogger.error("Image file not found", [{"file.path", image_path}])
        System.halt(1)
      end
    end)

    # Validate mask paths (must match number of images or be empty)
    if length(mask_paths) > 0 && length(mask_paths) != length(image_paths) do
      IO.puts("""
      Error: Number of mask paths (#{length(mask_paths)}) must match number of image paths (#{length(image_paths)})
      """)
      System.halt(1)
    end

    # Check mask files exist, set auto_generate flags
    {mask_paths_valid, auto_generate_flags} = Enum.with_index(mask_paths)
      |> Enum.map_reduce([], fn {mask_path, idx}, acc ->
        if mask_path && File.exists?(mask_path) do
          {mask_path, [false | acc]}
        else
          if mask_path do
            SpanCollector.add_span_attribute("mask.auto_generate.#{idx}", true)
            SpanCollector.add_span_attribute("mask.path.#{idx}", mask_path)
          end
          {nil, [true | acc]}
        end
      end)
    
    auto_generate_flags = Enum.reverse(auto_generate_flags)
    # If no masks provided, auto-generate all
    {mask_paths_valid, auto_generate_flags} = if length(mask_paths) == 0 do
      {List.duplicate(nil, length(image_paths)), List.duplicate(true, length(image_paths))}
    else
      {mask_paths_valid, auto_generate_flags}
    end

    segment_only = Keyword.get(opts, :segment_only, false)
    apply_merge = Keyword.get(opts, :apply_merge, false)
    
    # Validate mode
    if segment_only && apply_merge do
      # Error logs are kept for critical failures
      OtelLogger.error("Cannot use --segment-only and --apply-merge together")
      System.halt(1)
    end
    
    # Parse merge groups - support multiple (separated by ||)
    merge_groups_str = Keyword.get(opts, :merge_groups)
    merge_groups_list = if merge_groups_str do
      merge_groups_str
      |> String.split("||")
      |> Enum.map(&String.trim/1)
      |> Enum.filter(&(&1 != ""))
    else
      []
    end
    
    # Validate merge groups count matches images (or is empty)
    if length(merge_groups_list) > 0 && length(merge_groups_list) != length(image_paths) do
      IO.puts("""
      Error: Number of merge groups (#{length(merge_groups_list)}) must match number of image paths (#{length(image_paths)})
      Use || to separate merge groups for each image, e.g., "0,1;3,4||2,3||1,2"
      """)
      System.halt(1)
    end
    
    %{
      image_paths: image_paths,
      mask_paths: mask_paths_valid,
      auto_generate_masks: auto_generate_flags,
      segment_only: segment_only,
      apply_merge: apply_merge,
      output_dir: Keyword.get(opts, :output_dir, "output"),
      size_threshold: Keyword.get(opts, :size_threshold, 2000),
      merge_groups: merge_groups_list,
      num_inference_steps: Keyword.get(opts, :num_inference_steps, 25),
      guidance_scale: Keyword.get(opts, :guidance_scale, 7.5),
      simplify_ratio: Keyword.get(opts, :simplify_ratio, 0.15),
      gpu: Keyword.get(opts, :gpu, 0),
      seed: Keyword.get(opts, :seed, 42)
    }
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

mode = cond do
  config.segment_only -> "Segment Only"
  config.apply_merge -> "Apply Merge"
  true -> "Full Generation"
end

images_str = if is_list(config.image_paths) and length(config.image_paths) > 0 do
  if length(config.image_paths) == 1 do
    Path.basename(List.first(config.image_paths))
  else
    "#{length(config.image_paths)} image(s): #{Enum.join(Enum.map(config.image_paths, &Path.basename/1), ", ")}"
  end
else
  if config.image_paths do
    # Handle case where image_paths might be a single string (shouldn't happen, but be safe)
    if is_binary(config.image_paths) do
      Path.basename(config.image_paths)
    else
      "No images specified"
    end
  else
    "No images specified"
  end
end

masks_str = if is_list(config.mask_paths) and length(config.mask_paths) > 0 do
  "#{length(config.mask_paths)} mask(s)"
else
  if config.mask_paths && length(config.mask_paths) > 0 && hd(config.mask_paths) do
    Path.basename(hd(config.mask_paths))
  else
    "Auto-generate using SAM"
  end
end

merge_groups_str = if is_list(config.merge_groups) and length(config.merge_groups) > 0 do
  "#{length(config.merge_groups)} merge group set(s)"
else
  if config.merge_groups && length(config.merge_groups) > 0 do
    inspect(config.merge_groups)
  else
    nil
  end
end

IO.puts("""
=== OmniPart Generation ===
Mode: #{mode}
Images: #{images_str}
Masks: #{masks_str}
Output Directory: #{config.output_dir}
#{if config.auto_generate_masks && length(config.auto_generate_masks) > 0 && Enum.any?(config.auto_generate_masks) || config.segment_only, do: "Size Threshold: #{config.size_threshold}\n", else: ""}#{if merge_groups_str, do: "Merge Groups: #{merge_groups_str}\n", else: ""}#{if not config.segment_only, do: "Inference Steps: #{config.num_inference_steps}\nGuidance Scale: #{config.guidance_scale}\nSimplify Ratio: #{config.simplify_ratio}\n", else: ""}GPU: #{config.gpu}
Seed: #{config.seed}
""")

# Add paths to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  omnipart_dir: Path.join([base_dir, "thirdparty", "OmniPart"]),
  checkpoint_dir: Path.join([base_dir, "pretrained_weights", "OmniPart", "ckpt"]),
  segment_only: config.segment_only,
  apply_merge: config.apply_merge
})

# Validate that image_paths is a list with all images
if not is_list(config_with_paths.image_paths) or length(config_with_paths.image_paths) == 0 do
  IO.puts("Error: image_paths must be a non-empty list")
  IO.puts("  Got: #{inspect(config_with_paths.image_paths)}")
  System.halt(1)
end

# Debug: Log how many images are in config before serialization
IO.puts("[DEBUG] Config contains #{length(config_with_paths.image_paths)} image(s) before serialization")
if length(config_with_paths.image_paths) > 1 do
  IO.puts("[DEBUG] Image paths in config: #{inspect(Enum.map(config_with_paths.image_paths, &Path.basename/1))}")
  OtelLogger.info("Config contains multiple images", [{"image.count", length(config_with_paths.image_paths)}])
end

# Save config to JSON for Python to read
{config_file, config_file_normalized} = ConfigFile.create(config_with_paths, "omnipart_config")

# Debug: Verify config was written correctly
case File.read(config_file) do
  {:ok, content} ->
    case Jason.decode(content) do
      {:ok, decoded} ->
        image_count = if is_list(decoded["image_paths"]), do: length(decoded["image_paths"]), else: 0
        if image_count != length(config_with_paths.image_paths) do
          IO.puts("[WARN] Config file image count mismatch: expected #{length(config_with_paths.image_paths)}, got #{image_count}")
        end
      {:error, _} -> :ok
    end
  {:error, _} -> :ok
end

# Download checkpoints using Elixir downloader
SpanCollector.track_span("omnipart.download_weights", fn ->
  checkpoint_dir = config_with_paths.checkpoint_dir
  File.mkdir_p!(checkpoint_dir)
  
  # Download OmniPart modules (includes SAM checkpoint)
  omnipart_modules_dir = Path.join([checkpoint_dir, "..", "OmniPart_modules"])
  case HuggingFaceDownloader.download_repo("omnipart/OmniPart_modules", omnipart_modules_dir, "OmniPart Modules", true) do
    {:ok, _} ->
      SpanCollector.add_span_attribute("download.status", "success")
      # Move SAM checkpoint to checkpoint_dir if needed
      sam_ckpt_src = Path.join([omnipart_modules_dir, "sam_vit_h_4b8939.pth"])
      sam_ckpt_dst = Path.join([checkpoint_dir, "sam_vit_h_4b8939.pth"])
      if File.exists?(sam_ckpt_src) and not File.exists?(sam_ckpt_dst) do
        File.cp!(sam_ckpt_src, sam_ckpt_dst)
        SpanCollector.add_span_attribute("sam_checkpoint.copied", true)
      end
    {:error, reason} ->
      SpanCollector.add_span_attribute("download.status", "error")
      SpanCollector.add_span_attribute("download.error", inspect(reason))
      # Only log errors - metrics will show the status
      OtelLogger.error("OmniPart modules download failed", [{"error", inspect(reason)}])
  end
  
  SpanCollector.add_span_attribute("checkpoint.dir", checkpoint_dir)
end)

# Process using OmniPart
SpanCollector.track_span("omnipart.generation", fn ->
try do
  {_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
from pathlib import Path
import shutil

# Enable OpenEXR support in OpenCV (required for .exr file writing)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# Setup Python logging with proper levels
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Verify torch is available
try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    raise ImportError(
        f"torch is not available: {e}. "
        "This should have been installed by uv_init. "
        "Please check that uv_init completed successfully."
    )

# Get configuration from JSON file
""" <> ConfigFile.python_path_string(config_file_normalized) <> ~S"""

# Support multiple images - read as lists
image_paths = config.get('image_paths', [])
mask_paths = config.get('mask_paths', [])
auto_generate_masks = config.get('auto_generate_masks', [])
merge_groups_list = config.get('merge_groups', [])

# Ensure image_paths is a list
if not isinstance(image_paths, list):
    # If it's a string (single image), convert to list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    else:
        image_paths = []

# Backward compatibility: if old single image format, convert to list
if len(image_paths) == 0 and config.get('image_path'):
    image_paths = [config.get('image_path')]
    mask_paths = [config.get('mask_path')] if config.get('mask_path') else []
    auto_generate_masks = [config.get('auto_generate_mask', False)]
    merge_groups_str = config.get('merge_groups')
    if merge_groups_str:
        merge_groups_list = [merge_groups_str]
    else:
        merge_groups_list = []

# Debug: Verify images were read correctly
print(f"[DEBUG] Read {len(image_paths)} image(s) from config")
if len(image_paths) > 1:
    print(f"[DEBUG] Image paths: {[Path(p).name for p in image_paths]}")

segment_only = config.get('segment_only', False)
apply_merge = config.get('apply_merge', False)
output_dir = config.get('output_dir', 'output')
size_threshold = config.get('size_threshold', 2000)
num_inference_steps = config.get('num_inference_steps', 25)
guidance_scale = config.get('guidance_scale', 7.5)
simplify_ratio = config.get('simplify_ratio', 0.15)
gpu = config.get('gpu', 0)
seed = config.get('seed', 42)
omnipart_dir = config.get('omnipart_dir')
checkpoint_dir = config.get('checkpoint_dir')

# Resolve all paths to absolute
image_paths = [str(Path(p).resolve()) for p in image_paths]
mask_paths = [str(Path(p).resolve()) if p else None for p in mask_paths]
output_dir_base = Path(output_dir).resolve()
omnipart_dir = str(Path(omnipart_dir).resolve())
checkpoint_dir = str(Path(checkpoint_dir).resolve())

# Verify all input images exist
for image_path in image_paths:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

# Handle mask paths - ensure lists match length
while len(mask_paths) < len(image_paths):
    mask_paths.append(None)
while len(auto_generate_masks) < len(image_paths):
    auto_generate_masks.append(True)
while len(merge_groups_list) < len(image_paths):
    merge_groups_list.append(None)

for i, mask_path in enumerate(mask_paths):
    if mask_path:
        if not Path(mask_path).exists():
            logger.warning(f"Mask file not found: {mask_path}")
            logger.info("Will auto-generate mask using SAM")
            auto_generate_masks[i] = True
            mask_paths[i] = None
        elif not mask_path.lower().endswith('.exr'):
            logger.warning(f"Mask file is not .exr format: {mask_path}")
            logger.info("OmniPart expects .exr files with shape [h, w, 3] containing 2D part IDs")
    else:
        auto_generate_masks[i] = True

# Verify OmniPart directory exists (optional - can use from Hugging Face)
if not Path(omnipart_dir).exists():
    print(f"[WARN] OmniPart directory not found at {omnipart_dir}")
    print("Will attempt to use OmniPart from Python package or Hugging Face")

print("\n=== Step 2: Prepare Input Data ===")
if len(image_paths) == 1:
    print(f"Images (1): {Path(image_paths[0]).name}")
else:
    print(f"Images ({len(image_paths)}): {', '.join([Path(p).name for p in image_paths])}")

# Create output directory with timestamped subdirectory (matching other generation scripts)
output_dir_base.mkdir(parents=True, exist_ok=True)
import time
tag = time.strftime("%Y%m%d_%H_%M_%S")
output_dir = str(output_dir_base / tag)
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Process each image separately for segmentation
processed_images = []
processed_masks = []
processed_image_paths = []

# Load SAM models once before loop for efficiency (for regular segmentation mode)
# Note: apply_merge mode loads its own SAM models per image
sam_model = None
sam_mask_generator = None
if not apply_merge:
    # Only load SAM if we're doing regular segmentation (not apply_merge)
    if not Path(omnipart_dir).exists():
        raise FileNotFoundError(
            f"OmniPart directory not found at {omnipart_dir}. "
            "Cannot generate mask without OmniPart codebase."
        )
    
    sys.path.insert(0, str(omnipart_dir))
    
    # Import required modules
    try:
        from segment_anything import SamAutomaticMaskGenerator, build_sam
        from modules.label_2d_mask.label_parts import (
            get_sam_mask, 
            clean_segment_edges,
            resize_and_pad_to_square,
        )
        # Visualizer requires detectron2, but we can work around it
        try:
            from modules.label_2d_mask.visualizer import Visualizer
        except ImportError:
            print("[WARN] detectron2 not available, using simplified visualizer")
            # Create a minimal visualizer class that provides the needed methods
            class Visualizer:
                def __init__(self, image):
                    if isinstance(image, np.ndarray):
                        self.img = image.copy()
                    else:
                        self.img = np.array(image)
                    self.output = Image.fromarray(self.img)
                
                def draw_binary_mask(self, binary_mask, color=None, edge_color=None, text=None, alpha=0.7, area_threshold=10):
                    if np.sum(binary_mask) < area_threshold:
                        return self.output
                    return self.output
                
                def draw_binary_mask_with_number(self, binary_mask, color=None, edge_color=None, text=None, 
                                                 label_mode='1', alpha=0.1, anno_mode=['Mask'], area_threshold=10, font_size=None):
                    return self.draw_binary_mask(binary_mask, color, edge_color, text, alpha, area_threshold)
        from PIL import Image
        import numpy as np
        import cv2
        import io
    except ImportError as e:
        raise ImportError(
            f"Failed to import required modules for mask generation: {e}. "
            "Make sure OmniPart dependencies are installed."
        )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load SAM checkpoint (should be downloaded by Elixir)
    sam_ckpt_path = Path(checkpoint_dir) / "sam_vit_h_4b8939.pth"
    if not sam_ckpt_path.exists():
        # Fallback: try to find in OmniPart_modules directory
        omnipart_modules_dir = Path(checkpoint_dir).parent / "OmniPart_modules"
        sam_ckpt_fallback = omnipart_modules_dir / "sam_vit_h_4b8939.pth"
        if sam_ckpt_fallback.exists():
            sam_ckpt_path = sam_ckpt_fallback
        else:
            # Last resort: download via Python
            print("[WARN] SAM checkpoint not found, downloading via Python...")
            from huggingface_hub import hf_hub_download
            sam_ckpt_path = hf_hub_download(
                repo_id="omnipart/OmniPart_modules",
                filename="sam_vit_h_4b8939.pth",
                local_dir=str(checkpoint_dir)
            )
            sam_ckpt_path = Path(sam_ckpt_path)
    
    # Load SAM model once (will be reused for all images)
    print("[INFO] Loading SAM model for segmentation (will be reused for all images)...")
    sam_model = build_sam(checkpoint=str(sam_ckpt_path)).to(device=device)
    sam_mask_generator = SamAutomaticMaskGenerator(sam_model)

for img_idx, image_path in enumerate(image_paths):
    print(f"\n--- Processing image {img_idx + 1}/{len(image_paths)}: {Path(image_path).name} ---")
    mask_path = mask_paths[img_idx] if img_idx < len(mask_paths) else None
    auto_generate_mask = auto_generate_masks[img_idx] if img_idx < len(auto_generate_masks) else True
    merge_groups_str = merge_groups_list[img_idx] if img_idx < len(merge_groups_list) and merge_groups_list[img_idx] else None
    
    # State file for iterative workflow (per image)
    img_name = Path(image_path).stem
    state_file = Path(output_dir) / f"{img_name}_segmentation_state.json"
    
    try:
        # Handle apply_merge mode - load existing state
        if apply_merge:
            if not state_file.exists():
                raise FileNotFoundError(
                    f"Segmentation state not found: {state_file}. "
                    "Please run with --segment-only first to generate initial segmentation."
                )
            
            print("\n=== Step 2: Apply Merge Groups ===")
            print(f"Loading segmentation state from: {state_file}")
            
            import json
            import numpy as np
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            original_group_ids = np.array(state['original_group_ids'])
            processed_image_path = state['processed_image']
            image_array = np.array(state['image'])
            
            if not Path(omnipart_dir).exists():
                raise FileNotFoundError(f"OmniPart directory not found at {omnipart_dir}")
            
            sys.path.insert(0, str(omnipart_dir))
            
            from segment_anything import SamAutomaticMaskGenerator, build_sam
            from modules.label_2d_mask.label_parts import get_sam_mask, clean_segment_edges
            try:
                from modules.label_2d_mask.visualizer import Visualizer
            except ImportError:
                print("[WARN] detectron2 not available, using simplified visualizer")
                # Create a minimal visualizer class that provides the needed methods
                class Visualizer:
                    def __init__(self, image):
                        if isinstance(image, np.ndarray):
                            self.img = image.copy()
                        else:
                            self.img = np.array(image)
                        self.output = Image.fromarray(self.img)
                    
                    def draw_binary_mask(self, binary_mask, color=None, edge_color=None, text=None, alpha=0.7, area_threshold=10):
                        # Simplified binary mask drawing without detectron2
                        if np.sum(binary_mask) < area_threshold:
                            return self.output
                        return self.output
                    
                    def draw_binary_mask_with_number(self, binary_mask, color=None, edge_color=None, text=None, 
                                                     label_mode='1', alpha=0.1, anno_mode=['Mask'], area_threshold=10, font_size=None):
                        # Simplified binary mask drawing with number without detectron2
                        return self.draw_binary_mask(binary_mask, color, edge_color, text, alpha, area_threshold)
            from PIL import Image
            import numpy as np
            import cv2
            
            # Parse merge groups
            if not merge_groups_str:
                raise ValueError("--merge-groups is required when using --apply-merge")
            
            merge_groups = []
            group_sets = merge_groups_str.split(';')
            for group_set in group_sets:
                ids = [int(x.strip()) for x in group_set.split(',') if x.strip()]
                if ids:
                    merge_groups.append(ids)
            
            unique_ids = np.unique(original_group_ids)
            unique_ids = unique_ids[unique_ids >= 0]
            print(f"\n=== Applying Merge Groups ===")
            print(f"Original segment IDs: {sorted(unique_ids.tolist())}")
            print(f"Merge groups to apply: {merge_groups}")
            
            # Load models (lightweight - just SAM for merging)
            device_merge = "cuda" if torch.cuda.is_available() else "cpu"
            sam_ckpt_path = Path(checkpoint_dir) / "sam_vit_h_4b8939.pth"
            if not sam_ckpt_path.exists():
                # Fallback: try to find in OmniPart_modules directory
                omnipart_modules_dir = Path(checkpoint_dir).parent / "OmniPart_modules"
                sam_ckpt_fallback = omnipart_modules_dir / "sam_vit_h_4b8939.pth"
                if sam_ckpt_fallback.exists():
                    sam_ckpt_path = sam_ckpt_fallback
                else:
                    # Last resort: download via Python
                    print("[WARN] SAM checkpoint not found, downloading via Python...")
                    from huggingface_hub import hf_hub_download
                    sam_ckpt_path = hf_hub_download(
                        repo_id="omnipart/OmniPart_modules",
                        filename="sam_vit_h_4b8939.pth",
                        local_dir=str(checkpoint_dir)
                    )
                    sam_ckpt_path = Path(sam_ckpt_path)
            
            sam_model_merge = build_sam(checkpoint=str(sam_ckpt_path)).to(device=device_merge)
            sam_mask_generator_merge = SamAutomaticMaskGenerator(sam_model_merge)
            
            # Apply merge
            processed_image = Image.open(processed_image_path)
            visual = Visualizer(image_array)
            
            new_group_ids, merged_im = get_sam_mask(
                image_array,
                sam_mask_generator_merge,
                visual,
                merge_groups=merge_groups,
                existing_group_ids=original_group_ids,
                rgba_image=processed_image,
                skip_split=True,
                img_name=img_name,
                save_dir=str(output_dir),
                size_threshold=size_threshold
            )
            
            new_unique_ids = np.unique(new_group_ids)
            new_unique_ids = new_unique_ids[new_unique_ids >= 0]
            print(f"\n=== Merge Results ===")
            print(f"Original segment IDs: {sorted(unique_ids.tolist())}")
            print(f"Applied merge groups: {merge_groups}")
            print(f"Final segment IDs: {sorted(new_unique_ids.tolist())}")
            print(f"Result: {len(unique_ids)} segments -> {len(new_unique_ids)} segments")
            
            # Clean edges
            new_group_ids = clean_segment_edges(new_group_ids)
            
            # Save visualization with group numbers (merged_im already has numbers drawn)
            if merged_im is not None:
                vis_path = Path(output_dir) / f"{img_name}_mask_segments_merged_labeled.png"
                # Convert to PIL Image if it's a numpy array
                if isinstance(merged_im, np.ndarray):
                    # Ensure it's uint8 and has the right shape
                    if merged_im.dtype != np.uint8:
                        merged_im = (merged_im * 255).astype(np.uint8) if merged_im.max() <= 1.0 else merged_im.astype(np.uint8)
                    # Handle RGB vs RGBA
                    if len(merged_im.shape) == 3 and merged_im.shape[2] == 3:
                        merged_im = Image.fromarray(merged_im, mode='RGB')
                    elif len(merged_im.shape) == 3 and merged_im.shape[2] == 4:
                        merged_im = Image.fromarray(merged_im, mode='RGBA')
                    else:
                        merged_im = Image.fromarray(merged_im)
                merged_im.save(str(vis_path))
                print(f"[OK] Merged mask visualization with group numbers saved to: {vis_path}")
            
            # Also save simple colored visualization
            from modules.label_2d_mask.label_parts import get_mask
            get_mask(new_group_ids, image_array, ids=3, img_name=img_name, save_dir=str(output_dir))
            
            # Save new mask as .exr (OpenCV with OpenEXR enabled via environment variable)
            save_mask = new_group_ids + 1
            save_mask = save_mask.reshape(518, 518, 1).repeat(3, axis=-1)
            mask_path = str(Path(output_dir) / f"{img_name}_mask.exr")
            # OpenCV with OPENCV_IO_ENABLE_OPENEXR='1' can write .exr files
            cv2.imwrite(mask_path, save_mask.astype(np.float32))
            print(f"[OK] Updated mask saved to: {mask_path}")
            
            # Update state (keep original_group_ids unchanged for future merges)
            state['group_ids'] = new_group_ids.tolist()
            state['save_mask_path'] = mask_path
            with open(state_file, 'w') as f:
                json.dump(state, f)
            
            print("\n=== Merge Complete ===")
            print("You can:")
            print("  1. Run again with --apply-merge and different --merge-groups to refine further")
            print("  2. Run without --apply-merge to generate 3D model")
            
            # Clean up merge-specific SAM models
            del sam_model_merge, sam_mask_generator_merge
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Exit early - don't generate 3D (only for apply_merge mode)
            import sys
            sys.exit(0)
        
        # Handle mask - use provided mask or generate new one (only if not apply_merge)
        if not apply_merge:
            if not auto_generate_mask and mask_path:
                # Use provided mask
                print(f"\n=== Step 2a: Using Provided Mask ===")
                print(f"Using mask: {mask_path}")
                # Load the mask and create processed image
                mask_array = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_array is None:
                    print(f"[WARN] Failed to load mask from {mask_path}, will auto-generate")
                    auto_generate_mask = True
                else:
                    # Mask is provided and loaded successfully
                    # Load original image and process it
                    img = Image.open(image_path).convert("RGB")
                    # Use transparent-background for background removal
                    try:
                        from transparent_background import Remover
                        remover = Remover(mode='base', device=device)
                        processed_image = remover.process(img, type='rgba')
                        alpha_channel = np.array(processed_image.split()[3])
                        alpha_channel = cv2.GaussianBlur(alpha_channel.astype(np.float32), (3, 3), 0.5).astype(np.uint8)
                        processed_image.putalpha(Image.fromarray(alpha_channel))
                        print("[OK] Background removed successfully using transparent-background")
                    except Exception as e:
                        print(f"[WARN] transparent-background removal failed: {e}, using original image")
                        processed_image = img.convert("RGBA")
                    
                    # Resize and pad to square (imported before loop)
                    from modules.label_2d_mask.label_parts import resize_and_pad_to_square
                    processed_image = resize_and_pad_to_square(processed_image)
                    
                    # Save processed image
                    processed_image_path = str(Path(output_dir) / f"{img_name}_processed.png")
                    processed_image.save(processed_image_path)
                    print(f"[OK] Processed image saved to: {processed_image_path}")
                    
                    # Collect results
                    processed_images.append(Image.open(processed_image_path))
                    processed_masks.append(mask_path)
                    processed_image_paths.append(processed_image_path)
                    print(f"[OK] Image {img_idx + 1} processed: mask={mask_path}, processed_image={processed_image_path}")
            
            # Generate mask if needed
            elif auto_generate_mask:
                print("\n=== Step 2a: Generate Segmentation Mask ===")
                print("Mask not provided. Generating automatically using SAM...")
                
                # Use transparent-background (free open-source library) for background removal
                # Powered by InSPyReNet (ACCV 2022) - research-backed, high-quality results
                print("Removing background using transparent-background (InSPyReNet)...")
                try:
                    from transparent_background import Remover
                    
                    # Load image
                    img = Image.open(image_path).convert("RGB")
                    
                    # Initialize remover with 'fast' mode for speed, or 'base' for better quality
                    # 'fast' mode is faster but 'base' provides better edge quality
                    print("Initializing transparent-background remover...")
                    remover = Remover(mode='base', device=device)  # Use 'base' for best quality
                    
                    # Remove background - returns RGBA image with transparent background
                    print("Processing image with transparent-background...")
                    processed_image = remover.process(img, type='rgba')
                    
                    # Apply slight edge smoothing to reduce aliasing (if needed)
                    # transparent-background already produces clean edges, but we can smooth slightly
                    alpha_channel = np.array(processed_image.split()[3])
                    alpha_channel = cv2.GaussianBlur(alpha_channel.astype(np.float32), (3, 3), 0.5).astype(np.uint8)
                    processed_image.putalpha(Image.fromarray(alpha_channel))
                    
                    print("[OK] Background removed successfully using transparent-background")
                    
                except Exception as e:
                    print(f"[WARN] transparent-background removal failed: {e}, using original image")
                    img = Image.open(image_path).convert("RGB")
                    processed_image = img.convert("RGBA")
                
                # Use pre-loaded SAM models for part segmentation (loaded before loop)
                if sam_model is None or sam_mask_generator is None:
                    raise RuntimeError("SAM models not loaded. This should not happen in regular segmentation mode.")
        
                # Resize and pad to square
                processed_image = resize_and_pad_to_square(processed_image)
                
                # Create white background version
                white_bg = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
                white_bg_img = Image.alpha_composite(white_bg, processed_image.convert("RGBA"))
                image = np.array(white_bg_img.convert('RGB'))
                
                # Generate SAM masks
                print(f"Generating segmentation masks (size threshold: {size_threshold})...")
                visual = Visualizer(image)
                
                # Parse merge groups if provided
                merge_groups = None
                if merge_groups_str:
                    merge_groups = []
                    group_sets = merge_groups_str.split(';')
                    for group_set in group_sets:
                        ids = [int(x.strip()) for x in group_set.split(',') if x.strip()]
                        if ids:
                            merge_groups.append(ids)
                    print(f"\n=== Merge Groups Provided ===")
                    print(f"Merge groups to apply after segmentation: {merge_groups}")
                
                # Get segmentation
                group_ids, vis_image = get_sam_mask(
                    image, 
                    sam_mask_generator, 
                    visual, 
                    merge_groups=merge_groups, 
                    rgba_image=processed_image,
                    img_name=img_name,
                    save_dir=str(output_dir),
                    size_threshold=size_threshold
                )
                
                # Clean edges
                group_ids = clean_segment_edges(group_ids)
                
                # Ensure mask indices start from 0 for this image (independent index space per image)
                unique_ids = np.unique(group_ids)
                unique_ids = unique_ids[unique_ids >= 0]  # Exclude background (-1)
                if len(unique_ids) > 0:
                    # Create mapping to ensure indices start from 0
                    index_map = {}
                    for new_idx, old_idx in enumerate(sorted(unique_ids)):
                        index_map[old_idx] = new_idx
                    # Apply remapping
                    remapped_group_ids = np.zeros_like(group_ids) - 1  # Initialize with -1 (background)
                    for old_idx, new_idx in index_map.items():
                        remapped_group_ids[group_ids == old_idx] = new_idx
                    group_ids = remapped_group_ids
                    print(f"[INFO] Remapped mask indices to start from 0: {sorted(unique_ids.tolist())} -> {sorted([index_map[i] for i in unique_ids])}")
                
                # Save visualization with group numbers
                # get_sam_mask already saves original visualization when merge_groups are provided
                # We just need to ensure proper file naming per image
                if vis_image is not None:
                    # Helper function to save visualization
                    def save_vis_image(vis_img, save_path, label_type):
                        if isinstance(vis_img, np.ndarray):
                            # Ensure it's uint8 and has the right shape
                            if vis_img.dtype != np.uint8:
                                vis_img = (vis_img * 255).astype(np.uint8) if vis_img.max() <= 1.0 else vis_img.astype(np.uint8)
                            # Handle RGB vs RGBA
                            if len(vis_img.shape) == 3 and vis_img.shape[2] == 3:
                                vis_img = Image.fromarray(vis_img, mode='RGB')
                            elif len(vis_img.shape) == 3 and vis_img.shape[2] == 4:
                                vis_img = Image.fromarray(vis_img, mode='RGBA')
                            else:
                                vis_img = Image.fromarray(vis_img)
                        vis_img.save(str(save_path))
                        print(f"[OK] {label_type} mask visualization saved to: {save_path}")
                    
                    if merge_groups:
                        # get_sam_mask saves original as {img_name}_mask_segments_original_labeled.png
                        # Copy/rename it to {img_name}_mask_segments_labeled.png for consistency
                        original_vis_path = Path(output_dir) / f"{img_name}_mask_segments_original_labeled.png"
                        labeled_vis_path = Path(output_dir) / f"{img_name}_mask_segments_labeled.png"
                        if original_vis_path.exists():
                            import shutil
                            shutil.copy2(original_vis_path, labeled_vis_path)
                            print(f"[OK] Original labeled mask visualization saved to: {labeled_vis_path}")
                        
                        # Save merged version (current vis_image already has merge applied)
                        vis_path_merged = Path(output_dir) / f"{img_name}_mask_segments_merged_labeled.png"
                        save_vis_image(vis_image, vis_path_merged, "Merged labeled")
                    else:
                        # No merge applied, just save labeled version
                        vis_path_labeled = Path(output_dir) / f"{img_name}_mask_segments_labeled.png"
                        save_vis_image(vis_image, vis_path_labeled, "Labeled")
                
                # Also save the colored mask version (without labels) for comparison
                from modules.label_2d_mask.label_parts import get_mask
                get_mask(group_ids, image, ids="colored", img_name=img_name, save_dir=str(output_dir))
                
                # Show segment IDs
                unique_ids = np.unique(group_ids)
                unique_ids = unique_ids[unique_ids >= 0]  # Exclude background
                print(f"\n[INFO] Found {len(unique_ids)} segments with IDs: {sorted(unique_ids.tolist())}")
                print("[INFO] You can merge segments using --merge-groups (e.g., '0,1;3,4')")
                
                # Save segmentation state for iterative workflow
                state = {
                    'image': image.tolist(),
                    'processed_image': str(Path(output_dir) / f"{img_name}_processed.png"),
                    'group_ids': group_ids.tolist(),
                    'original_group_ids': group_ids.tolist(),  # Keep original for merge operations
                    'img_name': img_name,
                }
                with open(state_file, 'w') as f:
                    import json
                    json.dump(state, f)
                print(f"[OK] Segmentation state saved to: {state_file}")
                
                # Save mask as .exr (OpenCV with OpenEXR enabled via environment variable)
                save_mask = group_ids + 1  # Shift IDs so background is 0, parts start at 1
                
                # Ensure mask is the correct shape (518, 518)
                h, w = save_mask.shape
                if h != 518 or w != 518:
                    print(f"[WARN] Mask shape is {h}x{w}, resizing to 518x518")
                    save_mask = cv2.resize(save_mask.astype(np.float32), (518, 518), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                
                # Ensure mask values are valid (non-negative integers)
                save_mask = np.clip(save_mask, 0, 255).astype(np.int32)
                
                # Reshape to (518, 518, 3) for EXR format
                save_mask = save_mask.reshape(518, 518, 1).repeat(3, axis=-1).astype(np.float32)
                
                mask_path = str(Path(output_dir) / f"{img_name}_mask.exr")
                # OpenCV with OPENCV_IO_ENABLE_OPENEXR='1' can write .exr files
                success = cv2.imwrite(mask_path, save_mask)
                if not success:
                    raise RuntimeError(f"Failed to write mask to {mask_path}")
                print(f"[OK] Generated mask saved to: {mask_path}")
                print(f"[INFO] Mask shape: {save_mask.shape}, value range: [{save_mask.min():.1f}, {save_mask.max():.1f}]")
                
                # Also save processed image for inference
                processed_image_path = str(Path(output_dir) / f"{img_name}_processed.png")
                processed_image.save(processed_image_path)
                print(f"[OK] Processed image saved to: {processed_image_path}")
                
                # Collect processed image and mask for this image
                processed_images.append(Image.open(processed_image_path))
                processed_masks.append(mask_path)
                processed_image_paths.append(processed_image_path)
                print(f"[OK] Image {img_idx + 1} processed: mask={mask_path}, processed_image={processed_image_path}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Failed to process image {img_idx + 1} ({Path(image_path).name}): {type(e).__name__}: {error_msg[:200]}")
        import traceback
        traceback.print_exc()
        # Continue with next image
        continue

# After processing all images, unload SAM models
if len(processed_images) > 0:
    print("\n[INFO] Unloading SAM models and clearing GPU cache after segmentation stage...")
    if 'sam_model' in locals():
        del sam_model
    if 'sam_mask_generator' in locals():
        del sam_mask_generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    print("[OK] Resources cleaned up after segmentation stage")

# If segment-only mode, exit here
if segment_only:
    print("\n=== Segmentation Complete ===")
    print(f"Processed {len(processed_images)} image(s)")
    print("Next steps:")
    print("  1. Review the segmentation visualizations in the output directory")
    print("  2. Run with --apply-merge and --merge-groups to refine segmentation")
    print("     Example: elixir omnipart_generation.exs image.png --apply-merge --merge-groups '0,1;3,4'")
    print("  3. Run without --segment-only to generate 3D model")
    import sys
    sys.exit(0)

# Use first image's mask for 3D generation (or combine masks if needed)
mask_path = processed_masks[0] if processed_masks else None
image_path = processed_image_paths[0] if processed_image_paths else None

if not image_path:
    raise ValueError("No processed images available for 3D generation")

print(f"\nUsing {len(processed_image_paths)} image(s) for 3D generation")
if mask_path:
    print(f"Using mask from first image: {mask_path}")

print("\n=== Step 3: Run OmniPart Inference ===")
print(f"Generating 3D shape with part-aware control from {len(processed_image_paths)} view(s)...")

# Set up environment for SPMD (Single Program Multiple Data)
env = os.environ.copy()
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Memory optimization settings to reduce fragmentation
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
env["CUDA_LAUNCH_BLOCKING"] = "0"  # Set to 1 for debugging, 0 for performance
# Disable some optimizations that might cause numerical issues
env["TORCH_USE_CUDA_DSA"] = "0"
# Also set in os.environ for Python code
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# For SPMD: Use all available GPUs, don't restrict with CUDA_VISIBLE_DEVICES
# Verify CUDA is available and detect number of GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {num_gpus}")
    if num_gpus > 0:
        print(f"[INFO] SPMD mode: Using {num_gpus} GPU(s) for distributed processing")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB)")
        # Primary device for main operations
        device = f"cuda:{gpu}" if gpu < num_gpus else "cuda:0"
        torch.cuda.set_device(int(device.split(':')[1]))
        print(f"Primary device: {device}")
    else:
        device = "cpu"
        print("[WARN] No CUDA devices available, using CPU")
else:
    device = "cpu"
    num_gpus = 0
    print("[WARN] CUDA is not available. Inference will be slower on CPU.")

# Fix nvdiffrast path issue: ensure torch_extensions directory exists with absolute path
# nvdiffrast needs to compile its CUDA extension on first use, and it uses torch.utils.cpp_extension
# which stores compiled extensions in ~/.cache/torch_extensions/
# Use absolute path to avoid path resolution issues
torch_ext_dir = Path.home().resolve() / ".cache" / "torch_extensions"
torch_ext_dir.mkdir(parents=True, exist_ok=True)
# Ensure the directory is writable
os.chmod(str(torch_ext_dir), 0o755)
torch_ext_dir_abs = str(torch_ext_dir.resolve())
print(f"[INFO] torch_extensions cache directory: {torch_ext_dir_abs}")

# Set environment variable with absolute path (for subprocess)
env["TORCH_EXTENSIONS_DIR"] = torch_ext_dir_abs
os.environ["TORCH_EXTENSIONS_DIR"] = torch_ext_dir_abs

# Note: nvdiffrast is not needed since texture baking is disabled (textured=False)
# GLB files will be exported without baked textures, avoiding nvdiffrast dependency

# Set up checkpoint directory
# OmniPart expects checkpoints in ckpt/ directory relative to the OmniPart directory
if Path(omnipart_dir).exists():
    ckpt_dir = Path(omnipart_dir) / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    # If we have a checkpoint_dir from config, symlink or copy checkpoints there
    if Path(checkpoint_dir).exists() and Path(checkpoint_dir) != ckpt_dir:
        print(f"[INFO] Checkpoint directory: {checkpoint_dir}")
        print(f"[INFO] OmniPart will use: {ckpt_dir}")
        print("[INFO] Model weights will be downloaded automatically if needed")

# Set up OmniPart path and run inference directly (no subprocess)
if Path(omnipart_dir).exists():
    # Use local OmniPart installation
    sys.path.insert(0, str(omnipart_dir))
    print(f"[INFO] Using OmniPart from: {omnipart_dir}")
else:
    # Try to use OmniPart as installed package
    try:
        import omnipart
        omnipart_path = Path(omnipart.__file__).parent.parent
        sys.path.insert(0, str(omnipart_path))
        print(f"[INFO] Using OmniPart from installed package: {omnipart_path}")
    except ImportError:
        raise ImportError("OmniPart not found. Please ensure thirdparty/OmniPart exists or OmniPart is installed.")

# Change to OmniPart directory for config loading
original_cwd = os.getcwd()
try:
    if Path(omnipart_dir).exists():
        os.chdir(omnipart_dir)
    else:
        import omnipart
        omnipart_path = Path(omnipart.__file__).parent.parent
        os.chdir(str(omnipart_path))
    
    # Run OmniPart inference directly
    import numpy as np
    from PIL import Image
    from omegaconf import OmegaConf
    
    from modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
    from modules.part_synthesis.process_utils import save_parts_outputs
    from modules.inference_utils import load_img_mask, prepare_bbox_gen_input, prepare_part_synthesis_input, gen_mesh_from_bounds, vis_voxel_coords, merge_parts
    from modules.part_synthesis.pipelines import OmniPartImageTo3DPipeline
    from huggingface_hub import hf_hub_download
    
    # Set device explicitly for SPMD (use all available GPUs)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        # Use the specified GPU as primary, or default to 0
        primary_gpu = min(gpu, num_gpus - 1) if num_gpus > 0 else 0
        device = f"cuda:{primary_gpu}"
        torch.cuda.set_device(primary_gpu)
        print(f"[INFO] SPMD mode: {num_gpus} GPU(s) available")
        print(f"[INFO] Primary device: {device} ({torch.cuda.get_device_name(primary_gpu)})")
        use_spmd = num_gpus > 1
    else:
        device = "cpu"
        use_spmd = False
        print("[WARN] CUDA not available, using CPU")
    
    # Set up paths
    partfield_encoder_path = "ckpt/model_objaverse.ckpt"
    bbox_gen_ckpt = "ckpt/bbox_gen.ckpt"
    part_synthesis_ckpt = "omnipart/OmniPart"
    
    # Download checkpoints if needed
    if not os.path.exists(partfield_encoder_path):
        partfield_encoder_path = hf_hub_download(repo_id="omnipart/OmniPart_modules", filename="partfield_encoder.ckpt", local_dir="ckpt")
    if not os.path.exists(bbox_gen_ckpt):
        bbox_gen_ckpt = hf_hub_download(repo_id="omnipart/OmniPart_modules", filename="bbox_gen.ckpt", local_dir="ckpt")
    
    os.makedirs(output_dir, exist_ok=True)
    inference_output_dir = os.path.join(output_dir, Path(image_path).stem)
    os.makedirs(inference_output_dir, exist_ok=True)
    
    torch.manual_seed(seed)
    
    # Load part_synthesis model with half precision for memory efficiency
    # Load models (reduced verbosity)
    part_synthesis_pipeline = OmniPartImageTo3DPipeline.from_pretrained(part_synthesis_ckpt, use_fp16=True)
    # Explicitly move to device and ensure not using DataParallel (which uses multiple GPUs)
    part_synthesis_pipeline.to(device)
    # Ensure all sub-models are on the correct device and not wrapped in DataParallel
    if hasattr(part_synthesis_pipeline, 'models'):
        for model_name, model in part_synthesis_pipeline.models.items():
            if model is not None:
                # Remove DataParallel wrapper if present (would try to use multiple GPUs)
                if isinstance(model, torch.nn.DataParallel):
                    print(f"[WARN] {model_name} was wrapped in DataParallel, unwrapping...")
                    part_synthesis_pipeline.models[model_name] = model.module
                    model = model.module
                model.to(device)
                # Verify device placement
                if hasattr(model, 'parameters'):
                    first_param = next(model.parameters(), None)
                    if first_param is not None:
                        param_device = first_param.device
                        if str(param_device) != device:
                            print(f"[WARN] {model_name} parameters on {param_device}, moving to {device}")
                            model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Ensure all operations complete
    
    # Load bbox_gen model
    bbox_gen_config = OmegaConf.load("configs/bbox_gen.yaml").model.args
    bbox_gen_config.partfield_encoder_path = partfield_encoder_path
    bbox_gen_model = BboxGen(bbox_gen_config)
    bbox_gen_model.load_state_dict(torch.load(bbox_gen_ckpt), strict=False)
    bbox_gen_model.to(device)
    bbox_gen_model.eval().half()
    
    # Load all images for multiview generation
    print(f"[INFO] Loading {len(processed_image_paths)} image(s) for multiview 3D generation...")
    
    # Load mask from first image (for part layout)
    img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis = load_img_mask(processed_image_paths[0], mask_path)
    img_mask_vis.save(os.path.join(inference_output_dir, "img_mask_vis.png"))
    
    # Load all images as PIL Images for multiview
    images_list = [Image.open(img_path) for img_path in processed_image_paths]
    print(f"[DEBUG] Loaded {len(images_list)} image(s) for multiview: {[Path(p).name for p in processed_image_paths]}")
    
    # Generate voxel coordinates with performance optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Enable benchmark for faster inference
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
    # Generate voxel coordinates from all images (multiview)
    print(f"[INFO] Generating voxel coordinates from {len(images_list)} view(s)...")
    voxel_coords = part_synthesis_pipeline.get_coords(images_list, num_samples=1, seed=seed, sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5})
    voxel_coords = voxel_coords.cpu().numpy()
    np.save(os.path.join(inference_output_dir, "voxel_coords.npy"), voxel_coords)
    voxel_coords_ply = vis_voxel_coords(voxel_coords)
    voxel_coords_ply.export(os.path.join(inference_output_dir, "voxel_coords_vis.ply"))
    # Clear cache after voxel coordinate generation
    del voxel_coords_ply
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Generate bounding boxes (reduced verbosity)
    bbox_gen_input = prepare_bbox_gen_input(os.path.join(inference_output_dir, "voxel_coords.npy"), img_white_bg, ordered_mask_input)
    bbox_gen_output = bbox_gen_model.generate(bbox_gen_input)
    np.save(os.path.join(inference_output_dir, "bboxes.npy"), bbox_gen_output['bboxes'][0])
    bboxes_vis = gen_mesh_from_bounds(bbox_gen_output['bboxes'][0])
    bboxes_vis.export(os.path.join(inference_output_dir, "bboxes_vis.glb"))
    
    # Clear bbox generation intermediates
    del bbox_gen_input
    del bbox_gen_output
    del bboxes_vis
    
    # Unload bbox_gen model to free GPU memory before SLAT sampling (reduced verbosity)
    del bbox_gen_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Prepare part synthesis input
    part_synthesis_input = prepare_part_synthesis_input(os.path.join(inference_output_dir, "voxel_coords.npy"), os.path.join(inference_output_dir, "bboxes.npy"), ordered_mask_input)
    
    # Validate inputs (reduced verbosity - debug only)
    # logger.debug(f"Coords shape: {part_synthesis_input['coords'].shape if isinstance(part_synthesis_input['coords'], torch.Tensor) else len(part_synthesis_input['coords'])}")
    # logger.debug(f"Part layouts: {len(part_synthesis_input['part_layouts'])} parts")
    # logger.debug(f"Masks shape: {part_synthesis_input['masks'].shape}")
    
    # Sample SLAT with performance optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Enable benchmark for faster inference
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
    # Enable memory-efficient attention if available
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except:
        pass
    # Sampling SLAT (reduced verbosity)
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Use all images for multiview conditioning
    print(f"[INFO] Using {len(images_list)} image(s) for SLAT generation...")
    print(f"[DEBUG] Conditioning on images: {[Path(p).name for p in processed_image_paths]}")
    cond = part_synthesis_pipeline.get_cond(images_list)
    print(f"[DEBUG] Condition shape: {cond['cond'].shape if isinstance(cond, dict) and 'cond' in cond else 'N/A'}")
    
    # Clear memory before SLAT sampling
    torch.cuda.empty_cache()
    gc.collect()
    
    torch.manual_seed(seed)
    slat = part_synthesis_pipeline.sample_slat(
        cond, 
        part_synthesis_input['coords'], 
        [part_synthesis_input['part_layouts']], 
        part_synthesis_input['masks'],
        sampler_params={"steps": num_inference_steps, "cfg_strength": guidance_scale},
    )
    
    # Clear cond and intermediate tensors after SLAT sampling
    del cond
    if 'img_batched' in locals():
        del img_batched
    torch.cuda.empty_cache()
    gc.collect()
    
    # Divide SLAT into parts
    divided_slat = part_synthesis_pipeline.divide_slat(slat, [part_synthesis_input['part_layouts']])
    
    # Clear original slat after division
    del slat
    torch.cuda.empty_cache()
    gc.collect()
    
    # Clear memory after SLAT division
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # True SPMD: Split SLAT data across GPUs - each GPU processes different coordinate subsets
    part_synthesis_output = {}
    formats_to_generate = ['mesh', 'gaussian']  # Removed radiance_field - too memory intensive
    
    if use_spmd and num_gpus > 1 and hasattr(divided_slat, 'coords') and hasattr(divided_slat, 'feats'):
        # True SPMD: Split the coordinate/feature data across GPUs
        num_points = divided_slat.coords.shape[0]
        points_per_gpu = (num_points + num_gpus - 1) // num_gpus  # Ceiling division
        
        print(f"[INFO] SPMD: Splitting {num_points} points across {num_gpus} GPUs")
        print(f"[INFO] SPMD: ~{points_per_gpu} points per GPU (same decoder program, different data)")
        
        for fmt in formats_to_generate:
            try:
                print(f"\n[INFO] SPMD: Decoding {fmt} format across {num_gpus} GPUs...")
                decoder_key = f'slat_decoder_{fmt}'
                
                if not (hasattr(part_synthesis_pipeline, 'models') and decoder_key in part_synthesis_pipeline.models):
                    print(f"[WARN] Decoder {decoder_key} not found, skipping")
                    continue
                
                decoder = part_synthesis_pipeline.models[decoder_key]
                if decoder is None:
                    print(f"[WARN] Decoder {decoder_key} is None, skipping")
                    continue
                
                # Split SLAT data across GPUs and process sequentially with proper isolation
                gpu_outputs = []
                for gpu_id in range(num_gpus):
                    start_idx = gpu_id * points_per_gpu
                    end_idx = min((gpu_id + 1) * points_per_gpu, num_points)
                    
                    if start_idx >= num_points:
                        break
                    
                    target_device = f"cuda:{gpu_id}"
                    
                    # Clear ALL GPUs before processing this one
                    for d in range(num_gpus):
                        torch.cuda.set_device(d)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(d)
                    gc.collect()
                    
                    # Set target GPU as current device
                    torch.cuda.set_device(gpu_id)
                    
                    # Create subset of SLAT for this GPU (SPMD: same program, different data)
                    # Create on CPU first, then move to target GPU to avoid device conflicts
                    subset_coords = divided_slat.coords[start_idx:end_idx].cpu()
                    subset_feats = divided_slat.feats[start_idx:end_idx].cpu()
                    
                    # Adjust batch IDs to maintain continuity
                    if subset_coords.shape[0] > 0:
                        subset_coords[:, 0] = 0  # All points in same batch for this GPU
                    
                    # Move to target GPU
                    subset_coords = subset_coords.to(target_device)
                    subset_feats = subset_feats.to(target_device)
                    
                    # Create SparseTensor subset for this GPU
                    from modules.part_synthesis.modules.sparse.basic import SparseTensor
                    slat_subset = SparseTensor(
                        coords=subset_coords,
                        feats=subset_feats
                    )
                    
                    # Create a fresh decoder instance for this GPU to avoid device conflicts
                    # Deep copy the decoder state to ensure complete isolation
                    import copy
                    decoder_copy = copy.deepcopy(decoder)
                    decoder_copy.to(target_device)
                    decoder_copy.eval()
                    
                    # Verify all parameters are on correct device
                    all_on_device = all(p.device == torch.device(target_device) for p in decoder_copy.parameters())
                    if not all_on_device:
                        print(f"[WARN] SPMD: Some decoder parameters not on {target_device}, skipping GPU {gpu_id}")
                        del decoder_copy, slat_subset
                        gpu_outputs.append((gpu_id, None))
                        continue
                    
                    print(f"[INFO] SPMD: GPU {gpu_id} processing {subset_coords.shape[0]} points...")
                    try:
                        # Decode subset on this GPU (same program, different data = SPMD)
                        with torch.cuda.device(gpu_id):
                            gpu_output = decoder_copy(slat_subset)
                        # Move output to CPU to free GPU memory
                        if isinstance(gpu_output, list):
                            gpu_output = [out.cpu() if hasattr(out, 'cpu') else out for out in gpu_output]
                        elif hasattr(gpu_output, 'cpu'):
                            gpu_output = gpu_output.cpu()
                        gpu_outputs.append((gpu_id, gpu_output))
                        print(f"[OK] SPMD: GPU {gpu_id} decoded {subset_coords.shape[0]} points")
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "device" in error_msg.lower() or "cuda" in error_msg.lower():
                            print(f"[WARN] SPMD: GPU {gpu_id} device error: {error_msg[:200]}")
                        else:
                            print(f"[WARN] SPMD: GPU {gpu_id} failed: {type(e).__name__}: {error_msg[:200]}")
                        gpu_outputs.append((gpu_id, None))
                    except Exception as e:
                        print(f"[WARN] SPMD: GPU {gpu_id} failed: {type(e).__name__}: {str(e)[:200]}")
                        gpu_outputs.append((gpu_id, None))
                    
                    # Clean up this GPU's resources
                    del decoder_copy, slat_subset, subset_coords, subset_feats
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(gpu_id)
                    gc.collect()
                
                # Collect and merge results from all GPUs
                valid_outputs = [out for gpu_id, out in gpu_outputs if out is not None]
                if valid_outputs:
                    # If only one GPU succeeded, use that output
                    if len(valid_outputs) == 1:
                        part_synthesis_output[fmt] = valid_outputs[0]
                        print(f"[OK] SPMD: Successfully decoded {fmt} format on 1 GPU")
                    elif len(valid_outputs) == num_gpus:
                        # All GPUs succeeded - merge outputs
                        # For mesh/gaussian, we concatenate the results
                        if isinstance(valid_outputs[0], list):
                            # Flatten list of lists
                            merged = []
                            for out in valid_outputs:
                                if isinstance(out, list):
                                    merged.extend(out)
                                else:
                                    merged.append(out)
                            part_synthesis_output[fmt] = merged
                        else:
                            # Single outputs - use first (or merge if needed)
                            part_synthesis_output[fmt] = valid_outputs[0]
                        print(f"[OK] SPMD: Successfully decoded {fmt} format on all {num_gpus} GPUs")
                    else:
                        # Partial success - use first valid output
                        part_synthesis_output[fmt] = valid_outputs[0]
                        print(f"[WARN] SPMD: Partial success - {len(valid_outputs)}/{num_gpus} GPUs decoded {fmt} format")
                else:
                    print(f"[ERROR] SPMD: All GPUs failed for {fmt} format")
                    
            except Exception as e:
                print(f"[WARN] SPMD: Failed to decode {fmt} format: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        # Fallback: Sequential processing (single GPU or not enough parts)
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            print(f"[INFO] Available GPU memory: {free_memory_gb:.2f} GB")
            if free_memory_gb < 1.0:
                print("[WARN] Low GPU memory detected, only generating mesh format")
                formats_to_generate = ['mesh']
        
        for fmt in formats_to_generate:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                if torch.cuda.is_available():
                    allocated_before = torch.cuda.memory_allocated(0) / (1024**3)
                    print(f"[INFO] Decoding {fmt} format (GPU memory before: {allocated_before:.2f} GB)...")
                
                fmt_output = part_synthesis_pipeline.decode_slat(divided_slat, [fmt])
                if fmt in fmt_output:
                    part_synthesis_output[fmt] = fmt_output[fmt]
                    print(f"[OK] Successfully decoded {fmt} format")
                
                del fmt_output
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "CUDA" in error_msg:
                    print(f"[WARN] Out of memory while decoding {fmt} format, skipping...")
                else:
                    print(f"[WARN] Failed to decode {fmt} format: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                continue
            except Exception as e:
                print(f"[WARN] Failed to decode {fmt} format: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                continue
    
    if not part_synthesis_output:
        raise RuntimeError("Failed to generate any format")
    
    print(f"[INFO] Successfully generated formats: {list(part_synthesis_output.keys())}")
    
    # Save outputs
    print("[INFO] Saving outputs...")
    save_parts_outputs(
        part_synthesis_output, 
        output_dir=inference_output_dir, 
        simplify_ratio=simplify_ratio, 
        save_video=False,
        save_glb=True,
        textured=False,  # Disabled texture baking to prevent OOM errors
    )
    
    # Merge parts
    print("[INFO] Merging parts...")
    merge_parts(inference_output_dir)
    
    print("[OK] OmniPart inference completed successfully")
    
    # Unload models and clear GPU cache after 3D generation stage
    print("\n[INFO] Unloading models and clearing GPU cache...")
    if 'part_synthesis_pipeline' in locals():
        # Clear pipeline models
        if hasattr(part_synthesis_pipeline, 'models'):
            for model in part_synthesis_pipeline.models.values():
                del model
        del part_synthesis_pipeline
    if 'bbox_gen_model' in locals():
        del bbox_gen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations complete
        import gc
        gc.collect()
    print("[OK] Resources cleaned up after 3D generation stage")
    
finally:
    os.chdir(original_cwd)

# Find output files
print("\n=== Step 4: Locate Output Files ===")
# OmniPart creates a subdirectory based on image filename within the timestamped output directory
# and generates multiple outputs:
# - mesh_segment.glb (merged parts)
# - bboxes_vis.glb (bounding boxes visualization)
# - voxel_coords_vis.ply (voxel coordinates)
# - merged_gs.ply (merged 3D Gaussians)
# - exploded_gs.ply (exploded 3D Gaussians)
image_name = Path(image_path).stem
# inference_output_dir is the subdirectory created by inference (same as image_name)
search_dir = Path(inference_output_dir) if 'inference_output_dir' in locals() else Path(output_dir) / image_name
if search_dir.exists():
    print(f"Found output subdirectory: {search_dir}")
else:
    # Fallback: search in main output directory
    search_dir = Path(output_dir)
    print(f"Searching in output directory: {search_dir}")

output_files = []
for fmt in ["glb", "ply"]:
    files = list(search_dir.rglob(f"*.{fmt}"))
    output_files.extend(files)

if output_files:
    print(f"\n=== Complete ===")
    print(f"Generated 3D shape(s) saved to:")
    for output_file in sorted(output_files):
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  - {output_file} ({size_mb:.2f} MB)")
    
    # Find main output (mesh_segment.glb is the primary merged mesh)
    main_outputs = [f for f in output_files if "mesh_segment" in f.name or "merged" in f.name]
    if main_outputs:
        main_output = main_outputs[0]
        print(f"\nMain output file: {main_output}")
    elif output_files:
        main_output = max(output_files, key=lambda p: p.stat().st_size)
        print(f"\nLargest output file: {main_output}")
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

# Track completion as span attribute
SpanCollector.add_span_attribute("status", "completed")
end)

# Export spans and metrics as JSON
SpanCollector.display_trace()

