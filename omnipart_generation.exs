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
#   --image, -i <path>            Input image path (required, RGBA PNG recommended)
#   --mask, -m <path>             Segmentation mask path (.exr file with 2D part IDs) (optional)
#   --output-dir, -o <path>       Output directory (default: outputs)
#   --segment-only                 Only generate segmentation, don't generate 3D model
#   --apply-merge                  Apply merge groups to existing segmentation state
#   --size-threshold <int>        Minimum segment size in pixels for mask generation (default: 2000)
#   --merge-groups <string>       Merge groups for mask (e.g., "0,1;3,4" to merge segments 0&1 and 3&4)
#   --num-inference-steps <int>   Number of inference steps (default: 25)
#   --guidance-scale <float>      Guidance scale for SLat sampler (default: 7.5)
#   --simplify-ratio <float>      Mesh simplification ratio (default: 0.3)
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
      elixir omnipart_generation.exs --image <path> [--mask <path>] [options]
    
    Arguments:
      <image_path>                 Input image file (required)
      <mask_path>                  Segmentation mask file (.exr with 2D part IDs) (optional)
    
    Options:
      --image, -i <path>            Input image path (required if not positional, RGBA PNG recommended)
      --mask, -m <path>             Segmentation mask path (.exr file) (optional)
      --output-dir, -o <path>       Output directory (default: output)
      --segment-only                 Only generate segmentation, don't generate 3D model
      --apply-merge                  Apply merge groups to existing segmentation state
      --size-threshold <int>        Minimum segment size in pixels for mask generation (default: 2000)
      --merge-groups <string>       Merge groups for mask (e.g., "0,1;3,4" to merge segments 0&1 and 3&4)
      --num-inference-steps <int>   Number of inference steps (default: 25)
      --guidance-scale <float>      Guidance scale for SLat sampler (default: 7.5)
      --simplify-ratio <float>      Mesh simplification ratio (default: 0.3)
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

    # Get image and mask paths (positional or from options)
    image_path = Keyword.get(opts, :image) || Enum.at(args, 0)
    mask_path = Keyword.get(opts, :mask) || Enum.at(args, 1)

    if !image_path do
      IO.puts("""
      Error: Image path is required.
      
      Usage:
        elixir omnipart_generation.exs <image_path> [mask_path] [options]
        elixir omnipart_generation.exs --image <path> [--mask <path>] [options]
      
      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    # Check if image file exists
    if !File.exists?(image_path) do
      IO.puts("Error: Image file not found: #{image_path}")
      System.halt(1)
    end

    # Mask is optional - will be auto-generated if not provided
    auto_generate_mask = !mask_path || !File.exists?(mask_path)
    if mask_path && !File.exists?(mask_path) do
      IO.puts("[INFO] Mask file not found: #{mask_path}")
      IO.puts("[INFO] Will auto-generate mask using SAM")
      ^mask_path = nil
      ^auto_generate_mask = true
    end

    segment_only = Keyword.get(opts, :segment_only, false)
    apply_merge = Keyword.get(opts, :apply_merge, false)
    
    # Validate mode
    if segment_only && apply_merge do
      IO.puts("Error: Cannot use --segment-only and --apply-merge together")
      System.halt(1)
    end
    
    %{
      image_path: image_path,
      mask_path: mask_path,
      auto_generate_mask: auto_generate_mask,
      segment_only: segment_only,
      apply_merge: apply_merge,
      output_dir: Keyword.get(opts, :output_dir, "output"),
      size_threshold: Keyword.get(opts, :size_threshold, 2000),
      merge_groups: Keyword.get(opts, :merge_groups),
      num_inference_steps: Keyword.get(opts, :num_inference_steps, 25),
      guidance_scale: Keyword.get(opts, :guidance_scale, 7.5),
        simplify_ratio: Keyword.get(opts, :simplify_ratio, 0.3),
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

IO.puts("""
=== OmniPart Generation ===
Mode: #{mode}
Image: #{config.image_path}
Mask: #{if config.mask_path, do: config.mask_path, else: "Auto-generate using SAM"}
Output Directory: #{config.output_dir}
#{if config.auto_generate_mask || config.segment_only, do: "Size Threshold: #{config.size_threshold}\n", else: ""}#{if config.merge_groups, do: "Merge Groups: #{config.merge_groups}\n", else: ""}#{if not config.segment_only, do: "Inference Steps: #{config.num_inference_steps}\nGuidance Scale: #{config.guidance_scale}\nSimplify Ratio: #{config.simplify_ratio}\n", else: ""}GPU: #{config.gpu}
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

# Save config to JSON for Python to read
{config_file, config_file_normalized} = ConfigFile.create(config_with_paths, "omnipart_config")

# Download checkpoints using Elixir downloader
SpanCollector.track_span("omnipart.download_weights", fn ->
  IO.puts("\n=== Step 1: Download Pretrained Weights ===")
  
  checkpoint_dir = config_with_paths.checkpoint_dir
  File.mkdir_p!(checkpoint_dir)
  
  # Download OmniPart modules (includes SAM checkpoint)
  omnipart_modules_dir = Path.join([checkpoint_dir, "..", "OmniPart_modules"])
  case HuggingFaceDownloader.download_repo("omnipart/OmniPart_modules", omnipart_modules_dir, "OmniPart Modules", true) do
    {:ok, _} ->
      IO.puts("[OK] OmniPart modules downloaded successfully")
      # Move SAM checkpoint to checkpoint_dir if needed
      sam_ckpt_src = Path.join([omnipart_modules_dir, "sam_vit_h_4b8939.pth"])
      sam_ckpt_dst = Path.join([checkpoint_dir, "sam_vit_h_4b8939.pth"])
      if File.exists?(sam_ckpt_src) and not File.exists?(sam_ckpt_dst) do
        File.cp!(sam_ckpt_src, sam_ckpt_dst)
        IO.puts("[OK] SAM checkpoint copied to checkpoint directory")
      end
    {:error, _} ->
      IO.puts("[WARN] OmniPart modules download had errors, but continuing...")
      IO.puts("[INFO] Model weights will be downloaded automatically by Python if needed")
  end
  
  # Note: Using transparent-background (free open-source library) for background removal
  # Powered by InSPyReNet (ACCV 2022) - research-backed, high-quality results
  
  IO.puts("[OK] Checkpoint directory ready: #{checkpoint_dir}")
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

# Enable OpenEXR support in OpenCV (required for .exr file writing)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

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
auto_generate_mask = config.get('auto_generate_mask', False)
segment_only = config.get('segment_only', False)
apply_merge = config.get('apply_merge', False)
output_dir = config.get('output_dir', 'output')
size_threshold = config.get('size_threshold', 2000)
merge_groups_str = config.get('merge_groups')
num_inference_steps = config.get('num_inference_steps', 25)
guidance_scale = config.get('guidance_scale', 7.5)
simplify_ratio = config.get('simplify_ratio', 0.3)
gpu = config.get('gpu', 0)
seed = config.get('seed', 42)
omnipart_dir = config.get('omnipart_dir')
checkpoint_dir = config.get('checkpoint_dir')

# Resolve paths to absolute
image_path = str(Path(image_path).resolve())
output_dir_base = Path(output_dir).resolve()
omnipart_dir = str(Path(omnipart_dir).resolve())
checkpoint_dir = str(Path(checkpoint_dir).resolve())

# Verify input image exists
if not Path(image_path).exists():
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Handle mask path
if mask_path:
    mask_path = str(Path(mask_path).resolve())
    if not Path(mask_path).exists():
        print(f"[WARN] Mask file not found: {mask_path}")
        print("[INFO] Will auto-generate mask using SAM")
        auto_generate_mask = True
        mask_path = None
    elif not mask_path.lower().endswith('.exr'):
        print(f"[WARN] Mask file is not .exr format: {mask_path}")
        print("OmniPart expects .exr files with shape [h, w, 3] containing 2D part IDs")
else:
    auto_generate_mask = True

# Verify OmniPart directory exists (optional - can use from Hugging Face)
if not Path(omnipart_dir).exists():
    print(f"[WARN] OmniPart directory not found at {omnipart_dir}")
    print("Will attempt to use OmniPart from Python package or Hugging Face")

print("\n=== Step 2: Prepare Input Data ===")
print(f"Image: {image_path}")

# Create output directory with timestamped subdirectory (matching other generation scripts)
output_dir_base.mkdir(parents=True, exist_ok=True)
import time
tag = time.strftime("%Y%m%d_%H_%M_%S")
output_dir = str(output_dir_base / tag)
Path(output_dir).mkdir(parents=True, exist_ok=True)

# State file for iterative workflow
img_name = Path(image_path).stem
state_file = Path(output_dir) / f"{img_name}_segmentation_state.json"

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
    print(f"Original segment IDs: {sorted(unique_ids.tolist())}")
    print(f"Merge groups: {merge_groups}")
    
    # Load models (lightweight - just SAM for merging)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    sam_model = build_sam(checkpoint=str(sam_ckpt_path)).to(device=device)
    sam_mask_generator = SamAutomaticMaskGenerator(sam_model)
    
    # Apply merge
    processed_image = Image.open(processed_image_path)
    visual = Visualizer(image_array)
    
    new_group_ids, merged_im = get_sam_mask(
        image_array,
        sam_mask_generator,
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
    print(f"New segment IDs (after merging): {sorted(new_unique_ids.tolist())}")
    
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
    
    # Exit early - don't generate 3D
    import sys
    sys.exit(0)

# Generate mask if needed
if auto_generate_mask:
    print("\n=== Step 2a: Generate Segmentation Mask ===")
    print("Mask not provided. Generating automatically using SAM...")
    
    if not Path(omnipart_dir).exists():
        raise FileNotFoundError(
            f"OmniPart directory not found at {omnipart_dir}. "
            "Cannot generate mask without OmniPart codebase."
        )
    
    # Add OmniPart to path
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
                    # Simplified binary mask drawing without detectron2
                    if np.sum(binary_mask) < area_threshold:
                        return self.output
                    # Simple overlay - just return the image for now
                    # The visualization is optional, core functionality doesn't need it
                    return self.output
                
                def draw_binary_mask_with_number(self, binary_mask, color=None, edge_color=None, text=None, 
                                                 label_mode='1', alpha=0.1, anno_mode=['Mask'], area_threshold=10, font_size=None):
                    # Simplified binary mask drawing with number without detectron2
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
        print(f"[WARN] transparent-background removal failed: {e}")
        print("[INFO] Falling back to SAM-based background removal...")
        
        # Fallback to original SAM
        from segment_anything import SamAutomaticMaskGenerator, build_sam
        sam_model = build_sam(checkpoint=str(sam_ckpt_path)).to(device=device)
        sam_mask_generator = SamAutomaticMaskGenerator(sam_model)
        
        # Process image
        print("Removing background using SAM...")
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Generate SAM masks
        print("Generating SAM masks for background removal...")
        masks = sam_mask_generator.generate(img_array)
        
        # Sort masks by area (largest first) and combine
        sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        foreground_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=bool)
        
        for mask_data in sorted_masks:
            mask = mask_data["segmentation"]
            foreground_mask = np.logical_or(foreground_mask, mask)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        foreground_mask = foreground_mask.astype(bool)
        
        # Create RGBA image
        processed_image = Image.fromarray(img_array).convert("RGBA")
        alpha_channel = np.array(processed_image.split()[3])
        alpha_channel[~foreground_mask] = 0
        alpha_channel = cv2.GaussianBlur(alpha_channel.astype(np.float32), (5, 5), 1.0).astype(np.uint8)
        processed_image.putalpha(Image.fromarray(alpha_channel))
    
    # Also load SAM for later segmentation (still needed for part segmentation)
    print("Loading SAM model for part segmentation...")
    sam_model = build_sam(checkpoint=str(sam_ckpt_path)).to(device=device)
    sam_mask_generator = SamAutomaticMaskGenerator(sam_model)
    
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
        print(f"Merge groups: {merge_groups}")
    
    # Get segmentation
    group_ids, vis_image = get_sam_mask(
        image, 
        sam_mask_generator, 
        visual, 
        merge_groups=merge_groups, 
        rgba_image=processed_image,
        img_name=Path(image_path).stem,
        save_dir=str(output_dir),
        size_threshold=size_threshold
    )
    
    # Clean edges
    group_ids = clean_segment_edges(group_ids)
    
    # Save visualization with group numbers
    if vis_image is not None:
        vis_path = Path(output_dir) / f"{img_name}_mask_segments_labeled.png"
        # Convert to PIL Image if it's a numpy array
        if isinstance(vis_image, np.ndarray):
            # Ensure it's uint8 and has the right shape
            if vis_image.dtype != np.uint8:
                vis_image = (vis_image * 255).astype(np.uint8) if vis_image.max() <= 1.0 else vis_image.astype(np.uint8)
            # Handle RGB vs RGBA
            if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
                vis_image = Image.fromarray(vis_image, mode='RGB')
            elif len(vis_image.shape) == 3 and vis_image.shape[2] == 4:
                vis_image = Image.fromarray(vis_image, mode='RGBA')
            else:
                vis_image = Image.fromarray(vis_image)
        vis_image.save(str(vis_path))
        print(f"[OK] Mask visualization with group numbers saved to: {vis_path}")
    
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
    
    # Update image path to use processed image (with alpha channel)
    image_path = processed_image_path
    
    # If segment-only mode, exit here
    if segment_only:
        print("\n=== Segmentation Complete ===")
        print("Next steps:")
        print("  1. Review the segmentation visualizations in the output directory")
        print("  2. Run with --apply-merge and --merge-groups to refine segmentation")
        print("     Example: elixir omnipart_generation.exs image.png --apply-merge --merge-groups '0,1;3,4'")
        print("  3. Run without --segment-only to generate 3D model")
        import sys
        sys.exit(0)
else:
    print(f"Mask: {mask_path}")

print("\n=== Step 3: Run OmniPart Inference ===")
print(f"Generating 3D shape with part-aware control...")

# Set up environment
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu)
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Add numerical stability settings
env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
env["CUDA_LAUNCH_BLOCKING"] = "0"  # Set to 1 for debugging, 0 for performance
# Disable some optimizations that might cause numerical issues
env["TORCH_USE_CUDA_DSA"] = "0"
# Verify CUDA is available
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Using GPU {gpu}: {torch.cuda.get_device_name(0)}")
else:
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
        "--output_root", output_dir,
        "--seed", str(seed),
        "--num_inference_steps", str(num_inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--simplify_ratio", str(simplify_ratio),
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
            "--output_root", output_dir,
            "--seed", str(seed),
            "--num_inference_steps", str(num_inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--simplify_ratio", str(simplify_ratio),
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
            "--output_root", output_dir,
            "--seed", str(seed),
            "--num_inference_steps", str(num_inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--simplify_ratio", str(simplify_ratio),
        ]
        cwd = str(temp_omnipart)
        sys.path.insert(0, str(temp_omnipart))

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
# OmniPart creates a subdirectory based on image filename within the timestamped output directory
# and generates multiple outputs:
# - mesh_segment.glb (merged parts)
# - bboxes_vis.glb (bounding boxes visualization)
# - voxel_coords_vis.ply (voxel coordinates)
# - merged_gs.ply (merged 3D Gaussians)
# - exploded_gs.ply (exploded 3D Gaussians)
image_name = Path(image_path).stem
expected_subdir = Path(output_dir) / f"{image_name}_processed"
if expected_subdir.exists():
    search_dir = expected_subdir
    print(f"Found output subdirectory: {search_dir}")
else:
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
end)

IO.puts("\n=== Complete ===")
IO.puts("3D shape generation completed successfully!")

# Display OpenTelemetry trace
SpanCollector.display_trace()

