#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# TEXGen Texture Inpainting Script
# Generate, inpaint, or complete textures for 3D meshes using TEXGen
# Repository: https://github.com/CVMI-Lab/TEXGen
# Hugging Face: https://huggingface.co/Andyx/TEXGen
#
# Usage:
#   elixir texgen_texture_inpainting.exs <model_dir> <text_prompt> [options]
#   elixir texgen_texture_inpainting.exs --model-dir <dir> --prompt <text> [options]
#
# Options:
#   --model-dir <path>            Directory containing model.obj, model.mtl, model.png (required if not positional)
#   --prompt, -p <text>            Text prompt describing the desired texture (required if not positional)
#   --num-steps, --steps <int>    Number of inference steps (default: 30)
#   --guidance-scale, -g <float>  Guidance scale (default: 2.0)
#   --seed, -s <int>              Random seed (default: 42)
#   --gpu <int>                  GPU ID to use (default: 0)
#   --output-dir <path>          Output directory (default: outputs)

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
# TEXGen uses PyTorch Lightning, diffusers, and various 3D processing libraries
# All dependencies managed by uv (no pip)
# Based on official requirements: https://github.com/CVMI-Lab/TEXGen

# Get absolute path to thirdparty/torchsparse for local build
base_dir = Path.expand(".")
torchsparse_path = Path.join([base_dir, "thirdparty", "torchsparse"])
torchsparse_abs_path = Path.expand(torchsparse_path)

Pythonx.uv_init("""
[project]
name = "texgen-texture-inpainting"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "transformers==4.28.1",
  "diffusers==0.28.0",
  "omegaconf==2.3.0",
  "accelerate==0.24.1",
  "safetensors",
  "bitsandbytes",
  "wandb",
  "jaxtyping",
  "typeguard",
  "pillow",
  "matplotlib",
  "opencv-python",
  "imageio",
  "torchdyn",
  "lpips",
  "torch-geometric",
  "open3d",
  "openexr",
  "trimesh",
  "pytorch-lightning",
  "lightning",
  "einops",
  "huggingface-hub>=0.20.2",
  "pyyaml",
  "ninja",
  "scipy",
  "termcolor",
  "timm",
  "sharedarray",
  "tensorboard",
  "tensorboardx",
  "yapf",
  "addict",
  "plyfile",
  "h5py",
  "numpy<2.0",
  # nvdiffrast: required for TEXGen rendering
  # nvdiffrast wheel requires PyTorch 2.3+ (has ExchangeDevice symbol)
  # Updated to match nvdiffrast wheel requirements
  "torch==2.3.0",
  "torchvision==0.18.0",
  # torch-cluster: platform-specific wheel URLs for PyTorch 2.3.0+cu121
  "torch-cluster @ https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_cluster-1.6.3%2Bpt23cu121-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux'",
  "torch-cluster @ https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_cluster-1.6.3%2Bpt23cu121-cp310-cp310-win_amd64.whl ; sys_platform == 'win32'",
  # torch-sparse: platform-specific wheel URLs for PyTorch 2.3.0+cu121
  "torch-sparse @ https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux'",
  "torch-sparse @ https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp310-cp310-win_amd64.whl ; sys_platform == 'win32'",
  #"torchsparse @ https://github.com/Deathdadev/torchsparse/releases/download/v2.1.0-windows/torchsparse-2.1.0-cp310-cp310-win_amd64.whl", 
  # nvdiffrast: use pre-built wheel from Hugging Face (no build required)
  # Note: This wheel may require PyTorch 2.3+ due to ExchangeDevice symbol
  "nvdiffrast @ https://huggingface.co/spaces/microsoft/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux'",
  # torchsparse: build from local thirdparty/torchsparse directory
  # torchsparse now has torch in build-system.requires, so it can be built during uv_init
  "torchsparse @ file:///#{String.replace(torchsparse_abs_path, "\\", "/")}",
  # Note: TEXGen is used from local thirdparty/TEXGen directory
  # Additional dependencies that may need manual installation:
  # xformers: install from PyTorch index (commented out - may not be available for all platforms)
  "xformers",
  #"Pointcept @ https://github.com/Pointcept/Pointcept.git#subdirectory=libs/pointops.git",
  # "spconv-cu121",
  # - flash-attn (may require manual build)
]

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
""")

# Parse command-line arguments
defmodule ArgsParser do
  def show_help do
    IO.puts("""
    TEXGen Texture Inpainting Script
    Generate, inpaint, or complete textures for 3D meshes using TEXGen
    Repository: https://github.com/CVMI-Lab/TEXGen
    Hugging Face: https://huggingface.co/Andyx/TEXGen

    Usage:
      elixir texgen_texture_inpainting.exs <model_dir> <text_prompt> [options]
      elixir texgen_texture_inpainting.exs --model-dir <dir> --prompt <text> [options]

    Arguments:
      <model_dir>                 Directory containing model.obj, model.mtl, model.png
      <text_prompt>               Text prompt describing the desired texture

    Options:
      --model-dir, -m <path>      Directory containing model.obj, model.mtl, model.png
      --prompt, -p <text>         Text prompt describing the desired texture
      --num-steps, --steps <int>  Number of inference steps (default: 30)
      --guidance-scale, -g <float> Guidance scale (default: 2.0)
      --seed, -s <int>            Random seed (default: 42)
      --gpu <int>                GPU ID to use (default: 0)
      --output-dir <path>        Output directory (default: outputs)
      --help, -h                  Show this help message

    Example:
      elixir texgen_texture_inpainting.exs assets/models/3441609f539b46b38e7ab1213660cf3e "a wooden chair with a simple, modern design"
      elixir texgen_texture_inpainting.exs --model-dir assets/models/344 --prompt "a modern wooden chair" --num-steps 50
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        model_dir: :string,
        prompt: :string,
        num_steps: :integer,
        guidance_scale: :float,
        seed: :integer,
        gpu: :integer,
        output_dir: :string,
        help: :boolean
      ],
      aliases: [
        m: :model_dir,
        p: :prompt,
        steps: :num_steps,
        g: :guidance_scale,
        s: :seed,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    # Support both positional and named arguments
    model_dir = Keyword.get(opts, :model_dir) || List.first(args)
    prompt = Keyword.get(opts, :prompt) || Enum.at(args, 1)

    if !model_dir do
      IO.puts("""
      Error: Model directory is required.

      Usage:
        elixir texgen_texture_inpainting.exs <model_dir> <text_prompt> [options]
        elixir texgen_texture_inpainting.exs --model-dir <dir> --prompt <text> [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    if !prompt do
      IO.puts("""
      Error: Text prompt is required.

      Usage:
        elixir texgen_texture_inpainting.exs <model_dir> <text_prompt> [options]
        elixir texgen_texture_inpainting.exs --model-dir <dir> --prompt <text> [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    # Validate model directory
    model_dir = Path.expand(model_dir)
    if !File.exists?(model_dir) do
      IO.puts("Error: Model directory not found: #{model_dir}")
      System.halt(1)
    end

    model_obj = Path.join(model_dir, "model.obj")
    model_png = Path.join(model_dir, "model.png")

    if !File.exists?(model_obj) do
      IO.puts("Error: model.obj not found in: #{model_dir}")
      System.halt(1)
    end

    if !File.exists?(model_png) do
      IO.puts("Error: model.png not found in: #{model_dir}")
      System.halt(1)
    end

    num_steps = Keyword.get(opts, :num_steps, 30)
    if num_steps < 1 do
      IO.puts("Error: num_steps must be at least 1")
      System.halt(1)
    end

    guidance_scale = Keyword.get(opts, :guidance_scale, 2.0)
    if guidance_scale < 0.0 do
      IO.puts("Error: guidance_scale must be non-negative")
      System.halt(1)
    end

    config = %{
      model_dir: model_dir,
      prompt: prompt,
      num_steps: num_steps,
      guidance_scale: guidance_scale,
      seed: Keyword.get(opts, :seed, 42),
      gpu: Keyword.get(opts, :gpu, 0),
      output_dir: Keyword.get(opts, :output_dir, "outputs")
    }

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

# Check platform compatibility
if :os.type() == {:win32, :nt} do
  IO.puts("""
  ⚠️  WARNING: TEXGen has limited Windows support.

  TEXGen requires several Linux-specific dependencies:
  - Triton (no Windows wheels for PyTorch 2.1.0)
  - spconv-cu118 (Linux-only)
  - nvdiffrast (Linux-only)
  - torchsparse (Linux-only)
  - flash-attn (Linux-only)

  For best results, please use Linux or WSL (Windows Subsystem for Linux).

  Attempting to continue anyway...
  """)
end

IO.puts("""
=== TEXGen Texture Inpainting ===
Model Directory: #{config.model_dir}
Text Prompt: #{config.prompt}
Inference Steps: #{config.num_steps}
Guidance Scale: #{config.guidance_scale}
Seed: #{config.seed}
GPU: #{config.gpu}
Output Directory: #{config.output_dir}
""")

# Add paths to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  texgen_dir: Path.join([base_dir, "thirdparty", "TEXGen"]),
  checkpoint_dir: Path.join([base_dir, "pretrained_weights", "TEXGen"]),
  config_path: Path.join([base_dir, "thirdparty", "TEXGen", "configs", "texgen_test.yaml"])
})

# Save config to JSON for Python to read
{config_file, config_file_normalized} = ConfigFile.create(config_with_paths, "texgen_config")

# Download checkpoint using Elixir-native approach
SpanCollector.track_span("texgen.download_weights", fn ->
  IO.puts("\n=== Step 1: Download Pretrained Weights ===")
  IO.puts("Downloading TEXGen checkpoint from Hugging Face...")

  checkpoint_dir = config_with_paths.checkpoint_dir
  IO.puts("Using checkpoint directory: #{checkpoint_dir}")

  # Download TEXGen checkpoint (using OpenTelemetry integration)
  case HuggingFaceDownloader.download_repo("Andyx/TEXGen", checkpoint_dir, "TEXGen", true) do
    {:ok, _} -> :ok
    {:error, _} -> IO.puts("[WARN] TEXGen download had errors, but continuing...")
  end
end)

# Process using TEXGen
SpanCollector.track_span("texgen.texture_generation", fn ->
try do
  {_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
import subprocess
from pathlib import Path
import uuid
import torch

# Get configuration from JSON file
""" <> ConfigFile.python_path_string(config_file_normalized) <> ~S"""

model_dir = config.get('model_dir')
prompt = config.get('prompt')
num_steps = config.get('num_steps', 30)
guidance_scale = config.get('guidance_scale', 2.0)
seed = config.get('seed', 42)
gpu = config.get('gpu', 0)
output_dir = config.get('output_dir', 'outputs')
texgen_dir = config.get('texgen_dir')
checkpoint_dir = config.get('checkpoint_dir')
config_path = config.get('config_path')

# Resolve paths to absolute
model_dir = str(Path(model_dir).resolve())
texgen_dir = str(Path(texgen_dir).resolve())
checkpoint_dir = str(Path(checkpoint_dir).resolve())
config_path = str(Path(config_path).resolve())

# Verify TEXGen directory exists
if not Path(texgen_dir).exists():
    raise FileNotFoundError(
        f"TEXGen not found at {texgen_dir}. "
        "Please ensure TEXGen is cloned to thirdparty/TEXGen"
    )

# Verify config file exists
if not Path(config_path).exists():
    raise FileNotFoundError(
        f"Config file not found at {config_path}. "
        "Please ensure TEXGen configs are available"
    )

# Verify model directory structure
model_obj = Path(model_dir) / "model.obj"
model_png = Path(model_dir) / "model.png"
if not model_obj.exists():
    raise FileNotFoundError(f"model.obj not found in {model_dir}")
if not model_png.exists():
    raise FileNotFoundError(f"model.png not found in {model_dir}")

print("\n=== Step 2: Verify torchsparse ===")
# Check if torchsparse is available (should be installed via uv_init from local thirdparty/torchsparse)
try:
    import torchsparse
    print("[OK] torchsparse is available")
    print(f"TorchSparse version: {torchsparse.__version__}")
except ImportError as e:
    print(f"[WARN] torchsparse not found: {e}")
    print("Note: torchsparse should be built from thirdparty/torchsparse during uv_init")
    print("Continuing anyway - TEXGen will fail without torchsparse.")

print("\n=== Step 3: Verify nvdiffrast ===")
# Check if nvdiffrast is available (should be installed via uv_init)
try:
    import nvdiffrast.torch as dr
    print("[OK] nvdiffrast is available")
except (ImportError, ModuleNotFoundError) as e:
    print(f"\n[WARN] nvdiffrast not found: {e}")
    print("Note: nvdiffrast may require CUDA development tools (nvcc) to build.")
    print("If the build failed, you may need to install it manually.")
    print("Continuing anyway - some features may not work without nvdiffrast.")

print("\n=== Step 4: Prepare Input Data ===")

# Generate a unique model ID (use first 2 chars of directory name or generate UUID)
model_id = Path(model_dir).name
if len(model_id) < 2:
    model_id = str(uuid.uuid4()).replace("-", "")[:32]

# Create input JSONL file
input_list_dir = Path(texgen_dir) / "assets" / "input_list"
input_list_dir.mkdir(parents=True, exist_ok=True)
input_list_file = input_list_dir / f"input_{model_id}.jsonl"

# Create JSONL entry
jsonl_entry = {
    "id": model_id,
    "result": prompt,
    "root_dir": str(Path(model_dir).parent)
}

with open(input_list_file, 'w', encoding='utf-8') as f:
    f.write(json.dumps(jsonl_entry) + "\n")

print(f"[OK] Created input list: {input_list_file}")

# Find checkpoint file
checkpoint_files = list(Path(checkpoint_dir).glob("*.ckpt"))
if not checkpoint_files:
    # Try looking in subdirectories
    checkpoint_files = list(Path(checkpoint_dir).rglob("*.ckpt"))

if not checkpoint_files:
    raise FileNotFoundError(
        f"No checkpoint file found in {checkpoint_dir}. "
        "Please download the checkpoint from Hugging Face: https://huggingface.co/Andyx/TEXGen"
    )

checkpoint_path = str(checkpoint_files[0])
print(f"[OK] Using checkpoint: {checkpoint_path}")

print("\n=== Step 5: Run TEXGen Inference ===")
print(f"Running TEXGen with {num_steps} steps, guidance_scale={guidance_scale}, seed={seed}...")

# Change to TEXGen directory
original_cwd = Path.cwd()
os.chdir(texgen_dir)

try:
    # Force GPU usage by setting CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available. GPU {gpu} requested but not found.")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Using GPU {gpu}: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
    
    # Build launch.py command (use wrapper for compatibility)
    launch_script = "launch_wrapper.py" if os.path.exists(os.path.join(texgen_dir, "launch_wrapper.py")) else "launch.py"
    cmd = [
        sys.executable,
        launch_script,
        "--config", config_path,
        "--test",
        "--gpu", str(gpu),
        f"data.eval_scene_list={input_list_file}",
        f"exp_root_dir={output_dir}",
        f"name=texgen",
        f"tag=inference",
        f"system.weights={checkpoint_path}",
        f"system.test_num_steps={num_steps}",
        f"system.test_cfg_scale={guidance_scale}",
        f"seed={seed}"
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {texgen_dir}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Run TEXGen with environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    result = subprocess.run(
        cmd,
        cwd=texgen_dir,
        env=env,
        capture_output=False,
        text=True,
        check=True
    )

    print("[OK] TEXGen inference completed successfully")

    # Find output directory (TEXGen creates timestamped directories)
    output_base = Path(output_dir) / "texgen" / "inference"
    if output_base.exists():
        # Find the most recent timestamped directory
        timestamped_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()], reverse=True)
        if timestamped_dirs:
            output_path = timestamped_dirs[0]
            print(f"\n=== Complete ===")
            print(f"Generated textures saved to: {output_path}")
            print(f"\nOutput files:")
            for file in sorted(output_path.rglob("*")):
                if file.is_file():
                    rel_path = file.relative_to(output_path)
                    print(f"  - {rel_path}")
        else:
            print(f"\n[WARN] No timestamped output directory found in {output_base}")
    else:
        print(f"\n[WARN] Output directory not found: {output_base}")

finally:
    # Restore original working directory
    os.chdir(original_cwd)
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
IO.puts("Texture generation completed successfully!")

# Display OpenTelemetry trace
SpanCollector.display_trace()
