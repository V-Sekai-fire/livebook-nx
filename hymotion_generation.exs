#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# HY-Motion 1.0 Text-to-Motion Generation Script
# Generate 3D human motion animations from text prompts using Tencent HY-Motion 1.0
# Model: HY-Motion 1.0 by Tencent (1.0B parameters, DiT architecture with flow matching)
# Repository: https://github.com/Tencent-Hunyuan/HY-Motion-1.0
# Hugging Face: https://huggingface.co/tencent/HY-Motion-1.0
#
# Usage:
#   elixir hymotion_generation.exs "<prompt>" [options]
#   elixir hymotion_generation.exs --input-file <file> [options]
#
# Options:
#   --input-file, -i <path>         Read prompts from file (alternative to command-line text)
#   --model "HY-Motion-1.0-Lite"   Model variant: HY-Motion-1.0, HY-Motion-1.0-Lite (default: "HY-Motion-1.0-Lite")
#   --duration <float>               Target duration in seconds (default: 5.0, will be estimated if prompt engineering enabled)
#   --cfg-scale <float>             Classifier-free guidance scale (default: 5.0)
#   --num-seeds <int>               Number of random seeds for generation (default: 4)
#   --disable-rewrite               Disable LLM-based prompt rewriting
#   --disable-duration-est          Disable LLM-based duration estimation
#   --output-format "glb"            Output format: glb (GLB 3D model with animation) (default: "glb")
#   --help, -h                       Show this help message

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
Pythonx.uv_init("""
[project]
name = "hymotion-generation"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "torch==2.5.1",
  "torchvision==0.20.1",
  "torchdiffeq==0.2.5",
  "accelerate==0.30.1",
  "diffusers==0.26.3",
  "transformers==4.53.3",
  "einops==0.8.1",
  "safetensors==0.5.3",
  "bitsandbytes==0.49.0",
  "numpy>=1.24.0,<2.0",
  "scipy>=1.10.0",
  "transforms3d==0.4.2",
  "PyYAML==6.0",
  "omegaconf==2.3.0",
  "click==8.1.3",
  "requests==2.32.4",
  "openai==1.78.1",
  "huggingface-hub==0.30.0",
  # Blender Python API for GLB export
  # Note: bpy requires Python 3.11+, but we use 3.10 for compatibility
  # If GLB export is needed, install bpy separately or use Python 3.11
  # "bpy==4.5.*",
]

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[[tool.uv.index]]
name = "inria-gitlab"
url = "https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple"
explicit = true
""")

# Parse command-line arguments
defmodule ArgsParser do
  def show_help do
    IO.puts("""
    HY-Motion 1.0 Text-to-Motion Generation Script
    Generate 3D human motion animations from text prompts using Tencent HY-Motion 1.0
    
    Model: HY-Motion 1.0 by Tencent (1.0B parameters, DiT architecture with flow matching)
    Repository: https://github.com/Tencent-Hunyuan/HY-Motion-1.0
    Hugging Face: https://huggingface.co/tencent/HY-Motion-1.0

    Usage:
      elixir hymotion_generation.exs "<prompt>" [options]
      elixir hymotion_generation.exs --input-file <file> [options]

    Options:
      --input-file, -i <path>         Read prompts from file (alternative to command-line text)
      --model, -m <variant>           Model variant: HY-Motion-1.0, HY-Motion-1.0-Lite (default: "HY-Motion-1.0-Lite")
      --duration, -d <float>          Target duration in seconds (default: 5.0, will be estimated if prompt engineering enabled)
      --cfg-scale <float>            Classifier-free guidance scale (default: 5.0)
      --num-seeds <int>               Number of random seeds for generation (default: 4)
      --disable-rewrite               Disable LLM-based prompt rewriting
      --disable-duration-est          Disable LLM-based duration estimation
      --output-format, -f <format>   Output format: glb (GLB 3D model with animation) (default: "glb")
      --help, -h                      Show this help message

    Model Variants:
      - HY-Motion-1.0: Standard model, 1.0B parameters, 26GB VRAM minimum
      - HY-Motion-1.0-Lite: Lightweight model, 0.46B parameters, 24GB VRAM minimum

    Prompt Guidelines:
      - Use English prompts (under 60 words for optimal results)
      - Focus on action descriptions or detailed movements of limbs and torso
      - Not supported: Non-humanoid characters, emotions/clothing, environment/camera, multi-person interactions

    Examples:
      elixir hymotion_generation.exs "A person performs a squat, then pushes a barbell overhead"
      elixir hymotion_generation.exs "A person walks unsteadily, then slowly sits down" --duration 8.0
      elixir hymotion_generation.exs --input-file prompts.txt --model HY-Motion-1.0-Lite --num-seeds 1
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        input_file: :string,
        model: :string,
        duration: :float,
        cfg_scale: :float,
        num_seeds: :integer,
        disable_rewrite: :boolean,
        disable_duration_est: :boolean,
        output_format: :string,
        help: :boolean
      ],
      aliases: [
        i: :input_file,
        m: :model,
        d: :duration,
        f: :output_format,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    # Get prompt from various sources
    prompt = cond do
      input_file = Keyword.get(opts, :input_file) ->
        if File.exists?(input_file) do
          # Read from file - will be processed by Python script
          input_file
        else
          IO.puts("Error: Input file not found: #{input_file}")
          System.halt(1)
        end
      arg_prompt = List.first(args) ->
        arg_prompt
      true ->
        IO.puts("""
        Error: Prompt input is required.

        Usage:
          elixir hymotion_generation.exs "<prompt>" [options]
          elixir hymotion_generation.exs --input-file <file> [options]

        Use --help or -h for more information.
        """)
        System.halt(1)
    end

    # Validate model variant
    model = Keyword.get(opts, :model, "HY-Motion-1.0-Lite")
    valid_models = ["HY-Motion-1.0", "HY-Motion-1.0-Lite"]
    if model not in valid_models do
      IO.puts("Error: Invalid model variant '#{model}'. Valid variants: #{Enum.join(valid_models, ", ")}")
      System.halt(1)
    end

    # Validate output format
    output_format = Keyword.get(opts, :output_format, "glb")
    valid_formats = ["glb"]
    if output_format not in valid_formats do
      IO.puts("Error: Invalid output format. Must be one of: #{Enum.join(valid_formats, ", ")}")
      System.halt(1)
    end

    # Validate duration
    duration = Keyword.get(opts, :duration, 5.0)
    if duration < 1.0 or duration > 30.0 do
      IO.puts("Error: Duration must be between 1.0 and 30.0 seconds")
      System.halt(1)
    end

    # Validate cfg_scale
    cfg_scale = Keyword.get(opts, :cfg_scale, 5.0)
    if cfg_scale < 0.1 or cfg_scale > 20.0 do
      IO.puts("Error: cfg_scale must be between 0.1 and 20.0")
      System.halt(1)
    end

    # Validate num_seeds
    num_seeds = Keyword.get(opts, :num_seeds, 4)
    if num_seeds < 1 or num_seeds > 10 do
      IO.puts("Error: num_seeds must be between 1 and 10")
      System.halt(1)
    end

    config = %{
      prompt: prompt,
      is_file: is_binary(prompt) && String.contains?(prompt, ["/", "\\"]),
      model: model,
      duration: duration,
      cfg_scale: cfg_scale,
      num_seeds: num_seeds,
      disable_rewrite: Keyword.get(opts, :disable_rewrite, false),
      disable_duration_est: Keyword.get(opts, :disable_duration_est, false),
      output_format: output_format
    }

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== HY-Motion 1.0 Text-to-Motion Generation ===
Prompt: #{if config.is_file, do: "File: #{config.prompt}", else: String.slice(config.prompt, 0, 100) <> if(String.length(config.prompt) > 100, do: "...", else: "")}
Model: #{config.model}
Duration: #{config.duration} seconds
CFG Scale: #{config.cfg_scale}
Num Seeds: #{config.num_seeds}
Disable Rewrite: #{config.disable_rewrite}
Disable Duration Est: #{config.disable_duration_est}
Output Format: #{config.output_format}
""")

# Add paths to config for Python
base_dir = Path.expand(".")
thirdparty_dir = Path.join([base_dir, "thirdparty", "HY-Motion-1.0"])

# Check HY-Motion code (vendored in thirdparty directory)
IO.puts("\n=== Step 0: Check HY-Motion Code ===")
hymotion_code_exists = File.exists?(thirdparty_dir) && File.exists?(Path.join(thirdparty_dir, "hymotion"))

if !hymotion_code_exists do
  IO.puts("[ERROR] HY-Motion code not found at: #{thirdparty_dir}")
  IO.puts("[INFO] The code should be vendored in the thirdparty/HY-Motion-1.0 directory")
  IO.puts("[INFO] Repository: https://github.com/Tencent-Hunyuan/HY-Motion-1.0")
  System.halt(1)
else
  IO.puts("[OK] HY-Motion code found at: #{thirdparty_dir}")
end

config_with_paths = Map.merge(config, %{
  thirdparty_dir: thirdparty_dir,
  models_dir: Path.join([base_dir, "pretrained_weights", "hymotion"]),
  ckpts_dir: Path.join([base_dir, "pretrained_weights", "hymotion", "tencent"])
})

# Download models using Elixir-native approach
IO.puts("\n=== Step 1: Download Pretrained Weights ===")
IO.puts("Downloading HY-Motion 1.0 models from Hugging Face...")

base_dir = Path.expand(".")
hymotion_weights_dir = Path.join([base_dir, "pretrained_weights", "hymotion"])

IO.puts("Using weights directory: #{hymotion_weights_dir}")

# HY-Motion 1.0 repository on Hugging Face
# The models are in subdirectories: HY-Motion-1.0 and HY-Motion-1.0-Lite
repo_id = "tencent/HY-Motion-1.0"

# Determine which model variant to download
model_subdir = case config.model do
  "HY-Motion-1.0-Lite" -> "HY-Motion-1.0-Lite"
  _ -> "HY-Motion-1.0"
end

# Download the entire repo - it contains both model variants in subdirectories
download_dir = Path.join([hymotion_weights_dir, "tencent", "HY-Motion-1.0"])
model_weights_dir = Path.join([download_dir, model_subdir])

IO.puts("Downloading repository: #{repo_id}")
IO.puts("Target directory: #{download_dir}")
IO.puts("Model variant: #{model_subdir}")

# Download HY-Motion weights (using OpenTelemetry integration)
# The Hugging Face repo contains both model variants in subdirectories
case HuggingFaceDownloader.download_repo(repo_id, download_dir, "HY-Motion-1.0", true) do
  {:ok, _} -> 
    IO.puts("[OK] HY-Motion 1.0 download completed")
    # Verify the model subdirectory exists
    if File.exists?(model_weights_dir) do
      config_path = Path.join([model_weights_dir, "config.yaml"])
      ckpt_path = Path.join([model_weights_dir, "latest.ckpt"])
      stats_dir = Path.join([model_weights_dir, "stats"])
      if File.exists?(config_path) do
        IO.puts("[OK] Model variant directory found: #{model_subdir}")
        if File.exists?(ckpt_path) do
          file_size_mb = File.stat!(ckpt_path).size / 1024 / 1024
          IO.puts("[OK] Model checkpoint found: #{Float.round(file_size_mb, 1)} MB")
        else
          IO.puts("[WARN] Model checkpoint not found: latest.ckpt")
          IO.puts("[INFO] The checkpoint may need to be downloaded separately (large file)")
        end
        # Check for stats directory
        if File.exists?(stats_dir) do
          mean_file = Path.join([stats_dir, "Mean.npy"])
          std_file = Path.join([stats_dir, "Std.npy"])
          if File.exists?(mean_file) and File.exists?(std_file) do
            IO.puts("[OK] Stats files found in: #{stats_dir}")
          else
            IO.puts("[WARN] Stats directory exists but Mean.npy or Std.npy missing")
            IO.puts("[INFO] Motion generation may have normalization issues without these files")
          end
        else
          IO.puts("[WARN] Stats directory not found: #{stats_dir}")
          IO.puts("[INFO] Motion generation will work but may have normalization issues")
          IO.puts("[INFO] Stats files (Mean.npy, Std.npy) should be in the model directory")
        end
      else
        IO.puts("[WARN] Config file not found in: #{model_weights_dir}")
        IO.puts("[INFO] The model structure may be different than expected")
      end
    else
      IO.puts("[WARN] Model variant directory not found: #{model_subdir}")
      IO.puts("[INFO] Checking downloaded structure...")
      # List what was downloaded
      if File.exists?(download_dir) do
        case File.ls(download_dir) do
          {:ok, files} ->
            IO.puts("[INFO] Downloaded files/directories: #{Enum.join(files, ", ")}")
          _ -> :ok
        end
      end
      IO.puts("[INFO] Expected structure: #{model_weights_dir}/config.yaml and #{model_weights_dir}/latest.ckpt")
    end
  {:error, _} ->
    IO.puts("[WARN] HY-Motion 1.0 download had errors, but continuing...")
    IO.puts("[INFO] If the model is not on Hugging Face, you may need to download it manually")
    IO.puts("[INFO] Expected structure: #{model_weights_dir}/config.yaml and #{model_weights_dir}/latest.ckpt")
end

# Save config to JSON for Python to read
{config_file, config_file_normalized} = ConfigFile.create(config_with_paths, "hymotion_config")

# Import libraries and process using HY-Motion
SpanCollector.track_span("hymotion.generation", fn ->
  try do
    {_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
import re
import time
import warnings
import subprocess
import random
from pathlib import Path

# CRITICAL: Patch subprocess.run FIRST, before any imports that might use it
_original_subprocess_run = subprocess.run
def _patched_subprocess_run(*args, **kwargs):
    # Handle capture_output - if it's True, don't set stdout/stderr manually
    if kwargs.get('capture_output', False):
        # capture_output=True means stdout=subprocess.PIPE and stderr=subprocess.PIPE
        # Don't override these if capture_output is set
        if 'stdin' not in kwargs:
            kwargs['stdin'] = subprocess.DEVNULL
    else:
        # Only set stdout/stderr if capture_output is not set
        if 'stdin' not in kwargs:
            kwargs['stdin'] = subprocess.DEVNULL
        if 'stdout' not in kwargs:
            kwargs['stdout'] = subprocess.PIPE
        if 'stderr' not in kwargs:
            kwargs['stderr'] = subprocess.PIPE
    return _original_subprocess_run(*args, **kwargs)
subprocess.run = _patched_subprocess_run

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Use Hugging Face models directly instead of local ckpts
os.environ['USE_HF_MODELS'] = '1'

# Ensure stdin is not blocking
if hasattr(os, 'devnull'):
    try:
        devnull = open(os.devnull, 'r')
        sys.stdin = devnull
    except:
        pass

# Patch sys.stdout and sys.stderr to prevent blocking
# But make sure they don't raise StopIteration
_original_stdout_write = sys.stdout.write
_original_stderr_write = sys.stderr.write

def _safe_write(original_write):
    def _write_wrapper(data):
        try:
            return original_write(data)
        except (StopIteration, BrokenPipeError, OSError):
            # Silently ignore write errors
            return len(data) if isinstance(data, str) else len(str(data))
    return _write_wrapper

sys.stdout.write = _safe_write(_original_stdout_write)
sys.stderr.write = _safe_write(_original_stderr_write)

# Get configuration from JSON file
""" <> ConfigFile.python_path_string(config_file_normalized) <> ~S"""

# Get paths from config
thirdparty_dir = config.get('thirdparty_dir')
models_dir = config.get('models_dir')
ckpts_dir = config.get('ckpts_dir')

# Add thirdparty directory to Python path
if thirdparty_dir and os.path.exists(thirdparty_dir):
    sys.path.insert(0, thirdparty_dir)
    print(f"[OK] Added thirdparty directory to path: {thirdparty_dir}")
    
    # Set working directory to thirdparty for relative paths (like scripts/gradio/...)
    # Save original working directory
    original_cwd = os.getcwd()
    os.chdir(thirdparty_dir)
    print(f"[OK] Changed working directory to: {thirdparty_dir}")
else:
    print(f"[ERROR] Thirdparty directory not found: {thirdparty_dir}")
    sys.exit(1)

# Import HY-Motion modules
try:
    from hymotion.utils.t2m_runtime import T2MRuntime
    from hymotion.utils.visualize_mesh_web import get_output_dir as original_get_output_dir
    from hymotion.utils.visualize_mesh_web import sanitize_folder_name as original_sanitize_folder_name
    print("[OK] HY-Motion modules imported successfully")
    
    # Patch get_output_dir to handle absolute paths correctly
    # This allows us to use absolute paths from project root instead of thirdparty directory
    def patched_get_output_dir(sub_path: str = ""):
        # If sub_path is an absolute path, use it directly
        if os.path.isabs(sub_path):
            return sub_path
        # Otherwise, use the original function
        return original_get_output_dir(sub_path)
    
    # Patch sanitize_folder_name to preserve absolute paths
    def patched_sanitize_folder_name(folder_name: str) -> str:
        # If it's an absolute path, return it as-is (normalized)
        if os.path.isabs(folder_name):
            return os.path.normpath(folder_name)
        # Otherwise, use the original function
        return original_sanitize_folder_name(folder_name)
    
    # Apply the patches
    import hymotion.utils.visualize_mesh_web as visualize_module
    visualize_module.get_output_dir = patched_get_output_dir
    visualize_module.sanitize_folder_name = patched_sanitize_folder_name
    print("[OK] Patched get_output_dir and sanitize_folder_name to handle absolute paths")
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] Failed to import HY-Motion modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get configuration values
prompt = config.get('prompt')
is_file = config.get('is_file', False)
model_variant = config.get('model', 'HY-Motion-1.0-Lite')
duration = config.get('duration', 5.0)
cfg_scale = config.get('cfg_scale', 5.0)
num_seeds = config.get('num_seeds', 4)
disable_rewrite = config.get('disable_rewrite', False)
disable_duration_est = config.get('disable_duration_est', False)
output_format = config.get('output_format', 'dict')

print("\n=== Step 2: Setup Model Paths ===")
sys.stdout.flush()

# Setup checkpoint path
# The Hugging Face repo structure is: HY-Motion-1.0/{model_variant}/config.yml
# where model_variant is "HY-Motion-1.0" or "HY-Motion-1.0-Lite"
repo_base = os.path.join(ckpts_dir, "HY-Motion-1.0")
model_path = os.path.join(repo_base, model_variant)
os.makedirs(model_path, exist_ok=True)
print(f"Model checkpoint path: {model_path}")
sys.stdout.flush()

# Also check if files are directly in repo_base (flat structure)
flat_model_path = os.path.join(repo_base, model_variant)
if not os.path.exists(flat_model_path):
    # Check if model files are in root of repo_base
    root_config = os.path.join(repo_base, "config.yml")
    if os.path.exists(root_config):
        print(f"[INFO] Found config in repo root, checking structure...")
        sys.stdout.flush()

# Check if config file exists - try multiple path structures and file names
# Note: Hugging Face repo uses config.yml (not config.yaml)
config_path = None
found_config = False

# Try both .yaml and .yml extensions
alt_paths = [
    os.path.join(model_path, "config.yaml"),  # Primary path with .yaml
    os.path.join(model_path, "config.yml"),   # Primary path with .yml
    os.path.join(ckpts_dir, model_variant, "config.yaml"),  # Direct variant path
    os.path.join(ckpts_dir, model_variant, "config.yml"),
    os.path.join(repo_base, model_variant, "config.yaml"),  # Repo base + variant
    os.path.join(repo_base, model_variant, "config.yml"),
    os.path.join(ckpts_dir, "HY-Motion-1.0", model_variant, "config.yaml"),  # Full path
    os.path.join(ckpts_dir, "HY-Motion-1.0", model_variant, "config.yml"),
]

for alt_config in alt_paths:
    if os.path.exists(alt_config):
        print(f"[OK] Found config file: {alt_config}")
        config_path = alt_config
        model_path = os.path.dirname(alt_config)
        found_config = True
        break

if not found_config:
    print(f"[WARN] Config file not found")
    print(f"[INFO] Searched in: {model_path}")
    print(f"[INFO] Model weights should be downloaded from Hugging Face")
    print(f"[INFO] Expected structure: {model_path}/config.yml and {model_path}/latest.ckpt")
    print(f"[INFO] The model will use randomly initialized weights (for testing only)")
    print(f"[INFO] To use pretrained weights, download from: https://huggingface.co/tencent/HY-Motion-1.0")
    # Set a default config path even if not found (will use random weights)
    config_path = os.path.join(model_path, "config.yml")
    sys.stdout.flush()

print("\n=== Step 3: Check Stats Files ===")
sys.stdout.flush()

# Check for stats directory - CRITICAL for motion normalization
# Without these files, motion will be erratic/unusable
stats_dir = os.path.join(model_path, "stats")
stats_mean = os.path.join(stats_dir, "Mean.npy")
stats_std = os.path.join(stats_dir, "Std.npy")

stats_missing = False
if not os.path.exists(stats_dir):
    print(f"[ERROR] Stats directory not found: {stats_dir}")
    stats_missing = True
elif not os.path.exists(stats_mean):
    print(f"[ERROR] Stats file missing: {stats_mean}")
    stats_missing = True
elif not os.path.exists(stats_std):
    print(f"[ERROR] Stats file missing: {stats_std}")
    stats_missing = True

if stats_missing:
    print(f"[ERROR] Stats files are REQUIRED for proper motion generation")
    print(f"[ERROR] Without Mean.npy and Std.npy, motion will be erratic/unusable")
    print(f"[INFO] Expected location: {stats_dir}/")
    print(f"[INFO] These files should be in the model directory from Hugging Face")
    print(f"[INFO] Model repository: https://huggingface.co/tencent/HY-Motion-1.0")
    print(f"[INFO] Please ensure stats/Mean.npy and stats/Std.npy are downloaded")
    sys.exit(1)
else:
    print(f"[OK] Stats files found: {stats_dir}")
    print(f"  - Mean.npy: {os.path.exists(stats_mean)}")
    print(f"  - Std.npy: {os.path.exists(stats_std)}")
sys.stdout.flush()

print("\n=== Step 4: Initialize T2M Runtime ===")
sys.stdout.flush()

device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
print(f"Device: {device}")
sys.stdout.flush()

try:
    # Check if checkpoint exists - use absolute path
    ckpt_path = os.path.join(model_path, "latest.ckpt")
    ckpt_path = os.path.abspath(ckpt_path)  # Ensure absolute path
    skip_model_loading = not os.path.exists(ckpt_path)
    
    # If config file doesn't exist, we can't initialize (even with random weights)
    if not found_config:
        print(f"[ERROR] Config file is required but not found")
        print(f"[INFO] Please ensure the model is downloaded from Hugging Face")
        print(f"[INFO] Expected: {model_path}/config.yml or {model_path}/config.yaml")
        sys.exit(1)
    
    if skip_model_loading:
        print(f"[WARN] Checkpoint not found: {ckpt_path}")
        print(f"[INFO] Will use randomly initialized weights (for testing only)")
        print(f"[INFO] To use pretrained weights, ensure the model is downloaded from Hugging Face")
        print(f"[INFO] Checkpoint should be at: {ckpt_path}")
        sys.stdout.flush()
    else:
        # Verify checkpoint file size
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"[OK] Checkpoint found: {ckpt_path}")
        print(f"[OK] Checkpoint size: {ckpt_size_mb:.1f} MB")
        sys.stdout.flush()
    
    # Patch the config to use absolute path for mean_std_dir
    # This ensures stats files are found even when working directory changes
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Update mean_std_dir to absolute path if it's relative
    if 'train_pipeline_args' in config_dict and 'test_cfg' in config_dict['train_pipeline_args']:
        mean_std_dir = config_dict['train_pipeline_args']['test_cfg'].get('mean_std_dir', './stats/')
        if mean_std_dir.startswith('./') or not os.path.isabs(mean_std_dir):
            # Convert relative path to absolute path relative to model directory
            abs_stats_dir = os.path.join(model_path, mean_std_dir.lstrip('./'))
            config_dict['train_pipeline_args']['test_cfg']['mean_std_dir'] = abs_stats_dir
            print(f"[INFO] Updated mean_std_dir to absolute path: {abs_stats_dir}")
            
            # Write updated config to a temporary file
            import tempfile
            temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
            yaml.dump(config_dict, temp_config, default_flow_style=False)
            temp_config.close()
            config_path = temp_config.name
            print(f"[INFO] Using patched config: {config_path}")
    sys.stdout.flush()
    
    # Initialize runtime with absolute checkpoint path
    # load_in_demo() checks os.path.exists(ckpt_name) so it needs the full absolute path
    # The checkpoint should be in the same directory as the config file
    ckpt_name_for_runtime = ckpt_path if not skip_model_loading else "latest.ckpt"
    
    print(f"[INFO] Initializing T2MRuntime with:")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {ckpt_name_for_runtime}")
    print(f"  Skip model loading: {skip_model_loading}")
    sys.stdout.flush()
    
    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_name_for_runtime,  # Use absolute path if checkpoint exists
        skip_text=False,
        device_ids=None,  # Use all available GPUs
        skip_model_loading=skip_model_loading,
        force_cpu=(device == "cpu"),
        disable_prompt_engineering=(disable_rewrite and disable_duration_est),
        prompt_engineering_host=None,  # Can be set if prompt engineering service available
        prompt_engineering_model_path=None  # Can be set if local prompt engineering model available
    )
    
    print(f"[OK] T2M Runtime initialized")
    print(f"  Model loading: {'skipped (random weights)' if skip_model_loading else 'loaded from checkpoint'}")
    
    # Verify and fix stats dimensions if needed
    if runtime.pipelines:
        pipeline = runtime.pipelines[0]
        if hasattr(pipeline, 'mean') and hasattr(pipeline, 'std'):
            mean_shape = pipeline.mean.shape
            std_shape = pipeline.std.shape
            print(f"  Stats loaded: mean shape={mean_shape}, std shape={std_shape}")
            
            # Fix dimensions if stats are 1D instead of 2D
            if len(mean_shape) == 1 and mean_shape[0] == 201:
                print(f"  [INFO] Reshaping stats from 1D to 2D (1, 201)")
                pipeline.mean = pipeline.mean.unsqueeze(0)
                pipeline.std = pipeline.std.unsqueeze(0)
                print(f"  [OK] Stats reshaped: mean shape={pipeline.mean.shape}, std shape={pipeline.std.shape}")
            elif mean_shape == (1, 201) and std_shape == (1, 201):
                print(f"  [OK] Stats dimensions correct (1, 201)")
            else:
                print(f"  [WARN] Stats dimensions unexpected: {mean_shape}, {std_shape}")
            
            mean_range = f"[{pipeline.mean.min().item():.4f}, {pipeline.mean.max().item():.4f}]"
            std_range = f"[{pipeline.std.min().item():.4f}, {pipeline.std.max().item():.4f}]"
            print(f"  Stats ranges: mean={mean_range}, std={std_range}")
        else:
            print(f"  [WARN] Stats not found in pipeline (using blank mean/std)")
    sys.stdout.flush()
    
except Exception as e:
    print(f"[ERROR] Failed to initialize runtime: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("  1. Ensure model weights are downloaded from Hugging Face")
    print("  2. Check that config.yaml exists in the model directory")
    print("  3. Verify CUDA/GPU availability if using GPU mode")
    print("  4. Check Python dependencies are installed correctly")
    sys.exit(1)

print("\n=== Step 5: Generate Motion ===")
sys.stdout.flush()

# Create output directory with timestamp
# Use original_cwd (project root) instead of thirdparty directory
tag = time.strftime("%Y%m%d_%H_%M_%S")
output_dir = Path(original_cwd) / "output"
output_dir.mkdir(exist_ok=True, parents=True)
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True, parents=True)

try:
    if is_file:
        # Process file input
        print(f"Processing input file: {prompt}")
        sys.stdout.flush()
        
        # Use the file processing logic from local_infer.py
        # Supports both .txt and .json formats
        file_ext = os.path.splitext(prompt)[1].lower()
        
        if file_ext == '.json':
            # JSON format: {"category": ["prompt#duration#id", ...]}
            with open(prompt, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract first prompt from first category
            first_category = list(data.keys())[0] if data else None
            if first_category and data[first_category]:
                first_line = data[first_category][0]
            else:
                print("[ERROR] JSON file is empty or has no valid prompts")
                sys.exit(1)
        else:
            # TXT format: one prompt per line (prompt#duration#id)
            with open(prompt, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            
            if not lines:
                print("[ERROR] Input file is empty")
                sys.exit(1)
            
            first_line = lines[0]
        
        # Parse prompt line: format is "prompt#duration#id" or just "prompt"
        split_list = first_line.split("#")
        text_prompt = split_list[0].strip()
        
        if not text_prompt:
            print("[ERROR] No prompt text found in file")
            sys.exit(1)
        
        # Parse duration if provided (in frames, convert to seconds)
        if len(split_list) > 1 and split_list[1].strip():
            try:
                duration_frames = float(split_list[1].strip())
                duration = duration_frames / 30.0  # Convert frames to seconds (assuming 30 FPS)
                print(f"[INFO] Using duration from file: {duration_frames} frames ({duration:.2f}s)")
            except (ValueError, IndexError):
                duration = config.get('duration', 5.0)
                print(f"[INFO] Could not parse duration from file, using default: {duration}s")
        else:
            duration = config.get('duration', 5.0)
        
        print(f"Text prompt: {text_prompt}")
        print(f"Duration: {duration}s")
        sys.stdout.flush()
    else:
        # Single prompt
        text_prompt = prompt
        duration = config.get('duration', 5.0)
        print(f"Text prompt: {text_prompt}")
        print(f"Duration: {duration}s")
        sys.stdout.flush()
    
    # Generate random seeds
    seeds = [random.randint(0, 999) for _ in range(num_seeds)]
    seeds_csv = ",".join(map(str, seeds))
    
    print(f"Seeds: {seeds_csv}")
    print(f"CFG Scale: {cfg_scale}")
    sys.stdout.flush()
    
    # Generate motion (always use dict format, we'll convert to GLB with Blender)
    req_format = "dict"  # Always generate as dict, convert to GLB with Blender
    if output_format == "glb":
        print(f"[INFO] GLB format requested - will convert using Blender after generation")
        sys.stdout.flush()
    
    output_filename = "00000000"  # Default filename
    
    # Use absolute path - our patch to get_output_dir will handle it correctly
    export_dir_abs = os.path.abspath(str(export_dir))
    print(f"Using output directory: {export_dir_abs}")
    print(f"Generating motion with format: {req_format}")
    sys.stdout.flush()
    
    try:
        html, _, motion_dict = runtime.generate_motion(
            text=text_prompt,
            seeds_csv=seeds_csv,
            duration=duration,
            cfg_scale=cfg_scale,
            output_format=req_format,
            original_text=text_prompt,
            output_dir=export_dir_abs,  # Use absolute path - patched get_output_dir will handle it
            output_filename=output_filename,
        )
        
        print(f"[OK] Motion generated successfully")
        sys.stdout.flush()
    except Exception as gen_error:
        print(f"[ERROR] Motion generation failed: {gen_error}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check that the model weights are properly downloaded")
        print("  2. Verify GPU memory is sufficient (24GB+ for Lite, 26GB+ for standard)")
        print("  3. Try reducing --num-seeds or --duration")
        print("  4. Use --disable-rewrite and --disable-duration-est if prompt engineering fails")
        sys.exit(1)
    
    # Note: HTML visualization is already handled by runtime.generate_motion() 
    # which uses upstream save_visualization_data() and construct_smpl_data_dict()
    # No custom code needed - the upstream code handles everything correctly
    
    # Convert to GLB using Blender if requested
    glb_files = []
    if output_format == "glb" and motion_dict:
        print(f"\n=== Converting to GLB using Blender ===")
        sys.stdout.flush()
        try:
            # Build mesh and armature in Blender for GLB export
            from hymotion.pipeline.body_model import construct_smpl_data_dict
            import numpy as np
            import torch
            
            # Clear existing scene
            bpy.ops.wm.read_factory_settings(use_empty=True)
            
            # Load WoodenMesh template data
            wooden_mesh_path = os.path.join(thirdparty_dir, "scripts", "gradio", "static", "assets", "dump_wooden")
            
            # Load template data using the same function as body_model.py
            from pathlib import Path as PathLib
            
            def load_wooden_mesh_data(model_path):
                # Load wooden model data from binary files
                model_path = PathLib(model_path)
                
                # Load vertex template: (V*3,) -> (V, 3)
                with open(model_path / "v_template.bin", "rb") as f:
                    v_template_flat = np.frombuffer(f.read(), dtype=np.float32)
                num_verts = len(v_template_flat) // 3
                v_template = v_template_flat.reshape(num_verts, 3)
                
                # Load joint template: (J*3,) -> (J, 3)
                with open(model_path / "j_template.bin", "rb") as f:
                    j_template_flat = np.frombuffer(f.read(), dtype=np.float32)
                num_joints = len(j_template_flat) // 3
                j_template = j_template_flat.reshape(num_joints, 3)
                
                # Load skin weights: (V*4,) -> (V, 4)
                with open(model_path / "skinWeights.bin", "rb") as f:
                    skin_weights_flat = np.frombuffer(f.read(), dtype=np.float32)
                skin_weights = skin_weights_flat.reshape(num_verts, 4)
                
                # Load skin indices: (V*4,) -> (V, 4)
                with open(model_path / "skinIndice.bin", "rb") as f:
                    skin_indices_flat = np.frombuffer(f.read(), dtype=np.uint16)
                skin_indices = skin_indices_flat.reshape(num_verts, 4).astype(np.int64)
                
                # Load kintree (parent indices): (J,)
                with open(model_path / "kintree.bin", "rb") as f:
                    parents = np.frombuffer(f.read(), dtype=np.int32)
                
                # Load faces
                with open(model_path / "faces.bin", "rb") as f:
                    faces_flat = np.frombuffer(f.read(), dtype=np.uint16)
                faces = faces_flat.reshape(-1, 3)
                
                # Load joint names (already loaded above, but ensure consistency)
                joint_names_path = model_path / "joint_names.json"
                if joint_names_path.exists():
                    with open(joint_names_path, "r") as f:
                        joint_names = json.load(f)
                else:
                    joint_names = [f"Joint_{i}" for i in range(num_joints)]
                
                return {
                    "v_template": v_template,
                    "j_template": j_template,
                    "skin_weights": skin_weights,
                    "skin_indices": skin_indices,
                    "parents": parents,
                    "faces": faces,
                    "joint_names": joint_names,
                    "num_joints": num_joints,
                    "num_verts": num_verts,
                }
            
            # Load template data
            print(f"[INFO] Loading WoodenMesh template from: {wooden_mesh_path}")
            template_data = load_wooden_mesh_data(wooden_mesh_path)
            v_template = template_data["v_template"]
            j_template = template_data["j_template"]
            faces = template_data["faces"]
            # Use joint_names from template_data (should match what we loaded above)
            joint_names = template_data["joint_names"]
            parents = template_data["parents"]
            skin_weights = template_data["skin_weights"]
            skin_indices = template_data["skin_indices"]
            num_joints = template_data["num_joints"]
            
            print(f"[INFO] Loaded template: {len(v_template)} vertices, {len(faces)} faces, {num_joints} joints")
            
            # Create mesh from template vertices and faces
            mesh = bpy.data.meshes.new(name="WoodenMesh")
            mesh.from_pydata(v_template.tolist(), [], faces.tolist())
            mesh.update()
            
            # Create mesh object
            mesh_obj = bpy.data.objects.new("WoodenMesh", mesh)
            bpy.context.collection.objects.link(mesh_obj)
            
            # Get motion data
            # motion_dict has: 'latent_denorm', 'keypoints3d', 'rot6d', 'transl', 'root_rotations_mat', 'text'
            # The 'keypoints3d' contains joint positions for each frame
            # The 'vertices' from the pipeline output would be in the model_output, but we can reconstruct from keypoints
            
            # Import geometry utilities for rotation conversion
            from hymotion.utils.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_euler_angles
            
            # Parse motion data from motion_dict
            # These should be torch tensors or numpy arrays
            def parse_tensor(tensor_data):
                # Convert tensor data to numpy array
                if tensor_data is None:
                    return None
                if hasattr(tensor_data, 'cpu'):
                    # Torch tensor
                    return tensor_data.cpu().numpy()
                elif isinstance(tensor_data, np.ndarray):
                    return tensor_data
                elif isinstance(tensor_data, (list, tuple)):
                    return np.array(tensor_data)
                else:
                    # Try to convert to numpy
                    return np.array(tensor_data)
            
            # Extract motion data
            rot6d_data = parse_tensor(motion_dict.get('rot6d'))
            transl_data = parse_tensor(motion_dict.get('transl'))
            root_rotations_mat = parse_tensor(motion_dict.get('root_rotations_mat'))
            
            if rot6d_data is None:
                print("[ERROR] rot6d data not found in motion_dict")
                print(f"[INFO] Available keys in motion_dict: {list(motion_dict.keys())}")
                raise ValueError("rot6d data is required for animation")
            
            # Convert to torch tensors for processing
            import torch
            rot6d_tensor = torch.from_numpy(rot6d_data).float()
            
            # Handle transl data
            if transl_data is None:
                print("[WARN] transl data not found, using zeros")
                # Infer shape from rot6d
                if len(rot6d_tensor.shape) == 4:
                    num_frames = rot6d_tensor.shape[1]
                else:
                    num_frames = rot6d_tensor.shape[0]
                transl_data = np.zeros((num_frames, 3))
            
            transl_tensor = torch.from_numpy(transl_data).float()
            
            # Convert rot6d to rotation matrices: (num_frames, num_joints, 3, 3)
            # rot6d shape: (batch, frames, joints, 6) or (frames, joints, 6)
            # Remove batch dimension if present
            if len(rot6d_tensor.shape) == 4:
                # (batch, frames, joints, 6) -> take first batch
                rot6d_tensor = rot6d_tensor[0]
                if len(transl_tensor.shape) == 2 and transl_tensor.shape[0] == rot6d_tensor.shape[0]:
                    # transl is (frames, 3) - already correct
                    pass
                elif len(transl_tensor.shape) == 3:
                    # (batch, frames, 3) -> take first batch
                    transl_tensor = transl_tensor[0]
            
            if len(rot6d_tensor.shape) != 3 or rot6d_tensor.shape[2] != 6:
                raise ValueError(f"Invalid rot6d shape: {rot6d_tensor.shape}, expected (frames, joints, 6)")
            
            num_frames, num_joints_motion, _ = rot6d_tensor.shape
            
            # Check if number of joints matches template
            if num_joints_motion != num_joints:
                print(f"[WARN] Joint count mismatch: motion has {num_joints_motion} joints, template has {num_joints} joints")
                print(f"[INFO] Using minimum: {min(num_joints_motion, num_joints)} joints")
                num_joints_to_use = min(num_joints_motion, num_joints)
            else:
                num_joints_to_use = num_joints
            
            print(f"[INFO] Motion data shapes: rot6d={rot6d_tensor.shape}, transl={transl_tensor.shape}")
            print(f"[INFO] Using {num_joints_to_use} joints for animation")
            sys.stdout.flush()
            
            # Convert rot6d to rotation matrices (3x3) directly
            # Use only the joints that match the template
            rot6d_to_use = rot6d_tensor[:, :num_joints_to_use, :]
            
            # Reshape to (num_frames * num_joints_to_use, 6) for batch conversion
            rot6d_flat = rot6d_to_use.reshape(-1, 6)
            rot_matrices_flat = rot6d_to_rotation_matrix(rot6d_flat)  # (num_frames * num_joints_to_use, 3, 3)
            rot_matrices = rot_matrices_flat.reshape(num_frames, num_joints_to_use, 3, 3).numpy()
            
            # Keep rotation matrices as 3x3 - we'll use them directly in Blender
            # This avoids Euler angle discontinuities and gimbal lock issues
            print(f"[INFO] Parsed motion data: {num_frames} frames, {num_joints_to_use} joints")
            print(f"[INFO] Translation range: {transl_data.min(axis=0)} to {transl_data.max(axis=0)}")
            print(f"[INFO] Using 3x3 rotation matrices directly from 6D representation")
            sys.stdout.flush()
            
            # Create armature with full skeleton
            bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
            armature = bpy.context.active_object
            armature.name = "Motion_Armature"
            armature.data.name = "Motion_Armature"
            
            # Create all bones from joint structure
            edit_bones = armature.data.edit_bones
            
            # Clear default bone
            if len(edit_bones) > 0:
                edit_bones.remove(edit_bones[0])
            
            # Create bones for all joints
            bones = {}
            for i, joint_name in enumerate(joint_names):
                bone = edit_bones.new(joint_name)
                bone.head = Vector(j_template[i])
                
                # Set tail based on child joints or default direction
                parent_idx = parents[i]
                if parent_idx >= 0 and parent_idx < num_joints:
                    # Point tail towards parent (will be adjusted)
                    bone.tail = Vector(j_template[i]) + Vector((0, 0, 0.05))
                else:
                    # Root bone - point upward
                    bone.tail = Vector(j_template[i]) + Vector((0, 0, 0.1))
                
                # Set parent relationship
                if parent_idx >= 0 and parent_idx < num_joints:
                    parent_name = joint_names[parent_idx]
                    if parent_name in bones:
                        bone.parent = bones[parent_name]
                        # Adjust tail to point to parent
                        bone.tail = Vector(j_template[parent_idx])
                
                bones[joint_name] = bone
            
            # Adjust bone lengths to connect to children
            for i, joint_name in enumerate(joint_names):
                bone = bones[joint_name]
                # Find children
                children = [j for j in range(num_joints) if parents[j] == i]
                if children:
                    # Point tail to first child
                    child_pos = j_template[children[0]]
                    bone.tail = Vector(child_pos)
                elif bone.parent:
                    # If no children, keep current tail
                    pass
            
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Apply skinning to mesh
            # Add armature modifier
            armature_modifier = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
            armature_modifier.object = armature
            armature_modifier.use_vertex_groups = True
            
            # Create vertex groups for each bone
            for joint_name in joint_names:
                vg = mesh_obj.vertex_groups.new(name=joint_name)
            
            # Assign skinning weights
            for vert_idx in range(len(v_template)):
                # Get skin weights and indices for this vertex
                weights = skin_weights[vert_idx]
                indices = skin_indices[vert_idx]
                
                # Assign weights to vertex groups
                for w, bone_idx in zip(weights, indices):
                    if w > 0.0 and bone_idx < num_joints:
                        bone_name = joint_names[bone_idx]
                        vg = mesh_obj.vertex_groups[bone_name]
                        vg.add([vert_idx], float(w), 'REPLACE')
            
            # Parent mesh to armature
            mesh_obj.parent = armature
            mesh_obj.parent_type = 'ARMATURE'
            
            # Animate the armature with motion data
            print(f"\n=== Animating Armature ===")
            sys.stdout.flush()
            
            # Set up animation
            scene = bpy.context.scene
            fps = 30  # HY-Motion uses 30 FPS
            scene.frame_start = 1
            scene.frame_end = num_frames
            scene.render.fps = fps
            
            # Select armature and enter pose mode
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            
            # Get pose bones
            pose_bones = armature.pose.bones
            
            # Find root bone (usually first joint or joint with parent_idx == -1)
            root_bone_idx = None
            for i, joint_name in enumerate(joint_names):
                if parents[i] < 0:
                    root_bone_idx = i
                    break
            if root_bone_idx is None:
                root_bone_idx = 0  # Default to first joint
            
            root_bone_name = joint_names[root_bone_idx]
            print(f"[INFO] Root bone: {root_bone_name} (index {root_bone_idx})")
            sys.stdout.flush()
            
            # Animate each bone
            for joint_idx in range(num_joints_to_use):
                if joint_idx >= len(joint_names):
                    print(f"[WARN] Joint index {joint_idx} exceeds joint_names length, skipping")
                    continue
                
                joint_name = joint_names[joint_idx]
                if joint_name not in pose_bones:
                    print(f"[WARN] Bone '{joint_name}' not found in pose bones, skipping")
                    continue
                
                bone = pose_bones[joint_name]
                
                # Use rotation matrix mode (more stable than Euler)
                # Convert 3x3 rotation matrix to Blender's Matrix and apply
                bone.rotation_mode = 'QUATERNION'  # Use quaternion for stability
                
                # Animate rotations for this bone using 3x3 rotation matrices
                for frame_idx in range(num_frames):
                    scene.frame_set(frame_idx + 1)
                    
                    # Get 3x3 rotation matrix for this joint at this frame
                    rot_mat_3x3 = rot_matrices[frame_idx, joint_idx]  # (3, 3)
                    
                    # Convert to Blender Matrix (row-major)
                    # Blender uses row-major matrices
                    rot_matrix = Matrix(rot_mat_3x3.tolist())
                    
                    # Convert matrix to quaternion (more stable than Euler)
                    bone.rotation_quaternion = rot_matrix.to_quaternion()
                    
                    # Insert keyframe for rotation
                    bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)
                    
                    # Animate root translation (only for root bone)
                    if joint_idx == root_bone_idx:
                        # Get translation for this frame
                        if len(transl_data.shape) == 2:
                            trans = transl_data[frame_idx]
                        else:
                            # Handle different shapes
                            trans = transl_data[frame_idx] if frame_idx < len(transl_data) else transl_data[0]
                        # Convert from meters to Blender units (1 unit = 1 meter, but scale if needed)
                        # HY-Motion uses meters, Blender uses meters by default
                        bone.location = Vector(trans)
                        bone.keyframe_insert(data_path="location", frame=frame_idx + 1)
            
            # Return to object mode
            bpy.ops.object.mode_set(mode='OBJECT')
            
            print(f"[OK] Animation applied: {num_frames} frames at {fps} FPS")
            print(f"[INFO] Root bone '{root_bone_name}' animated with translation")
            sys.stdout.flush()
            
            # Export to GLB
            glb_path = os.path.join(str(export_dir), f"{output_filename}_{tag}.glb")
            bpy.ops.export_scene.gltf(
                filepath=glb_path,
                export_format='GLB',
                export_materials=True,
                export_animations=True,
                export_armatures=True,
                export_skins=True,
                export_normals=True,
                export_colors=True,
                export_texcoords=True,
                use_selection=False
            )
            
            if os.path.exists(glb_path):
                glb_files.append(glb_path)
                print(f"[OK] Converted to GLB: {glb_path}")
                print(f"[INFO] Mesh: {len(v_template)} vertices, {len(faces)} faces")
                print(f"[INFO] Armature: {num_joints} joints with skinning")
                print(f"[INFO] Animation: {num_frames} frames at {fps} FPS")
            else:
                print(f"[WARN] GLB file not created: {glb_path}")
            sys.stdout.flush()
        except Exception as glb_error:
            print(f"[WARN] GLB conversion failed: {glb_error}")
            import traceback
            traceback.print_exc()
            # Ensure glb_files is still a list even on error
            if 'glb_files' not in locals():
                glb_files = []
            sys.stdout.flush()
    
    # Save results
    saved_files = []
    
    if output_format == "glb":
        # Find GLB files that were created
        glb_pattern = os.path.join(str(export_dir), "*.glb")
        import glob
        glb_files = glob.glob(glb_pattern)
        for glb_file in glb_files:
            saved_files.append(glb_file)
    
    if not saved_files:
        print("[WARN] No output files were saved")
        print("[INFO] Check that generation completed successfully")
        sys.stdout.flush()
    
    # Save HTML visualization if available
    if html:
        html_path = export_dir / f"visualization_{tag}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"[OK] Saved visualization to {html_path}")
        sys.stdout.flush()
    
    # Save metadata
    metadata_path = export_dir / f"metadata_{tag}.txt"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {text_prompt}\n")
        f.write(f"Model: {model_variant}\n")
        f.write(f"Duration: {duration}s\n")
        f.write(f"CFG Scale: {cfg_scale}\n")
        f.write(f"Num Seeds: {num_seeds}\n")
        f.write(f"Seeds: {seeds_csv}\n")
        f.write(f"Disable Rewrite: {disable_rewrite}\n")
        f.write(f"Disable Duration Est: {disable_duration_est}\n")
        f.write(f"Output Format: {output_format}\n")
    print(f"[OK] Saved metadata to {metadata_path}")
    sys.stdout.flush()
    
except Exception as e:
    print(f"[ERROR] Failed to generate motion: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Complete ===")
print(f"Generated motion saved to: {export_dir}")
print(f"\nOutput files:")
for saved_file in saved_files:
    print(f"  - {saved_file}")
if html:
    html_path = export_dir / f"visualization_{tag}.html"
    if os.path.exists(str(html_path)):
        print(f"  - {html_path}")
    else:
        print(f"  - [INFO] HTML visualization was generated but file not found")
metadata_path = export_dir / f"metadata_{tag}.txt"
if os.path.exists(str(metadata_path)):
    print(f"  - {metadata_path}")
sys.stdout.flush()
sys.stderr.flush()
""", %{"config_file_normalized" => config_file_normalized})
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
IO.puts("Motion generation completed successfully!")

# Display OpenTelemetry trace - save to output directory
SpanCollector.display_trace("output")

