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
#   --model "HY-Motion-1.0"         Model variant: HY-Motion-1.0, HY-Motion-1.0-Lite (default: "HY-Motion-1.0")
#   --duration <float>               Target duration in seconds (default: 5.0, will be estimated if prompt engineering enabled)
#   --cfg-scale <float>             Classifier-free guidance scale (default: 5.0)
#   --num-seeds <int>               Number of random seeds for generation (default: 4)
#   --disable-rewrite               Disable LLM-based prompt rewriting
#   --disable-duration-est          Disable LLM-based duration estimation
#   --output-format "dict"           Output format: dict (JSON), fbx (if FBX SDK available) (default: "dict")
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
  # fbxsdkpy is optional - only needed for FBX export
  # "fbxsdkpy==2020.1.post2",
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
      --model, -m <variant>           Model variant: HY-Motion-1.0, HY-Motion-1.0-Lite (default: "HY-Motion-1.0")
      --duration, -d <float>          Target duration in seconds (default: 5.0, will be estimated if prompt engineering enabled)
      --cfg-scale <float>            Classifier-free guidance scale (default: 5.0)
      --num-seeds <int>               Number of random seeds for generation (default: 4)
      --disable-rewrite               Disable LLM-based prompt rewriting
      --disable-duration-est          Disable LLM-based duration estimation
      --output-format, -f <format>   Output format: dict (JSON), fbx (if FBX SDK available) (default: "dict")
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
    model = Keyword.get(opts, :model, "HY-Motion-1.0")
    valid_models = ["HY-Motion-1.0", "HY-Motion-1.0-Lite"]
    if model not in valid_models do
      IO.puts("Error: Invalid model variant '#{model}'. Valid variants: #{Enum.join(valid_models, ", ")}")
      System.halt(1)
    end

    # Validate output format
    output_format = Keyword.get(opts, :output_format, "dict")
    valid_formats = ["dict", "fbx"]
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
config_with_paths = Map.merge(config, %{
  thirdparty_dir: Path.join([base_dir, "thirdparty", "HY-Motion-1.0"]),
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
      if File.exists?(config_path) do
        IO.puts("[OK] Model variant directory found: #{model_subdir}")
        if File.exists?(ckpt_path) do
          file_size_mb = File.stat!(ckpt_path).size / 1024 / 1024
          IO.puts("[OK] Model checkpoint found: #{Float.round(file_size_mb, 1)} MB")
        else
          IO.puts("[WARN] Model checkpoint not found: latest.ckpt")
          IO.puts("[INFO] The checkpoint may need to be downloaded separately (large file)")
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

# Ensure stdin is not blocking
if hasattr(os, 'devnull'):
    try:
        devnull = open(os.devnull, 'r')
        sys.stdin = devnull
    except:
        pass

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
else:
    print(f"[ERROR] Thirdparty directory not found: {thirdparty_dir}")
    sys.exit(1)

# Import HY-Motion modules
try:
    from hymotion.utils.t2m_runtime import T2MRuntime
    print("[OK] HY-Motion modules imported successfully")
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] Failed to import HY-Motion modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get configuration values
prompt = config.get('prompt')
is_file = config.get('is_file', False)
model_variant = config.get('model', 'HY-Motion-1.0')
duration = config.get('duration', 5.0)
cfg_scale = config.get('cfg_scale', 5.0)
num_seeds = config.get('num_seeds', 4)
disable_rewrite = config.get('disable_rewrite', False)
disable_duration_est = config.get('disable_duration_est', False)
output_format = config.get('output_format', 'dict')

print("\n=== Step 2: Setup Model Paths ===")
sys.stdout.flush()

# Setup checkpoint path
# The structure is: pretrained_weights/hymotion/tencent/HY-Motion-1.0/{model_variant}/
# where model_variant is "HY-Motion-1.0" or "HY-Motion-1.0-Lite"
repo_base = os.path.join(ckpts_dir, "HY-Motion-1.0")
model_path = os.path.join(repo_base, model_variant)
os.makedirs(model_path, exist_ok=True)
print(f"Model checkpoint path: {model_path}")
sys.stdout.flush()

# Check if config file exists
config_path = os.path.join(model_path, "config.yaml")
if not os.path.exists(config_path):
    print(f"[WARN] Config file not found: {config_path}")
    print(f"[INFO] Model weights should be downloaded from Hugging Face")
    print(f"[INFO] Expected structure: {model_path}/config.yaml and {model_path}/latest.ckpt")
    # Try alternative path structure (if downloaded directly to ckpts_dir)
    alt_path = os.path.join(ckpts_dir, model_variant)
    alt_config = os.path.join(alt_path, "config.yaml")
    if os.path.exists(alt_config):
        print(f"[INFO] Found config at alternative path: {alt_config}")
        config_path = alt_config
        model_path = alt_path
    sys.stdout.flush()

print("\n=== Step 3: Initialize T2M Runtime ===")
sys.stdout.flush()

device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
print(f"Device: {device}")
sys.stdout.flush()

try:
    # Initialize runtime
    # Note: If model not found, it will use randomly initialized weights (for testing)
    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name="latest.ckpt",
        skip_text=False,
        device_ids=None,  # Use all available GPUs
        skip_model_loading=False,
        force_cpu=(device == "cpu"),
        disable_prompt_engineering=(disable_rewrite and disable_duration_est),
        prompt_engineering_host=None,  # Can be set if prompt engineering service available
        prompt_engineering_model_path=None  # Can be set if local prompt engineering model available
    )
    
    print(f"[OK] T2M Runtime initialized")
    print(f"  FBX available: {runtime.fbx_available}")
    sys.stdout.flush()
    
except Exception as e:
    print(f"[ERROR] Failed to initialize runtime: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 4: Generate Motion ===")
sys.stdout.flush()

# Create output directory with timestamp
tag = time.strftime("%Y%m%d_%H_%M_%S")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True)

try:
    if is_file:
        # Process file input
        print(f"Processing input file: {prompt}")
        sys.stdout.flush()
        
        # Use the file processing logic from local_infer.py
        # For now, we'll process it as a single prompt file
        # Full file processing would require more complex logic
        with open(prompt, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        
        if not lines:
            print("[ERROR] Input file is empty")
            sys.exit(1)
        
        # Process first line as prompt (can be extended to process all lines)
        first_line = lines[0]
        split_list = first_line.split("#")
        text_prompt = split_list[0].strip()
        if len(split_list) > 1:
            try:
                duration = float(split_list[1]) / 30.0  # Convert frames to seconds
            except:
                duration = config.get('duration', 5.0)
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
    
    # Generate motion
    req_format = output_format if output_format == "fbx" and runtime.fbx_available else "dict"
    output_filename = "00000000"  # Default filename
    
    html, fbx_files, motion_dict = runtime.generate_motion(
        text=text_prompt,
        seeds_csv=seeds_csv,
        duration=duration,
        cfg_scale=cfg_scale,
        output_format=req_format,
        original_text=text_prompt,
        output_dir=str(export_dir),
        output_filename=output_filename,
    )
    
    print(f"[OK] Motion generated successfully")
    sys.stdout.flush()
    
    # Save results
    if req_format == "fbx" and fbx_files:
        print(f"[OK] Saved {len(fbx_files)} FBX file(s)")
        for fbx_file in fbx_files:
            print(f"  - {fbx_file}")
        sys.stdout.flush()
    elif motion_dict:
        # Save motion dict as JSON
        motion_json_path = export_dir / f"motion_{tag}.json"
        with open(motion_json_path, 'w', encoding='utf-8') as f:
            json.dump(motion_dict, f, indent=2)
        print(f"[OK] Saved motion data to {motion_json_path}")
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
        f.write(f"Output Format: {req_format}\n")
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
if req_format == "fbx" and fbx_files:
    for fbx_file in fbx_files:
        print(f"  - {fbx_file}")
if motion_dict:
    print(f"  - motion_{tag}.json")
if html:
    print(f"  - visualization_{tag}.html")
print(f"  - metadata_{tag}.txt")
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

