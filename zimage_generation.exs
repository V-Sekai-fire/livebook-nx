#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Z-Image-Turbo Generation Script
# Generate photorealistic images from text prompts using Z-Image-Turbo
# Model: Z-Image-Turbo by Tongyi-MAI (6B parameters)
# Repository: https://replicate.com/prunaai/z-image-turbo
#
# Usage:
#   elixir zimage_generation.exs "<prompt>" [options]
#
# Options:
#   --width <int>                   Image width in pixels (default: 1024)
#   --height <int>                   Image height in pixels (default: 1024)
#   --seed <int>                     Random seed for generation (default: 0)
#   --num-steps <int>                Number of inference steps (default: 9, results in 8 DiT forwards)
#   --guidance-scale <float>         Guidance scale (default: 0.0 for turbo models)
#   --output-format "png"            Output format: png, jpg, jpeg (default: "png")
#
# Note: Image-to-image editing is not supported by Z-Image-Turbo.
#       Z-Image-Edit (a separate model) is required for image editing but is not yet released.

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
  {:req, "~> 0.5.0"}
])

# Suppress debug logs from Req to avoid showing long URLs
Logger.configure(level: :info)

# Initialize Python environment with required dependencies
# Z-Image-Turbo uses diffusers and transformers
# All dependencies managed by uv (no pip)
Pythonx.uv_init("""
[project]
name = "zimage-generation"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "diffusers @ git+https://github.com/huggingface/diffusers",
  "transformers",
  "accelerate",
  "pillow",
  "torch",
  "torchvision",
  "numpy",
  "huggingface-hub",
  "gitpython",
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
    Z-Image-Turbo Generation Script
    Generate photorealistic images from text prompts using Z-Image-Turbo
    Model: Z-Image-Turbo by Tongyi-MAI (6B parameters)
    Repository: https://replicate.com/prunaai/z-image-turbo

    Usage:
      elixir zimage_generation.exs "<prompt>" [options]

    Options:
      --width, -w <int>               Image width in pixels (default: 1024)
      --height <int>                  Image height in pixels (default: 1024)
      --seed, -s <int>                 Random seed for generation (default: 0)
      --num-steps, --steps <int>      Number of inference steps (default: 9, results in 8 DiT forwards)
      --guidance-scale, -g <float>     Guidance scale (default: 0.0 for turbo models)
      --output-format, -f "png"        Output format: png, jpg, jpeg (default: "png")
      --help, -h                       Show this help message

    Note: Image-to-image editing is not supported by Z-Image-Turbo.
          Z-Image-Edit (a separate model) is required for image editing but is not yet released.

    Example:
      elixir zimage_generation.exs "a beautiful sunset over mountains" --width 1024 --height 1024
      elixir zimage_generation.exs "a cat wearing a hat" -w 512 -h 512 -s 42
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        width: :integer,
        height: :integer,
        seed: :integer,
        num_steps: :integer,
        guidance_scale: :float,
        output_format: :string,
        help: :boolean
      ],
      aliases: [
        w: :width,
        h: :help,
        s: :seed,
        steps: :num_steps,
        g: :guidance_scale,
        f: :output_format
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    prompt = List.first(args)

    if !prompt do
      IO.puts("""
      Error: Text prompt is required.

      Usage:
        elixir zimage_generation.exs "<prompt>" [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    width = Keyword.get(opts, :width, 1024)
    height = Keyword.get(opts, :height, 1024)

    if width < 64 or width > 2048 or height < 64 or height > 2048 do
      IO.puts("Error: Width and height must be between 64 and 2048 pixels")
      System.halt(1)
    end

    output_format = Keyword.get(opts, :output_format, "png")
    valid_formats = ["png", "jpg", "jpeg"]
    if output_format not in valid_formats do
      IO.puts("Error: Invalid output format. Must be one of: #{Enum.join(valid_formats, ", ")}")
      System.halt(1)
    end

    num_steps = Keyword.get(opts, :num_steps, 9)
    if num_steps < 1 do
      IO.puts("Error: num_steps must be at least 1")
      System.halt(1)
    end

    guidance_scale = Keyword.get(opts, :guidance_scale, 0.0)
    if guidance_scale < 0.0 do
      IO.puts("Error: guidance_scale must be non-negative")
      System.halt(1)
    end

    config = %{
      prompt: prompt,
      width: width,
      height: height,
      seed: Keyword.get(opts, :seed, 0),
      num_steps: num_steps,
      guidance_scale: guidance_scale,
      output_format: output_format
    }

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== Z-Image-Turbo Generation ===
Prompt: #{config.prompt}
Width: #{config.width}
Height: #{config.height}
Seed: #{config.seed}
Inference Steps: #{config.num_steps}
Guidance Scale: #{config.guidance_scale}
Output Format: #{config.output_format}
""")

# Add weights directory to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  zimage_weights_dir: Path.join([base_dir, "pretrained_weights", "Z-Image-Turbo"])
})

# Save config to JSON for Python to read
config_json = Jason.encode!(config_with_paths)
File.write!("config.json", config_json)

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
IO.puts("Downloading Z-Image-Turbo models from Hugging Face...")

base_dir = Path.expand(".")
zimage_weights_dir = Path.join([base_dir, "pretrained_weights", "Z-Image-Turbo"])

IO.puts("Using weights directory: #{zimage_weights_dir}")

# Z-Image-Turbo repository on Hugging Face
repo_id = "Tongyi-MAI/Z-Image-Turbo"

# Download Z-Image-Turbo weights
case HuggingFaceDownloader.download_repo(repo_id, zimage_weights_dir, "Z-Image-Turbo") do
  {:ok, _} -> :ok
  {:error, _} ->
    IO.puts("[WARN] Z-Image-Turbo download had errors, but continuing...")
    IO.puts("[INFO] If the model is not on Hugging Face, you may need to download it manually")
end

# Import libraries and process using Z-Image-Turbo
{_, _python_globals} = Pythonx.eval("""
import json
import sys
import os
from pathlib import Path
from PIL import Image
import torch
from accelerate.utils import set_seed
from diffusers import ZImagePipeline

# Get configuration from JSON file
with open("config.json", 'r', encoding='utf-8') as f:
    config = json.load(f)

prompt = config.get('prompt')
width = config.get('width', 1024)
height = config.get('height', 1024)
seed = config.get('seed', 0)
num_steps = config.get('num_steps', 9)
guidance_scale = config.get('guidance_scale', 0.0)
output_format = config.get('output_format', 'png')

# Get weights directory from config
zimage_weights_dir = config.get('zimage_weights_dir')

# Fallback to default path if not in config
if not zimage_weights_dir:
    base_dir = Path.cwd()
    zimage_weights_dir = str(base_dir / "pretrained_weights" / "Z-Image-Turbo")

# Ensure path is string
zimage_weights_dir = str(Path(zimage_weights_dir).resolve())

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

print("\\n=== Step 3: Initialize Model ===")
print("Loading Z-Image-Turbo pipeline...")

device = "cuda" if torch.cuda.is_available() else "cpu"
# Use bfloat16 for optimal performance on supported GPUs, fallback to float16 or float32
if device == "cuda" and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
elif device == "cuda":
    dtype = torch.float16
else:
    dtype = torch.float32

try:
    # Load Z-Image-Turbo pipeline
    model_id = "Tongyi-MAI/Z-Image-Turbo"

    # Try loading from local directory first, then from Hugging Face
    if Path(zimage_weights_dir).exists() and (Path(zimage_weights_dir) / "model_index.json").exists():
        print(f"Loading from local directory: {zimage_weights_dir}")
        pipe = ZImagePipeline.from_pretrained(
            zimage_weights_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
    else:
        print(f"Loading from Hugging Face Hub: {model_id}")
        pipe = ZImagePipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

    pipe = pipe.to(device)
    print(f"[OK] Z-Image-Turbo pipeline loaded on {device} with dtype {dtype}")

    # [Optional] Attention Backend
    # Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
    try:
        # Try Flash Attention 3 first (newest)
        pipe.transformer.set_attention_backend("_flash_3")
        print("[OK] Enabled Flash-Attention-3")
    except:
        try:
            # Fallback to Flash Attention 2
            pipe.transformer.set_attention_backend("flash")
            print("[OK] Enabled Flash-Attention-2")
        except:
            print("[INFO] Using default SDPA attention backend")

    # [Optional] Model Compilation
    # Compiling the DiT model accelerates inference, but the first run will take longer to compile.
    # Uncomment the following line to enable compilation:
    # pipe.transformer.compile()
    # print("[OK] Model compilation enabled (first run will be slower)")

    # [Optional] CPU Offloading
    # Enable CPU offloading for memory-constrained devices.
    # Uncomment the following line to enable CPU offloading:
    # pipe.enable_model_cpu_offload()
    # print("[OK] CPU offloading enabled")

except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print("\\nMake sure you have")
    print("  1. All dependencies installed via uv (including diffusers from git)")
    print("  2. Sufficient GPU memory (16GB VRAM recommended)")
    print("  3. The latest version of diffusers installed from source")
    raise

print("\\n=== Step 4: Generate Image ===")
print(f"Generating image with prompt: '{prompt}'")
print(f"Resolution: {width}x{height}, Steps: {num_steps}, Guidance: {guidance_scale}")

# Generate random seed if seed is 0
if seed == 0:
    import secrets
    seed = secrets.randbelow(999999999)
    print(f"Generated random seed: {seed}")

set_seed(seed)

try:
    # Generate image
    generator = torch.Generator(device=device).manual_seed(seed)

    # Run inference (text-to-image only - Z-Image-Turbo doesn't support image editing)
    output = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    # Get the generated image
    image = output.images[0]

    print("[OK] Image generated successfully")

except Exception as e:
    print(f"[ERROR] Error during generation: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\\n=== Step 5: Save Image ===")

# Create output directory with timestamp
import time
tag = time.strftime("%Y%m%d_%H_%M_%S")
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True)

# Save image
output_filename = f"zimage_{tag}.{output_format}"
output_path = export_dir / output_filename

# Convert format if needed
if output_format.lower() in ["jpg", "jpeg"]:
    # Convert RGBA to RGB for JPEG
    if image.mode == "RGBA":
        # Create white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3] if image.mode == "RGBA" else None)
        image = background
    image.save(str(output_path), "JPEG", quality=95)
else:
    image.save(str(output_path), "PNG")

print(f"[OK] Saved image to {output_path}")

print("\\n=== Complete ===")
print(f"Generated image saved to: {output_path}")
print(f"\\nOutput file")
print(f"  - {output_path}")
""", %{})

IO.puts("\n=== Complete ===")
IO.puts("Image generation completed successfully!")
