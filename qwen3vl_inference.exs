#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Qwen3-VL Vision-Language Inference Script
# Generate text responses from images using Huihui-Qwen3-VL-4B-Thinking-abliterated
# Model: Huihui-Qwen3-VL-4B-Thinking-abliterated (uncensored version)
# Repository: https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated
#
# Usage:
#   elixir qwen3vl_inference.exs <image_path> "<prompt>" [options]
#
# Options:
#   --max-tokens <int>              Maximum number of tokens to generate (default: 4096)
#   --temperature <float>            Sampling temperature (default: 0.7)
#   --top-p <float>                  Top-p (nucleus) sampling (default: 0.9)
#   --output <path>                  Output file path for text response (optional)
#   --use-flash-attention            Use Flash Attention 2 for better performance (default: false)

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
  {:req, "~> 0.5.0"}
])

# Suppress debug logs from Req to avoid showing long URLs
Logger.configure(level: :info)

# Initialize Python environment with required dependencies
# Qwen3-VL uses transformers and accelerate
# All dependencies managed by uv (no pip)
Pythonx.uv_init("""
[project]
name = "qwen3vl-inference"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "transformers",
  "accelerate",
  "pillow",
  "torch>=2.0.0,<2.5.0",  # Pin to stable version range for Windows compatibility
  "torchvision>=0.15.0,<0.20.0",  # Pin torchvision to compatible version
  "numpy",
  "huggingface-hub",
  "bitsandbytes",
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
    Qwen3-VL Vision-Language Inference Script
    Generate text responses from images using Huihui-Qwen3-VL-4B-Thinking-abliterated
    Model: Huihui-Qwen3-VL-4B-Thinking-abliterated (uncensored version)
    Repository: https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated

    Usage:
      elixir qwen3vl_inference.exs <image_path> "<prompt>" [options]

    Options:
      --max-tokens, -m <int>        Maximum number of tokens to generate (default: 4096)
      --temperature, -t <float>     Sampling temperature (default: 0.7)
      --top-p <float>                Top-p (nucleus) sampling (default: 0.9)
      --output, -o <path>            Output file path for text response (optional)
      --use-flash-attention          Use Flash Attention 2 for better performance (default: false)
      --use-4bit                     Use 4-bit quantization (default: true, recommended for 4B model, ~2-3GB VRAM)
      --full-precision                Use full precision instead of 4-bit quantization (requires 8GB+ VRAM)
      --help, -h                      Show this help message

    Example:
      elixir qwen3vl_inference.exs image.jpg "What is in this image?"
      elixir qwen3vl_inference.exs photo.png "Describe this scene" -m 2048 -t 0.8 -o output.txt
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        max_tokens: :integer,
        temperature: :float,
        top_p: :float,
        output: :string,
        use_flash_attention: :boolean,
        use_4bit: :boolean,
        full_precision: :boolean,
        help: :boolean
      ],
      aliases: [
        m: :max_tokens,
        t: :temperature,
        o: :output,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    # Check if no arguments provided at all
    if Enum.empty?(args) do
      IO.puts("""
      Error: Image path and prompt are required.

      Usage:
        elixir qwen3vl_inference.exs <image_path> "<prompt>" [options]

      Example:
        elixir qwen3vl_inference.exs image.jpg "What is in this image?"

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    image_path = List.first(args)
    prompt = args |> Enum.at(1)

    if !image_path do
      IO.puts("""
      Error: Image path is required.

      Usage:
        elixir qwen3vl_inference.exs <image_path> "<prompt>" [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    if !prompt do
      IO.puts("""
      Error: Text prompt is required.

      Usage:
        elixir qwen3vl_inference.exs <image_path> "<prompt>" [options]

      Example:
        elixir qwen3vl_inference.exs image.jpg "What is in this image?"

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    if !File.exists?(image_path) do
      IO.puts("Error: Image file not found: #{image_path}")
      System.halt(1)
    end

    max_tokens = Keyword.get(opts, :max_tokens, 4096)
    if max_tokens < 1 do
      IO.puts("Error: max_tokens must be at least 1")
      System.halt(1)
    end

    temperature = Keyword.get(opts, :temperature, 0.7)
    if temperature < 0.0 do
      IO.puts("Error: temperature must be non-negative")
      System.halt(1)
    end

    top_p = Keyword.get(opts, :top_p, 0.9)
    if top_p < 0.0 or top_p > 1.0 do
      IO.puts("Error: top_p must be between 0.0 and 1.0")
      System.halt(1)
    end

    config = %{
      image_path: image_path,
      prompt: prompt,
      max_tokens: max_tokens,
      temperature: temperature,
      top_p: top_p,
      output_path: Keyword.get(opts, :output),
      use_flash_attention: Keyword.get(opts, :use_flash_attention, false),
      use_4bit: (if Keyword.get(opts, :full_precision, false), do: false, else: Keyword.get(opts, :use_4bit, true))
    }

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== Qwen3-VL Vision-Language Inference ===
Image: #{config.image_path}
Prompt: #{config.prompt}
Max Tokens: #{config.max_tokens}
Temperature: #{config.temperature}
Top-P: #{config.top_p}
Use Flash Attention: #{config.use_flash_attention}
Use 4-bit Quantization: #{config.use_4bit}
#{if config.output_path, do: "Output File: #{config.output_path}", else: ""}
""")

# Add weights directory to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  model_weights_dir: Path.join([base_dir, "pretrained_weights", "Huihui-Qwen3-VL-4B-Thinking-abliterated"])
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

        # Filter out GGUF files (not needed for transformers library)
        # We only need safetensors, config files, tokenizer files, etc.
        filtered_files =
          files_list
          |> Enum.reject(fn {path, _info} ->
            String.ends_with?(path, ".gguf") or
            String.contains?(path, "/GGUF/") or
            String.contains?(path, "\\GGUF\\")
          end)

        total = length(filtered_files)
        skipped = length(files_list) - total
        IO.puts("Found #{length(files_list)} files (#{skipped} GGUF files skipped, #{total} files to download)")

        filtered_files
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
IO.puts("Downloading Qwen3-VL models from Hugging Face...")

base_dir = Path.expand(".")
model_weights_dir = Path.join([base_dir, "pretrained_weights", "Huihui-Qwen3-VL-4B-Thinking-abliterated"])

IO.puts("Using weights directory: #{model_weights_dir}")

# Qwen3-VL repository on Hugging Face
repo_id = "huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated"

# Download model weights
case HuggingFaceDownloader.download_repo(repo_id, model_weights_dir, "Qwen3-VL") do
  {:ok, _} -> :ok
  {:error, _} ->
    IO.puts("[WARN] Qwen3-VL download had errors, but continuing...")
    IO.puts("[INFO] Model will be loaded from Hugging Face Hub if local files are incomplete")
end

# Import libraries and process using Qwen3-VL
{_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
from pathlib import Path

# Verify PyTorch installation before importing transformers
print("\nVerifying PyTorch installation...")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    # Verify torch.serialization exists
    from torch import serialization
    print("[OK] PyTorch installation verified")
except ImportError as e:
    print(f"\n[ERROR] Failed to import PyTorch: {e}")
    print("This may be due to missing DLLs or an incomplete installation.")
    print("Please ensure Visual C++ Redistributables are installed.")
    raise
except Exception as e:
    print(f"\n[ERROR] PyTorch import error: {e}")
    print("This may be due to missing DLLs. Try reinstalling PyTorch or installing Visual C++ Redistributables.")
    raise

from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# Set CPU thread optimization
cpu_count = os.cpu_count()
print(f"Number of CPU cores in the system: {cpu_count}")
half_cpu_count = cpu_count // 2
os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
torch.set_num_threads(half_cpu_count)

# Get configuration from JSON file
config_file = Path("config.json")
if not config_file.exists():
    raise FileNotFoundError("config.json not found. This should be created by the Elixir script.")

with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Validate config is a dictionary
if not isinstance(config, dict):
    raise ValueError(f"config.json must contain a JSON object, got {type(config)}")

image_path = config.get('image_path')
prompt = config.get('prompt')
max_tokens = config.get('max_tokens', 4096)
temperature = config.get('temperature', 0.7)
top_p = config.get('top_p', 0.9)
output_path = config.get('output_path')
use_flash_attention = config.get('use_flash_attention', False)
use_4bit = config.get('use_4bit', True)

# Validate required fields
if not image_path:
    raise ValueError(f"image_path is required but was not found in config.json. Config keys: {list(config.keys())}")
if not prompt:
    raise ValueError(f"prompt is required but was not found in config.json. Config keys: {list(config.keys())}")

# Get weights directory from config
model_weights_dir = config.get('model_weights_dir')

# Fallback to default path if not in config
if not model_weights_dir:
    base_dir = Path.cwd()
    model_weights_dir = str(base_dir / "pretrained_weights" / "Huihui-Qwen3-VL-4B-Thinking-abliterated")

# Ensure path is string
model_weights_dir = str(Path(model_weights_dir).resolve())

# Resolve image path
image_path = str(Path(image_path).resolve())

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

print("\n=== Step 3: Initialize Model ===")
print("Loading Qwen3-VL model...")

MODEL_ID = "huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Configure quantization if requested
quantization_config = None
if use_4bit and device == "cuda":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    print("[INFO] Using 4-bit quantization (NF4)")
    dtype = None  # Don't set dtype when using quantization

try:
    # Load model
    # Try loading from local directory first, then from Hugging Face
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "flash_attention_2" if use_flash_attention else "sdpa",
    }

    if quantization_config:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["dtype"] = dtype

    if Path(model_weights_dir).exists() and (Path(model_weights_dir) / "config.json").exists():
        print(f"Loading from local directory: {model_weights_dir}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_weights_dir,
            **load_kwargs
        )
    else:
        print(f"Loading from Hugging Face Hub: {MODEL_ID}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            **load_kwargs
        )

    if use_4bit:
        print(f"[OK] Model loaded on {device} with 4-bit quantization")
    else:
        print(f"[OK] Model loaded on {device} with dtype {dtype}")
    if use_flash_attention:
        print("[OK] Flash Attention 2 enabled")

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("[OK] Processor loaded")

except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print("\nMake sure you have")
    print("  1. All dependencies installed via uv")
    print("  2. Sufficient GPU memory:")
    print("     - 8GB+ VRAM for 4B model at full precision")
    print("     - 2-3GB VRAM for 4B model with --use-4bit (recommended)")
    print("  3. Flash Attention 2 installed if using --use-flash-attention")
    raise

print("\n=== Step 4: Process Image and Generate Response ===")
print(f"Loading image: {image_path}")

# Load and verify image
if not Path(image_path).exists():
    raise FileNotFoundError(f"Image file not found: {image_path}")

image = Image.open(image_path).convert("RGB")
print(f"[OK] Image loaded: {image.size[0]}x{image.size[1]}")

# Prepare messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
]

print(f"Prompt: '{prompt}'")

# Preparation for inference
print("Preparing inputs...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

print("[OK] Inputs prepared")

# Inference: Generation of the output
print(f"Generating response (max_tokens={max_tokens}, temperature={temperature}, top_p={top_p})...")
generated_ids = model.generate(
    **inputs,
    max_new_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    do_sample=temperature > 0.0,
)

# Extract only the newly generated tokens
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# Decode the response
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

response = output_text[0] if output_text else ""

print("[OK] Response generated")
print("\n=== Response ===")
print(response)
print("\n=== End Response ===")

# Save output if specified
if output_path:
    output_path_resolved = Path(output_path).resolve()
    output_path_resolved.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_resolved, 'w', encoding='utf-8') as f:
        f.write(response)

    print(f"\n[OK] Response saved to: {output_path_resolved}")
else:
    # Save to default output directory with timestamp
    import time
    tag = time.strftime("%Y%m%d_%H_%M_%S")
    export_dir = output_dir / tag
    export_dir.mkdir(exist_ok=True)

    output_filename = f"qwen3vl_response_{tag}.txt"
    output_path_default = export_dir / output_filename

    with open(output_path_default, 'w', encoding='utf-8') as f:
        f.write(response)

    print(f"\n[OK] Response saved to: {output_path_default}")

print("\n=== Complete ===")
""", %{})

IO.puts("\n=== Complete ===")
IO.puts("Vision-language inference completed successfully!")
