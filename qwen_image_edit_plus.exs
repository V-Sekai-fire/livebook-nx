#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Qwen Image Edit Plus Script
# Edit images with precise control using Qwen's 20B parameter image editing model
# Model: qwen/qwen-image-edit-plus (via Replicate API)
# Repository: https://replicate.com/qwen/qwen-image-edit-plus
# Paper: https://arxiv.org/abs/2508.02324
#
# Usage:
#   elixir qwen_image_edit_plus.exs <image_path> "<prompt>" [options]
#   elixir qwen_image_edit_plus.exs --image <path1> --image <path2> "<prompt>" [options]
#
# Options:
#   --image, -i <path>            Input image path(s) (required, can specify multiple times for multi-image editing)
#   --prompt, -p <string>        Text prompt describing the edit (required)
#   --output, -o <path>          Output file path (default: output/<timestamp>/qwen_edit_<timestamp>.png)
#   --output-dir <path>           Output directory (default: output)
#   --controlnet-depth <path>     Depth map image for ControlNet (optional)
#   --controlnet-edge <path>      Edge map image for ControlNet (optional)
#   --controlnet-keypoint <path> Keypoint map image for ControlNet (optional)
#   --num-inference-steps <int>  Number of inference steps (default: 25)
#   --guidance-scale <float>      Guidance scale (default: 7.5)
#   --seed <int>                  Random seed (optional)
#   --api-token <string>          Replicate API token (or set REPLICATE_API_TOKEN env var)
#   --help, -h                     Show help message
#
# Examples:
#   elixir qwen_image_edit_plus.exs photo.jpg "Change the background to a beach"
#   elixir qwen_image_edit_plus.exs --image person1.jpg --image person2.jpg "Both people standing together in a park"
#   elixir qwen_image_edit_plus.exs product.png "Create a professional poster with warm lighting" --output-dir results

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
  {:base64, "~> 0.1.0"},
])

Logger.configure(level: :info)

# Load shared utilities
Code.eval_file("shared_utils.exs")

# Initialize OpenTelemetry
OtelSetup.configure()

# Initialize Python environment with minimal dependencies for Replicate API
# Replicate API client only needs requests and PIL for image handling
Pythonx.uv_init("""
[project]
name = "qwen-image-edit-plus"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "replicate",
  "pillow",
  "requests",
]
""")

# Parse command-line arguments
defmodule ArgsParser do
  def show_help do
    IO.puts("""
    Qwen Image Edit Plus Script
    Edit images with precise control using Qwen's 20B parameter image editing model
    
    Model: qwen/qwen-image-edit-plus (via Replicate API)
    Repository: https://replicate.com/qwen/qwen-image-edit-plus
    
    Usage:
      elixir qwen_image_edit_plus.exs <image_path> "<prompt>" [options]
      elixir qwen_image_edit_plus.exs --image <path1> --image <path2> "<prompt>" [options]
    
    Arguments:
      <image_path>                 Input image file (required, can specify multiple)
      <prompt>                     Text prompt describing the edit (required)
    
    Options:
      --image, -i <path>            Input image path(s) (required, can specify multiple times for multi-image editing, 1-3 images recommended)
      --prompt, -p <string>        Text prompt describing the edit (required)
      --output, -o <path>          Output file path (default: output/<timestamp>/qwen_edit_<timestamp>.png)
      --output-dir <path>           Output directory (default: output)
      --controlnet-depth <path>     Depth map image for ControlNet (optional)
      --controlnet-edge <path>      Edge map image for ControlNet (optional)
      --controlnet-keypoint <path>  Keypoint map image for ControlNet (optional)
      --num-inference-steps <int>   Number of inference steps (default: 25)
      --guidance-scale <float>     Guidance scale (default: 7.5)
      --seed <int>                 Random seed (optional)
      --api-token <string>         Replicate API token (or set REPLICATE_API_TOKEN env var)
      --help, -h                    Show this help message
    
    Features:
      - Multi-image editing: Combine 1-3 images (person + person, person + product, person + scene)
      - Person consistency: Preserve facial identity across styles and poses
      - Product poster generation: Transform product photos into professional posters
      - Text editing: Edit text in images while preserving font and style
      - ControlNet support: Use depth, edge, or keypoint maps for precise control
    
    Examples:
      # Single image edit
      elixir qwen_image_edit_plus.exs photo.jpg "Change the background to a beach"
      
      # Multi-image composition
      elixir qwen_image_edit_plus.exs --image person1.jpg --image person2.jpg "Both people standing together in a park"
      
      # Product poster generation
      elixir qwen_image_edit_plus.exs product.png "Create a professional poster with warm lighting" --output-dir results
      
      # With ControlNet depth map
      elixir qwen_image_edit_plus.exs image.jpg "Edit the scene" --controlnet-depth depth.png
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        image: :string,
        prompt: :string,
        output: :string,
        output_dir: :string,
        controlnet_depth: :string,
        controlnet_edge: :string,
        controlnet_keypoint: :string,
        num_inference_steps: :integer,
        guidance_scale: :float,
        seed: :integer,
        api_token: :string,
        help: :boolean
      ],
      aliases: [
        i: :image,
        p: :prompt,
        o: :output,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    # Get image paths - support multiple images
    image_paths = case Keyword.get_values(opts, :image) do
      [] -> 
        # No --image flags, use positional args (first arg is image, rest might be prompt)
        if length(args) > 0 do
          [hd(args)]
        else
          []
        end
      images -> 
        # --image flags provided
        images
    end

    # Get prompt - from --prompt flag or positional args
    prompt = case Keyword.get(opts, :prompt) do
      nil ->
        # Try to get from positional args (after images)
        if length(args) > length(image_paths) do
          Enum.at(args, length(image_paths))
        else
          nil
        end
      prompt_val -> prompt_val
    end

    if length(image_paths) == 0 do
      IO.puts("""
      Error: At least one image path is required.
      
      Usage:
        elixir qwen_image_edit_plus.exs <image_path> "<prompt>" [options]
        elixir qwen_image_edit_plus.exs --image <path1> --image <path2> "<prompt>" [options]
      
      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    if !prompt || prompt == "" do
      IO.puts("""
      Error: Prompt is required.
      
      Usage:
        elixir qwen_image_edit_plus.exs <image_path> "<prompt>" [options]
      
      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    # Validate image count (1-3 recommended)
    if length(image_paths) > 3 do
      IO.puts("[WARN] More than 3 images provided. Model works best with 1-3 images.")
    end

    # Check if all image files exist
    Enum.each(image_paths, fn image_path ->
      if !File.exists?(image_path) do
        OtelLogger.error("Image file not found", [{"file.path", image_path}])
        System.halt(1)
      end
    end)

    # Get API token from option or environment
    api_token = Keyword.get(opts, :api_token) || System.get_env("REPLICATE_API_TOKEN")
    
    if !api_token || api_token == "" do
      IO.puts("""
      Error: Replicate API token is required.
      
      Set it via:
        --api-token <token>
        or
        REPLICATE_API_TOKEN environment variable
      
      Get your API token from: https://replicate.com/account/api-tokens
      """)
      System.halt(1)
    end

    %{
      image_paths: image_paths,
      prompt: prompt,
      output_path: Keyword.get(opts, :output),
      output_dir: Keyword.get(opts, :output_dir, "output"),
      controlnet_depth: Keyword.get(opts, :controlnet_depth),
      controlnet_edge: Keyword.get(opts, :controlnet_edge),
      controlnet_keypoint: Keyword.get(opts, :controlnet_keypoint),
      num_inference_steps: Keyword.get(opts, :num_inference_steps, 25),
      guidance_scale: Keyword.get(opts, :guidance_scale, 7.5),
      seed: Keyword.get(opts, :seed),
      api_token: api_token
    }
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== Qwen Image Edit Plus ===
Images: #{length(config.image_paths)} image(s)
Prompt: #{config.prompt}
Output Directory: #{config.output_dir}
Inference Steps: #{config.num_inference_steps}
Guidance Scale: #{config.guidance_scale}
#{if config.seed, do: "Seed: #{config.seed}\n", else: ""}#{if config.controlnet_depth, do: "ControlNet Depth: #{config.controlnet_depth}\n", else: ""}#{if config.controlnet_edge, do: "ControlNet Edge: #{config.controlnet_edge}\n", else: ""}#{if config.controlnet_keypoint, do: "ControlNet Keypoint: #{config.controlnet_keypoint}\n", else: ""}
""")

# Add paths to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  workspace_root: base_dir
})

# Save config to JSON for Python to read
{config_file, config_file_normalized} = ConfigFile.create(config_with_paths, "qwen_image_edit_plus_config")

# Process using Replicate API
SpanCollector.track_span("qwen_image_edit_plus.generation", fn ->
try do
  {_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
from pathlib import Path
import time
import base64
from io import BytesIO
from PIL import Image

# Replicate API client
import replicate

# Get configuration from JSON file
""" <> ConfigFile.python_path_string(config_file_normalized) <> """

image_paths = config.get('image_paths', [])
prompt = config.get('prompt')
output_path = config.get('output_path')
output_dir = config.get('output_dir', 'output')
controlnet_depth = config.get('controlnet_depth')
controlnet_edge = config.get('controlnet_edge')
controlnet_keypoint = config.get('controlnet_keypoint')
num_inference_steps = config.get('num_inference_steps', 25)
guidance_scale = config.get('guidance_scale', 7.5)
seed = config.get('seed')
api_token = config.get('api_token')

# Set Replicate API token
os.environ['REPLICATE_API_TOKEN'] = api_token

# Create output directory with timestamped subdirectory
output_dir_base = Path(output_dir).resolve()
output_dir_base.mkdir(parents=True, exist_ok=True)

tag = time.strftime("%Y%m%d_%H_%M_%S")
export_dir = output_dir_base / tag
export_dir.mkdir(parents=True, exist_ok=True)

# Copy input images to output directory for self-contained output
import shutil
for image_path in image_paths:
    src_path = Path(image_path)
    if src_path.exists():
        dst_path = export_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        print(f"[OK] Copied input image to: {dst_path}")

# Save input config JSON to output directory
output_config_path = export_dir / "input_config.json"
config_dict = {
    'image_paths': image_paths,
    'prompt': prompt,
    'num_inference_steps': num_inference_steps,
    'guidance_scale': guidance_scale,
    'seed': seed,
    'controlnet_depth': controlnet_depth,
    'controlnet_edge': controlnet_edge,
    'controlnet_keypoint': controlnet_keypoint,
}
with open(output_config_path, 'w') as f:
    json.dump(config_dict, f, indent=2)
print(f"[OK] Input config saved to: {output_config_path}")

print("\n=== Step 1: Prepare Images ===")
# Load and prepare images
images = []
for image_path in image_paths:
    img = Image.open(image_path).convert("RGB")
    images.append(img)
    print(f"[OK] Loaded image: {Path(image_path).name} ({img.size[0]}x{img.size[1]})")

# Prepare ControlNet inputs if provided
controlnet_inputs = {}
if controlnet_depth:
    depth_img = Image.open(controlnet_depth).convert("RGB")
    controlnet_inputs['depth'] = depth_img
    print(f"[OK] Loaded depth map: {Path(controlnet_depth).name}")
if controlnet_edge:
    edge_img = Image.open(controlnet_edge).convert("RGB")
    controlnet_inputs['edge'] = edge_img
    print(f"[OK] Loaded edge map: {Path(controlnet_edge).name}")
if controlnet_keypoint:
    keypoint_img = Image.open(controlnet_keypoint).convert("RGB")
    controlnet_inputs['keypoint'] = keypoint_img
    print(f"[OK] Loaded keypoint map: {Path(controlnet_keypoint).name}")

print("\n=== Step 2: Run Qwen Image Edit Plus ===")
print(f"Model: qwen/qwen-image-edit-plus")
print(f"Prompt: {prompt}")
print(f"Images: {len(images)} image(s)")

# Prepare input for Replicate API
# For multi-image, we need to pass images as a list
# The model expects: image (single) or images (list of 1-3 images)
input_data = {
    'prompt': prompt,
    'num_inference_steps': num_inference_steps,
    'guidance_scale': guidance_scale,
}

# Add images - if single image, use 'image', if multiple use 'images'
if len(images) == 1:
    input_data['image'] = images[0]
else:
    input_data['images'] = images

# Add seed if provided
if seed is not None:
    input_data['seed'] = seed

# Add ControlNet inputs if provided
if controlnet_inputs:
    input_data.update(controlnet_inputs)

# Run inference via Replicate API
print("[INFO] Calling Replicate API...")
try:
    output = replicate.run(
        "qwen/qwen-image-edit-plus",
        input=input_data
    )
    
    # Replicate returns a list of URLs or file objects
    # Handle both cases
    if isinstance(output, list):
        output_images = output
    else:
        output_images = [output]
    
    print(f"[OK] Generated {len(output_images)} output image(s)")
    
    # Save output images
    for idx, output_item in enumerate(output_images):
        if isinstance(output_item, str):
            # URL - download it
            import urllib.request
            output_filename = f"qwen_edit_{tag}_{idx+1}.png"
            output_path_final = export_dir / output_filename
            urllib.request.urlretrieve(output_item, str(output_path_final))
            print(f"[OK] Downloaded output image: {output_path_final}")
        else:
            # PIL Image or file-like object
            if hasattr(output_item, 'save'):
                # PIL Image
                output_filename = f"qwen_edit_{tag}_{idx+1}.png"
                output_path_final = export_dir / output_filename
                output_item.save(str(output_path_final))
                print(f"[OK] Saved output image: {output_path_final}")
            else:
                # File-like object
                output_filename = f"qwen_edit_{tag}_{idx+1}.png"
                output_path_final = export_dir / output_filename
                with open(output_path_final, 'wb') as f:
                    f.write(output_item.read())
                print(f"[OK] Saved output image: {output_path_final}")
    
    # If output_path was specified, also save there
    if output_path:
        output_path_resolved = Path(output_path).resolve()
        output_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        if len(output_images) > 0:
            # Copy first output to specified path
            first_output = export_dir / f"qwen_edit_{tag}_1.png"
            if first_output.exists():
                shutil.copy2(first_output, output_path_resolved)
                print(f"[OK] Copied to specified output path: {output_path_resolved}")
    
    print(f"\n=== Complete ===")
    print(f"Output directory: {export_dir}")
    print(f"Generated {len(output_images)} image(s)")
    
except Exception as e:
    print(f"[ERROR] Replicate API call failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise

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

# Export spans and metrics as JSON
SpanCollector.display_trace(config.output_dir)

