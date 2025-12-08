#!/usr/bin/env elixir

# Video Generation Script (Text-to-Video and Image-to-Video)
# Generate videos from text prompts or images using FOSS models
# Primary Model: Wan2.2 (Wan-Video) - Apache-2.0 License (FOSS)
# Repository: https://github.com/Wan-Video/Wan2.2
# Paper: https://arxiv.org/abs/2503.20314
# Website: https://wan.video
# Replicate: https://replicate.com/wan-video
#
# Usage:
#   elixir video_generation.exs "<prompt>" [options]
#   elixir video_generation.exs --image <image_path> [options]
#
# Options:
#   --image <path>                    Input image for image-to-video (I2V)
#   --prompt-json <path>              Read prompt from JSON file (e.g., thirdparty/video.json)
#   --model "wan-2.5-t2v"            Model to use: wan-2.5-t2v, wan-2.5-i2v, wan-2.5-t2v-fast, wan-2.5-i2v-fast, ti2v-5b (default: auto)
#   --width <int>                    Video width in pixels (default: 720)
#   --height <int>                   Video height in pixels (default: 1280)
#   --duration <int>                  Video duration in seconds: 5 or 10 (default: 5)
#   --fps <int>                      Frames per second: 8, 12, 16, 24 (default: 24)
#   --seed <int>                     Random seed for generation (default: 0)
#   --num-steps <int>                Number of inference steps (default: 50, lower for fast models)
#   --guidance-scale <float>         Guidance scale (default: 7.5)
#   --output-format "mp4"            Output format: mp4, gif (default: "mp4")
#   --use-replicate                  Use Replicate API instead of local model (requires API key)

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
  {:req, "~> 0.5.0"}
])

# Suppress debug logs from Req
Logger.configure(level: :info)

# Initialize Python environment with required dependencies
# Wan-Video uses diffusers, transformers, and video processing libraries
Pythonx.uv_init("""
[project]
name = "video-generation"
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
  "opencv-python",
  "imageio[ffmpeg]",
  "huggingface-hub",
  "gitpython",
  "replicate",
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
        image: :string,
        prompt_json: :string,
        model: :string,
        width: :integer,
        height: :integer,
        duration: :integer,
        fps: :integer,
        seed: :integer,
        num_steps: :integer,
        guidance_scale: :float,
        output_format: :string,
        use_replicate: :boolean
      ],
      aliases: [
        i: :image,
        m: :model,
        w: :width,
        h: :height,
        d: :duration,
        s: :seed,
        steps: :num_steps,
        g: :guidance_scale,
        f: :output_format
      ]
    )

    prompt = List.first(args)
    image_path = Keyword.get(opts, :image)
    prompt_json_path = Keyword.get(opts, :prompt_json)
    use_replicate = Keyword.get(opts, :use_replicate, false)

    # Read prompt from JSON file if provided
    if prompt_json_path && File.exists?(prompt_json_path) do
      json_content = File.read!(prompt_json_path)
      json_data = Jason.decode!(json_content)
      if is_map(json_data) && Map.has_key?(json_data, "input") do
        input_data = json_data["input"]
        if is_map(input_data) && Map.has_key?(input_data, "prompt") do
          prompt = input_data["prompt"]
          IO.puts("Read prompt from JSON: #{prompt}")
        end
      end
    end

    # Determine mode
    mode = cond do
      image_path != nil -> :image_to_video
      prompt != nil -> :text_to_video
      true -> nil
    end

    if mode == nil do
      IO.puts("""
      Error: Either a text prompt or image path is required.

      Usage:
        elixir video_generation.exs "<prompt>" [options]
        elixir video_generation.exs --image <image_path> [options]

      Options:
        --image, -i <path>              Input image for image-to-video (I2V)
        --model, -m "wan-2.5-t2v"      Model: wan-2.5-t2v, wan-2.5-i2v, wan-2.5-t2v-fast, wan-2.5-i2v-fast, ti2v-5b (default: auto)
        --width, -w <int>               Video width in pixels (default: 720)
        --height, -h <int>              Video height in pixels (default: 1280)
        --duration, -d <int>             Video duration: 5 or 10 seconds (default: 5)
        --fps <int>                     Frames per second: 8, 12, 16, 24 (default: 24)
        --seed, -s <int>                Random seed (default: 0)
        --num-steps, --steps <int>      Number of inference steps (default: 50)
        --guidance-scale, -g <float>    Guidance scale (default: 7.5)
        --output-format, -f "mp4"       Output format: mp4, gif (default: "mp4")
        --use-replicate                 Use Replicate API instead of local model
      """)
      System.halt(1)
    end

    # Validate inputs
    if image_path && !File.exists?(image_path) do
      IO.puts("Error: Image file not found: #{image_path}")
      System.halt(1)
    end

    # Model selection
    model = Keyword.get(opts, :model)
    model = if model do
      model
    else
      case mode do
        :text_to_video -> "wan-2.5-t2v"
        :image_to_video -> "wan-2.5-i2v"
      end
    end

    valid_models = ["wan-2.5-t2v", "wan-2.5-i2v", "wan-2.5-t2v-fast", "wan-2.5-i2v-fast", "ti2v-5b"]
    if model not in valid_models do
      IO.puts("Error: Invalid model '#{model}'. Valid models: #{Enum.join(valid_models, ", ")}")
      System.halt(1)
    end

    # Validate dimensions
    width = Keyword.get(opts, :width, 720)
    height = Keyword.get(opts, :height, 1280)

    if width < 256 or width > 1920 or height < 256 or height > 1920 do
      IO.puts("Error: Width and height must be between 256 and 1920 pixels")
      System.halt(1)
    end

    # Validate duration
    duration = Keyword.get(opts, :duration, 5)
    if duration not in [5, 10] do
      IO.puts("Error: Duration must be 5 or 10 seconds")
      System.halt(1)
    end

    # Validate FPS
    fps = Keyword.get(opts, :fps, 24)
    if fps not in [8, 12, 16, 24] do
      IO.puts("Error: FPS must be one of: 8, 12, 16, 24")
      System.halt(1)
    end

    # Validate output format
    output_format = Keyword.get(opts, :output_format, "mp4")
    valid_formats = ["mp4", "gif"]
    if output_format not in valid_formats do
      IO.puts("Error: Invalid output format. Must be one of: #{Enum.join(valid_formats, ", ")}")
      System.halt(1)
    end

    # Default steps based on model (fast models use fewer steps)
    default_steps = if String.contains?(model, "fast"), do: 25, else: 50
    num_steps = Keyword.get(opts, :num_steps, default_steps)

    config = %{
      mode: mode,
      prompt: prompt || "",
      image_path: image_path,
      model: model,
      width: width,
      height: height,
      duration: duration,
      fps: fps,
      seed: Keyword.get(opts, :seed, 0),
      num_steps: num_steps,
      guidance_scale: Keyword.get(opts, :guidance_scale, 7.5),
      output_format: output_format,
      use_replicate: use_replicate
    }

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

mode_str = case config.mode do
  :text_to_video -> "Text-to-Video"
  :image_to_video -> "Image-to-Video"
end

IO.puts("""
=== Video Generation (#{mode_str}) ===
Model: #{config.model}
Mode: #{config.mode}
Prompt: #{config.prompt}
Image: #{config.image_path || "N/A"}
Resolution: #{config.width}x#{config.height}
Duration: #{config.duration}s
FPS: #{config.fps}
Seed: #{config.seed}
Steps: #{config.num_steps}
Guidance Scale: #{config.guidance_scale}
Output Format: #{config.output_format}
Use Replicate: #{config.use_replicate}
""")

# Save config to JSON for Python to read
config_json = Jason.encode!(config)
File.write!("config.json", config_json)

# Handle Replicate API mode
if config.use_replicate do
  IO.puts("\n=== Using Replicate API ===")

  replicate_token = System.get_env("REPLICATE_API_TOKEN")

  if !replicate_token do
    IO.puts("""
    Error: REPLICATE_API_TOKEN environment variable is not set.

    To use Replicate API:
    1. Get your API token from https://replicate.com/account/api-tokens
    2. Set the environment variable:
       export REPLICATE_API_TOKEN=your_token_here
       (or on Windows: set REPLICATE_API_TOKEN=your_token_here)
    """)
    System.halt(1)
  end

  # Use Replicate API for generation
  {_, _python_globals} = Pythonx.eval("""
  import json
  import os
  import time
  from pathlib import Path
  import replicate
  from PIL import Image
  import cv2

  # Get configuration
  with open("config.json", 'r') as f:
      config = json.load(f)

  # Initialize Replicate client
  client = replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN"))

  # Map model names to Replicate model IDs
  # Note: Wan2.2 models on Replicate may use different naming
  # Check https://replicate.com/wan-video for actual model IDs
  model_map = {
      "wan-2.5-t2v": "wan-video/wan-2.5-t2v",
      "wan-2.5-i2v": "wan-video/wan-2.5-i2v",
      "wan-2.5-t2v-fast": "wan-video/wan-2.5-t2v-fast",
      "wan-2.5-i2v-fast": "wan-video/wan-2.5-i2v-fast",
      "ti2v-5b": "wan-video/wan-2.2-i2v-fast",  # TI2V-5B is the fast 5B model
  }

  model_id = model_map.get(config['model'], "wan-video/wan-2.5-t2v")
  mode = config['mode']
  prompt = config.get('prompt', '')
  image_path = config.get('image_path')

  print(f"\\n=== Step 1: Prepare Input ===")

  # Prepare input based on mode
  input_data = {
      "prompt": prompt if prompt else "a beautiful landscape",
      "width": config.get('width', 720),
      "height": config.get('height', 1280),
      "duration": config.get('duration', 5),
      "fps": config.get('fps', 24),
      "seed": config.get('seed', 0) if config.get('seed', 0) != 0 else None,
      "num_inference_steps": config.get('num_steps', 50),
      "guidance_scale": config.get('guidance_scale', 7.5),
  }

  # Add image input for I2V
  if mode == 'image_to_video':
      if image_path:
          print(f"Using input image: {image_path}")
          input_data["image"] = open(image_path, "rb")

  print(f"\\n=== Step 2: Generate Video via Replicate ===")
  print(f"Model: {model_id}")
  print(f"Input: {input_data}")

  # Run prediction
  output = client.run(model_id, input=input_data)

  print(f"\\n=== Step 3: Download Video ===")

  # Download video
  output_url = output if isinstance(output, str) else output[0] if isinstance(output, list) else str(output)

  import requests
  output_dir = Path("output")
  output_dir.mkdir(exist_ok=True)

  tag = time.strftime("%Y%m%d_%H_%M_%S")
  output_filename = f"video_{tag}.mp4"
  output_path = output_dir / output_filename

  print(f"Downloading video from: {output_url}")
  response = requests.get(output_url, stream=True)
  response.raise_for_status()

  with open(output_path, 'wb') as f:
      for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)

  print(f"✓ Video saved: {output_path}")

  # Clean up temp file
  if os.path.exists("temp_input_frame.png"):
      os.remove("temp_input_frame.png")

  print("\\n=== Complete ===")
  print(f"Generated video saved to: {output_path}")
  """, %{})

else
  # Use local model (requires model weights)
  IO.puts("\n=== Using Local Model ===")
  IO.puts("Note: Local model inference requires downloading model weights.")
  IO.puts("For faster setup, consider using --use-replicate flag with REPLICATE_API_TOKEN.")

  # Elixir-native Hugging Face download function
  defmodule HuggingFaceDownloader do
    @base_url "https://huggingface.co"
    @api_base "https://huggingface.co/api"

    def download_repo(repo_id, local_dir, repo_name \\ "model") do
      IO.puts("Downloading #{repo_name}...")

      File.mkdir_p!(local_dir)

      case get_file_tree(repo_id) do
        {:ok, files} ->
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
      case get_files_recursive(repo_id, revision, "") do
        {:ok, files} ->
          file_map =
            files
            |> Enum.map(fn file ->
              {file["path"], file}
            end)
            |> Map.new()

          {:ok, file_map}

        error -> error
      end
    end

    defp get_files_recursive(repo_id, revision, path) do
      url = if path == "" do
        "#{@api_base}/models/#{repo_id}/tree/#{revision}"
      else
        "#{@api_base}/models/#{repo_id}/tree/#{revision}/#{path}"
      end

      try do
        response = Req.get(url)

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
      url = "#{@base_url}/#{repo_id}/resolve/main/#{path}"
      local_path = Path.join(local_dir, path)

      file_size = info["size"] || 0
      size_mb = if file_size > 0, do: Float.round(file_size / 1024 / 1024, 1), else: 0

      filename = Path.basename(path)
      IO.write("\r  [#{current}/#{total}] Downloading: #{filename} (#{size_mb} MB)")

      if File.exists?(local_path) do
        IO.write("\r  [#{current}/#{total}] Skipped (exists): #{filename}")
      else
        local_path
        |> Path.dirname()
        |> File.mkdir_p!()

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

  IO.puts("\n=== Step 2: Download Model Weights ===")
  IO.puts("Note: Wan-Video models are large. Consider using --use-replicate for faster setup.")

  base_dir = Path.expand(".")
  model_weights_dir = Path.join([base_dir, "pretrained_weights", "Wan-Video"])

  IO.puts("Using weights directory: #{model_weights_dir}")

  # Note: Actual model repository would need to be determined
  # For now, we'll use a placeholder and note that Replicate is recommended
  IO.puts("[INFO] Local model inference requires model weights from Hugging Face.")
  IO.puts("[INFO] Model repository: wan-research/wan-video (check for specific model variants)")
  IO.puts("[INFO] For production use, --use-replicate is recommended.")

  # Local inference using Wan2.2 via diffusers
  {_, _python_globals} = Pythonx.eval("""
  import json
  import os
  import sys
  import time
  from pathlib import Path
  from PIL import Image
  import torch
  from accelerate.utils import set_seed
  import imageio

  with open("config.json", 'r') as f:
      config = json.load(f)

  print("\\n=== Step 1: Load Configuration ===")
  mode = config.get('mode')
  prompt = config.get('prompt', '')
  image_path = config.get('image_path')
  model_name = config.get('model', 'wan-2.5-i2v')
  width = config.get('width', 720)
  height = config.get('height', 1280)
  duration = config.get('duration', 5)
  fps = config.get('fps', 24)
  seed = config.get('seed', 0)
  num_steps = config.get('num_steps', 50)
  guidance_scale = config.get('guidance_scale', 7.5)
  output_format = config.get('output_format', 'mp4')

  print(f"Mode: {mode}")
  print(f"Model: {model_name}")
  print(f"Prompt: {prompt}")
  print(f"Resolution: {width}x{height}")
  print(f"Duration: {duration}s, FPS: {fps}")

  # Check if Wan2.2 is available in thirdparty
  wan2_path = Path("thirdparty") / "Wan2.2"
  if not wan2_path.exists():
      print("\\n=== Step 2: Setup Wan2.2 ===")
      print("Wan2.2 not found in thirdparty/Wan2.2")
      print("Attempting to use diffusers with Wan-Video models from Hugging Face...")
      print("\\nNote: For best results, clone Wan2.2:")
      print("  git clone https://github.com/Wan-Video/Wan2.2 thirdparty/Wan2.2")
      use_diffusers = True
  else:
      print(f"\\n=== Step 2: Using Wan2.2 from {wan2_path} ===")
      sys.path.insert(0, str(wan2_path))
      use_diffusers = False

  print("\\n=== Step 3: Load Model ===")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32

  print(f"Device: {device}, dtype: {dtype}")

  try:
      if use_diffusers:
          # Try using diffusers with Wan-Video models
          from diffusers import DiffusionPipeline

          # Map model names to Hugging Face model IDs
          model_map = {
              "wan-2.5-t2v": "Wan-Video/Wan2.2-T2V",
              "wan-2.5-i2v": "Wan-Video/Wan2.2-I2V",
              "wan-2.5-t2v-fast": "Wan-Video/Wan2.2-T2V-Fast",
              "wan-2.5-i2v-fast": "Wan-Video/Wan2.2-I2V-Fast",
              "ti2v-5b": "Wan-Video/Wan2.2-TI2V-5B",
          }

          hf_model_id = model_map.get(model_name, "Wan-Video/Wan2.2-I2V")
          print(f"Loading model from Hugging Face: {hf_model_id}")

          # Load pipeline
          pipe = DiffusionPipeline.from_pretrained(
              hf_model_id,
              torch_dtype=dtype,
              low_cpu_mem_usage=True,
          )
          pipe = pipe.to(device)
          print(f"✓ Model loaded")
      else:
          # Use local Wan2.2 generate.py
          from generate import generate_video
          print("Using local Wan2.2 generate.py")
          pipe = None
  except Exception as e:
      print(f"Error loading model: {e}")
      print("\\nFalling back to instructions...")
      print("\\nTo use local inference:")
      print("1. Clone Wan2.2: git clone https://github.com/Wan-Video/Wan2.2 thirdparty/Wan2.2")
      print("2. Install dependencies: cd thirdparty/Wan2.2 && pip install -r requirements.txt")
      print("3. Download model weights from Hugging Face")
      print("4. Run this script again")
      raise

  print("\\n=== Step 4: Generate Video ===")

  # Generate seed if needed
  if seed == 0:
      import secrets
      seed = secrets.randbelow(999999999)
      print(f"Generated random seed: {seed}")

  set_seed(seed)
  generator = torch.Generator(device=device).manual_seed(seed)

  # Create output directory
  output_dir = Path("output")
  output_dir.mkdir(exist_ok=True)
  tag = time.strftime("%Y%m%d_%H_%M_%S")
  output_filename = f"video_{tag}.{output_format}"
  output_path = output_dir / output_filename

  try:
      if use_diffusers and pipe:
          # Generate using diffusers
          if mode == 'image_to_video' and image_path:
              image = Image.open(image_path).convert("RGB")
              print(f"Generating video from image: {image_path}")
              output = pipe(
                  image=image,
                  prompt=prompt,
                  width=width,
                  height=height,
                  num_inference_steps=num_steps,
                  guidance_scale=guidance_scale,
                  generator=generator,
              )
          else:
              print(f"Generating video from prompt: {prompt}")
              output = pipe(
                  prompt=prompt,
                  width=width,
                  height=height,
                  num_inference_steps=num_steps,
                  guidance_scale=guidance_scale,
                  generator=generator,
              )

          # Get frames from output
          frames = output.frames[0] if hasattr(output, 'frames') else output.images

          # Save video
          print(f"Saving video to {output_path}...")
          imageio.mimwrite(str(output_path), frames, fps=fps, codec='libx264', quality=8)
          print(f"✓ Video saved: {output_path}")
      else:
          # Use local generate.py
          print("Using Wan2.2 generate.py (not yet implemented in this wrapper)")
          print("Please use the generate.py script directly from thirdparty/Wan2.2")
          raise NotImplementedError("Local generate.py integration not yet implemented")

  except Exception as e:
      print(f"Error during generation: {e}")
      import traceback
      traceback.print_exc()
      raise

  print("\\n=== Complete ===")
  print(f"Generated video saved to: {output_path}")
  """, %{})
end

IO.puts("\n=== Complete ===")
IO.puts("Video generation script completed!")
