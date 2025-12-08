#!/usr/bin/env elixir

# Video Generation Script (Text-to-Video and Image-to-Video)
# Generate videos from text prompts or images using FOSS models
# Primary Model: Wan2.2 (Wan-Video) - Apache-2.0 License (FOSS)
# Repository: https://github.com/Wan-Video/Wan2.2
# Paper: https://arxiv.org/abs/2503.20314
# Website: https://wan.video
#
# Usage:
#   elixir video_generation.exs "<prompt>" [options]
#   elixir video_generation.exs --image <image_path> [options]
#
# Options:
#   --image <path>                    Input image for image-to-video (I2V)
#   --model "wan-2.5-t2v"            Model to use: wan-2.5-t2v, wan-2.5-i2v, wan-2.5-t2v-fast, wan-2.5-i2v-fast, ti2v-5b, wan-2.2-animate (default: auto)
#   --width <int>                    Video width in pixels (default: 720)
#   --height <int>                   Video height in pixels (default: 1280)
#   --duration <int>                  Video duration in seconds: 5 or 10 (default: 5)
#   --fps <int>                      Frames per second: 8, 12, 16, 24 (default: 24)
#   --seed <int>                     Random seed for generation (default: 0)
#   --num-steps <int>                Number of inference steps (default: 50, lower for fast models)
#   --guidance-scale <float>         Guidance scale (default: 7.5)
#   --output-format "mp4"            Output format: mp4, gif (default: "mp4")

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
  "tokenizers>=0.20.3",
  "accelerate",
  "pillow",
  "torch",
  "torchvision",
  "numpy>=1.23.5,<2",
  "opencv-python",
  "imageio[ffmpeg]",
  "huggingface-hub",
  "gitpython",
  "easydict",
  "tqdm",
  "scipy",
  "ftfy",
  "einops",
  "decord",
  "librosa",
  "peft",
  "onnxruntime",
  "pandas",
  "matplotlib",
  "loguru",
  "sentencepiece",
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
        model: :string,
        width: :integer,
        height: :integer,
        duration: :integer,
        fps: :integer,
        seed: :integer,
        num_steps: :integer,
        guidance_scale: :float,
        output_format: :string
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

    # Determine mode
    # If image is provided, it's image-to-video (prompt is optional)
    # If no image but prompt is provided, it's text-to-video
    mode = cond do
      image_path != nil -> :image_to_video
      prompt != nil -> :text_to_video
      true -> nil
    end

    # For text-to-video, prompt is required
    if mode == :text_to_video && (prompt == nil || prompt == "") do
      IO.puts("Error: Text prompt is required for text-to-video mode")
      System.halt(1)
    end

    if mode == nil do
      IO.puts("""
      Error: Either a text prompt or image path is required.

      Usage:
        elixir video_generation.exs "<prompt>" [options]
        elixir video_generation.exs --image <image_path> [options]

      Options:
        --image, -i <path>              Input image for image-to-video (I2V)
        --model, -m "wan-2.5-t2v"      Model: wan-2.5-t2v, wan-2.5-i2v, wan-2.5-t2v-fast, wan-2.5-i2v-fast, ti2v-5b, wan-2.2-animate (default: auto)
        --width, -w <int>               Video width in pixels (default: 720)
        --height, -h <int>              Video height in pixels (default: 1280)
        --duration, -d <int>             Video duration: 5 or 10 seconds (default: 5)
        --fps <int>                     Frames per second: 8, 12, 16, 24 (default: 24)
        --seed, -s <int>                Random seed (default: 0)
        --num-steps, --steps <int>      Number of inference steps (default: 50)
        --guidance-scale, -g <float>    Guidance scale (default: 7.5)
        --output-format, -f "mp4"       Output format: mp4, gif (default: "mp4")
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

    valid_models = ["wan-2.5-t2v", "wan-2.5-i2v", "wan-2.5-t2v-fast", "wan-2.5-i2v-fast", "ti2v-5b", "wan-2.2-animate"]
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
      output_format: output_format
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
""")

# Save config to JSON for Python to read
config_json = Jason.encode!(config)
File.write!("config.json", config_json)

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

# Use local Wan2.2 model
IO.puts("\n=== Using Local Wan2.2 Model ===")

# Check and download model weights if needed
task_map = %{
  "wan-2.5-t2v" => "t2v-A14B",
  "wan-2.5-i2v" => "i2v-A14B",
  "wan-2.5-t2v-fast" => "t2v-A14B",
  "wan-2.5-i2v-fast" => "i2v-A14B",
  "ti2v-5b" => "ti2v-5B",
  "wan-2.2-animate" => "animate-14B"
}

task = Map.get(task_map, config.model, if config.mode == :image_to_video do "i2v-A14B" else "t2v-A14B" end)

# Map task to HuggingFace model ID
model_map = %{
  "t2v-A14B" => "Wan-AI/Wan2.2-T2V-A14B",
  "i2v-A14B" => "Wan-AI/Wan2.2-I2V-A14B",
  "ti2v-5B" => "Wan-AI/Wan2.2-TI2V-5B",
  "animate-14B" => "Wan-AI/Wan2.2-Animate-14B"
}

hf_model_id = Map.get(model_map, task)

# Check if model exists and is complete
if hf_model_id do
  # Get user home directory safely
  user_home = case System.user_home() do
    {:ok, home} -> home
    :error -> System.tmp_dir!()
    home when is_binary(home) -> home  # Handle case where it returns string directly
  end

  possible_paths = [
    Path.join(["pretrained_weights", "Wan2.2", task]),
    Path.join(["pretrained_weights", "Wan-Video", task]),
    Path.join([user_home, ".cache", "wan2.2", task])
  ]

  # Check if directory exists AND has required files (at least T5 checkpoint)
  model_path = Enum.find(possible_paths, fn path ->
    if File.exists?(path) and path != "" do
      # Check for at least one required file to verify model is complete
      t5_file = Path.join(path, "models_t5_umt5-xxl-enc-bf16.pth")
      vae_file_1 = Path.join(path, "Wan2.1_VAE.pth")
      vae_file_2 = Path.join(path, "Wan2.2_VAE.pth")
      File.exists?(t5_file) or File.exists?(vae_file_1) or File.exists?(vae_file_2)
    else
      false
    end
  end)

  if is_nil(model_path) or model_path == "" do
    # Check if directory exists but is incomplete
    incomplete_path = Enum.find(possible_paths, &File.exists?/1)

    if incomplete_path do
      IO.puts("\n⚠ Model checkpoint directory exists but appears incomplete.")
      IO.puts("Missing required files. Re-downloading from HuggingFace...")
    else
      IO.puts("\n⚠ Model checkpoint not found. Downloading from HuggingFace...")
    end

    IO.puts("Model: #{hf_model_id}")
    IO.puts("This may take a while depending on your internet connection...")

    download_path = Path.join(["pretrained_weights", "Wan2.2", task])

    # Note: If incomplete_path exists, the downloader will overwrite existing files
    # No need to remove the directory first - the downloader handles this

    case HuggingFaceDownloader.download_repo(hf_model_id, download_path, "Wan2.2-#{task}") do
      {:ok, _} ->
        IO.puts("✓ Model downloaded successfully to #{download_path}")

      {:error, reason} ->
        IO.puts("\n❌ Failed to download model: #{inspect(reason)}")
        IO.puts("\nPlease download the model manually:")
        IO.puts("  Option 1: Using huggingface-cli:")
        IO.puts("    huggingface-cli download #{hf_model_id} --local-dir #{download_path}")
        IO.puts("  Option 2: Set WAN2_CKPT_DIR environment variable to point to your model directory")
        IO.puts("Expected locations: #{inspect(possible_paths)}")
        System.halt(1)
    end
  else
    IO.puts("✓ Model checkpoint found at #{model_path}")
  end
end


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

  # Save original working directory before changing
  original_cwd = Path.cwd()

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
  wan2_path = original_cwd / "thirdparty" / "Wan2.2"
  if not wan2_path.exists():
      raise FileNotFoundError(
          f"Wan2.2 not found at {wan2_path}. "
          "Please clone it: git clone https://github.com/Wan-Video/Wan2.2 thirdparty/Wan2.2"
      )

  print(f"\\n=== Step 2: Using Wan2.2 from {wan2_path} ===")

  # Add Wan2.2 to Python path (use absolute path)
  wan2_path_abs = str(wan2_path.resolve())
  sys.path.insert(0, wan2_path_abs)
  os.chdir(wan2_path_abs)  # Change to Wan2.2 directory for relative imports

  # Map our model names to Wan2.2 task names
  task_map = {
      "wan-2.5-t2v": "t2v-A14B",
      "wan-2.5-i2v": "i2v-A14B",
      "wan-2.5-t2v-fast": "t2v-A14B",  # Use same task, adjust steps
      "wan-2.5-i2v-fast": "i2v-A14B",  # Use same task, adjust steps
      "ti2v-5b": "ti2v-5B",
      "wan-2.2-animate": "animate-14B",
  }

  task = task_map.get(model_name, "i2v-A14B" if mode == "image_to_video" else "t2v-A14B")

  print(f"Task: {task}")
  print(f"Model name: {model_name}")

  print("\\n=== Step 3: Load Model ===")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Device: {device}")

  # Clear CUDA cache if using GPU to free up memory
  if device == "cuda":
      torch.cuda.empty_cache()
      import gc
      gc.collect()
      print("Cleared CUDA cache")

  # Import Wan2.2 modules
  try:
      import wan
      from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
      from wan.utils.utils import save_video

      cfg = WAN_CONFIGS[task]
      print(f"✓ Loaded Wan2.2 config for {task}")
  except Exception as e:
      print(f"Error importing Wan2.2 modules: {e}")
      import traceback
      traceback.print_exc()
      raise

  # Determine checkpoint directory (should be in pretrained_weights or specified)
  # Use original_cwd for paths since we're now in wan2_path directory
  ckpt_dir = os.environ.get("WAN2_CKPT_DIR")
  if not ckpt_dir:
      # Try common locations (relative to original working directory)
      possible_paths = [
          original_cwd / "pretrained_weights" / "Wan2.2" / task,
          original_cwd / "pretrained_weights" / "Wan-Video" / task,
          Path.home() / ".cache" / "wan2.2" / task,
      ]
      for path in possible_paths:
          if path.exists():
              ckpt_dir = str(path.resolve())
              break

      if not ckpt_dir:
          print(f"\\n⚠ Warning: Checkpoint directory not found for {task}")
          print("Please set WAN2_CKPT_DIR environment variable or download model weights")
          print(f"Expected locations: {[str(p) for p in possible_paths]}")
          raise FileNotFoundError(f"Checkpoint directory not found for task {task}")

  print(f"Checkpoint directory: {ckpt_dir}")

  # Map size to Wan2.2 size format
  # Wan2.2 uses predefined sizes like "720p", "1080p", etc.
  # For custom sizes, we'll use the closest match or default
  size_key = None
  for key, (w, h) in SIZE_CONFIGS.items():
      if w == width and h == height:
          size_key = key
          break

  if not size_key:
      # Use closest match or default
      if width <= 720 and height <= 1280:
          size_key = "720p"
      elif width <= 1080 and height <= 1920:
          size_key = "1080p"
      else:
          size_key = "720p"  # Default
      print(f"Size {width}x{height} mapped to {size_key} ({SIZE_CONFIGS[size_key]})")

  size_tuple = SIZE_CONFIGS[size_key]
  max_area = MAX_AREA_CONFIGS.get(size_key, width * height)

  # Initialize model based on task
  try:
      if "animate" in task:
          print("Initializing WanAnimate model...")
          # Use memory-saving options
          model = wan.WanAnimate(
              config=cfg,
              checkpoint_dir=ckpt_dir,
              device_id=0 if device == "cuda" else -1,
              rank=0,
              t5_cpu=True,  # Always use CPU for T5 to save GPU memory
              init_on_cpu=True,  # Initialize on CPU to save GPU memory
              convert_model_dtype=True,  # Convert dtype to save memory
          )
      elif "i2v" in task:
          print("Initializing WanI2V model...")
          # Use memory-saving options: T5 on CPU, init on CPU, convert dtype
          # This helps reduce GPU memory usage
          model = wan.WanI2V(
              config=cfg,
              checkpoint_dir=ckpt_dir,
              device_id=0 if device == "cuda" else -1,
              rank=0,
              t5_cpu=True,  # Always use CPU for T5 to save GPU memory
              init_on_cpu=True,  # Initialize on CPU to save GPU memory
              convert_model_dtype=True,  # Convert dtype to save memory
          )
      elif "ti2v" in task:
          print("Initializing WanTI2V model...")
          # Use memory-saving options
          model = wan.WanTI2V(
              config=cfg,
              checkpoint_dir=ckpt_dir,
              device_id=0 if device == "cuda" else -1,
              rank=0,
              t5_cpu=True,  # Always use CPU for T5 to save GPU memory
              init_on_cpu=True,  # Initialize on CPU to save GPU memory
              convert_model_dtype=True,  # Convert dtype to save memory
          )
      else:
          print("Initializing WanT2V model...")
          # Use memory-saving options
          model = wan.WanT2V(
              config=cfg,
              checkpoint_dir=ckpt_dir,
              device_id=0 if device == "cuda" else -1,
              rank=0,
              t5_cpu=True,  # Always use CPU for T5 to save GPU memory
              init_on_cpu=True,  # Initialize on CPU to save GPU memory
              convert_model_dtype=True,  # Convert dtype to save memory
          )
      print("✓ Model initialized")

      # Clear cache again after model initialization
      if device == "cuda":
          torch.cuda.empty_cache()
          import gc
          gc.collect()
  except torch.cuda.OutOfMemoryError as e:
      print(f"\\n❌ CUDA Out of Memory Error: {e}")
      print("\\nTry the following:")
      print("  1. Close other applications using GPU memory")
      print("  2. Restart the script to clear GPU memory")
      print("  3. Use a smaller resolution (e.g., 480p instead of 720p)")
      print("  4. Reduce the number of steps")
      print("  5. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      import traceback
      traceback.print_exc()
      raise
  except Exception as e:
      print(f"Error initializing model: {e}")
      import traceback
      traceback.print_exc()
      raise

  print("\\n=== Step 4: Generate Video ===")

  # Generate seed if needed
  if seed == 0:
      import secrets
      seed = secrets.randbelow(999999999)
      print(f"Generated random seed: {seed}")

  set_seed(seed)

  # Create output directory (relative to original working directory)
  # original_cwd was set earlier, but we need to get it from the parent of current dir
  # since we changed to wan2_path
  output_dir = original_cwd / "output"
  output_dir.mkdir(exist_ok=True)
  tag = time.strftime("%Y%m%d_%H_%M_%S")
  output_filename = f"video_{tag}.{output_format}"
  output_path = output_dir / output_filename

  try:
      # Adjust steps for fast models
      if "fast" in model_name:
          num_steps = min(num_steps, 25)  # Fast models use fewer steps

      # Use config defaults if not specified
      frame_num = cfg.frame_num
      shift = cfg.sample_shift if hasattr(cfg, 'sample_shift') else 5.0
      sample_solver = 'unipc'  # Default solver

      print(f"Generating video...")
      print(f"  Prompt: {prompt}")
      print(f"  Size: {size_key} ({size_tuple})")
      print(f"  Steps: {num_steps}")
      print(f"  Guidance: {guidance_scale}")
      print(f"  Seed: {seed}")
      print(f"  Frames: {frame_num}")

      if mode == 'image_to_video' and image_path:
          # Resolve image path relative to original working directory
          if not Path(image_path).is_absolute():
              image_path = original_cwd / image_path
          image = Image.open(str(image_path)).convert("RGB")
          print(f"  Image: {image_path}")

          # Generate video using WanI2V or WanTI2V
          if "ti2v" in task:
              video_tensor = model.generate(
                  input_prompt=prompt,
                  img=image,
                  size=size_tuple,
                  max_area=max_area,
                  frame_num=frame_num,
                  shift=shift,
                  sample_solver=sample_solver,
                  sampling_steps=num_steps,
                  guide_scale=guidance_scale,
                  seed=seed,
                  offload_model=(device == "cpu"),
              )
          else:
              video_tensor = model.generate(
                  input_prompt=prompt,
                  img=image,
                  max_area=max_area,
                  frame_num=frame_num,
                  shift=shift,
                  sample_solver=sample_solver,
                  sampling_steps=num_steps,
                  guide_scale=guidance_scale,
                  seed=seed,
                  offload_model=(device == "cpu"),
              )
      else:
          # Generate video using WanT2V
          video_tensor = model.generate(
              input_prompt=prompt,
              size=size_tuple,
              frame_num=frame_num,
              shift=shift,
              sample_solver=sample_solver,
              sampling_steps=num_steps,
              guide_scale=guidance_scale,
              seed=seed,
              offload_model=(device == "cpu"),
          )

      # Save video (video_tensor is shape [C, N, H, W])
      print(f"\\nSaving video to {output_path}...")
      save_video(video_tensor, str(output_path), fps=fps)
      print(f"✓ Video saved: {output_path}")

  except Exception as e:
      print(f"Error during generation: {e}")
      import traceback
      traceback.print_exc()
      raise
  finally:
      # Restore original directory
      os.chdir(str(original_cwd))

  print("\\n=== Complete ===")
  print(f"Generated video saved to: {output_path}")
  """, %{})

IO.puts("\n=== Complete ===")
IO.puts("Video generation script completed!")
