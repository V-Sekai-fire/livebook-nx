#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Qwen Image Edit Plus Script
# Edit images using Qwen Image Edit 2509 model
# Model: Qwen-Image-Edit-2509 by Qwen Team
# Repository: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
#
# Architecture:
#   Native CLI API using GenServer, Behaviours, :gen_statem, Task/AsyncStream
#
# Usage:
#   elixir qwen_image_edit_plus.exs <image1> [image2] [image3] "<prompt>" [options]

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

# Initialize Python environment
Pythonx.uv_init("""
[project]
name = "qwen-image-edit-plus"
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

defmodule ArgsParser do
  def show_help do
    IO.puts("""
    Qwen Image Edit Plus Script
    Edit images using Qwen Image Edit 2509 model

    Model: Qwen-Image-Edit-2509 by Qwen Team
    Repository: https://huggingface.co/Qwen/Qwen-Image-Edit-2509

    Architecture:
      - Lazy loading: Model weights downloaded and pipeline loaded on first generation
      - Efficient: Subsequent generations reuse the loaded pipeline
      - Multi-image support: Process 1-3 input images (supports multiple images as list)

    Usage:
      elixir qwen_image_edit_plus.exs <image1> [image2] [image3] "<prompt>" [options]

    Arguments:
      image1, image2, image3    Input image file paths (1-3 images supported)
      prompt                    Text prompt describing the edit

    Options:
      --go-fast, -f             Use fast inference mode (fewer steps, default: true)
      --use-4bit                Use 4-bit quantization to reduce VRAM (default: false, requires bitsandbytes)
      --aspect-ratio <ratio>    Aspect ratio: "match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4" (default: "match_input_image", note: currently not used by pipeline)
      --output-format, -o <fmt>  Output format: webp, png, jpg, jpeg (default: "webp")
      --output-quality, -q <int> Output quality for webp/jpg (default: 95, range: 1-100)
      --help, -h                 Show this help message

    Examples:
      elixir qwen_image_edit_plus.exs image1.jpg "Make the background blue"
      elixir qwen_image_edit_plus.exs image1.jpg image2.jpg "The person adopts the pose from image 1"
      elixir qwen_image_edit_plus.exs image.jpg "Change the rabbit's color to purple" --go-fast false
      elixir qwen_image_edit_plus.exs image.jpg "Edit text in the image" --output-format png

    Notes:
      - First generation will download model weights if not present (~20GB full precision, ~10GB with 4-bit)
      - Model supports multiple images (pass as list to pipeline)
      - Use --use-4bit to reduce VRAM usage (recommended for GPUs with <24GB VRAM)
      - Output saved to output/<timestamp>/qwen_edit_<timestamp>.<format>
      - Based on Qwen-Image-Edit-2509: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
    """)
    System.halt(0)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        go_fast: :boolean,
        use_4bit: :boolean,
        aspect_ratio: :string,
        output_format: :string,
        output_quality: :integer,
        help: :boolean
      ],
      aliases: [
        f: :go_fast,
        o: :output_format,
        q: :output_quality,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
    end

    # Parse images and prompt from args
    # Images come first, then prompt (last argument in quotes)
    {images, prompt} = parse_images_and_prompt(args)

    if images == [] do
      OtelLogger.error("At least one image is required", [
        {"error.type", "missing_argument"},
        {"error.argument", "image"}
      ])
      System.halt(1)
    end

    if length(images) > 3 do
      OtelLogger.error("Maximum 3 images supported", [
        {"error.type", "validation_error"},
        {"error.argument", "image"},
        {"error.count", length(images)}
      ])
      System.halt(1)
    end

    if prompt == nil or prompt == "" do
      OtelLogger.error("Prompt is required", [
        {"error.type", "missing_argument"},
        {"error.argument", "prompt"}
      ])
      System.halt(1)
    end

    # Validate image files exist
    Enum.each(images, fn image_path ->
      if !File.exists?(image_path) do
        OtelLogger.error("Image file not found", [
          {"error.type", "file_not_found"},
          {"error.path", image_path}
        ])
        System.halt(1)
      end
    end)

    output_format = Keyword.get(opts, :output_format, "webp")
    valid_formats = ["webp", "png", "jpg", "jpeg"]
    if output_format not in valid_formats do
      OtelLogger.error("Invalid output format", [
        {"error.type", "validation_error"},
        {"error.field", "output_format"},
        {"error.valid_values", Enum.join(valid_formats, ", ")}
      ])
      System.halt(1)
    end

    output_quality = Keyword.get(opts, :output_quality, 95)
    if output_quality < 1 or output_quality > 100 do
      OtelLogger.error("output_quality must be between 1 and 100", [
        {"error.type", "validation_error"},
        {"error.field", "output_quality"}
      ])
      System.halt(1)
    end

    aspect_ratio = Keyword.get(opts, :aspect_ratio, "match_input_image")
    valid_aspect_ratios = ["match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4"]
    if aspect_ratio not in valid_aspect_ratios do
      OtelLogger.error("Invalid aspect ratio", [
        {"error.type", "validation_error"},
        {"error.field", "aspect_ratio"},
        {"error.valid_values", Enum.join(valid_aspect_ratios, ", ")}
      ])
      System.halt(1)
    end

    %{
      images: images,
      prompt: prompt,
      go_fast: Keyword.get(opts, :go_fast, true),
      use_4bit: Keyword.get(opts, :use_4bit, false),
      aspect_ratio: aspect_ratio,
      output_format: output_format,
      output_quality: output_quality
    }
  end

  defp parse_images_and_prompt(args) do
    # Find the prompt (usually the last argument, may be in quotes)
    # Images are all arguments before the prompt
    case args do
      [] -> {[], nil}
      [single_arg] ->
        # Could be image or prompt - check if it's a file
        if File.exists?(single_arg) do
          {[single_arg], nil}
        else
          {[], single_arg}
        end
      _ ->
        # Try to find where prompt starts (first non-file argument)
        {images, rest} = Enum.split_while(args, &File.exists?/1)
        prompt = Enum.join(rest, " ")
        if prompt == "", do: {images, nil}, else: {images, prompt}
    end
  end
end


# ============================================================================
# TIER 1: NATIVE CLI API
# ============================================================================

# Behaviour for image editing operations
defmodule QwenImageEdit.Behaviour do
  @moduledoc """
  Behaviour for Qwen Image Edit Plus model operations.
  """
  @callback setup() :: :ok | {:error, term()}
  @callback edit(list(String.t()), String.t(), boolean(), boolean(), String.t(), String.t(), integer()) :: {:ok, Path.t()} | {:error, term()}
end

# State machine for editing workflow using :gen_statem (OTP built-in)
defmodule QwenImageEdit.StateMachine do
  @moduledoc """
  State machine for image editing workflow.
  States: :idle -> :loading -> :editing -> :complete | :error
  """
  @behaviour :gen_statem

  defstruct [
    :images,
    :prompt,
    :go_fast,
    :aspect_ratio,
    :output_format,
    :output_quality,
    :output_path,
    :error
  ]

  # Client API
  def start_link(opts \\ []) do
    :gen_statem.start_link(__MODULE__, [], opts)
  end

  def call(server, event) do
    :gen_statem.call(server, event)
  end

  def get_state(server) do
    :gen_statem.call(server, :get_state)
  end

  # State machine callbacks
  @impl :gen_statem
  def callback_mode, do: :state_functions

  @impl :gen_statem
  def init(_), do: {:ok, :idle, %__MODULE__{}}

  # State: idle
  def idle({:call, from}, {:start_edit, params}, data) do
    new_data = struct(data, params)
    {:next_state, :loading, new_data, [{:reply, from, :ok}]}
  end

  def idle({:call, from}, :get_state, data) do
    {:keep_state_and_data, [{:reply, from, {:idle, data}}]}
  end

  def idle(_, _, _), do: :keep_state_and_data

  # State: loading
  def loading({:call, from}, {:model_loaded}, data) do
    {:next_state, :editing, data, [{:reply, from, :ok}]}
  end

  def loading({:call, from}, {:error, reason}, data) do
    {:next_state, :error, %{data | error: reason}, [{:reply, from, {:error, reason}}]}
  end

  def loading({:call, from}, :get_state, data) do
    {:keep_state_and_data, [{:reply, from, {:loading, data}}]}
  end

  def loading(_, _, _), do: :keep_state_and_data

  # State: editing
  def editing({:call, from}, {:complete, output_path}, data) do
    {:next_state, :complete, %{data | output_path: output_path}, [{:reply, from, {:ok, output_path}}]}
  end

  def editing({:call, from}, {:error, reason}, data) do
    {:next_state, :error, %{data | error: reason}, [{:reply, from, {:error, reason}}]}
  end

  def editing({:call, from}, :get_state, data) do
    {:keep_state_and_data, [{:reply, from, {:editing, data}}]}
  end

  def editing(_, _, _), do: :keep_state_and_data

  # State: complete
  def complete({:call, from}, {:reset}, _data) do
    {:next_state, :idle, %__MODULE__{}, [{:reply, from, :ok}]}
  end

  def complete({:call, from}, :get_state, data) do
    {:keep_state_and_data, [{:reply, from, {:complete, data}}]}
  end

  def complete(_, _, _), do: :keep_state_and_data

  # State: error
  def error({:call, from}, {:reset}, _data) do
    {:next_state, :idle, %__MODULE__{}, [{:reply, from, :ok}]}
  end

  def error({:call, from}, :get_state, data) do
    {:keep_state_and_data, [{:reply, from, {:error, data}}]}
  end

  def error(_, _, _), do: :keep_state_and_data
end

# GenServer for managing Qwen Image Edit Plus generation
defmodule QwenImageEdit.Server do
  @moduledoc """
  GenServer for managing Qwen Image Edit Plus image editing.
  Handles setup, editing requests, and state management.
  """
  use GenServer
  require OpenTelemetry.Tracer

  defstruct [
    :editor_impl,
    :state_machine,
    :setup_complete,
    :current_edit
  ]

  # Client API
  def start_link(opts \\ []) do
    editor_impl = Keyword.get(opts, :editor_impl, QwenImageEdit.Impl)
    GenServer.start_link(__MODULE__, editor_impl, name: __MODULE__)
  end

  def setup(server \\ __MODULE__) do
    GenServer.call(server, :setup)
  end

  def edit(server \\ __MODULE__, images, prompt, go_fast, use_4bit, aspect_ratio, output_format, output_quality) do
    # Timeout: 30 minutes (1,800,000 ms) - model loading can take 10-15 minutes on first run
    timeout_ms = 1_800_000
    case GenServer.call(server, {:edit, images, prompt, go_fast, use_4bit, aspect_ratio, output_format, output_quality}, timeout_ms) do
      {:error, _reason} = error -> error
      {:ok, _path} = ok -> ok
      other -> {:error, "Unexpected response: #{inspect(other)}"}
    end
  end

  def get_state(server \\ __MODULE__) do
    GenServer.call(server, :get_state)
  end

  # Server callbacks
  @impl true
  def init(editor_impl) do
    {:ok, state_machine} = QwenImageEdit.StateMachine.start_link([])
    state = %__MODULE__{
      editor_impl: editor_impl,
      state_machine: state_machine,
      setup_complete: false,
      current_edit: nil
    }
    {:ok, state}
  end

  @impl true
  def handle_call(:setup, _from, state) do
    if state.setup_complete do
      {:reply, :ok, state}
    else
      case state.editor_impl.setup() do
        :ok ->
          {:reply, :ok, %{state | setup_complete: true}}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end

  @impl true
  def handle_call({:edit, images, prompt, go_fast, use_4bit, aspect_ratio, output_format, output_quality}, from, state) do
    task = Task.async(fn ->
      # Transition to loading state
      QwenImageEdit.StateMachine.call(state.state_machine, {:start_edit, %{
        images: images,
        prompt: prompt,
        go_fast: go_fast,
        use_4bit: use_4bit,
        aspect_ratio: aspect_ratio,
        output_format: output_format,
        output_quality: output_quality
      }})

      # Model will be loaded during editing
      QwenImageEdit.StateMachine.call(state.state_machine, {:model_loaded})

      # Perform editing
      case QwenImageEdit.Impl.edit_with_pipeline(images, prompt, go_fast, use_4bit, aspect_ratio, output_format, output_quality) do
        {:ok, output_path} ->
          QwenImageEdit.StateMachine.call(state.state_machine, {:complete, output_path})
          {:ok, output_path}
        {:error, reason} ->
          QwenImageEdit.StateMachine.call(state.state_machine, {:error, reason})
          {:error, reason}
      end
    end)

    # Monitor the task and reply when done
    new_state = %{state | current_edit: {from, task}, setup_complete: true}
    {:noreply, new_state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    sm_state = QwenImageEdit.StateMachine.get_state(state.state_machine)
    {:reply, %{setup_complete: state.setup_complete, state_machine: sm_state}, state}
  end

  @impl true
  def handle_info({ref, result}, state) when is_reference(ref) do
    case state.current_edit do
      {from, %Task{ref: ^ref}} ->
        GenServer.reply(from, result)
        {:noreply, %{state | current_edit: nil}}
      _ ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case state.current_edit do
      {from, %Task{ref: ^ref}} ->
        GenServer.reply(from, {:error, "Task failed: #{inspect(reason)}"})
        {:noreply, %{state | current_edit: nil}}
      _ ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info(:timeout, state) do
    case state.current_edit do
      {from, _task} ->
        GenServer.reply(from, {:error, "Editing timeout - the process took too long. This may indicate a GPU/CUDA issue or memory problem."})
        {:noreply, %{state | current_edit: nil}}
      _ ->
        {:noreply, state}
    end
  end
end

# Real implementation
defmodule QwenImageEdit.Impl do
  @moduledoc """
  Real implementation of Qwen Image Edit Plus editing.
  """
  @behaviour QwenImageEdit.Behaviour
  require OpenTelemetry.Tracer

  @impl QwenImageEdit.Behaviour
  def setup do
    # Setup is now handled lazily during first edit
    :ok
  end

  @impl QwenImageEdit.Behaviour
  def edit(_images, _prompt, _go_fast, _use_4bit, _aspect_ratio, _output_format, _output_quality) do
    # This is called from the GenServer
    # The actual editing happens in the GenServer
    {:error, "edit should be called through QwenImageEdit.Server"}
  end

  # Internal function that loads the pipeline and edits
  def edit_with_pipeline(images, prompt, go_fast, use_4bit, aspect_ratio, output_format, output_quality) do
    SpanCollector.track_span("qwen_edit.edit", fn ->
    OpenTelemetry.Tracer.with_span "qwen_edit.edit" do
      OpenTelemetry.Tracer.set_attribute("image.count", length(images))
      OpenTelemetry.Tracer.set_attribute("image.go_fast", go_fast)
      OpenTelemetry.Tracer.set_attribute("image.use_4bit", use_4bit)
      OpenTelemetry.Tracer.set_attribute("image.aspect_ratio", aspect_ratio)
      OpenTelemetry.Tracer.set_attribute("image.output_format", output_format)
      OpenTelemetry.Tracer.set_attribute("image.output_quality", output_quality)
      OpenTelemetry.Tracer.set_attribute("prompt.length", String.length(prompt))

      base_dir = Path.expand(".")
      
      # Choose model directory and repo based on quantization
      {qwen_weights_dir, repo_id} = if use_4bit do
        {
          Path.join([base_dir, "pretrained_weights", "Qwen-Image-Edit-2509-4bit"]),
          "ovedrive/Qwen-Image-Edit-2509-4bit"
        }
      else
        {
          Path.join([base_dir, "pretrained_weights", "Qwen-Image-Edit-2509"]),
          "Qwen/Qwen-Image-Edit-2509"
        }
      end

      # Download weights if needed (only once)
      if !File.exists?(qwen_weights_dir) or !File.exists?(Path.join(qwen_weights_dir, "config.json")) do
        SpanCollector.track_span("qwen_edit.download_weights", fn ->
        OpenTelemetry.Tracer.with_span "qwen_edit.download_weights" do
          model_name = if use_4bit, do: "Qwen-Image-Edit-2509-4bit", else: "Qwen-Image-Edit-2509"
          OtelLogger.info("Downloading #{model_name} models from Hugging Face", [
            {"download.weights_dir", qwen_weights_dir},
            {"download.use_4bit", use_4bit}
          ])

          case HuggingFaceDownloader.download_repo(repo_id, qwen_weights_dir, model_name, true) do
            {:ok, _} ->
              OtelLogger.ok("Model weights downloaded", [
                {"download.status", "completed"}
              ])
              OpenTelemetry.Tracer.set_status(:ok)
            {:error, reason} ->
              OtelLogger.warn("Download had errors", [
                {"download.status", "partial"},
                {"error.reason", inspect(reason)}
              ])
              OtelLogger.info("Model will be loaded from Hugging Face Hub if local files are incomplete")
              OpenTelemetry.Tracer.set_status(:error, inspect(reason))
          end
        end
        end)
      else
        OtelLogger.ok("Model weights already present", [
          {"model.weights_dir", qwen_weights_dir},
          {"model.weights_cached", true}
        ])
        OpenTelemetry.Tracer.set_attribute("model.weights_cached", true)
      end

      # Convert image paths to absolute paths
      image_paths = Enum.map(images, &Path.expand/1)

      config = %{
        images: image_paths,
        prompt: prompt,
        go_fast: go_fast,
        use_4bit: use_4bit,
        aspect_ratio: aspect_ratio,
        output_format: output_format,
        output_quality: output_quality,
        qwen_weights_dir: qwen_weights_dir
      }

      config_json = Jason.encode!(config)
      # Use cross-platform temp directory
      tmp_dir = System.tmp_dir!()
      File.mkdir_p!(tmp_dir)
      config_file = Path.join(tmp_dir, "qwen_edit_config_#{System.system_time(:millisecond)}.json")
      File.write!(config_file, config_json)
      config_file_normalized = String.replace(config_file, "\\", "/")

      SpanCollector.track_span("qwen_edit.python_editing", fn ->
      OpenTelemetry.Tracer.with_span "qwen_edit.python_editing" do
        try do
          OtelLogger.info("Loading pipeline", [
            {"pipeline.load", "starting"}
          ])

          load_code = ~S"""
# Load pipeline
import json
import os
import sys
import logging
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

_original_tqdm_init = tqdm.__init__
def _silent_tqdm_init(self, *args, **kwargs):
    kwargs['disable'] = True
    return _original_tqdm_init(self, *args, **kwargs)
tqdm.__init__ = _silent_tqdm_init

cpu_count = os.cpu_count()
half_cpu_count = cpu_count // 2
os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
torch.set_num_threads(half_cpu_count)

# Get config first to determine model
""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
""" <> ~S"""
""" <> ~S"""

device = "cuda" if torch.cuda.is_available() else "cpu"
use_4bit = config.get('use_4bit', False)

# Choose model based on quantization
if use_4bit and device == "cuda":
    # Try using pre-quantized model if available
    MODEL_ID = "ovedrive/Qwen-Image-Edit-2509-4bit"
    print("[INFO] Using 4-bit quantized model: ovedrive/Qwen-Image-Edit-2509-4bit")
    print("[INFO] This reduces VRAM usage significantly (~10GB vs ~20GB)")
    dtype = None  # Quantized models handle dtype internally
    # Update weights directory for quantized model
    base_qwen_weights_dir = Path(r"#{qwen_weights_dir}").resolve()
    qwen_weights_dir = base_qwen_weights_dir.parent / "Qwen-Image-Edit-2509-4bit"
else:
    MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    qwen_weights_dir = Path(r"#{qwen_weights_dir}").resolve()

# Print device information
print(f"[INFO] Device: {device}")
if device == "cuda":
    print(f"[INFO] CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] CUDA Version: {torch.version.cuda}")
else:
    print("[INFO] Using CPU (CUDA not available)")
import sys
sys.stdout.flush()

# Performance optimizations (from Exa best practices)
if device == "cuda":
    torch.set_float32_matmul_precision("high")
    # Torch inductor optimizations for maximum speed
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True

print(f"[INFO] Checking for local model weights at: {qwen_weights_dir}")
if qwen_weights_dir.exists() and (qwen_weights_dir / "config.json").exists():
    print(f"[OK] Found local model weights, loading from: {qwen_weights_dir}")
    print("[INFO] Loading pipeline (this may take 5-15 minutes on first load)...")
    import sys
    sys.stdout.flush()
    try:
        # Try with local_files_only=True first (faster if all files are present)
        print("[INFO] Attempting to load with local_files_only=True...")
        sys.stdout.flush()
        load_kwargs = {"local_files_only": True}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(qwen_weights_dir),
            **load_kwargs
        )
        print("[OK] Pipeline loaded from local directory (local_files_only=True)")
    except Exception as e:
        print(f"[INFO] local_files_only failed: {str(e)[:200]}")
        print("[INFO] Retrying with local_files_only=False (may download missing files)...")
        sys.stdout.flush()
        try:
            load_kwargs = {"local_files_only": False}
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                str(qwen_weights_dir),
                **load_kwargs
            )
            print("[OK] Pipeline loaded from local directory (with fallback)")
        except Exception as e2:
            print(f"[ERROR] Failed to load from local directory: {str(e2)[:200]}")
            print(f"[INFO] Falling back to Hugging Face Hub: {MODEL_ID}")
            sys.stdout.flush()
            load_kwargs = {}
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                MODEL_ID,
                **load_kwargs
            )
            print("[OK] Pipeline loaded from Hugging Face Hub")
else:
    print(f"[INFO] Local weights not found, loading from Hugging Face Hub: {MODEL_ID}")
    print("[INFO] This may take 10-15 minutes on first load...")
    import sys
    sys.stdout.flush()
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )
    print("[OK] Pipeline loaded from Hugging Face Hub")

print(f"[INFO] Moving pipeline to device: {device}")
import sys
sys.stdout.flush()
pipe = pipe.to(device)
print(f"[OK] Pipeline moved to {device}")

# Performance optimizations for 2x speed (from Exa)
if device == "cuda":
    # Memory format optimization
    try:
        if hasattr(pipe, 'transformer'):
            pipe.transformer.to(memory_format=torch.channels_last)
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'decode'):
            pipe.vae.to(memory_format=torch.channels_last)
        print("[OK] Memory format optimized (channels_last)")
    except Exception as e:
        print(f"[INFO] Memory format optimization: {e}")

    # torch.compile for maximum speed (from Exa best practices)
    # Note: Requires Triton package. Falls back gracefully if not available.
    try:
        # Check if Triton is available before compiling
        import triton
        if hasattr(pipe, 'transformer'):
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False)
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'decode'):
            pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=False)
        print("[OK] torch.compile enabled (reduce-overhead mode for 2x speed boost)")
    except ImportError:
        print("[INFO] Triton not installed - skipping torch.compile (install triton for 2x speed boost)")
    except Exception as e:
        # Catch TritonMissing and other compilation errors
        if "Triton" in str(e) or "triton" in str(e).lower():
            print("[INFO] Triton not available - skipping torch.compile (install triton for 2x speed boost)")
        else:
            print(f"[INFO] torch.compile not available: {e}")

if dtype is not None:
    print(f"[OK] Pipeline fully loaded and optimized on {device} with dtype {dtype}")
else:
    print(f"[OK] Pipeline fully loaded and optimized on {device} with 4-bit quantization")
print("[INFO] Pipeline ready for inference")
import sys
sys.stdout.flush()
"""

          # Edit with loaded pipeline
          Pythonx.eval(load_code <> ~S"""
# Process editing
import json
import time
from pathlib import Path
""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
""" <> ~S"""
""" <> ~S"""

images = config.get('images', [])
prompt = config.get('prompt')
go_fast = config.get('go_fast', True)
aspect_ratio = config.get('aspect_ratio', 'match_input_image')
output_format = config.get('output_format', 'webp')
output_quality = config.get('output_quality', 95)

# Load input images
input_images = []
for img_path in images:
    img_path_resolved = Path(img_path).resolve()
    if not img_path_resolved.exists():
        raise FileNotFoundError(f"Image file not found: {img_path_resolved}")
    img = Image.open(str(img_path_resolved)).convert("RGB")
    input_images.append(img)
    print(f"[OK] Loaded image: {img_path_resolved.name} ({img.size[0]}x{img.size[1]})")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Generate image
print(f"[INFO] Starting edit: {prompt[:50]}...")
print(f"[INFO] Parameters: {len(input_images)} image(s), go_fast={go_fast}, aspect_ratio={aspect_ratio}")
print("[INFO] Preparing for inference...")
import sys
sys.stdout.flush()

# Check GPU memory if on CUDA
if device == "cuda":
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory stats properly
            # Note: memory_reserved can sometimes show cached allocations that exceed
            # what's actually available, so we use allocated as the primary metric
            allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            # Calculate free memory (use max to avoid negative values)
            # Reserved includes cached allocations, so actual free might be less
            free_estimate = max(0, total - reserved)
            free_actual = max(0, total - allocated)
            
            print(f"[INFO] GPU Memory Status:")
            print(f"  Total GPU Memory: {total:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
            print(f"  Reserved (cached): {reserved:.2f} GB")
            print(f"  Free (estimated): {free_estimate:.2f} GB")
            if allocated > total * 0.9:
                print(f"[WARNING] GPU memory usage is very high ({allocated/total*100:.1f}%)")
            sys.stdout.flush()
    except Exception as e:
        print(f"[INFO] Could not check GPU memory: {e}")
        sys.stdout.flush()

# Use inference_mode for faster execution
with torch.inference_mode():
    # Qwen-Image-Edit-2509 supports multiple images as a list
    # Prepare generator
    generator = torch.Generator(device=device)
    generator.manual_seed(0)  # Use fixed seed for reproducibility
    
    # Map go_fast to num_inference_steps
    # go_fast=True means fewer steps (faster), go_fast=False means more steps (higher quality)
    if go_fast:
        num_inference_steps = 20  # Faster inference
    else:
        num_inference_steps = 40  # Default from Hugging Face example for 2509
    
    # Call pipeline with Qwen-Image-Edit-2509 API
    # API: image (list), prompt, generator, true_cfg_scale, negative_prompt, num_inference_steps, guidance_scale, num_images_per_prompt
    print(f"[INFO] Starting inference with {num_inference_steps} steps...")
    print("[INFO] This may take several minutes. Please wait...")
    sys.stdout.flush()
    
    # Prepare pipeline kwargs
    pipeline_kwargs = {
        "image": input_images,  # Pass as list (supports 1-3 images)
        "prompt": prompt,
        "generator": generator,
        "true_cfg_scale": 4.0,  # Default from Hugging Face example
        "negative_prompt": " ",  # Default from Hugging Face example
        "num_inference_steps": num_inference_steps,
        "guidance_scale": 1.0,  # Default from Hugging Face example for 2509
        "num_images_per_prompt": 1,
    }
    
    # Try to add callback if supported
    try:
        # Check if callback parameter is supported
        import inspect
        sig = inspect.signature(pipe.__call__)
        if 'callback' in sig.parameters or 'callback_on_step_end' in sig.parameters:
            def progress_callback(pipe, step_index, timestep, callback_kwargs):
                progress = int(((step_index + 1) / num_inference_steps) * 100)
                if (step_index + 1) % max(1, num_inference_steps // 10) == 0 or step_index == 0:
                    print(f"[PROGRESS] Step {step_index + 1}/{num_inference_steps} ({progress}%)")
                    sys.stdout.flush()
                return callback_kwargs
            
            # Try callback_on_step_end first (newer API)
            if 'callback_on_step_end' in sig.parameters:
                pipeline_kwargs["callback_on_step_end"] = progress_callback
            else:
                pipeline_kwargs["callback"] = progress_callback
            print("[INFO] Progress callbacks enabled")
            sys.stdout.flush()
    except Exception as e:
        print(f"[INFO] Could not enable progress callbacks: {e}")
        sys.stdout.flush()
    
    try:
        output = pipe(**pipeline_kwargs)
        print("[OK] Inference completed successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Pipeline call failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"[INFO] Attempted with {len(input_images)} image(s)")
        print("[INFO] Pipeline parameters available:")
        import inspect
        sig = inspect.signature(pipe.__call__)
        print(f"  {sig}")
        sys.stdout.flush()
        raise

print("[INFO] Editing complete, processing image...")
sys.stdout.flush()

image = output.images[0]

tag = time.strftime("%Y%m%d_%H_%M_%S")
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True)

output_filename = f"qwen_edit_{tag}.{output_format}"
output_path = export_dir / output_filename

# Save with appropriate format and quality
if output_format.lower() == "webp":
    image.save(str(output_path), "WEBP", quality=output_quality)
elif output_format.lower() in ["jpg", "jpeg"]:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3] if image.mode == "RGBA" else None)
        image = background
    image.save(str(output_path), "JPEG", quality=output_quality)
else:
    image.save(str(output_path), "PNG")

print(f"[OK] Saved image to {output_path}")
print(f"OUTPUT_PATH:{output_path}")
""", %{"config_file_normalized" => config_file_normalized, "qwen_weights_dir" => qwen_weights_dir})

          {:ok, output_path} = find_latest_output(output_format)
          {:ok, output_path}
        rescue
          e ->
            # Clean up temp file on error
            if File.exists?(config_file) do
              File.rm(config_file)
            end
            OpenTelemetry.Tracer.record_exception(e, [])
            OpenTelemetry.Tracer.set_status(:error, Exception.message(e))
            {:error, Exception.message(e)}
        after
          # Clean up temp file
          if File.exists?(config_file) do
            File.rm(config_file)
          end
        end
      end
      end)
    end
  end)
  end

  defp find_latest_output(output_format) do
    output_dir = Path.expand("output")
    if File.exists?(output_dir) do
      dirs =
        output_dir
        |> File.ls!()
        |> Enum.filter(&File.dir?(Path.join(output_dir, &1)))
        |> Enum.sort(:desc)

      if dirs != [] do
        latest_dir = List.first(dirs)
        pattern = "qwen_edit_*.#{output_format}"
        files = Path.join([output_dir, latest_dir]) |> Path.join(pattern) |> Path.wildcard()
        if files != [] do
          {:ok, List.first(files)}
        else
          {:error, "No output file found"}
        end
      else
        {:error, "No output directory found"}
      end
    else
      {:error, "Output directory does not exist"}
    end
  end
end

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# Get configuration
config = ArgsParser.parse(System.argv())

# Configure OpenTelemetry for console logging
OtelSetup.configure()

# Start GenServer
{:ok, _pid} = QwenImageEdit.Server.start_link(editor_impl: QwenImageEdit.Impl)

# CLI mode - edit image(s)
IO.puts("=== Qwen Image Edit 2509 ===")
IO.puts("")

image_count = length(config.images)
IO.puts("Processing #{image_count} image(s)")
Enum.each(config.images, fn img -> IO.puts("  • #{img}") end)
IO.puts("Prompt: #{config.prompt}")
IO.puts("Go Fast: #{config.go_fast}")
IO.puts("Aspect Ratio: #{config.aspect_ratio}")
IO.puts("Format: #{String.upcase(config.output_format)}")
IO.puts("Quality: #{config.output_quality}")
IO.puts("")

OtelLogger.info("Starting image edit", [
  {"edit.image_count", image_count},
  {"edit.prompt", config.prompt},
  {"edit.go_fast", config.go_fast},
  {"edit.aspect_ratio", config.aspect_ratio},
  {"edit.output_format", config.output_format},
  {"edit.output_quality", config.output_quality}
])

IO.puts("[1/1] Editing: #{config.prompt}")

result = QwenImageEdit.Server.edit(config.images, config.prompt, config.go_fast, config.use_4bit, config.aspect_ratio, config.output_format, config.output_quality)

case result do
  {:ok, output_path} ->
    OtelLogger.ok("Image edit completed", [
      {"edit.status", "complete"},
      {"edit.output_path", output_path}
    ])
    IO.puts("  ✓ Success: #{output_path}")
    IO.puts("")
    IO.puts("=== SUCCESS ===")
    IO.puts("Image edited successfully!")
    IO.puts("  • #{output_path}")
  {:error, reason} ->
    OtelLogger.error("Image edit failed", [
      {"edit.status", "failed"},
      {"error.reason", inspect(reason)}
    ])
    IO.puts("  ✗ Failed: #{inspect(reason)}")
    System.halt(1)
end

# Display OpenTelemetry trace summary for performance debugging - save to output directory
SpanCollector.display_trace("output")

