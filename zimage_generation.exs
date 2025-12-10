#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Z-Image-Turbo Generation Script
# Generate photorealistic images from text prompts using Z-Image-Turbo
# Model: Z-Image-Turbo by Tongyi-MAI (6B parameters)
# Repository: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
#
# Architecture:
#   Native CLI API using GenServer, Behaviours, :gen_statem, Task/AsyncStream
#
# Usage:
#   elixir zimage_generation.exs "<prompt>" [options]

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

defmodule ArgsParser do
  def show_help do
    IO.puts("""
    Z-Image-Turbo Generation Script
    Generate photorealistic images from text prompts using Z-Image-Turbo

    Model: Z-Image-Turbo by Tongyi-MAI (6B parameters)
    Repository: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo

    Architecture:
      - Lazy loading: Model weights downloaded and pipeline loaded on first generation
      - Efficient: Subsequent generations reuse the loaded pipeline

    Usage:
      elixir zimage_generation.exs "<prompt>" [options]

    Options:
      --width, -w <int>               Image width in pixels (default: 1024, range: 64-2048)
      --height <int>                  Image height in pixels (default: 1024, range: 64-2048)
      --seed, -s <int>                 Random seed for generation (default: 0 = random)
      --num-steps, --steps <int>      Number of inference steps (default: 4 for turbo speed)
      --guidance-scale, -g <float>     Guidance scale (default: 0.0 for turbo models)
      --output-format, -f <format>    Output format: png, jpg, jpeg (default: "png")
      --help, -h                       Show this help message

    Examples:
      elixir zimage_generation.exs "a beautiful sunset over mountains" --width 1024 --height 1024
      elixir zimage_generation.exs "a cat wearing a hat" -w 512 -h 512 -s 42
      elixir zimage_generation.exs "futuristic cityscape" --steps 12 -f jpg

      # Multiple prompts
      elixir zimage_generation.exs "prompt 1" "prompt 2" "prompt 3"
      elixir zimage_generation.exs "cat" "dog" "bird" --width 512

    Notes:
      - First generation will download model weights (~6GB) if not present
      - Multiple prompts processed with the same loaded pipeline
      - Output saved to output/<timestamp>/zimage_<timestamp>.<format>
    """)
    System.halt(0)
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
    end

    prompts = args

    if prompts == [] do
      OtelLogger.error("At least one text prompt is required", [
        {"error.type", "missing_argument"},
        {"error.argument", "prompt"}
      ])
      System.halt(1)
    end

    width = Keyword.get(opts, :width, 1024)
    height = Keyword.get(opts, :height, 1024)

    if width < 64 or width > 2048 or height < 64 or height > 2048 do
      OtelLogger.error("Width and height must be between 64 and 2048 pixels", [
        {"error.type", "validation_error"},
        {"error.field", "width/height"}
      ])
      System.halt(1)
    end

    output_format = Keyword.get(opts, :output_format, "png")
    valid_formats = ["png", "jpg", "jpeg"]
    if output_format not in valid_formats do
      OtelLogger.error("Invalid output format", [
        {"error.type", "validation_error"},
        {"error.field", "output_format"},
        {"error.valid_values", Enum.join(valid_formats, ", ")}
      ])
      System.halt(1)
    end

    num_steps = Keyword.get(opts, :num_steps, 4)
    if num_steps < 1 do
      OtelLogger.error("num_steps must be at least 1", [
        {"error.type", "validation_error"},
        {"error.field", "num_steps"}
      ])
      System.halt(1)
    end

    guidance_scale = Keyword.get(opts, :guidance_scale, 0.0)
    if guidance_scale < 0.0 do
      OtelLogger.error("guidance_scale must be non-negative", [
        {"error.type", "validation_error"},
        {"error.field", "guidance_scale"}
      ])
      System.halt(1)
    end

    %{
      prompts: prompts,
      width: Keyword.get(opts, :width, 1024),
      height: Keyword.get(opts, :height, 1024),
      seed: Keyword.get(opts, :seed, 0),
      num_steps: Keyword.get(opts, :num_steps, 4),
      guidance_scale: Keyword.get(opts, :guidance_scale, 0.0),
      output_format: Keyword.get(opts, :output_format, "png")
    }
  end
end


# ============================================================================
# TIER 1: NATIVE CLI API
# ============================================================================

# Behaviour for image generation operations
defmodule ZImageGenerator.Behaviour do
  @moduledoc """
  Behaviour for Z-Image-Turbo model operations.
  """
  @callback setup() :: :ok | {:error, term()}
  @callback generate(String.t(), integer(), integer(), integer(), integer(), float(), String.t()) :: {:ok, Path.t()} | {:error, term()}
end

# State machine for generation workflow using :gen_statem (OTP built-in)
defmodule ZImageGeneration.StateMachine do
  @moduledoc """
  State machine for image generation workflow.
  States: :idle -> :loading -> :generating -> :complete | :error
  """
  @behaviour :gen_statem

  defstruct [
    :prompt,
    :width,
    :height,
    :seed,
    :num_steps,
    :guidance_scale,
    :output_format,
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
  def idle({:call, from}, {:start_generation, params}, data) do
    new_data = struct(data, params)
    {:next_state, :loading, new_data, [{:reply, from, :ok}]}
  end

  def idle({:call, from}, :get_state, data) do
    {:keep_state_and_data, [{:reply, from, {:idle, data}}]}
  end

  def idle(_, _, _), do: :keep_state_and_data

  # State: loading
  def loading({:call, from}, {:model_loaded}, data) do
    {:next_state, :generating, data, [{:reply, from, :ok}]}
  end

  def loading({:call, from}, {:error, reason}, data) do
    {:next_state, :error, %{data | error: reason}, [{:reply, from, {:error, reason}}]}
  end

  def loading({:call, from}, :get_state, data) do
    {:keep_state_and_data, [{:reply, from, {:loading, data}}]}
  end

  def loading(_, _, _), do: :keep_state_and_data

  # State: generating
  def generating({:call, from}, {:complete, output_path}, data) do
    {:next_state, :complete, %{data | output_path: output_path}, [{:reply, from, {:ok, output_path}}]}
  end

  def generating({:call, from}, {:error, reason}, data) do
    {:next_state, :error, %{data | error: reason}, [{:reply, from, {:error, reason}}]}
  end

  def generating({:call, from}, :get_state, data) do
    {:keep_state_and_data, [{:reply, from, {:generating, data}}]}
  end

  def generating(_, _, _), do: :keep_state_and_data

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

# GenServer for managing Z-Image-Turbo generation
defmodule ZImageGenerator.Server do
  @moduledoc """
  GenServer for managing Z-Image-Turbo image generation.
  Handles setup, generation requests, and state management.
  """
  use GenServer
  require OpenTelemetry.Tracer

  defstruct [
    :generator_impl,
    :state_machine,
    :setup_complete,
    :current_generation
  ]

  # Client API
  def start_link(opts \\ []) do
    generator_impl = Keyword.get(opts, :generator_impl, ZImageGenerator.Impl)
    GenServer.start_link(__MODULE__, generator_impl, name: __MODULE__)
  end

  def setup(server \\ __MODULE__) do
    GenServer.call(server, :setup)
  end

  def generate(server \\ __MODULE__, prompt, width, height, seed, num_steps, guidance_scale, output_format) do
    # Timeout: 10 minutes (600,000 ms) - image generation can take time, especially on first run
    timeout_ms = 600_000
    case GenServer.call(server, {:generate, prompt, width, height, seed, num_steps, guidance_scale, output_format}, timeout_ms) do
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
  def init(generator_impl) do
    {:ok, state_machine} = ZImageGeneration.StateMachine.start_link([])
    state = %__MODULE__{
      generator_impl: generator_impl,
      state_machine: state_machine,
      setup_complete: false,
      current_generation: nil
    }
    {:ok, state}
  end

  @impl true
  def handle_call(:setup, _from, state) do
    if state.setup_complete do
      {:reply, :ok, state}
    else
      case state.generator_impl.setup() do
        :ok ->
          {:reply, :ok, %{state | setup_complete: true}}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end

  @impl true
  def handle_call({:generate, prompt, width, height, seed, num_steps, guidance_scale, output_format}, from, state) do
    task = Task.async(fn ->
      # Transition to loading state
      ZImageGeneration.StateMachine.call(state.state_machine, {:start_generation, %{
        prompt: prompt,
        width: width,
        height: height,
        seed: seed,
        num_steps: num_steps,
        guidance_scale: guidance_scale,
        output_format: output_format
      }})

      # Model will be loaded during generation
      ZImageGeneration.StateMachine.call(state.state_machine, {:model_loaded})

      # Perform generation
      case ZImageGenerator.Impl.generate_with_pipeline(prompt, width, height, seed, num_steps, guidance_scale, output_format) do
        {:ok, output_path} ->
          ZImageGeneration.StateMachine.call(state.state_machine, {:complete, output_path})
          {:ok, output_path}
        {:error, reason} ->
          ZImageGeneration.StateMachine.call(state.state_machine, {:error, reason})
          {:error, reason}
      end
    end)

    # Monitor the task and reply when done
    new_state = %{state | current_generation: {from, task}, setup_complete: true}
    {:noreply, new_state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    sm_state = ZImageGeneration.StateMachine.get_state(state.state_machine)
    {:reply, %{setup_complete: state.setup_complete, state_machine: sm_state}, state}
  end

  @impl true
  def handle_info({ref, result}, state) when is_reference(ref) do
    case state.current_generation do
      {from, %Task{ref: ^ref}} ->
        GenServer.reply(from, result)
        {:noreply, %{state | current_generation: nil}}
      _ ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case state.current_generation do
      {from, %Task{ref: ^ref}} ->
        GenServer.reply(from, {:error, "Task failed: #{inspect(reason)}"})
        {:noreply, %{state | current_generation: nil}}
      _ ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info(:timeout, state) do
    case state.current_generation do
      {from, _task} ->
        GenServer.reply(from, {:error, "Generation timeout - the process took too long. This may indicate a GPU/CUDA issue or memory problem."})
        {:noreply, %{state | current_generation: nil}}
      _ ->
        {:noreply, state}
    end
  end
end

# Real implementation
defmodule ZImageGenerator.Impl do
  @moduledoc """
  Real implementation of Z-Image-Turbo generation.
  """
  @behaviour ZImageGenerator.Behaviour
  require OpenTelemetry.Tracer

  @impl ZImageGenerator.Behaviour
  def setup do
    # Setup is now handled lazily during first generation
    :ok
  end

  @impl ZImageGenerator.Behaviour
  def generate(_prompt, _width, _height, _seed, _num_steps, _guidance_scale, _output_format) do
    # This is called from the GenServer
    # The actual generation happens in the GenServer
    {:error, "generate should be called through ZImageGenerator.Server"}
  end

  # Internal function that loads the pipeline and generates
  def generate_with_pipeline(prompt, width, height, seed, num_steps, guidance_scale, output_format) do
    SpanCollector.track_span("zimage.generate", fn ->
    OpenTelemetry.Tracer.with_span "zimage.generate" do
      OpenTelemetry.Tracer.set_attribute("image.width", width)
      OpenTelemetry.Tracer.set_attribute("image.height", height)
      OpenTelemetry.Tracer.set_attribute("image.seed", seed)
      OpenTelemetry.Tracer.set_attribute("image.num_steps", num_steps)
      OpenTelemetry.Tracer.set_attribute("image.guidance_scale", guidance_scale)
      OpenTelemetry.Tracer.set_attribute("image.output_format", output_format)
      OpenTelemetry.Tracer.set_attribute("prompt.length", String.length(prompt))

      base_dir = Path.expand(".")
      zimage_weights_dir = Path.join([base_dir, "pretrained_weights", "Z-Image-Turbo"])
      repo_id = "Tongyi-MAI/Z-Image-Turbo"

      # Download weights if needed (only once)
      if !File.exists?(zimage_weights_dir) or !File.exists?(Path.join(zimage_weights_dir, "config.json")) do
        SpanCollector.track_span("zimage.download_weights", fn ->
        OpenTelemetry.Tracer.with_span "zimage.download_weights" do
          OtelLogger.info("Downloading Z-Image-Turbo models from Hugging Face", [
            {"download.weights_dir", zimage_weights_dir}
          ])

          case HuggingFaceDownloader.download_repo(repo_id, zimage_weights_dir, "Z-Image-Turbo", true) do
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
          {"model.weights_dir", zimage_weights_dir},
          {"model.weights_cached", true}
        ])
        OpenTelemetry.Tracer.set_attribute("model.weights_cached", true)
      end

      config = %{
        prompt: prompt,
        width: width,
        height: height,
        seed: seed,
        num_steps: num_steps,
        guidance_scale: guidance_scale,
        output_format: output_format,
        zimage_weights_dir: zimage_weights_dir
      }

      config_json = Jason.encode!(config)
      # Use cross-platform temp directory
      tmp_dir = System.tmp_dir!()
      File.mkdir_p!(tmp_dir)
      config_file = Path.join(tmp_dir, "zimage_config_#{System.system_time(:millisecond)}.json")
      File.write!(config_file, config_json)
      config_file_normalized = String.replace(config_file, "\\", "/")

      SpanCollector.track_span("zimage.python_generation", fn ->
      OpenTelemetry.Tracer.with_span "zimage.python_generation" do
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
from diffusers import DiffusionPipeline

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

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Performance optimizations (from Exa best practices)
if device == "cuda":
    torch.set_float32_matmul_precision("high")
    # Torch inductor optimizations for maximum speed
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True

zimage_weights_dir = Path(r"#{zimage_weights_dir}").resolve()

if zimage_weights_dir.exists() and (zimage_weights_dir / "config.json").exists():
    print(f"Loading from local directory: {zimage_weights_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(zimage_weights_dir),
        torch_dtype=dtype,
        local_files_only=False
    )
else:
    print(f"Loading from Hugging Face Hub: {MODEL_ID}")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )

pipe = pipe.to(device)

# Performance optimizations for 2x speed (from Exa)
if device == "cuda":
    # Memory format optimization
    try:
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

print(f"[OK] Pipeline loaded on {device} with dtype {dtype}")
"""

          # Generate with loaded pipeline
          Pythonx.eval(load_code <> ~S"""
# Process generation
import json
import time
from pathlib import Path
""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
""" <> ~S"""
""" <> ~S"""

prompt = config.get('prompt')
width = config.get('width', 1024)
height = config.get('height', 1024)
seed = config.get('seed', 0)
num_steps = config.get('num_steps', 4)
guidance_scale = config.get('guidance_scale', 0.0)
output_format = config.get('output_format', 'png')

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

generator = torch.Generator(device=device)
if seed == 0:
    seed = generator.seed()
else:
    generator.manual_seed(seed)

# Generate image
print(f"[INFO] Starting generation: {prompt[:50]}...")
print(f"[INFO] Parameters: {width}x{height}, {num_steps} steps, seed={seed}")
print("[INFO] Generating (optimized for speed)...")
import sys
sys.stdout.flush()

# Use inference_mode for faster execution (2x speed)
with torch.inference_mode():
    output = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

print("[INFO] Generation complete, processing image...")
sys.stdout.flush()

image = output.images[0]

tag = time.strftime("%Y%m%d_%H_%M_%S")
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True)

output_filename = f"zimage_{tag}.{output_format}"
output_path = export_dir / output_filename

if output_format.lower() in ["jpg", "jpeg"]:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3] if image.mode == "RGBA" else None)
        image = background
    image.save(str(output_path), "JPEG", quality=95)
else:
    image.save(str(output_path), "PNG")

print(f"[OK] Saved image to {output_path}")
print(f"OUTPUT_PATH:{output_path}")
""", %{"config_file_normalized" => config_file_normalized})

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
        pattern = "zimage_*.#{output_format}"
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
{:ok, _pid} = ZImageGenerator.Server.start_link(generator_impl: ZImageGenerator.Impl)

# CLI mode - generate multiple images
IO.puts("=== Z-Image-Turbo Generation ===")
IO.puts("")

prompt_count = length(config.prompts)
IO.puts("Processing #{prompt_count} prompt(s)")
IO.puts("Dimensions: #{config.width}x#{config.height}px")
IO.puts("Steps: #{config.num_steps}, Seed: #{if config.seed == 0, do: "random", else: config.seed}")
IO.puts("Format: #{String.upcase(config.output_format)}")
IO.puts("")

# Process all prompts
results = config.prompts
|> Enum.with_index(1)
|> Enum.map(fn {prompt, index} ->
  OtelLogger.info("Starting generation #{index}/#{prompt_count}", [
    {"generation.index", index},
    {"generation.total", prompt_count},
    {"generation.prompt", prompt},
    {"generation.prompt_length", String.length(prompt)},
    {"generation.width", config.width},
    {"generation.height", config.height},
    {"generation.seed", config.seed},
    {"generation.num_steps", config.num_steps},
    {"generation.guidance_scale", config.guidance_scale},
    {"generation.output_format", config.output_format}
  ])

  IO.puts("[#{index}/#{prompt_count}] Generating: #{prompt}")

  result = ZImageGenerator.Server.generate(prompt, config.width, config.height, config.seed, config.num_steps, config.guidance_scale, config.output_format)

  case result do
    {:ok, output_path} ->
      OtelLogger.ok("Generation #{index}/#{prompt_count} completed", [
        {"generation.index", index},
        {"generation.status", "complete"},
        {"generation.output_path", output_path}
      ])
      IO.puts("  ✓ Success: #{output_path}")
      {:ok, output_path}
    {:error, reason} ->
      OtelLogger.error("Generation #{index}/#{prompt_count} failed", [
        {"generation.index", index},
        {"generation.status", "failed"},
        {"error.reason", inspect(reason)}
      ])
      IO.puts("  ✗ Failed: #{inspect(reason)}")
      {:error, reason}
  end
end)

IO.puts("")

# Summary
success_count = Enum.count(results, fn r -> match?({:ok, _}, r) end)
failed_count = prompt_count - success_count

if failed_count == 0 do
  IO.puts("=== ALL SUCCESSFUL (#{success_count}/#{prompt_count}) ===")
  IO.puts("All #{success_count} image(s) generated successfully!")
  IO.puts("")
  results
  |> Enum.filter(fn r -> match?({:ok, _}, r) end)
  |> Enum.each(fn {:ok, path} -> IO.puts("  • #{path}") end)
else
  IO.puts("=== PARTIAL SUCCESS (#{success_count}/#{prompt_count} succeeded) ===")
  if success_count > 0 do
    IO.puts("Successful generations:")
    results
    |> Enum.with_index(1)
    |> Enum.filter(fn {r, _} -> match?({:ok, _}, r) end)
    |> Enum.each(fn {{:ok, path}, idx} -> IO.puts("  [#{idx}] #{path}") end)
    IO.puts("")
  end
  if failed_count > 0 do
    IO.puts("Failed generations:")
    results
    |> Enum.with_index(1)
    |> Enum.filter(fn {r, _} -> match?({:error, _}, r) end)
    |> Enum.each(fn {{:error, reason}, idx} -> IO.puts("  [#{idx}] #{inspect(reason)}") end)
  end

  System.halt(1)
end

# Display OpenTelemetry trace summary for performance debugging
SpanCollector.display_trace()
