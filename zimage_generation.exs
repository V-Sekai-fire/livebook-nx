#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Z-Image-Turbo Generation Script
# Generate photorealistic images from text prompts using Z-Image-Turbo
# Model: Z-Image-Turbo by Tongyi-MAI (6B parameters)
# Repository: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
#
# Usage:
#   elixir zimage_generation.exs "<prompt>" [options]
#   elixir zimage_generation.exs --mcp-server [options]  # Start as MCP HTTP server
#
# Options:
#   --width <int>                   Image width in pixels (default: 1024)
#   --height <int>                   Image height in pixels (default: 1024)
#   --seed <int>                     Random seed for generation (default: 0)
#   --num-steps <int>                Number of inference steps (default: 9, results in 8 DiT forwards)
#   --guidance-scale <float>         Guidance scale (default: 0.0 for turbo models)
#   --output-format "png"            Output format: png, jpg, jpeg (default: "png")
#   --mcp-server                     Start as MCP HTTP server instead of running generation
#   --mcp-port <int>                 Port for MCP HTTP server (default: 4000)
#   --no-analytics                   Disable analytics and telemetry collection
#
# Note: Image-to-image editing is not supported by Z-Image-Turbo.
#       Z-Image-Edit (a separate model) is required for image editing but is not yet released.
#
# MCP Server Mode:
#   When --mcp-server is used, the script starts an HTTP server that exposes the generation
#   functionality via Model Context Protocol (MCP). Connect with:
#     mcp connect http://localhost:4000/mcp
#   The server will call setup() once on startup to download weights, load the model,
#   and perform a test generation to verify everything works.

# Configure Logflare OpenTelemetry BEFORE Mix.install starts applications
# Set environment variables which are read early in the application lifecycle
logflare_source_id = "ee297a54-c48f-4795-8ca1-2c4cb6e57296"
logflare_api_key = System.get_env("LOGFLARE_API_KEY") || "00b958d441b10b33026109732c60f9b7378c374c0c9908993e473b98b6d992ae"

if logflare_api_key do
  # Set OpenTelemetry configuration BEFORE Mix.install starts applications
  # OpenTelemetry data will be routed through LogflareLogger instead of direct OTLP export
  Application.put_env(:opentelemetry, :span_processor, :batch)
  # Disable direct OTLP export - we'll route through LogflareLogger instead
  Application.put_env(:opentelemetry, :traces_exporter, :none)
  Application.put_env(:opentelemetry, :metrics_exporter, :none)
  Application.put_env(:opentelemetry, :logs_exporter, :none)
end

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
  {:req, "~> 0.5.0"},
  {:ex_mcp, git: "https://github.com/fire/ex-mcp.git"},
  {:plug, "~> 1.15"},
  {:plug_cowboy, "~> 2.7"},
  {:opentelemetry_api, "~> 1.3"},
  {:opentelemetry, "~> 1.3"},
  {:opentelemetry_exporter, "~> 1.0"},
  {:logflare_logger_backend, "~> 0.11.4"},
])

# Suppress debug logs from Req to avoid showing long URLs
Logger.configure(level: :info)

# Logflare configuration variables (already set above before Mix.install)
# Reference: https://docs.logflare.app/integrations/open-telemetry/

# Legal Notice and Analytics Configuration
# Analytics are enabled by default when LOGFLARE_API_KEY is set
# Use --no-analytics to opt out of all telemetry collection
defmodule AnalyticsConfig do
  def show_legal_notice do
    IO.puts("""
    ════════════════════════════════════════════════════════════════════════════
    ANALYTICS AND DATA COLLECTION NOTICE
    ════════════════════════════════════════════════════════════════════════════

    This software collects anonymous usage analytics and telemetry data to help
    improve the service. The following data may be collected:

    • Application logs (info, warnings, errors)
    • Performance metrics (generation time, resource usage)
    • Trace data (operation spans and events)
    • Error reports and stack traces

    Data is sent to Logflare (https://logflare.app) for analysis.

    You can opt out at any time by using the --no-analytics flag:
      elixir zimage_generation.exs --no-analytics "<prompt>"

    For more information about data collection and privacy, please refer to:
    https://logflare.app/privacy

    By continuing, you acknowledge that you have read and understood this notice.
    ════════════════════════════════════════════════════════════════════════════
    """)
  end
end

# Analytics configuration will be done after parsing command-line arguments
# Note: logflare_source_id and logflare_api_key are already defined above (before Mix.install)
# This function configures analytics based on the opt-out flag
defmodule AnalyticsSetup do
  def configure(analytics_enabled, api_key, source_id) do
    if api_key && analytics_enabled do
      # Show legal notice on first run
      AnalyticsConfig.show_legal_notice()

      # Configure Logger to use Logflare backend
      Application.put_env(:logger, :backends, [LogflareLogger.HttpBackend])

      Application.put_env(:logflare_logger_backend, :url, "https://api.logflare.app")
      Application.put_env(:logflare_logger_backend, :level, :info)
      Application.put_env(:logflare_logger_backend, :api_key, api_key)
      Application.put_env(:logflare_logger_backend, :source_id, source_id)
      Application.put_env(:logflare_logger_backend, :flush_interval, 1_000)
      Application.put_env(:logflare_logger_backend, :max_batch_size, 50)
      Application.put_env(:logflare_logger_backend, :metadata, :all)

      # OpenTelemetry configuration is already set before Mix.install (see above)
      # OpenTelemetry is used for creating spans, but data is routed through LogflareLogger
      # No direct OTLP export - all telemetry goes through Logflare HTTP API
      case Application.ensure_all_started(:opentelemetry) do
        {:ok, _} ->
          OtelLogger.info("OpenTelemetry started - spans will be logged via LogflareLogger", [
            {"otel.export_method", "logflare_logger"},
            {"otel.otlp_export", "disabled"}
          ])
        error ->
          OtelLogger.warn("Failed to start OpenTelemetry - spans will not be created", [
            {"otel.error", inspect(error)}
          ])
      end

      # Reconfigure Logger to apply backend changes
      Logger.add_backend(LogflareLogger.HttpBackend)

      :enabled
    else
      # Disable all analytics
      Application.put_env(:opentelemetry, :traces_exporter, :none)
      Application.put_env(:opentelemetry, :metrics_exporter, :none)
      Application.put_env(:opentelemetry, :logs_exporter, :none)
      :disabled
    end
  end
end

# Analytics will be configured after parsing command-line arguments

# Helper module for structured logging to Logflare
# IMPORTANT: This module writes to logflare_logger_backend (via Logger), NOT to OpenTelemetry logs
# OpenTelemetry logging is disabled - only traces and metrics are sent via OTLP
# This ensures logs go to Logflare's HTTP API while traces/metrics go to Logflare's OTLP endpoint
defmodule OtelLogger do
  require Logger

  # Convert list of tuples to keyword list for Logger
  # String keys are converted to atoms, handling dots by replacing with underscores
  defp to_keyword_list(attrs) when is_list(attrs) do
    Enum.map(attrs, fn
      {k, v} when is_binary(k) ->
        # Convert string keys to atoms, replacing dots with underscores
        atom_key = k |> String.replace(".", "_") |> String.to_atom()
        {atom_key, v}
      {k, v} when is_atom(k) ->
        {k, v}
      other ->
        other
    end)
  end

  defp to_keyword_list(attrs), do: attrs

  # All logging functions write to Logger, which is configured to use LogflareLogger.HttpBackend
  # This sends logs to Logflare's HTTP API endpoint, NOT to OpenTelemetry
  def info(message, attrs \\ []) do
    Logger.info(message, to_keyword_list(attrs))
  end

  def warn(message, attrs \\ []) do
    Logger.warning(message, to_keyword_list(attrs))
  end

  def error(message, attrs \\ []) do
    Logger.error(message, to_keyword_list(attrs))
  end

  def ok(message, attrs \\ []) do
    Logger.info(message, Keyword.merge([severity: "ok"], to_keyword_list(attrs)))
  end
end

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
      --mcp-server                     Start as MCP HTTP server instead of running generation
      --mcp-port <int>                 Port for MCP HTTP server (default: 4000)
      --mock                           Use mock implementation (skip expensive operations for testing)
      --help, -h                       Show this help message

    Note: Image-to-image editing is not supported by Z-Image-Turbo.
          Z-Image-Edit (a separate model) is required for image editing but is not yet released.

    Example:
      elixir zimage_generation.exs "a beautiful sunset over mountains" --width 1024 --height 1024
      elixir zimage_generation.exs "a cat wearing a hat" -w 512 -h 512 -s 42
      elixir zimage_generation.exs --mcp-server --mcp-port 4000
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
        mcp_server: :boolean,
        mcp_port: :integer,
        no_analytics: :boolean,
        mock: :boolean,
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

    mcp_server = Keyword.get(opts, :mcp_server, false)
    no_analytics = Keyword.get(opts, :no_analytics, false)
    mock_mode = Keyword.get(opts, :mock, false)
    analytics_enabled = !no_analytics
    prompt = List.first(args)

    # Prompt is only required in CLI mode (not MCP server mode)
    if !mcp_server and !prompt do
      OtelLogger.error("Text prompt is required", [
        {"error.type", "missing_argument"},
        {"error.argument", "prompt"}
      ])
      System.halt(1)
    end

    # Only validate parameters if not in MCP server mode (they'll be validated per-request in MCP mode)
    if !mcp_server do
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

      num_steps = Keyword.get(opts, :num_steps, 9)
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
    end

    config = %{
      prompt: prompt,
      width: Keyword.get(opts, :width, 1024),
      height: Keyword.get(opts, :height, 1024),
      seed: Keyword.get(opts, :seed, 0),
      num_steps: Keyword.get(opts, :num_steps, 9),
      guidance_scale: Keyword.get(opts, :guidance_scale, 0.0),
      output_format: Keyword.get(opts, :output_format, "png"),
      mcp_server: mcp_server,
      mcp_port: Keyword.get(opts, :mcp_port, 4000),
      mock_mode: mock_mode,
      analytics_enabled: analytics_enabled
    }

    config
  end
end

# Elixir-native Hugging Face download function (must be defined before use)
defmodule HuggingFaceDownloader do
  @base_url "https://huggingface.co"
  @api_base "https://huggingface.co/api"

  def download_repo(repo_id, local_dir, repo_name \\ "model") do
    OtelLogger.info("Downloading model repository", [
      {"download.repo_name", repo_name},
      {"download.repo_id", repo_id}
    ])

    # Create directory
    File.mkdir_p!(local_dir)

    # Get file tree from Hugging Face API
    case get_file_tree(repo_id) do
      {:ok, files} ->
        # files is a map, convert to list for counting and iteration
        files_list = Map.to_list(files)
        total = length(files_list)
        OtelLogger.info("Found files to download", [
          {"download.file_count", total}
        ])

        files_list
        |> Enum.with_index(1)
        |> Enum.each(fn {{path, info}, index} ->
          download_file(repo_id, path, local_dir, info, index, total)
        end)

        OtelLogger.ok("#{repo_name} downloaded successfully", [
          {"download.repo_name", repo_name},
          {"download.status", "completed"}
        ])
        {:ok, local_dir}

      {:error, reason} ->
        OtelLogger.error("#{repo_name} download failed", [
          {"download.repo_name", repo_name},
          {"download.status", "failed"},
          {"error.reason", inspect(reason)}
        ])
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
          OtelLogger.warn("Failed to download file", [
            {"download.file_path", path},
            {"download.status_code", status}
          ])

        %{status: status} ->
          OtelLogger.warn("Failed to download file", [
            {"download.file_path", path},
            {"download.status_code", status}
          ])

        {:error, reason} ->
          OtelLogger.warn("Failed to download file", [
            {"download.file_path", path},
            {"error.reason", inspect(reason)}
          ])
      end
    end
  end
end

# MCP Server Handler
defmodule ZImageMCPHandler do
  use ExMCP.Server.Handler
  require OpenTelemetry.Tracer

  @impl true
  def init(_args), do: {:ok, %{}}

  # Handle GenServer calls from ExMCP.HttpPlug
  # ExMCP.HttpPlug wraps the handler in a GenServer and calls these methods
  def handle_call({:initialize, params}, _from, state) do
    case handle_initialize(params, state) do
      {:ok, result, new_state} -> {:reply, {:ok, result}, new_state}
      {:error, reason, new_state} -> {:reply, {:error, reason}, new_state}
    end
  end

  def handle_call({:list_tools, params}, _from, state) do
    # Handle both nil params and map params
    case handle_list_tools(params || %{}, state) do
      {:ok, result, new_state} -> {:reply, result, new_state}
      {:error, reason, new_state} -> {:reply, {:error, reason}, new_state}
    end
  end

  def handle_call({:call_tool, params}, _from, state) when is_map(params) do
    tool_name = Map.get(params, "name")
    args = Map.get(params, "arguments", %{})
    case handle_call_tool(tool_name, args, state) do
      {:ok, content, new_state} -> {:reply, {:ok, content}, new_state}
      {:error, reason, new_state} -> {:reply, {:error, reason}, new_state}
    end
  end

  # Handle ExMCP.HttpPlug calling with {:call_tool, tool_name, args} format
  # Args may have atom keys, convert to string keys for handle_call_tool
  def handle_call({:call_tool, tool_name, args}, _from, state) when is_binary(tool_name) do
    # Convert atom keys to string keys if needed
    string_keyed_args = if is_map(args) do
      Enum.into(args, %{}, fn
        {k, v} when is_atom(k) -> {Atom.to_string(k), v}
        {k, v} when is_binary(k) -> {k, v}
        other -> other
      end)
    else
      args
    end
    case handle_call_tool(tool_name, string_keyed_args, state) do
      {:ok, content, new_state} -> {:reply, {:ok, content}, new_state}
      {:error, reason, new_state} -> {:reply, {:error, reason}, new_state}
    end
  end

  def handle_call(msg, _from, state) do
    {:reply, {:error, "Unknown call: #{inspect(msg)}"}, state}
  end

  @impl true
  def handle_initialize(_params, state) do
    OpenTelemetry.Tracer.with_span "mcp.initialize" do
      OpenTelemetry.Tracer.set_attribute("mcp.server.name", "zimage-generation")
      OpenTelemetry.Tracer.set_attribute("mcp.server.version", "1.0.0")

      result = %{
        "protocolVersion" => "2024-11-05",
        "serverInfo" => %{
          "name" => "zimage-generation",
          "version" => "1.0.0"
        },
        "capabilities" => %{
          "tools" => %{}
        }
      }
      
      {:ok, result, state}
    end
  end

  @impl true
  def handle_list_tools(_params, state) do
    OpenTelemetry.Tracer.with_span "mcp.list_tools" do
      tools = [
        %{
          "name" => "generate_image",
          "description" => "Generate photorealistic images from text prompts using Z-Image-Turbo",
          "inputSchema" => %{
            "type" => "object",
            "properties" => %{
              "prompt" => %{
                "type" => "string",
                "description" => "Text prompt describing the image to generate"
              },
              "width" => %{
                "type" => "integer",
                "description" => "Image width in pixels (default: 1024, range: 64-2048)",
                "default" => 1024
              },
              "height" => %{
                "type" => "integer",
                "description" => "Image height in pixels (default: 1024, range: 64-2048)",
                "default" => 1024
              },
              "seed" => %{
                "type" => "integer",
                "description" => "Random seed for generation (default: 0)",
                "default" => 0
              },
              "num_steps" => %{
                "type" => "integer",
                "description" => "Number of inference steps (default: 9)",
                "default" => 9
              },
              "guidance_scale" => %{
                "type" => "number",
                "description" => "Guidance scale (default: 0.0 for turbo models)",
                "default" => 0.0
              },
              "output_format" => %{
                "type" => "string",
                "description" => "Output format: png, jpg, jpeg (default: png)",
                "enum" => ["png", "jpg", "jpeg"],
                "default" => "png"
              }
            },
            "required" => ["prompt"]
          }
        }
      ]
      OpenTelemetry.Tracer.set_attribute("mcp.tools.count", length(tools))
      {:ok, tools, state}
    end
  end

  @impl true
  def handle_call_tool("generate_image", args, state) do
    OpenTelemetry.Tracer.with_span "mcp.tool.generate_image" do
      # Get mcp_port from application config or default
      mcp_port = Application.get_env(:zimage_generation, :mcp_port, 4001)
      prompt = Map.get(args, "prompt")

      if !prompt do
        OpenTelemetry.Tracer.set_status(:error, "prompt is required")
        {:error, "prompt is required", state}
      else
        try do
          # Extract parameters with defaults
          width = Map.get(args, "width", 1024)
          height = Map.get(args, "height", 1024)
          seed = Map.get(args, "seed", 0)
          num_steps = Map.get(args, "num_steps", 9)
          guidance_scale = Map.get(args, "guidance_scale", 0.0)
          output_format = Map.get(args, "output_format", "png")

          # Set span attributes
          OpenTelemetry.Tracer.set_attribute("image.width", width)
          OpenTelemetry.Tracer.set_attribute("image.height", height)
          OpenTelemetry.Tracer.set_attribute("image.seed", seed)
          OpenTelemetry.Tracer.set_attribute("image.num_steps", num_steps)
          OpenTelemetry.Tracer.set_attribute("image.guidance_scale", guidance_scale)
          OpenTelemetry.Tracer.set_attribute("image.output_format", output_format)
          OpenTelemetry.Tracer.set_attribute("prompt.length", String.length(prompt))

          # Call the generation function
          # Get the configured generator (real or mock)
          generator = Application.get_env(:zimage_generation, :generator, ZImageGenerator)
          
          case generator.generate(prompt, width, height, seed, num_steps, guidance_scale, output_format) do
          {:ok, output_path} ->
            OpenTelemetry.Tracer.set_attribute("image.output_path", output_path)
            OpenTelemetry.Tracer.set_status(:ok)
            
            # Convert file path to HTTP URL
            # output_path is like: output/20251209_12_34_56/zimage_20251209_12_34_56.png
            # We need: /images/20251209_12_34_56/zimage_20251209_12_34_56.png
            output_dir = Path.expand("output")
            abs_output_path = Path.expand(output_path)
            relative_path = Path.relative_to(abs_output_path, output_dir)
            # Normalize path separators for URL
            relative_path = String.replace(relative_path, "\\", "/")
            # Get port from state or config
            mcp_port = Application.get_env(:zimage_generation, :mcp_port, 4001)
            image_url = "http://localhost:#{mcp_port}/images/#{relative_path}"
            
            content = [
              %{
                "type" => "text",
                "text" => "Image generated successfully. View at: #{image_url}"
              },
              %{
                "type" => "image",
                "data" => image_url,
                "mimeType" => "image/#{output_format}"
              },
              %{
                "type" => "resource",
                "uri" => image_url,
                "mimeType" => "image/#{output_format}"
              }
            ]
            {:ok, content, state}
        {:error, reason} ->
          OpenTelemetry.Tracer.set_status(:error, "Generation failed: #{reason}")
          {:error, "Generation failed: #{reason}", state}
        end
      rescue
        e ->
          OpenTelemetry.Tracer.record_exception(e, [])
          OpenTelemetry.Tracer.set_status(:error, Exception.message(e))
          {:error, "Generation failed: #{Exception.message(e)}", state}
      end
    end
    end
  end

  @impl true
  def handle_call_tool(tool_name, _args, state) do
    {:error, "Unknown tool: #{tool_name}", state}
  end
end

# Behaviour for ZImageGenerator to allow mocking
defmodule ZImageGenerator.Behaviour do
  @moduledoc """
  Behaviour for Z-Image-Turbo model operations.
  Allows mocking expensive operations for testing.
  """
  @callback setup() :: :ok | {:error, term()}
  @callback generate(String.t(), integer(), integer(), integer(), integer(), float(), String.t()) :: {:ok, Path.t()} | {:error, term()}
end

# Generation function module
defmodule ZImageGenerator do
  @moduledoc """
  Handles Z-Image-Turbo model setup and generation.
  Setup is called once to prepare the model (download weights, load pipeline, and perform a test generation).
  """
  @behaviour ZImageGenerator.Behaviour
  require OpenTelemetry.Tracer

  @impl ZImageGenerator.Behaviour
  def setup do
    OpenTelemetry.Tracer.with_span "zimage.setup" do
      OtelLogger.info("Setting up Z-Image-Turbo Model", [
        {"setup.phase", "start"}
      ])

      base_dir = Path.expand(".")
      zimage_weights_dir = Path.join([base_dir, "pretrained_weights", "Z-Image-Turbo"])
      OpenTelemetry.Tracer.set_attribute("model.weights_dir", zimage_weights_dir)

      # Download model weights if not already present
      repo_id = "Tongyi-MAI/Z-Image-Turbo"
      OpenTelemetry.Tracer.set_attribute("model.repo_id", repo_id)

      if !File.exists?(zimage_weights_dir) or !File.exists?(Path.join(zimage_weights_dir, "config.json")) do
        OpenTelemetry.Tracer.with_span "zimage.download_weights" do
          OtelLogger.info("Downloading Z-Image-Turbo models from Hugging Face", [
            {"download.weights_dir", zimage_weights_dir}
          ])

          case HuggingFaceDownloader.download_repo(repo_id, zimage_weights_dir, "Z-Image-Turbo") do
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
      else
        OtelLogger.ok("Model weights already present", [
          {"model.weights_dir", zimage_weights_dir},
          {"model.weights_cached", true}
        ])
        OpenTelemetry.Tracer.set_attribute("model.weights_cached", true)
      end

      # Load pipeline and perform a test generation
      OtelLogger.info("Loading pipeline and performing test generation", [
        {"setup.phase", "load_pipeline"}
      ])
      OpenTelemetry.Tracer.with_span "zimage.load_pipeline" do
        try do
      {_, _python_globals} = Pythonx.eval("""
import json
import os
import sys
import logging
from pathlib import Path
from PIL import Image
import torch
from diffusers import DiffusionPipeline

# Suppress Hugging Face Hub and tqdm logging/progress bars during setup
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Suppress tqdm output
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Monkey patch tqdm to suppress output during setup
_original_tqdm_init = tqdm.__init__
def _silent_tqdm_init(self, *args, **kwargs):
    kwargs['disable'] = True
    return _original_tqdm_init(self, *args, **kwargs)
tqdm.__init__ = _silent_tqdm_init

base_dir = Path(".").resolve()
zimage_weights_dir = base_dir / "pretrained_weights" / "Z-Image-Turbo"

# Set CPU thread optimization
cpu_count = os.cpu_count()
half_cpu_count = cpu_count // 2
os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
torch.set_num_threads(half_cpu_count)

# Load model
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

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
print(f"[OK] Pipeline loaded on {device} with dtype {dtype}")

# Perform a test generation to verify everything works
print("Performing test generation...")
test_prompt = "a simple red circle"
generator = torch.Generator(device=device)
generator.manual_seed(42)

test_output = pipe(
    prompt=test_prompt,
    width=512,
    height=512,
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=generator,
)

test_image = test_output.images[0]
print(f"[OK] Test generation successful (image size: {test_image.size})")
print("[OK] Setup complete - model ready for generation")
""", %{})

          OtelLogger.ok("Model setup complete with test generation", [
            {"setup.phase", "complete"},
            {"setup.status", "success"}
          ])
          OpenTelemetry.Tracer.set_status(:ok)
          :ok
        rescue
          e ->
            OpenTelemetry.Tracer.record_exception(e, [])
            OpenTelemetry.Tracer.set_status(:error, Exception.message(e))
            OtelLogger.error("Setup failed", [
              {"setup.phase", "failed"},
              {"setup.status", "error"},
              {"error.message", Exception.message(e)}
            ])
            OtelLogger.error("Server cannot start without successful setup")
            raise e
        end
      end
    end
  end

  @impl ZImageGenerator.Behaviour
  def generate(prompt, width, height, seed, num_steps, guidance_scale, output_format) do
    OpenTelemetry.Tracer.with_span "zimage.generate" do
      OpenTelemetry.Tracer.set_attribute("image.width", width)
      OpenTelemetry.Tracer.set_attribute("image.height", height)
      OpenTelemetry.Tracer.set_attribute("image.seed", seed)
      OpenTelemetry.Tracer.set_attribute("image.num_steps", num_steps)
      OpenTelemetry.Tracer.set_attribute("image.guidance_scale", guidance_scale)
      OpenTelemetry.Tracer.set_attribute("image.output_format", output_format)
      OpenTelemetry.Tracer.set_attribute("prompt.length", String.length(prompt))

      base_dir = Path.expand(".")
      config = %{
      prompt: prompt,
      width: width,
      height: height,
      seed: seed,
      num_steps: num_steps,
      guidance_scale: guidance_scale,
      output_format: output_format,
      zimage_weights_dir: Path.join([base_dir, "pretrained_weights", "Z-Image-Turbo"])
    }

      config_json = Jason.encode!(config)
      File.write!("config.json", config_json)

      # Execute the Python generation code
      OpenTelemetry.Tracer.with_span "zimage.python_generation" do
        try do
      {_, _python_globals} = Pythonx.eval("""
import json
import sys
import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import DiffusionPipeline

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
zimage_weights_dir = config.get('zimage_weights_dir')

# Set CPU thread optimization
cpu_count = os.cpu_count()
half_cpu_count = cpu_count // 2
os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
torch.set_num_threads(half_cpu_count)

# Resolve paths
zimage_weights_dir = Path(zimage_weights_dir).resolve()
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Load model
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

if zimage_weights_dir.exists() and (zimage_weights_dir / "config.json").exists():
    print(f"Loading from local directory: {zimage_weights_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(zimage_weights_dir),
        torch_dtype=dtype
    )
else:
    print(f"Loading from Hugging Face Hub: {MODEL_ID}")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )

pipe = pipe.to(device)
print(f"[OK] Pipeline loaded on {device}")

# Generate image
generator = torch.Generator(device=device)
if seed == 0:
    seed = generator.seed()
else:
    generator.manual_seed(seed)

output = pipe(
    prompt=prompt,
    width=width,
    height=height,
    num_inference_steps=num_steps,
    guidance_scale=guidance_scale,
    generator=generator,
)

image = output.images[0]

# Save image
import time
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
""", %{})

      # Extract output path from Python output
      # The Python code prints the path, we need to capture it
      # For now, construct it from the timestamp pattern
          {:ok, output_path} = find_latest_output(output_format)
          {:ok, output_path}
        rescue
          e ->
            OpenTelemetry.Tracer.record_exception(e, [])
            OpenTelemetry.Tracer.set_status(:error, Exception.message(e))
            {:error, Exception.message(e)}
        end
      end
    end
  end

  defp find_latest_output(output_format) do
    output_dir = Path.expand("output")
    if File.exists?(output_dir) do
      # Find the most recent directory
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

# Helper function to extract output path from MCP content
defmodule ZImageHelpers do
  def extract_output_path(content) when is_list(content) do
    Enum.find_value(content, fn item ->
      case item do
        %{type: "resource", uri: uri} ->
          # Extract file path from file:// URI
          if String.starts_with?(uri, "file://") do
            String.replace_prefix(uri, "file://", "")
          else
            nil
          end
        %{type: "text", text: text} ->
          # Try to extract path from text
          case Regex.run(~r/(?:Output file|Saved|Path):\s*(.+)/i, text) do
            [_, path] -> String.trim(path)
            _ -> nil
          end
        _ ->
          nil
      end
    end)
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

# Configure analytics based on user preference
AnalyticsSetup.configure(config.analytics_enabled, logflare_api_key, logflare_source_id)

# Mock stub implementation for testing
defmodule ZImageGeneratorMock.Stub do
  @moduledoc """
  Stub implementation of ZImageGenerator.Behaviour for testing.
  Returns mock responses without expensive operations.
  """
  @behaviour ZImageGenerator.Behaviour
  
  @impl ZImageGenerator.Behaviour
  def setup do
    OtelLogger.info("Mock setup - skipping model download and loading", [
      {"test.mode", "mock"},
      {"setup.status", "mocked"}
    ])
    :ok
  end
  
  @impl ZImageGenerator.Behaviour
  def generate(prompt, width, height, _seed, _num_steps, _guidance_scale, output_format) do
    OtelLogger.info("Mock generation - creating solid color image", [
      {"test.mode", "mock"},
      {"prompt", prompt},
      {"width", width},
      {"height", height},
      {"output_format", output_format}
    ])
    
    # Create output directory with timestamp (matching real implementation)
    base_dir = Path.expand(".")
    output_base = Path.join([base_dir, "output"])
    File.mkdir_p!(output_base)
    
    # Create timestamped folder for this run (matching real implementation)
    timestamp = DateTime.utc_now() |> DateTime.to_string() |> String.replace(~r/[^\d]/, "") |> String.slice(0..13)
    export_dir = Path.join([output_base, timestamp])
    File.mkdir_p!(export_dir)
    
    # Generate filename matching real implementation format
    output_filename = "zimage_#{timestamp}.#{output_format}"
    output_path = Path.join([export_dir, output_filename])
    
    # Create a solid color image with correct dimensions and format
    # Use a pleasant blue-gray color
    {_, _python_globals} = Pythonx.eval("""
from PIL import Image
import os
from pathlib import Path

# Create a solid color image with the exact requested dimensions
width = #{width}
height = #{height}
output_format = "#{output_format}".lower()

# Use a pleasant blue-gray color (RGB: 100, 150, 200)
color = (100, 150, 200)

# Create the image
img = Image.new('RGB', (width, height), color=color)

# Ensure output directory exists
output_path = Path(r"#{output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save with correct format
if output_format in ['jpg', 'jpeg']:
    # Convert RGBA to RGB for JPEG if needed
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
        img = background
    img.save(str(output_path), "JPEG", quality=95)
else:
    img.save(str(output_path), "PNG")

print(f"OUTPUT_PATH:{output_path}")
""", %{})
    
    # Extract output path from Python print output or use the path we constructed
    output_path_actual = output_path
    
    OtelLogger.ok("Mock image generated successfully", [
      {"test.mode", "mock"},
      {"output_path", output_path_actual},
      {"image.width", width},
      {"image.height", height},
      {"image.format", output_format}
    ])
    
    {:ok, output_path_actual}
  end
end

# Setup mocking if --mock flag is set
if config.mock_mode do
  # Set the generator to use mock stub
  Application.put_env(:zimage_generation, :generator, ZImageGeneratorMock.Stub)
  
  OtelLogger.info("Mock mode enabled - using ZImageGeneratorMock.Stub", [
    {"test.mode", "mock"},
    {"test.generator", "ZImageGeneratorMock.Stub"}
  ])
else
  # Use real implementation
  Application.put_env(:zimage_generation, :generator, ZImageGenerator)
end

# Get the configured generator
generator = Application.get_env(:zimage_generation, :generator, ZImageGenerator)

# Check if MCP server mode
if config.mcp_server do
  OtelLogger.info("Starting Z-Image-Turbo MCP Server", [
    {"server.port", config.mcp_port},
    {"server.endpoint", "http://localhost:#{config.mcp_port}/mcp"},
    {"server.mode", "mcp"},
    {"server.transport", "http"}
  ])

  # Get the configured generator (real or mock) and call setup
  generator = Application.get_env(:zimage_generation, :generator, ZImageGenerator)
  
  # Call setup once when server starts - crash if it fails
  case generator.setup() do
    :ok ->
      OtelLogger.ok("Setup completed successfully", [
        {"setup.status", "success"}
      ])
    {:error, reason} ->
      OtelLogger.error("Setup failed", [
        {"setup.status", "failed"},
        {"error.reason", inspect(reason)}
      ])
      System.halt(1)
  end

  # Create a Plug router with static file serving for images
  defmodule ZImageMCPRouter do
    use Plug.Router

    # Serve static files from output directory
    # Plug.Static serves files relative to the "from" directory
    # So /images/20251209_12_34_56/file.png maps to output/20251209_12_34_56/file.png
    plug Plug.Static,
      at: "/images",
      from: Path.expand("output"),
      gzip: false,
      content_types: %{"png" => "image/png", "jpg" => "image/jpeg", "jpeg" => "image/jpeg"}

    plug :match
    plug :dispatch

    forward "/mcp", to: ExMCP.HttpPlug,
      handler: ZImageMCPHandler,
      server_info: %{name: "zimage-generation", version: "1.0.0"},
      sse_enabled: true,
      cors_enabled: true

    match _ do
      send_resp(conn, 404, "Not Found")
    end
  end

  # Store port in application config for use in handlers
  Application.put_env(:zimage_generation, :mcp_port, config.mcp_port)
  
  # Start the server
  {:ok, _} = Plug.Cowboy.http(ZImageMCPRouter, [], port: config.mcp_port)
  
  OtelLogger.info("Image server started", [
    {"server.images_endpoint", "http://localhost:#{config.mcp_port}/images"},
    {"server.mcp_endpoint", "http://localhost:#{config.mcp_port}/mcp"}
  ])

  # Keep the process alive
  Process.sleep(:infinity)
else
  # Normal CLI mode - route through MCP handler
  OtelLogger.info("Starting Z-Image-Turbo Generation", [
    {"generation.prompt_length", String.length(config.prompt)},
    {"generation.width", config.width},
    {"generation.height", config.height},
    {"generation.seed", config.seed},
    {"generation.num_steps", config.num_steps},
    {"generation.guidance_scale", config.guidance_scale},
    {"generation.output_format", config.output_format}
  ])

  # Get the configured generator (real or mock)
  generator = Application.get_env(:zimage_generation, :generator, ZImageGenerator)
  
  # Call generation directly (bypassing MCP handler for CLI mode)
  case generator.generate(config.prompt, config.width, config.height, config.seed, config.num_steps, config.guidance_scale, config.output_format) do
    {:ok, output_path} ->
      OtelLogger.ok("Image generation completed successfully", [
        {"generation.status", "complete"},
        {"generation.output_path", output_path}
      ])
      IO.puts("\n[OK] Image generated successfully!")
      IO.puts("Output: #{output_path}")
    {:error, reason} ->
      OtelLogger.error("Generation failed", [
        {"generation.status", "failed"},
        {"error.reason", inspect(reason)}
      ])
      IO.puts("\n[ERROR] Generation failed: #{inspect(reason)}")
      System.halt(1)
  end
end
