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
  {:logflare_logger_backend, "~> 0.11.4"}
])

# Suppress debug logs from Req to avoid showing long URLs
Logger.configure(level: :info)

# Configure Logflare integration BEFORE Mix.install starts applications
# This ensures OpenTelemetry exporter reads the correct endpoint at startup
# Reference: https://docs.logflare.app/integrations/open-telemetry/
logflare_source_id = "ee297a54-c48f-4795-8ca1-2c4cb6e57296"
logflare_api_key = System.get_env("LOGFLARE_API_KEY") || "00b958d441b10b33026109732c60f9b7378c374c0c9908993e473b98b6d992ae"

# Set OpenTelemetry configuration BEFORE applications start
# This ensures the exporter reads the correct endpoint instead of defaulting to localhost:4318
if logflare_api_key do
  Application.put_env(:opentelemetry, :span_processor, :batch)
  Application.put_env(:opentelemetry, :traces_exporter, :otlp)
  Application.put_env(:opentelemetry, :metrics_exporter, :otlp)
  Application.put_env(:opentelemetry, :logs_exporter, :none)
  
  Application.put_env(:opentelemetry_exporter, :otlp_protocol, :grpc)
  Application.put_env(:opentelemetry_exporter, :otlp_compression, :gzip)
  Application.put_env(:opentelemetry_exporter, :otlp_endpoint, "https://otel.logflare.app:443")
  Application.put_env(:opentelemetry_exporter, :otlp_headers, [
    {"x-source", logflare_source_id},
    {"x-api-key", logflare_api_key}
  ])
end

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

# Configure Logflare integration
# Reference: https://docs.logflare.app/integrations/open-telemetry/
# Reference: https://github.com/Logflare/logflare_logger_backend
logflare_source_id = "ee297a54-c48f-4795-8ca1-2c4cb6e57296"
logflare_api_key = System.get_env("LOGFLARE_API_KEY") || "00b958d441b10b33026109732c60f9b7378c374c0c9908993e473b98b6d992ae"

# Analytics configuration will be done after parsing command-line arguments
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
      # This ensures the exporter reads the correct endpoint at startup
      # Just ensure both opentelemetry and opentelemetry_exporter are started
      # The exporter must be started for traces to be sent to Logflare
      {:ok, _} = Application.ensure_all_started(:opentelemetry)
      {:ok, _} = Application.ensure_all_started(:opentelemetry_exporter)

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

  @impl true
  def handle_initialize(_params, state) do
    OpenTelemetry.Tracer.with_span "mcp.initialize" do
      OpenTelemetry.Tracer.set_attribute("mcp.server.name", "zimage-generation")
      OpenTelemetry.Tracer.set_attribute("mcp.server.version", "1.0.0")

      {:ok, %{
        name: "zimage-generation",
        version: "1.0.0",
        capabilities: %{
          tools: %{}
        }
      }, state}
    end
  end

  @impl true
  def handle_list_tools(_params, state) do
    OpenTelemetry.Tracer.with_span "mcp.list_tools" do
      tools = [
      %{
        name: "generate_image",
        description: "Generate photorealistic images from text prompts using Z-Image-Turbo",
        input_schema: %{
          type: "object",
          properties: %{
            prompt: %{
              type: "string",
              description: "Text prompt describing the image to generate"
            },
            width: %{
              type: "integer",
              description: "Image width in pixels (default: 1024, range: 64-2048)",
              default: 1024
            },
            height: %{
              type: "integer",
              description: "Image height in pixels (default: 1024, range: 64-2048)",
              default: 1024
            },
            seed: %{
              type: "integer",
              description: "Random seed for generation (default: 0)",
              default: 0
            },
            num_steps: %{
              type: "integer",
              description: "Number of inference steps (default: 9)",
              default: 9
            },
            guidance_scale: %{
              type: "number",
              description: "Guidance scale (default: 0.0 for turbo models)",
              default: 0.0
            },
            output_format: %{
              type: "string",
              description: "Output format: png, jpg, jpeg (default: png)",
              enum: ["png", "jpg", "jpeg"],
              default: "png"
            }
          },
          required: ["prompt"]
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
          case ZImageGenerator.generate(prompt, width, height, seed, num_steps, guidance_scale, output_format) do
          {:ok, output_path} ->
            OpenTelemetry.Tracer.set_attribute("image.output_path", output_path)
            OpenTelemetry.Tracer.set_status(:ok)
            {:ok, [
              %{
                type: "text",
                text: "Image generated successfully at: #{output_path}"
              },
              %{
                type: "resource",
                uri: "file://#{Path.expand(output_path)}",
                mimeType: "image/#{output_format}"
              }
            ], state}
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

# Generation function module
defmodule ZImageGenerator do
  @moduledoc """
  Handles Z-Image-Turbo model setup and generation.
  Setup is called once to prepare the model (download weights, load pipeline, and perform a test generation).
  """
  require OpenTelemetry.Tracer

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

# Check if MCP server mode
if config.mcp_server do
  OtelLogger.info("Starting Z-Image-Turbo MCP Server", [
    {"server.port", config.mcp_port},
    {"server.endpoint", "http://localhost:#{config.mcp_port}/mcp"},
    {"server.mode", "mcp"}
  ])

  # Call setup once when server starts - crash if it fails
  case ZImageGenerator.setup() do
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

  # Create a simple Plug router
  defmodule ZImageMCPRouter do
    use Plug.Router

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

  # Start the server
  {:ok, _} = Plug.Cowboy.http(ZImageMCPRouter, [], port: config.mcp_port)

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

  # Convert CLI config to MCP tool arguments
  mcp_args = %{
    "prompt" => config.prompt,
    "width" => config.width,
    "height" => config.height,
    "seed" => config.seed,
    "num_steps" => config.num_steps,
    "guidance_scale" => config.guidance_scale,
    "output_format" => config.output_format
  }

  # Call MCP handler
  case ZImageMCPHandler.handle_call_tool("generate_image", mcp_args, %{}) do
    {:ok, content, _state} ->
      # Extract output path from content
      output_path = ZImageHelpers.extract_output_path(content)
      OtelLogger.ok("Image generation completed successfully", [
        {"generation.status", "complete"}
      ])
      if output_path do
        OtelLogger.info("Output file generated", [
          {"generation.output_path", output_path}
        ])
      else
        # Log all content items
        Enum.each(content, fn item ->
          case item do
            %{type: "text", text: text} ->
              OtelLogger.info(text)
            %{type: "resource", uri: uri} ->
              OtelLogger.info("Resource available", [
                {"resource.uri", uri}
              ])
            _ ->
              OtelLogger.info("Content item", [
                {"content.type", inspect(item)}
              ])
          end
        end)
      end
    {:error, reason, _state} ->
      OtelLogger.error("Generation failed", [
        {"generation.status", "failed"},
        {"error.reason", reason}
      ])
      System.halt(1)
  end
end
