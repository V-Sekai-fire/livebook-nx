#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Shared Utilities for Generation Scripts
# Common functionality used across all generation scripts

# ============================================================================
# SHARED DEPENDENCIES
# ============================================================================

# Standard dependencies - call Mix.install before loading this file
# For scripts without OpenTelemetry:
#   Mix.install([{:pythonx, "~> 0.4.7"}, {:jason, "~> 1.4.4"}, {:req, "~> 0.5.0"}])
#   Logger.configure(level: :info)
#   Code.eval_file("shared_utils.exs")
#
# For scripts with OpenTelemetry:
#   Mix.install([{:pythonx, "~> 0.4.7"}, {:jason, "~> 1.4.4"}, {:req, "~> 0.5.0"},
#                {:opentelemetry_api, "~> 1.3"}, {:opentelemetry, "~> 1.3"}, {:opentelemetry_exporter, "~> 1.0"}])
#   Logger.configure(level: :info)
#   Code.eval_file("shared_utils.exs")

# ============================================================================
# HUGGING FACE DOWNLOADER
# ============================================================================

defmodule HuggingFaceDownloader do
  @moduledoc """
  Downloads model repositories from Hugging Face.
  """
  @base_url "https://huggingface.co"
  @api_base "https://huggingface.co/api"

  def download_repo(repo_id, local_dir, repo_name \\ "model", use_otel \\ false) do
    if use_otel do
      OtelLogger.info("Downloading model repository", [
        {"download.repo_name", repo_name},
        {"download.repo_id", repo_id}
      ])
    else
      IO.puts("Downloading #{repo_name}...")
    end

    File.mkdir_p!(local_dir)

    case get_file_tree(repo_id) do
      {:ok, files} ->
        files_list = Map.to_list(files)
        total = length(files_list)
        if use_otel do
          OtelLogger.info("Found files to download", [
            {"download.file_count", total}
          ])
        else
          IO.puts("Found #{total} files to download")
        end

        files_list
        |> Enum.with_index(1)
        |> Enum.each(fn {{path, info}, index} ->
          download_file(repo_id, path, local_dir, info, index, total, use_otel)
        end)

        if use_otel do
          OtelLogger.ok("#{repo_name} downloaded successfully", [
            {"download.repo_name", repo_name},
            {"download.status", "completed"}
          ])
        else
          IO.puts("[OK] #{repo_name} downloaded successfully")
        end
        {:ok, local_dir}

      {:error, reason} ->
        if use_otel do
          OtelLogger.error("#{repo_name} download failed", [
            {"download.repo_name", repo_name},
            {"download.status", "failed"},
            {"error.reason", inspect(reason)}
          ])
        else
          IO.puts("[ERROR] #{repo_name} download failed: #{inspect(reason)}")
        end
        {:error, reason}
    end
  end

  defp get_file_tree(repo_id, revision \\ "main") do
    case get_files_recursive(repo_id, revision, "") do
      {:ok, files} ->
        file_map =
          files
          |> Enum.map(fn file -> {file["path"], file} end)
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
        {:ok, %{status: status}} -> raise "API returned status #{status}"
        %{status: status} -> raise "API returned status #{status}"
        {:error, reason} -> raise inspect(reason)
        other -> raise "Unexpected response: #{inspect(other)}"
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

  defp download_file(repo_id, path, local_dir, info, current, total, use_otel \\ false) do
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
        {:ok, %{status: 200}} -> IO.write("\r  [#{current}/#{total}] ✓ #{filename}")
        %{status: 200} -> IO.write("\r  [#{current}/#{total}] ✓ #{filename}")
        {:ok, %{status: status}} ->
          if use_otel do
            OtelLogger.warn("Failed to download file", [
              {"download.file_path", path},
              {"download.status_code", status}
            ])
          else
            IO.puts("\n[WARN] Failed to download file: #{path} (status: #{status})")
          end
        %{status: status} ->
          if use_otel do
            OtelLogger.warn("Failed to download file", [
              {"download.file_path", path},
              {"download.status_code", status}
            ])
          else
            IO.puts("\n[WARN] Failed to download file: #{path} (status: #{status})")
          end
        {:error, reason} ->
          if use_otel do
            OtelLogger.warn("Failed to download file", [
              {"download.file_path", path},
              {"error.reason", inspect(reason)}
            ])
          else
            IO.puts("\n[WARN] Failed to download file: #{path} (#{inspect(reason)})")
          end
      end
    end
  end
end

# ============================================================================
# CONFIG FILE UTILITIES
# ============================================================================

defmodule ConfigFile do
  @moduledoc """
  Utilities for creating and managing temporary config files for Python scripts.
  """
  def create(config_data, prefix \\ "config") do
    tmp_dir = System.tmp_dir!()
    File.mkdir_p!(tmp_dir)
    config_file = Path.join(tmp_dir, "#{prefix}_#{System.system_time(:millisecond)}.json")
    config_json = Jason.encode!(config_data)
    File.write!(config_file, config_json)
    config_file_normalized = String.replace(config_file, "\\", "/")
    {config_file, config_file_normalized}
  end

  def cleanup(config_file) do
    if File.exists?(config_file) do
      File.rm(config_file)
    end
  end

  def python_path_string(config_file_normalized) do
    """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
"""
  end
end

# ============================================================================
# OUTPUT DIRECTORY UTILITIES
# ============================================================================

defmodule OutputDir do
  @moduledoc """
  Utilities for creating timestamped output directories.
  """
  def create do
    output_dir = Path.expand("output")
    File.mkdir_p!(output_dir)
    tag = timestamp()
    export_dir = Path.join(output_dir, tag)
    File.mkdir_p!(export_dir)
    export_dir
  end

  def timestamp do
    {{year, month, day}, {hour, min, sec}} = :calendar.local_time()
    :io_lib.format("~4..0B~2..0B~2..0B_~2..0B_~2..0B_~2..0B", [year, month, day, hour, min, sec])
    |> List.to_string()
  end
end

# ============================================================================
# PYTHON UV INIT HELPERS
# ============================================================================

defmodule PythonEnv do
  @moduledoc """
  Helpers for initializing Python environments with common dependencies.
  """
  def torch_index_config do
    """
[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
"""
  end

  def init(project_name, python_version \\ "==3.10.*", dependencies, extra_config \\ "") do
    config = """
[project]
name = "#{project_name}"
version = "0.0.0"
requires-python = "#{python_version}"
dependencies = [
#{Enum.map(dependencies, fn dep -> "  \"#{dep}\"," end) |> Enum.join("\n")}
]
#{extra_config}
"""
    Pythonx.uv_init(config)
  end
end

# ============================================================================
# OPENTELEMETRY UTILITIES
# ============================================================================

defmodule OtelSetup do
  @moduledoc """
  Configures OpenTelemetry for console-only logging.
  """
  def configure do
    Application.put_env(:opentelemetry, :span_processor, :batch)
    Application.put_env(:opentelemetry, :traces_exporter, :none)
    Application.put_env(:opentelemetry, :metrics_exporter, :none)
    Application.put_env(:opentelemetry, :logs_exporter, :none)

    case Application.ensure_all_started(:opentelemetry) do
      {:ok, _} ->
        OtelLogger.info("OpenTelemetry started - logging to console only", [
          {"otel.export_method", "console"},
          {"otel.local_only", true}
        ])
        # Start span collector for performance tracking
        SpanCollector.start_link()
        :ok
      error ->
        OtelLogger.warn("Failed to start OpenTelemetry - spans will not be created", [
          {"otel.error", inspect(error)}
        ])
        {:error, error}
    end
  end
end

defmodule SpanCollector do
  @moduledoc """
  Collects OpenTelemetry span timing data for performance debugging.
  """
  use Agent

  def start_link(_opts \\ []) do
    Agent.start_link(fn -> %{spans: [], start_time: System.monotonic_time(:millisecond)} end, name: __MODULE__)
  end

  def track_span(name, fun) do
    start_time = System.monotonic_time(:millisecond)
    parent_span = get_current_span()

    Agent.update(__MODULE__, fn state ->
      span_id = :erlang.unique_integer([:positive])
      new_span = %{
        id: span_id,
        name: name,
        parent: parent_span,
        start_time: start_time,
        end_time: nil,
        duration_ms: nil
      }
      %{state | spans: [new_span | state.spans]}
    end)

    set_current_span(name)

    try do
      result = fun.()
      end_time = System.monotonic_time(:millisecond)
      duration = end_time - start_time

      Agent.update(__MODULE__, fn state ->
        spans = Enum.map(state.spans, fn span ->
          if span.name == name && span.end_time == nil do
            %{span | end_time: end_time, duration_ms: duration}
          else
            span
          end
        end)
        %{state | spans: spans}
      end)

      clear_current_span()
      result
    rescue
      e ->
        end_time = System.monotonic_time(:millisecond)
        duration = end_time - start_time

        Agent.update(__MODULE__, fn state ->
          spans = Enum.map(state.spans, fn span ->
            if span.name == name && span.end_time == nil do
              %{span | end_time: end_time, duration_ms: duration, error: Exception.message(e)}
            else
              span
            end
          end)
          %{state | spans: spans}
        end)

        clear_current_span()
        raise e
    end
  end

  defp get_current_span do
    case Process.get(:current_span) do
      nil -> :root
      span -> span
    end
  end

  defp set_current_span(name) do
    Process.put(:current_span, name)
  end

  defp clear_current_span do
    Process.delete(:current_span)
  end

  def get_trace do
    Agent.get(__MODULE__, fn state ->
      total_time = System.monotonic_time(:millisecond) - state.start_time
      %{
        spans: Enum.reverse(state.spans),
        total_time_ms: total_time,
        start_time: state.start_time
      }
    end)
  end

  def display_trace do
    trace = get_trace()

    IO.puts("")
    IO.puts("=== OpenTelemetry Trace Summary ===")
    IO.puts("")

    if trace.spans == [] do
      IO.puts("  No spans recorded")
      IO.puts("")
    else
      # Build span tree
      span_map = Enum.reduce(trace.spans, %{}, fn span, acc ->
        Map.put(acc, span.name, span)
      end)

      # Calculate total time from root spans
      root_spans = Enum.filter(trace.spans, fn span -> span.parent == :root end)
      total_span_time = Enum.sum(Enum.map(root_spans, fn s -> s.duration_ms || 0 end))

      IO.puts("  Total Execution Time: #{format_duration(trace.total_time_ms)}")
      IO.puts("  Total Span Time: #{format_duration(total_span_time)}")
      IO.puts("  Overhead: #{format_duration(trace.total_time_ms - total_span_time)}")
      IO.puts("")
      IO.puts("  Span Breakdown:")
      IO.puts("")

      # Display spans in order with indentation
      display_spans(root_spans, span_map, 0)

      IO.puts("")
    end
  end

  defp display_spans(spans, span_map, indent) do
    Enum.each(spans, fn span ->
      indent_str = String.duplicate("  ", indent)
      duration_str = if span.duration_ms, do: format_duration(span.duration_ms), else: "N/A"
      error_str = if Map.has_key?(span, :error) && span.error, do: " [ERROR: #{span.error}]", else: ""

      percentage = if span.duration_ms do
        total = Enum.sum(Enum.map(Map.values(span_map), fn s -> s.duration_ms || 0 end))
        if total > 0, do: Float.round(span.duration_ms / total * 100, 1), else: 0.0
      else
        0.0
      end

      IO.puts("#{indent_str}├─ #{span.name}: #{duration_str} (#{percentage}%)#{error_str}")

      # Display child spans
      child_spans = Enum.filter(Map.values(span_map), fn s -> s.parent == span.name end)
      if child_spans != [] do
        display_spans(child_spans, span_map, indent + 1)
      end
    end)
  end

  defp format_duration(ms) when is_integer(ms) do
    cond do
      ms >= 1000 -> "#{Float.round(ms / 1000, 2)}s"
      true -> "#{ms}ms"
    end
  end

  defp format_duration(_), do: "N/A"
end

defmodule OtelLogger do
  @moduledoc """
  Logger wrapper for OpenTelemetry attributes.
  """
  require Logger

  defp to_keyword_list(attrs) when is_list(attrs) do
    Enum.map(attrs, fn
      {k, v} when is_binary(k) ->
        atom_key = k |> String.replace(".", "_") |> String.to_atom()
        {atom_key, v}
      {k, v} when is_atom(k) -> {k, v}
      other -> other
    end)
  end

  defp to_keyword_list(attrs), do: attrs

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
