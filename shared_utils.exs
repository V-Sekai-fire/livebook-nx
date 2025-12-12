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
  @compile {:no_warn_undefined, [Req]}
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

  defp download_file(repo_id, path, local_dir, info, current, total, use_otel) do
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
    # Use environment variable to avoid string escaping issues
    # This is the safest approach for paths with special characters
    ~s"""
config_file_path = os.environ.get('CONFIG_FILE_PATH')
if not config_file_path:
    raise ValueError("CONFIG_FILE_PATH environment variable not set")
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
"""
  end
  
  def set_config_env(config_file_normalized) do
    # Set the config file path as an environment variable
    System.put_env("CONFIG_FILE_PATH", config_file_normalized)
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
  Configures OpenTelemetry using the official library from hex.pm.
  Sets up a JSON console exporter for spans and metrics.
  """
  require OpenTelemetry.Tracer
  
  def configure do
    # Configure batch processor for better performance
    Application.put_env(:opentelemetry, :span_processor, :batch)
    
    # Start our JSON exporter agent to collect spans
    OtelJsonExporter.start_link([])
    
    # Use a custom span processor that collects spans
    # The exporter will be called by the SDK's batch processor
    Application.put_env(:opentelemetry, :traces_exporter, OtelJsonExporter)
    Application.put_env(:opentelemetry, :metrics_exporter, OtelJsonExporter)
    Application.put_env(:opentelemetry, :logs_exporter, OtelJsonExporter)
    
    # Set up Logger backend to capture logs
    setup_log_backend()

    case Application.ensure_all_started(:opentelemetry) do
      {:ok, _} ->
        :ok
      error ->
        OtelLogger.warn("Failed to start OpenTelemetry", [
          {"otel.error", inspect(error)}
        ])
        {:error, error}
    end
  end
  
  defp setup_log_backend do
    # Add a custom Logger backend to capture logs
    # Use Elixir's Logger.add_backend for compatibility
    case Logger.add_backend(OtelLogHandler) do
      {:ok, _} -> :ok
      {:error, :already_exists} -> :ok  # Already added, that's fine
      _error -> 
        # Fallback: try the :logger API
        try do
          :logger.add_handler(:otel_json_handler, OtelLogHandler, %{
            level: :info,
            config: %{}
          })
        rescue
          _ -> :ok  # If both fail, continue without log collection
        end
    end
  end
end

defmodule OtelLogHandler do
  @moduledoc """
  Logger backend that captures logs and sends them to OtelJsonExporter.
  """
  @behaviour :gen_event
  
  def init(_args) do
    {:ok, %{}}
  end
  
  def handle_event({level, _gl, {Logger, msg, _ts, md}}, state) do
    # Extract log message and metadata
    message = format_message(msg, md)
    metadata = extract_metadata(md)
    
    # Send to OtelJsonExporter
    OtelJsonExporter.add_log(level, message, metadata)
    
    {:ok, state}
  end
  
  def handle_event(_event, state) do
    {:ok, state}
  end
  
  def handle_call({:configure, _opts}, state) do
    {:ok, :ok, state}
  end
  
  def handle_info(_msg, state) do
    {:ok, state}
  end
  
  def code_change(_old_vsn, state, _extra) do
    {:ok, state}
  end
  
  def terminate(_reason, _state) do
    :ok
  end
  
  defp format_message(msg, _md) when is_binary(msg), do: msg
  defp format_message({:string, chardata}, _md), do: IO.chardata_to_string(chardata)
  defp format_message({:report, report}, _md) when is_map(report), do: inspect(report)
  defp format_message(other, _md), do: inspect(other)
  
  defp extract_metadata(md) when is_list(md) do
    Enum.map(md, fn
      {key, value} -> {key, value}
      other -> other
    end)
  end
  defp extract_metadata(md), do: md
end

defmodule OtelJsonExporter do
  @moduledoc """
  Custom OpenTelemetry exporter that collects spans and metrics, then exports as JSON.
  Implements the OpenTelemetry exporter behavior.
  """
  use Agent
  
  def start_link(_opts \\ []) do
    Agent.start_link(fn -> %{
      spans: [],
      metrics: [],
      logs: [],
      start_time: System.monotonic_time(:millisecond)
    } end, name: __MODULE__)
  end
  
  def add_log(level, message, metadata \\ []) do
    log_entry = %{
      timestamp: System.monotonic_time(:millisecond),
      level: level,
      message: message,
      metadata: convert_metadata(metadata)
    }
    
    Agent.update(__MODULE__, fn state ->
      %{state | logs: [log_entry | state.logs]}
    end)
  end

  def add_python_log(level, message, metadata \\ [], trace_context \\ nil) do
    log_entry = %{
      timestamp: System.monotonic_time(:millisecond),
      level: level,
      message: message,
      metadata: convert_metadata(metadata),
      source: "python",
      trace_context: trace_context
    }
    
    Agent.update(__MODULE__, fn state ->
      %{state | logs: [log_entry | state.logs]}
    end)
  end

  def add_python_span(span_data) do
    Agent.update(__MODULE__, fn state ->
      %{state | spans: [span_data | state.spans]}
    end)
  end
  
  # OpenTelemetry exporter callback
  def export(traces, _opts) when is_list(traces) do
    # Collect spans from traces
    spans = Enum.flat_map(traces, fn trace ->
      collect_spans_from_trace(trace)
    end)
    
    Agent.update(__MODULE__, fn state ->
      %{state | spans: state.spans ++ spans}
    end)
    
    :ok
  end
  
  def export(_other, _opts), do: :ok
  
  # OpenTelemetry metrics exporter callback  
  def export_metrics(metrics, _opts) when is_list(metrics) do
    exported_metrics = Enum.map(metrics, fn metric ->
      convert_metric(metric)
    end)
    
    Agent.update(__MODULE__, fn state ->
      %{state | metrics: state.metrics ++ exported_metrics}
    end)
    
    :ok
  end
  
  def export_metrics(_other, _opts), do: :ok
  
  def export_json(output_dir \\ nil) do
    state = Agent.get(__MODULE__, fn s -> s end)
    total_time = System.monotonic_time(:millisecond) - state.start_time
    
    json_data = %{
      spans: state.spans,
      metrics: state.metrics,
      logs: Enum.reverse(state.logs),  # Reverse to show chronological order
      metadata: %{
        total_time_ms: total_time,
        start_time: state.start_time,
        end_time: System.monotonic_time(:millisecond),
        span_count: length(state.spans),
        metric_count: length(state.metrics),
        log_count: length(state.logs)
      }
    }
    
    json_string = Jason.encode!(json_data, pretty: true)
    
    if output_dir do
      # Save to file instead of printing
      File.mkdir_p!(output_dir)
      output_file = Path.join(output_dir, "opentelemetry_trace.json")
      File.write!(output_file, json_string)
      IO.puts("")
      IO.puts("[OK] OpenTelemetry trace saved to: #{output_file}")
      IO.puts("")
    else
      # Fallback: print if no output directory provided
      IO.puts("")
      IO.puts("=== OpenTelemetry JSON Export ===")
      IO.puts("")
      IO.puts(json_string)
      IO.puts("")
    end
    
    json_string
  end
  
  defp convert_metadata(metadata) when is_list(metadata) do
    # Convert list of tuples to a map for JSON encoding
    metadata
    |> Enum.map(fn
      {key, value} -> {convert_key(key), convert_value(value)}
      other -> {"_other", inspect(other)}
    end)
    |> Enum.into(%{})
  end
  defp convert_metadata(metadata) when is_map(metadata) do
    # Convert map keys and values to JSON-encodable format
    metadata
    |> Enum.map(fn {key, value} -> {convert_key(key), convert_value(value)} end)
    |> Enum.into(%{})
  end
  defp convert_metadata(metadata), do: %{"_raw" => inspect(metadata)}
  
  defp convert_key(key) when is_atom(key), do: Atom.to_string(key)
  defp convert_key(key) when is_binary(key), do: key
  defp convert_key(key), do: to_string(key)
  
  defp collect_spans_from_trace(trace) do
    # Extract spans from trace - structure depends on OpenTelemetry SDK
    # Try to extract span data from the trace tuple/record
    try do
      # OpenTelemetry traces are typically tuples or records
      # Try common patterns
      case trace do
        {_module, spans} when is_list(spans) ->
          Enum.map(spans, &convert_span/1)
        spans when is_list(spans) ->
          Enum.map(spans, &convert_span/1)
        span when is_tuple(span) ->
          [convert_span(span)]
        _ ->
          # Fallback: try to extract using OpenTelemetry API functions
          try do
            # Use OpenTelemetry API if available
            [%{trace: inspect(trace, limit: 100)}]
          rescue
            _ -> []
          end
      end
    rescue
      _ -> []
    end
  end
  
  defp convert_span(span) do
    # Convert OpenTelemetry span to JSON-serializable format
    try do
      # Try to extract span data - OpenTelemetry spans are typically records
      # Use pattern matching on common span structures
      case span do
        {:span, name, trace_id, span_id, parent_span_id, start_time, end_time, attributes, events, status, _links} ->
          %{
            name: name,
            trace_id: format_id(trace_id),
            span_id: format_id(span_id),
            parent_span_id: format_id(parent_span_id),
            start_time: start_time,
            end_time: end_time,
            duration_ns: if(end_time && start_time, do: end_time - start_time, else: nil),
            attributes: convert_attributes(attributes),
            events: convert_events(events),
            status: inspect(status)
          }
        tuple when is_tuple(tuple) and tuple_size(tuple) >= 2 ->
          # Generic tuple extraction
          %{
            name: inspect(:erlang.element(2, tuple)),
            trace_id: format_id(safe_element(tuple, 3)),
            span_id: format_id(safe_element(tuple, 4)),
            start_time: safe_element(tuple, 5),
            end_time: safe_element(tuple, 6),
            attributes: safe_convert_attributes(tuple, 7),
            raw: inspect(tuple, limit: 200)
          }
        _ ->
          %{span: inspect(span, limit: 200)}
      end
    rescue
      e ->
        %{span: inspect(span, limit: 200), error: inspect(e, limit: 100)}
    end
  end
  
  defp convert_metric(metric) do
    try do
      %{
        name: inspect(metric, limit: 200),
        data: inspect(metric, limit: 500)
      }
    rescue
      _ ->
        %{metric: inspect(metric, limit: 200)}
    end
  end
  
  defp convert_attributes(attrs) when is_list(attrs) do
    # Convert list of tuples to a map for JSON encoding
    attrs
    |> Enum.map(fn
      {key, value} -> {convert_key(key), convert_value(value)}
      other -> {"_other", inspect(other)}
    end)
    |> Enum.into(%{})
  end
  defp convert_attributes(attrs) when is_map(attrs) do
    # Convert map keys and values to JSON-encodable format
    attrs
    |> Enum.map(fn {key, value} -> {convert_key(key), convert_value(value)} end)
    |> Enum.into(%{})
  end
  defp convert_attributes(attrs), do: %{"_raw" => inspect(attrs)}
  
  defp convert_events(events) when is_list(events) do
    Enum.map(events, &inspect/1)
  end
  defp convert_events(events), do: inspect(events)
  
  defp convert_value(value) when is_atom(value), do: Atom.to_string(value)
  defp convert_value(value) when is_binary(value), do: value
  defp convert_value(value) when is_number(value), do: value
  defp convert_value(value) when is_boolean(value), do: value
  defp convert_value(value) when is_list(value) do
    # Recursively convert list items
    Enum.map(value, &convert_value/1)
  end
  defp convert_value(value) when is_map(value) do
    # Recursively convert map keys and values
    value
    |> Enum.map(fn {k, v} -> {convert_key(k), convert_value(v)} end)
    |> Enum.into(%{})
  end
  defp convert_value(value) when is_tuple(value) do
    # Convert tuple to list for JSON encoding
    value |> Tuple.to_list() |> Enum.map(&convert_value/1)
  end
  defp convert_value(value), do: inspect(value, limit: 100)
  
  defp format_id(id) when is_integer(id) do
    Integer.to_string(id, 16) |> String.pad_leading(16, "0")
  end
  defp format_id(nil), do: nil
  defp format_id(id), do: inspect(id)
  
  defp safe_element(tuple, index) do
    try do
      :erlang.element(index, tuple)
    rescue
      _ -> nil
    end
  end
  
  defp safe_convert_attributes(tuple, index) do
    try do
      convert_attributes(:erlang.element(index, tuple))
    rescue
      _ -> []
    end
  end
end

defmodule SpanCollector do
  @moduledoc """
  Wrapper around OpenTelemetry.Tracer API for convenience.
  Uses the official OpenTelemetry library from hex.pm instead of custom implementation.
  """
  require OpenTelemetry.Tracer

  def track_span(name, fun, attrs \\ []) do
    OpenTelemetry.Tracer.with_span name do
      # Set attributes on the current span
      Enum.each(attrs, fn {key, value} ->
        OpenTelemetry.Tracer.set_attribute(key, value)
      end)
      
      try do
        result = fun.()
        OpenTelemetry.Tracer.set_status(:ok)
        result
      rescue
        e ->
          # Handle SystemExit(0) as success, not error
          case e do
            %Pythonx.Error{type: type, value: value} ->
              try do
                # Use Pythonx.eval to access Python object attributes
                # Create a temporary Python namespace with the objects
                python_globals = %{"_elixir_type" => type, "_elixir_value" => value}
                {type_name, _} = Pythonx.eval("_elixir_type.__name__", python_globals)
                if type_name == "SystemExit" do
                  exit_code = try do
                    {code, _} = Pythonx.eval("_elixir_value.__int__()", python_globals)
                    code
                  rescue
                    _ ->
                      # Try to get exit code from args
                      try do
                        {args, _} = Pythonx.eval("_elixir_value.args", python_globals)
                        case args do
                          [code] when is_integer(code) -> code
                          _ -> 0
                        end
                      rescue
                        _ -> 0
                      end
                  end
                  
                  if exit_code == 0 do
                    # SystemExit(0) is success, not an error
                    OpenTelemetry.Tracer.set_status(:ok)
                    :ok
                  else
                    # Non-zero exit code is an error
                    OpenTelemetry.Tracer.record_exception(e, [])
                    OpenTelemetry.Tracer.set_status(:error, "SystemExit(#{exit_code})")
                    raise e
                  end
                else
                  # Other Python exceptions are errors
                  OpenTelemetry.Tracer.record_exception(e, [])
                  try do
                    OpenTelemetry.Tracer.set_status(:error, Exception.message(e))
                  rescue
                    _ -> OpenTelemetry.Tracer.set_status(:error, "Python exception: #{inspect(type_name)}")
                  end
                  raise e
                end
              rescue
                _ ->
                  # If we can't determine the exception type, treat as error
                  OpenTelemetry.Tracer.record_exception(e, [])
                  try do
                    OpenTelemetry.Tracer.set_status(:error, Exception.message(e))
                  rescue
                    _ -> OpenTelemetry.Tracer.set_status(:error, "Python exception")
                  end
                  raise e
              end
            _ ->
              # Non-Pythonx exceptions
              OpenTelemetry.Tracer.record_exception(e, [])
              try do
                OpenTelemetry.Tracer.set_status(:error, Exception.message(e))
              rescue
                _ -> OpenTelemetry.Tracer.set_status(:error, inspect(e))
              end
              raise e
          end
      end
    end
  end

  def add_span_attribute(key, value) do
    OpenTelemetry.Tracer.set_attribute(key, value)
  end

  def get_trace_context do
    # Get current trace context for propagation to Python
    # Returns TRACEPARENT format string for W3C trace context propagation
    try do
      # Get current span context
      span_ctx = OpenTelemetry.Tracer.current_span_ctx()
      
      if span_ctx do
        # Extract trace_id and span_id from span context
        # OpenTelemetry span context is a tuple/record with trace_id and span_id
        # Try to extract using pattern matching or element access
        trace_id = try do
          :erlang.element(2, span_ctx)  # trace_id is typically at position 2
        rescue
          _ -> nil
        end
        
        span_id = try do
          :erlang.element(3, span_ctx)  # span_id is typically at position 3
        rescue
          _ -> nil
        end
        
        if trace_id && span_id do
          # Format as TRACEPARENT (version-trace_id-span_id-trace_flags)
          # version = 00, trace_id = 32 hex chars, span_id = 16 hex chars, trace_flags = 2 hex chars
          trace_id_hex = trace_id |> Integer.to_string(16) |> String.pad_leading(32, "0")
          span_id_hex = span_id |> Integer.to_string(16) |> String.pad_leading(16, "0")
          trace_flags_hex = "01"  # Default: sampled
          
          "00-#{trace_id_hex}-#{span_id_hex}-#{trace_flags_hex}"
        else
          nil
        end
      else
        nil
      end
    rescue
      _ -> nil
    end
  end

  def record_metric(_name, _value, _unit \\ nil) do
    # Use OpenTelemetry Metrics API when available
    # For now, we'll need to use a metrics exporter or custom collection
    # This is a placeholder - proper implementation would use OpenTelemetry.Metrics
    :ok
  end

  def display_trace(output_dir \\ nil) do
    # Export JSON trace data if available
    try do
      if Process.whereis(OtelJsonExporter) do
        OtelJsonExporter.export_json(output_dir)
      else
        # Fallback if exporter not available
        IO.puts("")
        IO.puts("=== OpenTelemetry Trace ===")
        IO.puts("")
        IO.puts("Note: Use OpenTelemetry exporters (OTLP, Jaeger, etc.) to view traces.")
        IO.puts("For JSON export, configure opentelemetry_exporter_otlp or similar.")
        IO.puts("")
      end
    rescue
      e ->
        IO.puts("")
        IO.puts("=== OpenTelemetry Trace ===")
        IO.puts("")
        IO.puts("Error exporting trace: #{inspect(e)}")
        IO.puts("")
    end
  end
end

defmodule OtelLogger do
  @moduledoc """
  Logger wrapper for OpenTelemetry attributes.
  Logs are captured by OtelLogHandler and included in JSON export.
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
