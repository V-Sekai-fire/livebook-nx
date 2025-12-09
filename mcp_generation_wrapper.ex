#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Generic MCP Generation Wrapper
# Provides a reusable ExMCP API wrapper for any Tier 1 GenServer-based generation service
#
# Usage:
#   This module provides a generic MCP handler that wraps any GenServer-based generation
#   service. Configure it with:
#   - generator_server: The GenServer module (must implement setup/0 and generate/7)
#   - tool_definitions: List of MCP tool definitions
#   - response_formatter: Function to format generation results as MCP content

defmodule MCPGenerationWrapper do
  @moduledoc """
  Generic MCP wrapper for GenServer-based generation services.
  
  This module provides a reusable ExMCP handler that can wrap any Tier 1 GenServer
  that implements the standard generation interface:
  - setup/0 or setup/1: Initialize the generator
  - generate/7: Generate content with parameters
  
  Configuration is done via application environment or function parameters.
  """
  
  @doc """
  Creates a configured MCP handler module for a specific generation service.
  
  ## Options
  
  - `:generator_server` - The GenServer module that handles generation (required)
  - `:server_name` - Name for the MCP server (default: "generation-server")
  - `:server_version` - Version string (default: "1.0.0")
  - `:tool_definitions` - List of MCP tool definitions (required)
  - `:response_formatter` - Function to format generation results (optional)
  - `:mcp_port` - Port for the HTTP server (default: 4000)
  - `:static_files_path` - Path to serve static files from (optional)
  - `:static_files_at` - URL path prefix for static files (default: "/images")
  
  ## Example
  
      defmodule MyImageMCPHandler do
        use MCPGenerationWrapper,
          generator_server: MyImageGenerator.Server,
          server_name: "my-image-generation",
          server_version: "1.0.0",
          tool_definitions: [
            %{
              "name" => "generate_image",
              "description" => "Generate images from text prompts",
              "inputSchema" => %{
                "type" => "object",
                "properties" => %{
                  "prompt" => %{"type" => "string", "description" => "Text prompt"}
                },
                "required" => ["prompt"]
              }
            }
          ],
          response_formatter: &MyImageMCPHandler.format_response/2
      end
  """
  defmacro __using__(opts) do
    generator_server = Keyword.fetch!(opts, :generator_server)
    server_name = Keyword.get(opts, :server_name, "generation-server")
    server_version = Keyword.get(opts, :server_version, "1.0.0")
    tool_definitions = Keyword.fetch!(opts, :tool_definitions)
    response_formatter = Keyword.get(opts, :response_formatter)
    mcp_port = Keyword.get(opts, :mcp_port, 4000)
    static_files_path = Keyword.get(opts, :static_files_path)
    static_files_at = Keyword.get(opts, :static_files_at, "/images")

    quote do
      use ExMCP.Server.Handler
      require OpenTelemetry.Tracer

      @generator_server unquote(generator_server)
      @server_name unquote(server_name)
      @server_version unquote(server_version)
      @tool_definitions unquote(tool_definitions)
      @response_formatter unquote(response_formatter)
      @mcp_port unquote(mcp_port)
      @static_files_path unquote(static_files_path)
      @static_files_at unquote(static_files_at)

      @impl true
      def init(_args), do: {:ok, %{}}

      def handle_call({:initialize, params}, _from, state) do
        case handle_initialize(params, state) do
          {:ok, result, new_state} -> {:reply, {:ok, result}, new_state}
          {:error, reason, new_state} -> {:reply, {:error, reason}, new_state}
        end
      end

      def handle_call({:list_tools, params}, _from, state) do
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

      def handle_call({:call_tool, tool_name, args}, _from, state) when is_binary(tool_name) do
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
          result = %{
            "protocolVersion" => "2024-11-05",
            "serverInfo" => %{
              "name" => @server_name,
              "version" => @server_version
            },
            "capabilities" => %{"tools" => %{}}
          }
          {:ok, result, state}
        end
      end

      @impl true
      def handle_list_tools(_params, state) do
        OpenTelemetry.Tracer.with_span "mcp.list_tools" do
          {:ok, %{"tools" => @tool_definitions}, state}
        end
      end

      @impl true
      def handle_call_tool(tool_name, args, state) do
        OpenTelemetry.Tracer.with_span "mcp.tool.#{tool_name}" do
          # Delegate to the specific tool handler
          case handle_tool(tool_name, args, state) do
            {:ok, result, new_state} ->
              # Apply response formatter if provided
              content = if @response_formatter do
                @response_formatter.(result, args)
              else
                default_format_response(result, args)
              end
              {:ok, content, new_state}
            {:error, reason, new_state} ->
              {:error, reason, new_state}
          end
        end
      end

      # Default tool handler - should be overridden by specific implementations
      def handle_tool(tool_name, _args, state) do
        {:error, "Unknown tool: #{tool_name}", state}
      end
      
      # Make handle_tool overridable
      defoverridable [handle_tool: 3]

      # Default response formatter
      defp default_format_response({:ok, output_path}, _args) when is_binary(output_path) do
        # Try to convert file path to HTTP URL if static files are configured
        if @static_files_path do
          output_dir = Path.expand(@static_files_path)
          abs_output_path = Path.expand(output_path)
          
          if String.starts_with?(abs_output_path, output_dir) do
            relative_path = Path.relative_to(abs_output_path, output_dir) |> String.replace("\\", "/")
            image_url = "http://localhost:#{@mcp_port}#{@static_files_at}/#{relative_path}"
            
            [
              %{"type" => "text", "text" => "Generated successfully. View at: #{image_url}"},
              %{"type" => "resource", "uri" => image_url}
            ]
          else
            [
              %{"type" => "text", "text" => "Generated successfully. Output: #{output_path}"},
              %{"type" => "resource", "uri" => "file://#{output_path}"}
            ]
          end
        else
          [
            %{"type" => "text", "text" => "Generated successfully. Output: #{output_path}"},
            %{"type" => "resource", "uri" => "file://#{output_path}"}
          ]
        end
      end

      defp default_format_response({:error, reason}, _args) do
        [
          %{"type" => "text", "text" => "Generation failed: #{inspect(reason)}"}
        ]
      end

      defp default_format_response(other, _args) do
        [
          %{"type" => "text", "text" => "Result: #{inspect(other)}"}
        ]
      end

    end
  end
end

