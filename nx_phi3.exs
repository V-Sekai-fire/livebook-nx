#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# NX PHI-3 Text Generation Script
# Uses Bumblebee to load GPT-2 model and generate text
#
# Usage:
#   elixir nx_phi3.exs [text] [options]
#
# Options:
#   --model, -m <model>        Hugging Face model name (default: "gpt2")
#   --max-tokens, -t <int>     Maximum number of new tokens to generate (default: 10)
#   --help, -h                 Show help message
#
# Examples:
#   elixir nx_phi3.exs "Yesterday, I was reading a book and"
#   elixir nx_phi3.exs "Hello world" --max-tokens 20 --model "gpt2"

Mix.install([
  {:bumblebee, git: "https://github.com/elixir-nx/bumblebee", tag: "main"},
  {:nx, "~> 0.7.0"},
  {:exla, "~> 0.7.1"},
], config: [nx: [default_backend: EXLA.Backend]])

require Logger

Logger.configure(level: :info)

# Parse command-line arguments
defmodule ArgsParser do
  def show_help do
    IO.puts("""
    NX PHI-3 Text Generation Script
    
    Usage:
      elixir nx_phi3.exs [text] [options]
    
    Arguments:
      <text>                    Input text for generation (optional, defaults to example text)
    
    Options:
      --model, -m <model>       Hugging Face model name (default: "gpt2")
      --max-tokens, -t <int>     Maximum number of new tokens to generate (default: 10)
      --help, -h                 Show help message
    
    Examples:
      elixir nx_phi3.exs "Yesterday, I was reading a book and"
      elixir nx_phi3.exs "Hello world" --max-tokens 20 --model "gpt2"
    
    Note: For CUDA support, ensure the following environment variables are set:
      XLA_BUILD=true
      XLA_TARGET=cuda
      XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
      CUDA_HOME=/usr/local/cuda
      PATH="/usr/local/cuda/bin:$PATH"
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        model: :string,
        max_tokens: :integer,
        help: :boolean
      ],
      aliases: [
        m: :model,
        t: :max_tokens,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    model = Keyword.get(opts, :model, "gpt2")
    max_tokens = Keyword.get(opts, :max_tokens, 10)
    text = case args do
      [text | _] -> text
      [] -> "Yesterday, I was reading a book and"
    end

    %{
      model: model,
      max_tokens: max_tokens,
      text: text
    }
  end
end

# Main execution
config = ArgsParser.parse(System.argv())

IO.puts("""
=== NX PHI-3 Text Generation ===
Model: #{config.model}
Input Text: #{config.text}
Max Tokens: #{config.max_tokens}
""")

IO.puts("[INFO] Loading model and tokenizer...")
{:ok, model} = Bumblebee.load_model({:hf, config.model})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, config.model})

IO.puts("[INFO] Creating text generation serving...")
serving = Bumblebee.Text.generation(model, tokenizer, max_new_tokens: config.max_tokens)

IO.puts("[INFO] Generating text...")
result = Nx.Serving.run(serving, config.text)

IO.puts("\n=== Generated Text ===")
# Bumblebee returns a map with :text key
case result do
  %{text: text} -> IO.puts(text)
  text when is_binary(text) -> IO.puts(text)
  other -> IO.inspect(other, label: "Result")
end
IO.puts("")

