#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# NX Examples Script
# Demonstrates basic tensor operations using NX (Numerical Elixir)
#
# Usage:
#   elixir nx.exs [options]
#
# Options:
#   --help, -h                 Show help message
#
# Examples:
#   elixir nx.exs

Mix.install([
  {:kino_bumblebee, "~> 0.2.1"},
  {:exla, "~> 0.5.0"}
], config: [nx: [default_backend: EXLA.Backend]])

require Logger

Logger.configure(level: :info)

# Parse command-line arguments
defmodule ArgsParser do
  def show_help do
    IO.puts("""
    NX Examples Script
    
    Demonstrates basic tensor operations using NX (Numerical Elixir).
    
    Usage:
      elixir nx.exs [options]
    
    Options:
      --help, -h                 Show help message
    
    Examples:
      elixir nx.exs
    
    Note: For CUDA support, ensure the following environment variable is set:
      EXLA_TARGET=cuda12
    """)
  end

  def parse(args) do
    {opts, _, _} = OptionParser.parse(args,
      switches: [
        help: :boolean
      ],
      aliases: [
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    %{}
  end
end

# Main execution
_ = ArgsParser.parse(System.argv())

IO.puts("=== NX Examples ===")
IO.puts("")

IO.puts("[INFO] Creating tensors...")
tensor1 = Nx.tensor([[1, 2], [3, 4]])
tensor2 = Nx.tensor([[5, 6], [7, 8]])

IO.puts("Tensor 1:")
IO.inspect(tensor1)
IO.puts("")
IO.puts("Tensor 2:")
IO.inspect(tensor2)
IO.puts("")

IO.puts("[INFO] Performing operations...")
sum = Nx.add(tensor1, tensor2)
product = Nx.multiply(tensor1, tensor2)
difference = Nx.subtract(tensor1, tensor2)

IO.puts("Sum (tensor1 + tensor2):")
IO.inspect(sum)
IO.puts("")

IO.puts("Product (tensor1 * tensor2):")
IO.inspect(product)
IO.puts("")

IO.puts("Difference (tensor1 - tensor2):")
IO.inspect(difference)
IO.puts("")

IO.puts("=== All Results ===")
IO.inspect({sum, product, difference})
IO.puts("")

