#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Toy problem to test texture optimization assumptions

Mix.install([
  {:pythonx, "~> 0.4.7"},
])

Logger.configure(level: :info)

Pythonx.uv_init("""
[project]
name = "texture-toy-test"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "torch==2.4.0",
  "torchvision==0.19.0",
  "numpy",
  "tqdm",
]
""")

IO.puts("=== Running Texture Optimization Toy Problem ===")

Pythonx.eval("""
import sys
sys.path.insert(0, '/mnt/c/Users/ernest.lee/Desktop/livebook-nx')

from test_texture_optimization import test_texture_optimization

if __name__ == "__main__":
    test_texture_optimization()
""", %{})

