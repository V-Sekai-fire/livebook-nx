#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# KVoiceWalk Voice Cloning Script
# Generate Kokoro voice tensors by cloning target voices using random walk algorithm
# Repository: https://github.com/RobViren/kvoicewalk
#
# Usage:
#   elixir kvoicewalk_generation.exs --target-audio <audio.wav> --target-text "<text>" [options]
#
# Options:
#   --target-audio <path>          Target audio file (must be 24000 Hz mono WAV)
#   --target-text <text>           Text spoken in the target audio (or path to .txt file)
#   --other-text <text>            Text for self-similarity comparison (default: provided default)
#   --starting-voice <path>        Path to starting voice tensor (.pt file)
#   --voice-folder <path>          Path to folder containing voice tensors (default: ./voices)
#   --interpolate-start            Run interpolation search before random walk
#   --transcribe-start             Transcribe target audio to text automatically
#   --population-limit <int>       Limit number of voices in random walk (default: 10)
#   --step-limit <int>             Limit number of random walk steps (default: 10000)
#   --output-name <name>           Output voice tensor filename (default: generated_voice.pt)
#   --help, -h                     Show this help message

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
  {:req, "~> 0.5.0"}
])

# Suppress debug logs from Req to avoid showing long URLs
Logger.configure(level: :info)

# Initialize Python environment with required dependencies
# KVoiceWalk uses Kokoro, Resemblyzer, and various audio processing libraries
# All dependencies managed by uv (no pip)
Pythonx.uv_init("""
[project]
name = "kvoicewalk-generation"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "kokoro>=0.9.4",
  "resemblyzer",
  "soundfile",
  "torch",
  "torchaudio",
  "numpy<2.0",
  "scipy",
  "librosa",
  "faster-whisper",
  "pillow",
  "huggingface-hub",
  "tokenizers>=0.20.0",
  "transformers>=4.30.0",
  "misaki[en]",
]

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchaudio = { index = "pytorch-cu118" }

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
""")

# Parse command-line arguments
defmodule ArgsParser do
  def show_help do
    IO.puts("""
    KVoiceWalk Voice Cloning Script
    Generate Kokoro voice tensors by cloning target voices using random walk algorithm
    Repository: https://github.com/RobViren/kvoicewalk

    Usage:
      elixir kvoicewalk_generation.exs --target-audio <audio.wav> --target-text "<text>" [options]

    Options:
      --target-audio <path>          Target audio file (must be 24000 Hz mono WAV)
      --target-text <text>           Text spoken in the target audio (or path to .txt file)
      --other-text <text>            Text for self-similarity comparison (default: provided default)
      --starting-voice <path>        Path to starting voice tensor (.pt file)
      --voice-folder <path>          Path to folder containing voice tensors (default: ./voices)
      --interpolate-start            Run interpolation search before random walk
      --transcribe-start             Transcribe target audio to text automatically
      --population-limit <int>       Limit number of voices in random walk (default: 10)
      --step-limit <int>             Limit number of random walk steps (default: 10000)
      --output-name <name>           Output voice tensor filename (default: generated_voice.pt)
      --help, -h                     Show this help message

    Example:
      elixir kvoicewalk_generation.exs --target-audio target.wav --target-text "Hello world"
      elixir kvoicewalk_generation.exs --target-audio target.wav --target-text transcript.txt --interpolate-start
      elixir kvoicewalk_generation.exs --target-audio target.wav --target-text "Text" --transcribe-start
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        target_audio: :string,
        target_text: :string,
        other_text: :string,
        starting_voice: :string,
        voice_folder: :string,
        interpolate_start: :boolean,
        transcribe_start: :boolean,
        population_limit: :integer,
        step_limit: :integer,
        output_name: :string,
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

    target_audio = Keyword.get(opts, :target_audio)
    target_text = Keyword.get(opts, :target_text)

    if !target_audio do
      IO.puts("""
      Error: Target audio file is required.

      Usage:
        elixir kvoicewalk_generation.exs --target-audio <audio.wav> --target-text "<text>" [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    if !target_text do
      IO.puts("""
      Error: Target text is required.

      Usage:
        elixir kvoicewalk_generation.exs --target-audio <audio.wav> --target-text "<text>" [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    # Check if target audio file exists
    if !File.exists?(target_audio) do
      IO.puts("Error: Target audio file not found: #{target_audio}")
      System.halt(1)
    end

    # Check if target_text is a file path
    target_text_content = if File.exists?(target_text) do
      # It's a file path, read it
      File.read!(target_text)
    else
      # It's text content
      target_text
    end

    # Validate numeric parameters
    population_limit = Keyword.get(opts, :population_limit, 10)
    if population_limit < 1 do
      IO.puts("Error: population_limit must be at least 1")
      System.halt(1)
    end

    step_limit = Keyword.get(opts, :step_limit, 10000)
    if step_limit < 1 do
      IO.puts("Error: step_limit must be at least 1")
      System.halt(1)
    end

    config = %{
      target_audio: target_audio,
      target_text: target_text_content,
      other_text: Keyword.get(opts, :other_text),
      starting_voice: Keyword.get(opts, :starting_voice),
      voice_folder: Keyword.get(opts, :voice_folder, "./voices"),
      interpolate_start: Keyword.get(opts, :interpolate_start, false),
      transcribe_start: Keyword.get(opts, :transcribe_start, false),
      population_limit: population_limit,
      step_limit: step_limit,
      output_name: Keyword.get(opts, :output_name, "generated_voice.pt")
    }

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== KVoiceWalk Voice Cloning ===
Target Audio: #{config.target_audio}
Target Text: #{String.slice(config.target_text, 0, 100)}#{if String.length(config.target_text) > 100, do: "...", else: ""}
Voice Folder: #{config.voice_folder}
Interpolate Start: #{config.interpolate_start}
Transcribe Start: #{config.transcribe_start}
Population Limit: #{config.population_limit}
Step Limit: #{config.step_limit}
Output Name: #{config.output_name}
""")

# Add paths to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  kvoicewalk_path: Path.join([base_dir, "thirdparty", "KVoiceWalk"]),
  workspace_root: base_dir
})

# Save config to JSON for Python to read
config_json = Jason.encode!(config_with_paths)
File.write!("config.json", config_json)

# Clone KVoiceWalk repository if it doesn't exist
IO.puts("\n=== Step 1: Setup KVoiceWalk Repository ===")
kvoicewalk_path = config_with_paths.kvoicewalk_path

if !File.exists?(kvoicewalk_path) do
  IO.puts("KVoiceWalk repository not found. Cloning from GitHub...")
  IO.puts("Repository: https://github.com/RobViren/kvoicewalk")

  # Create thirdparty directory if it doesn't exist
  File.mkdir_p!(Path.dirname(kvoicewalk_path))

  # Clone the repository
  case System.cmd("git", ["clone", "https://github.com/RobViren/kvoicewalk.git", kvoicewalk_path], stderr_to_stdout: true) do
    {output, 0} ->
      IO.puts("[OK] KVoiceWalk repository cloned successfully")
    {output, exit_code} ->
      IO.puts("[ERROR] Failed to clone KVoiceWalk repository")
      IO.puts(output)
      System.halt(1)
  end
else
  IO.puts("✓ KVoiceWalk repository found at: #{kvoicewalk_path}")
end

# Import libraries and run KVoiceWalk
{_, _python_globals} = Pythonx.eval("""
import json
import sys
import os
import subprocess
from pathlib import Path

# Get configuration from JSON file
with open("config.json", 'r', encoding='utf-8') as f:
    config = json.load(f)

target_audio = config.get('target_audio')
target_text = config.get('target_text')
other_text = config.get('other_text')
starting_voice = config.get('starting_voice')
voice_folder = config.get('voice_folder', './voices')
interpolate_start = config.get('interpolate_start', False)
transcribe_start = config.get('transcribe_start', False)
population_limit = config.get('population_limit', 10)
step_limit = config.get('step_limit', 10000)
output_name = config.get('output_name', 'generated_voice.pt')
kvoicewalk_path = config.get('kvoicewalk_path')
workspace_root = config.get('workspace_root')

# Resolve paths
target_audio = str(Path(target_audio).resolve())
kvoicewalk_path = str(Path(kvoicewalk_path).resolve())
workspace_root = str(Path(workspace_root).resolve())

# Check if KVoiceWalk exists
if not Path(kvoicewalk_path).exists():
    raise FileNotFoundError(f"KVoiceWalk repository not found at: {kvoicewalk_path}")

# Add KVoiceWalk to Python path
if str(kvoicewalk_path) not in sys.path:
    sys.path.insert(0, str(kvoicewalk_path))

# Change to KVoiceWalk directory
original_cwd = os.getcwd()
os.chdir(kvoicewalk_path)

# Create output directory
output_dir = Path(workspace_root) / "output"
output_dir.mkdir(exist_ok=True, parents=True)

# Create timestamped folder for this run
import time
tag = time.strftime("%Y%m%d_%H_%M_%S")
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True, parents=True)

print(f"\\n=== Step 2: Run KVoiceWalk Voice Cloning ===")
print(f"Target Audio: {target_audio}")
print(f"Target Text: {target_text[:100]}{'...' if len(target_text) > 100 else ''}")
print(f"Output Directory: {export_dir}")

# Build command arguments for KVoiceWalk main.py
cmd_args = [
    sys.executable, "main.py",
    "--target_audio", target_audio,
    "--target_text", target_text,
    "--population_limit", str(population_limit),
    "--step_limit", str(step_limit),
    "--output_name", output_name,
    "--voice_folder", voice_folder,
]

# Add optional arguments
if other_text:
    cmd_args.extend(["--other_text", other_text])

if starting_voice:
    cmd_args.extend(["--starting_voice", starting_voice])

if interpolate_start:
    cmd_args.append("--interpolate_start")

if transcribe_start:
    cmd_args.append("--transcribe_start")

print(f"\\nRunning KVoiceWalk with command:")
print(f"  {' '.join(cmd_args)}")
print(f"\\nThis may take a while depending on step_limit and population_limit...")

# Run KVoiceWalk
generated_voice = None
try:
    result = subprocess.run(
        cmd_args,
        cwd=kvoicewalk_path,
        check=True,
        capture_output=False  # Show output in real-time
    )

    print("\\n[OK] KVoiceWalk completed successfully")

    # Find the generated voice file
    # KVoiceWalk saves to 'out' directory by default, or uses --output_name
    possible_locations = [
        Path(kvoicewalk_path) / "out" / output_name,
        Path(kvoicewalk_path) / output_name,
        Path(kvoicewalk_path) / "out" / "out.pt",  # Default output name
    ]

    for loc in possible_locations:
        if loc.exists():
            generated_voice = loc
            break

    if generated_voice:
        # Copy to output directory
        output_voice = export_dir / output_name
        import shutil
        shutil.copy2(generated_voice, output_voice)
        print(f"\\n[OK] Generated voice tensor saved to: {output_voice}")
        print(f"\\nYou can use this voice tensor with kokoro_tts_generation.exs:")
        print(f"  elixir kokoro_tts_generation.exs \"<text>\" --voice-file {output_voice}")
    else:
        print(f"\\n[WARN] Could not find generated voice tensor")
        print(f"  Searched in: {possible_locations}")
        print(f"  Check the KVoiceWalk output directory manually")

except subprocess.CalledProcessError as e:
    print(f"\\n[ERROR] KVoiceWalk failed with exit code {e.returncode}")
    raise
except Exception as e:
    print(f"\\n[ERROR] Error running KVoiceWalk: {e}")
    import traceback
    traceback.print_exc()
    raise
finally:
    # Restore original working directory
    os.chdir(original_cwd)

print("\\n=== Complete ===")
print(f"Voice cloning completed!")
if generated_voice:
    print(f"\\nOutput files in {export_dir.name}/:")
    print(f"  - {output_name} (Generated voice tensor)")
""", %{})

IO.puts("\n=== Complete ===")
IO.puts("Voice cloning completed successfully!")
