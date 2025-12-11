#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# Kokoro TTS Generation Script
# Generate speech from text using Kokoro-82M (Text-to-Speech)
# Model: Kokoro-82M by hexgrad (82M parameters, Apache-2.0 License)
# Repository: https://github.com/hexgrad/kokoro
# Hugging Face: https://huggingface.co/hexgrad/Kokoro-82M
#
# Usage:
#   elixir kokoro_tts_generation.exs "<text>" [options]
#   elixir kokoro_tts_generation.exs --input-file <file> [options]
#   echo "<text>" | elixir kokoro_tts_generation.exs [options]
#
# Options:
#   --input-file, -i <path>         Read text from file (alternative to command-line text)
#   --lang-code "a"                  Language code: a (American English), b (British English), e (Spanish), f (French), h (Hindi), i (Italian), j (Japanese), p (Portuguese), z (Chinese) (default: "a")
#   --voice "af_heart"               Voice to use (default: "af_heart")
#   --voice-file <path>               Path to voice tensor file (.pt) - overrides --voice if provided
#   --speed <float>                  Speech speed multiplier (default: 1.0)
#   --split-pattern <regex>           Pattern to split text into segments (default: "\\n+")
#   --output-format "wav"             Output format: wav, mp3, flac (default: "wav")
#   --sample-rate <int>               Audio sample rate in Hz (default: 24000)

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

# Initialize OpenTelemetry
OtelSetup.configure()

# Initialize Python environment with required dependencies
# Kokoro uses kokoro package and misaki for G2P
# All dependencies managed by uv (no pip)
Pythonx.uv_init("""
[project]
name = "kokoro-tts-generation"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  "kokoro>=0.9.4",
  "soundfile",
  "torch",
  "torchaudio",
  "numpy<2.0",  # Pin to <2.0 for compatibility with thinc (spaCy dependency)
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
  "huggingface-hub",
  # Use newer tokenizers with pre-built wheels to avoid Rust compilation on Windows
  # This may require updating transformers to a compatible version
  "tokenizers>=0.20.0",
  "transformers>=4.30.0",
  # misaki for G2P - install based on language
  # For English: misaki[en]
  # For Japanese: misaki[ja]
  # For Chinese: misaki[zh]
  # For all: misaki[all]
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
    Kokoro TTS Generation Script
    Generate speech from text using Kokoro-82M (Text-to-Speech)
    Model: Kokoro-82M by hexgrad (82M parameters, Apache-2.0 License)
    Repository: https://github.com/hexgrad/kokoro
    Hugging Face: https://huggingface.co/hexgrad/Kokoro-82M

    Usage:
      elixir kokoro_tts_generation.exs "<text>" [options]
      elixir kokoro_tts_generation.exs --input-file <file> [options]
      echo "<text>" | elixir kokoro_tts_generation.exs [options]

    Options:
      --input-file, -i <path>         Read text from file (alternative to command-line text)
      --lang-code, -l "a"              Language code: a (American English), b (British English), e (Spanish), f (French), h (Hindi), i (Italian), j (Japanese), p (Portuguese), z (Chinese) (default: "a")
      --voice, -v "af_heart"          Voice to use (default: "af_heart")
      --voice-file <path>              Path to voice tensor file (.pt) - overrides --voice if provided
      --speed, -s <float>              Speech speed multiplier (default: 1.0)
      --split-pattern <regex>          Pattern to split text into segments (default: "\\n+")
      --output-format, -f "wav"        Output format: wav, mp3, flac (default: "wav")
      --sample-rate, -r <int>          Audio sample rate in Hz (default: 24000)
      --help, -h                       Show this help message

    Example:
      elixir kokoro_tts_generation.exs "Hello, world!" --lang-code a
      elixir kokoro_tts_generation.exs --input-file text.txt --voice af_heart
      echo "Hello" | elixir kokoro_tts_generation.exs
    """)
  end

  def parse(args) do
    {opts, args, _} = OptionParser.parse(args,
      switches: [
        lang_code: :string,
        voice: :string,
        voice_file: :string,
        speed: :float,
        split_pattern: :string,
        output_format: :string,
        sample_rate: :integer,
        input_file: :string,
        help: :boolean
      ],
      aliases: [
        l: :lang_code,
        v: :voice,
        s: :speed,
        f: :output_format,
        r: :sample_rate,
        i: :input_file,
        h: :help
      ]
    )

    if Keyword.get(opts, :help, false) do
      show_help()
      System.halt(0)
    end

    # Get text from various sources (priority: input_file > command-line arg > stdin)
    # Note: stdin reading is attempted only if no other input is provided
    text = cond do
      input_file = Keyword.get(opts, :input_file) ->
        if File.exists?(input_file) do
          File.read!(input_file)
        else
          IO.puts("Error: Input file not found: #{input_file}")
          System.halt(1)
        end
      arg_text = List.first(args) ->
        arg_text
      true ->
        # No command-line input provided - try stdin only if it's piped (not a TTY)
        # On Windows, reading from stdin when it's a TTY (console) will block indefinitely
        # So we only read if stdin is not a TTY (meaning it's piped)
        try do
          # Check if stdin is a TTY - if it is, don't read (would block)
          is_tty = case :io.getopts(:standard_io) do
            {:ok, opts} -> Keyword.get(opts, :tty, true)
            _ -> true  # Assume TTY if we can't determine
          end

          if is_tty do
            # Stdin is a TTY (console), don't read to avoid blocking
            nil
          else
            # Stdin is not a TTY (piped input), safe to read
            case IO.binread(:stdio, :eof) do
              {:error, _} -> nil
              :eof -> nil
              data when is_binary(data) and byte_size(data) > 0 -> data
              _ -> nil
            end
          end
        rescue
          # If any error occurs, return nil (will show error message)
          _ -> nil
        catch
          # Catch exit signals
          :exit, _ -> nil
        end
    end

    if !text || String.trim(text) == "" do
      IO.puts("""
      Error: Text input is required.

      Usage:
        elixir kokoro_tts_generation.exs "<text>" [options]
        elixir kokoro_tts_generation.exs --input-file <file> [options]
        echo "<text>" | elixir kokoro_tts_generation.exs [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    # Trim the text
    text = String.trim(text)

    lang_code = Keyword.get(opts, :lang_code, "a")
    valid_lang_codes = ["a", "b", "e", "f", "h", "i", "j", "p", "z"]
    if lang_code not in valid_lang_codes do
      IO.puts("Error: Invalid lang_code '#{lang_code}'. Valid codes: #{Enum.join(valid_lang_codes, ", ")}")
      System.halt(1)
    end

    voice_file = Keyword.get(opts, :voice_file)
    if voice_file && !File.exists?(voice_file) do
      IO.puts("Error: Voice file not found: #{voice_file}")
      System.halt(1)
    end

    output_format = Keyword.get(opts, :output_format, "wav")
    valid_formats = ["wav", "mp3", "flac"]
    if output_format not in valid_formats do
      IO.puts("Error: Invalid output format. Must be one of: #{Enum.join(valid_formats, ", ")}")
      System.halt(1)
    end

    sample_rate = Keyword.get(opts, :sample_rate, 24000)
    if sample_rate < 8000 or sample_rate > 48000 do
      IO.puts("Error: Sample rate must be between 8000 and 48000 Hz")
      System.halt(1)
    end

    speed = Keyword.get(opts, :speed, 1.0)
    if speed < 0.5 or speed > 2.0 do
      IO.puts("Error: Speed must be between 0.5 and 2.0")
      System.halt(1)
    end

    config = %{
      text: text,
      lang_code: lang_code,
      voice: Keyword.get(opts, :voice, "af_heart"),
      voice_file: voice_file,
      speed: speed,
      split_pattern: Keyword.get(opts, :split_pattern, "\\n+"),
      output_format: output_format,
      sample_rate: sample_rate
    }

    config
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

IO.puts("""
=== Kokoro TTS Generation ===
Text: #{String.slice(config.text, 0, 100)}#{if String.length(config.text) > 100, do: "...", else: ""}
Language Code: #{config.lang_code}
Voice: #{config.voice}
Voice File: #{config.voice_file || "N/A"}
Speed: #{config.speed}
Split Pattern: #{config.split_pattern}
Output Format: #{config.output_format}
Sample Rate: #{config.sample_rate} Hz
""")

# Add weights directory to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  kokoro_weights_dir: Path.join([base_dir, "pretrained_weights", "Kokoro-82M"])
})

# Save config to JSON for Python to read (use temp file to avoid conflicts)
config_json = Jason.encode!(config_with_paths)
# Use cross-platform temp directory
tmp_dir = System.tmp_dir!()
File.mkdir_p!(tmp_dir)
config_file = Path.join(tmp_dir, "kokoro_tts_config_#{System.system_time(:millisecond)}.json")
File.write!(config_file, config_json)
config_file_normalized = String.replace(config_file, "\\", "/")

# Download models using Elixir-native approach
IO.puts("\n=== Step 2: Download Pretrained Weights ===")
IO.puts("Downloading Kokoro-82M models from Hugging Face...")

base_dir = Path.expand(".")
kokoro_weights_dir = Path.join([base_dir, "pretrained_weights", "Kokoro-82M"])

IO.puts("Using weights directory: #{kokoro_weights_dir}")

# Kokoro-82M repository on Hugging Face
repo_id = "hexgrad/Kokoro-82M"

# Download Kokoro-82M weights (using OpenTelemetry integration)
case HuggingFaceDownloader.download_repo(repo_id, kokoro_weights_dir, "Kokoro-82M", true) do
  {:ok, _} -> :ok
  {:error, _} ->
    IO.puts("[WARN] Kokoro-82M download had errors, but continuing...")
    IO.puts("[INFO] If the model is not on Hugging Face, you may need to download it manually")
end

# spaCy model is installed via dependencies (line 44-45), no need to download separately
IO.puts("\n=== spaCy Model ===")
IO.puts("spaCy model 'en_core_web_sm' will be installed automatically via uv dependencies")

# Import libraries and process using Kokoro
SpanCollector.track_span("kokoro_tts.generation", fn ->
try do
{_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
import re
import time
import warnings
import subprocess
from pathlib import Path

# CRITICAL: Patch subprocess.run FIRST, before any imports that might use it
# This prevents misaki/spacy from hanging when they spawn subprocesses
_original_subprocess_run = subprocess.run
def _patched_subprocess_run(*args, **kwargs):
    # Always redirect stdin to prevent hanging
    if 'stdin' not in kwargs:
        kwargs['stdin'] = subprocess.DEVNULL
    # Also redirect stdout/stderr if not specified to avoid blocking
    if 'stdout' not in kwargs:
        kwargs['stdout'] = subprocess.PIPE
    if 'stderr' not in kwargs:
        kwargs['stderr'] = subprocess.PIPE
    return _original_subprocess_run(*args, **kwargs)
subprocess.run = _patched_subprocess_run

# Suppress warnings to prevent StopIteration issues with warning system
warnings.filterwarnings('ignore')

# Set environment variables to make HuggingFace non-interactive
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Ensure stdin is not blocking - redirect to devnull
# This must be done BEFORE any imports that might spawn subprocesses
if hasattr(os, 'devnull'):
    try:
        devnull = open(os.devnull, 'r')
        sys.stdin = devnull
    except:
        pass

# Now import libraries - subprocess.run is already patched
import torch
import soundfile as sf
import numpy as np

# spaCy model is installed via uv dependencies, just verify it's available
# Import spacy first to get access to cli module
import spacy

# Check if model exists (it should be installed via dependencies)
try:
    nlp = spacy.load('en_core_web_sm')
    print("[OK] spaCy model 'en_core_web_sm' is available")
    sys.stdout.flush()
except OSError:
    # Model not found - this shouldn't happen if dependencies installed correctly
    print("[WARN] spaCy model 'en_core_web_sm' not found")
    print("[INFO] It should have been installed via uv dependencies")
    print("[INFO] Will use patched download as fallback")
    sys.stdout.flush()

# Patch spacy.cli.download to handle misaki's download attempts
_original_spacy_download = spacy.cli.download
def _patched_spacy_download(name, **kwargs):
    # Check if model already exists
    try:
        spacy.load(name)
        print(f"[OK] spaCy model '{name}' already available, skipping download")
        sys.stdout.flush()
        return 0  # Success
    except OSError:
        # Model doesn't exist - try to download using patched subprocess
        print(f"[INFO] spaCy model '{name}' not found, attempting download...")
        sys.stdout.flush()
        result = subprocess.run(
            [sys.executable, '-m', 'spacy', 'download', name],
            check=False,
            timeout=600,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            print(f"[OK] spaCy model '{name}' downloaded successfully")
            sys.stdout.flush()
            return 0
        else:
            print(f"[WARN] spaCy model '{name}' download failed (code {result.returncode})")
            sys.stdout.flush()
            return 1  # Indicate failure, but don't crash
spacy.cli.download = _patched_spacy_download

# Also patch sys.exit to prevent it from killing the process
_original_sys_exit = sys.exit
def _patched_sys_exit(code=0):
    # Don't actually exit, just log it
    if code != 0:
        print(f"[WARN] sys.exit({code}) called, but continuing...")
        sys.stdout.flush()
    # Don't call original sys.exit - just return
    return
sys.exit = _patched_sys_exit

# Now import kokoro - spacy.cli.download is patched so misaki won't try to download
from kokoro import KPipeline

# Get configuration from JSON file
""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
""" <> ~S"""

""" <> ~S"""

text = config.get('text')
lang_code = config.get('lang_code', 'a')
voice = config.get('voice', 'af_heart')
voice_file = config.get('voice_file')
speed = config.get('speed', 1.0)
split_pattern = config.get('split_pattern', '\\n+')
output_format = config.get('output_format', 'wav')
sample_rate = config.get('sample_rate', 24000)

# Get weights directory from config
kokoro_weights_dir = config.get('kokoro_weights_dir')

# Fallback to default path if not in config
if not kokoro_weights_dir:
    base_dir = Path.cwd()
    kokoro_weights_dir = str(base_dir / "pretrained_weights" / "Kokoro-82M")

# Ensure path is string
kokoro_weights_dir = str(Path(kokoro_weights_dir).resolve())

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

print("\n=== Step 3: Initialize Kokoro Pipeline ===")
sys.stdout.flush()
print(f"Loading Kokoro pipeline for language code: {lang_code}")
sys.stdout.flush()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
sys.stdout.flush()

# Force flush all streams before initialization
sys.stdout.flush()
sys.stderr.flush()

try:
    # Initialize Kokoro pipeline
    # Kokoro automatically downloads weights from Hugging Face if not found locally
    # Pass repo_id explicitly to suppress warning and ensure non-interactive mode
    # Suppress warnings during initialization
    # Note: subprocess.run is already patched above, so misaki won't hang
    print("Initializing pipeline (this may take a moment)...")
    sys.stdout.flush()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use explicit repo_id to avoid warning and ensure non-interactive
        pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')

    print(f"[OK] Kokoro pipeline initialized for language code '{lang_code}'")
    sys.stdout.flush()
    sys.stderr.flush()

except Exception as e:
    print(f"[ERROR] Error initializing pipeline: {e}")
    import traceback
    traceback.print_exc()
    print("\nMake sure you have")
    print("  1. All dependencies installed via uv (including kokoro>=0.9.4)")
    print("  2. misaki[en] or appropriate language support installed")
    print("  3. espeak-ng installed (system dependency)")
    raise

print("\n=== Step 4: Load Voice ===")
sys.stdout.flush()

# Load voice (either from file or use default)
voice_tensor = None
if voice_file:
    print(f"Loading voice from file: {voice_file}")
    sys.stdout.flush()
    try:
        voice_tensor = torch.load(voice_file, weights_only=True)
        print(f"[OK] Voice tensor loaded from {voice_file}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[WARN] Could not load voice file: {e}")
        print(f"Falling back to default voice: {voice}")
        sys.stdout.flush()
        voice_tensor = None
else:
    print(f"Using default voice: {voice}")
    sys.stdout.flush()

print("\n=== Step 5: Generate Speech ===")
print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
print(f"Speed: {speed}x")
print(f"Split pattern: {split_pattern}")
sys.stdout.flush()

# Split text by pattern if provided
if split_pattern:
    segments = re.split(split_pattern, text)
    segments = [s.strip() for s in segments if s.strip()]
    print(f"Split into {len(segments)} segment(s)")
else:
    segments = [text]

# Generate audio for each segment
all_audio_segments = []
all_graphemes = []
all_phonemes = []

try:
    # Use voice tensor if provided, otherwise use voice string
    voice_input = voice_tensor if voice_tensor is not None else voice

    # Suppress warnings during generation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator = pipeline(
            text,
            voice=voice_input,
            speed=speed,
            split_pattern=split_pattern if split_pattern else None
        )

    # Process generator output
    # Convert generator to list to avoid blocking issues
    generator_items = list(generator)

    for i, (gs, ps, audio) in enumerate(generator_items):
        print(f"\nSegment {i+1}/{len(generator_items)}:")
        print(f"  Graphemes: {gs[:100]}{'...' if len(gs) > 100 else ''}")
        print(f"  Phonemes: {ps[:100]}{'...' if len(ps) > 100 else ''}")
        print(f"  Audio length: {len(audio) / sample_rate:.2f}s")
        sys.stdout.flush()

        all_audio_segments.append(audio)
        all_graphemes.append(gs)
        all_phonemes.append(ps)

    print(f"\n[OK] Generated {len(all_audio_segments)} audio segment(s)")
    sys.stdout.flush()

except Exception as e:
    print(f"[ERROR] Error during generation: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n=== Step 6: Save Audio ===")

# Create output directory with timestamp
tag = time.strftime("%Y%m%d_%H_%M_%S")
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True)

# Concatenate all audio segments
if all_audio_segments:
    # Concatenate audio arrays
    full_audio = np.concatenate(all_audio_segments)

    # Save main audio file
    output_filename = f"kokoro_{tag}.{output_format}"
    output_path = export_dir / output_filename

    # Save audio using soundfile
    sf.write(str(output_path), full_audio, sample_rate, format=output_format.upper())
    print(f"[OK] Saved audio to {output_path}")
    print(f"  Duration: {len(full_audio) / sample_rate:.2f}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Format: {output_format.upper()}")

    # Also save individual segments if multiple
    if len(all_audio_segments) > 1:
        for i, (audio_seg, gs, ps) in enumerate(zip(all_audio_segments, all_graphemes, all_phonemes)):
            segment_filename = f"kokoro_{tag}_segment_{i:02d}.{output_format}"
            segment_path = export_dir / segment_filename
            sf.write(str(segment_path), audio_seg, sample_rate, format=output_format.upper())
            print(f"[OK] Saved segment {i+1} to {segment_path}")

            # Save text metadata for each segment
            metadata_filename = f"kokoro_{tag}_segment_{i:02d}.txt"
            metadata_path = export_dir / metadata_filename
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"Graphemes: {gs}\n")
                f.write(f"Phonemes: {ps}\n")
            print(f"[OK] Saved metadata to {metadata_path}")
else:
    print("[ERROR] No audio segments generated")
    raise ValueError("No audio segments generated")

print("\n=== Complete ===")
print(f"Generated speech saved to: {output_path}")
print(f"\nOutput files")
print(f"  - {output_path} (Main audio file)")
if len(all_audio_segments) > 1:
    print(f"  - {len(all_audio_segments)} segment files")
    print(f"  - {len(all_audio_segments)} metadata files")
sys.stdout.flush()
sys.stderr.flush()
""", %{"config_file_normalized" => config_file_normalized})
rescue
  e ->
    # Clean up temp file on error
    if File.exists?(config_file) do
      File.rm(config_file)
    end
    reraise e, __STACKTRACE__
after
  # Clean up temp file
  if File.exists?(config_file) do
    File.rm(config_file)
  end
end
end)

IO.puts("\n=== Complete ===")
IO.puts("TTS generation completed successfully!")

# Display OpenTelemetry trace - save to output directory
SpanCollector.display_trace("output")
