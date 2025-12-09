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
#   elixir kvoicewalk_generation.exs --target-folder <folder> [options]
#
# Options:
#   --target-audio <path>          Target audio file (must be 24000 Hz mono WAV)
#   --target-text <text>           Text spoken in the target audio (or path to .txt file)
#   --target-folder <path>         Folder containing audio/text pairs (processes all pairs in folder/subfolders)
#   --other-text <text>            Text for self-similarity comparison (default: provided default)
#   --starting-voice <path>        Path to starting voice tensor (.pt file)
#   --voice-folder <path>          Path to folder containing voice tensors (default: ./voices)
#   --interpolate-start            Run interpolation search before random walk
#   --transcribe-start             Transcribe target audio to text automatically
#   --population-limit <int>       Limit number of voices in random walk (default: 10)
#   --step-limit <int>             Limit number of random walk steps (default: 10000)
#   --output-name <name>           Output voice tensor filename (default: generated_voice.pt)
#   --help, -h                     Show this help message

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
  "torch>=2.0.0,<2.5.0",  # Pin to stable version range for Windows compatibility
  "torchaudio>=2.0.0,<2.5.0",
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
  "scipy",
  "librosa",
  "faster-whisper",
  "tqdm",
  "pillow",
  "spacy",
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
      elixir kvoicewalk_generation.exs --target-folder <folder> [options]

    Options:
      --target-audio <path>          Target audio file (must be 24000 Hz mono WAV)
      --target-text <text>           Text spoken in the target audio (or path to .txt file)
      --target-folder <path>         Folder containing audio/text pairs (processes all pairs recursively)
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
    {opts, _args, _} = OptionParser.parse(args,
      switches: [
        target_audio: :string,
        target_text: :string,
        target_folder: :string,
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
    target_folder = Keyword.get(opts, :target_folder)

    # Check if using folder mode
    if target_folder do
      if !File.exists?(target_folder) or !File.dir?(target_folder) do
        IO.puts("Error: Target folder not found or is not a directory: #{target_folder}")
        System.halt(1)
      end

      # Folder mode - will process all audio/text pairs
      %{
        target_folder: target_folder,
        other_text: Keyword.get(opts, :other_text),
        starting_voice: Keyword.get(opts, :starting_voice),
        voice_folder: Keyword.get(opts, :voice_folder, "./voices"),
        interpolate_start: Keyword.get(opts, :interpolate_start, false),
        transcribe_start: Keyword.get(opts, :transcribe_start, false),
        population_limit: Keyword.get(opts, :population_limit, 10),
        step_limit: Keyword.get(opts, :step_limit, 10000),
        output_name: Keyword.get(opts, :output_name, "generated_voice") |> String.replace_suffix(".pt", "")
      }
    else
      # Single file mode
      if !target_audio do
      IO.puts("""
      Error: Either --target-audio or --target-folder is required.

      Usage:
        elixir kvoicewalk_generation.exs --target-audio <audio.wav> --target-text "<text>" [options]
        elixir kvoicewalk_generation.exs --target-folder <folder> [options]

      Use --help or -h for more information.
      """)
      System.halt(1)
    end

    if !target_text do
      IO.puts("""
      Error: Target text is required when using --target-audio.

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
      output_name: Keyword.get(opts, :output_name, "generated_voice") |> String.replace_suffix(".pt", "")
    }
    end
  end
end

# Get configuration
config = ArgsParser.parse(System.argv())

# Check if folder mode
if Map.has_key?(config, :target_folder) do
  IO.puts("""
  === KVoiceWalk Voice Cloning (Folder Mode) ===
  Target Folder: #{config.target_folder}
  Voice Folder: #{config.voice_folder}
  Interpolate Start: #{config.interpolate_start}
  Transcribe Start: #{config.transcribe_start}
  Population Limit: #{config.population_limit}
  Step Limit: #{config.step_limit}
  Output Name: #{config.output_name}
  """)
else
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
end

# Add paths to config for Python
base_dir = Path.expand(".")
config_with_paths = Map.merge(config, %{
  kvoicewalk_path: Path.join([base_dir, "thirdparty", "kvoicewalk"]),
  workspace_root: base_dir
})

# Save config to JSON for Python to read (use temp file to avoid conflicts)
config_json = Jason.encode!(config_with_paths)
# Use cross-platform temp directory
tmp_dir = System.tmp_dir!()
File.mkdir_p!(tmp_dir)
config_file = Path.join(tmp_dir, "kvoicewalk_config_#{System.system_time(:millisecond)}.json")
File.write!(config_file, config_json)
config_file_normalized = String.replace(config_file, "\\", "/")

# Check if KVoiceWalk repository exists
IO.puts("\n=== Step 1: Setup KVoiceWalk Repository ===")
kvoicewalk_path = config_with_paths.kvoicewalk_path

if !File.exists?(kvoicewalk_path) do
  IO.puts("[ERROR] KVoiceWalk repository not found at: #{kvoicewalk_path}")
  IO.puts("Please ensure the repository exists in thirdparty/kvoicewalk")
  System.halt(1)
else
  IO.puts("✓ KVoiceWalk repository found at: #{kvoicewalk_path}")
end

# Import libraries and run KVoiceWalk directly (no subprocess)
SpanCollector.track_span("kvoicewalk.generation", fn ->
try do
  {_, _python_globals} = Pythonx.eval(~S"""
import json
import sys
import os
import shutil
from pathlib import Path

# Get configuration from JSON file
""" <> """
config_file_path = r"#{String.replace(config_file_normalized, "\\", "\\\\")}"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
""" <> ~S"""

target_audio = config.get('target_audio')
target_text = config.get('target_text')
target_folder = config.get('target_folder')
other_text = config.get('other_text')
starting_voice = config.get('starting_voice')
voice_folder = config.get('voice_folder', './voices')
interpolate_start = config.get('interpolate_start', False)
transcribe_start = config.get('transcribe_start', False)
population_limit = config.get('population_limit', 10)
step_limit = config.get('step_limit', 10000)
output_name = config.get('output_name', 'generated_voice')
kvoicewalk_path = config.get('kvoicewalk_path')
workspace_root = config.get('workspace_root')

# Resolve paths
if target_audio:
    target_audio = Path(target_audio).resolve()
kvoicewalk_path = Path(kvoicewalk_path).resolve()
workspace_root = Path(workspace_root).resolve()

# Check if KVoiceWalk exists
if not kvoicewalk_path.exists():
    raise FileNotFoundError(f"KVoiceWalk repository not found at: {kvoicewalk_path}")

# Add KVoiceWalk to Python path
if str(kvoicewalk_path) not in sys.path:
    sys.path.insert(0, str(kvoicewalk_path))

# Change to KVoiceWalk directory (needed for relative imports)
original_cwd = os.getcwd()
os.chdir(kvoicewalk_path)

# Create output directory
output_dir = workspace_root / "output"
output_dir.mkdir(exist_ok=True, parents=True)

# Create timestamped folder for this run
import time
tag = time.strftime("%Y%m%d_%H_%M_%S")
export_dir = output_dir / tag
export_dir.mkdir(exist_ok=True, parents=True)

# Define function to process a single audio/text pair
def process_single_pair(target_audio, target_text, other_text, starting_voice, voice_folder,
                       interpolate_start, transcribe_start, population_limit, step_limit,
                       output_name, export_dir, kvoicewalk_path):
    # Process a single audio/text pair through KVoiceWalk
    # Verify PyTorch can be imported before proceeding (only once)
    if not hasattr(process_single_pair, '_pytorch_verified'):
        print("\nVerifying PyTorch installation...")
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            process_single_pair._pytorch_verified = True
        except ImportError as e:
            print(f"\n[ERROR] Failed to import PyTorch: {e}")
            print("This may be due to missing DLLs or an incomplete installation.")
            print("Please ensure Visual C++ Redistributables are installed.")
            raise
        except Exception as e:
            print(f"\n[ERROR] PyTorch import error: {e}")
            print("This may be due to missing DLLs. Try reinstalling PyTorch or installing Visual C++ Redistributables.")
            raise

    # Import KVoiceWalk modules
    from utilities.audio_processor import Transcriber, convert_to_wav_mono_24k
    from utilities.kvoicewalk import KVoiceWalk

    # Handle target_audio - convert to mono wav 24K if needed
    if target_audio.exists() and target_audio.is_file():
        target_audio = Path(convert_to_wav_mono_24k(target_audio))
    else:
        raise FileNotFoundError(f"Target audio file not found: {target_audio}")

    # Handle transcription if requested
    if transcribe_start:
        print(f"\nTranscribing target audio...")
        transcriber = Transcriber()
        target_text = transcriber.transcribe(audio_path=target_audio)
        print(f"Transcribed text: {target_text[:100]}{'...' if len(target_text) > 100 else ''}")

    # Handle text input - read from file if it's a .txt file path
    # Ensure target_text is a string before processing
    if not target_text:
        raise ValueError("target_text is required but was not provided")

    if isinstance(target_text, str) and target_text.endswith('.txt'):
        text_path = Path(target_text)
        if text_path.exists() and text_path.is_file():
            target_text = text_path.read_text(encoding='utf-8')
        else:
            print(f"Warning: Text file not found: {text_path}, using as literal text")
    elif not isinstance(target_text, str):
        # Convert to string if it's not already
        target_text = str(target_text)

    # Set default other_text if not provided
    if not other_text:
        other_text = "If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, the resulting eruption is both a visual and tactile delight, often used in classrooms to simulate volcanic activity on a miniature scale."

    # Initialize and run KVoiceWalk
    print(f"\nInitializing KVoiceWalk...")
    try:
        kvoicewalk = KVoiceWalk(
            target_audio=target_audio,
            target_text=target_text,
            other_text=other_text,
            voice_folder=voice_folder,
            interpolate_start=interpolate_start,
            population_limit=population_limit,
            starting_voice=starting_voice if starting_voice else None,
            output_name=output_name
        )

        print(f"\nStarting random walk with {step_limit} steps...")
        kvoicewalk.random_walk(step_limit)

        print("\n[OK] KVoiceWalk completed successfully")

        # Find the generated voice file
        from utilities.path_router import OUT_DIR
        out_dir = Path(OUT_DIR)
        generated_voice = None

        if out_dir.exists():
            result_dirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith(f"{output_name}_")]

            if result_dirs:
                latest_result_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
                pt_files = list(latest_result_dir.glob("*.pt"))

                if pt_files:
                    def get_step_number(pt_file):
                        try:
                            parts = pt_file.stem.split('_')
                            output_name_parts = output_name.split('_')
                            for i in range(len(parts) - len(output_name_parts) + 1):
                                if parts[i:i+len(output_name_parts)] == output_name_parts:
                                    if i + len(output_name_parts) < len(parts):
                                        return int(parts[i + len(output_name_parts)])
                        except (ValueError, IndexError):
                            pass
                        return -1

                    pt_files.sort(key=get_step_number, reverse=True)
                    generated_voice = pt_files[0]
                    print(f"\nFound generated voice tensor: {generated_voice.name}")
                else:
                    print(f"\n[WARN] No .pt files found in {latest_result_dir}")
            else:
                print(f"\n[WARN] No results directories found matching '{output_name}_*' in {out_dir}")
        else:
            print(f"\n[WARN] Output directory not found: {out_dir}")

        if generated_voice:
            output_voice = export_dir / generated_voice.name
            shutil.copy2(generated_voice, output_voice)
            print(f"\n[OK] Generated voice tensor saved to: {output_voice}")
            print(f"\nYou can use this voice tensor with kokoro_tts_generation.exs:")
            print(f"  elixir kokoro_tts_generation.exs \"<text>\" --voice-file {output_voice}")
            return output_voice
        else:
            print(f"\n[WARN] Could not find generated voice tensor")
            print(f"  Check the KVoiceWalk output directory manually: {out_dir}")
            return None

    except Exception as e:
        print(f"\n[ERROR] Error running KVoiceWalk: {e}")
        import traceback
        traceback.print_exc()
        raise

# Handle folder mode
if target_folder:
    # Resolve target_folder before changing directories
    target_folder = Path(target_folder)
    if not target_folder.is_absolute():
        # Make it relative to workspace_root
        target_folder = workspace_root / target_folder
    target_folder = target_folder.resolve()
    print(f"\n=== Step 2: Processing Folder Mode ===")
    print(f"Target Folder: {target_folder}")
    print(f"Output Directory: {export_dir}")

    # Find all audio/text pairs recursively
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
    audio_files = []

    # Scan for audio files
    for audio_file in target_folder.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            # Look for corresponding text file
            text_file = audio_file.with_suffix('.txt')
            if text_file.exists():
                audio_files.append((audio_file, text_file))

    print(f"\nFound {len(audio_files)} audio/text pairs to combine")

    if len(audio_files) == 0:
        print(f"\n[ERROR] No audio/text pairs found in {target_folder}")
        print("Expected: audio files (.mp3, .wav, etc.) with corresponding .txt files")
        raise ValueError("No audio/text pairs found")

    # Combine all audio/text pairs from the same speaker
    print(f"\n=== Combining {len(audio_files)} audio/text pairs ===")

    # Import audio processing libraries
    import soundfile as sf
    import librosa
    import numpy as np

    # Combine all audio files
    print("Combining audio files...")
    combined_audio_segments = []
    combined_text_parts = []

    for idx, (audio_file, text_file) in enumerate(audio_files, 1):
        print(f"  Loading {idx}/{len(audio_files)}: {audio_file.name}")

        # Read and convert audio to mono 24K
        from utilities.audio_processor import convert_to_wav_mono_24k
        converted_audio = convert_to_wav_mono_24k(audio_file)

        # Load audio data
        audio_data, sample_rate = sf.read(converted_audio)
        combined_audio_segments.append(audio_data)

        # Read text
        text_content = text_file.read_text(encoding='utf-8').strip()
        combined_text_parts.append(text_content)

    # Concatenate all audio segments
    print(f"\nConcatenating {len(combined_audio_segments)} audio segments...")
    combined_audio = np.concatenate(combined_audio_segments)
    total_duration = len(combined_audio) / sample_rate
    print(f"Combined audio duration: {total_duration:.2f} seconds")

    # Combine all text with spaces
    combined_text = " ".join(combined_text_parts)
    print(f"Combined text length: {len(combined_text)} characters")
    print(f"Combined text preview: {combined_text[:200]}{'...' if len(combined_text) > 200 else ''}")

    # Save combined audio to temporary file
    import tempfile
    temp_audio_file = Path(tempfile.gettempdir()) / f"combined_{target_folder.name}_{tag}.wav"
    sf.write(str(temp_audio_file), combined_audio, sample_rate, format='WAV')
    print(f"\nSaved combined audio to temporary file: {temp_audio_file.name}")

    # Process combined audio/text as single training run
    print(f"\n=== Processing Combined Audio/Text ===")
    print(f"Using combined audio from {len(audio_files)} clips")
    print(f"Total duration: {total_duration:.2f} seconds")

    try:
        process_single_pair(
            temp_audio_file, combined_text, other_text, starting_voice, voice_folder,
            interpolate_start, transcribe_start, population_limit, step_limit,
            output_name, export_dir, kvoicewalk_path
        )
        print(f"\n=== Folder Processing Complete ===")
        print(f"Successfully processed combined audio from {len(audio_files)} pairs")
        print(f"Output directory: {export_dir}")

        # Clean up temporary file
        if temp_audio_file.exists():
            temp_audio_file.unlink()
            print(f"Cleaned up temporary file: {temp_audio_file.name}")

    except Exception as e:
        print(f"\n[ERROR] Failed to process combined audio: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temporary file on error
        if temp_audio_file.exists():
            temp_audio_file.unlink()
        raise

else:
    # Single file mode
    print(f"\n=== Step 2: Run KVoiceWalk Voice Cloning ===")
    print(f"Target Audio: {target_audio}")
    if target_text:
        print(f"Target Text: {target_text[:100]}{'...' if len(target_text) > 100 else ''}")
    else:
        print(f"Target Text: (not provided)")
    print(f"Output Directory: {export_dir}")
    print(f"\nThis may take a while depending on step_limit ({step_limit}) and population_limit ({population_limit})...")

    process_single_pair(
        Path(target_audio).resolve(), target_text, other_text, starting_voice, voice_folder,
        interpolate_start, transcribe_start, population_limit, step_limit,
        output_name, export_dir, kvoicewalk_path
    )

    # Find the generated voice file
    # KVoiceWalk saves to 'out' directory with pattern:
    # out/{output_name}_{target_audio_stem}_{timestamp}/{output_name}_{step}_{score}_{similarity}_{audio_stem}.pt
    from utilities.path_router import OUT_DIR
    out_dir = Path(OUT_DIR)
    generated_voice = None

    if out_dir.exists():
        # Find the most recent results directory matching the output_name pattern
        # Directories are named: {output_name}_{target_audio_stem}_{timestamp}
        result_dirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith(f"{output_name}_")]

        if result_dirs:
            # Get the most recent results directory (by modification time)
            latest_result_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
            # Find all .pt files in this directory
            pt_files = list(latest_result_dir.glob("*.pt"))

            if pt_files:
                # Get the file with the highest step number
                # Files are named: {output_name}_{step}_{score}_{similarity}_{audio_stem}.pt
                def get_step_number(pt_file):
                    try:
                        # Extract step number from filename
                        # Format: {output_name}_{step}_{score}_{similarity}_{audio_stem}.pt
                        parts = pt_file.stem.split('_')
                        # Find where output_name ends - step should be the next part
                        output_name_parts = output_name.split('_')
                        # Find the index after output_name parts
                        for i in range(len(parts) - len(output_name_parts) + 1):
                            if parts[i:i+len(output_name_parts)] == output_name_parts:
                                if i + len(output_name_parts) < len(parts):
                                    return int(parts[i + len(output_name_parts)])
                    except (ValueError, IndexError):
                        pass
                    return -1

                # Sort by step number and get the highest (most recent step)
                pt_files.sort(key=get_step_number, reverse=True)
                generated_voice = pt_files[0]
                print(f"\nFound generated voice tensor: {generated_voice.name}")
            else:
                print(f"\n[WARN] No .pt files found in {latest_result_dir}")
        else:
            print(f"\n[WARN] No results directories found matching '{output_name}_*' in {out_dir}")
    else:
        print(f"\n[WARN] Output directory not found: {out_dir}")

    if generated_voice:
        # Copy to output directory
        output_voice = export_dir / generated_voice.name
        shutil.copy2(generated_voice, output_voice)
        print(f"\n[OK] Generated voice tensor saved to: {output_voice}")
        print(f"\nYou can use this voice tensor with kokoro_tts_generation.exs:")
        print(f"  elixir kokoro_tts_generation.exs \"<text>\" --voice-file {output_voice}")
    else:
        print(f"\n[WARN] Could not find generated voice tensor")
        print(f"  Check the KVoiceWalk output directory manually: {out_dir}")

# Restore original working directory
os.chdir(original_cwd)

print("\n=== Complete ===")
print(f"Voice cloning completed!")
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
IO.puts("Voice cloning completed successfully!")

# Display OpenTelemetry trace
SpanCollector.display_trace()
