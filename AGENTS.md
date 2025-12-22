# AGENTS.md

This document provides essential context and guidelines for AI coding agents working with this repository.

## Project Overview

This repository contains a collection of Elixir-based CLI scripts for AI/ML model inference and generation tasks. Most scripts use Python (via `Pythonx`) for model execution, while some use native Elixir ML libraries (NX/Bumblebee). All scripts integrate with Hugging Face for model downloads and follow a consistent architecture pattern with shared utilities, OpenTelemetry observability, and standardized error handling.

### Key Technologies

- **Language**: Elixir (CLI scripts using `Mix.install`)
- **Python Integration**: `Pythonx` library for executing Python code (most scripts)
- **Native Elixir ML**: `NX` (Numerical Elixir) and `Bumblebee` for native ML inference (some scripts)
- **Python Package Management**: `uv` (via Pythonx)
- **Observability**: OpenTelemetry with AppSignal integration
- **Model Source**: Hugging Face model repositories
- **JSON Processing**: `Jason` library
- **HTTP Client**: `Req` library

## Repository Structure

```
livebook-nx/
├── *.exs                    # Main generation/inference scripts
├── shared_utils.exs        # Shared utilities and modules
├── output/                 # Generated outputs (timestamped directories)
├── pretrained_weights/     # Cached model weights
├── thirdparty/             # Third-party dependencies and tools
└── AGENTS.md              # This file
```

### Main Scripts

#### Python-based Scripts (via Pythonx)

- **qwen3vl_inference.exs**: Vision-language inference using Qwen3-VL
- **kokoro_tts_generation.exs**: Text-to-speech generation using Kokoro-82M
- **kvoicewalk_generation.exs**: Voice cloning using KVoiceWalk
- **sam3_video_segmentation.exs**: Video segmentation using SAM3
- **zimage_generation.exs**: Text-to-image generation using Z-Image-Turbo
- **unirig_generation.exs**: 3D rigging using UniRig (supports VRM bone naming)
- **tris_to_quads_converter.exs**: Mesh conversion utilities (triangles to quads)
- **corrective_smooth_baker.exs**: Mesh smoothing utilities (bakes corrective smooth modifiers)

#### Native Elixir ML Scripts (NX/Bumblebee)

- **nx.exs**: Demonstrates basic tensor operations using NX (Numerical Elixir)
- **nx_phi3.exs**: Text generation using Bumblebee with GPT-2 or other Hugging Face models

## Architecture Patterns

### Script Types

This repository contains two types of scripts:

1. **Python-based Scripts**: Use `Pythonx` to execute Python code for model inference
   - Most scripts fall into this category
   - Use shared utilities from `shared_utils.exs`
   - Integrate with OpenTelemetry for observability
   - Examples: `unirig_generation.exs`, `qwen3vl_inference.exs`, `zimage_generation.exs`

2. **Native Elixir ML Scripts**: Use NX and Bumblebee for direct Elixir-based ML inference
   - Do not require Python or Pythonx
   - Use Elixir-native tensor operations
   - Examples: `nx.exs`, `nx_phi3.exs`

### Script Structure

Most scripts follow this consistent pattern:

1. **Header**: SPDX license, copyright, description
2. **Dependencies**: `Mix.install` with required packages
3. **OpenTelemetry Setup**: Configuration (can be disabled with `--disable-telemetry`) - *Python-based scripts only*
4. **Shared Utilities**: `Code.eval_file("shared_utils.exs")` - *Python-based scripts only*
5. **Argument Parsing**: `ArgsParser` module for CLI arguments
6. **Main Logic**: 
   - Python-based scripts: Orchestration of Python execution via `Pythonx`
   - Native Elixir scripts: Direct use of NX/Bumblebee APIs
7. **Error Handling**: Standardized error handling and logging

**Note**: Native Elixir ML scripts (`nx.exs`, `nx_phi3.exs`) do not use Pythonx or shared utilities, and instead use NX/Bumblebee directly for tensor operations and model inference.

### Example Script Templates

#### Python-based Script Template

```elixir
#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire

Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
  {:req, "~> 0.5.0"},
  {:opentelemetry_api, "~> 1.3"},
  {:opentelemetry, "~> 1.3"},
  {:opentelemetry_exporter, "~> 1.6"},
])

Logger.configure(level: :info)

# Load shared utilities
Code.eval_file("shared_utils.exs")

# Initialize OpenTelemetry (unless disabled)
unless disable_telemetry do
  OtelSetup.configure()
end

# Main script logic...
```

#### Native Elixir ML Script Template

```elixir
#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire

Mix.install([
  {:bumblebee, git: "https://github.com/elixir-nx/bumblebee", tag: "main"},
  {:nx, "~> 0.7.0"},
  {:exla, "~> 0.7.1"},
], config: [nx: [default_backend: EXLA.Backend]])

require Logger

Logger.configure(level: :info)

# Main script logic using NX/Bumblebee directly...
```

## Shared Utilities (`shared_utils.exs`)

The `shared_utils.exs` file provides common functionality used across Python-based scripts (not used by native Elixir ML scripts):

### Key Modules

1. **HuggingFaceDownloader**: Downloads model repositories from Hugging Face
   - Handles recursive file tree traversal
   - Progress tracking and error handling
   - Caching support

2. **ConfigFile**: Manages temporary JSON config files for Python scripts
   - Creates timestamped config files
   - Provides Python-safe path strings

3. **OutputDir**: Creates timestamped output directories
   - Format: `output/YYYYMMDD_HH_MM_SS/`
   - Automatic cleanup of old outputs

4. **PythonEnv**: Initializes Python environments with dependencies
   - Uses `uv` for package management
   - Handles dependency installation

5. **ArgsParser**: Standardized CLI argument parsing
   - Consistent help messages
   - Type validation
   - Default value handling

6. **OtelSetup**: OpenTelemetry configuration
   - AppSignal integration
   - OTLP exporter setup
   - Resource attribute configuration

7. **SpanCollector**: OpenTelemetry span tracking
   - Wrapper around OpenTelemetry.Tracer API
   - Trace context propagation
   - Metric recording

8. **OtelLogger**: Structured logging with OpenTelemetry
   - Automatic span attribute injection
   - Log level management

## OpenTelemetry Integration

### Configuration

OpenTelemetry is configured to send telemetry data to AppSignal:

- **Endpoint**: `https://fwbkb568.eu-central.appsignal-collector.net`
- **Protocol**: HTTP/protobuf
- **API Key**: Configured in `OtelSetup` module

### Disabling Telemetry

Users can disable telemetry with the `--disable-telemetry` flag:

```bash
elixir script.exs --disable-telemetry input_file
```

### Python OpenTelemetry

Python code also integrates with OpenTelemetry:

- Trace context propagation from Elixir to Python
- Explicit spans around key operations
- OTLP exporter for AppSignal
- Local file exporter for debugging

### Span Usage Patterns

**Elixir:**
```elixir
SpanCollector.track_span("operation.name", fn ->
  # Operation code
end, [{"attribute.key", "value"}])
```

**Python:**
```python
with tracer.start_as_current_span("operation.name") as span:
    span.set_attribute("key", "value")
    # Operation code
    span.set_status(Status(StatusCode.OK))
```

## Code Style and Conventions

### Elixir

- Use `Mix.install` for dependency management (no `mix.exs`)
- Follow Elixir naming conventions (snake_case for functions, PascalCase for modules)
- Use `IO.puts` for user-facing output, `Logger` for structured logging
- Prefer pattern matching over conditionals where possible
- Use `~S"""` sigils for raw strings, `~s"""` for interpolated strings

### Python (Embedded)

- Python code is embedded as strings in Elixir scripts
- Use `Pythonx.eval` or `Pythonx.spawn` for execution
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Handle errors gracefully with try/except blocks

### File Paths

- Use absolute paths when possible
- Use `Path` module for path manipulation
- Handle Windows/Unix path differences via `ConfigFile.python_path_string`

## Development Environment Setup

### Prerequisites

- **Elixir**: Version 1.19+ (tested with 1.19.4)
- **Erlang/OTP**: Version 16.1+ (comes with Elixir)
- **Python**: Version 3.10+ (managed via Pythonx)
- **CUDA**: For GPU-accelerated models (optional)
- **Rust/Cargo**: For some Elixir dependencies (e.g., `base64`)

### Dependencies

Dependencies are installed at runtime via `Mix.install`:

```elixir
Mix.install([
  {:pythonx, "~> 0.4.7"},
  {:jason, "~> 1.4.4"},
  {:req, "~> 0.5.0"},
  {:opentelemetry_api, "~> 1.3"},
  {:opentelemetry, "~> 1.3"},
  {:opentelemetry_exporter, "~> 1.6"},
])
```

### Python Dependencies

Python dependencies are managed via `uv` (installed by Pythonx) for Python-based scripts:

- Dependencies are specified in Python code strings via `Pythonx.uv_init/1`
- Installed automatically on first run
- Cached in `~/.cache/pythonx/`

### Native Elixir ML Dependencies

Native Elixir ML scripts use:

- **NX**: Numerical Elixir for tensor operations
- **EXLA**: XLA backend for GPU acceleration (optional, requires CUDA setup)
- **Bumblebee**: Pre-trained model loading and inference
- Dependencies installed via `Mix.install` at runtime

## Common Workflows

### Adding a New Script

1. Create a new `.exs` file following the script template
2. Define CLI arguments using `ArgsParser`
3. Implement main logic using shared utilities
4. Add OpenTelemetry spans for observability
5. Test with sample inputs
6. Document usage in script header

### Debugging

- Use `IO.puts("[DEBUG] ...")` for debug output
- Check OpenTelemetry traces in AppSignal dashboard
- Review local OpenTelemetry logs in `/tmp/python_otel.log`
- Use `--disable-telemetry` to isolate telemetry-related issues

### Error Handling

- Use `try/rescue` blocks for error handling
- Log errors with `OtelLogger.error/2`
- Provide user-friendly error messages
- Clean up resources (temp files, GPU memory) in `after` blocks

## Testing

### Manual Testing

Scripts are tested manually by running with sample inputs:

```bash
# Python-based script example
elixir unirig_generation.exs model.obj

# Native Elixir ML script example
elixir nx_phi3.exs "Hello world" --max-tokens 20
```

### Output Verification

- Check output directory for generated files
- Verify file formats and sizes
- Review logs for errors or warnings

## Performance Considerations

### GPU Memory Management

- Clear CUDA cache after model operations: `torch.cuda.empty_cache()`
- Delete large tensors explicitly: `del tensor`
- Use garbage collection: `gc.collect()`
- Monitor GPU memory usage in logs

### Model Caching

- Models are cached in `pretrained_weights/` directory
- Hugging Face models cached in `~/.cache/huggingface/`
- Use `local_files_only=True` when files are already cached

### Python Execution

- Use `Pythonx.spawn` for long-running operations to avoid GIL issues
- Use `Pythonx.eval` for quick operations
- Pass large data via config files, not command-line arguments

## Security Considerations

### Telemetry Data

- Telemetry includes file paths, system metadata, and performance metrics
- No file contents are transmitted
- Users can disable telemetry with `--disable-telemetry`
- See information collection notice in scripts for details

### Model Downloads

- Models downloaded from Hugging Face (trusted source)
- Files verified via Hugging Face API
- Local caching prevents re-downloading

### File Paths

- Validate user-provided paths
- Use absolute paths to prevent directory traversal
- Sanitize paths before passing to Python

## Troubleshooting

### Common Issues

1. **Python GIL Errors**: Use `Pythonx.spawn` instead of `Pythonx.eval` for long operations (Python-based scripts)
2. **OpenTelemetry Connection Errors**: Check network connectivity, verify AppSignal endpoint (Python-based scripts)
3. **GPU Out of Memory**: Reduce batch size, enable CPU offloading, use quantization
4. **Model Download Failures**: Check network connection, verify Hugging Face access
5. **Path Issues**: Use absolute paths, check file permissions
6. **EXLA/CUDA Setup Issues** (Native Elixir ML): Ensure proper environment variables are set for CUDA support (see `nx_phi3.exs` for examples)
7. **NX Backend Errors**: Verify EXLA backend is properly configured for GPU operations

### Debug Commands

```bash
# Check Elixir version
elixir --version

# Check Python version (via Pythonx)
elixir -e "Mix.install([{:pythonx, \"~> 0.4.7\"}]); {version, _} = Pythonx.eval(\"import sys; sys.version\"); IO.puts(version)"

# Check GPU availability (Python-based scripts)
elixir -e "Mix.install([{:pythonx, \"~> 0.4.7\"}]); {available, _} = Pythonx.eval(\"import torch; torch.cuda.is_available()\"); IO.puts(available)"

# Check NX backend (Native Elixir ML)
elixir -e "Mix.install([{:nx, \"~> 0.7.0\"}, {:exla, \"~> 0.7.1\"}], config: [nx: [default_backend: EXLA.Backend]]); IO.inspect(Nx.default_backend())"
```

## Contributing Guidelines

### Code Changes

- Follow existing code patterns and conventions
- Add OpenTelemetry spans for new operations
- Update script headers with usage examples
- Test with sample inputs before committing

### Commit Messages

Use descriptive commit messages:

```
Fix OpenTelemetry span usage in Python code

- Use trace.get_current_span() instead of direct span object
- Properly enter/exit context managers
- Add error handling for span operations
```

### Documentation

- Update this AGENTS.md file when adding new patterns
- Document new CLI options in script headers
- Add examples for complex use cases

## Key Files Reference

- **shared_utils.exs**: Core utilities and modules (used by Python-based scripts)
- **unirig_generation.exs**: Complex 3D rigging pipeline with VRM support
- **zimage_generation.exs**: Text-to-image generation with native CLI API architecture
- **nx_phi3.exs**: Example of native Elixir ML using Bumblebee
- **AGENTS.md**: This file (agent documentation)

## Additional Resources

- Elixir Documentation: https://elixir-lang.org/docs.html
- Pythonx Documentation: https://hexdocs.pm/pythonx/
- OpenTelemetry Elixir: https://hexdocs.pm/opentelemetry/
- Hugging Face API: https://huggingface.co/docs/api-inference/index

