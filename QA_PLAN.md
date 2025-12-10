# Manual Ad-Hoc QA Plan

## Overview
This QA plan covers manual testing of all Elixir CLI scripts after standardization with OpenTelemetry and shared utilities.

## Prerequisites
- [ ] Elixir installed and working
- [ ] Python 3.10+ available (for Pythonx)
- [ ] CUDA-capable GPU (for GPU-accelerated scripts)
- [ ] Sufficient disk space for model downloads
- [ ] Network connection for Hugging Face downloads

## Code Review Checklist
- [ ] Verify no duplicate `SpanCollector.display_trace()` calls in any script
- [ ] Fix compiler warnings (unused default parameters, undefined Req module warnings)
- [ ] Verify all scripts properly close span tracking blocks
- [ ] Check for any duplicate module definitions
- [ ] Verify all scripts have exactly one `SpanCollector.display_trace()` call at the end

---

## 1. Shared Utilities Testing

### 1.1 HuggingFaceDownloader
- [ ] Test download with `use_otel=true` (OtelLogger integration)
- [ ] Test download with `use_otel=false` (basic IO.puts)
- [ ] Verify files are downloaded correctly
- [ ] Test with existing files (should skip)
- [ ] Test error handling for invalid repo IDs
- [ ] Verify progress display works correctly

### 1.2 ConfigFile
- [ ] Test `ConfigFile.create/2` creates temp files
- [ ] Test `ConfigFile.cleanup/1` removes temp files
- [ ] Test `ConfigFile.python_path_string/1` generates correct Python code
- [ ] Verify cross-platform path handling (Windows backslashes)

### 1.3 OutputDir
- [ ] Test `OutputDir.create/0` creates timestamped directories
- [ ] Verify timestamp format is correct
- [ ] Test directory creation in `output/` folder

### 1.4 OpenTelemetry Integration
- [ ] Verify `OtelSetup.configure/0` starts successfully
- [ ] Test `SpanCollector.start_link/0` initializes
- [ ] Verify `SpanCollector.track_span/2` records spans correctly
- [ ] Test `SpanCollector.display_trace/0` shows correct hierarchy
- [ ] Verify trace shows parent-child relationships
- [ ] Test error spans are marked correctly

---

## 2. Image Generation Scripts

### 2.1 zimage_generation.exs
- [ ] Test basic generation: `elixir zimage_generation.exs "test prompt"`
- [ ] Test with custom dimensions: `--width 512 --height 512`
- [ ] Test with custom seed: `--seed 42`
- [ ] Test with custom steps: `--num-steps 8`
- [ ] Test multiple prompts: `"prompt1" "prompt2" "prompt3"`
- [ ] Test different output formats: `--output-format jpg`
- [ ] Verify OpenTelemetry trace displays at end
- [ ] Verify trace shows: `zimage.generate`, `zimage.download_weights`, `zimage.python_generation`
- [ ] Test with existing weights (should skip download)
- [ ] Verify output files are created in timestamped directories
- [ ] Test error handling with invalid prompts
- [ ] Test help message: `--help`

### 2.2 partcrafter_generation.exs
- [ ] Test basic generation: `elixir partcrafter_generation.exs "thirdparty/9zs80jkckhrma0ctz4as2vw900.jpeg"`
- [ ] Test with custom parts: `--num-parts 4`
- [ ] Test with custom steps: `--num-steps 10`
- [ ] Test with custom tokens: `--num-tokens 512`
- [ ] Test with custom seed: `--seed 42`
- [ ] Verify OpenTelemetry trace displays at end
- [ ] Verify trace shows: `partcrafter.download_weights`, `partcrafter.generation`
- [ ] Verify output GLB files are created
- [ ] Test error handling with invalid image path: `elixir partcrafter_generation.exs "nonexistent.jpg"`
- [ ] Test help message: `--help`

---

## 3. Vision-Language Inference

### 3.1 qwen3vl_inference.exs
- [ ] Test basic inference: `elixir qwen3vl_inference.exs "thirdparty/9zs80jkckhrma0ctz4as2vw900.jpeg" "What is in this image?"`
- [ ] Test with custom max tokens: `--max-tokens 2048`
- [ ] Test with custom temperature: `--temperature 0.8`
- [ ] Test with output file: `--output response.txt`
- [ ] Test with 4-bit quantization (default)
- [ ] Test with full precision: `--full-precision`
- [ ] Verify OpenTelemetry trace displays at end
- [ ] Verify trace shows: `qwen3vl.download_weights`, `qwen3vl.inference`
- [ ] Verify response is generated correctly
- [ ] Test error handling with invalid image path: `elixir qwen3vl_inference.exs "nonexistent.jpg" "What is this?"`
- [ ] Test error handling with missing prompt: `elixir qwen3vl_inference.exs "thirdparty/9zs80jkckhrma0ctz4as2vw900.jpeg"`
- [ ] Test help message: `--help`

---

## 4. Video Processing

### 4.1 sam3_video_segmentation.exs
- [ ] Test basic segmentation: `elixir sam3_video_segmentation.exs "thirdparty/rhy08tw6k9rma0ctz7m9y0xmgr.mp4"`
- [ ] Test with custom prompt: `--prompt "person"`
- [ ] Test with custom mask color: `--mask-color "red"`
- [ ] Test with custom opacity: `--mask-opacity 0.7`
- [ ] Test mask-only mode: `--mask-only`
- [ ] Test mask-video mode: `--mask-video`
- [ ] Test ZIP output: `--return-zip`
- [ ] Verify OpenTelemetry trace displays at end
- [ ] Verify trace shows: `sam3.segmentation`
- [ ] Verify output video is created
- [ ] Test error handling with invalid video path: `elixir sam3_video_segmentation.exs "nonexistent.mp4"`
- [ ] Test help message: `--help`

---

## 5. 3D Model Processing

### 5.1 tris_to_quads_converter.exs
- [x] Test basic conversion: `elixir tris_to_quads_converter.exs "thirdparty/monkey.usdc"`
- [x] Test with custom output: `--output "output.usdc"`
- [ ] Test with GLB input: `elixir tris_to_quads_converter.exs "thirdparty/UniRig/examples/bird.glb"`
- [ ] Test with GLTF input: `elixir tris_to_quads_converter.exs "thirdparty/UniRig/examples/giraffe.glb"`
- [ ] Test with USD input: `elixir tris_to_quads_converter.exs "thirdparty/monkey.usdc"`
- [ ] Test with FBX input: `elixir tris_to_quads_converter.exs "thirdparty/UniRig/examples/skeleton/example.fbx"` (if exists)
- [ ] Verify OpenTelemetry trace displays at end
- [ ] Verify trace shows: `tris_to_quads.conversion`
- [ ] Verify output USDC file is created
- [ ] Test error handling with invalid input file
- [ ] Test help message: `--help`

### 5.2 unirig_generation.exs
- [x] Test basic rigging: `elixir unirig_generation.exs "thirdparty/monkey.usdc"`
- [ ] Test skeleton-only: `elixir unirig_generation.exs "thirdparty/monkey.usdc" --skeleton-only`
- [ ] Test skin-only: `elixir unirig_generation.exs "thirdparty/monkey.usdc" --skin-only`
- [ ] Test with GLB input: `elixir unirig_generation.exs "thirdparty/UniRig/examples/bird.glb"`
- [ ] Test with custom seed: `elixir unirig_generation.exs "thirdparty/monkey.usdc" --seed 123`
- [ ] Verify OpenTelemetry trace displays at end
- [ ] Verify trace shows: `unirig.generation`
- [ ] Verify output USDC file is created
- [ ] Test error handling with invalid mesh path
- [ ] Test help message: `--help`

### 5.3 corrective_smooth_baker.exs
- [x] Test basic baking: `elixir corrective_smooth_baker.exs "output/20251209_16_34_24/rigged.usdc"` (or use output from unirig_generation)
- [ ] Test with custom output: `elixir corrective_smooth_baker.exs "output/.../rigged.usdc" --output "baked.usdc"`
- [ ] Test with custom bake range: `elixir corrective_smooth_baker.exs "output/.../rigged.usdc" --bake-range "Selected"`
- [ ] Test with custom deviation threshold: `elixir corrective_smooth_baker.exs "output/.../rigged.usdc" --deviation-threshold 0.05`
- [ ] Test with custom bake quality: `elixir corrective_smooth_baker.exs "output/.../rigged.usdc" --bake-quality 2.0`
- [ ] Verify OpenTelemetry trace displays at end
- [ ] Verify trace shows: `corrective_smooth.baking`
- [ ] Verify output USDC file is created
- [ ] Test error handling with invalid input file: `elixir corrective_smooth_baker.exs "nonexistent.usdc"`
- [ ] Test help message: `--help`

---

## 6. Audio Processing

### 6.1 kvoicewalk_generation.exs
- [ ] Test basic voice generation: `elixir kvoicewalk_generation.exs --target-audio "thirdparty/kvoicewalk/example/target.wav" --target-text "Hello world"`
- [ ] Test with folder: `--target-folder "thirdparty/young_adult_feminine_clips/344bf332f298134d"`
- [ ] Test with custom starting voice: `--starting-voice "thirdparty/kvoicewalk/voices/af_bella.pt"`
- [ ] Test with interpolation: `--interpolate-start`
- [ ] Test with transcription: `--transcribe-start`
- [ ] Test with custom population limit: `--population-limit 20`
- [ ] Test with custom step limit: `--step-limit 5000`
- [ ] Verify OpenTelemetry trace displays at end
- [ ] Verify trace shows: `kvoicewalk.generation`
- [ ] Verify output voice tensor is created
- [ ] Test error handling with invalid audio path: `elixir kvoicewalk_generation.exs --target-audio "nonexistent.wav" --target-text "test"`
- [ ] Test help message: `--help`

### 6.2 kokoro_tts_generation.exs
- [ ] Test basic TTS: `elixir kokoro_tts_generation.exs "Hello world"`
- [ ] Test with input file: `--input-file "text.txt"` (create a test file with sample text)
- [ ] Test with custom language: `--lang-code "j"` (Japanese)
- [ ] Test with custom voice: `--voice "af_bella"`
- [ ] Test with custom voice file: `--voice-file "thirdparty/kvoicewalk/voices/af_bella.pt"`
- [ ] Test with custom speed: `--speed 1.5`
- [ ] Test with custom output format: `--output-format "mp3"`
- [ ] Test with custom sample rate: `--sample-rate 44100`
- [ ] Verify OpenTelemetry trace displays at end
- [ ] Verify trace shows: `kokoro_tts.generation`
- [ ] Verify output audio file is created
- [ ] Test error handling with invalid text
- [ ] Test help message: `--help`

---

## 7. OpenTelemetry Trace Verification

### 7.1 Trace Display Format
- [ ] Verify all scripts show "=== OpenTelemetry Trace Summary ===" header
- [ ] Verify "Total Execution Time" is displayed
- [ ] Verify "Total Span Time" is displayed
- [ ] Verify "Overhead" is displayed
- [ ] Verify "Span Breakdown:" section is displayed
- [ ] Verify spans use tree structure with `├─` characters
- [ ] Verify child spans are indented correctly
- [ ] Verify percentages are shown for each span
- [ ] Verify durations are formatted correctly (ms or s)

### 7.2 Span Hierarchy
- [ ] Verify root spans have `parent: :root`
- [ ] Verify child spans have correct parent names
- [ ] Verify nested spans are displayed in correct order
- [ ] Verify all spans are accounted for in the tree
- [ ] Test with scripts that have multiple nested spans

### 7.3 Error Handling in Traces
- [ ] Test trace display when span has error
- [ ] Verify error message is shown: `[ERROR: ...]`
- [ ] Verify trace still displays even if some spans fail
- [ ] Test trace display with no spans recorded

---

## 8. Cross-Platform Testing (Windows Focus)

### 8.1 Path Handling
- [ ] Verify Windows backslashes are handled correctly
- [ ] Test with paths containing spaces
- [ ] Test with paths containing special characters
- [ ] Verify temp file paths work on Windows
- [ ] Verify output directory creation works

### 8.2 File Operations
- [ ] Test file downloads on Windows
- [ ] Test file cleanup on Windows
- [ ] Verify file permissions work correctly
- [ ] Test with long file paths (>260 chars if applicable)

---

## 9. Performance Testing

### 9.1 First Run (Cold Start)
- [ ] Measure time for first run of each script
- [ ] Verify model downloads complete successfully
- [ ] Note any timeouts or issues

### 9.2 Subsequent Runs (Warm Start)
- [ ] Measure time for subsequent runs (with cached models)
- [ ] Compare performance with first run
- [ ] Verify caching works correctly

### 9.3 Trace Accuracy
- [ ] Verify trace times match actual execution times
- [ ] Check that overhead calculation is reasonable
- [ ] Verify span durations add up correctly

---

## 10. Error Handling

### 10.1 Invalid Inputs
- [ ] Test each script with invalid arguments
- [ ] Test with missing required arguments
- [ ] Test with invalid file paths
- [ ] Test with invalid option values
- [ ] Verify helpful error messages

### 10.2 Network Issues
- [ ] Test behavior when Hugging Face is unreachable
- [ ] Test with partial downloads
- [ ] Verify graceful degradation

### 10.3 Resource Constraints
- [ ] Test with insufficient disk space
- [ ] Test with insufficient memory (if possible)
- [ ] Verify error messages are clear

---

## 11. Integration Testing

### 11.1 Workflow Testing
- [ ] Test complete workflow: zimage → qwen3vl (image to description)
  - Generate image: `elixir zimage_generation.exs "a cat"`
  - Describe image: `elixir qwen3vl_inference.exs "output/.../zimage_*.png" "What is in this image?"`
- [ ] Test workflow: partcrafter → unirig (parts to rigged model)
  - Generate parts: `elixir partcrafter_generation.exs "thirdparty/9zs80jkckhrma0ctz4as2vw900.jpeg"`
  - Rig model: `elixir unirig_generation.exs "output/.../partcrafter_*.glb"`
- [ ] Test workflow: tris_to_quads → unirig → corrective_smooth (convert then rig then bake)
  - Convert: `elixir tris_to_quads_converter.exs "thirdparty/monkey.usdc"`
  - Rig: `elixir unirig_generation.exs "output/.../monkey_quads.usdc"`
  - Bake: `elixir corrective_smooth_baker.exs "output/.../rigged.usdc"`
- [ ] Verify intermediate files are handled correctly

### 11.2 Concurrent Execution
- [ ] Test running multiple scripts simultaneously
- [ ] Verify no file conflicts occur
- [ ] Verify temp files are unique per process

---

## 12. Documentation Verification

### 12.1 Help Messages
- [ ] Verify all scripts have `--help` option
- [ ] Verify help messages are clear and complete
- [ ] Verify examples are provided
- [ ] Verify all options are documented

### 12.2 Code Comments
- [ ] Verify scripts have usage comments at top
- [ ] Verify complex logic is commented
- [ ] Verify shared utilities are documented

---

## 13. Regression Testing

### 13.1 Functionality
- [ ] Verify all original functionality still works
- [ ] Verify no features were broken during standardization
- [ ] Compare outputs before/after changes

### 13.2 Output Quality
- [ ] Verify generated images are correct
- [ ] Verify 3D models are correct
- [ ] Verify audio output is correct
- [ ] Verify text responses are correct

---

## 14. Cleanup and Maintenance

### 14.1 Temp File Cleanup
- [ ] Verify all temp config files are cleaned up
- [ ] Test cleanup on normal exit
- [ ] Test cleanup on error exit
- [ ] Test cleanup on interrupt (Ctrl+C)

### 14.2 Resource Management
- [ ] Verify Python processes are cleaned up
- [ ] Verify GPU memory is released
- [ ] Verify file handles are closed

---

## 15. Final Checklist

- [ ] All scripts tested individually
- [ ] All OpenTelemetry traces verified
- [ ] All error cases tested
- [ ] All help messages verified
- [ ] All output files verified
- [ ] Performance acceptable
- [ ] No regressions found
- [ ] Documentation complete
- [ ] Ready for production use

---

## Notes

### Known Issues
- Document any issues found during testing
- Include workarounds if applicable

### Test Environment
- OS: Windows 10/11
- Elixir version: [version]
- Python version: [version]
- GPU: [model]
- Date: [date]

### Test Results Summary
- Total tests: [count]
- Passed: [count]
- Failed: [count]
- Skipped: [count]

