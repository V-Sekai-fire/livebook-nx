# Tools Research - Matching Script Style

## Script Style Analysis

Your Elixir scripts follow a consistent pattern:
- **Language**: Elixir CLI scripts using `Mix.install`
- **Python Integration**: Via `Pythonx` for ML model execution
- **Observability**: OpenTelemetry for tracing and performance monitoring
- **Model Source**: Hugging Face model repositories
- **Structure**: Shared utilities, ArgsParser modules, consistent error handling
- **Dependencies**: Managed via `uv` (Python package manager)

## Existing Tools (Avoid These)

- **Text-to-Image**: Z-Image-Turbo
- **Image-to-3D**: PartCrafter
- **Vision-Language**: Qwen3-VL
- **Video Processing**: SAM3 (segmentation)
- **Audio TTS**: Kokoro TTS
- **Voice Cloning**: KVoiceWalk

## Recommended Tools by Category (NEW - Different from Existing)

**Note: All models listed below are FOSS (Free and Open Source Software) with permissive licenses (MIT, Apache 2.0, BSD, or similar).**

### 1. Image-to-Image (Inpainting, Outpainting, Style Transfer)

#### LanPaint (Universal Inpainting)
- **Model**: `charrywhite/LanPaint`
- **Style Match**: ✅ High - Training-free diffusion inpainting
- **Integration**: Universal inpainting sampler with "think mode"
- **Script Name**: `lanpaint_inpainting.exs`
- **Features**: 
  - Multiple iterations before denoising
  - Works with any diffusion model
  - Superior inpainting quality

#### Paint-by-Inpaint
- **Model**: `paint-by-inpaint`
- **Style Match**: ✅ High - Advanced inpainting model
- **Integration**: Fine-tuned for inpainting tasks
- **Script Name**: `paint_by_inpaint.exs`

#### Stable Diffusion Inpainting
- **Model**: Various SD inpainting models via diffusers
- **License**: CreativeML Open RAIL-M ✅ FOSS
- **Style Match**: ✅ High - Industry standard inpainting
- **Integration**: Uses diffusers pipeline
- **Script Name**: `stable_diffusion_inpainting.exs`
- **Note**: Can use SDXL or SD 1.5/2.1 base models, permissive license

### 2. Image Super-Resolution / Upscaling

#### UltraZoom-2X
- **Model**: `andrewdalpino/UltraZoom-2X`
- **Style Match**: ✅ High - Fast single image super-resolution
- **Integration**: Two-stage "zoom in and enhance" mechanism
- **Script Name**: `ultrazoom_upscaling.exs`
- **Features**:
  - Fast and scalable
  - Controllable enhancements (denoising, deblurring, deartifacting)
  - Full RGB support

#### FLUX Upscale
- **Model**: `wangkanai/flux-upscale`
- **Style Match**: ✅ High - Real-ESRGAN upscale models
- **Integration**: 2x and 4x upscaling with detail enhancement
- **Script Name**: `flux_upscale.exs`
- **Features**:
  - Post-processing for AI-generated images
  - Noise reduction and artifact removal
  - CPU and GPU compatible

#### Real-ESRGAN
- **Model**: Various Real-ESRGAN models
- **License**: BSD-3-Clause ✅ FOSS
- **Style Match**: ✅ High - Popular upscaling solution
- **Integration**: General purpose or specialized (anime, photo, text)
- **Script Name**: `realesrgan_upscaling.exs`
- **Note**: Multiple variants for different content types, fully open-source

### 3. OCR / Document Processing

#### DeepSeek-OCR
- **Model**: `deepseek-ai/DeepSeek-OCR`
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - High-accuracy OCR model
- **Integration**: Extracts text from complex visual inputs
- **Script Name**: `deepseek_ocr.exs`
- **Features**:
  - Documents, screenshots, receipts, natural scenes
  - Multilingual support
  - High accuracy
  - Fully open-source (Apache 2.0)

#### LightOnOCR-1B
- **Model**: `lightonai/LightOnOCR-1B-1025`
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Efficient OCR model
- **Integration**: 1B parameter model for text extraction
- **Script Name**: `lighton_ocr.exs`
- **Note**: Smaller, faster alternative, fully open-source

#### PaddleOCR
- **Model**: Various PaddleOCR models
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Open-source OCR solution
- **Integration**: End-to-end OCR with detector and recognizer
- **Script Name**: `paddleocr.exs`
- **Features**:
  - Multi-line, multi-block text
  - Natural scene text
  - Multiple languages
  - Fully open-source (Apache 2.0)

#### TrOCR (Microsoft)
- **Model**: `microsoft/trocr-base-printed` or variants
- **License**: MIT ✅ FOSS
- **Style Match**: ✅ High - Transformer-based OCR
- **Integration**: Text recognition from images
- **Script Name**: `trocrocr.exs`
- **Note**: Fully open-source, good for printed text

### 4. Text Generation (LLMs)

#### Mistral-7B
- **Model**: `mistralai/Mistral-7B-v0.1` or `Open-Orca/Mistral-7B-OpenOrca`
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Popular open-source LLM
- **Integration**: Text generation, chat, instruction following
- **Script Name**: `mistral_text_generation.exs`
- **Features**:
  - 7B parameters
  - Efficient inference
  - Multiple fine-tuned variants
  - Fully permissive Apache 2.0 license

#### Phi-3.5
- **Model**: `microsoft/Phi-3.5-mini-instruct`
- **License**: MIT ✅ FOSS
- **Style Match**: ✅ High - Small, efficient LLM
- **Integration**: Instruction-tuned for chat
- **Script Name**: `phi3_text_generation.exs`
- **Features**:
  - Small model size (3.8B)
  - High quality outputs
  - Good for resource-constrained environments
  - Fully permissive MIT license

#### Gemma (Google)
- **Model**: `google/gemma-2b` or `google/gemma-7b`
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Google's fully open-source LLM
- **Integration**: Text generation, chat
- **Script Name**: `gemma_text_generation.exs`
- **Note**: Fully permissive Apache 2.0 license, multiple sizes available

#### Qwen2.5 (Alibaba)
- **Model**: `Qwen/Qwen2.5-7B-Instruct` or variants
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Fully open-source LLM
- **Integration**: Text generation, chat
- **Script Name**: `qwen2_text_generation.exs`
- **Note**: Apache 2.0 license, strong performance

### 5. Code Generation

#### Qwen Coder
- **Model**: `Qwen/Qwen2.5-Coder-7B-Instruct` or variants
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - State-of-the-art code LLM
- **Integration**: Code generation, completion, understanding
- **Script Name**: `qwen_coder_generation.exs`
- **Features**:
  - High performance on code generation tasks
  - Multiple programming languages
  - Instruction-tuned for code tasks
  - Fully permissive Apache 2.0 license
  - Better performance than StarCoder2

#### StarCoder2
- **Model**: `bigcode/starcoder2-15b` or variants
- **License**: BigCode Open RAIL-M ✅ FOSS
- **Style Match**: ✅ High - State-of-the-art code LLM
- **Integration**: Code generation, completion, understanding
- **Script Name**: `starcoder2_generation.exs`
- **Features**:
  - Trained on permissively licensed code
  - 80+ programming languages
  - Git commits, GitHub issues, Jupyter notebooks
  - Open RAIL-M license (permissive with use restrictions)

#### DeepSeek Coder
- **Model**: `deepseek-ai/deepseek-coder-1.3b-base` or larger variants
- **License**: MIT ✅ FOSS
- **Style Match**: ✅ High - Fully open-source code generation model
- **Integration**: General code synthesis and understanding
- **Script Name**: `deepseek_coder_generation.exs`
- **Variants**:
  - Base models (1.3B, 6.7B, 33B)
  - Instruction-tuned variants
  - Fully permissive MIT license

#### WizardCoder
- **Model**: `WizardLMTeam/WizardCoder-15B-V1.0` or variants
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Evol-Instruct fine-tuned
- **Integration**: Complex coding instructions
- **Script Name**: `wizardcoder_generation.exs`
- **Features**:
  - High HumanEval scores
  - Follows complex instructions
  - Multiple size variants
  - Fully open-source (Apache 2.0)

### 6. Speech Recognition / Transcription

#### Whisper (OpenAI)
- **Model**: `openai/whisper-large-v3` or smaller variants
- **License**: MIT ✅ FOSS
- **Style Match**: ✅ High - State-of-the-art speech recognition
- **Integration**: Audio → Text transcription
- **Script Name**: `whisper_transcription.exs`
- **Features**:
  - Multilingual ASR
  - Speech translation
  - Multiple sizes (tiny to large)
  - High accuracy
  - Fully permissive MIT license

#### Distil-Whisper
- **Model**: `distil-whisper/distil-large-v3` or variants
- **License**: MIT ✅ FOSS
- **Style Match**: ✅ High - Faster, smaller Whisper
- **Integration**: Distilled version of Whisper
- **Script Name**: `distil_whisper_transcription.exs`
- **Features**:
  - 6x faster than Whisper
  - 50% smaller
  - Similar accuracy
  - Fully permissive MIT license

### 7. Image Classification / Object Detection

#### Vision Transformer (ViT)
- **Model**: Various ViT models (e.g., `google/vit-base-patch16-224`)
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Image classification
- **Integration**: Classify images into categories
- **Script Name**: `vit_classification.exs`
- **Note**: Multiple variants and sizes available, fully open-source

#### YOLO (Object Detection)
- **Model**: Various YOLO models (YOLOv8, YOLOv9, etc.)
- **License**: AGPL-3.0 ✅ FOSS
- **Style Match**: ✅ High - Real-time object detection
- **Integration**: Detect and locate objects in images
- **Script Name**: `yolo_detection.exs`
- **Features**:
  - Bounding box detection
  - Multiple object classes
  - Real-time performance
  - Fully open-source (AGPL-3.0)

#### DETR (Detection Transformer)
- **Model**: `facebook/detr-resnet-50` or variants
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Transformer-based detection
- **Integration**: End-to-end object detection
- **Script Name**: `detr_detection.exs`
- **Note**: Fully open-source (Apache 2.0)

#### RF-DETR (Roboflow)
- **Model**: `roboflow/rfdetr-base` or variants (nano, small, medium, base)
- **License**: Apache 2.0 ✅ FOSS
- **Repository**: https://github.com/roboflow/rf-detr
- **Style Match**: ✅ High - SOTA real-time object detection and segmentation
- **Integration**: Real-time transformer-based detection and instance segmentation
- **Script Name**: `rfdetr_detection.exs`
- **Features**:
  - First real-time model to exceed 60 AP on COCO
  - State-of-the-art on RF100-VL benchmark
  - Instance segmentation support (RF-DETR Seg)
  - Faster and more accurate than YOLO at similar sizes
  - Multiple model sizes (Nano, Small, Medium, Base)
  - Fully open-source (Apache 2.0)
- **Note**: Based on DETR architecture, optimized for real-time performance

### 8. Image Segmentation

#### Segment Anything Model (SAM)
- **Model**: `facebook/sam-vit-base` or variants
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Image segmentation
- **Integration**: Segment objects in images (different from SAM3 which is video)
- **Script Name**: `sam_segmentation.exs`
- **Note**: Different from SAM3 (video segmentation), fully open-source

#### SegFormer
- **Model**: `nvidia/segformer-b0-finetuned-ade-640-640`
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Semantic segmentation
- **Integration**: Segment images into semantic regions
- **Script Name**: `segformer_segmentation.exs`
- **Note**: Fully open-source (Apache 2.0)

### 9. Depth Estimation

#### DepthPro (Apple)
- **Model**: `apple/DepthPro` or `apple/DepthPro-hf`
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Sharp monocular metric depth estimation
- **Integration**: Zero-shot metric monocular depth estimation
- **Script Name**: `depthpro_estimation.exs`
- **Features**:
  - High-resolution depth maps with sharp boundaries
  - Metric depth with absolute scale (no camera intrinsics needed)
  - Fast inference (2.25MP depth map in 0.3s on GPU)
  - State-of-the-art performance on in-the-wild scenes
  - Fully open-source (Apache 2.0)
- **Repository**: https://github.com/apple/ml-depth-pro

#### DPT (Dense Prediction Transformer)
- **Model**: `Intel/dpt-large` or variants
- **License**: MIT ✅ FOSS
- **Style Match**: ✅ High - Monocular depth estimation
- **Integration**: Estimate depth from single images
- **Script Name**: `dpt_depth_estimation.exs`
- **Features**:
  - High accuracy
  - Transformer-based
  - Multiple variants
  - Fully permissive MIT license

#### MiDaS
- **Model**: `Intel/MiDaS` or variants
- **License**: MIT ✅ FOSS
- **Style Match**: ✅ High - Popular depth estimation
- **Integration**: Monocular depth estimation
- **Script Name**: `midas_depth_estimation.exs`
- **Note**: Well-established model, fully permissive MIT license

### 10. Face Recognition / Detection

#### AuraFace
- **Model**: Various AuraFace models
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Open-source face recognition
- **Integration**: Face recognition and identity preservation
- **Script Name**: `auraface_recognition.exs`
- **Features**:
  - Face recognition
  - Identity preservation
  - Fully open-source (Apache 2.0)

#### RetinaFace
- **Model**: Various RetinaFace models
- **License**: MIT ✅ FOSS
- **Style Match**: ✅ High - Face detection
- **Integration**: Detect faces in images
- **Script Name**: `retinaface_detection.exs`
- **Note**: Fully permissive MIT license

### 11. Pose Estimation

#### MediaPipe Pose
- **Model**: `qualcomm/MediaPipe-Pose-Estimation`
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Real-time pose estimation
- **Integration**: Detect and track human body poses
- **Script Name**: `mediapipe_pose_estimation.exs`
- **Features**:
  - Real-time performance
  - Bounding boxes and pose skeletons
  - Mobile-optimized
  - Fully open-source (Apache 2.0)

#### ViTPose
- **Model**: Various ViTPose models
- **License**: Apache 2.0 ✅ FOSS
- **Style Match**: ✅ High - Vision Transformer for pose
- **Integration**: Human pose estimation
- **Script Name**: `vitpose_estimation.exs`
- **Note**: Fully open-source (Apache 2.0)

## Integration Pattern Template

All new tools should follow this pattern:

```elixir
#!/usr/bin/env elixir

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 V-Sekai-fire
#
# [Tool Name] Script
# [Description]
# Model: [Model Name] ([parameters])
# Repository: https://huggingface.co/[repo-id]

# Configure OpenTelemetry
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

# Initialize Python environment
Pythonx.uv_init("""
[project]
name = "[tool-name]"
version = "0.0.0"
requires-python = "==3.10.*"
dependencies = [
  # Tool-specific dependencies
]

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
""")

# ArgsParser module
defmodule ArgsParser do
  # Argument parsing logic
end

# Main execution with OpenTelemetry spans
SpanCollector.track_span("[tool].generation", fn ->
  # Tool execution logic
end)

# Display trace
SpanCollector.display_trace()
```

## Priority Recommendations (NEW Categories Only - FOSS Only)

**All recommendations below are for fully FOSS models with permissive licenses (MIT, Apache 2.0, BSD, or similar).**

### High Priority (Best Style Match & Popularity)

1. **Whisper** (`whisper_transcription.exs`)
   - **License**: MIT ✅ FOSS
   - State-of-the-art speech recognition
   - Complements Kokoro TTS (transcription vs generation)
   - High accuracy, multilingual
   - Multiple size variants

2. **DeepSeek-OCR** (`deepseek_ocr.exs`)
   - **License**: Apache 2.0 ✅ FOSS
   - High-accuracy OCR
   - Extracts text from documents, screenshots, receipts
   - Multilingual support
   - Production-ready

3. **Qwen Coder** (`qwen_coder_generation.exs`)
   - **License**: Apache 2.0 ✅ FOSS
   - State-of-the-art code generation
   - Better performance than StarCoder2
   - Multiple programming languages
   - Fully permissive Apache 2.0 license
   - Well-documented

4. **UltraZoom-2X** (`ultrazoom_upscaling.exs`)
   - Fast image super-resolution
   - Controllable enhancements
   - Good for post-processing

5. **LanPaint** (`lanpaint_inpainting.exs`)
   - Universal inpainting
   - Works with any diffusion model
   - Superior quality

### Medium Priority

6. **DeepSeek Coder** (`deepseek_coder_generation.exs`)
   - Fully open-source code generation (MIT)
   - Multiple variants (1.3B, 6.7B, 33B)
   - Well-documented

7. **MediaPipe Pose** (`mediapipe_pose_estimation.exs`)
   - Real-time pose estimation
   - Mobile-optimized
   - Good for applications

8. **YOLO** (`yolo_detection.exs`)
   - Real-time object detection
   - Popular and well-supported
   - Multiple versions available

9. **DPT Depth Estimation** (`dpt_depth_estimation.exs`)
   - High-accuracy depth estimation
   - Transformer-based
   - Useful for 3D applications

10. **RF-DETR** (`rfdetr_detection.exs`)
    - **License**: Apache 2.0 ✅ FOSS
    - SOTA real-time object detection and segmentation
    - First real-time model to exceed 60 AP on COCO
    - Faster and more accurate than YOLO at similar sizes
    - Instance segmentation support
    - Repository: https://github.com/roboflow/rf-detr

## Model Repository Links (NEW Categories)

### Image-to-Image (Inpainting/Outpainting)
- `charrywhite/LanPaint`
- `paint-by-inpaint`
- Various Stable Diffusion inpainting models

### Image Super-Resolution
1. Various Real-ESRGAN models (BSD-3-Clause) - Most established, multiple variants for different content types
2. `andrewdalpino/UltraZoom-2X` - Fast, scalable, controllable enhancements
3. `wangkanai/flux-upscale` - Real-ESRGAN based, optimized for AI-generated images

### OCR / Document Processing
- `deepseek-ai/DeepSeek-OCR` (Apache 2.0)
- `lightonai/LightOnOCR-1B-1025` (Apache 2.0)
- `microsoft/trocr-base-printed` (MIT)
- Various PaddleOCR models (Apache 2.0)

### Code Generation
- `Qwen/Qwen2.5-Coder-7B-Instruct` (Apache 2.0)
- `bigcode/starcoder2-15b` (BigCode Open RAIL-M)
- `deepseek-ai/deepseek-coder-1.3b-base` (MIT)
- `WizardLMTeam/WizardCoder-15B-V1.0` (Apache 2.0)

### Speech Recognition
- `openai/whisper-large-v3`
- `distil-whisper/distil-large-v3`

### Image Classification / Object Detection
- `google/vit-base-patch16-224`
- Various YOLO models
- `facebook/detr-resnet-50`
- `roboflow/rfdetr-base` (Apache 2.0) - https://github.com/roboflow/rf-detr

### Depth Estimation
- `apple/DepthPro` or `apple/DepthPro-hf` (Apache 2.0) - Sharp monocular metric depth, fast inference
- `Intel/dpt-large` (MIT)
- `Intel/MiDaS` (MIT)

### Face Recognition / Detection
- Various AuraFace models
- Various RetinaFace models

### Pose Estimation
- `qualcomm/MediaPipe-Pose-Estimation`
- Various ViTPose models

## Next Steps

1. Prioritize based on use case
2. Create new `.exs` scripts following the template
3. Integrate with existing `shared_utils.exs`
4. Add OpenTelemetry tracing
5. Test with actual model downloads
