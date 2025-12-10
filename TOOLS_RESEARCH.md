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
- **Hugging Face**: https://huggingface.co/charrywhite/LanPaint
- **Repository**: https://github.com/scraed/LanPaint
- **Paper**: https://arxiv.org/abs/2502.03491
- **Published**: February 2025
- **License**: GPL-3.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Training-free diffusion inpainting
- **Integration**: Universal inpainting sampler with "think mode"
- **Script Name**: `lanpaint_inpainting.exs`
- **Features**: Multiple iterations before denoising, works with any diffusion model, superior quality

#### Paint-by-Inpaint

- **Model**: `paint-by-inpaint`
- **Hugging Face**: https://huggingface.co/paint-by-inpaint
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
- **Hugging Face**: https://huggingface.co/andrewdalpino/UltraZoom-2X
- **Paper**: https://arxiv.org/abs/2506.13756
- **Published**: June 2025
- **Style Match**: ✅ High - Fast single image super-resolution
- **Integration**: Two-stage "zoom in and enhance" mechanism
- **Script Name**: `ultrazoom_upscaling.exs`
- **Features**: Fast and scalable, controllable enhancements (denoising, deblurring, deartifacting), full RGB support

#### FLUX Upscale

- **Model**: `wangkanai/flux-upscale`
- **Hugging Face**: https://huggingface.co/wangkanai/flux-upscale
- **Style Match**: ✅ High - Real-ESRGAN upscale models
- **Integration**: 2x and 4x upscaling with detail enhancement
- **Script Name**: `flux_upscale.exs`
- **Features**: Post-processing for AI-generated images, noise reduction, CPU and GPU compatible

#### Real-ESRGAN

- **Model**: Various Real-ESRGAN models
- **Repository**: https://github.com/xinntao/Real-ESRGAN
- **Paper**: https://arxiv.org/abs/2107.10833
- **Published**: July 2021 (ICCV 2021)
- **License**: BSD-3-Clause ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Popular upscaling solution
- **Integration**: General purpose or specialized (anime, photo, text)
- **Script Name**: `realesrgan_upscaling.exs`
- **Note**: Multiple variants for different content types, fully open-source

### 3. OCR / Document Processing

#### DeepSeek-OCR

- **Model**: `deepseek-ai/DeepSeek-OCR`
- **Hugging Face**: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **Repository**: https://github.com/deepseek-ai/DeepSeek-OCR
- **Published**: October 2024
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - High-accuracy OCR model
- **Integration**: Extracts text from complex visual inputs
- **Script Name**: `deepseek_ocr.exs`
- **Features**: Documents, screenshots, receipts, natural scenes, multilingual support, high accuracy

#### LightOnOCR-1B

- **Model**: `lightonai/LightOnOCR-1B-1025`
- **Hugging Face**: https://huggingface.co/lightonai/LightOnOCR-1B-1025
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Efficient OCR model
- **Integration**: 1B parameter model for text extraction
- **Script Name**: `lighton_ocr.exs`
- **Note**: Smaller, faster alternative, fully open-source

#### PaddleOCR

- **Model**: Various PaddleOCR models
- **Repository**: https://github.com/PaddlePaddle/PaddleOCR
- **Paper**: https://arxiv.org/abs/2507.05595
- **Published**: May 2020 (initial release, ongoing development)
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
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
- **Hugging Face**: https://huggingface.co/microsoft/trocr-base-printed
- **License**: MIT ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Transformer-based OCR
- **Integration**: Text recognition from images
- **Script Name**: `trocrocr.exs`
- **Note**: Fully open-source, good for printed text

### 4. Text Generation (LLMs)

#### Qwen3 (Alibaba)
- **Model**: `Qwen/Qwen3-8B` or variants
- **Hugging Face**: https://huggingface.co/Qwen/Qwen3-8B
- **Paper**: https://arxiv.org/abs/2505.09388
- **Published**: May 2025
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Latest generation open-source LLM
- **Integration**: Text generation, chat, reasoning, agent capabilities
- **Script Name**: `qwen3_text_generation.exs`
- **Features**: Seamless switching between thinking mode (complex reasoning, math, coding) and non-thinking mode (efficient dialogue), superior reasoning capabilities, enhanced human preference alignment, multilingual support, multiple sizes available

### 5. Code Generation

#### Qwen Coder

- **Model**: `Qwen/Qwen2.5-Coder-7B-Instruct` or variants
- **Hugging Face**: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- **Repository**: https://github.com/QwenLM/Qwen2.5-Coder
- **Paper**: https://arxiv.org/abs/2409.12186
- **Published**: September 2024
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - State-of-the-art code LLM
- **Integration**: Code generation, completion, understanding
- **Script Name**: `qwen_coder_generation.exs`
- **Features**: High performance, multiple programming languages, instruction-tuned, better than StarCoder2

### 6. Speech Recognition / Transcription

#### Distil-Whisper

- **Model**: `distil-whisper/distil-large-v3` or variants
- **Hugging Face**: https://huggingface.co/distil-whisper/distil-large-v3
- **Repository**: https://github.com/huggingface/distil-whisper
- **Paper**: https://arxiv.org/abs/2311.00430
- **Published**: November 2023
- **License**: MIT ✅ FOSS (OSI-approved)
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
- **Hugging Face**: https://huggingface.co/google/vit-base-patch16-224
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Image classification
- **Integration**: Classify images into categories
- **Script Name**: `vit_classification.exs`
- **Note**: Multiple variants and sizes available, fully open-source

#### RF-DETR (Roboflow)

- **Model**: `roboflow/rfdetr-base` or variants (nano, small, medium, base)
- **Hugging Face**: https://huggingface.co/roboflow/rfdetr-base
- **Repository**: https://github.com/roboflow/rf-detr
- **Paper**: https://arxiv.org/abs/2511.09554
- **Published**: March 2025
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - SOTA real-time object detection and segmentation
- **Integration**: Real-time transformer-based detection and instance segmentation
- **Script Name**: `rfdetr_detection.exs`
- **Features**: First real-time model to exceed 60 AP on COCO, state-of-the-art on RF100-VL benchmark, instance segmentation support (RF-DETR Seg), faster and more accurate than YOLO at similar sizes, multiple model sizes (Nano, Small, Medium, Base)
- **Note**: Based on DETR architecture, optimized for real-time performance

### 8. Image Segmentation

#### Segment Anything Model (SAM)

- **Model**: `facebook/sam-vit-base` or variants
- **Hugging Face**: https://huggingface.co/facebook/sam-vit-base
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Image segmentation
- **Integration**: Segment objects in images (different from SAM3 which is video)
- **Script Name**: `sam_segmentation.exs`
- **Note**: Different from SAM3 (video segmentation), fully open-source

#### SegFormer

- **Model**: `nvidia/segformer-b0-finetuned-ade-640-640`
- **Hugging Face**: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-640-640
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Semantic segmentation
- **Integration**: Segment images into semantic regions
- **Script Name**: `segformer_segmentation.exs`
- **Note**: Fully open-source (Apache 2.0)

### 9. Depth Estimation

#### DepthPro (Apple)

- **Model**: `apple/DepthPro` or `apple/DepthPro-hf`
- **Hugging Face**: https://huggingface.co/apple/DepthPro
- **Repository**: https://github.com/apple/ml-depth-pro
- **Paper**: https://arxiv.org/abs/2410.02073
- **Published**: October 2024
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Sharp monocular metric depth estimation
- **Integration**: Zero-shot metric monocular depth estimation
- **Script Name**: `depthpro_estimation.exs`
- **Features**:
  - High-resolution depth maps with sharp boundaries
  - Metric depth with absolute scale (no camera intrinsics needed)
  - Fast inference (2.25MP depth map in 0.3s on GPU)
  - State-of-the-art performance on in-the-wild scenes
  - Fully open-source (Apache 2.0)

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

#### YOLO-NAS-POSE

- **Model**: YOLO-NAS-POSE (N, S, M, L variants)
- **Repository**: https://github.com/Deci-AI/super-gradients
- **Documentation**: https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md
- **License**: Open-source (pre-trained weights for non-commercial use) ⚠️ Custom license
- **Style Match**: ✅ High - SOTA pose estimation with real-time performance
- **Integration**: Keypoint detection and pose estimation
- **Script Name**: `yolo_nas_pose_estimation.exs`
- **Features**:
  - State-of-the-art performance, outperforms YOLOv8-Pose and DEKR
  - Real-time inference (2.35ms - 8.86ms on T4 GPU for 640x640 images)
  - Multiple model sizes (N, S, M, L)
  - Quantization-aware blocks for optimized performance
  - Based on YOLO-NAS architecture with pose-optimized head
  - Transfer learning from YOLO-NAS weights
- **Note**: Generated by Deci's Neural Architecture Search (AutoNAC™) technology

#### MediaPipe Pose

- **Model**: `qualcomm/MediaPipe-Pose-Estimation`
- **Hugging Face**: https://huggingface.co/qualcomm/MediaPipe-Pose-Estimation
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
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

### 12. 3D Texture Inpainting / Generation

#### TEXGen

- **Script Pattern**: Follow the Integration Pattern Template (see below) for `texgen_texture_inpainting.exs`
- **Model**: `Andyx/TEXGen`
- **Hugging Face**: https://huggingface.co/Andyx/TEXGen
- **Repository**: https://github.com/CVMI-Lab/TEXGen
- **Paper**: https://arxiv.org/abs/2411.14740
- **Published**: November 2024 (SIGGRAPH Asia 2024)
- **License**: Apache 2.0 ✅ FOSS (OSI-approved)
- **Style Match**: ✅ High - Generative diffusion model for mesh textures
- **Integration**: 3D texture inpainting, completion, and synthesis
- **Script Name**: `texgen_texture_inpainting.exs`
- **Features**:
  - High-resolution texture map generation in UV space
  - Supports inpainting, completion, and synthesis
  - Text and image-guided texture generation
  - 700M parameter diffusion model
  - Direct generation in UV texture space (feed-forward)
- **Note**: SIGGRAPH Asia 2024, Best Paper Honorable Mention

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
