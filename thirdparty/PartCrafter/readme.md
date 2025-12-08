# PartCrafter Replicate Cog

This repository contains a Replicate cog for PartCrafter, a structured 3D mesh generation model that creates multiple parts and objects from a single RGB image.

## About PartCrafter

PartCrafter is a structured 3D generative model based on compositional latent diffusion transformers. It can generate multiple 3D mesh parts from a single input image in one shot.

**Original Paper**: [PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers](https://arxiv.org/abs/2506.05573)

**Original Repository**: [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter)

## Usage

### Input Parameters

- **image**: Input image for 3D mesh generation (required)
- **num_parts**: Number of parts to generate (1-16, default: 16)
- **seed**: Random seed for reproducibility (0 for random, default: 0)
- **num_tokens**: Number of tokens for generation (256/512/1024/1536/2048, default: 1024)
- **num_inference_steps**: Number of inference steps (10-100, default: 50)
- **guidance_scale**: Guidance scale for generation (1.0-15.0, default: 7.0)
- **remove_background**: Remove background from input image (default: false)
- **use_flash_decoder**: Use flash decoder for faster inference (default: false)

### Output

The model returns a single GLB file containing the merged 3D mesh with all generated parts.

## Local Development

### Prerequisites

- [Cog](https://github.com/replicate/cog) installed
- NVIDIA GPU with CUDA support
- At least 16GB GPU memory recommended

### Building the Cog

```bash
cog build
```

### Running Predictions

```bash
cog predict -i image=@path/to/your/image.jpg -i num_parts=4
```

### Testing with Example Images

You can test with the provided example images in the `assets/images/` directory:

```bash
cog predict -i image=@assets/images/np3_2f6ab901c5a84ed6bbdf85a67b22a2ee.png -i num_parts=3
```

## Model Details

- **Model Type**: Diffusion Transformer for 3D Mesh Generation
- **Input**: RGB Image
- **Output**: Structured 3D Mesh (GLB format)
- **GPU Memory**: ~8-16GB VRAM required
- **Inference Time**: ~30-60 seconds depending on parameters

## Key Features

- **Multi-part Generation**: Generates multiple semantically meaningful parts
- **Single Shot**: Creates all parts in one inference pass
- **Structured Output**: Parts are properly colored and composed
- **Background Removal**: Optional automatic background removal
- **Configurable Quality**: Adjustable inference steps and guidance scale

## Technical Implementation

The cog implementation:

1. Downloads pretrained weights from HuggingFace during setup
2. Loads both PartCrafter and RMBG (background removal) models
3. Processes input images with optional background removal
4. Runs structured diffusion generation for multiple parts
5. Composes and exports the final mesh as GLB

## Limitations

- Requires significant GPU memory (8GB+ recommended)
- Best results with clear, well-lit objects
- Performance depends on input image quality
- Generation time scales with number of parts and inference steps

## Citation

If you use this model, please cite the original paper:

```bibtex
@misc{lin2025partcrafter,
  title={PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers},
  author={Yuchen Lin and Chenguo Lin and Panwang Pan and Honglei Yan and Yiqiang Feng and Yadong Mu and Katerina Fragkiadaki},
  year={2025},
  eprint={2506.05573},
  url={https://arxiv.org/abs/2506.05573}
}
```

## License

This project follows the MIT License from the original PartCrafter repository.
