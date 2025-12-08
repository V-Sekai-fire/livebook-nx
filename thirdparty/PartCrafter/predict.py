import os
import sys
import time
import tempfile
from typing import Any, Union
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from accelerate.utils import set_seed
from cog import BasePredictor, Input, Path

# Add src to path for imports
sys.path.append("src")

from src.utils.data_utils import get_colored_mesh_composition
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Using device: {self.device}")
        
        # Download pretrained weights
        partcrafter_weights_dir = "pretrained_weights/PartCrafter"
        rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
        
        print("Downloading PartCrafter weights...")
        snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)
        
        print("Downloading RMBG weights...")
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

        # Initialize RMBG model for background removal
        print("Loading RMBG model...")
        self.rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(self.device)
        self.rmbg_net.eval()

        # Initialize PartCrafter pipeline
        print("Loading PartCrafter pipeline...")
        self.pipe = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(self.device, self.dtype)
        
        print("Setup complete!")

    @torch.no_grad()
    def run_inference(
        self,
        image_input: Union[str, Image.Image],
        num_parts: int,
        seed: int,
        num_tokens: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        max_num_expanded_coords: int = 1000000000,
        use_flash_decoder: bool = False,
        rmbg: bool = False,
    ) -> tuple:
        """Run PartCrafter inference"""
        
        if rmbg:
            img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=self.rmbg_net)
        else:
            if isinstance(image_input, str):
                img_pil = Image.open(image_input)
            else:
                img_pil = image_input

        start_time = time.time()
        outputs = self.pipe(
            image=[img_pil] * num_parts,
            attention_kwargs={"num_parts": num_parts},
            num_tokens=num_tokens,
            generator=torch.Generator(device=self.pipe.device).manual_seed(seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_num_expanded_coords=max_num_expanded_coords,
            use_flash_decoder=use_flash_decoder,
        ).meshes
        end_time = time.time()
        
        print(f"Inference time: {end_time - start_time:.2f} seconds")
        
        # Handle None outputs (decoding errors)
        for i in range(len(outputs)):
            if outputs[i] is None:
                outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        
        return outputs, img_pil

    def predict(
        self,
        image: Path = Input(description="Input image for 3D mesh generation"),
        num_parts: int = Input(
            description="Number of parts to generate", 
            default=16, 
            ge=1, 
            le=16
        ),
        seed: int = Input(
            description="Random seed for reproducibility. Use 0 for random seed", 
            default=0, 
            ge=0, 
            le=10000
        ),
        num_tokens: int = Input(
            description="Number of tokens for generation", 
            default=2048, 
            choices=[256, 512, 1024, 1536, 2048]
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", 
            default=50, 
            ge=10, 
            le=100
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation", 
            default=7.0, 
            ge=1.0, 
            le=15.0
        ),
        remove_background: bool = Input(
            description="Remove background from input image", 
            default=False
        ),
        use_flash_decoder: bool = Input(
            description="Use flash decoder for faster inference (Tempermental?)", 
            default=False
        ),
    ) -> Path:
        """Generate structured 3D mesh from input image"""
        
        # Generate random seed if seed is 0
        if seed == 0:
            seed = int.from_bytes(os.urandom(2), "big")
        
        set_seed(seed)
        
        print(f"Generating {num_parts} parts with seed {seed}")
        
        # Run inference
        outputs, processed_image = self.run_inference(
            image_input=str(image),
            num_parts=num_parts,
            seed=seed,
            num_tokens=num_tokens,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            rmbg=remove_background,
            use_flash_decoder=use_flash_decoder,
        )
        
        # Create output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export individual parts
            for i, mesh in enumerate(outputs):
                part_path = os.path.join(temp_dir, f"part_{i:02d}.glb")
                mesh.export(part_path)
            
            # Create and export merged mesh
            merged_mesh = get_colored_mesh_composition(outputs)
            output_path = os.path.join(temp_dir, "object.glb")
            merged_mesh.export(output_path)
            
            print(f"Generated {len(outputs)} parts")
            
            # Copy to final output location
            final_output = Path(tempfile.mktemp(suffix=".glb"))
            os.rename(output_path, final_output)
            
            return final_output
