import os
import numpy as np
import torch
import argparse
from PIL import Image
from omegaconf import OmegaConf

from modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
from modules.part_synthesis.process_utils import save_parts_outputs
from modules.inference_utils import load_img_mask, prepare_bbox_gen_input, prepare_part_synthesis_input, gen_mesh_from_bounds, vis_voxel_coords, merge_parts
from modules.part_synthesis.pipelines import OmniPartImageTo3DPipeline

from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-input", "--image_input", type=str, required=True, dest="image_input")
    parser.add_argument("--mask-input", "--mask_input", type=str, required=True, dest="mask_input")
    parser.add_argument("--output-root", "--output_root", type=str, default="./output", dest="output_root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", "--num_inference_steps", type=int, default=25, dest="num_inference_steps")
    parser.add_argument("--guidance-scale", "--guidance_scale", type=float, default=7.5, dest="guidance_scale")
    parser.add_argument("--simplify-ratio", "--simplify_ratio", type=float, default=0.3, dest="simplify_ratio")
    parser.add_argument("--partfield-encoder-path", "--partfield_encoder_path", type=str, default="ckpt/model_objaverse.ckpt", dest="partfield_encoder_path")
    parser.add_argument("--bbox-gen-ckpt", "--bbox_gen_ckpt", type=str, default="ckpt/bbox_gen.ckpt", dest="bbox_gen_ckpt")
    parser.add_argument("--part-synthesis-ckpt", "--part_synthesis_ckpt", type=str, default="omnipart/OmniPart", dest="part_synthesis_ckpt")
    args = parser.parse_args()

    if not os.path.exists(args.partfield_encoder_path):
        args.partfield_encoder_path = hf_hub_download(repo_id="omnipart/OmniPart_modules", filename="partfield_encoder.ckpt", local_dir="ckpt")
    if not os.path.exists(args.bbox_gen_ckpt):
        args.bbox_gen_ckpt = hf_hub_download(repo_id="omnipart/OmniPart_modules", filename="bbox_gen.ckpt", local_dir="ckpt")

    os.makedirs(args.output_root, exist_ok=True)
    output_dir = os.path.join(args.output_root, args.image_input.split("/")[-1].split(".")[0])
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # load part_synthesis model
    part_synthesis_pipeline = OmniPartImageTo3DPipeline.from_pretrained(args.part_synthesis_ckpt)
    part_synthesis_pipeline.to(device)
    print("[INFO] PartSynthesis model loaded")

    # load bbox_gen model
    bbox_gen_config = OmegaConf.load("configs/bbox_gen.yaml").model.args
    bbox_gen_config.partfield_encoder_path = args.partfield_encoder_path
    bbox_gen_model = BboxGen(bbox_gen_config)
    bbox_gen_model.load_state_dict(torch.load(args.bbox_gen_ckpt), strict=False)
    bbox_gen_model.to(device)
    bbox_gen_model.eval().half()
    print("[INFO] BboxGen model loaded")
    
    img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis = load_img_mask(args.image_input, args.mask_input)
    img_mask_vis.save(os.path.join(output_dir, "img_mask_vis.png"))

    voxel_coords = part_synthesis_pipeline.get_coords(img_black_bg, num_samples=1, seed=args.seed, sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5})
    voxel_coords = voxel_coords.cpu().numpy()
    np.save(os.path.join(output_dir, "voxel_coords.npy"), voxel_coords)
    voxel_coords_ply = vis_voxel_coords(voxel_coords)
    voxel_coords_ply.export(os.path.join(output_dir, "voxel_coords_vis.ply"))
    print("[INFO] Voxel coordinates saved")

    bbox_gen_input = prepare_bbox_gen_input(os.path.join(output_dir, "voxel_coords.npy"), img_white_bg, ordered_mask_input)
    bbox_gen_output = bbox_gen_model.generate(bbox_gen_input)
    np.save(os.path.join(output_dir, "bboxes.npy"), bbox_gen_output['bboxes'][0])
    bboxes_vis = gen_mesh_from_bounds(bbox_gen_output['bboxes'][0])
    bboxes_vis.export(os.path.join(output_dir, "bboxes_vis.glb"))
    print("[INFO] BboxGen output saved")

    part_synthesis_input = prepare_part_synthesis_input(os.path.join(output_dir, "voxel_coords.npy"), os.path.join(output_dir, "bboxes.npy"), ordered_mask_input)
    
    # Validate inputs before processing
    print(f"[INFO] Validating inputs...")
    print(f"  Coords shape: {part_synthesis_input['coords'].shape if isinstance(part_synthesis_input['coords'], torch.Tensor) else len(part_synthesis_input['coords'])}")
    print(f"  Part layouts: {len(part_synthesis_input['part_layouts'])} parts")
    print(f"  Masks shape: {part_synthesis_input['masks'].shape}")
    print(f"  Mask value range: [{part_synthesis_input['masks'].min().item()}, {part_synthesis_input['masks'].max().item()}]")
    
    # Enable error handling for numerical issues
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False  # Disable benchmark to avoid numerical issues
    
    # Try to generate all formats together first (most efficient)
    # If SIGFPE occurs, it will crash the process, but that's a CUDA/driver issue
    print("[INFO] Generating mesh, gaussian, and radiance_field formats...")
    print("[DEBUG] Clearing CUDA cache...")
    torch.cuda.empty_cache()
    print("[DEBUG] CUDA cache cleared")
    
    print("[DEBUG] Starting get_slat call...")
    print(f"[DEBUG] Image shape: {img_black_bg.shape if hasattr(img_black_bg, 'shape') else type(img_black_bg)}")
    print(f"[DEBUG] Coords shape: {part_synthesis_input['coords'].shape}")
    print(f"[DEBUG] Part layouts count: {len(part_synthesis_input['part_layouts'])}")
    print(f"[DEBUG] Masks shape: {part_synthesis_input['masks'].shape}")
    print(f"[DEBUG] Seed: {args.seed}, Steps: {args.num_inference_steps}, Guidance: {args.guidance_scale}")
    
    try:
        part_synthesis_output = part_synthesis_pipeline.get_slat(
            img_black_bg, 
            part_synthesis_input['coords'], 
            [part_synthesis_input['part_layouts']], 
            part_synthesis_input['masks'],
            seed=args.seed,
            slat_sampler_params={"steps": args.num_inference_steps, "cfg_strength": args.guidance_scale},
            formats=['mesh', 'gaussian', 'radiance_field'],
            preprocess_image=False,
        )
        print("[DEBUG] get_slat completed successfully")
        print(f"[DEBUG] Output keys: {list(part_synthesis_output.keys())}")
        
        print("[DEBUG] Starting save_parts_outputs...")
        save_parts_outputs(
            part_synthesis_output, 
            output_dir=output_dir, 
            simplify_ratio=args.simplify_ratio, 
            save_video=False,
            save_glb=True,
            textured=False,
        )
        print("[DEBUG] save_parts_outputs completed")
        
        print("[DEBUG] Starting merge_parts...")
        merge_parts(output_dir)
        print("[DEBUG] merge_parts completed")
        
        print(f"[INFO] PartSynthesis output saved (formats: {', '.join(part_synthesis_output.keys())})")
    except Exception as e:
        print(f"[ERROR] Exception during part synthesis: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise