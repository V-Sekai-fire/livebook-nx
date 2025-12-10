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
    parser.add_argument("--image_input", type=str, required=True)
    parser.add_argument("--mask_input", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--simplify_ratio", type=float, default=0.3)
    parser.add_argument("--partfield_encoder_path", type=str, default="ckpt/model_objaverse.ckpt")
    parser.add_argument("--bbox_gen_ckpt", type=str, default="ckpt/bbox_gen.ckpt")
    parser.add_argument("--part_synthesis_ckpt", type=str, default="omnipart/OmniPart")
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
    
    # Sample SLAT once, then decode formats separately to avoid SIGFPE crashes
    # This reuses the expensive sampling calculation while isolating format-specific crashes
    print("[INFO] Sampling SLAT once, then decoding formats separately...")
    print("[DEBUG] Clearing CUDA cache...")
    torch.cuda.empty_cache()
    print("[DEBUG] CUDA cache cleared")
    
    print(f"[DEBUG] Image shape: {img_black_bg.shape if hasattr(img_black_bg, 'shape') else type(img_black_bg)}")
    print(f"[DEBUG] Coords shape: {part_synthesis_input['coords'].shape}")
    print(f"[DEBUG] Part layouts count: {len(part_synthesis_input['part_layouts'])}")
    print(f"[DEBUG] Masks shape: {part_synthesis_input['masks'].shape}")
    print(f"[DEBUG] Seed: {args.seed}, Steps: {args.num_inference_steps}, Guidance: {args.guidance_scale}")
    
    # Sample SLAT once (the expensive operation)
    print("[DEBUG] Getting condition...")
    # get_cond expects either a batched tensor (B, C, H, W) or list of PIL Images
    # img_black_bg from load_img_mask is a torch.Tensor with shape (C, H, W)
    if isinstance(img_black_bg, torch.Tensor):
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        img_batched = img_black_bg.unsqueeze(0)
    else:
        # If it's a PIL Image, wrap in list
        img_batched = [img_black_bg]
    cond = part_synthesis_pipeline.get_cond(img_batched)
    print(f"[DEBUG] Condition shape: {cond.shape if hasattr(cond, 'shape') else type(cond)}")
    torch.manual_seed(args.seed)
    print("[DEBUG] Starting sample_slat...")
    slat = part_synthesis_pipeline.sample_slat(
        cond, 
        part_synthesis_input['coords'], 
        [part_synthesis_input['part_layouts']], 
        part_synthesis_input['masks'],
        sampler_params={"steps": args.num_inference_steps, "cfg_strength": args.guidance_scale},
    )
    print("[DEBUG] SLAT sampling completed")
    print(f"[DEBUG] SLAT shape: {slat.shape if hasattr(slat, 'shape') else 'N/A'}")
    print(f"[DEBUG] SLAT coords shape: {slat.coords.shape if hasattr(slat, 'coords') else 'N/A'}")
    print(f"[DEBUG] SLAT coords batch IDs: {slat.coords[:, 0].unique() if hasattr(slat, 'coords') else 'N/A'}")
    print(f"[DEBUG] SLAT feats shape: {slat.feats.shape if hasattr(slat, 'feats') else 'N/A'}")
    
    # Divide SLAT into parts (once)
    print("[DEBUG] Dividing SLAT into parts...")
    print(f"[DEBUG] Part layouts: {len(part_synthesis_input['part_layouts'])} parts")
    divided_slat = part_synthesis_pipeline.divide_slat(slat, [part_synthesis_input['part_layouts']])
    print(f"[DEBUG] Divided SLAT, type: {type(divided_slat)}, length: {len(divided_slat) if isinstance(divided_slat, list) else 'N/A'}")
    
    # Decode formats separately from the same SLAT
    part_synthesis_output = {}
    formats_to_generate = ['mesh', 'gaussian', 'radiance_field']
    
    for fmt in formats_to_generate:
        print(f"[DEBUG] Decoding {fmt} format...")
        try:
            torch.cuda.empty_cache()
            fmt_output = part_synthesis_pipeline.decode_slat(divided_slat, [fmt])
            if fmt in fmt_output:
                part_synthesis_output[fmt] = fmt_output[fmt]
                print(f"[OK] Successfully decoded {fmt} format")
            else:
                print(f"[WARN] {fmt} format not in output")
        except Exception as e:
            print(f"[WARN] Failed to decode {fmt} format: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"[INFO] Continuing with other formats...")
            continue
    
    if not part_synthesis_output:
        raise RuntimeError("Failed to generate any format")
    
    print(f"[INFO] Successfully generated formats: {list(part_synthesis_output.keys())}")
    
    print("[DEBUG] Starting save_parts_outputs...")
    save_parts_outputs(
        part_synthesis_output, 
        output_dir=output_dir, 
        simplify_ratio=args.simplify_ratio, 
        save_video=True,  # Enable Gaussian splat video export
        save_glb=False,  # Disable GLB to speed up (texture baking is slow)
        textured=False,  # Not needed if save_glb=False
    )
    print("[DEBUG] save_parts_outputs completed")
    
    print("[DEBUG] Starting merge_parts...")
    merge_parts(output_dir)
    print("[DEBUG] merge_parts completed")
    
    print(f"[INFO] PartSynthesis output saved (formats: {', '.join(part_synthesis_output.keys())})")