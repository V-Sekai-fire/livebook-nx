import os
import imageio
import torch
from modules.part_synthesis.utils import render_utils, postprocessing_utils
from modules.part_synthesis.representations.gaussian.gaussian_model import Gaussian


def save_parts_outputs(outputs, output_dir, simplify_ratio, save_video=False, save_glb=True, textured=True, target_total_triangles=None):
    """
    Save part outputs with optional target total triangle count for merged mesh.
    
    Args:
        target_total_triangles: Target total triangles for merged mesh (15,000-70,000).
                               If None, uses simplify_ratio directly.
                               If specified, calculates simplify_ratio to achieve target.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Debug: Print settings
    print(f"[DEBUG] save_parts_outputs settings: save_video={save_video}, save_glb={save_glb}, textured={textured}")

    # Handle missing formats gracefully
    available_formats = [fmt for fmt in ['gaussian', 'mesh', 'radiance_field'] if fmt in outputs and len(outputs[fmt]) > 0]
    if not available_formats:
        print("[WARN] No valid output formats available. Skipping save_parts_outputs.")
        return
    
    # Use the format with the most parts, or gaussian if available
    if 'gaussian' in outputs and len(outputs['gaussian']) > 0:
        num_parts = len(outputs['gaussian'])
    elif 'mesh' in outputs and len(outputs['mesh']) > 0:
        num_parts = len(outputs['mesh'])
    elif 'radiance_field' in outputs and len(outputs['radiance_field']) > 0:
        num_parts = len(outputs['radiance_field'])
    else:
        print("[WARN] No valid output formats available. Skipping save_parts_outputs.")
        return
    
    # If target_total_triangles is specified, calculate simplify_ratio from total initial faces
    if target_total_triangles is not None and save_glb:
        # First pass: count total initial faces across all parts
        total_initial_faces = 0
        for i in range(num_parts):
            if i == 0:
                continue  # Skip part 0 (overall model)
            if 'mesh' in outputs and i < len(outputs['mesh']) and outputs['mesh'][i] is not None:
                faces = outputs['mesh'][i].faces
                if faces is not None:
                    total_initial_faces += faces.shape[0]
        
        if total_initial_faces > 0:
            # Calculate simplify_ratio to achieve target
            # simplify_ratio = target_triangles / total_initial_triangles
            target_ratio = target_total_triangles / total_initial_faces
            # Clamp to reasonable range (0.01 to 1.0)
            target_ratio = max(0.01, min(1.0, target_ratio))
            simplify_ratio = target_ratio
            print(f"[INFO] Target total triangles: {target_total_triangles:,}, initial total: {total_initial_faces:,}, calculated simplify_ratio: {simplify_ratio:.4f}")
        else:
            print(f"[WARN] Could not determine initial face count, using provided simplify_ratio: {simplify_ratio}")
    
    # Save Gaussian PLY files BEFORE mesh baking
    print("[INFO] Saving Gaussian PLY files before mesh baking...")
    gs_list = []
    
    # Collect gaussians for parts > 0
    for i in range(num_parts):
        if i == 0:
            # Save part 0 gaussian if available
            if 'gaussian' in outputs and 0 < len(outputs['gaussian']):
                ply_path = f"{output_dir}/part0_gs.ply"
                if os.path.exists(ply_path):
                    os.remove(ply_path)
                outputs['gaussian'][0].save_ply(ply_path)
                print(f"[OK] Saved part0_gs.ply")
        else:
            if 'gaussian' in outputs and i < len(outputs['gaussian']):
                gs_list.append(outputs['gaussian'][i])
    
    # Save merged and exploded gaussians
    if gs_list:
        merged_gaussian = merge_gaussians(gs_list)
        merged_gaussian.save_ply(f"{output_dir}/merged_gs.ply")
        print(f"[OK] Saved merged_gs.ply")
        
        exploded_gs = exploded_gaussians(gs_list, explosion_scale=0.3)
        exploded_gs.save_ply(f"{output_dir}/exploded_gs.ply")
        print(f"[OK] Saved exploded_gs.ply")
    else:
        print("[WARN] No gaussians to merge (gs_list is empty). Skipping merged/exploded gaussian export.")
    
    # Now proceed with mesh baking and GLB generation
    for i in range(num_parts):
        if i == 0:
            continue
        if save_video:
            # Render radiance field videos (original OmniPart default)
            if 'radiance_field' in outputs and i < len(outputs['radiance_field']):
                print(f"[INFO] Rendering radiance field video for part {i}...")
                try:
                    video = render_utils.render_video(
                        outputs['radiance_field'][i], 
                        resolution=512,  # Original OmniPart default
                        ssaa=1,  # Original OmniPart default
                        num_frames=300
                    )['color']
                    rf_video_path = f"{output_dir}/part{i}_rf.mp4"
                    if os.path.exists(rf_video_path):
                        os.remove(rf_video_path)
                    imageio.mimsave(rf_video_path, video, fps=30)
                    print(f"[OK] Radiance field video saved: {rf_video_path}")
                except Exception as e:
                    print(f"[WARN] Failed to render radiance field video for part {i}: {e}")
                    import traceback
                    traceback.print_exc()
            # Render Gaussian splat videos (original OmniPart default)
            if 'gaussian' in outputs and i < len(outputs['gaussian']):
                print(f"[INFO] Rendering Gaussian splat video for part {i}...")
                try:
                    video = render_utils.render_video(
                        outputs['gaussian'][i], 
                        resolution=512,  # Original OmniPart default
                        ssaa=1,  # Original OmniPart default for Gaussian
                        num_frames=300
                    )['color']
                    gs_video_path = f"{output_dir}/part{i}_gs.mp4"
                    if os.path.exists(gs_video_path):
                        os.remove(gs_video_path)
                    imageio.mimsave(gs_video_path, video, fps=30)
                    print(f"[OK] Gaussian splat video saved: {gs_video_path}")
                except Exception as e:
                    print(f"[WARN] Failed to render Gaussian splat video for part {i}: {e}")
                    import traceback
                    traceback.print_exc()
        if save_glb:
            # Check if mesh is available
            if 'mesh' not in outputs or i >= len(outputs['mesh']) or outputs['mesh'][i] is None:
                print(f"[WARN] Mesh not available for part {i}, skipping GLB generation")
                continue
            
            print(f"[DEBUG] Generating GLB for part {i} (save_glb=True)")
            # Get gaussian if available, otherwise None
            gaussian = outputs['gaussian'][i] if 'gaussian' in outputs and i < len(outputs['gaussian']) else None
            
            # Try with textured first, fall back to untextured if it fails
            glb = postprocessing_utils.to_glb(
                gaussian,
                outputs['mesh'][i],
                simplify=simplify_ratio,  # Mesh simplification factor
                texture_size=2048,  # High-resolution texture (2K)
                textured=textured,
            )
            # If textured failed, try untextured
            if glb is None and textured:
                print(f"[WARN] Textured GLB generation failed for part {i}, trying untextured...")
                glb = postprocessing_utils.to_glb(
                    gaussian,
                    outputs['mesh'][i],
                    simplify=simplify_ratio,
                    texture_size=2048,  # High-resolution texture (2K)
                    textured=False,  # Disable texture baking
                )
            if glb is None:
                print(f"[WARN] GLB generation failed for part {i}, skipping...")
                continue
            glb_path = f"{output_dir}/part{i}.glb"
            if os.path.exists(glb_path):
                os.remove(glb_path)
            glb.export(glb_path)
            print(f"[OK] Saved part{i}.glb")


def merge_gaussians(gaussians_list):
    if not gaussians_list:
        raise ValueError("gaussians_list is empty")

    first_gaussian = gaussians_list[0]
    merged_gaussian = Gaussian(**first_gaussian.init_params, device=first_gaussian.device)
    
    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    scaling_list = []
    rotation_list = []
    opacity_list = []
    
    for gaussian in gaussians_list:
        if (gaussian.sh_degree != first_gaussian.sh_degree or 
            not torch.allclose(gaussian.aabb, first_gaussian.aabb)):
            raise ValueError("All Gaussian objects must have the same sh_degree and aabb parameters")
            
        if gaussian._xyz is not None:
            xyz_list.append(gaussian._xyz)
        if gaussian._features_dc is not None:
            features_dc_list.append(gaussian._features_dc)
        if gaussian._features_rest is not None:
            features_rest_list.append(gaussian._features_rest)
        if gaussian._scaling is not None:
            scaling_list.append(gaussian._scaling)
        if gaussian._rotation is not None:
            rotation_list.append(gaussian._rotation)
        if gaussian._opacity is not None:
            opacity_list.append(gaussian._opacity)
    
    if xyz_list:
        merged_gaussian._xyz = torch.cat(xyz_list, dim=0)
    if features_dc_list:
        merged_gaussian._features_dc = torch.cat(features_dc_list, dim=0)
    if features_rest_list:
        merged_gaussian._features_rest = torch.cat(features_rest_list, dim=0)
    else:
        merged_gaussian._features_rest = None
    if scaling_list:
        merged_gaussian._scaling = torch.cat(scaling_list, dim=0)
    if rotation_list:
        merged_gaussian._rotation = torch.cat(rotation_list, dim=0)
    if opacity_list:
        merged_gaussian._opacity = torch.cat(opacity_list, dim=0)
    
    return merged_gaussian


def exploded_gaussians(gaussians_list, explosion_scale=0.4):

    if not gaussians_list:
        raise ValueError("gaussians_list is empty")

    first_gaussian = gaussians_list[0]
    merged_gaussian = Gaussian(**first_gaussian.init_params, device=first_gaussian.device)
    
    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    scaling_list = []
    rotation_list = []
    opacity_list = []
    
    all_centers = []
    for gaussian in gaussians_list:
        if gaussian._xyz is not None:
            center = gaussian.get_xyz.mean(dim=0)
            all_centers.append(center)
    
    if not all_centers:
        raise ValueError("No valid gaussians with xyz data found")
    
    all_centers = torch.stack(all_centers)
    global_center = all_centers.mean(dim=0)
    
    for i, gaussian in enumerate(gaussians_list):
        if (gaussian.sh_degree != first_gaussian.sh_degree or 
            not torch.allclose(gaussian.aabb, first_gaussian.aabb)):
            raise ValueError("All Gaussian objects must have the same sh_degree and aabb parameters")
        
        if i < len(all_centers):
            part_center = all_centers[i]
            direction = part_center - global_center
            direction_norm = torch.norm(direction)
            if direction_norm > 1e-6:
                direction = direction / direction_norm
            else:
                direction = torch.randn(3, device=gaussian.device)
                direction = direction / torch.norm(direction)
            
            offset = direction * explosion_scale
        else:
            offset = torch.zeros(3, device=gaussian.device)
            
        if gaussian._xyz is not None:
            original_xyz = gaussian.get_xyz
            exploded_xyz = original_xyz + offset
            exploded_xyz_normalized = (exploded_xyz - gaussian.aabb[None, :3]) / gaussian.aabb[None, 3:]
            xyz_list.append(exploded_xyz_normalized)
            
        if gaussian._features_dc is not None:
            features_dc_list.append(gaussian._features_dc)
        if gaussian._features_rest is not None:
            features_rest_list.append(gaussian._features_rest)
        if gaussian._scaling is not None:
            scaling_list.append(gaussian._scaling)
        if gaussian._rotation is not None:
            rotation_list.append(gaussian._rotation)
        if gaussian._opacity is not None:
            opacity_list.append(gaussian._opacity)
    
    if xyz_list:
        merged_gaussian._xyz = torch.cat(xyz_list, dim=0)
    if features_dc_list:
        merged_gaussian._features_dc = torch.cat(features_dc_list, dim=0)
    if features_rest_list:
        merged_gaussian._features_rest = torch.cat(features_rest_list, dim=0)
    else:
        merged_gaussian._features_rest = None
    if scaling_list:
        merged_gaussian._scaling = torch.cat(scaling_list, dim=0)
    if rotation_list:
        merged_gaussian._rotation = torch.cat(rotation_list, dim=0)
    if opacity_list:
        merged_gaussian._opacity = torch.cat(opacity_list, dim=0)
    
    return merged_gaussian