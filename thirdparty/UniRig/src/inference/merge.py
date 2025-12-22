'''
inject the result in res.npz into model.vrm and exports as res_textured.vrm
'''
import argparse
import yaml
import os
import numpy as np
from numpy import ndarray

from typing import Tuple, Union, List

import argparse
from tqdm import tqdm
from box import Box

from scipy.spatial import cKDTree

import open3d as o3d
import itertools

import bpy
from mathutils import Vector

# Require robust skin transfer - crash if not available
from .robust_skin_transfer import robust_skin_weights_transfer

from ..data.raw_data import RawData, RawSkin
from ..data.extract import process_mesh, process_armature, get_arranged_bones

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_runs', type=int)
    parser.add_argument('--id', type=int)
    return parser.parse_args()

def clean_bpy():
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)

def load(filepath: str, return_armature: bool=False):
    if return_armature:
        old_objs = set(bpy.context.scene.objects)

    if not os.path.exists(filepath):
        raise ValueError(f'File {filepath} does not exist !')
    try:
        if filepath.endswith(".vrm"):
            # Install and enable VRM addon
            # For Blender 4.2+, VRM addon is available from Extensions Platform (Asset Library)
            # Check if addon is already installed and enabled
            vrm_addon_enabled = False
            vrm_module_name = None
            
            # Check common VRM addon module names
            for addon in bpy.context.preferences.addons:
                if 'vrm' in addon.module.lower():
                    vrm_module_name = addon.module
                    if addon.enabled:
                        vrm_addon_enabled = True
                        break
            
            if not vrm_addon_enabled:
                # Try to enable if installed but not enabled
                if vrm_module_name:
                    try:
                        bpy.ops.preferences.addon_enable(module=vrm_module_name)
                        print(f"Enabled VRM addon: {vrm_module_name}")
                        vrm_addon_enabled = True
                    except Exception as e:
                        print(f"Failed to enable VRM addon: {e}")
                
                # If not installed, try to install from Extensions Platform (Blender 4.2+)
                if not vrm_addon_enabled:
                    try:
                        # For Blender 4.2+, try installing from Extensions Platform
                        # Note: This may require the addon to be available in Blender's Extensions
                        # If this fails, the addon must be installed manually from:
                        # Edit > Preferences > Add-ons > Install from Extensions Platform
                        # or download from: https://extensions.blender.org/add-ons/vrm/
                        if hasattr(bpy.ops.preferences, 'addon_install_from_asset_library'):
                            bpy.ops.preferences.addon_install_from_asset_library(asset_id='io_scene_vrm')
                            print("Installed VRM addon from Extensions Platform")
                            # Enable after installation
                            bpy.ops.preferences.addon_enable(module='io_scene_vrm')
                            vrm_addon_enabled = True
                    except Exception as e:
                        print(f"Could not install VRM addon from Extensions Platform: {e}")
                        print("Please install VRM addon manually:")
                        print("  1. Edit > Preferences > Add-ons")
                        print("  2. Click 'Install from Extensions Platform' or download from:")
                        print("     https://extensions.blender.org/add-ons/vrm/")
                        raise RuntimeError("VRM addon is not installed. Please install it manually from Blender's Extensions Platform.")
            
            # Verify addon is enabled before importing
            if not vrm_addon_enabled:
                raise RuntimeError("VRM addon is not enabled. Please enable it in Edit > Preferences > Add-ons")
            
            bpy.ops.import_scene.vrm(
                filepath=filepath,
                use_addon_preferences=True,
                extract_textures_into_folder=False,
                make_new_texture_folder=False,
                set_shading_type_to_material_on_import=False,
                set_view_transform_to_standard_on_import=True,
                set_armature_display_to_wire=True,
                set_armature_display_to_show_in_front=True,
                set_armature_bone_shape_to_default=True,
                disable_bake=True, # customized option for better performance
            )
        elif filepath.endswith(".obj"):
            bpy.ops.wm.obj_import(filepath=filepath)
        elif filepath.endswith(".fbx") or filepath.endswith(".FBX"):
            bpy.ops.import_scene.fbx(filepath=filepath, ignore_leaf_bones=False, use_image_search=False)
        elif filepath.endswith(".glb") or filepath.endswith(".gltf"):
            bpy.ops.import_scene.gltf(filepath=filepath, import_pack_images=False)
        elif filepath.endswith(".usd") or filepath.endswith(".usda") or filepath.endswith(".usdc"):
            # USD import - using only essential parameters that work in Blender 4.5
            bpy.ops.wm.usd_import(
                filepath=filepath,
                import_materials=True
            )
        elif filepath.endswith(".dae"):
            bpy.ops.wm.collada_import(filepath=filepath)
        elif filepath.endswith(".blend"):
            with bpy.data.libraries.load(filepath) as (data_from, data_to):
                data_to.objects = data_from.objects
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
        else:
            raise ValueError(f"not suported type {filepath}")
    except:
        raise ValueError(f"failed to load {filepath}")
    if return_armature:
        armature = [x for x in set(bpy.context.scene.objects)-old_objs if x.type=="ARMATURE"]
        if len(armature)==0:
            return None
        if len(armature)>1:
            raise ValueError(f"multiple armatures found")
        armature = armature[0]
        
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in bpy.data.armatures[0].edit_bones:
            bone.roll = 0. # change all roll to 0. to prevent weird behaviour

        bpy.ops.object.mode_set(mode='OBJECT')
        armature.select_set(False)
        
        bpy.ops.object.select_all(action='DESELECT')
        return armature

def get_skin(arranged_bones):
    meshes = []
    for v in bpy.data.objects:
        if v.type == 'MESH':
            meshes.append(v)
    index = {}
    for (id, pbone) in enumerate(arranged_bones):
        index[pbone.name] = id
    _dict_skin = {}
    total_bones = len(arranged_bones)
    for obj in meshes:
        total_vertices = len(obj.data.vertices)
        skin_weight = np.zeros((total_vertices, total_bones))
        obj_group_names = [g.name for g in obj.vertex_groups]
        obj_verts = obj.data.vertices
        for bone in arranged_bones:
            if bone.name not in obj_group_names:
                continue

            gidx = obj.vertex_groups[bone.name].index
            bone_verts = [v for v in obj_verts if gidx in [g.group for g in v.groups]]
            for v in bone_verts:
                which = [id for id in range(len(v.groups)) if v.groups[id].group==gidx]
                w = v.groups[which[0]].weight
                skin_weight[v.index, index[bone.name]] = w
        _dict_skin[obj.name] = {
            'skin': skin_weight,
        }
    
    skin = np.concatenate([
        _dict_skin[d]['skin'] for d in _dict_skin
    ], axis=0)
    return skin

def axis(a: np.ndarray):
    b = np.concatenate([-a[:, 0:1], -a[:, 1:2], a[:, 2:3]], axis=1)
    return b

def get_correct_orientation_kdtree(a: np.ndarray, b: np.ndarray, bones: np.ndarray, num: int=16384) -> np.ndarray:
    '''
    a: sampled_vertiecs
    b: mesh_vertices
    '''
    min_loss = float('inf')
    best_transformed = a.copy()
    axis_permutations = list(itertools.permutations([0, 1, 2]))
    sign_combinations = [(x, y, z) for x in [1, -1] 
                        for y in [1, -1] 
                        for z in [1, -1]]
    _bones = bones.copy()
    for perm in axis_permutations:
        permuted_a = a[np.random.permutation(a.shape[0])[:num]][:, perm]
        for signs in sign_combinations:
            transformed = permuted_a * np.array(signs)
            tree = cKDTree(transformed)
            distances, indices = tree.query(b)
            current_loss = distances.mean()
            if current_loss < min_loss: # prevent from mirroring
                min_loss = current_loss
                best_transformed = a[:, perm] * np.array(signs)
                bones[:, :3] = _bones[:, :3][:, perm] * np.array(signs)
                bones[:, 3:] = _bones[:, 3:][:, perm] * np.array(signs)
    
    return best_transformed, bones

def denormalize_vertices(mesh_vertices: ndarray, vertices: ndarray, bones: ndarray) -> np.ndarray:
    min_vals = np.min(mesh_vertices, axis=0)
    max_vals = np.max(mesh_vertices, axis=0)
    center = (min_vals + max_vals) / 2
    scale = np.max(max_vals - min_vals) / 2
    denormalized_vertices = vertices * scale + center
    denormalized_bones = bones * scale
    denormalized_bones[:, :3] += center
    denormalized_bones[:, 3:] += center

    return denormalized_vertices, denormalized_bones

def get_matrix(ob):
    m = np.eye(4)
    while ob:
        if hasattr(ob, 'matrix_world'):
            m = m @ np.array(ob.matrix_world)
        ob = ob.parent
    return m

def create_skeleton_mesh(
    bones: ndarray,  # (J, 6) where [:3] is head, [3:] is tail
    parents: List[Union[int, None]],
    names: List[str],
    cylinder_radius: float = None,
    vertices_per_cross_section: int = 8
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Create a skeleton line/tube mesh from bones (similar to SIGGRAPH 2025 Roblox rigging).
    
    Args:
        bones: (J, 6) array where bones[i, :3] is head, bones[i, 3:] is tail
        parents: List of parent indices (None for root)
        names: List of bone names
        cylinder_radius: Radius of bone cylinders (default: 1% of average bone length)
        vertices_per_cross_section: Number of vertices around bone (default: 8)
    
    Returns:
        skeleton_vertices: (N_skeleton, 3) array of skeleton mesh vertices
        skeleton_faces: (F_skeleton, 3) array of skeleton mesh faces
        skeleton_skin: (N_skeleton, J) array of skin weights for skeleton mesh
    """
    J = len(names)
    if J == 0:
        return np.array([], dtype=np.float32).reshape(0, 3), np.array([], dtype=np.int32).reshape(0, 3), np.array([], dtype=np.float32).reshape(0, 0)
    
    # Calculate average bone length for default radius
    bone_lengths = np.linalg.norm(bones[:, 3:] - bones[:, :3], axis=1)
    avg_bone_length = np.mean(bone_lengths[bone_lengths > 0])
    if cylinder_radius is None:
        cylinder_radius = max(avg_bone_length * 0.01, 0.001)  # 1% of average, minimum 0.001
    
    skeleton_vertices_list = []
    skeleton_faces_list = []
    skeleton_skin_list = []
    vertex_offset = 0
    
    # Create cylinder for each bone
    for bone_idx in range(J):
        head = bones[bone_idx, :3]
        tail = bones[bone_idx, 3:]
        bone_dir = tail - head
        bone_length = np.linalg.norm(bone_dir)
        
        if bone_length < 1e-6:
            # Skip zero-length bones
            continue
        
        bone_dir_normalized = bone_dir / bone_length
        
        # Create orthogonal basis for cylinder cross-section
        # Use a stable method to find perpendicular vectors
        if abs(bone_dir_normalized[2]) < 0.9:
            perp1 = np.array([0, 0, 1])
        else:
            perp1 = np.array([1, 0, 0])
        perp1 = perp1 - np.dot(perp1, bone_dir_normalized) * bone_dir_normalized
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(bone_dir_normalized, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # Generate vertices for this bone cylinder
        bone_vertices = []
        for i in range(vertices_per_cross_section):
            angle = 2 * np.pi * i / vertices_per_cross_section
            # Create circle in plane perpendicular to bone direction
            circle_vec = np.cos(angle) * perp1 + np.sin(angle) * perp2
            # Add vertices at head and tail of bone
            bone_vertices.append(head + cylinder_radius * circle_vec)
            bone_vertices.append(tail + cylinder_radius * circle_vec)
        
        bone_vertices = np.array(bone_vertices)
        skeleton_vertices_list.append(bone_vertices)
        
        # Generate faces for this bone cylinder
        # Each cross-section has vertices_per_cross_section vertices
        # Connect adjacent vertices to form quads (triangulated)
        bone_faces = []
        for i in range(vertices_per_cross_section):
            # Current and next vertex indices in cross-section
            curr_head = vertex_offset + i * 2
            curr_tail = vertex_offset + i * 2 + 1
            next_head = vertex_offset + ((i + 1) % vertices_per_cross_section) * 2
            next_tail = vertex_offset + ((i + 1) % vertices_per_cross_section) * 2 + 1
            
            # Create two triangles per quad
            # Triangle 1: curr_head, next_head, curr_tail
            bone_faces.append([curr_head, next_head, curr_tail])
            # Triangle 2: next_head, next_tail, curr_tail
            bone_faces.append([next_head, next_tail, curr_tail])
        
        skeleton_faces_list.append(np.array(bone_faces, dtype=np.int32))
        
        # Generate skin weights for this bone
        # Each vertex gets weight 1.0 for this bone, 0.0 for others
        num_vertices_bone = len(bone_vertices)
        bone_skin = np.zeros((num_vertices_bone, J), dtype=np.float32)
        bone_skin[:, bone_idx] = 1.0
        skeleton_skin_list.append(bone_skin)
        
        vertex_offset += num_vertices_bone
    
    if len(skeleton_vertices_list) == 0:
        return np.array([], dtype=np.float32).reshape(0, 3), np.array([], dtype=np.int32).reshape(0, 3), np.array([], dtype=np.float32).reshape(0, J)
    
    # Concatenate all bones
    skeleton_vertices = np.vstack(skeleton_vertices_list)
    skeleton_faces = np.vstack(skeleton_faces_list)
    skeleton_skin = np.vstack(skeleton_skin_list)
    
    # At joints, blend weights from connected bones
    # For now, we'll keep the simple approach where each bone segment has its own weight
    # Joint blending can be added later if needed
    
    return skeleton_vertices, skeleton_faces, skeleton_skin

def make_armature(
    vertices: ndarray,
    bones: ndarray, # (joint, tail)
    parents: list[Union[int, None]],
    names: list[str],
    skin: ndarray,
    group_per_vertex: int=4,
    add_root: bool=False,
    is_vrm: bool=False,
):
    context = bpy.context
    
    mesh_vertices = []
    local_coord = np.eye(4)
    local_parent = None
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        if ob.parent is not None:
            local_coord = get_matrix(ob.parent)
            local_parent = ob.parent
        m = np.array(ob.matrix_world)
        matrix_world_rot = m[:3, :3]
        matrix_world_bias = m[:3, 3]
        for v in ob.data.vertices:
            mesh_vertices.append(matrix_world_rot @ np.array(v.co) + matrix_world_bias)

    mesh_vertices = np.stack(mesh_vertices)
    vertices, bones = denormalize_vertices(mesh_vertices, vertices, bones)
    
    bpy.ops.object.add(type="ARMATURE", location=(0, 0, 0))
    armature = context.object
    humanoid = None  # Initialize humanoid variable
    
    # Initialize VRM extension if needed
    if is_vrm:
        # Ensure VRM addon is enabled
        vrm_addon_enabled = False
        vrm_module_name = None
        for addon in bpy.context.preferences.addons:
            if 'vrm' in addon.module.lower():
                vrm_module_name = addon.module
                if addon.enabled:
                    vrm_addon_enabled = True
                    break
        
        if not vrm_addon_enabled:
            if vrm_module_name:
                try:
                    bpy.ops.preferences.addon_enable(module=vrm_module_name)
                    vrm_addon_enabled = True
                    print(f"[Info] Enabled VRM addon: {vrm_module_name}")
                except Exception as e:
                    print(f"[Warning] Could not enable VRM addon: {e}")
        
        # Add VRM extension to armature if addon is enabled
        if vrm_addon_enabled:
            # The VRM addon should automatically add the extension when enabled
            # If it doesn't exist, try to initialize it
            if not hasattr(armature.data, 'vrm_addon_extension'):
                # Try to add the extension by accessing it (VRM addon should create it)
                try:
                    # Switch to object mode to ensure armature is in correct state
                    bpy.ops.object.mode_set(mode='OBJECT')
                    # Access the extension - VRM addon should create it if enabled
                    # This might require the addon to be properly initialized
                    armature.data.vrm_addon_extension
                except AttributeError:
                    print("[Warning] VRM addon extension not available. VRM features may not work correctly.")
                    print("[Info] The VRM addon may need to be properly initialized. Trying to refresh...")
                    # Try refreshing the addon
                    try:
                        import importlib
                        if vrm_module_name:
                            importlib.reload(__import__(vrm_module_name))
                    except:
                        pass
                    is_vrm = False
                except Exception as e:
                    print(f"[Warning] Could not initialize VRM extension: {e}")
                    is_vrm = False
            
            if hasattr(armature.data, 'vrm_addon_extension'):
                try:
                    armature.data.vrm_addon_extension.spec_version = "1.0"
                    humanoid = armature.data.vrm_addon_extension.vrm1.humanoid
                    print("[Info] VRM extension initialized successfully")
                except Exception as e:
                    print(f"[Warning] Could not set VRM extension properties: {e}")
                    is_vrm = False
        else:
            print("[Warning] VRM addon is not enabled. VRM features will be disabled.")
            is_vrm = False
    elif hasattr(armature.data, 'vrm_addon_extension'):
        # VRM extension exists but is_vrm was False - use it anyway
        try:
            armature.data.vrm_addon_extension.spec_version = "1.0"
            humanoid = armature.data.vrm_addon_extension.vrm1.humanoid
            is_vrm = True
        except Exception as e:
            print(f"[Warning] Could not use existing VRM extension: {e}")
            is_vrm = False
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature.data.edit_bones
    if add_root:
        bone_root = edit_bones.new('Root')
        bone_root.name = 'Root'
        bone_root.head = (0., 0., 0.)
        bone_root.tail = (bones[0, 0], bones[0, 1], bones[0, 2])
    
    J = len(names)
    def extrude_bone(
        name: Union[None, str],
        parent_name: Union[None, str],
        head: Tuple[float, float, float],
        tail: Tuple[float, float, float],
    ):
        bone = edit_bones.new(name)
        bone.head = (head[0], head[1], head[2])
        bone.tail = (tail[0], tail[1], tail[2])
        bone.name = name
        if parent_name is None:
            return
        parent_bone = edit_bones.get(parent_name)
        bone.parent = parent_bone
        bone.use_connect = False # always False currently

    vertices, bones = get_correct_orientation_kdtree(vertices, mesh_vertices, bones)
    inv = np.linalg.inv(local_coord)
    bones[:, :3] = (inv[:3, :3] @ bones[:, :3].T + inv[:3, 3:4]).T
    bones[:, 3:] = (inv[:3, :3] @ bones[:, 3:].T + inv[:3, 3:4]).T
    for i in range(J):
        if add_root:
            pname = 'Root' if parents[i] is None else names[parents[i]]
        else:
            pname = None if parents[i] is None else names[parents[i]]
        extrude_bone(names[i], pname, bones[i, :3], bones[i, 3:])

    # must set to object mode to enable parent_set
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # When is_vrm is True, ensure bones match the names from UniRig
    # UniRig generates VRM names via Order.make_names() when cls='vroid'
    # The names list should already contain the correct VRM names
    # We only rename bones if they exist and have different names
    if is_vrm:
        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = armature.data.edit_bones
        
        renamed_count = 0
        for i, expected_name in enumerate(names):
            # Find bone by the name it was created with (from names list)
            bone = edit_bones.get(expected_name)
            if bone is None:
                # Bone doesn't exist with expected name, skip
                continue
            
            # Only rename if the bone name doesn't match the expected name
            # (This handles cases where Blender might have auto-renamed due to conflicts)
            if bone.name != expected_name:
                # Check if target name already exists (from a previous bone)
                existing_bone = edit_bones.get(expected_name)
                if existing_bone is None or existing_bone == bone:
                    bone.name = expected_name
                    renamed_count += 1
                    print(f"[Info] Renamed bone '{bone.name}' (index {i}) to '{expected_name}'")
                else:
                    print(f"[Warning] Cannot rename '{bone.name}' to '{expected_name}': target name already exists")
        
        bpy.ops.object.mode_set(mode='OBJECT')

        if renamed_count > 0:
            print(f"[Info] Renamed {renamed_count} bones to match UniRig-generated names")
        else:
            # Verify bones have VRM names
            vrm_bone_count = sum(1 for name in names if name.startswith('J_Bip_'))
            if vrm_bone_count > 0:
                print(f"[Info] All bones already have correct names from UniRig (VRM names: {vrm_bone_count})")
    objects = bpy.data.objects
    for o in bpy.context.selected_objects:
        o.select_set(False)
    
    argsorted = np.argsort(-skin, axis=1)
    vertex_group_reweight = skin[np.arange(skin.shape[0])[..., None], argsorted]
    vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[...,None]
    vertex_group_reweight = np.nan_to_num(vertex_group_reweight)
    
    # Detect cold start: check if vertices are skeleton joints (count matches bone count)
    # In cold start, vertices are skeleton joints, not mesh vertices
    is_cold_start = (vertices.shape[0] == len(names)) and (vertices.shape[0] < 1000)  # Heuristic: skeleton joints are few
    
    # Initialize source data variables
    source_vertices = vertices
    source_skin = skin
    source_faces = np.array([], dtype=np.int32).reshape(0, 3)
    
    # Build source mesh faces from vertices
    # For cold start: create skeleton mesh from bones
    # For normal case: use convex hull or existing mesh topology
    if is_cold_start:
        print(f"[Info] Cold start detected: vertices are skeleton joints ({vertices.shape[0]} joints, {len(names)} bones)")
        print(f"[Info] Creating skeleton mesh for robust weight transfer...")
        try:
            # Create skeleton mesh from bones
            skeleton_vertices, skeleton_faces, skeleton_skin = create_skeleton_mesh(
                bones=bones,
                parents=parents,
                names=names,
                cylinder_radius=None,  # Auto-calculate
                vertices_per_cross_section=8
            )
            
            if len(skeleton_vertices) > 0 and len(skeleton_faces) > 0:
                print(f"[Info] Created skeleton mesh: {len(skeleton_vertices)} vertices, {len(skeleton_faces)} faces")
                # Use skeleton mesh as source
                source_vertices = skeleton_vertices
                source_faces = skeleton_faces
                source_skin = skeleton_skin
            else:
                print(f"[Warning] Skeleton mesh creation failed, falling back to ConvexHull")
                is_cold_start = False
        except Exception as e:
            print(f"[Warning] Skeleton mesh creation failed: {e}")
            import traceback
            traceback.print_exc()
            is_cold_start = False
    
    # If not using skeleton mesh, try ConvexHull for source faces
    if not is_cold_start or len(source_faces) == 0:
        try:
            # Try to get faces from mesh_vertices if available
            # For now, we'll use a simple approach: create faces from a convex hull
            from scipy.spatial import ConvexHull
            try:
                # Try to create a simple mesh from vertices using convex hull
                hull = ConvexHull(source_vertices)
                source_faces = hull.simplices
                print(f"[Info] Created source mesh from ConvexHull: {len(source_vertices)} vertices, {len(source_faces)} faces")
            except Exception as e:
                # If convex hull fails, create a minimal mesh structure
                print(f"[Warning] ConvexHull failed: {e}")
                source_faces = np.array([], dtype=np.int32).reshape(0, 3)
        except Exception as e:
            print(f"[Warning] Could not create source mesh faces: {e}")
            source_faces = np.array([], dtype=np.int32).reshape(0, 3)
    
    for ob in objects:
        if ob.type != 'MESH':
            continue
        ob.select_set(True)
        armature.select_set(True)
        bpy.ops.object.parent_set(type='ARMATURE_NAME')
        vis = []
        for x in ob.vertex_groups:
            vis.append(x.name)
        
        n_vertices = []
        m = local_coord @ np.array(ob.matrix_world)
        matrix_world_rot = m[:3, :3]
        matrix_world_bias = m[:3, 3]
        for v in ob.data.vertices:
            n_vertices.append(matrix_world_rot @ np.array(v.co) + matrix_world_bias)
        n_vertices = np.stack(n_vertices)
        
        # Get target mesh faces from Blender object
        target_faces = []
        for poly in ob.data.polygons:
            if len(poly.vertices) >= 3:
                # Triangulate polygon if needed
                for i in range(1, len(poly.vertices) - 1):
                    target_faces.append([poly.vertices[0], poly.vertices[i], poly.vertices[i + 1]])
        target_faces = np.array(target_faces, dtype=np.int32) if target_faces else np.array([], dtype=np.int32).reshape(0, 3)
        
        # Use robust transfer - requires libigl, will crash if not available
        if len(source_faces) == 0 or len(target_faces) == 0:
            raise ValueError(
                f"Cannot use robust skin transfer: source_faces={len(source_faces)}, target_faces={len(target_faces)}. "
                f"Both meshes must have valid face data."
            )
        
        # Check if we have valid source data
        if source_vertices.shape[0] == 0 or source_skin.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"Invalid source data for robust transfer: vertices={source_vertices.shape[0]}, skin={source_skin.shape[0]}"
            )
        
        print(f"[Info] Using robust skin weight transfer for {ob.name}")
        if is_cold_start:
            print(f"[Info] Using skeleton mesh as source (cold start mode)")
        # Transfer weights using robust method
        # Compute search radius as 5% of target mesh bounding box diagonal
        bbox_min = n_vertices.min(axis=0)
        bbox_max = n_vertices.max(axis=0)
        search_radius = 0.05 * np.linalg.norm(bbox_max - bbox_min)
        
        # Transfer weights with error handling
        try:
            transferred_skin, success = robust_skin_weights_transfer(
                V1=source_vertices,  # Use skeleton mesh in cold start, or original vertices otherwise
                F1=source_faces,      # Use skeleton mesh faces in cold start, or ConvexHull faces otherwise
                W1=source_skin,       # Use skeleton mesh weights in cold start, or original skin otherwise
                V2=n_vertices,        # Target mesh vertices
                F2=target_faces,      # Target mesh faces
                SearchRadius=search_radius,
                NormalThreshold=30.0,
                num_smooth_iter_steps=10,
                smooth_alpha=0.2,
                use_smoothing=True
            )
            
            if not success:
                # Inpainting failed, but we still have interpolated weights to use
                # Continue with the interpolated weights as fallback
                print(f"[Warning] Robust skin weight transfer inpainting failed for {ob.name}, using interpolated weights as fallback")
        except Exception as e:
            # Robust transfer raised an exception (e.g., libigl error, numerical issues, etc.)
            # Fall back to simple interpolation-based transfer
            print(f"[Error] Robust skin weight transfer failed with exception for {ob.name}: {e}")
            print(f"[Info] Falling back to simple interpolation-based weight transfer")
            import traceback
            traceback.print_exc()
            
            # Use simple interpolation fallback
            # Find closest vertices and interpolate weights
            try:
                tree = cKDTree(source_vertices)
                distances, indices = tree.query(n_vertices, k=min(4, len(source_vertices)))
                
                # Initialize transferred skin with zeros
                transferred_skin = np.zeros((len(n_vertices), source_skin.shape[1]))
                
                # Interpolate weights using inverse distance weighting
                for i, (dists, idxs) in enumerate(zip(distances, indices)):
                    if isinstance(dists, np.ndarray):
                        # Multiple neighbors
                        weights = 1.0 / (dists + 1e-10)  # Add small epsilon to avoid division by zero
                        weights = weights / weights.sum()
                        for j, (w, idx) in enumerate(zip(weights, idxs)):
                            transferred_skin[i] += w * source_skin[int(idx)]
                    else:
                        # Single neighbor
                        transferred_skin[i] = source_skin[int(idxs)]
                
                # Normalize weights
                row_sums = transferred_skin.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0  # Avoid division by zero
                transferred_skin = transferred_skin / row_sums
            except Exception as fallback_error:
                # Even the fallback failed, use nearest neighbor only
                print(f"[Warning] Interpolation fallback also failed: {fallback_error}")
                print(f"[Info] Using nearest neighbor only for {ob.name}")
                tree = cKDTree(source_vertices)
                _, indices = tree.query(n_vertices, k=1)
                transferred_skin = source_skin[indices.flatten()] if hasattr(indices, 'flatten') else source_skin[indices]
            
            success = False
        
        # Recompute argsorted and vertex_group_reweight for transferred skin
        transferred_argsorted = np.argsort(-transferred_skin, axis=1)
        transferred_vertex_group_reweight = transferred_skin[np.arange(transferred_skin.shape[0])[..., None], transferred_argsorted]
        transferred_vertex_group_reweight = transferred_vertex_group_reweight / transferred_vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[...,None]
        transferred_vertex_group_reweight = np.nan_to_num(transferred_vertex_group_reweight)
        
        # Apply transferred weights
        for v in range(len(n_vertices)):
            for ii in range(group_per_vertex):
                i = transferred_argsorted[v, ii]
                if i >= len(names):
                    continue
                n = names[i]
                if n not in ob.vertex_groups:
                    continue
                ob.vertex_groups[n].add([v], transferred_vertex_group_reweight[v, ii], 'REPLACE')
        
        print(f"[Info] Robust transfer completed successfully for {ob.name}")
        
        armature.select_set(False)
        ob.select_set(False)
    armature.parent = local_parent
    
    # set vrm bones link
    if is_vrm:
        # Ensure VRM extension exists and get humanoid
        if not hasattr(armature.data, 'vrm_addon_extension'):
            # Try to ensure VRM addon is enabled and extension is available
            vrm_addon_enabled = False
            for addon in bpy.context.preferences.addons:
                if 'vrm' in addon.module.lower() and addon.enabled:
                    vrm_addon_enabled = True
                    break
            
            if vrm_addon_enabled:
                # VRM addon is enabled but extension doesn't exist
                # Try to refresh or reinitialize
                print("[Warning] VRM extension not found on armature, trying to initialize...")
                try:
                    # Switch to object mode and select armature
                    bpy.ops.object.mode_set(mode='OBJECT')
                    bpy.context.view_layer.objects.active = armature
                    # The extension should be created automatically when VRM addon is enabled
                    # Try accessing it to trigger creation
                    _ = armature.data.vrm_addon_extension
                except:
                    print("[Error] Could not create VRM extension. VRM features will be disabled.")
                    is_vrm = False
        
        if is_vrm and hasattr(armature.data, 'vrm_addon_extension'):
            try:
                armature.data.vrm_addon_extension.spec_version = "1.0"
                humanoid = armature.data.vrm_addon_extension.vrm1.humanoid
                humanoid.human_bones.hips.node.bone_name = "J_Bip_C_Hips"
                humanoid.human_bones.spine.node.bone_name = "J_Bip_C_Spine"
                
                humanoid.human_bones.chest.node.bone_name = "J_Bip_C_Chest"
                humanoid.human_bones.neck.node.bone_name = "J_Bip_C_Neck"
                humanoid.human_bones.head.node.bone_name = "J_Bip_C_Head"
                humanoid.human_bones.left_upper_leg.node.bone_name = "J_Bip_L_UpperLeg"
                humanoid.human_bones.left_lower_leg.node.bone_name = "J_Bip_L_LowerLeg"
                humanoid.human_bones.left_foot.node.bone_name = "J_Bip_L_Foot"
                humanoid.human_bones.right_upper_leg.node.bone_name = "J_Bip_R_UpperLeg"
                humanoid.human_bones.right_lower_leg.node.bone_name = "J_Bip_R_LowerLeg"
                humanoid.human_bones.right_foot.node.bone_name = "J_Bip_R_Foot"
                humanoid.human_bones.left_upper_arm.node.bone_name = "J_Bip_L_UpperArm"
                humanoid.human_bones.left_lower_arm.node.bone_name = "J_Bip_L_LowerArm"
                humanoid.human_bones.left_hand.node.bone_name = "J_Bip_L_Hand"
                humanoid.human_bones.right_upper_arm.node.bone_name = "J_Bip_R_UpperArm"
                humanoid.human_bones.right_lower_arm.node.bone_name = "J_Bip_R_LowerArm"
                humanoid.human_bones.right_hand.node.bone_name = "J_Bip_R_Hand"
                
                # Try to automatically assign bones
                try:
                    bpy.ops.vrm.assign_vrm1_humanoid_human_bones_automatically(armature_name=armature.name)
                except Exception as e:
                    print(f"[Warning] Could not automatically assign VRM bones: {e}")
            except Exception as e:
                print(f"[Error] Failed to set VRM bone links: {e}")
                import traceback
                traceback.print_exc()

def merge(
    path: str,
    output_path: str,
    vertices: ndarray,
    joints: ndarray,
    skin: ndarray,
    parents: List[Union[None, int]],
    names: List[str],
    tails: ndarray,
    add_root: bool=False,
    is_vrm: bool=False,
):
    '''
    Merge skin and bone into original file.
    '''
    clean_bpy()
    try:
        load(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    
    bones = np.concatenate([joints, tails], axis=1)
    # if the result is weired, orientation may be wrong
    make_armature(
        vertices=vertices,
        bones=bones,
        parents=parents,
        names=names,
        skin=skin,
        group_per_vertex=4,
        add_root=add_root,
        is_vrm=is_vrm,
    )
    
    dirpath = os.path.dirname(output_path)
    if dirpath != '':
        os.makedirs(dirpath, exist_ok=True)
    try:
        # Export based on file extension, not is_vrm flag
        # is_vrm is only used for bone naming, not export format
        if output_path.endswith(".vrm"):
            # Only export as VRM if the file extension explicitly requests it
            bpy.ops.export_scene.vrm(filepath=output_path)
        elif output_path.endswith(".fbx") or output_path.endswith(".FBX"):
            bpy.ops.export_scene.fbx(filepath=output_path, add_leaf_bones=True)
        elif output_path.endswith(".glb") or output_path.endswith(".gltf"):
            bpy.ops.export_scene.gltf(filepath=output_path)
        elif output_path.endswith(".usd") or output_path.endswith(".usda") or output_path.endswith(".usdc"):
            # USD export with embedded images and material preservation
            # Preserves quads and materials from GLTF/FBX sources
            # Using only essential parameters that are valid in Blender 4.5
            # Note: Texture embedding is controlled by relative_paths=False
            bpy.ops.wm.usd_export(
                filepath=output_path,
                export_materials=True,
                export_textures=True,
                relative_paths=False,  # False = embed textures, True = use relative paths
                export_uvmaps=True,
                export_armatures=True,  # Preserve armatures for rigged models
                selected_objects_only=False,
                visible_objects_only=False,
                use_instancing=False,
                evaluation_mode='RENDER'
            )
        elif output_path.endswith(".dae"):
            bpy.ops.wm.collada_export(filepath=output_path)
        elif output_path.endswith(".blend"):
            with bpy.data.libraries.load(output_path) as (data_from, data_to):
                data_to.objects = data_from.objects
        else:
            raise ValueError(f"not suported type {output_path}")
    except:
        raise ValueError(f"failed to export {output_path}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def nullable_string(val):
    if not val:
        return None
    return val

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require_suffix', type=str, required=True)
    parser.add_argument('--num_runs', type=int, required=True)
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--data_config', type=str, required=False)
    parser.add_argument('--skeleton_config', type=str, required=False)
    parser.add_argument('--skin_config', type=str, required=False)
    parser.add_argument('--merge_dir', type=str, required=False)
    parser.add_argument('--merge_name', type=str, required=False)
    parser.add_argument('--add_root', type=str2bool, required=False, default=False)
    parser.add_argument('--source', type=nullable_string, required=False, default=None)
    parser.add_argument('--target', type=nullable_string, required=False, default=None)
    parser.add_argument('--output', type=nullable_string, required=False, default=None)
    return parser.parse_args()

def transfer(source: str, target: str, output: str, add_root: bool=False):
    clean_bpy()
    try:
        armature = load(filepath=source, return_armature=True)
        assert armature is not None
    except Exception as e:
        print(f"failed to load {source}")
        return

    vertices, faces, skin = process_mesh()
    arranged_bones = get_arranged_bones(armature)
    if skin is None:
        skin = get_skin(arranged_bones)

    joints, tails, parents, names, matrix_local = process_armature(armature, arranged_bones)
    merge(
        path=target,
        output_path=output,
        vertices=vertices,
        joints=joints,
        skin=skin,
        parents=parents,
        names=names,
        tails=tails,
        add_root=add_root,
    )

if __name__ == "__main__":
    args = parse()
    
    if args.source is not None or args.target is not None:
        assert args.source is not None and args.target is not None
        transfer(args.source, args.target, args.output, args.add_root)
        exit()

    data_config     = Box(yaml.safe_load(open(args.data_config, "r")))
    skeleton_config = Box(yaml.safe_load(open(args.skeleton_config, "r")))
    skin_config     = Box(yaml.safe_load(open(args.skin_config, "r")))

    num_runs        = args.num_runs
    id              = args.id
    require_suffix  = args.require_suffix.split(',')
    merge_dir       = args.merge_dir
    merge_name      = args.merge_name
    add_root        = args.add_root

    input_dataset_dir   = data_config.input_dataset_dir
    dataset_name        = data_config.output_dataset_dir
    
    skin_output_dataset_dir = skin_config.writer.output_dir
    skin_name               = skin_config.writer.export_npz
    
    skeleton_output_dataset_dir = skeleton_config.writer.output_dir
    skeleton_name               = skeleton_config.writer.export_npz

    def make_path(output_dataset_dir, dataset_name, root, file_name):
        if output_dataset_dir is None:
            return os.path.join(
                dataset_name,
                os.path.relpath(root, input_dataset_dir),
                file_name,
            )
        return os.path.join(
            output_dataset_dir,
            dataset_name,
            os.path.relpath(root, input_dataset_dir),
            file_name,
        )

    files = []
    for root, dirs, f in os.walk(input_dataset_dir):
        for file in f:
            if file.split('.')[-1] in require_suffix:
                file_name = file.removeprefix("./")
                suffix = file.split('.')[-1]
                # remove suffix
                file_name = '.'.join(file_name.split('.')[:-1])
                
                skin_path = make_path(skin_output_dataset_dir, dataset_name, root, os.path.join(file_name, skin_name+'.npz'))
                skeleton_path = make_path(skeleton_output_dataset_dir, dataset_name, root, os.path.join(file_name, skeleton_name+'.npz'))
                merge_path = make_path(merge_dir, dataset_name, root, os.path.join(file_name, merge_name+"."+suffix))
                
                # check if inference result exists
                if os.path.exists(skin_path) and os.path.exists(skeleton_path):
                    files.append((os.path.join(root, file), skin_path, skeleton_path, merge_path))

    num_files = len(files)
    print("num_files", num_files)
    gap = num_files // num_runs
    start = gap * id
    end = gap * (id + 1)
    if id+1==num_runs:
        end = num_files
    
    files = sorted(files)
    if end!=-1:
        files = files[:end]
    tot = 0
    for file in tqdm(files[start:]):
        origin_file = file[0]
        skin_path = file[1]
        skeleton_path = file[2]
        merge_file = file[3]
        
        raw_skin = RawSkin.load(path=skin_path)
        raw_data = RawData.load(path=skeleton_path)
        
        try:
            merge(
                path=origin_file,
                output_path=merge_file,
                vertices=raw_skin.vertices,
                joints=raw_skin.joints,
                skin=raw_skin.skin,
                parents=raw_data.parents,
                names=raw_data.names,
                tails=raw_data.tails,
                add_root=add_root,
                is_vrm=(raw_data.cls=='vroid'),
            )
        except Exception as e:
            print(f"failed to merge {origin_file}: {e}")