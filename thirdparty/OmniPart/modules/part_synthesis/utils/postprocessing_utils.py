from typing import *
import numpy as np
import torch
import utils3d
# nvdiffrast is required for texture baking
try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False
    dr = None

from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
import igraph
import cv2
from PIL import Image
from .random_utils import sphere_hammersley_sequence
from ..renderers import GaussianRenderer
from ..representations import Strivec, Gaussian, MeshExtractResult

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    """ 
    convert a tensor, in any form / dimension, from rgb space to srgb space 
    Args:
        f (torch.Tensor): input tensor

    """
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb_image(f: torch.Tensor) -> torch.Tensor:
    """ 
    convert an image tensor from rgb space to srgb space 
    Args:
        f (torch.Tensor): input tensor

    """
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=32,
    resolution=128,
    num_views=500,
    debug=False,
    verbose=False
):
    """
    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.

    Args:
        verts (torch.Tensor): Vertices of the mesh. Shape (V, 3).
        faces (torch.Tensor): Faces of the mesh. Shape (F, 3).
        max_hole_size (float): Maximum area of a hole to fill.
        max_hole_nbe (int): Maximum number of boundary edges in a hole to fill.
        resolution (int): Resolution of the rasterization.
        num_views (int): Number of views to rasterize the mesh.
        debug (bool): Whether to output debug information and meshes.
        verbose (bool): Whether to print progress.
    """
    # Ensure inputs are float32 (not float64/double)
    if isinstance(verts, torch.Tensor):
        verts = verts.float()
    else:
        verts = torch.tensor(verts, dtype=torch.float32)
    # nvdiffrast requires int32 for faces, not int64
    if isinstance(faces, torch.Tensor):
        faces = faces.int()  # Convert to int32 (not long/int64)
    else:
        faces = torch.tensor(faces, dtype=torch.int32)  # Use int32 for nvdiffrast
    
    # Move to GPU if not already
    if not verts.is_cuda:
        verts = verts.cuda()
    if not faces.is_cuda:
        faces = faces.cuda()
    
    # Ensure faces are int32 (nvdiffrast requirement)
    if faces.dtype != torch.int32:
        faces = faces.int()
    
    # Construct cameras at uniformly distributed positions on a sphere
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)  # Generate uniformly distributed points on sphere
        # Explicitly convert to float32 to avoid float64 propagation
        yaws.append(float(y))
        pitchs.append(float(p))
    # Ensure float32 (not float64) - convert Python floats explicitly
    yaws = torch.tensor(yaws, dtype=torch.float32, device=verts.device)
    pitchs = torch.tensor(pitchs, dtype=torch.float32, device=verts.device)
    radius = torch.tensor(2.0, dtype=torch.float32, device=verts.device)  # Camera distance from origin
    fov = torch.deg2rad(torch.tensor(40, dtype=torch.float32, device=verts.device))  # Camera field of view
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)  # Create projection matrix
    # Ensure projection is float32
    if projection.dtype != torch.float32:
        projection = projection.float()
    views = []
    for (yaw, pitch) in zip(yaws, pitchs):
        # Calculate camera position from spherical coordinates
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ], dtype=torch.float32, device=verts.device) * radius
        # Create view matrix looking at origin
        view = utils3d.torch.view_look_at(orig, torch.tensor([0, 0, 0], dtype=torch.float32, device=verts.device), torch.tensor([0, 0, 1], dtype=torch.float32, device=verts.device))
        # Ensure view is float32
        if view.dtype != torch.float32:
            view = view.float()
        views.append(view)
    views = torch.stack(views, dim=0)
    # Ensure views stack is float32
    if views.dtype != torch.float32:
        views = views.float()

    # Rasterize mesh from multiple viewpoints to determine visible faces
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    
    if not NVDIFFRAST_AVAILABLE:
        raise ImportError("nvdiffrast is required for hole filling. Please install nvdiffrast.")
    
    # Use utils3d (nvdiffrast) for rasterization
    rastctx = utils3d.torch.RastContext(backend='cuda')
    for i in tqdm(range(views.shape[0]), total=views.shape[0], disable=not verbose, desc='Rasterizing'):
        view = views[i]
        # Render from current viewpoint
        buffers = utils3d.torch.rasterize_triangle_faces(
            rastctx, verts[None], faces, resolution, resolution, view=view, projection=projection
        )
        # Collect face IDs that are visible from this view
        face_id = buffers['face_id'][0][buffers['mask'][0] > 0.95] - 1
        face_id = torch.unique(face_id).long()
        visblity[face_id] += 1
    # Normalize visibility to [0,1] - ensure float32
    visblity = visblity.float() / float(num_views)
    visblity = visblity.float()  # Ensure float32 (not float64)
    
    # Prepare for mincut-based mesh cleaning
    ## Construct edge data structures
    # Ensure faces are int32 for utils3d operations (compute_edges expects int32)
    if faces.dtype != torch.int32:
        faces = faces.int()
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = utils3d.torch.compute_connected_components(faces, edges, face2edge)
    
    ## Identify outer faces (those with high visibility)
    outer_face_indices = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    for i in range(len(connected_components)):
        # Use visibility threshold - faces visible in at least 25-50% of views are considered outer faces
        outer_face_indices[connected_components[i]] = visblity[connected_components[i]] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)
    
    ## Identify inner faces (completely invisible)
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if verbose:
        tqdm.write(f'Found {inner_face_indices.shape[0]} invisible faces out of {faces.shape[0]} total')
    
    # Safety check: if ALL faces are invisible, something is wrong with visibility detection
    # Fail explicitly rather than silently
    if inner_face_indices.shape[0] == faces.shape[0]:
        raise RuntimeError(f'Hole filling failed: All {faces.shape[0]} faces marked as invisible. This indicates a problem with visibility detection (likely camera setup or mesh orientation issue).')
    
    if inner_face_indices.shape[0] == 0:
        return verts, faces
    
    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    # Edge weights based on edge length - used for min-cut algorithm
    # Ensure float32 for all operations
    v0 = verts[dual_edge2edge[:, 0]].float()
    v1 = verts[dual_edge2edge[:, 1]].float()
    dual_edges_weights = torch.norm(v0 - v1, dim=1)
    # Ensure weights are float32 (not float64)
    if dual_edges_weights.dtype != torch.float32:
        dual_edges_weights = dual_edges_weights.float()
    if verbose:
        tqdm.write(f'Dual graph: {dual_edges.shape[0]} edges')

    ## Solve mincut problem using igraph
    ### Construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    # Ensure int32 for edges (igraph expects int)
    dual_edges_np = dual_edges.cpu().numpy().astype(np.int32)
    g.add_edges(dual_edges_np)
    # Ensure float32 for weights (not float64)
    dual_edges_weights_np = dual_edges_weights.cpu().numpy().astype(np.float32)
    g.es['weight'] = dual_edges_weights_np
    
    ### Add source and target nodes
    g.add_vertex('s')
    g.add_vertex('t')
    
    ### Connect invisible faces to source with weight 1
    g.add_edges([(f, 's') for f in inner_face_indices], attributes={'weight': torch.ones(inner_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
    
    ### Connect outer faces to target with weight 1
    g.add_edges([(f, 't') for f in outer_face_indices], attributes={'weight': torch.ones(outer_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
    
    # Safety check: ensure we have both inner and outer faces
    if inner_face_indices.shape[0] == 0 or outer_face_indices.shape[0] == 0:
        raise RuntimeError(f'Hole filling failed: Missing inner faces ({inner_face_indices.shape[0]}) or outer faces ({outer_face_indices.shape[0]}). Cannot perform mincut.')
                
    ### Solve mincut to separate inner from outer faces
    # Ensure float32 for weight calculation (not float64)
    weights = np.array(g.es['weight'], dtype=np.float32) * 1000.0
    cut = g.mincut('s', 't', weights.tolist())
    remove_face_indices = torch.tensor([v for v in cut.partition[0] if v < faces.shape[0]], dtype=torch.long, device=faces.device)
    
    # Safety check: don't remove all faces - fail explicitly
    if remove_face_indices.shape[0] >= faces.shape[0]:
        raise RuntimeError(f'Hole filling failed: Mincut would remove all {faces.shape[0]} faces. This indicates a problem with the mincut algorithm or mesh structure.')
    
    if verbose:
        tqdm.write(f'Mincut solved, start checking the cut')
    
    ### Validate the cut by checking each connected component
    to_remove_cc = utils3d.torch.compute_connected_components(faces[remove_face_indices])
    if debug:
        tqdm.write(f'Number of connected components of the cut: {len(to_remove_cc)}')
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### Check if the connected component has low visibility
        visblity_median = visblity[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f'visblity_median: {visblity_median}')
        if visblity_median > 0.25:
            continue
        
        #### Check if the cutting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(face2edge[remove_face_indices[cc]], return_counts=True)
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)]
        if len(cc_new_boundary_edge_indices) > 0:
            # Group boundary edges into connected components
            # Ensure edges are int32 for utils3d operations
            cc_new_boundary_edges = edges[cc_new_boundary_edge_indices]
            if cc_new_boundary_edges.dtype != torch.int32:
                cc_new_boundary_edges = cc_new_boundary_edges.int()
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(cc_new_boundary_edges)
            # Calculate the center of each boundary loop - ensure float32
            cc_new_boundary_edges_cc_center = [verts[edges[cc_new_boundary_edge_indices[edge_cc]]].mean(dim=1).mean(dim=0).float() for edge_cc in cc_new_boundary_edge_cc]
            cc_new_boundary_edges_cc_area = []
            # Calculate the area of each boundary loop
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = (verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]] - cc_new_boundary_edges_cc_center[i]).float()
                _e2 = (verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]] - cc_new_boundary_edges_cc_center[i]).float()
                cc_new_boundary_edges_cc_area.append(torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5)
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f'Area of the cutting loop: {cc_new_boundary_edges_cc_area}')
            # Skip if any loop is too large
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue
            
        valid_remove_cc.append(cc)
        
    # Generate debug visualizations if requested
    if debug:
        # Ensure float32 for debug visualizations (not float64)
        face_v = verts[faces].mean(dim=1).cpu().numpy().astype(np.float32)
        vis_dual_edges = dual_edges.cpu().numpy().astype(np.int32)
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]  # Blue for inner
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]  # Green for outer
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]  # Magenta for removed by mincut
        if len(valid_remove_cc) > 0:
            vis_colors[remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()] = [255, 0, 0]  # Red for valid removal
        utils3d.io.write_ply('dbg_dual.ply', face_v, edges=vis_dual_edges, vertex_colors=vis_colors)
        
        # Ensure float32/int32 for debug visualizations (not float64)
        vis_verts = verts.cpu().numpy().astype(np.float32)
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy().astype(np.int32)
        utils3d.io.write_ply('dbg_cut.ply', vis_verts, edges=vis_edges)
        
    # Remove the identified faces
    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        # Clean up disconnected vertices
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f'Removed {(~mask).sum()} faces by mincut')
    else:
        if verbose:
            tqdm.write(f'Removed 0 faces by mincut')
    
    # Use meshfix to fill small holes in the mesh        
    mesh = _meshfix.PyTMesh()
    # Ensure numpy arrays are float32 before passing to meshfix
    verts_np = verts.cpu().numpy().astype(np.float32)
    faces_np = faces.cpu().numpy().astype(np.int32)
    mesh.load_array(verts_np, faces_np)
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    # Ensure output from meshfix is converted to float32 (meshfix may return float64)
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    verts, faces = torch.tensor(verts, device='cuda', dtype=torch.float32), torch.tensor(faces, device='cuda', dtype=torch.int32)

    return verts, faces


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    debug: bool = False,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_hole_size (float): Maximum area of a hole to fill.
        fill_holes_max_hole_nbe (int): Maximum number of boundary edges of a hole to fill.
        fill_holes_resolution (int): Resolution of the rasterization.
        fill_holes_num_views (int): Number of views to rasterize the mesh.
        debug (bool): Whether to output debug visualizations.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')
    
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices, faces
    
    # Simplify mesh using meshoptimizer with screen-space arc angle error metric
    # Falls back to trimesh if meshoptimizer is not available
    if simplify and simplify_ratio > 0:
        try:
            # Ensure float32/int32
            vertices = np.asarray(vertices, dtype=np.float32)
            faces = np.asarray(faces, dtype=np.int32)
            
            # Calculate target number of indices (3 per face)
            target_indices = int(faces.shape[0] * simplify_ratio * 3)
            if target_indices < 12:  # Minimum 4 faces = 12 indices
                target_indices = 12
            
            if target_indices >= faces.shape[0] * 3:
                # No decimation needed
                if verbose:
                    tqdm.write(f'Decimation skipped: target indices ({target_indices}) >= current indices ({faces.shape[0] * 3})')
            else:
                # Try to use meshoptimizer with screen-space error
                try:
                    from .meshoptimizer_wrapper import simplify_with_screen_error
                    
                    # Convert faces to flat index array
                    indices = faces.flatten().astype(np.uint32)
                    
                    # Use default camera (origin looking at mesh center)
                    # Screen-space error will be calculated based on mesh extents
                    mesh_center = vertices.mean(axis=0)
                    mesh_extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
                    
                    # Calculate target error based on screen arc angle
                    # For screen-space: error should be relative to viewing distance
                    # Default: 1% of mesh extent, adjusted for screen projection
                    target_error = 0.01  # 1% relative error
                    
                    # Simplify with meshoptimizer
                    result = simplify_with_screen_error(
                        vertices,
                        indices,
                        target_index_count=target_indices,
                        target_error=target_error,
                        camera_position=mesh_center + np.array([0, 0, mesh_extent * 2], dtype=np.float32),
                        camera_direction=np.array([0, 0, -1], dtype=np.float32),
                        fov=40.0,
                        screen_width=1920,
                        screen_height=1080,
                        options=0,
                    )
                    
                    # Handle return value (may be 2 or 3 elements depending on version)
                    if len(result) == 2:
                        new_indices, result_error = result
                        new_vertices = vertices  # Vertices not modified
                    else:
                        new_indices, new_vertices, result_error = result
                    
                    # Convert back to faces
                    new_faces = new_indices.reshape(-1, 3)
                    
                    # Remap vertices if needed (meshoptimizer may not remap, so we keep original)
                    vertices = new_vertices
                    faces = new_faces.astype(np.int32)
                    
                    if verbose:
                        tqdm.write(f'After meshoptimizer decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces (error: {result_error:.6f})')
                
                except (ImportError, Exception) as e:
                    # Fallback to trimesh if meshoptimizer not available
                    if verbose:
                        tqdm.write(f'Meshoptimizer not available ({e}), falling back to trimesh')
                    
                    # Create trimesh object
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    # Calculate target number of faces
                    target_faces = int(faces.shape[0] * simplify_ratio)
                    if target_faces < 4:  # Minimum faces for a valid mesh
                        target_faces = 4
                    # Perform quadric decimation
                    mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
                    vertices = mesh.vertices
                    faces = mesh.faces
                    if verbose:
                        tqdm.write(f'After trimesh decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')
        except Exception as e:
            # Decimation failed, skip it but continue
            if verbose:
                tqdm.write(f'Warning: Decimation failed ({e}), skipping decimation but continuing with hole filling')
            # Keep original vertices/faces for hole filling

    # Remove invisible faces and fill small holes
    # This can work independently of decimation
    if fill_holes:
        # Save current mesh before attempting hole filling
        mesh_before_holes = (vertices.copy(), faces.copy())
        try:
            # Ensure float32 to avoid dtype mismatches
            vertices_t = torch.tensor(vertices, dtype=torch.float32).cuda()
            faces_t = torch.tensor(faces.astype(np.int32), dtype=torch.int32).cuda()
            vertices_t, faces_t = _fill_holes(
                vertices_t, faces_t,
                max_hole_size=fill_holes_max_hole_size,
                max_hole_nbe=fill_holes_max_hole_nbe,
                resolution=fill_holes_resolution,
                num_views=fill_holes_num_views,
                debug=debug,
                verbose=verbose,
            )
            vertices, faces = vertices_t.cpu().numpy(), faces_t.cpu().numpy()
            if verbose:
                tqdm.write(f'After hole filling: {vertices.shape[0]} vertices, {faces.shape[0]} faces')
        except Exception as e:
            # If hole filling fails, continue with mesh before hole filling
            if verbose:
                tqdm.write(f'Warning: Hole filling failed ({e}), continuing without hole filling')
            # Restore mesh before hole filling
            vertices, faces = mesh_before_holes

    return vertices, faces


def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.
    This creates UV coordinates for the mesh that can be used for texture mapping.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    
    Returns:
        tuple: (remapped_vertices, remapped_faces, uvs) where uvs are the texture coordinates
    """

    # Parametrize the mesh using xatlas
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    # Apply the vertex mapping to get the final vertices
    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs


def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal['fast', 'opt'] = 'opt',
    lambda_tv: float = 1e-2,
    verbose: bool = False,
    srgb_space: bool = False,
):
    """
    Bake texture to a mesh from multiple observations.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        texture_size (int): Size of the texture.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of texture baking:
            'fast': Simple weighted averaging of observed colors.
            'opt': Optimization-based texture generation with regularization.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
        
    Returns:
        np.array: The baked texture as an RGB image (H, W, 3)
    """
    # Move data to GPU and ensure float32 (not float64)
    # nvdiffrast requires int32 for faces, not int64
    vertices = torch.tensor(vertices, dtype=torch.float32).cuda()
    faces = torch.tensor(faces.astype(np.int32), dtype=torch.int32).cuda()
    uvs = torch.tensor(uvs, dtype=torch.float32).cuda()
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    masks = [torch.tensor(m>0).bool().cuda() for m in masks]
    views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
    projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]

    if not NVDIFFRAST_AVAILABLE:
        raise ImportError("nvdiffrast is required for texture baking. Please install nvdiffrast.")
    
    if mode == 'fast':
        # Fast texture baking - weighted average of observed colors
        texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).cuda()
        texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).cuda()
        rastctx = utils3d.torch.RastContext(backend='cuda')
        
        # Iterate through each observation and accumulate colors
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(observations), disable=not verbose, desc='Texture baking (fast)'):
            with torch.no_grad():
                # Rasterize the mesh from this viewpoint
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                uv_map = rast['uv'][0].detach().flip(0)  # Flip Y to match texture convention
                mask = rast['mask'][0].detach().bool() & masks[0]  # Only use valid mask pixels
            
            # Map UV coordinates to texture pixels
            # Use bilinear filtering for better quality - not expensive and produces smoother textures
            uv_map_scaled = uv_map * (texture_size - 1)  # Scale to [0, texture_size-1]
            obs = observation[mask]
            uv_map_masked = uv_map_scaled[mask]
            
            # Bilinear sampling for better quality
            u = uv_map_masked[:, 0]
            v = uv_map_masked[:, 1]
            u0 = u.floor().long().clamp(0, texture_size - 1)
            u1 = (u0 + 1).clamp(0, texture_size - 1)
            v0 = v.floor().long().clamp(0, texture_size - 1)
            v1 = (v0 + 1).clamp(0, texture_size - 1)
            
            wu = u - u0.float()
            wv = v - v0.float()
            
            # Convert 2D UV to 1D indices for scattering
            idx00 = u0 + (texture_size - v0 - 1) * texture_size
            idx01 = u0 + (texture_size - v1 - 1) * texture_size
            idx10 = u1 + (texture_size - v0 - 1) * texture_size
            idx11 = u1 + (texture_size - v1 - 1) * texture_size
            
            # Bilinear weights
            w00 = (1 - wu) * (1 - wv)
            w01 = (1 - wu) * wv
            w10 = wu * (1 - wv)
            w11 = wu * wv
            
            # Accumulate colors and weights with bilinear interpolation
            texture = texture.scatter_add(0, idx00.view(-1, 1).expand(-1, 3), obs * w00.view(-1, 1))
            texture = texture.scatter_add(0, idx01.view(-1, 1).expand(-1, 3), obs * w01.view(-1, 1))
            texture = texture.scatter_add(0, idx10.view(-1, 1).expand(-1, 3), obs * w10.view(-1, 1))
            texture = texture.scatter_add(0, idx11.view(-1, 1).expand(-1, 3), obs * w11.view(-1, 1))
            
            texture_weights = texture_weights.scatter_add(0, idx00, w00)
            texture_weights = texture_weights.scatter_add(0, idx01, w01)
            texture_weights = texture_weights.scatter_add(0, idx10, w10)
            texture_weights = texture_weights.scatter_add(0, idx11, w11)

        # Normalize by summed weights
        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255).astype(np.uint8)

        if srgb_space:
            # convert the texture from rgb space to srgb 
            texture = rgb_to_srgb_image(texture) 

        # Fill holes in texture using inpainting
        mask = (texture_weights == 0).cpu().numpy().astype(np.uint8).reshape(texture_size, texture_size)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    elif mode == 'opt':
        # Optimization-based texture baking with total variation regularization
        if not NVDIFFRAST_AVAILABLE:
            raise ImportError("nvdiffrast is required for optimization-based texture baking. Please install nvdiffrast.")
        
        rastctx = utils3d.torch.RastContext(backend='cuda')
        # Flip observations vertically to match rendering coordinate system (matches original OmniPart)
        # Observations are torch tensors at this point (converted from numpy in bake_texture)
        observations = [obs.flip(0) for obs in observations]  # Flip Y for rendering
        masks = [m.flip(0) for m in masks]
        
        # Precompute UV maps for efficiency
        _uv = []
        _uv_dr = []
        for view_idx, (observation, view, projection) in enumerate(tqdm(zip(observations, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV')):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                # Do NOT flip UV maps in opt mode - only flip final texture output (matches original OmniPart)
                # dr.texture expects (B, H, W, 2) shape with batch dimension
                _uv.append(rast['uv'].detach())
                _uv_dr.append(rast['uv_dr'].detach())  # Gradient information for differentiable rendering

        # Initialize texture as a learnable parameter
        # Use small random values instead of zeros to break symmetry and allow gradients to flow
        texture = torch.nn.Parameter(torch.randn((1, texture_size, texture_size, 3), dtype=torch.float32).cuda() * 0.01)
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        # Learning rate scheduling functions
        def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            """Exponential learning rate annealing"""
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            """Cosine learning rate annealing"""
            return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
        
        def tv_loss(texture):
            """Total variation loss for regularization"""
            return torch.nn.functional.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :]) + \
                   torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])
    
        # Optimization loop
        # Increased to 1500 iterations for higher quality optimization (texture size reduced to 512 for efficiency)
        total_steps = 1500
        
        with tqdm(total=total_steps, disable=not verbose, desc='Texture baking (opt): optimizing') as pbar:
            # Debug: Print initial statistics
            if verbose and len(_uv) > 0:
                uv_sample = _uv[0]
                obs_sample = observations[0]
                mask_sample = masks[0]
                if len(uv_sample.shape) == 4:
                    uv_sample = uv_sample.squeeze(0)
                print(f"[DEBUG] UV range: [{uv_sample.min():.3f}, {uv_sample.max():.3f}], shape: {uv_sample.shape}")
                print(f"[DEBUG] Observation range: [{obs_sample.min():.3f}, {obs_sample.max():.3f}], mean: {obs_sample.mean():.3f}, shape: {obs_sample.shape}")
                print(f"[DEBUG] Mask coverage: {mask_sample.sum().item()}/{mask_sample.numel()} ({100*mask_sample.sum().item()/mask_sample.numel():.1f}%)")
                print(f"[DEBUG] Texture init range: [{texture.min():.3f}, {texture.max():.3f}], mean: {texture.mean():.3f}")
            
            for step in range(total_steps):
                optimizer.zero_grad()
                # Random sample a view for stochastic optimization
                selected = np.random.randint(0, len(views))
                uv, uv_dr, observation, mask = _uv[selected], _uv_dr[selected], observations[selected], masks[selected]
                # Differentiable rendering of texture
                render = dr.texture(texture, uv, uv_dr)[0]
                # Loss calculation - L1 reconstruction loss + TV regularization
                if mask.sum() > 0:
                    loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                else:
                    # Skip if no valid pixels
                    pbar.update()
                    continue
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)
                loss_value = loss.item()
                loss.backward()
                optimizer.step()
                # Learning rate annealing
                optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, total_steps, 1e-2, 1e-5)
                pbar.set_postfix({'loss': loss_value})
                pbar.update()
                
                # Clear memory periodically to prevent OOM
                if (step + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        
        if srgb_space:
            # convert the texture from rgb space to srgb 
            texture = rgb_to_srgb_image(texture)
        
        # Convert optimized texture to numpy array
        texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        # Fill any remaining holes in the texture
        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
        )['mask'][0].detach().cpu().numpy().astype(np.uint8)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return texture


def to_glb(
    app_rep: Union[Strivec, Gaussian, None],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 2048,
    debug: bool = False,
    verbose: bool = True,
    textured: bool = True,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_size (float): Maximum area of a hole to fill.
        texture_size (int): Size of the texture.
        debug (bool): Whether to print debug information.
        verbose (bool): Whether to print progress.
        
    Returns:
        trimesh.Trimesh: The processed mesh with texture, ready for GLB export
    """
    # Extract mesh data from the result - ensure float32/int32 (not float64)
    vertices = mesh.vertices.detach().cpu().numpy().astype(np.float32)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32)
    
    # Apply mesh post-processing
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),  # Scale hole size by mesh complexity
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=debug,
        verbose=verbose,
    )
    
    # If postprocess_mesh failed completely, try with just decimation (no hole filling)
    if vertices is None or faces is None:
        if verbose:
            print("[WARN] postprocess_mesh failed, retrying with simplified processing (no hole filling)...")
        # Retry with just decimation, no hole filling
        vertices, faces = postprocess_mesh(
            mesh.vertices.detach().cpu().numpy(),  # Use original vertices
            mesh.faces.detach().cpu().numpy(),      # Use original faces
            simplify=simplify > 0,
            simplify_ratio=simplify,
            fill_holes=False,  # Disable hole filling
            fill_holes_max_hole_size=fill_holes_max_size,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),
            fill_holes_resolution=1024,
            fill_holes_num_views=1000,
            debug=debug,
            verbose=verbose,
        )
        # If still None, use original vertices/faces
        if vertices is None or faces is None:
            if verbose:
                print("[WARN] postprocess_mesh failed again, using original mesh without post-processing")
            # Use original mesh data - ensure float32/int32 (not float64)
            vertices = mesh.vertices.detach().cpu().numpy().astype(np.float32)
            faces = mesh.faces.detach().cpu().numpy().astype(np.int32)

    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return None

    texture = None
    uvs = None
    vertices_original = vertices.copy()  # Save original vertices for final mesh export
    
    if textured:
        # Check if app_rep is available (needed for texture baking)
        if app_rep is None:
            if verbose:
                print("[WARN] app_rep (gaussian/strivec) is None. Cannot bake texture. Skipping texture baking.")
            textured = False
        
    if textured:
        # Check if texture baking is possible
        if not NVDIFFRAST_AVAILABLE:
            if verbose:
                print("[WARN] nvdiffrast is not available. Texture baking disabled.")
            textured = False
        else:
            try:
                # Test if utils3d rasterization works
                test_ctx = utils3d.torch.RastContext(backend='cuda')
                del test_ctx
                if verbose:
                    print("[INFO] Using utils3d (nvdiffrast) for texture baking")
            except Exception as e:
                if verbose:
                    print(f"[WARN] utils3d rasterization unavailable: {e}. Texture baking disabled.")
                textured = False
        
        # Proceed with texture baking if backend is available
        if textured:
            # Create UV mapping for the mesh (matches original OmniPart implementation)
            # parametrize_mesh may remap vertices and reindex faces
            vertices, faces, uvs = parametrize_mesh(vertices, faces)
            if verbose:
                print(f"[DEBUG] After parametrize_mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces, {uvs.shape[0]} UVs")

            # Render single view from the appearance representation for texturing
            # Use front-facing view (yaw=0, pitch=0)
            from .render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, render_frames
            extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [0], [2], [40])
            res = render_frames(app_rep, extrinsics, intrinsics, {'resolution': 2048, 'bg_color': (0, 0, 0)}, verbose=False)
            observations = [res['color'][0]]
            # Create mask from the rendered image
            masks = [np.any(observations[0] > 0, axis=-1)]
            # Convert camera parameters to numpy
            extrinsics = [extrinsics[0].cpu().numpy()]
            intrinsics = [intrinsics[0].cpu().numpy()]
            
            # Bake texture from the rendered views onto the mesh
            # Use vertices as-is (no normalization) - mesh and Gaussian should already be in same coordinate system
            # Reduce texture size if GPU memory is low to prevent OOM
            try:
                import torch
                if torch.cuda.is_available():
                    # Check available GPU memory
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_memory_gb = free_memory / (1024**3)
                    # If less than 5GB free, reduce texture size from 2048 to 1024
                    if free_memory_gb < 5.0 and texture_size > 1024:
                        if verbose:
                            print(f"[INFO] Low GPU memory ({free_memory_gb:.2f} GB free), reducing texture size from {texture_size} to 1024")
                        texture_size = 1024
                    elif free_memory_gb < 2.0 and texture_size > 512:
                        if verbose:
                            print(f"[INFO] Very low GPU memory ({free_memory_gb:.2f} GB free), reducing texture size from {texture_size} to 512")
                        texture_size = 512
                        if verbose:
                            print(f"[INFO] Very low GPU memory ({free_memory_gb:.2f} GB free), reducing texture size from {texture_size} to 512")
                        texture_size = 512
                
                texture = bake_texture(
                    vertices, faces, uvs,
                    observations, masks, extrinsics, intrinsics,
                    texture_size=texture_size, mode='opt',  # Use optimization-based texturing
                    lambda_tv=0.01,  # Total variation regularization
                    verbose=verbose
                )
                texture = Image.fromarray(texture)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Texture baking failed: {e}")
                    print("[INFO] Falling back to untextured mesh")
                    import traceback
                    traceback.print_exc()
                textured = False
                texture = None
                uvs = None
    # Convert from z-up to y-up coordinate system (common in many 3D formats)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    
    if textured and texture is not None and uvs is not None:
        # Create PBR material for the mesh
        material = trimesh.visual.material.PBRMaterial(
            roughnessFactor=1.0,
            baseColorTexture=texture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
        )
        
        # Create the final trimesh object with texture
        mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    else:
        # Create untextured mesh
        mesh = trimesh.Trimesh(vertices, faces)
    
    return mesh


def simplify_gs(
    gs: Gaussian,
    simplify: float = 0.95,
    verbose: bool = True,
):
    """
    Simplify 3D Gaussians using an optimization-based approach
    NOTE: this function is not used in the current implementation for the unsatisfactory performance.
    
    Args:
        gs (Gaussian): 3D Gaussian representation to simplify.
        simplify (float): Ratio of Gaussians to remove in simplification.
        verbose (bool): Whether to print progress.
        
    Returns:
        Gaussian: The simplified Gaussian representation
    """
    if simplify <= 0:
        return gs
    
    # Render single view from the original Gaussian representation
    from .render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, render_frames
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [0], [2], [40])
    res = render_frames(gs, extrinsics, intrinsics, {'resolution': 1024, 'bg_color': (0, 0, 0)}, verbose=False)
    observations = [torch.tensor(res['color'][0] / 255.0).float().cuda().permute(2, 0, 1)]
    
    # Following https://arxiv.org/pdf/2411.06019
    # Initialize renderer
    renderer = GaussianRenderer({
            "resolution": 1024,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0,0,0),
        })
        
    # Clone the Gaussian representation
    new_gs = Gaussian(**gs.init_params)
    new_gs._features_dc = gs._features_dc.clone()
    new_gs._features_rest = gs._features_rest.clone() if gs._features_rest is not None else None
    new_gs._opacity = torch.nn.Parameter(gs._opacity.clone())
    new_gs._rotation = torch.nn.Parameter(gs._rotation.clone())
    new_gs._scaling = torch.nn.Parameter(gs._scaling.clone())
    new_gs._xyz = torch.nn.Parameter(gs._xyz.clone())
    
    # Set up optimizer with different learning rates for different parameters
    start_lr = [1e-4, 1e-3, 5e-3, 0.025]  # Position, rotation, scaling, opacity
    end_lr = [1e-6, 1e-5, 5e-5, 0.00025]
    optimizer = torch.optim.Adam([
        {"params": new_gs._xyz, "lr": start_lr[0]},
        {"params": new_gs._rotation, "lr": start_lr[1]},
        {"params": new_gs._scaling, "lr": start_lr[2]},
        {"params": new_gs._opacity, "lr": start_lr[3]},
    ], lr=start_lr[0])
    
    # Learning rate scheduling functions
    def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
        """Exponential learning rate annealing"""
        return start_lr * (end_lr / start_lr) ** (step / total_steps)

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        """Cosine learning rate annealing"""
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
    
    # Auxiliary variables for proximal optimization algorithm
    _zeta = new_gs.get_opacity.clone().detach().squeeze()
    _lambda = torch.zeros_like(_zeta)
    _delta = 1e-7  # Regularization parameter
    _interval = 10  # Interval for updates
    num_target = int((1 - simplify) * _zeta.shape[0])  # Target number of Gaussians after simplification
    
    # Optimization loop
    with tqdm(total=2500, disable=not verbose, desc='Simplifying Gaussian') as pbar:
        for i in range(2500):
            # Prune low-opacity Gaussians periodically
            if i % 100 == 0:
                mask = new_gs.get_opacity.squeeze() > 0.05
                mask = torch.nonzero(mask).squeeze()
                # Update all relevant parameters
                new_gs._xyz = torch.nn.Parameter(new_gs._xyz[mask])
                new_gs._rotation = torch.nn.Parameter(new_gs._rotation[mask])
                new_gs._scaling = torch.nn.Parameter(new_gs._scaling[mask])
                new_gs._opacity = torch.nn.Parameter(new_gs._opacity[mask])
                new_gs._features_dc = new_gs._features_dc[mask]
                new_gs._features_rest = new_gs._features_rest[mask] if new_gs._features_rest is not None else None
                _zeta = _zeta[mask]
                _lambda = _lambda[mask]
                # Update optimizer state
                for param_group, new_param in zip(optimizer.param_groups, [new_gs._xyz, new_gs._rotation, new_gs._scaling, new_gs._opacity]):
                    stored_state = optimizer.state[param_group['params'][0]]
                    if 'exp_avg' in stored_state:
                        stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                        stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                    del optimizer.state[param_group['params'][0]]
                    param_group['params'][0] = new_param
                    optimizer.state[param_group['params'][0]] = stored_state

            opacity = new_gs.get_opacity.squeeze()
            
            # Sparsify using proximal gradient method
            if i % _interval == 0:
                _zeta = _lambda + opacity.detach()
                if opacity.shape[0] > num_target:
                    # Keep only the top K Gaussians by importance
                    index = _zeta.topk(num_target)[1]
                    _m = torch.ones_like(_zeta, dtype=torch.bool)
                    _m[index] = 0
                    _zeta[_m] = 0
                _lambda = _lambda + opacity.detach() - _zeta
            
            # Sample a random view for this iteration
            view_idx = np.random.randint(len(observations))
            observation = observations[view_idx]
            extrinsic = extrinsics[view_idx]
            intrinsic = intrinsics[view_idx]
            
            # Render and compute loss
            color = renderer.render(new_gs, extrinsic, intrinsic)['color']
            rgb_loss = torch.nn.functional.l1_loss(color, observation)
            # Loss includes reconstruction and sparsity term
            loss = rgb_loss + \
                   _delta * torch.sum(torch.pow(_lambda + opacity - _zeta, 2))
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            for j in range(len(optimizer.param_groups)):
                optimizer.param_groups[j]['lr'] = cosine_anealing(optimizer, i, 2500, start_lr[j], end_lr[j])
            
            pbar.set_postfix({'loss': rgb_loss.item(), 'num': opacity.shape[0], 'lambda': _lambda.mean().item()})
            pbar.update()
    
    # Convert parameters back to data
    new_gs._xyz = new_gs._xyz.data
    new_gs._rotation = new_gs._rotation.data
    new_gs._scaling = new_gs._scaling.data
    new_gs._opacity = new_gs._opacity.data
    
    return new_gs
