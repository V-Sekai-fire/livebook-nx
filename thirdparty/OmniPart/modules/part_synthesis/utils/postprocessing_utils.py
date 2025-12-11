from typing import *
import numpy as np
import torch
import utils3d
# nvdiffrast is optional - PyTorch3D can be used as alternative for texture baking
try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False
    dr = None

# Try PyTorch3D as alternative to nvdiffrast
PYTORCH3D_AVAILABLE = False
PYTORCH3D_IMPORT_ERROR = None
try:
    from pytorch3d.renderer import (
        MeshRasterizer, RasterizationSettings, PerspectiveCameras,
        TexturesUV, TexturesVertex
    )
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.rasterizer import Fragments
    PYTORCH3D_AVAILABLE = True
except ImportError as e:
    PYTORCH3D_AVAILABLE = False
    # Store error for debugging
    PYTORCH3D_IMPORT_ERROR = str(e)
except Exception as e:
    # Catch other errors (e.g., CUDA initialization issues)
    PYTORCH3D_AVAILABLE = False
    PYTORCH3D_IMPORT_ERROR = str(e)

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
from .render_utils import render_multiview
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
    # Construct cameras at uniformly distributed positions on a sphere
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)  # Generate uniformly distributed points on sphere
        yaws.append(y)
        pitchs.append(p)
    yaws = torch.tensor(yaws).cuda()
    pitchs = torch.tensor(pitchs).cuda()
    radius = 2.0  # Camera distance from origin
    fov = torch.deg2rad(torch.tensor(40)).cuda()  # Camera field of view
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)  # Create projection matrix
    views = []
    for (yaw, pitch) in zip(yaws, pitchs):
        # Calculate camera position from spherical coordinates
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda().float() * radius
        # Create view matrix looking at origin
        view = utils3d.torch.view_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        views.append(view)
    views = torch.stack(views, dim=0)

    # Rasterize mesh from multiple viewpoints to determine visible faces
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    
    # Re-check PyTorch3D availability at runtime (in case import failed at module load but works now)
    pytorch3d_available_runtime = PYTORCH3D_AVAILABLE
    if not pytorch3d_available_runtime and not NVDIFFRAST_AVAILABLE:
        # Try importing again at runtime
        try:
            from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PerspectiveCameras
            from pytorch3d.structures import Meshes
            pytorch3d_available_runtime = True
        except Exception:
            pytorch3d_available_runtime = False
    
    # Use PyTorch3D if nvdiffrast is not available
    use_pytorch3d = not NVDIFFRAST_AVAILABLE and pytorch3d_available_runtime
    
    if not NVDIFFRAST_AVAILABLE and not pytorch3d_available_runtime:
        error_msg = "Either nvdiffrast or PyTorch3D is required for hole filling. Please install one of them."
        if PYTORCH3D_IMPORT_ERROR:
            error_msg += f" PyTorch3D import error: {PYTORCH3D_IMPORT_ERROR}"
        raise ImportError(error_msg)
    
    if use_pytorch3d:
        # Import PyTorch3D components at runtime (in case module-level import failed)
        try:
            from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PerspectiveCameras
            from pytorch3d.structures import Meshes
        except ImportError as e:
            raise ImportError(f"PyTorch3D is required for hole filling but import failed: {e}")
        
        # Use PyTorch3D for rasterization
        for i in tqdm(range(views.shape[0]), total=views.shape[0], disable=not verbose, desc='Rasterizing'):
            view = views[i]
            # Create mesh without textures for visibility detection
            meshes = Meshes(verts=verts[None], faces=faces[None])
            
            # Convert view and projection to PyTorch3D format
            R_w2c = view[:3, :3]
            T_w2c = view[:3, 3].unsqueeze(0)
            fx_pixels = projection[0, 0] / 2.0
            fy_pixels = projection[1, 1] / 2.0
            cx_pixels = (projection[0, 2] + 1.0) / 2.0
            cy_pixels = (1.0 - projection[1, 2]) / 2.0
            focal_length = torch.tensor([[fx_pixels, fy_pixels]], device=verts.device, dtype=torch.float32)
            principal_point = torch.tensor([[(cx_pixels - resolution / 2.0) / (resolution / 2.0), 
                                             (cy_pixels - resolution / 2.0) / (resolution / 2.0)]], 
                                           device=verts.device, dtype=torch.float32)
            
            cameras = PerspectiveCameras(
                R=R_w2c[None].to(verts.device),
                T=T_w2c.to(verts.device),
                focal_length=focal_length,
                principal_point=principal_point,
                image_size=torch.tensor([[resolution, resolution]], device=verts.device, dtype=torch.float32),
                device=verts.device,
            )
            
            raster_settings = RasterizationSettings(
                image_size=(resolution, resolution),
                blur_radius=0.0,
                faces_per_pixel=1,
                perspective_correct=True,
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(meshes)
            
            # Get face IDs from fragments
            # PyTorch3D uses -1 for background, 0-indexed face IDs for valid faces
            pix_to_face = fragments.pix_to_face[0]  # (H, W, 1)
            mask = (pix_to_face >= 0).squeeze(-1)  # (H, W)
            if mask.any():
                face_id = pix_to_face[mask].squeeze(-1)  # Face IDs are already 0-indexed in PyTorch3D
                face_id = torch.unique(face_id).long()
                # Clamp to valid face range
                face_id = face_id[(face_id >= 0) & (face_id < faces.shape[0])]
                if face_id.numel() > 0:
                    visblity[face_id] += 1
    else:
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
    # Normalize visibility to [0,1]
    visblity = visblity.float() / num_views
    
    # Prepare for mincut-based mesh cleaning
    ## Construct edge data structures
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
    dual_edges_weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)
    if verbose:
        tqdm.write(f'Dual graph: {dual_edges.shape[0]} edges')

    ## Solve mincut problem using igraph
    ### Construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    g.add_edges(dual_edges.cpu().numpy())
    g.es['weight'] = dual_edges_weights.cpu().numpy()
    
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
    cut = g.mincut('s', 't', (np.array(g.es['weight']) * 1000).tolist())
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
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(edges[cc_new_boundary_edge_indices])
            # Calculate the center of each boundary loop
            cc_new_boundary_edges_cc_center = [verts[edges[cc_new_boundary_edge_indices[edge_cc]]].mean(dim=1).mean(dim=0) for edge_cc in cc_new_boundary_edge_cc]
            cc_new_boundary_edges_cc_area = []
            # Calculate the area of each boundary loop
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]] - cc_new_boundary_edges_cc_center[i]
                _e2 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]] - cc_new_boundary_edges_cc_center[i]
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
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]  # Blue for inner
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]  # Green for outer
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]  # Magenta for removed by mincut
        if len(valid_remove_cc) > 0:
            vis_colors[remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()] = [255, 0, 0]  # Red for valid removal
        utils3d.io.write_ply('dbg_dual.ply', face_v, edges=vis_dual_edges, vertex_colors=vis_colors)
        
        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
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
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
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
    
    # Simplify mesh using quadric edge collapse decimation
    # Use trimesh instead of PyVista to avoid signal handler conflicts in Pythonx
    if simplify and simplify_ratio > 0:
        try:
            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            # Calculate target number of faces
            target_faces = int(faces.shape[0] * simplify_ratio)
            if target_faces < 4:  # Minimum faces for a valid mesh
                target_faces = 4
            if target_faces >= faces.shape[0]:
                # No decimation needed
                if verbose:
                    tqdm.write(f'Decimation skipped: target faces ({target_faces}) >= current faces ({faces.shape[0]})')
            else:
                # Perform quadric decimation
                mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
                vertices = mesh.vertices
                faces = mesh.faces
                if verbose:
                    tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')
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
            vertices_t = torch.tensor(vertices).cuda()
            faces_t = torch.tensor(faces.astype(np.int32)).cuda()
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


def _rasterize_with_pytorch3d(vertices, faces, uvs, height, width, view, projection):
    """
    Rasterize mesh using PyTorch3D as alternative to nvdiffrast.
    Returns same format as utils3d.torch.rasterize_triangle_faces for compatibility.
    
    Properly converts OpenGL-style view and projection matrices to PyTorch3D camera format.
    """
    # Try importing PyTorch3D at runtime if not available at module load
    try:
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PerspectiveCameras, TexturesUV
        from pytorch3d.structures import Meshes
    except ImportError as e:
        raise ImportError(f"PyTorch3D is required for texture baking without nvdiffrast. Import error: {e}")
    
    # Convert to torch tensors if needed and get device
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, dtype=torch.int64)
    if not isinstance(uvs, torch.Tensor):
        uvs = torch.tensor(uvs, dtype=torch.float32)
    
    device = vertices.device
    vertices = vertices.to(device)
    faces = faces.to(device)
    uvs = uvs.to(device)
    
    # Ensure vertices and faces have correct shapes: (batch_size, num_verts, 3) and (batch_size, num_faces, 3)
    # PyTorch3D Meshes requires batch dimension
    if len(vertices.shape) == 2:
        # Add batch dimension: (num_verts, 3) -> (1, num_verts, 3)
        vertices = vertices.unsqueeze(0)
    elif len(vertices.shape) != 3:
        raise ValueError(f"vertices must have shape (num_verts, 3) or (batch_size, num_verts, 3), got {vertices.shape}")
    
    if len(faces.shape) == 2:
        # Add batch dimension: (num_faces, 3) -> (1, num_faces, 3)
        faces = faces.unsqueeze(0)
    elif len(faces.shape) != 3:
        raise ValueError(f"faces must have shape (num_faces, 3) or (batch_size, num_faces, 3), got {faces.shape}")
    
    # Ensure faces are int64 (PyTorch3D requirement)
    faces = faces.long()
    
    batch_size = vertices.shape[0]
    
    # Verify shapes
    if vertices.shape[2] != 3:
        raise ValueError(f"vertices must have 3 coordinates per vertex, got {vertices.shape[2]}")
    if faces.shape[2] != 3:
        raise ValueError(f"faces must have 3 indices per face, got {faces.shape[2]}")
    
    # Convert view and projection to PyTorch3D camera format
    # view: 4x4 world-to-camera matrix (OpenGL convention)
    # projection: 4x4 OpenGL perspective projection matrix
    
    # Ensure view and projection are tensors
    if not isinstance(view, torch.Tensor):
        view = torch.tensor(view, device=device, dtype=torch.float32)
    else:
        view = view.to(device)
    if not isinstance(projection, torch.Tensor):
        projection = torch.tensor(projection, device=device, dtype=torch.float32)
    else:
        projection = projection.to(device)
    
    # Extract rotation and translation from view matrix
    # The view matrix is world-to-camera: [R | T; 0 0 0 | 1]
    # where R is rotation and T is translation in camera coordinates
    # For a point P_world in world coordinates:
    #   P_camera = R @ P_world + T
    R_w2c = view[:3, :3]  # Rotation: world-to-camera (3x3)
    T_w2c = view[:3, 3]    # Translation: in camera coordinates (3,)
    
    # PyTorch3D PerspectiveCameras expects:
    # - R: world-to-camera rotation matrix (3x3) ✓
    # - T: translation in camera coordinates (same as T_w2c) ✓
    #   The transform is: P_camera = R @ P_world + T
    T_pytorch3d = T_w2c.unsqueeze(0)  # Shape: (1, 3)
    
    # Extract intrinsics from projection matrix
    # The projection matrix from intrinsics_to_projection (OpenCV to OpenGL) stores:
    # - projection[0,0] = 2 * fx (fx from intrinsics[0,0] in pixels)
    # - projection[1,1] = 2 * fy (fy from intrinsics[1,1] in pixels)
    # - projection[0,2] = 2 * cx - 1 (cx from intrinsics[0,2] in pixels)
    # - projection[1,2] = -2 * cy + 1 (cy from intrinsics[1,2] in pixels)
    # Note: The intrinsics are in pixels (OpenCV format)
    fx_pixels = projection[0, 0] / 2.0
    fy_pixels = projection[1, 1] / 2.0
    cx_pixels = (projection[0, 2] + 1.0) / 2.0
    cy_pixels = (1.0 - projection[1, 2]) / 2.0
    
    # PyTorch3D PerspectiveCameras expects:
    # - focal_length: (fx, fy) in pixels - ✓ correct
    # - principal_point: offset from image center in NDC coordinates [-1, 1]
    #   Formula: offset = (pixel_coord - center) / (size/2)
    #   This normalizes the offset to [-1, 1] range
    focal_length = torch.tensor([[fx_pixels, fy_pixels]], device=device, dtype=torch.float32)
    principal_point = torch.tensor([[(cx_pixels - width / 2.0) / (width / 2.0), 
                                     (cy_pixels - height / 2.0) / (height / 2.0)]], 
                                   device=device, dtype=torch.float32)
    
    # Create camera with proper intrinsics
    cameras = PerspectiveCameras(
        R=R_w2c[None].to(device),  # (1, 3, 3) - world-to-camera rotation
        T=T_pytorch3d.to(device),  # (1, 3) - translation in camera coordinates
        focal_length=focal_length,  # (1, 2) - focal length in pixels
        principal_point=principal_point,  # (1, 2) - principal point offset from center (normalized)
        image_size=torch.tensor([[height, width]], device=device, dtype=torch.float32),  # (1, 2) - (H, W)
        device=device,
    )
    
    # Create mesh with UV coordinates
    # PyTorch3D needs vertices and faces, and can handle UVs via TexturesUV
    # TexturesUV expects:
    # - maps: (N, H, W, C) - texture maps, not (N, H, W, D, C)
    # - faces_uvs: (N, F, 3) - batch dimension required
    # - verts_uvs: (N, V, 2) - batch dimension required
    # For rasterization, we need a dummy texture map of proper shape
    # Use a minimal 1x1 texture map: (N, 1, 1, 3)
    faces_uvs_batch = faces.unsqueeze(0) if len(faces.shape) == 2 else faces  # (F, 3) -> (1, F, 3)
    verts_uvs_batch = uvs.unsqueeze(0) if len(uvs.shape) == 2 else uvs  # (V, 2) -> (1, V, 2)
    
    # Store references to UV data for later use (PyTorch3D TexturesUV doesn't expose these as public attributes)
    faces_uvs_indices = faces_uvs_batch  # (B, F, 3) - indices into verts_uvs
    verts_uvs = verts_uvs_batch  # (B, V, 2) - actual UV coordinates
    
    meshes = Meshes(
        verts=vertices,
        faces=faces,
        textures=TexturesUV(
            maps=torch.zeros((batch_size, 1, 1, 3), device=device),  # (N, H, W, C) - proper shape
            faces_uvs=faces_uvs_batch,
            verts_uvs=verts_uvs_batch
        )
    )
    
    # Create rasterizer
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    
    # Rasterize
    fragments = rasterizer(meshes)
    
    # Get barycentric coordinates and face indices
    bary_coords = fragments.bary_coords  # (B, H, W, 1, 3)
    pix_to_face = fragments.pix_to_face  # (B, H, W, 1)
    
    # Debug: Check if any faces were rasterized
    if pix_to_face.shape[0] > 0:
        valid_pixels = (pix_to_face >= 0).sum().item()
        total_pixels = pix_to_face.numel()
        print(f"[DEBUG] Rasterization result: {valid_pixels}/{total_pixels} pixels have valid faces (pix_to_face range: [{pix_to_face.min().item()}, {pix_to_face.max().item()}])")
    
    # Interpolate UV coordinates using barycentric coordinates
    # In PyTorch3D, faces_uvs contains indices into verts_uvs
    # faces_uvs: (B, F, 3) - indices into verts_uvs
    # verts_uvs: (B, V, 2) - actual UV coordinates
    # Note: We use the stored references since TexturesUV doesn't expose these as public attributes
    
    # Get face index for each pixel
    face_idx = pix_to_face.squeeze(-1)  # (B, H, W) - face indices, -1 for background
    
    # For each batch, gather face UV indices
    # We'll process each batch separately to avoid complex indexing
    uv_maps = []
    for b in range(batch_size):
        # Get face indices for this batch: (H, W)
        batch_face_idx = face_idx[b]  # (H, W)
        
        # Get valid pixels (face_idx >= 0)
        valid_mask = batch_face_idx >= 0  # (H, W)
        
        if valid_mask.sum() == 0:
            # No valid pixels, create zero UV map
            if b == 0:
                print(f"[DEBUG] Rasterization: NO VALID PIXELS in batch {b}! (valid_mask.sum() == 0)")
                print(f"[DEBUG] pix_to_face range: [{pix_to_face.min().item()}, {pix_to_face.max().item()}]")
            batch_uv_map = torch.zeros((height, width, 2), device=device, dtype=torch.float32)
            uv_maps.append(batch_uv_map)
            continue
        
        # Get UV vertex indices for valid faces: (num_valid, 3)
        valid_face_idx = batch_face_idx[valid_mask]  # (num_valid,)
        
        # Debug: Check face index range (only on first batch to avoid spam)
        debug_this_batch = (b == 0 and valid_face_idx.numel() > 0)
        if b == 0:
            # Always print debug info for first batch, even if no valid pixels
            if valid_face_idx.numel() > 0:
                max_face_idx = valid_face_idx.max().item()
                num_faces = faces_uvs_indices.shape[1]
                if max_face_idx >= num_faces:
                    raise RuntimeError(f"Face index {max_face_idx} out of range [0, {num_faces-1}]. This indicates a mismatch between rasterized faces and UV face indices.")
                print(f"[DEBUG] Rasterization: {valid_face_idx.numel()} valid pixels, max face idx: {max_face_idx}, num faces: {num_faces}")
            else:
                print(f"[DEBUG] Rasterization: NO VALID PIXELS in batch {b}!")
        
        face_uv_indices = faces_uvs_indices[b, valid_face_idx]  # (num_valid, 3)
        
        # Get UV coordinates: (num_valid, 3, 2)
        # face_uv_indices contains vertex indices into verts_uvs
        if debug_this_batch:
            max_uv_idx = face_uv_indices.max().item()
            min_uv_idx = face_uv_indices.min().item()
            num_uv_verts = verts_uvs.shape[1]
            print(f"[DEBUG] UV indices: min={min_uv_idx}, max={max_uv_idx}, num_uv_verts={num_uv_verts}")
            if max_uv_idx >= num_uv_verts:
                raise RuntimeError(f"UV vertex index {max_uv_idx} out of range [0, {num_uv_verts-1}]. This indicates a mismatch between face UV indices and UV vertices.")
        
        face_uvs = verts_uvs[b, face_uv_indices]  # (num_valid, 3, 2)
        
        if debug_this_batch:
            print(f"[DEBUG] Face UVs range: [{face_uvs.min():.3f}, {face_uvs.max():.3f}], mean: {face_uvs.mean():.3f}")
        
        # Get barycentric coordinates for valid pixels: (num_valid, 3)
        batch_bary = bary_coords[b, valid_mask].squeeze(1)  # (num_valid, 3)
        
        if debug_this_batch:
            print(f"[DEBUG] Bary coords range: [{batch_bary.min():.3f}, {batch_bary.max():.3f}], sum per pixel: [{batch_bary.sum(dim=1).min():.3f}, {batch_bary.sum(dim=1).max():.3f}]")
        
        # Interpolate: (num_valid, 3) @ (num_valid, 3, 2) -> (num_valid, 2)
        # We need to do element-wise multiplication and sum
        uv_valid = (batch_bary.unsqueeze(-1) * face_uvs).sum(dim=1)  # (num_valid, 2)
        
        if debug_this_batch:
            print(f"[DEBUG] Interpolated UV range: [{uv_valid.min():.3f}, {uv_valid.max():.3f}], mean: {uv_valid.mean():.3f}")
        
        # Create full UV map for this batch
        batch_uv_map = torch.zeros((height, width, 2), device=device, dtype=torch.float32)
        batch_uv_map[valid_mask] = uv_valid
        uv_maps.append(batch_uv_map)
    
    # Stack all batches: (B, H, W, 2)
    uv_map = torch.stack(uv_maps, dim=0)  # (B, H, W, 2)
    
    # Debug: Check UV map before flipping
    if uv_map.shape[0] > 0:
        valid_uv_mask = uv_map[0].abs().sum(dim=-1) > 0
        if valid_uv_mask.sum() > 0:
            print(f"[DEBUG] UV map before flip: range [{uv_map[0, valid_uv_mask].min():.3f}, {uv_map[0, valid_uv_mask].max():.3f}], valid pixels: {valid_uv_mask.sum()}/{valid_uv_mask.numel()}")
        else:
            print(f"[DEBUG] UV map before flip: ALL ZEROS!")
    
    # Create mask (pixels with valid faces)
    mask = (pix_to_face >= 0).float()  # (B, H, W, 1)
    
    # Flip Y coordinate to match nvdiffrast convention
    uv_map = uv_map.flip(1)  # Flip height dimension
    
    # Debug: Check UV map after flipping
    if uv_map.shape[0] > 0:
        valid_uv_mask = uv_map[0].abs().sum(dim=-1) > 0
        if valid_uv_mask.sum() > 0:
            print(f"[DEBUG] UV map after flip: range [{uv_map[0, valid_uv_mask].min():.3f}, {uv_map[0, valid_uv_mask].max():.3f}], valid pixels: {valid_uv_mask.sum()}/{valid_uv_mask.numel()}")
        else:
            print(f"[DEBUG] UV map after flip: ALL ZEROS!")
    
    return {
        'uv': uv_map,  # (B, H, W, 2)
        'mask': mask.squeeze(-1),  # (B, H, W)
        'face_id': pix_to_face.squeeze(-1),  # (B, H, W)
    }


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
    force_pytorch3d: bool = False,
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
    # Move data to GPU
    vertices = torch.tensor(vertices).cuda()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    uvs = torch.tensor(uvs).cuda()
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    masks = [torch.tensor(m>0).bool().cuda() for m in masks]
    views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
    projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]

    if mode == 'fast':
        # Fast texture baking - weighted average of observed colors
        texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).cuda()
        texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).cuda()
        
        # Determine which backend to use
        if force_pytorch3d:
            # Force PyTorch3D usage
            if not PYTORCH3D_AVAILABLE:
                raise ImportError("PyTorch3D is required when force_pytorch3d=True but PyTorch3D is not available.")
            use_pytorch3d = True
            use_utils3d = False
        else:
            # Auto-detect: Use PyTorch3D if nvdiffrast is not available and PyTorch3D works, otherwise try utils3d
            use_pytorch3d = not NVDIFFRAST_AVAILABLE and PYTORCH3D_AVAILABLE
            use_utils3d = not use_pytorch3d  # Use utils3d if not using PyTorch3D
            
            if use_utils3d:
                try:
                    rastctx = utils3d.torch.RastContext(backend='cuda')
                except Exception as e:
                    # If utils3d also fails, try PyTorch3D as last resort
                    if PYTORCH3D_AVAILABLE:
                        use_pytorch3d = True
                        use_utils3d = False
                    else:
                        raise ImportError(f"Rasterization backend unavailable. utils3d error: {e}. PyTorch3D also unavailable.")
        
        # Iterate through each observation and accumulate colors
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(observations), disable=not verbose, desc='Texture baking (fast)'):
            with torch.no_grad():
                # Rasterize the mesh from this viewpoint
                if use_pytorch3d:
                    rast = _rasterize_with_pytorch3d(
                        vertices[None], faces, uvs[None], 
                        observation.shape[0], observation.shape[1], 
                        view, projection
                    )
                    uv_map = rast['uv'][0].detach()
                    mask = rast['mask'][0].detach().bool() & masks[0]
                else:
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
        # Determine which backend to use
        if force_pytorch3d:
            # Force PyTorch3D usage
            if not PYTORCH3D_AVAILABLE:
                raise ImportError("PyTorch3D is required when force_pytorch3d=True but PyTorch3D is not available.")
            use_pytorch3d = True
        else:
            # Auto-detect: Check if we have a rasterization backend available
            if not NVDIFFRAST_AVAILABLE and not PYTORCH3D_AVAILABLE:
                raise ImportError("Either nvdiffrast or PyTorch3D is required for optimization-based texture baking. Please install one of them.")
            
            # Use PyTorch3D if nvdiffrast is not available, otherwise use nvdiffrast
            use_pytorch3d = not NVDIFFRAST_AVAILABLE and PYTORCH3D_AVAILABLE
            if not use_pytorch3d:
                rastctx = utils3d.torch.RastContext(backend='cuda')
        
        observations = [obs.flip(0) for obs in observations]  # Flip Y for rendering
        masks = [m.flip(0) for m in masks]
        
        # Precompute UV maps for efficiency
        _uv = []
        _uv_dr = []
        for view_idx, (observation, view, projection) in enumerate(tqdm(zip(observations, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV')):
            with torch.no_grad():
                if use_pytorch3d:
                    rast = _rasterize_with_pytorch3d(
                        vertices[None], faces, uvs[None],
                        observation.shape[0], observation.shape[1],
                        view, projection
                    )
                    uv_result = rast['uv'].detach()
                    _uv.append(uv_result)
                    # For PyTorch3D, we don't have uv_dr, so create a dummy gradient tensor
                    _uv_dr.append(torch.ones_like(uv_result))
                    # Debug: Check UV map for first view
                    if view_idx == 0 and verbose:
                        valid_uv_mask = uv_result[0].abs().sum(dim=-1) > 0
                        if valid_uv_mask.sum() > 0:
                            print(f"[DEBUG] First view UV map: range [{uv_result[0, valid_uv_mask].min():.3f}, {uv_result[0, valid_uv_mask].max():.3f}], valid pixels: {valid_uv_mask.sum()}/{valid_uv_mask.numel()}")
                        else:
                            print(f"[DEBUG] First view UV map: ALL ZEROS!")
                else:
                    rast = utils3d.torch.rasterize_triangle_faces(
                        rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                    )
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
        total_steps = 2500
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
                if use_pytorch3d:
                    # Use PyTorch3D texture sampling (simpler bilinear sampling)
                    # Sample texture using UV coordinates
                    # uv from _rasterize_with_pytorch3d has shape (B, H, W, 2) where B=1
                    # We need to ensure it's (H, W, 2) before normalizing, then (1, H, W, 2) for grid_sample
                    if len(uv.shape) == 4 and uv.shape[0] == 1:
                        # Remove batch dimension: (1, H, W, 2) -> (H, W, 2)
                        uv = uv.squeeze(0)
                    # Clamp UV coordinates to [0, 1] range to avoid out-of-bounds sampling
                    uv = torch.clamp(uv, 0.0, 1.0)
                    # Flip Y coordinate: PyTorch3D uses bottom-left origin, but grid_sample uses top-left
                    # Also, texture coordinates typically have Y flipped
                    uv_flipped = torch.stack([uv[..., 0], 1.0 - uv[..., 1]], dim=-1)
                    uv_normalized = uv_flipped * 2.0 - 1.0  # Convert [0,1] to [-1,1] for grid_sample
                    uv_grid = uv_normalized.unsqueeze(0)  # (H, W, 2) -> (1, H, W, 2)
                    # grid_sample expects (N, C, H, W) input and (N, H, W, 2) grid
                    texture_for_sampling = texture.permute(0, 3, 1, 2)  # (1, 3, H, W)
                    render = torch.nn.functional.grid_sample(
                        texture_for_sampling, uv_grid, mode='bilinear', 
                        padding_mode='zeros', align_corners=True
                    )
                    render = render.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
                    # Debug on first step
                    if step == 0 and verbose:
                        print(f"[DEBUG] Render range: [{render.min():.3f}, {render.max():.3f}], mean: {render.mean():.3f}")
                        if mask.sum() > 0:
                            print(f"[DEBUG] Render on mask range: [{render[mask].min():.3f}, {render[mask].max():.3f}], mean: {render[mask].mean():.3f}")
                            print(f"[DEBUG] Observation on mask range: [{observation[mask].min():.3f}, {observation[mask].max():.3f}], mean: {observation[mask].mean():.3f}")
                        else:
                            print(f"[DEBUG] Render on mask: no valid pixels")
                            print(f"[DEBUG] Observation on mask: no valid pixels")
                else:
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
                loss.backward()
                optimizer.step()
                # Learning rate annealing
                optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, total_steps, 1e-2, 1e-5)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
        
        if srgb_space:
            # convert the texture from rgb space to srgb 
            texture = rgb_to_srgb_image(texture)
        
        # Convert optimized texture to numpy array
        texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        # Fill any remaining holes in the texture
        if use_pytorch3d:
            # For PyTorch3D, rasterize UV mesh to find holes
            try:
                # Create a simple UV mesh (2D mesh in texture space)
                uv_verts_2d = (uvs * 2 - 1).unsqueeze(0)  # Convert to [-1,1] range
                # Add z=0 for 2D mesh
                uv_verts_3d = torch.cat([uv_verts_2d, torch.zeros((1, uvs.shape[0], 1), device=uvs.device)], dim=-1)
                # Simple orthographic camera for UV space
                from pytorch3d.renderer import OrthographicCameras
                uv_camera = OrthographicCameras(device=uvs.device)
                uv_mesh = Meshes(verts=uv_verts_3d, faces=faces.unsqueeze(0))
                uv_rasterizer = MeshRasterizer(
                    cameras=uv_camera,
                    raster_settings=RasterizationSettings(
                        image_size=(texture_size, texture_size),
                        blur_radius=0.0,
                        faces_per_pixel=1,
                    )
                )
                uv_fragments = uv_rasterizer(uv_mesh)
                mask = 1 - (uv_fragments.pix_to_face[0] >= 0).float().cpu().numpy().astype(np.uint8)
            except:
                # Fallback: use simple threshold on texture coverage
                mask = np.zeros((texture_size, texture_size), dtype=np.uint8)
        else:
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
    texture_size: int = 1024,
    debug: bool = False,
    verbose: bool = True,
    textured: bool = True,
    force_pytorch3d: bool = False,
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
    # Extract mesh data from the result
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    
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
            # Use original mesh data
            vertices = mesh.vertices.detach().cpu().numpy()
            faces = mesh.faces.detach().cpu().numpy()

    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return None

    texture = None
    uvs = None
    if textured:
        # Check if app_rep is available (needed for texture baking)
        if app_rep is None:
            if verbose:
                print("[WARN] app_rep (gaussian/strivec) is None. Cannot bake texture. Skipping texture baking.")
            textured = False
        
    if textured:
        # Check if texture baking is possible
        # Determine backend availability
        utils3d_available = False
        if not force_pytorch3d:
            try:
                # Test if utils3d rasterization works
                test_ctx = utils3d.torch.RastContext(backend='cuda')
                del test_ctx
                utils3d_available = True
                if verbose:
                    print("[INFO] Using utils3d for texture baking")
            except Exception as e:
                utils3d_available = False
                if verbose:
                    print(f"[INFO] utils3d rasterization unavailable: {e}")
        
        # Check PyTorch3D availability if needed
        if force_pytorch3d or not utils3d_available:
            if not PYTORCH3D_AVAILABLE:
                if verbose:
                    print("[WARN] Neither utils3d rasterization nor PyTorch3D is available. Texture baking disabled.")
                    if PYTORCH3D_IMPORT_ERROR:
                        print(f"[INFO] PyTorch3D import error: {PYTORCH3D_IMPORT_ERROR}")
                textured = False
            else:
                if verbose:
                    if force_pytorch3d:
                        print("[INFO] Using PyTorch3D for texture baking (forced)")
                    else:
                        print("[INFO] Using PyTorch3D for texture baking (utils3d unavailable)")
        
        # Proceed with texture baking if either backend is available
        if textured:
            # Create UV mapping for the mesh
            vertices, faces, uvs = parametrize_mesh(vertices, faces)
            if verbose:
                print(f"[DEBUG] After parametrize_mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces, {uvs.shape[0]} UVs")
                print(f"[DEBUG] UV range from xatlas: [{uvs.min():.3f}, {uvs.max():.3f}], mean: {uvs.mean():.3f}")

            # Render multi-view images from the appearance representation for texturing
            observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)
            # Create masks from the rendered images
            masks = [np.any(observation > 0, axis=-1) for observation in observations]
            # Convert camera parameters to numpy
            extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
            intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
            
            # Bake texture from the rendered views onto the mesh
            try:
                texture = bake_texture(
                    vertices, faces, uvs,
                    observations, masks, extrinsics, intrinsics,
                    texture_size=texture_size, mode='opt',  # Use optimization-based texturing
                    lambda_tv=0.01,  # Total variation regularization
                    verbose=verbose,
                    force_pytorch3d=force_pytorch3d
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
    
    # Render multi-view images from the original Gaussian representation
    observations, extrinsics, intrinsics = render_multiview(gs, resolution=1024, nviews=100)
    observations = [torch.tensor(obs / 255.0).float().cuda().permute(2, 0, 1) for obs in observations]
    
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
