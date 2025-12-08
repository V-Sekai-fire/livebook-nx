"""
Robust Skin Weights Transfer via Weight Inpainting
Based on: https://github.com/rin-23/RobustSkinWeightsTransferCode
SIGGRAPH Asia 2023 Technical Communications paper

This module provides robust skin weight transfer between meshes using:
1. Closest point matching with distance and normal thresholds
2. Barycentric interpolation for matched vertices
3. Laplacian-based inpainting for unmatched vertices
4. Optional smoothing for inpainted regions

REQUIRES: libigl-python-bindings
"""

import numpy as np
import math
from typing import Tuple, Optional
from scipy.sparse import csc_matrix, diags

# Require libigl - crash if not available
try:
    import igl
except ImportError as e:
    raise ImportError(
        "libigl-python-bindings is required for robust skin weight transfer. "
        "Please install it with: pip install libigl-python-bindings"
    ) from e


def compute_barycentric_coordinates(P: np.ndarray, V1: np.ndarray, V2: np.ndarray, V3: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates for points P in triangles (V1, V2, V3).
    
    Args:
        P: #P by 3, query points
        V1: #P by 3, first triangle vertices
        V2: #P by 3, second triangle vertices
        V3: #P by 3, third triangle vertices
    Returns:
        B: #P by 3, barycentric coordinates [b1, b2, b3]
    """
    # Compute vectors
    v0 = V1 - V3  # V1 relative to V3
    v1 = V2 - V3  # V2 relative to V3
    v2 = P - V3   # P relative to V3
    
    # Compute dot products
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    
    # Compute denominator
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)  # Avoid division by zero
    
    # Compute barycentric coordinates
    b1 = (d11 * d20 - d01 * d21) / denom
    b2 = (d00 * d21 - d01 * d20) / denom
    b3 = 1.0 - b1 - b2
    
    B = np.column_stack([b1, b2, b3])
    return B


def find_closest_point_on_surface(P: np.ndarray, V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given points, find their closest points on the surface of the V,F mesh.
    
    Args:
        P: #P by 3, where every row is a point coordinate
        V: #V by 3 mesh vertices
        F: #F by 3 mesh triangles indices
    Returns:
        sqrD: #P smallest squared distances
        I: #P primitive indices corresponding to smallest distances
        C: #P by 3 closest points
        B: #P by 3 of the barycentric coordinates of the closest point
    """
    sqrD, I, C = igl.point_mesh_squared_distance(P, V, F)
    F_closest = F[I, :]
    V1 = V[F_closest[:, 0], :]
    V2 = V[F_closest[:, 1], :]
    V3 = V[F_closest[:, 2], :]
    B = compute_barycentric_coordinates(C, V1, V2, V3)
    return sqrD, I, C, B


def interpolate_attribute_from_bary(A: np.ndarray, B: np.ndarray, I: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Interpolate per-vertex attributes A via barycentric coordinates B of the F[I,:] vertices.
    
    Args:
        A: #V by N per-vertex attributes
        B: #B by 3 array of the barycentric coordinates of some points
        I: #B primitive indices containing the closest point
        F: #F by 3 mesh triangle indices
    Returns:
        A_out: #B interpolated attributes
    """
    F_closest = F[I, :]
    a1 = A[F_closest[:, 0], :]
    a2 = A[F_closest[:, 1], :]
    a3 = A[F_closest[:, 2], :]
    
    b1 = B[:, 0].reshape(-1, 1)
    b2 = B[:, 1].reshape(-1, 1)
    b3 = B[:, 2].reshape(-1, 1)
    
    A_out = a1 * b1 + a2 * b2 + a3 * b3
    return A_out


def normalize_vec(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def find_matches_closest_surface(
    V1: np.ndarray, F1: np.ndarray, N1: np.ndarray,
    V2: np.ndarray, F2: np.ndarray, N2: np.ndarray,
    W1: np.ndarray,
    dDISTANCE_THRESHOLD_SQRD: float,
    dANGLE_THRESHOLD_DEGREES: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each vertex on the target mesh find a match on the source mesh.
    
    Args:
        V1: #V1 by 3 source mesh vertices
        F1: #F1 by 3 source mesh triangles indices
        N1: #V1 by 3 source mesh normals
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        N2: #V2 by 3 target mesh normals
        W1: #V1 by num_bones source mesh skin weights
        dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
        dANGLE_THRESHOLD_DEGREES: scalar normal threshold
    
    Returns:
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match
        W2: #V2 by num_bones, skinning weights copied from source using closest point method
    """
    Matched = np.zeros(V2.shape[0], dtype=bool)
    sqrD, I, C, B = find_closest_point_on_surface(V2, V1, F1)
    
    # Interpolate per-vertex attributes (skin weights and normals) using barycentric coordinates
    W2 = interpolate_attribute_from_bary(W1, B, I, F1)
    N1_match_interpolated = interpolate_attribute_from_bary(N1, B, I, F1)
    
    # Check that the closest point passes our distance and normal thresholds
    for RowIdx in range(V2.shape[0]):
        n1 = normalize_vec(N1_match_interpolated[RowIdx, :])
        n2 = normalize_vec(N2[RowIdx, :])
        rad_angle = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
        deg_angle = math.degrees(rad_angle)
        if sqrD[RowIdx] <= dDISTANCE_THRESHOLD_SQRD and deg_angle <= dANGLE_THRESHOLD_DEGREES:
            Matched[RowIdx] = True
    
    return Matched, W2


def is_valid_array(sparse_matrix) -> bool:
    """Check if sparse matrix has valid (non-NaN, non-Inf) values."""
    has_invalid_numbers = np.isnan(sparse_matrix.data).any() or np.isinf(sparse_matrix.data).any()
    return not has_invalid_numbers


def inpaint(V2: np.ndarray, F2: np.ndarray, W2: np.ndarray, Matched: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Inpaint weights for all vertices on the target mesh for which we didn't find a good match.
    
    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones, skinning weights copied from source
        Matched: #V2 array of bools, True if we found a good match
    
    Returns:
        W_inpainted: #V2 by num_bones, final skinning weights with inpainting
        success: True if inpainting succeeded, False otherwise
    """
    # Compute the laplacian
    try:
        L = 2 * igl.cotmatrix(V2, F2)
    except Exception as e:
        # If Laplacian computation fails, return interpolated weights
        return W2, False
    
    # Use MASSMATRIX_TYPE_VORONOI if available, otherwise use numeric constant (1)
    try:
        mass_type = igl.MASSMATRIX_TYPE_VORONOI
    except AttributeError:
        mass_type = 1  # MASSMATRIX_TYPE_VORONOI = 1
    
    try:
        M = igl.massmatrix(V2, F2, mass_type)
        Minv = diags(1 / M.diagonal())
    except Exception as e:
        # If mass matrix computation fails, return interpolated weights
        return W2, False
    
    if not is_valid_array(L):
        # Laplacian is invalid (contains NaN/Inf), return interpolated weights
        return W2, False
    
    if not is_valid_array(Minv):
        # Mass matrix inverse is invalid, return interpolated weights
        return W2, False
    
    try:
        Q = -L + L @ Minv @ L
    except Exception as e:
        # If matrix multiplication fails, return interpolated weights
        return W2, False
    
    if not is_valid_array(Q):
        # System matrix is invalid, return interpolated weights
        return W2, False
    
    # Convert Q to csc_matrix format explicitly
    Q = csc_matrix(Q)
    
    # Set up boundary conditions
    Aeq = csc_matrix((0, 0))
    Beq = np.empty((0, 0), dtype=np.float64)  # Must be 2D array with shape (0, 0)
    B = np.zeros((L.shape[0], W2.shape[1]), dtype=np.float64)
    
    b = np.array(range(0, int(V2.shape[0])), dtype=np.int64)
    b = b[Matched]
    bc = W2[Matched, :].astype(np.float64)
    
    # Check if we have any matched vertices - if not, can't do inpainting
    if len(b) == 0:
        # No matched vertices, return interpolated weights
        return W2, False
    
    # Use keyword arguments to match the expected function signature
    # The Python bindings expect: A, B, known, Y, Aeq, Beq, pd
    # The function returns a single array, not a tuple
    try:
        W_inpainted = igl.min_quad_with_fixed(A=Q, B=B, known=b, Y=bc, Aeq=Aeq, Beq=Beq, pd=True)
        # Verify the result is valid
        if np.any(np.isnan(W_inpainted)) or np.any(np.isinf(W_inpainted)):
            # Result contains invalid values, use interpolated weights
            return W2, False
        success = True
    except Exception as e:
        # If inpainting fails, return interpolated weights
        W_inpainted = W2
        success = False
    
    return W_inpainted, success


def smooth(
    V2: np.ndarray, F2: np.ndarray, W2: np.ndarray, Matched: np.ndarray,
    dDISTANCE_THRESHOLD: float, num_smooth_iter_steps: int = 10, smooth_alpha: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth weights in areas where weights were inpainted and their close neighbors.
    
    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones skinning weights
        Matched: #V2 array of bools, True if we found a good match
        dDISTANCE_THRESHOLD: scalar distance threshold
        num_smooth_iter_steps: scalar number of smoothing steps
        smooth_alpha: scalar the smoothing strength
    
    Returns:
        W2_smoothed: #V2 by num_bones new smoothed weights
        VIDs_to_smooth: 1D array of vertex IDs for which smoothing was applied
    """
    NotMatched = ~Matched
    VIDs_to_smooth = np.array(NotMatched, copy=True)
    
    # Build adjacency list
    adj_list = igl.adjacency_list(F2)
    
    def get_points_within_distance(V: np.ndarray, VID: int, distance: float = dDISTANCE_THRESHOLD):
        """Get all neighbours of vertex VID within distance threshold."""
        queue = [VID]
        visited = {VID}
        while len(queue) != 0:
            vv = queue.pop(0)
            neigh = adj_list[vv]
            for nn in neigh:
                if not VIDs_to_smooth[nn] and np.linalg.norm(V[VID, :] - V[nn, :]) < distance:
                    VIDs_to_smooth[nn] = True
                    if nn not in visited:
                        queue.append(nn)
                        visited.add(nn)
    
    for i in range(V2.shape[0]):
        if NotMatched[i]:
            get_points_within_distance(V2, i)
    
    W2_smoothed = np.array(W2, copy=True)
    for step_idx in range(num_smooth_iter_steps):
        for i in range(V2.shape[0]):
            if VIDs_to_smooth[i]:
                neigh = adj_list[i]
                num = len(neigh)
                if num > 0:
                    weight = W2_smoothed[i, :]
                    new_weight = (1 - smooth_alpha) * weight
                    for influence_idx in neigh:
                        weight_connected = W2_smoothed[influence_idx, :]
                        new_weight += (weight_connected / num) * smooth_alpha
                    W2_smoothed[i, :] = new_weight
    
    return W2_smoothed, VIDs_to_smooth


def propagate_weights_from_matched(
    V2: np.ndarray, F2: np.ndarray, W2: np.ndarray, Matched: np.ndarray,
    max_hops: int = 5
) -> np.ndarray:
    """
    Propagate weights from matched vertices to unmatched vertices using mesh connectivity.
    Uses edge-based propagation (geodesic distance) which is more natural than Euclidean distance.
    This creates a "rubbery" fallback when inpainting fails.
    
    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones skin weights (matched vertices have valid weights)
        Matched: #V2 array of bools, True if vertex has matched weights
        max_hops: Maximum number of edge hops to search for matched neighbors
    
    Returns:
        W2_propagated: #V2 by num_bones propagated weights
    """
    W2_propagated = np.array(W2, copy=True)
    NotMatched = ~Matched
    
    # Build adjacency list for edge-based propagation
    adj_list = igl.adjacency_list(F2)
    
    # For each unmatched vertex, find matched vertices reachable via mesh edges
    for vid in range(V2.shape[0]):
        if not NotMatched[vid]:
            continue  # Skip matched vertices
        
        # BFS to find matched vertices within max_hops
        queue = [(vid, 0)]  # (vertex_id, hop_count)
        visited = {vid}
        matched_neighbors = []  # (vertex_id, hop_count, edge_distance)
        
        while len(queue) > 0:
            current, hops = queue.pop(0)
            
            # Check if we've exceeded max hops
            if hops >= max_hops:
                continue
            
            # Check neighbors
            for neighbor in adj_list[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                
                # Check if this neighbor is matched
                if Matched[neighbor]:
                    # Compute edge distance (geodesic distance along mesh)
                    edge_dist = np.linalg.norm(V2[current] - V2[neighbor])
                    matched_neighbors.append((neighbor, hops + 1, edge_dist))
                else:
                    # Continue BFS
                    queue.append((neighbor, hops + 1))
        
        # If we found matched neighbors, blend their weights
        if len(matched_neighbors) > 0:
            # Sort by hop count (prefer closer in mesh topology), then by edge distance
            matched_neighbors.sort(key=lambda x: (x[1], x[2]))
            
            # Take closest neighbors (prefer 1-hop, then 2-hop, etc.)
            # Use up to 5 neighbors, prioritizing lower hop counts
            closest = matched_neighbors[:5]
            
            # Compute weights: inverse of (hop_count + edge_distance)
            # This gives more weight to closer neighbors (fewer hops, shorter edges)
            hop_counts = np.array([x[1] for x in closest])
            edge_dists = np.array([x[2] for x in closest])
            
            # Normalize edge distances by average edge length for fair comparison
            avg_edge_length = np.mean([np.linalg.norm(V2[i] - V2[j]) 
                                      for i in range(min(100, len(V2))) 
                                      for j in adj_list[i] if j < len(V2)])
            if avg_edge_length > 1e-6:
                normalized_dists = edge_dists / avg_edge_length
            else:
                normalized_dists = edge_dists
            
            # Weight = 1 / (hop_count + normalized_edge_distance)
            # Lower hop count and shorter edges = higher weight
            weights = 1.0 / (hop_counts + normalized_dists + 1e-6)
            weights = weights / weights.sum()
            
            # Blend weights from matched neighbors
            blended = np.zeros(W2.shape[1])
            for i, (nid, _, _) in enumerate(closest):
                blended += weights[i] * W2[nid]
            
            W2_propagated[vid] = blended
        # If no matched neighbors found via mesh connectivity, 
        # fall back to finding closest matched vertex by Euclidean distance
        # (but this should be rare if mesh is well-connected)
        else:
            matched_indices = np.where(Matched)[0]
            if len(matched_indices) > 0:
                # Find closest matched vertex by Euclidean distance as last resort
                distances = np.linalg.norm(V2[matched_indices] - V2[vid], axis=1)
                closest_idx = matched_indices[np.argmin(distances)]
                W2_propagated[vid] = W2[closest_idx]
    
    # Normalize weights to sum to 1 for each vertex
    row_sums = W2_propagated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-6, 1.0, row_sums)  # Avoid division by zero
    W2_propagated = W2_propagated / row_sums
    
    return W2_propagated


def robust_skin_weights_transfer(
    V1: np.ndarray, F1: np.ndarray, W1: np.ndarray,
    V2: np.ndarray, F2: np.ndarray,
    SearchRadius: Optional[float] = None,
    NormalThreshold: float = 30.0,
    num_smooth_iter_steps: int = 10,
    smooth_alpha: float = 0.2,
    use_smoothing: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Perform robust weight transfer from source mesh to target mesh.
    
    Args:
        V1: #V1 by 3 source mesh vertices
        F1: #F1 by 3 source mesh triangles indices
        W1: #V1 by num_bones source mesh skin weights
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        SearchRadius: Radius for searching closest point (default: 5% of bounding box diagonal)
        NormalThreshold: Maximum angle (degrees) between normals to be considered a match
        num_smooth_iter_steps: Number of smoothing iterations
        smooth_alpha: Smoothing strength
        use_smoothing: Whether to apply smoothing step
    
    Returns:
        W2: #V2 by num_bones target mesh skin weights
        success: True if transfer succeeded
    """
    # Compute normals
    N1 = igl.per_vertex_normals(V1, F1)
    N2 = igl.per_vertex_normals(V2, F2)
    
    # Set default search radius if not provided
    if SearchRadius is None:
        # Compute bounding box diagonal manually if igl.bounding_box_diagonal doesn't exist
        try:
            bbox_diag = igl.bounding_box_diagonal(V2)
        except AttributeError:
            # Manual computation: diagonal of axis-aligned bounding box
            bbox_min = np.min(V2, axis=0)
            bbox_max = np.max(V2, axis=0)
            bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        SearchRadius = 0.05 * bbox_diag
    
    SearchRadiusSqrd = SearchRadius * SearchRadius
    
    # Step 1: Find matches
    Matched, SkinWeights_interpolated = find_matches_closest_surface(
        V1, F1, N1, V2, F2, N2, W1, SearchRadiusSqrd, NormalThreshold
    )
    
    # Step 2: Inpaint unmatched vertices
    InpaintedWeights, success = inpaint(V2, F2, SkinWeights_interpolated, Matched)
    
    if not success:
        # Inpainting failed - use mesh connectivity-based propagation from matched vertices
        # This creates a "rubbery" fallback that propagates weights along mesh edges,
        # which is more natural than using far-away interpolated weights
        PropagatedWeights = propagate_weights_from_matched(
            V2, F2, SkinWeights_interpolated, Matched, max_hops=5
        )
        return PropagatedWeights, False
    
    # Step 3: Optional smoothing
    if use_smoothing:
        SmoothedInpaintedWeights, _ = smooth(
            V2, F2, InpaintedWeights, Matched, SearchRadius,
            num_smooth_iter_steps, smooth_alpha
        )
        return SmoothedInpaintedWeights, True
    else:
        return InpaintedWeights, True
