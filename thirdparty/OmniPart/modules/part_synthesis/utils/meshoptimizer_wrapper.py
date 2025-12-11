"""
Wrapper module that imports from the meshoptimizer package.
This provides backward compatibility while using the installed package.
"""
try:
    # Try to import from installed meshoptimizer package
    from meshoptimizer import simplify_with_screen_error
except ImportError:
    # Fallback: try to import from local path
    import sys
    from pathlib import Path
    
    meshopt_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "meshoptimizer"
    if meshopt_dir.exists():
        sys.path.insert(0, str(meshopt_dir))
        try:
            from meshoptimizer import simplify_with_screen_error
        except ImportError:
            # Final fallback: use trimesh
            def simplify_with_screen_error(vertices, indices, target_index_count, **kwargs):
                """Fallback to trimesh if meshoptimizer not available"""
                import trimesh
                mesh = trimesh.Trimesh(vertices=vertices, faces=indices.reshape(-1, 3))
                target_faces = target_index_count // 3
                if target_faces < 1:
                    target_faces = 1
                mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
                return mesh.faces.flatten().astype(indices.dtype), mesh.vertices, 0.0
    else:
        # Final fallback: use trimesh
        def simplify_with_screen_error(vertices, indices, target_index_count, **kwargs):
            """Fallback to trimesh if meshoptimizer not available"""
            import trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=indices.reshape(-1, 3))
            target_faces = target_index_count // 3
            if target_faces < 1:
                target_faces = 1
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            return mesh.faces.flatten().astype(indices.dtype), mesh.vertices, 0.0

