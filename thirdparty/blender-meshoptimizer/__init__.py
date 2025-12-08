import bmesh
import bpy
import ctypes
import os
import platform
from bpy.types import Operator, AddonPreferences, PropertyGroup
from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty

bl_info = {
    "name": "Mesh Decimator (meshopt)",
    "author": "Meshopt Integration",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "support": "COMMUNITY",
    "category": "Mesh",
    "description": "Mesh decimation using meshoptimizer library.",
    "location": "Editmode > Mesh",
    "warning": "",
    "doc_url": "",
}

# Try to load meshopt library
_meshopt_lib = None

def load_meshopt_library():
    """Try to load the meshopt shared library"""
    global _meshopt_lib
    
    if _meshopt_lib is not None:
        return _meshopt_lib
    
    system = platform.system()
    lib_name = None
    
    if system == "Windows":
        lib_name = "meshoptimizer.dll"
    elif system == "Darwin":  # macOS
        lib_name = "libmeshoptimizer.dylib"
    else:  # Linux
        lib_name = "libmeshoptimizer.so"
    
    # Try common locations
    search_paths = [
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), "lib"),
        os.path.join(os.path.dirname(__file__), "meshoptimizer", "lib"),
    ]
    
    for path in search_paths:
        lib_path = os.path.join(path, lib_name)
        if os.path.exists(lib_path):
            try:
                _meshopt_lib = ctypes.CDLL(lib_path)
                setup_meshopt_functions(_meshopt_lib)
                return _meshopt_lib
            except OSError:
                continue
    
    # Try loading from system path
    try:
        _meshopt_lib = ctypes.CDLL(lib_name)
        setup_meshopt_functions(_meshopt_lib)
        return _meshopt_lib
    except OSError:
        pass
    
    return None

def setup_meshopt_functions(lib):
    """Setup meshopt function signatures"""
    # meshopt_simplify signature
    lib.meshopt_simplify.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # destination
        ctypes.POINTER(ctypes.c_uint32),  # indices
        ctypes.c_size_t,  # index_count
        ctypes.POINTER(ctypes.c_float),  # vertex_positions
        ctypes.c_size_t,  # vertex_count
        ctypes.c_size_t,  # vertex_positions_stride
        ctypes.c_size_t,  # target_index_count
        ctypes.c_float,  # target_error
        ctypes.c_uint32,  # options
        ctypes.POINTER(ctypes.c_float),  # result_error (can be NULL)
    ]
    lib.meshopt_simplify.restype = ctypes.c_size_t
    
    # meshopt_simplifyScale signature
    lib.meshopt_simplifyScale.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # vertex_positions
        ctypes.c_size_t,  # vertex_count
        ctypes.c_size_t,  # vertex_positions_stride
    ]
    lib.meshopt_simplifyScale.restype = ctypes.c_float

class MeshDecimatorPreferences(AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        lib = load_meshopt_library()
        if lib:
            layout.label(text="✓ meshoptimizer library loaded", icon='CHECKMARK')
        else:
            layout.label(text="✗ meshoptimizer library not found", icon='ERROR')
            layout.label(text="Please build meshoptimizer as a shared library")
            layout.label(text="and place it in the add-on directory or system path")

class MESH_OT_decimate_meshopt(Operator):
    """Decimate mesh using meshoptimizer"""
    
    bl_idname = "mesh.decimate_meshopt"
    bl_label = "Decimate Mesh (meshopt)"
    bl_description = "Reduce triangle count using meshoptimizer library (error-based, like Godot Engine)"
    bl_options = {"REGISTER", "UNDO"}
    
    target_error: FloatProperty(
        name="Target Error",
        description="Maximum allowed error relative to mesh bounding box size (scene area)",
        default=0.0001,
        min=0.0,
        max=1.0,
        step=0.0001,
        precision=5,
    )
    
    lock_border: BoolProperty(
        name="Lock Border",
        description="Do not move vertices on mesh borders",
        default=True,
    )
    
    def execute(self, context):
        lib = load_meshopt_library()
        if lib is None:
            self.report({"ERROR"}, "meshoptimizer library not found. Please build and install it.")
            return {"CANCELLED"}
        
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        
        if len(selected_objects) == 0:
            self.report({"WARNING"}, "Select at least one mesh object.")
            return {"CANCELLED"}
        
        # Store the current mode and active object
        original_mode = context.mode
        original_active = context.active_object
        
        # Process each selected object
        processed_count = 0
        for obj in selected_objects:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode="OBJECT")
            
            context.view_layer.objects.active = obj
            
            result = self.decimate_mesh(context, obj, lib)
            if result:
                processed_count += 1
        
        # Restore original mode and active object
        if original_active:
            try:
                if original_active.name in bpy.data.objects:
                    context.view_layer.objects.active = original_active
                    if original_mode == 'EDIT_MESH' and original_active.type == 'MESH':
                        bpy.ops.object.mode_set(mode="EDIT")
                    else:
                        bpy.ops.object.mode_set(mode="OBJECT")
            except (ReferenceError, AttributeError):
                if original_mode == 'EDIT_MESH':
                    bpy.ops.object.mode_set(mode="OBJECT")
        
        if processed_count > 0:
            self.report({"INFO"}, f"Decimated {processed_count} object(s).")
        
        return {"FINISHED"}
    
    def decimate_mesh(self, context, obj, lib):
        """Decimate a single mesh object"""
        # Ensure we're in object mode
        bpy.ops.object.mode_set(mode="OBJECT")
        
        # Get mesh data
        mesh = obj.data
        
        # Triangulate if needed
        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bmesh.update_edit_mesh(mesh)
        bpy.ops.object.mode_set(mode="OBJECT")
        
        # Rebuild bmesh to get triangulated data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        
        # Extract vertex positions
        vertex_count = len(bm.verts)
        vertex_positions = (ctypes.c_float * (vertex_count * 3))()
        
        for i, vert in enumerate(bm.verts):
            co = vert.co
            vertex_positions[i * 3] = co.x
            vertex_positions[i * 3 + 1] = co.y
            vertex_positions[i * 3 + 2] = co.z
        
        # Extract indices
        index_count = len(bm.faces) * 3
        indices = (ctypes.c_uint32 * index_count)()
        
        for i, face in enumerate(bm.faces):
            for j, vert in enumerate(face.verts):
                indices[i * 3 + j] = vert.index
        
        # Calculate error scale based on mesh bounding box (scene area)
        error_scale = lib.meshopt_simplifyScale(
            vertex_positions,
            vertex_count,
            ctypes.sizeof(ctypes.c_float) * 3
        )
        
        # Use error-based simplification (like Godot Engine)
        # target_index_count = 0 means error is the only limiting factor
        target_index_count = 0
        target_error = self.target_error
        
        # Options
        options = 0
        if self.lock_border:
            options |= 1  # meshopt_SimplifyLockBorder
        
        # Allocate result buffer
        result_indices = (ctypes.c_uint32 * index_count)()
        result_error = ctypes.c_float()
        
        # Call meshopt_simplify
        result_count = lib.meshopt_simplify(
            result_indices,
            indices,
            index_count,
            vertex_positions,
            vertex_count,
            ctypes.sizeof(ctypes.c_float) * 3,
            target_index_count,
            target_error,
            options,
            ctypes.byref(result_error)
        )
        
        if result_count == 0:
            self.report({"WARNING"}, f"{obj.name}: Could not simplify mesh")
            bm.free()
            return False
        
        # Clear existing faces
        bm.faces.ensure_lookup_table()
        for face in list(bm.faces):
            bm.faces.remove(face)
        
        # Create faces from simplified indices
        face_count = result_count // 3
        for i in range(face_count):
            idx = i * 3
            v0_idx = result_indices[idx]
            v1_idx = result_indices[idx + 1]
            v2_idx = result_indices[idx + 2]
            
            # Skip degenerate faces (same vertex used multiple times)
            if v0_idx == v1_idx or v1_idx == v2_idx or v0_idx == v2_idx:
                continue
            
            try:
                v0 = bm.verts[v0_idx]
                v1 = bm.verts[v1_idx]
                v2 = bm.verts[v2_idx]
                bm.faces.new((v0, v1, v2))
            except (ValueError, IndexError):
                # Skip invalid faces
                pass
        
        # Remove unused vertices
        bm.verts.ensure_lookup_table()
        for vert in list(bm.verts):
            if not vert.link_faces:
                bm.verts.remove(vert)
        
        # Update mesh
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        
        original_face_count = index_count // 3
        self.report({"INFO"}, f"{obj.name}: Reduced to {face_count} triangles (from {original_face_count})")
        return True
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

ui_classes = (MESH_OT_decimate_meshopt, MeshDecimatorPreferences)

def menu_func_decimate(self, context):
    self.layout.separator()
    self.layout.operator(MESH_OT_decimate_meshopt.bl_idname)

def register():
    for ui_class in ui_classes:
        bpy.utils.register_class(ui_class)
    bpy.types.VIEW3D_MT_edit_mesh.append(menu_func_decimate)

def unregister():
    bpy.types.VIEW3D_MT_edit_mesh.remove(menu_func_decimate)
    for ui_class in ui_classes:
        bpy.utils.unregister_class(ui_class)

if __name__ == "__main__":
    register()
