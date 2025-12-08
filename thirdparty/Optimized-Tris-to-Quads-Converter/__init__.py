import bmesh
import bpy
import subprocess
import sys
from bpy.types import Operator, AddonPreferences

bl_info = {
    "name": "Optimized Tris to Quads Converter",
    "author": "Rulesobeyer (https://github.com/Rulesobeyer/)",
    "version": (1, 2, 2),
    "blender": (4, 0, 0),
    "support": "COMMUNITY",
    "category": "Mesh",
    "description": "Tris to quads by mathematical optimization.",
    "location": "Editmode > Face",
    "warning": "",
    "doc_url": "https://github.com/Rulesobeyer/Tris-Quads-Ex",
}

class InstallPulpOperator(Operator):
    bl_idname = "wm.install_pulp"
    bl_label = "Install Pulp"

    def execute(self, context):
        try:
            import pulp
            self.report({'INFO'}, "Pulp is already installed")
        except ImportError:
            subprocess.call([sys.executable, "-m", "pip", "install", "pulp"])
            self.report({'INFO'}, "Pulp has been installed")
        return {'FINISHED'}

class TrisToQuadsPreferences(AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        try:
            import pulp
            layout.label(text="Pulp is installed")
        except ImportError:
            layout.operator("wm.install_pulp", text="Install Pulp", icon='IMPORT')

class CEF_OT_tris_convert_to_quads_ex(Operator):
    """Tris to Quads"""

    bl_idname = "mesh.tris_convert_to_quads_ex"
    bl_label = "Optimized Tris to Quads Converter"
    bl_description = "Tris to quads by mathematical optimization."
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        
        if len(selected_objects) == 0:
            self.report({"WARNING"}, "Select at least one mesh object.")
            return {"CANCELLED"}

        try:
            # Store the current mode and active object
            original_mode = context.mode
            original_active = context.active_object
            
            # Process each selected object
            processed_count = 0
            for obj in selected_objects:
                # Ensure we're in object mode to change active object
                if context.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode="OBJECT")
                
                # Set this object as active
                context.view_layer.objects.active = obj
                
                # Convert tris to quads for this object (handles mode switching internally)
                self.convert_tris_to_quads(context)
                processed_count += 1
            
            # Restore original mode and active object
            if original_active:
                try:
                    # Check if object still exists by trying to access its name
                    _ = original_active.name
                    if original_active.name in bpy.data.objects:
                        context.view_layer.objects.active = original_active
                        if original_mode == 'EDIT_MESH' and original_active.type == 'MESH':
                            bpy.ops.object.mode_set(mode="EDIT")
                        else:
                            bpy.ops.object.mode_set(mode="OBJECT")
                except (ReferenceError, AttributeError):
                    # Object was deleted, just restore mode if possible
                    if original_mode == 'EDIT_MESH':
                        bpy.ops.object.mode_set(mode="OBJECT")
            
        except ImportError:
            self.report({"ERROR"}, "Pulp is not installed")
            return {"CANCELLED"}

        if processed_count > 1:
            self.report({"INFO"}, f"Processed {processed_count} objects.")
        
        return {"FINISHED"}

    def convert_tris_to_quads(self, context):
        from pulp import PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum, value

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.mode_set(mode="EDIT")
        obj = bpy.context.edit_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()

        m = LpProblem(sense=LpMaximize)
        edges = {}
        for edge in bm.edges:
            if not self.is_valid_edge(edge):
                continue
            ln = edge.calc_length()
            edges[edge] = LpVariable(f"v{len(edges):03}", cat="Binary"), ln
        mx = max([i[1] for i in edges.values()], default=1)
        m.setObjective(lpSum(v * (1 + 0.1 * ln / mx) for edge, (v, ln) in edges.items()))
        self.add_constraints(bm, m, edges)
        self.solve_problem(m, edges, obj)

        bm.free()

    def is_valid_edge(self, edge):
        return (edge.select and len(edge.link_faces) == 2 and
                edge.link_faces[0].select and edge.link_faces[1].select and
                len(edge.link_faces[0].edges) == 3 and len(edge.link_faces[1].edges) == 3)

    def add_constraints(self, bm, problem, edges):
        from pulp import lpSum
        for face in bm.faces:
            if len(face.edges) != 3:
                continue
            vv = [vln[0] for edge in face.edges if (vln := edges.get(edge)) is not None]
            if len(vv) > 1:
                problem += lpSum(vv) <= 1

    def solve_problem(self, problem, edges, obj):
        from pulp import PULP_CBC_CMD, value
        solver = PULP_CBC_CMD(gapRel=0.01, timeLimit=60, msg=False)
        problem.solve(solver)
        if problem.status != 1:
            self.report({"INFO"}, f"{obj.name}: Not solved.")
        else:
            bpy.ops.mesh.select_all(action="DESELECT")
            n = 0
            for edge, (v, _) in edges.items():
                if value(v) > 0.5:
                    edge.select_set(True)
                    n += 1
            self.report({"INFO"}, f"{obj.name}: {n} edges are dissolved.")
            bpy.ops.mesh.dissolve_edges(use_verts=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_face_by_sides(type="NOTEQUAL")
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.mode_set(mode="EDIT")

ui_classes = (CEF_OT_tris_convert_to_quads_ex, InstallPulpOperator, TrisToQuadsPreferences)

def menu_func_tris_to_quads(self, context):
    self.layout.operator(CEF_OT_tris_convert_to_quads_ex.bl_idname)

def register():
    for ui_class in ui_classes:
        bpy.utils.register_class(ui_class)
    bpy.types.VIEW3D_MT_edit_mesh_faces.append(menu_func_tris_to_quads)

def unregister():
    bpy.types.VIEW3D_MT_edit_mesh_faces.remove(menu_func_tris_to_quads)
    for ui_class in ui_classes:
        bpy.utils.unregister_class(ui_class)

if __name__ == "__main__":
    register()
