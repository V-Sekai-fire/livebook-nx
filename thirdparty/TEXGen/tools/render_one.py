import math
import os
import sys

import bpy


def render_single_mesh(model_path, output_path, rotation=(0, 0, math.pi / 1.5)):
    bpy.ops.import_scene.gltf(filepath=model_path)
    mesh_obj = bpy.context.scene.collection.objects[0]
    bpy.context.scene.render.film_transparent = True

    material = mesh_obj.material_slots[0].material
    if material is None:
        print("Material not found")
        return

    if not material.use_nodes:
        print(f"Material does not use nodes")
        return

    nodes = material.node_tree.nodes

    for node in nodes:
        if node.type == 'TEX_IMAGE':
            nodes.remove(node)

    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            node.inputs['Base Color'].default_value = (0.445, 0.800, 0.415, 1.0)  # set to (R, G, B, A)

    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.ops.transform.rotate(value=rotation[2], orient_axis='Z')

    mesh_obj.active_material.node_tree.nodes[0].inputs['Roughness'].default_value = 0.5

    mesh_obj.data.use_auto_smooth = True

    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    mesh_obj.select_set(True)
    bpy.ops.object.delete()

def render_single(model_path, output_path, rotation=(0, 0, math.pi / 4), texture_path=None):
    bpy.ops.import_scene.gltf(filepath=model_path)
    mesh_obj = bpy.context.scene.collection.objects[0]

    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.ops.transform.rotate(value=rotation[2], orient_axis='Z')

    mesh_obj.active_material.node_tree.nodes[0].inputs['Roughness'].default_value = 0.7

    if texture_path is not None:
        if mesh_obj.active_material and mesh_obj.active_material.use_nodes:
            for node in mesh_obj.active_material.node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    new_img = bpy.data.images.load(filepath=texture_path)
                    node.image = new_img

    mesh_obj.data.use_auto_smooth = True

    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    mesh_obj.select_set(True)
    bpy.ops.object.delete()

glb_path = sys.argv[5]
render_type = sys.argv[6]
pi = math.pi
views = 6
unit_rot = 2 * pi / views


demo_dir = os.path.dirname(glb_path)
filename = glb_path.split("/")[-1].split(".")[0]

if render_type == "mesh":
    view = 0
    output_path = os.path.join(demo_dir, f"mesh_{filename}_{view}.png")
    render_single_mesh(
        glb_path,
        output_path,
        rotation=(0, 0, view * unit_rot)
    )

elif render_type == "view":
    for view in range(views):
        output_path = os.path.join(demo_dir, f"{filename}_{view}.png")
        render_single(
            glb_path,
            output_path,
            rotation=(0, 0, view * unit_rot)
        )
