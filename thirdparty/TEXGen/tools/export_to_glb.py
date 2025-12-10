import argparse
import json
import os
import pathlib

import bpy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_indices_path", default="assets/input_list/test_input.jsonl", type=str, help="Path to the input indices jsonl file")
    parser.add_argument("--texture_exp_dir", required=True, type=str, help="Path to the texture experiment directory, e.g. outputs_test/test/test@20240831-163254/save/it0-test")
    parser.add_argument("--output_glb_dir", required=True, type=str, help="Path to the output directory")
    return parser.parse_args()

args = parse_args()
INPUT_INDICES_PATH = args.input_indices_path
YOUR_TEXTURE_EXP_DIR = args.texture_exp_dir
OUTPUT_GLB_DIR = args.output_glb_dir

gt_path_dict = {}
with open(INPUT_INDICES_PATH, "r") as f:
    lines = f.readlines()
    json_list = [json.loads(line) for line in lines]
    for json_dict in json_list:
        gt_path_dict[json_dict["id"]] = json_dict["root_dir"]

def load_obj_and_set_texture(obj_path, image_path, export_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    bpy.ops.wm.obj_import(filepath=obj_path)

    image = bpy.data.images.load(image_path)

    texture = bpy.data.textures.new('TextureName', type='IMAGE')
    texture.image = image

    material = bpy.data.materials.new(name="MaterialName")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    tex_image = material.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = image
    material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)

    bpy.ops.export_scene.gltf(filepath=export_path, export_format='GLB')

def get_file_path(name, project="texgen"):
    """
    return: base_name, mesh, img
    """
    if project == "texgen":
        root_dir = gt_path_dict[name]
        base_dir = os.path.join(root_dir, name[:2], name)
        mesh = os.path.join(base_dir, "model.obj")
        img = os.path.join(YOUR_TEXTURE_EXP_DIR, name+".png")
        breakpoint()
    else:
        raise NotImplementedError
    return name, mesh, img


if __name__ == "__main__":
    id_list = list(gt_path_dict.keys())
    for id in tqdm(id_list):
        name, mesh, img = get_file_path(id, "texgen")
        export_path = f"{OUTPUT_GLB_DIR}/{name}.glb"
        pathlib.Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        load_obj_and_set_texture(str(mesh), str(img), str(export_path))
        