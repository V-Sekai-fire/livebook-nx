import json
import math
import os
import random
from dataclasses import dataclass, field
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transform
from PIL import Image
import imageio
import cv2
from torch.utils.data import DataLoader, Dataset

import spuv
from spuv.utils.config import parse_structured
from spuv.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
)
from spuv.utils.typing import *
from spuv.utils.mesh_utils import *
from spuv.data.camera_strategy import camera_functions


def _parse_scene_list(scene_path):
    data = []
    with open(scene_path, 'r') as file:
        for line in file:
            try:
                # Parse each line as a JSON object
                json_data = json.loads(line.strip())
                data.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} in line: {line}")

    return data

def file_from_uv_dir(uv_dir, mes_name="model.obj", uv_name="model.png"):
    mesh_file_path = os.path.join(uv_dir, mes_name)
    uv_image_path = os.path.join(uv_dir, uv_name)

    return mesh_file_path, uv_image_path


@dataclass
class ObjaverseDataModuleConfig:
    scene_list: Any = ""
    eval_scene_list: Any = ""
    repeat: int = 1  # for debugging purpose
    camera_strategy: str = "strategy_1"
    eval_camera_strategy: str = "strategy_1"
    height: int = 128
    width: int = 128
    cond_views: int = 1
    sup_views: int = 4
    uv_height: int = 1024
    uv_width: int = 1024
    uv_channel: int = 3

    train_indices: Optional[Tuple[int, int]] = None
    val_indices: Optional[Tuple[int, int]] = None
    test_indices: Optional[Tuple[int, int]] = None

    batch_size: int = 1
    num_workers: int = 16

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_cond_views: int = 1
    eval_sup_views: int = 12

    vertex_transformation: bool = False
    mesh_name: str = "model.obj"
    uv_name: str = "model.png"


class ObjaverseDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: ObjaverseDataModuleConfig = cfg
        self.split = split

        if self.split == "train" and self.cfg.train_indices is not None:
            self.all_uvs = _parse_scene_list(self.cfg.scene_list)
            self.all_uvs = self.all_uvs[
                               self.cfg.train_indices[0]: self.cfg.train_indices[1]
                               ]
            self.all_uvs = self.all_uvs * self.cfg.repeat
        elif self.split == "val" and self.cfg.val_indices is not None:
            self.all_uvs = _parse_scene_list(self.cfg.eval_scene_list)
            self.all_uvs = self.all_uvs[
                               self.cfg.val_indices[0]: self.cfg.val_indices[1]
                               ]
        elif self.split == "test" and self.cfg.test_indices is not None:
            self.all_uvs = _parse_scene_list(self.cfg.eval_scene_list)
            self.all_uvs = self.all_uvs[
                               self.cfg.test_indices[0]: self.cfg.test_indices[1]
                               ]

    def __len__(self):
        return len(self.all_uvs)

    def __getitem__(self, index):
        try:
            return self.try_get_item(index)
        except Exception as e:
            print(f"Failed to load {index}: {e}")
            return None

    def vertex_transform(self, mesh):
        pre_vertices = mesh['v_pos']

        vertices = torch.clone(pre_vertices)
        vertices[:, 1] = -pre_vertices[:, 2]  # -z --> y
        vertices[:, 2] = pre_vertices[:, 1]  # y --> z

        bounding_box_max = vertices.max(0)[0]
        bounding_box_min = vertices.min(0)[0]
        mesh_scale = 1.0
        scale = mesh_scale / ((bounding_box_max - bounding_box_min).max() + 1e-6)
        center_offset = (bounding_box_max + bounding_box_min) * 0.5
        vertices = (vertices - center_offset) * scale

        mesh['v_pos'] = vertices

        return mesh

    def try_get_item(self, index):
        scene_index = index
        uv_data = self.all_uvs[scene_index]
        scene_id = uv_data["id"]
        data_root_dir = uv_data["root_dir"]
        uv_dir = f"{data_root_dir}/{scene_id[:2]}/{scene_id}"
        prompt = uv_data["result"]
        mesh_file_path, uv_image_path = file_from_uv_dir(uv_dir, self.cfg.mesh_name, self.cfg.uv_name)

        mesh, uv_map = load_mesh_and_uv(
            mesh_file_path,
            uv_image_path,
        )

        if self.cfg.vertex_transformation:
            mesh = self.vertex_transform(mesh)

        if self.split == "train":
            cond_views = self.cfg.cond_views
            sup_views = self.cfg.sup_views
            camera_func: Callable = camera_functions[self.cfg.camera_strategy]
            camera_info = camera_func(mesh, cond_views, sup_views,)
            cond_sup_fovy = camera_info["cond_sup_fovy"]
            cond_sup_c2w = camera_info["cond_sup_c2w"]
            height = self.cfg.height
            width = self.cfg.width
        elif self.split == "val" or self.split == "test":
            cond_views = self.cfg.eval_cond_views
            sup_views = self.cfg.eval_sup_views
            camera_func: Callable = camera_functions[self.cfg.eval_camera_strategy]
            camera_info = camera_func(mesh, cond_views, sup_views,)
            cond_sup_fovy = camera_info["cond_sup_fovy"]
            cond_sup_c2w = camera_info["cond_sup_c2w"]
            height = self.cfg.eval_height
            width = self.cfg.eval_width
        else:
            raise NotImplementedError

        cond_sup_proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            cond_sup_fovy, width / height, 0.1, 1000.0
        )
        cond_sup_mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(cond_sup_c2w, cond_sup_proj_mtx)

        assert cond_views + sup_views == len(cond_sup_c2w)

        data_out = {}
        data_out["c2w_cond"] = cond_sup_c2w[:cond_views]
        data_out["mvp_mtx_cond"] = cond_sup_mvp_mtx[:cond_views]
        data_out["fovy_cond"] = cond_sup_fovy[:cond_views]
        data_out["c2w"] = cond_sup_c2w[cond_views:cond_views + sup_views]
        data_out["mvp_mtx"] = cond_sup_mvp_mtx[cond_views:cond_views + sup_views]
        data_out["fovy"] = cond_sup_fovy[cond_views:cond_views + sup_views]

        out = {
            **data_out,
            "index": torch.as_tensor(scene_index),
            "scene_id": scene_id,
            "uv_dir": uv_dir,
            "mesh": mesh,
            "uv_map": uv_map[..., :3],
            "height": height,
            "width": width,
            "cond_views": cond_views,
            "sup_views": sup_views,
            "uv_height": self.cfg.uv_height,
            "uv_width": self.cfg.uv_width,
            "uv_channel": self.cfg.uv_channel,
            "prompt": prompt,
        }

        return out

    # Important! Handle mesh data
    def collate(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        meshes = [item['mesh'] for item in batch]
        for item in batch:
            del item['mesh']
        batch = torch.utils.data.default_collate(batch)
        batch['mesh'] = meshes

        if self.split == "train":
            batch.update({"height": self.cfg.height, "width": self.cfg.width})
        else:
            batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})

        return batch


class ObjaverseDataModule(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            # self.test_dataset = ObjaverseOrbitDataset(self.cfg, "test")
            self.test_dataset = ObjaverseDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = {
        "scene_list": "",
        "eval_scene_list": "",
        "test_indices": [0, 100],
        "uv_name": "",
        "height": 512,
        "width": 512,
        "batch_size": 1,
        "num_workers": 0,
    }

    dataset = ObjaverseDataModule(conf)
    dataset.setup()
    dataloader = dataset.test_dataloader()

    for batch in dataloader:
        breakpoint()

    breakpoint()
