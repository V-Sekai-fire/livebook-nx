import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spuv.utils.typing import *


def camera_strategy_1(
        mesh: Dict,
        cond_views: int = 1,
        sup_views: int = 4,
        **kwargs) -> Dict:
    """
    Basic camera strategy for training, fixed elevation and distance and fov, uniform azimuth.
    :param cond_views: number of conditional views
    :param sup_views: number of supervision views
    :param kwargs: additional arguments
    """
    # Default camera intrinsics
    default_elevation = 10
    default_camera_views = 36
    default_camera_lens = 50
    default_camera_sensor_width = 36
    default_fovy = 2 * np.arctan(default_camera_sensor_width / (2 * default_camera_lens))

    bbox_size = mesh['v_pos'].max(dim=0)[0] - mesh['v_pos'].min(dim=0)[0]
    distance = default_camera_lens / default_camera_sensor_width * \
               math.sqrt(bbox_size[0] ** 2 + bbox_size[1] ** 2 + bbox_size[2] ** 2)

    all_azimuth_deg = torch.linspace(0, 360.0, default_camera_views + 1)[:default_camera_views]
    all_elevation_deg = torch.full_like(all_azimuth_deg, default_elevation)
    all_view_idxs = torch.arange(default_camera_views)

    # Select front view ids: 0-8 and 27-35
    front_view_idxs = torch.cat([torch.arange(0, 9), torch.arange(27, 36)])
    # Randomly select conditional views
    shuffle_indices = torch.randperm(front_view_idxs.nelement())
    cond_view_idxs = front_view_idxs[shuffle_indices[:cond_views]]
    # Randomly select supervision views from the remaining views
    remaining_views = all_view_idxs[~torch.isin(all_view_idxs, cond_view_idxs)]
    shuffle_indices = torch.randperm(remaining_views.nelement())
    sup_view_idxs = remaining_views[shuffle_indices[:sup_views]]

    # Get the corresponding azimuth and elevation
    view_idxs = torch.cat([cond_view_idxs, sup_view_idxs])
    azimuth = all_azimuth_deg[view_idxs]
    elevation = all_elevation_deg[view_idxs]
    camera_distances = torch.full_like(elevation, distance)
    c2w = get_c2w(azimuth, elevation, camera_distances)

    fovy = torch.full_like(azimuth, default_fovy)

    return {
        'cond_sup_view_idxs': view_idxs,
        'cond_sup_c2w': c2w,
        'cond_sup_fovy': fovy,
    }

def camera_strategy_2(
        mesh: Dict,
        cond_views: int = 1,
        sup_vies: int = 4,
        **kwargs) -> Dict:
    """
    For sup views: Random elevation and azimuth, fixed distance and close fov.
    :param cond_views: number of conditional views
    :param sup_views: number of supervision views
    :param kwargs: additional arguments
    """
    # Default camera intrinsics
    default_elevation = 10
    default_camera_views = 36
    default_camera_lens = 50
    default_camera_sensor_width = 36
    default_fovy = 2 * np.arctan(default_camera_sensor_width / (2 * default_camera_lens))

    bbox_size = mesh['v_pos'].max(dim=0)[0] - mesh['v_pos'].min(dim=0)[0]
    distance = default_camera_lens / default_camera_sensor_width * \
               math.sqrt(bbox_size[0] ** 2 + bbox_size[1] ** 2 + bbox_size[2] ** 2)

    all_azimuth_deg = torch.linspace(0, 360.0, default_camera_views + 1)[:default_camera_views]
    all_elevation_deg = torch.full_like(all_azimuth_deg, default_elevation)
    all_view_idxs = torch.arange(default_camera_views)

    # Select front view ids: 0-8 and 27-35
    front_view_idxs = torch.cat([torch.arange(0, 9), torch.arange(27, 36)])
    # Randomly select conditional views
    shuffle_indices = torch.randperm(front_view_idxs.nelement())
    cond_view_idxs = front_view_idxs[shuffle_indices[:cond_views]]
    # Randomly select supervision views from the remaining views
    remaining_views = all_view_idxs[~torch.isin(all_view_idxs, cond_view_idxs)]
    shuffle_indices = torch.randperm(remaining_views.nelement())
    sup_view_idxs = remaining_views[shuffle_indices[:sup_vies]]

    # Get the corresponding azimuth and elevation
    view_idxs = torch.cat([cond_view_idxs, sup_view_idxs])
    azimuth = all_azimuth_deg[view_idxs]
    elevation = all_elevation_deg[view_idxs]
    camera_distances = torch.full_like(elevation, distance)

    # for sup view
    random_elevation_noise = torch.rand(sup_vies) * 70 - 30  # [-30, 40]
    elevation[cond_views:] += random_elevation_noise
    c2w = get_c2w(azimuth, elevation, camera_distances)

    fovy = torch.full_like(azimuth, default_fovy)
    random_fovy_scale = torch.rand(sup_vies) * 0.7 + 0.3  # [0.3, 1]
    fovy[cond_views:] *= random_fovy_scale

    return {
        'cond_sup_view_idxs': view_idxs,
        'cond_sup_c2w': c2w,
        'cond_sup_fovy': fovy,
    }

def camera_strategy_3(
        mesh: Dict,
        cond_views: int = 1,
        sup_views: int = 4,
        **kwargs) -> Dict:
    """
    Basic camera strategy for training, fixed elevation and distance and fov, uniform azimuth.
    :param cond_views: number of conditional views
    :param sup_views: number of supervision views
    :param kwargs: additional arguments
    """
    # Default camera intrinsics
    default_elevation = 10
    default_camera_views = 36
    default_camera_lens = 50
    default_camera_sensor_width = 36
    default_fovy = 2 * np.arctan(default_camera_sensor_width / (2 * default_camera_lens))

    bbox_size = mesh['v_pos'].max(dim=0)[0] - mesh['v_pos'].min(dim=0)[0]
    distance = default_camera_lens / default_camera_sensor_width * \
               math.sqrt(bbox_size[0] ** 2 + bbox_size[1] ** 2 + bbox_size[2] ** 2)

    all_azimuth_deg = torch.linspace(0, 360.0, default_camera_views + 1)[:default_camera_views]
    all_elevation_deg = torch.full_like(all_azimuth_deg, default_elevation)
    all_view_idxs = torch.arange(default_camera_views)

    # Select front view ids: 0-4 and 31-35
    front_view_idxs = torch.cat([torch.arange(0, 4), torch.arange(31, 36)])
    # Randomly select conditional views
    shuffle_indices = torch.randperm(front_view_idxs.nelement())
    cond_view_idxs = front_view_idxs[shuffle_indices[:cond_views]]
    # Randomly select supervision views from the remaining views
    remaining_views = all_view_idxs[~torch.isin(all_view_idxs, cond_view_idxs)]
    shuffle_indices = torch.randperm(remaining_views.nelement())
    sup_view_idxs = remaining_views[shuffle_indices[:sup_views]]

    # Get the corresponding azimuth and elevation
    view_idxs = torch.cat([cond_view_idxs, sup_view_idxs])
    azimuth = all_azimuth_deg[view_idxs]
    elevation = all_elevation_deg[view_idxs]
    camera_distances = torch.full_like(elevation, distance)
    c2w = get_c2w(azimuth, elevation, camera_distances)

    fovy = torch.full_like(azimuth, default_fovy)

    return {
        'cond_sup_view_idxs': view_idxs,
        'cond_sup_c2w': c2w,
        'cond_sup_fovy': fovy,
    }

def camera_strategy_120fps_uniform(
        mesh: Dict,
        cond_views: int = 1,
        sup_vies: int = 4,
        **kwargs) -> Dict:
    """
    For sup views: Random elevation and azimuth, fixed distance and close fov.
    :param cond_views: number of conditional views
    :param sup_views: number of supervision views
    :param kwargs: additional arguments
    """
    # Default camera intrinsics
    assert cond_views == 1 and sup_vies == 120
    default_elevation = 10
    default_camera_views = 120
    default_camera_lens = 50
    default_camera_sensor_width = 36
    default_fovy = 2 * np.arctan(default_camera_sensor_width / (2 * default_camera_lens))

    bbox_size = mesh['v_pos'].max(dim=0)[0] - mesh['v_pos'].min(dim=0)[0]
    distance = default_camera_lens / default_camera_sensor_width * \
               math.sqrt(bbox_size[0] ** 2 + bbox_size[1] ** 2 + bbox_size[2] ** 2)

    all_azimuth_deg = torch.linspace(0, 360.0, default_camera_views + 1)[:default_camera_views] \
                      + torch.randint(-90, 90, (1,))

    all_elevation_deg = torch.full_like(all_azimuth_deg, default_elevation)
    all_view_idxs = torch.arange(default_camera_views)

    cond_view_idxs = torch.arange(0, 1)
    sup_view_idxs = torch.arange(0, 120)

    # Get the corresponding azimuth and elevation
    view_idxs = torch.cat([cond_view_idxs, sup_view_idxs])
    azimuth = all_azimuth_deg[view_idxs]
    elevation = all_elevation_deg[view_idxs]
    camera_distances = torch.full_like(elevation, distance)
    c2w = get_c2w(azimuth, elevation, camera_distances)

    fovy = torch.full_like(azimuth, default_fovy)

    return {
        'cond_sup_view_idxs': view_idxs,
        'cond_sup_c2w': c2w,
        'cond_sup_fovy': fovy,
    }

def camera_strategy_test_1_to_4(
        mesh: Dict,
        cond_views: int = 1,
        sup_vies: int = 4,
        **kwargs) -> Dict:
    """
    For sup views: Random elevation and azimuth, fixed distance and close fov.
    :param cond_views: number of conditional views
    :param sup_views: number of supervision views
    :param kwargs: additional arguments
    """
    # Default camera intrinsics
    assert cond_views == 1 and sup_vies == 4
    default_elevation = 10
    default_camera_views = 4
    default_camera_lens = 50
    default_camera_sensor_width = 36
    default_fovy = 2 * np.arctan(default_camera_sensor_width / (2 * default_camera_lens))

    bbox_size = mesh['v_pos'].max(dim=0)[0] - mesh['v_pos'].min(dim=0)[0]
    distance = default_camera_lens / default_camera_sensor_width * \
               math.sqrt(bbox_size[0] ** 2 + bbox_size[1] ** 2 + bbox_size[2] ** 2)

    all_azimuth_deg = torch.linspace(0, 360.0, default_camera_views + 1)[:default_camera_views] - 60

    all_elevation_deg = torch.full_like(all_azimuth_deg, default_elevation)
    all_view_idxs = torch.arange(default_camera_views)

    cond_view_idxs = torch.arange(0, 1)
    sup_view_idxs = torch.arange(0, 4)

    # Get the corresponding azimuth and elevation
    view_idxs = torch.cat([cond_view_idxs, sup_view_idxs])
    azimuth = all_azimuth_deg[view_idxs]
    elevation = all_elevation_deg[view_idxs]
    camera_distances = torch.full_like(elevation, distance)
    c2w = get_c2w(azimuth, elevation, camera_distances)

    fovy = torch.full_like(azimuth, default_fovy)

    return {
        'cond_sup_view_idxs': view_idxs,
        'cond_sup_c2w': c2w,
        'cond_sup_fovy': fovy,
    }

def camera_strategy_test_1_to_4_90deg(
        mesh: Dict,
        cond_views: int = 1,
        sup_vies: int = 4,
        **kwargs) -> Dict:
    """
    For sup views: Random elevation and azimuth, fixed distance and close fov.
    :param cond_views: number of conditional views
    :param sup_views: number of supervision views
    :param kwargs: additional arguments
    """
    # Default camera intrinsics
    assert cond_views == 1 and sup_vies == 4
    default_elevation = 10
    default_camera_views = 4
    default_camera_lens = 50
    default_camera_sensor_width = 36
    default_fovy = 2 * np.arctan(default_camera_sensor_width / (2 * default_camera_lens))

    bbox_size = mesh['v_pos'].max(dim=0)[0] - mesh['v_pos'].min(dim=0)[0]
    distance = default_camera_lens / default_camera_sensor_width * \
               math.sqrt(bbox_size[0] ** 2 + bbox_size[1] ** 2 + bbox_size[2] ** 2)

    all_azimuth_deg = torch.linspace(0, 360.0, default_camera_views + 1)[:default_camera_views] - 90

    all_elevation_deg = torch.full_like(all_azimuth_deg, default_elevation)
    all_view_idxs = torch.arange(default_camera_views)

    cond_view_idxs = torch.arange(0, 1)
    sup_view_idxs = torch.arange(0, 4)

    # Get the corresponding azimuth and elevation
    view_idxs = torch.cat([cond_view_idxs, sup_view_idxs])
    azimuth = all_azimuth_deg[view_idxs]
    elevation = all_elevation_deg[view_idxs]
    camera_distances = torch.full_like(elevation, distance)
    c2w = get_c2w(azimuth, elevation, camera_distances)

    fovy = torch.full_like(azimuth, default_fovy)

    return {
        'cond_sup_view_idxs': view_idxs,
        'cond_sup_c2w': c2w,
        'cond_sup_fovy': fovy,
    }

def get_c2w(
        azimuth_deg,
        elevation_deg,
        camera_distances,):
    assert len(azimuth_deg) == len(elevation_deg) == len(camera_distances)
    n_views = len(azimuth_deg)
    #camera_distances = torch.full_like(elevation_deg, dis)
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )
    center = torch.zeros_like(camera_positions)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(n_views, 1)
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w

def get_c2w_from_uniform_azimuth(ele, dis, n_views):
    azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[:n_views]
    elevation_deg = torch.full_like(azimuth_deg, ele)
    camera_distances = torch.full_like(elevation_deg, dis)
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )
    center = torch.zeros_like(camera_positions)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(n_views, 1)
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w

camera_functions = {
    "strategy_1": camera_strategy_1,
    "strategy_2": camera_strategy_2,
    "strategy_3": camera_strategy_3,
    "strategy_test": camera_strategy_120fps_uniform,
    "strategy_test_1_to_4": camera_strategy_test_1_to_4,
    "strategy_test_1_to_4_90deg": camera_strategy_test_1_to_4_90deg
}