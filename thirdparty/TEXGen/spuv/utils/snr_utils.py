from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F
from einops import rearrange

import spuv
from spuv.systems.base import BaseLossConfig, BaseSystem
from spuv.utils.ops import binary_cross_entropy, get_plucker_rays
from spuv.utils.typing import *
from spuv.models.lpips import LPIPS
from spuv.utils.misc import time_recorder as tr

from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor


def compute_snr_from_scheduler(num_train_timesteps, noise_scheduler):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        snr_weights = []
        for timesteps in range(num_train_timesteps):
            alphas_cumprod = noise_scheduler.alphas_cumprod
            sqrt_alphas_cumprod = alphas_cumprod**0.5
            sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

            # Expand the tensors.
            # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].float()
            alpha = sqrt_alphas_cumprod

            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].float()
            sigma = sqrt_one_minus_alphas_cumprod

            # Compute SNR.
            snr = (alpha / sigma) ** 2
            snr_weights.append(snr)
        return snr_weights

def get_weights_from_timesteps(timesteps, snr_weights, mode='min_snr', prediction_type='sample'):
    """
    Returns the weights for the given timesteps.
    """
    if mode == 'snr':
        selected_weights = torch.tensor([snr_weights[i] for i in timesteps.tolist()], device=timesteps.device)
    elif mode == 'min_snr':
        gamma = 4 * torch.ones_like(timesteps, device=timesteps.device)
        snr = torch.tensor(snr_weights, device=timesteps.device)[timesteps]
        min_snr_weight = torch.stack([snr, gamma], dim=1).min(dim=1)[0]
        if prediction_type == 'sample':
            selected_weights = min_snr_weight
        elif prediction_type == 'noise':
            selected_weights = min_snr_weight / snr
        elif prediction_type == 'v_prediction':
            selected_weights = min_snr_weight / (snr + 1)
    elif mode == 'soft_min_snr':
        gamma = 4 * torch.ones_like(timesteps, device=timesteps.device)
        snr = torch.tensor(snr_weights, device=timesteps.device)[timesteps]
        soft_min_snr_weight = (snr * gamma)/(snr + gamma)
        if prediction_type == 'sample':
            selected_weights = soft_min_snr_weight
        elif prediction_type == 'noise':
            selected_weights = soft_min_snr_weight / snr
        elif prediction_type == 'v_prediction':
            selected_weights = soft_min_snr_weight / (snr + 1)

    return selected_weights


if __name__ == "__main__":
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    prediction_type='sample')
    snr_weights = compute_snr_from_scheduler(1000, noise_scheduler)
    timesteps = torch.randint(0, 1000, (8,))
    timesteps = timesteps.long()
    breakpoint()
    selected_weights = get_weights_from_timesteps(timesteps, snr_weights)

