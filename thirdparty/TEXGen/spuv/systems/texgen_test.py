from dataclasses import dataclass, field
import cv2
import torch
import torch.nn.functional as F
from einops import rearrange

import spuv
from spuv.utils.misc import get_device
from spuv.utils.typing import *
from spuv.utils.misc import time_recorder as tr
from spuv.utils.snr_utils import compute_snr_from_scheduler, get_weights_from_timesteps
from spuv.utils.mesh_utils import uv_padding
from spuv.utils.nvdiffrast_utils import *
from spuv.systems.texgen_base import TEXGenDiffusion as TEXGenBaseSystem


class TEXGenDiffusion(TEXGenBaseSystem):
    @dataclass
    class Config(TEXGenBaseSystem.Config):
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

    def configure(self):
        super().configure()
        self.image_tokenizer = spuv.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.sigma_min=0.000001

    def get_conditional_flow(self, noise, sample, t):
        t = t[:, None, None, None]
        return (1 - (1 - self.sigma_min) * t) * noise + t * sample

    def prepare_diffusion_data(self, batch, noisy_images=None):
        device = get_device()
        uv_channel, uv_height, uv_width = batch["uv_channel"][0], batch["uv_height"][0], batch["uv_width"][0]
        batch_size = len(batch["mesh"])
        uv_shape = (batch_size, uv_channel, uv_height, uv_width)
        if self.training or "uv_map" in batch:
            sample_images = rearrange(batch["uv_map"], "B H W C -> B C H W").to(dtype=self.dtype)
            if self.cfg.data_normalization:
                sample_images = (sample_images * 2 - 1)
        else:
            sample_images = None

        if "mask_map" not in batch or "position_map" not in batch:
            position_map_, mask_map_ = rasterize_batched_geometry_maps(
                self.ctx, batch["mesh"],
                uv_height,
                uv_width
            )
            mask_map = rearrange(mask_map_, "B H W C-> B C H W").to(dtype=self.dtype)
            position_map = rearrange(position_map_, "B H W C -> B C H W").to(dtype=self.dtype)
        else:
            mask_map = rearrange(batch["mask_map"], "B H W -> B 1 H W").to(dtype=self.dtype)
            position_map = rearrange(batch["position_map"], "B H W C -> B C H W").to(dtype=self.dtype)

        # timesteps = torch.rand(batch_size, device=device)
        # Sample uniformly
        uniform_samples = torch.rand(batch_size, device=device)
        # Apply power transformation to skew towards smaller t
        power = 2  # >1 to skew towards 0
        timesteps = uniform_samples ** power

        if noisy_images is not None:
            noisy_images = noisy_images.to(dtype=self.dtype)
        else:
            noise = torch.randn(uv_shape, device=device, dtype=self.dtype)
            if sample_images is not None:
                noisy_images = self.get_conditional_flow(
                        noise,
                        sample_images,
                        timesteps
                    )
            else:
                noisy_images = noise

        noisy_images *= mask_map

        loss_weights = torch.ones_like(timesteps, device=device, dtype=self.dtype)

        diffusion_data = {
            "sample_images": sample_images,
            "position_map": position_map,
            "mask_map": mask_map,
            "timesteps": timesteps,
            "noise": noise,
            "noisy_images": noisy_images,
            "batch_loss_weights": loss_weights,
        }

        return diffusion_data

    def forward(self,
                condition: Dict[str, Any],
                diffusion_data: Dict[str, Any],
                condition_drop=None,
                ) -> Dict[str, Any]:
        mask_map = diffusion_data["mask_map"]
        position_map = diffusion_data["position_map"]
        timesteps = diffusion_data["timesteps"]
        input_tensor = diffusion_data["noisy_images"]

        text_embeddings = condition["text_embeddings"]
        image_embeddings = condition["image_embeddings"]
        clip_embeddings = [text_embeddings, image_embeddings]

        mesh = condition["mesh"]

        image_info = {
            'mvp_mtx_cond': condition["mvp_mtx_cond"],
            'rgb_cond': condition["rgb_cond"],
        }

        if condition_drop is None and self.training:
            condition_drop = torch.rand(input_tensor.shape[0], device=input_tensor.device) < self.cfg.condition_drop_rate
            condition_drop = condition_drop.float()
        elif condition_drop is None:
            condition_drop = torch.zeros(input_tensor.shape[0], device=input_tensor.device)

        output, addition_info = self.backbone(
           input_tensor,
           mask_map,
           position_map,
           timesteps*1000,
           clip_embeddings,
           mesh,
           image_info,
           data_normalization=self.cfg.data_normalization,
           condition_drop=condition_drop,
        )

        return output, addition_info

    def prepare_condition_info(self, batch):
        mesh = batch["mesh"]
        mvp_mtx_cond = batch["mvp_mtx_cond"]
        uv_map_gt = batch["uv_map"]
        image_height = batch["height"]
        image_width = batch["width"]

        # Online rendering the condition image
        background_color = self.render_background_color
        rgb_cond = render_batched_meshes(self.ctx, mesh, uv_map_gt, mvp_mtx_cond, image_height, image_width, background_color)

        if self.cfg.cond_rgb_perturb and self.training:
            B, Nv, H, W, C = rgb_cond.shape
            rgb_cond = rearrange(rgb_cond, "B Nv H W C -> (B Nv) C H W")
            rgb_cond = self.data_augmentation(rgb_cond, background_color)
            rgb_cond = rearrange(rgb_cond, "(B Nv) C H W -> B Nv H W C", B=B, Nv=Nv)

        prompt = batch["prompt"]
        
        text_embeddings = self.image_tokenizer.process_text(prompt).to(dtype=self.dtype)
        image_embeddings = self.image_tokenizer.process_image(rgb_cond).to(dtype=self.dtype)

        condition_info = {
            "mesh": mesh,
            "mvp_mtx_cond": mvp_mtx_cond,
            "rgb_cond": rgb_cond,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "prompt": prompt,
        }

        return condition_info

    def on_check_train(self, batch, outputs):
        if (
                self.true_global_step < self.cfg.recon_warm_up_steps
                or self.cfg.train_regression
        ):
            self.train_regression = True
        else:
            self.train_regression = False

        if (
                self.global_rank == 0
                and self.cfg.check_train_every_n_steps > 0
                and self.true_global_step % (self.cfg.check_train_every_n_steps*10) == 0
        ):
            images = []
            texture_map_outputs = outputs["texture_map_outputs"]

            for key, value in texture_map_outputs.items():
                if self.cfg.data_normalization:
                    img = (value * 0.5 + 0.5) * outputs["mask_map"]
                else:
                    img = value * outputs["mask_map"]
                img_format = {
                    "type": "rgb",
                    "img": rearrange(img, "B C H W -> (B H) W C"),
                    "kwargs": {"data_format": "HWC"},
                }
                images.append(img_format)

            self.save_image_grid(
                f"it{self.true_global_step}-train.jpg",
                images,
                name="train_step_output",
                step=self.true_global_step,
            )

        if outputs['render_out'] is not None:
            images = [
                {
                    "type": "rgb",
                    "img": rearrange(outputs['render_out'], "B V H W C -> (B H) (V W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": rearrange(outputs['render_gt'], "B V H W C -> (B H) (V W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": rearrange(outputs['rgb_cond'], "B V H W C -> (B H) (V W) C"),
                    "kwargs": {"data_format": "HWC"},
                }
            ]

            self.save_image_grid(
                f"it{self.true_global_step}-train-render.jpg",
                images,
                name="train_step_output",
                step=self.true_global_step,
            )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch is None:
            spuv.info("Received None batch, skipping.")
            return None
        try:
            with torch.cuda.amp.autocast(enabled=False):
                if self.use_ema and self.val_with_ema:
                    with self.ema_scope("Validation with ema weights"):
                        texture_map_outputs = self.test_pipeline(batch)
                else:
                    spuv.info("Validation without ema weights")
                    texture_map_outputs = self.test_pipeline(batch)
        except Exception as e:
            spuv.info(f"Error in test pipeline: {e}")
            return None

        render_images = {}
        background_color = self.render_background_color 

        assert len(batch["scene_id"]) == 1
        save_str = batch["scene_id"][0]

        # save prediction to png file
        value = texture_map_outputs["pred_x0"]
        if self.cfg.data_normalization:
            img = (value * 0.5 + 0.5) * texture_map_outputs["mask_map"]
        else:
            img = value * texture_map_outputs["mask_map"]
        # Important to flip the uv map for possible meshlab loading, for rendering using NvDiffRasterizer, do not flip!
        flip_img = torch.flip(img, dims=[2])

        img_format = [{
            "type": "rgb",
            "img": rearrange(flip_img, "B C H W-> (B H) W C"),
            "kwargs": {"data_format": "HWC"},
        }]

        self.save_image_grid(
            f"it{self.true_global_step}-test/{save_str}.png",
            img_format,
            name=f"test_step_output_{self.global_rank}_{batch_idx}",
            step=self.true_global_step,
        )

        # save preview
        for key in ["pred_x0", "gt_x0", "baked_texture"]:
            value = texture_map_outputs[key]
            if self.cfg.data_normalization:
                img = (value * 0.5 + 0.5) * texture_map_outputs["mask_map"]
            else:
                img = value * texture_map_outputs["mask_map"]
            # Important to flip the uv map for possible meshlab loading, for rendering using NvDiffRasterizer, do not flip!
            flip_img = torch.flip(img, dims=[2])

            img_format = [{
                "type": "rgb",
                "img": rearrange(flip_img, "B C H W-> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            }]

            self.save_image_grid(
                f"it{self.true_global_step}-test/preview/{key}_{self.global_rank}_{batch_idx}.jpg",
                img_format,
                name=f"test_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

            img = rearrange(img, "B C H W -> B H W C")
            mvp_mtx = batch['mvp_mtx']
            mesh = batch['mesh']
            height = batch['height']
            width = batch['width']

            pad_img = uv_padding(img.squeeze(0), texture_map_outputs['mask_map'].squeeze(0).squeeze(0), iterations=2)
            
            render_out = render_batched_meshes(self.ctx, mesh, pad_img, mvp_mtx, height, width, background_color)

            img_format = [{
                "type": "rgb",
                "img": rearrange(render_out, "B (V1 V2) H W C -> (B V1 H) (V2 W) C", V1=4),
                "kwargs": {"data_format": "HWC"},
            }]

            self.save_image_grid(
                f"it{self.true_global_step}-test/preview/render_{key}_{self.global_rank}_{batch_idx}.jpg",
                img_format,
                name=f"test_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

            render_images[key] = torch.clamp(rearrange(render_out, "B V H W C -> (B V) C H W"), min=0, max=1)
        
    def test_pipeline(self, batch):
        diffusion_data = self.prepare_diffusion_data(batch)
        condition_info = self.prepare_condition_info(batch)

        device = get_device()
        test_num_steps = self.cfg.test_num_steps

        B, C, H, W = diffusion_data["mask_map"].shape
        noise = torch.randn((B, 3, H, W), device=device, dtype=self.dtype)
        noisy_images = noise

        t_span=torch.linspace(0, 1, test_num_steps, device=device, dtype=self.dtype)
        delta = 1.0 / test_num_steps

        for i, t in enumerate(t_span):
            timestep = t.repeat(B)
            diffusion_data["timesteps"] = timestep
            diffusion_data["noisy_images"] = noisy_images
            cond_step_out, addition_info = self(condition_info, diffusion_data)

            if (
                    self.cfg.test_cfg_scale != 0.0
                    and self.cfg.guidance_interval[0] <= t <= self.cfg.guidance_interval[1]
            ):
                uncond_step_out, _ = self(condition_info, diffusion_data, condition_drop=torch.ones(B, device=device))
                step_out = uncond_step_out + self.cfg.test_cfg_scale * (cond_step_out - uncond_step_out)
                # Apply guidance rescale. From paper [Common Diffusion Noise Schedules
                # and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) section 3.4.
                if self.cfg.guidance_rescale != 0:
                    std_pos = cond_step_out.std(dim=list(range(1, cond_step_out.ndim)), keepdim=True)
                    std_cfg = step_out.std(dim=list(range(1, step_out.ndim)), keepdim=True)
                    # Fuse equation 15,16 for more efficient computation.
                    step_out *= self.cfg.guidance_rescale * (std_pos / std_cfg) + (1 - self.cfg.guidance_rescale)
            else:
                step_out = cond_step_out

            noisy_images = noisy_images + delta * step_out

        pred_x0 = noisy_images
        texture_map_outputs = {
            "pred_x0": pred_x0,
            "baked_texture": addition_info['baked_texture'],
            "gt_x0": diffusion_data["sample_images"],
            "mask_map": diffusion_data["mask_map"],
        }

        return texture_map_outputs
