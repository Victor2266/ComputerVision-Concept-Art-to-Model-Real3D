import math
import os
import functools
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image

from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
    get_rays,
    get_ray_directions
)

class OptimizedTSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int
        image_tokenizer_cls: str
        image_tokenizer: dict
        tokenizer_cls: str
        tokenizer: dict
        backbone_cls: str
        backbone: dict
        post_processor_cls: str
        post_processor: dict
        decoder_cls: str
        decoder: dict
        renderer_cls: str
        renderer: dict
        use_half_precision: bool = True
        ray_chunk_size: int = 8192
        batch_size: int = 4

    cfg: Config

    def __init__(self, cfg):
        super().__init__(cfg)
        # Enable CUDA optimization
        torch.backends.cudnn.benchmark = True
        self.token_cache = {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str, device: str = "cuda"):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
            use_saved_ckpt = True
        else:
            config_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=config_name)
            weight_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=weight_name)
            use_saved_ckpt = False

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        
        # Load weights and move to specified device
        ckpt = torch.load(weight_path, map_location="cpu")
        if use_saved_ckpt:
            if "module" in list(ckpt["state_dict"].keys())[0]:
                ckpt = {key.replace('module.',''): item for key, item in ckpt["state_dict"].items()}
            else:
                ckpt = ckpt["state_dict"]
        
        model.load_state_dict(ckpt)
        model = model.to(device)
        
        # Convert to half precision if configured
        if cfg.use_half_precision:
            model = model.half()
            
        return model

    def configure(self):
        super().configure()
        # JIT compile the decoder for faster inference
        self.decoder = torch.jit.script(self.decoder)
        
    @functools.lru_cache(maxsize=32)
    def _cached_tokenize_image(self, image_hash):
        """Cache tokenization results for repeated images"""
        return self.image_tokenizer(image_hash)

    @torch.inference_mode()
    def forward(self, inputs: torch.FloatTensor, rays_o: torch.FloatTensor, rays_d: torch.FloatTensor):
        """Optimized forward pass using inference mode"""
        if self.cfg.use_half_precision:
            inputs = inputs.half()
            rays_o = rays_o.half()
            rays_d = rays_d.half()

        batch_size, n_views = rays_o.shape[:2]
        
        # Process image tokens with caching
        input_hash = hash(inputs.cpu().numpy().tobytes())
        if input_hash in self.token_cache:
            input_image_tokens = self.token_cache[input_hash]
        else:
            input_image_tokens = self.image_tokenizer(inputs)
            self.token_cache[input_hash] = input_image_tokens

        input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C')
        tokens = self.tokenizer(batch_size)
        tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens)
        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        
        # Batch process the views
        scene_codes = rearrange(scene_codes.unsqueeze(1).repeat(1,n_views,1,1,1,1),
                              'b Nv Np Ct Hp Wp -> (b Nv) Np Ct Hp Wp')
        
        # Process rays in chunks for memory efficiency
        rays_o = rearrange(rays_o, 'b Nv h w c -> (b Nv) (h w) c')
        rays_d = rearrange(rays_d, 'b Nv h w c -> (b Nv) (h w) c')
        
        h, w = rays_o.shape[1:3]
        chunks = torch.split(torch.arange(rays_o.shape[1]), self.cfg.ray_chunk_size)
        
        render_images = []
        render_masks = []
        
        for chunk in chunks:
            chunk_rays_o = rays_o[:, chunk]
            chunk_rays_d = rays_d[:, chunk]
            
            chunk_results = self.renderer(self.decoder, 
                                        scene_codes, 
                                        chunk_rays_o, 
                                        chunk_rays_d, 
                                        return_mask=True)
            
            render_images.append(chunk_results[0])
            render_masks.append(chunk_results[1])
            
        render_images = torch.cat(render_images, dim=1)
        render_masks = torch.cat(render_masks, dim=1)
        
        # Reshape back to original dimensions
        render_images = rearrange(render_images, '(b Nv) (h w) c -> b Nv c h w', 
                                Nv=n_views, h=h, w=w)
        render_masks = rearrange(render_masks, '(b Nv) (h w) c -> b Nv c h w', 
                               Nv=n_views, h=h, w=w)
        
        return {'images_rgb': render_images, 
                'images_weight': render_masks}

    @torch.inference_mode()
    def render_360(self, scene_codes, n_views: int, elevation_deg: float = 0.0,
                  camera_distance: float = 1.9, fovy_deg: float = 40.0,
                  height: int = 256, width: int = 256, return_type: str = "pil"):
        """Optimized 360 rendering with batched processing"""
        rays_o, rays_d = get_spherical_cameras(n_views, elevation_deg, camera_distance, 
                                             fovy_deg, height, width)
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)
        
        if self.cfg.use_half_precision:
            rays_o = rays_o.half()
            rays_d = rays_d.half()
            scene_codes = scene_codes.half()

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray((image.detach().cpu().numpy() * 255.0).astype(np.uint8))
            else:
                raise NotImplementedError

        # Process in batches
        batch_size = self.cfg.batch_size
        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(0, n_views, batch_size):
                batch_end = min(i + batch_size, n_views)
                batch_rays_o = rays_o[i:batch_end]
                batch_rays_d = rays_d[i:batch_end]
                
                # Flatten batch for processing
                flat_rays_o = batch_rays_o.reshape(-1, *batch_rays_o.shape[2:])
                flat_rays_d = batch_rays_d.reshape(-1, *batch_rays_d.shape[2:])
                
                image_batch = self.renderer(self.decoder, scene_code, flat_rays_o, flat_rays_d)
                
                # Process and store results
                for img in image_batch:
                    images_.append(process_output(img))
            
            images.append(images_)
        
        return images