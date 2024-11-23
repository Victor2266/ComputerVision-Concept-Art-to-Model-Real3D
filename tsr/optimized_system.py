import math
import os
from dataclasses import dataclass, field
from typing import List, Union
import functools

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

    cfg: Config

    def __init__(self, cfg):
        super().__init__(cfg)
        torch.backends.cudnn.benchmark = True
        self.use_half_precision = True
        self.batch_size = 4
        self.ray_chunk_size = 8192

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

    def to_half(self):
        """Convert all model components to half precision"""
        self.image_tokenizer = self.image_tokenizer.half()
        self.tokenizer = self.tokenizer.half()
        self.backbone = self.backbone.half()
        self.post_processor = self.post_processor.half()
        self.decoder = self.decoder.half()
        self.renderer = self.renderer.half()
        return self

    def get_latent_from_img(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        with torch.inference_mode():
            print("Input type:", type(image))
            print("First element type:", type(image[0]))
            if hasattr(image[0], 'size'):
                print("Size:", image[0].size)
            elif hasattr(image[0], 'shape'):
                print("Shape:", image[0].shape)
            
            # Process input image
            rgb_cond = self.image_processor(image, self.cfg.cond_image_size)
            print("After image processor shape:", rgb_cond.shape)
            
            # Move to device first
            rgb_cond = rgb_cond.to(device)
            
            # Convert to half precision if needed
            if self.use_half_precision:
                rgb_cond = rgb_cond.half()
            
            # Ensure correct shape before adding batch dimension
            if len(rgb_cond.shape) == 4:  # [B, H, W, C]
                rgb_cond = rgb_cond.permute(0, 3, 1, 2)  # [B, C, H, W]
            elif len(rgb_cond.shape) == 3:  # [H, W, C]
                rgb_cond = rgb_cond.unsqueeze(0).permute(0, 3, 1, 2)
                
            print("After permute shape:", rgb_cond.shape)
            
            # Add view dimension
            rgb_cond = rgb_cond.unsqueeze(1)  # [B, 1, C, H, W]
            print("Final shape before tokenizer:", rgb_cond.shape)
            
            batch_size = rgb_cond.shape[0]
            
            # Process with image tokenizer
            input_image_tokens = self.image_tokenizer(rgb_cond)
            print("Token shape:", input_image_tokens.shape)
            print("Token dtype:", input_image_tokens.dtype)
            
            # Ensure consistent dtype
            if self.use_half_precision:
                input_image_tokens = input_image_tokens.half()
            
            input_image_tokens = rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)
            tokens = self.tokenizer(batch_size)
            
            # Ensure tokens have the same dtype
            if self.use_half_precision:
                tokens = tokens.half()
            
            tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens)
            scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
            
            return scene_codes

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str):
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
        
        ckpt = torch.load(weight_path, map_location="cpu")
        if use_saved_ckpt:
            if "module" in list(ckpt["state_dict"].keys())[0]:
                ckpt = {key.replace('module.',''): item for key, item in ckpt["state_dict"].items()}
            else:
                ckpt = ckpt["state_dict"]
        
        model.load_state_dict(ckpt)
        
        if model.use_half_precision:
            model.to_half()
            
        return model

    @torch.inference_mode()
    def forward(self, inputs: torch.FloatTensor, rays_o: torch.FloatTensor, rays_d: torch.FloatTensor):
        if self.use_half_precision:
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
        
        # Batch process views
        scene_codes = rearrange(scene_codes.unsqueeze(1).repeat(1,n_views,1,1,1,1),
                              'b Nv Np Ct Hp Wp -> (b Nv) Np Ct Hp Wp')
        
        # Process rays in chunks
        rays_o = rearrange(rays_o, 'b Nv h w c -> (b Nv) (h w) c')
        rays_d = rearrange(rays_d, 'b Nv h w c -> (b Nv) (h w) c')
        
        chunk_size = self.ray_chunk_size
        h, w = rays_o.shape[1:3]
        all_render_images = []
        all_render_masks = []
        
        for i in range(0, rays_o.shape[1], chunk_size):
            chunk_rays_o = rays_o[:, i:i+chunk_size]
            chunk_rays_d = rays_d[:, i:i+chunk_size]
            
            chunk_render_images, chunk_render_masks = self.renderer(
                self.decoder, 
                scene_codes, 
                chunk_rays_o, 
                chunk_rays_d,
                return_mask=True
            )
            
            all_render_images.append(chunk_render_images)
            all_render_masks.append(chunk_render_masks)
        
        render_images = torch.cat(all_render_images, dim=1)
        render_masks = torch.cat(all_render_masks, dim=1)
        
        # Reshape back
        render_images = rearrange(render_images, '(b Nv) (h w) c -> b Nv c h w', 
                                Nv=n_views, h=h, w=w)
        render_masks = rearrange(render_masks, '(b Nv) (h w) c -> b Nv c h w', 
                               Nv=n_views, h=h, w=w)
        
        return {'images_rgb': render_images, 
                'images_weight': render_masks}

    @functools.lru_cache(maxsize=32)
    def _cached_tokenize_image(self, image_tensor_bytes):
        # Convert the bytes back to tensor and reshape correctly
        tensor = torch.from_numpy(np.frombuffer(image_tensor_bytes, dtype=np.float32).copy())
        # Calculate the proper shape based on the input size
        total_elements = tensor.numel()
        height = width = self.cfg.cond_image_size
        channels = 3
        batch = views = 1
        expected_elements = batch * views * channels * height * width
        
        if total_elements != expected_elements:
            # Adjust the shape if needed
            tensor = tensor[:batch * views * channels * height * width]
        
        tensor = tensor.reshape(batch, views, channels, height, width)
        tensor = tensor.to(next(self.image_tokenizer.parameters()).device)
        return self.image_tokenizer(tensor)

    def get_latent_from_img(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        with torch.inference_mode():
            print("Input type:", type(image))
            print("First element type:", type(image[0]))
            if hasattr(image[0], 'size'):
                print("Size:", image[0].size)
            elif hasattr(image[0], 'shape'):
                print("Shape:", image[0].shape)
            
            # Process input image
            rgb_cond = self.image_processor(image, self.cfg.cond_image_size)
            print("After image processor shape:", rgb_cond.shape)
            
            # Ensure correct shape before adding batch dimension
            if len(rgb_cond.shape) == 4:  # [B, H, W, C]
                rgb_cond = rgb_cond.permute(0, 3, 1, 2)  # [B, C, H, W]
            elif len(rgb_cond.shape) == 3:  # [H, W, C]
                rgb_cond = rgb_cond.unsqueeze(0).permute(0, 3, 1, 2)
                
            print("After permute shape:", rgb_cond.shape)
            
            # Add view dimension
            rgb_cond = rgb_cond.unsqueeze(1).to(device)  # [B, 1, C, H, W]
            print("Final shape before tokenizer:", rgb_cond.shape)
            
            if self.use_half_precision:
                rgb_cond = rgb_cond.half()
            
            batch_size = rgb_cond.shape[0]
            
            # Process with image tokenizer
            input_image_tokens = self.image_tokenizer(rgb_cond)
            print("Token shape:", input_image_tokens.shape)
            
            if self.use_half_precision:
                input_image_tokens = input_image_tokens.half()
            
            input_image_tokens = rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)
            tokens = self.tokenizer(batch_size)
            tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens)
            scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
            
            return scene_codes

    @torch.inference_mode()
    def render_360(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)
        
        # Convert cameras to half precision if needed
        if self.use_half_precision:
            scene_codes = scene_codes.half()
            rays_o = rays_o.half()
            rays_d = rays_d.half()

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                # Convert back to float32 for numpy
                if image.dtype == torch.float16:
                    image = image.float()
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                if image.dtype == torch.float16:
                    image = image.float()
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        # Batch processing
        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(0, n_views, self.batch_size):
                batch_end = min(i + self.batch_size, n_views)
                batch_rays_o = rays_o[i:batch_end]
                batch_rays_d = rays_d[i:batch_end]
                
                # Process rays in chunks
                h, w = batch_rays_o.shape[1:3]
                flat_rays_o = batch_rays_o.reshape(-1, 3)
                flat_rays_d = batch_rays_d.reshape(-1, 3)
                
                all_images = []
                for j in range(0, flat_rays_o.shape[0], self.ray_chunk_size):
                    chunk_rays_o = flat_rays_o[j:j+self.ray_chunk_size]
                    chunk_rays_d = flat_rays_d[j:j+self.ray_chunk_size]
                    
                    # Ensure all inputs are in half precision if needed
                    if self.use_half_precision:
                        chunk_rays_o = chunk_rays_o.half()
                        chunk_rays_d = chunk_rays_d.half()
                        scene_code = scene_code.half()
                    
                    chunk_images = self.renderer(
                        self.decoder,
                        scene_code,
                        chunk_rays_o,
                        chunk_rays_d
                    )
                    all_images.append(chunk_images)
                
                batch_images = torch.cat(all_images, dim=0)
                batch_images = batch_images.reshape(batch_end - i, h, w, -1)
                
                # Convert back to float32 for image processing
                if batch_images.dtype == torch.float16:
                    batch_images = batch_images.float()
                
                for img in batch_images:
                    images_.append(process_output(img))
            
            images.append(images_)
        
        return images

    def extract_mesh(self, scene_codes, resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        
        # Convert inputs to half precision if needed
        if self.use_half_precision:
            scene_codes = scene_codes.half()
            
        meshes = []
        for scene_code in scene_codes:
            grid_vertices = self.isosurface_helper.grid_vertices.to(scene_codes.device)
            if self.use_half_precision:
                grid_vertices = grid_vertices.half()
                
            scaled_vertices = scale_tensor(
                grid_vertices,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            
            with torch.inference_mode():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scaled_vertices,
                    scene_code,
                )["density_act"]
                
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            
            # Convert back to float32 for final processing
            v_pos = v_pos.float()
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            
            with torch.inference_mode():
                color = self.renderer.query_triplane(
                    self.decoder,
                    v_pos.half() if self.use_half_precision else v_pos,
                    scene_code,
                )["color"]
                
            # Convert to float32 for mesh creation
            color = color.float()
            
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy(),
            )
            meshes.append(mesh)
            
        return meshes