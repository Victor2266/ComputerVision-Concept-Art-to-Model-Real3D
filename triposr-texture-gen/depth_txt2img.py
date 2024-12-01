import argparse
import sys
import signal
from contextlib import contextmanager
import threading
import os

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
import numpy as np
from PIL import Image
import torch
import gc

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def get_optimal_device():
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # Disable TF32 for better compatibility
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

_DEFAULT_DEVICE = get_optimal_device()

class TextToObjectImage:
    def __init__(
        self,
        device=_DEFAULT_DEVICE,
        model='Lykon/dreamshaper-8',
        cn_model='lllyasviel/control_v11p_sd15_normalbae',
    ):
        self.device = device
        print(f"Initializing on device: {device}")
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        print(f"Loading ControlNet model from {cn_model}...")
        controlnet = ControlNetModel.from_pretrained(
            cn_model, 
            torch_dtype=dtype,
            variant='fp16' if device == 'cuda' else None,
            use_safetensors=True
        )

        print(f"Loading Stable Diffusion model from {model}...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model, 
            controlnet=controlnet, 
            torch_dtype=dtype,
            variant='fp16' if device == 'cuda' else None,
            safety_checker=None,
            use_safetensors=True,
            requires_safety_checker=False
        )
        
        # Move to device after full initialization
        self.pipe = self.pipe.to(device)
        
        if device == "cuda":
            self.pipe.enable_attention_slicing(slice_size="auto")
            self.pipe.enable_vae_slicing()
            torch.cuda.empty_cache()
            gc.collect()
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        print("Setup complete!")

    def resize_to_valid_dimensions(self, image: Image.Image, max_size: int = 768) -> Image.Image:
        """Resize image to valid dimensions for Stable Diffusion while maintaining aspect ratio"""
        # Get original dimensions
        width, height = image.size
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Calculate new dimensions
        if width > height:
            new_width = min(max_size, width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_size, height)
            new_width = int(new_height * aspect_ratio)
            
        # Ensure dimensions are multiples of 8
        new_width = ((new_width + 7) // 8) * 8
        new_height = ((new_height + 7) // 8) * 8
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def generate(self, desc: str, steps: int, control_image: Image.Image):
        print(f"\nStarting image generation with {steps} steps...")
        
        # Clean prompt
        prompt = desc.split('--')[0].strip()
        prompt = prompt[:77]
        print("Using prompt:", prompt)
        
        # Store original dimensions
        original_size = control_image.size
        print(f"Original image size: {original_size}")
        
        # Resize for processing
        control_image = self.resize_to_valid_dimensions(control_image)
        print(f"Processing at size: {control_image.size}")
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        try:
            with time_limit(300):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt='lighting, shadows, grid, dark, mesh',
                    num_inference_steps=steps,
                    num_images_per_prompt=1,
                    image=control_image,
                    guidance_scale=7.5,
                )
                print("\nGeneration complete!")
                # Resize back to original dimensions
                result_image = result.images[0].resize(original_size, Image.Resampling.LANCZOS)
                return result_image
                
        except TimeoutException:
            print("\nGeneration timed out! Trying again with reduced settings...")
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            reduced_image = self.resize_to_valid_dimensions(control_image, max_size=512)
            result = self.pipe(
                prompt=prompt,
                negative_prompt='lighting, shadows, grid, dark, mesh',
                num_inference_steps=max(steps // 2, 6),
                num_images_per_prompt=1,
                image=reduced_image,
                guidance_scale=7.0,
            )
            # Resize back to original dimensions
            result_image = result.images[0].resize(original_size, Image.Resampling.LANCZOS)
            return result_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('desc', help='Short description of desired model appearance')
    parser.add_argument('depth_img', help='Depth control image')
    parser.add_argument('output_path', help='Path for generated image')
    parser.add_argument(
        '--image-model',
        help='SD 1.5-based model for texture gen',
        default='Lykon/dreamshaper-8',
    )
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument(
        '--device',
        default=_DEFAULT_DEVICE,
        type=str,
        help='Device to prefer. Default: try to auto-detect from platform (CUDA, Metal)'
    )
    args = parser.parse_args()

    print("\nInitializing TextToObjectImage...")
    t2i = TextToObjectImage(args.device, args.image_model)
    
    print(f"\nLoading control image from {args.depth_img}...")
    control_image = Image.open(args.depth_img)
    
    result_image = t2i.generate(args.desc, args.steps, control_image)
    
    print(f"Saving generated image to {args.output_path}...")
    result_image.save(args.output_path)
    print("Done!")