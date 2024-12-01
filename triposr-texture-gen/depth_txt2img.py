import argparse
import sys
import signal
from contextlib import contextmanager
import threading

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

_DEFAULT_DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

class TextToObjectImage:
    def __init__(
        self,
        device=_DEFAULT_DEVICE,
        model='Lykon/dreamshaper-8',
        cn_model='lllyasviel/control_v11p_sd15_normalbae',
    ):
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Force model to use less memory
        torch.backends.cudnn.benchmark = False
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.7)
        
        # Adjust dtype based on device
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
        
        # Memory optimizations
        if device == "cuda":
            print("Enabling memory optimizations...")
            self.pipe.enable_attention_slicing(slice_size=1)
            self.pipe.enable_vae_slicing()
            self.pipe = self.pipe.to(device)
            
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        print("Setup complete!")

    def generate(self, desc: str, steps: int, control_image: Image):
        print(f"\nStarting image generation with {steps} steps...")
        print("Using prompt:", desc)
        
        # Resize image if too large
        max_size = 512
        if max(control_image.size) > max_size:
            ratio = max_size / max(control_image.size)
            new_size = tuple(int(dim * ratio) for dim in control_image.size)
            control_image = control_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Clear memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        prompt = f'{desc}, front and back view, 180, reverse, 3D rendering, high quality'
        
        try:
            with time_limit(300):  # 5 minute timeout
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt='lighting, shadows, grid, dark, mesh',
                    num_inference_steps=steps,
                    num_images_per_prompt=1,
                    image=control_image,
                    guidance_scale=7.5,
                    width=control_image.width,
                    height=control_image.height
                )
                print("\nGeneration complete!")
                return result.images[0]
                
        except TimeoutException:
            print("\nGeneration timed out! Trying again with reduced settings...")
            # Try again with reduced settings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            self.pipe.enable_attention_slicing(slice_size=1)
            return self.pipe(
                prompt=prompt,
                negative_prompt='lighting, shadows, grid, dark, mesh',
                num_inference_steps=max(steps // 2, 6),  # Reduce steps but keep minimum of 6
                num_images_per_prompt=1,
                image=control_image,
                guidance_scale=7.0,  # Reduced guidance scale
                width=min(control_image.width, 384),  # Reduced size
                height=min(control_image.height, 384)
            ).images[0]

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print("\nInitializing TextToObjectImage...")
    t2i = TextToObjectImage(args.device, args.image_model)
    
    print(f"\nLoading control image from {args.depth_img}...")
    control_image = Image.open(args.depth_img)
    
    result_image = t2i.generate(args.desc, args.steps, control_image)
    
    print(f"Saving generated image to {args.output_path}...")
    result_image.save(args.output_path)
    print("Done!")
    