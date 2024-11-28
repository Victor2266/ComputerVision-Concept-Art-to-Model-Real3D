import os
import pickle
import json
import tqdm
import cv2
import random
import torch
import numpy as np
import random
import math
import json
import time
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import functional as func_transforms
from torchvision import transforms
import torchvision
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset.constant import *
from utils.geo_utils import get_relative_pose
from utils.process_utils import process_images, get_rays_from_pose
from tsr.utils import get_spherical_cameras

SCALE_RAW = 0.5

def convert_windows_to_wsl_path(windows_path):
    """Convert Windows path to WSL path"""
    # Remove drive letter and convert backslashes to forward slashes
    path = windows_path.replace('\\', '/')
    drive_letter = path[0].lower()
    wsl_path = f"/mnt/{drive_letter}/{path[3:]}"
    return wsl_path

class ObjaverseWintoWSL(Dataset):
    def __init__(self, config, split='train', multiple_data=False, data_name=''):
        self.config = config
        self.split = split
        # Use raw Windows path directly instead of converting
        self.root = convert_windows_to_wsl_path(r"F:\Zero123 Objaverse\gobjaverse_5314_Food")
        self.multiple_data = multiple_data
        self.data_name = data_name
        assert split in ['train', 'val', 'test']

        self.img_size = config.dataset.img_size
        self.num_frame = config.dataset.num_frame if self.split == 'train' else config.dataset.num_frame + 1
        self.render_size = config.model.render_resolution
        
        self.normalization = config.dataset.normalization
        self.canonical_distance = SCALE_OBJAVERSE
        self.canonical_scale = SCALE_TRIPLANE_SAFE

        self.data_split = {}
        self._load_dataset()
        self.seq_names = []
        if self.split == 'train':
            self.seq_names += self.data_split['train']
        else:
            self.seq_names += self.data_split['test']
            if self.split == 'val':
                self.seq_names = self.seq_names[::config.eval_vis_freq]

        print(f"Loaded {len(self.seq_names)} sequences for {split}")
    def _split_data(self, data_split_file_path):
        all_info = {'train': [], 'test': []}
        all_instances_return = []

        print(f"Scanning root directory: {self.root}")
        
        # Navigate through directory structure
        for category in os.listdir(self.root):
            category_path = os.path.join(self.root, category)
            if not os.path.isdir(category_path):
                continue
                
            print(f"Processing category: {category}")
            instance_count = 0
                
            for instance in os.listdir(category_path):
                instance_path = os.path.join(category_path, instance)
                if not os.path.isdir(instance_path):
                    continue
                    
                for subdir in os.listdir(instance_path):
                    subdir_path = os.path.join(instance_path, subdir)
                    if not os.path.isdir(subdir_path):
                        continue
                    
                    #print(f"Self num_frame: {self.num_frame}") # 4
                    # Check for valid instance (should contain the PNG and JSON files)
                    png_file = os.path.join(subdir_path, f"{subdir}.png")
                    json_file = os.path.join(subdir_path, f"{subdir}.json")
                    
                    if os.path.isfile(png_file) and os.path.isfile(json_file):
                        relative_path = os.path.join(category, instance, subdir)
                        all_instances_return.append(relative_path)
                        instance_count += 1
            
            print(f"Found {instance_count} valid instances in category {category}")

        print(f"Total valid instances found: {len(all_instances_return)}")
        
        # Split into train and test
        num_instances = len(all_instances_return)
        num_instances_test = max(50, int(0.1 * num_instances))  # 10% for test, minimum 50
        
        random.shuffle(all_instances_return)
        all_info['train'] = all_instances_return[:-num_instances_test]
        all_info['test'] = all_instances_return[-num_instances_test:]

        print(f"Split data into {len(all_info['train'])} training and {len(all_info['test'])} testing instances")

        # Save split information
        os.makedirs(os.path.dirname(data_split_file_path), exist_ok=True)
        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f)
            
        return all_info

    def _load_dataset(self):
        data_root = './dataset/split_info'
        os.makedirs(data_root, exist_ok=True)
        data_split_file_path = os.path.join(data_root, 'objaverse_food.json')

        if not os.path.exists(data_split_file_path):
            print(f"Split file not found, creating new split at {data_split_file_path}")
            data_split_file = self._split_data(data_split_file_path)
        else:
            print(f"Loading existing split from {data_split_file_path}")
            with open(data_split_file_path, 'r') as f:
                data_split_file = json.load(f)

        print('Food Objaverse dataset instances: train {}, test {}'.format(
            len(data_split_file['train']), len(data_split_file['test'])))
        self.data_split.update(data_split_file)

    
    def _load_frame(self, seq_path, img_name):
        file_path = os.path.join(seq_path, img_name + '.png')
        img_pil = Image.open(file_path)
        img_np = np.asarray(img_pil)
        try:
            mask = Image.fromarray((img_np[:,:,3] > 0).astype(float))
        except:
            mask = Image.fromarray(np.logical_and(img_np[:,:,0]==0,
                                                  img_np[:,:,1]==0,
                                                  img_np[:,:,2]==0).astype(float))

        if self.config.dataset.white_bkg:
            # white background
            mask_255 = mask.point(lambda p: p * 255)
            white_background = Image.new('RGB', img_pil.size, (255, 255, 255))
            rgb = Image.composite(img_pil, white_background, mask_255.convert('L'))
        else:
            # black background
            rgb = Image.fromarray(img_np[:,:,:3])

        rgb = rgb.resize((self.img_size, self.img_size), Image.LANCZOS)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        rgb = np.asarray(rgb).transpose((2,0,1)) / 255.0                            # [3,H,W], in range [0,1]
        mask = np.asarray(mask)[:,:,np.newaxis].transpose((2,0,1))                  # [1,H,W], in range [0,1]

        if not self.config.dataset.white_bkg:
            rgb *= mask
        
        if self.config.train.normalize_img:
            normalization = transforms.Compose([
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
            rgb = torch.from_numpy(rgb)
            rgb = normalization(rgb).numpy()

        return rgb, mask
    def _load_frameNEW(self, seq_path, img_name):
        """Load image and mask from directory"""
        # Try direct path first
        file_path = os.path.join(seq_path, f"{img_name}.png")
        if not os.path.isfile(file_path):
            # Try subdirectory path
            file_path = os.path.join(seq_path, img_name, f"{img_name}.png")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"Image file not found: {file_path}")

        try:
            img_pil = Image.open(file_path)
            img_np = np.array(img_pil)
            
            # Handle RGBA images
            if img_np.shape[-1] == 4:
                mask = Image.fromarray((img_np[:,:,3] > 0).astype(float))
                rgb = Image.fromarray(img_np[:,:,:3])
            else:
                # Handle RGB images
                mask = Image.fromarray(np.logical_and(img_np[:,:,0]==0,
                                                    img_np[:,:,1]==0,
                                                    img_np[:,:,2]==0).astype(float))
                rgb = img_pil

            if self.config.dataset.white_bkg:
                mask_255 = mask.point(lambda p: p * 255)
                white_background = Image.new('RGB', img_pil.size, (255, 255, 255))
                rgb = Image.composite(rgb, white_background, mask_255.convert('L'))
            
            rgb = rgb.resize((self.img_size, self.img_size), Image.LANCZOS)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            rgb = np.array(rgb).transpose((2,0,1)) / 255.0
            mask = np.array(mask)[:,:,np.newaxis].transpose((2,0,1))

            if not self.config.dataset.white_bkg:
                rgb *= mask
            
            return rgb, mask
        except Exception as e:
            print(f"Error loading image {file_path}: {str(e)}")
            raise


    def _load_pose(self, seq_path, img_name):
        """Load camera pose and intrinsics from JSON file"""
        # Try direct path first
        file_path = os.path.join(seq_path, f"{img_name}.json")
        if not os.path.isfile(file_path):
            # Try subdirectory path
            file_path = os.path.join(seq_path, img_name, f"{img_name}.json")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"Pose file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                meta = json.load(f)
                
            camera_matrix = np.eye(4)
            camera_matrix[:3, 0] = np.array(meta['x'])
            camera_matrix[:3, 1] = np.array(meta['y'])
            camera_matrix[:3, 2] = np.array(meta['z'])
            camera_matrix[:3, 3] = np.array(meta['origin'])
            camera_matrix = torch.tensor(camera_matrix).float()
            
            fx = 0.5 / math.tan(0.5 * meta['x_fov'])
            fy = 0.5 / math.tan(0.5 * meta['y_fov'])
            K = torch.tensor([[fx * self.render_size, 0., 0.5 * self.render_size],
                            [0., fy * self.render_size, 0.5 * self.render_size],
                            [0., 0., 1.]]).float()
                            
            return camera_matrix, K
        except Exception as e:
            print(f"Error loading pose {file_path}: {str(e)}")
            raise

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        try:
            relative_seq_path = self.seq_names[idx]  # This is the relative path (e.g., "131/665232/00038")
            full_seq_path = os.path.join(self.root, relative_seq_path)  # Full path to sequence directory
            base_name = os.path.basename(relative_seq_path)  # e.g., "00038"
            # Get all PNG files in this directory
            png_files = []
            
            # Scan for all valid image-json pairs
            for i in range(40):  # Check all 40 possible views
                view_name = f"{base_name}"  # Use the base name directly
                png_path = os.path.join(full_seq_path, f"{view_name}.png")
                # print("png_path", png_path)
                json_path = os.path.join(full_seq_path, f"{view_name}.json")
                if os.path.isfile(png_path) and os.path.isfile(json_path):
                    png_files.append(view_name)
                else:
                    print(f"Missing PNG or JSON file for {view_name} in {full_seq_path}")

            if len(png_files) < self.num_frame:
                print(f"Not enough valid files in {full_seq_path}. Found {len(png_files)}, need {self.num_frame}")
                # Try a different sample
                alternative_idx = (idx + 1) % len(self)
                return self[alternative_idx]

            # Sample images safely
            if self.split == 'train':
                # Ensure safe sampling
                num_first_sample = min(len(png_files), 25)
                if num_first_sample <= 0:
                    raise ValueError(f"No valid files to sample from in {full_seq_path}")
                
                # Get first sample
                chosen_files = random.sample(png_files[:num_first_sample], k=1)
                
                # Get remaining samples
                remaining_files = [f for f in png_files if f not in chosen_files]
                if len(remaining_files) < (self.num_frame - 1):
                    # If not enough remaining files, use sampling with replacement
                    additional_files = random.choices(remaining_files or png_files, k=(self.num_frame - 1))
                else:
                    # If enough remaining files, sample without replacement
                    additional_files = random.sample(remaining_files, k=(self.num_frame - 1))
                
                chosen_files.extend(additional_files)
            else:
                # For validation/testing, just take sequential frames
                chosen_files = png_files[:self.num_frame]
                # If still not enough, repeat the last one
                while len(chosen_files) < self.num_frame:
                    chosen_files.append(chosen_files[-1])

            # Ensure we have exactly the right number of files
            assert len(chosen_files) == self.num_frame, \
                f"Wrong number of files selected: {len(chosen_files)} != {self.num_frame}"

            # Load images and poses
            imgs, masks = [], []
            Ks, cam_poses_cv2 = [], []
            
            for file_name in chosen_files:
                try:
                    img, mask = self._load_frame(full_seq_path, file_name)
                    pose, K = self._load_pose(full_seq_path, file_name)
                    
                    imgs.append(torch.tensor(img))
                    masks.append(torch.tensor(mask))
                    Ks.append(K)
                    cam_poses_cv2.append(pose)
                except Exception as e:
                    print(f"Error loading file {file_name} from {full_seq_path}: {str(e)}")
                    raise

            # Stack tensors
            try:
                imgs = torch.stack(imgs)           # [n,c,h,w]
                masks = torch.stack(masks)         # [n,1,h,w]
                Ks = torch.stack(Ks)              # [n,3,3]
                cam_poses_cv2 = torch.stack(cam_poses_cv2)  # [n,4,4]
            except Exception as e:
                print(f"Error stacking tensors for {full_seq_path}: {str(e)}")
                raise

            # Normalize camera poses
            cam_poses_normalized_cv2 = self._canonicalize_cam_poses(cam_poses_cv2)
            cam_poses_normalized_opengl = cam_poses_normalized_cv2 @ cv2_to_opengl.unsqueeze(0)

            # Get input image
            input, input_mask = imgs[0].float(), masks[0].float()
            input = input * input_mask + (1 - input_mask) * 0.5

            # Get rays
            rays_o, rays_d = get_rays_from_pose(cam_poses_normalized_opengl, focal=Ks[:,0,0], size=self.render_size)

            return {
                'input_image': input.unsqueeze(0),
                'rays_o': rays_o.float(),
                'rays_d': rays_d.float(),
                'render_images': imgs.float(),
                'render_masks': masks.float(),
                'seq_name': relative_seq_path
            }

        except Exception as e:
            print(f"Error processing sequence {relative_seq_path if 'relative_seq_path' in locals() else 'unknown'}: {str(e)}")
            # Try a different sample
            alternative_idx = (idx + 1) % len(self)
            if alternative_idx != idx:
                return self[alternative_idx]
            else:
                # If we've tried all indices, raise the error
                raise RuntimeError("No valid samples found in dataset")

    def _canonicalize_cam_poses(self, cam_poses):
        '''Normalize camera poses relative to the first pose'''
        cam_poses_rel = get_relative_pose(cam_poses[0], cam_poses)
        translation = cam_poses[0][:3,3]
        scale = torch.sqrt((translation ** 2).sum())
        canonical_pose = self._build_canonical_pose(scale)
        cam_poses_rotated = canonical_pose.unsqueeze(0) @ cam_poses_rel
        cam_poses_scaled = self._normalize_scale(cam_poses_rotated, scale)
        return cam_poses_scaled

    def _build_canonical_pose(self, scale):
        canonical_pose = torch.tensor([[0.0, 0.0, 1.0, scale],
                                     [1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]]) @ opengl_to_cv2
        return canonical_pose

    def _normalize_scale(self, camera_poses, distance):
        if self.normalization == 'constant_distance':
            distance_target = self.canonical_distance
            distance_factor = distance_target / distance
            cam_translation_normalized = camera_poses[:,:3,3] * distance_factor
            cam_poses_normalized = camera_poses.clone()
            cam_poses_normalized[:,:3,3] = cam_translation_normalized
            return cam_poses_normalized
        elif self.normalization == 'constant_scale':
            scale_target = self.canonical_scale
            scale_factor = scale_target / SCALE_RAW
            cam_translation_normalized = camera_poses[:,:3,3] * scale_factor
            cam_poses_normalized = camera_poses.clone()
            cam_poses_normalized[:,:3,3] = cam_translation_normalized
            return cam_poses_normalized
        else:
            raise NotImplementedError