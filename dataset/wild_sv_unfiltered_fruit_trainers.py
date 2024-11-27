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
from torchvision.transforms import functional as func_transforms
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageFilter
from dataset.constant import *
from utils.geo_utils import get_relative_pose
from utils.process_utils import process_images, get_cameras, get_rays_from_pose, get_cameras_curriculum, ImageMaskAug
from utils import data_utils
from scipy.optimize import minimize
from dataset.mvimagenet_sv import paste_crop
ImageFile.LOAD_TRUNCATED_IMAGES = True

def convert_windows_to_wsl_path(windows_path):
    """Convert Windows path to WSL path"""
    # Remove drive letter and convert backslashes to forward slashes
    path = windows_path.replace('\\', '/')
    drive_letter = path[0].lower()
    wsl_path = f"/mnt/{drive_letter}/{path[3:]}"
    return wsl_path

class WILD_SV_UNFILTERED_FRUIT_TRAINERS(Dataset):
    def __init__(self, config, split='train', length=1, multiple_data=False, data_name='',
                 root=r"F:\Zero123 Objaverse\food drawings"):
        self.config = config
        self.split = split
        self.root = convert_windows_to_wsl_path(root)  # Keep Windows path
        self.multiple_data = multiple_data
        self.data_name = data_name
        assert split in ['train', 'val', 'test']
        self.length = length

        # Dataset configuration
        self.use_consistency = config.train.use_consistency
        self.num_frame_consistency = config.train.num_frame_consistency
        self.consistency_curriculum = config.dataset.sv_curriculum
        self.rerender_consistency_input = config.train.rerender_consistency_input

        # Image and rendering settings
        self.img_size = config.dataset.img_size
        self.num_frame = 1
        self.render_size = config.model.render_resolution if split == 'train' else config.test.eval_resolution
        self.camera_distance = SCALE_OBJAVERSE if config.dataset.mv_data_name == 'objaverse' else SCALE_OBJAVERSE_BOTH
        self.render_views = config.dataset.sv_render_views
        self.render_views_sample = config.dataset.sv_render_views_sample

        # Augmentation
        self.transform = ImageMaskAug() if config.dataset.sv_use_aug else None

        # Background color (white or black)
        self.white_background = config.dataset.get('white_bkg', True)

        # Load dataset
        print(f"Initializing dataset from {self.root}")
        self.data_split = {}
        self._load_dataset()
        
        # Set up sequence names based on split
        self.seq_names = []
        if self.split == 'train':
            self.seq_names += self.data_split['train']
        else:
            self.seq_names += self.data_split['test']
            if self.split == 'val':
                self.seq_names = self.seq_names[::config.eval_vis_freq]
                
        print(f"Loaded {len(self.seq_names)} images for {split} split")

    def _load_dataset(self):
        data_root = './dataset/split_info'
        os.makedirs(data_root, exist_ok=True)
        data_split_file_path = os.path.join(data_root, 'food_singleview_unfiltered.json')

        if not os.path.exists(data_split_file_path):
            self._split_data(data_split_file_path)

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        print('Singleview food unfiltered (SV) dataset instances: train {}, test {}'.format(len(data_split_file['train']),
                                                                         len(data_split_file['test'])))
        self.data_split.update(data_split_file)


    # def _split_data(self, data_split_file_path):
    #     '''
    #     Select data that have both images and masks
    #     '''
    #     all_instances_valid = []
    #
    #     all_data = os.listdir(self.root)
    #     for dataset in all_data:
    #         all_instances = os.listdir(os.path.join(self.root, dataset))
    #         all_instances_valid += [os.path.join(dataset, it) for it in all_instances]
    #
    #     random.shuffle(all_instances_valid)
    #     all_info = {'train': all_instances_valid[:-1000], 'test': all_instances_valid[-1000:]}
    #
    #     with open(data_split_file_path, 'w') as f:
    #         json.dump(all_info, f, indent=4)

    def _split_data(self, data_split_file_path):
        '''
        Select images from flat directory structure
        '''
        all_instances_valid = []

        # Get all image files from the root directory
        valid_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        
        print(f"Scanning directory: {self.root}")
        for filename in os.listdir(self.root):
            if os.path.splitext(filename)[1] in valid_extensions:
                try:
                    # Verify the image can be opened
                    img_path = os.path.join(self.root, filename)
                    with Image.open(img_path) as img:
                        # Convert to RGB to ensure it's a valid image
                        img.convert('RGB')
                        all_instances_valid.append(filename)
                except Exception as e:
                    print(f"Skipping {filename} due to error: {str(e)}")

        print(f"Found {len(all_instances_valid)} valid images")

        # Split into train and test
        random.shuffle(all_instances_valid)
        num_test = min(1000, int(len(all_instances_valid) * 0.1))  # 10% or 1000, whichever is smaller
        
        all_info = {
            'train': all_instances_valid[:-num_test], 
            'test': all_instances_valid[-num_test:]
        }

        print(f"Split into {len(all_info['train'])} training and {len(all_info['test'])} testing images")

        # Save the split
        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f, indent=4)


    def __len__(self):
        return len(self.seq_names)
    

    def __getitem__(self, idx):
        img_name = self.seq_names[idx]
        img_path = os.path.join(self.root, img_name)

        # load image and mask
        img, mask = self._load_image(img_path)
        img, mask = self._process_image(img, mask)      # squared, centered

        if self.transform and self.split == 'train':
            img, mask = self.transform(img, mask)
            bkgd_color = 1.0 if self.config.dataset.white_bkg else 0.0
            img = img * mask + (1 - mask) * bkgd_color

        if self.config.train.normalize_img:
            imgs = self._normalize_img(imgs)

        fov = 0.691150367
        fx, fy = 0.5 / math.tan(0.5 * fov), 0.5 / math.tan(0.5 * fov)
        Ks = torch.tensor([[fx * self.render_size, 0., 0.5 * self.render_size],
                          [0., fy * self.render_size, 0.5 * self.render_size],
                          [0., 0., 1.]]).float().unsqueeze(0).repeat(self.num_frame,1,1)
        c2w = get_cameras(self.render_views, 0, self.camera_distance, sampling=self.render_views_sample)
        rays_o, rays_d = get_rays_from_pose(c2w, focal=Ks[:,0,0], size=self.render_size)

        input, input_mask = img.float(), mask.float()   # [c,h,w], [1,h,w]
        input = input * input_mask + (1 - input_mask) * 0.5

        sample = {
                'input_image': input.unsqueeze(0),              # [1,c,h,w]
                'rays_o': rays_o.float(),                       # [n,h,w,3], only used in training
                'rays_d': rays_d.float(),                       # [n,h,w,3], only used in training
                'render_images': img.unsqueeze(0).float(),      # [n=1,c,h,w]
                'render_masks': mask.unsqueeze(0).float(),      # [n=1,1,h,w]
                'Ks': Ks,                                       # [n,3,3]
            }
        
        if self.use_consistency:
            c2w_consistency = self._get_cameras_consistency(c2w)
            rays_o_consistency, rays_d_consistency = get_rays_from_pose(c2w_consistency, focal=Ks[:,0,0], size=self.render_size)
            sample['rays_o_consistency'] = rays_o_consistency
            sample['rays_d_consistency'] = rays_d_consistency

        if self.rerender_consistency_input and self.use_consistency:
            rays_o_hres, rays_d_hres = get_rays_from_pose(c2w, 
                                                          focal=Ks[:,0,0] / self.render_size * self.img_size, 
                                                          size=self.img_size)
            sample['rays_o_hres'] = rays_o_hres
            sample['rays_d_hres'] = rays_d_hres

        return sample


    def _load_imageOLD(self, img_path):
        img_pil = Image.open(img_path)
        assert img_pil.mode == 'RGBA'

        r, g, b, a = img_pil.split()
        r_array = np.array(r)
        g_array = np.array(g)
        b_array = np.array(b)
        a_array = np.array(a)

        img = np.stack([r_array, g_array, b_array], axis=-1)
        img = np.asarray(img).transpose((2,0,1)) / 255.0                            # [3,h,w]
        mask = np.asarray(a_array).squeeze()[:,:,np.newaxis].transpose((2,0,1))
        mask = (mask > 225)                                                         # [1,h,w]
        return torch.tensor(img), torch.tensor(mask).float()

    def _load_image(self, img_path):
        """
        Load image and create mask, supporting both RGBA and RGB formats.
        For RGB images, we can either use the whole image or detect the object automatically.
        """
        img_pil = Image.open(img_path)
        
        if img_pil.mode == 'RGBA':
            # Handle RGBA images using alpha channel as mask
            r, g, b, a = img_pil.split()
            img = np.array(img_pil.convert('RGB'))
            mask = np.array(a)
            mask = (mask > 225).astype(float)  # Binary mask from alpha channel
        else:
            # Handle RGB images
            img = np.array(img_pil.convert('RGB'))
            
            # Option 1: Use the whole image (simple mask of all ones)
            mask = np.ones(img.shape[:2], dtype=float)
            
            # Option 2: Try to detect the object using simple thresholding
            # Uncomment this section if you want automatic object detection
            """
            # Convert to grayscale and threshold to find potential foreground
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Use Otsu's thresholding
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = mask.astype(float) / 255.0
            
            # Optional: Clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            """

        # Convert to tensor format
        img = np.asarray(img).transpose((2,0,1)) / 255.0  # [3,h,w]
        mask = mask[np.newaxis, :, :]  # [1,h,w]
        
        return torch.tensor(img), torch.tensor(mask).float()
    

    def _process_image(self, img, mask, expand_ratio=1.7):
        # img in shape [3,h,w]
        # mask in shape [1,h,w]
        c,h,w = img.shape
        assert img.shape[-2:] == mask.shape[-2:]
        bkgd_color = 1.0 if self.config.dataset.white_bkg else 0.0
        if self.split == 'train':
            expand_ratio = random.random() * 0.5 + 1.45

        larger_dim = max(h, w)
        new_size = int(larger_dim * expand_ratio)
        pad_l = (new_size - w) // 2
        pad_r = new_size - w - pad_l
        pad_t = (new_size - h) // 2
        pad_b = new_size - h - pad_t

        img = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=bkgd_color)
        mask = F.pad(mask, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0.0)

        # squared image
        processed_img = F.interpolate(img.unsqueeze(0), 
                                        size=(self.img_size, self.img_size),
                                        mode='bilinear', align_corners=False)[0]
        processed_mask = F.interpolate(mask.unsqueeze(0), 
                                        size=(self.img_size, self.img_size),
                                        mode='nearest')[0]
        return processed_img, processed_mask
    

    def _get_cameras_consistency(self, c2w):
        # c2w in shape [n,4,4]
        # c2w[0] is the input view pose (identity rotation)
        c2w_input, c2w_last = c2w[:1], c2w[-self.num_frame_consistency:]
        c2w_invert = c2w_input @ torch.inverse(c2w_last) @ c2w_input
        return c2w_invert
    

    def _normalize_img(self, imgs):
        normalization = transforms.Compose([
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
        return normalization(imgs)

