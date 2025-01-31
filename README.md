<div align="center">

# This Repo is based on Real3D: Scaling Up Large Reconstruction Models with Real-World Images

**Abstract from Real3D**: As single-view 3D reconstruction is ill-posed due to the ambiguity from 2D to 3D, the reconstruction models have to learn generic shape and texture priors from large data. The default strategy for training single-view Large Reconstruction Models (LRMs) follows the fully supervised route, using synthetic 3D assets or multi-view captures. Although these resources simplify the training procedure, they are hard to scale up beyond the existing datasets and they are not necessarily representative of the real distribution of object shapes. To address these limitations, in this paper, we introduce Real3D, the first LRM system that can be trained using single-view real-world images. Real3D introduces a novel self-training framework that can benefit from both the existing 3D/multi-view synthetic data and diverse single-view real images. We propose two unsupervised losses that allow us to supervise LRMs at the pixel- and semantic-level, even for training examples without ground-truth 3D or novel views. To further improve performance and scale up the image data, we develop an automatic data curation approach to collect high-quality examples from in-the-wild images. Our experiments show that Real3D consistently outperforms prior work in four diverse evaluation settings that include real and synthetic data, as well as both in-domain and out-of-domain shapes.

</div>
# Demo


https://github.com/user-attachments/assets/f91a00a1-9ccb-40b0-83c0-ab8f47c2815f


# Installation Guide

## Environment Setup

### (Local Option)

First, setup the environment for the Mesh Generation.

```bash
# If you are running on windows you need to set up WSL and Conda (INSIDE WSL) first.

#After that do this:
conda create --name real3d python=3.9
conda activate real3d

# Install pytorch, we use:
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://github.com/hwjiang1510/Real3D.git
cd real3d

pip install -r requirements.txt

# I had to run this to update some of the versions 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# I had to reinstall torchmcubes
pip uninstall torchmcubes
pip install git+https://github.com/tatsy/torchmcubes.git

# Install other dependencies < probably don't need to do this
pip install rembg pillow numpy onnxruntime
```

Second, setup the environment for the Remesh and Texture Generation.

```bash
# If you are running on windows you need to set up WSL and Conda (INSIDE WSL) first.

conda create -n texture-gen python=3.10 -y
conda activate texture-gen

conda install pytorch==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Move on to the next step.

**This is a fix for an issue when installing torchmcubes where it doesn't detect GPU:**

```bash
conda install -c nvidia cuda-toolkit

# It wasn't detecting my gpu so had to change the cmake file to "86" but I think this only works for some gpus
git clone https://github.com/tatsy/torchmcubes.git
cd torchmcubes

# in CMakeLists.txt file, change set(CMAKE_CUDA_ARCHITECTURES "native") to set(CMAKE_CUDA_ARCHITECTURES "86")
pip install .

# Also had to rerun install requirements.txt after installing torchmcubes
```

### (Google Colab Option)

Visit this Google Colab Notebook: [Here](https://colab.research.google.com/drive/1sFt2UtVDTU171ZtouI5CUZ4gyRcVkvuV?usp=sharing) (Keep in mind that you only have a few hours of compute in the free version.)

Then run each cell in order to set up the environment.

There is a cell which downloads our model by default, it can be changed to other models.

## After Setting up The Environment

### (Local Checkpoint Setup)

Download the model weight:

- [model weight from TripoSR](https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt?download=true)
- [model weight from TripoSR, fine tuned on synthetic data to fix some issues](https://huggingface.co/hwjiang/Real3D/resolve/main/model_both.ckpt?download=true)
- [model weights from Real3d trained on Multi-View and Single-View Images](https://huggingface.co/hwjiang/Real3D/resolve/main/model_both_trained_v1.ckpt?download=true)
- [Our Model Weight Fine Tuned on Fantasy Sword Images](https://huggingface.co/hwjiang/Real3D/resolve/main/model_both_trained_v1.ckpt?download=true) (not uploaded yet)
- [Our Model Weight Fine Tuned on Food Items](https://huggingface.co/Victor2266/RealFruit3D/resolve/main/new_fruit_model_30k.ckpt?download=true)

and put it into `./checkpoint/<model_name>.ckpt`.

### (Google Collab Option)

To change the model it uses, you have to edit this cell in the note book to change the url to the model you want:
![image](https://github.com/user-attachments/assets/ff358c76-2b9e-4481-8285-0b688341e6bf)

You also have to change the run script to the path of the model you want to use:
![image](https://github.com/user-attachments/assets/1d887d84-ebf2-4db7-8d6a-bb0937cc8ac7)

Otherwise, just run every cell in order, it will take an input image, generate a mesh for it, then clean and remesh that output and generate a new texture for it using stable diffusion. At the end, it will output rough renderings of a few angles of the final model.

## Run The Mesh Generation Demo (Locally)

Use `./run.sh` and modify your image path and foreground segmentation config accordingly. Tune the chunk size to fit your GPU.

**To Fix “permission denied” error:**
Use `chmod +x run.sh` to add the “x” permission.

A good input image follows 3 criteria: It should have a clear subject that isn't smaller than 100 pixels, the subject should not be occluded by other objects or cutoff from the image, and extremely asynetrically objects have issues.

### **If you want to modify the parameters for run.py:**

[these are the parameters](https://github.com/Victor2266/ComputerVision-Concept-Art-to-Model-Real3D/blob/4e2323a7d527d56bd7fa0e62f0f24a59a8137bca/parameters.md)

## Run The Remesh and Texture Generation Demo (Locally)

```bash
cd triposr-texture-gen/

# first prepare mesh for texture generation
python prepare_mesh.py --input mesh.glb

# run texture generation
PROMPT="A single perfect red apple, flawless glossy skin, studio product photography, extreme close-up, photorealistic, 8k resolution, soft diffused lighting, pristine condition"

python text2texture.py prepared_mesh.glb "$PROMPT"
```

## How To Do Training

### Data preparation

This repo uses Gobjaverse and our collected real images.

### Step 0: (Optional) Fine-tune TripoSR

As TripoSR predicts 3D shapes with randomized scales, we first need to fine-tune it on Objaverse. We provide the [fine-tuned model weight](https://huggingface.co/hwjiang/Real3D/resolve/main/model_both.ckpt?download=true), so you can put it to `./checkpoint/model_both.ckpt` and skip this stage.

### Step 1: Self-training on real images

Use `./train_sv.sh`.
Or `train_sv_for_fruit.sh`
You have to change the YAML file in `./config` to set your hyper parameters and change the paths in the dataloader scripts in `./dataset` to point to your MV and SV datasets.

## Evaluation

Use `./eval.sh` and modify the script and config accordingly.
For example, to evaluate on CO3D with ground-truth multiviews, use `eval_mv.py` and `./config/eval/eval_mv_co3d.yaml`. To evaluate on single-view images, use `eval_sv.py` and `./config/eval/eval_sv.yaml`.

## Acknowledgements

This repo is developed based on [TripoSR](https://github.com/VAST-AI-Research/TripoSR/), [Real3D](https://github.com/hwjiang1510/Real3D/tree/main?tab=readme-ov-file), and [triposr-texture-gen](https://github.com/ejones/triposr-texture-gen/blob/main/README.md)
