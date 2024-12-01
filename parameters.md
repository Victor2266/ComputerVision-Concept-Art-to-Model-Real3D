#### **Device Configuration**
- `--device`:
  - **Type**: `str`
  - **Default**: `"cuda:0"`
  - **Description**: Specifies the computation device to use. Defaults to the first CUDA-compatible GPU (`cuda:0`). If no GPU is available, falls back to CPU.
  - **Example**: `--device cpu`

#### **Model Loading**
- `--pretrained-model-name-or-path`:
  - **Type**: `str`
  - **Default**: `"./checkpoint"`
  - **Description**: Path to the pretrained model, either as a local directory or a Hugging Face model ID.
  - **Example**: `--pretrained-model-name-or-path stabilityai/TripoSR`

#### **Chunk Size**
- `--chunk-size`:
  - **Type**: `int`
  - **Default**: `8192`
  - **Description**: Evaluation chunk size for surface extraction and rendering. Smaller values use less GPU memory but increase computation time.
  - **Example**: `--chunk-size 4096`

#### **Marching Cubes Resolution**
- `--mc-resolution`:
  - **Type**: `int`
  - **Default**: `256`
  - **Description**: Resolution for the marching cubes algorithm, which is used to generate a mesh. Higher values produce more detailed meshes but require more memory and time.
  - **Example**: `--mc-resolution 512`

#### **Background Removal**
- `--no-remove-bg`:
  - **Type**: `flag`
  - **Default**: `False`
  - **Description**: If specified, skips automatic background removal. Assumes input images already have a uniform background and a well-sized foreground.
  - **Example**: `--no-remove-bg`

- `--foreground-ratio`:
  - **Type**: `float`
  - **Default**: `0.65`
  - **Description**: Ratio of the foreground size relative to the entire image size. Used to resize the foreground when background removal is enabled.
  - **Example**: `--foreground-ratio 0.75`

#### **Output Settings**
- `--output-dir`:
  - **Type**: `str`
  - **Default**: `"output_demo/"`
  - **Description**: Directory where results (e.g., images, videos, meshes) will be saved.
  - **Example**: `--output-dir results/`

- `--model-save-format`:
  - **Type**: `str`
  - **Default**: `"obj"`
  - **Choices**: `["obj", "glb"]`
  - **Description**: Format for saving the extracted mesh. Can be Wavefront OBJ (`obj`) or GLTF Binary (`glb`).
  - **Example**: `--model-save-format glb`

#### **Rendering**
- `--render`:
  - **Type**: `flag`
  - **Default**: `False`
  - **Description**: If specified, renders a 360-degree video of the model using NeRF-like techniques.
  - **Example**: `--render`

- `--render-num-views`:
  - **Type**: `int`
  - **Default**: `30`
  - **Description**: Number of views to render for the 360-degree video.
  - **Example**: `--render-num-views 72`
