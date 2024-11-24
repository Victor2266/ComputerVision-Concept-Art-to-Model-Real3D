import json
import math
import pathlib

import bpy
from blendify import scene
from blendify.cameras import PerspectiveCamera
from mathutils import Vector  # type: ignore

DIR_PATH = pathlib.Path(__file__).parent
DATASET_INPUT_PATH = DIR_PATH / "multiview_input"
DATASET_OUTPUT_PATH = DIR_PATH / "multiview_dataset"

INPUT_NAME = "anvil"
OUTPUT_PATH = DATASET_OUTPUT_PATH / INPUT_NAME

"""
Format:
https://aigc3d.github.io/gobjaverse/
https://github.com/hwjiang1510/Real3D/blob/main/dataset/objaverse.py

Objaverse (Gobjaverse) is rendered with 38 fixed camera poses:
    - 24 on orbit with distance 1.6450
    - 2 for top-down / bottom up with distance 1.6450
    - 12 on orbit with distance 1.9547
And object is normalized in range [-0.5,0.5]
"""


def normalize_obj(obj):
    # find object bounding box size
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_center = sum(bbox_corners, Vector()) / 8.0
    bbox_size = max((corner - bbox_center).length for corner in bbox_corners)

    # scale object to fit range
    scale_factor = 1 / (2 * bbox_size)
    obj.scale *= scale_factor

    # center object at origin
    obj.location -= bbox_center


def add_cam(orbit_distance, elevation_angle, azimuth_angle):
    x = orbit_distance * math.cos(elevation_angle) * math.cos(azimuth_angle)
    y = orbit_distance * math.cos(elevation_angle) * math.sin(azimuth_angle)
    z = orbit_distance * math.sin(elevation_angle)

    return scene.set_perspective_camera(
        resolution=(512, 512),
        rotation_mode="look_at",
        rotation=(0, 0, 0),
        fov_x=0.7,
        translation=(x, y, z),
    )


def setup_scene():
    scene.lights.add_point(strength=1000, translation=(4, -2, 4))
    bpy.ops.import_scene.fbx(filepath=str(DATASET_INPUT_PATH / f"{INPUT_NAME}.fbx"))
    obj = bpy.context.selected_objects[0]
    normalize_obj(obj)
    DATASET_OUTPUT_PATH.mkdir(exist_ok=True)
    OUTPUT_PATH.mkdir(exist_ok=True)


def write_render_json(json_path, cam: PerspectiveCamera):
    rotation = cam.blender_camera.matrix_world.to_3x3()

    render_data = {
        "origin": list(cam.translation),
        "x_fov": cam.fov_x,
        "y_fov": cam.fov_y,
        "x": list(rotation.col[0]),
        "y": list(rotation.col[1]),
        "z": list(rotation.col[2]),
    }

    with json_path.open("w") as file:
        json.dump(render_data, file, indent=4)


def render_view(view_name, orbit_distance, elevation_angle, azimuth_angle):
    cam = add_cam(orbit_distance, elevation_angle, azimuth_angle)
    render_name = f"{INPUT_NAME}_{view_name}"
    new_dir = OUTPUT_PATH / render_name
    new_dir.mkdir()

    json_file = new_dir / f"{render_name}.json"
    write_render_json(json_file, cam)

    scene.render(
        filepath=str(new_dir / f"{render_name}.png"),
        samples=64,
        use_denoiser=True,
        save_depth=True,
        save_albedo=True,
    )


def render():
    # elevation range from 5° to 30°, rotation = {r × 15° | r ∈ [0, 23]}
    orbit_distance = 1.6450
    for view_idx in range(24):
        azimuth_angle = math.radians(view_idx * 15)
        elevation_angle = math.radians(5 + (25 * (view_idx / 23)))
        render_view(f"high_{view_idx}", orbit_distance, elevation_angle, azimuth_angle)

    # elevation from -5° to 5°, rotation = {r × 30° | r ∈ [0, 11]}
    orbit_distance = 1.9547
    for view_idx in range(12):
        azimuth_angle = math.radians(view_idx * 30)
        elevation_angle = math.radians(-5 + (10 * (view_idx / 11)))
        render_view(f"low_{view_idx}", orbit_distance, elevation_angle, azimuth_angle)

    # top and bottom view
    orbit_distance = 1.6450
    render_view("top", orbit_distance, math.radians(90), 0)
    render_view("bottom", orbit_distance, math.radians(-90), 0)


if __name__ == "__main__":
    setup_scene()
    render()
