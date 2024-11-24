import math
import pathlib

import bpy
from blendify import scene
from mathutils import Vector  # type: ignore

DIR_PATH = pathlib.Path(__file__).parent
INPUT_PATH = DIR_PATH / "multiview_input"
OUTPUT_PATH = DIR_PATH / "multiview_dataset"


def scale_to_camera(obj, cam):
    # find object bounding box size
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_center = sum(bbox_corners, Vector()) / 8.0
    bbox_size = max((corner - bbox_center).length for corner in bbox_corners)

    # get size that fits in fov of camera
    distance_x = bbox_size / math.tan(cam.data.angle_x / 2)
    distance_y = bbox_size / math.tan(cam.data.angle_y / 2)
    max_distance = max(distance_x, distance_y)

    # change scale to fit camera
    cam_distance = (bbox_center - cam.location).length
    scale_factor = cam_distance / max_distance
    obj.scale *= scale_factor


def add_cam(translation):
    return scene.set_perspective_camera(
        resolution=(512, 512),
        rotation_mode="look_at",
        rotation=(0, 0, 0),
        fov_x=0.7,
        translation=translation,
    )


scene.lights.add_point(strength=1000, translation=(4, -2, 4))

cam1 = add_cam((5, 5, 8))


bpy.ops.import_scene.fbx(filepath=str(INPUT_PATH / "anvil.fbx"))
obj = bpy.context.selected_objects[0]
scale_to_camera(obj, cam1.blender_camera)


OUTPUT_PATH.mkdir(exist_ok=True)

scene.render(filepath=str(OUTPUT_PATH / "view1.png"))

add_cam((5, -5, 8))
scene.render(filepath=str(OUTPUT_PATH / "view2.png"))

add_cam((-5, 5, 8))
scene.render(filepath=str(OUTPUT_PATH / "view3.png"))
