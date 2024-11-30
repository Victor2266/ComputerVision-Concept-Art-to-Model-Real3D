import json
import math
import pathlib

import bpy
import numpy
from blendify import scene
from blendify.cameras import PerspectiveCamera
from mathutils import Vector

DIR_PATH = pathlib.Path(__file__).parent
DATASET_INPUT_PATH = DIR_PATH / "multiview_input"
DATASET_OUTPUT_PATH = DIR_PATH / "multiview_dataset"


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

    # center object at origin
    vertices = numpy.empty((len(obj.data.vertices), 3), dtype=numpy.float32)
    obj.data.vertices.foreach_get("co", vertices.ravel())
    vertices -= bbox_center
    obj.data.vertices.foreach_set("co", vertices.ravel())
    obj.data.update()

    # scale object to fit range
    scale_factor = 1 / (2 * bbox_size)
    obj.scale *= scale_factor


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
    add_hdri(str(DIR_PATH / "sunrise.hdr"))


def join_objs():
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    base = objs[0]

    if len(objs) > 1:
        with bpy.context.temp_override(
            active_object=base, selected_editable_objects=objs
        ):
            bpy.ops.object.join()

    return base


def import_obj(input_path):
    bpy.ops.import_scene.gltf(filepath=str(input_path))
    obj = join_objs()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    normalize_obj(obj)


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


def render_view(
    output_path: pathlib.Path,
    view_count,
    orbit_distance,
    elevation_angle,
    azimuth_angle,
):
    padded_view_count = f"{view_count:05}"
    view_dir = output_path / padded_view_count
    view_dir.mkdir(exist_ok=True)

    cam = add_cam(orbit_distance, elevation_angle, azimuth_angle)

    json_file = view_dir / f"{padded_view_count}.json"
    write_render_json(json_file, cam)

    scene.render(
        filepath=str(view_dir / f"{padded_view_count}.png"),
        samples=64,
        use_denoiser=True,
        # save_depth=True,
        # save_albedo=True,
    )

    print(f"Rendered view {padded_view_count}")
    return view_count + 1


def render_multiview(input_idx):
    output_path = DATASET_OUTPUT_PATH / f"{input_idx:05}"
    output_path.mkdir(exist_ok=True)

    view_count = 0

    # elevation range from 5° to 30°, rotation = {r × 15° | r ∈ [0, 23]}
    orbit_distance = 1.6450
    for view_idx in range(25):
        azimuth_angle = math.radians(view_idx * 15)
        elevation_angle = math.radians(5 + (25 * (view_idx / 24)))
        view_count = render_view(
            output_path, view_count, orbit_distance, elevation_angle, azimuth_angle
        )

    # elevation from -5° to 5°, rotation = {r × 30° | r ∈ [0, 11]}
    orbit_distance = 1.9547
    for view_idx in range(13):
        azimuth_angle = math.radians(view_idx * 30)
        elevation_angle = math.radians(-5 + (10 * (view_idx / 12)))
        view_count = render_view(
            output_path, view_count, orbit_distance, elevation_angle, azimuth_angle
        )

    # top and bottom view
    orbit_distance = 1.6450
    view_count = render_view(
        output_path, view_count, orbit_distance, math.radians(90), 0
    )
    view_count = render_view(
        output_path, view_count, orbit_distance, math.radians(-90), 0
    )


def add_hdri(hdri_path):
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(hdri_path)
    background = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")

    links.new(env_tex.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


if __name__ == "__main__":
    DATASET_OUTPUT_PATH.mkdir(exist_ok=True)

    valid_extensions = [".glb"]
    for input_idx, input_path in enumerate(DATASET_INPUT_PATH.iterdir()):
        if input_path.suffix.lower() not in valid_extensions:
            print(f"Skipped {input_path.name}")
            continue

        setup_scene()
        import_obj(input_path)
        render_multiview(input_idx)
        scene.clear()
