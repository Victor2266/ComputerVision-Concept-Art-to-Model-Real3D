import argparse
import pathlib

import bpy
from blendify import scene

DIR_PATH = pathlib.Path(__file__).parent
DEFAULT_INPUT_PATH = DIR_PATH / "mesh.glb"
DEFAULT_EXPORT_PATH = DIR_PATH / "prepared_mesh.obj"


def get_all_meshes():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def prepare_mesh(obj, subdiv_levels):
    remesh_mod = obj.modifiers.new(name="Remesh", type="REMESH")
    remesh_mod.mode = "VOXEL"
    remesh_mod.voxel_size = 0.04

    obj.modifiers.new(name="Triangulate", type="TRIANGULATE")

    subdiv_mod = obj.modifiers.new(name="Subdivision", type="SUBSURF")
    subdiv_mod.levels = subdiv_levels
    subdiv_mod.render_levels = subdiv_levels

    obj.modifiers.new(name="Triangulate", type="TRIANGULATE")


def clean_mesh(subdiv_levels):
    # it comes in with small loose parts
    bpy.ops.mesh.separate(type="LOOSE")

    main_mesh = remove_loose_parts()

    prepare_mesh(main_mesh, subdiv_levels)


def remove_loose_parts():
    meshes = get_all_meshes()
    most_vertices = max(meshes, key=lambda obj: len(obj.data.vertices))
    main_mesh = None
    for obj in meshes:
        if obj != most_vertices:
            bpy.data.objects.remove(obj)
        else:
            main_mesh = obj
    return main_mesh


def import_mesh(filepath):
    bpy.ops.import_scene.gltf(filepath=str(filepath))
    objs = get_all_meshes()
    if len(objs) > 1:
        raise RuntimeError("Too many objs imported")
    print(f"Imported mesh from {filepath}")
    return objs[0]


def export_mesh(filepath):
    bpy.ops.wm.obj_export(
        filepath=str(filepath), export_uv=False, export_materials=False
    )
    print(f"Exported mesh to {filepath}")


def get_args():
    parser = argparse.ArgumentParser(description="Prepare mesh for texturing")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=f"Import path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_EXPORT_PATH),
        help=f"Export path (default: {DEFAULT_EXPORT_PATH})",
    )
    parser.add_argument(
        "--subdiv",
        type=int,
        default=2,
        help="Subdivision levels (default: 2)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_path = pathlib.Path(args.input)
    export_path = pathlib.Path(args.output)
    subdiv_levels = args.subdiv

    scene.clear()
    import_mesh(input_path)
    clean_mesh(subdiv_levels)
    export_mesh(export_path)
