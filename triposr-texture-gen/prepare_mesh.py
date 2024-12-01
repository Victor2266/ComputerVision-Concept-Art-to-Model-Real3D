import argparse
import pathlib

import bpy
import numpy as np
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

    subdiv_mod = obj.modifiers.new(name="Subdivision", type="SUBSURF")
    subdiv_mod.levels = subdiv_levels
    subdiv_mod.render_levels = subdiv_levels

    tri_mod = obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
    tri_mod.quad_method = "BEAUTY"


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


def smooth_shade(filepath):
    def load_obj(filepath):
        vertices = []
        faces = []
        with open(filepath, "r") as file:
            for line in file:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v":
                    vertices.append(list(map(float, parts[1:4])))
                elif parts[0] == "f":
                    face = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    faces.append(face)
        return np.array(vertices), np.array(faces)

    def calculate_smooth_normals(vertices, faces):
        normals = np.zeros(vertices.shape, dtype=np.float32)
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            normal /= np.linalg.norm(normal)
            for vert_idx in face:
                normals[vert_idx] += normal
        normals = (normals.T / np.linalg.norm(normals, axis=1)).T
        return normals

    def save_obj(filepath, vertices, faces, normals):
        with open(filepath, "w") as file:
            for vert_idx, vert in enumerate(vertices):
                file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
                file.write(
                    f"vn {normals[vert_idx][0]} "
                    f"{normals[vert_idx][1]} "
                    f"{normals[vert_idx][2]}\n"
                )
            for face in faces:
                face_indices = [f"{idx + 1}//{idx + 1}" for idx in face]
                file.write(f"f {' '.join(face_indices)}\n")

    vertices, faces = load_obj(filepath)
    normals = calculate_smooth_normals(vertices, faces)
    save_obj(filepath, vertices, faces, normals)

    with open(filepath, "r") as obj_file:
        lines = obj_file.readlines()

    with open(filepath, "w") as output_file:
        smooth_added = False
        for line in lines:
            if line.startswith("s 0"):
                continue
            if line.startswith("f ") and not smooth_added:
                output_file.write("s 1\n")
                smooth_added = True
            output_file.write(line)


def export_mesh(filepath):
    bpy.ops.wm.obj_export(
        filepath=str(filepath),
        export_uv=False,
        export_materials=False,
    )
    print(f"Exported mesh to {filepath}")

    # manually add smooth shading since it wasn't working with blender
    smooth_shade(filepath)


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
