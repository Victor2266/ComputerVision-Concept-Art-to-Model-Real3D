import pathlib

import bpy
from blendify import scene

DIR_PATH = pathlib.Path(__file__).parent
INPUT_PATH = DIR_PATH / "mesh.glb"
EXPORT_PATH = DIR_PATH / "prepared_mesh.obj"


def get_all_meshes():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def prepare_mesh(obj):
    remesh_mod = obj.modifiers.new(name="Remesh", type="REMESH")
    remesh_mod.mode = "VOXEL"
    remesh_mod.voxel_size = 0.04

    obj.modifiers.new(name="Triangulate", type="TRIANGULATE")

    subdiv_mod = obj.modifiers.new(name="Subdivision", type="SUBSURF")
    subdiv_mod.levels = 2
    subdiv_mod.render_levels = 2

    obj.modifiers.new(name="Triangulate", type="TRIANGULATE")


def clean_mesh():
    # it comes in with small loose parts
    bpy.ops.mesh.separate(type="LOOSE")

    main_mesh = remove_loose_parts()

    prepare_mesh(main_mesh)


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


if __name__ == "__main__":
    scene.clear()
    import_mesh(INPUT_PATH)
    clean_mesh()
    export_mesh(EXPORT_PATH)
