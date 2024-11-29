bl_info = {
    "name": "Custom Model Generator",
    "author": "Victor Do",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Custom Generator",
    "description": "Run custom model generation script from within Blender",
    "warning": "",
    "category": "3D View"
}

import bpy
import os
import subprocess
from bpy.props import StringProperty, IntProperty, BoolProperty, EnumProperty
from bpy.types import Panel, Operator

def convert_windows_to_wsl_path(windows_path):
    if windows_path.startswith('\\\\wsl$'):
        wsl_path = windows_path.replace('\\\\wsl$', '')
        wsl_path = wsl_path.replace('/Ubuntu-20.04', '')
    else:
        drive_letter = windows_path[0].lower()
        wsl_path = f'/mnt/{drive_letter}/{windows_path[3:]}'
    
    return wsl_path.replace('\\', '/')

class CMG_PT_main_panel(Panel):
    bl_label = "Custom Model Generator"
    bl_idname = "CMG_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Custom Generator'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        text = (
            "This is just a frontend which activates the scripts "
            "stored in the CustomReal3D directory of your WSL "
            "home path."
        )
        layout.label(text=text, icon='INFO')

        # Input image path
        layout.label(text="Input Image:")
        layout.prop(scene, "cmg_input_image", text="")
        
        # Output directory
        layout.label(text="Output Directory:")
        layout.prop(scene, "cmg_output_dir", text="")
        
        # Checkpoint path
        layout.label(text="Checkpoint Path or Name (Relative Path):")
        layout.prop(scene, "cmg_checkpoint_path", text="")
        
        # Additional settings
        layout.label(text="Settings:")
        layout.prop(scene, "cmg_render_views", text="Number of Views")
        layout.prop(scene, "cmg_mc_resolution", text="MC Resolution")
        layout.prop(scene, "cmg_model_format", text="Model Format")
        layout.prop(scene, "cmg_remove_bg", text="Remove Background")
        
        # Run button
        layout.operator("cmg.run", text="Generate Model", icon='PLAY')

class CMG_OT_run(Operator):
    bl_idname = "cmg.run"
    bl_label = "Generate Model"
    
    def execute(self, context):
        scene = context.scene
        
        # Convert Windows paths to WSL paths
        input_image = convert_windows_to_wsl_path(scene.cmg_input_image)
        output_dir = convert_windows_to_wsl_path(scene.cmg_output_dir)
        checkpoint_path = (scene.cmg_checkpoint_path)
        
        # Create the command string
        python_command = f"CUDA_VISIBLE_DEVICES=0 python run.py '{input_image}' --output-dir '{output_dir}' --render --render-num-views {scene.cmg_render_views} --model-save-format {scene.cmg_model_format} --mc-resolution {scene.cmg_mc_resolution} --pretrained-model-name-or-path '{checkpoint_path}'"
        
        if not scene.cmg_remove_bg:
            python_command += " --no-remove-bg"
        
        # Create a bash script that will be executed
        command = f"""
#!/bin/bash
source ~/.bashrc
source ~/.bash_profile
conda activate real3d
cd
cd CustomReal3D
{python_command}
"""
        
        try:
            # Run the command in WSL
            result = subprocess.run(
                ['wsl', 'bash', '-i', '-c', command],
                capture_output=True,
                text=True,
                check=True
            )
            self.report({'INFO'}, "Model generation completed successfully!")
            print("Command output:", result.stdout)
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"Model generation failed: {str(e)}\nOutput: {e.output}\nError: {e.stderr}")
            print("Command stderr:", e.stderr)
            return {'CANCELLED'}
        
        return {'FINISHED'}

# Property definitions
def register():
    bpy.types.Scene.cmg_input_image = StringProperty(
        name="Input Image",
        description="Path to input image",
        default="./assets/examples/apple.jpg",
        subtype='FILE_PATH'
    )
    bpy.types.Scene.cmg_output_dir = StringProperty(
        name="Output Directory",
        description="Output directory for results",
        default="output_demo",
        subtype='DIR_PATH'
    )
    bpy.types.Scene.cmg_checkpoint_path = StringProperty(
        name="Checkpoint Path",
        description="Path to the model checkpoint",
        default="./checkpoint/model_both_trained_v1.ckpt",
        subtype='FILE_PATH'
    )
    bpy.types.Scene.cmg_render_views = IntProperty(
        name="Render Views",
        description="Number of views to render",
        default=72,
        min=1
    )
    bpy.types.Scene.cmg_mc_resolution = IntProperty(
        name="MC Resolution",
        description="MC Resolution",
        default=256,
        min=1
    )
    bpy.types.Scene.cmg_model_format = EnumProperty(
        name="Model Format",
        description="Output model format",
        items=[
            ('obj', 'OBJ', 'Wavefront OBJ format'),
            ('glb', 'GLB', 'GL Binary format'),
        ],
        default='obj'
    )
    bpy.types.Scene.cmg_remove_bg = BoolProperty(
        name="Remove Background",
        description="Remove background from input image",
        default=True
    )
    
    bpy.utils.register_class(CMG_PT_main_panel)
    bpy.utils.register_class(CMG_OT_run)

def unregister():
    del bpy.types.Scene.cmg_input_image
    del bpy.types.Scene.cmg_output_dir
    del bpy.types.Scene.cmg_checkpoint_path
    del bpy.types.Scene.cmg_render_views
    del bpy.types.Scene.cmg_mc_resolution
    del bpy.types.Scene.cmg_model_format
    del bpy.types.Scene.cmg_remove_bg
    
    bpy.utils.unregister_class(CMG_PT_main_panel)
    bpy.utils.unregister_class(CMG_OT_run)

if __name__ == "__main__":
    register()