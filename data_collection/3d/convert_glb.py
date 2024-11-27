import subprocess
from pathlib import Path

# download: https://github.com/facebookincubator/FBX2glTF/releases
PATH_TO_FBX2GLTF = Path.home() / "Downloads" / "FBX2glTF-windows-x64.exe"
DIR_PATH = Path(__file__).parent
INPUT_DIR = DIR_PATH / "glb_input"
OUTPUT_DIR = DIR_PATH / "glb_output"


def fbx_to_gltf(fbx_path: Path):
    output_file = OUTPUT_DIR / f"{fbx_path.stem}.glb"

    command = [
        str(PATH_TO_FBX2GLTF),
        "--input",
        str(fbx_path),
        "--output",
        str(output_file.with_suffix("")),
        "--binary",
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Output file: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    for input_path in INPUT_DIR.iterdir():
        if input_path.suffix.lower() != ".fbx":
            print(f"Skipped {input_path.name}")
            continue

        fbx_to_gltf(input_path)
