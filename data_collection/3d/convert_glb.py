import shutil
import subprocess
import zipfile
from pathlib import Path

# download: https://github.com/facebookincubator/FBX2glTF/releases
PATH_TO_FBX2GLTF = Path.home() / "Downloads" / "FBX2glTF-windows-x64.exe"
DIR_PATH = Path(__file__).parent
INPUT_DIR = DIR_PATH / "glb_input"
OUTPUT_DIR = DIR_PATH / "glb_output"

CLEAR_INPUT_DIR = False

DOWNLOAD_ORIGIN = "turbosquid"


def fbx_to_gltf(fbx_path: Path):
    output_file = (
        OUTPUT_DIR
        / f"{DOWNLOAD_ORIGIN.lower()}_{fbx_path.stem.lower().replace(' ', '_')}.glb"
    )

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


def extract_zip(path, output_dir):
    with zipfile.ZipFile(path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if ".." in member or member.startswith("/") or member.startswith("\\"):
                raise ValueError(f"Unsafe path detected: {member}")
        zip_ref.extractall(output_dir)


def extract_rar(path, output_dir):
    subprocess.run(["7z", "x", str(path), f"-o{output_dir}"], check=True)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    for input_path in INPUT_DIR.iterdir():
        if input_path.suffix not in [".zip", ".rar"]:
            continue

        extract_path = INPUT_DIR / input_path.stem
        extract_path.mkdir(exist_ok=True)

        if input_path.suffix == ".rar":
            extract_rar(input_path, extract_path)
        elif input_path.suffix == ".zip":
            extract_zip(input_path, extract_path)

        for fbx_path in extract_path.rglob("*.fbx"):
            fbx_to_gltf(fbx_path)

            if CLEAR_INPUT_DIR:
                if extract_path.is_dir():
                    shutil.rmtree(extract_path)
                input_path.unlink()

            break
