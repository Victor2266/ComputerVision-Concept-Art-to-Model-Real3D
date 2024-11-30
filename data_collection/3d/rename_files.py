from pathlib import Path

PATH = Path.home() / "Downloads" / "rename"

PREFIX = "fab"

if __name__ == "__main__":
    for path in PATH.iterdir():
        if path.is_file():
            new_name = f"{PREFIX}_{path.name.lower()}"
            new_path = path.parent / new_name
            path.rename(new_path)
            print(f"Renamed {path} to {new_path}")
