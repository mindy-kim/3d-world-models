import os
from pathlib import Path

def rename_cam_folders(directory):
    """
    Renames folders like cam_00 → cam_00000, cam_01 → cam_00001, ..., cam_99 → cam_00099
    inside the given directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a valid directory")

    for i in range(100):
        old_name = directory / f"cam_{i+1:02d}"
        new_name = directory / f"cam{i+1:02d}"

        if old_name.exists() and old_name.is_dir():
            if new_name.exists():
                print(f"Skipping: {new_name} already exists.")
            else:
                print(f"Renaming: {old_name.name} → {new_name.name}")
                old_name.rename(new_name)
        else:
            print(f"Missing: {old_name} does not exist.")

if __name__ == "__main__":
    # CHANGE THIS TO YOUR DIRECTORY PATH
    root_dir = "data/multipleview/pose1"
    rename_cam_folders(root_dir)

