import os
from pathlib import Path
import re

def shift_frame_indices(directory):
    """
    For each cam_xxxxx folder inside the directory, shift frame_00000.jpg to frame_00001.jpg, etc.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    # Match folders like cam00000, cam00001, ...
    cam_folders = sorted([f for f in directory.iterdir() if f.is_dir() and f.name.startswith("cam")])

    for cam in cam_folders:
        print(f"Processing {cam.name}...")
        # Match files like frame_00000.jpg
        frame_files = sorted(cam.glob("frame_*.jpg"))

        # Rename in reverse to avoid overwriting
        for frame_path in reversed(frame_files):
            match = re.match(r"frame_(\d{5})\.jpg", frame_path.name)
            if match:
                idx = int(match.group(1))
                new_idx = idx + 1
                new_name = f"frame_{new_idx:05d}.jpg"
                new_path = frame_path.with_name(new_name)
                print(f"Renaming {frame_path.name} → {new_name}")
                frame_path.rename(new_path)
            else:
                print(f"Skipping non-matching file: {frame_path.name}")
if __name__ == "__main__":
    # CHANGE THIS to the directory that contains the cam_XXXXX folders
    root_dir = "data/multipleview/pose1"
    shift_frame_indices(root_dir)

