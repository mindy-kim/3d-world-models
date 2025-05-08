from pathlib import Path
from read_model import read_cameras_text, read_images_text, read_points3D_text

# Replace this path with your actual output path
sparse_path = Path("data/220923_yogi_body_hands_03596_Tree_Pose_or_Vrksasana/-a_stageii/colmap_scene/sparse/0")

cameras = read_cameras_text(sparse_path / "cameras.txt")
images = read_images_text(sparse_path / "images.txt")
points3D = read_points3D_text(sparse_path / "points3D.txt")

print("Loaded cameras:", len(cameras))
print("Loaded images:", len(images))
print("Loaded 3D points:", len(points3D))
