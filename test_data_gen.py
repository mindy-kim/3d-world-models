import numpy as np
import os
import trimesh
from PIL import Image
import io
from tqdm import tqdm
from fibonacci_sphere import *
from moyo.eval.com_evaluation import smplx_to_mesh, visualize_mesh
import pickle as pkl
import cv2

# Util to create a quaternion from a 3x3 rotation matrix
def matrix_to_quaternion(R):
    return trimesh.transformations.quaternion_from_matrix(R)

# Util to build COLMAP-style camera pose text file (images.txt)
def write_images_txt(image_dir, pose_dir, out_path):
    image_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n"]
    for i, pose_file in enumerate(sorted(os.listdir(pose_dir))):
        pose = np.load(os.path.join(pose_dir, pose_file))
        R = pose[:3, :3]
        t = pose[:3, 3]
        q = matrix_to_quaternion(pose)
        image_name = f"{i:06d}.png"
        image_lines.append(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {image_name}\n\n")
    with open(out_path, 'w') as f:
        f.writelines(image_lines)

# Util to build COLMAP-style cameras.txt
# Assuming all cameras are the same intrinsics
def write_cameras_txt(out_path, width, height, fx):
    cx, cy = width / 2, height / 2
    with open(out_path, 'w') as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 SIMPLE_RADIAL {width} {height} {fx} {cx} {cy} 0\n")

# Util to sample and write points3D.txt from mesh
def write_points3D_txt(mesh_path, out_path, num_points=10000):
    mesh = trimesh.load(mesh_path, process=False)
    # wtf is face_indices
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    colors = np.tile(np.array([[200, 150, 120]]), (num_points, 1))
    
    with open(out_path, 'w') as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, pt in enumerate(points):
            r, g, b = colors[i]
            f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {int(r)} {int(g)} {int(b)} 0.01\n")

# Main processing
def export_to_colmap_format(pose_name: str, video_name: str):
    viewpoints = fibonacci_sphere(100)
    support_dir = 'data'
    data_folder = "mosh/train/"
    width, height = 512, 512
    fov = np.pi / 3
    fx = width / (2 * np.tan(fov / 2))
    image_out_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/colmap_scene/images")
    pose_out_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/colmap_scene/poses")
    sparse_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/colmap_scene/sparse/0") # camera's intrinsic + extrinsic params in txt format
    mesh_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/moyo_mesh")

    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(pose_out_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # read in poses
    pose_file = os.path.join(support_dir, f"{data_folder}/{pose_name}/{video_name}")
    if pose_file.endswith(".pkl"):
        pp_params = pkl.load(open(pose_file, 'rb'))
        num_frames = len(pp_params['fullpose'])
    else:
        pp_params = np.load(pose_file)
        num_frames = len(pp_params['poses'])

    gender = 'neutral'
    pp_body_model_output, _, _, faces = smplx_to_mesh(
        pp_params,
        f'{support_dir}/models_lockedhead/smplx/SMPLX_NEUTRAL.npz',
        'smplx',
        gender=gender
    )

    # get mesh per frame
    for frame_ind in tqdm(range(num_frames)[50:-50:10]):
        pp_mesh = visualize_mesh(pp_body_model_output, faces, frame_id=frame_ind)
        mesh_path = os.path.join(mesh_dir, f"{pose_name}_t{frame_ind}.ply")
        pp_mesh.export(mesh_path)
        
        # get viewpoints (fib sphere) for given timestep
        for i, view in enumerate(viewpoints):
            eye = view * 2.0
            forward = np.array([0, 1, 0]) - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(np.array([0, 1, 0]), forward)
            up = np.cross(forward, right)
            cam_to_world = np.eye(4)
            cam_to_world[:3, 0] = right
            cam_to_world[:3, 1] = up
            cam_to_world[:3, 2] = forward
            cam_to_world[:3, 3] = eye

            np.save(os.path.join(pose_out_dir, f"{i:06d}.npy"), cam_to_world)

            # Render and save image
            scene = trimesh.Scene(pp_mesh)
            scene.set_camera(angles=view, distance=2)
            png = scene.save_image(resolution=(width, height))
            img_path = os.path.join(image_out_dir, f"{i:06d}.png")
            with Image.open(io.BytesIO(png)) as img:
                img.save(img_path)

        # Write COLMAP format files
        write_images_txt(image_out_dir, pose_out_dir, os.path.join(sparse_dir, "images.txt"))
        write_cameras_txt(os.path.join(sparse_dir, "cameras.txt"), width, height, fx)
        write_points3D_txt(mesh_path, os.path.join(sparse_dir, "points3D.txt")) # find what the face indices are?

if __name__ == "__main__":
    export_to_colmap_format()