import numpy as np
import trimesh
from PIL import Image
import io
from tqdm import tqdm
from fibonacci_sphere import *
from moyo.eval.com_evaluation import smplx_to_mesh, visualize_mesh
import pickle as pkl
import cv2
from pathlib import Path

import pyglet
import os
pyglet.options['osx_alt_loop'] = False
pyglet.options['shadow_window'] = False


W, H           = 1024, 1024             # resolution of every rendered frame
focal_px       = 1.2 * W                # pinhole focal length in **pixels**
n_views        = 100                    # how many cameras to sample
radius_factor  = 2   

cam_centers = fibonacci_sphere(n_views) * radius_factor   # (N,3)

# look‑at → rotation matrix (world→cam) and quaternion
def look_at(cam_pos, target=np.zeros(3), up=np.array([0, 1, 0])):
    fwd = (target - cam_pos)
    fwd /= (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(up, fwd); right /= (np.linalg.norm(right) + 1e-8)
    new_up = np.cross(fwd, right)
    Rwc = np.stack([right, new_up, fwd], 0)          # camera frame in world
    Rcw = Rwc.T                                      # world → cam
    qw, qx, qy, qz = rotmat2qvec(Rcw)
    return (qw, qx, qy, qz), (-Rcw @ cam_pos)        # qvec, tvec

def rotmat2qvec(R):
    # Compatible with COLMAP format
    qw = np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
    qx = np.sqrt(max(0, 1 + R[0,0] - R[1,1] - R[2,2])) / 2 * np.sign(R[2,1]-R[1,2])
    qy = np.sqrt(max(0, 1 - R[0,0] + R[1,1] - R[2,2])) / 2 * np.sign(R[0,2]-R[2,0])
    qz = np.sqrt(max(0, 1 - R[0,0] - R[1,1] + R[2,2])) / 2 * np.sign(R[1,0]-R[0,1])
    return qw, qx, qy, qz


# Main processing
def export_to_colmap_format(pose_name: str, video_name: str):
    viewpoints = fibonacci_sphere(100)
    support_dir = 'data'
    data_folder = "mosh/train/"
    width, height = 512, 512
    fov = np.pi / 3
    fx = width / (2 * np.tan(fov / 2))
    
    image_out_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/")
    # pose_out_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/colmap_scene/poses")
    sparse_dir = Path(os.path.join(support_dir, f"{pose_name}/{video_name}/sparse")) # camera's intrinsic + extrinsic params in txt format
    mesh_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/")
    face_ind_dir = os.path.join(support_dir, f"face_inds")

    sparse_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(image_out_dir): 
        os.makedirs(image_out_dir, exist_ok=True)
    # if not os.path.exists(pose_out_dir): 
    #     os.makedirs(pose_out_dir, exist_ok=True)
    if not os.path.exists(sparse_dir): 
        os.makedirs(sparse_dir, exist_ok=True)
    if not os.path.exists(mesh_dir): 
        os.makedirs(mesh_dir, exist_ok=True)
    if not os.path.exists(face_ind_dir): 
        os.makedirs(face_ind_dir, exist_ok=True)

    # read in poses
    pose_file = os.path.join(support_dir, f"{data_folder}/{pose_name}/{video_name}.pkl")
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

    face_inds = []

    # get mesh per fram
    for i, frame_ind in enumerate(range(num_frames)[50:-50:10]):
    # for frame_ind in tqdm(range(num_frames)[100:102]): #we'll probably have to tune this per video
        # frame_ind += 
        frames_from_start = (frame_ind - 50) // 10
        mesh = visualize_mesh(pp_body_model_output, faces, frame_id=frame_ind)
        mesh_path = os.path.join(mesh_dir, f"{pose_name}_t{frame_ind}.ply")

        if i == 0:
            mesh.export(mesh_path)
        # ---------- B. write cameras.txt ----------
            cam_lines = ["# Camera list with one line of data per camera:\n"
                        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                        "# Number of cameras: 1\n"]
            cam_lines.append(f"1 PINHOLE {W} {H} {focal_px} {focal_px} {W/2} {H/2}\n")

            (sparse_dir/"cameras.txt").write_text("".join(cam_lines))

            # ---------- C. write images.txt ----------
            img_hdr = ["# Image list with two lines per image:\n",
                    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
                    "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
                    f"# Number of images: {n_views}, mean observations per image: 0\n"]
            image_lines = img_hdr
            for i, C in enumerate(cam_centers, 1):
                qvec, tvec = look_at(C)
                image_name = f"render_{i:04d}.png"
                image_lines.append(f"{i} {' '.join(map(str,qvec))} "
                                f"{' '.join(map(str,tvec))} 1 {image_name}\n\n")
            (sparse_dir/"images.txt").write_text("".join(image_lines))

            # ---------- D. write points3D.txt ----------
            V = np.asarray(mesh.vertices)                   # (P,3)
            pt_hdr = ["# 3D point list with one line per point:\n",
                    "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as "
                    "(IMAGE_ID, POINT2D_IDX)\n",
                    f"# Number of points: {len(V)}, mean track length: {n_views}\n"]
            pt_lines = pt_hdr
            for pid, xyz in enumerate(V, 1):
                track = " ".join([f"{iid} 0" for iid in range(1, n_views+1)])
                pt_lines.append(f"{pid} {xyz[0]} {xyz[1]} {xyz[2]} 255 255 255 0 {track}\n")
            (sparse_dir/"points3D.txt").write_text("".join(pt_lines))
        
        # get viewpoints (fib sphere) for given timestep
        for i, view in enumerate(viewpoints):
            eye = view * 2.0
            forward = np.array([0, 1, 0]) - eye
            forward /= (np.linalg.norm(forward) + 1e-8)
            right = np.cross(np.array([0, 1, 0]), forward)
            up = np.cross(forward, right)
            cam_to_world = np.eye(4)
            cam_to_world[:3, 0] = right
            cam_to_world[:3, 1] = up
            cam_to_world[:3, 2] = forward
            cam_to_world[:3, 3] = eye

            # Render and save image
            scene = trimesh.Scene(mesh)
            scene.set_camera(angles=view, distance=2)
            png = scene.save_image(resolution=(width, height))

            img_cam_dir = os.path.join(image_out_dir, f"cam{i:02d}")
            os.makedirs(img_cam_dir, exist_ok=True)
            img_path = os.path.join(img_cam_dir, f"frame_{frames_from_start:05d}.jpg")
            with Image.open(io.BytesIO(png)) as img:
                img = img.convert('RGB')
                img.save(img_path)

        print('outside')


if __name__ == "__main__":
    POSE_NAME = 'Boat_Pose'
    VIDEO_NAME = '221004_yogi_nexus_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a_stageii'

    export_to_colmap_format(POSE_NAME, VIDEO_NAME)