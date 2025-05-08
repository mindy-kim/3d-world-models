# Complete standalone script to write and read COLMAP .bin and .txt files (compatible with LLFF/Fyusin readers)

import numpy as np
import os
import struct
import trimesh
from trimesh.transformations import euler_from_matrix, rotation_matrix
from PIL import Image
import io
from tqdm import tqdm
from fibonacci_sphere import *
from moyo.eval.com_evaluation import smplx_to_mesh, visualize_mesh
import pickle as pkl
import cv2

# Util to convert 4x4 pose to quaternion and translation

def matrix_to_qvec_tvec(pose):
    R_mat = pose[:3, :3]
    t = pose[:3, 3]
    q = R.from_matrix(R_mat).as_quat()
    return np.roll(q, 1), t

# Write cameras.bin using 8-byte fields (COLMAP compatible)

def write_cameras_bin(cameras_dict, filepath):
    with open(filepath, "wb") as f:
        f.write(struct.pack("Q", len(cameras_dict)))
        for cam_id, cam in cameras_dict.items():
            f.write(struct.pack("iiQQ", cam_id, cam['model_id'], cam['width'], cam['height']))
            f.write(struct.pack(f"{len(cam['params'])}d", *cam['params']))

# Write cameras.txt

def write_cameras_txt(cameras_dict, filepath):
    with open(filepath, "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS\n")
        for cam_id, cam in cameras_dict.items():
            model_str = "SIMPLE_RADIAL" if cam['model_id'] == 2 else f"MODEL_{cam['model_id']}"
            param_str = " ".join(map(str, cam['params']))
            f.write(f"{cam_id} {model_str} {cam['width']} {cam['height']} {param_str}\n")

# Read cameras.bin (8-byte version)

def read_cameras_bin(filepath):
    cameras = {}
    with open(filepath, "rb") as f:
        num_cams = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_cams):
            cam_id, model_id, width, height = struct.unpack("iiQQ", f.read(24))
            num_params = 3 if model_id in [0, 1] else 4
            params = struct.unpack(f"{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

# Write images.bin using 8-byte compliant format

def write_images_bin(images_dict, filepath):
    with open(filepath, "wb") as f:
        f.write(struct.pack("Q", len(images_dict)))
        for img_id, img in images_dict.items():
            f.write(struct.pack("Q", img_id))
            f.write(struct.pack("4d", *img['qvec']))
            f.write(struct.pack("3d", *img['tvec']))
            f.write(struct.pack("i", img['camera_id']))
            f.write(img['name'].encode("utf-8") + b"\x00")
            f.write(struct.pack("Q", 0))  # number of 2D points

# Write images.txt

def write_images_txt(images_dict, filepath):
    with open(filepath, "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# followed by: POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id, img in images_dict.items():
            q = " ".join(map(str, img['qvec']))
            t = " ".join(map(str, img['tvec']))
            f.write(f"{img_id} {q} {t} {img['camera_id']} {img['name']}\n\n")

# Write points3D.bin with dummy tracks

def write_points3D_bin(points_dict, filepath):
    with open(filepath, "wb") as f:
        f.write(struct.pack("Q", len(points_dict)))
        for pid, pt in points_dict.items():
            f.write(struct.pack("Q", pid))
            f.write(struct.pack("3d", *pt['xyz']))
            f.write(struct.pack("3B", *pt['rgb']))
            f.write(struct.pack("d", pt['error']))
            f.write(struct.pack("Q", 1))  # 1 dummy track
            f.write(struct.pack("ii", pt['image_id'], pt['point2D_idx']))

    # if os.path.exists(face_ind_path):
    #     face_indices = np.load(face_ind_path, np.array(face_indices))
    #     points = mesh[face_indices]
    # else:
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    print(face_indices)
    face_indices = np.array(face_indices)
    print(face_indices.shape)
    print(np.unique(face_indices).shape)
    np.save(face_ind_path, face_indices)

    
    with open(out_path, 'w') as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, pt in enumerate(points):
            r, g, b = colors[i]
            f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {int(r)} {int(g)} {int(b)} 0.01\n")

def export_to_colmap_bin(pose_name: str, video_name: str):
    viewpoints = fibonacci_sphere(100)
    support_dir = 'data'
    data_folder = "mosh/val/"
    width, height = 512, 512
    fov = np.pi / 3
    fx = width / (2 * np.tan(fov / 2))

    base_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/colmap_scene")
    image_out_dir = os.path.join(base_dir, "images")
    mesh_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/moyo_mesh")
    sparse_dir = os.path.join(base_dir, "sparse/0")

    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    pose_file = os.path.join(support_dir, f"{data_folder}{pose_name}_{video_name}.pkl")
    pp_params = pkl.load(open(pose_file, 'rb'))
    num_frames = len(pp_params['fullpose'])

    gender = 'neutral'
    pp_body_model_output, _, _, faces = smplx_to_mesh(
        pp_params,
        f'{support_dir}/models_lockedhead/smplx/SMPLX_NEUTRAL.npz',
        'smplx',
        gender=gender
    )

    cam_id = 1
    cameras = {
        cam_id: {
            'model_id': 2,  # SIMPLE_RADIAL
            'width': width,
            'height': height,
            'params': [fx, width / 2, height / 2, 0.0]
        }
    }

    images = {}
    points3D = {}
    frame_ind = 100
    pp_mesh = visualize_mesh(pp_body_model_output, faces, frame_id=frame_ind)
    mesh_path = os.path.join(mesh_dir, f"{pose_name}_t{frame_ind}.ply")
    pp_mesh.export(mesh_path)

    image_id = 1
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

        qvec, tvec = matrix_to_qvec_tvec(cam_to_world)

        scene = trimesh.Scene(pp_mesh)
        scene.set_camera(angles=view, distance=2)
        png = scene.save_image(resolution=(width, height))

        img_path = os.path.join(image_out_dir, f"{image_id:06d}.jpg")
        with Image.open(io.BytesIO(png)) as img:
            img = img.convert('RGB')
            img.save(img_path)

        images[image_id] = {
            'camera_id': cam_id,
            'qvec': qvec,
            'tvec': tvec,
            'name': f"{image_id:06d}.jpg"
        }
        image_id += 1

    points, _ = trimesh.sample.sample_surface(pp_mesh, 10000)
    for i, pt in enumerate(points):
        points3D[i + 1] = {
            'xyz': pt,
            'rgb': [200, 150, 120],
            'error': 0.01,
            'image_id': 1,  # dummy
            'point2D_idx': 0  # dummy
        }

    # Write BIN files
    write_cameras_bin(cameras, os.path.join(sparse_dir, 'cameras.bin'))
    write_images_bin(images, os.path.join(sparse_dir, 'images.bin'))
    write_points3D_bin(points3D, os.path.join(sparse_dir, 'points3D.bin'))

    # Also write TXT versions
    write_cameras_txt(cameras, os.path.join(sparse_dir, 'cameras.txt'))
    write_images_txt(images, os.path.join(sparse_dir, 'images.txt'))
    write_points3D_txt(points3D, os.path.join(sparse_dir, 'points3D.txt'))

    print("\nCOLMAP export complete (BIN and TXT).")

def trimesh_to_nerf_transform(camera_transform):
    # Step 1: Invert the camera transform (world-to-camera → camera-to-world)
    nerf_transform = np.linalg.inv(camera_transform)
    
    return nerf_transform


# Main processing
def export_to_4dgs_format(pose_name: str, video_name: str, start_frame: int, end_frame: int, step: int):
    support_dir = './data'
    data_folder = "mosh/train/"
    
    image_out_dir = os.path.join(support_dir, f"4dgs")
    mesh_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/moyo_mesh")
    json_file = os.path.join(support_dir, '4dgs/transforms_train.json')

    if not os.path.exists(image_out_dir): 
        os.makedirs(image_out_dir, exist_ok=True)
    if not os.path.exists(mesh_dir): 
        os.makedirs(mesh_dir, exist_ok=True)

    # read in poses
    pose_file = os.path.join(support_dir, f"{data_folder}/{pose_name}/{video_name}.pkl")
    pp_params = pkl.load(open(pose_file, 'rb'))
    print(len(pp_params['fullpose']))
    num_frames = ((end_frame - start_frame) // step) + 1

    print(num_frames)
    viewpoints = fibonacci_sphere(num_frames)
    width, height = 64, 64
    fov = np.pi / 3
    fx = width / (2 * np.tan(fov / 2))

    transforms = {}
    content = None

    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            content = f.read().strip()
    
    if content:
        with open(json_file, "r") as f:
            transforms = json.load(f)
        start_frame = len(transforms['frames']) * step + start_frame
    else:
        transforms['frames'] = []

    gender = 'neutral'
    pp_body_model_output, _, _, faces = smplx_to_mesh(
        pp_params,
        f'{support_dir}/models_lockedhead/smplx/SMPLX_NEUTRAL.npz',
        'smplx',
        gender=gender
    )

    # get mesh per frame
    for i in tqdm(range(start_frame, end_frame, step)):
        print(i)
        pp_mesh = visualize_mesh(pp_body_model_output, faces, frame_id=i)

        i = len(transforms['frames'])
        print(i)

    # get mesh per fram
    for frame_ind in tqdm(range(num_frames)[790:-50:10]):
    # for frame_ind in tqdm(range(num_frames)[100:102]): #we'll probably have to tune this per video
        # frame_ind += 
        frames_from_start = (frame_ind - 50) // 10
        pp_mesh = visualize_mesh(pp_body_model_output, faces, frame_id=frame_ind)
        mesh_path = os.path.join(mesh_dir, f"{pose_name}_t{frame_ind}.ply")
        pp_mesh.export(mesh_path)

        transform_frame = {}
        
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

            # pose_cam_dir = os.path.join(pose_out_dir, f"cam{i:02d}")
            # os.makedirs(pose_cam_dir, exist_ok=True)
            # np.save(os.path.join(pose_cam_dir, f"{i:06d}.npy"), cam_to_world)

        # Render and save image
        scene = trimesh.Scene(pp_mesh)
        pp_mesh.apply_translation(-pp_mesh.bounds.mean(axis=0))  # Center mesh at (0, 0, 0)

        # Set the camera transform
        params = nerf_to_trimesh_camera(np.array(old_transforms['frames'][i]['transform_matrix']))
        scene.set_camera(**params)

        # scene.camera_transform = np.array(old_transforms['frames'][i]['transform_matrix'])

        # scene.set_camera(angles=view, distance=2.5)
        # cam_to_world = trimesh_to_nerf_transform(scene.camera_transform)
        # cam_to_world = scene.camera_transform

        try:
            png = scene.save_image(resolution=(width, height))

            img_cam_dir = os.path.join(image_out_dir, f"cam{i:02d}")
            os.makedirs(img_cam_dir, exist_ok=True)
            img_path = os.path.join(img_cam_dir, f"frame_{frames_from_start:05d}.jpg")
            with Image.open(io.BytesIO(png)) as img:
                img.save(img_path)
        except Exception as e:
            break

        print('outside')
        # # Write COLMAP format files
        # write_images_txt(image_out_dir, pose_out_dir, os.path.join(sparse_dir, "images.txt"))
        # write_cameras_txt(os.path.join(sparse_dir, "cameras.txt"), width, height, fx)
        # # write_points3D_txt(face_ind_dir, mesh_path, os.path.join(sparse_dir, "points3D.txt")) # find what the face indices are?

        # mesh = trimesh.load(mesh_path, process=False)
        # colors = np.tile(np.array([[200, 150, 120]]), (10000, 1))
        # points, face_indices = trimesh.sample.sample_surface(mesh, 10000, 142)
        # face_indices = np.array(face_indices)
        # print(face_indices.shape)
        # print(np.unique(face_indices).shape)

        # if frame_ind == 101:
        #     print(np.array_equal(face_indices, face_inds))
        # else:
        #     face_inds = face_indices


if __name__ == "__main__":
    export_to_colmap_bin("220923_yogi_body_hands_03596_Tree_Pose_or_Vrksasana", "-a_stageii")
