import numpy as np
import os
import trimesh
from trimesh.transformations import euler_from_matrix, rotation_matrix
from PIL import Image
import io
from tqdm import tqdm
from fibonacci_sphere import *
from moyo.eval.com_evaluation import smplx_to_mesh, visualize_mesh
import pickle as pkl
import cv2
import json

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
def write_points3D_txt(face_ind_path, mesh_path, out_path, num_points=10000):
    mesh = trimesh.load(mesh_path, process=False)
    colors = np.tile(np.array([[200, 150, 120]]), (num_points, 1))

    # if os.path.exists(face_ind_path):
    #     face_indices = np.load(face_ind_path, np.array(face_indices))
    #     points = mesh[face_indices]
    # else:
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    face_indices = np.array(face_indices)
    
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
    face_ind_dir = os.path.join(support_dir, f"face_inds")

    if not os.path.exists(image_out_dir): 
        os.makedirs(image_out_dir, exist_ok=True)
    if not os.path.exists(pose_out_dir): 
        os.makedirs(pose_out_dir, exist_ok=True)
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

    # get mesh per frame
    # for frame_ind in tqdm(range(num_frames)[50:-50:10]):
    for frame_ind in tqdm(range(num_frames)[100:102]):
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

        print('outside')
        # Write COLMAP format files
        write_images_txt(image_out_dir, pose_out_dir, os.path.join(sparse_dir, "images.bin"))
        write_cameras_txt(os.path.join(sparse_dir, "cameras.bin"), width, height, fx)
        # write_points3D_txt(face_ind_dir, mesh_path, os.path.join(sparse_dir, "points3D.txt")) # find what the face indices are?

        mesh = trimesh.load(mesh_path, process=False)
        colors = np.tile(np.array([[200, 150, 120]]), (10000, 1))
        points, face_indices = trimesh.sample.sample_surface(mesh, 10000, 142)
        face_indices = np.array(face_indices)
        print(face_indices.shape)
        print(np.unique(face_indices).shape)

        if frame_ind == 101:
            print(np.array_equal(face_indices, face_inds))
        else:
            face_inds = face_indices

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

        mesh_path = os.path.join(mesh_dir, f"{pose_name}_t{i}.ply")
        pp_mesh.export(mesh_path)

        transform_frame = {}
        
        # get viewpoints (fib sphere) for given timestep
        view = viewpoints[i]

        # Render and save image
        scene = trimesh.Scene(pp_mesh)
        scene.set_camera(angles=view, distance=2.5)
        # cam_to_world = trimesh_to_nerf_transform(scene.camera_transform)
        cam_to_world = scene.camera_transform

        try:
            png = scene.save_image(resolution=(width, height))
            img_path = os.path.join(image_out_dir, f"r_{i:03d}.png")
            with Image.open(io.BytesIO(png)) as img:
                img.save(img_path)
        except Exception as e:
            break

        transforms['camera_angle_x'] = scene.camera.fov[0]
        transform_frame['file_path'] = f"./r_{i:03d}"
        transform_frame['rotation'] = 2 * np.pi / num_frames  # one full circle divided by number of frames
        transform_frame['time'] = float(i) / num_frames
        transform_frame['transform_matrix'] = cam_to_world.tolist()

        transforms['frames'].append(transform_frame)

    with open(json_file, "w") as f:
        json.dump(transforms, f, indent=2)





def nerf_to_trimesh_camera(transform_matrix):
    """
    Convert NeRF's transform_matrix (camera-to-world) into trimesh.set_camera() parameters,
    but force the camera to look at (0, 0, 0).

    Args:
        transform_matrix (np.ndarray): 4x4 camera-to-world matrix from NeRF.
        default_fov (float): Default field of view if not inferrable.

    Returns:
        dict: Parameters for trimesh.scene.Scene.set_camera().
    """
    # Extract camera position (translation part of the matrix)
    camera_position = transform_matrix[:3, 3]

    # Compute new rotation to look at origin
    # Trimesh expects the camera to face -Z, so we compute a rotation that makes -Z point toward (0,0,0)
    direction_to_origin = -camera_position  # Vector from camera to origin (flipped for look-at)
    direction_to_origin_normalized = direction_to_origin / np.linalg.norm(direction_to_origin)

    # Default "up" vector (Y-up, but you can adjust if needed)
    up_vector = np.array([0, 1, 0])  # Y-up convention

    # Compute rotation matrix that aligns -Z with direction_to_origin_normalized
    rotation = rotation_matrix_lookat(direction_to_origin_normalized, up_vector)

    # Convert rotation matrix to Euler angles (XYZ order)
    angles = euler_from_matrix(rotation, 'sxyz')

    # Distance is simply the norm from camera to origin
    distance = np.linalg.norm(camera_position)

    return {
        'angles': angles,
        'distance': distance,
        'center': [0, 0, 0],  # Force look-at origin
    }

def rotation_matrix_lookat(target_dir, up_vector):
    """
    Compute a rotation matrix that makes -Z point along `target_dir`.
    (Trimesh cameras face -Z, so we need to align -Z with the look direction.)
    """
    # Normalize target direction
    target_dir = target_dir / np.linalg.norm(target_dir)

    # Compute orthogonal axes
    right = np.cross(up_vector, target_dir)
    right = right / np.linalg.norm(right)
    new_up = np.cross(target_dir, right)

    # Construct rotation matrix (columns are right, up, -forward)
    rotation = np.eye(4)
    rotation[:3, 0] = right      # X-axis (right)
    rotation[:3, 1] = new_up     # Y-axis (up)
    rotation[:3, 2] = -target_dir  # Z-axis (-forward, since Trimesh faces -Z)
    return rotation


# Main processing
def json_to_4dgs_format(pose_name: str, video_name: str, old_json_file, new_json_file):
    support_dir = './data'
    data_folder = "mosh/train/"
    
    image_out_dir = os.path.join(support_dir, f"4dgs")
    mesh_dir = os.path.join(support_dir, f"{pose_name}/{video_name}/moyo_mesh")

    if not os.path.exists(image_out_dir): 
        os.makedirs(image_out_dir, exist_ok=True)
    if not os.path.exists(mesh_dir): 
        os.makedirs(mesh_dir, exist_ok=True)

    with open(old_json_file, "r") as f:
        old_transforms = json.load(f)

    transforms = {}
    content = None

    if os.path.exists(new_json_file):
        with open(new_json_file, "r") as f:
            content = f.read().strip()
    
    if content:
        with open(new_json_file, "r") as f:
            transforms = json.load(f)
        start_frame = len(transforms['frames'])
    else:
        transforms['frames'] = []

    # read in poses
    pose_file = os.path.join(support_dir, f"{data_folder}/{pose_name}/{video_name}.pkl")
    pp_params = pkl.load(open(pose_file, 'rb'))

    width, height = 64, 64
    fov = np.pi / 3
    fx = width / (2 * np.tan(fov / 2))

    gender = 'neutral'
    pp_body_model_output, _, _, faces = smplx_to_mesh(
        pp_params,
        f'{support_dir}/models_lockedhead/smplx/SMPLX_NEUTRAL.npz',
        'smplx',
        gender=gender
    )

    # get mesh per frame
    for i in tqdm(range(len(transforms['frames']), len(old_transforms['frames']))):
        pp_mesh = visualize_mesh(pp_body_model_output, faces, frame_id=i+50)

        mesh_path = os.path.join(mesh_dir, f"{pose_name}_t{i}.ply")
        pp_mesh.export(mesh_path)

        transform_frame = {}

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
            img_path = os.path.join(image_out_dir, f"r_{i:03d}.png")
            with Image.open(io.BytesIO(png)) as img:
                img.save(img_path)
        except Exception as e:
            break

        transforms['camera_angle_x'] = scene.camera.fov[0]
        transform_frame['file_path'] = f"./val/r_{i:03d}"
        transform_frame['rotation'] = old_transforms['frames'][i]['rotation']  # one full circle divided by number of frames
        transform_frame['time'] = old_transforms['frames'][i]['time']
        transform_frame['transform_matrix'] = old_transforms['frames'][i]['transform_matrix']

        transforms['frames'].append(transform_frame)

    with open(new_json_file, "w") as f:
        json.dump(transforms, f, indent=2)


# if __name__ == "__main__":
#     export_to_colmap_format()