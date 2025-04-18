import numpy as np
import pickle as pkl
from moyo.eval.com_evaluation import smplx_to_mesh, visualize_mesh
import trimesh
from PIL import Image
import io
from fibonacci_sphere import *
import time
from tqdm import tqdm
import os

def main():
    viewpoints = fibonacci_sphere(60)
    
    support_dir = 'data'
    data_folder = "mosh/train/"

    for pose_file in os.listdir(f"{support_dir}/{data_folder}")[:1]:
        pose_name = os.path.splitext(pose_file)[0].split("_", 1)[1]
        if pose_file.endswith(".pkl"):
            pp_params = pkl.load(open(f"{support_dir}/{data_folder}/{pose_file}", 'rb'))  
            num_frames = len(pp_params['fullpose'])
        else:
            pp_params = np.load(pose_file)
            num_frames = len(pp_params['poses'])
        gender = 'neutral'
        pp_body_model_output, _, _, faces = smplx_to_mesh(pp_params, f'{support_dir}/models_lockedhead/smplx/SMPLX_NEUTRAL.npz','smplx', gender=gender)

        for frame_ind in tqdm(range(num_frames)[200:201]):
            pp_mesh = visualize_mesh(pp_body_model_output, faces, frame_id=frame_ind)
            for view in viewpoints:
                scene = trimesh.scene.scene.Scene(pp_mesh)
                scene.set_camera(angles=view, distance=2)
                png = scene.save_image()
                with Image.open(io.BytesIO(png)) as img:
                    # time.sleep(1)
                    img.save(f"{support_dir}/moyo_png/{pose_name}___t{frame_ind}_v{view}.png") # 3 _'s
                # scene.reset()

if __name__ == "__main__":
    main()