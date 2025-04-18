import torch
import numpy as np
import smplx
import pickle as pkl
from moyo.eval.com_evaluation import smplx_to_mesh, visualize_mesh
from moyo.utils.constants import frame_select_dict_combined as frame_select_dict
import trimesh
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp
from PIL import Image
import io

support_dir = './data'

# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
comp_device

amass_npz_fname = osp.join(support_dir, 'YOGI_2_latest_smplx_neutral/train/220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a_stageii.npz') # the path to body data
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = bdata['gender']

print('Data keys available:%s'%list(bdata.keys()))

print('The subject of the mocap sequence is  {}.'.format(subject_gender))

from human_body_prior.body_model.body_model import BodyModel

bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))

num_betas = 16 # number of body parameters

bm = BodyModel(f'{support_dir}/models_lockedhead/smplx/SMPLX_NEUTRAL.npz', 'smplx', num_betas=num_betas).to(comp_device)
faces = c2c(bm.f)

time_length = len(bdata['trans'])

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
    'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
}

print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
print('time_length = {}'.format(time_length))

mosh = f"{support_dir}/mosh/train/220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a_stageii.pkl"

# all_pose_names = [pp_name.replace('_stageii', '') for pp_name in pp_names]
selected_frame = frame_select_dict['_'.join(mosh.split('_')[5:-1])]

pp_pkl = mosh
pp_params = pkl.load(open(pp_pkl, 'rb'))
# read height, weight, gender meta
height = None
weight = None
gender = 'neutral'
subj_id = 'unknown'
pp_body_model_output, smplx_params, transls, faces = smplx_to_mesh(pp_params, f'{support_dir}/models_lockedhead/smplx/SMPLX_NEUTRAL.npz','smplx', gender=gender)

pp_mesh = visualize_mesh(pp_body_model_output, faces, frame_id=550)
scene = trimesh.scene.scene.Scene(pp_mesh)

scene.set_camera(angles=(2,2,2), distance=5)
png = scene.save_image()
image = Image.open(io.BytesIO(png))
image.save("./sample.png")
# print(f"PNG file saved as {"./sample.png"}")