"""
mesh2colmap_model.py
Create a minimal COLMAP model from a watertight mesh.
"""

import numpy as np
from pathlib import Path
# from llff.poses.colmap_wrapper import qvec2rotmat  # or copy from COLMAP scripts


import numpy as np
import os
import struct
import trimesh
from PIL import Image
import io
from tqdm import tqdm
from fibonacci_sphere import *
from moyo.eval.com_evaluation import smplx_to_mesh, visualize_mesh
import pickle as pkl
from scipy.spatial.transform import Rotation as R


# ---------- user parameters ----------
mesh_file      = "scene.obj"
out_dir        = Path("sparse_txt")      # will hold cameras.txt, images.txt, ...
img_dir        = Path("images")          # your rendered RGBA frames live here
W, H           = 1920, 1080             # resolution of every rendered frame
focal_px       = 1.2 * W                # pinhole focal length in **pixels**
n_views        = 100                    # how many cameras to sample
radius_factor  = 1.3                    # distance of camera sphere
# -------------------------------------

out_dir.mkdir(parents=True, exist_ok=True)

# ---------- A. sample cameras ----------
# def fibonacci_sphere(n):
#     import math
#     golden = math.pi * (3 - np.sqrt(5))
#     points = []
#     for i in range(n):
#         y = 1 - (i / float(n - 1)) * 2       # y goes from 1 to -1
#         radius = np.sqrt(1 - y * y)
#         theta = golden * i
#         x, z = np.cos(theta) * radius, np.sin(theta) * radius
#         points.append(np.array([x, y, z]))
#     return np.stack(points)

cam_centers = fibonacci_sphere(n_views) * radius_factor   # (N,3)

# look‑at → rotation matrix (world→cam) and quaternion
def look_at(cam_pos, target=np.zeros(3), up=np.array([0, 1, 0])):
    fwd = (target - cam_pos)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
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
# ---------- B. write cameras.txt ----------
cam_lines = ["# Camera list with one line of data per camera:\n"
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
             "# Number of cameras: 1\n"]
cam_lines.append(f"1 PINHOLE {W} {H} {focal_px} {focal_px} {W/2} {H/2}\n")
(out_dir/"cameras.txt").write_text("".join(cam_lines))

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
(out_dir/"images.txt").write_text("".join(image_lines))

# ---------- D. write points3D.txt ----------
# Use mesh vertices as sparse points, assume every camera sees every point
mesh = trimesh.load("data/220923_yogi_body_hands_03596_Tree_Pose_or_Vrksasana/-a_stageii/moyo_mesh/220923_yogi_body_hands_03596_Tree_Pose_or_Vrksasana_t100.ply", process=False)
V = np.asarray(mesh.vertices)                   # (P,3)
pt_hdr = ["# 3D point list with one line per point:\n",
          "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as "
          "(IMAGE_ID, POINT2D_IDX)\n",
          f"# Number of points: {len(V)}, mean track length: {n_views}\n"]
pt_lines = pt_hdr
for pid, xyz in enumerate(V, 1):
    track = " ".join([f"{iid} 0" for iid in range(1, n_views+1)])
    pt_lines.append(f"{pid} {xyz[0]} {xyz[1]} {xyz[2]} 255 255 255 0 {track}\n")
(out_dir/"points3D.txt").write_text("".join(pt_lines))

print("✅  Text model written to", out_dir)
