#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
import copy
def multithread_write(image_list, path, scene_name=""):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, scene_name + '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, action, name, iteration, views, gaussians, pipeline, background, cam_type, scene_name = ""):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        rendering = render(view, gaussians, pipeline, background, action = action, cam_type=cam_type)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path, scene_name)

    multithread_write(render_list, render_path, scene_name)

    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), f'{scene_name}_2_video_rgb.mp4'), render_images, fps=30)
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        dataset2 = copy.deepcopy(dataset)
        dataset2.source_path = dataset2.z_second_source_path 
        scene1 = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scene2 = Scene(dataset2, gaussians, load_iteration=iteration, shuffle=False)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        action1 = torch.tensor([1, 1]).float()
        action2 = torch.tensor([0, 0]).float()
        
        cam_type=scene1.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, action1, "train", scene1.loaded_iter, scene1.getTrainCameras(), gaussians, pipeline, background,cam_type,"boat_pose")
            render_set(dataset.model_path, action2, "train", scene2.loaded_iter, scene2.getTrainCameras(), gaussians, pipeline, background,cam_type,"side_plank")
        if not skip_test:
            render_set(dataset.model_path, action1, "test", scene1.loaded_iter, scene1.getTestCameras(), gaussians, pipeline, background,cam_type,"boat_pose")
            render_set(dataset.model_path, action2, "test", scene2.loaded_iter, scene2.getTestCameras(), gaussians, pipeline, background,cam_type,"side_plank")
        if not skip_video:
            render_set(dataset.model_path, action1, "video",scene1.loaded_iter,scene1.getVideoCameras(),gaussians,pipeline,background,cam_type,"boat_pose")
            render_set(dataset.model_path, action2, "video",scene2.loaded_iter,scene2.getVideoCameras(),gaussians,pipeline,background,cam_type,"side_plank")
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)