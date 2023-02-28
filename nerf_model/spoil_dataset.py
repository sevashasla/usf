import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np


def get_name_dir(opt):
    dir_name = "loaded_"
    if opt.sparse_views:
        dir_name += f"sr_{opt.sparse_ratio:.3f}_"
    if opt.pixel_denoising:
        dir_name += f"pnr_{opt.pixel_noise_ratio:.3f}_"
    if opt.region_denoising:
        dir_name += f"rnr_{opt.region_noise_ratio:.3f}_"
    if opt.super_resolution:
        dir_name += f"dense_sr_{int(opt.dense_sr)}_"
        dir_name += f"sr_factor_{opt.sr_factor:.3f}_"
    if opt.label_propagation:
        dir_name += f"pp_{opt.partial_perc:.3f}_"
    return dir_name


def save_spoiled(opt, poses, images, semantic_images):
    dir_name = get_name_dir(opt)
    path_saved = os.path.join(opt.path, dir_name)
    print(f"[INFO] save spoled images to {path_saved}")
    if os.path.exists(os.path.join(path_saved, "poses.npy")):
        answer = input("Path to elements already exists. Overrite? Y/n")
        if answer != 'Y':
            print('Not saved!')
            return

    with open(os.path.join(path_saved, "poses.npy")) as f:
        for i in tqdm.tqdm(desc=f'Save poses'):
            np.save(f, poses[i])

    with open(os.path.join(path_saved, "images.npy")) as f:
        for i in tqdm.tqdm(desc=f'Save images'):
            np.save(f, images[i])

    with open(os.path.join(path_saved, "semantic_images.npy")) as f:
        for i in tqdm.tqdm(desc=f'Save semantic_images'):
            np.save(f, semantic_images[i])


def load_saved(opt, poses, images, semantic_images):
    dir_name = get_name_dir(opt)
    path_saved = os.path.join(opt.path, dir_name)
    print(f"[INFO] load images from {path_saved}")
    with open(os.path.join(path_saved, "poses.npy"), "rb") as f:
        while True:
            try:
                curr_arr = np.load(f, allow_pickle=True)
                poses.append(curr_arr)
            except:
                break

    with open(os.path.join(path_saved, "images.npy"), "rb") as f:
        while True:
            try:
                curr_arr = np.load(f, allow_pickle=True)
                images.append(curr_arr)
            except:
                break

    with open(os.path.join(path_saved, "semantic_images.npy"), "rb") as f:
        while True:
            try:
                curr_arr = np.load(f, allow_pickle=True)
                semantic_images.append(curr_arr)
            except:
                break
        

def apply_sparse(opt, frames, use_seed=True):
    if not opt.sparse_views:
        return frames

    if use_seed:
        np.random.seed(opt.seed)

    N_train = len(frames)
    
    K = int(N_train * opt.sparse_ratio)  # number of skipped training frames, mask=0

    N = N_train - K  # number of used training frames,  mask=1
    if K == 0: # in case sparse_ratio==0:
        return frames
    return np.random.choice(frames, N, replace=False)


def apply_pixel_denoise(opt, poses, images, semantic_images):
    pass

def apply_region_denoise(opt, poses, images, semantic_images):
    pass

def apply_super_resolution(opt, poses, images, semantic_images):
    pass

def apply_label_propagation(opt, poses, images, semantic_images):
    pass

