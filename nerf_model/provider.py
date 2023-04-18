import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, tm
from .semantic_utils import SemanticRemap
from sklearn.model_selection import train_test_split

from .spoil_dataset import (
    apply_sparse,
    load_spoiled,
    save_spoiled,
    apply_pixel_denoise,
    apply_region_denoise,
    apply_super_resolution,
    apply_label_propagation,
)


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=100, train_val_indexer=None, semantic_remap=None):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1
        self.train_val_indexer = train_val_indexer

        self.rand_pose = opt.rand_pose

        if semantic_remap is None:
            self.semantic_remap = SemanticRemap()
        else:
            self.semantic_remap = semantic_remap
            self.num_semantic_classes = len(self.semantic_remap.semantic_classes)

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # # choose two random poses, and interpolate between.
            # f0, f1 = np.random.choice(frames, 2, replace=False)
            # pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            # pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            # rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            # slerp = Slerp([0, 1], rots)

            # self.poses = []
            # self.images = None
            # self.semantic_images = None
            # for i in range(n_test + 1):
            #     ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
            #     pose = np.eye(4, dtype=np.float32)
            #     pose[:3, :3] = slerp(ratio).as_matrix()
            #     pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
            #     self.poses.append(pose)

            # the new way of making test video
            self.poses = []
            self.images = None
            self.semantic_images = None
            pose0 = np.array([
                [1, 0, 0, 0], 
                [0, 0, 1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]
            ], dtype=np.float32)
            pose0[:3, 3] = np.mean([np.array(f['transform_matrix'])[:3, 3] for f in frames], axis=0)
            pose0 = nerf_matrix_to_ngp(pose0, scale=self.scale, offset=self.offset) # [4, 4]
            poses_per_axis = n_test
            step = np.pi * 2 / poses_per_axis
            # # rotate over z
            # for i in range(poses_per_axis):
            #     curr_rot = Rotation.from_euler('xyz', [step * i, 0, 0]).as_matrix()
            #     curr_pose = pose0.copy()
            #     curr_pose[:3, :3] = curr_pose[:3, :3].copy() @ curr_rot
            #     self.poses.append(curr_pose)
            
            # rotate over y
            for i in range(poses_per_axis):
                curr_rot = Rotation.from_euler('xyz', [0, step * i, 0]).as_matrix()
                curr_pose = pose0.copy()
                curr_pose[:3, :3] = curr_pose[:3, :3].copy() @ curr_rot
                self.poses.append(curr_pose)

            # # rotate over z
            # for i in range(poses_per_axis):
            #     curr_rot = Rotation.from_euler('xyz', [0, 0, step * i]).as_matrix()
            #     curr_pose = pose0.copy()
            #     curr_pose[:3, :3] = curr_pose[:3, :3].copy() @ curr_rot
            #     self.poses.append(curr_pose)
            
        elif self.mode == 'colmap' and type == 'train' and opt.load_saved:
            self.poses = []
            self.images = []
            self.semantic_images = []
            self.depths = []
            load_spoiled(opt, self.poses, self.images, self.semantic_images, self.depths)
        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':

                    if np.isclose(opt.eval_ratio, 0.0):
                        # val_idx = [
                        #     492, 141, 409,  31, 570, 593, 873, 399, 406, 272, 691,  70, 312,
                        #     642, 500, 345, 351, 643, 772, 854,  14, 759, 692, 781, 526, 103,
                        #     158, 721, 458, 549, 150, 567, 717, 403, 656, 866, 362, 389, 830,
                        #     453, 676, 231,  97, 491, 620, 548, 204,  55,  65, 736, 635, 196,
                        #     308, 294, 701, 278,  77, 892, 386, 596, 503, 258, 299, 773, 142,
                        #     624, 523, 884, 775, 475, 842, 622,   8, 771, 380, 553, 862, 144,
                        #     145, 311, 390, 735, 542,  34, 794, 482, 895, 506,  27, 320
                        # ]
                        # train_idx = [0, 60, 120, 200, 305, 552, 764, 899]
                        train_idx, val_idx = np.arange(len(frames)), []
                    else:
                        train_idx, val_idx = train_test_split(np.arange(len(frames)), test_size=opt.eval_ratio, random_state=opt.seed)

                    self.train_val_indexer = {
                        "train_idx": train_idx,
                        "val_idx": val_idx
                    }
                
                    frames = apply_sparse(opt, [frames[i] for i in self.train_val_indexer['train_idx']])
                elif type == 'val':
                    frames = [frames[i] for i in self.train_val_indexer['val_idx']]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            self.semantic_images = []
            self.depths = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                semantic_path = os.path.join(self.root_path, f['semantic_path'])
                depth_path = os.path.join(self.root_path, f['depth_path']) if 'depth_path' in f else None
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path) or not os.path.exists(semantic_path):
                    continue

                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                ### image read
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                ### semantic read
                semantic = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
                if semantic.shape[0] != self.H or semantic.shape[1] != self.W:
                    semantic = cv2.resize(semantic, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

                ### depth read
                if not depth_path is None:
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
                    if depth.shape[0] != self.H or depth.shape[1] != self.W:
                        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                    self.depths.append(depth)

                self.poses.append(pose)
                self.images.append(image)
                self.semantic_images.append(semantic)

        ### spoil dataset
        if type == 'train' and not opt.load_saved:
            if opt.pixel_denoising:
                apply_pixel_denoise(opt, self.poses, self.images, self.semantic_images)
            if opt.region_denoising:
                apply_region_denoise(opt, self.poses, self.images, self.semantic_images)
            if opt.super_resolution:
                apply_super_resolution(opt, self.poses, self.images, self.semantic_images)
            if opt.label_propagation:
                apply_label_propagation(opt, self.poses, self.images, self.semantic_images)

            if opt.visualise_save:
                save_spoiled(opt, self.poses, self.images, self.semantic_images)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.semantic_images = torch.from_numpy(np.stack(self.semantic_images, axis=0)) # [N, H, W]
        
        # make semantic remap
        if self.semantic_images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
            self.semantic_remap.remap(self.semantic_images, inplace=True)
            self.num_semantic_classes = len(self.semantic_remap.semantic_classes)
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())
        # visualize_poses(self.poses.numpy()[:100:5])

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())
        

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
                self.semantic_images = self.semantic_images.to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1] # 3/4
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images

            semantic_images = self.semantic_images[index].to(self.device)
            if self.training:
                semantic_images = torch.gather(semantic_images.view(B, -1), 1, rays['inds']) # [B, N]
            results['semantic_images'] = semantic_images

        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader