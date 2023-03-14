'''
transform data from replica dataset to ngp format
'''

import numpy as np
import argparse
import os
import math
import json
from tqdm import trange


class Replica2NGP:
    def __init__(self, args):
        self.traj_file = args.traj_file
        self.out_file = args.out_file
        self.args = args

    def __find_instrincts(self):
        pass

    # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    @staticmethod
    def closest_point_2_lines(oa, da, ob, db): 
        da = da / np.linalg.norm(da)
        db = db / np.linalg.norm(db)
        c = np.cross(da, db)
        denom = np.linalg.norm(c)**2
        t = ob - oa
        ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
        tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
        if ta > 0:
            ta = 0
        if tb > 0:
            tb = 0
        return (oa+ta*da+ob+tb*db) * 0.5, denom

    @staticmethod
    def rotmat(a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        # handle exception for the opposite direction input
        if c < -1 + 1e-10:
            return Replica2NGP.rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

    @staticmethod
    def change_transform_matrix(poses, inplace=False):
        if not inplace:
            poses = poses.copy()
        N = poses.shape[0]
        poses[:, 0:3, 1] *= -1
        poses[:, 0:3, 2] *= -1
        poses = poses[:, [1, 0, 2, 3], :] # swap y and z
        poses[:, 2, :] *= -1 # flip whole world upside down

        up = poses[:, 0:3, 1].sum(0)
        up = up / np.linalg.norm(up)
        R = Replica2NGP.rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        poses = R @ poses

        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for i in trange(N):
            mf = poses[i, :3, :]
            for j in range(i + 1, N):
                mg = poses[j, :3, :]
                p, w = Replica2NGP.closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                #print(i, j, p, w)
                if w > 0.01:
                    totp += p * w
                    totw += w
        totp /= totw
        poses[:, :3, 3] -= totp
        avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
        poses[:, :3, 3] *= 4.0 / avglen
        return poses

    def __set_params_replica(self):
        self.H = self.args.H
        self.W = self.args.W

        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W/self.H

        self.hfov = 90
        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
        self.fy = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0
    

    def transform(self):
        self.__set_params_replica()
        result = {}
        result["fl_x"] = self.fx
        result["fl_y"] = self.fy
        result["cx"] = self.cx
        result["cy"] = self.cy
        result["w"] = self.W
        result["h"] = self.H
        result["aabb_scale"] = 2
        result["frames"] = []

        poses = []
        with open(self.traj_file) as f:
            while s := f.readline():
                pose = np.array([float(x) for x in s.split()]).reshape(4, 4)
                poses.append(pose)
        
        poses = np.array(poses)
        self.change_transform_matrix(poses, inplace=True)
        print("[INFO] poses are prepared!")

        for i in range(len(poses)):
            frames_item = {}
            frames_item["file_path"] = f"rgb/rgb_{i}.png"
            frames_item["semantic_path"] = f"semantic_class/semantic_class_{i}.png"
            if self.args.depth:
                frames_item["depth_path"] = f"depth/depth_{i}.png"
            frames_item["transform_matrix"] = poses[i].tolist()
            result["frames"].append(frames_item)
            
        with open(self.out_file, "w") as f:
            f.write(json.dumps(result, indent=4))
        
        print(f"[INFO] transforms.json at: {self.out_file}!")


def main():
    # scale factor is used for evaluation!
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_file", type=str, help='path to traj_w_c.txt')
    parser.add_argument("--out_file", type=str, help='path to output file')
    parser.add_argument("-W", type=int, default=160, help='width')
    parser.add_argument("-H", type=int, default=120, help='height')
    # parser.add_argument("--config_file", type=str, help='path to config file')
    parser.add_argument("--depth", action='store_true', help='do one need to use provide depth')
    args = parser.parse_args()
    print("[INFO] replica2nerf.py with parameters:")
    print(args)
    r2n = Replica2NGP(args)
    r2n.transform()

main()
