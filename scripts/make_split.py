'''
make transforms_train.json and etc.
'''

import numpy as np
import argparse
import os
import math
import json
from tqdm import trange
from copy import deepcopy


class Splitter():
    def __init__(self, args):
        self.args = args
        self.train_ratio = args.train_ratio
        self.eval_ratio = args.eval_ratio
        self.holdout_ratio = args.holdout_ratio
        self.test_ratio = args.test_ratio
        self.transform_file = args.transform_file

    def _split(self, n, random_state=42):
        # count sizes
        maybe_ratio = lambda x: int(x) if x > 1.0 else int(x * n)
        train_size = maybe_ratio(self.train_ratio)
        val_size = maybe_ratio(self.eval_ratio)
        holdout_size = maybe_ratio(self.holdout_ratio)
        test_size = maybe_ratio(self.test_ratio)

        # generate indices
        np.random.seed(random_state)
        all_ids = np.random.permutation(n)
        train_ids = all_ids[:train_size]
        val_ids = all_ids[train_size:train_size + val_size]
        holdout_ids = all_ids[train_size + val_size:train_size + val_size + holdout_size]
        test_ids = all_ids[-test_size - 1:] # take them from the end
        return train_ids, val_ids, holdout_ids, test_ids

    def make_split(self):
        with open(self.transform_file, "r") as f:
            transforms = json.load(f)
        not_frames = {}
        frames = transforms["frames"]
        for k, v in transforms.items():
            if k != "frames":
                not_frames[k] = v
        train_ids, val_ids, holdout_ids, test_ids = self._split(len(transforms["frames"]))
        transforms_dict = {
            "train": [frames[i] for i in train_ids],
            "val": [frames[i] for i in val_ids],
            "test": [frames[i] for i in test_ids],
            "holdout": [frames[i] for i in holdout_ids],
        }

        # save new file
        dir_to_tr = os.path.dirname(self.transform_file)
        for mode in transforms_dict:
            mode_file = deepcopy(not_frames)
            mode_file["frames"] = transforms_dict[mode]
            to_save_path = os.path.join(dir_to_tr, f"transforms_{mode}.json")
            with open(to_save_path, "w") as f:
                f.write(json.dumps(mode_file, indent=4))

        # rename old file
        os.rename(self.transform_file, os.path.join(dir_to_tr, "old_transforms.json"))

    def __call__(self):
        self.make_split()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform_file", type=str, help='path to transform.json')
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--eval_ratio", type=float)
    parser.add_argument("--holdout_ratio", type=float)
    parser.add_argument("--test_ratio", type=float)
    args = parser.parse_args()
    print("[INFO] split with parameters:")
    print(args)
    splitter = Splitter(args)
    splitter.make_split()

main()
