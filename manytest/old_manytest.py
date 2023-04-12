from tqdm import tqdm
import os
import sys
from copy import deepcopy
import json
import yaml

default_ssr = {
    "experiment": {
        "scene_file": "",
        "save_dir": "",
        "dataset_dir": "",
        "convention": "opencv",
        "width": 160,
        "height": 120,
        "gpu": "0",
        "enable_semantic": True,
        "enable_depth": True,
        "endpoint_feat": False,
        "eval_ratio": 0.1,
    },
    "model": {
        "netdepth": 8,
        "netwidth": 256,
        "netdepth_fine": 8,
        "netwidth_fine": 256,
        # "chunk": 1024,
        # "netchunk": 1024
        "chunk": 4096,
        "netchunk": 4096,
    },
    "render": {
        "N_rays": 512,
        "N_samples": 32,
        "N_importance": 64,
        "perturb": 1,
        "use_viewdirs": True,
        "i_embed": 0,
        "multires": 10,
        "multires_views": 4,
        "raw_noise_std": 1,
        "test_viz_factor": 1,
        "no_batching": True,
        "depth_range": [
            0.1,
            10
        ],
        "white_bkgd": False
    },
    "train": {
        "lrate": 0.0005,
        "lrate_decay": 250000,
        "N_iters": 1000,
        "wgt_sem": 0.04
    },
    "logging": {
        "step_log_print": 10,
        "step_log_tfb": 199,
        "step_save_ckpt": 199,
        "step_val": 199,
        "step_vis_train": 199
    }
}

# PLACE_NAMES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
# PLACE_NAMES = ["office_1", "office_2", "office_3", "office_4"]
PLACE_NAMES = ["room_1"]
SEQUENCES = ["Sequence_1",]
DATASET_ROOT = "/mnt/hdd8/Datasets/Replica/"
CURR_PWD = os.getcwd()
GROUP = "exp-15-04-2023"

# Results path
SEMANTIC_NGP_RES =      "/mnt/hdd8/skorokhodov_vs/results/semantic_ngp_results/"
TORCH_NGP_RES =         "/mnt/hdd8/skorokhodov_vs/results/torch-ngp_results"
SEMANTIC_NERF_RES =     "/mnt/hdd8/skorokhodov_vs/results/semantic_nerf_results"

# Convert script path
CONVERT_SCRIPT_PATH =   "/home/skorokhodov_vs/nerf/replica2nerf.py"

# Files to run
SEMANTIC_NGP_START =    "/home/skorokhodov_vs/nerf/from_laba/ngp_with_semantic_nerf/main_nerf.py"
TORCH_NGP_START =       "/home/skorokhodov_vs/nerf/from_laba/torch-ngp/main_nerf.py"
SEMANTIC_NERF_START =   "/home/skorokhodov_vs/nerf/from_laba/semantic_nerf/train_SSR_main_time_measure.py"

class Runner():
    def __init__(self, args, mode, semantic_nerf_args=None):
        '''
        mode: ['ngp_with_semantic_nerf', 'torch_ngp', 'semantic_nerf']
        args: arguments to run
        '''
        self.mode = mode
        self.args = args
        self.semantic_nerf_args = semantic_nerf_args

    def __run(self, to_run):
        path_from = os.path.dirname(to_run)
        print(f"[INFO] Starting {to_run} {self.args}")
        result = os.system(f"cd {path_from} && python3 {to_run} {self.args}")
        if result == 0:
            print(f"[INFO] End {to_run} {self.args}")
        else:
            print(f"[WARN] FAILED {to_run} {self.args}")

    def __call__(self):
        self.run()

    def run(self):
        if self.mode == "ngp_with_semantic_nerf":
            self.__run(
                to_run=SEMANTIC_NGP_START,
            )
            # delete
            # os.system(f"python3 my_clear_ngp.py --path={SEMANTIC_NGP_RES}")
        
        elif self.mode == "torch_ngp":
            self.__run(
                to_run=TORCH_NGP_START,
            )
            # delete
            # os.system(f"python3 my_clear_ngp.py --path={TORCH_NGP_RES}")
        
        elif self.mode == "semantic_nerf":
            os.makedirs("tmp", exist_ok=True)
            semantic_config = deepcopy(default_ssr)
            for k, v in self.semantic_nerf_args.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        semantic_config[k][kk] = vv
                else:
                    semantic_config[k] = v
            with open("tmp/config.yaml", "w") as f:
                yaml.dump(semantic_config, f)
            
            config = os.path.join(CURR_PWD, "tmp/config.yaml")
            self.args = self.args + f" --config_file={config}"
            self.__run(
                to_run=SEMANTIC_NERF_START
            )
            os.remove(config)
        else:
            print(f"[WARN] mode {self.mode} not in ['ngp_with_semantic_nerf', 'torch_ngp', 'semantic_nerf']")


def create_transforms(python_script, path, w, h):
    out_file = os.path.join(path, "transforms.json")
    traj_file = os.path.join(path, "traj_w_c.txt")
    os.system(f"python3 {python_script} --traj_file={traj_file} --out_file={out_file} -W={w} -H={h}")


def main():
    for w, h in [
            (320, 240), 
            (480, 360), 
            (640, 480)
        ]:
        for place in PLACE_NAMES:
            for seq in SEQUENCES:
                datapath = os.path.join(DATASET_ROOT, place, seq)
                # create_transforms(CONVERT_SCRIPT_PATH, datapath, w=w, h=h)

                scene_file = os.path.join(DATASET_ROOT, "semantic_info", place)

                runners = [
                    ### time complexity + quality
                    # Runner(
                    #     f"{datapath} --group={GROUP} --workspace={TORCH_NGP_RES}/{place}_{seq}_{w}_{h}_to_compare_tc --fp16 --scale=0.5 --bound=3 --epochs=100 --max_ray_batch=4096 --num_rays=512 --num_steps=96 --eval_interval=100 --eval_ratio=0.1", 
                    #     mode="torch_ngp"
                    # ),

                    # Runner(
                    #     f"{datapath} --group={GROUP} --workspace={TORCH_NGP_RES}/{place}_{seq}_{w}_{h}_to_compare_tc_cuda --fp16 --scale=0.5 --bound=3 --epochs=800 --max_ray_batch=4096 --num_rays=512 --max_steps=96 --cuda_ray --eval_interval=100 --eval_ratio=0.1", 
                    #     mode="torch_ngp"
                    # ),
                    
                    Runner(
                        f"{datapath} --group={GROUP} --workspace={SEMANTIC_NGP_RES}/{place}_{seq}_{w}_{h}_to_compare_tc_lambd=0.1 --fp16 --scale=0.5 --bound=3 --epochs=500 --max_ray_batch=4096 --num_rays=512 --num_steps=96 --eval_interval=100 --eval_ratio=0.1 --lambd=0.1", 
                        mode="ngp_with_semantic_nerf"
                    ),

                    # Runner(
                    #     f"{datapath} --group={GROUP} --workspace={SEMANTIC_NGP_RES}/{place}_{seq}_{w}_{h}_to_compare_tc_lambd=0.01 --fp16 --scale=0.5 --bound=3 --epochs=800 --max_ray_batch=4096 --num_rays=512 --num_steps=96 --eval_interval=100 --eval_ratio=0.1 --lambd=0.01", 
                    #     mode="ngp_with_semantic_nerf"
                    # ),


                    # Runner(
                    #     f"--group={GROUP} ", 
                    #     mode="semantic_nerf", 
                    #     semantic_nerf_args={
                    #         "experiment": {
                    #             "save_dir": f"{SEMANTIC_NERF_RES}/{place}_{seq}_{w}_{h}",
                    #             "eval_ratio": 0.1,
                    #             "scene_file": scene_file,
                    #             "dataset_dir": datapath,
                    #             "width": w,
                    #             "height": h,
                    #         },
                    #         "train": {
                    #             "N_iters": 100000,
                    #         },
                    #         "logging": {
                    #             "step_log_print": 10000,
                    #             "step_log_tfb": 1000000,
                    #             "step_save_ckpt": 49999,
                    #             "step_val": 20000,
                    #             "step_vis_train": 1000000
                    #         },
                    #         "render": {
                    #             "N_importance": 32 if w == 640 else 64 # костыль!!!
                    #         }
                    #     }
                    # ),
                    
                    # # to compare quality on small train dataset
                    # Runner(
                    #     f"{datapath} --workspace=results/to_eval_{place}_{seq}_small_dataset_lambd=0.1 --fp16 --scale=0.5 --bound=3 --epochs=100 --max_ray_batch=4096 --num_rays=1024 --eval_interval=20 --eval_ratio=0.9 --lambd=0.1", 
                    #     mode="ngp_with_semantic_nerf"
                    # ),

                    # Runner(
                    #     f"{datapath} --workspace=results/to_eval_{place}_{seq}_small_dataset --fp16 --scale=0.5 --bound=3 --epochs=100 --max_ray_batch=4096 --num_rays=1024 --eval_interval=20 --eval_ratio=0.9", 
                    #     mode="torch_ngp"
                    # ),

                    # Runner("", 
                    #     mode="semantic_nerf", 
                    #     semantic_nerf_args={
                    #         "experiment": {
                    #             "save_dir": f"results/{place}_{seq}_small_dataset",
                    #             "eval_ratio": 0.9,
                    #             "scene_file": scene_file,
                    #             "dataset_dir": datapath,
                    #         },
                    #         "train": {
                    #             "N_iters": 15000,
                    #         },
                    #         "logging": {
                    #             "step_log_print": 1000,
                    #             "step_log_tfb": 100000,
                    #             "step_save_ckpt": 14999,
                    #             "step_val": 7499,
                    #             "step_vis_train": 100000
                    #         },
                    #     }
                    # ),
                ]

                for r in runners:
                    r.run()

main()
