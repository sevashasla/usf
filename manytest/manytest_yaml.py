from tqdm import tqdm
import os
import sys
from copy import deepcopy
import json
import yaml
import argparse

class NgpRunner:
    def __init__(self):
        pass

    def change_default(self, default_params, config):
        params = deepcopy(default_params)
        for k, v in config.items():
            if isinstance(v, dict):
                # create array
                if not k in params:
                    params[k] = {}
                for kk, vv in v.items():
                    params[k][kk] = vv
            else:
                params[k] = v
        return params

    def run(self, gpu):
        path_from = os.path.dirname(self.start)
        print(f"[INFO] Starting {self.start}")
        # print(f"Wanna run: python3 {self.start} {self.launch}")
        result = os.system(f'cd {path_from} && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="{gpu}," python3 {self.start} {self.launch}')
        if result == 0:
            print(f"[INFO] END {self.start} {self.launch}") 
        else:
            print(f"[ERROR] FAILED {self.start} {self.launch}")
        return result

    @classmethod
    def prepare_config(cls, config, **kwargs):
        config["workspace"] = f"{cls.store_result}/{kwargs['place']}_{kwargs['sequence']}_{kwargs['w']}_{kwargs['h']}_{kwargs['i']}"
        config["datapath"] = kwargs['datapath']
        config["W"] = kwargs['w']
        config["H"] = kwargs['h']
        config["group"] = kwargs['group']
        config.setdefault("eval_interval", config["epochs"] // 10)

    def prepare_launch(self):
        other_params = deepcopy(self.params)
        other_params.pop("datapath")
        true_bool_params = [k for k, v in other_params.items() if isinstance(v, bool) and v]
        other_params = [(k, v) for k, v in other_params.items() if not isinstance(v, bool)]

        result = f"{self.params['datapath']} "
        result = result + " ".join(map(lambda pair: f"--{pair[0]}={pair[1]}", other_params)) + " "
        result = result + " ".join(map(lambda k: f"--{k}", true_bool_params)) + " "
        return result


class SemanticNgpRunner(NgpRunner):
    default_params = {
        "fp16": True,
        "scale": 0.5, 
        "bound": 3,
        "epochs": 100,
        "max_ray_batch": 4096,
        "num_rays": 512,
        "num_steps": 96,
        "eval_interval": 100,
        "eval_ratio": 0.1,
        "lambd": 0.01,
    }
    start = None
    store_result = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = self.change_default(self.default_params, self.config)
        self.launch = self.prepare_launch()


class TorchNgpRunner(NgpRunner):
    default_params = {
        "fp16": True,
        "scale": 0.5, 
        "bound": 3,
        "epochs": 100,
        "max_ray_batch": 4096,
        "num_rays": 512,
        "num_steps": 96,
        "eval_interval": 100,
        "eval_ratio": 0.1,
        "cuda_ray": False,
    }
    start = None
    store_result = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = self.change_default(self.default_params, self.config)
        self.launch = self.prepare_launch()


class SemanticNeRFRunner(NgpRunner):
    default_params = {
        "experiment": {
            "convention": "opencv",
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
            "step_log_print": 10000,
            "step_log_tfb": int(1e10),
            "step_vis_train": int(1e10),
        }
    }
    
    start = None
    store_result = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = self.change_default(self.default_params, self.config)
        self.launch = self.prepare_launch()

    @classmethod
    def prepare_config(cls, config, **kwargs):
        config.setdefault("experiment", {})
        config["experiment"]["save_dir"] = f"{cls.store_result}/{kwargs['place']}_{kwargs['sequence']}_{kwargs['w']}_{kwargs['h']}_{kwargs['i']}"
        config["experiment"]["scene_file"] = kwargs['scene_file']
        config["experiment"]["dataset_dir"] = kwargs['datapath']
        config["experiment"]["width"] = kwargs['w']
        config["experiment"]["height"] = kwargs['h']
        config["experiment"]["group"] = kwargs['group']
        config["experiment"]["gpu"] = kwargs['gpu']
        config.setdefault("logging", {})
        config["logging"].setdefault("step_save_ckpt", config["train"]["N_iters"] // 5)
        config["logging"].setdefault("step_val", config["train"]["N_iters"] // 5)
        config["logging"].setdefault("step_vis_train", config["train"]["N_iters"])
        
    def prepare_launch(self):
        os.makedirs("tmp", exist_ok=True)
        workspace_name = os.path.basename(self.config['experiment']['save_dir'])
        with open(f"tmp/{workspace_name}.yaml", "w") as f:
            yaml.dump(self.params, f)

        curr_pwd = os.getcwd()
        config = os.path.join(curr_pwd, "tmp/config.yaml")

        result = f" --config_file={config}"
        return result


def create_transforms(python_script, path, w, h):
    '''
    transforms traj_file
    '''
    out_file = os.path.join(path, "transforms.json")
    traj_file = os.path.join(path, "traj_w_c.txt")
    os.system(f"python3 {python_script} --traj_file={traj_file} --out_file={out_file} -W={w} -H={h}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./manytest.yaml", help="path to yaml file")
    args = parser.parse_args()

    with open(args.path) as f:
        exp_configs = yaml.safe_load(f)

    dataset_dir = exp_configs['dataset_dir']
    group = exp_configs['name']
    continue_on_fail = exp_configs['continue_on_fail']
    convert_script_path = exp_configs["convert_script_path"]
    gpu = exp_configs["gpu"]

    # init all classes:
    SemanticNgpRunner.start = exp_configs["start"]["semantic_ngp"]
    TorchNgpRunner.start = exp_configs["start"]["torch_ngp"]
    SemanticNeRFRunner.start = exp_configs["start"]["semantic_nerf"]

    SemanticNgpRunner.store_result = os.path.join(exp_configs["store_result"]["semantic_ngp"], group)
    TorchNgpRunner.store_result = os.path.join(exp_configs["store_result"]["torch_ngp"], group)
    SemanticNeRFRunner.store_result = os.path.join(exp_configs["store_result"]["semantic_nerf"], group)

    for exp in exp_configs["experiments"]:
        # prepare the experiment
        w, h = exp["w"], exp["h"]
        need_transforms = exp.get("need_transforms", True)
        place = exp["place"]
        sequence = exp["sequence"]
        scene_file = os.path.join(exp_configs["scene_dir"], place)
        datapath = os.path.join(dataset_dir, place, sequence)
        if need_transforms:
            create_transforms(convert_script_path, datapath, w=w, h=h)

        runners = []
        for i, torch_ngp_exp_config in enumerate(exp["torch_ngp"]):
            config = deepcopy(torch_ngp_exp_config)
            if config is None or not config.pop("do_run", True):
                continue
            TorchNgpRunner.prepare_config(
                config, 
                datapath=datapath,
                place=place, sequence=sequence,
                w=w, h=h, group=group, i=i,
            )

        for i, semantic_ngp_exp_config in enumerate(exp["semantic_ngp"]):
            config = deepcopy(semantic_ngp_exp_config)
            if config is None or not config.pop("do_run", True):
                continue
            SemanticNgpRunner.prepare_config(
                config, 
                datapath=datapath,
                place=place, sequence=sequence,
                w=w, h=h, scene_file=scene_file,
                group=group, i=i,
            )
            runners.append(SemanticNgpRunner(config))

        for i, semantic_nerf_exp_config in enumerate(exp["semantic_nerf"]):
            config = deepcopy(semantic_nerf_exp_config)
            if config is None or not config.pop("do_run", True):
                continue
            SemanticNeRFRunner.prepare_config(
                config, 
                datapath=datapath,
                place=place, sequence=sequence,
                gpu=str(gpu), 
                w=w, h=h, group=group, scene_file=scene_file,
                i=i,
            )
            runners.append(SemanticNeRFRunner(config))
        
        for r in runners:
            result_run = r.run(gpu)
            if result_run != 0 and not continue_on_fail:
                return

if __name__ == "__main__":
    main()
