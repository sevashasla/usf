from tqdm import tqdm
import os
import sys
from copy import deepcopy
import json
import yaml
import argparse
import wandb
from functools import partial

class NeRFRunner:
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

    def run(self, gpus):
        # hardcoded, choose the best gpu
        if 0 in gpus:
            gpu = 0 # 2080 Ti
        elif 2 in gpus:
            gpu = 2 # 2080 Ti
        else:
            gpu = 1 # not a 2080 Ti

        path_from = os.path.dirname(self.start)
        print(f"[INFO] Starting {self.start} on {gpu}!")
        # print(f"Wanna run: python3 {self.start} {self.launch}")
        result = os.system(f'cd {path_from} && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="{gpu}," python3 {self.start} {self.launch}')
        # result = os.system(f'sleep {1 + idx}')
        if result == 0:
            print(f"[INFO] END {self.start} {self.launch}") 
        else:
            print(f"[ERROR] FAILED {self.start} {self.launch}")
        return result

    @classmethod
    def prepare_config(cls, config, **kwargs):
        config["workspace"] = os.path.join(cls.store_result, kwargs["sweep_id"])
        config["datapath"] = kwargs['datapath']
        config['project'] = kwargs['project']
        config["group"] = kwargs['group']

    def prepare_launch(self):
        other_params = deepcopy(self.params)
        other_params.pop("datapath")
        semantic_remap = other_params.pop('semantic_remap', None)
        true_bool_params = [k for k, v in other_params.items() if isinstance(v, bool) and v]
        other_params = [(k, v) for k, v in other_params.items() if not isinstance(v, bool)]

        
        result = f"{self.params['datapath']} "
        result = result + " ".join(map(lambda pair: f"--{pair[0]}={pair[1]}", other_params)) + " "
        result = result + " ".join(map(lambda k: f"--{k}", true_bool_params)) + " "
        if semantic_remap:
            result = result + f"--semantic_remap='{semantic_remap}'"
        return result


class SemanticNgpRunner(NeRFRunner):
    default_params = {
        "fp16": True,
        "scale": 0.5, 
        "bound": 3,
        "epochs": 1500,
        "max_ray_batch": 2048,
        "num_rays": 1024,
        "num_steps": 512,
        "eval_interval": 50,
        "video_interval": 100,
        "save_interval": 50
    }
    start = None
    store_result = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = self.change_default(self.default_params, self.config)
        self.launch = self.prepare_launch()


def create_transforms(python_script, path, w, h):
    '''
    transforms traj_file
    '''
    out_file = os.path.join(path, "transforms.json")
    traj_file = os.path.join(path, "traj_w_c.txt")
    os.system(f"python3 {python_script} --traj_file={traj_file} --out_file={out_file} -W={w} -H={h}")


def make_split(python_script, datapath, train_ratio, eval_ratio, holdout_ratio, test_ratio):
    '''
    make split
    '''
    transform_file = os.path.join(datapath, "transforms.json")
    os.system(f"python3 {python_script} --transform_file={transform_file} --train_ratio={train_ratio} --eval_ratio={eval_ratio} --holdout_ratio={holdout_ratio} --test_ratio={test_ratio}")


def create_and_run(other_parameters, datapath, project, gpus, group):
    wandb.init()
    params = wandb.config
    run_id = wandb.run.id
    wandb.finish()

    config = deepcopy(other_parameters)
    for k, v in params.items():
        config[k] = v # rewrite parameters!

    SemanticNgpRunner.prepare_config(
        config, sweep_id=run_id, datapath=datapath, project=project, group=group
    )
    
    runner = SemanticNgpRunner(config)
    runner.run(gpus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./sweep.yaml", help="path to yaml file")
    args = parser.parse_args()

    with open(args.path) as f:
        exp_config = yaml.safe_load(f)
    
    project  = exp_config['project']
    dataset_dir = exp_config['dataset_dir']
    group = exp_config['name']
    convert_script_path = exp_config["convert_script_path"]
    split_script_path = exp_config["split_script_path"]
    gpus = exp_config["gpu"]
    if not isinstance(gpus, list):
        gpus = [gpus]
    
    # init all classes:
    SemanticNgpRunner.start = exp_config["start"]["semantic_ngp"]
    SemanticNgpRunner.store_result = os.path.join(exp_config["store_result"]["semantic_ngp"], group)
    
    # prepare the dataset
    pdconfig = exp_config["prepare_dataset_config"]
    w, h = pdconfig["w"], pdconfig["h"]
    need_transforms = pdconfig.get("need_transforms", True)
    place = pdconfig["place"]
    sequence = pdconfig["sequence"]
    datapath = os.path.join(dataset_dir, place, sequence)
    if need_transforms:
        create_transforms(convert_script_path, datapath, w=w, h=h)
        make_split(
            split_script_path, datapath, 
            pdconfig["train_ratio"], pdconfig["eval_ratio"], 
            pdconfig["holdout_ratio"], pdconfig["test_ratio"]
        )

    # start the sweep
    sweep_config = exp_config["sweep_config"]
    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project=project,
    )
    wandb.agent(
        sweep_id, 
        function=partial(
            create_and_run, 
            other_parameters=exp_config["other_parameters"],
            datapath=datapath,
            project=project,
            gpus=gpus, group=group,
        ), 
        count=exp_config["count"],
    )

if __name__ == "__main__":
    main()
