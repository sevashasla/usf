from tqdm import tqdm
import os
import sys
from copy import deepcopy
import json
import yaml
import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import current_thread

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
        # lists are thread-safe!
        # start of some multithreading here
        curr_thread = current_thread()
        idx = int(curr_thread.name.split("_")[-1])
        gpu = gpus[idx]
        # end of some multithreading here
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
        # add common
        for k, v in kwargs["common"].items():
            config[k] = v

        # other preparations
        workspace = config.pop("name", f"{kwargs['place']}_{kwargs['sequence']}_{kwargs['w']}_{kwargs['h']}_{kwargs['i']}")
        config["workspace"] = f"{cls.store_result}/{workspace}"
        config["datapath"] = kwargs['datapath']
        config["group"] = kwargs['group']
        config['project'] = kwargs['project']

    def prepare_launch(self):
        other_params = deepcopy(self.params)
        other_params.pop("datapath")
        true_bool_params = [k for k, v in other_params.items() if isinstance(v, bool) and v]
        other_params = [(k, v) for k, v in other_params.items() if not isinstance(v, bool)]

        result = f"{self.params['datapath']} "
        result = result + " ".join(map(lambda pair: f"--{pair[0]}={pair[1]}", other_params)) + " "
        result = result + " ".join(map(lambda k: f"--{k}", true_bool_params)) + " "
        return result
    
    @staticmethod
    def already_exists(config):
        return os.path.exists(config["workspace"])


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


class TorchNgpRunner(NeRFRunner):
    default_params = {
        "fp16": True,
        "scale": 1.0, 
        "bound": 8.0,
        "epochs": 1500,
        "max_ray_batch": 4096,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./manytest.yaml", help="path to yaml file")
    args = parser.parse_args()

    with open(args.path) as f:
        exp_configs = yaml.safe_load(f)

    project  = exp_configs['project']
    dataset_dir = exp_configs['dataset_dir']
    group = exp_configs['name']
    continue_on_fail = exp_configs['continue_on_fail']
    convert_script_path = exp_configs["convert_script_path"]
    split_script_path = exp_configs["split_script_path"]
    gpus = exp_configs["gpu"]
    if not isinstance(gpus, list):
        gpus = [gpus]

    # init all classes:
    SemanticNgpRunner.start = exp_configs["start"]["semantic_ngp"]
    TorchNgpRunner.start = exp_configs["start"]["torch_ngp"]

    SemanticNgpRunner.store_result = os.path.join(exp_configs["store_result"]["semantic_ngp"], group)
    TorchNgpRunner.store_result = os.path.join(exp_configs["store_result"]["torch_ngp"], group)

    runners = []
    for exp in exp_configs["experiments"]:
        # prepare the experiment
        w, h = exp["w"], exp["h"]
        need_transforms = exp.get("need_transforms", True)
        place = exp["place"]
        sequence = exp["sequence"]
        common = exp.get("common", {}) # common features
        scene_file = os.path.join(exp_configs["scene_dir"], place)
        datapath = os.path.join(dataset_dir, place, sequence)
        if need_transforms:
            create_transforms(convert_script_path, datapath, w=w, h=h)
            make_split(split_script_path, datapath, exp["train_ratio"], exp["eval_ratio"], exp["holdout_ratio"], exp["test_ratio"])

        for i, torch_ngp_exp_config in enumerate(exp.get("torch_ngp", [])):
            config = deepcopy(torch_ngp_exp_config)
            if config is None:
                continue
            TorchNgpRunner.prepare_config(
                config, 
                common=common, 
                datapath=datapath,
                place=place, sequence=sequence,
                w=w, h=h, group=group, i=i, project=project,
            )
            if TorchNgpRunner.already_exists(config) and not config.pop("do_run", False): # also delete do_run!
                continue
            config.pop("do_run", True)
            runners.append(TorchNgpRunner(config))

        for i, semantic_ngp_exp_config in enumerate(exp.get("semantic_ngp", [])):
            config = deepcopy(semantic_ngp_exp_config)
            if config is None:
                continue
            SemanticNgpRunner.prepare_config(
                config, 
                common=common, 
                datapath=datapath,
                place=place, sequence=sequence,
                w=w, h=h, scene_file=scene_file,
                group=group, i=i, project=project,
            )
            if SemanticNgpRunner.already_exists(config) and not config.pop("do_run", False): # also delete do_run!
                continue
            config.pop("do_run", True)
            runners.append(SemanticNgpRunner(config))

    # start multithreading here
    print(f"[INFO] Total: {len(runners)} runs")
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        all_result_runs = [] # features
        for r in runners:
            all_result_runs.append(executor.submit(r.run, gpus))
        for result_run in all_result_runs:
            if result_run.result() != 0 and not continue_on_fail:
                return
    # end multithreading here

if __name__ == "__main__":
    main()
