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
        if result == 0:
            print(f"[INFO] END {self.start} {self.launch}") 
        else:
            print(f"[ERROR] FAILED {self.start} {self.launch}")
        return result

    @classmethod
    def prepare_config(cls, config, **kwargs):
        workspace = config.pop("name", f"{kwargs['place']}_{kwargs['sequence']}_{kwargs['w']}_{kwargs['h']}_{kwargs['i']}")
        config["workspace"] = f"{cls.store_result}/{workspace}"
        config["datapath"] = kwargs['datapath']
        config["group"] = kwargs['group']
        config.setdefault("eval_interval", config["epochs"] // 5)

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


class TorchNgpRunner(NeRFRunner):
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
    gpus = exp_configs["gpu"]
    if not isinstance(gpus, list):
        gpus = [gpus]

    # init all classes:
    SemanticNgpRunner.start = exp_configs["start"]["semantic_ngp"]
    TorchNgpRunner.start = exp_configs["start"]["torch_ngp"]

    SemanticNgpRunner.store_result = os.path.join(exp_configs["store_result"]["semantic_ngp"], group)
    TorchNgpRunner.store_result = os.path.join(exp_configs["store_result"]["torch_ngp"], group)

    
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
        for i, torch_ngp_exp_config in enumerate(exp.get("torch_ngp", [])):
            config = deepcopy(torch_ngp_exp_config)
            if config is None:
                continue
            TorchNgpRunner.prepare_config(
                config, 
                datapath=datapath,
                place=place, sequence=sequence,
                w=w, h=h, group=group, i=i,
            )
            if TorchNgpRunner.already_exists(config) and not config.pop("do_run", False): # also delete do_run!
                continue
            runners.append(TorchNgpRunner(config))

        for i, semantic_ngp_exp_config in enumerate(exp.get("semantic_ngp", [])):
            config = deepcopy(semantic_ngp_exp_config)
            if config is None:
                continue
            SemanticNgpRunner.prepare_config(
                config, 
                datapath=datapath,
                place=place, sequence=sequence,
                w=w, h=h, scene_file=scene_file,
                group=group, i=i,
            )
            if SemanticNgpRunner.already_exists(config) and not config.pop("do_run", False): # also delete do_run!
                continue
            runners.append(SemanticNgpRunner(config))

        # start multithreading here
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
