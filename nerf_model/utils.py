import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import wandb

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics import ConfusionMatrix

from sklearn.metrics import confusion_matrix

from .time_measure import TimeMeasure
tm = TimeMeasure()
from imgviz import label_colormap

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def linear_transform(image: np.ndarray, lower=0.0, upper=255.0):
    image_min = image.min()
    image_max = image.max()

    convert_01 = (image - image_min) / (image_max - image_min + 1e-9)
    return (lower + convert_01 * (upper - lower)).astype(np.uint8)

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

class DirichletSemanticLoss(nn.Module):
    '''
    Implementation of dirichlet loss from paper
    https://arxiv.org/pdf/1806.01768.pdf
    '''
    def __init__(self):
        super().__init__()

    def forward(self, pred_parameters, gt_semantic):
        '''
        pred_parameters: [B, N, SC] <---> alphas
        gt_semantic: [B, N]
        ---
        e = alpha - 1
        S = sum(alphas)
        b = (alpha - 1) / S
        u = K / S
        gt_semantic: [B, N]
        '''

        K = pred_parameters.size(-1)
        dirichlet_strength = pred_parameters.sum(dim=-1) # [B, N]
        probs = (pred_parameters - 1) / dirichlet_strength
        uncert = K / dirichlet_strength

        # (p_1 - 1) ** 2 + p_2 ** 2 + ... + p_n ** 2 = 
        # 1 -2p_1 + p_1 ** 2 + p_2 ** 2 + ... + p_n ** 2 = 
        first = (probs ** 2).sum() + 1 - 2 * probs[gt_semantic]
        second = probs * (1 - probs) / (dirichlet_strength + 1)

        # add KL-Distance
        third = torch.lgamma(pred_parameters.sum(dim=-1)) - torch.lgamma(K) - torch.lgamma(pred_parameters).sum(dim=-1) +\
            (pred_parameters - 1.0) * (torch.digamma(pred_parameters) - torch.digamma(dirichlet_strength))

        return first + second + third


class RGBUncertaintyLoss(nn.Module):
    '''
    It calculates RGBUncertaintyLoss on one image
    '''
    def __init__(self, w):
        super().__init__()
        self.w = w
    
    def forward(self, pred_rgb, gt_rgb, uncert, alphas):
        '''
        calculates likelihood + regularization
        pred_rgb, gt: [B, N, 3]
        uncert: [B, N, 1]
        alphas: [B, N, T+t]
        '''
        idx_no_zero = ~torch.isclose(uncert, torch.zeros_like(uncert)).squeeze(-1)
        eps = 1e-9
        first = 0.5 * torch.mean((pred_rgb[idx_no_zero] - gt_rgb[idx_no_zero]) ** 2 / (uncert[idx_no_zero] + eps)) 
        second = 0.5 * torch.mean(torch.log(uncert[idx_no_zero] + eps))
        third = self.w * alphas[idx_no_zero].mean() + 4.0
        return first + second + third


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()


class SegmentationMeter:
    def __init__(self, num_semantic_classes, ignore_label=-1, device="cuda"):
        self.num_semantic_classes = num_semantic_classes
        self.ignore_label = ignore_label
        self.device = device
        self.conf_mat = torch.zeros(
            (self.num_semantic_classes, self.num_semantic_classes), 
            dtype=torch.long).to(self.device)
        self.conf_mat_calculate = ConfusionMatrix(
            task='multiclass', 
            num_classes=self.num_semantic_classes).to(self.device)

    def clear(self):
        self.conf_mat *= 0
    
    def measure(self):
        conf_mat = self.conf_mat.cpu().numpy()
        norm_conf_mat = np.transpose(
            np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

        missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
        exsiting_class_mask = ~ missing_class_mask

        class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
        ious = np.zeros(self.num_semantic_classes)
        for class_id in range(self.num_semantic_classes):
            ious[class_id] = (conf_mat[class_id, class_id] / (
                    np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                    conf_mat[class_id, class_id]))
        miou = nanmean(ious)
        miou_valid_class = np.mean(ious[exsiting_class_mask])
        return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious

    @torch.no_grad()
    def update(self, preds, truths):
        # TODO: may be bug here if [B, H, W] and W = self.num_semantic_classes
        if preds.ndim >= 3 and preds.shape[-1] == self.num_semantic_classes:
            preds = torch.argmax(preds, dim=-1)
        valid_pix_idx = truths.flatten() != self.ignore_label
        preds = preds.flatten()[valid_pix_idx].detach()
        truths = truths.flatten()[valid_pix_idx].detach()
        
        # update conf matrix
        self.conf_mat += self.conf_mat_calculate(preds, truths)

    def write(self, writer, global_step, prefix=""):
        writer.add_text(os.path.join(prefix, "---"), self.report(), global_step)

    def report(self):
        miou, miou_valid_class, total_accuracy, class_average_accuracy, ious = self.measure()
        return f"miou: {miou:.3f},\nmiou_valid_class: {miou_valid_class:.3f},\ntotal_accuracy: {total_accuracy:.3f},\nclass_average_accuracy: {class_average_accuracy:.3f},\nious: \n{ious}"

    def wandb_log(self, prefix):
        miou, miou_valid_class, total_accuracy, class_average_accuracy, ious = self.measure()
        return {
            f"{prefix}/miou" : miou, 
            f"{prefix}/miou_valid_class" : miou_valid_class, 
            f"{prefix}/total_accuracy" : total_accuracy, 
            f"{prefix}/class_average_accuracy" : class_average_accuracy, 
        }


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0
        self.name = "psnr"
        self.sign = 1.0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

    def wandb_log(self, prefix):
        return {f"{prefix}/PSNR": self.measure()}


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0
        self.name = "ssim"
        self.sign = 1.0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'
    
    def wandb_log(self, prefix):
        return {f"{prefix}/SSIM": self.measure()}


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net
        self.name = "lpips"
        self.sign = -1.0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

    def wandb_log(self, prefix):
        return {f"{prefix}/LPIPS": self.measure()}

@torch.no_grad()
def choose_new_k(model, holdout_dataset, k):
    '''
    Fucntion for active learning
    '''
    pres = []
    posts = []

    for data in holdout_dataset.dataloader():
        model.eval()

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]

        assert images.size(0) == 1 # batch_size must be equal to 1

        # eval with fixed background color
        bg_color = 1

        outputs = model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(holdout_dataset.opt))

        pred_uncert = outputs['uncertainty_image'] + 1e-9
        pred_alpha = outputs['alphas']

        alphas_shifted = torch.cat([torch.ones_like(pred_alpha[..., :1]), 1 - pred_alpha + 1e-15], dim=-1) # [N, T+t+1]
        weights = pred_alpha * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
        uncert_all = outputs['uncertainty_all'] + 1e-9

        pre = uncert_all.sum([1,2])
        post = (1. / (1. / uncert_all + weights ** 2.0 / pred_uncert)).sum([1 , 2])
        pres.append(pre)
        posts.append(post)
    
    pres = torch.cat(pres, 0)
    posts = torch.cat(posts, 0)
    index = torch.topk(pres-posts, k)[1].cpu().numpy()
    return index


class EarlyStop:
    def __init__(self, alpha=1e-3, relative=True, last=3, occ=2):
        '''
        Do one need to do early stopping

        ---
        arguments
        alpha: float
            - if effect if smaller than alpha, than stop
        relative: bool
            - relative effect of absolute effect
        last: int
            - how many last observation to consider
        bigger_is_better: bool
            - Bigger metric => better result
        occ: int
            - how many occurences in a row take into account
        '''
        self.alpha = alpha
        self.last = last
        self.metrics = []
        self.occ = occ
        self.relative = relative
        self.curr_occ = 0

    def _update_occ(self, new_res):
        last_mean = np.mean(self.metrics[-self.last:])
        denom = (np.abs(last_mean) + 1e-9) if self.relative else 1.0
        if np.abs(new_res - last_mean) / denom < self.alpha:
            self.curr_occ += 1
        else:
            self.curr_occ = 0

    def __call__(self, new_res):
        if len(self.metrics) == 0:
            self.metrics.append(new_res)
            return False
        
        self._update_occ(new_res)
        
        if self.curr_occ >= self.occ:
            return True
        else:
            self.metrics.append(new_res)
            return False


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 criterion_semantic=None, # loss function for semantic, if None, assume inline implementation in train_step
                 criterion_uncertainty=None, # loss function for uncertainty, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 lambd=1.0, # coeff for semantic loss
                 omega=1.0, # coeff in for uncertainty loss
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 segmentation_metrics=[],
                 depth_metrics=[],
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 semantic_remap=None, # just a dict
                 early_stop=None,
                 metric_to_monitor="lpips",
        ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.segmentation_metrics = segmentation_metrics
        self.depth_metrics = depth_metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.save_eval_images = opt.save_eval_images
        self.save_interval = opt.save_interval
        self.active_learning_interval = opt.active_learning_interval
        self.active_learning_num = opt.active_learning_num
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.semantic_remap = semantic_remap
        self.use_semantic = not opt.not_use_semantic
        self.lambd = lambd
        self.omega = omega
        self.metric_to_monitor = metric_to_monitor

        # create colormap
        self.sem_colormap = label_colormap(opt.total_num_classes)

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if isinstance(criterion_semantic, nn.Module):
            criterion_semantic.to(self.device)
        self.criterion_semantic = criterion_semantic
        
        if isinstance(criterion_uncertainty, nn.Module):
            criterion_uncertainty.to(self.device)
        self.criterion_uncertainty = criterion_uncertainty

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        if early_stop is None:
            early_stop = EarlyStop(alpha=5e-3, relative=False)
        self.early_stop = early_stop

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": {
                "metrics": [], 
                "segmentation_metrics": [], 
                "depth_metrics": []
            }, # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def _clear_metrics(self):
        for metric in self.metrics:
            metric.clear()
        for smetric in self.segmentation_metrics:
            smetric.clear()
        for dmetric in self.depth_metrics:
            dmetric.clear()

    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            pred_smntc = outputs['semantic_image']
            pred_uncert = outputs['uncertainty_image']
            pred_depth = outputs['depth']
            pred_uncert_smntc = outputs['semantic_uncertainty_image']

            # [debug] uncomment to plot the images used in train_step
            #torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)
            return pred_rgb, None, pred_smntc, None, pred_depth, loss

        images = data['images'] # [B, N, 3/4]
        if self.use_semantic:
            semantic_images = data['semantic_images']
        else:
            semantic_images = None
        B, N, C = images.shape
        SC = self.model.num_semantic_classes # number of semantic classes

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

    
        gt_smntc = semantic_images

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
        # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
    
        pred_rgb = outputs['image']
        pred_smntc = outputs['semantic_image']
        pred_uncert_smntc = outputs['semantic_uncertainty_image']
        pred_uncert = outputs['uncertainty_image']
        pred_depth = outputs['depth']
        pred_alpha = outputs['alphas']

        # MSE loss
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

        # # CrossEntropyLoss
        # if self.use_semantic:
        #     loss_smntc = self.criterion_semantic(pred_smntc.view(B * N, SC), gt_smntc.view(B * N)) # scalar
        # else:
        #     loss_smntc = torch.tensor(0.0)

        # CrossEntropyLoss
        if self.use_semantic:
            pred_smntc_probs = self.model.semantic_postprocess(pred_smntc, pred_uncert_smntc)
            loss_smntc = self.criterion_semantic(pred_smntc_probs.view(B * N, SC), gt_smntc.view(B * N)) # scalar
        else:
            loss_smntc = torch.tensor(0.0)

        # RGBUncertaintyLoss
        # TODO
        if pred_uncert.min() > 0:
            loss_uncert = self.criterion_uncertainty(pred_rgb, gt_rgb, pred_uncert, pred_alpha)
        else:
            loss_uncert = loss.mean()

        # patch-based rendering
        if self.opt.patch_size > 1:
            gt_rgb = gt_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()

            # torch_vis_2d(gt_rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss [not useful...]
            loss = loss + 1e-3 * self.criterion_lpips(pred_rgb, gt_rgb)

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()
        # they use weighted loss, but set lambda_ce = 1 it's ok
        losses_to_log = {"train/loss": loss.item(), "train/loss_ce": loss_smntc.item(), "train/loss_uncert": loss_uncert.item()}

        # TODO: Maybe delete loss of MSE?
        loss = (1 - self.omega) * loss + self.lambd * loss_smntc + self.omega * loss_uncert
        if not self.opt.no_wandb:
            wandb.log({**losses_to_log, "train/sum_loss": loss.item()})
        # loss = self.lambd * loss_smntc + self.omega * loss_uncert

        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss = loss + loss_ws.mean()

        return {
            "pred_rgb": pred_rgb, 
            "gt_rgb": gt_rgb, 
            "pred_smntc": pred_smntc, 
            "gt_smntc": gt_smntc, 
            "pred_depth": pred_depth, 
            "pred_uncert": pred_uncert, 
            "pred_uncert_smnth": pred_uncert_smntc, 
            "loss": loss
        }

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        if self.use_semantic:
            semantic_images = data['semantic_images'] # [B, H, W]
        else:
            semantic_images = None
        B, H, W, C = images.shape
        SC = self.model.num_semantic_classes

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        if self.use_semantic:
            gt_smntc = semantic_images
            pred_smntc = outputs['semantic_image'].reshape(B, H, W, SC)
            pred_uncert_smntc = outputs['semantic_uncertainty_image'].reshape(B, H, W, 1)
            pred_smntc_probs = self.model.semantic_postprocess(pred_smntc, pred_uncert_smntc)
        else:
            gt_smntc = None
            pred_smntc = None
            pred_uncert_smntc = None
            pred_smntc_probs = None
        pred_uncert = outputs['uncertainty_image'].reshape(B, H, W, 1)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_alpha = outputs['alphas'].reshape(B, H, W, -1)

        loss = self.criterion(pred_rgb, gt_rgb).mean()
        if self.use_semantic:
            loss_smntc = self.criterion_semantic(pred_smntc.view(B * H * W, SC), gt_smntc.view(B * H * W))
        else:
            loss_smntc = torch.tensor(0.0)
        
        loss_uncert = self.criterion_uncertainty(pred_rgb, gt_rgb, pred_uncert, pred_alpha)
        loss = (1 - self.omega) * loss + self.lambd * loss_smntc + self.omega * loss_uncert

        return {
            "pred_rgb": pred_rgb, 
            "gt_rgb": gt_rgb, 
            "pred_smntc": pred_smntc, 
            "pred_uncert_smntc": pred_uncert_smntc, 
            "pred_smntc_probs": pred_smntc_probs, 
            "gt_smntc": gt_smntc, 
            "pred_depth": pred_depth, 
            "pred_uncert": pred_uncert, 
            "loss": loss
        }

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        tm.start('render')
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))
        tm.end('render')
        SC = self.model.num_semantic_classes

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        if self.use_semantic:
            pred_uncert_smntc = outputs['semantic_uncertainty_image'].reshape(-1, H, W, 1)
            pred_smntc = outputs['semantic_image'].reshape(-1, H, W, SC)
            pred_smntc_probs = self.model.semantic_postprocess(pred_smntc, pred_uncert_smntc)
        else:
            pred_smntc = None
            pred_uncert_smntc = None
            pred_smntc_probs = None
        pred_uncert = outputs['uncertainty_image'].reshape(-1, H, W, 1)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return {
            "pred_rgb": pred_rgb, 
            "pred_smntc": pred_smntc, 
            "pred_uncert_smntc": pred_uncert_smntc, 
            "pred_smntc_probs": pred_smntc_probs, 
            "pred_uncert": pred_uncert,
            "pred_depth": pred_depth,
        }


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_dataset, valid_dataset, test_dataset, max_epochs, holdout_dataset=None):
        valid_loader = valid_dataset.dataloader()
        test_loader = test_dataset.dataloader()
        for epoch in range(self.epoch + 1, max_epochs + 1):
            train_loader = train_dataset.dataloader()
            # mark untrained region (i.e., not covered by any camera from the training dataset)
            if self.model.cuda_ray:
                self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

            # get a ref to error_map
            self.error_map = train_loader._data.error_map
        
            self.epoch = epoch
            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0 and self.epoch % self.save_interval == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.opt.video_interval == 0:
                self.test(test_loader, write_video=True)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

                # maybe do early stop. ONLY if use metrics
                if not self.use_loss_as_metric and self.early_stop(self.stats["results"]["metrics"][-1][self.metric_to_monitor]):
                    print(f"[INFO] Early stopping at {epoch}")
                    break

            if self.epoch % self.active_learning_interval == 0:
                print(f"[INFO] active learning at {epoch}")
                self.active_learning(train_dataset, holdout_dataset)
                print(f"[INFO] new train size {len(train_dataset):5}, new holdout size {len(holdout_dataset):5}")
                

    def active_learning(self, train_dataset, holdout_dataset):
        if holdout_dataset is None or len(holdout_dataset) == 0:
            return 
        # it is new indices to extend train dataset
        new_k = choose_new_k(
            self.model, holdout_dataset, 
            min(len(holdout_dataset), self.active_learning_num), # use all samples at the end
        )
        train_dataset.append(holdout_dataset, new_k)
        holdout_dataset.drop(new_k)

    def evaluate(self, loader, name=None):
        self.evaluate_one_epoch(loader, name)

    def test(self, loader, save_path=None, name=None, write_video=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_uncert = []
            all_preds_depth = []
            if self.use_semantic:
                all_preds_smntc = []
                all_preds_semantic_uncert = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    full_pred_test = self.test_step(data)
                    preds = full_pred_test['pred_rgb']
                    preds_smntc = full_pred_test['pred_smntc']
                    preds_uncert_semantic = full_pred_test['pred_uncert_semantic']
                    preds_smntc_probs = full_pred_test['pred_smntc_probs']
                    preds_uncert = full_pred_test['pred_uncert']
                    preds_depth = full_pred_test['pred_depth']
                    

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                
                pred_uncert = preds_uncert[0].detach().cpu().numpy()

                if self.use_semantic:
                    pred_smntc = preds_smntc_probs[0].detach().cpu().numpy().argmax(axis=-1).astype(np.uint8)
                    semantic_uncert = preds_uncert_semantic[0].detach().cpu().numpy()
                    
                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if self.use_semantic and self.semantic_remap:
                    self.semantic_remap.inv_remap(pred_smntc, inplace=True)
                if write_video:
                    all_preds.append(pred)
                    all_preds_uncert.append(pred_uncert)
                    if self.use_semantic:
                        all_preds_smntc.append(pred_smntc)
                        all_preds_semantic_uncert.append(semantic_uncert)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_uncert.png'), linear_transform(pred_uncert))
                    if self.use_semantic:
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_smntc.png'), pred_smntc)
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_semantic_uncert.png'), linear_transform(semantic_uncert))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_uncert = np.stack(all_preds_uncert, axis=0)
            if self.use_semantic:
                all_preds_smntc = np.stack(all_preds_smntc, axis=0)
                all_preds_semantic_uncert = np.stack(all_preds_semantic_uncert, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=10, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_uncert.mp4'), linear_transform(all_preds_uncert), fps=10, quality=8, macro_block_size=1)
            if self.use_semantic:
                imageio.mimwrite(os.path.join(save_path, f'{name}_smntc.mp4'), self.sem_colormap[all_preds_smntc], fps=10, quality=8, macro_block_size=1)
                imageio.mimwrite(os.path.join(save_path, f'{name}_smntc_uncert.mp4'), linear_transform(all_preds_semantic_uncert), fps=10, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=10, quality=8, macro_block_size=1)
            to_log_videos = {
                "video/rgb": wandb.Video(os.path.join(save_path, f'{name}_rgb.mp4'), format="mp4"),
                "video/uncert": wandb.Video(os.path.join(save_path, f'{name}_uncert.mp4'), format="mp4"),
                "video/depth": wandb.Video(os.path.join(save_path, f'{name}_depth.mp4'), format="mp4"),
            }
            if self.use_semantic:
                to_log_videos = {
                    **to_log_videos,
                    "video/semantic": wandb.Video(os.path.join(save_path, f'{name}_smntc.mp4'), format="mp4"),
                    "video/semantic_uncert": wandb.Video(os.path.join(save_path, f'{name}_smntc_uncert.mp4'), format="mp4"),
                }
            if not self.opt.no_wandb:
                wandb.log(to_log_videos)


        self.log(f"==> Finished Test.")
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, pred_smntc, gt_smntc, pred_depth, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_smntc, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_smntc = F.interpolate(pred_smntc.argmax(axis=-1).unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_smntc = preds_smntc[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'image_smntc': pred_smntc,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            self._clear_metrics()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                full_pred_train = self.train_step(data)
                preds = full_pred_train["pred_rgb"]
                truths = full_pred_train["gt_rgb"]
                preds_smntc = full_pred_train["pred_smntc"]
                preds_uncert_smntc = full_pred_train["pred_uncert_smntc"]
                preds_smntc_probs = full_pred_train["pred_smntc_probs"]
                gt_smntc = full_pred_train["gt_smntc"]
                preds_uncert = full_pred_train["pred_uncert"]
                pred_depth = full_pred_train["pred_depth"]
                loss = full_pred_train["loss"]
         
            self.scaler.scale(loss).backward()
            with torch.no_grad():
                sum_grads = []
                for p in self.model.parameters():
                    if p.requires_grad:
                        if p.grad is None:
                            continue
                        sum_grads.append(torch.abs(p.grad).mean().item())
                if len(sum_grads) == 0:
                    sum_grads = [0.0]
                if not self.opt.no_wandb:
                    wandb.log({"train/grad_norm": np.mean(sum_grads)})
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0) # TODO not hardcode?
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                    for smetric in self.segmentation_metrics:
                        smetric.update(preds_smntc_probs, gt_smntc) # IMPORTANT!
                    for dmetric in self.depth_metrics:
                        dmetric.update(pred_depth, ...)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                metrics_to_report = {}
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    metrics_to_report.update(metric.wandb_log("train"))
                for smetric in self.segmentation_metrics:
                    self.log(smetric.report(), style="red")
                    metrics_to_report.update(smetric.wandb_log("train"))
                for dmetric in self.depth_metrics:
                    self.log(dmetric.report(), style="red")
                    metrics_to_report.update(dmetric.wandb_log("train"))
                self._clear_metrics()
                if not self.opt.no_wandb:
                    wandb.log(metrics_to_report)

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            self._clear_metrics()
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    full_pred_eval = self.eval_step(data)
                    preds = full_pred_eval["pred_rgb"]
                    truths = full_pred_eval["gt_rgb"]
                    preds_smntc = full_pred_eval["pred_smntc"]
                    preds_uncert_smntc = full_pred_eval["pred_uncert_smntc"]
                    preds_smntc_probs = full_pred_eval["pred_smntc_probs"]
                    gt_smntc = full_pred_eval["gt_smntc"]
                    preds_uncert = full_pred_eval["pred_uncert"]
                    preds_depth = full_pred_eval["pred_depth"]
                    loss = full_pred_eval["loss"]
                    

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)


                    preds_uncert_list = [torch.zeros_like(preds_uncert).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_uncert_list, preds_uncert)
                    preds_uncert = torch.cat(preds_uncert_list, dim=0)

                    if self.use_semantic:
                        preds_smntc_list = [torch.zeros_like(preds_smntc).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_smntc_list, preds_smntc)
                        preds_smntc = torch.cat(preds_smntc_list, dim=0)

                        preds_smntc_probs_list = [torch.zeros_like(preds_smntc_probs).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_smntc_probs_list, preds_smntc)
                        preds_smntc = torch.cat(preds_smntc_probs_list, dim=0)

                        preds_uncert_smntc = [torch.zeros_like(preds_uncert_smntc).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_uncert_smntc, preds_smntc)
                        preds_smntc = torch.cat(preds_uncert_smntc, dim=0)

                        gt_smntc_list = [torch.zeros_like(gt_smntc).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(gt_smntc_list, gt_smntc)
                        gt_smntc = torch.cat(gt_smntc_list, dim=0)
                    
                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                    for smetric in self.segmentation_metrics:
                        smetric.update(preds_smntc_probs, gt_smntc)
                    for dmetric in self.depth_metrics:
                        dmetric.update(preds_depth, ...)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    if self.use_semantic:
                        save_path_smntc = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_smntc.png')
                    save_path_uncert = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_uncert.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    if self.save_eval_images:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        if self.opt.color_space == 'linear':
                            preds = linear_to_srgb(preds)

                        pred = preds[0].detach().cpu().numpy()
                        pred = (pred * 255).astype(np.uint8)
                        
                        if self.use_semantic:
                            pred_smntc = preds_smntc[0].detach().cpu().numpy()
                            pred_smntc = pred_smntc.argmax(axis=-1).astype(np.uint8)

                        pred_uncert = preds_uncert[0].detach().cpu().numpy()

                        pred_depth = preds_depth[0].detach().cpu().numpy()
                        pred_depth = (pred_depth * 255).astype(np.uint8)
                        
                        cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                        if self.use_semantic:
                            cv2.imwrite(save_path_smntc, pred_smntc)
                        cv2.imwrite(save_path_uncert, pred_uncert)
                        cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                rbg_metrics_item = {}
                for metric in self.metrics:
                    result = metric.measure()
                    rbg_metrics_item[metric.name] = result
                self.stats["results"]['metrics'].append(rbg_metrics_item) # if max mode, use -result
                # result if self.best_mode == 'min' else - result
            else:
                self.stats["results"]['metrics'].append(average_loss) # if no metric, choose best by min loss
            
            if not self.use_loss_as_metric and len(self.segmentation_metrics) > 0:
                result = self.segmentation_metrics[0].report()
                self.stats["results"]['segmentation_metrics'].append(result)

            if not self.use_loss_as_metric and len(self.depth_metrics) > 0:
                result = self.depth_metrics[0].report()
                self.stats["results"]['depth_metrics'].append(result)
            # calculate metrics
            metrics_to_report = {}
            for metric in self.metrics:    
                metrics_to_report.update(metric.wandb_log("eval"))
                self.log(metric.report(), style="blue")
            for smetric in self.segmentation_metrics:
                metrics_to_report.update(smetric.wandb_log("eval"))
                self.log(smetric.report(), style="blue")
            for dmetric in self.depth_metrics:
                metrics_to_report.update(dmetric.wandb_log("eval"))
                self.log(dmetric.report(), style="blue")
            self._clear_metrics()
            if not self.opt.no_wandb:
                wandb.log(metrics_to_report)

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]['metrics']) > 0:
                if self.use_loss_as_metric:
                    last = self.stats["results"]['metrics'][-1]
                else:
                    last = self.stats["results"]['metrics'][-1][self.metric_to_monitor]
                if self.stats["best_result"] is None or last < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {last}")
                    self.stats["best_result"] = last

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
