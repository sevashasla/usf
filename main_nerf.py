import torch
import argparse
import json
import os

from nerf_model.provider import NeRFDataset
from nerf_model.gui import NeRFGUI
from nerf_model.utils import *
from nerf_model.network import NeRFNetwork
from nerf_model.semantic_utils import *

from functools import partial
import wandb

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--n_video', type=int, default=100)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--save_eval_images', action='store_true', help="save eval images or not")
    parser.add_argument('--downscale', type=int, default=1)
    
    parser.add_argument('--eval_ratio', type=float, default=0.2)
    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--holdout_ratio', type=float, default=0.0)
    parser.add_argument('--test_ratio', type=float, default=0.0)
    parser.add_argument('--total_num_classes', type=int, default=101) # from replica dataset
    parser.add_argument('--metric_to_monitor', type=str, default="lpips") # from replica dataset

    ### training options
    # parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--epochs', type=int, default=20, help="training epochs")
    parser.add_argument('--video_interval', type=int, default=None, help="how often to make & save video")
    parser.add_argument('--video_mode', type=int, default=1, help="mode of how to choose poses for video making")
    parser.add_argument('--warmup_epochs', type=float, default=1, help="number of warmup epochs")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--lambd', type=float, default=0.0, help="coeff for semantic loss")
    parser.add_argument('--save_interval', type=int, default=10, help="how often to save")
    parser.add_argument('--use_loss_as_metric', action='store_true', help="")
    
    
    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    parser.add_argument('--first_encoding', type=str, choices=["hashgrid", "sphere_harmonics"], default="hashgrid", help="type of encoding for coordinates")
    parser.add_argument('--arc_rgb', type=int, default=0, choices=[0, 1, 2, 3], help="watch network. How deep rgb uncertainty from the out")
    parser.add_argument('--arc_smntc', type=int, default=0, choices=[0, 1, 2], help="watch network. How deep semantic uncertainty from the out")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    ### semantic
    parser.add_argument('--num_semantic_classes', type=int, required=False, help="number of semantic classes")
    parser.add_argument('--semantic_remap', type=json.loads, required=False, help="remap for semantic classes")
    parser.add_argument('--use_semantic', action="store_true", help="use and predict the semantic labels")

    # uncertrainty
    parser.add_argument('--alpha_uncert', type=float, default=0.01, help="coeff inside RGBUncertaintyLoss")
    parser.add_argument('--beta_min', type=float, default=0.01, help="beta_min in NeRFNetwork, min of uncertainty")
    parser.add_argument('--omega', type=float, default=0.0, help="weight of RGBUncertaintyLoss")
    parser.add_argument('--use_uncert', action="store_true", help="use and predict the semantic labels")
    parser.add_argument('--use_semantic_uncert', action="store_true", help="use and predict the semantic labels")

    ### active learning
    parser.add_argument('--active_learning_interval', type=int, default=None, help="how often to apply active learning")
    parser.add_argument('--active_learning_num', type=int, default=4, help="how often to apply active learning")
    parser.add_argument('--Ngen', type=int, default=10, help="How many samples to generate in semantic postprocess prob")
    parser.add_argument('--su_weight', type=float, default=0.0, help="weight of Semantic Uncertainty during active learning")

    ### SPECIAL PARAMETERS
    # sparse-views
    parser.add_argument("--sparse_views", action='store_true',
                        help='Use labels from a sparse set of frames')
    parser.add_argument("--sparse_ratio", type=float, default=0,
                        help='The portion of dropped labelling frames during training, which can be used along with all working modes.')    
    parser.add_argument("--label_map_ids", nargs='*', type=int, default=[],
                        help='In sparse view mode, use selected frame ids from sequences as supervision.')
    parser.add_argument("--random_sample", action='store_true', help='Whether to randomly/evenly sample frames from the sequence.')

    # denoising---pixel-wise
    parser.add_argument("--pixel_denoising", action='store_true',
                        help='Whether to work in pixel-denoising tasks.')
    parser.add_argument("--pixel_noise_ratio", type=float, default=0,
                        help='In sparse view mode, if pixel_noise_ratio > 0, the percentage of pixels to be perturbed in each sampled frame  for pixel-wise denoising task..')
                        
    # denoising---region-wise
    parser.add_argument("--region_denoising", action='store_true',
                        help='Whether to work in region-denoising tasks by flipping class labels of chair instances in Replica Room_2')
    parser.add_argument("--region_noise_ratio", type=float, default=0,
                        help='In region-wise denoising task, region_noise_ratio is the percentage of chair instances to be perturbed in each sampled frame for region-wise denoising task.')
    parser.add_argument("--uniform_flip", action='store_true',
                        help='In region-wise denoising task, whether to change chair labels uniformly or not, i.e., by ascending area ratios. This corresponds to two set-ups mentioned in the paper.')
    parser.add_argument("--instance_id", nargs='*', type=int, default=[3, 6, 7, 9, 11, 12, 13, 48],
                        help='In region-wise denoising task, the chair instance ids in Replica Room_2 to be randomly perturbed. The ids of all 8 chairs are [3, 6, 7, 9, 11, 12, 13, 48]')
       
    # super-resolution
    parser.add_argument("--super_resolution", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument('--dense_sr',  action='store_true', help='Whether to use dense or sparse labels for SR instead of dense labels.')
    parser.add_argument('--sr_factor',  type=int, default=8, help='Scaling factor of super-resolution.')

    # label propagation
    parser.add_argument("--label_propagation", action='store_true',
                        help='Label propagation using partial seed regions.')
    parser.add_argument("--partial_perc", type=float, default=0,
                        help='0: single-click propagation; 1: using 1-percent sub-regions for label propagation, 5: using 5-percent sub-regions for label propagation')

    # cache
    parser.add_argument('--visualise_save',  action='store_true', help='whether to save the noisy labels into harddrive for later usage')
    parser.add_argument('--load_saved',  action='store_true', help='use trained noisy labels for training to ensure consistency betwwen experiments')

    parser.add_argument('--path_to_save_tm', default="", type=str, help="where to store json with rendering time")

    # wandb
    parser.add_argument('--project', type=str, default="ngp_with_semantic_nerf")
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--sweep_id', type=str, default=None)
    parser.add_argument('--no_wandb', action="store_true")
    parser.add_argument('--wandbdir', type=str, default="/mnt/hdd8/skorokhodov_vs/wandb_logs")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    print(opt)

    if opt.test and opt.num_semantic_classes is None and opt.use_semantic:
        raise RuntimeError("'num_semantic_classes' must be known if test")
    
    if opt.train_ratio is None:
        opt.train_ratio = 1.0 - opt.eval_ratio - opt.holdout_ratio

    if opt.video_interval is None:
        opt.video_interval = opt.epochs

    if opt.active_learning_interval is None:
        opt.active_learning_interval = opt.epochs + 5

    if opt.use_semantic_uncert:
        opt.use_semantic = True 

    seed_everything(opt.seed)

    criterion = torch.nn.MSELoss(reduction='none')
    # criterion_semantic = torch.nn.CrossEntropyLoss()
    criterion_semantic = torch.nn.NLLLoss()
    criterion_uncertainty = RGBUncertaintyLoss(opt.alpha_uncert)
    # criterion float partial(educticoeff inside RGBUncertaintyLoss')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.test:
        model = NeRFNetwork(
            opt, 
            encoding=opt.first_encoding,
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            num_semantic_classes=opt.num_semantic_classes,
            beta_min=opt.beta_min, 
            Ngen=opt.Ngen, 
            arc_rgb=opt.arc_rgb, 
            arc_smntc=opt.arc_smntc, 
        )

        metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter(device=device)]
        if opt.use_semantic:
            segmentation_metrics = [SegmentationMeter(opt.num_semantic_classes)]
        else:
            segmentation_metrics = []
        # TODO
        depth_metrics = []
        trainer = Trainer(
            'ngp', opt, model, 
            device=device, workspace=opt.workspace, 
            criterion=criterion, 
            criterion_semantic=criterion_semantic, lambd=opt.lambd, 
            criterion_uncertainty=criterion_uncertainty, omega=opt.omega, 
            fp16=opt.fp16, 
            metrics=metrics, segmentation_metrics=segmentation_metrics, depth_metrics=depth_metrics,
            use_checkpoint=opt.ckpt,
            semantic_remap=SemanticRemap(opt.semantic_remap) if opt.semantic_remap else None, 
            metric_to_monitor=opt.metric_to_monitor, 
        )

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
    
    else:

        nerf_dataset = NeRFDataset(opt, device=device, type='train', downscale=opt.downscale, semantic_remap=SemanticRemap(opt.semantic_remap) if opt.semantic_remap else None, )

        num_semantic_classes = nerf_dataset.num_semantic_classes     
        train_loader = nerf_dataset.dataloader()

        model = NeRFNetwork(
            opt, 
            encoding=opt.first_encoding,
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            num_semantic_classes=num_semantic_classes,
            beta_min=opt.beta_min,
            Ngen=opt.Ngen,
            arc_rgb=opt.arc_rgb, 
            arc_smntc=opt.arc_smntc, 
        )

        print(model)

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        iters = len(train_loader) * opt.epochs

        # decay to 0.1 * init_lr at last iter step
        def increase_lr(step, gamma, i):
            return 1.0 + (gamma - 1.0) * (i // step)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 
                0.1 ** min(iter / iters, 1) * \
                min(1e-3 + 1 / (opt.warmup_epochs * len(train_loader)) * iter, 1) * \
                increase_lr(opt.active_learning_interval * len(train_loader), 1.2, iter)
            )

        metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter(device=device)]
        if opt.use_semantic:
            segmentation_metrics = [SegmentationMeter(num_semantic_classes)]
        else:
            segmentation_metrics = []
        # TODO
        depth_metrics = []

        trainer = Trainer(
            'ngp', opt, model, 
            device=device, workspace=opt.workspace, 
            optimizer=optimizer, 
            criterion=criterion, 
            criterion_semantic=criterion_semantic, lambd=opt.lambd, 
            criterion_uncertainty=criterion_uncertainty, omega=opt.omega, 
            ema_decay=0.95, fp16=opt.fp16, 
            lr_scheduler=scheduler, scheduler_update_every_step=True, 
            metrics=metrics, segmentation_metrics=segmentation_metrics, depth_metrics=depth_metrics, 
            use_checkpoint=opt.ckpt, 
            eval_interval=opt.eval_interval,
            semantic_remap=nerf_dataset.semantic_remap,
            metric_to_monitor=opt.metric_to_monitor, 
            use_loss_as_metric=opt.use_loss_as_metric
        )

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_dataset = NeRFDataset(
                opt, device=device, type='val', downscale=opt.downscale, 
                semantic_remap=nerf_dataset.semantic_remap,
                tvh_indexer=nerf_dataset.tvh_indexer
            )

            holdout_dataset = NeRFDataset(
                opt, device=device, type='holdout', downscale=opt.downscale, 
                semantic_remap=nerf_dataset.semantic_remap,
                tvh_indexer=nerf_dataset.tvh_indexer
            )

            print(f"[INFO] MAX_EPOCH: {opt.epochs}, ITERS: {iters}")
            print(f"[INFO] RESUME: {opt.resume}")
            if not opt.no_wandb:
                # sweep is running!
                if opt.sweep_id:
                    wandb.init(
                        id=opt.sweep_id, 
                        resume=True,
                        project=opt.project,
                        config={**vars(opt), "mode": "semantic_ngp"},
                        tags=["semantic_ngp"],
                        dir=opt.wandbdir,
                    )
                else:
                    wandb.init(
                        project=opt.project,
                        group=opt.group,
                        name=f"semantic_ngp: {os.path.basename(opt.workspace)}",
                        config={**vars(opt), "mode": "semantic_ngp"},
                        tags=["semantic_ngp"],
                        dir=opt.wandbdir,
                        resume=opt.resume,
                    )

            test_dataset = NeRFDataset(opt, device=device, type='test', semantic_remap=nerf_dataset.semantic_remap, tvh_indexer=nerf_dataset.tvh_indexer)
            video_dataset = NeRFDataset(opt, device=device, type='video', semantic_remap=nerf_dataset.semantic_remap, n_video=opt.n_video)
            test_loader = test_dataset.dataloader()
            trainer.train(
                nerf_dataset, valid_dataset, test_dataset, 
                video_dataset, opt.epochs, holdout_dataset
            )
            
            # trainer.save_mesh(resolution=256, threshold=10)

            if opt.path_to_save_tm == '':
                opt.path_to_save_tm = os.path.join(opt.workspace, "time_measurements.json")
            
            tm.save(opt.path_to_save_tm)
            if not opt.no_wandb:
                wandb.run.summary["t_mean_render"] = tm.mean('render')
                wandb.finish()

