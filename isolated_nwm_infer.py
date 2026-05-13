# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# === 修改开始：不再在线跑 VGGT，删除 predict_tracks / build_geom_from_tracks 的 import ===
# from vggt.dependency.track_predict import predict_tracks
# === 修改结束 ===

from distributed import init_distributed
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import yaml
import argparse
import os
import numpy as np

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

import misc
import distributed as dist
from models import CDiT_models
from datasets import EvalDataset, N_GEOM_FIXED
from PIL import Image


# === 修改开始：build_geom_from_tracks 不再在这里直接调用，删除 import ===
# from misc import build_geom_from_tracks
# === 修改结束 ===


def save_image(output_file, img, unnormalize_img):
    img = img.detach().cpu()
    if unnormalize_img:
        img = misc.unnormalize(img)

    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

    image.save(output_file)


def get_dataset_eval(config, dataset_name, eval_type, predefined_index=True):
    data_config = config["eval_datasets"][dataset_name]
    if predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/{eval_type}.pkl"
    else:
        predefined_index = None

    # === 修改开始：geom_cache_root 从顶层 config 读，eval 也用 cache ===
    geom_cache_root = config.get("geom_cache_root", None)
    # === 修改结束 ===

    dataset = EvalDataset(
        data_folder=data_config["data_folder"],
        data_split_folder=data_config["test"],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
        max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
        len_traj_pred=config["eval_len_traj_pred"],
        traj_stride=config["traj_stride"],
        context_size=config["eval_context_size"],
        normalize=config["normalize"],
        transform=misc.transform,
        goals_per_obs=4,
        predefined_index=predefined_index,
        traj_names='traj_names.txt',
        # === 修改开始 ===
        geom_cache_root=geom_cache_root,
        # === 修改结束 ===
    )

    return dataset


@torch.no_grad()
def model_forward_wrapper(all_models, curr_obs, curr_delta, num_timesteps, latent_size, device,
                          num_cond, num_goals=1, rel_t=None, progress=False,
                          geom=None):
    """
    === 修改开始 ===
    geom: [B, N_geom, 6] 或 None。
        - 训练 evaluate / 推理 main 都会传 geom（从 dataset 加载）。
        - planning_eval.py 目前没传，这里 fallback 到全零 geom（即 geom 不提供信息）。
        - rollout 时（generate_rollout）每一步用的是首步的 geom，
          因为 rollout 的 curr_obs 是逐步生成的预测图，没有对应 cache。
    === 修改结束 ===
    """
    model, diffusion, vae = all_models
    x = curr_obs.to(device)  # [B, T, 3, H, W]
    y = curr_delta.to(device)

    # === 修改开始：geom 不再在线跑 VGGT；接受外部传入或 fallback 全零 ===
    if geom is None:
        # fallback：planning 等没传 geom 的路径用全零
        geom = torch.zeros(x.shape[0], N_GEOM_FIXED, 6, device=device, dtype=torch.float32)
    else:
        geom = geom.to(device)
    # === 修改结束 ===

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        if rel_t is None:
            rel_t = (torch.ones(B) * (1. / 128.)).to(device)
            rel_t *= num_timesteps

        x = x.flatten(0, 1)
        x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
        x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3],
                                                     x.shape[4]).flatten(0, 1)
        z = torch.randn(B * num_goals, 4, latent_size, latent_size, device=device)
        y = y.flatten(0, 1)

        # 修改开始
        geom = geom.unsqueeze(1).expand(B, num_goals, geom.shape[1], geom.shape[2]).flatten(0, 1)
        # 修改结束，下面这行新增了一个geom参数
        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t, geom=geom)
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress, device=device
        )
        samples = vae.decode(samples / 0.18215).sample

        return torch.clip(samples, -1., 1.)


def generate_rollout(args, output_dir, rollout_fps, idxs, all_models, obs_image, gt_image, delta,
                     num_cond, device, geom=None):
    """
    === 修改开始：增加 geom 参数。整个 rollout 全程复用首步的 geom（因为后续步的
    curr_obs 是预测出来的图像，没有对应离线 cache 可用）。这是一个简化但常见的做法。 ===
    """
    rollout_stride = args.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride - 1::rollout_stride]
    delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)
    curr_obs = obs_image.clone().to(device)

    for i in range(gt_image.shape[1]):
        curr_delta = delta[:, i:i + 1].to(device)
        if args.gt:
            x_pred_pixels = gt_image[:, i].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models, curr_obs, curr_delta,
                rollout_stride, args.latent_size,
                num_cond=num_cond, num_goals=1, device=device,
                geom=geom,
            )

        curr_obs = torch.cat((curr_obs, x_pred_pixels.unsqueeze(1)), dim=1)  # append current prediction
        curr_obs = curr_obs[:, 1:]  # remove first observation
        visualize_preds(output_dir, idxs, i, x_pred_pixels)


def generate_time(args, output_dir, idxs, all_models, obs_image, gt_output, delta, secs, num_cond, device,
                  geom=None):
    """
    === 修改开始：增加 geom 参数。每次预测都用首步的 geom（输入 obs_image 不变）。 ===
    """
    eval_timesteps = [sec * args.input_fps for sec in secs]
    for sec, timestep in zip(secs, eval_timesteps):
        curr_delta = delta[:, :timestep].sum(dim=1, keepdim=True)
        if args.gt:
            x_pred_pixels = gt_output[:, timestep - 1].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models, obs_image, curr_delta,
                timestep, args.latent_size,
                num_cond=num_cond, num_goals=1, device=device,
                geom=geom,
            )
        visualize_preds(output_dir, idxs, sec, x_pred_pixels)


def visualize_preds(output_dir, idxs, sec, x_pred_pixels):
    for batch_idx, sample_idx in enumerate(idxs.squeeze()):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(output_dir, f'id_{sample_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f'{sec}.png')
        save_image(image_file, x_pred_pixels[batch_idx], True)


@torch.no_grad()
def main(args):
    _, _, device, _ = init_distributed()
    print(args)
    device = torch.device(device)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    exp_eval = args.exp

    # model & config setup
    if args.gt:
        args.save_output_dir = os.path.join(args.output_dir, 'gt')
    else:
        exp_name = os.path.basename(exp_eval).split('.')[0]
        args.save_output_dir = os.path.join(args.output_dir, exp_name)

    if args.ckp != '0100000':
        args.save_output_dir = args.save_output_dir + "_%s" % (args.ckp)

    os.makedirs(args.save_output_dir, exist_ok=True)

    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(exp_eval, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    latent_size = config['image_size'] // 8
    args.latent_size = config['image_size'] // 8

    num_cond = config['context_size']
    print("loading")
    model_lst = (None, None, None)
    if not args.gt:
        # ========================= 修改开始：LoRA 加载流程 =========================
        from misc import load_model_for_inference

        # base ckpt: NWM 原作者预训练权重（必须给）
        # lora ckpt: LoRA 微调出来的 ckpt
        # 两个路径都从 config 里读，这样切换实验只改 yaml 不改代码
        base_ckpt_path = config.get('from_checkpoint', None)
        lora_ckpt_path = f'{config["results_dir"]}/{config["run_name"]}/checkpoints/{args.ckp}.pth.tar'

        # LoRA 超参数必须与训练时一致，否则参数 shape 对不上
        lora_rank = int(config.get('lora_rank', 8))
        lora_alpha = int(config.get('lora_alpha', 16))
        lora_dropout = float(config.get('lora_dropout', 0.0))

        # 1 在 CPU 上建模
        model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4)

        # 2 load_model_for_inference 内部会：
        # 加载 base -> 注入 LoRA -> 加载 LoRA ckpt -> .to(device).eval()
        model = load_model_for_inference(
            model,
            base_ckpt_path=base_ckpt_path,
            lora_ckpt_path=lora_ckpt_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=device,
            verbose=True,
        )

        # === 修改：原来这里是 torch.compile (typo)。改回 torch.compile ===
        model = torch.compile(model)
        diffusion = create_diffusion(str(250))
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
        model_lst = (model, diffusion, vae)
        # ========================= 修改结束：LoRA 加载流程 =========================

    # Loading Datasets
    dataset_names = args.datasets.split(',')
    datasets = {}

    for dataset_name in dataset_names:
        dataset_val = get_dataset_eval(config, dataset_name, args.eval_type, predefined_index=True)

        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

        curr_data_loader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        datasets[dataset_name] = curr_data_loader

    print_freq = 1
    header = 'Evaluation: '
    metric_logger = dist.MetricLogger(delimiter="  ")

    for dataset_name in dataset_names:
        dataset_save_output_dir = os.path.join(args.save_output_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = datasets[dataset_name]

        # === 修改开始：EvalDataset 现在多返回一个 geom（第 5 个返回值） ===
        for data_iter_step, (idxs, obs_image, gt_image, delta, geom) in enumerate(
                metric_logger.log_every(curr_data_loader, print_freq, header)
        ):
            # 修改：去掉最外层的autocast，防止里头的predict_track方法精度不够
            # （目前 predict_track 已离线，没在这里跑，但保持原样不加 autocast 也无害）
            obs_image = obs_image[:, -num_cond:].to(device)
            gt_image = gt_image.to(device)
            geom = geom.to(device)  # [B, N_geom, 6]
            num_cond = config["context_size"]
            if args.eval_type == 'rollout':
                for rollout_fps in args.rollout_fps_values:
                    curr_rollout_output_dir = os.path.join(dataset_save_output_dir, f'rollout_{rollout_fps}fps')
                    os.makedirs(curr_rollout_output_dir, exist_ok=True)
                    generate_rollout(args, curr_rollout_output_dir, rollout_fps, idxs, model_lst,
                                     obs_image, gt_image, delta, num_cond, device, geom=geom)
            elif args.eval_type == 'time':
                secs = np.array([2 ** i for i in range(0, args.num_sec_eval)])
                curr_time_output_dir = os.path.join(dataset_save_output_dir, 'time')
                os.makedirs(curr_time_output_dir, exist_ok=True)
                generate_time(args, curr_time_output_dir, idxs, model_lst,
                              obs_image, gt_image, delta, secs, num_cond, device, geom=geom)
        # === 修改结束 ===


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='0100000')
    parser.add_argument("--num_sec_eval", type=int, default=5)
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--eval_type", type=str, default=None,
                        help="type of evaluation has to be either 'time' or 'rollout'")
    # Rollout Evaluation Args
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    parser.add_argument("--gt", type=int, default=0, help="set to 1 to produce ground truth evaluation set")
    args = parser.parse_args()

    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]

    main(args)