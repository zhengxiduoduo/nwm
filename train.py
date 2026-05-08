# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

# 修改开始 加 LoRA

import math
# import torch
import torch.nn as nn
import torch.nn.functional as F


# 修改结束 加 LoRA



from isolated_nwm_infer import model_forward_wrapper
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import matplotlib.pyplot as plt 
import yaml
from vggt.dependency.track_predict import predict_tracks


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL

from distributed import init_distributed
from models import CDiT_models
from diffusion import create_diffusion
from datasets import TrainingDataset
from misc import transform
from misc import build_geom_from_tracks


# 修改开始 加loRA
class LoRALinear(nn.Module):
    """
    Wrap an existing nn.Linear with LoRA.
    Original weight/bias are frozen.
    Only lora_A and lora_B are trainable.
    """

    def __init__(self, base_layer: nn.Linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear only supports nn.Linear, got {type(base_layer)}")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # keep original linear
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if rank > 0:
            # LoRA params
            self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

            # init: A kaiming, B zero -> start from no-op
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        base_out = self.base_layer(x)

        if self.rank <= 0:
            return base_out

        x_d = self.dropout(x)
        # (..., in_features) -> (..., rank) -> (..., out_features)
        lora_out = F.linear(F.linear(x_d, self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out

########### LoRA 的 help 函数 ###########

def replace_linear_with_lora(module: nn.Module, rank=8, alpha=16, dropout=0.0, verbose=False):
    """
    递归 替换所有 Linear 为 LoRA
    Recursively replace all nn.Linear in module with LoRALinear.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            if verbose:
                print(f"[LoRA] Replaced Linear layer: {name}")
        else:
            replace_linear_with_lora(child, rank=rank, alpha=alpha, dropout=dropout, verbose=verbose)



# 修改结束 加loRA


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new CDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:

    # 修改开始，device “GPU编号” 和 “PyTorch设备对象” 分开写
    # _, rank, device, _ = init_distributed()
    _, rank, device_id, _ = init_distributed()
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    # 修改结束

    # rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)
    
    # Setup an experiment folder:
    os.makedirs(config['results_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    tokenizer = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    latent_size = config['image_size'] // 8

    assert config['image_size'] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    num_cond = config['context_size']

    # ===================== LoRA修改开始：LoRA 微调流程 =====================
    # 关键顺序：
    #   1）在CPU上建模型
    #   2）加载 base 预训练 ckpt (strict=False, 容忍新加的 geometry 模块)
    #   3）注入LoRA
    #   4）冻结非可训练参数
    #   5）.to(device)
    #   6）deepcopy 出 EMA
    #   7）optimizer 只接收 requires_grad=True 的参数
    #   8）（如有）加载 LoRA-only 续训 ckpt
    #   9）DDP, find_unused_parameters=True

    # 1 建模型（先放 CPU）
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4)

    # 2 加载 base 预训练 ckpt（在注入 LoRA 之前）
    base_ckpt_path = config.get('from_checkpoint', 0)

    if base_ckpt_path:
        print("Load BASE pretrained checkpoint from", base_ckpt_path)
        base_ckpt = torch.load(base_ckpt_path, map_locaation='cpu', weights_only=False)
        # 优先用 ema 权重做 base，效果通常更好；没有就用 model
        base_state = base_ckpt.get('ema', base_ckpt.get('model'))
        base_state = {k.replace('_orig_mod.', ''): v for k, v in base_state.items()}
        res = model.load_state_dict(base_state, strict=False)
        print(f"[Base ckpt] missing keys: {len(res.missing_keys)}, unexpected: {len(res.unexpected_keys)}")
        if rank == 0:   # real logger
            print("     missing (新加板块，应当包含 geometry/gcttn/norm_geom/adaLN_modulation):")
            for k in res.missing_keys[:20]:
                print("         ", k)
            print("     unexpected:")
            for k in res.unexpected_keys[:20]:
                print("         ", k)


    # 3 注入 LoRA
    lora_rank = int(config.get('lora_rank', 8))
    lora_alpha = int(config.get('lora_alpha', 16))
    lora_dropout = float(config.get('lora_dropout', 0.0))
    replace_linear_with_lora(model, rank=lora_rank, alpha=lora_alpha,
                             dropout=lora_dropout, verbose=False)
    print(f"[LoRA] injected. rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")

    # 4 冻结非可训练参数
    # 可训练：LoRA 参数 + 新加的 geometry 板块 + cross-attn 到 geometry 的 norm / 层
    #        + adaLN_modulation
    # adaLN_modulation 因为从 11 → 16 维度变了，必须重训，整层放开
    trainable_keywords = ("lora_A", "lora_B",
                          "geometry_embedder",
                          "gcttn",
                          "norm_geom",
                          "adaLN_modulation",
                          # final_layer 的 adaLN 也归在adaLN_modulation 里，已包含
                          )
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in trainable_keywords)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable params: {n_trainable:,} / total {n_total:,}"
          f"({100.0 * n_trainable / n_total:.2f}%)")

    # 5 搬到 device
    model = model.to(device)

    # 6 EMA (deepcopy 之后再 to device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # 7 optimizer 只接收可训练参数
    lr = float(config.get('lr', 1e-4))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0)

    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)
    if bfloat_enable:
        scaler = torch.amp.GradScaler()


    # 8 (如有)加载LoRA-only 续训 ckpt
    # 注意：这里加载的是 LoRA fine-tune 的 ckpt，不是 base ckpt
    # base ckpt 通过 config['from_checkpoint] 在第2步加载
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    print('Searching for LoRA checkpoint from ', checkpoint_dir)
    start_epoch = 0
    train_steps = 0
    if os.path.isfile(latest_path):
        print("Loading LoRA fine-tune checkpoint from ", latest_path)
        latest_checkpoint = torch.load(latest_path, map_location=device, weights_only=False)

        # LoRA ckpt 只存了可训练参数，所以 strict=False
        if "model" in latest_checkpoint:
            model_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['model'].items()}
            res = model.load_state_dict(model_ckp, strict=False)
            print(f"[Resume] model: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")

            ema_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['ema'].items()}
            res = ema.load_state_dict(ema_ckp, strict=False)
            print(f"[Resume] ema: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")

        if "opt" in latest_checkpoint:
            opt.load_state_dict(latest_checkpoint['opt'])
            print("Loading optimizer params")

        if "epoch" in latest_checkpoint:
            start_epoch = latest_checkpoint['epoch'] + 1
        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint['train_steps']
        if "scaler" in latest_checkpoint and bfloat_enable:
            scaler.load_state_dict(latest_checkpoint["scaler"])

    else:
        # 没有续训 ckpt：用当前（已加载 base + 注入 LoRA 的）模型初始化 EMA
        update_ema(ema, model, decay=0)

    # ~40% speedup but might leads to worse performance depending on pytorch version
    if args.torch_compile:
        model = torch.compile(model)

    # 9 DDP, find_unused_parameters=True (因为有冻结参数)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    #=


    # ===================== LoRA修改结束：LoRA 微调流程 =====================


    # ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # requires_grad(ema, False)
    #
    # # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    # lr = float(config.get('lr', 1e-4))
    # opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    #
    # bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)
    # if bfloat_enable:
    #     scaler = torch.amp.GradScaler()
    #
    # # load existing checkpoint
    # latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    # print('Searching for model from ', checkpoint_dir)
    # start_epoch = 0
    # train_steps = 0
    # if os.path.isfile(latest_path) or config.get('from_checkpoint', 0):
    #     if os.path.isfile(latest_path) and config.get('from_checkpoint', 0):
    #         raise ValueError("Resuming from checkpoint, this might override latest.pth.tar!!")
    #     latest_path = latest_path if os.path.isfile(latest_path) else config.get('from_checkpoint', 0)
    #     print("Loading model from ", latest_path)
    #     latest_checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
    #
    #     if "model" in latest_checkpoint:
    #         model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['model'].items()}
    #         res = model.load_state_dict(model_ckp, strict=True)
    #         print("Loading model weights", res)
    #
    #         model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['ema'].items()}
    #         res = ema.load_state_dict(model_ckp, strict=True)
    #         print("Loading EMA model weights", res)
    #     else:
    #         update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    #
    #     if "opt" in latest_checkpoint:
    #         opt_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['opt'].items()}
    #         opt.load_state_dict(opt_ckp)
    #         print("Loading optimizer params")
    #
    #     if "epoch" in latest_checkpoint:
    #         start_epoch = latest_checkpoint['epoch'] + 1
    #
    #     if "train_steps" in latest_checkpoint:
    #         train_steps = latest_checkpoint["train_steps"]
    #
    #     if "scaler" in latest_checkpoint:
    #         scaler.load_state_dict(latest_checkpoint["scaler"])
    #
    # # ~40% speedup but might leads to worse performance depending on pytorch version
    # if args.torch_compile:
    #     model = torch.compile(model)
    #
    # # 修改开始
    # # model = DDP(model, device_ids=[device])
    # model = DDP(model, device_ids=[device_id])
    # # 修改结束

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = []
    test_dataset = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    goals_per_obs = int(data_config["goals_per_obs"])
                    if data_split_type == 'test':
                        goals_per_obs = 4 # standardize testing
                        # 其实不是很理解这个goals per observation是啥，看论文时注意
                    
                    if "distance" in data_config:
                        min_dist_cat=data_config["distance"]["min_dist_cat"]
                        max_dist_cat=data_config["distance"]["max_dist_cat"]
                    else:
                        min_dist_cat=config["distance"]["min_dist_cat"]
                        max_dist_cat=config["distance"]["max_dist_cat"]

                    if "len_traj_pred" in data_config:
                        len_traj_pred=data_config["len_traj_pred"]
                    else:
                        len_traj_pred=config["len_traj_pred"]

                    dataset = TrainingDataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        min_dist_cat=min_dist_cat,
                        max_dist_cat=max_dist_cat,
                        len_traj_pred=len_traj_pred,
                        context_size=config["context_size"],
                        normalize=config["normalize"],
                        goals_per_obs=goals_per_obs,
                        transform=transform,
                        predefined_index=None,
                        traj_stride=1,
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        test_dataset.append(dataset)
                    print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")

    # combine all the datasets from different robots
    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x, y, rel_t in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)
            B, T = x.shape[:2]

            # 修改开始（用原图的 context 帧提 geometry / tracks）
            # x: [B, T, 3, H, W]
            with torch.no_grad():
                geom_list = []
                # 修改提醒：想想优化（是否支持batch化geometry extraction；是否离线缓存tracks；是否只在dataset prepocessing阶段算一次）
                for b in range(B):

                    images_bt = x[b, :num_cond]  # 只取 context, 不取全部T [:num_cond, 3, H, W]
                    # vggt 输出 tracks
                    pred_tracks, pred_vis_scores, pred_confs, _, _ = predict_tracks(images_bt)

                    # 兼容 numpy / tensor
                    if not torch.is_tensor(pred_tracks):
                        pred_tracks = torch.from_numpy(pred_tracks)
                    if not torch.is_tensor(pred_vis_scores):
                        pred_vis_scores = torch.from_numpy(pred_vis_scores)
                    if not torch.is_tensor(pred_confs):
                        pred_confs = torch.from_numpy(pred_confs)

                    pred_tracks = pred_tracks.to(device)            # 位置
                    pred_vis_scores = pred_vis_scores.to(device)    # 每帧是否可见
                    pred_confs = pred_confs.to(device)              # 整个track的质量（没有这一维容易被noisy points干扰）

                    # 这里要做一个 geometry feature 组装
                    geom_b = build_geom_from_tracks(pred_tracks, pred_vis_scores, pred_confs)
                    # 目标：[N_geom, D] 这个D目前先定为4，后面看情况改成6（已加temporal）

                    geom_list.append(geom_b)

                geom = torch.stack(geom_list, dim=0)     # [B, N_geom, D]

            # 修改结束
            
            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    B, T = x.shape[:2]
                    x = x.flatten(0,1)
                    x = tokenizer.encode(x).latent_dist.sample().mul_(0.18215)
                    x = x.unflatten(0, (B, T))
                
                num_goals = T - num_cond
                x_start = x[:, num_cond:].flatten(0, 1)
                # 所以这一大段都不是很懂，就是跟x相关的，再着重看一下
                x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
                y = y.flatten(0, 1)
                rel_t = rel_t.flatten(0, 1)

                # 修改开始,每个 sample 的 geom 要复制给它的多个 goal
                geom = geom.unsqueeze(1).expand(B, num_goals, geom.shape[1], geom.shape[2]).flatten(0, 1)
                # 修改结束

                t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)

                # 修改开始
                model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t, geom=geom)
                # 修改结束
                loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            if not bfloat_enable:
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                if config.get('grad_clip_val', 0) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_val'])
                scaler.step(opt)
                scaler.update()
            
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.detach().item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = dist.get_world_size()*x_cond.shape[0]*steps_per_sec
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    # ===================== LoRA修改开始：LoRA ckpt 只存可训练参数 =====================
                    # 用一个外部 set 收集所有 requires_grad=True 的参数名
                    # 注意：DDP 包装后，参数前缀是 module., 所以从 model.module 取
                    raw_model = model.module if hasattr(model, "module") else model
                    raw_model_for_keys = getattr(raw_model, "_orig_mod", raw_model) # 兼容 torch.compile

                    trainable_names = {
                        n for n, p in raw_model_for_keys.named_parameters() if p.requires_grad
                    }

                    def _filter_trainable(state_dict, names):
                        out = {}
                        for k, v in state_dict.items():
                            k_norm = k.replace('_orig_mod.', '')
                            if k_norm in names:
                                out[k] = v
                        return out

                    full_model_sd = raw_model.state_dict()
                    full_ema_sd = ema.state_dict()

                    checkpoint = {
                        "model": _filter_trainable(full_model_sd, trainable_names),
                        "ema": _filter_trainable(full_ema_sd, trainable_names),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps,
                        "lora_rank": lora_rank,
                        "lora_alpha": lora_alpha,
                    }

                    # checkpoint = {
                    #     "model": model.module.state_dict(),
                    #     "ema": ema.state_dict(),
                    #     "opt": opt.state_dict(),
                    #     "args": args,
                    #     "epoch": epoch,
                    #     "train_steps": train_steps
                    # }
                    # ===================== LoRA修改结束：LoRA ckpt 只存可训练参数 =====================
                    if bfloat_enable:
                        checkpoint.update({"scaler": scaler.state_dict()})
                    checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
                    torch.save(checkpoint, checkpoint_path)
                    if train_steps % (10*args.ckpt_every) == 0 and train_steps > 0:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
                        torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if train_steps % args.eval_every == 0 and train_steps > 0:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, rank, config["batch_size"], config["num_workers"], latent_size, device, save_dir, args.global_seed, bfloat_enable, num_cond)
                dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


@torch.no_grad
def evaluate(model, vae, diffusion, test_dataloaders, rank, batch_size, num_workers, latent_size, device, save_dir, seed, bfloat_enable, num_cond):
    sampler = DistributedSampler(
        test_dataloaders,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )
    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True)
    score = torch.tensor(0.).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step
    for x, y, rel_t in loader:
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device).flatten(0, 1)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            samples = model_forward_wrapper((model, diffusion, vae), x, y, num_timesteps=None, latent_size=latent_size, device=device, num_cond=num_cond, num_goals=num_goals, rel_t=rel_t)
            x_start_pixels = x[:, num_cond:].flatten(0, 1)
            x_cond_pixels = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
            samples = samples * 0.5 + 0.5
            x_start_pixels = x_start_pixels * 0.5 + 0.5
            x_cond_pixels = x_cond_pixels * 0.5 + 0.5
            res = eval_model(x_start_pixels, samples)
            score += res.sum()
            n_samples += len(res)
        break
    
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(samples.shape[0], 10)):
            _, ax = plt.subplots(1,3,dpi=256)
            ax[0].imshow((x_cond_pixels[i, -1].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[1].imshow((x_start_pixels[i].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[2].imshow((samples[i].permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
            plt.savefig(f'{save_dir}/{i}.png')
            plt.close()

    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score/n_samples
    return sim_score

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
