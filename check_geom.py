import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from datasets import load_geom_npz

CACHE_ROOT = "/media/jingzhang/T7 Shield/datasets/scand_geom_cache"
DATASET = "scand"   # 改成你 yaml 里 datasets 下的那个 key
SPLIT = "train"

# 从 cache 目录里随便挑一个文件
import glob
files = glob.glob(os.path.join(CACHE_ROOT, DATASET, SPLIT, "*.npz"))
files = [f for f in files if not f.endswith(".failed.npz")]
assert len(files) > 0, "没找到成功 cache"
print(f"Total success cache: {len(files)}")

# 解析出 (traj, t)
fname = os.path.basename(files[0]).replace(".npz", "")
traj_name, t_str = fname.rsplit("__t", 1)
curr_time = int(t_str)
print(f"Testing: traj={traj_name}, t={curr_time}")

geom = load_geom_npz(CACHE_ROOT, DATASET, SPLIT, traj_name, curr_time)

print(f"geom.shape = {geom.shape}")             # 期望 [128, 6]
print(f"x_last range  = [{geom[:,0].min():.3f}, {geom[:,0].max():.3f}]")  # 期望 ~[0,1]
print(f"y_last range  = [{geom[:,1].min():.3f}, {geom[:,1].max():.3f}]")  # 期望 ~[0,1]
print(f"dx     range  = [{geom[:,2].min():.3f}, {geom[:,2].max():.3f}]")  # 期望 ~[-1,1] 但通常很小
print(f"dy     range  = [{geom[:,3].min():.3f}, {geom[:,3].max():.3f}]")
print(f"vis    range  = [{geom[:,4].min():.3f}, {geom[:,4].max():.3f}]")  # 期望 [0,1]
print(f"conf   range  = [{geom[:,5].min():.3f}, {geom[:,5].max():.3f}]")  # 期望 [0,1]

# 统计有效点（非 padding）
nonzero = (geom.abs().sum(dim=-1) > 1e-8).sum().item()
print(f"非 padding 点数: {nonzero} / 128")