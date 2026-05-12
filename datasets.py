# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import numpy as np
import torch
import os
from PIL import Image
from typing import Tuple
import yaml
import pickle
import tqdm
from torch.utils.data import Dataset
from misc import (
    angle_difference,
    get_data_path,
    get_delta_np,
    normalize_data,
    to_local_coords,
    build_geom_from_tracks,
    safe_traj_name,
)

# =============================== 修改开始：geom cache 加载工具 ===============================

# 与 precompute 时的 --max-query-pts 对齐：目前用的是 128，所以这里也是 128。
# 如果以后重跑 precompute 改成更大，这里也要同步改。
N_GEOM_FIXED = 128


def load_geom_npz(cache_root: str,
                  dataset_name: str,
                  split: str,
                  traj_name: str,
                  curr_time: int,
                  max_tokens: int = N_GEOM_FIXED) -> torch.Tensor:
    """
    从离线 cache 加载 (tracks, vis, confs) 并组装成 geom: [max_tokens, 6]。

    - 找不到 cache 文件 / 加载报错时，返回全零 geom（不抛异常，让训练正常进行，
      相当于这条样本"暂时不提供 geom 信息"）。
    - 多余的点 top-k by confidence，已在 build_geom_from_tracks 内做；
      不足 max_tokens 时这里 pad 0。
    - tracks 坐标系由 npz 内的 coord_h / coord_w 决定（precompute 时存的，
      默认 518）。这样下游不用关心 VGGT 内部分辨率。
    """
    stem = f"{safe_traj_name(traj_name)}__t{curr_time}"
    npz_path = os.path.join(cache_root, dataset_name, split, f"{stem}.npz")

    if not os.path.isfile(npz_path):
        return torch.zeros(max_tokens, 6, dtype=torch.float32)

    try:
        data = np.load(npz_path)
        tracks = torch.from_numpy(data["tracks"])  # [Tc, Nq, 2]
        vis = torch.from_numpy(data["vis"])  # [Tc, Nq]
        confs = torch.from_numpy(data["confs"])  # [Nq]

        # 向后兼容：旧 cache 没存 coord_h/coord_w，默认 518（=vggt_resolution）
        coord_h = float(data["coord_h"]) if "coord_h" in data.files else 518.0
        coord_w = float(data["coord_w"]) if "coord_w" in data.files else 518.0

        geom = build_geom_from_tracks(
            tracks, vis, confs,
            max_tokens=max_tokens,
            coord_h=coord_h, coord_w=coord_w,
        )
    except Exception:
        return torch.zeros(max_tokens, 6, dtype=torch.float32)

    # pad / truncate 到 max_tokens
    n = geom.shape[0]
    if n >= max_tokens:
        geom = geom[:max_tokens]
    else:
        pad = torch.zeros(max_tokens - n, 6, dtype=geom.dtype)
        geom = torch.cat([geom, pad], dim=0)
    return geom.float()


# =============================== 修改结束 ===============================


class BaseDataset(Dataset):
    def __init__(
            self,
            data_folder: str,
            data_split_folder: str,
            dataset_name: str,
            image_size: Tuple[int, int],
            min_dist_cat: int,
            max_dist_cat: int,
            len_traj_pred: int,
            traj_stride: int,
            context_size: int,
            transform: object,
            traj_names: str,
            normalize: bool = True,
            predefined_index: list = None,
            goals_per_obs: int = 1,
            # === 修改开始：geom cache 配置 ===
            geom_cache_root: str = None,
            # === 修改结束 ===
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs

        # === 修改开始：geom cache 配置 ===
        self.geom_cache_root = geom_cache_root
        # 从 data_split_folder 路径推断 split 名（"train" / "test"），
        # 用来在 cache 目录里找对应子目录。
        _split_basename = os.path.basename(data_split_folder.rstrip("/")).lower()
        if "train" in _split_basename:
            self._split_for_geom = "train"
        elif "test" in _split_basename or "val" in _split_basename:
            self._split_for_geom = "test"
        else:
            # fallback: 用 split folder 的最后一段
            self._split_for_geom = _split_basename if _split_basename else "train"
        # === 修改结束 ===

        traj_names_file = os.path.join(data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        # load data/data_config.yaml
        with open("config/data_config.yaml", "r") as f:
            all_data_config = yaml.safe_load(f)

        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.data_config = all_data_config[self.dataset_name]
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        if predefined_index:
            print(f"****** Using a predefined evaluation index... {predefined_index}******")
            with open(predefined_index, "rb") as f:
                self.index_to_data = pickle.load(f)
                return
        else:
            print("****** Evaluating from NON PREDEFINED index... ******")
            index_to_data_path = os.path.join(
                self.data_split_folder,
                f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}.pkl",
            )

            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))

        return samples_index, goals_index

    def _get_trajectory(self, trajectory_name):
        with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        for k, v in traj_data.items():
            traj_data[k] = v.astype('float')
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]

        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw = angle_difference(yaw[0], goal_yaw)

        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
            goal_pos[:, :2] /= self.data_config["metric_waypoint_spacing"]

        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
        return actions, goal_pos

        # === 修改开始：通用 geom 加载辅助 ===

    def _load_geom(self, f_curr: str, curr_time: int) -> torch.Tensor:
        """根据 self.geom_cache_root 决定是从 cache 加载，还是返回全零。"""
        if self.geom_cache_root is None:
            return torch.zeros(N_GEOM_FIXED, 6, dtype=torch.float32)
        return load_geom_npz(
            self.geom_cache_root,
            self.dataset_name,
            self._split_for_geom,
            f_curr,
            int(curr_time),
            max_tokens=N_GEOM_FIXED,
        )
    # === 修改结束 ===


class TrainingDataset(BaseDataset):
    def __init__(
            self,
            data_folder: str,
            data_split_folder: str,
            dataset_name: str,
            image_size: Tuple[int, int],
            min_dist_cat: int,
            max_dist_cat: int,
            len_traj_pred: int,
            traj_stride: int,
            context_size: int,
            transform: object,
            traj_names: str = 'traj_names.txt',
            normalize: bool = True,
            predefined_index: list = None,
            goals_per_obs: int = 1,
            # === 修改开始 ===
            geom_cache_root: str = None,
            # === 修改结束 ===
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
                         len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index,
                         goals_per_obs,
                         geom_cache_root=geom_cache_root)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs))
            goal_time = (curr_time + goal_offset).astype('int')
            rel_time = (goal_offset).astype('float') / (128.)  # TODO: refactor, currently a fixed const

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]

            obs_image = torch.stack(
                [self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

            # === 修改开始：加载 geom ===
            geom = self._load_geom(f_curr, curr_time)  # [N_GEOM_FIXED, 6]
            # === 修改结束 ===

            return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32),
                geom,  # === 修改开始：新增 geom ===
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


class EvalDataset(BaseDataset):
    def __init__(
            self,
            data_folder: str,
            data_split_folder: str,
            dataset_name: str,
            image_size: Tuple[int, int],
            min_dist_cat: int,
            max_dist_cat: int,
            len_traj_pred: int,
            traj_stride: int,
            context_size: int,
            transform: object,
            traj_names: str,
            normalize: bool = True,
            predefined_index: list = None,
            goals_per_obs: int = 1,
            # === 修改开始 ===
            geom_cache_root: str = None,
            # === 修改结束 ===
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
                         len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index,
                         goals_per_obs,
                         geom_cache_root=geom_cache_root)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, _, _ = self.index_to_data[i]
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))

            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack(
                [self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            pred_image = torch.stack(
                [self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in pred])

            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            actions, _ = self._compute_actions(curr_traj_data, curr_time,
                                               np.array([curr_time + 1]))  # last argument is dummy goal
            actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            delta = get_delta_np(actions)

            # === 修改开始：加载 geom ===
            geom = self._load_geom(f_curr, curr_time)  # [N_GEOM_FIXED, 6]
            # === 修改结束 ===

            return (
                torch.tensor([i], dtype=torch.float32),  # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
                geom,  # === 修改开始：新增 geom ===
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


class TrajectoryEvalDataset(BaseDataset):
    """
    Planning eval 用的 dataset。
    目前 planning 路径暂未接入 geom（geom 始终为全零），所以这里
    保持原有 4 元组返回，不引入 BC break。如果以后要接入 planning 的 geom，
    可以再在这里增加 geom 输出，并在 planning_eval.py 透传。
    """

    def __init__(
            self,
            data_folder: str,
            data_split_folder: str,
            dataset_name: str,
            image_size: Tuple[int, int],
            min_dist_cat: int,
            max_dist_cat: int,
            len_traj_pred: int,
            traj_stride: int,
            context_size: int,
            transform: object,
            traj_names: str,
            normalize: bool = True,
            predefined_index: list = None,
            goals_per_obs: int = 1,
            geom_cache_root: str = None,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
                         len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index,
                         goals_per_obs,
                         geom_cache_root=geom_cache_root)

    def _sample_goal(self, trajectory_name, curr_time, min_goal_dist, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
        goal_time = curr_time + int(goal_offset)
        return trajectory_name, goal_time, False

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            f_goal, goal_time, _ = self._sample_goal(f_curr, curr_time, min_goal_dist, max_goal_dist)

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            context = [(f_curr, t) for t in context_times]

            obs_image = torch.stack(
                [self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            goal_image = self.transform(Image.open(get_data_path(self.data_folder, f_goal, goal_time))).unsqueeze(0)
            curr_traj_data = self._get_trajectory(f_curr)

            actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, np.array([goal_time]))

            return (
                torch.tensor([i], dtype=torch.float32),  # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)