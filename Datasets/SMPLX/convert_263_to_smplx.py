#!/usr/bin/env python3
"""
convert_263_to_smplx.py

Convert 263-d representation -> IK-recovered SMPL-X 22-joint global skeleton and visualize using Wis3D.
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from .utils.wis3d_utils import HWis3D as Wis3D
from .utils.skeleton_structure import Skeleton_SMPL22 as Skeleton_SMPLX22
from .utils.path_manager_smplx import PathManager_SMPLX
pm = PathManager_SMPLX()

# SMPL-X 22 joints 对应索引
PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]

# 你项目里的 IK 函数，需要保持可用
from .visualization.motion_process import recover_from_ric, recover_root_rot_heading_ang

def inv_transform(x, mean=None, std=None):
    """Inverse normalization for 263-d motion vectors."""
    is_numpy = False
    if isinstance(x, np.ndarray):
        is_numpy = True
        x = torch.from_numpy(x).float()

    if mean is not None and std is not None:
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float().to(x.device)
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).float().to(x.device)
        while mean.dim() < x.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        x = x * std + mean

    if is_numpy:
        x = x.cpu().numpy()
    return x

def load_motion_263(path):
    """Load 263-d motion file"""
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "motion_263" in data:
            motion = data["motion_263"]
        else:
            keys = [k for k in data.files if data[k].ndim == 2 and data[k].shape[1] == 263]
            if len(keys) == 0:
                raise ValueError(f"No array with shape [T,263] found inside {path}")
            motion = data[keys[0]]
    else:
        motion = data
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    if not (motion.ndim == 2 and motion.shape[1] == 263):
        raise AssertionError(f"Expected shape [T,263], got {motion.shape} for file {path}")
    return motion.astype(np.float32)

def convert_263_to_skeleton(motion_263, mean=None, std=None, n_joints=22, hml_type=None, device=None, visualize=False):
    """Convert 263-d motion sequence to IK-recovered SMPL-X 22-joint global skeleton."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1️⃣ 转成 tensor 并逆归一化
    sample = inv_transform(torch.from_numpy(motion_263).unsqueeze(0).unsqueeze(0).to(device), mean=mean, std=std).float() # ([B,1,T,263])

    # 2️⃣ IK恢复骨架
    sample_ = recover_from_ric(sample, n_joints, hml_type)  # ([B,1,T,22,3])

    # 3️⃣ 处理 heading
    sample_for_heading = sample_.view(-1, *sample_.shape[2:]).permute(0,2,3,1)  # ([B,22,3,T])
    heading_all = recover_root_rot_heading_ang(sample_for_heading)  # [B,1,T]
    heading_all *= 180 / torch.pi
    heading_all = heading_all.cpu().numpy().round().squeeze()

    # 4️⃣ 取 IK 输出骨架
    skeleton_global = sample_.cpu().numpy()[0,0]  # [T,22,3]

    # 5️⃣ 修正根关节 Y 高度
    root_y0 = skeleton_global[0,0,1]
    skeleton_global[:, :, 1] = skeleton_global[:, :, 1] - skeleton_global[0,0,1] + root_y0

    # 6️⃣ 可视化骨架
    if visualize:
        vis = Wis3D(pm.outputs / 'wis3d', f'SMPLX-sequence-demo')

        # 1️⃣ 添加骨架动画
        vis.add_motion_skel(
            joints=skeleton_global,  # [T,22,3]
            bones=Skeleton_SMPLX22.bones,
            colors=Skeleton_SMPLX22.bone_colors,
            name='demo-skeleton',
            offset=0,
        )

        # 2️⃣ 添加根关节轨迹（以第0号关节为例）
        root_positions = torch.from_numpy(skeleton_global[:,0,:]).float()  # [T,3]
        vis.add_traj_xz(root_positions, name="root_traj")
        add_fixed_text(vis, text="a person throws their body forward and then to the side in an expressive way.", num_frames=skeleton_global.shape[0])

    return skeleton_global  # [T,22,3]

def add_fixed_text(vis, text: str, num_frames: int, offset: int = 0):
    """
    在 Wis3D 中的所有帧添加相同的文字（例如序列名、提示文字）
    """
    fake_verts = np.array([[0, 0, 0]])  # 虚拟点
    for i in range(num_frames):
        vis.set_scene_id(i + offset)
        vis.add_point_cloud(
            vertices=fake_verts,
            colors=None,
            name=text,   # 文字内容
        )
    vis.set_scene_id(0)


def process_file(poses_path, mean=None, std=None, hml_type=None, visualize=False):
    print(f"Processing: {poses_path}")
    motion_263 = load_motion_263(poses_path)
    skeleton_global = convert_263_to_skeleton(
        motion_263, mean=mean, std=std, hml_type=hml_type, visualize=visualize
    )
    print(f"Skeleton shape: {skeleton_global.shape}")  # [T,22,3]
    return skeleton_global

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root,f)
            file_path.append(fullname)
    return file_path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", type=str, required=True, help="path to .npy/.npz file or folder if --is_folder")
    p.add_argument("--is_folder", action="store_true", help="treat --poses as folder and process all .npy inside")
    p.add_argument("--mean", type=str, default=None, help="path to mean.npy for 263-d data")
    p.add_argument("--std", type=str, default=None, help="path to std.npy for 263-d data")
    p.add_argument("--hml_type", type=str, default=None)
    p.add_argument("--visualize", action="store_true", help="enable Wis3D visualization")
    return p.parse_args()

if __name__=="__main__":
    args = parse_args()
    mean = np.load(args.mean) if args.mean else None
    std = np.load(args.std) if args.std else None

    if args.is_folder:
        allfiles = findAllFile(args.poses)
        npy_list = [f for f in allfiles if f.endswith(".npy") or f.endswith(".npz")]
        for f in tqdm(npy_list):
            try:
                process_file(f, mean=mean, std=std, hml_type=args.hml_type, visualize=args.visualize)
            except Exception as e:
                print(f"❌ Failed {f}: {e}")
    else:
        process_file(args.poses, mean=mean, std=std, hml_type=args.hml_type, visualize=args.visualize)

    print("✅ Done.")

# python -m Datasets.SMPLX.convert_263_to_smplx \
#     --poses Datasets/SMPLX/demo_263_output/test001.npy \
#     --visualize \
#     --mean Datasets/SMPLX/demo_263_output/mean_std/mean.npy \
#     --std Datasets/SMPLX/demo_263_output/mean_std/std.npy
