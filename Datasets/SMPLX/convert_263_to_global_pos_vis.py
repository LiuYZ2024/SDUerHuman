#!/usr/bin/env python3
"""
convert_263_to_smplx.py

Convert 263-d representation -> IK-recovered SMPL-X 22-joint global skeleton and visualize using Wis3D.

python -m Datasets.SMPLX.convert_263_to_global_pos_vis \
    --poses Datasets/SMPLX/demo_263_output/test263_1.npy \
    --visualize \
    --mean Datasets/SMPLX/demo_263_output/mean_std/mean.npy \
    --std Datasets/SMPLX/demo_263_output/mean_std/std.npy \
    --idx 1 \
    --text "['a person throws their body forward and then to the side in an expressive way', 'the sim walks backwards down the plane.', 'a person raises their hand up towards their head and then outs it back down', 'a person standing with hands up by chest, right  leg steps back.', 'the sim walks forward, reaching the end of the plane and looping backwards again.', 'the left hand goes upward towards the face and back down to the hip.']"

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


def convert_263_to_skeleton(motion_263, mean=None, std=None, n_joints=22, hml_type=None, device=None, visualize=False):
    """Convert 263-d motion sequence to IK-recovered SMPL-X 22-joint global skeleton."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 转成 tensor 并逆归一化
    sample = inv_transform(torch.from_numpy(motion_263).unsqueeze(0).unsqueeze(0).to(device), mean=mean, std=std).float() # ([B,1,T,263])

    # IK恢复骨架
    sample_ = recover_from_ric(sample, n_joints, hml_type)  # ([B,1,T,22,3])

    # 处理 heading
    sample_for_heading = sample_.view(-1, *sample_.shape[2:]).permute(0,2,3,1)  # ([B,22,3,T])
    heading_all = recover_root_rot_heading_ang(sample_for_heading)  # [B,1,T]
    heading_all *= 180 / torch.pi
    heading_all = heading_all.cpu().numpy().round().squeeze()

    # 取 IK 输出骨架
    skeleton_global = sample_.cpu().numpy()[0,0]  # [T,22,3]

    # 修正根关节 Y 高度
    root_y0 = skeleton_global[0,0,1]
    skeleton_global[:, :, 1] = skeleton_global[:, :, 1] - skeleton_global[0,0,1] + root_y0

    # 可视化骨架
    if visualize:
        vis = Wis3D(pm.outputs / 'wis3d', f'SMPLX-sequence-demo')

        # 添加骨架动画
        vis.add_motion_skel(
            joints=skeleton_global,  # [T,22,3]
            bones=Skeleton_SMPLX22.bones,
            colors=Skeleton_SMPLX22.bone_colors,
            name='demo-skeleton',
            offset=0,
        )

        # 添加根关节轨迹（以第0号关节为例）
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


def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root,f)
            file_path.append(fullname)
    return file_path

def load_motion_263(path):
    """Load 263-d motion file, support multiple sequences."""
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "motion_263" in data:
            motion = data["motion_263"]
        else:
            keys = [k for k in data.files if data[k].ndim >= 2]
            motion = data[keys[0]]
    else:
        motion = data

    motion = np.array(motion, dtype=np.float32)

    # --- 维度自动纠正 ---
    if motion.ndim == 2 and motion.shape[1] == 263:
        # [T,263] 单动作
        motion = motion[None, :, :]  # [1,T,263]
    elif motion.ndim == 3:
        if motion.shape[1] == 263 and motion.shape[2] != 263:
            # [B,263,T] or [B,263,1,T]
            motion = np.swapaxes(motion, 1, -1)  # -> [B,T,263]
        elif motion.shape[-1] == 263:
            pass  # already [B,T,263]
        else:
            raise AssertionError(f"Unrecognized motion shape {motion.shape}")
    elif motion.ndim == 4:
        # 兼容 [B,263,1,T]
        if motion.shape[1] == 263 and motion.shape[2] == 1:
            motion = motion.squeeze(2).transpose(0, 2, 1)  # [B,T,263]
        else:
            raise AssertionError(f"Unexpected shape {motion.shape}")
    else:
        raise AssertionError(f"Expected ndim 2~4, got {motion.ndim}, shape={motion.shape}")

    return motion  # [B,T,263]


def process_file(poses_path, mean=None, std=None, hml_type=None, visualize=False, text=None, idx=0):
    print(f"Processing: {poses_path}")
    motion_263_all = load_motion_263(poses_path)  # [B,T,263]
    B, T, D = motion_263_all.shape
    print(f"Loaded {B} motions, each {T} frames, {D} dims.")

    skeletons_all = []

    # 初始化 Wis3D
    vis = Wis3D(pm.outputs / 'wis3d_263', f'SMPLX-sequence-263-{idx}') if visualize else None

    for b in range(B):
        motion_263 = motion_263_all[b]
        skeleton_global = convert_263_to_skeleton(
            motion_263, mean=mean, std=std, hml_type=hml_type, visualize=False
        )
        skeletons_all.append(skeleton_global)

        if visualize:
            vis.add_motion_skel(
                joints=skeleton_global,
                bones=Skeleton_SMPLX22.bones,
                colors=Skeleton_SMPLX22.bone_colors,
                name=f'{b}-skeleton',
                offset=0,
            )
            root_positions = torch.from_numpy(skeleton_global[:, 0, :]).float()
            vis.add_traj_xz(root_positions, name=f"{b}-root_traj")
            if text is not None:
                add_fixed_text(vis, text=text[b], num_frames=skeleton_global.shape[0])

    if visualize:
        print("✅ Visualization ready.")

    return skeletons_all

import ast  # 用于安全解析字符串表达式

def load_text_arg(text_arg):
    """
    支持以下输入格式：
    Python 列表字符串：--text "['desc1','desc2',...]"
    文本文件路径：--text path/to/descriptions.txt （每行一句）
    None：返回 None
    """
    if text_arg is None:
        return None

    # 如果是文件路径
    if os.path.exists(text_arg) and text_arg.endswith(".txt"):
        with open(text_arg, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    # 尝试解析 Python 列表字符串
    try:
        parsed = ast.literal_eval(text_arg)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass

    # 其他情况统一返回单元素列表
    return [text_arg]



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", type=str, required=True, help="path to .npy/.npz file or folder if --is_folder")
    p.add_argument("--is_folder", action="store_true", help="treat --poses as folder and process all .npy inside")
    p.add_argument("--mean", type=str, default=None, help="path to mean.npy for 263-d data")
    p.add_argument("--std", type=str, default=None, help="path to std.npy for 263-d data")
    p.add_argument("--hml_type", type=str, default=None)
    p.add_argument("--visualize", action="store_true", help="enable Wis3D visualization")
    p.add_argument("--text", type=str, default=None, help="text with descriptions for each motion sequence")
    p.add_argument("--idx", type=int, default=0, help="")
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
                process_file(f, mean=mean, std=std, hml_type=args.hml_type, visualize=args.visualize, text=load_text_arg(args.text), idx=args.idx)
            except Exception as e:
                print(f"❌ Failed {f}: {e}")
    else:
        process_file(args.poses, mean=mean, std=std, hml_type=args.hml_type, visualize=args.visualize, text=load_text_arg(args.text), idx=args.idx)

    print("✅ Done.")
