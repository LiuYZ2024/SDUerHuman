#!/usr/bin/env python3
"""
convert_272_to_smplx_v2.py

Convert MotionStreamer 272-d representation -> SMPL-like parameters and save as .npz
Support B motions in a single file and optional Wis3D visualization of all sequences together.

wis3d --vis_dir Datasets/SMPLX/data_output/wis3d --host 0.0.0.0 --port 19090

python Datasets/SMPLX/convert_272_to_smplx_vis_B.py \
    --poses Datasets/SMPLX/demo_272_output/test272_1.npy \
    --output Datasets/SMPLX/data_converted_to_smplx/ \
    --mean Datasets/SMPLX/demo_272_output/mean_std/Mean.npy \
    --std Datasets/SMPLX/demo_272_output/mean_std/Std.npy \
    --visualize \
    --every_n_frame 1 \
    --idx 1 \
    --text "['a person throws their body forward and then to the side in an expressive way', 'the sim walks backwards down the plane.', 'a person raises their hand up towards their head and then outs it back down', 'a person standing with hands up by chest, right  leg steps back.', 'the sim walks forward, reaching the end of the plane and looping backwards again.', 'the left hand goes upward towards the face and back down to the hip.']"


"""

import os
import argparse
import ast
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
from pathlib import Path

from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle

# Things you don't need to care about. They are just for driving the tutorials.
from utils.path_manager_smplx import PathManager_SMPLX
from utils.wis3d_utils import HWis3D as Wis3D
# wis3d --vis_dir /home/liu/lyz/Human/Datasets/SMPLX/data_output/wis3d --host 0.0.0.0 --port 19090

from utils.skeleton_structure import Skeleton_SMPL22 as Skeleton_SMPLX22
from smplx import SMPLX

pm = PathManager_SMPLX()

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    return np.array(R_total)


def rotations_matrix_to_smplx85(rotations_matrix, translation):
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
    trans = translation.reshape(nfrm, 3)
    base = axis_angle
    if base.shape[1] < 72:
        pad = np.zeros((nfrm, 72 - base.shape[1]))
        base = np.concatenate([base, pad], axis=-1)
    out = base
    if out.shape[1] < 75:
        out = np.concatenate([out, np.zeros((nfrm, 75 - out.shape[1]))], axis=-1)
    out[:, 72:75] = trans
    return out


def recover_from_local_rotation(final_x, njoint):
    nfrm, _ = final_x.shape
    jr_block = final_x[:, 8 + 6 * njoint: 8 + 12 * njoint]
    jr_reshaped = jr_block.reshape(nfrm, njoint, 6)
    rotations_matrix = np.zeros((nfrm, njoint, 3, 3), dtype=np.float32)
    for i in range(nfrm):
        for j in range(njoint):
            r6 = jr_reshaped[i, j]
            r6_t = torch.from_numpy(r6).float()
            Rm = rotation_6d_to_matrix(r6_t).numpy()
            rotations_matrix[i, j] = Rm

    global_heading_diff_rot = final_x[:, 2:8]
    global_heading_rot_mats = np.zeros((nfrm, 3, 3), dtype=np.float32)
    for i in range(nfrm):
        R = torch.from_numpy(global_heading_diff_rot[i]).float()
        global_heading_rot_mats[i] = rotation_6d_to_matrix(R).numpy()

    global_heading_rot = accumulate_rotations(global_heading_rot_mats)
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    rotations_matrix[:, 0, ...] = np.matmul(inv_global_heading_rot, rotations_matrix[:, 0, ...])

    velocities_root_xy_no_heading = final_x[:, :2]
    positions_no_heading = final_x[:, 8: 8 + 3 * njoint].reshape(nfrm, njoint, 3)
    height = positions_no_heading[:, 0, 1]
    velocities_root_xyz_no_heading = np.zeros((nfrm, 3), dtype=np.float32)
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    if nfrm > 1:
        for t in range(1, nfrm):
            velocities_root_xyz_no_heading[t] = inv_global_heading_rot[t - 1].dot(
                velocities_root_xyz_no_heading[t])
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height
    smplx_vec = rotations_matrix_to_smplx85(rotations_matrix, root_translation)
    return smplx_vec


def load_motion_272_multi(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "motion_272" in data:
            motion = data["motion_272"]
        else:
            keys = [k for k in data.files if data[k].ndim >= 2 and (data[k].shape[-1] == 272 or data[k].shape[1] == 272)]
            if len(keys) == 0:
                raise ValueError(f"No array with shape containing 272 found inside {path}. Keys: {data.files}")
            motion = data[keys[0]]
    else:
        motion = data

    motion = np.array(motion, dtype=np.float32)
    if motion.ndim == 2 and motion.shape[1] == 272:
        motion = motion[None, :, :]
    elif motion.ndim == 3:
        if motion.shape[1] == 272 and motion.shape[2] != 272:
            motion = np.swapaxes(motion, 1, 2)
    elif motion.ndim == 4:
        if motion.shape[1] == 272 and motion.shape[2] == 1:
            motion = motion.squeeze(2).transpose(0, 2, 1)
    else:
        raise AssertionError(f"Unexpected shape {motion.shape}")
    return motion


def save_smplx_params(out_path, root_orient, pose_body, trans, extra=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             root_orient=root_orient,
             pose_body=pose_body,
             trans=trans,
             extra=extra)
    print(f"Saved SMPL-X-style params to {out_path}")


def add_fixed_text(vis, text: str, num_frames: int, offset: int = 0):
    fake_verts = np.array([[0, 0, 0]])
    for i in range(num_frames):
        vis.set_scene_id(i + offset)
        vis.add_point_cloud(vertices=fake_verts, colors=None, name=text)
    vis.set_scene_id(0)

def visualize_smplx_sequence(npz_path, gender='neutral', every_n_frame=1, vis=None, b=0, text=None):
    """
    可视化 SMPL-X 动作序列

    Args:
        npz_path (str or Path): SMPL-X 参数 npz 文件路径
        gender (str): 'neutral' / 'male' / 'female'
        every_n_frame (int): 可选，下采样可视化的帧间隔（默认为 1 表示全帧）
    """

    # 加载数据
    data = np.load(npz_path, allow_pickle=True)

    # 获取参数
    betas = torch.tensor(data.get('betas', np.zeros((1, 10))), dtype=torch.float32)
    pose_body = torch.tensor(data.get('pose_body', np.zeros((1, 21, 3))), dtype=torch.float32)
    global_orient = torch.tensor(data.get('root_orient', np.zeros((1, 3))), dtype=torch.float32).unsqueeze(1)
    transl = torch.tensor(data.get('trans', np.zeros((1, 3))), dtype=torch.float32)

    # raise RuntimeError(body_pose.shape)

    n_frames = pose_body.shape[0]
    print(f"Total frames: {n_frames}, betas: {betas.shape}")

    # 只取部分帧（如果需要下采样）
    idx = np.arange(0, n_frames, every_n_frame)
    pose_body = pose_body[idx]
    global_orient = global_orient[idx]
    transl = transl[idx]
    n_frames = len(idx)

    body_models = SMPLX(
        model_path = pm.root_dataset / 'smplx' / 'models' / 'smplx',
        gender='neutral',
        use_hands=False,
        use_face=False,
        use_face_contour=False,
        num_pca_comps=0,
        use_pca=False,
        ext='npz',
        batch_size=n_frames
    )

    # Inference
    smpl_out = body_models(
        betas=betas.expand(n_frames, -1),
        global_orient=global_orient,
        body_pose=pose_body,
        transl=transl,
    )

    joints = smpl_out.joints
    verts = smpl_out.vertices
    faces = body_models.faces

    # 可视化
    vis.add_motion_mesh(
        verts=verts,
        faces=faces,
        name=f'{b}-surface',
        offset=0,
    )
    vis.add_motion_skel(
        joints=joints[:, :22],
        bones=Skeleton_SMPLX22.bones,
        colors=Skeleton_SMPLX22.bone_colors,
        name=f'{b}-skeleton',
        offset=0,
    )

    root_positions = transl
    vis.add_traj_xz(root_positions, name=f"{b}-root_traj")
    if text is not None:
        add_fixed_text(vis, text=text[b], num_frames=n_frames)

    print(f"✅ Visualization ready at Wis3D: {pm.outputs / 'wis3d'}")

def process_file(poses_path, output_root, mean=None, std=None, visualize=False, text_list=None, every_n_frame=1, vis_idx=None):
    print(f"Processing: {poses_path}")
    motions = load_motion_272_multi(poses_path)
    B, T, D = motions.shape
    print(f"Loaded {B} motions, each {T} frames, {D} dims.")

    out_paths = []
    from pathlib import Path
    vis = Wis3D(pm.outputs / 'wis3d', f'SMPLX-sequence-272-{vis_idx}')

    for b in range(B):
        motion_272 = motions[b].copy()
        if mean is not None and std is not None:
            motion_272 = motion_272 * std + mean

        recovered = recover_from_local_rotation(motion_272, njoint=22)
        axis_angle = recovered[:, :72].reshape(recovered.shape[0], 24, 3)
        root_orient = axis_angle[:, 0, :]
        pose_body = axis_angle[:, 1:22, :].reshape(recovered.shape[0], -1)
        trans = recovered[:, 72:75]

        base = os.path.splitext(os.path.basename(poses_path))[0]
        out_dir = output_root if os.path.isdir(output_root) else os.path.dirname(output_root) or "."
        outname = f"{base}_seq{b:03d}_smplx_params.npz"
        out_path = os.path.join(out_dir, outname)
        save_smplx_params(out_path, root_orient, pose_body, trans, extra={"full_recovered_shape": recovered.shape})
        out_paths.append(out_path)

        # -----------------------------
        # visualization: all sequences in one Wis3D
        # -----------------------------
        visualize_smplx_sequence(npz_path=pm.root_dataset/f'data_converted_to_smplx/test272_1_seq00{b}_smplx_params.npz', every_n_frame=1, vis=vis, b=b, text=text_list)

    return out_paths


def parse_text_arg(raw):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1 and isinstance(raw[0], str) and raw[0].strip().startswith('['):
            try:
                parsed = ast.literal_eval(raw[0])
                return [str(x) for x in parsed]
            except Exception:
                pass
        return [str(x) for x in raw]
    s = str(raw)
    if os.path.exists(s) and s.endswith('.txt'):
        with open(s, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    if s.strip().startswith('['):
        try:
            parsed = ast.literal_eval(s)
            return [str(x) for x in parsed]
        except Exception:
            pass
    return [s]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", type=str, required=True)
    p.add_argument("--output", type=str, default="./data_converted_to_smplx")
    p.add_argument("--is_folder", action="store_true")
    p.add_argument("--mean", type=str, default=None)
    p.add_argument("--std", type=str, default=None)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--every_n_frame", type=int, default=1)
    p.add_argument("--text", nargs="+", type=str, default=None)
    p.add_argument("--idx", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mean = np.load(args.mean) if args.mean else None
    std = np.load(args.std) if args.std else None
    text_list = parse_text_arg(args.text)

    if args.is_folder:
        allfiles = findAllFile(args.poses)
        npy_list = [f for f in allfiles if f.endswith(".npy") or f.endswith(".npz")]
        for f in tqdm(npy_list):
            rel = os.path.relpath(f, args.poses)
            out_path_dir = os.path.join(args.output, os.path.splitext(rel)[0])
            os.makedirs(out_path_dir, exist_ok=True)
            process_file(f, out_path_dir, mean=mean, std=std,
                         visualize=args.visualize, text_list=text_list, every_n_frame=args.every_n_frame, vis_idx=args.idx)
    else:
        process_file(args.poses, args.output, mean=mean, std=std,
                     visualize=args.visualize, text_list=text_list, every_n_frame=args.every_n_frame, vis_idx=args.idx)
