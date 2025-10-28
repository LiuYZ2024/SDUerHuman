#!/usr/bin/env python3
"""
convert_272_to_smplx_v2.py

Convert MotionStreamer 272-d representation -> SMPL-like parameters and save as .npz

Usage examples:
    python convert_272_to_smplx_v2.py --poses data_converted_to_272/P1_motion272.npy --output data_converted_to_smplx/
    python convert_272_to_smplx_v2.py --is_folder --poses data_converted_to_272/ --output data_converted_to_smplx/
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch

from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle

def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

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
    jr_block = final_x[:, 8 + 6 * njoint : 8 + 12 * njoint]
    jr_reshaped = jr_block.reshape(nfrm, njoint, 6)
    rotations_matrix = np.zeros((nfrm, njoint, 3, 3), dtype=np.float32)
    for i in range(nfrm):
        for j in range(njoint):
            r6 = jr_reshaped[i, j]
            # Rm = rotation_6d_to_matrix(r6)
            r6_t = torch.from_numpy(r6).float()
            Rm = rotation_6d_to_matrix(r6_t).numpy()
            rotations_matrix[i, j] = Rm

    global_heading_diff_rot = final_x[:, 2:8]
    global_heading_rot_mats = np.zeros((nfrm, 3, 3), dtype=np.float32)
  
    for i in range(nfrm):
        R = torch.from_numpy(global_heading_diff_rot[i]).float()
        global_heading_rot_mats[i] = rotation_6d_to_matrix(R).numpy()

    global_heading_rot = accumulate_rotations(global_heading_rot_mats)
    inv_global_heading_rot = np.transpose(global_heading_rot, (0,2,1))
    rotations_matrix[:,0,...] = np.matmul(inv_global_heading_rot, rotations_matrix[:,0,...])

    velocities_root_xy_no_heading = final_x[:, :2]
    positions_no_heading = final_x[:, 8 : 8 + 3 * njoint].reshape(nfrm, njoint, 3)
    height = positions_no_heading[:, 0, 1]
    velocities_root_xyz_no_heading = np.zeros((nfrm, 3), dtype=np.float32)
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    if nfrm > 1:
        for t in range(1, nfrm):
            velocities_root_xyz_no_heading[t] = inv_global_heading_rot[t-1].dot(velocities_root_xyz_no_heading[t])
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height
    smplx_vec = rotations_matrix_to_smplx85(rotations_matrix, root_translation)
    return smplx_vec

def load_motion_272(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "motion_272" in data:
            motion = data["motion_272"]
        else:
            keys = [k for k in data.files if data[k].ndim == 2 and data[k].shape[1] == 272]
            if len(keys) == 0:
                raise ValueError(f"No array with shape [T,272] found inside {path}. Keys: {data.files}")
            motion = data[keys[0]]
    else:
        motion = data
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
        print(f"⚙️  Removed leading batch dim, new shape: {motion.shape}")
    if not (motion.ndim == 2 and motion.shape[1] == 272):
        raise AssertionError(f"Expected shape [T,272], got {motion.shape} for file {path}")
    return motion.astype(np.float32)

def save_smplx_params(out_path, root_orient, pose_body, trans, extra=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             root_orient=root_orient,
             pose_body=pose_body,
             trans=trans,
             extra=extra)
    print(f"💾 Saved SMPL-X-style params to {out_path}")

def process_file(poses_path, output_path, mean=None, std=None):
    print(f"Processing: {poses_path}")
    motion_272 = load_motion_272(poses_path)
    if mean is not None and std is not None:
        motion_272 = motion_272 * std + mean  # 反标准化

    recovered = recover_from_local_rotation(motion_272, njoint=22)
    # raise RuntimeError(recovered.shape) # debug: (240, 75)
    # raise RuntimeError(recovered[0])
    if recovered.shape[1] < 75:
        raise RuntimeError(f"Recovered vector too short: {recovered.shape}")

    axis_angle = recovered[:, :72].reshape(recovered.shape[0], 24, 3)
    root_orient = axis_angle[:, 0, :]
    pose_body = axis_angle[:, 1:22, :].reshape(recovered.shape[0], -1)
    trans = recovered[:, 72:75]

    base = os.path.splitext(os.path.basename(poses_path))[0]
    if os.path.isdir(output_path):
        out_file = os.path.join(output_path, base + "_smplx_params.npz")
    else:
        out_file = output_path
    save_smplx_params(out_file, root_orient, pose_body, trans, extra={"full_recovered_shape": recovered.shape})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", type=str, required=True, help="path to .npy/.npz file or folder if --is_folder")
    p.add_argument("--output", type=str, default="./data_converted_to_smplx", help="output file or folder")
    p.add_argument("--is_folder", action="store_true", help="treat --poses as folder and process all .npy inside")
    p.add_argument("--mean", type=str, default=None, help="path to mean.npy for 272-d data")
    p.add_argument("--std", type=str, default=None, help="path to std.npy for 272-d data")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mean = np.load(args.mean) if args.mean is not None else None
    std = np.load(args.std) if args.std is not None else None

    if args.is_folder:
        allfiles = findAllFile(args.poses)
        npy_list = [f for f in allfiles if f.endswith(".npy") or f.endswith(".npz")]
        for f in tqdm(npy_list):
            rel = os.path.relpath(f, args.poses)
            out_path = os.path.join(args.output, os.path.splitext(rel)[0] + "_smplx_params.npz")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                process_file(f, out_path, mean=mean, std=std)
            except Exception as e:
                print(f"❌ Failed {f}: {e}")
    else:
        out = args.output
        if os.path.isdir(out):
            base = os.path.splitext(os.path.basename(args.poses))[0]
            out = os.path.join(out, base + "_smplx_params.npz")
        process_file(args.poses, out, mean=mean, std=std)
    print("Done.")

# python Datasets/SMPLX/convert_272_to_smplx.py \
#     --poses Datasets/SMPLX/demo_272_output/test001.npy \
#     --output Datasets/SMPLX/data_converted_to_smplx/ \
#     --mean Datasets/SMPLX/demo_272_output/mean_std/Mean.npy \
#     --std Datasets/SMPLX/demo_272_output/mean_std/Std.npy

