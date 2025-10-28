import os
import json
import numpy as np
import torch
from smplx import SMPLX
from scipy.spatial.transform import Rotation as R

from utils.path_manager_smplx import PathManager_SMPLX

pm = PathManager_SMPLX()


def smplx_to_motionstreamer_272_body(motion_parms, model_path, gender='neutral', dt=1/30, device='cpu'):
    """
    Convert SMPL-X motion parameters to 272D MotionStreamer representation.

    Args:
        motion_parms: dict with keys:
            - 'root_orient': (T,3) axis-angle
            - 'pose_body': (T,63) = 21*3 axis-angle
            - 'trans': (T,3)
        model_path: path to SMPL-X model directory
        gender: 'neutral', 'male', or 'female'
        dt: frame time interval (default 1/30)
        device: 'cpu' or 'cuda'
    Returns:
        motion_272: np.ndarray (T, 272)
    """

    T = motion_parms['trans'].shape[0]
    K = 22  # SMPL-X body joints (root + 21)

    # convert to torch tensors
    root_orient = torch.tensor(motion_parms['root_orient'], dtype=torch.float32).to(device)
    pose_body = torch.tensor(motion_parms['pose_body'], dtype=torch.float32).to(device)
    trans = torch.tensor(motion_parms['trans'], dtype=torch.float32).to(device)

    # 1️⃣ load SMPL-X model
    body_model = SMPLX(
        model_path=model_path,
        gender=gender,
        use_hands=False,
        use_face=False,
        num_pca_comps=0,
        batch_size=T
    ).to(device)

    # 2️⃣ forward SMPL-X
    smpl_out = body_model(
        global_orient=root_orient.unsqueeze(1),
        body_pose=pose_body,
        transl=trans
    )

    joints = smpl_out.joints[:, :K, :]  # (T,22,3)

    # 3️⃣ root planar velocity
    trans_np = trans.detach().cpu().numpy()
    root_vel = np.zeros((T, 2))
    root_vel[1:] = (trans_np[1:, [0, 2]] - trans_np[:-1, [0, 2]]) / dt
    root_vel[0] = root_vel[1]

    # 4️⃣ root 6D rotation
    root_orient_np = root_orient.detach().cpu().numpy()
    root_rot6d = np.zeros((T, 6))
    for t in range(T):
        root_rot6d[t] = R.from_rotvec(root_orient_np[t]).as_matrix()[:, :2].reshape(-1)

    # 5️⃣ joint local positions
    joints_np = joints.detach().cpu().numpy()
    root_pos = joints_np[:, 0:1, :]
    jp = joints_np - root_pos  # (T,22,3)
    jp = jp.reshape(T, -1)

    # 6️⃣ joint velocities
    jv = np.zeros_like(jp)
    jv[1:] = (jp[1:] - jp[:-1]) / dt
    jv[0] = jv[1]

    # 7️⃣ joint 6D rotations
    full_pose = torch.cat([root_orient.unsqueeze(1), pose_body.view(T, 21, 3)], dim=1)
    jr = np.zeros((T, K * 6))
    for t in range(T):
        Rm = R.from_rotvec(full_pose[t].detach().cpu().numpy()).as_matrix()  # (22,3,3)
        jr[t] = Rm[:, :, :2].reshape(-1)

    # 8️⃣ concatenate all to (T, 272)
    motion_272 = np.concatenate([root_vel, root_rot6d, jp, jv, jr], axis=1)
    assert motion_272.shape[1] == 272, f"Got {motion_272.shape[1]} dims, expected 272."

    return motion_272


def convert_motion_file(npz_path, model_path, output_dir='data_converted_to_272', gender='neutral', device='cpu'):
    os.makedirs(pm.root_dataset/output_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    motion_parms = {
        'root_orient': data['root_orient'],
        'pose_body': data['pose_body'],
        'trans': data['trans']
    }

    motion_272 = smplx_to_motionstreamer_272_body(
        motion_parms, model_path, gender=gender, device=device
    )

    base = os.path.splitext(os.path.basename(npz_path))[0]
    npz_out_path = os.path.join(pm.root_dataset/output_dir, f"{base}_motion272.npz")
    json_out_path = os.path.join(pm.root_dataset/output_dir, f"{base}_motion272.json")

    # 保存 npz
    np.savez_compressed(npz_out_path, motion_272=motion_272)

    # 保存 json（只保留少量数据检查）
    json_summary = {
        'file': npz_path,
        'num_frames': motion_272.shape[0],
        'dim': motion_272.shape[1],
        'example_first_row': motion_272[0].tolist(),
    }
    with open(json_out_path, 'w') as f:
        json.dump(json_summary, f, indent=2)

    print(f"✅ Saved to:\n  - {npz_out_path}\n  - {json_out_path}")


if __name__ == "__main__":
    # ======== 示例使用 ========
    model_path = "/home/liu/lyz/Human/Datasets/SMPLX/smplx/models/smplx"        # 你的 SMPL-X 模型目录
    input_npz = "/home/liu/lyz/Human/Datasets/SMPLX/P1.npz"    # 你的输入 npz 文件
    convert_motion_file(input_npz, model_path, device='cuda')
