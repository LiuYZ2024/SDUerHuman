import numpy as np
from scipy.spatial.transform import Rotation as R

def smplx_to_motionstreamer_272_body(motion_parms, dt=1/30):
    """
    Convert SMPL-X style motion parameters to MotionStreamer 272D representation
    considering only 22 body joints.

    motion_parms: dict with keys:
        - 'root_orient' : (T,3) axis-angle of root
        - 'pose_body'   : (T, 3*21) axis-angle of 21 body joints (excluding root)
        - 'trans'       : (T,3) global root positions
    dt: frame interval in seconds (default 1/30 for 30fps)
    
    Returns:
        motion_272: (T, 272) MotionStreamer 272D representation
    """
    T = motion_parms['trans'].shape[0]
    K = 22  # total body joints (root + 21)
    
    # -------------------------
    # 1. Root linear velocity (XZ plane)
    trans = motion_parms['trans']  # (T,3)
    root_vel = np.zeros((T,2))
    root_vel[1:] = (trans[1:, [0,2]] - trans[:-1, [0,2]]) / dt
    root_vel[0] = root_vel[1]  # first frame
    
    # -------------------------
    # 2. Root 6D rotation
    root_orient = motion_parms['root_orient']  # (T,3)
    root_rot6d = np.zeros((T,6))
    for t in range(T):
        root_rot6d[t] = R.from_rotvec(root_orient[t]).as_matrix()[:,:2].reshape(-1)
    
    # -------------------------
    # 3. Joint local positions (relative to root)
    pose_body = motion_parms['pose_body']  # (T, 3*21)
    jp = np.zeros((T, 3*K))
    # first joint is root at origin in local space
    jp[:, :3] = 0.0
    jp[:, 3:] = pose_body  # 其余 21 个关节局部位置，用 pose_body 替代（若已有局部位置，可直接替换）
    
    # -------------------------
    # 4. Joint local velocities
    jv = np.zeros((T, 3*K))
    jv[1:] = (jp[1:] - jp[:-1]) / dt
    jv[0] = jv[1]  # first frame
    
    # -------------------------
    # 5. Joint 6D rotations
    jr = np.zeros((T, 6*K))
    # 根节点旋转
    jr[:, :6] = root_rot6d
    # 其余关节
    for t in range(T):
        # pose_body: axis-angle (T, 3*21)
        body_rotvecs = pose_body[t].reshape(21,3)
        body_rot6d = np.zeros((21,6))
        for k in range(21):
            body_rot6d[k] = R.from_rotvec(body_rotvecs[k]).as_matrix()[:,:2].reshape(-1)
        jr[t,6:] = body_rot6d.flatten()
    
    # -------------------------
    # 6. Concatenate to 272D
    motion_272 = np.concatenate([root_vel, root_rot6d, jp, jv, jr], axis=1)
    return motion_272
