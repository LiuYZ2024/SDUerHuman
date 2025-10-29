# Packages you may use very often.
import torch
import numpy as np
from smplx import SMPLX

# Things you don't need to care about. They are just for driving the tutorials.
from utils.path_manager_smplx import PathManager_SMPLX
from utils.wis3d_utils import HWis3D as Wis3D
# wis3d --vis_dir /home/liu/lyz/Human/Datasets/SMPLX/data_output/wis3d --host 0.0.0.0 --port 19090
# wis3d --vis_dir /home/liu/lyz/Human/Datasets/SMPLX/data_output/wis3d_263 --host 0.0.0.0 --port 19093
from utils.skeleton_structure import Skeleton_SMPL22 as Skeleton_SMPLX22

pm = PathManager_SMPLX()

B = 150
body_models = {}
genders = ['neutral', 'female', 'male']  # case insensitive

# all joints
# for gender in genders:
#     body_models[gender] = SMPLX(
#         model_path = pm.root_dataset / 'smplx' / 'models' / 'smplx',
#         gender='neutral',
#         use_hands=True,
#         use_face=True,
#         use_face_contour=True,
#         num_pca_comps=6,
#         use_pca=True,
#         ext='npz'
#     )

# only contain body joints (bug)
for gender in genders:
    body_models[gender] = SMPLX(
        model_path = pm.root_dataset / 'smplx' / 'models' / 'smplx',
        gender='neutral',
        use_hands=False,
        use_face=False,
        use_face_contour=False,
        num_pca_comps=0,
        use_pca=False,
        ext='npz',
        batch_size=B
    )

# Prepare some parameters for later inference.
body_model : SMPLX = body_models['neutral']  # use neutral for example

# Prepare mesh template for later visualization.
# Tips: mesh = vertices + faces, and the faces are the indices of vertices, which won't change across SMPL's outputs.
mesh_temp : np.ndarray = body_model.faces  # (13776, 3)

# Inference.
betas         = torch.zeros(B, 10)          # 体型系数
global_orient = torch.zeros(B, 3)           # 根关节旋转 axis-angle
body_pose     = torch.zeros(B, 21, 3)       # 其余身体关节
left_hand_pose  = torch.zeros(B, 6)         # 手部 PCA
right_hand_pose = torch.zeros(B, 6)
jaw_pose      = torch.zeros(B, 3)
leye_pose     = torch.zeros(B, 3)
reye_pose     = torch.zeros(B, 3)
expression    = torch.zeros(B, 10)
transl       = torch.zeros(B, 3)

# ---------------------------
# 前向推理
# ---------------------------
# all joints
# smpl_out = body_model(
#     betas=betas,
#     global_orient=global_orient,
#     body_pose=body_pose,
#     left_hand_pose=left_hand_pose,
#     right_hand_pose=right_hand_pose,
#     jaw_pose=jaw_pose,
#     leye_pose=leye_pose,
#     reye_pose=reye_pose,
#     expression=expression,
#     transl=transl
# )

# only contain body joints (bug)
smpl_out = body_model(
    betas=betas,
    global_orient=global_orient,
    body_pose=body_pose,
    transl=transl
)

# Check output.
joints : torch.Tensor = smpl_out.joints    # [150, 117, 3]
verts  : torch.Tensor = smpl_out.vertices  # [150, 10475, 3]
print(joints.shape, verts.shape)

def learn_betas(
    selected_component : int = 0,
    lower_bound : int = -5,
    upper_bound : int = +5,
):
    def make_fake_data():
        fake_betas = torch.zeros(B, 10)
        fake_betas[:, selected_component] = torch.linspace(lower_bound, upper_bound, B)
        return fake_betas
    fake_betas = make_fake_data()
    print(fake_betas)

    # Inference.
    smpl_out = body_model(
            betas         = fake_betas,             # shape coefficients
            global_orient = torch.zeros(B, 1, 3),   # axis-angle representation
            body_pose     = torch.zeros(B, 21, 3),  # axis-angle representation
            transl        = torch.zeros(B, 3),
        )

    # Check output.
    joints : torch.Tensor = smpl_out.joints    # (B, 117, 3)
    verts  : torch.Tensor = smpl_out.vertices  # (B, 10475, 3)
    faces  : np.ndarray   = body_model.faces   # (20908, 3)
    print(joints.shape)
    print(verts.shape)
    print(faces.shape)


    def visualize_results():
        """ This part is to visualize the results. You are supposed to ignore this part. """
        shape_wis3d = Wis3D(
                pm.outputs / 'wis3d',
                'SMPLX-parameters-beta',
            )

        shape_wis3d.add_motion_verts(
            verts  = verts,
            name   = f'betas[:, {selected_component}] from {lower_bound} to {upper_bound}',
            offset = 0,
        )
        shape_wis3d.add_motion_mesh(
            verts  = verts,
            faces  = faces,
            name   = f'surface: betas[:, {selected_component}] from {lower_bound} to {upper_bound}',
            offset = 0,
        )
        shape_wis3d.add_motion_skel(
            joints = joints[:, :24],
            bones  = Skeleton_SMPLX22.bones,
            colors = Skeleton_SMPLX22.bone_colors,
            name   = f'skeleton: betas[:, {selected_component}] from {lower_bound} to {upper_bound}',
            offset = 0,
        )
    visualize_results()

# learn_betas(0)
# learn_betas(1)
# learn_betas(2)

def learn_orient():
    def make_fake_data():
        fake_orient = torch.zeros(B, 1, 3)
        fake_orient[   : 50, :, 0] = torch.linspace(0, 2 * np.pi, 50).reshape(50, 1)  # about x-axis
        fake_orient[ 50:100, :, 1] = torch.linspace(0, 2 * np.pi, 50).reshape(50, 1)  # about y-axis
        fake_orient[100:150, :, :] = torch.linspace(0, 2 * np.pi, 50).reshape(50, 1, 1).repeat(1, 1, 3)  # about x=y=z
        fake_orient[100:150, :, :] /= np.sqrt(3)  # Ensure the norm is 2pi.
        return fake_orient
    fake_orient = make_fake_data()

    # Inference.
    smpl_out = body_model(
            betas         = torch.zeros(B, 10),     # shape coefficients
            global_orient = fake_orient,            # axis-angle representation
            body_pose     = torch.zeros(B, 21, 3),  # axis-angle representation
            transl        = torch.zeros(B, 3),
        )

    # Check output.
    joints : torch.Tensor = smpl_out.joints    
    verts  : torch.Tensor = smpl_out.vertices  
    faces  : np.ndarray   = body_model.faces   

    def visualize_results():
        """ This part is to visualize the results. You are supposed to ignore this part. """
        orient_wis3d = Wis3D(
                pm.outputs / 'wis3d',
                'SMPL-parameters-global_orient',
            )

        # Prepare the rotation axis.
        axis_x   = torch.tensor([[0, 0, 0], [3, 0, 0]], dtype=torch.float32)
        axis_y   = torch.tensor([[0, 0, 0], [0, 3, 0]], dtype=torch.float32)
        axis_xyz = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        axis_all = torch.concat(
            [
                axis_x.reshape(1, 2, 3).repeat(50, 1, 1),
                axis_y.reshape(1, 2, 3).repeat(50, 1, 1),
                axis_xyz.reshape(1, 2, 3).repeat(50, 1, 1),
            ], dim = 0)
        axis_all[:, :, :] += joints[:, [0], :] # move the axis to the root joints


        orient_wis3d.add_vec_seq(
            vecs = axis_all,
            name = 'rotation axis',
        )
        orient_wis3d.add_motion_verts(
            verts  = verts,
            name   = f'vertices',
            offset = 0,
        )
        orient_wis3d.add_motion_mesh(
            verts  = verts,
            faces  = faces,
            name   = f'surface',
            offset = 0,
        )
        orient_wis3d.add_motion_skel(
            joints = joints[:, :24],
            bones  = Skeleton_SMPLX22.bones,
            colors = Skeleton_SMPLX22.bone_colors,
            name   = f'skeleton',
            offset = 0,
        )
    visualize_results()

# learn_orient()

def learn_body_pose(eg_path, frame_idx=0):
    """
    可视化指定动作文件中某一帧的 SMPL-X 姿态。
    
    Args:
        eg_path (str): .npz 文件路径
        frame_idx (int): 要可视化的帧索引（默认第0帧）
    """

    def load_eg_params(eg_path, frame_idx):
        data = np.load(eg_path, allow_pickle=True)
        if 'pose_body' not in data.files:
            raise KeyError(f"'body_pose' not found in {eg_path}. Found keys: {data.files}")
        body_pose = data['pose_body']
        body_pose_aa = torch.from_numpy(body_pose[min(frame_idx, body_pose.shape[0]-1)])  # 防止越界
        return body_pose_aa.unsqueeze(0)  # (1, 21, 3)


    def make_fake_data(target_pose):
        grad_weights = torch.linspace(0, 1, B).reshape(B, 1, 1)  # (B, 1, 1)
        fake_body_pose = grad_weights * target_pose  # (B, 21, 3)
        return fake_body_pose

    tgt_body_pose = load_eg_params(eg_path, frame_idx)
    fake_body_pose = make_fake_data(tgt_body_pose)

    smpl_out = body_model(
        betas=torch.zeros(B, 10),
        global_orient=torch.zeros(B, 1, 3),
        body_pose=fake_body_pose,
        transl=torch.zeros(B, 3),
    )

    joints: torch.Tensor = smpl_out.joints
    verts: torch.Tensor = smpl_out.vertices
    faces: np.ndarray = body_model.faces

    def visualize_results():
        orient_wis3d = Wis3D(
            pm.outputs / 'wis3d',
            f'SMPL-body-pose-frame{frame_idx}',
        )

        orient_wis3d.add_motion_verts(
            verts=verts,
            name='vertices',
            offset=0,
        )
        orient_wis3d.add_motion_mesh(
            verts=verts,
            faces=faces,
            name='surface',
            offset=0,
        )
        orient_wis3d.add_motion_skel(
            joints=joints[:, :22],
            bones=Skeleton_SMPLX22.bones,
            colors=Skeleton_SMPLX22.bone_colors,
            name='skeleton',
            offset=0,
        )

    visualize_results()

# 可视化第 0 帧（初始姿态）
# learn_body_pose('P1.npz', frame_idx=0)
# 可视化第 100 帧
# learn_body_pose( pm.root_dataset / 'SMPLX/P1.npz', frame_idx=100)

def learn_transl(rotation:bool = False):
    def make_fake_data():
        phase = torch.arange(50) / 50.0 * (2 * np.pi)  # 0 ~ 2𝜋

        # Generate fake translation.
        fake_transl = torch.zeros(B, 3)
        # Part 1, [0, 50)
        fake_transl[   : 25, 2] = torch.sin(phase[:25])  # along z-axis
        fake_transl[ 25: 50, 1] = torch.sin(phase[:25])  # along y-axis
        # Part 2, [50, 75) + [75, 100)
        fake_transl[ 50:100, 1] = torch.sin(phase)       # along y-axis
        fake_transl[ 50: 75, 0] = torch.sin(phase[::2])  # along y-axis
        fake_transl[ 75:100, 2] = torch.sin(phase[::2])  # along y-axis
        # Part 3, [100, 150)
        fake_transl[100:150, 0] = torch.cos(phase) * phase / (2 * np.pi)
        fake_transl[100:150, 2] = torch.sin(phase) * phase / (2 * np.pi)

        # Generate fake rotation (if needed).
        fake_orient = torch.zeros(B, 1, 3)
        if rotation:
            fake_orient[:, :, 1] = torch.linspace(0, 3 * (2 * np.pi), B).reshape(B, 1)  # about y-axis

        return fake_transl, fake_orient

    fake_transl, fake_orient = make_fake_data()

    # Inference.
    smpl_out = body_model(
            betas         = torch.zeros(B, 10),     # shape coefficients
            global_orient = fake_orient,            # axis-angle representation
            body_pose     = torch.zeros(B, 21, 3),  # axis-angle representation
            transl        = fake_transl,
        )

    # Check output.
    joints : torch.Tensor = smpl_out.joints  
    verts  : torch.Tensor = smpl_out.vertices
    faces  : np.ndarray   = body_model.faces 

    def visualize_results():
        """ This part is to visualize the results. You are supposed to ignore this part. """
        transl_wis3d = Wis3D(
                pm.outputs / 'wis3d',
                'SMPL-parameters-transl',
            )

        transl_wis3d.add_traj(
            positions = fake_transl,
            name      = f'trajectory (rotating)' if rotation else 'trajectory',
            offset    = 0,
        )
        transl_wis3d.add_motion_verts(
            verts  = verts,
            name   = f'vertices (rotating)' if rotation else 'vertices',
            offset = 0,
        )
        transl_wis3d.add_motion_mesh(
            verts  = verts,
            faces  = faces,
            name   = f'surface (rotating)' if rotation else 'surface',
            offset = 0,
        )
        transl_wis3d.add_motion_skel(
            joints = joints[:, :22],
            bones  = Skeleton_SMPLX22.bones,
            colors = Skeleton_SMPLX22.bone_colors,
            name   = f'skeleton (rotating)' if rotation else 'skeleton',
            offset = 0,
        )
    visualize_results()

# learn_transl(rotation=False)
# learn_transl(rotation=True)

def visualize_smplx_sequence(npz_path, gender='neutral', every_n_frame=1):
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
    vis = Wis3D(pm.outputs / 'wis3d', f'SMPLX-sequence-{npz_path.stem}')
    vis.add_motion_mesh(
        verts=verts,
        faces=faces,
        name=f'{npz_path.stem}-surface',
        offset=0,
    )
    vis.add_motion_skel(
        joints=joints[:, :22],
        bones=Skeleton_SMPLX22.bones,
        colors=Skeleton_SMPLX22.bone_colors,
        name=f'{npz_path.stem}-skeleton',
        offset=0,
    )

    print(f"✅ Visualization ready at Wis3D: {pm.outputs / 'wis3d'}")

visualize_smplx_sequence(npz_path=pm.root_dataset/'data_converted_to_smplx/test272_1_seq000_smplx_params.npz', every_n_frame=1)

def visualize_two_smplx_sequences(npz_path1, npz_path2, gender='neutral', every_n_frame=1):
    """
    在同一坐标系中可视化两个 SMPL-X 动作序列（例如交互动作 P1 和 P2）

    Args:
        npz_path1 (str or Path): 第一个 SMPL-X 参数 npz 文件路径（例如 P1.npz）
        npz_path2 (str or Path): 第二个 SMPL-X 参数 npz 文件路径（例如 P2.npz）
        gender (str): 'neutral' / 'male' / 'female'
        every_n_frame (int): 下采样帧间隔（默认 1 表示全部帧）
    """
    def load_sequence(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        betas = torch.tensor(data.get('betas', np.zeros((1, 10))), dtype=torch.float32)
        pose_body = torch.tensor(data.get('pose_body', np.zeros((1, 21, 3))), dtype=torch.float32)
        global_orient = torch.tensor(data.get('root_orient', np.zeros((1, 3))), dtype=torch.float32).unsqueeze(1)
        transl = torch.tensor(data.get('trans', np.zeros((1, 3))), dtype=torch.float32)
        return betas, pose_body, global_orient, transl

    # 分别加载两组参数
    betas1, pose_body1, global_orient1, transl1 = load_sequence(npz_path1)
    betas2, pose_body2, global_orient2, transl2 = load_sequence(npz_path2)

    n_frames = min(pose_body1.shape[0], pose_body2.shape[0])
    idx = np.arange(0, n_frames, every_n_frame)

    pose_body1 = pose_body1[idx]
    global_orient1 = global_orient1[idx]
    transl1 = transl1[idx]

    pose_body2 = pose_body2[idx]
    global_orient2 = global_orient2[idx]
    transl2 = transl2[idx]
    n_frames = len(idx)

    # 共享同一个模型定义
    body_model = SMPLX(
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

    # 推理两组动作
    smpl_out1 = body_model(
        betas=betas1.expand(n_frames, -1),
        global_orient=global_orient1,
        body_pose=pose_body1,
        transl=transl1,
    )
    smpl_out2 = body_model(
        betas=betas2.expand(n_frames, -1),
        global_orient=global_orient2,
        body_pose=pose_body2,
        transl=transl2,
    )

    joints1, verts1 = smpl_out1.joints, smpl_out1.vertices
    joints2, verts2 = smpl_out2.joints, smpl_out2.vertices
    faces = body_model.faces

    # Wis3D 可视化
    vis = Wis3D(pm.outputs / 'wis3d', f'SMPLX-P1P2-sequence')

    vis.add_motion_mesh(
        verts=verts1,
        faces=faces,
        name=f'{npz_path1.stem}-surface',
        offset=0,
    )
    vis.add_motion_mesh(
        verts=verts2,
        faces=faces,
        name=f'{npz_path2.stem}-surface',
        offset=0,
    )

    vis.add_motion_skel(
        joints=joints1[:, :22],
        bones=Skeleton_SMPLX22.bones,
        colors=Skeleton_SMPLX22.bone_colors,
        name=f'{npz_path1.stem}-skeleton',
        offset=0,
    )
    vis.add_motion_skel(
        joints=joints2[:, :22],
        bones=Skeleton_SMPLX22.bones,
        colors=Skeleton_SMPLX22.bone_colors,
        name=f'{npz_path2.stem}-skeleton',
        offset=0,
    )

    print(f"✅ Visualization ready at Wis3D: {pm.outputs / 'wis3d'}")

visualize_two_smplx_sequences(
    npz_path1 = pm.root_dataset / 'P1.npz',
    npz_path2 = pm.root_dataset / 'P2.npz',
    every_n_frame = 2
)