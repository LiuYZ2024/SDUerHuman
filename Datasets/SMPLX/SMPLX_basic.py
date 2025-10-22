# Packages you may use very often.
import torch
import numpy as np
from smplx import SMPLX

# Things you don't need to care about. They are just for driving the tutorials.
from utils.path_manager_smplx import PathManager_SMPLX
from utils.wis3d_utils import HWis3D as Wis3D
# wis3d --vis_dir /home/liu/lyz/Human/Datasets/data_output/wis3d --host 0.0.0.0 --port 19090 --verbose True 

from skeleton_structure import Skeleton_SMPL24 as Skeleton_SMPLX22

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

def learn_body_pose(eg_path):
    def load_eg_params(eg_path):
        eg_params = np.load(eg_path, allow_pickle=True).item()
        eg_body_pose_aa = torch.from_numpy(eg_params['body_pose'])  # (1, 21, 3)
        return eg_body_pose_aa

    def make_fake_data():
        tgt_body_pose = load_eg_params(eg_path)  # (1, 21, 3)
        grad_weights = torch.linspace(0, 1, B).reshape(B, 1, 1)  # (B, 1, 1)
        fake_body_pose = grad_weights * tgt_body_pose  # (B, 21, 3)
        return fake_body_pose

    fake_body_pose = make_fake_data()  # (B, 21, 3)

    # Inference.
    smpl_out = body_model(
            betas         = torch.zeros(B, 10),    # shape coefficients
            global_orient = torch.zeros(B, 1, 3),  # axis-angle representation
            body_pose     = fake_body_pose,        # axis-angle representation
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
                'SMPL-parameters-body_pose',
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
            joints = joints[:, :22],
            bones  = Skeleton_SMPLX22.bones,
            colors = Skeleton_SMPLX22.bone_colors,
            name   = f'skeleton',
            offset = 0,
        )
    visualize_results()

# learn_body_pose(pm.inputs / 'examples/ballerina.npy')