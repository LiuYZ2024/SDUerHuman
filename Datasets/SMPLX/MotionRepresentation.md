# 动作表示方法总结

本文档总结了四种常用的人体动作表示方法：  

* HumanML3D 的 263 维表示  
* HumanML3D-272 的 272 维表示（MotionStreamer）  
* SMPL 参数表示  
* SMPL-X 参数表示 (Inter-X)

---

## 🔹263D 表示（传统 HumanML3D / MoGlow 等用法）

$$
x = \{\dot r_x, \dot r_z, r_y, \dot r_a,\ j_p,\ j_v,\ j_r,\ c \}
$$

* **根部信息（4 维）**

  * $(\dot r_x, \dot r_z \in \mathbb{R}^2)$：根节点在 XZ 平面上的速度
  * $(r_y \in \mathbb{R})$：根节点高度（Y 方向）
  * $(\dot r_a \in \mathbb{R})$：根节点绕 Y 轴的角速度

* **关节信息**

  * $(j_p \in \mathbb{R}^{3(K-1)})$：**相对 root 的局部关节位置**（不含 root）
  * $(j_v \in \mathbb{R}^{3K})$：局部关节速度（含 root）
  * $(j_r \in \mathbb{R}^{6(K-1)})$：局部关节旋转（6D 形式，但根节点没有）

* **接触标签（4 维）**

  * $(c \in \mathbb{R}^4)$：左右脚、左右手是否接触地面

👉 总和起来：
$$
2 + 1 + 1 + 3 \times 21 + 3 \times 22 + 6 \times 21 + 4 = 263
$$

**特点：**

- 主要用于动作生成任务  
- 需要额外步骤转换回 SMPL 参数  
- 速度信息和旋转均编码在向量中

⚠️ **问题**：

* 关节旋转 (j_r) **不是 SMPL 原生旋转**，而是通过 **IK (Inverse Kinematics)** 从 (j_p)（局部位置）估算出来的。
* IK 会丢失 **twist rotation（扭转信息）**，导致 SMPL 动画时出现 **误差和抖动**。
* Post-processing （IK + SMPLify）耗时大（10 秒动作需要 ~60 秒后处理）。

---

## 🔹272D 表示（改进版）

$$
x = \{\dot r_x, \dot r_z, \dot r_a,\ j_p,\ j_v,\ j_r \}
$$

* **根部信息**

  * $(\dot r_x, \dot r_z \in \mathbb{R}^2)$：XZ 平面速度
  * $(\dot r_a \in \mathbb{R}^6)$：根节点旋转，直接用 **6D rotation**（替代原来的「角速度 + 高度」）

* **关节信息（包含 root）**

  * $(j_p \in \mathbb{R}^{3K})$：所有关节的局部位置（含 root）
  * $(j_v \in \mathbb{R}^{3K})$：所有关节的局部速度（含 root）
  * $(j_r \in \mathbb{R}^{6K})$：所有关节的局部旋转（含 root），直接取自 **SMPL 原始旋转**，6D 表示

👉 总和起来：
$$
2 + 6 + 3 \times 22 + 3 \times 22 + 6 \times 22 = 272
$$

⚡ **改进点**：

* **完全跳过 IK**，直接用 AMASS/SMPL 提供的旋转。
* 旋转用 **6D 表示**，避免四元数归一化和欧拉角奇异性。
* 不再需要显式存 root 高度 (r_y)，因为 root 的全局姿态已经体现在 6D rotation 和 root 的位置里。
* 移除了接触标签 (c)（可以另行预测或用物理约束代替）。

---

## 🔹SMPL 参数表示（传统骨骼驱动）

SMPL 模型直接使用人体骨骼参数进行动作表示，常用于重建、驱动和生成任务。其动作向量通常包含以下内容：

$$
x = \{ \mathbf{\theta}, \mathbf{\beta}, \mathbf{t} \}
$$

* **关节旋转**

  * $\theta \in \mathbb{R}^{3K}$ 或 $\mathbb{R}^{6K}$：每个关节的旋转

    * 3D 表示为 **axis-angle**
    * 6D 表示为 **6D rotation**（避免奇异性和归一化问题）

* **身体形状参数**

  * $\beta \in \mathbb{R}^{10}$（通常）表示个体体型（体型 PCA 系数）

* **全局根节点位置**

  * $t \in \mathbb{R}^3$：全局平移，用于定位 root

**特点：**

* 完全基于 SMPL 原生参数，无需 IK 或后处理
* 可直接驱动 SMPL 模型生成真实人体网格
* 可扩展到动作生成、姿态估计、动作迁移等任务

⚠️ **问题**：

* 高维旋转表示（尤其 axis-angle）在训练生成模型时可能不稳定
* 不包含速度信息，如果需要动作预测，需要另外计算关节速度或加速度

---

## 🔹SMPL-X 参数表示（Inter-X / 表情 + 手部扩展）


$$
x = \{ \mathbf{\theta}_{\text{body}}, \mathbf{\theta}_{\text{hands}}, \mathbf{\theta}_{\text{face}}, \mathbf{\beta}, \mathbf{t} \}
$$

* **身体旋转**

  * $\theta_{\text{body}} \in \mathbb{R}^{6K_\text{body}}$：身体主要关节旋转

* **手部旋转**

  * $\theta_{\text{hands}} \in \mathbb{R}^{6K_\text{hands}}$：手部每个关节旋转，6D 表示

* **面部表情**

  * $\theta_{\text{face}} \in \mathbb{R}^{K_\text{face}}$：通常是表情 PCA 或 blendshape 系数

* **身体形状与根节点**

  * $\beta \in \mathbb{R}^{10}$：体型
  * $t \in \mathbb{R}^3$：全局 root 平移

**特点：**

* 能同时表示全身动作 + 手势 + 面部表情
* 常用于高保真动作生成和交互动画（如 Inter-X 数据集）
* 可以用 6D rotation 表示，避免欧拉角或四元数问题

⚠️ **问题**：

* 参数维度高（可能超过 150），训练生成模型更具挑战
* 面部和手部的精细动作需要额外数据和监督

---

## 🔹核心区别总结

| 项目                 | 263D 表示                                        | 272D 表示                                   | SMPL 表示         | SMPL-X 表示      |
| ------------------ | ---------------------------------------------- | ----------------------------------------- | --------------- | -------------- |
| **Root 表示**        | (\dot r_x, \dot r_z, r_y, \dot r_a)（速度+高度+角速度） | (\dot r_x, \dot r_z, \dot r_a(6D))（速度+旋转） | t + 根关节旋转       | t + 根关节旋转      |
| **Joint Position** | (3(K-1))，不含 root                               | (3K)，含 root                               | 隐式由旋转生成网格       | 隐式由旋转生成网格，含手/面 |
| **Joint Rotation** | (6(K-1))，需 IK 解出                               | (6K)，直接取 SMPL 原始旋转                        | Axis-angle / 6D | 6D，包含手和身体关节    |
| **Joint Velocity** | (3K)                                           | (3K)                                      | 无               | 无              |
| **Contact Label**  | 有（4D）                                          | 无                                         | 无               | 无              |
| **形状参数**           | 无                                              | 无                                         | β (体型)          | β (体型)         |
| **面部 / 手部动作**      | 无                                              | 无                                         | 无               | 有（手部关节 + 面部表情） |
| **后处理需求**          | 必须做 IK/SMPLify，慢且有误差                           | 无需 IK，直接可驱动 SMPL                          | 无               | 无              |
| **总维度**            | 263                                            | 272                                       | ~85~90（身体）      | >150（含手和面）     |


---

## 将 272表示 转换成 SMPL-X参数表示：

```python
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
