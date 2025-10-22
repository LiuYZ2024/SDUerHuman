# SMPLX 

## 使用流程

### 1️⃣ 准备 SMPLX 模型文件

首先下载 SMPLX 官方提供的模型文件（`.pkl` 和 `.npz`），并放在SMPLX/smplx/models/smplx下面，形如：
```
models 
├── smplx 
    ├── SMPLX_FEMALE.npz 
    ├── SMPLX_FEMALE.pkl 
    ├── SMPLX_MALE.npz 
    ├── SMPLX_MALE.pkl 
    ├── SMPLX_NEUTRAL.npz 
    └── SMPLX_NEUTRAL.pkl
```
### 2️⃣ 创建 Conda 环境

根据仓库中的 `conda_env.yaml` 文件创建环境，执行命令：

```bash
# 创建 conda 环境
conda env create -f conda_env.yaml

# 激活环境
conda activate smplx
```
### 3️⃣ 运行 SMPLX_basic.ipynb 文件（要将kernal切换到conda环境）

---
## 参考资料

- [SMPLX GitHub 仓库](https://github.com/vchoutas/smplx)  
- [LearningHumans 仓库](https://github.com/IsshikiHugh/LearningHumans) 
