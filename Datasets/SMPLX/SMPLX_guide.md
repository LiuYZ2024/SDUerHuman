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

根据仓库中的 `environment.yaml` 文件创建环境，执行命令：

```bash
# 创建 conda 环境
conda env create -f environment.yaml

# 激活环境
conda activate human
```
### 3️⃣ 文件和文件夹说明：
[SMPL_basic.ipynb](./SMPL_basic.ipynb): 使用SMPL的示例文件，包括对一些fake数据的可视化脚本，在wis3d页面中查看可视化效果（来自[LearningHumans 仓库](https://github.com/IsshikiHugh/LearningHumans) ）

[SMPLX_basic.ipynb](./SMPLX_basic.ipynb): 使用SMPLX的示例文件，包括对一些fake数据的可视化脚本，在wis3d页面中查看可视化效果

[SMPLX_basic.py](./SMPLX_basic.py): SMPLX_basic.ipynb中py部分的汇总，方便运行和调试

[convert_272_to_smplx.py](./convert_272_to_smplx.py): 将272维表示的动作数据转成smplx参数的npz文件，可以利用SMPLX_basic.py中的visualize_smplx_sequence函数可视化

[convert_272_to_smplx_vis_B.py](./convert_272_to_smplx_vis_B.py): 将272维表示的动作数据转成smplx参数的npz文件，支持Batch输入，直接在当前脚本中调用可视化

[convert_263_to_global_pos_vis.py](./convert_263_to_global_pos_vis.py): 将263维表示的动作数据转成动作的全局关节坐标，支持Batch输入，直接在当前脚本中调用可视化

[P1.npz](./P1.npz): 示例smplx动作序列（取自Inter-X）

[smplx](./smplx/): 存放smplx模型文件

[utils](./utils/): 工具脚本

[visualization](./visualization/): 可视化相关脚本

[data_output](./data_output/): 输出的可视化数据

[demo_263_output](./demo_263_output/): 存放263维动作表示文件，以及mean_std

[demo_272_output](./demo_272_output/): 存放272维动作表示文件，以及mean_std

**convert_272_to_smplx.py， convert_272_to_smplx_vis_B.py， convert_263_to_global_pos_vis.py 的运行命令在文件开头的注释中给出**

---
## 参考资料

- [SMPLX GitHub 仓库](https://github.com/vchoutas/smplx)  
- [LearningHumans 仓库](https://github.com/IsshikiHugh/LearningHumans) 
