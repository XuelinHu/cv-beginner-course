# 计算机视觉入门：从像素矩阵到可运行模型

给第一次学习计算机视觉的学生使用的中文 Jupyter Notebook 项目。22 课各自包含输入图片、矩阵或张量、处理步骤、输出可视化、评价与练习。不需要显卡或API密钥。00–18基础实验安装依赖后可离线运行；19–21真实工业、手势和人体课程首次联网下载数据/模型，之后复用本地缓存。

## 第一次运行（推荐：Python 3.11 + CPU）

解压完整项目，**保留 notebooks、assets 和 course_utils.py 的相对位置**。建议至少 4 GB 可用内存。第一次安装需要互联网。

Windows PowerShell：

```powershell
cd D:\workspace\cv\cv-beginner-course
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

其他学生请将 `cd` 后的路径改为自己的解压位置。无需激活 PowerShell 脚本，也无需管理员权限。若 `py` 不存在，安装 Python 3.11，或使用下方 Conda 方式。

已有 Conda（本机已装好 `pyg`，直接执行最后一行即可）：

```powershell
cd D:\workspace\cv\cv-beginner-course
conda run -n pyg python -m jupyterlab notebooks
```

新建独立 Conda 环境：

```powershell
conda create -n cv-course python=3.11 pip -y
conda run -n cv-course python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
conda run -n cv-course python -m pip install -r requirements.txt
conda run -n cv-course python -m jupyterlab notebooks
```

Linux：

```bash
cd /path/to/cv-beginner-course
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab notebooks
```

打开 `00_start_here.ipynb`，选择 Python 内核，点击“运行 → 运行所有单元格”。每一课可重新启动内核后独立运行，无需先执行前一课。Notebook 已保留实际运行的图文输出，即使暂未装环境也能阅读。

## 学习顺序

|课|Notebook|实际实验|
|---|---|---|
|00|[环境与第一张图](notebooks/00_start_here.ipynb)|真实照片 → 8×8矩阵 → NCHW张量 → 输出|
|01|[像素、颜色、采样与量化](notebooks/01_pixels_matrices.ipynb)|手算矩阵；采样、量化、灰度、HSV、直方图|
|02|[滤波、卷积和边缘](notebooks/02_filtering_edges.ipynb)|核乘加、噪声、4种滤波、PSNR、Sobel/Canny|
|03|[形态学、轮廓和特征](notebooks/03_morphology_contours.ipynb)|零件掩膜清理、像素测量、真实照片ORB匹配|
|04|[数据集与评价](notebooks/04_datasets_evaluation.ipynb)|真实digits数据、训练/验证/测试、混淆矩阵、分布变化|
|05|[传统视觉模型](notebooks/05_classical_models.ipynb)|HOG+SVM；Haar/HOG检测与真实照片|
|06|[图像分类](notebooks/06_classification.ipynb)|真实手写数字上训练小CNN并独立测试|
|07|[目标检测](notebooks/07_detection.ipynb)|受控形状图像上学习类别和边界框；IoU与失败例|
|08|[图像分割](notebooks/08_segmentation.ipynb)|受控标注上训练微型U-Net；mask、叠加图、Dice/IoU|
|09|[图像检索](notebooks/09_retrieval.ipynb)|特征向量、余弦相似度、Top-K排序、Recall@K|
|10|[扩散图像生成](notebooks/10_generation.ipynb)|真实digits上训练微型DDPM；前向加噪与反向采样|
|11|[图像描述](notebooks/11_captioning.ipynb)|受控图形图像 → CNN编码 → GRU逐词生成|
|12|[超分辨率](notebooks/12_super_resolution.ipynb)|低清/高清配对、小残差CNN、双三次基线、PSNR|
|13|[双目深度](notebooks/13_depth.ipynb)|已知几何生成双目图 → 匹配视差 → 米制深度|
|14|[光流](notebooks/14_optical_flow.ipynb)|相邻帧、Farneback向量场、图像重映射、EPE|
|15|[三维重建](notebooks/15_reconstruction.ipynb)|投影矩阵 → 两相机观测 → 三角化 → 三维点云|
|16|[姿态估计](notebooks/16_pose.ipynb)|受控关节图 → 训练热图网络 → 解码关键点 → PCK|
|17|[CT预处理与分割入门](notebooks/17_ct_preprocessing.ipynb)|合成HU切片、窗宽窗位、掩膜与Dice；可选真实DICOM入口|
|18|[水下图像实验](notebooks/18_underwater.ipynb)|真实照片模拟水下退化、白平衡/CLAHE、对照与指标局限|
|19|[真实工业缺陷分类与ROI](notebooks/19_industrial_defect_classification.ipynb)|NEU-DET钢材图、VOC标注、分组划分、三层全部通道、卷积/池化/向量/双头损失、独立测试|
|20|[真实手指手势识别](notebooks/20_hand_gesture_real.ipynb)|21点标注、自动手掌ROI、CNN内部输出、672维向量到63坐标、误差与真实视频逐帧推理|
|21|[真实人体姿态与动作](notebooks/21_human_pose_real.ipynb)|人工17点与模型33点映射、CNN算子、768维向量到195输出、热图/mask支路、下蹲视频逐帧关节角度|

完整目录对应关系和各课前置知识见 [课程地图](docs/CURRICULUM.md)。

## 工业视觉专项实验

本仓库还包含独立组织的[工业视觉入门实验室](courses/industrial-vision-lab/README.md)：从暗场补光、二值化和纹理，到汇流排定位、镍片/七格检查、遮挡与摆歪时的拒判，以及实拍教程和数据留存。它与上面的 22 课共享同一仓库，但保留自己的依赖、运行入口和数据边界；现场原图、私有样本、SQLite 数据库和运行时文件不会提交或打入学生包。

00–18于2026-11-16在全新Python 3.11环境逐课执行，19–21于2026-11-17在同一独立环境执行；22课均保留实际图文输出。详情见 [运行验收记录](docs/VERIFICATION.md)。

## 真实场景三课怎么学

按19→20→21顺序。19课第一次下载固定版本NEU-DET镜像（约122MB），解析1800张图和XML。相同像素图分组后划为训练、验证、测试；归一化参数只从训练集拟合。8→16→32个通道逐层展示，手算卷积/最大池化并与PyTorch比对；32维分类向量与保留位置的框分支分别计算损失。数据中有多类共现，按文件名主类别做单标签分类，保留所有原框，定位主类别联合框；不是完整多目标检测器。

20–21首次下载约26MB预训练模型和少量真实标注照片/视频，使用已有OpenCV，无额外依赖。手部演示21点和几何手势规则；人体演示33主体点、6辅助点与下蹲时间序列。这两课查看固定预训练网络的真实内部张量，不声称重新训练大模型，也不把小样例误差称为独立基准成绩。图表、逐帧坐标、算子清单与指标保存到outputs，原始下载缓存位于data。

下载地址与模型SHA256固定，原始照片/视频保留各自权利；项目MIT不改变第三方数据许可。学生包保留已运行图文，原始数据与模型按需下载。详情见[来源记录](docs/SOURCES.md)。

## CPU 与 GPU

PyTorch 课程统一使用 `device()`，自动选 CUDA、MPS 或 CPU，训练数据与模型移到同一设备，绘图前转回 CPU。OpenCV 的传统处理使用 CPU；有 GPU 不会自动把 OpenCV 变成 CUDA 版本。

在有兼容 NVIDIA 驱动的机器上，可在新环境使用以下命令替换 CPU 的 torch 安装步骤，其余命令相同：

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

这是 PyTorch 2.5.1 官方提供的安装路径，驱动/硬件要求参见 [PyTorch 版本安装说明](https://pytorch.org/get-started/previous-versions/)。本次实际验证机器只有 CPU；GPU 路径已编写，尚未在真实显卡上执行。

强制 CPU：PowerShell 先执行 `$env:CV_DEVICE='cpu'`，Linux 先执行 `export CV_DEVICE=cpu`。设置后重启 Jupyter 内核。`CV_DEVICE=cuda` 在没有 CUDA 时会明确报错，避免误以为使用了显卡。

## 一次执行全部课程

在项目根目录运行（每篇独立新内核，结果写回 Notebook，明细写入 verification）：

```powershell
conda run --no-capture-output -n pyg python scripts/run_notebooks.py --device cpu
```

普通 Python 环境改成 `python scripts/run_notebooks.py --device cpu`。仅运行一篇可加 `--pattern '06*.ipynb'`。默认不会跳过训练，实际耗时取决于电脑。输出图保存在 `outputs/`，使用同名输出便于重复比较；不修改输入照片。

## 这些实验能说明什么

- **真实输入**：OpenCV 样例照片及内置手写数字；来源、版本和 SHA256 在 assets/SOURCES.json。
- **受控实验**：检测、分割、描述、姿态采用带已知标签的合成图，便于初学者理解端到端训练；不能把高分当成真实场景效果。
- **小型模型**：DDPM、CNN、GRU 等确实训练并推理；数据规模与模型容量面向课堂，不等同于大规模预训练系统。
- **几何实验**：深度与三维使用明确的相机、纹理和观测假设；三角化不是一个包含自动匹配、SfM、MVS、网格纹理的完整工程系统。
- **CT与水下**：合成HU、模拟退化均有显式说明，避免将演示输出解释为临床结果或真实水下恢复真值。

每课都有修改数据、阈值或参数的练习。先理解矩阵和输入输出，再尝试真实自有数据。

## 项目来源与分发

已经克隆 [applied-computer-vision-opencv](https://github.com/mohd-faizy/applied-computer-vision-opencv) 到开发工作区的 `upstream/`，核对了其 MIT 许可与教程。其依赖面向 Python 3.14，且缺少本课程所需的完整生成/描述/分割教学链路，因此本项目重新组织中文课程、压缩依赖，并加入对应实验。学生包不依赖该克隆才能运行。参考项目和改动范围见 [来源记录](docs/SOURCES.md)；原始许可保存在 `reference/`。

可直接把 `dist/cv-beginner-course-student.zip` 发给学生。生成命令：

```powershell
python scripts/package_course.py
```

GitHub 仓库：[XuelinHu/cv-beginner-course](https://github.com/XuelinHu/cv-beginner-course)。Gitee 镜像：[XuelinHu/cv-beginner-course](https://gitee.com/XuelinHu/cv-beginner-course)。两个仓库均已公开，学生可以直接克隆：

```bash
git clone https://gitee.com/XuelinHu/cv-beginner-course.git
cd cv-beginner-course
```

也可使用 `git clone https://github.com/XuelinHu/cv-beginner-course.git`。

也可以直接分发学生压缩包。第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 常见问题

- `No module named torch/cv2`：当前 Notebook 内核不是安装依赖的环境。在单元格运行 `import sys; print(sys.executable)` 核对。
- 图片找不到：检查是否只复制了 ipynb；应解压完整学生包。
- 中文变成方块：Windows 自动使用微软雅黑；Linux 可安装系统的 Noto Sans CJK 字体，或按英文标题理解图。
- 首次安装慢：国内可为普通依赖使用学校可信的 PyPI 镜像；PyTorch 的 CPU/CUDA 安装源应按上面的官方地址选择。
- 内存或运行时间不足：关闭其他训练 Notebook，按课独立运行；不要同时运行所有内核。
- 需要真实 DICOM：另行安装 `python -m pip install -r requirements-optional.txt`，按17课入口读取自己的文件并处理隐私信息。
