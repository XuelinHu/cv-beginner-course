# 计算机视觉入门：从像素矩阵到可运行模型

给第一次学习计算机视觉的学生使用的中文 Jupyter Notebook 项目。00–27 共 28 节，统一放在一个 `notebooks/` 目录、使用一套 Python 环境和一条 Jupyter 启动入口。课程包含基础视觉、模型实验，以及刀片电池汇流排焊接外观检测专题；专题从暗光成像开始，逐步定位汇流排、镍片与七格，并覆盖拒判、证据记录和当前实拍重跑。

## 第一次运行（推荐：Python 3.11 + CPU）

解压完整项目，**保留 notebooks、assets 和 course_utils.py 的相对位置**。建议至少 4 GB 可用内存。第一次安装需要互联网。

Windows PowerShell：

```powershell
cd D:\workspace\cv\cv-beginner-course
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:JUPYTER_CONFIG_DIR = (Join-Path (Get-Location) ".jupyter")
.\.venv\Scripts\python.exe -m jupyterlab --custom-css notebooks
```

其他学生请将 `cd` 后的路径改为自己的解压位置。无需激活 PowerShell 脚本，也无需管理员权限。若 `py` 不存在，安装 Python 3.11，或使用下方 Conda 方式。

已有 Conda（本机已装好 `pyg`，直接执行最后一行即可）：

```powershell
cd D:\workspace\cv\cv-beginner-course
$env:JUPYTER_CONFIG_DIR = (Join-Path (Get-Location) ".jupyter")
conda run -n pyg python -m jupyterlab --custom-css notebooks
```

新建独立 Conda 环境：

```powershell
conda create -n cv-course python=3.11 pip -y
conda run -n cv-course python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
conda run -n cv-course python -m pip install -r requirements.txt
$env:JUPYTER_CONFIG_DIR = (Join-Path (Get-Location) ".jupyter")
conda run -n cv-course python -m jupyterlab --custom-css notebooks
```

Linux：

```bash
cd /path/to/cv-beginner-course
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
export JUPYTER_CONFIG_DIR="$PWD/.jupyter"
.venv/bin/python -m jupyterlab --custom-css notebooks
```

Notebook 正文、代码和文本输出字号为 JupyterLab 默认值的 150%；该样式由项目内 `.jupyter/custom/custom.css` 提供。

打开 `00_start_here.ipynb`，选择 Python 内核，点击“运行 → 运行所有单元格”。每一课可重新启动内核后独立运行，无需先执行前一课。Notebook 已保留实际运行的图文输出，即使暂未装环境也能阅读。

## 学习顺序

课程共 28 篇，按 00–27 顺序学习。第 05 课专门讲解附件中的数据增强方法。

|课|Notebook|实际实验|
|---|---|---|
|00–04|基础与数据评价|像素、滤波、轮廓、数据集划分与指标|
|05|[数据增强与工业视觉鲁棒性](notebooks/05_data_augmentation.ipynb)|附件课件原图、几何/颜色/噪声、Mosaic、MixUp、Copy-Paste、Random Erasing、YOLO 参数|
|06–10|模型基础|HOG/SVM、CNN、检测、分割、橘子特征检索|
|11–15|几何与真实公开图像|光流、三维、姿态坐标、真实 DICOM CT、EUVP 水下图像|
|16–19|真实视觉与工业案例|手势、人体姿态、钢铁缺陷、焊缝/汇流排标签边界|
|20–27|刀片电池视觉专题|暗光、增强、二值、纹理、计数、质量门控、七格规则、实时汇流排流程|

完整目录对应关系和各课前置知识见 [课程地图](docs/CURRICULUM.md)。

## 刀片电池焊接检测专题

第 19–27 节是与前 18 节连续编号的专题内容，不另设第二套 Notebook、依赖或启动方式。专题源材料、许可清单、工具模块和现场样例保存在 `courses/industrial-vision-lab/`；完整私有数据集和运行数据库位于局域网服务器，不会推送到公开 Git 仓库。学生离线学习仍可使用课程中的公开样例及执行输出。

此前的运行验收记录见 [运行验收记录](docs/VERIFICATION.md)；统一后的 00–27 需在局域网服务器重新逐课执行并生成独立验收报告。

## 真实场景三课怎么学

按16→17→18顺序。18课第一次下载固定版本NEU-DET镜像（约122MB），解析1800张图和XML。相同像素图分组后划为训练、验证、测试；归一化参数只从训练集拟合。8→16→32个通道逐层展示，手算卷积/最大池化并与PyTorch比对；32维分类向量与保留位置的框分支分别计算损失。数据中有多类共现，按文件名主类别做单标签分类，保留所有原框，定位主类别联合框；不是完整多目标检测器。

16–17首次下载约26MB预训练模型和少量真实标注照片/视频，使用已有OpenCV，无额外依赖。手部演示21点和几何手势规则；人体演示33主体点、6辅助点与下蹲时间序列。这两课查看固定预训练网络的真实内部张量，不声称重新训练大模型，也不把小样例误差称为独立基准成绩。图表、逐帧坐标、算子清单与指标保存到outputs，原始下载缓存位于data。

下载地址与模型SHA256固定，原始照片/视频保留各自权利；项目MIT不改变第三方数据许可。学生包保留已运行图文，原始数据与模型按需下载。详情见[来源记录](docs/SOURCES.md)。

## CPU 与 GPU

PyTorch 课程统一使用 `device()`，自动选 CUDA、MPS 或 CPU，训练数据与模型移到同一设备，绘图前转回 CPU。OpenCV 的传统处理使用 CPU；有 GPU 不会自动把 OpenCV 变成 CUDA 版本。

在有兼容 NVIDIA 驱动的机器上，可在新环境使用以下命令替换 CPU 的 torch 安装步骤，其余命令相同：

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

这是 PyTorch 2.9.1 的 CUDA 12.8 安装路径，驱动/硬件要求参见 [PyTorch 版本安装说明](https://pytorch.org/get-started/previous-versions/)。服务器 RTX 5090 将使用此 GPU 环境并进行实机验证。

强制 CPU：PowerShell 先执行 `$env:CV_DEVICE='cpu'`，Linux 先执行 `export CV_DEVICE=cpu`。设置后重启 Jupyter 内核。`CV_DEVICE=cuda` 在没有 CUDA 时会明确报错，避免误以为使用了显卡。

## 一次执行全部课程

在项目根目录运行（每篇独立新内核，结果写回 Notebook，明细写入 verification）：

```powershell
conda run --no-capture-output -n pyg python scripts/run_notebooks.py --device cpu
```

普通 Python 环境改成 `python scripts/run_notebooks.py --device cpu`。仅运行一篇可加 `--pattern '06*.ipynb'`。默认不会跳过训练，实际耗时取决于电脑。输出图保存在 `outputs/`，使用同名输出便于重复比较；不修改输入照片。

## 这些实验能说明什么

- **真实输入**：OpenCV 样例照片及内置手写数字；来源、版本和 SHA256 在 assets/SOURCES.json。
- **受控实验**：检测、分割和姿态采用带已知标签的受控教学数据；真实汇流排图仅作场景示例，不作为未经标注的训练真值。
- **小型模型**：CNN 等模型面向课堂，数据规模与模型容量不等同于大规模预训练系统。
- **几何实验**：深度与三维使用明确的相机、纹理和观测假设；三角化不是一个包含自动匹配、SfM、MVS、网格纹理的完整工程系统。
- **CT与水下**：CT 使用随项目提供的公开 DICOM 原图；水下使用公开 EUVP 实拍输入与配对参考，不以水果或生成图代替真实数据。

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
