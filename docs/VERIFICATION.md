# 运行验收记录

验收日期：2026-11-16。

## 实测环境与结果

- Windows 10，Python 3.11.14，全新独立 venv，无继承 pyg 环境依赖。
- PyTorch 2.5.1+cpu，NumPy 2.2.6，OpenCV 4.12.0.88。
- 通过 README 的官方 CPU PyTorch 安装与 requirements.txt 安装；pip check 无依赖冲突。
- 全部 19 篇在各自独立新内核内顺序执行，内核强制使用运行脚本所属的同一个 Python。
- 共 107 个代码单元，77 幅嵌入 PNG；无执行错误或未运行单元。
- 本机总执行时间约 130.6 秒（含逐课启动内核；其他电脑耗时会不同）。
- 原始输入照片 SHA256、Notebook 结构、代码语法、图片文件有效性和说明文档相对链接检查通过。
- 学生压缩包已实际解压到独立目录，00 课通过相对路径读取素材并执行成功。

|Notebook|代码单元|嵌入图|秒|结果|
|---|---:|---:|---:|---|
|00_start_here.ipynb|4|3|10.56|通过|
|01_pixels_matrices.ipynb|6|6|4.3|通过|
|02_filtering_edges.ipynb|5|4|3.95|通过|
|03_morphology_contours.ipynb|6|5|4.04|通过|
|04_datasets_evaluation.ipynb|5|3|11.15|通过|
|05_classical_models.ipynb|6|5|14.33|通过|
|06_classification.ipynb|6|5|16.47|通过|
|07_detection.ipynb|6|4|7.63|通过|
|08_segmentation.ipynb|6|4|6.67|通过|
|09_retrieval.ipynb|6|3|3.13|通过|
|10_generation.ipynb|6|5|7.16|通过|
|11_captioning.ipynb|5|3|7.49|通过|
|12_super_resolution.ipynb|5|4|5.71|通过|
|13_depth.ipynb|5|3|3.25|通过|
|14_optical_flow.ipynb|5|4|3.32|通过|
|15_reconstruction.ipynb|6|2|3.1|通过|
|16_pose.ipynb|5|4|11.03|通过|
|17_ct_preprocessing.ipynb|7|4|3.42|通过|
|18_underwater.ipynb|7|6|3.91|通过|

## 复核方法

`powershell
python scripts/run_notebooks.py --device cpu
python scripts/audit_course.py
`

执行报告保存在 verification，图像和模型保存在 outputs；完整图文结果也保存在 Notebook 中。课程主流程不调用外部 API，不下载额外数据或预训练权重。可选真实 DICOM 入口需学生提供自己的文件。

## 实验范围

GPU 路径使用统一设备选择与张量迁移；本机无显卡，未做真实 CUDA/MPS 实测。Linux 的安装方式已写入 README，本次全量实测平台为 Windows。

检测、分割、图像描述与姿态采用明确标注的受控数据；深度与三维采用已知几何。CT 与水下章节注明合成/模拟数据。指标用于解释课堂实验，不代表真实业务或临床性能。
