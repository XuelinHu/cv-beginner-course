# GitHub 搜索、选型与改编记录

检索日期：2026-11-16。

|项目|核对结果|本课程处理|
|---|---|---|
|[applied-computer-vision-opencv](https://github.com/mohd-faizy/applied-computer-vision-opencv)|已浅克隆，MIT；有基础、滤波、轮廓、光流、双目和推理教程；pyproject 要求 Python>=3.14|保留原仓库在开发目录upstream；在当前Python3.11基础上重新组织中文课与CPU小模型，不继承其全部依赖|
|[OpenCV](https://github.com/opencv/opencv)|官方算法文档和样例；固定素材版本4.12.0|采用官方样例照片，并按官方API编写可读小实验|
|[PyTorch Tutorials](https://github.com/pytorch/tutorials)|官方张量、训练和图像任务教程；单独不覆盖整个传统视觉范围|参考张量/模型/优化器组织方式，使用现有torch2.5.1，离线从头训练|

本项目不是将上游全部 Notebook 翻译后原样发布。新增了矩阵手算、中文讲解、离线素材、独立划分与指标、错误/领域变化案例、微型DDPM、CNN+GRU描述、超分、关键点热图、CT与水下专题。每篇算法核心保留在 Notebook 可阅读单元中。

## 方法资料

- OpenCV：[平滑](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)、[形态学](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)、[光流](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)、[双目深度](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)。
- PyTorch：[学习基础](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)、[安装版本选择](https://pytorch.org/get-started/previous-versions/)。
- [DDPM](https://arxiv.org/abs/2006.11239)、[U-Net](https://arxiv.org/abs/1505.04597)、[图像描述 Show and Tell](https://arxiv.org/abs/1411.4555)。课程使用缩小且有明确说明的教学实现，未声称复现论文指标。
- [scikit-learn digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)。

具体超参数、来源、适用范围见各课，不依赖实时网站才能执行主流程。
