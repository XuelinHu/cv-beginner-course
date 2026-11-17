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

## 19–21真实场景新增来源（2026-11-17）

- [NEU官方数据页](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)：热轧钢材6种表面缺陷。NEU-DET有1800张200×200图像与框标注。课程从[固定版本镜像](https://github.com/allanwong/NEU-DET-Steel-Surface-Defect-Detection/tree/351bee21b56bbf97a1e1d18262bceb202eb7ef41)按需下载，保留所有原框；文件名主类用于教学分类，其他缺陷类别不丢弃。按像素哈希分组、种子42分层划分，不冒充官方分割或完整多目标检测。
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models)：手掌检测、手部21点、人体检测和人体姿态模型。原版包装代码与Apache-2.0许可在reference/opencv_zoo；四个权重的SHA256写在keypoint_utils.py。无新增Python依赖，CPU OpenCV DNN。
- [MMPose测试资源](https://github.com/open-mmlab/mmpose/tree/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/tests/data/coco)：COCO-WholeBody四张真实照片、14个人物标注，保留原JSON的许可/来源元数据。手部21点人工真值、身体17点与模型33点显式映射；不能把公开参考样例误差当成独立基准成绩。
- [Google Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)：清晰测试照片pointing_up、thumb_up、woman_hands，用于自动ROI的新图推理。手势角度规则与关键点CNN区分讲解，不伪造手语分类概率。
- [TensorFlow.js真实手部视频](https://github.com/tensorflow/tfjs-models/blob/c731b9ebbd6f4c9e8bf99b0df76bbdbf9c25b07f/hand-pose-detection/test_data/asl_hand.25fps.mp4)、[真实下蹲视频](https://github.com/tensorflow/tfjs-models/blob/c731b9ebbd6f4c9e8bf99b0df76bbdbf9c25b07f/pose-detection/test_data/pose_squats.mp4)。按顺序逐帧推理，未以平移静态照片冒充真实视频。
- 原始数据/模型不进入Git或学生包；首次下载后记录地址、版本及SHA256于data/keypoints/download_manifest.json和data/neu_det/manifest.json。公开的下载来源清单同时保存在reference/real-data-downloads.json。原图/标注的权利不因课程MIT许可改变。
