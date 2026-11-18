# 课程地图

全课程为 00–27 共 28 篇 Notebook，按编号顺序学习。数据增强独立成课，放在数据集评价之后、传统模型之前；旧 10–13 已从课程中移除。

|课程|Notebook|主题与完成标准|
|---:|---|---|
|00|[环境与第一张图](../notebooks/00_start_here.ipynb)|运行环境、图像读取与输出|
|01|[像素、颜色、采样与量化](../notebooks/01_pixels_matrices.ipynb)|解释矩阵、通道、采样与量化|
|02|[滤波、卷积和边缘](../notebooks/02_filtering_edges.ipynb)|在水果原图上比较滤波与 Canny|
|03|[形态学、轮廓和特征](../notebooks/03_morphology_contours.ipynb)|从水果与汇流排原图开始学习形态学和轮廓|
|04|[数据集与评价](../notebooks/04_datasets_evaluation.ipynb)|训练/验证/测试；理解数据泄漏和指标|
|05|[数据增强与工业视觉鲁棒性](../notebooks/05_data_augmentation.ipynb)|几何、颜色、噪声、Mosaic、MixUp、Copy-Paste、Random Erasing；保持框/掩膜同步|
|06|[传统视觉模型](../notebooks/06_classical_models.ipynb)|HOG 向量与线性 SVM|
|07|[图像分类](../notebooks/07_classification.ipynb)|CNN 输入、特征图、展平向量与概率|
|08|[目标检测](../notebooks/08_detection.ipynb)|框、类别与 IoU；真实汇流排图只作场景参照|
|09|[图像分割](../notebooks/09_segmentation.ipynb)|区分矩形框与逐像素掩膜|
|10|[图像检索](../notebooks/10_retrieval.ipynb)|橘子 ROI → 192 维颜色/空间特征 → Top-K|
|11|[光流](../notebooks/11_optical_flow.ipynb)|运动向量场、重映射与 EPE|
|12|[三维重建](../notebooks/12_reconstruction.ipynb)|相机观测、三角化和重投影误差|
|13|[姿态估计](../notebooks/13_pose.ipynb)|坐标矩阵与同一图像叠加显示|
|14|[CT 预处理](../notebooks/14_ct_preprocessing.ipynb)|真实公开 DICOM 原图、HU 和窗宽窗位|
|15|[水下图像](../notebooks/15_underwater.ipynb)|真实公开 EUVP 输入、参考图与传统增强|
|16|[真实手指手势识别](../notebooks/16_hand_gesture_real.ipynb)|关键点、ROI 逆变换与逐帧推理|
|17|[真实人体姿态与动作](../notebooks/17_human_pose_real.ipynb)|关键点映射、输出张量与动作角度|
|18|[真实工业缺陷分类与 ROI](../notebooks/18_industrial_defect_classification.ipynb)|NEU-DET 分组划分、特征通道与独立评价|
|19|[真实场景与工业案例](../notebooks/19_environment_and_real_cases.ipynb)|焊缝、金属表面与电池汇流排的标签边界|
|20|[暗光照明与夜间补光](../notebooks/20_lighting_and_night_fill.ipynb)|亮度、反光与画质门控|
|21|[灰度、对比度与 CLAHE](../notebooks/21_grayscale_contrast_clahe.ipynb)|暗光图像增强和限制|
|22|[二值化与形态学](../notebooks/22_binary_and_morphology.ipynb)|阈值、连通域和结构元素|
|23|[边缘与纹理](../notebooks/23_edges_and_texture.ipynb)|边缘密度不是七格焊接判定|
|24|[计数与小样本模板匹配](../notebooks/24_single_and_multi_count.ipynb)|计数、模板匹配和遮挡限制|
|25|[复杂场景与质量门控](../notebooks/25_complex_scenes_and_quality_gate.ipynb)|摆歪、遮挡、截断和拒判|
|26|[七格规则与证据数据库](../notebooks/26_evidence_database_and_private_case.ipynb)|七格至少五格合格、未知待复核和证据保存|
|27|[实时刀片电池汇流排检查](../notebooks/27_live_blade_busbar_inspection.ipynb)|汇流排 → 镍片 → 七个焊接位置|

## 数据边界

- CT 使用随项目提供的公开 DICOM 原图，不用生成图，也不用于诊断。
- 水下使用公开 EUVP 实拍输入与配对参考，不使用水果图或 AI 生成图。
- 数据增强附件是课件原图；正式实验由 Notebook 代码重新生成。增强图是派生图，不新增真实缺陷真值。
- 汇流排实拍候选框不是人工真值；七格未标注时必须待复核。
- 普通焊缝公开图不能代替刀片电池七格焊接标签。

统一图文结构：原始输入 → 处理步骤 → 中间矩阵/张量/向量 → 输出 → 局限与复核。
