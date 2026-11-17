# 运行验收记录

00–18验收日期2026-11-16；19–21新增真实场景验收日期2026-11-17。

## 实测环境与结果

- Windows 10，独立Python 3.11.14 venv，PyTorch2.5.1+cpu、NumPy2.2.6、OpenCV4.12.0.88，无新增依赖。
- 22篇均在各自独立新内核中运行，所有代码单元完成，无错误，图像嵌入Notebook。基础19篇于前次验证，新三篇于本次验证。
- 共149个代码单元，136幅嵌入PNG。耗时不包含首次下载，因硬件而异。
- 19课1800张真实图，1799个不同像素哈希；相同图不跨集合。训练1259、验证270、测试271。123张含多类框，保留全部标注，主类别单标签教学视角已明确。
- 19课仅由训练集拟合、验证集选权重（第23轮）：测试准确率0.9373、宏F1 0.9356、主类别联合框平均IoU 0.5683。定位分支增加LayerNorm以防Sigmoid饱和；整图框对照IoU为0.5889，说明平均IoU可能奖励过大的框，不能据此宣称定位优秀。保留失败例；不是完整多目标检测基准成绩。
- 20课实际运行21点与672维特征向量，处理真实手势视频前80帧；困难小手0.096置信度本应拒绝，oracle人工ROI仅作误差分析，不充当检测成功样本。
- 21课实际运行33主体点、17点人工映射、768维向量等价输出，处理下蹲视频全部65帧；低可见点不参与动作角度判断。小参考样例约10.92像素误差，不宣称独立测试准确率。
- 第一层卷积、19课池化、两种关键点输出头均与手算结果进行数值断言。59幅新增图经过文件有效性检查，人工查看代表性原图、通道、预测和视频骨架。
- 下载URL、版本、SHA256留档，模型与真实素材缓存核验通过。原始下载缓存不进入Git与学生包，首次执行19–21需要互联网。
- 基础学生包曾独立解压运行00课；新增包检查相对路径、辅助模块和第三方包装代码。

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
|19_industrial_defect_classification.ipynb|13|25|20.68|通过|
|20_hand_gesture_real.ipynb|15|17|14.47|通过|
|21_human_pose_real.ipynb|14|17|14.72|通过|

## 复核方法

```powershell
python scripts/run_notebooks.py --device cpu
python scripts/audit_course.py
```

执行明细在verification，生成图表/模型在outputs，两者不进入公共Git。Notebook完整保留实际运行图文。00–18不下载额外权重；19–21自动下载并校验缓存。GPU/CUDA/MPS、Linux执行及生产泛化未在本机实测。

基础课程的受控输入、模拟水下和CT范围说明继续保留。20–21固定预训练权重，几何手势和下蹲状态规则明确标识，不声称是已训练的通用手语/动作分类器。
