# 第三方来源

本项目课程代码使用 MIT。以下第三方文件及依赖保留各自许可证，项目根 LICENSE 不改变其许可。

- `assets/photos`：OpenCV 4.12.0 仓库 `samples/data` 的 fruits.jpg、chicky_512.png（别名 chicky.png）、basketball1.png。仓库许可证副本见 `reference/OpenCV-LICENSE.txt`；精确URL、哈希见 `assets/SOURCES.json`。图像保持原样，教学输出注明了所做变换。
- 参考教程 `mohd-faizy/applied-computer-vision-opencv`，MIT，Copyright (c) 2026 mohd-faizy，固定提交 `5e7f194cc1a6ece51fc387d91ee4c70409506582`。许可证副本见 `reference/upstream-LICENSE.txt`。本地上游克隆独立保留，不自动安装或执行其代码；参考其课程布局后新增中文、矩阵和端到端小模型实验。
- digits：通过 scikit-learn 自带 `load_digits` 读取 UCI 手写数字数据；数据来源与引用见 [官方数据文档](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)。数据未单独改名发布为原创。
- NumPy、OpenCV、PyTorch、Matplotlib、SciPy、scikit-learn、Jupyter 作为依赖安装，其许可证按各上游发行包执行。
- 其余合成输入由各 Notebook 中的可见代码生成；没有使用学生个人信息或患者资料。

- 19课NEU-DET由东北大学Song Kechen等作者发布。固定镜像只作为下载通道；原始图像与标注不纳入本项目MIT授权，未把整个数据包重新发布到仓库/学生包。
- reference/opencv_zoo中的四个原版Python包装文件、README与LICENSE来自OpenCV Zoo固定提交48b10fa505aeb51fbd127fc60a5c660040e823b7，Apache-2.0，保留各模型LICENSE。部署权重按原始URL下载并核验SHA256。
- COCO-WholeBody照片/人工标注小样例来自MMPose固定版本；照片许可ID和来源保留于原始JSON。Google测试照片和TensorFlow.js视频按原始来源的许可使用，原始数据缓存不随仓库/学生包分发。Notebook中的教学可视化注明来源，原作者权利不变。
