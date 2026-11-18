"""Create the ten source notebooks. This generator is not needed for normal use."""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf

INDUSTRIAL_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


SETUP = code('''
from pathlib import Path
import sys, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_utils.py").is_file())
INDUSTRIAL_ROOT = ROOT / "courses" / "industrial-vision-lab"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INDUSTRIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(INDUSTRIAL_ROOT))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import lab_utils as lab
print("OpenCV", cv2.__version__, "| NumPy", np.__version__)
''')
SOURCES = md('''
## 来源与练习边界

公开焊缝：sukmaadhiwijaya / rikkarth，固定提交 `8b80d1afe88c0b946500a498cbdb9d7a1eda5533`，数据卡声明 CC0-1.0；工业表面：Kolektor Group / ViCoS，CC BY-NC-SA 4.0；分散零件：Tung1213，CC BY-SA 3.0；重叠零件：Technologyfelix，CC BY-SA 4.0。相关照片和派生图保留原许可及署名。

完整来源、原许可页、下载路径及 SHA256 见 [DATA_SOURCES.md](../courses/industrial-vision-lab/DATA_SOURCES.md) 和 [source_manifest.json](../courses/industrial-vision-lab/sources/source_manifest.json)。本课是非商业教学。灰度、二值、叠框及扰动均为原照片的派生图，七格合成图另有醒目标记。它们不等于新增独立验收数据，也不证明电池焊接准确率或机械强度。
''')

LESSONS = [
("22_environment_and_real_cases", "22｜先看真实案例和环境", [
md('''
# 00｜先看真实案例和环境

这套课面向第一次使用 Python 图像处理的人。先观察照片，再运行一个小算法，最后检查算法在哪些条件下失效。每本都能单独从上到下执行，建议先读本课。

**操作：**在 JupyterLab 打开笔记本，选中代码格按 `Shift + Enter`。第一次从第一个代码格开始。变量保存当前笔记本的图像，下一格会继续使用；重启内核后应重新逐格运行。

学完本课应能区分：真实照片、原始标注、算法输出、人工真值、派生图。我们使用三个任务：焊缝外观、金属表面缺陷、真实零件数量。没有把它们伪装成刀片电池的七格数据。
'''),
SETUP,
md('''
## 1. 确认数据没有被换掉

读图之前核对 SHA256。它能发现文件变化，但不能证明标注正确或不同照片来自不同零件。下载器只取少量公开样本，完整工业档案保存在被忽略的缓存中。
'''),
code('''
rows = lab.manifest()["files"]
image_rows = [row for row in rows if row["kind"] == "image"]
for row in image_rows:
    image = lab.load_case(row["id"])
    print(row["id"], image.shape, row["license"], row["sha256"][:16])
assert len({row["dataset"] for row in image_rows}) >= 3
assert not lab.manifest()["private_images_included"]
'''),
md('''
## 2. 看三类真实照片

焊缝照片研究局部纹理；Kolektor 照片研究表面裂纹；螺钉照片研究数量。反光、边界、尺度不同，同一套阈值不应直接照搬。分散零件原作者说明有六颗螺钉和两颗金属球，这里把“所有可见零件”作为计数口径。
'''),
code('''
lab.show_images([lab.load_case("weld_good"), lab.load_case("kolektor_defect_0"),
                 lab.load_case("parts_separated")],
                ["真实焊缝：CC0 来源语义", "真实工业表面：原始缺陷掩膜", "真实零件：6 螺钉 + 2 球"], cols=3)
'''),
md('''
## 3. 标注不是算法预测

下面黄色框来自公开焊缝数据的原始 YOLO 标注；本课没有训练或运行检测器。显示原标注只是为了理解框坐标。YOLO 的中心坐标和宽高以整图归一化；显示时要乘原图宽高，不能固定乘 640。
'''),
code('''
image = lab.load_case("weld_good")
labels = lab.pixel_yolo_labels("weld_good", image)
display(labels)
lab.show_images([image, lab.draw_boxes(image, labels)], ["原始实拍", "上游标注叠框（不是推理）"], cols=2)
'''),
md('''
## 动手检查

把 `weld_good` 改成 `weld_crack`，比较标注。再看图片的 `shape`：顺序是高、宽、通道。OpenCV 读入彩色图片用 BGR 顺序，本课显示函数会转换成 RGB。若发现“数量正确但框错了”，也不能算定位完成。

产线迁移顺序：拍清楚 → 定位汇流排 → 找镍片 → 对齐焊格 → 判七格外观 → 保存证据。强度与内部熔合不能仅凭二维外观保证。
'''), SOURCES]),
("23_lighting_and_night_fill", "23｜先打好光：夜间补光和像素", [
md('''
# 01｜先打好光：夜间补光和像素

目标是知道“画面更亮”为什么不一定“焊纹更清楚”。本课读取真实焊缝照片，再人为降低亮度、加噪声和模糊，做可重复的派生演示。这些图不是现场夜间实测，也不是不同灯具的实拍对比。

暗环境表示环境光不足；**光学暗场**是一种低角度照明布置，让平面直接反光偏离镜头、让纹理和边缘散射入镜。二者不能混用。对于反光金属，可先比较大面积漫射光和左右低角度条形光：漫射光用于定位，侧光用于看纹理。偏振对裸金属的改善可能有限，需实际比较。

参考：[Basler 照明基础](https://www.baslerweb.com/en/learning/illumination/)、[暗场照明](https://www.baslerweb.com/en/illumination/dark-field-light/)。以下只演示像素变化，不宣称复现这些光学布置。
'''), SETUP,
code('''
image = lab.load_case("weld_good")
dark = lab.degrade(image, 0.25)
dark_noisy = lab.degrade(image, 0.25, noise=9)
blurred = lab.degrade(image, 0.65, blur=11)
lab.show_images([image, dark, dark_noisy, blurred],
                ["真实原图", "派生：亮度 ×0.25", "派生：低亮 + 噪声", "派生：模糊，模拟细节丢失"], cols=2)
for name, sample in [("原图", image), ("低亮", dark), ("低亮噪声", dark_noisy), ("模糊", blurred)]:
    print(name, lab.image_quality(sample))
'''),
md('''
## 1. 看四个指标，但不迷信它们

平均灰度描述整体明暗；过暗/过曝比例描述落在极端范围的像素；拉普拉斯方差可作为清晰度线索。噪声也会使方差变大，所以“分数高”不能独自证明对焦正确。整图指标还会被黑背景或亮金属误导，最终要在镍片焊纹 ROI 上测。
'''),
code('''
brightened = np.clip(dark_noisy.astype(np.float32) * 4, 0, 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab.gray(dark_noisy))
lab.show_images([lab.gray(image), lab.gray(brightened), clahe],
                ["原图灰度参考", "派生低亮噪声再放大：噪声也放大", "CLAHE：增强对比，不恢复真实细节"], cols=3)
display(HTML(lab.parameter_gallery([image, dark, dark_noisy, blurred],
    ["真实原图", "派生低亮", "派生低亮带噪声", "派生模糊"], "lighting")))
'''),
md('''
## 2. 算一格焊纹能占多少像素

物方采样约等于视野宽度 / 水平像素数。若一格最小可见纹理宽度为 0.15 mm，先用 8 像素做工程试验起点，这不是通用验收标准。镜头解析力、失焦、景深、压缩和运动都会使有效细节少于理论像素。

焦距主要改变视野和工作距离关系；远心镜头主要减少物体高度变化引起的倍率/透视变化，适合尺寸测量。远心镜头不会凭空增加像素或照亮焊纹。当前先实现“是否焊过”，应优先确认局部像素和打光，再决定是否需要远心。
'''),
code('''
sensor_width_px = 1920
feature_mm = 0.15
for fov_mm in [60, 120, 240]:
    feature_px = feature_mm * sensor_width_px / fov_mm
    print(f"视野 {fov_mm:3d} mm → 0.15 mm 特征约 {feature_px:.1f} px")
min_feature_px = 8  # 仅示例设计起点，必须用真实焊格复核
print("此假设下最大视野宽度 mm：", feature_mm * sensor_width_px / min_feature_px)
'''),
md('''
## 3. 现场四步试光记录

固定相机和工件距离，使用完整原图，不拍屏幕；分别记录环境光、漫射补光、左侧低角度、右侧低角度四种真实画面。尽量保持曝光、增益和白平衡一致；运动时缩短曝光，避免靠长曝光提亮。灯的频闪也要通过连续帧观察。

先用可见光比较清晰度与反光；只有真实对比表明特征分离改善，才决定红光、蓝光或近红外。RGB 通道拆分不是光谱实验，软件不能由一张普通照片证明换波长有效。保存请求值和实际回读，设备不支持曝光/对焦回读时填未知。
'''),
code('''
capture_recipe = {"camera_name": "OsmoAction4", "width": 1920, "height": 1080,
    "fps": 30, "max_detection_hz": 5,
    "lighting": {"scene": "night_fill_light", "lamp_on": True, "auto_exposure": True,
                 "exposure": None, "autofocus": True},
    "actual_exposure": None, "actual_autofocus": None,
    "note": "教学配置示例，不是相机已执行或已回读的证据"}
display(capture_recipe)
assert capture_recipe["actual_exposure"] is None
'''), SOURCES]),
("24_grayscale_contrast_clahe", "24｜灰度、直方图和局部对比", [
md('''
# 02｜灰度、直方图和局部对比

目标：把一张真实金属表面照片转成灰度，观察亮度分布，再比较全局均衡和 CLAHE。灰度不是缺陷检测，只是把每个像素表示为一个亮度值。CLAHE 在小网格中增强对比，并限制过强放大，但仍可能增强噪声。

原理参考：[OpenCV 官方均衡教程](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)。示例代码为本项目重新编写。
'''), SETUP,
code('''
image = lab.load_case("kolektor_defect_0")
g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(g)
enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
lab.show_images([image, g, equalized, enhanced],
                ["真实工业表面", "灰度", "派生：全局均衡", "派生：CLAHE"], cols=4)
'''),
md('''
## 1. 直方图回答“有多少明暗像素”

横轴是灰度 0–255，纵轴是像素数量。直方图没有位置关系：裂纹在哪里、两块金属是否相邻都不能由它直接读出。峰值不等于某一种物体。
'''),
code('''
fig, ax = plt.subplots(figsize=(10, 4))
for sample, name in [(g, "原灰度"), (equalized, "全局均衡"), (enhanced, "CLAHE")]:
    hist = cv2.calcHist([sample], [0], None, [256], [0, 256]).ravel()
    ax.plot(np.arange(256), hist / hist.sum(), label=name)
ax.set(xlabel="灰度值", ylabel="像素比例", title="同一真实照片的派生分布")
ax.legend(); fig.tight_layout(); plt.show(); plt.close(fig)
print("图像尺寸", g.shape, "类型", g.dtype, "范围", int(g.min()), int(g.max()))
'''),
md('''
## 2. 参数只改变效果，不创造真值

提高 `clipLimit` 可以更强地增强局部差异，也可能把噪声和正常加工纹理增强。观察下方三个值，并在原图上复核。不能因为处理后“看着有纹理”就判为焊好。
'''),
code('''
clips = [1.0, 2.0, 6.0]
alternatives = [cv2.createCLAHE(clipLimit=v, tileGridSize=(8, 8)).apply(g) for v in clips]
lab.show_images(alternatives, [f"CLAHE clipLimit={v}" for v in clips], cols=3)
display(HTML(lab.parameter_gallery(alternatives, [f"clipLimit={v}" for v in clips], "clahe")))
assert enhanced.shape == g.shape and enhanced.dtype == np.uint8
'''),
md('''
## 小练习

把案例改成 `kolektor_normal_0`，检查正常纹理是否同样被增强。再把 `tileGridSize` 从 `(8, 8)` 改为 `(4, 4)`，观察局部增强范围。记录参数和原文件摘要，避免日后无法重现。
'''), SOURCES]),
("25_binary_and_morphology", "25｜二值化与形态学", [
md('''
# 03｜二值化与形态学

目标：把真实照片变成黑白前景，理解阈值和形态学会怎样改变细小结构。白色表示算法当前选中的像素，**不自动等于缺陷、镍片或已焊**。

参考：[OpenCV 阈值](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)、[形态学](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)。
'''), SETUP,
code('''
g = lab.load_case("kolektor_defect_0", grayscale=True)
smooth = cv2.GaussianBlur(g, (5, 5), 0)
_, fixed = cv2.threshold(smooth, 110, 255, cv2.THRESH_BINARY_INV)
otsu_value, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 7)
lab.show_images([g, fixed, otsu, adaptive],
                ["真实原灰度", "固定阈值 110", f"Otsu 自动阈值 {otsu_value:.0f}", "自适应 31 / C=7"], cols=4)
'''),
md('''
## 1. 三种阈值的适用条件

固定阈值需要亮度稳定；Otsu 根据全图灰度分布选阈值，前景背景难分时也会失败；自适应阈值根据局部邻域判断，可能同时抓出划痕、噪声和原有纹理。`blockSize` 要是大于 1 的奇数。

现在改变同一张实拍照片的亮度，观察固定阈值漂移。下方是派生挑战，不是新的照片或独立准确率测试。
'''),
code('''
samples, titles = [], []
for factor in [0.6, 1.0, 1.3]:
    derived = lab.degrade(g, factor)
    _, mask = cv2.threshold(derived, 110, 255, cv2.THRESH_BINARY_INV)
    samples.append(mask); titles.append(f"派生亮度 ×{factor}，白像素 {np.mean(mask > 0):.1%}")
lab.show_images(samples, titles, cols=3)
'''),
md('''
## 2. 开闭运算是“修改形状”

开运算先腐蚀后膨胀，常移除小白点，也会删掉细裂纹。闭运算先膨胀后腐蚀，常连上断开的白线，也会把相邻零件粘在一起。核大小必须和原图像素尺度一起记录，不能在缩放后不变地沿用。
'''),
code('''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
big_open = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
lab.show_images([adaptive, opened, closed, big_open],
                ["原二值图", "3×3 开运算", "3×3 闭运算", "11×11 开运算：细结构可能被删"], cols=4)
assert set(np.unique(opened)).issubset({0, 255})
'''),
md('''
## 对照真实标注，而不是凭处理图下结论

掩膜来自上游人工标注，仅作观察参考。下图不把掩膜送给算法，也不以本课六张示例调出一个“已验收阈值”。
'''),
code('''
truth = lab.load_case("kolektor_defect_0_mask", grayscale=True)
image = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
overlay = image.copy(); overlay[truth > 0] = (40, 40, 230)
lab.show_images([image, overlay, adaptive],
                ["真实原图", "原始人工缺陷掩膜叠加", "算法二值候选：并非缺陷真值"], cols=3)
'''), SOURCES]),
("26_edges_and_texture", "26｜边缘与纹理：看见细节", [
md('''
# 04｜边缘与纹理：看见细节

目标：区分零件边界、表面纹理和加工缺陷。Canny 找亮度变化处；局部标准差描述一个小窗口的亮度变化强弱；Gabor 滤波器响应某个方向的纹理。这些都不是“焊接强度计”。

参考：[OpenCV Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)。本课在真实焊缝照片上计算派生特征。
'''), SETUP,
code('''
image = lab.load_case("weld_crack")
g = lab.gray(image)
smooth = cv2.GaussianBlur(g, (5, 5), 0)
edges = cv2.Canny(smooth, 40, 100)
std = lab.local_std(g, 11)
std_view = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
lab.show_images([image, edges, std_view],
                ["真实焊缝", "派生：Canny 边缘", "派生：11×11 局部标准差"], cols=3)
'''),
md('''
## 1. 方向纹理

Gabor 核的方向改变时，同一焊纹的响应会变化。这说明不能只靠一个固定方向阈值处理任意旋转。要么先对齐工件方向，要么显式考虑多个方向，并用真实例子检验。下方仅查看滤波响应，不训练分类器。
'''),
code('''
responses = []
angles = [0, 45, 90, 135]
for angle in angles:
    kernel = cv2.getGaborKernel((21, 21), 4.0, np.deg2rad(angle), 9.0, 0.6, 0, ktype=cv2.CV_32F)
    response = cv2.filter2D(g.astype(np.float32), cv2.CV_32F, kernel)
    responses.append(cv2.normalize(np.abs(response), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
lab.show_images(responses, [f"方向响应 {angle}°" for angle in angles], cols=4)
'''),
md('''
## 2. 光斑和噪声也会产生“纹理”

边缘密度只是边缘像素比例。把正常照片加噪声，会出现很多新的梯度。将这个结果直接当成“焊格数量”会出错。真实七格需要先定位镍片和焊纹区，再对齐、分格、判断可见性。
'''),
code('''
reference = lab.load_case("weld_good", grayscale=True)
noisy = lab.degrade(reference, 0.6, noise=25)
edge_ref = cv2.Canny(cv2.GaussianBlur(reference, (3, 3), 0), 40, 100)
edge_noise = cv2.Canny(cv2.GaussianBlur(noisy, (3, 3), 0), 40, 100)
lab.show_images([reference, edge_ref, noisy, edge_noise],
                ["真实焊缝灰度", "原图边缘", "派生：噪声挑战", "噪声边缘"], cols=4)
print("边缘密度，原图 / 派生噪声：", round(float(np.mean(edge_ref > 0)), 4), round(float(np.mean(edge_noise > 0)), 4))
'''), SOURCES]),
("27_single_and_multi_count", "27｜单件、多个与当前视野计数", [
md('''
# 05｜单件、多个与当前视野计数

目标：在真实分散零件照片中提取连通域、按当前画面编号，并理解“连通域数量”和“物理零件数量”的差别。原作者说明六颗螺钉、两颗小金属球，因此本例全部可见零件为 8 个；如果任务只数螺钉，真值应为 6，不能偷换口径。

编号只表示当前帧的顺序，不是跨帧追踪身份，也不是预先填好的 2×4 固定位置。
'''), SETUP,
code('''
image = lab.load_case("parts_separated")
g = lab.gray(image)
# 本照片背景集中在中灰；同时取暗轮廓与亮反光，参数仅针对这个受控案例。
raw = np.where((g < 110) | (g > 190), 255, 0).astype(np.uint8)
# 原照片最右有一条白边，排除贴图像边界的成分，不能当成零件。
raw[:, -2:] = 0
kernel = np.ones((13, 3), np.uint8)  # 竖向连断纹，过宽会合并邻件
closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
boxes = lab.component_boxes(closed, min_area=200)
boxes = [b for b in boxes if b["bbox"][2] > 5 and b["bbox"][3] > 5]
lab.show_images([image, closed, lab.draw_boxes(image, boxes)],
                ["真实照片：6 螺钉 + 2 球", "派生候选二值图", f"算法观察到 {len(boxes)} 个区域，黄色待复核"], cols=3)
print("人工口径：全部可见零件 8；算法区域：", len(boxes), "；差值：", len(boxes) - 8)
'''),
md('''
## 1. 看每个区域，不只看总数

总数相同仍可能同时漏一件、多一件。先显示每件框、面积和中心，再人工检查一一对应。当前函数按画面行和横坐标排序；若要稳定追踪同一个零件，需另加关联逻辑，不能把当前编号存成工件 ID。
'''),
code('''
for index, row in enumerate(boxes, 1):
    print(f"#{index}", "bbox(px)=", row["bbox"], "area=", row["area"])
if boxes:
    x, y, width, height = boxes[0]["bbox"]
    crop = image[y:y + height, x:x + width]
    lab.show_images([crop], ["由当前检测框切出的单件（需人工复核框是否完整）"], cols=1)
'''),
md('''
## 2. 先把变化背景扣掉，再观察受控小视野

上面的全局灰度范围产生粘连。现在用大窗口中值滤波估计缓变背景，再看偏离背景的亮暗像素。这里额外展示左侧人工选取的四个可见零件，让你比较局部成功与全景失败；这个人工裁切只是教学，实际系统必须由汇流排定位器产生 ROI，不能写死固定位置冒充实时检测。
'''),
code('''
background = cv2.medianBlur(g, 51)
contrast = cv2.absdiff(g, background)
local_mask = np.where(contrast > 25, 255, 0).astype(np.uint8)
local_mask[:, -3:] = 0
local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))
full_candidates = lab.component_boxes(local_mask, min_area=150)
left_roi = image[:, :300]  # 教学人工 ROI：两螺钉 + 两金属球，非生产定位逻辑
left_candidates = lab.component_boxes(local_mask[:, :300].copy(), min_area=150)
lab.show_images([background, lab.draw_boxes(image, full_candidates), lab.draw_boxes(left_roi, left_candidates)],
    ["派生：估计缓变背景", f"全图候选 {len(full_candidates)} / 真值 8：仍需复核",
     f"人工教学小视野 {len(left_candidates)} / 真值 4"], cols=3)
print("受控 ROI 的区域数量：", len(left_candidates), "；整图仍有两件粘连，不能绿灯。")
assert len(left_candidates) == 4  # 只验证这张公开教学照片的可重复性，不代表独立准确率
'''),
md('''
## 3. 核大小会改变数量

本例的亮暗反光使一个螺钉可能碎成多个区域；过宽的闭运算又可能把两颗相邻螺钉合在一起。不要自动挑一个“刚好等于 8”的参数后把它作为可靠性证明。
'''),
code('''
views, labels = [], []
for kw, kh in [(3, 3), (3, 13), (15, 15)]:
    candidate = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((kh, kw), np.uint8))
    candidates = lab.component_boxes(candidate, min_area=200)
    views.append(lab.draw_boxes(image, candidates))
    labels.append(f"核 {kw}×{kh}：{len(candidates)} 个区域")
lab.show_images(views, labels, cols=3)
display(HTML(lab.parameter_gallery(views, labels, "count")))
'''),
md('''
## 迁移到刀片电池

先检测当前视野中实际可见汇流排，再分割每块上的镍片。N×M 只可作配方中的预期数量，不能代替当前图像的实测数量。被遮挡、出画或粘连的目标应标出原因并待复核。多个摄像头若视野重叠，先校准和去重，再算整包数量。
'''), SOURCES]),
("28_few_sample_template", "28｜小样本模板：快速起步与边界", [
md('''
# 06｜小样本模板：快速起步与边界

目标：从真实零件图裁一个模板，在原图中找位置，并看到旋转、尺度变化的影响。模板匹配适合外观、角度和尺度受控的定位原型；匹配分数不是缺陷概率，更不是焊接合格概率。

参考：[OpenCV 模板匹配](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)。下面模板来自同一张图，属于自匹配演示，不能作为独立测试。
'''), SETUP,
code('''
image = lab.load_case("parts_separated")
g = lab.gray(image)
# 人工查看本公开照片后选第一颗螺钉的完整窗口，xyxy，像素单位。
x1, y1, x2, y2 = 20, 90, 75, 212
template = g[y1:y2, x1:x2]
assert template.size and template.std() > 1
scores = cv2.matchTemplate(g, template, cv2.TM_CCOEFF_NORMED)
matches = lab.suppress_matches(scores, template.shape, threshold=0.80)
lab.show_images([template, scores, lab.draw_boxes(image, matches)],
                ["人工裁出的模板", "位置相似度热图（非概率）", f"自匹配候选 {len(matches)} 处"], cols=3)
print("最大自匹配相似度", cv2.minMaxLoc(scores)[1])
'''),
md('''
## 1. 限制重复框

相邻位置会得到相近分数，因此需要空间抑制，避免一颗螺钉输出很多重叠框。本课做简单中心邻域抑制；它不处理可变尺度重叠的全部情况。实际系统应记录原图尺寸和目标框的坐标系。
'''),
code('''
views, captions = [], []
for angle in [0, 15, 45]:
    matrix = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), angle, 1)
    rotated = cv2.warpAffine(g, matrix, (g.shape[1], g.shape[0]), borderValue=int(np.median(g)))
    result = cv2.matchTemplate(rotated, template, cv2.TM_CCOEFF_NORMED)
    _, value, _, point = cv2.minMaxLoc(result)
    row = {"bbox": [point[0], point[1], template.shape[1], template.shape[0]], "status": "REVIEW"}
    views.append(lab.draw_boxes(rotated, [row]))
    captions.append(f"派生旋转 {angle}°，最高相似度 {value:.3f}")
lab.show_images(views, captions, cols=3)
'''),
md('''
## 2. 什么时候应停止堆模板

如果同一位置因打光、反光、旋转反复变样，先处理采集条件与工件对齐。模板可用于定位已知汇流排轮廓，但不能凭一张模板证明镍片焊接状态。对于多型号，应把相同尺寸零件的定位与具体配方规则分开。

要评估泛化，需把同一物理零件、同一视频连续帧及其所有增强保持在同一组。旋转、调亮、换背景不能增加独立来源数。当前自匹配分数绝不填入“准确率”。
'''), SOURCES]),
("29_complex_scenes_and_quality_gate", "29｜复杂场景与拒判", [
md('''
# 07｜复杂场景与拒判

目标：看见一个可重复运行的算法如何在真实遮挡和反光中失败，并设计黄色待复核路径。下面是真实螺钉盒照片，存在重叠、局部遮挡、反光和复杂背景。没有完整零件数量真值，所以只输出候选区域数，不编造计数准确率。
'''), SETUP,
code('''
image = lab.load_case("parts_occluded")
image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
g = lab.gray(image)
smooth = cv2.GaussianBlur(g, (5, 5), 0)
otsu_value, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
candidates = lab.component_boxes(closed, min_area=150)
lab.show_images([image, closed, lab.draw_boxes(image, candidates)],
                ["真实遮挡螺钉盒", f"Otsu 阈值 {otsu_value:.0f}", f"只有候选区域 {len(candidates)} 个，不等于物理数量"], cols=3)
print({"status": "REVIEW", "observed_regions": len(candidates), "true_count": None,
       "reason": "遮挡和反光，尚无逐件真值，不能确认全数检出"})
'''),
md('''
## 1. 画质门控需要在任务 ROI 标定

下面阈值是教学示例，不是工厂通用标准。模糊可能使清晰度下降，噪声也可能使其上升。质量通过表示“值得继续判断”，并不表示焊接合格。边界出画、遮挡和未知七格仍要单独拒判。
'''),
code('''
def quality_gate(roi):
    q = lab.image_quality(roi)
    reasons = []
    if q["dark_ratio"] > 0.35: reasons.append("过暗比例过高")
    if q["overexposed_ratio"] > 0.20: reasons.append("过曝比例过高")
    if q["sharpness"] < 35: reasons.append("缺少清晰纹理，需检查对焦或曝光")
    return {"eligible": not reasons, "status": "REVIEW", "quality": q,
            "reason": "；".join(reasons) if reasons else "画质门控通过，仍需工件与七格检查"}

real = lab.load_case("weld_good")
challenges = [real, lab.degrade(real, 0.08), lab.degrade(real, 1.8), lab.degrade(real, 0.7, blur=25)]
for label, candidate in zip(["实拍参考", "派生过暗", "派生过亮", "派生模糊"], challenges):
    print(label, quality_gate(candidate))
assert quality_gate(challenges[1])["eligible"] is False
'''),
md('''
## 2. 看“错误绿灯”更重要

假设焊格算法在模糊图上仍返回七个 OK，前置画质门控应拦下。这是控制流程测试，不是视觉识别成绩。实际评估要分别记漏检、误检、错误放行、不必要拒判，以及处理延迟。
'''),
code('''
fake_slots = ["OK"] * 7  # 明确：这里是流程测试输入，不是假装模型输出
quality_result = quality_gate(challenges[1])
decision = lab.seven_slot_verdict(fake_slots, quality_clear=quality_result["eligible"])
display(decision)
assert decision["status"] == "REVIEW"
'''),
md('''
## 最小现场验收表

每个新物理样本保留原图、逐件汇流排框、镍片存在性、七格人工标签、光照/角度、结果与原因。按来源分组冻结验收集，不能看完错例再在同一集上反复调参并称独立成绩。对“至少 90%”必须先确认口径是逐格、逐镍片还是整帧，以及黄色如何计入。
'''), SOURCES]),
("30_seven_slots_and_unknown", "30｜七格至少五格与未知状态", [
md('''
# 08｜七格至少五格与未知状态

目标：把业务规则写成不会悄悄“误绿”的代码。先看真实焊缝，然后用合成七格示意验证逻辑。**当前公开照片没有电池七格真值，本课没有训练七格识别器。**

正确顺序是汇流排定位 → 镍片存在性 → 镍片/焊纹对齐 → 七格可见性与外观判定 → 规则。七格中至少五格符合人工确认的外观标准才可能绿；本课采用保守策略，任一格看不清先黄，完整清晰但不足五格红。缺镍片确认后红。
'''), SETUP,
code('''
lab.show_images([lab.load_case("weld_good")],
                ["真实公开焊缝：类别语义不同，没有刀片电池七格标签"], cols=1)
'''),
md('''
## 1. 合成示意只用于规则

下面画的条纹不是实际焊点。七格沿着对齐后的焊接排布方向编号，而不是随意把整张图片平均切成七份。真实系统需要从镍片窗口剔除柱子、背景、强反光和边界后确定格位。
'''),
code('''
def schematic(slots):
    canvas = np.full((150, 630, 3), 225, np.uint8)
    colors = {"OK": (60, 160, 65), "NG": (40, 45, 205), None: (30, 185, 225)}
    for index, state in enumerate(slots):
        left = 10 + index * 88
        cv2.rectangle(canvas, (left, 30), (left + 75, 115), colors[state], 3)
        if state == "OK":
            for y in [55, 65, 75, 85]: cv2.line(canvas, (left + 13, y), (left + 62, y), (90, 90, 90), 2)
        cv2.putText(canvas, str(index + 1) + ("?" if state is None else ""), (left + 20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2)
    return canvas

examples = [["OK"] * 5 + ["NG"] * 2, ["OK"] * 4 + ["NG"] * 3,
            ["OK"] * 5 + ["NG", None]]
lab.show_images([schematic(slots) for slots in examples],
                ["合成规则示意：5/7 → OK", "合成规则示意：4/7 → NG", "合成规则示意：未知一格 → REVIEW"], cols=1)
'''),
md('''
## 2. 运行规则和异常输入

未知不能用 0.5 概率或文件名强行填补。缺片、出画、不可判、少于七格都要明确原因。二维外观 OK 仅表示满足本规则，不等于内部熔合或机械强度达标。
'''),
code('''
cases = [
    ("清晰5格", examples[0], {}, "OK"),
    ("清晰4格", examples[1], {}, "NG"),
    ("仍有未知", examples[2], {}, "REVIEW"),
    ("确认缺镍片", [], {"nickel_present": False}, "NG"),
    ("镍片未知", ["OK"] * 7, {"nickel_present": None}, "REVIEW"),
    ("仅有六格", ["OK"] * 6, {}, "REVIEW"),
    ("工件出画", ["OK"] * 7, {"busbar_visible": False}, "REVIEW"),
    ("画质差", ["OK"] * 7, {"quality_clear": False}, "REVIEW"),
]
for name, slots, options, expected in cases:
    result = lab.seven_slot_verdict(slots, **options)
    print(name, result["status"], result["reason"])
    assert result["status"] == expected, name
'''),
md('''
## 3. 别把坐标单位混起来

先记录整帧尺寸，再记录局部 ROI 的原图坐标。检测输入若缩放/补边，必须把框映射回原图；分格处理使用 ROI 自己的宽高。下面用简单数值展示矩形窗口转换，不宣称该窗口是实际电池镍片。
'''),
code('''
frame_width, frame_height = 1920, 1080
roi = [600, 300, 210, 70]  # xywh 原图像素，纯坐标示例
normalized = [roi[0] / frame_width, roi[1] / frame_height,
              roi[2] / frame_width, roi[3] / frame_height]
restored = np.array(normalized) * np.array([frame_width, frame_height, frame_width, frame_height])
assert np.allclose(restored, roi)
print("归一化 xywh", normalized)
print("ROI 七格边界（只有对齐后才有业务意义）", np.linspace(roi[0], roi[0] + roi[2], 8).tolist())
'''), SOURCES]),
("31_evidence_database_and_private_case", "31｜保存证据、数据库与私有样本", [
md('''
# 09｜保存证据、数据库与私有样本

目标：将原始图摘要、参数、质量、结果、原因一起保存，并避免重复写入同一事件。用 SQLite 演示本地存储，不需要数据库服务器或口令。现场系统可以再对接 PostgreSQL/MySQL 和 MES，但不能把“放进待同步表”显示成“MES 已接收”。

下方记录明确为教学规则输入，来源是公开照片；不伪装成实时检测器结果。数据库文件被 Git 忽略，执行笔记本不会连接任何远程系统。
'''), SETUP,
code('''
source = lab.case_row("weld_good")
image = lab.load_case("weld_good")
quality = lab.image_quality(image)
result = lab.seven_slot_verdict(["OK"] * 5 + ["NG"] * 2)
result["basis"] = "synthetic_rule_input_not_visual_inference"
recipe = {"name": "teaching-rule-v1", "required_good": 5, "total_slots": 7,
          "unknown_policy": "REVIEW", "lighting_actual": "unknown"}
database = lab.OUTPUT / "teaching_inspections.sqlite3"
connection = lab.create_database(database)
event_key = "teaching-v1:" + source["sha256"] + ":5-of-7-synthetic-rule"
record_id, inserted = lab.save_inspection(connection, event_key=event_key,
    source_id=source["id"], source_sha256=source["sha256"], evidence_path=source["file"],
    result=result, recipe=recipe, quality=quality)
same_id, inserted_twice = lab.save_inspection(connection, event_key=event_key,
    source_id=source["id"], source_sha256=source["sha256"], evidence_path=source["file"],
    result=result, recipe=recipe, quality=quality)
assert record_id == same_id and inserted_twice is False
print("记录 ID", record_id, "首次新增", inserted, "重复提交新增", inserted_twice)
'''),
md('''
## 1. 结果表与待同步表在同一事务中写入

本地结果和 outbox 一起提交，减少“结果已存但待发送事件丢失”的情况。发送服务需要用 `event_key` 做幂等、确认服务端回执后才改 `synced`；网络失败可以重试。这里没有 MES 地址，也不模拟成功回执。

实际实时视频的 `event_key` 应包含采集会话和帧/工件标识，不能只用图像摘要，否则两次合法检测可能被误合并。
'''),
code('''
print("检测结果", connection.execute("SELECT id,source_id,status,observed FROM inspections").fetchall())
print("真实同步状态", connection.execute("SELECT inspection_id,state,attempts FROM sync_outbox").fetchall())
assert connection.execute("SELECT state FROM sync_outbox WHERE inspection_id=?", (record_id,)).fetchone()[0] == "pending"
connection.close()
public_record = {"source": source["id"], "source_sha256": source["sha256"],
                 "quality": quality, "recipe": recipe, "result": result,
                 "sync": {"state": "pending", "remote_ack": None}}
(lab.OUTPUT / "teaching_inspection.json").write_text(json.dumps(public_record, ensure_ascii=False, indent=2), encoding="utf-8")
display(public_record)
'''),
md('''
## 2. 本地导入私有刀片电池样本

默认课程从不读取 `data/private`，公开网页只包含上述许可明确的公开照片。需要练现场九块汇流排时，在终端显式导入自己的图片，例如：

```powershell
conda run -n pyg python scripts/import_private.py "D:\\your-private-folder\\battery.jpg" --expected-count 9
```

这里的 9 必须来自人工确认，不是算法成绩。导入器把像素重新编码成 PNG、去掉原 EXIF，把 SHA256 和人工数量保存在 `data/private`；七格真值仍为未知。该目录和 `outputs/private` 均被 Git 忽略。请在私有笔记本另行处理，不把私人输出写回公开教材。

实时采集还需要真实摄像头服务，不能用轮播照片冒充。采集线程只保留最新帧；检测线程处理完成后再取新帧；图像、检测框和 frame_id 原子切换。设备不支持对焦/曝光回读时，数据库字段保留空值。
'''),
code('''
# 不读取私有目录；只确认默认数据入口确实是公开清单。
assert all(not row["file"].startswith("data/private/") for row in lab.manifest()["files"])
print("默认课程不会导出私有像素；真实相机、MES 和七格视觉识别尚需各自现场验收。")
'''),
md('''
## 最后的复现检查

重新执行本课，同一事件不应重复插入；换成新的会话事件应正常新增。打开 `outputs/public/teaching_inspection.json`，检查结果理由、来源摘要和 pending 状态。完成课程表示掌握一套可运行的图像与数据流程，不等于工业产品已达到任何准确率要求。
'''), SOURCES]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-generated", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "notebooks"
    directory.mkdir(parents=True, exist_ok=True)
    for slug, title, cells in LESSONS:
        output = directory / f"{slug}.ipynb"
        if output.exists() and not args.overwrite_generated:
            raise FileExistsError(f"Preserving existing notebook {output.name}; use --overwrite-generated only intentionally")
        notebook = nbf.v4.new_notebook(cells=[nbf.from_dict(cell) for cell in cells], metadata={
            "kernelspec": {"display_name": "Python (current environment)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "industrial_vision_lab": {"title": title, "private_data": False, "generated_source": True},
        })
        for index, cell in enumerate(notebook.cells):
            cell["id"] = f"{slug[:20]}-{index:02d}"
            if index == 0 and cell.cell_type == "markdown":
                lines = cell.source.splitlines()
                if lines and lines[0].startswith("# "):
                    lines[0] = "# " + title
                    cell.source = "\n".join(lines)
        nbf.write(notebook, output)
        print(output.name)


if __name__ == "__main__":
    main()
