"""生成核心教学 Notebook；仅使用标准库，源代码可复现。"""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip() + "\n"}


BOOT = code(r'''
from pathlib import Path
import sys, os
for folder in [Path.cwd(), *Path.cwd().parents]:
    if (folder / "course_utils.py").exists():
        sys.path.insert(0, str(folder)); break
else:
    raise FileNotFoundError("请从课程项目根目录或 notebooks 目录启动 Jupyter。")
from course_utils import setup, device, save_figure
ROOT = setup(seed=42)
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn
torch.set_num_threads(min(2, os.cpu_count() or 1))
DEVICE = device()
print("计算设备：", DEVICE)
''')


def write(name, cells):
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"{name.split('_')[0]}-cell-{i:03d}"
    (OUT / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


write("06_classical_models.ipynb", [
md(r'''
# 05 传统视觉：HOG + SVM 与 Haar 级联

**学习目标**：理解“像素 → 手工特征 → 分类器”与“滑动窗口检测”的区别。
本课的主实验使用 scikit-learn 自带的真实手写数字图像，不下载数据。另用 OpenCV 样例照片运行 Haar 人脸检测。
HOG + SVM 是经典特征与分类器组合；Haar + 级联用于演示候选窗口筛选，二者不是同一个算法。

输入：8×8 灰度数字。过程：放大、梯度、方向直方图、SVM。输出：数字类别、测试准确率和失败例。
'''), BOOT,
md(r'''
## 1. 查看真实输入与矩阵

`load_digits` 的像素范围是 0–16。这里先划分训练/测试集，避免测试信息进入拟合过程。
8×8 是一个极小的教学数据集；它的准确率不能代表真实街景 OCR 性能。
'''),
code(r'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
digits = load_digits()
train_idx, test_idx = train_test_split(np.arange(len(digits.images)), test_size=.25,
                                     random_state=42, stratify=digits.target)
images = (digits.images / 16 * 255).astype(np.uint8)
print("第一张输入的 8×8 像素矩阵：\n", images[0])
fig, axs = plt.subplots(1, 5, figsize=(12, 3))
for ax, i in zip(axs, train_idx[:5]):
    ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"label={digits.target[i]}"); ax.axis("off")
save_figure("05_digit_inputs.png")
'''),
md(r'''
## 2. 从梯度得到 HOG 特征

HOG 按局部区域统计梯度方向，并在更大的块内归一化。它保留轮廓信息，弱化均匀区域的绝对亮度。
为了便于观察，把 8×8 图像插值为 24×24；插值不会凭空增加真实细节。
窗口 24×24、块 8×8、步长 4×4、单元 4×4、9 个方向箱，得到固定长度向量。
'''),
code(r'''
hog = cv2.HOGDescriptor((24, 24), (8, 8), (4, 4), (4, 4), 9)
resized = np.stack([cv2.resize(im, (24, 24), interpolation=cv2.INTER_CUBIC) for im in images])
features = np.stack([hog.compute(im).ravel() for im in resized])
gx = cv2.Sobel(resized[0], cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(resized[0], cv2.CV_32F, 0, 1, ksize=3)
magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
print("图像张量：", resized.shape, "HOG 特征：", features.shape)
print("特征前 16 项：", np.round(features[0, :16], 3))
print("局部水平梯度：\n", np.round(gx[8:12, 8:12], 1))
fig, axs = plt.subplots(1, 3, figsize=(11, 3))
for ax, im, title in zip(axs, [resized[0], magnitude, angle % 180], ["Input", "Gradient magnitude", "Orientation [0,180)"]):
    ax.imshow(im, cmap="gray" if title == "Input" else "viridis"); ax.set_title(title); ax.axis("off")
save_figure("05_hog_process.png")
'''),
md(r'''
## 3. 拟合分类器与独立测试

标准化器和 SVM 都只拟合训练集；测试集只用于评价。`LinearSVC` 的输出是决策分数，不能直接当作概率。
'''),
code(r'''
classifier = make_pipeline(StandardScaler(), LinearSVC(C=.2, max_iter=5000, random_state=42))
classifier.fit(features[train_idx], digits.target[train_idx])
prediction = classifier.predict(features[test_idx])
test_accuracy = accuracy_score(digits.target[test_idx], prediction)
print(f"HOG + SVM 测试准确率：{test_accuracy:.3f}")
ConfusionMatrixDisplay.from_predictions(digits.target[test_idx], prediction, cmap="Blues")
save_figure("05_hog_confusion.png")
assert test_accuracy > .85, "准确率异常低：请检查输入、标签和特征流程。"
'''),
md(r'''
## 4. 查看错误，而不只看一个数字

下面优先展示误分类输入。它们可能字形模糊或与其他类别相似。
分类器为整张裁剪图输出一个类别；若要在大图中检测，需要增加候选区域/滑动窗口和去重。
'''),
code(r'''
wrong = np.flatnonzero(prediction != digits.target[test_idx])
selected = wrong[:6] if len(wrong) else np.arange(6)
fig, axs = plt.subplots(1, len(selected), figsize=(12, 3), squeeze=False)
for ax, k in zip(axs[0], selected):
    ax.imshow(images[test_idx[k]], cmap="gray")
    ax.set_title(f"true={digits.target[test_idx[k]]}, pred={prediction[k]}"); ax.axis("off")
save_figure("05_hog_errors.png")
'''),
md(r'''
## 5. Haar 级联：在照片中搜索窗口

Viola–Jones 方法把 Haar 矩形特征、积分图和级联分类器结合起来。OpenCV 自带的 XML 是已训练模型。
下面用真实篮球照片演示；远处、遮挡或非正面人脸可能漏检。画出的框必须人工复核，不应把“没有检出”理解为“没有人”。
'''),
code(r'''
photo_path = ROOT / "assets/photos/basketball1.png"
bgr = cv2.imread(str(photo_path))
if bgr is None:
    raise FileNotFoundError(f"缺少课程随附照片：{photo_path}")
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if cascade.empty():
    raise RuntimeError("OpenCV 安装中缺少 Haar 模型，请重新安装 requirements.txt。")
faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(12, 12))
result = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
for x, y, w, h in faces:
    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 80, 0), 2)
print("检测框数量：", len(faces), "；这是模型预测，不是真值标注。")
print("输入灰度局部矩阵：\n", gray[:4, :6])
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); axs[0].set_title("Input photograph")
axs[1].imshow(result); axs[1].set_title(f"Haar output: {len(faces)} candidates")
for ax in axs: ax.axis("off")
save_figure("05_haar_photo.png")
'''),
md(r'''
## 6. 方法边界、练习与来源

| 方法 | 学习的内容 | 优点 | 主要局限 |
|---|---|---|---|
| Haar 级联 | 从矩形特征构建级联分类器 | CPU 快、模型小 | 正面/姿态/尺度变化敏感 |
| HOG + SVM | 梯度描述子 + 判别边界 | 特征可解释 | 大幅形变、复杂背景较困难 |
| DPM | 多部件与空间形变 | 建模物体部件布局 | 训练/推理复杂；本课不把简单 HOG 当作 DPM |

练习：改变 HOG 方向箱数量，记录特征长度与独立测试准确率；降低图片亮度后重新评价；改变 Haar 的 `minNeighbors`，观察误检与漏检。

来源：[OpenCV HOGDescriptor](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html)、[OpenCV 级联检测教程](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)、[scikit-learn digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)。数据与照片许可见项目 `THIRD_PARTY.md`。
''')])

write("07_classification.ipynb", [
md(r'''
# 06 图像分类：真实手写数字与小型 CNN

输入灰度图像 → CNN 特征 → 十类 logits → Softmax 概率。你会看到像素矩阵、卷积特征图、训练曲线、混淆矩阵与失败例。
本课离线使用真实手写数字数据；从零训练，不下载权重。CPU 默认几秒到几十秒，GPU 自动兼容。
'''), BOOT,
md(r'''
## 1. 先划分训练、验证与测试

训练集负责更新参数，验证集负责选择模型，测试集只在训练结束后评价一次。
按类别分层划分，避免某个集合缺少数字；这里没有随机增强，方便初学者跟踪输入变化。
'''),
code(r'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import TensorDataset, DataLoader
data = load_digits()
X = torch.tensor(data.images[:, None] / 16., dtype=torch.float32)
y = torch.tensor(data.target, dtype=torch.long)
train_ids, rest_ids = train_test_split(np.arange(len(y)), test_size=.3, stratify=y.numpy(), random_state=42)
val_ids, test_ids = train_test_split(rest_ids, test_size=.5, stratify=y[rest_ids].numpy(), random_state=42)
print("train / val / test：", len(train_ids), len(val_ids), len(test_ids))
print("输入形状 N,C,H,W：", tuple(X.shape))
print("输入 8×8 浮点矩阵：\n", np.round(X[0,0].numpy(), 2))
fig, axs = plt.subplots(1, 6, figsize=(12, 3))
for ax, i in zip(axs, train_ids[:6]):
    ax.imshow(X[i,0], cmap="gray", vmin=0, vmax=1); ax.set_title(str(y[i].item())); ax.axis("off")
save_figure("07_classification_inputs.png")
'''),
md(r'''
## 2. 看清每一层的输入输出

第一层卷积：`[N,1,8,8] → [N,12,8,8]`；池化得到 4×4；第二层得到 24 个通道。
最后的线性层输出 10 个 logits。训练的 `CrossEntropyLoss` 内部包含 log-softmax，不要先做 Softmax 再传入它。
'''),
code(r'''
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1,12,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(12,24,3,padding=1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(24*4*4,48), nn.ReLU(), nn.Linear(48,10))
    def forward(self, x):
        return self.classifier(self.features(x))
model = DigitCNN().to(DEVICE)
example = X[:1].to(DEVICE)
with torch.no_grad():
    print("feature map：", tuple(model.features(example).shape))
    print("logits：", tuple(model(example).shape))
train_loader = DataLoader(TensorDataset(X[train_ids],y[train_ids]), batch_size=64, shuffle=True, num_workers=0)
optimizer = torch.optim.Adam(model.parameters(), lr=.003)
criterion = nn.CrossEntropyLoss()
'''),
md(r'''
## 3. 训练，并按验证准确率保留模型

`model.train()` 用于训练；`eval()` 和 `no_grad()` 用于评价。数据和模型都移动到同一个设备。
测试集在这个单元中完全不使用。设置 `CV_DEVICE=cpu` 可强制 CPU。
'''),
code(r'''
import copy
history, best_val, best_weights = [], -1., None
for epoch in range(18):
    model.train(); total_loss = 0.
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(); loss = criterion(model(xb),yb)
        loss.backward(); optimizer.step()
        total_loss += loss.item()*len(xb)
    model.eval()
    with torch.no_grad():
        val_accuracy = (model(X[val_ids].to(DEVICE)).argmax(1).cpu() == y[val_ids]).float().mean().item()
    history.append((total_loss/len(train_ids),val_accuracy))
    if val_accuracy > best_val:
        best_val = val_accuracy; best_weights = copy.deepcopy(model.state_dict())
model.load_state_dict(best_weights)
print(f"最佳验证准确率：{best_val:.3f}")
fig, axs = plt.subplots(1,2,figsize=(10,3))
axs[0].plot(np.array(history)[:,0]); axs[0].set_title("Train cross entropy"); axs[0].set_xlabel("Epoch")
axs[1].plot(np.array(history)[:,1]); axs[1].set_title("Validation accuracy"); axs[1].set_ylim(0,1)
save_figure("07_classification_training.png")
'''),
md(r'''
## 4. 单张图片：从像素到类别概率

本例是互斥的单标签十分类，因此 Softmax 后每行概率之和为 1。
最高概率并不等于模型一定正确；模型在分布外数据上仍可能错误且自信。
'''),
code(r'''
model.eval()
with torch.no_grad():
    logits = model(X[test_ids].to(DEVICE))
    probabilities = logits.softmax(1).cpu().numpy()
    pred = probabilities.argmax(1)
    maps = model.features(X[test_ids[:1]].to(DEVICE)).cpu().numpy()[0]
print("logits：", np.round(logits[0].cpu().numpy(),3))
print("概率：", np.round(probabilities[0],3), "；sum=", probabilities[0].sum())
print("第一张特征图左上角：\n", np.round(maps[0,:3,:3],3))
fig, axs = plt.subplots(1,4,figsize=(12,3))
axs[0].imshow(X[test_ids[0],0],cmap="gray"); axs[0].set_title(f"Input, true={y[test_ids[0]].item()}")
for i, ax in enumerate(axs[1:]): ax.imshow(maps[i],cmap="viridis"); ax.set_title(f"Feature channel {i}")
for ax in axs: ax.axis("off")
save_figure("07_classification_features.png")
'''),
md(r'''
## 5. 独立测试与错误分析

混淆矩阵的行是真实类别，列是预测类别。离对角线的数字越多，说明那些类别更容易互相混淆。
下方优先展示错例；若没有错例则展示普通测试样本。
'''),
code(r'''
test_accuracy = float(np.mean(pred == y[test_ids].numpy()))
print(f"独立测试准确率：{test_accuracy:.3f}")
print(classification_report(y[test_ids], pred, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y[test_ids],pred,cmap="Blues")
save_figure("07_classification_confusion.png")
wrong = np.flatnonzero(pred != y[test_ids].numpy())
selected = wrong[:6] if len(wrong) else np.arange(6)
fig, axs = plt.subplots(1,len(selected),figsize=(12,3),squeeze=False)
for ax,k in zip(axs[0],selected):
    ax.imshow(X[test_ids[k],0],cmap="gray"); ax.set_title(f"true={y[test_ids[k]].item()}, pred={pred[k]}"); ax.axis("off")
save_figure("07_classification_errors.png")
assert test_accuracy > .90, "学习结果异常：检查随机种子、训练轮数与归一化。"
'''),
md(r'''
## 6. 自己动手与进一步阅读

1. 去掉第二个卷积层，比较运行时间、参数量和测试准确率。
2. 对测试输入加噪声或旋转，观察分布变化带来的误差；不要根据测试结果反复调整参数。
3. 改用自己的十类图片前，需要重新准备标签、划分数据并训练；这个小模型不能识别任意物体。

来源：[PyTorch 基础训练教程](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)、[CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[digits 数据](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)。
''')])

write("08_detection.ipynb", [
md(r'''
# 07 目标检测：真正学习类别与边界框

本课从零训练一个单目标检测器：输入 32×32 RGB 图，输出类别和归一化边界框 `[xmin,ymin,xmax,ymax]`。
训练图像为程序生成的圆或方块，每张恰好一个目标；这让初学者能看懂所有标注和学习过程。
它不是 YOLO，也不能直接用于真实街景，但确实通过损失函数学习分类与定位，而不是把轮廓查找冒充深度检测。
'''), BOOT,
md(r'''
## 1. 生成独立的训练、验证、测试样本

框坐标归一化到 0–1。颜色、尺寸、位置、背景噪声随机变化；标签由生成时的几何参数给定。
三个集合使用不同随机种子，绝不把测试图像用于训练。

下面的数组按 **NCHW** 存放：`N` 是图片张数，`C=3` 对应红、绿、蓝通道，`H=W=32` 是高和宽。
`train_X[0,0,10:14,10:14]` 是第 1 张图红通道的一块 4×4 像素矩阵；数值为 0–1，越大表示该通道越亮。
类别标签形状为 `[N]`，`0` 表示方块、`1` 表示圆。框标签是 `[N,4]`，每行依次是左上角 `xmin,ymin` 和右下角 `xmax,ymax`。
例如 `[0.25,0.25,0.75,0.75] × 32 = [8,8,24,24]`，表示图片中从 `(8,8)` 到 `(24,24)` 的框；横坐标向右、纵坐标向下。
'''),
code(r'''
from torch.utils.data import DataLoader, TensorDataset
def make_detection_data(n,seed):
    rng = np.random.default_rng(seed)
    ims, labels, boxes = [],[],[]
    for _ in range(n):
        im = rng.uniform(.02,.12,(32,32,3)).astype(np.float32)
        size = 2*int(rng.integers(4,8))+1; x = int(rng.integers(2,30-size)); y0 = int(rng.integers(2,30-size))
        label = int(rng.integers(0,2)); color = rng.uniform(.55,1.,3).tolist()
        if label == 0:
            cv2.rectangle(im,(x,y0),(x+size-1,y0+size-1),color,-1)
        else:
            cv2.circle(im,(x+size//2,y0+size//2),size//2,color,-1)
        ims.append(im.transpose(2,0,1)); labels.append(label)
        boxes.append([x/32,y0/32,(x+size)/32,(y0+size)/32])
    return torch.tensor(np.array(ims)),torch.tensor(labels),torch.tensor(boxes,dtype=torch.float32)
train_X,train_y,train_box = make_detection_data(700,10)
val_X,val_y,val_box = make_detection_data(120,20)
test_X,test_y,test_box = make_detection_data(120,30)
names = ["square","circle"]
print("输入张量：",tuple(train_X.shape),"标签：",train_y[:4].tolist())
print("框矩阵：\n",train_box[:4].numpy())
print("第一个 RGB 图的 R 通道局部：\n",np.round(train_X[0,0,10:14,10:14].numpy(),2))
def draw_box(ax, box, color, label):
    x1,y1,x2,y2 = np.asarray(box)*32
    ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec=color,lw=2,label=label))
fig,axs=plt.subplots(1,4,figsize=(12,3))
for i,ax in enumerate(axs):
    ax.imshow(train_X[i].permute(1,2,0)); draw_box(ax,train_box[i],"lime","ground truth")
    ax.set_title(names[train_y[i]]); ax.axis("off")
save_figure("08_detection_inputs.png")
'''),
md(r'''
## 2. 一个共享特征骨干，两个预测分支

卷积提取空间特征。分类头输出 2 个 logits；定位头预测中心与宽高，再转为 `xyxy`。
框回归损失使用 Smooth L1；分类使用交叉熵。这里固定每张一个物体，所以无需候选框匹配或 NMS。

模型的两项输出分别是 `[N,2]` 的类别 **logits**（每张图两个未归一化分数）和 `[N,4]` 的预测框。
对 logits 按类别维做 `argmax(dim=1)` 得到 `[N]` 的预测类别；需要概率时才做 `softmax(dim=1)`，不能把原始分数直接当作概率。
'''),
code(r'''
class SingleObjectDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                      nn.Conv2d(12,20,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),
                                      nn.Linear(20*8*8,64),nn.ReLU())
        self.class_head=nn.Linear(64,2); self.box_head=nn.Linear(64,4)
    def forward(self,x):
        f=self.features(x); raw=self.box_head(f).sigmoid()
        center=.15+.7*raw[:,:2]; wh=.15+.45*raw[:,2:]
        box=torch.cat([center-wh/2,center+wh/2],dim=1)
        return self.class_head(f),box
model=SingleObjectDetector().to(DEVICE)
optimizer=torch.optim.Adam(model.parameters(),lr=.002)
loader=DataLoader(TensorDataset(train_X,train_y,train_box),batch_size=64,shuffle=True,num_workers=0)
def box_iou(a,b):
    lo=torch.maximum(a[:,:2],b[:,:2]); hi=torch.minimum(a[:,2:],b[:,2:])
    intersection=(hi-lo).clamp(min=0).prod(1)
    area_a=(a[:,2:]-a[:,:2]).clamp(min=0).prod(1)
    area_b=(b[:,2:]-b[:,:2]).clamp(min=0).prod(1)
    return intersection/(area_a+area_b-intersection+1e-8)
print("模型输出形状：",[tuple(v.shape) for v in model(train_X[:2].to(DEVICE))])
'''),
md(r'''
## 3. 学习分类与定位

损失为 `分类交叉熵 + 10 × SmoothL1(预测框,真值框)`，权重只适用于本教学例。
按验证集“类别正确且 IoU≥0.5”的比例选择模型；测试集仍未使用。
'''),
code(r'''
import copy
history=[]; best_score=-1.; best_state=None
for epoch in range(24):
    model.train(); loss_sum=0.
    for xb,yb,bb in loader:
        xb,yb,bb=xb.to(DEVICE),yb.to(DEVICE),bb.to(DEVICE)
        cls,box=model(xb)
        loss=nn.functional.cross_entropy(cls,yb)+10*nn.functional.smooth_l1_loss(box,bb)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); loss_sum+=loss.item()*len(xb)
    model.eval()
    with torch.no_grad():
        cls,box=model(val_X.to(DEVICE)); iou=box_iou(box,val_box.to(DEVICE))
        score=((cls.argmax(1)==val_y.to(DEVICE)) & (iou>=.5)).float().mean().item()
    history.append((loss_sum/len(train_X),score))
    if score>best_score: best_score=score; best_state=copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
fig,axs=plt.subplots(1,2,figsize=(10,3))
axs[0].plot(np.array(history)[:,0]); axs[0].set_title("Training loss")
axs[1].plot(np.array(history)[:,1]); axs[1].set_title("Validation: correct class and IoU >= 0.5"); axs[1].set_ylim(0,1)
save_figure("08_detection_training.png")
print("最佳验证成功率：",round(best_score,3))
'''),
md(r'''
## 4. 独立评价与可视化

IoU 是交并比：预测框与真值框交集面积 / 并集面积。下面报告类别准确率、平均 IoU 和定位成功率。
这些是单目标教学指标，**不是 COCO mAP**；完整多目标 mAP 需要按置信度排序、类别、IoU 阈值与一对一匹配计算 PR 曲线。
'''),
code(r'''
model.eval()
with torch.no_grad():
    logits,pred_box=model(test_X.to(DEVICE)); pred_class=logits.argmax(1).cpu()
    pred_box=pred_box.cpu(); ious=box_iou(pred_box,test_box)
accuracy=(pred_class==test_y).float().mean().item()
success=((pred_class==test_y)&(ious>=.5)).float().mean().item()
print(f"类别准确率={accuracy:.3f}，平均 IoU={ious.mean():.3f}，正确类别且 IoU≥0.5={success:.3f}")
print("真值框：",test_box[0].numpy(),"预测框：",np.round(pred_box[0].numpy(),3))
fig,axs=plt.subplots(2,4,figsize=(12,6))
for i,ax in enumerate(axs.flat):
    ax.imshow(test_X[i].permute(1,2,0)); draw_box(ax,test_box[i],"lime","truth"); draw_box(ax,pred_box[i],"red","prediction")
    ax.set_title(f"{names[pred_class[i]]}, IoU={ious[i]:.2f}"); ax.axis("off")
axs[0,0].legend(fontsize=7)
save_figure("08_detection_results.png")
assert ious.mean()>.5 and accuracy>.8,"模型未学会基础任务：检查训练与数据。"
'''),
md(r'''
## 5. 挑出最差结果，认识任务边界

每张只允许一个目标这一假设非常强。真实检测还需要处理背景、多个目标、遮挡、小目标和标签缺失。
以下按 IoU 最差排序，而不是只挑最好看的图。
'''),
code(r'''
worst=torch.argsort(ious)[:4]
fig,axs=plt.subplots(1,4,figsize=(12,3))
for ax,k in zip(axs,worst):
    ax.imshow(test_X[k].permute(1,2,0)); draw_box(ax,test_box[k],"lime","truth"); draw_box(ax,pred_box[k],"red","prediction")
    ax.set_title(f"true={names[test_y[k]]}, IoU={ious[k]:.2f}"); ax.axis("off")
save_figure("08_detection_failures.png")
'''),
md(r'''
## 6. 练习与后续真实数据路线

练习：增加背景纹理；让物体触及边缘；生成含两物体的图，观察本模型为何无法正确表达两个框。
下一步参考 TorchVision Faster R-CNN 教程，把样本改成 `image + {boxes,labels}`，使用自己标注的数据和预训练权重。

来源：[TorchVision 目标检测微调教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)、[Faster R-CNN 原论文](https://arxiv.org/abs/1506.01497)、[COCO 评价实现](https://github.com/cocodataset/cocoapi)。本例为课程原创简化实现，不声称复现上述完整模型。
''')])

write("09_segmentation.ipynb", [
md(r'''
# 08 图像分割：从像素标签到小型 U-Net

分类为整图输出类别，检测为目标输出框，分割为**每个像素**输出类别。本课用圆形/矩形的合成数据训练真正的编码器—解码器网络。
输入 `[N,3,32,32]`，输出 logits `[N,3,32,32]`；标签 `[N,32,32]`，类别为背景、圆形、矩形。
默认 CPU 运行，无需下载。数据简单，适合观察流程，不能当作真实场景性能。
'''), BOOT,
md(r'''
## 1. 输入图与真值掩膜一一对应

掩膜保存类别整数 0/1/2，不是彩色显示图。显示时使用色表，训练时仍使用原始整数。
先绘制圆再绘制矩形；重叠像素以矩形标签为准，这个规则对图像和掩膜完全一致。

图像采用 **NCHW**：`[N,3,32,32]` 依次为图片张数、RGB 通道、高、宽；浮点像素范围是 0–1。
掩膜采用 **NHW**：`[N,32,32]`，没有 RGB 通道；一个位置只保存一个类别编号，其中 `0=背景`、`1=圆形`、`2=矩形`。
因此打印的 8×8 掩膜局部矩阵中，每个整数对应一个像素的真值类别，不能把 `2` 理解成更亮的像素。
'''),
code(r'''
from torch.utils.data import DataLoader,TensorDataset
def make_segmentation_data(n,seed):
    rng=np.random.default_rng(seed); ims=[]; masks=[]
    for _ in range(n):
        im=rng.uniform(.02,.12,(32,32,3)).astype(np.float32); mask=np.zeros((32,32),np.uint8)
        x,y0=rng.integers(7,25,2); radius=int(rng.integers(4,7))
        cv2.circle(im,(int(x),int(y0)),radius,(.85,.18,.15),-1); cv2.circle(mask,(int(x),int(y0)),radius,1,-1)
        x,y0=rng.integers(2,20,2); w,h=rng.integers(6,12,2)
        cv2.rectangle(im,(int(x),int(y0)),(int(x+w),int(y0+h)),(.15,.8,.25),-1)
        cv2.rectangle(mask,(int(x),int(y0)),(int(x+w),int(y0+h)),2,-1)
        im=np.clip(im+rng.normal(0,.035,im.shape),0,1)
        ims.append(im.transpose(2,0,1)); masks.append(mask)
    return torch.tensor(np.array(ims),dtype=torch.float32),torch.tensor(np.array(masks),dtype=torch.long)
train_X,train_y=make_segmentation_data(240,10)
val_X,val_y=make_segmentation_data(60,20)
test_X,test_y=make_segmentation_data(60,30)
print("图像张量：",tuple(train_X.shape),"掩膜：",tuple(train_y.shape))
row,col=torch.nonzero(train_y[0],as_tuple=False)[0].tolist()
print("目标边缘的掩膜局部矩阵：\n",train_y[0,max(0,row-2):row+6,max(0,col-2):col+6].numpy())
fig,axs=plt.subplots(1,2,figsize=(8,3))
axs[0].imshow(train_X[0].permute(1,2,0)); axs[0].set_title("RGB input")
axs[1].imshow(train_y[0],cmap="viridis",vmin=0,vmax=2); axs[1].set_title("Mask: background/circle/rectangle")
for ax in axs: ax.axis("off")
save_figure("09_segmentation_inputs.png")
'''),
md(r'''
## 2. 一个带跳跃连接的小型 U-Net

编码器把分辨率从 32×32 降到 16×16，解码器上采样回原大小，并拼接浅层特征以恢复边界。
这是 U-Net 思想的极小教学版，不是原论文的完整配置。输出每个像素的三类 logits。

输出张量形状为 **NCHW** 的 `[N,3,32,32]`，但这里的 `C=3` 表示三个类别分数，含义与输入的 RGB 通道不同。
在固定像素 `(h,w)` 上，`logits[n,:,h,w]` 是背景/圆/矩形的三个分数；`argmax(dim=1)` 沿类别维取最大值，得到 `[N,32,32]` 的整数预测掩膜。
这个掩膜的形状和类别编号与真值完全一致，才能逐像素计算 Dice 和 IoU。
'''),
code(r'''
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU())
        self.low=nn.Sequential(nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU())
        self.dec=nn.Sequential(nn.Conv2d(24,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,3,1))
    def forward(self,x):
        skip=self.enc(x); low=self.low(skip)
        up=nn.functional.interpolate(low,size=skip.shape[-2:],mode="bilinear",align_corners=False)
        return self.dec(torch.cat([skip,up],dim=1))
model=MiniUNet().to(DEVICE)
optimizer=torch.optim.Adam(model.parameters(),lr=.004)
loader=DataLoader(TensorDataset(train_X,train_y),batch_size=32,shuffle=True,num_workers=0)
print("模型输出：",tuple(model(train_X[:2].to(DEVICE)).shape))
'''),
md(r'''
## 3. 用验证集选择模型

像素级交叉熵直接接受 logits 和整数掩膜。训练准确率可能被大量背景像素抬高，因此选择模型时关注前景 Dice。
对每类 $c$，$Dice_c=2|P_c∩G_c|/(|P_c|+|G_c|)$，$IoU_c=|P_c∩G_c|/|P_c∪G_c|$。
'''),
code(r'''
def segmentation_scores(pred,target):
    dice=[]; iou=[]
    for c in range(3):
        p=pred==c; g=target==c; inter=(p&g).sum().float()
        dice.append((2*inter/(p.sum()+g.sum()).clamp(min=1)).item())
        iou.append((inter/(p|g).sum().clamp(min=1)).item())
    return np.array(dice),np.array(iou)
import copy
history=[]; best=-1.; best_state=None
for epoch in range(18):
    model.train(); loss_sum=0.
    for xb,yb in loader:
        xb,yb=xb.to(DEVICE),yb.to(DEVICE)
        loss=nn.functional.cross_entropy(model(xb),yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); loss_sum+=loss.item()*len(xb)
    model.eval()
    with torch.no_grad():
        val_pred=model(val_X.to(DEVICE)).argmax(1).cpu()
    val_dice,_=segmentation_scores(val_pred,val_y); score=val_dice[1:].mean()
    history.append((loss_sum/len(train_X),score))
    if score>best: best=score; best_state=copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
fig,axs=plt.subplots(1,2,figsize=(10,3))
axs[0].plot(np.array(history)[:,0]); axs[0].set_title("Train pixel cross entropy")
axs[1].plot(np.array(history)[:,1]); axs[1].set_title("Validation foreground Dice"); axs[1].set_ylim(0,1)
save_figure("09_segmentation_training.png")
'''),
md(r'''
## 4. 测试掩膜、叠加图与指标

测试集从未参与拟合或模型选择。图中同一列展示同一输入的真值、预测和叠加，便于查看边界错位。
本例是语义分割：两个相同类别的圆不会自动得到独立 ID；实例分割还要区分个体。
'''),
code(r'''
model.eval()
with torch.no_grad():
    pred=model(test_X.to(DEVICE)).argmax(1).cpu()
dice,iou=segmentation_scores(pred,test_y)
print("各类 Dice [背景,圆,矩形]：",np.round(dice,3))
print("各类 IoU：",np.round(iou,3),"mIoU(含背景)=",round(iou.mean(),3))
print("预测掩膜局部：\n",pred[0,8:16,8:16].numpy())
fig,axs=plt.subplots(3,4,figsize=(12,8))
for i in range(3):
    rgb=test_X[i].permute(1,2,0).numpy()
    axs[i,0].imshow(rgb); axs[i,1].imshow(test_y[i],vmin=0,vmax=2,cmap="viridis")
    axs[i,2].imshow(pred[i],vmin=0,vmax=2,cmap="viridis")
    axs[i,3].imshow(rgb); axs[i,3].imshow(np.ma.masked_where(pred[i]==0,pred[i]),vmin=0,vmax=2,cmap="viridis",alpha=.5)
for ax,title in zip(axs[0],["Input","Ground truth","Prediction","Overlay"]): ax.set_title(title)
for ax in axs.flat: ax.axis("off")
save_figure("09_segmentation_results.png")
assert dice[1:].mean()>.85,"前景分割效果异常，请检查训练和标签。"
'''),
md(r'''
## 5. 检查颜色捷径与失败情况

训练中的颜色与类别关联，网络可能学到颜色捷径。将红绿通道互换，相同几何物体仍应保持原类别；若预测改变，说明模型对颜色依赖很强。
这个实验用来揭示局限，不把合成数据上的高分宣传为真实泛化能力。
'''),
code(r'''
shifted=test_X[:,[1,0,2]].clone()
with torch.no_grad(): shifted_pred=model(shifted.to(DEVICE)).argmax(1).cpu()
shifted_dice,_=segmentation_scores(shifted_pred,test_y)
print("颜色变化前前景 Dice：",round(dice[1:].mean(),3),"变化后：",round(shifted_dice[1:].mean(),3))
fig,axs=plt.subplots(1,3,figsize=(9,3))
axs[0].imshow(shifted[0].permute(1,2,0)); axs[0].set_title("Color-shifted input")
axs[1].imshow(test_y[0],vmin=0,vmax=2); axs[1].set_title("Unchanged ground truth")
axs[2].imshow(shifted_pred[0],vmin=0,vmax=2); axs[2].set_title("Prediction under shift")
for ax in axs: ax.axis("off")
save_figure("09_segmentation_domain_shift.png")
'''),
md(r'''
## 6. 练习与参考

练习：让每类拥有多种颜色，重新训练，比较颜色变化实验；改变掩膜尺寸后使用最近邻插值（不要用双线性插值生成非整数标签）；增加实例 ID，分析语义与实例分割的区别。

来源：[U-Net 原论文](https://arxiv.org/abs/1505.04597)、[PyTorch 像素交叉熵](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[TorchVision 分割模型](https://pytorch.org/vision/stable/models.html#semantic-segmentation)。
''')])

write("17_ct_preprocessing.ipynb", [
md(r'''
# 17 CT 图像预处理与分割入门

学习 DICOM 像素值到 HU、窗宽窗位、掩膜对齐、阈值基线和 Dice 指标。
**默认输入是程序生成的合成 HU 切片和合成肿瘤掩膜，不是病人影像，不训练或验证临床模型。**
这样没有医学数据下载、隐私或 GPU 门槛；另提供读取自己有权使用的 DICOM 文件的入口。
'''), BOOT,
md(r'''
## 1. 合成有明确真值的 CT 切片

设定空气约 −1000 HU、脂肪约 −100 HU、软组织约 40 HU，并构造一个低密度椭圆病灶。
真实 CT 的组织值会随设备、增强期相和病情变化；这些数值只是教学参数，不是诊断阈值。

`hu` 是 `[H,W]=[128,128]` 的二维浮点矩阵：一行对应图像一行、每列对应一个像素，数值表示 HU，而不是 0–255 的屏幕亮度。
例如本例的 `−1000`、`35`、`75` 分别模拟空气、软组织和肝脏区域，加入噪声后会在相应数值附近波动。
同尺寸的 `truth` 是布尔矩阵，`True/1` 表示合成病灶、`False/0` 表示其余位置；它与 HU 矩阵逐像素对齐。
'''),
code(r'''
rng=np.random.default_rng(42)
H=W=128
yy,xx=np.mgrid[:H,:W]
body=((xx-64)/50)**2+((yy-65)/55)**2<=1
liver=((xx-78)/26)**2+((yy-65)/30)**2<=1
tumor=((xx-83)/8)**2+((yy-61)/10)**2<=1
hu=np.full((H,W),-1000.,np.float32)
hu[body]=-100.; hu[((xx-64)/42)**2+((yy-65)/47)**2<=1]=35.
hu[liver]=75.; hu[tumor]=20.
hu+=rng.normal(0,5,hu.shape).astype(np.float32)
truth=tumor.astype(bool)
print("HU 矩阵片段：\n",np.round(hu[57:63,79:85],1))
print("掩膜片段：\n",truth[57:63,79:85].astype(int))
fig,axs=plt.subplots(1,2,figsize=(8,4))
im=axs[0].imshow(hu,cmap="gray",vmin=-1000,vmax=150); axs[0].set_title("Synthetic HU slice"); plt.colorbar(im,ax=axs[0])
axs[1].imshow(truth,cmap="gray"); axs[1].set_title("Synthetic lesion mask")
for ax in axs: ax.axis("off")
save_figure("17_ct_inputs.png")
'''),
md(r'''
## 2. 模拟存储值与 HU 的线性变换

对常见 CT 对象，`HU = StoredValue × RescaleSlope + RescaleIntercept`。实际文件还需检查 Modality、单位和相关元数据。
不能把任意图像数组都宣称为 HU。这里用已知参数把合成 HU 转成存储值，再验证逆变换。
'''),
code(r'''
slope,intercept=1.,-1024.
stored=np.rint((hu-intercept)/slope).astype(np.int16)
recovered=stored.astype(np.float32)*slope+intercept
print("存储值局部：\n",stored[57:61,79:83])
print("恢复 HU 局部：\n",recovered[57:61,79:83])
print("量化最大误差：",np.abs(recovered-hu).max())
assert np.abs(recovered-hu).max()<=.51
'''),
md(r'''
## 3. 窗宽窗位只影响显示映射

教学线性显示：先裁剪到 `[中心−宽度/2,中心+宽度/2]`，再映射到 0–1。
这不是完整的 DICOM VOI LUT 标准实现；读取真实文件时应使用 `apply_voi_lut` 等标准处理并检查窗参数。
显示窗改变不应覆盖原始 HU，也不能让掩膜随显示窗改变。

例如 `center=60,width=150` 时，窗口下限是 `−15 HU`、上限是 `135 HU`：小于等于 −15 显示为 0（黑），大于等于 135 显示为 1（白），60 HU 映射为 0.5（中灰）。
20 HU 会映射为 `(20−(−15))/150≈0.233`。映射后的显示矩阵仍是 `[128,128]`，但其单位已经是 0–1 的显示强度；后续分割仍使用原始 HU 矩阵。
'''),
code(r'''
def window_image(hu_image,center,width):
    if width<=0: raise ValueError("窗宽必须大于 0")
    return np.clip((hu_image-(center-width/2))/width,0,1)
windows=[(50,350),(60,150),(-600,1500)]
fig,axs=plt.subplots(1,3,figsize=(12,4))
for ax,(center,width) in zip(axs,windows):
    ax.imshow(window_image(recovered,center,width),cmap="gray",vmin=0,vmax=1)
    ax.set_title(f"Center={center}, Width={width}"); ax.axis("off")
save_figure("17_ct_windows.png")
'''),
md(r'''
## 4. 最简单的 ROI 阈值分割基线

本例已知合成肝脏区域，先用 ROI 限制，再对低 HU 区域阈值分割。真实医学图像通常没有这样的现成 ROI，不能把本实验分数等同于自动临床肝肿瘤分割。
对真实深度学习分割，需要独立病人划分、像素/体素空间对齐、训练与外部验证；第 08 课演示网络训练流程。
'''),
code(r'''
raw_pred=liver & (recovered<38)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
pred=cv2.morphologyEx(raw_pred.astype(np.uint8),cv2.MORPH_OPEN,kernel).astype(bool)
def overlap_metrics(pred,true):
    intersection=np.logical_and(pred,true).sum()
    dice=2*intersection/max(pred.sum()+true.sum(),1)
    iou=intersection/max(np.logical_or(pred,true).sum(),1)
    return dice,iou
dice,iou=overlap_metrics(pred,truth)
print(f"合成切片 Dice={dice:.3f}, IoU={iou:.3f}")
print("输入/预测/真值的形状：",recovered.shape,pred.shape,truth.shape)
fig,axs=plt.subplots(1,4,figsize=(14,4))
axs[0].imshow(window_image(recovered,60,150),cmap="gray"); axs[0].set_title("Input window")
axs[1].imshow(truth,cmap="gray"); axs[1].set_title("Ground truth")
axs[2].imshow(pred,cmap="gray"); axs[2].set_title("ROI + threshold + opening")
axs[3].imshow(window_image(recovered,60,150),cmap="gray")
axs[3].contour(truth,levels=[.5],colors="lime"); axs[3].contour(pred,levels=[.5],colors="red"); axs[3].set_title("Green truth / red prediction")
for ax in axs: ax.axis("off")
save_figure("17_ct_segmentation.png")
assert dice>.9,"合成基线与真值不一致，请检查坐标和阈值。"
'''),
md(r'''
## 5. 故意制造掩膜错位

相同大小不代表空间对齐。真实 DICOM 还涉及切片顺序、方向、原点、PixelSpacing 与层间距离。
把标签平移几像素，Dice 就会下降；可视化叠加能发现单纯检查 shape 发现不了的问题。
'''),
code(r'''
shifted_truth=np.roll(truth,7,axis=1)
bad_dice,_=overlap_metrics(pred,shifted_truth)
print(f"正确对齐 Dice={dice:.3f}；错位 7 像素 Dice={bad_dice:.3f}")
fig,axs=plt.subplots(1,2,figsize=(8,4))
for ax,mask,title in zip(axs,[truth,shifted_truth],["Correct alignment","Wrong alignment"]):
    ax.imshow(window_image(recovered,60,150),cmap="gray"); ax.contour(mask,levels=[.5],colors="red"); ax.set_title(title); ax.axis("off")
save_figure("17_ct_alignment.png")
'''),
md(r'''
## 6. 可选：读取自己的单张 CT DICOM

默认不执行、不额外安装依赖。只有设置 `CV_DICOM_PATH` 后才读取；你需要有使用该数据的权限。
如需启用，先安装 `pydicom`（用于解析 DICOM，不用于模型训练）。压缩像素可能还需要对应解码插件。
不要把患者姓名、标识、出生日期或完整原始文件提交到 GitHub。
'''),
code(r'''
dicom_path=os.environ.get("CV_DICOM_PATH")
if dicom_path:
    try:
        import pydicom
        from pydicom.pixels import apply_modality_lut
    except ImportError as exc:
        raise ImportError("真实 DICOM 入口需要 pydicom；按 README 的可选依赖说明安装。") from exc
    ds=pydicom.dcmread(dicom_path)
    if getattr(ds,"Modality",None)!="CT": raise ValueError("本入口仅接受 CT，不能假定其他影像单位为 HU。")
    pixels=ds.pixel_array
    if pixels.ndim!=2: raise ValueError("本入口仅示范单帧 2D CT；多帧/体数据需显式处理。")
    real_hu=apply_modality_lut(pixels,ds).astype(np.float32)
    print("读取单帧 CT：",real_hu.shape,"；请核查 RescaleType/单位、方向与 spacing。")
    plt.figure(figsize=(5,5)); plt.imshow(window_image(real_hu,50,350),cmap="gray"); plt.axis("off")
    save_figure("17_optional_ct_window.png")
else:
    print("未设置 CV_DICOM_PATH：继续使用明确标注的合成教学切片。")
'''),
md(r'''
## 7. 练习与参考

练习：改变噪声、病灶 HU 和阈值，记录 Dice；用合成切片训练第 08 课网络；按病人而非切片划分真实数据；分别统计肝脏和肿瘤而非只统计整体准确率。

来源：[pydicom modality LUT](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_modality_lut.html)、[pydicom VOI LUT](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_voi_lut.html)、[nnU-Net 原项目](https://github.com/MIC-DKFZ/nnUNet)。本课不声称复现 nnU-Net 或提供临床建议。
''')])

write("18_underwater.ipynb", [
md(r'''
# 18 水下图像：合成退化、增强与评价边界

用随附真实水果照片模拟色偏、衰减与雾化，再做灰世界白平衡和 CLAHE。输入 → 退化过程 → 增强输出全部可见。
**照片不是真实水下采集，退化模型是教学简化模型。** 只有在这个已知参考的合成实验中，才适合直接对参考图算 PSNR/SSIM 类指标。
真实水下通常没有无退化真值；色彩好看或无参考分数提高，不等于恢复了真实颜色或提升了检测性能。
'''), BOOT,
md(r'''
## 1. 真实原图、RGB 矩阵与颜色范围

OpenCV 读入 BGR，要转换成 RGB 再显示。统一转换为 float32 的 0–1，防止 uint8 乘加溢出。
课程随附 OpenCV `fruits.jpg`；不需要联网下载。
'''),
code(r'''
photo_path=ROOT/"assets/photos/fruits.jpg"
bgr=cv2.imread(str(photo_path))
if bgr is None: raise FileNotFoundError(f"缺少随附照片：{photo_path}")
reference=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
reference=cv2.resize(reference,(320,240),interpolation=cv2.INTER_AREA)
print("输入：",reference.shape,reference.dtype)
print("左上角 RGB 矩阵：\n",np.round(reference[:3,:3],3))
plt.figure(figsize=(6,4)); plt.imshow(reference); plt.title("Reference photograph (not underwater)"); plt.axis("off")
save_figure("18_underwater_reference.png")
'''),
md(r'''
## 2. 可观察的合成退化过程

简化形式 $I_c=J_c t_c + A_c(1-t_c)$，其中 $J$ 是参考、$t$ 是通道透射率、$A$ 是背景光。
红通道衰减更强，且模拟深度从左到右增加。它用于理解现象，不能表示所有真实水域或相机成像。
'''),
code(r'''
h,w=reference.shape[:2]
depth=np.linspace(.4,1.6,w,dtype=np.float32)[None,:,None]
beta=np.array([1.6,.65,.35],np.float32)[None,None,:]
transmission=np.exp(-depth*beta)
ambient=np.array([.02,.30,.36],np.float32)[None,None,:]
degraded=np.clip(reference*transmission+ambient*(1-transmission),0,1)
print("透射率 RGB（左/中/右）：\n",np.round(transmission[0,[0,w//2,-1]],3))
fig,axs=plt.subplots(1,3,figsize=(13,4))
axs[0].imshow(reference); axs[0].set_title("Reference")
axs[1].imshow(np.broadcast_to(transmission,(h,w,3))); axs[1].set_title("Transmission map")
axs[2].imshow(degraded); axs[2].set_title("Synthetic underwater-like degradation")
for ax in axs: ax.axis("off")
save_figure("18_underwater_degradation.png")
'''),
md(r'''
## 3. 灰世界白平衡

假设场景平均颜色应该接近灰色，按各通道均值的倒数调整增益。若场景本来大面积是绿色/红色，这个假设就会失效。
因此增益做了限制，并保留数值供检查，而不是盲目追求更鲜艳。
'''),
code(r'''
means=degraded.mean(axis=(0,1))
gain=np.clip(means.mean()/np.maximum(means,1e-6),.4,3.)
balanced=np.clip(degraded*gain,0,1)
print("退化图 RGB 均值：",np.round(means,3),"增益：",np.round(gain,3))
print("白平衡输出局部：\n",np.round(balanced[40:43,40:43,0],3))
fig,axs=plt.subplots(1,2,figsize=(10,4))
axs[0].imshow(degraded); axs[0].set_title("Degraded")
axs[1].imshow(balanced); axs[1].set_title("Gray-world white balance")
for ax in axs: ax.axis("off")
save_figure("18_underwater_white_balance.png")
'''),
md(r'''
## 4. 在 Lab 亮度上做 CLAHE

局部直方图均衡化改变明暗对比度，不直接逆转物理衰减。只处理 L 通道以减少额外的颜色失真。
过大的 clipLimit 可能放大噪声和边缘光晕。
'''),
code(r'''
balanced_u8=np.uint8(np.round(balanced*255))
lab=cv2.cvtColor(balanced_u8,cv2.COLOR_RGB2LAB)
clahe=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8))
lab_out=lab.copy(); lab_out[:,:,0]=clahe.apply(lab[:,:,0])
enhanced=cv2.cvtColor(lab_out,cv2.COLOR_LAB2RGB).astype(np.float32)/255
print("输入 L 局部：\n",lab[40:44,40:44,0])
print("输出 L 局部：\n",lab_out[40:44,40:44,0])
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,im,title in zip(axs,[reference,degraded,enhanced],["Reference","Degraded","White balance + CLAHE"]):
    ax.imshow(im); ax.set_title(title); ax.axis("off")
save_figure("18_underwater_results.png")
'''),
md(r'''
## 5. 与已知参考比较，同时报告不利结果

PSNR 使用相同 0–1 范围：$10\log_{10}(1/MSE)$。数值越高意味着本实验中逐像素误差越小，不衡量语义正确性。
增强后的 PSNR 可能下降，这不是程序失败，而是对比增强和参考重建目标不同。
下面也报告通道均值和亮度标准差；它们仅是描述统计，不能代替 UCIQE/UIQM 或下游目标检测评价。
'''),
code(r'''
def psnr(a,b):
    mse=float(np.mean((a-b)**2))
    return float("inf") if mse==0 else 10*np.log10(1/mse)
def luminance_std(im):
    return float((im@np.array([.299,.587,.114])).std())
for label,im in [("degraded",degraded),("white_balance",balanced),("enhanced",enhanced)]:
    print(f"{label:15s} PSNR={psnr(reference,im):.2f} dB, luminance std={luminance_std(im):.3f}, RGB mean={im.mean((0,1)).round(3)}")
fig,axs=plt.subplots(1,2,figsize=(10,3))
for c,color in enumerate(["r","g","b"]):
    axs[0].hist(degraded[:,:,c].ravel(),bins=40,alpha=.4,color=color)
    axs[1].hist(enhanced[:,:,c].ravel(),bins=40,alpha=.4,color=color)
axs[0].set_title("Degraded channel histograms"); axs[1].set_title("Enhanced channel histograms")
save_figure("18_underwater_histograms.png")
assert enhanced.shape==reference.shape and np.isfinite(enhanced).all()
'''),
md(r'''
## 6. 物理模型的“已知参数上界”与实际边界

在本合成实验中，生成时的 $t,A$ 已知，可以代数求逆来检验实现。真实图像的参数通常未知，估计误差和噪声会被放大。
此步骤是验证合成过程，不是可直接部署的真实水下恢复算法。
'''),
code(r'''
oracle=np.clip((degraded-ambient*(1-transmission))/transmission,0,1)
print("已知合成参数的逆变换最大误差：",float(np.abs(oracle-reference).max()))
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,im,title in zip(axs,[reference,enhanced,oracle],["Reference","Heuristic enhancement","Inverse with known synthetic parameters"]):
    ax.imshow(im); ax.set_title(title); ax.axis("off")
save_figure("18_underwater_oracle.png")
assert np.allclose(oracle,reference,atol=1e-5)
'''),
md(r'''
## 7. 练习与真实水下研究路线

练习：增大红通道衰减、加入噪声，比较增强效果；使用全绿色场景观察灰世界假设失败；如果有检测标注，分别对增强前后计算同一模型的 mAP，验证增强是否真正帮助任务。
后续可比较 UIEB 等有参考数据集与真实无参考数据；记录数据许可与划分。cGAN 是**条件生成对抗网络**，不应称为强化学习。

来源：[OpenCV CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)、[UIEB 官方项目](https://li-chongyi.github.io/proj_benchmark.html)、[Conditional GAN 原论文](https://arxiv.org/abs/1411.1784)。本课为可解释的传统基线，不声称复现水下深度恢复模型。
''')])

print("已生成 6 份核心 Notebook。")
