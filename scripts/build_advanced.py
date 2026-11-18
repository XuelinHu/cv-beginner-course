"""生成中文高级视觉入门 Notebook；教学代码保留在单元格中。"""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def md(s):
    s = s.replace(chr(92)*2, chr(92)).replace("$alpha_t", "$"+chr(92)+"alpha_t")
    s = s.replace(chr(8)+"eta", chr(92)+"beta").replace(chr(8)+"ar", chr(92)+"bar")
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip() + "\n"}


def code(s):
    return {"cell_type": "code", "metadata": {}, "source": textwrap.dedent(s).strip() + "\n", "outputs": [], "execution_count": None}


SETUP = r'''
from pathlib import Path
import sys, os
for folder in [Path.cwd(), *Path.cwd().parents]:
    if (folder / "course_utils.py").exists():
        sys.path.insert(0, str(folder))
        break
else:
    raise FileNotFoundError("请从完整项目的根目录或 notebooks 目录运行。")
from course_utils import setup, device, save_figure
ROOT = setup(seed=42)
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic("matplotlib", "inline")
import torch
from torch import nn
import torch.nn.functional as F
dev = device()
torch.set_num_threads(2)
print("计算设备:", dev)
'''


def write(name, cells):
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:03d}"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    (OUT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


write("10_retrieval.ipynb", [
md(r'''# 09 图像检索：图片怎样变成向量，再找到相似图片

学习目标：观察 RGB/HSV 数值，亲手提取向量、计算余弦相似度、返回 Top-K 并评价。
本课使用可复现的几何图像图库；另提供项目内真实照片的查询实验。不下载权重。
这是**颜色与空间外观特征基线**，不具备 CLIP 等模型的语义理解能力。'''),
code(SETUP),
md('''## 1. 准备输入：图库与独立查询图片

圆、方形、三角形各用红绿蓝生成多张图，图库与查询使用不同随机种子，标签仅用于评估。'''),
code(r'''
def make_picture(shape, color, seed):
    rng = np.random.default_rng(seed)
    image = np.full((64, 64, 3), 235, np.uint8)
    center = (int(rng.integers(27, 37)), int(rng.integers(27, 37)))
    size = int(rng.integers(15, 22))
    if shape == 0:
        cv2.circle(image, center, size, color, -1)
    elif shape == 1:
        cv2.rectangle(image, (center[0]-size, center[1]-size), (center[0]+size, center[1]+size), color, -1)
    else:
        pts = np.array([[center[0], center[1]-size], [center[0]-size, center[1]+size], [center[0]+size, center[1]+size]])
        cv2.fillPoly(image, [pts], color)
    return image  # 此处颜色直接按 RGB 解释

colors = [(220, 40, 40), (40, 180, 40), (40, 70, 220)]
gallery, labels = [], []
for shape in range(3):
    for color_id, color in enumerate(colors):
        for sample in range(4):
            gallery.append(make_picture(shape, color, 100*shape+10*color_id+sample))
            labels.append((shape, color_id))
query = make_picture(0, colors[0], 999)
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
for ax, im, title in zip(axs, [query, gallery[0], gallery[12], gallery[24]], ["独立查询", "图库：圆", "图库：方形", "图库：三角"]):
    ax.imshow(im); ax.set_title(title); ax.axis("off")
save_figure("10_retrieval_inputs")
print("输入形状:", query.shape, "类型:", query.dtype)
print("查询中心 3×3 RGB 矩阵:\n", query[30:33, 30:33])
'''),
md(r'''## 2. 编码：从像素到一个固定长度向量

先用饱和度分出彩色前景，再计算前景颜色直方图和紧致边界框内的二值轮廓图，拼接后做 L2 归一化。
颜色特征负责“颜色像不像”，轮廓特征保留“形状像不像”。前景提取依赖本课灰色背景的假设，不能直接用于任意真实场景。

数值怎么读：RGB 输入是 `[64,64,3]`，最后一维依次是红、绿、蓝；HSV 直方图有 `12×4=48` 项，轮廓缩略图有 `12×12=144` 项，拼成 192 维向量。36 张图库组成 `[36,192]` 矩阵，乘以 `[192]` 查询向量得到 36 个相似度。归一化后的特征值已不是原始像素值，不能按颜色强度解释。'''),
code(r'''
def encode(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] > 60).astype(np.uint8)
    if not mask.any():
        mask = np.ones(image.shape[:2], np.uint8)
    hist = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256]).reshape(-1)
    hist = hist / (np.linalg.norm(hist)+1e-8)
    x, y, w, h = cv2.boundingRect(mask)
    spatial = cv2.resize(mask[y:y+h, x:x+w].astype(np.float32), (12, 12), interpolation=cv2.INTER_AREA).reshape(-1)
    spatial /= np.linalg.norm(spatial)+1e-8
    vector = np.concatenate([hist, spatial])
    return vector / (np.linalg.norm(vector)+1e-8)

vectors = np.stack([encode(im) for im in gallery])
q = encode(query)
print("图库矩阵 [图像数, 特征维数]:", vectors.shape)
print("查询向量前 12 项:", np.round(q[:12], 3))
print("L2 范数:", np.linalg.norm(q))
'''),
md(r'''## 3. 检索：余弦相似度与排序

归一化后，$s(q,v)=q^Tv$。分数表示特征相似程度，不是“属于该类别的概率”。'''),
code(r'''
scores = vectors @ q
top = np.argsort(-scores)[:5]
print("Top-5 索引:", top, "分数:", np.round(scores[top], 3))
fig, axs = plt.subplots(1, 6, figsize=(14, 3))
axs[0].imshow(query); axs[0].set_title("查询"); axs[0].axis("off")
for ax, idx in zip(axs[1:], top):
    ax.imshow(gallery[idx]); ax.set_title(f"ID {idx}\n相似度 {scores[idx]:.3f}"); ax.axis("off")
save_figure("10_retrieval_results")
'''),
md('''## 4. 评价：在九个独立查询上计算 Precision@K 与 Recall@K

相关图片定义为“形状和颜色都相同”；图库每组有 4 张。
Precision@K 衡量返回结果中相关结果的比例；Recall@K 衡量图库相关图片被找回的比例。'''),
code(r'''
precision, recall = [], []
for shape in range(3):
    for c in range(3):
        test_query = make_picture(shape, colors[c], 2000+10*shape+c)
        rank = np.argsort(-(vectors @ encode(test_query)))[:4]
        relevant = sum(labels[i] == (shape, c) for i in rank)
        precision.append(relevant/4); recall.append(relevant/4)
print(f"独立查询 Precision@4={np.mean(precision):.3f}, Recall@4={np.mean(recall):.3f}")
assert vectors.shape[0] == len(labels) and np.isfinite(scores).all()
'''),
md('''## 5. 真实照片实验：相似变换能否被检索回来

同一张照片的亮度、缩放和旋转版本与三种形状一起进入图库。观察外观特征对旋转的敏感性。
这不是严格的语义检索评测；要评价语义检索应使用独立标注的真实图库。'''),
code(r'''
photo_path = ROOT / "assets/photos/fruits.jpg"
if photo_path.exists():
    photo = cv2.cvtColor(cv2.imread(str(photo_path)), cv2.COLOR_BGR2RGB)
    photo = cv2.resize(photo, (64, 64))
    candidates = [photo, np.clip(photo.astype(float)*0.65, 0, 255).astype(np.uint8), np.rot90(photo).copy(), *gallery[::12]]
    photo_scores = np.array([encode(im) @ encode(photo) for im in candidates])
    fig, axs = plt.subplots(1, len(candidates), figsize=(14, 3))
    for i, ax in enumerate(axs):
        ax.imshow(candidates[i]); ax.set_title(f"{i}: {photo_scores[i]:.3f}"); ax.axis("off")
    save_figure("10_retrieval_photo")
else:
    print("未找到可选真实照片；请运行项目的素材准备步骤。")
'''),
md('''## 6. 学生练习与扩展

1. 修改特征权重、灰度缩略图大小，观察 Recall@4。
2. 把查询图变成灰度，解释为什么颜色特征失效。
3. 进阶：用 torchvision 预训练编码器替换 encode，保持检索与评价代码不变；下载权重前先确认网络和授权。

参考：[OpenCV 直方图计算](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html)、[FAISS 项目](https://github.com/facebookresearch/faiss)。本课的小图库使用精确矩阵乘法，不依赖 FAISS。''')])


write("10_generation.ipynb", [
md(r'''# 10 图像生成：在 CPU 上训练一个小型 DDPM

从 sklearn 自带 8×8 手写数字开始，观察**干净图 → 加噪矩阵 → 噪声预测 → 反向生成**。
这是真实的噪声预测与反向采样，采用很小的 CNN、短训练和低分辨率以适合 CPU；输出质量不能代表成熟图像生成模型。'''),
code(SETUP),
md('''## 1. 输入数据：归一化与张量布局

sklearn 的 digits 包含 8×8 灰度手写数字，像素原范围是 0～16。选 0、1、2 三类降低任务难度；不读取外部模型。训练和保留集分开。'''),
code(r'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
images = digits.images[digits.target < 3].astype(np.float32)/8.0-1.0
train_images, heldout_images = train_test_split(images, test_size=0.2, random_state=42)
train_x = torch.from_numpy(train_images[:, None]).to(dev)
heldout_x = torch.from_numpy(heldout_images[:, None]).to(dev)
print("训练张量 [N,C,H,W]:", tuple(train_x.shape))
print("一张输入的前 4×4 矩阵:\n", train_images[0, :4, :4])
fig, axs = plt.subplots(1, 6, figsize=(10, 2))
for ax, im in zip(axs, train_images[:6]):
    ax.imshow(im, cmap="gray", vmin=-1, vmax=1); ax.axis("off")
save_figure("10_generation_inputs")
'''),
md(r'''## 2. 前向加噪

定义 $eta_t$，$alpha_t=1-eta_t$，$ar\alpha_t=\prod_{s=1}^{t}\alpha_s$。
可直接采样：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，其中 $\epsilon\sim\mathcal N(0,I)$。
这里用 40 步和较大的最终噪声率，让末端足够接近高斯噪声；它是教学配置。'''),
code(r'''
T = 40
betas = torch.linspace(0.001, 0.25, T, device=dev)
alphas = 1-betas
alpha_bar = torch.cumprod(alphas, 0)
def add_noise(clean, t, noise):
    a = alpha_bar[t][:, None, None, None]
    return a.sqrt()*clean+(1-a).sqrt()*noise

clean = train_x[:1].repeat(4, 1, 1, 1)
ts = torch.tensor([0, 9, 24, 39], device=dev)
noise = torch.randn_like(clean)
noisy = add_noise(clean, ts, noise)
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
for ax, im, t in zip(axs, noisy.detach().cpu().numpy()[:, 0], ts.cpu().tolist()):
    ax.imshow(im, cmap="gray", vmin=-2, vmax=2); ax.set_title(f"t={t}"); ax.axis("off")
save_figure("10_generation_forward")
print("末端 alpha_bar:", alpha_bar[-1].item())
print("加噪后的 4×4:\n", np.round(noisy[2, 0, :4, :4].cpu().numpy(), 2))
'''),
md(r'''## 3. 噪声预测网络

网络输入是 $x_t$ 和时间 $t$，输出与图像同形状的预测噪声。时间编码拼成一个额外通道。

形状怎么读：原图与预测噪声均为 `[N,1,8,8]`，表示 N 张单通道 8×8 图；时间通道与噪声图拼接后，网络实际接收 `[N,2,8,8]`。显示的 4×4 矩阵是某张图左上角的局部，不能当成整张 8×8 图。原始像素 0～16 经 `像素/8-1` 映射到 [-1,1]；加入高斯噪声后允许超出该范围。
损失 $\mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2$，目标是随机加入的**真实噪声**。'''),
code(r'''
class TinyNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 1, 3, padding=1))
    def forward(self, x, t):
        time_channel = (t.float()/(T-1))[:, None, None, None].expand(-1, 1, 8, 8)
        return self.net(torch.cat([x, time_channel], dim=1))

model = TinyNoisePredictor().to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
steps = int(os.environ.get("CV_DIFFUSION_STEPS", "400"))
losses = []
model.train()
for step in range(steps):
    ids = torch.randint(len(train_x), (64,), device=dev)
    x0 = train_x[ids]
    t = torch.randint(T, (len(x0),), device=dev)
    epsilon = torch.randn_like(x0)
    xt = add_noise(x0, t, epsilon)
    loss = F.mse_loss(model(xt, t), epsilon)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
print("初期/末期平均训练损失:", np.mean(losses[:30]), np.mean(losses[-30:]))
plt.figure(figsize=(7, 3)); plt.plot(losses); plt.xlabel("训练步"); plt.ylabel("噪声预测 MSE")
save_figure("10_generation_loss")
'''),
md(r'''## 4. 保留集检查：网络是否学到去噪

与“始终预测零噪声”比较同一批保留图的噪声 MSE。它衡量去噪能力，不等于 FID 或人类视觉质量。'''),
code(r'''
model.eval()
with torch.no_grad():
    val_t = torch.full((len(heldout_x),), T//2, device=dev, dtype=torch.long)
    val_e = torch.randn_like(heldout_x)
    val_xt = add_noise(heldout_x, val_t, val_e)
    pred_e = model(val_xt, val_t)
    val_mse = F.mse_loss(pred_e, val_e).item()
    zero_mse = val_e.square().mean().item()
print(f"保留集噪声 MSE={val_mse:.4f}; 零预测基线={zero_mse:.4f}")
assert np.isfinite(val_mse)
'''),
md(r'''## 5. 从纯噪声开始反向采样

DDPM 反向均值：$\mu_\theta=\alpha_t^{-1/2}(x_t-\beta_t\epsilon_\theta/\sqrt{1-\bar\alpha_t})$。
采用后验方差 $\tilde\beta_t=\beta_t(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)$，最后一步不再加噪。
为了小模型稳定，先估计并截断 $\hat x_0$ 到数据范围，再计算等价后验均值。'''),
code(r'''
snapshots = {}
with torch.no_grad():
    x = torch.randn(12, 1, 8, 8, device=dev)
    snapshots[T] = x[0, 0].cpu().numpy().copy()
    for t in reversed(range(T)):
        tt = torch.full((len(x),), t, device=dev, dtype=torch.long)
        eps = model(x, tt)
        a_bar = alpha_bar[t]
        prev_bar = alpha_bar[t-1] if t > 0 else torch.tensor(1., device=dev)
        x0_est = ((x-(1-a_bar).sqrt()*eps)/a_bar.sqrt()).clamp(-1, 1)
        mean = betas[t]*prev_bar.sqrt()/(1-a_bar)*x0_est + (1-prev_bar)*alphas[t].sqrt()/(1-a_bar)*x
        variance = betas[t]*(1-prev_bar)/(1-a_bar)
        x = mean + variance.sqrt()*torch.randn_like(x) if t > 0 else mean
        if t in [30, 20, 10, 0]:
            snapshots[t] = x[0, 0].cpu().numpy().copy()
generated = x.cpu().numpy()[:, 0]
fig, axs = plt.subplots(1, 5, figsize=(12, 3))
for ax, (t, im) in zip(axs, snapshots.items()):
    ax.imshow(im, cmap="gray", vmin=-1, vmax=1); ax.set_title(f"反向 t={t}"); ax.axis("off")
save_figure("10_generation_process")
fig, axs = plt.subplots(2, 6, figsize=(10, 4))
for ax, im in zip(axs.flat, generated):
    ax.imshow(im, cmap="gray", vmin=-1, vmax=1); ax.axis("off")
save_figure("10_generation_outputs")
print("生成输出 4×4:\n", np.round(generated[0, :4, :4], 2))
'''),
md('''## 6. 理解局限与练习

本课没有文本条件，也没有潜空间编码器，不能直接称为 Stable Diffusion。短训练可能生成模糊或难辨认的数字；请把实际结果与训练数据并排看，不能把训练图当生成图。

练习：把 CV_DIFFUSION_STEPS 改为 800；比较固定随机种子下的生成质量和耗时。扩展到全部十类后，为什么训练更难？

主要来源：[DDPM 原论文](https://arxiv.org/abs/2006.11239)、[作者代码](https://github.com/hojonathanho/diffusion)、[sklearn digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)。本课为独立教学简化实现。''')])


write("11_captioning.ipynb", [
md('''# 11 图像描述：CNN 编码图片，GRU 逐词生成句子

用受控的“颜色 × 形状”图片学习真正的编码器—解码器训练：RGB 输入 → 视觉向量 → 词元矩阵 → 句子。
默认不下载大模型。这是小词表、短句子的教学任务；不是能理解任意自然照片的通用视觉语言模型。'''),
code(SETUP),
md('''## 1. 数据：图像与完整句子成对

标签句子由数据生成器定义为 “a red circle”等。训练时只输入图像与前缀词，预测下一个词；测试生成时不提供答案。位置与大小随机，训练/测试种子分离。'''),
code(r'''
words = ["<pad>", "<bos>", "<eos>", "a", "red", "green", "blue", "circle", "square", "triangle"]
stoi = {word: i for i, word in enumerate(words)}
palette = [(230, 35, 35), (35, 200, 35), (35, 70, 230)]
color_names = ["red", "green", "blue"]
shape_names = ["circle", "square", "triangle"]
def caption_data(n, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        c, s = int(rng.integers(3)), int(rng.integers(3))
        im = np.full((32, 32, 3), 245, np.uint8)
        x, y = rng.integers(13, 20, size=2); r = int(rng.integers(7, 11))
        if s == 0: cv2.circle(im, (x, y), r, palette[c], -1)
        elif s == 1: cv2.rectangle(im, (x-r, y-r), (x+r, y+r), palette[c], -1)
        else: cv2.fillPoly(im, [np.array([[x, y-r], [x-r, y+r], [x+r, y+r]])], palette[c])
        xs.append(im.transpose(2, 0, 1)/255.)
        ys.append([stoi["<bos>"], stoi["a"], stoi[color_names[c]], stoi[shape_names[s]], stoi["<eos>"]])
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(ys, dtype=torch.long)

train_x, train_y = caption_data(360, 10)
test_x, test_y = caption_data(90, 999)
print("图像 NCHW:", tuple(train_x.shape), "句子 token 矩阵:\n", train_y[:5])
print("RGB 局部矩阵:\n", train_x[0, :, 14:17, 14:17].permute(1, 2, 0).numpy())
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axs):
    ax.imshow(train_x[i].permute(1, 2, 0)); ax.set_title(" ".join(words[t] for t in train_y[i, 1:-1])); ax.axis("off")
save_figure("11_caption_inputs")
'''),
md('''## 2. 网络结构

CNN 把图片编码为 64 维向量，用它初始化 GRU 的隐藏状态，并在每个时间步与词向量拼接。
Embedding 把词编号变成向量；GRU 输出每个位置的词表 logits。训练采用 teacher forcing，推理则使用自己上一步预测的词。

维度怎么读：图像是 `[N,3,32,32]`，句子是 `[N,5]` 个整数词编号；例如 `[1,3,6,9,2]` 对应 `<bos> a blue triangle <eos>`。编号只是查词表的索引，数值大小不表示语义相似。输入前 4 个词预测后 4 个词，故 logits 为 `[N,4,10]`；对最后一维取 argmax 后得到 `[N,4]` 预测词编号。'''),
code(r'''
class TinyCaptioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(12, 20, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Flatten(), nn.Linear(20*8*8, 64), nn.LayerNorm(64))
        self.embedding = nn.Embedding(len(words), 24)
        self.decoder = nn.GRU(24+64, 64, batch_first=True)
        self.head = nn.Linear(64, len(words))
    def forward(self, image, prefix):
        features = self.encoder(image)
        hidden = features[None]
        inputs = torch.cat([self.embedding(prefix), features[:, None].expand(-1, prefix.shape[1], -1)], dim=-1)
        output, _ = self.decoder(inputs, hidden)
        return self.head(output)
    def generate(self, image):
        features = self.encoder(image)
        hidden = features[None]
        token = torch.full((len(image), 1), stoi["<bos>"], dtype=torch.long, device=image.device)
        result = []
        for _ in range(4):
            inputs = torch.cat([self.embedding(token), features[:, None]], dim=-1)
            out, hidden = self.decoder(inputs, hidden)
            token = self.head(out).argmax(-1)
            result.append(token[:, 0])
        return torch.stack(result, dim=1)

model = TinyCaptioner().to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_x, train_y = train_x.to(dev), train_y.to(dev)
print("编码向量形状:", tuple(model.encoder(train_x[:2]).shape))
print("输出 logits [batch,时间,词表]:", tuple(model(train_x[:2], train_y[:2, :-1]).shape))
'''),
md('''## 3. 训练：预测每一个“下一个词”

交叉熵直接接收 logits，不先做 Softmax。批次随机抽样；固定种子方便比较。'''),
code(r'''
losses = []
model.train()
for step in range(420):
    ids = torch.randint(len(train_x), (32,), device=dev)
    logits = model(train_x[ids], train_y[ids, :-1])
    loss = F.cross_entropy(logits.reshape(-1, len(words)), train_y[ids, 1:].reshape(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
print("初期/末期损失:", np.mean(losses[:20]), np.mean(losses[-20:]))
plt.figure(figsize=(7, 3)); plt.plot(losses); plt.xlabel("训练步"); plt.ylabel("词元交叉熵")
save_figure("11_caption_loss")
'''),
md('''## 4. 独立图片 → 逐词输出 → 完整句子

推理时只给 `<bos>`；颜色与形状必须来自图像编码，不能从标签或文件名读取。'''),
code(r'''
model.eval()
with torch.no_grad():
    predictions = model.generate(test_x.to(dev)).cpu()
    visual_vectors = model.encoder(test_x[:2].to(dev)).cpu().numpy()
def decode(tokens):
    return " ".join(words[int(i)] for i in tokens if words[int(i)] not in ["<bos>", "<eos>", "<pad>"])
exact = (predictions == test_y[:, 1:]).all(1).float().mean().item()
token_acc = (predictions == test_y[:, 1:]).float().mean().item()
print(f"保留集句子全对率={exact:.3f}, token准确率={token_acc:.3f}")
print("图像编码向量前 8 维:\n", np.round(visual_vectors[:, :8], 3))
print("预测 token 矩阵:\n", predictions[:5].numpy())
fig, axs = plt.subplots(2, 4, figsize=(13, 6))
for i, ax in enumerate(axs.flat):
    ax.imshow(test_x[i].permute(1, 2, 0)); ax.set_title(f"预测: {decode(predictions[i])}\n真值: {decode(test_y[i])}"); ax.axis("off")
save_figure("11_caption_outputs")
assert predictions.shape == test_y[:, 1:].shape
'''),
md('''## 5. 为什么不能直接描述真实照片

当前词表只有颜色、几何形状，没有“人、动物、动作”等词，训练图也没有这些场景。
即使对自然照片输出一句通顺的句子，也可能是错误外推。通用图像描述需要真实图文数据与更强的视觉/语言编码器。

练习：添加 yellow 和 rectangle，并同步修改数据生成器与词表；将老师强制输入的前缀换成自回归训练，观察差异。'''),
md('''## 6. 参考与扩展

- [Show and Tell 原论文](https://arxiv.org/abs/1411.4555)：图像编码器与语言解码器的经典思路。
- [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)：接口与隐藏状态形状。
- 本课为独立的小型受控任务实现，没有复制论文训练结果。''')])


write("12_super_resolution.ipynb", [
md('''# 12 图像超分辨率：低清矩阵 → 插值 → 学习残差 → 高清预测

默认用 sklearn 自带的手写数字，在独立测试集上比较双三次插值与小型 CNN。
8×8 图像先降采样到 4×4，再重建到 8×8。这是尺寸很小的教学实验，不是修复真实照片细节的保证。'''),
code(SETUP),
md('''## 1. 构造可解释的低清—高清配对

只按训练集训练，保留集只评价。降采样用面积插值；基线使用双三次插值。

数值怎么读：一张低清图片是 4×4 矩阵，插值与目标各是 8×8 矩阵。批量输入 CNN 时增加通道维，变成 `[N,1,8,8]`。例如输出矩阵中 0.9 表示接近白色、0.1 表示接近黑色；范围为 [0,1]，不是原始数字数据的 0～16。插值只增加采样位置，学习残差才利用训练数据修正局部结构。'''),
code(r'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
images = load_digits().images.astype(np.float32)/16.
train, test = train_test_split(images, test_size=.2, random_state=42)
def degrade_and_resize(images):
    lr = np.stack([cv2.resize(im, (4, 4), interpolation=cv2.INTER_AREA) for im in images])
    up = np.stack([cv2.resize(im, (8, 8), interpolation=cv2.INTER_CUBIC) for im in lr])
    return lr, up
train_lr, train_up = degrade_and_resize(train)
test_lr, test_up = degrade_and_resize(test)
print("低清输入 4×4:\n", np.round(test_lr[0], 2))
print("高清目标前 4×4:\n", np.round(test[0, :4, :4], 2))
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for ax, im, title in zip(axs, [test_lr[0], test_up[0], test[0]], ["输入 LR 4×4", "双三次 8×8", "目标 HR 8×8"]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=1); ax.set_title(title); ax.axis("off")
save_figure("12_sr_inputs")
'''),
md(r'''## 2. 残差学习

$\hat I_{HR}=I_{bicubic}+f_\theta(I_{bicubic})$。CNN 只预测修正量。
输出层零初始化，使模型开始时等价于插值基线，避免比较一个故意很差的随机基线。'''),
code(r'''
class ResidualSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return x+self.net(x)

model = ResidualSR().to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=.002)
X = torch.from_numpy(train_up[:, None]).to(dev)
Y = torch.from_numpy(train[:, None]).to(dev)
losses = []
for step in range(250):
    ids = torch.randint(len(X), (64,), device=dev)
    loss = F.mse_loss(model(X[ids]), Y[ids])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
plt.figure(figsize=(7, 3)); plt.plot(losses); plt.xlabel("训练步"); plt.ylabel("像素 MSE")
save_figure("12_sr_loss")
'''),
md(r'''## 3. 评价：同一保留集上的 MSE 与 PSNR

图像范围为 [0,1]，$PSNR=10\log_{10}(1/MSE)$。按每张图先算 PSNR 再取均值。
只看视觉图容易选到有利的样本，因此同时报告整个测试集。'''),
code(r'''
model.eval()
with torch.no_grad():
    sr = model(torch.from_numpy(test_up[:, None]).to(dev)).clamp(0, 1).cpu().numpy()[:, 0]
def mean_psnr(prediction, target):
    mse = np.mean((prediction-target)**2, axis=(1, 2))
    return np.mean(10*np.log10(1/np.maximum(mse, 1e-12)))
print(f"双三次 PSNR={mean_psnr(test_up.clip(0,1),test):.2f} dB")
print(f"残差CNN PSNR={mean_psnr(sr,test):.2f} dB")
print("输出超分矩阵前 4×4:\n", np.round(sr[0, :4, :4], 2))
fig, axs = plt.subplots(3, 4, figsize=(10, 7))
for row in range(3):
    for ax, im, title in zip(axs[row], [test_lr[row], test_up[row], sr[row], test[row]], ["LR", "Bicubic", "CNN预测", "真实HR"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1); ax.set_title(title); ax.axis("off")
save_figure("12_sr_outputs")
assert sr.shape == test.shape and np.isfinite(sr).all()
'''),
md('''## 4. 真实照片域外实验

只在数字上训练的模型，不一定能提升水果照片。下图使用同一低清图比较插值与 CNN，并保留原图作为参考。
这种域外实验可以揭示训练分布限制，不应把“像素数变多”写成“真实细节恢复”。'''),
code(r'''
photo_path = ROOT / "assets/photos/fruits.jpg"
if photo_path.exists():
    photo = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
    photo = cv2.resize(photo, (64, 64)).astype(np.float32)/255.
    lr = cv2.resize(photo, (32, 32), interpolation=cv2.INTER_AREA)
    bicubic = cv2.resize(lr, (64, 64), interpolation=cv2.INTER_CUBIC)
    with torch.no_grad():
        output = model(torch.from_numpy(bicubic[None, None]).to(dev)).clamp(0, 1).cpu().numpy()[0, 0]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    for ax, im, title in zip(axs, [lr, bicubic, output, photo], ["照片 LR", "Bicubic", "数字模型外推", "原图参考"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1); ax.set_title(title); ax.axis("off")
    save_figure("12_sr_photo_domain_shift")
else:
    print("请先运行项目素材准备步骤，再尝试真实照片域外实验。")
'''),
md('''## 5. 学生练习

1. 改成 2×2 → 8×8，比较信息损失。
2. 在同一测试集上检查每张图的 PSNR，找出最差的 5 张。
3. 将 MSE 改为 L1，解释训练目标与视觉效果的关系。

来源：[SRCNN 原论文](https://arxiv.org/abs/1501.00092)、[EDSR 残差超分辨率](https://arxiv.org/abs/1707.02921)。本课是简化的残差 CNN，并非完整 SRCNN/EDSR 复现。''')])


write("13_depth.ipynb", [
md('''# 13 双目深度：左右图 → 视差矩阵 → 米制深度

用可复现的纹理平面生成一对已经极线校正的图片，运行真实 StereoSGBM 匹配。
已知焦距与基线才可以把视差换成米制深度；单张照片并不能直接套用该公式。'''),
code(SETUP),
md('''## 1. 输入：具有已知位移的左右图

世界是一个深度为 2 m 的平面，焦距 240 px，双目基线 0.1 m，所以真实视差为 12 px。
右图中同一点比左图向左 12 px；边界产生无法匹配的区域。'''),
code(r'''
rng = np.random.default_rng(42)
height, width = 128, 224
texture = rng.integers(0, 256, (height, width), dtype=np.uint8)
left = cv2.GaussianBlur(texture, (3, 3), .6)
focal_px, baseline_m, true_depth_m = 240., .1, 2.
d_true = int(focal_px*baseline_m/true_depth_m)
right = rng.integers(0, 256, left.shape, dtype=np.uint8)
right[:, :-d_true] = left[:, d_true:]
print("真实视差:", d_true, "px")
print("左图局部矩阵:\n", left[60:64, 100:106])
print("右图对应局部矩阵:\n", right[60:64, 100-d_true:106-d_true])
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, im, title, x in zip(axs, [left, right], ["左图", "右图"], [100,100-d_true]):
    ax.imshow(im, cmap="gray"); ax.scatter([x], [60], c="red"); ax.axhline(60,c="yellow",lw=.7); ax.set_title(title); ax.axis("off")
save_figure("13_depth_inputs")
'''),
md(r'''## 2. 匹配视差

StereoSGBM 综合局部匹配代价与平滑约束。`numDisparities` 必须是 16 的倍数；`blockSize` 取奇数。
OpenCV 返回有符号定点值，需要除以 16 才得到像素视差。非正视差视为无效。

矩阵怎么读：左图、右图和视差图都是 `[128,224]`，相同的行列索引指向左相机的对应像素。原始值 192 除以 16 得到 12 px；再计算 `240×0.1/12=2 m`。深度图中的 NaN 表示当前没有有效估计，不表示物体距离为零。'''),
code(r'''
matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=5,
                               P1=8*5**2, P2=32*5**2, uniquenessRatio=8,
                               speckleWindowSize=0, disp12MaxDiff=1)
raw = matcher.compute(left, right)
disparity = raw.astype(np.float32)/16.
valid = disparity > 0
evaluation_mask = valid.copy()
evaluation_mask[:, :40] = False
evaluation_mask[:, -20:] = False
evaluation_mask[:10] = False; evaluation_mask[-10:] = False
print("原始定点值 4×6:\n", raw[60:64, 100:106])
print("视差 px 4×6:\n", disparity[60:64, 100:106])
print("中央区域有效率:", evaluation_mask.sum()/((width-60)*(height-20)))
'''),
md(r'''## 3. 视差转深度

校正后的理想双目几何：$d=x_L-x_R$，$Z=fB/d$。
$f$ 单位为像素，$B$ 单位为米，则 $Z$ 为米。无效值用 NaN，而不是当作 0 m。'''),
code(r'''
depth = np.full(disparity.shape, np.nan, np.float32)
depth[valid] = focal_px*baseline_m/disparity[valid]
disparity_mae = np.mean(np.abs(disparity[evaluation_mask]-d_true))
depth_mae = np.mean(np.abs(depth[evaluation_mask]-true_depth_m))
print(f"中央有效像素视差 MAE={disparity_mae:.4f} px")
print(f"中央有效像素深度 MAE={depth_mae:.4f} m")
print("深度局部矩阵 m:\n", np.round(depth[60:64,100:106],3))
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
axs[0].imshow(left,cmap="gray"); axs[0].set_title("输入左图")
p = axs[1].imshow(np.where(valid,disparity,np.nan),cmap="turbo",vmin=0,vmax=20); axs[1].set_title("匹配视差 / px"); fig.colorbar(p,ax=axs[1])
p = axs[2].imshow(depth,cmap="viridis",vmin=1.5,vmax=2.5); axs[2].set_title("深度 / m"); fig.colorbar(p,ax=axs[2])
for ax in axs: ax.axis("off")
save_figure("13_depth_outputs")
assert evaluation_mask.any() and np.isfinite(depth[evaluation_mask]).all()
'''),
md('''## 4. 误差传播与失败条件

同样 1 px 视差误差对远处深度影响更大。纹理不足、重复图案、遮挡、反光、未校正左右图都会造成错误匹配。
本课只验证一个有纹理的平面，不能据此宣称复杂场景都能达到相同误差。'''),
code(r'''
ds = np.arange(1, 25, dtype=float)
z = focal_px*baseline_m/ds
z_perturbed = focal_px*baseline_m/(ds+1)
plt.figure(figsize=(7, 3)); plt.plot(ds,z,label="正确视差"); plt.plot(ds,z_perturbed,label="视差 +1 px")
plt.xlabel("视差 px"); plt.ylabel("深度 m"); plt.legend(); plt.grid(alpha=.3)
save_figure("13_depth_sensitivity")
'''),
md('''## 5. 学生练习与真实双目扩展

将纹理换成纯灰色，比较无效率；把位移改为 6 px，先预测深度再运行。
真实双目需要相机标定、畸变校正、极线校正和相机参数文件，不能仅把任意两张图片代入。

来源：[OpenCV 双目深度教程](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)、[StereoSGBM 参数](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)。''')])


write("14_optical_flow.ipynb", [
md('''# 14 光流：两帧图像 → 每个像素的运动向量

观察 `(u,v)` 位移矩阵、箭头图和彩色光流图；用已知平移量计算端点误差 EPE。
运行 OpenCV Farnebäck 稠密光流，不需要显卡或模型权重。'''),
code(SETUP),
md('''## 1. 可验证的输入

对同一纹理图像施加已知平移 dx=4、dy=2 px。二维图像坐标 x 向右、y 向下，因此正 v 表示向下移动。'''),
code(r'''
rng = np.random.default_rng(8)
frame0 = cv2.GaussianBlur(rng.integers(0,256,(128,160),dtype=np.uint8),(5,5),.9)
cv2.circle(frame0,(65,60),20,220,2)
dx, dy = 4., 2.
M = np.float32([[1,0,dx],[0,1,dy]])
frame1 = cv2.warpAffine(frame0,M,(160,128),borderMode=cv2.BORDER_REFLECT)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,im,title in zip(axs,[frame0,frame1],["t 帧", "t+1 帧：右移4/下移2"]):
    ax.imshow(im,cmap="gray");ax.set_title(title);ax.axis("off")
save_figure("14_flow_inputs")
print("输入局部像素矩阵:\n",frame0[60:64,60:65])
'''),
md('''## 2. 稠密光流

`flow[y,x,0]=u`、`flow[y,x,1]=v`，单位为相邻两帧间的像素位移。
只有已知帧间隔，才能转换成像素/秒；要变成米/秒还需要几何标定。

矩阵怎么读：输入两帧均为 `[128,160]` 灰度矩阵，输出是 `[128,160,2]`。若 `flow[60,60]=[4,2]`，表示当前帧 `(x=60,y=60)` 附近内容在下一帧移动到约 `(64,62)`；这两个值不是颜色，也不是下一帧的绝对坐标。`u`、`v` 的局部矩阵分别展示水平和垂直分量。'''),
code(r'''
flow = cv2.calcOpticalFlowFarneback(frame0,frame1,None,.5,3,21,5,7,1.5,0)
print("输出 [H,W,2]:",flow.shape)
print("u 局部矩阵:\n",np.round(flow[60:64,60:65,0],2))
print("v 局部矩阵:\n",np.round(flow[60:64,60:65,1],2))
y,x=np.mgrid[10:118:8,10:150:8]
plt.figure(figsize=(8,5));plt.imshow(frame0,cmap="gray")
plt.quiver(x,y,flow[y,x,0],flow[y,x,1],color="red",angles="xy",scale_units="xy",scale=1)
plt.title("箭头表示估计运动：右/下为正")
save_figure("14_flow_vectors")
'''),
md(r'''## 3. 用真实位移计算 EPE

$EPE=\sqrt{(u-u^*)^2+(v-v^*)^2}$。评价只使用远离图像边界的区域，避免边界填充影响。
还检查“用流场把下一帧采样回当前帧”后的亮度差。'''),
code(r'''
margin=20
interior=flow[margin:-margin,margin:-margin]
epe=np.sqrt((interior[...,0]-dx)**2+(interior[...,1]-dy)**2)
print(f"中央区域平均 EPE={epe.mean():.4f} px")
yy,xx=np.mgrid[0:128,0:160].astype(np.float32)
warped=cv2.remap(frame1,xx+flow[...,0],yy+flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
photometric=np.abs(warped.astype(float)-frame0.astype(float))
print("配准后中央亮度 MAE:",photometric[margin:-margin,margin:-margin].mean())
mag,ang=cv2.cartToPolar(flow[...,0],flow[...,1])
hsv=np.zeros((128,160,3),np.uint8)
hsv[...,0]=np.mod(ang*90/np.pi,180).astype(np.uint8)
hsv[...,1]=255;hsv[...,2]=np.clip(mag/8*255,0,255).astype(np.uint8)
flow_rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,im,title in zip(axs,[flow_rgb,warped,photometric],["方向=色相，幅度=亮度", "下一帧回采样", "亮度绝对误差"]):
    ax.imshow(im,cmap=None if im.ndim==3 else "gray");ax.set_title(title);ax.axis("off")
save_figure("14_flow_outputs")
assert np.isfinite(flow).all() and flow.shape[-1]==2
'''),
md('''## 4. 真实图片的可控运动

将项目内的水果照片平移后估计光流；注意低纹理区域与边界的误差变化。这仍是已知人工运动的照片实验，不是实拍视频评测。'''),
code(r'''
photo_path=ROOT/"assets/photos/fruits.jpg"
if photo_path.exists():
    a=cv2.resize(cv2.imread(str(photo_path),cv2.IMREAD_GRAYSCALE),(160,128))
    b=cv2.warpAffine(a,M,(160,128),borderMode=cv2.BORDER_REFLECT)
    f=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,21,5,7,1.5,0)
    err=np.linalg.norm(f[20:-20,20:-20]-np.array([dx,dy]),axis=-1)
    print(f"真实照片人工平移 EPE={err.mean():.3f} px")
    plt.figure(figsize=(8,5));plt.imshow(a,cmap="gray")
    plt.quiver(x,y,f[y,x,0],f[y,x,1],color="red",angles="xy",scale_units="xy",scale=1)
    plt.title("真实照片上的估计运动")
    save_figure("14_flow_photo")
else:
    print("请先准备项目真实照片素材。")
'''),
md('''## 5. 练习与边界

把位移改为 20 px、改变金字塔层数；再只改变亮度，观察亮度一致性假设的影响。
真实场景还会遇到遮挡、新出现区域、快速运动和运动模糊。可比较稀疏 Lucas–Kanade 与本课稠密 Farnebäck 的输出范围。

来源：[OpenCV 光流教程](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)。''')])


write("15_reconstruction.ipynb", [
md('''# 15 三维重建：投影图像 → 对应点 → 三角化 → 点云

从带已知相机内外参的合成立方体开始，每个输入点都能追溯到真实 3D 坐标。
这是双视角三角化实验；不冒充完整的 SfM/MVS 或自动从任意照片恢复三维。'''),
code(SETUP),
md(r'''## 1. 定义世界点和相机矩阵

针孔模型 $\lambda [u,v,1]^T=P[X,Y,Z,1]^T$，$P=K[R|t]$。
第二台相机在世界中向右移动 B，其外参平移是 -B；不要把相机位置与外参 t 混为一谈。'''),
code(r'''
f=180.;cx=120.;cy=90.;baseline=.4
K=np.array([[f,0,cx],[0,f,cy],[0,0,1]],float)
P1=K@np.column_stack([np.eye(3),np.zeros(3)])
P2=K@np.column_stack([np.eye(3),np.array([-baseline,0,0])])
cube=np.array([[x,y,z] for z in [2.,3.] for y in [-.4,.4] for x in [-.4,.4]],float)
rng=np.random.default_rng(42)
world=np.vstack([cube, rng.uniform([-.4,-.4,2],[.4,.4,3],size=(60,3))])
def project(P, xyz):
    h=np.column_stack([xyz,np.ones(len(xyz))])@P.T
    return h[:,:2]/h[:,2:3]
uv1=project(P1,world);uv2=project(P2,world)
print("相机 K:\n",K,"\nP2:\n",P2)
print("世界点前4个:\n",world[:4],"\n左图坐标:\n",np.round(uv1[:4],2))
'''),
md('''## 2. 将投影点画成两幅输入图

点的颜色和编号表示已知对应关系。真实重建必须自己检测/匹配特征并剔除错误匹配；本课把对应关系作为已知输入，集中学习几何过程。'''),
code(r'''
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,uv,title in zip(axs,[uv1,uv2],["左相机输入投影", "右相机输入投影"]):
    ax.set_facecolor("#f0f0f0");ax.scatter(uv[:,0],uv[:,1],c=world[:,2],cmap="viridis",s=25)
    for i in range(8):ax.text(uv[i,0]+1,uv[i,1],str(i))
    ax.set_xlim(0,240);ax.set_ylim(180,0);ax.set_aspect("equal");ax.set_title(title)
save_figure("15_reconstruction_inputs")
'''),
md(r'''## 3. 三角化与齐次归一化

维度怎么读：每台相机的二维对应点是 `[N,2]`，列按 `(u,v)` 排列；OpenCV 接口转置为 `[2,N]`。投影矩阵为 `[3,4]`。三角化给出 `[4,N]` 齐次点，前三行除以第四行后，转置得到 `[N,3]` 的 `(X,Y,Z)` 世界坐标。齐次坐标本身不直接表示米，也不是 RGB 像素值。

OpenCV 输出 4×N 齐次点 $[X_h,Y_h,Z_h,W]^T$，需除以 W 得到欧式坐标。'''),
code(r'''
homogeneous=cv2.triangulatePoints(P1,P2,uv1.T,uv2.T)
reconstructed=(homogeneous[:3]/homogeneous[3:4]).T
print("齐次矩阵前4列:\n",np.round(homogeneous[:,:4],3))
print("重建点前4个:\n",np.round(reconstructed[:4],3))
rmse=np.sqrt(np.mean(np.sum((reconstructed-world)**2,axis=1)))
reproj=np.mean(np.linalg.norm(project(P1,reconstructed)-uv1,axis=1))
print(f"无噪声 3D RMSE={rmse:.8f} m；左图重投影误差={reproj:.8f} px")
assert rmse<1e-5
'''),
md('''## 4. 加入匹配噪声，观察实际误差

给两张图的坐标分别加 0.3 px 高斯噪声，再重复三角化。重投影误差很小并不自动保证三维误差很小。'''),
code(r'''
uv1_noisy=uv1+rng.normal(0,.3,uv1.shape)
uv2_noisy=uv2+rng.normal(0,.3,uv2.shape)
h=cv2.triangulatePoints(P1,P2,uv1_noisy.T,uv2_noisy.T)
xyz_noisy=(h[:3]/h[3:4]).T
point_error=np.linalg.norm(xyz_noisy-world,axis=1)
print(f"有噪声 3D RMSE={np.sqrt(np.mean(point_error**2)):.4f} m")
fig=plt.figure(figsize=(11,4))
ax=fig.add_subplot(121,projection="3d")
ax.scatter(world[:,0],world[:,1],world[:,2],label="真值",s=15)
ax.scatter(xyz_noisy[:,0],xyz_noisy[:,1],xyz_noisy[:,2],label="重建",marker="x",s=15)
ax.set_xlabel("X / m");ax.set_ylabel("Y / m");ax.set_zlabel("Z / m");ax.legend();ax.set_title("稀疏点云")
ax2=fig.add_subplot(122);ax2.hist(point_error,bins=15);ax2.set_xlabel("点三维误差 / m");ax2.set_title("匹配噪声导致的误差")
save_figure("15_reconstruction_outputs")
'''),
md('''## 5. 输出可查看的点云文件

保存 PLY 文本点云，可用支持 PLY 的三维软件打开。它只有点，没有表面网格、纹理或完整物体表面。'''),
code(r'''
out_dir=ROOT/"outputs";out_dir.mkdir(exist_ok=True)
ply=out_dir/"15_reconstructed_points.ply"
with ply.open("w",encoding="utf-8") as file:
    file.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz_noisy)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    np.savetxt(file,xyz_noisy,fmt="%.6f")
print("点云已保存:",ply)
'''),
md('''## 6. 完整重建还缺什么

实际 SfM/MVS 流程包括：相机标定、特征检测与匹配、几何验证、位姿估计、三角化、束调整、稠密重建、网格与纹理。
练习：把基线从 0.4 m 改为 0.05 m，在同样坐标噪声下重新计算；解释误差为什么变化。

来源：[OpenCV triangulatePoints](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)、[COLMAP 官方教程](https://colmap.github.io/tutorial.html)。''')])


write("16_pose.ipynb", [
md('''# 16 姿态估计：骨架图片 → 关键点热图 → 坐标与骨架

用随机姿态的合成火柴人训练小型热图网络，每个输出通道对应一个关节。
这是“关键点热图”机制的可运行教学，不是已训练的人体姿态模型；不能据此对真实人物作可靠识别。'''),
code(SETUP),
md('''## 1. 生成图像、坐标和目标热图

关键点顺序：头、颈、左手、右手、髋、左脚、右脚。所有坐标按 (x,y) 保存。
训练与测试独立随机种子，尺度固定 32×32；关节角度与位置随机。

维度怎么读：图像数组原为 `[N,32,32]`，送入网络前补单通道维成为 **NCHW = `[N,1,32,32]`**。N 是图片数，C 是通道数，H/W 是高/宽。目标与预测热图是 `[N,K,32,32]`，本课 K=7；第 k 个通道只负责第 k 个关键点。坐标结果是 `[N,K,2]`，最后两项始终按 `(x,y)` 排列。

用一个仅供说明的 3×3 归一化热图读回坐标：

```text
[[0.01, 0.02, 0.01],
 [0.03, 0.70, 0.05],
 [0.02, 0.04, 0.12]]
```

这九个数的和为 1，每项对应一个候选像素位置，并非像素亮度。最大值 0.70 在第 1 行、第 1 列（从 0 开始数）；按行展开后 argmax 索引为 4。对于宽度 W=3，`x=4%3=1`、`y=4//3=1`，因此坐标是 `(1,1)`。实际代码用 W=32 做同样换算。某处响应最大只表示模型选了该位置，仍须与真值比较；示例中的 0.70 不是本次训练测量值。'''),
code(r'''
names=["head","neck","left hand","right hand","hip","left foot","right foot"]
edges=[(0,1),(1,2),(1,3),(1,4),(4,5),(4,6)]
def pose_data(n,seed):
    rng=np.random.default_rng(seed);images=[];points=[];maps=[]
    yy,xx=np.mgrid[:32,:32]
    for _ in range(n):
        cx=float(rng.uniform(14,18));y=float(rng.uniform(8,11))
        xy=np.array([[cx,y-4],[cx,y],[cx-rng.uniform(5,10),y+rng.uniform(-2,7)],
                     [cx+rng.uniform(5,10),y+rng.uniform(-2,7)],[cx,y+9],
                     [cx-rng.uniform(3,7),y+rng.uniform(15,18)],
                     [cx+rng.uniform(3,7),y+rng.uniform(15,18)]],np.float32)
        im=np.zeros((32,32),np.uint8)
        for a,b in edges:cv2.line(im,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),180,1,cv2.LINE_AA)
        for pt in xy:cv2.circle(im,tuple(pt.astype(int)),1,255,-1)
        heat=np.exp(-((xx[None]-xy[:,0,None,None])**2+(yy[None]-xy[:,1,None,None])**2)/(2*1.4**2))
        images.append(im/255.);points.append(xy);maps.append(heat)
    return np.array(images,np.float32),np.array(points,np.float32),np.array(maps,np.float32)

train_im,train_pts,train_hm=pose_data(240,10)
test_im,test_pts,test_hm=pose_data(48,999)
print("输入图:",train_im.shape,"热图:",train_hm.shape)
print("坐标矩阵 [关键点,xy]:\n",np.round(test_pts[0],2))
print("头部目标热图局部:\n",np.round(test_hm[0,0,3:7,13:17],2))
fig,axs=plt.subplots(1,3,figsize=(10,3))
axs[0].imshow(test_im[0],cmap="gray");axs[0].set_title("输入火柴人")
axs[1].imshow(test_hm[0,2],cmap="hot");axs[1].set_title("左手目标热图")
axs[2].imshow(test_im[0],cmap="gray");axs[2].scatter(test_pts[0,:,0],test_pts[0,:,1],c=range(7),cmap="tab10");axs[2].set_title("关键点真值")
for ax in axs:ax.axis("off")
save_figure("16_pose_inputs")
'''),
md('''## 2. 热图回归网络

使用卷积提取空间特征，输出 7 个 32×32 热图 logits，再对每个关节的空间位置做 Softmax。
将目标高斯归一化成概率分布，最小化分布交叉熵。额外输入 x/y 坐标通道帮助小网络区分左右与上下；这也是明确的建模假设。'''),
code(r'''
class HeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,5,padding=2),nn.ReLU(),
                               nn.Conv2d(16,24,5,padding=2),nn.ReLU(),
                               nn.Conv2d(24,24,5,padding=2),nn.ReLU(),
                               nn.Conv2d(24,7,1))
    def forward(self,x):
        yy,xx=torch.meshgrid(torch.linspace(-1,1,32,device=x.device),torch.linspace(-1,1,32,device=x.device),indexing="ij")
        xy=torch.stack([xx,yy])[None].expand(len(x),-1,-1,-1)
        return self.net(torch.cat([x,xy],dim=1))

model=HeatmapNet().to(dev)
optimizer=torch.optim.Adam(model.parameters(),lr=.003)
X=torch.from_numpy(train_im[:,None]).to(dev)
Y=torch.from_numpy(train_hm).to(dev)
losses=[]
for step in range(320):
    ids=torch.randint(len(X),(24,),device=dev)
    pred=model(X[ids]);target=Y[ids]
    target=target.flatten(2);target=target/target.sum(-1,keepdim=True)
    loss=-(target*pred.flatten(2).log_softmax(-1)).sum(-1).mean()
    optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
plt.figure(figsize=(7,3));plt.plot(losses);plt.xlabel("训练步");plt.ylabel("空间分布交叉熵")
save_figure("16_pose_loss")
'''),
md('''## 3. 从热图读出坐标

每通道最大值的位置作为关节坐标。它的值称为响应强度，未经校准不能直接等同于可靠概率。
使用像素定位误差和 PCK@3 px，分别评价误差大小与三像素阈值内的比例。'''),
code(r'''
model.eval()
with torch.no_grad():
    logits=model(torch.from_numpy(test_im[:,None]).to(dev))
    pred_hm=logits.flatten(2).softmax(-1).reshape_as(logits).cpu().numpy()
flat=pred_hm.reshape(len(test_im),7,-1).argmax(-1)
pred_pts=np.stack([flat%32,flat//32],axis=-1)
errors=np.linalg.norm(pred_pts-test_pts,axis=-1)
print(f"保留集平均关键点误差={errors.mean():.3f} px；PCK@3px={(errors<3).mean():.3f}")
print("预测坐标矩阵:\n",pred_pts[0])
print("预测左手热图局部:\n",np.round(pred_hm[0,2,9:13,6:10],2))
fig,axs=plt.subplots(2,4,figsize=(12,6))
for i,ax in enumerate(axs.flat):
    ax.imshow(test_im[i],cmap="gray")
    for a,b in edges:
        ax.plot(test_pts[i,[a,b],0],test_pts[i,[a,b],1],"g-",lw=1,alpha=.6)
        ax.plot(pred_pts[i,[a,b],0],pred_pts[i,[a,b],1],"r--",lw=1)
    ax.scatter(pred_pts[i,:,0],pred_pts[i,:,1],c="red",s=10)
    ax.set_title(f"绿=真值 红=预测\n平均误差 {errors[i].mean():.1f}px");ax.axis("off")
save_figure("16_pose_outputs")
assert pred_hm.shape==(len(test_im),7,32,32) and np.isfinite(pred_hm).all()
'''),
md('''## 4. 对比两类容易混淆的关节

左手/右手通道不同，即使外观类似也必须区分。遮挡会让热图不确定；低分辨率坐标还存在取整误差。'''),
code(r'''
fig,axs=plt.subplots(1,3,figsize=(10,3))
for ax,im,title in zip(axs,[test_im[0],pred_hm[0,2],pred_hm[0,3]],["输入", "预测左手热图", "预测右手热图"]):
    ax.imshow(im,cmap="hot" if "热图" in title else "gray");ax.set_title(title);ax.axis("off")
save_figure("16_pose_heatmaps")
'''),
md('''## 5. 学生练习与真实任务扩展

遮住一只手重新预测；去掉 x/y 坐标通道；改变高斯半径，观察 PCK。
真实人体姿态需要真实人物与关节标注、多人关联、尺度变化和遮挡处理，评价常见 PCK/OKS。

来源：[Stacked Hourglass 原论文](https://arxiv.org/abs/1603.06937)、[MMPose 官方项目](https://github.com/open-mmlab/mmpose)。本课仅学习热图回归机制，不复现这些大型模型的精度。''')])

if __name__ == "__main__":
    print("已生成 8 份高级视觉入门 Notebook。")
