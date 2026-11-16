"""生成本课程基础 Notebook；不需要 nbformat 参与生成。"""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
INIT = '''
from pathlib import Path
import sys
root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_utils.py").exists()), None)
if root is None:
    raise RuntimeError("请在完整的 cv-beginner-course 项目里打开 Notebook")
if str(root) not in sys.path: sys.path.insert(0, str(root))
from course_utils import setup, device, show_images, save_figure, ROOT
ROOT = setup(seed=42)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
print("计算设备:", device())
'''


def write(name, cells):
    target = ROOT / 'notebooks' / name
    target.parent.mkdir(exist_ok=True)
    out = []
    for i, (kind, text) in enumerate(cells):
        cell = {'id': f'{name[:2]}-{i:03}', 'cell_type': kind, 'metadata': {},
                'source': dedent(text).strip() + '\n'}
        if kind == 'code': cell.update(execution_count=None, outputs=[])
        out.append(cell)
    target.write_text(json.dumps({'cells': out, 'metadata': {
        'kernelspec': {'display_name': 'Python 3 (CV)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'}},
        'nbformat': 4, 'nbformat_minor': 5}, ensure_ascii=False, indent=1), encoding='utf-8')


M = lambda s: ('markdown', s)
C = lambda s: ('code', s)

write('00_start_here.ipynb', [
M('''# 00 从这里开始：一张图片怎样成为计算机里的数值

目标：确认环境，读入真实图片，把像素转为矩阵，执行一次 CPU/GPU 兼容计算。
先阅读项目根目录 README，安装完成后，选择 Python 内核，点击“运行 → 运行所有单元格”。
每一课都可以在新内核中独立运行；不要把安装命令混在算法单元中。'''), C(INIT),
M('''## 1 输入：项目自带的真实照片
图片来自 OpenCV 官方样例，下载来源和校验值见 assets/SOURCES.json。读取失败时立即报错，避免悄悄把随机图当成照片。'''),
C('''path = ROOT / "assets/photos/fruits.jpg"
bgr = cv2.imread(str(path))
if bgr is None: raise FileNotFoundError(path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
print("H,W,C=", rgb.shape, "类型=", rgb.dtype, "范围=", (rgb.min(), rgb.max()))
show_images([rgb], ["输入：真实水果照片（RGB）"])
save_figure("00_input_photo")'''),
M('''## 2 一个像素与一个局部矩阵
RGB 像素有三个通道。`rgb[y, x]` 表示第 y 行、第 x 列；OpenCV 读入时通道顺序是 BGR，显示前转换。'''),
C('''print("左上像素 RGB:", rgb[0, 0])
print("红通道左上 5×5 矩阵:\\n", rgb[:5, :5, 0])
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
print("8×8灰度矩阵:\\n", small)
fig, ax = plt.subplots(figsize=(6, 6)); ax.imshow(small, cmap="gray", vmin=0, vmax=255)
for y in range(8):
    for x in range(8): ax.text(x, y, str(small[y,x]), ha="center", va="center", color="red", fontsize=9)
ax.set_title("像素与矩阵逐项对应"); save_figure("00_pixel_matrix")'''),
M('''## 3 过程：数组变成张量
图像常为 H×W×C，PyTorch 卷积输入为 N×C×H×W。`N` 是批量大小，像素除以 255 后处于 [0,1]。'''),
C('''x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
x = x.to(device())
bright = (x * 1.25).clamp(0, 1)
output = bright[0].detach().cpu().permute(1, 2, 0).numpy()
print("输入张量:", tuple(x.shape), x.device, "输出:", output.shape)
show_images([rgb, output], ["输入", "输出：亮度乘1.25并截断"])
save_figure("00_input_output")
assert x.ndim == 4 and np.isfinite(output).all()'''),
M('''## 4 自己做
1. 把系数改成 0.6，再检查三个像素的前后变化。
2. 把图片替换为自己拍摄的照片，观察 H、W 是否改变。
3. 修改环境变量 `CV_DEVICE=cpu` 可以强制 CPU；默认自动选用 CUDA/MPS/CPU。

本机仅验证 CPU。GPU 代码使用相同算法、张量和设备迁移，实际 GPU 速度仍需在有显卡的机器测量。
下一课：01 中手算采样、量化和颜色转换。来源：[OpenCV 图像入门](https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html)、[PyTorch 张量](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)。''')])

write('01_pixels_matrices.ipynb', [
M('''# 01 像素、采样、量化、颜色与直方图
学习顺序：小矩阵手算 → 对真实照片执行同样操作 → 比较输入输出。
采样决定空间取样的密度；量化决定幅值可用的离散等级。两者不能混为“图片有多少种颜色”。'''), C(INIT),
M('''## 1 输入一个 5×5 灰度矩阵
8 位单通道可取 0–255，共 256 个值；RGB 图像是三个这样的通道。像素数不等于实际出现的颜色数量。'''),
C('''a = np.array([[0,30,60,90,120],[30,60,90,120,150],[60,90,120,150,180],
              [90,120,150,180,210],[120,150,180,210,255]], dtype=np.uint8)
print(a)
fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(a,cmap="gray",vmin=0,vmax=255)
for (y,x),value in np.ndenumerate(a): ax.text(x,y,str(value),ha="center",va="center",color="red")
save_figure("01_matrix")'''),
M('''## 2 空间采样与幅值量化
采样每隔两个位置取一个值；量化将 [0,255] 分成 4 个区间，输出 0、85、170、255。'''),
C('''sampled = a[::2, ::2]
quantized = (np.floor(a.astype(float)/64).clip(0,3) * 85).astype(np.uint8)
print("采样后:\\n",sampled, "\\n量化后:\\n",quantized)
assert len(np.unique(quantized)) <= 4
show_images([a,sampled,quantized],["输入","空间采样","4级量化"]); save_figure("01_sampling_quantization")'''),
M('''## 3 对真实照片采样与量化
最近邻放大只是把低分辨率像素画大，没有恢复原始细节。'''),
C('''bgr=cv2.imread(str(ROOT/"assets/photos/fruits.jpg")); assert bgr is not None
rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
low=cv2.resize(rgb,(32,32),interpolation=cv2.INTER_AREA)
up=cv2.resize(low,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
q=(np.floor(rgb.astype(float)/64).clip(0,3)*85).astype(np.uint8)
show_images([rgb,up,q],["原图","32×32采样再放大","每通道4级"]); save_figure("01_photo_sampling")'''),
M('''## 4 灰度、HSV 与直方图
OpenCV 的常用灰度近似为 0.299R+0.587G+0.114B（实际舍入可差一个灰度值）。HSV 是颜色表示方式，不代表“物理层面和画家层面”两个严格类别。
对 uint8 的 OpenCV HSV，H 通常范围 0–179，S、V 为 0–255。'''),
C('''gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
manual=np.round(rgb@np.array([0.299,0.587,0.114])).astype(np.uint8)
print("灰度局部矩阵:\\n",gray[:4,:4]); print("手算最大舍入差:",np.abs(gray.astype(int)-manual.astype(int)).max())
show_images([rgb,gray,hsv[:,:,0],hsv[:,:,2]],["RGB","灰度","H色相（灰度显示）","V明度"]); save_figure("01_color_channels")
fig,ax=plt.subplots(); ax.hist(gray.ravel(),bins=256,range=(0,256)); ax.set(xlabel="灰度值",ylabel="像素个数",title="输入灰度直方图")
save_figure("01_histogram")'''),
M('''## 5 亮度与对比度
`y=clip(alpha*x+beta)`：beta 平移灰度，alpha 改变对比度。uint8 直接加法可能溢出，先转浮点再截断。'''),
C('''brighter=np.clip(gray.astype(float)+50,0,255).astype(np.uint8)
contrast=np.clip((gray.astype(float)-128)*1.6+128,0,255).astype(np.uint8)
print("输入/亮度/对比度的局部值:",gray[0,:5],brighter[0,:5],contrast[0,:5])
show_images([gray,brighter,contrast],["输入","beta=50","alpha=1.6,中心128"]); save_figure("01_brightness_contrast")'''),
M('''## 6 练习与核对
手算 `a[2,2]` 的4邻域和8邻域；比较 16、32、64 像素采样；比较每通道 2、4、8 级量化。
应看到空间分辨率降低与色阶减少产生不同失真。
参考：[OpenCV 颜色空间](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)。''')])

write('02_filtering_edges.ipynb', [
M('''# 02 卷积、降噪与梯度边缘
先手算 3×3 核，再在真实图像上比较均值、高斯、中值和双边滤波，最后观察梯度与 Canny。
降噪输出仍是强度图；梯度和边缘输出描述的是变化，任务不同。'''), C(INIT),
M('''## 1 一个可手算的窗口
输入矩阵 a 的行、列对应图像 y、x 位置；每个数值是一处灰度强度。3×3 核矩阵每个元素为 1/9，输出等于窗口中9个像素乘权重后的和。奇数窗口 k 的下标范围为 −(k−1)/2 到 +(k−1)/2。下面手算中心像素，并与完整输出矩阵对应位置核对。'''),
C('''a=np.arange(25,dtype=np.float32).reshape(5,5)
kernel=np.ones((3,3),np.float32)/9
center=float((a[1:4,1:4]*kernel).sum())
filtered=cv2.filter2D(a,-1,kernel,borderType=cv2.BORDER_REFLECT)
print("输入:\\n",a,"\\n核:\\n",kernel,"\\n中心输出:",center,"\\n全部输出:\\n",filtered)
assert np.isclose(center,filtered[2,2])
show_images([a,filtered],["输入5×5","均值输出"]); save_figure("02_convolution_matrix")'''),
M('''## 2 输入真实图片，再加入可控噪声
噪声在课程中人为添加，因此有干净参考图，可以计算 MSE/PSNR。真实照片本身没有无噪声真值时，不应虚构 PSNR。'''),
C('''bgr=cv2.imread(str(ROOT/"assets/photos/fruits.jpg")); assert bgr is not None
clean=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(192,192))
rng=np.random.default_rng(42)
noisy=np.clip(clean.astype(float)+rng.normal(0,18,clean.shape),0,255).astype(np.uint8)
results={"干净参考":clean,"噪声输入":noisy,"均值":cv2.blur(noisy,(5,5)),
         "高斯":cv2.GaussianBlur(noisy,(5,5),1.2),"中值":cv2.medianBlur(noisy,5),
         "双边":cv2.bilateralFilter(noisy,7,35,35)}
show_images(results); save_figure("02_denoising_comparison")
for name,img in list(results.items())[1:]:
    mse=np.mean((img.astype(float)-clean.astype(float))**2)
    print(name,"MSE=",round(mse,2),"PSNR=",round(10*np.log10(255**2/max(mse,1e-12)),2))'''),
M('''## 3 Sobel 梯度的数值意义
保留浮点有符号梯度，不能先转 uint8 丢掉负值。幅值 sqrt(gx²+gy²) 表示变化强度，方向 atan2(gy,gx)。'''),
C('''gx=cv2.Sobel(clean,cv2.CV_32F,1,0,ksize=3)
gy=cv2.Sobel(clean,cv2.CV_32F,0,1,ksize=3)
magnitude=cv2.magnitude(gx,gy)
print("gx局部矩阵:\\n",gx[50:55,50:55],"\\n幅值:\\n",magnitude[50:55,50:55].round(1))
show_images([clean,gx,gy,magnitude],["输入","x梯度（有符号）","y梯度（有符号）","梯度幅值"])
save_figure("02_sobel")'''),
M('''## 4 Canny：模糊、梯度、非极大值抑制与双阈值连接
这里调用 OpenCV 实现，修改阈值比较保留的边缘。它输出二值边缘，不是灰度降噪结果。'''),
C('''smooth=cv2.GaussianBlur(clean,(5,5),1)
e1=cv2.Canny(smooth,30,80); e2=cv2.Canny(smooth,100,180)
show_images([clean,e1,e2],["输入","阈值30/80","阈值100/180"]); save_figure("02_canny")
print("边缘像素数:",np.count_nonzero(e1),np.count_nonzero(e2))
assert set(np.unique(e1)).issubset({0,255})'''),
M('''## 5 参数实验与常见错误
比较核 3/5/9：降噪更强是否同时损失边缘？给图片加入椒盐噪声后再比较中值和高斯。
不要把“某图PSNR更高”推广成所有场景最优；不要把 Canny 和自适应阈值当成同一种算法。
参考：[OpenCV 平滑](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)、[梯度](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html)、[Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)。''')])

write('03_morphology_contours.ipynb', [
M('''# 03 二值化、形态学、轮廓与局部特征
输入一张自制零件图，逐步得到二值掩膜、清理结果、轮廓和测量值。再用真实照片演示 ORB 特征匹配。
所有尺寸先用像素表示，没有标定不能写成毫米。'''), C(INIT),
M('''## 1 构造可追溯的零件图与矩阵
白色是前景；结构元素的形状决定邻域。腐蚀收缩前景，膨胀扩展前景。'''),
C('''mask=np.zeros((160,220),np.uint8)
cv2.rectangle(mask,(25,25),(95,125),255,-1); cv2.circle(mask,(155,80),38,255,-1)
cv2.circle(mask,(60,70),8,0,-1)
rng=np.random.default_rng(42); dirty=mask.copy(); yy=rng.integers(0,160,70); xx=rng.integers(0,220,70); dirty[yy,xx]=255
mini=(mask[20:30,20:30]>0).astype(np.uint8)
print("边界局部10×10:\\n",mini)
show_images([mask,dirty],["干净输入","噪点输入"]); save_figure("03_input")'''),
M('''## 2 形态学处理
开运算=先腐蚀后膨胀，常用于去小白点；闭运算=先膨胀后腐蚀，常用于补小黑洞。
能否去掉/补上取决于结构元素大小，并非对所有噪点和孔洞都有效。'''),
C('''k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)); print("结构元素:\\n",k)
eroded=cv2.erode(dirty,k); dilated=cv2.dilate(dirty,k)
opened=cv2.morphologyEx(dirty,cv2.MORPH_OPEN,k)
closed=cv2.morphologyEx(opened,cv2.MORPH_CLOSE,k)
show_images([dirty,eroded,dilated,opened,closed],["输入","腐蚀","膨胀","开运算","再闭运算"])
save_figure("03_morphology")'''),
M('''## 3 轮廓与测量输出
轮廓围成的几何面积与白像素个数并不严格相同。下面同时打印外接框、轮廓面积与周长。'''),
C('''contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
vis=cv2.cvtColor(closed,cv2.COLOR_GRAY2RGB)
for i,c in enumerate(contours):
    x,y,w,h=cv2.boundingRect(c); area=cv2.contourArea(c); perimeter=cv2.arcLength(c,True)
    cv2.rectangle(vis,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.putText(vis,str(i),(x,y+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,160,0),1)
    print(i, "bbox(x,y,w,h)=",(x,y,w,h),"面积像素平方=",area,"周长像素=",round(perimeter,2))
show_images([closed,vis],["清理后二值图","输出：轮廓测量"]); save_figure("03_contours")
assert len(contours)==2'''),
M('''## 4 阈值：固定与自适应
阈值分割按亮度决定前景，不是学习了语义类别；第08课再学习神经网络分割。'''),
C('''illum=np.tile(np.linspace(20,160,220),(160,1))
scene=np.clip(illum+mask*0.3,0,255).astype(np.uint8)
_,fixed=cv2.threshold(scene,125,255,cv2.THRESH_BINARY)
adaptive=cv2.adaptiveThreshold(scene,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-5)
show_images([scene,fixed,adaptive],["不均匀照明输入","固定阈值","自适应阈值"]); save_figure("03_thresholds")'''),
M('''## 5 真实照片的局部特征与匹配
ORB 检测关键点并计算二值描述子；Hamming 距离衡量不同位的个数。此例对同一图片做已知旋转，不代表跨场景一定匹配成功。'''),
C('''photo=cv2.imread(str(ROOT/"assets/photos/fruits.jpg")); assert photo is not None
photo=cv2.resize(photo,(256,256)); transform=cv2.getRotationMatrix2D((128,128),15,0.9)
warped=cv2.warpAffine(photo,transform,(256,256))
orb=cv2.ORB_create(nfeatures=350); p1,d1=orb.detectAndCompute(photo,None); p2,d2=orb.detectAndCompute(warped,None)
assert d1 is not None and d2 is not None
matches=sorted(cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(d1,d2),key=lambda m:m.distance)[:25]
print("描述子形状:",d1.shape,"前2行8列:\\n",d1[:2,:8])
matched=cv2.drawMatches(photo,p1,warped,p2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show_images([cv2.cvtColor(matched,cv2.COLOR_BGR2RGB)],["输入照片 → 旋转输出 → ORB匹配"]); save_figure("03_orb_matches")'''),
M('''## 6 练习与参考
扩大结构元素观察孔洞何时被填平；改变照明梯度比较阈值；把旋转改成大角度检验 ORB 的局限。
来源：[形态学](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)、[轮廓](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)、[特征匹配](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)。''')])

write('04_datasets_evaluation.ipynb', [
M('''# 04 数据集、任务格式、划分和评价
本课把“数据集整理”落实为可操作实验：读取真实手写数字数据，检查标签与矩阵，拆分训练/验证/测试，构造基线并观察领域变化。
数据与模型指标都要对应明确任务，不能拿分类准确率代替检测/分割质量。'''), C(INIT),
M('''## 1 输入：scikit-learn 内置手写数字
这是实际采集的 8×8 数字图，不需要联网。像素范围 0–16，标签为 0–9。
来源：[digits 数据文档](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)，UCI Optical Recognition of Handwritten Digits。'''),
C('''from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
data=load_digits(); images=data.images.astype(np.float32); labels=data.target
print("输入:",images.shape,"标签:",labels.shape,"范围:",(images.min(),images.max()))
print("第一张图矩阵:\\n",images[0]); print("标签:",labels[0])
show_images(list(images[:6]),[f"真实标签 {v}" for v in labels[:6]]); save_figure("04_digits_input")'''),
M('''## 2 分层划分，不把测试集用于选择参数
训练集拟合模型，验证集选择配置，测试集在确定模型后报告。此处演示按样本分层；医疗/同一学生/同一视频等数据需要按患者、身份或序列分组，防止泄漏。'''),
C('''indices=np.arange(len(labels))
train_val,test=train_test_split(indices,test_size=0.2,random_state=42,stratify=labels)
train,val=train_test_split(train_val,test_size=0.25,random_state=42,stratify=labels[train_val])
assert not (set(train)&set(test) or set(train)&set(val) or set(val)&set(test))
print("训练/验证/测试:",len(train),len(val),len(test))
for name,idx in [("train",train),("val",val),("test",test)]: print(name,np.bincount(labels[idx]))'''),
M('''## 3 基线：训练、验证选择与测试
用归一化像素训练逻辑回归。归一化常量 16 来自数据定义，不是用测试集估计。'''),
C('''x=images.reshape(len(images),-1)/16
candidates=[]
for c in [0.1,1.0,10.0]:
    model=LogisticRegression(C=c,max_iter=1000).fit(x[train],labels[train])
    score=accuracy_score(labels[val],model.predict(x[val])); candidates.append((score,model,c)); print("C",c,"验证accuracy",round(score,4))
best=max(candidates,key=lambda item:item[0])[1]
pred=best.predict(x[test]); accuracy=accuracy_score(labels[test],pred)
print("最终测试准确率:",round(accuracy,4))
ConfusionMatrixDisplay(confusion_matrix(labels[test],pred)).plot(cmap="Blues"); save_figure("04_confusion_matrix")'''),
M('''## 4 输出与错误分析
概率不是正确性的保证。先看误分类，再考虑增加数据或改变模型。'''),
C('''wrong=np.where(pred!=labels[test])[0]
selected=wrong[:6] if len(wrong) else np.arange(6)
show_images([images[test[i]] for i in selected],[f"真{labels[test[i]]}/预测{pred[i]}" for i in selected])
save_figure("04_predictions")
shifted=1-x[test]
print("原分布测试准确率:",accuracy)
print("合成黑白反转测试准确率:",accuracy_score(labels[test],best.predict(shifted)))
print("反转像素矩阵:\\n",shifted[0].reshape(8,8).round(2))'''),
M('''## 5 任务与指标对应

|任务|标签/输出|评价与本课程|
|---|---|---|
|分类|类别，N×C logits|Accuracy、混淆矩阵；06|
|检测|类别和边界框|IoU、匹配后的Precision/Recall；07（小样例不宣称COCO mAP）|
|分割|每像素类别|Dice、IoU；08|
|检索|查询与相关集合|Recall@K、AP；09|
|生成|图像分布/条件|样本质量与多样性；10不以小样本声称FID排名|
|描述|词序列|Token/句子准确率；11受控数据|
|超分|配对低清/高清|PSNR；12|
|深度/光流/姿态|连续值/关键点|深度误差、EPE、PCK；13/14/16|

数据清单还应记录：名称、版本、来源、许可、标注、划分、下载大小、隐私限制。本课程素材清单在 assets/SOURCES.json。
练习：增加噪声或旋转，在不改变测试集的前提下改进训练集增强，并保留前后指标。
后续模型索引：05 传统特征、06 CNN、07 检测、08 编码器解码器、10 DDPM、11 CNN+GRU；用任务选择模型，不把模型名混在一张无分类清单。''')])

print('生成完成：00–04')
