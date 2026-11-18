# 公开素材、许可与证据

核验日期：2026-11-17。代码的 MIT 许可不覆盖下面的照片和标注；它们各自保留原许可。教材用于非商业学习，含非商业素材的页面也受相应限制。没有用私人电池照片替换公开案例。

| 教学案例 | 原始来源及作者 | 素材许可 | 本课用途与限制 |
|---|---|---|---|
| 焊缝外观 | [sukmaadhiwijaya / Kaggle](https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection)，[固定版分发及数据卡](https://huggingface.co/datasets/rikkarth/welding-defect-object-detection/tree/8b80d1afe88c0b946500a498cbdb9d7a1eda5533) | 数据卡声明 [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/)，标明原 Kaggle 来源 | 只取三个实拍焊缝及其原 YOLO 标注。类别 Bad Weld / Good Weld / Defect 是来源语义，不是本项目七格电池标准；照片采集过程和独立性未验证。 |
| 工业金属表面 | [KolektorSDD 官方页](https://www.vicos.si/resources/kolektorsdd/)，Kolektor Group d.o.o. / ViCoS | 官方明确 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)；商业用途需联系作者 | 六个原始表面样本及掩膜，来自真实受控工业场景。仅作非商业教学，不把六图演示写成产线准确率。原档案由官方链接下载，选样保持原字节。 |
| 分散零件计数 | [Screws.png 原文件页](https://commons.wikimedia.org/w/index.php?title=File:Screws.png&oldid=504053334)，Tung1213，2012 | 作者明确 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | 真实照片，原说明为六颗螺钉、两颗金属球。它是零件实拍案例，不冒称工业基准测试集。框、灰度、二值图等改编保留相同许可及署名。 |
| 重叠零件挑战 | [Schrauben.jpg 原文件页](https://commons.wikimedia.org/w/index.php?title=File:Schrauben.jpg&oldid=927807040)，Technologyfelix，2023 | 作者明确 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 真实螺钉盒照片，用来展示遮挡导致计数不可靠；不虚构完整数量真值。相关改编保留相同许可及署名。 |

精确下载 URL、固定提交、原始档案成员路径、文件大小和 SHA256 在 [sources/source_manifest.json](sources/source_manifest.json)。`data/cache` 不提交完整数据集。`data/public` 只保存上述小样本及原标注；默认离线运行，不需要账号、令牌或摄像头。

## 本次新增的真实公开输入

`data/public/augmentation/` 中的六张图片由用户提供，记录了视频课件截图中的增强方法，包含原视频水印、字幕和批注。其著作权和二次分发许可尚未核实；课程代码重新实现其中的方法，截图只用于所提供课程项目内的学习引用。对外再分发前应核实权利人和许可。

|素材|来源|用途与权利说明|
|---|---|---|
|`data/public/medical/CT_small.dcm`|[pydicom 官方 DICOM 示例](https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html)，固定文件来自 [pydicom 测试数据](https://github.com/pydicom/pydicom/blob/7cda352958d0d9193b19dfa417b126dcf3286943/src/pydicom/data/test_files/CT_small.dcm)|真实 DICOM 像素数据，用于 HU 和窗宽窗位教学，不用于诊断。SHA256：`3dd31e5cc835b3f2cdd46c9da1982f59251e78518fefa8163d914631c66437d6`。|
|`data/public/underwater/euvp_test_input_1.jpg`、`euvp_test_reference_1.jpg`|[EUVP 官方数据集页](https://irvlab.cs.umn.edu/resources/euvp-dataset)；图像按固定版本从 [FUnIE-GAN 测试样例](https://github.com/xahidbuffon/FUnIE-GAN/tree/8f934c834c94e007b00866186b9ee624dc2b7b69/data/test)获取|真实水下输入及配对参考，不是生成图，也不是逐像素物理真值。图片权利遵循 EUVP 原数据集条款；FUnIE-GAN 代码许可不自动覆盖图片。SHA256 分别为 `8eae88a6349db0fed9d25355aaa9a93afb0dbc4cc6c97e7a318c4713bbc41870`、`94653e5b9046b25115441865b651b291cec28fcf7ab51316a828b2214b94d2c4`。|

## 派生图的标记

课程中的低亮度、噪声、旋转、模糊和阈值图，均是公开实拍照片的**派生演示**，不是新增独立照片，也不是实测灯光或光谱实验。第 08 课的七格条纹是明确标注的**合成规则示意**，仅验证软件逻辑，不替代任何真实焊格标签。图片裁切、叠框、灰度、阈值等修改会在课中注明。

Kolektor 派生图按 CC BY-NC-SA 4.0；分散零件派生图按 CC BY-SA 3.0；遮挡零件派生图按 CC BY-SA 4.0。混合包含这些图片的静态教程只用于非商业教学。转载时请保留每页来源说明和本文件。

## 参考文献与官方教程

KolektorSDD：Domen Tabernik, Samo Šela, Jure Skvarč, Danijel Skočaj. *Segmentation-Based Deep-Learning Approach for Surface-Defect Detection*. Journal of Intelligent Manufacturing, 2019. [doi:10.1007/s10845-019-01476-x](https://doi.org/10.1007/s10845-019-01476-x)。本课只用其图像和原始标注，不复制论文方法或声称复现其成绩。

- [Basler：照明方法、反光表面与偏振的局限](https://www.baslerweb.com/en/learning/illumination/)
- [Basler：暗场照明](https://www.baslerweb.com/en/illumination/dark-field-light/)
- [OpenCV：直方图均衡与 CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- [OpenCV：固定、自适应、Otsu 阈值](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [OpenCV：形态学](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [OpenCV：Canny 边缘](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [OpenCV：模板匹配](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
- [OpenCV：连通域与形状分析](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)

## 私有电池照片

只能通过 `scripts/import_private.py` 导入 `data/private`，该目录被 Git 忽略。默认笔记本、公开清单和网页不读取该目录。不要把现场照片、工厂名称、MES 地址、人员信息或访问凭据粘进公开笔记本输出。私有图片的本地实验请另存到 `outputs/private`，不运行公开导出流程。

## 用户授权的当前相机实拍（第10课）

用户于 2026-11-18 明确要求将当前实时拍摄的场景制作成同样的在线课程。为此，从其当前 Windows 笔记本连接的 OsmoAction4，经已有检测服务保存了一帧，拍摄时间为北京时间 2026-11-18 05:23:32（UTC 2026-11-17T21:23:32.356460+00:00）。仅导出九个候选工件的联合区域，847×700 像素，无损 PNG；没有导入其他工厂照片。

来源、原始 BGR8 摘要、PNG 摘要、裁切坐标、授权范围、灯光请求/设备回读、实际检测结果及入库核验见 [现场授权清单](sources/onsite_capture_manifest.json)。原始全画面与完整数据库记录不进入教材。`data/authorized_onsite` 是这次明确授权的单独入口；`data/private` 仍被忽略且不参与导出。

该现场照片及其灰度、二值、纹理、裁切、叠框派生图权利仍归原权利人，**仅获准用于本项目教程展示，没有 CC0、CC BY 或通用开源数据许可**。代码 MIT 许可不适用于这张照片。GitHub 仓库保持私有；在线教程按用户要求展示此限定样本。

补光条件来自操作者说明，不是光度/光谱测量。当前九个候选、九个镍片未知、全部七格待复核是系统输出，不是人工标签或准确率结论。第10课调用锁定版本主项目的传统引擎，不打开其私有参考图。`private_data_read:false` 表示未读忽略的私有目录；执行记录另以 `authorized_onsite_data_included:true` 明示本次授权现场数据。

新增的旋转23°、黑块遮挡、右侧裁边和全遮挡均为这同一帧的派生挑战，不是新实拍或独立验收样本，沿用该图的限定教学授权。采样时与重算时引擎摘要分开保存，原采样结论保持原样。几何可见性只提供疑似遮挡线索，不能证明已发现完全隐藏工件或精确测量隐藏表面积。

