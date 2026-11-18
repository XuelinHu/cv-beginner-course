# 工业视觉入门实验室

十一本可执行中文 Jupyter 教程：从真实照片理解打光、图像处理、计数、拒判与数据留存。面向第一次使用 OpenCV 的学习者，代码可直接改参数重跑。

本项目是独立的非商业教学项目。00–09 使用公开许可素材；新增第10课按用户明确要求，加入当前笔记本 OsmoAction4 的夜间工件实拍。仅发布工件裁切及白名单元数据，不读取 `data/private`，不公开完整数据库或访问凭据。**当前七格仍待复核，不声称已训练好焊点识别器或达到工业准确率要求**。

在线目录：[工业视觉入门实验室](http://47.120.48.245:14034/)；新增实拍课：[10｜用当前相机实拍走完汇流排检测](http://47.120.48.245:14034/10_live_blade_busbar_inspection.html)。网页保留本次拍摄记录，不是实时视频。

## 最快开始

本机已经有 Conda `pyg` 环境。无需重新安装依赖，推荐直接在项目文件夹执行：

```powershell
.\run_lab.ps1
```

它会执行全部十一本笔记本，将带输出副本放到 `executed/`，把离线网页导出到 `site/`。用浏览器打开 `site/index.html` 即可阅读。`site/execution_report.json` 记录实际软件版本、各课耗时、来源和文件摘要；任何一课报错会停止。

第10课调用主项目实际传统引擎。当前课程位于 `cv-beginner-course/courses/industrial-vision-lab`，须将有访问权限的主项目克隆到课程工作区同级目录 `../../../battery-weld`：`git clone git@github.com:XuelinHu/battery-weld.git ../../../battery-weld`。来源清单分别保留采样时引擎摘要和本次已复核的重算引擎摘要；变更后需复核再更新，不能自动绕过。前十课仍可单独执行，不依赖主项目。现场重跑禁用私有参考图，只运行传统轮廓流程。

需要自己改参数时：

```powershell
conda run --no-capture-output -n pyg python -m jupyterlab
```

打开 `notebooks/00_environment_and_real_cases.ipynb`，从上到下按 `Shift + Enter`。每本都自带环境入口，可单独执行。更改源笔记本后再次运行 `run_lab.ps1` 更新网页，不必运行生成器。

在其他已准备好依赖的 Python 环境中，默认命令为：

```bash
python scripts/run_all.py
```

`requirements.txt` 列出运行依赖。SQLite、JSON、HTTP 下载均用 Python 标准库，不需要 MySQL、Node、GPU 或新模型包。首次建环境时再按自己的环境管理方式安装这些依赖；本次交付未安装任何包。

建议 Python 3.10 或更新版本；本次实际版本记录在 `site/execution_report.json`。执行器为当前 Python 创建临时内核，避免 Jupyter 默认内核意外指向其他环境；每课结束后关闭自己的内核。

本次现有环境的直接依赖精确版本另存于 [requirements-verified.txt](requirements-verified.txt)，执行与计数的真实结论见 [VALIDATION.md](VALIDATION.md)。

## 十一课内容

| 课 | 重点 | 真实案例 / 明确标记的派生演示 |
|---|---|---|
| 00 | 环境、真实照片、原标注与输出的区别 | 焊缝、工业表面、分散零件 |
| 01 | 夜间补光、暗场、曝光与像素预算 | 焊缝原图 + 派生低亮、噪声、模糊 |
| 02 | 灰度、直方图、全局均衡、CLAHE | Kolektor 工业表面 |
| 03 | 固定/自适应/Otsu、开运算与闭运算 | 工业表面及原始人工缺陷掩膜 |
| 04 | Canny、局部标准差、方向纹理 | 真实焊缝及派生噪声 |
| 05 | 单件裁切、多对象计数、当前视野编号 | 实拍六螺钉两金属球，展示误合并风险 |
| 06 | 小样本模板定位与失效条件 | 实拍零件，自匹配和派生旋转明确区分 |
| 07 | 真实遮挡、反光、画质门控和待复核 | 真实重叠螺钉盒，不编造完整数量真值 |
| 08 | 七格至少五格、缺镍片、未知、坐标映射 | 真实公开焊缝 + 明确标记的合成规则图 |
| 09 | SQLite、证据摘要、幂等、outbox、私有导入 | 公开照片留存 + 合成规则测试输入 |
| 10 | 当前相机实拍、夜间补光、二值化、汇流排切分、镍片与七格拒判、实际入库 | 2026-11-18 05:23:32 北京时间 OsmoAction4 工件区域；9个候选全部待复核 |

第10课已补充摆歪、疑似遮挡、贴边/出画处理：实际旋转轮廓、方向校正图及五组原图/旋转/黑块遮挡/裁边/全遮挡对照。原实拍为9候选；旋转23°为7候选，黑块遮挡和裁边各为8候选，全遮挡为0候选，均保持REVIEW。漏检与粗分割局限如实保留，不宣称任意遮挡都能找全。运行结果见 `outputs/public/onsite_robustness.json`。

网页保留完整代码和执行输出，参数对照选择器不依赖外网。`executed/` 可直接在 GitHub/Jupyter 查看实际图表。

## 校验与只读部署

```powershell
conda run --no-capture-output -n pyg python scripts/verify_artifacts.py
conda run --no-capture-output -n pyg python -m unittest discover -s tests -v
```

前一个命令核对素材摘要、十一本执行状态、授权实拍摘要、图表和离线链接；后一个验证素材完整性、导入隔离、证据幂等和发布白名单。

在线部署通过专用只读入口发布 `site/` 中的十六个已确认发布文件（含授权实拍课和授权清单）。Linux 服务器的可重启管理命令为：

```bash
python3 scripts/manage_site.py start
python3 scripts/manage_site.py status
python3 scripts/manage_site.py restart
python3 scripts/manage_site.py stop
```

服务不提供目录浏览、笔记本执行或上传；私有目录和源码不在公开文件白名单中。日志、PID 与受管实例信息存放在被忽略的 `runtime`，更新时保留。完整启停、首次接管和 Git 更新方式见 [DEPLOYMENT.md](DEPLOYMENT.md)。

此入口不接摄像头、数据库或 MES。项目本身不需要 `.env`；`.env`、`.env.*`、私有图、缓存、运行目录与 SQLite 文件均被 Git 忽略。不要在笔记本输出或网页中填写访问密码、令牌、私有 MES 地址。实际服务器和外网映射由主项目部署流程单独配置。

## 资料来源和许可

三类公开任务来自 CC0 焊缝、CC BY-NC-SA 4.0 的 KolektorSDD、CC BY-SA 零件照片。[DATA_SOURCES.md](DATA_SOURCES.md) 有原作者和许可依据，[sources/source_manifest.json](sources/source_manifest.json) 固定文件 SHA256。第10课现场图单独存入 `data/authorized_onsite`，依据见 [现场授权清单](sources/onsite_capture_manifest.json)，权利归原权利人，仅授权本项目教学展示，没有通用开放许可。代码采用 MIT，**不能把代码许可套在照片上**。

仓库仅保留教学小样本及标注，完整 100 MB 工业档案不提交。已有小样本时完全离线核验。素材缺失时可以恢复：

```powershell
.\run_lab.ps1 -Download
```

或 `python scripts/download_data.py`。下载器不要求账号或令牌，拒绝校验值变化；网络受限或被素材站点限流时保留错误，不会悄悄换一个来源。初始策展用的 `--initialize-lock` 仅用于本次首次捕获，日常复现不要重新生成摘要来绕过校验。

## 使用现场私有电池照片

```powershell
conda run -n pyg python scripts/import_private.py "D:\your-private-folder\battery.jpg" --expected-count 9
```

人工确认九块才能填 9，否则省略数量。只会导入被忽略的 `data/private`，记录摘要、大小和可选人工数量，七格标签仍未知。默认笔记本和公开网页不读取该目录。私有实验请另存到 `outputs/private` 或自己的私有笔记本；不要把结果写回公开教材。原图重新编码为 PNG，不保留原 EXIF。

## 目录

```text
notebooks/                可编辑的十一本源笔记本
executed/                 实际执行后的公开输出副本
site/                     离线网页、来源清单和执行记录
data/public/              公开小样本及原标注，保留各自许可
data/authorized_onsite/    用户明确授权用于本教程的工件裁切
data/cache/               完整原档案缓存（Git 忽略）
data/private/             显式导入的私有图片（Git 忽略）
outputs/public/           教学结果 JSON 与本地 SQLite（数据库不提交）
outputs/private/          私有实验输出（Git 忽略）
sources/source_manifest.json   公开素材的固定摘要与来源
sources/onsite_capture_manifest.json   现场图授权范围、摘要与白名单结果
scripts/download_data.py      按固定摘要下载和校验
scripts/run_all.py            当前 Python 内核执行、离线网页导出
scripts/import_private.py     仅本地私有导入
scripts/create_notebooks.py   教材源生成器，日常学习无需运行
scripts/capture_onsite_case.py 显式授权后经现有相机API取帧并核验实际入库
scripts/create_onsite_lesson.py 仅生成新增实拍课，不改前十课
scripts/serve_site.py         十六个已审核发布文件的只读服务
scripts/manage_site.py        Linux 启停、重启、状态与进程归属核对
```

## 从教材走向实际检测

先固定相机、距离和照明，明确最小焊纹像素。然后按当前视野检测汇流排、切镍片窗口、对齐七格；清晰外观至少五格才按规则绿，确认不足五格红，看不清黄。N×M 只作配方预期，不能充当实测计数。二维视觉不能单独证明内部熔合或机械强度。

需要工业验收时先冻结新物理来源的真值和计分口径，分别统计误放行、漏检、误检、待复核与延迟。旋转、亮度和背景增强只能作挑战，不能增加独立样本数。本课模板自匹配和八个合成规则检查都不是产线识别准确率。
