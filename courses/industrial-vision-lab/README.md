# 刀片电池焊接检测专题

本专题是主课程的一部分，不提供独立网页或第二套 Notebook 入口。唯一课程目录为项目根目录 `notebooks/`：第 20–27 节讲解暗光成像、汇流排定位、镍片存在性、七格判断、拒判和证据记录；第 19 课用于区分焊缝、钢铁表面和汇流排标签语义；从项目根目录启动 JupyterLab。

```powershell
$env:JUPYTER_CONFIG_DIR = (Join-Path (Get-Location) ".jupyter")
conda run --no-capture-output -n pyg python -m jupyterlab --custom-css notebooks
```

局域网服务器部署使用同一套 JupyterLab，MySQL 仅由专题数据库课程及配套 Python 模块访问。完整教学数据保存在忽略 Git 的 `courses/industrial-vision-lab/data/dataset/`，不推送到 GitHub 或 Gitee，也不进入学生课程包。SQLite 文件仅作为 MySQL 迁移校验来源，线上运行数据库为 MySQL。

本目录只保存专题代码、公开教学样例、数据来源与许可、测试及运行结果。当前七格仍待复核；课程和传统视觉演示不代表已完成工业级模型验收。

公开数据来源见 [DATA_SOURCES.md](DATA_SOURCES.md)，原始来源与 SHA256 见 `sources/`。现场采集只使用经授权的样例裁切；完整私有照片和完整训练数据集不得放入公开仓库。完整课程的执行记录见根目录 [VERIFICATION.md](../../docs/VERIFICATION.md)。
