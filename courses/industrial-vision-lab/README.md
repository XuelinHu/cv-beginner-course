# 刀片电池焊接检测专题

本专题是主课程的一部分，不提供独立网页或第二套 Notebook 入口。唯一课程目录为项目根目录 `notebooks/`：第 22–32 节讲解暗光成像、汇流排定位、镍片存在性、七格判断、拒判和证据记录；从项目根目录启动 JupyterLab。

```powershell
conda run --no-capture-output -n pyg python -m jupyterlab notebooks
```

局域网服务器部署使用同一套 JupyterLab，MySQL 仅由专题数据库课程及配套 Python 模块访问。开发/教学数据完整保存在服务器 `courses/industrial-vision-lab/data/legacy_import/`，该目录被 Git 忽略，不推送到 GitHub 或 Gitee，也不进入学生课程包。SQLite 历史数据仅作为迁移校验来源，不是线上运行数据库。

本目录只保存专题代码、公开教学样例、数据来源与许可、测试及运行结果。当前七格仍待复核；课程和传统视觉演示不代表已完成工业级模型验收。

公开数据来源见 [DATA_SOURCES.md](DATA_SOURCES.md)，原始来源与 SHA256 见 `sources/`。现场采集只使用经授权的样例裁切；完整私有照片和完整训练数据集不得放入公开仓库。完整课程的执行记录见根目录 [VERIFICATION.md](../../docs/VERIFICATION.md)。
