# 统一课程验证

专题 22–32 已合并到主课程唯一的 `notebooks/` 目录。课程整体执行报告和 33 节 Notebook 的结构、图像输出检查见 [统一课程验收记录](../../docs/VERIFICATION.md)。专题单元测试在仓库根目录执行：

```powershell
conda run --no-capture-output -n pyg python -m unittest discover -s courses/industrial-vision-lab/tests -v
```

数据完整性由 `scripts/audit_project_data.py` 对照完整文件清单及 SHA256 检查；SQLite 到局域网 MySQL 的逐表逐行迁移校验由 `scripts/verify_mysql_migration.py` 执行。完整数据留在局域网服务器，不上传到 Git 远端。
