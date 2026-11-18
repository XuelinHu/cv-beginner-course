# 局域网 JupyterLab 部署

课程只通过 JupyterLab 使用。当前入口：

- 局域网：<https://192.168.1.9:4034/lab>
- FRP 远程：<https://47.120.48.245:14034/lab>

访问必须携带 Jupyter token。token 保存在服务器权限为 600 的 `/home/xuelin/.config/cv-beginner-course/jupyter-token`，不写入本文件、Notebook、Git 或日志。证书为本项目自签名证书，首次浏览器访问会提示证书未由公开 CA 签发；确认地址后继续访问。

服务器上的项目、Notebook 输出、完整训练数据（`courses/industrial-vision-lab/data/dataset/`）和 MySQL 均保存在同一局域网节点。GitHub/Gitee 只同步项目代码、公开教学素材和执行后的教学 Notebook，不包含完整数据集、数据库密码或现场私有图像。

数据库由 MySQL 保存运行记录；迁移前 SQLite 文件仅保留为行级校验来源。服务器管理员可通过用户级 `cv-beginner-course-jupyter.service` 检查和重启 JupyterLab。外网入口由 FRP 管理，不在本仓库保存 FRP 凭据。
