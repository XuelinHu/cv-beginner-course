# 教程静态服务管理

主仓库为 `XuelinHu/cv-beginner-course`，本课程位于 `/ds1/workspace/CV/cv-beginner-course/courses/industrial-vision-lab`。运行入口仅使用 Linux 自带 Python 3 标准库，不安装 Jupyter、OpenCV 或模型环境，不依赖主检测项目的虚拟环境。

## 日常命令

在服务器执行：

    cd /ds1/workspace/CV/cv-beginner-course/courses/industrial-vision-lab
    python3 scripts/manage_site.py start
    python3 scripts/manage_site.py status
    python3 scripts/manage_site.py restart
    python3 scripts/manage_site.py stop

默认监听 0.0.0.0:4034。重复 start 会返回现有受管进程，不会多开。restart 先等待自己的旧进程退出，再开新进程；如果旧进程未退出或 PID 已属于别的程序，会保留状态并报错，不会停止其他项目。

后台进程脱离 SSH 会话，日志追加到 runtime/site.log，当前 PID 写入 runtime/site.pid，进程身份与状态保存在 runtime/site-service.json。runtime 被 Git 忽略，更新源码保留这些文件。系统重启后执行 start 恢复服务；本次没有修改系统级开机配置。

首次从旧的 http.server 切换时，才使用下面的显式接管命令：

    python3 scripts/manage_site.py restart --adopt-legacy

它只接受 runtime/site.pid 所指向、端口相同、--directory 精确等于本项目 site 目录的旧 http.server。其他进程或端口占用会拒绝接管。

## 访问与验证

当前网络映射由既有 FRP 配置提供，本项目不保存 FRP 凭据：

- 局域网：http://192.168.1.9:4034/
- 公网：http://47.120.48.245:14034/
- 健康检查：在上述地址后加 /healthz
- 实际课程执行记录：在上述地址后加 /execution_report.json

healthz 返回进程、源码版本、实例标识和公开执行记录摘要；它不是登录凭证，不允许控制服务。

    curl --fail http://127.0.0.1:4034/healthz
    curl --fail http://127.0.0.1:4034/execution_report.json

运行层仅接受 GET 和 HEAD。只允许课程索引、十一课 HTML、来源说明页、公开来源清单、现场授权清单与执行记录这十六个已审核文件。新增现场课必须有明确授权、单独来源清单及匹配摘要，服务启动时验证。其他文件、目录列表、私图、笔记本、源码和 .env 返回 404；POST/PUT/PATCH/DELETE 等返回 405。没有执行代码、上传文件或连接 MES 的接口。

## Git 接入和后续更新

服务器原先是已部署的文件目录。首次接入使用 git init/fetch/read-tree/update-ref 更新 Git 元数据，没有 checkout 覆盖现有运行文件。修改部署文件时只同步本次明确变更的源码和文档；公开教学图片及执行产物保持原摘要。

后续先检查工作区，再进行快进更新：

    cd /ds1/workspace/CV/cv-beginner-course
    git status --short
    git pull --ff-only
    python3 courses/industrial-vision-lab/scripts/manage_site.py restart
    python3 courses/industrial-vision-lab/scripts/manage_site.py status

若工作区存在用户修改，先审阅并保存；不要使用 reset --hard 或强制覆盖。GitHub 使用服务器已有 SSH 授权，不需要复制本机令牌或提交密钥。runtime、私有图片、SQLite、缓存和 .env 均不进入 Git。

## 本地验证

Windows 现有 pyg 环境：

    conda run --no-capture-output -n pyg python -m unittest discover -s courses/industrial-vision-lab/tests -v
    conda run --no-capture-output -n pyg python courses/industrial-vision-lab/scripts/verify_artifacts.py

静态服务测试只依赖标准库，服务器也可单独执行：

    python3 -m unittest discover -s courses/industrial-vision-lab/tests -p test_static_site.py -v

Windows 如需临时前台浏览已生成的教材：

    conda run --no-capture-output -n pyg python courses/industrial-vision-lab/scripts/serve_site.py --bind 127.0.0.1 --port 4034

管理脚本依赖 Linux /proc 做进程归属核对；不会在 Windows 猜测 PID 后结束进程。
