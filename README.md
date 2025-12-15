# 环境
    创建一个虚拟环境  ： python -m venv huggingface_env
    在Windows上激活虚拟环境  ： huggingface_env
    安装依赖包  ： pip install -r requirements.txt
    导出依赖包  ： pip freeze > requirements.txt


pip install streamlit pandas plotly huggingface_hub
pip install matplotlib seaborn  networkx
pip install scikit-learn

pip install wordcloud pillow numpy

pip install pyvis


powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
reopen cmd
set Path=C:\Users\youzi\.local\bin;%Path%
uv --version
uv python list

# 安装Python时使用GitHub加速镜像
uv python install 3.13 --mirror "https://gitproxy.click/https://github.com/indygreg/python-build-standalone/releases/download"

# 安装依赖时使用清华源
uv add pandas --index "https://pypi.tuna.tsinghua.edu.cn/simple"

uv python dir       # 查看Python安装目录
uv cache dir        # 查看缓存目录
uv tool dir         # 查看工具目录
uv self update
2. 卸载uv
Windows：删除安装目录（如D:\Apps\uv）和环境变量中相关条目。
macOS/Linux：删除~/.local/bin/uv及配置文件~/.config/uv。


# 初始化默认项目（生成 pyproject.toml 和 .venv）
uv init myproject
cd myproject

# 指定 Python 版本初始化（如 3.13）
uv init myproject --python 3.13

# 初始化可打包应用（如 CLI 工具）
uv init --app --package mycli  # 生成可发布的应用结构

# 默认创建 .venv 目录（自动关联项目 Python 版本）
uv venv

# 指定 Python 版本创建
uv venv --python 3.11  # 使用已安装的 3.11 版本

# 自定义虚拟环境路径
uv venv /path/to/custom/venv

# 安装生产依赖（自动更新 pyproject.toml 和 uv.lock）
uv add requests  # 单个包
uv add numpy pandas  # 多个包
uv add "django>=4.2,<5.0"  # 指定版本范围

# 添加开发依赖（仅开发环境使用）
uv add --dev pytest black  # 开发依赖会写入 [project.optional-dependencies.dev]

# 从 requirements.txt 导入依赖
uv add -r requirements.txt

uv remove requests  # 移除生产依赖
uv remove --dev pytest  # 移除开发依赖
uv remove numpy pandas  # 批量移除

# 生成/更新锁文件（锁定精确版本）
uv lock

# 升级指定依赖（如 requests 到最新兼容版本）
uv lock --upgrade-package requests

# 根据锁文件安装依赖（适合多人协作或部署）
uv sync  # 等价于 "uv install"，优先使用缓存加速

uv tree  # 显示所有依赖及其层级
uv tree --outdated  # 标记可升级的依赖
uv tree --depth 2  # 限制显示深度（避免依赖树过大）

uv python list  # 列出已安装版本
uv python uninstall 3.10  # 卸载指定版本
uv python upgrade  # 升级当前版本到最新补丁版

# 运行项目内脚本（自动使用虚拟环境）
uv run main.py

# 临时指定 Python 版本运行
uv run --python 3.13 script.py

# 运行带依赖的独立脚本（无需项目配置）
uv run --with requests script.py  # 自动安装 requests 并运行

# 安装全局 CLI 工具（如代码格式化工具 ruff）
uv tool install ruff

# 运行已安装工具
uv tool run ruff format myscript.py

# 临时运行工具（无需安装，用完即删）
uvx pycowsay "Hello"  # uvx 是 "uv tool run" 的简写

uv cache dir  # 查看缓存路径（默认 ~/.cache/uv）
uv cache clean  # 清空所有缓存
uv cache purge  # 仅删除未使用的缓存

# 导出依赖到 requirements.txt（兼容 pip）
uv pip freeze > requirements.txt

# 从 requirements.in 编译为锁定文件（类似 pip-tools）
uv pip compile requirements.in --output-file requirements.txt