#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
TSINGHUA_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

# 根据平台选择 python 命令
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) PYTHON="python"  ;;
    *)                     PYTHON="python3" ;;
esac

if [ ! -d "$VENV_DIR" ]; then
    echo "未检测到 .venv，正在创建虚拟环境..."
    "$PYTHON" -m venv --system-site-packages --copies "$VENV_DIR"
    echo "虚拟环境创建完成。"
else
    echo "检测到已有 .venv，跳过创建。"
fi

# 根据平台确定 pip 路径和 pip.conf 位置
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        PIP="$VENV_DIR/Scripts/pip"
        PIP_CONF="$VENV_DIR/pip.ini"
        ;;
    *)
        PIP="$VENV_DIR/bin/pip"
        PIP_CONF="$VENV_DIR/pip.conf"
        ;;
esac

# 配置清华 pip 源
cat > "$PIP_CONF" <<EOF
[global]
index-url = $TSINGHUA_INDEX
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
echo "已配置 pip 清华源。"

# 安装依赖
if [ -f "$REQUIREMENTS" ]; then
    echo "正在安装 requirements.txt 中的依赖..."
    "$PIP" install --upgrade pip
    "$PIP" install -r "$REQUIREMENTS"
    echo "依赖安装完成。"
else
    echo "未找到 requirements.txt，跳过依赖安装。"
fi

echo "环境准备就绪，激活命令：source $VENV_DIR/bin/activate"
