#!/usr/bin/env bash
set -e
echo "[MolPropLab] 正在设置项目..."

# 检查是否在 Conda 环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到 Conda 环境"
    echo "请先创建并激活 Conda 环境:"
    echo "  conda create -n molproplab python=3.11 -y"
    echo "  conda activate molproplab"
    echo ""
    echo "是否继续？(y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 RDKit 是否已安装
echo "检查 RDKit..."
if ! python -c "import rdkit" 2>/dev/null; then
    echo "RDKit 未安装，正在通过 Conda 安装..."
    if command -v conda &> /dev/null; then
        conda install -c conda-forge rdkit -y
    else
        echo "错误: 未找到 conda 命令。请先安装 Anaconda 或 Miniconda"
        exit 1
    fi
else
    echo "✓ RDKit 已安装"
fi

# 安装 Python 依赖
echo "安装 Python 依赖..."
cd "$(dirname "$0")/../ml"
pip install -r ../requirements.txt

# 安装 Node.js 依赖
echo "安装后端依赖..."
cd ../server
npm install

echo "安装前端依赖..."
cd ../frontend
npm install

echo ""
echo "✓ 设置完成！"
echo ""
echo "下一步："
echo "1. 确保 Conda 环境已激活: conda activate molproplab"
echo "2. 启动后端: cd server && npm run dev"
echo "3. 启动前端: cd frontend && npm run dev"
