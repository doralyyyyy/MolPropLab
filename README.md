# MolPropLab — 分子性质预测平台

一个用于从 SMILES 预测分子性质的端到端全栈机器学习应用。

## 1. 核心特性

- **多性质预测：** 支持同时预测 10 种分子性质（分子量、LogP、LogS、pKa、沸点、熔点、折射率、蒸气压、密度、闪点）
- **Python 端 ML：** RDKit 预处理、分子描述符、ECFP（1024）、PyTorch Geometric PotentialNet/GIN、LightGBM/RandomForest 基线模型  
- **不确定性估计：** MC-Dropout（GNN）/ 小型集成（Baseline）  
- **可解释性：** SHAP（基线模型）、梯度显著性（GNN）→ 原子级重要性 JSON  
- **后端：** Node.js + Express + TypeScript；通过 `child_process.spawn` 调用 Python  
- **前端：** React + Vite + Tailwind + shadcn 风格组件 + 3Dmol.js  
- **批处理任务：** 简单内存队列 & CSV/XLSX 上传，支持模型选择

输入 SMILES，一次性获得所有性质的预测值、不确定性以及原子重要性热力图。

---

## 2. 环境要求

> 注：为实验者所用系统环境

### 系统环境

- **操作系统**：Windows 11
- **CPU**：64 位架构，Intel(R)Core(TM)i7-14650HX 处理器，16 核
- **内存**：16 GB
- **GPU**：支持 CUDA 的 NVIDIA GPU
  - **GPU 型号**：NVIDIA GeForce RTX4050 Laptop GPU
  - **CUDA 版本**：12.7
  - **驱动版本**：566.24

### Python 环境

- **Python 版本**：3.11.14
- **包管理**：推荐使用 Conda 来安装 RDKit（Windows 上不推荐直接用 pip）
- **主要依赖库版本**：见 `requirements.txt`

### Node.js 环境

- **Node.js 版本**：建议 20.19.5
  - 前往 [Node.js 官网](https://nodejs.org/) 下载并安装 **Node.js 20.19.5**
  - 或直接下载压缩包：[Node.js v20.19.5](https://nodejs.org/dist/v20.19.5/)，解压后将 `node.exe` 所在文件夹的路径添加到系统环境变量中

---

## 3. 安装步骤

这里以 **Conda + Windows** 举例，macOS / Linux 同理，只是命令行路径略有差别。

### 快速安装（推荐，适用于 Linux/macOS）

如果你在 Linux 或 macOS 系统上，可以使用一键安装脚本：

```bash
# 1. 创建并激活 Conda 环境
conda create -n molproplab python=3.11 -y
conda activate molproplab

# 2. 运行安装脚本（会自动安装所有依赖）
bash scripts/setup.sh
```

脚本会自动完成以下操作：
- 检查 Conda 环境
- 安装 RDKit（如果缺失）
- 安装所有 Python 依赖
- 安装 Node.js 后端和前端依赖
- 下载模型权重（如果可用）

> **注意**：Windows 用户可以使用 Git Bash 或 WSL 运行此脚本，或按照下面的手动安装步骤操作。

---

### 手动安装步骤

#### 1. 创建 Conda 环境

```bash
# 创建并激活环境（Python 3.11）
conda create -n molproplab python=3.11 -y
conda activate molproplab
```

#### 2. 安装 RDKit

```bash
# 安装 RDKit（推荐用 conda-forge）
conda install -c conda-forge rdkit -y
```

> **注意**：在 Windows 上不推荐直接用 `pip install rdkit`，使用 Conda 更稳定。

#### 3. 安装其余 Python 依赖

```bash
# 在项目根目录执行
pip install -r requirements.txt
```

> **重要提示**：
> - 当前使用 **NumPy 1.x**（<2.0），以确保在 Windows 上的稳定性
> - **PyTorch**：默认会安装 CPU 版本，如需 GPU 支持，可单独安装 GPU 版本：
>   ```bash
>   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
>   ```
> - 如果安装 PyTorch Geometric 有困难，可以先忽略；GNN 会被**自动降级为基线模型**，不影响整体运行

#### 4. 安装 Node.js 依赖

```bash
# 安装后端依赖
cd server
npm install

# 安装前端依赖
cd ../frontend
npm install
```

---

## 4. 启动服务

建议保持 **Python 环境处于已激活状态**（`conda activate molproplab`）。

### 启动后端 API

在项目根目录：

```bash
cd server
npm run dev
```

后端默认运行在：`http://localhost:3001`

> **注意**：`server/package.json` 中的 `dev` 脚本已配置 Python 路径。如果使用不同的 Conda 环境路径，可以：
> - 修改 `server/package.json` 中的 `PYTHON` 环境变量
> - 或在启动前设置环境变量：`set PYTHON=你的python路径 && npm run dev`

### 启动前端

另开一个终端（前端对 Python 环境没有强依赖，可以在普通终端运行）：

```bash
cd frontend
npm run dev
```

前端默认运行在：`http://localhost:5173`

---

## 5. 使用流程

### Web 界面使用

1. 浏览器打开 `http://localhost:5173`
2. 打开 **Single Prediction** 页面
3. 输入一个 SMILES，例如：
   - `CCO`（乙醇）
   - `CC(=O)O`（乙酸）
   - `c1ccccc1`（苯）
4. 选择模型类型（`baseline` 或 `gnn`）
5. 点击 **Predict**
6. 你会看到：
   - **所有性质的预测结果表格**，包括：
     - 分子量 (MW)
     - LogP（脂溶性）
     - LogS（水溶解度）
     - pKa
     - 沸点
     - 熔点
     - 折射率
     - 蒸气压
     - 密度
     - 闪点
   - 每个性质都显示**预测值**和**不确定性 (σ)**
   - 右侧 **3D 分子视图**，带原子级别热力图（重要性越高颜色越亮）

### 批处理预测

1. 进入 **Batch Prediction** 页面
2. 选择模型类型（`baseline` 或 `gnn`）
3. 上传包含 `smiles` 列的 CSV 或 XLSX 文件（可参考 `ml/data/logp.csv`）
4. 后端自动排队处理
5. 可轮询任务状态
6. 完成后下载结果文件（包含所有性质的预测列，格式：`{property}_prediction` 和 `{property}_uncertainty`）

---

## 6. Python CLI

### 训练模型

```bash
cd ml

# 训练所有性质的基线模型
python train_baseline.py

# 训练所有性质的 GNN 模型（需要 PyTorch 和 PyTorch Geometric）
python train_gnn.py
```

> **注意**：训练脚本会自动为所有 10 种性质训练模型。首次使用建议先运行 `python train_baseline.py` 来训练所有性质的基线模型，这样预测时才能显示所有性质的结果。

### 评估模型

```bash
# 评估基线模型（计算 RMSE、MAE、R2、MAPE 等多种指标）
python eval_baseline.py

# 评估 GNN 模型（计算 RMSE、MAE、R2、MAPE 等多种指标）
python eval_gnn.py
```

> **注意**：评估脚本使用 scaffold_split 进行数据集划分，与训练时保持一致，确保评估结果的可靠性。

### 模型比较

```bash
# 比较单个性质的 Baseline 和 GNN 模型
python compare_models.py --property logp

# 比较所有性质的模型表现
python compare_models.py
```

比较脚本会生成详细的性能报告，包括：
- RMSE、MAE、R2、MAPE、相关系数等指标
- 哪个模型在哪个性质上表现更好

### 推理

```bash
# 单条预测（返回所有性质的预测结果）
python inference.py --smiles "CCO" --model baseline --json
python inference.py --smiles "CCO" --model gnn --json

# 批量预测（支持CSV和XLSX格式）
python inference.py --csv data/logp.csv --output out.csv --model baseline
python inference.py --csv data/logp.csv --output out.csv --model gnn
```

> **注意**：单条预测会返回所有 10 种性质的预测结果。批量预测的输出文件会包含所有性质的列（格式：`{property}_prediction` 和 `{property}_uncertainty`）。

---

## 7. 常见问题与故障排除

### 1. RDKit 安装失败

**问题**：无法安装 RDKit 或导入失败

**解决方案**：
- 确保使用的是 **Conda 环境**
- 使用：`conda install -c conda-forge rdkit`
- 不推荐在 Windows 上直接用 `pip install rdkit`

### 2. PyTorch DLL 加载错误（Windows）

**问题**：遇到类似 `OSError: [WinError 127] Error loading "shm.dll"` 或 `[WinError 182] 操作系统无法运行 %1` 的错误

**解决方案**：
1. 确保已安装 **Visual C++ Redistributable**
   - 下载地址：https://aka.ms/vs/17/release/vc_redist.x64.exe
2. 当前 `requirements.txt` 已指定 PyTorch >= 2.0
3. 如果问题仍然存在，可以尝试重新安装：
   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install "numpy<2"
   ```
4. 代码中已自动设置 `TORCH_SHM_DISABLE=1` 环境变量，帮助避免共享内存相关问题

### 3. GNN 预测自动回退到 Baseline

**问题**：选择 GNN 模型，但预测结果中 `model` 字段显示为 `baseline`

**原因**：
- PyTorch 或 PyTorch Geometric 未正确安装
- Windows 上 DLL 加载失败

**诊断方法**：
- 检查服务器日志中的错误信息
- 查看 Python 脚本的 stderr 输出

**解决方案**：
- 参考"PyTorch DLL 加载错误"部分
- 确保已安装 Visual C++ Redistributable
- 检查 Python 环境是否正确激活

### 4. API 端口冲突

**问题**：端口已被占用

**解决方案**：
- 后端：修改 `server/src/index.ts` 中的 `PORT` 变量
- 前端：修改 `frontend/vite.config.ts` 中的 `server.port`
- 默认端口：
  - 后端：`http://localhost:3001`
  - 前端：`http://localhost:5173`

### 5. Python 路径配置

**问题**：服务器无法找到 Python 解释器

**解决方案**：
- 在启动前设置环境变量：`set PYTHON=你的python路径 && npm run dev`
- 默认逻辑：优先使用 `PYTHON` 环境变量 → 然后使用 `CONDA_PREFIX/python.exe` → 最后使用系统 `python`

---

## 8. 模型说明

### 数据预处理

系统在训练和评估前会自动进行数据预处理：

1. **缺失值处理**：自动移除 SMILES 或目标值为空的样本
2. **异常值处理**：使用 IQR（四分位距）方法检测并移除异常值
   - 移除超出 Q1-1.5×IQR 到 Q3+1.5×IQR 范围的值
   - 仅当样本数 > 10 时进行异常值检测，避免小数据集过度过滤
3. **特征归一化**：
   - Baseline 模型：对输入特征（描述符 + ECFP）进行 Z-score 归一化
   - GNN 模型：对目标值进行标准化（使用训练集的均值和标准差）

### 数据集划分

- **方法**：使用 scaffold_split（基于分子骨架的划分，按样本数分配）
  - 确保相同骨架的分子在同一集合中，避免数据泄露
  - 对大聚类进行拆分，使划分比例更接近目标比例
  - 默认比例：训练集 70%，验证集 15%，测试集 15%
  - 训练时：使用训练集（70%）进行训练，验证集（15%）用于早停
  - 评估时：使用测试集（15%）进行评估

### 多性质预测

系统支持同时预测以下 10 种分子性质：

1. **分子量 (MW)**：g/mol 单位
2. **LogP（脂溶性）**：衡量分子的亲脂性
3. **LogS（水溶解度）**：log(mol/L) 单位
4. **pKa**：酸解离常数
5. **沸点**：°C 单位
6. **熔点**：°C 单位
7. **折射率**：无量纲
8. **蒸气压**：Pa 单位
9. **密度**：g/cm³ 单位
10. **闪点**：°C 单位

每个性质都有独立的模型，训练时会为每个性质分别训练基线模型和 GNN 模型。预测时会同时调用所有性质的模型，返回完整的预测结果。

### Baseline 模型

- **算法**：LightGBM（优先）或 RandomForest（降级）
- **特征**：分子描述符 + ECFP（1024 位）
  - 特征归一化：使用 Z-score 归一化（基于训练集的均值和标准差）
- **不确定性**：集成模型的标准差（3 个模型的集成）
- **可解释性**：SHAP 值映射到原子重要性
- **过拟合防止**：
  - 根据数据集大小动态调整模型参数（树深度、叶子节点最小样本数等）
  - 使用 subsample 和 colsample_bytree 进行特征和样本采样
- **适用场景**：快速预测、小到中等数据集、无需 GPU
- **数据要求**：每个性质至少需要 20+ 样本，推荐 100+ 样本以获得更好性能

### GNN 模型

- **算法**：PotentialNet（默认）或 GIN（可配置）
- **特征**：图神经网络，直接学习分子图结构
  - 目标值标准化：使用训练集的均值和标准差进行标准化
- **不确定性**：MC-Dropout（Monte Carlo Dropout，20 次采样）
- **可解释性**：梯度显著性映射到原子重要性
- **过拟合防止**：
  - Dropout 正则化（根据数据集大小调整 dropout 率）
  - 早停机制（基于验证集 RMSE）
  - 根据数据集大小自动调整模型复杂度（隐藏层大小、层数、训练轮数）
- **适用场景**：大数据集（推荐 500+ 样本）、需要学习复杂分子模式、有 GPU 更佳
- **注意**：
  - 如果 PyTorch/PyTorch Geometric 不可用，会自动降级到 Baseline
  - 对于小数据集（<200 样本），基线模型通常表现更好
  - 系统会根据数据集大小自动调整模型参数（隐藏层大小、训练轮数等）

### 模型评估指标

系统使用多种指标评估模型性能：

- **RMSE**（Root Mean Squared Error）：均方根误差，衡量预测误差的大小
- **MAE**（Mean Absolute Error）：平均绝对误差，对异常值不敏感
- **R²**（Coefficient of Determination）：决定系数，衡量模型解释的方差比例
- **MAPE**（Mean Absolute Percentage Error）：平均绝对百分比误差，便于跨性质比较
- **相关系数**：预测值与真实值的线性相关性

所有评估指标在测试集上计算，使用 scaffold_split 确保与训练时的数据划分方式一致。

---

## 9. 测试

### Python 测试

```bash
pytest ml/tests/test_inference.py
```

### Node.js 测试

```bash
cd server
npm test
```