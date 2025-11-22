# MolPropLab — 分子性质预测平台

一个用于从 SMILES 预测分子性质的端到端全栈机器学习应用。

## 1. 核心特性

- **多性质预测：** 支持同时预测 10 种分子性质（分子量、LogP、LogS、pKa、沸点、熔点、折射率、蒸气压、密度、闪点）
- **Python 端 ML：** RDKit 预处理、分子描述符、ECFP（1024）、PyTorch Geometric GIN、LightGBM/RandomForest 基线模型  
- **不确定性估计：** MC-Dropout（GNN）/ 小型集成（Baseline）  
- **可解释性：** SHAP（基线模型）、梯度显著性（GNN）→ 原子级重要性 JSON  
- **后端：** Node.js + Express + TypeScript；通过 `child_process.spawn` 调用 Python  
- **前端：** React + Vite + Tailwind + shadcn 风格组件 + 3Dmol.js  
- **批处理任务：** 简单内存队列 & CSV/XLSX 上传，支持模型选择

输入 SMILES，一次性获得所有性质的预测值、不确定性以及原子重要性热力图。

---

## 2. 环境要求

- **Python**：建议 3.10 或 3.11
- **Node.js**：建议 20.19.5
  - 前往 [Node.js 官网](https://nodejs.org/) 下载并安装 **Node.js 20.19.5**
  - 或直接下载压缩包：[Node.js v20.19.5](https://nodejs.org/dist/v20.19.5/)，解压后将 `node.exe` 所在文件夹的路径添加到系统环境变量中
- **推荐**：使用 Conda 来安装 RDKit（Windows 上不推荐直接用 pip）

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
> - 当前使用 **PyTorch 2.3.1**（CPU 版本）和 **NumPy 1.x**（<2.0），以确保在 Windows 上的稳定性
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
# 评估基线模型
python eval_baseline.py

# 评估 GNN 模型
python eval_gnn.py
```

### 推理

```bash
# 单条预测（返回所有性质的预测结果）
python inference.py --smiles "CCO" --model baseline --json
python inference.py --smiles "CCO" --model gnn --json

# 批量预测（支持CSV和XLSX格式）
python inference.py --csv data/logp.csv --output out.csv --model baseline
python inference.py --xlsx data/logp.xlsx --output out.csv --model gnn
```

> **注意**：单条预测会返回所有 10 种性质的预测结果。批量预测的输出文件会包含所有性质的列（格式：`{property}_prediction` 和 `{property}_uncertainty`）。

---

## 7. API 说明

### `POST /predict`

单条预测端口，返回所有性质的预测结果。

**请求：**

```json
{
  "smiles": "CCO",
  "model": "baseline" | "gnn"
}
```

**返回：**

```json
{
  "properties": {
    "molecular_weight": {
      "name": "分子量 (MW)",
      "unit": "g/mol",
      "prediction": 46.07,
      "uncertainty": 0.5
    },
    "logp": {
      "name": "LogP (脂溶性)",
      "unit": "",
      "prediction": 0.3476,
      "uncertainty": 0.0117
    },
    "logs": {
      "name": "LogS (水溶解度)",
      "unit": "log(mol/L)",
      "prediction": -0.1301,
      "uncertainty": 0.0474
    },
    "pka": {
      "name": "pKa",
      "unit": "",
      "prediction": 15.9,
      "uncertainty": 0.5
    },
    "boiling_point": {
      "name": "沸点",
      "unit": "°C",
      "prediction": 78.4,
      "uncertainty": 3.0866
    },
    // ... 其他性质（熔点、折射率、蒸气压、密度、闪点）
  },
  "atom_importances": [0.0, 0.5, 1.0, ...],
  "sdf": "...",
  "model": "baseline",
  "version": "v1"
}
```

### `POST /batch_predict`

批量预测端口，支持 CSV 和 XLSX 文件。

**请求：** 上传表单（`multipart/form-data`），字段：
- `file`：CSV 或 XLSX 文件（需包含 `smiles` 列）
- `model`：模型类型（`baseline` 或 `gnn`，可选，默认为 `gnn`）

**返回：**

```json
{
  "jobId": "xxxx-xxxx-xxxx"
}
```

**轮询进度：** `GET /job/:id`

**下载结果：** `GET /job/:id/download`（CSV 文件，包含所有性质的预测列）

### 其他 API

- `GET /models` → 模型注册表 & 基本指标
- `POST /explain` → 与 `/predict` 类似，但强调解释性输出
- `GET /health` → 健康检查 `{ ok: true }`

---

## 8. 前端说明

- **React + Vite + Tailwind**
- **shadcn 风格组件**：压缩在单文件 `src/ui.tsx`
- **3Dmol.js**：通过 CDN 加载
- 3D 结构由后端 RDKit 生成 SDF（字符串），前端按原子重要性热度上色

---

## 9. 依赖说明

### Python 依赖

- **RDKit**：推荐使用 Conda 安装 `conda install -c conda-forge rdkit`，Windows 上不推荐直接用 pip
- **PyTorch**：当前使用 2.3.1 版本（CPU），在 Windows 上更稳定。若遇到 DLL 加载问题，请确保已安装 Visual C++ Redistributable
- **NumPy**：使用 1.x 版本（<2.0）以确保与 PyTorch 2.3.1 兼容
- **PyTorch Geometric**：在部分系统安装较麻烦；若安装失败，系统将**自动降级到基线模型**

### Node.js 依赖

见 `server/package.json` 和 `frontend/package.json`

---

## 10. 常见问题与故障排除

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
2. 当前 `requirements.txt` 已指定 PyTorch 2.3.1，这是经过测试的稳定版本
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
- 修改 `server/package.json` 中的 `dev` 脚本，设置正确的 `PYTHON` 环境变量
- 或在启动前设置：`set PYTHON=你的python路径 && npm run dev`
- 默认路径：`D:\anaconda3\envs\molproplab\python.exe`（Windows）

---

## 11. 模型说明

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
- **不确定性**：集成模型的标准差
- **可解释性**：SHAP 值映射到原子重要性
- **适用场景**：快速预测、小到中等数据集、无需 GPU
- **数据要求**：每个性质至少需要 20+ 样本，推荐 100+ 样本以获得更好性能

### GNN 模型

- **算法**：GIN（Graph Isomorphism Network）
- **特征**：图神经网络，直接学习分子图结构
- **不确定性**：MC-Dropout（Monte Carlo Dropout）
- **可解释性**：梯度显著性映射到原子重要性
- **适用场景**：大数据集（推荐 500+ 样本）、需要学习复杂分子模式、有 GPU 更佳
- **注意**：
  - 如果 PyTorch/PyTorch Geometric 不可用，会自动降级到 Baseline
  - 对于小数据集（<200 样本），基线模型通常表现更好
  - 系统会根据数据集大小自动调整模型参数（隐藏层大小、训练轮数等）

---

## 12. 测试

### Python 测试

```bash
pytest ml/tests/test_inference.py
```

### Node.js 测试

```bash
cd server
npm test
```

---

## 13. 项目结构（仅展示主要文件）

```
MolPropLab/
├── ml/                    # Python ML 代码
│   ├── core.py           # 核心 ML 功能（模型、特征提取、多性质预测等）
│   ├── inference.py      # CLI 推理接口
│   ├── train_baseline.py # 训练所有性质的基线模型
│   ├── train_gnn.py      # 训练所有性质的 GNN 模型
│   ├── eval_baseline.py  # 评估基线模型
│   ├── eval_gnn.py      # 评估 GNN 模型
│   ├── data/            # 数据目录（包含各性质的CSV文件）
│   │   ├── molecular_weight.csv
│   │   ├── logp.csv
│   │   ├── logs.csv
│   │   ├── pka.csv
│   │   ├── boiling_point.csv
│   │   ├── melting_point.csv
│   │   ├── refractive_index.csv
│   │   ├── vapor_pressure.csv
│   │   ├── density.csv
│   │   └── flash_point.csv
│   └── saved_models/    # 保存的模型（每个性质一个模型文件）
├── server/              # Node.js 后端
│   ├── src/
│   │   └── index.ts     # Express 服务器
│   └── package.json
├── frontend/            # React 前端
│   ├── src/
│   │   ├── App.tsx      # 主应用组件
│   │   └── ui.tsx       # UI 组件
│   └── package.json
└── requirements.txt     # Python 依赖
```

---

## 14. 许可证

MIT

详见仓库内 `LICENSE` 文件。