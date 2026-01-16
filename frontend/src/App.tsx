/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Routes, Route, Link, useNavigate } from "react-router-dom";
import axios from "axios";
import useSWR from "swr";
import { Button, Card, Input, Textarea, Badge, Progress, Table } from "./ui";
import { Chart, LineElement, PointElement, LinearScale, LogarithmicScale, CategoryScale, BarElement, Tooltip, Legend } from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import { FiUpload, FiSearch, FiDatabase, FiBarChart2, FiHome, FiFileText, FiDownload } from "react-icons/fi";

Chart.register(LineElement, PointElement, LinearScale, LogarithmicScale, CategoryScale, BarElement, Tooltip, Legend);

// 3Dmol global
declare global {
  interface Window { $3Dmol: any; }
}

const API = axios.create({ baseURL: "http://localhost:3001" });
const fetcher = (url: string) => API.get(url).then(r => r.data);

function colorForValue(v: number) {
  // 0..1 -> blue..red
  const x = Math.max(0, Math.min(1, v));
  const r = Math.floor(255 * x);
  const b = Math.floor(255 * (1 - x));
  return `rgb(${r},50,${b})`;
}

// 3D分子可视化组件，使用3Dmol.js渲染分子结构并显示原子重要性热力图
const Molecule3D: React.FC<{ sdf: string; atomImportances?: number[] }> = ({ sdf, atomImportances }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);

  useEffect(() => {
    if (!sdf || !window.$3Dmol || !containerRef.current) return;
    
    const container = containerRef.current;
    
    // 清理之前的 viewer
    if (viewerRef.current) {
      try {
        viewerRef.current.removeAll();
      } catch (e) {
        // 忽略清理错误
      }
      viewerRef.current = null;
    }
    
    // 清空容器内容（移除所有子元素，包括 canvas）
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    
    // 创建新的 viewer
    const viewer = new window.$3Dmol.GLViewer(container, { backgroundColor: "#ffffff" });
    viewerRef.current = viewer;
    
    viewer.addModel(sdf, "sdf");
    viewer.setStyle({}, { stick: { radius: 0.16 } });
    if (atomImportances && atomImportances.length > 0) {
      const max = Math.max(...atomImportances.map(v => Math.abs(v))) || 1;
      atomImportances.forEach((imp, idx) => {
        const norm = Math.abs(imp) / max;
        viewer.setStyle({ atomindex: idx }, { sphere: { radius: 0.35, color: colorForValue(norm) } });
      });
    }
    viewer.zoomTo();
    viewer.render();
    
    // 延迟 resize 确保容器已正确渲染
    const resizeTimeout = setTimeout(() => {
      if (viewerRef.current) {
        viewerRef.current.resize();
      }
    }, 100);
    
    const handle = () => {
      if (viewerRef.current) {
        viewerRef.current.resize();
      }
    };
    window.addEventListener("resize", handle);
    
    return () => {
      clearTimeout(resizeTimeout);
      window.removeEventListener("resize", handle);
      if (viewerRef.current) {
        try {
          viewerRef.current.removeAll();
        } catch (e) {
          // 忽略清理错误
        }
        viewerRef.current = null;
      }
      // 清理容器
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
    };
  }, [sdf, atomImportances]);

  return <div ref={containerRef} className="w-full h-[380px] rounded-lg border border-border relative" style={{ position: "relative" }} />;
};

// 页面布局组件，包含头部导航和页脚
const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="min-h-screen text-foreground bg-background">
    <header className="border-b border-border">
      <div className="mx-auto max-w-6xl px-5 py-4 flex items-center justify-between">
        <Link to="/" className="text-xl font-bold flex items-center gap-2">
          <FiHome /> MolPropLab
        </Link>
        <nav className="flex gap-4 text-sm">
          <Link to="/predict" className="hover:underline flex items-center gap-1">
            <FiSearch /> 单条预测
          </Link>
          <Link to="/batch" className="hover:underline flex items-center gap-1">
            <FiUpload /> 批量预测
          </Link>
          <Link to="/models" className="hover:underline flex items-center gap-1">
            <FiDatabase /> 模型浏览
          </Link>
          <Link to="/explain" className="hover:underline flex items-center gap-1">
            <FiBarChart2 /> 解释性分析
          </Link>
        </nav>
      </div>
    </header>
    <main className="mx-auto max-w-6xl px-5 py-6">{children}</main>
    <footer className="mx-auto max-w-6xl px-5 py-8 text-center text-sm text-muted">
      © 2025 MolPropLab. MIT License.
    </footer>
  </div>
);

// 首页组件，显示功能入口卡片
const Home: React.FC = () => {
  const items = [
    { title: "单条预测", href: "/predict", desc: "输入 SMILES → 获取性质预测、不确定性和原子级热力图", icon: FiSearch },
    { title: "批量预测", href: "/batch", desc: "上传 CSV/XLSX，跟踪任务进度，下载预测结果", icon: FiUpload },
    { title: "模型浏览", href: "/models", desc: "查看所有性质的模型评估结果和性能对比", icon: FiDatabase },
    { title: "解释性分析", href: "/explain", desc: "查看 Baseline 和 GNN 模型的性能对比图表", icon: FiBarChart2 }
  ];
  return (
    <Layout>
      <div className="grid md:grid-cols-2 gap-5">
        {items.map((x) => {
          const Icon = x.icon;
          return (
            <Card key={x.href}>
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  <Icon className="text-primary text-2xl mt-1" />
                  <div>
                    <h3 className="text-lg font-semibold">{x.title}</h3>
                    <p className="text-sm text-muted mt-1">{x.desc}</p>
                  </div>
                </div>
                <Link to={x.href}><Button>打开</Button></Link>
              </div>
            </Card>
          );
        })}
      </div>
      <Card className="mt-6">
        <div className="flex items-center gap-2">
          <Badge>Tech</Badge>
          <span className="text-sm text-muted">React + Vite + Tailwind + shadcn-style + 3Dmol.js</span>
        </div>
      </Card>
    </Layout>
  );
};

// 单条预测页面组件，允许用户输入SMILES并查看预测结果
const SinglePrediction: React.FC = () => {
  const [smiles, setSmiles] = useState("CCO");
  const [model, setModel] = useState<"baseline" | "gnn">("baseline");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [smilesError, setSmilesError] = useState<string | null>(null);
  const [netError, setNetError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // SMILES输入合法性校验
  function validateSmiles(s: string): string | null {
    const t = s.trim();
    // 空输入
    if (!t) return "输入的 SMILES 不合法！";
    // 含空格 / 换行
    if (/\s/.test(t)) return "输入的 SMILES 不合法！";
    // 非法字符
    const allowed = /^[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:]+$/;
    if (!allowed.test(t)) return "输入的 SMILES 不合法！";
    return null;
  }
  const handle = async () => {
    if (loading) return;

    // 校验 SMILES
    const err = validateSmiles(smiles);
    if (err) {
      setSmilesError(err);
      return;
    }
    setSmilesError(null);
    setNetError(null);

    // 取消预测
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);

    console.log("[ui] predict start", { smiles, model });

    try {
      const { data } = await API.post(
        "/predict",
        { smiles, model },
        {
          signal: controller.signal,
          timeout: 0,
        }
      );

      console.log("[ui] predict success, keys=", Object.keys(data || {}));
      setResult(data);
    } catch (e: any) {
      const isCanceled =
        e?.code === "ERR_CANCELED" ||
        e?.name === "CanceledError" ||
        e?.name === "AbortError";

      if (isCanceled) {
        console.log("[ui] predict canceled");
        return;
      }

      if (e?.code === "ECONNABORTED") {
        setNetError("请求超时：后端未及时返回结果。");
      } else {
        setNetError(e?.message || "请求失败");
      }
      console.error("[ui] predict error:", e);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setLoading(false);
        console.log("[ui] predict end");
      }
    }
  };

  return (
    <Layout>
      <Card>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="text-sm opacity-80">SMILES 字符串</label>
            <Textarea
              rows={5}
              value={smiles}
              onChange={(e) => {
                setSmiles(e.target.value);
                // 一旦用户重新输入，就清除旧的错误提示
                if (smilesError) setSmilesError(null);
              }}
              placeholder="请输入 SMILES 字符串，例如：CCO"
            />

            {smilesError && (
              <div className="mt-2 text-sm text-red-600">
                {smilesError}
              </div>
            )}
            <div className="flex items-center gap-3 mt-3">
              <label className="text-sm opacity-80">模型</label>
              <select
                className="bg-white border border-border rounded px-2 py-1 text-sm"
                value={model}
                onChange={(e) => setModel(e.target.value as any)}
              >
                <option value="baseline">基线模型 (Baseline)</option>
                <option value="gnn">图神经网络 (GNN)</option>
              </select>
              <Button onClick={handle} disabled={loading} className="flex items-center gap-2">
                <FiSearch /> {loading ? "预测中..." : "预测"}
              </Button>
              {loading && (
                <Button
                  type="button"
                  onClick={() => {
                    abortRef.current?.abort();
                    setNetError("已取消本次预测。");
                  }}
                  className="bg-red-400 hover:bg-red-500 text-white"
                >
                  取消
                </Button>
              )}

            </div>
                {result && (
              <div className="mt-4">
                <div className="flex flex-wrap items-center gap-2 mb-4">
                  <Badge>模型: {result.model === "baseline" ? "基线" : "GNN"}</Badge>
                </div>
                {result.properties ? (
                  <div className="space-y-2">
                    <h4 className="text-sm font-semibold mb-2">所有性质预测结果：</h4>
                    <div className="border border-border rounded-lg overflow-hidden">
                      <table className="w-full text-sm">
                        <thead className="bg-gray-50 border-b border-border">
                          <tr>
                            <th className="px-3 py-2 text-left font-medium">性质</th>
                            <th className="px-3 py-2 text-right font-medium">预测值</th>
                            <th className="px-3 py-2 text-right font-medium">不确定性 (σ)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(result.properties).map(([key, prop]: [string, any]) => (
                            <tr key={key} className="border-b border-border/50 last:border-0">
                              <td className="px-3 py-2">
                                <div className="font-medium">{prop.name}</div>
                                {prop.unit && <div className="text-xs text-muted">{prop.unit}</div>}
                              </td>
                              <td className="px-3 py-2 text-right">
                                {isNaN(prop.prediction) ? (
                                  <span className="text-muted">N/A</span>
                                ) : (
                                  <b>{Number(prop.prediction).toFixed(4)}</b>
                                )}
                              </td>
                              <td className="px-3 py-2 text-right">
                                {isNaN(prop.uncertainty) ? (
                                  <span className="text-muted">N/A</span>
                                ) : (
                                  <span className="text-muted">{Number(prop.uncertainty).toFixed(4)}</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : (
                  <div className="text-sm">
                    <p className="mt-2">预测值: <b>{Number(result.prediction).toFixed(4)}</b></p>
                    <p>不确定性 (σ): <b>{Number(result.uncertainty).toFixed(4)}</b></p>
                  </div>
                )}
              </div>
            )}
          </div>
          <div>
            {result?.sdf ? (
              <Molecule3D sdf={result.sdf} atomImportances={result.atom_importances} />
            ) : (
              <div className="h-[380px] flex items-center justify-center border border-border rounded bg-white text-muted">暂无分子结构</div>
            )}
          </div>
        </div>
      </Card>
    </Layout>
  );
};

// 批量预测页面组件，允许用户上传CSV/XLSX文件并跟踪任务进度
const BatchPrediction: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<"baseline" | "gnn">("baseline");
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<any>(null);
  const [uploading, setUploading] = useState(false);
  

  useEffect(() => {
    if (!jobId) return;
    const t = setInterval(async () => {
      const { data } = await API.get(`/job/${jobId}`);
      setStatus(data);
      setProgress(Math.round((data.progress || 0) * 100));
      if (data.state === "done" || data.state === "error") {
        clearInterval(t);
      }
    }, 1500);
    return () => clearInterval(t);
  }, [jobId]);

  const upload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("model", model);
      const { data } = await API.post("/batch_predict", fd, { headers: { "Content-Type": "multipart/form-data" } });
      setJobId(data.jobId);
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setUploading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setJobId(null);
    setProgress(0);
    setStatus(null);
  };

  return (
    <Layout>
      <div className="grid md:grid-cols-2 gap-6">
        {/* 左侧：文件上传区域 */}
        <Card>
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <FiUpload /> 上传文件
          </h3>
          <div className="flex flex-col gap-4">
            <div>
              <label className="flex items-center gap-2 text-sm font-medium mb-2">
                <FiFileText /> 选择文件（CSV 或 XLSX）
              </label>
              <input 
                type="file" 
                accept=".csv,.xlsx" 
                onChange={(e) => setFile(e.target.files?.[0] || null)} 
                className="block w-full text-sm text-foreground file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-white hover:file:opacity-90 cursor-pointer"
                disabled={!!jobId}
              />
              {file && (
                <p className="text-xs text-muted mt-2 flex items-center gap-1">
                  <FiFileText /> {file.name} ({(file.size / 1024).toFixed(1)} KB)
                </p>
              )}
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm font-medium mb-2">
                选择模型
              </label>
              <select
                className="w-full bg-white border border-border rounded-md px-3 py-2 text-sm text-foreground"
                value={model}
                onChange={(e) => setModel(e.target.value as "baseline" | "gnn")}
                disabled={!!jobId}
              >
                <option value="baseline">基线模型 (Baseline)</option>
                <option value="gnn">图神经网络 (GNN)</option>
              </select>
              <p className="text-xs text-muted mt-1">
                {model === "baseline" 
                  ? "使用 LightGBM，速度快，稳定性好" 
                  : "使用 GIN 图神经网络，适合复杂分子模式"}
              </p>
            </div>
            <div className="flex gap-2">
              <Button 
                onClick={upload} 
                disabled={!file || uploading || !!jobId} 
                className="flex items-center gap-2 flex-1"
              >
                <FiUpload /> {uploading ? "上传中..." : "开始预测"}
              </Button>
              {jobId && (
                <Button variant="ghost" onClick={reset} className="flex items-center gap-2">
                  重置
                </Button>
              )}
            </div>
          </div>
        </Card>

        {/* 右侧：任务状态区域 */}
        <Card>
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <FiDatabase /> 任务状态
          </h3>
          {!jobId ? (
            <div className="text-center py-8 text-muted">
              <p>暂无任务</p>
              <p className="text-xs mt-2">上传文件以开始批量预测</p>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm">
                  <span className="font-medium">任务 ID：</span>
                  <code className="text-xs bg-gray-100 px-2 py-1 rounded">{jobId}</code>
                </div>
                <Badge color={
                  status?.state === "done" ? "rgba(16,185,129,0.15)" :
                  status?.state === "error" ? "rgba(239,68,68,0.15)" :
                  status?.state === "running" ? "rgba(59,130,246,0.15)" :
                  "rgba(156,163,175,0.15)"
                }>
                  {status?.state === "done" ? "已完成" :
                   status?.state === "error" ? "错误" :
                   status?.state === "running" ? "运行中" :
                   "排队中"}
                </Badge>
              </div>
              
              {status?.state === "running" && (
                <div>
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span>进度</span>
                    <span className="font-medium">{progress}%</span>
                  </div>
                  <Progress value={progress} />
                </div>
              )}

              {status?.state === "done" && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <p className="text-sm font-medium text-green-800 mb-2">✓ 预测完成！</p>
                  <a 
                    className="inline-flex items-center gap-2 text-sm text-primary hover:underline" 
                    href={`http://localhost:3001/job/${jobId}/download`} 
                    target="_blank"
                  >
                    <FiDownload /> 下载结果
                  </a>
                </div>
              )}

              {status?.state === "error" && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <p className="text-sm font-medium text-red-800 mb-1">✗ 发生错误</p>
                  <p className="text-xs text-red-600">{status.error || "未知错误"}</p>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>

      {/* 使用说明 */}
      <Card className="mt-6">
        <h3 className="text-sm font-semibold mb-2">📋 文件格式要求</h3>
        <ul className="text-xs text-muted space-y-1 list-disc list-inside">
          <li>文件必须包含 <code className="bg-gray-100 px-1 rounded">smiles</code> 列</li>
          <li>支持格式：CSV (.csv) 或 Excel (.xlsx)</li>
          <li>每行应包含一个 SMILES 字符串</li>
          <li>结果将以 CSV 格式下载，包含预测值和不确定性</li>
        </ul>
      </Card>
    </Layout>
  );
};

// 模型浏览器组件，显示所有性质的评估信息
const ModelExplorer: React.FC = () => {
  const { data, error } = useSWR("/models", fetcher);
  const properties = data?.properties || [];
  
  const columns = ["性质", "Baseline RMSE", "Baseline R²", "GNN RMSE", "GNN R²", "更好模型"];
  const rows = properties.map((p: any) => {
    const baseline = p.baseline || {};
    const gnn = p.gnn || {};
    const hasBaseline = !baseline.error && baseline.rmse !== undefined;
    const hasGNN = !gnn.error && gnn.rmse !== undefined;
    
    return [
      p.property_name || p.property,
      hasBaseline ? Number(baseline.rmse).toFixed(4) : "未评估",
      hasBaseline ? Number(baseline.r2).toFixed(4) : "-",
      hasGNN ? Number(gnn.rmse).toFixed(4) : "未评估",
      hasGNN ? Number(gnn.r2).toFixed(4) : "-",
      p.better_model === "baseline" ? "Baseline" : p.better_model === "gnn" ? "GNN" : "-"
    ];
  });
  
  const hasEvaluated = properties.some((p: any) => 
    (!p.baseline?.error && p.baseline?.rmse !== undefined) || 
    (!p.gnn?.error && p.gnn?.rmse !== undefined)
  );
  
  return (
    <Layout>
      <Card>
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <FiDatabase /> 模型评估结果
        </h3>
        {error ? (
          <div className="text-red-600 text-sm">加载失败: {error.message}</div>
        ) : !hasEvaluated ? (
          <div className="space-y-3">
            <p className="text-muted text-sm">
              暂无模型评估数据。请先训练和评估模型。
            </p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm font-medium text-blue-800 mb-2">如何生成评估数据：</p>
              <ol className="text-xs text-blue-700 space-y-1 list-decimal list-inside">
                <li>训练模型：在<code className="bg-blue-100 px-1 rounded">ml</code>目录下运行 <code className="bg-blue-100 px-1 rounded">python train_baseline.py</code> 和 <code className="bg-blue-100 px-1 rounded">python train_gnn.py</code></li>
                <li>评估模型：在<code className="bg-blue-100 px-1 rounded">ml</code>目录下运行 <code className="bg-blue-100 px-1 rounded">python compare_models.py</code></li>
                <li>评估结果将自动保存到 <code className="bg-blue-100 px-1 rounded">*_comparison.json</code> 文件</li>
              </ol>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table columns={columns} rows={rows} />
            <p className="text-xs text-muted mt-3">
              共 {properties.length} 个性质，{properties.filter((p: any) => p.better_model).length} 个已评估
            </p>
          </div>
        )}
      </Card>
    </Layout>
  );
};

// 解释性可视化组件，显示两个模型的性能对比
const ExplanationViewer: React.FC = () => {
  const { data, error } = useSWR("/models", fetcher);
  const properties = data?.properties || [];
  const [metric, setMetric] = useState<"rmse" | "mape">("rmse");
  
  const hasMetric = (obj: any) => {
    if (!obj || obj.error) return false;
    const v = metric === "rmse" ? obj.rmse : obj.mape;
    return v !== undefined && v !== null && isFinite(Number(v));
  };

  // 根据当前指标筛选，避免 RMSE/MAPE 混用导致展示混乱
  const evaluatedProperties = properties.filter((p: any) =>
    hasMetric(p.baseline) || hasMetric(p.gnn)
  );
  
  const chart = useMemo(() => {
    if (evaluatedProperties.length === 0) return null;

    const pick = (obj: any) => {
      if (!hasMetric(obj)) return null;
      const n = Number(metric === "rmse" ? obj.rmse : obj.mape);
      // log scale 不能显示 <=0
      if (metric === "rmse" && n <= 0) return null;
      return n;
    };
    
    return {
      labels: evaluatedProperties.map((p: any) => p.property_name || p.property),
      datasets: [
        { 
          label: metric === "rmse" ? "Baseline RMSE (log y)" : "Baseline MAPE (%)", 
          data: evaluatedProperties.map((p: any) => pick(p.baseline)), 
          borderWidth: 2,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          pointBackgroundColor: "#3b82f6",
          pointBorderColor: "#3b82f6"
        },
        { 
          label: metric === "rmse" ? "GNN RMSE (log y)" : "GNN MAPE (%)", 
          data: evaluatedProperties.map((p: any) => pick(p.gnn)), 
          borderWidth: 2,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          pointBackgroundColor: "#10b981",
          pointBorderColor: "#10b981"
        }
      ]
    };
  }, [evaluatedProperties, metric]);

  const chartOptions = useMemo(() => {
    return {
      responsive: true,
      interaction: { mode: "index" as const, intersect: false },
      plugins: {
        legend: { position: "top" as const },
        tooltip: {
          callbacks: {
            label: (ctx: any) => {
              const v = ctx.parsed?.y;
              if (v === null || v === undefined || Number.isNaN(v)) return `${ctx.dataset.label}: N/A`;
              const num = Number(v);
              if (metric === "rmse") return `${ctx.dataset.label}: ${num.toExponential(3)}`;
              return `${ctx.dataset.label}: ${num.toFixed(2)}%`;
            }
          }
        }
      },
      scales: {
        y: metric === "rmse"
          ? {
              type: "logarithmic" as const,
              title: { display: true, text: "RMSE (log scale, unit varies by property)" },
              ticks: {
                maxTicksLimit: 6,
                callback: (value: any) => {
                  const n = Number(value);
                  if (!isFinite(n)) return "";
                  const exp = Math.log10(n);
                  const isPow10 = Math.abs(exp - Math.round(exp)) < 1e-6;
                  if (!isPow10) return "";
                  // 只显示 10^k 的主刻度，减少密度
                  if (n >= 1000 || n <= 0.01) return n.toExponential(0);
                  return n.toString();
                }
              }
            }
          : {
              type: "linear" as const,
              title: { display: true, text: "MAPE (%)" },
              ticks: { callback: (v: any) => `${v}%` }
            },
        x: { ticks: { maxRotation: 45, minRotation: 0 } }
      }
    };
  }, [metric]);

  const r2Chart = useMemo(() => {
    if (evaluatedProperties.length === 0) return null;
    const pickR2 = (obj: any) => {
      if (!obj || obj.error) return null;
      const v = obj.r2;
      if (v === undefined || v === null || !isFinite(Number(v))) return null;
      return Number(v);
    };
    return {
      labels: evaluatedProperties.map((p: any) => p.property_name || p.property),
      datasets: [
        {
          label: "Baseline R²",
          data: evaluatedProperties.map((p: any) => pickR2(p.baseline)),
          backgroundColor: "rgba(59, 130, 246, 0.4)",
          borderColor: "#3b82f6",
          borderWidth: 1,
        },
        {
          label: "GNN R²",
          data: evaluatedProperties.map((p: any) => pickR2(p.gnn)),
          backgroundColor: "rgba(16, 185, 129, 0.4)",
          borderColor: "#10b981",
          borderWidth: 1,
        },
      ],
    };
  }, [evaluatedProperties]);

  const r2Options = useMemo(() => {
    return {
      responsive: true,
      interaction: { mode: "index" as const, intersect: false },
      plugins: { legend: { position: "top" as const } },
      scales: {
        y: {
          title: { display: true, text: "R² (higher is better)" },
          suggestedMin: -1,
          suggestedMax: 1,
        },
        x: { ticks: { maxRotation: 45, minRotation: 0 } },
      },
    };
  }, []);

  const correlationChart = useMemo(() => {
    if (evaluatedProperties.length === 0) return null;
    const pickCorr = (obj: any) => {
      if (!obj || obj.error) return null;
      const v = obj.correlation;
      if (v === undefined || v === null || !isFinite(Number(v))) return null;
      return Number(v);
    };
    return {
      labels: evaluatedProperties.map((p: any) => p.property_name || p.property),
      datasets: [
        {
          label: "Baseline Correlation",
          data: evaluatedProperties.map((p: any) => pickCorr(p.baseline)),
          backgroundColor: "rgba(59, 130, 246, 0.35)",
          borderColor: "#3b82f6",
          borderWidth: 1,
        },
        {
          label: "GNN Correlation",
          data: evaluatedProperties.map((p: any) => pickCorr(p.gnn)),
          backgroundColor: "rgba(16, 185, 129, 0.35)",
          borderColor: "#10b981",
          borderWidth: 1,
        },
      ],
    };
  }, [evaluatedProperties]);

  const correlationOptions = useMemo(() => {
    return {
      responsive: true,
      interaction: { mode: "index" as const, intersect: false },
      plugins: {
        legend: { position: "top" as const },
        tooltip: {
          callbacks: {
            label: (ctx: any) => {
              const v = ctx.parsed?.y;
              if (v === null || v === undefined || Number.isNaN(v)) return `${ctx.dataset.label}: N/A`;
              return `${ctx.dataset.label}: ${Number(v).toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        y: {
          title: { display: true, text: "Correlation (Pearson, higher is better)" },
          suggestedMin: -1,
          suggestedMax: 1,
        },
        x: { ticks: { maxRotation: 45, minRotation: 0 } },
      },
    };
  }, []);

  const correlationSummary = useMemo(() => {
    const baselineVals = evaluatedProperties
      .map((p: any) => p.baseline?.correlation)
      .filter((v: any) => v !== undefined && v !== null && isFinite(Number(v)))
      .map((v: any) => Number(v));
    const gnnVals = evaluatedProperties
      .map((p: any) => p.gnn?.correlation)
      .filter((v: any) => v !== undefined && v !== null && isFinite(Number(v)))
      .map((v: any) => Number(v));
    const avg = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
    return { baselineAvg: avg(baselineVals), gnnAvg: avg(gnnVals) };
  }, [evaluatedProperties]);

  const summary = useMemo(() => {
    const total = properties.length;
    const baselineWins = properties.filter((p: any) => p.better_model === "baseline").length;
    const gnnWins = properties.filter((p: any) => p.better_model === "gnn").length;
    const evaluated = properties.filter((p: any) => p.better_model).length;
    return { total, evaluated, baselineWins, gnnWins };
  }, [properties]);

  return (
    <Layout>
      <Card>
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <FiBarChart2 /> 模型性能对比
        </h3>
        {error ? (
          <div className="text-red-600 text-sm">加载失败: {error.message}</div>
        ) : evaluatedProperties.length === 0 ? (
          <div className="space-y-3">
            <p className="text-muted text-sm">
              暂无模型评估数据。请先训练和评估模型。
            </p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm font-medium text-blue-800 mb-2">如何生成评估数据：</p>
              <ol className="text-xs text-blue-700 space-y-1 list-decimal list-inside">
                <li>训练模型：在<code className="bg-blue-100 px-1 rounded">ml</code>目录下运行 <code className="bg-blue-100 px-1 rounded">python train_baseline.py</code> 和 <code className="bg-blue-100 px-1 rounded">python train_gnn.py</code></li>
                <li>评估模型：在<code className="bg-blue-100 px-1 rounded">ml</code>目录下运行 <code className="bg-blue-100 px-1 rounded">python compare_models.py</code></li>
                <li>评估结果将自动保存到 <code className="bg-blue-100 px-1 rounded">*_comparison.json</code> 文件</li>
              </ol>
            </div>
          </div>
        ) : (
          <div>
            <p className="text-sm text-muted mb-4">
              {metric === "rmse"
                ? "此图表比较 Baseline 和 GNN 在各性质上的 RMSE。由于不同性质量纲差异很大，纵轴使用对数刻度（更便于同时观察多性质）。数值越小越好。"
                : "此图表比较 Baseline 和 GNN 在各性质上的 MAPE（百分比误差），更适合跨性质对比。数值越小越好。"}
            </p>

            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm text-muted">指标：</span>
              <select
                className="bg-white border border-border rounded px-2 py-1 text-sm"
                value={metric}
                onChange={(e) => setMetric(e.target.value as any)}
              >
                <option value="rmse">RMSE（对数纵轴）</option>
                <option value="mape">MAPE（%）</option>
              </select>
            </div>
            {chart && <Line data={chart} options={chartOptions} />}
            <div className="mt-4 grid md:grid-cols-2 gap-4">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="text-sm font-medium text-blue-800 mb-1">Baseline 模型</p>
                <p className="text-xs text-blue-700">
                  {metric === "rmse"
                    ? `平均 RMSE: ${(evaluatedProperties.reduce((sum: number, p: any) => sum + (p.baseline?.rmse || 0), 0) / Math.max(1, evaluatedProperties.filter((p: any) => hasMetric(p.baseline)).length)).toFixed(4)}`
                    : `平均 MAPE: ${(evaluatedProperties.reduce((sum: number, p: any) => sum + (p.baseline?.mape || 0), 0) / Math.max(1, evaluatedProperties.filter((p: any) => hasMetric(p.baseline)).length)).toFixed(2)}%`}
                </p>
              </div>
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <p className="text-sm font-medium text-green-800 mb-1">GNN 模型</p>
                <p className="text-xs text-green-700">
                  {metric === "rmse"
                    ? `平均 RMSE: ${(evaluatedProperties.reduce((sum: number, p: any) => sum + (p.gnn?.rmse || 0), 0) / Math.max(1, evaluatedProperties.filter((p: any) => hasMetric(p.gnn)).length)).toFixed(4)}`
                    : `平均 MAPE: ${(evaluatedProperties.reduce((sum: number, p: any) => sum + (p.gnn?.mape || 0), 0) / Math.max(1, evaluatedProperties.filter((p: any) => hasMetric(p.gnn)).length)).toFixed(2)}%`}
                </p>
              </div>
            </div>

            <div className="mt-6">
              <h4 className="text-sm font-semibold mb-2">R² 对比（同性质内越高越好）</h4>
              {r2Chart && <Bar data={r2Chart} options={r2Options} />}
            </div>

            <div className="mt-6">
              <h4 className="text-sm font-semibold mb-2">Correlation 对比（Pearson，越高越好）</h4>
              <p className="text-xs text-muted mb-2">
                相关系数反映预测值与真实值的线性一致性，适合补充 RMSE/MAPE 的整体趋势判断。
                {correlationSummary.baselineAvg !== null && correlationSummary.gnnAvg !== null
                  ? ` 平均相关系数：Baseline ${correlationSummary.baselineAvg.toFixed(4)}，GNN ${correlationSummary.gnnAvg.toFixed(4)}。`
                  : ""}
              </p>
              {correlationChart && <Bar data={correlationChart} options={correlationOptions} />}
            </div>

            <div className="mt-6 grid md:grid-cols-3 gap-3">
              <div className="bg-gray-50 border border-border rounded-lg p-3">
                <p className="text-xs text-muted mb-1">已评估性质</p>
                <p className="text-lg font-semibold">{summary.evaluated}/{summary.total}</p>
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="text-xs text-blue-700 mb-1">Baseline 优势</p>
                <p className="text-lg font-semibold text-blue-800">{summary.baselineWins}</p>
              </div>
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <p className="text-xs text-green-700 mb-1">GNN 优势</p>
                <p className="text-lg font-semibold text-green-800">{summary.gnnWins}</p>
              </div>
            </div>
          </div>
        )}
      </Card>
    </Layout>
  );
};

// 主应用组件，定义路由配置
export default function App() {
  const nav = useNavigate();
  useEffect(() => {
    // redirect root -> home
  }, []);
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/predict" element={<SinglePrediction />} />
      <Route path="/batch" element={<BatchPrediction />} />
      <Route path="/models" element={<ModelExplorer />} />
      <Route path="/explain" element={<ExplanationViewer />} />
      <Route path="*" element={<Layout><Card>404 页面未找到 <Button variant="ghost" onClick={() => nav("/")}>返回首页</Button></Card></Layout>} />
    </Routes>
  );
}
