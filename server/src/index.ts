import express from "express";
import cors from "cors";
import multer from "multer";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import crypto from "crypto";

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3001;
// 优先使用环境变量，如果没有则使用默认的Conda路径
const PYTHON = process.env.PYTHON || (process.env.CONDA_PREFIX ? path.join(process.env.CONDA_PREFIX, "python.exe") : "python");
const ROOT = path.resolve(__dirname, "../..");
const ML_DIR = path.join(ROOT, "ml");

// 设置PyTorch环境变量，避免DLL加载问题
process.env.TORCH_SHM_DISABLE = "1";
const TMP_DIR = path.join(__dirname, "..", "tmp");
if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR, { recursive: true });

// 定期清理旧的临时文件（超过1小时未访问的文件）
function cleanupOldFiles() {
  try {
    const files = fs.readdirSync(TMP_DIR);
    const now = Date.now();
    const maxAge = 60 * 60 * 1000; // 1小时
    
    for (const file of files) {
      const filePath = path.join(TMP_DIR, file);
      try {
        const stats = fs.statSync(filePath);
        const age = now - stats.mtimeMs;
        if (age > maxAge) {
          fs.unlinkSync(filePath);
          console.log(`[cleanup] Deleted old file: ${file} (age: ${Math.round(age / 1000 / 60)} minutes)`);
        }
      } catch (err) {
        // 忽略文件已删除等错误
      }
    }
  } catch (err) {
    console.error("[cleanup] Error cleaning old files:", err);
  }
}

// 每小时清理一次旧文件
setInterval(cleanupOldFiles, 60 * 60 * 1000);
// 启动时也清理一次
cleanupOldFiles();

// 调试信息：打印实际使用的Python路径
console.log(`[server] Using Python: ${PYTHON}`);
console.log(`[server] PYTHON env var: ${process.env.PYTHON || "not set"}`);

app.use(cors());
app.use(express.json({ limit: "2mb" }));

type Job = {
  id: string;
  state: "queued" | "running" | "done" | "error";
  progress: number;
  input: string;
  output: string;
  error?: string;
};
const jobs = new Map<string, Job>();

// 调用Python脚本并返回执行结果
function callPython(args: string[], onStdout?: (s: string) => void, onStderr?: (s: string) => void, req?: express.Request, res?: express.Response): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const env = { ...process.env, TORCH_SHM_DISABLE: "1" };
    const p = spawn(PYTHON, args, { cwd: ML_DIR, stdio: ["ignore", "pipe", "pipe"], env });
    
    // 前端断开，杀死子进程
    let killed = false;
    const killChild = () => {
      if (killed) return;
      killed = true;

      if (!p.pid) return;
      if (process.platform === "win32") {
        try {
          spawn("taskkill", ["/pid", String(p.pid), "/T", "/F"]);
        } catch {}
      } else {
        try {
          p.kill("SIGTERM");
        } catch {}
      }
    };
    if (req) req.on("aborted", killChild);
    if (res) {
      res.on("close", () => {
        if (!res.writableEnded) killChild();
      });
    }; 

    let stdout = "";
    let stderr = "";
    p.stdout.on("data", (d) => {
      const s = d.toString();
      stdout += s;
      onStdout?.(s);
    });
    p.stderr.on("data", (d) => {
      const s = d.toString();
      stderr += s;
      onStderr?.(s);
    });
    p.on("close", (code) => resolve({ code: code ?? 0, stdout, stderr }));
  });
}

// 健康检查端口
app.get("/health", (_req, res) => res.json({ ok: true }));

// 单条预测端口，接收SMILES字符串并返回预测结果
app.post("/predict", async (req, res) => {
  try {
    const { smiles, model } = req.body || {};
    if (!smiles) return res.status(400).json({ error: "Missing 'smiles'" });
    console.log(`[predict] Request: smiles="${smiles}", model="${model}"`);
    const args = ["inference.py", "--smiles", String(smiles)];
    if (model) args.push("--model", String(model));
    args.push("--json");
    console.log(`[predict] Calling Python with args: ${args.join(" ")}`);
    const { code, stdout, stderr } = await callPython(args, undefined, (s) => {
      // 打印所有stderr输出（包括调试信息）
      console.log(`[Python stderr] ${s.trim()}`);
      }, req,res
    );
    if (code !== 0) {
      console.error(`[predict] Python error (code ${code}):`, stderr);
      return res.status(500).json({ error: "Python error", stderr });
    }
    if (stderr) {
      console.log(`[predict] Python stderr output:`, stderr);
    }
    console.log("[predict] raw stdout head:", stdout.slice(0, 500));
    const data = JSON.parse(stdout.trim());
    console.log(`[predict] Result model: ${data.model}, prediction: ${data.prediction}`);
    return res.json(data);
  } catch (e: any) {
    console.error(`[predict] Exception:`, e);
    return res.status(500).json({ error: e?.message || "Unknown error" });
  }
});

const upload = multer({ dest: TMP_DIR });
// 批量预测端口，接收CSV/XLSX文件并创建后台任务
app.post("/batch_predict", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file" });
  const model = (req.body.model as string) || "baseline"; // 默认使用baseline，更稳定
  const id = crypto.randomUUID();
  const input = req.file.path;
  const output = path.join(TMP_DIR, `${id}-results.csv`);
  const job: Job = { id, state: "queued", progress: 0, input, output };
  jobs.set(id, job);
  res.json({ jobId: id });
  const uploadedFile = req.file;
  
  setImmediate(async () => {
    const job = jobs.get(id);
    if (!job) return;
    job.state = "running";
    jobs.set(id, job);
    const ext = path.extname(uploadedFile.originalname).toLowerCase();
    const args = ext === ".xlsx" 
      ? ["inference.py", "--xlsx", input, "--output", output, "--model", model]
      : ["inference.py", "--csv", input, "--output", output, "--model", model];
    const { code, stdout, stderr } = await callPython(args, (s) => {
      const m = /PROGRESS\s+(\d+)\/(\d+)/.exec(s);
      if (m) {
        const cur = Number(m[1]), tot = Number(m[2]);
        job.progress = tot ? cur / tot : 0;
        jobs.set(id, job);
      }
    });
    if (code === 0 && fs.existsSync(output)) {
      job.state = "done";
      job.progress = 1;
    } else {
      job.state = "error";
      job.error = stderr || "Unknown error";
      // 任务失败时清理文件
      cleanupJobFiles(job);
      jobs.delete(id);
    }
    jobs.set(id, job);
  });
});

// 查询任务状态的端口
app.get("/job/:id", (req, res) => {
  const j = jobs.get(String(req.params.id));
  if (!j) return res.status(404).json({ error: "Not found" });
  return res.json({ id: j.id, state: j.state, progress: j.progress, download: j.state === "done" ? `/job/${j.id}/download` : null, error: j.error || null });
});

// 清理任务文件的辅助函数
function cleanupJobFiles(job: Job) {
  try {
    // 清理输入文件
    if (job.input && fs.existsSync(job.input)) {
      fs.unlinkSync(job.input);
      console.log(`[cleanup] Deleted input file: ${job.input}`);
    }
    // 清理输出文件（在下载后或任务失败时）
    if (job.output && fs.existsSync(job.output)) {
      fs.unlinkSync(job.output);
      console.log(`[cleanup] Deleted output file: ${job.output}`);
    }
  } catch (err) {
    console.error(`[cleanup] Error cleaning up files for job ${job.id}:`, err);
  }
}

// 下载任务结果文件的端口
app.get("/job/:id/download", (req, res) => {
  const j = jobs.get(String(req.params.id));
  if (!j || j.state !== "done" || !fs.existsSync(j.output)) return res.status(404).send("Not ready");
  res.setHeader("Content-Type", "text/csv");
  res.setHeader("Content-Disposition", `attachment; filename="results-${j.id}.csv"`);
  
  // 下载完成后清理文件
  const stream = fs.createReadStream(j.output);
  stream.pipe(res);
  stream.on("end", () => {
    // 延迟清理，确保文件已完全发送
    setTimeout(() => {
      cleanupJobFiles(j);
      // 从 jobs Map 中移除已完成的任务
      jobs.delete(j.id);
    }, 1000);
  });
  stream.on("error", () => {
    res.status(500).send("Error reading file");
  });
});

// 获取模型注册表信息的端口
app.get("/models", async (_req, res) => {
  try {
    const regPath = path.join(ML_DIR, "saved_models", "registry.json");
    let registry: any = null;
    if (fs.existsSync(regPath)) {
      registry = JSON.parse(fs.readFileSync(regPath, "utf-8"));
    } else {
      registry = {
        models: [
          { name: "baseline", version: "v1", type: "LightGBM", metrics: { rmse: 0.5, r2: 0.85 } },
          { name: "gnn", version: "v1", type: "GIN", metrics: { rmse: 0.6, r2: 0.8 } }
        ],
        calibration: Array.from({ length: 30 }, (_, i) => ({ pred: Math.sin(i / 5) + 2, true: Math.sin(i / 5 + 0.1) + 2 }))
      };
    }
    res.json(registry);
  } catch (e: any) {
    res.status(500).json({ error: e?.message || "Error" });
  }
});

// 解释性预测端口，与predict类似但强调可解释性输出
app.post("/explain", async (req, res) => {
  // 与predict相同，但保留为显式端口
  const { smiles, model } = req.body || {};
  if (!smiles) return res.status(400).json({ error: "Missing 'smiles'" });
  const args = ["inference.py", "--smiles", String(smiles), "--json"];
  if (model) args.push("--model", String(model));
  const { code, stdout, stderr } = await callPython(args);
  if (code !== 0) return res.status(500).json({ error: "Python error", stderr });
  res.json(JSON.parse(stdout.trim()));
});

// 启动Express服务器
export function start() {
  app.listen(PORT, () => console.log(`[server] http://localhost:${PORT}`));
}

if (process.env.JEST_WORKER_ID === undefined) {
  start();
}

export default app;
