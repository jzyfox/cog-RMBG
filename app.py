"""
BRIA RMBG 批量抠图 - 本地 Web 界面
运行方式: python app.py
然后浏览器访问 http://localhost:5000
"""

import json
import queue
import shutil
import threading
import webbrowser
from pathlib import Path

from flask import Flask, Response, render_template_string, request
from PIL import Image

app = Flask(__name__)

_progress_queue: queue.Queue = queue.Queue()
_is_processing = False
_model_loaded = False

# ──────────────────────────────────────────────────────────────────────────────
# 页面 HTML
# ──────────────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BRIA RMBG · 批量抠图</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif; }
    .log-item { animation: fadeIn .2s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; } }
    .progress-bar { transition: width .4s ease; }
    input:focus { outline: none; box-shadow: 0 0 0 3px rgba(99,102,241,.25); }
  </style>
</head>
<body class="bg-gray-950 min-h-screen text-gray-100 flex flex-col">

  <!-- Header -->
  <header class="border-b border-gray-800 px-8 py-5 flex items-center gap-3">
    <div class="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center text-white font-bold text-sm">AI</div>
    <h1 class="text-lg font-semibold tracking-tight">BRIA RMBG &nbsp;·&nbsp; 批量背景移除</h1>
    <span id="status-badge" class="ml-auto text-xs px-3 py-1 rounded-full bg-gray-800 text-gray-400">空闲</span>
  </header>

  <!-- Main -->
  <main class="flex-1 px-8 py-8 max-w-3xl mx-auto w-full flex flex-col gap-6">

    <!-- 路径卡片 -->
    <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-4">
      <h2 class="text-sm font-medium text-gray-400 uppercase tracking-wider">目录设置</h2>

      <div class="flex flex-col gap-1.5">
        <label class="text-sm text-gray-300">输入文件夹 <span class="text-gray-600">（含原始图片）</span></label>
        <input id="input-dir" type="text" placeholder="例：C:/Users/xxx/images/input"
          class="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-600 transition" />
      </div>

      <div class="flex flex-col gap-1.5">
        <label class="text-sm text-gray-300">输出文件夹 <span class="text-gray-600">（抠图结果将保存在这里）</span></label>
        <input id="output-dir" type="text" placeholder="例：C:/Users/xxx/images/output"
          class="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-600 transition" />
      </div>

      <button id="start-btn" onclick="startProcessing()"
        class="mt-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed
               text-white font-medium rounded-xl px-6 py-3 text-sm transition-all duration-150 flex items-center justify-center gap-2">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        开始处理
      </button>
    </div>

    <!-- 进度卡片（初始隐藏） -->
    <div id="progress-card" class="hidden bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-4">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-medium text-gray-400 uppercase tracking-wider">处理进度</h2>
        <span id="progress-text" class="text-sm text-gray-400">0 / 0</span>
      </div>

      <!-- 进度条 -->
      <div class="bg-gray-800 rounded-full h-2.5 overflow-hidden">
        <div id="progress-bar" class="progress-bar h-full bg-indigo-500 rounded-full" style="width: 0%"></div>
      </div>

      <!-- 统计 -->
      <div class="grid grid-cols-3 gap-3 mt-1">
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <div id="stat-processed" class="text-2xl font-bold text-indigo-400">0</div>
          <div class="text-xs text-gray-500 mt-1">已抠图</div>
        </div>
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <div id="stat-skipped" class="text-2xl font-bold text-green-400">0</div>
          <div class="text-xs text-gray-500 mt-1">已跳过</div>
        </div>
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <div id="stat-error" class="text-2xl font-bold text-red-400">0</div>
          <div class="text-xs text-gray-500 mt-1">失败</div>
        </div>
      </div>

      <!-- 日志 -->
      <div id="log-area"
        class="bg-gray-950 border border-gray-800 rounded-xl p-3 h-48 overflow-y-auto flex flex-col gap-1 font-mono text-xs">
      </div>
    </div>

  </main>

  <script>
    let eventSource = null;

    function startProcessing() {
      const inputDir = document.getElementById('input-dir').value.trim();
      const outputDir = document.getElementById('output-dir').value.trim();

      if (!inputDir || !outputDir) {
        alert('请填写输入和输出文件夹路径');
        return;
      }

      // 重置 UI
      setProcessing(true);
      document.getElementById('progress-card').classList.remove('hidden');
      document.getElementById('log-area').innerHTML = '';
      setProgress(0, 0);
      setStats({ processed: 0, skipped: 0, error: 0 });

      // 启动后端处理
      fetch('/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_dir: inputDir, output_dir: outputDir })
      })
      .then(r => r.json())
      .then(data => {
        if (data.error) { alert(data.error); setProcessing(false); return; }
        listenProgress();
      })
      .catch(e => { alert('启动失败：' + e); setProcessing(false); });
    }

    function listenProgress() {
      if (eventSource) eventSource.close();
      eventSource = new EventSource('/stream');

      eventSource.onmessage = function(e) {
        const msg = JSON.parse(e.data);

        if (msg.type === 'ping') return;

        if (msg.type === 'start') {
          addLog(`发现 ${msg.total} 张图片，开始处理...`, 'text-gray-400');
          setProgress(0, msg.total);
        }

        if (msg.type === 'log') {
          addLog(msg.message, 'text-gray-400');
        }

        if (msg.type === 'progress') {
          setProgress(msg.current, msg.total);
          setStats(msg.stats);
          const color = msg.status === 'skipped' ? 'text-green-500'
                      : msg.status === 'error'   ? 'text-red-500'
                      : 'text-indigo-400';
          const icon  = msg.status === 'skipped' ? '→ 跳过'
                      : msg.status === 'error'   ? '✗ 失败'
                      : '✓ 完成';
          addLog(`${icon}  ${msg.file}`, color);
        }

        if (msg.type === 'error_item') {
          addLog(`✗ ${msg.file}  —  ${msg.message}`, 'text-red-500');
        }

        if (msg.type === 'done') {
          const s = msg.stats;
          addLog(`─────────────────────────────`, 'text-gray-700');
          addLog(`全部完成 · 抠图 ${s.processed} 张  跳过 ${s.skipped} 张  失败 ${s.error} 张`, 'text-white font-semibold');
          setStatus('完成', 'bg-green-900 text-green-400');
          setProcessing(false);
          eventSource.close();
        }

        if (msg.type === 'error') {
          addLog(`错误：${msg.message}`, 'text-red-400');
          setStatus('出错', 'bg-red-900 text-red-400');
          setProcessing(false);
          eventSource.close();
        }
      };
    }

    function setProgress(current, total) {
      const pct = total > 0 ? Math.round(current / total * 100) : 0;
      document.getElementById('progress-bar').style.width = pct + '%';
      document.getElementById('progress-text').textContent = `${current} / ${total}`;
    }

    function setStats(s) {
      document.getElementById('stat-processed').textContent = s.processed;
      document.getElementById('stat-skipped').textContent   = s.skipped;
      document.getElementById('stat-error').textContent     = s.error;
    }

    function setProcessing(active) {
      const btn = document.getElementById('start-btn');
      btn.disabled = active;
      btn.textContent = active ? '处理中...' : '开始处理';
      if (active) setStatus('处理中', 'bg-indigo-900 text-indigo-300');
      else if (document.getElementById('status-badge').textContent === '处理中')
        setStatus('空闲', 'bg-gray-800 text-gray-400');
    }

    function setStatus(text, cls) {
      const badge = document.getElementById('status-badge');
      badge.textContent = text;
      badge.className = `ml-auto text-xs px-3 py-1 rounded-full ${cls}`;
    }

    function addLog(text, colorClass) {
      const area = document.getElementById('log-area');
      const line = document.createElement('div');
      line.className = `log-item ${colorClass}`;
      line.textContent = text;
      area.appendChild(line);
      area.scrollTop = area.scrollHeight;
    }
  </script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────────
# 后端路由
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/start", methods=["POST"])
def start():
    global _is_processing, _progress_queue

    if _is_processing:
        return {"error": "正在处理中，请等待当前任务完成"}, 400

    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    output_dir = data.get("output_dir", "").strip()

    if not input_dir or not output_dir:
        return {"error": "请填写输入和输出目录"}, 400

    # 清空队列
    while not _progress_queue.empty():
        try:
            _progress_queue.get_nowait()
        except queue.Empty:
            break

    threading.Thread(target=_run_batch, args=(input_dir, output_dir), daemon=True).start()
    return {"status": "started"}


@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                msg = _progress_queue.get(timeout=20)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# 批处理逻辑（在后台线程中运行）
# ──────────────────────────────────────────────────────────────────────────────
def _run_batch(input_dir: str, output_dir: str) -> None:
    global _is_processing, _model_loaded

    _is_processing = True
    try:
        from batch_rmbg import (
            collect_tasks,
            has_real_transparency,
            load_model,
            remove_background,
        )

        input_path = Path(input_dir).resolve()
        output_path = Path(output_dir).resolve()

        if not input_path.exists():
            _progress_queue.put({"type": "error", "message": f"输入目录不存在：{input_dir}"})
            return

        output_path.mkdir(parents=True, exist_ok=True)
        tasks = collect_tasks(input_path, output_path)

        if not tasks:
            _progress_queue.put({"type": "error", "message": "未找到任何支持的图片文件（jpg / png / webp 等）"})
            return

        _progress_queue.put({"type": "start", "total": len(tasks)})

        if not _model_loaded:
            _progress_queue.put({"type": "log", "message": "正在加载 BRIA RMBG-1.4 模型（首次运行会下载权重，请稍候）..."})
            load_model()
            _model_loaded = True
            _progress_queue.put({"type": "log", "message": "模型加载完成 ✓"})

        stats = {"processed": 0, "skipped": 0, "error": 0}

        for i, (inp, out) in enumerate(tasks):
            status = "error"
            try:
                out.parent.mkdir(parents=True, exist_ok=True)

                with Image.open(inp) as img:
                    transparent = has_real_transparency(img)

                if transparent:
                    shutil.copy2(inp, out.with_suffix(inp.suffix))
                    stats["skipped"] += 1
                    status = "skipped"
                else:
                    remove_background(inp, out.with_suffix(".png"))
                    stats["processed"] += 1
                    status = "processed"

            except Exception as exc:
                stats["error"] += 1
                _progress_queue.put({"type": "error_item", "file": str(inp), "message": str(exc)})

            _progress_queue.put({
                "type": "progress",
                "current": i + 1,
                "total": len(tasks),
                "file": inp.name,
                "status": status,
                "stats": dict(stats),
            })

        _progress_queue.put({"type": "done", "stats": dict(stats)})

    except Exception as exc:
        _progress_queue.put({"type": "error", "message": str(exc)})
    finally:
        _is_processing = False


# ──────────────────────────────────────────────────────────────────────────────
# 启动
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def _open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open("http://localhost:5000")

    threading.Thread(target=_open_browser, daemon=True).start()
    print("启动中... 浏览器将自动打开 http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
