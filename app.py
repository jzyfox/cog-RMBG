"""
BRIA RMBG + CLIP 分类 - 本地 Web 界面
运行方式: python app.py
然后浏览器访问 http://localhost:5000
"""

import json
import io
import queue
import shutil
import threading
import webbrowser
from copy import deepcopy
from pathlib import Path

from asset_catalog_grid import DEFAULT_CATALOG_LAYOUT, normalize_catalog_type, normalize_layout_config
from flask import Flask, Response, abort, render_template_string, request, send_file
from hero_image_cleaner import DEFAULT_EDGE_BRIGHTNESS_THRESHOLD, DEFAULT_EDGE_WIDTH
from PIL import Image
from semantic_bundle_builder import (
    BUNDLE_FRONTEND_CONFIG,
    DEFAULT_BUNDLE_TARGET_COUNT,
    DEFAULT_SEED_CANDIDATE_COUNT,
    build_bundle_candidates,
    load_bundle_pool,
    load_bundle_review_data,
    regenerate_bundle_candidate,
    render_bundle_outputs,
    render_bundle_preview,
    save_bundle_review,
    seed_bundle_candidates,
)
from semantic_tagger import (
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_MAX_RETRIES as DEFAULT_SEMANTIC_MAX_RETRIES,
    DEFAULT_SLEEP_SECONDS as DEFAULT_SEMANTIC_SLEEP_SECONDS,
    build_semantic_frontend_config,
    build_semantic_tags,
    load_semantic_review_data,
    normalize_semantic_model,
    save_semantic_review_items,
)

app = Flask(__name__)

CATEGORIES_FILE = Path(__file__).parent / "categories.json"
CATALOG_LAYOUT_FILE = Path(__file__).parent / "catalog_layout.json"


def _load_categories() -> list[dict]:
    """读取已保存的分类配置，文件不存在时返回默认值。"""
    if CATEGORIES_FILE.exists():
        try:
            return json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_CATEGORIES


def _get_semantic_frontend_config(category_rows: list[dict] | None = None) -> dict:
    return build_semantic_frontend_config(category_rows if category_rows is not None else _load_categories())


def _resolve_semantic_request_category(raw_category: str | None, semantic_config: dict | None = None) -> str:
    config = semantic_config or _get_semantic_frontend_config()
    normalized = normalize_catalog_type(str(raw_category or "").strip().lower())
    if not normalized or normalized == "default":
        return str(config.get("default_category") or DEFAULT_CATEGORIES[0]["name"])

    category_items = {
        str(item.get("value") or ""): item
        for item in config.get("category_items", [])
        if isinstance(item, dict) and item.get("value")
    }
    matched_item = category_items.get(normalized)
    if matched_item and not bool(matched_item.get("supported")):
        raise ValueError(f"该品类暂未支持语义标签：{normalized}")
    return normalized


def _format_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _default_catalog_layout() -> dict:
    return deepcopy(DEFAULT_CATALOG_LAYOUT)


def _load_catalog_layout() -> dict:
    if CATALOG_LAYOUT_FILE.exists():
        try:
            return normalize_layout_config(
                json.loads(CATALOG_LAYOUT_FILE.read_text(encoding="utf-8"))
            )
        except Exception:
            pass
    return _default_catalog_layout()


# ── Hero 清洗任务状态 ────────────────────────────────────────────────────────────
_clean_queue: queue.Queue = queue.Queue()
_clean_processing = False

# ── 抠图任务状态 ──────────────────────────────────────────────────────────────
_rmbg_queue: queue.Queue = queue.Queue()
_rmbg_processing = False
_rmbg_model_loaded = False

# ── 分类任务状态 ──────────────────────────────────────────────────────────────
_classify_queue: queue.Queue = queue.Queue()
_classify_processing = False
_clip_processor = None
_clip_model = None
_clip_device = None
DEFAULT_CLASSIFY_THRESHOLD_STEP = 0.1
UNCERTAIN_CATEGORY = "uncertain"

_catalog_queue: queue.Queue = queue.Queue()
_catalog_processing = False

_semantic_queue: queue.Queue = queue.Queue()
_semantic_processing = False

_bundle_queue: queue.Queue = queue.Queue()
_bundle_processing = False

# ── 默认分类 ──────────────────────────────────────────────────────────────────
DEFAULT_CATEGORIES = [
    {"name": "sofa",         "prompt": "a product photo of a sofa or couch with cushions"},
    {"name": "bed",          "prompt": "a product photo of a bed with headboard and mattress"},
    {"name": "lounge_chair", "prompt": "a product photo of a lounge chair or armchair or accent chair"},
    {"name": "dining_chair", "prompt": "a product photo of a dining chair without armrests"},
    {"name": "coffee_table", "prompt": "a product photo of a low coffee table or side table"},
    {"name": "dining_table", "prompt": "a product photo of a dining table or desk"},
    {"name": "cabinet",      "prompt": "a product photo of a cabinet, sideboard, wardrobe or TV console"},
    {"name": "lamp",         "prompt": "a product photo of a floor lamp or table lamp or pendant light"},
    {"name": "shelf",        "prompt": "a product photo of a bookshelf or display shelf or rack"},
    {"name": "other",        "prompt": "a product photo of a decorative home furniture item"},
]

# ──────────────────────────────────────────────────────────────────────────────
# HTML 模板
# ──────────────────────────────────────────────────────────────────────────────
LEGACY_INLINE_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>家具 AI 工具箱</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: 'Inter','PingFang SC','Microsoft YaHei',sans-serif; }
    .log-item  { animation: fadeIn .15s ease; }
    @keyframes fadeIn { from{opacity:0;transform:translateY(3px)} to{opacity:1} }
    .progress-bar { transition: width .35s ease; }
    input[type=text],input[type=number] {
      background:#1f2937; border:1px solid #374151; border-radius:.5rem;
      padding:.6rem 1rem; font-size:.875rem; color:#f3f4f6; width:100%;
      transition:box-shadow .15s;
    }
    input[type=text]:focus,input[type=number]:focus {
      outline:none; box-shadow:0 0 0 3px rgba(99,102,241,.3);
    }
    input[type=range] { accent-color:#6366f1; cursor:pointer; }
    .tab-btn { transition: all .15s; }
    .tab-btn.active {
      background:#312e81; color:#a5b4fc; border-color:#4338ca;
    }
    .cat-row input { padding:.45rem .75rem; font-size:.8rem; }
    .size-row input { padding:.45rem .75rem; font-size:.8rem; }
  </style>
</head>
<body class="bg-gray-950 min-h-screen text-gray-100 flex flex-col">

<!-- ── Header ─────────────────────────────────────────────────────────────── -->
<header class="border-b border-gray-800 px-8 py-4 flex items-center gap-3">
  <div class="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center font-bold text-sm">AI</div>
  <h1 class="text-base font-semibold tracking-tight">家具 AI 工具箱</h1>

  <!-- Tab 切换 -->
  <nav class="ml-6 flex gap-2">
    <button id="tab-rmbg-btn" onclick="switchTab('rmbg')"
      class="tab-btn active text-xs px-4 py-1.5 rounded-lg border border-gray-700 text-gray-300">
      批量抠图
    </button>
    <button id="tab-classify-btn" onclick="switchTab('classify')"
      class="tab-btn text-xs px-4 py-1.5 rounded-lg border border-gray-700 text-gray-300">
      图片分类
    </button>
    <button id="tab-catalog-btn" onclick="switchTab('catalog')"
      class="tab-btn text-xs px-4 py-1.5 rounded-lg border border-gray-700 text-gray-300">
      资产拼图
    </button>
  </nav>

  <span id="status-badge" class="ml-auto text-xs px-3 py-1 rounded-full bg-gray-800 text-gray-400">空闲</span>
</header>

<!-- ── Main ───────────────────────────────────────────────────────────────── -->
<main class="flex-1 px-6 py-6 max-w-3xl mx-auto w-full">

  <!-- ════════════════════ 抠图面板 ════════════════════ -->
  <div id="panel-rmbg" class="flex flex-col gap-5">

    <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-4">
      <h2 class="text-xs font-medium text-gray-500 uppercase tracking-widest">目录设置</h2>

      <div class="flex flex-col gap-1.5">
        <label class="text-sm text-gray-300">输入文件夹 <span class="text-gray-600">（含原始图片）</span></label>
        <input id="rmbg-input-dir" type="text" placeholder="例：C:/Users/xxx/原图" />
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-sm text-gray-300">输出文件夹 <span class="text-gray-600">（抠图结果）</span></label>
        <input id="rmbg-output-dir" type="text" placeholder="例：C:/Users/xxx/抠图结果" />
      </div>

      <button id="rmbg-start-btn" onclick="startRmbg()"
        class="mt-1 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-600
               disabled:cursor-not-allowed text-white font-medium rounded-xl px-6 py-3 text-sm
               transition flex items-center justify-center gap-2">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        开始处理
      </button>
    </div>

    <!-- 抠图进度卡 -->
    <div id="rmbg-progress-card" class="hidden bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-4">
      <div class="flex justify-between items-center">
        <h2 class="text-xs font-medium text-gray-500 uppercase tracking-widest">处理进度</h2>
        <span id="rmbg-progress-text" class="text-sm text-gray-400">0 / 0</span>
      </div>
      <div class="bg-gray-800 rounded-full h-2 overflow-hidden">
        <div id="rmbg-progress-bar" class="progress-bar h-full bg-indigo-500 rounded-full" style="width:0%"></div>
      </div>
      <div class="grid grid-cols-3 gap-3">
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <div id="rmbg-stat-processed" class="text-2xl font-bold text-indigo-400">0</div>
          <div class="text-xs text-gray-500 mt-1">已抠图</div>
        </div>
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <div id="rmbg-stat-skipped" class="text-2xl font-bold text-green-400">0</div>
          <div class="text-xs text-gray-500 mt-1">已跳过</div>
        </div>
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <div id="rmbg-stat-error" class="text-2xl font-bold text-red-400">0</div>
          <div class="text-xs text-gray-500 mt-1">失败</div>
        </div>
      </div>
      <div id="rmbg-log"
        class="bg-gray-950 border border-gray-800 rounded-xl p-3 h-44 overflow-y-auto flex flex-col gap-0.5 font-mono text-xs">
      </div>
    </div>
  </div>

  <!-- ════════════════════ 分类面板 ════════════════════ -->
  <div id="panel-classify" class="hidden flex flex-col gap-5">

    <!-- 目录 + 阈值 -->
    <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-4">
      <h2 class="text-xs font-medium text-gray-500 uppercase tracking-widest">目录设置</h2>

      <div class="flex flex-col gap-1.5">
        <label class="text-sm text-gray-300">输入文件夹 <span class="text-gray-600">（待分类的透明 PNG）</span></label>
        <input id="cl-input-dir" type="text" placeholder="例：C:/Users/xxx/抠图结果" />
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-sm text-gray-300">输出文件夹 <span class="text-gray-600">（子文件夹将自动创建在这里）</span></label>
        <input id="cl-output-dir" type="text" placeholder="例：C:/Users/xxx/分类结果" />
      </div>

      <!-- 置信度阈值 -->
      <div class="flex flex-col gap-2">
        <div class="flex justify-between">
          <label class="text-sm text-gray-300">置信度阈值</label>
          <span id="threshold-label" class="text-sm text-indigo-400 font-mono">0%（不过滤）</span>
        </div>
        <input id="cl-threshold" type="range" min="0" max="95" step="5" value="0"
          oninput="updateThresholdLabel(this.value)"
          class="w-full h-1.5 rounded-full" />
        <p class="text-xs text-gray-600">低于阈值的图片将归入 <span class="text-gray-400">uncertain</span> 文件夹，供人工复查</p>
      </div>
    </div>

    <!-- 分类编辑器 -->
    <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <h2 class="text-xs font-medium text-gray-500 uppercase tracking-widest">目标分类</h2>
        <div class="flex gap-2">
          <button id="save-cat-btn" onclick="saveCategories()"
            class="text-xs text-green-400 hover:text-green-300 px-3 py-1 rounded-lg border border-green-900 hover:border-green-700 transition">
            保存配置
          </button>
          <button onclick="addCategoryRow()"
            class="text-xs text-indigo-400 hover:text-indigo-300 px-3 py-1 rounded-lg border border-indigo-800 hover:border-indigo-600 transition">
            + 添加分类
          </button>
        </div>
      </div>

      <!-- 表头 -->
      <div class="grid grid-cols-[140px_1fr_32px] gap-2 px-1">
        <span class="text-xs text-gray-600">文件夹名称</span>
        <span class="text-xs text-gray-600">CLIP 文本描述（英文，越具体越准）</span>
        <span></span>
      </div>

      <!-- 动态行列表 -->
      <div id="category-rows" class="flex flex-col gap-2"></div>
    </div>

    <!-- 开始按钮 -->
    <button id="cl-start-btn" onclick="startClassify()"
      class="bg-violet-600 hover:bg-violet-500 disabled:bg-gray-800 disabled:text-gray-600
             disabled:cursor-not-allowed text-white font-medium rounded-xl px-6 py-3 text-sm
             transition flex items-center justify-center gap-2">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
          d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
      </svg>
      开始分类
    </button>

    <!-- 分类进度卡 -->
    <div id="cl-progress-card" class="hidden bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-4">
      <div class="flex justify-between items-center">
        <h2 class="text-xs font-medium text-gray-500 uppercase tracking-widest">分类进度</h2>
        <span id="cl-progress-text" class="text-sm text-gray-400">0 / 0</span>
      </div>
      <div class="bg-gray-800 rounded-full h-2 overflow-hidden">
        <div id="cl-progress-bar" class="progress-bar h-full bg-violet-500 rounded-full" style="width:0%"></div>
      </div>

      <!-- 每类计数 -->
      <div id="cl-cat-stats" class="flex flex-wrap gap-2 min-h-[2rem]"></div>

      <div id="cl-log"
        class="bg-gray-950 border border-gray-800 rounded-xl p-3 h-52 overflow-y-auto flex flex-col gap-0.5 font-mono text-xs">
      </div>
    </div>
  </div>

  <div id="panel-catalog" class="hidden flex flex-col gap-5">

    <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-4">
      <h2 class="text-xs font-medium text-gray-500 uppercase tracking-widest">资产拼图设置</h2>

      <div class="flex flex-col gap-1.5">
        <label class="text-sm text-gray-300">输入文件夹<span class="text-gray-600">（包含各个套餐子目录）</span></label>
        <input id="cg-input-dir" type="text" placeholder="例如：C:/Users/xxx/classified_assets" />
      </div>

      <div class="flex flex-col gap-1.5">
        <label class="text-sm text-gray-300">输出文件夹<span class="text-gray-600">（每个子文件夹会生成一张图）</span></label>
        <input id="cg-output-dir" type="text" placeholder="例如：C:/Users/xxx/catalog_grids" />
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div class="flex flex-col gap-1.5">
          <label class="text-sm text-gray-300">画布宽度</label>
          <input id="cg-canvas-width" type="number" value="4096" min="512" step="256" />
        </div>
        <div class="flex flex-col gap-1.5">
          <label class="text-sm text-gray-300">间距 Padding</label>
          <input id="cg-padding" type="number" value="100" min="0" step="10" />
        </div>
        <div class="flex flex-col gap-1.5">
          <label class="text-sm text-gray-300">像素 / 厘米</label>
          <input id="cg-ppcm" type="number" value="10" min="0.1" step="0.5" />
        </div>
      </div>

      <div class="text-xs text-gray-500 leading-6 bg-gray-950 border border-gray-800 rounded-xl px-4 py-3">
        每个子文件夹会单独生成一张网格图，名称沿用子文件夹并追加
        <span class="font-mono text-gray-300">_grid.png</span>。
        文件名最后一个下划线后的后缀会被当作类别；真实物理尺寸映射表可以直接在
        <span class="font-mono text-gray-300">asset_catalog_grid.py</span> 顶部的
        <span class="font-mono text-gray-300">PHYSICAL_SIZES_CM</span> 中调整。
      </div>
    </div>

    <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <h2 class="text-xs font-medium text-gray-500 uppercase tracking-widest">物理尺寸映射</h2>
        <div class="flex gap-2">
          <button id="save-size-btn" onclick="savePhysicalSizes()"
            class="text-xs text-emerald-400 hover:text-emerald-300 px-3 py-1 rounded-lg border border-emerald-900 hover:border-emerald-700 transition">
            保存配置
          </button>
          <button onclick="addPhysicalSizeRow()"
            class="text-xs text-indigo-400 hover:text-indigo-300 px-3 py-1 rounded-lg border border-indigo-800 hover:border-indigo-600 transition">
            + 添加类别
          </button>
        </div>
      </div>

      <div class="grid grid-cols-[140px_110px_110px_32px] gap-2 px-1">
        <span class="text-xs text-gray-600">类别后缀</span>
        <span class="text-xs text-gray-600">宽度 cm</span>
        <span class="text-xs text-gray-600">高度 cm</span>
        <span></span>
      </div>

      <div id="physical-size-rows" class="flex flex-col gap-2"></div>
    </div>

    <button id="cg-start-btn" onclick="startCatalog()"
      class="bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-800 disabled:text-gray-600
             disabled:cursor-not-allowed text-white font-medium rounded-xl px-6 py-3 text-sm
             transition flex items-center justify-center gap-2">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
          d="M4 6h16M4 12h16M4 18h16"/>
      </svg>
      开始生成资产网格图
    </button>

    <div id="cg-progress-card" class="hidden bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col gap-4">
      <div class="flex justify-between items-center">
        <h2 class="text-xs font-medium text-gray-500 uppercase tracking-widest">拼图进度</h2>
        <span id="cg-progress-text" class="text-sm text-gray-400">0 / 0</span>
      </div>
      <div class="bg-gray-800 rounded-full h-2 overflow-hidden">
        <div id="cg-progress-bar" class="progress-bar h-full bg-emerald-500 rounded-full" style="width:0%"></div>
      </div>
      <div class="grid grid-cols-3 gap-3">
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <div id="cg-stat-rendered" class="text-2xl font-bold text-emerald-400">0</div>
          <div class="text-xs text-gray-500 mt-1">已处理</div>
        </div>
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <div id="cg-stat-packages" class="text-2xl font-bold text-sky-400">0</div>
          <div class="text-xs text-gray-500 mt-1">套餐数</div>
        </div>
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <div id="cg-stat-default" class="text-2xl font-bold text-yellow-400">0</div>
          <div class="text-xs text-gray-500 mt-1">默认尺寸</div>
        </div>
      </div>
      <div id="cg-result-path" class="text-xs text-gray-500 break-all min-h-[1rem]"></div>
      <div id="cg-log"
        class="bg-gray-950 border border-gray-800 rounded-xl p-3 h-52 overflow-y-auto flex flex-col gap-0.5 font-mono text-xs">
      </div>
    </div>
  </div>

</main>

<!-- ──────────────────────────── JavaScript ──────────────────────────────── -->
<script>
// ── 默认分类数据（从后端注入）────────────────────────────────────────────────
const DEFAULT_CATEGORIES = {{ default_categories | tojson }};
const DEFAULT_PHYSICAL_SIZES = {{ default_physical_sizes | tojson }};

// ── 标签页切换 ────────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.getElementById('panel-rmbg').classList.toggle('hidden', tab !== 'rmbg');
  document.getElementById('panel-classify').classList.toggle('hidden', tab !== 'classify');
  document.getElementById('panel-catalog').classList.toggle('hidden', tab !== 'catalog');
  document.getElementById('tab-rmbg-btn').classList.toggle('active', tab === 'rmbg');
  document.getElementById('tab-classify-btn').classList.toggle('active', tab === 'classify');
  document.getElementById('tab-catalog-btn').classList.toggle('active', tab === 'catalog');
}

// ── 阈值显示 ──────────────────────────────────────────────────────────────────
function updateThresholdLabel(val) {
  const v = parseInt(val);
  document.getElementById('threshold-label').textContent =
    v === 0 ? '0%（不过滤）' : v + '%';
}

// ── 分类行管理 ────────────────────────────────────────────────────────────────
function buildCategoryRow(name, prompt) {
  const row = document.createElement('div');
  row.className = 'cat-row grid grid-cols-[140px_1fr_32px] gap-2 items-center';
  row.innerHTML = `
    <input type="text" value="${escHtml(name)}"
      class="cat-name" placeholder="文件夹名" />
    <input type="text" value="${escHtml(prompt)}"
      class="cat-prompt" placeholder="CLIP 文本描述（英文）" />
    <button onclick="this.closest('.cat-row').remove()"
      class="w-8 h-8 flex items-center justify-center rounded-lg text-gray-600
             hover:text-red-400 hover:bg-gray-800 transition text-lg leading-none">×</button>`;
  return row;
}

function addCategoryRow(name='', prompt='') {
  document.getElementById('category-rows').appendChild(buildCategoryRow(name, prompt));
}

function getCategoriesFromUI() {
  const rows = document.querySelectorAll('.cat-row');
  const cats = [];
  rows.forEach(row => {
    const name   = row.querySelector('.cat-name').value.trim();
    const prompt = row.querySelector('.cat-prompt').value.trim();
    if (name && prompt) cats.push({ name, prompt });
  });
  return cats;
}

function buildPhysicalSizeRow(name, widthCm, heightCm) {
  const row = document.createElement('div');
  row.className = 'size-row grid grid-cols-[140px_110px_110px_32px] gap-2 items-center';
  row.innerHTML = `
    <input type="text" value="${escHtml(name)}"
      class="size-name" placeholder="category" />
    <input type="number" value="${widthCm ?? ''}"
      class="size-width" min="0.1" step="0.1" placeholder="width" />
    <input type="number" value="${heightCm ?? ''}"
      class="size-height" min="0.1" step="0.1" placeholder="height" />
    <button onclick="this.closest('.size-row').remove()"
      class="w-8 h-8 flex items-center justify-center rounded-lg text-gray-600
             hover:text-red-400 hover:bg-gray-800 transition text-lg leading-none">×</button>`;
  return row;
}

function addPhysicalSizeRow(name='', widthCm='', heightCm='') {
  document.getElementById('physical-size-rows').appendChild(
    buildPhysicalSizeRow(name, widthCm, heightCm)
  );
}

function getPhysicalSizesFromUI() {
  const rows = document.querySelectorAll('.size-row');
  const items = [];
  const seenNames = new Set();
  rows.forEach(row => {
    const name = row.querySelector('.size-name').value.trim().toLowerCase();
    const widthRaw = row.querySelector('.size-width').value.trim();
    const heightRaw = row.querySelector('.size-height').value.trim();
    const widthCm = parseFloat(widthRaw);
    const heightCm = parseFloat(heightRaw);
    if (!name && !widthRaw && !heightRaw) return;
    if (!name) throw new Error('物理尺寸表中存在未填写类别后缀的行');
    if (!Number.isFinite(widthCm) || !Number.isFinite(heightCm) || widthCm <= 0 || heightCm <= 0) {
      throw new Error(`类别 ${name} 的宽高必须是大于 0 的数字`);
    }
    if (seenNames.has(name)) throw new Error(`类别 ${name} 重复，请保留一条`);
    seenNames.add(name);
    items.push({ name, width_cm: widthCm, height_cm: heightCm });
  });
  return items;
}

// 初始化默认分类行
DEFAULT_CATEGORIES.forEach(c => addCategoryRow(c.name, c.prompt));
DEFAULT_PHYSICAL_SIZES.forEach(item => addPhysicalSizeRow(item.name, item.width_cm, item.height_cm));

function saveCategories() {
  const cats = getCategoriesFromUI();
  if (cats.length === 0) { alert('请至少保留一个分类'); return; }
  const btn = document.getElementById('save-cat-btn');
  btn.textContent = '保存中...'; btn.disabled = true;
  fetch('/classify/categories', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(cats)
  })
  .then(r => r.json())
  .then(d => {
    btn.textContent = `已保存 (${d.count} 项)`;
    btn.className = btn.className.replace('text-green-400','text-white').replace('border-green-900','border-green-600');
    setTimeout(() => {
      btn.textContent = '保存配置';
      btn.className = btn.className.replace('text-white','text-green-400').replace('border-green-600','border-green-900');
      btn.disabled = false;
    }, 2000);
  })
  .catch(() => { btn.textContent = '保存失败'; btn.disabled = false; });
}

function savePhysicalSizes() {
  let items = [];
  try {
    items = getPhysicalSizesFromUI();
  } catch (err) {
    alert(err.message || '尺寸配置有误');
    return;
  }
  const btn = document.getElementById('save-size-btn');
  btn.textContent = '保存中...'; btn.disabled = true;
  fetch('/catalog/physical_sizes', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(items)
  })
  .then(r => r.json())
  .then(d => {
    if (d.error) throw new Error(d.error);
    btn.textContent = `已保存(${d.count})`;
    btn.className = btn.className.replace('text-emerald-400','text-white').replace('border-emerald-900','border-emerald-600');
    setTimeout(() => {
      btn.textContent = '保存配置';
      btn.className = btn.className.replace('text-white','text-emerald-400').replace('border-emerald-600','border-emerald-900');
      btn.disabled = false;
    }, 2000);
  })
  .catch(err => {
    alert(err.message || '保存失败');
    btn.textContent = '保存失败';
    btn.disabled = false;
  });
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
}

// ════════════════════ 抠图逻辑 ════════════════════
let rmbgEventSource = null;

function startRmbg() {
  const inputDir  = document.getElementById('rmbg-input-dir').value.trim();
  const outputDir = document.getElementById('rmbg-output-dir').value.trim();
  if (!inputDir || !outputDir) { alert('请填写输入和输出文件夹路径'); return; }

  setRmbgProcessing(true);
  document.getElementById('rmbg-progress-card').classList.remove('hidden');
  document.getElementById('rmbg-log').innerHTML = '';
  setRmbgProgress(0, 0);
  document.getElementById('rmbg-stat-processed').textContent = '0';
  document.getElementById('rmbg-stat-skipped').textContent   = '0';
  document.getElementById('rmbg-stat-error').textContent     = '0';

  fetch('/start', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({input_dir: inputDir, output_dir: outputDir})
  })
  .then(r => r.json())
  .then(d => { if (d.error) { alert(d.error); setRmbgProcessing(false); return; } listenRmbg(); })
  .catch(e => { alert('启动失败：' + e); setRmbgProcessing(false); });
}

function listenRmbg() {
  if (rmbgEventSource) rmbgEventSource.close();
  rmbgEventSource = new EventSource('/stream');
  rmbgEventSource.onmessage = function(e) {
    const msg = JSON.parse(e.data);
    if (msg.type === 'ping') return;
    if (msg.type === 'start') { rmbgLog(`发现 ${msg.total} 张图片，开始处理...`, 'text-gray-500'); setRmbgProgress(0, msg.total); }
    if (msg.type === 'log')   { rmbgLog(msg.message, 'text-gray-500'); }
    if (msg.type === 'progress') {
      setRmbgProgress(msg.current, msg.total);
      document.getElementById('rmbg-stat-processed').textContent = msg.stats.processed;
      document.getElementById('rmbg-stat-skipped').textContent   = msg.stats.skipped;
      document.getElementById('rmbg-stat-error').textContent     = msg.stats.error;
      const color = msg.status==='skipped' ? 'text-green-500' : msg.status==='error' ? 'text-red-500' : 'text-indigo-400';
      const icon  = msg.status==='skipped' ? '→ 跳过' : msg.status==='error' ? '✗ 失败' : '✓ 完成';
      rmbgLog(`${icon}  ${msg.file}`, color);
    }
    if (msg.type === 'error_item') { rmbgLog(`✗ ${msg.file} — ${msg.message}`, 'text-red-500'); }
    if (msg.type === 'done')  {
      const s = msg.stats;
      rmbgLog('──────────────────────────────────────────', 'text-gray-800');
      rmbgLog(`完成 · 抠图 ${s.processed}  跳过 ${s.skipped}  失败 ${s.error}`, 'text-white');
      setStatus('完成', 'bg-green-900 text-green-400'); setRmbgProcessing(false); rmbgEventSource.close();
    }
    if (msg.type === 'error') {
      rmbgLog(`错误：${msg.message}`, 'text-red-400');
      setStatus('出错', 'bg-red-900 text-red-400'); setRmbgProcessing(false); rmbgEventSource.close();
    }
  };
}

function setRmbgProgress(cur, tot) {
  const pct = tot > 0 ? Math.round(cur/tot*100) : 0;
  document.getElementById('rmbg-progress-bar').style.width = pct + '%';
  document.getElementById('rmbg-progress-text').textContent = `${cur} / ${tot}`;
}

function setRmbgProcessing(active) {
  const btn = document.getElementById('rmbg-start-btn');
  btn.disabled = active;
  btn.childNodes[btn.childNodes.length-1].textContent = active ? ' 处理中...' : ' 开始处理';
  if (active) setStatus('处理中', 'bg-indigo-900 text-indigo-300');
  else if (document.getElementById('status-badge').textContent === '处理中')
    setStatus('空闲', 'bg-gray-800 text-gray-400');
}

function rmbgLog(text, cls) { addLog('rmbg-log', text, cls); }

// ════════════════════ 分类逻辑 ════════════════════
let clEventSource = null;
let clCatStats    = {};

function startClassify() {
  const inputDir   = document.getElementById('cl-input-dir').value.trim();
  const outputDir  = document.getElementById('cl-output-dir').value.trim();
  const threshold  = parseInt(document.getElementById('cl-threshold').value) / 100;
  const categories = getCategoriesFromUI();

  if (!inputDir || !outputDir) { alert('请填写输入和输出文件夹路径'); return; }
  if (categories.length === 0) { alert('请至少添加一个分类'); return; }

  setClProcessing(true);
  document.getElementById('cl-progress-card').classList.remove('hidden');
  document.getElementById('cl-log').innerHTML       = '';
  document.getElementById('cl-cat-stats').innerHTML = '';
  clCatStats = {};
  setClProgress(0, 0);

  fetch('/classify/start', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({input_dir: inputDir, output_dir: outputDir, threshold, categories})
  })
  .then(r => r.json())
  .then(d => { if (d.error) { alert(d.error); setClProcessing(false); return; } listenClassify(); })
  .catch(e => { alert('启动失败：' + e); setClProcessing(false); });
}

function listenClassify() {
  if (clEventSource) clEventSource.close();
  clEventSource = new EventSource('/classify/stream');
  clEventSource.onmessage = function(e) {
    const msg = JSON.parse(e.data);
    if (msg.type === 'ping') return;
    if (msg.type === 'start') { clLog(`发现 ${msg.total} 张图片，加载 CLIP 模型...`, 'text-gray-500'); setClProgress(0, msg.total); }
    if (msg.type === 'log')   { clLog(msg.message, 'text-gray-500'); }
    if (msg.type === 'progress') {
      setClProgress(msg.current, msg.total);
      updateCatStats(msg.category, msg.stats);
      const conf = (msg.confidence * 100).toFixed(0);
      const color = msg.category === 'uncertain' ? 'text-yellow-500' : 'text-violet-400';
      clLog(`→  ${msg.file.padEnd ? msg.file : msg.file}  →  ${msg.category}  (${conf}%)`, color);
    }
    if (msg.type === 'error_item') { clLog(`✗ ${msg.file} — ${msg.message}`, 'text-red-500'); }
    if (msg.type === 'done') {
      clLog('──────────────────────────────────────────', 'text-gray-800');
      const entries = Object.entries(msg.stats).map(([k,v]) => `${k}: ${v}`).join('  ');
      clLog(`完成 · ${entries}`, 'text-white');
      setStatus('完成', 'bg-green-900 text-green-400'); setClProcessing(false); clEventSource.close();
    }
    if (msg.type === 'error') {
      clLog(`错误：${msg.message}`, 'text-red-400');
      setStatus('出错', 'bg-red-900 text-red-400'); setClProcessing(false); clEventSource.close();
    }
  };
}

function updateCatStats(category, stats) {
  const container = document.getElementById('cl-cat-stats');
  let badge = document.getElementById('cat-badge-' + category);
  if (!badge) {
    badge = document.createElement('span');
    badge.id = 'cat-badge-' + category;
    const color = category === 'uncertain' ? 'bg-yellow-900 text-yellow-300'
                                           : 'bg-violet-900 text-violet-300';
    badge.className = `log-item ${color} text-xs px-2.5 py-1 rounded-full font-mono`;
    container.appendChild(badge);
  }
  badge.textContent = `${category}: ${stats[category] || 0}`;
}

function setClProgress(cur, tot) {
  const pct = tot > 0 ? Math.round(cur/tot*100) : 0;
  document.getElementById('cl-progress-bar').style.width = pct + '%';
  document.getElementById('cl-progress-text').textContent = `${cur} / ${tot}`;
}

function setClProcessing(active) {
  const btn = document.getElementById('cl-start-btn');
  btn.disabled = active;
  btn.childNodes[btn.childNodes.length-1].textContent = active ? ' 分类中...' : ' 开始分类';
  if (active) setStatus('分类中', 'bg-violet-900 text-violet-300');
  else if (document.getElementById('status-badge').textContent === '分类中')
    setStatus('空闲', 'bg-gray-800 text-gray-400');
}

function clLog(text, cls) { addLog('cl-log', text, cls); }

let cgEventSource = null;
let cgDefaultCount = 0;

function startCatalog() {
  const inputDir = document.getElementById('cg-input-dir').value.trim();
  const outputDir = document.getElementById('cg-output-dir').value.trim();
  const canvasWidth = parseInt(document.getElementById('cg-canvas-width').value, 10);
  const padding = parseInt(document.getElementById('cg-padding').value, 10);
  const pixelsPerCm = parseFloat(document.getElementById('cg-ppcm').value);
  let physicalSizes = [];
  try {
    physicalSizes = getPhysicalSizesFromUI();
  } catch (err) {
    alert(err.message || '尺寸配置有误');
    return;
  }

  if (!inputDir || !outputDir) { alert('请填写输入目录和输出文件夹'); return; }
  if (!Number.isFinite(canvasWidth) || canvasWidth <= 0) { alert('画布宽度必须大于 0'); return; }
  if (!Number.isFinite(padding) || padding < 0) { alert('Padding 不能小于 0'); return; }
  if (!Number.isFinite(pixelsPerCm) || pixelsPerCm <= 0) { alert('像素 / 厘米必须大于 0'); return; }

  setCgProcessing(true);
  cgDefaultCount = 0;
  document.getElementById('cg-progress-card').classList.remove('hidden');
  document.getElementById('cg-log').innerHTML = '';
  document.getElementById('cg-result-path').textContent = '';
  document.getElementById('cg-stat-rendered').textContent = '0';
  document.getElementById('cg-stat-packages').textContent = '0';
  document.getElementById('cg-stat-default').textContent = '0';
  setCgProgress(0, 0);

  fetch('/catalog/start', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      input_dir: inputDir,
      output_dir: outputDir,
      canvas_width: canvasWidth,
      padding,
      pixels_per_cm: pixelsPerCm,
      physical_sizes: physicalSizes
    })
  })
  .then(r => r.json())
  .then(d => { if (d.error) { alert(d.error); setCgProcessing(false); return; } listenCatalog(); })
  .catch(e => { alert('启动失败：' + e); setCgProcessing(false); });
}

function listenCatalog() {
  if (cgEventSource) cgEventSource.close();
  cgEventSource = new EventSource('/catalog/stream');
  cgEventSource.onmessage = function(e) {
    const msg = JSON.parse(e.data);
    if (msg.type === 'ping') return;
    if (msg.type === 'start') {
      document.getElementById('cg-stat-packages').textContent = msg.packages;
      document.getElementById('cg-result-path').textContent = `输出目录：${msg.output_dir}`;
      cgLog(`发现 ${msg.packages} 个套餐，共 ${msg.total} 张 PNG，开始逐个生成网格图...`, 'text-gray-400');
      setCgProgress(0, msg.total);
    }
    if (msg.type === 'package') {
      cgLog(`=== ${msg.package} === (${msg.count} 张)`, 'text-sky-400');
    }
    if (msg.type === 'log') {
      cgLog(msg.message, 'text-gray-500');
    }
    if (msg.type === 'progress') {
      setCgProgress(msg.current, msg.total);
      document.getElementById('cg-stat-rendered').textContent = msg.current;
      if (msg.used_default_size) {
        cgDefaultCount += 1;
        document.getElementById('cg-stat-default').textContent = cgDefaultCount;
        cgLog(`默认尺寸  ${msg.file}  ->  ${msg.category}  (${msg.target_width_px}x${msg.target_height_px})`, 'text-yellow-400');
      } else {
        cgLog(`已布局  ${msg.file}  ->  ${msg.category}  (${msg.target_width_px}x${msg.target_height_px})`, 'text-emerald-400');
      }
    }
    if (msg.type === 'package_done') {
      cgLog(`已输出  ${msg.package}_grid.png  (${msg.items} 张, ${msg.canvas_width}x${msg.canvas_height})`, 'text-sky-300');
    }
    if (msg.type === 'done') {
      document.getElementById('cg-stat-rendered').textContent = msg.items;
      document.getElementById('cg-stat-packages').textContent = msg.packages;
      document.getElementById('cg-stat-default').textContent = msg.default_size_count;
      document.getElementById('cg-result-path').textContent = `输出目录：${msg.output_dir}`;
      cgLog('────────────────────────────────────────', 'text-gray-700');
      cgLog(`完成 · 生成 ${msg.packages} 张拼图 · 物品 ${msg.items}`, 'text-white');
      if (msg.default_size_count > 0) {
        const detail = Object.entries(msg.default_categories || {}).map(([k, v]) => `${k}: ${v}`).join('  ');
        if (detail) cgLog(`默认尺寸类别：${detail}`, 'text-yellow-400');
      }
      cgLog(`输出目录：${msg.output_dir}`, 'text-gray-300');
      setStatus('完成', 'bg-green-900 text-green-400');
      setCgProcessing(false);
      cgEventSource.close();
    }
    if (msg.type === 'error') {
      cgLog(`错误：${msg.message}`, 'text-red-400');
      setStatus('出错', 'bg-red-900 text-red-400');
      setCgProcessing(false);
      cgEventSource.close();
    }
  };
}

function setCgProgress(cur, tot) {
  const pct = tot > 0 ? Math.round(cur / tot * 100) : 0;
  document.getElementById('cg-progress-bar').style.width = pct + '%';
  document.getElementById('cg-progress-text').textContent = `${cur} / ${tot}`;
}

function setCgProcessing(active) {
  const btn = document.getElementById('cg-start-btn');
  btn.disabled = active;
  btn.childNodes[btn.childNodes.length - 1].textContent = active ? ' 正在生成资产网格图...' : ' 开始生成资产网格图';
  if (active) setStatus('拼图中', 'bg-emerald-900 text-emerald-300');
  else if (document.getElementById('status-badge').textContent === '拼图中')
    setStatus('空闲', 'bg-gray-800 text-gray-400');
}

function cgLog(text, cls) { addLog('cg-log', text, cls); }

// ── 通用工具 ──────────────────────────────────────────────────────────────────
function setStatus(text, cls) {
  const b = document.getElementById('status-badge');
  b.textContent = text;
  b.className = `ml-auto text-xs px-3 py-1 rounded-full ${cls}`;
}

function addLog(areaId, text, cls) {
  const area = document.getElementById(areaId);
  const line = document.createElement('div');
  line.className = `log-item ${cls}`;
  line.textContent = text;
  area.appendChild(line);
  area.scrollTop = area.scrollHeight;
}
</script>
</body>
</html>
"""

# Runtime template lives in app_template.html. The legacy inline template is kept
# above only to avoid a large in-place string deletion in a dirty worktree.
HTML = (Path(__file__).parent / "app_template.html").read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# 路由
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    categories = _load_categories()
    semantic_config = _get_semantic_frontend_config(categories)
    return render_template_string(
        HTML,
        default_categories=categories,
        default_catalog_layout=_load_catalog_layout(),
        default_semantic_config=semantic_config,
        default_bundle_config=BUNDLE_FRONTEND_CONFIG,
        default_bundle_target_count=DEFAULT_BUNDLE_TARGET_COUNT,
        default_seed_candidate_count=DEFAULT_SEED_CANDIDATE_COUNT,
        default_semantic_sleep_seconds=DEFAULT_SEMANTIC_SLEEP_SECONDS,
        default_semantic_max_retries=DEFAULT_SEMANTIC_MAX_RETRIES,
        default_classify_threshold_step=int(DEFAULT_CLASSIFY_THRESHOLD_STEP * 100),
        default_clean_edge_width=DEFAULT_EDGE_WIDTH,
        default_clean_edge_threshold=int(DEFAULT_EDGE_BRIGHTNESS_THRESHOLD),
    )


@app.route("/classify/categories", methods=["POST"])
def save_categories():
    cats = request.json or []
    if not isinstance(cats, list) or not cats:
        return {"error": "分类列表为空"}, 400
    CATEGORIES_FILE.write_text(json.dumps(cats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "saved", "count": len(cats)}


# ── Hero 清洗 / 拼图 / 抠图路由 ────────────────────────────────────────────────
@app.route("/catalog/layout", methods=["POST"])
def save_catalog_layout():
    try:
        layout = normalize_layout_config(request.json or {})
    except ValueError as exc:
        return {"error": str(exc)}, 400

    CATALOG_LAYOUT_FILE.write_text(
        json.dumps(layout, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"status": "saved", "count": len(layout["boxes"])}


@app.route("/clean/start", methods=["POST"])
def start_clean():
    global _clean_processing

    if _clean_processing:
        return {"error": "Hero 清洗任务正在进行中，请等待完成"}, 400

    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()

    try:
        edge_width = int(data.get("edge_width", DEFAULT_EDGE_WIDTH))
        edge_brightness_threshold = float(
            data.get("edge_brightness_threshold", DEFAULT_EDGE_BRIGHTNESS_THRESHOLD)
        )
    except (TypeError, ValueError):
        return {"error": "边缘检测宽度和亮度阈值必须是合法数字"}, 400

    if not input_dir:
        return {"error": "请填写输入目录"}, 400
    if edge_width <= 0:
        return {"error": "边缘检测宽度必须大于 0"}, 400
    if not 0 <= edge_brightness_threshold <= 255:
        return {"error": "边缘亮度阈值必须在 0 到 255 之间"}, 400

    _drain(_clean_queue)
    threading.Thread(
        target=_run_clean,
        args=(input_dir, edge_width, edge_brightness_threshold),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.route("/clean/stream")
def stream_clean():
    return _sse_response(_clean_queue)


@app.route("/start", methods=["POST"])
def start_rmbg():
    global _rmbg_processing

    if _rmbg_processing:
        return {"error": "抠图任务正在进行中，请等待完成"}, 400

    data = request.json or {}
    input_dir  = data.get("input_dir", "").strip()
    output_dir = data.get("output_dir", "").strip()
    if not input_dir or not output_dir:
        return {"error": "请填写输入和输出目录"}, 400

    _drain(_rmbg_queue)
    threading.Thread(target=_run_rmbg, args=(input_dir, output_dir), daemon=True).start()
    return {"status": "started"}


@app.route("/stream")
def stream_rmbg():
    return _sse_response(_rmbg_queue)


# ── 分类路由 ──────────────────────────────────────────────────────────────────
@app.route("/classify/start", methods=["POST"])
def start_classify():
    global _classify_processing

    if _classify_processing:
        return {"error": "分类任务正在进行中，请等待完成"}, 400

    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    output_dir = data.get("output_dir", "").strip()
    categories = data.get("categories", [])

    try:
        threshold = float(data.get("threshold", 0))
        threshold_step = float(data.get("threshold_step", DEFAULT_CLASSIFY_THRESHOLD_STEP))
    except (TypeError, ValueError):
        return {"error": "置信度阈值和每轮降幅必须是合法数字"}, 400

    raw_auto_retry_uncertain = data.get("auto_retry_uncertain", False)
    if isinstance(raw_auto_retry_uncertain, str):
        auto_retry_uncertain = raw_auto_retry_uncertain.strip().lower() in {"1", "true", "yes", "on"}
    else:
        auto_retry_uncertain = bool(raw_auto_retry_uncertain)

    if not input_dir or not output_dir:
        return {"error": "请填写输入和输出目录"}, 400
    if not categories:
        return {"error": "请至少提供一个分类"}, 400
    if threshold < 0 or threshold > 1:
        return {"error": "置信度阈值必须在 0 到 1 之间"}, 400
    if threshold_step <= 0 or threshold_step > 1:
        return {"error": "每轮降幅必须大于 0 且不超过 1"}, 400

    _drain(_classify_queue)
    threading.Thread(
        target=_run_classify,
        args=(input_dir, output_dir, categories, threshold, auto_retry_uncertain, threshold_step),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.route("/classify/stream")
def stream_classify():
    return _sse_response(_classify_queue)


@app.route("/semantic/start", methods=["POST"])
def start_semantic():
    global _semantic_processing

    if _semantic_processing:
        return {"error": "语义标签任务正在进行中，请稍后再试"}, 400

    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    model = data.get("model", "").strip() or DEFAULT_SEMANTIC_MODEL
    semantic_config = _get_semantic_frontend_config()

    try:
        category = _resolve_semantic_request_category(data.get("category", ""), semantic_config)
        model = normalize_semantic_model(model)
        skip_existing = bool(data.get("skip_existing", True))
        sleep_seconds = float(data.get("sleep_seconds", DEFAULT_SEMANTIC_SLEEP_SECONDS))
        max_retries = int(data.get("max_retries", DEFAULT_SEMANTIC_MAX_RETRIES))
    except Exception as exc:
        return {"error": str(exc)}, 400

    if not input_dir:
        return {"error": "请填写输入目录"}, 400

    _drain(_semantic_queue)
    threading.Thread(
        target=_run_semantic,
        args=(input_dir, category, model, skip_existing, sleep_seconds, max_retries),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.route("/semantic/stream")
def stream_semantic():
    return _sse_response(_semantic_queue)


@app.route("/semantic/load", methods=["POST"])
def load_semantic():
    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()

    if not input_dir:
        return {"error": "请填写输入目录"}, 400

    try:
        category = _resolve_semantic_request_category(data.get("category", ""))
        payload = load_semantic_review_data(input_dir=input_dir, category=category)
    except Exception as exc:
        return {"error": str(exc)}, 400
    return payload


@app.route("/semantic/save", methods=["POST"])
def save_semantic():
    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    items = data.get("items", [])

    if not input_dir:
        return {"error": "请填写输入目录"}, 400

    try:
        category = _resolve_semantic_request_category(data.get("category", ""))
        payload = save_semantic_review_items(
            input_dir=input_dir,
            category=category,
            items=items,
        )
    except Exception as exc:
        return {"error": str(exc)}, 400
    return payload


@app.route("/semantic/image")
def semantic_image():
    raw_path = request.args.get("path", "").strip()
    if not raw_path:
        abort(404)

    path = Path(raw_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        abort(404)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        abort(404)
    return send_file(path)


@app.route("/bundle/start", methods=["POST"])
def start_bundle():
    global _bundle_processing

    if _bundle_processing:
        return {"error": "自动组套任务正在进行中，请稍后再试"}, 400

    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()

    try:
        target_count = int(data.get("target_count", DEFAULT_BUNDLE_TARGET_COUNT))
    except (TypeError, ValueError):
        return {"error": "目标数量必须是合法整数"}, 400

    if not input_dir:
        return {"error": "请填写输入目录"}, 400

    _drain(_bundle_queue)
    threading.Thread(
        target=_run_bundle,
        args=(input_dir, target_count),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.route("/bundle/stream")
def stream_bundle():
    return _sse_response(_bundle_queue)


@app.route("/bundle/pool", methods=["POST"])
def bundle_pool():
    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    category = data.get("category")

    if not input_dir:
        return {"error": "请填写输入目录"}, 400

    try:
        payload = load_bundle_pool(input_dir=input_dir, category=category)
    except Exception as exc:
        return {"error": str(exc)}, 400
    return payload


@app.route("/bundle/seed", methods=["POST"])
def bundle_seed():
    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    seed_image_path = data.get("seed_image_path", "").strip()

    try:
        candidate_count = int(data.get("candidate_count", DEFAULT_SEED_CANDIDATE_COUNT))
    except (TypeError, ValueError):
        return {"error": "候选数量必须是合法整数"}, 400

    if not input_dir:
        return {"error": "请填写输入目录"}, 400
    if not seed_image_path:
        return {"error": "请先选择种子图片"}, 400

    try:
        payload = seed_bundle_candidates(
            input_dir=input_dir,
            seed_image_path=seed_image_path,
            candidate_count=candidate_count,
        )
    except Exception as exc:
        return {"error": str(exc)}, 400
    return payload


@app.route("/bundle/load", methods=["POST"])
def bundle_load():
    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()

    if not input_dir:
        return {"error": "请填写输入目录"}, 400

    try:
        payload = load_bundle_review_data(input_dir=input_dir)
    except Exception as exc:
        return {"error": str(exc)}, 400
    return payload


@app.route("/bundle/review", methods=["POST"])
def bundle_review():
    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    action = str(data.get("action", "")).strip().lower()

    if not input_dir:
        return {"error": "请填写输入目录"}, 400

    try:
        if action == "regenerate":
            bundle_id = str(data.get("bundle_id", "")).strip()
            if not bundle_id:
                return {"error": "缺少 bundle_id"}, 400
            payload = regenerate_bundle_candidate(input_dir=input_dir, bundle_id=bundle_id)
        else:
            payload = save_bundle_review(
                input_dir=input_dir,
                bundles=data.get("bundles", []),
            )
    except Exception as exc:
        return {"error": str(exc)}, 400
    return payload


@app.route("/bundle/preview", methods=["POST"])
def bundle_preview():
    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    bundle_id = str(data.get("bundle_id", "")).strip()

    if not input_dir:
        return {"error": "请填写输入目录"}, 400
    if not bundle_id:
        return {"error": "缺少 bundle_id"}, 400

    try:
        payload = render_bundle_preview(
            input_dir=input_dir,
            bundle_id=bundle_id,
            layout=_load_catalog_layout(),
        )
    except Exception as exc:
        return {"error": str(exc)}, 400

    return send_file(io.BytesIO(payload), mimetype="image/png", download_name=f"{bundle_id}.png")


@app.route("/bundle/render", methods=["POST"])
def bundle_render():
    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    output_dir = data.get("output_dir", "").strip()
    bundle_ids = data.get("bundle_ids")

    if not input_dir or not output_dir:
        return {"error": "请填写输入目录和输出目录"}, 400

    try:
        payload = render_bundle_outputs(
            input_dir=input_dir,
            output_dir=output_dir,
            bundle_ids=bundle_ids if isinstance(bundle_ids, list) else None,
            layout=_load_catalog_layout(),
        )
    except Exception as exc:
        return {"error": str(exc)}, 400
    return payload


@app.route("/catalog/start", methods=["POST"])
def start_catalog():
    global _catalog_processing

    if _catalog_processing:
        return {"error": "资产拼图任务正在进行中，请稍后再试"}, 400

    data = request.json or {}
    input_dir = data.get("input_dir", "").strip()
    output_dir = data.get("output_dir", "").strip() or data.get("output_path", "").strip()

    try:
        layout = normalize_layout_config(data.get("layout"))
    except ValueError as exc:
        return {"error": str(exc) or "参数格式不正确"}, 400

    if not input_dir or not output_dir:
        return {"error": "请填写输入目录和输出文件夹"}, 400

    _drain(_catalog_queue)
    threading.Thread(
        target=_run_catalog,
        args=(input_dir, output_dir, layout),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.route("/catalog/stream")
def stream_catalog():
    return _sse_response(_catalog_queue)


# ──────────────────────────────────────────────────────────────────────────────
# 后台任务：Hero 清洗 / 抠图 / 分类 / 拼图
# ──────────────────────────────────────────────────────────────────────────────
def _run_clean(input_dir: str, edge_width: int, edge_brightness_threshold: float) -> None:
    global _clean_processing

    _clean_processing = True
    try:
        from hero_image_cleaner import clean_hero_images

        def _emit(payload: dict) -> None:
            _clean_queue.put(payload)

        summary = clean_hero_images(
            input_dir=input_dir,
            edge_width=edge_width,
            edge_brightness_threshold=edge_brightness_threshold,
            show_tqdm=False,
            progress_callback=_emit,
        )
        _clean_queue.put({"type": "done", **summary})
    except Exception as exc:
        _clean_queue.put({"type": "error", "message": str(exc)})
    finally:
        _clean_processing = False


def _run_rmbg(input_dir: str, output_dir: str) -> None:
    global _rmbg_processing, _rmbg_model_loaded

    _rmbg_processing = True
    try:
        from batch_rmbg import collect_tasks, has_real_transparency, load_model, remove_background

        inp = Path(input_dir).resolve()
        out = Path(output_dir).resolve()

        if not inp.exists():
            _rmbg_queue.put({"type": "error", "message": f"输入目录不存在：{input_dir}"})
            return

        out.mkdir(parents=True, exist_ok=True)
        tasks = collect_tasks(inp, out)

        if not tasks:
            _rmbg_queue.put({"type": "error", "message": "未找到任何支持的图片文件"})
            return

        _rmbg_queue.put({"type": "start", "total": len(tasks)})

        if not _rmbg_model_loaded:
            _rmbg_queue.put({"type": "log", "message": "正在加载 BRIA RMBG-1.4 模型（首次运行会下载权重）..."})
            load_model()
            _rmbg_model_loaded = True
            _rmbg_queue.put({"type": "log", "message": "模型加载完成"})

        stats = {"processed": 0, "skipped": 0, "error": 0}

        for i, (src, dst) in enumerate(tasks):
            status = "error"
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(src) as img:
                    transparent = has_real_transparency(img)
                if transparent:
                    shutil.copy2(src, dst.with_suffix(src.suffix))
                    stats["skipped"] += 1
                    status = "skipped"
                else:
                    remove_background(src, dst.with_suffix(".png"))
                    stats["processed"] += 1
                    status = "processed"
            except Exception as exc:
                stats["error"] += 1
                _rmbg_queue.put({"type": "error_item", "file": str(src), "message": str(exc)})

            _rmbg_queue.put({
                "type": "progress", "current": i + 1, "total": len(tasks),
                "file": src.name, "status": status, "stats": dict(stats),
            })

        _rmbg_queue.put({"type": "done", "stats": dict(stats)})

    except Exception as exc:
        _rmbg_queue.put({"type": "error", "message": str(exc)})
    finally:
        _rmbg_processing = False


# ──────────────────────────────────────────────────────────────────────────────
# 后台任务：分类
# ──────────────────────────────────────────────────────────────────────────────
def _run_classify(
    input_dir: str,
    output_dir: str,
    categories: list[dict],
    threshold: float,
    auto_retry_uncertain: bool = False,
    threshold_step: float = DEFAULT_CLASSIFY_THRESHOLD_STEP,
) -> None:
    global _classify_processing, _clip_processor, _clip_model, _clip_device

    _classify_processing = True
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        inp = Path(input_dir).resolve()
        out = Path(output_dir).resolve()

        if not inp.exists():
            _classify_queue.put({"type": "error", "message": f"输入目录不存在：{input_dir}"})
            return

        images = sorted(
            p for p in inp.rglob("*")
            if p.is_file() and p.suffix.lower() == ".png"
        )
        if not images:
            all_files = [p for p in inp.rglob("*") if p.is_file()]
            exts = sorted({p.suffix.lower() for p in all_files}) or ["（无文件）"]
            _classify_queue.put({"type": "error", "message": (
                f"在 {inp} 中未找到任何 PNG 文件。"
                f"共扫描到 {len(all_files)} 个文件，"
                f"扩展名：{', '.join(exts)}"
            )})
            return

        _classify_queue.put({
            "type": "start",
            "total": len(images),
            "threshold": threshold,
            "auto_retry_uncertain": auto_retry_uncertain,
            "threshold_step": threshold_step,
        })

        if _clip_model is None:
            _classify_queue.put({"type": "log", "message": "正在加载 CLIP 模型（首次运行会下载权重）..."})
            model_id = "openai/clip-vit-base-patch32"
            device = _select_clip_device(torch)
            _clip_processor = CLIPProcessor.from_pretrained(model_id)
            _clip_model = CLIPModel.from_pretrained(model_id).to(device)
            _clip_model.eval()
            _clip_device = device
            _classify_queue.put({"type": "log", "message": f"模型加载完成，设备: {device}"})

        cat_names = [c["name"] for c in categories]
        text_prompts = [c["prompt"] for c in categories]
        stats: dict[str, int] = {"error": 0}
        out.mkdir(parents=True, exist_ok=True)
        current_assignments: dict[str, str] = {}
        pending_tasks: list[dict] = [
            {
                "item_id": str(img_path.relative_to(inp).as_posix()),
                "source_path": img_path,
                "relative_dir": img_path.parent.relative_to(inp),
                "base_stem": _strip_classification_suffix(img_path.stem, cat_names),
                "suffix": img_path.suffix,
                "from_output": False,
            }
            for img_path in images
        ]
        current_threshold = threshold
        pass_index = 0
        scope = "all"

        while pending_tasks:
            pass_index += 1
            _classify_queue.put({
                "type": "pass_start",
                "pass_index": pass_index,
                "scope": scope,
                "threshold": current_threshold,
                "total": len(pending_tasks),
                "remaining_uncertain": len(pending_tasks) if scope == "uncertain" else 0,
            })

            next_pending_tasks: list[dict] = []

            for i, task in enumerate(pending_tasks, start=1):
                img_path = Path(task["source_path"])
                try:
                    with Image.open(img_path) as raw:
                        rgb_img = _to_white_rgb(raw)

                    inputs = _clip_processor(
                        text=text_prompts,
                        images=rgb_img,
                        return_tensors="pt",
                        padding=True,
                    ).to(_clip_device)

                    with torch.no_grad():
                        probs = _clip_model(**inputs).logits_per_image.softmax(dim=1).squeeze()

                    best_idx = int(probs.argmax().item())
                    confidence = float(probs[best_idx].item())
                    category = cat_names[best_idx] if confidence >= current_threshold else UNCERTAIN_CATEGORY

                    dest_dir = out / Path(task["relative_dir"])
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / _build_classified_name(
                        base_stem=str(task["base_stem"]),
                        category=category,
                        suffix=str(task["suffix"]),
                    )

                    if img_path != dest_path:
                        shutil.copy2(img_path, dest_path)
                        if bool(task.get("from_output")) and img_path.exists():
                            img_path.unlink()

                    previous_category = current_assignments.get(str(task["item_id"]))
                    if previous_category is None:
                        stats[category] = stats.get(category, 0) + 1
                    elif previous_category != category:
                        previous_count = stats.get(previous_category, 0)
                        if previous_count > 0:
                            stats[previous_category] = previous_count - 1
                        stats[category] = stats.get(category, 0) + 1
                    current_assignments[str(task["item_id"])] = category

                    if category == UNCERTAIN_CATEGORY:
                        next_pending_tasks.append({
                            **task,
                            "source_path": dest_path,
                            "from_output": True,
                        })

                    _classify_queue.put({
                        "type": "progress",
                        "current": i,
                        "total": len(pending_tasks),
                        "file": img_path.name,
                        "category": category,
                        "confidence": confidence,
                        "stats": dict(stats),
                        "pass_index": pass_index,
                        "threshold": current_threshold,
                    })

                except Exception as exc:
                    stats["error"] = stats.get("error", 0) + 1
                    _classify_queue.put({
                        "type": "error_item",
                        "file": img_path.name,
                        "message": str(exc),
                        "pass_index": pass_index,
                        "threshold": current_threshold,
                    })
                    _classify_queue.put({
                        "type": "progress",
                        "current": i,
                        "total": len(pending_tasks),
                        "file": img_path.name,
                        "category": "error",
                        "confidence": 0.0,
                        "stats": dict(stats),
                        "pass_index": pass_index,
                        "threshold": current_threshold,
                    })

            _classify_queue.put({
                "type": "pass_done",
                "pass_index": pass_index,
                "scope": scope,
                "threshold": current_threshold,
                "total": len(pending_tasks),
                "remaining_uncertain": stats.get(UNCERTAIN_CATEGORY, 0),
                "stats": dict(stats),
            })

            if (not auto_retry_uncertain) or (not next_pending_tasks) or current_threshold <= 0:
                break

            current_threshold = max(0.0, current_threshold - threshold_step)
            pending_tasks = next_pending_tasks
            scope = "uncertain"

        _classify_queue.put({
            "type": "done",
            "stats": dict(stats),
            "passes": pass_index,
            "threshold": current_threshold,
            "remaining_uncertain": stats.get(UNCERTAIN_CATEGORY, 0),
        })

    except Exception as exc:
        _classify_queue.put({"type": "error", "message": str(exc)})
    finally:
        _classify_processing = False


def _run_catalog(
    input_dir: str,
    output_dir: str,
    layout: dict,
) -> None:
    global _catalog_processing

    _catalog_processing = True
    try:
        from asset_catalog_grid import build_catalog_grids

        def _emit(payload: dict) -> None:
            _catalog_queue.put(payload)

        summary = build_catalog_grids(
            input_dir=input_dir,
            output_dir=output_dir,
            layout=layout,
            progress_callback=_emit,
        )
        _catalog_queue.put({"type": "done", **summary})

    except Exception as exc:
        _catalog_queue.put({"type": "error", "message": str(exc)})
    finally:
        _catalog_processing = False


def _run_semantic(
    input_dir: str,
    category: str,
    model: str,
    skip_existing: bool,
    sleep_seconds: float,
    max_retries: int,
) -> None:
    global _semantic_processing

    _semantic_processing = True
    try:
        def _emit(payload: dict) -> None:
            _semantic_queue.put(payload)

        summary = build_semantic_tags(
            input_dir=input_dir,
            category=category,
            model=model,
            skip_existing=skip_existing,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            progress_callback=_emit,
        )
        _semantic_queue.put({"type": "done", **summary})
    except Exception as exc:
        _semantic_queue.put({"type": "error", "message": str(exc)})
    finally:
        _semantic_processing = False


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────
def _run_bundle(
    input_dir: str,
    target_count: int,
) -> None:
    global _bundle_processing

    _bundle_processing = True
    try:
        def _emit(payload: dict) -> None:
            _bundle_queue.put(payload)

        summary = build_bundle_candidates(
            input_dir=input_dir,
            target_count=target_count,
            progress_callback=_emit,
        )
        _bundle_queue.put({"type": "done", **summary})
    except Exception as exc:
        _bundle_queue.put({"type": "error", "message": str(exc)})
    finally:
        _bundle_processing = False


def _to_white_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def _select_clip_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _strip_classification_suffix(stem: str, category_names: list[str]) -> str:
    original = str(stem or "")
    lowered = original.lower()
    suffixes = sorted(
        {str(name or "").strip().lower() for name in [*category_names, UNCERTAIN_CATEGORY] if str(name or "").strip()},
        key=len,
        reverse=True,
    )
    for suffix in suffixes:
        token = f"_{suffix}"
        if lowered.endswith(token):
            trimmed = original[:-len(token)]
            return trimmed or original
    return original


def _build_classified_name(base_stem: str, category: str, suffix: str) -> str:
    normalized_base_stem = str(base_stem or "").strip() or "item"
    return f"{normalized_base_stem}_{category}{suffix}"


def _drain(q: queue.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _sse_response(q: queue.Queue) -> Response:
    def generate():
        while True:
            try:
                msg = q.get(timeout=20)
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
