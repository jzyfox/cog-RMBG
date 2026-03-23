"""
批量背景移除脚本
基于 BRIA RMBG-1.4 模型，支持目录结构镜像、透明度检测与跳过、进度展示。

用法:
    python batch_rmbg.py --input_dir ./input --output_dir ./output
    python batch_rmbg.py --input_dir ./input --output_dir ./output --workers 4
"""

import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image


# ─────────────────────────────────────────────
# ① 模型全局实例（只加载一次，所有图片共享）
# ─────────────────────────────────────────────
_net: BriaRMBG | None = None
_device: torch.device | None = None


def load_model() -> None:
    """从 HuggingFace Hub 下载权重并加载模型到 GPU/CPU，只需调用一次。"""
    global _net, _device

    print("正在加载 BRIA RMBG-1.4 模型...")
    model_path = hf_hub_download("briaai/RMBG-1.4", "model.pth")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _net = BriaRMBG()
    _net.load_state_dict(torch.load(model_path, map_location=_device))
    _net.to(_device).eval()

    print(f"模型已加载，运行设备: {_device}\n")


def remove_background(input_path: Path, output_path: Path) -> None:
    """调用 BRIA RMBG-1.4 对单张图片进行背景移除，输出带 Alpha 通道的 PNG。"""
    assert _net is not None and _device is not None, "请先调用 load_model()"

    # 统一转为 RGB（处理 RGBA / 灰度 / 调色板等非标准格式）
    orig_im = np.array(Image.open(input_path).convert("RGB"))
    orig_im_size = orig_im.shape[0:2]

    img = preprocess_image(orig_im, [1024, 1024]).to(_device)

    with torch.no_grad():
        result = _net(img)

    result_image = postprocess_image(result[0][0], orig_im_size)

    pil_mask = Image.fromarray(result_image)
    no_bg = Image.new("RGBA", pil_mask.size, (0, 0, 0, 0))
    no_bg.paste(Image.open(input_path).convert("RGBA"), mask=pil_mask)
    no_bg.save(output_path)


# ─────────────────────────────────────────────
# ② 透明度检测
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def has_real_transparency(image: Image.Image) -> bool:
    """
    检查图片是否已完成背景移除（存在真正透明的像素）。

    判断标准：alpha < 10 的像素占全图 1% 以上。
    - alpha 253-254（近乎不透明）不算：这类图片通常是白背景 PNG，需要重新抠图
    - alpha 0-9（完全/接近透明）才算：真正被抠掉的背景
    """
    if image.mode not in ("RGBA", "LA", "PA"):
        return False

    alpha = np.array(image.getchannel("A"))
    total = alpha.size
    truly_transparent = int((alpha < 10).sum())
    return truly_transparent > total * 0.01


# ─────────────────────────────────────────────
# ③ 单张图片处理逻辑
# ─────────────────────────────────────────────
def process_image(input_path: Path, output_path: Path) -> str:
    """
    处理单张图片，返回处理结果标记：'skipped' | 'processed' | 'error:<msg>'
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(input_path) as img:
            already_transparent = has_real_transparency(img)

        if already_transparent:
            # 已是透明背景，直接复制，保留原文件名和格式
            dest = output_path.with_suffix(input_path.suffix)
            shutil.copy2(input_path, dest)
            return "skipped"
        else:
            # 需要抠图，输出统一为 .png（含 Alpha）
            dest = output_path.with_suffix(".png")
            remove_background(input_path, dest)
            return "processed"

    except Exception as exc:
        return f"error:{exc}"


# ─────────────────────────────────────────────
# ④ 目录遍历与结构镜像
# ─────────────────────────────────────────────
def collect_tasks(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """
    递归收集所有图片，并计算对应的输出路径（镜像目录结构）。
    返回 [(input_path, output_path), ...]
    """
    tasks = []
    for input_path in input_dir.rglob("*"):
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not input_path.is_file():
            continue

        relative = input_path.relative_to(input_dir)
        output_path = output_dir / relative
        tasks.append((input_path, output_path))

    return tasks


# ─────────────────────────────────────────────
# ⑤ 批处理主函数
# ─────────────────────────────────────────────
def batch_remove_background(
    input_dir: str | Path,
    output_dir: str | Path,
    workers: int = 1,
) -> None:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(input_dir, output_dir)
    if not tasks:
        print("未找到任何支持的图片文件，请检查输入目录。")
        return

    load_model()
    print(f"共发现 {len(tasks)} 张图片，开始处理...\n")

    stats = {"skipped": 0, "processed": 0, "error": 0}
    error_paths: list[str] = []

    def _run(args: tuple[Path, Path]) -> tuple[Path, str]:
        input_path, output_path = args
        result = process_image(input_path, output_path)
        return input_path, result

    executor_cls = ThreadPoolExecutor if workers > 1 else None

    with tqdm(total=len(tasks), unit="img", dynamic_ncols=True) as pbar:
        if executor_cls:
            with executor_cls(max_workers=workers) as executor:
                futures = {executor.submit(_run, t): t for t in tasks}
                for future in as_completed(futures):
                    input_path, result = future.result()
                    _update(pbar, stats, error_paths, input_path, result)
        else:
            for task in tasks:
                input_path, result = _run(task)
                _update(pbar, stats, error_paths, input_path, result)

    # 汇总报告
    print(f"\n{'─' * 50}")
    print(f"处理完成：")
    print(f"  ✓ 已抠图    {stats['processed']:>5} 张")
    print(f"  → 已跳过    {stats['skipped']:>5} 张（原本透明）")
    print(f"  ✗ 处理失败  {stats['error']:>5} 张")
    if error_paths:
        print("\n失败文件列表：")
        for p in error_paths:
            print(f"  {p}")
    print(f"{'─' * 50}")
    print(f"输出目录: {output_dir}")


def _update(
    pbar: tqdm,
    stats: dict,
    error_paths: list,
    input_path: Path,
    result: str,
) -> None:
    if result == "skipped":
        stats["skipped"] += 1
        pbar.set_postfix_str(f"跳过: {input_path.name}", refresh=False)
    elif result == "processed":
        stats["processed"] += 1
        pbar.set_postfix_str(f"完成: {input_path.name}", refresh=False)
    else:
        stats["error"] += 1
        msg = result.removeprefix("error:")
        error_paths.append(str(input_path))
        tqdm.write(f"[ERROR] {input_path}  →  {msg}")
    pbar.update(1)


# ─────────────────────────────────────────────
# ⑥ CLI 入口
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量背景移除工具（基于 BRIA RMBG-1.4）"
    )
    parser.add_argument("--input_dir", required=True, help="输入图片根目录")
    parser.add_argument("--output_dir", required=True, help="输出结果根目录")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发线程数（默认 1，GPU 推理通常不建议 > 1）",
    )
    args = parser.parse_args()

    batch_remove_background(args.input_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
