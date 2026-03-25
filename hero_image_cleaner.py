from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXCLUDED_DIR_NAME = "_excluded"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_EDGE_WIDTH = 10
DEFAULT_EDGE_BRIGHTNESS_THRESHOLD = 235.0
HERO_LABELS = [
    "a full body product shot of a furniture",
    "a close-up macro detail shot of furniture parts or textures",
]

ProgressCallback = Callable[[dict], None]

_clip_processor = None
_clip_model = None
_clip_device = None
_clip_model_id: str | None = None


def clean_hero_images(
    input_dir: str | Path,
    edge_width: int = DEFAULT_EDGE_WIDTH,
    edge_brightness_threshold: float = DEFAULT_EDGE_BRIGHTNESS_THRESHOLD,
    model_id: str = CLIP_MODEL_ID,
    labels: list[str] | None = None,
    show_tqdm: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Keep hero images in place and move rejected images into each folder's _excluded directory."""
    input_dir = Path(input_dir).resolve()
    labels = labels or HERO_LABELS

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹：{input_dir}")
    if edge_width <= 0:
        raise ValueError("edge_width 必须大于 0")
    if not 0 <= edge_brightness_threshold <= 255:
        raise ValueError("edge_brightness_threshold 必须在 0 到 255 之间")
    if len(labels) != 2:
        raise ValueError("labels 必须包含 2 条提示词")

    image_folders = _discover_image_folders(input_dir)
    if not image_folders:
        raise FileNotFoundError(f"在 {input_dir} 下未找到任何支持的图片文件")

    total_images = sum(len(files) for _, files in image_folders)
    stats = {
        "kept": 0,
        "excluded_edge": 0,
        "excluded_clip": 0,
        "broken": 0,
    }

    _emit(progress_callback, {
        "type": "start",
        "input_dir": str(input_dir),
        "folders": len(image_folders),
        "total": total_images,
        "edge_width": edge_width,
        "edge_brightness_threshold": edge_brightness_threshold,
        "excluded_dir_name": EXCLUDED_DIR_NAME,
    })

    current = 0
    clip_loaded = False
    device = "unused"

    with tqdm(
        total=total_images,
        desc="Hero 图片清洗",
        unit="img",
        dynamic_ncols=True,
        disable=not show_tqdm,
    ) as pbar:
        for folder, images in image_folders:
            excluded_dir = folder / EXCLUDED_DIR_NAME
            excluded_dir.mkdir(parents=True, exist_ok=True)
            _emit(progress_callback, {
                "type": "folder",
                "folder": folder.name,
                "path": str(folder),
                "count": len(images),
            })

            for image_path in images:
                current += 1
                edge_brightness = None
                hero_score = None
                detail_score = None
                confidence = None
                moved_to = None
                message = None
                status = "kept"

                try:
                    with Image.open(image_path) as image:
                        rgb_image = _to_white_rgb(image)
                        edge_brightness = _measure_edge_brightness(rgb_image, edge_width)

                        if edge_brightness < edge_brightness_threshold:
                            status = "excluded_edge"
                        else:
                            if not clip_loaded:
                                _emit(progress_callback, {
                                    "type": "log",
                                    "message": "正在加载 CLIP 模型（首次运行会下载权重）...",
                                })
                                _, _, device = load_clip_components(model_id=model_id)
                                clip_loaded = True
                                _emit(progress_callback, {
                                    "type": "log",
                                    "message": f"CLIP 模型加载完成，设备: {device}",
                                })

                            scores = predict_labels(
                                image=rgb_image,
                                labels=labels,
                                model_id=model_id,
                            )
                            hero_score = scores[0]
                            detail_score = scores[1]
                            confidence = max(scores)
                            status = "excluded_clip" if detail_score > hero_score else "kept"

                except Exception as exc:
                    status = "broken"
                    message = str(exc)

                if status == "kept":
                    stats["kept"] += 1
                else:
                    moved_to = str(_move_to_excluded(image_path, excluded_dir))
                    if status == "excluded_edge":
                        stats["excluded_edge"] += 1
                    elif status == "excluded_clip":
                        stats["excluded_clip"] += 1
                    else:
                        stats["broken"] += 1

                pbar.update(1)
                pbar.set_postfix_str(_postfix_text(status, image_path.name), refresh=False)

                _emit(progress_callback, {
                    "type": "progress",
                    "current": current,
                    "total": total_images,
                    "folder": folder.name,
                    "file": image_path.name,
                    "status": status,
                    "edge_brightness": _round_or_none(edge_brightness, 2),
                    "hero_score": _round_or_none(hero_score, 4),
                    "detail_score": _round_or_none(detail_score, 4),
                    "confidence": _round_or_none(confidence, 4),
                    "moved_to": moved_to,
                    "message": message,
                    "stats": dict(stats),
                })

    return {
        "input_dir": str(input_dir),
        "folders": len(image_folders),
        "items": total_images,
        "kept": stats["kept"],
        "excluded_edge": stats["excluded_edge"],
        "excluded_clip": stats["excluded_clip"],
        "broken": stats["broken"],
        "excluded_dir_name": EXCLUDED_DIR_NAME,
        "device": device,
    }


def load_clip_components(model_id: str = CLIP_MODEL_ID):
    global _clip_processor, _clip_model, _clip_device, _clip_model_id

    if _clip_model is not None and _clip_processor is not None and _clip_model_id == model_id:
        return _clip_processor, _clip_model, _clip_device

    import torch
    from transformers import CLIPModel, CLIPProcessor

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    _clip_processor = CLIPProcessor.from_pretrained(model_id)
    _clip_model = CLIPModel.from_pretrained(model_id).to(device)
    _clip_model.eval()
    _clip_device = device
    _clip_model_id = model_id
    return _clip_processor, _clip_model, _clip_device


def predict_labels(
    image: Image.Image,
    labels: list[str],
    model_id: str = CLIP_MODEL_ID,
) -> list[float]:
    import torch

    processor, model, device = load_clip_components(model_id=model_id)
    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1).squeeze(0)
    return [float(value) for value in probs.tolist()]


def _discover_image_folders(root: Path) -> list[tuple[Path, list[Path]]]:
    folders: list[tuple[Path, list[Path]]] = []
    candidate_dirs = sorted(
        path for path in root.rglob("*")
        if path.is_dir() and path.name != EXCLUDED_DIR_NAME and EXCLUDED_DIR_NAME not in path.parts
    )

    for folder in candidate_dirs:
        images = _list_images(folder)
        if images:
            folders.append((folder, images))

    if folders:
        return folders

    root_images = _list_images(root)
    return [(root, root_images)] if root_images else []


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _to_white_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def _measure_edge_brightness(image: Image.Image, edge_width: int) -> float:
    width, height = image.size
    strip = max(1, min(edge_width, width, height))
    samples = [
        np.asarray(image.crop((0, 0, width, strip)), dtype=np.float32),
        np.asarray(image.crop((0, height - strip, width, height)), dtype=np.float32),
        np.asarray(image.crop((0, 0, strip, height)), dtype=np.float32),
        np.asarray(image.crop((width - strip, 0, width, height)), dtype=np.float32),
    ]
    stacked = np.concatenate([sample.reshape(-1, 3) for sample in samples], axis=0)
    return float(stacked.mean())


def _move_to_excluded(source: Path, excluded_dir: Path) -> Path:
    excluded_dir.mkdir(parents=True, exist_ok=True)
    target = excluded_dir / source.name
    if not target.exists():
        shutil.move(str(source), str(target))
        return target

    index = 1
    while True:
        candidate = excluded_dir / f"{source.stem}_{index}{source.suffix}"
        if not candidate.exists():
            shutil.move(str(source), str(candidate))
            return candidate
        index += 1


def _emit(progress_callback: ProgressCallback | None, payload: dict) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _postfix_text(status: str, filename: str) -> str:
    if status == "kept":
        return f"保留: {filename}"
    if status == "excluded_edge":
        return f"边缘剔除: {filename}"
    if status == "excluded_clip":
        return f"CLIP 剔除: {filename}"
    return f"损坏: {filename}"


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗家具图库中的白底全貌 Hero 图片")
    parser.add_argument("--input_dir", required=True, help="根目录，递归扫描其中的品牌/型号子文件夹")
    parser.add_argument("--edge_width", type=int, default=DEFAULT_EDGE_WIDTH, help="边缘检测宽度，默认 10")
    parser.add_argument(
        "--edge_brightness_threshold",
        type=float,
        default=DEFAULT_EDGE_BRIGHTNESS_THRESHOLD,
        help="边缘平均亮度阈值，默认 235",
    )
    parser.add_argument("--model", default=CLIP_MODEL_ID, help="CLIP 模型 ID")
    args = parser.parse_args()

    summary = clean_hero_images(
        input_dir=args.input_dir,
        edge_width=args.edge_width,
        edge_brightness_threshold=args.edge_brightness_threshold,
        model_id=args.model,
        show_tqdm=True,
    )

    print(f"\n{'─' * 54}")
    print("Hero 图片清洗完成：")
    print(f"  保留                {summary['kept']:>6} 张")
    print(f"  边缘检测剔除        {summary['excluded_edge']:>6} 张")
    print(f"  CLIP 特写剔除       {summary['excluded_clip']:>6} 张")
    print(f"  损坏/异常剔除       {summary['broken']:>6} 张")
    print(f"  处理文件夹          {summary['folders']:>6} 个")
    print(f"  运行设备            {summary['device']}")
    print(f"{'─' * 54}")
    print(f"输入目录: {summary['input_dir']}")


if __name__ == "__main__":
    main()
