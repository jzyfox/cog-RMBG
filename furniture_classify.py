"""
furniture_classify.py
使用 CLIP (openai/clip-vit-base-patch32) 对透明背景家具 PNG 图片自动分类。

用法:
    python furniture_classify.py --input_dir ./furniture --output_dir ./sorted
    python furniture_classify.py --input_dir ./furniture --output_dir ./sorted --threshold 0.4
"""

import argparse
import shutil
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ──────────────────────────────────────────────────────────────────────────────
# ① 分类定义
#    key   = 子文件夹名称
#    value = 送给 CLIP 的英文文本提示（越具体识别率越高）
# ──────────────────────────────────────────────────────────────────────────────
CATEGORIES: dict[str, str] = {
    "sofa":         "a product photo of a sofa or couch with cushions",
    "bed":          "a product photo of a bed with headboard and mattress",
    "lounge_chair": "a product photo of a lounge chair or armchair or accent chair",
    "dining_chair": "a product photo of a dining chair without armrests",
    "coffee_table": "a product photo of a low coffee table or side table",
    "dining_table": "a product photo of a dining table or desk",
    "cabinet":      "a product photo of a cabinet, sideboard, wardrobe or TV console",
    "lamp":         "a product photo of a floor lamp or table lamp or pendant light",
    "shelf":        "a product photo of a bookshelf or display shelf or rack",
    "other":        "a product photo of a decorative home furniture item",
}


# ──────────────────────────────────────────────────────────────────────────────
# ② 模型加载（只加载一次）
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_id: str = "openai/clip-vit-base-patch32"):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"加载 CLIP 模型 ({model_id})  →  设备: {device}")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    print("模型加载完成 ✓\n")
    return processor, model, device


# ──────────────────────────────────────────────────────────────────────────────
# ③ 图像预处理：RGBA → 白底 RGB
#    直接把 RGBA 送给 CLIP 会让透明区域变成黑色，严重干扰识别
# ──────────────────────────────────────────────────────────────────────────────
def to_white_rgb(image_path: Path) -> Image.Image:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])   # split()[3] 是 Alpha 通道
        return background
    return img.convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# ④ 单张推理
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(
    image: Image.Image,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    text_prompts: list[str],
) -> tuple[int, float]:
    """
    返回 (最佳类别索引, 置信度概率)
    """
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze()   # shape: (n_classes,)
    best_idx = int(probs.argmax().item())
    confidence = float(probs[best_idx].item())
    return best_idx, confidence


# ──────────────────────────────────────────────────────────────────────────────
# ⑤ 批量分类主函数
# ──────────────────────────────────────────────────────────────────────────────
def classify_furniture(
    input_dir: str | Path,
    output_dir: str | Path,
    model_id: str = "openai/clip-vit-base-patch32",
    threshold: float = 0.0,
) -> None:
    """
    input_dir  : 存放所有待分类透明 PNG 的文件夹（只扫描顶层文件）
    output_dir : 分类结果输出目录，会自动创建子文件夹
    threshold  : 置信度阈值，低于此值归入 "uncertain" 文件夹，0 表示不过滤
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 只扫描顶层 PNG（大小写均兼容）
    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )
    if not images:
        print("未找到任何 PNG 文件，请检查输入目录。")
        return

    print(f"共发现 {len(images)} 张 PNG，开始分类...\n")

    categories = list(CATEGORIES.keys())
    text_prompts = list(CATEGORIES.values())

    processor, model, device = load_model(model_id)

    stats: dict[str, int] = {}
    errors: list[str] = []

    for img_path in tqdm(images, desc="分类进度", unit="张"):
        try:
            rgb_img = to_white_rgb(img_path)
            idx, confidence = predict(rgb_img, processor, model, device, text_prompts)
            category = categories[idx]

            # 置信度不足时归入 uncertain
            if threshold > 0 and confidence < threshold:
                category = "uncertain"

            dest_dir = output_dir / category
            dest_dir.mkdir(exist_ok=True)
            shutil.move(str(img_path), dest_dir / img_path.name)

            stats[category] = stats.get(category, 0) + 1
            tqdm.write(f"  {img_path.name:<40}  →  {category}  ({confidence:.1%})")

        except Exception as exc:
            errors.append(f"{img_path.name}: {exc}")
            tqdm.write(f"[跳过] {img_path.name} — {exc}")

    # ── 汇总报告 ──────────────────────────────────
    print(f"\n{'─' * 50}")
    print("分类完成：")
    for cat in list(CATEGORIES.keys()) + ["uncertain"]:
        count = stats.get(cat, 0)
        if count:
            print(f"  {cat:<18} {count:>4} 张")
    if errors:
        print(f"\n处理失败 {len(errors)} 张：")
        for e in errors:
            print(f"  {e}")
    print(f"{'─' * 50}")
    print(f"输出目录: {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# ⑥ CLI 入口
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP 家具自动分类工具")
    parser.add_argument("--input_dir",  required=True, help="待分类透明 PNG 所在文件夹")
    parser.add_argument("--output_dir", required=True, help="分类结果输出目录")
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="CLIP 模型 ID（默认 openai/clip-vit-base-patch32）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="置信度阈值 0~1，低于此值归入 uncertain 文件夹（默认 0，不过滤）",
    )
    args = parser.parse_args()
    classify_furniture(args.input_dir, args.output_dir, args.model, args.threshold)


if __name__ == "__main__":
    main()
