from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageColor, ImageDraw, ImageFont


# Internal baseline widths in centimeters. These preserve the previous scale
# semantics when global_scale=1.0 and category_scale=1.0.
BASE_WIDTHS_CM: dict[str, float] = {
    "sofa": 220,
    "bed": 180,
    "lounge_chair": 90,
    "dining_chair": 50,
    "chair": 60,
    "coffee_table": 120,
    "side_table": 55,
    "dining_table": 160,
    "table": 140,
    "cabinet": 180,
    "wardrobe": 180,
    "lamp": 45,
    "shelf": 100,
    "other": 100,
    "default": 100,
}

PIXELS_PER_CM = 10
DEFAULT_CATEGORY_SCALE = 1.0
DEFAULT_CANVAS_WIDTH = 4096
DEFAULT_PADDING = 100
DEFAULT_BACKGROUND = "#f5f5f5"
SUPPORTED_EXTENSIONS = {".png"}

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE = Image.LANCZOS


ProgressCallback = Callable[[dict], None]


@dataclass(frozen=True)
class PreparedItem:
    package_name: str
    source_path: Path
    category: str
    bbox: tuple[int, int, int, int]
    target_size: tuple[int, int]
    base_width_cm: float
    category_scale: float
    used_default_config: bool


@dataclass(frozen=True)
class PackageGroup:
    name: str
    items: list[PreparedItem]


@dataclass(frozen=True)
class TitlePlacement:
    text: str
    x: int
    y: int


@dataclass(frozen=True)
class ItemPlacement:
    item: PreparedItem
    x: int
    y: int


def build_catalog_grids(
    input_dir: str | Path,
    output_dir: str | Path,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    padding: int = DEFAULT_PADDING,
    global_scale: float = 1.0,
    background_color: str = DEFAULT_BACKGROUND,
    category_scales: dict[str, float] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Build one catalog grid image for each package subdirectory."""
    input_dir = Path(input_dir).resolve()
    output_dir, file_like_output = _normalize_output_dir(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if canvas_width <= 0:
        raise ValueError("canvas_width must be greater than 0")
    if padding < 0:
        raise ValueError("padding must be >= 0")
    if global_scale <= 0:
        raise ValueError("global_scale must be greater than 0")

    normalized_category_scales = _normalize_category_scales(category_scales or {})
    package_files = _discover_package_files(input_dir)
    total_items = sum(len(files) for _, files in package_files)
    if total_items == 0:
        raise FileNotFoundError(f"No PNG files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress_callback, {
        "type": "start",
        "total": total_items,
        "packages": len(package_files),
        "output_dir": str(output_dir),
        "canvas_width": canvas_width,
        "padding": padding,
        "global_scale": global_scale,
    })
    if file_like_output:
        _emit(progress_callback, {
            "type": "log",
            "message": (
                "Output target looked like a file path. "
                "Using its parent directory because one grid is generated per package."
            ),
        })

    current = 0
    total_default_count = 0
    total_default_categories: dict[str, int] = {}
    package_summaries: list[dict] = []

    for package_name, files in package_files:
        _emit(progress_callback, {
            "type": "package",
            "package": package_name,
            "count": len(files),
        })

        prepared_items: list[PreparedItem] = []
        package_default_count = 0
        package_default_categories: dict[str, int] = {}

        for file_path in files:
            prepared = _prepare_item(
                package_name=package_name,
                image_path=file_path,
                global_scale=global_scale,
                category_scales=normalized_category_scales,
            )
            prepared_items.append(prepared)
            current += 1

            if prepared.used_default_config:
                package_default_count += 1
                total_default_count += 1
                package_default_categories[prepared.category] = (
                    package_default_categories.get(prepared.category, 0) + 1
                )
                total_default_categories[prepared.category] = (
                    total_default_categories.get(prepared.category, 0) + 1
                )

            _emit(progress_callback, {
                "type": "progress",
                "current": current,
                "total": total_items,
                "package": package_name,
                "file": file_path.name,
                "category": prepared.category,
                "used_default_config": prepared.used_default_config,
                "target_width_px": prepared.target_size[0],
                "target_height_px": prepared.target_size[1],
                "category_scale": prepared.category_scale,
            })

        package_group = PackageGroup(name=package_name, items=prepared_items)
        package_summary = _render_package_grid(
            package_group=package_group,
            output_dir=output_dir,
            canvas_width=canvas_width,
            padding=padding,
            background_color=background_color,
            progress_callback=progress_callback,
        )
        package_summary["default_config_count"] = package_default_count
        package_summary["default_categories"] = package_default_categories
        package_summaries.append(package_summary)

        _emit(progress_callback, {
            "type": "package_done",
            **package_summary,
        })

    return {
        "output_dir": str(output_dir),
        "packages": len(package_summaries),
        "items": total_items,
        "generated_files": [item["output_path"] for item in package_summaries],
        "default_config_count": total_default_count,
        "default_categories": total_default_categories,
        "package_summaries": package_summaries,
    }


def build_catalog_grid(
    input_dir: str | Path,
    output_path: str | Path,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    padding: int = DEFAULT_PADDING,
    global_scale: float = 1.0,
    background_color: str = DEFAULT_BACKGROUND,
    category_scales: dict[str, float] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Backward-compatible wrapper for callers using the old function name."""
    return build_catalog_grids(
        input_dir=input_dir,
        output_dir=output_path,
        canvas_width=canvas_width,
        padding=padding,
        global_scale=global_scale,
        background_color=background_color,
        category_scales=category_scales,
        progress_callback=progress_callback,
    )


def _render_package_grid(
    package_group: PackageGroup,
    output_dir: Path,
    canvas_width: int,
    padding: int,
    background_color: str,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    widest_item = max(item.target_size[0] for item in package_group.items)
    effective_canvas_width = max(canvas_width, widest_item + padding * 2)
    if effective_canvas_width != canvas_width:
        _emit(progress_callback, {
            "type": "log",
            "message": (
                f"Package {package_group.name}: requested canvas width {canvas_width}px is too small. "
                f"Expanded to {effective_canvas_width}px to avoid clipping."
            ),
        })

    title_font = _load_font(max(24, padding // 2))
    title_color = ImageColor.getrgb("#334155")
    background_rgba = ImageColor.getrgb(background_color) + (255,)

    placements, title, total_height = _layout_package(
        package_group=package_group,
        canvas_width=effective_canvas_width,
        padding=padding,
        title_font=title_font,
    )

    canvas = Image.new("RGBA", (effective_canvas_width, total_height), background_rgba)
    draw = ImageDraw.Draw(canvas)
    draw.text((title.x, title.y), title.text, fill=title_color, font=title_font)

    for placement in placements:
        with Image.open(placement.item.source_path) as source_image:
            rgba = source_image.convert("RGBA")
            cropped = rgba.crop(placement.item.bbox)
            resized = cropped.resize(placement.item.target_size, RESAMPLE)
            canvas.alpha_composite(resized, dest=(placement.x, placement.y))

    output_path = output_dir / f"{package_group.name}_grid.png"
    canvas.save(output_path)

    return {
        "package": package_group.name,
        "output_path": str(output_path),
        "items": len(package_group.items),
        "canvas_width": effective_canvas_width,
        "canvas_height": total_height,
    }


def _prepare_item(
    package_name: str,
    image_path: Path,
    global_scale: float,
    category_scales: dict[str, float],
) -> PreparedItem:
    category = _parse_category(image_path.stem)
    has_base_width = category in BASE_WIDTHS_CM
    has_category_scale = category in category_scales
    base_width_cm = BASE_WIDTHS_CM.get(category, BASE_WIDTHS_CM["default"])
    category_scale = category_scales.get(category, category_scales.get("default", DEFAULT_CATEGORY_SCALE))
    used_default_config = (not has_base_width) or (not has_category_scale)

    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
        alpha_channel = rgba.getchannel("A")
        bbox = alpha_channel.getbbox()
        if bbox is None:
            bbox = (0, 0, rgba.width, rgba.height)

        cropped_width = max(1, bbox[2] - bbox[0])
        cropped_height = max(1, bbox[3] - bbox[1])

    target_width = max(
        1,
        round(base_width_cm * PIXELS_PER_CM * global_scale * category_scale),
    )
    target_height = max(1, round(cropped_height * target_width / cropped_width))

    return PreparedItem(
        package_name=package_name,
        source_path=image_path,
        category=category,
        bbox=bbox,
        target_size=(target_width, target_height),
        base_width_cm=base_width_cm,
        category_scale=category_scale,
        used_default_config=used_default_config,
    )


def _discover_package_files(input_dir: Path) -> list[tuple[str, list[Path]]]:
    package_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    package_files: list[tuple[str, list[Path]]] = []

    if package_dirs:
        for package_dir in package_dirs:
            files = sorted(
                path for path in package_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if files:
                package_files.append((package_dir.name, files))
        return package_files

    root_files = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if root_files:
        return [(input_dir.name, root_files)]
    return []


def _layout_package(
    package_group: PackageGroup,
    canvas_width: int,
    padding: int,
    title_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> tuple[list[ItemPlacement], TitlePlacement, int]:
    placements: list[ItemPlacement] = []
    current_y = padding
    title_gap = max(24, padding // 3)
    title_text = f"=== {package_group.name} ==="
    title = TitlePlacement(text=title_text, x=padding, y=current_y)
    current_y += _measure_text_height(title_text, title_font) + title_gap

    row_items: list[tuple[PreparedItem, int]] = []
    row_height = 0
    current_x = padding

    def flush_row() -> None:
        nonlocal current_y, current_x, row_items, row_height
        if not row_items:
            return
        baseline_y = current_y + row_height
        for row_item, row_x in row_items:
            placements.append(ItemPlacement(
                item=row_item,
                x=row_x,
                y=baseline_y - row_item.target_size[1],
            ))
        current_y = baseline_y + padding
        current_x = padding
        row_items = []
        row_height = 0

    for item in package_group.items:
        item_width, item_height = item.target_size
        if row_items and current_x + item_width > canvas_width - padding:
            flush_row()

        row_items.append((item, current_x))
        row_height = max(row_height, item_height)
        current_x += item_width + padding

    flush_row()

    total_height = max(current_y, padding * 2)
    return placements, title, total_height


def _parse_category(stem: str) -> str:
    if "_" not in stem:
        return "default"
    return stem.rsplit("_", 1)[-1].strip().lower() or "default"


def _normalize_output_dir(output_target: str | Path) -> tuple[Path, bool]:
    path = Path(output_target).expanduser()
    file_like_output = bool(path.suffix)
    if file_like_output:
        path = path.parent
    return path.resolve(), file_like_output


def _measure_text_height(
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> int:
    box = font.getbbox(text)
    return max(1, box[3] - box[1])


def _normalize_category_scales(category_scales: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for category, scale in category_scales.items():
        if not category:
            continue
        scale = float(scale)
        if scale <= 0:
            continue
        normalized[str(category).strip().lower()] = scale
    return normalized


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [
        windows_fonts / "msyh.ttc",
        windows_fonts / "msyhbd.ttc",
        windows_fonts / "simhei.ttf",
        windows_fonts / "simsun.ttc",
        windows_fonts / "arial.ttf",
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
    ]
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _emit(progress_callback: ProgressCallback | None, payload: dict) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one furniture asset grid per package subdirectory.",
    )
    parser.add_argument("--input_dir", required=True, help="Root directory that contains package subfolders.")
    parser.add_argument("--output_dir", help="Directory where package_name_grid.png files will be written.")
    parser.add_argument("--output_path", help=argparse.SUPPRESS)
    parser.add_argument("--canvas_width", type=int, default=DEFAULT_CANVAS_WIDTH, help="Requested canvas width in pixels.")
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING, help="Padding between assets in pixels.")
    parser.add_argument("--global_scale", type=float, default=1.0, help="Global multiplier for the whole catalog.")
    parser.add_argument("--background", default=DEFAULT_BACKGROUND, help="Canvas background color, e.g. #f5f5f5.")
    args = parser.parse_args()

    output_target = args.output_dir or args.output_path or "catalog_grids"
    summary = build_catalog_grids(
        input_dir=args.input_dir,
        output_dir=output_target,
        canvas_width=args.canvas_width,
        padding=args.padding,
        global_scale=args.global_scale,
        background_color=args.background,
    )
    print(f"Saved {summary['packages']} grid images to: {summary['output_dir']}")
    for path in summary["generated_files"]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
