from __future__ import annotations

import argparse
import copy
import io
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, ImageColor, ImageDraw, ImageFont


DEFAULT_CANVAS_WIDTH = 1024
DEFAULT_CANVAS_HEIGHT = 1024
DEFAULT_BACKGROUND = "#f5f5f5"
SUPPORTED_EXTENSIONS = {".png"}
CANVAS_PADDING = 40
TITLE_GAP = 24

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "sofa": ("sofa", "couch"),
    "coffee_table": ("coffee_table", "tea_table"),
    "lounge_chair": ("lounge_chair", "armchair", "accent_chair"),
    "dining_table": ("dining_table", "table"),
    "dining_chair": ("dining_chair", "chair"),
    "bed": ("bed", "bedframe"),
    "cabinet": ("cabinet", "bedside_table", "side_table", "nightstand", "night_table"),
}
KNOWN_CATEGORY_SUFFIXES: tuple[str, ...] = tuple(sorted({
    value
    for canonical, aliases in CATEGORY_ALIASES.items()
    for value in (canonical, *aliases)
}, key=len, reverse=True))

DEFAULT_CATALOG_LAYOUT: dict = {
    "canvas_width": DEFAULT_CANVAS_WIDTH,
    "canvas_height": DEFAULT_CANVAS_HEIGHT,
    "boxes": [
        {"type": "sofa", "x": 40, "y": 40, "width": 420, "height": 220, "allowed_types": []},
        {
            "type": "coffee_table",
            "x": 90,
            "y": 290,
            "width": 220,
            "height": 140,
            "allowed_types": ["dining_table", "cabinet"],
        },
        {"type": "lounge_chair", "x": 360, "y": 230, "width": 220, "height": 260, "allowed_types": []},
        {
            "type": "dining_table",
            "x": 40,
            "y": 540,
            "width": 360,
            "height": 200,
            "allowed_types": ["coffee_table", "cabinet"],
        },
        {"type": "dining_chair", "x": 430, "y": 540, "width": 150, "height": 230, "allowed_types": []},
        {"type": "bed", "x": 40, "y": 780, "width": 380, "height": 180, "allowed_types": []},
        {
            "type": "cabinet",
            "x": 430,
            "y": 790,
            "width": 140,
            "height": 160,
            "allowed_types": ["coffee_table", "dining_table"],
        },
    ],
}

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE = Image.LANCZOS


ProgressCallback = Callable[[dict], None]
CATALOG_GENERATION_MODES = {"standard", "exhaustive"}


@dataclass(frozen=True)
class LayoutBox:
    type: str
    x: int
    y: int
    width: int
    height: int
    allowed_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedItem:
    source_path: Path
    raw_category: str
    category: str
    bbox: tuple[int, int, int, int]
    cropped_size: tuple[int, int]


@dataclass(frozen=True)
class AssignedItem:
    box: LayoutBox
    item: PreparedItem
    match_mode: str
    target_size: tuple[int, int]


@dataclass(frozen=True)
class ItemPlacement:
    assignment: AssignedItem
    x: int
    y: int


@dataclass(frozen=True)
class BoxCandidatePlan:
    box: LayoutBox
    candidates: tuple[PreparedItem, ...]
    match_mode: str


def build_catalog_grids(
    input_dir: str | Path,
    output_dir: str | Path,
    layout: dict | None = None,
    background_color: str = DEFAULT_BACKGROUND,
    generation_mode: str = "standard",
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Build one layout-driven catalog grid image for each package subdirectory."""
    input_dir = Path(input_dir).resolve()
    output_dir, file_like_output = _normalize_output_dir(output_dir)
    normalized_layout = normalize_layout_config(layout)
    generation_mode = str(generation_mode or "standard").strip().lower() or "standard"
    if generation_mode not in CATALOG_GENERATION_MODES:
        allowed = ", ".join(sorted(CATALOG_GENERATION_MODES))
        raise ValueError(f"generation_mode must be one of: {allowed}")
    layout_boxes = _build_layout_boxes(normalized_layout["boxes"])
    accepted_types = _collect_accepted_types(layout_boxes)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

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
        "canvas_width": normalized_layout["canvas_width"],
        "canvas_height": normalized_layout["canvas_height"],
        "box_count": len(layout_boxes),
        "generation_mode": generation_mode,
    })
    if file_like_output:
        _emit(progress_callback, {
            "type": "log",
            "message": (
                "Output target looked like a file path. "
                "Using its parent directory because one grid is generated per package."
            ),
        })

    stats = {
        "scanned": 0,
        "selected": 0,
        "duplicate": 0,
        "ignored": 0,
        "error": 0,
        "packages_rendered": 0,
        "outputs_rendered": 0,
    }
    package_summaries: list[dict] = []

    for package_name, files in package_files:
        _emit(progress_callback, {
            "type": "package",
            "package": package_name,
            "count": len(files),
        })

        candidate_items: dict[str, list[PreparedItem]] = defaultdict(list)
        ignored_types: dict[str, int] = {}

        for file_path in files:
            raw_category = _parse_category(file_path.stem)
            category = normalize_catalog_type(raw_category)
            stats["scanned"] += 1

            if category not in accepted_types:
                stats["ignored"] += 1
                ignored_types[category] = ignored_types.get(category, 0) + 1
                _emit(progress_callback, {
                    "type": "progress",
                    "current": stats["scanned"],
                    "total": total_items,
                    "package": package_name,
                    "file": file_path.name,
                    "raw_category": raw_category,
                    "category": category,
                    "status": "ignored_type",
                    "stats": dict(stats),
                })
                continue

            try:
                prepared = _prepare_item(
                    image_path=file_path,
                    raw_category=raw_category,
                    category=category,
                )
            except Exception as exc:
                stats["error"] += 1
                _emit(progress_callback, {"type": "error_item", "file": file_path.name, "message": str(exc)})
                _emit(progress_callback, {
                    "type": "progress",
                    "current": stats["scanned"],
                    "total": total_items,
                    "package": package_name,
                    "file": file_path.name,
                    "raw_category": raw_category,
                    "category": category,
                    "status": "error",
                    "stats": dict(stats),
                })
                continue

            candidate_items[category].append(prepared)
            _emit(progress_callback, {
                "type": "progress",
                "current": stats["scanned"],
                "total": total_items,
                "package": package_name,
                "file": file_path.name,
                "raw_category": raw_category,
                "category": category,
                "status": "candidate",
                "candidate_count": len(candidate_items[category]),
                "stats": dict(stats),
            })

        if generation_mode == "exhaustive":
            package_summary = _render_exhaustive_package_grids(
                package_name=package_name,
                layout=normalized_layout,
                layout_boxes=layout_boxes,
                candidate_items=candidate_items,
                output_dir=output_dir,
                background_color=background_color,
            )
        else:
            assignments, assignment_summary = _assign_items(layout_boxes, candidate_items)
            package_summary = _render_package_grid(
                package_name=package_name,
                layout=normalized_layout,
                layout_boxes=layout_boxes,
                assignments=assignments,
                output_dir=output_dir,
                background_color=background_color,
            )
            package_summary.update(assignment_summary)
            package_summary["missing_types"] = [
                box.type for box in layout_boxes if box.type not in assignments
            ]

        stats["selected"] += package_summary["items_placed"]
        stats["outputs_rendered"] += package_summary.get("variant_count", 0)
        if package_summary.get("variant_count", 0) > 0:
            stats["packages_rendered"] += 1

        package_summary["ignored_types"] = ignored_types
        package_summary["generation_mode"] = generation_mode
        package_summaries.append(package_summary)
        _emit(progress_callback, {"type": "package_done", **package_summary, "stats": dict(stats)})

    generated_files: list[str] = []
    for item in package_summaries:
        output_paths = item.get("output_paths") or []
        if output_paths:
            generated_files.extend(str(path) for path in output_paths)
        elif item.get("output_path"):
            generated_files.append(str(item["output_path"]))

    return {
        "output_dir": str(output_dir),
        "packages": len(package_summaries),
        "items": total_items,
        "generated_files": generated_files,
        "stats": dict(stats),
        "package_summaries": package_summaries,
        "layout": normalized_layout,
        "generation_mode": generation_mode,
        "outputs_rendered": stats["outputs_rendered"],
    }


def build_catalog_grid(
    input_dir: str | Path,
    output_path: str | Path,
    layout: dict | None = None,
    background_color: str = DEFAULT_BACKGROUND,
    generation_mode: str = "standard",
    progress_callback: ProgressCallback | None = None,
) -> dict:
    return build_catalog_grids(
        input_dir=input_dir,
        output_dir=output_path,
        layout=layout,
        background_color=background_color,
        generation_mode=generation_mode,
        progress_callback=progress_callback,
    )


def build_catalog_grid_from_manifest(
    items: list[dict],
    output_path: str | Path,
    layout: dict | None = None,
    background_color: str = DEFAULT_BACKGROUND,
    title: str | None = None,
) -> dict:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas, summary = _render_manifest_canvas(
        items=items,
        layout=layout,
        background_color=background_color,
        title=title,
    )
    canvas.save(output_path)
    return {
        **summary,
        "output_path": str(output_path),
    }


def render_catalog_manifest_preview(
    items: list[dict],
    layout: dict | None = None,
    background_color: str = DEFAULT_BACKGROUND,
    title: str | None = None,
) -> bytes:
    canvas, _ = _render_manifest_canvas(
        items=items,
        layout=layout,
        background_color=background_color,
        title=title,
    )
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_layout_config(layout: dict | None) -> dict:
    source = copy.deepcopy(DEFAULT_CATALOG_LAYOUT if layout is None else layout)
    if not isinstance(source, dict):
        raise ValueError("layout must be an object")

    try:
        canvas_width = int(source.get("canvas_width", DEFAULT_CANVAS_WIDTH))
        canvas_height = int(source.get("canvas_height", DEFAULT_CANVAS_HEIGHT))
    except (TypeError, ValueError):
        raise ValueError("canvas_width and canvas_height must be numeric") from None

    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("canvas_width and canvas_height must be greater than 0")

    rows = source.get("boxes", [])
    if not isinstance(rows, list):
        raise ValueError("boxes must be an array")

    normalized_boxes: list[dict] = []
    seen_types: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each layout box must be an object")

        raw_type = str(row.get("type", "")).strip().lower()
        if not raw_type:
            raise ValueError("box type cannot be empty")
        box_type = normalize_catalog_type(raw_type)
        if box_type in seen_types:
            raise ValueError(f"duplicate box type: {box_type}")

        try:
            x = int(row.get("x"))
            y = int(row.get("y"))
            width = int(row.get("width"))
            height = int(row.get("height"))
        except (TypeError, ValueError):
            raise ValueError(f"box {box_type} must use numeric x/y/width/height") from None

        if width <= 0 or height <= 0:
            raise ValueError(f"box {box_type} width and height must be greater than 0")
        if x < 0 or y < 0:
            raise ValueError(f"box {box_type} cannot start outside the canvas")
        if x + width > canvas_width or y + height > canvas_height:
            raise ValueError(f"box {box_type} must stay within the canvas")

        raw_allowed_types = row.get("allowed_types", [])
        if raw_allowed_types is None:
            raw_allowed_types = []
        if not isinstance(raw_allowed_types, list):
            raise ValueError(f"box {box_type} allowed_types must be an array")

        allowed_types: list[str] = []
        seen_allowed: set[str] = set()
        for raw_allowed in raw_allowed_types:
            normalized_allowed = normalize_catalog_type(str(raw_allowed or "").strip().lower())
            if not normalized_allowed or normalized_allowed == "default":
                continue
            if normalized_allowed == box_type or normalized_allowed in seen_allowed:
                continue
            seen_allowed.add(normalized_allowed)
            allowed_types.append(normalized_allowed)

        seen_types.add(box_type)
        normalized_boxes.append({
            "type": box_type,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "allowed_types": allowed_types,
        })

    return {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "boxes": normalized_boxes,
    }


def normalize_catalog_type(type_name: str) -> str:
    normalized = str(type_name or "").strip().lower()
    if not normalized:
        return "default"
    for canonical, aliases in CATEGORY_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized


def parse_catalog_category_from_stem(stem: str) -> str:
    normalized_stem = str(stem or "").strip().lower()
    if not normalized_stem:
        return "default"

    for suffix in KNOWN_CATEGORY_SUFFIXES:
        if normalized_stem == suffix or normalized_stem.endswith(f"_{suffix}"):
            return normalize_catalog_type(suffix)

    if "_" not in normalized_stem:
        return "default"
    return normalize_catalog_type(normalized_stem.rsplit("_", 1)[-1])


def strip_catalog_category_suffix(
    stem: str,
    category_names: list[str] | tuple[str, ...] | set[str] | None = None,
    extra_suffixes: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    original = str(stem or "")
    lowered = original.lower()
    suffixes = set(KNOWN_CATEGORY_SUFFIXES)

    for values in (category_names or (), extra_suffixes or ()):
        for value in values:
            raw = str(value or "").strip().lower()
            if not raw:
                continue
            suffixes.add(raw)
            normalized = normalize_catalog_type(raw)
            if normalized and normalized != "default":
                suffixes.add(normalized)

    for suffix in sorted(suffixes, key=len, reverse=True):
        token = f"_{suffix}"
        if lowered.endswith(token):
            trimmed = original[:-len(token)]
            return trimmed or original
    return original


def build_classified_catalog_name(base_stem: str, category: str, suffix: str) -> str:
    normalized_base_stem = str(base_stem or "").strip() or "item"
    normalized_category = str(category or "").strip().lower() or "uncertain"
    return f"{normalized_base_stem}_{normalized_category}{suffix}"


def _render_manifest_canvas(
    items: list[dict],
    layout: dict | None,
    background_color: str,
    title: str | None,
) -> tuple[Image.Image, dict]:
    normalized_layout = normalize_layout_config(layout)
    layout_boxes = _build_layout_boxes(normalized_layout["boxes"])
    candidate_items = _prepare_manifest_items(items)
    assignments, assignment_summary = _assign_items(layout_boxes, candidate_items)
    missing_types = [box.type for box in layout_boxes if box.type not in assignments]
    if missing_types:
        raise ValueError(f"Manifest is missing required categories: {', '.join(missing_types)}")

    canvas, placements, base_summary = _compose_grid_canvas(
        title_text=title or "bundle_preview",
        layout=normalized_layout,
        layout_boxes=layout_boxes,
        assignments=assignments,
        background_color=background_color,
    )
    base_summary.update(assignment_summary)
    base_summary["items_placed"] = len(placements)
    base_summary["box_count"] = len(layout_boxes)
    return canvas, base_summary


def _render_package_grid(
    package_name: str,
    layout: dict,
    layout_boxes: list[LayoutBox],
    assignments: dict[str, AssignedItem],
    output_dir: Path,
    background_color: str,
    variant_suffix: str | None = None,
) -> dict:
    title_package_name = package_name if variant_suffix is None else f"{package_name}_{variant_suffix}"
    canvas, placements, base_summary = _compose_grid_canvas(
        title_text=f"=== {title_package_name} ===",
        layout=layout,
        layout_boxes=layout_boxes,
        assignments=assignments,
        background_color=background_color,
    )
    output_name = (
        f"{package_name}_grid.png"
        if variant_suffix is None
        else f"{package_name}_grid_{variant_suffix}.png"
    )
    output_path = output_dir / output_name
    canvas.save(output_path)

    return {
        "package": package_name,
        "output_path": str(output_path),
        "output_name": output_name,
        "output_paths": [str(output_path)],
        "variant_count": 1,
        "first_output_name": output_name,
        "last_output_name": output_name,
        **base_summary,
        "items_placed": len(placements),
        "box_count": len(layout_boxes),
    }


def _render_exhaustive_package_grids(
    package_name: str,
    layout: dict,
    layout_boxes: list[LayoutBox],
    candidate_items: dict[str, list[PreparedItem]],
    output_dir: Path,
    background_color: str,
) -> dict:
    candidate_plans, assignment_summary = _build_exhaustive_candidate_plans(layout_boxes, candidate_items)
    empty_summary = {
        "package": package_name,
        "output_path": "",
        "output_name": "",
        "output_paths": [],
        "variant_count": 0,
        "first_output_name": "",
        "last_output_name": "",
        "canvas_width": layout["canvas_width"] + CANVAS_PADDING * 2,
        "canvas_height": layout["canvas_height"] + CANVAS_PADDING * 2,
        "layout_canvas_width": layout["canvas_width"],
        "layout_canvas_height": layout["canvas_height"],
        "items_placed": 0,
        "box_count": len(layout_boxes),
        "placement_details": [],
        "unused_candidate_types": {},
        **assignment_summary,
    }
    if assignment_summary["missing_types"]:
        return {
            **empty_summary,
            "message": f"Missing required categories: {', '.join(assignment_summary['missing_types'])}",
        }

    rendered_variants: list[dict] = []
    total_items_placed = 0
    for variant_index, assignments in enumerate(_iter_exhaustive_assignments(candidate_plans)):
        variant_summary = _render_package_grid(
            package_name=package_name,
            layout=layout,
            layout_boxes=layout_boxes,
            assignments=assignments,
            output_dir=output_dir,
            background_color=background_color,
            variant_suffix=_index_to_alpha_suffix(variant_index),
        )
        rendered_variants.append(variant_summary)
        total_items_placed += variant_summary["items_placed"]

    if not rendered_variants:
        return {
            **empty_summary,
            "message": "No complete combinations available without reusing the same image",
        }

    first_variant = rendered_variants[0]
    last_variant = rendered_variants[-1]
    return {
        "package": package_name,
        "output_path": first_variant["output_path"],
        "output_name": first_variant["output_name"],
        "output_paths": [item["output_path"] for item in rendered_variants],
        "variant_count": len(rendered_variants),
        "first_output_name": first_variant["output_name"],
        "last_output_name": last_variant["output_name"],
        "canvas_width": first_variant["canvas_width"],
        "canvas_height": first_variant["canvas_height"],
        "layout_canvas_width": first_variant["layout_canvas_width"],
        "layout_canvas_height": first_variant["layout_canvas_height"],
        "items_placed": total_items_placed,
        "box_count": len(layout_boxes),
        "placement_details": [],
        "unused_candidate_types": {},
        "message": "",
        **assignment_summary,
    }


def _compose_grid_canvas(
    title_text: str,
    layout: dict,
    layout_boxes: list[LayoutBox],
    assignments: dict[str, AssignedItem],
    background_color: str,
) -> tuple[Image.Image, list[ItemPlacement], dict]:
    title_font = _load_font(max(24, layout["canvas_width"] // 24))
    title_height = _measure_text_height(title_text, title_font)
    canvas_width = layout["canvas_width"] + CANVAS_PADDING * 2
    content_origin_x = CANVAS_PADDING
    content_origin_y = CANVAS_PADDING + title_height + TITLE_GAP
    canvas_height = content_origin_y + layout["canvas_height"] + CANVAS_PADDING

    background_rgba = ImageColor.getrgb(background_color) + (255,)
    canvas = Image.new("RGBA", (canvas_width, canvas_height), background_rgba)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (CANVAS_PADDING, CANVAS_PADDING),
        title_text,
        fill=ImageColor.getrgb("#334155"),
        font=title_font,
    )

    placements: list[ItemPlacement] = []
    for box in layout_boxes:
        assignment = assignments.get(box.type)
        if assignment is None:
            continue
        offset_x = content_origin_x + box.x + max(0, (box.width - assignment.target_size[0]) // 2)
        offset_y = content_origin_y + box.y + max(0, box.height - assignment.target_size[1])
        placements.append(ItemPlacement(assignment=assignment, x=offset_x, y=offset_y))

    for placement in placements:
        with Image.open(placement.assignment.item.source_path) as source_image:
            rgba = source_image.convert("RGBA")
            cropped = rgba.crop(placement.assignment.item.bbox)
            resized = cropped.resize(placement.assignment.target_size, RESAMPLE)
            canvas.alpha_composite(resized, dest=(placement.x, placement.y))

    return canvas, placements, {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "layout_canvas_width": layout["canvas_width"],
        "layout_canvas_height": layout["canvas_height"],
    }


def _assign_items(
    layout_boxes: list[LayoutBox],
    candidate_items: dict[str, list[PreparedItem]],
) -> tuple[dict[str, AssignedItem], dict]:
    queues = {
        category: deque(items)
        for category, items in candidate_items.items()
    }
    assignments: dict[str, AssignedItem] = {}
    placement_details: list[dict] = []
    fallback_usage: list[str] = []
    exact_filled = 0
    fallback_filled = 0

    for box in layout_boxes:
        item = _pop_candidate(queues, box.type)
        if item is None:
            continue
        assignment = _build_assignment(box, item, match_mode="exact")
        assignments[box.type] = assignment
        exact_filled += 1

    for box in layout_boxes:
        if box.type in assignments:
            continue
        for allowed_type in box.allowed_types:
            item = _pop_candidate(queues, allowed_type)
            if item is None:
                continue
            assignment = _build_assignment(box, item, match_mode="fallback")
            assignments[box.type] = assignment
            fallback_filled += 1
            fallback_usage.append(f"{box.type} -> {item.category}")
            break

    for box in layout_boxes:
        assignment = assignments.get(box.type)
        if assignment is None:
            continue
        placement_details.append({
            "box_type": box.type,
            "source_category": assignment.item.category,
            "file": assignment.item.source_path.name,
            "match_mode": assignment.match_mode,
            "target_width_px": assignment.target_size[0],
            "target_height_px": assignment.target_size[1],
        })

    unused_candidate_types = {
        category: len(queue)
        for category, queue in queues.items()
        if len(queue) > 0
    }

    return assignments, {
        "exact_filled": exact_filled,
        "fallback_filled": fallback_filled,
        "fallback_usage": fallback_usage,
        "placement_details": placement_details,
        "unused_candidate_types": unused_candidate_types,
    }


def _build_exhaustive_candidate_plans(
    layout_boxes: list[LayoutBox],
    candidate_items: dict[str, list[PreparedItem]],
) -> tuple[list[BoxCandidatePlan], dict]:
    plans: list[BoxCandidatePlan] = []
    missing_types: list[str] = []
    exact_filled = 0
    fallback_filled = 0
    fallback_usage: list[str] = []

    for box in layout_boxes:
        exact_candidates = list(candidate_items.get(box.type, []))
        if exact_candidates:
            plans.append(BoxCandidatePlan(
                box=box,
                candidates=tuple(exact_candidates),
                match_mode="exact",
            ))
            exact_filled += 1
            continue

        selected_allowed_type = ""
        selected_candidates: list[PreparedItem] = []
        for allowed_type in box.allowed_types:
            allowed_candidates = list(candidate_items.get(allowed_type, []))
            if not allowed_candidates:
                continue
            selected_allowed_type = allowed_type
            selected_candidates = allowed_candidates
            break

        if selected_candidates:
            plans.append(BoxCandidatePlan(
                box=box,
                candidates=tuple(selected_candidates),
                match_mode="fallback",
            ))
            fallback_filled += 1
            fallback_usage.append(f"{box.type} -> {selected_allowed_type}")
            continue

        missing_types.append(box.type)

    return plans, {
        "exact_filled": exact_filled,
        "fallback_filled": fallback_filled,
        "fallback_usage": fallback_usage,
        "missing_types": missing_types,
    }


def _iter_exhaustive_assignments(
    candidate_plans: list[BoxCandidatePlan],
    index: int = 0,
    used_paths: set[Path] | None = None,
    current_assignments: dict[str, AssignedItem] | None = None,
) -> Iterator[dict[str, AssignedItem]]:
    active_paths = used_paths if used_paths is not None else set()
    active_assignments = current_assignments if current_assignments is not None else {}

    if index >= len(candidate_plans):
        yield dict(active_assignments)
        return

    plan = candidate_plans[index]
    for item in plan.candidates:
        if item.source_path in active_paths:
            continue

        active_paths.add(item.source_path)
        active_assignments[plan.box.type] = _build_assignment(
            plan.box,
            item,
            match_mode=plan.match_mode,
        )
        yield from _iter_exhaustive_assignments(
            candidate_plans,
            index=index + 1,
            used_paths=active_paths,
            current_assignments=active_assignments,
        )
        active_assignments.pop(plan.box.type, None)
        active_paths.remove(item.source_path)


def _index_to_alpha_suffix(index: int) -> str:
    if index < 0:
        raise ValueError("index must be greater than or equal to 0")

    chars: list[str] = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        chars.append(chr(ord("a") + remainder))
        if current == 0:
            break
        current -= 1

    return "".join(reversed(chars))


def _build_assignment(
    box: LayoutBox,
    item: PreparedItem,
    match_mode: str,
) -> AssignedItem:
    target_size = _fit_item_to_box(item.cropped_size, box)
    return AssignedItem(
        box=box,
        item=item,
        match_mode=match_mode,
        target_size=target_size,
    )


def _fit_item_to_box(
    cropped_size: tuple[int, int],
    box: LayoutBox,
) -> tuple[int, int]:
    cropped_width, cropped_height = cropped_size
    resize_ratio = min(box.width / cropped_width, box.height / cropped_height)
    target_width = max(1, int(round(cropped_width * resize_ratio)))
    target_height = max(1, int(round(cropped_height * resize_ratio)))
    return target_width, target_height


def _prepare_item(
    image_path: Path,
    raw_category: str,
    category: str,
) -> PreparedItem:
    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
        alpha_channel = rgba.getchannel("A")
        bbox = alpha_channel.getbbox()
        if bbox is None:
            bbox = (0, 0, rgba.width, rgba.height)

        cropped_width = max(1, bbox[2] - bbox[0])
        cropped_height = max(1, bbox[3] - bbox[1])

    return PreparedItem(
        source_path=image_path,
        raw_category=raw_category,
        category=category,
        bbox=bbox,
        cropped_size=(cropped_width, cropped_height),
    )


def _build_layout_boxes(rows: list[dict]) -> list[LayoutBox]:
    return [
        LayoutBox(
            type=row["type"],
            x=row["x"],
            y=row["y"],
            width=row["width"],
            height=row["height"],
            allowed_types=tuple(row.get("allowed_types", [])),
        )
        for row in rows
    ]


def _collect_accepted_types(layout_boxes: list[LayoutBox]) -> set[str]:
    accepted_types: set[str] = set()
    for box in layout_boxes:
        accepted_types.add(box.type)
        accepted_types.update(box.allowed_types)
    return accepted_types


def _pop_candidate(
    queues: dict[str, deque[PreparedItem]],
    category: str,
) -> PreparedItem | None:
    queue = queues.get(category)
    if not queue:
        return None
    return queue.popleft()


def _prepare_manifest_items(items: list[dict]) -> dict[str, list[PreparedItem]]:
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest items cannot be empty")

    candidate_items: dict[str, list[PreparedItem]] = defaultdict(list)
    seen_paths: set[Path] = set()

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each manifest item must be an object")

        category = normalize_catalog_type(item.get("category"))
        image_path_value = item.get("image_path") or item.get("source_image_path")
        if not image_path_value:
            raise ValueError("Each manifest item must include image_path")

        image_path = Path(str(image_path_value)).expanduser().resolve()
        if image_path in seen_paths:
            raise ValueError(f"Duplicate manifest image detected: {image_path.name}")
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Manifest image must be PNG: {image_path.name}")

        seen_paths.add(image_path)
        candidate_items[category].append(_prepare_item(
            image_path=image_path,
            raw_category=category,
            category=category,
        ))

    return candidate_items


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


def _parse_category(stem: str) -> str:
    return parse_catalog_category_from_stem(stem)


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
    parser.add_argument("--canvas_width", type=int, default=DEFAULT_CANVAS_WIDTH, help="Canvas width for the default layout.")
    parser.add_argument("--canvas_height", type=int, default=DEFAULT_CANVAS_HEIGHT, help="Canvas height for the default layout.")
    parser.add_argument("--background", default=DEFAULT_BACKGROUND, help="Canvas background color, e.g. #f5f5f5.")
    args = parser.parse_args()

    output_target = args.output_dir or args.output_path or "catalog_grids"
    layout = copy.deepcopy(DEFAULT_CATALOG_LAYOUT)
    layout["canvas_width"] = args.canvas_width
    layout["canvas_height"] = args.canvas_height
    summary = build_catalog_grids(
        input_dir=args.input_dir,
        output_dir=output_target,
        layout=layout,
        background_color=args.background,
    )
    print(f"Saved {summary['packages']} grid images to: {summary['output_dir']}")
    for path in summary["generated_files"]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
