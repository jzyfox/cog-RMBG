from __future__ import annotations

import base64
import io
import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from asset_catalog_grid import KNOWN_CATEGORY_SUFFIXES, normalize_catalog_type

ProgressCallback = Callable[[dict], None]

SEMANTIC_CATEGORY = "sofa"
SEMANTIC_MODEL = "qwen3.5-plus"
SEMANTIC_PROMPT_VERSION = "semantic_multicategory_v2"
LEGACY_SOFA_PROMPT_VERSION = "sofa_semantic_v1"
SEMANTIC_INDEX_FILE = "semantic_tags.jsonl"
SEMANTIC_ERROR_FILE = "semantic_tag_errors.jsonl"
DEFAULT_SLEEP_SECONDS = 0.5
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_IMAGE_SIDE = 1600
DEFAULT_JPEG_QUALITY = 90
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

STYLE_OPTIONS = [
    "现代",
    "极简",
    "北欧",
    "中式",
    "日式",
    "欧式",
    "美式",
    "工业风",
    "轻奢",
    "奶油风",
    "中古风",
    "混搭",
]
COLOR_FAMILY_OPTIONS = [
    "白色",
    "米色",
    "灰色",
    "棕色",
    "黑色",
    "绿色",
    "蓝色",
    "黄色",
    "橙色",
    "红色",
    "粉色",
    "紫色",
    "混色",
]
COLOR_BRIGHTNESS_OPTIONS = ["浅", "中", "深"]
MATERIAL_OPTIONS = [
    "布艺",
    "皮革",
    "丝绒",
    "羊羔绒",
    "麂皮",
    "木质",
    "金属",
    "藤编",
    "玻璃",
    "石材",
    "混合",
]


def _enum_field(name: str, label: str, options: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "type": "enum",
        "options": list(options),
    }


SEMANTIC_CATEGORY_DEFINITIONS: dict[str, dict[str, Any]] = {
    "sofa": {
        "label": "沙发",
        "detail_fields": [
            _enum_field("layout_shape", "布局形状", ["L型", "U型", "一字型", "弧形", "其他"]),
            _enum_field("footprint_shape", "平面轮廓", ["圆形", "正方形", "长方形", "异形", "其他"]),
        ],
    },
    "bed": {
        "label": "床",
        "detail_fields": [
            _enum_field("bed_form", "床体形式", ["平台床", "软包床", "架子床", "其他"]),
            _enum_field("headboard_shape", "床头形状", ["直板", "弧形", "翼形", "无", "其他"]),
            _enum_field("bed_base_style", "床底样式", ["高脚式", "落地式", "箱体式", "其他"]),
        ],
    },
    "lounge_chair": {
        "label": "休闲椅",
        "detail_fields": [
            _enum_field("armrest_style", "扶手形式", ["有扶手", "无扶手", "半扶手", "其他"]),
            _enum_field("backrest_height", "靠背高度", ["低背", "中背", "高背", "其他"]),
            _enum_field("base_style", "底座样式", ["四脚", "旋转底座", "雪橇脚", "板式", "其他"]),
        ],
    },
    "dining_chair": {
        "label": "餐椅",
        "detail_fields": [
            _enum_field("backrest_shape", "靠背形状", ["镂空", "实心", "条栅", "软包", "其他"]),
            _enum_field("seat_form", "坐面形式", ["硬座", "软包", "编织", "其他"]),
            _enum_field("base_style", "底座样式", ["四脚", "雪橇脚", "板式", "其他"]),
        ],
    },
    "coffee_table": {
        "label": "茶几",
        "detail_fields": [
            _enum_field("tabletop_shape", "台面形状", ["圆形", "正方形", "长方形", "椭圆形", "异形", "其他"]),
            _enum_field("base_style", "底座样式", ["四脚", "柱式", "框架", "板式", "其他"]),
            _enum_field("storage_form", "收纳形式", ["无", "层板", "抽屉", "其他"]),
        ],
    },
    "dining_table": {
        "label": "餐桌",
        "detail_fields": [
            _enum_field("tabletop_shape", "台面形状", ["圆形", "正方形", "长方形", "椭圆形", "异形", "其他"]),
            _enum_field("base_style", "底座样式", ["四脚", "柱式", "框架", "板式", "其他"]),
        ],
    },
    "bedside_table": {
        "label": "床头柜",
        "detail_fields": [
            _enum_field("tabletop_shape", "台面形状", ["圆形", "正方形", "长方形", "椭圆形", "异形", "其他"]),
            _enum_field("base_style", "底座样式", ["高脚式", "落地式", "悬浮式", "其他"]),
            _enum_field("storage_form", "收纳形式", ["无", "单抽", "双抽及以上", "开放格", "其他"]),
        ],
    },
}

SUPPORTED_SEMANTIC_CATEGORIES = tuple(SEMANTIC_CATEGORY_DEFINITIONS.keys())
SUPPORTED_SEMANTIC_CATEGORY_SET = set(SUPPORTED_SEMANTIC_CATEGORIES)

SEMANTIC_SHARED_ENUMS = {
    "styles": STYLE_OPTIONS,
    "color_families": COLOR_FAMILY_OPTIONS,
    "color_brightness": COLOR_BRIGHTNESS_OPTIONS,
    "materials": MATERIAL_OPTIONS,
}

SEMANTIC_FRONTEND_CONFIG = {
    "category_options": list(SUPPORTED_SEMANTIC_CATEGORIES),
    "category_labels": {
        category: definition["label"]
        for category, definition in SEMANTIC_CATEGORY_DEFINITIONS.items()
    },
    "category_schemas": {
        category: {
            "label": definition["label"],
            "detail_fields": [dict(field) for field in definition["detail_fields"]],
        }
        for category, definition in SEMANTIC_CATEGORY_DEFINITIONS.items()
    },
    "shared_enums": SEMANTIC_SHARED_ENUMS,
    "enums": SEMANTIC_SHARED_ENUMS,
    "default_category": SEMANTIC_CATEGORY,
    "default_sleep_seconds": DEFAULT_SLEEP_SECONDS,
    "default_max_retries": DEFAULT_MAX_RETRIES,
}


def build_semantic_tags(
    input_dir: str | Path,
    category: str = SEMANTIC_CATEGORY,
    skip_existing: bool = True,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    input_root = Path(input_dir).expanduser().resolve()
    category = _normalize_target_category(category)
    sleep_seconds = _coerce_sleep_seconds(sleep_seconds)
    max_retries = _coerce_max_retries(max_retries)

    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹：{input_root}")

    images = discover_semantic_images(input_root, category=category)
    if not images:
        raise FileNotFoundError(f"未找到任何 {category} PNG 图片：{input_root}")

    _emit(progress_callback, {
        "type": "start",
        "input_dir": str(input_root),
        "category": category,
        "total": len(images),
        "skip_existing": skip_existing,
        "sleep_seconds": sleep_seconds,
        "max_retries": max_retries,
    })

    stats = {"tagged": 0, "skipped_existing": 0, "error": 0}

    for index, image_path in enumerate(images, start=1):
        sidecar_path = image_path.with_suffix(".json")
        attempted_request = False

        if skip_existing and sidecar_path.exists():
            try:
                existing = _load_json(sidecar_path)
                normalized = _validate_semantic_record(
                    existing,
                    image_path=image_path,
                    input_root=input_root,
                )
                if existing != normalized:
                    _write_sidecar(sidecar_path, normalized)
                stats["skipped_existing"] += 1
                _emit(progress_callback, {
                    "type": "progress",
                    "current": index,
                    "total": len(images),
                    "file": image_path.name,
                    "status": "skipped_existing",
                    "stats": dict(stats),
                })
                continue
            except Exception:
                pass

        try:
            attempted_request = True
            record = _tag_single_image(
                image_path=image_path,
                input_root=input_root,
                category=category,
                sleep_seconds=sleep_seconds,
                max_retries=max_retries,
            )
            _write_sidecar(sidecar_path, record)
            stats["tagged"] += 1
            _emit(progress_callback, {
                "type": "progress",
                "current": index,
                "total": len(images),
                "file": image_path.name,
                "status": "tagged",
                "primary_style": record["primary_style"],
                "color_family": record["color_family"],
                "stats": dict(stats),
            })
        except Exception as exc:
            stats["error"] += 1
            _emit(progress_callback, {
                "type": "error_item",
                "file": image_path.name,
                "message": str(exc),
            })
            _emit(progress_callback, {
                "type": "progress",
                "current": index,
                "total": len(images),
                "file": image_path.name,
                "status": "error",
                "stats": dict(stats),
            })

        if attempted_request and sleep_seconds > 0 and index < len(images):
            time.sleep(sleep_seconds)

    index_stats = rebuild_semantic_indices(input_root)
    return {
        "input_dir": str(input_root),
        "category": category,
        "total": len(images),
        "stats": {
            **stats,
            **index_stats,
        },
    }


def load_semantic_review_data(
    input_dir: str | Path,
    category: str = SEMANTIC_CATEGORY,
) -> dict:
    input_root = Path(input_dir).expanduser().resolve()
    category = _normalize_target_category(category)

    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹：{input_root}")

    images = discover_semantic_images(input_root, category=category)
    items: list[dict[str, Any]] = []
    stats = {"valid": 0, "invalid": 0, "missing": 0}

    for image_path in images:
        item = _build_review_item(input_root, image_path)
        stats[item["status"]] += 1
        items.append(item)

    return {
        "input_dir": str(input_root),
        "category": category,
        "items": items,
        "stats": stats,
    }


def save_semantic_review_items(
    input_dir: str | Path,
    items: list[dict[str, Any]],
    category: str = SEMANTIC_CATEGORY,
) -> dict:
    input_root = Path(input_dir).expanduser().resolve()
    category = _normalize_target_category(category)

    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹：{input_root}")
    if not isinstance(items, list) or not items:
        raise ValueError("items 必须是非空数组")

    saved_items: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("每个保存项都必须是对象")

        image_path = _resolve_item_image_path(item=item, input_root=input_root)
        if _parse_category_from_stem(image_path.stem) != category:
            raise ValueError(f"仅允许保存 {category} 图片：{image_path.name}")

        raw_tag_data = item.get("tag_data", item)
        if not isinstance(raw_tag_data, dict):
            raise ValueError(f"标签数据格式不正确：{image_path.name}")

        existing_vlm_model = None
        existing_sidecar = image_path.with_suffix(".json")
        if existing_sidecar.exists():
            try:
                existing_record = _load_json(existing_sidecar)
                if isinstance(existing_record, dict):
                    existing_vlm_model = existing_record.get("vlm_model")
            except Exception:
                pass

        record = _validate_semantic_record(
            raw_tag_data,
            image_path=image_path,
            input_root=input_root,
            vlm_model=existing_vlm_model,
            refresh_updated_at=True,
        )
        _write_sidecar(existing_sidecar, record)
        saved_items.append(_build_review_item(input_root, image_path))

    index_stats = rebuild_semantic_indices(input_root)
    return {
        "status": "saved",
        "count": len(saved_items),
        "items": saved_items,
        "stats": index_stats,
    }


def rebuild_semantic_indices(
    input_dir: str | Path,
    category: str | None = None,
) -> dict:
    input_root = Path(input_dir).expanduser().resolve()
    if category not in (None, ""):
        _normalize_target_category(category)

    valid_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []

    for image_path in discover_semantic_images(input_root, category=None):
        sidecar_path = image_path.with_suffix(".json")
        if not sidecar_path.exists():
            error_records.append(_build_error_record(
                image_path=image_path,
                input_root=input_root,
                status="missing",
                message="未找到同名语义标签 JSON",
            ))
            continue

        try:
            record = _load_json(sidecar_path)
            normalized = _validate_semantic_record(
                record,
                image_path=image_path,
                input_root=input_root,
            )
            if record != normalized:
                _write_sidecar(sidecar_path, normalized)
            valid_records.append(normalized)
        except Exception as exc:
            error_records.append(_build_error_record(
                image_path=image_path,
                input_root=input_root,
                status="invalid",
                message=str(exc),
            ))

    _write_jsonl(input_root / SEMANTIC_INDEX_FILE, valid_records)
    _write_jsonl(input_root / SEMANTIC_ERROR_FILE, error_records)

    return {
        "valid": len(valid_records),
        "invalid": sum(1 for item in error_records if item["status"] == "invalid"),
        "missing": sum(1 for item in error_records if item["status"] == "missing"),
    }


def discover_semantic_images(
    input_dir: str | Path,
    category: str | None = SEMANTIC_CATEGORY,
) -> list[Path]:
    input_root = Path(input_dir).expanduser().resolve()
    normalized_category = None if category in (None, "") else _normalize_target_category(category)
    return sorted(
        path
        for path in input_root.rglob("*.png")
        if path.is_file()
        and _parse_category_from_stem(path.stem) in SUPPORTED_SEMANTIC_CATEGORY_SET
        and (normalized_category is None or _parse_category_from_stem(path.stem) == normalized_category)
    )


def _tag_single_image(
    image_path: Path,
    input_root: Path,
    category: str,
    sleep_seconds: float,
    max_retries: int,
) -> dict:
    last_error: Exception | None = None
    prompt = _build_prompt(category)
    data_url = _image_path_to_data_url(
        image_path=image_path,
        max_image_side=DEFAULT_MAX_IMAGE_SIDE,
        jpeg_quality=DEFAULT_JPEG_QUALITY,
    )

    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(sleep_seconds)

        try:
            raw_response = _call_qwen_vlm(data_url, prompt)
            payload = _parse_json_object(raw_response)
            return _validate_semantic_record(
                payload,
                image_path=image_path,
                input_root=input_root,
                refresh_updated_at=True,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"打标失败（重试 {max_retries} 次后仍未成功）：{last_error}") from last_error


def _call_qwen_vlm(
    image_data_url: str,
    prompt: str,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("缺少 openai 依赖，请先安装 requirements.txt 中的新依赖") from exc

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未配置环境变量 DASHSCOPE_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
    )
    response = client.chat.completions.create(
        model=SEMANTIC_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
        extra_body={
            "enable_thinking": False,
            "vl_high_resolution_images": False,
        },
    )
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def _image_path_to_data_url(
    image_path: Path,
    max_image_side: int,
    jpeg_quality: int,
) -> str:
    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))

        if max(background.size) > max_image_side:
            scale = max_image_side / max(background.size)
            resized = background.resize(
                (
                    max(1, int(round(background.width * scale))),
                    max(1, int(round(background.height * scale))),
                ),
                Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS,
            )
        else:
            resized = background

        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


def _build_review_item(input_root: Path, image_path: Path) -> dict[str, Any]:
    json_path = image_path.with_suffix(".json")
    raw_data = None
    error_message = None
    status = "missing"
    tag_data = _build_review_draft(image_path=image_path, input_root=input_root)

    if json_path.exists():
        try:
            raw_data = _load_json(json_path)
            validated = _validate_semantic_record(
                raw_data,
                image_path=image_path,
                input_root=input_root,
            )
            if raw_data != validated:
                _write_sidecar(json_path, validated)
            status = "valid"
            tag_data = validated
        except Exception as exc:
            status = "invalid"
            error_message = str(exc)
            tag_data = _build_review_draft(
                image_path=image_path,
                input_root=input_root,
                raw_data=raw_data,
            )

    return {
        "file_name": image_path.name,
        "image_path": str(image_path),
        "json_path": str(json_path),
        "status": status,
        "tag_data": tag_data,
        "error_message": error_message,
    }


def _build_review_draft(
    image_path: Path,
    input_root: Path,
    raw_data: Any | None = None,
) -> dict[str, Any]:
    category = _get_image_category(image_path)
    metadata = _build_metadata(image_path=image_path, input_root=input_root)
    payload = raw_data if isinstance(raw_data, dict) else {}

    primary_style = _safe_enum_value(payload.get("primary_style"), STYLE_OPTIONS)
    main_material = _safe_enum_value(payload.get("main_material"), MATERIAL_OPTIONS)

    secondary_materials: list[str] = []
    raw_secondary_materials = payload.get("secondary_materials", [])
    if isinstance(raw_secondary_materials, list):
        for value in raw_secondary_materials:
            normalized = _safe_enum_value(value, MATERIAL_OPTIONS)
            if (
                normalized
                and normalized != main_material
                and normalized not in secondary_materials
                and len(secondary_materials) < 2
            ):
                secondary_materials.append(normalized)

    return {
        "category": category,
        "primary_style": primary_style,
        "secondary_style": _safe_secondary_style(payload.get("secondary_style"), primary_style),
        "color_family": _safe_enum_value(payload.get("color_family"), COLOR_FAMILY_OPTIONS),
        "color_brightness": _safe_enum_value(payload.get("color_brightness"), COLOR_BRIGHTNESS_OPTIONS),
        "main_material": main_material,
        "secondary_materials": secondary_materials,
        "category_details": _build_category_details_draft(category, payload),
        "brand": None,
        "size": {
            "width_mm": None,
            "depth_mm": None,
            "height_mm": None,
        },
        **metadata,
        "prompt_version": str(payload.get("prompt_version") or SEMANTIC_PROMPT_VERSION),
        "vlm_model": str(payload.get("vlm_model") or SEMANTIC_MODEL),
        "updated_at": str(payload.get("updated_at") or ""),
    }


def _validate_semantic_record(
    raw: Any,
    image_path: Path,
    input_root: Path,
    vlm_model: str | None = None,
    prompt_version: str | None = None,
    refresh_updated_at: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("标签 JSON 必须是对象")

    category = _get_image_category(image_path)
    _validate_raw_field_names(raw, category)

    raw_category = raw.get("category")
    if raw_category not in (None, ""):
        normalized_raw_category = normalize_catalog_type(str(raw_category).strip().lower())
        if normalized_raw_category != category:
            raise ValueError(f"category 与图片类别不一致：{normalized_raw_category}")

    primary_style = _require_enum(raw.get("primary_style"), STYLE_OPTIONS, "primary_style")
    secondary_style = raw.get("secondary_style")
    if secondary_style in ("", "null"):
        secondary_style = None
    elif secondary_style is not None:
        secondary_style = _require_enum(secondary_style, STYLE_OPTIONS, "secondary_style")
        if secondary_style == primary_style:
            raise ValueError("secondary_style 不能与 primary_style 相同")

    main_material = _require_enum(raw.get("main_material"), MATERIAL_OPTIONS, "main_material")

    raw_secondary_materials = raw.get("secondary_materials", [])
    if raw_secondary_materials in (None, ""):
        raw_secondary_materials = []
    if not isinstance(raw_secondary_materials, list):
        raise ValueError("secondary_materials 必须是数组")

    secondary_materials: list[str] = []
    for value in raw_secondary_materials:
        material = _require_enum(value, MATERIAL_OPTIONS, "secondary_materials")
        if material == main_material:
            raise ValueError("secondary_materials 不能包含 main_material")
        if material in secondary_materials:
            raise ValueError("secondary_materials 不能重复")
        secondary_materials.append(material)
    if len(secondary_materials) > 2:
        raise ValueError("secondary_materials 最多允许 2 项")

    _validate_readonly_fields(raw)
    category_details = _validate_category_details(raw, category)
    metadata = _build_metadata(image_path=image_path, input_root=input_root)

    return {
        "category": category,
        "primary_style": primary_style,
        "secondary_style": secondary_style,
        "color_family": _require_enum(raw.get("color_family"), COLOR_FAMILY_OPTIONS, "color_family"),
        "color_brightness": _require_enum(raw.get("color_brightness"), COLOR_BRIGHTNESS_OPTIONS, "color_brightness"),
        "main_material": main_material,
        "secondary_materials": secondary_materials,
        "category_details": category_details,
        "brand": None,
        "size": {
            "width_mm": None,
            "depth_mm": None,
            "height_mm": None,
        },
        **metadata,
        "prompt_version": str(prompt_version or SEMANTIC_PROMPT_VERSION),
        "vlm_model": str(vlm_model or raw.get("vlm_model") or SEMANTIC_MODEL),
        "updated_at": _now_iso() if refresh_updated_at else str(raw.get("updated_at") or _now_iso()),
    }


def _build_prompt(category: str) -> str:
    definition = _get_category_definition(category)
    detail_fields = definition["detail_fields"]
    detail_schema_lines = "\n".join(
        f'    "{field["name"]}": "..."{"," if index < len(detail_fields) - 1 else ""}'
        for index, field in enumerate(detail_fields)
    )
    detail_rule_lines = "\n".join(
        f'{7 + index}. category_details.{field["name"]} 只能从 [{"、".join(field["options"])}] 中选 1 个。'
        for index, field in enumerate(detail_fields)
    )
    label = definition["label"]

    return (
        f"请分析这张单件{label}产品图。已知图片类别一定是“{label}”，不需要识别品牌和尺寸。\n\n"
        "请严格输出一个 JSON 对象，不要输出 markdown，不要输出解释，不要输出任何 JSON 之外的内容。\n\n"
        "字段必须且只能包含：\n"
        "{\n"
        f'  "category": "{category}",\n'
        '  "primary_style": "...",\n'
        '  "secondary_style": null,\n'
        '  "color_family": "...",\n'
        '  "color_brightness": "...",\n'
        '  "main_material": "...",\n'
        '  "secondary_materials": [],\n'
        '  "category_details": {\n'
        f"{detail_schema_lines}\n"
        "  },\n"
        '  "brand": null,\n'
        '  "size": {\n'
        '    "width_mm": null,\n'
        '    "depth_mm": null,\n'
        '    "height_mm": null\n'
        "  }\n"
        "}\n\n"
        "取值规则：\n"
        f'1. primary_style 只能从 [{"、".join(STYLE_OPTIONS)}] 中选 1 个。\n'
        "2. secondary_style 只能为 null，或从同一列表中再选 1 个，且不能与 primary_style 相同。\n"
        f'3. color_family 只能从 [{"、".join(COLOR_FAMILY_OPTIONS)}] 中选 1 个。\n'
        f'4. color_brightness 只能从 [{"、".join(COLOR_BRIGHTNESS_OPTIONS)}] 中选 1 个。\n'
        f'5. main_material 只能从 [{"、".join(MATERIAL_OPTIONS)}] 中选 1 个。\n'
        "6. secondary_materials 为 0 到 2 个数组项，每项都必须从同一材质列表中选择，且不能与 main_material 重复。\n"
        f"{detail_rule_lines}\n"
        f"{7 + len(detail_fields)}. brand 必须为 null。\n"
        f"{8 + len(detail_fields)}. size.width_mm / depth_mm / height_mm 必须全部为 null。\n"
        f"{9 + len(detail_fields)}. 如果无法可靠判断某一项，请使用允许的兜底值，例如 其他、无、混色、混合、null，但不要发明新标签。"
    )


def _get_category_definition(category: str) -> dict[str, Any]:
    normalized = _normalize_target_category(category)
    return SEMANTIC_CATEGORY_DEFINITIONS[normalized]


def _get_image_category(image_path: Path) -> str:
    category = _parse_category_from_stem(image_path.stem)
    if category not in SUPPORTED_SEMANTIC_CATEGORY_SET:
        raise ValueError(f"当前语义标签不支持该图片类别：{image_path.name}")
    return category


def _build_category_details_draft(category: str, payload: dict[str, Any]) -> dict[str, str]:
    definition = _get_category_definition(category)
    raw_details = payload.get("category_details")
    details_source = raw_details if isinstance(raw_details, dict) else {}
    legacy_sofa_details = {}
    if category == "sofa":
        legacy_sofa_details = {
            "layout_shape": payload.get("layout_shape"),
            "footprint_shape": payload.get("footprint_shape"),
        }

    normalized_details: dict[str, str] = {}
    for field in definition["detail_fields"]:
        value = details_source.get(field["name"])
        if value in (None, "") and field["name"] in legacy_sofa_details:
            value = legacy_sofa_details[field["name"]]
        normalized_details[field["name"]] = _safe_enum_value(value, field["options"])
    return normalized_details


def _validate_category_details(raw: dict[str, Any], category: str) -> dict[str, str]:
    definition = _get_category_definition(category)
    raw_details = raw.get("category_details")

    if raw_details in (None, ""):
        if category == "sofa":
            raw_details = {
                "layout_shape": raw.get("layout_shape"),
                "footprint_shape": raw.get("footprint_shape"),
            }
        else:
            raw_details = {}

    if not isinstance(raw_details, dict):
        raise ValueError("category_details 必须是对象")

    field_names = {field["name"] for field in definition["detail_fields"]}
    extra_names = sorted(set(raw_details) - field_names)
    if extra_names:
        raise ValueError(f"category_details 存在未允许字段：{', '.join(extra_names)}")

    normalized_details: dict[str, str] = {}
    for field in definition["detail_fields"]:
        normalized_details[field["name"]] = _require_enum(
            raw_details.get(field["name"]),
            field["options"],
            f'category_details.{field["name"]}',
        )
    return normalized_details


def _validate_readonly_fields(raw: dict[str, Any]) -> None:
    if raw.get("brand") not in (None, "", "null"):
        raise ValueError("brand 必须为 null")

    raw_size = raw.get("size")
    if raw_size in (None, "", {}):
        return
    if not isinstance(raw_size, dict):
        raise ValueError("size 必须是对象")

    for key in ("width_mm", "depth_mm", "height_mm"):
        if raw_size.get(key) not in (None, "", "null"):
            raise ValueError(f"size.{key} 必须为 null")


def _validate_raw_field_names(raw: dict[str, Any], category: str) -> None:
    allowed_names = {
        "category",
        "primary_style",
        "secondary_style",
        "color_family",
        "color_brightness",
        "main_material",
        "secondary_materials",
        "category_details",
        "brand",
        "size",
        "file_name",
        "relative_path",
        "package_name",
        "source_image_path",
        "prompt_version",
        "vlm_model",
        "updated_at",
    }
    if category == "sofa":
        allowed_names.update({"layout_shape", "footprint_shape"})

    extra_names = sorted(set(raw) - allowed_names)
    if extra_names:
        raise ValueError(f"存在未允许字段：{', '.join(extra_names)}")


def _build_metadata(image_path: Path, input_root: Path) -> dict[str, str]:
    relative_path = image_path.relative_to(input_root).as_posix()
    parts = Path(relative_path).parts
    package_name = parts[0] if len(parts) > 1 else input_root.name
    return {
        "file_name": image_path.name,
        "relative_path": relative_path,
        "package_name": package_name,
        "source_image_path": str(image_path),
    }


def _build_error_record(
    image_path: Path,
    input_root: Path,
    status: str,
    message: str,
) -> dict[str, Any]:
    return {
        **_build_metadata(image_path=image_path, input_root=input_root),
        "category": _get_image_category(image_path),
        "status": status,
        "message": message,
        "updated_at": _now_iso(),
    }


def _resolve_item_image_path(item: dict[str, Any], input_root: Path) -> Path:
    candidates: list[Any] = [
        item.get("image_path"),
        item.get("source_image_path"),
        item.get("relative_path"),
        item.get("file_name"),
    ]
    if isinstance(item.get("tag_data"), dict):
        candidates.extend([
            item["tag_data"].get("image_path"),
            item["tag_data"].get("source_image_path"),
            item["tag_data"].get("relative_path"),
            item["tag_data"].get("file_name"),
        ])

    candidate = next((value for value in candidates if value), None)
    if not candidate:
        raise ValueError("保存项缺少 image_path / source_image_path / relative_path")

    image_path = Path(str(candidate)).expanduser()
    if not image_path.is_absolute():
        image_path = (input_root / image_path).resolve()
    else:
        image_path = image_path.resolve()

    if input_root not in image_path.parents and image_path != input_root:
        raise ValueError(f"图片路径不在输入目录内：{image_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    if image_path.suffix.lower() != ".png":
        raise ValueError(f"只允许保存 PNG 图片：{image_path.name}")
    return image_path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_sidecar(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _parse_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("模型返回结果不是文本")

    candidate = text.strip()
    if not candidate:
        raise ValueError("模型返回空内容")

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError("模型返回中未找到 JSON 对象")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = candidate[start:index + 1]
                parsed = json.loads(snippet)
                if not isinstance(parsed, dict):
                    raise ValueError("模型返回的 JSON 不是对象")
                return parsed

    raise ValueError("模型返回的 JSON 不完整")


def _parse_category_from_stem(stem: str) -> str:
    normalized_stem = str(stem or "").strip().lower()
    for suffix in KNOWN_CATEGORY_SUFFIXES:
        if normalized_stem == suffix or normalized_stem.endswith(f"_{suffix}"):
            return normalize_catalog_type(suffix)
    if "_" not in normalized_stem:
        return "default"
    return normalize_catalog_type(normalized_stem.rsplit("_", 1)[-1])


def _normalize_target_category(category: Any) -> str:
    normalized = normalize_catalog_type(str(category or "").strip().lower())
    if normalized not in SUPPORTED_SEMANTIC_CATEGORY_SET:
        supported = ", ".join(SUPPORTED_SEMANTIC_CATEGORIES)
        raise ValueError(f"当前仅支持以下语义标签品类：{supported}")
    return normalized


def _coerce_sleep_seconds(value: Any) -> float:
    try:
        sleep_seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError("sleep_seconds 必须是数字") from None
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds 不能小于 0")
    return sleep_seconds


def _coerce_max_retries(value: Any) -> int:
    try:
        max_retries = int(value)
    except (TypeError, ValueError):
        raise ValueError("max_retries 必须是整数") from None
    if max_retries < 0:
        raise ValueError("max_retries 不能小于 0")
    return max_retries


def _require_enum(value: Any, allowed: list[str], field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} 不能为空")
    normalized = str(value).strip()
    if normalized not in allowed:
        raise ValueError(f"{field_name} 不在允许范围内：{normalized}")
    return normalized


def _safe_enum_value(value: Any, allowed: list[str]) -> str:
    normalized = str(value).strip()
    return normalized if normalized in allowed else ""


def _safe_secondary_style(value: Any, primary_style: str) -> str | None:
    if value in (None, "", "null"):
        return None
    normalized = str(value).strip()
    if normalized in STYLE_OPTIONS and normalized != primary_style:
        return normalized
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(payload)
