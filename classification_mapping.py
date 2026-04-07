import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asset_catalog_grid import (
    normalize_catalog_token,
    normalize_catalog_type,
    parse_catalog_category_from_stem,
)


CLASSIFICATION_INDEX_FILE_NAME = ".classification_index.json"
CLASSIFICATION_MAPPING_PRESET_FILE = Path(__file__).parent / "classification_mapping_preset.json"
CLASSIFICATION_MAPPING_SNAPSHOT_FILE_NAME = ".classification_mapping.json"
CLASSIFICATION_MAPPING_VERSION = 1
UNCERTAIN_CATEGORY = "uncertain"

MAPPING_TARGET_CATEGORIES: tuple[str, ...] = (
    "sofa",
    "coffee_table",
    "lounge_chair",
    "dining_table",
    "dining_chair",
    "bed",
    "cabinet",
)
MAPPING_TARGET_CATEGORY_SET = set(MAPPING_TARGET_CATEGORIES)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_mapping_source_label(value: Any) -> str:
    return normalize_catalog_token(str(value or "").strip().lower())


def normalize_mapping_target_category(value: Any) -> str:
    normalized = normalize_catalog_type(str(value or "").strip().lower())
    if normalized not in MAPPING_TARGET_CATEGORY_SET:
        allowed = ", ".join(MAPPING_TARGET_CATEGORIES)
        raise ValueError(f"Unsupported mapping target category: {value}. Allowed: {allowed}")
    return normalized


def _normalize_stored_mappings(raw_mappings: Any) -> dict[str, str]:
    if not isinstance(raw_mappings, dict):
        return {}

    normalized: dict[str, str] = {}
    for raw_source, raw_target in raw_mappings.items():
        source_label = normalize_mapping_source_label(raw_source)
        if not source_label or source_label == "default" or source_label in MAPPING_TARGET_CATEGORY_SET:
            continue
        try:
            target_category = normalize_mapping_target_category(raw_target)
        except Exception:
            continue
        normalized[source_label] = target_category
    return dict(sorted(normalized.items()))


def _load_mapping_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw_mappings = payload.get("mappings") if isinstance(payload, dict) else payload
    return _normalize_stored_mappings(raw_mappings)


def _write_mapping_file(path: Path, mappings: dict[str, str], extra: dict[str, Any] | None = None) -> dict[str, str]:
    normalized = _normalize_stored_mappings(mappings)
    payload = {
        "version": CLASSIFICATION_MAPPING_VERSION,
        "updated_at": _now_iso(),
        "mappings": normalized,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def load_global_mapping_preset() -> dict[str, str]:
    return _load_mapping_file(CLASSIFICATION_MAPPING_PRESET_FILE)


def save_global_mapping_preset(mappings: dict[str, str]) -> dict[str, str]:
    return _write_mapping_file(CLASSIFICATION_MAPPING_PRESET_FILE, mappings)


def directory_mapping_snapshot_path(output_root: str | Path) -> Path:
    return Path(output_root).expanduser().resolve() / CLASSIFICATION_MAPPING_SNAPSHOT_FILE_NAME


def load_directory_mapping_snapshot(output_root: str | Path) -> dict[str, str]:
    return _load_mapping_file(directory_mapping_snapshot_path(output_root))


def save_directory_mapping_snapshot(output_root: str | Path, mappings: dict[str, str]) -> dict[str, str]:
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return _write_mapping_file(
        directory_mapping_snapshot_path(output_path),
        mappings,
        extra={"output_dir": str(output_path)},
    )


def load_classification_index(output_root: str | Path) -> dict[str, str]:
    output_path = Path(output_root).expanduser().resolve()
    index_path = output_path / CLASSIFICATION_INDEX_FILE_NAME
    if not index_path.exists():
        return {}

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    raw_items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, dict):
        return {}

    items: dict[str, str] = {}
    for raw_relative_path, raw_category in raw_items.items():
        relative_path = str(raw_relative_path or "").replace("\\", "/").strip().lstrip("./")
        if not relative_path or Path(relative_path).suffix.lower() != ".png":
            continue
        normalized_category = normalize_mapping_source_label(raw_category)
        if not normalized_category or normalized_category == "default":
            continue
        items[relative_path] = normalized_category
    return items


def get_classified_source_label(
    image_path: str | Path,
    output_root: str | Path,
    *,
    index_items: dict[str, str] | None = None,
) -> str:
    output_path = Path(output_root).expanduser().resolve()
    image = Path(image_path).expanduser().resolve()
    if output_path not in image.parents and image != output_path:
        raise ValueError(f"Image path is outside output directory: {image}")

    effective_index = index_items if index_items is not None else load_classification_index(output_path)
    relative_path = image.relative_to(output_path).as_posix()
    indexed_label = normalize_mapping_source_label(effective_index.get(relative_path))
    if indexed_label and indexed_label != "default":
        return indexed_label

    candidate_labels = {
        value
        for value in (effective_index or {}).values()
        if normalize_mapping_source_label(value) not in {"", "default"}
    }
    source_label = parse_catalog_category_from_stem(
        image.stem,
        category_names=sorted(candidate_labels),
        extra_suffixes=[UNCERTAIN_CATEGORY, *MAPPING_TARGET_CATEGORIES],
    )
    return normalize_mapping_source_label(source_label)


def resolve_effective_target_category(
    source_label: Any,
    *,
    output_root: str | Path | None = None,
    snapshot: dict[str, str] | None = None,
    preset: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    normalized_source = normalize_mapping_source_label(source_label)
    if not normalized_source or normalized_source == "default":
        return None, "unmapped"
    if normalized_source in MAPPING_TARGET_CATEGORY_SET:
        return normalized_source, "identity"

    effective_snapshot = snapshot if snapshot is not None else (
        load_directory_mapping_snapshot(output_root) if output_root is not None else {}
    )
    target = effective_snapshot.get(normalized_source)
    if target:
        return target, "directory"

    effective_preset = preset if preset is not None else load_global_mapping_preset()
    target = effective_preset.get(normalized_source)
    if target:
        return target, "global"

    return None, "unmapped"


def get_mapped_semantic_category(
    image_path: str | Path,
    input_root: str | Path,
    *,
    index_items: dict[str, str] | None = None,
    snapshot: dict[str, str] | None = None,
    preset: dict[str, str] | None = None,
) -> str | None:
    source_label = get_classified_source_label(image_path, input_root, index_items=index_items)
    target_category, _ = resolve_effective_target_category(
        source_label,
        output_root=input_root,
        snapshot=snapshot,
        preset=preset,
    )
    return target_category
