from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable
from uuid import uuid4

from asset_catalog_grid import (
    DEFAULT_CATALOG_LAYOUT,
    build_catalog_grid_from_manifest,
    normalize_catalog_type,
    render_catalog_manifest_preview,
)
from semantic_tagger import (
    SEMANTIC_CATEGORY_LABELS,
    SEMANTIC_INDEX_FILE,
    rebuild_semantic_indices,
)

ProgressCallback = Callable[[dict], None]

BUNDLE_CANDIDATE_FILE = "semantic_bundle_candidates.jsonl"
BUNDLE_REVIEW_FILE = "semantic_bundle_review.jsonl"
BUNDLE_ERROR_FILE = "semantic_bundle_errors.jsonl"

DEFAULT_BUNDLE_TARGET_COUNT = 100
DEFAULT_SEED_CANDIDATE_COUNT = 8
SEED_TOP_K = 12
AUTO_TOP_K = 8
SEED_BEAM_WIDTH = 128
AUTO_BEAM_WIDTH = 48
SEED_NON_ANCHOR_MAX_APPEARANCES = 2

BUNDLE_CATEGORIES: tuple[str, ...] = (
    "sofa",
    "coffee_table",
    "lounge_chair",
    "dining_table",
    "dining_chair",
    "bed",
    "cabinet",
)
BUNDLE_CATEGORY_SET = set(BUNDLE_CATEGORIES)

AUTO_ANCHOR_CATEGORY = "sofa"
AUTO_CLUSTER_CAP_RATIO = 0.2
REUSE_LIMITS = {
    "sofa": 2,
    "bed": 2,
    "dining_table": 2,
    "coffee_table": 3,
    "lounge_chair": 3,
    "dining_chair": 3,
    "cabinet": 3,
}
STATUS_OPTIONS = {"pending", "accepted", "rejected"}
GENERATION_MODES = {"auto", "seeded"}

BUNDLE_FRONTEND_CONFIG = {
    "bundle_categories": list(BUNDLE_CATEGORIES),
    "category_labels": {
        category: SEMANTIC_CATEGORY_LABELS.get(category, category)
        for category in BUNDLE_CATEGORIES
    },
    "default_target_count": DEFAULT_BUNDLE_TARGET_COUNT,
    "default_seed_candidate_count": DEFAULT_SEED_CANDIDATE_COUNT,
    "reuse_limits": dict(REUSE_LIMITS),
}


def build_bundle_candidates(
    input_dir: str | Path,
    target_count: int = DEFAULT_BUNDLE_TARGET_COUNT,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    input_root = _resolve_input_root(input_dir)
    target_count = _coerce_positive_int(target_count, "target_count")

    semantic_records = _load_semantic_records(input_root, refresh=True)
    pool_by_category = _group_semantic_records(semantic_records)
    workspace, workspace_errors = _load_workspace(input_root, semantic_records)
    existing_errors = _load_error_rows(input_root)
    existing_signatures = {bundle_signature(bundle) for bundle in workspace}
    reuse_counts = _build_reuse_counts(workspace)
    cluster_cap = max(1, math.ceil(target_count * AUTO_CLUSTER_CAP_RATIO))
    cluster_counts = Counter(
        bundle.get("cluster_key")
        for bundle in workspace
        if bundle.get("generation_mode") == "auto" and bundle.get("cluster_key")
    )
    pool_counts = {category: len(pool_by_category.get(category, [])) for category in BUNDLE_CATEGORIES}

    _emit(progress_callback, {
        "type": "start",
        "input_dir": str(input_root),
        "target_count": target_count,
        "pool_counts": pool_counts,
        "existing_candidates": len(workspace),
        "cluster_cap": cluster_cap,
    })

    anchors = list(pool_by_category.get(AUTO_ANCHOR_CATEGORY, []))
    if not anchors:
        raise FileNotFoundError("No valid sofa semantic records found for auto bundle generation")

    new_bundles: list[dict[str, Any]] = []
    run_errors: list[dict[str, Any]] = []
    emitted_anchor_failures: set[str] = set()

    while len(new_bundles) < target_count:
        made_progress = False
        for anchor in anchors:
            if len(new_bundles) >= target_count:
                break

            cluster_key = _cluster_key(anchor)
            if cluster_counts[cluster_key] >= cluster_cap:
                continue

            variants = _generate_bundle_variants(
                anchor_record=anchor,
                generation_mode="auto",
                pool_by_category=pool_by_category,
                existing_signatures=existing_signatures,
                reuse_counts=reuse_counts,
                max_results=1,
                top_k=AUTO_TOP_K,
                beam_width=AUTO_BEAM_WIDTH,
                locked_categories=[AUTO_ANCHOR_CATEGORY],
            )
            if not variants:
                anchor_key = str(anchor["source_image_path"])
                if anchor_key not in emitted_anchor_failures:
                    emitted_anchor_failures.add(anchor_key)
                    run_errors.append(_build_error_row(
                        error_type="auto_anchor_failed",
                        message="Unable to build a complete non-duplicate bundle from this anchor",
                        extra={
                            "anchor_category": anchor["category"],
                            "anchor_image_path": anchor["source_image_path"],
                            "cluster_key": cluster_key,
                        },
                    ))
                continue

            bundle = variants[0]
            new_bundles.append(bundle)
            made_progress = True
            existing_signatures.add(bundle_signature(bundle))
            _bump_bundle_reuse_counts(reuse_counts, bundle)
            cluster_counts[cluster_key] += 1

            _emit(progress_callback, {
                "type": "progress",
                "current": len(new_bundles),
                "total": target_count,
                "status": "generated",
                "bundle_id": bundle["bundle_id"],
                "score": bundle["score"],
                "anchor_category": bundle["anchor_category"],
                "anchor_image_path": bundle["anchor_image_path"],
                "cluster_key": bundle["cluster_key"],
                "stats": {
                    "generated": len(new_bundles),
                    "existing": len(workspace),
                },
            })

        if not made_progress:
            break

    combined = workspace + new_bundles
    persisted_errors = existing_errors + workspace_errors + run_errors
    _persist_workspace(input_root, combined, persisted_errors)

    summary = {
        "input_dir": str(input_root),
        "target_count": target_count,
        "generated_count": len(new_bundles),
        "actual_generated": len(new_bundles),
        "items": combined,
        "stats": _workspace_stats(combined),
        "shortage_counts": _build_shortage_summary(pool_by_category, combined, target_count),
    }
    return summary


def load_bundle_pool(
    input_dir: str | Path,
    category: str | None = None,
) -> dict:
    input_root = _resolve_input_root(input_dir)
    semantic_records = _load_semantic_records(input_root, refresh=True)
    normalized_category = None
    if category not in (None, "", "all"):
        normalized_category = _normalize_bundle_category(category)

    items = []
    counts = Counter()
    for record in semantic_records:
        counts[record["category"]] += 1
        if normalized_category and record["category"] != normalized_category:
            continue
        items.append(_pool_item_from_record(record))

    return {
        "input_dir": str(input_root),
        "category": normalized_category,
        "items": items,
        "stats": {
            "total": len(items),
            "counts": {category_name: counts.get(category_name, 0) for category_name in BUNDLE_CATEGORIES},
        },
    }


def seed_bundle_candidates(
    input_dir: str | Path,
    seed_image_path: str | Path,
    candidate_count: int = DEFAULT_SEED_CANDIDATE_COUNT,
) -> dict:
    input_root = _resolve_input_root(input_dir)
    candidate_count = _coerce_positive_int(candidate_count, "candidate_count")
    semantic_records = _load_semantic_records(input_root, refresh=True)
    semantic_map = {str(Path(record["source_image_path"]).resolve()): record for record in semantic_records}
    pool_by_category = _group_semantic_records(semantic_records)

    seed_path = _resolve_image_path(seed_image_path, input_root)
    seed_record = semantic_map.get(str(seed_path))
    if not seed_record:
        raise ValueError("Seed image must belong to the current directory and have a valid semantic tag")

    workspace, workspace_errors = _load_workspace(input_root, semantic_records)
    existing_errors = _load_error_rows(input_root)
    existing_signatures = {bundle_signature(bundle) for bundle in workspace}
    reuse_counts = _build_reuse_counts(workspace)

    variants = _generate_bundle_variants(
        anchor_record=seed_record,
        generation_mode="seeded",
        pool_by_category=pool_by_category,
        existing_signatures=existing_signatures,
        reuse_counts=reuse_counts,
        max_results=max(candidate_count * 6, candidate_count),
        top_k=SEED_TOP_K,
        beam_width=SEED_BEAM_WIDTH,
        locked_categories=[seed_record["category"]],
    )
    selected, rejected_errors = _select_seeded_variants(
        variants=variants,
        existing_signatures=existing_signatures,
        reuse_counts=reuse_counts,
        candidate_count=candidate_count,
        seed_path=str(seed_path),
    )

    combined = workspace + selected
    persisted_errors = existing_errors + workspace_errors + rejected_errors
    _persist_workspace(input_root, combined, persisted_errors)

    return {
        "input_dir": str(input_root),
        "seed_image_path": str(seed_path),
        "seed_category": seed_record["category"],
        "requested_count": candidate_count,
        "generated_count": len(selected),
        "items": selected,
        "stats": _workspace_stats(combined),
    }


def load_bundle_review_data(input_dir: str | Path) -> dict:
    input_root = _resolve_input_root(input_dir)
    semantic_records = _load_semantic_records(input_root, refresh=False)
    workspace, workspace_errors = _load_workspace(input_root, semantic_records)
    persisted_errors = _load_error_rows(input_root)
    if workspace_errors:
        _persist_workspace(input_root, workspace, persisted_errors + workspace_errors)

    return {
        "input_dir": str(input_root),
        "items": workspace,
        "stats": _workspace_stats(workspace),
    }


def save_bundle_review(
    input_dir: str | Path,
    bundles: list[dict[str, Any]],
) -> dict:
    input_root = _resolve_input_root(input_dir)
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("bundles must be a non-empty array")

    semantic_records = _load_semantic_records(input_root, refresh=False)
    workspace, workspace_errors = _load_workspace(input_root, semantic_records)
    existing_errors = _load_error_rows(input_root)
    bundle_map = {bundle["bundle_id"]: bundle for bundle in workspace}

    updated: list[dict[str, Any]] = []
    for patch in bundles:
        bundle_id = str((patch or {}).get("bundle_id") or "").strip()
        if not bundle_id or bundle_id not in bundle_map:
            raise ValueError(f"Unknown bundle_id: {bundle_id or '<empty>'}")
        status = str((patch or {}).get("status") or "").strip().lower()
        if status not in STATUS_OPTIONS:
            raise ValueError(f"Invalid bundle status: {status}")
        bundle = bundle_map[bundle_id]
        bundle["status"] = status
        bundle["updated_at"] = _now_iso()
        updated.append(bundle)

    ordered_workspace = [bundle_map[bundle["bundle_id"]] for bundle in workspace]
    _persist_workspace(input_root, ordered_workspace, existing_errors + workspace_errors)
    return {
        "status": "saved",
        "count": len(updated),
        "items": updated,
        "stats": _workspace_stats(ordered_workspace),
    }


def regenerate_bundle_candidate(
    input_dir: str | Path,
    bundle_id: str,
) -> dict:
    input_root = _resolve_input_root(input_dir)
    semantic_records = _load_semantic_records(input_root, refresh=True)
    semantic_map = {str(Path(record["source_image_path"]).resolve()): record for record in semantic_records}
    pool_by_category = _group_semantic_records(semantic_records)
    workspace, workspace_errors = _load_workspace(input_root, semantic_records)
    existing_errors = _load_error_rows(input_root)

    bundle_map = {bundle["bundle_id"]: bundle for bundle in workspace}
    current = bundle_map.get(bundle_id)
    if not current:
        raise ValueError(f"Unknown bundle_id: {bundle_id}")

    remaining_workspace = [bundle for bundle in workspace if bundle["bundle_id"] != bundle_id]
    existing_signatures = {bundle_signature(bundle) for bundle in remaining_workspace}
    reuse_counts = _build_reuse_counts(remaining_workspace)

    anchor_path = _resolve_image_path(current["anchor_image_path"], input_root)
    anchor_record = semantic_map.get(str(anchor_path))
    if not anchor_record:
        raise ValueError("Anchor image no longer has a valid semantic tag")

    variants = _generate_bundle_variants(
        anchor_record=anchor_record,
        generation_mode=current["generation_mode"],
        pool_by_category=pool_by_category,
        existing_signatures=existing_signatures,
        reuse_counts=reuse_counts,
        max_results=24 if current["generation_mode"] == "seeded" else 8,
        top_k=SEED_TOP_K if current["generation_mode"] == "seeded" else AUTO_TOP_K,
        beam_width=SEED_BEAM_WIDTH if current["generation_mode"] == "seeded" else AUTO_BEAM_WIDTH,
        locked_categories=current.get("locked_categories") or [current["anchor_category"]],
        bundle_id=current["bundle_id"],
        created_at=current.get("created_at"),
    )
    if not variants:
        raise RuntimeError("Could not regenerate a new non-duplicate bundle from the current anchor")

    regenerated = variants[0]
    regenerated["status"] = "pending"
    regenerated["updated_at"] = _now_iso()

    merged = []
    for bundle in workspace:
        merged.append(regenerated if bundle["bundle_id"] == bundle_id else bundle)

    _persist_workspace(input_root, merged, existing_errors + workspace_errors)
    return {
        "status": "regenerated",
        "item": regenerated,
        "stats": _workspace_stats(merged),
    }


def render_bundle_preview(
    input_dir: str | Path,
    bundle_id: str,
    layout: dict | None = None,
) -> bytes:
    input_root = _resolve_input_root(input_dir)
    semantic_records = _load_semantic_records(input_root, refresh=False)
    workspace, _ = _load_workspace(input_root, semantic_records)
    bundle = next((item for item in workspace if item["bundle_id"] == bundle_id), None)
    if not bundle:
        raise ValueError(f"Unknown bundle_id: {bundle_id}")

    return render_catalog_manifest_preview(
        items=bundle["items"],
        layout=layout or DEFAULT_CATALOG_LAYOUT,
        title=bundle_id,
    )


def render_bundle_outputs(
    input_dir: str | Path,
    output_dir: str | Path,
    bundle_ids: list[str] | None = None,
    layout: dict | None = None,
) -> dict:
    input_root = _resolve_input_root(input_dir)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    semantic_records = _load_semantic_records(input_root, refresh=False)
    workspace, _ = _load_workspace(input_root, semantic_records)
    selected_ids = {str(item) for item in (bundle_ids or []) if str(item).strip()}
    selected = [
        bundle
        for bundle in workspace
        if (bundle["bundle_id"] in selected_ids if selected_ids else bundle["status"] == "accepted")
    ]
    if not selected:
        raise ValueError("No accepted bundles available for rendering")

    rendered: list[dict[str, Any]] = []
    for index, bundle in enumerate(selected, start=1):
        stem = f"bundle_{index:03d}"
        image_path = output_root / f"{stem}_grid.png"
        summary = build_catalog_grid_from_manifest(
            items=bundle["items"],
            output_path=image_path,
            layout=layout or DEFAULT_CATALOG_LAYOUT,
            title=stem,
        )
        manifest_path = output_root / f"{stem}.json"
        manifest_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        rendered.append({
            "bundle_id": bundle["bundle_id"],
            "image_path": str(image_path),
            "manifest_path": str(manifest_path),
            "score": bundle["score"],
            "status": bundle["status"],
            "render_summary": summary,
        })

    return {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "count": len(rendered),
        "items": rendered,
    }


def bundle_signature(bundle: dict[str, Any]) -> str:
    item_map = {
        normalize_catalog_type(item.get("category")): str(Path(item.get("image_path") or item.get("source_image_path")).resolve())
        for item in bundle.get("items", [])
    }
    return "||".join(item_map.get(category, "") for category in BUNDLE_CATEGORIES)


def _resolve_input_root(input_dir: str | Path) -> Path:
    input_root = Path(input_dir).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_root}")
    return input_root


def _load_semantic_records(input_root: Path, refresh: bool) -> list[dict[str, Any]]:
    if refresh or not (input_root / SEMANTIC_INDEX_FILE).exists():
        rebuild_semantic_indices(input_root)

    index_path = input_root / SEMANTIC_INDEX_FILE
    if not index_path.exists():
        return []

    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for row in _read_jsonl(index_path):
        if not isinstance(row, dict):
            continue
        category = normalize_catalog_type(row.get("category"))
        if category not in BUNDLE_CATEGORY_SET:
            continue
        image_path = _resolve_image_path(row.get("source_image_path"), input_root)
        normalized_path = str(image_path)
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        normalized = dict(row)
        normalized["category"] = category
        normalized["source_image_path"] = normalized_path
        normalized["file_name"] = str(row.get("file_name") or image_path.name)
        normalized["relative_path"] = str(row.get("relative_path") or image_path.relative_to(input_root).as_posix())
        normalized["package_name"] = str(row.get("package_name") or Path(normalized["relative_path"]).parts[0])
        normalized["secondary_materials"] = list(row.get("secondary_materials") or [])
        normalized["category_details"] = dict(row.get("category_details") or {})
        records.append(normalized)

    records.sort(key=lambda record: (record["category"], record["relative_path"]))
    return records


def _group_semantic_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    return grouped


def _load_workspace(
    input_root: Path,
    semantic_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semantic_map = {str(Path(record["source_image_path"]).resolve()): record for record in semantic_records}
    source_path = input_root / BUNDLE_CANDIDATE_FILE
    if not source_path.exists():
        source_path = input_root / BUNDLE_REVIEW_FILE
    if not source_path.exists():
        return [], []

    bundles: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in _read_jsonl(source_path):
        if not isinstance(row, dict):
            continue
        try:
            bundles.append(_normalize_bundle(row, input_root, semantic_map))
        except Exception as exc:
            errors.append(_build_error_row(
                error_type="invalid_bundle_record",
                message=str(exc),
                extra={"source_file": str(source_path)},
            ))
    return bundles, errors


def _normalize_bundle(
    raw: dict[str, Any],
    input_root: Path,
    semantic_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    bundle_id = str(raw.get("bundle_id") or "").strip() or f"bundle_{uuid4().hex[:12]}"
    generation_mode = str(raw.get("generation_mode") or "").strip().lower()
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"Invalid generation_mode in bundle {bundle_id}")

    anchor_category = _normalize_bundle_category(raw.get("anchor_category"))
    anchor_image_path = str(_resolve_image_path(raw.get("anchor_image_path"), input_root))
    locked_categories = [
        _normalize_bundle_category(category)
        for category in (raw.get("locked_categories") or [anchor_category])
    ]
    if anchor_category not in locked_categories:
        locked_categories.append(anchor_category)

    status = str(raw.get("status") or "pending").strip().lower()
    if status not in STATUS_OPTIONS:
        status = "pending"

    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        raise ValueError(f"Bundle {bundle_id} is missing items")

    item_map: dict[str, dict[str, Any]] = {}
    for raw_item in raw_items:
        item = _normalize_bundle_item(raw_item, input_root, semantic_map)
        if item["category"] in item_map:
            raise ValueError(f"Bundle {bundle_id} has duplicate category {item['category']}")
        item_map[item["category"]] = item

    missing = [category for category in BUNDLE_CATEGORIES if category not in item_map]
    if missing:
        raise ValueError(f"Bundle {bundle_id} is missing categories: {', '.join(missing)}")

    ordered_items = [item_map[category] for category in BUNDLE_CATEGORIES]
    return {
        "bundle_id": bundle_id,
        "generation_mode": generation_mode,
        "status": status,
        "score": round(float(raw.get("score") or _bundle_average_pairwise_score(ordered_items)), 4),
        "cluster_key": str(raw.get("cluster_key") or _cluster_key(item_map[anchor_category]["semantic_tag"])),
        "anchor_category": anchor_category,
        "anchor_image_path": anchor_image_path,
        "locked_categories": [category for category in BUNDLE_CATEGORIES if category in locked_categories],
        "items": ordered_items,
        "created_at": str(raw.get("created_at") or _now_iso()),
        "updated_at": str(raw.get("updated_at") or raw.get("created_at") or _now_iso()),
    }


def _normalize_bundle_item(
    raw_item: dict[str, Any],
    input_root: Path,
    semantic_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    image_path = str(_resolve_image_path(raw_item.get("image_path") or raw_item.get("source_image_path"), input_root))
    semantic_tag = raw_item.get("semantic_tag") if isinstance(raw_item.get("semantic_tag"), dict) else semantic_map.get(image_path)
    if not isinstance(semantic_tag, dict):
        semantic_tag = semantic_map.get(image_path)
    if not isinstance(semantic_tag, dict):
        raise ValueError(f"Missing semantic snapshot for bundle item: {image_path}")

    category = _normalize_bundle_category(raw_item.get("category") or semantic_tag.get("category"))
    semantic_snapshot = dict(semantic_map.get(image_path) or semantic_tag)
    if semantic_snapshot.get("category") != category:
        raise ValueError(f"Bundle item category mismatch for {image_path}")

    return {
        "category": category,
        "image_path": image_path,
        "relative_path": str(semantic_snapshot.get("relative_path") or Path(image_path).relative_to(input_root).as_posix()),
        "file_name": str(semantic_snapshot.get("file_name") or Path(image_path).name),
        "package_name": str(semantic_snapshot.get("package_name") or Path(image_path).parent.name),
        "semantic_tag": semantic_snapshot,
    }


def _generate_bundle_variants(
    anchor_record: dict[str, Any],
    generation_mode: str,
    pool_by_category: dict[str, list[dict[str, Any]]],
    existing_signatures: set[str],
    reuse_counts: Counter,
    max_results: int,
    top_k: int,
    beam_width: int,
    locked_categories: list[str],
    bundle_id: str | None = None,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    anchor_category = anchor_record["category"]
    fill_categories = [category for category in BUNDLE_CATEGORIES if category != anchor_category]
    anchor_path = str(anchor_record["source_image_path"])

    partials = [{
        "selected": {anchor_category: anchor_record},
        "used_paths": {anchor_path},
        "slot_scores": [],
        "partial_score": 0.0,
    }]

    for category in fill_categories:
        expanded: list[dict[str, Any]] = []
        for partial in partials:
            candidates: list[tuple[float, dict[str, Any]]] = []
            for candidate in pool_by_category.get(category, []):
                candidate_path = str(candidate["source_image_path"])
                if candidate_path in partial["used_paths"]:
                    continue
                if reuse_counts[candidate_path] >= REUSE_LIMITS[category]:
                    continue

                anchor_score = semantic_compatibility(anchor_record, candidate)
                selected_records = list(partial["selected"].values())
                average_selected_score = mean(
                    semantic_compatibility(candidate, selected_record)
                    for selected_record in selected_records
                ) if selected_records else anchor_score
                slot_score = (0.6 * anchor_score) + (0.4 * average_selected_score)
                if slot_score < 4:
                    continue
                candidates.append((slot_score, candidate))

            candidates.sort(
                key=lambda item: (
                    item[0],
                    semantic_compatibility(item[1], anchor_record),
                    item[1]["relative_path"],
                ),
                reverse=True,
            )

            for slot_score, candidate in candidates[:top_k]:
                selected = dict(partial["selected"])
                selected[category] = candidate
                expanded.append({
                    "selected": selected,
                    "used_paths": set(partial["used_paths"]) | {str(candidate["source_image_path"])},
                    "slot_scores": list(partial["slot_scores"]) + [slot_score],
                    "partial_score": partial["partial_score"] + slot_score,
                })

        if not expanded:
            return []
        expanded.sort(
            key=lambda item: (
                item["partial_score"],
                len(item["used_paths"]),
            ),
            reverse=True,
        )
        partials = expanded[:beam_width]

    variants: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for partial in partials:
        items = [_bundle_item_from_record(partial["selected"][category]) for category in BUNDLE_CATEGORIES]
        bundle = _build_bundle_record(
            generation_mode=generation_mode,
            anchor_record=anchor_record,
            locked_categories=locked_categories,
            items=items,
            bundle_id=bundle_id,
            created_at=created_at,
        )
        signature = bundle_signature(bundle)
        if signature in existing_signatures or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        variants.append(bundle)

    variants.sort(key=lambda bundle: (bundle["score"], bundle_signature(bundle)), reverse=True)
    return variants[:max_results]


def _select_seeded_variants(
    variants: list[dict[str, Any]],
    existing_signatures: set[str],
    reuse_counts: Counter,
    candidate_count: int,
    seed_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    local_non_anchor_usage: Counter = Counter()
    temp_reuse = Counter(reuse_counts)
    errors: list[dict[str, Any]] = []

    for bundle in variants:
        if len(selected) >= candidate_count:
            break
        signature = bundle_signature(bundle)
        if signature in existing_signatures:
            continue

        non_anchor_items = [
            item for item in bundle["items"]
            if str(item["image_path"]) != seed_path
        ]
        if any(local_non_anchor_usage[item["image_path"]] >= SEED_NON_ANCHOR_MAX_APPEARANCES for item in non_anchor_items):
            continue
        if any(temp_reuse[item["image_path"]] >= REUSE_LIMITS[item["category"]] for item in bundle["items"]):
            continue

        selected.append(bundle)
        existing_signatures.add(signature)
        for item in bundle["items"]:
            temp_reuse[item["image_path"]] += 1
            if item["image_path"] != seed_path:
                local_non_anchor_usage[item["image_path"]] += 1

    if len(selected) < candidate_count:
        errors.append(_build_error_row(
            error_type="seed_candidate_shortage",
            message="Could not produce the requested number of unique seeded bundles",
            extra={
                "requested_count": candidate_count,
                "generated_count": len(selected),
                "seed_image_path": seed_path,
            },
        ))

    return selected, errors


def _build_bundle_record(
    generation_mode: str,
    anchor_record: dict[str, Any],
    locked_categories: list[str],
    items: list[dict[str, Any]],
    bundle_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    now = _now_iso()
    return {
        "bundle_id": bundle_id or f"bundle_{uuid4().hex[:12]}",
        "generation_mode": generation_mode,
        "status": "pending",
        "score": round(_bundle_average_pairwise_score(items), 4),
        "cluster_key": _cluster_key(anchor_record),
        "anchor_category": anchor_record["category"],
        "anchor_image_path": str(anchor_record["source_image_path"]),
        "locked_categories": [category for category in BUNDLE_CATEGORIES if category in set(locked_categories)],
        "items": items,
        "created_at": str(created_at or now),
        "updated_at": now,
    }


def _bundle_average_pairwise_score(items: list[dict[str, Any]]) -> float:
    records = [item["semantic_tag"] if "semantic_tag" in item else item for item in items]
    scores: list[float] = []
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            scores.append(semantic_compatibility(left, right))
    return round(mean(scores), 4) if scores else 0.0


def semantic_compatibility(left: dict[str, Any], right: dict[str, Any]) -> float:
    score = 0.0
    left_primary = str(left.get("primary_style") or "")
    right_primary = str(right.get("primary_style") or "")
    left_secondary = str(left.get("secondary_style") or "")
    right_secondary = str(right.get("secondary_style") or "")
    left_color = str(left.get("color_family") or "")
    right_color = str(right.get("color_family") or "")
    left_brightness = str(left.get("color_brightness") or "")
    right_brightness = str(right.get("color_brightness") or "")
    left_main_material = str(left.get("main_material") or "")
    right_main_material = str(right.get("main_material") or "")
    left_secondary_materials = {
        str(value) for value in (left.get("secondary_materials") or [])
        if str(value).strip()
    }
    right_secondary_materials = {
        str(value) for value in (right.get("secondary_materials") or [])
        if str(value).strip()
    }

    if left_primary and left_primary == right_primary:
        score += 4
    if left_primary and left_primary == right_secondary:
        score += 2
    if right_primary and right_primary == left_secondary:
        score += 2
    if left_color and left_color == right_color:
        score += 3
    if left_brightness and left_brightness == right_brightness:
        score += 1
    if left_main_material and left_main_material == right_main_material:
        score += 2
    if left_main_material and left_main_material in right_secondary_materials:
        score += 1
    if right_main_material and right_main_material in left_secondary_materials:
        score += 1

    overlap = left_secondary_materials & right_secondary_materials
    if overlap:
        score += min(1.0, 0.5 * len(overlap))

    return round(score, 4)


def _bundle_item_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": record["category"],
        "image_path": str(record["source_image_path"]),
        "relative_path": record["relative_path"],
        "file_name": record["file_name"],
        "package_name": record["package_name"],
        "semantic_tag": dict(record),
    }


def _pool_item_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_path": str(record["source_image_path"]),
        "file_name": record["file_name"],
        "relative_path": record["relative_path"],
        "package_name": record["package_name"],
        "category": record["category"],
        "primary_style": record.get("primary_style"),
        "secondary_style": record.get("secondary_style"),
        "color_family": record.get("color_family"),
        "color_brightness": record.get("color_brightness"),
        "main_material": record.get("main_material"),
        "secondary_materials": list(record.get("secondary_materials") or []),
        "semantic_tag": dict(record),
    }


def _cluster_key(record: dict[str, Any]) -> str:
    return "|".join([
        str(record.get("primary_style") or ""),
        str(record.get("color_family") or ""),
        str(record.get("main_material") or ""),
    ])


def _workspace_stats(workspace: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(bundle["status"] for bundle in workspace)
    mode_counts = Counter(bundle["generation_mode"] for bundle in workspace)
    return {
        "total": len(workspace),
        "pending": status_counts.get("pending", 0),
        "accepted": status_counts.get("accepted", 0),
        "rejected": status_counts.get("rejected", 0),
        "auto": mode_counts.get("auto", 0),
        "seeded": mode_counts.get("seeded", 0),
    }


def _build_reuse_counts(workspace: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for bundle in workspace:
        for item in bundle["items"]:
            counts[item["image_path"]] += 1
    return counts


def _bump_bundle_reuse_counts(reuse_counts: Counter, bundle: dict[str, Any]) -> None:
    for item in bundle["items"]:
        reuse_counts[item["image_path"]] += 1


def _build_shortage_summary(
    pool_by_category: dict[str, list[dict[str, Any]]],
    workspace: list[dict[str, Any]],
    target_count: int,
) -> dict[str, Any]:
    usage = _build_reuse_counts(workspace)
    summary: dict[str, Any] = {}
    for category in BUNDLE_CATEGORIES:
        available_records = pool_by_category.get(category, [])
        remaining_capacity = sum(
            max(0, REUSE_LIMITS[category] - usage[str(record["source_image_path"])])
            for record in available_records
        )
        summary[category] = {
            "pool_size": len(available_records),
            "reuse_capacity": remaining_capacity,
            "target_count": target_count,
        }
    return summary


def _persist_workspace(
    input_root: Path,
    workspace: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> None:
    normalized_workspace = [_prepare_bundle_for_persist(bundle) for bundle in workspace]
    _write_jsonl(input_root / BUNDLE_CANDIDATE_FILE, normalized_workspace)
    _write_jsonl(input_root / BUNDLE_REVIEW_FILE, normalized_workspace)
    _write_jsonl(input_root / BUNDLE_ERROR_FILE, error_rows)


def _prepare_bundle_for_persist(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": bundle["bundle_id"],
        "generation_mode": bundle["generation_mode"],
        "status": bundle["status"],
        "score": round(float(bundle["score"]), 4),
        "cluster_key": bundle["cluster_key"],
        "anchor_category": bundle["anchor_category"],
        "anchor_image_path": bundle["anchor_image_path"],
        "locked_categories": list(bundle["locked_categories"]),
        "items": [
            {
                "category": item["category"],
                "image_path": item["image_path"],
                "relative_path": item["relative_path"],
                "file_name": item["file_name"],
                "package_name": item["package_name"],
                "semantic_tag": item["semantic_tag"],
            }
            for item in bundle["items"]
        ],
        "created_at": bundle["created_at"],
        "updated_at": bundle["updated_at"],
    }


def _load_error_rows(input_root: Path) -> list[dict[str, Any]]:
    error_path = input_root / BUNDLE_ERROR_FILE
    if not error_path.exists():
        return []
    return [row for row in _read_jsonl(error_path) if isinstance(row, dict)]


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _build_error_row(
    error_type: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "type": error_type,
        "message": message,
        "updated_at": _now_iso(),
    }
    if extra:
        payload.update(extra)
    return payload


def _resolve_image_path(path_value: Any, input_root: Path) -> Path:
    if not path_value:
        raise ValueError("Image path cannot be empty")
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (input_root / path).resolve()
    else:
        path = path.resolve()
    if input_root not in path.parents and path != input_root:
        raise ValueError(f"Path is outside the input directory: {path}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    return path


def _normalize_bundle_category(value: Any) -> str:
    normalized = normalize_catalog_type(str(value or "").strip().lower())
    if normalized not in BUNDLE_CATEGORY_SET:
        raise ValueError(f"Unsupported bundle category: {value}")
    return normalized


def _coerce_positive_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer") from None
    if result <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return result


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _emit(progress_callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(payload)
