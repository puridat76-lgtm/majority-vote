from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.retrain_support import DATASET_ROOT, split_counts


def relative_item_path(dataset_root: Path, absolute_path: str) -> Path:
    return Path(absolute_path).resolve().relative_to(dataset_root.resolve())


def add_reason(
    flagged: dict[Path, list[dict[str, Any]]],
    dataset_root: Path,
    absolute_path: str,
    *,
    reason: str,
    details: dict[str, Any],
) -> None:
    rel_path = relative_item_path(dataset_root, absolute_path)
    flagged.setdefault(rel_path, []).append({"reason": reason, **details})


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Retrain Dataset Cleanup Plan",
        "",
        f"- Source dataset: `{plan['source_dataset']}`",
        f"- Clean dataset: `{plan['clean_dataset']}`",
        f"- Cleaned splits: `{', '.join(plan['clean_splits'])}`",
        f"- Weak known threshold: `{plan['rules']['weak_known_threshold']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total removed files | {plan['removed_total']} |",
    ]
    for split, groups in sorted(plan["removed_counts"].items()):
        for group, count in sorted(groups.items()):
            lines.append(f"| Removed `{split}/{group}` | {count} |")
    lines.extend(
        [
            "",
            "## Source Counts",
            "",
            "```json",
            json.dumps(plan["source_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Clean Counts",
            "",
            "```json",
            json.dumps(plan["clean_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Remove / Replace List",
            "",
        ]
    )
    for rel_path, reasons in plan["removed_files"].items():
        reason_text = ", ".join(reason["reason"] for reason in reasons)
        lines.append(f"- `{rel_path}`: {reason_text}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned retrain dataset from audit report")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--audit-report", default=str(ROOT_DIR / "data" / "retrain_dataset_audit.json"))
    parser.add_argument("--output-root", default=str(ROOT_DIR / "data" / "prepared" / "cat_retrain_dataset_cleaned"))
    parser.add_argument("--json-output", default=str(ROOT_DIR / "data" / "retrain_dataset_cleanup_plan.json"))
    parser.add_argument("--markdown-output", default=str(ROOT_DIR / "data" / "retrain_dataset_cleanup_plan.md"))
    parser.add_argument("--clean-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--weak-known-threshold", type=float, default=0.25)
    parser.add_argument("--keep-duplicate-known", action="store_true")
    parser.add_argument("--keep-quality-issues", action="store_true")
    parser.add_argument("--keep-suspicious-not-cat", action="store_true")
    parser.add_argument("--keep-weak-known", action="store_true")
    parser.add_argument("--keep-weak-unknown", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    audit_report = json.loads(Path(args.audit_report).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)

    clean_splits = set(args.clean_splits)
    flagged: dict[Path, list[dict[str, Any]]] = {}
    remove_duplicate_known = not args.keep_duplicate_known
    remove_quality_issues = not args.keep_quality_issues
    remove_suspicious_not_cat = not args.keep_suspicious_not_cat
    remove_weak_known = not args.keep_weak_known
    remove_weak_unknown = not args.keep_weak_unknown

    if remove_duplicate_known:
        for group in audit_report.get("exact_duplicate_groups", []):
            paths = sorted(relative_item_path(dataset_root, path) for path in group)
            for duplicate_path in paths[1:]:
                if duplicate_path.parts and duplicate_path.parts[0] in clean_splits:
                    flagged.setdefault(duplicate_path, []).append({"reason": "exact_duplicate"})

    if remove_quality_issues:
        for item in audit_report.get("quality_issues", []):
            if item["split"] not in clean_splits:
                continue
            add_reason(
                flagged,
                dataset_root,
                item["file_path"],
                reason="quality_issue",
                details={
                    "quality_reasons": item.get("quality_reasons", []),
                    "cat_face_score": item.get("cat_face_score"),
                },
            )

    if remove_suspicious_not_cat:
        for item in audit_report.get("suspicious_not_cat", []):
            if item["split"] not in clean_splits:
                continue
            add_reason(
                flagged,
                dataset_root,
                item["file_path"],
                reason="suspicious_not_cat",
                details={
                    "localized_face_detected": item.get("localized_face_detected"),
                    "cat_face_score": item.get("cat_face_score"),
                },
            )

    for item in audit_report.get("weak_cat_faces", []):
        if item["split"] not in clean_splits:
            continue
        if item["group"] == "known" and remove_weak_known and float(item.get("cat_face_score") or 0.0) < args.weak_known_threshold:
            add_reason(
                flagged,
                dataset_root,
                item["file_path"],
                reason="weak_known_face",
                details={"cat_face_score": item.get("cat_face_score")},
            )
        if item["group"] == "unknown_cat" and remove_weak_unknown:
            add_reason(
                flagged,
                dataset_root,
                item["file_path"],
                reason="weak_unknown_face",
                details={"cat_face_score": item.get("cat_face_score")},
            )

    removed_counts: dict[str, Counter[str]] = defaultdict(Counter)
    removed_files_report: dict[str, list[dict[str, Any]]] = {}
    kept_files = 0

    for source_path in dataset_root.rglob("*"):
        if not source_path.is_file():
            continue
        rel_path = source_path.relative_to(dataset_root)
        if rel_path in flagged:
            split = rel_path.parts[0] if rel_path.parts else "unknown"
            group = rel_path.parts[1] if len(rel_path.parts) > 1 else "unknown"
            removed_counts[split][group] += 1
            removed_files_report[str(rel_path)] = flagged[rel_path]
            continue
        destination = output_root / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        kept_files += 1

    from tools.retrain_support import list_samples

    source_counts = split_counts(list_samples(dataset_root, splits=("train", "val", "test")))
    clean_counts = split_counts(list_samples(output_root, splits=("train", "val", "test")))

    plan = {
        "source_dataset": str(dataset_root),
        "clean_dataset": str(output_root),
        "clean_splits": sorted(clean_splits),
        "rules": {
            "weak_known_threshold": args.weak_known_threshold,
            "remove_duplicate_known": remove_duplicate_known,
            "remove_quality_issues": remove_quality_issues,
            "remove_suspicious_not_cat": remove_suspicious_not_cat,
            "remove_weak_known": remove_weak_known,
            "remove_weak_unknown": remove_weak_unknown,
        },
        "removed_total": sum(sum(counter.values()) for counter in removed_counts.values()),
        "kept_total": kept_files,
        "source_counts": source_counts,
        "clean_counts": clean_counts,
        "removed_counts": {split: dict(counter) for split, counter in removed_counts.items()},
        "removed_files": removed_files_report,
    }

    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_output = Path(args.markdown_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(plan), encoding="utf-8")

    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
