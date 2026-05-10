from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as predictor_app
from tools.retrain_support import (
    DATASET_ROOT,
    build_query_artifacts,
    build_reference_bank,
    classify_artifact,
    create_components,
    list_samples,
    summarize_rows,
)


def build_confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[row["expected_label"]][row["predicted_label"]] += 1
    return {
        expected: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
        for expected, counter in sorted(matrix.items())
    }


def evaluate(
    dataset_root: Path,
    index_split: str,
    query_splits: tuple[str, ...],
    model_path: Path | None,
    calibrator_path: Path | None,
    detector_mode: str,
) -> dict[str, Any]:
    encoder, cropper, face_classifier = create_components(model_path=model_path, detector_mode=detector_mode)
    all_samples = list_samples(dataset_root, splits=(index_split, *query_splits))
    index_samples = [sample for sample in all_samples if sample.split == index_split]
    query_samples = [sample for sample in all_samples if sample.split in query_splits]
    index_records, index_vectors = build_reference_bank(index_samples, encoder=encoder, cropper=cropper)
    artifacts = build_query_artifacts(query_samples, encoder=encoder, cropper=cropper, face_classifier=face_classifier)
    calibrator = predictor_app.OpenSetCalibrator(calibrator_path) if calibrator_path else None

    baseline_rows = [
        classify_artifact(
            artifact,
            index_records=index_records,
            index_vectors=index_vectors,
            top_k=predictor_app.TOP_K,
            vote_relative_margin=predictor_app.VOTE_RELATIVE_MARGIN,
            cat_face_support_threshold=predictor_app.CAT_FACE_SUPPORT_THRESHOLD,
            calibrator=None,
        )
        for artifact in artifacts
    ]
    calibrated_rows = [
        classify_artifact(
            artifact,
            index_records=index_records,
            index_vectors=index_vectors,
            top_k=predictor_app.TOP_K,
            vote_relative_margin=predictor_app.VOTE_RELATIVE_MARGIN,
            cat_face_support_threshold=predictor_app.CAT_FACE_SUPPORT_THRESHOLD,
            calibrator=calibrator,
        )
        for artifact in artifacts
    ]

    return {
        "dataset_root": str(dataset_root),
        "index_split": index_split,
        "query_splits": list(query_splits),
        "model_path": str(model_path or predictor_app.MODEL_PATH),
        "calibrator_path": str(calibrator_path) if calibrator_path else None,
        "detector_mode": detector_mode,
        "index_size": len(index_records),
        "query_size": len(artifacts),
        "baseline": {
            **summarize_rows(baseline_rows),
            "confusion": build_confusion(baseline_rows),
            "rows": baseline_rows,
        },
        "calibrated": {
            **summarize_rows(calibrated_rows),
            "confusion": build_confusion(calibrated_rows),
            "rows": calibrated_rows,
        },
    }


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_root": report["dataset_root"],
        "index_split": report["index_split"],
        "query_splits": report["query_splits"],
        "model_path": report["model_path"],
        "calibrator_path": report["calibrator_path"],
        "baseline": {
            "accuracy": report["baseline"]["accuracy"],
            "group_metrics": report["baseline"]["group_metrics"],
        },
        "calibrated": {
            "accuracy": report["calibrated"]["accuracy"],
            "group_metrics": report["calibrated"]["group_metrics"],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    def section(title: str, payload: dict[str, Any]) -> list[str]:
        lines = [
            f"## {title}",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Overall accuracy | {payload['accuracy']:.4f} |",
        ]
        for group, group_payload in payload["group_metrics"].items():
            lines.append(f"| {group} accuracy | {group_payload['accuracy']:.4f} |")
        return lines

    lines = [
        "# Retrain Dataset Evaluation",
        "",
        f"- Dataset: `{report['dataset_root']}`",
        f"- Index split: `{report['index_split']}`",
        f"- Query splits: `{', '.join(report['query_splits'])}`",
        f"- Model: `{report['model_path']}`",
        f"- Calibrator: `{report['calibrator_path'] or 'none'}`",
        f"- Detector: `{report['detector_mode']}`",
        "",
    ]
    lines.extend(section("Baseline", report["baseline"]))
    lines.extend([""])
    lines.extend(section("With Calibrator", report["calibrated"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictor pipeline on prepared retrain dataset")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--index-split", default="train")
    parser.add_argument("--query-splits", nargs="+", default=["test"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--calibrator-path", default=str(predictor_app.OPEN_SET_CALIBRATOR_PATH))
    parser.add_argument("--detector-mode", choices=("haar", "yolo"), default="haar")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else None
    calibrator_path = Path(args.calibrator_path) if args.calibrator_path else None
    report = evaluate(
        dataset_root=Path(args.dataset_root),
        index_split=args.index_split,
        query_splits=tuple(args.query_splits),
        model_path=model_path,
        calibrator_path=calibrator_path if calibrator_path and calibrator_path.exists() else None,
        detector_mode=args.detector_mode,
    )
    if args.summary_only:
        report = compact_report(report)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
