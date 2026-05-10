from __future__ import annotations

import argparse
import json
import sys
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
)


def rank_key(row: dict[str, Any]) -> tuple[float, float]:
    strongest_wrong = max(float(row.get("best_known_score") or 0.0), float(row.get("best_unknown_score") or 0.0), float(row.get("best_not_cat_score") or 0.0))
    cat_face_score = float(row.get("cat_face_score") or 0.0)
    return (strongest_wrong, cat_face_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives / confusing queries from retrain dataset")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--index-split", default="train")
    parser.add_argument("--query-split", default="test")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--calibrator-path", default=str(predictor_app.OPEN_SET_CALIBRATOR_PATH))
    parser.add_argument("--detector-mode", choices=("haar", "yolo"), default="haar")
    parser.add_argument("--topn", type=int, default=15)
    parser.add_argument("--output", default=str(ROOT_DIR / "data" / "hard_negatives.json"))
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model_path) if args.model_path else None
    calibrator_path = Path(args.calibrator_path) if args.calibrator_path else None
    encoder, cropper, face_classifier = create_components(model_path=model_path, detector_mode=args.detector_mode)
    samples = list_samples(dataset_root, splits=(args.index_split, args.query_split))
    index_samples = [sample for sample in samples if sample.split == args.index_split]
    query_samples = [sample for sample in samples if sample.split == args.query_split]
    index_records, index_vectors = build_reference_bank(index_samples, encoder=encoder, cropper=cropper)
    artifacts = build_query_artifacts(query_samples, encoder=encoder, cropper=cropper, face_classifier=face_classifier)
    calibrator = predictor_app.OpenSetCalibrator(calibrator_path) if calibrator_path and calibrator_path.exists() else None

    rows = [
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
    errors = [row for row in rows if not row["correct"]]
    by_group: dict[str, list[dict[str, Any]]] = {}
    for group in ("known", "unknown_cat", "not_cat"):
        group_errors = [row for row in errors if row["expected_group"] == group]
        by_group[group] = sorted(group_errors, key=rank_key, reverse=True)[: args.topn]

    output = {
        "dataset_root": str(dataset_root),
        "index_split": args.index_split,
        "query_split": args.query_split,
        "detector_mode": args.detector_mode,
        "calibrator_used": bool(calibrator and calibrator.available),
        "total_errors": len(errors),
        "top_errors_by_group": by_group,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
