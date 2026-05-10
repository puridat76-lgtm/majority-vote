from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as predictor_app
from tools.retrain_support import DATASET_ROOT, build_query_artifacts, create_components, list_samples


def exact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit retrain dataset quality and suspicious samples")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output", default=str(ROOT_DIR / "data" / "retrain_dataset_audit.json"))
    parser.add_argument("--detector-mode", choices=("haar", "yolo"), default="haar")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    samples = list_samples(dataset_root, splits=tuple(args.splits))
    encoder, cropper, face_classifier = create_components(detector_mode=args.detector_mode)
    artifacts = build_query_artifacts(samples, encoder=encoder, cropper=cropper, face_classifier=face_classifier)

    duplicates: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        duplicates[exact_hash(sample.file_path)].append(str(sample.file_path))
    exact_duplicates = [paths for paths in duplicates.values() if len(paths) > 1]

    suspicious_not_cat = []
    weak_cat_faces = []
    quality_issues = []
    split_summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for artifact in artifacts:
        sample = artifact.sample
        split_summary[sample.split][sample.group] += 1
        cat_face_score = float(artifact.face_check.get("cat_face_score") or 0.0)
        non_face_score = float(artifact.face_check.get("not_cat_face_score") or 0.0)
        localized_face = bool(artifact.detection.face_detected)
        quality_pass = bool(artifact.quality.get("quality_pass"))
        base_payload = {
            "split": sample.split,
            "group": sample.group,
            "label": sample.label,
            "file_path": str(sample.file_path),
            "source_name": sample.source_name,
            "cat_face_score": round(cat_face_score, 6),
            "non_face_score": round(non_face_score, 6),
            "localized_face_detected": localized_face,
            "detector_backend": artifact.detection.detector_backend,
            "quality_pass": quality_pass,
            "blur_score": round(float(artifact.quality.get("blur_score") or 0.0), 4),
            "brightness": round(float(artifact.quality.get("brightness") or 0.0), 4),
            "quality_reasons": artifact.quality.get("quality_reasons") or [],
        }
        if not quality_pass:
            quality_issues.append(base_payload)
        if sample.group == "not_cat" and (localized_face or cat_face_score >= 0.5):
            suspicious_not_cat.append(base_payload)
        if sample.group in {"known", "unknown_cat"} and (not localized_face and cat_face_score < 0.5):
            weak_cat_faces.append(base_payload)

    report = {
        "dataset_root": str(dataset_root),
        "detector_mode": args.detector_mode,
        "sample_count": len(samples),
        "split_summary": {split: dict(groups) for split, groups in split_summary.items()},
        "exact_duplicate_groups": exact_duplicates,
        "quality_issues": quality_issues,
        "suspicious_not_cat": suspicious_not_cat,
        "weak_cat_faces": weak_cat_faces,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
