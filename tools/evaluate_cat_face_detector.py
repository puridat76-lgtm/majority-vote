from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import (
    CAT_FACE_DETECTOR_MODEL_CANDIDATES,
    FaceCropper,
    CatFaceDetector,
    VALID_EXTENSIONS,
    read_image,
)


def iter_images(base: Path):
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS and not any(part.startswith(".") for part in path.parts):
            yield path


def classify_group(path: Path) -> str:
    parts = {part for part in path.parts}
    if "not_cat" in parts:
        return "not_cat"
    if "unknown_cat" in parts:
        return "unknown_cat"
    if "known" in parts:
        return "known"
    return "other"


def evaluate(dataset_root: Path) -> dict[str, object]:
    haar_cropper = FaceCropper()
    haar_cropper.prefer_yolo = False
    yolo_detector = CatFaceDetector(CAT_FACE_DETECTOR_MODEL_CANDIDATES)

    results: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)

    for path in iter_images(dataset_root):
        group = classify_group(path)
        image = read_image(path)
        if image is None:
            continue
        haar = haar_cropper.detect_and_crop(image)
        yolo_box = yolo_detector.detect_largest_face(image) if yolo_detector.available else None
        results[group]["total"] += 1
        results[group]["haar_face"] += int(bool(haar.face_detected))
        results[group]["yolo_face"] += int(bool(yolo_box))
        results[group]["either_face"] += int(bool(haar.face_detected or yolo_box))
        if len(examples[group]) < 10 and (haar.face_detected or yolo_box):
            examples[group].append({
                "file": str(path),
                "haar_face": bool(haar.face_detected),
                "yolo_face": bool(yolo_box),
                "haar_box": haar.face_box,
                "yolo_box": yolo_box,
            })

    summary = {
        group: {
            "total": int(counter["total"]),
            "haar_face_rate": round(counter["haar_face"] / max(counter["total"], 1), 4),
            "yolo_face_rate": round(counter["yolo_face"] / max(counter["total"], 1), 4),
            "either_face_rate": round(counter["either_face"] / max(counter["total"], 1), 4),
        }
        for group, counter in results.items()
    }
    return {
        "dataset_root": str(dataset_root),
        "yolo_available": yolo_detector.available,
        "yolo_backend": yolo_detector.backend_name,
        "summary": summary,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Haar vs YOLO cat-face detector on dataset folders.")
    parser.add_argument("--dataset", default="data/prepared/cat_retrain_dataset/test", help="Dataset split or root to evaluate")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    report = evaluate(Path(args.dataset).resolve())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"dataset: {report['dataset_root']}")
    print(f"yolo: {report['yolo_available']} ({report['yolo_backend']})")
    for group, metrics in report["summary"].items():
        print(
            f"[{group}] total={metrics['total']} "
            f"haar={metrics['haar_face_rate']:.4f} "
            f"yolo={metrics['yolo_face_rate']:.4f} "
            f"either={metrics['either_face_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
