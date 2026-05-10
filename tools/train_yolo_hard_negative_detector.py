from __future__ import annotations

import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT_DIR / "data" / "experiments" / "yolo_hard_negative_dataset" / "data.yaml"
DEFAULT_BASE = ROOT_DIR / "data" / "imports" / "cat_face_detector_zip" / "cat_face_detector" / "cat_face_detector.pt"
DEFAULT_PROJECT = ROOT_DIR / "data" / "experiments" / "yolo_hard_negative_runs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO detector on hard negative cat-face dataset.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--base-model", default=str(DEFAULT_BASE))
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="cat_face_detector_hardneg_v1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()

    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(Path(args.base_model)))
    model.train(
        data=str(Path(args.data)),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        fraction=args.fraction,
        project=str(Path(args.project)),
        name=args.name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
