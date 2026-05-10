from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT_DIR / "data" / "imports" / "cat_face_detector_zip" / "cat_face_detector" / "roboflow_dataset"
OUTPUT_ROOT = ROOT_DIR / "data" / "experiments" / "yolo_hard_negative_dataset"
PREPARED_ROOT = ROOT_DIR / "data" / "prepared" / "cat_retrain_dataset_cleaned_keep_notcat"
LIVE_NOT_CAT_ROOT = ROOT_DIR / "uploads" / "reference" / "not_cat"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def copy_labeled_split(source_split: str, target_split: str, output_root: Path) -> tuple[int, int]:
    source_images = SOURCE_ROOT / source_split / "images"
    source_labels = SOURCE_ROOT / source_split / "labels"
    target_images = output_root / "images" / target_split
    target_labels = output_root / "labels" / target_split
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    for image_path in sorted(source_images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        target_image = target_images / image_path.name
        shutil.copy2(image_path, target_image)
        image_count += 1

        source_label = source_labels / f"{image_path.stem}.txt"
        target_label = target_labels / f"{image_path.stem}.txt"
        if source_label.exists():
            shutil.copy2(source_label, target_label)
            label_count += 1
        else:
            target_label.write_text("", encoding="utf-8")
    return image_count, label_count


def add_negative_image(path: Path, target_split: str, output_root: Path, prefix: str) -> bool:
    if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    digest = file_hash(path)
    safe_name = f"{prefix}_{digest}{path.suffix.lower()}"
    target_image = output_root / "images" / target_split / safe_name
    target_label = output_root / "labels" / target_split / f"{Path(safe_name).stem}.txt"
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target_image)
    target_label.write_text("", encoding="utf-8")
    return True


def add_negative_sources(output_root: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}

    for path in sorted(LIVE_NOT_CAT_ROOT.iterdir()) if LIVE_NOT_CAT_ROOT.exists() else []:
        counts["train"] += int(add_negative_image(path, "train", output_root, "live_not_cat"))

    split_map = {"train": "train", "val": "val", "test": "test"}
    for prepared_split, target_split in split_map.items():
        not_cat_root = PREPARED_ROOT / prepared_split / "not_cat"
        if not not_cat_root.exists():
            continue
        for path in sorted(not_cat_root.iterdir()):
            counts[target_split] += int(add_negative_image(path, target_split, output_root, f"prepared_{prepared_split}_not_cat"))
    return counts


def write_data_yaml(output_root: Path) -> None:
    yaml_text = "\n".join([
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        "  0: cat_face",
        "",
    ])
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO cat-face dataset with hard negative images.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    labeled = {
        "train": copy_labeled_split("train", "train", output_root),
        "val": copy_labeled_split("valid", "val", output_root),
        "test": copy_labeled_split("test", "test", output_root),
    }
    negatives = add_negative_sources(output_root)
    write_data_yaml(output_root)

    print("dataset:", output_root)
    for split, (images, labels) in labeled.items():
        print(f"{split}: labeled_images={images} copied_labels={labels} hard_negatives={negatives.get(split, 0)}")


if __name__ == "__main__":
    main()
