from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_ALIASES = {"train": "train", "val": "val", " test": "test", "test": "test", " val": "val"}
CLASS_NAMES = ("known", "unknown_cat", "not_cat")


def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts) or "__MACOSX" in path.parts


def should_keep_file(path: Path) -> bool:
    return path.is_file() and not is_hidden(path) and path.suffix.lower() in VALID_EXTENSIONS


def resolve_dataset_root(source: Path) -> Path:
    if (source / "train").exists() or (source / " val").exists() or (source / "val").exists():
        return source
    children = [child for child in source.iterdir() if child.is_dir() and not is_hidden(child)]
    if len(children) == 1:
        return resolve_dataset_root(children[0])
    raise FileNotFoundError(f"Could not resolve dataset root under {source}")


def normalize_split_dirs(root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for child in root.iterdir():
        if not child.is_dir() or is_hidden(child):
            continue
        normalized = SPLIT_ALIASES.get(child.name)
        if normalized:
            mapping[normalized] = child
    missing = [split for split in ("train", "val", "test") if split not in mapping]
    if missing:
        raise FileNotFoundError(f"Missing expected splits: {missing}")
    return mapping


def copy_known_split(source: Path, target: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    target.mkdir(parents=True, exist_ok=True)
    for cat_dir in sorted(source.iterdir(), key=lambda p: p.name):
        if not cat_dir.is_dir() or is_hidden(cat_dir):
            continue
        dest_dir = target / cat_dir.name.strip()
        dest_dir.mkdir(parents=True, exist_ok=True)
        for image in sorted(cat_dir.iterdir(), key=lambda p: p.name):
            if not should_keep_file(image):
                continue
            shutil.copy2(image, dest_dir / image.name)
            counts[cat_dir.name.strip()] += 1
    return counts


def copy_flat_split(source: Path, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for image in sorted(source.iterdir(), key=lambda p: p.name):
        if not should_keep_file(image):
            continue
        shutil.copy2(image, target / image.name)
        count += 1
    return count


def prepare_dataset(source: Path, output: Path) -> dict[str, object]:
    root = resolve_dataset_root(source)
    split_dirs = normalize_split_dirs(root)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"source_root": str(root), "output_root": str(output), "splits": {}}
    for split, split_source in split_dirs.items():
        split_target = output / split
        split_target.mkdir(parents=True, exist_ok=True)
        known_counts = copy_known_split(split_source / "known", split_target / "known")
        unknown_count = copy_flat_split(split_source / "unknown_cat", split_target / "unknown_cat")
        not_cat_count = copy_flat_split(split_source / "not_cat", split_target / "not_cat")
        summary["splits"][split] = {
            "known_total": int(sum(known_counts.values())),
            "known_cats": int(len(known_counts)),
            "top_known_counts": known_counts.most_common(10),
            "unknown_cat": int(unknown_count),
            "not_cat": int(not_cat_count),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize cat retrain dataset into train/val/test structure.")
    parser.add_argument("--source", default="data/imports/cat_retrain_dataset_zip", help="Path to extracted or raw dataset folder")
    parser.add_argument("--output", default="data/prepared/cat_retrain_dataset", help="Output path for normalized dataset")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    summary = prepare_dataset(source, output)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("Prepared dataset")
    print(f"source: {summary['source_root']}")
    print(f"output: {summary['output_root']}")
    for split, payload in summary["splits"].items():
        print(f"[{split}] known_total={payload['known_total']} known_cats={payload['known_cats']} unknown_cat={payload['unknown_cat']} not_cat={payload['not_cat']}")


if __name__ == "__main__":
    main()
