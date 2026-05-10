from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


def split_rows(
    dataset_root: Path,
    *,
    model_path: Path | None,
    index_split: str,
    query_split: str,
    detector_mode: str,
) -> list[dict[str, Any]]:
    encoder, cropper, face_classifier = create_components(model_path=model_path, detector_mode=detector_mode)
    samples = list_samples(dataset_root, splits=(index_split, query_split))
    index_samples = [sample for sample in samples if sample.split == index_split]
    query_samples = [sample for sample in samples if sample.split == query_split]
    index_records, index_vectors = build_reference_bank(index_samples, encoder=encoder, cropper=cropper)
    artifacts = build_query_artifacts(query_samples, encoder=encoder, cropper=cropper, face_classifier=face_classifier)
    return [
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


def rows_to_matrix(rows: list[dict[str, Any]], feature_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if feature_names is None:
        feature_names = sorted(rows[0]["decision_features"].keys())
    x = np.asarray(
        [[float(row["decision_features"].get(name, 0.0)) for name in feature_names] for row in rows],
        dtype="float32",
    )
    y = np.asarray([row["expected_group"] for row in rows], dtype=object)
    return x, y, feature_names


def softmax_logits(x: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    logits = x @ coef.T + intercept
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-8, None)


def apply_override(rows: list[dict[str, Any]], probabilities: np.ndarray, classes: list[str], min_prob: float, min_margin: float) -> dict[str, Any]:
    correct = 0
    totals: dict[str, dict[str, int]] = {}
    for row, sample_probs in zip(rows, probabilities, strict=False):
        ranking = np.argsort(sample_probs)[::-1]
        top_idx = int(ranking[0])
        second_prob = float(sample_probs[int(ranking[1])]) if len(ranking) > 1 else 0.0
        top_prob = float(sample_probs[top_idx])
        top_group = classes[top_idx]
        confident = top_prob >= min_prob and (top_prob - second_prob) >= min_margin
        final_label = row["predicted_label"]
        if confident and top_group in {"unknown_cat", "not_cat"}:
            final_label = top_group
        is_correct = final_label == row["expected_label"]
        correct += int(is_correct)
        group_payload = totals.setdefault(row["expected_group"], {"total": 0, "correct": 0})
        group_payload["total"] += 1
        group_payload["correct"] += int(is_correct)
    group_metrics = {
        group: round(payload["correct"] / payload["total"], 4) if payload["total"] else 0.0
        for group, payload in sorted(totals.items())
    }
    macro = float(np.mean(list(group_metrics.values()))) if group_metrics else 0.0
    return {
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "macro_accuracy": round(macro, 4),
        "group_metrics": group_metrics,
    }


def tune_thresholds(val_rows: list[dict[str, Any]], val_probs: np.ndarray, classes: list[str]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for min_prob in (0.45, 0.5, 0.55, 0.6, 0.65):
        for min_margin in (0.0, 0.03, 0.05, 0.08, 0.1):
            metrics = apply_override(val_rows, val_probs, classes, min_prob, min_margin)
            candidate = {
                "min_probability": float(min_prob),
                "min_margin": float(min_margin),
                **metrics,
            }
            if best is None or (
                candidate["macro_accuracy"],
                candidate["accuracy"],
                -candidate["min_probability"],
                -candidate["min_margin"],
            ) > (
                best["macro_accuracy"],
                best["accuracy"],
                -best["min_probability"],
                -best["min_margin"],
            ):
                best = candidate
    return best or {
        "min_probability": predictor_app.OPEN_SET_CALIBRATOR_MIN_PROB,
        "min_margin": predictor_app.OPEN_SET_CALIBRATOR_MIN_MARGIN,
        "accuracy": 0.0,
        "macro_accuracy": 0.0,
        "group_metrics": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train open-set calibrator for known/unknown/not_cat group decision")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default=str(predictor_app.OPEN_SET_CALIBRATOR_PATH))
    parser.add_argument("--report", default=str(ROOT_DIR / "data" / "open_set_calibrator_report.json"))
    parser.add_argument("--detector-mode", choices=("haar", "yolo"), default="haar")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model_path) if args.model_path else None
    train_rows = split_rows(dataset_root, model_path=model_path, index_split="train", query_split="train", detector_mode=args.detector_mode)
    val_rows = split_rows(dataset_root, model_path=model_path, index_split="train", query_split="val", detector_mode=args.detector_mode)
    test_rows = split_rows(dataset_root, model_path=model_path, index_split="train", query_split="test", detector_mode=args.detector_mode)

    x_train, y_train, feature_names = rows_to_matrix(train_rows)
    x_val, _, _ = rows_to_matrix(val_rows, feature_names)
    x_test, _, _ = rows_to_matrix(test_rows, feature_names)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(x_train_scaled, y_train)

    classes = [str(name) for name in clf.classes_]
    val_probs = softmax_logits(x_val_scaled, clf.coef_, clf.intercept_)
    tuned = tune_thresholds(val_rows, val_probs, classes)
    test_probs = softmax_logits(x_test_scaled, clf.coef_, clf.intercept_)
    test_metrics = apply_override(
        test_rows,
        test_probs,
        classes,
        tuned["min_probability"],
        tuned["min_margin"],
    )

    payload = {
        "feature_names": feature_names,
        "classes": classes,
        "coefficients": clf.coef_.astype("float32").tolist(),
        "intercept": clf.intercept_.astype("float32").tolist(),
        "mean": scaler.mean_.astype("float32").tolist(),
        "scale": scaler.scale_.astype("float32").tolist(),
        "min_probability": tuned["min_probability"],
        "min_margin": tuned["min_margin"],
        "metadata": {
            "dataset_root": str(dataset_root),
            "index_split": "train",
            "trained_with_model": str(model_path or predictor_app.MODEL_PATH),
            "detector_mode": args.detector_mode,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "val_tuned": tuned,
        "test_metrics": test_metrics,
        "output": str(output_path),
        "classes": classes,
        "feature_names": feature_names,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
