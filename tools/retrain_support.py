from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as predictor_app


DATASET_ROOT = ROOT_DIR / "data" / "prepared" / "cat_retrain_dataset"


@dataclass
class PreparedSample:
    split: str
    group: str
    label: str
    file_path: Path
    source_name: str


@dataclass
class PreparedArtifact:
    sample: PreparedSample
    quality: dict[str, Any]
    detection: predictor_app.DetectionResult
    face_check: dict[str, Any]
    query_vector: np.ndarray


def build_decision_context(
    artifact: PreparedArtifact,
    *,
    index_records: list[dict[str, Any]],
    index_vectors: np.ndarray,
    top_k: int = predictor_app.TOP_K,
    vote_relative_margin: float = predictor_app.VOTE_RELATIVE_MARGIN,
    cat_face_support_threshold: float = predictor_app.CAT_FACE_SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    similarities = np.dot(index_vectors, artifact.query_vector).astype("float32")
    query_cat_face_score = float(artifact.face_check.get("cat_face_score") or 0.0)
    query_cat_like = (
        bool(artifact.face_check.get("available"))
        and query_cat_face_score >= predictor_app.CAT_LIKE_QUERY_THRESHOLD
    )
    exclude_file_path = str(artifact.sample.file_path.resolve())
    top_candidates = predictor_app.build_top_candidates(
        index_records,
        similarities,
        top_k=top_k,
        query_cat_like=query_cat_like,
        exclude_file_path=exclude_file_path,
    )
    score_summary = predictor_app.build_score_summary(
        index_records,
        similarities,
        face_check=artifact.face_check,
        exact_known_label=None,
        query_cat_like=query_cat_like,
        exclude_file_path=exclude_file_path,
    )
    effective_detection = predictor_app.effective_detection_from_face_support(
        artifact.detection,
        artifact.face_check,
        threshold=cat_face_support_threshold,
    )
    previous_margin = predictor_app.VOTE_RELATIVE_MARGIN
    predictor_app.VOTE_RELATIVE_MARGIN = vote_relative_margin
    try:
        vote_candidates = predictor_app.effective_vote_candidates(top_candidates)
    finally:
        predictor_app.VOTE_RELATIVE_MARGIN = previous_margin
    features = predictor_app.extract_group_decision_features(
        top_candidates,
        vote_candidates,
        effective_detection,
        artifact.quality,
        artifact.face_check,
        score_summary,
    )
    return {
        "similarities": similarities,
        "query_cat_like": query_cat_like,
        "top_candidates": top_candidates,
        "score_summary": score_summary,
        "effective_detection": effective_detection,
        "vote_candidates": vote_candidates,
        "features": features,
    }


def list_samples(
    dataset_root: Path = DATASET_ROOT,
    splits: Iterable[str] = ("train", "val", "test"),
) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    for split in splits:
        split_root = dataset_root / split
        known_root = split_root / "known"
        if known_root.exists():
            for cat_dir in sorted(path for path in known_root.iterdir() if path.is_dir()):
                for path in sorted(path for path in cat_dir.iterdir() if path.is_file()):
                    samples.append(PreparedSample(split=split, group="known", label=cat_dir.name, file_path=path, source_name=path.name))
        for group in ("unknown_cat", "not_cat"):
            group_root = split_root / group
            if not group_root.exists():
                continue
            for path in sorted(path for path in group_root.iterdir() if path.is_file()):
                samples.append(PreparedSample(split=split, group=group, label=group, file_path=path, source_name=path.name))
    return samples


def split_counts(samples: Iterable[PreparedSample]) -> dict[str, dict[str, int]]:
    payload: dict[str, dict[str, int]] = {}
    for sample in samples:
        split_payload = payload.setdefault(sample.split, {"known": 0, "unknown_cat": 0, "not_cat": 0})
        split_payload[sample.group] += 1
    return payload


def create_components(
    model_path: Path | None = None,
    *,
    detector_mode: str | None = None,
) -> tuple[predictor_app.EncoderBackend, predictor_app.FaceCropper, predictor_app.CatFaceBinaryClassifier]:
    encoder = predictor_app.EncoderBackend(model_path or predictor_app.MODEL_PATH)
    face_classifier = predictor_app.CatFaceBinaryClassifier(
        predictor_app.CAT_FACE_MODEL_CANDIDATES,
        predictor_app.CAT_FACE_LABELS_CANDIDATES,
    )
    cropper = predictor_app.FaceCropper(detector_mode=detector_mode, face_classifier=face_classifier)
    return encoder, cropper, face_classifier


def build_reference_bank(
    samples: Iterable[PreparedSample],
    *,
    encoder: predictor_app.EncoderBackend,
    cropper: predictor_app.FaceCropper,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    image_batch: list[np.ndarray] = []
    record_batch: list[dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal image_batch, record_batch
        if not image_batch:
            return
        encoded = encoder.encode_many(image_batch)
        for record, vector in zip(record_batch, encoded, strict=False):
            records.append(record)
            vectors.append(vector)
        image_batch = []
        record_batch = []

    for sample in samples:
        image = predictor_app.read_image(sample.file_path)
        if image is None:
            continue
        detection = cropper.detect_and_crop(image)
        image_batch.append(detection.crop_bgr)
        record_batch.append({
            "label": sample.label,
            "group": sample.group,
            "cat_id": None,
            "image_name": sample.source_name,
            "file_path": str(sample.file_path.resolve()),
            "split": sample.split,
            "crop_strategy": detection.crop_strategy,
            "face_detected": detection.face_detected,
            "detector_backend": detection.detector_backend,
        })
        if len(image_batch) >= predictor_app.ENCODE_BATCH_SIZE:
            flush_batch()
    flush_batch()
    if not vectors:
        return records, np.empty((0, 0), dtype="float32")
    return records, np.vstack(vectors).astype("float32")


def build_query_artifacts(
    samples: Iterable[PreparedSample],
    *,
    encoder: predictor_app.EncoderBackend,
    cropper: predictor_app.FaceCropper,
    face_classifier: predictor_app.CatFaceBinaryClassifier,
) -> list[PreparedArtifact]:
    artifacts: list[PreparedArtifact] = []
    image_batch: list[np.ndarray] = []
    pending: list[tuple[PreparedSample, dict[str, Any], predictor_app.DetectionResult, dict[str, Any]]] = []

    def flush_batch() -> None:
        nonlocal image_batch, pending
        if not image_batch:
            return
        encoded = encoder.encode_many(image_batch)
        for (sample, quality, detection, face_check), vector in zip(pending, encoded, strict=False):
            artifacts.append(PreparedArtifact(sample=sample, quality=quality, detection=detection, face_check=face_check, query_vector=vector))
        image_batch = []
        pending = []

    for sample in samples:
        image_bytes = sample.file_path.read_bytes()
        image = predictor_app.read_image_bytes(image_bytes)
        if image is None:
            continue
        quality = predictor_app.compute_quality(image)
        detection = cropper.detect_and_crop(image)
        face_check = face_classifier.predict(detection.crop_bgr)
        image_batch.append(detection.crop_bgr)
        pending.append((sample, quality, detection, face_check))
        if len(image_batch) >= predictor_app.ENCODE_BATCH_SIZE:
            flush_batch()
    flush_batch()
    return artifacts


def classify_artifact(
    artifact: PreparedArtifact,
    *,
    index_records: list[dict[str, Any]],
    index_vectors: np.ndarray,
    top_k: int = predictor_app.TOP_K,
    vote_relative_margin: float = predictor_app.VOTE_RELATIVE_MARGIN,
    cat_face_support_threshold: float = predictor_app.CAT_FACE_SUPPORT_THRESHOLD,
    calibrator: predictor_app.OpenSetCalibrator | None = None,
) -> dict[str, Any]:
    context = build_decision_context(
        artifact,
        index_records=index_records,
        index_vectors=index_vectors,
        top_k=top_k,
        vote_relative_margin=vote_relative_margin,
        cat_face_support_threshold=cat_face_support_threshold,
    )
    top_candidates = context["top_candidates"]
    score_summary = context["score_summary"]
    effective_detection = context["effective_detection"]
    vote_candidates = context["vote_candidates"]

    previous_margin = predictor_app.VOTE_RELATIVE_MARGIN
    predictor_app.VOTE_RELATIVE_MARGIN = vote_relative_margin
    try:
        decision = predictor_app.decide_prediction(top_candidates, artifact.quality, effective_detection, score_summary)
        decision, calibrator_result = predictor_app.apply_open_set_calibrator(
            calibrator,
            current_decision=decision,
            top_candidates=top_candidates,
            quality=artifact.quality,
            detection=effective_detection,
            face_check=artifact.face_check,
            score_summary=score_summary,
        )
    finally:
        predictor_app.VOTE_RELATIVE_MARGIN = previous_margin

    raw_summary = predictor_app.summarize_classes(top_candidates)
    effective_summary = predictor_app.summarize_classes(vote_candidates)
    winner = effective_summary[0] if effective_summary else {"label": None, "votes": 0}
    winner_top10_votes = next((int(row["votes"]) for row in raw_summary if row["label"] == winner.get("label")), 0)
    return {
        "file_path": str(artifact.sample.file_path),
        "source_name": artifact.sample.source_name,
        "split": artifact.sample.split,
        "expected_label": artifact.sample.label,
        "expected_group": artifact.sample.group,
        "predicted_label": str(decision["final_label"]),
        "predicted_group": (
            "known"
            if decision["final_label"] not in {"unknown_cat", "not_cat", "low_quality"}
            else str(decision["final_label"])
        ),
        "correct": str(decision["final_label"]) == artifact.sample.label,
        "decision_type": decision["decision_type"],
        "decision_reason": decision["reason"],
        "winner_label": decision.get("winner_label"),
        "winner_votes": int(decision.get("winner_votes", 0)),
        "winner_top10_votes": int(winner_top10_votes),
        "top1_label": top_candidates[0]["label"] if top_candidates else None,
        "top1_group": top_candidates[0]["group"] if top_candidates else None,
        "top1_score": top_candidates[0]["score"] if top_candidates else None,
        "best_known_label": score_summary.get("best_known_label"),
        "best_known_score": float(score_summary.get("best_known_score") or 0.0),
        "best_unknown_score": float(score_summary.get("best_unknown_score") or 0.0),
        "best_not_cat_score": float(score_summary.get("best_not_cat_score") or 0.0),
        "query_cat_like": bool(score_summary.get("query_cat_like")),
        "face_detected_raw": bool(artifact.detection.face_detected),
        "face_detected_effective": bool(effective_detection.face_detected),
        "detector_backend": artifact.detection.detector_backend,
        "cat_face_score": artifact.face_check.get("cat_face_score"),
        "cat_face_non_face_score": artifact.face_check.get("not_cat_face_score"),
        "quality_pass": bool(artifact.quality.get("quality_pass")),
        "blur_score": float(artifact.quality.get("blur_score") or 0.0),
        "brightness": float(artifact.quality.get("brightness") or 0.0),
        "calibrator_prediction": calibrator_result.get("predicted_group"),
        "calibrator_confidence": calibrator_result.get("confidence"),
        "calibrator_margin": calibrator_result.get("margin"),
        "calibrator_confident": bool(calibrator_result.get("confident")),
        "decision_features": context["features"],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        payload = groups.setdefault(row["expected_group"], {"total": 0, "correct": 0})
        payload["total"] += 1
        payload["correct"] += int(row["correct"])
    group_metrics = {
        group: {
            "total": payload["total"],
            "correct": payload["correct"],
            "accuracy": round(payload["correct"] / payload["total"], 4) if payload["total"] else 0.0,
        }
        for group, payload in sorted(groups.items())
    }
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "group_metrics": group_metrics,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
