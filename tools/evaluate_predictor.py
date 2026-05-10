from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as predictor_app


@dataclass
class QuerySample:
    file_path: str
    expected_label: str
    expected_group: str
    source_name: str


@dataclass
class QueryArtifacts:
    sample: QuerySample
    quality: dict[str, Any]
    detection: predictor_app.DetectionResult
    face_check: dict[str, Any]
    query_vector: np.ndarray
    similarities: np.ndarray


@dataclass
class EvalConfig:
    name: str
    top_k: int
    vote_relative_margin: float
    use_face_model: bool
    cat_face_support_threshold: float
    cat_like_query_threshold: float = predictor_app.CAT_LIKE_QUERY_THRESHOLD
    special_override_margin: float = predictor_app.SPECIAL_OVERRIDE_MARGIN
    use_group_calibrator: bool = False


def build_samples(service: predictor_app.PredictorService) -> list[QuerySample]:
    samples: list[QuerySample] = []
    for cat in service.state.get("cats", []):
        for image in cat.get("images", []):
            samples.append(
                QuerySample(
                    file_path=image["file_path"],
                    expected_label=cat["name"],
                    expected_group="known",
                    source_name=image.get("source_name") or image["name"],
                )
            )
    for reference_key in predictor_app.REFERENCE_KEYS:
        ref = service.state["reference_sets"][reference_key]
        for image in ref.get("images", []):
            samples.append(
                QuerySample(
                    file_path=image["file_path"],
                    expected_label=reference_key,
                    expected_group=reference_key,
                    source_name=image.get("source_name") or image["name"],
                )
            )
    return samples


def build_artifacts(service: predictor_app.PredictorService, samples: list[QuerySample]) -> list[QueryArtifacts]:
    if service.index_vectors is None:
        raise RuntimeError("index_vectors is empty")

    artifacts: list[QueryArtifacts] = []
    for sample in samples:
        image_path = predictor_app.BASE_DIR / sample.file_path
        image_bytes = image_path.read_bytes()
        image_bgr = predictor_app.read_image_bytes(image_bytes)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {sample.file_path}")
        quality = predictor_app.compute_quality(image_bgr)
        detection = service.cropper.detect_and_crop(image_bgr)
        face_check = service.face_classifier.predict(detection.crop_bgr)
        query_vector = service.encoder.encode(detection.crop_bgr)
        similarities = np.dot(service.index_vectors, query_vector).astype("float32")
        artifacts.append(
            QueryArtifacts(
                sample=sample,
                quality=quality,
                detection=detection,
                face_check=face_check,
                query_vector=query_vector,
                similarities=similarities,
            )
        )
    return artifacts


def summarize_group_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        groups[row["expected_group"]]["total"] += 1
        groups[row["expected_group"]]["correct"] += int(row["correct"])
    payload = {}
    for group, stats in groups.items():
        total = stats["total"]
        payload[group] = {
            "total": total,
            "correct": stats["correct"],
            "accuracy": round(stats["correct"] / total, 4) if total else 0.0,
        }
    return dict(sorted(payload.items()))


def build_confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[row["expected_label"]][row["predicted_label"]] += 1
    return {
        expected: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
        for expected, counter in sorted(matrix.items())
    }


def face_supported(face_check: dict[str, Any], threshold: float, enabled: bool) -> bool:
    if not enabled:
        return False
    if not face_check.get("available"):
        return False
    score = face_check.get("cat_face_score")
    if score is None:
        return False
    return float(score) >= threshold


def classify_with_config(
    service: predictor_app.PredictorService,
    artifact: QueryArtifacts,
    config: EvalConfig,
    record_paths: np.ndarray,
) -> dict[str, Any]:
    face_check = artifact.face_check if config.use_face_model else {
        "available": False,
        "predicted_label": None,
        "predicted_index": None,
        "cat_face_score": None,
        "not_cat_face_score": None,
        "cat_face_supported": False,
    }
    query_cat_face_score = float(face_check.get("cat_face_score") or 0.0)
    query_cat_like = (
        config.use_face_model
        and bool(face_check.get("available"))
        and query_cat_face_score >= config.cat_like_query_threshold
    )
    top_candidates = predictor_app.build_top_candidates(
        service.index_records,
        artifact.similarities,
        top_k=config.top_k,
        query_cat_like=query_cat_like,
        exclude_file_path=artifact.sample.file_path,
    )
    score_summary = predictor_app.build_score_summary(
        service.index_records,
        artifact.similarities,
        face_check=face_check,
        exact_known_label=None,
        query_cat_like=query_cat_like,
        exclude_file_path=artifact.sample.file_path,
    )
    effective_detection = predictor_app.effective_detection_from_face_support(
        artifact.detection,
        face_check,
        threshold=config.cat_face_support_threshold,
    )

    previous_margin = predictor_app.VOTE_RELATIVE_MARGIN
    previous_special_margin = predictor_app.SPECIAL_OVERRIDE_MARGIN
    predictor_app.VOTE_RELATIVE_MARGIN = config.vote_relative_margin
    predictor_app.SPECIAL_OVERRIDE_MARGIN = config.special_override_margin
    try:
        decision = predictor_app.decide_prediction(top_candidates, artifact.quality, effective_detection, score_summary)
        calibrator_result = {"available": False}
        if config.use_group_calibrator:
            decision, calibrator_result = predictor_app.apply_open_set_calibrator(
                service.group_calibrator,
                current_decision=decision,
                top_candidates=top_candidates,
                quality=artifact.quality,
                detection=effective_detection,
                face_check=face_check,
                score_summary=score_summary,
            )
        vote_candidates = predictor_app.effective_vote_candidates(top_candidates)
    finally:
        predictor_app.VOTE_RELATIVE_MARGIN = previous_margin
        predictor_app.SPECIAL_OVERRIDE_MARGIN = previous_special_margin

    raw_class_summary = predictor_app.summarize_classes(top_candidates)
    effective_class_summary = predictor_app.summarize_classes(vote_candidates)
    winner = effective_class_summary[0] if effective_class_summary else {"votes": 0, "label": None, "weighted_sum": 0.0}
    winner_top10_votes = 0
    for row in raw_class_summary:
        if row["label"] == winner.get("label"):
            winner_top10_votes = int(row["votes"])
            break

    predicted_label = str(decision["final_label"])
    return {
        "file_path": artifact.sample.file_path,
        "expected_label": artifact.sample.expected_label,
        "expected_group": artifact.sample.expected_group,
        "predicted_label": predicted_label,
        "correct": predicted_label == artifact.sample.expected_label,
        "decision_type": decision["decision_type"],
        "decision_reason": decision["reason"],
        "winner_label": decision.get("winner_label"),
        "winner_votes": int(decision.get("winner_votes", 0)),
        "winner_top10_votes": int(winner_top10_votes),
        "calibrator_prediction": calibrator_result.get("predicted_group"),
        "calibrator_confident": bool(calibrator_result.get("confident")),
        "face_detected_raw": bool(artifact.detection.face_detected),
        "face_detected_effective": bool(effective_detection.face_detected),
        "cat_face_score": face_check.get("cat_face_score"),
        "top1_label": top_candidates[0]["label"] if top_candidates else None,
        "top1_score": top_candidates[0]["score"] if top_candidates else None,
    }


def build_error_analysis(result: dict[str, Any]) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    groups = sorted({row["expected_group"] for row in result["rows"]})
    for group in groups:
        subset = [row for row in result["rows"] if row["expected_group"] == group and not row["correct"]]
        decision_counter = Counter(row["decision_type"] for row in subset)
        prediction_counter = Counter(row["predicted_label"] for row in subset)
        samples = [
            {
                "file_path": row["file_path"],
                "expected": row["expected_label"],
                "predicted": row["predicted_label"],
                "decision_type": row["decision_type"],
                "top1_label": row["top1_label"],
                "winner_label": row["winner_label"],
                "face_detected_raw": row["face_detected_raw"],
                "cat_face_score": row["cat_face_score"],
            }
            for row in subset[:8]
        ]
        analysis[group] = {
            "error_count": len(subset),
            "decision_types": dict(decision_counter.most_common()),
            "predicted_labels": dict(prediction_counter.most_common()),
            "sample_errors": samples,
        }
    return analysis


def evaluate_config(
    service: predictor_app.PredictorService,
    artifacts: list[QueryArtifacts],
    config: EvalConfig,
) -> dict[str, Any]:
    record_paths = np.array([str(record["file_path"]) for record in service.index_records], dtype=object)
    rows = [classify_with_config(service, artifact, config, record_paths) for artifact in artifacts]
    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    return {
        "config": {
            "name": config.name,
            "top_k": config.top_k,
            "vote_relative_margin": config.vote_relative_margin,
            "use_face_model": config.use_face_model,
            "cat_face_support_threshold": config.cat_face_support_threshold,
            "cat_like_query_threshold": config.cat_like_query_threshold,
            "special_override_margin": config.special_override_margin,
            "use_group_calibrator": config.use_group_calibrator,
        },
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "group_metrics": summarize_group_metrics(rows),
        "confusion": build_confusion(rows),
        "rows": rows,
    }


def diff_results(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_map = {row["file_path"]: row for row in before["rows"]}
    improved = []
    regressed = []
    changed = 0
    for row in after["rows"]:
        previous = before_map[row["file_path"]]
        if previous["predicted_label"] != row["predicted_label"]:
            changed += 1
        if not previous["correct"] and row["correct"]:
            improved.append(
                {
                    "file_path": row["file_path"],
                    "expected": row["expected_label"],
                    "before": previous["predicted_label"],
                    "after": row["predicted_label"],
                    "before_face": previous["face_detected_effective"],
                    "after_face": row["face_detected_effective"],
                }
            )
        elif previous["correct"] and not row["correct"]:
            regressed.append(
                {
                    "file_path": row["file_path"],
                    "expected": row["expected_label"],
                    "before": previous["predicted_label"],
                    "after": row["predicted_label"],
                    "before_face": previous["face_detected_effective"],
                    "after_face": row["face_detected_effective"],
                }
            )
    return {
        "changed_predictions": changed,
        "improved": improved,
        "regressed": regressed,
    }


def row_brief(result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    return {
        "name": config["name"],
        "accuracy": result["accuracy"],
        "known": result["group_metrics"].get("known", {}).get("accuracy"),
        "unknown_cat": result["group_metrics"].get("unknown_cat", {}).get("accuracy"),
        "not_cat": result["group_metrics"].get("not_cat", {}).get("accuracy"),
    }


def build_raw_retrieval_metrics(
    service: predictor_app.PredictorService,
    artifacts: list[QueryArtifacts],
) -> dict[str, Any]:
    record_paths = np.array([str(record["file_path"]) for record in service.index_records], dtype=object)
    rows = []
    for artifact in artifacts:
        mask = record_paths != artifact.sample.file_path
        valid_indices = np.flatnonzero(mask)
        filtered_scores = artifact.similarities[mask]
        top_index = valid_indices[np.argsort(filtered_scores)[::-1][0]]
        record = service.index_records[int(top_index)]
        rows.append(
            {
                "expected_group": artifact.sample.expected_group,
                "expected_label": artifact.sample.expected_label,
                "top1_group": record["group"],
                "top1_label": record["label"],
            }
        )

    known_rows = [row for row in rows if row["expected_group"] == "known"]
    unknown_rows = [row for row in rows if row["expected_group"] == "unknown_cat"]
    not_cat_rows = [row for row in rows if row["expected_group"] == "not_cat"]
    return {
        "known_top1_label_accuracy": round(sum(row["top1_label"] == row["expected_label"] for row in known_rows) / len(known_rows), 4),
        "unknown_cat_top1_group_accuracy": round(sum(row["top1_group"] == row["expected_group"] for row in unknown_rows) / len(unknown_rows), 4),
        "not_cat_top1_group_accuracy": round(sum(row["top1_group"] == row["expected_group"] for row in not_cat_rows) / len(not_cat_rows), 4),
    }


def render_dataset_section(report: dict[str, Any]) -> str:
    lines = [
        "## Dataset",
        "",
        "| Split | Images |",
        "| --- | ---: |",
        f"| Known | {report['dataset']['known']} |",
        f"| Unknown cat | {report['dataset']['unknown_cat']} |",
        f"| Not cat | {report['dataset']['not_cat']} |",
        f"| Total | {report['dataset']['total']} |",
        "",
        "| Known label | Images |",
        "| --- | ---: |",
    ]
    for label, count in report["dataset"]["known_by_label"].items():
        lines.append(f"| {label} | {count} |")
    return "\n".join(lines)


def render_current_thresholds_section(report: dict[str, Any]) -> str:
    lines = [
        "## Current Thresholds",
        "",
        "| Key | Value |",
        "| --- | ---: |",
    ]
    for key, value in report["current_thresholds"].items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def render_accuracy_summary_section(report: dict[str, Any]) -> str:
    lines = [
        "## Accuracy Summary",
        "",
        "| Config | Overall | Known | Unknown cat | Not cat |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    row = report["baseline"]["summary"]
    lines.append(f"| {row['name']} | {row['accuracy']:.4f} | {row['known']:.4f} | {row['unknown_cat']:.4f} | {row['not_cat']:.4f} |")
    return "\n".join(lines)


def render_top_k_section(report: dict[str, Any]) -> str:
    lines = [
        "## Top K Sweep",
        "",
        "| Config | Overall | Known | Unknown cat | Not cat |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["top_k_sweep"]:
        lines.append(f"| {row['name']} | {row['accuracy']:.4f} | {row['known']:.4f} | {row['unknown_cat']:.4f} | {row['not_cat']:.4f} |")
    return "\n".join(lines)


def render_threshold_sweeps_section(report: dict[str, Any]) -> str:
    lines = [
        "## Vote Margin Sweep",
        "",
        "| Config | Overall | Known | Unknown cat | Not cat |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["vote_margin_sweep"]:
        lines.append(f"| {row['name']} | {row['accuracy']:.4f} | {row['known']:.4f} | {row['unknown_cat']:.4f} | {row['not_cat']:.4f} |")
    lines.extend([
        "",
        "## Face Support Threshold Sweep",
        "",
        "| Config | Overall | Known | Unknown cat | Not cat |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in report["face_support_threshold_sweep"]:
        lines.append(f"| {row['name']} | {row['accuracy']:.4f} | {row['known']:.4f} | {row['unknown_cat']:.4f} | {row['not_cat']:.4f} |")
    return "\n".join(lines)


def render_raw_retrieval_section(report: dict[str, Any]) -> str:
    return "\n".join([
        "## Raw Retrieval",
        "",
        "| Metric | Accuracy |",
        "| --- | ---: |",
        f"| Known top-1 label | {report['raw_retrieval']['known_top1_label_accuracy']:.4f} |",
        f"| Unknown cat top-1 group | {report['raw_retrieval']['unknown_cat_top1_group_accuracy']:.4f} |",
        f"| Not cat top-1 group | {report['raw_retrieval']['not_cat_top1_group_accuracy']:.4f} |",
    ])


def render_before_after_section(report: dict[str, Any]) -> str:
    before = report["before_vs_after_face_model"]["before"]
    after = report["before_vs_after_face_model"]["after"]
    diff = report["before_vs_after_face_model"]["diff"]
    lines = [
        "## Before vs After Face Model",
        "",
        "| Config | Overall | Known | Unknown cat | Not cat |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| {before['name']} | {before['accuracy']:.4f} | {before['known']:.4f} | {before['unknown_cat']:.4f} | {before['not_cat']:.4f} |",
        f"| {after['name']} | {after['accuracy']:.4f} | {after['known']:.4f} | {after['unknown_cat']:.4f} | {after['not_cat']:.4f} |",
        "",
        "| Diff | Count |",
        "| --- | ---: |",
        f"| Changed predictions | {diff['changed_predictions']} |",
        f"| Improved | {len(diff['improved'])} |",
        f"| Regressed | {len(diff['regressed'])} |",
    ]
    return "\n".join(lines)


def render_error_analysis_section(report: dict[str, Any]) -> str:
    lines = [
        "## Error Analysis",
        "",
    ]
    for group, payload in report["baseline"]["error_analysis"].items():
        lines.extend([
            f"### {group}",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Error count | `{payload['error_count']}` |",
            f"| Decision types | `{payload['decision_types']}` |",
            f"| Predicted labels | `{payload['predicted_labels']}` |",
        ])
        if payload["sample_errors"]:
            lines.extend([
                "",
                "| File | Predicted | Decision | Top1 | Face score |",
                "| --- | --- | --- | --- | ---: |",
            ])
            for sample in payload["sample_errors"]:
                face_score = sample["cat_face_score"]
                face_text = "-" if face_score is None else f"{float(face_score):.6f}"
                lines.append(
                    f"| `{sample['file_path']}` | `{sample['predicted']}` | `{sample['decision_type']}` | `{sample['top1_label']}` | {face_text} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_markdown_report(report: dict[str, Any], section: str = "all") -> str:
    section_builders = {
        "dataset": [render_dataset_section],
        "current_thresholds": [render_current_thresholds_section],
        "summary": [render_accuracy_summary_section, render_raw_retrieval_section],
        "topk": [render_top_k_section],
        "thresholds": [render_threshold_sweeps_section],
        "before_after": [render_before_after_section],
        "errors": [render_error_analysis_section],
        "all": [
            render_dataset_section,
            render_current_thresholds_section,
            render_accuracy_summary_section,
            render_top_k_section,
            render_threshold_sweeps_section,
            render_raw_retrieval_section,
            render_before_after_section,
            render_error_analysis_section,
        ],
    }
    builders = section_builders[section]
    lines = ["# Predictor Evaluation Report", ""]
    for builder in builders:
        lines.append(builder(report))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictor on local dataset")
    parser.add_argument("--json", action="store_true", help="print raw JSON only")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="stdout output format")
    parser.add_argument("--section", choices=("all", "dataset", "current_thresholds", "summary", "topk", "thresholds", "before_after", "errors"), default="all", help="markdown section to print")
    parser.add_argument("--report-path", default=str(predictor_app.DATA_DIR / "evaluation_report.md"), help="write markdown report to this path")
    args = parser.parse_args()

    started = time.perf_counter()
    service = predictor_app.PredictorService()
    samples = build_samples(service)
    artifacts = build_artifacts(service, samples)

    current_face_threshold = predictor_app.CAT_FACE_SUPPORT_THRESHOLD
    current_vote_margin = predictor_app.VOTE_RELATIVE_MARGIN
    current_top_k = predictor_app.TOP_K

    baseline = EvalConfig(
        name="current",
        top_k=current_top_k,
        vote_relative_margin=current_vote_margin,
        use_face_model=True,
        cat_face_support_threshold=current_face_threshold,
    )
    before_face_model = EvalConfig(
        name="before_face_model",
        top_k=current_top_k,
        vote_relative_margin=current_vote_margin,
        use_face_model=False,
        cat_face_support_threshold=current_face_threshold,
    )

    top_k_results = [
        evaluate_config(service, artifacts, EvalConfig(name=f"top_{top_k}", top_k=top_k, vote_relative_margin=current_vote_margin, use_face_model=True, cat_face_support_threshold=current_face_threshold))
        for top_k in (10, 5, 3)
    ]
    vote_margin_results = [
        evaluate_config(service, artifacts, EvalConfig(name=f"margin_{margin:.2f}", top_k=current_top_k, vote_relative_margin=margin, use_face_model=True, cat_face_support_threshold=current_face_threshold))
        for margin in (0.08, 0.10, 0.12, 0.14, 0.16)
    ]
    face_threshold_results = [
        evaluate_config(service, artifacts, EvalConfig(name=f"face_thr_{threshold:.2f}", top_k=current_top_k, vote_relative_margin=current_vote_margin, use_face_model=True, cat_face_support_threshold=threshold))
        for threshold in (0.85, 0.90, 0.95, 0.97, 0.99)
    ]
    baseline_result = evaluate_config(service, artifacts, baseline)
    before_result = evaluate_config(service, artifacts, before_face_model)
    face_model_diff = diff_results(before_result, baseline_result)
    raw_retrieval_metrics = build_raw_retrieval_metrics(service, artifacts)

    report = {
        "dataset": {
            "total": len(samples),
            "known": sum(1 for sample in samples if sample.expected_group == "known"),
            "unknown_cat": sum(1 for sample in samples if sample.expected_group == "unknown_cat"),
            "not_cat": sum(1 for sample in samples if sample.expected_group == "not_cat"),
            "known_by_label": {
                cat["name"]: len(cat.get("images", []))
                for cat in service.state.get("cats", [])
            },
        },
        "current_thresholds": {
            "top_k": current_top_k,
            "vote_relative_margin": current_vote_margin,
            "cat_face_support_threshold": current_face_threshold,
            "cat_like_query_threshold": predictor_app.CAT_LIKE_QUERY_THRESHOLD,
            "strong_non_cat_face_score": predictor_app.STRONG_NON_CAT_FACE_SCORE,
            "not_cat_no_face_min_score": predictor_app.NOT_CAT_NO_FACE_MIN_SCORE,
            "known_high_confidence_score": predictor_app.KNOWN_HIGH_CONFIDENCE_SCORE,
            "known_no_face_score": predictor_app.KNOWN_NO_FACE_SCORE,
            "known_no_face_avg_score": predictor_app.KNOWN_NO_FACE_AVG_SCORE,
            "known_no_face_special_margin": predictor_app.KNOWN_NO_FACE_SPECIAL_MARGIN,
            "known_dominant_votes": predictor_app.KNOWN_DOMINANT_VOTES,
            "known_dominant_vote_margin": predictor_app.KNOWN_DOMINANT_VOTE_MARGIN,
            "known_dominant_avg_score": predictor_app.KNOWN_DOMINANT_AVG_SCORE,
            "known_dominant_best_score": predictor_app.KNOWN_DOMINANT_BEST_SCORE,
            "known_dominant_special_margin": predictor_app.KNOWN_DOMINANT_SPECIAL_MARGIN,
            "known_dominant_max_special_score": predictor_app.KNOWN_DOMINANT_MAX_SPECIAL_SCORE,
            "special_override_margin": predictor_app.SPECIAL_OVERRIDE_MARGIN,
        },
        "baseline": {
            "summary": row_brief(baseline_result),
            "confusion": baseline_result["confusion"],
            "error_analysis": build_error_analysis(baseline_result),
        },
        "raw_retrieval": raw_retrieval_metrics,
        "top_k_sweep": [row_brief(result) for result in top_k_results],
        "vote_margin_sweep": [row_brief(result) for result in vote_margin_results],
        "face_support_threshold_sweep": [row_brief(result) for result in face_threshold_results],
        "before_vs_after_face_model": {
            "before": row_brief(before_result),
            "after": row_brief(baseline_result),
            "diff": face_model_diff,
        },
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown_report(report, section="all"), encoding="utf-8")

    if args.json or args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(render_markdown_report(report, section=args.section), end="")


if __name__ == "__main__":
    main()
