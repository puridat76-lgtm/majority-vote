from __future__ import annotations

import argparse
import json
import re
import sys
import time
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as predictor_app
from tools import evaluate_predictor as live_eval


def macro_accuracy(result: dict[str, Any]) -> float:
    metrics = result.get("group_metrics", {})
    values = [float(payload.get("accuracy", 0.0)) for payload in metrics.values()]
    return float(mean(values)) if values else 0.0


def score_key(result: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = result.get("group_metrics", {})
    return (
        float(result.get("accuracy", 0.0)),
        macro_accuracy(result),
        float(metrics.get("known", {}).get("accuracy", 0.0)),
        float(metrics.get("unknown_cat", {}).get("accuracy", 0.0)),
        float(metrics.get("not_cat", {}).get("accuracy", 0.0)),
    )


def config_row(result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    metrics = result["group_metrics"]
    return {
        "name": config["name"],
        "detector_mode": config.get("detector_mode"),
        "top_k": config["top_k"],
        "vote_relative_margin": config["vote_relative_margin"],
        "cat_face_support_threshold": config["cat_face_support_threshold"],
        "cat_like_query_threshold": config.get("cat_like_query_threshold"),
        "special_override_margin": config.get("special_override_margin"),
        "use_group_calibrator": config.get("use_group_calibrator", False),
        "overall": round(float(result["accuracy"]), 4),
        "macro": round(macro_accuracy(result), 4),
        "known": round(float(metrics.get("known", {}).get("accuracy", 0.0)), 4),
        "unknown_cat": round(float(metrics.get("unknown_cat", {}).get("accuracy", 0.0)), 4),
        "not_cat": round(float(metrics.get("not_cat", {}).get("accuracy", 0.0)), 4),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# YOLO Experiment Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Live dataset total: `{report['live_dataset']['total']}`",
        "",
        "## Live Comparison",
        "",
        "| Config | Overall | Macro | Known | Unknown cat | Not cat | Detector | Calibrator | Top K | Vote margin | Face thr | Cat-like thr | Special margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["live_comparison_rows"]:
        lines.append(
            f"| {row['name']} | {row['overall']:.4f} | {row['macro']:.4f} | {row['known']:.4f} | {row['unknown_cat']:.4f} | {row['not_cat']:.4f} | "
            f"{row['detector_mode']} | {row['use_group_calibrator']} | {row['top_k']} | {row['vote_relative_margin']:.2f} | "
            f"{row['cat_face_support_threshold']:.2f} | {row['cat_like_query_threshold']:.2f} | {row['special_override_margin']:.2f} |"
        )
    lines.extend([
        "",
        "## YOLO Grid Top 10",
        "",
        "| Rank | Overall | Macro | Known | Unknown cat | Not cat | Calibrator | Top K | Vote margin | Face thr | Cat-like thr | Special margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for index, row in enumerate(report["live_yolo_top10"], start=1):
        lines.append(
            f"| {index} | {row['overall']:.4f} | {row['macro']:.4f} | {row['known']:.4f} | {row['unknown_cat']:.4f} | {row['not_cat']:.4f} | "
            f"{row['use_group_calibrator']} | {row['top_k']} | {row['vote_relative_margin']:.2f} | {row['cat_face_support_threshold']:.2f} | "
            f"{row['cat_like_query_threshold']:.2f} | {row['special_override_margin']:.2f} |"
        )
    lines.extend([
        "",
        "## Prepared Dataset Comparison",
        "",
        "| Config | Overall | Known | Unknown cat | Not cat | Detector |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in report["prepared_rows"]:
        lines.append(
            f"| {row['name']} | {row['overall']:.4f} | {row['known']:.4f} | {row['unknown_cat']:.4f} | {row['not_cat']:.4f} | {row['detector_mode']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
    ])
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def build_live_service(detector_mode: str) -> predictor_app.PredictorService:
    experiments_dir = predictor_app.DATA_DIR / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"live_{detector_mode}_index_cache.npz"
    return predictor_app.PredictorService(
        detector_mode=detector_mode,
        index_cache_path=experiments_dir / cache_name,
    )


def parse_json_tail(raw: str) -> dict[str, Any]:
    match = re.search(r"(\{[\s\S]*\})\s*$", raw)
    if not match:
        raise ValueError("Could not parse JSON tail from command output")
    return json.loads(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a serious YOLO-vs-Haar experiment on the live app dataset")
    parser.add_argument("--output", default=str(ROOT_DIR / "data" / "yolo_experiment_report.md"))
    parser.add_argument("--json-output", default=str(ROOT_DIR / "data" / "yolo_experiment_report.json"))
    parser.add_argument("--prepared-dataset-root", default=str(ROOT_DIR / "data" / "prepared" / "cat_retrain_dataset"))
    parser.add_argument("--prepared-haar-calibrator", default=str(ROOT_DIR / "data" / "models" / "open_set_calibrator.json"))
    parser.add_argument("--prepared-yolo-current-calibrator", default=str(ROOT_DIR / "data" / "models" / "open_set_calibrator_yolo_current.json"))
    parser.add_argument("--prepared-yolo-model", default=str(ROOT_DIR / "data" / "models" / "siamese_encoder_retrained_yolo.h5"))
    parser.add_argument("--prepared-yolo-calibrator", default=str(ROOT_DIR / "data" / "models" / "open_set_calibrator_retrained_yolo.json"))
    args = parser.parse_args()

    started = time.perf_counter()

    live_haar_service = build_live_service("haar")
    live_yolo_service = build_live_service("yolo")
    live_haar_samples = live_eval.build_samples(live_haar_service)
    live_haar_artifacts = live_eval.build_artifacts(live_haar_service, live_haar_samples)
    live_yolo_artifacts = live_eval.build_artifacts(live_yolo_service, live_haar_samples)

    baseline_current = live_eval.evaluate_config(
        live_haar_service,
        live_haar_artifacts,
        live_eval.EvalConfig(
            name="live_haar_current",
            top_k=predictor_app.TOP_K,
            vote_relative_margin=predictor_app.VOTE_RELATIVE_MARGIN,
            use_face_model=True,
            cat_face_support_threshold=predictor_app.CAT_FACE_SUPPORT_THRESHOLD,
            cat_like_query_threshold=predictor_app.CAT_LIKE_QUERY_THRESHOLD,
            special_override_margin=predictor_app.SPECIAL_OVERRIDE_MARGIN,
            use_group_calibrator=True,
        ),
    )
    yolo_current = live_eval.evaluate_config(
        live_yolo_service,
        live_yolo_artifacts,
        live_eval.EvalConfig(
            name="live_yolo_current",
            top_k=predictor_app.TOP_K,
            vote_relative_margin=predictor_app.VOTE_RELATIVE_MARGIN,
            use_face_model=True,
            cat_face_support_threshold=predictor_app.CAT_FACE_SUPPORT_THRESHOLD,
            cat_like_query_threshold=predictor_app.CAT_LIKE_QUERY_THRESHOLD,
            special_override_margin=predictor_app.SPECIAL_OVERRIDE_MARGIN,
            use_group_calibrator=True,
        ),
    )

    grid_results: list[dict[str, Any]] = []
    for top_k, vote_margin, face_thr, cat_like_thr, special_margin, use_calibrator in product(
        (10, 5),
        (0.10, 0.12, 0.16),
        (0.90, 0.95),
        (0.95, 1.01),
        (0.00, 0.02),
        (False, True),
    ):
        config = live_eval.EvalConfig(
            name=f"yolo_k{top_k}_m{vote_margin:.2f}_f{face_thr:.2f}_c{cat_like_thr:.2f}_s{special_margin:.2f}_{'cal' if use_calibrator else 'raw'}",
            top_k=top_k,
            vote_relative_margin=vote_margin,
            use_face_model=True,
            cat_face_support_threshold=face_thr,
            cat_like_query_threshold=cat_like_thr,
            special_override_margin=special_margin,
            use_group_calibrator=use_calibrator,
        )
        result = live_eval.evaluate_config(live_yolo_service, live_yolo_artifacts, config)
        result["config"]["detector_mode"] = "yolo"
        grid_results.append(result)

    baseline_current["config"]["detector_mode"] = "haar"
    yolo_current["config"]["detector_mode"] = "yolo"

    best_yolo = max(grid_results, key=score_key)
    sorted_yolo = sorted(grid_results, key=score_key, reverse=True)

    prepared_rows: list[dict[str, Any]] = []
    prepared_dataset_root = Path(args.prepared_dataset_root)
    prepared_evals = [
        {
            "name": "prepared_haar_current",
            "cmd": [
                sys.executable,
                "tools/evaluate_retrain_dataset.py",
                "--dataset-root",
                str(prepared_dataset_root),
                "--detector-mode",
                "haar",
                "--calibrator-path",
                str(Path(args.prepared_haar_calibrator)),
                "--format",
                "json",
                "--summary-only",
            ],
        },
        {
            "name": "prepared_yolo_current",
            "cmd": [
                sys.executable,
                "tools/evaluate_retrain_dataset.py",
                "--dataset-root",
                str(prepared_dataset_root),
                "--detector-mode",
                "yolo",
                "--calibrator-path",
                str(Path(args.prepared_yolo_current_calibrator)),
                "--format",
                "json",
                "--summary-only",
            ],
        },
    ]

    import subprocess

    for spec in prepared_evals:
        raw = subprocess.check_output(spec["cmd"], cwd=ROOT_DIR, text=True)
        payload = parse_json_tail(raw)
        prepared_rows.append(
            {
                "name": spec["name"],
                "detector_mode": payload["detector_mode"],
                "overall": payload["calibrated"]["accuracy"],
                "known": payload["calibrated"]["group_metrics"]["known"]["accuracy"],
                "unknown_cat": payload["calibrated"]["group_metrics"]["unknown_cat"]["accuracy"],
                "not_cat": payload["calibrated"]["group_metrics"]["not_cat"]["accuracy"],
            }
        )

    yolo_model_path = Path(args.prepared_yolo_model)
    yolo_calibrator_path = Path(args.prepared_yolo_calibrator)
    if yolo_model_path.exists() and yolo_calibrator_path.exists():
        raw = subprocess.check_output(
            [
                sys.executable,
                "tools/evaluate_retrain_dataset.py",
                "--dataset-root",
                str(prepared_dataset_root),
                "--detector-mode",
                "yolo",
                "--model-path",
                str(yolo_model_path),
                "--calibrator-path",
                str(yolo_calibrator_path),
                "--format",
                "json",
                "--summary-only",
            ],
            cwd=ROOT_DIR,
            text=True,
        )
        payload = parse_json_tail(raw)
        prepared_rows.append(
            {
                "name": "prepared_yolo_retrained",
                "detector_mode": payload["detector_mode"],
                "overall": payload["calibrated"]["accuracy"],
                "known": payload["calibrated"]["group_metrics"]["known"]["accuracy"],
                "unknown_cat": payload["calibrated"]["group_metrics"]["unknown_cat"]["accuracy"],
                "not_cat": payload["calibrated"]["group_metrics"]["not_cat"]["accuracy"],
            }
        )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "live_dataset": {
            "total": len(live_haar_samples),
            "known": sum(1 for sample in live_haar_samples if sample.expected_group == "known"),
            "unknown_cat": sum(1 for sample in live_haar_samples if sample.expected_group == "unknown_cat"),
            "not_cat": sum(1 for sample in live_haar_samples if sample.expected_group == "not_cat"),
        },
        "live_comparison_rows": [
            config_row(baseline_current),
            config_row(yolo_current),
            config_row(best_yolo),
        ],
        "live_yolo_top10": [config_row(row) for row in sorted_yolo[:10]],
        "prepared_rows": prepared_rows,
        "notes": [
            "live_haar_current mirrors the production app closest: Haar detector + current thresholds + current open-set calibrator.",
            "live_yolo_current swaps only the detector to YOLO and keeps the rest of the production behavior the same.",
            "best_yolo is selected from a grid over Top K, vote margin, face support threshold, cat-like threshold, special override margin, and calibrator on/off.",
            "prepared dataset rows measure the retrain dataset pipeline and should not be mixed with the live uploads accuracy number.",
        ],
    }

    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")

    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
