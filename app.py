from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import argparse
from functools import wraps

import cv2
import numpy as np
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
IS_SERVERLESS = os.environ.get("VERCEL") == "1"
SERVERLESS_RUNTIME_KEY = os.environ.get("VERCEL_GIT_COMMIT_SHA") or os.environ.get("VERCEL_DEPLOYMENT_ID") or "local"
RUNTIME_DIR = Path(os.environ.get("CAT_APP_RUNTIME_DIR", f"/tmp/cat-majority-vote-{SERVERLESS_RUNTIME_KEY}")) if IS_SERVERLESS else BASE_DIR
UPLOADS_DIR = BASE_DIR / "uploads"
CATS_DIR = UPLOADS_DIR / "cats"
REFERENCE_DIR = UPLOADS_DIR / "reference"
QUERY_DIR = (RUNTIME_DIR / "uploads" / "query") if IS_SERVERLESS else (UPLOADS_DIR / "query")
PACKAGE_DATA_DIR = BASE_DIR / "data"
PACKAGE_STORE_PATH = PACKAGE_DATA_DIR / "store.json"
DATA_DIR = (RUNTIME_DIR / "data") if IS_SERVERLESS else PACKAGE_DATA_DIR
STORE_PATH = DATA_DIR / "store.json"
INDEX_CACHE_PATH = DATA_DIR / "index_cache.npz"
MPLCONFIG_DIR = DATA_DIR / "matplotlib"
MODEL_PATH = BASE_DIR / "siamese_model3.h5"
OPEN_SET_CALIBRATOR_PATH = BASE_DIR / "data" / "models" / "open_set_calibrator.json"
YOLO_OPEN_SET_CALIBRATOR_PATH = BASE_DIR / "data" / "models" / "open_set_calibrator_yolo_keras_hardneg.json"
CAT_FACE_MODEL_CANDIDATES = (
    BASE_DIR / "data" / "keras_model.h5",
    BASE_DIR / "data" / "cat_face_classifier.h5",
    Path.home() / "Downloads" / "converted_keras" / "keras_model.h5",
)
CAT_FACE_LABELS_CANDIDATES = (
    BASE_DIR / "data" / "labels.txt",
    BASE_DIR / "data" / "cat_face_labels.txt",
    Path.home() / "Downloads" / "converted_keras" / "labels.txt",
)
CAT_FACE_DETECTOR_MODEL_CANDIDATES = (
    BASE_DIR / "data" / "cat_face_detector.pt",
    BASE_DIR / "data" / "imports" / "cat_face_detector_zip" / "cat_face_detector" / "cat_face_detector.pt",
)
ASSET_VERSION = "20260428-1406"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REFERENCE_KEYS = ("not_cat", "unknown_cat")
ADMIN_PASSWORD = "admin"
IMAGE_SIZE = 224
TOP_K = 10
ENCODE_BATCH_SIZE = 32
INDEX_CACHE_VERSION = 6
CLASSIC_FEATURE_WEIGHT = 0.15
FACE_DETECT_MAX_SIDE = 960
YOLO_DETECT_MAX_SIDE = 1280
YOLO_DETECT_IMGSZ = 640
YOLO_DETECT_CONF = 0.25
YOLO_FACE_VALIDATION_MIN_SCORE = 0.50
ENABLE_YOLO_DETECTOR = True
JPEG_REDUCED_2_SIZE = 3 * 1024 * 1024
JPEG_REDUCED_4_SIZE = 8 * 1024 * 1024
JPEG_REDUCED_8_SIZE = 16 * 1024 * 1024
QUERY_IMAGE_MAX_SIDE = 2048
QUALITY_MAX_SIDE = 1024
CAT_FACE_MODEL_INPUT_SIZE = 224
CAT_FACE_SUPPORT_THRESHOLD = 0.95
CAT_LIKE_QUERY_THRESHOLD = 1.01
STRONG_NON_CAT_FACE_SCORE = 0.01
NOT_CAT_NO_FACE_MIN_SCORE = 0.90
KNOWN_HIGH_CONFIDENCE_SCORE = 0.95
KNOWN_NO_FACE_SCORE = 0.82
KNOWN_NO_FACE_AVG_SCORE = 0.78
KNOWN_NO_FACE_SPECIAL_MARGIN = 0.08
KNOWN_DOMINANT_VOTES = 8
KNOWN_DOMINANT_VOTE_MARGIN = 5
KNOWN_DOMINANT_AVG_SCORE = 0.70
KNOWN_DOMINANT_BEST_SCORE = 0.72
KNOWN_DOMINANT_SPECIAL_MARGIN = 0.10
KNOWN_DOMINANT_MAX_SPECIAL_SCORE = 0.88
SPECIAL_OVERRIDE_MARGIN = 0.00
VOTE_RELATIVE_MARGIN = 0.12
OPEN_SET_CALIBRATOR_MIN_PROB = 0.55
OPEN_SET_CALIBRATOR_MIN_MARGIN = 0.05

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class EncoderBackend:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.backend_name = "fallback_classic"
        self._model = None
        self._load_error = None
        self._try_load_tensorflow_model()

    def _try_load_tensorflow_model(self) -> None:
        if not self.model_path.exists():
            self._load_error = f"missing_model:{self.model_path.name}"
            return

        try:
            import tensorflow as tf  # type: ignore
            from tensorflow.keras.models import load_model  # type: ignore

            class L2Norm(tf.keras.layers.Layer):
                def call(self, inputs):
                    return tf.math.l2_normalize(inputs, axis=-1)

            self._model = load_model(self.model_path, compile=False, custom_objects={"L2Norm": L2Norm})
            self.backend_name = "tensorflow_h5"
            self._load_error = None
        except Exception as exc:  # pragma: no cover - environment dependent
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._model = None
            self.backend_name = "fallback_classic"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = self._prepare_rgb(image_bgr)
        classic_vector = self._fallback_encode(resized)
        if self._model is not None:
            batch = np.expand_dims(resized / 255.0, axis=0)
            deep_vector = np.asarray(self._model.predict(batch, verbose=0)[0], dtype="float32")
            return self._combine_feature_vectors(deep_vector, classic_vector)
        return classic_vector

    def encode_many(self, images_bgr: list[np.ndarray]) -> list[np.ndarray]:
        resized_images = [self._prepare_rgb(image_bgr) for image_bgr in images_bgr]
        if not resized_images:
            return []
        classic_vectors = [self._fallback_encode(resized) for resized in resized_images]
        if self._model is None:
            return classic_vectors

        vectors: list[np.ndarray] = []
        for start in range(0, len(resized_images), ENCODE_BATCH_SIZE):
            batch_slice = resized_images[start:start + ENCODE_BATCH_SIZE]
            batch = np.stack(batch_slice).astype("float32") / 255.0
            predictions = np.asarray(self._model.predict(batch, verbose=0), dtype="float32")
            batch_classic_vectors = classic_vectors[start:start + ENCODE_BATCH_SIZE]
            for prediction, classic_vector in zip(predictions, batch_classic_vectors, strict=False):
                vectors.append(self._combine_feature_vectors(prediction, classic_vector))
            print(f"Encoded {min(start + len(batch_slice), len(resized_images))}/{len(resized_images)} crops...", flush=True)
        return vectors

    def _prepare_rgb(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return fit_image_to_canvas(rgb, IMAGE_SIZE, IMAGE_SIZE).astype("float32")

    def _fallback_encode(self, rgb_uint8: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb_uint8.astype("uint8"), cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype("float32").reshape(-1) / 255.0
        hist_parts = []
        for channel in cv2.split(rgb_uint8.astype("uint8")):
            hist = cv2.calcHist([channel], [0], None, [16], [0, 256]).astype("float32").reshape(-1)
            hist /= max(float(hist.sum()), 1.0)
            hist_parts.append(hist)
        edges = cv2.Canny(gray, 80, 160)
        edge_small = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).astype("float32").reshape(-1) / 255.0
        feature = np.concatenate([small, *hist_parts, edge_small]).astype("float32")
        return l2_normalize(feature)

    def _combine_feature_vectors(self, deep_vector: np.ndarray, classic_vector: np.ndarray) -> np.ndarray:
        deep_vector = l2_normalize(deep_vector)
        classic_vector = l2_normalize(classic_vector) * CLASSIC_FEATURE_WEIGHT
        combined = np.concatenate([deep_vector, classic_vector]).astype("float32")
        return l2_normalize(combined)


class CatFaceBinaryClassifier:
    def __init__(self, model_candidates: tuple[Path, ...], labels_candidates: tuple[Path, ...]):
        self.model_path = first_existing_path(model_candidates)
        self.labels_path = first_existing_path(labels_candidates)
        self.backend_name = "unavailable"
        self._model = None
        self._labels: list[str] = []
        self._cat_face_index: int | None = None
        self._load_error: str | None = None
        self._try_load_model()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def available(self) -> bool:
        return self._model is not None and self._cat_face_index is not None

    def _try_load_model(self) -> None:
        if self.model_path is None:
            self._load_error = "missing_model"
            return
        self._labels = self._load_labels()
        self._cat_face_index = self._infer_cat_face_index(self._labels)
        try:
            import tensorflow as tf  # type: ignore

            self._model = self._build_compatible_model(tf)
            self._model.load_weights(self.model_path)
            self.backend_name = "tm_cat_face_compat"
            self._load_error = None
        except Exception as exc:  # pragma: no cover - environment dependent
            self._model = None
            self.backend_name = "unavailable"
            self._load_error = f"{type(exc).__name__}: {exc}"

    def _load_labels(self) -> list[str]:
        if self.labels_path is None or not self.labels_path.exists():
            return ["รูปที่ไม่ใช่หน้าแมว", "หน้าแมว"]
        labels = []
        for raw in self.labels_path.read_text(encoding="utf-8").splitlines():
            text = raw.strip()
            if not text:
                continue
            parts = text.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(text)
        return labels or ["รูปที่ไม่ใช่หน้าแมว", "หน้าแมว"]

    def _infer_cat_face_index(self, labels: list[str]) -> int:
        for index, label in enumerate(labels):
            normalized = label.replace("_", " ").strip().lower()
            if ("หน้าแมว" in label and "ไม่ใช่" not in label) or ("cat face" in normalized and "not" not in normalized) or normalized == "cat":
                return index
        return 1 if len(labels) > 1 else 0

    def _build_compatible_model(self, tf: Any):
        base_core = tf.keras.applications.MobileNetV2(
            input_shape=(CAT_FACE_MODEL_INPUT_SIZE, CAT_FACE_MODEL_INPUT_SIZE, 3),
            alpha=0.35,
            include_top=False,
            weights=None,
            pooling=None,
        )
        base_core._name = "model2"
        base = tf.keras.Sequential([
            tf.keras.Input(shape=(CAT_FACE_MODEL_INPUT_SIZE, CAT_FACE_MODEL_INPUT_SIZE, 3), name="model2_input"),
            base_core,
            tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d_GlobalAveragePooling2D2"),
        ], name="sequential_5")
        head = tf.keras.Sequential([
            tf.keras.Input(shape=(1280,), name="dense_Dense3_input"),
            tf.keras.layers.Dense(100, activation="relu", name="dense_Dense3"),
            tf.keras.layers.Dense(2, activation="softmax", use_bias=False, name="dense_Dense4"),
        ], name="sequential_7")
        return tf.keras.Sequential([
            tf.keras.Input(shape=(CAT_FACE_MODEL_INPUT_SIZE, CAT_FACE_MODEL_INPUT_SIZE, 3), name="sequential_5_input"),
            base,
            head,
        ], name="sequential_8")

    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        if not self.available or image_bgr is None or image_bgr.size == 0:
            return {
                "available": False,
                "predicted_label": None,
                "predicted_index": None,
                "cat_face_score": None,
                "not_cat_face_score": None,
                "cat_face_supported": False,
            }
        fitted_bgr = fit_image_to_canvas(image_bgr, CAT_FACE_MODEL_INPUT_SIZE, CAT_FACE_MODEL_INPUT_SIZE)
        rgb = cv2.cvtColor(fitted_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        scores = np.asarray(self._model.predict(np.expand_dims(rgb, axis=0), verbose=0)[0], dtype="float32")
        predicted_index = int(np.argmax(scores))
        cat_face_index = int(self._cat_face_index or 0)
        cat_face_score = float(scores[cat_face_index])
        not_cat_indices = [index for index in range(len(scores)) if index != cat_face_index]
        not_cat_face_score = float(max((scores[index] for index in not_cat_indices), default=0.0))
        predicted_label = self._labels[predicted_index] if predicted_index < len(self._labels) else str(predicted_index)
        return {
            "available": True,
            "predicted_label": predicted_label,
            "predicted_index": predicted_index,
            "cat_face_score": round(cat_face_score, 6),
            "not_cat_face_score": round(not_cat_face_score, 6),
            "cat_face_supported": cat_face_score >= CAT_FACE_SUPPORT_THRESHOLD,
        }


class CatFaceDetector:
    def __init__(self, model_candidates: tuple[Path, ...]):
        self.model_path = first_existing_path(model_candidates)
        self._model = None
        self.backend_name = "unavailable"
        self._load_error: str | None = None
        self._try_load_model()

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _try_load_model(self) -> None:
        if self.model_path is None:
            self._load_error = "missing_model"
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(str(self.model_path))
            self.backend_name = "ultralytics_yolo"
            self._load_error = None
        except Exception as exc:  # pragma: no cover - environment dependent
            self._model = None
            self.backend_name = "unavailable"
            self._load_error = f"{type(exc).__name__}: {exc}"

    def detect_largest_face(self, image_bgr: np.ndarray) -> list[int] | None:
        if not self.available or image_bgr is None or image_bgr.size == 0:
            return None
        height, width = image_bgr.shape[:2]
        max_side = max(height, width)
        scale = 1.0
        inference_image = image_bgr
        if max_side > YOLO_DETECT_MAX_SIDE:
            scale = YOLO_DETECT_MAX_SIDE / float(max_side)
            inference_image = cv2.resize(
                image_bgr,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        try:
            results = self._model.predict(source=inference_image, verbose=False, conf=YOLO_DETECT_CONF, imgsz=YOLO_DETECT_IMGSZ)
        except Exception:
            return None
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype="float32")
        ranked = sorted(
            zip(xyxy.tolist(), confidences.tolist(), strict=False),
            key=lambda item: (((item[0][2] - item[0][0]) * (item[0][3] - item[0][1])), item[1]),
            reverse=True,
        )
        x1, y1, x2, y2 = ranked[0][0]
        if scale != 1.0:
            inv_scale = 1.0 / scale
            x1 *= inv_scale
            y1 *= inv_scale
            x2 *= inv_scale
            y2 *= inv_scale
        x = max(int(round(x1)), 0)
        y = max(int(round(y1)), 0)
        w = max(int(round(x2 - x1)), 1)
        h = max(int(round(y2 - y1)), 1)
        return [x, y, w, h]


class OpenSetCalibrator:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.available = False
        self.feature_names: list[str] = []
        self.classes: list[str] = []
        self.coefficients: np.ndarray | None = None
        self.intercept: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.min_probability = OPEN_SET_CALIBRATOR_MIN_PROB
        self.min_margin = OPEN_SET_CALIBRATOR_MIN_MARGIN
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.load_error = "missing_model"
            return
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            self.feature_names = [str(name) for name in payload["feature_names"]]
            self.classes = [str(name) for name in payload["classes"]]
            self.coefficients = np.asarray(payload["coefficients"], dtype="float32")
            self.intercept = np.asarray(payload["intercept"], dtype="float32")
            self.mean = np.asarray(payload["mean"], dtype="float32")
            self.scale = np.asarray(payload["scale"], dtype="float32")
            self.min_probability = float(payload.get("min_probability", OPEN_SET_CALIBRATOR_MIN_PROB))
            self.min_margin = float(payload.get("min_margin", OPEN_SET_CALIBRATOR_MIN_MARGIN))
            self.metadata = dict(payload.get("metadata") or {})
            self.available = (
                self.coefficients.ndim == 2
                and self.intercept.ndim == 1
                and len(self.classes) == int(self.coefficients.shape[0]) == int(self.intercept.shape[0])
                and len(self.feature_names) == int(self.coefficients.shape[1]) == int(self.mean.shape[0]) == int(self.scale.shape[0])
            )
            if not self.available:
                self.load_error = "invalid_shape"
        except Exception as exc:  # pragma: no cover - file/content dependent
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        if not self.available or self.coefficients is None or self.intercept is None or self.mean is None or self.scale is None:
            return {
                "available": False,
                "predicted_group": None,
                "confidence": None,
                "margin": None,
                "confident": False,
                "probabilities": {},
            }
        vector = np.asarray([float(features.get(name, 0.0)) for name in self.feature_names], dtype="float32")
        safe_scale = np.where(self.scale == 0.0, 1.0, self.scale)
        standardized = (vector - self.mean) / safe_scale
        logits = np.dot(self.coefficients, standardized) + self.intercept
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= max(float(np.sum(probs)), 1e-8)
        ranking = np.argsort(probs)[::-1]
        top_index = int(ranking[0])
        runner_prob = float(probs[int(ranking[1])]) if len(ranking) > 1 else 0.0
        top_prob = float(probs[top_index])
        margin = top_prob - runner_prob
        probabilities = {
            class_name: round(float(probs[idx]), 6)
            for idx, class_name in enumerate(self.classes)
        }
        return {
            "available": True,
            "predicted_group": self.classes[top_index],
            "confidence": round(top_prob, 6),
            "margin": round(margin, 6),
            "confident": top_prob >= self.min_probability and margin >= self.min_margin,
            "probabilities": probabilities,
        }


@dataclass
class DetectionResult:
    face_detected: bool
    face_box: list[int] | None
    crop_bgr: np.ndarray
    crop_strategy: str
    crop_box: list[int] | None = None
    human_face_detected: bool = False
    detector_backend: str = "none"


class DuplicateImageError(ValueError):
    def __init__(self, duplicates: list[dict[str, str]]):
        self.duplicates = duplicates
        super().__init__(format_duplicate_image_message(duplicates))


class FaceCropper:
    def __init__(
        self,
        detector_mode: str | None = None,
        face_classifier: CatFaceBinaryClassifier | None = None,
    ):
        haar_dir = Path(cv2.data.haarcascades)
        self.face_classifier = face_classifier
        self.yolo_detector = CatFaceDetector(CAT_FACE_DETECTOR_MODEL_CANDIDATES)
        requested_mode = (detector_mode or "default").strip().lower()
        self.detector_mode = requested_mode
        self.prefer_yolo = False
        if requested_mode == "yolo":
            self.prefer_yolo = self.yolo_detector.available
        elif requested_mode == "haar":
            self.prefer_yolo = False
        else:
            self.prefer_yolo = ENABLE_YOLO_DETECTOR and self.yolo_detector.available
        self.cat_cascades = [
            cv2.CascadeClassifier(str(haar_dir / "haarcascade_frontalcatface_extended.xml")),
            cv2.CascadeClassifier(str(haar_dir / "haarcascade_frontalcatface.xml")),
        ]
        self.human_cascade = cv2.CascadeClassifier(str(haar_dir / "haarcascade_frontalface_default.xml"))
        self.max_detection_side = FACE_DETECT_MAX_SIDE
        if self.prefer_yolo:
            validation_mode = f"validated_{YOLO_FACE_VALIDATION_MIN_SCORE:.2f}" if self._can_validate_yolo_crop() else "raw"
            self.signature = f"yolo_cat_v2_{validation_mode}_{YOLO_DETECT_MAX_SIDE}_{YOLO_DETECT_IMGSZ}_{YOLO_DETECT_CONF:.2f}"
        else:
            self.signature = f"haar_cat_v2_scaled_{FACE_DETECT_MAX_SIDE}"

    def _can_validate_yolo_crop(self) -> bool:
        return bool(self.face_classifier and self.face_classifier.available)

    def _prepare_detection_gray(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image_bgr.shape[:2]
        max_side = max(height, width)
        scale = 1.0
        detection_bgr = image_bgr
        if max_side > self.max_detection_side:
            scale = self.max_detection_side / float(max_side)
            detection_bgr = cv2.resize(
                image_bgr,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(detection_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray), scale

    def _build_yolo_result(self, image_bgr: np.ndarray, face_box: list[int]) -> DetectionResult:
        x, y, w, h = face_box
        crop_box = expanded_crop_box(image_bgr, x, y, w, h, pad_ratio=0.24)
        crop = crop_from_box(image_bgr, crop_box)
        return DetectionResult(True, [int(x), int(y), int(w), int(h)], crop, "cat_face", crop_box=crop_box, detector_backend="yolo")

    def _yolo_crop_supported(self, crop_bgr: np.ndarray) -> bool:
        if not self._can_validate_yolo_crop() or self.face_classifier is None:
            return True
        face_check = self.face_classifier.predict(crop_bgr)
        score = face_check.get("cat_face_score")
        return score is not None and float(score) >= YOLO_FACE_VALIDATION_MIN_SCORE

    def _detect_haar_face(self, image_bgr: np.ndarray) -> DetectionResult | None:
        gray, scale = self._prepare_detection_gray(image_bgr)
        for cascade in self.cat_cascades:
            if cascade.empty():
                continue
            faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(48, 48))
            if len(faces):
                x, y, w, h = max(faces, key=lambda row: row[2] * row[3])
                if scale != 1.0:
                    inv_scale = 1.0 / scale
                    x = int(round(x * inv_scale))
                    y = int(round(y * inv_scale))
                    w = int(round(w * inv_scale))
                    h = int(round(h * inv_scale))
                crop_box = expanded_crop_box(image_bgr, x, y, w, h, pad_ratio=0.24)
                crop = crop_from_box(image_bgr, crop_box)
                return DetectionResult(True, [int(x), int(y), int(w), int(h)], crop, "cat_face", crop_box=crop_box, detector_backend="haar")
        return None

    def _detect_center_square(self, image_bgr: np.ndarray) -> DetectionResult:
        gray, _scale = self._prepare_detection_gray(image_bgr)
        human_detected = False
        if not self.human_cascade.empty():
            human_faces = self.human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
            human_detected = len(human_faces) > 0
        crop_box = center_square_box(image_bgr)
        return DetectionResult(False, None, crop_from_box(image_bgr, crop_box), "center_square", crop_box=crop_box, human_face_detected=human_detected, detector_backend="center_square")

    def detect_and_crop(self, image_bgr: np.ndarray) -> DetectionResult:
        if self.prefer_yolo:
            yolo_face = self.yolo_detector.detect_largest_face(image_bgr)
            if yolo_face is not None:
                yolo_result = self._build_yolo_result(image_bgr, yolo_face)
                if self._yolo_crop_supported(yolo_result.crop_bgr):
                    return yolo_result
                haar_result = self._detect_haar_face(image_bgr)
                if haar_result is not None:
                    return DetectionResult(
                        haar_result.face_detected,
                        haar_result.face_box,
                        haar_result.crop_bgr,
                        haar_result.crop_strategy,
                        crop_box=haar_result.crop_box,
                        human_face_detected=haar_result.human_face_detected,
                        detector_backend="haar_after_yolo_reject",
                    )
                center_result = self._detect_center_square(image_bgr)
                return DetectionResult(
                    center_result.face_detected,
                    center_result.face_box,
                    center_result.crop_bgr,
                    center_result.crop_strategy,
                    crop_box=center_result.crop_box,
                    human_face_detected=center_result.human_face_detected,
                    detector_backend="center_square_after_yolo_reject",
                )

        haar_result = self._detect_haar_face(image_bgr)
        if haar_result is not None:
            return haar_result

        return self._detect_center_square(image_bgr)


class PredictorService:
    def __init__(
        self,
        *,
        model_path: Path | None = None,
        calibrator_path: Path | None = None,
        detector_mode: str | None = None,
        index_cache_path: Path | None = None,
    ):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CATS_DIR.mkdir(parents=True, exist_ok=True)
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        QUERY_DIR.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(model_path) if model_path is not None else MODEL_PATH
        self.index_cache_path = Path(index_cache_path) if index_cache_path is not None else INDEX_CACHE_PATH
        self.encoder = EncoderBackend(self.model_path)
        self.face_classifier = CatFaceBinaryClassifier(CAT_FACE_MODEL_CANDIDATES, CAT_FACE_LABELS_CANDIDATES)
        self.cropper = FaceCropper(detector_mode=detector_mode, face_classifier=self.face_classifier)
        if calibrator_path is not None:
            self.calibrator_path = Path(calibrator_path)
        elif self.cropper.prefer_yolo and YOLO_OPEN_SET_CALIBRATOR_PATH.exists():
            self.calibrator_path = YOLO_OPEN_SET_CALIBRATOR_PATH
        else:
            self.calibrator_path = OPEN_SET_CALIBRATOR_PATH
        self.group_calibrator = OpenSetCalibrator(self.calibrator_path)
        self._last_store_mtime = 0
        self.maybe_reload_store()
        self.index_records: list[dict[str, Any]] = []
        self.index_vectors: np.ndarray | None = None
        self.sync_reference_dirs()
        if self.state.get("index_dirty") or not self.load_index_cache():
            self.rebuild_index()

    def _default_store(self) -> dict[str, Any]:
        return {
            "next_cat_id": 1,
            "cats": [],
            "reference_sets": {
                "not_cat": {"label": "ไม่ใช่แมว", "images": []},
                "unknown_cat": {"label": "แมวที่ไม่อยู่ในระบบ", "images": []},
            },
            "index_dirty": True,
            "indexed_total_images": 0,
            "indexed_known_images": 0,
            "indexed_special_images": 0,
            "backend_name": self.encoder.backend_name,
        }

    def _load_store(self) -> dict[str, Any]:
        if IS_SERVERLESS and PACKAGE_STORE_PATH.exists():
            data = json.loads(PACKAGE_STORE_PATH.read_text(encoding="utf-8"))
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            self._save_store(data)
        elif STORE_PATH.exists():
            data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        else:
            data = self._default_store()
            self._save_store(data)
            return data

        if data:
            defaults = self._default_store()
            for key, value in defaults.items():
                data.setdefault(key, value)
            default_refs = defaults["reference_sets"]
            refs = data.setdefault("reference_sets", {})
            for reference_key, reference_default in default_refs.items():
                ref = refs.setdefault(reference_key, {})
                ref.setdefault("label", reference_default["label"])
                ref.setdefault("images", [])
            data["backend_name"] = self.encoder.backend_name
            return data

    def _save_store(self, data: dict[str, Any] | None = None) -> None:
        payload = data or self.state
        payload["backend_name"] = self.encoder.backend_name
        STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            self._last_store_mtime = STORE_PATH.stat().st_mtime
        except Exception:
            pass

    def maybe_reload_store(self) -> bool:
        if not STORE_PATH.exists():
            if not hasattr(self, "state"):
                self.state = self._default_store()
            return False
        try:
            mtime = STORE_PATH.stat().st_mtime
            if mtime > getattr(self, "_last_store_mtime", 0):
                new_state = self._load_store()
                if not hasattr(self, "state") or new_state.get("cats") != self.state.get("cats") or new_state.get("reference_sets") != self.state.get("reference_sets") or new_state.get("index_dirty"):
                    self.state = new_state
                    self.state["index_dirty"] = True
                    self._last_store_mtime = mtime
                    return True
                self._last_store_mtime = mtime
        except Exception:
            pass
        return False

    def sync_reference_dirs(self) -> bool:
        self.maybe_reload_store()
        changed = False
        for cat in self.state.get("cats", []):
            seen_paths: set[str] = set()
            kept_images = []
            for image in cat.get("images", []):
                file_path = image.get("file_path")
                if not isinstance(file_path, str) or file_path in seen_paths:
                    changed = True
                    continue
                path = BASE_DIR / file_path
                if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                    changed = True
                    continue
                content_hash = image.get("content_hash")
                if not isinstance(content_hash, str):
                    content_hash = file_sha256(path)
                    if content_hash:
                        image["content_hash"] = content_hash
                        changed = True
                seen_paths.add(file_path)
                kept_images.append(image)
            if kept_images != cat.get("images", []):
                cat["images"] = kept_images

        for reference_key in REFERENCE_KEYS:
            target_dir = REFERENCE_DIR / reference_key
            target_dir.mkdir(parents=True, exist_ok=True)
            ref = self.state["reference_sets"][reference_key]

            disk_records = {}
            for path in sorted(target_dir.iterdir(), key=reference_file_sort_key):
                if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                file_path = str(path.relative_to(BASE_DIR).as_posix())
                disk_records[file_path] = {"name": path.name, "file_path": file_path}

            seen_paths: set[str] = set()
            kept_images = []
            for image in ref.get("images", []):
                file_path = image.get("file_path")
                if not isinstance(file_path, str) or file_path in seen_paths or file_path not in disk_records:
                    changed = True
                    continue
                record = disk_records[file_path]
                kept_images.append(record)
                seen_paths.add(file_path)
                if image != record:
                    changed = True

            missing_images = [record for file_path, record in disk_records.items() if file_path not in seen_paths]
            if missing_images:
                changed = True
            ref["images"] = missing_images + kept_images

        if changed:
            self.state["index_dirty"] = True
            self._save_store()
        return changed

    def summary(self) -> dict[str, Any]:
        cat_images = sum(len(cat["images"]) for cat in self.state["cats"])
        not_cat_images = len(self.state["reference_sets"]["not_cat"]["images"])
        unknown_images = len(self.state["reference_sets"]["unknown_cat"]["images"])
        actual_total = cat_images + not_cat_images + unknown_images
        index_status = "empty" if actual_total == 0 else ("needs_train" if self.state.get("index_dirty") else "ready")
        return {
            "index_status": index_status,
            "actual_total_images": actual_total,
            "indexed_total_images": self.state.get("indexed_total_images", 0),
            "indexed_known_images": self.state.get("indexed_known_images", 0),
            "indexed_special_images": self.state.get("indexed_special_images", 0),
            "dataset_images": cat_images,
            "not_cat_folder_images": not_cat_images,
            "unknown_cat_folder_images": unknown_images,
            "known_labels": sorted(cat["name"] for cat in self.state["cats"]),
            "backend_name": self.encoder.backend_name,
            "backend_load_error": self.encoder.load_error,
            "cat_face_model_backend": self.face_classifier.backend_name,
            "cat_face_model_error": self.face_classifier.load_error,
            "cat_face_model_path": str(self.face_classifier.model_path) if self.face_classifier.model_path else None,
            "cat_face_detector_path": str(self.cropper.yolo_detector.model_path) if self.cropper.yolo_detector.model_path else None,
            "open_set_calibrator_path": str(self.calibrator_path),
            "detector_mode": self.cropper.detector_mode,
            "detector_prefer_yolo": self.cropper.prefer_yolo,
        }

    def serialize_cat(self, cat: dict[str, Any]) -> dict[str, Any]:
        images = [
            {
                "name": image["name"],
                "source_name": image.get("source_name") or image["name"],
                "content_hash": image.get("content_hash"),
                "url": url_for("uploaded_file", file_path=image["file_path"]),
            }
            for image in cat["images"]
        ]
        return {
            "id": cat["id"],
            "name": cat["name"],
            "owner": cat.get("owner", ""),
            "contact": cat.get("contact", ""),
            "location": cat.get("location", ""),
            "image_count": len(images),
            "images": images,
            "cover_image": images[0]["url"] if images else None,
        }

    def serialize_reference_set(self, reference_key: str, limit: int = 24) -> dict[str, Any]:
        ref = self.state["reference_sets"][reference_key]
        images = ref["images"][:limit]
        payload = [
            {
                "name": image["name"],
                "source_name": image.get("source_name") or image["name"],
                "content_hash": image.get("content_hash"),
                "url": url_for("uploaded_file", file_path=image["file_path"]),
            }
            for image in images
        ]
        total_images = len(ref["images"])
        return {
            "key": reference_key,
            "label": ref["label"],
            "image_count": total_images,
            "hidden_count": max(total_images - len(payload), 0),
            "images": payload,
        }

    def find_cat(self, cat_id: int) -> dict[str, Any] | None:
        return next((cat for cat in self.state["cats"] if cat["id"] == cat_id), None)

    def find_known_cat_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        for cat in self.state["cats"]:
            for image in cat["images"]:
                if image.get("content_hash") == content_hash:
                    return cat
        return None

    def prepare_upload(self, storage: FileStorage) -> dict[str, Any] | None:
        original_name = storage.filename or ""
        if not original_name:
            return None
        extension = Path(original_name).suffix.lower()
        if extension not in VALID_EXTENSIONS:
            raise ValueError("รองรับเฉพาะ JPG, PNG, WEBP")
        data = storage.read()
        return {
            "original_name": original_name,
            "safe_name": secure_filename(original_name) or f"upload{extension}",
            "extension": extension,
            "content_hash": hashlib.sha256(data).hexdigest(),
            "data": data,
        }

    def save_upload(self, storage: FileStorage, target_dir: Path) -> dict[str, str]:
        upload = self.prepare_upload(storage)
        if upload is None:
            raise ValueError("ไม่พบไฟล์อัปโหลด")
        target_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex}{upload['extension']}"
        (target_dir / stored_name).write_bytes(upload["data"])
        return {
            "name": stored_name,
            "source_name": upload["original_name"],
            "file_path": str((target_dir.relative_to(BASE_DIR) / stored_name).as_posix()),
            "content_hash": upload["content_hash"],
        }

    def cat_image_hash_index(self) -> dict[str, dict[str, str]]:
        index = {}
        for cat in self.state["cats"]:
            for image in cat["images"]:
                content_hash = image.get("content_hash")
                if not isinstance(content_hash, str):
                    content_hash = file_sha256(BASE_DIR / image["file_path"])
                    if content_hash:
                        image["content_hash"] = content_hash
                if isinstance(content_hash, str) and content_hash not in index:
                    index[content_hash] = {
                        "cat_name": str(cat["name"]),
                        "image_name": str(image.get("source_name") or image.get("name") or "รูปเดิม"),
                    }
        return index

    def reference_image_hash_index(self) -> dict[str, dict[str, str]]:
        index = {}
        for reference_key in REFERENCE_KEYS:
            ref = self.state["reference_sets"][reference_key]
            label = str(ref.get("label") or reference_key)
            for image in ref["images"]:
                content_hash = image.get("content_hash")
                if not isinstance(content_hash, str):
                    content_hash = file_sha256(BASE_DIR / image["file_path"])
                    if content_hash:
                        image["content_hash"] = content_hash
                if isinstance(content_hash, str) and content_hash not in index:
                    index[content_hash] = {
                        "cat_name": label,
                        "image_name": str(image.get("source_name") or image.get("name") or "รูปเดิม"),
                    }
        return index

    def reference_duplicate_hash_index(self) -> dict[str, dict[str, str]]:
        index = self.cat_image_hash_index()
        for content_hash, metadata in self.reference_image_hash_index().items():
            if content_hash not in index:
                index[content_hash] = metadata
        return index

    def save_cat_uploads(self, files: list[FileStorage], target_dir: Path) -> list[dict[str, str]]:
        uploads = [upload for storage in files if storage.filename for upload in [self.prepare_upload(storage)] if upload is not None]
        if not uploads:
            return []

        existing_hashes = self.cat_image_hash_index()
        seen_upload_hashes: dict[str, str] = {}
        duplicates: list[dict[str, str]] = []
        for upload in uploads:
            content_hash = upload["content_hash"]
            if content_hash in existing_hashes:
                existing = existing_hashes[content_hash]
                duplicates.append({
                    "filename": upload["original_name"],
                    "duplicate_of": existing["image_name"],
                    "cat_name": existing["cat_name"],
                    "source": "existing",
                })
                continue
            if content_hash in seen_upload_hashes:
                duplicates.append({
                    "filename": upload["original_name"],
                    "duplicate_of": seen_upload_hashes[content_hash],
                    "cat_name": "",
                    "source": "upload_batch",
                })
                continue
            seen_upload_hashes[content_hash] = upload["original_name"]

        if duplicates:
            raise DuplicateImageError(duplicates)

        target_dir.mkdir(parents=True, exist_ok=True)
        saved_images = []
        for upload in uploads:
            stored_name = f"{uuid.uuid4().hex}{upload['extension']}"
            (target_dir / stored_name).write_bytes(upload["data"])
            saved_images.append({
                "name": stored_name,
                "source_name": upload["original_name"],
                "file_path": str((target_dir.relative_to(BASE_DIR) / stored_name).as_posix()),
                "content_hash": upload["content_hash"],
            })
        return saved_images

    def save_reference_uploads(self, files: list[FileStorage], target_dir: Path) -> list[dict[str, str]]:
        uploads = [upload for storage in files if storage.filename for upload in [self.prepare_upload(storage)] if upload is not None]
        if not uploads:
            return []

        existing_hashes = self.reference_duplicate_hash_index()
        seen_upload_hashes: dict[str, str] = {}
        duplicates: list[dict[str, str]] = []
        for upload in uploads:
            content_hash = upload["content_hash"]
            if content_hash in existing_hashes:
                existing = existing_hashes[content_hash]
                duplicates.append({
                    "filename": upload["original_name"],
                    "duplicate_of": existing["image_name"],
                    "cat_name": existing["cat_name"],
                    "source": "existing",
                })
                continue
            if content_hash in seen_upload_hashes:
                duplicates.append({
                    "filename": upload["original_name"],
                    "duplicate_of": seen_upload_hashes[content_hash],
                    "cat_name": "",
                    "source": "upload_batch",
                })
                continue
            seen_upload_hashes[content_hash] = upload["original_name"]

        if duplicates:
            raise DuplicateImageError(duplicates)

        target_dir.mkdir(parents=True, exist_ok=True)
        saved_images = []
        for upload in uploads:
            stored_name = f"{uuid.uuid4().hex}{upload['extension']}"
            (target_dir / stored_name).write_bytes(upload["data"])
            saved_images.append({
                "name": stored_name,
                "source_name": upload["original_name"],
                "file_path": str((target_dir.relative_to(BASE_DIR) / stored_name).as_posix()),
                "content_hash": upload["content_hash"],
            })
        return saved_images

    def delete_image_file(self, file_path: str) -> None:
        path = (BASE_DIR / file_path).resolve()
        if path.is_file() and BASE_DIR.resolve() in path.parents:
            path.unlink(missing_ok=True)

    def index_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for cat in self.state["cats"]:
            for image in cat["images"]:
                sources.append({
                    "label": cat["name"],
                    "group": "known",
                    "cat_id": cat["id"],
                    "image_name": image["name"],
                    "file_path": image["file_path"],
                })
        for reference_key in ("unknown_cat", "not_cat"):
            for image in self.state["reference_sets"][reference_key]["images"]:
                sources.append({
                    "label": reference_key,
                    "group": reference_key,
                    "cat_id": None,
                    "image_name": image["name"],
                    "file_path": image["file_path"],
                })
        return sources

    def index_fingerprint(self, sources: list[dict[str, Any]] | None = None) -> str:
        payload = {
            **self.index_global_signature(),
            "sources": [
                {**source, "file": file_signature(BASE_DIR / source["file_path"])}
                for source in (sources or self.index_sources())
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def index_global_signature(self) -> dict[str, Any]:
        return {
            "version": INDEX_CACHE_VERSION,
            "backend_name": self.encoder.backend_name,
            "model": file_signature(self.model_path),
            "image_size": IMAGE_SIZE,
            "cropper": self.cropper.signature,
            "cat_face_model": file_signature(self.face_classifier.model_path) if self.face_classifier.model_path else None,
            "cat_face_detector": file_signature(self.cropper.yolo_detector.model_path) if self.cropper.yolo_detector.model_path else None,
            "input_preprocess": "fit_pad_replicate",
            "feature_fusion": {
                "classic_weight": CLASSIC_FEATURE_WEIGHT,
                "classic_descriptor": "gray32_rgbhist16_edge16",
            },
        }

    def vector_cache_key(self, source: dict[str, Any]) -> str:
        payload = {
            **self.index_global_signature(),
            "file_path": source["file_path"],
            "file": file_signature(BASE_DIR / source["file_path"]),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def load_cached_index_entries(self) -> tuple[dict[str, tuple[np.ndarray, dict[str, Any]]], dict[str, tuple[np.ndarray, dict[str, Any]]]]:
        current_entries: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        legacy_entries: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        if not self.index_cache_path.exists():
            return current_entries, legacy_entries
        try:
            with np.load(self.index_cache_path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata"].item()))
                records = metadata["records"]
                vectors = np.asarray(cache["vectors"], dtype="float32")
            if not isinstance(records, list) or len(records) != int(vectors.shape[0]):
                return current_entries, legacy_entries

            vector_keys = metadata.get("vector_keys")
            if isinstance(vector_keys, list) and len(vector_keys) == len(records):
                for key, record, vector in zip(vector_keys, records, vectors, strict=False):
                    if isinstance(key, str):
                        current_entries[key] = (np.asarray(vector, dtype="float32"), record)
            # Older caches only keyed vectors by file path, which is unsafe after
            # changing encoder backend, model file, cropper behavior, or image size.
            # Rebuild those entries instead of silently mixing incompatible vectors.
        except Exception:
            return {}, {}
        return current_entries, legacy_entries

    def load_index_cache(self) -> bool:
        if not self.index_cache_path.exists():
            return False
        try:
            expected_fingerprint = self.index_fingerprint()
            with np.load(self.index_cache_path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata"].item()))
                records = metadata["records"]
                vectors = np.asarray(cache["vectors"], dtype="float32")
            if metadata.get("fingerprint") != expected_fingerprint:
                return False
            if not isinstance(records, list) or len(records) != int(vectors.shape[0]):
                return False
            self.index_records = records
            self.index_vectors = vectors if len(records) else None
            return True
        except Exception:
            return False

    def save_index_cache(self, fingerprint: str, vector_keys: list[str]) -> None:
        if self.index_vectors is None:
            self.index_cache_path.unlink(missing_ok=True)
            return
        metadata = json.dumps({
            "fingerprint": fingerprint,
            "records": self.index_records,
            "vector_keys": vector_keys,
        }, ensure_ascii=False)
        np.savez_compressed(self.index_cache_path, vectors=self.index_vectors, metadata=np.array(metadata))

    def rebuild_index(self) -> None:
        self.sync_reference_dirs()
        sources = self.index_sources()
        cached_entries, legacy_entries = self.load_cached_index_entries()
        records, vectors, vector_keys = [], [], []
        pending_crops, pending_indices = [], []
        reused_vectors = 0
        total_sources = len(sources)
        if total_sources:
            print(f"Building image index from {total_sources} images...", flush=True)
        for position, source in enumerate(sources, start=1):
            path = BASE_DIR / source["file_path"]
            if not path.exists():
                continue

            vector_key = self.vector_cache_key(source)
            cached_entry = cached_entries.get(vector_key) or legacy_entries.get(source["file_path"])
            if cached_entry is not None:
                vector, cached_record = cached_entry
                records.append({
                    **source,
                    "crop_strategy": cached_record.get("crop_strategy", "cached"),
                    "face_detected": bool(cached_record.get("face_detected", False)),
                    "detector_backend": cached_record.get("detector_backend", "cached"),
                })
                vectors.append(vector)
                vector_keys.append(vector_key)
                reused_vectors += 1
            else:
                image_bgr = read_image(path)
                if image_bgr is None:
                    continue
                detection = self.cropper.detect_and_crop(image_bgr)
                records.append({
                    **source,
                    "crop_strategy": detection.crop_strategy,
                    "face_detected": detection.face_detected,
                    "detector_backend": detection.detector_backend,
                })
                vectors.append(None)
                vector_keys.append(vector_key)
                pending_indices.append(len(vectors) - 1)
                pending_crops.append(detection.crop_bgr)

            if total_sources and (position == total_sources or position % 25 == 0):
                print(f"Prepared {position}/{total_sources} images for indexing...", flush=True)
        if pending_crops:
            print(f"Encoding {len(pending_crops)} new/changed crops with {self.encoder.backend_name}; reused {reused_vectors} cached vectors.", flush=True)
        elif reused_vectors:
            print(f"Reused {reused_vectors} cached vectors; no image encoding needed.", flush=True)
        encoded_vectors = self.encoder.encode_many(pending_crops)
        for index, vector in zip(pending_indices, encoded_vectors, strict=False):
            vectors[index] = vector
        self.index_records = records
        completed_vectors = [vector for vector in vectors if vector is not None]
        self.index_vectors = np.vstack(completed_vectors).astype("float32") if completed_vectors else None
        self.state["indexed_total_images"] = len(records)
        self.state["indexed_known_images"] = sum(1 for row in records if row["group"] == "known")
        self.state["indexed_special_images"] = sum(1 for row in records if row["group"] != "known")
        self.state["index_dirty"] = False
        self.save_index_cache(self.index_fingerprint(sources), vector_keys)
        self._save_store()

    def create_cat(self, form: dict[str, str], files: list[FileStorage]) -> dict[str, Any]:
        name = form.get("name", "").strip()
        if not name:
            raise ValueError("กรอกชื่อแมว")
        cat_id = int(self.state["next_cat_id"])
        target_dir = CATS_DIR / str(cat_id)
        images = self.save_cat_uploads(files, target_dir)
        self.state["next_cat_id"] = cat_id + 1
        cat = {"id": cat_id, "name": name, "owner": form.get("owner", "").strip(), "contact": form.get("contact", "").strip(), "location": form.get("location", "").strip(), "images": images}
        self.state["cats"].append(cat)
        self.state["index_dirty"] = True
        self.rebuild_index()
        return cat

    def update_cat(self, cat_id: int, form: dict[str, str], files: list[FileStorage]) -> dict[str, Any]:
        cat = self.find_cat(cat_id)
        if not cat:
            raise KeyError("ไม่พบข้อมูลแมว")
        name = form.get("name", "").strip()
        target_dir = CATS_DIR / str(cat_id)
        new_images = self.save_cat_uploads(files, target_dir)
        if name:
            cat["name"] = name
        cat["owner"] = form.get("owner", "").strip()
        cat["contact"] = form.get("contact", "").strip()
        cat["location"] = form.get("location", "").strip()
        cat["images"].extend(new_images)
        self.state["index_dirty"] = True
        self.rebuild_index()
        return cat

    def delete_cat(self, cat_id: int) -> None:
        cat = self.find_cat(cat_id)
        if not cat:
            raise KeyError("ไม่พบข้อมูลแมว")
        self.state["cats"] = [row for row in self.state["cats"] if row["id"] != cat_id]
        for image in cat["images"]:
            self.delete_image_file(image["file_path"])
        target_dir = CATS_DIR / str(cat_id)
        if target_dir.exists():
            for child in target_dir.glob('*'):
                child.unlink(missing_ok=True)
            target_dir.rmdir()
        self.state["index_dirty"] = True
        self.rebuild_index()

    def delete_cat_image(self, cat_id: int, image_name: str) -> None:
        cat = self.find_cat(cat_id)
        if not cat:
            raise KeyError("ไม่พบข้อมูลแมว")
        image = next((row for row in cat["images"] if row["name"] == image_name), None)
        if not image:
            raise KeyError("ไม่พบรูป")
        cat["images"] = [row for row in cat["images"] if row["name"] != image_name]
        self.delete_image_file(image["file_path"])
        self.state["index_dirty"] = True
        self.rebuild_index()

    def upload_reference_images(self, reference_key: str, files: list[FileStorage]) -> None:
        if reference_key not in self.state["reference_sets"]:
            raise KeyError("ไม่พบ reference set")
        self.sync_reference_dirs()
        target_dir = REFERENCE_DIR / reference_key
        uploads = self.save_reference_uploads(files, target_dir)
        self.state["reference_sets"][reference_key]["images"] = uploads + self.state["reference_sets"][reference_key]["images"]
        self.state["index_dirty"] = True
        self.rebuild_index()

    def delete_reference_image(self, reference_key: str, image_name: str) -> None:
        if reference_key not in self.state["reference_sets"]:
            raise KeyError("ไม่พบ reference set")
        self.sync_reference_dirs()
        images = self.state["reference_sets"][reference_key]["images"]
        image = next((row for row in images if row["name"] == image_name), None)
        if not image:
            raise KeyError("ไม่พบรูป")
        self.state["reference_sets"][reference_key]["images"] = [row for row in images if row["name"] != image_name]
        self.delete_image_file(image["file_path"])
        self.state["index_dirty"] = True
        self.rebuild_index()

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        self.sync_reference_dirs()
        if self.state.get("index_dirty"):
            self.rebuild_index()
        if self.index_vectors is None or not self.index_records:
            raise ValueError("ยังไม่มีรูปอ้างอิงในระบบ")
        image_bgr = read_image_bytes(image_bytes)
        if image_bgr is None:
            raise ValueError("ไม่สามารถอ่านรูปได้")
        query_hash = hashlib.sha256(image_bytes).hexdigest()
        exact_cat = self.find_known_cat_by_hash(query_hash)
        quality = compute_quality(image_bgr)
        detection = self.cropper.detect_and_crop(image_bgr)
        face_check = self.face_classifier.predict(detection.crop_bgr)
        query_cat_face_score = float(face_check["cat_face_score"]) if face_check.get("cat_face_score") is not None else 0.0
        query_cat_like = bool(face_check.get("available")) and query_cat_face_score >= CAT_LIKE_QUERY_THRESHOLD
        effective_detection = effective_detection_from_face_support(detection, face_check)
        model_input_preview_bgr = fit_image_to_canvas(detection.crop_bgr, IMAGE_SIZE, IMAGE_SIZE)
        query_vector = self.encoder.encode(detection.crop_bgr)
        similarities = np.dot(self.index_vectors, query_vector)
        top_candidates = build_top_candidates(self.index_records, similarities, top_k=TOP_K, query_cat_like=query_cat_like)
        score_summary = build_score_summary(
            self.index_records,
            similarities,
            face_check=face_check,
            exact_known_label=exact_cat["name"] if exact_cat else None,
            query_cat_like=query_cat_like,
        )
        decision = decide_prediction(top_candidates, quality, effective_detection, score_summary)
        decision, calibrator_result = apply_open_set_calibrator(
            self.group_calibrator,
            current_decision=decision,
            top_candidates=top_candidates,
            quality=quality,
            detection=effective_detection,
            face_check=face_check,
            score_summary=score_summary,
        )
        matched_cat = None
        if decision["final_label"] not in {"unknown_cat", "not_cat", "low_quality"}:
            matched_cat = next((cat for cat in self.state["cats"] if cat["name"] == decision["final_label"]), None)
        vote_candidates = effective_vote_candidates(top_candidates)
        top_class_summary = summarize_classes(vote_candidates)
        top_class_summary_raw = summarize_classes(top_candidates)
        raw_summary_by_label = {str(row["label"]): row for row in top_class_summary_raw}
        for row in top_class_summary:
            raw_row = raw_summary_by_label.get(str(row["label"]))
            row["top10_votes"] = int(raw_row.get("votes", row["votes"])) if raw_row else int(row["votes"])
            row["top10_weighted_sum"] = round(float(raw_row.get("weighted_sum", row["weighted_sum"])), 6) if raw_row else round(float(row["weighted_sum"]), 6)
        display_known_name = score_summary["best_known_label"]
        display_known_score = float(score_summary["best_known_score"]) if score_summary["best_known_label"] else None
        if matched_cat:
            matched_summary = next((row for row in top_class_summary if row["label"] == matched_cat["name"]), None)
            if matched_summary:
                display_known_name = str(matched_summary["label"])
                display_known_score = float(matched_summary["best_score"])
        winner_raw = raw_summary_by_label.get(str(decision["winner_label"])) if decision.get("winner_label") else None
        runner_raw = raw_summary_by_label.get(str(decision["runner_up_label"])) if decision.get("runner_up_label") else None
        vote_cutoff_score = round(max(float(row["score"]) for row in top_candidates) - VOTE_RELATIVE_MARGIN, 6) if top_candidates else None
        face_detection_source = resolve_face_detection_source(detection, face_check)
        return {
            "final_label": decision["final_label"],
            "final_label_display": decision["final_label_display"],
            "decision_reason": decision["reason"],
            "decision_type": decision["decision_type"],
            "group_calibrator_available": bool(calibrator_result.get("available")),
            "group_calibrator_prediction": calibrator_result.get("predicted_group"),
            "group_calibrator_confidence": calibrator_result.get("confidence"),
            "group_calibrator_margin": calibrator_result.get("margin"),
            "group_calibrator_confident": bool(calibrator_result.get("confident")),
            "group_calibrator_probabilities": calibrator_result.get("probabilities"),
            "quality_pass": quality["quality_pass"],
            "quality_reasons": quality["quality_reasons"],
            "blur_score": round(quality["blur_score"], 4),
            "brightness": round(quality["brightness"], 4),
            "face_detected": effective_detection.face_detected,
            "localized_face_detected": bool(detection.face_detected),
            "face_box": detection.face_box,
            "crop_box": detection.crop_box,
            "image_width": int(image_bgr.shape[1]),
            "image_height": int(image_bgr.shape[0]),
            "crop_strategy": detection.crop_strategy,
            "human_face_detected": detection.human_face_detected,
            "detector_backend": detection.detector_backend,
            "face_detection_source": face_detection_source,
            "cat_face_model_available": bool(face_check.get("available")),
            "cat_face_model_label": face_check.get("predicted_label"),
            "cat_face_model_score": face_check.get("cat_face_score"),
            "cat_face_model_non_face_score": face_check.get("not_cat_face_score"),
            "cat_face_model_supported": bool(face_check.get("cat_face_supported")),
            "query_cat_like": query_cat_like,
            "best_known_name": display_known_name,
            "best_known_score": round(display_known_score, 6) if display_known_score is not None else None,
            "second_known_score": round(float(score_summary["second_known_score"]), 6) if score_summary["second_known_score"] else None,
            "best_unknown_score": round(float(score_summary["best_unknown_score"]), 6) if score_summary["best_unknown_score"] else None,
            "best_not_cat_score": round(float(score_summary["best_not_cat_score"]), 6) if score_summary["best_not_cat_score"] else None,
            "matched_cat": self.serialize_cat(matched_cat) if matched_cat else None,
            "model_input_preview": {
                "data_url": encode_image_data_url(model_input_preview_bgr, extension=".png"),
                "source_width": int(detection.crop_bgr.shape[1]),
                "source_height": int(detection.crop_bgr.shape[0]),
                "target_width": int(IMAGE_SIZE),
                "target_height": int(IMAGE_SIZE),
                "padding_mode": "replicate",
                "scan_box": relative_scan_box(detection.face_box, detection.crop_box, IMAGE_SIZE, IMAGE_SIZE),
            },
            "top_candidates": top_candidates,
            "top_class_summary": top_class_summary,
            "top_class_summary_raw": top_class_summary_raw,
            "vote_window": {
                "effective_candidate_count": len(vote_candidates),
                "top_candidate_count": len(top_candidates),
                "cutoff_score": vote_cutoff_score,
                "relative_margin": VOTE_RELATIVE_MARGIN,
            },
            "decision": {
                "winner_label": decision["winner_label"],
                "winner_votes": decision["winner_votes"],
                "winner_top10_votes": int(winner_raw.get("votes", decision["winner_votes"])) if winner_raw else int(decision["winner_votes"]),
                "winner_weighted_sum": decision["winner_weighted_sum"],
                "winner_avg_score": decision["winner_avg_score"],
                "runner_up_label": decision["runner_up_label"],
                "runner_up_votes": decision["runner_up_votes"],
                "runner_up_top10_votes": int(runner_raw.get("votes", decision["runner_up_votes"])) if runner_raw else int(decision["runner_up_votes"]),
                "runner_up_weighted_sum": decision["runner_up_weighted_sum"],
            },
            "model_backend": self.encoder.backend_name,
            "model_load_error": self.encoder.load_error,
        }


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype="float32")
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def format_duplicate_image_message(duplicates: list[dict[str, str]]) -> str:
    if not duplicates:
        return "พบรูปซ้ำ"
    details = []
    for duplicate in duplicates:
        filename = duplicate.get("filename", "ไฟล์ที่เลือก")
        duplicate_of = duplicate.get("duplicate_of", "รูปเดิม")
        if duplicate.get("source") == "upload_batch":
            details.append(f"{filename} ซ้ำกับไฟล์ {duplicate_of} ที่เลือกพร้อมกัน")
        else:
            cat_name = duplicate.get("cat_name") or "แมวในระบบ"
            details.append(f"{filename} ซ้ำกับรูปของ {cat_name} ({duplicate_of})")
    return "พบรูปซ้ำ: " + "; ".join(details)


def read_image(path: Path) -> np.ndarray | None:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size >= JPEG_REDUCED_8_SIZE:
            return cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_8)
        if size >= JPEG_REDUCED_4_SIZE:
            return cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_4)
        if size >= JPEG_REDUCED_2_SIZE:
            return cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_2)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def first_existing_path(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def file_signature(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        try:
            display_path = str(path.resolve().relative_to(BASE_DIR.resolve()).as_posix())
        except ValueError:
            display_path = str(path)
        return {"path": display_path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"path": str(path), "missing": True}


def reference_file_sort_key(path: Path) -> tuple[float, str]:
    try:
        modified_time = path.stat().st_mtime
    except OSError:
        modified_time = 0.0
    return (-modified_time, path.name.casefold())


def resize_to_max_side(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    current_max_side = max(height, width)
    if current_max_side <= max_side:
        return image_bgr
    scale = max_side / float(current_max_side)
    return cv2.resize(
        image_bgr,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def fit_image_to_canvas(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image
    scale = min(target_width / float(width), target_height / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    pad_left = max((target_width - resized_width) // 2, 0)
    pad_right = max(target_width - resized_width - pad_left, 0)
    pad_top = max((target_height - resized_height) // 2, 0)
    pad_bottom = max(target_height - resized_height - pad_top, 0)
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        return resized
    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )


def is_jpeg_bytes(image_bytes: bytes) -> bool:
    return len(image_bytes) >= 3 and image_bytes[:3] == b"\xff\xd8\xff"


def read_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    if is_jpeg_bytes(image_bytes):
        if len(image_bytes) >= JPEG_REDUCED_8_SIZE:
            image = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_8)
        elif len(image_bytes) >= JPEG_REDUCED_4_SIZE:
            image = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_4)
        elif len(image_bytes) >= JPEG_REDUCED_2_SIZE:
            image = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_2)
        else:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return resize_to_max_side(image, QUERY_IMAGE_MAX_SIDE)


def encode_image_data_url(image_bgr: np.ndarray, extension: str = ".png", quality: int = 90) -> str | None:
    if image_bgr is None or image_bgr.size == 0:
        return None
    suffix = extension.lower()
    if suffix == ".png":
        success, encoded = cv2.imencode(".png", image_bgr)
        mime_type = "image/png"
    else:
        success, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        mime_type = "image/jpeg"
    if not success:
        return None
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def crop_from_box(image_bgr: np.ndarray, box: list[int] | tuple[int, int, int, int] | None) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0 or not box:
        return image_bgr
    x, y, w, h = [int(value) for value in box]
    crop = image_bgr[y:y + h, x:x + w]
    return crop if crop.size else image_bgr


def expanded_crop_box(image_bgr: np.ndarray, x: int, y: int, w: int, h: int, pad_ratio: float = 0.2) -> list[int]:
    height, width = image_bgr.shape[:2]
    pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
    x1, y1 = max(x - pad_w, 0), max(y - pad_h, 0)
    x2, y2 = min(x + w + pad_w, width), min(y + h + pad_h, height)
    return [int(x1), int(y1), int(max(x2 - x1, 1)), int(max(y2 - y1, 1))]


def expand_crop(image_bgr: np.ndarray, x: int, y: int, w: int, h: int, pad_ratio: float = 0.2) -> np.ndarray:
    return crop_from_box(image_bgr, expanded_crop_box(image_bgr, x, y, w, h, pad_ratio=pad_ratio))


def center_square_box(image_bgr: np.ndarray) -> list[int]:
    height, width = image_bgr.shape[:2]
    size = min(height, width)
    x, y = max((width - size) // 2, 0), max((height - size) // 2, 0)
    return [int(x), int(y), int(max(size, 1)), int(max(size, 1))]


def center_square_crop(image_bgr: np.ndarray) -> np.ndarray:
    return crop_from_box(image_bgr, center_square_box(image_bgr))


def relative_scan_box(face_box: list[int] | None, crop_box: list[int] | None, target_width: int, target_height: int) -> dict[str, float] | None:
    if not face_box or not crop_box:
        return None
    face_x, face_y, face_w, face_h = [float(value) for value in face_box]
    crop_x, crop_y, crop_w, crop_h = [float(value) for value in crop_box]
    if crop_w <= 0 or crop_h <= 0:
        return None
    left = max((face_x - crop_x) / crop_w, 0.0)
    top = max((face_y - crop_y) / crop_h, 0.0)
    width = min(face_w / crop_w, 1.0)
    height = min(face_h / crop_h, 1.0)
    return {
        "left_pct": round(left * 100.0, 2),
        "top_pct": round(top * 100.0, 2),
        "width_pct": round(width * 100.0, 2),
        "height_pct": round(height * 100.0, 2),
        "target_width": int(target_width),
        "target_height": int(target_height),
    }


def compute_quality(image_bgr: np.ndarray) -> dict[str, Any]:
    quality_image = resize_to_max_side(image_bgr, QUALITY_MAX_SIDE)
    gray = cv2.cvtColor(quality_image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    min_side = int(min(quality_image.shape[:2]))
    reasons = []
    if min_side < 96:
        reasons.append("image_too_small")
    if blur_score < 18:
        reasons.append("image_too_blurry")
    if brightness < 28:
        reasons.append("image_too_dark")
    if brightness > 245:
        reasons.append("image_too_bright")
    severe = min_side < 72 or blur_score < 4 or brightness < 16
    return {"blur_score": blur_score, "brightness": brightness, "min_side": min_side, "quality_reasons": reasons, "quality_pass": not severe}


def face_model_support(face_check: dict[str, Any], threshold: float = CAT_FACE_SUPPORT_THRESHOLD) -> bool:
    if not face_check.get("available"):
        return False
    score = face_check.get("cat_face_score")
    if score is None:
        return False
    return float(score) >= threshold


def effective_detection_from_face_support(
    detection: DetectionResult,
    face_check: dict[str, Any],
    threshold: float = CAT_FACE_SUPPORT_THRESHOLD,
) -> DetectionResult:
    return DetectionResult(
        face_detected=bool(detection.face_detected) or face_model_support(face_check, threshold=threshold),
        face_box=detection.face_box,
        crop_bgr=detection.crop_bgr,
        crop_strategy=detection.crop_strategy,
        crop_box=detection.crop_box,
        human_face_detected=detection.human_face_detected,
        detector_backend=detection.detector_backend,
    )


def resolve_face_detection_source(detection: DetectionResult, face_check: dict[str, Any]) -> str:
    if detection.detector_backend == "yolo" and face_model_support(face_check):
        return "yolo+face_model"
    if detection.detector_backend == "yolo":
        return "yolo"
    if detection.detector_backend == "haar" and face_model_support(face_check):
        return "haar+face_model"
    if detection.detector_backend == "haar":
        return "haar"
    if face_model_support(face_check):
        return "face_model"
    return "none"


def build_top_candidates(
    index_records: list[dict[str, Any]],
    similarities: np.ndarray,
    *,
    top_k: int = TOP_K,
    query_cat_like: bool = False,
    exclude_file_path: str | None = None,
) -> list[dict[str, Any]]:
    ranked_indices = np.argsort(similarities)[::-1]
    top_candidates = []
    for idx in ranked_indices:
        record = index_records[int(idx)]
        if exclude_file_path and record["file_path"] == exclude_file_path:
            continue
        if query_cat_like and record["group"] == "not_cat":
            continue
        top_candidates.append({
            "rank": len(top_candidates) + 1,
            "label": record["label"],
            "group": record["group"],
            "score": round(float(similarities[int(idx)]), 6),
            "cat_id": record["cat_id"],
            "image_name": record["image_name"],
            "file_path": record["file_path"],
        })
        if len(top_candidates) >= min(top_k, len(similarities)):
            break
    return top_candidates


def build_score_summary(
    index_records: list[dict[str, Any]],
    similarities: np.ndarray,
    *,
    face_check: dict[str, Any] | None = None,
    exact_known_label: str | None = None,
    query_cat_like: bool = False,
    exclude_file_path: str | None = None,
) -> dict[str, Any]:
    known_candidates = []
    best_unknown = None
    best_not_cat = None
    for idx, record in enumerate(index_records):
        if exclude_file_path and record["file_path"] == exclude_file_path:
            continue
        if query_cat_like and record["group"] == "not_cat":
            continue
        score = float(similarities[idx])
        if record["group"] == "known":
            known_candidates.append({
                "label": record["label"],
                "group": record["group"],
                "score": score,
                "cat_id": record["cat_id"],
                "image_name": record["image_name"],
                "file_path": record["file_path"],
            })
        elif record["group"] == "unknown_cat":
            best_unknown = score if best_unknown is None else max(best_unknown, score)
        elif record["group"] == "not_cat":
            best_not_cat = score if best_not_cat is None else max(best_not_cat, score)
    known_sorted = sorted(known_candidates, key=lambda row: row["score"], reverse=True)
    best_known = known_sorted[0] if known_sorted else None
    second_known = known_sorted[1] if len(known_sorted) > 1 else None
    return {
        "exact_known_label": exact_known_label,
        "best_known_label": best_known["label"] if best_known else None,
        "best_known_score": float(best_known["score"]) if best_known else 0.0,
        "second_known_score": float(second_known["score"]) if second_known else 0.0,
        "best_unknown_score": float(best_unknown) if best_unknown is not None else 0.0,
        "best_not_cat_score": float(best_not_cat) if best_not_cat is not None else 0.0,
        "cat_face_model_available": bool(face_check.get("available")) if face_check else False,
        "cat_face_score": float(face_check["cat_face_score"]) if face_check and face_check.get("cat_face_score") is not None else None,
        "cat_face_model_non_face_score": (
            float(face_check["not_cat_face_score"])
            if face_check and face_check.get("not_cat_face_score") is not None
            else None
        ),
        "query_cat_like": query_cat_like,
    }


def summarize_classes(top_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in top_candidates:
        item = stats.setdefault(row["label"], {"label": row["label"], "group": row["group"], "votes": 0, "weighted_sum": 0.0, "avg_score": 0.0, "best_score": -1.0})
        item["votes"] += 1
        item["weighted_sum"] += float(row["score"])
        item["best_score"] = max(float(row["score"]), item["best_score"])
    for item in stats.values():
        item["avg_score"] = item["weighted_sum"] / max(item["votes"], 1)
    ordered = sorted(stats.values(), key=lambda row: (row["votes"], row["weighted_sum"], row["best_score"]), reverse=True)
    for item in ordered:
        item["weighted_sum"] = round(float(item["weighted_sum"]), 6)
        item["avg_score"] = round(float(item["avg_score"]), 6)
        item["best_score"] = round(float(item["best_score"]), 6)
    return ordered


def effective_vote_candidates(top_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not top_candidates:
        return []
    best_score = max(float(row["score"]) for row in top_candidates)
    cutoff = best_score - VOTE_RELATIVE_MARGIN
    filtered = [row for row in top_candidates if float(row["score"]) >= cutoff]
    return filtered


def extract_group_decision_features(
    top_candidates: list[dict[str, Any]],
    vote_candidates: list[dict[str, Any]],
    detection: DetectionResult,
    quality: dict[str, Any],
    face_check: dict[str, Any],
    score_summary: dict[str, Any],
) -> dict[str, float]:
    top_class_summary = summarize_classes(vote_candidates)
    winner = top_class_summary[0] if top_class_summary else {"label": None, "group": None, "votes": 0, "weighted_sum": 0.0, "avg_score": 0.0, "best_score": 0.0}
    runner = top_class_summary[1] if len(top_class_summary) > 1 else {"label": None, "group": None, "votes": 0, "weighted_sum": 0.0, "avg_score": 0.0, "best_score": 0.0}
    top1 = top_candidates[0] if top_candidates else None
    best_known = float(score_summary.get("best_known_score") or 0.0)
    second_known = float(score_summary.get("second_known_score") or 0.0)
    best_unknown = float(score_summary.get("best_unknown_score") or 0.0)
    best_not_cat = float(score_summary.get("best_not_cat_score") or 0.0)
    strongest_special = max(best_unknown, best_not_cat)
    cat_face_score_raw = face_check.get("cat_face_score")
    cat_face_non_face_raw = face_check.get("not_cat_face_score")
    cat_face_score = float(cat_face_score_raw) if cat_face_score_raw is not None else 0.0
    cat_face_non_face_score = float(cat_face_non_face_raw) if cat_face_non_face_raw is not None else 0.0
    weighted_margin = float(winner.get("weighted_sum", 0.0)) - float(runner.get("weighted_sum", 0.0))
    vote_margin = int(winner.get("votes", 0)) - int(runner.get("votes", 0))

    features = {
        "quality_pass": 1.0 if quality.get("quality_pass") else 0.0,
        "blur_score": float(quality.get("blur_score") or 0.0),
        "brightness": float(quality.get("brightness") or 0.0),
        "min_side": float(quality.get("min_side") or 0.0),
        "face_detected": 1.0 if detection.face_detected else 0.0,
        "localized_face_detected": 1.0 if detection.face_box else 0.0,
        "human_face_detected": 1.0 if detection.human_face_detected else 0.0,
        "cat_face_score": cat_face_score,
        "cat_face_non_face_score": cat_face_non_face_score,
        "query_cat_like": 1.0 if score_summary.get("query_cat_like") else 0.0,
        "best_known_score": best_known,
        "second_known_score": second_known,
        "best_unknown_score": best_unknown,
        "best_not_cat_score": best_not_cat,
        "best_special_score": strongest_special,
        "known_vs_unknown_margin": best_known - best_unknown,
        "known_vs_not_cat_margin": best_known - best_not_cat,
        "unknown_vs_not_cat_margin": best_unknown - best_not_cat,
        "winner_votes": float(winner.get("votes", 0)),
        "runner_votes": float(runner.get("votes", 0)),
        "vote_margin": float(vote_margin),
        "winner_weighted_sum": float(winner.get("weighted_sum", 0.0)),
        "runner_weighted_sum": float(runner.get("weighted_sum", 0.0)),
        "weighted_margin": float(weighted_margin),
        "winner_avg_score": float(winner.get("avg_score", 0.0)),
        "winner_best_score": float(winner.get("best_score", 0.0)),
        "top1_score": float(top1.get("score", 0.0)) if top1 else 0.0,
        "top_candidate_count": float(len(top_candidates)),
        "effective_vote_count": float(len(vote_candidates)),
        "top1_is_known": 1.0 if top1 and top1.get("group") == "known" else 0.0,
        "top1_is_unknown": 1.0 if top1 and top1.get("group") == "unknown_cat" else 0.0,
        "top1_is_not_cat": 1.0 if top1 and top1.get("group") == "not_cat" else 0.0,
        "winner_is_known": 1.0 if winner.get("group") == "known" else 0.0,
        "winner_is_unknown": 1.0 if winner.get("group") == "unknown_cat" else 0.0,
        "winner_is_not_cat": 1.0 if winner.get("group") == "not_cat" else 0.0,
    }
    return features


def apply_open_set_calibrator(
    calibrator: OpenSetCalibrator | None,
    *,
    current_decision: dict[str, Any],
    top_candidates: list[dict[str, Any]],
    quality: dict[str, Any],
    detection: DetectionResult,
    face_check: dict[str, Any],
    score_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if calibrator is None:
        return current_decision, {"available": False}
    vote_candidates = effective_vote_candidates(top_candidates)
    features = extract_group_decision_features(top_candidates, vote_candidates, detection, quality, face_check, score_summary)
    calibration = calibrator.predict(features)
    if (
        not calibration.get("available")
        or not calibration.get("confident")
        or current_decision.get("decision_type") in {"known_exact_hash", "quality_gate"}
    ):
        return current_decision, calibration

    class_summary = summarize_classes(vote_candidates)
    winner = class_summary[0] if class_summary else {"label": None, "votes": 0, "weighted_sum": 0.0, "avg_score": 0.0, "best_score": 0.0, "group": None}
    runner = class_summary[1] if len(class_summary) > 1 else {"label": None, "votes": 0, "weighted_sum": 0.0, "avg_score": 0.0, "best_score": 0.0, "group": None}
    predicted_group = calibration.get("predicted_group")
    current_is_known = current_decision.get("final_label") not in {"unknown_cat", "not_cat", "low_quality"}
    strong_known_majority = (
        winner.get("group") == "known"
        and int(winner.get("votes", 0)) >= 6
        and float(score_summary.get("best_known_score") or 0.0) >= 0.98
    )
    if predicted_group == "unknown_cat":
        if current_is_known and strong_known_majority:
            return current_decision, calibration
        return build_decision_payload("unknown_cat", "แมวที่ไม่อยู่ในระบบ", "open-set calibrator เอนเอียงไปที่ unknown_cat", "calibrator_unknown", winner, runner), calibration
    if predicted_group == "not_cat":
        non_face_score = float(score_summary.get("cat_face_model_non_face_score") or 0.0)
        if current_is_known and detection.face_detected and non_face_score < 0.98:
            return current_decision, calibration
        return build_decision_payload("not_cat", "ไม่ใช่แมว", "open-set calibrator เอนเอียงไปที่ not_cat", "calibrator_not_cat", winner, runner), calibration
    return current_decision, calibration


def build_decision_payload(final_label: str, display_label: str, reason: str, decision_type: str, winner: dict[str, Any], runner: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_label": final_label,
        "final_label_display": display_label,
        "reason": reason,
        "decision_type": decision_type,
        "winner_label": winner.get("label"),
        "winner_votes": int(winner.get("votes", 0)),
        "winner_weighted_sum": round(float(winner.get("weighted_sum", 0.0)), 6),
        "winner_avg_score": round(float(winner.get("avg_score", 0.0)), 6),
        "runner_up_label": runner.get("label"),
        "runner_up_votes": int(runner.get("votes", 0)),
        "runner_up_weighted_sum": round(float(runner.get("weighted_sum", 0.0)), 6),
    }


def decide_prediction(top_candidates: list[dict[str, Any]], quality: dict[str, Any], detection: DetectionResult, score_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    vote_candidates = effective_vote_candidates(top_candidates)
    class_summary = summarize_classes(vote_candidates)
    winner = class_summary[0] if class_summary else {"label": "unknown_cat", "votes": 0, "weighted_sum": 0.0, "avg_score": 0.0, "group": "unknown_cat"}
    runner = class_summary[1] if len(class_summary) > 1 else {"label": None, "votes": 0, "weighted_sum": 0.0, "avg_score": 0.0, "group": None}
    score_summary = score_summary or {}
    top1 = top_candidates[0] if top_candidates else None
    best_known = float(score_summary.get("best_known_score") or max((row["score"] for row in top_candidates if row["group"] == "known"), default=0.0))
    best_known_label = score_summary.get("best_known_label")
    best_unknown = float(score_summary.get("best_unknown_score") or max((row["score"] for row in top_candidates if row["group"] == "unknown_cat"), default=0.0))
    best_not_cat = float(score_summary.get("best_not_cat_score") or max((row["score"] for row in top_candidates if row["group"] == "not_cat"), default=0.0))
    cat_face_model_available = bool(score_summary.get("cat_face_model_available"))
    cat_face_score_raw = score_summary.get("cat_face_score")
    cat_face_score = float(cat_face_score_raw) if cat_face_score_raw is not None else None
    strong_non_cat_signal = cat_face_model_available and cat_face_score is not None and cat_face_score <= STRONG_NON_CAT_FACE_SCORE
    weighted_margin = float(winner["weighted_sum"]) - float(runner["weighted_sum"])
    vote_margin = int(winner["votes"]) - int(runner["votes"])

    exact_known_label = score_summary.get("exact_known_label")
    if exact_known_label:
        return build_decision_payload(str(exact_known_label), str(exact_known_label), "รูปตรงกับรูปแมวในฐานข้อมูลแบบ exact hash", "known_exact_hash", winner, runner)

    if not quality["quality_pass"]:
        return build_decision_payload("low_quality", "คุณภาพภาพไม่ผ่าน", "คุณภาพภาพต่ำเกินเกณฑ์ที่กำหนด", "quality_gate", winner, runner)

    strongest_special = max(best_unknown, best_not_cat)

    strong_not_cat_without_face = (
        not detection.face_detected
        and strong_non_cat_signal
        and winner["label"] == "not_cat"
        and winner["votes"] >= 3
        and best_not_cat >= max(best_known, NOT_CAT_NO_FACE_MIN_SCORE)
    )
    top1_not_cat_without_face = (
        not detection.face_detected
        and strong_non_cat_signal
        and bool(top1)
        and top1["label"] == "not_cat"
        and best_not_cat >= max(best_known, NOT_CAT_NO_FACE_MIN_SCORE)
    )
    if strong_not_cat_without_face or top1_not_cat_without_face:
        reason = "ไม่พบหน้าแมว และโมเดลหน้าแมวให้คะแนนต่ำมาก ขณะเดียวกัน not_cat เด่นในผลค้นหา"
        if detection.human_face_detected:
            reason += " รวมถึงพบใบหน้าคนในภาพ"
        return build_decision_payload("not_cat", "ไม่ใช่แมว", reason, "strong_not_cat_no_face", winner, runner)

    if winner["group"] == "known":
        winner_best = float(winner.get("best_score", 0.0))
        score_known_majority = (
            winner["votes"] >= 2
            and float(winner["avg_score"]) >= 0.78
            and winner_best >= 0.82
            and best_not_cat <= winner_best + SPECIAL_OVERRIDE_MARGIN
            and best_unknown <= winner_best + SPECIAL_OVERRIDE_MARGIN
            and (vote_margin >= 1 or weighted_margin >= 0.01 or winner_best >= KNOWN_HIGH_CONFIDENCE_SCORE)
        )
        dominant_known_majority = (
            winner["votes"] >= KNOWN_DOMINANT_VOTES
            and vote_margin >= KNOWN_DOMINANT_VOTE_MARGIN
            and float(winner["avg_score"]) >= KNOWN_DOMINANT_AVG_SCORE
            and winner_best >= KNOWN_DOMINANT_BEST_SCORE
            and strongest_special <= min(winner_best + KNOWN_DOMINANT_SPECIAL_MARGIN, KNOWN_DOMINANT_MAX_SPECIAL_SCORE)
        )
        known_majority_confident = score_known_majority or dominant_known_majority
        if known_majority_confident:
            reason = "ชนะ majority vote ของ known class และคะแนนผ่านเกณฑ์"
            decision_type = "known_majority"
            if dominant_known_majority and not score_known_majority:
                reason = "known class ชนะ majority vote ขาดมาก จึงยืนยันเป็นแมวในระบบ"
                decision_type = "known_dominant_majority"
            if not detection.face_detected:
                reason = "ไม่พบหน้าแมวชัด แต่ majority vote ของ known class ชัดเจน"
            return build_decision_payload(str(winner["label"]), str(winner["label"]), reason, decision_type, winner, runner)

    strong_known_match = bool(best_known_label) and best_known >= KNOWN_HIGH_CONFIDENCE_SCORE and best_known + SPECIAL_OVERRIDE_MARGIN >= strongest_special
    if strong_known_match and (detection.face_detected or best_known >= 0.98):
        reason = "known score สูงมากและไม่แพ้ special class เกิน margin"
        if not detection.face_detected:
            reason = "ไม่พบหน้าแมวชัด แต่ known score สูงมากและใกล้/ชนะ special class"
        return build_decision_payload(str(best_known_label), str(best_known_label), reason, "known_high_score", winner, runner)

    if detection.face_detected:
        if winner["label"] == "not_cat" and winner["votes"] >= 4 and best_not_cat >= max(best_known + SPECIAL_OVERRIDE_MARGIN, best_unknown, 0.90):
            return build_decision_payload("not_cat", "ไม่ใช่แมว", "Top 10 เอนเอียงไปที่คลาส not_cat", "special_majority", winner, runner)
        if winner["label"] == "unknown_cat" and winner["votes"] >= 4 and best_unknown >= max(best_known + SPECIAL_OVERRIDE_MARGIN, best_not_cat, 0.88):
            return build_decision_payload("unknown_cat", "แมวที่ไม่อยู่ในระบบ", "Top 10 เอนเอียงไปที่คลาส unknown_cat", "special_majority", winner, runner)
        if winner["group"] == "known":
            return build_decision_payload("unknown_cat", "แมวที่ไม่อยู่ในระบบ", "มีแมวในระบบใกล้เคียง แต่คะแนนยังไม่มั่นใจพอ", "fallback_unknown", winner, runner)
    if winner["label"] == "not_cat" and winner["votes"] >= 4 and best_not_cat >= max(best_known + SPECIAL_OVERRIDE_MARGIN, best_unknown + 0.02, 0.90):
        reason = "ไม่พบหน้าแมว และคะแนน not_cat เด่นกว่าคลาสอื่น"
        if detection.human_face_detected:
            reason += " รวมถึงพบใบหน้าคนในภาพ"
        return build_decision_payload("not_cat", "ไม่ใช่แมว", reason, "no_face_not_cat", winner, runner)
    if winner["group"] == "known":
        dominant_known_no_face = (
            winner["votes"] >= 7
            and float(winner["avg_score"]) >= KNOWN_NO_FACE_AVG_SCORE
            and best_known >= KNOWN_NO_FACE_SCORE
            and best_known >= strongest_special + KNOWN_NO_FACE_SPECIAL_MARGIN
        )
        if dominant_known_no_face:
            return build_decision_payload(str(winner["label"]), str(winner["label"]), "ไม่พบหน้าแมวชัด แต่ Top 10 เป็น known เกือบทั้งหมดและ special score ต่ำกว่าชัดเจน", "known_no_face_majority", winner, runner)
        strong_known = winner["votes"] >= 3 and float(winner["avg_score"]) >= 0.84 and best_known >= 0.90 and (vote_margin >= 2 or weighted_margin >= 0.05)
        if strong_known:
            return build_decision_payload(str(winner["label"]), str(winner["label"]), "ไม่พบหน้าแมวชัด แต่ผล known class เด่นมากพอให้ยืนยัน", "known_without_face", winner, runner)
    return build_decision_payload("unknown_cat", "แมวที่ไม่อยู่ในระบบ", "ไม่พบหน้าแมวชัดเจน หรือคะแนนยังไม่ชัดพอสำหรับการยืนยัน", "no_face_unknown", winner, runner)


def create_app(role: str = "both") -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "super_secret_cat_key"
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    service = PredictorService()
    lock = threading.RLock()

    def admin_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if role == "user":
                abort(404)
            if not session.get("admin_logged_in"):
                if request.path.startswith("/api/"):
                    return jsonify({"message": "Unauthorized"}), 401
                return redirect(url_for("login_page"))
            return f(*args, **kwargs)
        return decorated_function

    @app.context_processor
    def inject_globals():
        return {
            "asset_version": ASSET_VERSION,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "role": role
        }

    @app.errorhandler(RequestEntityTooLarge)
    def payload_too_large(_exc):
        return jsonify({"message": f"ไฟล์ที่ส่งมีขนาดรวมเกิน {MAX_UPLOAD_BYTES // (1024 * 1024)} MB กรุณาลดจำนวนรูปหรือย่อขนาดรูปก่อนบันทึก"}), 413

    @app.get("/")
    def index():
        return redirect(url_for("predict_page"))

    @app.get("/login")
    def login_page():
        if role == "user":
            abort(404)
        return render_template("login.html", active_page="login", page_title="Admin Login")

    @app.post("/api/login")
    def api_login():
        if role == "user":
            abort(404)
        payload = request.get_json(silent=True) or {}
        if payload.get("password") == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return jsonify({"success": True})
        return jsonify({"message": "Invalid password"}), 401

    @app.post("/api/logout")
    def api_logout():
        if role == "user":
            abort(404)
        session.pop("admin_logged_in", None)
        return jsonify({"success": True})

    @app.get("/profile/<int:cat_id>")
    def profile_page(cat_id: int):
        with lock:
            cat = service.find_cat(cat_id)
            if not cat:
                abort(404)
            return render_template("profile.html", cat=cat, active_page="profile", page_title=f"โปรไฟล์: {cat['name']}")

    @app.get("/cats")
    @admin_required
    def cats():
        return render_template("cats.html", active_page="cats", page_title="Cats")

    @app.get("/not-cat")
    @admin_required
    def not_cat_page():
        return render_template("reference.html", active_page="not_cat", page_title="Not cat", reference_key="not_cat", page_heading="Not cat", page_copy="คลาสอ้างอิงสำหรับรูปที่ไม่ใช่แมว")

    @app.get("/unknown-cat")
    @admin_required
    def unknown_cat_page():
        return render_template("reference.html", active_page="unknown_cat", page_title="Unknown_cat", reference_key="unknown_cat", page_heading="Unknown_cat", page_copy="คลาสอ้างอิงสำหรับแมวที่ไม่อยู่ในฐานข้อมูล")

    @app.get("/predict")
    def predict_page():
        if role == "admin" and not session.get("admin_logged_in"):
            return redirect(url_for("login_page"))
        return render_template("predict.html", active_page="predict", page_title="Predict")

    @app.get("/uploads/<path:file_path>")
    def uploaded_file(file_path: str):
        # Allow uploads access for both, but maybe protect it on admin port? 
        # Usually fine to leave as is for serving images.
        target = (BASE_DIR / file_path).resolve()
        if BASE_DIR.resolve() not in target.parents or not target.is_file():
            abort(404)
        return send_from_directory(target.parent, target.name)

    @app.get("/api/status")
    def api_status():
        with lock:
            service.sync_reference_dirs()
            return jsonify(service.summary())

    @app.get("/api/cats")
    @admin_required
    def api_cats():
        with lock:
            service.sync_reference_dirs()
            payload = [service.serialize_cat(cat) for cat in service.state["cats"]]
            return jsonify({"cats": payload, "summary": service.summary()})

    @app.post("/api/cats")
    @admin_required
    def api_create_cat():
        try:
            with lock:
                cat = service.create_cat(request.form, request.files.getlist("images"))
                return jsonify({"cat": service.serialize_cat(cat), "summary": service.summary()}), 201
        except DuplicateImageError as exc:
            return jsonify({"message": str(exc), "duplicate_images": exc.duplicates}), 400
        except ValueError as exc:
            return jsonify({"message": str(exc)}), 400

    @app.put("/api/cats/<int:cat_id>")
    @admin_required
    def api_update_cat(cat_id: int):
        try:
            with lock:
                cat = service.update_cat(cat_id, request.form, request.files.getlist("images"))
                return jsonify({"cat": service.serialize_cat(cat), "summary": service.summary()})
        except KeyError as exc:
            return jsonify({"message": exc.args[0]}), 404
        except DuplicateImageError as exc:
            return jsonify({"message": str(exc), "duplicate_images": exc.duplicates}), 400
        except ValueError as exc:
            return jsonify({"message": str(exc)}), 400

    @app.delete("/api/cats/<int:cat_id>")
    @admin_required
    def api_delete_cat(cat_id: int):
        try:
            with lock:
                service.delete_cat(cat_id)
                return jsonify({"deleted": True, "summary": service.summary()})
        except KeyError as exc:
            return jsonify({"message": exc.args[0]}), 404

    @app.delete("/api/cats/<int:cat_id>/images/<image_name>")
    @admin_required
    def api_delete_cat_image(cat_id: int, image_name: str):
        try:
            with lock:
                service.delete_cat_image(cat_id, image_name)
                return jsonify({"deleted": True, "summary": service.summary()})
        except KeyError as exc:
            return jsonify({"message": exc.args[0]}), 404

    @app.get("/api/reference-sets/<reference_key>")
    @admin_required
    def api_reference_set(reference_key: str):
        with lock:
            service.sync_reference_dirs()
            if reference_key not in service.state["reference_sets"]:
                return jsonify({"message": "ไม่พบ reference set"}), 404
            limit = int(request.args.get("limit", 24))
            return jsonify({
                "reference_set": service.serialize_reference_set(reference_key, limit),
                "duplicate_hash_index": service.reference_duplicate_hash_index(),
                "summary": service.summary(),
            })

    @app.post("/api/reference-sets/<reference_key>/images")
    @admin_required
    def api_upload_reference_images(reference_key: str):
        try:
            with lock:
                service.upload_reference_images(reference_key, request.files.getlist("images"))
                return jsonify({"reference_set": service.serialize_reference_set(reference_key), "summary": service.summary()})
        except KeyError as exc:
            return jsonify({"message": exc.args[0]}), 404
        except DuplicateImageError as exc:
            return jsonify({"message": str(exc), "duplicate_images": exc.duplicates}), 400
        except ValueError as exc:
            return jsonify({"message": str(exc)}), 400

    @app.delete("/api/reference-sets/<reference_key>/images/<image_name>")
    @admin_required
    def api_delete_reference_image(reference_key: str, image_name: str):
        try:
            with lock:
                service.delete_reference_image(reference_key, image_name)
                return jsonify({"deleted": True, "summary": service.summary()})
        except KeyError as exc:
            return jsonify({"message": exc.args[0]}), 404

    @app.post("/api/predict")
    def api_predict():
        try:
            if request.files.get("file"):
                image_bytes = request.files["file"].read()
            elif request.is_json:
                payload = request.get_json(silent=True) or {}
                raw = payload.get("image_base64")
                if not isinstance(raw, str) or "," not in raw:
                    return jsonify({"message": "รูปแบบ image_base64 ไม่ถูกต้อง"}), 400
                image_bytes = base64.b64decode(raw.split(",", 1)[1], validate=False)
            else:
                return jsonify({"message": "กรุณาส่ง file หรือ image_base64"}), 400
            with lock:
                return jsonify(service.predict(image_bytes))
        except ValueError as exc:
            return jsonify({"message": str(exc)}), 400

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Cat Identity App")
    parser.add_argument("--port", type=int, default=5056, help="Port to run the app on")
    parser.add_argument("--role", choices=["user", "admin", "both"], default="both", help="Role of this instance (restricts routes)")
    args = parser.parse_args()

    app = create_app(role=args.role)
    app.run(port=args.port, debug=True, use_reloader=False)
