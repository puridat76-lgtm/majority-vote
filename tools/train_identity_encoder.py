from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as predictor_app
from tools.retrain_support import DATASET_ROOT, PreparedSample, list_samples

os.environ.setdefault("MPLCONFIGDIR", str(predictor_app.DATA_DIR / "matplotlib"))


def load_known_split(
    split: str,
    cropper: predictor_app.FaceCropper,
    label_to_index: dict[str, int],
    dataset_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    samples = [sample for sample in list_samples(dataset_root, splits=(split,)) if sample.group == "known"]
    images = []
    labels = []
    for sample in samples:
        image = predictor_app.read_image(sample.file_path)
        if image is None:
            continue
        detection = cropper.detect_and_crop(image)
        fitted = predictor_app.fit_image_to_canvas(detection.crop_bgr, predictor_app.IMAGE_SIZE, predictor_app.IMAGE_SIZE)
        rgb = predictor_app.cv2.cvtColor(fitted, predictor_app.cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        images.append(rgb)
        labels.append(label_to_index[sample.label])
    x = np.asarray(images, dtype="float32")
    y = np.asarray(labels, dtype="int32")
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune identity encoder on prepared known-cat dataset")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--base-model", default=str(predictor_app.MODEL_PATH))
    parser.add_argument("--output", default=str(ROOT_DIR / "data" / "models" / "siamese_encoder_retrained.h5"))
    parser.add_argument("--history", default=str(ROOT_DIR / "data" / "models" / "siamese_encoder_retrained_history.json"))
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--detector-mode", choices=("haar", "yolo"), default="haar")
    args = parser.parse_args()

    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore

    class L2Norm(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.math.l2_normalize(inputs, axis=-1)

    dataset_root = Path(args.dataset_root)
    train_known = [sample for sample in list_samples(dataset_root, splits=("train",)) if sample.group == "known"]
    labels = sorted({sample.label for sample in train_known})
    label_to_index = {label: index for index, label in enumerate(labels)}

    face_classifier = predictor_app.CatFaceBinaryClassifier(
        predictor_app.CAT_FACE_MODEL_CANDIDATES,
        predictor_app.CAT_FACE_LABELS_CANDIDATES,
    )
    cropper = predictor_app.FaceCropper(detector_mode=args.detector_mode, face_classifier=face_classifier)
    x_train, y_train = load_known_split("train", cropper, label_to_index, dataset_root)
    x_val, y_val = load_known_split("val", cropper, label_to_index, dataset_root)
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(labels))
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(labels))

    encoder = load_model(Path(args.base_model), compile=False, custom_objects={"L2Norm": L2Norm})
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.08),
        ],
        name="augment",
    )
    inputs = tf.keras.Input(shape=(predictor_app.IMAGE_SIZE, predictor_app.IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    embeddings = encoder(x)
    x = tf.keras.layers.Dropout(0.2, name="identity_dropout")(embeddings)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax", name="identity_head")(x)
    train_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_encoder_finetune")

    encoder.trainable = False
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]
    head_history = train_model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=args.head_epochs,
        batch_size=32,
        verbose=2,
        callbacks=callbacks,
    )

    encoder.trainable = True
    for layer in encoder.layers[:2]:
        layer.trainable = False
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    finetune_history = train_model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=args.finetune_epochs,
        batch_size=32,
        verbose=2,
        callbacks=callbacks,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(output_path, include_optimizer=False)

    history_payload = {
        "labels": labels,
        "train_shape": list(x_train.shape),
        "val_shape": list(x_val.shape),
        "head_history": head_history.history,
        "finetune_history": finetune_history.history,
    }
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "history": str(history_path),
        "labels": len(labels),
        "train_images": int(x_train.shape[0]),
        "val_images": int(x_val.shape[0]),
        "best_val_accuracy": max(
            max(head_history.history.get("val_accuracy", [0.0]) or [0.0]),
            max(finetune_history.history.get("val_accuracy", [0.0]) or [0.0]),
        ),
        "detector_mode": args.detector_mode,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
