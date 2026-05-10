# Six-Point Improvement Report

Date: 2026-04-26

## Inputs

- Retrain dataset: `data/imports/cat_retrain_dataset_zip`
- Detector bundle: `data/imports/cat_face_detector_zip`

## 1. Dataset Audit

Prepared dataset:

- train: known `693`, unknown_cat `35`, not_cat `35`
- val: known `198`, unknown_cat `10`, not_cat `10`
- test: known `99`, unknown_cat `5`, not_cat `5`

Audit output:

- `data/retrain_dataset_audit.json`

Key findings:

- exact duplicate group: `1`
- quality issues: `2`
- suspicious `not_cat` images where Haar still localizes a face: multiple `IMG_82xx.JPG`
- many weak known/unknown images where no stable localized face was found

## 2. Better Detector

Benchmark script:

- `tools/evaluate_cat_face_detector.py`

Result on prepared `test` split:

| Group | Haar | YOLO |
| --- | ---: | ---: |
| known | `0.5657` | `0.9899` |
| unknown_cat | `1.0000` | `1.0000` |
| not_cat | `0.0000` | `0.0000` |

Status:

- YOLO detector support was added to `app.py`
- live switch is currently `OFF`
- current flag: `ENABLE_YOLO_DETECTOR = False`

Reason:

- YOLO improved face localization but hurt the current open-set behavior in the production-style evaluation

## 3. Two-Stage / Open-Set Decision

Added:

- reusable decision-context helpers in `app.py`
- runtime open-set calibrator support in `app.py`
- prediction response fields for calibrator diagnostics

Model path:

- `data/models/open_set_calibrator.json`

Status:

- implemented
- safe gating added so calibrator can only override into `unknown_cat` or `not_cat`
- known predictions are protected in strong-majority cases

## 4. Retrain Identity Encoder

Training script:

- `tools/train_identity_encoder.py`

Output:

- `data/models/siamese_encoder_retrained.h5`
- `data/models/siamese_encoder_retrained_history.json`

Status:

- retraining completed
- retrained encoder did not beat the current production encoder on the prepared test split

## 5. Hard Negative Mining

Script:

- `tools/mine_hard_negatives.py`

Output:

- `data/hard_negatives.json`

Key finding:

- the biggest remaining errors are known cats that get pushed to `unknown_cat` by the open-set layer

## 6. Validation / Reporting

Scripts:

- `tools/prepare_retrain_dataset.py`
- `tools/evaluate_retrain_dataset.py`
- `tools/train_open_set_calibrator.py`
- `tools/audit_retrain_dataset.py`
- `tools/evaluate_cat_face_detector.py`

Main outputs:

- `data/open_set_calibrator_report.json`
- `data/open_set_calibrator_retrained_report.json`
- `data/retrain_dataset_audit.json`
- `data/hard_negatives.json`

## Evaluation Summary

### Current live production-style evaluation

Command:

```bash
python tools/evaluate_predictor.py --section summary
```

Result:

| Config | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| current | `0.9245` | `0.9669` | `0.8824` | `0.7143` |

### Prepared retrain dataset evaluation

Current encoder + current calibrator:

| Mode | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.6697` | `0.6970` | `0.6000` | `0.2000` |
| calibrated | `0.6697` | `0.6364` | `1.0000` | `1.0000` |

Retrained encoder + retrained calibrator:

| Mode | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.6606` | `0.6869` | `0.6000` | `0.2000` |
| calibrated | `0.6789` | `0.6465` | `1.0000` | `1.0000` |

## Recommendation

Do now:

- keep the current production encoder
- keep YOLO detector support in code, but do not enable it yet
- use the audit report to clean suspicious `not_cat` and weak-face images
- use hard negatives to curate better `unknown_cat` and `not_cat` examples

Do next:

- clean the prepared dataset and retrain again
- retrain the open-set calibrator after dataset cleanup
- only switch production models after the prepared test split beats the current baseline without collapsing known-cat accuracy
