# YOLO Serious Experiment Summary

Date: 2026-04-26

## What Was Actually Tested

This round tested the parts that had not been tuned seriously before:

- YOLO detector in the same end-to-end pipeline
- downstream threshold sweep on the live app dataset
- current encoder + YOLO-specific calibrator
- YOLO-cropped encoder retraining
- YOLO-retrained encoder + YOLO-specific calibrator

## 1. Live App Dataset

This is the dataset the real app currently uses from `uploads/`.

Production-like baseline:

- detector: `haar`
- encoder: `siamese_model3.h5`
- face model: `keras_model.h5`
- calibrator: `open_set_calibrator.json`

Result:

| Config | Overall | Macro | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: | ---: |
| live_haar_current | `0.9560` | `0.9413` | `0.9669` | `1.0000` | `0.8571` |

YOLO with current production behavior:

| Config | Overall | Macro | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: | ---: |
| live_yolo_current | `0.9434` | `0.8722` | `0.9835` | `0.8235` | `0.8095` |

Best YOLO config found from grid search:

- `top_k = 10`
- `vote_relative_margin = 0.10`
- `cat_face_support_threshold = 0.90`
- `cat_like_query_threshold = 1.01`
- `special_override_margin = 0.00`
- `use_group_calibrator = true`

Result:

| Config | Overall | Macro | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: | ---: |
| best_yolo | `0.9434` | `0.8722` | `0.9835` | `0.8235` | `0.8095` |

Conclusion on live dataset:

- YOLO increased `known` accuracy
- but YOLO reduced `unknown_cat` and `not_cat`
- even after threshold/rule tuning, the best YOLO result was still below the Haar production baseline

## 2. Prepared Retrain Dataset

This is the dedicated train/val/test dataset from `cat_retrain_dataset.zip`.

### Haar current

| Mode | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.6697` | `0.6970` | `0.6000` | `0.2000` |
| calibrated | `0.6697` | `0.6364` | `1.0000` | `1.0000` |

### YOLO current

| Mode | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.7523` | `0.7677` | `0.8000` | `0.4000` |
| calibrated | `0.7431` | `0.7172` | `1.0000` | `1.0000` |

### YOLO retrained

Model:

- `data/models/siamese_encoder_retrained_yolo.h5`

Calibrator:

- `data/models/open_set_calibrator_retrained_yolo.json`

| Mode | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.7431` | `0.7677` | `0.6000` | `0.4000` |
| calibrated | `0.7523` | `0.7475` | `0.6000` | `1.0000` |

## Clear Comparison

What improved with YOLO:

- on the prepared dataset, YOLO + tuned downstream beat the Haar-based prepared baseline clearly
- YOLO-cropped retraining improved overall prepared accuracy further

What did not improve enough:

- on the live app dataset, YOLO still lost to the current Haar production pipeline
- the main loss remained in `unknown_cat` and `not_cat`

## Recommendation

Right now:

- keep `haar + current production pipeline` in the real app
- keep YOLO as an experiment path only

Why:

- the live production-style result is still better with Haar:
  - Haar current: `0.9560`
  - best YOLO: `0.9434`

Most important takeaway:

- YOLO is not a dead end
- YOLO does help once the whole pipeline is retuned
- but with the live app dataset today, it is still not better enough to replace the current production detector
