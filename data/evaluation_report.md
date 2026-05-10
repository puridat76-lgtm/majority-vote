# Predictor Evaluation Report

## Dataset

| Split | Images |
| --- | ---: |
| Known | 121 |
| Unknown cat | 17 |
| Not cat | 21 |
| Total | 159 |

| Known label | Images |
| --- | ---: |
| ไคลด์ | 30 |
| จูลส์ | 30 |
| โคลอี้ | 30 |
| จูลี่ | 31 |

## Current Thresholds

| Key | Value |
| --- | ---: |
| `top_k` | `10` |
| `vote_relative_margin` | `0.12` |
| `cat_face_support_threshold` | `0.95` |
| `cat_like_query_threshold` | `1.01` |
| `strong_non_cat_face_score` | `0.01` |
| `not_cat_no_face_min_score` | `0.9` |
| `known_high_confidence_score` | `0.95` |
| `known_no_face_score` | `0.82` |
| `known_no_face_avg_score` | `0.78` |
| `known_no_face_special_margin` | `0.08` |
| `known_dominant_votes` | `8` |
| `known_dominant_vote_margin` | `5` |
| `known_dominant_avg_score` | `0.7` |
| `known_dominant_best_score` | `0.72` |
| `known_dominant_special_margin` | `0.1` |
| `known_dominant_max_special_score` | `0.88` |
| `special_override_margin` | `0.0` |

## Accuracy Summary

| Config | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| current | 0.9245 | 0.9669 | 0.8824 | 0.7143 |

## Top K Sweep

| Config | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| top_10 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| top_5 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| top_3 | 0.9308 | 0.9669 | 0.9412 | 0.7143 |

## Vote Margin Sweep

| Config | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| margin_0.08 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| margin_0.10 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| margin_0.12 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| margin_0.14 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| margin_0.16 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |

## Face Support Threshold Sweep

| Config | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| face_thr_0.85 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| face_thr_0.90 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| face_thr_0.95 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| face_thr_0.97 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |
| face_thr_0.99 | 0.9245 | 0.9669 | 0.8824 | 0.7143 |

## Raw Retrieval

| Metric | Accuracy |
| --- | ---: |
| Known top-1 label | 0.9669 |
| Unknown cat top-1 group | 0.5294 |
| Not cat top-1 group | 0.7619 |

## Before vs After Face Model

| Config | Overall | Known | Unknown cat | Not cat |
| --- | ---: | ---: | ---: | ---: |
| before_face_model | 0.8553 | 0.9669 | 0.7647 | 0.2857 |
| current | 0.9245 | 0.9669 | 0.8824 | 0.7143 |

| Diff | Count |
| --- | ---: |
| Changed predictions | 11 |
| Improved | 11 |
| Regressed | 0 |

## Error Analysis

### known

| Metric | Value |
| --- | --- |
| Error count | `4` |
| Decision types | `{'fallback_unknown': 2, 'known_majority': 1, 'no_face_unknown': 1}` |
| Predicted labels | `{'unknown_cat': 3, 'ไคลด์': 1}` |

| File | Predicted | Decision | Top1 | Face score |
| --- | --- | --- | --- | ---: |
| `uploads/cats/4/477897e6527247e7bbace1ae34bcc9a3.png` | `unknown_cat` | `fallback_unknown` | `จูลส์` | 0.998007 |
| `uploads/cats/4/ed98306bea164e1aa46cf60e9f3d9218.png` | `unknown_cat` | `fallback_unknown` | `not_cat` | 0.212225 |
| `uploads/cats/5/468bb903bd4144749547ced533795293.png` | `ไคลด์` | `known_majority` | `ไคลด์` | 0.978871 |
| `uploads/cats/6/ecfd51c0b9b44f1cad0ee6da95b0630c.png` | `unknown_cat` | `no_face_unknown` | `not_cat` | 0.643710 |

### not_cat

| Metric | Value |
| --- | --- |
| Error count | `6` |
| Decision types | `{'no_face_unknown': 5, 'known_without_face': 1}` |
| Predicted labels | `{'unknown_cat': 5, 'จูลี่': 1}` |

| File | Predicted | Decision | Top1 | Face score |
| --- | --- | --- | --- | ---: |
| `uploads/reference/not_cat/fa418e9970a64fea9c12649a69e6f8d1.jpg` | `unknown_cat` | `no_face_unknown` | `unknown_cat` | 0.000000 |
| `uploads/reference/not_cat/7c39a48e42d346a6954c196551f43b6b.jpeg` | `unknown_cat` | `no_face_unknown` | `unknown_cat` | 0.000000 |
| `uploads/reference/not_cat/6e91cabf60f749b4a32e12be7d00bc52.jpeg` | `unknown_cat` | `no_face_unknown` | `unknown_cat` | 0.000000 |
| `uploads/reference/not_cat/0e3749228d1e44229217e2a3920ede9c.jpeg` | `unknown_cat` | `no_face_unknown` | `จูลส์` | 0.000046 |
| `uploads/reference/not_cat/9c9a621b2eb54eab9735514e0b230efe.jpeg` | `unknown_cat` | `no_face_unknown` | `not_cat` | 0.000003 |
| `uploads/reference/not_cat/8ff6f0d4238846488ec2c9d3c23b9ea3.jpeg` | `จูลี่` | `known_without_face` | `unknown_cat` | 0.000117 |

### unknown_cat

| Metric | Value |
| --- | --- |
| Error count | `2` |
| Decision types | `{'known_majority': 1, 'no_face_not_cat': 1}` |
| Predicted labels | `{'จูลส์': 1, 'not_cat': 1}` |

| File | Predicted | Decision | Top1 | Face score |
| --- | --- | --- | --- | ---: |
| `uploads/reference/unknown_cat/371b39629fe84feda495a1b56d4fe011.jpg` | `จูลส์` | `known_majority` | `จูลส์` | 0.998589 |
| `uploads/reference/unknown_cat/images (8).jpg` | `not_cat` | `no_face_not_cat` | `not_cat` | 0.900527 |
