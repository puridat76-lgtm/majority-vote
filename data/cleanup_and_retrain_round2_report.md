# Cleanup And Retrain Round 2

Date: 2026-04-26

## Goal

- build a concrete remove / replace list
- clean only `train/val`
- keep `test` untouched
- retrain and re-evaluate honestly on the original test split

## Remove / Replace List

Primary replace list for `unknown_cat`:

- `train/unknown_cat/34.jpeg`
- `val/unknown_cat/13.jpeg`

Reason:

- no stable localized face
- cat-face score is near zero
- these are weak samples for a face-based pipeline

Review list for `not_cat`:

- `train/not_cat/IMG_8258.JPG`
- `train/not_cat/IMG_8259.JPG`
- `train/not_cat/IMG_8272.JPG`
- `train/not_cat/IMG_8278.JPG`
- `train/not_cat/IMG_8279.JPG`
- `train/not_cat/IMG_8281.JPG`
- `val/not_cat/IMG_8245.JPG`
- `val/not_cat/IMG_8247.JPG`
- `val/not_cat/IMG_8249.JPG`
- `val/not_cat/IMG_8253.JPG`

Reason:

- Haar still localizes a face-like region on these images
- face-model score still says non-face strongly

Important note:

- removing these `not_cat` hard negatives made the baseline `not_cat` result worse
- keep them for now unless you replace them with better non-cat negatives

Known-cat cleanup applied in the experiment:

- exact duplicate removed: `train/known/ดิกซี่/Screenshot 2026-03-11 202701.png`
- quality issues removed: `2` images
- weak known-face images removed from `train/val`: `21` images

Full plans:

- `data/retrain_dataset_cleanup_plan.md`
- `data/retrain_dataset_cleanup_plan_keep_notcat.md`

## Variant Results

### Reference: original prepared dataset

Current encoder + current calibrator:

| Overall | Known | Unknown cat | Not cat |
| ---: | ---: | ---: | ---: |
| `0.6697` | `0.6364` | `1.0000` | `1.0000` |

### Variant A: strict cleaned

Rules:

- remove duplicates
- remove quality issues
- remove weak known / weak unknown
- remove suspicious `not_cat`

Current encoder + cleaned calibrator:

| Overall | Known | Unknown cat | Not cat |
| ---: | ---: | ---: | ---: |
| `0.6697` | `0.6364` | `1.0000` | `1.0000` |

Retrained encoder + retrained cleaned calibrator:

| Overall | Known | Unknown cat | Not cat |
| ---: | ---: | ---: | ---: |
| `0.6881` | `0.6566` | `1.0000` | `1.0000` |

### Variant B: keep `not_cat` hard negatives

Rules:

- remove duplicates
- remove quality issues
- remove weak known / weak unknown
- keep suspicious `not_cat`

Current encoder baseline:

| Overall | Known | Unknown cat | Not cat |
| ---: | ---: | ---: | ---: |
| `0.6697` | `0.6970` | `0.6000` | `0.2000` |

Retrained encoder baseline:

| Overall | Known | Unknown cat | Not cat |
| ---: | ---: | ---: | ---: |
| `0.6606` | `0.6869` | `0.6000` | `0.2000` |

## Conclusion

- the most useful concrete cleanup is replacing the `unknown_cat` weak-face images
- `not_cat` suspicious samples should be reviewed, but not auto-removed yet
- the best experiment in this round is:
  - strict cleaned dataset
  - retrained encoder
  - retrained cleaned calibrator
- it improves overall prepared-test accuracy from `0.6697` to `0.6881`
- this is still not strong enough to replace the production model used on the live `uploads` dataset
