# Retrain Dataset Cleanup Plan

- Source dataset: `/Users/p1650900770/Downloads/cat_majority_vote_ready/data/prepared/cat_retrain_dataset`
- Clean dataset: `/Users/p1650900770/Downloads/cat_majority_vote_ready/data/prepared/cat_retrain_dataset_cleaned`
- Cleaned splits: `train, val`
- Weak known threshold: `0.25`

## Summary

| Metric | Value |
| --- | ---: |
| Total removed files | 35 |
| Removed `train/known` | 16 |
| Removed `train/not_cat` | 6 |
| Removed `train/unknown_cat` | 1 |
| Removed `val/known` | 7 |
| Removed `val/not_cat` | 4 |
| Removed `val/unknown_cat` | 1 |

## Source Counts

```json
{
  "train": {
    "known": 693,
    "unknown_cat": 35,
    "not_cat": 35
  },
  "val": {
    "known": 198,
    "unknown_cat": 10,
    "not_cat": 10
  },
  "test": {
    "known": 99,
    "unknown_cat": 5,
    "not_cat": 5
  }
}
```

## Clean Counts

```json
{
  "train": {
    "known": 677,
    "unknown_cat": 34,
    "not_cat": 29
  },
  "val": {
    "known": 191,
    "unknown_cat": 9,
    "not_cat": 6
  },
  "test": {
    "known": 99,
    "unknown_cat": 5,
    "not_cat": 5
  }
}
```

## Remove / Replace List

- `train/not_cat/IMG_8272.JPG`: suspicious_not_cat
- `train/not_cat/IMG_8259.JPG`: suspicious_not_cat
- `train/not_cat/IMG_8258.JPG`: suspicious_not_cat
- `train/not_cat/IMG_8278.JPG`: suspicious_not_cat
- `train/not_cat/IMG_8279.JPG`: suspicious_not_cat
- `train/not_cat/IMG_8281.JPG`: suspicious_not_cat
- `train/unknown_cat/34.jpeg`: weak_unknown_face
- `train/known/เดรโก/Screenshot 2026-03-11 135411.png`: weak_known_face
- `train/known/พิกซี่/Screenshot 2026-03-11 214042.png`: weak_known_face
- `train/known/เทสซ่า/Screenshot 2026-03-11 131839.png`: weak_known_face
- `train/known/ลุค/Screenshot 2026-03-08 162248.png`: quality_issue
- `train/known/แซนดี้/Screenshot 2026-03-11 150715.png`: weak_known_face
- `train/known/แซนดี้/Screenshot 2026-03-11 145510.png`: weak_known_face
- `train/known/ไคลด์/Screenshot 2026-03-11 164141.png`: weak_known_face
- `train/known/ดิกซี่/Screenshot 2026-03-11 202852.png`: weak_known_face
- `train/known/ดิกซี่/Screenshot 2026-03-11 202701.png`: exact_duplicate
- `train/known/บอนนี่/Screenshot 2026-03-11 161019.png`: weak_known_face
- `train/known/บอนนี่/Screenshot 2026-03-11 161047.png`: weak_known_face
- `train/known/บอนนี่/Screenshot 2026-03-11 161315.png`: weak_known_face
- `train/known/โซอี้/Screenshot 2026-03-11 023818.png`: weak_known_face
- `train/known/โซอี้/Screenshot 2026-03-11 023849.png`: weak_known_face
- `train/known/แฮร์รี่/Screenshot 2026-03-09 191715.png`: weak_known_face
- `train/known/จูลี่/Screenshot 2026-03-11 181423.png`: weak_known_face
- `val/not_cat/IMG_8249.JPG`: suspicious_not_cat
- `val/not_cat/IMG_8247.JPG`: suspicious_not_cat
- `val/not_cat/IMG_8253.JPG`: suspicious_not_cat
- `val/not_cat/IMG_8245.JPG`: suspicious_not_cat
- `val/unknown_cat/13.jpeg`: weak_unknown_face
- `val/known/เทสซ่า/Screenshot 2026-03-11 132038.png`: weak_known_face
- `val/known/ลุค/Screenshot 2026-03-08 165743.png`: weak_known_face
- `val/known/โซอี้/Screenshot 2026-03-11 024032.png`: weak_known_face
- `val/known/โซอี้/Screenshot 2026-03-11 023906.png`: weak_known_face
- `val/known/โซอี้/Screenshot 2026-03-11 024017.png`: weak_known_face
- `val/known/จูลี่/Screenshot 2026-03-11 200854.png`: weak_known_face
- `val/known/จูลี่/Screenshot 2026-03-11 200741.png`: quality_issue
