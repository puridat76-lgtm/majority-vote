# API Contract

หน้าเว็บชุดนี้ใช้ endpoint หลักตามนี้

## 1. Global status

### `GET /api/status`

response:

```json
{
  "index_status": "ready",
  "actual_total_images": 12,
  "indexed_total_images": 12,
  "dataset_images": 8,
  "not_cat_folder_images": 2,
  "unknown_cat_folder_images": 2,
  "known_labels": ["Mimi", "Toby"]
}
```

ค่า `index_status` ที่ JS ใช้ตอนนี้มี:

- `ready`
- `needs_train`
- `empty`

## 2. Cats dataset

### `GET /api/cats`

response:

```json
{
  "cats": [
    {
      "id": 1,
      "name": "Mimi",
      "owner": "Demo Owner",
      "contact": "081-234-5678",
      "location": "Bangkok",
      "image_count": 2,
      "cover_image": "/uploads/cats/1/abc.png",
      "images": [
        {
          "name": "abc.png",
          "url": "/uploads/cats/1/abc.png"
        }
      ]
    }
  ],
  "summary": {
    "index_status": "needs_train",
    "actual_total_images": 12,
    "indexed_total_images": 10
  }
}
```

### `POST /api/cats`

form-data:

- `name`
- `owner`
- `contact`
- `location`
- `images` multiple files

### `PUT /api/cats/<cat_id>`

form-data เหมือน `POST /api/cats`

### `DELETE /api/cats/<cat_id>`

### `DELETE /api/cats/<cat_id>/images/<image_name>`

## 3. Reference sets

หน้า `Not cat` และ `Unknown_cat` ใช้ key เหล่านี้

- `not_cat`
- `unknown_cat`

### `GET /api/reference-sets/<reference_key>?limit=24`

response:

```json
{
  "reference_set": {
    "key": "not_cat",
    "label": "ไม่ใช่แมว",
    "image_count": 25,
    "hidden_count": 1,
    "images": [
      {
        "name": "sample.png",
        "url": "/uploads/reference/not_cat/sample.png"
      }
    ]
  },
  "summary": {
    "index_status": "needs_train",
    "actual_total_images": 12,
    "indexed_total_images": 10
  }
}
```

### `POST /api/reference-sets/<reference_key>/images`

form-data:

- `images` multiple files

### `DELETE /api/reference-sets/<reference_key>/images/<image_name>`

## 4. Predict

### `POST /api/predict`

รองรับ 2 แบบ

1. form-data

- `file`

2. json

```json
{
  "image_base64": "data:image/jpeg;base64,..."
}
```

response เมื่อทายเป็น known cat:

```json
{
  "final_label": "Mimi",
  "best_known_name": "Mimi",
  "best_known_score": 0.9642,
  "second_known_score": 0.9181,
  "best_unknown_score": 0.8224,
  "best_not_cat_score": 0.8013,
  "quality_pass": true,
  "quality_reasons": [],
  "blur_score": 66.54,
  "brightness": 122.18,
  "top_matches": [
    {"label": "Mimi", "score": 0.9642}
  ],
  "matched_cat": {
    "id": 1,
    "name": "Mimi",
    "owner": "Demo Owner",
    "contact": "081-234-5678",
    "location": "Bangkok",
    "image_count": 2,
    "cover_image": "/uploads/cats/1/abc.png",
    "images": [
      {"name": "abc.png", "url": "/uploads/cats/1/abc.png"}
    ]
  }
}
```

response เมื่อทายเป็น `unknown`, `not_cat`, `low_quality`:

```json
{
  "final_label": "unknown",
  "best_known_name": "Mimi",
  "best_known_score": 0.8841,
  "second_known_score": 0.9181,
  "best_unknown_score": 0.9426,
  "best_not_cat_score": 0.8119,
  "quality_pass": true,
  "quality_reasons": [],
  "blur_score": 66.54,
  "brightness": 122.18,
  "top_matches": [
    {"label": "Mimi", "score": 0.9642}
  ],
  "matched_cat": null
}
```

สิ่งสำคัญ:

- ถ้าเป็น known cat ให้ส่ง `matched_cat`
- ถ้าไม่ใช่ known cat ให้ `matched_cat = null`

## 5. Train

### `POST /api/train`

response:

```json
{
  "started": true,
  "job": {
    "job_id": "abc123",
    "status": "running",
    "stage": "prepare",
    "progress": 0.0
  },
  "summary": {
    "index_status": "needs_train",
    "actual_total_images": 12,
    "indexed_total_images": 10
  }
}
```

### `GET /api/train/status`

response:

```json
{
  "job": {
    "job_id": "abc123",
    "status": "completed",
    "stage": "completed",
    "progress": 1.0,
    "total_images": 12,
    "processed_images": 12,
    "valid_images": 12,
    "elapsed_seconds": 1.8,
    "current_label": "mock",
    "current_image": "step_5",
    "split_totals": {
      "gallery": 8,
      "not_cat": 2,
      "unknown_cat": 2
    },
    "split_processed": {
      "gallery": 8,
      "not_cat": 2,
      "unknown_cat": 2
    },
    "history": [
      {
        "elapsed_seconds": 0.9,
        "processed_images": 6,
        "valid_images": 6,
        "progress": 0.5,
        "gallery_images": 4,
        "not_cat_images": 1,
        "unknown_cat_images": 1
      }
    ],
    "summary": {}
  },
  "summary": {
    "index_status": "ready",
    "actual_total_images": 12,
    "indexed_total_images": 12
  }
}
```

## ถ้าจะย้ายไป backend อื่น

ถ้า route path ต่างจากนี้ คุณมี 2 ทาง

1. คง route ให้เหมือนเดิม
   วิธีนี้ง่ายสุด เพราะไม่ต้องแก้ JS

2. เปลี่ยน path ในไฟล์ JS

- `static/js/common.js`
- `static/js/cats.js`
- `static/js/reference.js`
- `static/js/train.js`
- `static/js/predict.js`
