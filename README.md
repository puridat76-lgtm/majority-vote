# Cat Identity Majority Vote

เว็บ Flask สำหรับระบุตัวตนแมวจากภาพ โดยใช้
- `siamese_model3.h5` เป็น encoder แปลงรูปเป็น embedding
- majority vote จาก `Top 10` similarity สูงสุด
- คลาส `unknown_cat` และ `not_cat` ร่วมในการตัดสิน
- แสดง Top 10 candidates และ class summary บนหน้า Predict

## ใช้งาน
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
เปิดที่ `http://127.0.0.1:5056`

ครั้งแรกหลังเพิ่ม/ลบรูป แอปจะสร้าง image index จากรูปใน `uploads/` และ cache ไว้ที่ `data/index_cache.npz`; รอบถัดไปจะโหลด cache นี้แทนการสร้าง index ใหม่ทั้งหมด

## สำหรับ Mac Apple Silicon
ถ้าต้องการใช้ `siamese_model3.h5` จริง แนะนำติดตั้งเพิ่ม:
```bash
pip install tensorflow-macos tensorflow-metal
```
ถ้ายังไม่มี TensorFlow ระบบจะ fallback ไปใช้ classic embedding เพื่อให้เว็บยังรันได้ แต่การใช้งานจริงควรติดตั้ง TensorFlow เพื่อให้ใช้ encoder จากโมเดลโดยตรง

## กติกาการตัดสิน
1. ตรวจคุณภาพภาพ
2. หาใบหน้าแมวและ crop ก่อนเข้า encoder
3. เทียบกับฐานข้อมูลทั้งหมด
4. ดึง Top 10 similarity สูงสุด
5. majority vote ตามจำนวนคลาส
6. tie-break ด้วย weighted sum และ best score
7. ถ้าไม่มั่นใจพอจะ fallback เป็น `unknown_cat`
