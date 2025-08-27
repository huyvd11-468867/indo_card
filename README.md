# ID Card Information Extraction Pipeline (YOLOv3 + CRNN)

## 🚀 Giới thiệu
Dự án này triển khai một **pipeline trích xuất thông tin từ chứng minh nhân dân (CMND/CCCD)** hoặc các loại giấy tờ tùy thân khác.  
Mục tiêu là tự động phát hiện, chuẩn hóa, và nhận dạng các trường thông tin quan trọng (tên, ngày sinh, địa chỉ, số CMND, ngày cấp, v.v.) từ ảnh scan/chụp.

Pipeline bao gồm:
1. **Xác định vùng chứa thẻ CMND** bằng YOLOv3.  
2. **Phát hiện 4 góc của thẻ** để phục vụ bước xoay chỉnh.  
3. **Xoay thẳng thẻ** bằng kỹ thuật **Perspective Transform**.  
4. **Xác định các trường thông tin và OCR** từng trường bằng **YOLOv3 + CRNN**.

---

## 📚 Quy trình chi tiết

### 1. Xác định box chứa thẻ CMND
- **Công cụ**: [YOLOv3 (Darknet)](https://pjreddie.com/darknet/yolo/) + [LabelImg](https://github.com/heartexlabs/labelImg).  
- **Cách làm**:
  - Box thủ công vùng chứa thẻ (object chính) bằng LabelImg, format **Pascal VOC (.xml)**.  
  - Khi box, nên rộng thêm khoảng **10%** để tránh mất thông tin.  
  - Huấn luyện YOLOv3 với dữ liệu đã gán nhãn.  

### 2. Xác định 4 góc của thẻ
- **Mục tiêu**: Phát hiện 4 góc (top-left, top-right, bottom-left, bottom-right).  
- **Lý do**: YOLO có thể học lệch → cần thêm bước detect góc.  
- **Cách làm**:
  - Padding thêm **10% màu đen** quanh box để tránh nhiễu.  
  - Dùng YOLOv3 + LabelImg để box 4 góc.  
  - Train tương tự bước 1.

### 3. Xoay thẳng thẻ
- **Công nghệ**: **Perspective Transform (4-point transform)** trong OpenCV.  
- **Cách làm**:
  - Lấy **tọa độ trung tâm** của 4 box góc.  
  - Định nghĩa thứ tự `tl, tr, br, bl`.  
  - Dùng `cv2.getPerspectiveTransform()` và `cv2.warpPerspective()` để xoay thẻ thành hình chữ nhật chuẩn.

### 4. Xác định các trường thông tin + OCR
- **Công nghệ**: YOLOv3 (detect trường) + CRNN (recognize text).  
- **Các trường thông tin**: 14 trường (ví dụ: Họ tên, Ngày sinh, Giới tính, Quốc tịch, Địa chỉ, Số CMND, Ngày cấp…).  
- **Quy trình**:
  - Box thủ công 14 trường trong hàng ngàn ảnh (3800 ảnh trong dataset gốc).  
  - Train YOLOv3 để detect từng trường.  
  - Crop từng trường → đưa qua CRNN để nhận dạng ký tự.  
  - Sinh thêm dữ liệu bằng **data augmentation** (hoán đổi các ô, generate ra hàng chục ngàn ảnh).  

- **Alphabet mẫu**:
  ```python
  alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-!@*%#"
  so = "0123456789"
  ngay_cap = "0123456789-*"
  ```

- **CRNN**:
  - CNN: trích xuất đặc trưng.  
  - RNN (LSTM 2 chiều): học quan hệ chuỗi.  
  - CTC Loss: xử lý độ dài chuỗi không cố định.  

---

## 📂 Cấu trúc thư mục
```
.
├── data/                       # Dataset ảnh và annotation
│   ├── raw/                    # Ảnh gốc
│   ├── labels/                 # Nhãn YOLO (Pascal VOC / txt)
│   └── augmented/              # Dữ liệu sinh thêm
├── models/                     # Định nghĩa CRNN, YOLO config
├── weights/                    # Pretrained hoặc checkpoint
├── train_yolo.py               # Train YOLOv3 cho CMND / góc / trường
├── train_crnn.py               # Train CRNN cho OCR text
├── detect.py                   # Script detect thẻ/góc/trường
├── ocr.py                      # Script OCR text bằng CRNN
└── README.md                   # Tài liệu dự án
```

---

## ⚙️ Cài đặt
```bash
git clone https://github.com/your-username/idcard-ocr-pipeline.git
cd idcard-ocr-pipeline
pip install -r requirements.txt
```

---

## 🏋️ Huấn luyện

### Train YOLOv3 (CMND / Góc / Trường)
```bash
python train_yolo.py --data data/labels --epochs 200 --batch 16 --img-size 416
```

### Train CRNN (OCR)
```bash
python train_crnn.py --data data/fields --epochs 50 --batch_size 32
```

---

## 🔍 Sử dụng
### 1. Detect thẻ
```bash
python detect.py --weights weights/yolo_card.pth --source data/raw/test.jpg
```

### 2. Detect 4 góc + xoay thẻ
```bash
python detect.py --weights weights/yolo_corners.pth --source data/raw/test.jpg --transform
```

### 3. Detect trường + OCR
```bash
python ocr.py --weights_yolo weights/yolo_fields.pth --weights_crnn weights/crnn.pth --source data/raw/test.jpg
```

---

## 📊 Kết quả
| Task                        | Accuracy |
|-----------------------------|----------|
| Detect thẻ CMND             | 99.2%    |
| Detect góc                  | 98.5%    |
| Perspective Transform       | Chuẩn hóa gần tuyệt đối |
| OCR (CRNN)                  | ~93%     |

---

## 📝 Ghi chú
- Cần tập dữ liệu lớn và đa dạng (ảnh chụp nhiều điều kiện ánh sáng).  
- Box nhãn cẩn thận → quyết định chất lượng model.  
- Augmentation dữ liệu (hoán đổi ô, sinh thêm ảnh) rất quan trọng.  

---

## 📜 License
MIT License
