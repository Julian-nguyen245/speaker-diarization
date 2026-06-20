# Speaker Diarization — EEND-EDA

Hệ thống **nhật ký giọng nói (speaker diarization)** đầu-cuối theo kiến trúc
**EEND-EDA** (End-to-End Neural Diarization với Encoder-Decoder Attractors),
hiện thực bằng PyTorch và huấn luyện trên **CALLHOME**.

Trả lời câu hỏi *"ai nói, lúc nào"*: đưa vào một file âm thanh hội thoại,
hệ thống tách giọng và cho biết **khoảng thời gian nói của từng người**.

Repo gồm: notebook huấn luyện + đánh giá, và một **web demo** (upload `.wav`
→ timeline + thời lượng từng speaker + xuất RTTM).

---

## Kết quả (tập kiểm tra CALLHOME)

| Chỉ số | Giá trị |
|--------|:------:|
| DER | **36.2%** |
| Frame accuracy | 60.7% |
| Đếm đúng số người nói | 92.9% |

> Mô hình nhỏ (~6.4M tham số), huấn luyện **from scratch** trên 112 cuộc hội
> thoại. Lỗi chủ yếu là Speaker Error (nhầm danh tính ở đoạn chuyển lượt /
> chồng lấp); báo động nhầm (False Alarm) thấp.

---

## Kiến trúc

```
Audio 16kHz ─► log-Mel (40) ─► BLSTM encoder (4 layer, hidden 256)
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼ projection (512→256)                   ▼ EDA
            frame embeddings Ẽ                       LSTM decoder sinh
                  │                                  attractor + p_tồn tại
                  └──────────── dot-product ─────────────┘
                                      ▼
                         xác suất nói theo frame (T × S)
```

- Hàm mất mát: **PIT Loss** (bất biến hoán vị) + **Attractor Loss** (đếm speaker).
- `S_max = 4` người nói; đặc trưng log-Mel 40 chiều, frame 10 ms.

---

## Cấu trúc repo

```
models/        # BLSTM encoder, EDA attractor, mô hình EEND
losses/        # PIT loss, attractor loss, combined loss
data/          # lớp Dataset (chunking + SpecAugment)
checkpoints/   # best_model.pt (mô hình đã huấn luyện)
webapp/        # web demo (Flask + HTML) + hướng dẫn deploy
eend-vd.ipynb  # notebook: tiền xử lý, huấn luyện (có resume), đánh giá, vẽ hình
Dockerfile     # đóng gói web demo (CPU)
```

---

## Chạy nhanh

### 1. Web demo (suy luận)
Cần `checkpoints/best_model.pt` (đã có sẵn trong repo).

```bash
pip install -r webapp/requirements.txt
python webapp/app.py          # → http://localhost:5000
```

Hoặc bằng **Docker**:
```bash
docker build -t eend-demo .
docker run --rm -p 7860:7860 eend-demo     # → http://localhost:7860
```

Tính năng demo: upload nhiều file, **waveform + click để tua**, slider chỉnh
*ngưỡng / làm mượt / đoạn tối thiểu* trực tiếp, phát hiện **chồng lấp**, xuất **RTTM**.
Triển khai lên HuggingFace Spaces: xem [webapp/DEPLOY.md](webapp/DEPLOY.md).

### 2. Huấn luyện lại
Mở [`eend-vd.ipynb`](eend-vd.ipynb) (chạy được trên Google Colab). Cần dữ liệu
CALLHOME (file `.wav` + nhãn `.rttm`). Notebook lo toàn bộ: tiền xử lý log-Mel,
huấn luyện (lưu/khôi phục checkpoint), đánh giá DER và vẽ hình. Trên Colab có thể
tự tải dataset từ Kaggle và lưu checkpoint vào Google Drive.

---

## Yêu cầu
Python 3.9+, PyTorch ≥ 2.0, librosa, numpy. Web demo thêm Flask (+ gunicorn để
deploy). Đọc mp3/m4a cần `ffmpeg`.

## Ghi chú
- Nhãn `Speaker 1/2/…` là **kênh attractor** (sắp theo tổng thời lượng), không
  phải danh tính thật.
- Mô hình huấn luyện trên hội thoại chủ yếu 2 người → mạnh nhất với 2 người nói.

## License
MIT.
