# EEND-EDA — Web Demo (Speaker Diarization)

Upload file âm thanh → model trả về **thời gian nói của từng người** (timeline + bảng + RTTM).

Model PyTorch không chạy thẳng trong trình duyệt nên demo gồm 2 phần:
- `app.py` — backend Flask: nạp `checkpoints/best_model.pt`, trích log-Mel, suy luận, trả **xác suất theo frame**.
- `index.html` — giao diện: upload nhiều file, waveform, timeline, slider, bảng, RTTM.

## Chạy
```bash
pip install flask librosa torch soundfile
python webapp/app.py        # → http://localhost:5000
```
Cần có `checkpoints/best_model.pt` và thư mục `models/` (đã có trong repo).
Đọc được `.wav/.mp3/.m4a/.flac/.ogg` (mp3/m4a cần `ffmpeg` trên máy).

## Tính năng
- **Đa file** song song, kéo-thả hoặc chọn.
- **Waveform** (vẽ bằng Web Audio, không cần mạng) + **click để tua** + **playhead** chạy theo audio.
- **Slider trực tiếp** (đổi tức thì, không gọi lại model):
  - *Ngưỡng phát hiện* (0.30–0.70)
  - *Làm mượt* (median filter, ms) — gộp nhiễu lẻ
  - *Đoạn tối thiểu* (ms) — bỏ đoạn quá ngắn
- **Phát hiện chồng lấp** (≥2 người cùng nói) → hàng riêng.
- **Đếm số người nói** theo xác suất tồn tại (existence) của attractor.
- **Tải RTTM** theo cấu hình slider hiện tại.

## Xử lý phía backend
1. Resample 16 kHz, log-Mel 40 chiều (đúng như khi huấn luyện).
2. **Cửa sổ chồng 50% (overlap-add)** + **căn permutation kênh giữa các chunk** → mượt ở biên, nhãn người nói ổn định hơn khi ghép.
3. Trả xác suất từng frame (gom bớt nếu file dài) để client tự đặt ngưỡng/làm mượt/cắt đoạn.

## Lưu ý
- "Speaker 1/2/…" là kênh attractor (xếp theo tổng thời lượng), **không phải danh tính thật**.
- Model huấn luyện trên CALLHOME (đa số 2 người) nên mạnh nhất ở 2 người; lỗi chính là nhầm danh tính ở đoạn chuyển lượt/chồng lấp.
