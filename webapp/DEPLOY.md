# Triển khai (Deploy)

Ba cách chạy demo, từ đơn giản đến online.

## 1. Local — chạy thẳng (dev)
```bash
pip install -r webapp/requirements.txt
python webapp/app.py            # → http://localhost:5000
```

## 2. Local — gunicorn (giống production)
```bash
pip install -r webapp/requirements.txt
cd webapp && gunicorn -b 0.0.0.0:7860 --timeout 300 -w 1 app:app
# → http://localhost:7860
```

## 3. Docker (tự chứa, CPU)
Chạy ở **gốc repo** (Dockerfile đã ở đó):
```bash
docker build -t eend-demo .
docker run --rm -p 7860:7860 eend-demo     # → http://localhost:7860
```
`.dockerignore` đã loại CallHome / thesis / notebook nên image gọn; chỉ đóng gói
`models/`, `checkpoints/best_model.pt`, `webapp/`.

## 4. HuggingFace Spaces (online, miễn phí)
Dùng script dựng thư mục tối giản rồi push:
```bash
bash webapp/make_space.sh          # tạo webapp/space/
cd webapp/space
git init && git lfs install
git add . && git commit -m "init"
git remote add origin https://huggingface.co/spaces/<user>/<ten-space>
git push -u origin main
```
- Tạo Space trước tại <https://huggingface.co/new-space>, chọn **SDK = Docker**.
- `best_model.pt` (~76 MB) đẩy qua **git-lfs** (script đã tạo sẵn `.gitattributes`).
- Space sẽ tự build Dockerfile và mở cổng `7860`.

## Biến môi trường
| Biến | Ý nghĩa | Mặc định |
|------|---------|----------|
| `PORT` | cổng khi chạy `python app.py` | `5000` |
| `MODEL_CKPT` | đường dẫn checkpoint | tự dò `../checkpoints/best_model.pt` |

## Ghi chú
- Image dùng **torch CPU** (`--extra-index-url .../whl/cpu`) cho nhẹ; có GPU thì model tự dùng CUDA khi chạy local.
- 1 worker gunicorn (`-w 1`) để chỉ nạp model một lần; `--timeout 300` cho file dài.
- Cần `ffmpeg` để đọc mp3/m4a (Docker đã cài sẵn).
