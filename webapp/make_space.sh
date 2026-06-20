#!/usr/bin/env bash
# Dung thu muc toi gian "space/" san sang day len HuggingFace Spaces (Docker SDK).
# Cach dung:  bash webapp/make_space.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/webapp/space"

rm -rf "$OUT"
mkdir -p "$OUT/models" "$OUT/checkpoints"

# App + giao dien + thu vien
cp "$ROOT/webapp/app.py" "$ROOT/webapp/index.html" "$ROOT/webapp/requirements.txt" "$OUT/"
# File mau (neu co)
[ -d "$ROOT/webapp/samples" ] && cp -r "$ROOT/webapp/samples" "$OUT/samples"
# Code model (chi 3 file can thiet)
cp "$ROOT/models/eend.py" "$ROOT/models/encoder.py" "$ROOT/models/attractor.py" "$OUT/models/"
# Checkpoint
cp "$ROOT/checkpoints/best_model.pt" "$OUT/checkpoints/"

# Dockerfile tu chua (app.py nam ngay goc /app)
cat > "$OUT/Dockerfile" <<'DOCKER'
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MODEL_CKPT=/app/checkpoints/best_model.pt
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "300", "-w", "1", "app:app"]
DOCKER

# README co header cho HuggingFace Spaces
cat > "$OUT/README.md" <<'MD'
---
title: EEND-EDA Diarization
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# EEND-EDA — Speaker Diarization
Tải lên file âm thanh → trả về thời gian nói của từng người (timeline, bảng, RTTM).
Mô hình BLSTM-EDA huấn luyện trên CALLHOME.
MD

# .gitattributes: checkpoint qua git-lfs
cat > "$OUT/.gitattributes" <<'GA'
*.pt filter=lfs diff=lfs merge=lfs -text
GA

echo "✅ Space san sang tai: $OUT"
echo
echo "Day len HuggingFace:"
echo "  1) Tao Space (SDK = Docker) tai https://huggingface.co/new-space"
echo "  2) cd $OUT"
echo "     git init && git lfs install && git add . && git commit -m 'init'"
echo "     git remote add origin https://huggingface.co/spaces/<user>/<ten-space>"
echo "     git push -u origin main"
