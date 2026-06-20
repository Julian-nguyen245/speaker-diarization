# EEND-EDA web demo — image CPU tu chua.
# Build:  docker build -t eend-demo .
# Run:    docker run --rm -p 7860:7860 eend-demo   ->  http://localhost:7860
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cai thu vien truoc (tan dung cache layer)
COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

# Chi copy thu can thiet (model code + checkpoint + webapp)
COPY models/ models/
COPY checkpoints/best_model.pt checkpoints/best_model.pt
COPY webapp/ webapp/

ENV PORT=7860
EXPOSE 7860
WORKDIR /app/webapp
# 1 worker de chi nap model 1 lan; timeout cao cho file dai
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "300", "-w", "1", "app:app"]
