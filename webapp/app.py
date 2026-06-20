# -*- coding: utf-8 -*-
"""
Backend suy luan EEND-EDA cho web demo.
Chay:  pip install flask librosa torch soundfile
       python webapp/app.py   ->  http://localhost:5000

Tra ve XAC SUAT theo frame (da overlap-add + can permutation giua cac chunk),
de phia client tu dat nguong / lam muot / cat doan TUC THI.
"""
import os, sys, tempfile, itertools
import numpy as np, torch, librosa
from flask import Flask, request, jsonify, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
# Tim thu muc chua code model (eend.py) — ho tro ca layout repo lan deploy goi
for _m in (os.path.join(HERE, "..", "models"), os.path.join(HERE, "models")):
    if os.path.exists(os.path.join(_m, "eend.py")):
        sys.path.insert(0, os.path.abspath(_m)); break
from eend import EEND

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tim checkpoint — uu tien bien moi truong MODEL_CKPT
CKPT = os.environ.get("MODEL_CKPT") or next(
    (p for p in (os.path.join(HERE, "..", "checkpoints", "best_model.pt"),
                 os.path.join(HERE, "checkpoints", "best_model.pt"),
                 os.path.join(HERE, "best_model.pt")) if os.path.exists(p)), None)
if CKPT is None:
    raise FileNotFoundError("Khong tim thay best_model.pt — dat bien MODEL_CKPT tro toi file.")

ck  = torch.load(CKPT, map_location=DEVICE, weights_only=False)
CFG = ck["config"]
model = EEND(CFG["input_size"], CFG["hidden_size"], CFG["num_layers"],
             CFG["dropout"], CFG["attractor_size"]).to(DEVICE)
model.load_state_dict(ck["model_state"]); model.eval()

SR, N_MELS, N_FFT, HOP = 16000, 40, 400, 160
FRAME = HOP / SR                       # 0.01 s / frame
CH    = CFG["chunk_size"]              # 500 frame
S     = CFG["num_speakers"]            # 4
PERMS = list(itertools.permutations(range(S)))
MAX_RET_FRAMES = 24000                 # gioi han so frame tra ve (gom lai neu dai hon)


def extract_logmel(y):
    mel = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=SR,
                                          n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
    return librosa.power_to_db(mel, ref=np.max).T


@torch.no_grad()
def infer(path):
    y, _ = librosa.load(path, sr=SR)
    feat = extract_logmel(y)
    T = feat.shape[0]
    if T == 0:
        raise ValueError("Audio rong hoac qua ngan")

    hop = max(1, CH // 2)                                   # cua so chong 50%
    starts = list(range(0, max(1, T - CH + 1), hop))
    if T > CH and starts[-1] != T - CH:
        starts.append(T - CH)
    if T <= CH:
        starts = [0]

    acc = np.zeros((T, S)); cnt = np.zeros(T); exist_list = []
    for s in starts:
        e = min(s + CH, T)
        c = torch.tensor(feat[s:e], dtype=torch.float32).unsqueeze(0)
        if c.shape[1] < CH:
            c = torch.cat([c, torch.zeros(1, CH - c.shape[1], feat.shape[1])], dim=1)
        logits, ex = model(c.to(DEVICE), num_speakers=S)
        p = torch.sigmoid(logits[0]).cpu().numpy()[:e - s]
        exist_list.append(torch.sigmoid(ex[0].squeeze(-1)).cpu().numpy())

        # Can kenh cua chunk hien tai voi phan da tich luy (vung chong lap)
        ov = cnt[s:e] > 0
        if ov.any():
            ref = acc[s:e][ov] / cnt[s:e][ov][:, None]
            cur = p[ov]
            best, bscore = PERMS[0], 1e18
            for perm in PERMS:
                d = np.abs(cur[:, perm] - ref).sum()
                if d < bscore:
                    bscore, best = d, perm
            p = p[:, list(best)]
        acc[s:e] += p; cnt[s:e] += 1

    probs = acc / np.maximum(cnt[:, None], 1.0)             # (T, S) da overlap-add
    exist = np.mean(exist_list, axis=0) if exist_list else np.zeros(S)

    # Gom frame de gioi han kich thuoc tra ve
    factor = max(1, int(np.ceil(T / MAX_RET_FRAMES)))
    if factor > 1:
        n = (T // factor) * factor
        probs = probs[:n].reshape(-1, factor, S).mean(axis=1)

    return {
        "duration": T * FRAME,
        "frame_shift": FRAME * factor,
        "n_channels": S,
        "exist": [round(float(x), 3) for x in exist],
        "est_speakers": int((exist > 0.5).sum()),
        "probs": [[round(float(v), 2) for v in probs[:, ch]] for ch in range(S)],
    }


app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/samples")
def list_samples():
    d = os.path.join(HERE, "samples")
    files = sorted(f for f in os.listdir(d)) if os.path.isdir(d) else []
    return jsonify(files)


@app.route("/samples/<path:name>")
def get_sample(name):
    return send_from_directory(os.path.join(HERE, "samples"), name)


@app.route("/infer", methods=["POST"])
def do_infer():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "Khong co file"}), 400
    suffix = os.path.splitext(f.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        f.save(tmp.name); tmp.close()
        res = infer(tmp.name)
        res["filename"] = f.filename
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", "5000"))
    print(f"Model loaded: {CKPT} | device={DEVICE} | "
          f"hidden={CFG['hidden_size']} layers={CFG['num_layers']} | port={PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
