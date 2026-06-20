
import torch
from torch.utils.data import Dataset
import numpy as np
import os

CACHE_DIR = "/home/ju1ian/Documents/eend/data/processed"

class CallHomeDataset(Dataset):
    """
    augment=True  -> bật SpecAugment (chỉ dùng cho tập train) để chống overfit.
    freq_mask/time_mask: độ rộng tối đa của dải mask; num_masks: số dải mỗi loại.
    """
    def __init__(self, indices, chunk_size=500, augment=False,
                 freq_mask=8, time_mask=50, num_masks=2):
        self.indices    = indices
        self.chunk_size = chunk_size
        self.augment    = augment
        self.freq_mask  = freq_mask
        self.time_mask  = time_mask
        self.num_masks  = num_masks
        self.chunks     = self._build_chunks()

    def _build_chunks(self):
        chunks = []
        for idx in self.indices:
            feat       = np.load(os.path.join(CACHE_DIR, f"feat_{idx}.npy"))
            num_frames = feat.shape[0]
            for start in range(0, num_frames - self.chunk_size, self.chunk_size):
                chunks.append((idx, start))
        return chunks

    def _spec_augment(self, feat):
        # feat: (chunk_size, n_mels)
        feat = feat.copy()
        T, F = feat.shape
        for _ in range(self.num_masks):
            # time mask
            t = np.random.randint(0, self.time_mask + 1)
            if t > 0 and T - t > 0:
                t0 = np.random.randint(0, T - t)
                feat[t0:t0 + t, :] = 0.0
            # frequency mask
            f = np.random.randint(0, self.freq_mask + 1)
            if f > 0 and F - f > 0:
                f0 = np.random.randint(0, F - f)
                feat[:, f0:f0 + f] = 0.0
        return feat

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        idx, start = self.chunks[i]
        feat  = np.load(os.path.join(CACHE_DIR, f"feat_{idx}.npy"))[start:start + self.chunk_size]
        label = np.load(os.path.join(CACHE_DIR, f"label_{idx}.npy"))[start:start + self.chunk_size]
        if self.augment:
            feat = self._spec_augment(feat)
        return (
            torch.tensor(feat,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

print("dataset.py written (with SpecAugment)")
