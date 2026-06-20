
import torch
import torch.nn as nn
from itertools import permutations

class PITLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # reduction="none" -> chon permutation tot nhat cho TUNG sample.
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, labels):
        """
        logits : (batch, frames, num_speakers)
        labels : (batch, frames, num_speakers)

        Returns:
            loss      : scalar (trung binh tren batch)
            best_perm : (batch, num_speakers) - best_perm[b][j] = chi so attractor
                        chiu trach nhiem cho cot label j cua sample b.
                        (Dung de gan existence target trong AttractorLoss.)
        """
        batch_size, frames, num_speakers = logits.shape
        perms = list(permutations(range(num_speakers)))

        losses = []
        for perm in perms:
            perm_logits = logits[:, :, perm]                       # (B, T, S)
            loss = self.bce(perm_logits, labels).mean(dim=(1, 2))  # (B,)
            losses.append(loss)

        losses = torch.stack(losses, dim=1)                        # (B, num_perms)
        min_loss, min_idx = losses.min(dim=1)                      # (B,)

        perm_tensor = torch.tensor(perms, device=logits.device, dtype=torch.long)  # (num_perms, S)
        best_perm = perm_tensor[min_idx]                           # (B, S)

        return min_loss.mean(), best_perm
