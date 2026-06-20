
import torch
import torch.nn as nn

class AttractorLoss(nn.Module):
    """
    Day model DEM so nguoi noi (eq. Attractor Loss trong tieu luan).
    Nhan ton tai q_s cho attractor thu s:
        q_s = 1 neu attractor khop voi 1 speaker THAT su co hoat dong
        q_s = 0 voi attractor du thua (cot label rong)
    Viec attractor nao khop speaker nao lay tu hoan vi toi uu phi* cua PIT.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, existence_logits, labels, best_perm, num_speakers):
        """
        existence_logits : (batch, num_speakers, 1)
        labels           : (batch, frames, num_speakers)
        best_perm        : (batch, num_speakers) tu PITLoss
        """
        batch_size = existence_logits.size(0)
        device = existence_logits.device

        # Cot label nao thuc su co hoat dong trong chunk nay (= so speaker S)
        active = (labels.sum(dim=1) > 0).float()                 # (B, S_max)

        # Gan theo hoan vi: q[b, best_perm[b][j]] = active[b][j]
        q = torch.zeros(batch_size, num_speakers, device=device)
        q.scatter_(1, best_perm, active)                         # (B, S_max)

        return self.bce(existence_logits.squeeze(-1), q)
