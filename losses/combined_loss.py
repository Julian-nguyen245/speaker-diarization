
import sys
sys.path.append("/home/ju1ian/Documents/eend/losses")
import torch.nn as nn
from pit_loss import PITLoss
from attractor_loss import AttractorLoss

class EENDLoss(nn.Module):
    def __init__(self, attractor_weight=1.0):
        super().__init__()
        self.pit_loss        = PITLoss()
        self.attractor_loss  = AttractorLoss()
        self.attractor_weight = attractor_weight

    def forward(self, logits, labels, existence_logits, num_speakers):
        """
        logits           : (batch, frames, num_speakers)
        labels           : (batch, frames, num_speakers)
        existence_logits : (batch, num_speakers + 1, 1)
        """
        pit, best_perm = self.pit_loss(logits, labels)
        attr = self.attractor_loss(existence_logits, labels, best_perm, num_speakers)
        total = pit + self.attractor_weight * attr
        return total, pit, attr
