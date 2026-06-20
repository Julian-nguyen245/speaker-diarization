
import torch
import torch.nn as nn
import sys
sys.path.append("/home/ju1ian/Documents/eend/models")

from encoder import BLSTMEncoder
from attractor import AttractorModule

class EEND(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_layers=4,
                 dropout=0.3, attractor_size=256):
        super().__init__()

        self.encoder = BLSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.attractor = AttractorModule(
            encoder_size=self.encoder.output_size,   # 2*hidden = 512
            attractor_size=attractor_size,
        )
        self.projection = nn.Linear(self.encoder.output_size, attractor_size)

    def forward(self, x, num_speakers=4):
        """
        Sinh num_speakers (= S_max) attractor.
        existence_logits cua tung attractor duoc dung de DEM speaker:
        attractor ung voi speaker that -> p->1, attractor du thua -> p->0.

        Returns:
            logits           : (batch, frames, num_speakers)
            existence_logits : (batch, num_speakers, 1)
        """
        enc_out  = self.encoder(x)            # (batch, frames, 512)
        enc_proj = self.projection(enc_out)   # (batch, frames, attractor_size)

        attractors, existence_logits = self.attractor(enc_out, num_speakers)
        logits = torch.bmm(enc_proj, attractors.transpose(1, 2))  # (batch, frames, num_speakers)
        return logits, existence_logits
