
import torch
import torch.nn as nn

class AttractorModule(nn.Module):
    """
    LSTM decoder sinh tuan tu cac attractor.
    De DEM so nguoi noi (EDA): luc train se sinh num_speakers + 1 attractor,
    attractor cuoi cung dong vai tro "STOP" (existence target = 0).
    """
    def __init__(self, encoder_size=512, attractor_size=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=encoder_size,
            hidden_size=attractor_size,
            batch_first=True,
        )
        self.linear = nn.Linear(attractor_size, 1)   # existence LOGIT (chua sigmoid)
        # Projection de khoi tao hidden state tu encoder (512 -> 256)
        self.initial_proj = nn.Linear(encoder_size, attractor_size)

    def forward(self, encoder_out, num_attractors):
        """
        encoder_out    : (batch, frames, encoder_size)
        num_attractors : so attractor can sinh

        Returns:
            attractors       : (batch, num_attractors, attractor_size)
            existence_logits : (batch, num_attractors, 1)
        """
        batch_size = encoder_out.size(0)
        encoder_size = encoder_out.size(-1)

        # Khoi tao h0, c0 tu trung binh encoder out
        mean_enc = encoder_out.mean(dim=1)                              # (batch, encoder_size)
        h = self.initial_proj(mean_enc).unsqueeze(0).contiguous()      # (1, batch, attractor_size)
        c = torch.zeros_like(h)

        dummy_input = torch.zeros(batch_size, 1, encoder_size, device=encoder_out.device)

        attractors, existence = [], []
        for _ in range(num_attractors):
            out, (h, c) = self.lstm(dummy_input, (h, c))
            attractors.append(out)                  # (batch, 1, attractor_size)
            existence.append(self.linear(out))      # (batch, 1, 1)

        attractors = torch.cat(attractors, dim=1)   # (batch, num_attractors, attractor_size)
        existence  = torch.cat(existence,  dim=1)   # (batch, num_attractors, 1)
        return attractors, existence
